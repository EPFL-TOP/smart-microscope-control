"""The Micro-Manager half of a survey: the install, its adapters, their devices.

Two kinds of call live here, and they must not be confused:

* :func:`system_section` and the adapter *names* load no DLL (MMCore lists
  file names), so they run in the ``smc discover`` process itself;
* :func:`list_adapters` and :func:`probe_adapter` load adapter DLLs. A vendor
  DLL can hang or crash the process that loads it, so these only ever run
  in the child process of :mod:`smc.discovery._mm_child`, started by
  :func:`mm_section` under a timeout. A dying child costs the device lists,
  not the survey — the lesson behind ``nikon-control``'s
  one-device-per-throwaway-core probe.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from smc import __version__
from smc.discovery import diagnosis
from smc.discovery.models import (
    AdapterDevice,
    AdapterInfo,
    BuildScan,
    ProbedDevice,
    ProbeResult,
    ProbeSource,
    SystemInfo,
)
from smc.hardware import core as core_mod
from smc.hardware.roles import device_type_name

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = [
    "BUILD_TIMEOUT_S",
    "CHILD_COMMAND",
    "CHILD_TIMEOUT_S",
    "BuildRun",
    "ChildResult",
    "build_section",
    "check_device_name",
    "check_port",
    "exit_detail",
    "list_adapters",
    "mm_section",
    "other_installs",
    "probe_adapter",
    "system_section",
    "walk_adapter",
]

logger = logging.getLogger("smc.discovery.mm_inventory")

#: Loading every DLL of a full nightly takes a while on a slow lab PC.
CHILD_TIMEOUT_S = 180.0

#: How the parent starts the child; a module attribute so tests can replace
#: it with a child that crashes or hangs.
CHILD_COMMAND: list[str] = [sys.executable, "-m", "smc.discovery._mm_child"]

_HUB_TYPE = "Hub"

#: A pre-init ``Port`` that means "not set", compared after ``strip().casefold()``.
#: Serial adapters are said to publish ``Undefined``; ``Utilities`` publishes
#: ``''``, which is the only one measured (design m2 §9.1).
_UNSET_PORTS = frozenset({"", "undefined"})

#: No serial port name is longer; a longer value was not typed as a port.
_MAX_PORT_CHARS = 200

#: No device name an adapter publishes is longer; a longer value was not typed
#: as a device name either.
_MAX_NAME_CHARS = 200

#: How long to wait for a killed child to go away before giving up on it.
_KILL_WAIT_S = 10.0

#: Python runs a pending signal handler between two waits (FM-83).
_WAIT_SLICE_S = 0.5

#: A child sharing the console can repaint it. ``0`` (no flag) off Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: POSIX-only; `_kill_child_tree`'s POSIX branch only runs where it exists,
#: but a test that simulates that branch on a real Windows interpreter
#: (there is no SIGKILL to send there) needs the same fallback.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _decode_console_bytes(data: bytes) -> str:
    """Decode a Windows console tool's raw output.

    ``taskkill`` writes the console (OEM) code page, not UTF-8 — the same
    mismatch ``os_inventory.py``'s PowerShell call works around by forcing
    ``[Console]::OutputEncoding``, which a plain ``.exe`` offers no way to
    do. The ``oem`` codec exists only on a real Windows interpreter; falling
    back keeps this decodable everywhere else (tests that simulate the
    Windows branch on macOS/Linux included).
    """
    try:
        return data.decode("oem", "replace").strip()
    except LookupError:
        return data.decode("utf-8", "replace").strip()


def _wait_in_slices(child: subprocess.Popen[bytes], timeout_s: float) -> int:
    """Wait for the child like ``wait(timeout_s)``, in slices of ``_WAIT_SLICE_S``.

    On Windows one ``Popen.wait(timeout)`` is one blocking call that a Ctrl-C
    does not end, and the child, created with ``CREATE_NO_WINDOW``, never
    receives the Ctrl-C either: the survey would sit out the whole timeout
    (180 s by default) with the user pressing keys (FM-83). Between two slices
    Python runs the pending signal handler and raises ``KeyboardInterrupt``.

    Raises:
        subprocess.TimeoutExpired: At the deadline, exactly as ``wait`` does.
            The last slice waits for the remainder, so a child that ends at
            the deadline is not reported as a timeout.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        try:
            return child.wait(timeout=max(min(_WAIT_SLICE_S, remaining), 0.0))
        except subprocess.TimeoutExpired:
            if deadline - time.monotonic() <= 0:
                raise


def _kill_child_tree(child: subprocess.Popen[bytes]) -> str | None:
    """Kill the child and any helper process it started (FM-35).

    ``kill()`` alone stops only the direct child: a helper it launched (a
    grandchild loading the vendor DLL, say) survives and may keep holding
    the hardware. POSIX: the child was started as the leader of its own
    session (``start_new_session=True``), so its whole process group shares
    its pid. Windows: ``taskkill /T`` walks the tree MMCore itself cannot
    report.

    Returns:
        A note fragment when the kill itself failed (e.g. ``taskkill``
        exited non-zero); ``None`` when there is nothing more to say.
    """
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(child.pid)],
                capture_output=True,
                timeout=10.0,
                creationflags=_NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            child.kill()
            return f"taskkill failed: {exc}"
        child.kill()
        if result.returncode != 0:
            detail = _decode_console_bytes(result.stderr or result.stdout or b"")
            return f"taskkill failed: {detail}" if detail else "taskkill failed"
        return None
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, _SIGKILL)
    return None


class ChildResult(BaseModel):
    """What the child prints on its last stdout line, as ASCII JSON."""

    listed: list[AdapterInfo] = Field(default_factory=list)
    probe: ProbeResult | None = None
    #: Set only by ``--build`` (design m2 §10.3).
    build: BuildScan | None = None


def other_installs(active: Path | None) -> list[str]:
    r"""Folder names of Micro-Manager installs other than the one in use.

    An MMStudio under ``C:\Program Files`` beside pymmcore-plus's own install
    is the usual source of a device-interface mismatch, so it is worth
    knowing it exists. Names only: the paths contain the user name.
    """
    roots: list[Path] = []
    if platform.system() == "Windows":
        for var in ("ProgramFiles", "ProgramFiles(x86)"):
            if os.environ.get(var):
                roots.append(Path(os.environ[var]))
        if os.environ.get("LOCALAPPDATA"):
            roots.append(Path(os.environ["LOCALAPPDATA"]) / "pymmcore-plus" / "mm")
    try:  # where `mmcore install` puts its builds, on every OS
        from pymmcore_plus._util import USER_DATA_MM_PATH

        roots.append(Path(USER_DATA_MM_PATH))
    except Exception:  # a private path of pymmcore-plus; its absence is harmless
        logger.debug("pymmcore-plus user data path not available")

    active_resolved = active.resolve() if active else None
    found: set[str] = set()
    for root in roots:
        try:
            candidates = [p for p in root.glob("Micro-Manager*") if p.is_dir()]
        except OSError:
            continue
        found.update(p.name for p in candidates if p.resolve() != active_resolved)
    return sorted(found)


def system_section() -> tuple[SystemInfo, list[str]]:
    """The machine, the software versions, and the installed adapter names.

    Runs in the parent: finding the install, reading the device API version
    and listing adapter names load no adapter.
    """
    import pymmcore_plus

    st = core_mod.status()
    info = SystemInfo(
        hostname=platform.node(),
        collected_at=datetime.now(timezone.utc),
        os=platform.platform(),
        os_build=platform.version(),
        python=platform.python_version(),
        smc=__version__,
        pymmcore_plus=str(pymmcore_plus.__version__),
        mm_install=str(st.install_dir) if st.install_dir else None,
        device_api=st.api_version,
        other_mm_installs=other_installs(st.install_dir),
    )
    return info, list(st.adapters)


def list_adapters(core: CMMCorePlus, names: list[str]) -> list[AdapterInfo]:
    """The devices each adapter offers, without loading any device.

    Asking loads the adapter's DLL, which on some adapters (``NikonTi2``)
    contacts the vendor SDK. One adapter failing is recorded and the next
    is asked: the error text is what the survey is for.
    """
    installed = set(core.getDeviceAdapterNames())
    result = []
    for name in names:
        if name not in installed:
            result.append(AdapterInfo(name=name, installed=False))
            continue
        try:
            devices = list(core.getAvailableDevices(name))
            types = list(core.getAvailableDeviceTypes(name))
            descriptions = list(core.getAvailableDeviceDescriptions(name))
        except Exception as exc:
            result.append(AdapterInfo(name=name, installed=True, error=str(exc)))
            continue
        listed = [
            AdapterDevice(name=str(d), type=device_type_name(t), description=str(desc))
            for d, t, desc in zip(devices, types, descriptions, strict=False)
        ]
        result.append(
            AdapterInfo(
                name=name,
                installed=True,
                devices=listed,
                list_meaning=diagnosis.listing_meaning(name, len(listed)),
            )
        )
    return result


def _is_invisible(char: str) -> bool:
    """A control character, or one that prints as nothing or as a blank that is no space.

    Unicode categories ``C*`` (control, format, surrogate, private, unassigned)
    and ``Z*`` (the line and paragraph separators, and every space separator:
    the no-break space among them) are refused; the plain space is not, since
    a port written with one is dropped at the edges and refused nowhere else.
    """
    if char == " ":
        return False
    category = unicodedata.category(char)
    return category[0] == "C" or category in ("Zl", "Zp", "Zs")


def check_port(port: str) -> str:
    """The port as it will be sent to a device, or why it cannot be.

    Spaces around it are dropped (a pasted ``" COM3 "``); a newline, a NUL or
    any other control character is refused rather than stripped, because a
    value that holds one was not typed as a port and would reach a vendor
    driver as it is (FM-32). So are the characters that look like nothing: a
    no-break space, a byte-order mark, a zero-width space, a line separator
    (what a port copied from a web page, an e-mail or a Notepad file carries,
    FM-80), which would look right on screen and name no port. An empty value
    is refused too: ``Path("")`` taught the project that an empty string is a
    silent default (FM-78), and an empty ``Port`` is exactly what ``Utilities``
    shows for "not set".

    Raises:
        ValueError: The port is empty, longer than 200 characters, or holds a
            control character or an invisible one.
    """
    value = port.strip(" ")
    if not value:
        raise ValueError("the port is empty")
    if len(value) > _MAX_PORT_CHARS:
        raise ValueError(f"the port is longer than {_MAX_PORT_CHARS} characters")
    if any(_is_invisible(c) for c in value):
        raise ValueError("the port holds an invisible or control character")
    return value


def check_device_name(name: str) -> str:
    """A device name as it will be compared with the adapter's, or why it cannot.

    The name of a device to leave out (``--skip``), with the rules of
    :func:`check_port` for the same reasons (FM-32, FM-78, FM-80): a pasted
    ``" TIDiaLamp "`` loses its spaces, and a name with a control or invisible
    character looks right on screen and matches no device, so the device that
    crashes the build would be loaded after all. Commas are allowed: a device
    whose name has one is a real thing to skip.

    Raises:
        ValueError: The name is empty, longer than 200 characters, or holds a
            control character or an invisible one.
    """
    value = name.strip(" ")
    if not value:
        raise ValueError("the device name is empty")
    if len(value) > _MAX_NAME_CHARS:
        raise ValueError(f"the device name is longer than {_MAX_NAME_CHARS} characters")
    if any(_is_invisible(c) for c in value):
        raise ValueError("the device name holds an invisible or control character")
    return value


def _loaded_type(core: CMMCorePlus, label: str) -> str:
    """The type of a loaded device, for one the adapter did not list."""
    try:
        return device_type_name(int(core.getDeviceType(label)))
    except Exception:  # the row is more useful with an unknown type than none
        logger.debug("device type of %s not available", label, exc_info=True)
        return "Unknown"


def _needs_port(core: CMMCorePlus, label: str) -> bool:
    """Whether a loaded, uninitialised device has a pre-init ``Port`` that is unset.

    A serial adapter cannot initialise without its port, and failing without
    it is not a hardware fault (FM-08). An exception while *reading* the
    property counts as "needs no port": the device is initialised, and a
    failure is reported raw (unknown beats guessed).
    """
    try:
        if not core.hasProperty(label, "Port"):
            return False
        if not core.isPropertyPreInit(label, "Port"):
            return False
        return str(core.getProperty(label, "Port")).strip().casefold() in _UNSET_PORTS
    except Exception:
        logger.debug("the Port of %s could not be read", label, exc_info=True)
        return False


class _Walk:
    """One walk of an adapter, in the order of design m2 §9.3.

    The state lives on an object because every rule appends rows, notes and
    checkpoints to the same result. ``untried`` counts the devices that were
    skipped without being looked at, so that a note about ``--port`` never
    claims to know what nobody checked.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        name: str,
        port: str | None,
        skip: Collection[str],
        progress: Callable[[str], None] | None,
        checkpoint: Callable[[ProbeResult], None] | None,
    ) -> None:
        self.core = core
        self.name = name
        self.port = port
        self.skip = frozenset(skip)
        self.progress = progress
        self.checkpoint = checkpoint
        self.result = ProbeResult(adapter=name)
        self.port_needed = False
        self.untried = 0

    # --- bookkeeping ---------------------------------------------------------
    def _say(self, line: str) -> None:
        if self.progress is not None:
            self.progress(line)

    def _checkpoint(self) -> None:
        """Hand over a copy of what is known, which is not the whole yet (§9.5)."""
        if self.checkpoint is None:
            return
        snapshot = self.result.model_copy(deep=True)
        snapshot.complete = False
        self.checkpoint(snapshot)

    def _add(self, row: ProbedDevice) -> None:
        self.result.devices.append(row)
        self._checkpoint()

    def _skip(self, name: str, type_name: str, reason: str) -> None:
        self.untried += 1
        self._add(
            ProbedDevice(name=name, type=type_name, status="not-probed", reason=reason)
        )

    # --- one device (rules 3 and 9) ------------------------------------------
    def _try(
        self,
        label: str,
        type_name: str | None,
        source: ProbeSource,
        parent: str | None,
    ) -> ProbedDevice:
        """Load, give its port, initialise: the row says what was tried.

        ``type_name`` is the listed type, or ``None`` for a device the adapter
        does not list, whose type is read after the load. ``KeyboardInterrupt``
        and ``SystemExit`` pass through; any other exception is this device's
        ``failed`` (rule 9).

        A device in ``skip`` is not touched at all, and no progress line names
        it: the parent blames the device of the last such line for a crash, and
        the device that crashes a build is the one left out (the Ti-E's
        ``TIDiaLamp``). It is not ``failed`` either: nothing was tried.
        """
        shown = type_name or "Unknown"
        if label in self.skip:
            self.untried += 1
            return ProbedDevice(
                name=label,
                type=shown,
                status="not-probed",
                source=source,
                reason=f"left out: --skip {label}",
            )
        self._say(f"probing {self.name}/{label}")
        port_set: str | None = None
        asked_port = False

        def failed(exc: Exception) -> ProbedDevice:
            if not asked_port:
                # Failed before its Port was looked at: the note about --port
                # must not claim to know whether this device needed one.
                self.untried += 1
            error = str(exc)
            return ProbedDevice(
                name=label,
                type=shown,
                status="failed",
                source=source,
                error=error,
                hint=diagnosis.error_hint(self.name, error),
                port=port_set,
            )

        try:
            self.core.loadDevice(label, self.name, label)
        except Exception as exc:
            return failed(exc)
        if type_name is None:
            shown = _loaded_type(self.core, label)
        try:
            if parent is not None:
                self.core.setParentLabel(label, parent)
        except Exception as exc:
            return failed(exc)
        needs_port = _needs_port(self.core, label)
        asked_port = True
        if needs_port:
            if self.port is None:
                return ProbedDevice(
                    name=label,
                    type=shown,
                    status="not-probed",
                    source=source,
                    reason="needs a port: pass --port COMx to probe it",
                )
            self.port_needed = True
            try:
                self.core.setProperty(label, "Port", self.port)
            except Exception as exc:
                return failed(exc)
            port_set = self.port
        try:
            self.core.initializeDevice(label)
        except Exception as exc:
            return failed(exc)
        return ProbedDevice(
            name=label, type=shown, status="ok", source=source, port=port_set
        )

    def _device(
        self,
        label: str,
        type_name: str | None,
        source: ProbeSource,
        parent: str | None,
    ) -> ProbedDevice:
        row = self._try(label, type_name, source, parent)
        if row.status == "failed" and type_name == _HUB_TYPE and not self.result.causes:
            # Set before the row's checkpoint, so that a crash that follows
            # keeps the diagnosis together with the failed hub (rule 7).
            self.result.causes = diagnosis.hub_failure_causes(self.name)
        self._add(row)
        return row

    # --- the walk ------------------------------------------------------------
    def _listing(self) -> list[AdapterDevice] | None:
        """Rule 1: the adapter's devices, or the reason there is nothing to probe."""
        result = self.result
        if self.name not in self.core.getDeviceAdapterNames():
            result.error = "adapter not installed"
            return None
        try:
            (info,) = list_adapters(self.core, [self.name])
        except Exception as exc:
            result.error = str(exc)
            return None
        if info.error is not None:
            result.error = info.error
            return None
        if not info.devices:
            result.error = (
                f"{self.name} lists no devices: "
                f"{diagnosis.empty_list_meaning(self.name)}"
            )
            return None
        return info.devices

    def _connect(self, hubs: list[AdapterDevice]) -> str | None:
        """Rule 2: the first hub that initialises; the others are not tried."""
        connected: str | None = None
        failed: list[str] = []
        for hub in hubs:
            if connected is not None:
                self._skip(
                    hub.name,
                    hub.type,
                    f"hub {connected} of this adapter is already initialised; "
                    "a probe connects one hub",
                )
                continue
            row = self._device(hub.name, hub.type, "adapter", parent=None)
            if row.status == "ok":
                connected = hub.name
            elif row.status == "failed":
                failed.append(hub.name)
        if connected is not None:
            for name in failed:
                self.result.notes.append(
                    f"hub {name} did not initialise but {connected} did: "
                    f"what follows is {connected}'s"
                )
        return connected

    def _below(self, hub: str, listed: list[AdapterDevice]) -> None:
        """Rule 4: what the initialised hub reports, loaded under it."""
        result = self.result
        self._say(f"probing {self.name}/{hub}: listing its peripherals")
        list_error: str | None = None
        try:
            names = [str(n) for n in self.core.getInstalledDevices(hub)]
        except Exception as exc:
            list_error = str(exc)
            names = []
        taken = {d.name for d in result.devices}
        reported: list[str] = []
        for name in names:
            if name not in taken and name not in reported:
                reported.append(name)

        if reported:
            by_name = {d.name: d for d in listed}
            for name in reported:
                known = by_name.get(name)
                self._device(
                    name,
                    known.type if known else None,
                    "both" if known else "hub",
                    parent=hub,
                )
            for device in listed:
                if device.name not in reported:
                    self._skip(
                        device.name,
                        device.type,
                        f"listed by the adapter but not reported by the hub {hub}: "
                        "not attached, or not detected",
                    )
        elif listed:
            # Some hubs do not detect, and their devices still work.
            if list_error is None:
                result.notes.append(
                    f"hub {hub} reports no peripherals: the adapter's own list was probed"
                )
            else:
                result.notes.append(
                    f"hub {hub} did not list its peripherals: {list_error}; "
                    "the adapter's own list was probed"
                )
            for device in listed:
                self._device(device.name, device.type, "adapter", parent=hub)
        else:
            what = (
                "reports no peripherals"
                if list_error is None
                else f"could not list its peripherals ({list_error})"
            )
            result.error = (
                f"hub {hub} initialised but {what}, and {self.name} lists no "
                "other device: nothing below it was probed; "
                f"{diagnosis.empty_list_meaning(self.name)}"
            )

    def run(self) -> ProbeResult:
        listed = self._listing()
        if listed is None:
            return self.result
        # Once before the first device, so a child that dies on the first one
        # still leaves a result that is marked as not whole (§9.5).
        self._checkpoint()

        hubs = [d for d in listed if d.type == _HUB_TYPE]
        others = [d for d in listed if d.type != _HUB_TYPE]
        connected = self._connect(hubs)
        if connected is not None:
            self._below(connected, others)
        elif hubs:
            # Rule 6: nothing below a hub that did not initialise is loaded.
            for device in others:
                self._skip(
                    device.name,
                    device.type,
                    f"its hub {hubs[0].name} was not initialised",
                )
        else:
            # Rule 5: an adapter with no hub (Utilities); no parent.
            for device in others:
                self._device(device.name, device.type, "adapter", parent=None)

        if self.port is not None and not self.port_needed:
            tried = (
                f"no device of {self.name} needs a port"
                if not self.untried
                else "no device that was initialised needs a port "
                f"({self.untried} not tried)"
            )
            self.result.notes.append(f"--port {self.port} was not used: {tried}")
        return self.result


def walk_adapter(
    core: CMMCorePlus,
    name: str,
    *,
    port: str | None = None,
    skip: Collection[str] = (),
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[ProbeResult], None] | None = None,
) -> ProbeResult:
    """Initialise the devices of one adapter on a core the caller owns.

    A row never says ``ok`` for something nobody initialised (design m2 §9.3):
    a device the adapter lists and its hub does not report, and a device that
    needs a port nobody gave, are ``not-probed`` with their reason, and are
    never initialised. A probe connects one hub: the Ti2-E lists the stand and
    then its simulator, and the simulator's expected failure on a connected
    stand is not shown as a fault of the stand.

    Every device it loaded stays loaded: ``smc config build`` (#15) builds on
    what initialised, so this never closes the core; :func:`probe_adapter` does.

    Args:
        core: The core to load into; the caller releases it.
        name: The adapter.
        port: A serial port to set, before initialising, on each device whose
            pre-init ``Port`` is unset. A device that already has one keeps it.
        skip: Names of devices never to load (``smc config build --skip``), for
            a device that crashes the process that initialises it. Each is a
            ``not-probed`` row, not a failure; a skipped hub lets the next
            listed hub connect (design m2 §10.2).
        progress: Called with ``"probing <adapter>/<device>"`` before each
            device is loaded, so a crash can be attributed to a device.
        checkpoint: Called with a deep copy of the result, ``complete=False``,
            once before the first device and after every row, so a child that
            dies keeps the rows it had finished (FM-02).

    Raises:
        ValueError: ``port`` is refused by :func:`check_port`.
    """
    checked = check_port(port) if port is not None else None
    return _Walk(core, name, checked, skip, progress, checkpoint).run()


def probe_adapter(
    name: str,
    *,
    port: str | None = None,
    skip: Collection[str] = (),
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[ProbeResult], None] | None = None,
    core: CMMCorePlus | None = None,
) -> ProbeResult:
    """Walk one adapter in a throwaway core, then release the stand.

    ``core`` exists so that a test hands over a stub: no test lists a vendor
    adapter (FM-40). The core is released in a ``finally`` whichever core it is,
    the one opened here or the one given, and whatever ends the walk (a Ctrl-C
    included): a stand stays held until its devices are unloaded.

    Args:
        name: The adapter.
        port: See :func:`walk_adapter`.
        skip: See :func:`walk_adapter`.
        progress: See :func:`walk_adapter`.
        checkpoint: See :func:`walk_adapter`.
        core: A core to use instead of a new ``CMMCorePlus``.
    """
    if core is None:
        from pymmcore_plus import CMMCorePlus

        core = CMMCorePlus()
    try:
        return walk_adapter(
            core,
            name,
            port=port,
            skip=skip,
            progress=progress,
            checkpoint=checkpoint,
        )
    finally:
        core_mod.close_core(core)


#: macOS signal numbers the child's crash guard turns into exit codes
#: (``_no_error_dialogs`` in :mod:`smc.discovery._mm_child`).
_DARWIN_GUARDED_SIGNALS = {
    4: "SIGILL",
    6: "SIGABRT",
    8: "SIGFPE",
    10: "SIGBUS",
    11: "SIGSEGV",
}


def exit_detail(returncode: int, platform_name: str) -> str:
    """The child's exit code, with the signal it most likely stands for.

    On macOS the child's crash guard ends a crashing process with
    ``_exit(signum)``, so an ``abort()`` shows up as exit code 6, not -6. A
    genuine exit code with the same value reads the same, hence "likely".
    """
    if platform_name == "darwin" and returncode in _DARWIN_GUARDED_SIGNALS:
        name = _DARWIN_GUARDED_SIGNALS[returncode]
        return f"{returncode} (likely {name}, crash guard)"
    return str(returncode)


#: The child's progress lines on stderr, so a note can name what was running.
_PROGRESS_PREFIXES = ("listing ", "probing ", "reading ")


def _last_progress(stderr_path: Path) -> str | None:
    try:
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        if line.startswith(_PROGRESS_PREFIXES):
            return line.strip()
    return None


def mm_section(
    adapters: list[str],
    *,
    all_adapters: bool = False,
    probe: str | None = None,
    port: str | None = None,
    timeout_s: float = CHILD_TIMEOUT_S,
) -> tuple[list[AdapterInfo], ProbeResult | None, list[str]]:
    """Run the adapter work in a child process and collect what it found.

    The result travels through a file, never the child's stdout: a vendor
    DLL may print while loading (FM-01). The child's stdout and stderr go to
    files too, so a helper process that inherits them cannot keep a pipe
    open and defeat the timeout (FM-04). The result file is read whatever
    the exit code: a crash or hang at teardown, after the result was
    written, costs a note and nothing else (FM-02). The child runs in the
    temporary folder, so the operator's working directory is not on its
    ``sys.path``.

    The child rewrites the file after every probed device, so a child that
    dies mid-probe leaves the rows it had finished, marked
    ``complete=False``.

    Args:
        adapters: The adapters to list.
        all_adapters: List every installed adapter too.
        probe: One adapter to initialise.
        port: The serial port for the probe's devices that need one. It
            travels as one argument, ``--port=<port>``: a port that starts
            with ``-`` would otherwise be read by ``argparse`` as an option.
            Ignored without ``probe``.
        timeout_s: How long the child may run.

    Returns:
        The listed adapters, the probe (``None`` unless asked, or when the
        child died before its first checkpoint), and notes naming what went
        wrong and what the child was doing at the time.

    Raises:
        ValueError: ``port`` is refused by :func:`check_port`.
    """
    checked_port = check_port(port) if probe and port is not None else None
    arguments = ["--adapters", ",".join(adapters)]
    if all_adapters:
        arguments.append("--all")
    if probe:
        arguments += ["--probe", probe]
    if checked_port is not None:
        arguments.append(f"--port={checked_port}")
    result, notes, _ = _run_child(
        arguments,
        prefix="adapters: the adapter child",
        timeout_s=timeout_s,
        missing_suffix="; device lists missing, adapter names kept",
    )
    if result is None:
        return [], None, notes
    return result.listed, result.probe, notes


def _run_child(
    arguments: Sequence[str],
    *,
    prefix: str,
    timeout_s: float,
    missing_suffix: str = "",
) -> tuple[ChildResult | None, list[str], str | None]:
    """Run the adapter child with ``arguments`` under a deadline; read what it wrote.

    The one runner of :func:`mm_section` and :func:`build_section` (design m2
    §10.3), so that the lessons of FM-01 to FM-04, FM-35 and FM-83 are kept in
    one place. The result travels through a file, never the child's stdout: a
    vendor DLL may print while loading (FM-01). The child's stdout and stderr go
    to files too, so a helper process that inherits them cannot keep a pipe open
    and defeat the timeout (FM-04). The result file is read whatever the exit
    code: a crash or hang at teardown, after the result was written, costs a
    note and nothing else (FM-02). The child runs in the temporary folder, so
    the operator's working directory is not on its ``sys.path``.
    ``CHILD_COMMAND`` is read here, when the child starts, so a test can replace
    it with a child that crashes or hangs.

    Args:
        arguments: The child's arguments, after ``--result FILE`` is added.
        prefix: What every note starts with: ``adapters: the adapter child``.
        timeout_s: How long the child may run.
        missing_suffix: Added to the note of a child that failed and left no
            result file; a child that left none and said nothing gets the note
            ``<prefix> wrote no result``.

    Returns:
        The result, ``None`` when the child wrote nothing usable; the notes;
        and the last progress line of a child that did not exit 0 or was
        stopped, ``None`` for one that finished.
    """
    with tempfile.TemporaryDirectory(
        prefix="smc-discover-", ignore_cleanup_errors=True
    ) as tmp:
        folder = Path(tmp)
        result_path = folder / "result.json"
        stderr_path = folder / "stderr.txt"
        command = [*CHILD_COMMAND, *arguments, "--result", str(result_path)]
        logger.info("adapter child: %s", " ".join(command[1:]))

        notes: list[str] = []
        with (
            (folder / "stdout.txt").open("wb") as out,
            stderr_path.open("wb") as err,
        ):
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    cwd=folder,
                    # So the whole tree can be killed together (FM-35);
                    # ignored on Windows, which uses creationflags instead.
                    start_new_session=True,
                    creationflags=_NO_WINDOW,
                )
            except OSError as exc:
                return None, [f"{prefix} could not start: {exc}"], None
            unstoppable = False
            kill_detail: str | None = None
            try:
                try:
                    returncode = _wait_in_slices(child, timeout_s)
                except subprocess.TimeoutExpired:
                    kill_detail = _kill_child_tree(child)
                    returncode = None
                    try:
                        # Bounded: a process stuck in a driver call may not die.
                        _wait_in_slices(child, _KILL_WAIT_S)
                    except subprocess.TimeoutExpired:
                        unstoppable = True
            except BaseException:
                # Ctrl-C while waiting must not leave a helper process running
                # and holding the hardware (FM-35).
                _kill_child_tree(child)
                raise

        doing = _last_progress(stderr_path)
        during = f" while {doing}" if doing else ""
        if returncode is None:
            # The direct child dying (unstoppable stays False) does not mean
            # the tree died: if the tree kill itself failed (kill_detail),
            # a helper it started may still be running and holding the
            # hardware, the exact case this wording exists to flag (FM-35).
            stopped = (
                "could not be stopped (it may still hold the hardware)"
                if unstoppable or kill_detail
                else "was stopped"
            )
            detail = f"; {kill_detail}" if kill_detail else ""
            notes.append(
                f"{prefix} did not finish in {timeout_s:.0f} s "
                f"and {stopped}{during}{detail}"
            )
        elif returncode != 0:
            notes.append(
                f"{prefix} exited {exit_detail(returncode, sys.platform)}{during}"
            )
        # A child that finished was not on any device when it ended.
        last = doing if returncode != 0 else None

        if not result_path.exists():
            if notes:
                notes[-1] += missing_suffix
            else:
                notes.append(f"{prefix} wrote no result")
            return None, notes, last
        try:
            result = ChildResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, ValidationError) as exc:
            notes.append(f"{prefix}'s result could not be read: {exc}")
            return None, notes, last
    return result, notes, last


#: What the walk says while it asks a hub for its peripherals: the hub is the
#: device it is on, and the one to leave out when the child dies there.
_LISTING_PERIPHERALS = ": listing its peripherals"


def _stopped_at(doing: str | None) -> str | None:
    """The device a child was on, from its last progress line.

    ``probing <adapter>/<device>`` and ``reading <adapter>/<device>`` name one;
    ``listing <adapter>`` does not (a crash there is the adapter's, not a
    device's). The adapter's name has no ``/``, so the first one splits it.
    """
    if doing is None:
        return None
    for prefix in ("probing ", "reading "):
        if doing.startswith(prefix):
            _, slash, device = doing[len(prefix) :].partition("/")
            device = device.removesuffix(_LISTING_PERIPHERALS)
            return device if slash and device else None
    return None


#: Two adapters, each a connection and a walk; a probe's 180 s is not verified
#: on a hub either (design m2 §9.10, §10.8).
BUILD_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class BuildRun:
    """What :func:`build_section` brings back."""

    #: ``None``: the child wrote nothing usable.
    scan: BuildScan | None
    notes: list[str]
    #: The device the child was on when it died or was stopped, the one
    #: ``--skip`` would leave out; ``None`` for a child that finished.
    stopped_at: str | None


def build_section(
    stand: str,
    *,
    camera: str | None = None,
    skip: Sequence[str] = (),
    search_paths: Sequence[Path] = (),
    timeout_s: float = BUILD_TIMEOUT_S,
) -> BuildRun:
    """Initialise the devices of a stand's adapter (and a camera's) in a child.

    ``smc config build`` writes a file from what answers here. It runs in the
    adapter child for the reasons of :func:`mm_section` (FM-01 to FM-04): a
    device that crashes or hangs costs the build, never this process, and
    ``stopped_at`` says which device it was, so the next run can leave it out.
    Every value travels as one ``--opt=value`` argument, so that a name that
    starts with ``-`` is a value and not an option (design m2 §9.1).

    Args:
        stand: The stand's adapter.
        camera: The adapter of a camera, walked on the same core.
        skip: Devices never to load; each is checked by
            :func:`check_device_name`.
        search_paths: Adapter folders to search first, as the profile's stand
            will be opened with (``Profile.adapter_search_dirs()``).
        timeout_s: How long the child may run.

    Raises:
        ValueError: ``stand`` is empty or a ``skip`` name is refused by
            :func:`check_device_name`; no child is started.
    """
    if not stand.strip():
        raise ValueError("the stand's adapter name is empty")
    arguments = [f"--build={stand}"]
    if camera:
        arguments.append(f"--camera={camera}")
    arguments += [f"--skip={check_device_name(name)}" for name in skip]
    arguments += [f"--search-path={path}" for path in search_paths]
    result, notes, doing = _run_child(
        arguments, prefix="build: the build child", timeout_s=timeout_s
    )
    scan = result.build if result is not None else None
    return BuildRun(scan=scan, notes=notes, stopped_at=_stopped_at(doing))
