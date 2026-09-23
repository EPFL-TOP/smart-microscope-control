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

import logging
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from smc import __version__
from smc.discovery.models import (
    AdapterDevice,
    AdapterInfo,
    ProbedDevice,
    ProbeResult,
    SystemInfo,
)
from smc.hardware import core as core_mod

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = [
    "CHILD_COMMAND",
    "CHILD_TIMEOUT_S",
    "ChildResult",
    "exit_detail",
    "list_adapters",
    "mm_section",
    "other_installs",
    "probe_adapter",
    "system_section",
]

logger = logging.getLogger("smc.discovery.mm_inventory")

#: Loading every DLL of a full nightly takes a while on a slow lab PC.
CHILD_TIMEOUT_S = 180.0

#: How the parent starts the child; a module attribute so tests can replace
#: it with a child that crashes or hangs.
CHILD_COMMAND: list[str] = [sys.executable, "-m", "smc.discovery._mm_child"]

_HUB_TYPE = "HubDevice"

#: How long to wait for a killed child to go away before giving up on it.
_KILL_WAIT_S = 10.0


class ChildResult(BaseModel):
    """What the child prints on its last stdout line, as ASCII JSON."""

    listed: list[AdapterInfo] = Field(default_factory=list)
    probe: ProbeResult | None = None


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
        python=platform.python_version(),
        smc=__version__,
        pymmcore_plus=str(pymmcore_plus.__version__),
        mm_install=str(st.install_dir) if st.install_dir else None,
        device_api=st.api_version,
        other_mm_installs=other_installs(st.install_dir),
    )
    return info, list(st.adapters)


def _type_name(value: object) -> str:
    from pymmcore_plus import DeviceType

    try:
        return str(DeviceType(int(value)).name)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return str(value)


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
        result.append(
            AdapterInfo(
                name=name,
                installed=True,
                devices=[
                    AdapterDevice(
                        name=str(d), type=_type_name(t), description=str(desc)
                    )
                    for d, t, desc in zip(devices, types, descriptions, strict=False)
                ],
            )
        )
    return result


def probe_adapter(
    name: str, *, progress: Callable[[str], None] | None = None
) -> ProbeResult:
    """Load and initialise every device of one adapter in a throwaway core.

    The ``Hub`` device is loaded first and becomes every other device's
    parent, because a hub-based adapter's peripherals only initialise below
    their hub. Each device's outcome is recorded; everything is unloaded
    afterwards so the stand is released.

    Args:
        name: The adapter.
        progress: Called with ``"probing <adapter>/<device>"`` before each
            device is loaded, so a crash can be attributed to a device.
    """
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    if name not in core.getDeviceAdapterNames():
        return ProbeResult(adapter=name, error="adapter not installed")
    try:
        (info,) = list_adapters(core, [name])
    except Exception as exc:
        return ProbeResult(adapter=name, error=str(exc))
    if info.error is not None:
        return ProbeResult(adapter=name, error=info.error)

    ordered = sorted(info.devices, key=lambda d: d.type != _HUB_TYPE)
    probed = []
    hub_label: str | None = None
    try:
        for device in ordered:
            if progress is not None:
                progress(f"probing {name}/{device.name}")
            try:
                core.loadDevice(device.name, name, device.name)
                if hub_label is not None and device.type != _HUB_TYPE:
                    core.setParentLabel(device.name, hub_label)
                core.initializeDevice(device.name)
            except Exception as exc:
                probed.append(
                    ProbedDevice(
                        name=device.name, type=device.type, ok=False, error=str(exc)
                    )
                )
                continue
            if device.type == _HUB_TYPE and hub_label is None:
                hub_label = device.name
            probed.append(ProbedDevice(name=device.name, type=device.type, ok=True))
    finally:
        core_mod.close_core(core)
    return ProbeResult(adapter=name, devices=probed)


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
_PROGRESS_PREFIXES = ("listing ", "probing ")


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

    Returns:
        The listed adapters, the probe (``None`` unless asked or when the
        child died during it), and notes naming what went wrong and what the
        child was doing at the time.
    """
    with tempfile.TemporaryDirectory(
        prefix="smc-discover-", ignore_cleanup_errors=True
    ) as tmp:
        folder = Path(tmp)
        result_path = folder / "result.json"
        stderr_path = folder / "stderr.txt"
        command = [
            *CHILD_COMMAND,
            "--adapters",
            ",".join(adapters),
            "--result",
            str(result_path),
        ]
        if all_adapters:
            command.append("--all")
        if probe:
            command += ["--probe", probe]
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
                )
            except OSError as exc:
                return (
                    [],
                    None,
                    [f"adapters: the adapter child could not start: {exc}"],
                )
            unstoppable = False
            try:
                returncode = child.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                child.kill()
                returncode = None
                try:
                    # Bounded: a process stuck in a driver call may not die.
                    child.wait(timeout=_KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    unstoppable = True

        doing = _last_progress(stderr_path)
        during = f" while {doing}" if doing else ""
        if returncode is None:
            stopped = (
                "could not be stopped (it may still hold the hardware)"
                if unstoppable
                else "was stopped"
            )
            notes.append(
                f"adapters: the adapter child did not finish in {timeout_s:.0f} s "
                f"and {stopped}{during}"
            )
        elif returncode != 0:
            notes.append(
                "adapters: the adapter child exited "
                f"{exit_detail(returncode, sys.platform)}{during}"
            )

        if not result_path.exists():
            if notes:
                notes[-1] += "; device lists missing, adapter names kept"
            else:
                notes.append("adapters: the adapter child wrote no result")
            return [], None, notes
        try:
            result = ChildResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, ValidationError) as exc:
            notes.append(
                f"adapters: the adapter child's result could not be read: {exc}"
            )
            return [], None, notes
    return result.listed, result.probe, notes
