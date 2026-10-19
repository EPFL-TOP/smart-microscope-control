"""What the layer knows about the two Nikon stands, as facts and file checks.

Pure Python and the file system: this module never imports ``pymmcore_plus`` and
never opens a core. The one thing that loads a DLL, the loader probe, runs in a
child process of its own (:mod:`smc.discovery.dll_probe`, design m2 §7.3).

Why a module of its own, not a plugin or a profile line (CLAUDE.md: roles, not
device names): the two adapters differ in what an empty device list means, where
their vendor driver is found and what usually keeps a stand from opening, and
those are facts about Nikon's SDKs, not about one microscope. Everything here is
measured on the Ti2-E (session 0, #126) except what says it comes from
``nikon-control``:

* ``NikonTi2`` asks Nikon's SDK what is on the stand, so an empty list means the
  SDK could not be reached. ``NikonTI`` publishes a fixed list of 18 devices,
  listed with no microscope attached, so a list proves nothing.
* ``NikonTi2`` loads ``Ti2_Mic_Driver.dll`` from **its own folder**; on our Ti2-E
  the copy beside the adapter had the SHA-256 of the one in the SDK folder.
  ``NikonTI`` finds ``NikonTi.dll`` on the system path, where Nikon's installer
  puts it (``nikon-control``, not seen here).
* MMCore says ``Failed to load device adapter`` for a missing dependency, a wrong
  SDK, the wrong bitness and an interface mismatch alike (FM-09), so
  :func:`diagnosis` lists what is certain, then what Windows said, then what
  usually keeps a stand from opening, most likely first.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from smc.discovery import dll_probe
from smc.discovery.dll_probe import DllProbe

__all__ = [
    "MAX_SEARCH_DEPTH",
    "ONE_CONNECTION",
    "STANDS",
    "TI2",
    "TIE",
    "Driver",
    "DriverLocation",
    "Stand",
    "StandCheck",
    "check_stand",
    "deep_check",
    "diagnosis",
    "find_driver",
    "install_driver",
    "readiness",
    "stand_for_adapter",
    "stand_for_key",
    "stands_in_config",
]

log = logging.getLogger("smc.hardware.backends.nikon")

#: How far below an adapter folder a misplaced driver is looked for. A profile
#: may list any folder as an adapter folder (FM-74): the search is bounded in
#: depth and in folders visited, and "not found there" is all it can claim.
MAX_SEARCH_DEPTH = 3
_MAX_FOLDERS = 2000

#: What a fingerprint shows: enough to tell two copies of a DLL apart.
_DIGEST_SHOWN = 12


@dataclass(frozen=True, slots=True)
class Driver:
    """The vendor DLL an adapter needs, and where it is found."""

    dll: str  # "Ti2_Mic_Driver.dll"
    sdk_dir: str  # a Windows path, where the vendor installer puts it
    installer: str  # what to install to get it
    beside_adapter: bool  # True: the adapter loads it from its own folder


@dataclass(frozen=True, slots=True)
class Stand:
    """One Nikon stand: its adapter, its driver and what usually goes wrong."""

    key: str  # "ti2" | "tie": the argument of `smc install-driver`
    adapter: str  # "NikonTi2" | "NikonTI"
    name: str  # "Nikon Ti2-E" | "Nikon Ti-E"
    hub: str  # "" when the SDK names it (Ti2-E__0 on ours) | "TIScope"
    dynamic: bool  # the device list comes from the SDK
    driver: Driver
    causes: tuple[str, ...]  # likely causes after ONE_CONNECTION, in order

    @property
    def adapter_file(self) -> str:
        """The adapter's file name; Nikon's adapters exist for Windows only."""
        return f"mmgr_dal_{self.adapter}.dll"

    @property
    def empty_list_meaning(self) -> str:
        """What an empty device list says about this adapter (used by ``smc discover``)."""
        if self.dynamic:
            return (
                "the adapter asks Nikon's SDK what is on the stand, so an empty "
                "device list means the SDK could not be reached, not that the "
                "adapter is missing"
            )
        return (
            "the adapter publishes a fixed device list, so a device list proves "
            "nothing about what is connected; only a probe does"
        )


@dataclass(frozen=True, slots=True)
class DriverLocation:
    """Where a stand's driver was found, and whether its adapter will find it there."""

    path: Path | None  # None: not found
    beside_adapter: bool  # in the adapter's own folder
    satisfied: bool  # where this stand's adapter finds it


@dataclass(frozen=True, slots=True)
class StandCheck:
    """What the file system says about one stand, before anything is opened."""

    stand: Stand
    adapter_path: Path | None  # the adapter MMCore would load, None: none
    searched: tuple[Path, ...]  # the adapter folders, in MMCore's order
    driver: DriverLocation


#: The first likely cause, shared by both stands: a stand accepts one connection
#: at a time, and the program that holds it is usually the operator's own.
ONE_CONNECTION = (
    "another program holds the stand, which accepts one connection at a time; "
    "close NIS-Elements, Ti2 Control, Micro-Manager and any other smc or "
    "pymmcore session, then run this again"
)

TI2 = Stand(
    key="ti2",
    adapter="NikonTi2",
    name="Nikon Ti2-E",
    hub="",
    dynamic=True,
    driver=Driver(
        dll="Ti2_Mic_Driver.dll",
        sdk_dir=r"C:\Program Files\Nikon\Ti2-SDK\bin",
        installer="Ti2 Control (the Ti2 SDK)",
        beside_adapter=True,
    ),
    causes=(
        "power: the stand or its controller is off, or was powered in the wrong "
        "order; power the stand before the controller box, or only "
        "Ti2-Simulator answers",
        "the driver: Ti2_Mic_Driver.dll is missing beside the adapter, or is not "
        "the version the adapter was built for; Ti2Sample.exe from the Ti2 SDK "
        "tests the SDK without Micro-Manager",
        "Ti2 Control 2.10 and 2.20 crash Micro-Manager when the nosepiece is "
        "added (mmCoreAndDevices#44); 2.00 is reported to work and 2.90.108 "
        "works on ours",
    ),
)

TIE = Stand(
    key="tie",
    adapter="NikonTI",
    name="Nikon Ti-E",
    hub="TIScope",
    dynamic=False,
    driver=Driver(
        dll="NikonTi.dll",
        sdk_dir=r"C:\Program Files\Nikon\Shared\Bin",
        installer="Nikon's Ti Setup Tool (or the TiSDKRedist package)",
        beside_adapter=False,
    ),
    causes=(
        "power and USB: the stand or its controller is off, or it is on a USB 3 "
        "port, a documented source of connection failures on this generation; "
        "use a USB 2 port",
        "the driver: NikonTi.dll is not installed, or not on the system path",
        "TIDiaLamp: it has crashed Micro-Manager with some driver versions; "
        "leave it out of the configuration unless it is needed",
    ),
)

STANDS: tuple[Stand, ...] = (TI2, TIE)

# Windows finds mmgr_dal_nikonti2.dll as it finds mmgr_dal_NikonTi2.dll.
_BY_ADAPTER = {stand.adapter.casefold(): stand for stand in STANDS}
_BY_KEY = {stand.key: stand for stand in STANDS}


def stand_for_adapter(adapter: str) -> Stand | None:
    """The Nikon stand an adapter belongs to; ``None`` for any other adapter."""
    return _BY_ADAPTER.get(adapter.strip().casefold())


def stand_for_key(key: str) -> Stand | None:
    """The stand for the argument of ``smc install-driver``; ``None`` if unknown."""
    return _BY_KEY.get(key.strip().casefold())


def stands_in_config(path: Path) -> list[Stand]:
    """The Nikon stands a Micro-Manager ``.cfg`` loads, in order of first appearance.

    It reads the *library* column of the ``Device,<label>,<library>,<name>``
    lines, never the label: a camera that someone labelled ``NikonTi2`` is not a
    stand. Comments are skipped and each stand counts once. A file that cannot be
    read gives ``[]``: ``doctor`` is about to open it and will say why it cannot.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list[Stand] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or fields[0] != "Device":
            continue
        stand = stand_for_adapter(fields[2])
        if stand is not None and stand not in found:
            found.append(stand)
    return found


def _is_file(path: Path) -> bool:
    """Whether ``path`` is a file; a folder this account may not search is "no"."""
    try:
        return path.is_file()
    except OSError:
        return False


def _find_below(root: Path, name: str) -> Path | None:
    """A file called ``name`` somewhere below ``root``, within the search bounds.

    Breadth first, so the nearest copy wins. A folder that cannot be listed is
    "not found there", never an error: ``doctor`` must go on to its next check.
    """
    wanted = name.casefold()
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    visited = 0
    while queue and visited < _MAX_FOLDERS:
        folder, depth = queue.popleft()
        visited += 1
        try:
            with os.scandir(folder) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
                for entry in children:
                    if entry.name.casefold() == wanted and entry.is_file():
                        return Path(entry.path)
                if depth < MAX_SEARCH_DEPTH:
                    queue.extend(
                        (Path(entry.path), depth + 1)
                        for entry in children
                        if entry.is_dir(follow_symlinks=False)
                    )
        except OSError:
            continue
    return None


def find_driver(
    stand: Stand, adapter_dir: Path | None, *, sdk_dir: Path | None = None
) -> DriverLocation:
    """Where ``stand``'s driver is, and whether its adapter will find it there.

    Looked for beside the adapter, then below the adapter's folder, then in the
    SDK folder (``sdk_dir``, by default the one the installer uses). ``satisfied``
    is what matters: the Ti2 adapter loads its driver from its own folder only,
    so a copy in a subfolder or in the SDK folder is a finding, not a pass; the
    Ti-E adapter finds its driver on the system path, i.e. in the SDK folder.

    Args:
        stand: The stand.
        adapter_dir: The folder of the adapter MMCore would load; ``None`` when
            there is none, and only the SDK folder is then looked at.
        sdk_dir: Where the vendor installer puts the driver; ``None`` for the
            registry's own (a Windows path, so tests pass a folder of their own).
    """
    driver = stand.driver
    sdk = Path(driver.sdk_dir) if sdk_dir is None else sdk_dir
    in_sdk = sdk / driver.dll
    sdk_has_it = _is_file(in_sdk)
    if adapter_dir is not None:
        beside = adapter_dir / driver.dll
        if _is_file(beside):
            return DriverLocation(
                beside,
                beside_adapter=True,
                satisfied=driver.beside_adapter or sdk_has_it,
            )
        nested = _find_below(adapter_dir, driver.dll)
        if nested is not None:
            return DriverLocation(
                nested,
                beside_adapter=False,
                satisfied=not driver.beside_adapter and sdk_has_it,
            )
    if sdk_has_it:
        return DriverLocation(
            in_sdk, beside_adapter=False, satisfied=not driver.beside_adapter
        )
    return DriverLocation(None, beside_adapter=False, satisfied=False)


def check_stand(
    stand: Stand, adapter_dirs: Sequence[Path], *, sdk_dir: Path | None = None
) -> StandCheck:
    """The adapter MMCore would load for ``stand``, and where its driver is.

    MMCore takes each adapter from the first search folder that holds its file
    and does not fall back to the next when that file fails to load (FM-76), so
    the first folder wins, and the driver is looked for beside *that* adapter.

    Args:
        stand: The stand.
        adapter_dirs: The adapter folders in the order MMCore searches them: the
            profile's ``adapter_search_paths``, then the install's.
        sdk_dir: As for :func:`find_driver`.
    """
    adapter_path = next(
        (
            folder / stand.adapter_file
            for folder in adapter_dirs
            if _is_file(folder / stand.adapter_file)
        ),
        None,
    )
    driver = find_driver(
        stand,
        adapter_path.parent if adapter_path is not None else None,
        sdk_dir=sdk_dir,
    )
    return StandCheck(
        stand=stand,
        adapter_path=adapter_path,
        searched=tuple(adapter_dirs),
        driver=driver,
    )


def readiness(check: StandCheck) -> list[tuple[str, str]]:
    """The rows ``smc doctor`` prints before it opens a stand: ``adapter``, ``driver``.

    They never decide an exit code: a driver that sits in the SDK folder instead
    of beside the adapter is exactly what the stand's own open will confirm or
    refute, and a ``.cfg`` can name an adapter this install does not need.
    """
    stand, driver, found = check.stand, check.stand.driver, check.driver
    if check.adapter_path is not None:
        adapter = str(check.adapter_path)
    elif check.searched:
        adapter = "not found in " + "; ".join(str(folder) for folder in check.searched)
    else:
        adapter = "not found (there is no adapter folder to search)"

    if found.path is None:
        where = "" if driver.beside_adapter else f" in {driver.sdk_dir}"
        text = f"{driver.dll} not found{where}: install {driver.installer}"
    elif found.satisfied and driver.beside_adapter:
        text = f"{driver.dll} beside the adapter"
    elif found.satisfied:
        text = f"{driver.dll} in {found.path.parent}"
    elif driver.beside_adapter and check.adapter_path is None:
        text = (
            f"{driver.dll} is in {found.path.parent}; there is no adapter folder "
            "to put it beside yet"
        )
    elif driver.beside_adapter:
        text = (
            f"{driver.dll} is in {found.path.parent}, not beside the adapter: "
            f"smc install-driver {stand.key} copies it"
        )
    else:
        text = (
            f"{driver.dll} is in {found.path.parent}, not in {driver.sdk_dir}, "
            "where the adapter finds it on the system path"
        )
    return [("adapter", adapter), ("driver", text)]


def deep_check(check: StandCheck, *, extra_path: Sequence[Path] = ()) -> list[DllProbe]:
    """Ask Windows to load the adapter, then the driver, each in a child of its own.

    Called after a stand failed to open, when MMCore has unloaded everything it
    had loaded (design m2 §7.1), so nothing here competes with a core for the
    stand or holds a vendor DLL in this process. What was not found is not
    probed: the ``found:`` lines of :func:`diagnosis` say why. Off Windows each
    probe says it was not tried; it is never a failure.

    Args:
        check: What :func:`check_stand` found.
        extra_path: The adapter folders, for the front of each child's ``PATH``,
            as ``open_core`` puts them there.
    """
    targets = (check.adapter_path, check.driver.path)
    return [
        dll_probe.probe_dll(target, extra_path=extra_path)
        for target in targets
        if target is not None
    ]


def _findings(check: StandCheck) -> list[str]:
    """What the file system proves, one ``found:`` line each."""
    stand, driver, found = check.stand, check.stand.driver, check.driver
    lines: list[str] = []
    if check.adapter_path is None:
        where = (
            "none of the adapter folders searched (first wins): "
            + "; ".join(str(folder) for folder in check.searched)
            if check.searched
            else "no folder: there is no adapter folder to search"
        )
        lines.append(f"found: {stand.adapter_file} is in {where}")
    if found.path is None:
        where = "" if driver.beside_adapter else f" in {driver.sdk_dir}"
        lines.append(
            f"found: {driver.dll} was not found{where}; install {driver.installer}"
        )
    elif not found.satisfied and driver.beside_adapter:
        if check.adapter_path is not None:
            lines.append(
                f"found: {driver.dll} is in {found.path.parent}, not beside the "
                f"adapter ({check.adapter_path.parent}), and the adapter loads it "
                f"from its own folder only; copy it with: smc install-driver "
                f"{stand.key}"
            )
    elif not found.satisfied:
        lines.append(
            f"found: {driver.dll} is in {found.path.parent}, not in "
            f"{driver.sdk_dir}, where the adapter finds it on the system path; "
            f"install {driver.installer}"
        )
    return lines


def _probe_line(probe: DllProbe) -> str:
    """One probe as a line; no verdict is "not probed", never a failure."""
    name = probe.path.name
    if probe.loads is True:
        return f"probe: {name} loads"
    if probe.loads is False:
        return f"probe: {name} will not load: {probe.detail}"
    return f"probe: {name} not probed: {probe.detail}"


def diagnosis(check: StandCheck, probes: Sequence[DllProbe]) -> list[str]:
    """Why a stand did not open: what is certain, what Windows said, what is likely.

    MMCore's message for a stand that does not open names no cause, so the lines
    come in the order of how much each one proves: ``found:`` for what the file
    system shows, ``probe:`` for what Windows' own loader said, then the likely
    causes in the order people hit them, the first being another program that
    already holds the stand. The text does not parse MMCore's multi-line message;
    it only follows it.

    Args:
        check: What :func:`check_stand` found.
        probes: What :func:`deep_check` returned, in its order.
    """
    lines = _findings(check)
    lines.extend(_probe_line(probe) for probe in probes)
    lines.append("in order of likelihood:")
    causes = (ONE_CONNECTION, *check.stand.causes)
    lines.extend(f"  {number}. {cause}" for number, cause in enumerate(causes, start=1))
    return lines


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _already_there(src: Path, dest: Path, digest: str) -> tuple[bool, str] | None:
    """The outcome when ``dest`` exists, or ``None`` when it does not."""
    try:
        if not dest.exists():
            return None
        there = _sha256(dest)
    except OSError as exc:
        return False, f"{dest} already exists and cannot be read: {exc}"
    if there == digest:
        return (
            True,
            f"{dest} is already there, identical to {src} "
            f"(SHA-256 {digest[:_DIGEST_SHOWN]})",
        )
    return (
        False,
        f"{dest} already exists and differs from {src} (SHA-256 "
        f"{there[:_DIGEST_SHOWN]} there, {digest[:_DIGEST_SHOWN]} here); smc never "
        "overwrites a driver, so remove it yourself if it is the wrong one",
    )


def install_driver(
    stand: Stand, adapter_dir: Path, *, source: Path | None = None
) -> tuple[bool, str]:
    """Copy ``stand``'s driver into the folder its adapter loads it from.

    **Copy, never move**: the vendor's own software still needs its copy. Never
    overwrite either: a driver that differs may be the version this adapter was
    built for, and smc cannot tell, so it says so and stops. The copy goes to
    ``<dll>.partial`` in the destination folder and is renamed onto the target,
    so an interrupted copy leaves no truncated DLL where the adapter would load it
    (FM-38). The look at the destination is repeated just before the rename,
    which narrows the window in which another installer's file could be replaced;
    ``os.replace`` cannot close it.

    Args:
        stand: The stand.
        adapter_dir: The folder of the stand's adapter in the Micro-Manager install.
        source: The driver to copy; by default the SDK folder's own.

    Returns:
        ``(ok, message)``: ``ok`` when the driver is in place (copied, or an
        identical copy was already there), with a one-line message either way.
    """
    driver = stand.driver
    if not driver.beside_adapter:
        return (
            False,
            f"{stand.name}'s adapter finds {driver.dll} on the system path, not "
            f"beside itself, so there is nothing to copy; install "
            f"{driver.installer} instead",
        )
    if not adapter_dir.is_dir():
        return False, f"{adapter_dir} is not a folder"
    src = Path(driver.sdk_dir) / driver.dll if source is None else source
    try:
        if not src.is_file():
            return (
                False,
                f"{src} does not exist; install {driver.installer}, or pass "
                "--from with the path of the DLL",
            )
        digest = _sha256(src)
    except OSError as exc:
        return False, f"could not read {src}: {exc}"

    dest = adapter_dir / driver.dll
    existing = _already_there(src, dest, digest)
    if existing is not None:
        return existing
    partial = dest.with_name(dest.name + ".partial")
    try:
        shutil.copy2(src, partial)
        if _sha256(partial) != digest:
            raise OSError("the copy does not match the source")
        late = _already_there(src, dest, digest)
        if late is not None:
            _discard(partial)
            return late
        os.replace(partial, dest)
    except OSError as exc:
        _discard(partial)
        return False, (
            f"could not copy to {dest}: {exc} (the folder may need an elevated prompt)"
        )
    except BaseException:
        _discard(partial)
        raise
    log.info("installed %s into %s", driver.dll, adapter_dir)
    return True, f"copied {src} to {dest} (SHA-256 {digest[:_DIGEST_SHOWN]})"


def _discard(partial: Path) -> None:
    """Remove a partial copy; one that cannot be removed is not worth a second error."""
    with contextlib.suppress(OSError):
        partial.unlink(missing_ok=True)
