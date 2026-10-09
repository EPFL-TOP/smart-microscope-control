"""Open a Micro-Manager core, on real hardware or on the demo devices.

Every hardware path in this project goes through one object: a
``CMMCorePlus`` from pymmcore-plus (ADR-0002). This module is the only place
that constructs one, so the tests, the CLI and the future UI agree on how a
core is created, how the Micro-Manager install is found, and what "no
hardware, use the simulator" means.

The demo configuration matters more than it looks: Micro-Manager's
``DemoCamera`` adapter simulates a camera, an XY stage, a Z drive, an
objective turret, a shutter and an autofocus. That is the regression bed of
this project (ADR-0005) — a plugin that cannot run on it is not finished.

Hardware behaviour encoded here (#77):

* **A stand may need an adapter that only the GUI install has.** A profile can
  name the installed MMStudio's folder and ``open_core`` searches it before
  pymmcore-plus's own. That does not fix a device-interface mismatch: an
  adapter whose version differs from pymmcore-plus's still does not load.
* **MMCore accepts a directory that does not exist without a word.** A typo
  would load nothing from it and say nothing, so every directory is checked
  before a core is built.
* **A Windows adapter loads its own DLLs through ``PATH``.** pymmcore-plus puts
  the search paths there, but skips one whose text occurs anywhere in ``PATH``
  (``MM-2.0`` inside ``MM-2.0.3``) and prepends the rest in reverse.
  ``open_core`` makes each directory an exact ``PATH`` entry, in search order.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from smc.hardware.errors import CoreError

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = [
    "INSTALL_HINT",
    "CoreError",
    "MicroManagerStatus",
    "close_core",
    "find_install",
    "open_core",
    "opened",
    "status",
]

log = logging.getLogger("smc.hardware.core")

#: What to run when the device adapters are missing. Kept in one place so every
#: error message and every doc says the same thing.
INSTALL_HINT = (
    "Micro-Manager device adapters are not installed for this Python. Run "
    "`mmcore install --test-adapters` for the demo devices (any OS), or "
    "`mmcore install` for a full nightly build (Windows / Intel macOS)."
)


@dataclass(frozen=True)
class MicroManagerStatus:
    """What this machine can drive, before any device is loaded."""

    install_dir: Path | None
    api_version: str
    core_version: str
    adapters: tuple[str, ...] = field(default_factory=tuple)

    @property
    def installed(self) -> bool:
        """Whether device adapters were found at all."""
        return self.install_dir is not None

    @property
    def has_demo_devices(self) -> bool:
        """Whether the simulator (``DemoCamera``) is available."""
        return "DemoCamera" in self.adapters


def find_install() -> Path | None:
    """Where the Micro-Manager device adapters live, if anywhere.

    pymmcore-plus looks at ``MICROMANAGER_PATH``, then its own install
    directory (``mmcore install``), then the usual application folders.
    """
    from pymmcore_plus import find_micromanager

    found = find_micromanager()
    return Path(found) if found else None


def status() -> MicroManagerStatus:
    """Report the Micro-Manager install without loading any device.

    Cheap and side-effect free, so a UI can call it at startup and a
    ``doctor`` command can print it.
    """
    from pymmcore_plus import CMMCorePlus

    install = find_install()
    core = CMMCorePlus()
    adapters: tuple[str, ...] = ()
    if install is not None:
        # A broken install reports as "no adapters" rather than crashing doctor.
        with suppress(Exception):
            adapters = tuple(sorted(str(a) for a in core.getDeviceAdapterNames()))
    return MicroManagerStatus(
        install_dir=install,
        api_version=str(core.getAPIVersionInfo()),
        core_version=str(core.getVersionInfo()),
        adapters=adapters,
    )


def _checked_dirs(paths: Sequence[str | Path]) -> list[str]:
    """Absolute form of each extra adapter directory; one that is not a directory is an error.

    MMCore accepts a missing directory silently, so this is the only place a
    typo in a profile's ``adapter_search_paths`` is caught for a profile built
    in code (a profile file was already checked when it was loaded).
    """
    if isinstance(paths, (str, os.PathLike)):
        # A str is a Sequence[str]: each character would be read as a directory.
        raise TypeError(
            "adapter_search_paths takes a sequence of directories, not a single "
            f"path ({paths!r}); wrap it in a list"
        )
    checked: list[str] = []
    for entry in paths:
        directory = Path(entry).expanduser().absolute()
        if not directory.is_dir():
            raise CoreError(
                f"adapter search path is not a directory: {entry} "
                "(set in [micromanager] adapter_search_paths of the profile)"
            )
        checked.append(str(directory))
    return checked


def _same_dir_key(entry: str) -> str:
    return os.path.normcase(os.path.abspath(entry))


def _merge_search_paths(extras: Sequence[str], current: Sequence[object]) -> list[str]:
    """Extras first, then what the core already searches; a repeat keeps its first place."""
    merged: list[str] = []
    seen: set[str] = set()
    for entry in [*extras, *(str(p) for p in current)]:
        key = _same_dir_key(entry)
        if key not in seen:
            seen.add(key)
            merged.append(entry)
    return merged


def _put_on_path(directories: Sequence[str]) -> None:
    """Make ``directories`` the first entries of ``PATH``, exactly and in order.

    A Windows adapter finds its vendor DLLs through ``PATH``. pymmcore-plus
    adds the search paths there, but by substring (so ``MM-2.0`` is skipped
    when ``MM-2.0.3`` is on ``PATH``) and one by one (so two directories end up
    in reverse order, and the later one would win a DLL that both hold).
    """
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    wanted = {os.path.normcase(d) for d in directories}
    rest = [e for e in entries if os.path.normcase(e) not in wanted]
    updated = os.pathsep.join([*directories, *rest])
    if updated != current:
        os.environ["PATH"] = updated


def open_core(
    config: str | Path | None = None,
    *,
    adapter_search_paths: Sequence[str | Path] = (),
) -> CMMCorePlus:
    """Create a core and load a configuration into it.

    Args:
        config: A Micro-Manager ``.cfg`` file. ``None`` loads the demo
            configuration shipped with the device adapters.
        adapter_search_paths: Directories to search for device adapters
            *before* the ones pymmcore-plus already searches, in this order.
            A profile's ``[micromanager] adapter_search_paths`` arrive here
            (``Profile.adapter_search_dirs()``). They matter when the stand's
            ``.cfg`` needs an adapter that only an installed MMStudio has.
            A repeated directory keeps its first place. The directories
            are put on ``PATH`` for the whole process, as pymmcore-plus does,
            and ``close_core`` does not take them off, since another core may
            need them.

    Returns:
        A fresh ``CMMCorePlus``. Callers own it and must release the devices
        (see :func:`opened`), because a Nikon or Zeiss stand accepts exactly
        one connection at a time and a core that is never closed keeps it.

    Raises:
        CoreError: When no adapters are installed, the file does not exist,
            or an entry of ``adapter_search_paths`` is not a directory. Load
            errors from Micro-Manager itself are re-raised as ``CoreError``
            too, with the file name attached.
        TypeError: When ``adapter_search_paths`` is a single path, not a
            sequence of them.
    """
    from pymmcore_plus import CMMCorePlus

    # Checked before any core exists: MMCore would take a missing directory
    # without a word, and a failed check must not leave a core behind.
    extras = _checked_dirs(adapter_search_paths)
    core = CMMCorePlus()
    # Done on every open, with or without extras, so that the log and PATH
    # look the same everywhere. It replaces the core's list, hence the merge.
    effective = _merge_search_paths(extras, core.getDeviceAdapterSearchPaths())
    core.setDeviceAdapterSearchPaths(effective)
    _put_on_path(effective)
    log.info(
        "Micro-Manager adapter search paths, first wins: %s",
        "; ".join(effective) or "(none)",
    )
    if config is None:
        if find_install() is None:
            raise CoreError(INSTALL_HINT)
        try:
            core.loadSystemConfiguration()
        except Exception as exc:
            raise CoreError(f"could not load the demo configuration: {exc}") from exc
        return core

    path = Path(config)
    if not path.exists():
        raise CoreError(f"configuration not found: {path}")
    try:
        core.loadSystemConfiguration(str(path))
    except Exception as exc:
        raise CoreError(f"could not load {path}: {exc}") from exc
    return core


def close_core(core: CMMCorePlus) -> None:
    """Release every device. Safe to call twice."""
    # Nothing useful can be done if unloading fails; the process is ending.
    with suppress(Exception):
        core.unloadAllDevices()


@contextmanager
def opened(
    config: str | Path | None = None,
    *,
    adapter_search_paths: Sequence[str | Path] = (),
) -> Iterator[CMMCorePlus]:
    """Context manager around :func:`open_core` that always releases the core."""
    core = open_core(config, adapter_search_paths=adapter_search_paths)
    try:
        yield core
    finally:
        close_core(core)
