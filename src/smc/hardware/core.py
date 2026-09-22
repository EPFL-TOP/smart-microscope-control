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
"""

from __future__ import annotations

from collections.abc import Iterator
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


def open_core(config: str | Path | None = None) -> CMMCorePlus:
    """Create a core and load a configuration into it.

    Args:
        config: A Micro-Manager ``.cfg`` file. ``None`` loads the demo
            configuration shipped with the device adapters.

    Returns:
        A fresh ``CMMCorePlus``. Callers own it and must release the devices
        (see :func:`opened`), because a Nikon or Zeiss stand accepts exactly
        one connection at a time and a core that is never closed keeps it.

    Raises:
        CoreError: When no adapters are installed or the file does not exist.
            Load errors from Micro-Manager itself are re-raised as
            ``CoreError`` too, with the file name attached.
    """
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
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
def opened(config: str | Path | None = None) -> Iterator[CMMCorePlus]:
    """Context manager around :func:`open_core` that always releases the core."""
    core = open_core(config)
    try:
        yield core
    finally:
        close_core(core)
