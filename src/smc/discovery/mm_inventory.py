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


def probe_adapter(name: str) -> ProbeResult:
    """Load and initialise every device of one adapter in a throwaway core.

    The ``Hub`` device is loaded first and becomes every other device's
    parent, because a hub-based adapter's peripherals only initialise below
    their hub. Each device's outcome is recorded; everything is unloaded
    afterwards so the stand is released.
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


def _last_json_line(stdout: str) -> str | None:
    # A vendor DLL may print to stdout while it loads; the result is the last
    # line the child writes.
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            return line
    return None


def mm_section(
    adapters: list[str],
    *,
    all_adapters: bool = False,
    probe: str | None = None,
    timeout_s: float = CHILD_TIMEOUT_S,
) -> tuple[list[AdapterInfo], ProbeResult | None, list[str]]:
    """Run the adapter work in a child process and collect what it found.

    Returns:
        The listed adapters, the probe (``None`` unless asked), and notes. A
        non-zero exit, a timeout or unparsable output returns no listing and
        a note that says which of the three happened.
    """
    command = [*CHILD_COMMAND, "--adapters", ",".join(adapters)]
    if all_adapters:
        command.append("--all")
    if probe:
        command += ["--probe", probe]
    logger.info("adapter child: %s", " ".join(command[1:]))
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            [],
            None,
            [
                f"adapters: the adapter child did not finish in {timeout_s:.0f} s "
                "and was stopped (an adapter hung while loading); device lists "
                "missing, adapter names kept"
            ],
        )
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-3:]
        detail = f": {' | '.join(tail)}" if tail else ""
        return (
            [],
            None,
            [
                f"adapters: the adapter child exited {done.returncode} (an "
                f"adapter crashed while loading?){detail}; device lists "
                "missing, adapter names kept"
            ],
        )
    line = _last_json_line(done.stdout or "")
    try:
        if line is None:
            raise ValueError("no JSON line on stdout")
        result = ChildResult.model_validate_json(line)
    except (ValueError, ValidationError) as exc:
        return (
            [],
            None,
            [f"adapters: the adapter child's output could not be read: {exc}"],
        )
    return result.listed, result.probe, []
