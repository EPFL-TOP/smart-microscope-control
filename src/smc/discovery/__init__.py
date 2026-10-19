"""Read-only survey of a microscope PC (``smc discover``).

Run on each microscope computer following
``docs/hardware/microscope-pc-setup.md``: what the OS sees (serial, USB/PnP,
PCI), which vendors that suggests, and what the installed Micro-Manager
adapters offer. Nothing is initialised unless one adapter is probed on
purpose, and nothing is written unless asked (:func:`smc.discovery.report.write`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from smc.discovery import dll_probe, mm_inventory, os_inventory
from smc.discovery.models import AdaptersSection, Inventory
from smc.discovery.vendors import LAB_ADAPTERS, apply_hints

__all__ = ["MAX_DLL_PROBES", "Inventory", "inventory"]

logger = logging.getLogger("smc.discovery")

#: Each loader probe may take ``dll_probe.DLL_PROBE_TIMEOUT_S``: a PC with a
#: dozen vendor adapters that will not load must not turn a survey into minutes
#: of probing (FM-74). The rest are named in one note.
MAX_DLL_PROBES = 10

_LOAD_FAILURE = "Failed to load device adapter"


def _probe_failed_loads(inv: Inventory, install: Path) -> None:
    """Ask Windows why each adapter that failed to load will not (FM-09).

    MMCore's message is the same for a missing dependency, a wrong SDK, the
    wrong bitness and a device-interface mismatch; Windows' own error code,
    read by loading the adapter's DLL alone in a child, tells them apart.
    Windows only: off it a ``.dll`` cannot be loaded, and a row saying so for
    every adapter would be noise. At most ``MAX_DLL_PROBES``, counted in the
    order of the listing. A probe that fails itself is a note, never the end of
    the survey.
    """
    if not dll_probe._WINDOWS:
        return
    probed = skipped = 0
    for adapter in inv.adapters.listed:
        if adapter.error is None or _LOAD_FAILURE not in adapter.error:
            continue
        if probed >= MAX_DLL_PROBES:
            skipped += 1
            continue
        probed += 1
        try:
            result = dll_probe.probe_dll(install / f"mmgr_dal_{adapter.name}.dll")
        except Exception as exc:
            inv.notes.append(
                f"adapters: the loader probe of {adapter.name} failed: {exc}"
            )
            continue
        adapter.load_diagnosis = result.detail
    if skipped:
        plural = "adapter" if skipped == 1 else "adapters"
        inv.notes.append(
            f"adapters: {skipped} {plural} failed to load and {'was' if skipped == 1 else 'were'} "
            f"not probed for Windows' reason (at most {MAX_DLL_PROBES} are probed "
            "per survey)"
        )


def inventory(
    *,
    all_adapters: bool = False,
    probe_adapter: str | None = None,
    include_os: bool = True,
    timeout_s: float = mm_inventory.CHILD_TIMEOUT_S,
) -> Inventory:
    """Survey this PC.

    The OS sections come first so the hints can add adapters to the device
    listing; the adapter work runs last, in a child process, so a vendor DLL
    that hangs or crashes costs the device lists and nothing else.

    Args:
        all_adapters: List the devices of every installed adapter instead of
            ``LAB_ADAPTERS`` plus the hinted ones. Slow: loads every DLL.
        probe_adapter: Load and initialise every device of this one adapter.
            Contacts the hardware; the stand must be powered and its vendor
            software closed.
        include_os: Collect serial, USB/PnP and PCI.
        timeout_s: How long the adapter child may run.
    """
    system, installed = mm_inventory.system_section()
    inv = Inventory(system=system, adapters=AdaptersSection(installed=installed))

    if include_os:
        inv.serial, inv.usb, inv.pci = os_inventory.os_sections(inv.notes)
    else:
        # An empty section must not read as "no devices on this PC".
        inv.notes.append("os: serial, USB / PnP and PCI not collected (--no-os)")
    hinted = apply_hints(inv)

    if system.mm_install is None:
        inv.notes.append(
            "adapters: no Micro-Manager install found, device lists skipped — "
            "run `mmcore install`"
        )
        return inv

    names = sorted(set(LAB_ADAPTERS) | hinted)
    listed, probe, notes = mm_inventory.mm_section(
        names, all_adapters=all_adapters, probe=probe_adapter, timeout_s=timeout_s
    )
    inv.adapters.listed = listed
    inv.probe = probe
    inv.notes.extend(notes)
    _probe_failed_loads(inv, Path(system.mm_install))
    logger.info(
        "inventory: %d adapters installed, %d listed, %d notes",
        len(installed),
        len(listed),
        len(inv.notes),
    )
    return inv
