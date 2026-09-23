"""Read-only survey of a microscope PC (``smc discover``).

Run on each microscope computer following
``docs/hardware/microscope-pc-setup.md``: what the OS sees (serial, USB/PnP,
PCI), which vendors that suggests, and what the installed Micro-Manager
adapters offer. Nothing is initialised unless one adapter is probed on
purpose, and nothing is written unless asked (:func:`smc.discovery.report.write`).
"""

from __future__ import annotations

import logging

from smc.discovery import mm_inventory, os_inventory
from smc.discovery.models import AdaptersSection, Inventory
from smc.discovery.vendors import LAB_ADAPTERS, apply_hints

__all__ = ["Inventory", "inventory"]

logger = logging.getLogger("smc.discovery")


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
    logger.info(
        "inventory: %d adapters installed, %d listed, %d notes",
        len(installed),
        len(listed),
        len(inv.notes),
    )
    return inv
