"""Vendor IDs and names seen on a PC → vendor → Micro-Manager adapters to list.

This is data, not code paths: recognising a new vendor is one entry in
:data:`VENDOR_HINTS`. The vendor IDs are best knowledge, not measurements; a
wrong one costs a hint, never a survey, and the first real surveys correct
the table.

IDs are bus-qualified (``usb:0403``, ``pci:1B6B``) because USB and PCI vendor
IDs are separate namespaces: the same four hex digits mean different
companies on the two buses.
"""

from __future__ import annotations

from dataclasses import dataclass

from smc.discovery.models import DeviceEntry, Hint, Inventory, SerialPort

__all__ = ["LAB_ADAPTERS", "VENDOR_HINTS", "VendorHint", "apply_hints"]


@dataclass(frozen=True, slots=True)
class VendorHint:
    """One vendor, how to recognise it, and what it suggests.

    An entry matches when its bus-qualified vendor ID matches (if ``ids`` is
    set) *and* one of the name fragments occurs in the entry's names (if
    ``name_fragments`` is set). Both together pin down a vendor whose chip is
    generic: a Xilinx FPGA is only a Zeiss realtime card when it says so.
    """

    ids: tuple[str, ...]
    name_fragments: tuple[str, ...]
    vendor: str
    adapters: tuple[str, ...]
    note: str = ""


_SERIAL_BRIDGE_ADAPTERS = (
    "Marzhauser",
    "MarzhauserLStep",
    "ASIStage",
    "ASITiger",
    "SutterLambda",
    "Prior",
    "CoherentOBIS",
    "Cobolt",
    "Omicron",
)
_SERIAL_BRIDGE_NOTE = (
    "generic USB-serial bridge: Märzhäuser, ASI, Sutter, Prior stages or a "
    "laser controller; the label on the device tells which"
)

VENDOR_HINTS: tuple[VendorHint, ...] = (
    VendorHint(("usb:0403",), (), "FTDI", _SERIAL_BRIDGE_ADAPTERS, _SERIAL_BRIDGE_NOTE),
    VendorHint(
        ("usb:10C4",),
        (),
        "Silicon Labs",
        _SERIAL_BRIDGE_ADAPTERS,
        _SERIAL_BRIDGE_NOTE,
    ),
    VendorHint(
        ("usb:067B",), (), "Prolific", _SERIAL_BRIDGE_ADAPTERS, _SERIAL_BRIDGE_NOTE
    ),
    VendorHint(("usb:0661",), (), "Hamamatsu", ("HamamatsuHam",)),
    VendorHint(("usb:04B0",), (), "Nikon", ("NikonTi2", "NikonTI")),
    VendorHint(("usb:3923",), (), "National Instruments", ("NIDAQ",)),
    VendorHint(
        ("usb:2341", "usb:2A03"),
        (),
        "Arduino",
        ("Arduino", "TriggerScopeMM"),
        "an Arduino, or a TriggerScope built on one",
    ),
    VendorHint(("usb:1313",), (), "Thorlabs", ()),
    VendorHint(("usb:1A72",), (), "Physik Instrumente", ("PI_GCS_2",)),
    VendorHint(
        ("usb:0529",),
        (),
        "Sentinel",
        (),
        "licence dongle of a vendor software; no adapter",
    ),
    VendorHint(("pci:1B6B",), (), "Photometrics", ("PVCAM",)),
    VendorHint(
        ("pci:1093",),
        (),
        "National Instruments",
        ("NIDAQ",),
        "a PCIe DAQ is a candidate light-sheet timing controller",
    ),
    VendorHint(
        ("pci:10EE",),
        ("CZMI", "MicoIf"),
        "Zeiss",
        (),
        "Zeiss realtime card (Xilinx FPGA); reached through MTB, no adapter",
    ),
)

#: Adapters this lab is likely to use; their devices are listed by default.
#: Names are checked against the install at run time: an absent one is
#: reported as not installed, never as an error.
LAB_ADAPTERS: tuple[str, ...] = (
    "NikonTi2",
    "NikonTI",
    "ZeissCAN29",
    "HamamatsuHam",
    "PVCAM",
    "AndorSDK3",
    "Marzhauser",
    "MarzhauserLStep",
    "ASIStage",
    "ASITiger",
    "PI_GCS_2",
    "NIDAQ",
    "Arduino",
    "TriggerScopeMM",
    "CoherentOBIS",
    "Cobolt",
    "Omicron",
    "SutterLambda",
    "ThorlabsFilterWheel",
    "DemoCamera",
)


def _matches(
    hint: VendorHint, bus: str, vendor_id: str | None, names: tuple[str | None, ...]
) -> bool:
    if hint.ids and (vendor_id is None or f"{bus}:{vendor_id.upper()}" not in hint.ids):
        return False
    if hint.name_fragments:
        text = " ".join(n for n in names if n).lower()
        return any(f.lower() in text for f in hint.name_fragments)
    return bool(hint.ids)


def apply_hints(inventory: Inventory) -> set[str]:
    """Annotate the serial, USB and PCI entries with the vendors they match.

    Appends one :class:`~smc.discovery.models.Hint` per matched entry to
    ``inventory.hints`` and sets each entry's ``hints``.

    Returns:
        The adapters the hints suggest, to be listed besides ``LAB_ADAPTERS``.
    """
    adapters: set[str] = set()
    targets: list[tuple[SerialPort | DeviceEntry, str, str, tuple[str | None, ...]]]
    targets = [
        (
            p,
            "usb",
            f"serial {p.device}",
            (p.manufacturer, p.description),
        )
        for p in inventory.serial
    ]
    for section, entries in (("usb", inventory.usb), ("pci", inventory.pci)):
        targets += [
            (
                e,
                section,
                f"{section} {e.name}",
                (e.name, e.manufacturer, e.instance_id),
            )
            for e in entries
        ]
    for entry, bus, label, names in targets:
        for hint in VENDOR_HINTS:
            if not _matches(hint, bus, entry.vendor_id, names):
                continue
            if hint.vendor not in entry.hints:
                entry.hints.append(hint.vendor)
            inventory.hints.append(
                Hint(
                    vendor=hint.vendor,
                    matched=label,
                    adapters=list(hint.adapters),
                    note=hint.note,
                )
            )
            adapters.update(hint.adapters)
    return adapters
