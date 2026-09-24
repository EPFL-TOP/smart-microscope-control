"""Serial ports, USB / PnP devices and PCI cards, as the operating system sees them.

This is the half of a survey that needs no Micro-Manager: what is plugged
into the PC, with vendor IDs the hints can map to adapters. Each OS has its
own tool (PowerShell ``Get-PnpDevice`` on Windows, ``system_profiler`` on
macOS, ``lsusb`` / ``lspci`` on Linux); any of them can be missing, slow or
broken on a locked-down lab PC, so every call has a timeout and a failure
becomes a note, never an exception.

The parsers are pure functions of the tool's text so they can be tested on
hand-written output from an OS the test does not run on.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from collections.abc import Callable
from typing import Any

from smc.discovery.models import DeviceEntry, SerialPort

__all__ = [
    "OS_TIMEOUT_S",
    "os_sections",
    "parse_lspci",
    "parse_lsusb",
    "parse_pnp_json",
    "parse_system_profiler",
    "pci_devices",
    "pnp_devices",
    "serial_ports",
    "usb_devices",
]

logger = logging.getLogger("smc.discovery.os_inventory")

#: Per OS call. ``Get-PnpDevice`` takes a few seconds on a PC with many devices.
OS_TIMEOUT_S = 20.0

#: A child sharing the console can repaint it (FM-22): PowerShell's own
#: ``[Console]::OutputEncoding=`` line is exactly that. ``0`` elsewhere.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The console output encoding is set inside the command: a child's stdout is
# the ANSI code page on Windows, which mangles device names; -Compress keeps
# the output one line.
_PNP_COMMAND = [
    "powershell",
    "-NoProfile",
    "-Command",
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
    "Get-PnpDevice -PresentOnly | "
    "Select-Object Status,Class,FriendlyName,InstanceId,Manufacturer | "
    "ConvertTo-Json -Compress",
]

#: PCI instance IDs (``PCI\VEN_1B6B&DEV_0001&...``) name the fields ``VEN_``/
#: ``DEV_``; every other bus (``USB\``, ``FTDIBUS\`` with ``+`` instead of
#: ``&``) names them ``VID_``/``PID_``. An ``ACPI\`` or ``HDAUDIO\`` ID has
#: neither and yields no IDs.
_PCI_VENDOR = re.compile(r"VEN_([0-9A-Fa-f]{4})")
_PCI_DEVICE = re.compile(r"DEV_([0-9A-Fa-f]{4})")
_USB_VENDOR = re.compile(r"VID_([0-9A-Fa-f]{4})")
_USB_DEVICE = re.compile(r"PID_([0-9A-Fa-f]{4})")
_HEX_ID = re.compile(r"0x([0-9A-Fa-f]{4})")
_LSUSB = re.compile(
    r"^Bus \d+ Device \d+: ID ([0-9A-Fa-f]{4}):([0-9A-Fa-f]{4})\s*(.*)$"
)
_LSPCI = re.compile(
    r"^\S+ (?P<cls>.+?) \[[0-9A-Fa-f]{4}\]: (?P<name>.+) "
    r"\[(?P<vid>[0-9A-Fa-f]{4}):(?P<pid>[0-9A-Fa-f]{4})\]"
)


def _run(command: list[str], what: str, notes: list[str]) -> str | None:
    """Run an OS tool; its stdout, or ``None`` with the reason in ``notes``."""
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=OS_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=_NO_WINDOW,
        )
    except FileNotFoundError:
        notes.append(f"{what}: `{command[0]}` not found on this PC")
        return None
    except subprocess.TimeoutExpired:
        notes.append(f"{what}: `{command[0]}` did not answer in {OS_TIMEOUT_S:.0f} s")
        return None
    except OSError as exc:
        notes.append(f"{what}: `{command[0]}` could not run: {exc}")
        return None
    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        reason = detail[0] if detail else "no message"
        notes.append(f"{what}: `{command[0]}` exited {done.returncode}: {reason}")
        return None
    return done.stdout


def _upper(match: re.Match[str] | None) -> str | None:
    return match.group(1).upper() if match else None


def _pnp_ids(instance_id: str | None) -> tuple[str | None, str | None]:
    """The vendor/product ID pair, read with the field names of its bus."""
    if not instance_id:
        return None, None
    if instance_id.upper().startswith("PCI\\"):
        return _upper(_PCI_VENDOR.search(instance_id)), _upper(
            _PCI_DEVICE.search(instance_id)
        )
    return _upper(_USB_VENDOR.search(instance_id)), _upper(
        _USB_DEVICE.search(instance_id)
    )


def parse_pnp_json(text: str) -> list[DeviceEntry]:
    """Parse ``Get-PnpDevice | Select-Object … | ConvertTo-Json`` output.

    ``ConvertTo-Json`` prints an object instead of a list when there is one
    device, and nothing at all when there is none.
    """
    text = text.strip()
    if not text:
        return []
    data: Any = json.loads(text)
    items = data if isinstance(data, list) else [data]
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        instance_id = item.get("InstanceId") or None
        vendor_id, product_id = _pnp_ids(instance_id)
        entries.append(
            DeviceEntry(
                source="pnp",
                name=item.get("FriendlyName") or instance_id or "?",
                vendor_id=vendor_id,
                product_id=product_id,
                manufacturer=item.get("Manufacturer") or None,
                device_class=item.get("Class") or None,
                status=item.get("Status") or None,
                instance_id=instance_id,
            )
        )
    return entries


def parse_lsusb(text: str) -> list[DeviceEntry]:
    """Parse ``lsusb`` lines (``Bus 001 Device 004: ID 0403:6001 Name``)."""
    entries = []
    for line in text.splitlines():
        m = _LSUSB.match(line.strip())
        if not m:
            continue
        vid, pid, name = m.group(1).upper(), m.group(2).upper(), m.group(3).strip()
        entries.append(
            DeviceEntry(
                source="lsusb",
                name=name or f"{vid.lower()}:{pid.lower()}",
                vendor_id=vid,
                product_id=pid,
            )
        )
    return entries


def parse_system_profiler(text: str) -> list[DeviceEntry]:
    """Parse ``system_profiler SPUSBDataType SPUSBHostDataType -json``.

    Devices nest under ``_items`` in either top-level key. Buses and host
    controllers carry no vendor ID and are not devices; ``SPUSBHostDataType``
    is host controllers only unless something is plugged into a port the
    other key does not already cover, but both are cheap to walk and neither
    is trusted to be complete on its own.
    """
    data: Any = json.loads(text)
    entries: list[DeviceEntry] = []

    def walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            vid = _upper(_HEX_ID.search(str(item.get("vendor_id", ""))))
            if vid is not None:
                entries.append(
                    DeviceEntry(
                        source="system_profiler",
                        name=str(item.get("_name", "?")),
                        vendor_id=vid,
                        product_id=_upper(
                            _HEX_ID.search(str(item.get("product_id", "")))
                        ),
                        manufacturer=item.get("manufacturer") or None,
                        serial_number=item.get("serial_num") or None,
                    )
                )
            walk(item.get("_items"))

    if isinstance(data, dict):
        for key in ("SPUSBDataType", "SPUSBHostDataType"):
            walk(data.get(key))
    return entries


def parse_lspci(text: str) -> list[DeviceEntry]:
    """Parse ``lspci -nn`` lines (``03:00.0 Class [1180]: Name [1093:c4c4]``)."""
    entries = []
    for line in text.splitlines():
        m = _LSPCI.match(line.strip())
        if not m:
            continue
        entries.append(
            DeviceEntry(
                source="lspci",
                name=m.group("name").strip(),
                vendor_id=m.group("vid").upper(),
                product_id=m.group("pid").upper(),
                device_class=m.group("cls").strip(),
            )
        )
    return entries


def _parsed(
    parse: Callable[[str], list[DeviceEntry]],
    text: str | None,
    what: str,
    notes: list[str],
) -> list[DeviceEntry]:
    if text is None:
        return []
    try:
        return parse(text)
    except (ValueError, AttributeError) as exc:
        notes.append(f"{what}: output could not be parsed: {exc}")
        return []


def serial_ports(notes: list[str]) -> list[SerialPort]:
    """Every serial port with its USB IDs, through ``pyserial``."""
    try:
        from serial.tools import list_ports

        ports = list_ports.comports()
    except Exception as exc:  # a driver can make enumeration itself fail
        notes.append(f"serial: ports could not be listed: {exc}")
        return []
    return [
        SerialPort(
            device=p.device,
            vendor_id=f"{p.vid:04X}" if p.vid is not None else None,
            product_id=f"{p.pid:04X}" if p.pid is not None else None,
            manufacturer=p.manufacturer or None,
            description=p.description or None,
            serial_number=p.serial_number or None,
        )
        for p in sorted(ports, key=lambda p: p.device)
    ]


def pnp_devices(notes: list[str]) -> list[DeviceEntry]:
    """Every present PnP device (Windows only)."""
    if platform.system() != "Windows":
        notes.append(
            f"pnp: Get-PnpDevice exists on Windows only, not {platform.system()}"
        )
        return []
    text = _run(_PNP_COMMAND, "pnp", notes)
    return _parsed(parse_pnp_json, text, "pnp", notes)


def _is_pci(entry: DeviceEntry) -> bool:
    return (entry.instance_id or "").upper().startswith("PCI\\")


def usb_devices(
    notes: list[str], pnp: list[DeviceEntry] | None = None
) -> list[DeviceEntry]:
    """USB devices; on Windows every PnP entry that is not a PCI card.

    Args:
        notes: Receives one line per thing that could not be collected.
        pnp: Windows PnP entries already collected, so PowerShell runs once.
    """
    system = platform.system()
    if system == "Windows":
        entries = pnp if pnp is not None else pnp_devices(notes)
        return [e for e in entries if not _is_pci(e)]
    if system == "Darwin":
        text = _run(
            ["system_profiler", "SPUSBDataType", "SPUSBHostDataType", "-json"],
            "usb",
            notes,
        )
        return _parsed(parse_system_profiler, text, "usb", notes)
    if system == "Linux":
        return _parsed(parse_lsusb, _run(["lsusb"], "usb", notes), "usb", notes)
    notes.append(f"usb: no USB listing for {system}")
    return []


def pci_devices(
    notes: list[str], pnp: list[DeviceEntry] | None = None
) -> list[DeviceEntry]:
    r"""PCI cards; on Windows the PnP entries whose instance ID starts ``PCI\``."""
    system = platform.system()
    if system == "Windows":
        entries = pnp if pnp is not None else pnp_devices(notes)
        return [e for e in entries if _is_pci(e)]
    if system == "Linux":
        return _parsed(parse_lspci, _run(["lspci", "-nn"], "pci", notes), "pci", notes)
    notes.append(f"pci: no PCI listing for {system}")
    return []


def os_sections(
    notes: list[str],
) -> tuple[list[SerialPort], list[DeviceEntry], list[DeviceEntry]]:
    """Serial, USB and PCI sections, running PowerShell once on Windows."""
    pnp = pnp_devices(notes) if platform.system() == "Windows" else None
    serial = serial_ports(notes)
    usb = usb_devices(notes, pnp)
    pci = pci_devices(notes, pnp)
    logger.info(
        "os inventory: %d serial, %d usb, %d pci", len(serial), len(usb), len(pci)
    )
    return serial, usb, pci
