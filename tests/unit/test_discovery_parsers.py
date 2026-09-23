import json

import pytest

from smc.discovery import os_inventory
from smc.discovery.os_inventory import (
    parse_lspci,
    parse_lsusb,
    parse_pnp_json,
    parse_system_profiler,
)

# What `Get-PnpDevice -PresentOnly | Select-Object ... | ConvertTo-Json -Compress`
# prints, abridged: an FTDI serial bridge, a Photometrics PCIe card, a device
# whose driver is missing (no FriendlyName, status Error).
PNP_JSON = json.dumps(
    [
        {
            "Status": "OK",
            "Class": "Ports",
            "FriendlyName": "USB Serial Port (COM3)",
            "InstanceId": "FTDIBUS\\VID_0403+PID_6001+A12345BA\\0000",
            "Manufacturer": "FTDI",
        },
        {
            "Status": "OK",
            "Class": "Camera",
            "FriendlyName": "Photometrics PCIe Interface",
            "InstanceId": "PCI\\VEN_1B6B&DEV_0001&SUBSYS_00011B6B&REV_01\\4&2A1&0&00E0",
            "Manufacturer": "Teledyne Photometrics",
        },
        {
            "Status": "Error",
            "Class": None,
            "FriendlyName": None,
            "InstanceId": "USB\\VID_04B0&PID_0A01\\5&1C2&0&3",
            "Manufacturer": None,
        },
    ]
)


def test_pnp_json_is_parsed_into_entries() -> None:
    entries = parse_pnp_json(PNP_JSON)

    assert [e.vendor_id for e in entries] == ["0403", "1B6B", "04B0"]
    assert [e.product_id for e in entries] == ["6001", "0001", "0A01"]
    ftdi, card, broken = entries
    assert ftdi.name == "USB Serial Port (COM3)"
    assert ftdi.device_class == "Ports"
    assert ftdi.source == "pnp"
    assert card.manufacturer == "Teledyne Photometrics"
    assert card.instance_id is not None
    assert card.instance_id.startswith("PCI\\")
    # A device without a friendly name is still reported, under its ID.
    assert broken.status == "Error"
    assert broken.name == broken.instance_id


def test_pnp_single_object_is_a_list_of_one() -> None:
    # ConvertTo-Json prints an object, not a one-element list, for one device.
    one = json.dumps(json.loads(PNP_JSON)[0])

    entries = parse_pnp_json(one)

    assert len(entries) == 1
    assert entries[0].vendor_id == "0403"


def test_pnp_empty_output_is_no_devices() -> None:
    assert parse_pnp_json("") == []


LSUSB = """\
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 004: ID 0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC
Bus 001 Device 003: ID 0661:0012
not a device line
"""


def test_lsusb_text_is_parsed() -> None:
    entries = parse_lsusb(LSUSB)

    assert [(e.vendor_id, e.product_id) for e in entries] == [
        ("1D6B", "0003"),
        ("0403", "6001"),
        ("0661", "0012"),
    ]
    assert entries[1].name.startswith("Future Technology Devices")
    assert entries[2].name == "0661:0012"
    assert {e.source for e in entries} == {"lsusb"}


SYSTEM_PROFILER = json.dumps(
    {
        "SPUSBDataType": [
            {
                "_name": "USB31Bus",
                "host_controller": "AppleT8112USBXHCI",
                "_items": [
                    {
                        "_name": "USB2.0 Hub",
                        "vendor_id": "0x05e3  (Genesys Logic, Inc.)",
                        "product_id": "0x0610",
                        "_items": [
                            {
                                "_name": "FT232R USB UART",
                                "vendor_id": "0x0403  (Future Technology Devices International Limited)",
                                "product_id": "0x6001",
                                "manufacturer": "FTDI",
                                "serial_num": "A12345BA",
                            }
                        ],
                    }
                ],
            }
        ]
    }
)


def test_system_profiler_json_is_parsed() -> None:
    entries = parse_system_profiler(SYSTEM_PROFILER)

    # Nested devices are found; the bus itself (no vendor ID) is not a device.
    assert [e.name for e in entries] == ["USB2.0 Hub", "FT232R USB UART"]
    ftdi = entries[1]
    assert (ftdi.vendor_id, ftdi.product_id) == ("0403", "6001")
    assert ftdi.manufacturer == "FTDI"
    assert ftdi.serial_number == "A12345BA"


LSPCI = """\
00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:a780] (rev 04)
03:00.0 Signal processing controller [1180]: National Instruments PCIe-6738 [1093:c4c4]
"""


def test_lspci_text_is_parsed() -> None:
    entries = parse_lspci(LSPCI)

    assert [(e.vendor_id, e.product_id) for e in entries] == [
        ("8086", "A780"),
        ("1093", "C4C4"),
    ]
    assert entries[1].name == "National Instruments PCIe-6738"
    assert entries[1].device_class == "Signal processing controller"


def test_unsupported_os_yields_empty_list_and_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os_inventory.platform, "system", lambda: "Plan9")
    notes: list[str] = []

    assert os_inventory.usb_devices(notes) == []
    assert os_inventory.pci_devices(notes) == []
    assert len(notes) == 2
    assert all("Plan9" in n for n in notes)


def test_missing_tool_yields_a_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setattr(os_inventory.platform, "system", lambda: "Linux")
    monkeypatch.setenv("PATH", str(tmp_path))
    notes: list[str] = []

    assert os_inventory.usb_devices(notes) == []
    assert os_inventory.pci_devices(notes) == []
    assert any("lsusb" in n for n in notes)
    assert any("lspci" in n for n in notes)


def test_os_sections_never_raise() -> None:
    # Whatever this machine is, the OS sections come back as lists.
    notes: list[str] = []
    serial, usb, pci = os_inventory.os_sections(notes)

    assert isinstance(serial, list)
    assert isinstance(usb, list)
    assert isinstance(pci, list)
