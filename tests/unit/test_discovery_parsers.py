import json
import subprocess
import sys
from pathlib import Path

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


def test_pnp_ids_are_read_by_bus() -> None:
    # PCI names its fields VEN_/DEV_; USB and FTDIBUS (+ instead of &) name
    # them VID_/PID_; ACPI and HDAUDIO have neither and yield no IDs.
    entries = parse_pnp_json(
        json.dumps(
            [
                {
                    "Status": "OK",
                    "Class": "System",
                    "FriendlyName": "System board",
                    "InstanceId": "ACPI\\VEN_INT&DEV_33A0\\0",
                },
                {
                    "Status": "OK",
                    "Class": "MEDIA",
                    "FriendlyName": "High Definition Audio Controller",
                    "InstanceId": "HDAUDIO\\FUNC_01&VEN_10EC&DEV_0000\\4&2A1&0&0001",
                },
            ]
        )
    )

    assert [(e.vendor_id, e.product_id) for e in entries] == [
        (None, None),
        (None, None),
    ]


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


def test_macos_usb_reads_both_data_types() -> None:
    # `system_profiler SPUSBDataType SPUSBHostDataType -json`: one call, two
    # top-level keys. SPUSBHostDataType is the real output of this command on
    # an Apple Silicon Mac (tests/data/system_profiler_usbhost.json, serial
    # numbers and location IDs stripped): host controllers only, no vendor ID,
    # so they must not be read as devices.
    host = json.loads(
        (Path(__file__).parents[1] / "data" / "system_profiler_usbhost.json").read_text(
            encoding="utf-8"
        )
    )
    combined = {**json.loads(SYSTEM_PROFILER), **host}

    entries = parse_system_profiler(json.dumps(combined))

    assert [e.name for e in entries] == ["USB2.0 Hub", "FT232R USB UART"]


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


def test_os_tools_run_without_a_console_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-22: a child sharing the console can repaint it (PowerShell's own
    # `[Console]::OutputEncoding=` line is exactly that). The Windows CI job
    # is the real check: creationflags is `CREATE_NO_WINDOW` there, `0` here.
    calls: list[dict[str, object]] = []
    real_run = os_inventory.subprocess.run

    def spy(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return real_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os_inventory.subprocess, "run", spy)
    notes: list[str] = []

    os_inventory._run([sys.executable, "-c", "pass"], "test", notes)

    assert calls
    assert calls[0]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
