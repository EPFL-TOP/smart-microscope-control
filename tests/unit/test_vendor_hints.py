from datetime import datetime, timezone

from smc.discovery.models import DeviceEntry, Inventory, SerialPort, SystemInfo
from smc.discovery.vendors import LAB_ADAPTERS, VENDOR_HINTS, apply_hints


def _inventory(**sections: object) -> Inventory:
    system = SystemInfo(
        hostname="test",
        collected_at=datetime.now(timezone.utc),
        os="test",
        python="3",
        smc="0",
        pymmcore_plus="0",
    )
    return Inventory(system=system, **sections)  # type: ignore[arg-type]


def test_ftdi_vid_gets_serial_bridge_hint() -> None:
    port = SerialPort(device="COM3", vendor_id="0403", product_id="6001")
    inv = _inventory(serial=[port])

    adapters = apply_hints(inv)

    assert "Marzhauser" in adapters
    assert "ASITiger" in adapters
    assert inv.serial[0].hints == ["FTDI"]
    (hint,) = inv.hints
    assert hint.vendor == "FTDI"
    assert "COM3" in hint.matched


def test_photometrics_pci_maps_to_pvcam() -> None:
    card = DeviceEntry(
        source="pnp",
        name="Photometrics PCIe card",
        vendor_id="1B6B",
        product_id="0001",
        instance_id="PCI\\VEN_1B6B&DEV_0001&SUBSYS_00000000",
    )
    inv = _inventory(pci=[card])

    assert apply_hints(inv) == {"PVCAM"}
    assert inv.pci[0].hints == ["Photometrics"]


def test_ni_pci_suggests_nidaq() -> None:
    card = DeviceEntry(
        source="lspci", name="National Instruments PCIe-6738", vendor_id="1093"
    )
    inv = _inventory(pci=[card])

    assert "NIDAQ" in apply_hints(inv)
    assert inv.hints[0].vendor == "National Instruments"


def test_pci_vendor_id_does_not_match_a_usb_hint() -> None:
    # USB and PCI vendor IDs are separate namespaces: FTDI's USB VID on a PCI
    # card says nothing about FTDI.
    card = DeviceEntry(source="lspci", name="Some card", vendor_id="0403")
    inv = _inventory(pci=[card])

    assert apply_hints(inv) == set()
    assert inv.hints == []


def test_xilinx_needs_the_zeiss_name_to_be_a_zeiss_card() -> None:
    plain = DeviceEntry(source="lspci", name="Xilinx FPGA", vendor_id="10EE")
    zeiss = DeviceEntry(source="pnp", name="CZMI realtime interface", vendor_id="10EE")
    inv = _inventory(pci=[plain, zeiss])

    assert apply_hints(inv) == set()
    assert [h.vendor for h in inv.hints] == ["Zeiss"]
    assert inv.pci[0].hints == []
    assert inv.pci[1].hints == ["Zeiss"]


def test_licence_dongle_is_hinted_without_an_adapter() -> None:
    dongle = DeviceEntry(source="pnp", name="Sentinel HASP Key", vendor_id="0529")
    inv = _inventory(usb=[dongle])

    assert apply_hints(inv) == set()
    assert inv.hints[0].adapters == []
    assert "licence" in inv.hints[0].note


def test_hint_ids_are_bus_qualified_upper_case_hex() -> None:
    for hint in VENDOR_HINTS:
        assert hint.ids or hint.name_fragments
        for vendor_id in hint.ids:
            bus, _, hex_id = vendor_id.partition(":")
            assert bus in {"usb", "pci"}
            assert len(hex_id) == 4
            assert hex_id == hex_id.upper()
    assert len(set(LAB_ADAPTERS)) == len(LAB_ADAPTERS)
