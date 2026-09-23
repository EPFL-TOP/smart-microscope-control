"""``smc discover`` on the demo install, and its failure modes.

The adapter work runs in a child process; these tests prove that a child
that crashes or hangs costs the device lists and nothing else.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from smc.cli import app
from smc.discovery import inventory, mm_inventory, os_inventory
from smc.discovery.mm_inventory import list_adapters
from smc.discovery.models import AdapterInfo, Inventory, SerialPort
from smc.discovery.report import JSON_NAME, TEXT_NAME
from smc.discovery.vendors import LAB_ADAPTERS

runner = CliRunner()

FTDI_PORT = SerialPort(device="COM3", vendor_id="0403", product_id="6001")


@pytest.fixture
def one_ftdi_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the OS sections by one FTDI serial port, the same on every OS."""

    def fake(notes: list[str]) -> tuple[list[SerialPort], list[object], list[object]]:
        return [FTDI_PORT.model_copy(deep=True)], [], []

    monkeypatch.setattr(os_inventory, "os_sections", fake)


@pytest.fixture
def need_mm(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


class _StubCore:
    """Two adapters installed; enumerating ``Broken`` raises like a missing DLL."""

    def getDeviceAdapterNames(self) -> tuple[str, ...]:  # noqa: N802
        return ("Good", "Broken")

    def getAvailableDevices(self, name: str) -> tuple[str, ...]:  # noqa: N802
        if name == "Broken":
            raise RuntimeError('Failed to load device adapter "Broken"')
        return ("Cam",)

    def getAvailableDeviceTypes(self, name: str) -> tuple[int, ...]:  # noqa: N802
        return (2,)

    def getAvailableDeviceDescriptions(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return ("A camera",)


def test_adapter_listing_records_enumeration_errors() -> None:
    good, broken, absent = list_adapters(_StubCore(), ["Good", "Broken", "Absent"])  # type: ignore[arg-type]

    assert good.error is None
    assert [(d.name, d.type) for d in good.devices] == [("Cam", "CameraDevice")]
    assert broken.installed
    assert broken.error is not None
    assert "Broken" in broken.error
    assert absent == AdapterInfo(name="Absent", installed=False)


def _listed(inv: Inventory) -> dict[str, AdapterInfo]:
    return {a.name: a for a in inv.adapters.listed}


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm", "one_ftdi_port")
def test_default_listing_covers_lab_and_hinted_adapters_only() -> None:
    inv = inventory()

    listed = _listed(inv)
    # LAB_ADAPTERS plus what the FTDI hint suggests (Prior is not a lab adapter).
    assert set(LAB_ADAPTERS) <= set(listed)
    assert "Prior" in listed
    assert listed["DemoCamera"].installed
    assert "DCam" in {d.name for d in listed["DemoCamera"].devices}
    # Absent adapters are reported as not installed, never as an error.
    assert not listed["NikonTi2"].installed
    assert listed["NikonTi2"].error is None
    # Installed but neither lab nor hinted: named, not enumerated.
    assert "SequenceTester" in inv.adapters.installed
    assert "SequenceTester" not in listed
    assert inv.probe is None


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_all_adapters_lists_every_installed_adapter() -> None:
    inv = inventory(all_adapters=True, include_os=False)

    listed = _listed(inv)
    assert set(inv.adapters.installed) <= set(listed)
    assert all(listed[name].installed for name in inv.adapters.installed)
    assert listed["DemoCamera"].devices
    assert not listed["NikonTi2"].installed


@pytest.mark.usefixtures("need_mm", "one_ftdi_port")
def test_mm_child_crash_becomes_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", "import os; os._exit(3)"]
    )

    inv = inventory()

    assert inv.adapters.listed == []
    assert inv.adapters.installed  # the names are kept
    assert any("exited 3" in n for n in inv.notes)
    assert [p.device for p in inv.serial] == ["COM3"]
    assert inv.hints


@pytest.mark.usefixtures("need_mm", "one_ftdi_port")
def test_mm_child_timeout_becomes_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mm_inventory,
        "CHILD_COMMAND",
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    inv = inventory(timeout_s=1)

    assert inv.adapters.listed == []
    assert any("did not finish in 1 s" in n for n in inv.notes)
    assert [p.device for p in inv.serial] == ["COM3"]


@pytest.mark.usefixtures("need_mm")
def test_mm_child_garbage_output_becomes_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", "print('{not json')"]
    )

    inv = inventory(include_os=False)

    assert inv.adapters.listed == []
    assert any("could not be read" in n for n in inv.notes)


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_probe_democamera_loads_hub_first() -> None:
    inv = inventory(probe_adapter="DemoCamera", include_os=False)

    assert inv.probe is not None
    assert inv.probe.error is None
    devices = inv.probe.devices
    assert devices[0].type == "HubDevice"
    assert devices[0].name == "DHub"
    assert len(devices) > 10
    failed = [(d.name, d.error) for d in devices if not d.ok]
    assert failed == []


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_discover_out_writes_utf8_json_and_text(tmp_path: object) -> None:
    from pathlib import Path

    out = Path(str(tmp_path)) / "surveys" / "nikon-ti2"

    result = runner.invoke(app, ["discover", "--out", str(out), "--no-os"])

    assert result.exit_code == 0, result.output
    inv = Inventory.model_validate_json((out / JSON_NAME).read_text(encoding="utf-8"))
    assert "DemoCamera" in _listed(inv)
    text = (out / TEXT_NAME).read_bytes().decode("utf-8")
    assert "Adapter devices" in text
    assert "DCam" in text


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_discover_runs_here_without_probe() -> None:
    result = runner.invoke(app, ["discover"])

    assert result.exit_code == 0, result.output
    assert "Installed adapters" in result.output
    assert "DemoCamera" in result.output
    assert "Probe of" not in result.output
