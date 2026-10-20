"""The probe on Micro-Manager's own test adapters (design m2 §9.3, measured 2026-10-20).

``SequenceTester`` is a hub that lists one device and reports 14 after it
initialises: the case that said ``THub ok`` and stopped. ``Utilities`` has a
serial shutter whose pre-init ``Port`` reads empty. Only the test adapters are
named here, never a vendor's (FM-40).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import smc.discovery as discovery
from smc.cli import app
from smc.discovery.mm_inventory import probe_adapter
from smc.discovery.models import Inventory, ProbedDevice

pytestmark = pytest.mark.demo

SERIAL_SHUTTER = "Serial port DTR Shutter"


@pytest.fixture(autouse=True)
def need_mm(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


@pytest.fixture(autouse=True)
def only_test_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lab list is the demo's: listing a vendor adapter contacts a stand (FM-06)."""
    monkeypatch.setattr(discovery, "LAB_ADAPTERS", ("DemoCamera",))


def _by_name(rows: list[ProbedDevice]) -> dict[str, ProbedDevice]:
    return {d.name: d for d in rows}


def test_probe_sequence_tester_loads_the_hubs_peripherals() -> None:
    result = probe_adapter("SequenceTester")

    assert result.error is None
    assert result.complete
    first = result.devices[0]
    assert (first.name, first.type, first.status, first.source) == (
        "THub",
        "Hub",
        "ok",
        "adapter",
    )
    peripherals = result.devices[1:]
    # Measured: 14. The count belongs to the adapter, not to this test.
    assert len(peripherals) >= 2
    assert {d.source for d in peripherals} == {"hub"}
    assert [d.name for d in peripherals if d.status != "ok"] == []
    types = {d.type for d in peripherals}
    assert {"Camera", "Shutter", "XYStage", "Stage", "AutoFocus", "State"} <= types


def test_probe_utilities_reports_the_serial_shutter_as_needing_a_port() -> None:
    without = _by_name(probe_adapter("Utilities").devices)

    shutter = without[SERIAL_SHUTTER]
    assert shutter.status == "not-probed"
    assert shutter.reason is not None
    assert "needs a port" in shutter.reason
    assert shutter.error is None

    with_port = _by_name(probe_adapter("Utilities", port="COMX").devices)

    tried = with_port[SERIAL_SHUTTER]
    # Tried, with the port on the row. Whether the demo's shutter then
    # initialises on a port that does not exist is the adapter's, and was only
    # measured off Windows: the walk's claim is that it was given the port.
    assert tried.status in ("ok", "failed")
    assert tried.port == "COMX"
    # No other Utilities row is asserted: its `DA ...` devices fail on their own.


def test_probe_through_the_child_gives_the_same_rows_and_is_complete() -> None:
    inv = discovery.inventory(probe_adapter="SequenceTester", include_os=False)

    assert inv.probe is not None
    assert inv.probe.complete
    direct = probe_adapter("SequenceTester")
    assert [(d.name, d.status, d.source) for d in inv.probe.devices] == [
        (d.name, d.status, d.source) for d in direct.devices
    ]
    back = Inventory.model_validate_json(inv.model_dump_json())
    assert back.probe == inv.probe
    # The old key is gone from the file: a reader cannot take a row for a pass.
    rows = json.loads(inv.model_dump_json())["probe"]["devices"]
    assert all("ok" not in row and "status" in row for row in rows)


def test_probe_through_the_child_gives_the_port_to_the_device_that_needs_it() -> None:
    inv = discovery.inventory(
        probe_adapter="Utilities", probe_port="COMX", include_os=False
    )

    assert inv.probe is not None
    shutter = _by_name(inv.probe.devices)[SERIAL_SHUTTER]
    assert shutter.port == "COMX"


def test_discover_cli_prints_the_probe_columns() -> None:
    runner = CliRunner(env={"COLUMNS": "200"})

    result = runner.invoke(
        app, ["discover", "--probe-adapter", "SequenceTester", "--no-os"]
    )

    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    for fragment in ("from", "result", "TCamera-0", "hub"):
        assert fragment in text
    assert "FAILED" not in text
    assert "INCOMPLETE" not in text
