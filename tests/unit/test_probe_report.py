"""The probe's models and how the report shows them (design m2 §9.2, §9.6).

A probe row never says ``ok`` for something nobody initialised: ``not-probed``
is a third result, and the report must show it with its reason.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from smc.discovery.models import (
    AdapterDevice,
    AdapterInfo,
    Inventory,
    ProbedDevice,
    ProbeResult,
    SystemInfo,
)
from smc.discovery.report import render_text


def _inventory(probe: ProbeResult | None = None) -> Inventory:
    system = SystemInfo(
        hostname="host",
        collected_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
        os="os",
        python="3.13",
        smc="0",
        pymmcore_plus="0",
    )
    inv = Inventory(system=system)
    inv.probe = probe
    return inv


def _flat(inv: Inventory) -> str:
    # The table folds long cells at 120 columns (FM-37): match whitespace-normalised.
    return " ".join(render_text(inv).split())


def test_a_probed_device_is_ok_only_when_its_status_is_ok_and_ok_is_not_serialised() -> (
    None
):
    ok = ProbedDevice(name="A", type="Camera", status="ok")
    failed = ProbedDevice(name="B", type="Camera", status="failed", error="boom")
    skipped = ProbedDevice(name="C", type="Camera", status="not-probed", reason="why")

    assert [d.ok for d in (ok, failed, skipped)] == [True, False, False]
    # A property, not a field: the old JSON key is gone, so no reader can take
    # a "not probed" row for a pass by reading it.
    # Keys, not a substring of the text: "status": "ok" holds the letters too.
    assert "ok" not in ok.model_dump()
    assert "ok" not in json.loads(ok.model_dump_json())
    assert ok.model_dump()["status"] == "ok"
    assert ok.source == "adapter"


def test_an_inventory_with_a_probe_round_trips_through_json() -> None:
    probe = ProbeResult(
        adapter="NikonTi2",
        devices=[
            ProbedDevice(name="Hub", type="Hub", status="ok"),
            ProbedDevice(
                name="Port dev",
                type="Shutter",
                status="ok",
                source="both",
                port="COM3",
            ),
            ProbedDevice(
                name="Bad",
                type="Camera",
                status="failed",
                source="hub",
                error="Native module failed to load (6)",
                hint="the vendor runtime could not be loaded",
            ),
            ProbedDevice(
                name="Y", type="Camera", status="not-probed", reason="not reported"
            ),
        ],
        error="nothing below it was probed",
        notes=["a note"],
        causes=["one", "two"],
        complete=False,
    )
    inv = _inventory(probe)
    inv.adapters.listed = [
        AdapterInfo(
            name="NikonTi2",
            installed=True,
            devices=[AdapterDevice(name="Hub", type="Hub")],
            list_meaning="the SDK could not be reached",
        )
    ]

    back = Inventory.model_validate_json(inv.model_dump_json())

    assert back == inv
    assert back.probe is not None
    assert back.probe.complete is False
    assert back.adapters.listed[0].list_meaning == "the SDK could not be reached"
