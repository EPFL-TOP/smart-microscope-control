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


def test_the_probe_table_shows_not_probed_and_its_reason() -> None:
    probe = ProbeResult(
        adapter="Fake",
        devices=[
            ProbedDevice(name="Hub1", type="Hub", status="ok"),
            ProbedDevice(
                name="Cam1", type="Camera", status="ok", source="hub", port="COM3"
            ),
            ProbedDevice(
                name="Bad1",
                type="Camera",
                status="failed",
                source="both",
                error="Native module failed to load (6)",
                hint="the vendor runtime could not be loaded",
            ),
            ProbedDevice(
                name="Skip1",
                type="Shutter",
                status="not-probed",
                reason="needs a port: pass --port COMx to probe it",
            ),
        ],
    )

    text = _flat(_inventory(probe))

    # The five headers of the probe table, in order and side by side: another
    # table of the report holds `device` and `type` too (FM-45).
    assert "device type from result detail" in text
    # Every result word, and what each one says in the detail column.
    assert "FAILED Native module failed to load (6) — the vendor runtime" in text
    assert "not probed needs a port: pass --port COMx to probe it" in text
    assert "ok port COM3" in text
    # The sources of the rows.
    assert "Cam1 Camera hub ok" in text
    assert "Bad1 Camera both FAILED" in text


def test_a_failed_row_shows_mmcores_text_with_or_without_a_hint() -> None:
    probe = ProbeResult(
        adapter="Fake",
        devices=[
            ProbedDevice(
                name="Bare",
                type="Camera",
                status="failed",
                error="Error in device X (10005)",
            ),
            ProbedDevice(
                name="Hinted",
                type="Camera",
                status="failed",
                error="Native module failed to load (6)",
                hint="the vendor runtime could not be loaded",
            ),
            ProbedDevice(
                name="Mute",
                type="Camera",
                status="failed",
                hint="only a hint is known",
            ),
        ],
    )

    text = _flat(_inventory(probe))

    # The raw error is the point of a probe: without a hint it must still show.
    assert "Bare Camera adapter FAILED Error in device X (10005)" in text
    assert "Hinted Camera adapter FAILED Native module failed to load (6) —" in text
    assert "Mute Camera adapter FAILED only a hint is known" in text


def test_a_probe_row_that_was_not_probed_never_reads_as_ok() -> None:
    probe = ProbeResult(
        adapter="Fake",
        devices=[
            ProbedDevice(
                name="Skip1", type="Shutter", status="not-probed", reason="why"
            )
        ],
    )

    text = _flat(_inventory(probe))

    assert "Skip1 Shutter adapter not probed why" in text
    assert "Skip1 Shutter adapter ok" not in text


def test_probe_text_is_never_read_as_markup() -> None:
    # FM-36: rich would drop `[red]` and `[roles.assign]` read as markup.
    probe = ProbeResult(
        adapter="Fake[x]",
        devices=[
            ProbedDevice(
                name="[b]Dev",
                type="Camera",
                status="failed",
                error="bad [roles.assign] here",
                hint="try [red]x",
            ),
            ProbedDevice(
                name="Dev2",
                type="Camera",
                status="not-probed",
                reason="why [roles.assign] [red]x",
            ),
        ],
        error="error [roles.assign] [red]x",
        notes=["note [roles.assign] [red]x"],
        causes=["cause [roles.assign] [red]x"],
    )

    text = _flat(_inventory(probe))

    for phrase in (
        "Probe of Fake[x]",
        "[b]Dev",
        "bad [roles.assign] here",
        "try [red]x",
        "why [roles.assign] [red]x",
        "error [roles.assign] [red]x",
        "- note [roles.assign] [red]x",
        "1. cause [roles.assign] [red]x",
    ):
        assert phrase in text


def test_an_incomplete_probe_says_so_in_its_title() -> None:
    whole = ProbeResult(adapter="Fake", devices=[], complete=True)
    partial = ProbeResult(adapter="Fake", devices=[], complete=False)

    assert "INCOMPLETE" not in _flat(_inventory(whole))
    text = _flat(_inventory(partial))
    assert "INCOMPLETE: the adapter child stopped before the last device" in text
    assert "Probe of Fake" in text


def test_causes_are_numbered_under_in_order_of_likelihood() -> None:
    probe = ProbeResult(
        adapter="Fake",
        devices=[ProbedDevice(name="H", type="Hub", status="failed", error="no")],
        causes=["first cause", "second cause", "third cause"],
    )

    text = _flat(_inventory(probe))

    assert (
        "In order of likelihood: 1. first cause 2. second cause 3. third cause" in text
    )


def test_a_probe_without_causes_or_notes_prints_neither() -> None:
    probe = ProbeResult(
        adapter="Fake", devices=[ProbedDevice(name="A", type="Camera", status="ok")]
    )

    text = _flat(_inventory(probe))

    assert "In order of likelihood" not in text


def test_the_listing_prints_the_meaning_under_the_devices() -> None:
    inv = _inventory()
    inv.adapters.listed = [
        AdapterInfo(
            name="NikonTI",
            installed=True,
            devices=[AdapterDevice(name="TIScope", type="Hub")],
            list_meaning="a fixed list proves nothing",
        ),
        AdapterInfo(
            name="NikonTi2",
            installed=True,
            devices=[],
            list_meaning="the SDK could not be reached",
        ),
        AdapterInfo(
            name="DemoCamera",
            installed=True,
            devices=[AdapterDevice(name="DCam", type="Camera")],
        ),
    ]

    text = _flat(inv)

    assert "TIScope Hub meaning: a fixed list proves nothing" in text
    assert "— no devices listed — meaning: the SDK could not be reached" in text
    # An adapter with nothing to add gets no row.
    assert text.count("meaning:") == 2
