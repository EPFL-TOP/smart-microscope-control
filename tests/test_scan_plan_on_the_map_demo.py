"""The scan plan on the plate page: its fields on the map and the time the scan will take (m5 §14, #320).

The owner's walk of the first run (#310, finding 4) ended at ``scan-plan done`` with a
map that had not changed and a card that did not say how long the scan would take. This
runs the page's controller as the page's buttons do, on the ``demo-sample`` stand, and
reads what the page reads: ``view().plan`` for the map's layer and the card's lines for
the estimate. The real ``plate-registration``, ``timing`` and ``scan-plan`` run on
Micro-Manager's demo devices; the stage is simulated, so no duration is asserted (the
estimate is of the demo's own timings, FM-44).

Measured 2026-11-09, Python 3.13.0: ``scan-plan wells=B2,B11 coverage=0.4`` holds 72 fields
(6 x 6 in each of the two wells) of 512 x 512 µm; ``timing`` takes 5.7 s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.hardware.profile import Profile
from smc.ui.controller import PlatePage
from smc.ui.guide import tool_title

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
UNKNOWN = "estimate: unknown, no usable timing: run timing, then scan-plan again"


def _run(page: PlatePage, plugin: str, params: list[str]) -> None:
    page.run(plugin, params)
    assert page.join(120.0), f"{plugin} did not end"
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith(f"{tool_title(plugin)} done in "), outcome


def _registered_and_planned(page: PlatePage, *, with_timing: bool) -> None:
    """Connect, register the plate from the sample, optionally time the stand, plan B2 and B11."""
    page.connect()
    assert page.join(60.0), "connect did not end"
    assert page.connected, page.status().outcome
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"
    _run(page, step.plugin, list(step.params))
    if with_timing:
        _run(page, "timing", [])
    # Typed in this order on purpose: the plan's own order is not the typing order.
    page.select_text("B2,B11")
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "scan-plan"
    assert step.blocked is None, step.blocked
    _run(page, "scan-plan", list(step.params))


def test_scan_plan_after_timing_draws_72_fields_and_says_about_how_long(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), tmp_path / "run.json")
    try:
        _registered_and_planned(page, with_timing=True)
        view = page.view()
        assert view.problems == ()
        plan = view.plan
        assert plan is not None
        assert (plan.count, plan.wells) == (72, 2)
        assert plan.fov_um == (512.0, 512.0)
        assert len(plan.fields) == 72
        lines = view.cards["scan-plan"].lines
        assert [line.split(": ")[0] for line in lines] == [
            "wells",
            "fields",
            "field",
            "estimate",
        ]
        assert lines[0] == "wells: 2 (B11, B2)"
        assert lines[3].startswith("estimate: about "), lines[3]
        assert not view.cards["scan-plan"].stale
    finally:
        page.close()
        page.release()


def test_scan_plan_with_no_timing_says_it_does_not_know(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), tmp_path / "run.json")
    try:
        _registered_and_planned(page, with_timing=False)
        view = page.view()
        plan = view.plan
        assert plan is not None
        assert plan.count == 72
        # An unknown is not a zero: the card says what is missing and what to run.
        assert view.cards["scan-plan"].lines[3] == UNKNOWN
    finally:
        page.close()
        page.release()
