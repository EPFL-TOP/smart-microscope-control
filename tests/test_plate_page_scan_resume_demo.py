"""A scan STOPped and continued from the plate page, on the demo's simulated stage (m5 §17.12 D17, #470).

The owner's walk of 2026-11-13 stopped the scan of ``demo-sample`` with STOP, and the page could
not go on: its record was ``failed`` (the STOP stopped a move, FM-64), its checkpoint stood, and
the page offered only ``smc run scan --resume`` after closing it. This runs the controller the
way the Scan screen's buttons do, without ``--dry-run`` (a dry-run scan of the demo's 72 fields
walks them in 0.03 s, before any STOP can land: measured while planning #470), on the real
plugins: the fields laid out, the scan started, STOP pressed in the second well (the scan
checkpoints after each well, so a STOP in the first leaves nothing to continue), *Resume*, then
*Continue the scan*. Every field of the plan has one frame, once.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import pytest

from smc.hardware.profile import Profile
from smc.plugins.slots import ScanPlanResult, ScanResult
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage
from smc.ui.guide import tool_title

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
#: A bound on each wait, never a timing (FM-44): the whole scan takes about 10 s here.
BOUND_S = 120.0


def _ran(page: PlatePage, plugin: str, params: list[str]) -> None:
    page.run(plugin, params)
    assert page.join(BOUND_S), f"{plugin} did not end"
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith(f"{tool_title(plugin)} done in "), outcome


def test_a_scan_stopped_and_continued_from_the_page_takes_every_field_once(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), run_doc)
    try:
        page.connect()
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome
        page.select_text("B2, B11")
        registration = page.view().next_step
        assert registration is not None
        assert registration.plugin == "plate-registration"
        _ran(page, "plate-registration", list(registration.params))
        # A smaller coverage than the profile's 0.4: fewer fields, the same two wells.
        _ran(page, "scan-plan", [*page.page_params("scan-plan"), "coverage=0.25"])
        planned = RunDocument.load(run_doc).stages["scan-plan"].result
        assert planned is not None
        plan = ScanPlanResult.model_validate(planned)
        first_well = plan.fields[0].well
        in_first = sum(field.well == first_well for field in plan.fields)
        assert 0 < in_first < len(plan.fields)

        page.run("scan", page.page_params("scan"))
        # STOP once the first well is checkpointed and the second one is under way: a bounded
        # count of polls (FM-44).
        stopped_at = None
        for _ in range(int(BOUND_S / 0.01)):
            status = page.status()
            if status.running != "scan":
                break
            if status.progress is not None and status.progress.done > in_first:
                stopped_at = status.progress.done
                page.stop()
                break
            threading.Event().wait(0.01)
        assert stopped_at is not None, "the scan ended before STOP could land"
        assert page.join(BOUND_S), "the stopped scan did not end"
        record = RunDocument.load(run_doc).stages["scan"]
        assert record.status in (StageStatus.stopped, StageStatus.failed)
        end = page.view().scan_end
        assert end is not None
        assert end.kind == "stopped"
        assert end.resumable
        assert end.total == len(plan.fields)
        assert end.done is not None
        assert end.done >= stopped_at

        page.resume()
        page.run("scan", page.page_params("scan"), resume=True)
        assert page.join(BOUND_S), "the continued scan did not end"
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith("Scan done in "), outcome
        done = RunDocument.load(run_doc).stages["scan"]
        assert done.status is StageStatus.done
        assert done.result is not None
        scan = ScanResult.model_validate(done.result)
        names = Counter(frame.name for frame in scan.frames)
        assert scan.failed == ()
        assert set(names) == {field.name for field in plan.fields}
        assert all(count == 1 for count in names.values()), names
        assert page.view().scan_end is None
    finally:
        page.close()
        page.release()
