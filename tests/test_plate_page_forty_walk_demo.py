"""A scan at the 40x from the page, end to end, on the demo-sample stand, through the view's own handlers (m5 §25 D7, #634).

What the owner did from the console in session 6, a button at a time and with no console: the camera
calibrated and the plate registered at the 10x, the turret turned to the 40x from the drive bar, the
camera calibrated there, a whole well laid out (its count is the claim) and a smaller plan scanned,
the cells found, a selector chosen **by name** with one of its parameters changed, and the list
exported. The second walk is the one with the focus map: a 10x scan taken with the autofocus locked,
which the page keeps by itself, the 40x scan with ``cell_focus`` ``map`` and the autofocus off, and
the kept file given to it.

The stand is ``demo-sample`` with the demo stage's ``Velocity`` raised on its own core (the demo
moves at 10 mm/s, and a calibration would take a minute). A click is ``button.clicks += 1``. A whole
well at the 40x is 2544 fields and 80 s of scanning on the demo, so the first walk lays it out once,
counts it, and scans a plan of ``coverage`` 0.4 (0.2 holds no cell of B2 on the demo, and the cells
stage then finds nothing to route). The selector is a throwaway class of this module
registered in place of the installed ones (``registry.entry_points``, with the built-ins kept), as
``test_plate_page_algorithms_demo.py`` does. The browser's part, and the 40x in a browser, are #667's
(FM-161): what is asserted here is what the handlers do.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # noqa: F401  (CI must fail, not skip, when the extra is missing)
else:
    pytest.importorskip("panel")

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile
from smc.plugins.nis_export.format import decode
from smc.plugins.slots import (
    NisExportResult,
    RouteResult,
    ScanPlanResult,
    ScanResult,
)
from smc.run import RunDocument, StageStatus
from smc.selection import registry
from smc.selection.api import Choice, SelectionInput, SelectorParams
from smc.selection.builtin import MostCells, MostCellsParams
from smc.ui.controller import PlatePage
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The fields of the whole of B2 at the 40x on the demo, ``coverage`` 1.0 (measured at ``2f2fab4``).
WHOLE_WELL_FIELDS = 2544
SCALE = 20.0


class WeightedParams(SelectorParams):
    """One parameter, so that the page's field for it and the run document's record of it are both seen."""

    scale: float = Field(10.0, gt=0, description="Cells per point of score.")


class Weighted:
    """``most-cells``' fields, each scored by the cells it holds over ``scale``."""

    name = "weighted"
    summary = "The fields most-cells chooses, scored by the cells they hold."
    Params = WeightedParams

    def select(self, data: SelectionInput, params: WeightedParams) -> list[Choice]:
        chosen = MostCells().select(data, MostCellsParams())
        found = []
        for choice in chosen:
            held = float(len(data.cells_in(choice.x_um, choice.y_um, choice.well)))
            found.append(
                replace(
                    choice,
                    score=held / params.scale,
                    measures={"cells_in_field": held},
                )
            )
        return found


def _register(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-ins and ``weighted``, in place of what is installed: nothing was read of the registry yet."""
    groups = {
        registry.SELECTOR_GROUP: [
            EntryPoint(
                "most-cells", "smc.selection.builtin:MostCells", registry.SELECTOR_GROUP
            ),
            EntryPoint("weighted", f"{__name__}:Weighted", registry.SELECTOR_GROUP),
        ],
        registry.ORDERER_GROUP: [
            EntryPoint(
                "nearest", "smc.selection.builtin:Nearest", registry.ORDERER_GROUP
            ),
            EntryPoint(
                "as-chosen", "smc.selection.builtin:AsChosen", registry.ORDERER_GROUP
            ),
        ],
    }
    monkeypatch.setattr(registry, "entry_points", lambda *, group: list(groups[group]))


def _click(button: Any) -> None:
    button.clicks += 1


@dataclass(frozen=True)
class Rig:
    """The page, its view, its stand and the run document of one walk."""

    page: PlatePage
    view: PlatePageView
    stand: Microscope
    run_doc: Path

    def finish(self) -> str:
        """Let the run end as a session would: ticks, then the redraw the end of a run asks for; the strip's outcome."""
        started = time.monotonic()
        while not self.page.join(0.01):
            self.view.tick()
            assert time.monotonic() - started < 600, "the run did not end"
        self.view.tick()
        self.view.refresh()
        outcome = self.page.status().outcome
        assert outcome is not None
        return outcome

    def press(self, screen: str, button: Any) -> str:
        """Show ``screen``, redraw as a session does, press ``button`` and wait for the run it started."""
        self.view.show_step(screen)  # type: ignore[arg-type]
        self.view.refresh()
        self.view.tick()
        _click(button)
        return self.finish()

    def stage(self, name: str) -> Any:
        return RunDocument.load(self.run_doc).stages[name]

    def calibrate(self, objective: str) -> None:
        """The Stand screen's *Calibrate the camera* at the objective in place."""
        assert self.stand.require(ObjectiveTurret).current() == objective
        outcome = self.press("stand", self.view.calibrate_button)
        assert outcome.startswith("Camera calibration done in "), outcome

    def register(self) -> None:
        """The Plate screen's *Register the plate*, from the sample's references."""
        outcome = self.press("plate", self.view.register_button)
        assert outcome.startswith("Registration done in "), outcome

    def turn_to(self, objective: str) -> None:
        """The drive bar's objective select, its button and its confirmation: the operator's turn (FM-14)."""
        self.view.objective_select.value = objective
        _click(self.view.objective_button)
        assert self.view.objective_confirm.visible is True
        _click(self.view.objective_confirm_button)
        self.finish()
        assert self.stand.require(ObjectiveTurret).current() == objective

    def lay_out(self, coverage: str, well: str = "B2") -> ScanPlanResult:
        """The Wells screen: ``well`` selected, ``coverage`` typed, *Lay out the fields*; the plan it wrote."""
        self.view.show_step("wells")
        self.page.select_text(well)
        self.view.tool_fields["scan-plan"]["coverage"].value = coverage
        self.view.refresh()
        self.view.tick()
        _click(self.view.layout_button)
        outcome = self.finish()
        assert outcome.startswith("Field layout done in "), outcome
        record = self.stage("scan-plan")
        assert record.status is StageStatus.done
        return ScanPlanResult.model_validate(record.result)

    def scan(self) -> ScanResult:
        """The Scan screen's *Start the scan*; the scan it wrote."""
        outcome = self.press("scan", self.view.scan_button)
        assert outcome.startswith("Scan done in "), outcome
        record = self.stage("scan")
        assert record.status is StageStatus.done
        return ScanResult.model_validate(record.result)

    def autofocus(self, on: bool) -> None:
        """The drive bar's *Autofocus on* or *off*."""
        _click(self.view.autofocus_on_button if on else self.view.autofocus_off_button)
        self.finish()


@pytest.fixture
def rig(
    mm_available: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Rig]:
    """A stand, a page over it and a view, with ``weighted`` registered; the stand and the page are closed afterwards."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    _register(monkeypatch)
    run_doc = tmp_path / "run.json"
    stand = Microscope.open(Profile.load(DEMO_SAMPLE))
    stand.core.setProperty("XY", "Velocity", "100000")
    page = PlatePage(stand, run_doc)
    try:
        yield Rig(page, PlatePageView(page), stand, run_doc)
    finally:
        page.close()
        stand.close()


def test_the_40x_walk_from_the_page_ends_with_the_exported_list(rig: Rig) -> None:
    view = rig.view
    # 1. At the 10x: the camera calibrated, the plate registered from the sample's references.
    rig.calibrate(TEN)
    rig.register()
    assert rig.stage("plate-registration").status is StageStatus.done

    # 2. The turn to the 40x is the drive bar's (a select, a button and a confirmation), and the
    #    camera is calibrated there: a registration from the sample's references measures no wall,
    #    so the page does not know the plate's objective and the calibration replaces the run
    #    document's own (a registration from walls keeps the 10x's and writes the 40x's into
    #    ``calibrations/``, which test_plate_page_plan_calibration_demo.py walks).
    rig.turn_to(FORTY)
    rig.calibrate(FORTY)
    calibration = rig.stage("camera-calibration")
    assert calibration.status is StageStatus.done
    assert calibration.result is not None
    assert calibration.result["objective"] == FORTY

    # 3. The whole well is laid out once, and its count is what the Wells screen says: B2 at the
    #    40x is 2544 fields, and the plan is the 40x's, from that calibration.
    whole = rig.lay_out("1.0")
    assert whole.objective == FORTY
    assert whole.matrix_source == "camera-calibration"
    assert len(whole.fields) == WHOLE_WELL_FIELDS
    lines = view.wells_known_block.object.split("\n")
    assert lines[0] == f"Objective: {FORTY}, in place"
    assert lines[1] == f"Fields: {WHOLE_WELL_FIELDS} in 1 well"
    assert lines[2] == "Time: not known (no timing measured yet)"
    assert len(view.fields_source.data["x"]) == WHOLE_WELL_FIELDS

    # 4. The scan is of a smaller plan, the middle of the well: 80 s a whole well on the demo is the
    #    stand's to spend, and the middle holds cells (a plan of coverage 0.2 holds none).
    plan = rig.lay_out("0.4")
    assert 0 < len(plan.fields) < WHOLE_WELL_FIELDS
    scanned = rig.scan()
    assert len(scanned.frames) == len(plan.fields)
    assert scanned.failed == ()
    assert scanned.objective == FORTY

    # 5. The Results screen: the selector chosen by name on the route row, one parameter changed;
    #    then *Run all that are left* takes the cells to the export.
    route = view.result_rows["route"]
    fields = view.tool_fields["route"]
    assert fields["selector"].value == ""
    fields["selector"].value = "weighted"
    (scale,) = view._algorithms["selector"].fields["weighted"].values()
    scale.value = str(int(SCALE))
    assert fields["selector_params"].value == '{"scale": 20.0}'
    assert "selector=weighted" in route.command.object
    outcome = rig.press("results", view.run_all_button)
    assert outcome.startswith("Run all that are left: done in "), outcome

    # 6. The run document records the name and the typed value, as `smc run route --param ...`
    #    would, and every stop is scored.
    stages = RunDocument.load(rig.run_doc).stages
    assert [
        stages[name].status
        for name in ("cells", "density", "route", "report", "nis-export")
    ] == [StageStatus.done] * 5
    assert stages["route"].params["selector"] == "weighted"
    assert stages["route"].params["selector_params"] == {"scale": SCALE}
    assert stages["route"].result is not None
    routed = RouteResult.model_validate(stages["route"].result)
    assert len(routed.positions) > 0
    assert routed.objective == FORTY
    assert all(stop.score is not None for stop in routed.positions)

    # 7. The export lists the stops of the route, in the route's order, at the 40x.
    assert stages["nis-export"].result is not None
    exported = NisExportResult.model_validate(stages["nis-export"].result)
    assert exported.positions == len(routed.positions)
    points: Any = decode((rig.run_doc.parent / exported.file).read_bytes())
    stops = sorted(routed.positions, key=lambda stop: stop.order)
    assert [point.name for point in points] == [stop.name for stop in stops]
    # The page kept no focus map for a walk with no locked frame (the autofocus was never on).
    assert not rig.page.focus_map_path.exists()


def test_the_40x_walk_with_the_map_uses_the_map_the_page_kept(rig: Rig) -> None:
    view = rig.view
    # 1. The 10x half, as at the stand: the camera calibrated, the plate registered, B2 laid out,
    #    *Autofocus on* (PFS locked at the 10x), and the scan. Nothing is copied by hand.
    rig.calibrate(TEN)
    rig.register()
    ten = rig.lay_out("0.4")
    assert ten.objective == TEN
    rig.autofocus(True)
    first = rig.scan()
    assert {frame.focus for frame in first.frames} == {"locked"}
    kept = rig.page.focus_map_path
    assert kept == rig.run_doc.parent / "run.focus-map.json"
    kept_bytes = kept.read_bytes()
    assert kept_bytes == rig.run_doc.read_bytes()

    # 2. The 40x half: the autofocus off, the turn, the camera calibrated there, the plan.
    rig.autofocus(False)
    rig.turn_to(FORTY)
    rig.calibrate(FORTY)
    forty = rig.lay_out("0.2")
    assert forty.objective == FORTY
    # The run document no longer holds the 10x plan the map was scanned from: this is why the page
    # keeps the copy itself and does not leave it to the operator (#682's review).
    assert RunDocument.load(kept).stages["scan-plan"].result is not None
    kept_plan = ScanPlanResult.model_validate(
        RunDocument.load(kept).stages["scan-plan"].result
    )
    assert kept_plan.objective == TEN
    assert (
        rig.stage("scan-plan").attempt
        != RunDocument.load(kept).stages["scan-plan"].attempt
    )

    # 3. The Scan screen: `map` typed, the autofocus off; the screen names the kept scan and the
    #    well to focus on by eye, and the command carries the kept file.
    view.show_step("scan")
    view.tool_fields["scan"]["cell_focus"].value = "map"
    view.tool_fields["scan"]["autofocus"].value = "off"
    view.refresh()
    view.tick()
    first_line, anchor = view.focus_map_text.object.split("\n")
    assert first_line.startswith(f"Focus map: the scan of B2 at {TEN}, ")
    assert first_line.endswith(" UTC, kept by the page in run.focus-map.json.")
    assert anchor == (
        "Before you start: with Live, focus by eye on the cells near the centre of B2. "
        "That focus is the scan's starting point; each well is then anchored near its centre."
    )
    assert f"focus_map={kept}" in view.scan_command.object.replace("'", "")
    assert view.scan_button.disabled is False, view.scan_wait_text.object

    # 4. *Start the scan*: done, with the kept file as its map.
    scanned = rig.scan()
    assert len(scanned.frames) == len(forty.fields)
    assert {frame.focus for frame in scanned.frames} == {"off"}
    record = rig.stage("scan")
    assert record.params["cell_focus"] == "map"
    assert os.path.samefile(record.params["focus_map"], kept)
    assert scanned.focus_map is not None
    assert Path(scanned.focus_map.run_doc).as_posix() == kept.as_posix()
    # The scan with the map was not locked, so it kept nothing: the map is still the 10x's.
    assert kept.read_bytes() == kept_bytes
    # The Results screen then goes on from this scan, as after any other (the list is the next walk).
    assert view.scanned_block.object.split("\n")[0] == (
        f"Scanned: {len(forty.fields)} frames, 0 failed"
    )
