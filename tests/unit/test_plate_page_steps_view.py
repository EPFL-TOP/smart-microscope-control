"""The plate page's step bar, its screens and the expert switch, without a browser (m5 §17, #443).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1``,
the switch is ``expert_toggle.value``, a typed range is ``select_input.value``. What this
cannot show is a browser: that a swap keeps a value the operator typed *in the DOM*, that
the figures keep their view, and the layout at 1280 x 800 are for ``/ui-check`` (FM-161):
here the model tree and the lists are asserted. Panel's warnings are errors (FM-147), and
a handler that failed fails the test that did not expect it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

import smc.ui.plate_page as plate_page
from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import PlateResult, ScanPlanResult
from smc.run import StageRecord, StageStatus, write_stage
from smc.ui.controller import PlatePage
from smc.ui.guide import SCREEN_LEADS, SCREEN_TITLES, STEPS, continue_label
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)

#: The registry's tools, loaded from their own ``value`` so that every screen has its
#: sections; ``plate-registration`` is this module's stub (it ends at once).
REAL = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "cells": "smc.plugins.cells.plugin:CellsPlugin",
    "density": "smc.plugins.density.plugin:DensityPlugin",
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "nis-export": "smc.plugins.nis_export.plugin:NisExportPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
    "report": "smc.plugins.report.plugin:ReportPlugin",
    "route": "smc.plugins.route.plugin:RoutePlugin",
    "scan": "smc.plugins.scan.plugin:ScanPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
    "timing": "smc.plugins.timing.plugin:TimingPlugin",
}


class RegistrationParams(PluginParams):
    plate: str = "96-well"
    refs: str = "[]"
    walls: str = "[]"
    orientation: str = "not-turned"


def _plate_result() -> PlateResult:
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=1,
    )


class Registration:
    """A registration that ends at once: no tick can fall inside its run."""

    name = "plate-registration"
    provides = "plate"
    summary = "A test registration."
    Params = RegistrationParams
    Result = PlateResult
    requires: tuple[type, ...] = (XYStage,)
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PlateResult:
        return _plate_result()


class Quick:
    """A tool that does nothing and ends at once, for a run that changes no step."""

    class Params(PluginParams):
        pass

    class Result(PluginResult):
        pass

    name = provides = "quick"
    summary = "A test tool."
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> Quick.Result:
        return Quick.Result()


def _entry_points() -> list[EntryPoint]:
    found = [EntryPoint(name=n, value=v, group=GROUP) for n, v in REAL.items()]
    for cls in (Registration, Quick):
        found.append(
            EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)
        )
    return found


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = PlatePage(fake_microscope, run_doc, entry_points=_entry_points())
    try:
        yield controller
    finally:
        controller.close()


def _seed_plate(run_doc: Path) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=T0,
            finished_utc=T0 + timedelta(minutes=1),
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=_plate_result().model_dump(mode="json"),
        ),
    )


def _scan_plan_result() -> ScanPlanResult:
    """A scan plan of 2 x 2 fields in each of B2 and B11 (8 fields), as test_plate_page.py builds one."""
    fov_um = (512.0, 512.0)
    half_w, half_h = fov_um[0] / 2, fov_um[1] / 2
    plate_plan, fields = plan_fields(
        _plate_result(),
        ["B2", "B11"],
        fov_um=fov_um,
        footprint_of=lambda x, y: [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ],
        rows=2,
        columns=2,
        overlap=0.1,
        clip_to_well=True,
    )
    return ScanPlanResult(
        plate_plan=plate_plan,
        wells=("B2", "B11"),
        objective="Nikon 10X S Fluor",
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        matrix_source="parameter",
        image_shape_px=(512, 512),
        fov_um=fov_um,
        overlap=0.1,
        rows=2,
        columns=2,
        fields=fields,
        estimate_s=None,
    )


def _seed_scan_plan(run_doc: Path) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin="scan-plan",
            provides="scan-plan",
            attempt="a1",
            status=StageStatus.done,
            started_utc=T0 + timedelta(minutes=1),
            finished_utc=T0 + timedelta(minutes=2),
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=_scan_plan_result().model_dump(mode="json"),
        ),
    )


def _ids_below(items: list[Any]) -> set[int]:
    """The id of every object under ``items``, the Accordion's panels too (``select`` stops at one)."""
    found: set[int] = set()
    for item in items:
        found.update(id(o) for o in item.select())
        for accordion in item.select(pn.Accordion):
            for panel in accordion.objects:
                found.update(id(o) for o in panel.select())
    return found


def _next_regions(items: list[Any]) -> list[Any]:
    return [
        item
        for item in items
        if isinstance(item, pn.Column)
        and getattr(item.objects[0], "object", None) == "Next step"
    ]


# -- the opening step ---------------------------------------------------------------


def test_the_page_opens_on_the_step_the_run_document_and_the_connection_suggest(
    page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    # Connected, the run document has no plate: the suggestion is the Plate step.
    assert PlatePageView(page).step == "plate"
    # With a plate in it, the next slot is the scan plan: the Wells step.
    _seed_plate(run_doc)
    assert PlatePageView(page).step == "wells"
    # A page that is not connected opens on the stand, whatever the document holds.
    closed = PlatePage.closed(Profile.demo(), run_doc, entry_points=_entry_points())
    try:
        view = PlatePageView(closed)
        assert view.step == "stand"
        assert (
            view.step_buttons["stand"].label,
            view.step_summaries["stand"].object,
        ) == (
            "1. Stand",
            "not connected",
        )
    finally:
        closed.close()


# -- the screens --------------------------------------------------------------------


def test_the_step_panel_shows_one_screen_at_a_time(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.step_panel.width == plate_page.STEP_PANEL_WIDTH_PX == 400
    assert view.step_panel.scroll is True
    for step in STEPS:
        view.show_step(step)
        shown = list(view.step_panel.objects)
        assert shown == view.screens[step]
        # A copy: a swap never hands the panel the screen's own list (§17.6).
        assert view.step_panel.objects is not view.screens[step]
        title, lead, *_rest = shown
        assert title.object == SCREEN_TITLES[step]
        assert title.styles["font-weight"] == "bold"
        assert lead.object == SCREEN_LEADS[step]
        # Every screen is designed now (B2: Stand and Plate; B3: Wells and Scan; B4: Results):
        # each holds its own buttons and none the next step, which is the expert view's. The
        # Scan screen's title and lead are the fallbacks of a page with no plan and no
        # selection, which is what this page is (m5 §17.10.3).
        assert _next_regions(shown) == []
        assert view.continue_button in shown
        # The step shown is the primary button, and the one `/ui-check` finds by its class.
        colours = {s: b.color for s, b in view.step_buttons.items()}
        assert colours == {s: ("primary" if s == step else "light") for s in STEPS}
        classes = {s: b.css_classes for s, b in view.step_buttons.items()}
        assert classes == {
            s: (["smc-step", "smc-step-shown"] if s == step else ["smc-step"])
            for s in STEPS
        }
        assert view.continue_button.label == (continue_label(step) or "")
        assert view.continue_button.visible is (step != "results")
    # Each screen's provisional contents are today's regions (B1), listed by step (m5
    # §17.7), written out here and not read from the code's table (FM-43): the tools of
    # a screen are exactly these, so that a tool of another step does not leak onto it
    # (found by the adversarial review of #443: each leak below survived every test).
    on = {
        step: {id(o) for o in _ids_objects(view.screens[step])}
        for step in (*STEPS, "expert")
    }
    tools_on = {
        step: {name for name, button in view.tool_runs.items() if id(button) in ids}
        for step, ids in on.items()
    }
    assert tools_on == {
        # Every screen is designed (B2: Stand and Plate; B3: Wells and Scan; B4: Results):
        # their actions are their own buttons and rows, and no tool section is listed on
        # them (the sections stay in the expert view). The Results rows' buttons are their
        # own: `tool_runs` holds the sections' *Run* buttons only.
        "stand": set(),
        "plate": set(),
        "wells": set(),
        "scan": set(),
        "results": set(),
        # The expert view lists every tool, the ones no step holds included (`goto-well`
        # is a third party's, as far as the steps go: in the expert view only).
        "expert": set(view.tool_runs),
    }
    assert "goto-well" in view.tool_runs
    # The page's own controls are each on their screen and on no other step's.
    homes = {
        "preset_select": {"stand", "expert"},
        "plate_select": {"plate", "expert"},
        "orientation_select": {"plate", "expert"},
        "plate_confirm_button": {"plate", "expert"},
        "select_input": {"wells", "expert"},
        # The run document's path closes the Results screen and the expert view; its
        # checklist and the next step are the expert view's alone (B4, m5 §17.4.5).
        "run_document_text": {"results", "expert"},
        "checklist_text": {"expert"},
        "continue_button": set(STEPS),
        "next_text": {"expert"},
        # B2's own widgets: each on its screen alone, `plate_from_text` with the menu it
        # explains (the expert view lists the same column).
        "stand_connect_button": {"stand"},
        "known_block": {"stand"},
        "calibrate_button": {"stand"},
        "register_button": {"plate"},
        "registered_block": {"plate"},
        "plate_from_text": {"plate", "expert"},
        # B3's: each on its screen alone. The *Wells* box and the tools' two fields are
        # on the Wells screen and in the expert view (one instance each, D5); the buttons
        # and the lines are the screens' own.
        "wells_from_text": {"wells"},
        "layout_button": {"wells"},
        "layout_wait": {"wells"},
        "wells_known_block": {"wells"},
        "wells_note": {"wells"},
        "layout_card": {"wells"},
        "scan_title_text": {"scan"},
        "scan_lead_text": {"scan"},
        "scan_button": {"scan"},
        "scan_wait_text": {"scan"},
        "scan_dry_note": {"scan"},
        "scan_progress": {"scan"},
        "scan_progress_text": {"scan"},
        "scanned_block": {"scan"},
        "scan_card": {"scan"},
        # B4's: each on the Results screen alone.
        "run_all_button": {"results"},
        "run_all_wait_text": {"results"},
        "chain_text": {"results"},
        "files_heading": {"results"},
        "files_text": {"results"},
    }
    for attribute, screens in homes.items():
        widget = getattr(view, attribute)
        assert {s for s, ids in on.items() if id(widget) in ids} == screens, attribute
    # Nothing in a screen is wider than the step panel's text (400 px less its scrollbar
    # and margins, m5 §17.3): a fixed width past 370 runs past the panel's edge.
    for step in STEPS:
        too_wide = [
            type(o).__name__
            for o in _ids_objects(view.screens[step])
            if isinstance(getattr(o, "width", None), int) and o.width > 370
        ]
        assert too_wide == [], f"{step} places a fixed width past 370 px: {too_wide}"


def _ids_objects(items: list[Any]) -> list[Any]:
    """Every object under ``items``, the Accordion's panels too."""
    found: list[Any] = []
    for item in items:
        found.extend(item.select())
        for accordion in item.select(pn.Accordion):
            for panel in accordion.objects:
                found.extend(panel.select())
    return found


def test_a_widget_on_two_screens_keeps_its_value_across_a_switch(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    root = view.layout.get_root()
    coverage = view.tool_fields["scan-plan"]["coverage"]
    view.show_step("wells")
    # One instance on both screens: the *Wells* box, and a tool's own field.
    assert view.select_input in view.screens["wells"]
    assert view.select_input in view.screens["expert"]
    assert coverage in _ids_objects(view.screens["wells"])
    assert coverage in _ids_objects(view.screens["expert"])
    view.select_input.value = "A1:B2"
    coverage.value = "0.4"
    assert page.selection == ("A1", "A2", "B1", "B2")
    for screen in (
        lambda: view.show_expert(True),
        lambda: view.show_step("plate"),
        lambda: view.show_step("wells"),
        lambda: view.show_expert(True),
        lambda: view.show_expert(False),
    ):
        screen()
        assert view.select_input.value == "A1:B2"
        assert coverage.value == "0.4"
        assert page.selection == ("A1", "A2", "B1", "B2")
        assert root.references()  # the model tree builds after every swap


# -- the expert view ----------------------------------------------------------------


def test_the_expert_view_holds_every_tool_and_the_step_comes_back(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.show_step("scan")
    assert view.expert_toggle.value is False
    # The switch, as the operator presses it.
    view.expert_toggle.value = True
    assert view.expert is True
    assert list(view.step_panel.objects) == view.screens["expert"]
    # Every tool, whatever its step: its Run button is in the accordion the view lists.
    placed = {id(o) for o in _ids_objects(list(view.step_panel.objects))}
    assert all(id(button) in placed for button in view.tool_runs.values())
    assert id(view.accordion) in placed
    assert id(view.run_document_text) in placed
    assert id(view.checklist_text) in placed
    assert id(view.preset_select) in placed
    assert id(view.select_input) in placed
    assert id(view.plate_select) in placed
    assert id(view.orientation_select) in placed
    assert id(view.plate_confirm_button) in placed
    # No step is shown while the expert view is: none is the primary button.
    assert {b.color for b in view.step_buttons.values()} == {"light"}
    assert not any(
        "smc-step-shown" in b.css_classes for b in view.step_buttons.values()
    )
    # Off again: the step it left comes back, and the continue button with it.
    view.expert_toggle.value = False
    assert (view.step, view.expert) == ("scan", False)
    assert list(view.step_panel.objects) == view.screens["scan"]
    assert view.step_buttons["scan"].color == "primary"
    assert view.continue_button.label == "Continue to Results"


def test_a_step_click_in_the_expert_view_leaves_it(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.expert_toggle.value = True
    assert view.expert
    view.step_buttons["wells"].clicks += 1
    assert (view.step, view.expert) == ("wells", False)
    # The switch follows, and the screen is the step's, not the expert view's.
    assert view.expert_toggle.value is False
    assert list(view.step_panel.objects) == view.screens["wells"]
    # `show_step` does it too, whoever calls it.
    view.show_expert(True)
    view.show_step("results")
    assert (view.step, view.expert, view.expert_toggle.value) == (
        "results",
        False,
        False,
    )


def test_continue_shows_the_next_step(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.show_step("stand")
    seen = []
    for expected in ("plate", "wells", "scan", "results"):
        assert view.continue_button.label == f"Continue to {expected.title()}"
        view.continue_button.clicks += 1
        seen.append(view.step)
        assert list(view.step_panel.objects) == view.screens[expected]
    assert seen == ["plate", "wells", "scan", "results"]
    # Nothing after the last: the button is hidden, and a click that still arrives does nothing.
    assert view.continue_button.visible is False
    view.continue_button.clicks += 1
    assert view.step == "results"
    assert list(view.step_panel.objects) == view.screens["results"]


# -- what stays on screen -----------------------------------------------------------


def test_both_figures_stay_in_the_model_tree_whatever_the_screen(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    root = view.layout.get_root()

    # A figure that leaves the model tree has its view destroyed in the browser, and
    # its ranges keep the bounds of the dead one (FM-174): what a dynamic tab did to the
    # figure of the tab not shown (#297). Neither figure is in a swapped list, so no
    # swap can take one out: every step, the expert view on and off, then a step.
    def both_in_tree() -> None:
        in_tree = {model.id for model in root.references()}
        assert view.figure.id in in_tree
        assert view.image_figure.id in in_tree

    for step in STEPS:
        view.show_step(step)
        both_in_tree()
    view.show_expert(True)
    both_in_tree()
    view.show_expert(False)
    both_in_tree()
    view.show_step("wells")
    both_in_tree()
    # No tab exists to hide one, in any screen (the accordion's panels included), and
    # no figure is in a screen's list.
    assert not view.layout.select(pn.Tabs)
    figures = {id(view.figure), id(view.image_figure)}
    for items in view.screens.values():
        assert not figures & {id(o) for o in _ids_objects(items)}
        assert not any(isinstance(o, pn.pane.Bokeh) for o in _ids_objects(items))
    # Both are the same height, side by side in the centre. The design said 460 px, then 400;
    # 360 is the floor (m5 §17.12 D15, measured at 1280 x 800 with the drive bar's toggle in view):
    # a figure lower than that is a plan deviation to report, not a fix, so the number is
    # pinned and not only the constant against itself.
    assert view.figure.height == view.image_figure.height == plate_page.FIGURE_HEIGHT_PX
    assert plate_page.FIGURE_HEIGHT_PX >= 360
    assert plate_page.STEP_PANEL_WIDTH_PX == 400


def test_the_strip_and_stop_are_outside_the_step_panel(page: PlatePage) -> None:
    view = PlatePageView(page)
    # The frame is the layout with an empty step panel, not the layout less the screen
    # shown: a widget of the frame that the screen shown also lists would be subtracted
    # with it and never compared (the Plate screen is the one the page opens on; found by
    # the adversarial review of #443).
    view.step_panel.objects = []
    always = {id(o) for o in view.layout.select()}
    view.show_step(view.step)
    for widget in (
        view.connect_button,
        view.disconnect_button,
        view.status_text,
        view.resume_button,
        view.stop_button,
        view.expert_toggle,
        view.label_select,
        view.refresh_button,
        view.snap_button,
        view.live_button,
        view.side_select,
        view.drive_toggle,
        view.xy_step_select,
        *view.step_buttons.values(),
        *view.step_summaries.values(),
        *view.jog_buttons.values(),
        *view.z_buttons.values(),
    ):
        assert id(widget) in always
    # No object of the always-rendered part is in any screen: it would be drawn in only
    # one of the two places, and nothing would fail (m5 §17.2 D5).
    for step, items in view.screens.items():
        assert not always & _ids_below(items), (
            f"{step} lists a widget of the page's frame"
        )
    # Neither is a screen's object in the frame while another is shown.
    view.show_expert(True)
    assert not always & _ids_below(list(view.step_panel.objects))


def test_the_next_step_is_in_the_expert_view_only_and_drawn_once(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    for name, items in view.screens.items():
        regions = _next_regions(items)
        if name != "expert":
            # Every step screen is designed (B2 to B4): the action is the screen's own, and
            # the walk found the next step's 12-line command (`plate-registration`, `Run
            # scan-plan`) on them.
            assert regions == [], f"{name} lists the next step"
            continue
        assert len(regions) == 1, f"{name} lists the next step {len(regions)} times"
        assert items.index(regions[0]) == 1
    # Listed once, so that it is drawn once and its place never depends on the state:
    # refresh() never rewrites a screen's list.
    before = {name: list(items) for name, items in view.screens.items()}
    view.refresh()
    view.tick()
    assert {name: list(items) for name, items in view.screens.items()} == before
    for name in (*STEPS, "expert"):
        if name == "expert":
            view.show_expert(True)
        else:
            view.show_step(name)  # type: ignore[arg-type]
        rendered = [o for o in _ids_objects(list(view.step_panel.objects))]
        drawn = 1 if name == "expert" else 0
        assert sum(o is view.next_text for o in rendered) == drawn
        assert sum(o is view.next_button for o in rendered) == drawn


# -- what changes the screen, and what never does ------------------------------------


def test_a_snap_or_a_runs_end_never_changes_the_screen(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    assert view.step == "plate"
    view.show_step("scan")
    assert view.step_buttons["plate"].label == "2. Plate"
    view.on_snap()
    page.join(10.0)
    view.tick()
    assert (view.step, view.expert) == ("scan", False)
    # A run that makes the Plate step done: its tick redraws the step bar, not the screen.
    page.choose_plate(page.plate)  # confirms the placeholder: a registration may run
    view.on_run("plate-registration")
    assert page.join(10.0)
    view.tick()
    assert view.step_buttons["plate"].label == "2. Plate ✓"
    assert view.step_summaries["plate"].object == "96-well, registered"
    assert (view.step, view.expert) == ("scan", False)
    assert list(view.step_panel.objects) == view.screens["scan"]
    # Nor does the expert view go off when a run ends in it.
    view.show_expert(True)
    view.on_run("quick")
    assert page.join(10.0)
    view.tick()
    view.on_refresh()
    assert (view.step, view.expert, view.expert_toggle.value) == ("scan", True, True)
    assert list(view.step_panel.objects) == view.screens["expert"]


def test_the_labels_and_summaries_follow_the_run_document_after_a_refresh(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    assert [b.label for b in view.step_buttons.values()] == [
        "1. Stand ✓",
        "2. Plate",
        "3. Wells",
        "4. Scan",
        "5. Results",
    ]
    assert [view.step_summaries[s].object for s in STEPS] == [
        "demo",
        "next",
        "",
        "",
        "",
    ]
    # Another program writes the run document: Refresh reads it (the tick alone does not).
    _seed_plate(run_doc)
    view.on_refresh()
    assert [b.label for b in view.step_buttons.values()] == [
        "1. Stand ✓",
        "2. Plate ✓",
        "3. Wells",
        "4. Scan",
        "5. Results",
    ]
    assert [view.step_summaries[s].object for s in STEPS] == [
        "demo",
        "96-well, registered",
        "next",
        "",
        "",
    ]
    # The summary is also the button's tooltip, and an empty one is none.
    assert view.step_buttons["plate"].description == "96-well, registered"
    assert view.step_buttons["scan"].description is None
    # The screen did not move for it (D2): the page opened on Plate and stays there, though
    # Wells is the step that is suggested now.
    assert view.step == "plate"
    assert view.step_summaries["wells"].object == "next"


def test_the_step_bar_shows_the_loaded_preset_and_the_scan_plans_counts(
    page: PlatePage,
    run_doc: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The view hands `guide.step_states` the loaded preset's name and the scan plan the
    # view read; the guide's own tests prove the sentences, this proves the wiring (found
    # by the adversarial review of #443: a view that passed neither stayed green).
    monkeypatch.chdir(tmp_path)  # presets are looked for in ./profiles/presets
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    (folder / "weekly.toml").write_text(
        'stand = "demo"\nplate = "96-well"\n', encoding="utf-8"
    )
    _seed_plate(run_doc)
    _seed_scan_plan(run_doc)
    view = PlatePageView(page)
    assert view.step_summaries["wells"].object == "2 wells, 8 fields"
    assert view.step_buttons["wells"].label == "3. Wells ✓"
    # No preset yet: the stand alone.
    assert page.preset is None
    assert view.step_summaries["stand"].object == "demo"
    view.preset_select.value = "weekly"
    view.preset_load_button.clicks += 1
    assert page.preset is not None
    assert view.message.object == ""
    assert view.step_summaries["stand"].object == "demo, preset weekly"
    # The tooltip says the same, and forgetting the preset takes it out again.
    assert view.step_buttons["stand"].description == "demo, preset weekly"
    view.preset_select.value = ""
    view.preset_load_button.clicks += 1
    assert page.preset is None
    assert view.step_summaries["stand"].object == "demo"


def test_two_views_show_their_own_screens(page: PlatePage, run_doc: Path) -> None:
    first, second = PlatePageView(page), PlatePageView(page)
    first.show_step("results")
    second.show_expert(True)
    # Each tab has its own screen and its own switch (D6) ...
    assert (first.step, first.expert) == ("results", False)
    assert second.expert is True
    assert second.step == "plate"
    assert list(first.step_panel.objects) == first.screens["results"]
    assert list(second.step_panel.objects) == second.screens["expert"]
    assert first.expert_toggle.value is False
    assert second.expert_toggle.value is True
    # ... and what each draws is the controller's: a registration that the first tab
    # starts and that ends between two ticks of the second is drawn in both (FM-151).
    page.choose_plate(page.plate)
    first.on_run("plate-registration")
    assert page.join(10.0)
    first.tick()
    second.tick()
    labels = [[b.label for b in view.step_buttons.values()] for view in (first, second)]
    assert labels[0] == labels[1]
    assert labels[0][1] == "2. Plate ✓"
    assert (first.step, second.step) == ("results", "plate")
    assert (first.expert, second.expert) == (False, True)


# -- what ``/ui-check`` found at 1280 x 800 (#443) -----------------------------------


def test_the_strip_stays_on_top_and_the_step_panel_scrolls_on_its_own(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    # Measured in headless Chromium (the scroll probe of #443): the page scrolls 189 px at
    # 1280 x 800 to bring the drive bar up, and STOP stayed at y = 5. A strip that scrolled
    # away would take STOP out of view (ADR-0006: always offered). From B2 (D12) the sticky
    # one is the header, which holds the strip, the outcome lines and the step bar: the
    # owner's walk saw the strip hold while the step bar scrolled away (#431).
    header = view.layout[0]
    assert header is view.header
    assert isinstance(header, pn.Column)
    assert header.styles["position"] == "sticky"
    assert header.styles["top"] == "0"
    # An opaque ground, or what scrolls under it is read through it.
    assert header.styles["background-color"] == "white"
    held = {id(o) for o in header.select()}
    for widget in (
        view.stop_button,
        view.connection_text,
        view.run_text,
        view.message,
        view.problems_text,
        *view.step_buttons.values(),
        *view.step_summaries.values(),
        view.expert_toggle,
    ):
        assert id(widget) in held
    # The strip is not sticky on its own any more: two sticky parts would stack at the top.
    strip = next(
        row for row in header if isinstance(row, pn.Row) and view.stop_button in row
    )
    assert "position" not in strip.styles
    # The panel has a height of its own to scroll in (the window's less the frame's): a
    # `scroll=True` column with no height grows with its screen, and the page scrolls.
    assert view.step_panel.scroll is True
    assert view.step_panel.styles["max-height"] == plate_page._STEP_PANEL_MAX_HEIGHT


def test_the_maps_title_wraps_in_a_pane_and_its_ticks_are_plain_numbers(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    # A figure's own title is one line cut at the figure's width: at 440 px the caveat
    # ("estimated ..., not registered") was clipped. The same words, wrapped, above the map.
    assert view.map_title_text.object == view.figure.title.text != ""
    assert view.figure.title.visible is False
    assert view.map_title_text.styles["white-space"] == "pre-wrap"
    assert type(view.map_title_text) is pn.pane.Str
    # Set where the title is, so a redraw keeps the two the same.
    view.select_input.value = "A1:B2"
    view.refresh()
    assert view.map_title_text.object == view.figure.title.text
    # Plain labels: Bokeh picks `4.000e+4` as soon as a tick reaches 1e5 (FM-174 reads it).
    for axis in (view.figure.xaxis[0], view.figure.yaxis[0]):
        assert axis.formatter.use_scientific is False


# -- the outcome lines --------------------------------------------------------------


def test_an_outcome_line_takes_no_height_while_its_text_is_empty(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    for pane in (view.message, view.problems_text):
        assert pane.object == ""
        assert pane.visible is False
    # The one outcome line is the exception: it always takes its height (D15).
    assert view.run_text.object == ""
    assert view.run_text.visible is True
    # Each line wraps at the window's width: a run's outcome is a sentence of 200
    # characters (`plate-orientation done ...; assumed: ...`) and its end ran past the
    # right edge in `/ui-check`'s `turned` state. Plain text, never Markdown (FM-139).
    for pane in (
        view.connection_text,
        view.run_text,
        view.message,
        view.problems_text,
        view.status_text,
    ):
        assert type(pane) is pn.pane.Str
        assert pane.styles["white-space"] == "pre-wrap"
        assert pane.sizing_mode == "stretch_width"
    # The connection line says nothing until a connect or a disconnect goes or fails.
    assert view.connection_text.object == ""
    assert view.connection_text.visible is False
    view.message.object = "that did not work"
    assert view.message.visible is True
    view.message.object = ""
    assert view.message.visible is False
    # `drive_text` has its own rule: it shows while the Drive toggle is on.
    assert view.drive_text.visible is False
    view.drive_toggle.value = True
    assert view.drive_text.visible is True
    # Their order under the strip, inside the sticky header, and the step bar after them;
    # the centre is below the header (B2, D12).
    top = list(view.header)
    assert top[1:6] == [
        view.connection_text,
        view.drive_text,
        view.run_text,
        view.message,
        view.problems_text,
    ]
    assert view.layout[0] is view.header


# --- round 2: the map's frame, the density select, the header (m5 §17.12 D15, D16, #475) ---


def _frame(view: PlatePageView) -> tuple[float, float, float, float]:
    """The frame's box as ``(x0, x1, y0, y1)``: flat, so that ``pytest.approx`` reads it."""
    data = view.frame_source.data
    return (min(data["x"]), max(data["x"]), min(data["y"]), max(data["y"]))


def test_the_frame_renderer_is_the_only_one_the_ranges_follow(page: PlatePage) -> None:
    from bokeh.models import DataRange1d

    view = PlatePageView(page)
    figure = view.figure
    # Set once, at the build: changing a range's renderers after the page is drawn does not reframe
    # (measured on Panel 1.10.0 / Bokeh 3.10.2), so what moves is the frame source's data.
    for axis_range in (figure.x_range, figure.y_range):
        assert isinstance(axis_range, DataRange1d)
        assert axis_range.renderers == [view.frame_renderer]
    assert view.frame_renderer.data_source is view.frame_source
    # Invisible, and in no hover's list (FM-174: the hover names the wells only).
    assert view.frame_renderer.glyph.fill_alpha == 0.0
    assert view.frame_renderer.glyph.line_alpha == 0.0
    assert view.frame_renderer not in (view.hover.renderers or [])
    assert figure.match_aspect is True
    # No other source feeds the ranges: the side labels have no glyph of their own any more.
    assert not any(
        getattr(r, "data_source", None) is view.side_source for r in figure.renderers
    )
    # Redrawing the wells does not replace the list the ranges hold.
    view.refresh()
    view.show_step("wells")
    assert figure.x_range.renderers == [view.frame_renderer]


def _grown(wells: list[Any], pitch: float) -> tuple[float, float, float, float]:
    """The wells' box grown by a pitch each way, as ``(x0, x1, y0, y1)`` (m5 §17.12 D16)."""
    return (
        min(w.center_um[0] - w.size_um[0] / 2 for w in wells) - pitch,
        max(w.center_um[0] + w.size_um[0] / 2 for w in wells) + pitch,
        min(w.center_um[1] - w.size_um[1] / 2 for w in wells) - pitch,
        max(w.center_um[1] + w.size_um[1] / 2 for w in wells) + pitch,
    )


def test_the_map_frames_the_plate_then_the_plans_wells(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    view = PlatePageView(page)
    view.refresh()
    assert view._view is not None
    pitch = 9000.0
    wells = list(view._view.wells)
    # The Stand and Plate screens and the expert view: the plate, a pitch of margin all round.
    plate_frame = _grown(wells, pitch)
    assert _frame(view) == pytest.approx(plate_frame)
    for step in ("stand", "plate"):
        view.show_step(step)
        assert _frame(view) == pytest.approx(plate_frame)
    view.show_expert(True)
    assert _frame(view) == pytest.approx(plate_frame)

    # A plan stands (B2 and B11): the Wells, Scan and Results screens frame its wells.
    _seed_scan_plan(run_doc)
    view.refresh()
    framed = _grown([w for w in wells if w.well in ("B2", "B11")], pitch)
    assert framed != pytest.approx(plate_frame)
    for step in ("wells", "scan", "results"):
        view.show_step(step)
        assert _frame(view) == pytest.approx(framed), step
    # Back to the plate for the screens that are about the plate; the expert view frames it too.
    view.show_step("plate")
    assert _frame(view) == pytest.approx(plate_frame)
    view.show_step("wells")
    view.show_expert(True)
    assert _frame(view) == pytest.approx(plate_frame)


def test_the_map_frames_the_plate_when_the_plan_names_no_well_of_it(
    page: PlatePage, run_doc: Path
) -> None:
    # A plan whose wells are not on this plate: no frame of nothing.
    _seed_plate(run_doc)
    plan = _scan_plan_result().model_copy(update={"wells": ("Z99",)})
    write_stage(
        run_doc,
        StageRecord(
            plugin="scan-plan",
            provides="scan-plan",
            attempt="a1",
            status=StageStatus.done,
            started_utc=T0 + timedelta(minutes=1),
            finished_utc=T0 + timedelta(minutes=2),
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=plan.model_dump(mode="json"),
        ),
    )
    view = PlatePageView(page)
    view.refresh()
    plate_frame = _frame(view)
    view.show_step("wells")
    assert _frame(view) == plate_frame


def test_the_density_select_shows_only_with_a_class(
    page: PlatePage, run_doc: Path
) -> None:
    from smc.plugins.slots import DensityResult, DensityWell

    view = PlatePageView(page)
    # Empty and above the map from the Stand screen on, before: the owner's walk.
    assert view.label_select.options == []
    assert view.label_select.visible is False
    density = DensityResult(
        labels=("live",),
        area_grid_um=10.0,
        wells=(
            DensityWell(
                well="A1",
                fields=1,
                failed_fields=0,
                area_mm2=2.0,
                counts={"live": 3},
                outside_well=0,
                per_mm2={"live": 1.5},
            ),
        ),
    )
    write_stage(
        run_doc,
        StageRecord(
            plugin="density",
            provides="density",
            attempt="a2",
            status=StageStatus.done,
            started_utc=T0 + timedelta(minutes=2),
            finished_utc=T0 + timedelta(minutes=3),
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=density.model_dump(mode="json"),
        ),
    )
    view.refresh()
    assert view.label_select.options == ["live"]
    assert view.label_select.visible is True


def test_the_header_holds_one_outcome_line_of_fixed_height(page: PlatePage) -> None:
    view = PlatePageView(page)
    # The one line that always takes its height: empty or not (the page jumped by a line each
    # time an outcome came and went, finding 11).
    assert view.run_text.visible is True
    assert view.run_text.min_height == plate_page._RUN_LINE_PX
    view.tick()
    view.run_text.object = "a short outcome"
    assert view.run_text.visible is True
    view.run_text.object = ""
    assert view.run_text.visible is True
    # The others only with a text.
    for pane in (view.connection_text, view.message, view.problems_text):
        assert pane.visible is bool(pane.object)
    # Summaries are one line (at most 22 characters, cut with an ellipsis): the step bar is
    # one line tall.
    assert all(len(pane.object) <= 22 for pane in view.step_summaries.values())
