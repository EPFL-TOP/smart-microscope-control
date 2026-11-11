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
from smc.plugins.slots import PlateResult
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
        title, lead, next_region, *_rest = shown
        assert title.object == SCREEN_TITLES[step]
        assert title.styles["font-weight"] == "bold"
        assert lead.object == SCREEN_LEADS[step]
        assert next_region in _next_regions(shown)
        assert view.continue_button in shown
        # The step shown is the primary button, and the one `/ui-check` finds by its class.
        colours = {s: b.color for s, b in view.step_buttons.items()}
        assert colours == {s: ("primary" if s == step else "light") for s in STEPS}
        classes = {s: b.css_classes for s, b in view.step_buttons.items()}
        assert classes == {
            s: (["smc-step", "smc-step-shown"] if s == step else ["smc-step"])
            for s in STEPS
        }
        # Never another step's tool: only the screen's own sections are listed.
        assert view.continue_button.label == (continue_label(step) or "")
        assert view.continue_button.visible is (step != "results")
    # Each screen's provisional contents are today's regions (B1), not another's.
    on = {step: {id(o) for o in _ids_objects(view.screens[step])} for step in STEPS}
    run = {name: id(button) for name, button in view.tool_runs.items()}
    assert run["scan-plan"] in on["wells"]
    assert run["scan"] in on["scan"]
    assert run["plate-registration"] in on["plate"]
    assert run["camera-calibration"] in on["stand"]
    assert run["timing"] in on["stand"]
    for name in ("cells", "density", "route", "report", "nis-export"):
        assert run[name] in on["results"]
    # A tool of no step (a third party's, `goto-well`) is in the expert view only.
    assert all(run["goto-well"] not in ids for ids in on.values())
    assert run["goto-well"] in {id(o) for o in _ids_objects(view.screens["expert"])}
    assert view.select_input in view.screens["wells"]
    assert view.select_input not in view.screens["plate"]


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
    # Both are 460 px high (m5 §17.3), side by side in the centre.
    assert view.figure.height == view.image_figure.height == plate_page.FIGURE_HEIGHT_PX


def test_the_strip_and_stop_are_outside_the_step_panel(page: PlatePage) -> None:
    view = PlatePageView(page)
    in_the_panel = _ids_below(list(view.step_panel.objects))
    always = {id(o) for o in view.layout.select()} - in_the_panel
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


def test_the_next_step_is_on_every_provisional_screen_and_drawn_once(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    seen: set[int] = set()
    for name, items in view.screens.items():
        regions = _next_regions(items)
        assert len(regions) == 1, f"{name} lists the next step {len(regions)} times"
        seen.add(id(regions[0]))
        assert items.index(regions[0]) == (2 if name != "expert" else 1)
    # One instance for all of them, so that it is drawn once and its place never depends
    # on the state: refresh() never rewrites a screen's list.
    assert len(seen) == 1
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
        assert sum(o is view.next_text for o in rendered) == 1
        assert sum(o is view.next_button for o in rendered) == 1


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


# -- the outcome lines --------------------------------------------------------------


def test_an_outcome_line_takes_no_height_while_its_text_is_empty(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    for pane in (view.message, view.problems_text, view.run_text):
        assert pane.object == ""
        assert pane.visible is False
    # The connection line is never empty once the tick has read the stand.
    assert view.connection_text.object != ""
    assert view.connection_text.visible is True
    view.message.object = "that did not work"
    assert view.message.visible is True
    view.message.object = ""
    assert view.message.visible is False
    # `drive_text` has its own rule: it shows while the Drive toggle is on.
    assert view.drive_text.visible is False
    view.drive_toggle.value = True
    assert view.drive_text.visible is True
    # Their order under the strip, and the step bar then the centre below them.
    top = list(view.layout)
    assert top[1:6] == [
        view.connection_text,
        view.drive_text,
        view.run_text,
        view.message,
        view.problems_text,
    ]
