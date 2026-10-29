"""The plate page's Panel layer, without a browser (m3 §15, #197).

``figure._trigger_event(Tap(...))`` is what a tap raises, ``button.clicks += 1``
what a click raises, and ``source.data`` what the map draws: the page is tested
through the methods and the models the way a session drives them. What no test
here does is run a browser (the tap of a real tab, the look, the periodic
callback of a session): that is for the owner at the stand (#198).

Panel's ``PendingDeprecationWarning`` is an error here: a widget spelling that
warns on the version the lab installs (``name=``, ``button_type=``) fails the
test at the call, and is not filtered.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import socket
import string
import threading
import time
import urllib.request
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

import useq
from bokeh.events import Tap
from bokeh.models import Circle, Rect
from pymmcore_plus import CMMCorePlus

import smc.ui.plate_page as plate_page
from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource, XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.slots import DensityResult, DensityWell, PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import (
    PageServerError,
    PageStatus,
    PlatePage,
)
from smc.ui.plate_page import PlatePageView, build, serve

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

T0 = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
A1_OFF = (1000.0, -500.0)
TURN_DEG = 7.0
PITCH_96_UM = 9000.0

REAL = {
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
}


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class SlowScan:
    """Registered as ``scan`` (a name ``STAGES`` shows): runs until released or stopped."""

    name = provides = "scan"
    summary = "A test scan that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        SlowScan.started.set()
        while not SlowScan.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


def _entry_points(*names: str) -> list[EntryPoint]:
    found = [EntryPoint(name=n, value=REAL[n], group=GROUP) for n in names]
    found.append(EntryPoint(name="scan", value=f"{__name__}:SlowScan", group=GROUP))
    return found


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    SlowScan.started = threading.Event()
    SlowScan.release = threading.Event()
    try:
        yield
    finally:
        SlowScan.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = PlatePage(
        fake_microscope,
        run_doc,
        entry_points=_entry_points("goto-well", "plate-registration", "scan-plan"),
    )
    try:
        yield controller
    finally:
        controller.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _seed(
    run_doc: Path,
    plugin: str,
    provides: str,
    result: PluginResult,
    *,
    minute: int = 0,
) -> None:
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt=f"a{minute}",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _plate(plate: str = "96-well") -> PlateResult:
    return PlateResult(
        plate=plate,
        a1_center_xy_um=A1_OFF,
        rotation_deg=TURN_DEG,
        method="references",
        residual_um=2.0,
        scale_error=0.001,
        n_refs=2,
    )


def _centre(
    well: str, pitch_um: float = PITCH_96_UM, rotation_deg: float = TURN_DEG
) -> tuple[float, float]:
    """The forward model of m3 §2.1 for a plate with A1 at ``A1_OFF``, written out."""
    row = string.ascii_uppercase.index(well[0])
    col = int(well[1:]) - 1
    angle = math.radians(rotation_deg)
    dx, dy = col * pitch_um, -row * pitch_um
    return (
        A1_OFF[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_OFF[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _tap(view: PlatePageView, x: float, y: float) -> None:
    view.figure._trigger_event(Tap(view.figure, x=x, y=y, sx=0, sy=0))


def _wells_by_name(view: PlatePageView) -> dict[str, dict[str, Any]]:
    data = view.wells_source.data
    return {
        name: {key: data[key][i] for key in data} for i, name in enumerate(data["name"])
    }


def _status(**changes: Any) -> PageStatus:
    base: dict[str, Any] = {
        "dry_run": False,
        "running": None,
        "progress": None,
        "outcome": None,
        "stop_note": None,
        "xy_um": (1000.0, -500.0),
        "z_um": 12.5,
        "objective": "Nikon 10X S Fluor",
        "autofocus": "autofocus off",
        "light_on": None,
        "moving": (),
        "halted": False,
        "stop_refused": (),
        "problems": (),
    }
    return PageStatus(**{**base, **changes})


# --- building ---------------------------------------------------------------------


def test_the_page_builds(page: PlatePage) -> None:
    layout = build(page)
    assert layout.get_root() is not None


def test_the_page_builds_every_tool_the_registry_holds_in_workflow_order(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    # `goto-well` is no stage of the workflow: it follows under `More`.
    assert list(view.tool_runs) == [
        "plate-registration",
        "scan-plan",
        "scan",
        "goto-well",
    ]
    assert list(view.tool_fields["scan-plan"]) == [
        "objective",
        "overlap",
        "coverage",
        "serpentine",
        "rows",
        "columns",
        "clip_to_well",
        "pixel_size_um",
    ]
    assert "wells" not in view.tool_fields["scan-plan"]
    assert set(view.tool_fields["plate-registration"]) == {"max_rotation_deg"}


# --- the map ----------------------------------------------------------------------


def test_the_map_draws_wells_in_stage_axes_with_the_plates_rotation(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    wells = _wells_by_name(view)
    assert len(wells) == 96
    for name, drawn in wells.items():
        expected = _centre(name)
        assert (drawn["x"], drawn["y"]) == pytest.approx(expected, abs=1e-6)
        assert drawn["angle"] == pytest.approx(math.radians(TURN_DEG))
        assert (drawn["w"], drawn["h"]) == pytest.approx((6400.0, 6400.0))
        assert drawn["r"] == pytest.approx(3200.0)
    # Stage axes, Y up: row A is above row H.
    assert wells["A1"]["y"] > wells["H1"]["y"]
    assert wells["A12"]["x"] > wells["A1"]["x"]


def test_round_wells_are_circles_and_square_ones_rectangles(
    run_doc: Path, page: PlatePage
) -> None:
    view = PlatePageView(page)
    assert isinstance(view.wells_renderer.glyph, Circle)
    view.plate_select.value = "384-well"
    assert isinstance(view.wells_renderer.glyph, Rect)
    drawing = [r for r in view.figure.renderers if r.data_source is view.wells_source]
    assert len(drawing) == 1
    assert view.hover.renderers == [view.wells_renderer]
    view.plate_select.value = "96-well"
    assert isinstance(view.wells_renderer.glyph, Circle)
    assert (
        len([r for r in view.figure.renderers if r.data_source is view.wells_source])
        == 1
    )


def test_the_map_says_whether_the_plate_is_registered(
    run_doc: Path, page: PlatePage
) -> None:
    view = PlatePageView(page)
    assert view.figure.title.text == (
        "96-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )
    _seed(run_doc, "plate-registration", "plate", _plate())
    view.refresh()
    assert view.figure.title.text == "96-well: the registered plate"
    view.plate_select.value = "384-well"
    assert view.figure.title.text == (
        "384-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )


def test_names_are_drawn_only_up_to_96_wells(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.names.visible
    view.plate_select.value = "384-well"
    assert not view.names.visible
    view.plate_select.value = "24-well"
    assert view.names.visible


def test_a_tap_selects_the_well_under_it(run_doc: Path, page: PlatePage) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    _tap(view, *_centre("C5"))
    assert page.selection == ("C5",)
    wells = _wells_by_name(view)
    assert wells["C5"]["outline_width"] == 3
    assert {w["outline_width"] for n, w in wells.items() if n != "C5"} == {1}
    assert wells["C5"]["outline"] != wells["C4"]["outline"]
    _tap(view, *_centre("C5"))
    assert page.selection == ()
    assert {w["outline_width"] for w in _wells_by_name(view).values()} == {1}


def test_a_tap_on_no_well_does_nothing(page: PlatePage) -> None:
    view = PlatePageView(page)
    _tap(view, -50000.0, 50000.0)
    assert page.selection == ()
    assert view.message.object == ""


def test_a_typed_range_selects_and_draws(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.select_input.value = "A1:A3"
    assert page.selection == ("A1", "A2", "A3")
    wells = _wells_by_name(view)
    assert [n for n, w in wells.items() if w["outline_width"] == 3] == [
        "A1",
        "A2",
        "A3",
    ]


def test_a_tap_in_drive_mode_moves_once(
    run_doc: Path, page: PlatePage, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    stage = fake_microscope.require(XYStage)
    assert not view.drive_text.visible
    view.drive_toggle.value = True
    assert view.drive_text.visible
    assert view.drive_text.object == (
        "DRIVE: the next click on a well moves the stage there"
    )
    _tap(view, *_centre("C5"))
    assert page.join(10.0)
    assert stage.position_um().x_um == pytest.approx(_centre("C5")[0], abs=1e-6)
    assert stage.position_um().y_um == pytest.approx(_centre("C5")[1], abs=1e-6)
    assert not view.drive_toggle.value
    assert not view.drive_text.visible
    assert page.selection == ()
    moved = stage.position_um()
    attempt = RunDocument.load(run_doc).stages["goto-well"].attempt
    # The second tap selects: the toggle went back to select mode, so nothing moves.
    _tap(view, *_centre("A1"))
    assert page.selection == ("A1",)
    assert page.status().running is None
    assert stage.position_um() == moved
    assert RunDocument.load(run_doc).stages["goto-well"].attempt == attempt


def test_a_drive_tap_that_is_refused_says_why_and_disarms_the_toggle(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    view.drive_toggle.value = True
    # No plate is registered: `goto-well` itself refuses, and the refusal is the outcome.
    _tap(view, 0.0, 0.0)
    assert page.join(10.0)
    assert page.status().outcome == (
        "goto-well: UpstreamError: goto-well reads plate, but the run document has "
        "no plate; smc run --list shows which plugin provides it"
    )
    assert stage.position_um() == before
    assert not view.drive_toggle.value


def test_a_drive_tap_on_another_plates_map_is_refused_before_anything_moves(
    run_doc: Path, page: PlatePage, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate("384-well"))
    view = PlatePageView(page)
    page.choose_plate("96-well")
    view.refresh()
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    view.drive_toggle.value = True
    _tap(view, 36000.0, -18000.0)
    assert view.message.object == (
        "the registered plate is a 384-well; calibrate the 96-well plate again, "
        "or choose 384-well"
    )
    assert page.status().running is None
    assert stage.position_um() == before
    assert not view.drive_toggle.value
    assert "goto-well" not in RunDocument.load(run_doc).stages


def test_density_colours_the_wells_and_unknown_is_grey(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    _seed(
        run_doc,
        "density",
        "density",
        DensityResult(
            labels=("live", "dead"),
            area_grid_um=10.0,
            wells=tuple(
                DensityWell(
                    well=well,
                    fields=1,
                    failed_fields=0,
                    area_mm2=2.0,
                    counts={"live": n, "dead": 1},
                    outside_well=0,
                    per_mm2={"live": n / 2.0, "dead": 0.5},
                )
                for well, n in (("A1", 2), ("A2", 8))
            ),
        ),
        minute=1,
    )
    view = PlatePageView(page)
    assert view.label_select.options == ["live", "dead"]
    assert view.label_select.value == "live"
    wells = _wells_by_name(view)
    assert (wells["A1"]["value"], wells["A2"]["value"]) == (1.0, 4.0)
    assert math.isnan(wells["B1"]["value"])
    assert view.mapper.low == 0.0
    assert view.mapper.high == 4.0
    assert view.mapper.nan_color == "#eeeeee"
    assert wells["A2"]["tip"] == "A2 — live: 8 (4.0/mm²), dead: 1 (0.5/mm²)"
    assert wells["B1"]["tip"] == "B1"
    view.label_select.value = "dead"
    wells = _wells_by_name(view)
    assert (wells["A1"]["value"], wells["A2"]["value"]) == (0.5, 0.5)
    assert view.mapper.high == 0.5


def test_a_selected_well_keeps_its_density_colour(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    page.select_text("A1")
    view.refresh()
    assert _wells_by_name(view)["A1"]["outline_width"] == 3


def test_the_route_is_drawn_in_order(run_doc: Path, page: PlatePage) -> None:
    from smc.plugins.slots import RouteResult

    _seed(
        run_doc,
        "route",
        "route",
        RouteResult.model_validate(
            {
                "objective": None,
                "matrix_um_per_px": [[1.0, 0.0], [0.0, 1.0]],
                "image_shape_px": [10, 10],
                "fov_um": [10.0, 10.0],
                "labels": [],
                "positions": [
                    {
                        "order": order,
                        "well": "A1",
                        "name": f"A1_{order}",
                        "x_um": x,
                        "y_um": y,
                        "z_um": None,
                        "pfs_offset": None,
                        "counts": {},
                    }
                    for order, x, y in ((2, 30.0, 3.0), (0, 10.0, 1.0), (1, 20.0, 2.0))
                ],
                "uncovered": {},
                "distance_um": 20.0,
                "travel_s": None,
            }
        ),
    )
    view = PlatePageView(page)
    assert view.route_source.data["x"] == [10.0, 20.0, 30.0]
    assert view.route_source.data["y"] == [1.0, 2.0, 3.0]
    assert view.route_source.data["order"] == [0, 1, 2]


def test_the_stage_is_a_cross_at_its_position(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.require(XYStage).move_to_um(1234.0, -567.0)
    view = PlatePageView(page)
    assert view.stage_source.data["x"] == [1234.0]
    assert view.stage_source.data["y"] == [-567.0]


# --- the strip, STOP and the tools ---------------------------------------------------


def test_a_position_that_rounds_to_zero_is_printed_without_a_sign() -> None:
    # The demo stage reads back -0.0 at its origin; the CLI prints it as 0.00 too.
    line = plate_page.status_line(_status(xy_um=(-0.0, -0.04), z_um=-0.0))
    assert line.startswith("XY (0.0, 0.0) µm | Z 0.00 µm | ")
    line = plate_page.status_line(_status(xy_um=(-0.06, 0.0), z_um=-0.001))
    assert line.startswith("XY (-0.1, 0.0) µm | Z 0.00 µm | ")


def test_the_stand_is_one_line_and_unknown_is_a_question_mark() -> None:
    assert plate_page.status_line(_status()) == (
        "XY (1000.0, -500.0) µm | Z 12.50 µm | Nikon 10X S Fluor | autofocus off | light ?"
    )
    assert plate_page.status_line(
        _status(
            xy_um=None,
            z_um=None,
            objective=None,
            light_on=True,
            dry_run=True,
            moving=("xy_stage",),
            halted=True,
            stop_refused=("xy_stage", "focus"),
        )
    ) == (
        "XY ? | Z ? | objective ? | autofocus off | light on | DRY RUN | "
        "moving: xy_stage | HALTED | STOP is refused by xy_stage, focus on this "
        "stand: a move under way runs to its target"
    )


def test_the_run_is_its_progress_its_outcome_and_the_last_stop() -> None:
    from smc.plugins.api import Progress

    assert plate_page.run_text(_status()) == ""
    assert plate_page.run_text(
        _status(
            running="scan",
            progress=Progress("scan", 3, 12, "well B2"),
            outcome="cells done in 2.0 s",
            stop_note="scan told to stop; stopped: the stand is halted (Resume clears the halt)",
        )
    ) == (
        "scan: 3/12 well B2\n"
        "cells done in 2.0 s\n"
        "scan told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert plate_page.run_text(_status(running="scan")) == "scan is running"
    assert (
        plate_page.run_text(
            _status(running="scan", progress=Progress("scan", 4, None, ""))
        )
        == "scan: 4"
    )


def test_stop_is_always_offered_and_resume_only_when_halted(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert not view.stop_button.disabled
    assert not view.resume_button.visible
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.tick()
    assert not view.stop_button.disabled
    view.on_stop()
    assert page.join(10.0)
    view.tick()
    assert view.resume_button.visible
    assert not view.stop_button.disabled
    assert view.run_text.object == (
        "scan stopped\n"
        "scan told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert "HALTED" in view.status_text.object
    view.on_resume()
    assert not view.resume_button.visible
    assert view.run_text.object == "scan stopped"


def test_stop_button_stops_through_a_click(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.stop_button.clicks += 1
    assert page.status().halted
    assert view.resume_button.visible
    view.resume_button.clicks += 1
    assert not page.status().halted
    assert not view.resume_button.visible


def test_a_blocked_tool_is_disabled_with_its_reason(page: PlatePage) -> None:
    view = PlatePageView(page)
    reason = (
        "the run document has no plate; smc run --list shows which plugin provides it"
    )
    assert view.tool_runs["scan-plan"].disabled
    assert view.tool_notes["scan-plan"].object == reason
    assert not view.tool_runs["plate-registration"].disabled
    assert view.tool_notes["plate-registration"].object == ""
    assert not view.tool_runs["scan"].disabled


def test_a_running_tool_disables_every_run_button_until_it_ends(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.on_run("scan")
    assert SlowScan.started.wait(5)
    assert all(button.disabled for button in view.tool_runs.values())
    assert view.run_text.object == "scan is running"
    SlowScan.release.set()
    assert page.join(10.0)
    view.tick()
    # The end of a run redraws the view: the run buttons are the view's blocked set again.
    assert not view.tool_runs["plate-registration"].disabled
    assert not view.tool_runs["scan"].disabled
    assert view.tool_runs["scan-plan"].disabled
    assert view.run_text.object.startswith("scan done in ")


def test_a_run_that_ends_redraws_the_view(run_doc: Path, page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.label_select.options == []
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.tick()
    _seed(
        run_doc,
        "density",
        "density",
        DensityResult(labels=("live",), area_grid_um=10.0, wells=()),
        minute=1,
    )
    assert (
        view.label_select.options == []
    )  # not read yet: the tick never reads the view
    SlowScan.release.set()
    assert page.join(10.0)
    view.tick()
    assert view.label_select.options == ["live"]


def test_an_empty_field_is_not_sent(page: PlatePage) -> None:
    view = PlatePageView(page)
    page.select_text("A1")
    sent: list[tuple[str, list[str]]] = []

    def record(plugin: str, params: list[str]) -> None:
        sent.append((plugin, list(params)))

    view.tool_fields["scan-plan"]["overlap"].value = "0.2"
    view.tool_fields["scan-plan"]["coverage"].value = "  "
    with _replaced(page, "run", record):
        view.on_run("scan-plan")
    assert sent == [("scan-plan", [*page.page_params("scan-plan"), "overlap=0.2"])]
    assert sent == [("scan-plan", ["wells=A1", "overlap=0.2"])]


def test_the_page_shows_a_message_and_keeps_no_exception(page: PlatePage) -> None:
    view = PlatePageView(page)
    # A run that is refused for the operator's mistake is a message, not an error.
    view.on_run("scan-plan")
    assert view.message.object == "select at least one well first"
    page.select_text("A1")
    view.on_select_text("Z99")
    assert view.message.object == "wells not understood on a 96-well plate: 'Z99'"
    assert page.selection == ("A1",)
    view.on_select_text("A2")
    assert view.message.object == ""
    assert page.selection == ("A2",)


def test_a_handler_keeps_no_error_it_showed(page: PlatePage) -> None:
    view = PlatePageView(page)
    refs: list[weakref.ref[BaseException]] = []

    def refuse(text: str) -> None:
        raise _track(_RefusedError("refused"), refs)

    gc.disable()
    try:
        with _replaced(page, "select_text", refuse):
            view.on_select_text("A1")
        assert view.message.object == "refused"
        assert len(refs) == 1
        assert (
            refs[0]() is None
        )  # dead without the cyclic collector: nothing holds it (FM-98)
    finally:
        gc.enable()


class _RefusedError(ValueError):
    """A subclass: a builtin exception cannot be weakly referenced, its subclass can."""


def _track(
    error: _RefusedError, refs: list[weakref.ref[BaseException]]
) -> _RefusedError:
    refs.append(weakref.ref(error))
    return error


def test_capture_buttons_act_on_the_current_well(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    view.on_capture("centre")
    assert view.message.object == "click a well first"
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, 0.0, 0.0)
    assert page.current == "A1"
    view.capture_centre_button.clicks += 1
    view.capture_side_select.value = "top"
    view.capture_wall_button.clicks += 1
    assert [(r.well, r.x_um, r.y_um) for r in page.refs] == [("A1", 10.0, 20.0)]
    assert [(w.well, w.side) for w in page.walls] == [("A1", "top")]
    assert (
        view.captures_text.object == "A1 centre (10.0, 20.0)\nA1 top wall (10.0, 20.0)"
    )
    view.forget_button.clicks += 1
    assert (page.refs, page.walls) == ((), ())
    assert view.captures_text.object == "no capture yet"
    view.on_capture(
        "nonsense"
    )  # not a button: the handler still refuses it, by message
    assert (
        view.message.object == "cannot capture 'nonsense'; capture a centre or a wall"
    )


def test_the_checklist_says_what_each_slot_is(run_doc: Path, page: PlatePage) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    assert view.checklist_text.object == "\n".join(
        [
            "· camera-calibration",
            "✓ plate",
            "· timings",
            "· scan-plan",
            "· scan",
            "· cells",
            "· density",
            "· route",
            "· report",
            "· nis-export",
        ]
    )


def test_tick_does_not_rebuild_the_view(page: PlatePage) -> None:
    view = PlatePageView(page)
    counts = {"view": 0, "status": 0}
    real_view, real_status = page.view, page.status

    def counting_view() -> Any:
        counts["view"] += 1
        return real_view()

    def counting_status() -> Any:
        counts["status"] += 1
        return real_status()

    with (
        _replaced(page, "view", counting_view),
        _replaced(page, "status", counting_status),
    ):
        for _ in range(10):
            view.tick()
    assert counts == {"view": 0, "status": 10}


def test_a_problem_is_shown_above_the_map(run_doc: Path, page: PlatePage) -> None:
    run_doc.write_bytes(b"not json")
    view = PlatePageView(page)
    assert view.problems_text.object == (
        f"run document: {run_doc}: is not valid JSON: Expecting value (line 1, "
        f"column 1); it was left as it is, restore it from a copy"
    )


def test_a_closed_or_reloaded_tab_does_not_stop_a_run(page: PlatePage) -> None:
    first = PlatePageView(page)
    first.on_run("scan")
    assert SlowScan.started.wait(5)
    PlatePageView(page)  # a second tab, built and dropped
    del first  # and the first one closed
    gc.collect()
    status = page.status()
    assert status.running == "scan"
    assert not status.halted
    assert RunDocument.load(page._run_doc).stages["scan"].status is StageStatus.running
    SlowScan.release.set()
    assert page.join(10.0)
    assert RunDocument.load(page._run_doc).stages["scan"].status is StageStatus.done


def test_the_plate_choice_redraws_the_map(page: PlatePage) -> None:
    view = PlatePageView(page)
    page.select_text("A1")
    view.plate_select.value = "24-well"
    assert page.plate == "24-well"
    assert page.selection == ()
    assert len(_wells_by_name(view)) == len(
        useq.WellPlatePlan(
            plate=useq.WellPlate.from_str("24-well"), a1_center_xy=(0.0, 0.0)
        ).all_well_names.ravel()
    )
    assert view.plate_select.value == "24-well"


# --- serve ----------------------------------------------------------------------------


class _Thread:
    """What ``pn.serve(..., threaded=True)`` returns, as far as ``serve`` uses it."""

    def __init__(
        self,
        *,
        alive: bool,
        interrupt: bool = False,
        events: list[str] | None = None,
        stop_waits_for: threading.Event | None = None,
    ) -> None:
        self.alive = alive
        self.interrupt = interrupt
        self.events = events if events is not None else []
        # Panel's `stop` ends with a join that has no limit: this one waits for the test.
        self.stop_waits_for = stop_waits_for
        self.calls: list[tuple[str, float | None]] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))
        if self.interrupt and timeout == 0.1:
            self.interrupt = False
            raise KeyboardInterrupt  # raised from the stub, never a signal (FM-44)

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self.events.append("server stop")
        if self.stop_waits_for is not None:
            self.stop_waits_for.wait(30)
        self.alive = False


@pytest.fixture
def served_thread(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., list[str]]]:
    """Put a ``_Thread`` where ``pn.serve`` would; the list is what happened, in order.

    The page's own ``close`` and the console's ``_say`` are recorded in it, so a test
    reads the order of the stop of the run and the stop of the server (FM-94: the
    assertion is after, never inside a callback).
    """
    released: list[threading.Event] = []

    def install(
        thread: _Thread, *, say: Callable[[str], None] | None = None
    ) -> list[str]:
        events = thread.events

        def close() -> None:
            events.append("page close")

        monkeypatch.setattr(pn, "serve", lambda apps, **kwargs: thread)
        monkeypatch.setattr(
            "smc.cli._say", say or (lambda text: events.append(f"say: {text}"))
        )
        setattr(page, "close", close)  # noqa: B010  (an instance attribute, removed below)
        if thread.stop_waits_for is not None:
            released.append(thread.stop_waits_for)
        return events

    try:
        yield install
    finally:
        for event in released:
            event.set()  # a stopper thread a failed test left behind ends
        if "close" in vars(page):
            delattr(page, "close")


def test_serve_returns_on_ctrl_c_and_stops_the_server(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    thread = _Thread(alive=True, interrupt=True, events=events)
    served: list[dict[str, Any]] = []
    said: list[str] = []
    bokeh_levels: list[int] = []
    bokeh_log = logging.getLogger("bokeh")

    def fake_serve(apps: dict[str, Any], **kwargs: Any) -> _Thread:
        served.append({"apps": list(apps), **kwargs})
        bokeh_levels.append(
            bokeh_log.level
        )  # asserted after: the call is not ours (FM-94)
        return thread

    monkeypatch.setattr(pn, "serve", fake_serve)
    monkeypatch.setattr("smc.cli._say", said.append)
    before = bokeh_log.level
    with _replaced(page, "close", lambda: events.append("page close")):
        serve(page, port=5123)
    # The run is told to stop before the server is touched (it can wait on a busy
    # tick, measured at twice the tick), and the server is then stopped and waited for.
    assert events == ["page close", "server stop"]
    assert thread.calls[0] == ("join", 0.1)
    assert ("stop", None) in thread.calls
    assert served == [
        {
            "apps": [""],
            "address": "localhost",
            "port": 5123,
            "show": False,
            "threaded": True,
            "verbose": False,
        }
    ]
    assert said == [
        "serving the plate page on http://localhost:5123/ — Ctrl-C stops it"
    ]
    # Bokeh is quiet while the page serves (its console would block the loop that
    # handles STOP), and its level is put back.
    assert bokeh_levels == [logging.WARNING]
    assert bokeh_log.level == before


def test_serve_says_when_the_server_stopped_on_its_own(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pn, "serve", lambda apps, **kwargs: _Thread(alive=False))
    monkeypatch.setattr("smc.cli._say", lambda text: None)
    before = logging.getLogger("bokeh").level
    with pytest.raises(PageServerError) as stopped:
        serve(page, port=5006)
    assert str(stopped.value) == (
        "the plate page's server stopped on its own (is port 5006 taken?)"
    )
    assert logging.getLogger("bokeh").level == before


_ANNOUNCED = "say: serving the plate page on http://localhost:5006/ — Ctrl-C stops it"


def test_a_server_that_will_not_stop_is_waited_for_a_while_and_said(
    page: PlatePage,
    served_thread: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Panel's `stop` ends with a join that has no limit, and the server waits for the
    # callback in flight: asked directly it would hold the console for as long as a slow
    # device read. The stub's `stop` waits for the test, as that join would.
    release = threading.Event()
    thread = _Thread(alive=True, interrupt=True, stop_waits_for=release)
    events = served_thread(thread)
    monkeypatch.setattr(plate_page, "SERVER_STOP_WAIT_S", 1.0)
    started = time.monotonic()
    serve(page, port=5006)
    waited = time.monotonic() - started
    release.set()
    assert events == [
        _ANNOUNCED,
        "page close",
        "server stop",
        "say: the plate page's server did not stop within 1 s",
    ]
    assert waited < 10  # not the 30 s that the stub's stop would take


def test_a_ctrl_c_before_the_server_is_announced_still_stops_the_run_and_the_server(
    page: PlatePage, served_thread: Callable[..., list[str]]
) -> None:
    def say(text: str) -> None:
        raise KeyboardInterrupt  # from the stub, never a signal (FM-44)

    thread = _Thread(alive=True)
    events = served_thread(thread, say=say)
    serve(page, port=5006)
    assert events == ["page close", "server stop"]


def test_a_console_that_fails_still_stops_the_server(
    page: PlatePage, served_thread: Callable[..., list[str]]
) -> None:
    def say(text: str) -> None:
        raise BrokenPipeError("the console is closed")

    thread = _Thread(alive=True)
    events = served_thread(thread, say=say)
    with pytest.raises(BrokenPipeError):
        serve(page, port=5006)
    # Panel's thread is not a daemon: left running it would keep the process alive.
    assert events == ["server stop"]


def test_a_second_ctrl_c_while_the_run_is_stopped_still_stops_the_server(
    page: PlatePage, served_thread: Callable[..., list[str]]
) -> None:
    thread = _Thread(alive=True, interrupt=True)
    events = served_thread(thread)

    def close() -> None:
        events.append("page close")
        raise KeyboardInterrupt  # the operator's second Ctrl-C (FM-71)

    with _replaced(page, "close", close), pytest.raises(KeyboardInterrupt):
        serve(page, port=5006)
    assert events == [_ANNOUNCED, "page close", "server stop"]


def test_the_served_page_answers(page: PlatePage) -> None:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    thread = pn.serve(
        {"": lambda: build(page)},
        address="localhost",
        port=port,
        show=False,
        threaded=True,
    )
    try:
        deadline = time.monotonic() + 20
        status = 0
        body = b""
        while time.monotonic() < deadline:
            try:
                # No proxy: a PC's proxy settings (the environment, and Windows' Internet
                # Options) would send a request for localhost to the proxy.
                with direct.open(f"http://localhost:{port}/", timeout=20) as reply:
                    status, body = reply.status, reply.read()
                break
            except OSError:
                time.sleep(0.2)
        assert status == 200
        assert b"bokeh" in body.lower()
    finally:
        thread.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()


# --- the widgets are wired to what they say -----------------------------------------


def test_a_drive_tap_goes_to_the_wall_the_side_select_says(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    view.side_select.value = "left"
    view.drive_toggle.value = True
    _tap(view, *_centre("C5"))
    assert page.join(10.0)
    record = RunDocument.load(run_doc).stages["goto-well"]
    assert record.params == {"well": "C5", "side": "left"}
    assert record.status is StageStatus.done


def test_each_run_button_starts_its_own_tool(run_doc: Path, page: PlatePage) -> None:
    view = PlatePageView(page)
    page.select_text("A1")
    sent: list[tuple[str, list[str]]] = []

    def record(plugin: str, params: list[str]) -> None:
        sent.append((plugin, list(params)))

    with _replaced(page, "run", record):
        for name in ("plate-registration", "scan-plan", "scan"):
            sent.clear()
            view.tool_runs[
                name
            ].disabled = False  # a click is what is tested, not `blocked`
            view.tool_runs[name].clicks += 1
            assert [plugin for plugin, _ in sent] == [name]


def test_the_refresh_button_redraws_the_view_after_the_run_document_changed(
    run_doc: Path, page: PlatePage
) -> None:
    view = PlatePageView(page)
    unregistered = (
        "96-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )
    assert view.figure.title.text == unregistered
    _seed(run_doc, "plate-registration", "plate", _plate())
    assert view.figure.title.text == unregistered  # a tick never reads the view
    view.refresh_button.clicks += 1
    assert view.figure.title.text == "96-well: the registered plate"
    assert "✓ plate" in view.checklist_text.object


def test_build_registers_one_callback_that_ticks_the_view_every_second(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: list[tuple[Any, Any]] = []

    def recorder(callback: Any, period: Any = 500, *args: Any, **kwargs: Any) -> None:
        registered.append((callback, period))

    monkeypatch.setattr(pn.state, "add_periodic_callback", recorder)
    build(page)
    [(callback, period)] = registered
    assert isinstance(callback.__self__, PlatePageView)
    assert callback.__func__ is PlatePageView.tick
    # One second, and no faster: a scan must not be slowed by a status poll.
    assert period == 1000


def test_a_device_error_reaches_the_problems_line_after_a_tick(
    page: PlatePage, fake_core: FakeCore
) -> None:
    def refuses(*args: object, **kwargs: object) -> list[float]:
        raise RuntimeError("the stage does not answer")

    with _replaced(fake_core, "getXYPosition", refuses):
        view = PlatePageView(page)
        view.tick()
        problems = page.status().problems
    assert problems == (
        "xy: HardwareError: xy_stage: getXPosition('XY') failed (the stage does not "
        "answer); check the device: power, cable, another program holding it",
    )
    assert view.problems_text.object == "\n".join(problems)
    assert view.status_text.object.startswith("XY ? | ")


def test_the_light_that_is_on_is_said_on(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    class LampOn:
        def is_on(self) -> bool:
            return True

    real_get = fake_microscope.get

    def get(capability: object) -> object:
        return LampOn() if capability is LightSource else real_get(capability)

    with _replaced(fake_microscope, "get", get):
        view = PlatePageView(page)
        status = page.status()
    assert status.light_on is True
    assert "light on" in view.status_text.object


def test_a_stage_that_is_moving_is_listed(
    page: PlatePage, fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    label = fake_core.getXYStageDevice()
    fake_core.busy_devices.add(label)
    stage = fake_microscope.require(XYStage)
    mover = threading.Thread(
        target=lambda: stage.move_to_um(5.0, 5.0), daemon=True, name="smc-test-move"
    )
    mover.start()
    seen: tuple[str, ...] = ()
    try:
        deadline = time.monotonic() + 5.0
        while not seen and time.monotonic() < deadline:
            seen = page.status().moving
            time.sleep(0.01)
        assert seen == ("xy_stage XY",)
        view = PlatePageView(page)
        assert "moving: xy_stage XY" in view.status_text.object
    finally:
        fake_core.busy_devices.discard(label)
        mover.join(10.0)
    assert not mover.is_alive()


def test_a_dry_run_is_said_on_the_strip(run_doc: Path, fake_core: FakeCore) -> None:
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    controller = PlatePage(dry, run_doc, entry_points=_entry_points("goto-well"))
    try:
        assert controller.status().dry_run
        assert "DRY RUN" in PlatePageView(controller).status_text.object
    finally:
        controller.close()
        dry.close()


def test_a_selection_made_in_another_tab_is_drawn_by_the_next_tick(
    page: PlatePage,
) -> None:
    # The controller is the server's: a tab that does not redraw would run
    # `scan-plan` with wells it does not show.
    first, second = PlatePageView(page), PlatePageView(page)
    _tap(first, *_centre("A1"))
    assert page.selection == ("A1",)
    assert _wells_by_name(second)["A1"]["outline_width"] == 1
    second.tick()
    assert _wells_by_name(second)["A1"]["outline_width"] == 3
    page.choose_plate("24-well")
    second.tick()
    assert second.plate_select.value == "24-well"
    assert len(_wells_by_name(second)) == 24


def test_a_tick_redraws_the_view_only_when_something_changed(page: PlatePage) -> None:
    view = PlatePageView(page)
    counted = {"view": 0}
    real_view = page.view

    def counting_view() -> Any:
        counted["view"] += 1
        return real_view()

    with _replaced(page, "view", counting_view):
        view.tick()
        view.tick()
        assert counted["view"] == 0
        page.select_text("A1")
        view.tick()
        assert counted["view"] == 1
        view.tick()
        assert counted["view"] == 1
