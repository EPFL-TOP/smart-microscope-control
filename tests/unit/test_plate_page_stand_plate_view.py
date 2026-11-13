"""The Stand and Plate screens of the guided page, without a browser (m5 §17.4.1, §17.4.2, #444).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1``. A
click on a *disabled* button still calls its handler on the server (measured on Panel 1.10.0
and 1.9.4), so the controller's refusals are the guard (FM-169) and a test clicks a disabled
button to prove it. What this cannot show is a browser: the layout at 1280 x 620 and that
the title and the first action are inside the page are for ``/ui-check`` (FM-161). Panel's
warnings are errors (FM-147), and a handler that failed fails the test that did not expect it.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, ObjectiveOffset, ObjectiveTurret, XYStage
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.guide import (
    CAPTURE_WELLS,
    GOTO_ESTIMATED,
    GOTO_NOT_DURING_LIVE,
    GOTO_UNPLACED,
    OTHER_WAYS_TO_REGISTER,
    PLATE_NOT_CHOSEN,
    SCREEN_LEADS,
    NextStep,
)
from smc.ui.plate_page import _WARNING, PlatePageView, next_step_text

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
TEN_X = "Nikon 10X S Fluor"
FORTY_X = "Nikon 40X Plan Fluor ELWD"
PITCH_UM = 9000.0

REAL = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
}


class GotoParams(PluginParams):
    well: str
    side: str = "centre"
    plate: str | None = None
    orientation: str | None = None


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class StubGoto:
    """A ``goto-well`` that records the well and moves nothing."""

    name = provides = "goto-well"
    summary = "A test goto-well."
    Params = GotoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ("plate",)
    ran: ClassVar[list[str]] = []

    def run(self, microscope: Any, params: GotoParams, ctx: RunContext) -> NoResult:
        StubGoto.ran.append(params.well)
        return NoResult()


class FailingGoto(StubGoto):
    """A ``goto-well`` whose run ends ``failed``: the stage did not arrive."""

    def run(self, microscope: Any, params: GotoParams, ctx: RunContext) -> NoResult:
        raise PluginRunError("goto-well", "the stage did not arrive")


class BlockingGoto(StubGoto):
    """A ``goto-well`` that is under way until released: the drive of a row that is in flight."""

    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: GotoParams, ctx: RunContext) -> NoResult:
        StubGoto.ran.append(params.well)
        BlockingGoto.started.set()
        while not BlockingGoto.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


class Slow:
    """A tool that runs until released: an activity that is going."""

    name = provides = "slow"
    summary = "A test tool that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        Slow.started.set()
        while not Slow.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


class LiveCamera:
    """The six methods of ``Camera``, giving one 6 x 10 frame at once: enough for a Live."""

    def snap(self) -> np.ndarray:
        return np.arange(60, dtype=np.uint16).reshape(6, 10)

    def exposure_ms(self) -> float:
        return 5.0

    def set_exposure_ms(self, value_ms: float) -> float:
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (6, 10)

    def bit_depth(self) -> int:
        return 16

    def pixel_size_um(self) -> float:
        return 0.65


def _entry_points(*extra: type) -> list[EntryPoint]:
    found = [
        EntryPoint(name=name, value=value, group=GROUP)
        for name, value in REAL.items()
        if name not in {cls.name for cls in extra}
    ]
    for cls in extra:
        found.append(
            EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)
        )
    return found


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    StubGoto.ran = []
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    BlockingGoto.started = threading.Event()
    BlockingGoto.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()
        BlockingGoto.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_page(
    fake_microscope: Microscope, run_doc: Path
) -> Iterator[Callable[..., PlatePage]]:
    """A page over the fake stand with the real calibration and registration, and the given tools."""
    pages: list[PlatePage] = []

    def make(*extra: type) -> PlatePage:
        page = PlatePage(fake_microscope, run_doc, entry_points=_entry_points(*extra))
        pages.append(page)
        return page

    try:
        yield make
    finally:
        Slow.release.set()
        for page in pages:
            page.close()


@pytest.fixture
def page(make_page: Callable[..., PlatePage]) -> PlatePage:
    return make_page(StubGoto, Slow)


@pytest.fixture
def holder_page(fake_core: FakeCore, run_doc: Path) -> Iterator[PlatePage]:
    """A page on a stand whose profile says where its holder puts a plate (m3 §5.5, m5 §15).

    The map is *estimated* until a plate is registered, and a *Go to* drives to the holder's
    estimate. The tools are a ``goto-well`` that is under way until released, and ``Slow``.
    """
    profile = Profile.demo().model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=(400.0, -325.0), measured="a test's own centre"
            )
        }
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), profile)
    controller = PlatePage(
        microscope, run_doc, entry_points=_entry_points(BlockingGoto, Slow)
    )
    try:
        yield controller
    finally:
        BlockingGoto.release.set()
        Slow.release.set()
        controller.close()
        microscope.close()


def _click(button: Any) -> None:
    button.clicks += 1


def _objects(items: list[Any]) -> list[Any]:
    """Every object under ``items``, the Accordion's panels too."""
    found: list[Any] = []
    for item in items:
        found.extend(item.select())
        for accordion in item.select(pn.Accordion):
            for panel in accordion.objects:
                found.extend(panel.select())
    return found


def _well_centre(well: str) -> tuple[float, float]:
    """The stage position of a well of a 96-well plate whose A1 is at the origin, not turned."""
    row, column = ord(well[0]) - ord("A"), int(well[1:]) - 1
    return column * PITCH_UM, -row * PITCH_UM


def _capture(page: PlatePage, fake_microscope: Microscope, *wells: str) -> None:
    for well in wells:
        fake_microscope.require(XYStage).move_to_um(*_well_centre(well))
        page.capture_centre(well)


def _seed_plate(
    run_doc: Path,
    *,
    plugin: str = "plate-registration",
    assumed: tuple[str, ...] = (),
    minute: int = 0,
) -> None:
    """A registered plate (the pose is *registered*): *Go to* then needs no holder.

    ``plugin`` is the tool that wrote it and ``minute`` orders the starts: a later record of
    another tool is the one the page's plate comes from.
    """
    result = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="references",
        residual_um=1.5,
        scale_error=0.0,
        n_refs=3,
        assumed=assumed,
    )
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides="plate",
            attempt=f"a{minute}",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then + timedelta(minutes=1),
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={"plate": "96-well"},
            result=result.model_dump(mode="json"),
        ),
    )


def _row_texts(view: PlatePageView) -> list[str]:
    return [slot.text.object for slot in view.centre_slots if slot.box.visible]


# -- the Stand screen ------------------------------------------------------------------


def test_the_stand_screen_shows_the_preset_before_anything_else(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.show_step("stand")
    title, lead, stand_region, preset_region, known, calibrate, cont = list(
        view.step_panel.objects
    )
    assert lead.object == SCREEN_LEADS["stand"]
    # D12: the stand line and the preset with its *Load* come first, then what is known,
    # then the calibration; the next step is not here, and each action's command is under
    # its own *Command* disclosure.
    assert title.object == "Which stand, and which preset?"
    assert view.stand_text in stand_region
    assert view.preset_select in _objects([preset_region])
    assert view.preset_load_button in _objects([preset_region])
    assert known is view.known_block
    assert calibrate is view.calibrate_box
    assert cont is view.continue_button
    assert view.next_text not in _objects(list(view.step_panel.objects))
    cards = [o for o in _objects([calibrate]) if isinstance(o, pn.Card)]
    assert [(c.title, c.collapsed) for c in cards] == [("Command", True)]


def test_the_stand_screen_says_whether_the_stand_is_connected_and_connects_it(
    run_doc: Path, fake_microscope: Microscope
) -> None:
    page = PlatePage.closed(
        Profile.demo(),
        run_doc,
        opener=lambda: fake_microscope,
        entry_points=_entry_points(),
    )
    try:
        view = PlatePageView(page)
        view.show_step("stand")
        assert view.stand_text.object == "demo: not connected"
        # A second instance of the strip's *Connect*, through the same handler.
        assert view.stand_connect_button is not view.connect_button
        assert view.stand_connect_button.label == "Connect demo"
        assert view.stand_connect_button.visible
        assert view.known_block.object.split("\n")[0] == "Objective: not known"
        _click(view.stand_connect_button)
        assert page.join(10.0)
        view.tick()
        assert page.connected
        assert view.stand_text.object == "demo: connected"
        assert not view.stand_connect_button.visible
        assert view.known_block.object.split("\n")[0] == (
            f"Objective: {TEN_X}, in place"
        )
    finally:
        page.close()


def test_the_stand_screen_offers_the_calibration_until_the_objective_in_place_has_one(
    page: PlatePage,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view = PlatePageView(page)
    assert view.calibrate_box.visible
    assert view.known_block.object.split("\n") == [
        f"Objective: {TEN_X}, in place",
        "Camera: not calibrated (the arrows cannot say which way the image moves)",
        "Light: not known",
    ]
    # Calibrated at the objective in place: nothing to offer, and the line says when.
    seed_camera_calibration(run_doc, objective=TEN_X)
    view.on_refresh()
    assert not view.calibrate_box.visible
    assert view.known_block.object.split("\n")[1] == (
        f"Camera: calibrated at {TEN_X} on 2026-11-09 09:30:00 UTC"
    )
    # Calibrated at another objective: offered again, and the line says both.
    seed_camera_calibration(run_doc, objective=FORTY_X)
    view.on_refresh()
    assert view.calibrate_box.visible
    assert view.known_block.object.split("\n")[1] == (
        f"Camera: calibrated at {FORTY_X} on 2026-11-09 09:30:00 UTC, not at {TEN_X}"
    )


def test_calibrate_the_camera_is_the_same_call_as_the_tools_own_run(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = PlatePageView(page)
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        page, "run", lambda plugin, params: calls.append((plugin, list(params)))
    )
    # The form starts with what the profile suggests, and a typed value is sent by both.
    view.tool_fields["camera-calibration"]["step_um"].value = "40"
    _click(view.calibrate_button)
    _click(view.tool_runs["camera-calibration"])
    assert calls == [
        ("camera-calibration", ["step_um=40"]),
        ("camera-calibration", ["step_um=40"]),
    ]
    # Its command is under a closed *Command* disclosure, and follows the form.
    words = shlex.split(view.calibrate_command.object)
    assert words[:3] == ["smc", "run", "camera-calibration"]
    assert words[-2:] == ["--param", "step_um=40"]
    view.tool_fields["camera-calibration"]["step_um"].value = "55"
    assert shlex.split(view.calibrate_command.object)[-2:] == ["--param", "step_um=55"]


# -- the Plate screen ------------------------------------------------------------------


def test_the_designed_screens_list_no_next_step_region(page: PlatePage) -> None:
    # Stand and Plate (B2, #444), Wells and Scan (B3, #445), Results (B4, #446): each action
    # is the screen's own.
    view = PlatePageView(page)
    for step in ("stand", "plate", "wells", "scan", "results"):
        view.show_step(step)
        shown = _objects(list(view.step_panel.objects))
        assert view.next_text not in shown, step
        assert view.next_button not in shown, step
        assert view.next_note not in shown, step
    # The next step is still in the expert view.
    view.show_expert(True)
    assert view.next_text in _objects(list(view.step_panel.objects))


def test_the_plate_screen_has_one_register_button_and_no_wall_widgets(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.show_step("plate")
    shown = _objects(list(view.step_panel.objects))
    buttons = [o for o in shown if isinstance(o, pn.widgets.Button)]
    # One *Register the plate*, and no *Run plate-registration* (the walk found two Run
    # buttons, one under the other): the tool's form and its Run are the expert view's.
    assert [b.label for b in buttons if "egister" in b.label] == ["Register the plate"]
    assert view.tool_runs["plate-registration"] not in shown
    assert not [b for b in buttons if b.label.startswith("Run ")]
    # The walls and the capture buttons of the section stay in the expert view.
    for hidden in (
        view.capture_wall_button,
        view.capture_side_select,
        view.walls_capture_button,
        view.walls_forget_button,
        view.capture_centre_button,
        view.forget_button,
        view.captures_text,
        view.image_note,
        view.capture_note,
    ):
        assert hidden not in shown
    view.show_expert(True)
    in_expert = _objects(list(view.step_panel.objects))
    assert view.tool_runs["plate-registration"] in in_expert
    assert view.capture_wall_button in in_expert
    # The other ways are named, with where to run them.
    assert view.other_ways_text.object == OTHER_WAYS_TO_REGISTER
    assert "walls" in view.other_ways_text.object
    assert "expert view" in view.other_ways_text.object


def test_before_any_capture_nothing_on_the_plate_screen_is_in_the_warning_colour(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.show_step("plate")
    # The placeholder asks which plate is on the stage, and nothing is captured: the rows
    # say what to do and the wait is the button's alone (the walk: a warning before any action).
    assert not page.plate_chosen
    assert view.plate_note.visible
    assert view.register_wait.object == PLATE_NOT_CHOSEN
    coloured = [
        o
        for o in _objects(list(view.step_panel.objects))
        if isinstance(o, pn.pane.Str) and o.styles.get("color") == _WARNING
    ]
    assert coloured == []
    page.choose_plate("96-well")
    view.refresh()
    assert view.register_wait.object == CAPTURE_WELLS


def test_the_plate_screen_lists_three_rows_and_names_where_the_plate_comes_from(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.show_step("plate")
    assert view.plate_from_text.object == (
        "the page's placeholder: choose the plate on the stage"
    )
    page.choose_plate("96-well")
    view.refresh()
    assert view.plate_from_text.object == "chosen on this page"
    assert [slot.box.visible for slot in view.centre_slots] == [True, True, True]
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to A1",
        "Go to A12",
        "Go to H1",
    ]
    assert [slot.capture_button.label for slot in view.centre_slots] == [
        "Capture the centre of A1",
        "Capture the centre of A12",
        "Capture the centre of H1",
    ]
    # The map is *unplaced* on this stand (no holder): the rows say to jog.
    assert _row_texts(view) == [
        "A1: jog there with the arrows, centre it, capture",
        "A12: jog there with the arrows, centre it, capture",
        "H1: jog there with the arrows, centre it, capture",
    ]
    # Another plate: its own far corners; a coverslip has one well, so one row.
    page.choose_plate("384-well")
    view.refresh()
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to A1",
        "Go to A24",
        "Go to P1",
    ]
    page.choose_plate("coverslip-18mm-square")
    view.refresh()
    assert [slot.box.visible for slot in view.centre_slots] == [True, False, False]
    # The slots are built once and never rebuilt (D5): the same widgets after every redraw.
    slots = list(view.centre_slots)
    view.refresh()
    assert view.centre_slots == slots


def test_a_capture_row_captures_the_well_it_names_and_becomes_redo(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    view.refresh()
    fake_microscope.require(XYStage).move_to_um(*_well_centre("A12"))
    _click(view.centre_slots[1].capture_button)
    assert [(r.well, r.x_um, r.y_um) for r in page.refs] == [
        ("A12", *_well_centre("A12"))
    ]
    assert page.capture_objectives == {"A12": TEN_X}
    assert view.centre_slots[1].text.object == (
        f"A12: captured at ({_well_centre('A12')[0]:.1f}, "
        f"{_well_centre('A12')[1]:.1f}) µm with {TEN_X}"
    )
    assert view.centre_slots[1].capture_button.label == "Redo"
    assert view.centre_slots[0].capture_button.label == "Capture the centre of A1"
    # A capture of a well that is not one of the three (clicked on the map) is listed too,
    # as text after the rows: the suggestion is not a rule.
    fake_microscope.require(XYStage).move_to_um(*_well_centre("C5"))
    page.capture_centre("C5")
    view.refresh()
    assert view.other_captures.visible
    assert view.other_captures.object == (
        "Other captures:\n"
        f"C5: captured at ({_well_centre('C5')[0]:.1f}, {_well_centre('C5')[1]:.1f}) "
        f"µm with {TEN_X}"
    )
    # Forgetting it (the expert view's button) takes the line away.
    page.forget("C5")
    view.refresh()
    assert not view.other_captures.visible
    assert view.other_captures.object == ""


def test_a_capture_row_shows_a_turn_between_two_captures(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    page.choose_plate("96-well")
    _capture(page, fake_microscope, "A1")
    fake_microscope.require(ObjectiveTurret).select(FORTY_X, confirm=True)
    _capture(page, fake_microscope, "A12")
    view.refresh()
    texts = _row_texts(view)
    assert texts[0].endswith(f"µm with {TEN_X}")
    assert texts[1].endswith(f"µm with {FORTY_X}")


def test_a_row_goes_to_the_well_it_names(
    page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    _seed_plate(run_doc)
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    _click(view.centre_slots[1].goto_button)
    assert page.join(10.0)
    view.tick()
    # The same call as a drive click: the controller drove, and made the well current.
    assert StubGoto.ran == ["A12"]
    assert page.current == "A12"
    # The answer is drawn on the row that started it, in the plain colour.
    note = view.centre_slots[1].note
    assert note.object.startswith("goto-well done in ")
    assert note.styles.get("color") != _WARNING
    assert view.centre_slots[0].note.object == ""
    assert view.message.object == ""


def test_a_refused_go_to_is_drawn_on_its_row(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The real `goto-well` on a stand whose profile says nothing of its holder, with the
    # plate not registered: the map is *unplaced* and the layer refuses. On the owner's
    # walk a drive before the registration moved nothing and said nothing (#431).
    page = make_page(Slow)
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    view.refresh()
    _click(view.centre_slots[2].goto_button)
    assert view.centre_slots[2].note.object == GOTO_UNPLACED
    assert view.centre_slots[2].note.styles["color"] == _WARNING
    # Only on that row, and not in the header's line far from the button.
    assert view.centre_slots[0].note.object == ""
    assert view.message.object == ""
    # The same text, kept by the controller for every tab and drawn by the drive bar.
    assert view.goto_text.object == GOTO_UNPLACED


def test_an_empty_note_takes_no_height_and_a_said_one_does(
    make_page: Callable[..., PlatePage],
) -> None:
    # An empty text pane still draws a line of its own, which spread the three rows out by
    # about 25 px each in `/ui-check` (D12 has 620 px): the notes show only with a text.
    page = make_page(Slow)
    view = PlatePageView(page)
    page.choose_plate("96-well")
    view.refresh()
    assert [slot.note.visible for slot in view.centre_slots] == [False] * 3
    assert all(slot.note.object == "" for slot in view.centre_slots)
    # The wait under *Register the plate* is a text: shown. The calibration's note has none
    # while the stand is connected and idle: hidden.
    assert view.register_wait.visible
    assert view.register_wait.object == CAPTURE_WELLS
    assert not view.calibrate_note.visible
    _click(view.centre_slots[1].goto_button)
    assert view.centre_slots[1].note.visible
    assert view.centre_slots[1].note.object == GOTO_UNPLACED
    assert not view.centre_slots[0].note.visible


def test_a_goto_well_that_ends_failed_is_drawn_on_its_row(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = make_page(FailingGoto)
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    _click(view.centre_slots[0].goto_button)
    assert page.join(10.0)
    view.tick()
    # The run's own outcome, whole (FM-45), in the warning colour: it did not end `done`.
    note = view.centre_slots[0].note
    assert (
        note.object == "goto-well: PluginRunError: goto-well: the stage did not arrive"
    )
    assert note.styles["color"] == _WARNING
    assert view.message.object == ""


def test_go_to_is_disabled_during_live_and_says_why(
    page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    _seed_plate(run_doc)
    fake_microscope.override(Camera, LiveCamera())
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    assert not any(slot.goto_button.disabled for slot in view.centre_slots)
    view.on_live()
    try:
        assert page.is_live
        view.tick()
        for slot in view.centre_slots:
            assert slot.goto_button.disabled
            assert GOTO_NOT_DURING_LIVE in slot.note.object.split("\n")
            assert slot.note.styles.get("color") != _WARNING
        # A click that still arrives is the controller's refusal (FM-169): the state's, so
        # the note every row already has says it, and the click moves no row (a refusal
        # because an activity goes is not the drive of the row that was clicked).
        _click(view.centre_slots[0].goto_button)
        assert page.status().goto == "live is running"
        assert view.centre_slots[0].note.object.split("\n") == [GOTO_NOT_DURING_LIVE]
        assert view.message.object == ""
        assert StubGoto.ran == []
    finally:
        page.live(False)
    assert page.join(10.0)
    view.tick()
    assert not any(slot.goto_button.disabled for slot in view.centre_slots)
    assert GOTO_NOT_DURING_LIVE not in view.centre_slots[1].note.object


def test_capture_stays_enabled_during_live(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, LiveCamera())
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    view.refresh()
    view.on_live()
    try:
        assert page.is_live
        view.tick()
        # The operator centres in Live with the arrows and captures there (D11).
        assert not any(slot.capture_button.disabled for slot in view.centre_slots)
        fake_microscope.require(XYStage).move_to_um(*_well_centre("A12"))
        _click(view.centre_slots[1].capture_button)
        assert [r.well for r in page.refs] == ["A12"]
        assert view.message.object == ""
    finally:
        page.live(False)
    assert page.join(10.0)


def test_register_runs_the_pairs_it_drew(
    page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    _capture(page, fake_microscope, "A1", "A12", "H1")
    view.refresh()
    pairs, wait = view._register_drawn
    assert wait is None
    assert not view.register_button.disabled
    assert view.register_wait.object == ""
    # The *Command* disclosure shows the call the button makes.
    assert view.register_card.visible
    assert view.register_card.collapsed is True
    words = shlex.split(view.register_command.object)
    assert words[:3] == ["smc", "run", "plate-registration"]
    assert [w for w in words if w.startswith(("plate=", "refs=", "walls="))] == [
        p for p in pairs if p.startswith(("plate=", "refs=", "walls="))
    ]
    # Another tab forgets a centre between the redraw and the click: the button runs what
    # was drawn, the pairs the operator read (#287), not pairs computed at the click.
    page.forget("A12")
    _click(view.register_button)
    assert page.join(30.0)
    record = RunDocument.load(run_doc).stages["plate-registration"]
    assert record.status is StageStatus.done
    assert [ref["well"] for ref in record.params["refs"]] == ["A1", "A12", "H1"]
    # The result line follows the run's end, and says what it measured.
    view.tick()
    assert view.registered_block.visible
    assert view.registered_block.object.startswith("Registered: residual ")
    assert ", turned " in view.registered_block.object
    assert view.step_buttons["plate"].label == "2. Plate ✓"


def test_a_disabled_register_or_go_to_is_refused_by_the_controller(
    make_page: Callable[..., PlatePage], run_doc: Path, fake_microscope: Microscope
) -> None:
    # The plate is registered before the page starts, as a plate a tool left in the run
    # document: the page takes its name (chosen), and *Go to* needs no holder.
    _seed_plate(run_doc)
    page = make_page(StubGoto, Slow)
    assert page.plate_chosen
    view = PlatePageView(page)
    view.show_step("plate")
    # A wait is drawn (no centre captured yet): the button is disabled and says what it
    # waits for, and a click that arrives anyway runs nothing.
    view.refresh()
    assert view.register_button.disabled
    assert view.register_wait.object == CAPTURE_WELLS
    _click(view.register_button)
    assert view.message.object == CAPTURE_WELLS
    assert RunDocument.load(run_doc).stages["plate-registration"].attempt == "a0"
    # An activity goes: *Go to*, *Register* and *Calibrate* are disabled with the reason
    # beside them, and every click that arrives is the controller's refusal (FM-169).
    _capture(page, fake_microscope, "A1", "A12", "H1")
    view.refresh()
    assert not view.register_button.disabled
    _click(view.tool_runs["slow"])
    try:
        assert Slow.started.wait(5)
        view.tick()
        reason = "slow is running: wait for it to end, or press STOP."
        assert view.register_button.disabled
        assert view.register_wait.object == reason
        assert view.calibrate_button.disabled
        assert view.calibrate_note.object == reason
        assert all(slot.goto_button.disabled for slot in view.centre_slots)
        assert reason in view.centre_slots[0].note.object
        # Capture reads the stage, so it is disabled while another activity goes too.
        assert all(slot.capture_button.disabled for slot in view.centre_slots)
        _click(view.register_button)
        assert view.message.object == "slow is running"
        _click(view.centre_slots[0].goto_button)
        assert page.status().goto == "slow is running"
        assert StubGoto.ran == []
        view.message.object = ""
        _click(view.centre_slots[1].capture_button)
        assert view.message.object == "slow is running"
    finally:
        Slow.release.set()
    assert page.join(10.0)
    assert RunDocument.load(run_doc).stages["plate-registration"].attempt == "a0"


def test_a_closed_stand_disables_the_stand_dependent_buttons_and_refuses_them(
    run_doc: Path,
) -> None:
    page = PlatePage.closed(
        Profile.demo(), run_doc, entry_points=_entry_points(StubGoto)
    )
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        page.choose_plate("96-well")
        view.refresh()
        assert all(slot.goto_button.disabled for slot in view.centre_slots)
        assert all(slot.capture_button.disabled for slot in view.centre_slots)
        assert view.register_button.disabled
        _click(view.centre_slots[0].capture_button)
        assert view.message.object == "the stand is not connected"
        assert page.refs == ()
    finally:
        page.close()


def test_the_rows_follow_a_capture_another_tab_made(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # What each tab draws is the controller's (FM-151): a capture one tab makes is in the
    # other's rows at its next tick, objective included.
    first, second = PlatePageView(page), PlatePageView(page)
    page.choose_plate("96-well")
    first.refresh()
    second.refresh()
    fake_microscope.require(XYStage).move_to_um(*_well_centre("A1"))
    _click(first.centre_slots[0].capture_button)
    second.tick()
    assert second.centre_slots[0].capture_button.label == "Redo"
    assert second.centre_slots[0].text.object.endswith(f"µm with {TEN_X}")
    # A recapture at the same place with another objective changes no centre, only its
    # label: the tick must still redraw the row.
    fake_microscope.require(ObjectiveTurret).select(FORTY_X, confirm=True)
    _click(first.centre_slots[0].capture_button)
    second.tick()
    assert second.centre_slots[0].text.object.endswith(f"µm with {FORTY_X}")


def test_no_widget_is_placed_twice_on_a_screen(page: PlatePage) -> None:
    # A widget placed twice in one layout is drawn once and nothing fails: so each screen's
    # own list is checked for a second place (m5 §17.2 D5), every widget of the Stand and
    # Plate screens included.
    view = PlatePageView(page)
    for step in ("stand", "plate"):
        view.show_step(step)
        found = [id(o) for o in _objects(list(view.step_panel.objects))]
        assert len(found) == len(set(found)), f"{step} places a widget twice"


# -- what the adversarial review of #444 found ------------------------------------------

HOSTILE = "<img src=x onerror=alert(1)>"


class _Turret:
    """A turret that names its objective with ``label``: a stand's text, which may hold anything."""

    def __init__(self, label: str) -> None:
        self._label = label

    def labels(self) -> list[str]:
        return [self._label]

    def current(self) -> str:
        return self._label

    def select(self, label: str, *, confirm: bool = False) -> str:
        return label

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0

    def offset(self, label: str) -> ObjectiveOffset:
        return ObjectiveOffset(label, None, None, None, "")


class HostileGoto(StubGoto):
    """A ``goto-well`` whose run ends ``failed`` with a message that is markup."""

    def run(self, microscope: Any, params: GotoParams, ctx: RunContext) -> NoResult:
        raise PluginRunError("goto-well", HOSTILE)


def test_the_rows_follow_the_maps_pose_and_an_estimated_drive_is_said_in_plain_words(
    holder_page: PlatePage, run_doc: Path
) -> None:
    # The post-#462 path, the owner's walk's main finding: a drive before the registration
    # went nowhere and said nothing. On a stand with a holder the map is *estimated*, the
    # rows say so, and the row that started the drive says where it is going, in the plain
    # colour (nothing in the warning colour before a capture).
    page = holder_page
    page.choose_plate("96-well")
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    assert view._view is not None
    assert view._view.pose.kind == "estimated"
    assert _row_texts(view) == [
        f"{well}: go there (estimated position), centre it by eye, capture"
        for well in ("A1", "A12", "H1")
    ]
    _click(view.centre_slots[0].goto_button)
    assert BlockingGoto.started.wait(5)
    view.tick()
    note = view.centre_slots[0].note
    assert note.object.split("\n") == [
        "goto-well is running: wait for it to end, or press STOP.",
        GOTO_ESTIMATED.format(well="A1"),
    ]
    assert note.styles.get("color") != _WARNING
    BlockingGoto.release.set()
    assert page.join(10.0)
    view.tick()
    assert note.object.startswith("goto-well done in ")
    assert note.styles.get("color") != _WARNING
    # A plate registered since: the rows are those of a registered map, and what the drive
    # said (an estimate) is history, not left under a plate that is now registered.
    _seed_plate(run_doc)
    view.refresh()
    assert view._view.pose.kind == "registered"
    assert _row_texts(view)[0] == "A1: go there, centre it, capture"
    assert note.object == ""
    assert not note.visible


def test_a_rows_drive_text_is_history_once_something_it_was_about_changes(
    make_page: Callable[..., PlatePage],
    run_doc: Path,
    fake_microscope: Microscope,
) -> None:
    page = make_page(Slow)  # the real goto-well, on a stand with no holder: unplaced
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    view.refresh()
    note = view.centre_slots[0].note
    _click(view.centre_slots[0].goto_button)
    assert note.object == GOTO_UNPLACED
    # A redraw that finds nothing changed keeps it: the operator is reading it.
    view.refresh()
    assert note.object == GOTO_UNPLACED
    # A centre captured is something it was about; it goes.
    _capture(page, fake_microscope, "A1")
    view.refresh()
    assert note.object == ""
    # The refusal again, and then a registered plate: the same.
    page.forget()
    view.refresh()
    _click(view.centre_slots[0].goto_button)
    assert note.object == GOTO_UNPLACED
    _seed_plate(run_doc)
    view.refresh()
    assert view.registered_block.visible
    assert note.object == ""
    assert not note.visible


def test_a_refused_click_while_a_drive_goes_leaves_the_drive_on_its_row(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The controller keeps one drive text for every tab. A click that arrives while a drive
    # goes (a second tab, a stale page: FM-169) is refused, and must not take the running
    # drive's `done` to a well the stage never visited.
    _seed_plate(run_doc)
    page = make_page(BlockingGoto, Slow)
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    _click(view.centre_slots[0].goto_button)
    assert BlockingGoto.started.wait(5)
    view.tick()
    _click(view.centre_slots[1].goto_button)
    assert StubGoto.ran == ["A1"]
    assert view.message.object == ""
    BlockingGoto.release.set()
    assert page.join(10.0)
    view.tick()
    assert view.centre_slots[0].note.object.startswith("goto-well done in ")
    assert view.centre_slots[1].note.object == ""
    assert view.centre_slots[2].note.object == ""


def test_a_map_drive_takes_the_answer_from_the_row(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    _click(view.centre_slots[1].goto_button)
    assert page.join(10.0)
    view.tick()
    assert view.centre_slots[1].note.object.startswith("goto-well done in ")
    # A drive armed on the map and clicked on a well: the answer is its, and no row's.
    view.drive_toggle.value = True
    view.on_tap(SimpleNamespace(x=0.0, y=0.0))
    assert page.join(10.0)
    view.tick()
    assert view.goto_text.object.startswith("goto-well done in ")
    assert view.centre_slots[1].note.object == ""
    assert not view.centre_slots[1].note.visible


def test_register_says_what_it_replaces_before_the_click(
    make_page: Callable[..., PlatePage],
    run_doc: Path,
    fake_microscope: Microscope,
) -> None:
    # m5 §11.4 and FM-172: a failed registration leaves no plate where a usable one stood,
    # and the page says so before the click. The tool's own section, which said it, is in
    # the expert view only: the Plate screen's button says it in its own words.
    page = make_page(Slow)
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    view.refresh()
    assert view.register_replaces.object == ""
    assert not view.register_replaces.visible
    _seed_plate(run_doc)
    _capture(page, fake_microscope, "A1", "A12", "H1")
    view.refresh()
    assert view.register_replaces.visible
    assert view.register_replaces.object == (
        "Registering again replaces this plate (references, 3 wells). A failed run "
        "leaves no plate; the centres captured stay, so Redo the wrong one and register "
        "again to get a plate back."
    )
    assert view.register_replaces.styles.get("color") != _WARNING
    assert not view.register_button.disabled


def test_a_windows_page_says_its_commands_are_quoted_for_a_posix_shell(
    make_page: Callable[..., PlatePage], fake_microscope: Microscope
) -> None:
    # FM-175: `shlex.join` quotes for a POSIX shell, and cmd.exe and PowerShell 5.1 read
    # quotes differently. The next step said so under its line; the *Command* disclosures
    # that replaced it on these two screens say it too, with the same words, on Windows only.
    page = make_page(Slow)
    view = PlatePageView(page)
    page.choose_plate("96-well")
    _capture(page, fake_microscope, "A1", "A12", "H1")
    view.refresh()
    words = (
        "that line is quoted for a POSIX shell (bash, zsh); PowerShell and cmd.exe "
        "read quotes differently: press the button, which makes the same call"
    )
    # Both ways are set, never left to the runner's own platform: on a Windows runner the
    # unset default is the note showing (found by CI on the first push of #444).
    view._windows = False
    view.refresh()
    for pane in (view.register_command_note, view.calibrate_command_note):
        assert pane.object == ""
        assert not pane.visible
    view._windows = True
    view.refresh()
    for pane, card_of in (
        (view.register_command_note, view.register_card),
        (view.calibrate_command_note, view.calibrate_box.objects[2]),
    ):
        assert pane.object == words
        assert pane.visible
        assert pane in card_of.objects
        assert pane.styles.get("color") != _WARNING
    # The next step's text says the same thing, from one source.
    step = NextStep("scan", (), (), "smc run scan", None)
    assert next_step_text(step, windows=True).endswith(f"  {words}")
    assert words not in next_step_text(step, windows=False)


def test_the_text_panes_of_the_two_screens_are_plain_text_whatever_the_stand_says(
    make_page: Callable[..., PlatePage],
    run_doc: Path,
    fake_microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-139. Free text reaches these screens from the stand (the turret's label, an
    # exception's message) and from a result (`assumed`): it is shown as it is, never as
    # markup, or a label of `<img onerror=...>` runs on the stand PC.
    _seed_plate(run_doc, assumed=(HOSTILE,))
    fake_microscope.override(ObjectiveTurret, _Turret(HOSTILE))
    page = make_page(HostileGoto)
    real_status = page.status
    monkeypatch.setattr(
        page,
        "status",
        lambda: dataclasses.replace(real_status(), objective=HOSTILE),
    )
    view = PlatePageView(page)
    _capture(page, fake_microscope, "A1")
    _click(view.centre_slots[1].goto_button)
    assert page.join(10.0)
    view.refresh()
    view.tick()
    # Said as it was given, in each of the three places that hold a stand's or a result's words.
    assert view.known_block.object.split("\n")[0] == f"Objective: {HOSTILE}, in place"
    assert HOSTILE in view.centre_slots[0].text.object
    assert view.registered_block.object.split("\n")[1] == f"assumed: {HOSTILE}"
    assert HOSTILE in view.centre_slots[1].note.object
    # And every text pane of both screens is a `Str`: counted together, so that a pane that
    # becomes Markdown is a different count and not a pass.
    shown: list[Any] = []
    for step in ("stand", "plate"):
        view.show_step(step)
        objects = _objects(list(view.step_panel.objects))
        # A card draws its title as an HTML pane of its own, from a constant of the page's.
        titles = {f"<h3>{o.title}</h3>" for o in objects if isinstance(o, pn.Card)}
        shown += [
            o
            for o in objects
            if isinstance(o, pn.pane.PaneBase)
            and not (type(o) is pn.pane.HTML and o.object in titles)
        ]
    assert len(shown) > 10
    assert [type(o).__name__ for o in shown if type(o) is not pn.pane.Str] == []


def test_the_known_block_reads_the_light_and_the_plate_line_reads_the_preset(
    make_page: Callable[..., PlatePage],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)  # presets are looked for in ./profiles/presets
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    (folder / "weekly.toml").write_text(
        'stand = "demo"\nplate = "96-well"\n', encoding="utf-8"
    )
    page = make_page(Slow)
    real_status = page.status
    light: list[bool | None] = [None]
    monkeypatch.setattr(
        page,
        "status",
        lambda: dataclasses.replace(real_status(), light_on=light[0]),
    )
    view = PlatePageView(page)
    # The light as the stand read it: the strip's own value, on the Stand screen too.
    for value, said in ((None, "not known"), (True, "on"), (False, "off")):
        light[0] = value
        view.tick()
        assert view.known_block.object.split("\n")[2] == f"Light: {said}"
    # The plate's source: the placeholder, then the loaded preset's plate.
    assert view.plate_from_text.object == (
        "the page's placeholder: choose the plate on the stage"
    )
    view.preset_select.value = "weekly"
    _click(view.preset_load_button)
    assert view.plate_from_text.object == "from the preset weekly"


def test_the_result_line_shows_the_assumed_lines_of_the_tool_that_wrote_the_plate(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # An older registration and a newer wall registration: the plate is the newer's, and
    # so are the assumptions under the result line (not an older tool's).
    _seed_plate(run_doc, assumed=("from the registration",))
    _seed_plate(run_doc, plugin="plate-walls", assumed=("from the walls",), minute=2)
    page = make_page(Slow)
    view = PlatePageView(page)
    view.show_step("plate")
    assert view._view is not None
    assert view._view.registered_by == "plate-walls"
    assert view.registered_block.object.split("\n")[1:] == ["assumed: from the walls"]


def test_a_pane_with_nothing_to_say_takes_no_height_on_a_fresh_plate_screen(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("96-well")
    view.refresh()
    for pane in (
        view.registered_block,
        view.other_captures,
        view.register_replaces,
        view.register_card,
        view.calibrate_note,
    ):
        assert not pane.visible
    _capture(page, fake_microscope, "A1", "A12", "H1")
    view.refresh()
    assert view.register_card.visible
    assert not view.registered_block.visible


def test_a_stand_that_lacks_what_a_tool_needs_says_why_beside_its_button(
    make_page: Callable[..., PlatePage], fake_microscope: Microscope
) -> None:
    # The controller refuses these runs on its own (FM-169); the reason beside the button is
    # the courtesy, and it is the tool's own (the next step's precedence).
    page = make_page(Slow)
    real_has = fake_microscope.has

    def has(capability: type) -> bool:
        return capability not in {Camera, XYStage} and real_has(capability)

    fake_microscope.has = has  # type: ignore[method-assign]
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        page.choose_plate("96-well")
        view.refresh()
        blocked = view._blocked
        assert blocked["camera-calibration"]
        assert view.calibrate_button.disabled
        assert view.calibrate_note.object == blocked["camera-calibration"]
    finally:
        del fake_microscope.has


def test_the_tools_own_reason_comes_before_the_screens_wait(run_doc: Path) -> None:
    # With the stand closed, *Register the plate* has two things to say: the screen waits
    # for a centre, and the tool needs the stand. What the stand says first wins (the
    # next step's precedence): it is the reason that holds longer.
    page = PlatePage.closed(
        Profile.demo(), run_doc, entry_points=_entry_points(StubGoto, Slow)
    )
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        page.choose_plate("96-well")
        view.refresh()
        assert view._register_drawn[1] == CAPTURE_WELLS
        assert view._blocked["plate-registration"] == "the stand is not connected"
        assert view.register_button.disabled
        assert view.register_wait.object == "the stand is not connected"
    finally:
        page.close()


def test_a_registration_that_did_not_load_is_said_and_its_button_is_off(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    entry_points = [
        EntryPoint(name="goto-well", value=REAL["goto-well"], group=GROUP),
        EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP),
    ]
    page = PlatePage(fake_microscope, run_doc, entry_points=entry_points)
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        page.choose_plate("96-well")
        view.refresh()
        assert view.register_button.disabled
        assert view.register_wait.object == (
            "plate-registration did not load: the problems line says why"
        )
        # And with no calibration tool, *Calibrate the camera* is not offered at all.
        assert not view.calibrate_box.visible
    finally:
        page.close()


def test_the_second_connect_is_off_while_the_stand_is_connecting(
    run_doc: Path, fake_microscope: Microscope
) -> None:
    release = threading.Event()

    def opener() -> Microscope:
        release.wait(10.0)
        return fake_microscope

    page = PlatePage.closed(
        Profile.demo(), run_doc, opener=opener, entry_points=_entry_points()
    )
    try:
        view = PlatePageView(page)
        view.show_step("stand")
        _click(view.stand_connect_button)
        view.tick()
        assert view.stand_text.object == "demo: connecting…"
        assert view.stand_connect_button.disabled
        assert view.stand_connect_button.visible
        release.set()
        assert page.join(10.0)
        view.tick()
        assert view.stand_text.object == "demo: connected"
        assert not view.stand_connect_button.visible
    finally:
        release.set()
        page.close()


# -- the header and the scroll (D12) ---------------------------------------------------


def test_the_header_holds_the_strip_the_outcomes_and_the_step_bar(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    header = view.header
    assert header.styles["position"] == "sticky"
    held = {id(o) for o in header.select()}
    for widget in (
        view.connect_button,
        view.stop_button,
        view.status_text,
        view.connection_text,
        view.drive_text,
        view.run_text,
        view.message,
        view.problems_text,
        view.expert_toggle,
        *view.step_buttons.values(),
        *view.step_summaries.values(),
    ):
        assert id(widget) in held
    # The panel and the centre are below it, and no screen lists a widget of the header:
    # it would be drawn in one of the two places only.
    for items in view.screens.values():
        assert not held & {id(o) for o in _objects(items)}
    below = {id(o) for o in view.layout[1].select()}
    assert not held & below
    assert id(view.step_panel) in below
    # Both figures are in the centre, below the header, whatever screen the panel shows.
    drawn = [p.object for p in view.layout[1].select(pn.pane.Bokeh)]
    assert view.figure in drawn
    assert view.image_figure in drawn


def test_switching_screens_scrolls_the_panel_to_the_top(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.show_step("stand")
    # The browser reports a scrolled panel's position to the server (measured on Panel
    # 1.10.0); the next screen must open at its title, not as far down as the last was.
    view.step_panel.scroll_position = 900
    view.show_step("plate")
    assert view.step_panel.scroll_position == 0
    view.step_panel.scroll_position = 900
    view.show_expert(True)
    assert view.step_panel.scroll_position == 0
    view.step_panel.scroll_position = 900
    view.continue_button.visible = True
    view.show_expert(False)
    assert view.step_panel.scroll_position == 0


# -- the whole path on a stand with a sample -------------------------------------------


@pytest.mark.demo
def test_a_plate_registered_from_the_plate_screen_alone(
    mm_available: bool, tmp_path: Path
) -> None:
    # The issue's second criterion: on `demo-sample`, Connect, the Plate screen,
    # *Register the plate*, and the result line drawn, with no tool section and no next step.
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    shipped = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
    page = PlatePage.closed(Profile.load(shipped), tmp_path / "run.json")
    try:
        view = PlatePageView(page)
        assert view.step == "stand"
        _click(view.stand_connect_button)
        assert page.join(60.0)
        view.tick()
        assert page.connected, page.status().outcome
        view.show_step("plate")
        assert view.plate_from_text.object == "from the profile's sample"
        # The sample knows the three centres: known, not measured, and *Register* is on.
        assert _row_texts(view) == [
            "A1: known from the profile's sample, not measured",
            "A12: known from the profile's sample, not measured",
            "H1: known from the profile's sample, not measured",
        ]
        assert not view.register_button.disabled, view.register_wait.object
        _click(view.register_button)
        assert page.join(60.0)
        view.tick()
        assert page.status().outcome is not None
        assert page.status().outcome.startswith("plate-registration done in ")
        assert view.registered_block.visible
        lines = view.registered_block.object.split("\n")
        assert lines[0].startswith("Registered: residual ")
        # What the registration did not measure is said under it (m5 §8.8.4).
        assert any(
            line.startswith("assumed: ") and "known, not measured" in line
            for line in lines[1:]
        ), lines
        assert view.step_buttons["plate"].label == "2. Plate ✓"
        assert view.continue_button.label == "Continue to Wells"
    finally:
        page.close()
        page.release()
