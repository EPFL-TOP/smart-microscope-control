"""The Stand and Plate screens of the guided page, without a browser (m5 §17.4.1, §17.4.2, #444).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1``. A
click on a *disabled* button still calls its handler on the server (measured on Panel 1.10.0
and 1.9.4), so the controller's refusals are the guard (FM-169) and a test clicks a disabled
button to prove it. What this cannot show is a browser: the layout at 1280 x 620 and that
the title and the first action are inside the page are for ``/ui-check`` (FM-161). Panel's
warnings are errors (FM-147), and a handler that failed fails the test that did not expect it.
"""

from __future__ import annotations

import os
import shlex
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, ObjectiveTurret, XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.ui.controller import PlatePage
from smc.ui.guide import (
    CAPTURE_WELLS,
    GOTO_NOT_DURING_LIVE,
    GOTO_UNPLACED,
    OTHER_WAYS_TO_REGISTER,
    PLATE_NOT_CHOSEN,
    SCREEN_LEADS,
)
from smc.ui.plate_page import _WARNING, PlatePageView

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
    try:
        yield
    finally:
        Slow.release.set()


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


def _seed_plate(run_doc: Path) -> None:
    """A registered plate (the pose is *registered*): *Go to* then needs no holder."""
    result = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="references",
        residual_um=1.5,
        scale_error=0.0,
        n_refs=3,
    )
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


def test_neither_screen_lists_the_next_step_region(page: PlatePage) -> None:
    view = PlatePageView(page)
    for step in ("stand", "plate"):
        view.show_step(step)
        shown = _objects(list(view.step_panel.objects))
        assert view.next_text not in shown, step
        assert view.next_button not in shown, step
        assert view.next_note not in shown, step
    # The next step is still in the expert view, and on the provisional screens.
    view.show_expert(True)
    assert view.next_text in _objects(list(view.step_panel.objects))
    view.show_step("wells")
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
        # A click that still arrives is the controller's refusal, drawn on its row (FM-169).
        _click(view.centre_slots[0].goto_button)
        assert page.status().goto == "live is running"
        assert view.centre_slots[0].note.object.split("\n") == [
            GOTO_NOT_DURING_LIVE,
            "live is running",
        ]
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
