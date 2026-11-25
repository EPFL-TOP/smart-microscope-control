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
from smc.plugins.slots import PlateResult, WallImages, WellFit
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
    REGISTER_ACROSS_A_TURN,
    SAMPLE_REGISTERS,
    SCREEN_LEADS,
    NextStep,
    local_time,
)
from smc.ui.plate_page import _WARNING, PlatePageView, TextPane, next_step_text

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
#: The Ti2-E's ``[safety] xy_soft_limits_um``. With its holder's centre (400, -325) a
#: Greiner 24-well has A1 at (-48 225, 28 850), A6 at (49 025, 28 850) and D1 at
#: (-48 225, -29 500) µm, and B2 at (-28 775, 9 400): measured 2026-11-16 (m5 §19.1).
TI2E_LIMITS = ((-49000.0, 49000.0), (-29000.0, 29000.0))
GREINER_CORNERS_NOTE = (
    "Not the plate's far corners (A1, A6 and D1): the stage's soft limits cut them, "
    "as the holder places this plate."
)

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


def _ti2e_profile(base: Profile) -> Profile:
    """``base`` with the Ti2-E's holder and XY soft limits (profiles/nikon-ti2.toml).

    A Greiner 24-well in that holder has A6 and D1 outside that travel (m5 §19.1).
    """
    return base.model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=(400.0, -325.0), measured="a test's own centre"
            ),
            "safety": base.safety.model_copy(update={"xy_soft_limits_um": TI2E_LIMITS}),
        }
    )


@pytest.fixture
def ti2e_microscope(fake_core: FakeCore) -> Iterator[Microscope]:
    """A stand with the Ti2-E's holder and soft limits."""
    microscope = Microscope.from_core(
        cast("CMMCorePlus", fake_core), _ti2e_profile(Profile.demo())
    )
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def ti2e_page(ti2e_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    """A page on that stand: the map is *estimated*, and some wells are out of reach."""
    controller = PlatePage(
        ti2e_microscope, run_doc, entry_points=_entry_points(StubGoto, Slow)
    )
    try:
        yield controller
    finally:
        Slow.release.set()
        controller.close()


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
    plate: str = "96-well",
    a1_center_xy_um: tuple[float, float] = (0.0, 0.0),
) -> None:
    """A registered plate (the pose is *registered*): *Go to* then needs no holder.

    ``plugin`` is the tool that wrote it and ``minute`` orders the starts: a later record of
    another tool is the one the page's plate comes from. ``plate`` and ``a1_center_xy_um`` are
    what it registers.
    """
    result = PlateResult(
        plate=plate,
        a1_center_xy_um=a1_center_xy_um,
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
            params={"plate": plate},
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
    title, lead, stand_region, preset_region, known, numbers, calibrate, cont = list(
        view.step_panel.objects
    )
    assert lead.object == SCREEN_LEADS["stand"]
    # D12: the stand line and the preset with its *Load* come first, then what is known and
    # the numbers of its calibration, then the calibration; the next step is not here, and each action's command is under
    # its own *Command* disclosure.
    assert title.object == "Which stand, and which preset?"
    assert view.stand_text in stand_region
    assert view.preset_select in _objects([preset_region])
    assert view.preset_load_button in _objects([preset_region])
    assert known is view.known_block
    # The numbers of the calibration in place, under what is known and outside the box that
    # hides when the objective needs no calibration (m5 §23.2 D2, #627).
    assert numbers is view.calibration_numbers
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
        # What Disconnect does is said once, here, while the stand is connected (D15).
        "Disconnect releases demo for NIS-Elements or ZEN.",
    ]
    # Calibrated at the objective in place: nothing to offer, and the line says when.
    seed_camera_calibration(run_doc, objective=TEN_X)
    view.on_refresh()
    assert not view.calibrate_box.visible
    # In the PC's own zone, to the minute (m5 §17.12 D20): the test asks the same clock.
    when = local_time(datetime(2026, 11, 9, 9, 30, tzinfo=timezone.utc))
    assert view.known_block.object.split("\n")[1] == (
        f"Camera: calibrated at {TEN_X} on {when}"
    )
    # Calibrated at another objective: offered again, and the line says both.
    seed_camera_calibration(run_doc, objective=FORTY_X)
    view.on_refresh()
    assert view.calibrate_box.visible
    assert view.known_block.object.split("\n")[1] == (
        f"Camera: calibrated at {FORTY_X} on {when}, not at {TEN_X}"
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
        if isinstance(o, pn.pane.Str)
        and o.styles.get("color") == _WARNING
        and o.visible
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
    # The answer is drawn on the row of the well, in its own words and the plain colour:
    # not the tool's whole outcome, which names assumptions that no card here shows.
    note = view.centre_slots[1].note
    assert note.object == "At A12."
    assert note.styles.get("color") != _WARNING
    assert view.centre_slots[0].note.object == ""
    assert view.message.object == ""


def test_a_reason_common_to_every_row_is_drawn_once_above_them(
    page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    # The owner's walk (finding 4): `goto-well is running: wait for it to end, or press STOP.`
    # four times under a drive, `End live to go to another well` three times in Live. A reason
    # that holds for every action of a screen is said once (m5 §17.12 D19), and a row says only
    # what is its own.
    _seed_plate(run_doc)
    fake_microscope.override(Camera, LiveCamera())
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    assert view.rows_note.object == ""
    assert not view.rows_note.visible

    _click(view.tool_runs["slow"])
    try:
        assert Slow.started.wait(5)
        view.tick()
        reason = "slow is running: wait for it to end, or press STOP."
        assert view.rows_note.object == reason
        assert view.rows_note.visible
        everything = [
            o.object
            for o in _objects(list(view.step_panel.objects))
            if isinstance(o, pn.pane.Str)
        ]
        assert sum(reason in text for text in everything) == 1
    finally:
        Slow.release.set()
    assert page.join(10.0)
    view.tick()
    assert view.rows_note.object == ""

    view.on_live()
    try:
        view.tick()
        everything = [
            o.object
            for o in _objects(list(view.step_panel.objects))
            if isinstance(o, pn.pane.Str)
        ]
        assert sum(GOTO_NOT_DURING_LIVE in text for text in everything) == 1
        assert view.rows_note.object == GOTO_NOT_DURING_LIVE
    finally:
        page.live(False)
    assert page.join(10.0)


def test_the_rows_note_says_what_register_uses_while_nothing_is_captured(
    fake_core: FakeCore, run_doc: Path
) -> None:
    # D23: the map says the plate is estimated, the rows say each in its own words, and one
    # line above them says what the button will do.
    shipped = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
    microscope = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.load(shipped)
    )
    page = PlatePage(microscope, run_doc, entry_points=_entry_points(StubGoto, Slow))
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        view.refresh()
        assert page.registers_from_sample
        assert view.rows_note.object == SAMPLE_REGISTERS
        assert view.rows_note.visible
        # A capture ends the sample's rows: the note goes with them.
        microscope.require(XYStage).move_to_um(*_well_centre("A1"))
        page.capture_centre("A1")
        view.refresh()
        assert not page.registers_from_sample
        assert view.rows_note.object == ""
    finally:
        page.close()
        microscope.close()


def test_a_row_answers_a_goto_in_its_own_words(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = make_page(BlockingGoto, Slow)
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    assert [slot.note.object for slot in view.centre_slots] == ["", "", ""]

    _click(view.centre_slots[2].goto_button)
    assert BlockingGoto.started.wait(5)
    view.tick()
    assert view.centre_slots[2].note.object == "Going to H1…"
    assert [slot.note.object for slot in view.centre_slots[:2]] == ["", ""]
    BlockingGoto.release.set()
    assert page.join(10.0)
    view.tick()
    assert view.centre_slots[2].note.object == "At H1."
    assert view.centre_slots[2].note.styles.get("color") != _WARNING
    # Another tab draws the same answer from the controller's, on the same row.
    other = PlatePageView(page)
    other.show_step("plate")
    other.tick()
    assert other.centre_slots[2].note.object == "At H1."


def test_a_far_capture_is_drawn_in_the_warning_colour_on_its_row(
    page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    # The owner's walk (finding 6): H1 captured at A12's place, 99 mm off, with no word.
    _seed_plate(run_doc)
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(*_well_centre("A1"))
    page.capture_centre("A1")
    stage.move_to_um(*_well_centre("A12"))
    page.capture_centre("H1")
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()

    a1, _a12, h1 = view.centre_slots
    assert not a1.capture_note.visible
    assert h1.capture_note.visible
    assert h1.capture_note.object.startswith("H1 was captured 117.")
    assert "Redo it there." in h1.capture_note.object
    assert h1.capture_note.styles["color"] == _WARNING
    # A note and never a refusal: the button stays as it was.
    assert not h1.capture_button.disabled

    # Redone where it belongs, the note goes.
    stage.move_to_um(*_well_centre("H1"))
    page.capture_centre("H1")
    view.refresh()
    assert not h1.capture_note.visible


def test_two_captures_at_one_place_are_noted_on_both_rows_of_an_unplaced_map(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # No plate registered and no holder: the map puts no well anywhere, so nothing is far
    # from where it is not, and only two captures at one point can be said (D16's pose rule).
    page.choose_plate("96-well")
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(1000.0, 2000.0)
    page.capture_centre("A1")
    stage.move_to_um(1010.0, 2000.0)
    page.capture_centre("A12")
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    assert view._view is not None
    assert view._view.pose.kind == "unplaced"
    a1, a12, h1 = view.centre_slots
    assert a1.capture_note.object == (
        "A1 and A12 were captured at the same place: redo the one that is wrong."
    )
    assert a12.capture_note.object == (
        "A12 and A1 were captured at the same place: redo the one that is wrong."
    )
    assert a12.capture_note.styles["color"] == _WARNING
    assert not h1.capture_note.visible


def test_register_shows_the_refusal_across_a_turn_before_the_click(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # The centres captured with the 10x, the turret turned to the 40x (m5 §17.12 D13, #373):
    # the button waits with the reason, the click that still arrives starts nothing, and the
    # outcome names both objectives.
    page.choose_plate("96-well")
    _capture(page, fake_microscope, "A1", "A12", "H1")
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    assert not view.register_button.disabled

    fake_microscope.require(ObjectiveTurret).select(FORTY_X, confirm=True)
    view.refresh()

    refusal = REGISTER_ACROSS_A_TURN.format(captured=TEN_X, in_place=FORTY_X)
    assert view.register_button.disabled
    assert view.register_wait.object == refusal
    assert view.register_wait.visible
    assert not page.run_doc.exists()
    _click(view.register_button)
    view.tick()
    assert page.status().running is None
    assert page.status().outcome == f"Registration: {refusal}"
    assert view.run_text.object.startswith(f"Registration: {refusal}")
    # Nothing was started or written.
    assert not page.run_doc.exists()
    # Turned back, the wait goes and the button is on.
    fake_microscope.require(ObjectiveTurret).select(TEN_X, confirm=True)
    view.refresh()
    assert not view.register_button.disabled
    assert view.register_wait.object == ""


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
    # The tool's one line, whole (FM-45) and without the exception's type, in the warning
    # colour: it did not end `done`. The drive bar says the outcome with the tool's title.
    note = view.centre_slots[0].note
    assert note.object == "goto-well: the stage did not arrive"
    assert note.styles["color"] == _WARNING
    assert view.goto_text.object == "Go to failed: goto-well: the stage did not arrive"
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
        # Said once, above the rows, and not under each of them (m5 §17.12 D19).
        assert view.rows_note.object.split("\n") == [GOTO_NOT_DURING_LIVE]
        assert view.rows_note.visible
        for slot in view.centre_slots:
            assert slot.goto_button.disabled
            assert slot.note.object == ""
        # A click that still arrives is the controller's refusal (FM-169): the state's, which
        # the note above the rows already says, and it leaves the drive's text as it was.
        _click(view.centre_slots[0].goto_button)
        assert page.status().goto == ""
        assert view.rows_note.object.split("\n") == [GOTO_NOT_DURING_LIVE]
        assert all(slot.note.object == "" for slot in view.centre_slots)
        assert view.message.object == ""
        assert StubGoto.ran == []
    finally:
        page.live(False)
    assert page.join(10.0)
    view.tick()
    assert not any(slot.goto_button.disabled for slot in view.centre_slots)
    assert GOTO_NOT_DURING_LIVE not in view.rows_note.object
    assert not view.rows_note.visible


def test_a_stale_go_to_row_click_under_live_says_the_same(
    page: PlatePage,
    run_doc: Path,
    fake_microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # m5 §23.2 D6: every caller of `goto` gets the refusal of the Live text, the rows' too. The
    # row's own screen says it once above the rows (`GOTO_NOT_DURING_LIVE`), so the click that
    # still arrives leaves the message alone, as it always did: the controller's text is what
    # the call raised, and nothing moved.
    _seed_plate(run_doc)
    fake_microscope.override(Camera, LiveCamera())
    view = PlatePageView(page)
    view.show_step("plate")
    view.refresh()
    raised: list[str] = []
    real_goto = page.goto

    def spy(well: str, side: str = "centre") -> None:
        try:
            real_goto(well, side)
        except RuntimeError as exc:
            raised.append(str(exc))
            raise

    monkeypatch.setattr(page, "goto", spy)
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    view.on_live()
    try:
        assert page.is_live
        view.tick()
        _click(view.centre_slots[0].goto_button)

        assert raised == [
            "live is running, so the stage did not move: press End live, switch Drive "
            "on and click the well again. The arrows work during Live."
        ]
        assert view.rows_note.object.split("\n") == [GOTO_NOT_DURING_LIVE]
        assert view.message.object == ""
        assert page.is_live
        assert stage.position_um() == before
        assert StubGoto.ran == []
    finally:
        page.live(False)
    assert page.join(10.0)


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
        # The reason common to every action of the screen is said once, above the rows; the
        # button's own wait is its own reason, and it has none (m5 §17.12 D19).
        assert view.rows_note.object == reason
        assert view.register_wait.object == ""
        assert not view.register_wait.visible
        assert view.calibrate_button.disabled
        assert view.calibrate_note.object == reason
        assert all(slot.goto_button.disabled for slot in view.centre_slots)
        assert all(slot.note.object == "" for slot in view.centre_slots)
        # Capture reads the stage, so it is disabled while another activity goes too.
        assert all(slot.capture_button.disabled for slot in view.centre_slots)
        _click(view.register_button)
        assert view.message.object == "slow is running"
        _click(view.centre_slots[0].goto_button)
        assert page.status().goto == ""
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
    # What runs is said once above the rows; the row says where it is going.
    assert view.rows_note.object == (
        "goto-well is running: wait for it to end, or press STOP."
    )
    assert note.object == "Going to A1…"
    assert page.status().goto == GOTO_ESTIMATED.format(well="A1")
    assert note.styles.get("color") != _WARNING
    BlockingGoto.release.set()
    assert page.join(10.0)
    view.tick()
    # At the estimate: centre it by eye (the row knows the drive went to the holder's estimate).
    assert note.object == (
        "At A1, the holder's estimate: centre it by eye with Live and the arrows, then "
        "capture."
    )
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
    # The refused click left the running drive's text: A1 is still where it is going.
    assert view.centre_slots[0].note.object == "Going to A1…"
    BlockingGoto.release.set()
    assert page.join(10.0)
    view.tick()
    assert view.centre_slots[0].note.object == "At A1."
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
    assert view.centre_slots[1].note.object == "At A12."
    # A drive armed on the map and clicked on a well: the answer is the click's, on the row of
    # the well it names (A1 is one of the three rows), and the drive bar says the outcome.
    view.drive_toggle.value = True
    view.on_tap(SimpleNamespace(x=0.0, y=0.0))
    assert page.join(10.0)
    view.tick()
    assert view.goto_text.object.startswith("Go to done in ")
    assert view.centre_slots[0].note.object == "At A1."
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
        (view.calibrate_command_note, view.calibrate_box.objects[3]),
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
    browser_text: Callable[[Any], str],
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
    # What the browser is sent for the strip's panes and for the known block is one `pre`
    # with those words in it (FM-271). At least one pane of the strip holds the failed
    # goto's words, so that the check cannot pass on panes that are empty.
    strip = [view.status_text, view.run_text, view.problems_text, view.message]
    holding = [pane for pane in strip if HOSTILE in pane.object]
    assert holding, [pane.object for pane in strip]
    for pane in strip:
        assert browser_text(pane) == (pane.object or " ")
    assert browser_text(view.known_block) == view.known_block.object
    # And every text pane of both screens is the page's escaping text pane: counted together,
    # so that a pane that becomes Markdown, or a bare `Str`, is a different count and not a pass.
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
    assert [type(o).__name__ for o in shown if type(o) is not TextPane] == []
    for pane in shown:
        assert browser_text(pane) == (pane.object or " ")


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
        assert page.status().outcome.startswith("Registration done in ")
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


def test_the_plate_screen_places_the_rows_note_every_rows_notes_and_the_replace_note_under_the_result(
    page: PlatePage, run_doc: Path
) -> None:
    # The layout, walked (FM-201: a widget that is built and set but never placed draws nothing, and no
    # test of its attributes sees it): a note that is not in the screen's list is a warning nobody reads.
    _seed_plate(run_doc)
    view = PlatePageView(page)
    view.show_step("plate")
    placed = {id(o) for o in _objects(list(view.step_panel.objects))}
    assert id(view.rows_note) in placed
    for slot in view.centre_slots:
        assert id(slot.note) in placed
        assert id(slot.capture_note) in placed
    # What the result's replacement note follows is the result line, not what is above it.
    screen = view.screens["plate"]
    assert screen.index(view.register_replaces) > screen.index(view.registered_block)
    assert screen.index(view.rows_note) < screen.index(view.centre_slots[0].box)


# -- the rows on a stand whose soft limits cut the corners (m5 §19, #512) ----------------


def test_the_plate_screen_suggests_wells_inside_the_stands_soft_limits(
    ti2e_page: PlatePage,
) -> None:
    page = ti2e_page
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("greiner-24-well")
    view.refresh()
    # The Ti2-E's A6 is 25 µm and its D1 500 µm outside the travel: *Go to* refuses both.
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to B2",
        "Go to B5",
        "Go to C2",
    ]
    assert [slot.capture_button.label for slot in view.centre_slots] == [
        "Capture the centre of B2",
        "Capture the centre of B5",
        "Capture the centre of C2",
    ]
    assert _row_texts(view) == [
        f"{well}: go there (estimated position), centre it by eye, capture"
        for well in ("B2", "B5", "C2")
    ]
    # The pane says why the rows are not the plate's far corners, in plain text, once.
    assert view.reach_note.visible
    assert view.reach_note.object == GREINER_CORNERS_NOTE
    assert page.suggested_centres() == ("B2", "B5", "C2")
    assert page.centres_reach_note() == GREINER_CORNERS_NOTE
    # A turn renames the wells in place and keeps the block: the same three names.
    page.choose_orientation("turned-180")
    view.refresh()
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to B2",
        "Go to B5",
        "Go to C2",
    ]
    assert view.reach_note.object == GREINER_CORNERS_NOTE


def test_a_click_on_a_row_goes_to_the_well_the_row_names(
    ti2e_page: PlatePage,
) -> None:
    # The row's *Go to* names B2, and the stub tool records the well it was run with.
    page = ti2e_page
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("greiner-24-well")
    view.refresh()
    _click(view.centre_slots[0].goto_button)
    assert page.join(10.0)
    assert StubGoto.ran == ["B2"]


def test_the_rows_keep_their_wells_after_a_capture_and_a_registered_plate(
    ti2e_page: PlatePage, ti2e_microscope: Microscope, run_doc: Path
) -> None:
    # D2: the estimate is the holder's, whatever the pose, so that the rows do not move
    # under the operator's hands when a capture or a registration changes the page's idea
    # of A1 by hundreds of µm.
    page = ti2e_page
    view = PlatePageView(page)
    view.show_step("plate")
    page.choose_plate("greiner-24-well")
    ti2e_microscope.require(XYStage).move_to_um(-28775.0, 9400.0)  # B2's estimate
    page.capture_centre("B2")
    view.refresh()
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to B2",
        "Go to B5",
        "Go to C2",
    ]
    assert _row_texts(view)[0].startswith("B2: captured at (-28775.0, 9400.0) µm")
    assert not view.other_captures.visible
    # A registered Greiner whose A1 is 300 µm off the holder's (-48 225, 28 850): the map is
    # registered now and its wells moved; the rows are the same three, and B2 is captured.
    _seed_plate(run_doc, plate="greiner-24-well", a1_center_xy_um=(-47925.0, 28850.0))
    view.refresh()
    assert view._view is not None
    assert view._view.pose.kind == "registered"
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to B2",
        "Go to B5",
        "Go to C2",
    ]
    assert _row_texts(view)[0].startswith("B2: captured")
    assert view.reach_note.object == GREINER_CORNERS_NOTE


def test_the_page_gives_the_holders_estimate_the_orientation_it_shows(
    fake_core: FakeCore, run_doc: Path
) -> None:
    # On the Ti2-E's travel a turn keeps the names, so the orientation cannot be seen there.
    # A 96-well centred on the origin in a travel that holds the left half only: the left
    # half is columns 1 to 6 not turned and 7 to 12 turned (the geometry's, typed out).
    base = Profile.demo()
    profile = base.model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=(0.0, 0.0), measured="a test's own centre"
            ),
            "safety": base.safety.model_copy(
                update={"xy_soft_limits_um": ((-60000.0, 0.0), (-40000.0, 40000.0))}
            ),
        }
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), profile)
    page = PlatePage(microscope, run_doc, entry_points=_entry_points(StubGoto, Slow))
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        page.choose_plate("96-well")
        view.refresh()
        assert page.suggested_centres() == ("A1", "A6", "H1")
        assert [slot.goto_button.label for slot in view.centre_slots] == [
            "Go to A1",
            "Go to A6",
            "Go to H1",
        ]
        page.choose_orientation("turned-180")
        view.refresh()
        assert page.suggested_centres() == ("A7", "A12", "H7")
        assert [slot.goto_button.label for slot in view.centre_slots] == [
            "Go to A7",
            "Go to A12",
            "Go to H7",
        ]
        assert page.centres_reach_note().startswith(
            "Not the plate's far corners (A1, A12 and H1)"
        )
    finally:
        page.close()
        microscope.close()


def test_the_reach_note_takes_no_height_where_nothing_is_cut(
    page: PlatePage, ti2e_page: PlatePage
) -> None:
    # No holder and no limits: today's rows, and no pane.
    plain = PlatePageView(page)
    plain.show_step("plate")
    page.choose_plate("96-well")
    plain.refresh()
    assert not plain.reach_note.visible
    assert plain.reach_note.object == ""
    assert [slot.goto_button.label for slot in plain.centre_slots] == [
        "Go to A1",
        "Go to A12",
        "Go to H1",
    ]
    # The Ti2-E's limits and a plate whose corners are all inside its travel: the same.
    view = PlatePageView(ti2e_page)
    view.show_step("plate")
    ti2e_page.choose_plate("6-well")
    view.refresh()
    assert not view.reach_note.visible
    assert view.reach_note.object == ""
    assert [slot.goto_button.label for slot in view.centre_slots] == [
        "Go to A1",
        "Go to A3",
        "Go to B1",
    ]


def test_the_reach_note_is_placed_after_the_rows_and_before_the_other_captures(
    ti2e_page: PlatePage,
) -> None:
    # FM-201: a widget built and set but not placed draws nothing, and no test of its
    # attributes would see it. Each screen's own list is walked.
    view = PlatePageView(ti2e_page)
    view.show_step("plate")
    placed = {id(o) for o in _objects(list(view.step_panel.objects))}
    assert id(view.reach_note) in placed
    screen = view.screens["plate"]
    assert screen.index(view.centre_slots[-1].box) < screen.index(view.reach_note)
    assert screen.index(view.reach_note) < screen.index(view.other_captures)


def test_a_profile_sample_of_the_plate_keeps_the_corners(
    fake_core: FakeCore, run_doc: Path
) -> None:
    # D7: the sample's own references are the plate's corners and a row then says `known from
    # the profile's sample`: the rows name the wells the references name, so a profile that
    # has both a [sample] of the page's plate and soft limits keeps the corners, and the note
    # (which would explain a cut) is empty.
    shipped = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
    profile = _ti2e_profile(Profile.load(shipped))
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), profile)
    page = PlatePage(microscope, run_doc, entry_points=_entry_points(StubGoto, Slow))
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        view.refresh()
        assert page.registers_from_sample
        assert page.suggested_centres() == ("A1", "A12", "H1")
        assert page.centres_reach_note() == ""
        assert _row_texts(view) == [
            f"{well}: known from the profile's sample, not measured"
            for well in ("A1", "A12", "H1")
        ]
        assert not view.reach_note.visible
        # The first capture ends `registers_from_sample`: the rows must not reshuffle then.
        microscope.require(XYStage).move_to_um(*_well_centre("A1"))
        page.capture_centre("A1")
        view.refresh()
        assert not page.registers_from_sample
        assert [slot.goto_button.label for slot in view.centre_slots] == [
            "Go to A1",
            "Go to A12",
            "Go to H1",
        ]
        assert not view.reach_note.visible
        # Another plate than the sample's is cut by the same limits, as on any stand.
        page.choose_plate("greiner-24-well")
        view.refresh()
        assert page.suggested_centres() == ("B2", "B5", "C2")
        assert view.reach_note.object == GREINER_CORNERS_NOTE
    finally:
        page.close()
        microscope.close()


# -- a calibration that failed, kept on the Stand screen (m5 §22, #625) -----------------------

#: What the real tool says over ``fake_microscope``, whose frame is flat: the reference frame is
#: judged before anything moves (m3 §4.2), so the words are the tool's own and need no stage.
FLAT_FRAME_WORDS = (
    "the camera's frame is dark or flat (the 1st to 99th percentile spans 0 counts, "
    "under 50): is the light on and the light path on the camera? nothing was moved"
)
LAST_FAILED = "The last calibration failed: "


def test_a_failed_calibration_from_the_stand_screen_stays_on_the_screen_in_the_warning_colour(
    page: PlatePage,
) -> None:
    """F14: the strip said it once and the screen kept nothing; the line stays under the button."""
    view = PlatePageView(page)
    assert not view.calibrate_failed.visible
    assert view.calibrate_failed.object == ""

    _click(view.calibrate_button)
    assert page.join(10.0)
    # The end of a run reads the run document again (the tick does it when it saw the run
    # going; this test did not tick in between), then the tick draws.
    view.refresh()
    view.tick()

    assert view.calibrate_failed.object == LAST_FAILED + FLAT_FRAME_WORDS
    assert view.calibrate_failed.visible
    assert view.calibrate_failed.styles["color"] == _WARNING
    # Between the note and the Command card, and the strip says what it always said, in its colour.
    assert view.calibrate_box.objects[:3] == [
        view.calibrate_button,
        view.calibrate_note,
        view.calibrate_failed,
    ]
    assert view.run_text.object == (
        f"Camera calibration failed: camera-calibration: {FLAT_FRAME_WORDS}"
    )
    assert view.run_text.styles.get("color") != _WARNING
    assert view.calibrate_box.visible
    assert view.known_block.object.split("\n")[1] == (
        "Camera: not calibrated (the arrows cannot say which way the image moves)"
    )

    # Refresh forgets the strip's outcome and leaves the line (D4).
    view.on_refresh()

    assert view.run_text.object == ""
    assert view.calibrate_failed.object == LAST_FAILED + FLAT_FRAME_WORDS
    assert view.calibrate_failed.visible


def _seed_failed_calibration(path: Path, words: str, *, minute: int = 0) -> None:
    """A ``camera-calibration`` record that ended ``failed``, as the runner writes its error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    then = T0 + timedelta(minutes=minute)
    write_stage(
        path,
        StageRecord(
            plugin="camera-calibration",
            provides="camera-calibration",
            attempt=f"a{minute}",
            status=StageStatus.failed,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            error=f"PluginRunError: camera-calibration: {words}",
        ),
    )


def test_a_calibration_that_ends_done_takes_the_failed_line_away(
    page: PlatePage,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    """D4: the line lives as long as the record, and a calibration that ends well replaces the record."""
    view = PlatePageView(page)
    _seed_failed_calibration(run_doc, "the light is off")
    view.on_refresh()
    assert view.calibrate_failed.object == LAST_FAILED + "the light is off"
    assert view.calibrate_failed.visible

    # Calibrated at the objective in place: the box is gone and so is the line.
    seed_camera_calibration(run_doc, objective=TEN_X)
    view.on_refresh()
    assert not view.calibrate_box.visible
    assert view.calibrate_failed.object == ""
    assert not view.calibrate_failed.visible

    # And calibrated at another objective: the box is offered again, with no failure to say.
    _seed_failed_calibration(run_doc, "the light is off", minute=1)
    view.on_refresh()
    assert view.calibrate_failed.object == LAST_FAILED + "the light is off"
    seed_camera_calibration(run_doc, objective=FORTY_X)
    view.on_refresh()
    assert view.calibrate_box.visible
    assert view.calibrate_failed.object == ""
    assert not view.calibrate_failed.visible


def test_the_failed_line_stays_through_a_disconnect_and_the_next_activity(
    page: PlatePage, run_doc: Path
) -> None:
    """D4: *Refresh*, a closed stand and the next activity leave it; the strip's outcome does not stay."""
    view = PlatePageView(page)
    _seed_failed_calibration(run_doc, "the light is off")
    view.on_refresh()
    line = LAST_FAILED + "the light is off"
    assert view.calibrate_failed.object == line

    page.disconnect()
    assert page.join(10.0)
    view.refresh()
    view.tick()

    assert not page.connected
    assert view.calibrate_failed.object == line
    assert view.calibrate_failed.visible


def test_a_record_that_cannot_be_read_gives_no_failed_line(
    page: PlatePage, run_doc: Path
) -> None:
    """FM-98: a run document that is not JSON, and a failed record with no error, say nothing and raise nothing."""
    view = PlatePageView(page)
    run_doc.write_text("{ this is not a run document", encoding="utf-8")
    view.on_refresh()
    assert view.calibrate_failed.object == ""
    assert not view.calibrate_failed.visible

    run_doc.unlink()
    write_stage(
        run_doc,
        StageRecord(
            plugin="camera-calibration",
            provides="camera-calibration",
            attempt="a0",
            status=StageStatus.failed,
            started_utc=T0,
            finished_utc=T0,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            error=None,
        ),
    )
    view.on_refresh()
    assert view.calibrate_failed.object == ""
    assert not view.calibrate_failed.visible


def test_the_failed_line_is_plain_text_and_bounded(
    page: PlatePage, run_doc: Path, browser_text: Callable[[Any], str]
) -> None:
    """FM-205: markup, a path and 700 characters in a tool's words; the line is one text pane, on one line, cut.

    The pane is the page's escaping text pane and the words pass as given. What a browser
    shows is not proved by a pane's type (a bare `Str` is parsed as HTML, FM-271, read in
    Chromium): ``browser_text`` reads the model the browser is sent, and it holds one `pre`
    with the markup as words.
    """
    words = f"{HOSTILE} in C:\\Users\\lab\\run\nand " + "x" * 700
    _seed_failed_calibration(run_doc, words)
    view = PlatePageView(page)

    view.on_refresh()

    assert type(view.calibrate_failed) is TextPane
    shown = view.calibrate_failed.object
    assert browser_text(view.calibrate_failed) == shown
    assert shown.startswith(f"{LAST_FAILED}{HOSTILE} in C:\\Users\\lab\\run and xxx")
    assert "\n" not in shown
    # The tool's words are cut at the card's bound, as the Plate screen's block cuts them.
    assert len(shown) == len(LAST_FAILED) + 601
    assert shown.endswith("…")


def _seed_walls_plate(run_doc: Path, objective: str) -> None:
    """A plate registered from images of its walls under ``objective``: the block is offered."""

    def fit(well: str, centre: tuple[float, float]) -> WellFit:
        return WellFit(
            well=well,
            recorded_xy_um=centre,
            center_xy_um=centre,
            diameter_um=6000.0,
            size_error=0.0,
            rms_um=0.1,
            n_points=100,
            walls=(),
        )

    walls = WallImages(
        objective=objective,
        pixel_size_um=1.1,
        image_shape_px=(100, 100),
        search_um=1000.0,
        wells=(fit("A1", (0.0, 0.0)), fit("H12", (99000.0, -63000.0))),
        frames=8,
        folder=None,
        elapsed_s=10.0,
    )
    result = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="wall-images",
        residual_um=1.0,
        scale_error=None,
        n_refs=2,
        walls=walls,
        objective=objective,
    )
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-walls",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=T0,
            finished_utc=T0,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={"plate": "96-well"},
            result=result.model_dump(mode="json"),
        ),
    )


def test_a_routed_click_reads_the_failure_of_the_position_document_not_the_run_document(
    page: PlatePage, run_doc: Path
) -> None:
    """D2: the click goes into the objective's own document when the plate was registered under another.

    The 10X is in place and the plate was registered under the 40X: the click is routed
    into ``calibrations/position-2.json`` (the 10X is position 2 of the fake turret), so the
    failure the line says is that document's, never the run document's own record of an
    older calibration. A problem of that document that is not a failure (no file, not
    JSON) is the Plate screen's block's, and gives no line here.
    """
    view = PlatePageView(page)
    _seed_walls_plate(run_doc, FORTY_X)
    _seed_failed_calibration(run_doc, "THE RUN DOCUMENT'S WORDS")
    folder = run_doc.parent / "calibrations"
    held = folder / "position-2.json"

    # Routed, with no document at the position: nothing of the run document's is said.
    view.on_refresh()
    assert page.status().objective == TEN_X
    assert view._stand_destination() == (True, 2)
    assert view.calibrate_failed.object == ""
    assert not view.calibrate_failed.visible

    # The position's own document failed: its words, whatever the run document says.
    _seed_failed_calibration(held, "THE POSITION'S WORDS")
    view.on_refresh()
    assert view.calibrate_failed.object == LAST_FAILED + "THE POSITION'S WORDS"
    assert view.calibrate_failed.visible
    assert view.calibrate_failed.styles["color"] == _WARNING

    # A document that is not one: the block says why, this line says nothing.
    held.write_text("{ this is not a run document", encoding="utf-8")
    view.on_refresh()
    assert view.calibrate_failed.object == ""
    assert not view.calibrate_failed.visible

    # Registered under the objective in place the click is not routed: the run document's words.
    _seed_walls_plate(run_doc, TEN_X)
    view.on_refresh()
    assert view._stand_destination() == (False, None)
    assert view.calibrate_failed.object == LAST_FAILED + "THE RUN DOCUMENT'S WORDS"
