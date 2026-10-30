"""The plate page's Drive region, without a browser (m5 §6.4, #263).

``button.clicks += 1`` is what a click raises, as in ``test_plate_page.py``.
Measured on Panel 1.10.0 and 1.9.4: a click on a *disabled* button still calls its
handler on the server, so the controller's refusals are the guard and a test
clicks a disabled button to prove it. Panel's deprecation warnings are errors here
(FM-147).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel  # noqa: F401  # CI must fail, not skip, when the extra is missing
else:
    pytest.importorskip("panel")

import numpy as np
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    LightSource,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.plate_page import NO_LABEL, PlatePageView

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

TEN_X = "Nikon 10X S Fluor"
TWENTY_X = "Nikon 20X Plan Fluor ELWD"
FORTY_X = "Nikon 40X Plan Fluor ELWD"
SHUTTER_LIGHT = "White Light Shutter"
CONFIRM_TEXT = (
    "Turn the turret to '{}'? A turn can drive an objective into a loaded plate: "
    "check that the stage is clear. The autofocus is switched off and not engaged "
    "again."
)


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class SlowScan:
    """Registered as ``scan``: runs until released or stopped."""

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


class FrameCamera:
    """The six methods of ``Camera``, giving one 6 x 10 frame at once."""

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


class StubTurret:
    """A turret with the labels it is given; ``selected`` are the turns it was asked."""

    def __init__(self, labels: list[str], current: str) -> None:
        self._labels = labels
        self._current = current
        self.selected: list[tuple[str, bool]] = []

    def labels(self) -> list[str]:
        return list(self._labels)

    def current(self) -> str:
        return self._current

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.selected.append((label, confirm))
        self._current = label
        return label

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0


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


def _stand(profile: Profile | None = None) -> Microscope:
    """The fake demo stand with the White Light Shutter as its light: every control usable."""
    if profile is None:
        profile = Profile.demo().model_copy(deep=True)
        profile.roles.assign[Role.light_source] = SHUTTER_LIGHT
    return Microscope.from_core(cast("CMMCorePlus", FakeCore.demo_like()), profile)


@pytest.fixture
def stand() -> Iterator[Microscope]:
    microscope = _stand()
    try:
        yield microscope
    finally:
        microscope.close()


def _controller(microscope: Microscope, run_doc: Path) -> PlatePage:
    scan = EntryPoint(name="scan", value=f"{__name__}:SlowScan", group=GROUP)
    return PlatePage(microscope, run_doc, entry_points=[scan])


@pytest.fixture
def page(stand: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = _controller(stand, run_doc)
    try:
        yield controller
    finally:
        controller.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _click(button: Any) -> None:
    button.clicks += 1


def _drive_controls(view: PlatePageView) -> dict[str, Any]:
    return {
        **{f"jog {k}": b for k, b in view.jog_buttons.items()},
        **{f"z {k}": b for k, b in view.z_buttons.items()},
        "autofocus on": view.autofocus_on_button,
        "autofocus off": view.autofocus_off_button,
        "objective": view.objective_button,
        "light on": view.light_on_button,
        "light off": view.light_off_button,
    }


def _disabled(view: PlatePageView) -> dict[str, bool]:
    return {name: bool(w.disabled) for name, w in _drive_controls(view).items()}


def _wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, f"{what} not seen within {timeout_s} s"
        time.sleep(0.005)


# --- disabled, and why -------------------------------------------------------------


def test_the_drive_region_disables_each_control_with_the_reason_its_stand_lacks(
    stand: Microscope, run_doc: Path, fake_microscope: Microscope
) -> None:
    lacking = {XYStage, ZStage, Autofocus, ObjectiveTurret, LightSource}
    with _replaced(stand, "has", lambda capability: capability not in lacking):
        page = _controller(stand, run_doc)
        view = PlatePageView(page)
        assert all(_disabled(view).values()), _disabled(view)
        assert view.drive_note.object == (
            "pad: this stand has no XYStage\n"
            "focus: this stand has no ZStage\n"
            "autofocus: this stand has no Autofocus\n"
            "objective: this stand has no ObjectiveTurret\n"
            "light: this stand has no LightSource"
        )
        page.close()
    # A stand with every capability: nothing disabled, nothing to say.
    view = PlatePageView(_controller(stand, run_doc))
    assert not any(_disabled(view).values()), _disabled(view)
    assert view.drive_note.object == ""
    # The demo's LED has no on/off that smc knows: both light buttons, one note.
    led = PlatePageView(_controller(fake_microscope, run_doc))
    assert led.drive_note.object == (
        "light: the light source LED has no on/off that smc knows"
    )
    assert [name for name, off in _disabled(led).items() if off] == [
        "light on",
        "light off",
    ]


def test_a_turret_whose_positions_cannot_be_read_is_disabled_and_says_so(
    stand: Microscope, run_doc: Path
) -> None:
    turret = stand.require(ObjectiveTurret)

    def failing() -> list[str]:
        raise RuntimeError("the turret did not answer")

    with _replaced(turret, "labels", failing):
        view = PlatePageView(_controller(stand, run_doc))
    assert view.objective_button.disabled
    assert view.drive_note.object == "objective: its positions could not be read"


def test_drive_controls_are_disabled_while_a_run_goes_and_enabled_while_live_goes(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    view.objective_select.value = TWENTY_X
    _click(view.objective_button)
    assert view.objective_confirm.visible
    _click(view.tool_runs["scan"])
    try:
        assert SlowScan.started.wait(5)
        view.tick()
        assert all(_disabled(view).values()), _disabled(view)
        # A tick that finds the controls disabled disarms the confirmation.
        assert not view.objective_confirm.visible
    finally:
        SlowScan.release.set()
    assert page.join(10.0)
    view.tick()
    assert not any(_disabled(view).values()), _disabled(view)
    stand.override(Camera, FrameCamera())
    view.on_live()
    try:
        assert page.is_live
        view.tick()
        # Live goes: the drive is what live is for (focusing, finding a field).
        assert not any(_disabled(view).values()), _disabled(view)
        assert view.snap_button.disabled
    finally:
        page.live(False)
    assert page.join(10.0)


# --- the pad and the focus -----------------------------------------------------------


def test_an_arrow_jogs_by_the_chosen_step_in_the_stages_axes(
    page: PlatePage, stand: Microscope
) -> None:
    # FM-73: X and Y are never swapped, and each sign is the stage's own.
    view = PlatePageView(page)
    view.xy_step_select.value = 10.0
    _click(view.jog_buttons["right"])
    assert page.join(10.0)
    view.xy_step_select.value = 100.0
    _click(view.jog_buttons["up"])
    assert page.join(10.0)
    assert stand.require(XYStage).position_um() == XY(10.0, 100.0)
    view.xy_step_select.value = 1.0
    _click(view.jog_buttons["left"])
    assert page.join(10.0)
    view.xy_step_select.value = 1000.0
    _click(view.jog_buttons["down"])
    assert page.join(10.0)
    assert stand.require(XYStage).position_um() == XY(9.0, -900.0)
    assert page.status().outcome == "stage jog: XY (9.00, -900.00) µm"
    assert view.message.object == ""


def test_z_buttons_move_the_focus_by_the_chosen_step(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    view.z_step_select.value = 5.0
    _click(view.z_buttons["up"])
    assert page.join(10.0)
    view.z_step_select.value = 0.5
    _click(view.z_buttons["down"])
    assert page.join(10.0)
    assert stand.require(ZStage).position_um() == 4.5
    assert page.status().outcome == "z jog: Z 4.50 µm"


def test_the_default_steps_are_100_and_1_um(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.xy_step_select.value == 100.0
    assert view.z_step_select.value == 1.0
    assert view.xy_step_select.options == {
        "1 µm": 1.0,
        "10 µm": 10.0,
        "100 µm": 100.0,
        "1000 µm": 1000.0,
    }
    assert view.z_step_select.options == {
        "0.5 µm": 0.5,
        "1 µm": 1.0,
        "5 µm": 5.0,
        "10 µm": 10.0,
        "50 µm": 50.0,
    }


@pytest.mark.parametrize(
    ("max_jog", "max_z", "xy_options", "xy_value", "z_options", "z_value"),
    [
        (
            50.0,
            3.0,
            {"1 µm": 1.0, "10 µm": 10.0},
            10.0,
            {"0.5 µm": 0.5, "1 µm": 1.0},
            1.0,
        ),
        (0.5, 0.2, {"1 µm": 1.0}, 1.0, {"0.5 µm": 0.5}, 0.5),
    ],
    ids=["under-some", "under-every"],
)
def test_steps_never_exceed_the_profiles_jog_limits(
    run_doc: Path,
    max_jog: float,
    max_z: float,
    xy_options: dict[str, float],
    xy_value: float,
    z_options: dict[str, float],
    z_value: float,
) -> None:
    profile = Profile.demo().model_copy(deep=True)
    profile.safety.max_jog_um = max_jog
    profile.safety.max_z_jog_um = max_z
    microscope = _stand(profile)
    try:
        view = PlatePageView(_controller(microscope, run_doc))
        assert view.xy_step_select.options == xy_options
        assert view.xy_step_select.value == xy_value
        assert view.z_step_select.options == z_options
        assert view.z_step_select.value == z_value
    finally:
        microscope.close()


def test_an_unknown_direction_is_a_message_and_moves_nothing(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    view.on_jog("north")
    assert view.message.object == "no jog 'north'; up, left, right or down"
    view.on_z("sideways")
    assert view.message.object == "no Z jog 'sideways'; up or down"
    assert page.join(10.0)
    assert stand.require(XYStage).position_um() == XY(0.0, 0.0)


# --- the objective's confirmation ------------------------------------------------------


def test_an_objective_change_turns_only_after_the_confirm_button(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    assert view.objective_select.value == TEN_X
    assert not view.objective_confirm.visible
    view.objective_select.value = TWENTY_X

    _click(view.objective_button)

    assert view.objective_confirm.visible
    assert view.objective_confirm_text.object == CONFIRM_TEXT.format(TWENTY_X)
    assert page.join(10.0)
    assert stand.require(ObjectiveTurret).current() == TEN_X

    _click(view.objective_confirm_button)
    assert page.join(10.0)

    assert stand.require(ObjectiveTurret).current() == TWENTY_X
    assert not view.objective_confirm.visible
    assert page.status().outcome == f"objective set: objective '{TWENTY_X}'"


def test_cancel_disarms_and_turns_nothing(page: PlatePage, stand: Microscope) -> None:
    view = PlatePageView(page)
    view.objective_select.value = TWENTY_X
    _click(view.objective_button)
    _click(view.objective_cancel_button)
    assert not view.objective_confirm.visible
    _click(view.objective_confirm_button)
    assert view.message.object == "choose the objective again"
    assert page.join(10.0)
    assert stand.require(ObjectiveTurret).current() == TEN_X


def test_changing_the_objective_choice_disarms_the_confirmation(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    view.objective_select.value = TWENTY_X
    _click(view.objective_button)
    assert view.objective_confirm.visible

    view.objective_select.value = FORTY_X

    assert not view.objective_confirm.visible
    # A stale browser that still shows the confirm button: the armed label is gone.
    _click(view.objective_confirm_button)
    assert view.message.object == "choose the objective again"
    assert page.join(10.0)
    assert stand.require(ObjectiveTurret).current() == TEN_X


def test_a_confirm_click_with_nothing_armed_turns_nothing(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    threads = threading.active_count()

    _click(view.objective_confirm_button)

    assert view.message.object == "choose the objective again"
    assert threading.active_count() == threads
    assert stand.require(ObjectiveTurret).current() == TEN_X


def test_the_tick_never_moves_the_objective_choice(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    view.objective_select.value = FORTY_X
    view.tick()
    assert view.objective_select.value == FORTY_X
    stand.require(ObjectiveTurret).select(TWENTY_X, confirm=True)
    view.tick()
    assert view.objective_select.value == FORTY_X


def test_an_empty_turret_label_is_offered_as_no_label(
    stand: Microscope, run_doc: Path
) -> None:
    turret = StubTurret(["", TEN_X], current=TEN_X)
    stand.override(ObjectiveTurret, turret)
    page = _controller(stand, run_doc)
    try:
        view = PlatePageView(page)
        assert view.objective_select.options == {NO_LABEL: "", TEN_X: TEN_X}
        view.objective_select.value = ""
        _click(view.objective_button)
        assert view.objective_confirm_text.object == CONFIRM_TEXT.format("")
        _click(view.objective_confirm_button)
        assert page.join(10.0)
        assert turret.selected == [("", True)]
        assert page.status().outcome == "objective set: objective ''"
    finally:
        page.close()


# --- the guard is the controller, not the button -----------------------------------------


def test_a_click_on_a_disabled_drive_button_is_refused_by_the_controller(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    _click(view.tool_runs["scan"])
    try:
        assert SlowScan.started.wait(5)
        view.tick()
        assert view.jog_buttons["right"].disabled
        # Panel still calls the handler of a disabled button (measured).
        _click(view.jog_buttons["right"])
        assert view.message.object == "scan is running"
        _click(view.z_buttons["up"])
        assert view.message.object == "scan is running"
        _click(view.light_on_button)
        assert view.message.object == "scan is running"
    finally:
        SlowScan.release.set()
    assert page.join(10.0)
    assert stand.require(XYStage).position_um() == XY(0.0, 0.0)
    assert stand.require(ZStage).position_um() == 0.0


# --- the other sessions --------------------------------------------------------------------


def test_a_jog_in_one_view_shows_on_another_views_strip_after_its_tick(
    page: PlatePage,
) -> None:
    # FM-151: the controller is the server's, and every tab reads the same stand.
    one = PlatePageView(page)
    other = PlatePageView(page)
    one.xy_step_select.value = 10.0
    _click(one.jog_buttons["right"])
    assert page.join(10.0)

    other.tick()

    assert "XY (10.0, 0.0) µm" in other.status_text.object.split(" | ")
    assert other.run_text.object == "stage jog: XY (10.00, 0.00) µm"


def test_light_buttons_switch_the_light(page: PlatePage, stand: Microscope) -> None:
    # FM-47: the state the test starts from is set.
    cast("FakeCore", stand.core).setShutterOpen(SHUTTER_LIGHT, False)
    view = PlatePageView(page)
    _click(view.light_on_button)
    assert page.join(10.0)
    assert stand.require(LightSource).is_on() is True
    _click(view.light_off_button)
    assert page.join(10.0)
    assert stand.require(LightSource).is_on() is False
    assert page.status().outcome == "light off: light off"


def test_autofocus_buttons_engage_and_disengage(
    page: PlatePage, stand: Microscope
) -> None:
    view = PlatePageView(page)
    _click(view.autofocus_on_button)
    assert page.join(10.0)
    assert page.status().outcome == "autofocus engage: autofocus locked"
    _click(view.autofocus_off_button)
    assert page.join(10.0)
    assert page.status().outcome == "autofocus disengage: autofocus off"
    assert not stand.require(Autofocus).is_engaged()


def test_a_jog_during_live_reaches_the_stage_from_the_page(
    page: PlatePage, stand: Microscope
) -> None:
    stand.override(Camera, FrameCamera())
    view = PlatePageView(page)
    view.on_live()
    try:
        view.xy_step_select.value = 10.0
        _click(view.jog_buttons["right"])
        assert view.message.object == ""
        _wait_for(
            lambda: page.status().outcome == "stage jog: XY (10.00, 0.00) µm",
            "the jog's outcome",
        )
        assert page.is_live
    finally:
        page.live(False)
    assert page.join(10.0)


_GROUPS: list[tuple[type, str, Callable[[PlatePageView], list[Any]]]] = [
    (
        XYStage,
        "pad: this stand has no XYStage",
        lambda v: [*v.jog_buttons.values(), v.xy_step_select],
    ),
    (
        ZStage,
        "focus: this stand has no ZStage",
        lambda v: [*v.z_buttons.values(), v.z_step_select],
    ),
    (
        Autofocus,
        "autofocus: this stand has no Autofocus",
        lambda v: [v.autofocus_on_button, v.autofocus_off_button],
    ),
    (
        ObjectiveTurret,
        "objective: this stand has no ObjectiveTurret",
        lambda v: [v.objective_button, v.objective_select],
    ),
    (
        LightSource,
        "light: this stand has no LightSource",
        lambda v: [v.light_on_button, v.light_off_button],
    ),
]


@pytest.mark.parametrize(
    ("missing", "note", "widgets"), _GROUPS, ids=[g[0].__name__ for g in _GROUPS]
)
def test_a_stand_lacking_one_capability_disables_that_group_and_no_other(
    stand: Microscope,
    run_doc: Path,
    missing: type,
    note: str,
    widgets: Callable[[PlatePageView], list[Any]],
) -> None:
    # FM-43: the table of groups is checked row by row, not all at once.
    with _replaced(stand, "has", lambda capability: capability is not missing):
        page = _controller(stand, run_doc)
        try:
            view = PlatePageView(page)
            lacking = widgets(view)
            everything = [
                *_drive_controls(view).values(),
                view.xy_step_select,
                view.z_step_select,
                view.objective_select,
            ]
            assert [w for w in everything if w.disabled] == lacking
            assert view.drive_note.object == note
        finally:
            page.close()
