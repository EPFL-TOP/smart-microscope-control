"""The plate page's drive actions on ``fake_microscope``, with and without live (m5 §6.3, #263).

No Panel here. A drive action is the same ``smc.drive`` call as its command,
run as an activity of its own, or handed to the live worker and run between two
frames. Texts are asserted whole (FM-45); threads meet on ``Event`` and every
wait is bounded (FM-44); a frame is asserted by what one taken *after* the
outcome shows, never by a number fixed in advance (FM-96), and a blocked snap is
released in a ``finally``.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus

import smc.drive as drive_module
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
from smc.hardware.errors import HardwareError
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.hardware.safety import DEFAULT_MAX_Z_JOG_UM
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.testing import FakeCore
from smc.ui.controller import PlatePage

TEN_X = "Nikon 10X S Fluor"
TWENTY_X = "Nikon 20X Plan Fluor ELWD"
SHUTTER_LIGHT = "White Light Shutter"


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class Slow:
    """Runs until released or stopped."""

    name = provides = "slow"
    summary = "A test plugin that waits."
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


class FrameCamera:
    """The six methods of ``Camera``, giving one frame; every wait is bounded (FM-44).

    ``entered`` is set when a snap starts; with ``block`` the snap waits for
    ``release``, as a driver in the middle of an exposure does.
    """

    def __init__(
        self,
        *,
        block: bool = False,
        fail_with: Callable[[], Exception] | None = None,
    ) -> None:
        self.frame = np.arange(60, dtype=np.uint16).reshape(6, 10)
        self._block = block
        self._fail_with = fail_with
        self.entered = threading.Event()
        self.release = threading.Event()

    def snap(self) -> np.ndarray:
        self.entered.set()
        if self._block:
            assert self.release.wait(10), "the test never released the snap"
        if self._fail_with is not None:
            raise self._fail_with()
        return self.frame

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


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _page(microscope: Microscope, run_doc: Path) -> PlatePage:
    slow = EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP)
    return PlatePage(microscope, run_doc, entry_points=[slow])


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = _page(fake_microscope, run_doc)
    try:
        yield controller
    finally:
        controller.close()


@pytest.fixture
def shutter_light() -> Iterator[Microscope]:
    """The fake demo stand with the White Light Shutter as its light source: one that switches."""
    profile = Profile.demo().model_copy(deep=True)
    profile.roles.assign[Role.light_source] = SHUTTER_LIGHT
    microscope = Microscope.from_core(
        cast("CMMCorePlus", FakeCore.demo_like()), profile
    )
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def light_page(shutter_light: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = _page(shutter_light, run_doc)
    try:
        yield controller
    finally:
        controller.close()


def _fake(microscope: Microscope) -> FakeCore:
    return cast("FakeCore", microscope.core)


def _refuses(error: type[Exception], text: str) -> AbstractContextManager[Any]:
    """``pytest.raises`` that insists on the whole message (FM-45)."""
    return pytest.raises(error, match=f"^{re.escape(text)}$")


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _without(missing: type) -> Callable[[type], bool]:
    def has(capability: type) -> bool:
        return capability is not missing

    return has


def _wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, f"{what} not seen within {timeout_s} s"
        time.sleep(0.005)


def _wait_for_outcome(page: PlatePage, outcome: str) -> None:
    _wait_for(
        lambda: page.status().outcome == outcome,
        f"the outcome {outcome!r} (last: {page.status().outcome!r})",
    )


def _wait_for_frame_after(page: PlatePage, number: int) -> Any:
    _wait_for(
        lambda: (frame := page.frame()) is not None and frame.number > number,
        f"a frame after {number}",
    )
    return page.frame()


def _refusal(call: Callable[[], object]) -> str:
    """``<Type>: <message on one line>`` of what ``call`` raises: the facade's own words."""
    with pytest.raises(HardwareError) as raised:
        call()
    text = f"{type(raised.value).__name__}: {' '.join(str(raised.value).split())}"
    del raised
    return text


def _on_thread(
    monkeypatch: pytest.MonkeyPatch, name: str, page: PlatePage
) -> list[tuple[str, bool]]:
    """Wrap ``smc.drive.<name>``: each call records its thread and whether it could take ``_lock``.

    ``Lock.locked()`` does not say which thread holds it (the thread that starts an
    activity holds it while the worker starts): the action takes it with a bound,
    which it can only fail to do when its own thread holds it across the call.
    """
    seen: list[tuple[str, bool]] = []
    original = getattr(drive_module, name)

    def recording(*args: Any, **kwargs: Any) -> Any:
        took = page._lock.acquire(timeout=2.0)
        if took:
            page._lock.release()
        seen.append((threading.current_thread().name, took))
        return original(*args, **kwargs)

    monkeypatch.setattr(drive_module, name, recording)
    return seen


# --- without live ------------------------------------------------------------------

_ACTIONS: list[tuple[str, str, Callable[[PlatePage], None], str]] = [
    ("stage jog", "jog_xy", lambda p: p.jog_xy(10.0, -20.0), "XY (10.00, -20.00) µm"),
    ("z jog", "jog_z", lambda p: p.jog_z(2.5), "Z 2.50 µm"),
    (
        "objective set",
        "set_objective",
        lambda p: p.objective(TWENTY_X, confirm=True),
        f"objective '{TWENTY_X}'",
    ),
    (
        "autofocus engage",
        "engage_autofocus",
        lambda p: p.autofocus(True),
        "autofocus locked",
    ),
    (
        "autofocus disengage",
        "disengage_autofocus",
        lambda p: p.autofocus(False),
        "autofocus off",
    ),
    ("light on", "switch_light", lambda p: p.light(True), "light on"),
    ("light off", "switch_light", lambda p: p.light(False), "light off"),
]


@pytest.mark.parametrize(
    ("name", "function", "act", "text"), _ACTIONS, ids=[a[0] for a in _ACTIONS]
)
def test_each_drive_action_runs_its_commands_call_on_the_worker(
    light_page: PlatePage,
    shutter_light: Microscope,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    function: str,
    act: Callable[[PlatePage], None],
    text: str,
) -> None:
    if name == "light off":
        _fake(shutter_light).setShutterOpen(SHUTTER_LIGHT, True)
    else:
        # FM-47: the state a light test starts from is set, never assumed.
        _fake(shutter_light).setShutterOpen(SHUTTER_LIGHT, False)
    seen = _on_thread(monkeypatch, function, light_page)

    act(light_page)
    assert light_page.join(10.0)

    assert light_page.status().outcome == f"{name}: {text}"
    # The same smc.drive function as the command's, once, on the activity's worker,
    # and with no lock held across it (FM-70).
    assert seen == [(f"smc-ui-{name}", True)]


def test_a_drive_action_is_refused_while_a_run_or_a_snap_goes(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    calls: list[Callable[[], None]] = [
        lambda: page.jog_xy(10.0, 0.0),
        lambda: page.jog_z(1.0),
        lambda: page.objective(TWENTY_X, confirm=True),
        lambda: page.autofocus(True),
        lambda: page.autofocus(False),
        lambda: page.light(True),
        lambda: page.light(False),
    ]
    page.run("slow", [])
    try:
        assert Slow.started.wait(5)
        for call in calls:
            with _refuses(RuntimeError, "slow is running"):
                call()
    finally:
        Slow.release.set()
    assert page.join(10.0)
    camera = FrameCamera(block=True)
    fake_microscope.override(Camera, camera)
    page.snap()
    try:
        assert camera.entered.wait(5)
        for call in calls:
            with _refuses(RuntimeError, "snap is running"):
                call()
    finally:
        camera.release.set()
    assert page.join(10.0)
    state = fake_microscope.state()
    assert state.xy == XY(0.0, 0.0)
    assert state.z_um == 0.0
    assert state.objective == TEN_X


@pytest.mark.parametrize(
    ("capability", "act", "outcome"),
    [
        (XYStage, lambda p: p.jog_xy(1.0, 0.0), "stage jog: this stand has no XYStage"),
        (ZStage, lambda p: p.jog_z(1.0), "z jog: this stand has no ZStage"),
        (
            ObjectiveTurret,
            lambda p: p.objective(TWENTY_X, confirm=True),
            "objective set: this stand has no ObjectiveTurret",
        ),
        (
            Autofocus,
            lambda p: p.autofocus(True),
            "autofocus engage: this stand has no Autofocus",
        ),
        (
            LightSource,
            lambda p: p.light(False),
            "light off: this stand has no LightSource",
        ),
    ],
    ids=["xy", "z", "turret", "autofocus", "light"],
)
def test_a_drive_action_on_a_stand_without_its_capability_is_an_outcome_and_starts_no_thread(
    page: PlatePage,
    fake_microscope: Microscope,
    capability: type,
    act: Callable[[PlatePage], None],
    outcome: str,
) -> None:
    threads = threading.active_count()
    before = page.activities_ended
    with _replaced(fake_microscope, "has", _without(capability)):
        act(page)
    status = page.status()
    assert status.outcome == outcome
    assert status.running is None
    assert page.activities_ended == before
    assert threading.active_count() == threads


def test_light_lacks_names_a_light_source_without_on_off(
    page: PlatePage, fake_microscope: Microscope, light_page: PlatePage
) -> None:
    # The demo's LED is a State device: no on/off that smc knows.
    lack = "the light source LED has no on/off that smc knows"
    assert page.light_lacks() == lack
    threads = threading.active_count()
    page.light(True)
    assert page.status().outcome == f"light on: {lack}"
    assert threading.active_count() == threads
    # Off is never held back: the facade's own error says to use the shutter.
    page.light(False)
    assert page.join(10.0)
    assert page.status().outcome == (
        "light off: HardwareError: light_source 'LED' (a State device) has no on/off "
        "that smc knows; switch the light with its shutter (Shutter)"
    )
    assert light_page.light_lacks() is None
    with _replaced(fake_microscope, "has", _without(LightSource)):
        assert _page(fake_microscope, page._run_doc).light_lacks() == (
            "this stand has no LightSource"
        )


def test_light_lacks_is_none_when_the_read_fails_and_is_read_again(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    light = fake_microscope.require(LightSource)
    reads: list[int] = []

    def failing() -> bool | None:
        reads.append(1)
        raise HardwareError("the light did not answer")

    with _replaced(light, "is_on", failing):
        assert page.light_lacks() is None
    # Not kept: a read that raised is no answer.
    assert page.light_lacks() == "the light source LED has no on/off that smc knows"
    with _replaced(light, "is_on", failing):
        # Kept now: the device's type does not change while the stand is held.
        assert page.light_lacks() == "the light source LED has no on/off that smc knows"
    assert len(reads) == 1


def test_objective_without_confirm_is_refused_by_the_layer(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    refusal = _refusal(
        lambda: fake_microscope.require(ObjectiveTurret).select(TWENTY_X)
    )

    page.objective(TWENTY_X, confirm=False)
    assert page.join(10.0)

    assert page.status().outcome == f"objective set: {refusal}"
    assert refusal.startswith("SafetyRefusedError: turning the objective turret")
    assert fake_microscope.require(ObjectiveTurret).current() == TEN_X


def test_objective_labels_are_the_turrets_and_empty_without_one(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    assert page.objective_labels() == tuple(
        fake_microscope.require(ObjectiveTurret).labels()
    )
    assert TWENTY_X in page.objective_labels()
    with _replaced(fake_microscope, "get", lambda capability: None):
        assert page.objective_labels() == ()


def test_objective_labels_that_cannot_be_read_are_empty_and_logged(
    page: PlatePage, fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    turret = fake_microscope.require(ObjectiveTurret)

    def failing() -> list[str]:
        raise HardwareError("the turret did not answer")

    with _replaced(turret, "labels", failing):
        assert page.objective_labels() == ()
    assert (
        "smc.ui",
        "objective: its positions could not be read: HardwareError: the turret did "
        "not answer",
    ) in [(r.name, r.getMessage()) for r in caplog.records]


def test_a_drive_action_without_live_counts_as_an_ended_activity(
    page: PlatePage,
) -> None:
    before = page.activities_ended

    page.jog_xy(10.0, 0.0)
    assert page.join(10.0)

    assert page.activities_ended == before + 1
    assert page.status().outcome == "stage jog: XY (10.00, 0.00) µm"


def test_a_closed_page_refuses_a_drive_action(page: PlatePage) -> None:
    page.close()
    with _refuses(RuntimeError, "the page is closed"):
        page.jog_xy(10.0, 0.0)


def test_while_halted_moves_are_refused_and_switching_off_runs(
    light_page: PlatePage, shutter_light: Microscope
) -> None:
    _fake(shutter_light).setShutterOpen(SHUTTER_LIGHT, True)
    light_page.stop()
    halted = _refusal(lambda: shutter_light.require(XYStage).move_by_um(1.0, 0.0))
    assert halted.startswith("MicroscopeHaltedError: ")

    for act, name in (
        (lambda: light_page.jog_xy(10.0, 0.0), "stage jog"),
        (lambda: light_page.jog_z(1.0), "z jog"),
        (lambda: light_page.objective(TWENTY_X, confirm=True), "objective set"),
        (lambda: light_page.autofocus(True), "autofocus engage"),
        (lambda: light_page.light(True), "light on"),
    ):
        act()
        assert light_page.join(10.0)
        assert light_page.status().outcome == f"{name}: {halted}"
    light_page.autofocus(False)
    assert light_page.join(10.0)
    assert light_page.status().outcome == "autofocus disengage: autofocus off"
    light_page.light(False)
    assert light_page.join(10.0)
    assert light_page.status().outcome == "light off: light off"
    state = shutter_light.state()
    assert state.xy == XY(0.0, 0.0)
    assert state.objective == TEN_X


def test_jog_limits_follow_the_profile_and_the_assumed_z_default(
    run_doc: Path,
) -> None:
    for max_z, expected_z in ((20.0, 20.0), (None, DEFAULT_MAX_Z_JOG_UM)):
        profile = Profile.demo().model_copy(deep=True)
        profile.safety.max_jog_um = 250.0
        profile.safety.max_z_jog_um = max_z
        microscope = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()), profile
        )
        try:
            assert _page(microscope, run_doc).jog_limits_um() == (250.0, expected_z)
        finally:
            microscope.close()


# --- during live ---------------------------------------------------------------------


def test_a_drive_action_during_live_runs_between_two_frames_and_live_goes_on(
    page: PlatePage, fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_microscope.override(Camera, FrameCamera())
    seen = _on_thread(monkeypatch, "jog_xy", page)
    page.live(True)
    try:
        _wait_for_frame_after(page, 1)
        before = page.activities_ended

        page.jog_xy(100.0, -50.0)
        _wait_for_outcome(page, "stage jog: XY (100.00, -50.00) µm")
        at_outcome = page.frame()
        assert at_outcome is not None
        after = _wait_for_frame_after(page, at_outcome.number)

        # A frame taken after the outcome shows where the stage went (FM-96).
        assert after.xy_um == (100.0, -50.0)
        assert page.is_live
        assert page.status().running == "live"
        # Handed to the live worker: not an activity of its own.
        assert page.activities_ended == before
        assert seen == [("smc-ui-live", True)]
    finally:
        page.live(False)
    assert page.join(10.0)
    outcome = page.status().outcome
    assert outcome is not None
    assert re.fullmatch(r"live ended: \d+ frames in \d+\.\d s", outcome), outcome


def test_a_drive_action_during_live_is_logged(
    page: PlatePage, fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    fake_microscope.override(Camera, FrameCamera())
    page.live(True)
    try:
        _wait_for_frame_after(page, 1)
        with caplog.at_level("INFO", logger="smc.ui"):
            page.jog_z(1.0)
            _wait_for_outcome(page, "z jog: Z 1.00 µm")
    finally:
        page.live(False)
    assert page.join(10.0)
    assert ("smc.ui", "z jog: Z 1.00 µm") in [
        (r.name, r.getMessage()) for r in caplog.records
    ]


def test_a_second_drive_action_while_one_waits_is_refused(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(block=True)
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        assert camera.entered.wait(5)
        page.jog_xy(10.0, 0.0)
        with _refuses(RuntimeError, "stage jog is waiting"):
            page.jog_z(1.0)
        with _refuses(RuntimeError, "stage jog is waiting"):
            page.jog_xy(10.0, 0.0)
    finally:
        camera.release.set()
    try:
        _wait_for_outcome(page, "stage jog: XY (10.00, 0.00) µm")
    finally:
        page.live(False)
    assert page.join(10.0)
    state = fake_microscope.state()
    # One jog, not two, and no Z jog: a refused click is never queued.
    assert state.xy == XY(10.0, 0.0)
    assert state.z_um == 0.0


def test_stop_drops_a_waiting_drive_action_and_sends_nothing(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(block=True)
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        assert camera.entered.wait(5)
        _fake(fake_microscope).log.clear()
        page.jog_xy(10.0, 0.0)
        page.stop()
    finally:
        camera.release.set()
    assert page.join(10.0)
    status = page.status()
    assert status.outcome is not None
    assert re.fullmatch(
        r"live ended: \d+ frames in \d+\.\d s; stage jog was not run", status.outcome
    ), status.outcome
    assert status.halted
    assert fake_microscope.state().xy == XY(0.0, 0.0)
    assert not [
        line for line in _fake(fake_microscope).log if line.startswith("setXYPosition")
    ]
    assert page._drive_waiting is None


def test_end_live_drops_a_waiting_drive_action_and_says_so(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(block=True)
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        assert camera.entered.wait(5)
        page.jog_z(1.0)
        page.live(False)
    finally:
        camera.release.set()
    assert page.join(10.0)
    status = page.status()
    assert status.outcome is not None
    assert re.fullmatch(
        r"live ended: \d+ frames in \d+\.\d s; z jog was not run", status.outcome
    ), status.outcome
    assert not status.halted
    assert fake_microscope.state().z_um == 0.0


def test_close_drops_a_waiting_drive_action_and_says_so(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(block=True)
    fake_microscope.override(Camera, camera)
    page.live(True)
    assert camera.entered.wait(5)
    page.jog_xy(10.0, 0.0)
    closer = threading.Thread(target=page.close, daemon=True)
    closer.start()
    try:
        _wait_for(lambda: page.status().halted, "the halt of close()")
    finally:
        camera.release.set()
    closer.join(15.0)
    assert not closer.is_alive()
    assert page.close_ended
    assert re.fullmatch(
        r"live ended: \d+ frames in \d+\.\d s; stage jog was not run",
        page.close_lines[-1],
    ), page.close_lines
    assert fake_microscope.state().xy == XY(0.0, 0.0)


def test_a_failed_frame_drops_a_waiting_drive_action(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(
        block=True, fail_with=lambda: HardwareError("the camera timed out")
    )
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        assert camera.entered.wait(5)
        page.jog_xy(10.0, 0.0)
    finally:
        camera.release.set()
    assert page.join(10.0)
    assert page.status().outcome == (
        "live ended after 0 frames: HardwareError: the camera timed out; "
        "stage jog was not run"
    )
    assert fake_microscope.state().xy == XY(0.0, 0.0)
    # Live is over: a jog now is an activity of its own, and runs.
    before = page.activities_ended
    page.jog_xy(10.0, 0.0)
    assert page.join(10.0)
    assert page.status().outcome == "stage jog: XY (10.00, 0.00) µm"
    assert page.activities_ended == before + 1


def test_a_drive_action_once_live_is_ending_is_refused(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(block=True)
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        assert camera.entered.wait(5)
        page.live(False)
        # The event is set and the frame in flight still holds the activity.
        assert page.is_live
        with _refuses(RuntimeError, "live is ending"):
            page.jog_xy(10.0, 0.0)
    finally:
        camera.release.set()
    assert page.join(10.0)
    outcome = page.status().outcome
    assert outcome is not None
    assert "was not run" not in outcome
    assert fake_microscope.state().xy == XY(0.0, 0.0)


def test_a_failed_drive_action_does_not_end_live(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # 500 µm against the demo's 100 µm Z jog limit: the facade refuses it.
    refusal = _refusal(lambda: fake_microscope.require(ZStage).move_by_um(500.0))
    fake_microscope.override(Camera, FrameCamera())
    page.live(True)
    try:
        _wait_for_frame_after(page, 1)
        page.jog_z(500.0)
        _wait_for_outcome(page, f"z jog: {refusal}")
        at_outcome = page.frame()
        assert at_outcome is not None
        _wait_for_frame_after(page, at_outcome.number)
        assert page.is_live
    finally:
        page.live(False)
    assert page.join(10.0)
    assert fake_microscope.state().z_um == 0.0


def test_a_click_after_a_failed_frame_and_before_live_ends_is_refused(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A failed frame ends the loop without the event. Between the drop and the end of
    # the activity, live is still the activity going: a click kept then would be taken
    # by no loop, run by nobody and reported by nobody. The live's event is set first,
    # so it is refused. The click is made at that very point, from the worker.
    fake_microscope.override(
        Camera, FrameCamera(fail_with=lambda: HardwareError("the camera timed out"))
    )
    original = page._live
    said: list[str] = []

    def live_then_click(event: threading.Event) -> str:
        outcome = original(event)
        try:
            page.jog_xy(10.0, 0.0)
        except RuntimeError as exc:
            said.append(str(exc))
        return outcome

    with _replaced(page, "_live", live_then_click):
        page.live(True)
        assert page.join(10.0)
    assert said == ["live is ending"]
    assert page._drive_waiting is None
    assert fake_microscope.state().xy == XY(0.0, 0.0)


def test_stop_during_a_drive_action_ends_live_as_a_stop(
    page: PlatePage, fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fake core's own camera, not an override: an override bypasses the halt. A
    # frame after STOP would end live with the halt's refusal, not as the stop it is.
    original = drive_module.jog_xy

    def stopped_while_moving(*args: Any, **kwargs: Any) -> Any:
        page.stop()
        return original(*args, **kwargs)

    monkeypatch.setattr(drive_module, "jog_xy", stopped_while_moving)
    page.live(True)
    try:
        _wait_for_frame_after(page, 1)
        page.jog_xy(10.0, 0.0)
    except BaseException:
        page.live(False)
        raise
    assert page.join(10.0)
    outcome = page.status().outcome
    assert outcome is not None
    assert re.fullmatch(r"live ended: \d+ frames in \d+\.\d s", outcome), outcome
    assert page.status().halted
