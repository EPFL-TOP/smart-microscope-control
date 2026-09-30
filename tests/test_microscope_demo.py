"""The ``Microscope`` facade on the demo devices (design §7, §13; #8).

Most tests open the demo stand exactly as a tool would. The others wrap the
demo core in ``_Spy``, a proxy that records stops and unloads and can make
one device misbehave: a stage that never reports idle, a Z drive that does
not answer, a snap that hangs. The demo devices themselves never do.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import smc.hardware.microscope as microscope_mod
from smc.hardware import Microscope, MicroscopeState
from smc.hardware.capabilities import XY, Camera, Properties, Shutter, XYStage, ZStage
from smc.hardware.errors import (
    CapabilityMissingError,
    DeviceTimeoutError,
    HardwareError,
    MicroscopeBusyError,
    MicroscopeHaltedError,
    MotionStoppedError,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import Role

pytestmark = [pytest.mark.demo, pytest.mark.usefixtures("need_mm")]

LOGGER = "smc.hardware.microscope"
JOIN_S = 5.0


@pytest.fixture
def need_mm(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


def _profile(*, timeout_ms: int = 60_000) -> Profile:
    data = Profile.demo().model_dump()
    data["micromanager"]["device_timeout_ms"] = timeout_ms
    return Profile.model_validate(data)


class _Spy:
    """The demo core, with stops and unloads recorded and some calls made to misbehave."""

    def __init__(self, core: Any) -> None:
        self._core = core
        self.calls: list[str] = []
        #: Labels that read busy; a stop clears one unless ``sticky``.
        self.busy: set[str] = set()
        self.sticky = False
        self.polled = threading.Event()
        #: When set, ``snapImage`` blocks until it is released.
        self.snap_release: threading.Event | None = None
        self.snap_entered = threading.Event()
        #: Called as each stop is sent, to see what was logged before it.
        self.on_stop: Callable[[], None] = lambda: None
        #: Called once, at the next busy poll: code running inside an action.
        self.on_poll: Callable[[], None] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def stop(self, label: str) -> None:
        self.on_stop()
        self.calls.append(f"stop {label}")
        if not self.sticky:
            self.busy.discard(label)
        self._core.stop(label)

    def unloadAllDevices(self) -> None:  # noqa: N802 - MMCore's name
        self.calls.append("unloadAllDevices")
        self._core.unloadAllDevices()

    def deviceBusy(self, label: str) -> bool:  # noqa: N802
        self.polled.set()
        hook, self.on_poll = self.on_poll, None
        if hook is not None:
            hook()
        return label in self.busy or bool(self._core.deviceBusy(label))

    def snapImage(self) -> None:  # noqa: N802
        if self.snap_release is not None:
            self.snap_entered.set()
            self.snap_release.wait(JOIN_S)
        self._core.snapImage()


class _StubCamera:
    """A ``Camera`` that needs no device: what the synthetic sample (#10) plugs in."""

    def snap(self) -> np.ndarray:
        return np.full((2, 3), 7, dtype=np.uint16)

    def exposure_ms(self) -> float:
        return 42.0

    def set_exposure_ms(self, value_ms: float) -> float:
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (2, 3)

    def bit_depth(self) -> int:
        return 12

    def pixel_size_um(self) -> float:
        return 0.5


class _Jammed:
    """A stage whose controller does not answer its stop."""

    def __init__(self, halted: Callable[[], bool], axis: str) -> None:
        self._halted = halted
        self._axis = axis
        self.halted_when_stopped: bool | None = None

    def wait(self, timeout_s: float | None = None) -> None:
        pass

    def is_busy(self) -> bool:
        return False

    def stop(self) -> None:
        self.halted_when_stopped = self._halted()
        raise RuntimeError(f"{self._axis} controller not answering")

    def limits_um(self) -> None:
        return None


class _JammedXY(_Jammed):
    def position_um(self) -> XY:
        return XY(0.0, 0.0)

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        return XY(x_um, y_um)

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        return XY(dx_um, dy_um)


class _JammedZ(_Jammed):
    def position_um(self) -> float:
        return 0.0

    def move_to_um(self, z_um: float) -> float:
        return z_um

    def move_by_um(self, dz_um: float) -> float:
        return dz_um


# --- open, roles, capabilities ----------------------------------------------


def test_open_demo_resolves_the_five_roles_and_closes() -> None:
    with Microscope.open("demo") as m:
        assert {
            role: m.roles.get(role)
            for role in (
                Role.camera,
                Role.xy_stage,
                Role.focus,
                Role.autofocus,
                Role.shutter,
            )
        } == {
            Role.camera: "Camera",
            Role.xy_stage: "XY",
            Role.focus: "Z",
            Role.autofocus: "Autofocus",
            Role.shutter: "White Light Shutter",
        }
        assert m.available() == [XYStage, ZStage, Camera, Shutter, Properties]
        assert not m.dry_run
        landed = m.require(XYStage).move_to_um(100, 50)
        assert landed.x_um == pytest.approx(100.0, abs=0.5)
        assert landed.y_um == pytest.approx(50.0, abs=0.5)
        core = m.core
        assert "XY" in core.getLoadedDevices()
    # One connection per stand: every device is released.
    assert list(core.getLoadedDevices()) == ["Core"]
    m.close()  # a second close is harmless


def test_require_returns_the_same_instance_twice() -> None:
    with Microscope.open("demo") as m:
        stage = m.require(XYStage)
        assert isinstance(stage, XYStage)
        assert m.require(XYStage) is stage
        assert m.get(XYStage) is stage
        assert m.require(Properties) is m.require(Properties)


def test_require_missing_role_names_the_role(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "blind.toml"
    path.write_text(
        '[microscope]\nname = "blind"\n\n[roles.exclude]\nshutter = ["shutter"]\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER), Microscope.open(path) as m:
        assert not m.has(Shutter)
        assert m.get(Shutter) is None
        assert Shutter not in m.available()
        with pytest.raises(CapabilityMissingError) as info:
            m.require(Shutter)
    assert info.value.role is Role.shutter
    assert "Shutter needs the 'shutter' role" in str(info.value)
    assert "available roles: camera, xy_stage, focus, autofocus" in str(info.value)
    assert "shutter" not in info.value.available
    # FM-13: the configuration's shutter was refused out loud, not silently.
    assert (
        "shutter: the configuration names 'White Light Shutter', which never "
        "fills shutter" in caplog.text
    )


def test_profile_assignment_picks_the_device_the_capability_drives() -> None:
    data = Profile.demo().model_dump()
    data["roles"]["assign"] = {"shutter": "LED Shutter"}
    with Microscope.open(Profile.model_validate(data)) as m:
        assert m.roles.get(Role.shutter) == "LED Shutter"
        assert m.roles.sources[Role.shutter] == "profile"
        m.core.setShutterOpen("White Light Shutter", False)
        m.core.setShutterOpen("LED Shutter", False)
        assert m.require(Shutter).set_open(True) is True
        assert m.core.getShutterOpen("LED Shutter")
        assert not m.core.getShutterOpen("White Light Shutter")


def test_override_replaces_a_capability() -> None:
    with Microscope.open("demo") as m:
        assert m.require(Camera).image_shape() == (512, 512)
        stub = _StubCamera()
        m.override(Camera, stub)
        assert m.require(Camera) is stub
        assert m.require(Camera).snap().shape == (2, 3)
        state = m.state()
        assert (state.exposure_ms, state.image_shape, state.pixel_size_um) == (
            42.0,
            (2, 3),
            0.5,
        )
        assert "| 42 ms |" in m.describe()
        with pytest.raises(TypeError, match="object does not implement Camera"):
            m.override(Camera, object())
        assert m.require(Camera) is stub


def test_pixel_size_falls_back_to_the_profile_for_the_current_objective() -> None:
    with Microscope.open("demo") as m:
        # Without MMCore's calibration, only the objective turret can tell.
        for name in m.core.getAvailablePixelSizeConfigs():
            m.core.deletePixelSizeConfig(name)
        camera, props = m.require(Camera), m.require(Properties)
        props.set("Objective", "State", 1)  # "Nikon 10X S Fluor"
        assert camera.pixel_size_um() == pytest.approx(0.65)
        props.set("Objective", "State", 0)  # "Nikon 40X Plan Fluor ELWD"
        assert camera.pixel_size_um() == pytest.approx(0.1625)
        props.set("Objective", "State", 2)  # "Objective-2": not in the profile
        assert camera.pixel_size_um() == 0.0


def test_open_releases_the_stand_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Any] = []
    real_open_core = microscope_mod.open_core

    def recording_open_core(config: Path | None) -> Any:
        core = real_open_core(config)
        opened.append(core)
        return core

    def broken(core: Any) -> Any:
        raise RuntimeError("inventory failed")

    monkeypatch.setattr(microscope_mod, "open_core", recording_open_core)
    monkeypatch.setattr(microscope_mod, "devices_from_core", broken)
    with pytest.raises(RuntimeError, match="inventory failed"):
        Microscope.open("demo")
    assert len(opened) == 1
    assert list(opened[0].getLoadedDevices()) == ["Core"]


# --- the core timeout and the lock timeout (FM-10, FM-15) ---------------------


def test_device_timeout_is_set_from_the_profile() -> None:
    with Microscope.open("demo") as m:
        # MMCore's own default is 5000 ms, shorter than a plate traverse.
        assert m.core.getTimeoutMs() == 60_000
    with Microscope.open(_profile(timeout_ms=45_000)) as m:
        assert m.core.getTimeoutMs() == 45_000


def test_the_lock_timeout_follows_the_device_timeout(demo_core: Any) -> None:
    spy = _Spy(demo_core)
    spy.snap_release = threading.Event()
    m = Microscope.from_core(spy, _profile(timeout_ms=500))
    assert demo_core.getTimeoutMs() == 500
    camera, shutter = m.require(Camera), m.require(Shutter)
    snapper = threading.Thread(target=camera.snap)
    snapper.start()
    try:
        assert spy.snap_entered.wait(JOIN_S)
        started = time.monotonic()
        with pytest.raises(MicroscopeBusyError) as info:
            shutter.set_auto_shutter(True)
        waited_s = time.monotonic() - started
    finally:
        spy.snap_release.set()
        snapper.join(timeout=JOIN_S)
        m.close()
    assert not snapper.is_alive()
    assert info.value.holder == "camera: snap"
    # 0.5 s, not the Executor's 60 s default.
    assert 0.5 <= waited_s < JOIN_S


# --- dry-run -----------------------------------------------------------------


def test_dry_run_reads_positions_but_never_moves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        Microscope.open("demo", dry_run=True) as m,
    ):
        assert m.dry_run
        xy, z = m.require(XYStage), m.require(ZStage)
        camera, shutter = m.require(Camera), m.require(Shutter)
        props = m.require(Properties)
        start_xy, start_z = xy.position_um(), z.position_um()
        exposure, auto = camera.exposure_ms(), shutter.auto_shutter()
        objective = props.get("Objective", "State")
        other = "2" if objective != "2" else "3"

        assert xy.move_to_um(500.0, 400.0) == XY(500.0, 400.0)
        assert xy.move_by_um(5.0, 5.0) == XY(start_xy.x_um + 5.0, start_xy.y_um + 5.0)
        assert z.move_to_um(12.5) == 12.5
        assert z.move_by_um(-1.0) == start_z - 1.0
        assert camera.set_exposure_ms(exposure + 25.0) == exposure + 25.0
        assert shutter.set_auto_shutter(not auto) is (not auto)
        assert props.set("Objective", "State", other) == other
        assert camera.snap().shape == camera.image_shape()  # acquisitions run

        assert xy.position_um() == start_xy
        assert z.position_um() == start_z
        assert camera.exposure_ms() == exposure
        assert shutter.auto_shutter() is auto
        assert props.get("Objective", "State") == objective
    dry = [msg for msg in caplog.messages if msg.startswith("[dry-run] ")]
    assert dry[0] == "[dry-run] xy_stage: move_to (500.0, 400.0) µm"
    assert len(dry) == 7
    # Nothing was logged as sent.
    sent = ("xy_stage:", "z:", "camera:", "shutter: auto", "property:")
    assert not [msg for msg in caplog.messages if msg.startswith(sent)]


# --- state and describe ------------------------------------------------------


def test_state_survives_one_failing_device(demo_core: Any) -> None:
    spy = _Spy(demo_core)

    def broken(label: str) -> float:
        raise RuntimeError("Z drive not responding")

    spy.getPosition = broken  # type: ignore[method-assign]
    m = Microscope.from_core(spy, Profile.demo())
    try:
        state = m.state()
        line = m.describe()
    finally:
        m.close()
    assert state.z_um is None
    assert state.errors == ["z: RuntimeError: Z drive not responding"]
    assert state.xy is not None
    assert state.exposure_ms is not None
    assert state.image_shape == (512, 512)
    assert state.shutter_open is not None
    assert state.roles["focus"] == "Z"
    assert "| Z ? |" in line


def test_describe_is_one_status_line() -> None:
    with Microscope.open("demo") as m:
        m.require(XYStage).move_to_um(100.0, 50.0)
        m.require(Camera).set_exposure_ms(25.0)
        m.require(Shutter).set_open(False)
        line = m.describe()
    assert line.startswith("XY (100.0, 50.0) µm | Z ")
    assert line.endswith(" µm | 25 ms | shutter closed")
    assert "\n" not in line


def test_state_reports_moving_and_halted(demo_core: Any) -> None:
    spy = _Spy(demo_core)
    spy.busy.add("XY")
    spy.sticky = True  # the stage never reports idle, even after its stop
    m = Microscope.from_core(spy, _profile(timeout_ms=100))
    try:
        state = m.state()
        assert isinstance(state, MicroscopeState)
        assert (state.moving, state.halted) == ((), False)
        assert "moving" not in m.describe()
        with pytest.raises(DeviceTimeoutError, match="the stop was sent"):
            m.require(XYStage).move_to_um(10.0, 0.0)
        assert m.state().moving == ("xy_stage XY",)
        assert m.describe().endswith("| moving: xy_stage XY")
        m.stop()
        state = m.state()
        assert state.halted
        assert state.moving == ("xy_stage XY",)  # a stop that did not take
        assert m.describe().endswith("| moving: xy_stage XY | HALTED")
        m.resume()
        assert not m.state().halted
        assert m.state().moving == ()
    finally:
        spy.busy.clear()
        m.close()


# --- stop, resume, close -----------------------------------------------------


def test_stop_halts_then_resume_allows_moves() -> None:
    with Microscope.open("demo") as m:
        xy, z = m.require(XYStage), m.require(ZStage)
        camera, shutter = m.require(Camera), m.require(Shutter)
        props = m.require(Properties)
        before = xy.move_to_um(0.0, 0.0)
        m.stop()
        refused: dict[str, Callable[[], object]] = {
            "move_to": lambda: xy.move_to_um(10.0, 10.0),
            "move_by": lambda: xy.move_by_um(1.0, 0.0),
            "z": lambda: z.move_by_um(1.0),
            "snap": camera.snap,
            "exposure": lambda: camera.set_exposure_ms(20.0),
            "open": lambda: shutter.set_open(True),
            "auto_shutter": lambda: shutter.set_auto_shutter(False),
            "property": lambda: props.set("Objective", "State", 2),
        }
        for name, action in refused.items():
            with pytest.raises(MicroscopeHaltedError):
                action()
            assert m.state().halted, name
        assert xy.position_um() == before  # reads still run ...
        assert shutter.set_open(False) is False  # ... and the light can be cut
        m.resume()
        landed = xy.move_to_um(10.0, 10.0)
    assert landed.x_um == pytest.approx(10.0, abs=0.5)


@pytest.mark.parametrize(
    ("jammed_axis", "other_stop", "named", "not_named"),
    [
        ("XY", "stop Z", "XYStage 'XY'", "ZStage"),
        ("Z", "stop XY", "ZStage 'Z'", "XYStage"),
    ],
)
def test_stop_continues_past_a_failing_stage_and_reports_it(
    demo_core: Any, jammed_axis: str, other_stop: str, named: str, not_named: str
) -> None:
    spy = _Spy(demo_core)
    m = Microscope.from_core(spy, Profile.demo())

    def halted() -> bool:
        return m.state().halted

    try:
        if jammed_axis == "XY":
            jammed: _Jammed = _JammedXY(halted, "XY")
            m.override(XYStage, jammed)
        else:
            jammed = _JammedZ(halted, "Z")
            m.override(ZStage, jammed)
        with pytest.raises(HardwareError) as info:
            m.stop()
        assert m.state().halted
    finally:
        m.close()
    assert jammed.halted_when_stopped is True  # halted before the stops
    assert spy.calls[0] == other_stop  # the other stage was still stopped
    message = str(info.value)
    assert (
        f"the emergency stop failed for {named} "
        f"(RuntimeError('{jammed_axis} controller not answering'))" in message
    )
    assert not_named not in message


def test_stop_during_a_move_ends_it(demo_core: Any) -> None:
    spy = _Spy(demo_core)
    spy.busy.add("XY")  # moving until stopped
    m = Microscope.from_core(spy, Profile.demo())
    errors: list[BaseException] = []

    def move() -> None:
        try:
            m.require(XYStage).move_to_um(10.0, 0.0)
        except BaseException as exc:
            errors.append(exc)

    mover = threading.Thread(target=move)
    mover.start()
    try:
        assert spy.polled.wait(JOIN_S)
        assert m.state().moving == ("xy_stage XY",)
        m.stop()
        mover.join(timeout=JOIN_S)
        assert not mover.is_alive()
    finally:
        spy.busy.clear()
        mover.join(timeout=JOIN_S)
        m.close()
    assert len(errors) == 1
    assert isinstance(errors[0], MotionStoppedError)
    assert spy.calls[:2] == ["stop XY", "stop Z"]


def test_close_stops_the_stages_when_something_moves(
    demo_core: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # §13: a move that timed out while the stage still reads busy is a motion
    # that outlived its action.
    spy = _Spy(demo_core)
    spy.busy.add("XY")
    spy.sticky = True
    m = Microscope.from_core(spy, _profile(timeout_ms=100))
    with pytest.raises(DeviceTimeoutError, match="the stop was sent"):
        m.require(XYStage).move_to_um(10.0, 0.0)
    assert m.state().moving == ("xy_stage XY",)
    spy.calls.clear()
    logged_before_stop: list[bool] = []
    spy.on_stop = lambda: logged_before_stop.append(
        any(msg.startswith("close:") for msg in caplog.messages)
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        m.close()
    assert spy.calls == ["stop XY", "stop Z", "unloadAllDevices"]
    assert "close: xy_stage XY still moving; the stages were stopped" in (
        caplog.messages
    )
    # FM-62: the stops went out before the line saying so.
    assert logged_before_stop == [False, False]


def test_close_during_a_move_stops_it_instead_of_waiting_for_it(
    demo_core: Any,
) -> None:
    # The mover holds the lock for its whole traverse (up to 10 s here); close
    # must not wait that long for it.
    spy = _Spy(demo_core)
    spy.busy.add("XY")  # moving until stopped
    m = Microscope.from_core(spy, _profile(timeout_ms=10_000))
    errors: list[BaseException] = []

    def move() -> None:
        try:
            m.require(XYStage).move_to_um(10.0, 0.0)
        except BaseException as exc:
            errors.append(exc)

    mover = threading.Thread(target=move)
    mover.start()
    try:
        assert spy.polled.wait(JOIN_S)
        started = time.monotonic()
        m.close()
        closed_in_s = time.monotonic() - started
        mover.join(timeout=JOIN_S)
        assert not mover.is_alive()
    finally:
        spy.busy.clear()
        mover.join(timeout=JOIN_S)
    assert closed_in_s < JOIN_S
    assert len(errors) == 1
    assert isinstance(errors[0], MotionStoppedError)
    assert spy.calls == ["stop XY", "stop Z", "unloadAllDevices"]


def test_close_never_raises_because_a_stop_failed(
    demo_core: Any, caplog: pytest.LogCaptureFixture
) -> None:
    spy = _Spy(demo_core)
    spy.busy.add("XY")
    spy.sticky = True
    m = Microscope.from_core(spy, _profile(timeout_ms=100))
    with pytest.raises(DeviceTimeoutError):
        m.require(XYStage).move_to_um(10.0, 0.0)

    def broken(label: str) -> None:
        raise RuntimeError(f"{label} does not answer")

    spy.stop = broken  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        m.close()
    assert list(demo_core.getLoadedDevices()) == ["Core"]
    assert (
        "close: the stop failed for XYStage 'XY' (RuntimeError('XY does not answer'))"
        in caplog.messages
    )


def test_close_from_inside_an_action_stops_but_does_not_unload(
    demo_core: Any,
) -> None:
    # Adversarial review of #8: a Ctrl-C handler that calls close() runs on
    # the thread whose move is waiting. Unloading there pulled the devices
    # from under that move, which raised a bare RuntimeError from MMCore.
    spy = _Spy(demo_core)
    spy.busy.add("XY")  # moving until stopped
    m = Microscope.from_core(spy, _profile(timeout_ms=10_000))
    spy.on_poll = m.close  # runs inside the move's wait, holding the lock
    with pytest.raises(HardwareError) as info:
        m.require(XYStage).move_to_um(10.0, 0.0)
    assert type(info.value) is HardwareError
    assert "close() was called from inside an action on this thread" in str(info.value)
    assert spy.calls[:2] == ["stop XY", "stop Z"]  # the move was stopped ...
    assert "unloadAllDevices" not in spy.calls  # ... and nothing was unloaded
    assert m.state().moving == ()
    m.close()  # once the action has returned, close releases the stand
    assert spy.calls[-1] == "unloadAllDevices"


def test_close_behind_a_hung_snap_fails_with_a_diagnosis(demo_core: Any) -> None:
    spy = _Spy(demo_core)
    spy.snap_release = threading.Event()
    m = Microscope.from_core(spy, _profile(timeout_ms=500))
    snapper = threading.Thread(target=m.require(Camera).snap)
    snapper.start()
    try:
        assert spy.snap_entered.wait(JOIN_S)
        started = time.monotonic()
        with pytest.raises(MicroscopeBusyError) as info:
            m.close()
        waited_s = time.monotonic() - started
        assert "unloadAllDevices" not in spy.calls
    finally:
        spy.snap_release.set()
        snapper.join(timeout=JOIN_S)
    assert not snapper.is_alive()
    # Bounded by the lock timeout (0.5 s), well before the snap would end.
    assert info.value.at_least
    assert 0.5 <= info.value.held_s <= waited_s < 2.5
    m.close()  # the snap is over: the retry unloads
    assert spy.calls[-1] == "unloadAllDevices"
    m.close()  # and a third close is a no-op
    assert spy.calls.count("unloadAllDevices") == 1
