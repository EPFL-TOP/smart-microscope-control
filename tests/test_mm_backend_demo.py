"""The Micro-Manager backend on the demo devices, labels taken from the core's own slots.

Role resolution (#6) and the facade (#8) do not exist yet, so each test
builds the capability directly from ``core.getXYStageDevice()`` and friends.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np
import pytest

from smc.hardware.backends.mm import (
    MMCamera,
    MMProperties,
    MMShutter,
    MMXYStage,
    MMZStage,
)
from smc.hardware.capabilities import (
    XY,
    Camera,
    Limits,
    Properties,
    Shutter,
    XYStage,
    ZStage,
)
from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.safety import Executor, Safety

LOGGER = logging.getLogger("smc.hardware.test")


def _executor(*, dry_run: bool = False) -> Executor:
    return Executor(dry_run=dry_run, lock=threading.RLock(), logger=LOGGER)


def _xy(core: Any, *, dry_run: bool = False, **safety: Any) -> MMXYStage:
    safety.setdefault("max_jog_um", 1000.0)
    return MMXYStage(
        core, core.getXYStageDevice(), _executor(dry_run=dry_run), Safety(**safety)
    )


def _z(core: Any, *, dry_run: bool = False, **safety: Any) -> MMZStage:
    safety.setdefault("max_jog_um", 1000.0)
    return MMZStage(
        core, core.getFocusDevice(), _executor(dry_run=dry_run), Safety(**safety)
    )


def _camera(
    core: Any,
    pixel_sizes_um: dict[str, float] | None = None,
    objective: str | None = None,
) -> MMCamera:
    return MMCamera(
        core,
        core.getCameraDevice(),
        _executor(),
        pixel_sizes_um or {},
        lambda: objective,
    )


@pytest.mark.demo
class TestOnDemoDevices:
    def test_xy_move_to_lands_within_half_a_micron(self, demo_core: Any) -> None:
        stage = _xy(demo_core)
        landed = stage.move_to_um(123.0, -45.0)
        assert landed.x_um == pytest.approx(123.0, abs=0.5)
        assert landed.y_um == pytest.approx(-45.0, abs=0.5)
        assert stage.position_um() == landed
        assert not stage.is_busy()

    def test_xy_move_by_is_relative(self, demo_core: Any) -> None:
        stage = _xy(demo_core)
        start = stage.move_to_um(100.0, 200.0)
        landed = stage.move_by_um(10.0, -20.0)
        assert landed.x_um == pytest.approx(start.x_um + 10.0, abs=0.5)
        assert landed.y_um == pytest.approx(start.y_um - 20.0, abs=0.5)

    def test_xy_jog_above_limit_is_refused_and_does_not_move(
        self, demo_core: Any
    ) -> None:
        stage = _xy(demo_core, max_jog_um=50.0)
        start = stage.move_to_um(0.0, 0.0)
        with pytest.raises(SafetyRefusedError):
            stage.move_by_um(60.0, 0.0)
        assert stage.position_um() == start
        landed = stage.move_by_um(60.0, 0.0, force=True)
        assert landed.x_um == pytest.approx(start.x_um + 60.0, abs=0.5)

    def test_xy_absolute_move_outside_soft_limits_is_refused(
        self, demo_core: Any
    ) -> None:
        stage = _xy(demo_core, xy_soft_limits_um=((-100.0, 100.0), (-100.0, 100.0)))
        start = stage.move_to_um(0.0, 0.0)
        with pytest.raises(SafetyRefusedError) as info:
            stage.move_to_um(0.0, 100.5)
        assert info.value.how_to_force == ""
        assert stage.position_um() == start

    def test_xy_relative_move_into_the_soft_limit_is_refused(
        self, demo_core: Any
    ) -> None:
        stage = _xy(demo_core, xy_soft_limits_um=((-100.0, 100.0), (-100.0, 100.0)))
        start = stage.move_to_um(90.0, 0.0)
        with pytest.raises(SafetyRefusedError):
            stage.move_by_um(20.0, 0.0, force=True)
        assert stage.position_um() == start
        assert stage.limits_um() == (Limits(-100.0, 100.0), Limits(-100.0, 100.0))

    def test_xy_dry_run_returns_the_command_and_does_not_move(
        self, demo_core: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        _xy(demo_core).move_to_um(0.0, 0.0)
        before = _xy(demo_core).position_um()
        dry = _xy(demo_core, dry_run=True)
        with caplog.at_level(logging.INFO, logger=LOGGER.name):
            assert dry.move_to_um(500.0, 500.0) == XY(500.0, 500.0)
        assert "[dry-run] xy_stage: move_to (500.0, 500.0) µm" in caplog.messages
        assert dry.move_by_um(5.0, 5.0) == XY(before.x_um + 5.0, before.y_um + 5.0)
        assert dry.position_um() == before

    def test_z_round_trip(
        self, demo_core: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        z = _z(demo_core)
        with caplog.at_level(logging.INFO, logger=LOGGER.name):
            landed = z.move_to_um(12.5)
        assert "z: move_to 12.5 µm" in caplog.messages
        assert landed == pytest.approx(12.5, abs=0.5)
        assert z.position_um() == pytest.approx(landed)
        assert z.move_by_um(-2.5) == pytest.approx(10.0, abs=0.5)
        assert z.limits_um() is None

    def test_z_target_outside_soft_limits_is_refused(self, demo_core: Any) -> None:
        z = _z(demo_core, z_soft_limits_um=(0.0, 100.0))
        start = z.move_to_um(50.0)
        with pytest.raises(SafetyRefusedError):
            z.move_to_um(100.5)
        with pytest.raises(SafetyRefusedError):
            z.move_by_um(60.0)
        assert z.position_um() == pytest.approx(start)
        assert z.limits_um() == Limits(0.0, 100.0)

    def test_snap_shape_matches_image_shape(self, demo_core: Any) -> None:
        camera = _camera(demo_core)
        frame = camera.snap()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 2
        assert frame.shape == camera.image_shape()
        assert camera.bit_depth() > 0

    def test_exposure_round_trip(
        self, demo_core: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        camera = _camera(demo_core)
        with caplog.at_level(logging.INFO, logger=LOGGER.name):
            assert camera.set_exposure_ms(37.0) == pytest.approx(37.0)
        assert "camera: exposure 37.0 ms" in caplog.messages
        assert camera.exposure_ms() == pytest.approx(37.0)

    def test_pixel_size_prefers_the_core_calibration(self, demo_core: Any) -> None:
        # The demo configuration ships pixel-size presets, so MMCore knows it;
        # an unusual value proves it is MMCore's number that comes back.
        demo_core.setPixelSizeUm(demo_core.getCurrentPixelSizeConfig(), 0.37)
        assert demo_core.getPixelSizeUm() == pytest.approx(0.37)
        camera = _camera(demo_core, {"Nikon 10X S Fluor": 99.0}, "Nikon 10X S Fluor")
        assert camera.pixel_size_um() == pytest.approx(0.37)

    def test_pixel_size_falls_back_to_the_objective_map(self, demo_core: Any) -> None:
        for name in demo_core.getAvailablePixelSizeConfigs():
            demo_core.deletePixelSizeConfig(name)
        assert demo_core.getPixelSizeUm() == 0
        sizes = {"Nikon 10X S Fluor": 0.65}
        assert _camera(demo_core, sizes, "Nikon 10X S Fluor").pixel_size_um() == 0.65
        assert _camera(demo_core, sizes, "Nikon 40X Plan Fluor").pixel_size_um() == 0.0
        assert _camera(demo_core, sizes, None).pixel_size_um() == 0.0

    def test_pixel_size_fallback_scales_with_binning(self, demo_core: Any) -> None:
        # MMCore's own value includes binning; the profile map is for binning 1.
        for name in demo_core.getAvailablePixelSizeConfigs():
            demo_core.deletePixelSizeConfig(name)
        demo_core.setProperty(demo_core.getCameraDevice(), "Binning", "2")
        camera = _camera(demo_core, {"Nikon 10X S Fluor": 0.65}, "Nikon 10X S Fluor")
        assert camera.pixel_size_um() == pytest.approx(1.3)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -0.65, 0.0])
    def test_pixel_size_bad_profile_entry_is_unknown(
        self, demo_core: Any, bad: float, caplog: pytest.LogCaptureFixture
    ) -> None:
        for name in demo_core.getAvailablePixelSizeConfigs():
            demo_core.deletePixelSizeConfig(name)
        camera = _camera(demo_core, {"Nikon 10X S Fluor": bad}, "Nikon 10X S Fluor")
        with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm"):
            assert camera.pixel_size_um() == 0.0
        assert "not a positive number" in caplog.text

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
    def test_exposure_refuses_non_finite_or_negative(
        self, demo_core: Any, bad: float
    ) -> None:
        camera = _camera(demo_core)
        before = camera.exposure_ms()
        with pytest.raises(ValueError, match="exposure must be finite"):
            camera.set_exposure_ms(bad)
        assert camera.exposure_ms() == before

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("snap", ()),
            ("exposure_ms", ()),
            ("set_exposure_ms", (10.0,)),
            ("image_shape", ()),
            ("bit_depth", ()),
            ("pixel_size_um", ()),
        ],
    )
    def test_camera_refuses_when_it_is_not_the_current_camera(
        self, demo_core: Any, method: str, args: tuple[float, ...]
    ) -> None:
        camera = MMCamera(demo_core, "NotTheCamera", _executor(), {}, lambda: None)
        with pytest.raises(HardwareError, match="not the core's current camera"):
            getattr(camera, method)(*args)

    def test_shutter_open_close_and_auto_shutter(
        self, demo_core: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        shutter = MMShutter(demo_core, demo_core.getShutterDevice(), _executor())
        with caplog.at_level(logging.INFO, logger=LOGGER.name):
            assert shutter.set_open(True) is True
            assert shutter.is_open() is True
            assert shutter.set_open(False) is False
        assert shutter.is_open() is False
        assert "shutter: open" in caplog.messages
        assert "shutter: close" in caplog.messages
        assert shutter.set_auto_shutter(False) is False
        assert shutter.auto_shutter() is False
        assert shutter.set_auto_shutter(True) is True
        assert shutter.auto_shutter() is True

    def test_properties_lists_devices_and_refuses_read_only_set(
        self, demo_core: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        props = MMProperties(demo_core, _executor())
        devices = props.devices()
        assert "Core" not in devices
        assert demo_core.getCameraDevice() in devices
        camera = demo_core.getCameraDevice()
        infos = props.describe(camera)
        assert infos
        assert all(i.device == camera for i in infos)

        read_only = next(i for i in infos if i.read_only)
        with pytest.raises(HardwareError, match="read-only"):
            props.set(camera, read_only.name, "anything")
        assert props.get(camera, read_only.name) == read_only.value

        with caplog.at_level(logging.INFO, logger=LOGGER.name):
            assert float(props.set(camera, "Exposure", 25)) == pytest.approx(25.0)
        assert float(props.get(camera, "Exposure")) == pytest.approx(25.0)
        assert f"property: {camera}.Exposure=25" in caplog.messages


class _AlwaysBusyCore:
    """The only MMCore calls ``wait`` makes, with a device that never settles."""

    def __init__(self, timeout_ms: float = 30.0) -> None:
        self.timeout_ms = timeout_ms
        self.polls = 0
        self.stops: list[str] = []

    def deviceBusy(self, label: str) -> bool:  # noqa: N802 - MMCore's name
        self.polls += 1
        if self.polls > 1000:
            # About 10 s of polling: a wait that never expires fails the test
            # instead of hanging the suite (FM-44).
            raise AssertionError("wait() kept polling past any sane deadline")
        return True

    def stop(self, label: str) -> None:
        self.stops.append(label)

    def getTimeoutMs(self) -> float:  # noqa: N802 - MMCore's name
        return self.timeout_ms


class _SteppingCore:
    """A stage that lands on whole micrometres, as a real stepper rounds to its step."""

    def __init__(self) -> None:
        self.xy = (0.0, 0.0)
        self.z = 0.0

    def setXYPosition(self, label: str, x: float, y: float) -> None:  # noqa: N802
        self.xy = (float(round(x)), float(round(y)))

    def getXPosition(self, label: str) -> float:  # noqa: N802
        return self.xy[0]

    def getYPosition(self, label: str) -> float:  # noqa: N802
        return self.xy[1]

    def setPosition(self, label: str, z: float) -> None:  # noqa: N802
        self.z = float(round(z))

    def getPosition(self, label: str) -> float:  # noqa: N802
        return self.z

    def deviceBusy(self, label: str) -> bool:  # noqa: N802
        return False

    def getTimeoutMs(self) -> float:  # noqa: N802
        return 1000.0


def test_moves_return_the_readback_not_the_command() -> None:
    core: Any = _SteppingCore()
    safety = Safety(max_jog_um=100.0)
    assert MMXYStage(core, "XY", _executor(), safety).move_to_um(1.4, -2.6) == XY(
        1.0, -3.0
    )
    assert MMZStage(core, "Z", _executor(), safety).move_to_um(7.6) == 8.0


class _BinningCore:
    """A current camera with no calibration and a given ``Binning`` property."""

    def __init__(self, binning: str | None) -> None:
        self.binning = binning

    def getCameraDevice(self) -> str:  # noqa: N802
        return "Cam"

    def getPixelSizeUm(self) -> float:  # noqa: N802
        return 0.0

    def hasProperty(self, label: str, name: str) -> bool:  # noqa: N802
        return self.binning is not None

    def getProperty(self, label: str, name: str) -> str:  # noqa: N802
        assert self.binning is not None
        return self.binning


@pytest.mark.parametrize(
    ("binning", "expected"),
    [
        ("1", 0.5),
        ("4", 2.0),
        ("2x2", 1.0),
        ("2X2", 1.0),
        ("1x2", 0.0),
        ("0", 0.0),
        ("abc", 0.0),
        ("2x2x2", 0.0),
        (None, 0.0),
    ],
)
def test_pixel_size_fallback_reads_binning_or_reports_unknown(
    binning: str | None, expected: float, caplog: pytest.LogCaptureFixture
) -> None:
    core: Any = _BinningCore(binning)
    camera = MMCamera(core, "Cam", _executor(), {"10x": 0.5}, lambda: "10x")
    with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm"):
        assert camera.pixel_size_um() == pytest.approx(expected)
    # Unknown is said out loud, not just returned as 0.0.
    assert ("cannot be read" in caplog.text) == (expected == 0.0)


def test_wait_times_out_cleanly() -> None:
    core = _AlwaysBusyCore()
    stage = MMXYStage(core, "XY", _executor(), Safety(max_jog_um=1.0))  # type: ignore[arg-type]
    started = time.monotonic()
    with pytest.raises(DeviceTimeoutError, match=r"XY is still busy after 0\.05 s"):
        stage.wait(0.05)
    assert 0.05 <= time.monotonic() - started < 2.0
    assert core.polls > 1
    # §13: a wait that gives up stops the stage before it raises (FM-17).
    assert core.stops == ["XY"]
    assert stage.is_busy()


def test_wait_defaults_to_the_core_timeout() -> None:
    core = _AlwaysBusyCore(timeout_ms=30.0)
    z = MMZStage(core, "Z", _executor(), Safety(max_jog_um=1.0))  # type: ignore[arg-type]
    with pytest.raises(DeviceTimeoutError, match=r"after 0\.03 s"):
        z.wait()


@pytest.mark.parametrize("bad", [math.nan, -1.0])
def test_wait_rejects_a_timeout_that_would_never_expire(bad: float) -> None:
    z = MMZStage(_AlwaysBusyCore(), "Z", _executor(), Safety(max_jog_um=1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        z.wait(bad)


def test_mm_classes_satisfy_the_protocols() -> None:
    core: Any = object()
    safety = Safety(max_jog_um=1.0)
    assert isinstance(MMXYStage(core, "XY", _executor(), safety), XYStage)
    assert isinstance(MMZStage(core, "Z", _executor(), safety), ZStage)
    assert isinstance(MMCamera(core, "Camera", _executor(), {}, lambda: None), Camera)
    assert isinstance(MMShutter(core, "Shutter", _executor()), Shutter)
    assert isinstance(MMProperties(core, _executor()), Properties)
    # The check can fail: a stage is not a camera.
    assert not isinstance(MMXYStage(core, "XY", _executor(), safety), Camera)


# --- Motion, stop and the lock (design §13, #54) -----------------------------

JOIN_S = 5.0


@pytest.mark.demo
class TestMotionOnDemoDevices:
    def test_move_with_wait_leaves_nothing_moving(self, demo_core: Any) -> None:
        executor = _executor()
        safety = Safety(max_jog_um=1000.0)
        xy = MMXYStage(demo_core, demo_core.getXYStageDevice(), executor, safety)
        z = MMZStage(demo_core, demo_core.getFocusDevice(), executor, safety)
        xy.move_to_um(10.0, 20.0)
        z.move_by_um(1.5)
        assert executor.moving() == ()

    def test_stop_reaches_the_demo_stages(
        self, demo_core: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        executor = _executor()
        safety = Safety(max_jog_um=1000.0)
        xy = MMXYStage(demo_core, demo_core.getXYStageDevice(), executor, safety)
        z = MMZStage(demo_core, demo_core.getFocusDevice(), executor, safety)
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            xy.stop()
            z.stop()
        assert f"xy_stage {demo_core.getXYStageDevice()}: stop" in caplog.messages
        assert f"z {demo_core.getFocusDevice()}: stop" in caplog.messages
        # A move stopped before its wait reports that it did not complete.
        xy.move_to_um(50.0, 50.0, wait=False)
        xy.stop()
        with pytest.raises(MotionStoppedError):
            xy.wait()
        assert executor.moving() == ()

    def test_properties_set_on_a_state_device_waits_and_reads_back(
        self, demo_core: Any
    ) -> None:
        executor = _executor()
        props = MMProperties(demo_core, executor)
        assert props.set("Objective", "State", 2) == "2"
        assert props.get("Objective", "State") == "2"
        assert executor.moving() == ()


class _MovingCore:
    """A stage that stays busy until the test releases it, and a camera.

    The demo stages report idle at once, so no guard test can rely on them
    being busy (#54 plan).
    """

    def __init__(self) -> None:
        self.xy = (0.0, 0.0)
        self.moving = False
        self.release = threading.Event()
        self.polled = threading.Event()
        self.stops: list[str] = []
        self.snaps = 0

    def setXYPosition(self, label: str, x: float, y: float) -> None:  # noqa: N802
        self.xy = (x, y)
        self.moving = True
        self.release.clear()

    def getXPosition(self, label: str) -> float:  # noqa: N802
        return self.xy[0]

    def getYPosition(self, label: str) -> float:  # noqa: N802
        return self.xy[1]

    def deviceBusy(self, label: str) -> bool:  # noqa: N802
        self.polled.set()
        if self.moving and self.release.is_set():
            self.moving = False
        return self.moving

    def stop(self, label: str) -> None:
        self.stops.append(label)
        self.moving = False

    def getTimeoutMs(self) -> float:  # noqa: N802
        return 30_000.0

    def getCameraDevice(self) -> str:  # noqa: N802
        return "Cam"

    def snapImage(self) -> None:  # noqa: N802
        self.snaps += 1

    def getImage(self) -> np.ndarray:  # noqa: N802
        return np.zeros((4, 4), dtype=np.uint16)


def test_move_by_waits_outside_the_lock() -> None:
    # A short lock timeout makes a wait that holds the lock fail the read
    # below with MicroscopeBusyError instead of hanging (FM-44).
    core = _MovingCore()
    executor = Executor(
        dry_run=False, lock=threading.RLock(), logger=LOGGER, lock_timeout_s=2.0
    )
    stage = MMXYStage(core, "XY", executor, Safety(max_jog_um=100.0))  # type: ignore[arg-type]
    result: list[XY] = []
    mover = threading.Thread(target=lambda: result.append(stage.move_by_um(5.0, 0.0)))
    mover.start()
    try:
        assert core.polled.wait(JOIN_S)
        assert stage.position_um() == XY(5.0, 0.0)  # another thread's read gets through
        assert mover.is_alive()  # ... while the move is still waiting
    finally:
        core.release.set()
        mover.join(timeout=JOIN_S)
    assert not mover.is_alive()
    assert result == [XY(5.0, 0.0)]
    assert executor.moving() == ()


def test_snap_after_move_without_wait_is_refused_until_wait() -> None:
    core = _MovingCore()
    executor = _executor()
    stage = MMXYStage(core, "XY", executor, Safety(max_jog_um=100.0))  # type: ignore[arg-type]
    camera = MMCamera(core, "Cam", executor, {}, lambda: None)  # type: ignore[arg-type]
    stage.move_to_um(10.0, 0.0, wait=False)
    with pytest.raises(MotionInProgressError, match="xy_stage XY"):
        camera.snap()
    assert core.snaps == 0
    core.release.set()
    stage.wait()
    assert camera.snap().shape == (4, 4)
    assert core.snaps == 1


def test_stop_during_a_move_ends_the_waiting_move() -> None:
    core = _MovingCore()
    executor = _executor()
    stage = MMXYStage(core, "XY", executor, Safety(max_jog_um=100.0))  # type: ignore[arg-type]
    errors: list[BaseException] = []

    def move() -> None:
        try:
            stage.move_to_um(10.0, 0.0)
        except BaseException as exc:
            errors.append(exc)

    mover = threading.Thread(target=move)
    mover.start()
    assert core.polled.wait(JOIN_S)
    stage.stop()
    mover.join(timeout=JOIN_S)
    assert not mover.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], MotionStoppedError)
    assert core.stops == ["XY"]


class _TurretCore:
    """A ``State`` device that reports busy for a few polls after each set."""

    def __init__(self, busy_polls: int = 3) -> None:
        self.value = "0"
        self.busy_polls = busy_polls
        self.busy_left = 0
        self.polls = 0

    def isPropertyReadOnly(self, device: str, name: str) -> bool:  # noqa: N802
        return False

    def getDeviceType(self, device: str) -> int:  # noqa: N802
        from pymmcore_plus import DeviceType

        return int(DeviceType.State)

    def setProperty(self, device: str, name: str, value: object) -> None:  # noqa: N802
        self.value = str(value)
        self.busy_left = self.busy_polls

    def getProperty(self, device: str, name: str) -> str:  # noqa: N802
        return self.value

    def deviceBusy(self, device: str) -> bool:  # noqa: N802
        self.polls += 1
        if self.busy_left > 0:
            self.busy_left -= 1
            return True
        return False

    def getTimeoutMs(self) -> float:  # noqa: N802
        return 30_000.0


def test_properties_set_on_a_state_device_waits_until_idle() -> None:
    core = _TurretCore(busy_polls=3)
    executor = _executor()
    props = MMProperties(core, executor)  # type: ignore[arg-type]
    assert props.set("Turret", "State", 2) == "2"
    assert core.busy_left == 0  # it waited for the device, not just the command
    assert core.polls >= 4
    assert executor.moving() == ()


def test_properties_set_times_out_without_a_stop_on_a_state_device() -> None:
    core = _TurretCore(busy_polls=10**9)
    core.getTimeoutMs = lambda: 50.0  # type: ignore[method-assign]
    props = MMProperties(core, _executor())  # type: ignore[arg-type]
    with pytest.raises(DeviceTimeoutError, match="cannot be stopped from smc"):
        props.set("Turret", "State", 2)
