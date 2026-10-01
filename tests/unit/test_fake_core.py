"""What ``FakeCore`` does, without Micro-Manager (design §8, #9).

The fake is only useful if it fails the way MMCore fails: the messages
asserted here are the demo's, measured 2026-10-01. ``test_fake_core_demo.py``
keeps its inventory equal to the demo's.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from pymmcore_plus import DeviceType

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, Shutter, XYStage, ZStage
from smc.hardware.errors import DeviceTimeoutError
from smc.hardware.profile import MicroManagerSection, Profile
from smc.testing import FakeCore, FakeDevice, FakeProperty

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


@contextmanager
def _facade(core: FakeCore, profile: Profile | None = None) -> Iterator[Microscope]:
    """A facade over ``core``, with the log cleared after building it."""
    microscope = Microscope.from_core(
        cast("CMMCorePlus", core), profile or Profile.demo()
    )
    core.log.clear()
    try:
        yield microscope
    finally:
        microscope.close()


def _raises_exactly(message: str) -> Any:
    """``pytest.raises`` for a ``RuntimeError`` whose whole message is ``message`` (FM-45)."""
    return pytest.raises(RuntimeError, match=f"^{re.escape(message)}$")


def test_fake_core_records_every_mutation() -> None:
    core = FakeCore.demo_like()
    with _facade(core) as microscope:
        microscope.require(XYStage).move_to_um(20, -10)
        microscope.require(Camera).set_exposure_ms(20)
        microscope.require(Shutter).set_open(True)
        assert core.log == [
            "setXYPosition('XY', 20.0, -10.0)",
            "setExposure(20.0)",
            "setShutterOpen('White Light Shutter', True)",
        ]


def test_building_a_facade_logs_the_timeout() -> None:
    core = FakeCore.demo_like()
    Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    assert core.log == ["setTimeoutMs(60000)"]


def test_fake_core_does_not_log_reads() -> None:
    core = FakeCore.demo_like()
    with _facade(core) as microscope:
        microscope.state()
        microscope.require(XYStage).position_um()
        microscope.require(XYStage).is_busy()
        microscope.require(ZStage).position_um()
        microscope.require(Camera).pixel_size_um()
        microscope.require(Shutter).is_open()
        core.getLoadedDevices()
        core.getDevicePropertyNames("Camera")
        core.getProperty("Camera", "Binning")
        core.getAllowedPropertyValues("Camera", "Binning")
        core.isPropertyReadOnly("Camera", "Binning")
        core.getStateLabels("Objective")
        core.getStateLabel("Objective")
        core.getState("Objective")
        core.getShutterOpen("White Light Shutter")
        core.isContinuousFocusEnabled()
        core.getTimeoutMs()
        assert core.log == []


def test_fake_busy_device_times_out() -> None:
    # FM-17: a stuck device; the move gives up at the core timeout and stops it.
    core = FakeCore.demo_like()
    core.busy_devices = {"Z"}
    profile = Profile.demo().model_copy(
        update={"micromanager": MicroManagerSection(config="", device_timeout_ms=200)}
    )
    with _facade(core, profile) as microscope:
        with pytest.raises(DeviceTimeoutError):
            microscope.require(ZStage).move_to_um(-1.0)
        assert core.log == ["setPosition('Z', -1.0)", "stop('Z')"]
        # A stuck device stays stuck: stop() does not clear it.
        assert core.deviceBusy("Z")


def test_fake_failing_device_raises_on_every_call_naming_it() -> None:
    core = FakeCore.demo_like()
    boom = RuntimeError("Z: no answer")
    core.failing["Z"] = boom
    calls = [
        lambda: core.getPosition("Z"),
        lambda: core.setPosition("Z", 1.0),
        lambda: core.deviceBusy("Z"),
        lambda: core.stop("Z"),
        lambda: core.getDeviceType("Z"),
        lambda: core.getDeviceName("Z"),
        lambda: core.getDevicePropertyNames("Z"),
        lambda: core.getProperty("Z", "Name"),
        lambda: core.isPropertyReadOnly("Z", "Name"),
        # The focus slot's device, reached without naming it.
        lambda: core.getPosition(),
    ]
    for call in calls:
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is boom
    # Another device still answers.
    assert core.getXYPosition("XY") == [0.0, 0.0]


def test_fake_failing_camera_raises_from_the_current_camera_calls() -> None:
    core = FakeCore.demo_like()
    boom = RuntimeError("camera: no answer")
    core.failing["Camera"] = boom
    calls = [
        core.snapImage,
        core.getImage,
        core.getExposure,
        lambda: core.setExposure(5.0),
        core.getImageHeight,
        core.getImageWidth,
        core.getImageBitDepth,
        core.getPixelSizeUm,
    ]
    for call in calls:
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is boom


def test_fake_unknown_label_raises_like_mmcore() -> None:
    core = FakeCore.demo_like()
    with _raises_exactly('No device with label "Nope"'):
        core.getDeviceType("Nope")
    with _raises_exactly('No device with label "Nope"'):
        core.getPosition("Nope")
    with _raises_exactly('Cannot get value of property "Nope"'):
        core.getProperty("Camera", "Nope")


def test_fake_wrong_device_type_raises_like_mmcore() -> None:
    core = FakeCore.demo_like()
    with _raises_exactly(
        'Device "XY" is of the wrong type for the requested operation'
    ):
        core.getPosition("XY")
    with _raises_exactly('Cannot stop "Camera": not a stage'):
        core.stop("Camera")


def test_fake_read_only_set_is_ignored_like_mmcore() -> None:
    core = FakeCore.demo_like()
    core.setProperty("Camera", "CameraName", "smc-contract")
    assert core.getProperty("Camera", "CameraName") == "DemoCamera-MultiMode"
    assert core.log == ["setProperty('Camera', 'CameraName', 'smc-contract')"]


def test_fake_disallowed_value_raises_like_mmcore() -> None:
    core = FakeCore.demo_like()
    with _raises_exactly('Cannot set property "Binning" to "3"'):
        core.setProperty("Camera", "Binning", "3")
    assert core.getProperty("Camera", "Binning") == "1"
    with _raises_exactly('Cannot set property "Exposure" to "20000"'):
        core.setProperty("Camera", "Exposure", "20000")


def test_fake_z_move_disables_continuous_focus() -> None:
    # FM-12: moving the Z drive switches PFS off on the Nikon stands.
    core = FakeCore.demo_like()
    with _facade(core) as microscope:
        core.enableContinuousFocus(True)
        microscope.require(ZStage).move_to_um(-1.0)
        assert core.isContinuousFocusEnabled() is False
        assert core.isContinuousFocusLocked() is False


def test_fake_z_move_keeps_continuous_focus_without_the_quirk() -> None:
    core = FakeCore.demo_like()
    core.quirk_z_move_disables_autofocus = False
    with _facade(core) as microscope:
        core.enableContinuousFocus(True)
        microscope.require(ZStage).move_to_um(-1.0)
        assert core.isContinuousFocusEnabled() is True
        assert core.isContinuousFocusLocked() is True


def test_fake_quirk_applies_only_to_the_focus_device() -> None:
    focus = FakeDevice("Z", DeviceType.Stage)
    other = FakeDevice("Piezo", DeviceType.Stage)
    core = FakeCore([focus, other], focus="Z")
    core.enableContinuousFocus(True)
    core.setPosition("Piezo", 1.0)
    assert core.isContinuousFocusEnabled() is True
    core.setPosition("Z", 1.0)
    assert core.isContinuousFocusEnabled() is False


def test_fake_frame_source_sees_the_stage_position() -> None:
    core = FakeCore.demo_like()
    seen: list[tuple[float, float, float]] = []
    frame = np.arange(6, dtype=np.uint16).reshape(2, 3)

    def source(x_um: float, y_um: float, z_um: float) -> np.ndarray:
        seen.append((x_um, y_um, z_um))
        return frame

    core.frame_source = source
    with _facade(core) as microscope:
        microscope.require(XYStage).move_to_um(20, -10)
        microscope.require(ZStage).move_to_um(-1)
        snapped = microscope.require(Camera).snap()
    assert seen == [(20.0, -10.0, -1.0)]
    assert snapped is frame


@pytest.mark.parametrize(
    ("bit_depth", "dtype"),
    [(8, np.uint8), (12, np.uint16), (16, np.uint16), (32, np.uint32)],
)
def test_fake_default_frame_matches_image_shape_and_bit_depth(
    bit_depth: int, dtype: type
) -> None:
    core = FakeCore.demo_like()
    core.image_shape = (4, 6)
    core.bit_depth = bit_depth
    core.snapImage()
    frame = core.getImage()
    assert frame.shape == (4, 6)
    assert frame.dtype == dtype
    assert not frame.any()


def test_fake_get_image_before_any_snap_raises_like_mmcore() -> None:
    core = FakeCore.demo_like()
    with _raises_exactly("Issue snapImage before getImage."):
        core.getImage()


def test_fake_state_and_label_properties_agree() -> None:
    core = FakeCore.demo_like()
    assert core.getState("Objective") == 1
    assert core.getStateLabel("Objective") == "Nikon 10X S Fluor"
    core.setState("Objective", 3)
    assert core.getStateLabel("Objective") == "Nikon 20X Plan Fluor ELWD"
    assert core.getProperty("Objective", "State") == "3"
    assert core.getProperty("Objective", "Label") == "Nikon 20X Plan Fluor ELWD"
    core.setProperty("Objective", "Label", "Nikon 40X Plan Fluor ELWD")
    assert core.getState("Objective") == 0
    assert core.getProperty("Objective", "State") == "0"
    core.setProperty("Objective", "State", "2")
    assert core.getStateLabel("Objective") == "Objective-2"


def test_fake_state_out_of_range_raises_and_keeps_the_position() -> None:
    core = FakeCore.demo_like()
    with _raises_exactly(
        'Error in device "Objective": (Error message unavailable) (103)'
    ):
        core.setState("Objective", 99)
    with _raises_exactly('Cannot set property "Label" to "Nope"'):
        core.setProperty("Objective", "Label", "Nope")
    assert core.getState("Objective") == 1


def test_fake_exposure_is_the_camera_exposure_property() -> None:
    core = FakeCore.demo_like()
    core.setExposure(20)
    assert core.getProperty("Camera", "Exposure") == "20.0000"
    core.setProperty("Camera", "Exposure", "12.5")
    assert core.getExposure() == 12.5
    assert core.getProperty("Camera", "Exposure") == "12.5000"


def test_fake_unload_device_clears_its_slot() -> None:
    core = FakeCore.demo_like()
    core.unloadDevice("White Light Shutter")
    assert core.getShutterDevice() == ""
    assert "White Light Shutter" not in core.getLoadedDevices()
    with _raises_exactly('No device with label "White Light Shutter"'):
        core.getShutterOpen("White Light Shutter")
    core.unloadAllDevices()
    core.unloadAllDevices()
    assert core.getLoadedDevices() == ("Core",)
    assert core.getCameraDevice() == core.getXYStageDevice() == ""
    assert core.getFocusDevice() == core.getAutoFocusDevice() == ""


def test_fake_slot_naming_no_device_is_refused() -> None:
    with pytest.raises(ValueError, match="camera='Cam' names no loaded device"):
        FakeCore([], camera="Cam")
    # A device of the wrong type does not fill a slot either.
    with pytest.raises(ValueError, match="focus='XY' names no loaded device"):
        FakeCore([FakeDevice("XY", DeviceType.XYStage)], focus="XY")


def test_fake_slot_setter_refuses_a_device_of_another_type_like_pymmcore() -> None:
    core = FakeCore.demo_like()
    with pytest.raises(
        ValueError, match=r'^Cannot set Core property Camera to invalid value "XY"$'
    ):
        core.setCameraDevice("XY")
    core.setCameraDevice("")
    assert core.getCameraDevice() == ""


def test_fake_device_without_state_properties_gets_them() -> None:
    wheel = FakeDevice("Wheel", DeviceType.State, state_labels=("A", "B"))
    camera = FakeDevice(
        "Cam", DeviceType.Camera, properties={"Gain": FakeProperty("1")}
    )
    core = FakeCore([wheel, camera], camera="Cam")
    assert core.getStateLabel("Wheel") == "A"
    core.setState("Wheel", 1)
    assert core.getProperty("Wheel", "Label") == "B"
    assert core.getExposure() == 10.0
