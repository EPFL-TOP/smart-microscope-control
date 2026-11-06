"""What ``FakeCore`` does, without Micro-Manager (design §8, #9).

The fake is only useful if it fails the way MMCore fails: the messages
asserted here are the demo's, measured 2026-10-01. ``test_fake_core_demo.py``
keeps its inventory equal to the demo's.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from pymmcore_plus import DeviceType
from test_mm_channels import ti2_core

from smc.hardware import Microscope
from smc.hardware.backends.mm import as_hardware_errors
from smc.hardware.capabilities import Camera, Shutter, XYStage, ZStage
from smc.hardware.errors import DeviceTimeoutError, HardwareError
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


def test_a_refused_stop_raises_the_ti2_text_and_leaves_the_device_busy() -> None:
    # m2 §11.8: the Ti2-E's answer to every stop of its stages (#208).
    core = FakeCore.demo_like()
    core.stop_refused = {"XY"}
    core.busy_devices = {"XY"}
    with _raises_exactly('Error in device "XY": Unsupported device command (11)'):
        core.stop("XY")
    # Logged before it raised, and the move goes on.
    assert core.log == ["stop('XY')"]
    assert core.deviceBusy("XY")
    # A fresh error each time: one kept would hold the fake's frames (FM-98).
    errors = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            core.stop("XY")
        errors.append(caught.value)
    assert errors[0] is not errors[1]
    # Another stage still takes its stop.
    core.stop("Z")
    # Through the layer's translator, the text the Executor sees.
    with pytest.raises(HardwareError) as translated:
        as_hardware_errors(cast("CMMCorePlus", core), "xy_stage").stop("XY")
    assert str(translated.value).startswith(
        "xy_stage: stop('XY') failed (Error in device \"XY\": Unsupported device "
        "command (11))"
    )


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
    # FM-12: moving the Z drive switches PFS off on the Nikon stands. On the
    # core directly: the facade's Z move engages it again (m2 §3.4).
    core = FakeCore.demo_like()
    core.enableContinuousFocus(True)
    core.setPosition("Z", -1.0)
    assert core.isContinuousFocusEnabled() is False
    assert core.isContinuousFocusLocked() is False


def test_fake_z_move_keeps_continuous_focus_without_the_quirk() -> None:
    core = FakeCore.demo_like()
    core.quirk_z_move_disables_autofocus = False
    core.enableContinuousFocus(True)
    core.setPosition("Z", -1.0)
    assert core.isContinuousFocusEnabled() is True
    assert core.isContinuousFocusLocked() is True


def test_fake_quirk_applies_only_to_the_focus_device() -> None:
    focus = FakeDevice("Z", DeviceType.Stage)
    other = FakeDevice("Piezo", DeviceType.Stage)
    pfs = FakeDevice("PFS", DeviceType.AutoFocus)
    core = FakeCore([focus, other, pfs], focus="Z", autofocus="PFS")
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


# --- continuous focus (m2 §3.12) ---------------------------------------------


def _two_autofocus_devices() -> FakeCore:
    return FakeCore(
        [
            FakeDevice("Z", DeviceType.Stage),
            FakeDevice("PFS", DeviceType.AutoFocus),
            FakeDevice("PFS2", DeviceType.AutoFocus),
        ],
        focus="Z",
        autofocus="PFS",
    )


def test_autofocus_offset_round_trips_and_is_logged() -> None:
    core = _two_autofocus_devices()
    assert core.getAutoFocusOffset() == 0.0
    core.setAutoFocusOffset(12.5)
    assert core.getAutoFocusOffset() == 12.5
    assert core.log == ["setAutoFocusOffset(12.5)"]
    # It acts on the slot's device: another AutoFocus device keeps its own.
    core.setAutoFocusDevice("PFS2")
    assert core.getAutoFocusOffset() == 0.0
    core.enableContinuousFocus(True)
    core.setAutoFocusDevice("PFS")
    assert core.isContinuousFocusEnabled() is False
    # An offset moves nothing (measured on the demo).
    assert core.getPosition("Z") == 0.0


def test_empty_autofocus_slot_raises_as_measured() -> None:
    core = _two_autofocus_devices()
    core.setAutoFocusDevice("")
    core.log.clear()
    with _raises_exactly("Auto-focus focus device not defined."):
        core.enableContinuousFocus(True)
    core.enableContinuousFocus(False)  # measured: no error
    assert core.isContinuousFocusEnabled() is False
    assert core.isContinuousFocusLocked() is False
    with _raises_exactly("Error (code 43)"):
        core.getAutoFocusOffset()
    with _raises_exactly("Error (code 43)"):
        core.setAutoFocusOffset(1.0)
    # Logged even when it then raises, as every mutation is.
    assert core.log == [
        "enableContinuousFocus(True)",
        "enableContinuousFocus(False)",
        "setAutoFocusOffset(1.0)",
    ]


def test_failing_autofocus_label_reaches_the_continuous_focus_calls() -> None:
    core = _two_autofocus_devices()
    boom = RuntimeError("PFS: no answer")
    core.failing["PFS"] = boom
    calls = [
        lambda: core.enableContinuousFocus(True),
        lambda: core.enableContinuousFocus(False),
        core.isContinuousFocusEnabled,
        core.isContinuousFocusLocked,
        core.getAutoFocusOffset,
        lambda: core.setAutoFocusOffset(1.0),
        lambda: core.deviceBusy("PFS"),
    ]
    for call in calls:
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is boom
    # The slot's device, not any AutoFocus device: PFS2 answers once named.
    core.setAutoFocusDevice("PFS2")
    assert core.isContinuousFocusEnabled() is False


def test_autofocus_lockable_false_never_locks() -> None:
    core = _two_autofocus_devices()
    core.autofocus_lockable = False
    core.enableContinuousFocus(True)
    assert core.isContinuousFocusEnabled() is True
    assert core.isContinuousFocusLocked() is False
    core.autofocus_lockable = True
    assert core.isContinuousFocusLocked() is True
    core.enableContinuousFocus(False)
    assert core.isContinuousFocusLocked() is False


def test_the_autofocus_drives_the_focus_device_after_the_reads_set() -> None:
    # Run 1 (#205): locked read True, then Z jumped to the offset's plane.
    core = _two_autofocus_devices()
    core.setPosition("Z", 10.0)
    core.autofocus_target_um = lambda: 245.0
    core.autofocus_drive_after_reads = 2
    core.enableContinuousFocus(True)
    assert [core.isContinuousFocusLocked() for _ in range(2)] == [True, True]
    assert core.getPosition("Z") == 10.0
    assert core.isContinuousFocusLocked() is True
    assert core.getPosition("Z") == 245.0
    # The count restarts at each switch-on from off, not at one sent while on.
    core.setPosition("Z", 10.0)  # the quirk switches it off
    core.enableContinuousFocus(True)
    core.isContinuousFocusLocked()
    core.enableContinuousFocus(True)
    core.isContinuousFocusLocked()
    assert core.getPosition("Z") == 10.0
    core.isContinuousFocusLocked()
    assert core.getPosition("Z") == 245.0
    # Off, or on and not locked, it moves nothing.
    core.setPosition("Z", 10.0)
    core.isContinuousFocusLocked()
    core.autofocus_lockable = False
    core.enableContinuousFocus(True)
    for _ in range(4):
        core.isContinuousFocusLocked()
    assert core.getPosition("Z") == 10.0


def test_the_autofocus_drive_leaves_the_autofocus_on() -> None:
    # Set on the store: a setPosition would trip the quirk and switch it off.
    core = _two_autofocus_devices()
    core.autofocus_target_um = lambda: 245.0
    core.enableContinuousFocus(True)
    core.log.clear()
    assert core.isContinuousFocusLocked() is True
    assert core.getPosition("Z") == 245.0
    assert core.isContinuousFocusEnabled() is True
    assert core.log == []


def test_unloading_the_autofocus_empties_its_slot() -> None:
    core = _two_autofocus_devices()
    core.enableContinuousFocus(True)
    core.unloadDevice("PFS")
    assert core.getAutoFocusDevice() == ""
    assert core.isContinuousFocusEnabled() is False


# --- a device that holds whole units (m2 §3.12, FM-119) ----------------------


def _core_with_offset_devices() -> FakeCore:
    return FakeCore(
        [
            FakeDevice("Z", DeviceType.Stage),
            FakeDevice("Off", DeviceType.Stage),
            FakeDevice("PFS", DeviceType.AutoFocus),
        ],
        focus="Z",
        autofocus="PFS",
    )


def test_hold_whole_units_truncates_a_stage_and_an_autofocus_offset() -> None:
    # The Ti2-E's PFSOffset (#207): 7927.5 reads 7927, 7925.75 lands on 7925.
    core = _core_with_offset_devices()
    core.hold_whole_units("Off")
    core.hold_whole_units("PFS")
    core.setPosition("Off", 7925.75)
    core.setAutoFocusOffset(7927.5)
    assert core.getPosition("Off") == 7925.0
    assert core.getAutoFocusOffset() == 7927.0
    # A float still, as MMCore's reads are: 7925 == 7925.0 would pass an int.
    assert isinstance(core.getPosition("Off"), float)
    assert isinstance(core.getAutoFocusOffset(), float)
    # The log keeps what the caller sent: the layer's call, not the device's reading.
    assert core.log == ["setPosition('Off', 7925.75)", "setAutoFocusOffset(7927.5)"]
    # A device that was not asked keeps its fraction.
    core.setPosition("Z", 12.5)
    assert core.getPosition("Z") == 12.5


def test_hold_whole_units_truncates_towards_zero_and_stores_what_is_not_finite() -> (
    None
):
    core = _core_with_offset_devices()
    core.hold_whole_units("Off")
    for sent, held in [(-0.75, 0.0), (-7925.75, -7925.0), (3.0, 3.0), (0.99, 0.0)]:
        core.setPosition("Off", sent)
        assert core.getPosition("Off") == held
    core.setPosition("Off", float("inf"))
    assert core.getPosition("Off") == float("inf")
    core.setPosition("Off", float("nan"))
    assert math.isnan(core.getPosition("Off"))


def test_hold_whole_units_names_a_missing_or_mistyped_device() -> None:
    # A misspelled label must fail the test, not build a stand that holds fractions.
    core = FakeCore.demo_like()
    for label in ("Zed", "Camera"):
        with pytest.raises(
            ValueError, match=f"{label!r} names no loaded Stage or AutoFocus"
        ):
            core.hold_whole_units(label)
    core.hold_whole_units("Z")  # a loaded Stage is accepted


# --- State labels and config groups (m2 §8.8) -------------------------------


def test_set_state_label_moves_the_turret_and_refuses_an_unknown_label() -> None:
    core = FakeCore.demo_like()
    core.setStateLabel("Objective", "Nikon 20X Plan Fluor ELWD")
    assert (core.getState("Objective"), core.getStateLabel("Objective")) == (
        3,
        "Nikon 20X Plan Fluor ELWD",
    )
    with _raises_exactly(
        'Error in device "Objective": Label not defined (10).  Available Labels: '
        "('Nikon 40X Plan Fluor ELWD', 'Nikon 10X S Fluor', 'Objective-2', "
        "'Nikon 20X Plan Fluor ELWD', 'Objective-4', 'Objective-5')"
    ):
        core.setStateLabel("Objective", "Nope")
    assert core.getState("Objective") == 3
    assert core.log == [
        "setStateLabel('Objective', 'Nikon 20X Plan Fluor ELWD')",
        "setStateLabel('Objective', 'Nope')",
    ]


def test_define_config_merges_into_a_preset_and_creates_the_group() -> None:
    # Measured: a redefinition keeps the settings it does not name, so a
    # replacement must delete first (m2 §8.1).
    core = FakeCore.demo_like()
    core.defineConfig("Channel", "DAPI", "Excitation", "Label", "Chroma-HQ480")
    core.defineConfig("Channel", "DAPI", "Ghost", "Prop", 1)
    assert core.getConfigData("Channel", "DAPI") == (
        ("Dichroic", "Label", "400DCLP"),
        ("Emission", "Label", "Chroma-HQ620"),
        ("Excitation", "Label", "Chroma-HQ480"),
        ("Core", "Shutter", "White Light Shutter"),
        ("Ghost", "Prop", "1"),
    )
    core.defineConfig("New", "a", "Camera", "Exposure", "10")
    assert "New" in core.getAvailableConfigGroups()
    assert core.getAvailableConfigs("New") == ("a",)


def test_delete_config_names_the_group_and_leaves_it_in_place() -> None:
    core = FakeCore.demo_like()
    with pytest.raises(
        ValueError, match=r'^Configuration group "Channel" does not exist$'
    ):
        core.deleteConfig("Channel", "Nope")
    for name in core.getAvailableConfigs("Objective"):
        core.deleteConfig("Objective", name)
    assert "Objective" in core.getAvailableConfigGroups()
    assert core.getAvailableConfigs("Objective") == ()
    assert core.getAvailableConfigs("Nope") == ()


def test_set_config_applies_each_setting_and_moves_the_shutter_slot() -> None:
    core = FakeCore.demo_like()
    core.setConfig("Channel-Multiband", "FITC")
    assert core.getStateLabel("LED") == "470nm"
    assert core.getStateLabel("Excitation") == "Empty"
    assert core.getShutterDevice() == "LED Shutter"
    assert core.log == ["setConfig('Channel-Multiband', 'FITC')"]
    with pytest.raises(
        ValueError,
        match=r'^Preset "Nope" of configuration group "Channel" does not exist$',
    ):
        core.setConfig("Channel", "Nope")
    core.defineConfig("Channel", "Ghosted", "Ghost", "Prop", "1")
    with _raises_exactly('No device with label "Ghost"'):
        core.setConfig("Channel", "Ghosted")


def test_get_config_data_of_a_missing_preset_raises_the_measured_error() -> None:
    core = FakeCore.demo_like()
    with pytest.raises(
        ValueError,
        match=r'^Configuration group "Channel" or its preset "Nope" does not exist$',
    ):
        core.getConfigData("Channel", "Nope")


def test_the_channel_group_takes_only_an_existing_group() -> None:
    core = FakeCore.demo_like()
    assert core.getChannelGroup() == "Channel"
    with pytest.raises(
        ValueError,
        match=r'^Cannot set Core property ChannelGroup to invalid value "Nope"$',
    ):
        core.setChannelGroup("Nope")
    core.setChannelGroup("")
    assert core.getChannelGroup() == ""
    core.setChannelGroup("Channel-Multiband")
    assert core.getChannelGroup() == "Channel-Multiband"


# --- The magnification factor (m2 §8.10) ------------------------------------


def _magnifier(label: str) -> FakeDevice:
    return FakeDevice(label, DeviceType.Magnifier, library="FakeCore", name=label)


def test_magnification_factor_is_the_product_of_the_magnifiers() -> None:
    assert FakeCore.demo_like().getMagnificationFactor() == 1.0
    core = FakeCore.demo_like(extra_devices=[_magnifier("A"), _magnifier("B")])
    # A magnifier with no entry reads 1.0, as the product of none does.
    assert core.getMagnificationFactor() == 1.0
    core.magnification = {"A": 1.5, "B": 2.0}
    assert core.getMagnificationFactor() == 3.0
    # Only a Magnifier counts: an entry for the camera is ignored.
    core.magnification["Camera"] = 7.0
    assert core.getMagnificationFactor() == 3.0
    # An entry for a label that is not loaded is ignored as well.
    core.magnification["Nowhere"] = 7.0
    assert core.getMagnificationFactor() == 3.0


@pytest.mark.parametrize(
    "label", ["Camera", "Z", "XY", "Objective", "White Light Shutter"]
)
def test_only_a_magnifier_counts_in_the_factor(label: str) -> None:
    # One device of every other type the demo has: a magnification entry for it
    # does not scale the factor, and its failing entry does not reach the call.
    core = FakeCore.demo_like(extra_devices=[_magnifier("A")])
    core.magnification = {"A": 1.5, label: 7.0}
    core.failing[label] = RuntimeError(f"{label}: no answer")
    assert core.getMagnificationFactor() == 1.5


def test_demo_like_appends_its_extra_devices_after_the_demos_own() -> None:
    plain = FakeCore.demo_like().getLoadedDevices()
    core = FakeCore.demo_like(extra_devices=[_magnifier("A")])
    assert core.getLoadedDevices() == (*plain[:-1], "A", "Core")
    assert core.getDeviceType("A") == DeviceType.Magnifier


def test_a_failing_magnifier_raises_its_entry_from_the_factor_call() -> None:
    core = FakeCore.demo_like(extra_devices=[_magnifier("A"), _magnifier("B")])
    boom = RuntimeError("A: no answer")
    core.failing["A"] = boom
    with pytest.raises(RuntimeError) as caught:
        core.getMagnificationFactor()
    assert caught.value is boom
    # The entry of a device that is not a Magnifier does not reach the call.
    del core.failing["A"]
    core.failing["Camera"] = boom
    assert core.getMagnificationFactor() == 1.0


# --- the two quirks of the Ti2-E (m2 §8.11 D6, #330) ---------------------------

FOUR, TEN = "1-Plan Fluor 4x", "2-Plan Fluor 10x"


def _lamp_reads(core: FakeCore, count: int) -> list[str]:
    return [core.getProperty("DiaLamp", "Intensity") for _ in range(count)]


def test_make_reads_stale_returns_the_previous_value_for_that_many_reads_then_the_new_one() -> (
    None
):
    # The Ti2-E's DiaLamp read 210 nine milliseconds after a set of 501 (#330).
    core = ti2_core()
    core.make_reads_stale("DiaLamp", "Intensity", reads=3)
    core.setProperty("DiaLamp", "Intensity", 210)
    assert _lamp_reads(core, 5) == ["501", "501", "501", "210", "210"]


def test_a_write_of_the_value_already_held_leaves_nothing_stale() -> None:
    core = ti2_core()
    core.make_reads_stale("DiaLamp", "Intensity", reads=3)
    core.setProperty("DiaLamp", "Intensity", 501)
    assert _lamp_reads(core, 2) == ["501", "501"]
    # A write of the value it holds also ends what an earlier write left stale.
    core.setProperty("DiaLamp", "Intensity", 210)
    assert _lamp_reads(core, 1) == ["501"]
    core.setProperty("DiaLamp", "Intensity", 210)
    assert _lamp_reads(core, 2) == ["210", "210"]


def test_each_write_restarts_the_stale_count() -> None:
    core = ti2_core()
    core.make_reads_stale("DiaLamp", "Intensity", reads=3)
    core.setProperty("DiaLamp", "Intensity", 210)
    assert _lamp_reads(core, 1) == ["501"]
    core.setProperty("DiaLamp", "Intensity", 300)
    # What was held before the second write is 210, not the 501 of the first.
    assert _lamp_reads(core, 5) == ["210", "210", "210", "300", "300"]


@pytest.mark.parametrize(
    ("device", "prop", "write", "read", "before", "after"),
    [
        pytest.param(
            "Nosepiece",
            "Label",
            lambda core: core.setStateLabel("Nosepiece", FOUR),
            lambda core: core.getStateLabel("Nosepiece"),
            TEN,
            FOUR,
            id="Label",
        ),
        pytest.param(
            "Nosepiece",
            "State",
            lambda core: core.setState("Nosepiece", 0),
            lambda core: core.getState("Nosepiece"),
            1,
            0,
            id="State",
        ),
        pytest.param(
            "Camera-1",
            "Exposure",
            lambda core: core.setExposure(20.0),
            lambda core: core.getExposure(),
            10.0,
            20.0,
            id="Exposure",
        ),
    ],
)
def test_stale_reads_cover_the_dedicated_calls_of_the_same_store(
    device: str,
    prop: str,
    write: Callable[[FakeCore], None],
    read: Callable[[FakeCore], object],
    before: object,
    after: object,
) -> None:
    core = ti2_core()
    core.make_reads_stale(device, prop, reads=2)
    write(core)
    assert [read(core) for _ in range(4)] == [before, before, after, after]


@pytest.mark.parametrize(
    ("device", "prop", "write", "before", "after"),
    [
        pytest.param(
            "DiaLamp",
            "Intensity",
            lambda core: core.setProperty("DiaLamp", "Intensity", 210),
            "501",
            "210",
            id="setProperty",
        ),
        pytest.param(
            "DiaLamp",
            "Intensity",
            lambda core: core.setConfig("Channel", "BF"),
            "501",
            "210",
            id="setConfig",
        ),
        pytest.param(
            "Nosepiece",
            "State",
            lambda core: core.setState("Nosepiece", 0),
            "1",
            "0",
            id="setState",
        ),
        pytest.param(
            "Nosepiece",
            "Label",
            lambda core: core.setStateLabel("Nosepiece", FOUR),
            TEN,
            FOUR,
            id="setStateLabel",
        ),
        pytest.param(
            "Nosepiece",
            "Label",
            lambda core: core.setProperty("Nosepiece", "Label", FOUR),
            TEN,
            FOUR,
            id="setProperty-on-a-State-device",
        ),
        pytest.param(
            "Camera-1",
            "Exposure",
            lambda core: core.setExposure(20.0),
            "10.0000",
            "20.0000",
            id="setExposure",
        ),
    ],
)
def test_every_path_that_writes_a_property_restarts_the_stale_count(
    device: str,
    prop: str,
    write: Callable[[FakeCore], None],
    before: str,
    after: str,
) -> None:
    core = ti2_core()
    core.make_reads_stale(device, prop, reads=1)
    write(core)
    assert [core.getProperty(device, prop) for _ in range(3)] == [before, after, after]


def test_a_property_that_was_not_made_stale_reads_fresh() -> None:
    core = ti2_core()
    core.make_reads_stale("Nosepiece", "Label", reads=3)
    core.setProperty("DiaLamp", "Intensity", 210)
    assert _lamp_reads(core, 1) == ["210"]


def test_make_reads_stale_refuses_a_misspelt_device_a_missing_or_read_only_property_and_zero_reads() -> (
    None
):
    # FM-119: a misspelt label would build a stand that reads fresh, and a test
    # that passes whatever the layer does.
    core = FakeCore.demo_like()
    refusals = [
        ("Camera", "Exposure", 0, "make_reads_stale: reads must be at least 1, got 0"),
        (
            "Camera",
            "Exposure",
            -2,
            "make_reads_stale: reads must be at least 1, got -2",
        ),
        ("Camra", "Exposure", 1, "make_reads_stale: no device 'Camra' is loaded"),
        ("Core", "Camera", 1, "make_reads_stale: no device 'Core' is loaded"),
        (
            "Camera",
            "Exposur",
            1,
            "make_reads_stale: device 'Camera' has no property 'Exposur'",
        ),
        (
            "Camera",
            "CameraName",
            1,
            "make_reads_stale: 'Camera' 'CameraName' is read-only, so a write never "
            "changes it and it is never stale",
        ),
    ]
    for device, prop, reads, message in refusals:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            core.make_reads_stale(device, prop, reads=reads)
    # None of the refusals registered anything.
    core.setExposure(20.0)
    assert core.getExposure() == 20.0


def test_a_stale_read_of_a_failing_device_raises_its_failure() -> None:
    core = ti2_core()
    core.make_reads_stale("DiaLamp", "Intensity", reads=2)
    core.setProperty("DiaLamp", "Intensity", 210)
    boom = RuntimeError("DiaLamp: no answer")
    core.failing["DiaLamp"] = boom
    with pytest.raises(RuntimeError) as caught:
        core.getProperty("DiaLamp", "Intensity")
    assert caught.value is boom
    core.failing.clear()
    # The failed read was no read of the lamp: both stale reads are still owed.
    assert _lamp_reads(core, 3) == ["501", "501", "210"]


def _ti2_at(z_um: float) -> FakeCore:
    core = ti2_core()
    core.setPosition("ZDrive", z_um)
    core.log.clear()
    return core


def test_move_z_on_turn_adds_the_delta_and_serves_the_escape_for_that_many_reads() -> (
    None
):
    # Session 2: Z read 500.0 µm at the return of the turn, then 570.26, and
    # settled 8.7 µm above where it was (m2 §8.9).
    core = _ti2_at(561.52)
    core.move_z_on_turn(FOUR, 8.70, escape_um=500.0, escape_reads=3)
    assert core.getPosition("ZDrive") == 561.52  # nothing happens before the turn
    core.setStateLabel("Nosepiece", FOUR)
    assert core.getPosition("ZDrive") == 500.0
    assert core.getPosition() == 500.0  # the focus slot's device, named or not
    assert core.getPosition("ZDrive") == 500.0
    assert core.getPosition("ZDrive") == pytest.approx(570.22)
    assert core.getPosition("ZDrive") == pytest.approx(570.22)
    # The turn is no Z move that the layer sent: only the turn is logged.
    assert core.log == [f"setStateLabel('Nosepiece', {FOUR!r})"]


def test_the_escape_is_served_again_by_each_turn_that_moves_z() -> None:
    core = _ti2_at(561.52)
    core.move_z_on_turn(FOUR, 8.70, escape_um=500.0, escape_reads=2)
    core.move_z_on_turn(TEN, -8.70)
    core.setStateLabel("Nosepiece", FOUR)
    assert core.getPosition("ZDrive") == 500.0
    core.setStateLabel("Nosepiece", TEN)  # the escape of the 4x is dropped
    assert core.getPosition("ZDrive") == pytest.approx(561.52)
    core.setStateLabel("Nosepiece", FOUR)
    assert [core.getPosition("ZDrive") for _ in range(3)] == [
        500.0,
        500.0,
        pytest.approx(570.22),
    ]


def test_a_turn_to_the_position_it_is_at_moves_no_z_and_serves_no_escape() -> None:
    core = _ti2_at(561.52)  # the nosepiece starts at the 10x
    core.move_z_on_turn(TEN, -8.70, escape_um=500.0, escape_reads=2)
    core.setStateLabel("Nosepiece", TEN)
    core.setState("Nosepiece", 1)
    assert core.getPosition("ZDrive") == 561.52
    # A turn to a label nobody registered moves nothing either.
    core.setStateLabel("Nosepiece", FOUR)
    assert core.getPosition("ZDrive") == 561.52


def test_a_turn_moves_z_without_the_nikon_quirk_switching_the_autofocus_off() -> None:
    # FM-12: the stand switches PFS off itself in a real turn, and the layer's
    # own switch-off is what a test of it must see; the fake's Z move on a
    # turn is not a setPosition.
    core = _ti2_at(561.52)
    core.move_z_on_turn(FOUR, 8.70)
    core.enableContinuousFocus(True)
    core.setStateLabel("Nosepiece", FOUR)
    assert core.isContinuousFocusEnabled() is True
    assert core.getPosition("ZDrive") == pytest.approx(570.22)
    # A real Z move still trips the quirk.
    core.setPosition("ZDrive", 561.52)
    assert core.isContinuousFocusEnabled() is False


def test_move_z_on_turn_takes_the_turn_of_any_state_device_and_several_labels() -> None:
    core = _ti2_at(100.0)
    core.move_z_on_turn("2-Ph1", 1.5)
    core.move_z_on_turn(FOUR, 8.0)
    core.setStateLabel("CondenserTurret", "2-Ph1")
    assert core.getPosition("ZDrive") == 101.5
    core.setStateLabel("Nosepiece", FOUR)
    assert core.getPosition("ZDrive") == 109.5


def test_move_z_on_turn_refuses_an_unknown_label_no_focus_slot_and_an_inconsistent_escape() -> (
    None
):
    core = ti2_core()
    refusals: list[tuple[str, dict[str, Any], str]] = [
        (
            "Nope",
            {},
            "move_z_on_turn: 'Nope' is no position label of a loaded State device",
        ),
        (
            "Intensity",
            {},
            "move_z_on_turn: 'Intensity' is no position label of a loaded State device",
        ),
        (
            FOUR,
            {"escape_um": 500.0, "escape_reads": -1},
            "move_z_on_turn: escape_reads must not be negative, got -1",
        ),
        (
            FOUR,
            {"escape_um": 500.0},
            "move_z_on_turn: escape_um is given with escape_reads 0, which serves "
            "none of it",
        ),
        (
            FOUR,
            {"escape_reads": 2},
            "move_z_on_turn: escape_reads is 2 but escape_um is missing",
        ),
    ]
    for label, options, message in refusals:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            core.move_z_on_turn(label, 8.70, **options)
    # None of them registered anything.
    core.setStateLabel("Nosepiece", FOUR)
    assert core.getPosition("ZDrive") == 0.0
    no_focus = FakeCore(
        [FakeDevice("Nosepiece", DeviceType.State, state_labels=(TEN, FOUR))]
    )
    with pytest.raises(
        ValueError,
        match=f"^{re.escape('move_z_on_turn: no focus device is in the Focus slot')}$",
    ):
        no_focus.move_z_on_turn(FOUR, 8.70)
