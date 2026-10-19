"""``LightSource`` on the Ti2-E's layout and the demo's (m2 §8.4, #14).

On the Ti2-E (``test_mm_channels.ti2_core()``) the light source is
``DiaLamp``, a ``Shutter`` with a free numeric ``Intensity`` that also fills
the shutter role. On the demo it is ``LED``, a ``State`` device with no knob
and no switch that smc knows. Messages are compared whole (FM-45).
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, cast

import pytest
from pymmcore_plus import DeviceType
from test_mm_channels import (
    JOIN_S,
    Thread,
    lock_held_by_a_snap,
    ti2,
    ti2_core,
    ti2_profile,
    wait_for,
)

from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource, PropertyInfo, Shutter, XYStage
from smc.hardware.errors import HardwareError
from smc.hardware.profile import LightSourceSection, Profile
from smc.testing import FakeCore, FakeDevice, FakeProperty

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_plus import CMMCorePlus

NO_KNOB = (
    "light_source {label!r} has no intensity property that smc knows; set "
    "[light_source] intensity_property in the profile, or use Properties"
)


def _lamp(core: FakeCore, **properties: FakeProperty) -> None:
    """Give ``DiaLamp`` more properties, in addition to ``Intensity`` and ``State``."""
    core._devices["DiaLamp"].properties.update(properties)


def _demo() -> Iterator[tuple[FakeCore, Microscope]]:
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    core.log.clear()
    try:
        yield core, microscope
    finally:
        microscope.close()


def test_intensity_is_found_on_the_lamp_not_the_camera() -> None:
    # nikon-control's test: a camera's BeadBrightness is no lamp.
    core = ti2_core()
    core._devices["Camera-1"].properties["BeadBrightness"] = FakeProperty(
        "1", lower=0.0, upper=10.0
    )
    with ti2(core=core) as (_, microscope):
        light = microscope.require(LightSource)
        assert light.device() == "DiaLamp"
        assert light.intensity_knob() == PropertyInfo("DiaLamp", "Intensity", "501")
        assert light.intensity_native() == 501.0
        assert microscope.state().light_intensity_native == 501.0


def test_setting_the_intensity_reads_back() -> None:
    with ti2() as (core, microscope):
        assert microscope.require(LightSource).set_intensity_native(210) == 210.0
        assert core.log == ["setProperty('DiaLamp', 'Intensity', 210.0)"]


def test_the_profile_property_wins_over_the_hints() -> None:
    core = ti2_core()
    _lamp(core, LampVoltage=FakeProperty("3.5"))
    profile = ti2_profile().model_copy(
        update={"light_source": LightSourceSection(intensity_property="LampVoltage")}
    )
    with ti2(profile, core=core) as (_, microscope):
        knob = microscope.require(LightSource).intensity_knob()
        assert knob is not None
        assert knob.name == "LampVoltage"


def test_a_profile_property_the_device_lacks_falls_back_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    for missing in ("Brightnes", "Model"):  # absent; read-only
        core = ti2_core()  # a closed facade unloads its core
        _lamp(core, Model=FakeProperty("Ti2 LED", read_only=True))
        profile = ti2_profile().model_copy(
            update={"light_source": LightSourceSection(intensity_property=missing)}
        )
        with ti2(profile, core=core) as (_, microscope):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                knob = microscope.require(LightSource).intensity_knob()
            assert knob is not None
            assert knob.name == "Intensity"
            assert caplog.messages == [
                f"light_source 'DiaLamp' has no writable property {missing!r} "
                f"([light_source] intensity_property in the profile); its writable "
                f"properties are: Intensity, State. Searching by name instead"
            ]


def test_a_non_numeric_property_is_never_a_knob() -> None:
    lamp = FakeDevice(
        "Lamp",
        DeviceType.Shutter,
        properties={
            "Power": FakeProperty("On", allowed=("On", "Off")),
            "Level": FakeProperty("n/a"),
        },
    )
    core = FakeCore([lamp], shutter="Lamp")
    with Microscope.from_core(cast("CMMCorePlus", core), ti2_profile()) as microscope:
        light = microscope.require(LightSource)
        assert light.intensity_knob() is None
        assert light.intensity_native() is None


def test_a_knob_with_limits_wins_over_one_without() -> None:
    core = ti2_core()
    # "level" ranks after "intensity", but only it declares a range.
    _lamp(core, DimmerLevel=FakeProperty("40", lower=0.0, upper=100.0))
    with ti2(core=core) as (_, microscope):
        knob = microscope.require(LightSource).intensity_knob()
        assert knob is not None
        assert (knob.name, knob.lower, knob.upper) == ("DimmerLevel", 0.0, 100.0)


def test_the_knobs_name_is_kept_and_its_value_read_every_time() -> None:
    core = ti2_core()
    with ti2(core=core) as (_, microscope):
        light = microscope.require(LightSource)
        assert light.intensity_native() == 501.0
        # A better knob appearing later does not move the resolved one.
        _lamp(core, Brightness=FakeProperty("9", lower=0.0, upper=10.0))
        core._devices["DiaLamp"].properties["Intensity"].value = "77"
        assert light.intensity_native() == 77.0


def test_no_knob_is_reported_not_faked(caplog: pytest.LogCaptureFixture) -> None:
    for core, microscope in _demo():
        light = microscope.require(LightSource)
        assert light.device() == "LED"
        with caplog.at_level(logging.INFO):
            assert light.intensity_knob() is None
            assert light.intensity_native() is None
        with pytest.raises(
            HardwareError, match=f"^{re.escape(NO_KNOB.format(label='LED'))}$"
        ):
            light.set_intensity_native(10.0)
        assert core.log == []
        assert microscope.state().light_intensity_native is None
        # Logged once: the knob's absence is kept like its name.
        assert caplog.messages == [
            "light_source 'LED': no intensity property (searched for intensity, "
            "brightness, power, level, voltage); set [light_source] "
            "intensity_property if the device has one"
        ]


def test_an_intensity_outside_the_limits_is_refused_not_clamped() -> None:
    core = ti2_core()
    intensity = core._devices["DiaLamp"].properties["Intensity"]
    intensity.lower, intensity.upper = 0.0, 255.0
    with ti2(core=core) as (_, microscope):
        light = microscope.require(LightSource)
        for value in (300.0, -1.0):
            message = (
                f"light_source 'DiaLamp': intensity {value:g} is outside Intensity's "
                f"limits [0, 255]; it is never clamped"
            )
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                light.set_intensity_native(value)
        assert core.log == []
        # Both limits are inside the range.
        assert light.set_intensity_native(255.0) == 255.0
        assert light.set_intensity_native(0.0) == 0.0


def test_the_setters_return_what_the_lamp_reads_not_what_was_asked() -> None:
    with ti2() as (core, microscope):
        light = microscope.require(LightSource)

        def ignored(*args: object) -> None:
            return None

        core.setProperty = ignored  # type: ignore[method-assign]
        core.setShutterOpen = ignored  # type: ignore[method-assign]
        assert light.set_intensity_native(210.0) == 501.0
        assert light.set_on(True) is False
        del core.setProperty, core.setShutterOpen


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_intensity_is_refused(value: float) -> None:
    with ti2() as (core, microscope):
        with pytest.raises(ValueError, match=r"intensity must be finite"):
            microscope.require(LightSource).set_intensity_native(value)
        assert core.log == []


def test_an_unreadable_intensity_is_a_hardware_error() -> None:
    core = ti2_core()
    with ti2(core=core) as (_, microscope):
        light = microscope.require(LightSource)
        assert light.intensity_native() == 501.0
        core._devices["DiaLamp"].properties["Intensity"].value = "n/a"
        message = (
            "light_source 'DiaLamp': Intensity reads 'n/a', which is not a finite "
            "number; check the device, or set [light_source] intensity_property in "
            "the profile"
        )
        with pytest.raises(HardwareError, match=f"^{re.escape(message)}$"):
            light.intensity_native()
        state = microscope.state()
        assert state.light_intensity_native is None
        assert state.errors == [f"light: HardwareError: {message}"]
        core.failing["DiaLamp"] = RuntimeError("DiaLamp: no answer")
        with pytest.raises(
            HardwareError, match=r"^light_source: getProperty\('DiaLamp'"
        ):
            light.intensity_native()
        core.failing.clear()


def test_the_lamp_switches_on_and_off() -> None:
    with ti2() as (core, microscope):
        light = microscope.require(LightSource)
        assert light.is_on() is False
        assert light.set_on(True) is True
        assert light.set_on(False) is False
        assert core.log == [
            "setShutterOpen('DiaLamp', True)",
            "setShutterOpen('DiaLamp', False)",
        ]


def test_switching_the_lamp_off_is_a_safe_call_while_another_action_waits() -> None:
    # FM-16: the light can always be cut, even while a move holds the lock.
    with ti2() as (core, microscope):
        light = microscope.require(LightSource)
        light.set_on(True)
        core.busy_devices.add("XYStage")
        mover = Thread(lambda: microscope.require(XYStage).move_to_um(5.0, 5.0))
        wait_for(lambda: microscope.state().moving == ("xy_stage XYStage",), "the move")
        assert light.set_on(False) is False
        assert core.getShutterOpen("DiaLamp") is False
        core.busy_devices.clear()
        mover.join()
        assert mover.error is None


def test_a_shutter_close_cancels_a_queued_lamp_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DiaLamp is both the shutter and the light source: one key, so a close
    # from either capability wins over an on from the other.
    with ti2() as (core, microscope):
        light, shutter = microscope.require(LightSource), microscope.require(Shutter)
        with lock_held_by_a_snap(core, microscope, monkeypatch) as queued:
            opener = Thread(lambda: light.set_on(True))
            assert queued.wait(JOIN_S)
            shutter.set_open(False)
        opener.join()
        assert isinstance(opener.error, HardwareError)
        # The Executor names the action's own safe call: the key is shared.
        assert str(opener.error).startswith(
            "`light_source: on` was not sent: `light_source: off` was requested after "
            "it"
        )
        assert "setShutterOpen('DiaLamp', True)" not in core.log
        assert core.getShutterOpen("DiaLamp") is False


def test_a_state_device_light_has_no_switch() -> None:
    for core, microscope in _demo():
        light = microscope.require(LightSource)
        assert light.is_on() is None
        message = (
            "light_source 'LED' (a State device) has no on/off that smc knows; switch "
            "the light with its shutter (Shutter)"
        )
        for on in (True, False):
            with pytest.raises(HardwareError, match=f"^{re.escape(message)}$"):
                light.set_on(on)
        assert core.log == []
