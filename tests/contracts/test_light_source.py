"""The ``LightSource`` contract, on every backend (m2 §8.4, §8.8).

The intensity is set only to the value it already reads; the light is never
switched. The demo's ``LED`` has neither a knob nor a switch that smc knows.
"""

from __future__ import annotations

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource
from smc.hardware.errors import HardwareError
from smc.hardware.roles import Role


def test_the_knob_is_none_or_a_property_of_the_role_device(
    light_source: LightSource, microscope: Microscope
) -> None:
    assert light_source.device() == microscope.roles.get(Role.light_source)
    knob = light_source.intensity_knob()
    if knob is not None:
        assert knob.device == light_source.device()
        assert not knob.read_only


def test_without_a_knob_nothing_is_faked(light_source: LightSource) -> None:
    if light_source.intensity_knob() is not None:
        pytest.skip("this light source has a knob")
    assert light_source.intensity_native() is None
    with pytest.raises(
        HardwareError, match=r"has no intensity property that smc knows"
    ):
        light_source.set_intensity_native(1.0)


def test_setting_the_current_intensity_reads_it_back(light_source: LightSource) -> None:
    now = light_source.intensity_native()
    if now is None:
        pytest.skip("this light source has no knob")
    assert light_source.set_intensity_native(now) == pytest.approx(now)


def test_without_a_switch_set_on_is_refused(light_source: LightSource) -> None:
    if light_source.is_on() is not None:
        pytest.skip("this light source has a switch")
    for on in (True, False):
        with pytest.raises(HardwareError, match=r"has no on/off that smc knows"):
            light_source.set_on(on)


def test_the_demo_led_has_neither_a_knob_nor_a_switch(
    light_source: LightSource, backend: str
) -> None:
    if backend not in ("fake", "demo"):
        pytest.skip("the demo's LED")
    assert light_source.device() == "LED"
    assert light_source.intensity_knob() is None
    assert light_source.is_on() is None
