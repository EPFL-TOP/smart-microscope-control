"""The objective turret, the light source and the channels on the demo (m2 §8.1, #14).

What the fake copies, checked on Micro-Manager's own demo devices: the
roles, the channel group, and a capture written into a configuration file
that a fresh core then loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Channels, LightSource, ObjectiveTurret
from smc.hardware.core import close_core, open_core
from smc.hardware.profile import MicroManagerSection, Profile

pytestmark = pytest.mark.demo


def test_demo_roles_give_led_without_a_knob(demo_microscope: Microscope) -> None:
    light = demo_microscope.require(LightSource)
    assert light.device() == "LED"
    assert light.intensity_knob() is None
    assert light.is_on() is None
    turret = demo_microscope.require(ObjectiveTurret)
    assert turret.current() == "Nikon 10X S Fluor"
    assert turret.pixel_size_um() == 0.65


def test_demo_turret_pixel_size_follows_a_magnifier(
    demo_microscope: Microscope,
) -> None:
    # m2 §8.10: the profile's sizes are for a factor of 1.0, so a magnifier
    # (the demo's optovar stands in for the Ti2-E's lever) divides them.
    turret = demo_microscope.require(ObjectiveTurret)
    assert turret.pixel_size_um() == 0.65
    core = demo_microscope.core
    core.loadDevice("Optovar", "DemoCamera", "DOptovar")
    core.initializeDevice("Optovar")
    core.setProperty("Optovar", "Position", "1.6x")
    assert turret.pixel_size_um() == pytest.approx(0.65 / 1.6)
    assert turret.pixel_size_um("Nikon 40X Plan Fluor ELWD") == pytest.approx(
        0.1625 / 1.6
    )


def test_demo_channel_group_is_channel_and_reads_dapi(
    demo_microscope: Microscope,
) -> None:
    channels = demo_microscope.require(Channels)
    assert channels.group() == "Channel"
    assert channels.names() == ["Cy5", "DAPI", "FITC", "Rhodamine"]
    assert channels.current() == "DAPI"
    assert channels.select("Rhodamine") == "Rhodamine"
    assert channels.current() == "Rhodamine"
    state = demo_microscope.state()
    assert (state.objective, state.channel, state.light_intensity_native) == (
        "Nikon 10X S Fluor",
        "Rhodamine",
        None,
    )
    assert state.errors == []


def test_a_captured_preset_in_a_written_cfg_loads_in_a_fresh_core(
    demo_core: Any, tmp_path: Path
) -> None:
    cfg = tmp_path / "demo.cfg"
    demo_core.saveSystemConfiguration(str(cfg))
    profile = Profile.demo().model_copy(
        update={"micromanager": MicroManagerSection(config=str(cfg))}
    )
    with Microscope.open(profile) as microscope:
        channels = microscope.require(Channels)
        channels.select("FITC")
        preset = channels.capture("smc-captured")
        assert preset.saved_to == cfg
    assert preset.settings == (
        ("Dichroic", "Label", "Q505LP"),
        ("Emission", "Label", "Chroma-HQ535"),
        ("Excitation", "Label", "Chroma-HQ480"),
        ("Path", "Label", "State-0"),
        ("LED", "Label", "Closed"),
        ("Core", "Shutter", "White Light Shutter"),
    )
    fresh = open_core(cfg)
    try:
        assert "smc-captured" in fresh.getAvailableConfigs("Channel")
        assert tuple(
            tuple(s) for s in fresh.getConfigData("Channel", "smc-captured")
        ) == (preset.settings)
        # The demo's own presets survive the edit.
        assert list(fresh.getAvailableConfigs("Channel")) == [
            "Cy5",
            "DAPI",
            "FITC",
            "Rhodamine",
            "smc-captured",
        ]
    finally:
        close_core(fresh)
