"""The ``Channels`` contract, on every backend (m2 §8.5, §8.8).

**On ``hardware`` no capture is ever run** (it writes the stand's ``.cfg``),
and the only select is of the current channel. ``FITC``, the capture and the
dry-run case run on ``fake`` and ``demo``, whose channel group is ``Channel``
and whose stand starts on ``DAPI``.
"""

from __future__ import annotations

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Channels
from smc.hardware.channels import named_for_channels


def _demo_only(backend: str) -> None:
    if backend == "hardware":
        pytest.skip("only the current channel is selected on hardware (m2 §8.8)")


def test_the_group_is_named_for_channels(
    channels: Channels, microscope: Microscope
) -> None:
    group = channels.group()
    assert named_for_channels(group) or group == microscope.profile.channels.group


def test_current_is_a_preset_or_none(channels: Channels) -> None:
    current = channels.current()
    assert current is None or current in channels.names()


def test_selecting_the_current_preset_reads_back(channels: Channels) -> None:
    current = channels.current()
    if current is None:
        pytest.skip("the stand matches no preset")
    assert channels.select(current) == current


def test_selecting_another_preset_and_back(channels: Channels, backend: str) -> None:
    _demo_only(backend)
    assert channels.current() == "DAPI"
    assert channels.select("FITC") == "FITC"
    assert channels.current() == "FITC"
    assert channels.select("DAPI") == "DAPI"


def test_an_unknown_preset_is_refused(channels: Channels) -> None:
    with pytest.raises(ValueError, match=r"has no preset 'smc-no-such-channel'"):
        channels.select("smc-no-such-channel")


def test_dry_run_changes_nothing(dry_channels: Channels, backend: str) -> None:
    _demo_only(backend)
    start = dry_channels.current()
    assert dry_channels.select("FITC") == "FITC"
    assert dry_channels.current() == start
    assert dry_channels.capture("smc-contract").saved_to is None
    assert "smc-contract" not in dry_channels.names()


def test_a_capture_records_no_objective_and_the_shutter(
    channels: Channels, microscope: Microscope, backend: str
) -> None:
    _demo_only(backend)
    assert microscope.profile.config_path() is None  # the demo: nothing written
    preset = channels.capture("smc-contract")
    devices = [device for device, _, _ in preset.settings]
    assert "Objective" not in devices
    assert ("Core", "Shutter", "White Light Shutter") in preset.settings
    assert preset.saved_to is None
    assert channels.preset("smc-contract") == preset
