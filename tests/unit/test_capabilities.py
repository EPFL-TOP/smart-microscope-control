"""Value types of the capability layer: what a caller can rely on without a device."""

from __future__ import annotations

import dataclasses

import pytest

from smc.hardware.capabilities import (
    XY,
    Autofocus,
    ChannelPreset,
    Channels,
    FocusStep,
    LightSource,
    ObjectiveTurret,
    PropertyInfo,
    XYStage,
    ZStage,
)


def test_value_types_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        XY(1.0, 2.0).x_um = 3.0  # type: ignore[misc]


def test_property_number_is_none_rather_than_a_guess() -> None:
    assert PropertyInfo("Cam", "Exposure", "12.5").number == 12.5
    assert PropertyInfo("Cam", "Mode", "Color Test Pattern").number is None


def test_property_numeric_needs_both_limits() -> None:
    assert PropertyInfo("Cam", "Gain", "1", lower=0.0, upper=10.0).numeric
    assert not PropertyInfo("Cam", "Gain", "1", lower=0.0).numeric


def test_property_describe_names_the_constraints() -> None:
    info = PropertyInfo(
        "Cam", "Binning", "1", read_only=True, allowed=("1", "2"), lower=1.0, upper=2.0
    )
    text = info.describe()
    assert text.startswith("Cam.Binning = 1")
    assert "[1 .. 2]" in text
    assert "{1 | 2}" in text
    assert "(read-only)" in text


class _StageWithoutStop:
    """Every ``XYStage``/``ZStage`` method except ``stop`` (design §13)."""

    def position_um(self) -> float:
        return 0.0

    def move_to_um(self, *args: float, wait: bool = True) -> float:
        return 0.0

    def move_by_um(self, *args: float, wait: bool = True, force: bool = False) -> float:
        return 0.0

    def wait(self, timeout_s: float | None = None) -> None:
        return None

    def is_busy(self) -> bool:
        return False

    def limits_um(self) -> None:
        return None


class _StageWithStop(_StageWithoutStop):
    def stop(self) -> None:
        return None


def test_stage_protocols_require_stop() -> None:
    assert not isinstance(_StageWithoutStop(), XYStage)
    assert not isinstance(_StageWithoutStop(), ZStage)
    assert isinstance(_StageWithStop(), XYStage)
    assert isinstance(_StageWithStop(), ZStage)


class _AutofocusWithoutFocusBy:
    """Every ``Autofocus`` method except ``focus_by_um`` (m2 §3.1)."""

    def is_engaged(self) -> bool:
        return False

    def is_locked(self) -> bool:
        return False

    def is_in_range(self) -> bool | None:
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        return False

    def disengage(self) -> None:
        return None

    def offset_native(self) -> float:
        return 0.0

    def set_offset_native(self, value: float) -> float:
        return value

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        return delta

    def offset_unit(self) -> str:
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        return None


class _AutofocusWithFocusBy(_AutofocusWithoutFocusBy):
    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        return FocusStep("z", dz_um, None)


def test_a_class_without_focus_by_um_is_not_an_autofocus() -> None:
    # focus_by_um is the one call that takes µm: an implementation that has
    # only the native offset calls must not pass for an Autofocus.
    assert not isinstance(_AutofocusWithoutFocusBy(), Autofocus)
    assert isinstance(_AutofocusWithFocusBy(), Autofocus)


def test_focus_step_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        FocusStep("z", 1.0, None).via = "offset"  # type: ignore[misc]


def test_channel_preset_is_frozen_and_unsaved_by_default() -> None:
    preset = ChannelPreset("Channel", "BF", (("DiaLamp", "Intensity", "210"),))
    assert preset.saved_to is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        preset.name = "GFP"  # type: ignore[misc]


class _TurretWithoutSelect:
    def labels(self) -> list[str]:
        return []

    def current(self) -> str:
        return ""

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0


class _Turret(_TurretWithoutSelect):
    def select(self, label: str, *, confirm: bool = False) -> str:
        return label


class _LightWithoutSwitch:
    def device(self) -> str:
        return "LED"

    def intensity_knob(self) -> None:
        return None

    def intensity_native(self) -> None:
        return None

    def set_intensity_native(self, value: float) -> float:
        return value

    def is_on(self) -> None:
        return None


class _Light(_LightWithoutSwitch):
    def set_on(self, on: bool) -> bool:
        return on


class _ChannelsWithoutCapture:
    def group(self) -> str:
        return "Channel"

    def names(self) -> list[str]:
        return []

    def current(self) -> None:
        return None

    def preset(self, name: str) -> ChannelPreset:
        return ChannelPreset("Channel", name, ())

    def select(self, name: str) -> str:
        return name


class _Channels(_ChannelsWithoutCapture):
    def capture(self, name: str) -> ChannelPreset:
        return ChannelPreset("Channel", name, ())


@pytest.mark.parametrize(
    ("protocol", "partial", "whole"),
    [
        (ObjectiveTurret, _TurretWithoutSelect, _Turret),
        (LightSource, _LightWithoutSwitch, _Light),
        (Channels, _ChannelsWithoutCapture, _Channels),
    ],
)
def test_the_m2_protocols_need_every_method(
    protocol: type, partial: type, whole: type
) -> None:
    # Microscope.override checks isinstance: a double missing the guarded
    # call (select, set_on, capture) must not pass for the capability.
    assert not isinstance(partial(), protocol)
    assert isinstance(whole(), protocol)
