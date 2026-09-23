"""Value types of the capability layer: what a caller can rely on without a device."""

from __future__ import annotations

import dataclasses

import pytest

from smc.hardware.capabilities import XY, PropertyInfo, XYStage, ZStage


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
