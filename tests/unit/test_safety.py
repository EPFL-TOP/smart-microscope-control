"""The safety guards: what the layer refuses, and what ``force`` can and cannot unlock."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.hardware.errors import SafetyRefusedError
from smc.hardware.safety import DEFAULT_MAX_Z_JOG_UM, Safety

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)


def test_jog_within_limit_passes() -> None:
    Safety(max_jog_um=100.0).check_jog_um(100.0, -100.0, force=False)


def test_jog_above_limit_is_refused_unless_forced() -> None:
    safety = Safety(max_jog_um=100.0)
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_jog_um(0.0, -100.5, force=False)
    assert info.value.how_to_force == "pass force=True"
    safety.check_jog_um(0.0, -100.5, force=True)


@given(dx=finite, dy=finite, limit=st.floats(min_value=0.0, max_value=1e5))
def test_jog_guard_refuses_exactly_the_jogs_above_the_limit(
    dx: float, dy: float, limit: float
) -> None:
    safety = Safety(max_jog_um=limit)
    if max(abs(dx), abs(dy)) > limit:
        with pytest.raises(SafetyRefusedError):
            safety.check_jog_um(dx, dy, force=False)
    else:
        safety.check_jog_um(dx, dy, force=False)
    safety.check_jog_um(dx, dy, force=True)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_jog_is_refused_even_when_forced(bad: float) -> None:
    # nan compares false against the limit and would slip through a plain `>`.
    with pytest.raises(SafetyRefusedError) as info:
        Safety(max_jog_um=100.0).check_jog_um(bad, 0.0, force=True)
    assert info.value.how_to_force == ""


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_z_jog_at_the_limit_passes_and_above_is_refused_unless_forced(
    sign: float,
) -> None:
    safety = Safety(max_jog_um=5000.0, max_z_jog_um=100.0)
    safety.check_z_jog_um(sign * 100.0, force=False)  # the limit is inclusive
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_z_jog_um(sign * 100.5, force=False)
    assert info.value.how_to_force == "pass force=True"
    safety.check_z_jog_um(sign * 100.5, force=True)


def test_z_jog_refusal_names_the_distance_and_the_limit() -> None:
    safety = Safety(max_jog_um=5000.0, max_z_jog_um=100.0)
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_z_jog_um(150.0, force=False)
    assert info.value.reason == "Z jog 150.0 µm exceeds the Z jog limit of 100.0 µm"
    assert not safety.z_jog_assumed


def test_omitted_max_z_jog_um_assumes_100_um_and_says_so() -> None:
    safety = Safety(max_jog_um=5000.0)
    assert safety.z_jog_assumed is True
    assert safety.max_z_jog_um == DEFAULT_MAX_Z_JOG_UM == 100.0
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_z_jog_um(150.0, force=False)
    assert info.value.reason.endswith(
        "(assumed: the profile sets no [safety] max_z_jog_um)"
    )
    # Naming the limit as assumed must not change what is refused or forceable.
    assert info.value.how_to_force == "pass force=True"
    safety.check_z_jog_um(100.0, force=False)
    safety.check_z_jog_um(150.0, force=True)


def test_a_given_max_z_jog_um_is_not_assumed() -> None:
    safety = Safety(max_jog_um=5000.0, max_z_jog_um=10.0)
    assert safety.z_jog_assumed is False
    assert safety.max_z_jog_um == 10.0


@given(dz=finite, limit=st.floats(min_value=0.0, max_value=1e5))
def test_z_jog_guard_refuses_exactly_the_jogs_above_the_limit(
    dz: float, limit: float
) -> None:
    safety = Safety(max_jog_um=5000.0, max_z_jog_um=limit)
    if abs(dz) > limit:
        with pytest.raises(SafetyRefusedError):
            safety.check_z_jog_um(dz, force=False)
    else:
        safety.check_z_jog_um(dz, force=False)
    safety.check_z_jog_um(dz, force=True)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_z_jog_is_refused_even_forced(bad: float) -> None:
    # nan compares false against the limit and would slip through a plain `>`.
    with pytest.raises(SafetyRefusedError) as info:
        Safety(max_jog_um=5000.0, max_z_jog_um=100.0).check_z_jog_um(bad, force=True)
    assert info.value.reason == f"Z jog {bad} µm is not a finite distance"
    assert info.value.how_to_force == ""


def test_z_jog_limit_is_independent_of_the_xy_jog_limit() -> None:
    safety = Safety(max_jog_um=5000.0, max_z_jog_um=10.0)
    safety.check_jog_um(4000.0, 0.0, force=False)
    with pytest.raises(SafetyRefusedError):
        safety.check_z_jog_um(11.0, force=False)


def test_target_outside_soft_limits_is_refused_and_not_forceable() -> None:
    safety = Safety(
        max_jog_um=100.0,
        z_soft_limits_um=(0.0, 5000.0),
        xy_soft_limits_um=((-1000.0, 1000.0), (-500.0, 500.0)),
    )
    safety.check_xy_target_um(1000.0, -500.0)
    safety.check_z_target_um(5000.0)
    with pytest.raises(SafetyRefusedError) as xy:
        safety.check_xy_target_um(0.0, 500.1)
    assert xy.value.how_to_force == ""
    with pytest.raises(SafetyRefusedError) as z:
        safety.check_z_target_um(-0.1)
    assert z.value.how_to_force == ""


def test_no_limits_means_no_check() -> None:
    safety = Safety(max_jog_um=100.0)
    safety.check_xy_target_um(1e9, -1e9)
    safety.check_z_target_um(-1e9)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_target_is_refused_without_limits(bad: float) -> None:
    safety = Safety(max_jog_um=100.0)
    with pytest.raises(SafetyRefusedError):
        safety.check_xy_target_um(bad, 0.0)
    with pytest.raises(SafetyRefusedError):
        safety.check_z_target_um(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_jog_um": math.nan},
        {"max_jog_um": -1.0},
        {"max_jog_um": 1.0, "max_z_jog_um": math.nan},
        {"max_jog_um": 1.0, "max_z_jog_um": math.inf},
        {"max_jog_um": 1.0, "max_z_jog_um": -1.0},
        {"max_jog_um": 1.0, "z_soft_limits_um": (10.0, 0.0)},
        {"max_jog_um": 1.0, "xy_soft_limits_um": ((0.0, 1.0), (0.0, math.inf))},
    ],
)
def test_invalid_configuration_is_rejected_at_construction(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Safety(**kwargs)  # type: ignore[arg-type]


# -- the autofocus offset (m2 §3.8) -------------------------------------------


def _offset_safety(**kwargs: object) -> Safety:
    return Safety(max_jog_um=5000.0, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_offset_jog_above_the_limit_is_refused_with_how_to_force(sign: float) -> None:
    safety = _offset_safety(max_autofocus_offset_jog=1.0, offset_unit="steps")
    safety.check_offset_jog_native(sign * 1.0, force=False)  # inclusive
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_offset_jog_native(sign * 1.5, force=False)
    assert info.value.reason == (
        f"autofocus offset jog {sign * 1.5} steps exceeds the offset jog limit of "
        f"1.0 steps"
    )
    assert info.value.how_to_force == "pass force=True"
    safety.check_offset_jog_native(sign * 1.5, force=True)


def test_offset_jog_without_a_limit_needs_force_and_zero_passes() -> None:
    # The unit is the stand's own: no default limit can be assumed.
    safety = _offset_safety()
    assert safety.max_autofocus_offset_jog is None
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_offset_jog_native(0.001, force=False)
    assert info.value.reason == (
        "autofocus offset jog 0.001 device units: the profile sets no [safety] "
        "max_autofocus_offset_jog, so every relative offset change needs force=True"
    )
    assert info.value.how_to_force == (
        "pass force=True, or set [safety] max_autofocus_offset_jog in the profile"
    )
    safety.check_offset_jog_native(0.0, force=False)
    safety.check_offset_jog_native(1e6, force=True)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("limit", [None, 1.0])
def test_offset_jog_non_finite_is_refused_even_forced(
    bad: float, limit: float | None
) -> None:
    safety = _offset_safety(max_autofocus_offset_jog=limit)
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_offset_jog_native(bad, force=True)
    assert info.value.reason == (
        f"autofocus offset jog {bad} device units is not a finite distance"
    )
    assert info.value.how_to_force == ""


def test_offset_target_outside_soft_limits_is_refused_even_forced() -> None:
    # There is no force for a target: the limits are where the lens stops.
    safety = _offset_safety(autofocus_offset_soft_limits=(7000.0, 8000.0))
    assert safety.autofocus_offset_soft_limits == (7000.0, 8000.0)
    safety.check_offset_target_native(7000.0)
    safety.check_offset_target_native(8000.0)
    with pytest.raises(SafetyRefusedError) as info:
        safety.check_offset_target_native(8000.5)
    assert info.value.reason == (
        "autofocus offset target 8000.5 device units is outside the soft limits "
        "[7000.0, 8000.0] device units; change [safety] autofocus_offset_soft_limits "
        "in the profile if the limits are wrong"
    )
    assert info.value.how_to_force == ""
    with pytest.raises(SafetyRefusedError) as non_finite:
        _offset_safety().check_offset_target_native(math.nan)
    assert non_finite.value.reason == (
        "autofocus offset target nan device units is not a finite position"
    )
    assert non_finite.value.how_to_force == ""
    # No limits configured: no check.
    _offset_safety().check_offset_target_native(-1e9)


@pytest.mark.parametrize(
    ("unit", "shown"), [("PFS offset steps", "PFS offset steps"), ("", "device units")]
)
def test_offset_refusals_name_the_unit_or_device_units(unit: str, shown: str) -> None:
    safety = _offset_safety(
        max_autofocus_offset_jog=1.0,
        autofocus_offset_soft_limits=(0.0, 10.0),
        offset_unit=unit,
    )
    with pytest.raises(SafetyRefusedError) as jog:
        safety.check_offset_jog_native(2.0, force=False)
    assert jog.value.reason.endswith(f"limit of 1.0 {shown}")
    with pytest.raises(SafetyRefusedError) as target:
        safety.check_offset_target_native(11.0)
    assert f"target 11.0 {shown} is outside the soft limits [0.0, 10.0] {shown};" in (
        target.value.reason
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_autofocus_offset_jog": math.nan},
        {"max_autofocus_offset_jog": math.inf},
        {"max_autofocus_offset_jog": -1.0},
        {"autofocus_offset_soft_limits": (10.0, 0.0)},
        {"autofocus_offset_soft_limits": (0.0, math.inf)},
    ],
)
def test_invalid_offset_configuration_is_rejected_at_construction(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _offset_safety(**kwargs)
