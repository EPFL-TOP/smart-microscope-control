"""The timing model's pure part: what a position costs and how many fit in an interval (m3 §12, #24).

The arithmetic is what a ten-hour experiment gets planned on, so it is tested
against fixed numbers, never measured ones. The first twelve tests are the port
of ``nikon-control``'s ``tests/test_timing.py`` (its pure half); the rest are
what the port lacked: a channel nobody measured, a stand with no switch time, a
refocus that does not exist, and arguments that are not numbers (FM-32).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins import ParamsError
from smc.plugins.params import parse_params
from smc.plugins.slots import TimingsResult
from smc.plugins.timing.algorithm import (
    ASSUMED_CHANNEL_SWITCH_MS,
    ASSUMED_REFOCUS_MS,
    Budget,
    channel_cost_ms,
    describe_timings,
    fit_line,
    plan,
    position_ms,
)
from smc.plugins.timing.plugin import TimingParams

_FIXED: dict[str, Any] = {
    "move_overhead_ms": 100.0,
    "move_ms_per_mm": 50.0,
    "refocus_ms": 400.0,
    "snap_ms": 200.0,
    "channel_ms": {"BF": 50.0, "GFP": 300.0, "mCherry": 300.0},
    "channel_switch_ms": 100.0,
    "samples": 0,
}


def fixed(**changes: Any) -> TimingsResult:
    """Round numbers, so the expected answers are obvious by hand; ``changes`` replaces fields."""
    return TimingsResult.model_validate({**_FIXED, **changes})


# ------------------------------------------------------------ the port (twelve)


def test_move_time_grows_with_distance() -> None:
    t = fixed()
    assert t.move_ms(1000) == 150.0  # 1 mm
    assert t.move_ms(40000) == 2100.0  # a well-to-well hop


def test_not_moving_costs_nothing() -> None:
    """The overhead is the price of issuing and settling a move."""
    assert fixed().move_ms(0) == 0.0


def test_position_cost_adds_every_channel_and_its_switch() -> None:
    t = fixed()
    # 100 + 50*2mm move, 400 refocus, then (100+50)+(100+300)+(100+300)
    assert position_ms(t, ["BF", "GFP", "mCherry"], 2000) == pytest.approx(
        200 + 400 + 150 + 400 + 400
    )
    assert position_ms(t, ["BF", "GFP", "mCherry"], 2000) == 1550.0


def test_an_unmeasured_channel_falls_back_to_the_mean() -> None:
    t = fixed()
    solo = position_ms(t, ["YFP"], move_um=0, refocus=False)
    mean = (50 + 300 + 300) / 3
    assert solo == pytest.approx(100 + mean)


def test_more_channels_means_fewer_positions() -> None:
    t = fixed()
    one = plan(t, ["BF"], interval_s=300)
    three = plan(t, ["BF", "GFP", "mCherry"], interval_s=300)
    assert three.positions < one.positions


def test_headroom_is_held_back_from_the_interval() -> None:
    """A timelapse scheduled to 100% of its interval drifts later forever."""
    t = fixed()
    full = plan(t, ["BF"], interval_s=300, headroom=1.0)
    default = plan(t, ["BF"], interval_s=300)
    # One channel at 2000 um costs 200 + 400 + 100 + 50 = 750 ms.
    assert full.per_position_ms == 750.0
    assert full.positions == 400
    assert default.positions == 320


def test_a_request_that_does_not_fit_says_so() -> None:
    t = fixed()
    budget = plan(
        t, ["BF", "GFP", "mCherry"], interval_s=300, requested=10000, move_um=2000
    )
    assert not budget.fits
    assert budget.duty > 1.0
    assert "DOES NOT FIT" in " ".join(budget.describe())


def test_a_request_that_fits_says_so_with_the_duty_cycle() -> None:
    t = fixed()
    budget = plan(t, ["BF"], interval_s=300, requested=10, move_um=2000, duration_h=10)
    assert budget.fits
    assert 0 < budget.duty < 1
    text = " ".join(budget.describe())
    assert "fits" in text
    assert "120 timepoints over 10 h" in text  # 10 h / 5 min


def test_timepoint_count_follows_the_interval() -> None:
    t = fixed()
    assert plan(t, ["BF"], interval_s=300, duration_h=10).timepoints == 120
    assert plan(t, ["BF"], interval_s=600, duration_h=10).timepoints == 60


def test_line_fit_recovers_overhead_and_rate() -> None:
    overhead, rate = fit_line([(1000.0, 150.0), (9000.0, 550.0)])
    assert overhead == pytest.approx(100.0)
    assert rate == pytest.approx(50.0)


def test_line_fit_never_returns_a_negative_cost() -> None:
    """Noise at short distances can fit a negative intercept; physics cannot."""
    overhead, rate = fit_line([(200.0, 10.0), (9000.0, 500.0)])
    assert overhead >= 0.0
    assert rate >= 0.0
    # And the clamp is what did it: the unclamped intercept of these two points is below 0.
    assert overhead == 0.0


def test_describe_flags_what_was_assumed_rather_than_measured() -> None:
    t = fixed(assumed=("PFS lock 400 ms (not measured)",))
    lines = describe_timings(t)
    assert any("assumed" in line for line in lines)
    assert any("PFS lock 400 ms (not measured)" in line for line in lines)


# ------------------------------------------------------------- what is new here


def test_a_channel_nobody_measured_costs_a_bare_snap() -> None:
    """The port returned 0, which counted an exposure as free."""
    t = fixed(channel_ms={}, snap_ms=200.0)
    assert channel_cost_ms(t, "GFP") == 200.0


def test_a_measured_channel_costs_what_was_measured_and_a_stranger_the_mean() -> None:
    t = fixed()
    assert channel_cost_ms(t, "GFP") == 300.0
    assert channel_cost_ms(t, "YFP") == pytest.approx((50 + 300 + 300) / 3)


def test_without_channels_a_position_is_one_bare_snap_and_no_switch() -> None:
    t = fixed()
    # No move, no refocus: only the snap, and not the 100 ms switch.
    assert position_ms(t, [], 0.0, refocus=False) == 200.0
    # 1 mm is 150 ms, the refocus 400 ms, the snap 200 ms.
    assert position_ms(t, [], 1000.0) == 750.0


def test_an_unmeasured_switch_is_assumed_and_the_budget_says_so() -> None:
    t = fixed(channel_switch_ms=None)
    # (100 + 50) + (100 + 300): the assumed switch is counted once per channel.
    assert position_ms(t, ["BF", "GFP"], 0.0, refocus=False) == 550.0
    budget = plan(t, ["BF", "GFP"], interval_s=300, move_um=0.0, refocus=False)
    assert budget.per_position_ms == 550.0
    assert budget.assumed == (
        f"channel switch {ASSUMED_CHANNEL_SWITCH_MS:.0f} ms is assumed: "
        "the timings hold none",
    )


def test_a_switch_that_is_not_needed_is_not_assumed() -> None:
    """With no channel there is no switch, so the budget has nothing to say about one."""
    t = fixed(channel_switch_ms=None)
    assert plan(t, [], interval_s=300).assumed == ()


def test_a_budget_says_which_channel_cost_is_not_a_measurement() -> None:
    t = fixed()
    budget = plan(t, ["BF", "YFP", "YFP"], interval_s=300)
    mean = (50 + 300 + 300) / 3
    assert budget.assumed == (
        f"channel YFP costs {mean:.0f} ms, the mean of the measured channels: "
        "it was not measured",
    )
    bare = plan(fixed(channel_ms={}), ["BF"], interval_s=300)
    assert bare.assumed == (
        "channel BF costs 200 ms, a bare snap: no channel was measured",
    )


def test_a_budget_carries_what_the_timings_assumed() -> None:
    line = "refocus 400 ms is assumed: the autofocus was off"
    budget = plan(fixed(assumed=(line,)), ["BF"], interval_s=300)
    assert budget.assumed == (line,)


def test_no_refocus_costs_nothing() -> None:
    """A stand without autofocus has ``refocus_ms`` None: asking for a refocus adds none."""
    t = fixed(refocus_ms=None)
    with_request = position_ms(t, ["BF"], 0.0, refocus=True)
    without = position_ms(t, ["BF"], 0.0, refocus=False)
    assert with_request == without == 150.0


def test_a_refocus_that_is_not_asked_for_is_left_out() -> None:
    assert position_ms(fixed(), ["BF"], 0.0, refocus=False) == 150.0
    assert position_ms(fixed(), ["BF"], 0.0, refocus=True) == 550.0


@pytest.mark.parametrize("interval_s", [0.0, -1.0, math.nan, math.inf])
def test_plan_refuses_an_interval_that_is_not_positive_and_finite(
    interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="interval_s"):
        plan(fixed(), ["BF"], interval_s=interval_s)


@pytest.mark.parametrize("headroom", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_plan_refuses_a_headroom_outside_zero_to_one(headroom: float) -> None:
    with pytest.raises(ValueError, match="headroom"):
        plan(fixed(), ["BF"], interval_s=300, headroom=headroom)


def test_plan_accepts_the_edges_of_the_headroom() -> None:
    assert plan(fixed(), ["BF"], interval_s=300, headroom=1.0).positions == 400
    assert plan(fixed(), ["BF"], interval_s=300, headroom=1e-9).positions == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("requested", -1),
        ("duration_h", -1.0),
        ("duration_h", math.nan),
        ("duration_h", math.inf),
        ("move_um", -1.0),
        ("move_um", math.nan),
        ("move_um", math.inf),
    ],
)
def test_plan_refuses_a_request_duration_or_move_that_is_negative_or_not_finite(
    name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=name):
        plan(fixed(), ["BF"], interval_s=300, **{name: value})


@pytest.mark.parametrize(
    ("interval_s", "duration_h"), [(1e-320, 10.0), (1e306, 10.0), (300.0, 1e308)]
)
def test_plan_refuses_a_count_that_overflows(
    interval_s: float, duration_h: float
) -> None:
    """``inf // x`` is ``nan`` and ``int(nan)`` raised a ``ValueError`` with no key named."""
    with pytest.raises(ValueError, match="too large"):
        plan(fixed(), ["BF"], interval_s=interval_s, duration_h=duration_h)


def test_a_position_that_costs_nothing_fits_no_positions() -> None:
    """The port's rule: a zero cost is "unknown", not "infinitely many"."""
    t = fixed(snap_ms=0.0, refocus_ms=None, channel_ms={}, move_ms_per_mm=0.0)
    budget = plan(t, [], interval_s=300, move_um=0.0)
    assert budget.per_position_ms == 0.0
    assert budget.positions == 0


def test_a_single_point_fits_a_rate_and_no_overhead() -> None:
    """The port's one-point branch: a rate through the origin, never a division by 0."""
    assert fit_line([(1000.0, 150.0)]) == (0.0, 150.0)
    assert fit_line([(0.0, 150.0)]) == (0.0, 0.0)
    assert fit_line([]) == (0.0, 0.0)


def test_two_points_at_one_distance_fit_no_rate() -> None:
    """The port's ``denom == 0`` branch: no slope, the mean as the overhead."""
    assert fit_line([(500.0, 100.0), (500.0, 200.0)]) == (150.0, 0.0)


@given(
    overhead=st.floats(0.0, 5000.0),
    rate=st.floats(0.0, 500.0),
    short=st.floats(10.0, 1000.0),
    extra=st.floats(100.0, 50000.0),
)
def test_a_line_through_two_exact_points_is_recovered(
    overhead: float, rate: float, short: float, extra: float
) -> None:
    long = short + extra
    points = [
        (short, overhead + rate * short / 1000.0),
        (long, overhead + rate * long / 1000.0),
    ]
    got_overhead, got_rate = fit_line(points)
    assert got_overhead == pytest.approx(overhead, abs=1e-6)
    assert got_rate == pytest.approx(rate, abs=1e-6)


def test_a_budget_is_frozen() -> None:
    budget = plan(fixed(), ["BF"], interval_s=300)
    assert isinstance(budget, Budget)
    with pytest.raises(dataclasses.FrozenInstanceError):
        budget.positions = 1  # type: ignore[misc]


def test_describe_names_a_missing_refocus_and_a_missing_switch() -> None:
    lines = describe_timings(fixed(refocus_ms=None, channel_switch_ms=None))
    assert "refocus      not applicable: no autofocus" in lines
    assert "channel swap not measured" in lines


def test_describe_lists_the_model_the_channels_and_the_snap() -> None:
    lines = describe_timings(fixed())
    assert lines[0] == "stage move   100 ms + 50.0 ms/mm"
    assert "refocus      400 ms" in lines
    assert "channel swap 100 ms" in lines
    assert "  GFP        300 ms" in lines
    assert "snap         200 ms" in lines
    assert not any("assumed" in line for line in lines)


def test_a_budget_describes_what_it_assumed() -> None:
    budget = plan(fixed(assumed=("refocus 400 ms is assumed",)), ["BF"], 300)
    assert budget.describe()[-1] == "assumed (not measured): refocus 400 ms is assumed"


def test_the_assumed_constants_are_the_ports() -> None:
    assert ASSUMED_CHANNEL_SWITCH_MS == 100.0
    assert ASSUMED_REFOCUS_MS == 400.0


# --------------------------------------------------------------- the parameters


@pytest.mark.parametrize(
    "distances",
    ["[9000,200]", "[200,200]", "[0,100]", "[-1,5]", "[200,50001]", "[200,1e300]"],
)
def test_distances_must_be_short_then_long(distances: str) -> None:
    """Two equal distances fit no line, and a probe is an absolute move (FM-128)."""
    with pytest.raises(ParamsError, match="distances_um"):
        parse_params(TimingParams, [f"distances_um={distances}"])


def test_distances_that_are_not_numbers_are_a_params_error() -> None:
    with pytest.raises(ParamsError, match="distances_um"):
        parse_params(TimingParams, ["distances_um=[200,NaN]"])
    with pytest.raises(ParamsError, match="distances_um"):
        parse_params(TimingParams, ["distances_um=[[200],[400]]"])


def test_distances_in_range_are_accepted() -> None:
    parsed = parse_params(TimingParams, ["distances_um=[200,50000]"])
    assert parsed.distances_um == (200.0, 50000.0)
    assert parse_params(TimingParams, []).distances_um == (200.0, 9000.0)


@pytest.mark.parametrize(
    "channels", ['["A","A"]', '[""]', '["A",""]', '[["A"]]', "[1]", '[{"a":1}]']
)
def test_channels_refuse_an_empty_name_a_repeat_and_a_nested_list(
    channels: str,
) -> None:
    """A nested list is a ``ParamsError`` (FM-128), never the ``TypeError`` of a set lookup."""
    with pytest.raises(ParamsError, match="channels"):
        parse_params(TimingParams, [f"channels={channels}"])


def test_channels_are_bounded_and_may_be_empty() -> None:
    assert parse_params(TimingParams, []).channels == ()
    ok = [f"c{n}" for n in range(32)]
    quoted = ",".join(f'"{n}"' for n in ok)
    assert parse_params(TimingParams, [f"channels=[{quoted}]"]).channels == tuple(ok)
    too_many = ",".join(f'"c{n}"' for n in range(33))
    with pytest.raises(ParamsError, match="channels"):
        parse_params(TimingParams, [f"channels=[{too_many}]"])


@pytest.mark.parametrize("repeats", [0, 21, -1])
def test_repeats_outside_one_to_twenty_are_refused(repeats: int) -> None:
    with pytest.raises(ParamsError, match="repeats"):
        parse_params(TimingParams, [f"repeats={repeats}"])


def test_repeats_are_bounded() -> None:
    assert parse_params(TimingParams, ["repeats=1"]).repeats == 1
    assert parse_params(TimingParams, ["repeats=20"]).repeats == 20
