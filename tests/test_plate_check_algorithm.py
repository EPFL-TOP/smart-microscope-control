"""The pure part of ``plate-check``: the pair of wells it re-measures, and the turn and slide it reads (m3 §27.3).

The numbers of session 5's move are the run documents' (#542, #550, §27.1): B2 and
B5 as ``plate-walls`` registered them at 11:27 UTC and as the recheck measured them
after the plate was lifted and put back. The expected turn and slide are written
out from those numbers, never read from the code (FM-203).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.plate_check.algorithm import (
    DEFAULT_MAX_SHIFT_UM,
    DEFAULT_SEARCH_UM,
    farthest_pair,
    rigid_motion,
)

#: Session 5 (#550, F15): registered, then re-measured after the plate was lifted.
B2_REGISTERED = (29601.8, -9936.3)
B2_MEASURED = (29587.35, -9964.71)
B5_REGISTERED = (-28924.5, -9968.3)
B5_MEASURED = (-28941.19, -9939.30)


def test_the_defaults_are_the_designs() -> None:
    # §27.2 D3 and §27.4: chosen, revisited at the stand.
    assert DEFAULT_MAX_SHIFT_UM == 25.0
    assert DEFAULT_SEARCH_UM == 1000.0


def test_session_5s_move_reads_as_a_turn_of_minus_0_056_degrees() -> None:
    turn_deg, slide = rigid_motion(
        [B2_REGISTERED, B5_REGISTERED], [B2_MEASURED, B5_MEASURED]
    )

    assert turn_deg == pytest.approx(-0.0562, abs=0.0005)
    # The mean of the measured minus the mean of the registered:
    # ((29587.35 - 28941.19) - (29601.8 - 28924.5)) / 2 = -15.57 and
    # ((-9964.71 - 9939.30) - (-9936.3 - 9968.3)) / 2 = +0.295.
    assert slide == pytest.approx((-15.57, 0.295), abs=0.01)


def test_a_pure_slide_reads_no_turn() -> None:
    expected = [(1000.0, 2000.0), (-50_000.0, 7000.0), (20_000.0, -30_000.0)]
    measured = [(x + 50.0, y - 12.5) for x, y in expected]

    turn_deg, slide = rigid_motion(expected, measured)

    assert turn_deg == pytest.approx(0.0, abs=1e-9)
    assert slide == pytest.approx((50.0, -12.5), abs=1e-9)


def test_a_turn_about_the_origin_reads_its_angle_and_the_slide_of_the_mean() -> None:
    # Two wells either side of the origin, turned by 0.06° about it: the mean stays at
    # the origin, so the slide is nothing, and the turn is the angle.
    angle = math.radians(0.06)
    expected = [(-49_500.0, 31_500.0), (49_500.0, -31_500.0)]
    measured = [
        (
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle),
        )
        for x, y in expected
    ]

    turn_deg, slide = rigid_motion(expected, measured)

    assert turn_deg == pytest.approx(0.06, abs=1e-9)
    assert slide == pytest.approx((0.0, 0.0), abs=1e-6)


@given(
    turn=st.floats(min_value=-5.0, max_value=5.0),
    sx=st.floats(min_value=-1000.0, max_value=1000.0),
    sy=st.floats(min_value=-1000.0, max_value=1000.0),
)
def test_a_rigid_motion_is_recovered_whatever_it_is(
    turn: float, sx: float, sy: float
) -> None:
    expected = [(-49_500.0, 31_500.0), (49_500.0, -31_500.0), (10_000.0, 9_000.0)]
    angle = math.radians(turn)
    cx = sum(x for x, _ in expected) / 3
    cy = sum(y for _, y in expected) / 3
    measured = [
        (
            cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + sx,
            cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + sy,
        )
        for x, y in expected
    ]

    turn_deg, slide = rigid_motion(expected, measured)

    assert turn_deg == pytest.approx(turn, abs=1e-6)
    assert slide == pytest.approx((sx, sy), abs=1e-6)


def test_rigid_motion_refuses_fewer_than_two_points_and_a_nan() -> None:
    with pytest.raises(ValueError, match="two"):
        rigid_motion([(0.0, 0.0)], [(1.0, 1.0)])
    with pytest.raises(ValueError, match="two"):
        rigid_motion([], [])
    with pytest.raises(ValueError, match="as many"):
        rigid_motion([(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0)])
    with pytest.raises(ValueError, match="finite"):
        rigid_motion([(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (math.nan, 0.0)])
    with pytest.raises(ValueError, match="finite"):
        rigid_motion([(math.inf, 0.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)])


def test_the_pair_farthest_apart_is_chosen_and_a_tie_keeps_the_registrations_order() -> (
    None
):
    # Session 5's registration: B2 and B5 are 58.5 mm apart, C3 lies between them.
    session_5 = [
        ("C3", (10056.7, 9544.1)),
        ("B2", B2_REGISTERED),
        ("B5", B5_REGISTERED),
    ]
    assert farthest_pair(session_5) == ("B2", "B5")

    # A square: both diagonals are the same length. The first pair met in the order
    # (i, j), i < j, is (A1, H12): A1 is first, and H12 is the first well whose
    # distance from A1 is the largest.
    square = [
        ("A1", (0.0, 0.0)),
        ("A12", (1000.0, 0.0)),
        ("H12", (1000.0, 1000.0)),
        ("H1", (0.0, 1000.0)),
    ]
    assert farthest_pair(square) == ("A1", "H12")

    # Two wells are the pair, in the order given.
    assert farthest_pair([("H12", (5.0, 5.0)), ("A1", (0.0, 0.0))]) == ("H12", "A1")


def test_farthest_pair_refuses_fewer_than_two_wells() -> None:
    with pytest.raises(ValueError, match="two"):
        farthest_pair([("A1", (0.0, 0.0))])
    with pytest.raises(ValueError, match="two"):
        farthest_pair([])
