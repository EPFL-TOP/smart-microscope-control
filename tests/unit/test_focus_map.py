"""The 10x lock map and the anchor rules (m3 §21.12.3): pure, on points placed by hand."""

from __future__ import annotations

import math

import pytest

from smc.plugins.cell_focus.focus_map import (
    ANCHOR_PEAKS,
    ANCHOR_TRIES,
    MAP_NEIGHBOURS,
    MAP_SNAP_UM,
    FocusMap,
    MapPoint,
    anchor_of,
    nearest_first,
)

#: The corners of a 10 x 20 µm rectangle, not square (FM-73): the two at y = 0
#: read Z 0, the two at y = 20 read Z 4.
_CORNERS = (
    MapPoint("B2", 0.0, 0.0, 0.0),
    MapPoint("B2", 10.0, 0.0, 0.0),
    MapPoint("B2", 0.0, 20.0, 4.0),
    MapPoint("B2", 10.0, 20.0, 4.0),
)
#: Far from the rectangle, of the same well: a fifth neighbour, never read.
_FAR = MapPoint("B2", 500.0, 500.0, 100.0)


def test_the_constants_are_the_designs() -> None:
    # m3 §21.12.3: measured or chosen there, not tuned here.
    assert (MAP_NEIGHBOURS, MAP_SNAP_UM, ANCHOR_TRIES, ANCHOR_PEAKS) == (4, 1.0, 8, 3)


def test_the_centre_of_four_frames_is_their_mean_and_a_far_fifth_is_not_read() -> None:
    assert FocusMap(_CORNERS).z_at("B2", 5.0, 10.0) == pytest.approx(2.0, abs=1e-12)
    # The fifth comes first and last in the input: never one of the 4 nearest.
    assert FocusMap((_FAR, *_CORNERS)).z_at("B2", 5.0, 10.0) == pytest.approx(
        2.0, abs=1e-12
    )
    assert FocusMap((*_CORNERS, _FAR)).z_at("B2", 5.0, 10.0) == pytest.approx(
        2.0, abs=1e-12
    )


def test_a_point_on_a_frame_gives_its_own_z() -> None:
    focus_map = FocusMap(_CORNERS)
    assert focus_map.z_at("B2", 0.0, 20.0) == 4.0
    # Within MAP_SNAP_UM, no division by about 0: the frame's own Z, exactly.
    assert focus_map.z_at("B2", 0.6, 20.7) == 4.0
    assert focus_map.z_at("B2", 1e-12, 0.0) == 0.0
    # Just beyond it, the weights decide (and give something other than 4).
    assert focus_map.z_at("B2", 1.1, 20.0) != 4.0


def test_the_weights_are_one_over_the_distance() -> None:
    # At (5, 2): the near pair is sqrt(29) away, the far pair sqrt(349); by hand,
    # 2 * 4 / sqrt(349) / (2 / sqrt(29) + 2 / sqrt(349)) = 0.8950. The plain mean
    # of the four is 2.0, and 1/d^2 gives 0.3069.
    near, far = 1.0 / math.sqrt(29.0), 1.0 / math.sqrt(349.0)
    by_hand = (2.0 * 4.0 * far) / (2.0 * near + 2.0 * far)
    assert by_hand == pytest.approx(0.8950, abs=1e-4)
    assert FocusMap(_CORNERS).z_at("B2", 5.0, 2.0) == pytest.approx(by_hand, rel=1e-12)


def test_another_wells_frames_are_never_read() -> None:
    # C2's frames lie nearer the point than B2's, and read Z 100: a wall lies
    # between, so B2's answer does not move.
    other = tuple(MapPoint("C2", p.x_um + 4.0, p.y_um, 100.0) for p in _CORNERS)
    focus_map = FocusMap((*other, *_CORNERS))
    assert focus_map.z_at("B2", 5.0, 10.0) == pytest.approx(2.0, abs=1e-12)
    assert focus_map.z_at("C2", 9.0, 10.0) == pytest.approx(100.0, abs=1e-12)
    assert (focus_map.count("B2"), focus_map.count("C2"), focus_map.count("D4")) == (
        4,
        4,
        0,
    )


def test_a_well_with_three_frames_is_refused_by_name() -> None:
    focus_map = FocusMap(_CORNERS[:3])
    with pytest.raises(
        ValueError,
        match=r"\Athe focus map holds 3 locked frames in B2, fewer than 4\Z",
    ):
        focus_map.z_at("B2", 5.0, 10.0)
    with pytest.raises(ValueError, match=r"holds 0 locked frames in C3, fewer than 4"):
        FocusMap(_CORNERS).z_at("C3", 5.0, 10.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_point_or_an_argument_that_is_not_finite_is_refused(bad: float) -> None:
    for point in (
        MapPoint("B2", bad, 0.0, 0.0),
        MapPoint("B2", 0.0, bad, 0.0),
        MapPoint("B2", 0.0, 0.0, bad),
    ):
        with pytest.raises(ValueError, match=r"of well B2 is not finite"):
            FocusMap((*_CORNERS, point))
    focus_map = FocusMap(_CORNERS)
    with pytest.raises(ValueError, match=r"not finite"):
        focus_map.z_at("B2", bad, 0.0)
    with pytest.raises(ValueError, match=r"not finite"):
        focus_map.z_at("B2", 0.0, bad)
    with pytest.raises(ValueError, match=r"not finite"):
        focus_map.nearest(bad, 0.0)


def test_nearest_names_the_well_and_the_distance_and_takes_the_first_on_a_tie() -> None:
    other = MapPoint("C2", 30.0, 0.0, 9.0)
    focus_map = FocusMap((*_CORNERS, other))
    point, distance = focus_map.nearest(27.0, 4.0)
    assert point == other
    assert distance == pytest.approx(5.0, abs=1e-12)
    # (5, 0) is 5 µm from both bottom corners: the first in the input wins.
    assert focus_map.nearest(5.0, 0.0) == (_CORNERS[0], 5.0)
    swapped = FocusMap((_CORNERS[1], _CORNERS[0]))
    assert swapped.nearest(5.0, 0.0) == (_CORNERS[1], 5.0)


def test_an_empty_map_is_allowed_and_refuses_to_answer() -> None:
    empty = FocusMap(())
    assert empty.count("B2") == 0
    with pytest.raises(ValueError, match=r"the focus map holds no locked frame"):
        empty.nearest(0.0, 0.0)
    with pytest.raises(ValueError, match=r"holds 0 locked frames in B2"):
        empty.z_at("B2", 0.0, 0.0)


def test_the_anchor_is_the_median_and_none_without_a_peak() -> None:
    assert anchor_of([1.0, 5.0, 2.0]) == 2.0
    assert anchor_of([-81.5, -80.5]) == -81.0
    assert anchor_of([]) is None
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match=r"not finite"):
            anchor_of([1.0, bad, 2.0])


def test_nearest_first_orders_by_distance_and_takes_the_lower_index_on_a_tie() -> None:
    points = [(10.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -1.0), (7.0, 7.0)]
    assert nearest_first(points, (0.0, 0.0), 3) == (3, 1, 2)
    # All of them when n is larger; the tie between 1 and 2 goes to 1.
    assert nearest_first(points, (0.0, 0.0), 10) == (3, 1, 2, 4, 0)
    # The centre is not the origin, and x and y are not swapped (FM-73).
    assert nearest_first(points, (10.0, 1.0), 2) == (0, 4)
    assert nearest_first([], (0.0, 0.0), 8) == ()
    with pytest.raises(ValueError, match=r"n must be at least 1"):
        nearest_first(points, (0.0, 0.0), 0)
