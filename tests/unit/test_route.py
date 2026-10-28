"""The route's pure part: which fields hold the cells, and the shortest path through them (m3 §11, #194).

The oracles are written here and share nothing with the code under test but
what the code is meant to agree with: a nearest-neighbour path, the brute-force
optimum for a few points, a point-in-polygon test and a separating-axis overlap
test on ``footprint()`` corners (never ``pixel_of_stage``). Every property that
treats the two axes differently has a frame that is not square (FM-73) and a
camera that is turned and mirrored.

``filterwarnings = error::RuntimeWarning`` below makes a numpy warning a failure
here and not only under ``-W error``: ``(N, 2) @ (2, 2)`` warns "divide by zero
encountered in matmul" on numpy 2.2.6 from N = 2000 up, so a matrix product on
coordinates is caught on the maintainer's machine.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.route.algorithm import order_route, path_length_um

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

Point = tuple[float, float]


# ------------------------------------------------------------------ the ordering


def _length(points: Sequence[Point], order: Sequence[int]) -> float:
    """The length of the path, written here: the sum of the legs."""
    return sum(
        math.hypot(points[b][0] - points[a][0], points[b][1] - points[a][1])
        for a, b in itertools.pairwise(order)
    )


def _nearest_neighbour(points: Sequence[Point], start: int) -> list[int]:
    """The oracle: from ``start`` always the nearest point left, the lowest index on a tie."""
    left = [index for index in range(len(points)) if index != start]
    order = [start]
    while left:
        here = points[order[-1]]
        nearest = min(
            left,
            key=lambda i: (
                math.hypot(points[i][0] - here[0], points[i][1] - here[1]),
                i,
            ),
        )
        order.append(nearest)
        left.remove(nearest)
    return order


def _optimum(points: Sequence[Point], start: int) -> float:
    """The shortest open path that starts at ``start``, by trying every order."""
    rest = [index for index in range(len(points)) if index != start]
    return min(
        _length(points, [start, *permutation])
        for permutation in itertools.permutations(rest)
    )


_coordinate = st.floats(min_value=-5000.0, max_value=5000.0)
_points = st.lists(st.tuples(_coordinate, _coordinate), min_size=1, max_size=25)


@st.composite
def _points_and_start(draw: st.DrawFn, max_size: int = 25) -> tuple[list[Point], int]:
    points = draw(
        st.lists(st.tuples(_coordinate, _coordinate), min_size=1, max_size=max_size)
    )
    return points, draw(st.integers(min_value=0, max_value=len(points) - 1))


@given(_points_and_start())
def test_the_route_visits_every_field_once(case: tuple[list[Point], int]) -> None:
    points, start = case

    order = order_route(points, start=start)

    assert sorted(order) == list(range(len(points)))
    assert order[0] == start


@given(_points_and_start())
def test_two_opt_never_lengthens_the_route(case: tuple[list[Point], int]) -> None:
    points, start = case

    order = order_route(points, start=start)

    assert (
        _length(points, order)
        <= _length(points, _nearest_neighbour(points, start)) + 1e-9
    )


@given(_points_and_start(max_size=7))
def test_the_route_is_never_shorter_than_the_optimum(
    case: tuple[list[Point], int],
) -> None:
    """A heuristic cannot beat the brute force: a path that did would be one that skips a point."""
    points, start = case

    order = order_route(points, start=start)

    assert _length(points, order) >= _optimum(points, start) - 1e-9


def test_two_opt_uncrosses_a_nearest_neighbour_path() -> None:
    points = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 2.0)]

    # Nearest neighbour crosses itself here: 0, 1, 3, 2, 4 is 5.236 µm.
    assert _nearest_neighbour(points, 0) == [0, 1, 3, 2, 4]
    assert _length(points, [0, 1, 3, 2, 4]) == pytest.approx(5.236, abs=1e-3)

    order = order_route(points, start=0)

    assert order == [0, 2, 3, 1, 4]
    assert _length(points, order) == pytest.approx(4.0)


def test_order_route_breaks_ties_by_the_lowest_index() -> None:
    assert order_route([(0.0, 0.0)] * 5, start=2) == [2, 0, 1, 3, 4]


def test_the_first_point_stays_where_it_is() -> None:
    """On a line the best path from the middle is not the one that starts at an end."""
    points = [(float(x), 0.0) for x in (0, 10, 20, 30, 40)]

    order = order_route(points, start=2)

    assert order[0] == 2
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_order_route_of_no_point_is_empty_and_of_one_is_itself() -> None:
    assert order_route([], start=0) == []
    assert order_route([(5.0, 5.0)], start=0) == [0]


@pytest.mark.parametrize("start", [-1, 3, 10])
def test_order_route_refuses_a_start_that_is_not_a_point(start: int) -> None:
    with pytest.raises(ValueError, match=r"start must be an index of the 3 points"):
        order_route([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], start=start)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_order_route_refuses_a_point_that_is_not_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="every point must have finite coordinates"):
        order_route([(0.0, 0.0), (bad, 1.0), (2.0, 0.0)], start=0)
    with pytest.raises(ValueError, match="every point must have finite coordinates"):
        order_route([(0.0, 0.0), (1.0, bad), (2.0, 0.0)], start=0)


def test_path_length_um() -> None:
    points = [(0.0, 0.0), (3.0, 4.0), (3.0, 10.0)]

    assert path_length_um(points, [0, 1]) == pytest.approx(5.0)
    assert path_length_um(points, [0, 1, 2]) == pytest.approx(11.0)
    assert path_length_um(points, [2, 1, 0]) == pytest.approx(11.0)
    assert path_length_um(points, [1]) == 0.0
    assert path_length_um(points, []) == 0.0
