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
from hypothesis import given, settings
from hypothesis import strategies as st

from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.route.algorithm import (
    _distance_to_polygon,
    allowed,
    order_route,
    path_length_um,
)
from smc.plugins.slots import WellGeometry

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

Point = tuple[float, float]

#: 60 rows by 100 columns at 1 µm, upright: a footprint of 100 x 60 µm.
FRAME = (60, 100)  # (height, width): not square, on purpose (FM-73)
UPRIGHT = ((1.0, 0.0), (0.0, -1.0))


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


def _well(
    diameter_um: float,
    *,
    circular: bool = True,
    height_um: float | None = None,
    rotation_deg: float = 0.0,
    centre: Point = (0.0, 0.0),
) -> WellGeometry:
    return WellGeometry(
        well="A1",
        center_um=centre,
        size_um=(diameter_um, diameter_um if height_um is None else height_um),
        circular=circular,
        rotation_deg=rotation_deg,
    )


def _plate_to_stage(well: WellGeometry, u: float, v: float) -> Point:
    """A point of the plate's axes in stage axes, written here: ``centre + R(theta) (u, v)``."""
    angle = math.radians(well.rotation_deg)
    return (
        well.center_um[0] + math.cos(angle) * u - math.sin(angle) * v,
        well.center_um[1] + math.sin(angle) * u + math.cos(angle) * v,
    )


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


# ------------------------------------------------------------------- what is allowed


def _inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """The oracle: all four cross products of one sign, whichever way the corners turn."""
    crosses = [
        (polygon[(k + 1) % 4][0] - polygon[k][0]) * (point[1] - polygon[k][1])
        - (polygon[(k + 1) % 4][1] - polygon[k][1]) * (point[0] - polygon[k][0])
        for k in range(4)
    ]
    return all(c >= 0.0 for c in crosses) or all(c <= 0.0 for c in crosses)


def _edge_samples(polygon: Sequence[Point], per_edge: int = 200) -> list[Point]:
    """The oracle's boundary: the corners and ``per_edge`` points along each edge."""
    samples: list[Point] = []
    for k in range(4):
        (ax, ay), (bx, by) = polygon[k], polygon[(k + 1) % 4]
        for step in range(per_edge + 1):
            t = step / per_edge
            samples.append((ax + t * (bx - ax), ay + t * (by - ay)))
    return samples


def _field_allowed(
    well: WellGeometry,
    x_um: float,
    y_um: float,
    *,
    exclude: float = 0.0,
    margin: float = 0.0,
    matrix: Matrix2 = UPRIGHT,
    shape: tuple[int, int] = FRAME,
) -> bool:
    return allowed(
        x_um,
        y_um,
        well,
        matrix=matrix,
        image_shape_px=shape,
        exclude_centre_um=exclude,
        wall_margin_um=margin,
    )


def test_allowed_measures_the_distance_to_the_footprint_not_to_its_corners() -> None:
    """A footprint centred at (0, 40) has its lower edge 10 µm above the centre and its corners 50 µm away.

    20 and 5 differ from the 10 µm distance, so measuring to the corners (or to
    the footprint's centre) is told from measuring to the edge.
    """
    well = _well(10_000.0)

    assert not _field_allowed(well, 0.0, 40.0, exclude=20.0)
    assert _field_allowed(well, 0.0, 40.0, exclude=5.0)


@pytest.mark.parametrize("exclude", [1e-9, 0.001, 20.0, 1000.0])
def test_a_field_over_the_centre_is_refused_for_any_exclusion(exclude: float) -> None:
    well = _well(10_000.0)

    assert not _field_allowed(well, 0.0, 0.0, exclude=exclude)
    assert not _field_allowed(well, 30.0, 10.0, exclude=exclude)
    assert _field_allowed(well, 0.0, 0.0, exclude=0.0)


def test_the_exclusion_is_the_distance_to_the_edge_of_a_turned_and_mirrored_footprint() -> (
    None
):
    """Whichever way the corners turn, the distance is to the polygon (a mirror reverses them)."""
    well = _well(10_000.0)
    for mirrored in (False, True):
        matrix = _matrix(0.8, 25.0, mirrored)
        corners = footprint(300.0, 200.0, matrix=matrix, image_shape_px=FRAME)
        nearest = min(math.hypot(x, y) for x, y in _edge_samples(corners, 2000))

        assert not _inside_polygon((0.0, 0.0), corners)
        assert _field_allowed(well, 300.0, 200.0, exclude=nearest - 0.5, matrix=matrix)
        assert not _field_allowed(
            well, 300.0, 200.0, exclude=nearest + 0.5, matrix=matrix
        )


def test_distance_to_polygon_is_zero_inside_and_to_the_nearest_edge_outside() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]
    for corners in (square, square[::-1]):
        assert _distance_to_polygon((5.0, 3.0), corners) == 0.0
        assert _distance_to_polygon((10.0, 3.0), corners) == 0.0  # on the edge
        assert _distance_to_polygon((13.0, 3.0), corners) == pytest.approx(3.0)
        assert _distance_to_polygon((5.0, -4.0), corners) == pytest.approx(4.0)
        # Beside a corner: the distance is to the corner, not to the edge's line.
        assert _distance_to_polygon((13.0, 10.0), corners) == pytest.approx(5.0)


def test_allowed_needs_every_corner_inside_the_shrunk_well() -> None:
    """The centre of this field is inside the well and one corner is not."""
    round_well = _well(200.0)

    assert round_well.contains(60.0, 0.0)
    assert not _field_allowed(round_well, 60.0, 0.0)  # its corner at x = 110
    assert _field_allowed(round_well, 40.0, 0.0)  # its corner at x = 90, y = +-30

    # Square well of 2 000 x 2 000: the field's far corner is at x = 900, 100 µm inside.
    square = _well(2000.0, circular=False)
    assert _field_allowed(square, 850.0, 0.0, margin=50.0)
    assert not _field_allowed(square, 850.0, 0.0, margin=150.0)
    # The same on the height: the field's top edge is at y = 970, 30 µm inside.
    assert _field_allowed(square, 0.0, 940.0, margin=20.0)
    assert not _field_allowed(square, 0.0, 940.0, margin=40.0)


def test_allowed_on_a_square_well_follows_the_plates_turn() -> None:
    """A well of 2 000 x 1 000 turned by 30°: one field is inside it and outside the axis-aligned well, and one the reverse.

    The well is not square (FM-73) and the camera's frame is not either.
    """
    turned = _well(2000.0, circular=False, height_um=1000.0, rotation_deg=30.0)
    axis_aligned = _well(2000.0, circular=False, height_um=1000.0, rotation_deg=0.0)
    small = (6, 10)

    in_the_turned_corner = _plate_to_stage(turned, 950.0, 450.0)
    in_the_axis_aligned_corner = (940.0, 460.0)

    assert _field_allowed(turned, *in_the_turned_corner, shape=small)
    assert not _field_allowed(axis_aligned, *in_the_turned_corner, shape=small)
    assert _field_allowed(axis_aligned, *in_the_axis_aligned_corner, shape=small)
    assert not _field_allowed(turned, *in_the_axis_aligned_corner, shape=small)


def test_allowed_refuses_a_margin_that_is_not_a_distance() -> None:
    with pytest.raises(ValueError, match="margin_um must be finite and not negative"):
        _field_allowed(_well(1000.0), 0.0, 0.0, margin=-1.0)


@st.composite
def _setups(draw: st.DrawFn) -> dict[str, object]:
    """A well (round or square, turned, not square), a camera (turned, mirrored, not square) and a field."""
    size = draw(st.floats(min_value=300.0, max_value=3000.0))
    circular = draw(st.booleans())
    height = size if circular else draw(st.floats(min_value=300.0, max_value=3000.0))
    well = _well(
        size,
        circular=circular,
        height_um=height,
        rotation_deg=draw(st.floats(min_value=-30.0, max_value=30.0)),
        centre=(
            draw(st.floats(min_value=-5000.0, max_value=5000.0)),
            draw(st.floats(min_value=-5000.0, max_value=5000.0)),
        ),
    )
    shape = (
        draw(st.integers(min_value=20, max_value=120)),
        draw(st.integers(min_value=20, max_value=120)),
    )
    matrix = _matrix(
        draw(st.floats(min_value=0.5, max_value=2.0)),
        draw(st.floats(min_value=0.0, max_value=360.0)),
        draw(st.booleans()),
    )
    reach = max(size, height) / 2.0 + 100.0
    return {
        "well": well,
        "shape": shape,
        "matrix": matrix,
        "x": well.center_um[0] + draw(st.floats(min_value=-reach, max_value=reach)),
        "y": well.center_um[1] + draw(st.floats(min_value=-reach, max_value=reach)),
        "margin": draw(st.floats(min_value=0.0, max_value=200.0)),
        "exclude": draw(st.one_of(st.just(0.0), st.floats(0.0, 300.0))),
    }


@settings(max_examples=300, deadline=None)
@given(_setups())
def test_allowed_agrees_with_a_sampled_footprint(setup: dict[str, object]) -> None:
    """What is allowed keeps to the well and away from the centre, and what is clearly so is allowed.

    The oracle samples the footprint's boundary (the corners and 200 points per
    edge, from ``footprint()`` corners) and asks ``contains`` of each, and
    measures the distance to the centre on the samples. Sampling can only
    overstate the distance, so ``allowed`` must imply the oracle exactly and the
    oracle with a 1 µm slack must imply ``allowed``.
    """
    well = setup["well"]
    assert isinstance(well, WellGeometry)
    x, y = float(setup["x"]), float(setup["y"])  # type: ignore[arg-type]
    margin, exclude = float(setup["margin"]), float(setup["exclude"])  # type: ignore[arg-type]
    matrix: Matrix2 = setup["matrix"]  # type: ignore[assignment]
    shape: tuple[int, int] = setup["shape"]  # type: ignore[assignment]
    corners = footprint(x, y, matrix=matrix, image_shape_px=shape)
    samples = _edge_samples(corners)
    centre = well.center_um
    nearest = (
        0.0
        if _inside_polygon(centre, corners)
        else min(math.dist(centre, s) for s in samples)
    )

    verdict = _field_allowed(
        well, x, y, exclude=exclude, margin=margin, matrix=matrix, shape=shape
    )

    if verdict:
        assert all(
            well.contains(sx, sy, margin_um=max(0.0, margin - 1e-6))
            for sx, sy in samples
        )
        assert exclude <= 0.0 or nearest >= exclude - 1e-6
    clearly_inside = all(
        well.contains(sx, sy, margin_um=margin + 1.0) for sx, sy in samples
    )
    if clearly_inside and (exclude <= 0.0 or nearest >= exclude + 1.0):
        assert verdict
