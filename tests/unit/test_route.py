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
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from smc.plugins.geometry import Matrix2, footprint, stage_of_pixel
from smc.plugins.route import algorithm
from smc.plugins.route.algorithm import (
    LatticeTooLarge,
    TooMuchWork,
    _candidate_counts,
    _distance_to_polygon,
    _inverse_of,
    _lattice,
    _pixel_offsets,
    allowed,
    choose_fields,
    fov_of,
    order_route,
    path_length_um,
)
from smc.plugins.slots import Cell, WellGeometry

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


# -------------------------------------------------------------- choosing the fields

#: A round well of 2 000 µm at the origin, the 60 x 100 upright frame at 1 µm and a
#: lattice step of 15 µm: the scenarios below were measured on this. Keep the wells
#: of the unit tests this small: a 10 000 µm well at a 15 µm step is 444 889
#: lattice points, which ``LatticeTooLarge`` refuses.
WELL = _well(2000.0)
STEP = 15.0


def _cell(
    x: float, y: float, label: str = "single", well: str = "A1", frame: int = 0
) -> Cell:
    return Cell(
        well=well,
        x_um=x,
        y_um=y,
        label=label,
        score=0.9,
        frame=frame,
        box_px=(0.0, 0.0, 1.0, 1.0),
    )


def _choose(
    cells: Sequence[Cell],
    well: WellGeometry = WELL,
    **overrides: object,
) -> list[tuple[float, float, int]]:
    """``choose_fields`` with the scenarios' defaults; each field as ``(x, y, cells held)``."""
    arguments: dict[str, object] = {
        "labels": ["single"],
        "matrix": UPRIGHT,
        "image_shape_px": FRAME,
        "n": 10,
        "min_cells": 1,
        "exclude_centre_um": 0.0,
        "wall_margin_um": 0.0,
        "step_um": STEP,
    }
    arguments.update(overrides)
    chosen = choose_fields(cells, well, **arguments)  # type: ignore[arg-type]
    return [(f.x_um, f.y_um, sum(f.counts.values())) for f in chosen]


def test_fields_that_touch_are_both_chosen() -> None:
    """Only fields at y = 30 and y = 90 hold a pair whole; they touch at y = 60, which is not an overlap.

    The leftmost x that holds x = 0 is -45: the lowest-numbered candidate.
    """
    cells = [_cell(0, 0), _cell(0, 59.9), _cell(0, 60.1), _cell(0, 120)]

    assert _choose(cells, n=2) == [(-45.0, 90.0, 2), (-45.0, 30.0, 2)]


DENSE = (
    [_cell(-500 + k, 500) for k in range(5)]
    + [_cell(500 + k, 0) for k in range(3)]
    + [_cell(0, -700)]
)


def test_the_greedy_takes_the_densest_field_first() -> None:
    assert _choose(DENSE) == [(-540.0, 525.0, 5), (465.0, 30.0, 3), (-45.0, -675.0, 1)]
    assert _choose(DENSE, n=2) == [(-540.0, 525.0, 5), (465.0, 30.0, 3)]
    assert _choose(DENSE, min_cells=2) == [(-540.0, 525.0, 5), (465.0, 30.0, 3)]
    assert _choose(DENSE, min_cells=6) == []


def test_a_cell_on_the_edge_of_two_touching_fields_never_makes_an_empty_field() -> None:
    """The three cells at y = 0 are on the lower edge of the field at y = 30 and the upper edge of the one at y = -30.

    The first holds them, so the second holds none that is new: without the
    refresh of a count that went down, it is chosen anyway with 0 cells (a
    fourth field at (465, -30), measured).
    """
    chosen = choose_fields(
        DENSE,
        WELL,
        labels=["single"],
        matrix=UPRIGHT,
        image_shape_px=FRAME,
        n=10,
        min_cells=1,
        exclude_centre_um=0.0,
        wall_margin_um=0.0,
        step_um=STEP,
    )

    assert len(chosen) == 3
    assert all(sum(f.counts.values()) >= 1 for f in chosen)
    assert [f.counts["single"] for f in chosen] == [5, 3, 1]


def test_ties_go_to_the_lowest_numbered_candidate() -> None:
    upper_row_first = [_cell(0, 600), _cell(1, 600), _cell(0, -600), _cell(1, -600)]
    left_one_first = [_cell(-600, 0), _cell(-599, 0), _cell(600, 0), _cell(601, 0)]

    assert _choose(upper_row_first, n=2) == [(-45.0, 630.0, 2), (-45.0, -570.0, 2)]
    assert _choose(left_one_first, n=2) == [(-645.0, 30.0, 2), (555.0, 30.0, 2)]


def test_candidates_are_numbered_row_by_row_from_plus_y_then_from_minus_x() -> None:
    """Right-top against left-bottom: row by row takes the top one, column by column would take the left one."""
    cells = [_cell(600, 600), _cell(-600, -600)]

    assert _choose(cells, n=2) == [(555.0, 630.0, 1), (-645.0, -570.0, 1)]


def test_cells_of_other_wells_and_other_labels_are_ignored() -> None:
    cells = [
        _cell(0, 0),
        _cell(0, 0, well="A2"),
        _cell(0, 0, label="doublet"),
        _cell(500, 500, well="A2"),
        _cell(-500, -500, label="debris"),
    ]

    chosen = choose_fields(
        cells,
        WELL,
        labels=["single"],
        matrix=UPRIGHT,
        image_shape_px=FRAME,
        n=10,
        min_cells=1,
        exclude_centre_um=0.0,
        wall_margin_um=0.0,
        step_um=STEP,
    )

    assert [dict(f.counts) for f in chosen] == [{"single": 1}]
    assert _choose(cells, labels=["single", "doublet"]) == [(-45.0, 30.0, 2)]


def test_every_label_asked_for_is_a_key_zeros_included() -> None:
    chosen = choose_fields(
        [_cell(0, 0), _cell(0, 0, label="debris")],
        WELL,
        labels=["single", "doublet", "debris"],
        matrix=UPRIGHT,
        image_shape_px=FRAME,
        n=1,
        min_cells=1,
        exclude_centre_um=0.0,
        wall_margin_um=0.0,
        step_um=STEP,
    )

    assert [dict(f.counts) for f in chosen] == [
        {"single": 1, "doublet": 0, "debris": 1}
    ]


def test_a_field_larger_than_the_well_gives_no_field() -> None:
    small = _well(50.0)

    assert _choose([_cell(0, 0), _cell(5, 5)], small) == []


def test_no_cell_or_no_candidate_gives_no_field() -> None:
    assert _choose([]) == []
    assert _choose([_cell(0, 0)], exclude_centre_um=500.0, wall_margin_um=600.0) == []
    # A field that holds the cell at the well's centre has the centre inside it.
    assert _choose([_cell(0, 0)], exclude_centre_um=40.0) == []


def test_a_matrix_that_cannot_be_inverted_is_refused() -> None:
    with pytest.raises(ValueError, match="the camera matrix cannot be inverted"):
        _choose([_cell(0, 0)], matrix=((1.0, 2.0), (0.5, 1.0)))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n": 0}, "n must be at least 1"),
        ({"min_cells": 0}, "min_cells must be at least 1"),
        ({"labels": []}, "labels must name at least one label"),
        ({"step_um": 0.0}, "step_um must be finite and positive"),
        ({"step_um": -15.0}, "step_um must be finite and positive"),
        ({"step_um": math.nan}, "step_um must be finite and positive"),
        ({"step_um": math.inf}, "step_um must be finite and positive"),
    ],
)
def test_choose_fields_refuses_what_is_not_a_choice(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _choose([_cell(0, 0)], **overrides)


# ----------------------------------------------------------------- the bounds (FM-74)


@pytest.mark.parametrize("step", [1e-3, 5e-324, 1e-300, 1e-9])
def test_a_lattice_beyond_the_bound_is_refused_before_allocating(step: float) -> None:
    """``3200.0 // 5e-324`` is ``inf`` and ``int(inf)`` is an ``OverflowError``: the size is bounded on floats."""
    well = _well(6400.0)

    with pytest.raises(
        LatticeTooLarge,
        match="the candidate lattice of A1 would hold more than 250000 fields at a step of",
    ):
        _choose([_cell(0, 0)], well, step_um=step)
    with pytest.raises(LatticeTooLarge):
        _choose([], well, step_um=step)


def test_a_lattice_that_is_not_a_number_of_points_is_refused_on_floats() -> None:
    with pytest.raises(LatticeTooLarge):
        _lattice(
            _well(6400.0),
            math.nan,
            matrix=UPRIGHT,
            image_shape_px=FRAME,
            exclude_centre_um=0.0,
            wall_margin_um=0.0,
        )
    with pytest.raises(LatticeTooLarge):
        _lattice(
            _well(6400.0),
            0.0,
            matrix=UPRIGHT,
            image_shape_px=FRAME,
            exclude_centre_um=0.0,
            wall_margin_um=0.0,
        )


def test_the_product_of_the_two_axes_is_bounded_too() -> None:
    """Each axis is under 250 000 and their product is not: 600 x 600 points are 361 201."""
    with pytest.raises(LatticeTooLarge):
        _choose([_cell(0, 0)], _well(1200.0), step_um=1.0)


def test_the_largest_legal_lattice_is_still_allowed() -> None:
    """497 x 497 = 247 009 points, under 250 000: 0.43 s on the machine that measured it."""
    well = _well(34_800.0)

    assert _choose([_cell(100, 100)], well, step_um=70.0) == [(70.0, 70.0, 1)]


def test_too_many_tests_are_refused_before_the_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(algorithm, "_MAX_TESTS", 10)

    with pytest.raises(
        TooMuchWork,
        match="1 cells × 63 lattice points per cell would be tested, over 10",  # noqa: RUF001
    ):
        _choose([_cell(0, 0)])


def test_a_tiny_step_is_too_much_work_and_not_an_overflow() -> None:
    index = np.zeros((1, 1), dtype=np.int32)
    centres = np.zeros((1, 2))

    with pytest.raises(TooMuchWork):
        _candidate_counts(
            np.array([[0.0, 0.0]]),
            index,
            centres,
            origin=(0.0, 0.0),
            step_um=5e-324,
            fov_um=(100.0, 60.0),
            inverse=_inverse_of(UPRIGHT, FRAME),
            half_px=(50.0, 30.0),
        )


# ---------------------------------------------------- counting against the oracles


def _footprint_polygon(
    x: float, y: float, matrix: Matrix2, shape: tuple[int, int]
) -> list[Point]:
    return list(footprint(x, y, matrix=matrix, image_shape_px=shape))


def _overlap_has_area(
    a: Sequence[Point], b: Sequence[Point], tol: float = 1e-6
) -> bool:
    """The oracle: two convex polygons share area when no edge normal separates them by ``tol`` or more.

    Touching (a gap of 0) is not an overlap; a separating axis is found on the
    edge normals of either polygon.
    """
    for polygon in (a, b):
        for k in range(len(polygon)):
            (px, py), (qx, qy) = polygon[k], polygon[(k + 1) % len(polygon)]
            nx_, ny_ = qy - py, px - qx
            norm = math.hypot(nx_, ny_)
            nx_, ny_ = nx_ / norm, ny_ / norm
            proj_a = [x * nx_ + y * ny_ for x, y in a]
            proj_b = [x * nx_ + y * ny_ for x, y in b]
            if max(proj_a) <= min(proj_b) + tol or max(proj_b) <= min(proj_a) + tol:
                return False
    return True


def _extent(matrix: Matrix2, shape: tuple[int, int]) -> tuple[float, float]:
    """The oracle's field of view: the extent of ``footprint()`` corners at the origin."""
    corners = _footprint_polygon(0.0, 0.0, matrix, shape)
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


@dataclass(frozen=True)
class World:
    """A well, a camera and the cells on a plate: turned, mirrored and not square (FM-73)."""

    well: WellGeometry
    matrix: Matrix2
    shape: tuple[int, int]
    step_um: float
    cells: tuple[Cell, ...]
    labels: tuple[str, ...]
    n: int
    min_cells: int
    margin_um: float
    exclude_um: float

    def wanted(self) -> list[Cell]:
        return [
            c for c in self.cells if c.well == self.well.well and c.label in self.labels
        ]

    def choose(self, **overrides: object) -> tuple[algorithm.ChosenField, ...]:
        arguments: dict[str, object] = {
            "labels": self.labels,
            "matrix": self.matrix,
            "image_shape_px": self.shape,
            "n": self.n,
            "min_cells": self.min_cells,
            "exclude_centre_um": self.exclude_um,
            "wall_margin_um": self.margin_um,
            "step_um": self.step_um,
        }
        arguments.update(overrides)
        return choose_fields(self.cells, self.well, **arguments)  # type: ignore[arg-type]


@st.composite
def _worlds(draw: st.DrawFn) -> World:
    """Small wells and coarse steps: about 4 000 lattice points at most, so an example stays well under 200 ms."""
    circular = draw(st.booleans())
    width = draw(st.floats(min_value=400.0, max_value=800.0))
    height = width if circular else draw(st.floats(min_value=400.0, max_value=800.0))
    well = _well(
        width,
        circular=circular,
        height_um=height,
        rotation_deg=draw(st.floats(min_value=-30.0, max_value=30.0)),
        centre=(
            draw(st.floats(min_value=-3000.0, max_value=3000.0)),
            draw(st.floats(min_value=-3000.0, max_value=3000.0)),
        ),
    )
    shape = (
        draw(st.integers(min_value=30, max_value=70)),
        draw(st.integers(min_value=30, max_value=70)),
    )
    matrix = _matrix(
        draw(st.floats(min_value=1.0, max_value=1.5)),
        draw(st.floats(min_value=0.0, max_value=360.0)),
        draw(st.booleans()),
    )
    step = draw(st.floats(min_value=0.4, max_value=1.0)) * min(_extent(matrix, shape))

    # Several clusters a little wider than a field, so that more than one field is chosen.
    cells: list[Cell] = []
    for _ in range(draw(st.integers(min_value=3, max_value=6))):
        u = draw(st.floats(min_value=-0.35, max_value=0.35)) * width
        v = draw(st.floats(min_value=-0.35, max_value=0.35)) * height
        x, y = _plate_to_stage(well, u, v)
        for _ in range(draw(st.integers(min_value=2, max_value=8))):
            cells.append(
                _cell(
                    x + draw(st.floats(min_value=-70.0, max_value=70.0)),
                    y + draw(st.floats(min_value=-70.0, max_value=70.0)),
                    label=draw(
                        st.sampled_from(["single", "single", "doublet", "debris"])
                    ),
                )
            )
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        cells.append(_cell(*well.center_um, well="A2"))
    return World(
        well=well,
        matrix=matrix,
        shape=shape,
        step_um=step,
        cells=tuple(cells),
        labels=draw(
            st.sampled_from([("single",), ("single", "doublet"), ("doublet", "debris")])
        ),
        n=draw(st.integers(min_value=1, max_value=6)),
        min_cells=draw(st.sampled_from([1, 1, 1, 2, 3])),
        margin_um=draw(st.sampled_from([0.0, 0.0, 15.0, 60.0])),
        exclude_um=draw(st.sampled_from([0.0, 0.0, 30.0, 120.0])),
    )


@settings(max_examples=100)
@given(_worlds())
def test_no_field_leaves_its_well_or_enters_an_exclusion(world: World) -> None:
    """Samples along the four edges and at the corners of every chosen footprint are inside the shrunk well and outside the disc."""
    for field in world.choose():
        polygon = _footprint_polygon(field.x_um, field.y_um, world.matrix, world.shape)
        samples = _edge_samples(polygon)
        assert all(
            world.well.contains(x, y, margin_um=max(0.0, world.margin_um - 1e-6))
            for x, y in samples
        )
        if world.exclude_um > 0.0:
            centre = world.well.center_um
            assert not _inside_polygon(centre, polygon)
            assert min(math.dist(centre, s) for s in samples) >= world.exclude_um - 1e-6


@settings(max_examples=100)
@given(_worlds())
def test_chosen_fields_never_overlap(world: World) -> None:
    """No two footprints share area by a separating-axis test on ``footprint()`` corners (touching is allowed)."""
    polygons = [
        _footprint_polygon(f.x_um, f.y_um, world.matrix, world.shape)
        for f in world.choose()
    ]

    for a, b in itertools.combinations(polygons, 2):
        assert not _overlap_has_area(a, b, tol=1e-4)


@settings(max_examples=100)
@given(_worlds())
def test_every_cell_is_counted_once_and_counts_match_the_oracle(world: World) -> None:
    """A field's counts are the wanted cells inside its polygon that no earlier field held, for every label asked for."""
    wanted = world.wanted()
    chosen = world.choose()
    held: set[int] = set()

    for field in chosen:
        polygon = _footprint_polygon(field.x_um, field.y_um, world.matrix, world.shape)
        fresh = {
            i
            for i, cell in enumerate(wanted)
            if i not in held and _inside_polygon((cell.x_um, cell.y_um), polygon)
        }
        assert dict(field.counts) == {
            label: sum(1 for i in fresh if wanted[i].label == label)
            for label in world.labels
        }
        assert len(fresh) >= world.min_cells
        held |= fresh

    assert len(chosen) <= world.n
    assert len(held) <= len(wanted)


@settings(max_examples=60)
@given(_worlds())
def test_the_window_counts_equal_the_brute_force_counts(world: World) -> None:
    """Counting per cell over the window equals testing every cell against every candidate; and the first field is the densest."""
    wanted = world.wanted()
    index, centres, origin = _lattice(
        world.well,
        world.step_um,
        matrix=world.matrix,
        image_shape_px=world.shape,
        exclude_centre_um=world.exclude_um,
        wall_margin_um=world.margin_um,
    )
    # A cell nowhere near the well is in the input, to show that it is dropped and counted nowhere.
    xy = np.array(
        [(c.x_um, c.y_um) for c in wanted] + [(1e12, -1e12)], dtype=np.float64
    )
    height, width = world.shape

    counts = _candidate_counts(
        xy,
        index,
        centres,
        origin=origin,
        step_um=world.step_um,
        fov_um=_extent(world.matrix, world.shape),
        inverse=_inverse_of(world.matrix, world.shape),
        half_px=(width / 2.0, height / 2.0),
    )

    reach = math.hypot(*_extent(world.matrix, world.shape))
    brute = []
    for x, y in centres.tolist():
        polygon = _footprint_polygon(x, y, world.matrix, world.shape)
        brute.append(
            sum(
                1
                for cx, cy in xy.tolist()
                if abs(cx - x) <= reach
                and abs(cy - y) <= reach
                and _inside_polygon((cx, cy), polygon)
            )
        )
    assert counts.tolist() == brute
    first = [sum(f.counts.values()) for f in world.choose(n=1, min_cells=1)]
    densest = max(brute, default=0)
    assert first == ([densest] if densest else [])


@given(
    st.floats(min_value=0.5, max_value=3.0),
    st.floats(min_value=0.0, max_value=360.0),
    st.booleans(),
    st.integers(min_value=10, max_value=200),
    st.integers(min_value=10, max_value=200),
    st.floats(min_value=0.0, max_value=199.0),
    st.floats(min_value=0.0, max_value=199.0),
)
def test_pixel_offsets_undo_stage_of_pixel(
    ps: float,
    rotation: float,
    mirrored: bool,
    height: int,
    width: int,
    col: float,
    row: float,
) -> None:
    """The offsets are read through ``pixel_of_stage`` and agree with the forward map, written out and without ``@``."""
    matrix = _matrix(ps, rotation, mirrored)
    x, y = stage_of_pixel(
        0.0, 0.0, col, row, matrix=matrix, image_shape_px=(height, width)
    )

    d_col, d_row = _pixel_offsets(
        np.array([x]), np.array([y]), _inverse_of(matrix, (height, width))
    )

    assert d_col[0] == pytest.approx(col - (width - 1) / 2.0, abs=1e-6)
    assert d_row[0] == pytest.approx(row - (height - 1) / 2.0, abs=1e-6)


def test_the_field_of_view_is_the_extent_of_the_footprint() -> None:
    assert fov_of(UPRIGHT, FRAME) == pytest.approx((100.0, 60.0))
    # Turned by 90°, the frame's width runs along Y: the extents swap (FM-73).
    assert fov_of(_matrix(1.0, 90.0, False), FRAME) == pytest.approx((60.0, 100.0))
    # At 45° both extents are (W + H) / sqrt(2): the bounding box of the turned rectangle.
    expected = (100.0 + 60.0) / math.sqrt(2.0)
    for mirrored in (False, True):
        assert fov_of(_matrix(1.0, 45.0, mirrored), FRAME) == pytest.approx(
            (expected, expected)
        )


def test_counting_in_chunks_of_one_cell_equals_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix(1.3, 33.0, True)
    shape = (50, 90)
    generator = random.Random(7)
    cells = [
        _cell(generator.uniform(-400.0, 400.0), generator.uniform(-400.0, 400.0))
        for _ in range(60)
    ]

    def chosen() -> list[tuple[float, float, int]]:
        return _choose(
            cells,
            matrix=matrix,
            image_shape_px=shape,
            step_um=0.5 * min(_extent(matrix, shape)),
        )

    whole = chosen()
    monkeypatch.setattr(algorithm, "_WORK_CHUNK", 1)

    assert whole
    assert chosen() == whole


# ----------------------------------------------- the greedy against an exact greedy

#: A small round well, so that the oracle below stays under hypothesis's deadline.
SMALL_WELL = _well(600.0)


def _exact_greedy(
    cells: Sequence[Point], *, n: int, min_cells: int
) -> list[tuple[float, float, int]]:
    """The greedy written out: recount every candidate against the cells not yet held, every round.

    The candidates are the lattice of ``STEP`` centred on the well that keep
    the whole 100 x 60 frame inside it, numbered from +Y down and then from -X
    up; fields that touch are allowed, overlapping ones are not.
    """
    radius = 300.0
    half = int(radius // STEP)
    candidates = [
        ((c - half) * STEP, (half - r) * STEP)
        for r in range(2 * half + 1)
        for c in range(2 * half + 1)
    ]
    candidates = [
        (x, y)
        for x, y in candidates
        if all(
            math.hypot(x + dx, y + dy) <= radius
            for dx in (-50.0, 50.0)
            for dy in (-30.0, 30.0)
        )
    ]

    def holds(x: float, y: float, cx: float, cy: float) -> bool:
        return abs(cx - x) <= 50.0 and abs(cy - y) <= 30.0

    held: set[int] = set()
    alive = list(range(len(candidates)))
    chosen: list[tuple[float, float, int]] = []
    while len(chosen) < n:
        best, best_count = -1, 0
        for k in alive:
            x, y = candidates[k]
            count = sum(
                1
                for i, (cx, cy) in enumerate(cells)
                if i not in held and holds(x, y, cx, cy)
            )
            if count > best_count:
                best, best_count = k, count
        if best < 0 or best_count < min_cells:
            break
        x, y = candidates[best]
        held |= {
            i
            for i, (cx, cy) in enumerate(cells)
            if i not in held and holds(x, y, cx, cy)
        }
        chosen.append((x, y, best_count))
        alive = [
            k
            for k in alive
            if not (
                abs(candidates[k][0] - x) < 100.0 and abs(candidates[k][1] - y) < 60.0
            )
        ]
    return chosen


@settings(max_examples=150)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-40, max_value=40),
            st.integers(min_value=-40, max_value=40),
        ),
        min_size=1,
        max_size=14,
    ),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=3),
)
def test_the_greedy_equals_an_exact_greedy_with_cells_on_the_fields_edges(
    grid: list[tuple[int, int]], n: int, min_cells: int
) -> None:
    """Cells on multiples of 5 µm sit exactly on the edges of fields 100 x 60 µm, and fields touch (no empty field is chosen)."""
    points = [(5.0 * i, 5.0 * j) for i, j in grid]
    cells = [_cell(x, y) for x, y in points]

    chosen = _choose(cells, SMALL_WELL, n=n, min_cells=min_cells)

    assert chosen == _exact_greedy(points, n=n, min_cells=min_cells)
    assert all(count >= min_cells for _, _, count in chosen)
