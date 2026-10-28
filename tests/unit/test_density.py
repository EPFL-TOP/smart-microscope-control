"""The area that was imaged and the counting per well (m3 §10, #193): pure arithmetic on stage µm.

The frame is 60 x 80 pixels (``(height, width)``, not square, FM-73) at 5 µm: a
field of 400 x 300 µm, 0.12 mm². Expected values are written here, from the
geometry and from the measurements of the plan, never read from the result's own
fields (FM-124). The module runs with ``RuntimeWarning`` as an error (FM-131):
numpy warns where Python's floats do not, and on one numpy only.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from smc.plugins.density import algorithm
from smc.plugins.density.algorithm import GridTooFineError, density, scanned_area_mm2
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.slots import Cell, DensityWell, WellGeometry

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

SHAPE = (60, 80)  # (height, width)
PIXEL_UM = 5.0
FIELD_MM2 = 400.0 * 300.0 / 1e6  # 0.12


def _exactly(text: str) -> str:
    return rf"\A{re.escape(text)}\Z"


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


UPRIGHT = _matrix(PIXEL_UM, 0.0, False)


def _disc(
    radius_um: float = 3200.0, centre: tuple[float, float] = (0.0, 0.0)
) -> WellGeometry:
    return WellGeometry("A1", centre, (2 * radius_um, 2 * radius_um), True, 0.0)


def _field(
    x_um: float,
    y_um: float,
    matrix: Matrix2 = UPRIGHT,
    shape: tuple[int, int] = SHAPE,
) -> tuple[tuple[float, float], ...]:
    return footprint(x_um, y_um, matrix=matrix, image_shape_px=shape)


# ----------------------------------------------------------------- the area


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("rotation_deg", [0.0, 30.0, -12.0])
def test_a_full_disc_scan_measures_the_well_area(
    rotation_deg: float, mirrored: bool
) -> None:
    """Fields on a 200 µm lattice cover the plane for every camera turn.

    A 400 x 300 µm rectangle holds the disc of radius 150 around its centre,
    and every point of the plane is within 141 µm of a lattice point. The
    mutant that does not clip to the well gives 37.3 mm² (+16 %).
    """
    well = _disc(3200.0, (1000.0, -500.0))
    matrix = _matrix(PIXEL_UM, rotation_deg, mirrored)
    footprints = [
        _field(1000.0 + 200.0 * i, -500.0 + 200.0 * j, matrix)
        for i in range(-18, 19)
        for j in range(-18, 19)
        if math.hypot(200.0 * i, 200.0 * j) <= 3500.0
    ]

    area = scanned_area_mm2(footprints, well, grid_um=5.0)

    assert area == pytest.approx(math.pi * 3.2**2, rel=0.01)


def test_overlapping_fields_are_not_counted_twice_in_the_area() -> None:
    well = _disc()

    overlapping = scanned_area_mm2(
        [_field(-50.0, 0.0), _field(50.0, 0.0)], well, grid_um=5.0
    )
    twice = scanned_area_mm2([_field(0.0, 0.0), _field(0.0, 0.0)], well, grid_um=5.0)

    # 500 x 300 µm: the sum of the two fields (0.24) is the mutant.
    assert overlapping == pytest.approx(0.15, abs=1e-9)
    assert twice == pytest.approx(FIELD_MM2, abs=1e-9)


def test_touching_fields_add_up_without_a_gap_or_an_overlap() -> None:
    area = scanned_area_mm2(
        [_field(-200.0, 0.0), _field(200.0, 0.0)], _disc(), grid_um=5.0
    )

    assert area == pytest.approx(0.24, abs=1e-9)


def test_a_rotated_mirrored_field_has_the_area_of_the_frame() -> None:
    """The edges' error is at most g / (2 w) per axis: 0.7 % at 2 µm for a 300 µm side."""
    corners = _field(40.0, -25.0, _matrix(PIXEL_UM, 30.0, True))

    area = scanned_area_mm2([corners], _disc(), grid_um=2.0)

    assert area == pytest.approx(FIELD_MM2, rel=0.015)


def test_a_field_is_its_width_by_its_height_and_not_the_other_way_round() -> None:
    """A field of 400 x 300 µm straddling a disc's wall: swapped sides clip a different area (35 % more).

    The disc has radius 1000 µm; the field is centred 400 µm right and 1000 µm
    up of the well's centre. An independent sampling at 0.25 µm gives 0.024864
    mm² clipped, the raster at 5 µm 0.0249; with the sides swapped it is
    0.033482.
    """
    well = _disc(1000.0)
    corners = _field(400.0, 1000.0)

    area = scanned_area_mm2([corners], well, grid_um=5.0)

    assert area == pytest.approx(0.024864, rel=0.03)
    assert area != pytest.approx(0.033482, rel=0.05)


def test_a_square_well_clips_at_its_walls_after_the_plates_turn() -> None:
    """FM-73: a 2000 x 1000 µm well turned by 30°, inside one 3000 x 2500 µm field.

    The well's whole area is 2.0 mm². A circumscribed square taken from
    ``size_um[1]`` gives 0.923, one from ``max(size_um) / 2`` 1.969, and the
    clip to a disc 1.57: none of them is within a tenth of a percent.
    """
    well = WellGeometry("A1", (300.0, -200.0), (2000.0, 1000.0), False, 30.0)
    camera = _matrix(25.0, 12.0, True)
    corners = _field(300.0, -200.0, camera, (100, 120))  # 3000 wide, 2500 high

    area = scanned_area_mm2([corners], well, grid_um=5.0)

    assert area == pytest.approx(2.0, rel=0.001)


def test_footprints_far_from_the_well_take_no_area_and_no_grid() -> None:
    """A frame read back at a wrong stage position leaves the grid and the area alone."""
    well = _disc()
    good = _field(0.0, 0.0)
    turned = _matrix(PIXEL_UM, 12.0, True)

    alone = scanned_area_mm2([good], well, grid_um=5.0)
    assert alone == pytest.approx(FIELD_MM2, abs=1e-9)
    for far in (1e9, 1e11, 9.9e11, -1e11):
        for matrix in (UPRIGHT, turned):
            away = _field(far, far, matrix)
            assert scanned_area_mm2([away], well, grid_um=5.0) == 0.0
            assert scanned_area_mm2([good, away], well, grid_um=5.0) == alone
            assert scanned_area_mm2([away, good], well, grid_um=5.0) == alone


def test_a_grid_point_exactly_on_each_edge_of_a_field_is_inside_it() -> None:
    """FM-132: the windows start and end at the field's edge, and the edge counts (as ``contains`` counts it).

    A square of one µm at the field's lower left, 5 µm out, makes the grid start
    half a step before the field, so that the field's four edges lie on grid
    points: 41 by 31 of them, 0.1271 mm², with the sizes exact in binary. The
    marker holds no grid point itself.
    """
    field = _field(0.0, 0.0)  # x in [-200, 200], y in [-150, 150], exact
    marker = ((-205.0, -155.0), (-204.0, -155.0), (-204.0, -154.0), (-205.0, -154.0))

    area = scanned_area_mm2([field, marker], _disc(), grid_um=10.0)

    assert area == pytest.approx(41 * 31 * 100.0 / 1e6, abs=1e-12)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("beyond", [1e13, -1e13])
def test_a_footprint_beyond_the_coordinate_bound_is_refused(
    axis: int, beyond: float
) -> None:
    """On either axis, either way: FM-73, a bound read on x alone passes a corner at y = 1e13."""
    corners = [list(corner) for corner in _field(0.0, 0.0)]
    corners[1][axis] = beyond

    with pytest.raises(
        ValueError, match=_exactly("footprint 1 has a corner beyond 1e12 µm")
    ):
        scanned_area_mm2(
            [_field(0.0, 0.0), [tuple(c) for c in corners]], _disc(), grid_um=5.0
        )


def test_no_footprint_has_no_area() -> None:
    assert scanned_area_mm2([], _disc(), grid_um=5.0) == 0.0


def test_a_footprint_the_well_does_not_touch_has_no_area() -> None:
    assert scanned_area_mm2([_field(5000.0, 0.0)], _disc(), grid_um=5.0) == 0.0


@pytest.mark.parametrize(("x_um", "y_um"), [(-1200.0, 0.0), (0.0, 1150.0)])
def test_a_field_that_only_touches_the_wells_extent_has_no_area(
    x_um: float, y_um: float
) -> None:
    """The field's edge is exactly the extent's (x = -1000, or y = 1000): an empty extent, not a grid of no points.

    A ``>=`` there would be a grid of 0 points and a division by zero in the bands.
    """
    well = _disc(1000.0)

    assert scanned_area_mm2([_field(x_um, y_um)], well, grid_um=5.0) == 0.0


def _brute_force_mm2(
    footprints: Sequence[Sequence[tuple[float, float]]], well: WellGeometry, g: float
) -> float:
    """Every grid point against every footprint by cross products, then the scalar ``contains``.

    The grid is the module's specification: the footprints' bounding box cut
    to the well's circumscribed square, points at the centres of cells of ``g``.
    """
    half = (
        well.size_um[0] / 2.0
        if well.circular
        else math.hypot(well.size_um[0] / 2.0, well.size_um[1] / 2.0)
    )
    cx, cy = well.center_um
    all_x = [x for corners in footprints for x, _ in corners]
    all_y = [y for corners in footprints for _, y in corners]
    x_lo, x_hi = max(min(all_x), cx - half), min(max(all_x), cx + half)
    y_lo, y_hi = max(min(all_y), cy - half), min(max(all_y), cy + half)
    if not (x_hi > x_lo and y_hi > y_lo):
        return 0.0
    nx, ny = math.ceil((x_hi - x_lo) / g), math.ceil((y_hi - y_lo) / g)
    px, py = np.meshgrid(
        x_lo + (np.arange(nx) + 0.5) * g, y_lo + (np.arange(ny) + 0.5) * g
    )
    covered = np.zeros(px.shape, dtype=bool)
    for corners in footprints:
        crosses = []
        for k in range(4):
            (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
            crosses.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))
        covered |= np.all([c >= 0.0 for c in crosses], axis=0) | np.all(
            [c <= 0.0 for c in crosses], axis=0
        )
    count = sum(
        1
        for x, y in zip(px[covered], py[covered], strict=True)
        if well.contains(float(x), float(y))
    )
    return count * g * g / 1e6


def _random_case(
    rng: random.Random,
) -> tuple[WellGeometry, list[tuple[tuple[float, float], ...]]]:
    circular = rng.random() < 0.5
    width = rng.uniform(500.0, 2000.0)
    well = WellGeometry(
        "A1",
        (rng.uniform(-900.0, 900.0), rng.uniform(-900.0, 900.0)),
        (width, width if circular else rng.uniform(400.0, 2000.0)),
        circular,
        rng.uniform(-180.0, 180.0),
    )
    fields = []
    for _ in range(rng.randint(1, 8)):
        matrix = _matrix(
            rng.uniform(2.0, 8.0), rng.uniform(-180.0, 180.0), rng.random() < 0.5
        )
        shape = (rng.randint(20, 90), rng.randint(20, 120))
        fields.append(
            _field(
                well.center_um[0] + rng.uniform(-0.8, 0.8) * width,
                well.center_um[1] + rng.uniform(-0.8, 0.8) * width,
                matrix,
                shape,
            )
        )
    return well, fields


def test_the_windows_lose_no_point() -> None:
    """60 seeded cases equal a brute force over every point and every footprint, to 1e-12 mm².

    No ``hypothesis`` (FM-132): its exact zeros put points on edges, where two
    correct definitions differ by one ulp; the centres of the grid here sit
    ``g / 2`` off any aligned edge, and the footprints are turned at random.
    """
    rng = random.Random(20261028)
    for case in range(60):
        well, fields = _random_case(rng)
        g = (7.3, 11.0, 13.7)[case % 3]

        area = scanned_area_mm2(fields, well, grid_um=g)

        assert area == pytest.approx(_brute_force_mm2(fields, well, g), abs=1e-12), case


@pytest.mark.parametrize("band", [1, 7, 64])
def test_banding_does_not_change_the_area(
    band: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(7)
    cases = [_random_case(rng) for _ in range(5)]
    whole = [scanned_area_mm2(f, w, grid_um=9.0) for w, f in cases]
    assert all(area > 0.0 for area in whole)

    monkeypatch.setattr(algorithm, "_BAND_ELEMENTS", band)

    assert [scanned_area_mm2(f, w, grid_um=9.0) for w, f in cases] == whole


# ---------------------------------------------------------- what is refused


def test_a_footprint_that_is_not_a_parallelogram_is_refused() -> None:
    w, h = 400.0, 300.0
    z_order = ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))
    trapezoid = ((0.0, 0.0), (w, 0.0), (300.0, h), (0.0, h))

    with pytest.raises(
        ValueError,
        match=_exactly(
            "footprint 0 is not a parallelogram: corner 2 is 800 µm from where "
            "corners 0, 1 and 3 put it"
        ),
    ):
        scanned_area_mm2([z_order], _disc(), grid_um=5.0)
    with pytest.raises(
        ValueError,
        match=_exactly(
            "footprint 1 is not a parallelogram: corner 2 is 100 µm from where "
            "corners 0, 1 and 3 put it"
        ),
    ):
        scanned_area_mm2([_field(0.0, 0.0), trapezoid], _disc(), grid_um=5.0)


def test_a_footprint_with_no_area_is_refused() -> None:
    line = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    point = ((5.0, 5.0),) * 4

    for corners in (line, point):
        with pytest.raises(ValueError, match=_exactly("footprint 0 has no area")):
            scanned_area_mm2([corners], _disc(), grid_um=5.0)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_footprint_with_a_corner_that_is_not_finite_is_refused(
    bad: float, axis: int
) -> None:
    corners = [list(corner) for corner in _field(0.0, 0.0)]
    corners[2][axis] = bad

    with pytest.raises(
        ValueError, match=_exactly("footprint 1 has a corner that is not finite")
    ):
        scanned_area_mm2(
            [_field(0.0, 0.0), [tuple(c) for c in corners]], _disc(), grid_um=5.0
        )


@pytest.mark.parametrize("axis", [0, 1])
def test_a_corner_two_that_is_off_on_one_axis_only_is_refused(axis: int) -> None:
    """FM-73: the offset is a distance in the plane, not its x: 50 µm in y alone is a different shape."""
    corners = [list(corner) for corner in _field(0.0, 0.0)]
    corners[2][axis] += 50.0

    with pytest.raises(
        ValueError,
        match=_exactly(
            "footprint 0 is not a parallelogram: corner 2 is 50 µm from where "
            "corners 0, 1 and 3 put it"
        ),
    ):
        scanned_area_mm2([[tuple(c) for c in corners]], _disc(), grid_um=5.0)


def test_the_parallelogram_tolerance_is_a_micrometre_and_not_more() -> None:
    """A thousandth of a µm off is a shape that is not one; a billionth is the rounding of a corner."""
    clean = scanned_area_mm2([_field(0.0, 0.0)], _disc(), grid_um=5.0)
    off = [list(corner) for corner in _field(0.0, 0.0)]
    off[2][0] += 1e-3
    near = [list(corner) for corner in _field(0.0, 0.0)]
    near[2][0] += 1e-9

    with pytest.raises(
        ValueError,
        match=_exactly(
            "footprint 0 is not a parallelogram: corner 2 is 0.001 µm from where "
            "corners 0, 1 and 3 put it"
        ),
    ):
        scanned_area_mm2([[tuple(c) for c in off]], _disc(), grid_um=5.0)
    assert scanned_area_mm2(
        [[tuple(c) for c in near]], _disc(), grid_um=5.0
    ) == pytest.approx(clean, abs=1e-12)


@pytest.mark.parametrize("count", [0, 3, 5])
def test_a_footprint_that_is_not_four_corners_is_refused(count: int) -> None:
    corners = (_field(0.0, 0.0) * 2)[:count]

    with pytest.raises(
        ValueError,
        match=_exactly(f"footprint 1 must have four corners, got {count}"),
    ):
        scanned_area_mm2([_field(0.0, 0.0), corners], _disc(), grid_um=5.0)


def test_the_first_footprint_to_fail_is_the_one_named() -> None:
    no_area = ((0.0, 0.0),) * 4
    not_finite = ((math.nan, 0.0),) * 4

    with pytest.raises(ValueError, match=_exactly("footprint 2 has no area")):
        scanned_area_mm2(
            [_field(0.0, 0.0), _field(9.0, 9.0), no_area, not_finite],
            _disc(),
            grid_um=5.0,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_a_grid_that_is_not_a_size_is_refused(bad: float) -> None:
    message = f"grid_um must be finite and positive, got {bad!r}"
    with pytest.raises(ValueError, match=_exactly(message)) as info:
        scanned_area_mm2([_field(0.0, 0.0)], _disc(), grid_um=bad)

    # A size that is not a size is not a grid that is too fine.
    assert info.type is ValueError


def test_a_grid_that_is_not_a_size_is_refused_even_with_nothing_to_measure() -> None:
    with pytest.raises(
        ValueError, match=_exactly("grid_um must be finite and positive, got 0.0")
    ):
        scanned_area_mm2([], _disc(), grid_um=0.0)


# ------------------------------------------------------ the bounds (FM-74)


class _NoAllocation:
    """``numpy`` as the algorithm sees it, except that a grid or a band cannot be made.

    ``np.zeros`` and ``np.arange`` are what allocate; the refusal must come
    before either. Everything else is numpy's own.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    def zeros(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("allocated before the refusal")

    arange = zeros


def _forbid_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(algorithm, "np", _NoAllocation())


def test_the_stand_in_for_numpy_is_reached_by_a_legal_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control of the tests below: a grid that is allowed does allocate."""
    _forbid_allocation(monkeypatch)

    with pytest.raises(AssertionError, match="allocated before the refusal"):
        scanned_area_mm2([_field(0.0, 0.0)], _disc(), grid_um=5.0)


@pytest.mark.parametrize("grid_um", [5e-324, 1e-300, 1e-9, 0.01])
def test_a_grid_too_fine_for_the_well_is_refused_before_allocating(
    grid_um: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``400.0 / 5e-324`` is ``inf`` and ``math.ceil(inf)`` an ``OverflowError``; 1e-300 would print 300 digits."""
    _forbid_allocation(monkeypatch)
    sides = {
        5e-324: "inf by inf points, more than 131072 on a side",
        1e-300: "4e+302 by 3e+302 points, more than 131072 on a side",
        1e-9: "4e+11 by 3e+11 points, more than 131072 on a side",
        0.01: "40000 by 30000 points, more than 1073741824 in all",
    }

    with pytest.raises(GridTooFineError) as info:
        scanned_area_mm2([_field(0.0, 0.0)], _disc(), grid_um=grid_um)

    assert str(info.value) == (
        f"well A1: a grid of {grid_um:g} µm over its imaged area would hold {sides[grid_um]}"
    )


@pytest.mark.parametrize(
    ("constant", "allowed", "refusal"),
    [
        (
            "_MAX_SIDE_POINTS",
            40,
            "well A1: a grid of 10 µm over its imaged area would hold 40 by 30 points, "
            "more than 39 on a side",
        ),
        (
            "_MAX_GRID_POINTS",
            1200,
            "well A1: a grid of 10 µm over its imaged area would hold 40 by 30 points, "
            "more than 1199 in all",
        ),
        (
            "_MAX_TESTS",
            1200,
            "well A1: counting its fields on a grid of 10 µm would make 1200 tests, "
            "more than 1199",
        ),
    ],
)
def test_the_largest_legal_grid_is_still_allowed(
    constant: str, allowed: int, refusal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grid of 40 by 30 points, 1200 in all, 1200 tests: at each bound it is allowed, one less refuses it."""
    corners = _field(0.0, 0.0)

    monkeypatch.setattr(algorithm, constant, allowed)
    assert scanned_area_mm2([corners], _disc(), grid_um=10.0) == pytest.approx(
        FIELD_MM2, abs=1e-9
    )

    monkeypatch.setattr(algorithm, constant, allowed - 1)
    with pytest.raises(GridTooFineError) as info:
        scanned_area_mm2([corners], _disc(), grid_um=10.0)
    assert str(info.value) == refusal


def test_too_many_tests_are_refused_before_the_first_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fifty identical fields, each a window of 1200 points: 60 000 tests."""
    footprints = [_field(0.0, 0.0)] * 50
    monkeypatch.setattr(algorithm, "_MAX_TESTS", 59_999)
    _forbid_allocation(monkeypatch)

    with pytest.raises(GridTooFineError) as info:
        scanned_area_mm2(footprints, _disc(), grid_um=10.0)

    assert str(info.value) == (
        "well A1: counting its fields on a grid of 10 µm would make 60000 tests, "
        "more than 59999"
    )
    monkeypatch.setattr(algorithm, "_MAX_TESTS", 60_000)
    with pytest.raises(AssertionError, match="allocated before the refusal"):
        scanned_area_mm2(footprints, _disc(), grid_um=10.0)


def test_a_far_footprint_is_no_test_of_the_window_and_no_grid() -> None:
    """The windows of a field 1e11 µm away are empty: the bound counts only what meets the well."""
    far = [_field(1e11, 0.0) for _ in range(1000)]

    assert scanned_area_mm2(
        [*far, _field(0.0, 0.0)], _disc(), grid_um=10.0
    ) == pytest.approx(FIELD_MM2, abs=1e-9)


def test_a_tall_field_meets_the_side_bound_on_its_height() -> None:
    """FM-73, FM-74: every other bounds test is wider than tall, where x binds; this one is 300 wide by 400 high."""
    tall = _field(0.0, 0.0, UPRIGHT, (80, 60))  # (height, width): 300 x 400 µm
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(algorithm, "_MAX_SIDE_POINTS", 35)
        with pytest.raises(GridTooFineError) as info:
            scanned_area_mm2([tall], _disc(), grid_um=10.0)
        assert str(info.value) == (
            "well A1: a grid of 10 µm over its imaged area would hold 30 by 40 "
            "points, more than 35 on a side"
        )
        patch.setattr(algorithm, "_MAX_SIDE_POINTS", 40)
        assert scanned_area_mm2([tall], _disc(), grid_um=10.0) == pytest.approx(
            FIELD_MM2, abs=1e-9
        )


def test_a_field_below_the_resolution_of_the_grid_is_what_the_raster_makes_of_it() -> (
    None
):
    """FM-138, recorded and not endorsed: the points are cell centres, so a field under half a step has none.

    The area of one field is read to the step, not better. At 100 µm a field of
    40 µm is 0.0 mm² (true 0.0016) and one of 130 µm is 0.01 (true 0.0169); at
    5 µm both are exact. A change of the grid's origin or of the sampling must
    change this test on purpose.
    """
    small = _field(0.0, 0.0, UPRIGHT, (8, 8))  # 40 x 40 µm
    larger = _field(0.0, 0.0, UPRIGHT, (26, 26))  # 130 x 130 µm

    assert scanned_area_mm2([small], _disc(), grid_um=100.0) == 0.0
    assert scanned_area_mm2([larger], _disc(), grid_um=100.0) == pytest.approx(
        0.01, abs=1e-12
    )
    assert scanned_area_mm2([small], _disc(), grid_um=5.0) == pytest.approx(
        0.0016, abs=1e-12
    )
    assert scanned_area_mm2([larger], _disc(), grid_um=5.0) == pytest.approx(
        0.0169, abs=1e-12
    )


def test_a_far_footprint_that_straddles_a_power_of_two_is_ignored_and_not_refused() -> (
    None
):
    """Where the spacing of the floats doubles inside a footprint, corner 2 is off by an ulp or two.

    A frame read back at 2**33 µm or further (8.6 km of stage) with its corners on
    both sides of a power of two is a parallelogram only to a few ulps, and an
    absolute 1e-6 µm refuses it, though a far frame is to be ignored. The guard
    counts the positions that the plain tolerance would have refused: they are
    the case, and without them the test would not be one.
    """
    turned = _matrix(PIXEL_UM, 30.0, True)
    well = _disc()
    good = _field(0.0, 0.0)
    alone = scanned_area_mm2([good], well, grid_um=5.0)
    rng = random.Random(33)
    beyond_plain_tolerance = 0
    for power in (33, 36, 39):
        for _ in range(100):
            far = _field(
                2.0**power + rng.uniform(-150.0, 150.0), rng.uniform(-1e3, 1e3), turned
            )
            (c0x, c0y), (c1x, c1y), (c2x, c2y), (c3x, c3y) = far
            off = math.hypot(c2x - (c1x + c3x - c0x), c2y - (c1y + c3y - c0y))
            beyond_plain_tolerance += off > 1e-6

            assert scanned_area_mm2([good, far], well, grid_um=5.0) == alone
    assert beyond_plain_tolerance > 20


def test_the_tolerance_that_grows_with_the_coordinates_still_refuses_a_real_trapezoid() -> (
    None
):
    """At 5.5e11 µm a few ulps are 1e-3 µm: a corner 100 µm off is still a shape that is not a parallelogram."""
    base = 2.0**39
    trapezoid = [
        (base, base),
        (base + 400.0, base),
        (base + 300.0, base + 300.0),
        (base, base + 300.0),
    ]

    with pytest.raises(
        ValueError,
        match=_exactly(
            "footprint 0 is not a parallelogram: corner 2 is 100 µm from where "
            "corners 0, 1 and 3 put it"
        ),
    ):
        scanned_area_mm2([trapezoid], _disc(), grid_um=5.0)


# ---------------------------------------------------------------- density()

LABELS = ("single", "doublet", "debris")


def _cell(well: str, x_um: float, y_um: float, label: str = "single") -> Cell:
    return Cell(
        well=well,
        x_um=x_um,
        y_um=y_um,
        label=label,
        score=0.9,
        frame=0,
        box_px=(0.0, 0.0, 1.0, 1.0),
    )


def _named(
    name: str, centre: tuple[float, float], radius_um: float = 1000.0
) -> WellGeometry:
    return WellGeometry(name, centre, (2 * radius_um, 2 * radius_um), True, 0.0)


def _count(
    cells: Sequence[Cell],
    wells: Sequence[WellGeometry],
    *,
    areas: dict[str, float] | None = None,
    fields: dict[str, int] | None = None,
    failed: dict[str, int] | None = None,
    labels: Sequence[str] = LABELS,
) -> tuple[DensityWell, ...]:
    names = [w.well for w in wells]
    return density(
        cells,
        wells=wells,
        areas_mm2=areas if areas is not None else dict.fromkeys(names, 0.25),
        fields=fields if fields is not None else dict.fromkeys(names, 1),
        failed_fields=failed if failed is not None else dict.fromkeys(names, 0),
        labels=labels,
    )


def test_counts_are_per_label_zeros_included_and_per_mm2_divides_by_the_area() -> None:
    well = _named("A1", (0.0, 0.0))
    cells = [_cell("A1", 10.0 * i, 5.0) for i in range(6)] + [
        _cell("A1", -20.0 * i, -5.0, "doublet") for i in range(2)
    ]

    (result,) = _count(cells, [well], fields={"A1": 4}, failed={"A1": 1})

    assert result.well == "A1"
    assert result.counts == {"single": 6, "doublet": 2, "debris": 0}
    assert list(result.counts) == list(LABELS)
    assert result.per_mm2 == {"single": 24.0, "doublet": 8.0, "debris": 0.0}
    assert result.area_mm2 == 0.25
    assert (result.fields, result.failed_fields, result.outside_well) == (4, 1, 0)


def test_a_cell_on_the_plastic_counts_outside() -> None:
    """Radius 1000 µm: 999 is inside, exactly 1000 is the edge and counts, 1001 is on the plastic."""
    well = _named("A1", (500.0, -300.0))
    cells = [
        _cell("A1", 500.0 + 999.0, -300.0),
        _cell("A1", 500.0 + 1000.0, -300.0),
        _cell("A1", 500.0 + 1001.0, -300.0, "doublet"),
    ]

    (result,) = _count(cells, [well])

    assert result.counts == {"single": 2, "doublet": 0, "debris": 0}
    assert result.outside_well == 1
    assert result.per_mm2 == {"single": 8.0, "doublet": 0.0, "debris": 0.0}


def test_a_well_with_no_frame_has_unknown_density() -> None:
    well = _named("A1", (0.0, 0.0))

    (result,) = _count(
        [], [well], areas={"A1": 0.0}, fields={"A1": 0}, failed={"A1": 4}
    )

    assert result.per_mm2 is None
    assert result.counts == {"single": 0, "doublet": 0, "debris": 0}
    assert (result.fields, result.failed_fields, result.area_mm2) == (0, 4, 0.0)


def test_cells_of_different_wells_are_counted_apart_and_the_order_of_wells_is_kept() -> (
    None
):
    b1 = _named("B1", (0.0, -9000.0))
    a1 = _named("A1", (0.0, 0.0))
    cells = [
        _cell("A1", 1.0, 1.0),
        _cell("B1", 1.0, -9000.0, "doublet"),
        _cell("A1", 2.0, 2.0),
        _cell("B1", 2.0, -9000.0, "doublet"),
        _cell("B1", 3.0, -9000.0, "doublet"),
    ]

    result = _count(
        cells, [b1, a1], areas={"B1": 0.5, "A1": 0.25}, fields={"B1": 2, "A1": 3}
    )

    assert [w.well for w in result] == ["B1", "A1"]
    assert result[0].counts == {"single": 0, "doublet": 3, "debris": 0}
    assert result[0].per_mm2 == {"single": 0.0, "doublet": 6.0, "debris": 0.0}
    assert result[1].counts == {"single": 2, "doublet": 0, "debris": 0}
    assert result[1].per_mm2 == {"single": 8.0, "doublet": 0.0, "debris": 0.0}
    assert (result[0].fields, result[1].fields) == (2, 3)


def test_every_cell_is_counted_once_or_outside() -> None:
    """Seeded cells over two wells, some on the plastic: each is classified as the scalar ``contains`` says."""
    rng = random.Random(11)
    wells = [_named("A1", (0.0, 0.0), 1000.0), _named("A2", (3000.0, 100.0), 800.0)]
    cells = [
        _cell(
            well.well,
            well.center_um[0] + rng.uniform(-1300.0, 1300.0),
            well.center_um[1] + rng.uniform(-1300.0, 1300.0),
            rng.choice(LABELS),
        )
        for well in wells
        for _ in range(200)
    ]

    result = _count(cells, wells)

    for well, counted in zip(wells, result, strict=True):
        mine = [c for c in cells if c.well == well.well]
        inside = [c for c in mine if well.contains(c.x_um, c.y_um)]
        assert 0 < len(inside) < len(mine)
        assert sum(counted.counts.values()) + counted.outside_well == len(mine)
        assert counted.outside_well == len(mine) - len(inside)
        for label in LABELS:
            assert counted.counts[label] == sum(1 for c in inside if c.label == label)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"labels": ()}, "labels must name at least one label"),
        ({"labels": ("single", "doublet", "single")}, "label 'single' is repeated"),
        ({"wells": "twice"}, "well 'A1' is listed twice"),
        ({"areas": {"B1": 0.25}}, "areas_mm2 has no entry for well 'A1'"),
        ({"fields": {"A1": 1}}, "fields has no entry for well 'B1'"),
        ({"failed": {}}, "failed_fields has no entry for well 'A1'"),
        (
            {"areas": {"A1": math.nan, "B1": 0.25}},
            "areas_mm2 of well 'A1' must be finite and not negative, got nan",
        ),
        (
            {"areas": {"A1": 0.25, "B1": -0.5}},
            "areas_mm2 of well 'B1' must be finite and not negative, got -0.5",
        ),
        (
            {"areas": {"A1": math.inf, "B1": 0.25}},
            "areas_mm2 of well 'A1' must be finite and not negative, got inf",
        ),
        (
            {"fields": {"A1": 1, "B1": -1}},
            "fields of well 'B1' must not be negative, got -1",
        ),
        (
            {"failed": {"A1": -2, "B1": 0}},
            "failed_fields of well 'A1' must not be negative, got -2",
        ),
        (
            {"cells": [_cell("A1", 0.0, 0.0), _cell("C3", 0.0, 0.0)]},
            "a cell is in well 'C3', which is not among the wells",
        ),
        (
            {"cells": [_cell("A1", 0.0, 0.0, "mitotic"), _cell("C3", 0.0, 0.0)]},
            "a cell of well 'A1' has label 'mitotic', which is not among the labels "
            "['single', 'doublet', 'debris']",
        ),
        (
            {"cells": [_cell("C3", 0.0, 0.0, "mitotic")]},
            "a cell is in well 'C3', which is not among the wells",
        ),
    ],
)
def test_density_refuses_what_it_cannot_count(
    kwargs: dict[str, Any], reason: str
) -> None:
    """The text is the whole of it (FM-45); the first cell that fails is the one named, its well before its label."""
    a1, b1 = _named("A1", (0.0, 0.0)), _named("B1", (3000.0, 0.0))
    wells = [a1, a1] if kwargs.pop("wells", None) == "twice" else [a1, b1]
    cells = kwargs.pop("cells", [])

    with pytest.raises(ValueError, match=_exactly(reason)):
        _count(cells, wells, **kwargs)


def test_a_missing_entry_is_named_well_by_well_and_then_mapping_by_mapping() -> None:
    """A1 lacks ``fields`` and B1 lacks ``areas_mm2``: the first well's is the one named."""
    a1, b1 = _named("A1", (0.0, 0.0)), _named("B1", (3000.0, 0.0))

    with pytest.raises(ValueError, match=_exactly("fields has no entry for well 'A1'")):
        _count([], [a1, b1], areas={"A1": 0.25}, fields={"B1": 1})
