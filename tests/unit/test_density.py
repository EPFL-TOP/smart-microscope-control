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
from smc.plugins.density.algorithm import GridTooFineError, scanned_area_mm2
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.slots import WellGeometry

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


def test_a_footprint_beyond_the_coordinate_bound_is_refused() -> None:
    with pytest.raises(
        ValueError, match=_exactly("footprint 1 has a corner beyond 1e12 µm")
    ):
        scanned_area_mm2([_field(0.0, 0.0), _field(1e13, 0.0)], _disc(), grid_um=5.0)


def test_no_footprint_has_no_area() -> None:
    assert scanned_area_mm2([], _disc(), grid_um=5.0) == 0.0


def test_a_footprint_the_well_does_not_touch_has_no_area() -> None:
    assert scanned_area_mm2([_field(5000.0, 0.0)], _disc(), grid_um=5.0) == 0.0


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


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_footprint_with_a_corner_that_is_not_finite_is_refused(bad: float) -> None:
    corners = list(_field(0.0, 0.0))
    corners[2] = (corners[2][0], bad)

    with pytest.raises(
        ValueError, match=_exactly("footprint 1 has a corner that is not finite")
    ):
        scanned_area_mm2([_field(0.0, 0.0), corners], _disc(), grid_um=5.0)


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
