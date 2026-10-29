"""The field grid of a scan plan: size, clipping to the well, estimate, gaps; pure (m3 §7.2, #21).

Ported from the geometry of ``nikon-control``'s ``test_scan.py``, plus the
clipping. The clipping is checked against an oracle that this file writes
itself and that shares nothing with the code but ``WellGeometry.contains`` and
``stage_of_pixel``: a 40 x 40 sampling of each field's frame, a field being kept
when one sample is inside the well. The frames are not square (FM-73), and one
camera is turned and mirrored, because a mirror reverses the corners of the
footprint and an inside test that assumes one orientation passes every
unmirrored test.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import pytest
import useq

from smc.plugins.geometry import (
    Matrix2,
    footprint,
    fov_of,
    pixel_matrix,
    stage_of_pixel,
)
from smc.plugins.scan_plan.algorithm import (
    _disc_meets,
    _polygons_overlap,
    estimate_s,
    gap_overlap,
    grid_size,
    plan_fields,
    turn_from_axes_deg,
)
from smc.plugins.slots import PlannedField, PlateResult, TimingsResult, well_geometry

Footprint = Callable[[float, float], Sequence[tuple[float, float]]]

# Their rig: 2048 px of 6.5 µm at 40x -> 0.1625 µm/px -> a 332.8 µm field.
FOV = (332.8, 332.8)
FRAME = (300, 512)  # (height, width): not square, on purpose (FM-73)
SAMPLES = 40


def _plate(
    name: str = "96-well",
    *,
    a1_um: tuple[float, float] = (1000.0, -500.0),
    rotation_deg: float = 0.0,
) -> PlateResult:
    return PlateResult(
        plate=name,
        a1_center_xy_um=a1_um,
        rotation_deg=rotation_deg,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


def _footprint_of(matrix: Matrix2, shape: tuple[int, int]) -> Footprint:
    return lambda x, y: footprint(x, y, matrix=matrix, image_shape_px=shape)


def _plan(
    plate: PlateResult,
    wells: Sequence[str],
    *,
    matrix: Matrix2,
    shape: tuple[int, int],
    rows: int,
    columns: int,
    overlap: float = 0.1,
    clip_to_well: bool = True,
) -> tuple[useq.WellPlatePlan, tuple[PlannedField, ...]]:
    return plan_fields(
        plate,
        wells,
        fov_um=fov_of(matrix, shape),
        footprint_of=_footprint_of(matrix, shape),
        rows=rows,
        columns=columns,
        overlap=overlap,
        clip_to_well=clip_to_well,
    )


def _names_touching_the_well(
    plan: useq.WellPlatePlan, matrix: Matrix2, shape: tuple[int, int]
) -> set[str]:
    """The oracle: a field is kept when one of 40 x 40 samples of its frame is inside the well."""
    height, width = shape
    pixels = [
        (-0.5 + (i + 0.5) * width / SAMPLES, -0.5 + (j + 0.5) * height / SAMPLES)
        for i in range(SAMPLES)
        for j in range(SAMPLES)
    ]
    geometries = {}
    kept: set[str] = set()
    for position in plan.image_positions:
        assert position.x is not None
        assert position.y is not None
        name = str(position.name)
        well = name.rsplit("_", 1)[0]
        if well not in geometries:
            geometries[well] = well_geometry(plan, well)
        for col, row in pixels:
            x, y = stage_of_pixel(
                position.x,
                position.y,
                col,
                row,
                matrix=matrix,
                image_shape_px=shape,
            )
            if geometries[well].contains(x, y):
                kept.add(name)
                break
    return kept


# --------------------------------------------------------------- grid size


def test_covering_a_96_well_at_40x_takes_the_expected_field_count() -> None:
    """The number that decides whether a survey is minutes or hours."""
    well = (6400.0, 6400.0)

    assert grid_size(FOV, well, coverage=1.0, overlap=0.0) == (20, 20)  # 19.2 -> 20
    assert grid_size(FOV, well, coverage=0.33, overlap=0.0) == (7, 7)


def test_grid_size_takes_rows_from_heights_and_columns_from_widths() -> None:
    """FM-73: a swap is invisible on a square frame and on a square well."""
    assert grid_size((512.0, 300.0), (6400.0, 6400.0), coverage=1.0, overlap=0.1) == (
        24,
        14,
    )
    # Rows from the heights, columns from the widths, on a well that is not square either.
    assert grid_size((512.0, 300.0), (3200.0, 6400.0), coverage=1.0, overlap=0.1) == (
        24,
        7,
    )


def test_a_well_that_one_field_already_covers_takes_one_field_per_axis() -> None:
    """A 1536-well well is 1.7 mm and a 4x field 3 mm: a second row of fields images nothing new."""
    assert grid_size((3000.0, 3000.0), (1700.0, 1700.0), coverage=1.0, overlap=0.1) == (
        1,
        1,
    )
    # And the edge: a field exactly as wide as the span needs no second one.
    assert grid_size((500.0, 500.0), (500.0, 500.0), coverage=1.0, overlap=0.0) == (
        1,
        1,
    )
    assert grid_size((500.0, 500.0), (501.0, 501.0), coverage=1.0, overlap=0.0) == (
        2,
        2,
    )


def test_a_coverage_far_below_one_field_still_takes_one_field_per_axis() -> None:
    """85 µm of a 1.7 mm well under a 3 mm field is a negative span, not a negative count."""
    assert grid_size(
        (3000.0, 3000.0), (1700.0, 1700.0), coverage=0.05, overlap=0.1
    ) == (
        1,
        1,
    )


@pytest.mark.parametrize(
    ("fov", "well", "coverage", "overlap"),
    [
        ((0.0, 300.0), (6400.0, 6400.0), 1.0, 0.1),
        ((300.0, -1.0), (6400.0, 6400.0), 1.0, 0.1),
        ((float("nan"), 300.0), (6400.0, 6400.0), 1.0, 0.1),
        ((300.0, 300.0), (float("inf"), 6400.0), 1.0, 0.1),
        ((300.0, 300.0), (6400.0, 6400.0), 0.0, 0.1),
        ((300.0, 300.0), (6400.0, 6400.0), 1.0, 1.0),
        ((300.0, 300.0), (6400.0, 6400.0), 1.0, -0.1),
    ],
)
def test_grid_size_refuses_what_is_not_a_size(
    fov: tuple[float, float],
    well: tuple[float, float],
    coverage: float,
    overlap: float,
) -> None:
    with pytest.raises(ValueError, match="grid_size needs"):
        grid_size(fov, well, coverage=coverage, overlap=overlap)


@pytest.mark.parametrize(
    ("field_um", "overlap"),
    [
        (1e-300, 0.1),  # a ratio of 1e304: finite
        (1e-320, 0.0),  # a denormal field: the ratio overflows to infinity
        (5e-324, 0.99),  # the step underflows to 0: a division by zero
    ],
    ids=["tiny", "denormal", "no-step"],
)
def test_a_field_far_smaller_than_the_well_gives_a_huge_grid_not_an_error(
    field_um: float, overlap: float
) -> None:
    """The plugin refuses what exceeds 100 per axis; the arithmetic must reach it (FM-121)."""
    rows, columns = grid_size(
        (field_um, field_um), (6400.0, 6400.0), coverage=1.0, overlap=overlap
    )

    assert rows > 100
    assert columns > 100


# ---------------------------------------------------------------- clipping


def test_a_96_well_at_40x_keeps_305_of_361_fields() -> None:
    """The Ti2-E's assumed camera (1412 x 1412 at 0.275 µm), 10 % overlap (measured)."""
    shape = (1412, 1412)
    matrix = pixel_matrix(0.275)
    plate = _plate()
    well_um = well_geometry(plate.to_plan(), "A1").size_um
    rows, columns = grid_size(fov_of(matrix, shape), well_um, coverage=1.0, overlap=0.1)
    assert (rows, columns) == (19, 19)

    plan, fields = _plan(
        plate, ["A1"], matrix=matrix, shape=shape, rows=rows, columns=columns
    )

    assert len(list(plan.image_positions)) == 361
    assert len(fields) == 305
    assert {field.name for field in fields} == _names_touching_the_well(
        plan, matrix, shape
    )


def test_clipping_a_round_well_matches_a_sampled_footprint() -> None:
    plate = _plate(rotation_deg=13.0)
    matrix = pixel_matrix(1.0)

    plan, fields = _plan(plate, ["B2"], matrix=matrix, shape=FRAME, rows=25, columns=15)

    assert len(list(plan.image_positions)) == 375
    assert len(fields) == 299
    assert {field.name for field in fields} == _names_touching_the_well(
        plan, matrix, FRAME
    )


def test_clipping_a_square_well_follows_the_plate_rotation() -> None:
    plate = _plate("384-well", rotation_deg=20.0)
    matrix = pixel_matrix(1.0)

    plan, fields = _plan(plate, ["B2"], matrix=matrix, shape=FRAME, rows=13, columns=9)

    assert len(list(plan.image_positions)) == 117
    # Ignoring the plate's rotation keeps all 117: the corners of the grid are outside.
    assert len(fields) == 105
    assert {field.name for field in fields} == _names_touching_the_well(
        plan, matrix, FRAME
    )


def test_clipping_with_a_mirrored_turned_camera_matches_a_sampled_footprint() -> None:
    plate = _plate("384-well", rotation_deg=20.0)
    matrix = _matrix(1.0, 30.0, True)

    plan, fields = _plan(plate, ["B2"], matrix=matrix, shape=FRAME, rows=13, columns=9)

    assert len(fields) == 59
    assert {field.name for field in fields} == _names_touching_the_well(
        plan, matrix, FRAME
    )


@pytest.mark.parametrize("mirrored", [False, True])
def test_a_mirror_does_not_change_which_fields_touch_a_round_well(
    mirrored: bool,
) -> None:
    """The footprint is the same area either way: only its corners turn the other way round."""
    plate = _plate(rotation_deg=13.0)

    _, fields = _plan(
        plate,
        ["B2"],
        matrix=_matrix(1.0, 30.0, mirrored),
        shape=FRAME,
        rows=25,
        columns=15,
    )
    _, reference = _plan(
        plate,
        ["B2"],
        matrix=_matrix(1.0, 30.0, False),
        shape=FRAME,
        rows=25,
        columns=15,
    )

    assert [field.name for field in fields] == [field.name for field in reference]


def test_without_clipping_every_field_is_kept() -> None:
    plate = _plate("384-well", rotation_deg=20.0)
    matrix = pixel_matrix(1.0)

    plan, fields = _plan(
        plate,
        ["B2", "A1"],
        matrix=matrix,
        shape=FRAME,
        rows=13,
        columns=9,
        clip_to_well=False,
    )

    assert len(fields) == 2 * 117
    assert [field.name for field in fields] == [
        str(position.name) for position in plan.image_positions
    ]


def test_a_square_well_is_not_a_disc() -> None:
    """The swapped shape (FM-73): on a 384-well well a disc of the same width loses the corners."""
    plate = _plate("384-well", rotation_deg=0.0)
    matrix = pixel_matrix(1.0)
    shape = (300, 300)

    plan, fields = _plan(
        plate, ["A1"], matrix=matrix, shape=shape, rows=13, columns=13, overlap=0.0
    )

    corner = f"A1_{0:04d}"
    assert corner in {field.name for field in fields}
    assert len(plan.image_positions) == 169
    assert {field.name for field in fields} == _names_touching_the_well(
        plan, matrix, shape
    )


# ---------------------------------------------------------- order and names


def test_wells_are_visited_in_the_given_order_not_raster() -> None:
    plan, fields = _plan(
        _plate(),
        ("B1", "A6"),
        matrix=pixel_matrix(1.0),
        shape=FRAME,
        rows=2,
        columns=3,
        clip_to_well=False,
    )

    assert [str(name) for name in plan.selected_well_names] == ["B1", "A6"]
    assert [field.well for field in fields] == ["B1"] * 6 + ["A6"] * 6


def test_index_counts_from_zero_in_each_well_and_names_stay_useqs() -> None:
    shape = (1412, 1412)
    matrix = pixel_matrix(0.275)

    _, fields = _plan(
        _plate(), ["A1", "A2"], matrix=matrix, shape=shape, rows=19, columns=19
    )

    for well in ("A1", "A2"):
        mine = [field for field in fields if field.well == well]
        assert [field.index for field in mine] == list(range(len(mine)))
        assert len(mine) == 305
        # The corner of the grid misses the round well, so useq's name has a gap
        # while the index does not.
        assert f"{well}_0000" not in {field.name for field in mine}
        assert mine[0].name != f"{well}_0000"
        assert all(field.name.startswith(f"{well}_") for field in mine)


def test_fields_are_centred_on_the_well() -> None:
    """A 3 x 3 grid's middle field sits at the well centre."""
    plate = _plate(a1_um=(1000.0, 2000.0))
    plan, fields = _plan(
        plate,
        ["A1"],
        matrix=pixel_matrix(0.1625),
        shape=(2048, 2048),
        rows=3,
        columns=3,
        overlap=0.0,
        clip_to_well=False,
    )

    assert plan.a1_center_xy == (1000.0, 2000.0)
    middle = [
        field
        for field in fields
        if abs(field.x_um - 1000.0) < 1 and abs(field.y_um - 2000.0) < 1
    ]
    assert len(middle) == 1, [(f.x_um, f.y_um) for f in fields]


def test_field_spacing_follows_the_overlap() -> None:
    plate = _plate()
    matrix = pixel_matrix(0.1625)
    shape = (2048, 2048)

    _, flush = _plan(
        plate,
        ["A1"],
        matrix=matrix,
        shape=shape,
        rows=1,
        columns=2,
        overlap=0.0,
        clip_to_well=False,
    )
    _, lapped = _plan(
        plate,
        ["A1"],
        matrix=matrix,
        shape=shape,
        rows=1,
        columns=2,
        overlap=0.5,
        clip_to_well=False,
    )

    assert abs(flush[1].x_um - flush[0].x_um) == pytest.approx(FOV[0], rel=0.01)
    assert abs(lapped[1].x_um - lapped[0].x_um) == pytest.approx(FOV[0] * 0.5, rel=0.01)


def test_the_fields_are_positions_in_the_stage_frame_of_the_registered_plate() -> None:
    """A well's fields are centred on the well's own centre, which the plate's rotation moves."""
    plate = _plate(rotation_deg=13.0)
    centre = well_geometry(plate.to_plan(), "B2").center_um

    _, fields = _plan(
        plate,
        ["B2"],
        matrix=pixel_matrix(1.0),
        shape=FRAME,
        rows=3,
        columns=3,
        clip_to_well=False,
    )

    assert sum(f.x_um for f in fields) / 9 == pytest.approx(centre[0])
    assert sum(f.y_um for f in fields) / 9 == pytest.approx(centre[1])


# ---------------------------------------------------------------- refusals


def test_a_plan_with_no_wells_or_a_nonsense_grid_is_refused() -> None:
    matrix = pixel_matrix(1.0)

    with pytest.raises(ValueError, match="no wells to plan"):
        _plan(_plate(), [], matrix=matrix, shape=FRAME, rows=3, columns=3)
    with pytest.raises(
        ValueError, match="a grid needs at least one row and one column"
    ):
        _plan(_plate(), ["A1"], matrix=matrix, shape=FRAME, rows=0, columns=3)
    with pytest.raises(
        ValueError, match="a grid needs at least one row and one column"
    ):
        _plan(_plate(), ["A1"], matrix=matrix, shape=FRAME, rows=3, columns=0)
    with pytest.raises(ValueError, match=r"overlap must be in \[0, 1\)"):
        _plan(
            _plate(), ["A1"], matrix=matrix, shape=FRAME, rows=1, columns=1, overlap=1.0
        )
    with pytest.raises(ValueError, match=r"overlap must be in \[0, 1\)"):
        _plan(
            _plate(),
            ["A1"],
            matrix=matrix,
            shape=FRAME,
            rows=1,
            columns=1,
            overlap=float("nan"),
        )


def test_a_well_that_is_not_on_the_plate_is_refused() -> None:
    with pytest.raises(ValueError, match="wells not on the 96-well plate: 'I1'"):
        _plan(
            _plate(),
            ["A1", "I1"],
            matrix=pixel_matrix(1.0),
            shape=FRAME,
            rows=1,
            columns=1,
        )


@pytest.mark.parametrize(
    ("wells", "named"),
    [(["A1", "A2"], "wells A1, A2"), (["A1"], "well A1")],
    ids=["two-wells", "one-well"],
)
def test_a_well_that_no_field_touches_is_refused_not_dropped(
    wells: list[str], named: str
) -> None:
    """A well with no field would be skipped by the scan and named by nobody."""

    def far_away(_x: float, _y: float) -> list[tuple[float, float]]:
        return [(1e6, 1e6), (1e6 + 10, 1e6), (1e6 + 10, 1e6 + 10), (1e6, 1e6 + 10)]

    message = (
        f"no field of the grid touches {named}: give rows and columns, "
        "a larger overlap, or clip_to_well=false"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        plan_fields(
            _plate(),
            wells,
            fov_um=(10.0, 10.0),
            footprint_of=far_away,
            rows=2,
            columns=2,
            overlap=0.0,
            clip_to_well=True,
        )


def test_only_the_wells_no_field_touches_are_named() -> None:
    """A1 is at x = 1000 µm and A2 at 10000: a frame that is only ever near the first."""

    def near_a1_only(x: float, y: float) -> list[tuple[float, float]]:
        shift = 0.0 if x < 5000.0 else 1e6
        return [
            (x + shift, y),
            (x + shift + 10, y),
            (x + shift + 10, y + 10),
            (x + shift, y + 10),
        ]

    message = (
        "no field of the grid touches well A2: give rows and columns, "
        "a larger overlap, or clip_to_well=false"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        plan_fields(
            _plate(),
            ["A1", "A2"],
            fov_um=(10.0, 10.0),
            footprint_of=near_a1_only,
            rows=1,
            columns=1,
            overlap=0.0,
            clip_to_well=True,
        )


def test_a_well_listed_twice_is_refused_not_imaged_twice() -> None:
    message = "wells are listed twice: A1, a1"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _plan(
            _plate(),
            ["A1", "a1"],
            matrix=pixel_matrix(1.0),
            shape=FRAME,
            rows=1,
            columns=1,
        )


def test_a_position_useq_gives_to_another_well_or_without_x_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """useq changed, not the plate: say so instead of planning from what cannot be read."""

    class Position:
        def __init__(self, name: str, x: float | None, y: float | None) -> None:
            self.name, self.x, self.y = name, x, y

    def positions(*items: Position) -> None:
        monkeypatch.setattr(
            useq.WellPlatePlan, "image_positions", property(lambda self: list(items))
        )

    kwargs = {
        "fov_um": (10.0, 10.0),
        "footprint_of": _footprint_of(pixel_matrix(1.0), (10, 10)),
        "rows": 1,
        "columns": 1,
        "overlap": 0.0,
        "clip_to_well": False,
    }

    positions(Position("A2_0000", 0.0, 0.0))
    with pytest.raises(
        ValueError, match="position 'A2_0000' is not of a well of the plan"
    ):
        plan_fields(_plate(), ["A1"], **kwargs)  # type: ignore[arg-type]

    positions(Position("A1_0000", None, 0.0))
    with pytest.raises(ValueError, match="position 'A1_0000' has no x or y"):
        plan_fields(_plate(), ["A1"], **kwargs)  # type: ignore[arg-type]

    positions(Position("A1_0000", 0.0, None))
    with pytest.raises(ValueError, match="position 'A1_0000' has no x or y"):
        plan_fields(_plate(), ["A1"], **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------- intersection tests


def _square(
    x: float, y: float, side: float, *, reverse: bool = False
) -> list[tuple[float, float]]:
    corners = [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]
    return corners[::-1] if reverse else corners


@pytest.mark.parametrize("reverse", [False, True], ids=["one-way", "the-other-way"])
def test_a_disc_meets_a_polygon_by_the_distance_from_its_centre(reverse: bool) -> None:
    polygon = _square(0.0, 0.0, 10.0, reverse=reverse)

    assert _disc_meets((5.0, 5.0), 0.1, polygon)  # the centre is inside
    assert _disc_meets((15.0, 5.0), 5.1, polygon)  # 5 from the nearest edge
    assert not _disc_meets((15.0, 5.0), 4.9, polygon)
    assert not _disc_meets(
        (15.0, 5.0), 5.0, polygon
    )  # touching is below the radius's `<`
    assert _disc_meets((13.0, 14.0), 5.1, polygon)  # 5 from the corner (10, 10)
    assert not _disc_meets((13.0, 14.0), 4.9, polygon)
    # A centre inside is at distance 0 whatever the radius that follows.
    assert _disc_meets((9.9, 0.1), 1e-9, polygon)


@pytest.mark.parametrize("reverse", [False, True], ids=["one-way", "the-other-way"])
def test_two_polygons_overlap_unless_an_edge_normal_separates_them(
    reverse: bool,
) -> None:
    base = _square(0.0, 0.0, 10.0, reverse=reverse)

    assert _polygons_overlap(base, _square(5.0, 5.0, 10.0))  # a corner overlap
    assert _polygons_overlap(base, _square(10.0, 0.0, 10.0))  # touching along an edge
    assert not _polygons_overlap(base, _square(10.5, 0.0, 10.0))
    assert not _polygons_overlap(base, _square(0.0, -10.5, 10.0))
    assert _polygons_overlap(_square(2.0, 2.0, 2.0), base)  # one inside the other

    # Diamonds (a square turned 45 degrees) of half-diagonal 5, whose bounding box
    # overlaps the square's corner at (10, 10) in both cases. Only the diagonal edge
    # normal tells them apart: the diamond at (14, 14) lies beyond x + y = 23 and the
    # square never gets past x + y = 20, the one at (12, 12) reaches back to x + y = 19.
    def diamond(x: float, y: float) -> list[tuple[float, float]]:
        corners = [(x - 5.0, y), (x, y - 5.0), (x + 5.0, y), (x, y + 5.0)]
        return corners[::-1] if reverse else corners

    assert not _polygons_overlap(base, diamond(14.0, 14.0))
    assert _polygons_overlap(base, diamond(12.0, 12.0))


# ---------------------------------------------------------------- estimate


def _timings(**kwargs: float | None) -> TimingsResult:
    values = {
        "move_overhead_ms": 100.0,
        "move_ms_per_mm": 50.0,
        "refocus_ms": 400.0,
        "snap_ms": 20.0,
        "samples": 3,
        **kwargs,
    }
    return TimingsResult(**values)  # type: ignore[arg-type]


def _field(index: int, x: float, y: float) -> PlannedField:
    return PlannedField(well="A1", index=index, name=f"A1_{index:04d}", x_um=x, y_um=y)


def test_the_estimate_adds_moves_refocus_and_snaps() -> None:
    fields = (_field(0, 0.0, 0.0), _field(1, 1000.0, 0.0), _field(2, 1000.0, 2000.0))
    timings = _timings()

    # Moves: none for the first field, 100 + 50 ms for 1 mm, 100 + 100 ms for 2 mm.
    assert estimate_s(fields, timings, refocus=True) == pytest.approx(1.61)
    assert estimate_s(fields, timings, refocus=False) == pytest.approx(0.41)


def test_an_unknown_refocus_cost_is_left_out_not_guessed() -> None:
    fields = (_field(0, 0.0, 0.0), _field(1, 1000.0, 0.0), _field(2, 1000.0, 2000.0))

    assert estimate_s(fields, _timings(refocus_ms=None), refocus=True) == pytest.approx(
        0.41
    )


def test_the_estimate_of_no_fields_is_zero() -> None:
    assert estimate_s((), _timings(), refocus=True) == 0.0


def test_the_estimate_moves_by_euclidean_distance() -> None:
    """3-4-5: a diagonal of 5 mm is one 5000 µm move, not 7000 µm of two."""
    fields = (_field(0, 0.0, 0.0), _field(1, 3000.0, 4000.0))
    timings = _timings(refocus_ms=None, snap_ms=0.0)

    assert estimate_s(fields, timings, refocus=False) == pytest.approx(
        (100.0 + 50.0 * 5.0) / 1000.0
    )


# --------------------------------------------------------------------- gaps


def test_a_rotated_camera_reports_gaps_folded_to_ninety_degrees() -> None:
    assert gap_overlap(2.5, 0.1) is None
    assert gap_overlap(3.5, 0.1) == pytest.approx(0.13)
    assert gap_overlap(90.0, 0.1) is None
    assert gap_overlap(180.0, 0.1) is None
    assert gap_overlap(93.5, 0.1) == pytest.approx(0.13)
    assert gap_overlap(30.0, 0.1) == math.inf


def test_a_suggestion_of_exactly_0_9_is_not_one_the_plugin_could_use() -> None:
    """The parameter refuses 0.9 (`lt=0.9`), so 'use overlap=0.90' would send the operator to an error."""
    assert gap_overlap(26.5, 0.1) == math.inf  # 200 sin(26.5 degrees) = 89.2 -> 0.90
    assert gap_overlap(26.4, 0.1) == pytest.approx(0.89)


def test_gaps_fold_negative_angles_too() -> None:
    assert gap_overlap(-3.5, 0.1) == pytest.approx(0.13)
    assert gap_overlap(-86.5, 0.1) == pytest.approx(0.13)
    assert gap_overlap(-2.5, 0.1) is None


def test_the_suggested_overlap_closes_the_gaps_it_reports() -> None:
    for angle in (3.5, 5.0, 7.3, 12.0):
        suggested = gap_overlap(angle, 0.1)
        assert suggested is not None
        assert suggested >= 2 * math.sin(math.radians(angle))
        assert gap_overlap(angle, suggested) is None


def test_gap_overlap_refuses_an_angle_that_is_not_a_number() -> None:
    with pytest.raises(ValueError, match="rotation_deg must be finite"):
        gap_overlap(float("nan"), 0.1)


@pytest.mark.parametrize(
    ("rotation_deg", "expected"),
    [
        (0.0, 0.0),
        (3.5, 3.5),
        (-3.5, 3.5),
        (93.5, 3.5),
        (-86.5, 3.5),
        (180.0, 0.0),
        (-180.0, 0.0),
        (45.0, 45.0),
        (135.0, 45.0),
        (-135.0, 45.0),
        (30.0, 30.0),
        (60.0, 30.0),
    ],
)
def test_the_turn_from_the_axes_is_the_distance_to_a_multiple_of_ninety(
    rotation_deg: float, expected: float
) -> None:
    assert turn_from_axes_deg(rotation_deg) == pytest.approx(expected)


def test_the_turn_from_the_axes_refuses_an_angle_that_is_not_a_number() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="rotation_deg must be finite"):
            turn_from_axes_deg(bad)
