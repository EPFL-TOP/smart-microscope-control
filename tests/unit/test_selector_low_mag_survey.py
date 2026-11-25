"""``low-mag-survey``: the 40x fields worth scanning, chosen from the 10x frames themselves (#636).

The survey is tested on run folders written by ``select_fields_seed.seed_run``
with frames painted here (dark discs at known stage points, on a camera turned by
178.3 degrees and not square, FM-73); the candidates and the bans on hand-built
surveys over ``make_input``; and the whole selector once through ``route``'s own
checks (FM-124).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pytest

from select_fields_seed import CENTRES, matrix_of, seed_run
from smc.data import load_scan
from smc.plugins.geometry import Matrix2, pixel_of_stage
from smc.plugins.slots import ScanFrame
from smc.selection import SelectionError, SelectionInput, check_choices
from smc.selection.low_mag_survey import (
    Candidate,
    LowMagSurvey,
    LowMagSurveyParams,
    WellSurvey,
    _merged,
    candidates_of,
    frame_scale_um_per_px,
    rank_candidates,
    survey_well,
)
from smc.selection.registry import selector, selector_names
from smc.selection.testing import make_input

PIXEL_UM = 1.124
SCAN_MATRIX = matrix_of(PIXEL_UM, 178.3, False)
#: The 40x of the 10x: a quarter of the pixel, the same camera.
IMAGING_MATRIX = matrix_of(PIXEL_UM / 4.0, 178.3, False)
A1 = CENTRES["A1"]
WELL_RADIUS_UM = 3200.0
LEVEL = 20000.0


def _painter(
    discs: Sequence[tuple[float, float, float]], shape: tuple[int, int], seed: int = 0
):  # type: ignore[no-untyped-def]
    """A ``frame_pixels`` that paints dark discs ``(x_um, y_um, diameter_um)`` at their stage points."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)

    def paint(frame: ScanFrame) -> npt.NDArray[np.uint16]:
        rng = np.random.default_rng(seed + frame.index)
        image = LEVEL * (1.0 + 0.01 * rng.standard_normal(shape))
        for x_um, y_um, diameter_um in discs:
            col, row = pixel_of_stage(
                frame.x_um,
                frame.y_um,
                x_um,
                y_um,
                matrix=SCAN_MATRIX,
                image_shape_px=shape,
            )
            inside = np.hypot(rows - row, cols - col) <= diameter_um / PIXEL_UM / 2.0
            image[inside] *= 0.8
        return np.clip(image, 0, 65535).astype(np.uint16)

    return paint


def _input(
    path: Path,
    *,
    fields_per_well: int = 500,
    wall_margin_um: float = 0.0,
    imaging: Matrix2 = IMAGING_MATRIX,
    shape: tuple[int, int] = (300, 400),
) -> SelectionInput:
    return SelectionInput(
        scan=load_scan(path),
        labels=("single",),
        fields_per_well=fields_per_well,
        wall_margin_um=wall_margin_um,
        exclude_centre_um=0.0,
        objective="40x",
        matrix_um_per_px=imaging,
        image_shape_px=shape,
    )


def _seed(
    tmp_path: Path,
    discs: Sequence[tuple[float, float, float]],
    *,
    shape: tuple[int, int] = (300, 400),
    rows: int = 2,
    columns: int = 2,
    focus_of: dict[tuple[str, int], str] | None = None,
) -> Path:
    seed = seed_run(
        tmp_path,
        cells=[],
        matrix=SCAN_MATRIX,
        shape=shape,
        rows=rows,
        columns=columns,
        focus_of=focus_of,  # type: ignore[arg-type]
        frame_pixels=_painter(discs, shape),
    )
    return seed.path


def _near(survey: WellSurvey, x_um: float, y_um: float, within_um: float) -> int:
    return int(np.sum(np.hypot(survey.x_um - x_um, survey.y_um - y_um) <= within_um))


# --- the survey: frames to objects and tiles on the stage -------------------------


def test_objects_are_placed_on_the_stage_through_the_scans_matrix(
    tmp_path: Path,
) -> None:
    point = (A1[0] + 137.0, A1[1] - 61.0)
    path = _seed(tmp_path, [(*point, 13.0)])
    survey = survey_well(_input(path), "A1", LowMagSurveyParams())
    assert len(survey.x_um) == 1
    assert math.hypot(survey.x_um[0] - point[0], survey.y_um[0] - point[1]) < 1.0
    assert abs(survey.diameter_um[0] - 13.0) < 0.15 * 13.0


def test_an_object_in_the_overlap_of_two_frames_is_counted_once(
    tmp_path: Path,
) -> None:
    # The well's centre is in all four frames of a 2 x 2 plan with 10 % overlap.
    path = _seed(tmp_path, [(A1[0], A1[1], 13.0)])
    data = _input(path)
    frames = data.scan.frames("A1")
    seen_by = [
        f
        for f in frames
        if abs(
            pixel_of_stage(
                f.x_um, f.y_um, *A1, matrix=SCAN_MATRIX, image_shape_px=(300, 400)
            )[0]
            - 199.5
        )
        < 200
    ]
    assert len(seen_by) >= 2
    survey = survey_well(data, "A1", LowMagSurveyParams())
    assert _near(survey, *A1, within_um=5.0) == 1


def test_an_object_outside_the_well_is_not_counted(tmp_path: Path) -> None:
    # Two frames 3600 um wide reach past the wall (radius 3200 um) on each side.
    inside = (A1[0] + 3000.0, A1[1])
    outside = (A1[0] + 3300.0, A1[1])
    path = _seed(
        tmp_path,
        [(*inside, 13.0), (*outside, 13.0)],
        shape=(400, 3200),
        rows=1,
        columns=2,
    )
    survey = survey_well(_input(path, shape=(400, 3200)), "A1", LowMagSurveyParams())
    assert _near(survey, *inside, within_um=5.0) == 1
    assert _near(survey, *outside, within_um=5.0) == 0
    radii = np.hypot(survey.tile_x_um - A1[0], survey.tile_y_um - A1[1])
    assert radii.max() <= WELL_RADIUS_UM


def test_an_unlocked_frame_is_not_read_and_its_fields_lose_coverage(
    tmp_path: Path,
) -> None:
    locked = _seed(tmp_path / "locked", [])
    frames = load_scan(locked).frames("A1")
    lost = frames[0]
    point = (lost.x_um, lost.y_um)
    path = _seed(
        tmp_path / "unlocked",
        [(*point, 13.0)],
        focus_of={("A1", lost.index): "unlocked"},
    )
    data = _input(path)
    params = LowMagSurveyParams(min_cells=0)
    survey = survey_well(data, "A1", params)
    assert _near(survey, *point, within_um=5.0) == 0
    full = survey_well(_input(locked), "A1", params)
    assert len(survey.tile_x_um) < 0.85 * len(full.tile_x_um)
    near_lost = [
        c
        for c in candidates_of(data, "A1", survey, params)
        if math.hypot(c.x_um - point[0], c.y_um - point[1]) < 60.0
    ]
    assert near_lost
    assert all(c.coverage < params.min_coverage for c in near_lost)


def test_a_singular_or_empty_matrix_is_a_selection_error() -> None:
    assert abs(frame_scale_um_per_px(SCAN_MATRIX) - PIXEL_UM) < 1e-9
    for matrix in (
        ((0.0, 0.0), (0.0, 0.0)),
        ((1.0, 2.0), (2.0, 4.0)),
        ((math.nan, 0.0), (0.0, 1.0)),
    ):
        with pytest.raises(SelectionError, match="matrix"):
            frame_scale_um_per_px(matrix)


# --- the candidates and the bans, on hand-built surveys ---------------------------

FIELD_UM = (400.0, 300.0)
#: The bans are tested on the touching lattice, where a field's place is known.
LATTICE = "lattice"


def _hand_input(
    fields_per_well: int = 500, wall_margin_um: float = 0.0
) -> SelectionInput:
    return make_input(
        [],
        wells=("B2",),
        field_um=FIELD_UM,
        fields_per_well=fields_per_well,
        wall_margin_um=wall_margin_um,
    )


def _hand_survey(
    data: SelectionInput,
    objects: Sequence[tuple[float, float, float]] = (),
    *,
    rough: Sequence[tuple[float, float]] = (),
    holes: Sequence[tuple[float, float]] = (),
) -> WellSurvey:
    """A survey of B2: objects ``(dx, dy, diameter)`` from the centre; tiles of 25 um everywhere in the well, texture 1.0 in the fields centred at ``rough``, none read in those at ``holes``."""
    cx, cy = data.scan.geometry["B2"].center_um
    radius = data.scan.geometry["B2"].size_um[0] / 2.0
    step = 25.0
    grid = np.arange(-radius + step / 2, radius, step)
    tx, ty = np.meshgrid(grid, grid)
    tx, ty = tx.ravel(), ty.ravel()
    keep = np.hypot(tx, ty) <= radius
    tx, ty = tx[keep], ty[keep]
    texture = np.zeros(len(tx))
    w, h = FIELD_UM
    for dx, dy in rough:
        texture[(np.abs(tx - dx) <= w / 2) & (np.abs(ty - dy) <= h / 2)] = 1.0
    read = np.ones(len(tx), dtype=bool)
    for dx, dy in holes:
        read &= ~((np.abs(tx - dx) <= w / 2) & (np.abs(ty - dy) <= h / 2))
    ox = np.array([cx + o[0] for o in objects], dtype=np.float64)
    oy = np.array([cy + o[1] for o in objects], dtype=np.float64)
    return WellSurvey(
        well="B2",
        x_um=ox,
        y_um=oy,
        diameter_um=np.array([o[2] for o in objects], dtype=np.float64),
        min_contrast=np.full(len(objects), -0.2),
        tile_x_um=cx + tx[read],
        tile_y_um=cy + ty[read],
        tile_texture=texture[read],
        tile_area_um2=step * step,
    )


def _at(
    candidates: Sequence[Candidate], data: SelectionInput, dx: float, dy: float
) -> Candidate:
    cx, cy = data.scan.geometry["B2"].center_um
    return min(candidates, key=lambda c: math.hypot(c.x_um - cx - dx, c.y_um - cy - dy))


def _chosen_at(
    chosen: Sequence[Candidate], data: SelectionInput, dx: float, dy: float
) -> bool:
    cx, cy = data.scan.geometry["B2"].center_um
    return any(
        abs(c.x_um - cx - dx) < 1e-6 and abs(c.y_um - cy - dy) < 1e-6 for c in chosen
    )


def _rank(
    data: SelectionInput, survey: WellSurvey, params: LowMagSurveyParams
) -> list[Candidate]:
    return rank_candidates(candidates_of(data, "B2", survey, params), params, data)


def test_candidates_are_the_imaging_objectives_footprint_and_never_overlap() -> None:
    data = _hand_input(fields_per_well=1000)
    params = LowMagSurveyParams(min_cells=0, placement=LATTICE)
    candidates = candidates_of(data, "B2", _hand_survey(data), params)
    xs = sorted({round(c.x_um, 6) for c in candidates})
    ys = sorted({round(c.y_um, 6) for c in candidates})
    assert min(np.diff(xs)) == pytest.approx(FIELD_UM[0])
    assert min(np.diff(ys)) == pytest.approx(FIELD_UM[1])
    assert all(data.allowed(c.x_um, c.y_um, "B2") for c in candidates)
    assert len(candidates) > 100
    chosen = [c.choice("B2") for c in rank_candidates(candidates, params, data)]
    check_choices(chosen, data)


def test_a_field_holding_a_large_object_is_banned() -> None:
    data = _hand_input()
    cells = [(0.0, 0.0, 13.0)] * 3 + [(800.0, 0.0, 13.0)] * 3
    survey = _hand_survey(data, [*cells, (10.0, 10.0, 45.0)])
    chosen = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    assert not _chosen_at(chosen, data, 0.0, 0.0)
    assert _chosen_at(chosen, data, 800.0, 0.0)
    assert (
        _at(
            candidates_of(data, "B2", survey, LowMagSurveyParams()), data, 0.0, 0.0
        ).n_large
        == 1
    )


def test_a_field_on_rough_tiles_is_banned() -> None:
    data = _hand_input()
    cells = [(0.0, 0.0, 13.0), (800.0, 0.0, 13.0)]
    survey = _hand_survey(data, cells, rough=[(0.0, 0.0)])
    chosen = _rank(data, survey, LowMagSurveyParams(max_texture=0.5, placement=LATTICE))
    assert not _chosen_at(chosen, data, 0.0, 0.0)
    assert _chosen_at(chosen, data, 800.0, 0.0)
    no_ban = _rank(
        data, survey, LowMagSurveyParams(max_texture=None, placement=LATTICE)
    )
    assert _chosen_at(no_ban, data, 0.0, 0.0)


def test_a_field_with_a_few_rough_tiles_is_not_banned() -> None:
    # One rough tile among the ~190 of a 400 x 300 um field: the clean glass has
    # such tiles (1.6 % of C4's interior on 2026-11-24), and banning on one emptied
    # a fifth of the well.
    data = _hand_input()
    survey = _hand_survey(data, [(0.0, 0.0, 13.0)])
    cx, cy = data.scan.geometry["B2"].center_um
    nearest = int(np.argmin(np.hypot(survey.tile_x_um - cx, survey.tile_y_um - cy)))
    survey.tile_texture[nearest] = 1.0
    params = LowMagSurveyParams(max_texture=0.5, placement=LATTICE)
    candidate = _at(candidates_of(data, "B2", survey, params), data, 0.0, 0.0)
    assert 0.0 < candidate.rough_fraction < params.max_rough_fraction
    assert _chosen_at(_rank(data, survey, params), data, 0.0, 0.0)
    strict = LowMagSurveyParams(
        max_texture=0.5, max_rough_fraction=0.0, placement=LATTICE
    )
    assert not _chosen_at(_rank(data, survey, strict), data, 0.0, 0.0)


def test_a_fully_read_well_covers_every_field_whatever_the_tiles_alignment() -> None:
    # Tiles 72 um wide, shifted by a third of a tile against the fields' lattice: a
    # count of the tile centres inside each field read some fully read fields as
    # 80 % covered, and min_coverage 0.9 banned them (315 of C4's 885).
    data = _hand_input()
    cx, cy = data.scan.geometry["B2"].center_um
    radius = data.scan.geometry["B2"].size_um[0] / 2.0
    step = 72.0
    grid = np.arange(-radius, radius + step, step) + step / 3.0
    tx, ty = (a.ravel() for a in np.meshgrid(grid, grid))
    keep = np.hypot(tx, ty) <= radius
    survey = WellSurvey(
        well="B2",
        x_um=np.zeros(0),
        y_um=np.zeros(0),
        diameter_um=np.zeros(0),
        min_contrast=np.zeros(0),
        tile_x_um=cx + tx[keep],
        tile_y_um=cy + ty[keep],
        tile_texture=np.zeros(int(keep.sum())),
        tile_area_um2=step * step,
    )
    params = LowMagSurveyParams(min_cells=0)
    inner = [
        c
        for c in candidates_of(data, "B2", survey, params)
        if c.wall_distance_um > step
    ]
    assert len(inner) > 50
    assert min(c.coverage for c in inner) >= params.min_coverage


def test_a_field_with_fewer_than_min_cells_is_not_chosen() -> None:
    data = _hand_input()
    survey = _hand_survey(
        data, [(0.0, 0.0, 13.0), (800.0, 0.0, 13.0), (810.0, 10.0, 13.0)]
    )
    chosen = _rank(data, survey, LowMagSurveyParams(min_cells=2, placement=LATTICE))
    assert [
        (round(c.x_um - data.scan.geometry["B2"].center_um[0]), c.n_cells)
        for c in chosen
    ] == [(800, 2)]


def test_a_field_above_max_density_is_not_chosen() -> None:
    data = _hand_input()
    # A 400 x 300 um field is 0.12 mm2: 3 cells are 25 per mm2, 6 are 50.
    survey = _hand_survey(data, [(0.0, 0.0, 13.0)] * 6 + [(800.0, 0.0, 13.0)] * 3)
    chosen = _rank(
        data, survey, LowMagSurveyParams(max_density_per_mm2=40.0, placement=LATTICE)
    )
    assert not _chosen_at(chosen, data, 0.0, 0.0)
    assert _chosen_at(chosen, data, 800.0, 0.0)
    assert _at(chosen, data, 800.0, 0.0).density_per_mm2 == pytest.approx(25.0)


def test_a_field_short_of_min_coverage_is_not_chosen() -> None:
    data = _hand_input()
    survey = _hand_survey(
        data, [(0.0, 0.0, 13.0), (800.0, 0.0, 13.0)], holes=[(0.0, 0.0)]
    )
    chosen = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    assert not _chosen_at(chosen, data, 0.0, 0.0)
    assert _chosen_at(chosen, data, 800.0, 0.0)


def test_fields_are_ranked_by_cells_then_by_distance_from_the_wall() -> None:
    data = _hand_input()
    survey = _hand_survey(
        data,
        [(0.0, 0.0, 13.0)] * 2
        + [(-1600.0, 0.0, 13.0)] * 2
        + [(800.0, 900.0, 13.0)] * 3,
    )
    chosen = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    cx, cy = data.scan.geometry["B2"].center_um
    first = [(round(c.x_um - cx), round(c.y_um - cy)) for c in chosen[:3]]
    # (-1600, 0) comes before (0, 0) in the lattice: only the wall distance puts it after.
    assert first == [(800, 900), (0, 0), (-1600, 0)]
    assert chosen[1].wall_distance_um > chosen[2].wall_distance_um


def test_no_more_than_fields_per_well_are_chosen_and_they_are_the_best() -> None:
    data = _hand_input(fields_per_well=2)
    survey = _hand_survey(
        data,
        [(0.0, 0.0, 13.0)] * 1 + [(800.0, 0.0, 13.0)] * 3 + [(0.0, 900.0, 13.0)] * 2,
    )
    chosen = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    assert [c.n_cells for c in chosen] == [3, 2]


def test_the_cell_class_is_the_diameter_band_and_large_starts_above_it() -> None:
    data = _hand_input()
    survey = _hand_survey(
        data,
        [
            (0.0, 0.0, 5.0),
            (0.0, 0.0, 8.0),
            (0.0, 0.0, 25.0),
            (0.0, 0.0, 27.0),
            (0.0, 0.0, 30.0),
        ],
    )
    candidate = _at(
        candidates_of(data, "B2", survey, LowMagSurveyParams()), data, 0.0, 0.0
    )
    assert candidate.n_cells == 2
    assert candidate.n_large == 1


def test_every_parameter_says_whether_its_default_was_measured_or_chosen() -> None:
    # A number the algorithm decides with is a parameter, never a constant of the
    # module, and its description and schema say what its default rests on.
    schema = LowMagSurveyParams.model_json_schema()["properties"]
    assert len(schema) >= 25
    for name, field in schema.items():
        basis = field.get("basis")
        assert basis in ("measured", "chosen"), name
        prefix = "Chosen, not measured:" if basis == "chosen" else "Measured on "
        assert field["description"].startswith(prefix), name
    measured = {n for n, f in schema.items() if f["basis"] == "measured"}
    assert measured == {"cell_min_diameter_um", "max_texture", "placement"}


@pytest.mark.parametrize("placement", ["lattice", "greedy"])
def test_the_lattice_offset_moves_every_candidate_by_it(placement: str) -> None:
    # 37 um is no multiple of greedy's quarter step (100 um): a candidate at
    # centre + 37 exists only if the offset is applied.
    data = _hand_input()
    survey = _hand_survey(data)
    params = LowMagSurveyParams(min_cells=0, placement=placement)
    moved = params.model_copy(
        update={"lattice_offset_x_um": 37.0, "lattice_offset_y_um": -50.0}
    )
    cx, cy = data.scan.geometry["B2"].center_um

    def at(candidates: Sequence[Candidate], dx: float, dy: float) -> int:
        return sum(
            1
            for c in candidates
            if abs(c.x_um - cx - dx) < 1e-6 and abs(c.y_um - cy - dy) < 1e-6
        )

    plain = candidates_of(data, "B2", survey, params)
    shifted = candidates_of(data, "B2", survey, moved)
    assert at(plain, 0.0, 0.0) == 1
    assert at(shifted, 37.0, -50.0) == 1
    assert at(shifted, 0.0, 0.0) == 0
    assert at(shifted, 37.0, 0.0) == 0
    assert at(shifted, 0.0, -50.0) == 0


def test_greedy_keeps_fields_that_touch_and_stops_at_fields_per_well() -> None:
    # Three clusters 305 um apart in y: the best fields are at 0 and +-300 um, one
    # field (300 um) apart, touching and not overlapping, so greedy keeps all three;
    # the cap drops the fourth. (Exactly 300 um apart, one field would hold two
    # clusters on its edges: route's inside test includes the edge.)
    data = _hand_input(fields_per_well=3)
    cells = (
        [(0.0, 0.0, 13.0)] * 6
        + [(0.0, 305.0, 13.0)] * 5
        + [(0.0, -305.0, 13.0)] * 4
        + [(-1200.0, 0.0, 13.0)] * 3
    )
    chosen = _rank(data, _hand_survey(data, cells), LowMagSurveyParams())
    cx, cy = data.scan.geometry["B2"].center_um
    assert [c.n_cells for c in chosen] == [6, 5, 4]
    assert sorted(round(c.y_um - cy) for c in chosen) == [-300, 0, 300]
    assert all(round(c.x_um - cx) == 0 for c in chosen)
    check_choices([c.choice("B2") for c in chosen], data)


def _with_geometry(data: SelectionInput, **changes: object) -> SelectionInput:
    geometry = dict(data.scan.geometry)
    geometry["B2"] = dataclasses.replace(geometry["B2"], **changes)  # type: ignore[arg-type]
    scan = dataclasses.replace(data.scan, geometry=MappingProxyType(geometry))
    return dataclasses.replace(data, scan=scan)


def test_a_well_that_is_not_circular_is_a_selection_error() -> None:
    data = _with_geometry(_hand_input(), circular=False)
    with pytest.raises(SelectionError, match="not circular"):
        candidates_of(data, "B2", _hand_survey(_hand_input()), LowMagSurveyParams())


def test_a_well_that_is_not_the_scans_is_a_selection_error() -> None:
    data = _hand_input()
    with pytest.raises(SelectionError, match="Z9"):
        candidates_of(data, "Z9", _hand_survey(data), LowMagSurveyParams())


def test_an_imaging_camera_that_cannot_be_inverted_is_a_selection_error() -> None:
    data = _hand_input()
    singular = dataclasses.replace(data, matrix_um_per_px=((4.0, 0.0), (8.0, 0.0)))
    with pytest.raises(SelectionError, match="cannot be inverted"):
        candidates_of(singular, "B2", _hand_survey(data), LowMagSurveyParams())


def test_a_frame_that_cannot_be_read_is_a_selection_error(tmp_path: Path) -> None:
    path = _seed(tmp_path, [(A1[0], A1[1], 13.0)])
    data = _input(path)
    first = data.scan.frames("A1")[0]
    data.scan.image(first)  # readable before
    frame_files = sorted(p for p in tmp_path.rglob("*.tif*") if p.is_file())
    assert frame_files
    for f in frame_files:
        f.unlink()
    with pytest.raises(SelectionError):
        survey_well(_input(path), "A1", LowMagSurveyParams())


def test_two_objects_of_one_frame_closer_than_the_merge_radius_are_both_kept() -> None:
    # Rows: x, y, diameter, contrast, frame, distance to the frame's centre. A and
    # B are two objects of frame 0, 3 um apart; C is frame 1's sighting of A.
    objects = np.array(
        [
            [0.0, 0.0, 9.0, -0.2, 0.0, 1.0],
            [3.0, 0.0, 9.0, -0.2, 0.0, 2.0],
            [0.5, 0.0, 9.0, -0.2, 1.0, 5.0],
        ]
    )
    kept = _merged(objects, 5.0)
    assert sorted(kept[:, 0].tolist()) == [0.0, 3.0]


def test_the_selector_is_registered_by_name() -> None:
    assert "low-mag-survey" in selector_names()
    assert isinstance(selector("low-mag-survey"), LowMagSurvey)


def test_greedy_placement_centres_a_field_on_a_cluster_the_lattice_cuts() -> None:
    # Four cells astride the lattice's edge at x = +200 um: the lattice splits them
    # two and two, a field moved by half a step holds all four.
    data = _hand_input()
    cluster = [(x, 0.0, 13.0) for x in (185.0, 195.0, 205.0, 215.0)]
    survey = _hand_survey(data, cluster)
    lattice = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    greedy = _rank(data, survey, LowMagSurveyParams())
    assert [c.n_cells for c in lattice] == [2, 2]
    assert [c.n_cells for c in greedy] == [4]


def test_greedy_fields_never_overlap_and_pass_routes_checks() -> None:
    data = _hand_input(fields_per_well=500)
    rng = np.random.default_rng(3)
    cells = [(dx, dy, 13.0) for dx, dy in rng.uniform(-2000, 2000, size=(400, 2))]
    survey = _hand_survey(data, cells)
    params = LowMagSurveyParams()
    chosen = _rank(data, survey, params)
    assert len(chosen) > 20
    for a, c in enumerate(chosen):
        for b in chosen[a + 1 :]:
            assert (
                abs(b.x_um - c.x_um) >= FIELD_UM[0] - 1e-6
                or abs(b.y_um - c.y_um) >= FIELD_UM[1] - 1e-6
            )
    check_choices([c.choice("B2") for c in chosen], data)


def test_greedy_takes_more_cells_in_its_first_fields_than_the_lattice() -> None:
    data = _hand_input(fields_per_well=5)
    rng = np.random.default_rng(5)
    clusters = rng.uniform(-1500, 1500, size=(12, 2))
    cells = [
        (cx + dx, cy + dy, 13.0)
        for cx, cy in clusters
        for dx, dy in rng.normal(0.0, 60.0, size=(8, 2))
    ]
    survey = _hand_survey(data, cells)
    lattice = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    greedy = _rank(data, survey, LowMagSurveyParams())
    assert len(greedy) == len(lattice) == 5
    assert sum(c.n_cells for c in greedy) > sum(c.n_cells for c in lattice)


def test_greedy_candidates_are_the_lattice_shifted_by_each_fraction_of_a_step() -> None:
    data = _hand_input()
    survey = _hand_survey(data)
    cx, cy = data.scan.geometry["B2"].center_um
    candidates = candidates_of(
        data, "B2", survey, LowMagSurveyParams(min_cells=0, greedy_subdivisions=4)
    )
    phases = {
        (round((c.x_um - cx) % FIELD_UM[0], 6), round((c.y_um - cy) % FIELD_UM[1], 6))
        for c in candidates
    }
    assert phases == {
        (x, y) for x in (0.0, 100.0, 200.0, 300.0) for y in (0.0, 75.0, 150.0, 225.0)
    }


def test_greedy_on_one_subdivision_is_the_lattice() -> None:
    data = _hand_input()
    rng = np.random.default_rng(9)
    cells = [(dx, dy, 13.0) for dx, dy in rng.uniform(-2000, 2000, size=(200, 2))]
    survey = _hand_survey(data, cells)
    lattice = _rank(data, survey, LowMagSurveyParams(placement=LATTICE))
    one = _rank(data, survey, LowMagSurveyParams(greedy_subdivisions=1))
    assert one == lattice


def test_params_refuse_a_cell_band_that_is_empty_or_overlaps_large() -> None:
    with pytest.raises(ValueError, match="cell_min_diameter_um"):
        LowMagSurveyParams(cell_min_diameter_um=20.0, cell_max_diameter_um=10.0)
    with pytest.raises(ValueError, match="large_min_diameter_um"):
        LowMagSurveyParams(cell_max_diameter_um=30.0, large_min_diameter_um=25.0)


# --- the whole selector ------------------------------------------------------------


def _seeded_selection(tmp_path: Path) -> tuple[SelectionInput, list]:  # type: ignore[type-arg]
    rng = np.random.default_rng(7)
    discs = [
        (A1[0] + dx, A1[1] + dy, 13.0)
        for dx, dy in rng.uniform(-300, 300, size=(25, 2))
    ]
    path = _seed(tmp_path, discs)
    data = _input(path, fields_per_well=8)
    return data, list(LowMagSurvey().select(data, LowMagSurveyParams()))


def test_the_choices_pass_routes_checks_on_a_seeded_run(tmp_path: Path) -> None:
    data, chosen = _seeded_selection(tmp_path)
    assert 1 <= len(chosen) <= 8
    check_choices(chosen, data)
    assert all(choice.well == "A1" for choice in chosen)


def test_measures_carry_their_units_and_are_finite(tmp_path: Path) -> None:
    _, chosen = _seeded_selection(tmp_path)
    expected = {
        "n_cells",
        "density_per_mm2",
        "n_large",
        "rough_fraction",
        "coverage",
        "wall_distance_um",
    }
    for choice in chosen:
        assert set(choice.measures) == expected
        assert all(math.isfinite(v) for v in choice.measures.values())
        assert choice.score == choice.measures["n_cells"]


def test_the_same_input_gives_the_same_choices(tmp_path: Path) -> None:
    data, chosen = _seeded_selection(tmp_path)
    again = list(LowMagSurvey().select(data, LowMagSurveyParams()))
    assert again == chosen
