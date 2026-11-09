"""scripts/dev/select_fields.py scores the scan's own frames as candidate fields (#343, m3 §11).

The script is a prototype for the owner to tune on real 40x scans: these tests
pin what it computes, not what a good field is. The camera is 60 x 80 pixels (not
square, FM-73), turned or mirrored where a test says so; every expected value is
built here from ``stage_of_pixel``, ``footprint`` and the seed, never from the
function under test (FM-124).
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial import cKDTree

from select_fields_seed import (
    CENTRES,
    CLASSES,
    PIXEL_UM,
    SHAPE,
    UPRIGHT,
    Seed,
    cell_at,
    load_script,
    matrix_of,
    seed_run,
)
from select_fields_seed import record as write_record
from smc.plugins.geometry import Matrix2, footprint, stage_of_pixel
from smc.plugins.route.algorithm import allowed
from smc.plugins.slots import (
    Cell,
    RouteResult,
    ScanFrame,
    ScanResult,
    WellGeometry,
    load_cells,
    load_frame,
    well_geometry,
)
from smc.run import RunDocument, RunDocumentError, StageStatus

HEIGHT, WIDTH = SHAPE


@pytest.fixture(scope="module")
def sf() -> ModuleType:
    """The script, loaded as ``tests/test_dev_point.py`` loads ``point.py``."""
    return load_script()


# --------------------------------------------------------------------- helpers

MATRICES: dict[str, Matrix2] = {
    "upright": UPRIGHT,
    "mirrored": matrix_of(PIXEL_UM, 0.0, True),
    "turned 90": matrix_of(PIXEL_UM, 90.0, False),
    "turned 30": matrix_of(PIXEL_UM, 30.0, False),
}


def _stage_at(
    matrix: Matrix2, x_um: float, y_um: float, d_col: float, d_row: float
) -> tuple[float, float]:
    """The stage point ``(d_col, d_row)`` pixels from the middle of a frame at ``(x_um, y_um)``, by the code of geometry."""
    centre_col, centre_row = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
    return stage_of_pixel(
        x_um,
        y_um,
        centre_col + d_col,
        centre_row + d_row,
        matrix=matrix,
        image_shape_px=SHAPE,
    )


def _geometry(seed: Seed, well: str = "A1") -> WellGeometry:
    return well_geometry(seed.plan.plate_plan, well)


def _candidates(
    sf: ModuleType,
    seed: Seed,
    *,
    well: str = "A1",
    labels: Sequence[str] = ("single",),
    min_spacing_um: float = 30.0,
    margin_um: float = 0.0,
    exclude_centre_um: float = 0.0,
    stats: dict[int, tuple[float | None, float] | None] | None = None,
) -> tuple[tuple, object]:
    cells = load_cells(seed.path.parent, seed.result)
    return sf.candidates_of(
        well,
        seed.scan,
        cells,
        _geometry(seed, well),
        labels=labels,
        classes=CLASSES,
        min_spacing_um=min_spacing_um,
        margin_um=margin_um,
        exclude_centre_um=exclude_centre_um,
        stats=stats if stats is not None else {},
    )


def _one_field(tmp_path: Path, cells) -> Seed:
    """One frame at the well's centre (readback ``(1000.3, -500.2)``), the cells as the caller places them."""
    return seed_run(tmp_path, cells=cells, rows=1, columns=1)


def test_the_seed_is_a_run_document_every_stage_can_read(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[cell_at("A1", 0.0, 0.0)], wells=("A1", "A2"))
    document = RunDocument.load(seed.path)
    for slot in ("plate", "scan-plan", "scan", "cells"):
        assert document.unusable(slot, profile="demo", dry_run=False) is None
    assert len(seed.scan.frames) == 8
    for frame in seed.scan.frames:
        assert load_frame(tmp_path, frame).shape == SHAPE
    assert load_cells(tmp_path, seed.result) == seed.cells


# ------------------------------------------------------------------ features


@pytest.mark.parametrize("name", list(MATRICES))
def test_a_cell_is_in_a_field_when_it_is_inside_the_cameras_footprint(
    sf: ModuleType, name: str
) -> None:
    matrix = MATRICES[name]
    x0, y0 = 1000.0, -500.0
    inside = [
        (WIDTH / 2 - 1, 0.0),
        (-(WIDTH / 2 - 1), 0.0),
        (0.0, HEIGHT / 2 - 1),
        (0.0, -(HEIGHT / 2 - 1)),
        # Not on an axis: 35 columns is inside a 40-column half-width and outside a
        # 30-row half-height, so a width and a height that are exchanged show here.
        (35.0, 0.0),
        (0.0, 25.0),
    ]
    outside = [
        (WIDTH / 2 + 1, 0.0),
        (-(WIDTH / 2 + 1), 0.0),
        (0.0, HEIGHT / 2 + 1),
        (0.0, -(HEIGHT / 2 + 1)),
        (0.0, 35.0),
        (36.0, 31.0),
    ]
    points = [_stage_at(matrix, x0, y0, c, r) for c, r in inside + outside]
    xy = np.array(points, dtype=np.float64)
    tree = cKDTree(xy)
    found = sf.cells_in_field(xy, tree, x0, y0, matrix=matrix, image_shape_px=SHAPE)
    assert sorted(found.tolist()) == list(range(len(inside)))


def test_nearest_neighbour_distance_looks_at_every_cell_of_the_well_not_only_the_field(
    tmp_path: Path, sf: ModuleType
) -> None:
    # A debris cell 10 um from the wanted cell and just outside the field's edge: the
    # wanted cell is not isolated at a spacing of 30 um, and is at 5.
    inner = 195.0

    def cells(scan):
        frame = scan.frames[0]
        return [
            Cell(
                well="A1",
                x_um=frame.x_um + inner,
                y_um=frame.y_um,
                label="single",
                score=0.9,
                frame=0,
                box_px=(0.0, 0.0, 10.0, 10.0),
            ),
            Cell(
                well="A1",
                x_um=frame.x_um + inner + 10.0,
                y_um=frame.y_um,
                label="debris",
                score=0.9,
                frame=0,
                box_px=(0.0, 0.0, 10.0, 10.0),
            ),
        ]

    seed = _one_field(tmp_path, cells)
    wide, _ = _candidates(sf, seed, min_spacing_um=30.0)
    narrow, _ = _candidates(sf, seed, min_spacing_um=5.0)
    assert (wide[0].n_wanted, wide[0].isolated) == (1, 0)
    assert (narrow[0].n_wanted, narrow[0].isolated) == (1, 1)
    assert wide[0].nn_median_um == pytest.approx(10.0)
    assert wide[0].counts == {"single": 1, "doublet": 0, "debris": 0}
    assert wide[0].n_all == 1  # the debris cell is outside the field: 195 + 10 > 200


def test_a_cell_alone_in_its_well_is_isolated_and_has_no_distance(
    tmp_path: Path, sf: ModuleType
) -> None:
    assert sf.neighbour_distances_um(np.array([[1.0, 2.0]])).tolist() == [math.inf]
    seed = _one_field(tmp_path, [cell_at("A1", 3.0, 4.0)])
    (candidate,), _ = _candidates(sf, seed, min_spacing_um=1e5)
    assert (candidate.n_wanted, candidate.isolated) == (1, 1)
    assert candidate.nn_median_um is None
    assert candidate.nn_min_um is None


def test_two_cells_at_one_position_are_not_isolated(
    tmp_path: Path, sf: ModuleType
) -> None:
    assert sf.neighbour_distances_um(np.array([[1.0, 2.0], [1.0, 2.0]])).tolist() == [
        0.0,
        0.0,
    ]
    seed = _one_field(tmp_path, [cell_at("A1", 3.0, 4.0), cell_at("A1", 3.0, 4.0)])
    (candidate,), _ = _candidates(sf, seed, min_spacing_um=1e-3)
    assert (candidate.n_wanted, candidate.isolated) == (2, 0)
    assert candidate.nn_median_um == 0.0
    assert candidate.nn_min_um == 0.0


@pytest.mark.parametrize("circular", [True, False], ids=["round", "square"])
@pytest.mark.parametrize("rotation_deg", [0.0, 17.0])
@pytest.mark.parametrize("matrix_name", ["upright", "turned 30"])
def test_wall_distance_agrees_with_contains(
    sf: ModuleType, circular: bool, rotation_deg: float, matrix_name: str
) -> None:
    well = WellGeometry(
        well="A1",
        center_um=(1000.0, -500.0),
        size_um=(6400.0, 6400.0) if circular else (6400.0, 4400.0),
        circular=circular,
        rotation_deg=rotation_deg,
    )
    matrix = MATRICES[matrix_name]
    checked = 0
    for dx in range(-3000, 3001, 500):
        for dy in range(-2000, 2001, 400):
            corners = footprint(
                1000.0 + dx + 0.37,
                -500.0 + dy - 0.21,
                matrix=matrix,
                image_shape_px=SHAPE,
            )
            d = sf.wall_distance_um(well, corners)
            if d < 10.0:
                continue
            checked += 1
            # 1e-6 um is far above the ulp-sized disagreement at 1e4 um (FM-132).
            assert all(well.contains(x, y, margin_um=d - 1e-6) for x, y in corners)
            assert not all(well.contains(x, y, margin_um=d + 1e-6) for x, y in corners)
    assert checked >= 20


def test_wall_distance_is_negative_when_a_corner_is_outside(sf: ModuleType) -> None:
    round_well = WellGeometry(
        well="A1",
        center_um=(0.0, 0.0),
        size_um=(1000.0, 1000.0),
        circular=True,
        rotation_deg=0.0,
    )
    corners = footprint(480.0, 0.0, matrix=UPRIGHT, image_shape_px=SHAPE)
    assert sf.wall_distance_um(round_well, corners) < 0.0


@pytest.mark.parametrize("matrix_name", ["upright", "turned 30"])
def test_the_default_min_spacing_is_three_median_cell_lengths(
    sf: ModuleType, matrix_name: str
) -> None:
    matrix = MATRICES[matrix_name]
    # Longest sides of 10, 20 and 30 pixels (the side is the larger of the two).
    boxes = [(0.0, 0.0, 10.0, 4.0), (5.0, 5.0, 8.0, 25.0), (0.0, 0.0, 30.0, 12.0)]
    cells = [cell_at("A1", 0.0, float(i), box_px=box) for i, box in enumerate(boxes)]
    assert sf.default_min_spacing_um(cells, matrix) == pytest.approx(
        3.0 * 20.0 * PIXEL_UM
    )
    # An anisotropic camera: the pixel is the geometric mean, sqrt(|det|) = sqrt(24).
    anisotropic: Matrix2 = ((6.0, 0.0), (0.0, -4.0))
    assert sf.default_min_spacing_um(cells, anisotropic) == pytest.approx(
        3.0 * 20.0 * math.sqrt(24.0)
    )
    assert sf.default_min_spacing_um([], matrix) is None


def test_the_laplacian_does_not_wrap_on_unsigned_frames(sf: ModuleType) -> None:
    rows, cols = np.indices((8, 8))
    frame = np.where((rows + cols) % 2 == 0, 0, 1000).astype(np.uint16)
    sharp, background = sf.frame_stats(frame)
    # Every interior pixel's Laplacian is +-4000: the variance is 4000^2, over the
    # median (500) squared, which a uint16 Laplacian (wrapping) would not give.
    assert background == 500.0
    assert sharp == pytest.approx(4000.0**2 / 500.0**2)
    assert sf.frame_stats(frame.astype(np.float64)) == (sharp, background)


def test_sharpness_does_not_depend_on_the_exposure(sf: ModuleType) -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(100, 4000, size=(40, 50)).astype(np.uint16)
    sharp, _ = sf.frame_stats(frame)
    brighter, _ = sf.frame_stats(frame * np.uint16(16))
    assert sharp is not None
    assert brighter is not None
    assert brighter == pytest.approx(sharp, rel=1e-9)


def test_a_blurred_frame_is_less_sharp(sf: ModuleType) -> None:
    rng = np.random.default_rng(1)
    frame = rng.integers(100, 4000, size=(40, 50)).astype(np.float64)
    padded = np.pad(frame, 1, mode="edge")
    blurred = (
        sum(padded[i : i + 40, j : j + 50] for i in range(3) for j in range(3)) / 9.0
    )
    sharp, _ = sf.frame_stats(frame)
    soft, _ = sf.frame_stats(blurred)
    assert sharp is not None
    assert soft is not None
    assert soft < sharp


def test_a_frame_of_zero_median_has_no_sharpness(sf: ModuleType) -> None:
    frame = np.zeros((10, 10), dtype=np.uint16)
    frame[4, 4] = 500
    assert sf.frame_stats(frame) == (None, 0.0)


def test_a_frame_that_is_not_two_dimensional_or_not_finite_is_refused(
    sf: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sf.frame_stats(np.zeros((4, 4, 3), dtype=np.uint16))
    bad = np.ones((6, 6), dtype=np.float32)
    bad[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sf.frame_stats(bad)


# ---------------------------------------------------------------- candidates


def test_an_unlocked_frame_is_never_a_candidate(tmp_path: Path, sf: ModuleType) -> None:
    seed = seed_run(
        tmp_path,
        cells=[cell_at("A1", 0.0, 0.0)],
        rows=2,
        columns=2,
        focus_of={("A1", 1): "unlocked"},
    )
    found, skipped = _candidates(sf, seed)
    assert [c.frame for c in found] == [0, 2, 3]
    assert skipped.unlocked == 1
    assert skipped.refused == 0


def test_off_and_none_frames_are_candidates_with_no_offset(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = seed_run(
        tmp_path,
        cells=[cell_at("A1", 0.0, 0.0)],
        focus_of={("A1", 0): "off", ("A1", 1): "none"},
    )
    found, _ = _candidates(sf, seed)
    by_frame = {c.frame: c for c in found}
    assert [c.frame for c in found] == [0, 1, 2, 3]
    assert by_frame[0].pfs_offset is None
    assert by_frame[1].pfs_offset is None
    assert by_frame[0].z_um == seed.scan.frames[0].z_um
    assert by_frame[2].pfs_offset == seed.scan.frames[2].pfs_offset
    assert (by_frame[0].focus, by_frame[1].focus, by_frame[2].focus) == (
        "off",
        "none",
        "locked",
    )


def test_a_footprint_over_the_wall_is_not_a_candidate(
    tmp_path: Path, sf: ModuleType
) -> None:
    # 3 x 3 fields round the centre of a 3200 um well: the centre field's farthest
    # corner is 250 um from the well's centre, the others' 580 or 700, so a margin of
    # 2800 um leaves only the centre one.
    seed = seed_run(tmp_path, cells=[cell_at("A1", 0.0, 0.0)], rows=3, columns=3)
    found, skipped = _candidates(sf, seed, margin_um=2800.0)
    well = _geometry(seed)
    expected = [
        k
        for k, frame in enumerate(seed.scan.frames)
        if all(
            well.contains(x, y, margin_um=2800.0)
            for x, y in footprint(
                frame.x_um, frame.y_um, matrix=UPRIGHT, image_shape_px=SHAPE
            )
        )
    ]
    assert expected == [4]
    assert [c.frame for c in found] == expected
    assert skipped.refused == 8


def test_a_footprint_over_the_excluded_centre_is_not_a_candidate(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = seed_run(tmp_path, cells=[cell_at("A1", 0.0, 0.0)], rows=3, columns=3)
    found, skipped = _candidates(sf, seed, exclude_centre_um=100.0)
    frame = seed.scan.frames[4]
    assert not allowed(
        frame.x_um,
        frame.y_um,
        _geometry(seed),
        matrix=UPRIGHT,
        image_shape_px=SHAPE,
        exclude_centre_um=100.0,
        wall_margin_um=0.0,
    )
    assert [c.frame for c in found] == [0, 1, 2, 3, 5, 6, 7, 8]
    assert skipped.refused == 1


def test_a_candidate_has_the_features_of_its_own_frame(
    tmp_path: Path, sf: ModuleType
) -> None:
    # Eight single cells and one doublet inside the field of frame 0: density per mm2
    # is the wanted count over the field's area (400 x 300 um = 0.12 mm2).
    def cells(scan):
        frame = scan.frames[0]
        singles = [
            Cell(
                well="A1",
                x_um=frame.x_um - 100.0 + 25.0 * i,
                y_um=frame.y_um + 10.0,
                label="single",
                score=0.9,
                frame=0,
                box_px=(0.0, 0.0, 10.0, 10.0),
            )
            for i in range(8)
        ]
        doublet = Cell(
            well="A1",
            x_um=frame.x_um,
            y_um=frame.y_um - 100.0,
            label="doublet",
            score=0.9,
            frame=0,
            box_px=(0.0, 0.0, 10.0, 10.0),
        )
        return [*singles, doublet]

    seed = seed_run(tmp_path, cells=cells, rows=1, columns=1)
    (candidate,), _ = _candidates(
        sf, seed, stats={0: (0.25, 1234.5)}, min_spacing_um=10.0
    )
    assert candidate.name == seed.scan.frames[0].name
    assert (candidate.x_um, candidate.y_um) == (
        seed.scan.frames[0].x_um,
        seed.scan.frames[0].y_um,
    )
    assert candidate.counts == {"single": 8, "doublet": 1, "debris": 0}
    assert candidate.n_all == 9
    assert candidate.n_wanted == 8
    assert candidate.density_per_mm2 == pytest.approx(8 / 0.12)
    assert candidate.isolated == 8  # 25 um apart, and the doublet is 110 um away
    assert candidate.nn_median_um == pytest.approx(25.0)
    assert (candidate.sharpness, candidate.background) == (0.25, 1234.5)
    # The frame is read back at (1000.3, -500.2), the well's centre is (1000, -500): the
    # farthest corner is 200.3 um across and 150.2 um up or down from it.
    assert candidate.wall_um == pytest.approx(3200.0 - math.hypot(200.3, 150.2))
    assert candidate.focus == "locked"


def test_a_frame_that_could_not_be_read_has_no_pixel_features(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _one_field(tmp_path, [cell_at("A1", 0.0, 0.0)])
    (unread,), _ = _candidates(sf, seed, stats={0: None})
    (unasked,), _ = _candidates(sf, seed, stats={})
    assert (unread.sharpness, unread.background) == (None, None)
    assert (unasked.sharpness, unasked.background) == (None, None)


# --------------------------------------------------------------------- score


def _cand(
    sf: ModuleType,
    frame: int,
    *,
    n_wanted: int = 1,
    n_all: int | None = None,
    isolated: int = 0,
    wall_um: float = 1000.0,
    sharpness: float | None = None,
    background: float | None = None,
    at: tuple[float, float] | None = None,
):
    """A candidate with the features a test gives it; the density is the wanted count over a 0.12 mm2 field.

    ``at`` is where its frame was read back (``(frame, 0)`` when a test does not care).
    """
    x_um, y_um = at if at is not None else (float(frame), 0.0)
    return sf.Candidate(
        frame=frame,
        name=f"A1_{frame:04d}",
        x_um=x_um,
        y_um=y_um,
        z_um=None,
        pfs_offset=None,
        focus="locked",
        counts={"single": n_wanted},
        n_all=n_wanted if n_all is None else n_all,
        n_wanted=n_wanted,
        density_per_mm2=n_wanted / 0.12,
        isolated=isolated,
        nn_median_um=None,
        nn_min_um=None,
        wall_um=wall_um,
        sharpness=sharpness,
        background=background,
    )


def _weights(sf: ModuleType, **given: float) -> dict[str, float]:
    """All six weights at 0 but the ones given."""
    return {name: 0.0 for name in sf.WEIGHT_NAMES} | given


def _ranked(scored) -> list[int]:
    """The frames by score, best first, ties to the lowest frame."""
    return [
        s.candidate.frame
        for s in sorted(scored, key=lambda s: (-s.score, s.candidate.frame))
    ]


def test_default_weights_rank_by_the_number_of_cells(sf: ModuleType) -> None:
    # What route prefers: the field with the most wanted cells, ties to the lowest frame.
    counts = [3, 7, 7, 0, 5]
    candidates = [_cand(sf, k, n_wanted=n) for k, n in enumerate(counts)]
    scored = sf.score_candidates(candidates, sf.parse_weights([]))
    assert _ranked(scored) == [1, 2, 4, 0, 3]


def test_the_defaults_are_the_density_alone_and_a_given_weight_replaces_only_its_own(
    sf: ModuleType,
) -> None:
    assert sf.parse_weights([]) == _weights(sf, density=1.0)
    assert sf.parse_weights(["isolation=2.5", "wall=0"]) == _weights(
        sf, density=1.0, isolation=2.5
    )
    assert sf.parse_weights(["density=0", "purity=3"]) == _weights(sf, purity=3.0)


def test_a_weight_of_zero_removes_a_feature(sf: ModuleType) -> None:
    few = [_cand(sf, 0, n_wanted=4, isolated=0), _cand(sf, 1, n_wanted=4, isolated=0)]
    many = [_cand(sf, 0, n_wanted=4, isolated=0), _cand(sf, 1, n_wanted=4, isolated=4)]
    weights = _weights(sf, density=1.0)
    assert [s.score for s in sf.score_candidates(few, weights)] == [
        s.score for s in sf.score_candidates(many, weights)
    ]
    with_isolation = _weights(sf, density=1.0, isolation=1.0)
    scores = [s.score for s in sf.score_candidates(many, with_isolation)]
    assert scores[1] > scores[0]


def test_weights_are_normalised_so_that_the_score_stays_in_zero_one(
    sf: ModuleType,
) -> None:
    candidates = [
        _cand(
            sf,
            0,
            n_wanted=2,
            n_all=4,
            isolated=1,
            wall_um=500.0,
            sharpness=0.1,
            background=900.0,
        ),
        _cand(
            sf,
            1,
            n_wanted=9,
            n_all=9,
            isolated=9,
            wall_um=3000.0,
            sharpness=0.9,
            background=100.0,
        ),
        _cand(
            sf, 2, n_wanted=0, n_all=0, wall_um=10.0, sharpness=0.5, background=500.0
        ),
    ]
    base = {
        "density": 1.0,
        "isolation": 2.0,
        "purity": 0.5,
        "wall": 1.0,
        "sharpness": 1.5,
        "background": 0.25,
    }
    scored = sf.score_candidates(candidates, base)
    tenfold = sf.score_candidates(candidates, {k: 10.0 * v for k, v in base.items()})
    assert all(0.0 <= s.score <= 1.0 for s in scored)
    assert [s.score for s in tenfold] == pytest.approx([s.score for s in scored])
    # Field 1 is the best on the four counted parts (1 each) and on the two ranks the
    # best of three (5/6 each; the darkest background is 1 - 1/6).
    expected = (1.0 + 2.0 + 0.5 + 1.0 + (1.5 + 0.25) * (5.0 / 6.0)) / 6.25
    assert scored[1].score == pytest.approx(expected)
    assert all(0.0 <= p <= 1.0 for s in scored for p in s.parts.values())
    assert set(scored[0].parts) == set(sf.WEIGHT_NAMES)
    # Two wanted cells of four, one of them isolated: half of the wanted ones, half pure.
    assert scored[0].parts["isolation"] == 0.5
    assert scored[0].parts["purity"] == 0.5


def test_the_isolation_weight_prefers_the_field_of_isolated_cells(
    tmp_path: Path, sf: ModuleType
) -> None:
    # Field A (frame 0) holds 6 cells within 8 um of each other, field B (frame 1) holds 4
    # cells 80 um apart. Density alone chooses A; isolation=5 chooses B.
    def cells(scan):
        a, b = scan.frames[0], scan.frames[1]
        crowd = [(a.x_um + 2.0 * i, a.y_um) for i in range(6)]
        spread = [(b.x_um - 120.0 + 80.0 * i, b.y_um + 20.0) for i in range(4)]
        return [
            Cell(
                well="A1",
                x_um=x,
                y_um=y,
                label="single",
                score=0.9,
                frame=0,
                box_px=(0.0, 0.0, 10.0, 10.0),
            )
            for x, y in crowd + spread
        ]

    seed = seed_run(tmp_path, cells=cells, rows=1, columns=2, overlap=0.0)
    found, _ = _candidates(sf, seed, min_spacing_um=30.0)
    assert [(c.n_wanted, c.isolated) for c in found] == [(6, 0), (4, 4)]
    density_only = sf.score_candidates(found, sf.parse_weights([]))
    isolation = sf.score_candidates(found, sf.parse_weights(["isolation=5"]))
    assert _ranked(density_only) == [0, 1]
    assert _ranked(isolation) == [1, 0]


def test_the_purity_weight_penalises_doublets_and_debris(sf: ModuleType) -> None:
    clean = _cand(sf, 0, n_wanted=5, n_all=5)
    dirty = _cand(sf, 1, n_wanted=5, n_all=10)
    scored = sf.score_candidates([clean, dirty], _weights(sf, purity=1.0))
    assert [s.parts["purity"] for s in scored] == [1.0, 0.5]
    assert _ranked(scored) == [0, 1]
    flat = sf.score_candidates([clean, dirty], _weights(sf, density=1.0))
    assert flat[0].score == flat[1].score
    nothing = sf.score_candidates(
        [_cand(sf, 2, n_wanted=0, n_all=0)], _weights(sf, purity=1.0)
    )
    assert nothing[0].parts["purity"] == 0.0


def test_the_wall_weight_prefers_the_centre(sf: ModuleType) -> None:
    edge = _cand(sf, 0, wall_um=100.0)
    middle = _cand(sf, 1, wall_um=2500.0)
    scored = sf.score_candidates([edge, middle], _weights(sf, wall=1.0))
    assert [s.parts["wall"] for s in scored] == pytest.approx([0.04, 1.0])
    assert _ranked(scored) == [1, 0]


def test_sharpness_and_background_are_ranks_within_the_well(sf: ModuleType) -> None:
    candidates = [
        _cand(sf, k, sharpness=value, background=value)
        for k, value in enumerate([1.0, 2.0, 3.0])
    ]
    sharp = sf.score_candidates(candidates, _weights(sf, sharpness=1.0))
    dark = sf.score_candidates(candidates, _weights(sf, background=1.0))
    assert [s.parts["sharpness"] for s in sharp] == pytest.approx([1 / 6, 3 / 6, 5 / 6])
    assert [s.parts["background"] for s in dark] == pytest.approx([5 / 6, 3 / 6, 1 / 6])
    tied = [_cand(sf, k, sharpness=2.0, background=2.0) for k in range(3)]
    assert [
        s.parts["sharpness"]
        for s in sf.score_candidates(tied, _weights(sf, sharpness=1.0))
    ] == [0.5] * 3
    assert [
        s.parts["background"]
        for s in sf.score_candidates(tied, _weights(sf, background=1.0))
    ] == [0.5] * 3


def test_percentile_rank_counts_the_values_below_and_half_the_ties(
    sf: ModuleType,
) -> None:
    values = [10.0, 20.0, 20.0, 40.0]
    assert sf.percentile_rank(5.0, values) == 0.0
    assert sf.percentile_rank(10.0, values) == pytest.approx(0.5 / 4)
    assert sf.percentile_rank(20.0, values) == pytest.approx((1 + 1.0) / 4)
    assert sf.percentile_rank(40.0, values) == pytest.approx(3.5 / 4)
    assert sf.percentile_rank(99.0, values) == 1.0


def test_an_unreadable_frame_drops_the_candidate_only_when_a_pixel_weight_is_set(
    sf: ModuleType,
) -> None:
    candidates = [
        _cand(sf, 0, n_wanted=3, sharpness=0.5, background=100.0),
        _cand(sf, 1, n_wanted=9),  # not read: no pixel features
        _cand(sf, 2, n_wanted=2, sharpness=0.2, background=200.0),
    ]
    assert [
        s.candidate.frame
        for s in sf.score_candidates(candidates, _weights(sf, density=1.0))
    ] == [0, 1, 2]
    for name in ("sharpness", "background"):
        scored = sf.score_candidates(
            candidates, _weights(sf, density=1.0, **{name: 1.0})
        )
        assert [s.candidate.frame for s in scored] == [0, 2]
    # The pool the ranks are taken over is the one that was kept.
    kept = sf.score_candidates(candidates, _weights(sf, sharpness=1.0))
    assert [s.parts["sharpness"] for s in kept] == [0.75, 0.25]


def test_the_density_part_is_over_the_pools_largest_and_is_zero_for_an_empty_pool(
    sf: ModuleType,
) -> None:
    scored = sf.score_candidates(
        [_cand(sf, 0, n_wanted=2), _cand(sf, 1, n_wanted=8)], _weights(sf, density=1.0)
    )
    assert [s.parts["density"] for s in scored] == pytest.approx([0.25, 1.0])
    empty = sf.score_candidates([_cand(sf, 0, n_wanted=0)], _weights(sf, density=1.0))
    assert empty[0].parts["density"] == 0.0
    assert empty[0].score == 0.0


@pytest.mark.parametrize(
    ("pair", "reason"),
    [
        ("density=-1", "from 0 to"),
        ("density=nan", "from 0 to"),
        ("density=inf", "from 0 to"),
        ("density=1e7", "from 0 to"),
        ("density=abc", "from 0 to"),
        ("density", "NAME=VALUE"),
    ],
)
def test_a_negative_or_non_finite_weight_is_refused(
    sf: ModuleType, pair: str, reason: str
) -> None:
    with pytest.raises(sf.Refusal, match=reason):
        sf.parse_weights([pair])


def test_all_weights_zero_is_refused(sf: ModuleType) -> None:
    with pytest.raises(sf.Refusal, match="at least one weight"):
        sf.parse_weights(["density=0"])


def test_an_unknown_weight_name_is_refused_with_the_names(sf: ModuleType) -> None:
    with pytest.raises(sf.Refusal) as info:
        sf.parse_weights(["sharpnes=1"])
    for name in sf.WEIGHT_NAMES:
        assert name in str(info.value)


def test_a_weight_given_twice_is_refused(sf: ModuleType) -> None:
    with pytest.raises(sf.Refusal, match="twice"):
        sf.parse_weights(["wall=1", "wall=2"])


# -------------------------------------------------------------------- choice

# The oracles of the geometry, written here from the polygon of ``footprint()`` and not
# taken from the script or from the route's helpers (FM-124, FM-132).


def _inside(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """All four cross products of one sign, whichever way the corners turn."""
    crosses = [
        (polygon[(k + 1) % 4][0] - polygon[k][0]) * (point[1] - polygon[k][1])
        - (polygon[(k + 1) % 4][1] - polygon[k][1]) * (point[0] - polygon[k][0])
        for k in range(4)
    ]
    return all(c >= 0.0 for c in crosses) or all(c <= 0.0 for c in crosses)


def _distance_to_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> float:
    """0 inside; else the smallest distance to 400 points along each edge (an over-estimate by under a um)."""
    if _inside(point, polygon):
        return 0.0
    best = math.inf
    for k in range(4):
        (ax, ay), (bx, by) = polygon[k], polygon[(k + 1) % 4]
        for step in range(401):
            t = step / 400
            best = min(best, math.dist(point, (ax + t * (bx - ax), ay + t * (by - ay))))
    return best


def _overlap_has_area(
    a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]
) -> bool:
    """Two convex polygons share area unless an edge normal separates them (touching is not an overlap)."""
    for polygon in (a, b):
        for k in range(4):
            (px, py), (qx, qy) = polygon[k], polygon[(k + 1) % 4]
            nx, ny = qy - py, px - qx
            norm = math.hypot(nx, ny)
            proj_a = [(x * nx + y * ny) / norm for x, y in a]
            proj_b = [(x * nx + y * ny) / norm for x, y in b]
            if max(proj_a) <= min(proj_b) + 1e-4 or max(proj_b) <= min(proj_a) + 1e-4:
                return False
    return True


def _choose(sf: ModuleType, scored, *, matrix: Matrix2 = UPRIGHT, **given):
    options = {"n": 500, "min_cells": 1, "min_score": 0.0} | given
    return sf.choose(scored, matrix=matrix, image_shape_px=SHAPE, **options)


def test_a_cell_on_the_edge_is_in_and_counted_once(sf: ModuleType) -> None:
    # Two fields that touch exactly (400 um apart for an upright 80-pixel frame): the
    # cell on their shared edge is in both by the footprint and held by the first.
    x1, x2, y = 800.0, 1200.0, -500.0
    xy = np.array([(1000.0, -500.0), (x1 - 50.0, y)], dtype=np.float64)
    tree = cKDTree(xy)
    for x in (x1, x2):
        found = sf.cells_in_field(xy, tree, x, y, matrix=UPRIGHT, image_shape_px=SHAPE)
        assert 0 in found
    cells = [cell_at("A1", 0.0, 0.0), cell_at("A1", -250.0, 0.0)]
    counts, uncovered = sf.tally(
        [(x1, y), (x2, y)],
        cells,
        labels=("single",),
        matrix=UPRIGHT,
        image_shape_px=SHAPE,
    )
    assert counts == [{"single": 2}, {"single": 0}]
    assert uncovered == 0
    # In the other order the second field of the list holds the edge cell, which is
    # the order of the choice and not of the stage: the first chosen holds it.
    swapped, _ = sf.tally(
        [(x2, y), (x1, y)],
        cells,
        labels=("single",),
        matrix=UPRIGHT,
        image_shape_px=SHAPE,
    )
    assert swapped == [{"single": 1}, {"single": 1}]


def test_the_counts_and_the_uncovered_add_up_to_the_wanted_cells(
    sf: ModuleType,
) -> None:
    cells = [
        cell_at("A1", dx, dy, label)
        for dx, dy, label in [
            (0.0, 0.0, "single"),
            (10.0, 5.0, "single"),
            (500.3, 100.7, "doublet"),
            (505.3, 101.7, "single"),
            (2000.0, 2000.0, "single"),
            (-30.1, 40.2, "debris"),
        ]
    ]
    fields = [(1000.0, -500.0), (1500.0, -400.0)]
    counts, uncovered = sf.tally(
        fields,
        cells,
        labels=("single", "doublet"),
        matrix=MATRICES["turned 30"],
        image_shape_px=SHAPE,
    )
    wanted = sum(1 for c in cells if c.label in ("single", "doublet"))
    assert counts == [
        {"single": 2, "doublet": 0},
        {"single": 1, "doublet": 1},
    ]
    assert sum(sum(c.values()) for c in counts) + uncovered == wanted
    assert uncovered == 1  # the cell 2000 um from the centre is in no field


def test_chosen_fields_never_overlap(sf: ModuleType) -> None:
    # A grid of candidates 150.7 um apart (decimals keep any pair off a field's edge), on a
    # camera turned 30 degrees: many pairs overlap, and the choice must keep none of them.
    matrix = MATRICES["turned 30"]
    positions = [
        (1000.0 + 150.7 * i, -500.0 + 131.3 * j) for i in range(8) for j in range(6)
    ]
    candidates = [
        _cand(sf, k, n_wanted=1 + k % 5, at=at) for k, at in enumerate(positions)
    ]
    scored = sf.score_candidates(candidates, sf.parse_weights([]))
    chosen = _choose(sf, scored, matrix=matrix)
    assert 2 <= len(chosen) < len(candidates)
    polygons = [
        footprint(
            s.candidate.x_um, s.candidate.y_um, matrix=matrix, image_shape_px=SHAPE
        )
        for s in chosen
    ]
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            assert not _overlap_has_area(polygons[i], polygons[j])


def test_touching_fields_may_both_be_chosen(sf: ModuleType) -> None:
    # 400 um apart for an 80-pixel frame of 5 um: they touch. 399.9 um apart they overlap.
    for gap, kept in [(400.0, 2), (399.9, 1)]:
        candidates = [
            _cand(sf, 0, n_wanted=3, at=(1000.0, -500.0)),
            _cand(sf, 1, n_wanted=2, at=(1000.0 + gap, -500.0)),
        ]
        scored = sf.score_candidates(candidates, sf.parse_weights([]))
        assert len(_choose(sf, scored)) == kept
    # The 60 rows exchange with the 80 columns when the camera is turned 90 degrees:
    # the same 400 um apart along Y is then a touching pair and 300 um along Y is not.
    turned = MATRICES["turned 90"]
    for dx, dy, kept in [
        (0.0, 400.0, 2),
        (0.0, 300.0, 1),
        (300.0, 0.0, 2),
        (399.9, 0.0, 2),
    ]:
        candidates = [
            _cand(sf, 0, n_wanted=3, at=(1000.0, -500.0)),
            _cand(sf, 1, n_wanted=2, at=(1000.0 + dx, -500.0 + dy)),
        ]
        scored = sf.score_candidates(candidates, sf.parse_weights([]))
        assert len(_choose(sf, scored, matrix=turned)) == kept, (dx, dy)


def test_fields_per_well_bounds_the_choice(sf: ModuleType) -> None:
    candidates = [
        _cand(sf, k, n_wanted=k + 1, at=(1000.0 + 1000.0 * k, 0.0)) for k in range(6)
    ]
    scored = sf.score_candidates(candidates, sf.parse_weights([]))
    chosen = _choose(sf, scored, n=3)
    assert [s.candidate.frame for s in chosen] == [5, 4, 3]


def test_min_cells_drops_the_empty_field(sf: ModuleType) -> None:
    near_wall = _cand(sf, 0, n_wanted=1, wall_um=3000.0, at=(0.0, 0.0))
    inside = _cand(sf, 1, n_wanted=3, wall_um=100.0, at=(1000.0, 0.0))
    empty = _cand(sf, 2, n_wanted=0, wall_um=2000.0, at=(2000.0, 0.0))
    scored = sf.score_candidates(
        [near_wall, inside, empty], sf.parse_weights(["density=0", "wall=1"])
    )
    assert [s.candidate.frame for s in _choose(sf, scored, min_cells=1)] == [0, 1]
    assert [s.candidate.frame for s in _choose(sf, scored, min_cells=2)] == [1]
    # Best score first, which is the order of the choice and not of the frames.
    assert [s.candidate.frame for s in _choose(sf, scored, min_cells=0)] == [0, 2, 1]


def test_min_score_stops_the_choice(sf: ModuleType) -> None:
    candidates = [
        _cand(sf, k, n_wanted=n, at=(1000.0 * k, 0.0)) for k, n in enumerate([10, 5, 2])
    ]
    scored = sf.score_candidates(candidates, sf.parse_weights([]))  # 1.0, 0.5, 0.2
    assert len(_choose(sf, scored, min_score=0.0)) == 3
    assert len(_choose(sf, scored, min_score=0.4)) == 2
    assert len(_choose(sf, scored, min_score=0.5)) == 2  # a score at the bar is kept
    assert len(_choose(sf, scored, min_score=1.0)) == 1


def test_equal_scores_go_to_the_lowest_frame(sf: ModuleType) -> None:
    candidates = [_cand(sf, k, n_wanted=4, at=(1000.0 * k, 0.0)) for k in (5, 2, 9)]
    scored = sf.score_candidates(candidates, sf.parse_weights([]))
    assert [s.candidate.frame for s in _choose(sf, scored, n=2)] == [2, 5]


def test_the_stops_start_at_the_field_nearest_the_previous_well(
    sf: ModuleType,
) -> None:
    points = [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)]
    # Coming from the right, the route starts at the right-hand field and walks left.
    assert sf.order_stops(points, (2500.0, 0.0)) == [2, 1, 0]
    assert sf.order_stops(points, (-500.0, 0.0)) == [0, 1, 2]
    # Two fields as near as each other: the one chosen first.
    assert sf.order_stops(points, (500.0, 0.0))[0] == 0
    assert sf.order_stops([], (0.0, 0.0)) == []


@given(
    rotation_deg=st.floats(0.0, 360.0),
    margin_um=st.floats(0.0, 2000.0),
    exclude_um=st.floats(0.0, 1500.0),
    mirrored=st.booleans(),
    offsets=st.lists(
        st.tuples(st.floats(-3600.0, 3600.0), st.floats(-3600.0, 3600.0)),
        min_size=1,
        max_size=14,
    ),
)
@settings(max_examples=60, deadline=None)
def test_no_chosen_field_leaves_its_well_or_enters_an_exclusion(
    sf: ModuleType,
    rotation_deg: float,
    margin_um: float,
    exclude_um: float,
    mirrored: bool,
    offsets: list[tuple[float, float]],
) -> None:
    matrix = matrix_of(PIXEL_UM, rotation_deg, mirrored)
    centre = CENTRES["A1"]
    well = WellGeometry(
        well="A1",
        center_um=centre,
        size_um=(6400.0, 6400.0),
        circular=True,
        rotation_deg=0.0,
    )
    frames = tuple(
        ScanFrame(
            well="A1",
            index=k,
            name=f"A1_{k:04d}",
            file=f"scan/t/A1_{k:04d}.tif",
            x_um=centre[0] + dx,
            y_um=centre[1] + dy,
            z_um=None,
            focus="none",
            pfs_offset=None,
            t_s=float(k),
        )
        for k, (dx, dy) in enumerate(offsets)
    )
    scan = ScanResult(
        folder="scan/t",
        objective=None,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        image_shape_px=SHAPE,
        exposure_ms=10.0,
        frames=frames,
        failed=(),
    )
    candidates, _ = sf.candidates_of(
        "A1",
        scan,
        [],
        well,
        labels=("single",),
        classes=CLASSES,
        min_spacing_um=30.0,
        margin_um=margin_um,
        exclude_centre_um=exclude_um,
        stats={},
    )
    scored = sf.score_candidates(candidates, sf.parse_weights([]))
    for s in _choose(sf, scored, matrix=matrix, min_cells=0):
        corners = footprint(
            s.candidate.x_um, s.candidate.y_um, matrix=matrix, image_shape_px=SHAPE
        )
        assert all(well.contains(x, y, margin_um=margin_um) for x, y in corners)
        # The oracle over-estimates a distance by under a um.
        assert _distance_to_polygon(centre, corners) >= exclude_um - 1.0


# --------------------------------------------------------------- pixels, PNG

#: A well of 1000 um: a canvas of 1600 pixels is then 0.64 um a pixel, finer than every
#: camera of these tests, so that a marker of a few pixels is several canvas pixels.
SMALL_WELL = WellGeometry(
    well="A1",
    center_um=(1000.0, -500.0),
    size_um=(1000.0, 1000.0),
    circular=True,
    rotation_deg=0.0,
)
TI2E: Matrix2 = ((-1.1, 0.0), (0.0, 1.1))
GREEN = (0, 255, 0)


def _canvas_pixel_of(canvas, x_um: float, y_um: float) -> tuple[float, float]:
    """The canvas pixel ``(col, row)`` whose centre is at a stage point: the design's formula, ``x = x_min + (c + 0.5) s``."""
    return (
        (x_um - canvas.x_min_um) / canvas.um_per_px - 0.5,
        (canvas.y_max_um - y_um) / canvas.um_per_px - 0.5,
    )


def _marked_frame(rows: int = 3, cols: int = 3, at=(9, 59)) -> np.ndarray:
    """A 60 x 80 frame of 100 with a bright marker whose top-left pixel is ``at = (row, col)``, off the axes on purpose."""
    frame = np.full(SHAPE, 100.0, dtype=np.float32)
    frame[at[0] : at[0] + rows, at[1] : at[1] + cols] = 255.0
    return frame


def _lit_centroid(pixels: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(pixels == 255.0)
    assert len(rows) > 0, "the marker was not placed"
    return float(cols.mean()), float(rows.mean())


def test_the_mosaic_puts_a_marker_where_the_stage_says(sf: ModuleType) -> None:
    canvas = sf.canvas_of(SMALL_WELL, 1600)
    seen: list[tuple[float, float]] = []
    for matrix in [
        UPRIGHT,
        TI2E,
        MATRICES["turned 90"],
        MATRICES["mirrored"],
        MATRICES["turned 30"],
    ]:
        pixels = canvas.blank()
        sf.place_frame(
            pixels,
            _marked_frame(),
            1000.0,
            -500.0,
            matrix=matrix,
            x_min_um=canvas.x_min_um,
            y_max_um=canvas.y_max_um,
            um_per_px=canvas.um_per_px,
        )
        # The marker's middle is pixel (col 60, row 10) of the frame.
        x, y = stage_of_pixel(
            1000.0, -500.0, 60.0, 10.0, matrix=matrix, image_shape_px=SHAPE
        )
        want = _canvas_pixel_of(canvas, x, y)
        got = _lit_centroid(pixels)
        assert got == pytest.approx(want, abs=1.0)
        seen.append(want)
    # The markers of the five cameras are far apart, so that a placement that ignored the
    # matrix would not pass for one of them.
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert math.dist(seen[i], seen[j]) > 10.0


def test_a_downsampled_tile_keeps_the_marker_in_place(sf: ModuleType) -> None:
    canvas = sf.canvas_of(SMALL_WELL, 1600)
    for matrix in [UPRIGHT, MATRICES["turned 30"], MATRICES["mirrored"]]:
        frame = _marked_frame(
            rows=4, cols=4, at=(8, 20)
        )  # on even pixels: whole blocks
        tile = sf.block_mean(frame, 2)
        assert tile.shape == (30, 40)
        pixels = canvas.blank()
        (m00, m01), (m10, m11) = matrix
        sf.place_frame(
            pixels,
            tile,
            1000.0,
            -500.0,
            matrix=((2 * m00, 2 * m01), (2 * m10, 2 * m11)),
            x_min_um=canvas.x_min_um,
            y_max_um=canvas.y_max_um,
            um_per_px=canvas.um_per_px,
        )
        x, y = stage_of_pixel(
            1000.0, -500.0, 21.5, 9.5, matrix=matrix, image_shape_px=SHAPE
        )
        assert _lit_centroid(pixels) == pytest.approx(
            _canvas_pixel_of(canvas, x, y), abs=1.0
        )


def test_block_mean_averages_whole_blocks_and_crops_the_remainder(
    sf: ModuleType,
) -> None:
    frame = np.arange(35, dtype=np.uint16).reshape(5, 7)
    tile = sf.block_mean(frame, 2)
    assert tile.shape == (2, 3)  # the fifth row and seventh column are cropped
    assert tile.dtype == np.float32
    assert tile[0, 0] == np.mean([0, 1, 7, 8])
    assert tile[1, 2] == np.mean([18, 19, 25, 26])
    assert np.array_equal(sf.block_mean(frame, 1), frame.astype(np.float32))
    with pytest.raises(ValueError, match="at least 1"):
        sf.block_mean(frame, 0)
    with pytest.raises(ValueError, match="larger than"):
        sf.block_mean(frame, 6)


def test_the_reduction_factor_is_bounded_and_refuses_a_pixel_size_that_is_not_one(
    sf: ModuleType,
) -> None:
    assert sf.reduction_factor(4.0, 0.16, (1412, 1412)) == 25
    assert sf.reduction_factor(0.5, 5.0, SHAPE) == 1
    assert sf.reduction_factor(4.0, 1e-300, SHAPE) == min(
        SHAPE
    )  # bounded on floats first
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="pixel size"):
            sf.reduction_factor(4.0, bad, SHAPE)


def test_the_canvas_is_the_wells_not_the_frames(sf: ModuleType) -> None:
    import warnings

    round_well = WellGeometry(
        well="A1",
        center_um=(1000.0, -500.0),
        size_um=(6400.0, 6400.0),
        circular=True,
        rotation_deg=0.0,
    )
    canvas = sf.canvas_of(round_well, 1600)
    # The well's circumscribed square plus 2 % on each side, at most png_max_px wide.
    assert canvas.x_min_um == pytest.approx(1000.0 - 3200.0 * 1.02)
    assert canvas.y_max_um == pytest.approx(-500.0 + 3200.0 * 1.02)
    assert canvas.um_per_px == pytest.approx(2 * 3200.0 * 1.02 / 1600)
    assert (canvas.width_px, canvas.height_px) == (1600, 1600)
    pixels = canvas.blank()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for far in (1e9, 1e300, -1e9):
            sf.place_frame(
                pixels,
                _marked_frame(),
                far,
                -far,
                matrix=UPRIGHT,
                x_min_um=canvas.x_min_um,
                y_max_um=canvas.y_max_um,
                um_per_px=canvas.um_per_px,
            )
    assert pixels.shape == (1600, 1600)
    assert np.isnan(pixels).all()


def test_the_canvas_of_a_square_well_is_the_turned_rectangles_box(
    sf: ModuleType,
) -> None:
    well = WellGeometry(
        well="A1",
        center_um=(0.0, 0.0),
        size_um=(6400.0, 4400.0),
        circular=False,
        rotation_deg=17.0,
    )
    canvas = sf.canvas_of(well, 1000)
    cos, sin = math.cos(math.radians(17.0)), math.sin(math.radians(17.0))
    half_x = (cos * 3200.0 + sin * 2200.0) * 1.02
    half_y = (sin * 3200.0 + cos * 2200.0) * 1.02
    assert canvas.x_min_um == pytest.approx(-half_x)
    assert canvas.y_max_um == pytest.approx(half_y)
    assert canvas.um_per_px == pytest.approx(2 * max(half_x, half_y) / 1000)
    assert max(canvas.width_px, canvas.height_px) <= 1000
    assert canvas.width_px > canvas.height_px  # a wider box than tall


def test_the_png_stretch_clips_before_it_casts(sf: ModuleType) -> None:
    import warnings

    rng = np.random.default_rng(3)
    pixels = rng.permutation(np.linspace(100.0, 200.0, 2000)).astype(np.float32)
    pixels = pixels.reshape(40, 50)
    pixels[0, :20] = np.nan
    pixels[1, 0] = -1e9
    pixels[1, 1] = 1e9
    pixels[1, 2] = 150.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grey = sf.stretch_to_uint8(pixels)
    assert grey.dtype == np.uint8
    assert grey[1, 0] == 0
    assert grey[1, 1] == 255
    assert 0 < grey[1, 2] < 255
    assert (grey[0, :20] == 40).all()  # the unplaced pixels are dark grey
    flat = sf.stretch_to_uint8(np.full((4, 4), 7.0, dtype=np.float32))
    assert flat.dtype == np.uint8
    assert len(np.unique(flat)) == 1
    nothing = sf.stretch_to_uint8(np.full((4, 4), np.nan, dtype=np.float32))
    assert (nothing == 40).all()


def _render(sf: ModuleType, tmp_path: Path, *, stops, cells=(), matrix=UPRIGHT):
    from PIL import Image

    canvas = sf.canvas_of(SMALL_WELL, 800)
    grey = np.full((canvas.height_px, canvas.width_px), 40, dtype=np.uint8)
    image = sf.render_png(
        grey,
        canvas,
        SMALL_WELL,
        cells=cells,
        classes=CLASSES,
        wanted=("single",),
        stops=stops,
        matrix=matrix,
        image_shape_px=SHAPE,
        caption="A1: test",
    )
    path = tmp_path / "A1.png"
    image.save(path, "PNG")
    reopened = Image.open(path)
    reopened.load()
    return canvas, reopened


@pytest.mark.parametrize("name", ["upright", "mirrored", "turned 30"])
def test_the_png_draws_a_chosen_field_in_green_with_its_number(
    sf: ModuleType, tmp_path: Path, name: str
) -> None:
    matrix = MATRICES[name]
    canvas, image = _render(
        sf, tmp_path, stops=[(0, 1000.0, -500.0), (1, 1050.0, -450.0)], matrix=matrix
    )
    assert image.mode == "RGB"
    assert max(image.size) <= 800
    assert image.size == (canvas.width_px, canvas.height_px)
    corners = footprint(1000.0, -500.0, matrix=matrix, image_shape_px=SHAPE)
    for k in range(4):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
        c, r = _canvas_pixel_of(canvas, (ax + bx) / 2.0, (ay + by) / 2.0)
        around = {
            image.getpixel((round(c) + dc, round(r) + dr))
            for dc in (-1, 0, 1)
            for dr in (-1, 0, 1)
        }
        assert GREEN in around, f"edge {k} of the footprint is not drawn"
    # The number of the stop is written at the footprint's centre, in green.
    centre_col, centre_row = _canvas_pixel_of(canvas, 1000.0, -500.0)

    def greens_around_the_centre(picture) -> int:
        """Pixels within 8 of the centre that are green-dominant: Pillow's default font is
        anti-aliased, so a glyph's pixels shade into the background and are not always the
        outline's pure green (they are on a build without FreeType)."""
        found = 0
        for dc in range(-8, 9):
            for dr in range(-8, 9):
                red, green, blue = picture.getpixel(
                    (round(centre_col) + dc, round(centre_row) + dr)
                )
                found += green > 150 and red < 60 and blue < 60
        return found

    assert greens_around_the_centre(image) > 0
    # And with no stop there is no green anywhere near the centre.
    _, bare = _render(sf, tmp_path, stops=[], matrix=matrix)
    assert greens_around_the_centre(bare) == 0


def test_the_png_colours_a_cell_by_its_class_and_rings_the_wanted_ones(
    sf: ModuleType, tmp_path: Path
) -> None:
    cells = [
        cell_at_small(0.0, 0.0, "single"),
        cell_at_small(60.0, 0.0, "doublet"),
    ]
    canvas, image = _render(sf, tmp_path, stops=[], cells=cells)
    gold, cyan, grey = (255, 215, 0), (0, 200, 255), (40, 40, 40)
    c, r = _canvas_pixel_of(canvas, 1000.0, -500.0)
    assert image.getpixel((round(c), round(r))) == gold
    assert (
        image.getpixel((round(c) + 5, round(r))) == gold
    )  # the ring of a wanted class
    c2, r2 = _canvas_pixel_of(canvas, 1060.0, -500.0)
    assert image.getpixel((round(c2), round(r2))) == cyan
    assert (
        image.getpixel((round(c2) + 5, round(r2))) == grey
    )  # a class nobody wanted has none


def cell_at_small(dx: float, dy: float, label: str) -> Cell:
    """A cell ``(dx, dy)`` um from the centre of ``SMALL_WELL``."""
    return Cell(
        well="A1",
        x_um=1000.0 + dx,
        y_um=-500.0 + dy,
        label=label,
        score=0.9,
        frame=0,
        box_px=(0.0, 0.0, 10.0, 10.0),
    )


# -------------------------------------------------------------------- record


def _options(sf: ModuleType, **given):
    base = {
        "labels": ("single",),
        "weights": sf.parse_weights([]),
        "fields_per_well": 10,
        "min_cells": 1,
        "min_score": 0.0,
        "wall_margin_um": 0.0,
        "exclude_centre_um": 0.0,
        "min_spacing_um": 30.0,
        "spacing_given": True,
        "spacing_cells": 0,
    }
    return sf.Options(**{**base, **given})


def _choices(sf: ModuleType, seed: Seed, options, wells: Sequence[str] | None = None):
    cells = load_cells(seed.path.parent, seed.result)
    return [
        sf.select_well(
            well,
            seed.scan,
            cells,
            _geometry(seed, well),
            classes=CLASSES,
            options=options,
            stats={},
        )
        for well in (wells if wells is not None else seed.plan.wells)
    ]


def _result(sf: ModuleType, seed: Seed, *, options=None, unreadable: int = 0, **given):
    options = options if options is not None else _options(sf, **given)
    return sf.route_result(
        seed.scan,
        seed.plan,
        _choices(sf, seed, options),
        options=options,
        unreadable=unreadable,
    )


def _one_cell_in_each_frame(scan: ScanResult) -> list[Cell]:
    """One single cell at the centre of every frame, so that every frame is a candidate worth choosing."""
    return [
        Cell(
            well=frame.well,
            x_um=frame.x_um,
            y_um=frame.y_um,
            label="single",
            score=0.9,
            frame=k,
            box_px=(0.0, 0.0, 10.0, 10.0),
        )
        for k, frame in enumerate(scan.frames)
    ]


def _row_of_three(tmp_path: Path, **given) -> Seed:
    """Three fields side by side that touch (overlap 0), 400 um apart: all of them can be chosen."""
    return seed_run(
        tmp_path,
        cells=_one_cell_in_each_frame,
        rows=1,
        columns=3,
        overlap=0.0,
        **given,
    )


def test_a_stop_takes_z_and_offset_from_its_own_frame(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path, focus_of={("A1", 1): "off"})
    result = _result(sf, seed)
    assert len(result.positions) == 3
    by_x = {round(f.x_um, 3): f for f in seed.scan.frames}
    for stop in result.positions:
        frame = by_x[round(stop.x_um, 3)]
        # The readback, which is 0.3 um off the planned field, and not the target.
        planned = next(
            f
            for f in seed.plan.fields
            if f.well == frame.well and f.index == frame.index
        )
        assert (stop.x_um, stop.y_um) == (frame.x_um, frame.y_um)
        assert (stop.x_um, stop.y_um) != (planned.x_um, planned.y_um)
        assert (stop.z_um, stop.pfs_offset) == (frame.z_um, frame.pfs_offset)
    off = by_x[round(seed.scan.frames[1].x_um, 3)]
    stop_of_off = next(s for s in result.positions if s.x_um == off.x_um)
    assert (stop_of_off.z_um, stop_of_off.pfs_offset) == (off.z_um, None)


def test_stops_are_named_numbered_and_counted_like_the_routes(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = seed_run(
        tmp_path,
        cells=lambda scan: [
            *_one_cell_in_each_frame(scan),
            Cell(
                well="A1",
                x_um=scan.frames[0].x_um + 10.0,
                y_um=scan.frames[0].y_um,
                label="doublet",
                score=0.9,
                frame=0,
                box_px=(0.0, 0.0, 10.0, 10.0),
            ),
        ],
        wells=("A1", "A2"),
        rows=1,
        columns=3,
        overlap=0.0,
    )
    result = _result(sf, seed, labels=("single", "doublet"))
    assert [s.order for s in result.positions] == list(range(6))
    assert [s.name for s in result.positions if s.well == "A1"] == [
        "A1_r000",
        "A1_r001",
        "A1_r002",
    ]
    assert [s.name for s in result.positions if s.well == "A2"] == [
        "A2_r000",
        "A2_r001",
        "A2_r002",
    ]
    assert all(set(s.counts) == {"single", "doublet"} for s in result.positions)
    first = next(s for s in result.positions if s.x_um == seed.scan.frames[0].x_um)
    assert first.counts == {"single": 1, "doublet": 1}
    assert result.labels == ("single", "doublet")
    assert result.objective == seed.scan.objective
    assert result.matrix_um_per_px == seed.matrix
    assert result.image_shape_px == seed.shape
    assert result.fov_um == pytest.approx((400.0, 300.0))


def test_the_first_well_starts_nearest_its_first_planned_field(
    tmp_path: Path, sf: ModuleType
) -> None:
    # Cells in the second and third field only: the first planned field is the left one
    # (x = 600), so the route starts at the nearer of the two, the middle one, and not at
    # the last planned (which would start at the right-hand field).
    seed = seed_run(
        tmp_path,
        cells=lambda scan: _one_cell_in_each_frame(scan)[1:],
        rows=1,
        columns=3,
        overlap=0.0,
    )
    result = _result(sf, seed)
    assert [s.x_um for s in result.positions] == [
        seed.scan.frames[1].x_um,
        seed.scan.frames[2].x_um,
    ]


def test_a_later_well_starts_nearest_the_previous_wells_last_stop(
    tmp_path: Path, sf: ModuleType
) -> None:
    # A2 is scanned first, left to right, and ends at its right-hand field; A1 lies to the
    # left, so its route starts at its right-hand field, not at its first planned one.
    seed = seed_run(
        tmp_path,
        cells=_one_cell_in_each_frame,
        wells=("A2", "A1"),
        rows=1,
        columns=3,
        overlap=0.0,
    )
    result = _result(sf, seed)
    a2 = [s.x_um for s in result.positions if s.well == "A2"]
    a1 = [s.x_um for s in result.positions if s.well == "A1"]
    assert a2 == sorted(a2)
    assert a1 == sorted(a1, reverse=True)
    assert [s.well for s in result.positions] == ["A2"] * 3 + ["A1"] * 3


def test_every_scanned_well_is_a_key_of_uncovered(
    tmp_path: Path, sf: ModuleType
) -> None:
    # A1 has a cell in every field, A2 none that is wanted, B1 has one cell that no field
    # holds (the margin leaves no candidate): the keys are the wells, zeros included.
    def cells(scan):
        return [
            *_one_cell_in_each_frame(scan)[:3],
            cell_at("A2", 0.0, 0.0, "debris"),
            cell_at("B1", 3000.0, 0.0),
        ]

    seed = seed_run(
        tmp_path, cells=cells, wells=("A1", "A2", "B1"), rows=1, columns=3, overlap=0.0
    )
    result = _result(sf, seed)
    assert list(result.uncovered) == ["A1", "A2", "B1"]
    assert result.uncovered == {"A1": 0, "A2": 0, "B1": 1}
    assert {s.well for s in result.positions} == {"A1"}


def test_the_route_is_the_distance_of_its_legs_and_travel_is_unknown(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path)
    result = _result(sf, seed)
    points = [(s.x_um, s.y_um) for s in result.positions]
    assert result.distance_um == pytest.approx(
        sum(math.dist(a, b) for a, b in itertools.pairwise(points))
    )
    assert result.travel_s is None  # unknown is not zero


def test_the_assumed_lines_name_the_score_the_candidates_and_the_spacing(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path, focus_of={("A1", 0): "unlocked"})
    given = _result(
        sf,
        seed,
        options=_options(
            sf, weights=sf.parse_weights(["isolation=2.5", "sharpness=0.5"])
        ),
        unreadable=2,
    )
    assert given.assumed == (
        "positions chosen by scripts/dev/select_fields.py, a prototype and not the "
        "route plugin; score weights density=1, isolation=2.5, purity=0, wall=0, "
        "sharpness=0.5, background=0",
        "the candidate fields are the scan's own frames at their readback positions, "
        "not a lattice",
        "min_spacing_um=30, given",
        "the route minimises the Euclidean distance between fields; whether the "
        "stage's move time follows it is not measured",
        "travel_s is not computed by the prototype",
        "1 frames were not candidates because the autofocus had not locked on them "
        "(unlocked)",
        "2 frames could not be read and were left out of the mosaic and of the pixel "
        "features",
    )
    measured = _result(
        sf,
        seed,
        options=_options(sf, min_spacing_um=45.0, spacing_given=False, spacing_cells=7),
    )
    assert measured.assumed[2] == (
        "min_spacing_um=45, measured as 3 times the median cell length of 7 wanted "
        "cells (the factor 3 is chosen, not measured)"
    )
    # Nothing left out: neither of the two last lines.
    (tmp_path / "clean").mkdir()
    clean = _result(sf, _row_of_three(tmp_path / "clean"))
    assert len(clean.assumed) == 5


def _record_of(sf: ModuleType, seed: Seed, result=None):
    doc = RunDocument.load(seed.path)
    result = result if result is not None else _result(sf, seed)
    return doc, sf.route_record(
        doc, result, params={"labels": ["single"], "weights": sf.parse_weights([])}
    )


def test_the_record_is_written_to_a_copy_and_the_original_is_untouched(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path)
    before = seed.path.read_bytes()
    doc, record = _record_of(sf, seed)
    copy = tmp_path / "run-selected.json"
    sf.write_copy(seed.path, copy, record)
    assert seed.path.read_bytes() == before
    written = RunDocument.load(copy)
    assert set(written.stages) == set(doc.stages) | {"route-prototype"}
    for name, stage in doc.stages.items():  # every other stage is kept, as it was
        assert written.stages[name] == stage
    stored = written.stages["route-prototype"]
    assert (stored.plugin, stored.provides, stored.status) == (
        "route-prototype",
        "route",
        StageStatus.done,
    )
    assert written.provider("route") == stored
    assert RouteResult.model_validate(stored.result).positions
    scan_record = doc.stages["scan"]
    assert (stored.profile, stored.dry_run) == (
        scan_record.profile,
        scan_record.dry_run,
    )


def test_the_record_reads_the_current_attempts_and_goes_stale_when_cells_runs_again(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path)
    doc, record = _record_of(sf, seed)
    assert {slot: (r.plugin, r.attempt) for slot, r in record.reads.items()} == {
        slot: (doc.provider(slot).plugin, doc.provider(slot).attempt)
        for slot in ("scan-plan", "scan", "cells")
    }
    copy = tmp_path / "run-selected.json"
    sf.write_copy(seed.path, copy, record)
    assert RunDocument.load(copy).is_stale("route-prototype") is False
    _cells_run_again(copy, seed, attempt="a2")
    assert RunDocument.load(copy).is_stale("route-prototype") is True


def _cells_run_again(path: Path, seed: Seed, *, attempt: str) -> None:
    """``cells`` run again: a new attempt of the stage, later than the first."""
    write_record(path, "cells", "cells", seed.result, attempt=attempt, day=29)


def test_an_older_route_record_stays_in_the_copy_and_loses(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path)
    older = _result(sf, seed, fields_per_well=1)
    write_record(seed.path, "route", "route", older)
    doc, prototype = _record_of(sf, seed)
    copy = tmp_path / "run-selected.json"
    sf.write_copy(seed.path, copy, prototype)
    written = RunDocument.load(copy)
    assert written.stages["route"] == doc.stages["route"]
    assert (
        len(RouteResult.model_validate(written.stages["route"].result).positions) == 1
    )
    assert written.provider("route").plugin == "route-prototype"


def test_no_absolute_path_reaches_the_record(tmp_path: Path, sf: ModuleType) -> None:
    seed = _row_of_three(tmp_path)
    _, record = _record_of(sf, seed)
    assert str(tmp_path) not in json.dumps(record.model_dump(mode="json"))
    assert str(tmp_path.parent) not in json.dumps(record.model_dump(mode="json"))


def test_the_copy_is_refused_when_it_is_the_original(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path)
    with pytest.raises(sf.Refusal, match="original"):
        sf.check_copy_target(seed.path, seed.path)
    link = tmp_path / "hard.json"
    try:
        os.link(seed.path, link)
    except OSError:
        pytest.skip("this file system has no hard links")
    with pytest.raises(sf.Refusal, match="original"):
        sf.check_copy_target(seed.path, link)  # the same file under another name


def test_the_copy_is_refused_when_its_folder_is_missing_or_it_is_a_folder(
    tmp_path: Path, sf: ModuleType
) -> None:
    seed = _row_of_three(tmp_path)
    with pytest.raises(sf.Refusal, match="folder"):
        sf.check_copy_target(seed.path, tmp_path / "missing" / "copy.json")
    (tmp_path / "a-folder").mkdir()
    with pytest.raises(sf.Refusal, match="folder"):
        sf.check_copy_target(seed.path, tmp_path / "a-folder")
    sf.check_copy_target(seed.path, tmp_path / "fine.json")  # no error


def test_a_copy_that_cannot_be_completed_is_removed(
    tmp_path: Path, sf: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _row_of_three(tmp_path)
    _, prototype = _record_of(sf, seed)
    copy = tmp_path / "run-selected.json"

    def fail(path: Path, rec: object) -> None:
        raise RunDocumentError(
            path, "could not be written (disk full); it is as it was"
        )

    monkeypatch.setattr(sf, "write_stage", fail)
    with pytest.raises(RunDocumentError):
        sf.write_copy(seed.path, copy, prototype)
    # A copy without the record would read as the original with a route nobody chose.
    assert not copy.exists()


def test_the_scans_camera_is_the_camera_not_the_plans(
    tmp_path: Path, sf: ModuleType
) -> None:
    # The plan was laid for a camera of 2.5 um pixels (fields of 200 x 150 um), the scan
    # imaged with 5 um pixels (400 x 300): a cell 190 um from a frame's centre is in the
    # scan's footprint and outside the plan's (FM-124).
    plan_camera = matrix_of(2.5, 0.0, False)
    seed = seed_run(
        tmp_path,
        cells=lambda scan: [
            Cell(
                well="A1",
                x_um=scan.frames[0].x_um + 190.0,
                y_um=scan.frames[0].y_um,
                label="single",
                score=0.9,
                frame=0,
                box_px=(0.0, 0.0, 10.0, 10.0),
            )
        ],
        rows=1,
        columns=1,
        plan_matrix=plan_camera,
    )
    assert seed.plan.matrix_um_per_px == plan_camera != seed.scan.matrix_um_per_px
    found, _ = _candidates(sf, seed)
    assert found[0].n_wanted == 1
    result = _result(sf, seed)
    assert result.matrix_um_per_px == seed.scan.matrix_um_per_px
    assert result.fov_um == pytest.approx((400.0, 300.0))
    assert result.positions[0].counts == {"single": 1}


# ------------------------------------------------------------------- command


def _two_wells(tmp_path: Path, **given) -> Seed:
    """Two wells of three touching fields, a cell in each: six stops, whatever the weights."""
    return seed_run(
        tmp_path,
        cells=_one_cell_in_each_frame,
        wells=("A1", "A2"),
        rows=1,
        columns=3,
        overlap=0.0,
        **given,
    )


def _main(sf: ModuleType, capsys: pytest.CaptureFixture[str], path: Path, *args: str):
    code = sf.main([str(path), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _stops_of(copy: Path) -> list:
    stage = RunDocument.load(copy).stages["route-prototype"]
    return list(RouteResult.model_validate(stage.result).positions)


def test_the_command_prints_the_table_and_writes_the_files(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    before = seed.path.read_bytes()
    code, out, err = _main(sf, capsys, seed.path)
    assert (code, err) == (0, "")
    assert "== A1: 3 fields chosen of 3 candidates, 3 wanted cells" in out
    assert "== A2: 3 fields chosen of 3 candidates, 3 wanted cells" in out
    assert "min_spacing_um 150 (measured, 3 times the median cell length of 6" in out
    for name in (
        "selection/selection.json",
        "selection/A1.png",
        "selection/A2.png",
        "run-selected.json",
    ):
        assert (tmp_path / name).is_file(), name
        assert name.split("/")[-1] in out  # every file written is named
    assert seed.path.read_bytes() == before
    stops = _stops_of(tmp_path / "run-selected.json")
    assert [s.name for s in stops] == [
        f"{w}_r{k:03d}" for w in ("A1", "A2") for k in range(3)
    ]
    document = RunDocument.load(tmp_path / "run-selected.json")
    assert document.provider("route").plugin == "route-prototype"


def test_the_table_lists_the_features_of_the_ranked_fields_and_marks_the_chosen(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # Four fields that overlap their neighbours (10 %), with 1 to 4 cells each: the table
    # ranks all four, and the stop column is the route number of the chosen ones only.
    def cells(scan):
        return [
            Cell(
                well="A1",
                x_um=frame.x_um - 150.0 + 40.0 * i,
                y_um=frame.y_um,
                label="single",
                score=0.9,
                frame=k,
                box_px=(0.0, 0.0, 10.0, 10.0),
            )
            for k, frame in enumerate(scan.frames)
            for i in range(k + 1)
        ]

    seed = seed_run(tmp_path, cells=cells)
    code, out, _ = _main(sf, capsys, seed.path, "--min-spacing-um", "50")
    assert code == 0
    lines = out.splitlines()
    header = next(
        i for i, line in enumerate(lines) if line.split()[:2] == ["rank", "frame"]
    )
    assert lines[header].split() == [
        "rank", "frame", "x_um", "y_um", "score", "n", "dens/mm2", "iso", "purity",
        "nn_um", "wall_um", "sharp", "bg", "focus", "stop",
    ]  # fmt: skip
    rows = []
    for line in lines[header + 1 :]:
        if line.strip().startswith("chosen"):
            break
        rows.append(line.split())
    assert len(rows) == 4
    assert [int(row[0]) for row in rows] == [1, 2, 3, 4]
    scores = [float(row[4]) for row in rows]
    assert scores == sorted(scores, reverse=True)
    chosen = {
        (f"{s.x_um:.1f}", f"{s.y_um:.1f}"): int(s.name.rsplit("_r", 1)[1])
        for s in _stops_of(tmp_path / "run-selected.json")
    }
    assert 1 <= len(chosen) < 4  # the 10 % overlap rules some out
    for row in rows:
        expected = chosen.get((row[2], row[3]))
        assert row[-1] == ("-" if expected is None else str(expected))
    # The best field holds the 4 cells, which are 40 um apart: not isolated at 50 um.
    assert (rows[0][5], rows[0][7]) == ("4", "0")
    assert "chosen, in route order:" in out
    assert "uncovered:" in out


def test_selection_json_holds_the_printed_numbers(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    code, out, _ = _main(
        sf, capsys, seed.path, "--weight", "wall=1", "--min-spacing-um", "40"
    )
    assert code == 0
    data = json.loads(
        (tmp_path / "selection" / "selection.json").read_text(encoding="utf-8")
    )
    assert data["schema_version"] == 1
    assert data["run_doc"] == "run.json"
    assert data["labels"] == ["single"]
    assert data["weights"] == {
        "density": 1.0, "isolation": 0.0, "purity": 0.0, "wall": 1.0,
        "sharpness": 0.0, "background": 0.0,
    }  # fmt: skip
    assert (data["min_spacing_um"], data["min_spacing_source"]) == (40.0, "given")
    assert list(data["wells"]) == ["A1", "A2"]
    stops = {
        s.name: (s.x_um, s.y_um) for s in _stops_of(tmp_path / "run-selected.json")
    }
    for well, entry in data["wells"].items():
        assert entry["uncovered"] == 0
        assert len(entry["candidates"]) == 3
        for candidate in entry["candidates"]:
            assert 0.0 <= candidate["score"] <= 1.0
            assert set(candidate["parts"]) == set(sf.WEIGHT_NAMES)
            assert f"{candidate['score']:.3f}" in out  # a score printed in the table
            name = f"{well}_r{candidate['stop']:03d}"
            assert stops[name] == (candidate["x_um"], candidate["y_um"])


def test_the_output_is_ascii(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # A file name with an accent and an objective with a Greek letter reach the console
    # as escapes (FM-20): a Windows console decodes with a code page.
    seed = seed_run(
        tmp_path,
        cells=_one_cell_in_each_frame,
        scan_objective="Plan Apo \N{GREEK SMALL LETTER LAMDA} 20x",
        rows=1,
        columns=3,
        overlap=0.0,
    )
    named = tmp_path / "run-\N{LATIN SMALL LETTER E WITH ACUTE}.json"
    named.write_bytes(seed.path.read_bytes())
    code, out, err = _main(sf, capsys, named)
    assert code == 0
    out.encode("ascii")  # does not raise
    err.encode("ascii")
    assert "run-\\xe9.json" in out
    assert "Plan Apo \\u03bb 20x" in out
    # And a refusal's text, which names the file, is ASCII as well.
    code, _, err = _main(
        sf, capsys, tmp_path / "missing-\N{LATIN SMALL LETTER E WITH ACUTE}.json"
    )
    assert code == 2
    err.encode("ascii")


def test_no_absolute_path_reaches_the_record_or_the_json(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    code, _, _ = _main(sf, capsys, seed.path)
    assert code == 0
    document = (tmp_path / "selection" / "selection.json").read_text(encoding="utf-8")
    stage = RunDocument.load(tmp_path / "run-selected.json").stages["route-prototype"]
    record = json.dumps(stage.model_dump(mode="json"))
    for text in (document, record):
        assert str(tmp_path) not in text
        assert str(tmp_path.parent) not in text


def test_the_table_is_printed_before_a_failed_write_and_the_exit_is_1(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    (tmp_path / "selection" / "A1.png").mkdir(
        parents=True
    )  # a folder where a picture goes
    code, out, err = _main(sf, capsys, seed.path)
    assert code == 1
    assert "== A1:" in out  # the table was printed
    assert "route: 6 stops" in out
    assert "A1.png" in err
    assert err.startswith("select_fields: could not write")
    # The other files were written all the same.
    assert (tmp_path / "selection" / "selection.json").is_file()
    assert (tmp_path / "selection" / "A2.png").is_file()
    assert (tmp_path / "run-selected.json").is_file()


def _rewritten(seed: Seed, edit) -> None:
    """The run document of ``seed`` with its stages edited by ``edit``."""
    document = RunDocument.load(seed.path)
    edited = document.model_copy(update={"stages": edit(dict(document.stages))})
    seed.path.write_text(
        json.dumps(edited.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )


def test_an_upstream_that_is_not_usable_says_what_to_run(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    _rewritten(seed, lambda stages: {k: v for k, v in stages.items() if k != "cells"})
    reason = RunDocument.load(seed.path).unusable(
        "cells", profile="demo", dry_run=False
    )
    assert reason is not None
    assert "no cells" in reason
    code, out, err = _main(sf, capsys, seed.path)
    assert (code, out, err) == (2, "", f"select_fields: {reason}\n")
    assert not (tmp_path / "selection").exists()
    assert not (tmp_path / "run-selected.json").exists()

    seed = _two_wells(tmp_path)
    _rewritten(
        seed,
        lambda stages: {
            **stages,
            "scan": stages["scan"].model_copy(update={"status": StageStatus.failed}),
        },
    )
    reason = RunDocument.load(seed.path).unusable("scan", profile="demo", dry_run=False)
    assert reason is not None
    assert "ended failed" in reason
    code, out, err = _main(sf, capsys, seed.path)
    assert (code, out, err) == (2, "", f"select_fields: {reason}\n")


def test_a_run_document_that_is_missing_or_not_a_run_document_is_refused(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _main(sf, capsys, tmp_path / "nope.json")
    assert code == 2
    assert "nope.json" in err
    assert "does not exist" in err
    (tmp_path / "text.json").write_text("not json", encoding="utf-8")
    code, _, err = _main(sf, capsys, tmp_path / "text.json")
    assert code == 2
    assert "not valid JSON" in err


def test_a_label_the_cells_do_not_have_is_refused_with_the_classes(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    code, out, err = _main(sf, capsys, seed.path, "--labels", "single,nope")
    assert (code, out) == (2, "")
    assert "'nope'" in err
    assert "['single', 'doublet', 'debris']" in err
    code, _, err = _main(sf, capsys, seed.path, "--labels", "single,single")
    assert code == 2
    assert "repeats" in err


def test_a_well_without_a_wanted_cell_has_no_field_and_the_table_says_so(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = seed_run(
        tmp_path,
        cells=lambda scan: [
            *_one_cell_in_each_frame(scan)[:3],
            cell_at("A2", 0.0, 0.0, "debris"),
        ],
        wells=("A1", "A2"),
        rows=1,
        columns=3,
        overlap=0.0,
    )
    code, out, _ = _main(sf, capsys, seed.path, "--min-spacing-um", "30")
    assert code == 0
    assert "== A2: 0 fields chosen of 3 candidates, 0 wanted cells" in out
    assert "no wanted cell in this well: no field chosen" in out
    stage = RunDocument.load(tmp_path / "run-selected.json").stages["route-prototype"]
    result = RouteResult.model_validate(stage.result)
    assert {p.well for p in result.positions} == {"A1"}
    assert result.uncovered == {"A1": 0, "A2": 0}  # every scanned well is a key


def test_no_field_in_any_well_writes_no_copy(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = seed_run(
        tmp_path,
        cells=[cell_at("A1", 0.0, 0.0, "debris")],
        rows=1,
        columns=3,
        overlap=0.0,
    )
    # The spacing cannot be measured from no wanted cell: asked for, it is refused.
    code, _, err = _main(sf, capsys, seed.path)
    assert code == 2
    assert "spacing cannot be measured" in err
    code, out, _ = _main(sf, capsys, seed.path, "--min-spacing-um", "30")
    assert code == 0
    assert "no field was chosen in any well: no run document copy written" in out
    assert not (tmp_path / "run-selected.json").exists()
    assert (tmp_path / "selection" / "selection.json").is_file()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--fields-per-well", "0"),
        ("--fields-per-well", "501"),
        ("--fields-per-well", "abc"),
        ("--min-cells", "0"),
        ("--min-score", "1.5"),
        ("--min-score", "-0.1"),
        ("--min-score", "nan"),
        ("--wall-margin-um", "-1"),
        ("--wall-margin-um", "inf"),
        ("--exclude-centre-um", "nan"),
        ("--exclude-centre-um", "2e6"),
        ("--png-max-px", "100"),
        ("--png-max-px", "8001"),
        ("--show", "0"),
        ("--min-spacing-um", "nan"),
        ("--min-spacing-um", "0"),
        ("--min-spacing-um", "inf"),
        ("--min-spacing-um", "1e6"),
        ("--weight", "density=-1"),
        ("--weight", "nope=1"),
    ],
)
def test_options_are_bounded(
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    # The run document is not even there: the options are refused first (FM-32), with the
    # option's name, and nothing is created.
    code, out, err = _main(sf, capsys, tmp_path / "missing.json", option, value)
    assert (code, out) == (2, "")
    assert err.startswith("select_fields: ")
    assert err.count("\n") == 1
    assert option.removeprefix("--") in err.replace("--", "")
    assert list(tmp_path.iterdir()) == []


def test_a_pixel_weight_without_pixels_is_refused_and_says_so(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    for name in ("sharpness", "background"):
        code, out, err = _main(
            sf, capsys, seed.path, "--no-pixels", "--weight", f"{name}=1"
        )
        assert (code, out) == (2, "")
        assert name in err
        assert "--no-pixels" in err
    assert not (tmp_path / "selection").exists()


def test_the_copy_is_refused_before_any_frame_is_read(
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reading of a frame swallows every failure on purpose (a bad frame is a finding),
    # so a stub that raises would prove nothing: it records, and the test asserts.
    reads: list[object] = []
    monkeypatch.setattr(sf, "load_frame", lambda *args: reads.append(args))
    seed = _two_wells(tmp_path)
    before = seed.path.read_bytes()
    code, _, err = _main(sf, capsys, seed.path, "--run-doc-out", str(seed.path))
    assert code == 2
    assert "original" in err
    code, _, err = _main(
        sf, capsys, seed.path, "--run-doc-out", str(tmp_path / "no" / "c.json")
    )
    assert code == 2
    assert "folder" in err
    (tmp_path / "taken").write_text("a file", encoding="utf-8")
    code, _, err = _main(sf, capsys, seed.path, "--out-dir", str(tmp_path / "taken"))
    assert code == 2
    assert "is a file" in err
    assert seed.path.read_bytes() == before
    assert not (tmp_path / "selection").exists()
    assert reads == []  # no frame was opened by any of the three refusals


def test_an_unreadable_frame_is_counted_and_never_ends_the_run(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path, write_frames=False)
    code, out, _ = _main(sf, capsys, seed.path)
    assert code == 0
    assert "6 frames could not be read (first: scan/t/A1_0000.tif" in out
    stops = RouteResult.model_validate(
        RunDocument.load(tmp_path / "run-selected.json")
        .stages["route-prototype"]
        .result
    )
    assert len(stops.positions) == 6  # no pixel weight: the candidates stay
    assert (
        "6 frames could not be read and were left out of the mosaic and of the pixel features"
        in stops.assumed
    )
    # With a pixel weight a candidate with no pixels cannot be ranked: dropped, and said.
    (tmp_path / "run-selected.json").unlink()
    code, out, _ = _main(sf, capsys, seed.path, "--weight", "sharpness=1")
    assert code == 0
    assert "6 dropped for a pixel weight" in out
    assert "no candidate to rank" in out
    assert "no field was chosen in any well" in out
    assert not (tmp_path / "run-selected.json").exists()


def test_a_frame_of_another_shape_is_left_out_of_the_pixel_features(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    def pixels(frame: ScanFrame) -> np.ndarray:
        shape = (50, 80) if frame.index == 1 else SHAPE
        return np.full(shape, 1000, dtype=np.uint16)

    seed = seed_run(
        tmp_path,
        cells=_one_cell_in_each_frame,
        rows=1,
        columns=3,
        overlap=0.0,
        frame_pixels=pixels,
    )
    code, out, _ = _main(sf, capsys, seed.path)
    assert code == 0
    assert "1 frames could not be read" in out
    assert "A1_0001 is 80 x 50 pixels, the scan says 80 x 60" in out


def test_wells_can_be_chosen_and_an_unscanned_one_is_refused(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    code, out, _ = _main(sf, capsys, seed.path, "--wells", "a2")
    assert code == 0
    assert "== A2:" in out
    assert "== A1:" not in out
    assert {s.well for s in _stops_of(tmp_path / "run-selected.json")} == {"A2"}
    assert not (tmp_path / "selection" / "A1.png").exists()
    code, _, err = _main(sf, capsys, seed.path, "--wells", "A2,Z9")
    assert code == 2
    assert "'Z9'" in err
    assert "A1, A2" in err


def test_no_pixels_reads_no_frame_and_no_png_writes_no_picture(
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _two_wells(tmp_path)
    code, out, _ = _main(sf, capsys, seed.path, "--no-png")
    assert code == 0
    assert not list((tmp_path / "selection").glob("*.png"))
    assert (tmp_path / "selection" / "selection.json").is_file()

    reads: list[object] = []  # recorded, not raised: a frame's failure is swallowed
    monkeypatch.setattr(sf, "load_frame", lambda *args: reads.append(args))
    code, out, _ = _main(sf, capsys, seed.path, "--no-pixels")
    assert code == 0
    assert reads == []
    assert (tmp_path / "selection" / "A1.png").is_file()  # over a dark canvas
    first = next(
        line for line in out.splitlines() if line.split()[:2] == ["rank", "frame"]
    )
    row = out.splitlines()[out.splitlines().index(first) + 1].split()
    assert row[11:13] == ["-", "-"]  # no sharpness, no background
    data = json.loads(
        (tmp_path / "selection" / "selection.json").read_text(encoding="utf-8")
    )
    assert data["options"]["pixels"] is False


def test_a_second_run_rebuilds_the_copy_from_the_original(
    tmp_path: Path, sf: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = _two_wells(tmp_path)
    assert _main(sf, capsys, seed.path)[0] == 0
    first = RunDocument.load(tmp_path / "run-selected.json")
    assert _main(sf, capsys, seed.path, "--fields-per-well", "1")[0] == 0
    second = RunDocument.load(tmp_path / "run-selected.json")
    assert set(second.stages) == set(first.stages)  # replaced, not stacked
    assert len(_stops_of(tmp_path / "run-selected.json")) == 2
    assert (
        second.stages["route-prototype"].attempt
        != first.stages["route-prototype"].attempt
    )


@pytest.mark.parametrize("matrix_name", ["upright", "turned 30 and mirrored"])
def test_the_picture_of_a_well_puts_the_frames_where_the_stage_says(
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
    matrix_name: str,
) -> None:
    from PIL import Image

    # 256 pixels for a 6400 um well is 25.5 um a pixel: the 5 um frame is reduced by 5.
    # A bright 15 x 15 pixel block (rows 5-19, columns 55-69, whole 5 x 5 tiles) is the
    # only thing the stretch maps to white; it must land where the stage puts it.
    matrix = {
        "upright": UPRIGHT,
        "turned 30 and mirrored": matrix_of(PIXEL_UM, 30.0, True),
    }[matrix_name]

    def pixels(frame: ScanFrame) -> np.ndarray:
        data = np.full(SHAPE, 100, dtype=np.uint16)
        data[5:20, 55:70] = 4000
        return data

    seed = seed_run(
        tmp_path,
        cells=[cell_at("A1", 3000.0, 3000.0)],  # far from the frame, out of the way
        rows=1,
        columns=1,
        matrix=matrix,
        frame_pixels=pixels,
    )
    code, _, _ = _main(
        sf,
        capsys,
        seed.path,
        "--png-max-px",
        "256",
        "--min-spacing-um",
        "30",
        "--min-cells",
        "1",
    )
    assert code == 0
    image = Image.open(tmp_path / "selection" / "A1.png")
    image.load()
    canvas = sf.canvas_of(_geometry(seed), 256)
    frame = seed.scan.frames[0]
    x, y = stage_of_pixel(
        frame.x_um, frame.y_um, 62.0, 12.0, matrix=matrix, image_shape_px=SHAPE
    )
    col, row = _canvas_pixel_of(canvas, x, y)
    assert image.getpixel((round(col), round(row))) == (255, 255, 255)
    # The point of the frame opposite through its centre is the dark background.
    opposite = stage_of_pixel(
        frame.x_um, frame.y_um, 17.0, 47.0, matrix=matrix, image_shape_px=SHAPE
    )
    col, row = _canvas_pixel_of(canvas, *opposite)
    assert image.getpixel((round(col), round(row))) == (0, 0, 0)
