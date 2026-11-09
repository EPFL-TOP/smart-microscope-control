"""scripts/dev/select_fields.py scores the scan's own frames as candidate fields (#343, m3 §11).

The script is a prototype for the owner to tune on real 40x scans: these tests
pin what it computes, not what a good field is. The camera is 60 x 80 pixels (not
square, FM-73), turned or mirrored where a test says so; every expected value is
built here from ``stage_of_pixel``, ``footprint`` and the seed, never from the
function under test (FM-124).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial import cKDTree

from select_fields_seed import (
    CLASSES,
    PIXEL_UM,
    SHAPE,
    UPRIGHT,
    Seed,
    cell_at,
    matrix_of,
    seed_run,
)
from smc.plugins.geometry import Matrix2, footprint, stage_of_pixel
from smc.plugins.route.algorithm import allowed
from smc.plugins.slots import (
    Cell,
    WellGeometry,
    load_cells,
    load_frame,
    well_geometry,
)
from smc.run import RunDocument

ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH = SHAPE


@pytest.fixture(scope="module")
def sf() -> ModuleType:
    """The script, loaded as ``tests/test_dev_point.py`` loads ``point.py``."""
    spec = importlib.util.spec_from_file_location(
        "select_fields_script", ROOT / "scripts" / "dev" / "select_fields.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
):
    """A candidate with the features a test gives it; the density is the wanted count over a 0.12 mm2 field."""
    return sf.Candidate(
        frame=frame,
        name=f"A1_{frame:04d}",
        x_um=float(frame),
        y_um=0.0,
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
