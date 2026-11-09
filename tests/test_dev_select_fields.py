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
