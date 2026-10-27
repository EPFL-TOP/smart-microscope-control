"""The cells stage's algorithm: suppression, pixels to stage µm, and one cell counted once (m3 §9.2).

Nothing here has a model or a microscope. Frames are not square on purpose
(FM-73): a swapped ``(height, width)`` is invisible on a square frame. The
merge tests build cells by hand, with the box position that makes the case
(``_cell``), and one test goes the whole way: the synthetic plate rendered by a
rotated, mirrored camera, a thresholding stub for a detector, and the stage
positions the blobs were put at.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from skimage import measure

from smc.plugins.cells.algorithm import (
    edge_distance_px,
    merge_across_fields,
    suppress_in_frame,
    to_cells,
)
from smc.plugins.cells.detector import Detection
from smc.plugins.geometry import Matrix2, pixel_matrix
from smc.plugins.slots import Cell, ScanFrame
from smc.testing.synthetic import Blob, PlateSample

SHAPE = (200, 320)  # (height, width): not square, on purpose (FM-73)
#: The frame centre as ``(col, row)``: the most uncut place of a frame.
CENTRE = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)``, the formula of §3.1, not the code under test."""
    angle = math.radians(rotation_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    flip = np.diag([-1.0 if mirrored else 1.0, -1.0])
    m = pixel_size_um * (rotation @ flip)
    return ((float(m[0, 0]), float(m[0, 1])), (float(m[1, 0]), float(m[1, 1])))


def _scan_frame(
    index: int = 0, *, well: str = "A1", x_um: float = 0.0, y_um: float = 0.0
) -> ScanFrame:
    return ScanFrame(
        well=well,
        index=index,
        name=f"f{index}",
        file=f"scan/f{index}.tif",
        x_um=x_um,
        y_um=y_um,
        z_um=None,
        focus="none",
        pfs_offset=None,
        t_s=0.0,
    )


def _detection(
    row0: float,
    col0: float,
    row1: float,
    col1: float,
    *,
    label: str = "single",
    score: float = 0.9,
) -> Detection:
    return Detection(row0, col0, row1, col1, label, score)


def _cell(
    x_um: float,
    frame: int,
    *,
    y_um: float = 0.0,
    well: str = "A1",
    centre_px: tuple[float, float] = CENTRE,
    score: float = 0.9,
    label: str = "single",
) -> Cell:
    """A cell whose 10-pixel box is centred at ``centre_px`` (col, row) of its frame.

    The merge reads the stage position and the box (for the distance to the
    frame's edge), so a test places a view near or far from an edge by
    ``centre_px``.
    """
    col, row = centre_px
    return Cell(
        well=well,
        x_um=x_um,
        y_um=y_um,
        label=label,
        score=score,
        frame=frame,
        box_px=(row - 5.0, col - 5.0, row + 5.0, col + 5.0),
    )


def _merge(cells: list[Cell], radius_um: float = 5.0) -> tuple[tuple[Cell, ...], int]:
    return merge_across_fields(cells, image_shape_px=SHAPE, radius_um=radius_um)


# --- the whole way: pixels of a rotated, mirrored camera to stage µm ----------


def _threshold_detector(frame: npt.NDArray[np.uint16]) -> list[Detection]:
    """A stand-in detector: every blob above 1000 counts is one ``single``.

    The box is the extreme pixel indices (``regionprops``' max is exclusive,
    hence the ``- 1``), so its midpoint is the blob's own pixel.
    """
    found = measure.regionprops(measure.label(frame > 1000))
    return [
        Detection(
            float(p.bbox[0]),
            float(p.bbox[1]),
            float(p.bbox[2] - 1),
            float(p.bbox[3] - 1),
            "single",
            0.9,
        )
        for p in found
    ]


def _stage_point(
    origin_um: tuple[float, float], matrix: Matrix2, col: float, row: float
) -> tuple[float, float]:
    """The stage point of pixel ``(col, row)``, written out here from §3.1."""
    d_col, d_row = col - CENTRE[0], row - CENTRE[1]
    (m00, m01), (m10, m11) = matrix
    return (
        origin_um[0] + m00 * d_col + m01 * d_row,
        origin_um[1] + m10 * d_col + m11 * d_row,
    )


@pytest.mark.parametrize(
    ("rotation_deg", "mirrored"),
    [(0.0, False), (30.0, False), (30.0, True), (-12.0, True)],
)
def test_a_rotated_mirrored_matrix_puts_cells_at_their_sample_position(
    rotation_deg: float, mirrored: bool
) -> None:
    matrix = _matrix(1.0, rotation_deg, mirrored)
    origin_0 = (0.0, 0.0)
    # Frame 1 is 288 pixels along the camera's columns: 32 columns overlap.
    origin_1 = (matrix[0][0] * 288.0, matrix[1][0] * 288.0)
    blobs = [
        Blob(*_stage_point(origin, matrix, col, row), radius_um=3.0, intensity=5000.0)
        for origin, col, row in [
            (origin_0, 20, 30),
            (origin_0, 150, 100),
            (origin_0, 300, 60),  # the three of the overlap
            (origin_0, 305, 150),
            (origin_0, 295, 180),
            (origin_1, 250, 100),  # seen by frame 1 only
        ]
    ]
    sample = PlateSample(
        noise_std=0.0,
        texture_std=0.0,
        well_level=0.0,
        plastic_level=0.0,
        blobs=blobs,
    )

    cells: list[Cell] = []
    for index, origin in enumerate([origin_0, origin_1]):
        frame = sample.render(
            *origin,
            0.0,
            shape=SHAPE,
            pixel_size_um=1.0,
            camera_rotation_deg=rotation_deg,
            mirrored=mirrored,
        )
        cells += to_cells(
            _threshold_detector(frame),
            _scan_frame(index, x_um=origin[0], y_um=origin[1]),
            index,
            matrix=matrix,
            image_shape_px=SHAPE,
        )
    kept, removed = _merge(cells)

    assert len(cells) == 9
    assert removed == 3
    assert len(kept) == 6
    for blob in blobs:
        near = [
            c for c in kept if math.hypot(c.x_um - blob.x_um, c.y_um - blob.y_um) < 0.1
        ]
        assert len(near) == 1, (blob.x_um, blob.y_um)


# --- to_cells ----------------------------------------------------------------


def test_to_cells_puts_the_box_centre_at_its_stage_point() -> None:
    frame = _scan_frame(7, well="B2", x_um=100.0, y_um=-50.0)
    shape = (3, 5)
    matrix = pixel_matrix(0.5)
    # The frame centre of a 3 x 5 frame is pixel (col 2, row 1): its own stage point.
    at_centre = _detection(0.0, 1.0, 2.0, 3.0, label="doublet", score=0.8)
    # Two columns right and one row up: +1 µm in X, and +0.5 µm in Y (rows grow
    # downwards, the stage's Y upwards).
    off_centre = _detection(-1.0, 3.0, 1.0, 5.0)

    centre_cell, off_cell = to_cells(
        [at_centre, off_centre], frame, 7, matrix=matrix, image_shape_px=shape
    )

    assert (centre_cell.x_um, centre_cell.y_um) == (100.0, -50.0)
    assert centre_cell.box_px == (0.0, 1.0, 2.0, 3.0)
    assert centre_cell.frame == 7
    assert centre_cell.well == "B2"
    assert (centre_cell.label, centre_cell.score) == ("doublet", 0.8)
    assert (off_cell.x_um, off_cell.y_um) == (101.0, -49.5)


# --- suppress_in_frame -------------------------------------------------------


def test_the_more_confident_label_wins_on_one_object() -> None:
    single = _detection(10, 10, 30, 30, label="single", score=0.6)
    doublet = _detection(10, 10, 30, 30, label="doublet", score=0.9)

    assert suppress_in_frame([single, doublet]) == [doublet]
    # Confidence decides, not size: a small single inside a bigger doublet wins
    # when the model is surer of it.
    sure_single = _detection(15, 15, 25, 25, label="single", score=0.95)
    big_doublet = _detection(10, 10, 30, 30, label="doublet", score=0.6)
    assert suppress_in_frame([big_doublet, sure_single]) == [sure_single]
    # Half of the smaller box is shared: below max_overlap 0.7, so both stay.
    half = _detection(10, 20, 30, 40, label="doublet", score=0.9)
    assert suppress_in_frame([single, half], max_overlap=0.7) == [single, half]


def test_suppression_returns_the_survivors_in_input_order() -> None:
    low = _detection(0, 0, 10, 10, score=0.5)
    apart = _detection(100, 100, 110, 110, score=0.6)
    high = _detection(0, 0, 10, 10, label="doublet", score=0.9)
    # The survivors rank as high, apart; in input order they are apart, high.
    assert suppress_in_frame([low, apart, high]) == [apart, high]


# --- merge_across_fields -----------------------------------------------------


def test_a_cell_seen_by_two_overlapping_fields_is_counted_once() -> None:
    cells = [_cell(0.0, 0), _cell(0.8, 1, y_um=0.3)]

    kept, removed = _merge(cells)

    assert removed == 1
    assert len(kept) == 1


def test_the_least_cut_view_is_kept() -> None:
    near_the_edge = _cell(0.0, 0, centre_px=(SHAPE[1] - 0.5 - 3.0, 99.5), score=0.95)
    uncut = _cell(0.4, 1, centre_px=CENTRE, score=0.6)

    kept, removed = _merge([near_the_edge, uncut])

    assert removed == 1
    assert kept == (uncut,)


def test_a_tie_of_edge_distance_goes_to_the_score_then_the_frame() -> None:
    low_score = _cell(0.0, 0, score=0.7)
    high_score = _cell(0.5, 1, score=0.9)
    assert _merge([low_score, high_score])[0] == (high_score,)

    first = _cell(0.0, 0, score=0.8)
    second = _cell(0.5, 1, score=0.8)
    assert _merge([second, first])[0] == (first,)


def test_two_cells_of_one_frame_are_never_merged() -> None:
    cells = [_cell(0.0, 0), _cell(1.0, 0)]

    kept, removed = _merge(cells)

    assert removed == 0
    assert kept == tuple(cells)


def test_a_box_between_two_cells_of_another_frame_removes_only_one() -> None:
    """Single linkage chains the three into one; the nearest-per-view rule does not."""
    left, right = _cell(0.0, 0), _cell(6.0, 0)
    between = _cell(3.0, 1, centre_px=(30.0, 99.5))

    kept, removed = _merge([left, right, between])

    assert removed == 1
    assert kept == (left, right)


def test_dense_neighbours_seen_by_two_fields_stay_two_cells() -> None:
    """A 0.8 µm calibration error between two fields must not merge neighbours."""
    first = [_cell(0.0, 0), _cell(5.5, 0)]
    second = [
        _cell(0.8, 1, centre_px=(30.0, 99.5)),
        _cell(5.0, 1, centre_px=(30.0, 99.5)),
    ]

    kept, removed = _merge([*first, *second])

    assert removed == 2
    assert kept == tuple(first)


def test_a_neighbour_the_other_field_missed_is_not_swallowed() -> None:
    """A kept view removes one view of another frame, not every cell within the radius."""
    seen_by_both = _cell(0.0, 0)
    same_object = _cell(0.8, 1, centre_px=(30.0, 99.5))
    only_frame_1_found = _cell(4.0, 1, centre_px=(30.0, 99.5))

    kept, removed = _merge([seen_by_both, same_object, only_frame_1_found])

    assert removed == 1
    assert kept == (seen_by_both, only_frame_1_found)


def test_two_equidistant_views_of_another_frame_lose_the_one_first_in_the_input() -> (
    None
):
    """Ties of distance go by input order, so a run and its rerun agree."""
    kept_view = _cell(0.0, 0)
    left = _cell(-3.0, 1, centre_px=(30.0, 99.5))
    right = _cell(3.0, 1, centre_px=(30.0, 99.5))

    assert _merge([kept_view, left, right])[0] == (kept_view, right)
    assert _merge([kept_view, right, left])[0] == (kept_view, left)


def test_a_cell_seen_by_the_four_fields_of_a_corner_is_kept_once() -> None:
    # Four fields meet at a corner; each sees the cell at another distance from
    # its own edge, the first one at the centre of its frame.
    views = [
        _cell(0.0, 0, centre_px=CENTRE),
        _cell(0.2, 1, centre_px=(CENTRE[0], 60.0)),
        _cell(0.4, 2, centre_px=(60.0, CENTRE[1])),
        _cell(0.6, 3, centre_px=(40.0, 40.0)),
    ]

    kept, removed = _merge(views)

    assert removed == 3
    assert kept == (views[0],)


def test_merge_keeps_cells_of_different_wells_apart() -> None:
    cells = [_cell(0.0, 0, well="A1"), _cell(0.0, 1, well="A2")]

    kept, removed = _merge(cells)

    assert removed == 0
    assert kept == tuple(cells)


def test_a_distance_equal_to_the_radius_merges() -> None:
    assert _merge([_cell(0.0, 0), _cell(3.0, 1, y_um=4.0)])[1] == 1
    assert _merge([_cell(0.0, 0), _cell(3.0, 1, y_um=4.01)])[1] == 0


def test_a_merge_radius_of_zero_merges_nothing() -> None:
    cells = [_cell(2.0, 0), _cell(2.0, 1)]

    kept, removed = _merge(cells, radius_um=0.0)

    assert removed == 0
    assert kept == tuple(cells)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_a_merge_radius_that_is_not_a_distance_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="radius_um must be finite and 0 or more"):
        _merge([_cell(0.0, 0)], radius_um=bad)


def test_a_radius_too_small_for_the_grid_still_merges_what_coincides() -> None:
    """``x / radius`` of a stage position must not overflow the grid index."""
    cells = [_cell(90_000.0, 0, y_um=-70_000.0), _cell(90_000.0, 1, y_um=-70_000.0)]

    kept, removed = _merge(cells, radius_um=1e-310)

    assert removed == 1
    assert len(kept) == 1


def test_merge_returns_the_kept_cells_in_input_order() -> None:
    cut = _cell(0.0, 1, centre_px=(30.0, 99.5))
    alone = _cell(50.0, 0)
    uncut = _cell(0.3, 0)

    kept, removed = _merge([cut, alone, uncut])

    assert removed == 1
    assert kept == (alone, uncut)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((4.9, 0.0), (5.1, 0.0)),  # two grid cells of 5 µm apart in X
        ((-0.1, 0.0), (0.1, 0.0)),  # either side of the origin, where floor flips
        ((0.0, 4.9), (0.0, 5.1)),  # and in Y
        ((0.0, -0.1), (0.0, 0.1)),
        ((4.9, 4.9), (5.1, 5.1)),  # across a corner of the grid
    ],
)
def test_two_views_in_neighbouring_grid_cells_are_still_one_object(
    a: tuple[float, float], b: tuple[float, float]
) -> None:
    cells = [_cell(a[0], 0, y_um=a[1]), _cell(b[0], 1, y_um=b[1])]

    assert _merge(cells, radius_um=5.0)[1] == 1


def _merge_by_scan(cells: list[Cell], radius_um: float) -> tuple[tuple[Cell, ...], int]:
    """The rule of ``merge_across_fields`` with no grid: every pair is looked at.

    What the grid hash is an optimisation of, written as plainly as the rule is
    said: walk the least cut view first, keep it, and remove from each other
    frame the nearest cell that is still undecided and within the radius.
    """
    if radius_um == 0.0:
        return tuple(cells), 0
    walk = sorted(
        range(len(cells)),
        key=lambda i: (
            -edge_distance_px(cells[i].box_px, SHAPE),
            -cells[i].score,
            cells[i].frame,
            i,
        ),
    )
    kept: set[int] = set()
    gone: set[int] = set()
    for i in walk:
        if i in gone:
            continue
        kept.add(i)
        nearest: dict[int, tuple[float, int]] = {}
        for j, other in enumerate(cells):
            if j in kept or j in gone:
                continue
            if other.well != cells[i].well or other.frame == cells[i].frame:
                continue
            distance = math.hypot(
                other.x_um - cells[i].x_um, other.y_um - cells[i].y_um
            )
            if distance <= radius_um and (
                other.frame not in nearest or (distance, j) < nearest[other.frame]
            ):
                nearest[other.frame] = (distance, j)
        gone.update(j for _, j in nearest.values())
    return tuple(c for j, c in enumerate(cells) if j not in gone), len(gone)


@given(
    st.lists(
        st.tuples(
            st.integers(0, 3),
            st.floats(0.0, 20.0),
            st.floats(0.0, 20.0),
            st.floats(10.0, 300.0),
            st.floats(10.0, 190.0),
            st.floats(0.05, 1.0),
        ),
        max_size=25,
    ),
    st.floats(0.0, 8.0),
)
def test_the_grid_finds_what_a_scan_of_every_pair_finds(
    drawn: list[tuple[int, float, float, float, float, float]], radius_um: float
) -> None:
    # Frames 0 and 1 are of A1, 2 and 3 of A2: a frame index belongs to one well.
    cells = [
        _cell(
            x,
            frame,
            y_um=y,
            well="A1" if frame < 2 else "A2",
            centre_px=(col, row),
            score=score,
        )
        for frame, x, y, col, row, score in drawn
    ]

    kept, removed = _merge(cells, radius_um)

    reference_kept, reference_removed = _merge_by_scan(cells, radius_um)
    assert removed == reference_removed
    assert [id(cell) for cell in kept] == [id(cell) for cell in reference_kept]
    # And the result is what the rule promises, whatever the reference says: by
    # identity, since two cells can be equal in value and still be two cells.
    survivors = {id(cell) for cell in kept}
    assert removed == len(cells) - len(kept)
    assert [cell for cell in cells if id(cell) in survivors] == list(kept)
    for cell in cells:
        if id(cell) in survivors:
            continue
        assert any(
            k.well == cell.well
            and k.frame != cell.frame
            and math.hypot(k.x_um - cell.x_um, k.y_um - cell.y_um) <= radius_um
            for k in kept
        )


# --- edge_distance_px --------------------------------------------------------


def test_edge_distance_counts_from_the_outer_edge_of_the_edge_pixels() -> None:
    corner = (-1.0, -1.0, 1.0, 1.0)  # a box centred at pixel (0, 0)
    centre_box = (
        CENTRE[1] - 1.0,
        CENTRE[0] - 1.0,
        CENTRE[1] + 1.0,
        CENTRE[0] + 1.0,
    )

    assert edge_distance_px(corner, SHAPE) == 0.5
    # The short axis decides: 100 rows against 160 columns (swapped: 40.0).
    assert edge_distance_px(centre_box, SHAPE) == 100.0


@pytest.mark.parametrize(
    ("centre_px", "expected"),
    [
        ((4.0, 100.0), 4.5),  # left: col + 0.5
        ((310.0, 100.0), 9.5),  # right: W - 0.5 - col
        ((160.0, 7.0), 7.5),  # top: row + 0.5
        ((160.0, 190.0), 9.5),  # bottom: H - 0.5 - row
    ],
)
def test_each_edge_of_the_frame_decides_when_it_is_the_nearest(
    centre_px: tuple[float, float], expected: float
) -> None:
    col, row = centre_px
    box = (row - 1.0, col - 1.0, row + 1.0, col + 1.0)

    assert edge_distance_px(box, SHAPE) == expected
