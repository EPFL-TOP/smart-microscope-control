"""The cells stage's algorithm: pure functions on detections, cells and numbers (m3 §9.2).

It has no model, no file and no microscope (ADR-0004): the loop that feeds it
frames is ``plugin.py``, and it is tested on synthetic detections and on the
synthetic plate. Boxes are ``(row0, col0, row1, col1)`` and a shape is
``(height, width)``; the stage position of a pixel is ``smc.plugins.geometry``'s
and nothing here re-derives a sign (FM-73).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from smc.plugins.cells.detector import Detection
from smc.plugins.cells.overlap import (
    DEFAULT_MAX_OVERLAP,
    OverlapCandidate,
    select_survivors,
)
from smc.plugins.geometry import Matrix2, stage_of_pixel
from smc.plugins.slots import Cell, ScanFrame

__all__ = [
    "edge_distance_px",
    "merge_across_fields",
    "suppress_in_frame",
    "to_cells",
]

#: The grid of the merge is at least this fine, in µm. A radius below it is a
#: cell size smaller than a pixel by orders of magnitude, and ``x / radius`` of a
#: stage position would stop being a number; the distance test still uses the
#: radius as given.
_MIN_GRID_CELL_UM = 1e-3

_UNDECIDED, _KEPT, _REMOVED = 0, 1, 2


def suppress_in_frame(
    detections: Sequence[Detection], *, max_overlap: float = DEFAULT_MAX_OVERLAP
) -> list[Detection]:
    """One detection per object of a frame: the model's per-class NMS can return one cell twice.

    The model suppresses overlapping boxes class by class, so one object can come
    back as a ``single`` and again as a ``doublet``. The boxes are compared by the
    intersection over the smaller one (a ``single`` inside a ``doublet`` is a
    containment that IoU would miss) and the more confident label wins. The
    survivors are returned in the order they came in, not in rank order.
    """
    candidates = [
        OverlapCandidate(
            ref=index,
            bbox=(d.row0_px, d.col0_px, d.row1_px, d.col1_px),
            score=d.score,
        )
        for index, d in enumerate(detections)
    ]
    kept, _ = select_survivors(candidates, max_overlap, prefer="score")
    return [detections[i] for i in sorted(i for i in kept if isinstance(i, int))]


def to_cells(
    detections: Sequence[Detection],
    frame: ScanFrame,
    frame_index: int,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> list[Cell]:
    """The detections of one frame as cells in stage µm, through the scan's matrix.

    A cell is at the midpoint of its box, as the detector gave it: there is no
    half-pixel correction (at most 0.14 µm at 40x, and the training's box
    convention is not known). ``frame_index`` is the frame's position in
    ``ScanResult.frames``, which the cells file stores so that the report can
    find the frame again.
    """
    cells: list[Cell] = []
    for d in detections:
        x_um, y_um = stage_of_pixel(
            frame.x_um,
            frame.y_um,
            (d.col0_px + d.col1_px) / 2.0,
            (d.row0_px + d.row1_px) / 2.0,
            matrix=matrix,
            image_shape_px=image_shape_px,
        )
        cells.append(
            Cell(
                well=frame.well,
                x_um=x_um,
                y_um=y_um,
                label=d.label,
                score=d.score,
                frame=frame_index,
                box_px=(d.row0_px, d.col0_px, d.row1_px, d.col1_px),
            )
        )
    return cells


def edge_distance_px(
    box_px: tuple[float, float, float, float], image_shape_px: tuple[int, int]
) -> float:
    """How far the box's centre is from the nearest outer edge of its frame, in pixels.

    The edge is the outer edge of the edge pixels (``-0.5`` and ``W - 0.5`` in
    pixel-centre coordinates), so a box centred on the first pixel is 0.5 from
    it. The least cut view of a cell is the one that is farthest from every
    edge; the shorter side of a non-square frame decides.
    """
    row0, col0, row1, col1 = box_px
    height, width = image_shape_px
    col_centre = (col0 + col1) / 2.0
    row_centre = (row0 + row1) / 2.0
    return min(
        col_centre + 0.5,
        width - 0.5 - col_centre,
        row_centre + 0.5,
        height - 0.5 - row_centre,
    )


def merge_across_fields(
    cells: Sequence[Cell],
    *,
    image_shape_px: tuple[int, int],
    radius_um: float,
) -> tuple[tuple[Cell, ...], int]:
    """Count a cell once when two overlapping fields both saw it.

    Cells of the same well and of **different** frames, within ``radius_um``
    (inclusive) of each other, are views of one object. The cells are walked
    from the least cut view (the box centre farthest from its frame's edge, then
    the higher score, then the lower frame, then the input order); one not yet
    removed is kept, and for each other frame the **nearest** cell of that frame
    that is still undecided is removed. A cell that was kept is never removed.

    Why not the connected components of "within the radius" (single linkage,
    union-find): they chain. One box of frame 1 between two cells of frame 0
    left one cell for two, and two neighbours seen twice, with a 0.8 µm
    calibration error between the fields, lost one of the two (measured while
    planning #23). Each kept view here removes at most one view of each other
    frame, so a neighbour in the same frame survives. Two cells of one frame are
    never merged: ``suppress_in_frame`` already decided that.

    Neighbours come from a grid of cells of ``radius_um`` keyed by well, so a
    well of thousands of cells is not compared pairwise; the 3 x 3 block around a
    cell holds every cell within the radius.

    Returns:
        The kept cells in the order they came in, and the number removed.

    Raises:
        ValueError: ``radius_um`` is negative or not finite.
    """
    if not (math.isfinite(radius_um) and radius_um >= 0.0):
        raise ValueError(f"radius_um must be finite and 0 or more, got {radius_um!r}")
    if radius_um == 0.0:
        return tuple(cells), 0

    cell_um = max(radius_um, _MIN_GRID_CELL_UM)
    grid: defaultdict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        grid[_bin(cell, cell_um)].append(index)

    edge = [edge_distance_px(cell.box_px, image_shape_px) for cell in cells]
    walk = sorted(
        range(len(cells)),
        key=lambda i: (-edge[i], -cells[i].score, cells[i].frame, i),
    )
    state = [_UNDECIDED] * len(cells)
    removed = 0
    for index in walk:
        if state[index] == _REMOVED:
            continue
        state[index] = _KEPT
        cell = cells[index]
        well, bin_x, bin_y = _bin(cell, cell_um)
        # Per other frame, the nearest undecided cell: (distance, input index).
        nearest: dict[int, tuple[float, int]] = {}
        for near_x in (bin_x - 1, bin_x, bin_x + 1):
            for near_y in (bin_y - 1, bin_y, bin_y + 1):
                for other in grid.get((well, near_x, near_y), ()):
                    candidate = cells[other]
                    if state[other] != _UNDECIDED or candidate.frame == cell.frame:
                        continue
                    distance = math.hypot(
                        candidate.x_um - cell.x_um, candidate.y_um - cell.y_um
                    )
                    if distance > radius_um:
                        continue
                    best = nearest.get(candidate.frame)
                    if best is None or (distance, other) < best:
                        nearest[candidate.frame] = (distance, other)
        for _, other in nearest.values():
            state[other] = _REMOVED
            removed += 1
    return tuple(c for i, c in enumerate(cells) if state[i] != _REMOVED), removed


def _bin(cell: Cell, cell_um: float) -> tuple[str, int, int]:
    """The grid cell of a cell: its well and the floors of its position over the cell size."""
    return (cell.well, math.floor(cell.x_um / cell_um), math.floor(cell.y_um / cell_um))
