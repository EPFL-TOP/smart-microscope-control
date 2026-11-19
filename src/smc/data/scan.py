"""A scan's run folder, loaded with no hardware (algorithms design §3).

A collaborator develops a selection algorithm on a laptop that has no
Micro-Manager and no stand: this module reads what a scan wrote (the plan, the
frames and their records, the detected cells) from a run folder and hands it
over as models, arrays and columns. It imports no hardware layer, and
``tests/test_kit_imports.py`` keeps it so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from smc.plugins.slots import Cell

__all__ = ["DatasetError", "cells_table"]


class DatasetError(ValueError):
    """A run folder or dataset that cannot be read; the message names the file and why.

    A ``ValueError`` so that a caller that already handles a bad value handles
    this too, and one class for every refusal of this package, so that a tool
    built on it catches one thing and prints the message as it is.
    """


def cells_table(cells: Sequence[Cell]) -> dict[str, NDArray[Any]]:
    """The cells as columns, one entry per cell, in the input's order.

    The keys are, in this order: ``well`` and ``label`` (``str``), ``frame``
    (``int64``, the index into the scan's frames), ``score``, ``x_um``,
    ``y_um``, ``row_px`` and ``col_px`` (the box's centre, in the frame's
    pixels), then ``row0_px``, ``col0_px``, ``row1_px`` and ``col1_px`` (the
    box, as ``Cell.box_px`` holds it: ``(row0, col0, row1, col1)``); all
    ``float64`` but the first three. ``pandas.DataFrame(cells_table(cells))``
    is the same table, and pandas is not a dependency of smc.

    An empty input gives twelve empty arrays of the same dtypes (``well`` and
    ``label`` as ``<U1``): a table that has columns and no row, never ``{}``.
    """
    boxes = np.array([cell.box_px for cell in cells], dtype=np.float64).reshape(
        len(cells), 4
    )
    row0, col0, row1, col1 = (np.ascontiguousarray(boxes[:, k]) for k in range(4))
    return {
        "well": np.array([cell.well for cell in cells], dtype=str),
        "label": np.array([cell.label for cell in cells], dtype=str),
        "frame": np.array([cell.frame for cell in cells], dtype=np.int64),
        "score": np.array([cell.score for cell in cells], dtype=np.float64),
        "x_um": np.array([cell.x_um for cell in cells], dtype=np.float64),
        "y_um": np.array([cell.y_um for cell in cells], dtype=np.float64),
        "row_px": (row0 + row1) / 2.0,
        "col_px": (col0 + col1) / 2.0,
        "row0_px": row0,
        "col0_px": col0,
        "row1_px": row1,
        "col1_px": col1,
    }
