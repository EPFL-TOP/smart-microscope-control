"""``smc.data``: a scan's run folder, loaded with no stand (algorithms design §3, #564).

The run folders are written by ``tests/select_fields_seed.py::seed_run`` with no
Micro-Manager, on a camera of 60 x 80 pixels (not square, FM-73). Every expected
value is written here from the seed or by hand, never read back through the
loader (FM-124).
"""

from __future__ import annotations

import numpy as np
import pytest

from smc.data import cells_table
from smc.plugins.slots import Cell

# --- cells_table ---------------------------------------------------------------


def _three_cells() -> tuple[Cell, Cell, Cell]:
    """Three cells whose every field differs; the first box is not square, so its centre's two axes differ (FM-73)."""
    return (
        Cell(
            well="A1",
            x_um=1.5,
            y_um=-2.5,
            label="single",
            score=0.9,
            frame=0,
            box_px=(10.0, 20.0, 30.0, 60.0),
        ),
        Cell(
            well="B2",
            x_um=100.25,
            y_um=-200.5,
            label="doublet",
            score=0.5,
            frame=3,
            box_px=(0.0, 5.0, 8.0, 9.0),
        ),
        Cell(
            well="A1",
            x_um=-7.0,
            y_um=8.0,
            label="debris",
            score=0.25,
            frame=1,
            box_px=(100.0, 200.0, 140.0, 260.0),
        ),
    )


COLUMNS = [
    "well",
    "label",
    "frame",
    "score",
    "x_um",
    "y_um",
    "row_px",
    "col_px",
    "row0_px",
    "col0_px",
    "row1_px",
    "col1_px",
]


def test_cells_table_has_the_columns_of_the_design_in_the_inputs_order() -> None:
    table = cells_table(_three_cells())
    assert list(table) == COLUMNS
    expected = {
        "well": ["A1", "B2", "A1"],
        "label": ["single", "doublet", "debris"],
        "frame": [0, 3, 1],
        "score": [0.9, 0.5, 0.25],
        "x_um": [1.5, 100.25, -7.0],
        "y_um": [-2.5, -200.5, 8.0],
        # The box is (row0, col0, row1, col1): the centre's row is the mean of
        # the first and the third number, its column of the second and the fourth.
        "row_px": [20.0, 4.0, 120.0],
        "col_px": [40.0, 7.0, 230.0],
        "row0_px": [10.0, 0.0, 100.0],
        "col0_px": [20.0, 5.0, 200.0],
        "row1_px": [30.0, 8.0, 140.0],
        "col1_px": [60.0, 9.0, 260.0],
    }
    for name in COLUMNS:
        assert table[name].tolist() == expected[name], name


def test_cells_table_dtypes_and_the_empty_table() -> None:
    full = cells_table(_three_cells())
    assert full["frame"].dtype == np.int64
    assert full["well"].dtype.kind == "U"
    assert full["label"].dtype.kind == "U"
    for name in COLUMNS[3:]:
        assert full[name].dtype == np.float64, name
    empty = cells_table(())
    assert list(empty) == COLUMNS
    for name in COLUMNS:
        assert empty[name].shape == (0,), name
        # The kind, not the width: an empty string array is <U1 and a full one <U7.
        assert empty[name].dtype.kind == full[name].dtype.kind, name
    assert empty["frame"].dtype == np.int64


def test_cells_table_columns_do_not_share_memory() -> None:
    # A caller that edits one column of the table must not edit another: the box
    # columns are cut from one array and each is a copy of its own.
    table = cells_table(_three_cells())
    table["row0_px"][0] = -1.0
    assert table["row_px"][0] == 20.0
    assert table["row1_px"][0] == 30.0


def test_cells_table_is_a_pandas_data_frame_input() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(cells_table(_three_cells()))
    assert frame.shape == (3, 12)
    assert list(frame.columns) == COLUMNS
