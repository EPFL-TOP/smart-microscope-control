"""``smc.data``: a scan's run folder, loaded with no stand (algorithms design §3, #564).

The run folders are written by ``tests/select_fields_seed.py::seed_run`` with no
Micro-Manager, on a camera of 60 x 80 pixels (not square, FM-73). Every expected
value is written here from the seed or by hand, never read back through the
loader (FM-124).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import tifffile

import smc.data.scan as scan_module
from select_fields_seed import (
    CENTRES,
    PIXEL_UM,
    SHAPE,
    Seed,
    cell_at,
    matrix_of,
    seed_run,
)
from smc.data import DatasetError, Scan, cells_table
from smc.plugins.slots import Cell, ScanFrame
from smc.run import RunDocument

HEIGHT, WIDTH = SHAPE


def _scan_of(seed: Seed, **changes: object) -> Scan:
    """A scan built from the seed's results, with no file read: ``from_results`` checks nothing."""
    arguments: dict[str, object] = {
        "plan": seed.plan,
        "result": seed.scan,
        "cells_result": seed.result,
        "cells": seed.cells,
    }
    arguments.update(changes)
    return Scan.from_results(
        seed.path.parent,
        RunDocument.load(seed.path),
        **arguments,  # type: ignore[arg-type]
    )


def _frame_k(frame: ScanFrame) -> int:
    """The position of a seeded frame in the scan: ``z_um`` is ``1000 + 10 k`` (``seed_run``'s docstring)."""
    assert frame.z_um is not None
    return round((frame.z_um - 1000.0) / 10.0)


def _ramp(k: int) -> np.ndarray:
    """What frame ``k`` holds: a ramp (so that a transposed read shows) offset by ``100 + k``."""
    return (np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(SHAPE) + 100 + k).astype(
        np.uint16
    )


# --- the camera -----------------------------------------------------------------


def test_a_scan_answers_with_the_scans_camera_and_not_the_plans(tmp_path: Path) -> None:
    # FM-124: the plan was laid for another matrix and another frame than the
    # scan took, and the frames were taken with the scan's.
    taken = matrix_of(PIXEL_UM, 30.0, True)
    seed = seed_run(
        tmp_path,
        cells=[],
        matrix=taken,
        plan_matrix=matrix_of(2.0, 0.0, False),
        plan_shape=(100, 100),
    )
    scan = _scan_of(seed)
    assert seed.plan.image_shape_px == (100, 100)  # the oracle: the plan differs
    assert scan.image_shape_px == (HEIGHT, WIDTH)
    assert scan.matrix_um_per_px == taken
    assert scan.pixel_size_um == PIXEL_UM
    assert scan.wells == ("A1",)
    assert scan.geometry["A1"].center_um == pytest.approx(CENTRES["A1"])


# --- frames ---------------------------------------------------------------------


def test_frames_of_a_well_keep_the_scans_order_and_a_typed_well_is_normalised(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"))
    scan = _scan_of(seed)
    in_order = {
        well: tuple(f for f in seed.scan.frames if f.well == well)
        for well in ("A1", "B2")
    }
    assert len(in_order["A1"]) > 1
    assert len(in_order["B2"]) > 1
    assert scan.frames() == seed.scan.frames
    assert scan.frames("A1") == in_order["A1"]
    for typed in ("b2", "B02", " B2 "):
        assert scan.frames(typed) == in_order["B2"], typed
    for stranger in ("Z9", "C3", ""):
        with pytest.raises(DatasetError) as caught:
            scan.frames(stranger)
        assert "(A1, B2)" in str(caught.value), stranger


def test_a_well_of_the_plan_with_no_frame_has_no_frames_and_is_not_an_error(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"))
    only_a1 = tuple(f for f in seed.scan.frames if f.well == "A1")
    scan = _scan_of(seed, result=seed.scan.model_copy(update={"frames": only_a1}))
    assert scan.wells == ("A1", "B2")
    assert scan.frames("B2") == ()
    assert scan.frames("A1") == only_a1


# --- pixels ---------------------------------------------------------------------


def test_image_reads_each_frames_own_pixels(tmp_path: Path) -> None:
    seed = seed_run(
        tmp_path,
        cells=[],
        wells=("A1", "B2"),
        frame_pixels=lambda frame: _ramp(_frame_k(frame)),
    )
    scan = _scan_of(seed)
    assert len(seed.scan.frames) > 2
    for k, frame in enumerate(seed.scan.frames):
        pixels = scan.image(frame)
        assert pixels.dtype == np.uint16
        assert pixels.shape == (HEIGHT, WIDTH)
        np.testing.assert_array_equal(pixels, _ramp(k))


def _break_first_frame(seed: Seed, how: str) -> ScanFrame:
    """Make the first frame of the seed unreadable in the way ``how`` says and return the frame to read."""
    frame = seed.scan.frames[0]
    target = seed.path.parent / frame.file
    if how == "garbage":
        target.write_bytes(b"this is not a tiff file " * 20)
    elif how == "empty":
        target.write_bytes(b"")
    elif how == "escaping":
        return frame.model_copy(update={"file": "../x.tif"})
    return frame


@pytest.mark.parametrize("how", ["missing", "garbage", "empty", "escaping"])
def test_image_of_a_missing_corrupt_empty_or_escaping_frame_is_a_dataset_error_naming_the_frame(
    tmp_path: Path, how: str
) -> None:
    seed = seed_run(tmp_path, cells=[], write_frames=how != "missing")
    frame = _break_first_frame(seed, how)
    scan = _scan_of(seed)
    with pytest.raises(DatasetError) as caught:
        scan.image(frame)
    assert frame.name in str(caught.value)
    assert frame.file in str(caught.value)
    assert caught.value.__cause__ is not None


def test_image_catches_tiffiles_error_by_name_and_not_through_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tifffile's error is a ValueError only from 2026.x, and the floor is 2024.1
    # (#23): on an older release it is its own class, and a read that catches
    # (OSError, ValueError) lets it through. The release the test runs on does
    # not say which, so the error is made the older kind.
    class OldTiffFileError(Exception):
        """tifffile before 2026: not a ValueError."""

    def refuse(run_dir: Path, frame: ScanFrame) -> np.ndarray:
        raise OldTiffFileError("not a TIFF file")

    seed = seed_run(tmp_path, cells=[])
    scan = _scan_of(seed)
    monkeypatch.setattr(tifffile, "TiffFileError", OldTiffFileError)
    monkeypatch.setattr(scan_module, "load_frame", refuse)
    with pytest.raises(DatasetError, match=seed.scan.frames[0].name):
        scan.image(seed.scan.frames[0])


# --- cells ----------------------------------------------------------------------


def _cells_in_two_wells() -> tuple[Cell, Cell, Cell, Cell]:
    def at(well: str, frame: int, dx: float) -> Cell:
        return cell_at(well, dx, 0.0).model_copy(update={"frame": frame})

    return (
        at("A1", 0, 1.0),
        at("A1", 2, 2.0),
        at("B2", 2, 3.0),
        at("B2", 5, 4.0),
    )


def test_cells_of_filters_by_well_and_by_frame_and_both(tmp_path: Path) -> None:
    c0, c1, c2, c3 = _cells_in_two_wells()
    seed = seed_run(tmp_path, cells=[c0, c1, c2, c3], wells=("A1", "B2"))
    scan = _scan_of(seed)
    assert len(seed.scan.frames) > 5
    assert scan.cells_of() == (c0, c1, c2, c3)
    assert scan.cells_of(well="A1") == (c0, c1)
    assert scan.cells_of(well="b2") == (c2, c3)
    assert scan.cells_of(frame=2) == (c1, c2)
    assert scan.cells_of(frame=0) == (c0,)  # frame 0 is a frame, not "no filter"
    assert scan.cells_of(well="B2", frame=2) == (c2,)
    assert scan.cells_of(well="A1", frame=5) == ()
    for bad_frame in (99, len(seed.scan.frames), -1):
        with pytest.raises(DatasetError, match="not in this scan"):
            scan.cells_of(frame=bad_frame)
    with pytest.raises(DatasetError, match=r"\(A1, B2\)"):
        scan.cells_of(well="Z9")


def test_cells_of_a_scan_with_no_cells_raises_instead_of_answering_none_found(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"))
    never_detected = _scan_of(seed, cells=None, cells_result=None)
    assert never_detected.cells is None
    with pytest.raises(DatasetError, match="no cells record"):
        never_detected.cells_of(well="A1")
    with pytest.raises(DatasetError, match="no cells record"):
        never_detected.cells_of()
    why = dataclasses.replace(
        never_detected, cells_absent="the last run of cells ended failed"
    )
    with pytest.raises(DatasetError, match="ended failed"):
        why.cells_of(well="A1")
    # Detected and found none is an answer, and a different one.
    found_none = _scan_of(seed, cells=[])
    assert found_none.cells_of(well="A1") == ()


def test_from_results_builds_a_scan_with_no_file_and_says_what_it_was_given(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"), profile="stand-x")
    path = seed.path.parent
    nothing = RunDocument.new()
    from_memory = Scan.from_results(
        path, nothing, plan=seed.plan, result=seed.scan, cells_result=None, cells=None
    )
    assert from_memory.profile == ""  # unknown, not a guessed name
    assert from_memory.cells is None
    assert from_memory.cells_absent == "no cells record"
    assert from_memory.run_dir == path
    assert from_memory.document is nothing
    none_found = Scan.from_results(
        path, nothing, plan=seed.plan, result=seed.scan, cells_result=None, cells=[]
    )
    assert none_found.cells == ()
    assert none_found.cells_absent is None
    given = [cell_at("A1", 1.0, 2.0), cell_at("B2", 3.0, 4.0)]
    listed = _scan_of(seed, cells=given)
    assert listed.cells == tuple(given)
    assert isinstance(listed.cells, tuple)
    assert listed.profile == "stand-x"  # the document's scan record says
    assert listed.cells_absent is None


def test_a_well_the_plate_does_not_have_is_refused_naming_it(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[])
    stranger = seed.plan.model_copy(update={"wells": ("A1", "Z99")})
    with pytest.raises(DatasetError, match="Z99"):
        _scan_of(seed, plan=stranger)


def test_a_scan_is_frozen_and_its_geometry_is_read_only(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"))
    scan = _scan_of(seed)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scan.profile = "another"  # type: ignore[misc]
    assert isinstance(scan.geometry, MappingProxyType)
    with pytest.raises(TypeError):
        scan.geometry["A1"] = scan.geometry["B2"]  # type: ignore[index]
    assert list(scan.geometry) == ["A1", "B2"]
    assert scan.geometry["B2"].center_um == pytest.approx(CENTRES["B2"])


# --- cells_table ----------------------------------------------------------------


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
