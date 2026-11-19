"""``smc.data``: a scan's run folder, loaded with no stand (algorithms design §3, #564).

The run folders are written by ``tests/select_fields_seed.py::seed_run`` with no
Micro-Manager, on a camera of 60 x 80 pixels (not square, FM-73). Every expected
value is written here from the seed or by hand, never read back through the
loader (FM-124).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
    record,
    seed_run,
)
from smc.data import DatasetError, Scan, cells_table, load_scan
from smc.plugins.slots import Cell, CellsFile, ScanFrame
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
    # The seed writes the same pixel size in the plan and the scan whatever the
    # matrix says, so the plan's is made another by hand.
    plan = seed.plan.model_copy(update={"pixel_size_um": 2.0})
    scan = _scan_of(seed, plan=plan)
    assert seed.plan.image_shape_px == (100, 100)  # the oracle: the plan differs
    assert scan.plan.pixel_size_um == 2.0
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
    elif how == "header_only":
        # What a power cut leaves: the TIFF header and nothing after it.
        target.write_bytes(target.read_bytes()[:8])
    elif how == "header_and_zeros":
        # NTFS keeps the file's length and zero-fills what was never flushed.
        target.write_bytes(target.read_bytes()[:8] + bytes(1000))
    elif how == "three_bytes":
        target.write_bytes(target.read_bytes()[:3])
    elif how == "escaping":
        return frame.model_copy(update={"file": "../x.tif"})
    return frame


@pytest.mark.parametrize(
    "how",
    [
        "missing",
        "garbage",
        "empty",
        "escaping",
        "header_only",
        "header_and_zeros",
        "three_bytes",
    ],
)
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
    # A file cut after its header: tifffile 2026 reads an empty array and says
    # nothing, so the shape finds it and there is no cause; an older release may
    # raise instead. Either way the error says what was wrong.
    if how in ("header_only", "header_and_zeros") and caught.value.__cause__ is None:
        assert "(60, 80)" in str(caught.value)
    else:
        assert caught.value.__cause__ is not None


@pytest.mark.parametrize("shape", [(WIDTH, HEIGHT), (10, 10), (HEIGHT, WIDTH, 3)])
def test_image_of_a_frame_that_is_not_the_scans_shape_is_a_dataset_error_naming_both(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    # FM-73: a frame written transposed is the right size and the wrong shape.
    seed = seed_run(tmp_path, cells=[])
    frame = seed.scan.frames[0]
    tifffile.imwrite(tmp_path / frame.file, np.zeros(shape, dtype=np.uint16))
    with pytest.raises(DatasetError) as caught:
        _scan_of(seed).image(frame)
    text = str(caught.value)
    assert frame.name in text
    assert str(shape) in text
    assert str((HEIGHT, WIDTH)) in text


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


# --- load_scan ------------------------------------------------------------------


def _edit(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    """Change ``run.json`` as a hand edit, a crash or a later run would have left it."""
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _refusal(path: Path | str) -> str:
    """The text of the ``DatasetError`` that ``load_scan`` raises for ``path``."""
    with pytest.raises(DatasetError) as caught:
        load_scan(path)
    return str(caught.value)


def test_load_scan_reads_the_wells_the_camera_and_the_frames_a_scan_wrote(
    tmp_path: Path,
) -> None:
    taken = matrix_of(PIXEL_UM, 30.0, True)
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"), matrix=taken)
    scan = load_scan(seed.path)
    assert scan.wells == ("A1", "B2")
    assert scan.profile == "demo"
    assert scan.run_dir == tmp_path
    assert scan.image_shape_px == (HEIGHT, WIDTH)
    assert scan.matrix_um_per_px == seed.matrix == taken
    assert scan.pixel_size_um == PIXEL_UM
    assert scan.frames() == seed.scan.frames
    assert scan.frames("B2") == tuple(f for f in seed.scan.frames if f.well == "B2")
    assert scan.plan == seed.plan
    assert scan.result == seed.scan
    assert scan.geometry["A1"].center_um == pytest.approx(CENTRES["A1"])
    assert scan.geometry["B2"].center_um == pytest.approx(CENTRES["B2"])
    assert list(scan.document.stages) == [
        "plate-registration",
        "scan-plan",
        "scan",
        "cells",
    ]


def test_load_scan_takes_a_path_as_text_and_as_a_path(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[])
    assert load_scan(str(seed.path)) == load_scan(seed.path)


def test_load_scan_takes_the_scans_camera_and_not_the_plans(tmp_path: Path) -> None:
    # FM-124: the plan was laid for another matrix and another frame.
    seed = seed_run(
        tmp_path,
        cells=[],
        plan_matrix=matrix_of(2.0, 0.0, False),
        plan_shape=(100, 100),
    )
    _edit(
        seed.path,
        lambda data: data["stages"]["scan-plan"]["result"].update(pixel_size_um=2.0),
    )
    scan = load_scan(seed.path)
    assert scan.plan.pixel_size_um == 2.0  # the oracle: the plan differs
    assert scan.pixel_size_um == PIXEL_UM
    assert scan.plan.image_shape_px == (100, 100)
    assert scan.image_shape_px == (HEIGHT, WIDTH)
    assert scan.matrix_um_per_px == seed.matrix
    assert scan.matrix_um_per_px != seed.plan.matrix_um_per_px


def test_a_well_of_the_plan_whose_frames_are_all_missing_loads_with_none(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"))
    only_a1 = tuple(f for f in seed.scan.frames if f.well == "A1")
    record(seed.path, "scan", "scan", seed.scan.model_copy(update={"frames": only_a1}))
    scan = load_scan(seed.path)
    assert scan.wells == ("A1", "B2")
    assert scan.frames("B2") == ()
    assert scan.frames("A1") == only_a1


def test_cells_come_with_their_result_and_in_the_files_order(tmp_path: Path) -> None:
    cells = [
        cell_at("B2", 5.0, 1.0, "doublet"),
        cell_at("A1", 3.0, 2.0),
        cell_at("B2", 1.0, 1.0, "debris"),
        cell_at("A1", -4.0, 0.5),
    ]
    seed = seed_run(tmp_path, cells=cells, wells=("A1", "B2"))
    scan = load_scan(seed.path)
    assert scan.cells == tuple(cells)  # the file's order, not sorted by well
    assert scan.cells == seed.cells
    assert scan.cells_result == seed.result
    assert scan.cells_absent is None


def _no_cells_record(data: dict[str, Any]) -> None:
    del data["stages"]["cells"]


def _cells_ended_failed(data: dict[str, Any]) -> None:
    data["stages"]["cells"]["status"] = "failed"


def _cells_on_another_stand(data: dict[str, Any]) -> None:
    data["stages"]["cells"]["profile"] = "another"


def _cells_were_a_dry_run(data: dict[str, Any]) -> None:
    data["stages"]["cells"]["dry_run"] = True


def _cells_result_is_not_one(data: dict[str, Any]) -> None:
    data["stages"]["cells"]["result"] = {"file": "cells/t1.json"}


@pytest.mark.parametrize(
    ("how", "reason"),
    [
        (_no_cells_record, "no cells record"),
        (_cells_ended_failed, "ended failed"),
        (_cells_on_another_stand, "was run on another"),
        (_cells_were_a_dry_run, "dry run"),
        (_cells_result_is_not_one, "not a cells result this smc reads"),
    ],
)
def test_cells_are_none_with_the_reason_when_there_is_no_usable_cells_record(
    tmp_path: Path, how: Callable[[dict[str, Any]], None], reason: str
) -> None:
    seed = seed_run(tmp_path, cells=[cell_at("A1", 3.0, 2.0)])
    _edit(seed.path, how)
    scan = load_scan(seed.path)
    assert scan.cells is None
    assert scan.cells_result is None
    assert scan.cells_absent is not None
    assert reason in scan.cells_absent
    # The scan itself is fine: its frames load, and its pixels read.
    assert scan.frames() == seed.scan.frames
    assert scan.image(scan.frames()[0]).shape == (HEIGHT, WIDTH)


def test_cells_are_none_when_the_cells_file_is_missing_or_does_not_match_its_result(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[cell_at("A1", 3.0, 2.0)], file_token="gone")
    kept = (tmp_path / seed.result.file).read_bytes()
    (tmp_path / seed.result.file).unlink()
    scan = load_scan(seed.path)
    assert scan.cells is None
    assert scan.cells_result is None
    assert scan.cells_absent is not None
    assert "missing" in scan.cells_absent
    assert seed.result.file in scan.cells_absent
    # A file the result did not count: one cell edited out of it.
    (tmp_path / seed.result.file).write_bytes(
        CellsFile(cells=()).model_dump_json().encode("utf-8")
    )
    scan = load_scan(seed.path)
    assert scan.cells is None
    assert scan.cells_absent is not None
    assert "differs from the result" in scan.cells_absent
    # And the file as it was is read again.
    (tmp_path / seed.result.file).write_bytes(kept)
    assert load_scan(seed.path).cells == seed.cells


def _no_scan(data: dict[str, Any]) -> None:
    del data["stages"]["scan"]


def _scan_stopped(data: dict[str, Any]) -> None:
    data["stages"]["scan"]["status"] = "stopped"


def _scan_running(data: dict[str, Any]) -> None:
    data["stages"]["scan"]["status"] = "running"


def _scan_failed(data: dict[str, Any]) -> None:
    data["stages"]["scan"]["status"] = "failed"


def _scan_stale(data: dict[str, Any]) -> None:
    # The scan read the plan's first attempt, and the plan was run again.
    data["stages"]["scan"]["reads"] = {
        "scan-plan": {"plugin": "scan-plan", "attempt": "a1"}
    }
    data["stages"]["scan-plan"]["attempt"] = "a2"


def _plan_failed(data: dict[str, Any]) -> None:
    data["stages"]["scan-plan"]["status"] = "failed"


def _plan_on_another_stand(data: dict[str, Any]) -> None:
    data["stages"]["scan-plan"]["profile"] = "another"


def _scan_result_is_not_one(data: dict[str, Any]) -> None:
    data["stages"]["scan"]["result"] = {"folder": "scan/t"}


def _scan_has_no_result(data: dict[str, Any]) -> None:
    data["stages"]["scan"]["result"] = None


def _plan_result_is_not_one(data: dict[str, Any]) -> None:
    data["stages"]["scan-plan"]["result"] = {"wells": ["A1"]}


def _plan_has_no_result(data: dict[str, Any]) -> None:
    data["stages"]["scan-plan"]["result"] = None


def _scan_of_another_plugin_with_no_usable_result(data: dict[str, Any]) -> None:
    # A slot is filled by whichever plugin provides it: the refusal says to run
    # that plugin again, which is not always the slot's name.
    record = data["stages"].pop("scan")
    record["plugin"] = "scan-by-hand"
    record["result"] = {"folder": "scan/t"}
    data["stages"]["scan-by-hand"] = record


@pytest.mark.parametrize(
    ("how", "reasons"),
    [
        (_no_scan, ["no scan"]),
        (_scan_stopped, ["ended stopped"]),
        (_scan_running, ["interrupted"]),
        (_scan_failed, ["ended failed"]),
        (_scan_stale, ["earlier run"]),
        (_plan_failed, ["scan-plan", "ended failed"]),
        (_plan_on_another_stand, ["scan-plan was run on another"]),
        (
            _scan_result_is_not_one,
            ["not a scan result this smc reads", "run scan again"],
        ),
        (_scan_has_no_result, ["not a scan result this smc reads", "no result"]),
        (
            _plan_result_is_not_one,
            [
                "the scan-plan record is not a scan-plan result this smc reads",
                "(plate_plan: Field required)",
                "run scan-plan again",
            ],
        ),
        (
            _plan_has_no_result,
            ["not a scan-plan result this smc reads", "(it holds no result)"],
        ),
        (
            _scan_of_another_plugin_with_no_usable_result,
            ["the scan record is not a scan result", "run scan-by-hand again"],
        ),
    ],
)
def test_load_scan_refuses_what_it_cannot_use_naming_the_file_and_the_reason(
    tmp_path: Path, how: Callable[[dict[str, Any]], None], reasons: list[str]
) -> None:
    seed = seed_run(tmp_path, cells=[])
    _edit(seed.path, how)
    text = _refusal(seed.path)
    assert text.startswith(f"{seed.path}: ")
    for reason in reasons:
        assert reason in text


def test_load_scan_refuses_a_dry_run_scan(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[], dry_run=True)
    text = _refusal(seed.path)
    assert str(seed.path) in text
    assert "dry run" in text


def test_load_scan_refuses_a_document_it_cannot_read_naming_the_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "nothing.json"
    assert _refusal(missing).startswith(str(missing))
    assert "does not exist" in _refusal(missing)

    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert str(broken) in _refusal(broken)
    assert "not valid JSON" in _refusal(broken)

    newer = tmp_path / "newer.json"
    newer.write_text(
        json.dumps(
            {"schema_version": 2, "created_utc": "2026-11-19T09:00:00Z", "stages": {}}
        ),
        encoding="utf-8",
    )
    assert str(newer) in _refusal(newer)
    assert "newer smc" in _refusal(newer)

    folder = tmp_path / "a folder"
    folder.mkdir()
    assert _refusal(folder) == (
        f"{folder}: is a folder, not a run document; name a file"
    )
    seed = seed_run(folder, cells=[])
    hint = _refusal(folder)
    assert hint.startswith(f"{folder}: is a folder")
    assert f"did you mean {seed.path}?" in hint
    assert seed.path == folder / "run.json"


def test_an_unreadable_document_is_not_an_empty_scan(tmp_path: Path) -> None:
    # FM-229: a document that cannot be read says so; it is never a document
    # with no stage, which would read as "no scan" and send the owner to run one.
    path = tmp_path / "run.json"
    path.write_bytes(b"\xff\xfe\x00")
    text = _refusal(path)
    assert "not UTF-8" in text
    assert "no scan" not in text


def test_a_folder_that_cannot_be_looked_at_is_a_dataset_error_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-24: "Access is denied" on a share or another user's folder. Path.is_dir
    # ignores only a missing path before Python 3.14 and lets the rest through, so
    # the probe that looks for a folder must not end the load with a traceback.
    # Patched, not chmod: a read-only folder proves nothing on Windows.
    seed = seed_run(tmp_path, cells=[])
    original = Path.is_dir

    def is_dir(self: Path) -> bool:
        if self == seed.path:
            raise PermissionError(13, "Access is denied")
        return original(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    text = _refusal(seed.path)
    assert text.startswith(f"{seed.path}: ")
    assert "could not be read" in text
    assert "Access is denied" in text


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
    columns = [name for name in COLUMNS if table[name].dtype == np.float64]
    assert len(columns) == 9
    for first in columns:
        assert table[first].flags.c_contiguous, first
        for second in columns:
            if first != second:
                assert not np.shares_memory(table[first], table[second]), (
                    first,
                    second,
                )
    table["row0_px"][0] = -1.0
    assert table["row_px"][0] == 20.0
    assert table["row1_px"][0] == 30.0


def test_a_dataset_error_is_a_value_error() -> None:
    # A caller that already handles a bad value handles an unreadable dataset
    # too, and #565, #567 and #568 raise it for their own refusals.
    assert issubclass(DatasetError, ValueError)


def test_cells_table_is_a_pandas_data_frame_input() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(cells_table(_three_cells()))
    assert frame.shape == (3, 12)
    assert list(frame.columns) == COLUMNS
