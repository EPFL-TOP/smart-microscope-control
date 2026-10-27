"""The loaders of the files a result names (m3 §3.4, §3.5): frames and cells.

Test files are written as bytes (FM-101): a text-mode write would turn a ``\\n``
into ``\\r\\n`` on Windows and a byte-order mark would be a matter of luck.
"""

from __future__ import annotations

import errno
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from smc.plugins.slots import (
    Cell,
    CellsResult,
    ScanFrame,
    load_cells,
    load_frame,
)
from smc.run import files as files_mod


def _tiff_bytes(image: np.ndarray[Any, Any]) -> bytes:
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, image)
    return buffer.getvalue()


def _frame(file: str) -> ScanFrame:
    return ScanFrame(
        well="A1",
        index=0,
        name="A1_0000",
        file=file,
        x_um=10.0,
        y_um=20.0,
        z_um=None,
        focus="none",
        pfs_offset=None,
        t_s=0.0,
    )


# --- load_frame --------------------------------------------------------------


def test_load_frame_reads_back_the_tiff_a_scan_wrote(tmp_path: Path) -> None:
    # Non-square on purpose (FM-73): a transposed read would show.
    image = np.arange(60, dtype=np.uint16).reshape(6, 10) * 1000
    target = tmp_path / "scan" / "0123456789ab" / "A1_0000.tif"
    target.parent.mkdir(parents=True)
    target.write_bytes(_tiff_bytes(image))
    loaded = load_frame(tmp_path, _frame("scan/0123456789ab/A1_0000.tif"))
    assert loaded.dtype == np.uint16
    assert loaded.shape == (6, 10)
    assert np.array_equal(loaded, image)


def test_load_frame_of_a_missing_file_names_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as raised:
        load_frame(tmp_path, _frame("scan/0123456789ab/A1_0000.tif"))
    assert str(raised.value) == (
        f"frame file scan/0123456789ab/A1_0000.tif is missing from the run folder "
        f"{tmp_path}"
    )
    # A folder is not a frame either.
    (tmp_path / "scan").mkdir()
    with pytest.raises(FileNotFoundError, match="frame file scan is missing"):
        load_frame(tmp_path, _frame("scan"))


def test_load_frame_refuses_a_path_that_leaves_the_run_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "outside.tif").write_bytes(_tiff_bytes(np.zeros((2, 3), np.uint16)))
    read: list[object] = []
    real_imread = tifffile.imread

    def spy(*args: Any, **kwargs: Any) -> Any:
        read.append(args)
        return real_imread(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imread", spy)
    # A ValueError, not a FileNotFoundError: the file exists, it is just not ours.
    with pytest.raises(ValueError, match=r"\A'\.\./outside\.tif' has a '\.\.' part"):
        load_frame(run, _frame("../outside.tif"))
    assert read == []


def test_load_frame_retries_a_read_that_antivirus_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-24: the scan wrote the file a moment ago; Windows may refuse the read.
    image = np.ones((2, 3), np.uint16)
    (tmp_path / "f.tif").write_bytes(_tiff_bytes(image))
    real_imread = tifffile.imread
    refused: list[int] = []
    waits: list[float] = []

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if len(refused) < 2:
            refused.append(1)
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_imread(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imread", flaky)
    monkeypatch.setattr(files_mod.time, "sleep", waits.append)
    assert np.array_equal(load_frame(tmp_path, _frame("f.tif")), image)
    assert (len(refused), waits) == (2, [0.05] * 2)


# --- load_cells --------------------------------------------------------------

CELLS_FILE = "cells/0123456789ab.json"
PREFIX = f"cells file {CELLS_FILE} "


def _cell(well: str, label: str, x_um: float = 1.5) -> dict[str, Any]:
    return {
        "well": well,
        "x_um": x_um,
        "y_um": 2.5,
        "label": label,
        "score": 0.9,
        "frame": 0,
        "box_px": [1.0, 2.0, 30.0, 40.0],
    }


# What the cells stage wrote: A1 has two singles and a doublet, B2 one single.
CELLS = [
    _cell("A1", "single", 1.5),
    _cell("A1", "single", 2.5),
    _cell("A1", "doublet", 3.5),
    _cell("B2", "single", 4.5),
]
# ``counts`` carries every label of ``classes``, zeros included; the file does not.
COUNTS = {
    "A1": {"single": 2, "doublet": 1, "debris": 0},
    "B2": {"single": 1, "doublet": 0, "debris": 0},
    "C3": {"single": 0, "doublet": 0, "debris": 0},
}


def _result(counts: dict[str, dict[str, int]] | None = None) -> CellsResult:
    return CellsResult(
        file=CELLS_FILE,
        model="cell_classes_model-cat-v2.pth",
        model_sha256="0" * 64,
        classes=("single", "doublet", "debris"),
        device="cpu",
        score_threshold=0.5,
        merge_radius_um=8.0,
        frames=4,
        counts=COUNTS if counts is None else counts,
        merged=0,
        seconds_per_frame=None,
    )


def _write(run: Path, content: dict[str, Any] | bytes, *, bom: bool = False) -> Path:
    """The file as the cells stage writes it: ASCII JSON; ``bom`` as Notepad would."""
    data = (
        content
        if isinstance(content, bytes)
        else json.dumps(content, ensure_ascii=True).encode("ascii")
    )
    path = run / CELLS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)
    return path


def _error(run: Path, result: CellsResult | None = None) -> ValueError:
    # Every refusal starts with the file's name, whatever its reason.
    with pytest.raises(ValueError, match=rf"\A{re.escape(PREFIX)}") as raised:
        load_cells(run, result or _result())
    return raised.value


def _refused(run: Path, result: CellsResult | None = None) -> str:
    return str(_error(run, result))


@pytest.mark.parametrize("bom", [False, True], ids=["plain", "with-bom"])
def test_load_cells_reads_back_what_a_cells_stage_wrote(
    tmp_path: Path, bom: bool
) -> None:
    # FM-80: Notepad adds a byte-order mark when a person saves the file.
    _write(tmp_path, {"schema_version": 1, "cells": CELLS}, bom=bom)
    loaded = load_cells(tmp_path, _result())
    assert loaded == tuple(Cell.model_validate(cell) for cell in CELLS)
    assert [cell.x_um for cell in loaded] == [1.5, 2.5, 3.5, 4.5]


def test_load_cells_with_no_cells_matches_counts_of_zeros(tmp_path: Path) -> None:
    _write(tmp_path, {"schema_version": 1, "cells": []})
    zeros = {"A1": {"single": 0, "doublet": 0, "debris": 0}}
    assert load_cells(tmp_path, _result(zeros)) == ()


@pytest.mark.parametrize(
    ("cells", "counts", "where", "have", "want"),
    [
        # One cell removed from the file behind the result.
        (CELLS[1:], COUNTS, "A1/single", 1, 2),
        # One count changed in the result.
        (
            CELLS,
            {**COUNTS, "B2": {"single": 2, "doublet": 0, "debris": 0}},
            "B2/single",
            1,
            2,
        ),
        # A label the counts do not know.
        ([*CELLS, _cell("A1", "mystery")], COUNTS, "A1/mystery", 1, 0),
        # A well whose count is zero, with a cell in the file.
        ([*CELLS, _cell("C3", "single")], COUNTS, "C3/single", 1, 0),
        # Counts that name a cell the file does not hold (a whole well gone).
        (CELLS[:3], COUNTS, "B2/single", 0, 1),
    ],
    ids=["removed", "count-changed", "unknown-label", "zero-count", "well-gone"],
)
def test_load_cells_refuses_a_file_that_differs_from_its_counts(
    tmp_path: Path,
    cells: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
    where: str,
    have: int,
    want: int,
) -> None:
    _write(tmp_path, {"schema_version": 1, "cells": cells})
    assert _refused(tmp_path, _result(counts)) == (
        f"{PREFIX}differs from the result at {where} "
        f"(the file holds {have}, the result counted {want})"
    )


@pytest.mark.parametrize("version", [2, 0, "1", True, 1.0, 1.5, None])
def test_load_cells_refuses_another_schema_version_before_it_validates_the_fields(
    tmp_path: Path, version: object
) -> None:
    # A newer smc's file has fields this one does not know: say it is newer,
    # not that "future_field" is not allowed.
    _write(
        tmp_path,
        {"schema_version": version, "cells": CELLS, "future_field": 1},
    )
    assert _refused(tmp_path) == (
        f"{PREFIX}has schema {version!r}; this smc reads schema 1"
    )


def test_load_cells_names_the_file_it_cannot_read(tmp_path: Path) -> None:
    run = tmp_path
    assert _refused(run) == f"{PREFIX}is missing from the run folder {run}"

    _write(run, b"not json at all")
    assert _refused(run) == (
        f"{PREFIX}is not valid JSON (Expecting value at line 1, column 1)"
    )

    # cp1252 bytes (what Windows Notepad writes as "ANSI") are not UTF-8.
    ansi = '{"schema_version": 1, "cells": [], "note": "caf\xe9"}'.encode("cp1252")
    _write(run, ansi)
    with pytest.raises(UnicodeDecodeError) as decoding:
        ansi.decode("utf-8-sig")
    assert _refused(run) == (
        f"{PREFIX}is not UTF-8 text ({decoding.value.reason} at byte "
        f"{decoding.value.start})"
    )

    _write(run, b'{"schema_version": 1, "cells": [{"score": NaN}]}')
    assert _refused(run) == f"{PREFIX}contains NaN, which JSON does not allow"

    _write(run, b"[]")
    assert _refused(run) == f"{PREFIX}is not a JSON object"


def test_load_cells_chains_the_cause_of_a_failure_that_has_one(tmp_path: Path) -> None:
    # What went wrong underneath stays on the traceback for whoever debugs it.
    assert isinstance(_error(tmp_path).__cause__, FileNotFoundError)
    _write(tmp_path, b"not json")
    assert isinstance(_error(tmp_path).__cause__, json.JSONDecodeError)
    _write(tmp_path, b"\xe9")
    assert isinstance(_error(tmp_path).__cause__, UnicodeDecodeError)
    broken = {**CELLS[0], "score": 1.5}
    _write(tmp_path, {"schema_version": 1, "cells": [broken]})
    assert isinstance(_error(tmp_path).__cause__, ValidationError)


def test_load_cells_says_which_field_does_not_validate(tmp_path: Path) -> None:
    broken = {**CELLS[0], "score": 1.5}
    _write(tmp_path, {"schema_version": 1, "cells": [broken, broken]})
    assert _refused(tmp_path) == (
        f"{PREFIX}does not validate: cells.0.score: Input should be less than or "
        f"equal to 1 (and 1 more)"
    )


def test_load_cells_refuses_a_path_that_leaves_the_run_folder(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "outside.json").write_bytes(b'{"schema_version": 1, "cells": []}')
    result = _result({}).model_copy(update={"file": "../outside.json"})
    message = (
        "cells file ../outside.json cannot be used: '../outside.json' has a '..' "
        "part; run-folder paths stay inside the run folder"
    )
    with pytest.raises(ValueError, match=rf"\A{re.escape(message)}\Z"):
        load_cells(run, result)


def test_load_cells_names_a_folder_where_the_file_should_be(tmp_path: Path) -> None:
    (tmp_path / CELLS_FILE).mkdir(parents=True)
    # IsADirectoryError on POSIX, PermissionError on Windows: either way it is named.
    assert _refused(tmp_path).startswith(f"{PREFIX}cannot be read (")


def test_load_cells_retries_a_read_that_antivirus_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, {"schema_version": 1, "cells": CELLS})
    real_read = Path.read_bytes
    refused: list[int] = []
    waits: list[float] = []

    def flaky(self: Path) -> bytes:
        if len(refused) < 2:
            refused.append(1)
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    monkeypatch.setattr(files_mod.time, "sleep", waits.append)
    assert len(load_cells(tmp_path, _result())) == 4
    assert (len(refused), waits) == (2, [0.05] * 2)
