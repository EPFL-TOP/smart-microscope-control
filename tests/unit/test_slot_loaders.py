"""The loaders of the files a result names (m3 §3.4, §3.5): frames and cells.

Test files are written as bytes (FM-101): a text-mode write would turn a ``\\n``
into ``\\r\\n`` on Windows and a byte-order mark would be a matter of luck.
"""

from __future__ import annotations

import errno
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from smc.plugins.slots import ScanFrame, load_frame
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
