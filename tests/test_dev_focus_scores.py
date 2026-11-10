"""``scripts/dev/focus_scores.py`` on a stack ``focus-on-cells`` wrote on the demo-sample (m3 §21.8)."""

from __future__ import annotations

import csv
import importlib.util
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import tifffile

from smc.hardware import Microscope
from smc.hardware.capabilities import ZStage
from smc.hardware.profile import Profile
from smc.plugins import run_plugin
from smc.plugins.cell_focus.plugin import FocusOnCellsParams, FocusOnCellsPlugin
from smc.plugins.slots import CellFocusResult
from smc.run import StageStatus

pytestmark = pytest.mark.demo

ROOT = Path(__file__).resolve().parents[1]
DEMO_SAMPLE = ROOT / "profiles" / "demo-sample.toml"


@pytest.fixture(scope="module")
def fs() -> ModuleType:
    """The script, loaded as ``tests/test_dev_point.py`` loads ``point.py``."""
    path = ROOT / "scripts" / "dev" / "focus_scores.py"
    spec = importlib.util.spec_from_file_location("focus_scores_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stack(mm_available: bool, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stack of 11 planes, 1 µm apart around Z = 2 µm, that the tool kept; the demo is sharp at 0."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path_factory.mktemp("focus") / "run.json"
    stand = Microscope.open(Profile.load(DEMO_SAMPLE))
    try:
        stand.require(ZStage).move_to_um(2.0)
        record = run_plugin(FocusOnCellsPlugin(), stand, FocusOnCellsParams(), run_doc)
    finally:
        stand.close()
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    result = CellFocusResult.model_validate(record.result)
    return run_doc.parent / result.folder


def test_a_stack_the_tool_kept_is_scored_and_peaks_near_the_sharp_plane(
    fs: ModuleType,
    stack: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_csv = tmp_path / "scores.csv"

    code = fs.main([str(stack), "--csv", str(out_csv)])

    assert code == 0
    printed = capsys.readouterr().out
    rows = re.findall(r"^\s+(-?\d+\.\d\d)\s+\S+\s+\S+\s+\S+\s+\S+$", printed, re.M)
    assert [float(z) for z in rows] == pytest.approx(
        [2.0 + k for k in range(-5, 6)], abs=0.01
    )
    brenner = re.search(
        r"brenner: peak, best plane \S+ um, refined: (-?\d+\.\d\d) um", printed
    )
    assert brenner is not None, printed
    assert abs(float(brenner.group(1))) <= 1.0
    # ASCII only: a Windows console decodes with its own code page.
    assert printed.isascii()
    with out_csv.open(encoding="utf-8", newline="") as handle:
        written = list(csv.reader(handle))
    assert written[0] == [
        "folder",
        "z_um",
        "brenner",
        "normalized-variance",
        "gradient-energy",
        "laplacian-variance",
    ]
    assert len(written) == 12
    assert {row[0] for row in written[1:]} == {stack.as_posix()}


def test_the_best_frame_is_not_a_plane_of_the_stack(
    fs: ModuleType, stack: Path
) -> None:
    planes, why = fs.read_stack(stack)

    assert why == ""
    assert len(planes) == 11
    assert (stack / "best.tif").is_file()


def test_a_folder_of_two_frames_is_skipped_and_nothing_scored_is_exit_1(
    fs: ModuleType,
    stack: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    short = tmp_path / "short"
    short.mkdir()
    for name in ("000.tif", "001.tif"):
        shutil.copy(stack / name, short / name)

    code = fs.main([str(short)])

    assert code == 1
    err = capsys.readouterr().err
    assert (
        f"{short}: 2 frames with z_um in their metadata, a stack needs 3: skipped"
        in err
    )
    assert "no stack was scored" in err


def test_a_skipped_folder_does_not_stop_the_others(
    fs: ModuleType,
    stack: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = fs.main([str(tmp_path / "missing"), str(stack)])

    assert code == 0
    captured = capsys.readouterr()
    assert "not a folder" in captured.err
    assert str(stack) in captured.out


def test_a_pattern_is_expanded_by_the_script_as_windows_shells_do_not(
    fs: ModuleType, stack: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # cmd.exe and PowerShell hand the program `cell-focus/*` as typed.
    code = fs.main([str(stack.parent / "*")])

    assert code == 0
    assert f"{stack}\n" in capsys.readouterr().out
    # A pattern that matches nothing is still a folder that is not there.
    assert fs.expand([str(stack.parent / "nothing-*")]) == [stack.parent / "nothing-*"]


def _refusing(
    monkeypatch: pytest.MonkeyPatch, fs: ModuleType, name: str, times: int
) -> list[str]:
    """Make reads of ``name`` raise ``PermissionError`` ``times`` times, as antivirus does on Windows."""
    refused: list[str] = []
    real = fs.tifffile.TiffFile

    def opener(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == name and len(refused) < times:
            refused.append(name)
            raise PermissionError(13, "The process cannot access the file", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(fs.tifffile, "TiffFile", opener)
    return refused


def test_a_frame_refused_for_a_moment_is_read_again(
    fs: ModuleType, stack: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = _refusing(monkeypatch, fs, "003.tif", times=2)

    planes, why = fs.read_stack(stack)

    assert refused == ["003.tif", "003.tif"]
    assert (len(planes), why) == (11, "")


def test_a_frame_that_stays_unreadable_skips_its_stack(
    fs: ModuleType,
    stack: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A curve with a hole in it would choose a score on planes that are not all there.
    _refusing(monkeypatch, fs, "003.tif", times=1000)

    code = fs.main([str(stack)])

    assert code == 1
    err = capsys.readouterr().err
    assert (
        f"{stack}: 003.tif cannot be read (The process cannot access the file): skipped"
        in err
    )


def test_a_tiff_without_z_is_left_out_and_counted(
    fs: ModuleType,
    stack: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(stack, copy)
    tifffile.imwrite(copy / "other.tif", np.zeros((8, 8), dtype=np.uint16))

    assert fs.main([str(copy)]) == 0

    err = capsys.readouterr().err
    assert f"{copy}: 1 TIFFs without z_um in their metadata left out" in err


def test_a_csv_that_cannot_be_written_is_said_and_is_exit_1(
    fs: ModuleType,
    stack: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A folder where the CSV should go: refused as Excel's hold on it is.
    held = tmp_path / "scores.csv"
    held.mkdir()

    code = fs.main([str(stack), "--csv", str(held)])

    assert code == 1
    captured = capsys.readouterr()
    assert f"{held} could not be written (" in captured.err
    assert str(stack) in captured.out


def test_plots_are_written_or_their_absence_is_said(
    fs: ModuleType,
    stack: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plots = tmp_path / "plots"

    code = fs.main([str(stack), "--plot", str(plots)])

    assert code == 0
    captured = capsys.readouterr()
    if importlib.util.find_spec("matplotlib") is None:
        assert "matplotlib is not installed: no plot written" in captured.err
    else:
        assert [p.suffix for p in plots.iterdir()] == [".png"]
