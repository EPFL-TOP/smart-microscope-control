"""``smc dataset show`` (algorithms design §3, #564): a scan's run folder, summarised with no stand.

The run folders come from ``tests/select_fields_seed.py::seed_run`` and no
Micro-Manager is opened. The runner pins the width (FM-37) and clears
``SMC_PROFILE`` and ``SMC_PROFILES``; lines are found by their first words and
compared as ``line.split()``, so a test says what is printed and not how many
spaces separate it, and the padding has a test of its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from select_fields_seed import SHAPE, Seed, cell_at, record, seed_run
from smc.cli import app
from smc.data import Scan, load_scan
from smc.data.cli import summary_lines
from smc.plugins.slots import FailedField
from smc.run import RunDocument

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
HEIGHT, WIDTH = SHAPE


def _lines(output: str) -> list[str]:
    """The printed lines, with the colour a forced terminal adds taken out."""
    return _ANSI.sub("", output).splitlines()


def _words(lines: list[str], *first: str) -> list[str]:
    """The words of the one line that starts with ``first``."""
    found = [line.split() for line in lines if line.split()[: len(first)] == [*first]]
    assert len(found) == 1, (first, found, lines)
    return found[0]


def _show(path: Path | str, *, columns: str = "200") -> tuple[int, list[str]]:
    result = CliRunner(
        env={"COLUMNS": columns, "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(app, ["dataset", "show", str(path)])
    return result.exit_code, _lines(result.output)


def _seed_with_known_cells(tmp_path: Path) -> Seed:
    cells = [
        cell_at("A1", 1.0, 1.0, "single"),
        cell_at("A1", 2.0, 1.0, "single"),
        cell_at("A1", 3.0, 1.0, "doublet"),
        cell_at("B2", 1.0, 1.0, "debris"),
        cell_at("B2", 2.0, 1.0, "single"),
    ]
    return seed_run(
        tmp_path, cells=cells, wells=("A1", "B2"), focus_of={("A1", 0): "unlocked"}
    )


def test_show_prints_the_profile_the_camera_the_wells_the_focus_states_and_the_cells_per_class(
    tmp_path: Path,
) -> None:
    seed = _seed_with_known_cells(tmp_path)
    # Three of the eight frames (0, 3 and 6) know their place in their well: a
    # count that is neither 0, all, nor half, which the complement would give too.
    placed = tuple(
        f.model_copy(update={"well_dx_um": 10.0, "well_dy_um": -20.0})
        if k % 3 == 0
        else f
        for k, f in enumerate(seed.scan.frames)
    )
    record(seed.path, "scan", "scan", seed.scan.model_copy(update={"frames": placed}))
    n = len(seed.scan.frames)
    in_a1 = sum(1 for f in seed.scan.frames if f.well == "A1")
    in_b2 = n - in_a1
    assert (in_a1, in_b2) == (4, 4)

    code, lines = _show(seed.path)

    assert code == 0
    assert _words(lines, "run", "document") == f"run document {seed.path}".split()
    assert _words(lines, "profile") == ["profile", "demo"]
    assert _words(lines, "objective") == ["objective", "Nikon", "10X", "S", "Fluor"]
    assert _words(lines, "pixel", "size") == ["pixel", "size", "5.0000", "µm/px"]
    assert _words(lines, "image", "shape") == (
        f"image shape {HEIGHT} x {WIDTH} px (height x width)".split()
    )
    assert _words(lines, "frames", str(n)) == f"frames {n} (0 fields failed)".split()
    assert _words(lines, "place", "in", "well") == (
        f"place in well 3 of {n} frames".split()
    )
    assert _words(lines, "well", "frames") == ["well", "frames", "focus"]
    assert _words(lines, "A1", str(in_a1)) == (
        f"A1 {in_a1} locked {in_a1 - 1}, unlocked 1".split()
    )
    assert _words(lines, "B2", str(in_b2)) == f"B2 {in_b2} locked {in_b2}".split()
    assert _words(lines, "cells") == ["cells", "5,", "detected", "with", "m.pth"]
    assert _words(lines, "well", "single") == ["well", "single", "doublet", "debris"]
    assert _words(lines, "A1", "2") == ["A1", "2", "1", "0"]
    assert _words(lines, "B2", "1") == ["B2", "1", "0", "1"]


def test_show_pads_each_column_to_its_widest_cell_and_leaves_no_trailing_blank(
    tmp_path: Path,
) -> None:
    seed = _seed_with_known_cells(tmp_path)
    code, lines = _show(seed.path)
    assert code == 0
    assert all(line == line.rstrip() for line in lines)
    # Two spaces between the columns, each as wide as its widest cell ("frames").
    assert "well  frames  focus" in lines
    assert "A1    4       locked 3, unlocked 1" in lines
    assert "B2    4       locked 4" in lines
    assert "well  single  doublet  debris" in lines
    assert "A1    2       1        0" in lines
    # The first block's values start in one column, after the longest label.
    assert "profile        demo" in lines
    assert "place in well  0 of 8 frames" in lines


def test_show_says_why_there_are_no_cells_and_exits_0(tmp_path: Path) -> None:
    seed = _seed_with_known_cells(tmp_path)
    data = json.loads(seed.path.read_text(encoding="utf-8"))
    del data["stages"]["cells"]
    seed.path.write_text(json.dumps(data), encoding="utf-8")

    code, lines = _show(seed.path)

    assert code == 0
    assert _words(lines, "cells") == ["cells", "none:", "no", "cells", "record"]
    assert _words(lines, "A1", "4") == ["A1", "4", "locked", "3,", "unlocked", "1"]
    # No class table without cells: the classes are the detector's, which never ran.
    assert not any("single" in line for line in lines)


def test_show_refuses_with_exit_2_and_names_the_file(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[], dry_run=True)
    result = runner.invoke(app, ["dataset", "show", str(seed.path)])
    assert result.exit_code == 2
    (refusal,) = [line for line in _lines(result.output) if "✗" in line]
    assert str(seed.path) in refusal
    assert "dry run" in refusal


def test_show_keeps_a_bracketed_path_and_does_not_wrap(tmp_path: Path) -> None:
    # FM-36 (rich reads "[bold]" as a style and drops it; "[1]" it leaves alone,
    # so the name has both) and FM-37 (rich wraps at the console's width): at 40
    # columns a path of this length would wrap.
    good = tmp_path / "run-[bold]-[1]"
    good.mkdir()
    seed = seed_run(good, cells=[cell_at("A1", 1.0, 1.0)])
    code, lines = _show(seed.path, columns="40")
    assert code == 0
    assert f"run document   {seed.path}" in lines

    broken = tmp_path / "broken-[bold]-[2]"
    broken.mkdir()
    bad = seed_run(broken, cells=[], dry_run=True)
    code, lines = _show(bad.path, columns="40")
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert refusal.startswith(f"✗ {bad.path}: ")
    assert "[bold]-[2]" in refusal
    assert refusal.endswith("run it without --dry-run")


def test_show_names_the_run_document_when_given_a_folder(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[])
    code, lines = _show(tmp_path)
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert f"{tmp_path}: is a folder" in refusal
    assert f"did you mean {seed.path}?" in refusal


def test_dataset_group_lists_show_in_its_help_and_in_smcs() -> None:
    group = runner.invoke(app, ["dataset", "--help"])
    assert group.exit_code == 0
    assert "show" in _ANSI.sub("", group.output)
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    assert "dataset" in _ANSI.sub("", root.output)


def _cp1252_env() -> dict[str, str]:
    """A cp1252 console and no operator profile, as ``tests/unit/test_cli_streams.py::_cp1252_env`` (FM-40)."""
    dropped = {"PYTHONUTF8", "SMC_PROFILE", "SMC_PROFILES"}
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def _run_redirected(args: list[str], out: Path, cwd: Path) -> int:
    with out.open("wb") as stream:
        done = subprocess.run(
            [sys.executable, "-m", "smc.cli", *args],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=_cp1252_env(),
            cwd=cwd,
            timeout=120,
            check=False,
        )
    return done.returncode


def test_show_survives_a_cp1252_redirect(tmp_path: Path) -> None:
    # FM-42: CliRunner hides what a redirected Windows console does: the "✗"
    # of a refusal has no cp1252 encoding, and "µ" has one.
    good = tmp_path / "good"
    good.mkdir()
    seed = _seed_with_known_cells(good)
    out = tmp_path / "show.txt"
    assert _run_redirected(["dataset", "show", str(seed.path)], out, tmp_path) == 0
    text = out.read_text(encoding="cp1252")
    assert "profile        demo" in text
    assert "pixel size     5.0000 µm/px" in text

    bad = tmp_path / "bad"
    bad.mkdir()
    dry = seed_run(bad, cells=[], dry_run=True)
    refused = tmp_path / "refused.txt"
    assert _run_redirected(["dataset", "show", str(dry.path)], refused, tmp_path) == 2
    assert "dry run" in refused.read_text(encoding="cp1252")


# --- the summary, as a function ---------------------------------------------------


def test_summary_lines_says_unknown_for_an_objective_and_a_cells_record_it_lacks(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"))
    failed = FailedField(
        well="B2", index=9, name="B2_0009", x_um=1.0, y_um=2.0, error="did not arrive"
    )
    scan = Scan.from_results(
        tmp_path,
        RunDocument.new(),
        plan=seed.plan,
        result=seed.scan.model_copy(update={"objective": None, "failed": (failed,)}),
        cells_result=None,
        cells=None,
    )
    lines = summary_lines(scan, "run.json")
    assert "run document   run.json" in lines  # the path as given, not resolved
    assert _words(lines, "profile") == ["profile", "unknown"]  # a scan built in memory
    assert _words(lines, "objective") == ["objective", "unknown"]
    assert _words(lines, "frames", "8") == ["frames", "8", "(1", "fields", "failed)"]
    assert _words(lines, "cells") == ["cells", "none:", "no", "cells", "record"]


def test_summary_lines_shows_a_label_or_a_well_the_detector_does_not_list(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",))
    scan = Scan.from_results(
        tmp_path,
        RunDocument.load(seed.path),
        plan=seed.plan,
        result=seed.scan,
        cells_result=seed.result,
        cells=[cell_at("A1", 1.0, 1.0, "mitosis"), cell_at("A2", 1.0, 1.0, "single")],
    )
    lines = summary_lines(scan, seed.path)
    assert _words(lines, "cells") == ["cells", "2,", "detected", "with", "m.pth"]
    assert _words(lines, "well", "single") == [
        "well",
        "single",
        "doublet",
        "debris",
        "mitosis",
    ]
    assert _words(lines, "A1", "0") == ["A1", "0", "0", "0", "1"]
    assert _words(lines, "A2", "1") == ["A2", "1", "0", "0", "0"]


def test_summary_lines_says_the_detector_is_unknown_when_the_scan_has_cells_and_no_result(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",))
    scan = Scan.from_results(
        tmp_path,
        RunDocument.load(seed.path),
        plan=seed.plan,
        result=seed.scan,
        cells_result=None,
        cells=[cell_at("A1", 1.0, 1.0, "mitosis")],
    )
    lines = summary_lines(scan, seed.path)
    assert _words(lines, "cells") == ["cells", "1,", "detector", "unknown"]
    assert _words(lines, "well", "mitosis") == ["well", "mitosis"]
    assert _words(lines, "A1", "1") == ["A1", "1"]


def test_show_lists_every_focus_state_a_well_has_in_the_order_locked_unlocked_off_none(
    tmp_path: Path,
) -> None:
    # A scan of an objective with no autofocus is "none" in every frame, and one
    # that did not use it is "off": a state that is not told reads as no frame.
    seed = seed_run(
        tmp_path,
        cells=[],
        wells=("A1", "B2"),
        focus_of={
            ("A1", 0): "unlocked",
            ("A1", 1): "off",
            ("A1", 2): "off",
            ("A1", 3): "none",
            ("B2", 1): "none",
            ("B2", 2): "none",
            ("B2", 3): "none",
        },
    )
    code, lines = _show(seed.path)
    assert code == 0
    assert _words(lines, "A1", "4") == [
        "A1",
        "4",
        "unlocked",
        "1,",
        "off",
        "2,",
        "none",
        "1",
    ]
    assert _words(lines, "B2", "4") == ["B2", "4", "locked", "1,", "none", "3"]


def test_show_prints_a_records_text_as_it_is_and_not_as_markup_or_an_emoji(
    tmp_path: Path,
) -> None:
    # FM-36: the objective is a record's text, and a label with brackets or a
    # colon pair is printed whole.
    seed = seed_run(tmp_path, cells=[])
    data = json.loads(seed.path.read_text(encoding="utf-8"))
    data["stages"]["scan"]["result"]["objective"] = "[bold]10x[/bold] :smile:"
    seed.path.write_text(json.dumps(data), encoding="utf-8")
    code, lines = _show(seed.path)
    assert code == 0
    assert "objective      [bold]10x[/bold] :smile:" in lines


def test_show_prints_the_scans_pixel_size_and_not_the_plans(tmp_path: Path) -> None:
    # FM-124: the plan was laid for another pixel size than the scan took.
    seed = seed_run(tmp_path, cells=[])
    data = json.loads(seed.path.read_text(encoding="utf-8"))
    data["stages"]["scan-plan"]["result"]["pixel_size_um"] = 2.0
    seed.path.write_text(json.dumps(data), encoding="utf-8")
    code, lines = _show(seed.path)
    assert code == 0
    assert "pixel size     5.0000 µm/px" in lines


def test_show_names_the_file_when_the_file_system_will_not_say_what_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-24: "Access is denied" is a refusal with a line and an exit code 2, not a traceback.
    seed = seed_run(tmp_path, cells=[])
    original = Path.is_dir

    def is_dir(self: Path) -> bool:
        if self == seed.path:
            raise PermissionError(13, "Access is denied")
        return original(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    code, lines = _show(seed.path)
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert refusal.startswith(f"✗ {seed.path}: ")
    assert "Access is denied" in refusal


def test_summary_lines_of_a_loaded_scan_is_what_the_command_prints(
    tmp_path: Path,
) -> None:
    seed = _seed_with_known_cells(tmp_path)
    code, printed = _show(seed.path)
    assert code == 0
    assert printed == summary_lines(load_scan(seed.path), seed.path)
