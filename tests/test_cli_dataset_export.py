"""``smc dataset export`` and ``smc dataset check`` (algorithms design §2, #565).

The run folders are ``tests/select_fields_seed.py::seed_run``'s and no
Micro-Manager is opened. The runner pins the width (FM-37) and clears
``SMC_PROFILE`` and ``SMC_PROFILES``; lines are found by their first words and
compared as ``line.split()``, so a test says what is printed and not how many
spaces separate it; help and usage text go through a plain-text helper that
takes the colour out (FM-72: ``GITHUB_ACTIONS=1`` forces it, and it splits an
option name in two).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import smc.data.cli as data_cli
import smc.data.dataset as dataset_module
from select_fields_seed import Seed, cell_at, seed_run
from smc.cli import app
from smc.data.cli import _progress_printer, dataset_app, export_lines
from smc.data.dataset import DatasetManifest, export_dataset

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """The text with the colour a forced terminal adds taken out (FM-72)."""
    return _ANSI.sub("", text)


def _lines(output: str) -> list[str]:
    return _plain(output).splitlines()


def _words(lines: list[str], *first: str) -> list[str]:
    """The words of the one line that starts with ``first``."""
    found = [line.split() for line in lines if line.split()[: len(first)] == [*first]]
    assert len(found) == 1, (first, found, lines)
    return found[0]


def _source(folder: Path, **changes: object) -> Seed:
    folder.mkdir(exist_ok=True)
    arguments: dict[str, object] = {
        "cells": [
            cell_at("A1", 1.0, 1.0, "single"),
            cell_at("A1", 2.0, 1.0, "doublet"),
            cell_at("B2", 1.0, 1.0, "single"),
        ],
        "wells": ("A1", "B2"),
    }
    arguments.update(changes)
    return seed_run(folder, **arguments)  # type: ignore[arg-type]


def _export(run_doc: Path, out: Path, *, columns: str = "200") -> tuple[int, list[str]]:
    result = CliRunner(
        env={"COLUMNS": columns, "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(app, ["dataset", "export", str(run_doc), "--out", str(out)])
    return result.exit_code, _lines(result.output)


def _check(directory: Path, *, columns: str = "200") -> tuple[int, list[str]]:
    result = CliRunner(
        env={"COLUMNS": columns, "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(app, ["dataset", "check", str(directory)])
    return result.exit_code, _lines(result.output)


def _on_disk_but_the_manifest(out: Path) -> list[Path]:
    return [p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"]


def test_export_prints_the_summary_and_writes_the_dataset(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src")
    out = tmp_path / "out"

    code, lines = _export(seed.path, out)

    assert code == 0, lines
    files = _on_disk_but_the_manifest(out)
    assert len(files) == 8 + 1 + 2  # the frames, the cells file, run.json, README.md
    megabytes = sum(p.stat().st_size for p in files) / 1e6
    # The progress lines come first, the `✓` line after them, the summary last.
    assert "copied: 1/9" in lines
    assert "copied: 9/9" in lines
    assert lines.index(f"✓ dataset written to {out}") > lines.index("copied: 9/9")
    summary = lines[lines.index(f"✓ dataset written to {out}") + 1 :]
    assert _words(summary, "frames") == ["frames", "8"]
    assert _words(summary, "wells") == ["wells", "A1,", "B2"]
    assert _words(summary, "cells") == ["cells", "3", "in", "cells/t1.json"]
    assert _words(summary, "size") == f"size {megabytes:.1f} MB in 11 files".split()
    assert _words(summary, "scrubbed") == ["scrubbed", "0", "values"]
    assert (out / "manifest.json").is_file()


def test_export_without_out_is_a_usage_error(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src")
    result = runner.invoke(app, ["dataset", "export", str(seed.path)])
    assert result.exit_code == 2
    assert "Missing option" in _plain(result.output)


def test_export_refuses_with_exit_2_and_names_the_file(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel.txt").write_text("keep", encoding="utf-8")

    code, lines = _export(seed.path, out)
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert str(out) in refusal
    assert "not empty" in refusal
    assert (out / "sentinel.txt").read_text(encoding="utf-8") == "keep"

    dry = _source(tmp_path / "dry", dry_run=True)
    code, lines = _export(dry.path, tmp_path / "other")
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert str(dry.path) in refusal
    assert "dry run" in refusal
    assert not (tmp_path / "other").exists()


def test_export_io_error_exits_1_and_says_the_folder_is_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-34: an error in the middle is told with how far it got, and the exit
    # code is not the one of a refusal: the folder holds part of the dataset.
    seed = _source(tmp_path / "src")
    out = tmp_path / "out"
    copy = dataset_module._copy_and_hash
    calls: list[int] = []

    def fails_on_the_second(source: Path, destination: Path) -> tuple[int, str]:
        calls.append(1)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return copy(source, destination)

    monkeypatch.setattr(dataset_module, "_copy_and_hash", fails_on_the_second)
    code, lines = _export(seed.path, out)

    assert code == 1
    (failure,) = [line for line in lines if "✗" in line]
    assert "stopped after 1 of 12 files" in failure
    assert seed.scan.frames[1].file in failure
    assert "No space left on device" in failure
    assert "no manifest.json" in failure
    assert not (out / "manifest.json").exists()


def test_export_keeps_a_bracketed_path_on_one_line(tmp_path: Path) -> None:
    # FM-36 and FM-37: rich reads "[bold]" as a style and drops it, and wraps at
    # the console's width; at 40 columns these paths would wrap.
    seed = _source(tmp_path / "run-[bold]-[1]")
    out = tmp_path / "out-[bold]-[2]"

    code, lines = _export(seed.path, out, columns="40")
    assert code == 0
    assert f"✓ dataset written to {out}" in lines

    code, lines = _export(seed.path, out, columns="40")
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert refusal.startswith(f"✗ {out}: ")
    assert "[bold]-[2]" in refusal

    code, lines = _export(
        tmp_path / "run-[bold]-[1]" / "nowhere.json", tmp_path / "x", columns="40"
    )
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert "run-[bold]-[1]" in refusal


def test_the_progress_printer_is_paced_and_prints_the_last_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [100.0]
    show = _progress_printer("copied", clock=lambda: now[0])

    show(1, 10)  # the first call is printed
    now[0] += 0.4
    show(2, 10)  # 0.4 s later: not
    now[0] += 0.5
    show(3, 10)  # 0.9 s after the first: not
    now[0] += 0.2
    show(4, 10)  # 1.1 s after the first: printed
    now[0] += 0.1
    show(5, 10)  # 0.1 s later: not
    now[0] += 0.0
    show(10, 10)  # the last is always printed

    printed = [line for line in capsys.readouterr().out.splitlines() if line]
    assert printed == ["copied: 1/10", "copied: 4/10", "copied: 10/10"]


def test_check_exits_0_1_and_2_and_never_1_for_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path / "src")
    out = tmp_path / "out"
    export_dataset(seed.path, out)

    # 0: everything matches.
    code, lines = _check(out)
    assert code == 0, lines
    assert "✓ 11 files match manifest.json (" in " ".join(lines)
    ok = [line for line in lines if line.startswith("✓")]
    assert len(ok) == 1
    assert ok[0].startswith("✓ 11 files match manifest.json (")
    assert ok[0].endswith(" MB)")
    assert _words(lines, "profile") == ["profile", "demo"]
    assert _words(lines, "frames") == ["frames", "8", "in", "wells", "A1,", "B2"]
    assert _words(lines, "cells") == ["cells", "3", "in", "cells/t1.json"]

    # 1: a frame was changed on the way.
    changed = seed.scan.frames[2].file
    data = bytearray((out / changed).read_bytes())
    data[-1] ^= 0xFF
    (out / changed).write_bytes(bytes(data))
    code, lines = _check(out)
    assert code == 1
    failures = [line for line in lines if "✗" in line]
    assert failures[0] == f"✗ {changed}: changed (same size, other content)"
    assert failures[-1] == (
        "✗ 1 of 11 files are missing or changed: copy the dataset again, then check again"
    )
    assert len(failures) == 2

    # 2: no manifest, a refusal.
    (out / "manifest.json").unlink()
    code, lines = _check(out)
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert "no manifest.json" in refusal
    assert "did not finish" in refusal

    # 2, and not 1: whatever `check` did not expect is not "a file differs" (FM-82).
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(data_cli, "check_dataset", boom)
    code, lines = _check(out)
    assert code == 2
    assert "✗ unexpected RuntimeError: boom" in lines


def test_check_names_at_most_twenty_files_and_counts_the_rest(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src", rows=4, columns=4)
    out = tmp_path / "out"
    manifest = export_dataset(seed.path, out)
    assert manifest.frames == 32
    for frame in seed.scan.frames[:25]:
        (out / frame.file).unlink()

    code, lines = _check(out)

    assert code == 1
    failures = [line for line in lines if "✗" in line]
    assert len(failures) == 20 + 1 + 1
    assert failures[0] == f"✗ {seed.scan.frames[0].file}: missing"
    assert failures[19] == f"✗ {seed.scan.frames[19].file}: missing"
    assert failures[20] == "✗ and 5 more"
    assert failures[21].startswith("✗ 25 of 35 files are missing or changed")
    assert not any(seed.scan.frames[20].file in line for line in failures)


def test_check_does_not_print_a_success_line_for_a_damaged_dataset(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path / "src")
    out = tmp_path / "out"
    export_dataset(seed.path, out)
    (out / seed.scan.frames[0].file).unlink()

    code, lines = _check(out)

    assert code == 1
    assert not any(line.startswith("✓") for line in lines)


def test_export_lines_say_why_there_are_no_cells_and_count_values_in_the_singular() -> (
    None
):
    manifest = DatasetManifest(
        smc_version="0",
        exported_utc="2026-11-18T12:12:47Z",  # type: ignore[arg-type]
        source="run.json",
        profile="demo",
        objective=None,
        pixel_size_um=5.0,
        image_shape_px=(60, 80),
        wells=("A1",),
        frames=0,
        cells=False,
        cells_absent="no cells record",
        files={},
        scrubbed=("/stages/scan/params/out",),
    )
    lines = export_lines(manifest)
    assert _words(lines, "cells") == ["cells", "none:", "no", "cells", "record"]
    assert _words(lines, "scrubbed") == (
        ["scrubbed", "1", "value,", "listed", "in", "manifest.json"]
    )
    # The labels are padded to the widest, two spaces apart.
    assert "frames    0" in lines
    assert "scrubbed  1 value, listed in manifest.json" in lines


def _help_texts(command: Any) -> list[str]:
    """The help text of a command and of each of its parameters, as typer holds them."""
    texts = [command.help or ""]
    texts += [getattr(param, "help", None) or "" for param in command.params]
    return texts


def test_the_dataset_group_lists_export_and_check_and_their_help_holds_no_bracket() -> (
    None
):
    group = runner.invoke(app, ["dataset", "--help"])
    assert group.exit_code == 0
    text = _plain(group.output)
    assert "export" in text
    assert "check" in text

    export_help = runner.invoke(app, ["dataset", "export", "--help"])
    assert export_help.exit_code == 0
    assert "RUN_DOC" in _plain(export_help.output)
    assert "--out" in _plain(export_help.output)

    # A square bracket in help text is rich markup (FM-36), and an option name
    # in it is split in two by the colour GITHUB_ACTIONS forces (FM-72).
    group_command: Any = typer.main.get_command(dataset_app)
    for name in ("export", "check"):
        for text in _help_texts(group_command.commands[name]):
            assert "[" not in text
            assert "]" not in text
            assert "--" not in text


def _cp1252_env() -> dict[str, str]:
    """A cp1252 console and no operator profile, as ``tests/unit/test_cli_streams.py::_cp1252_env`` (FM-40, FM-42)."""
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


def test_export_and_check_survive_a_cp1252_redirect(tmp_path: Path) -> None:
    # FM-42: CliRunner hides what a redirected Windows console does: the "✓"
    # and "✗" have no cp1252 encoding, and the files must still be written.
    seed = _source(tmp_path / "src")
    out = tmp_path / "out"
    log = tmp_path / "export.txt"

    assert (
        _run_redirected(
            ["dataset", "export", str(seed.path), "--out", str(out)], log, tmp_path
        )
        == 0
    ), log.read_text(encoding="cp1252")
    assert (out / "manifest.json").is_file()
    assert "dataset written to" in log.read_text(encoding="cp1252")

    checked = tmp_path / "check.txt"
    assert _run_redirected(["dataset", "check", str(out)], checked, tmp_path) == 0
    assert "files match manifest.json" in checked.read_text(encoding="cp1252")

    (out / seed.scan.frames[0].file).write_bytes(b"damaged")
    damaged = tmp_path / "damaged.txt"
    assert _run_redirected(["dataset", "check", str(out)], damaged, tmp_path) == 1
    assert "changed (size 7 instead of" in damaged.read_text(encoding="cp1252")

    json.loads((out / "manifest.json").read_text(encoding="utf-8"))  # still readable
