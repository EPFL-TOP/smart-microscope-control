"""``smc dataset draw`` (algorithms design §6, #568).

The run folders are ``tests/select_fields_seed.py::seed_run``'s and no
Micro-Manager is opened. The runner pins the width (FM-37) and clears
``SMC_PROFILE`` and ``SMC_PROFILES``; lines are found by their first words and
compared as ``line.split()``, so a test says what is printed and not how many spaces
separate it; help text goes through a plain-text helper that takes the colour out
(FM-72: ``GITHUB_ACTIONS=1`` forces it, and it splits an option name in two).
"""

from __future__ import annotations

import errno
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

import smc.data.cli as data_cli
from select_fields_seed import SHAPE, UPRIGHT, Seed, cell_at, seed_run
from smc.cli import app
from smc.data import draw as draw_module
from smc.data.cli import dataset_app
from smc.plugins.slots import RoutePosition, RouteResult
from smc.run import StageRecord, StageStatus, write_stage
from smc.run.files import AtomicWriteError

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


def _draw(run_doc: Path, *options: str, columns: str = "200") -> tuple[int, list[str]]:
    result = CliRunner(
        env={"COLUMNS": columns, "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(app, ["dataset", "draw", str(run_doc), *options])
    return result.exit_code, _lines(result.output)


def _stop(order: int, well: str, index: int, x_um: float, y_um: float) -> RoutePosition:
    return RoutePosition(
        order=order,
        well=well,
        name=f"{well}_r{index:03d}",
        x_um=x_um,
        y_um=y_um,
        z_um=None,
        pfs_offset=None,
        counts={"single": 1},
    )


def _write_route(
    path: Path,
    *,
    params: dict[str, Any] | None = None,
    extra: tuple[RoutePosition, ...] = (),
) -> None:
    """A ``route`` record, ``done``, with three stops: A1 has two, B2 one, the order plate-wide; ``extra`` stops follow."""
    result = RouteResult(
        objective="Nikon 10X S Fluor",
        matrix_um_per_px=UPRIGHT,
        image_shape_px=SHAPE,
        fov_um=(400.0, 300.0),
        labels=("single",),
        positions=(
            _stop(0, "A1", 0, 1000.0, -500.0),
            _stop(1, "A1", 1, 1300.0, -500.0),
            _stop(2, "B2", 0, 10000.0, -9500.0),
            *extra,
        ),
        uncovered={"A1": 0, "B2": 0},
        distance_um=300.0,
        travel_s=None,
    )
    then = datetime(2026, 10, 29, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin="route",
            provides="route",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params=params if params is not None else {},
            result=result.model_dump(mode="json"),
        ),
    )


def _source(
    folder: Path,
    *,
    route: bool = True,
    extra: tuple[RoutePosition, ...] = (),
    **changes: object,
) -> Seed:
    folder.mkdir(exist_ok=True)
    arguments: dict[str, object] = {
        "cells": [
            cell_at("A1", 1.0, 1.0, "single"),
            cell_at("B2", 1.0, 1.0, "single"),
        ],
        "wells": ("A1", "B2"),
        "rows": 1,
        "columns": 1,
    }
    arguments.update(changes)
    seed = seed_run(folder, **arguments)  # type: ignore[arg-type]
    if route:
        _write_route(
            seed.path,
            params={"selector": "most-cells", "orderer": "nearest"},
            extra=extra,
        )
    return seed


def test_draw_writes_a_picture_per_well_and_says_what_it_drew(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src")
    code, lines = _draw(seed.path)
    out = tmp_path / "src" / "draw"
    assert code == 0, lines
    assert (out / "A1.png").is_file()
    assert (out / "B2.png").is_file()
    assert _words(lines, "route")[1:] == [
        "route",
        "(selector",
        "most-cells,",
        "orderer",
        "nearest),",
        "3",
        "stops",
    ]
    assert _words(lines, "cells") == ["cells", "2", "detected"]
    # The path ends the line and may hold a space (a Windows user name does): the
    # words before it are compared, and the line must end with the path.
    assert _words(lines, "A1")[:9] == [
        "A1",
        "1",
        "of",
        "1",
        "frames",
        "2",
        "stops",
        "1",
        "cell",
    ]
    (well_line,) = [line for line in lines if line.startswith("A1 ")]
    assert well_line.endswith(f"  {out / 'A1.png'}")
    assert _words(lines, "B2")[1:9] == [
        "1",
        "of",
        "1",
        "frames",
        "1",
        "stop",
        "1",
        "cell",
    ]
    assert "drawing: 1/2" in lines
    assert "drawing: 2/2" in lines
    assert lines[-1] == f"✓ 2 pictures written to {out}"
    with Image.open(out / "A1.png") as picture:
        assert picture.format == "PNG"
        assert max(picture.size) == 1600


def test_draw_out_and_wells_and_max_px_do_what_they_say(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src")
    elsewhere = tmp_path / "elsewhere" / "pictures"
    code, lines = _draw(
        seed.path, "--out", str(elsewhere), "--wells", "b2", "--max-px", "300"
    )
    assert code == 0, lines
    assert sorted(p.name for p in elsewhere.iterdir()) == ["B2.png"]
    assert not (tmp_path / "src" / "draw").exists()
    with Image.open(elsewhere / "B2.png") as picture:
        assert max(picture.size) == 300
    assert lines[-1] == f"✓ 1 picture written to {elsewhere}"
    # Blanks and a repeated well are what a person types.
    code, lines = _draw(seed.path, "--out", str(elsewhere), "--wells", " A1 , a01,,")
    assert code == 0, lines
    assert sorted(p.name for p in elsewhere.iterdir()) == ["A1.png", "B2.png"]


def test_draw_says_why_there_is_no_route_and_still_draws(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src", route=False)
    code, lines = _draw(seed.path)
    assert code == 0, lines
    (route,) = [line for line in lines if line.startswith("route ")]
    assert route.split(None, 2)[1:] == [
        "not",
        "drawn: the run document has no route; smc run --list shows which "
        "plugin provides it",
    ]
    assert (tmp_path / "src" / "draw" / "A1.png").is_file()
    assert _words(lines, "A1")[5:7] == ["0", "stops"]


def test_draw_refuses_with_exit_2_what_it_cannot_use(tmp_path: Path) -> None:
    dry = _source(tmp_path / "dry", dry_run=True)
    code, lines = _draw(dry.path)
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert str(dry.path) in refusal
    assert "dry run" in refusal
    assert not (tmp_path / "dry" / "draw").exists()

    seed = _source(tmp_path / "src")
    code, lines = _draw(seed.path, "--wells", "Z9")
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert "well 'Z9' is not in this scan (A1, B2)" in refusal
    assert not (tmp_path / "src" / "draw").exists()

    for given in ("100", "4001", "0", "-5"):
        code, lines = _draw(seed.path, "--max-px", given)
        assert code == 2, given
        assert "✗ --max-px must be an integer from 256 to 4000" in lines
    assert not (tmp_path / "src" / "draw").exists()

    target = tmp_path / "a-file"
    target.write_bytes(b"not a folder")
    code, lines = _draw(seed.path, "--out", str(target))
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert "is a file, not a folder" in refusal
    assert target.read_bytes() == b"not a folder"


@pytest.mark.parametrize("given", ["", ",", " , "])
def test_draw_refuses_a_wells_option_that_names_no_well(
    tmp_path: Path, given: str
) -> None:
    seed = _source(tmp_path / "src")
    code, lines = _draw(seed.path, "--wells", given)
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert refusal.startswith("✗ --wells: no well is named")
    assert not (tmp_path / "src" / "draw").exists()


def test_draw_refuses_a_folder_a_missing_file_and_a_max_px_that_is_not_a_number(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path / "src")
    code, lines = _draw(tmp_path / "src")
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert "is a folder, not a run document" in refusal
    code, lines = _draw(tmp_path / "src" / "nowhere.json")
    assert code == 2
    assert any("✗" in line for line in lines)
    code, _ = _draw(seed.path, "--max-px", "many")
    assert code == 2


def test_draw_exit_1_when_a_picture_is_not_written_and_the_others_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path / "src")
    real = draw_module.write_atomically

    def flaky(path: Path, data: bytes, **kwargs: Any) -> None:
        if path.name == "B2.png":
            raise AtomicWriteError(OSError(errno.EIO, "disk"))
        real(path, data, **kwargs)

    monkeypatch.setattr(draw_module, "write_atomically", flaky)
    code, lines = _draw(seed.path)
    out = tmp_path / "src" / "draw"
    assert code == 1, lines
    assert (out / "A1.png").is_file()
    assert not (out / "B2.png").exists()
    assert f"✗ {out / 'B2.png'} was not written: disk" in lines
    assert any(line.startswith("✗ 1 of 2 pictures written to ") for line in lines), (
        lines
    )
    assert _words(lines, "B2")[-3:] == ["not", "written:", "disk"]
    assert not any(line.startswith("✓") for line in lines)


def test_an_unexpected_exception_is_exit_2_not_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-82: an uncaught exception ends Python with 1, which is "a picture was not written".
    seed = _source(tmp_path / "src")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the drawing broke")

    monkeypatch.setattr(draw_module, "draw_scan", boom)
    code, lines = _draw(seed.path)
    assert code == 2
    assert "✗ unexpected RuntimeError: the drawing broke" in lines


def test_draw_keeps_a_bracketed_path_on_one_line(tmp_path: Path) -> None:
    # FM-36 and FM-37: rich reads "[bold]" as a style and drops it, and wraps at the
    # console's width; at 40 columns this path would wrap.
    seed = _source(tmp_path / "run [bold] [1]")
    code, lines = _draw(seed.path, columns="40")
    out = tmp_path / "run [bold] [1]" / "draw"
    assert code == 0, lines
    assert f"✓ 2 pictures written to {out}" in lines
    assert any(str(out / "A1.png") in line for line in lines)

    code, lines = _draw(tmp_path / "run [bold] [1]" / "nowhere.json", columns="40")
    assert code == 2
    (refusal,) = [line for line in lines if "✗" in line]
    assert "run [bold] [1]" in refusal


def _cp1252_env() -> dict[str, str]:
    """A cp1252 console and no operator profile, as ``tests/unit/test_cli_streams.py::_cp1252_env`` (FM-40, FM-42)."""
    dropped = {"PYTHONUTF8", "SMC_PROFILE", "SMC_PROFILES"}
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def test_draw_survives_a_cp1252_redirect(tmp_path: Path) -> None:
    # FM-20, FM-42: CliRunner hides what a redirected Windows console does. The marks
    # and a Greek letter in the folder's name have no cp1252 encoding, and the
    # pictures must still be written.
    folder = tmp_path / "run-\N{GREEK SMALL LETTER LAMDA}"
    seed = _source(folder)
    log = tmp_path / "draw.txt"
    with log.open("wb") as stream:
        done = subprocess.run(
            [sys.executable, "-m", "smc.cli", "dataset", "draw", str(seed.path)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=_cp1252_env(),
            cwd=tmp_path,
            timeout=120,
            check=False,
        )
    assert done.returncode == 0, log.read_text(encoding="cp1252", errors="replace")
    assert (folder / "draw" / "A1.png").is_file()
    assert "pictures written to" in log.read_text(encoding="cp1252", errors="replace")


def test_dataset_help_lists_draw_and_its_help_holds_no_bracket_and_no_option() -> None:
    runner = CliRunner(
        env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None}
    )
    group = runner.invoke(app, ["dataset", "--help"])
    assert group.exit_code == 0
    assert "draw" in _plain(group.output)
    command = runner.invoke(app, ["dataset", "draw", "--help"])
    assert command.exit_code == 0
    text = _plain(command.output)
    for name in ("RUN_DOC", "--out", "--wells", "--max-px"):
        assert name in text
    # A square bracket in help text is rich markup (FM-36), and an option name in it is
    # split in two by the colour GITHUB_ACTIONS forces (FM-72).
    group_command: Any = typer.main.get_command(dataset_app)
    drawing = group_command.commands["draw"]
    texts = [drawing.help or ""] + [
        getattr(param, "help", None) or "" for param in drawing.params
    ]
    for help_text in texts:
        assert "[" not in help_text
        assert "]" not in help_text
        assert "--" not in help_text


def test_the_commands_numbers_are_the_drawings_and_the_command_does_not_import_it() -> (
    None
):
    assert data_cli._DRAW_DEFAULT_MAX_PX == draw_module.DEFAULT_MAX_PX
    assert data_cli._DRAW_MAX_PX_RANGE == (
        draw_module.MIN_MAX_PX,
        draw_module.MAX_MAX_PX,
    )
    assert not hasattr(data_cli, "draw_module")


def test_the_command_accepts_the_two_ends_of_the_max_px_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 4000 pixels is a 64 MB canvas a well: the end is checked against a recorder.
    seed = _source(tmp_path / "src")
    given: list[int] = []

    def record(scan: Any, out: Path, **kwargs: Any) -> draw_module.Drawing:
        given.append(kwargs["max_px"])
        picture = draw_module.WellPicture(
            "A1", out / "A1.png", True, None, 1, 1, 0, 0, 0, 0
        )
        return draw_module.Drawing(out, (picture,), None, "none", True, ())

    monkeypatch.setattr(draw_module, "draw_scan", record)
    for end in ("256", "4000"):
        code, lines = _draw(seed.path, "--max-px", end)
        assert code == 0, lines
    assert given == [256, 4000]


def test_draw_says_the_stops_of_a_well_the_scan_does_not_have(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src", extra=(_stop(3, "C3", 0, 50.0, 50.0),))
    code, lines = _draw(seed.path)
    assert code == 0, lines
    assert (
        "the route holds 1 stops in wells the scan does not have (C3): not drawn"
        in lines
    )
    assert _words(lines, "route")[-2:] == ["4", "stops"]


def test_draw_says_why_a_scan_has_no_cells_and_still_draws(tmp_path: Path) -> None:
    seed = _source(tmp_path / "src")
    (tmp_path / "src" / seed.result.file).unlink()
    code, lines = _draw(seed.path)
    assert code == 0, lines
    (cells,) = [line for line in lines if line.startswith("cells ")]
    assert cells.split(None, 2)[1] == "none:"
    assert "is missing from the run folder" in cells
    assert _words(lines, "A1")[7:9] == ["0", "cells"]
    assert (tmp_path / "src" / "draw" / "A1.png").is_file()


class _GoneConsole:
    """A console whose reader went away after ``alive`` writes, as a pipe closed by ``| more`` then q (FM-30)."""

    def __init__(self, alive: int) -> None:
        self.alive = alive

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.alive <= 0:
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")
        self.alive -= 1


@pytest.mark.parametrize("alive", [0, 2, 3])
def test_a_console_that_goes_away_does_not_cost_a_picture_or_change_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alive: int
) -> None:
    # The route and cells lines are two writes, then each well's progress line: the
    # wells after the one that found the pipe closed must still be drawn.
    seed = _source(tmp_path / "src", wells=("A1", "B2", "B1"))
    monkeypatch.setattr(data_cli, "console", _GoneConsole(alive))
    result = CliRunner(
        env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(app, ["dataset", "draw", str(seed.path)])
    assert result.exit_code == 0, result.output
    out = tmp_path / "src" / "draw"
    assert sorted(p.name for p in out.iterdir()) == ["A1.png", "B1.png", "B2.png"]


def test_a_console_that_goes_away_still_ends_a_refusal_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_cli, "console", _GoneConsole(0))
    result = CliRunner(
        env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(app, ["dataset", "draw", str(tmp_path / "nowhere.json")])
    assert result.exit_code == 2
