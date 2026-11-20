"""The algorithm developer's guide asks for commands smc has (algorithms design §7.6, #554).

The guide is the first thing a collaborator follows, on a laptop with no
microscope, and a command that was renamed after it was written is the first
thing that fails for them. These tests read the guide's own fenced blocks: every
line that starts with ``smc `` must name a command smc has, with options it has,
and every tool it runs must be one that never drives the stand. What a flag does
is checked by the walk the pull request carries, once; a renamed flag is found
here by its name.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from smc.cli import app
from smc.plugins.registry import discover

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "algorithms" / "README.md"

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})


def _smc_lines(text: str) -> list[str]:
    """The lines of the guide's fenced blocks that start with ``smc ``, in the order they are written."""
    lines: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside and line.startswith("smc "):
            lines.append(line.strip())
    return lines


def _guide_lines() -> list[str]:
    return _smc_lines(GUIDE.read_text(encoding="utf-8"))


def _resolve(line: str) -> tuple[list[str], Any, list[str]]:
    """The command path of ``line`` (``["dataset", "check"]``), the command it ends on, and its other words.

    The command is typed ``Any``: typer 0.28 carries its own copy of click, and a
    group is told from a command by its ``commands`` and not by an import.
    """
    current: Any = typer.main.get_command(app)
    words = shlex.split(line)[1:]
    path: list[str] = []
    for word in words:
        if word in getattr(current, "commands", {}):
            current = current.commands[word]
            path.append(word)
        else:
            break
    return path, current, words[len(path) :]


def test_the_guide_has_commands_to_check() -> None:
    lines = _guide_lines()
    # a guide with no command line would pass every test below
    assert len(lines) >= 10, lines
    paths = {tuple(_resolve(line)[0]) for line in lines}
    assert {("dataset", "check"), ("dataset", "show"), ("dataset", "draw")} <= paths
    assert ("run",) in paths


def test_every_smc_command_of_the_guide_exists() -> None:
    for line in _guide_lines():
        path, command, rest = _resolve(line)
        assert path, f"{line!r} names no smc command"
        # A group is not a command: `smc dataset render` would stop at `dataset`,
        # whose --help is fine, and the line would pass although it cannot run.
        assert not hasattr(command, "commands"), (
            f"{line!r}: {' '.join(path)} is a group of commands; "
            f"it has no command {rest[0] if rest else ''!r}"
        )
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, f"{line!r}: {result.output}"
        options = {
            name
            for parameter in command.params
            for name in (*parameter.opts, *parameter.secondary_opts)
        }
        for word in rest:
            if word.startswith("--") and word != "--help":
                assert word.split("=")[0] in options, (
                    f"{line!r}: smc {' '.join(path)} has no option {word!r}"
                )


def test_the_guide_names_no_command_that_drives_the_stand() -> None:
    plugins = discover().plugins
    runs = 0
    for line in _guide_lines():
        path, _, rest = _resolve(line)
        if path != ["run"] or not rest or rest[0].startswith("-"):
            continue  # `smc run --list` names no plugin
        runs += 1
        name = rest[0]
        assert name in plugins, f"{line!r}: no plugin is named {name!r}"
        assert getattr(plugins[name].plugin, "offline", False) is True, (
            f"{line!r}: {name} drives the stand, and the guide never asks for that"
        )
        if "--show-params" not in rest:
            # without --offline, smc run opens a stand: the guide has none to open
            assert "--offline" in rest, f"{line!r} has no --offline"
    assert runs >= 5, "the guide runs no tool offline"


@pytest.mark.parametrize(
    "heading",
    [
        "What this is for",
        "Set up",
        "Get a dataset",
        "Load it in Python",
        "Run the chain offline",
        "Write a selector",
        "Test it",
        "What never to assume",
        "What never to touch",
        "The issues and the pull request",
        "What comes next",
    ],
)
def test_the_guide_has_the_sections_a_newcomer_follows(heading: str) -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(heading)}$", text, flags=re.M), heading
