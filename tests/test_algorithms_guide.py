"""The algorithm developer's guide asks for commands smc has (algorithms design §7.6, #554).

The guide is the first thing a collaborator follows, on a laptop with no
microscope, and a command that was renamed after it was written is the first
thing that fails for them. These tests read the guide's own fenced blocks: every
line that starts with ``smc `` must name a command smc has, with options and
``--param`` keys it has, and the guide asks only for commands that never drive the
stand. What a flag does is checked by the walk the pull request carries, once; a
renamed flag or key is found here by its name.

The lines are read with POSIX rules (``shlex``), so a test can say nothing of how
PowerShell or ``cmd.exe`` read a quote (FM-175): the guide therefore puts JSON on a
command line in a POSIX block only, a test holds it to that, and the Windows way in
is a preset file, which no shell reads.
"""

from __future__ import annotations

import itertools
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from smc.cli import app
from smc.plugins.registry import discover

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "algorithms" / "README.md"
#: The skill that scaffolds a selector gives the same commands to an agent.
SKILL = ROOT / ".claude" / "skills" / "new-selector" / "SKILL.md"

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

#: The commands the guide may name: reading a dataset and running an offline tool.
#: Every other command of smc opens a stand or writes to one (`snap`, `ui`, `stage`,
#: `objective`, ...), and the guide never asks for that.
ALLOWED_COMMANDS = {"dataset", "run"}


def _blocks(text: str) -> list[tuple[str, list[str]]]:
    """The fenced blocks of ``text`` as ``(language, lines)``, in the order they are written.

    A fence may be indented (a block under a list item): a block that is not
    seen is a block whose commands are not checked.
    """
    blocks: list[tuple[str, list[str]]] = []
    language = ""
    lines: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if lines is None:
                language, lines = stripped[3:].strip(), []
            else:
                blocks.append((language, lines))
                lines = None
        elif lines is not None:
            lines.append(stripped)
    assert lines is None, "a fenced block of the guide is never closed"
    return blocks


def _smc_commands() -> list[tuple[str, str]]:
    """Every line of a fenced block of the guide or the skill that starts with ``smc ``, as ``(block language, line)``."""
    return [
        (language, line)
        for source in (GUIDE, SKILL)
        for language, lines in _blocks(source.read_text(encoding="utf-8"))
        for line in lines
        if line.startswith("smc ")
    ]


def _guide_lines() -> list[str]:
    return [line for _, line in _smc_commands()]


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


def _param_items(rest: list[str]) -> list[str]:
    """The ``KEY=VALUE`` items given to ``--param`` in the words of a command."""
    return [word for before, word in itertools.pairwise(rest) if before == "--param"]


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
        # Any other command of smc opens a stand or writes to one: `smc snap`,
        # `smc ui` and `smc stage move` exist, and a test that looked at `smc run`
        # lines only let them through (adversarial review of #554).
        assert path[0] in ALLOWED_COMMANDS, (
            f"{line!r}: smc {path[0]} is not a command the guide may ask for "
            f"({', '.join(sorted(ALLOWED_COMMANDS))}): it can drive a stand"
        )
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


def test_every_param_key_of_the_guide_is_a_parameter_of_its_tool() -> None:
    plugins = discover().plugins
    checked = 0
    for line in _guide_lines():
        path, _, rest = _resolve(line)
        if path != ["run"] or not rest or rest[0].startswith("-"):
            continue
        fields = plugins[rest[0]].plugin.Params.model_fields
        for item in _param_items(rest):
            key = item.split("=", 1)[0]
            checked += 1
            assert key in fields, (
                f"{line!r}: {rest[0]} has no parameter {key!r} "
                f"(it has {', '.join(sorted(fields))})"
            )
    assert checked >= 6, "the guide gives no --param to check"


def test_json_is_on_a_command_line_in_a_posix_block_only_and_is_json() -> None:
    seen = 0
    for language, line in _smc_commands():
        _, _, rest = _resolve(line)
        for item in _param_items(rest):
            value = item.split("=", 1)[1] if "=" in item else ""
            if not value.startswith(("[", "{")):
                continue
            seen += 1
            assert language == "sh", (
                f"{line!r}: JSON on a command line is quoted for a POSIX shell, "
                f"and this block is {language!r}: use a preset file (FM-175)"
            )
            json.loads(value)  # what smc itself does with it
    assert seen >= 1, "the guide has no JSON parameter to check"


def test_the_preset_file_of_the_guide_has_parameters_the_tool_has() -> None:
    plugins = discover().plugins
    found = 0
    for language, lines in _blocks(GUIDE.read_text(encoding="utf-8")):
        if language != "toml":
            continue
        document = tomllib.loads("\n".join(lines))
        for name, items in document["params"].items():
            fields = plugins[name].plugin.Params.model_fields
            for item in items:
                found += 1
                key, _, value = item.partition("=")
                assert key in fields, f"preset: {name} has no parameter {key!r}"
                if value.startswith(("[", "{")):
                    json.loads(value)
    assert found >= 2, "the guide has no preset file to check"


def test_no_command_of_the_guide_is_continued_on_a_second_line() -> None:
    # A command carried over by a backslash or a backtick has its second line read
    # by no test above, and the shells disagree on the character.
    for _, line in _smc_commands():
        assert not line.endswith(("\\", "`")), f"{line!r} is continued on another line"


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
