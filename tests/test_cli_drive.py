"""``smc objective``, ``smc autofocus`` and ``smc light`` on the demo devices (m5 §6.2, #263).

Each command is the same ``smc.drive`` call as the page's control (ADR-0006,
principle 5). Lines are asserted whole (FM-45), and a light test sets the state it
starts from: the demo's shutter reads open on some opens (FM-47).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smc.cli import app

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

TEN_X = "Nikon 10X S Fluor"
TWENTY_X = "Nikon 20X Plan Fluor ELWD"
DEMO_LABELS = (
    "Nikon 40X Plan Fluor ELWD",
    TEN_X,
    "Objective-2",
    TWENTY_X,
    "Objective-4",
    "Objective-5",
)

#: Typer forces colour when GITHUB_ACTIONS is set (FM-72): help is compared plain.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _restore_smc_logging() -> Iterator[None]:
    # The root callback replaces the smc logger's handler and level.
    logger = logging.getLogger("smc")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    logger.propagate = saved[2]


@pytest.fixture
def demo(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


@pytest.fixture
def shutter_light_profile(tmp_path: Path) -> Path:
    """The demo devices with the White Light Shutter as the light source: a light that switches."""
    path = tmp_path / "shutterlight.toml"
    path.write_text(
        '[microscope]\nname = "shutterlight"\n\n'
        '[roles.assign]\nlight_source = "White Light Shutter"\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_list_marks_the_current_label() -> None:
    result = runner.invoke(app, ["objective", "list"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"'{label}'  (current)" if label == TEN_X else f"'{label}'"
        for label in DEMO_LABELS
    ]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_get_prints_the_current_label() -> None:
    result = runner.invoke(app, ["objective", "get"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [f"objective '{TEN_X}'"]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_set_without_confirm_exits_2_and_names_the_confirm_flag() -> None:
    result = runner.invoke(app, ["objective", "set", TWENTY_X])

    assert result.exit_code == 2, result.output
    assert (
        "✗ turning the objective turret can drive an objective into a loaded plate "
        "(objective_turret 'Objective') — re-run with --confirm once the stage is clear"
    ) in result.output.splitlines()
    assert "confirm=True" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_set_with_confirm_prints_the_new_objective() -> None:
    result = runner.invoke(app, ["objective", "set", TWENTY_X, "--confirm"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == f"objective '{TWENTY_X}'"


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_set_with_an_unknown_label_exits_2_and_lists_the_labels() -> None:
    result = runner.invoke(app, ["objective", "set", "Nikon 100X", "--confirm"])

    assert result.exit_code == 2, result.output
    labels = ", ".join(repr(label) for label in DEMO_LABELS)
    assert (
        f"✗ objective_turret 'Objective' has no position 'Nikon 100X'; its labels "
        f"are: {labels}"
    ) in result.output.splitlines()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_set_dry_run_leaves_the_turret() -> None:
    result = runner.invoke(
        app, ["objective", "set", TWENTY_X, "--confirm", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == (
        f"[dry-run] would turn to '{TWENTY_X}'; the turret is at '{TEN_X}'"
    )


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_engage_then_disengage_print_the_state() -> None:
    engaged = runner.invoke(app, ["autofocus", "engage"])
    disengaged = runner.invoke(app, ["autofocus", "disengage"])
    read = runner.invoke(app, ["autofocus", "get"])

    assert engaged.exit_code == 0, engaged.output
    assert engaged.output.splitlines()[-1] == "autofocus locked"
    assert disengaged.exit_code == 0, disengaged.output
    assert disengaged.output.splitlines()[-1] == "autofocus off"
    assert read.exit_code == 0, read.output
    assert read.output.splitlines()[-1] == "autofocus off"


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_light_on_a_light_without_on_off_exits_1_with_the_layers_reason() -> None:
    # The demo's light source is LED, a State device: no on/off that smc knows.
    result = runner.invoke(app, ["light", "on"])

    assert result.exit_code == 1, result.output
    assert (
        "✗ light_source 'LED' (a State device) has no on/off that smc knows; switch "
        "the light with its shutter (Shutter)"
    ) in result.output.splitlines()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_light_on_and_off_switch_a_shutter_typed_light(
    shutter_light_profile: Path,
) -> None:
    profile = ["-p", str(shutter_light_profile)]
    # FM-47: the state the test starts from is set, never assumed.
    first = runner.invoke(app, ["light", "off", *profile])
    on = runner.invoke(app, ["light", "on", *profile])
    off = runner.invoke(app, ["light", "off", *profile])

    assert first.exit_code == 0, first.output
    assert on.exit_code == 0, on.output
    assert on.output.splitlines()[-1] == "light on"
    assert off.exit_code == 0, off.output
    assert off.output.splitlines()[-1] == "light off"


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_light_off_in_dry_run_says_that_it_is_sent(
    shutter_light_profile: Path,
) -> None:
    result = runner.invoke(
        app, ["light", "off", "--dry-run", "-p", str(shutter_light_profile)]
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == (
        "light off (dry run: switching off is sent all the same)"
    )


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_light_get_on_a_light_without_on_off_says_so_and_exits_0() -> None:
    result = runner.invoke(app, ["light", "get"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["light LED: no on/off that smc knows"]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_stand_without_a_turret_exits_1_with_the_facades_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "noturret.toml"
    path.write_text(
        '[microscope]\nname = "noturret"\n\n'
        '[roles.exclude]\nobjective_turret = ["Objective"]\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["objective", "set", TWENTY_X, "--confirm", "-p", str(path)]
    )

    assert result.exit_code == 1, result.output
    failed = [line for line in result.output.splitlines() if line.startswith("✗ ")]
    assert len(failed) == 1, result.output
    assert failed[0].startswith(
        "✗ ObjectiveTurret needs the 'objective_turret' role, which no loaded "
        "device fills"
    ), result.output


def test_objective_set_help_documents_the_confirm_flag() -> None:
    result = runner.invoke(app, ["objective", "set", "--help"])

    assert result.exit_code == 0, result.output
    plain = _ANSI.sub("", result.output)
    # The flag and its own help on one row: the docstring says --confirm too.
    help_text = "The stage is clear: turning the turret can drive an objective"
    rows = [
        row
        for row in plain.splitlines()
        if re.search(r"--confirm\s", row) and help_text in row
    ]
    assert rows, plain


def test_the_three_groups_are_listed_with_their_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    plain = _ANSI.sub("", result.output)
    for name, text in (
        ("objective", "The objective turret; a turn needs --confirm."),
        ("autofocus", "The autofocus (Nikon PFS, Zeiss Definite Focus)"),
        ("light", "The light source: on, off, read."),
    ):
        assert any(
            re.search(rf"\b{name}\s", row) and text in row for row in plain.splitlines()
        ), (name, plain)
