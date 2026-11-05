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

import smc.cli as cli
import smc.drive as drive
from smc.cli import app
from smc.drive import AutofocusReading, DriveResult

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


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_drive_result_not_reached_exits_1_with_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A turret that stops short of its label: a script chaining `smc objective set
    # ... && smc scan` must stop there.
    text = f"objective '{TEN_X}', although '{TWENTY_X}' was asked"

    def stops_short(microscope: object, label: str, *, confirm: bool) -> DriveResult:
        return DriveResult(text, reached=False)

    monkeypatch.setattr(cli, "set_objective", stops_short)

    result = runner.invoke(app, ["objective", "set", TWENTY_X, "--confirm"])

    assert result.exit_code == 1, result.output
    assert result.output.splitlines()[-1] == f"✗ {text}"
    assert text not in result.output.splitlines()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_get_that_cannot_read_exits_1_with_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unknown is not off: a script must not take "autofocus ?" for a state.
    text = "autofocus ? (autofocus: HardwareError: autofocus 'PFS': cannot read)"
    monkeypatch.setattr(
        cli,
        "read_autofocus",
        lambda microscope: AutofocusReading(text, None, False),
    )

    result = runner.invoke(app, ["autofocus", "get"])

    assert result.exit_code == 1, result.output
    assert result.output.splitlines()[-1] == f"✗ {text}"


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_get_decides_from_the_state_and_not_from_the_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-155: the exit code follows `known`, the state's own answer. The demo's
    # autofocus is off and known; words that happen to start like an unknown one
    # (a rewording of describe_autofocus) must not turn the reading into a failure.
    monkeypatch.setattr(
        drive, "describe_autofocus", lambda state: "autofocus ? (reworded)"
    )

    result = runner.invoke(app, ["autofocus", "get"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "autofocus ? (reworded)"


# --- smc autofocus offset (m2 §12.10.6) ---------------------------------------------


@pytest.fixture
def offset_limits_profile(tmp_path: Path) -> Path:
    """The demo devices with the offset's soft limits [5000, 12000]: a floor of 5000."""
    path = tmp_path / "floor.toml"
    path.write_text(
        '[microscope]\nname = "floor"\n\n'
        "[safety]\nautofocus_offset_soft_limits = [5000.0, 12000.0]\n",
        encoding="utf-8",
    )
    return path


def _never_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the stand was opened")

    monkeypatch.setattr(cli.Microscope, "open", refuse)


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_offset_reads_without_sending(
    monkeypatch: pytest.MonkeyPatch, offset_limits_profile: Path
) -> None:
    def no_write(microscope: object, value: float | None) -> DriveResult:
        raise AssertionError("the offset was written")

    monkeypatch.setattr(cli, "set_autofocus_offset", no_write)

    plain = runner.invoke(app, ["autofocus", "offset"])
    limited = runner.invoke(
        app, ["autofocus", "offset", "-p", str(offset_limits_profile)]
    )

    assert plain.exit_code == 0, plain.output
    assert plain.output.splitlines()[-1] == (
        "offset 0 device units, no soft limits in the profile (no floor: no first lock)"
    )
    # The brackets survive the console: no rich markup is read in them.
    assert limited.exit_code == 0, limited.output
    assert limited.output.splitlines()[-1] == (
        "offset 0 device units, soft limits [5000, 12000]"
    )


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_offset_to_sets_and_prints_the_readback() -> None:
    result = runner.invoke(app, ["autofocus", "offset", "--to", "12.5"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "offset 12.5 device units"


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_offset_floor_sets_the_lower_bound(
    offset_limits_profile: Path,
) -> None:
    result = runner.invoke(
        app, ["autofocus", "offset", "--floor", "-p", str(offset_limits_profile)]
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "offset 5000 device units"


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_offset_floor_without_soft_limits_exits_1() -> None:
    result = runner.invoke(app, ["autofocus", "offset", "--floor"])

    assert result.exit_code == 1, result.output
    assert result.output.splitlines()[-1] == (
        "✗ no floor: [safety] autofocus_offset_soft_limits is not set (m2 §12.10)"
    )


def test_autofocus_offset_with_both_options_exits_2_before_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _never_opened(monkeypatch)

    result = runner.invoke(app, ["autofocus", "offset", "--to", "5000", "--floor"])

    assert result.exit_code == 2, result.output
    assert "use --to or --floor, not both" in _ANSI.sub("", result.output)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_autofocus_offset_to_nan_exits_2_before_opening(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # typer parses them as floats (FM-32): the command refuses them itself.
    _never_opened(monkeypatch)

    result = runner.invoke(app, ["autofocus", "offset", "--to", value])

    assert result.exit_code == 2, result.output
    shown = str(float(value))
    assert f"--to must be a finite number, got {shown}" in _ANSI.sub("", result.output)


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_offset_dry_run_sends_nothing(offset_limits_profile: Path) -> None:
    profile = ["-p", str(offset_limits_profile)]

    # The offset is read again inside the same stand: it stayed where it was.
    result = runner.invoke(
        app, ["autofocus", "offset", "--floor", "--dry-run", *profile]
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == (
        "[dry-run] would set the offset to 5000 device units; it is at 0 device units"
    )
