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
import smc.hardware.backends.mm as mm
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


#: Written from the numbers of the profile below, never from the command's own helpers (FM-124).
MEASURED = "2026-11-12, session 3 (#400)"


def _write_offsets_profile(path: Path, reference: str, entries: str) -> Path:
    # `max_z_jog_um` is set so that the open warns of nothing: the lines are asserted whole.
    path.write_text(
        '[microscope]\nname = "offsets"\n\n[safety]\nmax_z_jog_um = 100.0\n\n'
        f'[objectives]\nreference = "{reference}"\n' + entries,
        encoding="utf-8",
    )
    return path


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_offset_prints_each_label_with_its_provenance(tmp_path: Path) -> None:
    path = _write_offsets_profile(
        tmp_path / "offsets.toml",
        TWENTY_X,
        f'\n[objectives.offsets."{TEN_X}"]\nxy_offset_um = [5.0, -2.5]\n'
        f'z_offset_um = -1.25\nmeasured = "{MEASURED}"\n'
        f'\n[objectives.offsets."Nikon 40X Plan Fluor ELWD"]\n'
        f'xy_offset_um = [-32.5, 9.75]\nmeasured = "{MEASURED}"\n',
    )

    result = runner.invoke(app, ["objective", "offset", "--profile", str(path)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"'Nikon 40X Plan Fluor ELWD'  xy (-32.50, +9.75) µm  z unknown  measured {MEASURED}",
        f"'{TEN_X}'  (current)  xy (+5.00, -2.50) µm  z -1.25 µm  measured {MEASURED}",
        "'Objective-2'  xy unknown  z unknown",
        f"'{TWENTY_X}'  reference",
        "'Objective-4'  xy unknown  z unknown",
        "'Objective-5'  xy unknown  z unknown",
    ]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_offset_names_a_profile_label_not_on_the_turret(
    tmp_path: Path,
) -> None:
    path = _write_offsets_profile(
        tmp_path / "offsets.toml",
        "Nikon 100X",
        f'\n[objectives.offsets."{TEN_X}"]\nxy_offset_um = [0.0, 0.0]\n'
        f'measured = "{MEASURED}"\n'
        f'\n[objectives.offsets."Nikon 4X"]\nz_offset_um = 3.0\nmeasured = "{MEASURED}"\n',
    )

    result = runner.invoke(app, ["objective", "offset", "--profile", str(path)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    # A zero that was measured is printed as one; the unmeasured labels are not.
    assert (
        f"'{TEN_X}'  (current)  xy (+0.00, +0.00) µm  z unknown  measured {MEASURED}"
    ) in lines
    assert "'Objective-2'  xy unknown  z unknown" in lines
    # The profile's labels that no turret position carries: the entry first, then
    # the reference; "[objectives]" reaches the terminal as written (no markup).
    assert lines[-2:] == [
        "'Nikon 4X' is in the profile's [objectives] but not on the turret: check the label",
        "'Nikon 100X' is in the profile's [objectives] but not on the turret: check the label",
    ]
    assert not any(line.endswith("reference") for line in lines)


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_offset_skips_an_empty_turret_position_and_prints_a_zero_without_its_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The protocol lists empty positions, whose label is ``""``: no objective to
    # measure an offset for, so no line. A `-0.0` read from the profile is `+0.00`.
    labels = mm.MMObjectiveTurret.labels
    monkeypatch.setattr(
        mm.MMObjectiveTurret, "labels", lambda self: ["", *labels(self)]
    )
    path = _write_offsets_profile(
        tmp_path / "offsets.toml",
        TWENTY_X,
        f'\n[objectives.offsets."{TEN_X}"]\nxy_offset_um = [-0.0, 0.0]\n'
        f'z_offset_um = -0.0\nmeasured = "{MEASURED}"\n',
    )

    result = runner.invoke(app, ["objective", "offset", "--profile", str(path)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    # Six lines for the six labels, none for the empty position put first.
    assert len(lines) == 6
    assert lines[0] == "'Nikon 40X Plan Fluor ELWD'  xy unknown  z unknown"
    assert (
        f"'{TEN_X}'  (current)  xy (+0.00, +0.00) µm  z +0.00 µm  measured {MEASURED}"
        in lines
    )
    assert not any(line.startswith("''") for line in lines)


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_objective_offset_without_a_reference_says_every_offset_is_unknown() -> None:
    result = runner.invoke(app, ["objective", "offset"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[:6] == [
        (
            f"'{label}'  (current)  xy unknown  z unknown"
            if label == TEN_X
            else f"'{label}'  xy unknown  z unknown"
        )
        for label in DEMO_LABELS
    ]
    assert lines[6:] == [
        "the profile names no reference objective ([objectives] reference): every "
        "offset is unknown"
    ]


def test_objective_offset_help_says_what_the_offsets_are_for() -> None:
    result = runner.invoke(app, ["objective", "offset", "--help"])

    assert result.exit_code == 0, result.output
    text = " ".join(_ANSI.sub("", result.output).split())
    assert "scanned with another" in text
    assert "objectives.reference and objectives.offsets" in text
    assert "never 0" in text


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
    # The brackets survive the console: no rich markup is read in them. The demo's
    # offset 0 lies below the floor 5000, and the line says so (#407).
    assert limited.exit_code == 0, limited.output
    assert limited.output.splitlines()[-1] == (
        "offset 0 device units, soft limits [5000, 12000]: below them; smc would not "
        "set an offset there"
    )


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_autofocus_offset_outside_the_limits_says_so_and_exits_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Limits [-10, -5]: the demo's offset 0 lies above them. A finding is no failure
    # of the read, so the exit code is 0, and nothing is written (m2 §12.10.6).
    def no_write(microscope: object, value: float | None) -> DriveResult:
        raise AssertionError("the offset was written")

    monkeypatch.setattr(cli, "set_autofocus_offset", no_write)
    path = tmp_path / "above.toml"
    path.write_text(
        '[microscope]\nname = "above"\n\n'
        "[safety]\nautofocus_offset_soft_limits = [-10.0, -5.0]\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["autofocus", "offset", "-p", str(path)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == (
        "offset 0 device units, soft limits [-10, -5]: above them; smc would not set "
        "an offset there"
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
