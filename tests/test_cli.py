"""The ``smc`` command line (design §9): CliRunner on the demo devices.

The runner pins the width (FM-37: rich wraps at 80 columns when it is not
writing to a terminal) and clears ``SMC_PROFILE``, so an operator's
``SMC_PROFILE=nikon-ti2`` never makes ``pytest`` open a real stand (FM-40).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner, Result

from smc import __version__
from smc.cli import app
from smc.discovery import dll_probe
from smc.discovery.dll_probe import DllProbe, describe_winerror
from smc.hardware import Microscope
from smc.hardware import core as core_mod
from smc.hardware import microscope as microscope_mod
from smc.hardware.backends import nikon
from smc.hardware.backends.mm import MMXYStage
from smc.hardware.capabilities import XY
from smc.hardware.errors import CoreError, HardwareError
from smc.hardware.microscope import MicroscopeState

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})


@pytest.fixture(autouse=True)
def _restore_smc_logging() -> Iterator[None]:
    # The root callback replaces the smc logger's handler and level; a later
    # caplog test would otherwise lose its INFO records.
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
def noxy_profile(tmp_path: Path) -> Path:
    """The demo devices with the XY stage excluded, so the role stays unfilled."""
    path = tmp_path / "noxy.toml"
    path.write_text(
        '[microscope]\nname = "noxy"\n\n[roles.exclude]\nxy_stage = ["xy"]\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def no_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command opens a microscope."""

    def refuse(cls: type[Microscope], *args: object, **kwargs: object) -> Microscope:
        pytest.fail("the command opened a microscope")

    monkeypatch.setattr(Microscope, "open", classmethod(refuse))


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_the_cli_leaves_the_smc_logger_as_it_found_it() -> None:
    # A handler left on CliRunner's closed stream turned every later smc log
    # record into a "Logging error" traceback in other test modules (CI, #11).
    logger = logging.getLogger("smc")
    before = (logger.level, list(logger.handlers))

    result = runner.invoke(app, ["-v", "version"])

    assert result.exit_code == 0, result.output
    assert (logger.level, list(logger.handlers)) == before


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "doctor" in result.output


@pytest.mark.demo
def test_doctor_passes_on_the_demo_devices(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "loads and answers" in result.output


def test_doctor_fails_cleanly_on_a_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", "--config", str(tmp_path / "nope.cfg")])
    assert result.exit_code == 1
    assert "not found" in result.output


# --- profiles ------------------------------------------------------------------


def test_profiles_lists_the_built_in_demo_when_no_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["profiles"])

    assert result.exit_code == 0, result.output
    assert re.search(r"^demo\s+\(built in\)$", result.output, re.M), result.output


def test_profiles_lists_smc_profiles_and_reports_an_invalid_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "good.toml").write_text('[microscope]\nname = "good"\n', encoding="utf-8")
    (sites / "bad.toml").write_text(
        '[microscope]\nname = "bad"\n\n[safety]\nmax_jog_um = -1.0\n', encoding="utf-8"
    )
    (sites / "other.toml").write_text('[tool]\nname = "x"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["profiles"], env={"SMC_PROFILES": str(sites)})

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    good = next(line for line in lines if line.startswith("good"))
    assert str((sites / "good.toml").resolve()) in good
    bad = lines.index(next(line for line in lines if line.startswith("bad")))
    assert lines[bad + 1].lstrip().startswith("✗")
    # FM-45: the key, not a word the tmp path could contain.
    assert "safety.max_jog_um" in lines[bad + 1]
    assert not any(line.startswith("other") for line in lines)
    searched = lines[lines.index("Search paths:") + 1 :]
    assert searched[0].strip() == str(sites)


def test_profiles_lists_a_cp1252_profile_with_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Notepad's "ANSI" (FM-80): the listing says what is wrong with the file
    # and goes on, as it does for any other profile that does not load.
    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "latin.toml").write_bytes(
        '[microscope]\nname = "t"\n# café\n'.encode("cp1252")
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["profiles"], env={"SMC_PROFILES": str(sites)})

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    latin = lines.index(next(line for line in lines if line.startswith("latin")))
    assert lines[latin + 1].lstrip().startswith("✗")
    assert "line 3 is not UTF-8 text" in lines[latin + 1]
    # The path is on the name's line; the error does not repeat it.
    shown = str((sites / "latin.toml").resolve())
    assert shown in lines[latin]
    assert shown not in lines[latin + 1]


@pytest.fixture
def unreadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``sites/aaa.toml`` names an ``mm.cfg`` that exists but cannot be looked at.

    ``Path.is_file`` is patched to raise, because a folder the account may not
    search (``chmod 000``) cannot be built on Windows, nor a share that wants a
    login anywhere (#93, FM-77). ``sites/zzz.toml`` is a good profile.
    """
    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "aaa.toml").write_text(
        '[microscope]\nname = "aaa"\n\n[micromanager]\nconfig = "mm.cfg"\n',
        encoding="utf-8",
    )
    (sites / "mm.cfg").write_text("", encoding="utf-8")
    (sites / "zzz.toml").write_text('[microscope]\nname = "zzz"\n', encoding="utf-8")
    real_is_file = Path.is_file

    def is_file(self: Path, *args: object, **kwargs: object) -> bool:
        if self.name == "mm.cfg":
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_file(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.chdir(tmp_path)
    return sites


def test_profiles_lists_a_profile_whose_config_cannot_be_looked_at_and_goes_on(
    unreadable_config: Path,
) -> None:
    result = runner.invoke(
        app, ["profiles"], env={"SMC_PROFILES": str(unreadable_config)}
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    aaa = lines.index(next(line for line in lines if line.startswith("aaa")))
    assert lines[aaa + 1].lstrip().startswith("✗ micromanager.config: cannot look at ")
    assert "Permission denied" in lines[aaa + 1]
    # The profiles after it, and the search paths, are still printed.
    assert any(line.startswith("zzz") for line in lines[aaa + 2 :])
    assert "Search paths:" in lines
    assert "unexpected" not in result.output


def test_stage_get_on_a_profile_whose_config_cannot_be_looked_at_exits_1(
    unreadable_config: Path,
) -> None:
    path = unreadable_config / "aaa.toml"

    result = runner.invoke(app, ["stage", "get", "-p", str(path)])

    assert result.exit_code == 1, result.output
    assert f"✗ {path}: micromanager.config: cannot look at " in result.output
    assert "unexpected" not in result.output


def test_stage_get_on_a_cp1252_profile_exits_1_naming_the_file_and_the_line(
    tmp_path: Path,
) -> None:
    # The report of #93: this printed "✗ unexpected UnicodeDecodeError: …", with
    # no file name, and asked the operator to file an issue.
    path = tmp_path / "latin.toml"
    path.write_bytes('[microscope]\n# café\nname = "t"\n'.encode("cp1252"))

    result = runner.invoke(app, ["stage", "get", "-p", str(path)])

    assert result.exit_code == 1, result.output
    assert f"✗ {path}: line 2 is not UTF-8 text (" in result.output
    assert "Save the file as UTF-8" in result.output
    assert "unexpected" not in result.output


def test_missing_profile_exits_1_with_the_search_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["stage", "get", "-p", "nosuch"])

    assert result.exit_code == 1, result.output
    assert "No profile named 'nosuch'" in result.output
    assert str(Path.cwd() / "profiles") in result.output


def test_smc_profile_env_selects_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["stage", "get"], env={"SMC_PROFILE": "nosuch"})

    assert result.exit_code == 1, result.output
    assert "No profile named 'nosuch'" in result.output


# --- devices, doctor -----------------------------------------------------------


def _demo_role_lines() -> list[str]:
    with Microscope.open("demo") as microscope:
        return microscope.roles.describe()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_devices_lists_each_device_with_the_roles_it_fills() -> None:
    result = runner.invoke(app, ["devices"])

    assert result.exit_code == 0, result.output
    out = result.output
    assert re.search(r"^XY\s+XYStage\s+DemoCamera/DXYStage\s+xy_stage$", out, re.M)
    assert re.search(r"^DHub\s+Hub\s+DemoCamera/DHub$", out, re.M)
    # A shutter candidate that is not assigned: it fills no role.
    assert re.search(
        r"^LED Shutter\s+Shutter\s+Utilities/State Device Shutter$", out, re.M
    ), out


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_devices_prints_every_role_line_with_its_source() -> None:
    expected = _demo_role_lines()

    result = runner.invoke(app, ["devices"])

    assert result.exit_code == 0, result.output
    # FM-36: "[core]" and "(also: LED Shutter)" survive the printing.
    assert any("[core]" in line for line in expected)
    assert any("(also: LED Shutter)" in line for line in expected)
    for line in expected:
        assert line in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_with_profile_prints_the_role_table() -> None:
    expected = _demo_role_lines()

    result = runner.invoke(app, ["doctor", "-p", "demo"])

    assert result.exit_code == 0, result.output
    assert "Profile demo" in result.output
    assert "loads and answers" in result.output
    for line in expected:
        assert line in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_prints_the_role_warnings(noxy_profile: Path) -> None:
    result = runner.invoke(app, ["doctor", "-p", str(noxy_profile)])

    assert result.exit_code == 0, result.output
    assert "! xy_stage: the configuration names 'XY'" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_fails_when_a_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # describe() shows the failed read as "XY ?"; a session script that
    # gates on doctor (#16) must not go on to drive that stage.
    def timed_out(self: MMXYStage) -> XY:
        raise RuntimeError("Serial port timed out")

    monkeypatch.setattr(MMXYStage, "position_um", timed_out)

    result = runner.invoke(app, ["doctor", "-p", "demo"])

    assert result.exit_code == 1, result.output
    assert "XY ?" in result.output
    assert "✗ xy: RuntimeError: Serial port timed out" in result.output
    assert "loads and answers" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
@pytest.mark.parametrize("failing_read", [1, 2])
def test_doctor_judges_the_snapshot_it_prints(
    failing_read: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A flaky serial line fails one read and answers the next. Read twice,
    # the line could show "XY ?" beside the ✓ and exit 0.
    real = MMXYStage.position_um
    reads: list[int] = []

    def flaky(self: MMXYStage) -> XY:
        reads.append(1)
        if len(reads) == failing_read:
            raise RuntimeError("Serial port timed out")
        return real(self)

    monkeypatch.setattr(MMXYStage, "position_um", flaky)

    result = runner.invoke(app, ["doctor", "-p", "demo"])

    failed = result.exit_code == 1
    assert result.exit_code in (0, 1), result.output
    assert ("XY ?" in result.output) == failed, result.output
    assert ("✗ xy: RuntimeError" in result.output) == failed, result.output
    assert ("loads and answers" in result.output) != failed, result.output


_SNAPSHOTS = {
    "readings": MicroscopeState(
        xy=XY(1234.5, -56.25), z_um=12.5, exposure_ms=25.0, shutter_open=False
    ),
    # The autofocus part, in each of its states (m2 §3.11).
    "autofocus-off": MicroscopeState(
        autofocus_engaged=False, autofocus_locked=False, autofocus_offset_native=0.0
    ),
    "autofocus-locked": MicroscopeState(
        autofocus_engaged=True, autofocus_locked=True, autofocus_offset_native=0.0
    ),
    "autofocus-searching": MicroscopeState(
        autofocus_engaged=True, autofocus_locked=False, autofocus_offset_native=0.0
    ),
    "autofocus-unreadable-lock": MicroscopeState(
        autofocus_engaged=True, errors=["autofocus: HardwareError: d"]
    ),
    "failed-reads": MicroscopeState(
        errors=["xy: RuntimeError: a", "z: RuntimeError: b", "shutter: OSError: c"]
    ),
    "moving-halted": MicroscopeState(
        xy=XY(0.0, 0.0),
        z_um=0.0,
        exposure_ms=10.0,
        shutter_open=True,
        moving=("XY", "Z"),
        halted=True,
    ),
}


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
@pytest.mark.parametrize("snapshot", sorted(_SNAPSHOTS))
@pytest.mark.parametrize("stand", ["demo", "noxy"])
def test_doctor_prints_the_status_line_of_describe(
    stand: str, snapshot: str, noxy_profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # doctor builds the line from the snapshot it judges; it must stay the
    # line Microscope.describe() prints for that snapshot, a missing stage
    # left out included. Both get the same one: the demo's shutter reads
    # open or closed from one open to the next.
    state = _SNAPSHOTS[snapshot]
    monkeypatch.setattr(Microscope, "state", lambda self: state)
    profile = "demo" if stand == "demo" else str(noxy_profile)
    with Microscope.open(profile) as microscope:
        expected = microscope.describe()

    result = runner.invoke(app, ["doctor", "-p", profile])

    assert result.exit_code == (1 if state.errors else 0), result.output
    assert re.search(f"^{re.escape(expected)}$", result.output, re.M), result.output
    assert expected.startswith("XY") == (stand == "demo")
    # Both stands have the demo's autofocus: the line names its state.
    shown = {
        "autofocus-off": "autofocus off",
        "autofocus-locked": "autofocus locked",
        "autofocus-searching": "autofocus on, not locked",
        "autofocus-unreadable-lock": "autofocus ?",
    }
    if snapshot in shown:
        assert f"| {shown[snapshot]}" in expected


# --- stage, z ------------------------------------------------------------------

_XY_LINE = re.compile(r"^XY \((-?\d+\.\d\d), (-?\d+\.\d\d)\) µm$", re.M)
_Z_LINE = re.compile(r"^Z (-?\d+\.\d\d) µm$", re.M)


def _xy_printed(output: str) -> tuple[float, float]:
    match = _XY_LINE.search(output)
    assert match, output
    return float(match[1]), float(match[2])


def _z_printed(output: str) -> float:
    match = _Z_LINE.search(output)
    assert match, output
    return float(match[1])


def test_a_misspelt_dry_run_flag_exits_2_before_opening_the_stand(
    no_stand: None,
) -> None:
    result = runner.invoke(app, ["stage", "move", "100", "0", "--dryrun"])

    assert result.exit_code == 2, result.output


def test_stage_z_and_snap_help_name_their_units() -> None:
    for args, unit in [
        (["stage", "--help"], "µm"),
        (["stage", "jog", "--help"], "µm"),
        (["z", "--help"], "µm"),
        (["snap", "--help"], "ms"),
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert unit in result.output, args


#: Typer forces colour when GITHUB_ACTIONS is set, and rich then styles an
#: option name in pieces (``-`` and ``-force``): the plain text is what a
#: help test can compare, in CI and in a terminal alike.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_z_jog_help_documents_the_force_option() -> None:
    result = runner.invoke(app, ["z", "jog", "--help"])

    assert result.exit_code == 0, result.output
    plain = _ANSI.sub("", result.output)
    # One row of the options table carries the flag and its own help text. The
    # command's docstring also says "--force", so the flag alone, or the help
    # text alone, would pass without the option being declared.
    help_text = "Move further than the profile's safety.max_z_jog_um allows."
    rows = [
        row
        for row in plain.splitlines()
        if re.search(r"--force\s", row) and help_text in row
    ]
    assert rows, plain


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_stage_get_prints_the_position_in_um() -> None:
    result = runner.invoke(app, ["stage", "get"])

    assert result.exit_code == 0, result.output
    # The demo reads -0.0 at the origin; the line shows 0.00.
    assert "XY (0.00, 0.00) µm" in result.output.splitlines()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_stage_move_prints_the_readback_at_the_target() -> None:
    result = runner.invoke(app, ["stage", "move", "1234.5", "-56.25"])

    assert result.exit_code == 0, result.output
    x, y = _xy_printed(result.output)
    assert abs(x - 1234.5) < 0.5
    assert abs(y + 56.25) < 0.5


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_stage_jog_takes_negative_distances() -> None:
    result = runner.invoke(app, ["stage", "jog", "-100", "-50"])

    assert result.exit_code == 0, result.output
    x, y = _xy_printed(result.output)
    assert abs(x + 100) < 0.5
    assert abs(y + 50) < 0.5


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_stage_jog_above_the_limit_exits_2_and_names_the_force_flag() -> None:
    result = runner.invoke(app, ["stage", "jog", "6000", "0"])

    assert result.exit_code == 2, result.output
    assert (
        "✗ jog (6000.0, 0.0) µm exceeds the jog limit of 5000.0 µm — re-run with "
        "--force if the distance is intended"
    ) in result.output.splitlines()
    assert "force=True" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_stage_jog_with_force_passes_the_guard() -> None:
    result = runner.invoke(app, ["stage", "jog", "6000", "0", "--force"])

    assert result.exit_code == 0, result.output
    x, y = _xy_printed(result.output)
    assert abs(x - 6000) < 0.5
    assert abs(y) < 0.5


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_stage_jog_dry_run_prints_the_target_and_leaves_the_stage() -> None:
    result = runner.invoke(app, ["stage", "jog", "100", "0", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert (
        "[dry-run] would move to XY (100.00, 0.00) µm; the stage is at "
        "XY (0.00, 0.00) µm"
    ) in result.output.splitlines()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_non_finite_target_exits_2() -> None:
    result = runner.invoke(app, ["stage", "move", "nan", "0"])

    assert result.exit_code == 2, result.output
    assert "is not a finite position" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_z_move_and_jog_print_the_readback() -> None:
    moved = runner.invoke(app, ["z", "move", "12.5"])
    jogged = runner.invoke(app, ["z", "jog", "-2.5"])

    assert moved.exit_code == 0, moved.output
    assert "Z 12.50 µm" in moved.output.splitlines()
    assert jogged.exit_code == 0, jogged.output
    assert abs(_z_printed(jogged.output) + 2.5) < 0.5


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_z_move_outside_the_soft_limits_exits_2_without_a_force_hint() -> None:
    result = runner.invoke(app, ["z", "move", "5000"])

    assert result.exit_code == 2, result.output
    assert "outside the soft limits [-1000.0, 1000.0] µm" in result.output
    assert "--force" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
@pytest.mark.parametrize("distance", ["150", "-150"])
def test_z_jog_above_the_limit_exits_2_and_names_the_force_flag(distance: str) -> None:
    result = runner.invoke(app, ["z", "jog", distance])

    assert result.exit_code == 2, result.output
    assert (
        f"✗ Z jog {float(distance)} µm exceeds the Z jog limit of 100.0 µm — "
        "re-run with --force if the distance is intended"
    ) in result.output.splitlines()
    assert "force=True" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_z_jog_with_force_passes_the_guard() -> None:
    result = runner.invoke(app, ["z", "jog", "150", "--force"])

    assert result.exit_code == 0, result.output
    assert abs(_z_printed(result.output) - 150) < 0.5


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_forced_z_jog_still_obeys_the_soft_limits() -> None:
    result = runner.invoke(app, ["z", "jog", "5000", "--force"])

    assert result.exit_code == 2, result.output
    assert "outside the soft limits [-1000.0, 1000.0] µm" in result.output
    # The soft limit cannot be forced, so the line must not suggest it.
    assert "--force" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_z_jog_refusal_says_when_the_limit_is_assumed(tmp_path: Path) -> None:
    # FM-36: "[safety]" is bracketed text that rich would read as markup and drop.
    path = tmp_path / "nokey.toml"
    path.write_text('[microscope]\nname = "nokey"\n', encoding="utf-8")

    result = runner.invoke(app, ["z", "jog", "150", "-p", str(path)])

    assert result.exit_code == 2, result.output
    assert (
        "✗ Z jog 150.0 µm exceeds the Z jog limit of 100.0 µm (assumed: the "
        "profile sets no [safety] max_z_jog_um) — re-run with --force if the "
        "distance is intended"
    ) in result.output.splitlines()


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_z_jog_dry_run_refuses_an_oversized_jog() -> None:
    result = runner.invoke(app, ["z", "jog", "150", "--dry-run"])

    assert result.exit_code == 2, result.output
    assert "exceeds the Z jog limit of 100.0 µm" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_z_jog_dry_run_leaves_the_drive() -> None:
    result = runner.invoke(app, ["z", "jog", "5", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert (
        "[dry-run] would move to Z 5.00 µm; the drive is at Z 0.00 µm"
        in result.output.splitlines()
    )


# --- errors and the release of the stand ---------------------------------------


@pytest.fixture
def close_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``Microscope.close`` calls; the real close still runs."""
    calls: list[int] = []
    real_close = Microscope.close

    def counting_close(self: Microscope) -> None:
        calls.append(1)
        real_close(self)

    monkeypatch.setattr(Microscope, "close", counting_close)
    return calls


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_missing_role_exits_1_and_points_at_roles_assign(noxy_profile: Path) -> None:
    result = runner.invoke(app, ["stage", "get", "-p", str(noxy_profile)])

    assert result.exit_code == 1, result.output
    assert "XYStage needs the 'xy_stage' role" in result.output
    assert "[roles.assign]" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_the_stand_is_released_after_a_refusal(close_calls: list[int]) -> None:
    result = runner.invoke(app, ["stage", "jog", "6000", "0"])

    assert result.exit_code == 2, result.output
    assert len(close_calls) == 1


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_ctrl_c_during_a_move_exits_130_and_releases_the_stand(
    monkeypatch: pytest.MonkeyPatch, close_calls: list[int]
) -> None:
    def interrupted(self: MMXYStage, x_um: float, y_um: float) -> XY:
        raise KeyboardInterrupt

    monkeypatch.setattr(MMXYStage, "move_to_um", interrupted)

    result = runner.invoke(app, ["stage", "move", "10", "0"])

    assert result.exit_code == 130, result.output
    assert "✗ interrupted — any move under way was stopped" in result.output
    assert len(close_calls) == 1
    assert "Traceback" not in result.output
    assert "Aborted" not in result.output


@pytest.fixture
def broken_position(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: MMXYStage) -> XY:
        raise ValueError("boom")

    monkeypatch.setattr(MMXYStage, "position_um", boom)


@pytest.mark.demo
@pytest.mark.usefixtures("demo", "broken_position")
def test_an_unexpected_error_is_one_line_that_points_at_debug() -> None:
    result = runner.invoke(app, ["stage", "get"])

    assert result.exit_code == 1, result.output
    assert "unexpected ValueError: boom" in result.output
    assert "smc --debug" in result.output
    assert "Traceback" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo", "broken_position")
def test_debug_lets_the_traceback_through() -> None:
    result = runner.invoke(app, ["--debug", "stage", "get"])

    assert isinstance(result.exception, ValueError), result.output


class _FaultyCore:
    """The demo core, except that reading the X position raises, as a dead serial line does."""

    def __init__(self, core: object) -> None:
        self._core = core

    def __getattr__(self, name: str) -> object:
        return getattr(self._core, name)

    def getXPosition(self, label: str | None = None) -> float:  # noqa: N802 - MMCore's name
        raise RuntimeError("[COM3] Serial port timed out")


@pytest.fixture
def faulty_core(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fault is raised by the *core*: patching MMXYStage.position_um, as
    # broken_position does, would skip the layer this test is about (FM-79).
    real_open_core = microscope_mod.open_core

    def wrapped(config: object = None, **kwargs: object) -> _FaultyCore:
        return _FaultyCore(real_open_core(config, **kwargs))  # type: ignore[arg-type]

    monkeypatch.setattr(microscope_mod, "open_core", wrapped)


@pytest.mark.demo
@pytest.mark.usefixtures("demo", "faulty_core")
def test_a_device_fault_exits_1_naming_the_role_and_the_call() -> None:
    result = runner.invoke(app, ["stage", "get"])

    assert result.exit_code == 1, result.output
    # The bracket proves the message is printed as text, not as rich markup (FM-36).
    assert (
        "✗ xy_stage: getXPosition('XY') failed ([COM3] Serial port timed out); "
        "check the device: power, cable, another program holding it"
    ) in result.output
    assert "unexpected" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo", "faulty_core")
def test_debug_shows_the_device_fault_chained_from_the_core_error() -> None:
    result = runner.invoke(app, ["--debug", "stage", "get"])

    assert type(result.exception) is HardwareError, result.output
    assert type(result.exception.__cause__) is RuntimeError


@pytest.mark.demo
@pytest.mark.usefixtures("demo", "faulty_core")
def test_doctor_prints_the_translated_fault_it_read() -> None:
    result = runner.invoke(app, ["doctor", "-p", "demo"])

    assert result.exit_code == 1, result.output
    # Nothing else pinned this line: the profile path already reports a device
    # fault, and a later change to the proxy must not turn it into "unexpected".
    assert any(
        line.startswith(
            "✗ xy: HardwareError: xy_stage: getXPosition('XY') failed "
            "([COM3] Serial port timed out)"
        )
        for line in result.output.splitlines()
    ), result.output
    assert "unexpected" not in result.output


# --- doctor --config -------------------------------------------------------------


#: Printed by the fake when the core is released, so that the output shows
#: whether the stand was let go before or after the ``✗`` line.
_UNLOADED = "-- unloadAllDevices --"

#: What the demo's configuration answers for each row of the role table.
_DEMO_CORE_ROLES = {
    "camera": "Camera",
    "xy stage": "XY",
    "focus": "Z",
    "autofocus": "Autofocus",
    "shutter": "White Light Shutter",
}


class _FailingCore:
    """The demo core, except that the named calls raise; it counts calls and releases.

    ``doctor --config`` has no profile, so what fails is chosen per MMCore call:
    ``getXYPosition`` is the serial read of a dead line, ``getFocusDevice``
    stands for a bug that is not a device fault. ``calls`` is how a test sees
    that one read decided both the table and the verdict (FM-39).
    """

    def __init__(self, core: object, fails: dict[str, Exception]) -> None:
        self._core = core
        self._fails = fails
        self.calls: Counter[str] = Counter()
        self.unloads = 0

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._core, name)
        if not callable(attr):
            return attr

        def call(*args: object, **kwargs: object) -> object:
            self.calls[name] += 1
            if name in self._fails:
                raise self._fails[name]
            return attr(*args, **kwargs)

        return call

    def unloadAllDevices(self) -> None:  # noqa: N802 - MMCore's name
        self.unloads += 1
        print(_UNLOADED)
        self._core.unloadAllDevices()  # type: ignore[attr-defined]


@pytest.fixture
def config_core(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[_FailingCore]]:
    """Make ``doctor --config`` open the demo core, with the named calls failing.

    The seam is ``core.open_core``, which ``core.opened`` calls: the demo is
    opened whatever path the command is given, so no real stand and no
    install detection is involved (FM-40). The returned list holds the cores
    the command opened. The XY stage sits at (10, 20) µm, so that a swapped X
    and Y cannot read the same; ``no_xy_stage=True`` clears the core's XY
    stage, as on a stand that has none.
    """
    real_open_core = core_mod.open_core

    def install(*, no_xy_stage: bool = False, **fails: Exception) -> list[_FailingCore]:
        opened: list[_FailingCore] = []

        def wrapped(config: object = None, **kwargs: object) -> _FailingCore:
            real = real_open_core(None)
            real.setXYPosition(10.0, 20.0)
            # The demo stage simulates its travel: read at once, it is
            # a fraction of the way there.
            real.waitForSystem()
            if no_xy_stage:
                real.setXYStageDevice("")
            core = _FailingCore(real, fails)
            opened.append(core)
            return core

        monkeypatch.setattr(core_mod, "open_core", wrapped)
        return opened

    return install


def _doctor_config(tmp_path: Path, *options: str) -> Result:
    return runner.invoke(
        app, [*options, "doctor", "--config", str(tmp_path / "demo.cfg")]
    )


def _fail_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.lstrip().startswith("✗")]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_config_prints_the_core_roles_and_the_position(
    tmp_path: Path, config_core: Callable[..., list[_FailingCore]]
) -> None:
    opened = config_core()

    result = _doctor_config(tmp_path)

    assert result.exit_code == 0, result.output
    # One row per role, from its own call: the labels differ, so a row fed by
    # the wrong getter shows.
    for role, label in _DEMO_CORE_ROLES.items():
        assert re.search(rf"^\s*{role}\s+{label}\s*$", result.output, re.M), role
    # X before Y. The decimals are #94's (it may print two): not pinned.
    assert re.search(r"^\s*xy position\s+\(10\.0+, 20\.0+\)", result.output, re.M), (
        result.output
    )
    assert "loads and answers" in result.output
    # One read of each decides the table and the verdict (FM-39): a flaky line
    # cannot show "xy stage XY" and then skip the position, or the reverse.
    (core,) = opened
    reads = ["getCameraDevice", "getXYStageDevice", "getFocusDevice"]
    reads += ["getAutoFocusDevice", "getShutterDevice", "getXYPosition"]
    assert {name: core.calls[name] for name in reads} == dict.fromkeys(reads, 1)


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_config_without_an_xy_stage_reads_no_position_and_passes(
    tmp_path: Path, config_core: Callable[..., list[_FailingCore]]
) -> None:
    # MMCore raises on getXYPosition() with no XY stage; the proxy would make
    # that a ✗, and a healthy camera-only stand would fail a script that gates
    # on doctor.
    opened = config_core(
        no_xy_stage=True, getXYPosition=RuntimeError('No device with label ""')
    )

    result = _doctor_config(tmp_path)

    assert result.exit_code == 0, result.output
    assert re.search(r"^\s*xy stage\s+— not set —\s*$", result.output, re.M), (
        result.output
    )
    assert "xy position" not in result.output
    assert _fail_lines(result.output) == []
    assert "loads and answers" in result.output
    assert opened[0].calls["getXYPosition"] == 0


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_config_device_fault_exits_1_naming_the_call(
    tmp_path: Path, config_core: Callable[..., list[_FailingCore]]
) -> None:
    config_core(getXYPosition=RuntimeError("[COM3] Serial port timed out"))

    result = _doctor_config(tmp_path)

    assert result.exit_code == 1, result.output
    # The bracket proves the message is printed as text, not as rich markup
    # (FM-36). The "check the device" tail is #121's to reword: not pinned.
    assert (
        "✗ core: getXYPosition() failed ([COM3] Serial port timed out)" in result.output
    )
    assert len(_fail_lines(result.output)) == 1, result.output
    for absent in ("unexpected", "Traceback", "loads and answers"):
        assert absent not in result.output
    # The operator sees which device is the XY stage, hence which cable; the
    # position row is the one that is missing.
    stage = re.search(r"^\s*xy stage\s+XY\s*$", result.output, re.M)
    assert stage is not None, result.output
    assert stage.start() < result.output.index("✗ core:"), result.output
    assert "xy position" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_config_releases_the_core_when_a_read_faults(
    tmp_path: Path, config_core: Callable[..., list[_FailingCore]]
) -> None:
    # A Nikon stand takes one connection at a time: the core is released
    # before the next command, or the next smc cannot connect. And before the
    # line is printed (design §9), so the diagnosis never waits on a release
    # that blocks on a dead serial line.
    opened = config_core(getXYPosition=RuntimeError("[COM3] Serial port timed out"))

    result = _doctor_config(tmp_path)

    assert result.exit_code == 1, result.output
    assert [core.unloads for core in opened] == [1]
    assert result.output.index(_UNLOADED) < result.output.index("✗ core:")


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
@pytest.mark.parametrize("failing_call", ["getFocusDevice", "getXYPosition"])
def test_doctor_config_an_unexpected_error_is_one_line(
    failing_call: str,
    tmp_path: Path,
    config_core: Callable[..., list[_FailingCore]],
) -> None:
    # Not a device fault, so the proxy lets it through: what reports it is
    # _reported around the branch, like every other command. The position read
    # catches a HardwareError and nothing wider: a bug there is not turned into
    # a finding under the role table.
    config_core(**{failing_call: ValueError("boom")})

    result = _doctor_config(tmp_path)

    assert result.exit_code == 1, result.output
    assert "✗ unexpected ValueError: boom" in result.output
    assert "Traceback" not in result.output
    assert "Core roles" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
@pytest.mark.parametrize(
    "failing_call",
    [
        "getCameraDevice",
        "getXYStageDevice",
        "getFocusDevice",
        "getAutoFocusDevice",
        "getShutterDevice",
    ],
)
def test_doctor_config_a_registry_read_that_faults_is_one_line(
    failing_call: str,
    tmp_path: Path,
    config_core: Callable[..., list[_FailingCore]],
) -> None:
    # Each of the five reads goes through the proxy: left on the raw core, a
    # serial fault there would be "unexpected RuntimeError ... file an issue".
    config_core(**{failing_call: RuntimeError("[COM3] Serial port timed out")})

    result = _doctor_config(tmp_path)

    assert result.exit_code == 1, result.output
    assert (
        f"✗ core: {failing_call}() failed ([COM3] Serial port timed out)"
        in result.output
    )
    assert len(_fail_lines(result.output)) == 1, result.output
    assert "unexpected" not in result.output
    # The registry is read before the table: a fault there prints no table.
    assert "Core roles" not in result.output
    assert "xy stage" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_debug_shows_the_doctor_config_fault_chained_from_the_core_error(
    tmp_path: Path, config_core: Callable[..., list[_FailingCore]]
) -> None:
    config_core(getXYPosition=RuntimeError("[COM3] Serial port timed out"))

    result = _doctor_config(tmp_path, "--debug")

    assert type(result.exception) is HardwareError, result.output
    assert type(result.exception.__cause__) is RuntimeError


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_verbose_logs_each_command_sent() -> None:
    logged = runner.invoke(app, ["-v", "stage", "move", "10", "20"])
    quiet = runner.invoke(app, ["stage", "move", "10", "20"])

    assert logged.exit_code == 0, logged.output
    assert "xy_stage: move_to (10.0, 20.0) µm" in logged.output
    assert quiet.exit_code == 0, quiet.output
    assert "xy_stage: move_to" not in quiet.output


# --- snap ----------------------------------------------------------------------


@pytest.mark.parametrize("where", ["missing-folder", "png-suffix", "directory"])
def test_snap_bad_out_exits_2_before_opening_the_stand(
    where: str, tmp_path: Path, no_stand: None
) -> None:
    out = {
        "missing-folder": tmp_path / "nowhere" / "frame.tif",
        "png-suffix": tmp_path / "frame.png",
        "directory": tmp_path / "frames.tif",
    }[where]
    if where == "directory":
        out.mkdir()

    result = runner.invoke(app, ["snap", "--out", str(out)])

    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "60001"])
def test_snap_refuses_an_exposure_out_of_range(
    value: str, tmp_path: Path, no_stand: None
) -> None:
    out = tmp_path / "frame.tif"

    result = runner.invoke(app, ["snap", "--out", str(out), "--exposure-ms", value])

    assert result.exit_code == 2, result.output
    assert "at most 60000 ms" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_snap_writes_a_16_bit_tiff_with_its_metadata(tmp_path: Path) -> None:
    out = tmp_path / "frame.tif"

    result = runner.invoke(app, ["snap", "--out", str(out), "--exposure-ms", "25"])

    assert result.exit_code == 0, result.output
    with tifffile.TiffFile(out) as tif:
        frame = tif.asarray()
        meta = tif.shaped_metadata[0]
    assert frame.dtype == np.uint16
    assert frame.shape == (512, 512)
    assert meta["exposure_ms"] == 25.0
    assert meta["pixel_size_um"] == 1.0
    assert meta["profile"] == "demo"
    assert meta["camera"] == "Camera"
    assert meta["xy_um"] == [0.0, 0.0]
    assert "512x512 uint16, exposure 25 ms" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_snap_write_failure_still_prints_the_frame_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tifffile, "imwrite", denied)

    result = runner.invoke(app, ["snap", "--out", str(tmp_path / "frame.tif")])

    assert result.exit_code == 1, result.output
    assert "snapped (not written): 512x512 uint16" in result.output
    assert "✗ could not write" in result.output
    # FM-36: the bracketed errno is not eaten as markup.
    assert "[Errno 13]" in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_snap_overwrites_an_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "frame.tif"
    out.write_bytes(b"an earlier capture")

    result = runner.invoke(app, ["snap", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert tifffile.imread(out).shape == (512, 512)
    assert [p.name for p in tmp_path.iterdir()] == ["frame.tif"]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_an_interrupted_write_keeps_the_earlier_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tifffile opens its target with "wb": a Ctrl-C half-way through the
    # write would otherwise leave a truncated file where a good one was.
    out = tmp_path / "frame.tif"
    out.write_bytes(b"an earlier capture")

    def interrupted(path: str | Path, *args: object, **kwargs: object) -> None:
        Path(path).write_bytes(b"\x00" * 8)
        raise KeyboardInterrupt

    monkeypatch.setattr(tifffile, "imwrite", interrupted)

    result = runner.invoke(app, ["snap", "--out", str(out)])

    assert result.exit_code == 130, result.output
    assert out.read_bytes() == b"an earlier capture"
    assert [p.name for p in tmp_path.iterdir()] == ["frame.tif"]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_failed_replace_keeps_the_frame_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows refuses to replace a TIFF that Fiji holds open. The frame was
    # written in full next to it and must not be deleted with the error.
    out = tmp_path / "frame.tif"
    out.write_bytes(b"an earlier capture")

    def held_open(src: object, dst: object) -> None:
        # As os.replace raises it: str() names both files ('src' -> 'dst').
        raise PermissionError(13, "Permission denied", str(src), None, str(dst))

    monkeypatch.setattr(os, "replace", held_open)

    result = runner.invoke(app, ["snap", "--out", str(out)])

    assert result.exit_code == 1, result.output
    assert out.read_bytes() == b"an earlier capture"
    kept = [p for p in tmp_path.iterdir() if p != out]
    assert len(kept) == 1, kept
    assert tifffile.imread(kept[0]).shape == (512, 512)
    assert "✗ could not write" in result.output
    assert f"Permission denied; the new file is kept as {kept[0]}" in result.output
    assert result.output.count(kept[0].name) == 1, result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_ctrl_c_before_the_rename_removes_the_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing names a file left behind by an interrupt, so none is left.
    out = tmp_path / "frame.tif"
    out.write_bytes(b"an earlier capture")

    def interrupted(src: object, dst: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)

    result = runner.invoke(app, ["snap", "--out", str(out)])

    assert result.exit_code == 130, result.output
    assert out.read_bytes() == b"an earlier capture"
    assert [p.name for p in tmp_path.iterdir()] == ["frame.tif"]


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
@pytest.mark.skipif(
    sys.platform == "win32", reason="a Windows file takes its ACLs from the folder"
)
def test_snap_gives_the_tiff_the_mode_of_a_plain_write(tmp_path: Path) -> None:
    # A temporary file is created 0600; replaced onto --out, it would leave
    # every capture readable by its owner only.
    out = tmp_path / "frame.tif"
    previous = os.umask(0o022)
    try:
        result = runner.invoke(app, ["snap", "--out", str(out)])
    finally:
        os.umask(previous)

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(out.stat().st_mode) == 0o644


# --- doctor on a Nikon stand (design m2 §7.4) ---------------------------------------
# No Nikon adapter is ever loaded (FM-40): the open is patched to fail the way
# MMCore does, the loader probe is canned, and every folder is under tmp_path.

TI2_ADAPTER = "mmgr_dal_NikonTi2.dll"
TI2_DRIVER = "Ti2_Mic_Driver.dll"
LOAD_ERROR = (
    "could not load {path}: Line 4: Device,Ti2-E__0,NikonTi2,Ti2-E__0\n"
    'Failed to load device adapter "NikonTi2" from {adapter}'
)


@pytest.fixture
def isolated_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both stands at an SDK folder under tmp_path: a Ti2 PC has the real one."""
    sdk = tmp_path / "sdk"
    for stand in nikon.STANDS:
        moved = dataclasses.replace(
            stand, driver=dataclasses.replace(stand.driver, sdk_dir=str(sdk))
        )
        monkeypatch.setitem(nikon._BY_ADAPTER, stand.adapter.casefold(), moved)
        monkeypatch.setitem(nikon._BY_KEY, stand.key, moved)
    return sdk


@pytest.fixture
def mm_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_sdk: Path
) -> Path:
    """An install folder holding the Ti2 adapter, as ``doctor`` finds the install."""
    install = tmp_path / "install"
    install.mkdir()
    (install / TI2_ADAPTER).write_bytes(b"MZ")
    status = core_mod.MicroManagerStatus(
        install_dir=install, api_version="test", core_version="test", adapters=()
    )
    monkeypatch.setattr(core_mod, "status", lambda: status)
    return install


@pytest.fixture
def ti2_profile(tmp_path: Path) -> Path:
    """A profile whose configuration loads a Ti2, and whose own folder holds the adapter."""
    own = tmp_path / "own"
    own.mkdir()
    (own / TI2_ADAPTER).write_bytes(b"MZ")
    (tmp_path / "ti2.cfg").write_text(
        "Device,Ti2-E__0,NikonTi2,Ti2-E__0\nDevice,ZDrive,NikonTi2,ZDrive\n",
        encoding="utf-8",
    )
    path = tmp_path / "ti2-test.toml"
    path.write_text(
        '[microscope]\nname = "ti2-test"\n\n[micromanager]\nconfig = "ti2.cfg"\n'
        'adapter_search_paths = ["own"]\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the loader probe by a canned WinError 126; the list holds what was probed."""
    probed: list[Path] = []

    def canned(path: Path, **kwargs: object) -> DllProbe:
        probed.append(path)
        return DllProbe(path, False, 126, describe_winerror(126, "Module not found."))

    monkeypatch.setattr(dll_probe, "probe_dll", canned)
    return probed


def _fails_to_load(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    """Make ``module.open_core`` fail as MMCore does for an adapter that will not load."""

    def fail(config: object = None, **kwargs: object) -> object:
        raise CoreError(LOAD_ERROR.format(path=config, adapter=TI2_ADAPTER))

    monkeypatch.setattr(module, "open_core", fail)


def _positions(output: str, *markers: str) -> list[int]:
    """The line number of the first line holding each marker; each must be there."""
    lines = output.splitlines()
    found = []
    for marker in markers:
        hits = [i for i, line in enumerate(lines) if marker in line]
        assert hits, f"{marker!r} is not in the output:\n{output}"
        found.append(hits[0])
    return found


@pytest.mark.usefixtures("mm_install")
def test_doctor_prints_the_nikon_readiness_before_opening(
    ti2_profile: Path, probes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fails_to_load(monkeypatch, microscope_mod)

    result = runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    own = ti2_profile.parent / "own"
    heading, adapter, driver, failure = _positions(
        result.output, "Nikon Ti2-E (NikonTi2)", "  adapter  ", "  driver ", "✗"
    )
    assert heading < adapter < driver < failure
    # The profile's own folder comes first and wins, as MMCore searches it (FM-76).
    assert str(own / TI2_ADAPTER) in result.output.splitlines()[adapter]
    assert f"{TI2_DRIVER} not found" in result.output.splitlines()[driver]


@pytest.mark.usefixtures("mm_install")
def test_doctor_diagnoses_a_nikon_stand_that_would_not_open(
    ti2_profile: Path, probes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fails_to_load(monkeypatch, microscope_mod)

    result = runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    assert result.exit_code == 1, result.output
    (
        failure,
        heading,
        found,
        probe,
        likelihood,
        first,
    ) = _positions(
        result.output,
        "✗ could not load",
        "Nikon Ti2-E: what to check",
        "  found: ",
        "  probe: ",
        "in order of likelihood:",
        "1. another program holds the stand",
    )
    assert failure < heading < found < probe < likelihood < first
    assert first == likelihood + 1
    # MMCore's own message is kept whole, then the findings follow it.
    assert 'Failed to load device adapter "NikonTi2"' in result.output
    # What Windows said, and what it means.
    assert "will not load: WinError 126: Module not found." in result.output
    # The adapter is probed from its first folder, with the driver not found: one probe.
    assert probes == [ti2_profile.parent / "own" / TI2_ADAPTER]


def test_doctor_probes_with_the_adapter_folders_on_the_path(
    ti2_profile: Path, monkeypatch: pytest.MonkeyPatch, mm_install: Path
) -> None:
    _fails_to_load(monkeypatch, microscope_mod)
    seen: list[list[Path]] = []

    def record(
        path: Path, *, extra_path: Sequence[Path] = (), **kwargs: object
    ) -> DllProbe:
        seen.append(list(extra_path))
        return DllProbe(path, True, None, "loads")

    monkeypatch.setattr(dll_probe, "probe_dll", record)

    runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    # The profile's folder, then the install: the order MMCore searches them in.
    assert seen == [[ti2_profile.parent / "own", mm_install]]


@pytest.mark.usefixtures("mm_install")
def test_doctor_reports_another_error_without_a_diagnosis(
    ti2_profile: Path, probes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(config: object = None, **kwargs: object) -> object:
        raise ValueError("boom")

    monkeypatch.setattr(microscope_mod, "open_core", boom)

    result = runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    assert result.exit_code == 1, result.output
    assert "✗ unexpected ValueError: boom" in result.output
    assert "Nikon Ti2-E (NikonTi2)" in result.output
    assert "what to check" not in result.output
    assert probes == []


@pytest.mark.usefixtures("mm_install")
def test_doctor_leaves_the_stand_alone_while_it_diagnoses(
    ti2_profile: Path, probes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The probes run after the open has failed and everything is unloaded, never
    # while a core may still hold the stand.
    events: list[str] = []

    def fail(config: object = None, **kwargs: object) -> object:
        events.append("open")
        raise CoreError(LOAD_ERROR.format(path=config, adapter=TI2_ADAPTER))

    def canned(path: Path, **kwargs: object) -> DllProbe:
        events.append("probe")
        return DllProbe(path, True, None, "loads")

    monkeypatch.setattr(microscope_mod, "open_core", fail)
    monkeypatch.setattr(dll_probe, "probe_dll", canned)

    runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    assert events == ["open", "probe"]


@pytest.mark.usefixtures("mm_install")
def test_doctor_a_diagnosis_that_fails_does_not_hide_the_open_error(
    ti2_profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fails_to_load(monkeypatch, microscope_mod)

    def broken(path: Path, **kwargs: object) -> DllProbe:
        raise RuntimeError("probe bug")

    monkeypatch.setattr(dll_probe, "probe_dll", broken)

    result = runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    assert result.exit_code == 1, result.output
    assert "✗ could not load" in result.output
    assert "could not diagnose the stand: probe bug" in result.output


def test_doctor_config_diagnoses_a_nikon_cfg_that_would_not_open(
    tmp_path: Path,
    mm_install: Path,
    probes: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "stand.cfg"
    cfg.write_text("Device,Ti2-E__0,NikonTi2,Ti2-E__0\n", encoding="utf-8")
    _fails_to_load(monkeypatch, core_mod)

    result = runner.invoke(app, ["doctor", "--config", str(cfg)])

    assert result.exit_code == 1, result.output
    heading, adapter, failure, diagnosis = _positions(
        result.output, "Nikon Ti2-E (NikonTi2)", "  adapter  ", "✗", "what to check"
    )
    assert heading < adapter < failure < diagnosis
    # With --config the install alone is searched.
    assert str(mm_install / TI2_ADAPTER) in result.output.splitlines()[adapter]
    assert "in order of likelihood:" in result.output
    assert probes == [mm_install / TI2_ADAPTER]


@pytest.mark.usefixtures("mm_install")
def test_doctor_config_without_a_nikon_device_prints_no_nikon_block(
    tmp_path: Path, probes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "other.cfg"
    cfg.write_text("Device,Camera,DemoCamera,DCam\n", encoding="utf-8")

    def fail(config: object = None, **kwargs: object) -> object:
        raise CoreError('Failed to load device adapter "DemoCamera"')

    monkeypatch.setattr(core_mod, "open_core", fail)

    result = runner.invoke(app, ["doctor", "--config", str(cfg)])

    assert result.exit_code == 1, result.output
    assert 'Failed to load device adapter "DemoCamera"' in result.output
    assert "Nikon" not in result.output
    assert "what to check" not in result.output
    assert probes == []


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_doctor_without_a_nikon_stand_prints_no_nikon_block() -> None:
    result = runner.invoke(app, ["doctor", "-p", "demo"])

    assert result.exit_code == 0, result.output
    assert "Nikon" not in result.output
    assert "what to check" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("demo", "isolated_sdk")
def test_nikon_readiness_rows_never_change_the_exit_code(
    ti2_profile: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stand that opens is a stand that opens, whatever the file checks said:
    # the open is replaced by the demo's, so no Nikon adapter is loaded (FM-40).
    real_open_core = core_mod.open_core
    monkeypatch.setattr(
        microscope_mod, "open_core", lambda config=None, **kw: real_open_core(None)
    )
    status = core_mod.status()
    monkeypatch.setattr(
        core_mod,
        "status",
        lambda: core_mod.MicroManagerStatus(
            install_dir=tmp_path / "no-adapter-here",
            api_version=status.api_version,
            core_version=status.core_version,
            adapters=status.adapters,
        ),
    )
    (tmp_path / "own" / TI2_ADAPTER).unlink()

    result = runner.invoke(app, ["doctor", "-p", str(ti2_profile)])

    assert result.exit_code == 0, result.output
    assert "adapter  not found in" in result.output
    assert "loads and answers" in result.output


# --- install-driver (design m2 §7.5) ---------------------------------------------------


@pytest.fixture
def no_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command looks for the Micro-Manager install."""

    def refuse() -> Path | None:
        pytest.fail("the command looked for the Micro-Manager install")

    monkeypatch.setattr(core_mod, "find_install", refuse)


def _driver(folder: Path, data: bytes = b"vendor driver") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / TI2_DRIVER
    path.write_bytes(data)
    return path


@pytest.mark.usefixtures("no_install")
def test_install_driver_copies_into_the_folder_given(tmp_path: Path) -> None:
    source = _driver(tmp_path / "sdk")
    to = tmp_path / "mm"
    to.mkdir()

    result = runner.invoke(
        app, ["install-driver", "ti2", "--from", str(source), "--to", str(to)]
    )

    assert result.exit_code == 0, result.output
    assert f"✓ copied {source} to {to / TI2_DRIVER}" in result.output
    assert (to / TI2_DRIVER).read_bytes() == b"vendor driver"
    # The vendor's own software still needs its copy.
    assert source.read_bytes() == b"vendor driver"


def test_install_driver_defaults_to_the_adapter_folder_of_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install = tmp_path / "install"
    install.mkdir()
    (install / TI2_ADAPTER).write_bytes(b"MZ")
    monkeypatch.setattr(core_mod, "find_install", lambda: install)
    source = _driver(tmp_path / "sdk")

    result = runner.invoke(app, ["install-driver", "ti2", "--from", str(source)])

    assert result.exit_code == 0, result.output
    assert (install / TI2_DRIVER).read_bytes() == b"vendor driver"


def test_install_driver_takes_the_sdk_folder_by_default(
    tmp_path: Path, isolated_sdk: Path
) -> None:
    _driver(isolated_sdk, b"from the sdk")
    to = tmp_path / "mm"
    to.mkdir()

    result = runner.invoke(app, ["install-driver", "ti2", "--to", str(to)])

    assert result.exit_code == 0, result.output
    assert (to / TI2_DRIVER).read_bytes() == b"from the sdk"


def test_install_driver_without_the_adapter_says_to_pass_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty folder, whatever OS the test runs on: macOS has no Nikon
    # adapter and the Windows runner has the full nightly (FM-43).
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setattr(core_mod, "find_install", lambda: install)
    source = _driver(tmp_path / "sdk")

    result = runner.invoke(app, ["install-driver", "ti2", "--from", str(source)])

    assert result.exit_code == 1, result.output
    assert f"✗ {TI2_ADAPTER} is not in the Micro-Manager install" in result.output
    assert "pass --to" in result.output
    assert not (install / TI2_DRIVER).exists()


def test_install_driver_without_any_install_says_to_pass_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core_mod, "find_install", lambda: None)
    source = _driver(tmp_path / "sdk")

    result = runner.invoke(app, ["install-driver", "ti2", "--from", str(source)])

    assert result.exit_code == 1, result.output
    assert "no Micro-Manager install" in result.output
    assert "pass --to" in result.output


@pytest.mark.usefixtures("no_install")
def test_install_driver_tie_exits_1_naming_the_system_path(tmp_path: Path) -> None:
    # The Ti-E adapter finds NikonTi.dll on the system path: nothing to copy,
    # and no look at the install either.
    result = runner.invoke(app, ["install-driver", "tie"])

    assert result.exit_code == 1, result.output
    assert "system path" in result.output
    assert nikon.TIE.driver.installer in result.output


@pytest.mark.usefixtures("no_install")
def test_install_driver_never_overwrites_a_different_dll(tmp_path: Path) -> None:
    source = _driver(tmp_path / "sdk", b"the sdk version")
    to = tmp_path / "mm"
    there = _driver(to, b"another version")

    result = runner.invoke(
        app, ["install-driver", "ti2", "--from", str(source), "--to", str(to)]
    )

    assert result.exit_code == 1, result.output
    assert "differs" in result.output
    assert there.read_bytes() == b"another version"


@pytest.mark.usefixtures("no_install")
def test_install_driver_is_content_with_an_identical_copy(tmp_path: Path) -> None:
    source = _driver(tmp_path / "sdk", b"same")
    to = tmp_path / "mm"
    _driver(to, b"same")

    result = runner.invoke(
        app, ["install-driver", "ti2", "--from", str(source), "--to", str(to)]
    )

    assert result.exit_code == 0, result.output
    assert "already there" in result.output


@pytest.mark.usefixtures("no_install")
def test_install_driver_with_a_missing_source_names_the_installer_and_from(
    tmp_path: Path,
) -> None:
    to = tmp_path / "mm"
    to.mkdir()
    missing = tmp_path / "nowhere" / TI2_DRIVER

    result = runner.invoke(
        app, ["install-driver", "ti2", "--from", str(missing), "--to", str(to)]
    )

    assert result.exit_code == 1, result.output
    assert f"{missing} does not exist" in result.output
    assert nikon.TI2.driver.installer in result.output
    assert "--from" in result.output


@pytest.mark.usefixtures("no_install")
def test_install_driver_to_something_that_is_not_a_folder_exits_1(
    tmp_path: Path,
) -> None:
    source = _driver(tmp_path / "sdk")
    missing = tmp_path / "no-such-folder"

    result = runner.invoke(
        app, ["install-driver", "ti2", "--from", str(source), "--to", str(missing)]
    )

    assert result.exit_code == 1, result.output
    assert f"{missing} is not a folder" in result.output


@pytest.mark.usefixtures("no_install")
def test_install_driver_of_an_unknown_stand_is_a_usage_error_naming_the_stands() -> (
    None
):
    result = runner.invoke(app, ["install-driver", "zeiss"])

    assert result.exit_code == 2, result.output
    assert "unknown stand 'zeiss'" in result.output
    assert "ti2" in result.output
    assert "tie" in result.output


def test_install_driver_help_documents_from_and_to() -> None:
    result = runner.invoke(app, ["install-driver", "--help"])

    assert result.exit_code == 0, result.output
    plain = _ANSI.sub("", result.output)
    # One row per option carries the flag and its own help text (FM-72).
    for flag, text in (
        ("--from", "The driver DLL to copy"),
        ("--to", "The folder of the stand's adapter"),
    ):
        rows = [
            row
            for row in plain.splitlines()
            if re.search(rf"{flag}\s", row) and text in row
        ]
        assert rows, (flag, plain)
