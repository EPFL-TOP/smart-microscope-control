"""The ``smc`` command line (design §9): CliRunner on the demo devices.

The runner pins the width (FM-37: rich wraps at 80 columns when it is not
writing to a terminal) and clears ``SMC_PROFILE``, so an operator's
``SMC_PROFILE=nikon-ti2`` never makes ``pytest`` open a real stand (FM-40).
"""

from __future__ import annotations

import logging
import os
import re
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from smc import __version__
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.backends.mm import MMXYStage
from smc.hardware.capabilities import XY
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
        (["z", "jog", "--help"], "--force"),
        (["snap", "--help"], "ms"),
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert unit in result.output, args


def test_z_jog_help_documents_the_force_option() -> None:
    result = runner.invoke(app, ["z", "jog", "--help"])

    assert result.exit_code == 0, result.output
    # A phrase only the option's own help supplies: the command's docstring
    # also says "--force", so that word alone would pass without the option.
    assert (
        "Move further than the profile's safety.max_z_jog_um allows." in result.output
    )


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
