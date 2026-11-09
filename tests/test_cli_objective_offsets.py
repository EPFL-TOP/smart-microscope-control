"""``smc run objective-offsets`` and ``smc objective offset --from-run-doc`` on the demo-sample stand (m3 §20.7, #380).

The profile is a file of this module's own: ``demo-sample.toml`` with its ``[sample]``
turned as the design measured it (camera turned 30° and mirrored, plate turned 1.2°, a
shift per objective) and A1 at the stage origin, because each ``smc run`` opens a stand
at the origin with the demo's 10 mm/s: the wells it measures must be near it. A module
fixture builds the run documents once through ``run_plugin`` (the seam, FM-124) on a
faster stage: ``camera-calibration`` under the 40X, and ``camera-calibration`` then
``plate-walls`` on A1 and B2 under the 10X. The command line then runs on that.

The expected offsets are the declared shifts' differences, written out: the 40X's
``(-32.5, 9.75)`` minus the 10X's ``(5.0, -2.5)`` is ``(-37.5, 12.25)``. They are
compared within 1.0 µm, one pixel of the 10X: ``-37.494`` is written ``-37.49`` and
never compared as text. Every command's output is read as text, with markup off
(FM-36), and the profile's bytes are compared before and after.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import tomli_w
from typer.testing import CliRunner, Result

from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import ObjectiveOffsetEntry, ObjectivesSection, Profile
from smc.hardware.profile_write import BLOCK_BEGIN, BLOCK_END, objectives_block
from smc.plugins import run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.slots import ObjectiveOffsetsResult, WellRef
from smc.run import RunDocument, StageStatus

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

pytestmark = pytest.mark.demo

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
SHIFT = {TEN: [5.0, -2.5], TWENTY: [0.0, 0.0], FORTY: [-32.5, 9.75]}
PLATE_DEG = 1.2
PITCH_UM = 9000.0
#: Where the operator "centred" each well: a few hundred µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "B2": (-250.0, 150.0)}
SESSION = "session 3 (#412)"


def _truth(well: str) -> tuple[float, float]:
    """A1 is at the origin; B2 is one column right and one row down, turned with the plate."""
    col, row = {"A1": (0, 0), "B2": (1, 1)}[well]
    angle = math.radians(PLATE_DEG)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        math.cos(angle) * dx - math.sin(angle) * dy,
        math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _profile_text(*, reference: str = "", objectives: str = "") -> str:
    """``demo-sample`` turned as the design measured it, A1 at the origin: a profile file's text."""
    data = tomllib.loads(DEMO_SAMPLE.read_text(encoding="utf-8"))
    data["sample"].update(
        {
            "a1_center_xy_um": [0.0, 0.0],
            "rotation_deg": PLATE_DEG,
            "camera_rotation_deg": 30.0,
            "mirrored": True,
            "objective_shift_um": SHIFT,
        }
    )
    if reference:
        data["objectives"] = {"reference": reference}
    return tomli_w.dumps(data) + objectives


@dataclass(frozen=True)
class Documents:
    """The profile and the run documents the commands start from."""

    profile: Path
    run: Path
    forty: Path


@pytest.fixture(scope="module")
def documents(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Documents:
    """The calibration of the 40X and the registration under the 10X, built through ``run_plugin``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("cli-offsets")
    profile = folder / "stand.toml"
    profile.write_text(_profile_text(), encoding="utf-8")
    docs = Documents(profile, folder / "run.json", folder / "cal-40x.json")
    stand = Microscope.open(Profile.load(profile))
    try:
        # The demo stage moves at its `Velocity` property, 10 mm/s by default: this
        # stand builds the documents, and the commands below open their own.
        stand.core.setProperty("XY", "Velocity", "100000")
        turret = stand.require(ObjectiveTurret)
        turret.select(FORTY, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.forty
        )
        assert record.status is StageStatus.done
        turret.select(TEN, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.run
        )
        assert record.status is StageStatus.done
        refs = tuple(
            WellRef(
                well=well,
                x_um=_truth(well)[0] + ROUGH_UM[well][0],
                y_um=_truth(well)[1] + ROUGH_UM[well][1],
            )
            for well in ("A1", "B2")
        )
        record = run_plugin(
            PlateWallsPlugin(),
            stand,
            PlateWallsParams(plate="96-well", refs=refs),
            docs.run,
        )
        assert record.status is StageStatus.done
    finally:
        stand.close()
    return docs


#: Typer forces colour when ``GITHUB_ACTIONS`` is set, and rich styles an option name in
#: pieces: the text is compared with the escape codes stripped (FM-72).
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _out(result: Result) -> str:
    """What the command printed, whole and plain: a ``✗`` line is on the same stream a script reads."""
    return _ANSI.sub("", result.output)


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def _run_args(
    documents: Documents, run_doc: Path, profile: Path | None = None, **extra: str
) -> list[str]:
    params = {
        "targets": json.dumps([FORTY]),
        "calibrations": json.dumps([documents.forty.as_posix()]),
        "confirm_turns": "true",
        **extra,
    }
    args = [
        "run",
        "objective-offsets",
        "-p",
        str(profile or documents.profile),
        "--run-doc",
        str(run_doc),
    ]
    for key, value in params.items():
        args += ["--param", f"{key}={value}"]
    return args


@dataclass(frozen=True)
class Measured:
    """A copy of ``run.json`` on which ``smc run objective-offsets`` has run once, and what it printed."""

    run_doc: Path
    printed: str


@pytest.fixture(scope="module")
def measured(
    documents: Documents, tmp_path_factory: pytest.TempPathFactory
) -> Measured:
    folder = tmp_path_factory.mktemp("cli-offsets-run")
    run_doc = folder / "run.json"
    shutil.copy(documents.run, run_doc)
    result = _invoke(*_run_args(documents, run_doc))
    assert result.exit_code == 0, _out(result)
    return Measured(run_doc, _out(result))


#: A signed number as the lines print it: ``-37.49``, ``+12.26``.
NUMBER = r"([+-]\d+\.\d\d)"


def _profile_copy(
    documents: Documents, tmp_path: Path, text: str | None = None
) -> Path:
    """A profile file of this test's own, so that the bytes before and after are its."""
    path = tmp_path / "stand.toml"
    if text is None:
        shutil.copy(documents.profile, path)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _finished_date(run_doc: Path) -> str:
    """The date the run finished, UTC, as the record has it: what ``measured`` starts with."""
    finished = RunDocument.load(run_doc).stages["objective-offsets"].finished_utc
    assert finished is not None
    return finished.date().isoformat()


def _from_run_doc(profile: Path, run_doc: Path, *extra: str) -> Result:
    return _invoke(
        "objective",
        "offset",
        "-p",
        str(profile),
        "--from-run-doc",
        str(run_doc),
        *extra,
    )


def _numbers(match: re.Match[str] | None) -> tuple[float, float]:
    assert match is not None
    return (float(match[1]), float(match[2]))


# ---------------------------------------------------------------------- the run


def test_smc_run_prints_the_result_with_the_offsets_the_spread_and_the_wells(
    measured: Measured,
) -> None:
    printed = measured.printed

    assert re.search(r"✓ objective-offsets done in [\d.]+ s", printed)
    result = ObjectiveOffsetsResult.model_validate(
        json.loads(printed[printed.index("{") :])
    )
    # The 40X's declared shift (-32.5, 9.75) minus the 10X's (5.0, -2.5): one pixel
    # of the 10X is 1.0 µm.
    (offset,) = result.offsets
    assert offset.objective == FORTY
    assert offset.xy_offset_um == pytest.approx((-37.5, 12.25), abs=1.0)
    (measure,) = result.measures
    assert measure.spread_um < 0.5
    assert [w.well for w in measure.wells] == ["A1", "B2"]
    assert result.plate_objective == TEN


def test_a_run_without_confirm_turns_is_refused_before_anything_turns(
    documents: Documents, tmp_path: Path
) -> None:
    run_doc = tmp_path / "run.json"
    shutil.copy(documents.run, run_doc)
    args = [
        "run",
        "objective-offsets",
        "-p",
        str(documents.profile),
        "--run-doc",
        str(run_doc),
        "--param",
        f"targets={json.dumps([FORTY])}",
        "--param",
        f"calibrations={json.dumps([documents.forty.as_posix()])}",
    ]

    result = _invoke(*args)

    assert result.exit_code != 0
    assert "confirm_turns=true" in _out(result)
    assert "objective-offsets" not in RunDocument.load(run_doc).stages or (
        RunDocument.load(run_doc).stages["objective-offsets"].status
        is not StageStatus.done
    )


# -------------------------------------------------------------------- the preview


def test_the_preview_prints_what_would_be_written_and_leaves_the_profile_as_it_was(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path)
    before = profile.read_bytes()

    result = _from_run_doc(profile, measured.run_doc, "--session", SESSION)

    assert result.exit_code == 0, _out(result)
    lines = _out(result).splitlines()
    text = (
        f"{_finished_date(measured.run_doc)}, {SESSION}, objective-offsets in run.json"
    )
    shift = re.fullmatch(
        r"'Nikon 40X Plan Fluor ELWD'  shift from 'Nikon 10X S Fluor' "
        rf"\({NUMBER}, {NUMBER}\) µm, spread (\d\.\d\d) µm over A1, B2 "
        r"\(bound 2\.00 µm\)",
        lines[0],
    )
    assert shift is not None
    assert _numbers(shift) == pytest.approx((-37.5, 12.25), abs=1.0)
    assert float(shift[3]) < 0.5
    written = re.fullmatch(
        rf"'Nikon 40X Plan Fluor ELWD'  xy \({NUMBER}, {NUMBER}\) µm  was unknown",
        lines[1],
    )
    assert _numbers(written) == pytest.approx((-37.5, 12.25), abs=1.0)
    assert lines[2:] == [
        "reference 'Nikon 10X S Fluor'  (new: the profile named none)",
        f'measured = "{text}"',
        f"nothing was written: add --write to write these into {profile.resolve()}",
    ]
    assert profile.read_bytes() == before


def test_write_writes_the_block_and_the_readout_gives_it_back_with_its_measured_line(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path)
    before = profile.read_bytes()
    text = (
        f"{_finished_date(measured.run_doc)}, {SESSION}, objective-offsets in run.json"
    )

    result = _from_run_doc(profile, measured.run_doc, "--session", SESSION, "--write")

    assert result.exit_code == 0, _out(result)
    # FM-36: the brackets of `[objectives]` are text, not markup.
    assert _out(result).splitlines()[-1] == (
        f"✓ wrote [objectives] into {profile.resolve()}; smc objective offset -p "
        f"{profile} reads them back against the turret"
    )
    after = profile.read_bytes()
    # Nothing outside the block was touched: the profile's bytes come first.
    assert after.startswith(before)
    assert after.count(BLOCK_BEGIN.encode("utf-8")) == 1
    assert after.count(BLOCK_END.encode("utf-8")) == 1
    loaded = Profile.load(profile)
    assert loaded.objectives.reference == TEN
    entry = loaded.objectives.offsets[FORTY]
    assert entry.xy_offset_um == pytest.approx((-37.5, 12.25), abs=1.0)
    # Rounded to 0.01 µm: a profile holds a number a person can read, not 15 digits.
    assert entry.xy_offset_um is not None
    assert all(round(value, 2) == value for value in entry.xy_offset_um)
    assert entry.z_offset_um is None
    assert entry.measured == text
    # The read-out of the profile, against the turret: the 10X is the reference, the
    # 40X has its offset and the line that says where it came from.
    readout = _invoke("objective", "offset", "-p", str(profile))
    assert readout.exit_code == 0, _out(readout)
    forty = re.search(
        rf"^'Nikon 40X Plan Fluor ELWD'  xy \({NUMBER}, {NUMBER}\) µm  z unknown  "
        rf"measured {re.escape(text)}$",
        _out(readout),
        re.MULTILINE,
    )
    assert _numbers(forty) == pytest.approx((-37.5, 12.25), abs=1.0)
    assert "'Nikon 10X S Fluor'  (current)  reference" in _out(readout)


def test_a_second_write_replaces_the_block_and_says_what_it_replaces(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path)
    first = "session 3 (#412)"
    assert (
        _from_run_doc(
            profile, measured.run_doc, "--session", first, "--write"
        ).exit_code
        == 0
    )
    first_bytes = profile.read_bytes()
    # The profile now names a reference: the run that is read is one made with it.
    run_doc = tmp_path / "run2.json"
    shutil.copy(documents.run, run_doc)
    assert _invoke(*_run_args(documents, run_doc, profile)).exit_code == 0

    result = _from_run_doc(profile, run_doc, "--session", "session 4 (#413)", "--write")

    assert result.exit_code == 0, _out(result)
    lines = _out(result).splitlines()
    # The old value is printed before it is replaced (FM-172), with where it came from.
    replaced = re.fullmatch(
        rf"'Nikon 40X Plan Fluor ELWD'  xy \({NUMBER}, {NUMBER}\) µm  was xy "
        rf"\({NUMBER}, {NUMBER}\) µm, measured {re.escape(_finished_date(measured.run_doc))}"
        rf", {re.escape(first)}, objective-offsets in run\.json",
        lines[1],
    )
    assert replaced is not None
    # The profile names its reference already: nothing is "new".
    assert "reference 'Nikon 10X S Fluor'" in lines
    after = profile.read_bytes()
    assert after.count(BLOCK_BEGIN.encode("utf-8")) == 1
    # Everything before the block is what it was, and the block is the new one.
    block_at = first_bytes.index(BLOCK_BEGIN.encode("utf-8"))
    assert after[:block_at] == first_bytes[:block_at]
    entry = Profile.load(profile).objectives.offsets[FORTY]
    assert "session 4 (#413)" in entry.measured
    assert "session 3" not in after.decode("utf-8")


def test_a_z_offset_the_profile_held_is_dropped_and_said(
    documents: Documents, tmp_path: Path
) -> None:
    # An entry with a Z, inside the block smc writes: someone edited the values there.
    # One in a hand-written table outside the block is the writer's refusal (below).
    held = "\n" + objectives_block(
        ObjectivesSection(
            reference=TEN,
            offsets={
                FORTY: ObjectiveOffsetEntry(
                    xy_offset_um=(1.0, 2.0),
                    z_offset_um=3.5,
                    measured="2026-10-01, session 2 (#300)",
                )
            },
        )
    )
    profile = _profile_copy(documents, tmp_path, _profile_text() + held)
    run_doc = tmp_path / "run.json"
    shutil.copy(documents.run, run_doc)
    assert _invoke(*_run_args(documents, run_doc, profile)).exit_code == 0

    result = _from_run_doc(profile, run_doc, "--write")

    assert result.exit_code == 0, _out(result)
    lines = _out(result).splitlines()
    assert re.fullmatch(
        rf"'Nikon 40X Plan Fluor ELWD'  xy \({NUMBER}, {NUMBER}\) µm  was xy "
        r"\(\+1\.00, \+2\.00\) µm, measured 2026-10-01, session 2 \(#300\)",
        lines[1],
    )
    assert lines[2] == (
        f"'{FORTY}': its z_offset_um +3.50 µm (measured 2026-10-01, session 2 (#300)) "
        "is dropped: objective-offsets measures XY only"
    )
    # The entry is replaced, not merged (FM-99): the Z is gone from the file too.
    written = profile.read_text(encoding="utf-8")
    assert "z_offset_um" not in written
    assert written.count(BLOCK_BEGIN) == 1
    assert Profile.load(profile).objectives.offsets[FORTY].z_offset_um is None


# ------------------------------------------------------------------- the refusals


def test_a_run_document_of_another_profile_is_refused_and_nothing_is_written(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    text = _profile_text().replace('name = "demo-sample"', 'name = "another-stand"')
    profile = _profile_copy(documents, tmp_path, text)
    before = profile.read_bytes()

    result = _from_run_doc(profile, measured.run_doc, "--write")

    assert result.exit_code == 1
    assert _out(result).splitlines() == [
        f"✗ in {measured.run_doc.as_posix()}: objective-offsets was run on "
        "demo-sample, this is another-stand; run objective-offsets again on this "
        "profile"
    ]
    assert profile.read_bytes() == before


def test_a_reference_changed_since_the_run_is_refused_and_nothing_is_written(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path, _profile_text(reference=TWENTY))
    before = profile.read_bytes()

    result = _from_run_doc(profile, measured.run_doc, "--write")

    assert result.exit_code == 1
    assert _out(result).splitlines() == [
        f"✗ the profile's offsets are now from {TWENTY}, and the run measured them "
        "from (none): run objective-offsets again"
    ]
    assert profile.read_bytes() == before


def test_a_run_document_without_the_result_is_refused(
    documents: Documents, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path)

    result = _from_run_doc(profile, documents.run, "--write")

    assert result.exit_code == 1
    assert _out(result).splitlines() == [
        f"✗ in {documents.run.as_posix()}: the run document has no "
        "objective-offsets; smc run --list shows which plugin provides it"
    ]


def test_a_run_document_that_is_not_there_is_refused(
    documents: Documents, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path)

    result = _from_run_doc(profile, tmp_path / "none.json")

    assert result.exit_code == 1
    assert _out(result).startswith("✗ ")
    assert "none.json" in _out(result)


def test_a_result_this_smc_does_not_read_is_refused(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    from smc.run import StageRecord, write_stage

    run_doc = tmp_path / "run.json"
    shutil.copy(measured.run_doc, run_doc)
    record = RunDocument.load(run_doc).stages["objective-offsets"]
    write_stage(
        run_doc,
        StageRecord.model_validate(
            {**record.model_dump(mode="json"), "result": {"x": 1}}
        ),
    )
    profile = _profile_copy(documents, tmp_path)

    result = _from_run_doc(profile, run_doc)

    assert result.exit_code == 1
    assert _out(result).splitlines() == [
        f"✗ the objective-offsets result in {run_doc.as_posix()} is not one this smc "
        "reads; run objective-offsets again"
    ]


def test_a_profile_that_is_not_a_file_is_refused(
    measured: Measured, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No `demo.toml` on the search path: `-p demo` is the profile built in code.
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        "objective", "offset", "-p", "demo", "--from-run-doc", str(measured.run_doc)
    )

    assert result.exit_code == 1
    assert _out(result).splitlines() == [
        "✗ demo is not loaded from a file: give the path of its .toml to -p"
    ]


def test_an_objectives_table_the_profile_holds_by_hand_is_refused_by_the_write_only(
    documents: Documents, measured: Measured, tmp_path: Path
) -> None:
    profile = _profile_copy(documents, tmp_path, _profile_text() + "\n[objectives]\n")
    before = profile.read_bytes()

    preview = _from_run_doc(profile, measured.run_doc)
    result = _from_run_doc(profile, measured.run_doc, "--write")

    # The preview changes nothing and says what it would write; the write refuses.
    assert preview.exit_code == 0, _out(preview)
    assert result.exit_code == 1
    assert _out(result).splitlines()[-1] == (
        f"✗ {profile.resolve()}: the profile has an [objectives] table outside the "
        "block smc writes: delete it (its values are printed above) or edit it by "
        "hand instead of writing"
    )
    assert profile.read_bytes() == before


# ------------------------------------------------------------------ usage errors


@pytest.mark.parametrize(
    "extra", [["--write"], ["--session", SESSION]], ids=["write", "session"]
)
def test_write_and_session_without_a_run_document_are_usage_errors(
    documents: Documents, extra: list[str]
) -> None:
    result = _invoke("objective", "offset", "-p", str(documents.profile), *extra)

    assert result.exit_code == 2
    assert "--from-run-doc" in _out(result)


@pytest.mark.parametrize("session", ["line one\nline two", "tab\there", "bell\x07"])
def test_a_session_of_more_than_one_line_is_a_usage_error(
    documents: Documents, measured: Measured, tmp_path: Path, session: str
) -> None:
    profile = _profile_copy(documents, tmp_path)
    before = profile.read_bytes()

    result = _from_run_doc(profile, measured.run_doc, "--session", session, "--write")

    assert result.exit_code == 2
    assert "--session" in _out(result)
    assert profile.read_bytes() == before


def test_the_help_names_both_uses(documents: Documents) -> None:
    result = _invoke("objective", "offset", "--help")

    assert result.exit_code == 0
    text = " ".join(_out(result).split())
    # Each flag is declared, with its own help: the whole flag followed by a blank
    # (`--write` is no prefix of another), not a word the docstring also holds (FM-72).
    assert re.search(r"--from-run-doc\s+<str>\s+Show what the objective-offsets", text)
    assert re.search(r"--session\s+<str>\s+With --from-run-doc: the session", text)
    assert re.search(r"--write\s+With --from-run-doc: write the offsets", text)
    # The command's own help names both uses.
    assert (
        "or, with --from-run-doc, show and write what objective-offsets measured"
        in text
    )
    # FM-36: rich reads brackets in a help text as markup and drops the word in them,
    # so the help says "objectives section" and a phrase of it reaches the output whole.
    assert (
        "would put into the profile's objectives section, and write it with --write"
        in text
    )
