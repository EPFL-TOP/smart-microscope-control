"""The preview and the write of ``objective-offsets`` as a library (m3 §20.11, #382).

The command (``smc objective offset --from-run-doc``) and the plate page share
``plan_offsets`` and ``write_offsets``, so the lines they show and what they write
are one thing. ``tests/test_cli_objective_offsets.py`` proves the command's text
unchanged; these tests prove the library on its own, with no stand: a result put into
a run document by ``write_stage`` (the seam, FM-124) and a copy of
``profiles/nikon-ti2.toml``. Every expected line is written out by hand.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hardware.profile import ObjectiveOffsetEntry, ObjectivesSection, Profile
from smc.plugins.objective_offsets.profile_offsets import (
    OffsetsWriteError,
    plan_offsets,
    write_offsets,
)
from smc.plugins.slots import (
    MeasuredOffset,
    ObjectiveOffsetsResult,
    OffsetWell,
    TargetMeasure,
    WellFit,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

NIKON = Path(__file__).resolve().parents[2] / "profiles" / "nikon-ti2.toml"
FOUR = "1-Plan Fluor 4x"
TEN = "2-Plan Fluor 10x"
FORTY = "3-Plan Apo LmbdD0.25 40x"
#: 22:30 at UTC-5 on the 14th is the 15th in UTC: the date in `measured` is the UTC one.
FINISHED = datetime(2026, 11, 14, 22, 30, tzinfo=timezone(timedelta(hours=-5)))
WELLS = ("A1", "H12", "A12")


def _measure(objective: str, shift: tuple[float, float]) -> TargetMeasure:
    wells = tuple(
        OffsetWell(
            well=well,
            origin_xy_um=(0.0, 0.0),
            delta_xy_um=shift,
            fit=WellFit(
                well=well,
                recorded_xy_um=(0.0, 0.0),
                center_xy_um=shift,
                diameter_um=6000.0,
                size_error=0.0,
                rms_um=0.1,
                n_points=100,
                walls=(),
            ),
        )
        for well in WELLS
    )
    return TargetMeasure(
        objective=objective,
        calibration="cal.json",
        calibration_measured="2026-11-10T10:00:00+00:00",
        pixel_size_um=2.5,
        z_after_turn_um=None,
        wells=wells,
        shift_xy_um=shift,
        spread_um=0.03,
        max_spread_um=2.0,
        frames=24,
        folder=None,
    )


def _result(
    *, reference_in_profile: bool = False, targets: tuple[str, ...] = (FOUR, FORTY)
) -> ObjectiveOffsetsResult:
    """A result of the plate registered under the 10X: the 4X at (-12.341, 6.789), the 40X at (-0.0, 12.25)."""
    shifts = {FOUR: (-12.341, 6.789), FORTY: (-0.0, 12.25)}
    return ObjectiveOffsetsResult(
        plate_objective=TEN,
        reference=TEN,
        reference_in_profile=reference_in_profile,
        measures=tuple(_measure(label, shifts[label]) for label in targets),
        offsets=tuple(
            MeasuredOffset(objective=label, xy_offset_um=shifts[label])
            for label in targets
        ),
        objective_in_place=targets[-1],
        elapsed_s=62.8,
    )


def _document(
    folder: Path,
    result: ObjectiveOffsetsResult,
    *,
    profile: str = "nikon-ti2",
    name: str = "run.json",
) -> Path:
    """A run document in a folder of its own, holding the result as ``objective-offsets`` left it."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_stage(
        path,
        StageRecord(
            plugin="objective-offsets",
            provides="objective-offsets",
            attempt="a1",
            status=StageStatus.done,
            started_utc=FINISHED - timedelta(seconds=63),
            finished_utc=FINISHED,
            profile=profile,
            dry_run=False,
            smc_version="0.0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )
    return path


def _profile(folder: Path, extra: str = "") -> Profile:
    """A copy of ``nikon-ti2.toml`` with ``extra`` TOML after it; it has a ``source``."""
    path = folder / "stand.toml"
    # The profile names its Micro-Manager configuration next to itself, and the loader
    # checks the file is there.
    (folder / "nikon-ti2.cfg").write_bytes(NIKON.with_suffix(".cfg").read_bytes())
    text = NIKON.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    path.write_bytes((text + extra).encode("utf-8"))
    return Profile.load(path)


#: What a profile that was written by hand or by an earlier write holds.
HELD = f"""
[objectives]
reference = "{TEN}"

[objectives.offsets."{FOUR}"]
xy_offset_um = [1.0, 2.0]
z_offset_um = 0.5
measured = "2026-11-01, session 2 (#400)"

[objectives.offsets."{FORTY}"]
xy_offset_um = [3.0, -4.0]
measured = "2026-11-02"
"""


def _plan(
    tmp_path: Path,
    result: ObjectiveOffsetsResult | None = None,
    *,
    profile: Profile | None = None,
) -> tuple[Profile, Path]:
    loaded = profile or _profile(tmp_path)
    run_doc = _document(tmp_path / "a folder" / "of the run", result or _result())
    return loaded, run_doc


def test_the_plan_says_each_target_each_label_the_reference_and_the_text(
    tmp_path: Path,
) -> None:
    profile, run_doc = _plan(tmp_path)

    plan = plan_offsets(
        profile,
        RunDocument.load(run_doc),
        run_doc,
        session="session 3 (#412)",
    )

    assert plan.lines == (
        f"'{FOUR}'  shift from '{TEN}' (-12.34, +6.79) µm, spread 0.03 µm over "
        "A1, H12, A12 (bound 2.00 µm)",
        f"'{FORTY}'  shift from '{TEN}' (+0.00, +12.25) µm, spread 0.03 µm over "
        "A1, H12, A12 (bound 2.00 µm)",
        f"'{FOUR}'  xy (-12.34, +6.79) µm  was unknown",
        f"'{FORTY}'  xy (+0.00, +12.25) µm  was unknown",
        f"reference '{TEN}'  (new: the profile named none)",
        'measured = "2026-11-15, session 3 (#412), objective-offsets in run.json"',
    )
    assert plan.new_reference is True
    assert plan.profile == "nikon-ti2"
    assert plan.source == profile.source
    assert plan.section == ObjectivesSection(
        reference=TEN,
        offsets={
            FOUR: ObjectiveOffsetEntry(
                xy_offset_um=(-12.34, 6.79),
                measured="2026-11-15, session 3 (#412), objective-offsets in run.json",
            ),
            FORTY: ObjectiveOffsetEntry(
                xy_offset_um=(0.0, 12.25),
                measured="2026-11-15, session 3 (#412), objective-offsets in run.json",
            ),
        },
    )


def test_a_shift_that_rounds_to_zero_prints_no_minus_sign(tmp_path: Path) -> None:
    # #514 point 5: `smc objective offset --from-run-doc` previewed `(-0.00, -0.00)`. `+ 0.0`
    # cures an exact `-0.0` only; -0.001 is a real negative number that `:+.2f` prints
    # with its sign. -0.006 rounds to -0.01 and keeps its sign: only a zero loses it.
    result = ObjectiveOffsetsResult(
        plate_objective=TEN,
        reference=TEN,
        reference_in_profile=False,
        measures=(
            _measure(FOUR, (-0.001, -0.004)),
            _measure(FORTY, (-0.006, 0.004)),
        ),
        offsets=(
            MeasuredOffset(objective=FOUR, xy_offset_um=(-0.001, -0.004)),
            MeasuredOffset(objective=FORTY, xy_offset_um=(-0.006, 0.004)),
        ),
        objective_in_place=FORTY,
        elapsed_s=62.8,
    )
    profile, run_doc = _plan(tmp_path, result)

    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)

    assert plan.lines[0].startswith(f"'{FOUR}'  shift from '{TEN}' (+0.00, +0.00) µm, ")
    assert plan.lines[1].startswith(
        f"'{FORTY}'  shift from '{TEN}' (-0.01, +0.00) µm, "
    )
    assert plan.lines[2] == f"'{FOUR}'  xy (+0.00, +0.00) µm  was unknown"
    assert plan.lines[3] == f"'{FORTY}'  xy (-0.01, +0.00) µm  was unknown"
    assert "-0.00" not in "".join(plan.lines)


def test_a_held_offset_that_rounds_to_zero_prints_no_minus_sign(tmp_path: Path) -> None:
    # The same rule for what the profile held: its xy and the Z that is dropped.
    held = HELD.replace("xy_offset_um = [1.0, 2.0]", "xy_offset_um = [-0.001, 2.0]")
    held = held.replace("z_offset_um = 0.5", "z_offset_um = -0.002")
    profile = _profile(tmp_path, held)
    result = _result(reference_in_profile=True, targets=(FOUR,))
    _, run_doc = _plan(tmp_path, result, profile=profile)

    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)

    assert "was xy (+0.00, +2.00) µm" in plan.lines[1]
    assert plan.lines[2].startswith(f"'{FOUR}': its z_offset_um +0.00 µm (measured ")
    assert "-0.00" not in "".join(plan.lines)


def test_the_plan_names_the_run_document_and_never_its_folder(tmp_path: Path) -> None:
    # FM-50: a folder is the operator's machine; `measured` goes into a file in git.
    profile, run_doc = _plan(tmp_path)

    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)

    assert plan.lines[-1] == 'measured = "2026-11-15, objective-offsets in run.json"'
    assert "of the run" not in "".join(plan.lines)
    assert all(
        "of the run" not in entry.measured for entry in plan.section.offsets.values()
    )


def test_the_plan_replaces_a_label_says_what_it_held_and_drops_a_z(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, HELD)
    result = _result(reference_in_profile=True, targets=(FOUR,))
    _, run_doc = _plan(tmp_path, result, profile=profile)

    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)

    assert plan.lines == (
        f"'{FOUR}'  shift from '{TEN}' (-12.34, +6.79) µm, spread 0.03 µm over "
        "A1, H12, A12 (bound 2.00 µm)",
        f"'{FOUR}'  xy (-12.34, +6.79) µm  was xy (+1.00, +2.00) µm, measured "
        "2026-11-01, session 2 (#400)",
        f"'{FOUR}': its z_offset_um +0.50 µm (measured 2026-11-01, session 2 "
        "(#400)) is dropped: objective-offsets measures XY only",
        f"reference '{TEN}'",
        'measured = "2026-11-15, objective-offsets in run.json"',
    )
    assert plan.new_reference is False
    assert plan.section.offsets[FOUR] == ObjectiveOffsetEntry(
        xy_offset_um=(-12.34, 6.79),
        measured="2026-11-15, objective-offsets in run.json",
    )
    # A label the run did not measure keeps what the profile holds (replaced, not wiped).
    assert plan.section.offsets[FORTY] == ObjectiveOffsetEntry(
        xy_offset_um=(3.0, -4.0), measured="2026-11-02"
    )


@pytest.mark.parametrize(
    ("held", "in_profile", "now", "then"),
    [
        (f'\n[objectives]\nreference = "{FOUR}"\n', False, FOUR, "(none)"),
        ("", True, "(none)", TEN),
    ],
    ids=["profile-gained-a-reference", "profile-lost-its-reference"],
)
def test_the_plan_refuses_a_reference_changed_since_the_run(
    tmp_path: Path, held: str, in_profile: bool, now: str, then: str
) -> None:
    profile = _profile(tmp_path, held)
    _, run_doc = _plan(
        tmp_path, _result(reference_in_profile=in_profile), profile=profile
    )

    with pytest.raises(OffsetsWriteError) as raised:
        plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)

    assert str(raised.value) == (
        f"the profile's offsets are now from {now}, and the run measured them from "
        f"{then}: run objective-offsets again"
    )


def test_the_plan_refuses_a_result_of_another_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    run_doc = _document(tmp_path / "x", _result(), profile="another-stand")

    with pytest.raises(OffsetsWriteError) as raised:
        plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)

    assert str(raised.value) == (
        f"in {run_doc.as_posix()}: objective-offsets was run on another-stand, this "
        "is nikon-ti2; run objective-offsets again on this profile"
    )


def test_the_plan_refuses_a_document_with_no_result_and_a_profile_with_no_file(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    empty = tmp_path / "empty.json"
    write_stage(
        empty,
        StageRecord(
            plugin="objective-offsets",
            provides="objective-offsets",
            attempt="a1",
            status=StageStatus.failed,
            started_utc=FINISHED,
            finished_utc=FINISHED,
            profile="nikon-ti2",
            dry_run=False,
            smc_version="0.0",
            params={},
            error="PluginRunError: no",
        ),
    )

    with pytest.raises(OffsetsWriteError) as failed:
        plan_offsets(profile, RunDocument.load(empty), empty, session=None)
    built_into_smc = Profile.load("demo")
    built_into_smc.source = None
    with pytest.raises(OffsetsWriteError) as built_in:
        plan_offsets(built_into_smc, RunDocument.load(empty), empty, session=None)

    assert str(failed.value) == (
        f"in {empty.as_posix()}: the last run of objective-offsets ended failed; "
        "run it again"
    )
    assert str(built_in.value) == (
        "demo is not loaded from a file: give the path of its .toml to -p"
    )


@pytest.mark.parametrize("session", ["line one\nline two", "tab\there", "bell\x07"])
def test_the_plan_refuses_a_session_with_a_line_break(
    tmp_path: Path, session: str
) -> None:
    profile, run_doc = _plan(tmp_path)

    with pytest.raises(OffsetsWriteError) as raised:
        plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=session)

    assert str(raised.value) == (
        "the session is one line of text: no tab, no line break"
    )


def _outside_the_block(text: str) -> str:
    """The profile without the lines from the block's opening marker to its closing one."""
    from smc.hardware.profile_write import BLOCK_BEGIN, BLOCK_END

    lines = text.splitlines(keepends=True)
    first = next(
        i for i, line in enumerate(lines) if line.rstrip("\r\n") == BLOCK_BEGIN
    )
    last = next(i for i, line in enumerate(lines) if line.rstrip("\r\n") == BLOCK_END)
    return "".join(lines[:first] + lines[last + 1 :])


def test_the_write_replaces_the_block_and_keeps_every_other_byte(
    tmp_path: Path,
) -> None:
    profile, run_doc = _plan(tmp_path)
    original = profile.source.read_bytes().decode("utf-8")  # type: ignore[union-attr]
    first = plan_offsets(
        profile, RunDocument.load(run_doc), run_doc, session="session 3 (#412)"
    )

    write_offsets(first)

    path = first.source
    written = path.read_bytes().decode("utf-8")
    # A first write appends the block after one empty line and touches nothing else.
    assert written.startswith(original)
    assert Profile.load(path).objectives == first.section
    # The operator edits around the block; the next write replaces the block alone.
    edited = f"# kept comment\n{written}# trailing comment\n"
    path.write_bytes(edited.encode("utf-8"))
    reloaded = Profile.load(path)
    second_doc = _document(
        tmp_path / "second",
        _result(reference_in_profile=True, targets=(FOUR,)),
        name="run2.json",
    )
    second = plan_offsets(
        reloaded, RunDocument.load(second_doc), second_doc, session=None
    )

    write_offsets(second)

    final = path.read_bytes().decode("utf-8")
    assert _outside_the_block(final) == _outside_the_block(edited)
    assert final.startswith("# kept comment\n")
    assert final.endswith("# trailing comment\n")
    assert Profile.load(path).objectives == second.section
    assert Profile.load(path).objectives.offsets[FORTY].measured == (
        "2026-11-15, session 3 (#412), objective-offsets in run.json"
    )
    assert Profile.load(path).objectives.offsets[FOUR].measured == (
        "2026-11-15, objective-offsets in run2.json"
    )


def test_a_hand_written_objectives_table_is_refused_by_the_write_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, HELD)
    _, run_doc = _plan(tmp_path, _result(reference_in_profile=True), profile=profile)
    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)
    before = plan.source.read_bytes()

    with pytest.raises(OffsetsWriteError) as raised:
        write_offsets(plan)

    assert str(raised.value).startswith(f"{plan.source}: ")
    assert "has an [objectives] table outside the block smc writes" in str(raised.value)
    assert plan.source.read_bytes() == before


@pytest.mark.parametrize("kept", [True, False], ids=["kept", "not-kept"])
def test_a_write_that_cannot_replace_the_file_says_where_the_new_text_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kept: bool
) -> None:
    # Windows refuses the rename while an editor or a sync client holds the file
    # (FM-24, FM-38): the profile is as it was and the text says whether the whole new
    # text is kept.
    from smc.run.files import AtomicWriteError

    profile, run_doc = _plan(tmp_path)
    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)
    before = plan.source.read_bytes()
    temporary = tmp_path / ".stand.toml.0123abcd.smc-tmp"

    def refuse(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        raise AtomicWriteError(
            PermissionError(13, "Access is denied"), temporary if kept else None
        )

    monkeypatch.setattr("smc.run.files.write_atomically", refuse)

    with pytest.raises(OffsetsWriteError) as raised:
        write_offsets(plan)

    expected = f"{plan.source} was not changed: Access is denied"
    if kept:
        expected += (
            f"; the whole new text is kept in {temporary}: close what holds the "
            "profile and write again, then delete that file"
        )
    assert str(raised.value) == expected
    assert plan.source.read_bytes() == before


def test_a_write_without_tomli_w_says_what_is_missing_and_the_plan_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An install updated by `git pull` and not reinstalled lacks `tomli-w`: the preview
    # must not need it, and the write says what to do (FM-112).
    monkeypatch.setitem(sys.modules, "tomli_w", None)
    monkeypatch.delitem(sys.modules, "smc.hardware.profile_write", raising=False)
    profile, run_doc = _plan(tmp_path)
    plan = plan_offsets(profile, RunDocument.load(run_doc), run_doc, session=None)
    before = plan.source.read_bytes()

    with pytest.raises(OffsetsWriteError) as raised:
        write_offsets(plan)

    assert str(raised.value) == (
        "writing the profile needs the tomli-w package, which this install lacks: "
        "run pip install -e . in the smc folder, then write again"
    )
    assert plan.source.read_bytes() == before
