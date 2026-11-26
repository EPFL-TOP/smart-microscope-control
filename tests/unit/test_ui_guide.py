"""The next step's pure pieces: which slot is next, and the command that makes its call (m5 §8.3, #293).

No Panel and no stand. The command is read back with ``shlex.split``, the way a
shell reads it, never compared with the string the code builds (FM-45).
"""

from __future__ import annotations

import ast
import dataclasses
import html
import importlib
import json
import logging
import math
import re
import shlex
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path, PureWindowsPath
from typing import Any, ClassVar, Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

import smc
from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionInProgressError,
    SafetyRefusedError,
)
from smc.hardware.safety import Executor, Motion, Safety
from smc.plugins.api import Progress
from smc.plugins.geometry import Matrix2
from smc.plugins.presets import Preset, PresetCheck, PresetFile
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    MeasuredOffset,
    ObjectiveOffsetsResult,
    OffsetWell,
    PlanSelection,
    PlateResult,
    ScanPlanResult,
    TargetMeasure,
    WallFrame,
    WallImages,
    WellFit,
    WellRef,
)
from smc.ui.controller import (
    _CHECKLIST,
    ChainEnd,
    ChainStatus,
    CheckItem,
    LoadedPreset,
    MapPose,
    PlanLayer,
    ResultCard,
    RunClock,
    ScanEnd,
    ScanSummary,
    WellView,
)
from smc.ui.guide import (
    AGAIN_NOTE,
    ASSUMED_LINE_CHARS,
    AUTOFOCUS_WORDS,
    CALIBRATE_DRY_RUN,
    CALIBRATE_LEAD,
    CALIBRATE_LIGHT_OFF,
    CALIBRATE_NAMES_NONE,
    CALIBRATE_NO_OFFER,
    CALIBRATE_NOT_KNOWN,
    CALIBRATE_REUSE_ROUTED,
    CALIBRATIONS_FOLDER,
    CAMERA_NOT_CALIBRATED_NOTE,
    CAPTURE_FAR,
    CAPTURE_SAME_PLACE,
    CAPTURE_WELLS,
    CELL_FOCUS_WORDS,
    CENTRES_CUT_NOTE,
    CENTRES_UNREACHABLE_NOTE,
    COVERAGE_WORDS,
    DISCONNECT_RELEASES,
    DRY_RUN_NO_FRAME,
    END_LIVE_LABEL,
    FILES_HEADING,
    FOCUS_MAP_NONE,
    FRAME_DARK,
    FRAME_SATURATED,
    GOTO_NOT_DURING_LIVE,
    GOTO_UNPLACED,
    LAY_OUT_FIRST,
    MAP_FIELDS_UP_TO,
    MAX_POSITIONS,
    MEASURE_THE_TIME,
    NO_CALIBRATION_FOR_SIDES,
    NO_FILES_YET,
    NOT_CONNECTED,
    NOTHING_LEFT,
    OFFSETS_MEASURE_LABEL,
    OFFSETS_NO_DOCUMENT,
    OFFSETS_NOTHING_ARMED,
    OFFSETS_TOO_MANY,
    ORIENTATION_IN_STAGE_TERMS,
    OTHER_WAYS_TO_REGISTER,
    PLAN_CALIBRATION_NONE_MADE,
    PLATE_NOT_CHOSEN,
    REGISTER_ACROSS_A_TURN,
    REGISTER_FIRST,
    REGISTER_OBJECTIVES_DIFFER,
    RELOCK_NOTE,
    RESULT_CHOICES,
    RESULT_NAMES,
    RESULT_OPTIONAL_READS,
    RESULT_READS,
    RESULT_SHORT,
    RESULT_TOOLS,
    RUN_ALL_LABEL,
    SAMPLE_REGISTERS,
    SCAN_CONTINUE,
    SCAN_FIRST,
    SCAN_HALTED,
    SCAN_MAP_RESUME_NOTE,
    SCAN_NEEDS_A_LOCK,
    SCAN_REPLACES,
    SCAN_START_AGAIN,
    SCAN_START_AGAIN_CONFIRM,
    SCAN_WITHOUT_A_LOCK,
    SCREEN_LEADS,
    SCREEN_TITLES,
    SELECT_WELLS,
    STAND_HALTED,
    STEP_OF_SLOT,
    STEP_TITLES,
    STEPS,
    STOP_SEGMENT_AXES,
    TIME_NOT_KNOWN,
    TOOL_TITLES,
    WALLS_DRY_RUN,
    WALLS_MAX_CENTRES,
    WALLS_MIN_CENTRES,
    WALLS_NEED_CALIBRATION,
    CalibrationNote,
    GotoAnswer,
    KeptMap,
    NextStep,
    ResultRow,
    StepName,
    StepState,
    _join_names,
    a1_corner,
    again_note,
    busy_note,
    calibrate_changed,
    calibrate_command_folder_note,
    calibrate_is_the_plates,
    calibrate_label,
    calibrate_measured_other,
    calibrate_position_unknown,
    calibrate_replaces,
    calibrate_stand_routed,
    calibrate_wait,
    calibration_at_its_position,
    calibration_denied,
    calibration_failed,
    calibration_file,
    calibration_goes_to_a_document,
    calibration_held_line,
    calibration_how_text,
    calibration_light,
    calibration_missing_line,
    calibration_note_line,
    calibration_paths,
    calibration_read_line,
    calibrations_held_heading,
    capture_centre_label,
    capture_note,
    capture_wall_label,
    centre_rows,
    centres_reach_note,
    centres_text,
    chain_line,
    checkpoint_note,
    colour_bar_title,
    command_line,
    continue_label,
    drive_image_note,
    duration_text,
    error_words,
    field_source_line,
    files_html,
    files_lines,
    focus_map_file,
    focus_map_lines,
    forget_label,
    frame_hint,
    goto_row_text,
    image_sides_note,
    known_text,
    lay_out_wait,
    local_time,
    map_frame,
    map_title,
    needs_calibration,
    next_slot,
    objective_in_place_line,
    objective_line,
    offsets_card_lines,
    offsets_changed,
    offsets_confirm_text,
    offsets_in_two_documents,
    offsets_is_the_plates,
    offsets_lead,
    offsets_lead_unread,
    offsets_not_on_turret,
    offsets_others,
    offsets_plan_heading,
    offsets_plate_refusal,
    offsets_run_line,
    offsets_written_line,
    opening_step,
    orientation_label,
    page_words,
    plain_default,
    plan_calibration_frame,
    plan_calibration_missing,
    plan_calibration_other_objective,
    plan_calibration_used,
    plan_selection_note,
    plan_title,
    plate_from_line,
    plate_rows_note,
    preset_differences,
    preset_note_text,
    register_objective_refusal,
    register_replace_note,
    registered_line,
    relock_note,
    replace_note,
    result_row,
    results_left,
    run_all_wait,
    scan_actions_wait,
    scan_again_confirm,
    scan_end_lines,
    scan_lead,
    scan_lock_note,
    scan_plan_card_lines,
    scan_progress_lines,
    scan_title,
    scan_wait,
    scanned_line,
    side_words,
    stage_label,
    stand_calibration_failed,
    step_button_label,
    step_states,
    stop_refused_text,
    stop_segments_clause,
    suggested_centres,
    time_line,
    tool_title,
    turret_position,
    wall_touches_note,
    walls_card_lines,
    walls_centres_problem,
    walls_square_wells,
    wells_from_line,
    wells_known_lines,
    wells_text,
)


def _item(
    slot: str,
    state: str = "not run",
    *,
    optional: bool = False,
    reason: str | None = None,
) -> CheckItem:
    return CheckItem(slot, optional, state, reason)  # type: ignore[arg-type]


def test_next_slot_skips_optional_slots_and_done_ones() -> None:
    checklist = (
        _item("camera-calibration", optional=True),
        _item("plate", "ok"),
        _item("timings", optional=True),
        _item("scan-plan"),
        _item("scan"),
    )
    found = next_slot(checklist)
    assert found is not None
    assert found.slot == "scan-plan"


def test_next_slot_stops_at_an_unusable_one_with_its_reason() -> None:
    reason = "the last run of plate-registration ended failed; run it again"
    checklist = (
        _item("plate", "unusable", reason=reason),
        _item("scan-plan"),
    )
    found = next_slot(checklist)
    assert found is not None
    assert (found.slot, found.state, found.reason) == ("plate", "unusable", reason)


def test_next_slot_is_none_when_every_required_slot_is_ok() -> None:
    checklist = (
        _item("camera-calibration", optional=True),
        _item("plate", "ok"),
        _item("timings", "unusable", optional=True, reason="stale"),
        _item("scan", "ok"),
        # A third party's slot is optional: the workflow does not wait for it.
        _item("their-slot", optional=True),
    )
    assert next_slot(checklist) is None
    assert next_slot(()) is None


def test_next_slot_takes_the_first_of_several_missing_in_the_checklists_order() -> None:
    checklist = (_item("plate"), _item("scan-plan"), _item("scan"))
    found = next_slot(checklist)
    assert found is not None
    assert found.slot == "plate"


def test_the_two_waiting_texts_are_the_designs() -> None:
    assert SELECT_WELLS == (
        "select the wells to scan: click them on the map, or type a range in Wells "
        "(A1:D6, C*, all)"
    )
    assert CAPTURE_WELLS == (
        "centre a well under the objective (Live and the Drive pad), click it on the "
        "map and press Capture centre; three wells far apart fit the plate and check "
        "each other"
    )


def test_capture_wells_says_three_wells_far_apart() -> None:
    # Three centres are what fits a plate and checks itself (m5 §15 D7): two fit it and
    # check only their spacing, so the text must not keep the old "a second well".
    assert "three wells far apart" in CAPTURE_WELLS
    assert "second well" not in CAPTURE_WELLS
    assert "check each other" in CAPTURE_WELLS


def test_the_dry_run_note_says_what_to_do() -> None:
    # Whole (FM-45): the page cannot take the way out, so the text must name both.
    assert DRY_RUN_NO_FRAME == (
        "this page is a dry run: scan walks the fields and takes no frame, so cells and "
        "the tools after it have nothing to read. Serve the page without --dry-run on a "
        "stand whose stage is simulated (demo-sample), or run the chain with smc run "
        "from plate-registration, without --dry-run: a run that is not a dry run "
        "refuses what a dry run made"
    )


def test_a_next_step_is_frozen_and_holds_what_it_was_given() -> None:
    step = NextStep(
        plugin="scan",
        params=("a=1",),
        why=("the run document has no scan yet",),
        command="smc run scan",
        blocked=None,
    )
    assert (step.plugin, step.params, step.blocked) == ("scan", ("a=1",), None)
    # The page keeps the step it drew and runs it at the click: it must not change.
    with pytest.raises(AttributeError):
        step.plugin = "other"  # type: ignore[misc]


_REFS = json.dumps(
    [
        {"well": "A1", "x_um": -49500.0, "y_um": 31500.0},
        {"well": "H1", "x_um": -49500.0, "y_um": -31500.0},
    ]
)


def test_command_line_round_trips_json_through_shlex() -> None:
    pairs = ["plate=96-well", f"refs={_REFS}", "walls=[]", 'note=it\'s "quoted"']
    line = command_line(
        "plate-registration",
        pairs,
        profile="demo-sample",
        run_doc=Path("/runs/exp one/run.json"),
        dry_run=False,
    )
    words = shlex.split(line)
    assert words[:3] == ["smc", "run", "plate-registration"]
    taken = [words[i + 1] for i, word in enumerate(words) if word == "--param"]
    assert taken == pairs
    # The JSON is one word and parses back to what was dumped.
    assert json.loads(taken[1].removeprefix("refs=")) == json.loads(_REFS)


def test_command_line_names_the_profile_the_dry_run_and_the_run_doc() -> None:
    run_doc = Path("/runs/exp one/run.json")
    wet = shlex.split(
        command_line(
            "scan",
            ["a=1"],
            profile="/p/demo sample.toml",
            run_doc=run_doc,
            dry_run=False,
        )
    )
    assert wet == [
        "smc",
        "run",
        "scan",
        "-p",
        "/p/demo sample.toml",
        "--run-doc",
        str(run_doc),
        "--param",
        "a=1",
    ]
    dry = shlex.split(
        command_line("scan", [], profile="demo", run_doc=run_doc, dry_run=True)
    )
    assert dry == [
        "smc",
        "run",
        "scan",
        "-p",
        "demo",
        "--run-doc",
        str(run_doc),
        "--dry-run",
    ]


def test_command_line_puts_the_preset_after_dry_run_and_before_the_pairs() -> None:
    run_doc = Path("/runs/run.json")

    said = shlex.split(
        command_line(
            "scan-plan",
            ["wells=B2,B11", "coverage=0.4"],
            profile="demo-sample",
            run_doc=run_doc,
            dry_run=True,
            preset="/p/my presets/weekly.toml",
        )
    )

    assert said == [
        "smc",
        "run",
        "scan-plan",
        "-p",
        "demo-sample",
        "--run-doc",
        str(run_doc),
        "--dry-run",
        "--preset",
        "/p/my presets/weekly.toml",
        "--param",
        "wells=B2,B11",
        "--param",
        "coverage=0.4",
    ]
    without = shlex.split(
        command_line("scan-plan", [], profile="demo", run_doc=run_doc, dry_run=False)
    )
    assert "--preset" not in without


def test_preset_differences_name_each_changed_field_with_default_for_empty() -> None:
    expected = {
        "scan-plan": {"coverage": "0.4", "objective": "", "overlap": "0.1"},
        "cells": {"detector": "blobs"},
        "density": {"labels": "x"},
    }
    actual = {
        "scan-plan": {"coverage": " 0.5 ", "objective": "10x", "overlap": ""},
        "cells": {"detector": "blobs "},
        # `density` is not on the page any more: its fields read as empty.
    }

    assert preset_differences(expected, actual) == (
        "scan-plan coverage: 0.5 (the preset: 0.4)",
        "scan-plan objective: 10x (the preset: (default))",
        "scan-plan overlap: (default) (the preset: 0.1)",
        "density labels: (default) (the preset: x)",
    )
    assert preset_differences(expected, expected) == ()
    assert preset_differences({}, actual) == ()


def test_preset_differences_compare_stripped_text_on_both_sides() -> None:
    # A hand-edited `coverage= 0.4` puts " 0.4" in the field; what a run sends is the
    # stripped text, so the page right after the load differs in nothing.
    expected = {"scan-plan": {"coverage": " 0.4", "overlap": "0.1"}}

    assert (
        preset_differences(
            expected, {"scan-plan": {"coverage": " 0.4 ", "overlap": "0.1"}}
        )
        == ()
    )
    assert (
        preset_differences(
            expected, {"scan-plan": {"coverage": "0.4", "overlap": "0.1"}}
        )
        == ()
    )
    assert preset_differences(
        expected, {"scan-plan": {"coverage": "0.5", "overlap": "0.1"}}
    ) == ("scan-plan coverage: 0.5 (the preset: 0.4)",)


def test_preset_differences_show_a_long_value_on_one_cut_line_and_compare_it_whole() -> (
    None
):
    long = "[" + ",".join(["1"] * 100) + "]"
    expected = {"route": {"labels": long}}

    same = preset_differences(expected, {"route": {"labels": long}})
    other = preset_differences(expected, {"route": {"labels": long + "\n,2"}})

    assert same == ()
    assert len(other) == 1
    line = other[0]
    assert "\n" not in line
    assert line.startswith("route labels: [1,1,1")
    assert line.count("…") == 2  # both values are cut
    assert len(line) < 220


#: The note writes the path as the OS spells it (`\p\weekly.toml` on Windows, FM-24), so
#: the expected text is made from the path and not typed.
PRESET_PATH = Path("/p/weekly.toml")
LOADED_LINE = f"loaded: weekly  ({PRESET_PATH})"


def _loaded(mismatches: Sequence[str] = ()) -> LoadedPreset:
    file = PresetFile("weekly", PRESET_PATH, Preset())
    return LoadedPreset(
        file=file,
        check=PresetCheck("weekly", None, None, {}, ()),
        fields={},
        selection=None,
        mismatches=tuple(mismatches),
    )


def test_preset_note_text_says_none_loaded_and_when_the_page_holds_the_values() -> None:
    assert preset_note_text(None, ()) == "no preset loaded"
    assert preset_note_text(_loaded(), ()) == (
        f"{LOADED_LINE}\nthe page holds the preset's values"
    )


def test_preset_note_text_lists_the_mismatches_then_what_changed() -> None:
    note = preset_note_text(
        _loaded(["preset weekly was saved on nikon-ti2; this stand is demo"]),
        ["scan-plan coverage: 0.5 (the preset: 0.4)"],
    )

    assert note.split("\n") == [
        LOADED_LINE,
        "  preset weekly was saved on nikon-ti2; this stand is demo",
        "changed since loading:",
        "  scan-plan coverage: 0.5 (the preset: 0.4)",
    ]


def test_preset_note_text_cuts_mismatches_and_differences_at_eight() -> None:
    mismatches = [f"mismatch {n}" for n in range(11)]
    differences = [f"difference {n}" for n in range(9)]

    lines = preset_note_text(_loaded(mismatches), differences).split("\n")

    assert lines == [
        LOADED_LINE,
        *[f"  mismatch {n}" for n in range(8)],
        "  … 3 more",
        "changed since loading:",
        *[f"  difference {n}" for n in range(8)],
        "  … 1 more",
    ]
    # Eight is not cut: nothing is hidden behind a "… 0 more".
    exact = preset_note_text(_loaded(mismatches[:8]), differences[:8])
    assert "more" not in exact


@given(
    st.lists(
        st.tuples(
            st.from_regex(r"[a-z_]{1,8}", fullmatch=True),
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs",), blacklist_characters="\x00"
                ),
                max_size=30,
            ),
        ),
        max_size=5,
    )
)
def test_command_line_gives_back_any_pairs_through_shlex(
    pairs: list[tuple[str, str]],
) -> None:
    items = [f"{key}={value}" for key, value in pairs]
    line = command_line(
        "scan", items, profile="demo", run_doc=Path("/r/run.json"), dry_run=False
    )
    words = shlex.split(line)
    taken = [words[i + 1] for i, word in enumerate(words) if word == "--param"]
    assert taken == items


# --- the captures' labels, the busy note, the replace note (m5 §11, #332) ----------


def test_capture_labels_name_the_well_and_the_side() -> None:
    assert capture_centre_label("B3") == "Capture centre of B3"
    assert forget_label("B3") == "Forget B3"
    # With no well the labels are the plain three: there is nothing to name.
    assert capture_centre_label(None) == "Capture centre"
    assert capture_wall_label(None, "left") == "Capture wall"
    assert forget_label(None) == "Forget"


def test_capture_wall_label_names_the_well_and_the_image_side() -> None:
    # The side is the select's value, a side on the image (m5 §13.2): the button
    # says so, so that nobody reads it as a stage direction.
    assert capture_wall_label("B3", "left") == "Capture B3 wall at image left"
    assert capture_wall_label("B3", "bottom") == "Capture B3 wall at image bottom"
    assert capture_wall_label("H12", "top") == "Capture H12 wall at image top"
    assert capture_wall_label(None, "right") == "Capture wall"


# --- the image's sides and what the arrows do (m5 §13.2, #327) ----------------------

_TI2E: Matrix2 = ((-1.1, 0.0), (0.0, 1.1))  # the Ti2-E at 10x, a camera at 180 degrees
_UPRIGHT: Matrix2 = ((1.1, 0.0), (0.0, -1.1))  # a camera at 0 degrees, not mirrored
_THIRTY_DEGREES: Matrix2 = ((0.9526, 0.55), (0.55, -0.9526))


def test_the_no_calibration_text_says_to_run_camera_calibration() -> None:
    assert NO_CALIBRATION_FOR_SIDES == (
        "walls are named on the image, and no camera calibration says which stage "
        "side that is: run camera-calibration (Set up), then try again"
    )


def test_the_image_sides_note_says_what_each_side_is_on_the_stage() -> None:
    assert image_sides_note(_TI2E) == (
        "On the image: left is the stage's +X, right -X, top -Y, bottom +Y "
        "(from the camera calibration)."
    )
    # The table is the matrix's, not one camera's: an upright one reads the other way.
    assert image_sides_note(_UPRIGHT) == (
        "On the image: left is the stage's -X, right +X, top +Y, bottom -Y "
        "(from the camera calibration)."
    )


def test_the_image_sides_note_without_a_calibration_says_to_run_it() -> None:
    assert image_sides_note(None) == (
        "Walls are named on the image. No camera calibration says which stage side "
        "each one is: run camera-calibration (Set up) to capture a wall or drive to "
        "one."
    )


def test_the_drive_note_says_what_the_arrows_do_to_the_image() -> None:
    assert drive_image_note(_TI2E) == (
        "On the image, ↑ moves it up, → moves it right; ↓ and ← the opposite."
    )
    assert drive_image_note(_UPRIGHT) == (
        "On the image, ↑ moves it down, → moves it left; ↓ and ← the opposite."
    )
    # The Stand screen's button, not a tab the guided page no longer has (#475, finding 12).
    assert drive_image_note(None) == CAMERA_NOT_CALIBRATED_NOTE
    assert "Calibrate the camera" in CAMERA_NOT_CALIBRATED_NOTE
    assert "Set up" not in CAMERA_NOT_CALIBRATED_NOTE


def test_the_notes_of_a_camera_off_a_quarter_turn_are_frames_text() -> None:
    # `frames`' own words, taken from its refusal: a camera 30 degrees off names no
    # side, and the page says why instead of guessing (FM-98).
    for note in (image_sides_note(_THIRTY_DEGREES), drive_image_note(_THIRTY_DEGREES)):
        assert note.startswith("the camera is turned 30.0° away from the stage's axes")
        assert "an image side is not a stage side" in note
    singular: Matrix2 = ((1.0, 2.0), (2.0, 4.0))
    assert "cannot be inverted" in image_sides_note(singular)
    assert "cannot be inverted" in drive_image_note(singular)


def test_busy_note_for_live_names_the_button_and_for_another_activity_says_wait_or_stop() -> (
    None
):
    live = busy_note("live", live=True)
    assert live == "Live is running: press End live, then run the tool."
    # The button's own label, from the one constant the page's button draws.
    assert END_LIVE_LABEL in live
    other = busy_note("scan", live=False)
    assert other == "scan is running: wait for it to end, or press STOP."


def _plate_result(method: str, n_refs: int) -> PlateResult:
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method=method,  # type: ignore[arg-type]
        residual_um=None,
        scale_error=None,
        n_refs=n_refs,
    )


def test_replace_note_names_the_time_in_utc_the_method_and_the_wells() -> None:
    started = datetime(2026, 11, 3, 14, 20, 41, tzinfo=timezone.utc)
    # The expert view's note sits beside a card that says `done <time> UTC`: one zone, said
    # (m5 §17.12 D20; #514 point 4). A literal is right here, since it depends on no zone.
    when = "2026-11-03 14:20 UTC"
    tail = (
        "A failed run leaves no plate; the captures stay, so Forget the wrong one "
        "and Run again to get a plate back."
    )

    one = replace_note(_plate_result("seed", 1), started, keeps_captures=True)
    assert one == f"Run replaces the plate registered {when} (seed, 1 well). {tail}"
    two = replace_note(_plate_result("references", 2), started, keeps_captures=True)
    assert two == (
        f"Run replaces the plate registered {when} (references, 2 wells). {tail}"
    )
    # A method the registration names is the result's own, never a paraphrase.
    assert "(wall-images, 3 wells)" in replace_note(
        _plate_result("wall-images", 3), started, keeps_captures=True
    )


def test_replace_note_converts_a_start_that_is_not_in_utc_to_utc() -> None:
    # A start with an offset that is not zero: 14:20 at +05:00 is 09:20 UTC. A note that
    # skips the conversion says 14:20 UTC, which is wrong by the offset on every PC, so
    # the literal fails wherever it runs.
    started = datetime(2026, 11, 3, 14, 20, tzinfo=timezone(timedelta(hours=5)))
    note = replace_note(_plate_result("seed", 1), started, keeps_captures=True)
    assert note.startswith(
        "Run replaces the plate registered 2026-11-03 09:20 UTC (seed, 1 well). "
    )
    # A start with no zone is UTC, as the run document's times are (`local_time` reads it the
    # same way); `astimezone()` would read it as the PC's own and move it by the PC's offset.
    naive = datetime(2026, 11, 3, 14, 20)
    assert "registered 2026-11-03 14:20 UTC (" in replace_note(
        _plate_result("seed", 1), naive, keeps_captures=True
    )
    # A start across midnight changes the day too, not only the hour.
    late = datetime(2026, 11, 3, 1, 5, tzinfo=timezone(timedelta(hours=2)))
    assert "registered 2026-11-02 23:05 UTC (" in replace_note(
        _plate_result("seed", 1), late, keeps_captures=True
    )


class _NoLocalTime(datetime):
    """A start whose conversion to the PC's zone fails, as the PC's C library may refuse it."""

    error: ClassVar[BaseException] = OSError(22, "Invalid argument")

    def astimezone(self, tz: Any = None) -> datetime:  # type: ignore[override]
        raise self.error


@pytest.mark.parametrize(
    "error",
    [OSError(22, "Invalid argument"), OverflowError("date value out of range")],
    ids=["os-error", "overflow"],
)
def test_replace_note_shows_the_records_own_start_when_the_pc_cannot_convert_it(
    error: BaseException,
) -> None:
    # Windows refuses a time before 1970 or after year 3000, and any system a year 1 or
    # 9999 under an offset. A run document is data (FM-125): the note says the value
    # as recorded, and nothing raises (it is called by `view()`, on every redraw).
    _NoLocalTime.error = error
    started = _NoLocalTime(1, 1, 1, 0, 0, tzinfo=timezone.utc)
    note = replace_note(_plate_result("seed", 1), started, keeps_captures=True)
    assert note == (
        "Run replaces the plate registered 0001-01-01T00:00+00:00 (seed, 1 well). "
        "A failed run leaves no plate; the captures stay, so Forget the wrong one "
        "and Run again to get a plate back."
    )


# --- the page registers from images (m5 §12, #346) ----------------------------------


def test_replace_note_without_captures_does_not_tell_to_forget_them() -> None:
    # plate-finding fills the plate from the image and has no capture to forget: the
    # sentence about the captures is false under it (m5 §12.7, #367 item 3).
    started = datetime(2026, 11, 3, 14, 20, 41, tzinfo=timezone.utc)
    note = replace_note(_plate_result("references", 3), started, keeps_captures=False)
    assert note == (
        "Run replaces the plate registered 2026-11-03 14:20 UTC (references, 3 wells). "
        "A failed run leaves no plate."
    )
    assert "captures" not in note
    assert "Forget" not in note
    # And the keyword is required, so that no call forgets to say which note it wants.
    with pytest.raises(TypeError):
        replace_note(_plate_result("seed", 1), started)  # type: ignore[call-arg]


def test_the_walls_texts_are_the_designs() -> None:
    assert WALLS_DRY_RUN == (
        "plate-walls images the walls, which a dry run does not do: serve the page "
        "without --dry-run (demo-sample's stage is simulated), or register with "
        "plate-registration"
    )
    assert WALLS_NEED_CALIBRATION == (
        "plate-walls reads the camera's calibration: run camera-calibration first "
        "(Set up), with the objective you register with in place, on a field with "
        "texture (a well's wall will do)"
    )


@pytest.mark.parametrize("plate", ["384-well", "1536-well"])
def test_the_square_wells_reason_names_the_plate_and_says_to_register_with_plate_registration(
    plate: str,
) -> None:
    # The plugin's own words (m5 §12.3, #375): the operator reads the same sentence
    # beside the button as the plugin's refusal would have said.
    assert walls_square_wells(plate) == (
        f"plate-walls fits round wells, and the {plate} plate's are square: "
        "register it with plate-registration"
    )


def test_walls_centres_problem_names_what_is_missing_and_what_is_too_many() -> None:
    assert walls_centres_problem(0) == (
        "plate-walls needs the centres of 2 or 3 wells: centre a well under the "
        "objective (Live and the Drive pad), click it on the map and press Capture "
        "centre; then a second well, far from the first"
    )
    assert walls_centres_problem(1) == (
        "1 centre captured, plate-walls needs 2 or 3 wells: centre a second well, far "
        "from the first, click it on the map and press Capture centre"
    )
    # From the plugin's minimum to its maximum there is nothing to say.
    assert walls_centres_problem(2) is None
    assert walls_centres_problem(3) is None
    assert walls_centres_problem(6) is None
    assert walls_centres_problem(7) == (
        "plate-walls takes at most 6 wells and 7 centres are captured: Forget some"
    )
    assert "12 centres" in (walls_centres_problem(12) or "")


def test_the_pages_walls_bounds_are_the_plugins() -> None:
    # The page's copy of the plugin's bounds: the plugin refuses a count the page let
    # through, and the page refuses one the plugin accepts, if they drift apart.
    from smc.plugins.plate_walls import plugin

    assert WALLS_MIN_CENTRES == plugin._MIN_WELLS
    assert WALLS_MAX_CENTRES == plugin._MAX_WELLS


def test_centres_text_lists_each_centre_in_capture_order() -> None:
    assert centres_text(()) == "no centre captured yet"
    refs = (
        WellRef(well="D5", x_um=1.0, y_um=2.0),
        WellRef(well="B3", x_um=10182.04, y_um=-10087.9),
    )
    # The wells in capture order, never sorted; two spaces; one decimal; plain text.
    assert centres_text(refs) == (
        "centres captured (2):\n  D5 (1.0, 2.0) µm\n  B3 (10182.0, -10087.9) µm"
    )


def test_wall_touches_note_says_they_are_ignored_and_is_empty_for_none() -> None:
    assert wall_touches_note(0) == ""
    assert wall_touches_note(1) == (
        "1 wall capture is for plate-registration; plate-walls images the walls "
        "itself and ignores them"
    )
    assert wall_touches_note(2) == (
        "2 wall captures are for plate-registration; plate-walls images the walls "
        "itself and ignores them"
    )


def _fit(well: str, centre: tuple[float, float], frames: tuple[int, ...]) -> WellFit:
    return WellFit(
        well=well,
        recorded_xy_um=(centre[0] + 5.0, centre[1] - 5.0),
        center_xy_um=centre,
        diameter_um=15298.4,
        size_error=0.0,
        rms_um=3.0,
        n_points=40,
        walls=tuple(
            WallFrame(side=side, x_um=0.0, y_um=0.0, n_points=10, frames=n)
            for side, n in zip(("+X", "+Y", "-X", "-Y"), frames, strict=True)
        ),
    )


def _walls_result(
    residual: float | None, *, wells: tuple[WellFit, ...], plate: str = "24-well"
) -> PlateResult:
    return PlateResult(
        plate=plate,
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.1234,
        method="wall-images",
        residual_um=residual,
        scale_error=None,
        n_refs=len(wells),
        walls=WallImages(
            objective="10x",
            pixel_size_um=1.0,
            image_shape_px=(100, 100),
            search_um=3900.0,
            wells=wells,
            frames=sum(w.n_points for w in wells),
            folder="plate-walls/abc",
            elapsed_s=16.5,
        ),
    )


def test_walls_card_lines_show_each_well_then_the_plate() -> None:
    result = _walls_result(
        3.24,
        wells=(
            # The order the wells were measured in, which is not alphabetical.
            _fit("D5", (1.0, 2.0), (2, 3, 2, 3)),
            _fit("B3", (10182.04, -10087.93), (3, 4, 3, 4)),
        ),
    )
    # §12.6, character for character: the frames of the four sides summed, one decimal
    # on the centre, none on the diameter, two on the rotation.
    assert walls_card_lines(result) == (
        "D5: centre (1.0, 2.0) µm, 15298 µm across, 10 frames",
        "B3: centre (10182.0, -10087.9) µm, 15298 µm across, 14 frames",
        "plate 24-well: residual 3.2 µm, rotation 0.12°",
    )


def test_walls_card_lines_say_unknown_when_nothing_measured_the_residual() -> None:
    result = _walls_result(None, wells=(_fit("B3", (1.0, 2.0), (1, 1, 1, 1)),))
    assert walls_card_lines(result)[-1] == (
        "plate 24-well: residual unknown, rotation 0.12°"
    )


def test_walls_card_lines_stay_under_the_cards_width() -> None:
    # Six wells and the plate: seven lines, each shorter than the card's 80 characters.
    wells = tuple(
        _fit(f"A{i}", (-12345.6 * i, 98765.4 * i), (12, 12, 12, 12))
        for i in range(1, 7)
    )
    lines = walls_card_lines(_walls_result(123.4, wells=wells))
    assert len(lines) == 7
    assert max(len(line) for line in lines) <= 80


# --- the scan plan on the map and its card (m5 §14, #320) ----------------------------


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0.0, "0 s"),
        (0.4, "0 s"),
        (42.16, "42 s"),
        (59.4, "59 s"),
        (59.6, "1 min 00 s"),
        (192.0, "3 min 12 s"),
        (3599.4, "59 min 59 s"),
        (3599.6, "1 h 00 min"),
        (3900.0, "1 h 05 min"),
        # From an hour the seconds are dropped, not rounded up into the minute.
        (3659.0, "1 h 00 min"),
        (36000.0, "10 h 00 min"),
        # Python's `round` goes to the even second at a half, as §14.3 says: an estimate
        # is a sum of timings in ms, so an exact half does occur.
        (2.5, "2 s"),
        (12.5, "12 s"),
        (13.5, "14 s"),
    ],
)
def test_duration_text_rounds_to_the_second_and_names_the_unit(
    seconds: float, text: str
) -> None:
    assert duration_text(seconds) == text


def _layer(count: int, wells: int, *, drawn: bool = True) -> PlanLayer:
    return PlanLayer(
        count=count,
        wells=wells,
        fov_um=(512.0, 512.0),
        fields=tuple((float(i), 0.0) for i in range(count)) if drawn else (),
    )


def test_plan_title_says_how_many_fields_in_how_many_wells_and_the_bound() -> None:
    assert plan_title(None) == ""
    assert plan_title(_layer(72, 2)) == "; scan plan: 72 fields in 2 wells"
    # The singular, for the field and for the well.
    assert plan_title(_layer(1, 1)) == "; scan plan: 1 field in 1 well"
    # Over the bound the layer holds no field, and the title says why nothing is drawn.
    assert plan_title(_layer(20000, 1536, drawn=False)) == (
        "; scan plan: 20000 fields in 1536 wells, too many to draw (over 5000)"
    )
    assert MAP_FIELDS_UP_TO == 5000
    # The bound is a keyword, for the test of the boundary; the text quotes it.
    assert plan_title(_layer(5, 1), drawn_up_to=5) == "; scan plan: 5 fields in 1 well"
    assert plan_title(_layer(6, 1, drawn=False), drawn_up_to=5) == (
        "; scan plan: 6 fields in 1 well, too many to draw (over 5)"
    )
    # The title reads the count and not whether fields were kept: an empty plan is not
    # "too many to draw", and a count over the bound is, whatever the layer holds.
    assert plan_title(_layer(0, 3, drawn=False)) == "; scan plan: 0 fields in 3 wells"
    assert plan_title(_layer(6, 1), drawn_up_to=5) == (
        "; scan plan: 6 fields in 1 well, too many to draw (over 5)"
    )


def _scan_plan(
    wells: Sequence[str] = ("B11", "B2"),
    *,
    rows: int = 6,
    columns: int = 6,
    fov_um: tuple[float, float] = (512.0, 512.0),
    overlap: float = 0.1,
    objective: str | None = "Nikon 10X S Fluor",
    estimate_s: float | None = None,
    estimate_without_autofocus_s: float | None = None,
) -> ScanPlanResult:
    """A scan plan as ``scan-plan`` writes it, built as test_report.py builds one.

    ``wells`` are in visiting order, which is not the alphabetical one (FM-204).
    """
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    half_w, half_h = fov_um[0] / 2, fov_um[1] / 2
    plate_plan, fields = plan_fields(
        plate,
        list(wells),
        fov_um=fov_um,
        footprint_of=lambda x, y: [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ],
        rows=rows,
        columns=columns,
        overlap=overlap,
        clip_to_well=True,
    )
    return ScanPlanResult(
        plate_plan=plate_plan,
        wells=tuple(wells),
        objective=objective,
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        matrix_source="parameter",
        image_shape_px=(512, 512),
        fov_um=fov_um,
        overlap=overlap,
        rows=rows,
        columns=columns,
        fields=fields,
        estimate_s=estimate_s,
        estimate_without_autofocus_s=estimate_without_autofocus_s,
    )


def test_scan_plan_card_lines_say_the_wells_the_fields_the_field_and_the_estimate() -> (
    None
):
    result = _scan_plan(estimate_s=42.16)
    assert len(result.fields) == 72
    # §14.3, character for character: the multiplication sign is U+00D7, the micro U+00B5.
    assert scan_plan_card_lines(result) == (
        "wells: 2 (B11, B2)",
        "fields: 72, a grid of 6 × 6 in each well",  # noqa: RUF001
        "field: 512 × 512 µm at Nikon 10X S Fluor, overlap 0.1",  # noqa: RUF001
        "estimate: about 42 s (the moves, refocus and snaps of the timings)",
    )
    # A plan made without an objective says no `at ...` rather than `at None`.
    without = scan_plan_card_lines(_scan_plan(objective=None, estimate_s=1.0))
    assert without[2] == "field: 512 × 512 µm, overlap 0.1"  # noqa: RUF001
    # A real field of view is not a whole number of micrometres (0.65 µm x 2048 px):
    # `:g` shows it, where a rounding to whole micrometres would hide a tenth.
    real = scan_plan_card_lines(
        _scan_plan(("B2",), rows=1, columns=1, fov_um=(1331.2, 1331.2), overlap=0.15)
    )
    assert real[2] == "field: 1331.2 × 1331.2 µm at Nikon 10X S Fluor, overlap 0.15"  # noqa: RUF001


def test_the_card_of_a_plan_from_a_selection_names_the_selector_and_no_grid() -> None:
    # m3 §28.9: no grid was laid, so neither a grid nor an overlap is named; still
    # four lines, the wells and the estimate as for any plan.
    grid = _scan_plan(("B3", "B2"), rows=1, columns=3, estimate_s=42.16)
    selection = PlanSelection(
        run_doc="runs/run-10x.json",
        route_attempt="0123456789ab",
        measured="2026-11-26T08:15:02+00:00",
        selector="low-mag-survey",
        source_objective="Nikon 10X S Fluor",
        shift_um=(-37.5, 12.25),
        wells_without_fields=(),
        fields_left_out=0,
    )
    chosen = grid.model_copy(
        update={"rows": 1, "columns": 1, "overlap": 0.0, "selection": selection}
    )
    lines = scan_plan_card_lines(chosen)
    assert lines == (
        "wells: 2 (B3, B2)",
        "fields: 6, chosen by selector 'low-mag-survey' in runs/run-10x.json",
        "field: 512 × 512 µm at Nikon 10X S Fluor",  # noqa: RUF001
        "estimate: about 42 s (the moves, refocus and snaps of the timings)",
    )
    # A selection that left named wells without a field says how many.
    left = chosen.model_copy(
        update={
            "selection": selection.model_copy(
                update={"wells_without_fields": ("A1", "C4", "H12")}
            )
        }
    )
    assert scan_plan_card_lines(left)[1] == (
        "fields: 6, chosen by selector 'low-mag-survey' in runs/run-10x.json; none "
        "in 3 of the wells named"
    )


def test_scan_plan_card_says_what_to_run_when_the_estimate_is_unknown() -> None:
    unknown = scan_plan_card_lines(_scan_plan(estimate_s=None))[-1]
    assert unknown == (
        "estimate: unknown, no usable timing: run timing, then scan-plan again"
    )
    # An unknown is not a zero (FM-170): 0.0 is an estimate, and says so.
    zero = scan_plan_card_lines(_scan_plan(estimate_s=0.0))[-1]
    assert zero == "estimate: about 0 s (the moves, refocus and snaps of the timings)"


def test_the_scan_plan_card_says_what_its_estimate_sums_in_each_of_three_cases() -> (
    None
):
    # A lock was counted: the second figure is below the first (m5 §14.3, #549).
    locked = scan_plan_card_lines(
        _scan_plan(estimate_s=90.4, estimate_without_autofocus_s=88.4)
    )[-1]
    assert locked == (
        "estimate: about 1 min 30 s (the moves, the autofocus locked again in each "
        "well, and the snaps of the timings)"
    )
    # None was (no autofocus, or no lock time known): the two figures are equal.
    none = scan_plan_card_lines(
        _scan_plan(estimate_s=42.16, estimate_without_autofocus_s=42.16)
    )[-1]
    assert none == "estimate: about 42 s (the moves and snaps of the timings)"
    # A plan written before #549 has no second figure, and charged a refocus at each
    # field: it says so, in the words this card had.
    older = scan_plan_card_lines(_scan_plan(estimate_s=42.16))[-1]
    assert older == "estimate: about 42 s (the moves, refocus and snaps of the timings)"
    # The longest forms (FM-205): hours, and the text of a counted lock.
    longest = scan_plan_card_lines(
        _scan_plan(estimate_s=3900.0, estimate_without_autofocus_s=3800.0)
    )[-1]
    assert longest == (
        "estimate: about 1 h 05 min (the moves, the autofocus locked again in each "
        "well, and the snaps of the timings)"
    )
    # An unknown estimate is the same line whatever the second figure: it is None then.
    assert scan_plan_card_lines(_scan_plan(estimate_s=None))[-1] == (
        "estimate: unknown, no usable timing: run timing, then scan-plan again"
    )
    # A figure of 0 is a figure (FM-170): a second one of 0 under 2 s is a counted lock,
    # and two of 0 are none, neither read as the older plan's missing figure.
    zero_without = scan_plan_card_lines(
        _scan_plan(estimate_s=2.0, estimate_without_autofocus_s=0.0)
    )[-1]
    assert zero_without == (
        "estimate: about 2 s (the moves, the autofocus locked again in each well, "
        "and the snaps of the timings)"
    )
    both_zero = scan_plan_card_lines(
        _scan_plan(estimate_s=0.0, estimate_without_autofocus_s=0.0)
    )[-1]
    assert both_zero == "estimate: about 0 s (the moves and snaps of the timings)"


def test_scan_plan_card_names_the_wells_in_the_plans_order_and_cuts_after_six() -> None:
    # FM-204: B11 before B2, and a later well that sorts first. A sort would pass the
    # demo's own order only by luck of the string compare ("B11" < "B2").
    names = ("B11", "B2", "A1", "D4", "C3", "E5")
    assert scan_plan_card_lines(_scan_plan(names, rows=1, columns=1))[0] == (
        "wells: 6 (B11, B2, A1, D4, C3, E5)"
    )
    seven = (*names, "F6")
    assert scan_plan_card_lines(_scan_plan(seven, rows=1, columns=1))[0] == (
        "wells: 7 (B11, B2, A1, D4, C3, E5, … 1 more)"
    )
    # A hand-edited document with no well says so, without an empty pair of brackets.
    none = _scan_plan(rows=1, columns=1).model_copy(update={"wells": ()})
    assert scan_plan_card_lines(none)[0] == "wells: 0"


def test_scan_plan_card_says_how_many_fields_clipping_left_out() -> None:
    whole = _scan_plan(rows=2, columns=2)
    assert len(whole.fields) == 8
    assert scan_plan_card_lines(whole)[1] == "fields: 8, a grid of 2 × 2 in each well"  # noqa: RUF001
    # Clipping to the well took fields away: 14 by 14 in 2 wells is 392, the plan holds 344.
    clipped = whole.model_copy(
        update={
            "rows": 14,
            "columns": 14,
            "fields": tuple(whole.fields[i % 8] for i in range(344)),
        }
    )
    assert scan_plan_card_lines(clipped)[1] == (
        "fields: 344 of 392, a grid of 14 × 14 in each well, "  # noqa: RUF001
        "the fields that miss the well left out"
    )


# -- the map's title and the cross's label (m5 §15 D4, D5, #410) ---------------------

# Written as names, not as characters: the design says U+00B5 (the micro sign, which a
# font and an editor take for the Greek mu) and U+00B0.
UM = "\N{MICRO SIGN}m"
DEG = "\N{DEGREE SIGN}"


def test_map_title_says_which_pose_in_the_words_of_the_design() -> None:
    assert (
        map_title("96-well", MapPose("registered")) == "96-well: the registered plate"
    )
    # The registered plate names A1's corner, the check the operator makes (m5 §16 D8).
    assert (
        map_title("96-well", MapPose("registered", a1_corner="the back left"))
        == "96-well: the registered plate, A1 at the back left"
    )
    assert map_title("96-well", MapPose("seed", seed_well="B3")) == (
        "96-well: placed from one well (B3), not registered: "
        "the rotation is taken from the declared orientation, not measured"
    )
    # A seed with no reference well says so without an empty pair of brackets.
    assert map_title("96-well", MapPose("seed")) == (
        "96-well: placed from one well, not registered: "
        "the rotation is taken from the declared orientation, not measured"
    )
    assert map_title(
        "24-well", MapPose("estimated", holder_centre_um=(400.0, -325.0))
    ) == (
        "24-well: estimated from the stand's holder, not registered: "
        f"centred at stage (400, -325) {UM}, declared not turned, "
        "A1 at the top left of this map"
    )
    assert map_title(
        "24-well",
        MapPose(
            "estimated", holder_centre_um=(400.0, -325.0), orientation="turned-180"
        ),
    ) == (
        "24-well: estimated from the stand's holder, not registered: "
        f"centred at stage (400, -325) {UM}, declared turned 180{DEG}, "
        "A1 at the bottom right of this map"
    )
    assert map_title("96-well", MapPose("unplaced")) == (
        "96-well: not registered, and this stand does not say where its plate lies: "
        "A1 is drawn at stage (0, 0), so the cross shows the stage and not a well"
    )


def test_the_estimated_title_rounds_the_holder_to_whole_microns() -> None:
    # `:.0f` would print "-0" for a point a fraction of a micron below zero.
    near = MapPose("estimated", holder_centre_um=(400.4, -324.6))
    assert "centred at stage (400, -325) " in map_title("96-well", near)
    zero = MapPose("estimated", holder_centre_um=(-0.4, 0.4))
    assert "centred at stage (0, 0) " in map_title("96-well", zero)


def test_the_title_is_plain_text_whatever_the_plate_is_called() -> None:
    # FM-139: a title is never markup. The plate name is quoted as it is.
    title = map_title("<b>96-well</b>", MapPose("unplaced"))
    assert title.startswith("<b>96-well</b>: ")


# -- which way the plate lies, in words (m5 §16 D3, D8, #422) ------------------------


def test_the_a1_corner_is_in_the_operators_words_when_the_stand_says_its_sides() -> (
    None
):
    # A stand whose back is +Y and whose operator's right is +X (the demo sample's).
    assert a1_corner("not-turned", "+Y", "+X") == "the back left"
    assert a1_corner("turned-180", "+Y", "+X") == "the front right"
    assert orientation_label("not-turned", "+Y", "+X") == "A1 at the back left"
    assert (
        orientation_label("turned-180", "+Y", "+X")
        == f"Turned 180{DEG}: A1 at the front right"
    )
    # A stand whose back is +X: a word tied to an axis, or to the sign of Y, would say
    # the first stand's corner here (FM-73). A1 at -X +Y is the front right of it, and
    # +X -Y the back left.
    assert a1_corner("not-turned", "+X", "+Y") == "the front right"
    assert a1_corner("turned-180", "+X", "+Y") == "the back left"
    # A back of -Y and a right of -X, the Ti2-E's (#540): every sign flipped from the
    # first stand.
    assert a1_corner("not-turned", "-Y", "-X") == "the front right"
    assert a1_corner("turned-180", "-Y", "-X") == "the back left"


def test_the_a1_corner_is_in_stage_terms_when_the_stand_does_not() -> None:
    assert a1_corner("not-turned", None, None) == "stage -X +Y"
    assert a1_corner("turned-180", None, None) == "stage +X -Y"
    assert (
        orientation_label("not-turned", None, None) == "Not turned: A1 at stage -X +Y"
    )
    assert (
        orientation_label("turned-180", None, None)
        == f"Turned 180{DEG}: A1 at stage +X -Y"
    )
    # One direction alone names no corner (the profile refuses it; this reads it safely).
    assert a1_corner("not-turned", "+Y", None) == "stage -X +Y"
    assert "[holder] back and right" in ORIENTATION_IN_STAGE_TERMS


def test_side_words_follow_back_and_right() -> None:
    assert side_words("+Y", "+X") == {
        "+Y": "back of the stand",
        "-Y": "front of the stand",
        "+X": "operator's right",
        "-X": "operator's left",
    }
    assert side_words("-X", "+Y") == {
        "-X": "back of the stand",
        "+X": "front of the stand",
        "+Y": "operator's right",
        "-Y": "operator's left",
    }
    assert side_words(None, None) == {}
    assert side_words("+Y", None) == {}
    assert side_words(None, "+X") == {}


def test_a_seeds_title_takes_its_rotation_from_the_declared_orientation() -> None:
    title = map_title(
        "24-well",
        MapPose(
            "seed",
            seed_well="C4",
            orientation="turned-180",
            a1_corner="the front right",
        ),
    )
    assert title == (
        "24-well: placed from one well (C4), not registered: the rotation is taken "
        "from the declared orientation, A1 at the front right, not measured"
    )
    # Not the 0° that a seed assumed before an orientation could be declared.
    assert f"0{DEG}" not in title


def _well(
    name: str,
    centre: tuple[float, float],
    *,
    size: tuple[float, float] = (6000.0, 6000.0),
    circular: bool = True,
    rotation_deg: float = 0.0,
) -> WellView:
    return WellView(
        well=name,
        center_um=centre,
        size_um=size,
        circular=circular,
        rotation_deg=rotation_deg,
        selected=False,
        counts=None,
        per_mm2=None,
    )


WELLS = (
    _well("A1", (0.0, 0.0)),
    _well("B3", (18000.0, -9000.0)),
    # A turned square well: a 4000 by 2000 µm rectangle, its long side 30° from +X.
    _well(
        "C5",
        (50000.0, 0.0),
        size=(4000.0, 2000.0),
        circular=False,
        rotation_deg=30.0,
    ),
)


def test_stage_label_names_the_well_and_the_pose() -> None:
    on_b3 = (18000.0, -9000.0)
    assert stage_label(WELLS, MapPose("registered"), on_b3) == "over B3"
    assert stage_label(WELLS, MapPose("estimated"), on_b3) == "over B3 (estimated)"
    assert stage_label(WELLS, MapPose("seed"), on_b3) == "over B3 (rotation assumed)"
    # The edge counts, as `WellGeometry.contains` has it.
    assert stage_label(WELLS, MapPose("registered"), (18000.0 + 3000.0, -9000.0)) == (
        "over B3"
    )


def test_stage_label_is_empty_where_the_map_cannot_name_a_well() -> None:
    between = (9000.0, -4500.0)
    assert stage_label(WELLS, MapPose("registered"), between) == ""
    # No position (a closed stand), and a map that does not say where its plate is.
    assert stage_label(WELLS, MapPose("registered"), None) == ""
    assert stage_label(WELLS, MapPose("unplaced"), (0.0, 0.0)) == ""
    # No wells at all, and a point that is not a number: unknown is not inside.
    assert stage_label((), MapPose("registered"), (0.0, 0.0)) == ""
    assert stage_label(WELLS, MapPose("registered"), (float("nan"), 0.0)) == ""


def test_stage_label_follows_the_rotation_of_a_square_well() -> None:
    inside = (50000.0 + 1800.0 * 0.8660254, 1800.0 * 0.5)
    assert stage_label(WELLS, MapPose("registered"), inside) == "over C5"
    # Inside the well's axis-aligned box (1900, 800 from its centre), outside the
    # rectangle turned by 30°: a label that ignored the rotation would name C5.
    outside = (50000.0 + 1900.0, 800.0)
    assert stage_label(WELLS, MapPose("registered"), outside) == ""


def _turned(
    centre: tuple[float, float], u_um: float, v_um: float, degrees: float
) -> tuple[float, float]:
    """The stage point at ``(u, v)`` from ``centre`` in a plate turned by ``degrees``, written out."""
    angle = math.radians(degrees)
    return (
        centre[0] + math.cos(angle) * u_um - math.sin(angle) * v_um,
        centre[1] + math.sin(angle) * u_um + math.cos(angle) * v_um,
    )


def test_stage_label_tells_a_rectangle_from_a_disc_and_its_height_from_its_width() -> (
    None
):
    # C5 is the turned 4000 by 2000 µm rectangle of WELLS: half-sizes 2000 (u) and 1000 (v).
    c5 = (50000.0, 0.0)
    registered = MapPose("registered")
    # (1900, 900) is in the rectangle and outside the 2000 µm disc (2102 µm from the
    # centre): a well passed on as a disc would not name it.
    assert stage_label(WELLS, registered, _turned(c5, 1900.0, 900.0, 30.0)) == "over C5"
    # (0, 1500) is in the disc and 500 µm above the rectangle's 1000: a well whose height
    # was taken for its width (a 4000 µm square) would name it.
    assert stage_label(WELLS, registered, _turned(c5, 0.0, 1500.0, 30.0)) == ""
    # A round well is not its bounding square: A1's 6000 µm disc leaves out the corner.
    assert stage_label(WELLS, registered, (2900.0, 2900.0)) == ""
    assert stage_label(WELLS, registered, (2100.0, 2100.0)) == "over A1"


def test_stage_label_takes_the_first_well_in_the_views_order() -> None:
    # Wells do not overlap on a real plate; if a view's did, its order decides.
    twin = _well("Z9", (0.0, 0.0))
    assert stage_label((WELLS[0], twin), MapPose("registered"), (0.0, 0.0)) == "over A1"
    assert stage_label((twin, WELLS[0]), MapPose("registered"), (0.0, 0.0)) == "over Z9"


# -- the page's steps (m5 §17.5, #443) --------------------------------------------------


def _checklist(**states: str) -> tuple[CheckItem, ...]:
    """The page's checklist, in its order, with ``states`` by slot (``plate="ok"``), else ``not run``.

    A slot is written with an underscore for the dash (``scan_plan``): a keyword cannot
    hold a ``-``. The optional flags are the controller's own, so a change to the order
    or the optional slots shows up here.
    """
    wanted = {slot.replace("-", "_"): state for slot, state in states.items()}
    return tuple(
        _item(slot, wanted.get(slot.replace("-", "_"), "not run"), optional=optional)
        for slot, optional in _CHECKLIST
    )


def _states(
    checklist: Sequence[CheckItem],
    *,
    connected: bool = True,
    stand: str = "demo-sample",
    preset: str | None = None,
    plate: str = "96-well",
    plan: PlanLayer | None = None,
) -> dict[StepName, StepState]:
    return {
        state.step: state
        for state in step_states(
            checklist,
            connected=connected,
            stand=stand,
            preset=preset,
            plate=plate,
            plan=plan,
        )
    }


def test_every_checklist_slot_has_a_step() -> None:
    # A slot of the controller's checklist that no step holds would be in the expert
    # view only, and the step bar could never say it is waiting for it.
    assert {slot for slot, _optional in _CHECKLIST} <= set(STEP_OF_SLOT)
    assert set(STEP_OF_SLOT.values()) <= set(STEPS)
    # Every step holds something: a step with no slot could never be done or suggested.
    assert set(STEP_OF_SLOT.values()) == set(STEPS)
    assert tuple(STEP_TITLES) == STEPS
    assert set(SCREEN_TITLES) == set(STEPS) == set(SCREEN_LEADS)


def test_a_step_is_done_when_its_required_slots_are_ok() -> None:
    nothing = _states(_checklist())
    assert [nothing[step].done for step in STEPS] == [True, False, False, False, False]

    plate = _states(_checklist(plate="ok"))
    assert [plate[step].done for step in STEPS] == [True, True, False, False, False]

    # The optional slots never decide: the stand is done by the connection alone, and
    # a camera calibration that is ok does not make any other step done.
    calibrated = _states(_checklist(camera_calibration="ok"))
    assert [calibrated[step].done for step in STEPS] == [
        True,
        False,
        False,
        False,
        False,
    ]
    away = _states(_checklist(camera_calibration="ok", plate="ok"), connected=False)
    assert away["stand"].done is False
    assert away["plate"].done is True

    # Results is done when all five of its slots are, and not one before.
    four = _states(_checklist(cells="ok", density="ok", route="ok", report="ok"))
    assert four["results"].done is False
    five = _states(
        _checklist(cells="ok", density="ok", route="ok", report="ok", nis_export="ok")
    )
    assert five["results"].done is True

    # A slot that is unusable is not ok: the step it belongs to is not done.
    assert _states(_checklist(scan="unusable"))["scan"].done is False


def test_a_step_whose_slots_the_checklist_does_not_hold_is_not_done() -> None:
    # Unknown beats guessed: an empty checklist says nothing about the plate, and a step
    # with no required slot in it is not done (all([]) would say it is).
    states = _states(())
    assert [states[step].done for step in STEPS] == [True, False, False, False, False]


def test_the_suggested_step_holds_the_next_slot() -> None:
    # Walk the workflow: each slot made ok in turn moves the suggestion to the step of
    # the next one, and nothing is suggested once everything is done.
    order = [
        ("plate", "plate"),
        ("scan-plan", "wells"),
        ("scan", "scan"),
        ("cells", "results"),
        ("density", "results"),
        ("route", "results"),
        ("report", "results"),
        ("nis-export", "results"),
    ]
    done: dict[str, str] = {}
    for slot, step in order:
        states = _states(_checklist(**done))
        assert [s.step for s in states.values() if s.suggested] == [step]
        done[slot] = "ok"
    everything = _states(_checklist(**done))
    assert not any(state.suggested for state in everything.values())
    assert all(state.done for state in everything.values())

    # An unusable slot is still the next one: the step is suggested to run it again.
    stale = _states(_checklist(plate="ok", scan_plan="unusable"))
    assert [s.step for s in stale.values() if s.suggested] == ["wells"]


def test_the_page_opens_on_the_stand_until_it_is_connected() -> None:
    # A page not connected suggests the stand, whatever the run document holds.
    away = step_states(
        _checklist(plate="ok", scan_plan="ok"),
        connected=False,
        stand="demo-sample",
        preset=None,
        plate="96-well",
        plan=None,
    )
    assert [s.step for s in away if s.suggested] == ["stand"]
    assert opening_step(away) == "stand"

    # Connected, it opens on the step of the next slot, and on Results when nothing is
    # left (the owner reads what was made).
    connected = step_states(
        _checklist(plate="ok"),
        connected=True,
        stand="demo-sample",
        preset=None,
        plate="96-well",
        plan=None,
    )
    assert opening_step(connected) == "wells"
    all_ok = {slot: "ok" for slot, _optional in _CHECKLIST}
    finished = step_states(
        _checklist(**all_ok),
        connected=True,
        stand="demo-sample",
        preset=None,
        plate="96-well",
        plan=None,
    )
    assert opening_step(finished) == "results"
    assert opening_step(()) == "results"


@given(
    st.lists(
        st.sampled_from(["ok", "not run", "unusable"]),
        min_size=len(_CHECKLIST),
        max_size=len(_CHECKLIST),
    ),
    st.booleans(),
)
def test_at_most_one_step_is_suggested_and_never_a_done_one(
    states: list[str], connected: bool
) -> None:
    checklist = tuple(
        _item(slot, state, optional=optional)
        for (slot, optional), state in zip(_CHECKLIST, states, strict=True)
    )
    result = step_states(
        checklist,
        connected=connected,
        stand="s",
        preset=None,
        plate="96-well",
        plan=None,
    )
    suggested = [state for state in result if state.suggested]
    assert len(suggested) <= 1
    assert not any(state.done for state in suggested)
    # The page opens on the suggested step, else on Results.
    assert opening_step(result) == (suggested[0].step if suggested else "results")
    # A stand that is not connected is the one suggestion, and the stand is not done.
    if not connected:
        assert [state.step for state in suggested] == ["stand"]


def test_a_done_steps_summary_names_what_it_holds() -> None:
    all_ok = {slot: "ok" for slot, _optional in _CHECKLIST}
    plan = PlanLayer(count=72, wells=2, fov_um=(512.0, 512.0), fields=())
    states = _states(
        _checklist(**all_ok), stand="demo-sample", preset="weekly", plan=plan
    )
    # 22 characters at most (m5 §17.12 D15): the step bar is one line tall.
    assert states["stand"].summary == "demo-sample, preset w…"
    assert states["plate"].summary == "96-well, registered"
    assert states["wells"].summary == "2 wells, 72 fields"
    assert states["scan"].summary == "scanned"
    assert states["results"].summary == "5 of 5 done"

    # No preset: the stand alone. Not connected: it says so, and it is the one asked for.
    assert _states(_checklist(), preset=None)["stand"].summary == "demo-sample"
    away = _states(_checklist(), connected=False)["stand"]
    assert (away.summary, away.suggested) == ("not connected", True)

    # The singular, for a field and for a well.
    one = PlanLayer(count=1, wells=1, fov_um=(1.0, 1.0), fields=())
    assert _states(_checklist(**all_ok), plan=one)["wells"].summary == "1 well, 1 field"
    # A plan the page could not read still says what the slot says, nothing it invents.
    assert _states(_checklist(**all_ok), plan=None)["wells"].summary == "laid out"

    # Results counts what is done, and says so before it is all done.
    partial = _states(_checklist(plate="ok", scan_plan="ok", scan="ok", cells="ok"))
    assert partial["results"].summary == "1 of 5 done"
    assert partial["results"].suggested is True


def test_a_step_neither_done_nor_suggested_says_nothing_and_the_suggested_says_next() -> (
    None
):
    states = _states(_checklist(plate="ok"))
    assert states["wells"].summary == "next"
    assert states["scan"].summary == ""
    assert states["results"].summary == ""


def test_a_step_summary_is_cut_at_22_characters() -> None:
    long = "a-stand-with-a-very-long-profile-name-indeed"
    summary = _states(_checklist(), stand=long, preset="a-preset-with-a-long-name")[
        "stand"
    ].summary
    assert len(summary) == 22
    assert summary == "a-stand-with-a-very-l…"
    assert "\n" not in summary
    # Exactly 22 is not cut; 23 is.
    assert (
        _states(_checklist(), stand="s" * 22, preset=None)["stand"].summary == "s" * 22
    )
    assert (
        _states(_checklist(), stand="s" * 23, preset=None)["stand"].summary
        == "s" * 21 + "…"
    )
    # A short one is never cut.
    assert (
        _states(_checklist(), stand="x", preset="p")["stand"].summary == "x, preset p"
    )


def test_step_labels_are_unique_and_numbered() -> None:
    states = step_states(
        _checklist(plate="ok"),
        connected=True,
        stand="demo-sample",
        preset=None,
        plate="96-well",
        plan=None,
    )
    labels = [step_button_label(i, state) for i, state in enumerate(states, start=1)]
    assert labels == [
        "1. Stand ✓",
        "2. Plate ✓",
        "3. Wells",
        "4. Scan",
        "5. Results",
    ]
    assert len(set(labels)) == len(labels)
    # None of them is a bare title: a step's name starts with its number, so that it is
    # never the one of the *Plate* select or of the map's title.
    assert all(label[0].isdigit() for label in labels)


def test_the_screens_titles_and_leads_are_the_designs() -> None:
    assert SCREEN_TITLES == {
        "stand": "Which stand, and which preset?",
        "plate": "Where is the plate?",
        "wells": "Which wells?",
        "scan": "Scan the selected wells",
        "results": "Results",
    }
    assert SCREEN_LEADS == {
        "stand": (
            "The preset fills in the plate, the scan's coverage and the cell "
            "detector. You can change any of them later."
        ),
        "plate": (
            "Centre three wells far apart by eye, with Live and the arrows, and "
            "capture each centre."
        ),
        "wells": (
            "Click wells on the map, or type them. The map shows the fields and "
            "the time the scan will take."
        ),
        "scan": "STOP, at the top right, stops the scan.",
        "results": (
            "Each step reads the one before. Run the next one, or all that are left."
        ),
    }
    assert STEP_TITLES == {
        "stand": "Stand",
        "plate": "Plate",
        "wells": "Wells",
        "scan": "Scan",
        "results": "Results",
    }


def test_continue_label_names_the_next_step_and_none_after_results() -> None:
    assert continue_label("stand") == "Continue to Plate"
    assert continue_label("plate") == "Continue to Wells"
    assert continue_label("wells") == "Continue to Scan"
    assert continue_label("scan") == "Continue to Results"
    assert continue_label("results") is None


# -- the Stand and Plate screens (m5 §17.4.1, §17.4.2, #444) ------------------------------


def test_suggested_centres_are_three_far_corners_as_printed() -> None:
    # Measured on useq-schema 0.9.2 when #444 was planned: A1, the last column of row A and
    # the last row of column 1, as the plate prints them. Typed out, not read from the
    # function (FM-43): a plate whose rows run past Z (1536-well) is named `AF`.
    assert suggested_centres("96-well") == ("A1", "A12", "H1")
    assert suggested_centres("384-well") == ("A1", "A24", "P1")
    assert suggested_centres("1536-well") == ("A1", "A48", "AF1")
    assert suggested_centres("6-well") == ("A1", "A3", "B1")
    assert suggested_centres("24-well") == ("A1", "A6", "D1")


def test_a_one_well_plate_suggests_one_centre() -> None:
    # A coverslip has one well: the list has one row, never A1 three times.
    assert suggested_centres("coverslip-18mm-square") == ("A1",)
    assert suggested_centres("coverslip-22mm-square") == ("A1",)


# -- the centres a stand can reach (m5 §19, #512) -------------------------------------------

#: The Ti2-E's ``[safety] xy_soft_limits_um`` and ``[holder] plate_centre_xy_um``
#: (profiles/nikon-ti2.toml). Measured 2026-11-16 with ``holder_estimate("greiner-24-well",
#: (400, -325), orientation)``: A1 (-48 225, 28 850), A6 (49 025, 28 850) and D1
#: (-48 225, -29 500) µm, not turned; B2 (-28 775, 9 400), B5 (29 575, 9 400) and C2
#: (-28 775, -10 050). Typed out, never read back from the function (FM-43).
TI2E_LIMITS = ((-49000.0, 49000.0), (-29000.0, 29000.0))
TI2E_HOLDER = (400.0, -325.0)
GREINER_CORNERS_NOTE = (
    "Not the plate's far corners (A1, A6 and D1): the stage's soft limits cut them, "
    "as the holder places this plate."
)
NOTHING_REACHABLE_NOTE = (
    "No well of this plate lies inside the stage's soft limits as the holder places it, "
    "so Go to will be refused: check the profile's [holder] and [safety]."
)


@pytest.mark.parametrize("orientation", ["not-turned", "turned-180"])
def test_suggested_centres_on_the_ti2e_stay_inside_its_soft_limits(
    orientation: Literal["not-turned", "turned-180"],
) -> None:
    # A turn renames the wells in place and keeps the block (m5 §19.2 D2), so both
    # orientations suggest the same three names.
    assert suggested_centres(
        "greiner-24-well",
        holder_centre_um=TI2E_HOLDER,
        orientation=orientation,
        xy_limits_um=TI2E_LIMITS,
    ) == ("B2", "B5", "C2")


def test_suggested_centres_are_the_corners_without_a_holder_or_without_limits() -> None:
    # Nothing placed, or nothing cut: the rows are what they were (m5 §19.2 D2).
    corners = ("A1", "A6", "D1")
    assert suggested_centres("greiner-24-well") == corners
    assert suggested_centres("greiner-24-well", holder_centre_um=TI2E_HOLDER) == corners
    assert suggested_centres("greiner-24-well", xy_limits_um=TI2E_LIMITS) == corners
    # A stand whose travel holds the plate in its holder: the corners again.
    wide = ((-100000.0, 100000.0), (-100000.0, 100000.0))
    assert (
        suggested_centres(
            "greiner-24-well", holder_centre_um=TI2E_HOLDER, xy_limits_um=wide
        )
        == corners
    )


def test_suggested_centres_keep_the_corners_when_no_well_is_reachable() -> None:
    # The rows are built once and never empty (m5 §19.2 D4): the corners stay and the
    # note says why *Go to* will be refused.
    far = ((1e6, 2e6), (1e6, 2e6))
    assert suggested_centres(
        "greiner-24-well", holder_centre_um=TI2E_HOLDER, xy_limits_um=far
    ) == ("A1", "A6", "D1")


def test_an_unknown_plate_is_refused_with_useqs_error_by_both() -> None:
    with pytest.raises(ValueError, match="Unknown plate name 'nope'"):
        suggested_centres("nope")
    with pytest.raises(ValueError, match="Unknown plate name 'nope'"):
        centres_reach_note("nope")
    with pytest.raises(ValueError, match="Unknown plate name 'nope'"):
        suggested_centres(
            "nope", holder_centre_um=TI2E_HOLDER, xy_limits_um=TI2E_LIMITS
        )
    with pytest.raises(ValueError, match="Unknown plate name 'nope'"):
        centres_reach_note(
            "nope", holder_centre_um=TI2E_HOLDER, xy_limits_um=TI2E_LIMITS
        )


def test_the_reach_note_says_why_the_corners_are_not_the_rows() -> None:
    # Some wells reachable and the rows are not the corners: the corners are named.
    assert (
        centres_reach_note(
            "greiner-24-well",
            holder_centre_um=TI2E_HOLDER,
            xy_limits_um=TI2E_LIMITS,
        )
        == GREINER_CORNERS_NOTE
    )
    assert CENTRES_CUT_NOTE.format(corners="A1, A6 and D1") == GREINER_CORNERS_NOTE
    # No well reachable: the rows are the corners and *Go to* is refused; the note says so.
    far = ((1e6, 2e6), (1e6, 2e6))
    assert (
        centres_reach_note(
            "greiner-24-well", holder_centre_um=TI2E_HOLDER, xy_limits_um=far
        )
        == NOTHING_REACHABLE_NOTE
    )
    assert CENTRES_UNREACHABLE_NOTE == NOTHING_REACHABLE_NOTE


def test_the_reach_note_is_empty_where_nothing_is_cut() -> None:
    # The rows are the corners, so there is nothing to explain (m5 §19.2 D5): no holder,
    # no limits, neither, and a travel that holds every well.
    assert centres_reach_note("greiner-24-well") == ""
    assert centres_reach_note("greiner-24-well", holder_centre_um=TI2E_HOLDER) == ""
    assert centres_reach_note("greiner-24-well", xy_limits_um=TI2E_LIMITS) == ""
    wide = ((-100000.0, 100000.0), (-100000.0, 100000.0))
    assert (
        centres_reach_note(
            "greiner-24-well", holder_centre_um=TI2E_HOLDER, xy_limits_um=wide
        )
        == ""
    )
    # The 6-well's corners are all inside the Ti2-E's travel (m5 §19.1).
    assert (
        centres_reach_note(
            "6-well", holder_centre_um=TI2E_HOLDER, xy_limits_um=TI2E_LIMITS
        )
        == ""
    )


def test_the_reach_note_names_the_corners_of_another_plate() -> None:
    # A 96-well whose holder centre is the origin, in a 40 mm square travel: the corners
    # A1, A12 and H1 are far outside and the rows are the block that is inside. Typed
    # out: the columns are 9 mm apart, so only x = +-4.5 and +-13.5 mm are 1 mm inside.
    square = ((-20000.0, 20000.0), (-20000.0, 20000.0))
    kwargs = {"holder_centre_um": (0.0, 0.0), "xy_limits_um": square}
    assert suggested_centres("96-well", **kwargs) == ("C5", "C8", "F5")
    assert centres_reach_note("96-well", **kwargs) == (
        "Not the plate's far corners (A1, A12 and H1): the stage's soft limits cut them, "
        "as the holder places this plate."
    )


def test_a_turn_gives_its_own_rows_on_a_travel_that_is_not_symmetric() -> None:
    # On the Ti2-E's travel a turn only renames wells in place and the block keeps its
    # names, so no test there can see the orientation. A 96-well centred on the origin in
    # a travel that holds its left half only: not turned, the left half is columns 1 to 6;
    # turned 180 degrees about the holder, those wells are named by the other half, 7 to
    # 12. Typed out from the geometry (9 mm pitch, A1 at (-49 500, 31 500) not turned and
    # at (49 500, -31 500) turned).
    left_half = ((-60000.0, 0.0), (-40000.0, 40000.0))
    kwargs = {"holder_centre_um": (0.0, 0.0), "xy_limits_um": left_half}
    assert suggested_centres("96-well", orientation="not-turned", **kwargs) == (
        "A1",
        "A6",
        "H1",
    )
    assert suggested_centres("96-well", orientation="turned-180", **kwargs) == (
        "A7",
        "A12",
        "H7",
    )
    assert centres_reach_note("96-well", orientation="turned-180", **kwargs) == (
        "Not the plate's far corners (A1, A12 and H1): the stage's soft limits cut them, "
        "as the holder places this plate."
    )


def test_the_corners_are_joined_in_words() -> None:
    assert _join_names(("A1",)) == "A1"
    assert _join_names(("A1", "A4")) == "A1 and A4"
    assert _join_names(("A1", "A6", "D1")) == "A1, A6 and D1"


def test_every_suggested_centre_is_a_target_the_layers_guard_accepts() -> None:
    # FM-169: the guard stays the guard and the list is a courtesy. The layer's own
    # check accepts the estimated centre of each suggested well and refuses the corners
    # the list used to offer (m5 §19.4). Positions typed out from the measurement above.
    safety = Safety(max_jog_um=5000.0, xy_soft_limits_um=TI2E_LIMITS)
    chosen = suggested_centres(
        "greiner-24-well",
        holder_centre_um=TI2E_HOLDER,
        xy_limits_um=TI2E_LIMITS,
    )
    centres = {
        "B2": (-28775.0, 9400.0),
        "B5": (29575.0, 9400.0),
        "C2": (-28775.0, -10050.0),
    }
    assert chosen == ("B2", "B5", "C2")
    for well in chosen:
        safety.check_xy_target_um(*centres[well])
    for corner in ((49025.0, 28850.0), (-48225.0, -29500.0)):  # A6, D1
        with pytest.raises(SafetyRefusedError, match="outside the soft limits"):
            safety.check_xy_target_um(*corner)


def test_known_text_says_where_each_value_comes_from_or_that_it_is_not_known() -> None:
    here = CalibrationNote(objective="Nikon 10X", when_utc="2026-11-12 09:30:00")
    lines = known_text(objective="Nikon 10X", calibration=here, light_on=True)
    assert lines.split("\n") == [
        "Objective: Nikon 10X, in place",
        "Camera: calibrated at Nikon 10X on 2026-11-12 09:30:00 UTC",
        "Light: on",
    ]
    # Nothing is read: each line says it is not known, never a guess (CLAUDE.md).
    nothing = known_text(objective=None, calibration=None, light_on=None)
    assert nothing.split("\n") == [
        "Objective: not known",
        "Camera: not calibrated (the arrows cannot say which way the image moves)",
        "Light: not known",
    ]
    off = known_text(objective="Nikon 10X", calibration=here, light_on=False)
    assert off.split("\n")[2] == "Light: off"
    # A calibration at another objective says both, and which one is in place.
    other = CalibrationNote(objective="Nikon 40X", when_utc="2026-11-12 09:30:00")
    another = known_text(objective="Nikon 10X", calibration=other, light_on=True)
    assert another.split("\n")[1] == (
        "Camera: calibrated at Nikon 40X on 2026-11-12 09:30:00 UTC, not at Nikon 10X"
    )
    # A calibration whose objective was not recorded does not claim one.
    unrecorded = CalibrationNote(objective=None, when_utc="2026-11-12 09:30:00")
    said = known_text(objective="Nikon 10X", calibration=unrecorded, light_on=True)
    assert said.split("\n")[1] == (
        "Camera: calibrated on 2026-11-12 09:30:00 UTC, objective not recorded"
    )
    # An objective that is not read says what the calibration was made at, and no more.
    blind = known_text(objective=None, calibration=here, light_on=True)
    assert blind.split("\n")[1] == (
        "Camera: calibrated at Nikon 10X on 2026-11-12 09:30:00 UTC"
    )
    # The turret's label is a stand's text: printed as it is, never as markup (FM-139).
    markup = known_text(objective="<b>10x</b>", calibration=None, light_on=None)
    assert markup.split("\n")[0] == "Objective: <b>10x</b>, in place"


def test_calibration_is_offered_when_none_or_another_objectives() -> None:
    here = CalibrationNote(objective="Nikon 10X", when_utc="2026-11-12 09:30:00")
    unrecorded = CalibrationNote(objective=None, when_utc="2026-11-12 09:30:00")
    # None at all: offered, whatever is in place.
    assert needs_calibration("Nikon 10X", None) is True
    assert needs_calibration(None, None) is True
    # Made at the objective in place: not offered.
    assert needs_calibration("Nikon 10X", here) is False
    # Both known and different: the matrix of another objective is not this one's.
    assert needs_calibration("Nikon 40X", here) is True
    # One of them unknown: the page does not claim a difference it cannot see.
    assert needs_calibration(None, here) is False
    assert needs_calibration("Nikon 10X", unrecorded) is False


def _ref(well: str, x_um: float = 0.0, y_um: float = 0.0) -> WellRef:
    return WellRef(well=well, x_um=x_um, y_um=y_um)


def test_centre_rows_list_the_suggested_wells_then_other_captures() -> None:
    rows = centre_rows(
        "96-well",
        [_ref("C5", 10.0, 20.0), _ref("A12", 30.0, 40.0)],
        {"C5": "Nikon 10X", "A12": "Nikon 10X"},
        pose="estimated",
        from_sample=False,
    )
    # The three suggested wells first, in the suggestion's order, then a capture of
    # another well in capture order: the suggestion is not a rule.
    assert [row.well for row in rows] == ["A1", "A12", "H1", "C5"]
    assert [row.captured for row in rows] == [False, True, False, True]
    assert [row.text for row in rows] == [
        "A1: go there (estimated position), centre it by eye, capture",
        "A12: captured at (30.0, 40.0) µm with Nikon 10X",
        "H1: go there (estimated position), centre it by eye, capture",
        "C5: captured at (10.0, 20.0) µm with Nikon 10X",
    ]
    # Nothing captured: the three, none of them captured.
    bare = centre_rows("96-well", [], {}, pose="registered", from_sample=False)
    assert [(row.well, row.captured) for row in bare] == [
        ("A1", False),
        ("A12", False),
        ("H1", False),
    ]
    # Each pose says how to get there (m5 §17.4.2, D10): registered, estimated, unplaced.
    assert bare[0].text == "A1: go there, centre it, capture"
    unplaced = centre_rows("96-well", [], {}, pose="unplaced", from_sample=False)
    assert unplaced[0].text == "A1: jog there with the arrows, centre it, capture"
    # `seed` is a plate placed from one well: the wells are drawn where they were given.
    seed = centre_rows("96-well", [], {}, pose="seed", from_sample=False)
    assert seed[0].text == "A1: go there, centre it, capture"
    # On a stand whose profile has a sample of this plate and nothing captured, the page
    # registers from the sample's centres: known, not measured (m5 §8.2).
    sample = centre_rows("96-well", [], {}, pose="estimated", from_sample=True)
    assert [row.text for row in sample] == [
        "A1: known from the profile's sample, not measured",
        "A12: known from the profile's sample, not measured",
        "H1: known from the profile's sample, not measured",
    ]
    assert [row.captured for row in sample] == [False, False, False]


def test_centre_rows_list_the_wells_they_are_given_and_the_other_captures_after() -> (
    None
):
    # The page gives the wells the stand can reach (m5 §19.3): the rows are those, in
    # that order, and a capture of A1, which is not among them, is listed after.
    rows = centre_rows(
        "greiner-24-well",
        [_ref("A1", 5.0, 6.0)],
        {"A1": "Nikon 10X"},
        pose="estimated",
        from_sample=False,
        suggested=("B2", "B5", "C2"),
    )
    assert [row.well for row in rows] == ["B2", "B5", "C2", "A1"]
    assert [row.captured for row in rows] == [False, False, False, True]
    assert (
        rows[0].text == "B2: go there (estimated position), centre it by eye, capture"
    )
    assert rows[3].text == "A1: captured at (5.0, 6.0) µm with Nikon 10X"
    # A capture of a suggested well is that row, not a second one.
    again = centre_rows(
        "greiner-24-well",
        [_ref("B2", 1.0, 2.0)],
        {"B2": "Nikon 10X"},
        pose="estimated",
        from_sample=False,
        suggested=("B2", "B5", "C2"),
    )
    assert [(row.well, row.captured) for row in again] == [
        ("B2", True),
        ("B5", False),
        ("C2", False),
    ]
    # Without `suggested` the rows are the plate's own suggestion, as they were.
    plain = centre_rows("greiner-24-well", [], {}, pose="estimated", from_sample=False)
    assert [row.well for row in plain] == ["A1", "A6", "D1"]
    # An empty suggestion is the page's choice too: it lists the captures only.
    only = centre_rows(
        "greiner-24-well",
        [_ref("A1")],
        {},
        pose="registered",
        from_sample=False,
        suggested=(),
    )
    assert [row.well for row in only] == ["A1"]


def test_a_capture_row_names_its_objective_or_says_it_is_not_known() -> None:
    # The list shows the objective of each capture, so that a turn between two captures
    # is visible (m5 §17.4.2, #373 is not decided here).
    rows = centre_rows(
        "96-well",
        [_ref("A1", 1.0, 2.0), _ref("A12", -0.04, 5.0), _ref("H1", 3.0, 4.0)],
        {"A1": "Nikon 10X", "A12": None},
        pose="registered",
        from_sample=False,
    )
    assert [row.text for row in rows] == [
        "A1: captured at (1.0, 2.0) µm with Nikon 10X",
        # -0.04 rounds to a tenth as -0.0, which the page does not print (as `_um`).
        "A12: captured at (0.0, 5.0) µm with the objective not known",
        # No entry at all (a capture restored from a run document): the same words.
        "H1: captured at (3.0, 4.0) µm with the objective not known",
    ]


def test_registered_line_says_when_the_residual_was_not_measured() -> None:
    def result(residual_um: float | None, rotation_deg: float) -> PlateResult:
        return PlateResult(
            plate="96-well",
            a1_center_xy_um=(0.0, 0.0),
            rotation_deg=rotation_deg,
            method="references",
            residual_um=residual_um,
            scale_error=None,
            n_refs=3,
        )

    assert (
        registered_line(result(121.64, 0.1))
        == "Registered: residual 121.6 µm of the 300 µm allowed, turned 0.10°"
    )
    # A seed measures no residual: an unknown is not a zero (CLAUDE.md).
    assert (
        registered_line(result(None, -0.456))
        == "Registered: residual not measured, turned -0.46°"
    )
    assert (
        registered_line(result(0.0, 0.0))
        == "Registered: residual 0.0 µm of the 300 µm allowed, turned 0.00°"
    )


def test_register_replace_note_names_the_plate_it_replaces_and_the_way_back() -> None:
    def result(method: str, n_refs: int) -> PlateResult:
        return PlateResult(
            plate="96-well",
            a1_center_xy_um=(0.0, 0.0),
            rotation_deg=0.0,
            method=method,  # type: ignore[arg-type]
            residual_um=None,
            scale_error=None,
            n_refs=n_refs,
        )

    # m5 §11.4: a failed registration leaves no plate; said before the click, in this
    # screen's own words (*Redo*, not the tool's *Forget* and *Run*), whole (FM-45).
    assert register_replace_note(result("references", 3)) == (
        "Registering again replaces this plate (references, 3 wells). A failed run "
        "leaves no plate; the centres captured stay, so Redo the wrong one and "
        "register again to get a plate back."
    )
    assert register_replace_note(result("seed", 1)).startswith(
        "Registering again replaces this plate (seed, 1 well). "
    )


def _loaded_for_plate(plate: str | None) -> LoadedPreset:
    file = PresetFile("weekly", PRESET_PATH, Preset(plate=plate))
    return LoadedPreset(
        file=file,
        check=PresetCheck("weekly", None, None, {}, ()),
        fields={},
        selection=None,
        mismatches=(),
    )


def test_plate_from_line_names_the_placeholder_the_preset_the_sample_or_the_page() -> (
    None
):
    def say(
        *,
        chosen: bool = True,
        loaded: LoadedPreset | None = None,
        sample_plate: str | None = None,
    ) -> str:
        return plate_from_line(
            plate="96-well", chosen=chosen, loaded=loaded, sample_plate=sample_plate
        )

    weekly = _loaded_for_plate("96-well")
    # The first that holds. Nothing named the plate: it is the page's placeholder, whatever
    # a preset or a sample says of another plate.
    assert (
        say(chosen=False, loaded=weekly, sample_plate="96-well")
        == "the page's placeholder: choose the plate on the stage"
    )
    assert say(loaded=weekly, sample_plate="96-well") == "from the preset weekly"
    assert say(sample_plate="96-well") == "from the profile's sample"
    assert say() == "chosen on this page"
    # A preset of another plate, or with no plate, names nothing of this one.
    assert say(loaded=_loaded_for_plate("24-well")) == "chosen on this page"
    assert say(loaded=_loaded_for_plate(None)) == "chosen on this page"
    # A sample of another plate neither.
    assert say(sample_plate="384-well") == "chosen on this page"


def test_go_to_is_not_offered_during_live_says_why_and_that_capture_works() -> None:
    # The sentence of D11 (m5 §17.2), pinned as written: it is what the operator reads
    # beside a disabled *Go to*.
    assert GOTO_NOT_DURING_LIVE == (
        "End live to go to another well; Capture works during Live"
    )


def test_the_other_ways_to_register_name_the_walls_and_the_survey_and_the_expert_view() -> (
    None
):
    # D3: the guided path offers three centres by eye; the two slower ways are named, and
    # where to run them. Plain words, whole (FM-45).
    assert OTHER_WAYS_TO_REGISTER == (
        "Other ways to register: from the walls of 2 or 3 wells (slower; measures the "
        "wells' size too) or automatically (surveys the plate, about 7 minutes): in the "
        "expert view."
    )


# -- the Wells and Scan screens (m5 §17.4.3, §17.4.4, §17.10, #445) -----------------------


NOT_CONNECTED_TEXT = "the stand is not connected"


def _plan(
    *,
    count: int = 72,
    wells: int = 2,
    objective: str | None = "Nikon 10X S Fluor",
    estimate_s: float | None = 42.16,
    estimate_without_autofocus_s: float | None = None,
) -> PlanLayer:
    return PlanLayer(
        count=count,
        wells=wells,
        fov_um=(512.0, 512.0),
        fields=(),
        objective=objective,
        estimate_s=estimate_s,
        estimate_without_autofocus_s=estimate_without_autofocus_s,
    )


def test_the_two_waits_the_new_screens_add_are_pinned_as_written() -> None:
    assert REGISTER_FIRST == "register the plate first: step 2, Plate"
    assert LAY_OUT_FIRST == "lay out the fields first: step 3, Wells"


def test_the_objective_line_is_the_one_the_stand_screen_prints() -> None:
    assert (
        objective_line("Nikon 10X S Fluor") == "Objective: Nikon 10X S Fluor, in place"
    )
    assert objective_line(None) == "Objective: not known"
    # The Stand screen's block and the Wells screen's say the objective in one way.
    first = known_text(objective="20x", calibration=None, light_on=None).splitlines()[0]
    assert first == objective_line("20x")


def test_wells_text_names_up_to_three_wells_in_plate_order_then_counts() -> None:
    # The demo visits B11 before B2; the title reads the plate, not the route.
    assert wells_text(["B11", "B2"]) == "B2 and B11"
    assert wells_text(["B2"]) == "B2"
    assert wells_text(["C3", "A1", "B2"]) == "A1, B2 and C3"
    assert wells_text(["A10", "A9", "A2"]) == "A2, A9 and A10"
    # A 1536-well plate's rows go Z, then AA: the length of the letters comes first.
    assert wells_text(["AA1", "Z1"]) == "Z1 and AA1"
    # Four wells or more are counted, whatever they are.
    assert wells_text(["A1", "A2", "A3", "A4"]) == "4 wells"
    assert wells_text([f"{row}{col}" for row in "AB" for col in range(1, 7)]) == (
        "12 wells"
    )
    # A repeated name is one well, and nothing is no wells.
    assert wells_text(["A1", "A1"]) == "A1"
    assert wells_text([]) == "no wells"
    # A name the plate does not print sorts last and is still said.
    assert wells_text(["B2", "?"]) == "B2 and ?"


@given(
    st.lists(
        st.tuples(st.sampled_from("ABCDEFGH"), st.integers(1, 12)),
        min_size=1,
        max_size=3,
        unique=True,
    ),
    st.randoms(use_true_random=False),
)
def test_wells_text_does_not_depend_on_the_order_the_wells_come_in(
    cells: list[tuple[str, int]], rng: Any
) -> None:
    names = [f"{row}{col}" for row, col in cells]
    shuffled = list(names)
    rng.shuffle(shuffled)
    assert wells_text(shuffled) == wells_text(names)


def test_scan_title_names_the_plans_wells_then_the_selection_then_the_static_title() -> (
    None
):
    # The plan is what `scan` does: it names the wells, and the selection changed since
    # does not rename it.
    assert scan_title(["B11", "B2"], ["A1"]) == "Scan B2 and B11"
    assert scan_title(["B2"], []) == "Scan B2"
    assert scan_title([f"A{n}" for n in range(1, 13)], ["A1"]) == "Scan 12 wells"
    # No plan: the selection, in plate order.
    assert scan_title([], ["B11", "B2"]) == "Scan B2 and B11"
    # Neither: the static title, unchanged.
    assert scan_title([], []) == SCREEN_TITLES["scan"] == "Scan the selected wells"


def test_scan_lead_says_the_fields_the_objective_and_the_move_in_flight_a_stop_leaves() -> (
    None
):
    stop = "STOP, at the top right, stops the scan."
    assert scan_lead(_plan(), stop_refused=False) == (
        f"72 fields at Nikon 10X S Fluor. {stop}"
    )
    # The singular, and a plan that does not say its objective says no objective.
    assert scan_lead(_plan(count=1, objective=None), stop_refused=False) == (
        f"1 field. {stop}"
    )
    # A stand whose profile says a role refuses its stop: what STOP does not do.
    assert scan_lead(_plan(), stop_refused=True) == (
        f"72 fields at Nikon 10X S Fluor. {stop} "
        "A move already sent finishes first on this stand."
    )
    # No plan: the static lead, with the same addition.
    assert scan_lead(None, stop_refused=False) == SCREEN_LEADS["scan"] == stop
    assert scan_lead(None, stop_refused=True) == (
        f"{stop} A move already sent finishes first on this stand."
    )


def test_stop_segments_clause_names_each_cut_axis_with_its_size() -> None:
    both = (
        "; a long one ends at the end of its current segment "
        "(at most 5000 µm in XY, 100 µm in Z)"
    )
    assert stop_segments_clause((("xy_stage", 5000.0), ("focus", 100.0))) == both
    # `:g`, as the strip and the CLI write a size: 25 µm, never 25.0 µm.
    assert stop_segments_clause((("focus", 25.0),)) == (
        "; a long one ends at the end of its current segment (at most 25 µm in Z)"
    )
    assert stop_segments_clause((("xy_stage", 500.5),)) == (
        "; a long one ends at the end of its current segment (at most 500.5 µm in XY)"
    )
    # The profile's order is kept, not the order of the axes.
    assert stop_segments_clause((("focus", 100.0), ("xy_stage", 5000.0))) == (
        "; a long one ends at the end of its current segment "
        "(at most 100 µm in Z, 5000 µm in XY)"
    )
    # A role that is not one of the two is skipped, the others stay.
    assert stop_segments_clause((("autofocus_offset", 3.0), ("xy_stage", 5000.0))) == (
        "; a long one ends at the end of its current segment (at most 5000 µm in XY)"
    )
    # Nothing to say: no clause, so the sentence is today's.
    assert stop_segments_clause(()) == ""
    assert stop_segments_clause((("autofocus_offset", 3.0),)) == ""
    assert dict(STOP_SEGMENT_AXES) == {"xy_stage": "XY", "focus": "Z"}


def test_stop_refused_text_names_the_devices_that_refused_and_the_autofocus_switch_off() -> (
    None
):
    # The Ti2-E's three refusals, in the layer's order, and its profile's two cuts.
    said = stop_refused_text(
        ("autofocus_offset PFSOffset", "xy_stage XYStage", "z ZDrive"),
        autofocus="PFS",
        segments_um=(("xy_stage", 500.0), ("focus", 100.0)),
    )
    assert said == (
        "the stand is halted (Resume clears the halt) and the autofocus (PFS) is "
        "switched off. PFSOffset, XYStage and ZDrive refuse a stop on this stand: a "
        "move already sent finishes first; a long one ends at the end of its current "
        "segment (at most 500 µm in XY, 100 µm in Z). Use the stand's own controls to "
        "stop a move that must not finish."
    )
    assert said.startswith(STAND_HALTED)
    assert STAND_HALTED == "the stand is halted (Resume clears the halt)"
    assert not said.startswith("stopped")
    for word in ("resume()", "halted, but", "`", "stop()"):
        assert word not in said


def test_stop_refused_text_agrees_with_the_number_of_devices_and_drops_what_does_not_apply() -> (
    None
):
    # One device: it refuses. The label is what follows the role word.
    one = stop_refused_text(("xy_stage XY",), autofocus=None)
    assert one == (
        "the stand is halted (Resume clears the halt). XY refuses a stop on this "
        "stand: a move already sent finishes first. Use the stand's own controls to "
        "stop a move that must not finish."
    )
    # A motion with no label is named as it is, never as an empty word.
    assert stop_refused_text(("xy_stage",), autofocus=None).startswith(
        "the stand is halted (Resume clears the halt). xy_stage refuses a stop"
    )
    # A label may hold a space: the role word is the first one only.
    assert stop_refused_text(("xy_stage XY Stage",), autofocus=None).startswith(
        "the stand is halted (Resume clears the halt). XY Stage refuses a stop"
    )
    # Two devices: `and`, and the verb follows the number.
    two = stop_refused_text(("xy_stage XY", "z Z"), autofocus="PFS")
    assert "(PFS) is switched off. XY and Z refuse a stop on this stand" in two
    # The autofocus clause is the stand's: no role, no clause (the layer said nothing).
    assert "autofocus" not in stop_refused_text(("z Z",), autofocus=None)
    # A cut the profile makes on a role that did not refuse is the strip's business,
    # not this sentence's: the pairs given are the ones said.
    assert "segment" not in stop_refused_text(
        ("z Z",), autofocus=None, segments_um=(("autofocus_offset", 3.0),)
    )


def test_page_words_says_press_resume_where_the_layer_says_call_resume() -> None:
    assert page_words("the microscope was stopped; call `resume()` to continue") == (
        "the microscope was stopped; press Resume to continue"
    )
    assert page_words("stop the stage if it still moves, then call `resume()`") == (
        "stop the stage if it still moves, then press Resume"
    )
    assert page_words("the microscope was stopped (`Microscope.stop()`); go on") == (
        "the microscope was stopped; go on"
    )
    # Only those phrases: any other text is returned as it is, backticks included.
    for plain in ("the stop failed: cable unplugged", "a `wait()` that never ends", ""):
        assert page_words(plain) == plain


def test_the_layers_own_stop_texts_say_no_api_call_once_the_page_has_said_them() -> (
    None
):
    # The real errors, not copies of their words: a rewording in the layer fails here
    # and not at the stand (FM-254).
    halted = page_words(" ".join(str(MicroscopeHaltedError()).split()))
    assert halted == "the microscope was stopped; press Resume to continue"
    assert "`" not in halted


#: The stop family's call names the page says as its buttons (m5 §17.12.12 D36, D37): the
#: tuple is the one place a next family (the autofocus's calls) is added to the walk.
STOP_FAMILY_CALLS = ("stop", "resume", "wait")

#: A call written ``name()`` after one of the family's names; a device name in backticks is
#: not a call, so the backtick is never part of the pattern.
STOP_FAMILY_CALL = re.compile(r"\b(?:" + "|".join(STOP_FAMILY_CALLS) + r")\(\)")


def _real_motion() -> Motion:
    """A motion the layer tracks that cannot be stopped from smc: the stand's State devices."""
    return Motion(device="XY", name="xy_stage XYStage", is_busy=lambda: True, stop=None)


def _executor() -> Executor:
    return Executor(
        dry_run=False,
        lock=threading.RLock(),
        logger=logging.getLogger("smc.hardware.test"),
    )


def _timeout_error() -> Exception:
    try:
        _executor().wait(_real_motion(), 0.05)
    except DeviceTimeoutError as error:
        return error
    raise AssertionError("the wait did not time out")


def _cannot_be_stopped_error() -> Exception:
    try:
        _executor().stop(_real_motion())
    except HardwareError as error:
        return error
    raise AssertionError("the stop of a motion with no stop did not fail")


_MOTION_ENDING = (
    "wait for it or press STOP; if it never settles, press STOP, check the stand, "
    "then press Resume"
)
_MOTION_ENDING_MANY = (
    "wait for them or press STOP; if they never settle, press STOP, check the stand, "
    "then press Resume"
)


@pytest.mark.parametrize(
    ("make_error", "page_text"),
    [
        pytest.param(
            lambda: MotionInProgressError(("xy_stage XYStage",)),
            f"`xy_stage XYStage` is still moving; {_MOTION_ENDING}",
            id="one device",
        ),
        pytest.param(
            lambda: MotionInProgressError(
                ("xy_stage XYStage", "z ZDrive"), "the stage reports busy"
            ),
            "`xy_stage XYStage`, `z ZDrive` are still moving; "
            f"{_MOTION_ENDING_MANY} (the stage reports busy)",
            id="two devices with a detail",
        ),
        pytest.param(
            _timeout_error,
            "xy_stage XYStage is still busy after 0.05 s; the wait stops nothing, so "
            "it may still be moving. Press STOP if it should not be, or wait longer.",
            id="a wait that timed out",
        ),
        pytest.param(
            _cannot_be_stopped_error,
            "xy_stage XYStage cannot be stopped from smc; wait for it, or check the "
            "stand and press Resume",
            id="a motion that cannot be stopped",
        ),
    ],
)
def test_page_words_says_a_motion_the_stand_still_tracks_as_stop_and_resume(
    make_error: Callable[[], Exception], page_text: str
) -> None:
    # The layer's real errors (FM-254), made one line as the page does (`_one_line`): a
    # rewording in the layer fails here, not at the stand. The whole text is asserted, so
    # that an advice the page cannot follow (Resume while nothing is halted) cannot return.
    text = page_words(" ".join(str(make_error()).split()))
    assert text == page_text
    assert STOP_FAMILY_CALL.search(text) is None, text


def _docstring_constants(tree: ast.AST) -> set[int]:
    """The ``id`` of every docstring ``Constant`` of ``tree``: prose for a reader, never shown."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def _layer_literals(root: Path | None = None) -> Iterator[tuple[str, str]]:
    """Every string a part of ``smc`` that is not the page or the CLI can write, as ``(file, text)`` (m5 §17.12.12 D37).

    The walk reads the syntax tree of each module under ``src/smc`` (``root``) except ``ui/``
    (it holds the table whose keys name the calls) and ``cli.py`` (the CLI's texts are the
    CLI's). A docstring is skipped; an f-string is one text with each formatted value written
    ``it``, which is what the layer's plural texts read as (``wait for it (...)``), and the
    pieces of an f-string are not read a second time on their own. A file is parsed from its
    **bytes**, as the interpreter reads it: an editor that saves a UTF-8 byte-order mark
    (Notepad before Windows 10 1903, FM-80) leaves a module that imports and runs, and a
    text read of it would be a ``SyntaxError`` that names no file.
    """
    root = Path(smc.__file__).parent if root is None else root
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if parts[0] == "ui" or parts == ("cli.py",):
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        skipped = _docstring_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                skipped.update(id(part) for part in node.values)
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield "/".join(parts), node.value
            elif isinstance(node, ast.JoinedStr):
                yield (
                    "/".join(parts),
                    "".join(
                        part.value
                        if isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        else "it"
                        for part in node.values
                    ),
                )


def test_the_walk_reads_a_module_that_starts_with_a_byte_order_mark(
    tmp_path: Path,
) -> None:
    # A module saved with a BOM imports and runs, so the walk must read it too: the verifier
    # of #592's review reproduced a SyntaxError (`<unknown>`) from a text read of such a file.
    package = tmp_path / "layer"
    package.mkdir()
    bom = b"\xef\xbb\xbf"
    (package / "texts.py").write_bytes(
        bom + b'MESSAGE = "wait for it (`wait()`) or stop it (`stop()`)"\n'
    )
    # `ui/` and `cli.py` are not walked, BOM or not.
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "page.py").write_bytes(bom + b'SKIPPED = "stop()"\n')
    (tmp_path / "cli.py").write_bytes(bom + b'SKIPPED = "resume()"\n')
    found = list(_layer_literals(tmp_path))
    assert found == [
        ("layer/texts.py", "wait for it (`wait()`) or stop it (`stop()`)"),
    ]


def test_no_literal_of_the_layer_names_stop_resume_or_wait_after_page_words() -> None:
    # A layer that words a new advice with a call fails here until the table says it (D37).
    left = [
        f"{where}: {text!r}"
        for where, text in _layer_literals()
        if STOP_FAMILY_CALL.search(page_words(text))
    ]
    assert left == []


@pytest.mark.parametrize(
    "sentinel",
    [
        "still moving; wait for it (",
        "stops nothing, so it may still be moving",
        "cannot be stopped from smc; wait for it",
        "it may still move Z: stop() and check the stand, then resume()",
    ],
)
def test_the_walk_finds_the_literals_it_guards(sentinel: str) -> None:
    # FM-45: moving a text in the source must not leave the walk empty and the test above
    # vacuous; each family the table says has a literal here that the walk reads.
    texts = [text for _, text in _layer_literals()]
    assert any(sentinel in text for text in texts), sentinel
    assert any(STOP_FAMILY_CALL.search(text) for text in texts if sentinel in text), (
        sentinel
    )


def test_scan_lead_says_where_a_cut_move_ends_on_a_stand_that_cuts_its_moves() -> None:
    stop = "STOP, at the top right, stops the scan."
    segments = (("xy_stage", 5000.0), ("focus", 100.0))
    sentence = (
        "A move already sent finishes first on this stand; a long one ends at the "
        "end of its current segment (at most 5000 µm in XY, 100 µm in Z)."
    )
    # With a plan: the fields and the objective, then STOP, then the cut.
    assert scan_lead(_plan(), stop_refused=True, stop_segments_um=segments) == (
        f"72 fields at Nikon 10X S Fluor. {stop} {sentence}"
    )
    # Without a plan: the static lead, with the same sentence.
    assert scan_lead(None, stop_refused=True, stop_segments_um=segments) == (
        f"{SCREEN_LEADS['scan']} {sentence}"
    )
    # A stand with no refusal says nothing about a stop it does not refuse, whatever
    # the segments hold: the lead is the stop alone.
    assert scan_lead(None, stop_refused=False, stop_segments_um=segments) == stop
    assert scan_lead(_plan(), stop_refused=False, stop_segments_um=segments) == (
        f"72 fields at Nikon 10X S Fluor. {stop}"
    )
    # A refusal with no segment, or with only a role the page does not name: today's text.
    today = f"{stop} A move already sent finishes first on this stand."
    assert scan_lead(None, stop_refused=True, stop_segments_um=()) == today
    assert (
        scan_lead(
            None, stop_refused=True, stop_segments_um=(("autofocus_offset", 3.0),)
        )
        == today
    )


def test_the_wells_known_lines_say_what_is_laid_out_out_of_date_or_not_known() -> None:
    # A plan with an estimate: four lines, then the plan card's assumptions, in order.
    assert wells_known_lines(
        objective="Nikon 10X S Fluor",
        plan=_plan(),
        out_of_date=None,
        assumed=(
            "no camera calibration for Nikon 10X S Fluor: rotation 0 and no mirror assumed",
        ),
    ) == (
        "Objective: Nikon 10X S Fluor, in place",
        "Fields: 72 in 2 wells",
        "Time: about 42 s",
        "assumed: no camera calibration for Nikon 10X S Fluor: rotation 0 and no mirror "
        "assumed",
    )
    # An estimate of None is an unknown said with what to do, never `0 s` (FM-170).
    unknown = wells_known_lines(
        objective=None, plan=_plan(estimate_s=None), out_of_date=None, assumed=()
    )
    assert unknown == (
        "Objective: not known",
        "Fields: 72 in 2 wells",
        TIME_NOT_KNOWN,
    )
    assert not any("0 s" in line for line in unknown)
    # The singular.
    one = wells_known_lines(
        objective="10x",
        plan=_plan(count=1, wells=1),
        out_of_date=None,
        assumed=(),
    )
    assert one[1] == "Fields: 1 field in 1 well"
    # No plan: not laid out yet, and nothing else is claimed about time or assumptions.
    assert wells_known_lines(
        objective="10x", plan=None, out_of_date=None, assumed=("an old assumption",)
    ) == ("Objective: 10x, in place", "Fields: not laid out yet")
    # A plan the document no longer holds usable says why, in the checklist's words.
    assert wells_known_lines(
        objective="10x",
        plan=None,
        out_of_date="scan-plan read its inputs from an earlier run; run it again",
        assumed=(),
    )[1] == (
        "Fields: out of date (scan-plan read its inputs from an earlier run; run it "
        "again)"
    )


LOCKED_EACH_WELL = "with the autofocus locked again in each well"


@pytest.mark.parametrize(
    ("autofocus", "expected"),
    [
        ("off", "Time: about 1 min 28 s, with the autofocus not used"),
        # The text is read stripped: what the operator typed around the word is not the mode.
        (" off ", "Time: about 1 min 28 s, with the autofocus not used"),
        ("required", f"Time: about 1 min 30 s, {LOCKED_EACH_WELL}"),
        (
            "if-engaged",
            f"Time: about 1 min 30 s, {LOCKED_EACH_WELL}; "
            "about 1 min 28 s if it is off at the start",
        ),
        # The empty field is the tool's default, `if-engaged`; a mistyped text reads as
        # it too, and `scan` refuses it with its own words when it runs.
        (
            "",
            f"Time: about 1 min 30 s, {LOCKED_EACH_WELL}; "
            "about 1 min 28 s if it is off at the start",
        ),
        (
            "banana",
            f"Time: about 1 min 30 s, {LOCKED_EACH_WELL}; "
            "about 1 min 28 s if it is off at the start",
        ),
    ],
)
def test_the_time_line_follows_the_autofocus_mode(
    autofocus: str, expected: str
) -> None:
    plan = _plan(estimate_s=90.4, estimate_without_autofocus_s=88.4)
    assert time_line(plan, autofocus) == expected
    # The Wells screen's block carries the same line, in its place after the fields.
    lines = wells_known_lines(
        objective="Nikon 10X S Fluor",
        plan=plan,
        out_of_date=None,
        assumed=(),
        autofocus=autofocus,
    )
    assert lines == (
        "Objective: Nikon 10X S Fluor, in place",
        "Fields: 72 in 2 wells",
        expected,
    )


@pytest.mark.parametrize("autofocus", ["off", "required", "if-engaged", "", "banana"])
def test_the_time_line_is_the_plain_figure_when_there_is_nothing_to_say_about_the_autofocus(
    autofocus: str,
) -> None:
    # An older plan has no second figure; a plan with none to leave out has two equal ones.
    for plan in (
        _plan(estimate_s=42.16),
        _plan(estimate_s=42.16, estimate_without_autofocus_s=42.16),
        # A hand-edited plan whose second figure is above the first is not a time that
        # gets longer with the autofocus off: the plain line (the model refuses it, the
        # layer is built from a record that may not have been validated first).
        _plan(estimate_s=42.16, estimate_without_autofocus_s=99.0),
    ):
        assert time_line(plan, autofocus) == "Time: about 42 s"
    # Unknown stays unknown: never `0 s` (FM-170).
    unknown = _plan(estimate_s=None)
    assert time_line(unknown, autofocus) == TIME_NOT_KNOWN


@pytest.mark.parametrize("autofocus", ["off", "required", "if-engaged", ""])
def test_a_measured_zero_is_a_time_and_not_an_unknown(autofocus: str) -> None:
    # FM-170 both ways: 0 is a figure (a plan of nothing, or a stand whose scan without
    # the autofocus costs nothing), and only `None` is not known.
    assert time_line(_plan(estimate_s=0.0), autofocus) == "Time: about 0 s"
    assert (
        time_line(_plan(estimate_s=0.0, estimate_without_autofocus_s=0.0), autofocus)
        == "Time: about 0 s"
    )
    # A second figure of 0 under a first of 2 s is a lock counted, not an absent figure.
    plan = _plan(estimate_s=2.0, estimate_without_autofocus_s=0.0)
    expected = {
        "off": "Time: about 0 s, with the autofocus not used",
        "required": f"Time: about 2 s, {LOCKED_EACH_WELL}",
        "if-engaged": (
            f"Time: about 2 s, {LOCKED_EACH_WELL}; about 0 s if it is off at the start"
        ),
        "": (
            f"Time: about 2 s, {LOCKED_EACH_WELL}; about 0 s if it is off at the start"
        ),
    }
    assert time_line(plan, autofocus) == expected[autofocus]


def test_the_if_engaged_clause_is_left_out_when_the_two_durations_read_the_same() -> (
    None
):
    # 42.4 and 41.6 both read `42 s`: saying `about 42 s if it is off` would say nothing.
    plan = _plan(estimate_s=42.4, estimate_without_autofocus_s=41.6)
    assert time_line(plan, "if-engaged") == f"Time: about 42 s, {LOCKED_EACH_WELL}"
    assert time_line(plan, "") == f"Time: about 42 s, {LOCKED_EACH_WELL}"
    # `off` and `required` say their own figure, whatever the other reads.
    assert time_line(plan, "off") == "Time: about 42 s, with the autofocus not used"
    assert time_line(plan, "required") == f"Time: about 42 s, {LOCKED_EACH_WELL}"


@pytest.mark.parametrize(
    ("plan", "autofocus", "line"),
    [
        # Every form of today's line takes the tail, `Time: not known` included.
        (_plan(estimate_s=None), "", TIME_NOT_KNOWN),
        (_plan(estimate_s=42.16), "off", "Time: about 42 s"),
        (
            _plan(estimate_s=90.4, estimate_without_autofocus_s=88.4),
            "off",
            "Time: about 1 min 28 s, with the autofocus not used",
        ),
        (
            _plan(estimate_s=90.4, estimate_without_autofocus_s=88.4),
            "required",
            f"Time: about 1 min 30 s, {LOCKED_EACH_WELL}",
        ),
        (
            _plan(estimate_s=90.4, estimate_without_autofocus_s=88.4),
            "",
            f"Time: about 1 min 30 s, {LOCKED_EACH_WELL}; "
            "about 1 min 28 s if it is off at the start",
        ),
    ],
)
def test_the_time_line_says_the_focus_on_the_cells_is_not_counted(
    plan: PlanLayer, autofocus: str, line: str
) -> None:
    for mode in ("map", "per-field", "per-well"):
        assert time_line(plan, autofocus, mode) == (
            f"{line}; the focus on the cells (cell_focus={mode}) is not counted"
        )
    # The text is read stripped, and what the operator typed is what is named (the tool
    # refuses a mistyped value with its own words when it runs).
    assert time_line(plan, autofocus, " map ") == (
        f"{line}; the focus on the cells (cell_focus=map) is not counted"
    )
    # `off`, the empty field and no argument add nothing.
    for mode in ("off", " off ", ""):
        assert time_line(plan, autofocus, mode) == line
    assert time_line(plan, autofocus) == line
    # The Wells screen's block carries the same line, in its place after the fields.
    lines = wells_known_lines(
        objective="Nikon 10X S Fluor",
        plan=plan,
        out_of_date=None,
        assumed=("the offset between the objectives is not measured",),
        autofocus=autofocus,
        cell_focus="map",
    )
    assert lines[2] == (
        f"{line}; the focus on the cells (cell_focus=map) is not counted"
    )
    assert lines[3] == "assumed: the offset between the objectives is not measured"


def test_the_focus_texts_are_pinned_as_written() -> None:
    assert CELL_FOCUS_WORDS == "How the scan focuses on the cells"
    assert FOCUS_MAP_NONE == (
        "No focus map kept yet: scan the wells with the autofocus locked first; "
        "the page keeps that scan as the map."
    )
    assert SCAN_MAP_RESUME_NOTE == (
        "Continue the scan keeps the focus this scan started with: focusing again now "
        "changes nothing. Start again takes the focus as it is now."
    )


ANCHOR_AT_B2 = (
    "Before you start: with Live, focus by eye on the cells near the centre of B2. "
    "That focus is the scan's starting point; each well is then anchored near its centre."
)


def _kept(**changes: Any) -> KeptMap:
    kept = KeptMap(
        file="run.focus-map.json",
        wells=("B2",),
        objective="Nikon 10X S Fluor",
        scanned_utc=datetime(2026, 11, 26, 9, 5, 41, tzinfo=timezone.utc),
        problem=None,
    )
    return dataclasses.replace(kept, **changes)


def test_focus_map_lines_name_the_kept_scan_its_wells_objective_and_time() -> None:
    lines = focus_map_lines(
        cell_focus="map", typed_map="", kept=_kept(), first_well="B2"
    )
    assert lines == (
        "Focus map: the scan of B2 at Nikon 10X S Fluor, 2026-11-26 09:05 UTC, "
        "kept by the page in run.focus-map.json.",
        ANCHOR_AT_B2,
    )
    # Wells as every screen says them, in plate order, counted from four; the time is UTC
    # whatever zone the record carries.
    many = focus_map_lines(
        cell_focus="map",
        typed_map="",
        kept=_kept(
            wells=("B11", "B2"),
            scanned_utc=datetime(
                2026, 11, 26, 11, 5, tzinfo=timezone(timedelta(hours=2))
            ),
        ),
        first_well="B11",
    )
    assert many[0] == (
        "Focus map: the scan of B2 and B11 at Nikon 10X S Fluor, 2026-11-26 09:05 UTC, "
        "kept by the page in run.focus-map.json."
    )
    assert (
        "4 wells at"
        in focus_map_lines(
            cell_focus="map",
            typed_map="",
            kept=_kept(wells=("A1", "A2", "A3", "A4")),
            first_well="A1",
        )[0]
    )
    # What the copy does not say is said as such, never invented.
    bare = focus_map_lines(
        cell_focus="map",
        typed_map="",
        kept=_kept(objective=None, scanned_utc=None, wells=()),
        first_well=None,
    )
    assert bare == (
        "Focus map: the scan at objective not recorded, kept by the page in "
        "run.focus-map.json.",
        "Before you start: with Live, focus by eye on the cells near the centre of the "
        "first well. That focus is the scan's starting point; each well is then "
        "anchored near its centre.",
    )


def test_focus_map_lines_say_none_kept_and_how_to_get_one() -> None:
    lines = focus_map_lines(cell_focus="map", typed_map="", kept=None, first_well="B2")
    assert lines == (FOCUS_MAP_NONE, ANCHOR_AT_B2)
    # A blank field is no field: the tool's default is none.
    assert (
        focus_map_lines(cell_focus="map", typed_map="  ", kept=None, first_well="B2")
        == lines
    )


def test_focus_map_lines_name_a_typed_map_as_typed_on_this_page() -> None:
    # A typed `focus_map` wins over the kept file, and is said as the operator typed it.
    for kept in (None, _kept(), _kept(problem="it is not a run document")):
        assert focus_map_lines(
            cell_focus="map", typed_map=" D:/runs/ten.json ", kept=kept, first_well="B2"
        ) == ("Focus map: D:/runs/ten.json, typed on this page.", ANCHOR_AT_B2)
    # `null` reads as none, as the parser reads it: the kept file is what the click gives.
    assert (
        focus_map_lines(cell_focus="map", typed_map="null", kept=None, first_well="B2")[
            0
        ]
        == FOCUS_MAP_NONE
    )


def test_focus_map_lines_name_a_kept_file_that_cannot_be_read() -> None:
    kept = _kept(
        wells=(),
        objective=None,
        scanned_utc=None,
        problem="run.focus-map.json is not valid JSON",
    )
    assert focus_map_lines(
        cell_focus="map", typed_map="", kept=kept, first_well="B2"
    ) == (
        "The kept focus map run.focus-map.json cannot be read "
        "(run.focus-map.json is not valid JSON).",
        ANCHOR_AT_B2,
    )


@pytest.mark.parametrize("cell_focus", ["", "off", "per-field", "per-well", "banana"])
def test_focus_map_lines_are_empty_unless_cell_focus_is_map(cell_focus: str) -> None:
    assert (
        focus_map_lines(
            cell_focus=cell_focus, typed_map="x.json", kept=_kept(), first_well="B2"
        )
        == ()
    )
    # The field is read stripped.
    assert focus_map_lines(cell_focus=" map ", typed_map="", kept=None, first_well="B2")


def test_focus_map_file_is_the_stem_dot_focus_map_json_beside_the_run_document(
    tmp_path: Path,
) -> None:
    assert focus_map_file(tmp_path / "run.json") == tmp_path / "run.focus-map.json"
    assert (
        focus_map_file(tmp_path / "session 6" / "B2 scan.json")
        == tmp_path / "session 6" / "B2 scan.focus-map.json"
    )
    # The same folder as the run document, so the run folder stays the one place.
    assert focus_map_file(Path("run.json")) == Path("run.focus-map.json")


def test_plan_selection_note_is_empty_when_the_wells_agree_and_names_both_otherwise() -> (
    None
):
    # The same wells in another order, or the plan's visiting order: no difference.
    assert plan_selection_note(["B11", "B2"], ["B2", "B11"]) == ""
    # No plan, nothing to compare.
    assert plan_selection_note([], ["A1"]) == ""
    assert plan_selection_note(["B2", "B11"], ["B2"]) == (
        "The fields laid out are for B2 and B11; the wells selected now are B2. "
        "Lay out the fields again to scan these."
    )
    assert plan_selection_note(["B2"], []) == (
        "The fields laid out are for B2; no well is selected now."
    )
    # A selection that adds a well to the plan's is a difference too.
    assert plan_selection_note(["B2"], ["B2", "C3"]).startswith(
        "The fields laid out are for B2; the wells selected now are B2 and C3."
    )


def test_field_source_line_says_default_preset_suggestion_or_typed_in_that_order() -> (
    None
):
    def say(
        value: str,
        *,
        preset: str | None = "weekly",
        preset_value: str | None = "0.4",
        suggested: str | None = "0.4",
        default: str = "'if-engaged'",
    ) -> str:
        return field_source_line(
            value=value,
            default=default,
            preset=preset,
            preset_value=preset_value,
            suggested=suggested,
        )

    # An empty field is the tool's default, shown without a string's quotes.
    assert say("") == "the tool's default: if-engaged"
    assert say("  ") == "the tool's default: if-engaged"
    assert say("", default="1.0") == "the tool's default: 1.0"
    # The preset wins over the suggestion when both give the value.
    assert say("0.4") == "from the preset weekly"
    assert say("0.4", preset=None) == "from the profile's suggestions"
    assert say("0.4", preset="weekly", preset_value="0.5") == (
        "from the profile's suggestions"
    )
    assert say("0.4", preset=None, suggested=None) == "typed on this page"
    assert say("0.7") == "typed on this page"
    # A preset that does not set the field names nothing of it.
    assert say("0.4", preset="weekly", preset_value=None, suggested=None) == (
        "typed on this page"
    )


def test_wells_from_line_says_none_preset_or_page() -> None:
    assert wells_from_line([], "weekly", ["B2"]) == "no well chosen yet"
    assert wells_from_line([], None, None) == "no well chosen yet"
    assert wells_from_line(["B2", "B11"], "weekly", ["B2", "B11"]) == (
        "from the preset weekly"
    )
    # The box no longer holds what the preset selected, or the preset selected nothing.
    assert wells_from_line(["B2"], "weekly", ["B2", "B11"]) == "chosen on this page"
    assert wells_from_line(["B2"], "weekly", None) == "chosen on this page"
    assert wells_from_line(["B2"], None, None) == "chosen on this page"
    # A preset that is not loaded names nothing, whatever a stale selection says.
    assert wells_from_line(["B2"], None, ["B2"]) == "chosen on this page"


def test_lay_out_wait_orders_the_stand_the_plate_the_registration_the_tool_the_selection() -> (
    None
):
    def wait(
        *,
        connected: bool = True,
        plate_chosen: bool = True,
        registered: bool = True,
        blocked: str | None = None,
        selected: int = 2,
    ) -> str | None:
        return lay_out_wait(
            connected=connected,
            plate_chosen=plate_chosen,
            registered=registered,
            blocked=blocked,
            selected=selected,
        )

    assert wait() is None
    # Every wait at once, taken off one by one: the first of them is said, in the order of
    # the design. The tool's own reason is the controller's, and its words are not the guide's.
    reason = "scan-plan: a reason of the tool"
    assert (
        wait(
            connected=False,
            plate_chosen=False,
            registered=False,
            blocked=NOT_CONNECTED_TEXT,
            selected=0,
        )
        == NOT_CONNECTED_TEXT
    )
    assert (
        wait(
            plate_chosen=False,
            registered=False,
            blocked=PLATE_NOT_CHOSEN,
            selected=0,
        )
        == PLATE_NOT_CHOSEN
    )
    assert wait(registered=False, blocked=reason, selected=0) == REGISTER_FIRST
    assert wait(blocked=reason, selected=0) == reason
    assert wait(blocked=None, selected=0) == SELECT_WELLS
    assert wait(selected=1) is None


def test_scan_wait_orders_the_stand_the_plan_and_the_tools_reason() -> None:
    assert scan_wait(connected=True, blocked=None, plan_usable=True) is None
    assert scan_wait(connected=False, blocked="not connected", plan_usable=False) == (
        "not connected"
    )
    assert scan_wait(
        connected=True, blocked="scan: a checkpoint", plan_usable=False
    ) == (LAY_OUT_FIRST)
    assert scan_wait(
        connected=True, blocked="scan: a checkpoint", plan_usable=True
    ) == ("scan: a checkpoint")


def test_scan_progress_lines_estimate_from_the_first_report_and_not_from_the_click() -> (
    None
):
    # First report at 30.0 s with 1 field done, the latest at 36.0 s with 4 done, 40.0 s
    # since the click, 10 fields: 6 left at 2 s a field is 12 s. A rate from the click
    # (40 s for 4 fields) gives 1 min 00 s: the start-up of a scan is not its pace.
    clock = RunClock(elapsed_s=40.0, first=(30.0, 1), last=(36.0, 4))
    assert scan_progress_lines(4, 10, clock, running=True) == (
        "Field 4 of 10",
        "Elapsed 40 s",
        "Left about 12 s (estimated from the fields so far)",
    )


def test_scan_progress_lines_say_the_time_is_not_known_when_the_plan_has_no_timing() -> (
    None
):
    # The Ti2-E rehearsal (#514 point 1): fields of one well come at 0.19 s, so 9 left said
    # `Left about 1 s` for the 8 s the stage took to the next well. A plan with no timing
    # has no figure for that travel (the Wells screen says `Time: not known`), so *Left* says
    # it is not known and gives no number, whatever the clock holds.
    clock = RunClock(elapsed_s=12.0, first=(4.0, 1), last=(6.0, 9))
    assert scan_progress_lines(9, 18, clock, running=True, timing_known=True)[-1] == (
        "Left about 2 s (estimated from the fields so far)"
    )
    untimed = scan_progress_lines(9, 18, clock, running=True, timing_known=False)
    assert untimed == (
        "Field 9 of 18",
        "Elapsed 12 s",
        "Left: not known (no timing measured yet)",
    )
    assert all("about" not in line for line in untimed)
    # A run that ended, and a scan whose last field is done, draw no *Left* at all.
    assert scan_progress_lines(9, 18, clock, running=False, timing_known=False) == (
        "Field 9 of 18",
        "Elapsed 12 s",
    )
    assert scan_progress_lines(18, 18, clock, running=True, timing_known=False) == (
        "Field 18 of 18",
        "Elapsed 12 s",
    )
    # The timing is the default's assumption: the unknown is said only when it is passed.
    assert scan_progress_lines(9, 18, clock, running=True) == scan_progress_lines(
        9, 18, clock, running=True, timing_known=True
    )


def test_scan_progress_lines_leave_the_time_unknown_with_one_report_or_a_clock_that_did_not_move() -> (
    None
):
    unknown = "Left: not known yet (it needs two fields)"
    # No report yet, and no clock at all.
    assert scan_progress_lines(0, 72, RunClock(1.0, None, None), running=True)[-1] == (
        unknown
    )
    assert scan_progress_lines(0, 72, None, running=True) == ("Field 0 of 72", unknown)
    # One report: the first is the latest.
    one = RunClock(5.0, (4.65, 1), (4.65, 1))
    assert scan_progress_lines(1, 72, one, running=True)[-1] == unknown
    # Two reports with one stamp, as a 15.625 ms clock gives (FM-197): no division by zero.
    same_stamp = RunClock(5.0, (4.65, 1), (4.65, 2))
    assert scan_progress_lines(2, 72, same_stamp, running=True)[-1] == unknown
    # Two reports of one field (a retry) have no pace either.
    same_field = RunClock(5.0, (4.65, 3), (5.0, 3))
    assert scan_progress_lines(3, 72, same_field, running=True)[-1] == unknown


def test_scan_progress_lines_draw_no_left_once_the_last_field_is_done_or_the_run_ended() -> (
    None
):
    clock = RunClock(elapsed_s=23.0, first=(4.65, 1), last=(18.08, 72))
    # The last field is done: nothing is left to estimate.
    assert scan_progress_lines(72, 72, clock, running=True) == (
        "Field 72 of 72",
        "Elapsed 23 s",
    )
    # A run that ended (or stopped) at 30 of 72 left nothing to wait for either.
    stopped = RunClock(elapsed_s=9.0, first=(4.0, 1), last=(8.0, 30))
    assert scan_progress_lines(30, 72, stopped, running=False) == (
        "Field 30 of 72",
        "Elapsed 9 s",
    )
    # Without a clock there is no Elapsed.
    assert scan_progress_lines(30, 72, None, running=False) == ("Field 30 of 72",)


@given(
    done=st.integers(0, 200),
    total=st.integers(0, 200),
    first_t=st.floats(0.0, 1000.0),
    last_t=st.floats(0.0, 1000.0),
    first_done=st.integers(0, 200),
    last_done=st.integers(0, 200),
    elapsed=st.floats(0.0, 2000.0),
    running=st.booleans(),
)
def test_scan_progress_lines_never_raise_and_never_say_a_negative_time(
    done: int,
    total: int,
    first_t: float,
    last_t: float,
    first_done: int,
    last_done: int,
    elapsed: float,
    running: bool,
) -> None:
    clock = RunClock(elapsed, (first_t, first_done), (last_t, last_done))
    lines = scan_progress_lines(done, total, clock, running=running)
    assert lines[0] == f"Field {done} of {total}"
    assert all("-" not in line for line in lines)
    assert sum(line.startswith("Left") for line in lines) <= 1


def test_scanned_line_counts_frames_and_says_a_dry_run_took_none() -> None:
    assert scanned_line(ScanSummary(frames=72, failed=0, dry_run=False)) == (
        "Scanned: 72 frames, 0 failed"
    )
    assert scanned_line(ScanSummary(frames=1, failed=2, dry_run=False)) == (
        "Scanned: 1 frame, 2 failed"
    )
    # A dry run walks the fields and takes no frame: it is not a scan that found nothing.
    assert scanned_line(ScanSummary(frames=0, failed=0, dry_run=True)) == (
        "Scanned in a dry run: no frame was taken"
    )


def test_plain_default_drops_one_pair_of_quotes_only() -> None:
    assert plain_default("'if-engaged'") == "if-engaged"
    assert plain_default('"if-engaged"') == "if-engaged"
    # One pair, not all of them; the inner ones are the operator's text.
    assert plain_default("''x''") == "'x'"
    # Not a pair: mismatched, one-sided, a lone quote, nothing to drop.
    assert plain_default("'x\"") == "'x\""
    assert plain_default("'x") == "'x"
    assert plain_default("'") == "'"
    assert plain_default("1.0") == "1.0"
    assert plain_default("") == ""
    assert plain_default("None") == "None"
    # A default whose first and last characters are the same and are not a quote: four
    # real tools have one (`'0.0'` as a repr), and a check that tested only "same ends" would
    # strip a `.` from each.
    for default in ("0.0", "1e1", "1x1", "-5-", "11"):
        assert plain_default(default) == default
    assert plain_default("''") == ""


def test_an_empty_field_is_the_default_before_it_is_the_presets() -> None:
    # The order of the lines: a field that is empty says the tool's default even when the
    # preset's value for it is empty too (a preset that clears a field), and never `from the
    # preset`.
    assert field_source_line(
        value="", default="1.0", preset="weekly", preset_value="", suggested=""
    ) == ("the tool's default: 1.0")
    assert field_source_line(
        value="", default="'off'", preset=None, preset_value=None, suggested=""
    ) == ("the tool's default: off")


def test_a_well_name_with_absurd_digits_is_sorted_last_and_never_raises() -> None:
    # FM-125: a name from a record, as it comes. Python refuses `int()` of more than 4300
    # digits, and the texts are drawn by the tick before the strip is: one such name would
    # stop the strip, the problems line and the buttons every second.
    huge = "A" + "9" * 5000
    assert wells_text([huge, "B2"]) == f"B2 and {huge}"
    assert wells_text([huge]) == huge
    assert scan_title([huge], []) == f"Scan {huge}"
    assert scan_title([], [huge, "A1"]) == f"Scan A1 and {huge}"
    assert plan_selection_note([huge], ["A1"]).startswith(
        f"The fields laid out are for {huge}; "
    )


# -- the Results screen (m5 §17.4.5, §17.11, #446) --------------------------------------

_PLUGIN_OF = {
    "cells": "smc.plugins.cells.plugin:CellsPlugin",
    "density": "smc.plugins.density.plugin:DensityPlugin",
    "route": "smc.plugins.route.plugin:RoutePlugin",
    "report": "smc.plugins.report.plugin:ReportPlugin",
    "nis-export": "smc.plugins.nis_export.plugin:NisExportPlugin",
}


def _real_plugin(tool: str) -> Any:
    module, _, name = _PLUGIN_OF[tool].partition(":")
    return getattr(importlib.import_module(module), name)


def _results(
    reasons: dict[str, str] | None = None, **states: str
) -> tuple[CheckItem, ...]:
    """The checklist with ``states`` by slot (``scan="ok"``), and the reason of each ``unusable`` one."""
    checklist = _checklist(**states)
    given = {slot.replace("-", "_"): why for slot, why in (reasons or {}).items()}
    return tuple(
        dataclasses.replace(item, reason=given[item.slot.replace("-", "_")])
        if item.slot.replace("-", "_") in given
        else item
        for item in checklist
    )


def _result_card(**changes: Any) -> ResultCard:
    base: dict[str, Any] = {
        "plugin": "cells",
        "status": "done",
        "when_utc": "2026-11-12 09:30:00",
        "profile": "demo",
        "dry_run": False,
        "lines": (),
        "assumed": (),
        "error": None,
    }
    return ResultCard(**{**base, **changes})


def _row(
    plugin: str,
    checklist: Sequence[CheckItem],
    *,
    card: ResultCard | None = None,
    blocked: str | None = None,
    dry_scan: bool = False,
    running: str | None = None,
    progress: tuple[int, int | None] | None = None,
) -> ResultRow:
    return result_row(
        plugin,
        checklist=checklist,
        card=card,
        blocked=blocked,
        dry_scan=dry_scan,
        running=running,
        progress=progress,
    )


def test_the_results_tools_are_the_checklists_in_its_order_and_each_has_a_plain_name() -> (
    None
):
    slots = [slot for slot, _optional in _CHECKLIST]
    assert [slot for slot in slots if slot in RESULT_TOOLS] == list(RESULT_TOOLS)
    assert set(RESULT_NAMES) == set(RESULT_TOOLS) == set(RESULT_READS)
    # §17.4.5's words, and the screen they belong to (§17.5).
    assert RESULT_NAMES == {
        "cells": "Count the cells",
        "density": "Density per well",
        "route": "Route through the densest fields",
        "report": "Report",
        "nis-export": "Export for NIS-Elements",
    }
    assert {STEP_OF_SLOT[tool] for tool in RESULT_TOOLS} == {"results"}
    assert RUN_ALL_LABEL == "Run all that are left"


def test_the_results_reads_are_the_plugins_reads() -> None:
    # The guide copies what the plugins read, so that it stays pure (§17.11.2): a plugin that
    # reads one more slot must change this table, or a row would name the wrong input.
    for tool in RESULT_TOOLS:
        plugin = _real_plugin(tool)
        assert plugin.name == plugin.provides == tool
        assert RESULT_READS[tool] == tuple(plugin.reads), tool
        # The optional reads make a tool out of date too (`report` reads the route when
        # there is one): `again_note` walks them, so they are pinned like the others.
        assert RESULT_OPTIONAL_READS[tool] == tuple(plugin.reads_optional), tool
    assert set(RESULT_OPTIONAL_READS) == set(RESULT_TOOLS)


def test_the_results_choices_are_fields_of_the_tools_parameters() -> None:
    for tool, keys in RESULT_CHOICES.items():
        fields = _real_plugin(tool).Params.model_fields
        for key in keys:
            assert key in fields, (tool, key)


def test_results_left_is_the_slots_that_are_not_ok_in_checklist_order() -> None:
    checklist = _results(
        {"route": "route read its inputs from an earlier run; run it again"},
        plate="ok",
        scan_plan="ok",
        scan="ok",
        cells="ok",
        route="unusable",
        report="ok",
    )
    # `not run` and `unusable` are both left; `ok` never is, whatever comes after it.
    assert results_left(checklist) == ("density", "route", "nis-export")
    assert (
        results_left(
            _results(cells="ok", density="ok", route="ok", report="ok", nis_export="ok")
        )
        == ()
    )
    # The slots of the other screens are never a Results tool.
    assert results_left(_results()) == RESULT_TOOLS
    assert results_left(()) == ()


def test_a_done_row_says_when_and_offers_again_and_an_unusable_one_says_why_and_offers_run() -> (
    None
):
    checklist = _results(
        {"route": "route read its inputs from an earlier run; run it again"},
        scan_plan="ok",
        scan="ok",
        cells="ok",
        route="unusable",
    )
    done = _row("cells", checklist, card=_result_card())
    assert (done.state, done.button, done.done) == (
        "Done 2026-11-12 09:30:00 UTC",
        "Again",
        True,
    )
    assert done.name == "Count the cells"
    assert not done.missing
    dry = _row("cells", checklist, card=_result_card(dry_run=True))
    assert dry.state == "Done 2026-11-12 09:30:00 UTC (dry run)"
    # A done tool whose card the view could not make still says it is done.
    assert _row("cells", checklist, card=None).state == "Done"

    unusable = _row("route", checklist)
    assert unusable.state == (
        "Not usable: route read its inputs from an earlier run; run it again"
    )
    assert (unusable.button, unusable.done, unusable.missing) == ("Run", False, False)
    # An unusable result with no reason, still a state of its own.
    bare = tuple(
        dataclasses.replace(item, reason=None) if item.slot == "route" else item
        for item in checklist
    )
    assert _row("route", bare).state == "Not usable"


def test_a_row_whose_input_is_missing_says_after_the_tool_that_makes_it() -> None:
    nothing = _results()
    assert _row("cells", nothing).state == SCAN_FIRST
    assert SCAN_FIRST == "scan first: step 4, Scan"
    # `density` reads the plan first, then the scan, then the cells.
    assert _row("density", nothing).state == LAY_OUT_FIRST
    assert _row("density", _results(scan_plan="ok")).state == SCAN_FIRST
    assert (
        _row("density", _results(scan_plan="ok", scan="ok")).state
        == "after Count the cells"
    )
    # `route` waits for `cells` and not for `density`; `report` waits for `density`.
    assert (
        _row("route", _results(scan_plan="ok", scan="ok", cells="ok")).state == "Ready"
    )
    assert (
        _row("report", _results(scan_plan="ok", scan="ok", cells="ok")).state
        == "after Density per well"
    )
    assert _row("nis-export", _results(cells="ok")).state == (
        "after Route through the densest fields"
    )
    # An unusable input is as missing as an absent one.
    unusable = _results({"cells": "x"}, scan_plan="ok", scan="ok", cells="unusable")
    assert _row("density", unusable).state == "after Count the cells"
    assert _row("density", unusable).missing
    # The document's own words, for a terminal, are nowhere in a missing row.
    for tool in RESULT_TOOLS:
        assert "run document" not in _row(tool, nothing).state
        assert "smc run" not in _row(tool, nothing).state
    # A slot that is no tool of the screen is named as it is.
    assert _row("cells", ()).state == SCAN_FIRST


def test_a_wait_is_the_stands_reason_then_the_dry_runs_and_never_both_with_a_missing_input() -> (
    None
):
    ready = _results(scan_plan="ok", scan="ok", cells="ok", density="ok")
    stand = "the stand is not connected"
    # The controller's reason first, over the dry run's.
    assert _row("route", ready, blocked=stand, dry_scan=True).wait == stand
    assert _row("route", ready, dry_scan=True).wait == DRY_RUN_NO_FRAME
    assert _row("route", ready).wait == ""
    # The tool that reads no scan has nothing to hold back (`nis-export` reads the route).
    assert _row("nis-export", _results(route="ok"), dry_scan=True).wait == ""
    # A row that waits for an input says it in its state line and adds nothing.
    missing = _row(
        "density", _results(scan_plan="ok", scan="ok"), blocked=stand, dry_scan=True
    )
    assert missing.missing
    assert missing.wait == ""


def test_a_running_row_says_running_now_with_its_count_and_without_a_total() -> None:
    ready = _results(scan_plan="ok", scan="ok", cells="ok")
    assert _row("route", ready, running="route").state == "Running now"
    assert _row("route", ready, running="route", progress=(3, 12)).state == (
        "Running now: 3 of 12"
    )
    assert _row("route", ready, running="route", progress=(3, None)).state == (
        "Running now: 3"
    )
    # Running wins over done: *Again* is running now, and the row says so.
    done = _results(scan_plan="ok", scan="ok", cells="ok", route="ok")
    assert (
        _row("route", done, running="route", card=_result_card()).state == "Running now"
    )
    # Another tool's run is not this row's.
    assert _row("route", ready, running="cells", progress=(3, 12)).state == "Ready"


def test_run_all_waits_for_the_first_tool_left_and_says_nothing_is_left() -> None:
    assert run_all_wait([], ()) == NOTHING_LEFT
    assert NOTHING_LEFT == "Nothing is left: every step is done."

    nothing = _results()
    rows = [_row(tool, nothing) for tool in RESULT_TOOLS]
    # The first tool left needs the scan: its state is the reason, never its (empty) wait.
    assert run_all_wait(rows, RESULT_TOOLS) == SCAN_FIRST

    ready = _results(scan_plan="ok", scan="ok")
    rows = [_row(tool, ready, blocked="stand closed") for tool in RESULT_TOOLS]
    assert run_all_wait(rows, RESULT_TOOLS) == "stand closed"
    rows = [_row(tool, ready, dry_scan=True) for tool in RESULT_TOOLS]
    assert run_all_wait(rows, RESULT_TOOLS) == DRY_RUN_NO_FRAME

    assert (
        run_all_wait([_row(tool, ready) for tool in RESULT_TOOLS], RESULT_TOOLS) is None
    )
    # Only the first tool left counts: `density` waits for `cells`, which the chain runs.
    assert run_all_wait(
        [_row(tool, ready) for tool in RESULT_TOOLS], RESULT_TOOLS[1:]
    ) == ("after Count the cells")
    # A name that has no row is not a wait.
    assert run_all_wait([], ("their-tool",)) is None


def test_again_names_the_later_steps_that_are_done_and_is_empty_for_the_last() -> None:
    everything = _results(
        scan_plan="ok",
        scan="ok",
        cells="ok",
        density="ok",
        route="ok",
        report="ok",
        nis_export="ok",
    )
    note = again_note("cells", everything)
    assert note == (
        "Again makes Density, Route, Report, Export out of date; Run all that are left "
        "runs them again."
    )
    # What reads it, and no more: the route does not read the density, and the export reads
    # the route alone, so each of these names what really becomes out of date.
    assert again_note("density", everything) == AGAIN_NOTE.format(names="Report")
    # A new route makes the export out of date, and the report too: it reads the route when
    # there is one (`reads_optional`), which the checklist shows after *Again* on the route.
    assert again_note("route", everything) == AGAIN_NOTE.format(names="Report, Export")
    assert again_note("report", everything) == ""
    assert again_note("nis-export", everything) == ""
    # Only the tools that are done are named, in the tools' order.
    partly = _results(scan_plan="ok", scan="ok", cells="ok", route="ok")
    assert again_note("cells", partly) == AGAIN_NOTE.format(names="Route")
    # A tool that is not done has nothing to replace; neither has a name no table has.
    assert again_note("cells", _results()) == ""
    assert again_note("their-tool", everything) == ""


def _chain(
    ended: Sequence[tuple[str, str, str]],
    current: str | None = None,
    plugins: tuple[str, ...] = RESULT_TOOLS,
) -> ChainStatus:
    return ChainStatus(
        plugins=plugins,
        ended=tuple(ChainEnd(plugin, how, outcome) for plugin, how, outcome in ended),  # type: ignore[arg-type]
        current=current,
    )


def test_the_chain_line_says_which_tool_stopped_and_why_for_each_way_it_ends() -> None:
    assert chain_line(None) == ""
    # Going: which tool, and how far along (the first of five; the third).
    assert (
        chain_line(_chain([], current="cells")) == "Running Count the cells (1 of 5)."
    )
    assert (
        chain_line(
            _chain([("cells", "done", "x"), ("density", "done", "x")], current="route")
        )
        == "Running Route through the densest fields (3 of 5)."
    )
    # Every tool done: no "not run".
    every = [(tool, "done", f"{tool} done") for tool in RESULT_TOOLS]
    assert chain_line(_chain(every)) == (
        "Done: Count the cells, Density per well, Route through the densest fields, "
        "Report, Export for NIS-Elements."
    )
    # The chain of the tools left only: the names are those it ran.
    assert chain_line(_chain([("report", "done", "x")], plugins=("report",))) == (
        "Done: Report."
    )

    done_two = [("cells", "done", "x"), ("density", "done", "x")]
    failed = _chain(
        [*done_two, ("route", "failed", "route: PluginRunError: route: it broke")]
    )
    assert chain_line(failed) == (
        "Stopped at Route through the densest fields, which failed: route: "
        "PluginRunError: route: it broke. Done: Count the cells, Density per well. "
        "Not run: Report, Export for NIS-Elements."
    )
    refused = _chain(
        [*done_two, ("route", "refused", "route: ParamsError: labels: bad.")]
    )
    assert chain_line(refused) == (
        "Stopped at Route through the densest fields, which cannot run: route: "
        "ParamsError: labels: bad. Done: Count the cells, Density per well. "
        "Not run: Report, Export for NIS-Elements."
    )
    stopped = _chain([*done_two, ("route", "stopped", "route stopped")])
    assert chain_line(stopped) == (
        "Stopped at Route through the densest fields by STOP. Done: Count the cells, "
        "Density per well. Not run: Report, Export for NIS-Elements."
    )
    # A STOP between two tools: the last that ended is done and some are left.
    between = _chain(done_two)
    assert chain_line(between) == (
        "Stopped before Route through the densest fields by STOP. Done: Count the cells, "
        "Density per well. Not run: Report, Export for NIS-Elements."
    )
    # A STOP before the first tool: nothing was done.
    assert chain_line(_chain([])) == (
        "Stopped before Count the cells by STOP. Not run: Density per well, Route "
        "through the densest fields, Report, Export for NIS-Elements."
    )
    # The first tool stopped, none done and none after it: no "Done", no "Not run".
    assert chain_line(_chain([("report", "stopped", "x")], plugins=("report",))) == (
        "Stopped at Report by STOP."
    )
    # A name no table has is its own, and an outcome of several lines is one.
    odd = _chain(
        [("their-tool", "failed", "boom\nin two lines")], plugins=("their-tool",)
    )
    assert chain_line(odd) == (
        "Stopped at their-tool, which failed: boom in two lines."
    )


def test_files_lines_name_each_steps_file_without_its_key_and_only_when_done() -> None:
    cards = {
        "nis-export": _result_card(
            plugin="nis-export", files=("file: /run/export.xml",)
        ),
        "scan": _result_card(plugin="scan", files=("folder: /run/scan",)),
        "report": _result_card(
            plugin="report",
            files=("file: C:\\runs\\report: final.html", "file: /run/data.csv"),
        ),
        # A stopped record's card holds no files (FM-154): nothing to name.
        "cells": _result_card(plugin="cells", status="stopped", files=()),
        # An entry with no key is shown whole; a path that has a colon keeps it.
        "route": _result_card(plugin="route", files=("/run/route.json",)),
    }
    assert files_lines(cards) == (
        "Scan frames: /run/scan",
        "Route through the densest fields: /run/route.json",
        "Report: C:\\runs\\report: final.html",
        "Report: /run/data.csv",
        "Export for NIS-Elements: /run/export.xml",
    )
    assert files_lines({}) == ()
    assert FILES_HEADING == "Where the files are"
    assert NO_FILES_YET == "No step has written a file yet."
    # A card of a tool the screen does not list is not named.
    assert (
        files_lines({"timing": _result_card(plugin="timing", files=("file: x",))}) == ()
    )


def test_the_colour_bar_names_the_class_and_is_empty_without_one() -> None:
    assert colour_bar_title("single") == "single cells per mm²"
    assert colour_bar_title("doublet") == "doublet cells per mm²"
    assert colour_bar_title(None) == ""
    assert colour_bar_title("") == ""
    # A class comes from a record: one line, never markup.
    assert colour_bar_title("a\nb") == "a b cells per mm²"


_STATES = st.sampled_from(["ok", "not run", "unusable"])


@given(
    states=st.lists(_STATES, min_size=5, max_size=5),
    inputs=st.lists(_STATES, min_size=2, max_size=2),
    blocked=st.sampled_from([None, "stand closed"]),
    dry_scan=st.booleans(),
    running=st.sampled_from([None, *RESULT_TOOLS]),
)
def test_a_row_is_never_both_done_and_missing_and_a_missing_row_has_no_wait(
    states: list[str],
    inputs: list[str],
    blocked: str | None,
    dry_scan: bool,
    running: str | None,
) -> None:
    # Whatever the document holds, the rows and the chain's button agree: *Again* is exactly
    # a done tool, a missing input is said by the state alone, and the chain waits exactly
    # for what its first tool waits for.
    checklist = _checklist(
        scan_plan=inputs[0],
        scan=inputs[1],
        **dict(
            zip(
                ("cells", "density", "route", "report", "nis_export"),
                states,
                strict=True,
            )
        ),
    )
    rows = [
        _row(tool, checklist, blocked=blocked, dry_scan=dry_scan, running=running)
        for tool in RESULT_TOOLS
    ]
    left = results_left(checklist)
    for row in rows:
        assert row.done == (row.button == "Again")
        assert not (row.done and row.missing)
        assert not (row.missing and row.wait)
        assert not (row.missing and running == row.plugin)
    expected = None
    if not left:
        expected = NOTHING_LEFT
    else:
        first = next(row for row in rows if row.plugin == left[0])
        expected = (first.state if first.missing else first.wait) or None
    assert run_all_wait(rows, left) == expected


# -- round 2: plain words, notes drawn once, the frame, the files (m5 §17.12, #475) -----


def test_the_round_2_texts_are_the_designs() -> None:
    """Whole (FM-45): the design fixed these words with the owner, and a reword is a decision."""
    assert FRAME_DARK == (
        "The frame is dark: is the light on, and the light path on the camera?"
    )
    assert FRAME_SATURATED == "The frame is saturated: lower the exposure."
    assert REGISTER_ACROSS_A_TURN == (
        "The centres were captured with the {captured}; the {in_place} is in place: turn "
        "back to the {captured}, or redo the centres with the {in_place}."
    )
    assert REGISTER_OBJECTIVES_DIFFER == (
        "The centres were captured with different objectives ({which}): redo them with "
        "one objective."
    )
    assert AGAIN_NOTE == (
        "Again makes {names} out of date; Run all that are left runs them again."
    )
    assert RUN_ALL_LABEL in AGAIN_NOTE
    assert DISCONNECT_RELEASES == "Disconnect releases {stand} for NIS-Elements or ZEN."
    assert TIME_NOT_KNOWN == "Time: not known (no timing measured yet)"
    assert COVERAGE_WORDS == (
        "How much of each well's width the fields span: 0.4 is 40 %."
    )
    # `off` is refused when the autofocus is on (m3 §5.7 D2, #478): the page must not promise
    # that it "never touches" it (#498).
    assert AUTOFOCUS_WORDS == (
        "Autofocus: if-engaged uses it when it is on and locked; required refuses to "
        "start unless it is; off never uses it and refuses to start when it is on. "
        "The scan never switches it on."
    )
    assert SAMPLE_REGISTERS == (
        "Register the plate uses the profile's sample while no centre is captured; "
        "capture three to measure this plate."
    )


# The frame's verdict (#457): a frame is dark, saturated or lit.
_FRAME = (96, 128)  # not square (FM-73)


def _lit(shape: tuple[int, int] = _FRAME) -> Any:
    import numpy as np

    return (
        np.linspace(100.0, 3000.0, shape[0] * shape[1]).reshape(shape).astype(np.uint16)
    )


def test_frame_hint_says_dark_for_a_flat_frame_and_saturated_before_dark_and_nothing_for_a_lit_one() -> (
    None
):
    import numpy as np

    dark = np.full(_FRAME, 100, dtype=np.uint16)
    full_scale = np.full(_FRAME, 65535, dtype=np.uint16)

    assert frame_hint(dark, 16) == FRAME_DARK
    # A frame at full scale everywhere spans no counts either: saturated is asked first,
    # because the light path is not its remedy.
    assert frame_hint(full_scale, 16) == FRAME_SATURATED
    assert frame_hint(_lit(), 16) == ""
    # A camera whose depth is not known (a plugin's frame says 0) is not judged at all.
    assert frame_hint(dark, 0) == ""
    assert frame_hint(full_scale, 0) == ""
    # The same lit counts on a 12-bit camera are lit, and 4095 is its full scale.
    assert frame_hint(np.full(_FRAME, 4095, dtype=np.uint16), 12) == FRAME_SATURATED
    # A frame that is not 2-D says nothing rather than raising in the page's tick.
    assert frame_hint(np.zeros((4, 4, 3), dtype=np.uint16), 16) == ""


def test_frame_hint_tests_a_large_frame_on_every_other_row() -> None:
    import numpy as np

    def striped(side: int) -> Any:
        """Odd rows lit, even rows dark: a full test sees light, an every-other-row one does not."""
        frame = np.full((side, side), 100, dtype=np.uint16)
        frame[1::2, :] = 3000
        return frame

    # 1600 x 1600 is not larger: the whole frame is tested, and it carries signal.
    assert frame_hint(striped(1600), 16) == ""
    # 1602 x 1602 is: every other row and column, which here is the dark rows only.
    assert frame_hint(striped(1602), 16) == FRAME_DARK


def test_capture_note_names_a_capture_more_than_a_pitch_from_its_well() -> None:
    pitch = 9000.0
    far = capture_note("A1", (0.0, 0.0), (9001.0, 0.0), pitch, {})
    assert far == CAPTURE_FAR.format(well="A1", distance_mm=9.001)
    assert far.startswith("A1 was captured 9.0 mm from where the map puts it")
    # One µm under the bound: said nothing. The bound is a pitch, as the map draws it.
    assert capture_note("A1", (0.0, 0.0), (8999.0, 0.0), pitch, {}) == ""
    # H1 at A12's place, 99 mm off (the owner's walk): the distance is said in mm.
    assert "99.0 mm" in capture_note("H1", (49500.0, 0.0), (-49500.0, 0.0), pitch, {})


def test_capture_note_names_two_captures_at_one_place() -> None:
    pitch = 9000.0
    same = capture_note(
        "H1", (1000.0, 1000.0), (1000.0, 1000.0), pitch, {"A12": (1200.0, 1000.0)}
    )
    assert same == CAPTURE_SAME_PLACE.format(well="H1", other="A12")
    # The bound is half a pitch (4500 µm here): one µm each side of it.
    assert capture_note("A2", (0.0, 4499.0), None, pitch, {"A1": (0.0, 0.0)}) != ""
    assert capture_note("A2", (0.0, 4501.0), None, pitch, {"A1": (0.0, 0.0)}) == ""
    # Two neighbouring wells, each centred by eye, are one pitch apart and are right.
    assert (
        capture_note("A2", (8999.0, 0.0), (9000.0, 0.0), pitch, {"A1": (0.0, 0.0)})
        == ""
    )
    # A capture is never the same place as itself.
    assert capture_note("A1", (0.0, 0.0), (0.0, 0.0), pitch, {"A1": (0.0, 0.0)}) == ""
    # Far is said first when both hold.
    both = capture_note(
        "H1", (50000.0, 0.0), (0.0, 0.0), pitch, {"A12": (50000.0, 10.0)}
    )
    assert both.startswith("H1 was captured 50.0 mm")


def test_capture_note_is_empty_on_an_unplaced_map() -> None:
    # An unplaced map puts no well anywhere: nothing is far from where it is not.
    assert capture_note("A1", (99999.0, 99999.0), None, 9000.0, {}) == ""
    # Two captures at one place are still two captures at one place.
    assert capture_note("A1", (0.0, 0.0), None, 9000.0, {"B1": (0.0, 50.0)}) != ""


def test_register_objective_refusal_across_a_turn_names_both_objectives() -> None:
    said = register_objective_refusal(
        {"A1": "Nikon 10X", "H1": "Nikon 10X"}, "Nikon 40X"
    )
    assert said == REGISTER_ACROSS_A_TURN.format(
        captured="Nikon 10X", in_place="Nikon 40X"
    )
    assert said is not None
    assert "Nikon 10X" in said
    assert "Nikon 40X" in said
    # The same objective, one capture or several: nothing to say.
    assert register_objective_refusal({"A1": "Nikon 10X"}, "Nikon 10X") is None
    assert register_objective_refusal({}, "Nikon 10X") is None


def test_register_objective_refusal_names_captures_made_with_different_objectives() -> (
    None
):
    said = register_objective_refusal(
        {"A1": "Nikon 10X", "H1": "Nikon 40X", "A12": "Nikon 10X"}, "Nikon 40X"
    )
    assert said == REGISTER_OBJECTIVES_DIFFER.format(which="Nikon 10X and Nikon 40X")
    three = register_objective_refusal({"A1": "4X", "H1": "10X", "A12": "40X"}, "40X")
    assert three == REGISTER_OBJECTIVES_DIFFER.format(which="4X, 10X and 40X")
    # Whatever the turret says: two objectives are two objectives.
    assert register_objective_refusal({"A1": "10X", "H1": "40X"}, "10X") == (
        REGISTER_OBJECTIVES_DIFFER.format(which="10X and 40X")
    )


def test_register_objective_refusal_ignores_captures_whose_objective_is_not_known() -> (
    None
):
    # A capture restored from a run document has no label, and one whose turret could not be
    # read has `None`: neither is compared, so neither refuses anything.
    assert register_objective_refusal({"A1": None, "H1": None}, "Nikon 40X") is None
    assert (
        register_objective_refusal({"A1": "Nikon 10X", "H1": None}, "Nikon 10X") is None
    )
    # The known one still refuses a turn.
    assert register_objective_refusal(
        {"A1": "Nikon 10X", "H1": None}, "Nikon 40X"
    ) == REGISTER_ACROSS_A_TURN.format(captured="Nikon 10X", in_place="Nikon 40X")
    # An empty label is unknown too: *with* nothing would read as a fact.
    assert register_objective_refusal({"A1": ""}, "Nikon 40X") is None


def test_register_objective_refusal_is_none_when_the_turret_was_not_read() -> None:
    assert register_objective_refusal({"A1": "Nikon 10X"}, None) is None


def test_goto_row_text_says_going_at_the_estimate_at_and_the_refusal_without_the_tools_name() -> (
    None
):
    def answer(phase: str, *, estimated: bool = False, text: str = "") -> GotoAnswer:
        return GotoAnswer("A1", phase, estimated, text)  # type: ignore[arg-type]

    assert goto_row_text(None, "A1") == ""
    # The row of another well says nothing of this drive.
    assert goto_row_text(answer("going"), "H1") == ""
    assert goto_row_text(answer("going"), "A1") == "Going to A1…"
    assert goto_row_text(answer("at"), "A1") == "At A1."
    assert goto_row_text(answer("at", estimated=True), "A1") == (
        "At A1, the holder's estimate: centre it by eye with Live and the arrows, then "
        "capture."
    )
    # The refusal's or failure's one line is the controller's, as it is, with no tool name.
    assert goto_row_text(answer("refused", text=GOTO_UNPLACED), "A1") == GOTO_UNPLACED
    failed = goto_row_text(answer("failed", text="the move was stopped"), "A1")
    assert failed == "the move was stopped"
    assert "goto-well" not in failed


def test_map_frame_is_the_outlines_grown_by_one_pitch() -> None:
    frame = map_frame([(0.0, 0.0, 10.0, 20.0), (100.0, 50.0, 110.0, 70.0)], 9.0)
    assert frame == ((-9.0, 119.0), (-9.0, 79.0))
    # An outline given with its corners the other way round is the same outline.
    assert map_frame([(10.0, 20.0, 0.0, 0.0)], 9.0) == ((-9.0, 19.0), (-9.0, 29.0))
    with pytest.raises(ValueError, match="at least one outline"):
        map_frame([], 9.0)


def test_plate_rows_note_says_what_holds_for_every_row_once() -> None:
    assert plate_rows_note(running=None, live=False, from_sample=False) == ""
    assert plate_rows_note(running="goto-well", live=False, from_sample=False) == (
        busy_note("goto-well", live=False)
    )
    # Live is told apart: the way out is a button, and Capture still works.
    assert plate_rows_note(running="live", live=True, from_sample=False) == (
        GOTO_NOT_DURING_LIVE
    )
    assert plate_rows_note(running=None, live=False, from_sample=True) == (
        SAMPLE_REGISTERS
    )
    both = plate_rows_note(running="goto-well", live=False, from_sample=True)
    assert both.splitlines() == [busy_note("goto-well", live=False), SAMPLE_REGISTERS]


def _file_card(plugin: str, *files: str) -> ResultCard:
    return _result_card(plugin=plugin, files=tuple(files))


def test_files_html_escapes_every_text_and_links_only_files_inside_the_run_folder() -> (
    None
):
    run = Path("/runs/x")
    nasty = 'a<b>&"c.html'
    cards = {
        "scan": _file_card("scan", "folder: /runs/x/frames"),
        "report": _file_card("report", f"file: /runs/x/report/{nasty}"),
        "nis-export": _file_card("nis-export", "file: /data/p.xml"),
        "route": _file_card("route", "file: /runs/x/../etc/passwd"),
    }

    page = files_html(cards, run)

    # The name is data: escaped in the text and URL-quoted in the link.
    assert (
        '<a href="/run-files/report/a%3Cb%3E%26%22c.html" target="_blank" '
        'rel="noopener">report/a&lt;b&gt;&amp;&quot;c.html</a>'
    ) in page
    assert nasty not in page
    # The scan's folder, a file outside the run folder and a `..` are text, never links.
    assert page.count("<a ") == 1
    assert "Scan frames: /runs/x/frames" in page
    assert "Export for NIS-Elements: /data/p.xml" in page
    assert "Route through the densest fields: /runs/x/../etc/passwd" in page
    # The plain names, in the screen's order: the scan first, then each tool.
    assert (
        page.index("Scan frames") < page.index("Route through") < page.index("Report")
    )


def test_files_html_escapes_the_text_only_entries_as_well() -> None:
    # `nis-export --param out=` can be any absolute path on the PC, and a POSIX file name may hold
    # `<`: what is not a link is text in the same pane, and escaped (FM-139).
    hostile = '<img src=x onerror=1> & "q"'
    cards = {
        "scan": _file_card("scan", f"folder: /runs/x/{hostile}"),
        "report": _file_card("report", f"file: /elsewhere/{hostile}.html"),
        "route": _file_card("route", f"file: /runs/x/../{hostile}"),
        "nis-export": _file_card("nis-export", f"file: {hostile}"),
    }

    page = files_html(cards, Path("/runs/x"))

    assert "<img" not in page
    assert page.count(html.escape(hostile)) == 4
    assert "&lt;img src=x onerror=1&gt; &amp; &quot;q&quot;" in page
    assert "<a " not in page


def test_files_html_is_div_and_link_elements_only_whatever_a_file_is_called() -> None:
    """FM-271: the one HTML pane is read as a browser reads it, not by looking for substrings.

    A file called ``<img src=x onerror=1>.html``, one called ``"><b>.html`` (a quote and a
    bracket close the ``href`` attribute and the element if they are not escaped) and a
    text-only entry that holds ``<its run document>`` come out as ``div`` and ``a`` elements
    and nothing else; every link has exactly its ``href``, ``target`` and ``rel``; and the
    text the reader sees holds the hostile characters as they were written.
    """
    run = Path("/runs/x")
    image = "<img src=x onerror=1>.html"
    quote = '"><b>.html'
    placeholder = "reuse it with from_run_doc=<its run document>"
    cards = {
        "report": _file_card(
            "report", f"file: /runs/x/report/{image}", f"file: /runs/x/report/{quote}"
        ),
        "nis-export": _file_card("nis-export", f"file: /data/{placeholder}"),
    }

    class Elements(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.starts: list[tuple[str, dict[str, str | None]]] = []
            self.data: list[str] = []

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            self.starts.append((tag, dict(attrs)))

        def handle_data(self, data: str) -> None:
            self.data.append(data)

    parsed = Elements()
    parsed.feed(files_html(cards, run))
    parsed.close()

    assert {tag for tag, _ in parsed.starts} == {"div", "a"}
    links = [attrs for tag, attrs in parsed.starts if tag == "a"]
    assert len(links) == 2
    for attrs in links:
        assert sorted(attrs) == ["href", "rel", "target"]
        assert (attrs["href"] or "").startswith("/run-files/")
    seen = "".join(parsed.data)
    assert f"report/{image}" in seen
    assert f"report/{quote}" in seen
    assert placeholder in seen


def test_files_html_quotes_a_windows_relative_path_as_posix() -> None:
    # A page served on Windows names its files with backslashes; the route takes `/`.
    run = PureWindowsPath("C:/runs/x")
    cards = {"report": _file_card("report", "file: C:\\runs\\x\\report\\a b.html")}

    page = files_html(cards, run)

    assert 'href="/run-files/report/a%20b.html"' in page
    assert ">report/a b.html</a>" in page
    # Another drive is outside the run folder.
    other = {"report": _file_card("report", "file: D:\\runs\\x\\report\\a.html")}
    assert "<a " not in files_html(other, run)


def test_files_html_says_so_when_no_step_wrote_a_file() -> None:
    assert files_html({}, Path("/runs/x")) == f"<div>{NO_FILES_YET}</div>"
    # A card with no files is none; no run folder links nothing.
    assert (
        files_html({"report": _file_card("report")}, None)
        == f"<div>{NO_FILES_YET}</div>"
    )
    nothing_to_link = files_html(
        {"report": _file_card("report", "file: /r/a.html")}, None
    )
    assert "<a " not in nothing_to_link
    assert "Report: /r/a.html" in nothing_to_link


def test_tool_title_names_every_built_in_tool_and_a_third_partys_by_its_plugin_name() -> (
    None
):
    from smc.plugins import discover

    shipped = {
        name
        for name, entry in discover().plugins.items()
        if entry.target.startswith("smc.")
    }
    assert shipped, (
        "no plugin of the repository was found: the entry points are not installed"
    )
    assert shipped <= set(TOOL_TITLES), sorted(shipped - set(TOOL_TITLES))
    assert tool_title("goto-well") == "Go to"
    assert tool_title("plate-registration") == "Registration"
    assert tool_title("scan") == "Scan"
    # A name nobody listed is its own, and so is an activity that is no tool.
    assert tool_title("their-tool") == "their-tool"
    assert tool_title("live") == "live"
    # The plain titles name no plugin: that was the defect.
    assert all(title != plugin for plugin, title in TOOL_TITLES.items())
    assert not any(title == title.lower() for title in TOOL_TITLES.values())


def test_local_time_is_the_local_clock_to_the_minute_and_reads_a_naive_time_as_utc() -> (
    None
):
    noon = datetime(2026, 11, 13, 12, 0, 59, tzinfo=timezone.utc)
    assert local_time(noon) == noon.astimezone().strftime("%Y-%m-%d %H:%M")
    # To the minute: the seconds are dropped, not rounded.
    assert local_time(noon).count(":") == 1
    # Naive is UTC (the document's times are), whatever zone the PC is in.
    assert local_time(datetime(2026, 11, 13, 12, 0)) == local_time(noon)
    # An offset is converted: 14:20 at +05:00 is 09:20 UTC, the same instant.
    five = timezone(timedelta(hours=5))
    assert local_time(datetime(2026, 11, 13, 14, 20, tzinfo=five)) == local_time(
        datetime(2026, 11, 13, 9, 20, tzinfo=timezone.utc)
    )


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="tzset is POSIX only")
def test_local_time_is_in_the_zone_of_the_pc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "UTC-2")  # POSIX: two hours east of UTC
    time.tzset()
    try:
        assert local_time(datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)) == (
            "2026-11-13 14:00"
        )
        # A naive time is UTC and not the PC's: on a PC at UTC+2 it is still 14:00 local (a CI
        # runner at UTC could not tell the two apart).
        assert local_time(datetime(2026, 11, 13, 12, 0)) == "2026-11-13 14:00"
    finally:
        monkeypatch.undo()
        time.tzset()


@pytest.mark.parametrize("error", [OSError, OverflowError, ValueError])
def test_local_time_of_a_start_the_pc_cannot_show_is_the_values_own_text(
    error: type[Exception],
) -> None:
    # A date the PC cannot show in its zone (Windows before 1970 or after year 3000, year 1 or 9999
    # under an offset) raises one of these from `astimezone()`: the page's redraw must not end on it
    # (FM-125). A subclass whose `astimezone` raises makes it happen on any PC.
    class Unshowable(datetime):
        def astimezone(self, tz: Any = None) -> datetime:
            raise error("no zone shows this date")

    odd = Unshowable(2026, 11, 13, 14, 20, tzinfo=timezone(timedelta(hours=5)))

    assert local_time(odd) == "2026-11-13T14:20+05:00"


def test_the_again_note_is_one_line_of_short_names() -> None:
    everything = _results(
        scan_plan="ok",
        scan="ok",
        cells="ok",
        density="ok",
        route="ok",
        report="ok",
        nis_export="ok",
    )
    note = again_note("cells", everything)
    assert "\n" not in note
    assert len(note) < 100
    # Short names, never the rows' long ones.
    assert "Density per well" not in note
    assert "Route through the densest fields" not in note
    assert set(RESULT_SHORT) == set(RESULT_TOOLS)


def test_registered_line_names_the_bound_the_tool_refuses_beyond() -> None:
    result = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=-0.01,
        method="references",
        residual_um=62.3,
        scale_error=None,
        n_refs=3,
    )
    assert registered_line(result) == (
        "Registered: residual 62.3 µm of the 300 µm allowed, turned -0.01°"
    )
    # The bound is the registration's own, not a copy that could drift.
    from smc.plugins.plate_registration.algorithm import GOOD_RESIDUAL_UM

    assert f"{GOOD_RESIDUAL_UM:g} µm allowed" in registered_line(result)


def test_plain_default_reads_a_tuple_as_its_items() -> None:
    assert plain_default("('single',)") == "single"
    assert plain_default("('a', 'b')") == "a, b"
    assert plain_default("(1, 2.5)") == "1, 2.5"
    # What does not read as a tuple of literals is as it came, never an exception.
    assert plain_default("(x, y)") == "(x, y)"
    assert plain_default("(1 +)") == "(1 +)"
    assert plain_default("()") == "()"
    assert plain_default("(1)") == "(1)"
    assert plain_default("(" * 500 + ")" * 500) == "(" * 500 + ")" * 500
    # The quote rule of before still holds.
    assert plain_default("'if-engaged'") == "if-engaged"
    assert field_source_line(
        value="", default="('single',)", preset=None, preset_value=None, suggested=None
    ) == ("the tool's default: single")


def test_known_text_says_the_calibration_in_local_time_and_what_disconnect_does() -> (
    None
):
    here = CalibrationNote(
        objective="Nikon 10X",
        when_utc="2026-11-12 09:30:00",
        when_local="2026-11-12 10:30",
    )
    lines = known_text(
        objective="Nikon 10X", calibration=here, light_on=True, stand="demo-sample"
    ).split("\n")
    assert lines == [
        "Objective: Nikon 10X, in place",
        "Camera: calibrated at Nikon 10X on 2026-11-12 10:30",
        "Light: on",
        "Disconnect releases demo-sample for NIS-Elements or ZEN.",
    ]
    # Not connected: no word of a release, and a note with no local time keeps its UTC one.
    old = CalibrationNote(objective=None, when_utc="2026-11-12 09:30:00")
    said = known_text(objective=None, calibration=old, light_on=None).split("\n")
    assert said == [
        "Objective: not known",
        "Camera: calibrated on 2026-11-12 09:30:00 UTC, objective not recorded",
        "Light: not known",
    ]
    # The stand's name is a text of the profile, printed as it is (FM-139).
    markup = known_text(
        objective=None, calibration=None, light_on=None, stand="<b>x</b>"
    )
    assert (
        markup.split("\n")[-1]
        == "Disconnect releases <b>x</b> for NIS-Elements or ZEN."
    )


def test_a_results_row_says_when_it_was_done_in_local_time() -> None:
    card = _result_card(when_local="2026-11-12 10:30")
    row = _row("cells", _results(scan="ok", cells="ok"), card=card)
    assert row.state == "Done 2026-11-12 10:30"
    assert "UTC" not in row.state
    dry = _row(
        "cells",
        _results(scan="ok", cells="ok"),
        card=_result_card(when_local="2026-11-12 10:30", dry_run=True),
    )
    assert dry.state == "Done 2026-11-12 10:30 (dry run)"


# --- part B: the Scan screen after a STOP or a failure (m5 §17.12 D17, D18, #470) ------------


def test_part_bs_texts_are_the_design_docs() -> None:
    # Copied from §17.12.5: a change of words is a change of the design.
    assert SCAN_CONTINUE == "Continue the scan"
    assert SCAN_START_AGAIN == "Start again"
    assert SCAN_START_AGAIN_CONFIRM.format(done=36) == (
        "Start again drops the 36 fields already taken: press again to drop them."
    )
    assert SCAN_HALTED == "Press Resume (top right) first: the stand is halted."
    assert SCAN_REPLACES.format(frames=72) == (
        "Starting the scan again replaces this one (72 frames)."
    )


def _end(
    kind: Literal["stopped", "failed"] = "stopped",
    done: int | None = 45,
    total: int | None = 72,
    text: str = "",
) -> ScanEnd:
    return ScanEnd(kind=kind, done=done, total=total, text=text, resumable=True)


def test_a_stopped_scan_says_where_it_stopped() -> None:
    assert scan_end_lines(_end()) == ("Stopped by STOP at field 45 of 72.",)


def test_a_failed_scan_says_where_and_the_errors_line_without_its_type() -> None:
    end = _end(
        "failed",
        text="PluginRunError: scan: the camera timed out after   5 s\nretry",
    )
    assert scan_end_lines(end) == (
        "Failed at field 45 of 72: scan: the camera timed out after 5 s retry",
    )
    # A message that begins like a name but is no exception type keeps its words.
    plain = _end("failed", text="`xy_stage XY`: the move was stopped")
    assert scan_end_lines(plain) == (
        "Failed at field 45 of 72: `xy_stage XY`: the move was stopped",
    )


@pytest.mark.parametrize(
    ("done", "total"), [(None, None), (None, 72), (45, None)], ids=str
)
def test_a_scan_end_with_no_count_never_says_none(
    done: int | None, total: int | None
) -> None:
    # A page opened after the stop kept no progress: no `field None of None`.
    stopped = scan_end_lines(_end("stopped", done, total))
    failed = scan_end_lines(_end("failed", done, total, "RuntimeError: dark"))
    assert stopped == ("Stopped by STOP.",)
    assert failed == ("Failed: dark",)
    assert not any("None" in line for line in (*stopped, *failed))


def test_a_failure_with_no_recorded_error_says_so() -> None:
    assert scan_end_lines(_end("failed", text="")) == (
        "Failed at field 45 of 72: no reason was recorded",
    )


def test_the_first_click_on_start_again_says_what_the_second_drops() -> None:
    assert scan_again_confirm(36) == SCAN_START_AGAIN_CONFIRM.format(done=36)
    assert scan_again_confirm(None) == (
        "Start again drops the fields already taken: press again to drop them."
    )


def test_the_scan_actions_wait_for_the_stand_then_the_halt_then_a_run() -> None:
    wait = scan_actions_wait
    assert wait(connected=False, halted=True, running="scan") == NOT_CONNECTED
    assert wait(connected=True, halted=True, running="scan") == SCAN_HALTED
    assert wait(connected=True, halted=False, running="snap") == busy_note(
        "snap", live=False
    )
    assert wait(connected=True, halted=False, running="live") == busy_note(
        "live", live=True
    )
    assert wait(connected=True, halted=False, running=None) is None


def test_a_resume_command_names_resume_once_and_the_same_words_otherwise() -> None:
    common: dict[str, Any] = {
        "profile": "demo-sample",
        "run_doc": Path("run.json"),
        "dry_run": False,
    }
    plain = shlex.split(command_line("scan", ["autofocus=off"], **common))
    resumed = shlex.split(
        command_line("scan", ["autofocus=off"], **common, resume=True)
    )
    assert "--resume" not in plain
    assert resumed.count("--resume") == 1
    assert [word for word in resumed if word != "--resume"] == plain


def test_the_chain_line_names_a_tool_outside_results_by_its_plain_title() -> None:
    going = ChainStatus(("timing", "scan-plan"), (), "timing")
    assert chain_line(going) == "Running Timing (1 of 2)."
    done = ChainStatus(
        ("timing", "scan-plan"),
        (
            ChainEnd("timing", "done", "Timing done in 3.0 s"),
            ChainEnd("scan-plan", "done", "Field layout done in 0.1 s"),
        ),
        None,
    )
    assert chain_line(done) == "Done: Timing, Field layout."


def test_chain_line_ends_an_all_done_chain_with_its_first_assumed_line() -> None:
    """*Measure the time* dropped the ``left the autofocus off`` warning, which only a card showed (m5 §17.12.11 D32, FM-165)."""
    done = ChainStatus(
        ("timing", "scan-plan"),
        (
            ChainEnd(
                "timing", "done", "Timing done in 3.0 s", ("left the autofocus off",)
            ),
            ChainEnd("scan-plan", "done", "Field layout done in 0.1 s"),
        ),
        None,
    )
    assert chain_line(done) == (
        "Done: Timing, Field layout. Assumed: left the autofocus off."
    )
    # The first line of the first tool that has some; no count (the guided screens have no card).
    several = ChainStatus(
        ("timing", "scan-plan"),
        (
            ChainEnd("timing", "done", "Timing done in 3.0 s"),
            ChainEnd(
                "scan-plan",
                "done",
                "Field layout done in 0.1 s",
                ("pixel size from the profile", "a second line"),
            ),
        ),
        None,
    )
    assert chain_line(several) == (
        "Done: Timing, Field layout. Assumed: pixel size from the profile."
    )
    # A line that already ends with a full stop gets no second one; line breaks are spaces.
    stopped_at_a_dot = ChainStatus(
        ("timing",),
        (ChainEnd("timing", "done", "x", ("left it as it was.\nand more",)),),
        None,
    )
    assert chain_line(stopped_at_a_dot) == (
        "Done: Timing. Assumed: left it as it was. and more."
    )
    # A long line is cut as the strip cuts the first assumption: the card holds the whole.
    long = "x" * (ASSUMED_LINE_CHARS + 40)
    cut = chain_line(
        ChainStatus(("timing",), (ChainEnd("timing", "done", "x", (long,)),), None)
    )
    assert cut == f"Done: Timing. Assumed: {'x' * ASSUMED_LINE_CHARS}…."
    # A chain that stopped or failed says nothing of what a tool assumed, and a chain still
    # going names the tool.
    stopped = ChainStatus(
        ("timing", "scan-plan"),
        (ChainEnd("timing", "done", "x", ("left the autofocus off",)),),
        None,
    )
    assert "Assumed" not in chain_line(stopped)
    failed = ChainStatus(
        ("timing", "scan-plan"),
        (
            ChainEnd("timing", "done", "x", ("left the autofocus off",)),
            ChainEnd("scan-plan", "failed", "Field layout failed: no"),
        ),
        None,
    )
    assert "Assumed" not in chain_line(failed)
    going = ChainStatus(
        ("timing", "scan-plan"),
        (ChainEnd("timing", "done", "x", ("left the autofocus off",)),),
        "scan-plan",
    )
    assert chain_line(going) == "Running Field layout (2 of 2)."
    # ChainEnd of three arguments still builds: the default is no assumed line.
    assert ChainEnd("timing", "done", "x").assumed == ()


def test_checkpoint_note_names_the_status_and_the_way_for_scan_and_for_any_other_stage() -> (
    None
):
    """What a single run of a tool that feeds a stopped stage says before the click (m5 §17.12.11 D30)."""
    assert checkpoint_note("scan", "stopped") == (
        "This puts the stopped scan out of date: Continue the scan will not be offered "
        "any more, and Start the scan takes it from the first field."
    )
    assert checkpoint_note("scan", "failed").startswith(
        "This puts the failed scan out of date: "
    )
    assert checkpoint_note("plate-finding", "failed") == (
        "This puts the failed plate-finding out of date: "
        "smc run plate-finding --resume cannot continue it afterwards."
    )
    assert checkpoint_note("plate-finding", "stopped").startswith(
        "This puts the stopped plate-finding out of date: "
    )
    # One sentence, plain text: a stage's name is a plugin's.
    for stage in ("scan", "plate-finding"):
        assert "\n" not in checkpoint_note(stage, "stopped")


# -- the offsets between objectives (m5 §18, #382) ----------------------------------------

_TEN_X = "Nikon 10X S Fluor"
_FOUR_X = "Nikon 4X Plan Fluor"
_FORTY_X = "Nikon 40X Plan Fluor ELWD"


def _offsets_measure(
    objective: str,
    shift: tuple[float, float],
    wells: tuple[str, ...] = ("A1", "H12", "A12"),
) -> TargetMeasure:
    fits = tuple(
        OffsetWell(
            well=well,
            origin_xy_um=(0.0, 0.0),
            delta_xy_um=shift,
            fit=_fit(well, shift, (1, 1, 1, 1)),
        )
        for well in wells
    )
    return TargetMeasure(
        objective=objective,
        calibration="cal.json",
        calibration_measured="2026-11-10T10:00:00+00:00",
        pixel_size_um=2.5,
        z_after_turn_um=None,
        wells=fits,
        shift_xy_um=shift,
        spread_um=0.03,
        max_spread_um=2.0,
        frames=24,
        folder=None,
    )


def _offsets_result(
    measures: tuple[TargetMeasure, ...],
    offsets: tuple[MeasuredOffset, ...],
    *,
    plate_objective: str = _TEN_X,
    in_place: str = _FORTY_X,
) -> ObjectiveOffsetsResult:
    return ObjectiveOffsetsResult(
        plate_objective=plate_objective,
        reference=plate_objective,
        reference_in_profile=False,
        measures=measures,
        offsets=offsets,
        objective_in_place=in_place,
        elapsed_s=62.8,
    )


def test_the_other_objectives_leave_out_the_plates_and_the_empty_positions() -> None:
    labels = ("", _FOUR_X, _TEN_X, "   ", _FORTY_X)

    assert offsets_others(labels, _TEN_X) == (_FOUR_X, _FORTY_X)
    assert offsets_others((_TEN_X, ""), _TEN_X) == ()


def test_calibration_paths_drop_blank_lines_and_one_pair_of_quotes() -> None:
    text = (
        '  cal-10x.json \n\n"C:\\Users\\me\\My Docs\\cal-40x.json"\r\n   \n'
        '""\n""twice""\n"unclosed\n'
    )

    assert calibration_paths(text) == (
        "cal-10x.json",
        "C:\\Users\\me\\My Docs\\cal-40x.json",
        '"twice"',
        '"unclosed',
    )
    assert calibration_paths("") == ()
    assert calibration_paths("\n \n") == ()


def test_the_offsets_lead_names_the_plates_objective_and_the_others() -> None:
    assert offsets_lead(_TEN_X, (_FOUR_X, _FORTY_X)) == (
        f"The plate was registered with {_TEN_X}. Other objectives on the turret: "
        f"{_FOUR_X}, {_FORTY_X}. The page turns the turret to each objective of the "
        "calibrations below, finds the walls of the wells it registered again, and "
        "says how far the same wells moved."
    )


def test_the_offsets_lead_with_no_other_objective_says_so() -> None:
    assert offsets_lead(_TEN_X, ()) == (
        f"The plate was registered with {_TEN_X}. The turret has no other objective."
    )


def test_the_objective_in_place_line_says_not_known_and_may_be_turning_while_it_runs() -> (
    None
):
    assert objective_in_place_line(_FORTY_X, running=False) == (
        f"Objective in place: {_FORTY_X}"
    )
    assert objective_in_place_line(_FORTY_X, running=True) == (
        f"Objective in place: {_FORTY_X}"
    )
    assert objective_in_place_line(None, running=False) == (
        "Objective in place: not known"
    )
    assert objective_in_place_line(None, running=True) == (
        "Objective in place: not known (the turret may be turning)"
    )
    # An empty turret position is no objective in place.
    assert objective_in_place_line("", running=True) == (
        "Objective in place: not known (the turret may be turning)"
    )


def test_the_offsets_run_line_says_no_report_yet_then_the_report_and_its_count() -> (
    None
):
    assert offsets_run_line(None) == (
        "no report yet: the turn and the walk to the first well come first"
    )
    assert offsets_run_line(Progress("objective-offsets", 3, 24, "40X: A12 -Y")) == (
        "40X: A12 -Y (3 of 24)"
    )
    assert offsets_run_line(Progress("objective-offsets", 3, None, "40X: A12 -Y")) == (
        "40X: A12 -Y"
    )
    assert offsets_run_line(Progress("objective-offsets", 0, 24, "")) == "(0 of 24)"


def test_the_confirmation_names_what_is_in_place_the_turns_and_where_the_turret_is_left() -> (
    None
):
    assert offsets_confirm_text(_TEN_X, (_FOUR_X, _FORTY_X)) == (
        f"Turn the turret to {_FOUR_X}, then {_FORTY_X}, with the plate on the stage? "
        "Check that each objective clears the plate (its working distance, the "
        "plate's height) and that the stage is clear. "
        f"In place now: {_TEN_X}. The turret is left at {_FORTY_X}, not turned back, "
        "and the autofocus is switched off and not engaged again."
    )
    assert "In place now: not known." in offsets_confirm_text(None, (_FOUR_X,))
    assert offsets_confirm_text(None, (_FOUR_X,)).startswith(
        f"Turn the turret to {_FOUR_X}, with the plate on the stage?"
    )
    with pytest.raises(ValueError, match="at least one objective"):
        offsets_confirm_text(_TEN_X, ())


def test_the_offsets_card_lines_name_each_target_its_shift_spread_wells_and_bound() -> (
    None
):
    # Asymmetric on purpose, with a -0.0 that must not print as "-0.00" (FM-203); the
    # expected text is written from the numbers.
    result = _offsets_result(
        (
            _offsets_measure(_FOUR_X, (-12.341, 6.789)),
            _offsets_measure(_FORTY_X, (-0.0, 12.25), wells=("A1", "B2")),
        ),
        (
            MeasuredOffset(objective=_FOUR_X, xy_offset_um=(-12.341, 6.789)),
            MeasuredOffset(objective=_FORTY_X, xy_offset_um=(-0.0, 19.039)),
        ),
    )

    assert offsets_card_lines(result) == (
        f"{_FOUR_X}: shift from {_TEN_X} (-12.34, +6.79) µm, spread 0.03 µm over A1, "
        "H12, A12 (bound 2.00 µm)",
        f"{_FORTY_X}: shift from {_TEN_X} (+0.00, +12.25) µm, spread 0.03 µm over A1, "
        "B2 (bound 2.00 µm)",
        f"offsets to write: {_FOUR_X} (-12.34, +6.79) µm, {_FORTY_X} (+0.00, +19.04) µm",
        f"the turret is left at {_FORTY_X}",
    )


def test_a_shift_that_rounds_to_zero_prints_no_minus_sign_on_the_offsets_card() -> None:
    # -0.001 is not -0.0: `+ 0.0` alone left it as "-0.00" (#514 point 5, the CLI's twin).
    # -0.005 rounds to -0.01 and keeps its sign: only a value that prints as zero loses it.
    result = _offsets_result(
        (_offsets_measure(_FOUR_X, (-0.001, -0.004)),),
        (MeasuredOffset(objective=_FOUR_X, xy_offset_um=(-0.001, -0.006)),),
    )

    lines = offsets_card_lines(result)

    assert lines[0].startswith(f"{_FOUR_X}: shift from {_TEN_X} (+0.00, +0.00) µm, ")
    assert lines[1] == f"offsets to write: {_FOUR_X} (+0.00, -0.01) µm"


def test_the_offsets_card_lines_are_one_line_each_for_a_label_with_a_line_break_and_700_characters() -> (
    None
):
    label = "40X\nlens " + "x" * 700
    result = _offsets_result(
        (_offsets_measure(label, (1.0, 2.0)),),
        (MeasuredOffset(objective=label, xy_offset_um=(1.0, 2.0)),),
        in_place=label,
    )

    lines = offsets_card_lines(result)

    assert len(lines) == 3
    assert all("\n" not in line and "\r" not in line for line in lines)
    assert lines[0].startswith("40X lens xxxx")
    # The card's own bound cuts them (controller._card); the pure texts keep what they say.
    assert all(len(line) > 700 for line in lines)


def test_error_words_drop_the_exception_type_and_the_tools_name_and_leave_other_text_alone() -> (
    None
):
    tool = "no camera calibration of Nikon 40X in calibrations: turn to it"

    assert error_words(f"PluginRunError: objective-offsets: {tool}") == tool
    assert error_words(f"objective-offsets: {tool}") == tool
    assert error_words(f"ValueError: {tool}") == tool
    assert error_words(tool) == tool
    # Another tool's name, and a colon that is not a type's, stay.
    assert error_words("plate-walls: it failed") == "plate-walls: it failed"
    assert error_words("disk full: D:\\run.json") == "disk full: D:\\run.json"
    # The type is taken once, at the start only.
    assert error_words("see PluginRunError: x") == "see PluginRunError: x"


def test_the_plate_refusals_the_page_foresees_are_the_tools_words() -> None:
    from smc.plugins.errors import PluginRunError
    from smc.plugins.objective_offsets.plugin import _checked_plate

    wells = (
        _fit("A1", (0.0, 0.0), (1, 1, 1, 1)),
        _fit("H12", (9.0, 9.0), (1, 1, 1, 1)),
    )
    walls = _walls_result(None, wells=wells)
    assert walls.walls is not None
    fine = walls.model_copy(update={"objective": walls.walls.objective})
    cases = {
        "by eye": fine.model_copy(update={"method": "references"}),
        "no objective": fine.model_copy(update={"objective": None}),
        "mixed up": fine.model_copy(update={"objective": "another"}),
    }

    for plate in cases.values():
        with pytest.raises(PluginRunError) as raised:
            _checked_plate(plate)
        assert offsets_plate_refusal(plate) == raised.value.reason
    assert offsets_plate_refusal(fine) is None
    assert _checked_plate(fine)[0] == walls.walls.objective


def test_the_offsets_refusals_and_waits_are_one_line_in_the_tools_words() -> None:
    assert offsets_is_the_plates(_TEN_X) == (
        f"{_TEN_X} is the objective the plate was registered with: its offset from "
        "itself is zero; leave its document out"
    )
    assert offsets_not_on_turret(_FORTY_X, ("", _FOUR_X, _TEN_X)) == (
        f"{_FORTY_X} is not on the turret: {_FOUR_X}, {_TEN_X}"
    )
    assert offsets_in_two_documents(_FORTY_X, "a.json", "b.json") == (
        f"{_FORTY_X} is in two documents: a.json and b.json"
    )
    assert offsets_changed((_FOUR_X, _FORTY_X), (_FORTY_X,)) == (
        f"the objectives changed since you confirmed them ({_FOUR_X}, {_FORTY_X} → "
        f"{_FORTY_X}): measure again"
    )
    assert OFFSETS_TOO_MANY == "at most 5 objectives in one run"
    assert OFFSETS_NO_DOCUMENT == (
        "Calibrate the camera at each objective to measure (the button above), or "
        "give one run document per line."
    )
    assert OFFSETS_NOTHING_ARMED == (
        "choose the documents and press Measure the offsets again"
    )
    assert offsets_plan_heading("/p/stand.toml") == (
        "Writing these replaces what /p/stand.toml holds:"
    )
    assert offsets_written_line("/p/stand.toml") == (
        "Written into /p/stand.toml. The page keeps the offsets it was started with: "
        "restart it to use these."
    )


def test_a_calibration_line_says_its_objective_and_its_date_or_that_none_is_known() -> (
    None
):
    assert calibration_read_line(
        "cal-40x.json", _FORTY_X, "2026-11-14T14:06:37+00:00"
    ) == (f"cal-40x.json: calibrated at {_FORTY_X} on 2026-11-14")
    assert calibration_read_line("c.json", None, "at an unknown time") == (
        "c.json: calibrated at an unknown objective, at an unknown time"
    )


def test_the_lead_while_the_turret_is_not_read_does_not_say_it_has_no_other_objective() -> (
    None
):
    assert offsets_lead_unread(_TEN_X) == (
        f"The plate was registered with {_TEN_X}. The turret's other objectives are "
        "read when the stand is connected."
    )


def test_the_how_to_names_the_three_steps_the_command_and_where_a_path_is_read_from() -> (
    None
):
    command = (
        "smc run camera-calibration -p /p/stand.toml --run-doc cal-<objective>.json"
    )

    assert calibration_how_text(command).splitlines() == [
        "1. In the drive bar, Change objective to the objective, and bring a well's "
        "wall into the image (Live and the arrows).",
        "2. Press Disconnect on the Stand screen: the terminal needs the stand.",
        "3. In a terminal, run this, then press Connect again:",
        f"   {command}",
        "A relative path is read from where smc ui was started.",
    ]


def test_calibration_paths_drop_the_invisible_characters_a_windows_dialog_puts_before_a_path() -> (
    None
):
    # Windows' Properties dialog puts U+202A before the path it copies, and the OS refuses the
    # name with it: an operator sees a path that looks right and an error that says nothing.
    pasted = (
        "\u202aC:\\Users\\lab\\cal-40x.json\n"
        "\u200e D:\\cal.json \u200f\n"
        '"\u202aE:\\quoted.json"\n'
        "F:\\in\u200bside.json"
    )

    assert calibration_paths(pasted) == (
        "C:\\Users\\lab\\cal-40x.json",
        "D:\\cal.json",
        "E:\\quoted.json",
        # Inside a name nothing is touched: only the two ends of a line are.
        "F:\\in\u200bside.json",
    )
    assert calibration_paths("\u202a\n\u200b \u200f") == ()


# -- the page makes the camera calibration of each objective (m5 \u00a718.6, #501) -------------

_TWENTY_X = "Nikon 20X Plan Fluor ELWD"
#: The demo turret's labels (position 1 to 6), with empty positions among them.
_TURRET = (_FORTY_X, _TEN_X, "", _TWENTY_X, "", "")


def test_a_turret_position_is_the_labels_index_plus_one_and_unknown_for_none_a_blank_a_duplicate_a_missing_label_and_a_position_past_twelve() -> (
    None
):
    assert turret_position(_TURRET, _FORTY_X) == 1
    assert turret_position(_TURRET, _TEN_X) == 2
    # The empty position before it counts: the index of the 20X is 3, its position 4.
    assert turret_position(_TURRET, _TWENTY_X) == 4
    assert turret_position(_TURRET, None) is None
    assert turret_position(_TURRET, "") is None
    assert turret_position(_TURRET, "   ") is None
    assert turret_position(_TURRET, "Nikon 4X Plan Fluor") is None
    # Two positions carry the label: which one is in place cannot be said.
    assert turret_position((_TEN_X, _FORTY_X, _TEN_X), _TEN_X) is None
    assert turret_position((_TEN_X, _FORTY_X, _TEN_X), _FORTY_X) == 2
    # A position past the page's last document.
    far = ("",) * MAX_POSITIONS + (_TEN_X,)
    assert turret_position(far, _TEN_X) is None
    last = ("",) * (MAX_POSITIONS - 1) + (_TEN_X,)
    assert turret_position(last, _TEN_X) == MAX_POSITIONS
    assert turret_position((), _TEN_X) is None


def test_the_calibration_file_is_named_by_position_and_refuses_a_position_outside_the_turret() -> (
    None
):
    assert CALIBRATIONS_FOLDER == "calibrations"
    assert MAX_POSITIONS == 12
    assert calibration_file(1) == "position-1.json"
    assert calibration_file(12) == "position-12.json"
    for outside in (0, -1, 13):
        with pytest.raises(ValueError, match="1 to 12"):
            calibration_file(outside)


def _wait(**changes: Any) -> str | None:
    kwargs: dict[str, Any] = {
        "connected": True,
        "running": None,
        "live": False,
        "dry_run": False,
        "objective": _FORTY_X,
        "plate_objective": _TEN_X,
        "light_on": True,
    }
    kwargs.update(changes)
    return calibrate_wait(**kwargs)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"connected": False}, NOT_CONNECTED),
        ({"running": "scan"}, busy_note("scan", live=False)),
        ({"running": "live", "live": True}, busy_note("live", live=True)),
        ({"dry_run": True}, CALIBRATE_DRY_RUN),
        ({"objective": None}, CALIBRATE_NOT_KNOWN),
        ({"objective": "  "}, CALIBRATE_NOT_KNOWN),
        ({"objective": _TEN_X}, calibrate_is_the_plates(_TEN_X)),
        # Two at once: the first in D12's order wins.
        ({"connected": False, "running": "scan"}, NOT_CONNECTED),
        ({"running": "scan", "dry_run": True}, busy_note("scan", live=False)),
        ({"dry_run": True, "objective": None}, CALIBRATE_DRY_RUN),
        ({"objective": None, "plate_objective": ""}, CALIBRATE_NOT_KNOWN),
        ({}, None),
    ],
)
def test_the_calibrate_wait_gives_the_first_reason_in_its_order(
    changes: dict[str, Any], expected: str | None
) -> None:
    assert _wait(**changes) == expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        # The light is the last reason in D12's order (D20): every other one comes first.
        ({"light_on": False}, CALIBRATE_LIGHT_OFF),
        ({"light_on": False, "connected": False}, NOT_CONNECTED),
        ({"light_on": False, "running": "scan"}, busy_note("scan", live=False)),
        ({"light_on": False, "dry_run": True}, CALIBRATE_DRY_RUN),
        ({"light_on": False, "objective": None}, CALIBRATE_NOT_KNOWN),
        ({"light_on": False, "objective": " "}, CALIBRATE_NOT_KNOWN),
        ({"light_on": False, "objective": _TEN_X}, calibrate_is_the_plates(_TEN_X)),
        # A light that is on, or one nobody can read, refuses nothing: the tool decides.
        ({"light_on": True}, None),
        ({"light_on": None}, None),
    ],
)
def test_calibrate_wait_says_the_light_is_off_last_and_refuses_nothing_for_an_unknown_light(
    changes: dict[str, Any], expected: str | None
) -> None:
    assert _wait(**changes) == expected


def test_calibrate_wait_has_no_default_for_the_light() -> None:
    # Required, so that the view, the controller and the Stand screen cannot forget it (FM-169).
    with pytest.raises(TypeError, match="light_on"):
        calibrate_wait(  # type: ignore[call-arg]
            connected=True,
            running=None,
            live=False,
            dry_run=False,
            objective=_FORTY_X,
            plate_objective=_TEN_X,
        )


@pytest.mark.parametrize(
    ("light_on", "auto_shutter", "expected"),
    [
        # A light that reads off while the core opens the shutter around the exposure is unknown:
        # the Ti2-E's lamp is the core's shutter, closed at rest (the review of #557).
        (False, True, None),
        # Without auto-shutter, or when it could not be read, a light that reads off stands.
        (False, False, False),
        (False, None, False),
        # A light that reads on or is unknown is what it is, whatever the auto-shutter says.
        (True, True, True),
        (True, False, True),
        (True, None, True),
        (None, True, None),
        (None, False, None),
        (None, None, None),
    ],
)
def test_calibration_light_is_unknown_for_a_light_off_under_auto_shutter_and_as_read_otherwise(
    light_on: bool | None, auto_shutter: bool | None, expected: bool | None
) -> None:
    assert calibration_light(light_on, auto_shutter) is expected
    # What the controller and the view hand to calibrate_wait refuses exactly when it is False.
    refused = _wait(light_on=calibration_light(light_on, auto_shutter))
    assert (refused == CALIBRATE_LIGHT_OFF) is (expected is False)


@pytest.mark.parametrize(
    ("plate", "objective", "expected"),
    [
        # The block is offered (the plate is registered from its walls) and the objective is
        # known and is not the plate's: the one case that goes to a document of its own.
        (_TEN_X, _FORTY_X, True),
        (_TEN_X, _TWENTY_X, True),
        # The plate's own objective: its calibration is the run document's.
        (_TEN_X, _TEN_X, False),
        # No offer: no registration from walls to protect.
        (None, _FORTY_X, False),
        (None, None, False),
        # An objective not known, or an empty position: the destination cannot be chosen.
        (_TEN_X, None, False),
        (_TEN_X, "", False),
        (_TEN_X, "  ", False),
        # A blank plate objective is still an offer: only ``None`` is no offer.
        ("", _FORTY_X, True),
    ],
)
def test_calibration_goes_to_a_document_only_when_the_block_is_offered_and_the_objective_is_known_and_not_the_plates(
    plate: str | None, objective: str | None, expected: bool
) -> None:
    assert calibration_goes_to_a_document(plate, objective) is expected


def test_calibration_at_its_position_is_none_without_labels_false_past_them_and_for_no_objective() -> (
    None
):
    labels = (_FORTY_X, _TEN_X, "", _TWENTY_X)

    # No labels: they were not read, nothing is known (and not False).
    assert calibration_at_its_position((), 1, _FORTY_X) is None
    assert calibration_at_its_position((), 1, None) is None
    # The label at the position, counting empty positions: the index plus one.
    assert calibration_at_its_position(labels, 1, _FORTY_X) is True
    assert calibration_at_its_position(labels, 2, _TEN_X) is True
    assert calibration_at_its_position(labels, 4, _TWENTY_X) is True
    # Another label at that position.
    assert calibration_at_its_position(labels, 1, _TEN_X) is False
    assert calibration_at_its_position(labels, 4, _FORTY_X) is False
    # The last position is in range, the one after it is not (an off-by-one at len(labels)).
    assert calibration_at_its_position(labels, len(labels), _TWENTY_X) is True
    assert calibration_at_its_position(labels, len(labels) + 1, _TWENTY_X) is False
    assert calibration_at_its_position(labels, 0, _FORTY_X) is False
    # Position 0 and a negative one are no position, whatever Python's negative indexes say:
    # the last label is the 20X, and position 0 would read it.
    assert calibration_at_its_position(labels, 0, _TWENTY_X) is False
    assert calibration_at_its_position(labels, -1, _TWENTY_X) is False
    assert calibration_at_its_position(labels, -len(labels), _FORTY_X) is False
    # No objective, or a blank one, is at no position, an empty position's included.
    assert calibration_at_its_position(labels, 1, None) is False
    assert calibration_at_its_position(labels, 3, "") is False
    assert calibration_at_its_position(labels, 3, "  ") is False


def test_the_calibrate_label_names_the_objective_says_again_when_one_is_held_and_names_another_for_the_plates_or_none() -> (
    None
):
    another = "Calibrate the camera at another objective"

    assert calibrate_label(_FORTY_X, _TEN_X, again=False) == (
        f"Calibrate the camera at {_FORTY_X}"
    )
    assert calibrate_label(_FORTY_X, _TEN_X, again=True) == (
        f"Calibrate the camera again at {_FORTY_X}"
    )
    # Nothing to name: the button waits, and `again` has nothing to say.
    for objective in (None, "", "  ", _TEN_X):
        assert calibrate_label(objective, _TEN_X, again=False) == another
        assert calibrate_label(objective, _TEN_X, again=True) == another


def test_the_calibrate_texts_say_what_they_say() -> None:
    assert CALIBRATE_DRY_RUN == (
        "a dry run cannot measure the camera; run it without --dry-run"
    )
    assert CALIBRATE_NOT_KNOWN == (
        "the objective in place is not known: wait for the turret to stop turning, or "
        "check the stand"
    )
    assert CALIBRATE_NO_OFFER == (
        "the plate is not registered from its walls: calibrate with the Stand "
        "screen's button, into the run document"
    )
    assert CALIBRATE_NAMES_NONE == (
        "the button names no objective: wait until it does, then press it again"
    )
    assert CALIBRATE_LEAD == (
        "To calibrate the camera at an objective: turn the turret to it (Stand "
        "screen, Change objective), bring a well's wall into the image (the arrows, "
        "Live), then press the button. The stage moves by about a quarter of the "
        "field in X, then in Y, and comes back."
    )
    assert calibrate_is_the_plates(_TEN_X) == (
        f"{_TEN_X} is the objective the plate was registered with: its calibration "
        "is the run document's (Stand screen, Calibrate the camera); turn the turret "
        "to an objective to measure"
    )
    assert calibrate_changed(_FORTY_X, _TWENTY_X) == (
        f"the objective in place is {_TWENTY_X}, not {_FORTY_X} as the button said: "
        "press it again"
    )
    assert calibrate_position_unknown(_FORTY_X) == (
        f"the turret position of {_FORTY_X} is not known: its labels could not be "
        "read, two positions carry that label, or it is past position 12"
    )
    assert calibrate_replaces(_FORTY_X, "2026-11-17T10:20:30+00:00") == (
        "A new calibration replaces the one of 2026-11-17: if it fails or is stopped, "
        f"{_FORTY_X} has none until one succeeds."
    )
    assert calibrate_replaces(_FORTY_X, "at an unknown time") == (
        "A new calibration replaces the one of an unknown time: if it fails or is "
        f"stopped, {_FORTY_X} has none until one succeeds."
    )
    assert calibrations_held_heading("run/calibrations") == (
        "Camera calibrations the page holds, one document per objective, in "
        "run/calibrations:"
    )
    assert calibration_missing_line(_TWENTY_X) == f"{_TWENTY_X}: no calibration yet"
    assert calibration_note_line(_FORTY_X, "the pixel is square") == (
        f"  {_FORTY_X}: the pixel is square"
    )
    assert calibration_failed("the frame is dark or flat") == (
        "the last calibration failed: the frame is dark or flat"
    )
    # The Stand screen's line is the same words with a capital (m5 §22.2 D1), on one line.
    assert stand_calibration_failed("the frame is dark or flat") == (
        "The last calibration failed: the frame is dark or flat"
    )
    assert stand_calibration_failed("the frame\nis dark") == (
        "The last calibration failed: the frame is dark"
    )
    assert stand_calibration_failed("") == "The last calibration failed: "


def test_the_stand_routed_note_the_folder_note_and_the_measured_other_text_are_the_designs() -> (
    None
):
    """Whole texts of m5 §18.6.3, written out here and not read back from the code (FM-45)."""
    assert OFFSETS_MEASURE_LABEL == "Measure the offsets"
    assert CALIBRATE_LIGHT_OFF == (
        "the light reads off, so the camera's frame would be dark and the "
        "calibration would fail: switch it on (Light on, in the drive bar), then "
        "press the button again"
    )
    assert CALIBRATE_REUSE_ROUTED == (
        "the form's from_run_doc reuses a calibration into the run document, where "
        "it would replace the plate's: clear it to calibrate into the objective's own "
        "document, or use Run camera-calibration in the expert view"
    )
    assert calibrate_stand_routed(_TEN_X, _FORTY_X, "position-1.json") == (
        f"The plate was registered with {_TEN_X}: a calibration at {_FORTY_X} here is "
        "kept in its own document, calibrations/position-1.json, and the plate's "
        "calibration stays as it is. Lay out the fields (Wells screen) and Measure "
        "the offsets between objectives (Plate screen) use it."
    )
    assert calibrate_command_folder_note("run/calibrations") == (
        "The folder run/calibrations must exist first: the page makes it, a "
        "terminal does not."
    )
    assert calibrate_measured_other(_FORTY_X, _TWENTY_X, "position-1.json") == (
        f"it measured {_TWENTY_X}, not {_FORTY_X} as the button said (the turret was "
        f"turned during the run): position-1.json holds a calibration of {_TWENTY_X} "
        f"and {_FORTY_X} has none; put {_FORTY_X} in place and press the button again"
    )
    # A record that names no objective: the words say so.
    assert calibrate_measured_other(_FORTY_X, None, "position-1.json") == (
        "it measured an unknown objective, not "
        f"{_FORTY_X} as the button said (the turret was turned during the run): "
        "position-1.json holds a calibration of an unknown objective and "
        f"{_FORTY_X} has none; put {_FORTY_X} in place and press the button again"
    )


def test_the_plan_calibration_texts_are_the_designs_whole() -> None:
    """Whole texts of m5 §18.6.3 "Added by #617's plan", written out here (FM-45)."""
    assert PLAN_CALIBRATION_NONE_MADE == "none has been made"
    assert plan_calibration_used(
        _FORTY_X, "position-1.json", "2026-11-25T10:20:30+00:00"
    ) == (
        f"Lay out the fields uses the camera calibration of {_FORTY_X} made on "
        "2026-11-25 (calibrations/position-1.json): the camera's turn and mirror are "
        "measured, not assumed."
    )
    # A time that is not an ISO one is an unknown time, as the held list says it.
    assert plan_calibration_used(_FORTY_X, "position-1.json", "at an unknown time") == (
        f"Lay out the fields uses the camera calibration of {_FORTY_X} made at an "
        "unknown time (calibrations/position-1.json): the camera's turn and mirror "
        "are measured, not assumed."
    )
    assert plan_calibration_used(_FORTY_X, "position-1.json", "") == (
        f"Lay out the fields uses the camera calibration of {_FORTY_X} made at an "
        "unknown time (calibrations/position-1.json): the camera's turn and mirror "
        "are measured, not assumed."
    )
    assert plan_calibration_missing(_FORTY_X, PLAN_CALIBRATION_NONE_MADE) == (
        f"No camera calibration of {_FORTY_X} can be used (none has been made): Lay "
        "out the fields assumes the profile's pixel size, rotation 0 and no mirror, "
        "and the positions of cells in the fields will be wrong if the camera is "
        "turned or mirrored. To get one: Stand screen, Calibrate the camera, with "
        f"{_FORTY_X} in place."
    )
    assert plan_calibration_other_objective("position-1.json", _TEN_X) == (
        f"calibrations/position-1.json holds a calibration of {_TEN_X}"
    )
    assert plan_calibration_other_objective("position-1.json", None) == (
        "calibrations/position-1.json holds a calibration of an unknown objective"
    )
    assert plan_calibration_frame(
        _FORTY_X, "position-1.json", (1024, 1024), (384, 512)
    ) == (
        f"the camera calibration of {_FORTY_X} in calibrations/position-1.json is of "
        "a frame of 1024 by 1024 px, the camera gives 384 by 512 px (binning or the "
        f"region changed): calibrate the camera at {_FORTY_X} again (Stand screen, "
        "Calibrate the camera)"
    )
    # The transposed frame is another frame, and the words keep the order they were given.
    assert "512 by 384 px, the camera gives 384 by 512 px" in plan_calibration_frame(
        _FORTY_X, "position-1.json", (512, 384), (384, 512)
    )


def test_the_plan_calibration_texts_are_one_line_for_a_label_with_a_line_break() -> (
    None
):
    label = "40X lens\nwith a break\r\nand [markup] *stars*"
    texts = (
        plan_calibration_used(label, "position-1.json", "2026-11-25T10:20:30+00:00"),
        plan_calibration_missing(label, "why\nnot"),
        plan_calibration_other_objective("position-1.json", label),
        plan_calibration_frame(label, "position-1.json", (1, 2), (3, 4)),
    )

    for text in texts:
        assert "\n" not in text
        assert "\r" not in text
    # Plain text for the view (FM-139): the markup characters are kept as typed.
    assert "[markup] *stars*" in texts[0]


def test_calibration_denied_names_the_folder_and_the_reason_and_never_a_file() -> None:
    assert calibration_denied(
        "run/calibrations", "PermissionError: Permission denied"
    ) == (
        "run/calibrations: the page cannot look into this folder (PermissionError: "
        "Permission denied); which objectives have a calibration is not known"
    )
    # One line whatever the folder's or the reason's text holds (FM-139).
    odd = calibration_denied("a\nb", "PermissionError: no\r\nright")
    assert "\n" not in odd
    assert "\r" not in odd
    assert "position-" not in calibration_denied(
        "run/calibrations", "PermissionError: Permission denied"
    )


def test_the_held_calibration_lines_say_objective_date_problem_and_why_a_document_is_left_out() -> (
    None
):
    when = "2026-11-17T10:20:30+00:00"

    def line(**changes: Any) -> str:
        kwargs: dict[str, Any] = {
            "position": 1,
            "label": _FORTY_X,
            "objective": _FORTY_X,
            "measured": when,
            "problem": None,
            "plate_objective": _TEN_X,
            "at_position": True,
        }
        kwargs.update(changes)
        return calibration_held_line(**kwargs)

    assert line() == f"{_FORTY_X}: calibrated on 2026-11-17"
    assert line(measured="at an unknown time") == (
        f"{_FORTY_X}: calibrated, at an unknown time"
    )
    assert line(objective=_TEN_X) == (
        f"{_TEN_X}: calibrated on 2026-11-17; the plate's objective, not measured "
        "from itself"
    )
    # Not known whether the document is its position's: nothing is said about it.
    assert line(at_position=None) == f"{_FORTY_X}: calibrated on 2026-11-17"
    # A problem is the position's line, with the label when the page knows it.
    problem = "in c/position-4.json: the last run of camera-calibration ended stopped"
    assert line(position=4, objective=None, measured="", problem=problem) == (
        f"position 4 ({_FORTY_X}): {problem}"
    )
    for unknown in (None, "", "  "):
        assert line(position=4, label=unknown, objective=None, problem=problem) == (
            f"position 4: {problem}"
        )


def test_a_document_that_is_not_at_its_position_has_a_line_of_its_own_with_or_without_a_label() -> (
    None
):
    when = "2026-11-17T10:20:30+00:00"

    def line(**changes: Any) -> str:
        kwargs: dict[str, Any] = {
            "position": 1,
            "label": _FORTY_X,
            "objective": _TEN_X,
            "measured": when,
            "problem": None,
            "plate_objective": _TEN_X,
            "at_position": False,
        }
        kwargs.update(changes)
        return calibration_held_line(**kwargs)

    tail = (
        "not of the objective at this position now (the turret was turned during a "
        "run, or relabelled since): "
    )
    # The position and its label say where, the document's objective says what it holds, and
    # the way out names the objective that file should hold.
    assert line() == (
        f"position 1 ({_FORTY_X}): the document holds a calibration of {_TEN_X}, "
        f"{tail}calibrate {_FORTY_X} again, or type the documents to use"
    )
    # The old suffix of the usable line is gone, and so is its date: it is not that line.
    assert "not on the turret now" not in line()
    assert "calibrated on" not in line()
    # A position whose label the page does not know: no objective to calibrate again.
    for unknown in (None, "", "  "):
        assert line(position=4, label=unknown, objective=_FORTY_X) == (
            f"position 4: the document holds a calibration of {_FORTY_X}, "
            f"{tail}type the documents to use"
        )
    # A record that names no objective says so.
    assert line(objective=None) == (
        "position 1 (" + _FORTY_X + "): the document holds a calibration of an unknown "
        f"objective, {tail}calibrate {_FORTY_X} again, or type the documents to use"
    )
    # A problem still has the position's own line, whatever ``at_position`` says.
    problem = "in c/position-4.json: the last run of camera-calibration ended stopped"
    assert line(position=4, objective=None, measured="", problem=problem) == (
        f"position 4 ({_FORTY_X}): {problem}"
    )


def test_the_held_lines_are_one_line_each_for_a_label_with_a_line_break() -> None:
    label = "40X lens\nwith a break\r\nand [markup] *stars*"
    words = "it failed\nbecause of a break"
    lines = (
        calibration_held_line(
            1,
            label,
            label,
            "2026-11-17T10:20:30+00:00",
            None,
            plate_objective=_TEN_X,
            at_position=False,
        ),
        calibration_held_line(
            2,
            label,
            None,
            "",
            calibration_failed(words),
            plate_objective=_TEN_X,
            at_position=None,
        ),
        calibration_missing_line(label),
        calibration_note_line(label, "an assumed\nline"),
        calibrate_label(label, _TEN_X, again=True),
        calibrate_changed(label, label),
        calibrate_position_unknown(label),
        calibrate_replaces(label, "2026-11-17T10:20:30+00:00"),
        calibrate_stand_routed(label, label, "position-1.json"),
        calibrate_command_folder_note(label),
        calibrate_measured_other(label, label, "position-1.json"),
        calibration_denied(label, label),
        calibrations_held_heading(label),
        calibrate_is_the_plates(label),
    )

    for text in lines:
        assert "\n" not in text
        assert "\r" not in text
    # The markup characters are kept as typed: the view draws plain text, never markdown.
    assert "[markup] *stars*" in lines[0]


def test_error_words_also_drop_the_calibration_tools_name_and_leave_other_text_alone() -> (
    None
):
    words = "the frame of Nikon 40X is dark or flat: check the light path"

    assert error_words(f"PluginRunError: camera-calibration: {words}") == words
    assert error_words(f"camera-calibration: {words}") == words
    assert error_words(f"ValueError: camera-calibration: {words}") == words
    # One tool's name at most, and only at the start.
    assert error_words("PluginRunError: objective-offsets: camera-calibration: x") == (
        "camera-calibration: x"
    )
    assert error_words(f"see camera-calibration: {words}") == (
        f"see camera-calibration: {words}"
    )
    assert error_words("camera-calibrationx: it") == "camera-calibrationx: it"


# --- the lock notes after Measure the time and before the scan (m5 §20, #546) --------------


def _measured_chain(
    *,
    label: str = MEASURE_THE_TIME,
    current: str | None = None,
    plugins: tuple[str, ...] = ("timing",),
) -> ChainStatus:
    """*Measure the time*'s chain after its one tool ended ``done`` (``current`` ``None``)."""
    ended = (ChainEnd("timing", "done", "Timing done"),)
    return ChainStatus(plugins, ended, current, label)


def test_the_lock_notes_are_m5_section_20_3s_words() -> None:
    """Whole texts, written out here and not read back from the constants (FM-45)."""
    assert RELOCK_NOTE == (
        "The autofocus is not locked: Autofocus on, under the image, locks it again at the "
        "offset now set if this page has read that offset locked since Connect. After a "
        "Disconnect, or from smc autofocus engage in a terminal, no lock has been read, and "
        "the engage is refused."
    )
    assert SCAN_NEEDS_A_LOCK == (
        "Autofocus is required and it is not locked, so the scan will refuse to start: "
        "press Autofocus on, under the image, first, if this page has read the offset now "
        "set locked since Connect."
    )
    assert SCAN_WITHOUT_A_LOCK == (
        "The autofocus is not locked, so the scan will not use it (if-engaged): press "
        "Autofocus on, under the image, to scan with it, if this page has read the offset "
        "now set locked since Connect."
    )


def test_relock_note_is_said_only_after_measure_the_time_with_the_autofocus_not_locked() -> (
    None
):
    """One input changed at a time from the one full case: each other value says nothing (D2, D4, D40)."""

    def note(
        chain: ChainStatus | None,
        *,
        locked: bool | None = False,
        dry_run: bool = False,
        halted: bool = False,
    ) -> str:
        return relock_note(
            chain, autofocus_locked=locked, dry_run=dry_run, halted=halted
        )

    full = note(_measured_chain())
    assert full == RELOCK_NOTE

    assert note(_measured_chain(label=RUN_ALL_LABEL)) == ""
    assert note(_measured_chain(current="timing")) == ""
    assert note(_measured_chain(), locked=True) == ""
    assert note(_measured_chain(), locked=None) == ""
    assert note(_measured_chain(), dry_run=True) == ""
    assert note(None) == ""
    assert note(_measured_chain(plugins=())) == ""
    # D40: after a STOP the autofocus reads off and the stand is halted, so *Autofocus on*
    # would be refused (``press Resume to continue``): the one input changed alone says nothing.
    assert note(_measured_chain(), halted=True) == ""


@pytest.mark.parametrize(
    ("mode", "said"),
    [
        pytest.param("required", SCAN_NEEDS_A_LOCK, id="required"),
        pytest.param("if-engaged", SCAN_WITHOUT_A_LOCK, id="if-engaged"),
        pytest.param("off", "", id="off"),
        pytest.param("bogus", "", id="a value the tool would refuse"),
        pytest.param("", "", id="empty"),
        pytest.param("Required", "", id="the tool's words are lower case"),
    ],
)
def test_scan_lock_note_names_autofocus_on_for_required_and_if_engaged_only(
    mode: str, said: str
) -> None:
    assert (
        scan_lock_note(mode, autofocus_locked=False, dry_run=False, halted=False)
        == said
    )


@pytest.mark.parametrize("mode", ["required", "if-engaged"])
def test_scan_lock_note_says_nothing_for_a_lock_that_reads_locked_or_unknown_or_in_a_dry_run_or_while_halted(
    mode: str,
) -> None:
    assert (
        scan_lock_note(mode, autofocus_locked=True, dry_run=False, halted=False) == ""
    )
    assert (
        scan_lock_note(mode, autofocus_locked=None, dry_run=False, halted=False) == ""
    )
    assert (
        scan_lock_note(mode, autofocus_locked=False, dry_run=True, halted=False) == ""
    )
    # D40: a halted stand refuses the click the note advises, so the note is silent; the
    # one input changed alone, from the full case below it.
    assert (
        scan_lock_note(mode, autofocus_locked=False, dry_run=False, halted=True) == ""
    )
    assert (
        scan_lock_note(mode, autofocus_locked=False, dry_run=False, halted=False) != ""
    )
