"""The next step's pure pieces: which slot is next, and the command that makes its call (m5 §8.3, #293).

No Panel and no stand. The command is read back with ``shlex.split``, the way a
shell reads it, never compared with the string the code builds (FM-45).
"""

from __future__ import annotations

import json
import math
import shlex
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.geometry import Matrix2
from smc.plugins.presets import Preset, PresetCheck, PresetFile
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    PlateResult,
    ScanPlanResult,
    WallFrame,
    WallImages,
    WellFit,
    WellRef,
)
from smc.ui.controller import (
    _CHECKLIST,
    CheckItem,
    LoadedPreset,
    MapPose,
    PlanLayer,
    RunClock,
    ScanSummary,
    WellView,
)
from smc.ui.guide import (
    CAPTURE_WELLS,
    DRY_RUN_NO_FRAME,
    END_LIVE_LABEL,
    GOTO_NOT_DURING_LIVE,
    LAY_OUT_FIRST,
    MAP_FIELDS_UP_TO,
    NO_CALIBRATION_FOR_SIDES,
    ORIENTATION_IN_STAGE_TERMS,
    OTHER_WAYS_TO_REGISTER,
    PLATE_NOT_CHOSEN,
    REGISTER_FIRST,
    SCREEN_LEADS,
    SCREEN_TITLES,
    SELECT_WELLS,
    STEP_OF_SLOT,
    STEP_TITLES,
    STEPS,
    WALLS_DRY_RUN,
    WALLS_MAX_CENTRES,
    WALLS_MIN_CENTRES,
    WALLS_NEED_CALIBRATION,
    CalibrationNote,
    NextStep,
    StepName,
    StepState,
    a1_corner,
    busy_note,
    capture_centre_label,
    capture_wall_label,
    centre_rows,
    centres_text,
    command_line,
    continue_label,
    drive_image_note,
    duration_text,
    field_source_line,
    forget_label,
    image_sides_note,
    known_text,
    lay_out_wait,
    map_title,
    needs_calibration,
    next_slot,
    objective_line,
    opening_step,
    orientation_label,
    plain_default,
    plan_selection_note,
    plan_title,
    plate_from_line,
    preset_differences,
    preset_note_text,
    register_replace_note,
    registered_line,
    replace_note,
    scan_lead,
    scan_plan_card_lines,
    scan_progress_lines,
    scan_title,
    scan_wait,
    scanned_line,
    side_words,
    stage_label,
    step_button_label,
    step_states,
    suggested_centres,
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
    assert drive_image_note(None) == (
        "Which way the image moves for an arrow is not known: run camera-calibration "
        "(Set up)."
    )


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


def test_replace_note_names_the_time_the_method_and_the_wells() -> None:
    started = datetime(2026, 11, 3, 14, 20, 41, tzinfo=timezone.utc)
    # The record's start in the PC's local time, to the minute: computed here the
    # way a reader's clock would show it, never a literal that depends on the zone.
    local = started.astimezone().strftime("%Y-%m-%d %H:%M")
    tail = (
        "A failed run leaves no plate; the captures stay, so Forget the wrong one "
        "and Run again to get a plate back."
    )

    one = replace_note(_plate_result("seed", 1), started, keeps_captures=True)
    assert one == f"Run replaces the plate registered {local} (seed, 1 well). {tail}"
    two = replace_note(_plate_result("references", 2), started, keeps_captures=True)
    assert two == (
        f"Run replaces the plate registered {local} (references, 2 wells). {tail}"
    )
    # A method the registration names is the result's own, never a paraphrase.
    assert "(wall-images, 3 wells)" in replace_note(
        _plate_result("wall-images", 3), started, keeps_captures=True
    )


def test_replace_note_converts_a_start_that_is_not_in_utc_to_the_pcs_time() -> None:
    # A start with an offset that is not zero: 14:20 at +05:00 is 09:20 UTC. On a PC
    # whose zone is UTC (a CI runner's) a note that skips the conversion says 14:20,
    # so the test fails there, where a UTC start could not (it reads the same either
    # way). The expected text is the PC's own conversion of this instant.
    started = datetime(2026, 11, 3, 14, 20, tzinfo=timezone(timedelta(hours=5)))
    assert started.astimezone(timezone.utc).strftime("%H:%M") == "09:20"
    local = started.astimezone().strftime("%Y-%m-%d %H:%M")
    note = replace_note(_plate_result("seed", 1), started, keeps_captures=True)
    assert note.startswith(
        f"Run replaces the plate registered {local} (seed, 1 well). "
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
    local = started.astimezone().strftime("%Y-%m-%d %H:%M")
    note = replace_note(_plate_result("references", 3), started, keeps_captures=False)
    assert note == (
        f"Run replaces the plate registered {local} (references, 3 wells). "
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


def test_scan_plan_card_says_what_to_run_when_the_estimate_is_unknown() -> None:
    unknown = scan_plan_card_lines(_scan_plan(estimate_s=None))[-1]
    assert unknown == (
        "estimate: unknown, no usable timing: run timing, then scan-plan again"
    )
    # An unknown is not a zero (FM-170): 0.0 is an estimate, and says so.
    zero = scan_plan_card_lines(_scan_plan(estimate_s=0.0))[-1]
    assert zero == "estimate: about 0 s (the moves, refocus and snaps of the timings)"


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
    # The Ti2-E: +Y is the back and +X the operator's right.
    assert a1_corner("not-turned", "+Y", "+X") == "the back left"
    assert a1_corner("turned-180", "+Y", "+X") == "the front right"
    assert orientation_label("not-turned", "+Y", "+X") == "A1 at the back left"
    assert (
        orientation_label("turned-180", "+Y", "+X")
        == f"Turned 180{DEG}: A1 at the front right"
    )
    # A stand whose back is +X: a word tied to an axis, or to the sign of Y, would say
    # the Ti2-E's corner here (FM-73). A1 at -X +Y is the front right of it, and +X -Y
    # the back left.
    assert a1_corner("not-turned", "+X", "+Y") == "the front right"
    assert a1_corner("turned-180", "+X", "+Y") == "the back left"
    # A back of -Y and a right of -X: every sign flipped from the Ti2-E.
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
    assert states["stand"].summary == "demo-sample, preset weekly"
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


def test_a_summary_is_one_line_of_at_most_forty_characters() -> None:
    long = "a-stand-with-a-very-long-profile-name-indeed"
    summary = _states(_checklist(), stand=long, preset="a-preset-with-a-long-name")[
        "stand"
    ].summary
    assert len(summary) <= 40
    assert summary.startswith("a-stand-with-a-very-long")
    assert "\n" not in summary
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
        == "Registered: residual 121.6 µm, turned 0.10°"
    )
    # A seed measures no residual: an unknown is not a zero (CLAUDE.md).
    assert (
        registered_line(result(None, -0.456))
        == "Registered: residual not measured, turned -0.46°"
    )
    assert (
        registered_line(result(0.0, 0.0)) == "Registered: residual 0.0 µm, turned 0.00°"
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
) -> PlanLayer:
    return PlanLayer(
        count=count,
        wells=wells,
        fov_um=(512.0, 512.0),
        fields=(),
        objective=objective,
        estimate_s=estimate_s,
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
        "Time: not known (no timing measured; the expert view's timing tool measures "
        "it, then lay out the fields again)",
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
