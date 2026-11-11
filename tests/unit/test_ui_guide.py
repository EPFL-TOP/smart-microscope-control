"""The next step's pure pieces: which slot is next, and the command that makes its call (m5 §8.3, #293).

No Panel and no stand. The command is read back with ``shlex.split``, the way a
shell reads it, never compared with the string the code builds (FM-45).
"""

from __future__ import annotations

import json
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
from smc.ui.controller import CheckItem, LoadedPreset, MapPose, PlanLayer, WellView
from smc.ui.guide import (
    CAPTURE_WELLS,
    DRY_RUN_NO_FRAME,
    END_LIVE_LABEL,
    MAP_FIELDS_UP_TO,
    NO_CALIBRATION_FOR_SIDES,
    SELECT_WELLS,
    WALLS_DRY_RUN,
    WALLS_MAX_CENTRES,
    WALLS_MIN_CENTRES,
    WALLS_NEED_CALIBRATION,
    NextStep,
    busy_note,
    capture_centre_label,
    capture_wall_label,
    centres_text,
    command_line,
    drive_image_note,
    duration_text,
    forget_label,
    image_sides_note,
    map_title,
    next_slot,
    plan_title,
    preset_differences,
    preset_note_text,
    replace_note,
    scan_plan_card_lines,
    stage_label,
    wall_touches_note,
    walls_card_lines,
    walls_centres_problem,
    walls_square_wells,
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
    assert map_title("96-well", MapPose("seed", seed_well="B3")) == (
        "96-well: placed from one well (B3), not registered: "
        f"the rotation is assumed 0{DEG}, not measured"
    )
    # A seed with no reference well says so without an empty pair of brackets.
    assert map_title("96-well", MapPose("seed")) == (
        "96-well: placed from one well, not registered: "
        f"the rotation is assumed 0{DEG}, not measured"
    )
    assert map_title(
        "24-well", MapPose("estimated", holder_centre_um=(400.0, -325.0))
    ) == (
        "24-well: estimated from the stand's holder, not registered: "
        f"centred at stage (400, -325) {UM}, not turned, A1 at the top left of this map"
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


def test_stage_label_takes_the_first_well_in_the_views_order() -> None:
    # Wells do not overlap on a real plate; if a view's did, its order decides.
    twin = _well("Z9", (0.0, 0.0))
    assert stage_label((WELLS[0], twin), MapPose("registered"), (0.0, 0.0)) == "over A1"
    assert stage_label((twin, WELLS[0]), MapPose("registered"), (0.0, 0.0)) == "over Z9"
