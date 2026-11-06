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

from smc.plugins.presets import Preset, PresetCheck, PresetFile
from smc.plugins.slots import PlateResult
from smc.ui.controller import CheckItem, LoadedPreset
from smc.ui.guide import (
    CAPTURE_WELLS,
    DRY_RUN_NO_FRAME,
    END_LIVE_LABEL,
    SELECT_WELLS,
    NextStep,
    busy_note,
    capture_centre_label,
    capture_wall_label,
    command_line,
    forget_label,
    next_slot,
    preset_differences,
    preset_note_text,
    replace_note,
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
        "map and press Capture centre; a second well, far from the first, fits the "
        "rotation too"
    )


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
    assert capture_wall_label("B3", "left") == "Capture B3 left wall"
    assert capture_wall_label("B3", "bottom") == "Capture B3 bottom wall"
    assert forget_label("B3") == "Forget B3"
    # With no well the labels are the plain three: there is nothing to name.
    assert capture_centre_label(None) == "Capture centre"
    assert capture_wall_label(None, "left") == "Capture wall"
    assert forget_label(None) == "Forget"


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

    one = replace_note(_plate_result("seed", 1), started)
    assert one == f"Run replaces the plate registered {local} (seed, 1 well). {tail}"
    two = replace_note(_plate_result("references", 2), started)
    assert two == (
        f"Run replaces the plate registered {local} (references, 2 wells). {tail}"
    )
    # A method the registration names is the result's own, never a paraphrase.
    assert "(wall-images, 3 wells)" in replace_note(
        _plate_result("wall-images", 3), started
    )


def test_replace_note_converts_a_start_that_is_not_in_utc_to_the_pcs_time() -> None:
    # A start with an offset that is not zero: 14:20 at +05:00 is 09:20 UTC. On a PC
    # whose zone is UTC (a CI runner's) a note that skips the conversion says 14:20,
    # so the test fails there, where a UTC start could not (it reads the same either
    # way). The expected text is the PC's own conversion of this instant.
    started = datetime(2026, 11, 3, 14, 20, tzinfo=timezone(timedelta(hours=5)))
    assert started.astimezone(timezone.utc).strftime("%H:%M") == "09:20"
    local = started.astimezone().strftime("%Y-%m-%d %H:%M")
    note = replace_note(_plate_result("seed", 1), started)
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
    note = replace_note(_plate_result("seed", 1), started)
    assert note == (
        "Run replaces the plate registered 0001-01-01T00:00+00:00 (seed, 1 well). "
        "A failed run leaves no plate; the captures stay, so Forget the wrong one "
        "and Run again to get a plate back."
    )
