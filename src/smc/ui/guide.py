"""What the plate page says to do next: a value and a command, no widget (m5 §8.3).

The guide is a function of the run document's checklist, the page's selection and
captures, and the profile; it keeps no state, and the page draws it above its tools.
It holds no Panel and no device, so every sentence it builds is tested as text. Its
one promise is the one the page's *Run* button keeps: the pairs the step shows are the
pairs it runs, and ``command_line`` is the same call typed in a terminal (principle 5).
"""

from __future__ import annotations

import ast
import html
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import numpy.typing as npt

from smc.plugins.frame_signal import carries_signal, saturated
from smc.plugins.frames import image_direction_of_stage_step, stage_side_of_image_side
from smc.plugins.plate_registration.algorithm import GOOD_RESIDUAL_UM, suggested_refs
from smc.plugins.slots import WellGeometry

if TYPE_CHECKING:
    from datetime import datetime

    from smc.plugins.geometry import Matrix2
    from smc.plugins.slots import PlateResult, ScanPlanResult, WellRef
    from smc.plugins.slots.plate import PlateOrientation
    from smc.ui.controller import (
        ChainStatus,
        CheckItem,
        LoadedPreset,
        MapPose,
        PlanLayer,
        PoseKind,
        ResultCard,
        RunClock,
        ScanEnd,
        ScanSummary,
        WellView,
    )

__all__ = [
    "AGAIN_NOTE",
    "AUTOFOCUS_WORDS",
    "CAMERA_NOT_CALIBRATED_NOTE",
    "CAPTURE_FAR",
    "CAPTURE_SAME_PLACE",
    "CAPTURE_WELLS",
    "COVERAGE_WORDS",
    "DISCONNECT_RELEASES",
    "DRY_RUN_NO_FRAME",
    "END_LIVE_LABEL",
    "FILES_HEADING",
    "FRAME_DARK",
    "FRAME_SATURATED",
    "GOTO_ESTIMATED",
    "GOTO_NOT_DURING_LIVE",
    "GOTO_ROW_AT",
    "GOTO_ROW_AT_ESTIMATE",
    "GOTO_ROW_GOING",
    "GOTO_UNPLACED",
    "GOTO_WALL_NEEDS_REGISTRATION",
    "LAY_OUT_FIRST",
    "MAP_FIELDS_UP_TO",
    "MEASURE_THE_TIME",
    "NOTHING_LEFT",
    "NOT_CONNECTED",
    "NO_CALIBRATION_FOR_SIDES",
    "NO_FILES_YET",
    "ORIENTATION_IN_STAGE_TERMS",
    "OTHER_WAYS_TO_REGISTER",
    "PLATE_NOT_CHOSEN",
    "REGISTER_ACROSS_A_TURN",
    "REGISTER_FIRST",
    "REGISTER_OBJECTIVES_DIFFER",
    "RESULT_CHOICES",
    "RESULT_NAMES",
    "RESULT_OPTIONAL_READS",
    "RESULT_READS",
    "RESULT_SHORT",
    "RESULT_TOOLS",
    "RUN_ALL_LABEL",
    "SAMPLE_REGISTERS",
    "SCAN_CHECKPOINT_KEPT",
    "SCAN_CONTINUE",
    "SCAN_FAILED",
    "SCAN_FAILED_UNCOUNTED",
    "SCAN_FIRST",
    "SCAN_HALTED",
    "SCAN_REPLACES",
    "SCAN_START_AGAIN",
    "SCAN_START_AGAIN_CONFIRM",
    "SCAN_START_AGAIN_CONFIRM_UNCOUNTED",
    "SCAN_STOPPED",
    "SCAN_STOPPED_UNCOUNTED",
    "SCREEN_LEADS",
    "SCREEN_TITLES",
    "SELECTION_KEPT_ITS_NAMES",
    "SELECT_WELLS",
    "STEPS",
    "STEP_OF_SLOT",
    "STEP_TITLES",
    "TIME_NOT_KNOWN",
    "TOOL_TITLES",
    "WALLS_DRY_RUN",
    "WALLS_MAX_CENTRES",
    "WALLS_MIN_CENTRES",
    "WALLS_NEED_CALIBRATION",
    "CalibrationNote",
    "CentreRow",
    "GotoAnswer",
    "NextStep",
    "ResultRow",
    "StepName",
    "StepState",
    "a1_corner",
    "again_note",
    "busy_note",
    "capture_centre_label",
    "capture_note",
    "capture_wall_label",
    "centre_rows",
    "centres_text",
    "chain_line",
    "colour_bar_title",
    "command_line",
    "continue_label",
    "drive_image_note",
    "duration_text",
    "field_source_line",
    "files_html",
    "files_lines",
    "forget_label",
    "frame_hint",
    "goto_row_text",
    "image_sides_note",
    "known_text",
    "lay_out_wait",
    "local_time",
    "map_frame",
    "map_title",
    "needs_calibration",
    "next_slot",
    "objective_line",
    "opening_step",
    "orientation_label",
    "plain_default",
    "plan_selection_note",
    "plan_title",
    "plate_from_line",
    "plate_rows_note",
    "preset_differences",
    "preset_note_text",
    "register_objective_refusal",
    "register_replace_note",
    "registered_line",
    "replace_note",
    "result_row",
    "results_left",
    "run_all_wait",
    "scan_actions_wait",
    "scan_again_confirm",
    "scan_end_lines",
    "scan_lead",
    "scan_plan_card_lines",
    "scan_progress_lines",
    "scan_title",
    "scan_wait",
    "scanned_line",
    "side_words",
    "stage_label",
    "step_button_label",
    "step_states",
    "suggested_centres",
    "tool_title",
    "wall_touches_note",
    "walls_card_lines",
    "walls_centres_problem",
    "walls_square_wells",
    "wells_from_line",
    "wells_known_lines",
    "wells_text",
]

#: What the scan plan waits for. The page cannot choose wells for the operator:
#: which wells to image is the experiment (m5 §8.6).
SELECT_WELLS = (
    "select the wells to scan: click them on the map, or type a range in Wells "
    "(A1:D6, C*, all)"
)

#: What plate registration waits for on a stand that has no sample to register from:
#: the plate's place is measured by centring wells, never guessed. Three centres fit the
#: plate and check each other; two fit it and check only their spacing (m5 §15 D7).
CAPTURE_WELLS = (
    "centre a well under the objective (Live and the Drive pad), click it on the "
    "map and press Capture centre; three wells far apart fit the plate and check "
    "each other"
)

#: Why a tool that takes the plate does not run while the page shows the placeholder
#: (m5 §11.3): the page opens on a 96-well that nothing named, and a registration run
#: with it on a 24-well plate puts A1 two pitches wrong, with no residual to say so.
PLATE_NOT_CHOSEN = (
    "choose the plate on the stage first: the page shows a 96-well placeholder"
)

#: The label of the live button while a live view runs: ``busy_note`` quotes it and the
#: page draws it, from this one constant, so that the note never names a button the
#: page does not have (m5 §11.2).
END_LIVE_LABEL = "End live"

#: Why the step stops at ``scan`` on a page served with ``--dry-run`` (m5 §8.8.1): a dry
#: run walks the fields and takes no frame, so ``cells`` and what follows fail on a
#: missing frame, and the way out of a terminal run is not obvious (a run that is not a
#: dry run refuses what a dry run made).
DRY_RUN_NO_FRAME = (
    "this page is a dry run: scan walks the fields and takes no frame, so cells and "
    "the tools after it have nothing to read. Serve the page without --dry-run on a "
    "stand whose stage is simulated (demo-sample), or run the chain with smc run "
    "from plate-registration, without --dry-run: a run that is not a dry run "
    "refuses what a dry run made"
)

#: Past this many fields the map draws none of the scan plan (m5 §14.1): a first part
#: would show an arbitrary corner of the plan as if it were all of it, and a figure's
#: fields are about 33 bytes each on the websocket at every redraw (measured, §14), so
#: 5000 is about 165 kB. The title and the card say the count instead.
MAP_FIELDS_UP_TO = 5000

#: How many well centres ``plate-walls`` takes: the page's copy of the plugin's bounds
#: (``plate_walls.plugin._MIN_WELLS`` and ``_MAX_WELLS``), pinned by a test. The plugin
#: refuses a count it does not accept before anything moves; the page says so first.
WALLS_MIN_CENTRES = 2
WALLS_MAX_CENTRES = 6

#: Why ``plate-walls`` does not run on a page served with ``--dry-run`` (m5 §12.3). The
#: plugin refuses a dry run before it moves anything, and its refusal is a ``failed``
#: record that would replace a usable plate (m5 §11.4): the controller refuses first.
WALLS_DRY_RUN = (
    "plate-walls images the walls, which a dry run does not do: serve the page "
    "without --dry-run (demo-sample's stage is simulated), or register with "
    "plate-registration"
)

#: Why ``plate-walls`` waits when the run document has no ``camera-calibration`` at all
#: (m5 §12.3). The document's own words say the same for a terminal; the page has the
#: tool one region up.
WALLS_NEED_CALIBRATION = (
    "plate-walls reads the camera's calibration: run camera-calibration first "
    "(Set up), with the objective you register with in place, on a field with "
    "texture (a well's wall will do)"
)

#: Why a wall cannot be captured, nor driven to, without a ``camera-calibration``
#: (m5 §13.1): the operator names it on the image, and only the calibration's
#: rotation and mirror say which stage side that is (FM-189). The same fix as
#: ``WALLS_NEED_CALIBRATION``, a text of its own: here a side cannot be mapped, there
#: a tool cannot run.
NO_CALIBRATION_FOR_SIDES = (
    "walls are named on the image, and no camera calibration says which stage side "
    "that is: run camera-calibration (Set up), then try again"
)

#: Why a drive click goes nowhere on an *unplaced* map (m3 §5.5 D6): no registered plate,
#: and no holder to estimate one from, so the page cannot say where a well is.
GOTO_UNPLACED = (
    "the plate is not registered, and this stand's profile does not say where its "
    "holder puts a plate ([holder] plate_centre_xy_um): register the plate, or "
    "drive with the arrows"
)

#: Why a wall of an *estimated* well is not driven to (m3 §5.5 D6): the holder's estimate
#: can be a few hundred µm off, more than a wall's offset from the centre.
GOTO_WALL_NEEDS_REGISTRATION = (
    "a wall is reached through a registered plate: go to the centre, and register "
    "the plate"
)

#: What the drive's answer says while it goes to an *estimated* well (m3 §5.5 D7), a
#: ``str.format`` template with ``{well}``: the operator then centres it by eye.
GOTO_ESTIMATED = (
    "going to {well}: the holder's estimate, not registered; centre it by eye"
)


# -- round 2: the texts of m5 §17.12.5 (#475), copied from the design, never retyped -----

#: What Live and Snap say of a frame that carries no light (m5 §17.12 D22, #457): the light
#: path is the usual cause (F8, Ti2-E session 3: the light path on the eyepiece).
FRAME_DARK = "The frame is dark: is the light on, and the light path on the camera?"

#: What they say of a frame at the camera's full scale: the remedy is the exposure, not the light.
FRAME_SATURATED = "The frame is saturated: lower the exposure."

#: A capture further than one pitch from where the map puts its well (found on the owner's walk:
#: H1 captured at A12's place, 99 mm off). A note, never a refusal: the registration's own
#: residual bound refuses a bad fit.
CAPTURE_FAR = (
    "{well} was captured {distance_mm:.1f} mm from where the map puts it: is the "
    "stage over {well}? Redo it there."
)

#: Two captures of two wells at one place (the same walk).
CAPTURE_SAME_PLACE = (
    "{well} and {other} were captured at the same place: redo the one that is wrong."
)

#: Why *Register the plate* is refused with the turret turned since the centres were captured
#: (m5 §17.12 D13, #373): the matrix, the centres' scale and the offsets are an objective's.
REGISTER_ACROSS_A_TURN = (
    "The centres were captured with the {captured}; the {in_place} is in place: turn "
    "back to the {captured}, or redo the centres with the {in_place}."
)

#: Why it is refused when the centres were captured with more than one objective.
REGISTER_OBJECTIVES_DIFFER = (
    "The centres were captured with different objectives ({which}): redo them with "
    "one objective."
)

#: What a Plate row says of the *Go to* sent from it (m5 §17.12.3), in place of the tool's
#: whole outcome, which names assumptions that the guided screens have no card for.
GOTO_ROW_GOING = "Going to {well}…"
GOTO_ROW_AT_ESTIMATE = (
    "At {well}, the holder's estimate: centre it by eye with Live and the arrows, then "
    "capture."
)
GOTO_ROW_AT = "At {well}."

#: What *Again* replaces, on one line under a Results row (D20): the short names of what it
#: makes out of date, since the long note was drawn under every row at all times.
AGAIN_NOTE = "Again makes {names} out of date; Run all that are left runs them again."

#: The arrows' note on a stand whose camera is not calibrated: it names the Stand screen's
#: button, not a tab that the guided page no longer has.
CAMERA_NOT_CALIBRATED_NOTE = (
    "Which way the image moves for an arrow is not known: Calibrate the camera, on "
    "the Stand screen, measures it."
)

#: The Stand screen says once what the header said on every screen (D15).
DISCONNECT_RELEASES = "Disconnect releases {stand} for NIS-Elements or ZEN."

#: Above the centre rows: *Register the plate* uses the profile's sample while nothing is
#: captured (D23). The rows say it each in its own words, and the map says the plate is only
#: estimated; this says once what the button will do.
SAMPLE_REGISTERS = (
    "Register the plate uses the profile's sample while no centre is captured; "
    "capture three to measure this plate."
)

#: The Wells screen's plan line while no timing was measured (part B replaces it with a button).
TIME_NOT_KNOWN = "Time: not known (no timing measured yet)"

#: The Wells screen's two fields, in plain words, above each (D20): the fields are the tools'
#: own and keep their names in the expert view.
COVERAGE_WORDS = "How much of each well's width the fields span: 0.4 is 40 %."
AUTOFOCUS_WORDS = (
    "Autofocus: if-engaged uses it when it is on and locked; required refuses to "
    "start unless it is; off never touches it. The scan never switches it on."
)

#: Why everything that needs the stand is refused while the page holds none (m5 §7.2):
#: the one reason a disabled control gives, and the first one. The controller's too.
NOT_CONNECTED = "the stand is not connected"

#: The Scan screen after a scan that did not end ``done`` (m5 §17.12 D17, part B). The two
#: ``UNCOUNTED`` variants are for a page that kept no progress of that scan (it was opened
#: after the stop, or another activity ran since): never ``field None of None``.
SCAN_STOPPED = "Stopped by STOP at field {done} of {total}."
SCAN_STOPPED_UNCOUNTED = "Stopped by STOP."
SCAN_FAILED = "Failed at field {done} of {total}: {text}"
SCAN_FAILED_UNCOUNTED = "Failed: {text}"
SCAN_CONTINUE = "Continue the scan"
SCAN_START_AGAIN = "Start again"
SCAN_START_AGAIN_CONFIRM = (
    "Start again drops the {done} fields already taken: press again to drop them."
)
SCAN_START_AGAIN_CONFIRM_UNCOUNTED = (
    "Start again drops the fields already taken: press again to drop them."
)
SCAN_HALTED = "Press Resume (top right) first: the stand is halted."
SCAN_REPLACES = "Starting the scan again replaces this one ({frames} frames)."
#: Why the scan's own *Run* (the expert view) does not start it afresh over a checkpoint the
#: Scan screen continues: the page's words, not ``close this page first`` (D17).
SCAN_CHECKPOINT_KEPT = (
    "{status} with a checkpoint: Continue the scan, on the Scan screen, takes it up; "
    "Start again there drops it"
)
#: The Wells screen's button while the plan has no time (m5 §17.12 D18).
MEASURE_THE_TIME = "Measure the time"


@dataclass(frozen=True, slots=True)
class NextStep:
    """The tool the run document needs next, with the values to run it and why.

    ``params`` are ``smc run --param`` pairs, the call the button makes. ``why``
    holds one line for the tool and one for each value it carries, each saying
    where the value comes from; the others are the other ways to fill the slot.
    ``plugin`` is ``None`` when there is nothing to run (the workflow is done, or
    no registered plugin fills the slot), and ``blocked`` is why the tool cannot run
    now: the step is still drawn, so the operator reads what to do. ``command`` is
    ``""`` without a tool.
    """

    plugin: str | None
    params: tuple[str, ...]
    why: tuple[str, ...]
    command: str
    blocked: str | None


def next_slot(checklist: Sequence[CheckItem]) -> CheckItem | None:
    """The first slot the workflow needs and does not have in a usable state, or ``None``.

    The optional slots (``camera-calibration``, ``timings``, a third party's) are
    never a next step: the workflow runs without them.
    """
    for item in checklist:
        if not item.optional and item.state != "ok":
            return item
    return None


# -- the steps of the guided page (m5 §17.5, #443) -------------------------------------

#: The five screens of the guided page, in the order an operator walks them.
StepName = Literal["stand", "plate", "wells", "scan", "results"]

STEPS: tuple[StepName, ...] = ("stand", "plate", "wells", "scan", "results")

STEP_TITLES: Mapping[StepName, str] = {
    "stand": "Stand",
    "plate": "Plate",
    "wells": "Wells",
    "scan": "Scan",
    "results": "Results",
}

#: Which screen holds the tools of a checklist slot (m5 §17.5). A slot not listed (a third
#: party's) belongs to no screen: it is in the expert view only, and the step bar never
#: says it is waiting for it.
STEP_OF_SLOT: Mapping[str, StepName] = {
    "camera-calibration": "stand",
    "timings": "stand",
    "plate": "plate",
    "scan-plan": "wells",
    "scan": "scan",
    "cells": "results",
    "density": "results",
    "route": "results",
    "report": "results",
    "nis-export": "results",
}

#: The question each screen answers (m5 §17.4, D8). The scan's title is the fallback of a
#: page with neither a plan nor a selection: ``scan_title`` names the wells otherwise.
SCREEN_TITLES: Mapping[StepName, str] = {
    "stand": "Which stand, and which preset?",
    "plate": "Where is the plate?",
    "wells": "Which wells?",
    "scan": "Scan the selected wells",
    "results": "Results",
}

#: The one sentence under a screen's title (m5 §17.4).
SCREEN_LEADS: Mapping[StepName, str] = {
    "stand": (
        "The preset fills in the plate, the scan's coverage and the cell detector. "
        "You can change any of them later."
    ),
    "plate": (
        "Centre three wells far apart by eye, with Live and the arrows, and capture "
        "each centre."
    ),
    "wells": (
        "Click wells on the map, or type them. The map shows the fields and the time "
        "the scan will take."
    ),
    "scan": "STOP, at the top right, stops the scan.",
    "results": "Each step reads the one before. Run the next one, or all that are left.",
}

#: A step's summary is one line under its button and never wider than the button's column:
#: 22 characters, so that the step bar is one line tall (m5 §17.12 D15).
_SUMMARY_CHARS = 22


@dataclass(frozen=True, slots=True)
class StepState:
    """What the step bar draws for one step: a tick, the suggestion, one line (m5 §17.5).

    The state is the run document's and the connection's, never a view's: the step
    *shown* belongs to the browser tab (D6), whereas *done* and *suggested* are the same
    in every tab (FM-151).
    """

    step: StepName
    done: bool
    suggested: bool
    summary: str


def _cut(text: str) -> str:
    """``text`` as a summary line: cut with an ellipsis rather than overflow the button."""
    if len(text) <= _SUMMARY_CHARS:
        return text
    return text[: _SUMMARY_CHARS - 1] + "…"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def step_states(
    checklist: Sequence[CheckItem],
    *,
    connected: bool,
    stand: str,
    preset: str | None,
    plate: str,
    plan: PlanLayer | None,
) -> tuple[StepState, ...]:
    """The five steps' states, in ``STEPS`` order, from the checklist and the connection.

    The *stand* step is done by the connection alone: the optional slots (the camera's
    calibration, the timings) never decide a step, since the workflow runs without them.
    Another step is done when every required slot it holds is ``ok``, and a step the
    checklist holds no required slot of is not done (unknown beats guessed: an empty
    checklist says nothing about the plate). At most one step is suggested, the one that
    holds ``next_slot``, or the stand while the page is not connected: nothing else can
    run before it. A step that is neither done nor suggested has an empty summary, and
    the suggested one says ``next``; *results* says how many of its tools are done as
    soon as one is, since five tools are not one decision.

    ``stand`` is the profile's name (``PlatePage.stand_name``, read without the stand),
    ``preset`` the loaded preset's name, ``plate`` the plate the page is about and
    ``plan`` the usable scan plan (``None``: none, or not readable).
    """
    held: dict[StepName, list[CheckItem]] = {step: [] for step in STEPS}
    for item in checklist:
        step = STEP_OF_SLOT.get(item.slot)
        if step is not None:
            held[step].append(item)

    following = next_slot(checklist)
    suggested: StepName | None
    if not connected:
        suggested = "stand"
    elif following is None:
        suggested = None
    else:
        suggested = STEP_OF_SLOT.get(following.slot)

    states: list[StepState] = []
    for step in STEPS:
        required = [item for item in held[step] if not item.optional]
        done_slots = sum(1 for item in held[step] if item.state == "ok")
        done = (
            connected
            if step == "stand"
            else bool(required) and all(item.state == "ok" for item in required)
        )
        summary = ""
        if step == "stand":
            if not connected:
                summary = "not connected"
            elif preset:
                summary = f"{stand}, preset {preset}"
            else:
                summary = stand
        elif step == "results" and done_slots:
            summary = f"{done_slots} of {len(held[step])} done"
        elif done and step == "plate":
            summary = f"{plate}, registered"
        elif done and step == "wells":
            summary = (
                f"{_count(plan.wells, 'well')}, {_count(plan.count, 'field')}"
                if plan is not None
                else "laid out"
            )
        elif done and step == "scan":
            summary = "scanned"
        elif step == suggested:
            summary = "next"
        states.append(
            StepState(
                step=step,
                done=done,
                suggested=step == suggested,
                summary=_cut(summary),
            )
        )
    return tuple(states)


def opening_step(states: Sequence[StepState]) -> StepName:
    """The step the page opens on: the suggested one, else *results* (m5 §17.5, D2).

    Nothing left to do means the operator comes to read what was made. Only the
    opening is the page's choice: afterwards the screen changes by the operator's
    click alone (D2).
    """
    for state in states:
        if state.suggested:
            return state.step
    return "results"


def step_button_label(index: int, state: StepState) -> str:
    """A step button's label: ``2. Plate ✓`` when done, ``2. Plate`` otherwise.

    The number makes each button's accessible name unique: *Plate* is also a select's
    label and the map's title, which ``/ui-check`` locates by text (m5 §17.5).
    """
    tick = " ✓" if state.done else ""
    return f"{index}. {STEP_TITLES[state.step]}{tick}"


def continue_label(step: StepName) -> str | None:
    """The *Continue* button's label on ``step``: the next step's title, ``None`` after the last."""
    position = STEPS.index(step)
    if position + 1 >= len(STEPS):
        return None
    return f"Continue to {STEP_TITLES[STEPS[position + 1]]}"


# -- the Stand and Plate screens (m5 §17.4.1, §17.4.2, #444) ---------------------------

#: Why *Go to* is disabled while Live runs, and what is still possible (m5 §17.2 D11): a
#: ``goto-well`` run is refused during Live (§5.2), whereas a capture reads the stage and
#: is accepted (measured on the owner's walk, #431). Per well the path is Go to, Live,
#: the arrows, Capture, End live, and nothing ends Live on its own.
GOTO_NOT_DURING_LIVE = "End live to go to another well; Capture works during Live"

#: What the Plate screen says of the two other ways to register (m5 §17.2 D3, §17.4.2): the
#: guided path offers three centres by eye only, and these two are run from the expert view.
OTHER_WAYS_TO_REGISTER = (
    "Other ways to register: from the walls of 2 or 3 wells (slower; measures the "
    "wells' size too) or automatically (surveys the plate, about 7 minutes): in the "
    "expert view."
)


@dataclass(frozen=True, slots=True)
class CalibrationNote:
    """What the Stand screen says of the camera's calibration: the objective it was made at, and when.

    ``objective`` is the turret label the latest usable ``camera-calibration`` recorded,
    ``None`` when it recorded none (*unknown*, not "the one in place"). ``when_utc`` is
    the record's time in UTC as a result card prints it (``YYYY-MM-DD HH:MM:SS``, or its
    unknown-time placeholder): the page says the zone beside it. ``when_local`` is the
    same time in the PC's local zone to the minute (``local_time``), which the guided
    screens say; ``""`` for a note built without it, and then ``when_utc`` is said
    (m5 §17.12 D20).
    """

    objective: str | None
    when_utc: str
    when_local: str = ""


def needs_calibration(
    objective: str | None, calibration: CalibrationNote | None
) -> bool:
    """Whether the Stand screen offers *Calibrate the camera* (m5 §17.4.1).

    When there is no calibration at all, or when both objectives are known and differ:
    rotation and mirror belong to the optical path, but the matrix's scale is an
    objective's, and the arrows' note must not say which way the image moves from the
    matrix of another one. An objective that is not known, on either side, is no
    difference the page can see, so it offers nothing it cannot justify.
    """
    if calibration is None:
        return True
    return (
        objective is not None
        and calibration.objective is not None
        and objective != calibration.objective
    )


def objective_line(objective: str | None) -> str:
    """The "known" blocks' first line: the objective in place as the strip reads it, or that it is not known (m5 §17.2 D7).

    One line for the Stand screen's block and the Wells screen's, so that the two never
    say the objective in two ways. The label is the stand's own (FM-139: plain text).
    """
    return (
        "Objective: not known"
        if objective is None
        else f"Objective: {objective}, in place"
    )


def known_text(
    *,
    objective: str | None,
    calibration: CalibrationNote | None,
    light_on: bool | None,
    stand: str | None = None,
) -> str:
    """The "known" block of the Stand screen: three lines, each saying what it rests on (m5 §17.2 D7, §17.4.1).

    Plain text, never markup (FM-139): the turret's label is the stand's. A value the
    page has not read says ``not known`` and a camera with no calibration says what the
    operator loses (the arrows cannot say which way the image moves): an unknown is
    never shown as a value (CLAUDE.md). ``objective`` is the one in place as the strip
    reads it, ``calibration`` the latest usable one, ``light_on`` the light as read. The
    calibration's time is the PC's local one (``CalibrationNote.when_local``), the UTC
    one for a note that has none. ``stand`` is the stand's name **while it is
    connected**, ``None`` otherwise: a fourth line says that Disconnect releases it for
    the vendor's software, which is a fact of the Stand screen and no longer of the
    header (m5 §17.12 D15).
    """
    first = objective_line(objective)
    if calibration is None:
        camera = (
            "Camera: not calibrated (the arrows cannot say which way the image moves)"
        )
    else:
        when = (
            calibration.when_local
            if calibration.when_local
            else f"{calibration.when_utc} UTC"
        )
        if calibration.objective is None:
            camera = f"Camera: calibrated on {when}, objective not recorded"
        else:
            camera = f"Camera: calibrated at {calibration.objective} on {when}"
            if objective is not None and objective != calibration.objective:
                camera += f", not at {objective}"
    light = (
        "Light: not known"
        if light_on is None
        else f"Light: {'on' if light_on else 'off'}"
    )
    lines = [first, camera, light]
    if stand is not None:
        lines.append(DISCONNECT_RELEASES.format(stand=stand))
    return "\n".join(lines)


def suggested_centres(plate: str) -> tuple[str, ...]:
    """The wells the Plate screen suggests centring: A1, the last column of row A and the last row of column 1 (m5 §17.4.2).

    The names are the ones the plate prints, whatever way it lies in its holder (a turn
    renames the operator's captures, §16 D6, never this list). They are far apart, so
    they pin the rotation down: adjacent wells barely constrain it. It is the layer's
    own list for the same purpose (``plate-registration``'s ``suggested_refs``, which
    the sample's references use), so that the screen and the tool never disagree about
    which three wells make a good calibration; a one-well plate (a coverslip) has one.

    Raises:
        ValueError: ``plate`` is not a plate of ``useq``'s registry.
    """
    return suggested_refs(plate)


@dataclass(frozen=True, slots=True)
class CentreRow:
    """One row of the Plate screen's centres: the well, the sentence about it, and whether it is captured.

    ``captured`` picks the row's second button: *Capture the centre of <well>*, or *Redo*.
    """

    well: str
    text: str
    captured: bool


def _um_one(value: float) -> str:
    """``value`` to a tenth of a micrometre; ``-0.0`` is printed ``0.0``, as the strip's ``_um`` does."""
    text = f"{value:.1f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def centre_rows(
    plate: str,
    refs: Sequence[WellRef],
    objectives: Mapping[str, str | None],
    *,
    pose: PoseKind,
    from_sample: bool,
) -> tuple[CentreRow, ...]:
    """The Plate screen's centres: the suggested wells in their order, then the other captures in capture order (m5 §17.4.2).

    The suggestion is not a rule: a well clicked on the map and captured is listed
    too, after the three. A captured row says where and with which objective, so that a
    turn of the turret between two captures is visible (#373 is not decided here); an
    objective the page did not read, or a capture restored from a run document, says
    ``the objective not known`` and never the one in place. A well not captured says
    what to do, in the words the map's pose allows (D10): on an *estimated* map the
    stage goes to the holder's estimate and the operator centres it by eye, on an
    *unplaced* one nothing can go there (``goto`` refuses) and the arrows do. With
    ``from_sample`` (a profile ``[sample]`` of this plate and nothing captured) the
    registration runs from the sample's own centres: *known, not measured* (m5 §8.2).
    """
    captured = {ref.well: ref for ref in refs}
    suggested = suggested_centres(plate)
    wells = [*suggested, *(ref.well for ref in refs if ref.well not in suggested)]
    rows: list[CentreRow] = []
    for well in wells:
        ref = captured.get(well)
        if ref is not None:
            label = objectives.get(well)
            used = "the objective not known" if label is None else label
            text = (
                f"{well}: captured at ({_um_one(ref.x_um)}, {_um_one(ref.y_um)}) "
                f"µm with {used}"
            )
        elif from_sample:
            text = f"{well}: known from the profile's sample, not measured"
        elif pose == "estimated":
            text = f"{well}: go there (estimated position), centre it by eye, capture"
        elif pose == "unplaced":
            text = f"{well}: jog there with the arrows, centre it, capture"
        else:
            text = f"{well}: go there, centre it, capture"
        rows.append(CentreRow(well=well, text=text, captured=ref is not None))
    return tuple(rows)


def registered_line(result: PlateResult) -> str:
    """The Plate screen's result line: the residual and the turn of the plate just registered (m5 §17.4.2).

    A seed or a lattice fit measures no residual, and says so: an unknown is not a
    zero (CLAUDE.md). The rotation is the plate's own, to a hundredth of a degree. A
    measured residual says the bound the tool refuses beyond (``GOOD_RESIDUAL_UM``), so
    that 62.3 µm reads as good or as bad without a manual (m5 §17.12 D20).
    """
    residual = (
        "residual not measured"
        if result.residual_um is None
        else f"residual {result.residual_um:.1f} µm of the {GOOD_RESIDUAL_UM:g} µm allowed"
    )
    return f"Registered: {residual}, turned {result.rotation_deg:.2f}\N{DEGREE SIGN}"


def register_replace_note(result: PlateResult) -> str:
    """What *Register the plate* replaces when a plate is already registered, said before the click (m5 §11.4, FM-172).

    The latest attempt decides the ``plate`` slot even when it fails, so a failed
    registration leaves no plate where a usable one stood. The page does not refuse: the
    captures stay, so the way back is to *Redo* the centre that was wrong and register
    again. The tool's own note (``replace_note``) names the expert view's *Run* and
    *Forget*, which this screen does not have: this one names its own buttons, and the
    plate it speaks of is the result line's, a few lines above. Plain text (FM-139).
    """
    wells = "1 well" if result.n_refs == 1 else f"{result.n_refs} wells"
    return (
        f"Registering again replaces this plate ({result.method}, {wells}). A failed "
        "run leaves no plate; the centres captured stay, so Redo the wrong one and "
        "register again to get a plate back."
    )


def plate_from_line(
    *,
    plate: str,
    chosen: bool,
    loaded: LoadedPreset | None,
    sample_plate: str | None,
) -> str:
    """Where the Plate screen's plate comes from, on one line under the menu (m5 §17.2 D7).

    The first that holds: nothing named it (the page's placeholder, which no tool runs
    on, §11.3); the loaded preset's plate; the profile's ``[sample]``'s; else the
    operator's own choice on this page.
    """
    if not chosen:
        return "the page's placeholder: choose the plate on the stage"
    if (
        loaded is not None
        and loaded.file.preset is not None
        and loaded.file.preset.plate == plate
    ):
        return f"from the preset {loaded.file.name}"
    if sample_plate == plate:
        return "from the profile's sample"
    return "chosen on this page"


def capture_centre_label(well: str | None) -> str:
    """The *Capture centre* button's label: the well it records, as the page drew it (m5 §11.1).

    The operator reads the well on the button before pressing it: a capture
    recorded under a well the operator did not mean is a wall at the wrong place,
    and nothing else on the page showed which well that was.
    """
    return "Capture centre" if well is None else f"Capture centre of {well}"


def capture_wall_label(well: str | None, side: str) -> str:
    """The *Capture wall* button's label: the well and the *Wall* select's value as it stands.

    The select's values are sides on the image (m5 §13.1), and the label says so
    (``at image left``): a button that said only ``left`` could be read as the
    stage's.
    """
    return "Capture wall" if well is None else f"Capture {well} wall at image {side}"


def image_sides_note(matrix: Matrix2 | None) -> str:
    """What each side of the image is on the stage, under the capture buttons (m5 §13.2).

    The operator names a wall as the image shows it and the layer takes stage
    directions: the note says the mapping the page will use, so that a wall named
    on the image is not a surprise in the capture list. A matrix that ``frames``
    refuses gives ``frames``' own text (a camera too far from a quarter turn names
    no side), and no matrix says what to run. Plain text (FM-139).
    """
    if matrix is None:
        return (
            "Walls are named on the image. No camera calibration says which stage "
            "side each one is: run camera-calibration (Set up) to capture a wall or "
            "drive to one."
        )
    try:
        left, right, top, bottom = (
            stage_side_of_image_side(side, matrix)
            for side in ("left", "right", "top", "bottom")
        )
    except ValueError as exc:
        return str(exc)
    return (
        f"On the image: left is the stage's {left}, right {right}, top {top}, "
        f"bottom {bottom} (from the camera calibration)."
    )


def drive_image_note(matrix: Matrix2 | None) -> str:
    """What the Drive pad's arrows do to the image, under the pad (m5 §13.2).

    The arrows stay in the stage's axes (the owner's decision of 2026-11-03: it is
    the stage that moves, so the image is normally inverted, and microscopy's
    standards are kept); the note says what each does to the image on this stand,
    from the camera calibration, and *not known* when there is none. ``↑`` is a
    +Y step and ``→`` a +X step. Plain text (FM-139).
    """
    if matrix is None:
        return CAMERA_NOT_CALIBRATED_NOTE
    try:
        up = image_direction_of_stage_step("+Y", matrix)
        right = image_direction_of_stage_step("+X", matrix)
    except ValueError as exc:
        return str(exc)
    return f"On the image, ↑ moves it {up}, → moves it {right}; ↓ and ← the opposite."


def forget_label(well: str | None) -> str:
    """The *Forget* button's label: the well whose captures it drops."""
    return "Forget" if well is None else f"Forget {well}"


def busy_note(running: str, *, live: bool) -> str:
    """Why a tool's *Run* is disabled while an activity goes, beside the button (m5 §11.2).

    A tool does not run during another activity (one at a time, m5 §2.1) and the
    strip that says what runs is three regions away: the reason is said where the
    operator looks. Live is told apart because the way out is one button, whose
    label the note quotes; any other activity ends by itself or by STOP.
    """
    if live:
        return f"Live is running: press {END_LIVE_LABEL}, then run the tool."
    return f"{running} is running: wait for it to end, or press STOP."


def walls_centres_problem(captured: int) -> str | None:
    """Why ``plate-walls`` cannot run with ``captured`` centres, or ``None`` (m5 §12.3).

    The count is the page's copy of the plugin's (``WALLS_MIN_CENTRES`` to
    ``WALLS_MAX_CENTRES``). Fewer than two says what to do next, in the operator's
    words (Live, the Drive pad, the map); more than six says to forget some. The
    texts say ``2 or 3`` as the design does: two wells measure, a third checks the
    fit (m3 §17.4), and a fourth to sixth are allowed, not advised.
    """
    if captured == 0:
        return (
            "plate-walls needs the centres of 2 or 3 wells: centre a well under the "
            "objective (Live and the Drive pad), click it on the map and press "
            "Capture centre; then a second well, far from the first"
        )
    if captured < WALLS_MIN_CENTRES:
        return (
            f"{captured} centre captured, plate-walls needs 2 or 3 wells: centre a "
            "second well, far from the first, click it on the map and press Capture "
            "centre"
        )
    if captured > WALLS_MAX_CENTRES:
        return (
            f"plate-walls takes at most {WALLS_MAX_CENTRES} wells and {captured} "
            "centres are captured: Forget some"
        )
    return None


def walls_square_wells(plate: str) -> str:
    """Why ``plate-walls`` does not run on ``plate``, whose wells are square (m5 §12.3, #375).

    The plugin's own refusal, word for word: it fits circles and refuses a 384- or
    1536-well plate before it moves, and its refusal is a ``failed`` record that
    replaces a usable plate (m5 §11.4). The page says it beside the button and the
    controller refuses first, so the operator reads one sentence whoever refuses.
    A test pins the words to the plugin's.
    """
    return (
        f"plate-walls fits round wells, and the {plate} plate's are square: "
        "register it with plate-registration"
    )


def centres_text(refs: Sequence[WellRef]) -> str:
    """The centres captured on the page, one line each in capture order (m5 §12.5).

    Plain text, never markup (FM-139): a well's name comes from the page, but the
    positions are what the operator checks against the stage, so they are shown,
    to a tenth of a micrometre.
    """
    if not refs:
        return "no centre captured yet"
    lines = [f"centres captured ({len(refs)}):"]
    lines += [f"  {ref.well} ({ref.x_um:.1f}, {ref.y_um:.1f}) µm" for ref in refs]
    return "\n".join(lines)


def wall_touches_note(touches: int) -> str:
    """What becomes of the wall captures under ``plate-walls``, said once (m5 §12.5).

    They are ``plate-registration``'s; ``plate-walls`` images the walls itself and
    ignores them. The operator who touched a wall would wonder; ``""`` for none.
    """
    if touches <= 0:
        return ""
    these = "1 wall capture is" if touches == 1 else f"{touches} wall captures are"
    return (
        f"{these} for plate-registration; plate-walls images the walls itself and "
        "ignores them"
    )


def walls_card_lines(result: PlateResult) -> tuple[str, ...]:
    """The card of a plate registered from images: each well measured, then the plate (m5 §12.6).

    One line per well of ``walls.wells``, in the order given: the fitted centre,
    the diameter and the frames its four sides took; then one for the plate, with
    the residual and the rotation. The generic card shows a result's first keys and
    cuts the walls, the part an operator reads. ``residual unknown`` when nothing
    measured it: an unknown is not a zero.
    """
    lines: list[str] = []
    if result.walls is not None:
        for fit in result.walls.wells:
            frames = sum(side.frames for side in fit.walls)
            x_um, y_um = fit.center_xy_um
            lines.append(
                f"{fit.well}: centre ({x_um:.1f}, {y_um:.1f}) µm, "
                f"{fit.diameter_um:.0f} µm across, {frames} frames"
            )
    residual = (
        "residual unknown"
        if result.residual_um is None
        else f"residual {result.residual_um:.1f} µm"
    )
    lines.append(
        f"plate {result.plate}: {residual}, rotation {result.rotation_deg:.2f}°"
    )
    return tuple(lines)


#: A1's corner in the stage's terms for each orientation (m3 §24 D1): ``useq``'s layout
#: puts A1 at -X +Y, and a plate turned 180° has it at +X -Y.
_A1_STAGE_CORNER: Mapping[str, tuple[str, str]] = {
    "not-turned": ("-X", "+Y"),
    "turned-180": ("+X", "-Y"),
}

#: Where A1 is on the map, which is drawn X right and Y up whatever the stand's sides.
_A1_MAP_CORNER: Mapping[str, str] = {
    "not-turned": "top left",
    "turned-180": "bottom right",
}

#: What the note under *Plate lies* says while the profile names no side of the stand
#: (m5 §16 D3): the choices are then in the stage's terms, and this says why.
ORIENTATION_IN_STAGE_TERMS = (
    "this stand does not say which side of it is the back, so the corners are named "
    "in stage terms: its profile's [holder] back and right would name them"
)


#: What the strip adds when a turn renamed the plate under a selection (m5 §16 D6): a
#: selection is names, the printed wells, which now lie elsewhere on the stage.
SELECTION_KEPT_ITS_NAMES = (
    "the selection kept its names; with the plate turned they lie elsewhere"
)


def _opposite(direction: str) -> str:
    """``-X`` for ``+X``: the other end of one stage axis."""
    return ("-" if direction[0] == "+" else "+") + direction[1:]


def side_words(back: str | None, right: str | None) -> dict[str, str]:
    """The words for each stage direction, from the profile's ``[holder] back`` and ``right`` (m5 §16 D3).

    ``{}`` unless both are set: one side alone would name a corner by half. The
    profile refuses one alone or two on one axis; this only reads what it accepted.
    """
    if back is None or right is None:
        return {}
    return {
        back: "back of the stand",
        _opposite(back): "front of the stand",
        right: "operator's right",
        _opposite(right): "operator's left",
    }


def a1_corner(
    orientation: PlateOrientation, back: str | None, right: str | None
) -> str:
    """The corner A1 is at, in the operator's words when the stand says its sides (m5 §16 D3).

    ``the back left``, back-or-front first, from the stage corner of the orientation
    and the profile's directions; each word is found by the direction, never by its
    axis, so a stand whose back is +X says it right (FM-73). Without the directions,
    ``stage -X +Y``.
    """
    x, y = _A1_STAGE_CORNER[orientation]
    if back is None or right is None:
        return f"stage {x} {y}"
    # The corner's direction on the back's axis, and the one on the right's: the
    # profile keeps them on different axes, so each is found exactly once.
    depth = x if x[1] == back[1] else y
    width = x if x[1] == right[1] else y
    front_or_back = "back" if depth == back else "front"
    left_or_right = "right" if width == right else "left"
    return f"the {front_or_back} {left_or_right}"


def orientation_label(
    orientation: PlateOrientation, back: str | None, right: str | None
) -> str:
    """The *Plate lies* choice's label: ``A1 at the back left``, ``Turned 180°: A1 at the front right`` (m5 §16 D3).

    In stage terms when the stand does not say its sides, both labelled then, since
    "A1 at stage -X +Y" alone does not say it is the plate as ``useq`` lays it.
    """
    corner = f"A1 at {a1_corner(orientation, back, right)}"
    if orientation == "turned-180":
        return f"Turned 180\N{DEGREE SIGN}: {corner}"
    if back is None or right is None:
        return f"Not turned: {corner}"
    return corner


def map_title(plate: str, pose: MapPose) -> str:
    """What the map's title says about how its plate was placed, before the scan plan's suffix (m5 §15 D4, §16 D8).

    A well name read off the map is trusted, so the title says on its face which of
    *registered* (measured), *seed* (one well given, the rotation taken from the
    declared orientation), *estimated* (from the stand's holder) or *unplaced* (the
    stand says nothing) the map is, and for the last two what the operator would need
    to act on it: where the estimate puts the plate, which orientation was declared and
    which corner of the map A1 is at, or that the cross then shows the stage and no
    well. A registered plate and a seed name A1's corner (``pose.a1_corner``), the check
    the operator makes against the plate. Plain text, never markup (FM-139): the plate
    name is quoted as it is. The holder's centre is shown to whole µm, which is finer
    than it is known.
    """
    corner = f", A1 at {pose.a1_corner}" if pose.a1_corner else ""
    if pose.kind == "registered":
        return f"{plate}: the registered plate{corner}"
    if pose.kind == "seed":
        well = f" ({pose.seed_well})" if pose.seed_well else ""
        return (
            f"{plate}: placed from one well{well}, not registered: "
            f"the rotation is taken from the declared orientation{corner}, "
            "not measured"
        )
    if pose.kind == "estimated":
        # `round` is an int: no "-0" for a point a fraction of a micron below zero. A
        # pose with no centre (the controller never builds one) claims no position.
        centre = pose.holder_centre_um
        where = (
            ""
            if centre is None
            else (
                f"centred at stage ({round(centre[0])}, {round(centre[1])}) "
                "\N{MICRO SIGN}m, "
            )
        )
        # "declared": the estimate measured no orientation, the operator said it (FM-217).
        turned = (
            "turned 180\N{DEGREE SIGN}"
            if pose.orientation == "turned-180"
            else "not turned"
        )
        return (
            f"{plate}: estimated from the stand's holder, not registered: "
            f"{where}declared {turned}, A1 at the "
            f"{_A1_MAP_CORNER[pose.orientation]} of this map"
        )
    return (
        f"{plate}: not registered, and this stand does not say where its plate lies: "
        "A1 is drawn at stage (0, 0), so the cross shows the stage and not a well"
    )


def stage_label(
    wells: Sequence[WellView], pose: MapPose, xy_um: tuple[float, float] | None
) -> str:
    """The text by the cross: the well the stage is over, and on what that name rests (m5 §15 D5).

    ``""`` when there is no position (a closed stand), when the map is *unplaced* (its
    A1 is the stage's origin, so a name would be a coincidence of the drawing), or when
    the point is on no well. Otherwise ``over B3``, then ``(estimated)`` or
    ``(rotation assumed)`` for the poses that did not measure the plate: a name read off
    an estimate is an assumption, and a plate turned 180° swaps every name (m5 §15 D6).

    Each ``WellView`` is turned into a ``WellGeometry``, the one place that knows a well's
    shape (a disc, or a rectangle turned with the plate), rather than copying the test:
    0.19 ms over the 1536 wells of the largest plate. The first well in the view's order
    that contains the point wins; wells do not overlap.
    """
    if xy_um is None or pose.kind == "unplaced":
        return ""
    x_um, y_um = xy_um
    for well in wells:
        geometry = WellGeometry(
            well=well.well,
            center_um=well.center_um,
            size_um=well.size_um,
            circular=well.circular,
            rotation_deg=well.rotation_deg,
        )
        if geometry.contains(x_um, y_um):
            note = {"estimated": " (estimated)", "seed": " (rotation assumed)"}
            return f"over {well.well}{note.get(pose.kind, '')}"
    return ""


def plan_title(plan: PlanLayer | None, *, drawn_up_to: int = MAP_FIELDS_UP_TO) -> str:
    """The suffix the map's title gets for the scan plan it draws, ``""`` for none (m5 §14.3).

    Past ``drawn_up_to`` fields the layer holds none (a first part of the plan would
    pass for all of it), so the title is where the operator reads that the plan exists
    and why it is not drawn. The bound is a keyword so that the boundary is tested
    without a plan of five thousand fields.
    """
    if plan is None:
        return ""
    fields = "1 field" if plan.count == 1 else f"{plan.count} fields"
    wells = "1 well" if plan.wells == 1 else f"{plan.wells} wells"
    text = f"; scan plan: {fields} in {wells}"
    if plan.count > drawn_up_to:
        text += f", too many to draw (over {drawn_up_to})"
    return text


def duration_text(seconds: float) -> str:
    """``seconds`` as the page says a duration: to the second, then minutes, then hours (m5 §14.3).

    Rounded to the nearest second (Python's ``round``): ``42.16`` is ``42 s``. Under
    a minute ``<s> s``, under an hour ``<m> min <ss> s``, from an hour ``<h> h <mm>
    min`` with the seconds dropped, because an estimate of hours is not read to the
    second. The rounding comes first, so ``59.6`` is ``1 min 00 s``, never ``60 s``.
    """
    total = round(seconds)
    if total < 60:
        return f"{total} s"
    if total < 3600:
        minutes, rest = divmod(total, 60)
        return f"{minutes} min {rest:02d} s"
    hours, rest = divmod(total, 3600)
    return f"{hours} h {rest // 60:02d} min"


#: How many wells the scan plan's card names before it counts the others.
_CARD_WELLS = 6


def scan_plan_card_lines(result: ScanPlanResult) -> tuple[str, ...]:
    """The card of a scan plan: the wells, the fields, the field and the estimate (m5 §14.3).

    Built from the result because the generic card shows its first eight keys and
    cut ``estimate_s`` away, the figure the operator wants before pressing *scan*.
    The wells are named in the plan's order (the visiting order, not the
    alphabetical one). The estimate says what ``estimate_s()`` sums, the moves, a
    refocus when the timings know its cost and the snaps, and no more: it leaves out a
    channel change, the cells, the report and the export. An unknown estimate says what
    makes it unknown and what to run, and is never a zero (CLAUDE.md: unknown beats
    guessed).
    """
    names = list(result.wells[:_CARD_WELLS])
    if len(result.wells) > _CARD_WELLS:
        names.append(f"… {len(result.wells) - _CARD_WELLS} more")
    wells = f"wells: {len(result.wells)}"
    if names:
        wells += f" ({', '.join(names)})"
    grid = f"a grid of {result.rows} × {result.columns} in each well"  # noqa: RUF001
    planned = result.rows * result.columns * len(result.wells)
    if len(result.fields) == planned:
        fields = f"fields: {len(result.fields)}, {grid}"
    else:
        fields = (
            f"fields: {len(result.fields)} of {planned}, {grid}, "
            "the fields that miss the well left out"
        )
    width_um, height_um = result.fov_um
    at = "" if result.objective is None else f" at {result.objective}"
    field = f"field: {width_um:g} × {height_um:g} µm{at}, overlap {result.overlap:g}"  # noqa: RUF001
    if result.estimate_s is None:
        estimate = (
            "estimate: unknown, no usable timing: run timing, then scan-plan again"
        )
    else:
        estimate = (
            f"estimate: about {duration_text(result.estimate_s)} "
            "(the moves, refocus and snaps of the timings)"
        )
    return (wells, fields, field, estimate)


# -- the Wells and Scan screens (m5 §17.4.3, §17.4.4, §17.10, #445) ---------------------

#: What *Lay out the fields* waits for before a plate is registered (m5 §17.10.3): the
#: scan plan reads the plate's place, and the stand's holder estimate is not a registration.
REGISTER_FIRST = "register the plate first: step 2, Plate"

#: What *Start the scan* waits for while there is no usable plan (m5 §17.10.3): a stale,
#: foreign or missing plan is not what the scan would do.
LAY_OUT_FIRST = "lay out the fields first: step 3, Wells"

#: One well name as a plate prints it: the row's letters, then the column's number. Bounded:
#: a plate has rows of one or two letters and columns of one or two digits, and a name taken
#: from a record as it comes (FM-125) may hold five thousand digits, which ``int()`` refuses
#: (Python's limit is 4300): the page's texts are drawn before its strip, so one such name
#: would stop the strip, the problems line and the buttons at every tick.
_WELL_NAME = re.compile(r"([A-Za-z]{1,6})(\d{1,6})")


def _plate_order(name: str) -> tuple[int, int, str, int, str]:
    """A well's sort key in plate order: ``B2`` before ``B11``, ``Z1`` before ``AA1`` (m5 §17.10.3).

    A name that is not a letter group and a number sorts last, by its text: the page names
    what a plan holds and does not refuse an odd name.
    """
    found = _WELL_NAME.fullmatch(name)
    if found is None:
        return (1, 0, "", 0, name)
    row, column = found.groups()
    return (0, len(row), row.upper(), int(column), name)


def wells_text(wells: Sequence[str]) -> str:
    """Wells as a title says them: one, two, three by name in plate order, then a count (m5 §17.10.3).

    ``B2``, ``B2 and B11``, ``A1, A2 and A3``, and from four wells ``12 wells``. The order
    is the plate's, never the order they come in (the demo scans B11 first): the operator
    reads which wells, not the route. A repeated name counts once.
    """
    ordered = sorted(set(wells), key=_plate_order)
    if not ordered:
        return "no wells"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) <= 3:
        return f"{', '.join(ordered[:-1])} and {ordered[-1]}"
    return f"{len(ordered)} wells"


def scan_title(plan_wells: Sequence[str], selection: Sequence[str]) -> str:
    """The Scan screen's title, from the **plan's** wells (m5 §17.4.4, §17.10.3).

    The plan is what ``scan`` does (§14.1), so it names the wells; the selection stands in
    while there is no plan, and the static title while there is neither.
    """
    if plan_wells:
        return f"Scan {wells_text(plan_wells)}"
    if selection:
        return f"Scan {wells_text(selection)}"
    return SCREEN_TITLES["scan"]


def scan_lead(plan: PlanLayer | None, *, stop_refused: bool) -> str:
    """The Scan screen's lead: the plan's count and objective, where STOP is, and what it does not stop (m5 §17.10.3).

    On a stand whose profile says a role refuses its stop (``stop_refused_by``) a move
    already sent runs to its target, and the lead says so (FM-117): STOP is offered, never
    promised more than it does. Without a plan the static lead stands, with the same addition.
    """
    stop = SCREEN_LEADS["scan"]
    if stop_refused:
        stop += " A move already sent finishes first on this stand."
    if plan is None:
        return stop
    fields = _count(plan.count, "field")
    at = "" if plan.objective is None else f" at {plan.objective}"
    return f"{fields}{at}. {stop}"


def wells_known_lines(
    *,
    objective: str | None,
    plan: PlanLayer | None,
    out_of_date: str | None,
    assumed: Sequence[str],
) -> tuple[str, ...]:
    """The Wells screen's "known" block under *Lay out the fields*: objective, fields, time, assumptions (m5 §17.4.3, §17.10.3).

    ``plan`` is the usable scan plan, ``None`` when there is none: then the line says the
    plan is out of date and why (``out_of_date``: the checklist's reason for the slot) or
    that none is laid out yet. A plan made with no timing says ``Time: not known`` with
    what to run, never ``0 s`` (FM-170, CLAUDE.md: unknown beats guessed). ``assumed`` are
    the plan card's own lines, drawn only beside a plan they are about. Plain text (FM-139).
    """
    lines = [objective_line(objective)]
    if plan is None:
        lines.append(
            "Fields: not laid out yet"
            if out_of_date is None
            else f"Fields: out of date ({out_of_date})"
        )
        return tuple(lines)
    # The label already says "Fields", so the plural carries the bare number (``72 in 2
    # wells``) and only one says its noun (``1 field in 1 well``): ``Fields: 1`` alone
    # would read as a count of nothing (m5 §17.10.3).
    count = "1 field" if plan.count == 1 else str(plan.count)
    lines.append(f"Fields: {count} in {_count(plan.wells, 'well')}")
    if plan.estimate_s is None:
        lines.append(TIME_NOT_KNOWN)
    else:
        lines.append(f"Time: about {duration_text(plan.estimate_s)}")
    lines.extend(f"assumed: {line}" for line in assumed)
    return tuple(lines)


def plan_selection_note(plan_wells: Sequence[str], selection: Sequence[str]) -> str:
    """What the screens say when the wells selected are not the plan's wells, or ``""`` (m5 §17.4.3).

    The plan is what ``scan`` does and a selection changed since has not changed it
    (§14.1): the note says which wells each is for, and what to do. Compared as sets: the
    order a selection was made in is not a difference. No plan, nothing to compare.
    """
    if not plan_wells or set(plan_wells) == set(selection):
        return ""
    laid_out = f"The fields laid out are for {wells_text(plan_wells)}; "
    if not selection:
        return laid_out + "no well is selected now."
    return (
        laid_out
        + f"the wells selected now are {wells_text(selection)}. "
        + "Lay out the fields again to scan these."
    )


def field_source_line(
    *,
    value: str,
    default: str,
    preset: str | None,
    preset_value: str | None,
    suggested: str | None,
) -> str:
    """Where a tool's field on the Wells screen gets its value, on one line under it (m5 §17.2 D7, §17.10.3).

    The first that holds: the field is empty, so the tool's own default applies (shown
    without the quotes of a string default, as the placeholder is); the value is the loaded
    preset's; it is the profile's suggestion; else the operator typed it here.
    """
    if not value.strip():
        return f"the tool's default: {plain_default(default)}"
    if preset is not None and value == preset_value:
        return f"from the preset {preset}"
    if suggested is not None and value == suggested:
        return "from the profile's suggestions"
    return "typed on this page"


def wells_from_line(
    selection: Sequence[str],
    preset: str | None,
    preset_selection: Sequence[str] | None,
) -> str:
    """Where the Wells box's selection comes from, on one line under it (m5 §17.10.3).

    No well chosen is said as such; the loaded preset's selection, while the box still
    holds exactly it; else the operator's own choice (a click on the map or a typed range).
    """
    if not selection:
        return "no well chosen yet"
    if (
        preset is not None
        and preset_selection is not None
        and tuple(preset_selection) == tuple(selection)
    ):
        return f"from the preset {preset}"
    return "chosen on this page"


def lay_out_wait(
    *,
    connected: bool,
    plate_chosen: bool,
    registered: bool,
    blocked: str | None,
    selected: int,
) -> str | None:
    """Why *Lay out the fields* waits, or ``None``: the first of the stand, the plate, the registration, the tool, the selection (m5 §17.10.3).

    ``blocked`` is the controller's reason for ``scan-plan`` (its order puts a closed stand
    and a plate nothing named first, and then what the tool reads), so a missing
    registration says the step to take (``REGISTER_FIRST``) and not the document's words.
    """
    if not connected or not plate_chosen:
        return blocked
    if not registered:
        return REGISTER_FIRST
    if blocked is not None:
        return blocked
    if selected == 0:
        return SELECT_WELLS
    return None


def scan_wait(*, connected: bool, blocked: str | None, plan_usable: bool) -> str | None:
    """Why *Start the scan* waits, or ``None``: the stand, then the plan, then the tool's own reason (m5 §17.10.3).

    ``blocked`` is the controller's reason for ``scan``, which is ``None`` when nothing
    waits (a checkpoint it will not start over is one such reason: the page never starts
    a stage afresh over one, §2).
    """
    if not connected:
        return blocked
    if not plan_usable:
        return LAY_OUT_FIRST
    return blocked


def scan_progress_lines(
    done: int, total: int, clock: RunClock | None, *, running: bool
) -> tuple[str, ...]:
    """The Scan screen's lines under the bar: the field, the time so far and the time left (m5 §17.4.4, §17.10.3).

    *Left* is the rate **between the first report and the latest** times the fields that
    remain, never the rate from the click: the demo's scan takes 4.6 s to its first field
    and 0.19 s for each of the next, so a rate from the click said 2 min 47 s at the second
    field, where 13 s were (measured, §17.10.1). It is drawn only while the scan runs and
    has fields left, and says it is not known until there are two reports a clock tick
    apart (FM-197: Windows below Python 3.13 ticks every 15.6 ms, so two reports can carry
    one stamp): never a division by zero, never a negative time.
    """
    lines = [f"Field {done} of {total}"]
    if clock is not None:
        lines.append(f"Elapsed {duration_text(max(0.0, clock.elapsed_s))}")
    if running and done < total:
        first, last = (None, None) if clock is None else (clock.first, clock.last)
        if (
            first is not None
            and last is not None
            and last[1] > first[1]
            and last[0] > first[0]
        ):
            left = (total - done) * (last[0] - first[0]) / (last[1] - first[1])
            lines.append(
                f"Left about {duration_text(left)} (estimated from the fields so far)"
            )
        else:
            lines.append("Left: not known yet (it needs two fields)")
    return tuple(lines)


def scanned_line(summary: ScanSummary) -> str:
    """The Scan screen's result line, from the usable ``scan`` record (m5 §17.4.4, D8: the verb of the button).

    A dry run walks the fields and takes no frame: it says so, rather than ``0 frames, 0
    failed``, which would read as a scan that found nothing.
    """
    if summary.dry_run:
        return "Scanned in a dry run: no frame was taken"
    return f"Scanned: {_count(summary.frames, 'frame')}, {summary.failed} failed"


#: A record's ``error`` is ``<Type>: <message>`` (the runner's ``_error_line``); the screen says
#: the message alone, and the type stays in the run document and the log (m5 §17.12 D20).
_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt): ")


def _without_type(text: str) -> str:
    """``text`` without a leading exception type name, on one line."""
    flat = _flat(text).strip()
    return _ERROR_TYPE.sub("", flat, count=1) or flat


def scan_end_lines(end: ScanEnd) -> tuple[str, ...]:
    """What the Scan screen says of a scan that ended ``stopped`` or ``failed`` (m5 §17.12 D17).

    A STOP is said as a stop whatever the record says: a STOP that lands in a move ends the
    record ``failed`` (FM-64), and ``end.kind`` is the controller's, which knows STOP was
    pressed. How far it got is the progress the controller kept of that scan; a page that kept
    none (opened after the stop, or another activity ran since) drops the count rather than
    say ``field None of None``. A failure says its one line without the exception's type.
    Plain text (FM-139): the error comes from a plugin.
    """
    counted = end.done is not None and end.total is not None
    if end.kind == "stopped":
        if counted:
            return (SCAN_STOPPED.format(done=end.done, total=end.total),)
        return (SCAN_STOPPED_UNCOUNTED,)
    text = _without_type(end.text) or "no reason was recorded"
    if counted:
        return (SCAN_FAILED.format(done=end.done, total=end.total, text=text),)
    return (SCAN_FAILED_UNCOUNTED.format(text=text),)


def scan_again_confirm(done: int | None) -> str:
    """What the first click on *Start again* says: the second one drops the checkpoint (m5 §17.12 D17, FM-172)."""
    if done is None:
        return SCAN_START_AGAIN_CONFIRM_UNCOUNTED
    return SCAN_START_AGAIN_CONFIRM.format(done=done)


def scan_actions_wait(
    *, connected: bool, halted: bool, running: str | None
) -> str | None:
    """Why *Continue the scan* and *Start again* wait, or ``None`` (m5 §17.12 D17).

    The first that holds: the stand is closed; it is halted (the run would fail at its first
    move: *Resume* comes first); an activity goes (``busy_note``, Live told apart by its name,
    the one activity that ends by a button of its own). The tool's other reasons are the
    runner's, said as the outcome at the click (``check_ready(..., resume=True)``).
    """
    if not connected:
        return NOT_CONNECTED
    if halted:
        return SCAN_HALTED
    if running is not None:
        return busy_note(running, live=running == "live")
    return None


def plain_default(default: str) -> str:
    """A tool field's default as a placeholder shows it: one pair of matching quotes dropped (m5 §17.10.3).

    ``describe_params`` prints a string default with its quotes (``'if-engaged'``), which
    the operator would type if the placeholder were a value to copy. Only one pair, and
    only a ``'`` or a ``"`` at both ends: anything else is as it came. A tuple's repr
    (``('single',)``, the default of a field that takes a list) is its items joined by
    ``, `` (``single``): the repr's parentheses, quotes and trailing comma are Python's, not
    the operator's words; a text that does not read as a tuple of literals, and an empty
    tuple, are left as they came (m5 §17.12.3).
    """
    if len(default) >= 2 and default[0] == default[-1] and default[0] in "'\"":
        return default[1:-1]
    if default.startswith("(") and default.endswith(")"):
        try:
            found = ast.literal_eval(default)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return default
        if isinstance(found, tuple) and found:
            return ", ".join(str(item) for item in found)
    return default


def replace_note(
    result: PlateResult, started_utc: datetime, *, keeps_captures: bool
) -> str:
    """What a *Run* of a tool that provides the plate replaces, said before the click (m5 §11.4).

    The latest attempt decides the ``plate`` slot even when it fails, so a failed
    run leaves no plate where the older one stood. The page does not refuse: the
    registration is a pure function of the captures, which stay, so the plate is
    had again by the same command minus the capture that was wrong. The time is
    the record's start in the PC's local time, to the minute, and the method and
    the count are the result's own.

    ``keeps_captures`` is whether the tool runs from the page's captures
    (``plate-registration``, ``plate-walls``): under a tool that has none
    (``plate-finding``) the sentence about forgetting one is false, and the note
    ends after the failed run (m5 §12.7). It is required, so no call forgets it.
    """
    try:
        when = started_utc.astimezone().strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        # A start the PC cannot show in its own zone (Windows before 1970 or after
        # year 3000; any system at year 1 or 9999 under an offset): the record's own
        # value, as it is, not an exception that ends the page's redraw. A run
        # document is data (FM-125) and ``view()`` raises nothing for it.
        when = started_utc.isoformat(timespec="minutes")
    wells = "1 well" if result.n_refs == 1 else f"{result.n_refs} wells"
    note = (
        f"Run replaces the plate registered {when} ({result.method}, {wells}). "
        "A failed run leaves no plate"
    )
    if not keeps_captures:
        return note + "."
    return (
        note
        + "; the captures stay, so Forget the wrong one and Run again to get a plate back."
    )


def command_line(
    plugin: str,
    params: Sequence[str],
    *,
    profile: str,
    run_doc: Path,
    dry_run: bool,
    preset: str | None = None,
    resume: bool = False,
) -> str:
    """The ``smc run`` command that makes the call a step makes, quoted for a POSIX shell.

    ``shlex.join`` makes a JSON value one single-quoted word, so the operator
    copies it whole. How it pastes into PowerShell or ``cmd.exe`` is not verified
    (m5 §9.1); the run document's record of the run is the reference, whichever
    shell reads the line.

    ``preset`` is what ``smc run --preset`` takes (the page passes the file's
    path): given, the terminal's run applies the preset as the page's does, and
    its record names the preset as the page's would (m5 §10.5). The pairs that
    follow repeat what the preset gives, so the command says all it runs.

    ``resume`` adds ``--resume``: the command that continues a stage's checkpoint, which
    the same command without it drops (m5 §17.12 D17).
    """
    words = ["smc", "run", plugin, "-p", profile, "--run-doc", str(run_doc)]
    if dry_run:
        words.append("--dry-run")
    if resume:
        words.append("--resume")
    if preset is not None:
        words += ["--preset", preset]
    for pair in params:
        words += ["--param", pair]
    return shlex.join(words)


def preset_differences(
    expected: Mapping[str, Mapping[str, str]],
    actual: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    """What the page's fields hold that the loaded preset's load did not set, one line each (m5 §10.6).

    ``expected`` is what the load put in every field (the preset's value, else
    the profile's suggestion, else ``""``), per tool in the page's order, and
    ``actual`` what the fields hold now. A line is
    ``<tool> <key>: <value> (the preset: <expected>)`` for each field whose text,
    stripped, is not the expected one, with ``(default)`` for an empty text: the
    plugin's own default is what an empty field sends. A field the page no longer
    shows is read as empty. A value is shown on one line and cut, since a field
    may hold the whole of a JSON list; it is compared whole.
    """
    lines: list[str] = []
    for plugin, fields in expected.items():
        now = actual.get(plugin, {})
        for key, given in fields.items():
            # Both sides stripped: what a run sends is the stripped text, and a value
            # a hand edit left a space in is not a difference.
            wanted = given.strip()
            found = now.get(key, "").strip()
            if found != wanted:
                lines.append(
                    f"{plugin} {key}: {_shown(found)} (the preset: {_shown(wanted)})"
                )
    return tuple(lines)


def preset_note_text(loaded: LoadedPreset | None, differences: Sequence[str]) -> str:
    """The note under the *Preset* region: which preset, what it could not apply, what changed since.

    Plain text with line breaks, never markup (FM-139): the names come from a
    file a person edited. The mismatches and the differences are each cut at
    ``_NOTE_LINES`` lines, then ``… <n> more``, as a result card's lines are.
    """
    if loaded is None:
        return "no preset loaded"
    lines = [f"loaded: {loaded.file.name}  ({loaded.file.path})"]
    lines += [f"  {line}" for line in _cut_lines(loaded.mismatches)]
    if differences:
        lines.append("changed since loading:")
        lines += [f"  {line}" for line in _cut_lines(differences)]
    else:
        lines.append("the page holds the preset's values")
    return "\n".join(lines)


#: A note shows this many lines of each kind, as a result card shows its values (§2.4).
_NOTE_LINES = 8
#: A value in a difference is shown at most this long: a field may hold a whole JSON list.
_VALUE_CHARS = 80


def _cut_lines(lines: Sequence[str]) -> list[str]:
    if len(lines) <= _NOTE_LINES:
        return [_flat(line) for line in lines]
    return [
        *(_flat(line) for line in lines[:_NOTE_LINES]),
        f"… {len(lines) - _NOTE_LINES} more",
    ]


def _flat(text: str) -> str:
    """``text`` on one line: its line breaks are spaces."""
    return " ".join(text.split("\n")).replace("\r", " ")


def _shown(text: str) -> str:
    """A field's text as a difference shows it: ``(default)`` when empty, one line, cut."""
    if not text:
        return "(default)"
    flat = _flat(text)
    return flat if len(flat) <= _VALUE_CHARS else flat[:_VALUE_CHARS] + "…"


# -- the Results screen (m5 §17.4.5, §17.11, #446) ---------------------------------------

#: The tools of the Results screen in the checklist's order. Each is the slot it provides.
RESULT_TOOLS: tuple[str, ...] = ("cells", "density", "route", "report", "nis-export")

#: What each row calls its tool (m5 §17.4.5).
RESULT_NAMES: Mapping[str, str] = {
    "cells": "Count the cells",
    "density": "Density per well",
    "route": "Route through the densest fields",
    "report": "Report",
    "nis-export": "Export for NIS-Elements",
}

#: What each tool reads: the page's copy of the plugins' ``reads``, as ``WALLS_MIN_CENTRES``
#: is of ``plate-walls``' bounds, pinned by a test. A row finds the input it waits for from
#: this and the checklist, and never from the run document's sentence (``the run document
#: has no cells; smc run --list ...``), which is for a terminal. The guide stays pure: it
#: does not read the registry.
RESULT_READS: Mapping[str, tuple[str, ...]] = {
    "cells": ("scan",),
    "density": ("scan-plan", "scan", "cells"),
    "route": ("scan-plan", "scan", "cells"),
    "report": ("scan-plan", "scan", "cells", "density"),
    "nis-export": ("route",),
}

#: What each tool reads **if it is there** (the plugins' ``reads_optional``), pinned by the same
#: test. A tool that read an optional slot is out of date when that slot is made again, as one
#: that required it is (``RunDocument.is_stale``): ``report`` reads the route when there is one,
#: so a new route makes it stale. It is for ``again_note`` only: an optional input that is not
#: there never makes a row wait, so a row's state line does not read it.
RESULT_OPTIONAL_READS: Mapping[str, tuple[str, ...]] = {
    "cells": (),
    "density": (),
    "route": ("timings",),
    "report": ("route",),
    "nis-export": (),
}

#: The parameters a row shows under *Choices and command*, which a profile or a preset fills.
RESULT_CHOICES: Mapping[str, tuple[str, ...]] = {
    "cells": ("detector",),
    "route": ("labels", "fields_per_well"),
}

#: What a row says while the scan it reads has not run: its step is the screen before.
SCAN_FIRST = "scan first: step 4, Scan"

#: The label of the chain's button and what it says when there is none to run.
RUN_ALL_LABEL = "Run all that are left"
NOTHING_LEFT = "Nothing is left: every step is done."

#: The heading of the files block, and what it says before any step wrote one.
FILES_HEADING = "Where the files are"
NO_FILES_YET = "No step has written a file yet."


@dataclass(frozen=True, slots=True)
class ResultRow:
    """What one row of the Results screen says (m5 §17.11.3).

    ``state`` is the state line; ``button`` is ``Again`` exactly when the tool is done, which
    ``done`` says too; ``wait`` is what it waits for beside its button (``""`` when nothing),
    and ``missing`` is true exactly when an input it reads is not done, which ``state`` then
    says and ``wait`` does not repeat.
    """

    plugin: str
    name: str
    state: str
    button: Literal["Run", "Again"]
    wait: str
    missing: bool
    done: bool


def _result_name(plugin: str) -> str:
    """A tool's plain name: the row's, else its plain title (``Timing``, D20), else its own name."""
    return RESULT_NAMES.get(plugin) or tool_title(plugin)


def _names(plugins: Sequence[str]) -> str:
    return ", ".join(_result_name(plugin) for plugin in plugins)


def _is_ok(items: Mapping[str, CheckItem], slot: str) -> bool:
    item = items.get(slot)
    return item is not None and item.state == "ok"


def results_left(checklist: Sequence[CheckItem]) -> tuple[str, ...]:
    """The tools of the Results screen that are not ``ok``, in the checklist's order (m5 §17.11.3).

    A tool that is ``not run`` and one whose result is ``unusable`` are both left: a chain
    runs them. One that is ``ok`` is never run by a chain, since running it again would make
    every tool that read it out of date.
    """
    return tuple(
        item.slot
        for item in checklist
        if item.slot in RESULT_TOOLS and item.state != "ok"
    )


def result_row(
    plugin: str,
    *,
    checklist: Sequence[CheckItem],
    card: ResultCard | None,
    blocked: str | None,
    dry_scan: bool,
    running: str | None,
    progress: tuple[int, int | None] | None,
) -> ResultRow:
    """One row of the Results screen: the tool's state line, its button and what it waits for (m5 §17.11.3).

    The state line is the first that holds: the tool runs now (with its own count, and
    without a total when the plugin reports none); it is done, when and whether in a dry
    run; its result is not usable, with the run document's reason; an input it reads is not
    done, named by the step that makes it; else ``Ready``. ``blocked`` is the controller's
    reason (the stand is closed, a checkpoint, a capability), and ``dry_scan`` is a page
    served with ``--dry-run`` over a scan that was one: such a scan took no frame, so a tool
    that reads ``scan`` has nothing to read (m5 §8.8.1).
    """
    items = {item.slot: item for item in checklist}
    item = items.get(plugin)
    done = item is not None and item.state == "ok"
    missing = False
    if running == plugin:
        if progress is None:
            state = "Running now"
        elif progress[1] is None:
            state = f"Running now: {progress[0]}"
        else:
            state = f"Running now: {progress[0]} of {progress[1]}"
    elif done:
        state = "Done"
        if card is not None:
            state = (
                f"Done {_flat(card.when_local)}"
                if card.when_local
                else f"Done {_flat(card.when_utc)} UTC"
            )
            if card.dry_run:
                state += " (dry run)"
    elif item is not None and item.state == "unusable":
        state = "Not usable"
        if item.reason:
            state = f"Not usable: {_flat(item.reason)}"
    else:
        absent = next(
            (slot for slot in RESULT_READS.get(plugin, ()) if not _is_ok(items, slot)),
            None,
        )
        if absent is None:
            state = "Ready"
        else:
            missing = True
            if absent == "scan-plan":
                state = LAY_OUT_FIRST
            elif absent == "scan":
                state = SCAN_FIRST
            elif absent in RESULT_TOOLS:
                state = f"after {_result_name(absent)}"
            else:
                state = f"after {absent}"
    if missing:
        wait = ""
    elif blocked is not None:
        wait = blocked
    elif dry_scan and "scan" in RESULT_READS.get(plugin, ()):
        wait = DRY_RUN_NO_FRAME
    else:
        wait = ""
    return ResultRow(
        plugin=plugin,
        name=_result_name(plugin),
        state=state,
        button="Again" if done else "Run",
        wait=wait,
        missing=missing,
        done=done,
    )


def run_all_wait(rows: Sequence[ResultRow], left: Sequence[str]) -> str | None:
    """Why *Run all that are left* waits, or ``None`` when it can start (m5 §17.11.3).

    The chain starts with the first tool left and stops at once if that tool cannot run, so
    the button waits for what that tool waits for: nothing is left, an input is missing (the
    row's state line says which step makes it), or the tool's own wait.
    """
    if not left:
        return NOTHING_LEFT
    first = next((row for row in rows if row.plugin == left[0]), None)
    if first is None:
        return None
    if first.missing:
        return first.state
    return first.wait or None


def again_note(plugin: str, checklist: Sequence[CheckItem]) -> str:
    """What *Again* replaces, or ``""`` (m5 §17.11.3; §11.4's idea, for the one place a click replaces a result others read).

    Named are the tools after this one that are done and read it, directly or through
    another: running it again makes their results out of date, and *Run all that are left*
    runs them again. A tool no later tool reads (``report``, ``nis-export``) says nothing.
    The rule is the plugins' ``reads`` and ``reads_optional`` (``RESULT_READS``,
    ``RESULT_OPTIONAL_READS``) and not "every tool after it": ``route`` does not read
    ``density``, so a new density leaves the route as it was, while ``report`` reads the
    route when there is one, so a new route makes it out of date.
    """
    items = {item.slot: item for item in checklist}
    if plugin not in RESULT_TOOLS or not _is_ok(items, plugin):
        return ""
    reached = {plugin}
    later: list[str] = []
    for tool in RESULT_TOOLS[RESULT_TOOLS.index(plugin) + 1 :]:
        if reached.intersection(RESULT_READS[tool]) or reached.intersection(
            RESULT_OPTIONAL_READS[tool]
        ):
            reached.add(tool)
            if _is_ok(items, tool):
                later.append(tool)
    if not later:
        return ""
    return AGAIN_NOTE.format(names=", ".join(RESULT_SHORT[tool] for tool in later))


def _sentence(text: str) -> str:
    """``text`` on one line and ending with a full stop: an outcome comes from outside and may end with none."""
    flat = _flat(text).strip()
    return flat if flat.endswith(".") else flat + "."


def chain_line(chain: ChainStatus | None) -> str:
    """The one line that says what *Run all that are left* is doing or did (m5 §17.11.3).

    ``""`` before any chain. While a tool goes it says which, and how far along. After the
    end: all done, or where it stopped and why (the tool failed, was refused, or STOP landed
    in it or between two tools), what was done and what was not run, in the tools' plain
    names. Plain text, never markup: an outcome and a name come from outside (FM-139). A chain
    whose own code raised is told by the strip's outcome; its line reads like a stop.
    """
    if chain is None or not chain.plugins:
        return ""
    if chain.current is not None:
        return (
            f"Running {_result_name(chain.current)} "
            f"({len(chain.ended) + 1} of {len(chain.plugins)})."
        )
    done = [end.plugin for end in chain.ended if end.how == "done"]
    after = chain.plugins[len(chain.ended) :]
    last = chain.ended[-1] if chain.ended else None
    if last is not None and last.how == "done" and not after:
        return f"Done: {_names(done)}."
    if last is None or last.how == "done":
        # A STOP between two tools (or before the first): nothing was refused or failed.
        head = f"Stopped before {_result_name(after[0])} by STOP."
        not_run = after[1:]
    else:
        name = _result_name(last.plugin)
        not_run = after
        if last.how == "failed":
            # A tool's own outcome begins with its title (``Route failed: ...``, D20): it
            # already says which tool and that it failed, so it is the head as it is.
            if last.outcome.startswith(f"{tool_title(last.plugin)} failed"):
                head = _sentence(last.outcome)
            else:
                head = _sentence(f"Stopped at {name}, which failed: {last.outcome}")
        elif last.how == "refused":
            head = _sentence(f"Stopped at {name}, which cannot run: {last.outcome}")
        else:
            head = f"Stopped at {name} by STOP."
    parts = [head]
    if done:
        parts.append(f"Done: {_names(done)}.")
    if not_run:
        parts.append(f"Not run: {_names(not_run)}.")
    return " ".join(parts)


def files_lines(cards: Mapping[str, ResultCard]) -> tuple[str, ...]:
    """A line for each file a step wrote: ``<the step's plain name>: <path>`` (m5 §17.11.3).

    The scan first (``Scan frames``), then each tool of the screen in its order. A card
    holds files only when its record ended ``done`` (FM-154), so a stopped or failed tool
    names none. A card's entry is ``<key>: <path>``: the key (``file``, ``folder``) is the
    run document's word and not the operator's, so the path is what follows the first
    ``": "`` (the whole entry when there is none).
    """
    lines: list[str] = []
    for plugin in ("scan", *RESULT_TOOLS):
        card = cards.get(plugin)
        if card is None:
            continue
        name = "Scan frames" if plugin == "scan" else _result_name(plugin)
        for entry in card.files:
            _key, separator, path = entry.partition(": ")
            lines.append(f"{name}: {_flat(path if separator else entry)}")
    return tuple(lines)


def colour_bar_title(label: str | None) -> str:
    """The map's colour bar title: the class the *Density of* select names, or nothing (m5 §17.4.5, #293 item 6)."""
    if not label:
        return ""
    return f"{_flat(label)} cells per mm²"


# -- round 2: plain words, notes drawn once, the map's frame, the files (m5 §17.12, #475) ---

#: The plain title of each built-in tool, for the outcome line (m5 §17.12 D20). A tool not
#: listed (a third party's) is named by its plugin name.
TOOL_TITLES: Mapping[str, str] = {
    "camera-calibration": "Camera calibration",
    "timing": "Timing",
    "goto-well": "Go to",
    "plate-registration": "Registration",
    "plate-walls": "Registration from the walls",
    "plate-finding": "Plate finding",
    "plate-orientation": "Plate orientation",
    "scan-plan": "Field layout",
    "scan": "Scan",
    "cells": "Count the cells",
    "density": "Density per well",
    "route": "Route",
    "report": "Report",
    "nis-export": "Export for NIS-Elements",
    "stage-report": "Stage report",
    # Two tools the design's table did not list; the test of the table asks for every one
    # the repository ships.
    "focus-on-cells": "Focus on cells",
    "objective-offsets": "Objective offsets",
}

#: The short names the *Again* note lists (D20).
RESULT_SHORT: Mapping[str, str] = {
    "cells": "Count",
    "density": "Density",
    "route": "Route",
    "report": "Report",
    "nis-export": "Export",
}

#: A frame larger than this many pixels (1600 x 1600) is tested on every other row and column:
#: ``carries_signal`` takes 16.5 ms on the Prime 95B's 1200 x 1200 and 70.8 ms on 2048 x 2048
#: (measured, m5 §17.12 D22), and the test runs once per new frame.
_FULL_TEST_PIXELS = 1600 * 1600

#: Two captures closer than this fraction of a pitch are at one place. Not one pitch: two
#: neighbouring wells, each centred by eye, are one pitch apart and are right.
_SAME_PLACE_PITCH_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class GotoAnswer:
    """What the last *Go to* said, for the Plate row it was sent from (m5 §17.12.3).

    ``phase`` is ``going`` while the run goes, ``at`` once it ended ``done``, ``refused``
    when the page refused it before anything started, ``failed`` when it ended any other
    way (a failure, a STOP). ``estimated`` is whether the target was the holder's estimate
    and not a registered plate, so that the row says to centre it by eye. ``text`` is the
    refusal's or failure's one line, without the tool's name (``""`` for the other phases).
    It is a value, replaced whole and never edited, so that a tab reads one or the other
    (FM-151).
    """

    well: str
    phase: Literal["going", "at", "refused", "failed"]
    estimated: bool
    text: str


def tool_title(plugin: str) -> str:
    """A tool's plain title for the outcome line; a plugin the table has not is named by its own name (m5 §17.12 D20)."""
    return TOOL_TITLES.get(plugin, plugin)


def local_time(when_utc: datetime) -> str:
    """``when_utc`` in the PC's own zone, to the minute, as the guided screens say a time (m5 §17.12 D20).

    A naive datetime is taken as UTC (the run document's times are; one without a zone
    would otherwise be read as the PC's). A time the PC cannot show in its zone (Windows
    before 1970 or after year 3000, year 1 or 9999 under an offset) is the value's own
    ISO text, never an exception: a run document is data (FM-125) and ``view()`` raises
    nothing for it.
    """
    aware = (
        when_utc
        if when_utc.tzinfo is not None
        else when_utc.replace(tzinfo=timezone.utc)
    )
    try:
        return aware.astimezone().strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return aware.isoformat(timespec="minutes")


def frame_hint(image: npt.NDArray[Any], bit_depth: int) -> str:
    """What the image says about a frame that carries no useful light, or ``""`` (m5 §17.12 D22, #457).

    ``bit_depth`` is the camera's: a frame whose depth is not known (zero, a plugin's frame)
    is not judged, since the full scale and the dark floor both depend on it. Saturated is
    asked **before** dark: a frame at full scale everywhere spans no counts, and the light
    path is not its remedy. A frame larger than 1600 x 1600 is tested on every other row and
    column, which is a quarter of the work and the same verdict for a frame (a dark one is
    dark everywhere). A frame that is not a 2-D array of numbers says nothing.
    """
    if bit_depth <= 0:
        return ""
    sample = image[::2, ::2] if image.size > _FULL_TEST_PIXELS else image
    try:
        if saturated(sample, bit_depth):
            return FRAME_SATURATED
        if not carries_signal(sample, bit_depth):
            return FRAME_DARK
    except ValueError:
        return ""
    return ""


def capture_note(
    well: str,
    captured_um: tuple[float, float],
    expected_um: tuple[float, float] | None,
    pitch_um: float,
    others: Mapping[str, tuple[float, float]],
) -> str:
    """A note on one captured centre that is far from its well or on another's place, or ``""`` (m5 §17.12.3).

    ``expected_um`` is where the map's pose puts the centre of this well (registered, seed or
    estimated), ``None`` on an *unplaced* map, which puts none: then nothing is far. A capture
    more than one pitch of the plate from it is ``CAPTURE_FAR``: it is another well's place
    (H1 captured at A12's, 99 mm off, on the owner's walk). A capture within half a pitch of
    another well's capture is ``CAPTURE_SAME_PLACE``: two wells cannot be closer than a pitch,
    so one of the two is wrong. The first that holds. A note, never a refusal: a bad fit is
    refused by the registration's own residual bound, and a capture by eye is the operator's.
    ``others`` are the other wells' captures, never this one's.
    """
    if expected_um is not None:
        distance_um = _distance_um(captured_um, expected_um)
        if distance_um > pitch_um:
            return CAPTURE_FAR.format(well=well, distance_mm=distance_um / 1000.0)
    for other, place in others.items():
        if other != well and (
            _distance_um(captured_um, place) < _SAME_PLACE_PITCH_FRACTION * pitch_um
        ):
            return CAPTURE_SAME_PLACE.format(well=well, other=other)
    return ""


def _distance_um(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def register_objective_refusal(
    objectives: Mapping[str, str | None], in_place: str | None
) -> str | None:
    """Why *Register the plate* is refused for the objectives of its captures, or ``None`` (m5 §17.12 D13, #373).

    A plate registered from centres captured with one objective and then another puts the
    wells where neither objective saw them, and the offsets of #382 read the objective a
    registration records. Only the **known** labels are compared: a capture restored from a
    run document, or taken while the turret could not be read, has none and refuses nothing,
    and a turret that is not read (``in_place`` is ``None``) refuses the turn nothing (the
    tool's own ``assumed`` says the objective). Captures made with two objectives are refused
    whatever the turret says. Plain text: the labels are the stand's (FM-139).
    """
    known = list(dict.fromkeys(label for label in objectives.values() if label))
    if len(known) > 1:
        which = ", ".join(known[:-1]) + f" and {known[-1]}"
        return REGISTER_OBJECTIVES_DIFFER.format(which=which)
    if not known or in_place is None or known[0] == in_place:
        return None
    return REGISTER_ACROSS_A_TURN.format(captured=known[0], in_place=in_place)


def goto_row_text(answer: GotoAnswer | None, well: str) -> str:
    """What the Plate row of ``well`` says of the *Go to* that was sent from it; ``""`` for another row (m5 §17.12.3).

    The row of the well the drive was sent to, and no other: the answer is one for the whole
    page. The refusal's or failure's line is the controller's, without the tool's name.
    """
    if answer is None or answer.well != well:
        return ""
    if answer.phase == "going":
        return GOTO_ROW_GOING.format(well=well)
    if answer.phase == "at":
        template = GOTO_ROW_AT_ESTIMATE if answer.estimated else GOTO_ROW_AT
        return template.format(well=well)
    return answer.text


def map_frame(
    outlines: Sequence[tuple[float, float, float, float]], pitch_um: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The frame the map is drawn in: the bounding box of ``outlines`` grown by one pitch, as ``((x0, x1), (y0, y1))`` (m5 §17.12 D16).

    Each outline is ``(x0, y0, x1, y1)`` in stage µm, a well's (or the plate's). The margin is
    one pitch on each side, so that the side labels and the stage's cross have room inside
    the frame.

    Raises:
        ValueError: ``outlines`` is empty: a frame of nothing is no frame, and the caller keeps
            the one it has.
    """
    if not outlines:
        raise ValueError("a frame needs at least one outline")
    x0 = min(min(o[0], o[2]) for o in outlines) - pitch_um
    x1 = max(max(o[0], o[2]) for o in outlines) + pitch_um
    y0 = min(min(o[1], o[3]) for o in outlines) - pitch_um
    y1 = max(max(o[1], o[3]) for o in outlines) + pitch_um
    return ((x0, x1), (y0, y1))


def plate_rows_note(*, running: str | None, live: bool, from_sample: bool) -> str:
    """The one note above the Plate screen's centre rows: what holds for every row, once (m5 §17.12 D19, D23).

    The busy sentence and ``GOTO_NOT_DURING_LIVE`` were drawn under each row, four times
    during a *Go to*. Live is told apart from any other activity, as ``busy_note`` does.
    ``from_sample`` adds what *Register the plate* does while nothing is captured.
    """
    lines: list[str] = []
    if live:
        lines.append(GOTO_NOT_DURING_LIVE)
    elif running is not None:
        lines.append(busy_note(running, live=False))
    if from_sample:
        lines.append(SAMPLE_REGISTERS)
    return "\n".join(lines)


def _inside(path_text: str, run_folder: PurePath) -> PurePath | None:
    """``path_text`` relative to ``run_folder``, or ``None`` when it is not a path inside it.

    The text is parsed in the run folder's own flavour, so that a Windows path is read as one
    on any PC (a page served on Windows names its files with backslashes); a ``..`` is no
    file of the folder, whatever the text says.
    """
    try:
        relative = type(run_folder)(path_text).relative_to(run_folder)
    except (ValueError, TypeError):
        return None
    if not relative.parts or ".." in relative.parts:
        return None
    return relative


def files_html(cards: Mapping[str, ResultCard], run_folder: PurePath | None) -> str:
    """*Where the files are* as HTML: a link for each file inside the run folder, the others as text (m5 §17.12 D21).

    The one HTML pane of the page, so that every text in it is escaped with ``html.escape``
    (FM-139: a file name is data, and ``<``, ``&`` and ``"`` are legal in one). A ``file`` of a
    card inside ``run_folder`` is a link to ``/run-files/<its path relative to the folder, as
    POSIX and URL-quoted>``, opened in a new tab; a Windows relative path is quoted as POSIX
    too, the route takes ``/``. A ``folder`` (the scan's frames), a file outside the run folder
    (``nis-export``'s absolute ``out=``) and a path that cannot be read as one stay text: the
    route serves only the files of the run document (``PlatePage.run_file``). Without
    any file, ``NO_FILES_YET``.
    """
    lines: list[str] = []
    for plugin in ("scan", *RESULT_TOOLS):
        card = cards.get(plugin)
        if card is None:
            continue
        name = html.escape("Scan frames" if plugin == "scan" else _result_name(plugin))
        for entry in card.files:
            key, separator, path = entry.partition(": ")
            text = _flat(path if separator else entry)
            relative = (
                _inside(text, run_folder)
                if run_folder is not None and separator and key == "file"
                else None
            )
            if relative is None:
                lines.append(f"<div>{name}: {html.escape(text)}</div>")
                continue
            href = "/run-files/" + quote(relative.as_posix(), safe="/")
            shown = html.escape(relative.as_posix())
            lines.append(
                f'<div>{name}: <a href="{html.escape(href)}" target="_blank" '
                f'rel="noopener">{shown}</a></div>'
            )
    if not lines:
        return f"<div>{html.escape(NO_FILES_YET)}</div>"
    return "".join(lines)
