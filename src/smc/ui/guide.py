"""What the plate page says to do next: a value and a command, no widget (m5 §8.3).

The guide is a function of the run document's checklist, the page's selection and
captures, and the profile; it keeps no state, and the page draws it above its tools.
It holds no Panel and no device, so every sentence it builds is tested as text. Its
one promise is the one the page's *Run* button keeps: the pairs the step shows are the
pairs it runs, and ``command_line`` is the same call typed in a terminal (principle 5).
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from smc.plugins.frames import image_direction_of_stage_step, stage_side_of_image_side
from smc.plugins.plate_registration.algorithm import suggested_refs
from smc.plugins.slots import WellGeometry

if TYPE_CHECKING:
    from datetime import datetime

    from smc.plugins.geometry import Matrix2
    from smc.plugins.slots import PlateResult, ScanPlanResult, WellRef
    from smc.plugins.slots.plate import PlateOrientation
    from smc.ui.controller import (
        CheckItem,
        LoadedPreset,
        MapPose,
        PlanLayer,
        PoseKind,
        WellView,
    )

__all__ = [
    "CAPTURE_WELLS",
    "DRY_RUN_NO_FRAME",
    "END_LIVE_LABEL",
    "GOTO_ESTIMATED",
    "GOTO_NOT_DURING_LIVE",
    "GOTO_UNPLACED",
    "GOTO_WALL_NEEDS_REGISTRATION",
    "MAP_FIELDS_UP_TO",
    "NO_CALIBRATION_FOR_SIDES",
    "ORIENTATION_IN_STAGE_TERMS",
    "OTHER_WAYS_TO_REGISTER",
    "PLATE_NOT_CHOSEN",
    "SCREEN_LEADS",
    "SCREEN_TITLES",
    "SELECTION_KEPT_ITS_NAMES",
    "SELECT_WELLS",
    "STEPS",
    "STEP_OF_SLOT",
    "STEP_TITLES",
    "WALLS_DRY_RUN",
    "WALLS_MAX_CENTRES",
    "WALLS_MIN_CENTRES",
    "WALLS_NEED_CALIBRATION",
    "CalibrationNote",
    "CentreRow",
    "NextStep",
    "StepName",
    "StepState",
    "a1_corner",
    "busy_note",
    "capture_centre_label",
    "capture_wall_label",
    "centre_rows",
    "centres_text",
    "command_line",
    "continue_label",
    "drive_image_note",
    "duration_text",
    "forget_label",
    "image_sides_note",
    "known_text",
    "map_title",
    "needs_calibration",
    "next_slot",
    "opening_step",
    "orientation_label",
    "plan_title",
    "plate_from_line",
    "preset_differences",
    "preset_note_text",
    "register_replace_note",
    "registered_line",
    "replace_note",
    "scan_plan_card_lines",
    "side_words",
    "stage_label",
    "step_button_label",
    "step_states",
    "suggested_centres",
    "wall_touches_note",
    "walls_card_lines",
    "walls_centres_problem",
    "walls_square_wells",
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

#: The question each screen answers (m5 §17.4, D8). The scan's title is static until the
#: Wells and Scan screens name the wells (B3, #445).
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

#: A step's summary is one line under its button and never wider than the button's column.
_SUMMARY_CHARS = 40


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
    unknown-time placeholder): the page says the zone beside it.
    """

    objective: str | None
    when_utc: str


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


def known_text(
    *,
    objective: str | None,
    calibration: CalibrationNote | None,
    light_on: bool | None,
) -> str:
    """The "known" block of the Stand screen: three lines, each saying what it rests on (m5 §17.2 D7, §17.4.1).

    Plain text, never markup (FM-139): the turret's label is the stand's. A value the
    page has not read says ``not known`` and a camera with no calibration says what the
    operator loses (the arrows cannot say which way the image moves): an unknown is
    never shown as a value (CLAUDE.md). ``objective`` is the one in place as the strip
    reads it, ``calibration`` the latest usable one, ``light_on`` the light as read.
    """
    first = (
        "Objective: not known"
        if objective is None
        else f"Objective: {objective}, in place"
    )
    if calibration is None:
        camera = (
            "Camera: not calibrated (the arrows cannot say which way the image moves)"
        )
    elif calibration.objective is None:
        camera = (
            f"Camera: calibrated on {calibration.when_utc} UTC, objective not recorded"
        )
    else:
        camera = f"Camera: calibrated at {calibration.objective} on {calibration.when_utc} UTC"
        if objective is not None and objective != calibration.objective:
            camera += f", not at {objective}"
    light = (
        "Light: not known"
        if light_on is None
        else f"Light: {'on' if light_on else 'off'}"
    )
    return "\n".join((first, camera, light))


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
    zero (CLAUDE.md). The rotation is the plate's own, to a hundredth of a degree.
    """
    residual = (
        "residual not measured"
        if result.residual_um is None
        else f"residual {result.residual_um:.1f} µm"
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
        return (
            "Which way the image moves for an arrow is not known: run "
            "camera-calibration (Set up)."
        )
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
    """
    words = ["smc", "run", plugin, "-p", profile, "--run-doc", str(run_doc)]
    if dry_run:
        words.append("--dry-run")
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
