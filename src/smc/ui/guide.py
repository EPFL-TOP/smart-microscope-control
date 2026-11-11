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
from typing import TYPE_CHECKING

from smc.plugins.frames import image_direction_of_stage_step, stage_side_of_image_side
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
        WellView,
    )

__all__ = [
    "CAPTURE_WELLS",
    "DRY_RUN_NO_FRAME",
    "END_LIVE_LABEL",
    "MAP_FIELDS_UP_TO",
    "NO_CALIBRATION_FOR_SIDES",
    "ORIENTATION_IN_STAGE_TERMS",
    "PLATE_NOT_CHOSEN",
    "SELECTION_KEPT_ITS_NAMES",
    "SELECT_WELLS",
    "WALLS_DRY_RUN",
    "WALLS_MAX_CENTRES",
    "WALLS_MIN_CENTRES",
    "WALLS_NEED_CALIBRATION",
    "NextStep",
    "a1_corner",
    "busy_note",
    "capture_centre_label",
    "capture_wall_label",
    "centres_text",
    "command_line",
    "drive_image_note",
    "duration_text",
    "forget_label",
    "image_sides_note",
    "map_title",
    "next_slot",
    "orientation_label",
    "plan_title",
    "preset_differences",
    "preset_note_text",
    "replace_note",
    "scan_plan_card_lines",
    "side_words",
    "stage_label",
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
