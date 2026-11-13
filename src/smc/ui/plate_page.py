"""The plate page: Panel widgets over ``PlatePage`` (m3 §15, m5 §3, ADR-0006).

The only module besides the ``smc ui`` command that imports Panel and
Bokeh. It holds no decision: a tap, a button and a typed range each call the
controller, and an exception the controller raises for the operator's mistake is
shown as its message and kept as nothing else (FM-98). Four rules the page
keeps, each learned by measuring (m3 §15.3, #277):

* **The tick never rebuilds the view.** The periodic callback reads
  ``PlatePage.status()`` once a second and redraws the strip; the plate's wells
  are redrawn only when something changed them (a click, a typed range, a
  capture, the end of a run), because building 1536 wells takes 339 ms. The
  image is drawn by a callback of its own, every 0.1 s, that compares
  ``PlatePage.frame().number`` and reads nothing else: a frame is drawn only
  when its number is one the session has not drawn, whichever tab took it
  (FM-151), and a callback that sees the same number draws nothing (FM-145).
* **Bokeh models are touched only by server callbacks**, never by the run's
  worker thread, which only fills a queue.
* **A closed or reloaded tab does nothing.** The controller belongs to the
  server; a session's widgets and its periodic callback are the tab's.
* **A handler is registered through ``PlatePageView._handler``**: Panel runs one
  without the document lock when an event flushes a change it had queued (the
  Enter of a text field), and the page's own Bokeh models are then written
  outside the lock (FM-171). The periodic callbacks are locked and are not
  wrapped.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import mimetypes
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import panel as pn
import tornado.web
import useq
from bokeh.events import Tap
from bokeh.models import (
    BasicTickFormatter,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LabelSet,
    LinearColorMapper,
)
from bokeh.palettes import Greys256, Viridis256
from bokeh.plotting import figure

from smc.hardware.capabilities import (
    Autofocus,
    Camera,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.plugins.presets import PresetError, PresetExistsError
from smc.plugins.slots import IMAGE_SIDES
from smc.plugins.slots.plate import ORIENTATION_DEG, PlateOrientation
from smc.ui.controller import (
    NOT_CONNECTED,
    LoadedPreset,
    PageServerError,
    PageStatus,
    PlatePage,
    PlateView,
    ResultCard,
    ToolView,
    WellView,
)
from smc.ui.frames import Frame, display_image
from smc.ui.guide import (
    AUTOFOCUS_WORDS,
    COVERAGE_WORDS,
    DRY_RUN_NO_FRAME,
    END_LIVE_LABEL,
    FILES_HEADING,
    NO_FILES_YET,
    ORIENTATION_IN_STAGE_TERMS,
    OTHER_WAYS_TO_REGISTER,
    PLATE_NOT_CHOSEN,
    RESULT_CHOICES,
    RESULT_NAMES,
    RESULT_TOOLS,
    RUN_ALL_LABEL,
    SCREEN_LEADS,
    SCREEN_TITLES,
    STEPS,
    CentreRow,
    NextStep,
    ResultRow,
    StepName,
    again_note,
    busy_note,
    capture_centre_label,
    capture_note,
    capture_wall_label,
    centre_rows,
    centres_text,
    chain_line,
    colour_bar_title,
    continue_label,
    drive_image_note,
    field_source_line,
    files_html,
    forget_label,
    frame_hint,
    goto_row_text,
    image_sides_note,
    known_text,
    lay_out_wait,
    map_frame,
    map_title,
    needs_calibration,
    opening_step,
    orientation_label,
    plain_default,
    plan_selection_note,
    plan_title,
    plate_from_line,
    plate_rows_note,
    preset_differences,
    preset_note_text,
    register_replace_note,
    registered_line,
    result_row,
    results_left,
    run_all_wait,
    scan_lead,
    scan_progress_lines,
    scan_title,
    scan_wait,
    scanned_line,
    side_words,
    stage_label,
    step_button_label,
    step_states,
    wall_touches_note,
    wells_from_line,
    wells_known_lines,
)

__all__ = [
    "PlatePageView",
    "RunFileHandler",
    "build",
    "card_text",
    "next_step_text",
    "serve",
]

#: Names are drawn on a plate of at most this many wells; on more they are noise.
NAMES_UP_TO_WELLS = 96

#: How often the page reads the stand and the run, in milliseconds. A scan must not
#: be slowed by a status poll, and the cost of ``Microscope.state()`` on a real
#: stand is not measured (m3 §15.3): a click or a button acts at once through its
#: own handler.
TICK_MS = 1000

#: How often the page looks for a new frame, in milliseconds. It matches
#: ``LIVE_PERIOD_S``: a faster callback would find no new frame, a slower one would
#: drop frames the camera took. It reads ``PlatePage.frame()`` and, once it has drawn
#: one, ``PlatePage.is_live`` (to move the cross only while Live runs, m5 §15 D1).
FRAME_TICK_MS = 100

#: How long ``serve`` waits for Panel's server to stop, in seconds.
SERVER_STOP_WAIT_S = 10.0

DRIVE_TEXT = "DRIVE: the next click on a well moves the stage there"

#: The jog steps the Drive region offers, in µm (m5 §6.4): one jog per click, by a
#: step the operator chooses, never a number typed (a typo is how a stage leaves the
#: well). Only the steps up to the profile's jog limits are offered.
XY_STEPS_UM = (1.0, 10.0, 100.0, 1000.0)
Z_STEPS_UM = (0.5, 1.0, 5.0, 10.0, 50.0)

_XY_DEFAULT_UM = 100.0
_Z_DEFAULT_UM = 1.0

#: The pad is in the stage's axes, as ``smc stage`` and the plate map (X right, Y
#: up): a direction is the sign of each axis.
_XY_DIRECTIONS = {
    "up": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "down": (0.0, -1.0),
}
_Z_DIRECTIONS = {"up": 1.0, "down": -1.0}

#: How an empty turret position is offered: a label too (FM-125), but one a select
#: cannot show as nothing.
NO_LABEL = "(no label)"

#: The step panel's width and the two figures' height (m5 §17.3, §17.12 D15). The design says
#: 460 px so that the strip, the step bar, both figures and the drive bar are on a 1280 x 800
#: window; measured by ``/ui-check``, 460 left the whole drive bar below the fold (#443), and 400
#: left its last control, the *Drive* toggle, 20 px below the fold once the bar was two columns
#: and two rows (#475, ``wells-selected``: y 788 to 820 of 800 with the header at 143 px). 360 is
#: the floor D15 allows, and it is not lowered again: ``test_both_figures_stay_in_the_model_tree_whatever_the_screen``
#: pins ``>= 360``.
STEP_PANEL_WIDTH_PX = 400
FIGURE_HEIGHT_PX = 360

#: The height the outcome line keeps while it is empty, in pixels: one line of a text pane.
#: Measured by ``/ui-check``'s ``header`` figure with and without a text (m5 §17.12 D15).
_RUN_LINE_PX = 17

#: The drive bar's widgets sit closer than Panel's default margin (5 px above, 10 px on each side):
#: two columns of them are 440 px at most (the image column at 1280 px), and with the default
#: the *Z step* select ran past the window (``/ui-check``, #475).
_DRIVE_MARGIN = (3, 4)

#: How many centre rows the Plate screen builds (m5 §17.4.2): the suggestion is three wells,
#: and a capture past them is text, never a widget built later (D5).
_CENTRE_SLOTS = 3

#: The step panel scrolls on its own, below the header: a screen taller than the window
#: scrolls inside the panel, and the map, the image and the header stay where they are
#: (m5 §17.3). Its max-height is the window's less the header's, which is **137 px** at
#: 1280 px wide with the outcome lines empty (the strip, the connection line, the step bar
#: and its summaries), measured with ``scripts/dev/ui_snap.py``'s ``stand-620`` and
#: ``plate-620`` states on 2026-11-12 (Chromium, Panel 1.10.0): with 130 the panel ended 7 px
#: below a 620 px window. 140 leaves 3 px. A long outcome grows the header and pushes the panel
#: down: it is reported, not clipped (m5 §17.2 D12).
_STEP_PANEL_MAX_HEIGHT = "calc(100vh - 140px)"

#: The strip stays at the top of the window when the page scrolls: a white ground, so that
#: what scrolls under it is not read through it.
_STICKY_STRIP = {
    "position": "sticky",
    "top": "0",
    "z-index": "100",
    "background-color": "white",
}

#: The screen the expert view shows: today's regions, behind one switch (m5 §17.2 D4).
ScreenName = StepName | Literal["expert"]

#: The tool sections each screen lists (m5 §17.7): none. B1 listed today's sections on every
#: screen, provisionally, and B2 to B4 replaced each screen's list with its designed contents
#: (B2, #444; B3, #445; B4, #446): every screen's actions are its own buttons and rows, and the
#: tool sections stay in the expert view. Kept as a table so that a screen that lists one again
#: says it here.
_SCREEN_TOOLS: Mapping[StepName, tuple[str, ...]] = {
    "stand": (),
    "plate": (),
    "wells": (),
    "scan": (),
    "results": (),
}

#: The colour of the fields a scan has a frame for, over the plan's orange (m5 §17.10.2): a
#: green that differs from Viridis (the wells' fill), the route's red and the plan's orange.
#: A first choice: ``/ui-check``'s PNG decides whether it reads.
_IMAGED_GREEN = "#2ca02c"

#: The two fields of the tools that the Wells screen lists (m5 §17.4.3, D5): ``scan-plan``'s
#: coverage and ``scan``'s autofocus, as ``(tool, key)``. Each is the one instance the expert
#: view's section holds too.
_COVERAGE_FIELD = ("scan-plan", "coverage")
_AUTOFOCUS_FIELD = ("scan", "autofocus")

#: The width of a step's button and of the line under it in the step bar, in pixels.
_STEP_WIDTH_PX = 170

#: The two labels of the live button: ``_draw_buttons`` draws them and ``on_live`` reads
#: them, so that a click does what the button says (m5 §5.3, #287). The second is the
#: guide's, which ``busy_note`` quotes: the note and the button cannot differ (m5 §11.2).
_LIVE_LABEL = "Live"
_END_LIVE_LABEL = END_LIVE_LABEL

#: What the captures' note says while the buttons name no well (m5 §11.1).
_NO_WELL_NOTE = (
    "click a well on the map first: a capture records the well you clicked last"
)

NO_FRAME_TEXT = "no frame yet"

#: The *Preset* select's first option: no preset, which *Load* turns into a forget (m5 §10.6).
NO_PRESET = "(none)"

log = logging.getLogger("smc.ui")

#: The longest exception text a handler's failure line shows, in characters: the line
#: is for the operator, and the traceback with the whole text is in the server's log.
_FAILURE_TEXT_CHARS = 300

_WARNING = "#b45309"

#: A text pane that wraps: a path or an assumption longer than the column is read
#: whole, where a pane's own ``<pre>`` cut it at the column's edge (m5 §8.4). ``styles``
#: land on the pane's host and the ``<pre>`` inside its shadow root keeps its own
#: ``white-space`` (measured in Chromium, #252), so the same rule is given to the
#: ``<pre>`` through a stylesheet. Text, never markup (FM-139).
_WRAP_STYLES = {"white-space": "pre-wrap", "overflow-wrap": "anywhere"}
_WRAP_SHEET = "pre { white-space: pre-wrap; overflow-wrap: anywhere; }"
#: The width of the text panes of the step panel, inside its 400 px (m5 §17.3): every fixed
#: width placed in a screen is at most this, so that nothing runs past the panel's edge.
_TEXT_WIDTH = 370

_SELECTED_OUTLINE = "#1f77b4"
_PLAIN_OUTLINE = "#888888"
_UNKNOWN_FILL = "#eeeeee"
#: The sides the selects offer are sides **on the image**, as the operator names a wall
#: (m5 §13.1): the controller maps them to the stage's with the camera calibration.
_SIDES = ("centre", *IMAGE_SIDES)
_WALL_SIDES = IMAGE_SIDES


def _um(value: float, digits: int) -> str:
    """A position in µm; the demo stage's ``-0.0`` is printed as ``0.0``, as the CLI does."""
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def status_line(status: PageStatus) -> str:
    """The stand on one line: ``<stand> | XY (x, y) µm | Z z µm | objective | autofocus | light | ...``.

    The stand's name comes first, where ADR-0006 puts it. ``?`` is an unknown,
    never a zero; ``DRY RUN``, ``HALTED`` and ``moving`` are added only when they
    apply, and so is what STOP does not stop (FM-117). With the stand closed it is
    ``<stand> | not connected``, then ``DRY RUN`` when it applies, and nothing
    else: a row of ``?`` would read as a stand that does not answer.
    """
    if not status.connected:
        closed = [status.stand, "not connected"]
        if status.dry_run:
            closed.append("DRY RUN")
        return " | ".join(closed)
    parts = [
        status.stand,
        "XY ?"
        if status.xy_um is None
        else f"XY ({_um(status.xy_um[0], 1)}, {_um(status.xy_um[1], 1)}) µm",
        "Z ?" if status.z_um is None else f"Z {_um(status.z_um, 2)} µm",
        status.objective or "objective ?",
        status.autofocus,
        "light ?"
        if status.light_on is None
        else ("light on" if status.light_on else "light off"),
    ]
    if status.dry_run:
        parts.append("DRY RUN")
    if status.moving:
        parts.append(f"moving: {', '.join(status.moving)}")
    if status.halted:
        parts.append("HALTED")
    if status.stop_refused:
        parts.append(
            f"STOP is refused by {', '.join(status.stop_refused)} on this stand: "
            f"a move under way runs to its target"
        )
    return " | ".join(parts)


def _a_connection_failed(outcome: str, stand: str) -> bool:
    """Whether the activity that ended last was a connect or a disconnect that did not succeed (m5 §17.12 D15).

    The controller's outcomes begin ``connect:`` and ``disconnect:``; a success says
    ``<stand> connected`` or ``<stand> released``. The outcome stays until another activity
    starts, so the line goes with the next one.
    """
    if outcome.startswith("connect:"):
        return f"connect: {stand} connected" not in outcome
    if outcome.startswith("disconnect:"):
        return f"disconnect: {stand} released" not in outcome
    return False


def run_text(status: PageStatus) -> str:
    """The run: its progress as ``done/total message``, its outcome and what the last stop did."""
    lines: list[str] = []
    if status.running is not None:
        progress = status.progress
        if progress is None:
            lines.append(f"{status.running} is running")
        else:
            counted = (
                str(progress.done)
                if progress.total is None
                else f"{progress.done}/{progress.total}"
            )
            lines.append(f"{status.running}: {counted} {progress.message}".rstrip())
    if status.outcome is not None:
        lines.append(status.outcome)
    if status.stop_note is not None:
        lines.append(status.stop_note)
    return "\n".join(lines)


def card_text(card: ResultCard | None, *, stand: str | None = None) -> str:
    """A result card as plain text, for a ``pn.pane.Str``: a result's words are never markup (FM-139).

    The header is ``<status> <when_utc> UTC``, then `` on <profile>`` when the
    record was made on another stand than ``stand`` (when ``stand`` is not
    known, the profile is always told: a card never claims it is this stand's),
    then `` (dry run)``, then `` (stale)`` when the record read an input that has
    been produced again since; the result's lines follow, indented by two spaces,
    then its files the same way (``in the run document`` for a ``done`` card that
    names none, so that every result says where it is, m5 §8.4), then one
    ``assumed:`` line for each assumption and the record's error.
    """
    if card is None:
        return ""
    header = f"{card.status} {card.when_utc} UTC"
    if card.profile != stand:
        header += f" on {card.profile}"
    if card.dry_run:
        header += " (dry run)"
    if card.stale:
        header += " (stale)"
    lines = [header, *(f"  {line}" for line in card.lines)]
    if card.files:
        lines += [f"  {entry}" for entry in card.files]
    elif card.status == "done":
        lines.append("  in the run document")
    lines += [f"assumed: {text}" for text in card.assumed]
    if card.error is not None:
        lines.append(f"error: {card.error}")
    return "\n".join(lines)


def next_step_text(step: NextStep | None, *, windows: bool | None = None) -> str:
    """The next step as plain text, for a ``pn.pane.Str``: never markup (FM-139).

    With a tool, its name, then each line of ``why`` indented by two spaces and the
    command last. Without one (the workflow is done, or no plugin fills the slot),
    the lines of ``why`` as they are. ``""`` before the first read.

    ``command`` is quoted for a POSIX shell (``shlex.join``), and ``cmd.exe`` and
    PowerShell 5.1 read quotes differently from each other, so no one quoting is
    right for a Windows PC (FM-175): a page served there says so under the line, and
    the button, which makes the same call, is the way on. ``windows`` is ``None`` for
    "the PC this runs on" and is given by a test (m5 §8.8.6).
    """
    if step is None:
        return ""
    if step.plugin is None:
        return "\n".join(step.why)
    lines = [step.plugin, *(f"  {line}" for line in step.why)]
    lines.append(f"  command: {step.command}")
    note = posix_quoting_note(windows)
    if note:
        lines.append(f"  {note}")
    return "\n".join(lines)


#: What a page served on Windows says under a command it shows (FM-175, m5 §8.8.6).
_POSIX_QUOTING_NOTE = (
    "that line is quoted for a POSIX shell (bash, zsh); PowerShell and "
    "cmd.exe read quotes differently: press the button, which makes the "
    "same call"
)


def posix_quoting_note(windows: bool | None = None) -> str:
    """The note that a command is quoted for a POSIX shell, on a Windows page; ``""`` elsewhere.

    One text for every command the page prints for pasting: the next step's, and the
    *Command* disclosure under a Stand or Plate screen's button (m5 §17.2 D8), which
    replaced the next step there. ``windows`` is ``None`` for "the PC this runs on" and is
    given by a test.
    """
    return (
        _POSIX_QUOTING_NOTE
        if (sys.platform == "win32" if windows is None else windows)
        else ""
    )


def _wrapped_text(
    text: str = "",
    extra: dict[str, str] | None = None,
    *,
    width: int | None = _TEXT_WIDTH,
) -> Any:
    """A text pane whose lines wrap at its width, with ``extra`` styles added to the wrapping ones."""
    return pn.pane.Str(
        text,
        styles={**_WRAP_STYLES, **(extra or {})},
        width=width,
        stylesheets=[_WRAP_SHEET],
    )


def _steps(steps: tuple[float, ...], limit_um: float) -> dict[str, float]:
    """The steps up to ``limit_um``, as a select's options; the smallest one when none is.

    A profile whose limit is under every step still offers the smallest one, and
    the layer's refusal then says why (m5 §6.4).
    """
    offered = [step for step in steps if step <= limit_um] or [steps[0]]
    return {f"{step:g} µm": step for step in offered}


def _default_step(options: dict[str, float], preferred: float) -> float:
    """``preferred`` when it is offered, else the largest step offered."""
    values = list(options.values())
    return preferred if preferred in values else max(values)


def _outline(well: WellView) -> tuple[float, float, float, float]:
    """The box a well occupies on the map, ``(x0, y0, x1, y1)`` in stage µm, for the frame (m5 §17.12 D16).

    A square well is turned with the plate, so its box is the turned rectangle's; a round one
    has the same box whatever the turn.
    """
    x_um, y_um = well.center_um
    width_um, height_um = well.size_um
    if well.circular:
        half_x, half_y = width_um / 2.0, height_um / 2.0
    else:
        angle = math.radians(well.rotation_deg)
        cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
        half_x = (width_um * cos + height_um * sin) / 2.0
        half_y = (width_um * sin + height_um * cos) / 2.0
    return (x_um - half_x, y_um - half_y, x_um + half_x, y_um + half_y)


def _tip(well: WellView, label: str | None) -> str:
    """What a hover says of a well: its name, and the counts and density when there are some."""
    if well.counts is None:
        return well.well
    found = []
    for name, count in well.counts.items():
        density = None if well.per_mm2 is None else well.per_mm2.get(name)
        found.append(
            f"{name}: {count}"
            if density is None
            else f"{name}: {count} ({density:.1f}/mm²)"
        )
    return f"{well.well} — " + ", ".join(found)


@dataclass(frozen=True, slots=True)
class _CentreSlot:
    """One of the Plate screen's three centre rows: built once, rewritten by ``refresh()`` (m5 §17.2 D5).

    ``text`` is the row's sentence, ``goto_button`` and ``capture_button`` its two actions,
    ``note`` what the last drive to this well came to (``goto_row_text``) and ``capture_note``
    a note on its capture, in the warning colour: a reason common to every row is not here but
    in the one note above the rows (m5 §17.12 D19). ``box`` is the column that holds them,
    hidden when the plate has no such row.
    """

    text: Any
    goto_button: Any
    capture_button: Any
    note: Any
    capture_note: Any
    box: Any


@dataclass(frozen=True, slots=True)
class _ResultRow:
    """One of the Results screen's five rows: built once, rewritten by the redraws (m5 §17.4.5, §17.11.2).

    ``state`` is the state line, ``button`` is *Run* or *Again* (its label follows the
    tool), ``wait`` what it waits for and ``replaces`` what *Again* makes out of date, each
    hidden while empty. ``from_lines`` say where each choice's value comes from, by key;
    ``command`` and ``command_note`` are the tool's call and the Windows note under it, in
    the closed ``card``, which also holds the tool's own fields for ``RESULT_CHOICES``
    (the same instances as the accordion's, D5). ``box`` is the column the screen lists.
    """

    state: Any
    button: Any
    wait: Any
    replaces: Any
    from_lines: dict[str, Any]
    command: Any
    command_note: Any
    card: Any
    box: Any


class PlatePageView:
    """One session's widgets over one controller; every handler is a method the tests call.

    The attributes are the widgets and sources a test reads. Nothing here keeps
    an exception, and nothing runs on a thread but Bokeh's server loop.
    """

    def __init__(self, page: PlatePage) -> None:
        self.page = page
        self._view: PlateView | None = None
        self._running: str | None = None
        self._blocked: dict[str, str] = {}
        self._problems_view: tuple[str, ...] = ()
        self._problems_status: tuple[str, ...] = ()
        self._circular: bool | None = None
        self._seen_state: tuple[Any, ...] | None = None
        # The number of the frame this session last drew, and the stand's name the
        # tick last read (a result card names a profile only when it is another).
        self._seen_frame: int | None = None
        # What the frame last drawn says of its light (``frame_hint``): dark, saturated or
        # nothing, computed when the frame's number changes and drawn beside *Snap* and *Live*.
        self._frame_hint = ""
        self._camera_lacks: str | None = None
        self._stand: str | None = None
        # The cross has two writers (m5 §15 D1): the tick, from the stand's status once a
        # second, and the frame tick, from each Live frame ten times a second. The flag
        # says the frames hold it: set by the frame tick when it draws a frame's position
        # while Live runs, cleared by the tick when Live is not running. ``_cross_xy`` is
        # the position last drawn, so that a redraw of the view relabels the cross
        # without reading the stand again.
        self._cross_from_frames = False
        self._cross_xy: tuple[float, float] | None = None
        self._tools = page.tools()
        # The next step the last redraw drew: the button runs this one, never a step
        # computed at the click (m5 §8.5). And the selection the *Wells* box stands
        # for, with the guard that keeps an echo of it from selecting again.
        self._next_drawn: NextStep | None = None
        # The well the capture buttons were last drawn for: a capture records this
        # one or nothing, never the controller's ``current`` of the moment (FM-177).
        self._capture_well: str | None = None
        self._wells_shown: tuple[str, ...] = ()
        self._echoing = False
        self._next_tool: str | None = None
        # The Preset region (m5 §10.6). What a load put in every field, with the
        # loaded preset it belongs to: another tab's fields are its own, and its note
        # compares them with what the preset would have put. ``_filling`` keeps the
        # field watchers quiet while a load assigns dozens of fields, so that the note
        # is drawn once, after the last. ``_preset_armed`` holds the name an
        # overwrite waits for, in a tuple.
        self._preset_expected: dict[str, dict[str, str]] = {}
        self._expected_of: LoadedPreset | None = None
        self._filling = False
        self._preset_armed: tuple[str] | None = None
        # The Plate screen (m5 §17.4.2): the centre rows and the registration the last
        # redraw drew. A button runs what was drawn, never what would be computed at the
        # click (#287, FM-177). The drive's answer is the controller's, on the row of the well
        # it names (``PageStatus.goto_answer``), so every tab draws the same (FM-151).
        self._rows_drawn: tuple[CentreRow, ...] = ()
        self._register_drawn: tuple[tuple[str, ...], str | None] = ((), None)

        # The Wells and Scan screens (m5 §17.10): the last status the tick read, for the
        # texts ``refresh()`` redraws (it never reads the stand, #333), and how many of the
        # plan's fields the map shows as imaged *from the progress*: ``None`` while it shows
        # the run document's record. Per view (D6, FM-151): another tab draws its own.
        self._status: PageStatus | None = None
        self._imaged_drawn: int | None = None
        # The Results screen (m5 §17.11): the rows the last redraw drew and the tools it
        # found left, so that a button's wait is drawn from what the rows said and not from
        # a second reading, and how many tools of the chain the tick last saw ended. All
        # per view (D6, FM-151): another tab draws its own, from the shared status.
        self._results_drawn: dict[str, ResultRow] = {}
        self._results_left: tuple[str, ...] = ()
        self._chain_ended = 0
        # Whether the commands say they are quoted for a POSIX shell: ``None`` is the PC this
        # runs on (FM-175); a test sets it, as ``next_step_text`` takes ``windows=``.
        self._windows: bool | None = None

        self.wells_source = ColumnDataSource(
            {
                key: []
                for key in (
                    "name",
                    "x",
                    "y",
                    "r",
                    "w",
                    "h",
                    "angle",
                    "value",
                    "outline",
                    "outline_width",
                    "tip",
                )
            }
        )
        self.route_source = ColumnDataSource(
            {"x": [], "y": [], "order": [], "well": []}
        )
        # ``x``, ``y`` and ``text`` are always set together and of one length: a LabelSet
        # over columns of different lengths raises in Bokeh, and the map stops redrawing.
        self.stage_source = ColumnDataSource({"x": [], "y": [], "text": []})
        # The map's frame (m5 §17.12 D16): two points, the corners of the box the map shows.
        self.frame_source = ColumnDataSource({"x": [], "y": []})
        self._frame_drawn: tuple[tuple[float, float], tuple[float, float]] | None = None
        # The usable scan plan's fields, one rectangle each (m5 §14.2).
        self.fields_source = ColumnDataSource({"x": [], "y": [], "w": [], "h": []})
        # The plan's fields a scan has a frame for, drawn over them in green (m5 §17.10.2).
        self.imaged_source = ColumnDataSource({"x": [], "y": [], "w": [], "h": []})
        self.mapper = LinearColorMapper(
            palette=Viridis256, low=0.0, high=1.0, nan_color=_UNKNOWN_FILL
        )
        self.figure = figure(
            match_aspect=True,
            height=FIGURE_HEIGHT_PX,
            sizing_mode="stretch_width",
            x_axis_label="stage X (µm)",
            y_axis_label="stage Y (µm)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        # The frame: an invisible glyph at the two corners of the box the map shows, and the
        # only renderer the two ranges follow (D16). A view sets its two points and the map
        # reframes, keeping ``match_aspect``; changing a range's ``renderers`` after the page
        # is drawn does not reframe (measured on Panel 1.10.0 / Bokeh 3.10.2), so they are set
        # **once, here**, and what moves is the source's data. It is the map's own and in
        # no hover's list (``self.hover.renderers`` names the wells only, FM-174). No padding:
        # the frame already holds a pitch of margin on each side.
        self.frame_renderer = self.figure.scatter(
            "x", "y", source=self.frame_source, size=1, alpha=0.0
        )
        self.figure.x_range.renderers = [self.frame_renderer]
        self.figure.y_range.renderers = [self.frame_renderer]
        self.figure.x_range.range_padding = 0.0
        self.figure.y_range.range_padding = 0.0
        # Stage positions are whole micrometres up to a few hundred thousand: plain labels
        # (-60000), never ``4.000e+4``, which Bokeh picks as soon as one tick reaches 1e5
        # and which, in a 440 px map, ran the labels of the x axis into one another
        # (``/ui-check``, #443; FM-174 reads the tick labels for it).
        self.figure.xaxis.formatter = BasicTickFormatter(use_scientific=False)
        self.figure.yaxis.formatter = BasicTickFormatter(use_scientific=False)
        self.wells_renderer: Any = None
        self.hover = HoverTool(tooltips="@tip")
        self.figure.add_tools(self.hover)
        self.names = LabelSet(
            x="x",
            y="y",
            text="name",
            source=self.wells_source,
            text_align="center",
            text_baseline="middle",
            text_font_size="9px",
        )
        self.figure.add_layout(self.names)
        # The stand's sides around the wells and A1's name (m5 §16 D8): plain text, each
        # source's ``x``, ``y`` and ``text`` set together, as the cross's. A1 is drawn on
        # every plate, bold: the one name the operator checks against the plate, which
        # the names above leave out past ``NAMES_UP_TO_WELLS``.
        self.side_source = ColumnDataSource({"x": [], "y": [], "text": [], "angle": []})
        # The labels sit half a pitch outside the wells, inside the frame (which holds a pitch
        # of margin, D16), so no glyph is needed for the range to take them in: a label alone
        # beyond the top row was drawn off the canvas under the auto range (#422). The left
        # and the right one are turned 90° along the plate's sides, which is where a long word
        # fits (``operator's left`` was clipped, the owner's walk).
        self.figure.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="text",
                angle="angle",
                angle_units="deg",
                source=self.side_source,
                text_align="center",
                text_baseline="middle",
                text_font_size="11px",
                text_color="#555555",
            )
        )
        self.a1_source = ColumnDataSource({"x": [], "y": [], "text": []})
        self.figure.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="text",
                source=self.a1_source,
                text_align="center",
                text_baseline="middle",
                text_font_size="11px",
                text_font_style="bold",
            )
        )
        # Kept: its title names the class the *Density of* select names (m5 §17.4.5).
        self.colour_bar = ColorBar(color_mapper=self.mapper)
        self.figure.add_layout(self.colour_bar, "right")
        # Before the route's line, so that the route is drawn over the fields; the wells
        # are `underlay`, below both, and the only renderer the hover reads. The orange
        # differs from Viridis (the wells' fill) and from the route's red.
        self.figure.rect(
            "x",
            "y",
            "w",
            "h",
            source=self.fields_source,
            fill_color="#ff7f0e",
            fill_alpha=0.2,
            line_color="#ff7f0e",
            line_width=1,
        )
        # After the plan's fields and before the route: the green is over the orange and the
        # route over both.
        self.figure.rect(
            "x",
            "y",
            "w",
            "h",
            source=self.imaged_source,
            fill_color=_IMAGED_GREEN,
            fill_alpha=0.45,
            line_color=_IMAGED_GREEN,
            line_width=1,
        )
        self.figure.line(
            "x", "y", source=self.route_source, line_width=2, color="#d62728"
        )
        self.figure.scatter(
            "x", "y", source=self.route_source, marker="circle", size=5, color="#d62728"
        )
        self.figure.scatter(
            "x",
            "y",
            source=self.stage_source,
            marker="cross",
            size=18,
            line_width=3,
            color="#000000",
        )
        # The well the cross is over, beside it: on a white ground, so that it reads over
        # the wells' fill and the names (m5 §15 D5). The ground is whiter than it was (0.8), and
        # the label is drawn **away from the frame's nearer edge** (``_draw_cross``): beside the
        # right edge it ran off the map (``over A12`` was cut, the owner's walk, D16).
        self.cross_label = LabelSet(
            x="x",
            y="y",
            text="text",
            source=self.stage_source,
            x_offset=12,
            y_offset=12,
            text_align="left",
            text_baseline="bottom",
            text_font_size="11px",
            text_font_style="bold",
            background_fill_color="#ffffff",
            background_fill_alpha=0.8,
        )
        self.figure.add_layout(self.cross_label)
        self.figure.on_event(Tap, self._handler(self.on_tap))
        # The map's title says how far the wells can be trusted (registered, estimated,
        # unplaced: m5 §15 D4) and the plan's size, in a sentence a figure's own title
        # cuts at its width: at 440 px (m5 §17.3) it was clipped, caveat and all
        # (``/ui-check``, #443). The text is set on both from one place (``_draw_wells``);
        # the figure's own title is hidden, a pane above it wraps the same words.
        self.figure.title.visible = False
        self.map_title_text = _wrapped_text(
            width=None, extra={"font-weight": "bold", "font-size": "12px"}
        )
        self.map_title_text.sizing_mode = "stretch_width"

        # The camera's frame: 8-bit grey from 0 to 255 (``display_image``), whatever
        # the camera's bit depth is.
        self.image_source = ColumnDataSource({"image": [], "dw": [], "dh": []})
        self.image_figure = figure(
            match_aspect=True,
            height=FIGURE_HEIGHT_PX,
            sizing_mode="stretch_width",
            title=NO_FRAME_TEXT,
            x_axis_label="camera x (px)",
            y_axis_label="camera y (px, from the bottom)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        self.image_figure.image(
            image="image",
            x=0,
            y=0,
            dw="dw",
            dh="dh",
            source=self.image_source,
            color_mapper=LinearColorMapper(palette=Greys256, low=0, high=255),
        )

        # The outcome lines under the strip take no height while their text is empty
        # (m5 §17.3): ``_hide_when_empty`` is registered for each below, so the rule is
        # one, wherever the text is set.
        #
        # Each wraps at the window's width: an outcome is a sentence (``plate-orientation
        # done ...; assumed: turned 180° ...`` is 200 characters) and ran past the right
        # edge, its end unreadable (``/ui-check`` ``turned``, #443). Text, never
        # Markdown (FM-139).
        self.message = self._outcome_line({"color": _WARNING})
        self.problems_text = self._outcome_line({"color": _WARNING})
        # One line, wrapped when a refused STOP makes it long: it shares the strip's row
        # with the four buttons, which keep their width.
        self.status_text = _wrapped_text(width=None)
        self.status_text.sizing_mode = "stretch_width"
        # The one outcome line that always takes its height, empty or not: an outcome that
        # comes and goes pushed the page down and up by a line each time (the owner's walk,
        # #431 finding 11; m5 §17.12 D15). A second line (a stop note) grows it.
        self.run_text = self._outcome_line()
        self.run_text.visible = True
        self.run_text.min_height = _RUN_LINE_PX
        self.drive_text = pn.pane.Str(
            DRIVE_TEXT, styles={"color": _WARNING}, visible=False
        )
        self.checklist_text = pn.pane.Str("")
        self.captures_text = pn.pane.Str("")
        # The run document's path heads its checklist; a path is as long as the
        # folder it is in, so it wraps (m5 §8.4).
        self.run_document_text = _wrapped_text(
            f"Run document  {page.run_doc}", {"font-weight": "bold"}, width=None
        )
        # ADR-0006's head of the tools column: the tool the run document needs next.
        self.next_text = _wrapped_text()
        self.next_note = _wrapped_text(extra={"color": _WARNING})
        self.next_button = pn.widgets.Button(
            label="Run", color="primary", visible=False
        )

        self.plate_select = pn.widgets.Select(
            label="Plate",
            options=sorted(useq.registered_well_plate_keys()),
            value=page.plate,
        )
        # What the page asks while nothing named the plate (m5 §11.3): the note says
        # what the 96-well is, and the button confirms it. Both visible only then.
        #
        # In the plain colour: it asks a question the operator answers, and the Plate screen
        # shows nothing in the warning colour before a capture (m5 §17.4.2, #431's walk).
        self.plate_note = _wrapped_text()
        self.plate_confirm_button = pn.widgets.Button(label="", color="primary")
        # Which way the plate lies (m5 §16 D2): beside the plate menu, worded from the
        # stand's sides, and the only place it is said (no tool's form offers it).
        holder = page.holder
        self.orientation_select = pn.widgets.Select(
            label="Plate lies",
            options={
                orientation_label(value, holder.back, holder.right): value
                for value in cast(
                    "tuple[PlateOrientation, ...]", tuple(ORIENTATION_DEG)
                )
            },
            value=page.orientation,
        )
        # Under the menu, only on a stand that does not say its sides (m5 §16 D3).
        self.orientation_note = _wrapped_text(ORIENTATION_IN_STAGE_TERMS)
        self.orientation_note.visible = not side_words(holder.back, holder.right)
        # Set while ``refresh`` re-syncs the select, so its watcher starts no run.
        self._syncing_orientation = False
        self.label_select = pn.widgets.Select(
            label="Density of", options=[], value=None
        )
        self.drive_toggle = pn.widgets.Toggle(
            label="Drive (click moves the stage)",
            value=False,
            width=210,
            margin=_DRIVE_MARGIN,
            align="end",
        )
        # What the last drive to a well came to, beside the toggle the operator pressed
        # (m3 §5.5 D7, #431 finding 4): a refusal, what it goes to, then the run's end.
        # The controller's, so every tab draws the same (FM-151); text, never Markdown.
        self.goto_text = self._outcome_line()
        self.side_select = pn.widgets.Select(
            label="Drive to (wall as on the image)",
            options=list(_SIDES),
            value="centre",
            width=200,
            margin=_DRIVE_MARGIN,
        )
        self.select_input = pn.widgets.TextInput(
            label="Wells", placeholder="A1:D6, C*, all"
        )
        self.refresh_button = pn.widgets.Button(label="Refresh", align="end")
        self.stop_button = pn.widgets.Button(label="STOP", color="danger")
        self.resume_button = pn.widgets.Button(label="Resume", visible=False)
        self.capture_centre_button = pn.widgets.Button(label="Capture centre")
        self.capture_wall_button = pn.widgets.Button(label="Capture wall")
        self.capture_side_select = pn.widgets.Select(
            label="Wall, as on the image", options=list(_WALL_SIDES), value="left"
        )
        self.forget_button = pn.widgets.Button(label="Forget")
        # Why the three buttons above are disabled with no well: wrapped, as a tool's note.
        self.capture_note = _wrapped_text(extra={"color": _WARNING})
        # What each side of the image is on the stage, from the camera calibration (m5
        # §13.2): informative, so not in the warning colour, and it disables nothing.
        self.image_note = _wrapped_text()
        # `plate-walls`' own captures (m5 §12.5): the centres only, in its section. Widgets
        # of their own, never the registration's placed twice: a Panel widget placed
        # twice in one layout is drawn once, so the second place would be empty and
        # nothing would fail. `_draw_captures` sets both sets from the one `current`.
        self.walls_capture_button = pn.widgets.Button(label="Capture centre")
        self.walls_forget_button = pn.widgets.Button(label="Forget")
        self.walls_capture_note = _wrapped_text(extra={"color": _WARNING})
        self.walls_captures_text = pn.pane.Str("")
        self.walls_touches_note = _wrapped_text()
        self.snap_button = pn.widgets.Button(label="Snap")
        self.live_button = pn.widgets.Button(label=_LIVE_LABEL)
        self.snap_note = pn.pane.Str("", styles={"color": _WARNING})
        # ADR-0006's *Connect* (m5 §7.3): the one line that says why every control
        # that needs the stand is disabled while it is closed.
        self.connect_button = pn.widgets.Button(label="Connect")
        self.disconnect_button = pn.widgets.Button(label="Disconnect")
        self.connection_text = self._outcome_line()

        # Every handler goes through `_handler` (the document lock, a failure is a line).
        for pane in (
            self.connection_text,
            self.message,
            self.problems_text,
            self.goto_text,
        ):
            pane.param.watch(self._handler(self._hide_when_empty), "object")
        self.connect_button.on_click(self._handler(lambda event: self.on_connect()))
        self.disconnect_button.on_click(
            self._handler(lambda event: self.on_disconnect())
        )
        self.plate_select.param.watch(
            self._handler(lambda event: self.on_choose_plate(event.new)), "value"
        )
        # Registered after the choice's own watcher, so that the note reads the plate
        # the page holds by then.
        self.plate_select.param.watch(
            self._handler(lambda event: self._draw_preset_note()), "value"
        )
        self.orientation_select.param.watch(
            self._handler(lambda event: self.on_choose_orientation(event.new)), "value"
        )
        self.label_select.param.watch(
            self._handler(lambda event: self._draw_wells()), "value"
        )
        self.drive_toggle.param.watch(
            self._handler(
                lambda event: setattr(self.drive_text, "visible", bool(event.new))
            ),
            "value",
        )
        # The *Wall* select names the side on *Capture wall*: redrawn at once, not at
        # the next tick, so the label never says a side the press will not record.
        self.capture_side_select.param.watch(
            self._handler(lambda event: self._draw_captures()), "value"
        )
        self.select_input.param.watch(
            self._handler(lambda event: self.on_select_text(event.new)), "value"
        )
        self.select_input.param.watch(
            self._handler(lambda event: self._draw_preset_note()), "value"
        )
        self.plate_confirm_button.on_click(
            self._handler(lambda event: self.on_confirm_plate())
        )
        self.refresh_button.on_click(self._handler(lambda event: self.on_refresh()))
        self.next_button.on_click(self._handler(lambda event: self.on_next()))
        self.stop_button.on_click(self._handler(lambda event: self.on_stop()))
        self.resume_button.on_click(self._handler(lambda event: self.on_resume()))
        self.capture_centre_button.on_click(
            self._handler(lambda event: self.on_capture("centre"))
        )
        self.capture_wall_button.on_click(
            self._handler(lambda event: self.on_capture("wall"))
        )
        self.forget_button.on_click(self._handler(lambda event: self.on_forget()))
        self.walls_capture_button.on_click(
            self._handler(lambda event: self.on_capture("centre"))
        )
        self.walls_forget_button.on_click(self._handler(lambda event: self.on_forget()))
        self.snap_button.on_click(self._handler(lambda event: self.on_snap()))
        self.live_button.on_click(self._handler(lambda event: self.on_live()))
        self._build_drive()
        self._build_presets()
        self._build_stand_plate()
        self._build_wells_scan()
        self._build_steps()

        self.tool_runs: dict[str, Any] = {}
        self.tool_fields: dict[str, dict[str, Any]] = {}
        # Each field's default as ``describe_params`` prints it, by ``(tool, key)``.
        self._field_defaults: dict[tuple[str, str], str] = {}
        self.tool_notes: dict[str, Any] = {}
        self.tool_cards: dict[str, Any] = {}
        self.layout = self._layout()
        self.refresh()
        # The page opens on the step the run document and the connection suggest (m5
        # §17.5), once: afterwards only the operator changes the screen (D2). Before the
        # first render, so that no model of the screen it replaces was ever built.
        self.show_step(opening_step(self._step_states))
        self.tick()

    # --- layout ---------------------------------------------------------------

    def _build_drive(self) -> None:
        """The Drive region's widgets (m5 §6.4): each one calls the controller's action.

        A button's disabled state is a courtesy, not a guard: a click on a disabled
        Panel button still calls its handler on the server (measured), so the
        controller's refusals are what keeps the stand still.
        """
        page = self.page
        xy_limit_um, z_limit_um = page.jog_limits_um()
        xy_steps = _steps(XY_STEPS_UM, xy_limit_um)
        z_steps = _steps(Z_STEPS_UM, z_limit_um)
        self.xy_step_select = pn.widgets.Select(
            label="XY step",
            options=xy_steps,
            value=_default_step(xy_steps, _XY_DEFAULT_UM),
            width=110,
            margin=_DRIVE_MARGIN,
        )
        self.z_step_select = pn.widgets.Select(
            label="Z step",
            options=z_steps,
            value=_default_step(z_steps, _Z_DEFAULT_UM),
            width=90,
            margin=_DRIVE_MARGIN,
        )
        self.jog_buttons = {
            direction: pn.widgets.Button(label=label, width=50, margin=_DRIVE_MARGIN)
            for direction, label in (
                ("up", "↑"),
                ("left", "←"),
                ("right", "→"),
                ("down", "↓"),
            )
        }
        self.z_buttons = {
            "up": pn.widgets.Button(label="Z +", width=50, margin=_DRIVE_MARGIN),
            # A minus sign, as wide as the plus: a hyphen reads as a dash.
            "down": pn.widgets.Button(
                label="Z \N{MINUS SIGN}", width=50, margin=_DRIVE_MARGIN
            ),
        }
        self.autofocus_on_button = pn.widgets.Button(
            label="Autofocus on", margin=_DRIVE_MARGIN
        )
        self.autofocus_off_button = pn.widgets.Button(
            label="Autofocus off", margin=_DRIVE_MARGIN
        )
        # Built once per session, while the stand may be closed: the tick reads the
        # labels again when the connection it sees changes (m5 §7.3).
        self._seen_connected = page.connected
        self._objective_labels = page.objective_labels()
        self.objective_select = pn.widgets.Select(
            label="Objective",
            options={label or NO_LABEL: label for label in self._objective_labels},
            width=220,
            margin=_DRIVE_MARGIN,
        )
        self.objective_button = pn.widgets.Button(
            label="Change objective", margin=_DRIVE_MARGIN, align="end"
        )
        self.objective_confirm_text = pn.pane.Str(
            "", styles={"color": _WARNING, "white-space": "pre-wrap"}, width=380
        )
        self.objective_confirm_button = pn.widgets.Button(
            label="Turn the turret", color="danger"
        )
        self.objective_cancel_button = pn.widgets.Button(label="Cancel")
        self.objective_confirm = pn.Row(
            pn.Column(
                self.objective_confirm_text,
                pn.Row(self.objective_confirm_button, self.objective_cancel_button),
            ),
            visible=False,
        )
        self.light_on_button = pn.widgets.Button(label="Light on", margin=_DRIVE_MARGIN)
        self.light_off_button = pn.widgets.Button(
            label="Light off", margin=_DRIVE_MARGIN
        )
        self.drive_note = pn.pane.Str(
            "", styles={"color": _WARNING, "white-space": "pre-wrap"}, width=380
        )
        # What an arrow does to the image on this stand (m5 §13.2): the arrows are in
        # the stage's axes and stay so, the note says what they do to what is seen.
        self.drive_image_note = _wrapped_text(width=380)
        # The label the confirmation was armed with, in a tuple: an empty label is
        # one too (FM-125). The select follows the turret until the tick has read the
        # objective in place and set it, once; never after.
        self._armed: tuple[str] | None = None
        self._objective_shown = False

        for direction, button in self.jog_buttons.items():
            button.on_click(self._handler(self._jogger(direction)))
        for direction, button in self.z_buttons.items():
            button.on_click(self._handler(self._focuser(direction)))
        self.autofocus_on_button.on_click(
            self._handler(lambda event: self.on_autofocus(True))
        )
        self.autofocus_off_button.on_click(
            self._handler(lambda event: self.on_autofocus(False))
        )
        self.objective_button.on_click(self._handler(lambda event: self.on_objective()))
        self.objective_confirm_button.on_click(
            self._handler(lambda event: self.on_objective_confirm())
        )
        self.objective_cancel_button.on_click(
            self._handler(lambda event: self.on_objective_cancel())
        )
        self.objective_select.param.watch(
            self._handler(lambda event: self._disarm()), "value"
        )
        self.light_on_button.on_click(self._handler(lambda event: self.on_light(True)))
        self.light_off_button.on_click(
            self._handler(lambda event: self.on_light(False))
        )

    def _jogger(self, direction: str) -> Callable[[Any], None]:
        def clicked(event: Any) -> None:
            self.on_jog(direction)

        return clicked

    def _focuser(self, direction: str) -> Callable[[Any], None]:
        def clicked(event: Any) -> None:
            self.on_z(direction)

        return clicked

    def _build_presets(self) -> None:
        """The *Preset* region's widgets (m5 §10.6): none reaches the stand, so none is ever disabled.

        Every button goes through ``_handler`` (FM-171) and carries its text as
        ``label=`` (FM-147). The select's options are read from the folders by
        ``refresh()`` and after a save, never by the tick (FM-151).
        """
        self.preset_select = pn.widgets.Select(
            label="Preset", options={NO_PRESET: ""}, value="", width=280
        )
        self.preset_load_button = pn.widgets.Button(label="Load", width=80)
        self.preset_name = pn.widgets.TextInput(
            label="Save as", placeholder="weekly", width=280
        )
        self.preset_save_button = pn.widgets.Button(label="Save", width=80)
        # Wrapped anywhere: the question names the file that exists, a path with no
        # space in it, which `pre-wrap` alone cut at the column's edge (/ui-check, FM-153).
        self.preset_confirm_text = _wrapped_text(extra={"color": _WARNING})
        self.preset_overwrite_button = pn.widgets.Button(
            label="Overwrite", color="danger"
        )
        self.preset_cancel_button = pn.widgets.Button(label="Cancel")
        self.preset_confirm = pn.Row(
            pn.Column(
                self.preset_confirm_text,
                pn.Row(self.preset_overwrite_button, self.preset_cancel_button),
            ),
            visible=False,
        )
        # Text, never Markdown (FM-139): a preset's name and path come from a file.
        self.preset_note = _wrapped_text("no preset loaded")
        self.preset_load_button.on_click(
            self._handler(lambda event: self.on_load_preset())
        )
        self.preset_save_button.on_click(
            self._handler(lambda event: self.on_save_preset())
        )
        self.preset_overwrite_button.on_click(
            self._handler(lambda event: self.on_preset_overwrite())
        )
        self.preset_cancel_button.on_click(
            self._handler(lambda event: self.on_preset_cancel())
        )
        # A new name asks nothing about the old one: the confirmation goes.
        self.preset_name.param.watch(
            self._handler(lambda event: self._disarm_preset()), "value"
        )

    def _preset_region(self) -> Any:
        """The head of the tools column: the preset the assay's choices come from, loaded and saved by name.

        Above the next step, because a preset is what the operator chooses before
        the first step.
        """
        return pn.Column(
            pn.pane.Str("Preset", styles={"font-weight": "bold"}),
            pn.Row(self.preset_select, self.preset_load_button),
            pn.Row(self.preset_name, self.preset_save_button),
            self.preset_confirm,
            self.preset_note,
        )

    def _build_stand_plate(self) -> None:
        """The widgets of the Stand and Plate screens, each action a button of its own (m5 §17.4.1, §17.4.2).

        A button holds no value, so these are second instances of what the strip and the
        tools' sections already offer, going through the same handlers and the same
        controller calls (D5); every one is registered through ``_handler`` (FM-171) and
        carries its text as ``label=`` (FM-147). A disabled button is a courtesy: the
        controller refuses each of these calls on its own (FM-169). Text is plain, never
        markup (FM-139): a well's name, a preset's name and a path come from outside.
        """
        stand = self.page.stand_name
        # The Stand screen: the stand line with a *Connect* of its own, what is known
        # about the stand, and the calibration offered only when it is missing.
        self.stand_text = _wrapped_text()
        self.stand_connect_button = pn.widgets.Button(label=f"Connect {stand}")
        self.known_block = _wrapped_text()
        self.calibrate_button = pn.widgets.Button(label="Calibrate the camera")
        self.calibrate_note = _wrapped_text()
        self.calibrate_command = _wrapped_text()
        # Under each command, on a Windows page only: the line is quoted for a POSIX shell
        # (FM-175). Hidden, and taking no height, anywhere else.
        self.calibrate_command_note = _wrapped_text()
        self.calibrate_command_note.visible = False
        self.calibrate_box = pn.Column(
            self.calibrate_button,
            self.calibrate_note,
            pn.Card(
                self.calibrate_command,
                self.calibrate_command_note,
                title="Command",
                collapsed=True,
                sizing_mode="stretch_width",
            ),
        )
        # The Plate screen: where the plate comes from, three centre rows built once, the
        # captures past three as text, one *Register the plate* and what it came to.
        self.plate_from_text = _wrapped_text()
        # What holds for every row, said once above them (m5 §17.12 D19): a drive that goes, Live,
        # and what *Register the plate* uses while nothing is captured.
        self.rows_note = _wrapped_text()
        self.rows_note.visible = False
        self.centre_slots = [self._centre_slot(i) for i in range(_CENTRE_SLOTS)]
        self.other_captures = _wrapped_text()
        self.other_captures.visible = False
        self.register_button = pn.widgets.Button(
            label="Register the plate", color="primary"
        )
        self.register_wait = _wrapped_text()
        # What registering again would replace, said before the click (m5 §11.4, FM-172): the
        # tool's own section, which carried that note, is in the expert view only.
        self.register_replaces = _wrapped_text()
        self.register_replaces.visible = False
        self.register_command = _wrapped_text()
        self.register_command_note = _wrapped_text()
        self.register_command_note.visible = False
        self.register_card = pn.Card(
            self.register_command,
            self.register_command_note,
            title="Command",
            collapsed=True,
            sizing_mode="stretch_width",
            visible=False,
        )
        self.registered_block = _wrapped_text()
        self.registered_block.visible = False
        self.other_ways_text = _wrapped_text(OTHER_WAYS_TO_REGISTER)
        self.stand_connect_button.on_click(
            self._handler(lambda event: self.on_connect())
        )
        # The same handler as the *Run camera-calibration* of the expert view's section.
        self.calibrate_button.on_click(
            self._handler(self._runner("camera-calibration"))
        )
        self.register_button.on_click(self._handler(lambda event: self.on_register()))

    def _build_wells_scan(self) -> None:
        """The widgets of the Wells and Scan screens, each action a button of its own (m5 §17.4.3, §17.4.4, §17.10, #445).

        As the Stand and Plate screens': a button holds no value, so each is a second
        instance of what the tool's section offers, going through the same handler and the
        same controller call (D5), registered through ``_handler`` (FM-171) and carrying its
        text as ``label=`` (FM-147). A disabled button is a courtesy: the controller refuses
        each of these calls on its own (FM-169). The two tool fields the Wells screen lists
        are the tools' own instances, built later with their sections (D5); the lines under
        them say where each value comes from (D7). Text is plain, never markup (FM-139): a
        well's name, a preset's name and a path come from outside. Every text pane that
        draws nothing is hidden and takes no height (``_say``, D12).
        """
        # The Wells screen.
        self.wells_from_text = _wrapped_text()
        self.coverage_from_text = _wrapped_text()
        self.autofocus_from_text = _wrapped_text()
        # What each of the two fields is, in plain words, above it (m5 §17.12 D20): the fields
        # are the tools' own and keep their names in the expert view.
        self.coverage_words = _wrapped_text(COVERAGE_WORDS)
        self.autofocus_words = _wrapped_text(AUTOFOCUS_WORDS)
        self.layout_button = pn.widgets.Button(
            label="Lay out the fields", color="primary"
        )
        self.layout_wait = _wrapped_text()
        self.layout_wait.visible = False
        self.wells_known_block = _wrapped_text()
        self.wells_note = _wrapped_text()
        self.wells_note.visible = False
        self.layout_command = _wrapped_text()
        self.layout_command_note = _wrapped_text()
        self.layout_command_note.visible = False
        self.layout_card = pn.Card(
            self.layout_command,
            self.layout_command_note,
            title="Command",
            collapsed=True,
            sizing_mode="stretch_width",
            visible=False,
        )
        # The Scan screen: its title and lead name the plan's wells and fields, so they are
        # attributes now (``_draw_scan_screen``); the bar and the lines under it follow the
        # scan's progress and stay after it, and the result line follows the run document.
        self.scan_title_text = _wrapped_text(
            SCREEN_TITLES["scan"], {"font-weight": "bold"}
        )
        self.scan_lead_text = _wrapped_text(SCREEN_LEADS["scan"])
        self.scan_button = pn.widgets.Button(label="Start the scan", color="primary")
        self.scan_wait_text = _wrapped_text()
        self.scan_wait_text.visible = False
        self.scan_dry_note = _wrapped_text()
        self.scan_dry_note.visible = False
        self.scan_progress = pn.indicators.Progress(
            value=0, max=1, width=_TEXT_WIDTH, bar_color="primary", visible=False
        )
        self.scan_progress_text = _wrapped_text()
        self.scan_progress_text.visible = False
        self.scanned_block = _wrapped_text()
        self.scanned_block.visible = False
        self.scan_command = _wrapped_text()
        self.scan_command_note = _wrapped_text()
        self.scan_command_note.visible = False
        self.scan_card = pn.Card(
            self.scan_command,
            self.scan_command_note,
            title="Command",
            collapsed=True,
            sizing_mode="stretch_width",
            visible=False,
        )
        # The same handlers as the *Run scan-plan* and *Run scan* of the expert view's
        # sections: the pairs are the page's and the typed fields', read at the click.
        self.layout_button.on_click(self._handler(self._runner("scan-plan")))
        self.scan_button.on_click(self._handler(self._runner("scan")))

    def _build_results(self) -> None:
        """The Results screen's widgets: *Run all that are left*, a row for each tool and where the files are (m5 §17.4.5, §17.11.2, #446).

        Built with the screens, after the tools' sections, because a row's closed *Choices
        and command* holds the tool's own fields (D5: one instance, the accordion's too) and
        those are made with the sections. Every button goes through ``_handler`` (FM-171) and
        carries its text as ``label=`` (FM-147); a row's *Run* is the tool's own handler
        (``_runner``), so the pairs it sends are the page's and the typed fields', read at the
        click. A disabled button is a courtesy: the controller refuses each call on its own
        (FM-169). A tool that did not load has a row with no field and no card, which says so.
        Text is plain, never markup (FM-139): a class, a path and an outcome come from outside.
        A pane that says nothing is hidden and takes no height (``_say``, D12).
        """
        self.run_all_button = pn.widgets.Button(label=RUN_ALL_LABEL, color="primary")
        self.run_all_wait_text = _wrapped_text()
        self.run_all_wait_text.visible = False
        self.chain_text = _wrapped_text()
        self.chain_text.visible = False
        self.run_all_button.on_click(self._handler(lambda event: self.on_run_all()))
        self.result_rows: dict[str, _ResultRow] = {}
        for tool in RESULT_TOOLS:
            fields = self.tool_fields.get(tool, {})
            from_lines: dict[str, Any] = {}
            inside: list[Any] = []
            for key in RESULT_CHOICES.get(tool, ()):
                field = fields.get(key)
                if field is not None:
                    from_lines[key] = _wrapped_text()
                    inside += [field, from_lines[key]]
            command = _wrapped_text()
            command_note = _wrapped_text()
            command_note.visible = False
            card = pn.Card(
                *inside,
                command,
                command_note,
                title="Choices and command",
                collapsed=True,
                sizing_mode="stretch_width",
                visible=False,
            )
            state = _wrapped_text()
            button = pn.widgets.Button(label="Run")
            wait = _wrapped_text()
            wait.visible = False
            replaces = _wrapped_text()
            replaces.visible = False
            button.on_click(self._handler(self._runner(tool)))
            box = pn.Column(
                _wrapped_text(RESULT_NAMES[tool], {"font-weight": "bold"}),
                state,
                button,
                wait,
                replaces,
                card,
            )
            self.result_rows[tool] = _ResultRow(
                state,
                button,
                wait,
                replaces,
                from_lines,
                command,
                command_note,
                card,
                box,
            )
        self.files_heading = _wrapped_text(FILES_HEADING, {"font-weight": "bold"})
        # The one HTML pane of the page (m5 §17.12 D21): links to the files the run document
        # names, through the server's ``/run-files/`` route. ``files_html`` escapes every text
        # in it, and no other text of the page becomes markup (FM-139).
        self.files_text = pn.pane.HTML(
            f"<div>{NO_FILES_YET}</div>", width=_TEXT_WIDTH, margin=(5, 10)
        )

    def _centre_slot(self, index: int) -> _CentreSlot:
        """One centre row's widgets: its sentence, *Go to*, *Capture the centre of* (or *Redo*) and its note (m5 §17.4.2)."""
        text = _wrapped_text()
        goto_button = pn.widgets.Button(label="Go to")
        capture_button = pn.widgets.Button(label="Capture the centre")
        note = _wrapped_text()
        note.visible = False
        far = _wrapped_text(extra={"color": _WARNING})
        far.visible = False
        goto_button.on_click(self._handler(lambda event: self.on_goto_row(index)))
        capture_button.on_click(self._handler(lambda event: self.on_capture_row(index)))
        box = pn.Column(
            text, pn.Row(goto_button, capture_button), note, far, visible=False
        )
        return _CentreSlot(text, goto_button, capture_button, note, far, box)

    def _drive_bar(self) -> Any:
        """ADR-0006's *Drive*, under the image where Live is watched (m5 §17.3): a bar that wraps.

        The pad, the focus, the autofocus, the objective and the light of today's Drive
        region, plus the map's *Drive to* select and its *Drive (next click moves the
        stage)* toggle, which leave the map's controls: they are what the operator
        reaches for with the image in view, and in the old row the toggle was drawn over
        the preset's *Save as* (``/ui-check``, #431). A flex box wraps its groups at the
        image column's width, so no group runs past the window at 1280 px.
        """
        jog = self.jog_buttons
        pad = pn.Column(
            pn.Row(pn.Spacer(width=60), jog["up"]),
            pn.Row(jog["left"], jog["down"], jog["right"]),
        )
        # Two columns and two rows (m5 §17.12 D15): the pad and its step on the left; the
        # focus, the autofocus and the light on the right; then the objective on one row
        # and the drive on another, with the notes under them. The old flex box wrapped
        # nine groups into five lines and needed a scroll in every state (#452). Each
        # row is narrower than the image column (440 px at 1280): a group wider than the
        # column is not wrapped, it runs past the window (``/ui-check``, #443).
        pad_column = pn.Column(pad, self.xy_step_select)
        focus_column = pn.Column(
            pn.Row(self.z_buttons["up"], self.z_buttons["down"], self.z_step_select),
            pn.Row(self.autofocus_on_button, self.autofocus_off_button),
            pn.Row(self.light_on_button, self.light_off_button),
        )
        return pn.Column(
            pn.Row(pad_column, focus_column),
            pn.Row(self.objective_select, self.objective_button),
            self.objective_confirm,
            pn.Row(self.side_select, self.drive_toggle),
            self.goto_text,
            self.drive_image_note,
            self.drive_note,
            sizing_mode="stretch_width",
        )

    def _next_region(self) -> Any:
        """The head of the tools column: the tool the run document needs next, and one button for it (m5 §8.5)."""
        return pn.Column(
            pn.pane.Str("Next step", styles={"font-weight": "bold"}),
            self.next_text,
            self.next_button,
            self.next_note,
        )

    def _build_steps(self) -> None:
        """The step bar's widgets and the view's own state: the step shown and the expert switch (m5 §17.6).

        Both belong to the browser tab, not to the controller (D6): two tabs may show two
        screens, while what each draws is still the controller's (FM-151). A step's
        button is not disabled whatever the run document says: every step opens at any
        time (D2), and an action that cannot run says why beside itself.
        """
        self.step: StepName = "stand"
        self.expert = False
        self._step_states = step_states(
            (),
            connected=self.page.connected,
            stand=self.page.stand_name,
            preset=None,
            plate=self.page.plate,
            plan=None,
        )
        # Set while `show_*` re-syncs the switch, so that its watcher does not show again.
        self._syncing_expert = False
        self.step_buttons = {
            step: pn.widgets.Button(
                label=step_button_label(index, state),
                width=_STEP_WIDTH_PX,
                css_classes=["smc-step"],
                color="light",
            )
            for index, (step, state) in enumerate(
                zip(STEPS, self._step_states, strict=True), start=1
            )
        }
        # Text, never Markdown (FM-139): a preset's name is a file's.
        self.step_summaries = {
            step: _wrapped_text(width=_STEP_WIDTH_PX, extra={"font-size": "12px"})
            for step in STEPS
        }
        self.expert_toggle = pn.widgets.Toggle(
            label="Expert view", value=False, color="light", align="start"
        )
        self.continue_button = pn.widgets.Button(label="Continue", color="primary")
        # The panel that scrolls on its own (§17.3): the strip, the step bar and the
        # figures stay on screen while a long screen is read.
        self.step_panel = pn.Column(
            width=STEP_PANEL_WIDTH_PX,
            scroll=True,
            styles={"max-height": _STEP_PANEL_MAX_HEIGHT},
        )
        for step, button in self.step_buttons.items():
            button.on_click(self._handler(self._stepper(step)))
        self.expert_toggle.param.watch(
            self._handler(lambda event: self._on_expert_toggle(bool(event.new))),
            "value",
        )
        self.continue_button.on_click(self._handler(lambda event: self.on_continue()))

    def _stepper(self, step: StepName) -> Callable[[Any], None]:
        def clicked(event: Any) -> None:
            self.show_step(step)

        return clicked

    def _layout(self) -> Any:
        strip = pn.Row(
            self.connect_button,
            self.disconnect_button,
            self.status_text,
            self.resume_button,
            self.stop_button,
            sizing_mode="stretch_width",
        )
        step_bar = pn.Row(
            *(
                pn.Column(self.step_buttons[step], self.step_summaries[step])
                for step in STEPS
            ),
            pn.Spacer(sizing_mode="stretch_width"),
            self.expert_toggle,
            sizing_mode="stretch_width",
        )

        sections = {tool.name: self._tool_section(tool) for tool in self._tools}
        stages: dict[str, list[Any]] = {}
        for tool in self._tools:
            stages.setdefault(tool.stage, []).append(sections[tool.name])
        # Where each tool's stage is in the accordion, so that the next step can open it.
        self._stage_index = {
            tool.name: list(stages).index(tool.stage) for tool in self._tools
        }
        self.accordion = pn.Accordion(
            *((stage, pn.Column(*sections_)) for stage, sections_ in stages.items()),
            active=list(range(len(stages))),
        )
        self._build_screens(sections)

        # Neither figure is ever in a swapped list, and there are no tabs: a tab that is
        # not dynamic keeps a figure's view, but a figure that leaves the model tree has
        # its view destroyed, and bokehjs's DataRange1d keeps the bounds of every view it
        # has had, so the map was framed on the union of the plates it had drawn and
        # neither Reset nor Refresh took the old ones out (#297, FM-174). Both are
        # always on screen, side by side (m5 §17.2 D1).
        map_column = pn.Column(
            pn.Row(self.label_select, self.refresh_button),
            self.map_title_text,
            pn.pane.Bokeh(self.figure, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        image_column = pn.Column(
            pn.Row(self.snap_button, self.live_button, self.snap_note),
            pn.pane.Bokeh(self.image_figure, sizing_mode="stretch_width"),
            self._drive_bar(),
            sizing_mode="stretch_width",
        )
        centre = pn.Row(map_column, image_column, sizing_mode="stretch_width")
        # One sticky header (m5 §17.2 D12): the strip, the outcome lines and the step bar
        # with its summaries. At 1280 x 800 the drive bar is partly below the fold
        # (``/ui-check``), and a page scrolled to reach it must not take STOP out of view
        # (ADR-0006: STOP is always offered), nor the step bar, which scrolled away on the
        # owner's walk while the strip held (#431): they stay on screen together. An
        # outcome line takes no height while it is empty.
        self.header = pn.Column(
            strip,
            self.connection_text,
            self.drive_text,
            self.run_text,
            self.message,
            self.problems_text,
            step_bar,
            sizing_mode="stretch_width",
            styles=_STICKY_STRIP,
        )
        return pn.Column(
            self.header,
            pn.Row(self.step_panel, centre, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def _build_screens(self, sections: dict[str, Any]) -> None:
        """The lists the step panel shows, built once (m5 §17.2 D5, §17.6): never rewritten.

        Each screen is its title, its lead, its own regions and *Continue*. A value widget
        is one instance listed on every screen that shows it: only one list is rendered,
        and a swap keeps what was typed (measured, §17.2 D5). The Stand, Plate, Wells and
        Scan screens are designed (B2, #444, §17.4.1 and §17.4.2; B3, #445, §17.4.3 and
        §17.4.4): each holds its own buttons, none of the next step (the walk found *Run
        scan-plan* on the Plate screen) and none of the tool sections, which stay in the
        expert view; the Wells screen lists two of the tools' own fields, the same
        instances the accordion holds. So is Results (B4, #446, §17.4.5): *Run all that are
        left*, a row for each of its five tools (whose closed *Choices and command* holds
        the same instances too) and where the files are; the next step, the sections and the
        checklist are in the expert view only.
        """
        self._build_results()
        preset_region = self._preset_region()
        next_region = self._next_region()
        # The plate, where it comes from, which way it lies and the confirmation of the
        # placeholder: one container on the Plate screen and in the expert view.
        plate_controls = pn.Column(
            self.plate_select,
            self.plate_from_text,
            self.orientation_select,
            pn.Column(
                self.plate_confirm_button, self.plate_note, self.orientation_note
            ),
        )
        # The order is D12's: the stand line and the preset with its *Load* come first,
        # then what is known, then the calibration, so that the title, the lead and the
        # first action are on a 1280 x 620 page under the header.
        stand_region = pn.Column(self.stand_text, self.stand_connect_button)
        own: dict[StepName, list[Any]] = {
            "stand": [
                stand_region,
                preset_region,
                self.known_block,
                self.calibrate_box,
            ],
            "plate": [
                plate_controls,
                self.rows_note,
                *(slot.box for slot in self.centre_slots),
                self.other_captures,
                self.register_button,
                self.register_wait,
                self.register_card,
                self.registered_block,
                # Under the result line it follows (the owner's walk, finding 10).
                self.register_replaces,
                self.other_ways_text,
            ],
            "wells": self._wells_region(),
            "scan": [
                self.scan_button,
                self.scan_wait_text,
                self.scan_dry_note,
                self.scan_progress,
                self.scan_progress_text,
                self.scanned_block,
                self.scan_card,
            ],
            # The action first, so that it is on the page without a scroll (§17.4.5); the
            # run document's path closes the list, before *Continue* (which is hidden here).
            "results": [
                self.run_all_button,
                self.run_all_wait_text,
                self.chain_text,
                *(row.box for row in self.result_rows.values()),
                self.files_heading,
                self.files_text,
                self.run_document_text,
            ],
        }
        # The Scan screen's title and lead name the plan's wells and fields: panes the page
        # rewrites (``_draw_scan_screen``), not the static text of the others.
        titles = {
            step: _wrapped_text(SCREEN_TITLES[step], {"font-weight": "bold"})
            for step in STEPS
        }
        leads = {step: _wrapped_text(SCREEN_LEADS[step]) for step in STEPS}
        titles["scan"], leads["scan"] = self.scan_title_text, self.scan_lead_text
        self.screens: dict[ScreenName, list[Any]] = {}
        for step in STEPS:
            self.screens[step] = [
                titles[step],
                leads[step],
                *own[step],
                *(sections[name] for name in _SCREEN_TOOLS[step] if name in sections),
                self.continue_button,
            ]
        # The expert view keeps the next step, the checklist and every tool section: what
        # the Results screen's rows replace (§17.4.5).
        self.screens["expert"] = [
            preset_region,
            next_region,
            plate_controls,
            self.select_input,
            self.accordion,
            self.run_document_text,
            self.checklist_text,
        ]

    def _wells_region(self) -> list[Any]:
        """The Wells screen's own widgets, in the order of m5 §17.4.3 (the action is above what is known).

        The *Wells* box, then the two tool fields with the line that says where each value
        comes from, the action, its wait, what is known about the plan, the note that the
        selection is not the plan's, and the *Command* disclosure. The two fields are the
        tools' own instances (D5, ``tool_fields``), the ones the accordion holds too: a tool
        that did not load has no field, and its line is left out with it.
        """
        listed: list[Any] = [self.select_input, self.wells_from_text]
        for (tool, key), words, line in (
            (_COVERAGE_FIELD, self.coverage_words, self.coverage_from_text),
            (_AUTOFOCUS_FIELD, self.autofocus_words, self.autofocus_from_text),
        ):
            field = self.tool_fields.get(tool, {}).get(key)
            if field is not None:
                listed += [words, field, line]
        return [
            *listed,
            self.layout_button,
            self.layout_wait,
            self.wells_known_block,
            self.wells_note,
            self.layout_card,
        ]

    def show_step(self, step: StepName) -> None:
        """Show ``step``'s screen; the expert view, if it is on, goes off first (m5 §17.2 D4)."""
        self.step = step
        self.expert = False
        self._sync_expert_toggle()
        self._show()

    def show_expert(self, on: bool) -> None:
        """Show the expert view, or the screen of the step the view keeps (m5 §17.2 D4)."""
        self.expert = on
        self._sync_expert_toggle()
        self._show()

    def _on_expert_toggle(self, on: bool) -> None:
        """The switch was pressed (or set by ``show_*``, which says so by ``_syncing_expert``)."""
        if not self._syncing_expert:
            self.show_expert(on)

    def _sync_expert_toggle(self) -> None:
        if self.expert_toggle.value == self.expert:
            return
        self._syncing_expert = True
        try:
            self.expert_toggle.value = self.expert
        finally:
            self._syncing_expert = False

    def on_continue(self) -> None:
        """*Continue to …*: the next step's screen; nothing after the last (a click on a hidden button still arrives)."""
        position = STEPS.index(self.step) + 1
        if position < len(STEPS):
            self.show_step(STEPS[position])

    def _show(self) -> None:
        """Put the screen in the step panel, as a copy of its list, and restyle the step bar.

        A copy: ``Column.objects`` takes the list it is given, and the screen's own must
        never be the one a later swap mutates (m5 §17.6).
        """
        screen: ScreenName = "expert" if self.expert else self.step
        label = continue_label(self.step)
        self.continue_button.label = label or ""
        self.continue_button.visible = label is not None
        self.step_panel.objects = list(self.screens[screen])
        # After the swap: a panel keeps the scroll of the screen it replaced, and the walk
        # opened Plate scrolled as far as Stand had been, its title out of view (m5 §17.3,
        # D12). Measured on Panel 1.10.0: a column scrolled in the browser reports its
        # position to the server, and setting 0 after its objects changed scrolls it back.
        self.step_panel.scroll_position = 0
        self._style_steps()
        # The map frames what the screen is about: the plate, or the plan's wells (D16).
        self._draw_map_frame()
        self._draw_cross(self._cross_xy)

    def _style_steps(self) -> None:
        """The step shown is the primary button, and the class ``/ui-check`` reads it by; none is while the expert view is on."""
        for step, button in self.step_buttons.items():
            shown = step == self.step and not self.expert
            button.color = "primary" if shown else "light"
            button.css_classes = (
                ["smc-step", "smc-step-shown"] if shown else ["smc-step"]
            )

    def _draw_steps(self) -> None:
        """The step bar's labels and summaries, from the run document and the connection (m5 §17.5).

        Read from ``guide.step_states`` and never from the step panel's list: a screen is
        not rewritten by a redraw, and the tick draws none of this (the stand's own
        status is not read to name it). A summary is also the button's tooltip.
        """
        view = self._view
        preset = self.page.preset
        self._step_states = step_states(
            () if view is None else view.checklist,
            connected=self.page.connected,
            stand=self.page.stand_name,
            preset=None if preset is None else preset.file.name,
            plate=self.page.plate,
            plan=None if view is None else view.plan,
        )
        for index, state in enumerate(self._step_states, start=1):
            button = self.step_buttons[state.step]
            button.label = step_button_label(index, state)
            button.description = state.summary or None
            self.step_summaries[state.step].object = state.summary
        self._style_steps()

    @staticmethod
    def _outcome_line(extra: dict[str, str] | None = None) -> Any:
        """A line under the strip: empty and hidden, wrapped at the window's width (m5 §17.3)."""
        line = _wrapped_text(extra=extra, width=None)
        line.sizing_mode = "stretch_width"
        line.visible = False
        return line

    def _hide_when_empty(self, event: Any) -> None:
        """An outcome line takes no height while its text is empty (m5 §17.3)."""
        event.obj.visible = bool(event.new)

    def _tool_section(self, tool: ToolView) -> Any:
        fields: dict[str, Any] = {}
        suggested = dict(tool.suggested)
        for name, type_name, default, notes in (*tool.key, *tool.advanced):
            description = f"{type_name}  {notes}".strip()
            if name in suggested:
                # The field starts with what the stand suggests, so that this tool's own
                # *Run* sends what the next step shows (m5 §8.3).
                description += f"\nsuggested by the profile: {tool.suggested_why}"
            # The placeholder is what an empty field means, not a value to type: a string
            # default comes without the quotes ``describe_params`` prints (finding 13 of the
            # walk), and the raw default is kept for the "from" lines (m5 §17.10.3).
            self._field_defaults[(tool.name, name)] = default
            fields[name] = pn.widgets.TextInput(
                label=name,
                placeholder=plain_default(default),
                description=description,
                value=suggested.get(name, ""),
            )
            # After the field has its first value: a change of any field redraws the
            # note, unless a load is assigning them all (m5 §10.6).
            fields[name].param.watch(self._handler(self._on_field_changed), "value")
        self.tool_fields[tool.name] = fields
        key_widgets = [fields[row[0]] for row in tool.key]
        advanced_widgets = [fields[row[0]] for row in tool.advanced]
        run = pn.widgets.Button(label=f"Run {tool.name}")
        run.on_click(self._handler(self._runner(tool.name)))
        self.tool_runs[tool.name] = run
        # Wrapped: the reason of a stage with a checkpoint names the command that
        # continues it, which a line cut at the column's edge would lose (#252). A
        # `white-space` in `styles` lands on the pane's host, and the `<pre>` inside
        # its shadow root keeps its own `pre` (measured in Chromium), so the rule
        # goes to the `<pre>` itself.
        note = pn.pane.Str(
            "",
            styles={"color": _WARNING},
            width=_TEXT_WIDTH,
            stylesheets=["pre { white-space: pre-wrap; }"],
        )
        self.tool_notes[tool.name] = note
        # Plain text, never Markdown: a result's words are not markup (FM-139). It wraps:
        # a file's path is as long as the folder it is in (m5 §8.4).
        card = _wrapped_text()
        self.tool_cards[tool.name] = card
        # Wrapped: a summary is a sentence, and the step panel is 400 px wide (m5 §17.3).
        parts: list[Any] = [_wrapped_text(tool.summary), *key_widgets]
        if tool.name == "plate-registration":
            parts += [
                pn.Row(self.capture_centre_button, self.forget_button),
                # One under the other: the wall button names the well and the side
                # (*Capture B2 wall at image left*) and the select is 300 px wide, which
                # together ran past the panel's edge (``/ui-check``, #443).
                pn.Column(self.capture_wall_button, self.capture_side_select),
                self.image_note,
                self.capture_note,
                self.captures_text,
            ]
        elif tool.name == "plate-walls":
            # The owner's path (m5 §12.1): the centres of two or three wells, recorded
            # where the operator reads the tool, above its Run. Not the wall captures:
            # the tool images the walls itself.
            parts += [
                pn.Row(self.walls_capture_button, self.walls_forget_button),
                self.walls_capture_note,
                self.walls_captures_text,
                self.walls_touches_note,
            ]
        if advanced_widgets:
            parts.append(pn.Card(*advanced_widgets, title="Advanced", collapsed=True))
        parts += [run, note, card]
        return pn.Column(pn.pane.Str(tool.name, styles={"font-weight": "bold"}), *parts)

    def _runner(self, plugin: str) -> Callable[[Any], None]:
        def clicked(event: Any) -> None:
            self.on_run(plugin)

        return clicked

    # --- handlers ---------------------------------------------------------------

    def _handler(self, fn: Callable[..., Any]) -> Callable[..., None]:
        """``fn`` as Panel and Bokeh should call it: under the document lock, and never raising.

        Panel runs a handler without the session's lock when an event flushes a change
        it had queued: the Enter of a text field arrives with the field's change, and
        the watcher then writes the page's own Bokeh models (a ``ColumnDataSource``, a
        title) outside the lock, where Bokeh raises and the browser shows nothing
        (#277, FM-171). ``pn.io.unlocked()`` is what Panel's widgets use for each of
        their own writes; inside a locked callback it hands the events back to Bokeh,
        so it is safe for a tap and a click too. What a handler raises is a line of the
        page, not only a traceback in a console the operator does not read; ``Exception``
        only, so that Ctrl-C passes. The exception is formatted and kept nowhere
        (FM-98). Every handler the page registers goes through here.
        """

        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> None:
            try:
                # Looked up at call time: a test replaces it with a spy.
                with pn.io.unlocked():
                    fn(*args, **kwargs)
            except Exception as exc:
                log.exception("a handler of the plate page failed")
                text = str(exc)
                if len(text) > _FAILURE_TEXT_CHARS:
                    text = text[:_FAILURE_TEXT_CHARS] + "…"
                self.message.object = (
                    f"that action failed on the page: {type(exc).__name__}: {text}"
                    " (see the server's log)"
                )

        return run

    def _try(self, action: Callable[[], object]) -> bool:
        """Run ``action``; an operator's mistake becomes the message, and nothing else is kept."""
        try:
            action()
        except (ValueError, RuntimeError, PresetError) as exc:
            self.message.object = str(exc)
            return False
        self.message.object = ""
        return True

    def on_tap(self, event: Any) -> None:
        """A tap on the map: select the well under it, or (drive mode) drive to it, once."""
        drive = bool(self.drive_toggle.value)
        side = str(self.side_select.value)
        self._try(
            lambda: self.page.click(
                float(event.x), float(event.y), drive=drive, side=side
            )
        )
        if drive:
            # One click: an armed stage that stays armed after a mis-click is the danger.
            self.drive_toggle.value = False
        self.refresh()
        self.tick()

    def on_refresh(self) -> None:
        """The *Refresh* button: forget the last outcome, then read the run document and the stand again.

        Only the button forgets the strip's outcome (m5 §8.8.3): ``refresh()`` is also
        what every other handler and the end of a run call, and must not drop the
        outcome it is about to show. The tick then redraws the strip, so the old
        text is gone at once and not at the next period.
        """
        self.page.clear_outcome()
        self.refresh()
        self.tick()

    def on_select_text(self, text: str) -> None:
        """A typed range (``A1:D6``): the selection becomes exactly those wells.

        The box keeps what was typed: ``_wells_shown`` is the selection the box now
        stands for, so the redraw does not rewrite it. The box's own echo of a click
        (``_echo_wells``) comes back here, synchronously, and selects nothing: the
        ``_echoing`` guard alone stops it. A box the operator empties is a blank
        text, which clears the selection (m5 §8.8.5), and ``_wells_shown`` is then
        ``()``, so the next click is echoed again.
        """
        if self._echoing:
            return
        if self._try(lambda: self.page.select_text(text)):
            self._wells_shown = self.page.selection
        self.refresh()

    def on_next(self) -> None:
        """Run the next step the last redraw drew: its tool with its pairs, as shown.

        Never the step as it would be computed now, and never the tool's form: the
        operator runs the pairs they read (m5 §8.5, #287). A click on the disabled
        button still gets here (FM-169): the refusal is the controller's, as for
        every Run.
        """
        drawn = self._next_drawn
        if drawn is None or drawn.plugin is None:
            self.message.object = "nothing to run: press Refresh"
            return
        plugin, params = drawn.plugin, list(drawn.params)
        self._try(lambda: self.page.run(plugin, params))
        self.tick()

    def _row(self, index: int) -> CentreRow | None:
        """The centre row ``index`` as the last redraw drew it, or ``None`` when it has none."""
        rows = self._rows_drawn
        return (
            rows[index] if 0 <= index < min(len(rows), len(self.centre_slots)) else None
        )

    def on_goto_row(self, index: int) -> None:
        """*Go to <well>* of centre row ``index``: ``PlatePage.goto`` of the well the row named when it was drawn (m5 §17.4.2, D10).

        The same call as a drive click, so before a registration it drives to the holder's
        estimate (#462) and refuses where the layer does. What it came to is drawn on the row of
        the well, from the controller's answer (``status().goto_answer``): a refusal beside the
        button, not in the header's line far from it (on the owner's walk a drive before the
        plate was registered moved nothing and said nothing, #431). A click on a disabled button
        still arrives here (FM-169): the refusal is the controller's. A click refused because
        an activity goes (a drive, Live) is the state's, which the one note above the rows
        already says, and the controller leaves the running drive's text as it was.
        """
        row = self._row(index)
        if row is None:
            self.message.object = "nothing to go to: press Refresh"
            return
        # Both refusals are the controller's to say: the busy one by the note above the rows,
        # a refusal of the click's own by the answer it kept for this well.
        with contextlib.suppress(RuntimeError, ValueError):
            self.page.goto(row.well)
        self.message.object = ""
        self.tick()

    def on_capture_row(self, index: int) -> None:
        """*Capture the centre of <well>* (or *Redo*) of centre row ``index``: the well the row named when it was drawn.

        Never the controller's ``current``: the label names the well, and what the label
        said is what is recorded (FM-177). Accepted while Live runs, so that the operator
        centres in Live and captures there (D11); the controller refuses while another
        activity goes or the stage moves (FM-169).
        """
        row = self._row(index)
        if row is None:
            self.message.object = "nothing to capture: press Refresh"
            return
        well = row.well
        self._try(lambda: self.page.capture_centre(well))
        self.refresh()

    def on_register(self) -> None:
        """*Register the plate*: ``plate-registration`` with the pairs the last redraw drew, as drawn (m5 §17.4.2, #287).

        Never pairs computed at the click: the operator runs what the *Command* disclosure
        shows, as ``on_next`` does. With a wait drawn there are no pairs, and the click, which
        a disabled button still delivers, says what the screen waits for (FM-169); a busy
        stand or a plate nothing named is the controller's refusal.
        """
        pairs, wait = self._register_drawn
        if not pairs:
            self.message.object = wait or "nothing to register: press Refresh"
            return
        self._try(lambda: self.page.run("plate-registration", list(pairs)))
        self.tick()

    def on_choose_plate(self, name: str) -> None:
        """Another plate: the selection and the captures of the old one are dropped."""
        if name == self.page.plate:
            return
        self._try(lambda: self.page.choose_plate(name))
        self.refresh()

    def on_choose_orientation(self, value: PlateOrientation) -> None:
        """The plate lies another way: the controller decides what that does (m5 §16 D4).

        On a registered plate it starts ``plate-orientation`` and the select shows the
        new value once that run ends (the refresh after it); a refusal is the outcome
        and the refresh puts the select back. A re-sync by ``refresh`` does nothing.
        """
        if self._syncing_orientation or value == self.page.orientation:
            return
        self._try(lambda: self.page.choose_orientation(value))
        self.refresh()

    def _well_the_buttons_name(self) -> str:
        """The well the capture buttons were drawn for, when it is still the controller's current well (FM-177).

        The label said ``B3``; another tab, or a click between two ticks, may have made
        ``B4`` current since. Recording ``B3``'s wall at the position the operator took
        for ``B4``'s is a wrong plate registration with nothing to show it, and
        recording what the label did not say is what the page must not do. Neither is
        safe, so the press records nothing and asks again; the redraw that follows
        shows the well it would record now.

        Raises:
            ValueError: No well was drawn, or ``current`` is another well now.
        """
        drawn, now = self._capture_well, self.page.current
        if drawn is None:
            raise ValueError("click a well on the map first")
        if now != drawn:
            raise ValueError(
                f"the well to record changed from {drawn} to "
                f"{'none' if now is None else now}: check the button, then press it again"
            )
        return drawn

    def on_confirm_plate(self) -> None:
        """The *The plate is a …* button: the plate shown is the one on the stage (m5 §11.3).

        ``choose_plate`` with the plate already shown confirms it and clears nothing:
        the captures made on the placeholder are the operator's and are kept. A click
        on a button that is hidden still arrives here (FM-169); confirming a plate
        already confirmed is harmless.
        """
        self._try(lambda: self.page.choose_plate(self.page.plate))
        self.refresh()

    def on_capture(self, kind: str) -> None:
        """Capture the stage position as the centre, or a wall, of the well the button names."""

        def capture() -> None:
            well = self._well_the_buttons_name()
            if kind == "centre":
                self.page.capture_centre(well)
            elif kind == "wall":
                self.page.capture_wall(well, str(self.capture_side_select.value))
            else:
                raise ValueError(f"cannot capture {kind!r}; capture a centre or a wall")

        self._try(capture)
        self.refresh()

    def on_forget(self) -> None:
        """Forget the captures of the well the button names."""

        def forget() -> None:
            self.page.forget(self._well_the_buttons_name())

        self._try(forget)
        self.refresh()

    def on_run(self, plugin: str) -> None:
        """Run a tool with what the page fills and the fields the operator typed."""

        def run() -> None:
            self.page.run(
                plugin, [*self.page.page_params(plugin), *self._typed(plugin)]
            )

        self._try(run)
        self.tick()

    def on_run_all(self) -> None:
        """*Run all that are left*: the Results tools that loaded, each with the pairs its own *Run* sends (m5 §17.11.2).

        The pairs are read here, at the click, from the page's values and the fields the
        operator typed: the controller knows no widget. Which tools are left is the
        controller's to decide from the run document read now, so a list this tab drew before
        another one ran a tool can never run it again. A tool that did not load has a row that
        says so and is not passed. Whatever the controller refuses is the message, as a tool's
        own *Run* does.
        """
        tools = [tool for tool in RESULT_TOOLS if tool in self.tool_runs]

        def run_all() -> None:
            params = {
                tool: [*self.page.page_params(tool), *self._typed(tool)]
                for tool in tools
            }
            self.page.run_chain(tools, params)

        self._try(run_all)
        self.tick()

    def _typed(self, plugin: str) -> list[str]:
        """The ``key=value`` pairs of ``plugin``'s non-empty fields: what a run sends and a preset keeps."""
        return [
            f"{key}={widget.value.strip()}"
            for key, widget in self.tool_fields.get(plugin, {}).items()
            if widget.value.strip() != ""
        ]

    def on_connect(self) -> None:
        """*Connect*: open the profile's stand on the worker; the tick that sees it connected enables the controls."""
        self._try(self.page.connect)
        self.tick()

    def on_disconnect(self) -> None:
        """*Disconnect*: release the stand for another program; refused while an activity goes."""
        self._try(self.page.disconnect)
        self.tick()

    def on_stop(self) -> None:
        """STOP: always offered, a run or not; what it did is the stop note."""
        self.page.stop()
        self.tick()

    def on_resume(self) -> None:
        """Resume: clears the halt; refused while a run is going."""
        self._try(self.page.resume)
        self.tick()

    def on_snap(self) -> None:
        """Snap one frame; the tick that first sees the frame draws it, beside the map.

        A refusal (an activity is going) is the page's message, as for a run. The image
        is always on screen (m5 §17.2 D1), so a snap changes no screen (D2).
        """
        self._try(self.page.snap)
        self.tick()

    def on_live(self) -> None:
        """Start live, or end it.

        Which of the two is read from the label the button shows, not from the
        controller at the click (m5 §5.3, #287): a click does what the button says,
        and a button up to a second out of date must not do the opposite. A stale
        *End live* after a camera fault, with no live running, ends nothing and
        starts no new live on the camera that just failed (a live that another tab
        started meanwhile is ended, as the label says: ``live(False)`` ends the
        page's live, not the tab's); a stale *Live* in a second tab is the
        controller's refusal (``live is running``) and does not end the first
        tab's live. The tick that follows draws the label again. Ending is not STOP: the stand is
        not halted. A refusal is the page's message, as for a run.
        """
        starting = self.live_button.label != _END_LIVE_LABEL
        self._try(lambda: self.page.live(starting))
        self.tick()

    def on_jog(self, direction: str) -> None:
        """An arrow of the pad: one jog by the chosen step, in the stage's axes (FM-73)."""

        def jog() -> None:
            sign = _XY_DIRECTIONS.get(direction)
            if sign is None:
                raise ValueError(f"no jog {direction!r}; up, left, right or down")
            step = float(self.xy_step_select.value)
            self.page.jog_xy(sign[0] * step, sign[1] * step)

        self._try(jog)
        self.tick()

    def on_z(self, direction: str) -> None:
        """``Z +`` or ``Z -``: one focus jog by the chosen step; the layer's interlock applies."""

        def jog() -> None:
            sign = _Z_DIRECTIONS.get(direction)
            if sign is None:
                raise ValueError(f"no Z jog {direction!r}; up or down")
            self.page.jog_z(sign * float(self.z_step_select.value))

        self._try(jog)
        self.tick()

    def on_autofocus(self, on: bool) -> None:
        """Engage or disengage the autofocus."""
        self._try(lambda: self.page.autofocus(on))
        self.tick()

    def on_objective(self) -> None:
        """*Change objective*: arm the confirmation with the chosen label; nothing turns yet (FM-14)."""

        def arm() -> None:
            label = self.objective_select.value
            if label is None:
                raise ValueError("choose an objective first")
            self._armed = (str(label),)
            self.objective_confirm_text.object = (
                f"Turn the turret to {str(label)!r}? A turn can drive an objective "
                f"into a loaded plate: check that the stage is clear. The autofocus "
                f"is switched off and not engaged again."
            )
            self.objective_confirm.visible = True

        self._try(arm)

    def on_objective_confirm(self) -> None:
        """*Turn the turret*: the armed label, with ``confirm=True``; then disarmed.

        With nothing armed (a stale browser shows the button after a disarm), it
        turns nothing and says so.
        """
        armed = self._armed
        self._disarm()
        if armed is None:
            self.message.object = "choose the objective again"
            self.tick()
            return
        self._try(lambda: self.page.objective(armed[0], confirm=True))
        self.tick()

    def on_objective_cancel(self) -> None:
        """*Cancel*: disarm; nothing turns."""
        self._disarm()

    def on_light(self, on: bool) -> None:
        """Switch the light on or off."""
        self._try(lambda: self.page.light(on))
        self.tick()

    def _disarm(self) -> None:
        self._armed = None
        self.objective_confirm.visible = False

    # --- presets (m5 §10.6) --------------------------------------------------------

    def on_load_preset(self) -> None:
        """*Load*: the chosen preset's values in the plate, the wells and every field; ``(none)`` forgets it.

        A load is choosing the preset's values: every field of every tool is set to
        the preset's value, else the profile's suggestion, else empty, so the form
        shows what a fresh page with this preset shows, and a field typed before is
        replaced. The fields are set under ``_filling``, so the note is drawn once.
        A preset the plate's captures refuse to change the plate for is loaded all the
        same, its mismatch in the note (the controller's rule).
        """
        name = str(self.preset_select.value or "")
        if name == "":
            self.page.forget_preset()
            self._expected_of = None
            self._preset_expected = {}
            self._disarm_preset()
            self.message.object = ""
            self.refresh()
            return
        loaded: list[LoadedPreset] = []
        if not self._try(lambda: loaded.append(self.page.load_preset(name))):
            self._draw_preset_note()
            return
        self._fill_fields(loaded[0])
        # So that a Save after edits asks to overwrite this preset, not to make another.
        self.preset_name.value = loaded[0].file.name
        self._disarm_preset()
        self.refresh()

    def _fill_fields(self, loaded: LoadedPreset) -> None:
        """Every field of every tool set to what ``loaded`` gives it, and remembered as what the preset put there."""
        expected = self._expected_from(loaded)
        self._filling = True
        try:
            for tool, fields in self.tool_fields.items():
                for key, widget in fields.items():
                    widget.value = expected[tool][key]
        finally:
            # Always: a guard left up would stop the note following the edits.
            self._filling = False
        self._preset_expected = expected
        self._expected_of = loaded

    def on_save_preset(self) -> None:
        """*Save*: the plate, the *Wells* box and the non-empty fields, under the name typed.

        An existing name arms the confirmation instead of replacing the file. Nothing
        is written for a value its tool refuses: the message says which.
        """
        name = str(self.preset_name.value).strip()
        if name == "":
            self.message.object = "type a name to save the preset under"
            return
        self._save_preset(name, overwrite=False)

    def on_preset_overwrite(self) -> None:
        """*Overwrite*: save the armed name over the file that holds it, with the fields as they are now.

        With nothing armed (a stale tab shows the button after a cancel), or a name
        typed since, nothing is saved and the message says so.
        """
        armed = self._preset_armed
        self._disarm_preset()
        if armed is None:
            self.message.object = "nothing to overwrite: press Save again"
            return
        if str(self.preset_name.value).strip() != armed[0]:
            self.message.object = "the name changed: press Save again"
            return
        self._save_preset(armed[0], overwrite=True)

    def on_preset_cancel(self) -> None:
        """*Cancel*: nothing is overwritten."""
        self._disarm_preset()

    def _disarm_preset(self) -> None:
        self._preset_armed = None
        self.preset_confirm.visible = False

    def _save_preset(self, name: str, *, overwrite: bool) -> None:
        saved: list[LoadedPreset] = []
        exists: list[str] = []

        def save() -> None:
            fields = {tool: self._typed(tool) for tool in self.tool_fields}
            try:
                saved.append(
                    self.page.save_preset(
                        name, fields, str(self.select_input.value), overwrite=overwrite
                    )
                )
            except PresetExistsError as exc:
                # Text, never the error (FM-98).
                exists.append(str(exc))

        if not self._try(save):
            return
        if exists:
            self._preset_armed = (name,)
            self.preset_confirm_text.object = f"{exists[0]}: overwrite it?"
            self.preset_confirm.visible = True
            return
        self._disarm_preset()
        loaded = saved[0]
        # The saved preset is what the fields hold: nothing differs until they change.
        self._preset_expected = self._actual()
        self._expected_of = loaded
        self._offer_presets(selected=name)
        self.refresh()
        self.message.object = f"saved {name} to {loaded.file.path}"

    def _on_field_changed(self, event: Any) -> None:
        """A tool's field changed: the note says how the page differs from the preset, unless a load is filling them."""
        if self._filling:
            return
        self._draw_preset_note()
        # *Calibrate the camera*, *Lay out the fields* and *Start the scan* run the same
        # fields: their commands follow them, and so do the lines that say where the Wells
        # screen's two values come from (m5 §17.10.3). A load draws them once, at its end.
        self._draw_calibrate_command()
        self._draw_field_lines()
        self._draw_layout_command()
        self._draw_scan_command()
        self._draw_result_lines()
        self._draw_result_commands()

    def _actual(self) -> dict[str, dict[str, str]]:
        """What every tool's fields hold now, by tool and key."""
        return {
            tool: {key: str(widget.value) for key, widget in fields.items()}
            for tool, fields in self.tool_fields.items()
        }

    def _expected_from(self, loaded: LoadedPreset) -> dict[str, dict[str, str]]:
        """What a load of ``loaded`` puts in every field: its value, else the profile's suggestion, else ``""``."""
        expected: dict[str, dict[str, str]] = {}
        for tool in self._tools:
            given = loaded.fields.get(tool.name, {})
            suggested = dict(tool.suggested)
            expected[tool.name] = {
                key: given.get(key, suggested.get(key, ""))
                for key in self.tool_fields[tool.name]
            }
        return expected

    def _offer_presets(self, selected: str | None = None) -> None:
        """The select's options, read from the folders: a broken file is left out, its error is ``smc presets list``'s.

        Read by ``refresh()`` and after a save, never by the tick (FM-151). The
        choice stays when it is still offered.
        """
        options = {NO_PRESET: ""}
        for entry in self.page.presets():
            if entry.preset is not None:
                options[entry.name] = entry.name
        keep = selected if selected is not None else self.preset_select.value
        self.preset_select.options = options
        self.preset_select.value = keep if keep in options.values() else ""

    def _draw_preset_note(self) -> None:
        """The note under the region: the preset, what it could not apply, and how the page differs from it."""
        loaded = self.page.preset
        differences: tuple[str, ...] = ()
        if loaded is not None:
            expected = (
                self._preset_expected
                if self._expected_of is loaded
                else self._expected_from(loaded)
            )
            differences = (
                *preset_differences(expected, self._actual()),
                *self._page_differences(loaded),
            )
        self.preset_note.object = preset_note_text(loaded, differences)

    def _page_differences(self, loaded: LoadedPreset) -> tuple[str, ...]:
        """The two differences that are not a field's: the plate, and the wells a load selected."""
        found: list[str] = []
        applied = loaded.check.plate
        if applied is not None and applied != self.page.plate:
            found.append(f"plate: {self.page.plate} (the preset: {applied})")
        if loaded.selection is not None and self.page.selection != loaded.selection:
            found.append(
                f"wells: {len(self.page.selection)} selected "
                f"(the preset: {loaded.check.wells})"
            )
        return tuple(found)

    # --- drawing ------------------------------------------------------------------

    def tick(self) -> None:
        """The periodic callback's body: the strip, the run, the buttons and a new frame, never the view.

        It reads the stand once (``status()``) and redraws what depends on it. The
        plate's wells are redrawn by ``refresh`` when a run ends, not here, and
        the image only when the controller holds a frame this session has not drawn.
        """
        status = self.page.status()
        previous, self._running = self._running, status.running
        if status.stand != self._stand:
            # The first tick: the cards drawn by `refresh` did not know the stand.
            self._stand = status.stand
            self._draw_cards()
        if status.connected != self._seen_connected:
            self._follow_connection(status.connected)
        elif status.connected and not self.objective_select.options:
            self._read_objectives_again()
        self._draw_connection(status)
        self._draw_stand(status)
        # The Wells and Scan screens, from this status and the last view: text only, the
        # tick reads nothing new (FM-145). The status is kept for ``refresh()``, which
        # redraws the same texts from the view it has just read and never reads the stand.
        self._status = status
        self._guarded(self._draw_wells_screen, status)
        self._guarded(self._draw_scan_screen, status)
        self._draw_results_screen(status)
        self._draw_imaged(status)
        # How many tools of a chain have ended since the last tick (per view, D6): a tool's
        # end inside a chain changes the run document with no activity ending, so the rows
        # are read again for it (m5 §17.11.2). A chain that is gone, or a new one, is no end.
        chain = status.chain
        ended = 0 if chain is None else len(chain.ended)
        chain_moved = ended > 0 and ended != self._chain_ended
        self._chain_ended = ended
        self.status_text.object = status_line(status)
        self.run_text.object = run_text(status)
        self.goto_text.object = status.goto
        self.resume_button.visible = status.halted
        self.stop_button.disabled = False
        # While Live runs and its frames hold the cross, the tick leaves it to them: a
        # frame's position is newer than the status read a moment ago. Outside Live the
        # flag is cleared and the status draws it, so the cross never stays at an older
        # position than the stage's after Live ends (m5 §15 D1).
        if not self.page.is_live:
            self._cross_from_frames = False
        if not self._cross_from_frames:
            self._draw_cross(status.xy_um)
        self._problems_status = status.problems
        if not self._objective_shown and status.objective in self._objective_labels:
            # Until it has, once: the select starts at the objective in place, and
            # the tick never moves it after (the strip says which one is in place).
            # A read that failed gives ``None``, which is no option: the next tick
            # tries again, so a first ``current()`` that raised does not leave
            # the select on its first option, with *Change objective* turning to
            # that one and not to the objective in place (#295).
            self._objective_shown = True
            self.objective_select.value = status.objective
        self._draw_problems()
        self._follow_frame(self._draw_frame())
        self._draw_buttons()
        self._draw_drive()
        if previous is not None and status.running is None:
            # The end of a run, a chain's last tool included: one redraw covers the end the
            # chain's count also reports in the same tick.
            self.refresh()
        elif chain_moved or self._shared_state() != self._seen_state:
            # Another tab chose wells or a plate, or took a capture: the controller is
            # the server's, so a tab that does not redraw would run `scan-plan` with
            # wells it does not show. Reading five attributes is no view: the view is
            # rebuilt only when one of them changed.
            self.refresh()
        # The chain's line last: after the rows it is about have been redrawn (D20's note).
        self._draw_chain_line(status)

    def _guarded(
        self, draw: Callable[[PageStatus | None], None], status: PageStatus | None
    ) -> None:
        """Run one screen's draw so that a raise in it is logged and the tick goes on (m5 §17.12.3, #470).

        ``tick()`` draws the Wells and Scan screens before the strip's text and the buttons: a raise
        in either skipped them, and the operator was left with a strip that said nothing and a STOP
        that did not follow the stand. The failure is logged with its traceback (the server's log),
        never kept, and the screen keeps what it drew last.
        """
        try:
            draw(status)
        except Exception:
            log.exception(
                "%s failed: the screen keeps what it drew last", draw.__name__
            )

    def _follow_connection(self, connected: bool) -> None:
        """The objective's choices of the stand now held, or none: read again when the connection changes.

        The select is set to the objective in place by the tick's own once-only
        step, armed again here; a confirmation armed for the old stand is dropped.
        """
        self._seen_connected = connected
        self._offer_objectives(self.page.objective_labels())

    def _read_objectives_again(self) -> None:
        """Read the turret's labels again on a connected stand that offered none (#295).

        One failed ``labels()`` at the build, or at Connect, left the control
        disabled for the life of the tab. The tick reads again while the choices
        are empty, and offers them when the read gives some; a stand without a
        turret answers ``()`` without a device read and without a log line, and a
        read that keeps failing is logged once by the controller.
        """
        labels = self.page.objective_labels()
        if labels:
            self._offer_objectives(labels)

    def _offer_objectives(self, labels: tuple[str, ...]) -> None:
        """Offer ``labels`` as the select's choices, and let the tick set it to the objective in place.

        A confirmation armed for other choices is dropped.
        """
        self._objective_labels = labels
        self.objective_select.options = {label or NO_LABEL: label for label in labels}
        self._objective_shown = False
        self._disarm()

    def _draw_connection(self, status: PageStatus) -> None:
        """The *Connect* row: which of the two buttons applies, and a line only while a connect or a disconnect goes or after one failed.

        Both buttons are disabled while an activity goes (live included), so that
        Disconnect is never offered under a run; the controller refuses a click
        all the same (FM-169). While ``connect`` or ``disconnect`` is the activity
        the line says so: on a slow stand it would otherwise read ``Connect opens it``
        beside a greyed-out button, and the page would look stuck (m5 §7.3, #305). After a
        connect that failed it says why the controls are disabled; at any other time it is
        empty and takes no height (m5 §17.12 D15: the line on every screen took 27 px of
        a 623 px page, and what Disconnect does is said once, on the Stand screen).
        """
        busy = status.running is not None
        self.connect_button.disabled = busy or status.connected
        self.disconnect_button.disabled = busy or not status.connected
        outcome = status.outcome or ""
        if status.running == "connect":
            text = f"connecting {status.stand}…"
        elif status.running == "disconnect":
            text = f"disconnecting {status.stand}…"
        elif not status.connected and _a_connection_failed(outcome, status.stand):
            text = (
                f"{status.stand} is not connected: Connect opens it, and the "
                f"controls that need the stand are disabled until then"
            )
        else:
            # Neither goes nor failed: the strip says the stand is connected or not, and
            # what Disconnect does is on the Stand screen (m5 §17.12 D15).
            self.connection_text.object = ""
            return
        if status.dry_run:
            text += " (dry run)"
        self.connection_text.object = text

    def frame_tick(self) -> None:
        """The frame callback's body: draw the controller's frame when it is new, and the cross from it while Live runs.

        It reads no status and no view (FM-145): it runs ten times a second, and the
        stand's read and the run document's belong to ``tick``. It reads the controller's
        frame and, for a frame it has just drawn, whether Live runs: every ``Frame``
        carries the stage's position read right after its snap, so the cross follows the
        frames with no device read of its own (m5 §15 D1). Outside Live a frame is old by
        the time it is drawn (a snap's), and the cross stays where the tick put it; a frame
        with no position leaves the cross and the flag as they were.
        """
        self._follow_frame(self._draw_frame())

    def _follow_frame(self, frame: Frame | None) -> None:
        """Move the cross to the position of ``frame``, the frame just drawn, while Live runs.

        Both callbacks draw frames, so both call it: the tick sees a new frame first now and
        then, and a cross that only the frame tick moved would stay a frame behind the
        picture (the lag is one frame period, which a long exposure makes visible). ``None``
        is no new frame; a frame with no position, or one drawn when Live is not running (a
        snap's), moves nothing and leaves the flag as it was.
        """
        if frame is None or frame.xy_um is None or not self.page.is_live:
            return
        self._cross_from_frames = True
        self._draw_cross(frame.xy_um)

    def _draw_cross(self, xy_um: tuple[float, float] | None) -> None:
        """Draw the cross at ``xy_um``, or none, with the label of the well it is over.

        The one place that sets ``stage_source``, so that ``x``, ``y`` and ``text`` are
        set at once and of one length (a closed stand gives three empty lists). The label
        is computed from the last drawn view, which is ``None`` before the first refresh
        (the label is then empty): it never asks the controller for a new one (FM-145).
        """
        self._cross_xy = xy_um
        view = self._view
        label = "" if view is None else stage_label(view.wells, view.pose, xy_um)
        self._align_cross_label(xy_um)
        self.stage_source.data = {
            "x": [] if xy_um is None else [xy_um[0]],
            "y": [] if xy_um is None else [xy_um[1]],
            "text": [] if xy_um is None else [label],
        }

    def _align_cross_label(self, xy_um: tuple[float, float] | None) -> None:
        """Put the cross's label on the side of the cross that faces the frame's middle (m5 §17.12 D16).

        Beside the right edge the label is drawn to the left of the cross, beside the top edge
        below it: the label is a single point's, so its alignment is the layer's, set at each
        draw. Without a frame or a position the defaults stand (up and to the right).
        """
        frame = self._frame_drawn
        if xy_um is None or frame is None:
            return
        (x0, x1), (y0, y1) = frame
        near_right = xy_um[0] > (x0 + x1) / 2.0
        near_top = xy_um[1] > (y0 + y1) / 2.0
        self.cross_label.text_align = "right" if near_right else "left"
        self.cross_label.x_offset = -12 if near_right else 12
        self.cross_label.text_baseline = "top" if near_top else "bottom"
        self.cross_label.y_offset = -12 if near_top else 12

    def _frame_outlines(self) -> list[tuple[float, float, float, float]]:
        """The outlines the map is framed on for the screen shown (m5 §17.12 D16): the plan's wells, or the plate's.

        The Stand and Plate screens and the expert view frame the plate; the Wells, Scan and
        Results screens frame the wells of the plan the document holds, **even one that is not
        usable** (an out-of-date plan is still the wells the operator laid out), and the plate when
        none stands or none of its wells is on this plate. An operator's own zoom is Bokeh's, and
        stops the range following until the toolbar's *Reset*; nothing here overrides it.
        """
        view = self._view
        if view is None or not view.wells:
            return []
        planned = set(view.frame_wells)
        if planned and not self.expert and self.step in ("wells", "scan", "results"):
            chosen = [well for well in view.wells if well.well in planned]
            if chosen:
                return [_outline(well) for well in chosen]
        return [_outline(well) for well in view.wells]

    def _draw_map_frame(self) -> None:
        """Set the frame's two points for the screen shown, when they changed (m5 §17.12 D16).

        Called by every redraw of the wells and by every change of screen, never by the tick (it
        reads the last view, FM-145). With no outline (no wells drawn) the frame is left as it
        is: a frame of nothing is no frame. The margin is one pitch of the plate, from ``useq``
        as the controller's sides use it; a plate with no spacing (a coverslip) falls back to
        the size of its well.
        """
        outlines = self._frame_outlines()
        view = self._view
        if view is None or not outlines:
            return
        spacing_mm = useq.WellPlate.from_str(view.plate).well_spacing
        pitch_um = max(spacing_mm) * 1000.0
        if pitch_um <= 0.0:
            pitch_um = max(max(abs(o[2] - o[0]), abs(o[3] - o[1])) for o in outlines)
        frame = map_frame(outlines, pitch_um)
        if frame == self._frame_drawn:
            return
        self._frame_drawn = frame
        (x0, x1), (y0, y1) = frame
        self.frame_source.data = {"x": [x0, x1], "y": [y0, y1]}
        # The label's side depends on where the frame's middle is.
        self._align_cross_label(self._cross_xy)

    def _shared_state(self) -> tuple[Any, ...]:
        """What the controller holds that every tab draws: compared, never rebuilt.

        ``activities_ended`` is there for the tab that did not start a tool: one that
        ends between two of its ticks is never seen running, and its result card,
        the checklist and the blocked tools are new all the same (FM-151).
        """
        page = self.page
        return (
            page.plate,
            # Another tab's confirmation of the placeholder changes the note, the
            # button and the tools' notes without changing the plate (FM-151).
            page.plate_chosen,
            # Another tab's choice on an estimate changes no attribute above and no
            # activity ends for it (FM-151); a turn's run does, and is here too.
            page.orientation,
            page.selection,
            page.refs,
            # Each centre's objective, which another tab's recapture changes with no change
            # to the centres themselves when the stage did not move (FM-151).
            page.capture_objectives,
            page.walls,
            page.current,
            page.activities_ended,
            # The controller's loaded preset, by identity: another tab's load changes
            # the next step and the note without changing anything above (FM-151).
            page.preset,
        )

    def refresh(self) -> None:
        """Read the run document again and redraw the map, the route and the checklist."""
        # Taken before the view is: a change made meanwhile is seen by the next tick.
        self._seen_state = self._shared_state()
        view = self.page.view()
        self._view = view
        self._blocked = dict(view.blocked)
        self._problems_view = view.problems
        labels = list(view.labels)
        self.label_select.options = labels
        if self.label_select.value not in labels:
            self.label_select.value = labels[0] if labels else None
        # The *Density of* menu has nothing to choose before a density exists, and was drawn
        # empty above the map from the Stand screen on (the owner's walk, D16).
        self.label_select.visible = bool(labels)
        if self.plate_select.value != view.plate:
            self.plate_select.value = view.plate
        if self.orientation_select.value != view.orientation:
            self._syncing_orientation = True
            try:
                self.orientation_select.value = view.orientation
            finally:
                self._syncing_orientation = False
        self._draw_wells()
        self._echo_selection()
        self.route_source.data = {
            "x": [stop.x_um for stop in view.route],
            "y": [stop.y_um for stop in view.route],
            "order": [stop.order for stop in view.route],
            "well": [stop.well for stop in view.route],
        }
        plan = view.plan
        fields = () if plan is None else plan.fields
        width_um, height_um = (0.0, 0.0) if plan is None else plan.fov_um
        self.fields_source.data = {
            "x": [x_um for x_um, _y_um in fields],
            "y": [y_um for _x_um, y_um in fields],
            "w": [width_um] * len(fields),
            "h": [height_um] * len(fields),
        }
        if self._running != "scan":
            # A scan that goes draws its imaged fields from its progress (``_draw_imaged``),
            # and this redraw, another tab's end of a run or a click, must not draw an older
            # scan's record over them (m5 §17.10.2). The end of the run calls this again,
            # with no scan running: the layer is then the record's.
            self._imaged_drawn = None
            self._set_imaged(() if plan is None else plan.imaged, (width_um, height_um))
        self.checklist_text.object = "\n".join(
            {
                "ok": f"✓ {item.slot}",
                "not run": f"· {item.slot}",
                "unusable": f"✗ {item.slot}: {item.reason}",
            }[item.state]
            for item in view.checklist
        )
        refs = [f"{r.well} centre ({r.x_um:.1f}, {r.y_um:.1f})" for r in self.page.refs]
        walls = [
            f"{w.well} {w.side} wall ({w.x_um:.1f}, {w.y_um:.1f})"
            for w in self.page.walls
        ]
        self.captures_text.object = "\n".join([*refs, *walls]) or "no capture yet"
        # One read of each for the two texts: another tab may capture between two reads.
        centres, touches = self.page.refs, self.page.walls
        self.walls_captures_text.object = centres_text(centres)
        self.walls_touches_note.object = wall_touches_note(len(touches))
        # From the same view as the rest (FM-151); a calibration run from the page shows
        # after the run's own refresh. Each is text, with three cases (m5 §13.2).
        self.image_note.object = image_sides_note(view.camera_matrix)
        self.drive_image_note.object = drive_image_note(view.camera_matrix)
        self._draw_cards()
        self._draw_next()
        self._draw_steps()
        self._draw_problems()
        self._draw_plate_ask()
        self._draw_plate_screen()
        self._draw_commands()
        self._draw_buttons()
        if self._status is not None:
            # Before the first tick there is no status; that tick draws both screens.
            self._guarded(self._draw_wells_screen, self._status)
            self._guarded(self._draw_scan_screen, self._status)
            self._draw_results_screen(self._status)
            self._draw_chain_line(self._status)
        self._draw_drive()
        # The options are read here and after a save, never by the tick (§10.6).
        self._offer_presets()
        self._draw_preset_note()

    def _echo_selection(self) -> None:
        """Show the selection in the *Wells* box when it is not the one the box stands for (m5 §8.5).

        A click on the map, another tab, a plate chosen or a run document read changes
        the selection without the box: it is then set to the wells, comma-separated.
        A range the operator typed stays as typed (``on_select_text`` records it).
        The assignment calls the box's watcher at once, so it runs under ``_echoing``,
        which makes ``on_select_text`` return before it selects the same wells again.
        """
        selection = self.page.selection
        if selection == self._wells_shown:
            return
        self._wells_shown = selection
        self._echoing = True
        try:
            self.select_input.value = ",".join(selection)
        finally:
            self._echoing = False

    def _draw_next(self) -> None:
        """Draw the view's next step: its text, its button, and its stage opened when the tool changed.

        The step is the one this redraw's ``view()`` built (FM-151); the tick never
        builds it. It opens the stage's card in the accordion but never closes one the
        operator opened.
        """
        view = self._view
        step = None if view is None else view.next_step
        self._next_drawn = step
        self.next_text.object = next_step_text(step)
        tool = None if step is None else step.plugin
        self.next_button.visible = tool is not None
        if tool is None:
            self._next_tool = None
            return
        self.next_button.label = f"Run {tool}"
        if tool != self._next_tool:
            self._next_tool = tool
            index = self._stage_index.get(tool)
            if index is not None:
                self.accordion.active = sorted({*self.accordion.active, index})

    def _draw_cards(self) -> None:
        """The result card under each tool, from the view: a result follows the run document, not the tick."""
        cards = {} if self._view is None else self._view.cards
        for name, pane in self.tool_cards.items():
            pane.object = card_text(cards.get(name), stand=self._stand)

    def _draw_frame(self) -> Frame | None:
        """Draw the controller's frame when it is one this session has not drawn, and return it; else ``None``.

        The axes are the frame's own pixels: ``dw`` and ``dh`` are its width and
        height, not those of the strided picture (FM-136). Bokeh draws an image's
        first row at the bottom, so the rows are flipped to put the camera's top
        row on top.
        """
        frame = self.page.frame()
        if frame is None or frame.number == self._seen_frame:
            return None
        self._seen_frame = frame.number
        self._draw_image(frame)
        # Once for each new frame, never at a tick: 4.5 ms at 512 x 512 and 16.5 ms at the Prime
        # 95B's 1200 x 1200, on every other row and column past 1600 x 1600 (m5 §17.12 D22, FM-145).
        self._frame_hint = frame_hint(frame.image, frame.bit_depth)
        # ``_camera_lacks`` is the tick's last read: this callback asks the controller for the
        # frame and nothing else (FM-145), so a camera that is gone is the tick's to say.
        self.snap_note.object = self._camera_lacks or self._frame_hint
        return frame

    def _draw_image(self, frame: Frame) -> None:
        shown = display_image(frame.image, bit_depth=frame.bit_depth)
        height, width = frame.image.shape
        self.image_source.data = {
            "image": [np.ascontiguousarray(shown[::-1])],
            "dw": [width],
            "dh": [height],
        }
        self.image_figure.title.text = frame.summary

    def _draw_wells(self) -> None:
        view = self._view
        if view is None:
            return
        label = self.label_select.value
        values = [
            math.nan
            if label is None or w.per_mm2 is None or label not in w.per_mm2
            else w.per_mm2[label]
            for w in view.wells
        ]
        self._ensure_renderer(bool(view.wells[0].circular) if view.wells else True)
        self.wells_source.data = {
            "name": [w.well for w in view.wells],
            "x": [w.center_um[0] for w in view.wells],
            "y": [w.center_um[1] for w in view.wells],
            "r": [w.size_um[0] / 2.0 for w in view.wells],
            "w": [w.size_um[0] for w in view.wells],
            "h": [w.size_um[1] for w in view.wells],
            "angle": [math.radians(w.rotation_deg) for w in view.wells],
            "value": values,
            "outline": [
                _SELECTED_OUTLINE if w.selected else _PLAIN_OUTLINE for w in view.wells
            ],
            "outline_width": [3 if w.selected else 1 for w in view.wells],
            "tip": [_tip(w, label) for w in view.wells],
        }
        known = [v for v in values if math.isfinite(v)]
        self.mapper.low = 0.0
        self.mapper.high = max(known) if known and max(known) > 0.0 else 1.0
        # The bar says which class its colours count: the select's, so that a change of the
        # select, which draws the wells again, changes it too (m5 §17.4.5).
        self.colour_bar.title = colour_bar_title(label) or None
        self.names.visible = len(view.wells) <= NAMES_UP_TO_WELLS
        self.side_source.data = {
            "x": [side.x_um for side in view.sides],
            "y": [side.y_um for side in view.sides],
            "text": [side.text for side in view.sides],
            "angle": [side.angle_deg for side in view.sides],
        }
        a1 = [w for w in view.wells if w.well == "A1"]
        self.a1_source.data = {
            "x": [w.center_um[0] for w in a1],
            "y": [w.center_um[1] for w in a1],
            "text": [w.well for w in a1],
        }
        # Where the wells are drawn is only as good as the registration: say which map it is
        # (registered, seed, estimated or unplaced, m5 §15 D4). The scan plan's suffix is
        # where a plan too large to draw says so (m5 §14.3).
        title = map_title(view.plate, view.pose) + plan_title(view.plan)
        self.figure.title.text = title
        self.map_title_text.object = title
        # The frame before the cross: the label's side is the frame's (D16).
        self._draw_map_frame()
        # The cross's label names a well of *this* view: a registration that moved the wells
        # relabels it now, from the position last drawn, and reads nothing (FM-145).
        self._draw_cross(self._cross_xy)

    def _ensure_renderer(self, circular: bool) -> None:
        """Round wells are circles and square ones rectangles: swap the glyph when the plate changes."""
        if self._circular == circular and self.wells_renderer is not None:
            return
        if self.wells_renderer is not None:
            self.figure.renderers.remove(self.wells_renderer)
        fill = {"field": "value", "transform": self.mapper}
        common: dict[str, Any] = {
            "source": self.wells_source,
            "fill_color": fill,
            "fill_alpha": 0.85,
            "line_color": "outline",
            "line_width": "outline_width",
        }
        if circular:
            renderer = self.figure.circle(x="x", y="y", radius="r", **common)
        else:
            renderer = self.figure.rect(
                x="x", y="y", width="w", height="h", angle="angle", **common
            )
        renderer.level = "underlay"
        self.hover.renderers = [renderer]
        self.wells_renderer = renderer
        self._circular = circular

    def _draw_problems(self) -> None:
        self.problems_text.object = "\n".join(
            [*self._problems_view, *self._problems_status]
        )

    def _draw_buttons(self) -> None:
        """A tool that is blocked, or any tool or snap while an activity goes, cannot be started.

        A disabled *Run* says why beside the button (m5 §11.2): its own reason, which
        holds longer than any activity (the stand closed, a missing input, the plate
        not chosen), else what the running activity is and how to end it. A tool
        that would replace a usable plate adds that, on the next line (§11.4).

        The captures read the stage, so they are disabled while the stand is
        closed (the *Connect* row says why) and while an activity other than this
        page's live goes (the controller refuses a capture then, #295).
        """
        running = self._running
        live = self.page.is_live
        busy = None if running is None else busy_note(running, live=live)
        replaces = {} if self._view is None else self._view.replaces
        for name, button in self.tool_runs.items():
            reason = self._blocked.get(name)
            button.disabled = running is not None or reason is not None
            lines = (reason or busy, replaces.get(name))
            self.tool_notes[name].object = "\n".join(line for line in lines if line)
        # The next step's button: while an activity goes, or when the step says why not.
        blocked = None if self._next_drawn is None else self._next_drawn.blocked
        self.next_button.disabled = running is not None or blocked is not None
        self.next_note.object = blocked or busy or ""
        self._draw_captures()
        self._draw_stand_plate_buttons()
        self._draw_wells_scan_buttons()
        self._draw_results_buttons()
        lacks = self.page.lacks(Camera)
        self.snap_button.disabled = running is not None or lacks is not None
        # While live goes its own button is the one that ends it; any other activity
        # leaves it nothing to do.
        self.live_button.label = _END_LIVE_LABEL if live else _LIVE_LABEL
        self.live_button.disabled = lacks is not None or (
            running is not None and not live
        )
        self._camera_lacks = lacks
        self.snap_note.object = lacks or self._frame_hint
        # A turn on a registered plate is a run (m5 §16 D4): offered when one can start,
        # so not while Live goes either, as every Run. On an estimate it runs nothing and
        # stays offered beside Live. The controller refuses a run all the same when the
        # select is not disabled (FM-169).
        kind = None if self._view is None else self._view.pose.kind
        self.orientation_select.disabled = running is not None and (
            not live or kind in ("registered", "seed")
        )

    def _draw_plate_ask(self) -> None:
        """The note and the button that ask which plate is on the stage, visible only while nothing named it (m5 §11.3).

        The *Plate* select shows the placeholder (its options and value are
        unchanged) and the page says what it is, with the one button that confirms
        it. ``refresh`` draws it: ``plate_chosen`` is in the shared state, so another
        tab's confirmation redraws this one.
        """
        asking = not self.page.plate_chosen
        plate = self.page.plate
        self.plate_note.visible = asking
        self.plate_confirm_button.visible = asking
        if asking:
            self.plate_note.object = (
                f"Which plate is on the stage? {plate} is only a placeholder: "
                f"choose the plate, or press the button if it is a {plate}."
            )
            self.plate_confirm_button.label = f"The plate is a {plate}"

    def _draw_stand(self, status: PageStatus) -> None:
        """The Stand screen's line, its *Connect* and its "known" block, from the status the tick just read (m5 §17.4.1).

        The block says what the page knows and where from (D7): the objective in place and
        the light as the status read them, the camera's calibration as the last redraw read
        it from the run document, which a closed stand still has. ``*Calibrate the camera*``
        is offered only when the calibration is missing or made at another objective
        (``needs_calibration``), and only while the tool is loaded. The strip and the screen
        draw one stand, so the second *Connect* follows the strip's rule: it is there while
        the stand is closed and disabled while any activity goes.
        """
        if status.running == "connect":
            line = f"{status.stand}: connecting…"
        elif status.running == "disconnect":
            line = f"{status.stand}: disconnecting…"
        else:
            line = f"{status.stand}: {'connected' if status.connected else 'not connected'}"
        self.stand_text.object = line
        self.stand_connect_button.visible = not status.connected
        self.stand_connect_button.disabled = status.running is not None
        view = self._view
        calibration = None if view is None else view.camera_calibration
        objective = status.objective or None
        self.known_block.object = known_text(
            objective=objective,
            calibration=calibration,
            light_on=status.light_on,
            stand=status.stand if status.connected else None,
        )
        self.calibrate_box.visible = "camera-calibration" in self.tool_runs and (
            needs_calibration(objective, calibration)
        )

    def _draw_plate_screen(self) -> None:
        """The Plate screen from this redraw's view: where the plate comes from, the centre rows, what *Register* runs, what it came to (m5 §17.4.2).

        The rows are rewritten here and never rebuilt (D5), from the controller's captures
        and the view's pose; the pairs *Register the plate* runs are the ones drawn now
        (``on_register``), so a capture, a plate or a preset changed in any tab redraws
        them at the next tick (``_shared_state``). A slot with no row is hidden; the rows
        past the slots are text. The result line is the registered plate's, with the
        ``assumed`` lines of the card of the tool that wrote it: a registration from a
        seed or from the sample says what it did not measure.
        """
        page = self.page
        view = self._view
        self.plate_from_text.object = plate_from_line(
            plate=page.plate,
            chosen=page.plate_chosen,
            loaded=page.preset,
            sample_plate=page.sample_plate,
        )
        rows = centre_rows(
            page.plate,
            page.refs,
            page.capture_objectives,
            pose="unplaced" if view is None else view.pose.kind,
            from_sample=page.registers_from_sample,
        )
        self._rows_drawn = rows
        for index, slot in enumerate(self.centre_slots):
            row = rows[index] if index < len(rows) else None
            slot.box.visible = row is not None
            if row is None:
                continue
            slot.text.object = row.text
            slot.goto_button.label = f"Go to {row.well}"
            slot.capture_button.label = (
                "Redo" if row.captured else f"Capture the centre of {row.well}"
            )
            self._say(slot.capture_note, self._capture_note_of(row, view))
        others = rows[len(self.centre_slots) :]
        self.other_captures.object = (
            "\n".join(["Other captures:", *(row.text for row in others)])
            if others
            else ""
        )
        self.other_captures.visible = bool(others)
        self._register_drawn = page.registration_step()
        registered = None if view is None else view.registered
        lines: list[str] = []
        if registered is not None and view is not None:
            lines.append(registered_line(registered))
            card = view.cards.get(view.registered_by or "")
            if card is not None:
                lines += [f"assumed: {text}" for text in card.assumed]
        self.registered_block.object = "\n".join(lines)
        self.registered_block.visible = bool(lines)
        # Said before the click, as the tool's own section says it in the expert view (§11.4).
        self._say(
            self.register_replaces,
            "" if registered is None else register_replace_note(registered),
        )

    def _capture_note_of(self, row: CentreRow, view: PlateView | None) -> str:
        """The note under a captured row that is far from its well or on another's place, else ``""`` (m5 §17.12.3).

        The expected place is the centre the map draws for that well in the pose drawn
        (registered, seed or estimated), and none on an *unplaced* map, which puts no well
        anywhere: then only two captures at one place can be said. The pitch is the plate's,
        from ``useq``, as the map and the frame use it. A capture the controller no longer
        holds (another tab forgot it) has no note.
        """
        if not row.captured or view is None:
            return ""
        held = {ref.well: (ref.x_um, ref.y_um) for ref in self.page.refs}
        place = held.get(row.well)
        if place is None:
            return ""
        centres = {well.well: well.center_um for well in view.wells}
        expected = None if view.pose.kind == "unplaced" else centres.get(row.well)
        pitch_um = max(useq.WellPlate.from_str(view.plate).well_spacing) * 1000.0
        others = {well: xy for well, xy in held.items() if well != row.well}
        return capture_note(row.well, place, expected, pitch_um, others)

    def _draw_commands(self) -> None:
        """The *Command* disclosures of the Stand and Plate screens: the call each button makes (ADR-0006, principle 5, D8).

        *Register the plate* shows the pairs it runs, drawn with them; *Calibrate the camera*
        the pairs of the tool's own form, which it reads at the click and which
        ``_on_field_changed`` draws again.
        """
        pairs, _wait = self._register_drawn
        self.register_command.object = (
            self.page.command_for("plate-registration", pairs) if pairs else ""
        )
        self.register_card.visible = bool(pairs)
        # A page served on Windows says its commands are quoted for a POSIX shell (FM-175).
        note = posix_quoting_note(self._windows)
        self._say(self.register_command_note, note)
        self._say(self.calibrate_command_note, note)
        self._say(self.layout_command_note, note)
        self._say(self.scan_command_note, note)
        self._draw_calibrate_command()
        self._draw_layout_command()
        self._draw_scan_command()
        self._draw_result_commands()

    def _draw_calibrate_command(self) -> None:
        """The command of ``camera-calibration`` with the page's pairs and the form's typed fields."""
        try:
            pairs = [
                *self.page.page_params("camera-calibration"),
                *self._typed("camera-calibration"),
            ]
        except ValueError:
            self.calibrate_command.object = ""
            return
        self.calibrate_command.object = self.page.command_for(
            "camera-calibration", pairs
        )

    def _draw_stand_plate_buttons(self) -> None:
        """Which buttons of the Stand and Plate screens are disabled, and the reason beside each (m5 §11.2, §17.2 D11, §17.4).

        Drawn at every tick and every redraw. *Go to* is disabled while any activity goes,
        Live included; the capture stays enabled while Live runs, since the operator centres
        in Live and captures there (D11). **What holds for every row is said once**, in the note
        above them (``plate_rows_note``: what runs, or that Live has to end first, and what
        *Register the plate* uses while nothing is captured, m5 §17.12 D19, D23); a row says only
        what the last drive to its well came to (``goto_row_text``, a refusal or a failure in the
        warning colour, going and arrived in the plain one). *Register the plate* is disabled
        while a wait is drawn, with the wait beside it: the tool's own reason (the next step's
        precedence: what the run document or the stand says first wins, and ``CAPTURE_WELLS`` is
        only this wait: before any capture nothing is in the warning colour), or the refusal that
        a turn of the turret since the captures would meet (``register_refusal``, D13). The
        controller refuses every one of these calls itself (FM-169).
        """
        page = self.page
        running = self._running
        live = page.is_live
        connected = page.connected
        busy = None if running is None else busy_note(running, live=live)
        blocked = self._blocked.get("camera-calibration")
        self.calibrate_button.disabled = running is not None or blocked is not None
        self._say(self.calibrate_note, blocked or busy or "")
        stage_off = not connected or (running is not None and not live)
        self._say(
            self.rows_note,
            plate_rows_note(
                running=running, live=live, from_sample=page.registers_from_sample
            ),
        )
        # The controller's answer as it is now, not the tick's copy: a redraw that follows a
        # capture or a registration must not show the one it ended (FM-151, FM-145: a read of
        # one attribute, no stand).
        answer = page.goto_answer
        trouble = answer is not None and answer.phase in ("refused", "failed")
        for slot, row in zip(self.centre_slots, self._rows_drawn, strict=False):
            slot.goto_button.disabled = not connected or running is not None
            slot.capture_button.disabled = stage_off
            self._say(slot.note, goto_row_text(answer, row.well))
            slot.note.styles = {
                **_WRAP_STYLES,
                **({"color": _WARNING} if trouble and answer.well == row.well else {}),  # type: ignore[union-attr]
            }
        for slot in self.centre_slots[len(self._rows_drawn) :]:
            slot.goto_button.disabled = not connected or running is not None
            slot.capture_button.disabled = stage_off
            self._say(slot.note, "")
        _pairs, wait = self._register_drawn
        view = self._view
        reason = (
            self._blocked.get("plate-registration")
            or wait
            or (None if view is None else view.register_refusal)
        )
        if "plate-registration" not in self.tool_runs:
            reason = "plate-registration did not load: the problems line says why"
        self.register_button.disabled = running is not None or reason is not None
        self._say(self.register_wait, reason or "")

    def _field_line(self, field: tuple[str, str]) -> str:
        """The line under one of the Wells screen's two tool fields: where its value comes from (m5 §17.2 D7, §17.10.3).

        From the field as it is now and from what a load or the profile would put there: the
        preset's own value, else the profile's suggestion (``_expected_from``'s rule), else
        the operator typed it, and an empty field is the tool's default, in the words of the
        placeholder. ``""`` for a tool that did not load: its field is not on the screen.
        """
        tool, key = field
        widget = self.tool_fields.get(tool, {}).get(key)
        view = next((t for t in self._tools if t.name == tool), None)
        if widget is None or view is None:
            return ""
        loaded = self.page.preset
        given = None if loaded is None else loaded.fields.get(tool, {}).get(key)
        return field_source_line(
            value=str(widget.value).strip(),
            default=self._field_defaults.get(field, ""),
            preset=None if loaded is None else loaded.file.name,
            preset_value=None if given is None else given.strip(),
            suggested=dict(view.suggested).get(key),
        )

    def _draw_field_lines(self) -> None:
        """The three "from" lines of the Wells screen: the selection, the coverage and the autofocus (m5 §17.10.3).

        Called by the screen's redraw and by a field's own change (``_on_field_changed``),
        never while a load assigns the fields: a load ends with ``refresh()``.
        """
        loaded = self.page.preset
        self.wells_from_text.object = wells_from_line(
            self.page.selection,
            None if loaded is None else loaded.file.name,
            None if loaded is None else loaded.selection,
        )
        self.coverage_from_text.object = self._field_line(_COVERAGE_FIELD)
        self.autofocus_from_text.object = self._field_line(_AUTOFOCUS_FIELD)

    def _draw_wells_screen(self, status: PageStatus | None) -> None:
        """The Wells screen's lines from the last view and this status: the "from" lines, what is known and the selection's note (m5 §17.4.3, §17.10).

        The objective is the one in place as the status read it, the plan is the view's
        usable plan, and a plan that is not usable says why in the checklist's words (the
        card of the tool carried them before this screen replaced its section: FM-226). The
        plan card's ``assumed`` lines go under it, plain text (FM-139). The button and its
        wait are ``_draw_wells_scan_buttons``'s. A text pane that says nothing takes no height.
        """
        view = self._view
        plan = None if view is None else view.plan
        self._draw_field_lines()
        item = (
            None
            if view is None
            else next((i for i in view.checklist if i.slot == "scan-plan"), None)
        )
        card = None if view is None else view.cards.get("scan-plan")
        self.wells_known_block.object = "\n".join(
            wells_known_lines(
                objective=None if status is None else (status.objective or None),
                plan=plan,
                out_of_date=(
                    item.reason
                    if item is not None and item.state == "unusable"
                    else None
                ),
                assumed=() if card is None else card.assumed,
            )
        )
        self._say(
            self.wells_note,
            plan_selection_note(
                () if plan is None else plan.well_names, self.page.selection
            ),
        )

    def _draw_scan_screen(self, status: PageStatus | None) -> None:
        """The Scan screen from the last view and this status: its title and lead, the dry-run note, the progress and the result line (m5 §17.4.4, §17.10).

        The title and the lead name the **plan's** wells and fields, the selection while there
        is no plan. The bar and its lines are drawn only for a report of ``scan``: Live's
        report is ``live``, and another activity's start clears the report, so the block goes
        with the next activity and stays after the scan until then. ``max`` is at least 1 and
        ``value`` never above it (Panel refuses a ``value`` above the ``max`` it holds at that
        moment, so ``max`` is set first), and the stripes move only while the scan goes. The
        result line follows the run document, not
        the progress, so it survives a restart (``view.scanned``); the card's ``assumed``
        lines and its ``folder:`` line go under it. The dry-run note is plain and said once.
        """
        view = self._view
        plan = None if view is None else view.plan
        self.scan_title_text.object = scan_title(
            () if plan is None else plan.well_names, self.page.selection
        )
        self.scan_lead_text.object = scan_lead(
            plan, stop_refused=bool(status is not None and status.stop_refused)
        )
        self._say(
            self.scan_dry_note,
            DRY_RUN_NO_FRAME if status is not None and status.dry_run else "",
        )
        progress = None if status is None else status.progress
        shown = False
        if status is not None and progress is not None and progress.plugin == "scan":
            shown = True
            total = progress.done if progress.total is None else progress.total
            going = status.running == "scan"
            maximum = max(total, 1)
            # ``max`` first: Panel refuses a ``value`` above the ``max`` it holds at that
            # moment (``param.update`` sets ``value`` first and raised, measured), and does
            # not refuse a ``max`` below the ``value`` it holds, which is what a shorter
            # scan after a longer one asks for.
            self.scan_progress.max = maximum
            self.scan_progress.value = min(max(progress.done, 0), maximum)
            self.scan_progress.active = going
            self._say(
                self.scan_progress_text,
                "\n".join(
                    scan_progress_lines(
                        progress.done, total, status.clock, running=going
                    )
                ),
            )
        else:
            self._say(self.scan_progress_text, "")
        self.scan_progress.visible = shown
        summary = None if view is None else view.scanned
        lines: list[str] = []
        if view is not None and summary is not None:
            lines.append(scanned_line(summary))
            card = view.cards.get("scan")
            if card is not None:
                lines += [f"assumed: {text}" for text in card.assumed]
                lines += list(card.files)
        self._say(self.scanned_block, "\n".join(lines))

    def _draw_imaged(self, status: PageStatus) -> None:
        """While a scan runs, the plan's first ``done`` fields are drawn as imaged: from the progress, only when it changed (m5 §17.10.2).

        The scan visits the plan's fields in the plan's order (measured: the reports' names
        are the plan's), so the first ``done`` are the ones with a frame, a failed field
        counted as visited (the end of the run redraws from the record, which holds the frames
        that were taken). A **dry run** takes none, whatever the progress says, and the page
        says so (the note, the result line): nothing is drawn while one runs, as the record
        draws none after it. Nothing is drawn before the first report either: the record of an
        earlier scan, drawn by the last redraw, is cleared here. A ``done`` below what was drawn
        is a new run and is drawn from its start. It reads the status and the last view only
        (FM-145); with no scan running ``refresh()`` draws the record, and the end of the run
        calls it.
        """
        if status.running != "scan":
            return
        view = self._view
        plan = None if view is None else view.plan
        fields = () if plan is None else plan.fields
        progress = status.progress
        done = (
            min(max(progress.done, 0), len(fields))
            if progress is not None and progress.plugin == "scan" and not status.dry_run
            else 0
        )
        if done == self._imaged_drawn or plan is None:
            return
        self._imaged_drawn = done
        self._set_imaged(fields[:done], plan.fov_um)

    def _set_imaged(
        self, centres: Sequence[tuple[float, float]], fov_um: tuple[float, float]
    ) -> None:
        """The imaged fields' source set whole, ``fov_um`` wide and high at each centre (m5 §17.10.2)."""
        width_um, height_um = fov_um
        self.imaged_source.data = {
            "x": [x_um for x_um, _y_um in centres],
            "y": [y_um for _x_um, y_um in centres],
            "w": [width_um] * len(centres),
            "h": [height_um] * len(centres),
        }

    def _draw_wells_scan_buttons(self) -> None:
        """Which of *Lay out the fields* and *Start the scan* is disabled, and the reason beside each (m5 §11.2, §17.4.3, §17.4.4).

        Each button is disabled while a wait is drawn or an activity goes, and says its wait
        (the first that holds, ``lay_out_wait`` and ``scan_wait``) or, with none, what runs
        and how to end it (``busy_note``). The tool's own reason is the controller's
        (``_blocked``): the stand closed, what the tool reads, a checkpoint the page never
        starts over (FM-172). The controller refuses each of these calls itself (FM-169).
        """
        page = self.page
        view = self._view
        running = self._running
        busy = None if running is None else busy_note(running, live=page.is_live)
        connected = page.connected
        plate_chosen = page.plate_chosen
        blocked = self._blocked.get("scan-plan")
        if connected and not plate_chosen:
            # ``scan-plan`` takes no plate parameter, so the controller's reasons for it never
            # name the placeholder: what it says for a plate nobody registered is the run
            # document's, written for a terminal. The screen's own words (m5 §17.4.3).
            blocked = PLATE_NOT_CHOSEN
        lay_out = lay_out_wait(
            connected=connected,
            plate_chosen=plate_chosen,
            registered=view is not None and view.registered is not None,
            blocked=blocked,
            selected=len(page.selection),
        )
        if "scan-plan" not in self.tool_runs:
            lay_out = "scan-plan did not load: the problems line says why"
        self.layout_button.disabled = running is not None or lay_out is not None
        self._say(self.layout_wait, lay_out or busy or "")
        start = scan_wait(
            connected=connected,
            blocked=self._blocked.get("scan"),
            plan_usable=view is not None and view.plan is not None,
        )
        if "scan" not in self.tool_runs:
            start = "scan did not load: the problems line says why"
        self.scan_button.disabled = running is not None or start is not None
        self._say(self.scan_wait_text, start or busy or "")

    def _draw_layout_command(self) -> None:
        """The command of *Lay out the fields*: ``scan-plan`` with the page's wells and the form's typed fields (m5 §17.2 D8).

        Read at a redraw and at a field's change, never by the tick (it reads the run
        document). A selection the controller refuses for (none, or on another plate) draws
        no command, as ``_draw_calibrate_command`` does, and the card is hidden with it.
        """
        try:
            pairs = [*self.page.page_params("scan-plan"), *self._typed("scan-plan")]
        except ValueError:
            self.layout_command.object = ""
        else:
            self.layout_command.object = self.page.command_for("scan-plan", pairs)
        self.layout_card.visible = bool(self.layout_command.object)

    def _draw_scan_command(self) -> None:
        """The command of *Start the scan*: ``scan`` with the form's typed fields (m5 §17.2 D8)."""
        try:
            pairs = [*self.page.page_params("scan"), *self._typed("scan")]
        except ValueError:
            self.scan_command.object = ""
        else:
            self.scan_command.object = self.page.command_for("scan", pairs)
        self.scan_card.visible = bool(self.scan_command.object)

    def _draw_results_screen(self, status: PageStatus | None) -> None:
        """The Results screen from the last view and this status: the rows, the chain's line and where the files are (m5 §17.4.5, §17.11).

        Text only, from the last view and the status: it reads nothing new (FM-145), which is
        also why a second tab draws the chain the first one started, from the shared status
        (FM-151). A row's state line is the checklist's, the card's time and what is running
        (``result_row``); the tool's own progress is the status's report, and only while that
        tool runs. ``dry_scan`` is a page served with ``--dry-run`` over a scan that was one,
        the rule the next step used for ``DRY_RUN_NO_FRAME``. The buttons and their waits are
        ``_draw_results_buttons``'s, called last so that a redraw never leaves them behind.
        """
        view = self._view
        checklist = () if view is None else view.checklist
        cards = {} if view is None else view.cards
        scan = cards.get("scan")
        dry_scan = bool(
            status is not None and status.dry_run and scan is not None and scan.dry_run
        )
        running = None if status is None else status.running
        report = None if status is None else status.progress
        rows: dict[str, ResultRow] = {}
        for tool, widgets in self.result_rows.items():
            progress = (
                (report.done, report.total)
                if report is not None and report.plugin == tool
                else None
            )
            row = result_row(
                tool,
                checklist=checklist,
                card=cards.get(tool),
                blocked=self._blocked.get(tool),
                dry_scan=dry_scan,
                running=running,
                progress=progress,
            )
            rows[tool] = row
            widgets.state.object = row.state
            widgets.button.label = row.button
            self._say(widgets.replaces, again_note(tool, checklist))
        self._results_drawn = rows
        self._results_left = tuple(
            tool for tool in results_left(checklist) if tool in self.tool_runs
        )
        # The chain's line is not drawn here: a tool of the chain that ended is also a new
        # row, and the line is drawn after the rows' redraw, in the same tick (``_draw_chain_line``).
        self.files_text.object = files_html(cards, self.page.run_doc.parent)
        self._draw_result_lines()
        self._draw_results_buttons()

    def _draw_chain_line(self, status: PageStatus | None) -> None:
        """The chain's line under *Run all that are left*, drawn **after** the rows it is about (m5 §17.12.3, #446's Report).

        A tool of a chain that ended changes the run document with no activity ending, and the tick
        reads the rows again for it (``refresh()``); the line is the status's, so drawn before
        that it said ``Running Route (3 of 5)`` over a density row that still said ``Ready``. It is
        called last by the tick and by ``refresh()``.
        """
        self._say(self.chain_text, chain_line(None if status is None else status.chain))

    def _draw_results_buttons(self) -> None:
        """Which of *Run all that are left* and the five rows' buttons is disabled, and the reason beside each (m5 §11.2, §17.11.2).

        A button is disabled while an activity goes (a chain's own tools included: the strip
        names the tool that runs) or a wait is drawn, and says the wait or, with none, what
        runs and how to end it. A row whose input is missing is disabled and its state line
        says which step makes it. A tool that did not load says so, and the chain's wait is
        what its first tool left waits for. The controller refuses each of these calls itself
        (FM-169).
        """
        running = self._running
        busy = None if running is None else busy_note(running, live=self.page.is_live)
        for tool, widgets in self.result_rows.items():
            row = self._results_drawn.get(tool)
            reason = "" if row is None else row.wait
            if tool not in self.tool_runs:
                reason = f"{tool} did not load: the problems line says why"
            widgets.button.disabled = (
                running is not None or bool(reason) or row is None or row.missing
            )
            # A row's own reason only: what runs holds for every row, and is said once, under
            # *Run all that are left* (m5 §17.12 D19).
            self._say(widgets.wait, reason)
        wait = (
            "the results tools did not load: the problems line says why"
            if not any(tool in self.tool_runs for tool in RESULT_TOOLS)
            else run_all_wait(list(self._results_drawn.values()), self._results_left)
        )
        self.run_all_button.disabled = running is not None or wait is not None
        self._say(self.run_all_wait_text, wait or busy or "")

    def _draw_result_lines(self) -> None:
        """The *from …* line under each choice of the rows: where its value comes from (m5 §17.2 D7).

        Called by the screen's redraw and by a field's own change (``_on_field_changed``),
        never while a load assigns the fields: a load ends with ``refresh()``.
        """
        for tool, widgets in self.result_rows.items():
            for key, line in widgets.from_lines.items():
                line.object = self._field_line((tool, key))

    def _draw_result_commands(self) -> None:
        """The command of each row, with the page's pairs and the tool's typed fields, and the Windows note under it (m5 §17.2 D8, FM-175).

        Read at a redraw and at a field's change, never by the tick (it reads the run
        document). A tool that did not load, or whose pairs the controller refuses, draws no
        command, and its card is hidden with it unless it holds a choice.
        """
        note = posix_quoting_note(self._windows)
        for tool, widgets in self.result_rows.items():
            try:
                pairs = [*self.page.page_params(tool), *self._typed(tool)]
            except ValueError:
                widgets.command.object = ""
            else:
                widgets.command.object = (
                    self.page.command_for(tool, pairs) if tool in self.tool_runs else ""
                )
            self._say(widgets.command_note, note if widgets.command.object else "")
            widgets.card.visible = bool(widgets.command.object) or bool(
                widgets.from_lines
            )

    @staticmethod
    def _say(pane: Any, text: str) -> None:
        """``text`` in ``pane``, which takes no height while it is empty (m5 §17.3, D12).

        An empty text pane still draws a line of its own: under each of the three centre
        rows it spread them out by about 25 px, which D12's 620 px page cannot spare.
        """
        pane.object = text
        pane.visible = bool(text)

    def _draw_captures(self) -> None:
        """The three capture buttons name the well they record and the side of the *Wall* select (m5 §11.1).

        The well is the controller's ``current`` as it is now, kept as ``_capture_well``:
        a press records that well, or nothing when ``current`` has moved since
        (``_well_the_buttons_name``). With no well the buttons are the plain three,
        disabled, and the note says what to do. The captures read the stage, so they
        are also disabled with the stand closed (the *Connect* row says why) and
        while an activity other than this page's live goes, which the controller
        refuses a capture for; while live goes they stay enabled: only the
        controller knows whether a drive action waits or runs, and its refusal is
        the message. *Forget* reads no stage: only a missing well disables it. A
        disabled button is a courtesy, not a guard (FM-169).

        ``plate-walls``' centre buttons are set here too, from the same ``well``
        (m5 §12.5): ``_capture_well`` stays one value, so a press on either set
        records the well its label named, and FM-177's check is one check.

        Called by ``_draw_buttons`` and by the *Wall* select's own watcher.
        """
        well = self.page.current
        self._capture_well = well
        side = str(self.capture_side_select.value)
        centre_label = capture_centre_label(well)
        drop_label = forget_label(well)
        self.capture_centre_button.label = centre_label
        self.capture_wall_button.label = capture_wall_label(well, side)
        self.forget_button.label = drop_label
        self.walls_capture_button.label = centre_label
        self.walls_forget_button.label = drop_label
        stage_off = not self.page.connected or (
            self._running is not None and not self.page.is_live
        )
        self.capture_centre_button.disabled = stage_off or well is None
        self.capture_wall_button.disabled = stage_off or well is None
        self.forget_button.disabled = well is None
        self.walls_capture_button.disabled = stage_off or well is None
        self.walls_forget_button.disabled = well is None
        note = "" if well is not None else _NO_WELL_NOTE
        self.capture_note.object = note
        self.walls_capture_note.object = note

    def _draw_drive(self) -> None:
        """Every drive control disabled while an activity other than this page's live goes.

        Enabled while live goes: the live worker runs a drive action between two
        frames (m5 §6.3). A control whose stand lacks its capability is disabled
        too, and ``drive_note`` says why, one line each. A tick that finds the
        controls disabled disarms the objective's confirmation.
        """
        page = self.page
        busy = self._running is not None and not page.is_live
        objective = page.lacks(ObjectiveTurret)
        if objective is None and not self._objective_labels:
            objective = "its positions could not be read"
        reasons = {
            "pad": page.lacks(XYStage),
            "focus": page.lacks(ZStage),
            "autofocus": page.lacks(Autofocus),
            "objective": objective,
            "light": page.light_lacks(),
        }
        controls: dict[str, list[Any]] = {
            "pad": [*self.jog_buttons.values()],
            "focus": [*self.z_buttons.values()],
            "autofocus": [self.autofocus_on_button, self.autofocus_off_button],
            "objective": [self.objective_button, self.objective_confirm_button],
            "light": [self.light_on_button, self.light_off_button],
        }
        for name, widgets in controls.items():
            for widget in widgets:
                widget.disabled = busy or reasons[name] is not None
        self.xy_step_select.disabled = reasons["pad"] is not None
        self.z_step_select.disabled = reasons["focus"] is not None
        # *Drive* arms the next click on a well, which runs `goto-well` on the stand: with the
        # stand closed there is nothing to arm (the owner's walk, finding 14).
        self.drive_toggle.disabled = not page.connected
        if not page.connected and self.drive_toggle.value:
            self.drive_toggle.value = False
        self.objective_select.disabled = reasons["objective"] is not None
        if busy or reasons["objective"] is not None:
            self._disarm()
        if NOT_CONNECTED in reasons.values():
            # One reason for all five, said once (m5 §7.3).
            self.drive_note.object = NOT_CONNECTED
        else:
            self.drive_note.object = "\n".join(
                f"{name}: {reason}" for name, reason in reasons.items() if reason
            )


def build(page: PlatePage) -> Any:
    """One session's layout over the server's controller, with its own periodic callbacks.

    The callbacks (the tick and the frame tick) belong to the session (Panel stops
    them with it); the controller does not, so a closed or reloaded tab leaves a run,
    or a live view, going (FM-146).
    """
    view = PlatePageView(page)
    pn.state.add_periodic_callback(view.tick, period=TICK_MS)
    pn.state.add_periodic_callback(view.frame_tick, period=FRAME_TICK_MS)
    return view.layout


class RunFileHandler(tornado.web.RequestHandler):
    """``GET /run-files/<path>``: a file the run document names, and nothing else (m5 §17.12 D21).

    *Where the files are* links the report and the export to this route so that they open from
    the page. The route is the page's own and serves what ``PlatePage.run_file`` finds: the
    files a usable result of the run document lists, inside its folder, read at each request;
    **404** for anything else (a sibling, a parent, an absolute path, a folder, a file that is
    gone). It never lists a directory. The bytes are read in binary (``Path.read_bytes``), so no
    encoding is guessed, and the type is ``mimetypes``' for the name (``text/html`` for the
    report, ``application/xml`` for the export, ``application/json``), ``application/octet-stream``
    for one it does not know; a ``text/`` type says UTF-8, which is what the layer writes
    (CLAUDE.md). A file name is data: the path is never echoed into a page.
    """

    page: PlatePage

    def initialize(self, page: PlatePage) -> None:
        """Tornado's constructor hook: the page whose run document is served."""
        self.page = page

    def get(self, relative: str) -> None:
        """Answer with the file ``relative`` names, or 404 (see the class)."""
        path = self.page.run_file(relative)
        if path is None:
            raise tornado.web.HTTPError(404)
        try:
            body = path.read_bytes()
        except OSError:
            # A file the document names that cannot be read now (gone since the check, locked).
            raise tornado.web.HTTPError(404) from None
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if kind.startswith("text/"):
            kind += "; charset=utf-8"
        self.set_header("Content-Type", kind)
        # The bytes are shown as the type says, never sniffed into another (an export is not HTML).
        self.set_header("X-Content-Type-Options", "nosniff")
        self.write(body)


def serve(page: PlatePage, *, port: int) -> None:
    """Serve the page on ``localhost`` until Ctrl-C; a server that ends alone is an error.

    ``panel.serve`` blocking would swallow the operator's Ctrl-C (Bokeh installs its
    own handler, so it returns normally and nothing says that the operator
    interrupted), and a port that is taken kills the server thread silently, after
    ``serve`` has returned. So the server runs in a thread and the main thread waits
    in slices of 0.1 s (also what lets Ctrl-C through on Windows), where a
    ``KeyboardInterrupt`` is ours and a thread that ended alone is not.

    The page has no login and listens on ``localhost`` only: whoever reaches the
    port can move the stage.

    Raises:
        PageServerError: The server stopped without Ctrl-C (is the port taken?).
    """
    # The command's own printing: plain text, no markup (FM-36, FM-37).
    from smc.cli import _say

    # Bokeh logs through its own console handler at the root's level, and
    # pymmcore-plus leaves the root at DEBUG once a core is open: a route table at
    # the start and chatter through the session. A console that blocks (a QuickEdit
    # selection on Windows) would block the server's loop, which is the one that
    # handles the click on STOP (FM-62). The level is the page's for as long as it
    # serves, and put back after.
    bokeh_log = logging.getLogger("bokeh")
    level = bokeh_log.level
    bokeh_log.setLevel(logging.WARNING)
    interrupted = False
    try:
        thread = pn.serve(
            {"": lambda: build(page)},
            address="localhost",
            port=port,
            show=False,
            threaded=True,
            verbose=False,
            # The files of the run document open from the page (m5 §17.12 D21): one route, an
            # allowlist the controller computes at each request. ``static_dirs`` would refuse a
            # folder that does not exist yet and serve the whole run folder.
            extra_patterns=[(r"/run-files/(.*)", RunFileHandler, {"page": page})],
        )
        # Panel's thread is not a daemon: whatever ends this block, it is stopped, or
        # the process waits for it at exit.
        try:
            try:
                _say(
                    f"serving the plate page on http://localhost:{port}/ — Ctrl-C stops it"
                )
                while thread.is_alive():
                    thread.join(0.1)
            except KeyboardInterrupt:
                interrupted = True
                # The run is told to stop before the server is touched: stopping the
                # server waits for the callback in flight (measured: 12 s for a tick
                # that takes 6 s), and the stop of a scan must not wait behind it.
                page.close()
        finally:
            if thread.is_alive():
                with _quiet_session_close() as closing:
                    stopped = _stop_server(thread)
                if closing.dropped:
                    log.info(
                        "the server's stop closed %d open sessions", closing.dropped
                    )
                if not stopped:
                    _say(
                        f"the plate page's server did not stop within {SERVER_STOP_WAIT_S:.0f} s"
                    )
    finally:
        bokeh_log.setLevel(level)
    if interrupted:
        return
    raise PageServerError(
        f"the plate page's server stopped on its own (is port {port} taken?)"
    )


class _ServerStopFilter(logging.Filter):
    """Drops the one record Bokeh's server logs when it closes an open session, and counts it (m5 §11.5).

    A stop of the server discards its sessions and detaches each connection from its
    session before the socket's own close reaches ``on_close``, which then reads
    ``connection.session``: an ``AssertionError`` that Tornado logs as ``Uncaught
    exception GET /ws`` on every Ctrl-C with the page open in a browser, which is
    every Ctrl-C of a session. It is Bokeh's order of work, harmless at that point,
    and a traceback in front of the operator who just stopped the page on purpose.

    The message alone would hide a different failure with the same words, so the
    exception's class is checked too, and the filter is attached only while the
    server stops (``_quiet_session_close``). It does not patch Bokeh and closes no
    socket: both are a vendor's private API and a version's.
    """

    def __init__(self) -> None:
        super().__init__()
        #: How many records it dropped: the sessions the stop closed.
        self.dropped = 0

    def filter(self, record: logging.LogRecord) -> bool:
        error = None if record.exc_info is None else record.exc_info[1]
        if isinstance(error, AssertionError) and record.getMessage().startswith(
            "Uncaught exception GET /ws"
        ):
            self.dropped += 1
            return False
        return True


@contextlib.contextmanager
def _quiet_session_close() -> Iterator[_ServerStopFilter]:
    """``_ServerStopFilter`` on Tornado's application logger for the length of the block, removed after."""
    closing = _ServerStopFilter()
    logger = logging.getLogger("tornado.application")
    logger.addFilter(closing)
    try:
        yield closing
    finally:
        logger.removeFilter(closing)


def _stop_server(thread: Any) -> bool:
    """Stop Panel's server thread within ``SERVER_STOP_WAIT_S``; whether it ended.

    ``StoppableThread.stop`` ends with a join that has no limit, and the server's own
    shutdown waits for the callback in flight and then 5 s, so asking for the stop
    directly holds the console for as long as a slow device read does, and on Windows
    Ctrl-C does not interrupt that wait (FM-83). It is asked from a daemon thread and
    waited for in slices, and Panel 1.9's ``stop`` does not join, so the server
    thread is waited for too.
    """
    stopper = threading.Thread(
        target=thread.stop, name="smc-ui-stop-server", daemon=True
    )
    stopper.start()
    deadline = time.monotonic() + SERVER_STOP_WAIT_S
    while time.monotonic() < deadline:
        stopper.join(0.1)
        if not stopper.is_alive():
            thread.join(min(0.1, max(0.0, deadline - time.monotonic())))
            if not thread.is_alive():
                return True
    return False
