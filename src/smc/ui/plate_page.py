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
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import panel as pn
import useq
from bokeh.events import Tap
from bokeh.models import (
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
    END_LIVE_LABEL,
    NextStep,
    busy_note,
    capture_centre_label,
    capture_wall_label,
    centres_text,
    drive_image_note,
    forget_label,
    image_sides_note,
    map_title,
    plan_title,
    preset_differences,
    preset_note_text,
    stage_label,
    wall_touches_note,
)

__all__ = ["PlatePageView", "build", "card_text", "next_step_text", "serve"]

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

#: The centre's tabs, in the order they are built.
_PLATE_TAB = 0
_IMAGE_TAB = 1

#: The two labels of the live button: ``_draw_buttons`` draws them and ``on_live`` reads
#: them, so that a click does what the button says (m5 §5.3, #287). The second is the
#: guide's, which ``busy_note`` quotes: the note and the button cannot differ (m5 §11.2).
_LIVE_LABEL = "Live"
_END_LIVE_LABEL = END_LIVE_LABEL

#: The width of the note under the *Plate* select, in pixels: it wraps inside the map's column.
_PLATE_NOTE_WIDTH = 520

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
#: The width of the text panes of the tools column, inside its 420 px.
_TEXT_WIDTH = 380

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
    if sys.platform == "win32" if windows is None else windows:
        lines.append(
            "  that line is quoted for a POSIX shell (bash, zsh); PowerShell and "
            "cmd.exe read quotes differently: press the button, which makes the "
            "same call"
        )
    return "\n".join(lines)


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
        # The usable scan plan's fields, one rectangle each (m5 §14.2).
        self.fields_source = ColumnDataSource({"x": [], "y": [], "w": [], "h": []})
        self.mapper = LinearColorMapper(
            palette=Viridis256, low=0.0, high=1.0, nan_color=_UNKNOWN_FILL
        )
        self.figure = figure(
            match_aspect=True,
            height=560,
            sizing_mode="stretch_width",
            x_axis_label="stage X (µm)",
            y_axis_label="stage Y (µm)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
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
        self.figure.add_layout(ColorBar(color_mapper=self.mapper), "right")
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
        # the wells' fill and the names (m5 §15 D5).
        self.figure.add_layout(
            LabelSet(
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
                background_fill_alpha=0.7,
            )
        )
        self.figure.on_event(Tap, self._handler(self.on_tap))

        # The camera's frame: 8-bit grey from 0 to 255 (``display_image``), whatever
        # the camera's bit depth is.
        self.image_source = ColumnDataSource({"image": [], "dw": [], "dh": []})
        self.image_figure = figure(
            match_aspect=True,
            height=560,
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

        self.message = pn.pane.Str("", styles={"color": _WARNING})
        self.problems_text = pn.pane.Str("", styles={"color": _WARNING})
        self.status_text = pn.pane.Str("")
        self.run_text = pn.pane.Str("")
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
        self.plate_note = _wrapped_text(
            extra={"color": _WARNING}, width=_PLATE_NOTE_WIDTH
        )
        self.plate_confirm_button = pn.widgets.Button(label="", color="primary")
        self.label_select = pn.widgets.Select(
            label="Density of", options=[], value=None
        )
        self.drive_toggle = pn.widgets.Toggle(
            label="Drive (next click moves the stage)", value=False
        )
        self.side_select = pn.widgets.Select(
            label="Drive to (wall as on the image)",
            options=list(_SIDES),
            value="centre",
        )
        self.select_input = pn.widgets.TextInput(
            label="Wells", placeholder="A1:D6, C*, all"
        )
        self.refresh_button = pn.widgets.Button(label="Refresh")
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
        self.connection_text = pn.pane.Str("")

        # Every handler goes through `_handler` (the document lock, a failure is a line).
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

        self.tool_runs: dict[str, Any] = {}
        self.tool_fields: dict[str, dict[str, Any]] = {}
        self.tool_notes: dict[str, Any] = {}
        self.tool_cards: dict[str, Any] = {}
        self.layout = self._layout()
        self.refresh()
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
        )
        self.z_step_select = pn.widgets.Select(
            label="Z step",
            options=z_steps,
            value=_default_step(z_steps, _Z_DEFAULT_UM),
            width=110,
        )
        self.jog_buttons = {
            direction: pn.widgets.Button(label=label, width=50)
            for direction, label in (
                ("up", "↑"),
                ("left", "←"),
                ("right", "→"),
                ("down", "↓"),
            )
        }
        self.z_buttons = {
            "up": pn.widgets.Button(label="Z +", width=60),
            # A minus sign, as wide as the plus: a hyphen reads as a dash.
            "down": pn.widgets.Button(label="Z \N{MINUS SIGN}", width=60),
        }
        self.autofocus_on_button = pn.widgets.Button(label="Autofocus on")
        self.autofocus_off_button = pn.widgets.Button(label="Autofocus off")
        # Built once per session, while the stand may be closed: the tick reads the
        # labels again when the connection it sees changes (m5 §7.3).
        self._seen_connected = page.connected
        self._objective_labels = page.objective_labels()
        self.objective_select = pn.widgets.Select(
            label="Objective",
            options={label or NO_LABEL: label for label in self._objective_labels},
            width=220,
        )
        self.objective_button = pn.widgets.Button(label="Change objective")
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
        self.light_on_button = pn.widgets.Button(label="Light on")
        self.light_off_button = pn.widgets.Button(label="Light off")
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

    def _drive_region(self) -> Any:
        """ADR-0006's *Drive*: the pad, the focus, the autofocus, the objective, the light."""
        jog = self.jog_buttons
        pad = pn.Column(
            pn.Row(pn.Spacer(width=60), jog["up"]),
            pn.Row(jog["left"], jog["down"], jog["right"]),
        )
        return pn.Column(
            pn.pane.Str("Drive", styles={"font-weight": "bold"}),
            pn.Row(pad, self.xy_step_select),
            self.drive_image_note,
            pn.Row(self.z_buttons["up"], self.z_buttons["down"], self.z_step_select),
            pn.Row(self.autofocus_on_button, self.autofocus_off_button),
            pn.Row(self.objective_select, self.objective_button),
            self.objective_confirm,
            pn.Row(self.light_on_button, self.light_off_button),
            self.drive_note,
        )

    def _next_region(self) -> Any:
        """The head of the tools column: the tool the run document needs next, and one button for it (m5 §8.5)."""
        return pn.Column(
            pn.pane.Str("Next step", styles={"font-weight": "bold"}),
            self.next_text,
            self.next_button,
            self.next_note,
        )

    def _layout(self) -> Any:
        connection = pn.Row(
            self.connect_button,
            self.disconnect_button,
            self.connection_text,
            sizing_mode="stretch_width",
        )
        strip = pn.Row(
            self.status_text,
            pn.Spacer(sizing_mode="stretch_width"),
            self.resume_button,
            self.stop_button,
            sizing_mode="stretch_width",
        )
        controls = pn.Row(
            self.plate_select, self.label_select, self.side_select, self.drive_toggle
        )
        plate_tab = pn.Column(
            controls,
            pn.Row(self.plate_confirm_button, self.plate_note),
            pn.Row(self.select_input, self.refresh_button),
            pn.pane.Bokeh(self.figure, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        image_tab = pn.Column(
            pn.Row(self.snap_button, self.live_button, self.snap_note),
            pn.pane.Bokeh(self.image_figure, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        # Not dynamic: a dynamic tab destroys the figure's view when another tab is
        # shown, and bokehjs's DataRange1d keeps the bounds of every view it has had,
        # so the map was framed on the union of the plates it had drawn and neither
        # Reset nor Refresh took the old ones out (#297). Bokeh keeps the view of a
        # hidden tab, and its size, so no figure is laid out at zero size.
        self.tabs = pn.Tabs(
            ("Plate", plate_tab),
            ("Image", image_tab),
            dynamic=False,
            sizing_mode="stretch_width",
        )
        left = pn.Column(
            self.problems_text,
            self.tabs,
            self.run_document_text,
            self.checklist_text,
            sizing_mode="stretch_width",
        )
        stages: dict[str, list[Any]] = {}
        for tool in self._tools:
            stages.setdefault(tool.stage, []).append(self._tool_section(tool))
        # Where each tool's stage is in the accordion, so that the next step can open it.
        self._stage_index = {
            tool.name: list(stages).index(tool.stage) for tool in self._tools
        }
        self.accordion = pn.Accordion(
            *((stage, pn.Column(*sections)) for stage, sections in stages.items()),
            active=list(range(len(stages))),
        )
        right = pn.Column(
            self._preset_region(),
            self._next_region(),
            self._drive_region(),
            self.accordion,
            width=420,
        )
        return pn.Column(
            connection,
            strip,
            self.drive_text,
            self.run_text,
            self.message,
            pn.Row(left, right, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def _tool_section(self, tool: ToolView) -> Any:
        fields: dict[str, Any] = {}
        suggested = dict(tool.suggested)
        for name, type_name, default, notes in (*tool.key, *tool.advanced):
            description = f"{type_name}  {notes}".strip()
            if name in suggested:
                # The field starts with what the stand suggests, so that this tool's own
                # *Run* sends what the next step shows (m5 §8.3).
                description += f"\nsuggested by the profile: {tool.suggested_why}"
            fields[name] = pn.widgets.TextInput(
                label=name,
                placeholder=default,
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
            width=380,
            stylesheets=["pre { white-space: pre-wrap; }"],
        )
        self.tool_notes[tool.name] = note
        # Plain text, never Markdown: a result's words are not markup (FM-139). It wraps:
        # a file's path is as long as the folder it is in (m5 §8.4).
        card = _wrapped_text()
        self.tool_cards[tool.name] = card
        parts: list[Any] = [pn.pane.Str(tool.summary), *key_widgets]
        if tool.name == "plate-registration":
            parts += [
                pn.Row(self.capture_centre_button, self.forget_button),
                pn.Row(self.capture_wall_button, self.capture_side_select),
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

    def on_choose_plate(self, name: str) -> None:
        """Another plate: the selection and the captures of the old one are dropped."""
        if name == self.page.plate:
            return
        self._try(lambda: self.page.choose_plate(name))
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
        """Snap one frame and show the *Image* tab; the tick that first sees the frame draws it.

        A refusal (an activity is going) is the page's message, as for a run.
        """
        self._try(self.page.snap)
        self.tabs.active = _IMAGE_TAB
        self.tick()

    def on_live(self) -> None:
        """Start live, or end it, and show the *Image* tab when it starts.

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
        if starting:
            self.tabs.active = _IMAGE_TAB
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
        self.status_text.object = status_line(status)
        self.run_text.object = run_text(status)
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
        self._draw_frame()
        self._draw_buttons()
        self._draw_drive()
        if previous is not None and status.running is None:
            self.refresh()
        elif self._shared_state() != self._seen_state:
            # Another tab chose wells or a plate, or took a capture: the controller is
            # the server's, so a tab that does not redraw would run `scan-plan` with
            # wells it does not show. Reading five attributes is no view: the view is
            # rebuilt only when one of them changed.
            self.refresh()

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
        """The *Connect* row: what the stand's state is, and which of the two buttons applies.

        Both are disabled while an activity goes (live included), so that
        Disconnect is never offered under a run; the controller refuses a click
        all the same (FM-169). While ``connect`` or ``disconnect`` is the activity
        the row says so: on a slow stand it would otherwise read ``Connect opens it``
        beside a greyed-out button, and the page would look stuck (m5 §7.3, #305).
        """
        busy = status.running is not None
        self.connect_button.disabled = busy or status.connected
        self.disconnect_button.disabled = busy or not status.connected
        if status.running == "connect":
            text = f"connecting {status.stand}…"
        elif status.running == "disconnect":
            text = f"disconnecting {status.stand}…"
        elif status.connected:
            text = (
                f"Disconnect releases {status.stand} for another program "
                f"(NIS-Elements, ZEN)"
            )
        else:
            text = (
                f"{status.stand} is not connected: Connect opens it, and the "
                f"controls that need the stand are disabled until then"
            )
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
        frame = self._draw_frame()
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
        self.stage_source.data = {
            "x": [] if xy_um is None else [xy_um[0]],
            "y": [] if xy_um is None else [xy_um[1]],
            "text": [] if xy_um is None else [label],
        }

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
            page.selection,
            page.refs,
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
        if self.plate_select.value != view.plate:
            self.plate_select.value = view.plate
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
        self._draw_problems()
        self._draw_plate_ask()
        self._draw_buttons()
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
        self.names.visible = len(view.wells) <= NAMES_UP_TO_WELLS
        # Where the wells are drawn is only as good as the registration: say which map it is
        # (registered, seed, estimated or unplaced, m5 §15 D4). The scan plan's suffix is
        # where a plan too large to draw says so (m5 §14.3).
        self.figure.title.text = map_title(view.plate, view.pose) + plan_title(
            view.plan
        )
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
        lacks = self.page.lacks(Camera)
        self.snap_button.disabled = running is not None or lacks is not None
        # While live goes its own button is the one that ends it; any other activity
        # leaves it nothing to do.
        self.live_button.label = _END_LIVE_LABEL if live else _LIVE_LABEL
        self.live_button.disabled = lacks is not None or (
            running is not None and not live
        )
        self.snap_note.object = lacks or ""

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
