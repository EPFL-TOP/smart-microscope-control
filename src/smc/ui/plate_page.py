"""The plate page: Panel widgets over ``PlatePage`` (m3 §15, m5 §3, ADR-0006).

The only module besides the ``smc ui`` command that imports Panel and
Bokeh. It holds no decision: a tap, a button and a typed range each call the
controller, and an exception the controller raises for the operator's mistake is
shown as its message and kept as nothing else (FM-98). Three rules the page
keeps, each learned by measuring (m3 §15.3):

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
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
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

from smc.hardware.capabilities import Camera
from smc.ui.controller import (
    PageServerError,
    PageStatus,
    PlatePage,
    PlateView,
    ResultCard,
    ToolView,
    WellView,
)
from smc.ui.frames import Frame, display_image

__all__ = ["PlatePageView", "build", "card_text", "serve"]

#: Names are drawn on a plate of at most this many wells; on more they are noise.
NAMES_UP_TO_WELLS = 96

#: How often the page reads the stand and the run, in milliseconds. A scan must not
#: be slowed by a status poll, and the cost of ``Microscope.state()`` on a real
#: stand is not measured (m3 §15.3): a click or a button acts at once through its
#: own handler.
TICK_MS = 1000

#: How often the page looks for a new frame, in milliseconds. It matches
#: ``LIVE_PERIOD_S``: a faster callback would find no new frame, a slower one would
#: drop frames the camera took. It reads ``PlatePage.frame()`` and nothing else.
FRAME_TICK_MS = 100

#: How long ``serve`` waits for Panel's server to stop, in seconds.
SERVER_STOP_WAIT_S = 10.0

DRIVE_TEXT = "DRIVE: the next click on a well moves the stage there"

#: The centre's tabs, in the order they are built.
_PLATE_TAB = 0
_IMAGE_TAB = 1

NO_FRAME_TEXT = "no frame yet"

_WARNING = "#b45309"
_SELECTED_OUTLINE = "#1f77b4"
_PLAIN_OUTLINE = "#888888"
_UNKNOWN_FILL = "#eeeeee"
_SIDES = ("centre", "left", "right", "top", "bottom")
_WALL_SIDES = ("left", "right", "top", "bottom")


def _um(value: float, digits: int) -> str:
    """A position in µm; the demo stage's ``-0.0`` is printed as ``0.0``, as the CLI does."""
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def status_line(status: PageStatus) -> str:
    """The stand on one line: ``<stand> | XY (x, y) µm | Z z µm | objective | autofocus | light | ...``.

    The stand's name comes first, where ADR-0006 puts it. ``?`` is an unknown,
    never a zero; ``DRY RUN``, ``HALTED`` and ``moving`` are added only when they
    apply, and so is what STOP does not stop (FM-117).
    """
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
    then one ``assumed:`` line for each assumption and the record's error.
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
    lines += [f"assumed: {text}" for text in card.assumed]
    if card.error is not None:
        lines.append(f"error: {card.error}")
    return "\n".join(lines)


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
        self._tools = page.tools()

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
        self.stage_source = ColumnDataSource({"x": [], "y": []})
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
        self.figure.on_event(Tap, self.on_tap)

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

        self.plate_select = pn.widgets.Select(
            label="Plate",
            options=sorted(useq.registered_well_plate_keys()),
            value=page.plate,
        )
        self.label_select = pn.widgets.Select(
            label="Density of", options=[], value=None
        )
        self.drive_toggle = pn.widgets.Toggle(
            label="Drive (next click moves the stage)", value=False
        )
        self.side_select = pn.widgets.Select(
            label="Drive to", options=list(_SIDES), value="centre"
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
            label="Wall", options=list(_WALL_SIDES), value="left"
        )
        self.forget_button = pn.widgets.Button(label="Forget")
        self.snap_button = pn.widgets.Button(label="Snap")
        self.live_button = pn.widgets.Button(label="Live")
        self.snap_note = pn.pane.Str("", styles={"color": _WARNING})

        self.plate_select.param.watch(
            lambda event: self.on_choose_plate(event.new), "value"
        )
        self.label_select.param.watch(lambda event: self._draw_wells(), "value")
        self.drive_toggle.param.watch(
            lambda event: setattr(self.drive_text, "visible", bool(event.new)), "value"
        )
        self.select_input.param.watch(
            lambda event: self.on_select_text(event.new), "value"
        )
        self.refresh_button.on_click(lambda event: self.refresh())
        self.stop_button.on_click(lambda event: self.on_stop())
        self.resume_button.on_click(lambda event: self.on_resume())
        self.capture_centre_button.on_click(lambda event: self.on_capture("centre"))
        self.capture_wall_button.on_click(lambda event: self.on_capture("wall"))
        self.forget_button.on_click(lambda event: self.on_forget())
        self.snap_button.on_click(lambda event: self.on_snap())
        self.live_button.on_click(lambda event: self.on_live())

        self.tool_runs: dict[str, Any] = {}
        self.tool_fields: dict[str, dict[str, Any]] = {}
        self.tool_notes: dict[str, Any] = {}
        self.tool_cards: dict[str, Any] = {}
        self.layout = self._layout()
        self.refresh()
        self.tick()

    # --- layout ---------------------------------------------------------------

    def _layout(self) -> Any:
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
            pn.Row(self.select_input, self.refresh_button),
            pn.pane.Bokeh(self.figure, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        image_tab = pn.Column(
            pn.Row(self.snap_button, self.live_button, self.snap_note),
            pn.pane.Bokeh(self.image_figure, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        # Only the active tab is rendered, so a figure is never laid out hidden.
        self.tabs = pn.Tabs(
            ("Plate", plate_tab),
            ("Image", image_tab),
            dynamic=True,
            sizing_mode="stretch_width",
        )
        left = pn.Column(
            self.problems_text,
            self.tabs,
            pn.pane.Str("Run document", styles={"font-weight": "bold"}),
            self.checklist_text,
            sizing_mode="stretch_width",
        )
        stages: dict[str, list[Any]] = {}
        for tool in self._tools:
            stages.setdefault(tool.stage, []).append(self._tool_section(tool))
        right = pn.Column(
            pn.Accordion(
                *((stage, pn.Column(*sections)) for stage, sections in stages.items()),
                active=list(range(len(stages))),
            ),
            width=420,
        )
        return pn.Column(
            strip,
            self.drive_text,
            self.run_text,
            self.message,
            pn.Row(left, right, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def _tool_section(self, tool: ToolView) -> Any:
        fields: dict[str, Any] = {}
        for name, type_name, default, notes in (*tool.key, *tool.advanced):
            fields[name] = pn.widgets.TextInput(
                label=name,
                placeholder=default,
                description=f"{type_name}  {notes}".strip(),
            )
        self.tool_fields[tool.name] = fields
        key_widgets = [fields[row[0]] for row in tool.key]
        advanced_widgets = [fields[row[0]] for row in tool.advanced]
        run = pn.widgets.Button(label=f"Run {tool.name}")
        run.on_click(self._runner(tool.name))
        self.tool_runs[tool.name] = run
        note = pn.pane.Str("", styles={"color": _WARNING})
        self.tool_notes[tool.name] = note
        # Plain text, never Markdown: a result's words are not markup (FM-139).
        card = pn.pane.Str("")
        self.tool_cards[tool.name] = card
        parts: list[Any] = [pn.pane.Str(tool.summary), *key_widgets]
        if tool.name == "plate-registration":
            parts += [
                pn.Row(self.capture_centre_button, self.forget_button),
                pn.Row(self.capture_wall_button, self.capture_side_select),
                self.captures_text,
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

    def _try(self, action: Callable[[], object]) -> bool:
        """Run ``action``; an operator's mistake becomes the message, and nothing else is kept."""
        try:
            action()
        except (ValueError, RuntimeError) as exc:
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

    def on_select_text(self, text: str) -> None:
        """A typed range (``A1:D6``): the selection becomes exactly those wells."""
        if text.strip() == "":
            return
        self._try(lambda: self.page.select_text(text))
        self.refresh()

    def on_choose_plate(self, name: str) -> None:
        """Another plate: the selection and the captures of the old one are dropped."""
        if name == self.page.plate:
            return
        self._try(lambda: self.page.choose_plate(name))
        self.refresh()

    def on_capture(self, kind: str) -> None:
        """Capture the stage position as the centre, or a wall, of the current well."""

        def capture() -> None:
            well = self.page.current
            if well is None:
                raise ValueError("click a well first")
            if kind == "centre":
                self.page.capture_centre(well)
            elif kind == "wall":
                self.page.capture_wall(well, str(self.capture_side_select.value))
            else:
                raise ValueError(f"cannot capture {kind!r}; capture a centre or a wall")

        self._try(capture)
        self.refresh()

    def on_forget(self) -> None:
        """Forget the captures of the current well."""

        def forget() -> None:
            well = self.page.current
            if well is None:
                raise ValueError("click a well first")
            self.page.forget(well)

        self._try(forget)
        self.refresh()

    def on_run(self, plugin: str) -> None:
        """Run a tool with what the page fills and the fields the operator typed."""

        def run() -> None:
            typed = [
                f"{key}={widget.value.strip()}"
                for key, widget in self.tool_fields.get(plugin, {}).items()
                if widget.value.strip() != ""
            ]
            self.page.run(plugin, [*self.page.page_params(plugin), *typed])

        self._try(run)
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

        Which of the two is read from the controller at the click (``is_live``, m5
        §5.3), not from the label, which the last tick drew: a click in the second
        before the label follows the state does what the state says now. Ending is
        not STOP: the stand is not halted. A refusal is the page's message, as for
        a run.
        """
        starting = not self.page.is_live
        self._try(lambda: self.page.live(starting))
        if starting:
            self.tabs.active = _IMAGE_TAB
        self.tick()

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
        self.status_text.object = status_line(status)
        self.run_text.object = run_text(status)
        self.resume_button.visible = status.halted
        self.stop_button.disabled = False
        self.stage_source.data = {
            "x": [] if status.xy_um is None else [status.xy_um[0]],
            "y": [] if status.xy_um is None else [status.xy_um[1]],
        }
        self._problems_status = status.problems
        self._draw_problems()
        self._draw_frame()
        self._draw_buttons()
        if previous is not None and status.running is None:
            self.refresh()
        elif self._shared_state() != self._seen_state:
            # Another tab chose wells or a plate, or took a capture: the controller is
            # the server's, so a tab that does not redraw would run `scan-plan` with
            # wells it does not show. Reading five attributes is no view: the view is
            # rebuilt only when one of them changed.
            self.refresh()

    def frame_tick(self) -> None:
        """The frame callback's body: draw the controller's frame when it is new, and nothing else.

        It reads no status and no view (FM-145): it runs ten times a second, and the
        stand's read and the run document's belong to ``tick``.
        """
        self._draw_frame()

    def _shared_state(self) -> tuple[Any, ...]:
        """What the controller holds that every tab draws: compared, never rebuilt.

        ``activities_ended`` is there for the tab that did not start a tool: one that
        ends between two of its ticks is never seen running, and its result card,
        the checklist and the blocked tools are new all the same (FM-151).
        """
        page = self.page
        return (
            page.plate,
            page.selection,
            page.refs,
            page.walls,
            page.current,
            page.activities_ended,
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
        self.route_source.data = {
            "x": [stop.x_um for stop in view.route],
            "y": [stop.y_um for stop in view.route],
            "order": [stop.order for stop in view.route],
            "well": [stop.well for stop in view.route],
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
        self._draw_cards()
        self._draw_problems()
        self._draw_buttons()

    def _draw_cards(self) -> None:
        """The result card under each tool, from the view: a result follows the run document, not the tick."""
        cards = {} if self._view is None else self._view.cards
        for name, pane in self.tool_cards.items():
            pane.object = card_text(cards.get(name), stand=self._stand)

    def _draw_frame(self) -> None:
        """Draw the controller's frame when it is one this session has not drawn.

        The axes are the frame's own pixels: ``dw`` and ``dh`` are its width and
        height, not those of the strided picture (FM-136). Bokeh draws an image's
        first row at the bottom, so the rows are flipped to put the camera's top
        row on top.
        """
        frame = self.page.frame()
        if frame is None or frame.number == self._seen_frame:
            return
        self._seen_frame = frame.number
        self._draw_image(frame)

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
        # Where the wells are drawn is only as good as the registration: say which map it is.
        self.figure.title.text = (
            f"{view.plate}: nominal map (A1 at 0, 0, not turned), the plate is not registered"
            if view.nominal
            else f"{view.plate}: the registered plate"
        )

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
        """A tool that is blocked, or any tool or snap while an activity goes, cannot be started."""
        running = self._running
        for name, button in self.tool_runs.items():
            reason = self._blocked.get(name)
            button.disabled = running is not None or reason is not None
            self.tool_notes[name].object = reason or ""
        lacks = self.page.lacks(Camera)
        self.snap_button.disabled = running is not None or lacks is not None
        # While live goes its own button is the one that ends it; any other activity
        # leaves it nothing to do.
        live = self.page.is_live
        self.live_button.label = "End live" if live else "Live"
        self.live_button.disabled = lacks is not None or (
            running is not None and not live
        )
        self.snap_note.object = lacks or ""


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
            if thread.is_alive() and not _stop_server(thread):
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
