"""What the plate page shows and does, with no Panel in sight (m3 §15, ADR-0006).

The page is a view over plugins and capabilities, so everything it decides is
here and testable on the demo devices: which well a click lands on, what a stage
of the workflow can run now and why not, what a stop did. ``plate_page.py`` only
turns these values into widgets.

The page runs a plugin exactly as ``smc run`` does (``load``, ``parse_params``,
``check_ready``, ``run_plugin``), in one daemon worker thread, so every button
has its command-line equivalent. Three rules shape the code:

* **The stop path takes no lock** (FM-70). ``stop()`` reads one attribute, sets
  an event and calls ``Microscope.stop()``; the lock guards the few attributes a
  worker and a status poll both write and is never held across a device call.
* **No exception is kept** (FM-98). A run's failure becomes one line of text in
  ``outcome`` before the worker ends, so no traceback keeps the facade, and so
  the core, alive after the page is closed.
* **A view's life is not the work's.** The controller belongs to the server: a
  closed or reloaded browser tab never stops a scan; only STOP, the end of the
  server (``close()``) and the plugin's own end do.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
import useq
from pydantic import ValidationError

from smc.hardware.capabilities import Autofocus, Camera, LightSource, XYStage
from smc.hardware.errors import HardwareError, StopRefusedError
from smc.plugins import (
    ParamsError,
    PluginEntry,
    PluginError,
    PluginParams,
    Progress,
    Registry,
    check_ready,
    describe_params,
    discover,
    load,
    parse_params,
    run_plugin,
)
from smc.plugins.scan_plan.wells import parse_wells
from smc.plugins.slots import (
    DensityResult,
    PlateResult,
    RouteResult,
    WallTouch,
    WellGeometry,
    WellRef,
    normalise_well,
    well_geometry,
)
from smc.run import RunDocument, RunDocumentError, StageRecord, StageStatus
from smc.ui.frames import Frame

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "KEY_PARAMS",
    "LIVE_PERIOD_S",
    "MORE_STAGE",
    "PAGE_PARAMS",
    "STAGES",
    "CheckItem",
    "PageServerError",
    "PageStatus",
    "PlatePage",
    "PlateView",
    "ResultCard",
    "RouteStop",
    "ToolView",
    "WellView",
]

log = logging.getLogger("smc.ui")

#: The tools of the workflow in the order the page shows them, grouped by stage.
#: Every other registered plugin is on the page too, under ``MORE_STAGE``, by name
#: (m5 §2.3): a plugin nobody listed here is usable all the same.
STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Set up", ("camera-calibration", "plate-registration", "plate-finding", "timing")),
    ("Choose", ("scan-plan",)),
    ("Acquire", ("scan",)),
    ("Find", ("cells", "density", "route")),
    ("Report", ("report", "nis-export")),
)

#: The stage of a tool that ``STAGES`` does not name: the literal name the page shows.
MORE_STAGE = "More"

#: The parameters worth a field of their own, in the order the page shows them;
#: every other parameter is under the tool's *Advanced* fold, except a required one,
#: which is never folded.
KEY_PARAMS: dict[str, tuple[str, ...]] = {
    "camera-calibration": ("step_um",),
    "scan-plan": ("objective", "overlap", "coverage"),
    "scan": ("autofocus",),
    "cells": ("detector", "score_threshold"),  # detector: #260
    "route": ("labels", "fields_per_well"),
}

#: The parameters the page fills itself (``PlatePage.page_params``), so a form
#: does not offer them.
PAGE_PARAMS: dict[str, tuple[str, ...]] = {
    "plate-registration": ("plate", "refs", "walls"),
    "scan-plan": ("wells",),
}

#: The checklist, in the order of the stages of m3 §2.2; the second item says
#: whether a workflow can do without the slot.
_CHECKLIST: tuple[tuple[str, bool], ...] = (
    ("camera-calibration", True),
    ("plate", False),
    ("timings", True),
    ("scan-plan", False),
    ("scan", False),
    ("cells", False),
    ("density", False),
    ("route", False),
    ("report", False),
    ("nis-export", False),
)

_SIDES = ("left", "right", "top", "bottom")

#: A result card shows this many values, each at most this many characters long.
_CARD_LINES = 8
_CARD_CHARS = 80

#: An assumption and an error are diagnoses whose end often says what to do: they are
#: cut later than a value (more than twice the longest assumption a plugin writes).
_CARD_TEXT_CHARS = 600

#: A card's time when its record's date cannot be put in UTC: unknown, never a guess.
_UNKNOWN_TIME = "????-??-?? ??:??:??"

#: How long ``close`` waits for a stopped run to end, in seconds.
_CLOSE_WAIT_S = 10.0

#: A wait on the worker is made of slices this long (FM-83: Windows does not deliver
#: Ctrl-C during one long wait on a thread).
_WAIT_SLICE_S = 0.1

#: A live view takes a frame at most once in this many seconds: the cap is a period,
#: so a camera slower than 10 frames a second runs at its own rate (m5 §5.1).
LIVE_PERIOD_S = 0.1

#: What ``stop`` says when the stand took the stop: the one wording that may be told
#: as "stopped" (FM-117).
_STOP_ACCEPTED = "stopped: the stand is halted (Resume clears the halt)"

#: Wells built for a plate are kept (339 ms for 1536 of them); a registration
#: changes the key rarely, so a handful is plenty.
_GEOMETRY_KEPT = 4

_Capture = TypeVar("_Capture", WellRef, WallTouch)


class PageServerError(RuntimeError):
    """The page's server ended without the operator stopping it."""


@dataclass(frozen=True, slots=True)
class WellView:
    """One well as the map draws it, in stage µm; ``None`` is unknown, never zero."""

    well: str
    center_um: tuple[float, float]
    size_um: tuple[float, float]
    circular: bool
    rotation_deg: float
    selected: bool
    counts: dict[str, int] | None
    per_mm2: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class RouteStop:
    """One stop of the route, in stage µm."""

    order: int
    well: str
    x_um: float
    y_um: float


@dataclass(frozen=True, slots=True)
class CheckItem:
    """One slot of the run document: ``reason`` is ``RunDocument.unusable``'s words for it."""

    slot: str
    optional: bool
    state: Literal["ok", "not run", "unusable"]
    reason: str | None


@dataclass(frozen=True, slots=True)
class ResultCard:
    """What a tool's last record says, as lines of text: never markup (FM-139).

    ``lines`` are the result's values and are empty unless the record ended
    ``done``: a stopped or failed record's values are not a result (FM-154).
    ``assumed`` are the result's own ``assumed`` lines, under the same rule.
    ``stale`` marks a ``done`` record that read an input produced again since.
    Every text is bounded, whatever the record holds: the card is a summary.
    """

    plugin: str
    status: str
    when_utc: str
    profile: str
    dry_run: bool
    lines: tuple[str, ...]
    assumed: tuple[str, ...]
    error: str | None
    stale: bool = False


@dataclass(frozen=True, slots=True)
class PlateView:
    """A snapshot of what the run document says, as the page needs it (a pure read).

    ``nominal`` is true when the wells are those of the plate type with A1 at
    (0, 0) and no rotation, not a registered plate. ``blocked`` maps a registered
    tool to why it cannot run now. ``problems`` are what ``view()`` could not read
    and did not hide. ``cards`` has one card for each tool that has a record in
    the run document, and none for the others.
    """

    plate: str
    nominal: bool
    wells: tuple[WellView, ...]
    labels: tuple[str, ...]
    route: tuple[RouteStop, ...]
    checklist: tuple[CheckItem, ...]
    blocked: dict[str, str]
    problems: tuple[str, ...]
    cards: dict[str, ResultCard]


@dataclass(frozen=True, slots=True)
class ToolView:
    """A tool the page can run: its rows are ``describe_params`` rows (name, type, default, notes)."""

    name: str
    summary: str
    stage: str
    key: tuple[tuple[str, str, str, str], ...]
    advanced: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True, slots=True)
class PageStatus:
    """The stand and the run, read now.

    ``stand`` is the profile's ``microscope.name``. ``xy_um`` to ``halted`` are
    ``Microscope.state()``'s; ``stop_refused`` are the roles the profile declares
    refuse their stop, so that the page can say that STOP does not stop a move in
    flight on this stand (FM-117).
    """

    stand: str
    dry_run: bool
    running: str | None
    progress: Progress | None
    outcome: str | None
    stop_note: str | None
    xy_um: tuple[float, float] | None
    z_um: float | None
    objective: str | None
    autofocus: str
    light_on: bool | None
    moving: tuple[str, ...]
    halted: bool
    stop_refused: tuple[str, ...]
    problems: tuple[str, ...]


def _one_line(error: BaseException) -> str:
    """The message of ``error`` on one line; an error whose message cannot be built says nothing."""
    try:
        return " ".join(str(error).split())
    except (
        Exception
    ):  # a message that cannot be printed is not worth losing the outcome
        return ""


def _autofocus_words(engaged: bool | None, locked: bool | None) -> str:
    """The words of ``Microscope.describe``: ``?`` whenever a read the answer needs is unknown."""
    if engaged is None:
        return "autofocus ?"
    if not engaged:
        return "autofocus off"
    if locked is None:
        return "autofocus ?"
    return "autofocus locked" if locked else "autofocus on, not locked"


def _flat(text: str) -> str:
    """``text`` on one line: its line breaks are spaces."""
    return " ".join(text.splitlines())


def _cut(text: str, limit: int = _CARD_CHARS) -> str:
    """``text`` of at most ``limit`` characters, and a ``…`` when it was longer."""
    return text if len(text) <= limit else text[:limit] + "…"


def _value_line(key: str, value: object) -> str:
    """One ``<key>: <value>`` line of a result card; a list or an object is only counted.

    A value is shown by what it is, never by a ``repr`` that could be as long as the
    result: the card is a summary, and the run document holds the rest.
    """
    label = _cut(_flat(key))
    if isinstance(value, str):
        return f"{label}: {_cut(_flat(value))}"
    if isinstance(value, bool | int) or value is None:
        return f"{label}: {value}"
    if isinstance(value, float):
        return f"{label}: {value:g}"
    if isinstance(value, list):
        return f"{label}: {len(value)} items"
    if isinstance(value, dict):
        return f"{label}: {len(value)} entries"
    return f"{label}: {type(value).__name__}"


def _card(record: StageRecord, *, stale: bool = False) -> ResultCard:
    """The card of ``record``; its values only when it ended ``done`` (FM-154).

    ``stale`` is ``RunDocument.is_stale`` of the record, and marks only a ``done``
    one: a record that did not end done has no values to mislead, and its status
    already says to run it again. The stale card keeps its values (they are a result,
    of inputs that are no longer the document's).
    """
    when = record.finished_utc or record.started_utc
    try:
        when_utc = when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError):
        # A date a model accepts (year 1 with an offset) that UTC cannot hold: a
        # hand-edited document must not stop ``view()``, which raises nothing for one.
        when_utc = _UNKNOWN_TIME
    lines: list[str] = []
    assumed: tuple[str, ...] = ()
    if record.status is StageStatus.done and record.result is not None:
        lines = [
            _value_line(key, value)
            for key, value in record.result.items()
            if key != "assumed"
        ]
        if len(lines) > _CARD_LINES:
            lines = [*lines[:_CARD_LINES], f"… {len(lines) - _CARD_LINES} more"]
        found = record.result.get("assumed")
        if isinstance(found, list):
            items = [
                _cut(_flat(str(item)), _CARD_TEXT_CHARS) for item in found[:_CARD_LINES]
            ]
            if len(found) > _CARD_LINES:
                items.append(f"… {len(found) - _CARD_LINES} more")
            assumed = tuple(items)
        elif found is not None:
            assumed = (_cut(_flat(str(found)), _CARD_TEXT_CHARS),)
    return ResultCard(
        plugin=record.plugin,
        status=record.status.value,
        when_utc=when_utc,
        profile=_cut(_flat(record.profile)),
        dry_run=record.dry_run,
        lines=tuple(lines),
        assumed=assumed,
        error=(
            None
            if record.error is None
            else _cut(_flat(record.error), _CARD_TEXT_CHARS)
        ),
        stale=stale and record.status is StageStatus.done,
    )


def _first_problem(error: ValidationError) -> str:
    """``<where>: <what>`` of the first validation error."""
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "result"
    return f"{where}: {first['msg']}"


class PlatePage:
    """The plate page's logic over one stand and one run document.

    The controller never opens or closes the stand: the command that serves the
    page does, once, for the page's life. Everything an operator does here is a
    plugin run with the strings ``smc run`` takes, or a read of a capability.

    Attributes:
        selection: The wells the operator chose, in the order they chose them.
        refs: The wells centred under the objective, for plate registration.
        walls: The wall touches, for plate registration.
        current: The well last clicked or driven to; where *Capture* and
            *Forget* act.
        close_lines: What ``close()`` has to tell the operator about the run it
            stopped, as lines: ``<plugin> was stopped`` when the stand took the
            stop and the run ended stopped, else what the stop did and how the run
            ended. ``()`` when no run was going: what ``smc ui`` says last.
        close_ended: ``False`` when the run it stopped was still alive after the
            wait.
    """

    selection: tuple[str, ...]
    refs: tuple[WellRef, ...]
    walls: tuple[WallTouch, ...]
    current: str | None
    close_lines: tuple[str, ...]
    close_ended: bool

    def __init__(
        self,
        microscope: Microscope,
        run_doc: Path,
        *,
        entry_points: Iterable[EntryPoint] | None = None,
    ) -> None:
        """Start from the run document: its plate, the wells chosen and the references taken.

        Never raises for a document: one that cannot be read starts the page
        empty, and ``view()`` says why in its ``problems``.
        """
        self._microscope = microscope
        self._run_doc = run_doc
        # A list, so that a generator handed in is not spent by the first use.
        self._entry_points = None if entry_points is None else list(entry_points)
        self._registry_found: Registry | None = None
        self._tools_found: tuple[tuple[ToolView, ...], tuple[str, ...]] | None = None
        self._geometry: dict[
            tuple[str, tuple[float, float], float], tuple[WellGeometry, ...]
        ] = {}
        self._names: dict[str, frozenset[str]] = {}
        # What a worker and a status poll both write, under ``_lock``; never held
        # across a device call, and ``stop()`` never takes it (FM-70).
        self._lock = threading.Lock()
        self._current_run: tuple[str, threading.Event] | None = None
        self._progress: Progress | None = None
        self._outcome: str | None = None
        self._frame: Frame | None = None
        self._queue: queue.SimpleQueue[Progress] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None
        # The event of the live this page started, to tell it from any other activity
        # by identity. Written on the server's loop only; read without a lock (FM-70).
        self._live_event: threading.Event | None = None
        self._closed = False
        self._ended = 0
        # A plain attribute, written by ``stop()`` and ``resume()``, which take no lock.
        self._stop_note: str | None = None
        self.current = None
        self.close_lines = ()
        self.close_ended = True
        self.selection = ()
        self.refs = ()
        self.walls = ()
        doc, _unreadable = self._document()
        self._plate = self._initial_plate(doc)
        self._restore(doc)

    # --- plate and selection -------------------------------------------------

    @property
    def activities_ended(self) -> int:
        """How many activities (runs, snaps and live sessions) have ended since the page began.

        What every session compares to see that something ended while it was not
        looking: a tool that takes less than a tick is never seen running by a
        second tab, and its record, its result card and the checklist are new all
        the same (FM-151).
        """
        return self._ended

    @property
    def plate(self) -> str:
        """The plate the page is about, a name of ``useq``'s registry."""
        return self._plate

    def choose_plate(self, name: str) -> None:
        """Change the plate; the selection, the captures and ``current`` belong to the old one.

        Raises:
            ValueError: ``name`` is not a plate of ``useq``'s registry; the
                message lists them.
        """
        plates = sorted(useq.registered_well_plate_keys())
        if name not in plates:
            raise ValueError(
                f"{name!r} is not a plate; the plates are {', '.join(plates)}"
            )
        if name == self._plate:
            return
        self._plate = name
        self.selection = ()
        self.refs = ()
        self.walls = ()
        self.current = None

    def select_text(self, text: str) -> None:
        """Replace the selection by the wells ``text`` names (``A1:D6``, ``C*``, ``all``).

        Raises:
            ValueError: ``parse_wells``' message; the selection is as it was.
        """
        self.selection = parse_wells(self._plate, text)

    def toggle(self, well: str) -> None:
        """Add ``well`` to the selection, or remove it.

        Raises:
            ValueError: The well is not on the plate; the selection is as it was.
        """
        name = self._on_plate(well)
        if name in self.selection:
            self.selection = tuple(w for w in self.selection if w != name)
        else:
            self.selection = (*self.selection, name)

    def click(
        self, x_um: float, y_um: float, *, drive: bool, side: str = "centre"
    ) -> str | None:
        """The well under a point of the map: selected, or (``drive``) driven to.

        A point that is on no well, or is not finite, does nothing and returns
        ``None``. In select mode nothing moves, whatever the map shows.

        Raises:
            ValueError: ``drive`` on a map of another plate than the registered
                one (see ``goto``).
            RuntimeError: ``drive`` while a run is going, or after ``close``.
        """
        geometries, _nominal = self._geometries(self._document()[0])
        for geometry in geometries:
            if geometry.contains(x_um, y_um):
                well = geometry.well
                break
        else:
            return None
        if drive:
            self.goto(well, side)
        else:
            self.current = well
            self.toggle(well)
        return well

    def goto(self, well: str, side: str = "centre") -> None:
        """Run ``goto-well`` for ``well``.

        ``goto-well`` takes a well's name and reads the registered plate, while the
        map of a plate chosen with ``choose_plate`` is the nominal one when the
        registered plate is another: a click on C5 of the nominal 96-well map would
        drive the stage to C5 of the registered 384-well plate with nothing on the
        page to say so. So it refuses that mismatch before anything changes.

        Raises:
            ValueError: A usable ``plate`` slot names another plate than ``plate``.
            RuntimeError: A run is going, or the page is closed.
        """
        self._refuse_if_busy()
        self._refuse_another_registered_plate(self._document()[0])
        name = normalise_well(well)
        self.current = name
        self.run("goto-well", [f"well={name}", f"side={side}"])

    # --- captures --------------------------------------------------------------

    def capture_centre(self, well: str) -> WellRef:
        """Record the stage position as the centre of ``well``; a second capture replaces the first.

        Raises:
            ValueError: The well is not on the plate.
            CapabilityMissingError: The stand has no XY stage.
        """
        name = self._on_plate(well)
        position = self._microscope.require(XYStage).position_um()
        ref = WellRef(well=name, x_um=position.x_um, y_um=position.y_um)
        kept = tuple(r for r in self.refs if normalise_well(r.well) != name)
        self.refs = (*kept, ref)
        return ref

    def capture_wall(self, well: str, side: str) -> WallTouch:
        """Record the stage position as a point of ``well``'s ``side`` wall; replaces the same side.

        Raises:
            ValueError: ``side`` is not left, right, top or bottom, or the well is
                not on the plate.
            CapabilityMissingError: The stand has no XY stage.
        """
        if side not in _SIDES:
            raise ValueError(f"side must be one of {', '.join(_SIDES)}, got {side!r}")
        name = self._on_plate(well)
        position = self._microscope.require(XYStage).position_um()
        touch = WallTouch(
            well=name,
            side=side,  # type: ignore[arg-type]  # checked against _SIDES above
            x_um=position.x_um,
            y_um=position.y_um,
        )
        kept = tuple(
            w
            for w in self.walls
            if not (normalise_well(w.well) == name and w.side == side)
        )
        self.walls = (*kept, touch)
        return touch

    def forget(self, well: str | None = None) -> None:
        """Drop the captures of ``well`` (its centre and its walls), or all of them."""
        if well is None:
            self.refs = ()
            self.walls = ()
            return
        name = normalise_well(well)
        self.refs = tuple(r for r in self.refs if normalise_well(r.well) != name)
        self.walls = tuple(w for w in self.walls if normalise_well(w.well) != name)

    def page_params(self, plugin: str) -> list[str]:
        """The ``key=value`` pairs the page fills itself for ``plugin`` (``PAGE_PARAMS``).

        Raises:
            ValueError: ``scan-plan`` with nothing selected, or with a selection
                on another plate than the registered one.
        """
        if plugin == "scan-plan":
            if not self.selection:
                raise ValueError("select at least one well first")
            self._refuse_another_registered_plate(self._document()[0])
            return [f"wells={','.join(self.selection)}"]
        if plugin == "plate-registration":
            refs = json.dumps([r.model_dump(mode="json") for r in self.refs])
            walls = json.dumps([w.model_dump(mode="json") for w in self.walls])
            return [f"plate={self._plate}", f"refs={refs}", f"walls={walls}"]
        return []

    # --- runs, stop, status ---------------------------------------------------------

    def run(self, plugin: str, params: Sequence[str]) -> None:
        """Run ``plugin`` with ``key=value`` strings, as ``smc run`` would, in one worker thread.

        What ``smc run`` checks before it opens the stand is checked here before a
        thread starts: the plugin, its parameters, the run document and what the
        plugin reads. A refusal there is the ``outcome`` (``<plugin>: <Type>:
        <message>``); nothing is written and no thread starts. What the run does is
        shown by ``status()``.

        Raises:
            RuntimeError: A run is going (``<plugin> is running``), or the page is
                closed.
        """
        self._refuse_if_busy()
        try:
            entry = load(plugin, self._entry_points)
            try:
                parsed = parse_params(entry.plugin.Params, params)
            except ParamsError as exc:
                # parse_params knows the model, not the plugin: name the plugin, as
                # `smc run` does, so the advice (--show-params) is for the right one.
                raise ParamsError(plugin, exc.problems) from None
            except Exception as exc:
                # A third party's validator is code nobody here reviewed, and pydantic
                # turns only ValueError and AssertionError into a validation error:
                # any other exception is the plugin's, and an outcome, never a click
                # that ends with no word (FM-113).
                self._set_outcome(f"{plugin}: {type(exc).__name__}: {_one_line(exc)}")
                return
            doc = RunDocument.load_or_new(self._run_doc)
            check_ready(
                entry.plugin,
                doc,
                parsed,
                profile=self._microscope.profile.microscope.name,
                dry_run=self._microscope.dry_run,
                resume=False,
            )
        except (PluginError, RunDocumentError) as exc:
            # Text, never the error (FM-98).
            self._set_outcome(f"{plugin}: {type(exc).__name__}: {_one_line(exc)}")
            return
        self._start_activity(
            plugin, lambda event: self._run_plugin(plugin, entry, parsed, event)
        )

    def _start_activity(
        self, name: str, body: Callable[[threading.Event], str]
    ) -> threading.Event:
        """Start the activity ``name`` on the daemon worker; ``body`` returns its outcome text.

        Returns the activity's event, which ends it when set: a caller that must
        tell its own activity from another one (``live``) keeps it.

        Every action that reaches the stand through a call that may last is such
        an activity (m5 §2.1): it holds ``_current_run`` under its name, so that a
        second one is refused and ``stop()``, ``resume()``, ``close()`` and
        ``join()`` treat it as they treat a run. The lock discipline is that of the
        class: ``_lock`` guards the activity's name, the progress, the outcome and
        what the activity keeps, and is never held across a device call.

        Raises:
            RuntimeError: An activity is going (``<name> is running``), or the page
                is closed.
        """
        event = threading.Event()
        with self._lock:
            if self._closed:
                raise RuntimeError("the page is closed")
            if self._current_run is not None:
                raise RuntimeError(f"{self._current_run[0]} is running")
            self._current_run = (name, event)
            self._progress = None
            self._outcome = None
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            # Daemon: a plugin stuck in a vendor call must not keep the process
            # alive after the page is closed.
            worker = threading.Thread(
                target=self._work,
                args=(name, body, event),
                name=f"smc-ui-{name}",
                daemon=True,
            )
            self._worker = worker
            worker.start()
        return event

    def _work(
        self,
        name: str,
        body: Callable[[threading.Event], str],
        event: threading.Event,
    ) -> None:
        """The worker's body: the activity, and its outcome as text before anything else ends."""
        outcome = f"{name}: ended without an outcome"
        try:
            outcome = body(event)
        finally:
            with self._lock:
                self._outcome = outcome
                self._current_run = None
                # After the activity's own writes (a run's record is on disk by now),
                # so a session that sees the count change reads them.
                self._ended += 1
            log.info("%s", outcome)

    def _run_plugin(
        self,
        plugin: str,
        entry: PluginEntry,
        parsed: PluginParams,
        event: threading.Event,
    ) -> str:
        """A plugin's run on the worker, and its outcome as text."""
        log.info("%s: started from the plate page", plugin)
        started = time.monotonic()
        try:
            record = run_plugin(
                entry.plugin,
                self._microscope,
                parsed,
                self._run_doc,
                on_progress=self._queue.put,
                stop=event,
                plugin_version=entry.version,
            )
        except RunDocumentError as exc:
            if exc.record is not None and exc.record.result is not None:
                return (
                    f"{plugin} ran, but its record could not be written: "
                    f"{_one_line(exc)}"
                )
            return f"{plugin}: {type(exc).__name__}: {_one_line(exc)}"
        except Exception as exc:
            return f"{plugin}: {type(exc).__name__}: {_one_line(exc)}"
        if record.status is StageStatus.done:
            return f"{plugin} done in {time.monotonic() - started:.1f} s"
        return f"{plugin} stopped"

    def lacks(self, capability: type) -> str | None:
        """Why the stand cannot offer ``capability``, or ``None``: the one text of a disabled control.

        A control the stand cannot honour says so instead of failing when it is
        pressed (ADR-0006, principle 1).
        """
        if self._microscope.has(capability):
            return None
        return f"this stand has no {capability.__name__}"

    def snap(self) -> None:
        """Snap one frame on the worker and keep it as ``frame()``; what it did is the ``outcome``.

        An activity like a run (m5 §2.1): refused while another goes. A stand
        without a camera is an outcome and no thread starts. A snap writes nothing
        to disk and nothing to the run document: ``smc snap --out`` writes a
        frame.

        Raises:
            RuntimeError: An activity is going (``<name> is running``), or the page
                is closed.
        """
        self._refuse_if_busy()
        reason = self.lacks(Camera)
        if reason is not None:
            self._set_outcome(f"snap: {reason}")
            return
        self._start_activity("snap", self._snap)

    def live(self, on: bool) -> None:
        """Start a live view, or end the one this page started; at most ten frames a second.

        ``on`` is an activity like a snap (m5 §2.1) that takes frames until its event
        is set, and keeps each one as ``frame()``. It refuses as a snap does, and a
        stand without a camera is an outcome and no thread starts. Run and Snap are
        refused while it goes, and it is never ended for them (m5 §5.2).

        ``not on`` sets the event of the live this page started, and only that one
        (compared by identity, never by the name, so a plugin called ``live`` is
        not stopped), and returns at once. Ending a live view is not an emergency:
        it never calls ``Microscope.stop()`` (STOP is that), never joins, takes no
        lock (FM-70) and never raises, a closed page included.

        Raises:
            RuntimeError: ``on`` while an activity is going (``<name> is running``),
                or the page is closed.
        """
        if on:
            self._refuse_if_busy()
            reason = self.lacks(Camera)
            if reason is not None:
                self._set_outcome(f"live: {reason}")
                return
            self._live_event = self._start_activity("live", self._live)
            return
        current = self._current_run
        if current is not None and current[1] is self._live_event:
            current[1].set()

    @property
    def is_live(self) -> bool:
        """Whether the activity going is the live view ``live(True)`` started: what the button reads."""
        current = self._current_run
        return current is not None and current[1] is self._live_event

    def frame(self) -> Frame | None:
        """The latest frame, ``None`` before the first; the same for every session."""
        with self._lock:
            return self._frame

    def _live(self, event: threading.Event) -> str:
        """Frames until ``event`` is set or one fails, and the outcome as text.

        The cap is a period: the wait is what is left of ``LIVE_PERIOD_S`` since the
        frame began, so a camera slower than that runs at its own rate, and the wait
        is on the event, so STOP ends it within one period (FM-83). A frame in flight
        when the event is set completes and is kept. Nothing is logged inside the
        loop (FM-62, FM-148): the start and the outcome are the log's.
        """
        log.info("live: started from the plate page")
        started = time.monotonic()
        frames = 0
        while not event.is_set():
            began = time.monotonic()
            taken = self._take_frame()
            if isinstance(taken, str):
                return f"live ended after {frames} frames: {taken}"
            frames += 1
            self._queue.put(
                Progress(plugin="live", done=frames, total=None, message="frames")
            )
            event.wait(max(0.0, LIVE_PERIOD_S - (time.monotonic() - began)))
        return f"live ended: {frames} frames in {time.monotonic() - started:.1f} s"

    def _snap(self, event: threading.Event) -> str:
        """One snap on the worker, and its outcome as text: no exception is kept (FM-98)."""
        log.info("snap: started from the plate page")
        taken = self._take_frame()
        if isinstance(taken, str):
            return f"snap: {taken}"
        return f"snap: {taken.summary}"

    def _take_frame(self) -> Frame | str:
        """One frame, kept as ``frame()`` and returned; or the failure as text, with no prefix.

        The step a snap and a live view share. Read as ``smc snap`` reads: the
        exposure, the frame, then the stand's position and pixel size. The camera's
        array is copied: MMCore may hand out a view of its own buffer, and the page
        keeps the frame for as long as a session draws it. It logs nothing: a live
        view calls it ten times a second, and a console that blocks must not hold
        the worker (FM-62, FM-148). A failure is text, never the exception (FM-98),
        so no traceback keeps the facade, and so the core, alive.
        """
        try:
            camera = self._microscope.require(Camera)
            exposure_ms = camera.exposure_ms()
            image = camera.snap()
            if not (
                image.ndim == 2
                and image.dtype.kind == "u"
                and image.dtype.itemsize <= 2
            ):
                return (
                    f"the camera returned a {image.dtype} frame of shape "
                    f"{image.shape}; the page shows 2-D uint8 or uint16 frames"
                )
            bit_depth = camera.bit_depth()
            state = self._microscope.state()
            kept = np.array(image)
            kept.flags.writeable = False
            height, width = kept.shape
            summary = (
                f"{height}x{width} {kept.dtype}, exposure {exposure_ms:g} ms, "
                f"min {int(kept.min())}, max {int(kept.max())}"
            )
            taken_utc = datetime.now(timezone.utc)
            xy_um = None if state.xy is None else (state.xy.x_um, state.xy.y_um)
            with self._lock:
                previous = self._frame
                taken = Frame(
                    number=1 if previous is None else previous.number + 1,
                    image=cast("npt.NDArray[np.uint8] | npt.NDArray[np.uint16]", kept),
                    bit_depth=bit_depth,
                    exposure_ms=exposure_ms,
                    # 0.0 is the camera (or a failed read) saying "unknown".
                    pixel_size_um=state.pixel_size_um or 0.0,
                    xy_um=xy_um,
                    taken_utc=taken_utc,
                    summary=summary,
                )
                self._frame = taken
        except Exception as exc:
            return f"{type(exc).__name__}: {_one_line(exc)}"
        return taken

    def stop(self) -> None:
        """Ask the run to stop, then stop the stand; say what the stop did, never "stopped" by default.

        The event is set first, so the plugin leaves its loop as the stand halts,
        and the log line comes after the stop (FM-62). It takes no lock the worker
        holds (FM-70) and never raises for a stop that is refused or fails
        (FM-117): the outcome is ``stop_note``. STOP with nothing running still
        halts the stand.
        """
        current = self._current_run
        if current is not None:
            current[1].set()
        prefix = "" if current is None else f"{current[0]} told to stop; "
        try:
            self._microscope.stop()
        except StopRefusedError as exc:
            note = f"{prefix}halted, but {_one_line(exc)}"
        except HardwareError as exc:
            note = f"{prefix}the stop failed: {_one_line(exc)}"
        else:
            note = f"{prefix}{_STOP_ACCEPTED}"
        log.info("stop: %s", note)
        self._stop_note = note

    def resume(self) -> None:
        """Clear the halt; refused while a run is going.

        Raises:
            RuntimeError: A run is going (``<plugin> is running``).
        """
        current = self._current_run
        if current is not None:
            raise RuntimeError(f"{current[0]} is running")
        self._microscope.resume()
        self._stop_note = None

    def status(self) -> PageStatus:
        """The run and the stand now: the progress queue is drained and the stand is read.

        The latest ``Progress`` is kept, so that every session sees the same. The
        stand's read takes no lock and tolerates a failed device: what failed is in
        ``problems``.
        """
        latest: Progress | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if latest is not None:
                self._progress = latest
            running = None if self._current_run is None else self._current_run[0]
            progress = self._progress
            outcome = self._outcome
        state = self._microscope.state()
        problems = list(state.errors)
        light_on: bool | None = None
        light = self._microscope.get(LightSource)
        if light is not None:
            try:
                light_on = light.is_on()
            except Exception as exc:
                problems.append(f"light: {type(exc).__name__}: {_one_line(exc)}")
        autofocus = (
            _autofocus_words(state.autofocus_engaged, state.autofocus_locked)
            if self._microscope.has(Autofocus)
            else "no autofocus"
        )
        return PageStatus(
            stand=self._microscope.profile.microscope.name,
            dry_run=self._microscope.dry_run,
            running=running,
            progress=progress,
            outcome=outcome,
            stop_note=self._stop_note,
            xy_um=None if state.xy is None else (state.xy.x_um, state.xy.y_um),
            z_um=state.z_um,
            objective=state.objective,
            autofocus=autofocus,
            light_on=light_on,
            moving=tuple(state.moving),
            halted=state.halted,
            stop_refused=tuple(
                role.value for role in self._microscope.profile.safety.stop_refused_by
            ),
            problems=tuple(problems),
        )

    def join(self, timeout_s: float | None = None) -> bool:
        """Wait for the worker, in slices of 0.1 s; whether none is left alive.

        On Windows one long wait on a thread is not interrupted by Ctrl-C (FM-83),
        so the wait returns to Python between two slices.
        """
        worker = self._worker
        if worker is None:
            return True
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while worker.is_alive():
            if deadline is None:
                worker.join(_WAIT_SLICE_S)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(min(_WAIT_SLICE_S, remaining))
        return not worker.is_alive()

    def close(self) -> None:
        """Stop a run that is going and wait for it; afterwards ``run`` refuses.

        Idempotent: the first call does the work, and a later one (the command's
        ``finally`` after ``serve`` already closed the page) returns at once, so a
        second Ctrl-C never waits twice. It never closes the stand (the command
        that serves the page owns it) and never raises for a refused stop. What it
        tells the operator is ``close_lines``: what the stop did and how the run
        ended, never "stopped" by default (FM-117). A Ctrl-C during the wait is
        not caught: the stop was sent before it (FM-71).
        """
        if self._closed:
            return
        self._closed = True
        current = self._current_run
        if current is None:
            return
        plugin = current[0]
        self.stop()
        ended = self.join(_CLOSE_WAIT_S)
        if not ended:
            self._set_outcome(f"{plugin} did not end within {_CLOSE_WAIT_S:.0f} s")
        with self._lock:
            outcome = self._outcome
        note = self._stop_note
        if (
            ended
            and outcome == f"{plugin} stopped"
            and note == f"{plugin} told to stop; {_STOP_ACCEPTED}"
        ):
            # The stand took the stop and the run ended stopped: the one case in which
            # "was stopped" is the whole truth.
            lines: tuple[str, ...] = (f"{plugin} was stopped",)
        else:
            lines = tuple(line for line in (note, outcome) if line)
        self.close_lines = lines
        self.close_ended = ended

    def _refuse_if_busy(self) -> None:
        if self._closed:
            raise RuntimeError("the page is closed")
        current = self._current_run
        if current is not None:
            raise RuntimeError(f"{current[0]} is running")

    def _set_outcome(self, outcome: str) -> None:
        with self._lock:
            self._outcome = outcome

    # --- the view ---------------------------------------------------------------

    def tools(self) -> tuple[ToolView, ...]:
        """Every registered tool: those ``STAGES`` names in the workflow's order, then the others under ``More``, by name.

        A stage with no registered tool has none in the result; the stage's name
        is on each tool. A tool's ``key`` rows are ``KEY_PARAMS``' in its order and
        then every **required** field it does not name, so a plugin with no line
        in ``KEY_PARAMS`` is usable all the same; ``advanced`` is the rest. A
        plugin whose parameters cannot be described is left out and is a line of
        ``view().problems``, never an error here (FM-113).
        """
        return self._tools_and_problems()[0]

    def _tools_and_problems(self) -> tuple[tuple[ToolView, ...], tuple[str, ...]]:
        """The tools and what could not be read for them, built once: the registry does not change (FM-112)."""
        if self._tools_found is None:
            plugins = self._registry().plugins
            staged = {name: stage for stage, names in STAGES for name in names}
            order = [
                name for _stage, names in STAGES for name in names if name in plugins
            ]
            order += sorted(name for name in plugins if name not in staged)
            views: list[ToolView] = []
            problems: list[str] = []
            for name in order:
                entry = plugins[name]
                try:
                    described = describe_params(entry.plugin.Params)
                except Exception as exc:
                    # Third-party code: a model whose defaults or types cannot be
                    # described is a finding, not the end of the page (FM-113).
                    problems.append(
                        f"plugin {name}: {type(exc).__name__}: {_one_line(exc)}"
                    )
                    continue
                hidden = PAGE_PARAMS.get(name, ())
                rows = [r for r in described if r[0] not in hidden]
                by_name = {row[0]: row for row in rows}
                keys = [k for k in KEY_PARAMS.get(name, ()) if k in by_name]
                keys += [
                    row[0]
                    for row in rows
                    if row[2] == "(required)" and row[0] not in keys
                ]
                views.append(
                    ToolView(
                        name=name,
                        summary=entry.plugin.summary,
                        stage=staged.get(name, MORE_STAGE),
                        key=tuple(by_name[k] for k in keys),
                        advanced=tuple(r for r in rows if r[0] not in keys),
                    )
                )
            self._tools_found = (tuple(views), tuple(problems))
        return self._tools_found

    def view(self) -> PlateView:
        """A snapshot built from a fresh read of the run document; it raises nothing for it.

        An unreadable document, or a result its model refuses, is a line in
        ``problems`` and is drawn as absent (FM-125); the file is never written.
        """
        doc, unreadable = self._document()
        problems: list[str] = []
        if unreadable is not None:
            problems.append(f"run document: {unreadable}")
        registered = self._slot(doc, "plate", PlateResult, problems)
        density = self._slot(doc, "density", DensityResult, problems)
        route = self._slot(doc, "route", RouteResult, problems)
        geometries, nominal = self._registered_geometries(registered)
        selected = frozenset(self.selection)
        measured = {} if density is None else {w.well: w for w in density.wells}
        wells: list[WellView] = []
        for geometry in geometries:
            found = measured.get(geometry.well)
            wells.append(
                WellView(
                    well=geometry.well,
                    center_um=geometry.center_um,
                    size_um=geometry.size_um,
                    circular=geometry.circular,
                    rotation_deg=geometry.rotation_deg,
                    selected=geometry.well in selected,
                    counts=None if found is None else dict(found.counts),
                    per_mm2=(
                        None
                        if found is None or found.per_mm2 is None
                        else dict(found.per_mm2)
                    ),
                )
            )
        stops = (
            ()
            if route is None
            else tuple(
                RouteStop(p.order, p.well, p.x_um, p.y_um)
                for p in sorted(route.positions, key=lambda p: p.order)
            )
        )
        return PlateView(
            plate=self._plate,
            nominal=nominal,
            wells=tuple(wells),
            labels=() if density is None else density.labels,
            route=stops,
            checklist=self._checklist(doc),
            blocked=self._blocked(doc),
            problems=(*problems, *self._plugin_problems()),
            cards={
                tool.name: _card(doc.stages[tool.name], stale=doc.is_stale(tool.name))
                for tool in self.tools()
                if tool.name in doc.stages
            },
        )

    # --- internals of the view -----------------------------------------------------

    def _registry(self) -> Registry:
        """The plugins, found once: a registration changes only with an install (FM-112)."""
        if self._registry_found is None:
            self._registry_found = discover(self._entry_points)
        return self._registry_found

    def _plugin_problems(self) -> tuple[str, ...]:
        """A line for each plugin that did not load, and for each whose parameters could not be described."""
        failed = tuple(
            f"plugin {failure.name}: {' '.join(failure.reason.split())}"
            for failure in self._registry().failures
        )
        return (*failed, *self._tools_and_problems()[1])

    def _document(self) -> tuple[RunDocument, str | None]:
        """The run document as it is now, or a new one and why the file could not be read."""
        try:
            return RunDocument.load_or_new(self._run_doc), None
        except RunDocumentError as exc:
            # The text, never the error: a kept error keeps its traceback (FM-98).
            return RunDocument.new(), str(exc)

    def _unusable(self, doc: RunDocument, slot: str) -> str | None:
        return doc.unusable(
            slot,
            profile=self._microscope.profile.microscope.name,
            dry_run=self._microscope.dry_run,
        )

    def _slot(
        self, doc: RunDocument, slot: str, model: type[Any], problems: list[str]
    ) -> Any | None:
        """The result of a usable ``slot`` as ``model``; ``None`` and a problem when it refuses."""
        if self._unusable(doc, slot) is not None:
            return None
        provider = doc.provider(slot)
        if provider is None:  # unreachable once the slot is usable
            return None
        try:
            return model.model_validate(provider.result)
        except ValidationError as exc:
            problem = _first_problem(exc)
        problems.append(f"{slot}: result does not validate: {problem}")
        return None

    def _checklist(self, doc: RunDocument) -> tuple[CheckItem, ...]:
        """M3's slots in their order, then every other slot a record of the document provides.

        A third-party plugin's stage shows without a line in ``_CHECKLIST``: the
        extra slots are optional (no workflow of ours needs them) and by name.
        """
        named = {slot for slot, _optional in _CHECKLIST}
        extra = sorted({r.provides for r in doc.stages.values()} - named)
        items: list[CheckItem] = []
        for slot, optional in (*_CHECKLIST, *((slot, True) for slot in extra)):
            if doc.provider(slot) is None:
                items.append(CheckItem(slot, optional, "not run", None))
                continue
            reason = self._unusable(doc, slot)
            if reason is None:
                items.append(CheckItem(slot, optional, "ok", None))
            else:
                items.append(CheckItem(slot, optional, "unusable", reason))
        return tuple(items)

    def _blocked(self, doc: RunDocument) -> dict[str, str]:
        """Why each registered tool cannot run now; a tool that can is absent."""
        blocked: dict[str, str] = {}
        for name, entry in self._registry().plugins.items():
            reason = next(
                (r for r in (self._unusable(doc, s) for s in entry.plugin.reads) if r),
                None,
            )
            if reason is None:
                reason = next(
                    (
                        lack
                        for lack in (self.lacks(cap) for cap in entry.plugin.requires)
                        if lack
                    ),
                    None,
                )
            if reason is not None:
                blocked[name] = reason
        return blocked

    def _geometries(self, doc: RunDocument) -> tuple[tuple[WellGeometry, ...], bool]:
        """The wells the map draws for ``doc``: what ``view`` and ``click`` both use."""
        registered = self._slot(doc, "plate", PlateResult, [])
        return self._registered_geometries(registered)

    def _registered_geometries(
        self, registered: PlateResult | None
    ) -> tuple[tuple[WellGeometry, ...], bool]:
        """The registered plate's wells when it is the page's plate, else the nominal plate's.

        Built once per ``(plate, a1_center_xy_um, rotation_deg)`` and kept: 1536
        wells cost 339 ms, which a click or a refresh must not pay again.
        """
        if registered is not None and registered.plate == self._plate:
            key = (self._plate, registered.a1_center_xy_um, registered.rotation_deg)
            nominal = False
        else:
            key = (self._plate, (0.0, 0.0), 0.0)
            nominal = True
        found = self._geometry.get(key)
        if found is None:
            if nominal or registered is None:
                plan = useq.WellPlatePlan(
                    plate=useq.WellPlate.from_str(self._plate), a1_center_xy=(0.0, 0.0)
                )
            else:
                plan = registered.to_plan()
            found = tuple(
                well_geometry(plan, str(name))
                for name in np.asarray(plan.all_well_names).ravel()
            )
            if len(self._geometry) >= _GEOMETRY_KEPT:
                self._geometry.clear()
            self._geometry[key] = found
        return found, nominal

    def _plate_names(self) -> frozenset[str]:
        found = self._names.get(self._plate)
        if found is None:
            plan = useq.WellPlatePlan(
                plate=useq.WellPlate.from_str(self._plate), a1_center_xy=(0.0, 0.0)
            )
            found = frozenset(str(n) for n in np.asarray(plan.all_well_names).ravel())
            self._names[self._plate] = found
        return found

    def _on_plate(self, well: str) -> str:
        """``well`` as ``useq`` names it, or a ``ValueError`` when the plate has no such well."""
        name = normalise_well(well)
        if name not in self._plate_names():
            raise ValueError(f"{name} is not a well of the {self._plate} plate")
        return name

    def _refuse_another_registered_plate(self, doc: RunDocument) -> None:
        """Refuse a usable ``plate`` slot that names another plate than the page's.

        A slot whose result its model refuses is no case for this check: the plugin
        that reads it says so.
        """
        registered = self._slot(doc, "plate", PlateResult, [])
        if registered is not None and registered.plate != self._plate:
            raise ValueError(
                f"the registered plate is a {registered.plate}; calibrate the "
                f"{self._plate} plate again, or choose {registered.plate}"
            )

    def _initial_plate(self, doc: RunDocument) -> str:
        """The plate of the latest registration, else of the usable ``plate`` slot, else 96-well."""
        known = useq.registered_well_plate_keys()
        record = doc.stages.get("plate-registration")
        if record is not None:
            plate = record.params.get("plate")
            if isinstance(plate, str) and plate in known:
                return plate
        registered = self._slot(doc, "plate", PlateResult, [])
        if registered is not None and registered.plate in known:
            return str(registered.plate)
        return "96-well"

    def _restore(self, doc: RunDocument) -> None:
        """The selection and the captures the document already holds; what does not parse is dropped."""
        scan_plan = doc.stages.get("scan-plan")
        if scan_plan is not None:
            wells = scan_plan.params.get("wells")
            try:
                if isinstance(wells, str):
                    self.selection = parse_wells(self._plate, wells)
            except ValueError as exc:
                log.info(
                    "the scan-plan record's wells %r are not read: %s",
                    wells,
                    _one_line(exc),
                )
        registration = doc.stages.get("plate-registration")
        if registration is not None:
            self.refs = self._captured(registration.params, "refs", WellRef)
            self.walls = self._captured(registration.params, "walls", WallTouch)

    @staticmethod
    def _captured(
        params: dict[str, Any], key: str, model: type[_Capture]
    ) -> tuple[_Capture, ...]:
        raw = params.get(key)
        try:
            return tuple(model.model_validate(item) for item in raw or ())
        except (ValueError, TypeError) as exc:
            log.info(
                "the plate-registration record's %s are not read: %s",
                key,
                _one_line(exc),
            )
            return ()
