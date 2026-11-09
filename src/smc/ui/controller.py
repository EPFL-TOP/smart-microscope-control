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
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
import useq
from pydantic import BaseModel, ValidationError

from smc import drive
from smc.drive import DriveResult
from smc.hardware.capabilities import (
    Autofocus,
    Camera,
    LightSource,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import HardwareError, MicroscopeBusyError, StopRefusedError
from smc.hardware.microscope import describe_autofocus
from smc.hardware.safety import Safety
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
from smc.plugins.presets import (
    NOT_IN_PARAMS,
    Preset,
    PresetCheck,
    PresetError,
    PresetFile,
    check_name,
    check_preset,
    list_presets,
    registered_plate,
)
from smc.plugins.presets import load_preset as read_preset
from smc.plugins.presets import save_preset as write_preset
from smc.plugins.runner import reachable_checkpoint
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
from smc.ui.guide import (
    CAPTURE_WELLS,
    DRY_RUN_NO_FRAME,
    PLATE_NOT_CHOSEN,
    SELECT_WELLS,
    WALLS_DRY_RUN,
    WALLS_NEED_CALIBRATION,
    NextStep,
    command_line,
    next_slot,
    replace_note,
    walls_card_lines,
    walls_centres_problem,
)

if TYPE_CHECKING:
    from types import TracebackType

    from smc.hardware.microscope import Microscope
    from smc.hardware.profile import Profile

__all__ = [
    "KEY_PARAMS",
    "LIVE_PERIOD_S",
    "MORE_STAGE",
    "NOT_CONNECTED",
    "PAGE_PARAMS",
    "STAGES",
    "CheckItem",
    "LoadedPreset",
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
    (
        "Set up",
        (
            "camera-calibration",
            "plate-registration",
            "plate-walls",
            "plate-finding",
            "timing",
        ),
    ),
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
    # The same captured centres as the registration, never its wall touches: the tool
    # images the walls itself (m5 §12.2). Naming `plate` is what gates it on §11.3.
    "plate-walls": ("plate", "refs"),
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

#: The strip says a finished run's first assumption, cut here: a pointer to the card, which
#: holds every line at ``_CARD_TEXT_CHARS`` (m5 §8.8.2).
_STRIP_ASSUMED_CHARS = 160

#: Two reference centres are the same when they differ by at most this: a JSON dump
#: rounds, and a capture made by hand is never within a hundredth of a micrometre of the
#: sample's (m5 §8.8.4).
_SAME_REF_UM = 0.01

#: The keys of a ``done`` result that name a file or a folder the tool wrote: ``scan``'s
#: ``folder`` and the ``file`` of ``cells``, ``report`` and ``nis-export``. A third party
#: that follows the convention is shown the same way (m5 §8.4).
_FILE_KEYS = ("file", "folder")

#: A card's time when its record's date cannot be put in UTC: unknown, never a guess.
_UNKNOWN_TIME = "????-??-?? ??:??:??"

#: How long ``close`` waits for a stopped run to end, in seconds.
_CLOSE_WAIT_S = 10.0

#: How long a Disconnect or ``release()`` waits for the reads in flight before it
#: unloads, in seconds: as long as a slow device read, and a hung one is not waited
#: on for ever (FM-15, m5 §7.2). Polled every ``_READS_POLL_S``.
_READS_WAIT_S = 5.0
_READS_POLL_S = 0.01

#: How long ``release()`` waits for a connect that ``close()`` gave up on, in
#: seconds. 10 (``close``) + 10 (the server's stop) + 8 is 28 s, inside the service's
#: ``AppStopMethodConsole 30000`` (``docs/ui/service.md``).
_RELEASE_WAIT_S = 8.0

#: A wait on the worker is made of slices this long (FM-83: Windows does not deliver
#: Ctrl-C during one long wait on a thread).
_WAIT_SLICE_S = 0.1

#: A live view takes a frame at most once in this many seconds: the cap is a period,
#: so a camera slower than 10 frames a second runs at its own rate (m5 §5.1).
LIVE_PERIOD_S = 0.1

#: What ``stop`` says when the stand took the stop: the one wording that may be told
#: as "stopped" (FM-117).
_STOP_ACCEPTED = "stopped: the stand is halted (Resume clears the halt)"

#: Why everything that needs the stand is refused while the page holds none (m5 §7.2):
#: the one reason a disabled control gives, and the first one.
NOT_CONNECTED = "the stand is not connected"

#: Wells built for a plate are kept (339 ms for 1536 of them); a registration
#: changes the key rarely, so a handful is plenty.
_GEOMETRY_KEPT = 4

_Capture = TypeVar("_Capture", WellRef, WallTouch)


#: The providers of ``plate`` whose ``plate`` parameter is the plate on the stage. Another
#: provider's parameter of that name means something else (``plate-finding``'s sizes its
#: survey), so what it holds is read from its result only (``_plate_named_by``).
_PLATE_PARAM_PLUGINS = ("plate-registration", "plate-walls")


def _needs_plate(plugin: str) -> bool:
    """Whether the page fills ``plugin``'s ``plate`` itself, so that it needs a plate the operator chose (m5 §11.3).

    Read from ``PAGE_PARAMS``, so that a tool the page learns to fill the plate for
    (``plate-walls``, #346) is gated by naming ``plate`` there, with no further code.
    """
    return "plate" in PAGE_PARAMS.get(plugin, ())


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
    ``files`` are ``<key>: <path>`` for each file the result names, under the same
    rule as ``lines``: where the result is, so that the operator can open it.
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
    files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlateView:
    """A snapshot of what the run document says, as the page needs it (a pure read).

    ``nominal`` is true when the wells are those of the plate type with A1 at
    (0, 0) and no rotation, not a registered plate. ``blocked`` maps a registered
    tool to why it cannot run now: what it reads, what the stand lacks, or a
    stopped or failed run's checkpoint that ``smc run --resume`` could continue
    (``run()`` refuses to start such a stage afresh). ``problems`` are what
    ``view()`` could not read and did not hide. ``cards`` has one card for each
    tool that has a record in the run document, and none for the others.
    ``next_step`` is the tool the run document needs next and the values to run it
    with (m5 §8.3), built here from this read and never by the tick. ``replaces``
    maps each tool that provides the ``plate`` slot to what a Run of it would replace,
    while that slot is usable (m5 §11.4); a tool never refused for it, and no other
    slot has one.
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
    next_step: NextStep | None = None
    replaces: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolView:
    """A tool the page can run: its rows are ``describe_params`` rows (name, type, default, notes).

    ``suggested`` are the ``(key, value)`` pairs the profile suggests for the tool
    and ``suggested_why`` its one-line reason: the page starts those fields with
    their value, so a click on the tool's own *Run* sends what the next step shows.
    """

    name: str
    summary: str
    stage: str
    key: tuple[tuple[str, str, str, str], ...]
    advanced: tuple[tuple[str, str, str, str], ...]
    suggested: tuple[tuple[str, str], ...] = ()
    suggested_why: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedPreset:
    """The preset the page was given, what of it applied and what did not (m5 §10.5).

    ``check`` is the preset checked against this stand and run document, whose
    ``params`` are the pairs that apply; ``fields`` are those pairs split at their
    first ``=``, per tool, for the view to put in its fields. ``selection`` is the
    selection the load set, ``None`` when it set none. ``mismatches`` are the
    check's lines and then the page's own: every one a note, never a refusal.
    """

    file: PresetFile
    check: PresetCheck
    fields: dict[str, dict[str, str]]
    selection: tuple[str, ...] | None
    mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Suggestion:
    """A profile's ``[suggest.<plugin>]`` that passed the registry's check: pairs and why."""

    pairs: tuple[tuple[str, str], ...]
    why: str


class _Handed(NamedTuple):
    """A drive action handed to the live worker, waiting for the next turn of its loop (m5 §6.3).

    ``run`` returns the outcome text and never raises an ``Exception``. ``switch_off``
    is ``True`` for the two switch-offs, ``light off`` and ``autofocus disengage``:
    the one kind of action that is still run when live ends, because the facade
    never refuses it, also halted, and it sends no move (#295).
    """

    name: str
    run: Callable[[], str]
    switch_off: bool


@dataclass(frozen=True, slots=True)
class PageStatus:
    """The stand and the run, read now.

    ``stand`` is the profile's ``microscope.name``. ``xy_um`` to ``halted`` are
    ``Microscope.state()``'s; ``stop_refused`` are the roles the profile declares
    refuse their stop, so that the page can say that STOP does not stop a move in
    flight on this stand (FM-117). With ``connected`` false the page holds no
    stand, and every read of it is unknown (``None``, ``?``), never a fault.
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
    connected: bool = True


def _one_line(error: BaseException) -> str:
    """The message of ``error`` on one line; an error whose message cannot be built says nothing."""
    try:
        return " ".join(str(error).split())
    except (
        Exception
    ):  # a message that cannot be printed is not worth losing the outcome
        return ""


def _checkpoint_reason(name: str, status: StageStatus) -> str:
    """Why the page will not start ``name`` afresh, and the command that continues it (m3 §15.3).

    The page has no Resume (``smc run --resume`` is the way in M3), so a stage the
    operator stopped, or that failed, with a checkpoint would be lost by one click
    on Run. No path is named: the run document is the one this page was started
    with, and the command needs ``--run-doc`` whatever the text says. ``smc run``
    without ``--resume`` is the deliberate way to give the checkpoint up.
    """
    return (
        f"{status.value} with a checkpoint: continue it with `smc run {name} "
        f"--resume` (close this page first); `smc run {name}` without --resume "
        f"starts afresh and drops it"
    )


def _light_lacks_text(light: LightSource, on: bool | None) -> str | None:
    """Why a light source whose ``is_on()`` read ``on`` cannot be switched on from the page, or ``None``.

    ``None`` is a ``State`` device (the demo's ``LED``): smc knows no on/off for it.
    """
    if on is None:
        return f"the light source {light.device()} has no on/off that smc knows"
    return None


def _then(text: str, last: str | None) -> str:
    """``text``, then ``; <last>`` when there is a last drive action's outcome to keep."""
    return text if last is None else f"{text}; {last}"


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


def _file_path(value: str, run_dir: PurePath | None) -> str:
    """Where a result's ``file`` is: an absolute path as written, a relative one under the run folder.

    The plugins write a relative one (``report/<token>.html``) beside the run
    document, which is the folder ``ctx.run_dir`` names; ``nis-export --param out=``
    writes an absolute one, as POSIX text (``/data/p.xml``) whatever the OS it ran
    on. A run document moves between PCs, so a value that is absolute in *either*
    flavour is shown as written: joined on Windows, ``/data/p.xml`` is rooted
    but not absolute there, and the run folder's drive would be put in front of
    it, naming a path that does not exist. The path is shown, never opened or
    checked: a card reads the document only. With no run folder (a card built
    alone) it is as written.
    """
    if (
        run_dir is None
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        return value
    return str(run_dir / value)


def _plate_from_images(result: Mapping[str, Any]) -> PlateResult | None:
    """``result`` as a ``PlateResult`` that holds ``walls``, or ``None``: anything else is the generic card.

    A document is data (FM-125): a result that does not validate, or a plate
    registered by another method, is no case for the card of this one, and the
    validation is tried only when the result carries a ``walls`` object.
    """
    if not isinstance(result.get("walls"), dict):
        return None
    try:
        found = PlateResult.model_validate(result)
    except ValidationError:
        return None
    return found if found.walls is not None else None


def _card(
    record: StageRecord, *, stale: bool = False, run_dir: PurePath | None = None
) -> ResultCard:
    """The card of ``record``; its values only when it ended ``done`` (FM-154).

    ``stale`` is ``RunDocument.is_stale`` of the record, and marks only a ``done``
    one: a record that did not end done has no values to mislead, and its status
    already says to run it again. The stale card keeps its values (they are a result,
    of inputs that are no longer the document's).

    A ``_FILE_KEYS`` key with a text in it is a file, not a value: it is a
    ``files`` entry, and does not count against the card's lines. A number, a
    list or a blank text under such a key is no path and stays a line (FM-125).

    A result that is a plate registered from images (``PlateResult`` with
    ``walls``) shows the wells measured and the plate instead of its first keys,
    which cut the walls away (m5 §12.6); its frames' folder is a file. A result that
    does not validate as one is the generic card, never an exception (FM-125).
    """
    when = record.finished_utc or record.started_utc
    try:
        when_utc = when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError):
        # A date a model accepts (year 1 with an offset) that UTC cannot hold: a
        # hand-edited document must not stop ``view()``, which raises nothing for one.
        when_utc = _UNKNOWN_TIME
    lines: list[str] = []
    files: list[str] = []
    assumed: tuple[str, ...] = ()
    if record.status is StageStatus.done and record.result is not None:
        from_images = _plate_from_images(record.result)
        if from_images is not None:
            lines = [_cut(_flat(line)) for line in walls_card_lines(from_images)]
            folder = None if from_images.walls is None else from_images.walls.folder
            if folder is not None and folder.strip():
                files.append(
                    _cut(
                        _flat(f"folder: {_file_path(folder, run_dir)}"),
                        _CARD_TEXT_CHARS,
                    )
                )
        else:
            for key, value in record.result.items():
                if key == "assumed":
                    continue
                if key in _FILE_KEYS and isinstance(value, str) and value.strip():
                    files.append(
                        _cut(
                            _flat(f"{key}: {_file_path(value, run_dir)}"),
                            _CARD_TEXT_CHARS,
                        )
                    )
                else:
                    lines.append(_value_line(key, value))
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
        files=tuple(files),
    )


def _sample_refs_text(refs: Sequence[WellRef]) -> str:
    """Where references that are the profile's ``[sample]``'s come from: the step's line and the card's.

    One text, so that the step before the click and the card after it say the
    same thing (m5 §8.8.4): the plugin was given the references and cannot tell a
    synthetic plate's from a measured one, so the page says it.
    """
    return (
        f"refs: the centres of {', '.join(ref.well for ref in refs)} where the "
        "profile's [sample] puts the plate; this stand's plate is synthetic, so "
        "they are known, not measured"
    )


def _are_the_refs(found: object, refs: Sequence[WellRef]) -> bool:
    """Whether ``found``, a record's ``params["refs"]`` as it came, is ``refs`` within ``_SAME_REF_UM``.

    A run document is a file an operator may have edited: anything that is not a
    list of objects with a text ``well`` and numbers ``x_um`` and ``y_um``, with
    the wells of ``refs`` and no other, is not the sample's, and is never an
    exception (FM-125). A flag is not a number.
    """
    if not isinstance(found, list) or len(found) != len(refs):
        return False
    centres: dict[str, tuple[float, float]] = {}
    for item in found:
        if not isinstance(item, dict):
            return False
        well = item.get("well")
        if not isinstance(well, str):
            return False
        numbers: list[float] = []
        for value in (item.get("x_um"), item.get("y_um")):
            if isinstance(value, bool) or not isinstance(value, int | float):
                return False
            numbers.append(value)
        centres[well] = (numbers[0], numbers[1])
    return len(centres) == len(refs) and all(
        ref.well in centres
        and abs(centres[ref.well][0] - ref.x_um) <= _SAME_REF_UM
        and abs(centres[ref.well][1] - ref.y_um) <= _SAME_REF_UM
        for ref in refs
    )


def _assumed_tail(record: StageRecord) -> str:
    """What the strip adds to a ``done`` run: its first assumption, and how many more the card holds.

    A dry-run ``scan`` ended ``scan done in 0.0 s`` while its record said ``dry run:
    36 fields walked, no frame taken``, which only the card showed (m5 §8.8.2). The
    lines are the card's, so they are flat, bounded and never markup (FM-125,
    FM-139), and empty unless the record ended ``done`` (FM-154).
    """
    assumed = _card(record).assumed
    if not assumed:
        return ""
    tail = f"; assumed: {_cut(assumed[0], _STRIP_ASSUMED_CHARS)}"
    if len(assumed) > 1:
        tail += f" (+{len(assumed) - 1} more, on the card)"
    return tail


def _open_stand(profile: Profile, dry_run: bool) -> Microscope:
    """What Connect calls when the page was given no opener: ``Microscope.open``.

    ``Microscope`` is imported, and ``open`` looked up, at the call: a test's
    ``monkeypatch.setattr(Microscope, "open", ...)`` made after the page was built is
    the one used, and the module's import opens nothing.
    """
    from smc.hardware.microscope import Microscope

    return Microscope.open(profile, dry_run=dry_run)


def _first_problem(error: ValidationError) -> str:
    """``<where>: <what>`` of the first validation error."""
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "result"
    return f"{where}: {first['msg']}"


class _Reading:
    """A read of the stand that takes no lock, registered so that a Disconnect can wait for it (m5 §7.2).

    The facade's reads (``state()``, the light's and the turret's, a capture's
    position, STOP) take no lock by design (FM-70), so ``unloadAllDevices`` could
    run while one was inside the core. ``__enter__`` adds a token to the page's
    ``_readers`` **first**, then returns the facade the page holds, or ``None``:
    a Disconnect drops the facade and only then looks at ``_readers``, so a reader
    that took the facade before the drop is in the set when it looks, and one that
    comes after gets ``None``. Reversing either order reopens the race, and no
    test on a fast fake shows it.

    A class, not a ``@contextmanager`` (FM-98, FM-92: a generator frame would hold
    the facade), and no lock: ``set.add`` and ``set.discard`` are atomic, so STOP
    can use it (FM-70). A second Ctrl-C landing in ``stop()`` between ``add`` and
    the ``with`` body leaks a token (FM-65); it costs one ``_READS_WAIT_S`` of a
    Disconnect, never a hang, and is accepted.
    """

    def __init__(self, page: PlatePage) -> None:
        self._page = page
        self._token = object()

    def __enter__(self) -> Microscope | None:
        self._page._readers.add(self._token)
        return self._page._microscope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._page._readers.discard(self._token)


class PlatePage:
    """The plate page's logic over one stand and one run document.

    The stand's lifetime is the operator's (m5 §7): ``connect()`` opens the
    profile's stand and ``disconnect()`` releases it, each an activity, so a stand
    held by smc can be handed to NIS-Elements or ZEN without stopping the server.
    ``PlatePage(microscope, ...)`` starts connected to that facade;
    ``PlatePage.closed(profile, ...)``, what ``smc ui`` builds, starts with none.
    The profile is the page's from the start, so the map, the checklist and the
    cards work with the stand closed. Everything an operator does here is a
    plugin run with the strings ``smc run`` takes, or a read of a capability.

    The facade may be dropped by a ``disconnect`` between two reads on the
    server's loop: a method that reads it there takes it into a local once, never
    ``self._microscope`` twice, and an activity reaches it through ``_stand()``.

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
        opener: Callable[[], Microscope] | None = None,
        entry_points: Iterable[EntryPoint] | None = None,
    ) -> None:
        """Start connected to ``microscope``, from the run document: its plate, the wells chosen and the references taken.

        ``opener`` is what a ``connect()`` after a ``disconnect()`` calls; it
        defaults to ``Microscope.open`` of the facade's profile and dry run.
        Never raises for a document: one that cannot be read starts the page
        empty, and ``view()`` says why in its ``problems``.
        """
        self._setup(
            microscope,
            microscope.profile,
            microscope.dry_run,
            run_doc,
            opener=opener,
            entry_points=entry_points,
        )

    @classmethod
    def closed(
        cls,
        profile: Profile,
        run_doc: Path,
        *,
        dry_run: bool = False,
        opener: Callable[[], Microscope] | None = None,
        entry_points: Iterable[EntryPoint] | None = None,
    ) -> PlatePage:
        """A page with the stand closed: ``connect()`` opens it with ``opener``.

        What ``smc ui`` builds, so that a service that starts at boot, or restarts
        after a crash, never takes the stand from the vendor's software (m5 §7).
        ``opener`` defaults to ``Microscope.open(profile, dry_run=dry_run)``,
        looked up when Connect calls it.
        """
        page = cls.__new__(cls)
        page._setup(
            None,
            profile,
            dry_run,
            run_doc,
            opener=opener,
            entry_points=entry_points,
        )
        return page

    def _setup(
        self,
        microscope: Microscope | None,
        profile: Profile,
        dry_run: bool,
        run_doc: Path,
        *,
        opener: Callable[[], Microscope] | None,
        entry_points: Iterable[EntryPoint] | None,
    ) -> None:
        """What both constructors do; the profile and the dry run are the page's, held or not."""
        self._microscope: Microscope | None = microscope
        # The reads in flight, one token each (``_Reading``): no lock, so that STOP
        # can register (FM-70). A Disconnect waits for it to empty (``_wait_for_reads``).
        self._readers: set[object] = set()
        self._profile = profile
        self._dry_run = dry_run
        self._opener: Callable[[], Microscope] = (
            (lambda: _open_stand(profile, dry_run)) if opener is None else opener
        )
        self._run_doc = run_doc
        # A list, so that a generator handed in is not spent by the first use.
        self._entry_points = None if entry_points is None else list(entry_points)
        self._registry_found: Registry | None = None
        self._tools_found: tuple[tuple[ToolView, ...], tuple[str, ...]] | None = None
        # Checked once, against the registry, before any tool is described: a tool's
        # form starts with a suggested value (m5 §8.3).
        self._suggested, self._suggest_problems = self._check_suggestions(profile)
        # The preset loaded on this page, shared by every tab and kept under
        # ``_lock``, as the plate and the selection are (m5 §10.5).
        self._preset: LoadedPreset | None = None
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
        # The one drive action handed to the live worker and not yet taken, under
        # ``_lock`` (m5 §6.3), and the name of the one the live worker is running
        # now, which a capture reads (#295): the stage is about to move or moves.
        self._drive_waiting: _Handed | None = None
        self._drive_running: str | None = None
        # ``light_lacks()``'s answer, kept after the first read that did not raise.
        self._light_lacks_found: tuple[str | None] | None = None
        # The facade whose light ``status()`` read and which raised: ``light_lacks()``
        # does not read it again for that facade (one read a tick, m5 §6.3).
        # A weak reference: the page keeps text and flags about a released facade,
        # never the facade, which would keep its core alive after ``release()``
        # (FM-92).
        self._light_read_raised: weakref.ref[Microscope] | None = None
        # Whether ``objective_labels()`` has logged the failure of its streak.
        self._labels_failure_logged = False
        self._closed = False
        # The name of the activity ``close()`` gave up waiting for, so that
        # ``release()`` waits for a ``connect`` whether it ended before it began or not.
        self._gave_up: str | None = None
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
        self._plate, self._plate_chosen = self._initial_plate(doc)
        self._restore(doc)

    # --- plate and selection -------------------------------------------------

    @property
    def activities_ended(self) -> int:
        """How many activities (runs, snaps, live sessions, drive actions) have ended since the page began.

        A drive action handed to the live worker is not an activity (m5 §6.3), and
        is not counted.

        What every session compares to see that something ended while it was not
        looking: a tool that takes less than a tick is never seen running by a
        second tab, and its record, its result card and the checklist are new all
        the same (FM-151).
        """
        return self._ended

    @property
    def run_doc(self) -> Path:
        """The run document's path, made absolute where the server runs (m5 §8.4).

        What the operator reads on the page and what a copied command names:
        a relative ``--run-doc`` would mean another file from another folder.
        Symbolic links are not resolved: it is the path as the command was given.
        """
        return self._run_doc.absolute()

    @property
    def plate(self) -> str:
        """The plate the page is about, a name of ``useq``'s registry."""
        return self._plate

    @property
    def plate_chosen(self) -> bool:
        """Whether something named the plate: the run document, the profile's sample, or the operator.

        ``False`` is the placeholder (m5 §11.3): the page shows a 96-well that
        nothing said was on the stage, and a tool that takes the plate does not run
        on it.
        """
        return self._plate_chosen

    def choose_plate(self, name: str) -> None:
        """Change the plate; the selection, the captures and ``current`` belong to the old one.

        Choosing the plate already shown confirms it (``plate_chosen``) and clears
        nothing: it is the operator saying that the placeholder is right.

        Raises:
            ValueError: ``name`` is not a plate of ``useq``'s registry; the
                message lists them.
        """
        plates = sorted(useq.registered_well_plate_keys())
        if name not in plates:
            raise ValueError(
                f"{name!r} is not a plate; the plates are {', '.join(plates)}"
            )
        self._plate_chosen = True
        if name == self._plate:
            return
        self._plate = name
        self.selection = ()
        self.refs = ()
        self.walls = ()
        self.current = None

    def select_text(self, text: str) -> None:
        """Replace the selection by the wells ``text`` names (``A1:D6``, ``C*``, ``all``).

        A blank text (empty, or only white space) is the *Wells* box emptied: no
        wells, so the selection is cleared and nothing is raised (m5 §8.8.5).
        ``parse_wells`` itself refuses an empty text, which is right for ``smc run
        scan-plan --param wells=``: that is not a selection but a missing value.

        Raises:
            ValueError: ``parse_wells``' message for any other text that names no
                well; the selection is as it was.
        """
        if text.strip() == "":
            self.selection = ()
        else:
            self.selection = parse_wells(self._plate, text)
        self._drop_unselected_current()

    def toggle(self, well: str) -> None:
        """Add ``well`` to the selection, or remove it; removing the well that is ``current`` drops it.

        Raises:
            ValueError: The well is not on the plate; the selection is as it was.
        """
        name = self._on_plate(well)
        if name in self.selection:
            self.selection = tuple(w for w in self.selection if w != name)
            # Only the well that was deselected: a deselect of another well leaves
            # ``current`` (a well driven to is not selected, and stays) (m5 §11.1).
            if self.current == name:
                self._drop_unselected_current()
        else:
            self.selection = (*self.selection, name)

    def _drop_unselected_current(self) -> None:
        """``current`` is dropped when the selection no longer holds it (m5 §11.1).

        ``current`` is the well *Capture* and *Forget* act on, and the page names it
        on their labels: a well the map shows as deselected must not stay the one a
        capture is recorded under. One helper called wherever the selection is
        replaced, so that no path forgets it.
        """
        if self.current is not None and self.current not in self.selection:
            self.current = None

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
            self.toggle(well)
            # After the toggle: a click that deselects sets nothing, and the one that
            # selects makes this well the one the captures act on (m5 §11.1).
            if well in self.selection:
                self.current = well
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
            RuntimeError: The stand is not connected (``NOT_CONNECTED``), or the
                stage is about to move or moves (``<name> is running`` or ``<name>
                is waiting``, ``_refuse_if_moving``).
            CapabilityMissingError: The stand has no XY stage.
        """
        name = self._on_plate(well)
        self._refuse_if_moving()
        with self._reading() as microscope:
            if microscope is None:
                raise RuntimeError(NOT_CONNECTED)
            position = microscope.require(XYStage).position_um()
        ref = WellRef(well=name, x_um=position.x_um, y_um=position.y_um)
        kept = tuple(r for r in self.refs if normalise_well(r.well) != name)
        self.refs = (*kept, ref)
        return ref

    def capture_wall(self, well: str, side: str) -> WallTouch:
        """Record the stage position as a point of ``well``'s ``side`` wall; replaces the same side.

        Raises:
            ValueError: ``side`` is not left, right, top or bottom, or the well is
                not on the plate.
            RuntimeError: As ``capture_centre``.
            CapabilityMissingError: The stand has no XY stage.
        """
        if side not in _SIDES:
            raise ValueError(f"side must be one of {', '.join(_SIDES)}, got {side!r}")
        name = self._on_plate(well)
        self._refuse_if_moving()
        with self._reading() as microscope:
            if microscope is None:
                raise RuntimeError(NOT_CONNECTED)
            position = microscope.require(XYStage).position_um()
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
                on another plate than the registered one; a tool that takes the
                plate while nothing named it (``PLATE_NOT_CHOSEN``);
                ``plate-walls`` with fewer than two centres captured or more than
                six, after the plate's gate (m5 §12.3).
        """
        if _needs_plate(plugin) and not self._plate_chosen:
            raise ValueError(PLATE_NOT_CHOSEN)
        if plugin == "plate-walls":
            # One read of the captures for the count and the pairs: another tab may
            # capture between two reads of `self.refs`.
            refs = self.refs
            problem = walls_centres_problem(len(refs))
            if problem is not None:
                raise ValueError(problem)
            return self._dump_refs(refs)
        return self._page_pairs(plugin)

    def _page_pairs(self, plugin: str) -> list[str]:
        """``page_params`` without the plate's gate: what the page would fill if the plate were chosen.

        Without the count of ``plate-walls``' centres either: that is the gate's
        (``page_params``), as the plate's is, so that a preset's check of the
        tool's other fields is not refused for it (m5 §12.2).
        """
        if plugin == "scan-plan":
            if not self.selection:
                raise ValueError("select at least one well first")
            self._refuse_another_registered_plate(self._document()[0])
            return [f"wells={','.join(self.selection)}"]
        if plugin == "plate-registration":
            return self._registration_pairs(self.refs, self.walls)
        if plugin == "plate-walls":
            return self._dump_refs(self.refs)
        return []

    def _dump_refs(self, refs: Sequence[WellRef]) -> list[str]:
        """The ``plate`` and ``refs`` pairs of the two tools that register from captured centres.

        One place dumps them, so that ``plate-registration`` and ``plate-walls``
        are given the same text for the same captures, and a step and a click agree.
        """
        dumped = json.dumps([r.model_dump(mode="json") for r in refs])
        return [f"plate={self._plate}", f"refs={dumped}"]

    def _registration_pairs(
        self, refs: Sequence[WellRef], walls: Sequence[WallTouch]
    ) -> list[str]:
        """The pairs of ``plate-registration`` for this plate: one place dumps them, so a step and a click agree."""
        dumped_walls = json.dumps([w.model_dump(mode="json") for w in walls])
        return [*self._dump_refs(refs), f"walls={dumped_walls}"]

    # --- presets (m5 §10.5) -----------------------------------------------------------

    @property
    def preset(self) -> LoadedPreset | None:
        """The preset loaded on the page, or ``None``; every tab sees it."""
        with self._lock:
            return self._preset

    def presets(self) -> tuple[PresetFile, ...]:
        """The presets of this stand's folders, a broken file with its error (never read by the tick)."""
        return list_presets(self._profile)

    def load_preset(self, source: str) -> LoadedPreset:
        """Apply the preset ``source`` names (a name or a ``.toml`` path) and keep it as the loaded one.

        The plate and the wells are applied, and the pairs are for the view to put
        in the tools' fields; every mismatch is a line of the result, never a
        refusal. It needs no stand and is allowed while an activity goes: it reaches
        no device, and a run in flight has already read its pairs.

        A plate that this page holds captures on is not changed for a preset of
        another one, and neither are its wells: ``choose_plate`` would drop the
        captures, which are measured and cannot be brought back by a click.

        Raises:
            PresetError: No such preset, or the file cannot be read or is invalid;
                the message is the diagnosis, on one line.
        """
        file = read_preset(source, self._profile)
        models: dict[str, type[BaseModel]] = {
            name: entry.plugin.Params
            for name, entry in self._registry().plugins.items()
        }
        check = check_preset(
            file,
            models,
            stand=self._profile.microscope.name,
            registered=registered_plate(self._document()[0]),
        )
        mismatches = list(check.mismatches)
        plate, wells = check.plate, check.wells
        if plate is not None and plate != self._plate and (self.refs or self.walls):
            mismatches.append(
                f"preset {file.name} is for a {plate} plate; this page has captures "
                f"on a {self._plate} plate: its plate and wells are not applied"
            )
            plate = wells = None
        if plate is not None:
            self.choose_plate(plate)
        selection: tuple[str, ...] | None = None
        if wells is not None:
            try:
                self.select_text(wells)
            except ValueError as exc:
                mismatches.append(f"preset {file.name}: wells {wells}: {exc}")
            else:
                selection = self.selection
        loaded = LoadedPreset(
            file=file,
            check=check,
            fields={
                plugin: dict(item.split("=", 1) for item in items)
                for plugin, items in check.params.items()
            },
            selection=selection,
            mismatches=tuple(mismatches),
        )
        with self._lock:
            self._preset = loaded
        return loaded

    def forget_preset(self) -> None:
        """No preset is loaded: a run records none, and the next step suggests none."""
        with self._lock:
            self._preset = None

    def save_preset(
        self,
        name: str,
        fields: Mapping[str, Sequence[str]],
        wells: str,
        *,
        overwrite: bool = False,
    ) -> LoadedPreset:
        """Save the page's choices as the preset ``name`` and make it the loaded one.

        ``fields`` are the ``key=value`` pairs of each tool's non-empty fields, as
        a run collects them. Each tool's pairs must be ones its ``Params`` accepts,
        so that a preset its next run would refuse is never written. What the page
        itself fills when a tool runs (``PAGE_PARAMS``: the wells, the plate and the
        captures) is added for that check only, since a tool's other fields cannot
        be checked without it; the wells it adds when none are selected are a
        stand-in, and the preset keeps none.

        ``wells`` is the *Wells* box's text: kept as typed when it names exactly the
        selection (``A1:D6`` stays ``A1:D6``), else the selection is joined. With
        no selection the preset has no wells.

        Raises:
            PresetError: The name or a value is refused (nothing is written), the
                file cannot be written, or the name is taken and ``overwrite`` is
                false (``PresetExistsError``).
        """
        check_name(name)
        plugins = self._registry().plugins
        params: dict[str, tuple[str, ...]] = {}
        for plugin, pairs in fields.items():
            if not pairs:
                continue
            entry = plugins.get(plugin)
            if entry is None:
                raise PresetError(f"no plugin named {plugin}; nothing saved")
            for pair in pairs:
                key = pair.partition("=")[0]
                if key in NOT_IN_PARAMS.get(plugin, {}):
                    raise PresetError(
                        f"{plugin}: {key} is not kept under params: "
                        f"{NOT_IN_PARAMS[plugin][key]}; nothing saved"
                    )
            try:
                parse_params(
                    entry.plugin.Params, [*pairs, *self._pairs_the_page_adds(plugin)]
                )
            except ParamsError as exc:
                raise PresetError(
                    f"{plugin}: {'; '.join(exc.problems)}; nothing saved"
                ) from None
            params[plugin] = tuple(pairs)
        preset = Preset(
            stand=self._profile.microscope.name,
            # The placeholder is not a plate anyone named: written down, it would be
            # read back as a deliberate choice, and the load that follows (and any
            # later load) would confirm it and open the gate (m5 §11.3).
            plate=self._plate if self._plate_chosen else None,
            wells=self._wells_to_save(wells),
            saved_utc=datetime.now(timezone.utc).replace(microsecond=0),
            params=params,
        )
        path = write_preset(name, preset, self._profile, overwrite=overwrite)
        return self.load_preset(str(path))

    def _pairs_the_page_adds(self, plugin: str) -> list[str]:
        """What a run of ``plugin`` adds to its fields, so that its other fields can be checked alone."""
        if plugin not in PAGE_PARAMS:
            return []
        try:
            return self._page_pairs(plugin)
        except ValueError:
            # `scan-plan` with nothing selected, or a selection on another plate than
            # the registered one: a run is refused for that, a preset is not (it keeps
            # no wells then). Its other fields still need a well to be looked at. The
            # plate's gate is not asked: a preset is not a run (m5 §11.3), and the
            # fields are checked against the pairs the page would fill.
            return ["wells=A1"]

    def _wells_to_save(self, text: str) -> str | None:
        """The ``wells`` a preset keeps: the box's text when it names the selection, else the selection."""
        if not self.selection:
            return None
        typed = text.strip()
        if typed and "\n" not in typed and "\r" not in typed:
            try:
                if parse_wells(self._plate, typed) == self.selection:
                    return typed
            except ValueError:
                pass
        return ",".join(self.selection)

    # --- runs, stop, status ---------------------------------------------------------

    def run(self, plugin: str, params: Sequence[str]) -> None:
        """Run ``plugin`` with ``key=value`` strings, as ``smc run`` would, in one worker thread.

        What ``smc run`` checks before it opens the stand is checked here before a
        thread starts: the plugin, its parameters, the run document and what the
        plugin reads. A refusal there is the ``outcome`` (``<plugin>: <Type>:
        <message>``); nothing is written and no thread starts. What the run does is
        shown by ``status()``. ``smc run`` opens the stand for every plugin, so a
        run with the stand closed is the outcome ``<plugin>: the stand is not
        connected``.

        The page never starts a stage afresh over a checkpoint that ``smc run
        --resume`` could still continue (#252): a stage whose record is ``stopped``
        or ``failed`` with such a checkpoint is the outcome ``<plugin>: <status>
        with a checkpoint: continue it with ...``, and nothing is written. It is
        not an exception, and ``view().blocked`` says the same beforehand.

        Raises:
            RuntimeError: A run is going (``<plugin> is running``), or the page is
                closed.
        """
        self._refuse_if_busy()
        if self._microscope is None:
            self._set_outcome(f"{plugin}: {NOT_CONNECTED}")
            return
        if _needs_plate(plugin) and not self._plate_chosen:
            # The controller's own refusal (FM-169): a click on a disabled button
            # still arrives here, and so does a call that spells `plate=` itself. The
            # stand's being closed is the reason that holds longer, so it came first.
            self._set_outcome(f"{plugin}: {PLATE_NOT_CHOSEN}")
            return
        if plugin == "plate-walls" and self._dry_run:
            # The plugin refuses a dry run too, but its refusal is a `failed` record,
            # and the latest attempt decides the `plate` slot (m5 §11.4): a usable plate
            # would be replaced by a failure. The page refuses first, nothing written,
            # before the parameters are read (FM-169, m5 §12.3).
            self._set_outcome(f"{plugin}: {WALLS_DRY_RUN}")
            return
        profile = self._profile.microscope.name
        dry_run = self._dry_run
        # Read at the click, not when the worker starts: the preset the operator
        # saw loaded is the one the record names (m5 §10.5).
        loaded = self.preset
        preset = None if loaded is None else loaded.file.name
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
                profile=profile,
                dry_run=dry_run,
                resume=False,
            )
            # After `check_ready`, so that a stage that reads what is not usable says
            # that first, as `view().blocked` does. The stand's capabilities are the
            # runner's, in the worker, and come after.
            kept = reachable_checkpoint(
                entry.plugin.name, doc, profile=profile, dry_run=dry_run
            )
        except (PluginError, RunDocumentError) as exc:
            # Text, never the error (FM-98).
            self._set_outcome(f"{plugin}: {type(exc).__name__}: {_one_line(exc)}")
            return
        if kept is not None:
            self._set_outcome(f"{plugin}: {_checkpoint_reason(plugin, kept)}")
            return
        self._start_activity(
            plugin,
            lambda event: self._run_plugin(plugin, entry, parsed, event, preset),
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
        preset: str | None = None,
    ) -> str:
        """A plugin's run on the worker, and its outcome as text."""
        log.info("%s: started from the plate page", plugin)
        started = time.monotonic()
        try:
            record = run_plugin(
                entry.plugin,
                self._stand(),
                parsed,
                self._run_doc,
                on_progress=self._queue.put,
                stop=event,
                plugin_version=entry.version,
                preset=preset,
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
            seconds = time.monotonic() - started
            return f"{plugin} done in {seconds:.1f} s{_assumed_tail(record)}"
        return f"{plugin} stopped"

    def lacks(self, capability: type) -> str | None:
        """Why the stand cannot offer ``capability``, or ``None``: the one text of a disabled control.

        A control the stand cannot honour says so instead of failing when it is
        pressed (ADR-0006, principle 1). With the stand closed it is
        ``NOT_CONNECTED``, whatever the capability.
        """
        return self._lacks(self._microscope, capability)

    @staticmethod
    def _lacks(microscope: Microscope | None, capability: type) -> str | None:
        """``lacks`` of the facade a caller already took into a local."""
        if microscope is None:
            return NOT_CONNECTED
        if microscope.has(capability):
            return None
        return f"this stand has no {capability.__name__}"

    @property
    def connected(self) -> bool:
        """Whether the page holds a stand: what *Connect* and *Disconnect* read."""
        return self._microscope is not None

    def _reading(self) -> _Reading:
        """The registration of a lock-free read: ``with self._reading() as microscope:``.

        The facade comes from the ``with`` and nowhere else, so that a Disconnect
        that dropped it can see the read and wait for it (``_Reading``).
        """
        return _Reading(self)

    def _wait_for_reads(self) -> int:
        """Wait, bounded, for the reads in flight to end; how many were left.

        Called after the facade is dropped and before it is closed, so that
        ``unloadAllDevices`` never runs under a read that took the facade before the
        drop. A read that does not end within ``_READS_WAIT_S`` is not waited on for
        ever (FM-15): the unload goes ahead, the read fails with a line on the page
        (``_handler``), and the Disconnect's outcome says so.
        """
        deadline = time.monotonic() + _READS_WAIT_S
        while self._readers and time.monotonic() < deadline:
            time.sleep(_READS_POLL_S)
        return len(self._readers)

    def _stand(self) -> Microscope:
        """The facade the page holds, for an activity or a capture.

        Raises:
            RuntimeError: ``NOT_CONNECTED``.
        """
        microscope = self._microscope
        if microscope is None:
            raise RuntimeError(NOT_CONNECTED)
        return microscope

    def connect(self) -> None:
        """Open the profile's stand on the worker, as the activity ``connect``.

        Its outcome is ``connect: <stand> connected, <n> devices``, or what the
        opener raised, as one line (a stand that NIS-Elements holds). STOP pressed
        while it opens halts the new facade once it is kept. A connect that ends
        after ``close()`` gave up waiting releases what it opened: a stand is never
        left open with no owner. ``release()`` waits for it once more, bounded, and
        says what became of it, because that late release only runs while the
        process lives.

        Raises:
            RuntimeError: The page is closed, or an activity is going (``<name> is
                running``).
        """
        self._refuse_if_busy()
        if self._microscope is not None:
            self._set_outcome(
                f"connect: {self._profile.microscope.name} is already connected"
            )
            return
        self._start_activity("connect", self._connect)

    def _connect(self, event: threading.Event) -> str:
        """``connect``'s body on the worker; ``_lock`` is never held across the opener or a release."""
        stand = self._profile.microscope.name
        log.info("connect: opening %s from the plate page", stand)
        try:
            microscope = self._opener()
        except Exception as exc:
            # Text, never the error (FM-98): a CoreError from a stand another
            # program holds, a ProfileError from a `[sample]` stand.
            return f"connect: {type(exc).__name__}: {_one_line(exc)}"
        with self._lock:
            closed = self._closed
            if not closed:
                self._microscope = microscope
                # A new facade is not halted, and its light is read again.
                self._stop_note = None
                self._light_lacks_found = None
                self._light_read_raised = None
                self._labels_failure_logged = False
        if closed:
            try:
                microscope.close()
            except Exception as exc:
                return (
                    f"connect: {stand} was opened after the page closed, and its "
                    f"release failed: {type(exc).__name__}: {_one_line(exc)}"
                )
            return f"connect: {stand} was opened after the page closed, and released"
        outcome = f"connect: {stand} connected, {len(microscope.devices)} devices"
        if microscope.dry_run:
            outcome += " (dry run)"
        if event.is_set():
            # STOP was pressed while the stand opened: it found nothing to stop, and
            # the operator asked for a halted stand. Outside the lock (FM-70).
            self.stop()
            outcome += ", and halted: STOP was pressed while it connected"
        return outcome

    def disconnect(self) -> None:
        """Release the stand on the worker, as the activity ``disconnect``, for another program.

        The facade is dropped under the lock **before** the reads in flight are
        waited for and before it is closed (m5 §7.2): a read that took it before
        the drop is in ``_readers`` when the wait looks, and one that starts after
        gets none, so ``unloadAllDevices`` never runs under a read and none reaches
        a closed facade. Nothing is switched off: a light or an autofocus stays as
        the stand keeps it. Refused while a run, a snap, live or a drive action
        goes, so none finds the facade gone under it.

        The outcome never says ``released`` for a stand that is still held: a
        ``close()`` that raised ``MicroscopeBusyError`` unloaded nothing, so the page
        keeps the facade and Disconnect can retry; a failed unload is ``may still be
        held`` (``Microscope.unload_error``); any other error lets the facade go,
        for it may be half unloaded.

        Raises:
            RuntimeError: The page is closed, or an activity is going (``<name> is
                running``).
        """
        self._refuse_if_busy()
        if self._microscope is None:
            self._set_outcome(f"disconnect: {NOT_CONNECTED}")
            return
        self._start_activity("disconnect", self._disconnect)

    def _disconnect(self, event: threading.Event) -> str:
        """``disconnect``'s body on the worker: the drop under the lock, the wait and the close outside it."""
        with self._lock:
            microscope, self._microscope = self._microscope, None
            self._light_lacks_found = None
            self._light_read_raised = None
            self._labels_failure_logged = False
        if microscope is None:  # unreachable: only an activity drops it
            return f"disconnect: {NOT_CONNECTED}"
        stand = self._profile.microscope.name
        in_flight = self._wait_for_reads()
        # The reads that did not end in time: the readers of a facade that is about
        # to be gone (a reader that starts after the drop holds ``None``).
        stuck = frozenset(self._readers) if in_flight else frozenset()
        try:
            microscope.close()
        except MicroscopeBusyError as exc:
            # Nothing was unloaded and close() may be called again (m1 §7): the
            # page still holds the stand, and says so. Not on a closed page: the
            # command's release() has already looked, and a facade put back then
            # would have no owner.
            with self._lock:
                if not self._closed:
                    self._microscope = microscope
            return (
                f"disconnect: MicroscopeBusyError: {_one_line(exc)}; "
                f"the stand is still connected"
            )
        except Exception as exc:
            # close() has logged what to do when unloading fails. The facade may
            # be half unloaded: the page has let go of it for good.
            self._readers.difference_update(stuck)
            return f"disconnect: {type(exc).__name__}: {_one_line(exc)}"
        # The facade is gone: a read that never ends belongs to it, and its token
        # must not make every later Disconnect wait the bound again and report a read
        # that no stand has. Not on a busy close above, which gives the facade (and
        # the read still inside it) back to the page.
        self._readers.difference_update(stuck)
        unload_error = microscope.unload_error
        if unload_error is not None:
            return (
                f"disconnect: {stand} may still be held: its devices did not "
                f"unload ({' '.join(unload_error.split())}); restart smc ui "
                f"before another program takes it"
            )
        outcome = f"disconnect: {stand} released"
        if in_flight:
            outcome += (
                f", with {in_flight} read(s) still in flight after {_READS_WAIT_S:g} s"
            )
        return outcome

    def release(self, unwinding: BaseException | None = None) -> None:
        """Release the stand the page holds, if any: the command's, after ``close()``.

        Idempotent, and it does nothing on a page that holds no stand. It releases
        a facade handed to the constructor too.

        A ``connect`` that ``close()`` gave up on is waited for, at most
        ``_RELEASE_WAIT_S``, because its late release of what it opens can only run
        while the process lives (m5 §7.2): what became of it is added to
        ``close_lines``, and ``close_ended`` is whether it ended. Then the facade is
        dropped, the reads in flight are waited for, and the facade is ended as the
        ``with Microscope.open(...)`` block did: ``unwinding`` is the exception the
        command is leaving with (``None`` on the normal path), so that a Ctrl-C is
        noted with the motions still tracked (m2 §11.11 E6), which a second Ctrl-C
        during ``close()`` lost when ``release()`` called ``close()`` directly. The
        waits and the drop are one ``try`` whose ``finally`` ends the facade, so a
        second Ctrl-C in a wait still unloads the stand and is not swallowed (FM-187).

        Raises:
            RuntimeError: ``release() comes after close()``: before it, the facade
                could be pulled from under an activity.
            MicroscopeBusyError: What ``Microscope.close()`` raises; the page has
                let go of the facade all the same.
            HardwareError: As ``MicroscopeBusyError``.
        """
        if not self._closed:
            raise RuntimeError("release() comes after close()")
        microscope: Microscope | None = None
        try:
            if self._gave_up == "connect":
                self._wait_for_the_connect()
            with self._lock:
                microscope, self._microscope = self._microscope, None
            self._wait_for_reads()
        finally:
            # In a `finally`: a second Ctrl-C lands in one of the two waits above, and
            # the facade, dropped from the page and held by this local only, would
            # never be unloaded (FM-71, FM-49). A Ctrl-C before the drop finds the
            # page still holding it.
            if microscope is None:
                with self._lock:
                    microscope, self._microscope = self._microscope, None
            if microscope is not None:
                microscope.__exit__(
                    None if unwinding is None else type(unwinding),
                    unwinding,
                    None if unwinding is None else unwinding.__traceback__,
                )

    def _wait_for_the_connect(self) -> None:
        """The wait of ``release()`` for the ``connect`` that ``close()`` gave up on, and a closing line for it.

        ``join()`` returns at once when the connect ended meanwhile. The line is the
        connect's own outcome when it ended (it released what it opened, or the
        opener raised), else that its end is not known: the process ends now.
        """
        stand = self._profile.microscope.name
        ended = self.join(_RELEASE_WAIT_S)
        if ended:
            with self._lock:
                line = self._outcome or f"connect: {stand} ended without an outcome"
        else:
            line = (
                f"connect: {stand} is still opening after "
                f"{_CLOSE_WAIT_S + _RELEASE_WAIT_S:g} s: the process ends now, "
                f"and whether the stand is left connected is not known"
            )
        self.close_lines = (*self.close_lines, line)
        self.close_ended = ended

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
        refused while it goes, and it is never ended for them (m5 §5.2). A drive
        action is not refused: the live worker runs it between two frames (m5 §6.3).

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
        turn began, so a camera slower than that runs at its own rate, and the wait
        is on the event, so STOP ends it within one period (FM-83). A frame in flight
        when the event is set completes and is kept. Nothing is logged for a frame
        (FM-62, FM-148): the start, a drive action and the outcome are the log's.

        At the top of each turn, before the frame, the worker takes the drive action
        that waits (m5 §6.3) and runs it outside the lock: one thread on the
        devices, and the next frame shows where it went. A drive action that fails
        is its outcome, and live goes on; only a frame that fails ends live.
        Whatever ends the loop, the event is set **first**, then the action still
        waiting is taken under the lock: a failed frame ends the loop without the
        event, and a ``_drive`` that took the lock after the drop would otherwise
        keep an action that no loop takes. ``_drive`` reads the event under the
        lock, so an action is either taken by the loop, run or dropped at the end,
        or refused. **A switch-off is run** (``light off``, ``autofocus
        disengage``, #295), after the lock is released and as any drive action runs;
        every other action is dropped and never sent (FM-88: a move, a turn or an
        engage must not be sent after a STOP). A switch-off is the safe call the
        facade never refuses, also halted, and it sends no move, so what makes a
        drop right for the others does not apply: dropping it left the light on
        after a STOP. STOP does not wait for it (``stop()`` joins nothing, FM-70);
        ``close()`` waits for it as for a frame. The outcome ends with what the end
        did: ``; <the switch-off's outcome>`` or ``; <name> was not run``.
        """
        log.info("live: started from the plate page")
        ran: str | None = None
        try:
            outcome = self._live_frames(event)
        finally:
            event.set()
            with self._lock:
                waiting, self._drive_waiting = self._drive_waiting, None
                if waiting is not None and waiting.switch_off:
                    # Held in the same critical section as the take, so a capture
                    # never finds the action neither waiting nor running.
                    self._drive_running = waiting.name
            if waiting is not None and waiting.switch_off:
                ran = self._run_handed(waiting)
        if waiting is None:
            return outcome
        if ran is not None:
            return f"{outcome}; {ran}"
        return f"{outcome}; {waiting.name} was not run"

    def _run_handed(self, handed: _Handed) -> str:
        """Run an action handed to the live worker, outside ``_lock``; its outcome text, kept and logged.

        Whoever took ``handed`` from ``_drive_waiting`` set ``_drive_running`` in
        the same critical section; it is cleared here whatever the action does, so
        that an escaping ``BaseException`` never leaves a capture refused for ever.
        A drive action is logged (a frame never is, FM-148), after its outcome is
        kept (FM-62).
        """
        try:
            done = handed.run()
        except BaseException:
            with self._lock:
                self._drive_running = None
            raise
        with self._lock:
            self._drive_running = None
            self._outcome = done
        log.info("%s", done)
        return done

    def _live_frames(self, event: threading.Event) -> str:
        """``_live``'s loop: a waiting drive action, then a frame, then the rest of the period.

        Its text is live's own, then ``; <outcome of the last action it ran>`` when
        there was one (#295): the earlier ones were on the strip while live went,
        and the log has every one.
        """
        started = time.monotonic()
        frames = 0
        last: str | None = None
        while not event.is_set():
            began = time.monotonic()
            with self._lock:
                waiting = None
                if not event.is_set():
                    waiting, self._drive_waiting = self._drive_waiting, None
                    if waiting is not None:
                        self._drive_running = waiting.name
            if waiting is not None:
                last = self._run_handed(waiting)
                if event.is_set():
                    # STOP during the action: the stand is halted, and a frame now
                    # would end live with the halt's refusal instead of the stop.
                    break
            taken = self._take_frame()
            if isinstance(taken, str):
                return _then(f"live ended after {frames} frames: {taken}", last)
            frames += 1
            # Not on the progress queue: its only reader is a session's tick, and a
            # live view outlives every session, so with no tab open it would grow by
            # ten items a second for as long as the camera runs. The latest count is
            # all ``status()`` ever keeps of a queue.
            with self._lock:
                self._progress = Progress(
                    plugin="live", done=frames, total=None, message="frames"
                )
            event.wait(max(0.0, LIVE_PERIOD_S - (time.monotonic() - began)))
        return _then(
            f"live ended: {frames} frames in {time.monotonic() - started:.1f} s", last
        )

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
            microscope = self._stand()
            camera = microscope.require(Camera)
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
            state = microscope.state()
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

    # --- drive (m5 §6.3) ---------------------------------------------------------

    def jog_xy(self, dx_um: float, dy_um: float) -> None:
        """``smc stage jog dx dy``: one activity per click, or between two frames of live.

        Raises:
            RuntimeError: Another activity is going (``<name> is running``), the
                page is closed, or live is ending or holds a waiting action.
        """
        self._drive(
            "stage jog",
            XYStage,
            lambda: drive.jog_xy(self._stand(), dx_um, dy_um),
        )

    def jog_z(self, dz_um: float) -> None:
        """``smc z jog dz``; the layer's interlock suspends and relocks a locked autofocus (FM-12).

        Raises:
            RuntimeError: As ``jog_xy``.
        """
        self._drive("z jog", ZStage, lambda: drive.jog_z(self._stand(), dz_um))

    def objective(self, label: str, *, confirm: bool) -> None:
        """``smc objective set label``: the turret turns only with ``confirm`` (FM-14).

        The page passes the ``confirm`` its *Turn the turret* button gives; without
        it the layer refuses and the refusal is the outcome.

        Raises:
            RuntimeError: As ``jog_xy``.
        """
        self._drive(
            "objective set",
            ObjectiveTurret,
            lambda: drive.set_objective(self._stand(), label, confirm=confirm),
        )

    def autofocus(self, on: bool) -> None:
        """``smc autofocus engage`` or ``disengage``; switching off runs while halted.

        Raises:
            RuntimeError: As ``jog_xy``.
        """
        if on:
            self._drive(
                "autofocus engage",
                Autofocus,
                lambda: drive.engage_autofocus(self._stand()),
            )
        else:
            self._drive(
                "autofocus disengage",
                Autofocus,
                lambda: drive.disengage_autofocus(self._stand()),
                switch_off=True,
            )

    def light(self, on: bool) -> None:
        """``smc light on`` or ``off``; on is held back by ``light_lacks()``, off never is.

        A light whose on/off smc does not know gets the facade's own error for an
        off, which says to use the shutter.

        Raises:
            RuntimeError: As ``jog_xy``.
        """
        self._drive(
            "light on" if on else "light off",
            LightSource,
            lambda: drive.switch_light(self._stand(), on),
            lacks=self.light_lacks if on else None,
            switch_off=not on,
        )

    def objective_labels(self) -> tuple[str, ...]:
        """The turret's labels in position order; ``()`` with the stand closed, without a turret or when the read fails.

        A failure is logged at WARNING **once** per failing streak (#295): the tick
        reads again while the choices are empty, and a line a tick would fill the
        log and block the loop that handles STOP (FM-62, FM-148). A read that
        succeeds ends the streak, and so do Connect and Disconnect (a new stand
        has not failed yet).
        """
        with self._reading() as microscope:
            if microscope is None:
                return ()
            turret = microscope.get(ObjectiveTurret)
            if turret is None:
                return ()
            try:
                labels = tuple(turret.labels())
            except Exception as exc:
                if not self._labels_failure_logged:
                    self._labels_failure_logged = True
                    log.warning(
                        "objective: its positions could not be read: %s: %s",
                        type(exc).__name__,
                        _one_line(exc),
                    )
                return ()
            self._labels_failure_logged = False
            return labels

    def light_lacks(self) -> str | None:
        """Why the light cannot be switched on from the page, or ``None``.

        ``lacks(LightSource)``, else a light source whose ``is_on()`` reads
        ``None`` (a ``State`` device such as the demo's ``LED``). A read that
        raises is ``None``: the click then says what fails. The answer is kept
        after the first read that does not raise, since a device's type does not
        change while the stand is held; Connect and Disconnect forget it. It
        takes no lock: the page calls it.

        One read a tick: ``status()`` reads ``is_on()`` on every tick and keeps
        the answer of a read that did not raise (``_note_light_read``), so this
        reads for itself only when no ``status()`` has read this stand's light,
        and not at all for a stand whose last ``status()`` read raised.
        """
        with self._reading() as microscope:
            lack = self._lacks(microscope, LightSource)
            if lack is not None or microscope is None:
                return lack
            found = self._light_lacks_found
            if found is not None:
                return found[0]
            raised = self._light_read_raised
            if raised is not None and raised() is microscope:
                return None
            light = microscope.get(LightSource)
            if light is None:  # unreachable once lacks() said it is there
                return None
            try:
                answer = _light_lacks_text(light, light.is_on())
            except Exception:
                return None
            if self._microscope is microscope:
                # Not kept for a facade dropped during the read: a Connect that ended
                # meanwhile has cleared the answer for its own stand.
                self._light_lacks_found = (answer,)
            return answer

    def _note_light_read(
        self,
        microscope: Microscope,
        light: LightSource,
        on: bool | None,
        *,
        raised: bool,
    ) -> None:
        """What ``status()``'s own read of ``microscope``'s light tells ``light_lacks()`` (one read a tick).

        A read that did not raise gives the answer, kept as ``light_lacks()`` keeps
        its own; one that raised makes ``light_lacks()`` answer ``None`` without a
        second read, until a read succeeds. Nothing is kept for a facade the page
        let go of during the read: its Connect or Disconnect has cleared both for
        its own stand.
        """
        if self._microscope is not microscope:
            return
        if raised:
            self._light_read_raised = weakref.ref(microscope)
            return
        self._light_read_raised = None
        if self._light_lacks_found is None:
            try:
                self._light_lacks_found = (_light_lacks_text(light, on),)
            except Exception:
                # ``device()`` failing is no reason to fail the tick: the answer is
                # not kept, and ``light_lacks()`` reads for itself, as it always did.
                return

    def jog_limits_um(self) -> tuple[float, float]:
        """The profile's XY and Z jog limits, in µm: what the page's steps are bounded by.

        The numbers are the ones of a ``Safety`` built from the profile's two,
        the class the facade enforces a jog with and the one that owns "no
        ``max_z_jog_um`` means the layer's assumed default": this page repeats no
        rule, and a test checks that what it offers the facade accepts. Built from
        the profile and not read from the facade, so the same with the stand closed
        (the page builds its steps before Connect, m5 §7.3).
        """
        safety = self._profile.safety
        guard = Safety(max_jog_um=safety.max_jog_um, max_z_jog_um=safety.max_z_jog_um)
        return (guard.max_jog_um, guard.max_z_jog_um)

    def _drive(
        self,
        name: str,
        capability: type,
        action: Callable[[], DriveResult],
        *,
        lacks: Callable[[], str | None] | None = None,
        switch_off: bool = False,
    ) -> None:
        """What every drive action does, in m5 §6.3's order.

        The page is closed; another activity than this page's live goes; the stand
        lacks the capability (an outcome, nothing starts or waits); live goes (the
        action is handed to its worker); else the action is an activity of its own.
        ``lacks`` is a second reason after ``lacks(capability)``. ``switch_off`` is
        ``True`` for ``light off`` and ``autofocus disengage`` only: handed to the
        live worker, such an action is still run when live ends (#295), where any
        other is dropped.

        Raises:
            RuntimeError: ``the page is closed``, ``<name> is running``, ``live is
                ending`` or ``<name> is waiting``.
        """
        if self._closed:
            raise RuntimeError("the page is closed")
        current = self._current_run
        live = current is not None and current[1] is self._live_event
        if current is not None and not live:
            raise RuntimeError(f"{current[0]} is running")
        reason = self.lacks(capability)
        if reason is None and lacks is not None:
            reason = lacks()
        if reason is not None:
            self._set_outcome(f"{name}: {reason}")
            return

        def outcome() -> str:
            return self._drive_outcome(name, action)

        if live:
            with self._lock:
                now = self._current_run
                if now is None or now[1] is not self._live_event or now[1].is_set():
                    raise RuntimeError("live is ending")
                if self._drive_waiting is not None:
                    # Refused, never queued: a queue would move the stage after the
                    # operator stopped clicking.
                    raise RuntimeError(f"{self._drive_waiting.name} is waiting")
                self._drive_waiting = _Handed(name, outcome, switch_off)
            return
        self._start_activity(name, lambda event: outcome())

    @staticmethod
    def _drive_outcome(name: str, action: Callable[[], DriveResult]) -> str:
        """Run ``action``; its outcome text, an exception kept only as text (FM-98)."""
        try:
            result = action()
        except Exception as exc:
            return f"{name}: {type(exc).__name__}: {_one_line(exc)}"
        return f"{name}: {result.text}"

    def stop(self) -> None:
        """Ask the run to stop, then stop the stand; say what the stop did, never "stopped" by default.

        The event is set first, so the plugin leaves its loop as the stand halts,
        and the log line comes after the stop (FM-62). It takes no lock the worker
        holds (FM-70) and never raises for a stop that is refused or fails
        (FM-117): the outcome is ``stop_note``. STOP with nothing running still
        halts the stand. With the stand closed it sets the event all the same (a
        ``connect`` that is opening halts the stand once it holds it), and the
        note says there was nothing to stop.

        STOP is a read that never waits (m5 §7.2): it registers, sends the stop to
        the facade it took and unregisters, so a Disconnect that began meanwhile
        waits for it and the stop reaches loaded devices. One that finds the facade
        dropped says there was nothing to stop, and ``Microscope.close()`` stops
        what still moves before it unloads.
        """
        current = self._current_run
        if current is not None:
            current[1].set()
        prefix = "" if current is None else f"{current[0]} told to stop; "
        with self._reading() as microscope:
            try:
                if microscope is None:
                    note = f"{prefix}{NOT_CONNECTED}: nothing to stop"
                else:
                    microscope.stop()
                    note = f"{prefix}{_STOP_ACCEPTED}"
            except StopRefusedError as exc:
                note = f"{prefix}halted, but {_one_line(exc)}"
            except HardwareError as exc:
                note = f"{prefix}the stop failed: {_one_line(exc)}"
        # Kept before it is logged: the stop is sent, and a handler that blocks (a
        # Windows console in QuickEdit) must not keep the page from its note (FM-148).
        self._stop_note = note
        log.info("stop: %s", note)

    def resume(self) -> None:
        """Clear the halt; refused while a run is going.

        Raises:
            RuntimeError: A run is going (``<plugin> is running``), or the stand is
                not connected (``NOT_CONNECTED``).
        """
        current = self._current_run
        if current is not None:
            raise RuntimeError(f"{current[0]} is running")
        self._stand().resume()
        self._stop_note = None

    def status(self) -> PageStatus:
        """The run and the stand now: the progress queue is drained and the stand is read.

        The latest ``Progress`` is kept, so that every session sees the same. The
        stand's read takes no lock and tolerates a failed device: what failed is in
        ``problems``. The facade is read once; when a ``disconnect`` dropped it
        during the read, the status is the closed stand's, so a read of a released
        stand is never told as its fault (``another program holding it``).
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
        stand = self._profile.microscope.name
        stop_refused = tuple(
            role.value for role in self._profile.safety.stop_refused_by
        )
        # The whole read of the facade, from ``state()`` to the status built for it, is
        # registered: a Disconnect waits for it before it unloads (m5 §7.2).
        with self._reading() as microscope:
            if microscope is not None:
                state = microscope.state()
                problems = list(state.errors)
                light_on: bool | None = None
                light = microscope.get(LightSource)
                if light is not None:
                    try:
                        light_on = light.is_on()
                    except Exception as exc:
                        problems.append(
                            f"light: {type(exc).__name__}: {_one_line(exc)}"
                        )
                        self._note_light_read(microscope, light, None, raised=True)
                    else:
                        self._note_light_read(microscope, light, light_on, raised=False)
                autofocus = (
                    describe_autofocus(state)
                    if microscope.has(Autofocus)
                    else "no autofocus"
                )
                if self._microscope is microscope:
                    return PageStatus(
                        stand=stand,
                        dry_run=self._dry_run,
                        running=running,
                        progress=progress,
                        outcome=outcome,
                        stop_note=self._stop_note,
                        xy_um=None
                        if state.xy is None
                        else (state.xy.x_um, state.xy.y_um),
                        z_um=state.z_um,
                        objective=state.objective,
                        autofocus=autofocus,
                        light_on=light_on,
                        moving=tuple(state.moving),
                        halted=state.halted,
                        stop_refused=stop_refused,
                        problems=tuple(problems),
                    )
        return PageStatus(
            stand=stand,
            dry_run=self._dry_run,
            running=running,
            progress=progress,
            outcome=outcome,
            stop_note=self._stop_note,
            xy_um=None,
            z_um=None,
            objective=None,
            autofocus="?",
            light_on=None,
            moving=(),
            halted=False,
            stop_refused=stop_refused,
            problems=(),
            connected=False,
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
        that serves the page owns it, through ``release()``) and never raises for a
        refused stop. What it tells the operator is ``close_lines``: what the stop
        did and how the run ended, never "stopped" by default (FM-117). A Ctrl-C
        during the wait is not caught: the stop was sent before it (FM-71).

        An activity that does not end within ``_CLOSE_WAIT_S`` is kept by name
        (``_gave_up``): a ``connect`` is waited for again by ``release()``, which
        adds a line for it to ``close_lines`` (m5 §7.2). A ``connect`` that ends
        after ``close()`` releases what it opened.
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
            self._gave_up = plugin
            with self._lock:
                # Not over an outcome the worker has just written: ``release()``
                # tells a connect's outcome as a closing line.
                if self._current_run is not None:
                    self._outcome = f"{plugin} did not end within {_CLOSE_WAIT_S:.0f} s"
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

    def _refuse_if_moving(self) -> None:
        """Refuse a capture while the stage is about to move or moves (m5 §6.3, #295).

        A capture records the stage's position, so a position read while a move
        waits behind a frame, or runs on the live worker, is the one *before* the
        move. Refused: an activity other than this page's live (a run, a snap, a
        drive action of its own, ``connect``, ``disconnect``: ``goto-well`` and
        every plugin that moves the stage are runs), an action running on the live
        worker (``_drive_running``) and one waiting for it (``_drive_waiting``). A
        live with nothing handed to it does not refuse: a frame moves nothing.

        The three are read under ``_lock``, never across a device call, and the
        check is not atomic with the read that follows: the window is one
        ``position_um()``, for an operator clicking in two tabs.

        Raises:
            RuntimeError: ``<name> is running`` or ``<name> is waiting``.
        """
        with self._lock:
            current = self._current_run
            running = self._drive_running
            waiting = self._drive_waiting
        if current is not None and current[1] is not self._live_event:
            raise RuntimeError(f"{current[0]} is running")
        if running is not None:
            raise RuntimeError(f"{running} is running")
        if waiting is not None:
            raise RuntimeError(f"{waiting.name} is waiting")

    def _refuse_if_busy(self) -> None:
        if self._closed:
            raise RuntimeError("the page is closed")
        current = self._current_run
        if current is not None:
            raise RuntimeError(f"{current[0]} is running")

    def _set_outcome(self, outcome: str) -> None:
        with self._lock:
            self._outcome = outcome

    def clear_outcome(self) -> None:
        """Forget the last activity's outcome, when no activity goes (m5 §8.8.3).

        The outcome stays on the strip until the next activity replaces it, so that
        ``cells: ... run scan again`` was still there after the chain had been run
        from a terminal and the page refreshed. Only the *Refresh* button calls
        this: ``view()``'s redraw is also what the end of a run triggers, and must
        not drop the outcome it is about to show. The stop note is not touched.
        """
        with self._lock:
            if self._current_run is None:
                self._outcome = None

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
                # What the stand suggests is never folded under *Advanced*, even for a
                # key `KEY_PARAMS` does not name.
                suggestion = self._suggested.get(name)
                suggested = () if suggestion is None else suggestion.pairs
                keys += [k for k, _value in suggested if k in by_name and k not in keys]
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
                        suggested=suggested,
                        suggested_why=None if suggestion is None else suggestion.why,
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
        run_dir = self.run_doc.parent
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
        checklist = self._checklist(doc)
        blocked = self._blocked(doc)
        cards = {
            tool.name: _card(
                doc.stages[tool.name],
                stale=doc.is_stale(tool.name),
                run_dir=run_dir,
            )
            for tool in self.tools()
            if tool.name in doc.stages
        }
        cards = self._with_sample_source(doc, cards)
        return PlateView(
            plate=self._plate,
            nominal=nominal,
            wells=tuple(wells),
            labels=() if density is None else density.labels,
            route=stops,
            checklist=checklist,
            blocked=blocked,
            problems=(*problems, *self._plugin_problems(), *self._suggest_problems),
            cards=cards,
            next_step=self._next_step(checklist, blocked, cards),
            replaces=self._replace_notes(doc),
        )

    def _replace_notes(self, doc: RunDocument) -> dict[str, str]:
        """What a Run of each tool that provides ``plate`` would replace, while the slot is usable (m5 §11.4).

        For the ``plate`` slot only: a card already says the last result of every
        other slot, and a note under every tool would teach the operator to read
        none of them. The slot's problem, if its result does not validate, is
        ``view()``'s line already; it is not said twice.
        """
        result = self._slot(doc, "plate", PlateResult, [])
        record = doc.provider("plate")
        if result is None or record is None:
            return {}
        plugins = self._registry().plugins
        # The second sentence of the note says the captures stay: true of the tools
        # that run from them (their pairs hold `refs`), false of plate-finding (m5 §12.7).
        return {
            tool.name: replace_note(
                result,
                record.started_utc,
                keeps_captures="refs" in PAGE_PARAMS.get(tool.name, ()),
            )
            for tool in self.tools()
            if plugins[tool.name].plugin.provides == "plate"
        }

    # --- the next step (m5 §8.3) ---------------------------------------------------

    def _check_suggestions(
        self, profile: Profile
    ) -> tuple[dict[str, _Suggestion], tuple[str, ...]]:
        """The profile's ``[suggest]`` entries that name a registered plugin and its parameters.

        The profile checks the shape of the pairs and imports no plugin; this is
        where the plugin and its keys are checked, once. An entry that fails is a
        line of ``view().problems`` and is ignored whole, so that no step offers a
        value the plugin would refuse by key. A value is not checked: a value the
        plugin refuses is that run's own ``ParamsError``, which names the
        profile's mistake when the step is run.
        """
        if not profile.suggest:
            return {}, ()
        plugins = self._registry().plugins
        found: dict[str, _Suggestion] = {}
        problems: list[str] = []
        for name, entry in profile.suggest.items():
            where = f"profile [suggest.{name}]"
            registered = plugins.get(name)
            if registered is None:
                problems.append(f"{where}: no plugin named {name}")
                continue
            fields = registered.plugin.Params.model_fields
            pairs: list[tuple[str, str]] = []
            wrong: list[str] = []
            for item in entry.params:
                key, _separator, value = item.partition("=")
                if key not in fields:
                    wrong.append(
                        f"{where}: {key} is not a parameter of {name} "
                        f"(keys: {', '.join(fields)})"
                    )
                elif key in PAGE_PARAMS.get(name, ()):
                    wrong.append(f"{where}: {key} is filled by the page")
                pairs.append((key, value))
            if wrong:
                problems += wrong
            else:
                found[name] = _Suggestion(tuple(pairs), entry.why)
        return found, tuple(problems)

    def _next_step(
        self,
        checklist: Sequence[CheckItem],
        blocked: dict[str, str],
        cards: dict[str, ResultCard],
    ) -> NextStep:
        """The tool the run document needs next, with the values to run it and why each (m5 §8.3).

        Computed from this ``view()``'s read, never by the tick. The pairs are the
        call the page's button makes and the command names, so what the operator
        reads is what runs. The step never reads the tool's form.
        """
        item = next_slot(checklist)
        if item is None:
            files = tuple(entry for card in cards.values() for entry in card.files)
            return NextStep(
                plugin=None,
                params=(),
                why=("every stage of the workflow is done; what it wrote:", *files),
                command="",
                blocked=None,
            )
        plugins = self._registry().plugins
        providers = [
            tool
            for tool in self.tools()
            if plugins[tool.name].plugin.provides == item.slot
        ]
        if not providers:
            return NextStep(
                plugin=None,
                params=(),
                why=(
                    f"no registered plugin provides {item.slot}; "
                    "smc run --list shows the plugins",
                ),
                command="",
                blocked=None,
            )
        tool, others = providers[0], providers[1:]
        why = [
            item.reason
            if item.state == "unusable" and item.reason
            else f"the run document has no {item.slot} yet"
        ]
        pairs: list[str] = []
        stuck: str | None = None
        if tool.name == "plate-registration":
            pairs, stuck = self._registration_step(why)
        elif tool.name == "scan-plan":
            try:
                pairs = self.page_params("scan-plan")
            except ValueError as exc:
                # `page_params` refuses an empty selection, and a selection on another
                # plate than the registered one: the first says what to do, the second
                # says what is wrong. Text only (FM-98).
                stuck = SELECT_WELLS if not self.selection else str(exc)
            else:
                why.append(
                    f"wells: the {len(self.selection)} wells selected on the map"
                )
        if (
            stuck is None
            and self._dry_run
            and (
                tool.name == "scan"
                or (
                    "scan" in plugins[tool.name].plugin.reads
                    and "scan" in cards
                    and cards["scan"].dry_run
                )
            )
        ):
            # By the plugin's name, not by a capability: `scan-plan` also requires a
            # camera and computes the same in a dry run. A dry run may read a real
            # scan, so `cells` is stuck only when the scan it would read was a dry run.
            stuck = DRY_RUN_NO_FRAME
        # The preset wins over the profile's suggestion key by key (m5 §10.5): a
        # suggestion the preset gives another value for is not offered at all.
        loaded = self.preset
        from_preset = (
            [] if loaded is None else list(loaded.fields.get(tool.name, {}).items())
        )
        suggestion = self._suggested.get(tool.name)
        if suggestion is not None:
            overridden = {key for key, _value in from_preset}
            given = [
                f"{key}={value}"
                for key, value in suggestion.pairs
                if key not in overridden
            ]
            if given:
                pairs += given
                why.append(
                    f"{' '.join(given)}: suggested by the profile "
                    f"{self._profile.microscope.name}: {suggestion.why}"
                )
        if loaded is not None and from_preset:
            named = [f"{key}={value}" for key, value in from_preset]
            pairs += named
            why.append(f"{' '.join(named)}: from the preset {loaded.file.name}")
        why += [f"or {other.name}: {_flat(other.summary)}" for other in others]
        source = self._profile.source
        command = command_line(
            tool.name,
            pairs,
            profile=self._profile.microscope.name if source is None else str(source),
            run_doc=self.run_doc,
            dry_run=self._dry_run,
            preset=None if loaded is None else str(loaded.file.path),
        )
        # What the run document or the stand says first wins: it is the reason that holds.
        return NextStep(
            plugin=tool.name,
            params=tuple(pairs),
            why=tuple(why),
            command=command,
            blocked=blocked.get(tool.name) or stuck,
        )

    def _registration_step(self, why: list[str]) -> tuple[list[str], str | None]:
        """The pairs of ``plate-registration`` and what it waits for, with a line of ``why`` for them.

        While the plate is the placeholder nothing is offered, whatever the captures
        are (m5 §11.3): the plugin's own default (``plate=96-well``) must never be
        what a click runs. Then the captures made on this page win: they are
        measured. With none, a stand whose profile has a ``[sample]`` of this plate
        registers it from the sample's own geometry, which is known and not
        measured, and the line says so. With neither, the plate is found by
        centring wells, which only the operator can do.
        """
        if not self._plate_chosen:
            return [], PLATE_NOT_CHOSEN
        if self.refs or self.walls:
            why.append(
                f"plate, refs, walls: the {len(self.refs)} centres and "
                f"{len(self.walls)} walls captured on this page"
            )
            return self._registration_pairs(self.refs, self.walls), None
        sample = self._profile.sample
        if sample is None:
            return [], CAPTURE_WELLS
        if sample.plate != self._plate:
            return [], (
                f"the profile's sample is a {sample.plate} plate: "
                f"choose {sample.plate} under plate-registration"
            )
        # Imported here: `smc.testing` is a stand-in for a stand, reached only by a
        # profile that names a sample (m5 §8.2).
        from smc.testing.sample_stand import sample_refs

        refs = sample_refs(sample)
        why.append(_sample_refs_text(refs))
        return self._registration_pairs(refs, ()), None

    def _with_sample_source(
        self, doc: RunDocument, cards: dict[str, ResultCard]
    ) -> dict[str, ResultCard]:
        """``cards``, with the plate card's last ``assumed`` line saying the references are the sample's (m5 §8.8.4).

        Afterwards the card read ``method: references``, ``residual_um: 0`` and
        nothing about where the references came from. The line is added when the
        profile has a ``[sample]`` and the record ended ``done`` on this stand with
        the sample's own references (``_are_the_refs``): captures an operator made
        are measured, read differently, and never match. ``smc.testing`` is reached
        only for a profile that names a sample, as in the step (FM-176).
        """
        sample = self._profile.sample
        card = cards.get("plate-registration")
        record = doc.stages.get("plate-registration")
        if sample is None or card is None or record is None:
            return cards
        if (
            record.status is not StageStatus.done
            or record.profile != self._profile.microscope.name
        ):
            return cards
        from smc.testing.sample_stand import sample_refs

        refs = sample_refs(sample)
        if not _are_the_refs(record.params.get("refs"), refs):
            return cards
        said = replace(card, assumed=(*card.assumed, _sample_refs_text(refs)))
        return {**cards, "plate-registration": said}

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
            profile=self._profile.microscope.name,
            dry_run=self._dry_run,
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
        """Why each registered tool cannot run now; a tool that can is absent.

        Every tool needs the stand, as ``smc run`` opens it for every plugin, so
        with the stand closed ``NOT_CONNECTED`` is every tool's reason, ahead of
        any other. Otherwise the reasons, in the order the first that applies
        wins: the placeholder plate, for a tool that takes the plate (m5 §11.3);
        what the tool reads, what it requires of the stand, and last a
        checkpoint that ``smc run --resume`` could continue (#252), which ``run()``
        refuses to drop. A tool that cannot run at all says so before it offers a
        way to continue.

        ``plate-walls`` has its own words in this order (m5 §12.3): a dry run after
        the plate (the tool images the walls, which a dry run does not), the page's
        own sentence when the document has no calibration at all, and the count of
        centres after what the stand requires.
        """
        microscope = self._microscope
        if microscope is None:
            return {name: NOT_CONNECTED for name in self._registry().plugins}
        blocked: dict[str, str] = {}
        profile = self._profile.microscope.name
        dry_run = self._dry_run
        for name, entry in self._registry().plugins.items():
            reason = (
                PLATE_NOT_CHOSEN
                if _needs_plate(name) and not self._plate_chosen
                else None
            )
            if reason is None and name == "plate-walls" and dry_run:
                reason = WALLS_DRY_RUN
            if reason is None:
                reason = next(
                    (
                        r
                        for r in (self._unusable(doc, s) for s in entry.plugin.reads)
                        if r
                    ),
                    None,
                )
                if (
                    reason is not None
                    and name == "plate-walls"
                    and doc.provider("camera-calibration") is None
                ):
                    # The document's own sentence is for a terminal, and the tool is
                    # one region up: a calibration that exists keeps the document's
                    # words (another stand, a dry run, stale).
                    reason = WALLS_NEED_CALIBRATION
            if reason is None:
                reason = next(
                    (
                        lack
                        for lack in (
                            self._lacks(microscope, cap)
                            for cap in entry.plugin.requires
                        )
                        if lack
                    ),
                    None,
                )
            if reason is None and name == "plate-walls":
                reason = walls_centres_problem(len(self.refs))
            if reason is None:
                kept = reachable_checkpoint(name, doc, profile=profile, dry_run=dry_run)
                if kept is not None:
                    reason = _checkpoint_reason(name, kept)
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

    def _initial_plate(self, doc: RunDocument) -> tuple[str, bool]:
        """The plate the page starts with, and whether anything named it (m5 §11.3).

        In order: the plate of the latest provider of ``plate`` (what the operator was
        working on, whatever its status), the profile's ``[sample]``, which is what
        the stand's camera images, and last the ``96-well`` placeholder, which
        nothing named: the page then asks.
        """
        named = self._plate_named_by(doc)
        if named is not None:
            return named, True
        sample = self._profile.sample
        if sample is not None and sample.plate in useq.registered_well_plate_keys():
            return sample.plate, True
        return "96-well", False

    def _plate_named_by(self, doc: RunDocument) -> str | None:
        """The plate the latest provider of ``plate`` holds, read as data (FM-125), or ``None``.

        A usable result is what the record found: its ``plate`` is the truth. Without
        one (the run failed or was stopped) the record's ``plate`` parameter is what
        it was asked for, which is the plate on the stage for ``plate-registration``
        and ``plate-walls`` only: ``plate-finding``'s is the plate its survey was
        *sized* for (a default of 96-well), and a failed survey found no plate. Read
        as the plate on the stage, it opened the page on the wrong plate and said it
        had been chosen (adversarial review of #332).
        """
        record = doc.provider("plate")
        if record is None:
            return None
        found = self._slot(doc, "plate", PlateResult, [])
        if found is not None and found.plate in useq.registered_well_plate_keys():
            return str(found.plate)
        if record.plugin in _PLATE_PARAM_PLUGINS:
            return self._plate_of(record)
        return None

    @staticmethod
    def _plate_of(record: StageRecord | None) -> str | None:
        """The plate ``record`` was run for, read as data (FM-125): a name ``useq`` knows, else ``None``."""
        if record is None:
            return None
        plate = record.params.get("plate")
        if isinstance(plate, str) and plate in useq.registered_well_plate_keys():
            return plate
        return None

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
        # The captures come from the record the plate does: the latest provider of
        # ``plate`` (m5 §11.3), not the last registration by name, so that a newer
        # wall registration's centres are the ones the page holds. A record whose
        # plate the page did not take is no source: its wells are named for a plate
        # this page cannot build.
        provider = doc.provider("plate")
        if provider is None or self._plate_of(provider) != self._plate:
            return
        if provider.plugin == "plate-registration":
            self.refs = self._captured(provider, "refs", WellRef)
            self.walls = self._captured(provider, "walls", WallTouch)
        elif provider.plugin == "plate-walls":
            # It takes centres and measures the walls itself: it has no walls to give.
            self.refs = self._captured(provider, "refs", WellRef)

    @staticmethod
    def _captured(
        record: StageRecord, key: str, model: type[_Capture]
    ) -> tuple[_Capture, ...]:
        raw = record.params.get(key)
        try:
            return tuple(model.model_validate(item) for item in raw or ())
        except (ValueError, TypeError) as exc:
            log.info(
                "the %s record's %s are not read: %s",
                record.plugin,
                key,
                _one_line(exc),
            )
            return ()
