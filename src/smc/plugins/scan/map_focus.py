"""``cell_focus="map"`` in the scan: each field's Z from the 10x lock map plus its well's anchor (m3 §21.12.5).

Until PFS locks at the Ti2-E's 40x (#539), a 40x scan runs with the autofocus
off. A scan taken under the 10x's lock has measured the plate's bottom at each
of its frames; this module reads that map at every 40x field and adds one
offset per well, measured at the 40x by a few sweeps near the well's centre.
``plugin.py`` only calls it; the decisions it encodes:

* **Everything is checked before anything moves** (FM-30), dry run included:
  the map's document as ``calibration_from`` reads one (the runner's own
  ``unusable``, the record validated, never trusted), the plate registration it
  was scanned on (a map of another registration puts every field's Z at another
  place), enough locked frames in every well left, the map of a resumed scan,
  the operator's focus inside a mapped well, and every field's reach inside the
  soft limits and the Z jog limit (D19: the moves are absolute, so the layer's
  jog guard does not reach them).
* **Nothing drifts from well to well** (D2's point, D18). Every anchor's sweeps
  are centred on the map plus the operator's anchor, never on the previous
  well's nor on a field's best plane.
* **A flat well keeps the operator's anchor only where the operator focused**
  (D23). The operator's focus is a measurement in one well, ``start_well``; in
  any other well the 10x -> 40x difference is not known (it moved by 48 µm
  between two wells), so the well is not scanned and the scan goes on.
* **A field far from every locked frame is scanned, and counted** (D22). The
  map's Z is a weighted mean of frames that may lie millimetres away; the count
  and the farthest field are said, against a bound that is assumed.
* **The map is read at the readback** (FM-142), never at the target, and only
  from locked frames (FM-154): the others name no bottom.
* **A check measures, it does not correct** (D20): a checked field sweeps
  around its target, then moves back there, and its frame is the map's.
* **The autofocus is not touched** (D21): no engage, no switch-off, no offset.
* **A failed anchor try is not a failed field**: it is recorded with its error,
  a WARNING names it, and the next try runs. A halt, a motion stop and a stop
  pass (FM-64); nothing moves to put something back (FM-88).
* **Every target is bounded by the Z jog limit from the Z the scan started at**
  (D19): the moves are absolute, and a peak found at a field lies up to
  ``focus_range_um`` from the start anchor, so the bound checked before anything
  moved does not hold for a target, and is checked again at each one.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from smc.hardware.capabilities import XY, Camera, Limits, XYStage, ZStage
from smc.hardware.errors import (
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
)
from smc.plugins.api import RunContext
from smc.plugins.cell_focus.algorithm import (
    LANDING_FRACTION,
    find_peak,
    plan_planes,
)
from smc.plugins.cell_focus.focus_map import (
    ANCHOR_PEAKS,
    ANCHOR_TRIES,
    MAP_NEIGHBOURS,
    FocusMap,
    MapPoint,
    anchor_of,
    nearest_first,
)
from smc.plugins.cell_focus.loop import (
    FocusDriveOffTarget,
    SweepBounds,
    move_checked,
    outside_limits,
    sweep,
    z_jog_limit_um,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import fov_of
from smc.plugins.scan.algorithm import landed_off_target, landing_error_um
from smc.plugins.slots import (
    AnchorTry,
    FieldFocus,
    FocusAnchor,
    FocusMapUse,
    FocusSweep,
    PlannedField,
    ScanResult,
)
from smc.run.document import RunDocument, RunDocumentError

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope
    from smc.plugins.scan.plugin import ScanParams

__all__ = [
    "MapFocus",
    "MapStep",
    "anchor_well",
    "focus_field",
    "prepare_map",
]

_NAME = "scan"
#: The scan's own logger: the operator reads the map's WARNINGs with the scan's.
_LOG = logging.getLogger(f"smc.plugins.{_NAME}")
#: What a refusal of the start's place tells the operator.
_FOCUS_INSIDE = "focus by eye on the cells inside a mapped well, then start the scan"
#: The two errors of an anchor try that D19's bounds refused before its sweep: Z
#: did not move, and the same plan meets the same refusal again. ``_anchor_try``
#: builds them and ``_failed_by_stand`` reads them back, so both share these.
_OUTSIDE = "SweepOutsideLimits"
_BEYOND = "SweepBeyondJog"


@dataclass
class MapFocus:
    """The map of this attempt: what the checks found, the anchors, and the counts for ``assumed``."""

    focus_map: FocusMap
    #: This attempt's record: its start anchor set, no anchor (``record`` adds them).
    use: FocusMapUse
    limits: Limits
    z_start_um: float
    jog_um: float
    #: The map scan's larger field-of-view side: the bound of D22 (assumed, not measured).
    fov_side_um: float
    #: The anchors a resumed checkpoint holds, kept as they are (D18).
    resumed: tuple[FocusAnchor, ...] = ()
    #: This attempt's anchors, in scan order.
    anchors: dict[str, FocusAnchor] = field(default_factory=dict)
    #: The targets the fields left would take with the start anchor (the dry run's line).
    planned: int = 0
    planned_lo_um: float | None = None
    planned_hi_um: float | None = None
    #: D22: the fields left whose nearest locked frame of their well lies beyond
    #: ``fov_side_um``, the farthest of them (the first in the plan's order on a tie).
    beyond: int = 0
    farthest_name: str | None = None
    farthest_um: float = 0.0
    #: Checked fields whose frame was written, and the distances of their peaks.
    checks: int = 0
    distances_um: list[float] = field(default_factory=list)
    #: Fields whose check sweep would have left the soft limits or the jog limit.
    checks_skipped: int = 0

    def anchored(self, well: str) -> FocusAnchor | None:
        """The anchor of ``well``, this attempt's or a resumed one; ``None`` before it is measured."""
        if well in self.anchors:
            return self.anchors[well]
        for kept in self.resumed:
            if kept.well == well:
                return kept
        return None

    def anchor_um(self, well: str) -> float | None:
        """The offset added to the map in ``well``; ``None`` before it is measured and in a well not scanned (D23)."""
        found = self.anchored(well)
        return None if found is None else found.anchor_um

    def not_scanned(self, well: str) -> str | None:
        """A field's error when ``well`` was anchored ``none`` (D23): it is not scanned; else ``None``."""
        found = self.anchored(well)
        if found is None or found.source != "none":
            return None
        why = _why(found, f"the centre of {well}")
        return f"MapNotAnchored: {why}{_not_scanned_tail(self.use.start_well)}"

    def beyond_line(self) -> str | None:
        """D22's line, for ``assumed`` and the WARNING; ``None`` when no field lies beyond the bound."""
        if not self.beyond or self.farthest_name is None:
            return None
        return (
            f"{self.beyond} of {self.planned} fields lie more than "
            f"{self.fov_side_um:.1f} µm from every locked frame of their well, the "
            f"farthest {self.farthest_name} at {self.farthest_um:.1f} µm: where they "
            "are scanned, their Z is read from frames that far (the bound, the map "
            "scan's larger field side, is assumed, not measured)"
        )

    def record(self) -> FocusMapUse:
        """The ``focus_map`` of the result and the checkpoint: the resumed anchors first."""
        return self.use.model_copy(
            update={"anchors": (*self.resumed, *self.anchors.values())}
        )

    def count(self, done: FieldFocus) -> None:
        """Count a field whose frame was written: its check, when it has one (D20)."""
        if done.sweep is None or done.map_z_um is None or done.anchor_um is None:
            return
        self.checks += 1
        if done.sweep.best_z_um is not None:
            target = done.map_z_um + done.anchor_um
            self.distances_um.append(abs(done.sweep.best_z_um - target))

    def assumed(
        self, *, dry_run: bool, fields: int, step_s: float, every: int
    ) -> list[str]:
        """``assumed``'s lines for the map (m3 §21.12.5), two decimals."""
        start = self.use.start_anchor_um
        beyond = self.beyond_line()
        if dry_run:
            reach = (
                f", Z {self.planned_lo_um:.2f}…{self.planned_hi_um:.2f} µm"
                if self.planned_lo_um is not None and self.planned_hi_um is not None
                else ""
            )
            planned = [
                f"dry run: the focus from the map was planned for {self.planned} "
                f"fields{reach} with the operator's anchor {start:.2f} µm, not run"
            ]
            return planned if beyond is None else [*planned, beyond]
        lines = [
            f"the focus came from the map in {self.use.run_doc}, scanned "
            f"{self.use.measured}: each field's Z is the map's at its XY, from the "
            f"{MAP_NEIGHBOURS} nearest locked frames of its well, plus its well's anchor"
        ]
        if beyond is not None:
            lines.append(beyond)
        # The wells an earlier attempt anchored come first, in the checkpoint's
        # order: this attempt did not measure them, and its time line below does
        # not count them (#712).
        for anchor in self.resumed:
            line = _anchor_line(anchor, self.use.start_well)
            lines.append(f"{line} (an earlier attempt)")
        for anchor in self.anchors.values():
            lines.append(_anchor_line(anchor, self.use.start_well))
        anchor_s = sum(a.duration_s for a in self.anchors.values())
        if fields or self.anchors:
            per_field = step_s / fields if fields else 0.0
            lines.append(
                f"the focus from the map added {per_field:.2f} s per field ({fields} "
                f"fields), and the anchors {anchor_s:.2f} s ({len(self.anchors)} wells)"
            )
        lines.append(self._check_line(every))
        if self.checks_skipped:
            lines.append(
                f"the map's Z was not checked on {self.checks_skipped} fields whose "
                "check sweep would have left the soft limits or the Z jog limit"
            )
        return lines

    def _check_line(self, every: int) -> str:
        if every == 0:
            return "the map's Z was not checked against a sweep (map_check_every=0)"
        peaks = len(self.distances_um)
        within_1 = sum(d <= 1.0 for d in self.distances_um)
        within_2 = sum(d <= 2.0 for d in self.distances_um)
        median = (
            f", the median distance {statistics.median(self.distances_um):.2f} µm"
            if self.distances_um
            else ""
        )
        return (
            f"the map's Z was checked on {self.checks} fields: a peak on {peaks}, "
            f"within 1 µm of the map's Z on {within_1} and within 2 µm on "
            f"{within_2}{median}"
        )


def _operator_focus(start_well: str | None) -> str:
    """Where the operator focused by eye, for the lines of a well that is not scanned (D23)."""
    if start_well is None:
        return "the operator focused in no well of this scan"
    return f"the operator focused in {start_well}"


def _not_scanned_tail(start_well: str | None) -> str:
    """What ends the text of a well anchored ``none``, a field's error and the well's line alike (D23)."""
    return f", and {_operator_focus(start_well)}: the well was not scanned"


def _why(anchor: FocusAnchor, place: str) -> str:
    """Why no peak anchored the well: that no sweep found one, or that no sweep could run (#712).

    A try that ended before its sweep (a refusal of D19's bounds, a landing off
    target, a drive or a stage that failed) is not a flat try: saying that "no
    sweep found a peak" when none ran sent the operator to look at the sample
    when it was the stage, or the plan, that stopped them. ``place`` is where
    the sweeps were centred, in the words of the line it goes into.
    """
    tried = len(anchor.tries)
    if not anchor.tries or any(t.sweep is not None for t in anchor.tries):
        return f"no sweep near {place} found a peak ({tried} tried)"
    return (
        f"no sweep near {place} could run ({tried} tried, the first: "
        f"{anchor.tries[0].error})"
    )


def _anchor_line(anchor: FocusAnchor, start_well: str | None) -> str:
    """A well's line in ``assumed``, the same text as its WARNING when no peak came."""
    tried = len(anchor.tries)
    offset = anchor.anchor_um
    if offset is None:
        # D23: no peak in a well the operator did not focus in.
        why = _why(anchor, "its centre")
        return f"{anchor.well}: {why}{_not_scanned_tail(start_well)}"
    if anchor.source == "swept":
        peaks = sum(
            t.sweep is not None and t.sweep.outcome == "peak" for t in anchor.tries
        )
        return (
            f"{anchor.well}: anchored {offset:.2f} µm from the map by "
            f"{peaks} peaks of {tried} sweeps near its centre"
        )
    return (
        f"{anchor.well}: {_why(anchor, 'its centre')}: the operator's focus anchors "
        f"it, {offset:.2f} µm from the map"
    )


@dataclass(frozen=True)
class _Loaded:
    """What ``_read_map`` reads, with the names of the frames for the start's refusal."""

    focus_map: FocusMap
    use: FocusMapUse
    fov_side_um: float
    names: dict[MapPoint, str]


def _read_map(
    path: Path, *, profile: str, dry_run: bool, plate_attempt: str | None
) -> _Loaded:
    """The map in another run document, its record with no anchor, and the larger side of its field.

    Read as ``calibration_from`` reads a calibration (m3 §20.3): the runner's
    own ``unusable`` decides whether the document's ``scan`` can be used (not
    ``done``, another profile, a dry run read by a real one, stale), and its
    result is validated, never trusted. ``path`` is named as typed, with
    forward slashes (FM-24), and never resolved (FM-50): a relative path is
    read from the working folder and recorded as typed. ``plate_attempt`` is
    the attempt of this run document's ``plate`` provider: a map scanned on
    another registration of the plate is refused, and so is one when either
    side has none. The record's ``start_anchor_um`` is 0.0: the caller sets it.
    The names of the locked frames come with it, for check 5's text.

    Raises:
        PluginRunError: The document cannot be read, holds no usable ``scan``,
            a result this smc does not read, or a scan of another registration.
    """
    where = path.as_posix()
    problem: str | None = None
    try:
        other = RunDocument.load(path)
    except RunDocumentError as exc:
        problem = f"the focus map {where}: {exc.reason}"
    if problem is not None:
        # Raised outside the handler: nothing here keeps the load's error (FM-98).
        raise PluginRunError(_NAME, problem)

    reason = other.unusable(_NAME, profile=profile, dry_run=dry_run)
    record = other.provider(_NAME)
    if reason is not None or record is None or record.result is None:
        raise PluginRunError(
            _NAME, f"the focus map in {where}: {reason or 'it holds no result'}"
        )
    found: ScanResult | None = None
    try:
        found = ScanResult.model_validate(record.result)
    except ValidationError:
        problem = (
            f"the scan in {where} is not one this smc reads: it cannot be the focus "
            "map; run scan again there"
        )
    if problem is not None or found is None:
        raise PluginRunError(_NAME, problem or f"the scan in {where} holds no result")

    read = record.reads.get("plate")
    map_attempt = read.attempt if read is not None else None
    if map_attempt is None or plate_attempt is None or map_attempt != plate_attempt:
        raise PluginRunError(
            _NAME,
            f"the focus map in {where} was scanned on another registration of the "
            f"plate ({map_attempt or 'none'} against this run's "
            f"{plate_attempt or 'none'})",
        )

    points: list[MapPoint] = []
    names: dict[MapPoint, str] = {}
    for frame in found.frames:
        # Only a locked frame names the bottom: an unlocked one's Z is where a
        # search left it (FM-154).
        if frame.focus != "locked" or frame.z_um is None:
            continue
        point = MapPoint(frame.well, frame.x_um, frame.y_um, frame.z_um)
        points.append(point)
        names.setdefault(point, frame.name)
    measured = (
        record.finished_utc.isoformat(timespec="seconds")
        if record.finished_utc is not None
        else "at an unknown time"
    )
    use = FocusMapUse(
        run_doc=where,
        scan_attempt=record.attempt,
        measured=measured,
        start_anchor_um=0.0,
        anchors=(),
    )
    fov = fov_of(found.matrix_um_per_px, found.image_shape_px)
    return _Loaded(FocusMap(points), use, max(fov), names)


def prepare_map(
    microscope: Microscope,
    xy: XYStage,
    params: ScanParams,
    ctx: RunContext,
    bounds: SweepBounds,
    groups: Sequence[tuple[str, Sequence[PlannedField]]],
    *,
    autofocus_used: bool,
    resumed: FocusMapUse | None,
) -> MapFocus:
    """§21.12.5's checks 1-6, in their order, before anything moves; dry run included.

    ``groups`` are the wells left to scan with their fields, in scan order;
    ``bounds`` what §21.6's checks read (the soft limits, ``z_start``);
    ``resumed`` the checkpoint's ``focus_map`` on a resume.

    Raises:
        PluginRunError: A check refused the scan; nothing moved.
    """
    path = params.focus_map
    assert path is not None  # ScanParams refuses cell_focus=map without it
    where = path.as_posix()
    # 1. D21: under a lock, per-field and per-well start from the plane PFS holds.
    if autofocus_used:
        raise PluginRunError(
            _NAME,
            "cell_focus=map takes each field's Z from the map with the autofocus "
            "off, and the autofocus is locked: switch it off, or use per-field or "
            "per-well",
        )
    # 2. The map's document.
    plate = ctx.document.provider("plate")
    loaded = _read_map(
        path,
        profile=microscope.profile.microscope.name,
        dry_run=ctx.dry_run,
        plate_attempt=plate.attempt if plate is not None else None,
    )
    focus_map = loaded.focus_map
    # 3. Every well left holds enough locked frames.
    short = [well for well, _ in groups if focus_map.count(well) < MAP_NEIGHBOURS]
    if short:
        raise PluginRunError(
            _NAME,
            f"the focus map in {where} holds fewer than {MAP_NEIGHBOURS} locked "
            f"frames in {', '.join(short)}: scan them with the autofocus locked first",
        )
    # 4. A resume goes on with the map its anchors were measured against (FM-111).
    kept: tuple[FocusAnchor, ...] = ()
    if resumed is not None:
        if resumed.scan_attempt != loaded.use.scan_attempt:
            raise PluginRunError(
                _NAME,
                f"the focus map in {where} is not the one the scan being resumed "
                "used: run scan again without --resume",
            )
        kept = resumed.anchors
    # 5. The operator's focus, inside a mapped well.
    z_start = bounds.z_start_um
    if resumed is not None:
        # A resume keeps the operator's anchor of the attempt it resumes: a stop
        # during an anchor's or a check's sweep leaves Z on a sweep plane, up to
        # focus_range_um from the operator's focus, and an anchor read from it
        # would centre every later well's sweeps there (#623's review).
        start_anchor = resumed.start_anchor_um
        start_well = resumed.start_well
    elif groups:
        start_anchor, start_well = _start_anchor(xy, loaded, z_start, where)
    else:
        # Every well was done by an earlier attempt: nothing is read, nothing
        # moves, and no well is the operator's.
        start_anchor, start_well = 0.0, None
    state = MapFocus(
        focus_map=focus_map,
        use=loaded.use.model_copy(
            update={"start_anchor_um": start_anchor, "start_well": start_well}
        ),
        limits=bounds.limits,
        z_start_um=z_start,
        jog_um=z_jog_limit_um(microscope),
        fov_side_um=loaded.fov_side_um,
        resumed=kept,
    )
    # 6. D19: every field's reach, its anchor's sweeps and its target; and D22's
    # count of the fields far from every locked frame of their well.
    _check_reach(state, groups, params.focus_range_um)
    line = state.beyond_line()
    if line is not None:
        _LOG.warning("%s", line)
    return state


def _um(value: float) -> str:
    """One decimal, with no ``-0.0``: a stage read as -0.0 is at 0.0 for the operator."""
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text


def _start_anchor(
    xy: XYStage, loaded: _Loaded, z_start: float, where: str
) -> tuple[float, str]:
    """Check 5: the operator focused inside a mapped well; ``z_start`` minus the map there, and that well.

    The well is the one of the map's locked frame nearest the stage, whether or
    not the plan holds it: it is the only well where a flat anchor keeps the
    operator's anchor (D23).

    Raises:
        PluginRunError: The XY read is not finite, the map's nearest locked frame
            is farther than the map scan's larger field side, or its well holds
            too few locked frames to be read.
    """
    here: XY = xy.position_um()
    x0, y0 = here.x_um, here.y_um
    if not (math.isfinite(x0) and math.isfinite(y0)):
        raise PluginRunError(
            _NAME,
            f"the stage reads ({x0!r}, {y0!r}) µm: the map cannot be read there: "
            f"{_FOCUS_INSIDE}",
        )
    point, distance = loaded.focus_map.nearest(x0, y0)
    name = loaded.names.get(point, "?")
    if not distance <= loaded.fov_side_um:
        raise PluginRunError(
            _NAME,
            f"the stage at ({_um(x0)}, {_um(y0)}) µm is {distance:.1f} µm from the "
            f"map's nearest locked frame ({name} in {point.well}): {_FOCUS_INSIDE}",
        )
    count = loaded.focus_map.count(point.well)
    if count < MAP_NEIGHBOURS:
        raise PluginRunError(
            _NAME,
            f"the stage at ({_um(x0)}, {_um(y0)}) µm is nearest the map's locked frame "
            f"{name} in {point.well}, and the focus map in {where} holds {count} "
            f"locked frames there, fewer than {MAP_NEIGHBOURS}: {_FOCUS_INSIDE}",
        )
    return z_start - loaded.focus_map.z_at(point.well, x0, y0), point.well


def _check_reach(
    state: MapFocus,
    groups: Sequence[tuple[str, Sequence[PlannedField]]],
    range_um: float,
) -> None:
    """Check 6 (D19): each field's ``map + start anchor ± range`` inside the limits and the jog bound.

    Field by field in the plan's order, at the planned XY. The moves are
    absolute, so the layer's Z jog guard does not reach them (D3): the plugin
    bounds them by the same number, from the Z the scan starts at. The targets'
    span is kept for the dry run's line.

    Each field's distance to the nearest locked frame of its well is counted
    against the map scan's larger field side (D22): a field beyond it is
    scanned, and the count and the farthest are said, not refused.

    Raises:
        PluginRunError: A field's reach leaves the soft limits or the jog bound.
    """
    low, high = state.limits.low_um, state.limits.high_um
    start = state.use.start_anchor_um
    for well, fields in groups:
        for planned in fields:
            centre = state.focus_map.z_at(well, planned.x_um, planned.y_um) + start
            ends = (centre - range_um, centre + range_um)
            for reach in ends:
                if not low <= reach <= high:
                    raise PluginRunError(
                        _NAME,
                        f"the focus from the map would reach {reach:.2f} µm at "
                        f"{planned.name}, outside the soft limits [{low:g}, {high:g}] µm",
                    )
            far = max(ends, key=lambda z: abs(z - state.z_start_um))
            distance = abs(far - state.z_start_um)
            if not distance <= state.jog_um:
                raise PluginRunError(
                    _NAME,
                    f"the focus from the map would reach {far:.2f} µm at "
                    f"{planned.name}, {distance:.2f} µm from the Z the scan starts "
                    f"at, beyond the Z jog limit of {state.jog_um:g} µm",
                )
            # At the planned XY, not the readback: it is a count made before
            # anything moves, and the difference is a landing tolerance against a
            # bound of a field side.
            gap = state.focus_map.distance_um(well, planned.x_um, planned.y_um)
            if gap > state.fov_side_um:
                state.beyond += 1
                if state.farthest_name is None or gap > state.farthest_um:
                    state.farthest_name, state.farthest_um = planned.name, gap
            state.planned += 1
            state.planned_lo_um = (
                centre
                if state.planned_lo_um is None
                else min(state.planned_lo_um, centre)
            )
            state.planned_hi_um = (
                centre
                if state.planned_hi_um is None
                else max(state.planned_hi_um, centre)
            )


def anchor_well(
    well: str,
    fields: Sequence[PlannedField],
    centre_xy: tuple[float, float],
    *,
    xy: XYStage,
    z: ZStage,
    camera: Camera,
    state: MapFocus,
    params: ScanParams,
    tolerance_um: float,
    ctx: RunContext,
    done: int,
    total: int,
) -> FocusAnchor:
    """Measure ``well``'s anchor at the 40x near its centre, and keep it in ``state`` (D18).

    Up to ``ANCHOR_TRIES`` of the well's planned fields nearest its centre, each
    swept around the map there plus the operator's anchor, until
    ``ANCHOR_PEAKS`` find a peak; the anchor is the median of their offsets
    from the map. When none does, the operator's anchor in the well the operator
    focused in (``source="start"``), and none in any other well
    (``source="none"``, D23: the well is not scanned), unless no try ran a
    sweep and one of them failed by the stand: that is a stand that failed, not
    a well without a peak. A try that D19's bounds refused did not fail by the
    stand (#712). ``ctx.progress`` keeps the fields' count. The tries' frames
    are neither written nor shown.

    Raises:
        PluginRunError: No try ran a sweep in a well the operator did not focus
            in, and at least one ended in a failure of the stand (a refusal of
            D19's bounds is not one): the scan ends, and a resume anchors the
            well again.
        MicroscopeHaltedError, MotionStoppedError, StopRequested: Passed (FM-64);
            Z stays where it was read last (FM-88).
    """
    started = time.perf_counter()
    order = nearest_first(
        [(planned.x_um, planned.y_um) for planned in fields], centre_xy, ANCHOR_TRIES
    )
    tries: list[AnchorTry] = []
    offsets: list[float] = []
    for k, i in enumerate(order, start=1):
        ctx.raise_if_stopped()
        ctx.progress(done, total, f"anchoring {well} ({k} of {len(order)})")
        tried, offset = _anchor_try(
            fields[i], xy=xy, z=z, camera=camera, state=state, params=params,
            tolerance_um=tolerance_um, ctx=ctx,
        )  # fmt: skip
        tries.append(tried)
        if tried.error is not None:
            _LOG.warning(
                "%s: the anchor's try at %s failed (%s); the next field near the "
                "centre is tried",
                well,
                tried.name,
                tried.error,
            )
        if offset is not None:
            offsets.append(offset)
            if len(offsets) >= ANCHOR_PEAKS:
                break
    found = anchor_of(offsets)
    source: Literal["swept", "start", "none"]
    if found is not None:
        source, anchor_um = "swept", found
    elif well == state.use.start_well:
        # The operator's focus is a measurement in this well (D23).
        source, anchor_um = "start", state.use.start_anchor_um
    else:
        failures = [t for t in tries if _failed_by_stand(t)]
        if failures and all(t.sweep is None for t in tries):
            # No sweep ran, and the stand failed in at least one try: the stage,
            # the camera or the Z drive, which is not "no peak". Not scanning the
            # well would turn a dead stage into a scan that ends ``done`` (the
            # backstop of ``max_failed_in_row`` does not count these fields), so
            # the scan ends and a resume anchors the well again (#688's review,
            # FM-277). Tries that D19's bounds refused are not failures of the
            # stand: Z did not move, a resume would meet the same refusal, and a
            # well where every try was refused is ``none`` like a flat one (#712).
            raise PluginRunError(
                _NAME,
                f"the anchor of {well} could not be measured: none of its "
                f"{len(tries)} tries ran a sweep and {len(failures)} of them failed "
                f"(the first: {failures[0].error}): check the stage and the camera, "
                "then run scan --resume",
            )
        # Not in this one: the 10x -> 40x difference is not known here.
        source, anchor_um = "none", None
    anchor = FocusAnchor(
        well=well,
        source=source,
        anchor_um=anchor_um,
        tries=tuple(tries),
        duration_s=time.perf_counter() - started,
    )
    if found is None:
        _LOG.warning("%s", _anchor_line(anchor, state.use.start_well))
    state.anchors[well] = anchor
    return anchor


def _anchor_try(
    planned: PlannedField,
    *,
    xy: XYStage,
    z: ZStage,
    camera: Camera,
    state: MapFocus,
    params: ScanParams,
    tolerance_um: float,
    ctx: RunContext,
) -> tuple[AnchorTry, float | None]:
    """One try: §8's move and landing, the map at the readback, the sweep; its record and offset.

    A landing off target, a plane outside the soft limits or beyond the Z jog
    limit, ``FocusDriveOffTarget`` or a ``HardwareError`` end the try with its
    error, and Z makes no move for a try that ends before its sweep; the offset
    is ``None`` unless the sweep found a peak.
    """
    readback: XY | None = None
    map_z: float | None = None
    record: FocusSweep | None = None
    offset: float | None = None
    error: str | None = None
    try:
        moved = xy.move_to_um(planned.x_um, planned.y_um)
        off_um = landing_error_um(
            (planned.x_um, planned.y_um), (moved.x_um, moved.y_um)
        )
        if landed_off_target(off_um, tolerance_um):
            error = (
                f"StageOffTarget: the readback is {off_um:.1f} µm from the target, "
                f"more than {tolerance_um:.1f} µm (target ({planned.x_um:.1f}, "
                f"{planned.y_um:.1f}), readback ({moved.x_um:.1f}, {moved.y_um:.1f}))"
            )
        else:
            readback = moved
            # At the readback, never the target (FM-142).
            map_z = state.focus_map.z_at(planned.well, moved.x_um, moved.y_um)
            centre = map_z + state.use.start_anchor_um
            planes = plan_planes(centre, params.focus_range_um, params.focus_step_um)
            refusal = outside_limits(centre, planes, state.limits)
            if refusal is not None:
                error = f"{_OUTSIDE}: {refusal}"
            elif (
                # D19 (#688): the plane's distance from the Z the scan started at,
                # as the field's target is bounded; the moves are absolute, and
                # this centre is the map at the readback, not at the planned XY
                # check 6 read. Asked only when the limits did not refuse (#712).
                beyond := _beyond_jog(state, "the anchor sweep", centre, planes)
            ) is not None:
                error = f"{_BEYOND}: {beyond}"
            else:
                readbacks, scores = sweep(
                    z,
                    camera,
                    planes,
                    params.focus_step_um,
                    params.focus_score,
                    raise_if_stopped=ctx.raise_if_stopped,
                    wait_s=ctx.wait_s,
                )
                peak = find_peak(readbacks, scores)
                record = FocusSweep(
                    score=params.focus_score,
                    centre_um=centre,
                    z_um=readbacks,
                    scores=scores,
                    outcome=peak.outcome,
                    best_z_um=peak.z_um,
                )
                if peak.z_um is not None:
                    offset = peak.z_um - map_z
    except (MicroscopeHaltedError, MotionStoppedError):
        raise
    except (HardwareError, FocusDriveOffTarget) as exc:
        error = _one_line(exc)
        record = None
        offset = None
    # The readback and the map's Z go together: both read, or the move failed.
    read = readback is not None and map_z is not None
    return (
        AnchorTry(
            name=planned.name,
            x_um=readback.x_um if read and readback is not None else None,
            y_um=readback.y_um if read and readback is not None else None,
            map_z_um=map_z if read else None,
            sweep=record,
            error=error,
        ),
        offset,
    )


@dataclass(frozen=True)
class MapStep:
    """A field's step up to its snap: when it started, its check and what the frame records."""

    started: float
    sweep: FocusSweep | None
    map_z_um: float
    anchor_um: float


def focus_field(
    z: ZStage,
    camera: Camera,
    state: MapFocus,
    planned: PlannedField,
    readback: XY,
    params: ScanParams,
    ctx: RunContext,
) -> MapStep | str:
    """§21.12.5's per-field step: Z to the map plus the well's anchor, and the check on every n-th field.

    ``readback`` is the field's XY readback, already checked against its target
    (§8). Returns the step, or the field's error when nothing moved for it.

    Raises:
        FocusDriveOffTarget, HardwareError: The field fails (§8).
        MicroscopeHaltedError, MotionStoppedError, StopRequested: Passed.
    """
    anchor = state.anchor_um(planned.well)
    if anchor is None:
        # The anchor pass runs before a well's first field: only a bug gets here.
        return f"MapNotAnchored: the well {planned.well} has no anchor"
    map_z = state.focus_map.z_at(planned.well, readback.x_um, readback.y_um)
    target = map_z + anchor
    low, high = state.limits.low_um, state.limits.high_um
    if not low <= target <= high:
        return (
            f"MapOutsideLimits: the focus from the map at {planned.name} is "
            f"{target:.2f} µm, outside the soft limits [{low:g}, {high:g}] µm"
        )
    # D19 (#688): the peak an anchor found lies up to focus_range_um from the
    # start anchor, and the map is read at the readback, not at the planned XY
    # check 6 read: a target can pass that bound and fail this one.
    distance = abs(target - state.z_start_um)
    if not distance <= state.jog_um:
        return (
            f"MapBeyondJog: the focus from the map at {planned.name} is "
            f"{target:.2f} µm, {distance:.2f} µm from the Z the scan starts at, "
            f"beyond the Z jog limit of {state.jog_um:g} µm"
        )
    started = time.perf_counter()
    tolerance_um = LANDING_FRACTION * params.focus_step_um
    move_checked(z, target, tolerance_um, wait_s=ctx.wait_s)
    check: FocusSweep | None = None
    every = params.map_check_every
    if every > 0 and planned.index % every == 0:
        planes = plan_planes(target, params.focus_range_um, params.focus_step_um)
        refusal = outside_limits(target, planes, state.limits) or _beyond_jog(
            state, "the check sweep", target, planes
        )
        if refusal is not None:
            # The frame is still the map's: a check that cannot be made is said.
            state.checks_skipped += 1
            _LOG.warning("%s: the map's Z was not checked (%s)", planned.name, refusal)
        else:
            readbacks, scores = sweep(
                z,
                camera,
                planes,
                params.focus_step_um,
                params.focus_score,
                raise_if_stopped=ctx.raise_if_stopped,
                wait_s=ctx.wait_s,
            )
            peak = find_peak(readbacks, scores)
            check = FocusSweep(
                score=params.focus_score,
                centre_um=target,
                z_um=readbacks,
                scores=scores,
                outcome=peak.outcome,
                best_z_um=peak.z_um,
            )
            # The check measures, it does not correct (D20): back to the map's Z.
            move_checked(z, target, tolerance_um, wait_s=ctx.wait_s)
    return MapStep(started, check, map_z, anchor)


def _beyond_jog(
    state: MapFocus, subject: str, centre: float, planes: Sequence[float]
) -> str | None:
    """The refusal of a sweep with a plane beyond the jog bound from ``z_start`` (D3), or ``None``.

    ``subject`` names the sweep in the text (``the check sweep``, ``the anchor
    sweep``); ``centre`` is the Z it is centred on.
    """
    for plane in planes:
        distance = abs(plane - state.z_start_um)
        if not distance <= state.jog_um:
            return (
                f"{subject} around {centre:.2f} µm would reach {plane:.2f} µm, "
                f"{distance:.2f} µm from the Z the scan starts at, beyond the Z jog "
                f"limit of {state.jog_um:g} µm"
            )
    return None


def _failed_by_stand(tried: AnchorTry) -> bool:
    """Whether an anchor try ended in a failure of the stand, not in a refusal of D19's bounds (#712).

    The error carries its type (``<Type>: <message>``), so a later reader of
    ``focus_map.anchors`` classifies a try the same way: ``StageOffTarget``,
    ``FocusDriveOffTarget`` and every ``HardwareError`` are the stand's; a try
    that swept has no error.
    """
    error = tried.error
    return error is not None and not error.startswith((f"{_OUTSIDE}:", f"{_BEYOND}:"))


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``, as the scan's failed fields."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()
