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
  well's nor on a field's best plane; a well where no sweep finds a peak keeps
  the operator's anchor, and the result says so.
* **The map is read at the readback** (FM-142), never at the target, and only
  from locked frames (FM-154): the others name no bottom.
* **A check measures, it does not correct** (D20): a checked field sweeps
  around its target, then moves back there, and its frame is the map's.
* **The autofocus is not touched** (D21): no engage, no switch-off, no offset.
* **A failed anchor try is not a failed field**: it is recorded with its error,
  a WARNING names it, and the next try runs. A halt, a motion stop and a stop
  pass (FM-64); nothing moves to put something back (FM-88).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from smc.hardware.capabilities import XY, Camera, Limits, XYStage, ZStage
from smc.hardware.errors import (
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
)
from smc.hardware.safety import DEFAULT_MAX_Z_JOG_UM
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
    "load_focus_map",
    "prepare_map",
]

_NAME = "scan"
#: The scan's own logger: the operator reads the map's WARNINGs with the scan's.
_LOG = logging.getLogger(f"smc.plugins.{_NAME}")
#: What a refusal of the start's place tells the operator.
_FOCUS_INSIDE = "focus by eye on the cells inside a mapped well, then start the scan"


@dataclass
class MapFocus:
    """The map of this attempt: what the checks found, the anchors, and the counts for ``assumed``."""

    focus_map: FocusMap
    #: This attempt's record: its start anchor set, no anchor (``record`` adds them).
    use: FocusMapUse
    limits: Limits
    z_start_um: float
    jog_um: float
    #: The anchors a resumed checkpoint holds, kept as they are (D18).
    resumed: tuple[FocusAnchor, ...] = ()
    #: This attempt's anchors, in scan order.
    anchors: dict[str, FocusAnchor] = field(default_factory=dict)
    #: The targets the fields left would take with the start anchor (the dry run's line).
    planned: int = 0
    planned_lo_um: float | None = None
    planned_hi_um: float | None = None
    #: Checked fields whose frame was written, and the distances of their peaks.
    checks: int = 0
    distances_um: list[float] = field(default_factory=list)
    #: Fields whose check sweep would have left the soft limits or the jog limit.
    checks_skipped: int = 0

    def anchor_um(self, well: str) -> float | None:
        """The anchor of ``well``, this attempt's or a resumed one; ``None`` before it is measured."""
        if well in self.anchors:
            return self.anchors[well].anchor_um
        for kept in self.resumed:
            if kept.well == well:
                return kept.anchor_um
        return None

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
        if dry_run:
            reach = (
                f", Z {self.planned_lo_um:.2f}…{self.planned_hi_um:.2f} µm"
                if self.planned_lo_um is not None and self.planned_hi_um is not None
                else ""
            )
            return [
                f"dry run: the focus from the map was planned for {self.planned} "
                f"fields{reach} with the operator's anchor {start:.2f} µm, not run"
            ]
        lines = [
            f"the focus came from the map in {self.use.run_doc}, scanned "
            f"{self.use.measured}: each field's Z is the map's at its XY, from the "
            f"{MAP_NEIGHBOURS} nearest locked frames of its well, plus its well's anchor"
        ]
        for anchor in self.anchors.values():
            lines.append(_anchor_line(anchor))
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
            f", the median distance {_median(self.distances_um):.2f} µm"
            if self.distances_um
            else ""
        )
        return (
            f"the map's Z was checked on {self.checks} fields: a peak on {peaks}, "
            f"within 1 µm of the map's Z on {within_1} and within 2 µm on "
            f"{within_2}{median}"
        )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _anchor_line(anchor: FocusAnchor) -> str:
    """A well's line in ``assumed``, the same text as its WARNING when no peak came."""
    tried = len(anchor.tries)
    if anchor.source == "swept":
        peaks = sum(
            t.sweep is not None and t.sweep.outcome == "peak" for t in anchor.tries
        )
        return (
            f"{anchor.well}: anchored {anchor.anchor_um:.2f} µm from the map by "
            f"{peaks} peaks of {tried} sweeps near its centre"
        )
    return (
        f"{anchor.well}: no sweep near its centre found a peak ({tried} tried): the "
        f"operator's focus anchors it, {anchor.anchor_um:.2f} µm from the map"
    )


@dataclass(frozen=True)
class _Loaded:
    """What ``load_focus_map`` reads, with the names of the frames for the start's refusal."""

    focus_map: FocusMap
    use: FocusMapUse
    fov_side_um: float
    names: dict[MapPoint, str]


def _read_map(
    path: Path, *, profile: str, dry_run: bool, plate_attempt: str | None
) -> _Loaded:
    """``load_focus_map``'s body; the names of the locked frames too, for check 5's text."""
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


def load_focus_map(
    path: Path, *, profile: str, dry_run: bool, plate_attempt: str | None
) -> tuple[FocusMap, FocusMapUse, float]:
    """The map in another run document, its record with no anchor, and the larger side of its field.

    Read as ``calibration_from`` reads a calibration (m3 §20.3): the runner's
    own ``unusable`` decides whether the document's ``scan`` can be used (not
    ``done``, another profile, a dry run read by a real one, stale), and its
    result is validated, never trusted. ``path`` is named as typed, with
    forward slashes (FM-24), and never resolved (FM-50). ``plate_attempt`` is
    the attempt of this run document's ``plate`` provider: a map scanned on
    another registration of the plate is refused, and so is one when either
    side has none. The record's ``start_anchor_um`` is 0.0: the caller sets it.

    Raises:
        PluginRunError: The document cannot be read, holds no usable ``scan``,
            a result this smc does not read, or a scan of another registration.
    """
    loaded = _read_map(
        path, profile=profile, dry_run=dry_run, plate_attempt=plate_attempt
    )
    return loaded.focus_map, loaded.use, loaded.fov_side_um


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
    if groups:
        start_anchor = _start_anchor(xy, loaded, z_start, where)
    else:
        # Every well was done by an earlier attempt: nothing is read, nothing moves.
        start_anchor = resumed.start_anchor_um if resumed is not None else 0.0
    configured = microscope.profile.safety.max_z_jog_um
    jog_um = DEFAULT_MAX_Z_JOG_UM if configured is None else float(configured)
    state = MapFocus(
        focus_map=focus_map,
        use=loaded.use.model_copy(update={"start_anchor_um": start_anchor}),
        limits=bounds.limits,
        z_start_um=z_start,
        jog_um=jog_um,
        resumed=kept,
    )
    # 6. D19: every field's reach, its anchor's sweeps and its target.
    _check_reach(state, groups, params.focus_range_um)
    return state


def _start_anchor(xy: XYStage, loaded: _Loaded, z_start: float, where: str) -> float:
    """Check 5: the operator focused inside a mapped well; ``z_start`` minus the map there.

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
            f"the stage at ({x0:.1f}, {y0:.1f}) µm is {distance:.1f} µm from the "
            f"map's nearest locked frame ({name} in {point.well}): {_FOCUS_INSIDE}",
        )
    count = loaded.focus_map.count(point.well)
    if count < MAP_NEIGHBOURS:
        raise PluginRunError(
            _NAME,
            f"the stage at ({x0:.1f}, {y0:.1f}) µm is nearest the map's locked frame "
            f"{name} in {point.well}, and the focus map in {where} holds {count} "
            f"locked frames there, fewer than {MAP_NEIGHBOURS}: {_FOCUS_INSIDE}",
        )
    return z_start - loaded.focus_map.z_at(point.well, x0, y0)


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
    from the map, or the operator's when none does. ``ctx.progress`` keeps the
    fields' count. The tries' frames are neither written nor shown.

    Raises:
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
    anchor = FocusAnchor(
        well=well,
        source="start" if found is None else "swept",
        anchor_um=state.use.start_anchor_um if found is None else found,
        tries=tuple(tries),
        duration_s=time.perf_counter() - started,
    )
    if found is None:
        _LOG.warning("%s", _anchor_line(anchor))
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

    A landing off target, a plane outside the limits, ``FocusDriveOffTarget``
    or a ``HardwareError`` end the try with its error; the offset is ``None``
    unless the sweep found a peak.
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
                error = f"SweepOutsideLimits: {refusal}"
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
    started = time.perf_counter()
    tolerance_um = LANDING_FRACTION * params.focus_step_um
    move_checked(z, target, tolerance_um, wait_s=ctx.wait_s)
    check: FocusSweep | None = None
    every = params.map_check_every
    if every > 0 and planned.index % every == 0:
        planes = plan_planes(target, params.focus_range_um, params.focus_step_um)
        refusal = outside_limits(target, planes, state.limits) or _beyond_jog(
            state, target, planes
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


def _beyond_jog(state: MapFocus, target: float, planes: Sequence[float]) -> str | None:
    """The refusal of a check sweep with a plane beyond the jog bound from ``z_start`` (D3), or ``None``."""
    for plane in planes:
        distance = abs(plane - state.z_start_um)
        if not distance <= state.jog_um:
            return (
                f"the check sweep around {target:.2f} µm would reach {plane:.2f} µm, "
                f"{distance:.2f} µm from the Z the scan starts at, beyond the Z jog "
                f"limit of {state.jog_um:g} µm"
            )
    return None


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``, as the scan's failed fields."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()
