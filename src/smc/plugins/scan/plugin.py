"""``scan`` moves to every field of a scan plan, takes a frame and writes it as a TIFF (m3 §8).

A port of ``nikon-control``'s ``scan.run_iter``, redesigned around the layer.
The pure part (the fields left, the paths, the disk, the focus state, the
encoding) is ``algorithm.py``. This is the loop, and the decisions in it that
keep a night's scan from producing frames nobody can trust:

* **Everything that can be checked is checked before anything moves** (FM-30):
  the checkpoint being resumed and its files, the plan's field of view, every
  frame's path (resolved once, here; on Windows, its length while it is
  written), the objective in place, the camera's frame and pixel size against
  the plan's, the autofocus, the free disk. A refusal leaves the stage where it
  was and writes no file.
* **The stand is checked again at both ends of every well.** A turret turned
  by hand, a binning changed, the magnifier lever moved (FM-103) make every
  later frame the wrong size for the plan's matrix: the scan ends, and the
  checkpoint keeps only the wells whose stand was the same from first to last
  field.
* **An autofocus switched off during the scan stays off.** A lost lock is
  engaged again once per field, and so is the scan's own failed engage (one
  that raised too, FM-129); one the operator switched off is not, and its
  fields are counted as unlocked.
* **PFS never crosses a well's wall locked** (#416). Locked over a wall, it
  follows its reference to the plane its offset names in the next well, with
  no guard: before every well's first field the scan switches an engaged
  autofocus off and reads it back off, and that field engages it again under
  the layer's bound. A switch-off that cannot be made or read back ends the
  scan before anything moves to the well; a relock that moves Z out of its
  bound ends it too (the layer forgot the offset), with the wells done kept.
* **A move that returned is not a move that arrived** (FM-167). The readback is
  compared with the target, on both axes, before the autofocus, Z or the
  camera are touched: a field that landed off target is a failed field, not a
  frame of the wrong place. A dry run's readback is the target, so it is not
  compared.
* **A field that fails is recorded and skipped; a halt is not a failure**
  (FM-64). ``max_failed_in_row`` failures in a row end the scan: a stage that
  refuses every move must not read as a scan of failed fields. A scan that ends
  with no frame at all is a failure too (FM-166), not a ``done`` that ``cells``
  refuses hours later.
* **A frame that cannot be written ends the scan** as a ``PluginRunError`` that
  names the frame and says what to do (a file a viewer holds open, a full disk,
  a share that dropped), with the checkpoint keeping the wells done.
* **The autofocus is used only as the operator left it.** One the operator
  left off is never engaged, and one that is on must have locked on this
  sample: engaging drives Z to the plane of whatever offset is stored (#205),
  which only a lock on this sample shows is the operator's.
* **Nothing moves to put something back** (FM-88): no Z move, no turret turn,
  no return to a start position, no ``finally`` that moves.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smc.hardware.capabilities import (
    Autofocus,
    Camera,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import (
    AutofocusExcursionError,
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
)
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.cell_focus.algorithm import (
    DEFAULT_SCORE,
    LANDING_FRACTION,
    MAX_RANGE_UM,
    MAX_STEP_UM,
    MIN_STEP_UM,
    ScoreName,
    find_peak,
    plan_planes,
    score_frame,
    sweep_size_problem,
)
from smc.plugins.cell_focus.loop import (
    AutofocusStillOn,
    FocusDriveOffTarget,
    SweepBounds,
    check_sweep_bounds,
    move_checked,
    outside_limits,
    sweep,
    switch_off_for_sweep,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.scan.algorithm import (
    CALIBRATION_TOLERANCE,
    PROFILE_TOLERANCE,
    bytes_needed,
    collision_key,
    encode_frame,
    focus_state,
    frame_path,
    itemsize_of,
    landed_off_target,
    landing_error_um,
    landing_tolerance_um,
    offset_in_well_um,
    offset_lines,
    size_differs,
    well_centres_um,
    wells_to_scan,
)
from smc.plugins.slots import (
    FailedField,
    FieldFocus,
    FocusSweep,
    PlannedField,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)
from smc.run.files import (
    AtomicWriteError,
    new_token,
    resolve_in_run,
    temporary_path,
    write_atomically,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["ScanCheckpoint", "ScanParams", "ScanPlugin"]

_NAME = "scan"
_LOG = logging.getLogger(f"smc.plugins.{_NAME}")
_AGAIN = "run scan-plan again with the stand as it is now"
#: Added to §8's refusals when the focus on the cells switched the autofocus
#: off and nothing locked it again (FM-88: a failure restores nothing).
_FOCUS_LEFT_OFF = "; the focus on the cells left the autofocus off"

#: Windows' MAX_PATH, which counts the terminating NUL: a path of 260
#: characters or more fails where long paths are not enabled. Module
#: attributes, read at the call, so that a test sets them.
MAX_PATH: int = 260
ON_WINDOWS: bool = os.name == "nt"


class ScanParams(PluginParams):
    """How the autofocus is used, how many bad fields in a row end the scan, how far a landing may be."""

    autofocus: Literal["if-engaged", "required", "off"] = Field(
        "if-engaged",
        description=(
            "if-engaged: use it when it is on and locked; required: refuse to "
            "start unless it is; off: never touch it. It is never switched on "
            "by the scan."
        ),
    )
    max_failed_in_row: int = Field(
        3,
        ge=1,
        le=100,
        description="This many fields failing in a row end the scan.",
    )
    max_unlocked_in_row: int = Field(
        5,
        ge=1,
        le=1000,
        description=(
            "This many fields in a row on which the autofocus would not lock "
            "end the scan."
        ),
    )
    landing_tolerance: float = Field(
        0.05,
        gt=0.0,
        le=0.5,
        description=(
            "How far a field may land from its target, as a fraction of the "
            "smaller side of the plan's field of view; a field that landed "
            "farther is failed, not photographed. 0.05 is 4.8 µm for the "
            "demo's 10x (a field of view of 128 x 96 µm)."
        ),
    )
    cell_focus: Literal["off", "per-field", "per-well"] = Field(
        "off",
        description=(
            "The focus on the cells (m3 §21): per-field sweeps Z around every "
            "field and takes its frame at the sharpest plane; per-well sweeps "
            "until one field of the well finds a peak, then reuses its offset. "
            "The autofocus is off for the sweep and the frame, and the next "
            "field locks it again."
        ),
    )
    focus_range_um: float = Field(
        5.0,
        gt=0.0,
        le=MAX_RANGE_UM,
        description=(
            "The sweep's half-range in µm. Chosen, not measured: session 3 "
            "measures it on cells."
        ),
    )
    focus_step_um: float = Field(
        1.0,
        ge=MIN_STEP_UM,
        le=MAX_STEP_UM,
        description="The distance between two planes of the sweep, in µm. Chosen.",
    )
    focus_score: ScoreName = Field(
        DEFAULT_SCORE,
        description=(
            "The focus score. brenner is chosen from the literature, not "
            "measured on cells."
        ),
    )

    @model_validator(mode="after")
    def _a_sweep_with_planes(self) -> ScanParams:
        problem = sweep_size_problem(
            self.focus_range_um, self.focus_step_um, "focus_range_um", "focus_step_um"
        )
        if problem is not None:
            raise ValueError(problem)
        return self


class ScanCheckpoint(BaseModel):
    """What a resume needs: the frames' token, the wells done, their frames and failed fields.

    The token is a path part, so it is held to what ``new_token`` makes: a
    checkpoint edited to ``../..`` is refused when it is read, before
    anything moves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    token: str = Field(pattern=r"^[0-9a-f]{12}$")
    wells_done: tuple[str, ...]
    frames: tuple[ScanFrame, ...]
    failed: tuple[FailedField, ...]


@dataclass
class _Stand:
    """What the checks before the scan found, and what the loop reads again."""

    turret: ObjectiveTurret | None
    #: The camera's pixel size at the start; ``None`` when it reported none.
    p0: float | None
    exposure_ms: float
    autofocus: Autofocus | None
    #: Whether the scan uses the autofocus: it was on and locked at the start.
    used: bool
    assumed: list[str] = field(default_factory=list)


@dataclass
class _Focus:
    """The focus on the cells across fields (m3 §21.6): the flag, the counts and the times.

    ``off_by_focus`` says a focus step switched the autofocus off and no field
    has read it locked since: the next field engages it again, and a failure
    says it was left off. It is cleared as §8's ``off_by_the_scan`` is.
    """

    off_by_focus: bool = False
    #: Whether this field's step switched the autofocus off; reset per field.
    switched: bool = False
    #: Fields whose frame has a focus step, and their steps' time.
    fields: int = 0
    step_s: float = 0.0
    #: The engages that follow a focus step, and their time (FM-165).
    engages: int = 0
    engage_s: float = 0.0
    outcomes: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("peak", "flat", "at-edge", "two-peaks"), 0
        )
    )
    #: Fields whose autofocus read unlocked: no step (FM-154).
    skipped: int = 0
    dz_um: list[float] = field(default_factory=list)
    #: ``per-well``: the ``dz_um`` of the first field of each well with a peak.
    well_dz_um: dict[str, float] = field(default_factory=dict)

    def engage(self, af: Autofocus) -> bool:
        """Engage after a focus step, timed whether it locks, gives up or raises."""
        self.engages += 1
        started = time.perf_counter()
        try:
            return af.engage()
        finally:
            self.engage_s += time.perf_counter() - started

    def record(self, done: FieldFocus) -> None:
        """Count a field whose frame was taken after its focus step."""
        self.fields += 1
        self.step_s += done.duration_s
        self.dz_um.append(done.dz_um)

    def assumed(self, *, dry_run: bool, with_autofocus: bool, planes: int) -> list[str]:
        """``assumed``'s lines for the step (m3 §21.6), means to two decimals."""
        if dry_run:
            return [
                f"dry run: the focus on the cells was planned ({planes} planes per "
                "sweep), not run"
            ]
        lines: list[str] = []
        if self.fields:
            sweep_s = self.step_s / self.fields
            lock_s = self.engage_s / self.fields if with_autofocus else 0.0
            lock = f", {lock_s:.2f} s for the lock after" if with_autofocus else ""
            lines.append(
                f"the focus on the cells added {sweep_s + lock_s:.2f} s per field "
                f"({self.fields} fields: {sweep_s:.2f} s to sweep and move{lock})"
            )
        sweeps = sum(self.outcomes.values())
        if sweeps:
            lines.append(
                f"the focus on the cells found a peak on {self.outcomes['peak']} of "
                f"{sweeps} sweeps (flat {self.outcomes['flat']}, at an edge "
                f"{self.outcomes['at-edge']}, two peaks {self.outcomes['two-peaks']}): "
                "the others kept the centre"
            )
        if self.skipped:
            lines.append(
                f"the focus on the cells was skipped on {self.skipped} fields whose "
                "autofocus did not lock"
            )
        if with_autofocus and self.dz_um:
            lines.append(
                "each frame's pfs_offset names the plane PFS held, not the cells' "
                f"plane {min(self.dz_um):.2f}…{max(self.dz_um):.2f} µm from it: an "
                "export that engages PFS at that offset images the plane PFS held"
            )
        return lines


@dataclass(frozen=True)
class _Prepared:
    """A field's focus step up to its snap: when it started, and what the frame will record."""

    started: float
    source: Literal["swept", "well"]
    sweep: FocusSweep | None
    dz_um: float


def _focus_before_snap(
    z: ZStage,
    camera: Camera,
    af: Autofocus | None,
    params: ScanParams,
    bounds: SweepBounds,
    well: str,
    focus: _Focus,
    ctx: RunContext,
) -> _Prepared | str:
    """§21.6's steps 2-6 for one field, up to its snap; or the field's error when nothing moved.

    ``af`` is the autofocus when the scan uses it (the field read it locked),
    else ``None``. The centre is the Z PFS holds at the field, else the Z read
    before anything moved: it never follows the previous field's best plane,
    so nothing drifts from field to field (D2).

    Raises:
        FocusDriveOffTarget, AutofocusStillOn, HardwareError: The field fails;
            the autofocus stays off when it was switched off.
        MicroscopeHaltedError, MotionStoppedError, StopRequested: Passed.
    """
    centre = z.position_um() if af is not None else bounds.z_start_um
    if not math.isfinite(centre):
        return f"FocusDriveOffTarget: Z read {centre!r} at the field: no sweep is centred on it"
    known = focus.well_dz_um.get(well) if params.cell_focus == "per-well" else None
    planes: tuple[float, ...] = (
        (centre + known,)
        if known is not None
        else plan_planes(centre, params.focus_range_um, params.focus_step_um)
    )
    refusal = outside_limits(centre, planes, bounds.limits)
    if refusal is not None:
        return f"SweepOutsideLimits: {refusal}"

    started = time.perf_counter()
    if af is not None:
        focus.off_by_focus = True
        switch_off_for_sweep(af)
        focus.switched = True
    tolerance_um = LANDING_FRACTION * params.focus_step_um
    if known is not None:
        move_checked(z, centre + known, tolerance_um)
        return _Prepared(started, "well", None, known)

    readbacks, scores = sweep(
        z,
        camera,
        planes,
        params.focus_step_um,
        params.focus_score,
        raise_if_stopped=ctx.raise_if_stopped,
    )
    peak = find_peak(readbacks, scores)
    focus.outcomes[peak.outcome] += 1
    target = centre if peak.z_um is None else peak.z_um
    dz_um = target - centre
    if peak.z_um is not None and params.cell_focus == "per-well":
        focus.well_dz_um.setdefault(well, dz_um)
    move_checked(z, target, tolerance_um)
    return _Prepared(
        started,
        "swept",
        FocusSweep(
            score=params.focus_score,
            centre_um=centre,
            z_um=readbacks,
            scores=scores,
            outcome=peak.outcome,
            best_z_um=peak.z_um,
        ),
        dz_um,
    )


class ScanPlugin:
    """Moves to each field of the scan plan and writes one TIFF per field."""

    name = _NAME
    summary = (
        "Move to every field of the scan plan and write one TIFF per field, "
        "with the stage's readback and the focus state."
    )
    Params = ScanParams
    Result = ScanResult
    requires: tuple[type, ...] = (XYStage, Camera)
    provides = "scan"
    reads: tuple[str, ...] = ("scan-plan",)
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: ScanParams,
        ctx: RunContext,
    ) -> ScanResult:
        """Check the stand, then scan well by well, checkpointing after each.

        Raises:
            PluginRunError: A check before the scan refused it (nothing moved);
                the stand changed at a well; ``max_failed_in_row`` fields
                failed or ``max_unlocked_in_row`` would not lock in a row; a
                frame came back in another shape; every field failed (a scan
                with no frame at all); a frame could not be
                written (a full disk, a share that dropped or a file a
                viewer holds open is not a field's fault); the autofocus could
                not be switched off before a well, or an engage moved Z out of
                its bound. The checkpoint keeps the wells done.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped;
                the scan ends where it is.
            StopRequested: The operator stopped it; ``--resume`` continues.
        """
        focus = _Focus()
        reason: str | None = None
        try:
            return self._run(microscope, params, ctx, focus)
        except PluginRunError as exc:
            if not focus.off_by_focus:
                raise
            reason = exc.reason
        # Raised outside the handler (FM-98). The focus step switched the
        # autofocus off and nothing locked it again: a failure restores
        # nothing (FM-88), so the operator is told.
        raise PluginRunError(_NAME, reason + _FOCUS_LEFT_OFF)

    def _run(
        self,
        microscope: Microscope,
        params: ScanParams,
        ctx: RunContext,
        focus: _Focus,
    ) -> ScanResult:
        """``run``'s body; ``focus`` carries whether the focus step left the autofocus off."""
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        plan = ctx.upstream("scan-plan", ScanPlanResult)

        token, wells_done, frames, failed = _resume(ctx, plan)
        resumed_frames = len(frames)
        # Counted before the loop appends: a checkpoint of an smc that did not
        # record a frame's place in its well keeps its frames as they were, and
        # the result says how many (m3 §8, #430).
        kept_without_offsets = sum(frame.well_dx_um is None for frame in frames)
        if not all(side > 0.0 for side in plan.fov_um):
            # An edited run document (a ``nan`` is refused too): a tolerance of
            # nothing would fail every field.
            raise PluginRunError(
                _NAME,
                f"the plan's field of view {tuple(plan.fov_um)} µm is not positive: "
                f"{_AGAIN}",
            )
        tolerance_um = landing_tolerance_um(plan.fov_um, params.landing_tolerance)
        paths = _paths(ctx, plan, token)
        # Once per well, before anything moves, after the path refusals: a
        # well the plan lists and its plate does not hold is an edited run
        # document, and its field would be photographed with no place in a well.
        centres: dict[str, tuple[float, float]] = {}
        problem = ""
        try:
            centres = well_centres_um(plan)
        except ValueError as exc:
            problem = str(exc)
        if problem:
            # Raised outside the handler (FM-98).
            raise PluginRunError(_NAME, f"{problem}: {_AGAIN}")
        stand = _check_stand(microscope, camera, plan, params)
        if (
            params.autofocus != "off"
            and not stand.used
            and any(frame.focus == "locked" for frame in frames)
        ):
            # Without this, if-engaged would resume silently without PFS; a
            # stop during the focus on the cells leaves it off (m3 §21.6).
            raise PluginRunError(
                _NAME,
                "the scan being resumed used the autofocus, which is not locked now "
                "(the scan switches it off between wells, and a stop during the "
                "focus on the cells leaves it off): lock it on the sample, then run "
                "scan --resume",
            )
        groups = wells_to_scan(plan.wells, plan.fields, set(wells_done))
        total = sum(len(fields) for _, fields in groups)
        _check_disk(ctx, camera, plan, total)

        z_stage = microscope.get(ZStage)
        af = stand.autofocus
        focusing = params.cell_focus != "off"
        bounds: SweepBounds | None = None
        if focusing:
            if z_stage is None:
                raise PluginRunError(
                    _NAME, "cell_focus needs a focus drive, and this stand has none"
                )
            bounds = check_sweep_bounds(
                _NAME,
                "cell_focus",
                microscope,
                z_stage,
                "focus_range_um",
                params.focus_range_um,
                params.focus_step_um,
            )
            if params.autofocus == "off" and af is not None and af.is_engaged():
                # The sweep's Z moves go through the layer's interlock, which
                # switches an engaged autofocus off and leaves it off: with
                # autofocus=off nothing would say so, or lock it again. The
                # one read of an autofocus that "off" otherwise never reads.
                raise PluginRunError(
                    _NAME,
                    "cell_focus moves Z, which switches the autofocus off, and it "
                    "is on: run with autofocus=if-engaged to use it, or switch it "
                    "off first",
                )
        # The autofocus the focus step switches off: the one the scan uses.
        focus_af = af if stand.used else None
        unused_af_line: str | None = None
        if focusing and params.autofocus != "off" and af is not None and not stand.used:
            # A stop during the focus on the cells leaves the autofocus off,
            # and a stop in the first well leaves no checkpoint for the resume
            # refusal to read: the next run must not go on without it silently.
            unused_af_line = (
                "the autofocus was off when the scan started: the focus on the "
                "cells swept around the Z read then, and no frame was taken with "
                "it (a stop during the focus on the cells leaves it off: if it "
                "was locked on this plate, lock it again and run scan again)"
            )
            _LOG.warning("%s", unused_af_line)
        done = 0
        # The fields of this attempt that ended without an error: a frame
        # written, or in a dry run a move walked.
        walked_ok = 0
        failed_in_row = 0
        unlocked_in_row = 0
        engage_attempts = 0
        engage_locks = 0
        # Whether the scan's own engage left the autofocus off (the protocol
        # switches it off when no lock comes, whether it returns or raises):
        # that one is the scan's to retry, until a field reads it locked.
        off_by_the_scan = False
        # The wells whose wall the scan switched the autofocus off at, and the
        # engages that followed (m3 §8, #416): counted apart from §8's.
        wall_offs = 0
        wall_engages = 0
        wall_locks = 0
        # Whether the next engage is the first after a wall's switch-off; a
        # field that reads locked, or any engage made, ends its claim.
        after_wall = False
        t0 = time.monotonic()
        for well, fields in groups:
            _check_well(camera, plan, stand, f" at well {well}")
            # After the well's start check (FM-130), before anything moves to
            # it: PFS locked over a wall follows its reference to another plane,
            # with no guard (F11). A dry run never engages, so it switches nothing.
            if (
                stand.used
                and af is not None
                and not ctx.dry_run
                and _switch_off_at_wall(af, well)
            ):
                off_by_the_scan = True
                wall_offs += 1
                after_wall = True
            for planned in fields:
                ctx.raise_if_stopped()
                error: str | None = None
                tripped: str | None = None
                prepared: _Prepared | None = None
                cell_focus: FieldFocus | None = None
                pfs_offset: float | None = None
                offset_read = False
                in_step = False
                focus.switched = False
                try:
                    readback = xy.move_to_um(planned.x_um, planned.y_um)
                    if not ctx.dry_run:
                        # Before anything else is read or moved: a field that
                        # did not arrive is not photographed, and engaging the
                        # autofocus for it would drive Z for nothing. A dry
                        # run's readback is the target, so nothing is measured.
                        off_um = landing_error_um(
                            (planned.x_um, planned.y_um),
                            (readback.x_um, readback.y_um),
                        )
                        if landed_off_target(off_um, tolerance_um):
                            error = (
                                f"StageOffTarget: the readback is {off_um:.1f} µm "
                                f"from the target, more than {tolerance_um:.1f} µm "
                                f"(target ({planned.x_um:.1f}, {planned.y_um:.1f}), "
                                f"readback ({readback.x_um:.1f}, {readback.y_um:.1f}))"
                            )
                    if not ctx.dry_run and error is None:
                        locked = False
                        if stand.used and af is not None:
                            locked = af.is_locked()
                            # A lost lock is engaged again; an autofocus the
                            # operator switched off during the scan stays off,
                            # and its fields count as unlocked.
                            if locked:
                                # A lock someone else took ends the scan's own
                                # failure: a switch-off after it is not ours.
                                off_by_the_scan = False
                                focus.off_by_focus = False
                                after_wall = False
                            elif (
                                off_by_the_scan or focus.off_by_focus or af.is_engaged()
                            ):
                                # Set before the call: one that raises leaves
                                # the autofocus off too (FM-129).
                                off_by_the_scan = True
                                wall_engage = after_wall
                                after_wall = False
                                if focus.off_by_focus:
                                    # The lock after a focus step: timed and
                                    # counted apart from §8's re-engages (FM-165).
                                    locked = focus.engage(af)
                                elif wall_engage:
                                    wall_engages += 1
                                    locked = af.engage()
                                    wall_locks += int(locked)
                                else:
                                    engage_attempts += 1
                                    locked = af.engage()
                                    engage_locks += int(locked)
                                off_by_the_scan = not locked
                                if locked:
                                    focus.off_by_focus = False
                            unlocked_in_row = 0 if locked else unlocked_in_row + 1
                            if unlocked_in_row >= params.max_unlocked_in_row:
                                raise PluginRunError(
                                    _NAME,
                                    f"the autofocus did not lock on {unlocked_in_row} "
                                    f"fields in a row, the last {planned.name}; the "
                                    f"checkpoint keeps {len(wells_done)} wells: "
                                    "lock it on the sample, then run scan --resume",
                                )
                        if focusing and bounds is not None and z_stage is not None:
                            if focus_af is not None and not locked:
                                # Its Z is where a search left it (FM-154).
                                focus.skipped += 1
                            else:
                                in_step = True
                                if focus_af is not None:
                                    # The lock the sweep is centred on, read
                                    # before the switch-off.
                                    pfs_offset = focus_af.offset_native()
                                    offset_read = True
                                outcome = _focus_before_snap(
                                    z_stage,
                                    camera,
                                    focus_af,
                                    params,
                                    bounds,
                                    planned.well,
                                    focus,
                                    ctx,
                                )
                                if isinstance(outcome, str):
                                    error = outcome
                                else:
                                    prepared = outcome
                    if not ctx.dry_run and error is None:
                        z_um = z_stage.position_um() if z_stage is not None else None
                        if not offset_read:
                            pfs_offset = (
                                af.offset_native()
                                if stand.used and af is not None
                                else None
                            )
                        pixels = camera.snap()
                        if prepared is not None:
                            cell_focus = FieldFocus(
                                source=prepared.source,
                                sweep=prepared.sweep,
                                dz_um=prepared.dz_um,
                                frame_score=score_frame(
                                    np.asarray(pixels), params.focus_score
                                ),
                                duration_s=time.perf_counter() - prepared.started,
                            )
                except (MicroscopeHaltedError, MotionStoppedError):
                    raise
                except AutofocusExcursionError as exc:
                    # Its text only: the exception kept in a local of this
                    # frame, raised through it, would hold the frame and the
                    # stand until the cyclic collector (FM-98).
                    tripped = _one_line(exc)
                except (HardwareError, FocusDriveOffTarget, AutofocusStillOn) as exc:
                    error = _one_line(exc)
                if tripped is not None:
                    # Not a failed field: the layer switched the autofocus off and
                    # forgot its offset (m2 §12.2 D2), so every later engage would
                    # be refused and the scan would fail fields with a message
                    # that blames the stage. Raised outside the handler (FM-98).
                    # The trip, not the focus step, is what left PFS off: the
                    # text stays §8's, without ``run``'s suffix.
                    focus.off_by_focus = False
                    raise PluginRunError(
                        _NAME,
                        "the autofocus moved Z out of its bound while it locked at "
                        f"{planned.name} (well {planned.well}): {tripped}; "
                        f"the checkpoint keeps {len(wells_done)} wells. Its offset "
                        "names another plane here than where it held (m2 §12.11): "
                        f"with PFS off, focus on the sample in well {planned.well} by "
                        "eye and lock it from below (m2 §12.10), then run scan "
                        "--resume (the well is scanned again whole)",
                    )

                done += 1
                if error is not None and in_step:
                    _LOG.warning(
                        "%s: the focus on the cells failed (%s)%s",
                        planned.name,
                        error,
                        "; the autofocus is left off" if focus.switched else "",
                    )
                if error is not None:
                    failed.append(
                        FailedField(
                            well=planned.well,
                            index=planned.index,
                            name=planned.name,
                            x_um=planned.x_um,
                            y_um=planned.y_um,
                            error=error,
                        )
                    )
                    failed_in_row += 1
                    ctx.progress(done, total, planned.name)
                    if failed_in_row >= params.max_failed_in_row:
                        raise PluginRunError(
                            _NAME,
                            f"{failed_in_row} fields failed in a row, the last "
                            f"{planned.name}: {error}; the checkpoint keeps "
                            f"{len(wells_done)} wells: check the stage, then run "
                            "scan --resume",
                        )
                    continue

                failed_in_row = 0
                if cell_focus is not None:
                    focus.record(cell_focus)
                if not ctx.dry_run:
                    target = paths[planned.well, planned.name]
                    # The place in the well is the readback's, not the target's:
                    # a frame that landed 3 µm off reads 3 µm more (m3 §8, #430).
                    dx_um, dy_um = offset_in_well_um(
                        centres[planned.well], readback.x_um, readback.y_um
                    )
                    frames.append(
                        _write_frame(
                            plan,
                            planned,
                            target,
                            np.asarray(pixels),
                            ScanFrame(
                                well=planned.well,
                                index=planned.index,
                                name=planned.name,
                                file=target.relative,
                                x_um=readback.x_um,
                                y_um=readback.y_um,
                                z_um=z_um,
                                focus=focus_state(af is not None, stand.used, locked),
                                pfs_offset=pfs_offset,
                                t_s=time.monotonic() - t0,
                                cell_focus=cell_focus,
                                well_dx_um=dx_um,
                                well_dy_um=dy_um,
                            ),
                        )
                    )
                walked_ok += 1
                ctx.progress(done, total, planned.name)

            # Again after the last field: a lever moved in the middle of the
            # well made some of its frames at another scale, and a well is
            # only marked done when the stand was the same at both ends.
            _check_well(camera, plan, stand, f" at the end of well {well}")
            wells_done.append(well)
            ctx.save_checkpoint(
                ScanCheckpoint(
                    token=token,
                    wells_done=tuple(wells_done),
                    frames=tuple(frames),
                    failed=tuple(failed),
                )
            )

        if failed and walked_ok == 0 and resumed_frames == 0:
            # A scan of nothing is not ``done``: ``cells`` refuses it hours
            # later. The checkpoint keeps every well, so a resume meets this
            # again: the way out is a scan without it.
            raise PluginRunError(
                _NAME,
                f"every field failed ({len(failed)}), the first {failed[0].name}: "
                f"{failed[0].error}: check the stage and the camera, then run scan "
                "again without --resume (a resume skips the wells already walked)",
            )

        assumed = stand.assumed
        if focus.off_by_focus and af is not None and not ctx.dry_run:
            # The last field's step left the autofocus off and no next field
            # locks it: one engage, on a normal end only (a failure or a stop
            # leaves it off, FM-88). Every frame is written by now, so a lock
            # that does not come is said, not a failure.
            why: str | None = None
            try:
                if focus.engage(af):
                    focus.off_by_focus = False
                else:
                    why = "no lock came"
            except (MicroscopeHaltedError, MotionStoppedError):
                raise
            except HardwareError as exc:
                why = _one_line(exc)
            if why is not None:
                line = (
                    f"the autofocus did not lock again after the last field ({why}): "
                    "it is off"
                )
                _LOG.warning("%s", line)
                assumed.append(line)
        if engage_attempts:
            # Attempts and locks: an engage that gave up is one, and not the other.
            assumed.append(
                f"the autofocus was engaged again on {engage_attempts} fields and "
                f"locked on {engage_locks} of them"
            )
        if wall_offs:
            assumed.append(
                f"the autofocus was switched off before the first field of "
                f"{wall_offs} wells, so that no lock crossed a well's wall, and "
                f"engaged again after it on {wall_engages} fields, locked on "
                f"{wall_locks} of them"
            )
        if ctx.dry_run:
            assumed.append(f"dry run: {done} fields walked, no frame taken")
        if unused_af_line is not None:
            assumed.append(unused_af_line)
        if focusing and bounds is not None:
            assumed.extend(
                focus.assumed(
                    dry_run=ctx.dry_run,
                    with_autofocus=focus_af is not None,
                    planes=len(
                        plan_planes(
                            bounds.z_start_um,
                            params.focus_range_um,
                            params.focus_step_um,
                        )
                    ),
                )
            )
        if not ctx.dry_run:
            # What the frames' places in their wells rest on: last, after every
            # line the scan already writes. A dry run takes no frame.
            assumed.extend(
                offset_lines(
                    plan,
                    kept=resumed_frames,
                    kept_without_offsets=kept_without_offsets,
                )
            )
        return ScanResult(
            folder=f"scan/{token}",
            objective=plan.objective,
            pixel_size_um=plan.pixel_size_um,
            matrix_um_per_px=plan.matrix_um_per_px,
            image_shape_px=plan.image_shape_px,
            exposure_ms=stand.exposure_ms,
            # A dry run took no frame, whatever a resumed checkpoint holds.
            frames=() if ctx.dry_run else tuple(frames),
            failed=tuple(failed),
            assumed=tuple(assumed),
        )


def _resume(
    ctx: RunContext, plan: ScanPlanResult
) -> tuple[str, list[str], list[ScanFrame], list[FailedField]]:
    """The token, wells done, frames and failed fields to continue from; or a fresh start.

    A checkpoint's wells must be the plan's, and every frame it names must
    still be in the run folder: a result that names a file nobody can read is
    not a scan, and the cells stage would find out hours later.
    """
    checkpoint = ctx.checkpoint(ScanCheckpoint)
    if checkpoint is None:
        return new_token(), [], [], []
    foreign = [well for well in checkpoint.wells_done if well not in plan.wells]
    if foreign:
        raise PluginRunError(
            _NAME,
            f"the checkpoint names well {foreign[0]}, which the scan plan does not "
            "have: run scan without --resume",
        )
    missing = [
        frame.file
        for frame in checkpoint.frames
        if not _is_file_in_run(ctx, frame.file)
    ]
    if missing:
        raise PluginRunError(
            _NAME,
            f"{len(missing)} of the checkpoint's {len(checkpoint.frames)} frame files "
            f"are missing from the run folder, the first {missing[0]}: run scan "
            "without --resume",
        )
    return (
        checkpoint.token,
        list(checkpoint.wells_done),
        list(checkpoint.frames),
        list(checkpoint.failed),
    )


def _is_file_in_run(ctx: RunContext, relative: str) -> bool:
    """Whether ``relative`` is a file inside the run folder; a path that leaves it is not.

    A folder the OS cannot read now (a network share that dropped for a
    moment: Windows raises out of ``resolve``, FM-116) is not a missing file:
    reading it as one would tell the operator to drop a night's checkpoint.

    Raises:
        PluginRunError: The run folder cannot be read; ``--resume`` again later.
    """
    cause: OSError | None = None
    try:
        return resolve_in_run(ctx.run_dir, relative).is_file()
    except ValueError as exc:
        if not isinstance(exc.__cause__, OSError):
            return False
        cause = exc.__cause__
    except OSError as exc:
        cause = exc
    # Raised outside the handler: nothing here keeps the OS's error.
    raise _unreadable(cause)


def _unreadable(cause: OSError) -> PluginRunError:
    """The refusal for a run folder the OS will not read: it says why and what to do."""
    return PluginRunError(
        _NAME,
        f"the run folder cannot be read ({cause.strerror or cause}): check the "
        "disk or the network share, then run scan --resume",
    )


class _Target(NamedTuple):
    """Where a field's frame goes: the text the result names, and the path it is written to."""

    #: ``scan/<token>/<well>/<name>.tif``, POSIX, as ``ScanFrame.file``.
    relative: str
    #: What ``resolve_in_run`` made of it, once, before anything moved (FM-116).
    absolute: Path


def _paths(
    ctx: RunContext, plan: ScanPlanResult, token: str
) -> dict[tuple[str, str], _Target]:
    """Every field's frame path, keyed by well and name; refused before anything moves.

    The plan is a run document's, which may have been edited or copied: a
    field of a well the plan does not list, a name that is not a file name (or
    is a Windows device), two fields sharing one file (compared without case:
    NTFS and APFS do not tell ``A1_0001`` from ``a1_0001``), or a path Windows
    would refuse are refused here, not at the field. Each path is resolved
    once, here: resolving it again at every frame would let a share that drops
    mid-scan end the run as a ``ValueError`` (FM-116).
    """
    wells = set(plan.wells)
    targets: dict[tuple[str, str], _Target] = {}
    first_of: dict[tuple[str, str], tuple[str, str]] = {}
    for planned in plan.fields:
        if planned.well not in wells:
            raise PluginRunError(
                _NAME,
                f"field {planned.name!r} is of well {planned.well!r}, which the "
                f"scan plan does not list: {_AGAIN}",
            )
        problem = ""
        unreadable: OSError | None = None
        try:
            relative = frame_path(token, planned.well, planned.name)
            absolute = resolve_in_run(ctx.run_dir, relative)
        except ValueError as exc:
            if isinstance(exc.__cause__, OSError):
                unreadable = exc.__cause__
            else:
                problem = str(exc)
        # Raised outside the handler. The OSError's traceback holds the
        # resolver's frame, whose caller is this one: the local goes before this
        # frame can be left, or the plan and the context stay alive with it (FM-98).
        if unreadable is not None:
            try:
                raise _unreadable(unreadable)
            finally:
                del unreadable
        if problem:
            raise PluginRunError(
                _NAME, f"field {planned.name!r} of well {planned.well!r}: {problem}"
            )
        key = (planned.well, planned.name)
        if key in targets:
            raise PluginRunError(
                _NAME,
                f"two fields of well {planned.well} are named {planned.name}, so "
                f"they would write one file: {_AGAIN}",
            )
        folded = collision_key(planned.well, planned.name)
        if folded in first_of:
            first_well, first_name = first_of[folded]
            raise PluginRunError(
                _NAME,
                f"fields {first_name!r} of well {first_well!r} and "
                f"{planned.name!r} of well {planned.well!r} would write one file, "
                f"as the file system does not tell upper from lower case: {_AGAIN}",
            )
        if ON_WINDOWS:
            # The temporary file is the longest path the write makes.
            length = len(str(temporary_path(absolute)))
            if length >= MAX_PATH:
                raise PluginRunError(
                    _NAME,
                    f"a frame's path would reach {length} characters while it is "
                    f"written, and Windows takes {MAX_PATH - 1} at most unless "
                    "long paths are enabled (not assumed): move the run document "
                    "to a shorter folder, then run scan again",
                )
        first_of[folded] = key
        targets[key] = _Target(relative, absolute)
    return targets


def _check_stand(
    microscope: Microscope,
    camera: Camera,
    plan: ScanPlanResult,
    params: ScanParams,
) -> _Stand:
    """The objective, the camera and the autofocus, checked in this order before anything moves (FM-30)."""
    assumed: list[str] = []

    turret = microscope.get(ObjectiveTurret)
    if turret is None:
        assumed.append("the objective was not checked: the stand has no turret")
    elif plan.objective is None:
        assumed.append(
            "the plan names no objective: the objective in place was not checked"
        )
        turret = None
    else:
        current = turret.current()
        if current != plan.objective:
            raise PluginRunError(
                _NAME,
                f"the objective in place is {current!r}; the scan was planned for "
                f"{plan.objective!r}. Turn the turret to {plan.objective!r} (smc "
                "never turns it during a scan), or run scan-plan again with "
                f"objective={current}",
            )

    _check_shape(camera, plan, "")
    exposure_ms = camera.exposure_ms()
    p0: float | None = camera.pixel_size_um()
    if not (p0 is not None and math.isfinite(p0) and p0 > 0.0):
        # FM-104: 0 is "not measured", and infinity is above 0.
        p0 = None
        assumed.append(
            "the camera reports no pixel size: the plan's size and a change during "
            "the scan were not checked"
        )
    elif plan.matrix_source == "parameter":
        assumed.append(
            "the plan's pixel size was given as a parameter: not compared with "
            "the camera's"
        )
    else:
        tolerance = (
            PROFILE_TOLERANCE
            if plan.matrix_source == "profile"
            else CALIBRATION_TOLERANCE
        )
        if size_differs(p0, plan.pixel_size_um, tolerance):
            # A plan that read another document's calibration (m3 §23) is sent
            # there: run.json holds the registration's calibration, and running
            # one into it makes the plate stale.
            into = (
                f" into {plan.calibration_from}"
                if plan.calibration_from is not None
                else ""
            )
            again = (
                f"run camera-calibration{into}, then scan-plan, again with the stand "
                "as it is now"
                if plan.matrix_source == "camera-calibration"
                else _AGAIN
            )
            raise PluginRunError(
                _NAME,
                f"the camera reports {p0:g} µm per pixel; the plan was made for "
                f"{plan.pixel_size_um:g} µm ({plan.matrix_source}), more than "
                f"{tolerance:.0%} away (a binning, the magnifier lever): {again}",
            )

    af, used = _check_autofocus(microscope, params)
    return _Stand(
        turret=turret,
        p0=p0,
        exposure_ms=exposure_ms,
        autofocus=af,
        used=used,
        assumed=assumed,
    )


def _check_autofocus(
    microscope: Microscope, params: ScanParams
) -> tuple[Autofocus | None, bool]:
    """The autofocus and whether the scan uses it; refused when it cannot be used as asked.

    ``off`` reads nothing from it. Otherwise it is used only when it is on and
    locked now: engaging it would drive Z to the plane of whatever offset is
    stored, and only a lock on this sample shows that offset is the
    operator's (#205).
    """
    af = microscope.get(Autofocus)
    if params.autofocus == "off":
        return af, False
    engaged = af is not None and af.is_engaged()
    locked = engaged and af is not None and af.is_locked()
    if params.autofocus == "required" and not locked:
        raise PluginRunError(
            _NAME,
            "autofocus=required, and the autofocus is "
            f"{'missing' if af is None else 'on but not locked' if engaged else 'off'}: "
            "engage PFS on the stand and let it lock on the sample before the scan",
        )
    if engaged and not locked:
        raise PluginRunError(
            _NAME,
            "the autofocus is on but not locked: lock it on the sample, or switch "
            "it off and run with autofocus=off",
        )
    return af, locked


def _switch_off_at_wall(af: Autofocus, well: str) -> bool:
    """Switch an engaged autofocus off before ``well``; whether this switched it off (m3 §8).

    PFS locked over a well's wall follows its reference to the plane its
    offset names in the next well, which moved by at least 104 µm between two
    wells of one plate at the 40x (F11), with no action holding the microscope
    and no guard (m2 §12.2 D2). Switched off, Z stays where PFS held it, and
    the well's first field engages it again under D2's band. One that is off
    is left as it is: the operator's switch-off is not undone (FM-129).

    Raises:
        PluginRunError: The switch-off or a read of it raised, or it still
            reads on: PFS may be on, so nothing may move to the well (FM-19).
        MicroscopeHaltedError, MotionStoppedError: Passed (FM-64).
    """
    cause: str | None = None
    still_on = False
    try:
        if not af.is_engaged():
            return False
        af.disengage()
        still_on = af.is_engaged()
    except (MicroscopeHaltedError, MotionStoppedError):
        raise
    except HardwareError as exc:
        cause = _one_line(exc)
    # Raised outside the handler (FM-98).
    if cause is not None:
        raise PluginRunError(
            _NAME,
            f"the autofocus could not be switched off before well {well} ({cause}): "
            "nothing moved to it, and PFS may still be on: switch it off at the "
            "stand, then run scan --resume",
        )
    if still_on:
        raise PluginRunError(
            _NAME,
            f"the autofocus still reads on after it was switched off before well "
            f"{well}: nothing moved to it: switch it off at the stand, then run "
            "scan --resume",
        )
    return True


def _check_disk(
    ctx: RunContext, camera: Camera, plan: ScanPlanResult, n_fields: int
) -> None:
    """Refuse a scan whose frames the run folder's disk cannot hold, before the first move."""
    need = bytes_needed(n_fields, plan.image_shape_px, itemsize_of(camera.bit_depth()))
    unread: OSError | None = None
    try:
        free = shutil.disk_usage(ctx.run_dir).free
    except OSError as exc:
        unread = exc
    if unread is not None:
        raise _unreadable(unread)
    if free <= need:
        raise PluginRunError(
            _NAME,
            f"the run folder's disk has {free / 1e6:,.1f} MB free and the "
            f"{n_fields} frames need {need / 1e6:,.1f} MB: free some space, or "
            "put the run document on another disk",
        )


def _check_shape(camera: Camera, plan: ScanPlanResult, where: str) -> None:
    """The camera's frame must be the plan's: the matrix is only right on it (FM-73)."""
    height, width = camera.image_shape()
    shape = (int(height), int(width))
    if shape != tuple(plan.image_shape_px):
        raise PluginRunError(
            _NAME,
            f"the camera gives frames of {shape} pixels{where}; the plan was made "
            f"for {tuple(plan.image_shape_px)}: set the binning and region of "
            f"interest back, or {_AGAIN}",
        )


def _check_well(camera: Camera, plan: ScanPlanResult, stand: _Stand, at: str) -> None:
    """At both ends of a well: the objective, the frame and the pixel size are still the start's (FM-103).

    ``at`` says where, for the message (`` at well A2``). A ``HardwareError``
    from these reads is the stand's, not a field's: it ends the scan.
    """
    resume = "then run scan --resume"
    if stand.turret is not None:
        current = stand.turret.current()
        if current != plan.objective:
            raise PluginRunError(
                _NAME,
                f"the objective in place is now {current!r}{at}; the scan was "
                f"planned for {plan.objective!r}: turn the turret back to "
                f"{plan.objective!r}, {resume}",
            )
    _check_shape(camera, plan, at)
    if stand.p0 is not None:
        now = camera.pixel_size_um()
        if not math.isfinite(now) or size_differs(now, stand.p0, PROFILE_TOLERANCE):
            raise PluginRunError(
                _NAME,
                f"the camera reports {now:g} µm per pixel{at}, {stand.p0:g} at the "
                "start of the scan (the magnifier lever, a binning): set it back, "
                f"{resume}",
            )


def _write_frame(
    plan: ScanPlanResult,
    planned: PlannedField,
    target: _Target,
    pixels: np.ndarray,
    meta: ScanFrame,
) -> ScanFrame:
    """Write one frame whole (FM-38) and return its record; a frame of another shape is not written.

    An ``OSError`` writing it is not the field's fault (a full disk, a share
    that dropped, a file a viewer holds open) and every next frame would fail
    the same way: it ends the scan as a ``PluginRunError`` that names the
    frame and says what to do, with the checkpoint keeping the wells done. A
    ``BaseException`` that is not an ``OSError`` (a Ctrl-C) passes.
    """
    if tuple(pixels.shape) != tuple(plan.image_shape_px):
        raise PluginRunError(
            _NAME,
            f"the frame of {planned.name} has shape {tuple(pixels.shape)}; the "
            f"plan's frames are {tuple(plan.image_shape_px)}: check the camera's "
            "binning and region of interest, then run scan --resume",
        )
    data = encode_frame(pixels, meta)
    failure: OSError | None = None
    try:
        write_atomically(target.absolute, data)
    except OSError as exc:
        failure = exc
    # Raised outside the handler (FM-98).
    if failure is not None:
        # ``AtomicWriteError`` is an ``OSError`` that is not a ``PermissionError``:
        # its cause says which refusal it wraps.
        cause = failure.cause if isinstance(failure, AtomicWriteError) else failure
        kept = failure.kept if isinstance(failure, AtomicWriteError) else None
        if kept is not None and isinstance(cause, PermissionError):
            # Only the final rename failed, and Windows refuses to replace a
            # file another program holds open, as a viewer on a well that a
            # ``--resume`` rewrites.
            reason = (
                f"the frame {target.relative} cannot be replaced: another program "
                "holds it open (a viewer, a backup or antivirus agent): close it, "
                "then run scan --resume"
            )
        else:
            reason = (
                f"the frame {target.relative} could not be written "
                f"({failure.strerror or failure}): free some space if the disk is "
                "full, or check the network share, then run scan --resume"
            )
        # The error's traceback reaches this frame: a local that keeps it is a
        # cycle that holds this frame, and what it holds, until the collector.
        del failure, cause
        raise PluginRunError(_NAME, reason)
    return meta


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``: a failed field's error, for a table."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()
