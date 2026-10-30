"""``scan`` moves to every field of a scan plan, takes a frame and writes it as a TIFF (m3 §8).

A port of ``nikon-control``'s ``scan.run_iter``, redesigned around the layer.
The pure part (the fields left, the paths, the disk, the focus state, the
encoding) is ``algorithm.py``. This is the loop, and the decisions in it that
keep a night's scan from producing frames nobody can trust:

* **Everything that can be checked is checked before anything moves** (FM-30):
  the checkpoint being resumed and its files, every frame's path, the objective
  in place, the camera's frame and pixel size against the plan's, the
  autofocus, the free disk. A refusal leaves the stage where it was and writes
  no file.
* **The stand is checked again at both ends of every well.** A turret turned
  by hand, a binning changed, the magnifier lever moved (FM-103) make every
  later frame the wrong size for the plan's matrix: the scan ends, and the
  checkpoint keeps only the wells whose stand was the same from first to last
  field.
* **An autofocus switched off during the scan stays off.** A lost lock is
  engaged again once per field; one the operator switched off is not, and its
  fields are counted as unlocked.
* **A field that fails is recorded and skipped; a halt is not a failure**
  (FM-64). ``max_failed_in_row`` failures in a row end the scan: a stage that
  refuses every move must not read as a scan of failed fields.
* **The autofocus is used only as the operator left it.** One the operator
  left off is never engaged, and one that is on must have locked on this
  sample: engaging drives Z to the plane of whatever offset is stored (#205),
  which only a lock on this sample shows is the operator's.
* **Nothing moves to put something back** (FM-88): no Z move, no turret turn,
  no return to a start position, no ``finally`` that moves.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from smc.hardware.capabilities import (
    Autofocus,
    Camera,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import HardwareError, MicroscopeHaltedError, MotionStoppedError
from smc.plugins.api import PluginParams, RunContext
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
    size_differs,
    wells_to_scan,
)
from smc.plugins.slots import (
    FailedField,
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
_AGAIN = "run scan-plan again with the stand as it is now"

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
                frame came back in another shape; a frame could not be
                written (a full disk, a share that dropped or a file a
                viewer holds open is not a field's fault). The checkpoint
                keeps the wells done.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped;
                the scan ends where it is.
            StopRequested: The operator stopped it; ``--resume`` continues.
        """
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        plan = ctx.upstream("scan-plan", ScanPlanResult)

        token, wells_done, frames, failed = _resume(ctx, plan)
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
        stand = _check_stand(microscope, camera, plan, params)
        groups = wells_to_scan(plan.wells, plan.fields, set(wells_done))
        total = sum(len(fields) for _, fields in groups)
        _check_disk(ctx, camera, plan, total)

        z_stage = microscope.get(ZStage)
        af = stand.autofocus
        done = 0
        failed_in_row = 0
        unlocked_in_row = 0
        engage_attempts = 0
        engage_locks = 0
        # Whether the scan's own engage left the autofocus off (the protocol
        # switches it off when no lock comes, whether it returns or raises):
        # that one is the scan's to retry, until a field reads it locked.
        off_by_the_scan = False
        t0 = time.monotonic()
        for well, fields in groups:
            _check_well(camera, plan, stand, f" at well {well}")
            for planned in fields:
                ctx.raise_if_stopped()
                error: str | None = None
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
                            elif off_by_the_scan or af.is_engaged():
                                engage_attempts += 1
                                # Set before the call: one that raises leaves
                                # the autofocus off too (FM-129).
                                off_by_the_scan = True
                                locked = af.engage()
                                engage_locks += int(locked)
                                off_by_the_scan = not locked
                            unlocked_in_row = 0 if locked else unlocked_in_row + 1
                            if unlocked_in_row >= params.max_unlocked_in_row:
                                raise PluginRunError(
                                    _NAME,
                                    f"the autofocus did not lock on {unlocked_in_row} "
                                    f"fields in a row, the last {planned.name}; the "
                                    f"checkpoint keeps {len(wells_done)} wells: "
                                    "lock it on the sample, then run scan --resume",
                                )
                        z_um = z_stage.position_um() if z_stage is not None else None
                        pfs_offset = (
                            af.offset_native()
                            if stand.used and af is not None
                            else None
                        )
                        pixels = camera.snap()
                except (MicroscopeHaltedError, MotionStoppedError):
                    raise
                except HardwareError as exc:
                    error = _one_line(exc)

                done += 1
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
                if not ctx.dry_run:
                    target = paths[planned.well, planned.name]
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
                            ),
                        )
                    )
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

        assumed = stand.assumed
        if engage_attempts:
            # Attempts and locks: an engage that gave up is one, and not the other.
            assumed.append(
                f"the autofocus was engaged again on {engage_attempts} fields and "
                f"locked on {engage_locks} of them"
            )
        if ctx.dry_run:
            assumed.append(f"dry run: {done} fields walked, no frame taken")
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
        # Raised outside the handler: nothing here keeps the path's error.
        if unreadable is not None:
            raise _unreadable(unreadable)
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
            again = (
                "run camera-calibration, then scan-plan, again with the stand as it is now"
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
    # Raised outside the handler: nothing here keeps the OS's error (FM-98).
    if failure is not None:
        # ``AtomicWriteError`` is an ``OSError`` that is not a ``PermissionError``:
        # its cause says which refusal it wraps.
        cause = failure.cause if isinstance(failure, AtomicWriteError) else failure
        kept = failure.kept if isinstance(failure, AtomicWriteError) else None
        if kept is not None and isinstance(cause, PermissionError):
            # Only the final rename failed, and Windows refuses to replace a
            # file another program holds open, as a viewer on a well that a
            # ``--resume`` rewrites.
            raise PluginRunError(
                _NAME,
                f"the frame {target.relative} cannot be replaced: another program "
                "holds it open (a viewer, a backup or antivirus agent): close it, "
                "then run scan --resume",
            )
        raise PluginRunError(
            _NAME,
            f"the frame {target.relative} could not be written "
            f"({failure.strerror or failure}): free some space if the disk is "
            "full, or check the network share, then run scan --resume",
        )
    return meta


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``: a failed field's error, for a table."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()
