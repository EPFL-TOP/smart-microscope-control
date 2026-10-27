"""``camera-calibration`` moves the stage a known distance and measures how the image moved (m3 §4.2).

The pure part (phase correlation, the matrix) is ``algorithm.py``. This is the
hardware loop around it: a reference frame, a probe in +X and one in +Y, a
return, and the decisions at each move that keep a calibration from doing
harm.

* **A return is a move, and it runs only when the measurement returned
  normally** (FM-88). After a stop, a halt or a fault the stage is left where
  it is and a WARNING says where; nothing is caught and carried on after
  (FM-64), so the operator's stop ends the stage and is not a failed field.
* **The return runs before the fit is judged.** An operator who calibrated on
  an empty field gets the stage back where they chose it, and then the reason
  the field was no good.
* **The stage is waited for before its start is read** (FM-85), and the step is
  sized from the camera's own pixel size only when that is finite and positive
  (FM-104): a pixel size of 0 means "not measured".
* **Nothing is written to the layer or the profile.** The camera has no setter
  and a stage that changed the stand's configuration would be hidden state; the
  scan plan picks this slot by objective and frame shape, and a WARNING names
  the profile value to update when the camera's own pixel size is off.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, ValidationError

from smc.hardware.capabilities import XY, Camera, ObjectiveTurret, XYStage
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.camera_calibration.algorithm import (
    DEFAULT_STEP_FRACTION,
    FALLBACK_STEP_UM,
    MIN_RESPONSE,
    Fit,
    fit,
    measure_shift,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.slots import CameraCalibration
from smc.run.document import RunDocument, RunDocumentError

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["CameraCalibrationParams", "CameraCalibrationPlugin"]

#: A step this fraction of the smaller side (in pixels) or more may have wrapped
#: in the correlation (m3 §4.1): the hint is added to a failed fit's reason.
_WRAP_FRACTION = 0.4
#: The camera's own pixel size is contradicted when the measured one differs by more.
_PIXEL_SIZE_TOLERANCE = 0.02
#: The stage coming back further than ``max(2 µm, 4 pixels)`` is reported.
_RETURN_FLOOR_UM = 2.0
_RETURN_PIXELS = 4.0


class CameraCalibrationParams(PluginParams):
    """The probe distance, the wait after a move, and an earlier calibration to reuse."""

    step_um: float | None = Field(
        None,
        gt=0.0,
        le=2000.0,
        description=(
            "Probe distance in µm; default a quarter of the smaller side of the "
            "frame, from the camera's pixel size (50 µm when it reports none)."
        ),
    )
    settle_ms: float = Field(
        0.0,
        ge=0.0,
        le=10_000.0,
        description="Wait after each move, before the frame is taken.",
    )
    from_run_doc: Path | None = Field(
        None,
        description=(
            "Reuse the calibration in this run document instead of measuring: "
            "nothing moves, and the objective and frame shape must be the same."
        ),
    )


class CameraCalibrationPlugin:
    """Measures the camera matrix by moving the stage in X and Y, or reuses an earlier one."""

    name = "camera-calibration"
    summary = (
        "Measure how the camera's pixels map to the stage: size, rotation, mirror."
    )
    Params = CameraCalibrationParams
    Result = CameraCalibration
    requires: tuple[type, ...] = (XYStage, Camera)
    provides = "camera-calibration"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: CameraCalibrationParams,
        ctx: RunContext,
    ) -> CameraCalibration:
        """Measure (two probes and a return) or reuse; raise when nothing usable comes out.

        Raises:
            PluginRunError: A dry run was asked to measure, the fit is
                unusable (no texture, a stage that did not move), or the
                calibration to reuse does not fit this stand.
            StopRequested: The operator stopped it; the stage is where it was.
        """
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        turret = microscope.get(ObjectiveTurret)
        objective = turret.current() if turret is not None else None
        height, width = camera.image_shape()
        shape = (int(height), int(width))

        if params.from_run_doc is not None:
            return _reuse(microscope, params.from_run_doc, ctx, objective, shape)
        if ctx.dry_run:
            raise PluginRunError(
                self.name,
                "a dry run cannot measure the camera; run it without --dry-run",
            )

        reported_um = camera.pixel_size_um()
        assumed: list[str] = []
        used_fallback = False
        if params.step_um is not None:
            step_um = params.step_um
        elif _usable(reported_um):
            step_um = DEFAULT_STEP_FRACTION * min(shape) * reported_um
        else:
            step_um = FALLBACK_STEP_UM
            used_fallback = True
            assumed.append(f"step of {FALLBACK_STEP_UM:g} µm: no pixel size was known")

        # A stage still travelling from an earlier move reads a fraction of it.
        xy.wait()
        start = xy.position_um()
        reference = camera.snap()
        probes = (("+X probe", step_um, 0.0), ("+Y probe", 0.0, step_um))
        shifts = []
        settle_s = params.settle_ms / 1000.0
        try:
            for done, (label, dx_um, dy_um) in enumerate(probes, start=1):
                ctx.raise_if_stopped()
                xy.move_to_um(start.x_um + dx_um, start.y_um + dy_um)
                ctx.wait_s(settle_s)
                shifts.append(measure_shift(reference, camera.snap()))
                ctx.progress(done, 3, label)
            # Both probes returned normally, so the return runs (FM-88), and
            # before the fit is judged: an empty field still gets the stage back.
            ctx.raise_if_stopped()
            xy.move_to_um(start.x_um, start.y_um)
            ctx.wait_s(settle_s)
        except BaseException:
            # A stop, a halt, a fault or a refused move: the stage stays where it
            # is, the exception goes on (FM-88, FM-64), and nothing is caught.
            _warn_left(ctx, xy, start)
            raise
        shift_x, shift_y = shifts

        fitted = fit(shift_x, shift_y, step_um=step_um, image_shape_px=shape)
        if not fitted.trustworthy:
            raise PluginRunError(
                self.name,
                _unusable_reason(fitted, step_um, shape, reported_um, used_fallback),
            )

        after_return = measure_shift(reference, camera.snap())
        ctx.progress(3, 3, "return")
        return_error_um: float | None = None
        # `>=` as the positive test: a response that is nan goes to the else.
        if after_return.response >= MIN_RESPONSE:
            (m00, m01), (m10, m11) = fitted.matrix_um_per_px
            return_error_um = math.hypot(
                m00 * after_return.d_col_px + m01 * after_return.d_row_px,
                m10 * after_return.d_col_px + m11 * after_return.d_row_px,
            )
            if return_error_um > max(
                _RETURN_FLOOR_UM, _RETURN_PIXELS * fitted.pixel_size_um
            ):
                assumed.append(
                    f"the stage came back {return_error_um:.1f} µm off: its "
                    "repeatability limits the scan's stitching"
                )
        else:
            assumed.append(
                "the return could not be checked: the frame after it did not "
                f"correlate ({after_return.response:.2f})"
            )

        if (
            _usable(reported_um)
            and abs(fitted.pixel_size_um / reported_um - 1.0) > _PIXEL_SIZE_TOLERANCE
        ):
            ctx.logger.warning(
                "the camera reports %g µm per pixel for %s, measured %g: update "
                "the profile's pixel size",
                reported_um,
                objective if objective is not None else "this stand",
                fitted.pixel_size_um,
            )
        return CameraCalibration(
            objective=objective,
            image_shape_px=shape,
            matrix_um_per_px=fitted.matrix_um_per_px,
            pixel_size_um=fitted.pixel_size_um,
            rotation_deg=fitted.rotation_deg,
            mirrored=fitted.mirrored,
            residual_um=fitted.residual_um,
            response=min(shift_x.response, shift_y.response),
            return_error_um=return_error_um,
            step_um=step_um,
            reused_from=None,
            assumed=tuple(assumed),
        )


def _usable(pixel_size_um: float) -> bool:
    """Whether a pixel size can size a move: finite and positive (FM-104).

    Infinity is above 0, and 0 means "not measured": neither is a size.
    """
    return math.isfinite(pixel_size_um) and pixel_size_um > 0.0


def _warn_left(ctx: RunContext, xy: XYStage, start: XY) -> None:
    """Say where the stage was left, reading it in a ``try`` of its own.

    A read that fails must not hide the error that is already on its way: the
    caller re-raises that one, whatever happens here.
    """
    try:
        here = xy.position_um()
    except Exception:
        ctx.logger.warning(
            "the stage is left at an unknown position, not returned to (%.1f, %.1f) µm",
            start.x_um,
            start.y_um,
        )
        return
    ctx.logger.warning(
        "the stage is left at (%.1f, %.1f) µm, not returned to (%.1f, %.1f) µm",
        here.x_um,
        here.y_um,
        start.x_um,
        start.y_um,
    )


def _unusable_reason(
    fitted: Fit,
    step_um: float,
    shape: tuple[int, int],
    reported_um: float,
    used_fallback: bool,
) -> str:
    """Why the fit is no good, the step that was tried, and a hint when it may have wrapped."""
    reason = "; ".join(fitted.notes) + f"; step {step_um:g} µm"
    wide = _usable(reported_um) and step_um / reported_um >= _WRAP_FRACTION * min(shape)
    if used_fallback or wide:
        reason += "; a step over 40 % of the field may wrap: give a smaller step_um"
    return reason


def _reuse(
    microscope: Microscope,
    path: Path,
    ctx: RunContext,
    objective: str | None,
    shape: tuple[int, int],
) -> CameraCalibration:
    """The calibration of another run document, if it is of this stand as it is now.

    Nothing moves, so a dry run may reuse. The runner's own rule decides
    whether the other document's stage is usable (``unusable``: not ``done``, another
    profile, a dry run read by a real one), then the objective in place and the
    frame shape must be the ones it was measured with: a calibration of another
    objective or binning is silently wrong, not merely stale. ``path`` is kept as
    typed in the result (FM-50: an absolute one names a user's folder).
    """
    name = CameraCalibrationPlugin.name
    where = path.as_posix()
    problem: str | None = None
    try:
        other = RunDocument.load(path)
    except RunDocumentError as exc:
        problem = f"{where}: {exc.reason}"
    if problem is not None:
        # Raised outside the handler: nothing here keeps the load's error.
        raise PluginRunError(name, problem)

    reason = other.unusable(
        name, profile=microscope.profile.microscope.name, dry_run=ctx.dry_run
    )
    record = other.provider(name)
    if reason is not None or record is None or record.result is None:
        raise PluginRunError(name, f"in {where}: {reason or 'it holds no result'}")
    try:
        found = CameraCalibration.model_validate(record.result)
    except ValidationError:
        problem = (
            f"the result in {where} is not one this smc reads; run "
            f"camera-calibration again there"
        )
    if problem is not None:
        raise PluginRunError(name, problem)

    if objective is not None and found.objective != objective:
        was = found.objective if found.objective is not None else "an unknown objective"
        raise PluginRunError(
            name,
            f"the calibration in {where} is of {was}, the objective in place "
            f"is {objective}; put {was} in place or run camera-calibration "
            f"with {objective}",
        )
    if found.image_shape_px != shape:
        raise PluginRunError(
            name,
            f"the calibration in {where} is of a frame of {found.image_shape_px}, "
            f"the camera gives {shape} (binning or the region changed); run "
            f"camera-calibration again with this camera setting",
        )
    when = (
        record.finished_utc.isoformat(timespec="seconds")
        if record.finished_utc is not None
        else "at an unknown time"
    )
    line = f"camera calibration reused from {where}, measured {when}"
    # Through validation, never `model_copy`, which skips it.
    return CameraCalibration.model_validate(
        {
            **found.model_dump(),
            "reused_from": where,
            "return_error_um": None,
            "assumed": [*found.assumed, line],
        }
    )
