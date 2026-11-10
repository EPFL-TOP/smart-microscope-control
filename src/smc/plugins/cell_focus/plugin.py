"""``focus-on-cells`` sweeps Z around where the stage is and finds the sharpest plane (m3 §21.7).

The operator's tool, and the Z-stack maker of D6: every frame of the sweep is
kept as a TIFF with its Z, so a run at a stand is a stack that
``scripts/dev/focus_scores.py`` scores offline. The pure part is
``algorithm.py`` and the sweep is ``loop.py``; this is the run around them:

* **Everything that can be checked is checked before anything moves**, dry run
  included (FM-30): an autofocus that is on must be locked (its offset is then
  the plane the sweep is centred on), the stand must have Z soft limits, the
  range must not exceed the Z jog limit, and every plane must lie inside the
  limits. A refusal moves nothing and writes nothing.
* **The autofocus is off for the sweep** (D1), switched off explicitly and
  read back, and locked again at the end only when the run ends normally and
  ``relock`` asks for it.
* **A failure keeps the frames** (each is written as it is taken) and says
  where the sweep ended, whether the autofocus was left off, and that Z was
  left where it stopped. Nothing moves to put something back (FM-88), and a
  halt or a stop is not a failure (FM-64).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import tifffile
from pydantic import Field, model_validator

from smc.hardware.capabilities import (
    Autofocus,
    Camera,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import HardwareError, MicroscopeHaltedError, MotionStoppedError
from smc.plugins.api import PluginParams, RunContext, StopRequested
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
    check_sweep_bounds,
    move_checked,
    sweep,
    switch_off_for_sweep,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.slots import CellFocusResult, FocusSweep
from smc.run.files import (
    AtomicWriteError,
    new_token,
    resolve_in_run,
    temporary_path,
    write_atomically,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["FocusOnCellsParams", "FocusOnCellsPlugin"]

_NAME = "focus-on-cells"
_LOG = logging.getLogger(f"smc.plugins.{_NAME}")
_BEST = "best.tif"
#: Logged whenever a run that switched the autofocus off ends early: nothing
#: locks it again or moves Z back (FM-88), so the operator is told.
_LEFT_OFF_WARNING = (
    "focus-on-cells ended early: the autofocus was left off and Z was left where "
    "it stopped; {n} frames are kept in {folder}/. Lock the autofocus on the "
    "sample again before imaging"
)
#: The autofocus at the start, as ``CellFocusResult.autofocus`` names it.
_AutofocusState = Literal["locked", "off", "none"]

#: Windows' MAX_PATH, as the scan's (m3 §8). Module attributes, read at the
#: call, so that a test sets them.
MAX_PATH: int = 260
ON_WINDOWS: bool = os.name == "nt"

#: The errors of a sweep that end the run with its frames kept; a halt, a stop
#: and an operator's stop are not among them (FM-64).
_SWEEP_ERRORS = (HardwareError, FocusDriveOffTarget, AutofocusStillOn)


class FocusOnCellsParams(PluginParams):
    """How far and how finely to sweep, which score, and whether to lock the autofocus again."""

    range_um: float = Field(
        5.0,
        gt=0.0,
        le=MAX_RANGE_UM,
        description=(
            "The sweep's half-range in µm, around the Z where the stage is (or "
            "the plane the autofocus holds). Chosen, not measured: session 3 "
            "measures it on cells."
        ),
    )
    step_um: float = Field(
        1.0,
        ge=MIN_STEP_UM,
        le=MAX_STEP_UM,
        description="The distance between two planes, in µm. Chosen, not measured.",
    )
    score: ScoreName = Field(
        DEFAULT_SCORE,
        description=(
            "The focus score. brenner is chosen from the literature, not "
            "measured on cells: scripts/dev/focus_scores.py compares them on "
            "the stacks this tool keeps."
        ),
    )
    relock: bool = Field(
        True,
        description=(
            "Lock the autofocus again after the sweep, when it was locked at the "
            "start: it takes Z back to the plane it holds."
        ),
    )

    @model_validator(mode="after")
    def _a_sweep_with_planes(self) -> FocusOnCellsParams:
        problem = sweep_size_problem(self.range_um, self.step_um, "range_um", "step_um")
        if problem is not None:
            raise ValueError(problem)
        return self


class FocusOnCellsPlugin:
    """Sweeps Z around where the stage is, keeps every frame, and moves to the sharpest plane."""

    name = _NAME
    summary = (
        "Sweep Z around where the stage is, keep every frame as a Z-stack, and "
        "move to the sharpest plane (the autofocus is off for the sweep)."
    )
    Params = FocusOnCellsParams
    Result = CellFocusResult
    requires: tuple[type, ...] = (ZStage, Camera)
    provides = "cell-focus"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: FocusOnCellsParams,
        ctx: RunContext,
    ) -> CellFocusResult:
        """Check, sweep, move to the best plane (or back to the centre), and lock again.

        Raises:
            PluginRunError: A check refused it (nothing moved, nothing written):
                an autofocus on and not locked, no Z soft limits, a range above
                the Z jog limit, a plane outside the limits, a frame path too
                long for Windows. Or the sweep, or a move after it, failed: the
                frames taken are kept, Z is where it stopped and the autofocus
                is left off. Or a frame could not be written.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped;
                Z is where it is.
            StopRequested: The operator stopped it; Z is where it is.
        """
        z = microscope.require(ZStage)
        camera = microscope.require(Camera)
        xy = microscope.get(XYStage)
        af = microscope.get(Autofocus)
        turret = microscope.get(ObjectiveTurret)

        used, state, pfs_offset = _check_autofocus(af)
        bounds = check_sweep_bounds(
            _NAME,
            _NAME,
            microscope,
            z,
            "range_um",
            params.range_um,
            params.step_um,
        )
        centre = bounds.z_start_um
        planes = plan_planes(centre, params.range_um, params.step_um)
        n = len(planes)
        tolerance_um = LANDING_FRACTION * params.step_um

        token = new_token()
        folder = f"cell-focus/{token}"
        relatives = [f"{folder}/{k:03d}.tif" for k in range(n)]
        targets = _resolve(ctx, [*relatives, f"{folder}/{_BEST}"])
        objective = turret.current() if turret is not None else None
        here = xy.position_um() if xy is not None else None
        assumed: list[str] = []

        if ctx.dry_run:
            # The layer's dry run answers a move with its command: nothing
            # moves, no frame is taken and the autofocus is not touched
            # (`disengage` acts in a dry run too, m2 §3.2).
            walked: list[float] = []
            for plane in planes:
                ctx.raise_if_stopped()
                walked.append(move_checked(z, plane, tolerance_um))
            assumed.append(f"dry run: {n} planes walked, no frame taken")
            return CellFocusResult(
                folder=folder,
                objective=objective,
                x_um=here.x_um if here is not None else None,
                y_um=here.y_um if here is not None else None,
                autofocus=state,
                pfs_offset=pfs_offset,
                sweep=FocusSweep(
                    score=params.score,
                    centre_um=centre,
                    z_um=tuple(walked),
                    scores=(0.0,) * n,
                    outcome="flat",
                    best_z_um=None,
                ),
                files=(),
                best_file=None,
                z_um=z.position_um(),
                relocked=None,
                assumed=tuple(assumed),
            )

        written: list[str] = []
        plane_no = 0

        def raise_if_stopped() -> None:
            nonlocal plane_no
            plane_no += 1
            ctx.raise_if_stopped()

        def on_plane(k: int, readback: float, frame: npt.NDArray[Any]) -> None:
            _write(
                targets[k],
                relatives[k],
                frame,
                {
                    "z_um": readback,
                    "z_command_um": planes[k],
                    "score": score_frame(frame, params.score),
                    "score_name": params.score,
                },
            )
            written.append(relatives[k])
            ctx.progress(k + 1, n, f"plane {k + 1} of {n}")

        switched_off = False
        step = "the switch-off of the autofocus"
        error: str | None = None
        unwritten: str | None = None
        try:
            if used and af is not None:
                switch_off_for_sweep(af)
                switched_off = True
            step = ""
            readbacks, scores = sweep(
                z,
                camera,
                planes,
                params.step_um,
                params.score,
                raise_if_stopped=raise_if_stopped,
                on_plane=on_plane,
            )
            peak = find_peak(readbacks, scores)
            best_file: str | None = None
            if peak.z_um is not None:
                step = "the move to the best plane"
                landed = move_checked(z, peak.z_um, tolerance_um)
                step = "the snap at the best plane"
                frame = np.asarray(camera.snap())
                _write(
                    targets[n],
                    f"{folder}/{_BEST}",
                    frame,
                    {
                        "z_um": landed,
                        "z_command_um": peak.z_um,
                        "score": score_frame(frame, params.score),
                        "score_name": params.score,
                    },
                )
                best_file = f"{folder}/{_BEST}"
            else:
                step = "the move back to the centre"
                move_checked(z, centre, tolerance_um)
        except (MicroscopeHaltedError, MotionStoppedError, StopRequested):
            # A halt or a stop is not a failure (FM-64), and nothing is put
            # back after it (FM-88): what was left is said, then it passes.
            if switched_off:
                _LOG.warning(
                    "%s", _LEFT_OFF_WARNING.format(n=len(written), folder=folder)
                )
            raise
        except PluginRunError as exc:
            # A frame that could not be written (`_write`): its reason names
            # the frame and the fix, and the operator must also learn that the
            # autofocus is off and Z off its plane.
            if not switched_off:
                raise
            unwritten = exc.reason
        except _SWEEP_ERRORS as exc:
            error = _one_line(exc)
        # Raised outside the handlers (FM-98).
        if unwritten is not None:
            _LOG.warning("%s", _LEFT_OFF_WARNING.format(n=len(written), folder=folder))
            raise PluginRunError(
                _NAME,
                f"{unwritten}; the autofocus was left off; Z was left where it stopped",
            )
        if error is not None:
            where = step or f"plane {plane_no} of {n}"
            left_off = "; the autofocus was left off" if switched_off else ""
            if switched_off:
                _LOG.warning(
                    "%s", _LEFT_OFF_WARNING.format(n=len(written), folder=folder)
                )
            raise PluginRunError(
                _NAME,
                f"the focus sweep ended at {where} ({error}); the {len(written)} "
                f"frames taken are kept in {folder}/{left_off}; Z was left where "
                "it stopped",
            )

        relocked: bool | None = None
        at = "at the best plane" if peak.z_um is not None else "at the centre"
        if used and af is not None:
            if params.relock:
                relocked, why = _relock(af)
                if not relocked:
                    line = (
                        "the autofocus did not lock again: it is off"
                        if why is None
                        else f"the autofocus did not lock again ({why}): it is off"
                    )
                    _LOG.warning("%s", line)
                    assumed.append(line)
            else:
                assumed.append(f"the autofocus was left off, {at}")

        return CellFocusResult(
            folder=folder,
            objective=objective,
            x_um=here.x_um if here is not None else None,
            y_um=here.y_um if here is not None else None,
            autofocus=state,
            pfs_offset=pfs_offset,
            sweep=FocusSweep(
                score=params.score,
                centre_um=centre,
                z_um=readbacks,
                scores=scores,
                outcome=peak.outcome,
                best_z_um=peak.z_um,
            ),
            files=tuple(written),
            best_file=best_file,
            z_um=z.position_um(),
            relocked=relocked,
            assumed=tuple(assumed),
        )


def _check_autofocus(
    af: Autofocus | None,
) -> tuple[bool, _AutofocusState, float | None]:
    """Whether the autofocus is used, its state for the result, and its offset; refused when on and not locked.

    Used means on and locked: then the plane it holds is the sweep's centre,
    and its offset is the lock the sweep was centred on. One that is on and
    searching would move Z under the sweep.
    """
    if af is None:
        return False, "none", None
    if not af.is_engaged():
        return False, "off", None
    if not af.is_locked():
        raise PluginRunError(
            _NAME,
            "the autofocus is on and not locked: lock it on the sample, or switch "
            "it off to sweep around the Z where it is",
        )
    return True, "locked", af.offset_native()


def _relock(af: Autofocus) -> tuple[bool, str | None]:
    """``engage()``'s answer, and why it did not lock when it raised; a halt or a stop passes.

    The frames are written by now: a relock that fails is said, not a
    failure of the run (m3 §21.7).
    """
    why: str | None = None
    try:
        return af.engage(), None
    except (MicroscopeHaltedError, MotionStoppedError):
        raise
    except HardwareError as exc:
        why = _one_line(exc)
    return False, why


def _resolve(ctx: RunContext, relatives: list[str]) -> list[Path]:
    """Every frame path, resolved once before anything moves, with Windows' length checked on the longest.

    Raises:
        PluginRunError: A path leaves the run folder, the folder cannot be
            read, or the longest path is over ``MAX_PATH`` on Windows.
    """
    problem = ""
    resolved: list[Path] = []
    try:
        resolved = [resolve_in_run(ctx.run_dir, relative) for relative in relatives]
    except (ValueError, OSError) as exc:
        # `resolve_in_run` wraps the OS's refusal (a share that dropped,
        # FM-116) in a ValueError whose cause it is.
        unreadable = exc if isinstance(exc, OSError) else exc.__cause__
        if isinstance(unreadable, OSError):
            problem = (
                f"the run folder cannot be read ({unreadable.strerror or unreadable}): "
                f"check the disk or the network share, then run {_NAME} again"
            )
        else:
            problem = str(exc)
        del unreadable
    # Raised outside the handler (FM-98).
    if problem:
        raise PluginRunError(_NAME, problem)
    if ON_WINDOWS:
        length = max(len(str(temporary_path(path))) for path in resolved)
        if length >= MAX_PATH:
            raise PluginRunError(
                _NAME,
                f"a frame's path would reach {length} characters while it is "
                f"written, and Windows takes {MAX_PATH - 1} at most unless long "
                "paths are enabled (not assumed): move the run document to a "
                f"shorter folder, then run {_NAME} again",
            )
    return resolved


def _write(
    target: Path, relative: str, frame: npt.NDArray[Any], meta: dict[str, Any]
) -> None:
    """Write one frame whole (FM-38) as a TIFF in its own dtype, with ``meta`` as its JSON metadata.

    Raises:
        PluginRunError: The file could not be written (a full disk, a share
            that dropped, a file a viewer holds open); the frames before it
            are kept.
    """
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, frame, metadata=meta)
    failure: OSError | None = None
    try:
        write_atomically(target, buffer.getvalue())
    except OSError as exc:
        failure = exc
    # Raised outside the handler (FM-98).
    if failure is not None:
        cause = failure.cause if isinstance(failure, AtomicWriteError) else failure
        kept = failure.kept if isinstance(failure, AtomicWriteError) else None
        if kept is not None and isinstance(cause, PermissionError):
            reason = (
                f"the frame {relative} cannot be replaced: another program holds "
                "it open (a viewer, a backup or antivirus agent): close it, then "
                f"run {_NAME} again"
            )
        else:
            reason = (
                f"the frame {relative} could not be written "
                f"({failure.strerror or failure}): free some space if the disk is "
                f"full, or check the network share, then run {_NAME} again"
            )
        del failure, cause
        raise PluginRunError(_NAME, reason)


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()
