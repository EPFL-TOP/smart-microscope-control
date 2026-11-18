"""``plate-check`` checks that a registered plate still lies where ``plate-walls`` registered it (m3 §27.6).

At Ti2-E session 5 (#550, F15) the owner lifted the registered plate and put it
back; it turned by -0.056° and slid, and nothing in smc noticed. Experimenters
do this without noticing. This tool measures again, with ``plate-walls``' own
loop, the centres of two wells the registration measured (the pair farthest
apart by default), under the objective in place with its own calibration, and
compares them with the registered centres moved into that objective's frame
(§18). Beyond ``max_shift_um`` it refuses; otherwise its slot records when the
registration was last confirmed, before a drive, a route or an NIS export. The
scan makes the same check before its first field by itself (§27.5).

The decisions that keep the check from lying:

* **Everything that can be checked is checked before anything moves** (FM-30):
  a dry run, the objective, the offset between the plate's objective and the
  one in place, the calibration (of this objective and frame), the wells, the
  walk and the soft limits. A refusal moves nothing.
* **An unknown offset is a refusal, not a zero** (§27.2 D2): under another
  objective the offset reads as a move (39.4 µm on the demo for a plate that
  did not move, §27.1).
* **An engaged autofocus is switched off once, before anything moves** (m3 §5.7
  D1), with ``plate-walls``' own function and texts: PFS locked in one well
  follows its reference over the wall into the next (FM-223). Nothing engages
  it again; a refusal after the switch-off says so (D3).
* **Nothing moves to put something back** (FM-88): the run ends with one move
  to the last well's fitted centre, as ``plate-walls`` does, and on any
  exception a WARNING says where the stage was left.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from smc.hardware.capabilities import Autofocus, Camera, ObjectiveTurret, XYStage
from smc.hardware.profile import MAX_XY_OFFSET_UM
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.camera_calibration.plugin import calibration_from
from smc.plugins.errors import PluginRunError
from smc.plugins.objectives import objective_for, plate_for_objective
from smc.plugins.plate_check.algorithm import DEFAULT_MAX_SHIFT_UM
from smc.plugins.plate_check.loop import (
    checked_line,
    objective_text,
    prepare_check,
    run_check,
    unsaved_line,
)
from smc.plugins.plate_walls.loop import (
    AUTOFOCUS_OFF_LINE,
    SwitchOff,
    Where,
    check_calibration,
    move,
    run_saying_autofocus_left_off,
    switch_off_autofocus,
    warn_left,
)
from smc.plugins.slots import CameraCalibration, PlateCheckResult, PlateResult
from smc.run.files import new_token

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["AGAIN", "PlateCheckParams", "PlateCheckPlugin"]

_NAME = "plate-check"
#: What the operator does about a plate that moved, at the end of the refusal.
AGAIN = (
    "register it again (plate-walls) before a scan, a drive to a well, a route or "
    "an NIS export"
)


class PlateCheckParams(PluginParams):
    """The wells to measure again, the bound, and the calibration of the objective in place."""

    wells: tuple[str, ...] = Field(
        (),
        max_length=6,
        description=(
            "Wells the registration measured, two to six; default the pair "
            "farthest apart."
        ),
    )
    max_shift_um: float = Field(
        DEFAULT_MAX_SHIFT_UM,
        gt=0.0,
        le=500.0,
        description="How far a checked well may lie from its registered centre, in µm.",
    )
    calibration: Path | None = Field(
        None,
        description=(
            "A run document holding a camera-calibration made with the objective "
            "in place, when it is not the plate's own (relative to the working "
            "directory, as from_run_doc); default this document's."
        ),
    )
    search_um: float | None = Field(
        None,
        gt=0.0,
        le=MAX_XY_OFFSET_UM,
        description=(
            "How far each wall may lie from where the registered centre puts it, "
            "in µm; default min(1000, the well's radius); never above the radius."
        ),
    )
    settle_ms: float = Field(
        0.0,
        ge=0.0,
        le=10_000.0,
        description="Wait this long after each move before the frame is taken.",
    )
    save_frames: bool = Field(
        True,
        description=(
            "Save the frame centred on each wall as plate-check/<token>/"
            "<well>_<side>.tif beside the run document."
        ),
    )


class PlateCheckPlugin:
    """Measures two registered wells again and refuses a plate that moved since plate-walls registered it."""

    name = _NAME
    summary = (
        "Check that the plate still lies where plate-walls registered it: "
        "re-measure two registered wells and refuse a plate that moved."
    )
    Params = PlateCheckParams
    Result = PlateCheckResult
    requires: tuple[type, ...] = (XYStage, Camera)
    provides = "plate-check"
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ("camera-calibration",)

    def run(
        self,
        microscope: Microscope,
        params: PlateCheckParams,
        ctx: RunContext,
    ) -> PlateCheckResult:
        """Check, measure the wells again, and refuse a plate that moved.

        Raises:
            PluginRunError: A dry run, an empty turret position, an offset
                between the plate's objective and the one in place that is not
                measured, no calibration of the objective in place or one of
                another objective or frame, a plate ``plate-walls`` did not
                register, a well it did not measure, a walk that is too long or
                leaves the soft limits, an engaged autofocus that could not be
                switched off (nothing moved); or, the stage then where it was
                left: a wall that is not found, a well whose walls do not fit,
                a well that moved more than ``max_shift_um``.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped;
                the stage is where it is.
            StopRequested: The operator stopped it; the stage is where it is.

        A ``PluginRunError`` raised after the autofocus was switched off ends
        with ``AUTOFOCUS_LEFT_OFF`` (m3 §5.7 D3).
        """
        return run_saying_autofocus_left_off(
            lambda off: self._run(microscope, params, ctx, off)
        )

    def _run(
        self,
        microscope: Microscope,
        params: PlateCheckParams,
        ctx: RunContext,
        off: SwitchOff,
    ) -> PlateCheckResult:
        """``run``'s body; ``off`` records whether the switch-off was made, for a refusal after it."""
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        turret = microscope.get(ObjectiveTurret)
        if ctx.dry_run:
            raise PluginRunError(
                _NAME, "a dry run cannot image the walls; run it without --dry-run"
            )
        plate = ctx.upstream("plate", PlateResult)
        objective, _ = objective_for(_NAME, None, turret)
        frame = plate_for_objective(plate, objective, turret)
        if frame.shift_um is None:
            raise PluginRunError(_NAME, _unknown_frame(plate.objective, objective))
        calibration = self._calibration(microscope, params, ctx, objective)
        height, width = camera.image_shape()
        shape = (int(height), int(width))
        check_calibration(_NAME, calibration, objective, turret is not None, shape)

        token = new_token()
        setup = prepare_check(
            _NAME,
            xy,
            plate,
            params.wells,
            objective=objective,
            shift_um=frame.shift_um,
            matrix=calibration.matrix_um_per_px,
            pixel_size_um=calibration.pixel_size_um,
            shape=shape,
            max_shift_um=params.max_shift_um,
            search_um=params.search_um,
            settle_s=params.settle_ms / 1000.0,
            folder=ctx.run_dir / "plate-check" / token if params.save_frames else None,
            folder_text=f"plate-check/{token}" if params.save_frames else None,
            again=AGAIN,
        )
        assumed = list(setup.assumed)

        # After every refusal that can come before the first move, and outside
        # any `try`: nothing has moved. A stop asked now keeps the lock (§5.7 D1).
        ctx.raise_if_stopped()
        if switch_off_autofocus(_NAME, microscope.get(Autofocus), ctx):
            off.switched = True
            assumed.insert(0, AUTOFOCUS_OFF_LINE)

        sides_done = 0
        total_sides = 4 * len(setup.refs)

        def on_side(text: str) -> None:
            nonlocal sides_done
            sides_done += 1
            ctx.progress(sides_done, total_sides, text)

        # Where the run started, for the warning of the last move: run_check says
        # its own. A stage still travelling from an earlier move reads a fraction.
        xy.wait()
        start = xy.position_um()
        check = run_check(setup, xy, camera, ctx, on_side)

        # The last well, where the check found it: a well centred under the
        # objective in place, which Live shows (as plate-walls ends, §17.4).
        last = check.wells[-1].fit
        where = Where()
        where.text = f"{last.well}'s fitted centre"
        try:
            ctx.raise_if_stopped()
            move(_NAME, xy, *last.center_xy_um, where.text)
        except BaseException:
            warn_left(_NAME, ctx, xy, start, where)
            raise

        unsaved = unsaved_line(check)
        if unsaved is not None:
            assumed.append(unsaved)
        assumed.append(checked_line(check, before_the_first_field=False))
        return PlateCheckResult(check=check, assumed=tuple(assumed))

    @staticmethod
    def _calibration(
        microscope: Microscope,
        params: PlateCheckParams,
        ctx: RunContext,
        objective: str | None,
    ) -> CameraCalibration:
        """The calibration given, else this document's; none is a refusal that says which to give."""
        if params.calibration is not None:
            calibration, _ = calibration_from(
                _NAME,
                params.calibration,
                profile=microscope.profile.microscope.name,
                dry_run=ctx.dry_run,
            )
            return calibration
        own = ctx.upstream_or_none("camera-calibration", CameraCalibration)
        if own is None:
            raise PluginRunError(
                _NAME,
                f"no camera calibration of {objective_text(objective)}: give "
                "calibration=<a run document whose camera-calibration was made with "
                "it in place> (m5 §18.6)",
            )
        return own


def _unknown_frame(plate_objective: str | None, objective: str | None) -> str:
    """Why the plate cannot be put in the frame of the objective in place, and what to do."""
    if plate_objective is not None and objective is not None:
        return (
            f"the offset between {plate_objective} and {objective} is not measured "
            f"(the profile's [objectives]): check the plate with {plate_objective} "
            "in place, or measure the offsets (objective-offsets)"
        )
    if plate_objective is None:
        return (
            "the plate does not record the objective it was registered with, so "
            f"its frame under {objective_text(objective)} is not known: register it "
            "again with plate-walls"
        )
    return (
        f"the objective in use is unknown, and the plate was registered with "
        f"{plate_objective}: check the plate on the stand it was registered on"
    )
