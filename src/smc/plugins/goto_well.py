"""``goto-well`` drives the stage to a well's centre or one of its walls, through the registered plate (m3 §5).

It sends one absolute XY move and nothing else. What it decides, because it is
the first stage that moves the stand on a position computed from a measurement:

* **A well the plate does not have is refused before any move**, with the name
  of the plate, so a typo is read at the console and not as a stage that went
  somewhere else.
* **The layer's guards are the guards.** An absolute move is not jog-guarded
  (crossing a plate is legitimate travel) but a target outside the soft limits
  is refused by the layer and cannot be forced; nothing is caught here, so a
  refusal, a halt and a device fault all reach the runner, which records the
  stage ``failed`` and raises again (FM-64).
* **The result is where the stage says it is.** ``x_um`` and ``y_um`` are the
  facade's readback after arrival, taken once and not read again (FM-85); the
  target is kept beside them, so a stage that stopped short of it shows.
* **In a dry run nothing moved, and the result says so.** The layer returns the
  target for a move it skipped; the readback is then the target, which the
  result's ``assumed`` states, so that nobody takes it for a measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from smc.hardware.capabilities import XYStage
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.plate_registration.algorithm import wall_position
from smc.plugins.slots import PlateResult, well_geometry

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["GotoWellParams", "GotoWellPlugin", "GotoWellResult"]

_NAME = "goto-well"


class GotoWellParams(PluginParams):
    """The well to go to, and where in it: its centre or the point on one of its walls."""

    well: str = Field(description="The well to go to, as useq names it (A1, H12).")
    side: Literal["centre", "left", "right", "top", "bottom"] = Field(
        "centre",
        description=(
            "Where in the well: its centre, or the point on a wall (left and right "
            "along the plate's X, top is +Y towards row A) from which to touch it."
        ),
    )


class GotoWellResult(PluginResult):
    """Where the stage was sent and where it arrived, in µm."""

    well: str
    side: str
    target_x_um: float
    target_y_um: float
    x_um: float
    y_um: float


class GotoWellPlugin:
    """Moves the XY stage to a well of the registered plate."""

    name = _NAME
    summary = "Drive the stage to a well's centre or wall through the registered plate."
    Params = GotoWellParams
    Result = GotoWellResult
    requires: tuple[type, ...] = (XYStage,)
    provides = "goto-well"
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: GotoWellParams,
        ctx: RunContext,
    ) -> GotoWellResult:
        """Send one absolute move to the well's centre or wall and report the readback.

        Raises:
            PluginRunError: The well is not on the plate; nothing was moved.
            StopRequested: The operator stopped it before the move was sent.
            SafetyRefusedError: The target is outside the soft limits; the stage
                did not move. The layer's errors are not caught (FM-64).
        """
        plate = ctx.upstream("plate", PlateResult)

        problem = ""
        well = ""
        target = (0.0, 0.0)
        try:
            well = well_geometry(plate.to_plan(), params.well).well
            target = (
                plate.well_center_um(well)
                if params.side == "centre"
                else wall_position(plate, well, params.side)
            )
        except ValueError as exc:
            problem = f"{exc}; give a well of the {plate.plate} plate"
        if problem:
            # Raised outside the handler: nothing here keeps the algorithm's error (FM-98).
            raise PluginRunError(_NAME, problem)

        ctx.raise_if_stopped()
        landed = microscope.require(XYStage).move_to_um(*target)

        assumed = [f"plate: {line}" for line in plate.assumed]
        if ctx.dry_run:
            assumed.append(
                "dry run: the stage did not move; x_um and y_um are the target"
            )
        return GotoWellResult(
            well=well,
            side=params.side,
            target_x_um=target[0],
            target_y_um=target[1],
            x_um=landed.x_um,
            y_um=landed.y_um,
            assumed=tuple(assumed),
        )
