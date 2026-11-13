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
* **The target is in the frame of the objective in use** (m3 §18.3). A plate
  registered with the 10x is not at the same stage positions under the 40x; the
  profile's offset between them moves the plate before the target is computed.
  An offset that is not measured, or a plate that does not record its objective,
  moves to the registration's own positions and says so in ``assumed`` and in a
  WARNING: unknown is not zero. An empty turret position is no objective and is
  refused before any move.
* **Before registration, the holder's estimate** (m3 §5.5). With no usable
  ``plate`` slot and a ``plate`` parameter, the target is the well's centre where
  the profile's ``[holder]`` puts that plate (``holder_estimate``, the map's
  *estimated* pose), so the operator can reach a well to centre by eye before
  any registration. Only the centre: the estimate can be a few hundred µm off,
  so a wall is refused. The record says so first (``from_holder``, its first
  ``assumed`` line), applies no objective offset, and fills no slot that any
  tool reads. A usable slot always wins: the parameters then only check that it
  is the same plate.
* **An engaged autofocus is switched off before the move** (m3 §5.6 D1). PFS
  locked in one well follows its reference over the wall into the next, whose
  plane lay at least 104 µm away at the 40x (F11), with nothing holding the
  microscope (m2 §12.2 D2, FM-223). So before every move, the same well's too,
  PFS on (or searching) is switched off and read back, and the record says so
  first; the operator engages it again in the well. A switch-off that fails or
  does not hold fails the run before anything moves (FM-19). Not in a dry run:
  nothing moves, and a switch-off would act on the stand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import Field, field_validator

from smc.hardware.capabilities import Autofocus, ObjectiveTurret, XYStage
from smc.hardware.errors import HardwareError, MicroscopeHaltedError, MotionStoppedError
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError, UpstreamError
from smc.plugins.objectives import log_frame, objective_for, plate_for_objective
from smc.plugins.plate_registration.algorithm import wall_position
from smc.plugins.slots import (
    PlateResult,
    holder_estimate,
    refuse_image_side,
    well_geometry,
)
from smc.plugins.slots.plate import PlateOrientation

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["GotoWellParams", "GotoWellPlugin", "GotoWellResult"]

_NAME = "goto-well"

#: The last ``assumed`` line of a dry run, on both paths.
_DRY_LINE = "dry run: the stage did not move; x_um and y_um are the target"

#: How the estimate's ``assumed`` line names an orientation (m3 §5.5 D4).
_ORIENTATION_WORDS = {"not-turned": "not turned", "turned-180": "turned 180°"}


class GotoWellParams(PluginParams):
    """The well to go to, and where in it: its centre or the point on one of its walls.

    A wall is named by a stage direction, never by what an image shows (m3 §19).
    """

    well: str = Field(description="The well to go to, as useq names it (A1, H12).")
    side: Literal["centre", "+X", "-X", "+Y", "-Y"] = Field(
        "centre",
        description=(
            "Where in the well: its centre, or the point on a wall from which to "
            "touch it, named by the stage direction that reaches it from the centre "
            "(-X and +X along the plate's X, -Y and +Y along its Y; +Y is towards "
            "row A on a plate that is not turned). A side as seen on an image "
            "(left, right, top, bottom) is "
            "refused: the plate page maps those with the camera calibration."
        ),
    )
    plate: str | None = Field(
        None,
        description=(
            "Used only when the run document has no usable plate: the useq plate "
            "name (96-well, greiner-24-well) of the plate in the holder, to drive "
            "to the well's centre where the profile's [holder] plate_centre_xy_um "
            "puts it, an estimate and not a registration. With a usable plate, it "
            "must name the registered plate."
        ),
    )
    orientation: PlateOrientation = Field(
        "not-turned",
        description=(
            "Used only when the run document has no usable plate: which way the "
            "plate lies in its holder (not-turned: A1 at the -X +Y corner; "
            "turned-180). A registered plate's own orientation is used otherwise."
        ),
    )

    @field_validator("side", mode="before")
    @classmethod
    def _side_is_not_an_image_side(cls, value: Any) -> Any:
        return refuse_image_side(value)


class GotoWellResult(PluginResult):
    """Where the stage was sent and where it arrived, in µm.

    ``side`` echoes the parameter: ``centre`` or a stage direction (``-X``, ``+X``,
    ``-Y``, ``+Y``). It stays a ``str`` so that a run document written when the
    words were ``left`` and ``right`` still loads; nothing writes those now.
    ``objective`` is the one in use, ``plate_objective`` the one the plate was
    registered with, and ``shift_um`` what the target was moved by between the
    two: ``(0.0, 0.0)`` for the same objective, ``None`` when no offset could be
    applied (the target is then the registration's own, and ``assumed`` says so).
    All three default to unknown, so that older run documents load.
    ``from_holder`` is true when the target is the holder's estimate and not a
    registered plate (m3 §5.5 D4); an older record loads as registered.
    ``autofocus_switched_off`` is true when an engaged autofocus was switched off
    before the move, so that no lock crossed a well's wall (m3 §5.6 D1); the
    operator engages it again in the well. An older record loads as ``False``.
    """

    well: str
    side: str
    target_x_um: float
    target_y_um: float
    x_um: float
    y_um: float
    objective: str | None = None
    plate_objective: str | None = None
    shift_um: tuple[float, float] | None = None
    from_holder: bool = False
    autofocus_switched_off: bool = False


class GotoWellPlugin:
    """Moves the XY stage to a well of the registered plate, or of the holder's estimate."""

    name = _NAME
    summary = (
        "Drive the stage to a well's centre or wall through the registered plate, "
        "or to a well's centre estimated from the stand's holder."
    )
    Params = GotoWellParams
    Result = GotoWellResult
    requires: tuple[type, ...] = (XYStage,)
    provides = "goto-well"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ("plate",)

    def run(
        self,
        microscope: Microscope,
        params: GotoWellParams,
        ctx: RunContext,
    ) -> GotoWellResult:
        """Send one absolute move to the well's centre or wall and report the readback.

        A usable ``plate`` slot is the target's frame; without one, the holder's
        estimate of the ``plate`` parameter (m3 §5.5 D3). Every refusal comes before
        the one move.

        Raises:
            UpstreamError: No usable plate, and no ``plate`` parameter or no
                ``[holder] plate_centre_xy_um`` to estimate one from; or a usable
                slot whose result the model refuses. Nothing was moved.
            PluginRunError: The well is not on the plate, the ``plate`` parameter
                is not the registered plate or not a plate ``useq`` knows, a wall
                was asked of an estimate, the turret is at an empty position, or
                an engaged autofocus could not be switched off; nothing was moved.
            StopRequested: The operator stopped it before the move was sent.
            SafetyRefusedError: The target is outside the soft limits; the stage
                did not move. The layer's errors are not caught (FM-64), and a
                turret that cannot be read raises its own (FM-19).
        """
        plate = ctx.upstream_or_none("plate", PlateResult)
        if plate is None:
            estimate = _estimate_target(microscope, params, ctx)
            ctx.logger.warning(
                "%s: driving to %s of the %s estimated from the stand's holder, not "
                "registered",
                _NAME,
                estimate.well,
                params.plate,
            )
            ctx.raise_if_stopped()
            switched_off = _switch_off_before_move(microscope, ctx, estimate.well)
            arrived = microscope.require(XYStage).move_to_um(*estimate.target)
            lines = [_switched_off_line(estimate.well)] if switched_off else []
            lines.extend(estimate.assumed)
            if ctx.dry_run:
                lines.append(_DRY_LINE)
            return GotoWellResult(
                well=estimate.well,
                side=params.side,
                target_x_um=estimate.target[0],
                target_y_um=estimate.target[1],
                x_um=arrived.x_um,
                y_um=arrived.y_um,
                objective=estimate.objective,
                plate_objective=None,
                shift_um=None,
                from_holder=True,
                autofocus_switched_off=switched_off,
                assumed=tuple(lines),
            )
        if params.plate is not None and params.plate != plate.plate:
            raise PluginRunError(
                _NAME,
                f"the registered plate is a {plate.plate}, not a {params.plate}: "
                f"leave plate out, or register the {params.plate}",
            )
        turret = microscope.get(ObjectiveTurret)
        # Before anything moves, and before the well is looked up: the target is
        # worked out on the plate in the frame of the objective in use.
        objective, _ = objective_for(_NAME, None, turret)
        frame = plate_for_objective(plate, objective, turret)
        log_frame(ctx.logger, plate, frame, turret)
        framed = frame.plate

        problem = ""
        well = ""
        target = (0.0, 0.0)
        try:
            well = well_geometry(framed.to_plan(), params.well).well
            target = (
                framed.well_center_um(well)
                if params.side == "centre"
                else wall_position(framed, well, params.side)
            )
        except ValueError as exc:
            problem = f"{exc}; give a well of the {plate.plate} plate"
        if problem:
            # Raised outside the handler: nothing here keeps the algorithm's error (FM-98).
            raise PluginRunError(_NAME, problem)

        ctx.raise_if_stopped()
        switched_off = _switch_off_before_move(microscope, ctx, well)
        landed = microscope.require(XYStage).move_to_um(*target)

        assumed = [_switched_off_line(well)] if switched_off else []
        assumed.extend(f"plate: {line}" for line in plate.assumed)
        assumed.extend(frame.assumed)
        if ctx.dry_run:
            assumed.append(_DRY_LINE)
        return GotoWellResult(
            well=well,
            side=params.side,
            target_x_um=target[0],
            target_y_um=target[1],
            x_um=landed.x_um,
            y_um=landed.y_um,
            objective=objective,
            plate_objective=plate.objective,
            shift_um=frame.shift_um,
            autofocus_switched_off=switched_off,
            assumed=tuple(assumed),
        )


def _switched_off_line(well: str) -> str:
    """The first ``assumed`` line of a run that switched the autofocus off (m3 §5.6 D1).

    First, so that the page's strip shows it (``_assumed_tail``).
    """
    return (
        "the autofocus was switched off before the move, so that no lock crossed a "
        f"well's wall: engage it again in {well}"
    )


def _switch_off_before_move(microscope: Microscope, ctx: RunContext, well: str) -> bool:
    """Switch an engaged autofocus off and read it back; whether this switched it off (m3 §5.6 D1).

    Called just before the one move, after the stop check, with nothing between:
    a lock carried over a well's wall drives Z to the plane its offset names in
    the next well, unguarded (m2 §12.2 D2). ``is_engaged()`` decides, not
    ``is_locked()``: PFS searching drives Z the moment it finds its reference.
    No call at all without an autofocus or in a dry run, where nothing moves
    and a switch-off would act on the stand (m2 §3.2). One that is off is left
    off, and nothing is said.

    Raises:
        PluginRunError: A read or the switch-off raised, or it still reads on:
            PFS may be on, so nothing may move (FM-19). Raised outside the
            handler, with the error's text only (FM-98).
        MicroscopeHaltedError, MotionStoppedError: Passed first (FM-64).
    """
    af = microscope.get(Autofocus)
    if af is None or ctx.dry_run:
        return False
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
    if cause is not None:
        raise PluginRunError(
            _NAME,
            f"the autofocus could not be switched off before the move to {well} "
            f"({cause}): the stage did not move, and PFS may still be on: switch it "
            "off at the stand, then go to the well again",
        )
    if still_on:
        raise PluginRunError(
            _NAME,
            f"the autofocus still reads on after it was switched off before the move "
            f"to {well}: the stage did not move: switch it off at the stand, then go "
            "to the well again",
        )
    ctx.logger.warning(
        "%s: the autofocus was switched off before the move to %s, so that no lock "
        "crosses a well's wall (m2 §12.2 D2): engage it again there",
        _NAME,
        well,
    )
    return True


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``: a device's error, for the run's text."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()


class _Estimate(NamedTuple):
    """Where the holder's estimate puts a well, and what the record says about it."""

    well: str
    target: tuple[float, float]
    objective: str | None
    assumed: tuple[str, ...]


def _estimate_target(
    microscope: Microscope, params: GotoWellParams, ctx: RunContext
) -> _Estimate:
    """The well's centre where the profile's ``[holder]`` puts ``params.plate`` (m3 §5.5 D3).

    In D3's order, and nothing moves here: no ``plate`` parameter, no holder, the
    turret (an empty position is refused as on the registered path), a wall, then
    the plate and the well. Each refusal names what is missing and the fix; a
    ``useq`` error is taken as text in the handler and raised outside it (FM-98).

    Raises:
        UpstreamError: No ``plate`` parameter, or no ``[holder]
            plate_centre_xy_um``: the text starts with why the slot was not read.
        PluginRunError: A wall side, a plate ``useq`` does not know, a well not
            on the plate, or an empty turret position.
    """
    why = ctx.why_not_read("plate")
    if params.plate is None:
        raise UpstreamError(
            _NAME,
            "plate",
            f"{why}; or give plate=<a useq plate name> to drive to the estimate of "
            f"the profile's [holder]",
        )
    centre = microscope.profile.holder.plate_centre_xy_um
    if centre is None:
        raise UpstreamError(
            _NAME,
            "plate",
            f"{why}; and this stand's profile has no [holder] plate_centre_xy_um to "
            f"estimate the plate from",
        )
    objective, _ = objective_for(_NAME, None, microscope.get(ObjectiveTurret))
    if params.side != "centre":
        raise PluginRunError(
            _NAME,
            "a wall is reached through a registered plate: the holder's estimate can "
            "be a few hundred µm off; go to the centre, and register the plate",
        )
    problem = ""
    geometry = None
    try:
        plan = holder_estimate(params.plate, centre, params.orientation)
    except ValueError as exc:
        problem = f"{exc}; give a useq plate name (96-well, greiner-24-well)"
    else:
        try:
            geometry = well_geometry(plan, params.well)
        except ValueError as exc:
            problem = f"{exc}; give a well of the {params.plate} plate"
    if geometry is None:
        # Raised outside the handlers: nothing here keeps useq's error (FM-98).
        raise PluginRunError(_NAME, problem)
    assumed = (
        f"target estimated from the stand's holder, not registered: the "
        f"{params.plate} centred at stage ({centre[0]:.0f}, {centre[1]:.0f}) µm, "
        f"{_ORIENTATION_WORDS[params.orientation]}",
        f"plate not read: {why}",
        "no objective offset applied: the holder's estimate names no objective",
    )
    return _Estimate(geometry.well, geometry.center_um, objective, assumed)
