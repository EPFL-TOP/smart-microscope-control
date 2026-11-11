"""``plate-walls`` registers the plate from images of the walls of 2-3 wells the operator centred by eye (m3 §17).

The pure part (the wall finder, the circle fit) is ``algorithm.py``. This is the
hardware loop around it, which moves the XY stage around positions the operator
recorded, and the decisions in it that keep a registration from lying:

* **Everything that can be checked is checked before anything moves** (FM-30):
  a dry run, the wells and the plate, the calibration (it must be of this
  objective and this frame, and its matrix must invert), the frame, the walk's
  size and the stage's soft limits. A refusal moves nothing.
* **The camera's turn is measured, never assumed.** The Ti2-E's 10x camera is
  turned 180 degrees, so the matrix of an upright camera would invert every wall
  offset. A calibration of another objective or frame is a refusal, and so is
  none.
* **A frame is placed where the stage says it was**, not where it was sent
  (FM-142), and the wall of each side is measured in a frame centred on it: every
  side is framed the same way, so the threshold's bias (the meniscus) is the same
  on opposite walls and cancels in the centre.
* **A fit that cannot be trusted is a refusal, not a result** and it names the
  well: a wall that is not found, a well of another size, a circle that does not
  hold, a well that the others do not put where it is. A refusal leaves the
  stage where it is.
* **Nothing moves to put something back** (FM-88): no return and no ``finally``
  that moves. On any exception a WARNING says where the stage was left. No Z,
  turret, autofocus or light is touched.
* **A halt is not a refusal** (FM-64): ``SafetyRefusedError`` from a move is a
  refusal that names the well and the side; ``MicroscopeHaltedError`` and
  ``MotionStoppedError`` are not ``SafetyRefusedError``s and pass unchanged.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import useq
from pydantic import Field, field_validator

from smc.hardware.capabilities import Camera, ObjectiveTurret, XYStage
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.plate_registration.algorithm import (
    MAX_ROTATION_DEG,
    calibrate,
    nominal_offsets,
)
from smc.plugins.plate_walls.algorithm import (
    DEFAULT_SEARCH_FRACTION,
    SIDES,
    STEP_FRACTION,
    walk_offsets_um,
)
from smc.plugins.plate_walls.loop import (
    WallWalk,
    Where,
    check_calibration,
    check_frame,
    check_soft_limits,
    check_walk_size,
    checked_search,
    measure_well,
    move,
    warn_left,
)
from smc.plugins.slots import (
    CameraCalibration,
    PlateResult,
    WallImages,
    WellFit,
    WellRef,
    normalise_well,
)
from smc.plugins.slots.plate import PlateOrientation
from smc.run.files import new_token

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["PlateWallsParams", "PlateWallsPlugin"]

_NAME = "plate-walls"
_MIN_WELLS = 2
_MAX_WELLS = 6
#: No stage travels a metre: a recorded centre this far out is a unit slip (FM-32).
_MAX_CENTRE_UM = 1_000_000.0
#: The end of the "no wall found" refusal (m3 §20.2): what to check when a
#: registration cannot find a wall. ``{well}`` is the well's name.
_NO_WALL_HINT = (
    "check that the light is on (the inside of a well is brighter than the "
    "plastic in brightfield) and that the stage was in {well} when it was "
    "recorded, then record it again"
)
_TWO_WELLS_LINE = (
    "two wells: the residual checks their distance only; a third well checks the fit"
)


class PlateWallsParams(PluginParams):
    """The plate, the wells centred by eye, and how far to look for their walls."""

    plate: str = Field(
        "96-well",
        description=(
            "A useq plate name (24-well, 96-well); names are exact. The wells "
            "must be round: the square ones are registered with plate-registration."
        ),
    )
    refs: tuple[WellRef, ...] = Field(
        (),
        description=(
            "Two to six wells, each with the stage position where it was centred "
            'by eye with the joystick, JSON: refs=[{"well": "B3", "x_um": '
            '10200, "y_um": -9700}], the position from smc stage get. Far apart '
            "wells pin the rotation down."
        ),
    )
    search_um: float | None = Field(
        None,
        gt=0.0,
        le=20_000.0,
        description=(
            "How far, on each axis, the wall may lie from where the recorded "
            "centre puts it, in µm. Default: half the well's radius. It may not "
            "exceed the radius."
        ),
    )
    max_residual_um: float = Field(
        100.0,
        gt=0.0,
        le=1000.0,
        description=(
            "The largest distance, in µm, between a well's measured centre and "
            "where the plate's fit puts it; a well further out is refused."
        ),
    )
    max_rotation_deg: float = Field(
        MAX_ROTATION_DEG,
        gt=0.0,
        le=45.0,
        description=(
            "The largest tilt of the plate from its declared orientation that the "
            "fit accepts; a tilt past it is read as a misnamed well. Raise it "
            "only for a plate that is really tilted in its holder."
        ),
    )
    orientation: PlateOrientation = Field(
        "not-turned",
        description=(
            "Which way the plate lies in its holder: not-turned (A1 at the "
            "stage's -X +Y corner) or turned-180 (A1 at the +X -Y corner)."
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
            "Save the frame centred on each wall as plate-walls/<token>/"
            "<well>_<side>.tif beside the run document."
        ),
    )

    @field_validator("plate")
    @classmethod
    def _plate_is_one_useq_knows(cls, value: str) -> str:
        """Refuse a plate name ``useq`` has no definition for, before the stand opens."""
        useq.WellPlate.from_str(value)
        return value


class PlateWallsPlugin:
    """Registers a plate from the walls of 2-3 wells centred by eye: A1's centre and the plate's rotation."""

    name = _NAME
    summary = (
        "Register the plate from images of the walls of 2-3 wells you centred by "
        "eye: fits each well's centre and size, then the plate."
    )
    Params = PlateWallsParams
    Result = PlateResult
    requires: tuple[type, ...] = (XYStage, Camera)
    provides = "plate"
    reads: tuple[str, ...] = ("camera-calibration",)
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: PlateWallsParams,
        ctx: RunContext,
    ) -> PlateResult:
        """Check, look at each well's four walls, fit the wells and the plate; refuse what cannot be trusted.

        Raises:
            PluginRunError: A dry run, square wells, references that are fewer
                than two, more than six, off the plate, twice or beyond the
                stage's travel, a calibration that is not of this objective and
                frame, a frame that is too small or holds a whole well, a walk
                that is too long or leaves the soft limits (nothing moved); or,
                the stage then where it was left: a wall that is not found or
                was lost, a move the layer refused, a frame of another shape, a
                well of another size or whose walls do not make a circle, a
                well that the others do not put where it is, a rotation beyond
                ``max_rotation_deg``.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped;
                the stage is where it is.
            StopRequested: The operator stopped it; the stage is where it is.
        """
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        turret = microscope.get(ObjectiveTurret)
        if ctx.dry_run:
            raise PluginRunError(
                _NAME, "a dry run cannot image the walls; run it without --dry-run"
            )
        plate = useq.WellPlate.from_str(params.plate)
        if not plate.circular_wells:
            raise PluginRunError(
                _NAME,
                f"plate-walls fits round wells, and the {params.plate} plate's are "
                "square: register it with plate-registration",
            )
        refs = _checked_refs(params)

        objective = turret.current() if turret is not None else None
        calibration = ctx.upstream("camera-calibration", CameraCalibration)
        height, width = camera.image_shape()
        shape = (int(height), int(width))
        check_calibration(_NAME, calibration, objective, turret is not None, shape)
        matrix = calibration.matrix_um_per_px
        diameter_um = float(plate.well_size[0]) * 1000.0
        check_frame(_NAME, matrix, shape, diameter_um)

        radius_um = diameter_um / 2.0
        search_um = checked_search(
            _NAME,
            params.search_um,
            radius_um,
            default_um=DEFAULT_SEARCH_FRACTION * radius_um,
        )
        step_um = STEP_FRACTION * min(shape) * calibration.pixel_size_um
        check_walk_size(_NAME, len(refs), search_um, step_um)
        offsets = walk_offsets_um(radius_um, search_um, step_um)
        assumed = check_soft_limits(_NAME, xy, refs, offsets)

        token = new_token()
        walk = WallWalk(
            stage=_NAME,
            plate=params.plate,
            radius_um=radius_um,
            search_um=search_um,
            step_um=step_um,
            matrix=matrix,
            shape=shape,
            pixel_size_um=calibration.pixel_size_um,
            settle_s=params.settle_ms / 1000.0,
            folder=ctx.run_dir / "plate-walls" / token if params.save_frames else None,
            folder_text=f"plate-walls/{token}" if params.save_frames else None,
            no_wall_hint=_NO_WALL_HINT,
        )

        # A stage still travelling from an earlier move reads a fraction of it.
        xy.wait()
        start = xy.position_um()
        started = time.monotonic()
        where = Where()
        save_errors: list[str] = []
        fits: list[WellFit] = []
        frames = 0
        sides_done = 0
        total_sides = len(refs) * len(SIDES)

        def on_side(text: str) -> None:
            nonlocal sides_done
            sides_done += 1
            ctx.progress(sides_done, total_sides, text)

        try:
            for ref in refs:
                fit, taken = measure_well(
                    xy, camera, ctx, walk, ref, offsets, where, save_errors, on_side
                )
                fits.append(fit)
                frames += taken
            measured = tuple(
                WellRef(well=f.well, x_um=f.center_xy_um[0], y_um=f.center_xy_um[1])
                for f in fits
            )
            fitted, residual_um = _fit_plate(params, measured)
            ctx.raise_if_stopped()
            where.text = f"{fits[-1].well}'s fitted centre"
            move(_NAME, xy, *fits[-1].center_xy_um, where.text)
        except BaseException:
            # A stop, a halt, a fault or a refusal: the stage stays where it is, the
            # exception goes on (FM-88, FM-64), and nothing is caught.
            warn_left(_NAME, ctx, xy, start, where)
            raise

        if len(refs) == _MIN_WELLS:
            assumed.append(_TWO_WELLS_LINE)
        if save_errors:
            assumed.append(
                f"{len(save_errors)} frames could not be saved: {save_errors[0]}"
            )
        return PlateResult(
            plate=params.plate,
            a1_center_xy_um=fitted.a1_center_xy_um,
            rotation_deg=fitted.rotation_deg,
            orientation=fitted.orientation,
            method="wall-images",
            residual_um=residual_um,
            scale_error=fitted.scale_error,
            n_refs=len(measured),
            refs=measured,
            fit=None,
            # The plate's own objective (m3 §18.1): the one the frames were taken
            # with, which `check_calibration` has already refused to be an empty position.
            objective=objective,
            walls=WallImages(
                objective=objective,
                pixel_size_um=calibration.pixel_size_um,
                image_shape_px=shape,
                search_um=search_um,
                wells=tuple(fits),
                frames=frames,
                folder=walk.folder_text,
                elapsed_s=time.monotonic() - started,
            ),
            assumed=tuple(assumed),
        )


def _checked_refs(params: PlateWallsParams) -> tuple[WellRef, ...]:
    """The references with their names normalised, or the refusal; nothing has moved.

    A well that is not on the plate, a well given twice and a coordinate no
    stage reaches are each a refusal that names the well: a typo in a well's
    name or a unit slip would otherwise send the stage to look for a wall where
    there is none (FM-32).
    """
    refs = params.refs
    if len(refs) < _MIN_WELLS:
        raise PluginRunError(
            _NAME,
            "give at least two wells in refs, each with the stage position where "
            "it was centred by eye",
        )
    if len(refs) > _MAX_WELLS:
        raise PluginRunError(
            _NAME, f"give at most {_MAX_WELLS} wells in refs, got {len(refs)}"
        )
    wells = nominal_offsets(params.plate)
    seen: set[str] = set()
    checked: list[WellRef] = []
    for ref in refs:
        name = normalise_well(ref.well)
        if name not in wells:
            raise PluginRunError(
                _NAME, f"{ref.well} is not a well of the {params.plate} plate"
            )
        if name in seen:
            raise PluginRunError(_NAME, f"{name} is given twice in refs")
        seen.add(name)
        if max(abs(ref.x_um), abs(ref.y_um)) > _MAX_CENTRE_UM:
            raise PluginRunError(
                _NAME,
                f"{name}'s recorded position ({ref.x_um:g}, {ref.y_um:g}) µm is "
                f"beyond ±{_MAX_CENTRE_UM:g} µm: is it in µm?",
            )
        checked.append(WellRef(well=name, x_um=ref.x_um, y_um=ref.y_um))
    return tuple(checked)


def _fit_plate(
    params: PlateWallsParams, measured: tuple[WellRef, ...]
) -> tuple[PlateResult, float]:
    """The plate from the fitted centres, and its worst well's distance from where the others put it.

    ``calibrate`` refuses a spacing off the plate's definition and a rotation
    beyond the bound (FM-121: two wells the same distance apart fit 90° with no
    residual); the residual bound is this plugin's, 100 µm, because a wall's
    centre is measured to a micrometre or two and a well further out than that
    is a wrong well or a wrong plate.
    """
    problem = ""
    try:
        fitted = calibrate(
            params.plate,
            measured,
            max_rotation_deg=params.max_rotation_deg,
            orientation=params.orientation,
        )
    except ValueError as exc:
        problem = str(exc) or "the plate fit was refused"
    if problem:
        # Raised outside the handler: nothing here keeps the fit's error (FM-98).
        raise PluginRunError(_NAME, problem)
    worst, distance = max(
        (
            (m.well, math.dist(fitted.well_center_um(m.well), (m.x_um, m.y_um)))
            for m in measured
        ),
        key=lambda pair: pair[1],
    )
    if distance > params.max_residual_um:
        raise PluginRunError(
            _NAME,
            f"{worst} is {distance:.0f} µm from where the other wells put it "
            f"(bound {params.max_residual_um:g} µm): record it again, or check "
            "the plate type",
        )
    return fitted, distance
