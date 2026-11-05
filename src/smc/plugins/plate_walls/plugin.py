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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import useq
from pydantic import Field, field_validator

from smc.hardware.capabilities import XY, Camera, ObjectiveTurret, XYStage
from smc.hardware.errors import SafetyRefusedError
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import Matrix2, fov_of
from smc.plugins.plate_registration.algorithm import (
    DIAMETER_TOLERANCE,
    MAX_ROTATION_DEG,
    calibrate,
    nominal_offsets,
)
from smc.plugins.plate_walls.algorithm import (
    DEFAULT_SEARCH_FRACTION,
    MAX_FRAMES,
    MAX_RMS_FRACTION,
    MIN_FRAME_SIDE_PX,
    OUTWARD,
    SIDES,
    STEP_FRACTION,
    along_um,
    encode_wall_frame,
    find_wall,
    fit_circle,
    walk_offsets_um,
    wall_points_um,
)
from smc.plugins.slots import (
    CameraCalibration,
    PlateResult,
    WallFrame,
    WallImages,
    WallSide,
    WellFit,
    WellRef,
    normalise_well,
)
from smc.run.files import new_token, write_atomically

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["PlateWallsParams", "PlateWallsPlugin"]

_NAME = "plate-walls"
_MIN_WELLS = 2
_MAX_WELLS = 6
#: No stage travels a metre: a recorded centre this far out is a unit slip (FM-32).
_MAX_CENTRE_UM = 1_000_000.0
#: Below this a matrix is a camera that images nothing (``smc.plugins.geometry``'s
#: own bound): the walls' directions cannot be turned into pixels.
_MIN_DETERMINANT = 1e-12
_NO_OBJECTIVE = "an unknown objective"
_NO_LIMITS_LINE = (
    "the stage reports no soft limits, so the frames' moves were not checked "
    "against any travel"
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
            "The largest rotation of the plate that the fit accepts; a rotation "
            "past it is read as a misnamed well or a plate with its columns the "
            "other way. Raise it only for a plate that is really turned."
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


@dataclass(frozen=True, slots=True)
class _Walk:
    """What one run looks with: the camera, the nominal well and the walk, fixed before the first move."""

    plate: str
    radius_um: float
    search_um: float
    step_um: float
    matrix: Matrix2
    shape: tuple[int, int]
    pixel_size_um: float
    settle_s: float
    folder: Path | None
    folder_text: str | None


class _Where:
    """What the loop is doing, for the warning that says where the stage was left.

    Text such as ``B3's +X wall``: a move's refusal names the same thing, so the
    operator reads one name in both.
    """

    __slots__ = ("text",)

    def __init__(self) -> None:
        self.text = "before the first move"


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
        _check_calibration(calibration, objective, turret is not None, shape)
        matrix = calibration.matrix_um_per_px
        diameter_um = float(plate.well_size[0]) * 1000.0
        _check_frame(matrix, shape, diameter_um)

        radius_um = diameter_um / 2.0
        search_um = _checked_search(params, radius_um)
        step_um = STEP_FRACTION * min(shape) * calibration.pixel_size_um
        _check_walk_size(len(refs), search_um, step_um)
        offsets = walk_offsets_um(radius_um, search_um, step_um)
        assumed = _check_soft_limits(xy, refs, offsets)

        token = new_token()
        walk = _Walk(
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
        )

        # A stage still travelling from an earlier move reads a fraction of it.
        xy.wait()
        start = xy.position_um()
        started = time.monotonic()
        where = _Where()
        save_errors: list[str] = []
        fits: list[WellFit] = []
        frames = 0
        try:
            for index, ref in enumerate(refs):
                fit, taken = _measure_well(
                    xy,
                    camera,
                    ctx,
                    walk,
                    ref,
                    offsets,
                    index,
                    len(refs),
                    where,
                    save_errors,
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
            _move(xy, *fits[-1].center_xy_um, where.text)
        except BaseException:
            # A stop, a halt, a fault or a refusal: the stage stays where it is, the
            # exception goes on (FM-88, FM-64), and nothing is caught.
            _warn_left(ctx, xy, start, where)
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
            method="wall-images",
            residual_um=residual_um,
            scale_error=fitted.scale_error,
            n_refs=len(measured),
            refs=measured,
            fit=None,
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


def _check_calibration(
    calibration: CameraCalibration,
    objective: str | None,
    has_turret: bool,
    shape: tuple[int, int],
) -> None:
    """Refuse a calibration that is not of this objective and frame, or whose matrix cannot be inverted.

    A calibration of another objective or binning is silently wrong, not merely
    stale: the camera's turn decides which way every wall lies (the Ti2-E's 10x
    camera is turned 180°). A turret at an empty position reports the label
    ``""``, which is no objective and not one called nothing (FM-125). The
    matrix's determinant is checked the way its downstream user would
    (``not >=``, so that ``nan`` is refused too).
    """
    ask = (
        f"run camera-calibration with {objective} in place, on a field with "
        "texture: a well's wall will do"
        if objective
        else "run camera-calibration on a field with texture: a well's wall will do"
    )
    if has_turret and not objective:
        raise PluginRunError(
            _NAME,
            "no objective is in place: the turret is at an empty position; turn "
            "it to an objective and run camera-calibration there",
        )
    if calibration.objective != objective:
        was = (
            calibration.objective
            if calibration.objective is not None
            else _NO_OBJECTIVE
        )
        now = objective if objective is not None else _NO_OBJECTIVE
        raise PluginRunError(
            _NAME,
            f"the camera calibration is not of this stand: it is of {was}, the camera is at {now}; {ask}",
        )
    if calibration.image_shape_px != shape:
        raise PluginRunError(
            _NAME,
            "the camera calibration is not of this stand: it is of a frame of "
            f"{calibration.image_shape_px}, the camera gives {shape}; {ask}",
        )
    (m00, m01), (m10, m11) = calibration.matrix_um_per_px
    determinant = m00 * m11 - m01 * m10
    if not abs(determinant) >= _MIN_DETERMINANT:
        raise PluginRunError(
            _NAME,
            "the camera calibration's matrix cannot be inverted (determinant "
            f"{determinant!r}); {ask}",
        )


def _check_frame(matrix: Matrix2, shape: tuple[int, int], diameter_um: float) -> None:
    """Refuse a frame too small to tell a wall from its own corner, or a field that holds a whole well."""
    if min(shape) < MIN_FRAME_SIDE_PX:
        raise PluginRunError(
            _NAME,
            f"a frame of {shape[0]} \N{MULTIPLICATION SIGN} {shape[1]} px is too "
            "small to find a wall in",
        )
    field_um = max(fov_of(matrix, shape))
    if field_um >= diameter_um:
        raise PluginRunError(
            _NAME,
            f"the field of {field_um:.0f} µm holds a whole well of "
            f"{diameter_um:.0f} µm: use a higher magnification",
        )


def _checked_search(params: PlateWallsParams, radius_um: float) -> float:
    """The search distance: the parameter, or half the radius; never more than the radius (FM-121).

    A search past the radius would look for a wall beyond the well's centre
    on the other side. The default obeys the bound by construction, and the
    parameter is held to the same one.
    """
    if params.search_um is None:
        return DEFAULT_SEARCH_FRACTION * radius_um
    if params.search_um > radius_um:
        raise PluginRunError(
            _NAME,
            f"search_um {params.search_um:g} is more than the well's radius "
            f"{radius_um:g} µm",
        )
    return params.search_um


def _check_walk_size(n_wells: int, search_um: float, step_um: float) -> None:
    """Refuse a run that could take more than ``MAX_FRAMES`` frames, counted before any walk is built (FM-74).

    The worst case is every side walking its whole list and then taking its
    centred frame. The length of a walk is arithmetic (``1 + 2 * floor(search /
    step)``), in Python ints and capped, so that a pixel size far too small does
    not make the count itself an allocation.
    """
    ratio = search_um / step_um
    steps = min(math.floor(ratio), MAX_FRAMES) if math.isfinite(ratio) else MAX_FRAMES
    worst = n_wells * len(SIDES) * (1 + 2 * steps + 1)
    if worst > MAX_FRAMES:
        raise PluginRunError(
            _NAME,
            f"up to {worst} frames: use a lower magnification or a smaller search_um",
        )


def _check_soft_limits(
    xy: XYStage, refs: tuple[WellRef, ...], offsets: tuple[float, ...]
) -> list[str]:
    """Refuse a walk that leaves the stage's soft limits; say when there are none to check.

    The test is the layer's own (``low <= v <= high`` on both axes), so a frame
    that passes here is one the move will not refuse. ``limits_um()`` is
    ``None`` when the stand does not say, and then the result says that no
    travel was checked: what is true of the layer, not what to do about it.
    """
    limits = xy.limits_um()
    if limits is None:
        return [_NO_LIMITS_LINE]
    x_limits, y_limits = limits
    for ref in refs:
        for side in SIDES:
            ox, oy = OUTWARD[side]
            for d in offsets:
                x, y = ref.x_um + ox * d, ref.y_um + oy * d
                if not (
                    x_limits.low_um <= x <= x_limits.high_um
                    and y_limits.low_um <= y <= y_limits.high_um
                ):
                    raise PluginRunError(
                        _NAME,
                        f"{ref.well}'s {side} wall would be looked for at "
                        f"({x + 0.0:.1f}, {y + 0.0:.1f}) µm, outside the stage's soft limits",
                    )
    return []


def _move(xy: XYStage, x_um: float, y_um: float, what: str) -> tuple[float, float]:
    """Move there and return the readback: where the stage says it is, never where it was sent (FM-142).

    A refusal of the layer (the soft limits, a move already under way) names what
    the move was for, the well and the side. A halt and a stop are not
    ``SafetyRefusedError``s and pass unchanged (FM-64). A readback that is not
    finite is refused.
    """
    problem = ""
    try:
        reached = xy.move_to_um(x_um, y_um)
    except SafetyRefusedError as exc:
        problem = (
            f"{what}: the move to ({x_um + 0.0:.1f}, {y_um + 0.0:.1f}) µm was "
            f"refused: {exc}"
        )
    else:
        if not (math.isfinite(reached.x_um) and math.isfinite(reached.y_um)):
            problem = "the stage's readback is not a finite position"
        else:
            return (reached.x_um, reached.y_um)
    # Raised outside the handler: nothing here keeps the layer's error (FM-98).
    raise PluginRunError(_NAME, problem)


def _snap(camera: Camera, ctx: RunContext, walk: _Walk) -> npt.NDArray[np.generic]:
    """Settle, then one frame, which must be the calibration's shape (a binning or region change is a refusal)."""
    ctx.wait_s(walk.settle_s)
    frame = np.asarray(camera.snap())
    if frame.shape != walk.shape:
        raise PluginRunError(
            _NAME,
            f"the frame has shape {frame.shape}, the calibration {walk.shape}: "
            "check the camera's binning and region",
        )
    return frame


def _measure_well(
    xy: XYStage,
    camera: Camera,
    ctx: RunContext,
    walk: _Walk,
    ref: WellRef,
    offsets: tuple[float, ...],
    index: int,
    n_wells: int,
    where: _Where,
    save_errors: list[str],
) -> tuple[WellFit, int]:
    """One well: its four walls, then the circle fitted to them and the refusals of that fit."""
    well = ref.well
    recorded = (ref.x_um, ref.y_um)
    points: list[npt.NDArray[np.float64]] = []
    wall_frames: list[WallFrame] = []
    taken = 0
    for done, side in enumerate(SIDES, start=1):
        where.text = f"{well}'s {side} wall"
        wall, wall_points = _measure_side(
            xy, camera, ctx, walk, well, recorded, side, offsets, save_errors
        )
        wall_frames.append(wall)
        points.append(wall_points)
        taken += wall.frames
        ctx.progress(index * len(SIDES) + done, n_wells * len(SIDES), f"{well} {side}")

    problem = ""
    try:
        fit = fit_circle(np.vstack(points), floor_um=2.0 * walk.pixel_size_um)
    except ValueError as exc:
        problem = f"{well}'s walls do not fit a circle: {exc}"
    if problem:
        # Raised outside the handler: nothing here keeps the fit's error (FM-98).
        raise PluginRunError(_NAME, problem)

    nominal_um = 2.0 * walk.radius_um
    size_error = fit.diameter_um / nominal_um - 1.0
    seen_at = ", ".join(f"{w.side} ({w.x_um:.0f}, {w.y_um:.0f})" for w in wall_frames)
    if abs(size_error) > DIAMETER_TOLERANCE:
        raise PluginRunError(
            _NAME,
            f"{well} measures {fit.diameter_um:.0f} µm across and the {walk.plate} "
            f"plate's wells {nominal_um:.0f} µm: is it the right plate type? Its "
            f"walls were imaged at {seen_at} µm",
        )
    bound_um = MAX_RMS_FRACTION * walk.radius_um
    if fit.rms_um > bound_um:
        raise PluginRunError(
            _NAME,
            f"{well}'s wall points lie {fit.rms_um:.3g} µm from a circle on average, "
            f"more than {bound_um:.3g} µm: a frame may show something else than the "
            "wall (a bubble, a scratch, a label); look at its frames, imaged at "
            f"{seen_at} µm",
        )
    return (
        WellFit(
            well=well,
            recorded_xy_um=recorded,
            center_xy_um=fit.center_xy_um,
            diameter_um=fit.diameter_um,
            size_error=size_error,
            rms_um=fit.rms_um,
            n_points=fit.n_used,
            walls=tuple(wall_frames),
        ),
        taken,
    )


def _measure_side(
    xy: XYStage,
    camera: Camera,
    ctx: RunContext,
    walk: _Walk,
    well: str,
    recorded: tuple[float, float],
    side: WallSide,
    offsets: tuple[float, ...],
    save_errors: list[str],
) -> tuple[WallFrame, npt.NDArray[np.float64]]:
    """One wall: walk until it is found, centre a frame on it, and keep that frame's points alone.

    Only the centred frame's points are kept: every side is framed the same way,
    so the threshold's bias (the meniscus darkens the inside towards the wall)
    is the same on opposite walls and cancels in the centre (§17.1). The walk
    ends at the first frame whose wall lies within ``search_um`` of the nominal
    radius, and goes on otherwise.
    """
    outward = OUTWARD[side]
    what = f"{well}'s {side} wall"
    frames = 0
    found: float | None = None
    for d in offsets:
        ctx.raise_if_stopped()
        at = _move(xy, recorded[0] + outward[0] * d, recorded[1] + outward[1] * d, what)
        frame = _snap(camera, ctx, walk)
        frames += 1
        edge = find_wall(frame, outward_xy=outward, matrix=walk.matrix)
        if edge is None:
            continue
        out = along_um(
            wall_points_um(
                edge.points_px, *at, matrix=walk.matrix, image_shape_px=walk.shape
            ),
            recorded,
            outward,
        )
        if abs(out - walk.radius_um) <= walk.search_um:
            found = out
            break
    if found is None:
        raise PluginRunError(
            _NAME,
            f"no wall found on {well}'s {side} side in {frames} frames, from "
            f"{walk.radius_um - walk.search_um:.0f} to "
            f"{walk.radius_um + walk.search_um:.0f} µm out of the recorded centre "
            f"({recorded[0] + 0.0:.1f}, {recorded[1] + 0.0:.1f}) µm: check that the "
            "light is on (the inside of a well is brighter than the plastic in "
            f"brightfield) and that the stage was in {well} when it was recorded, "
            "then record it again",
        )

    ctx.raise_if_stopped()
    at = _move(
        xy, recorded[0] + outward[0] * found, recorded[1] + outward[1] * found, what
    )
    frame = _snap(camera, ctx, walk)
    frames += 1
    edge = find_wall(frame, outward_xy=outward, matrix=walk.matrix)
    kept: npt.NDArray[np.float64] | None = None
    if edge is not None:
        kept = wall_points_um(
            edge.points_px, *at, matrix=walk.matrix, image_shape_px=walk.shape
        )
        if abs(along_um(kept, recorded, outward) - walk.radius_um) > walk.search_um:
            kept = None
    if edge is None or kept is None:
        raise PluginRunError(
            _NAME,
            f"the wall of {well}'s {side} side was lost in the frame centred on it "
            f"at ({at[0] + 0.0:.1f}, {at[1] + 0.0:.1f}) µm",
        )

    file = _save(walk, frame, well, side, at, save_errors)
    return (
        WallFrame(
            side=side,
            x_um=at[0],
            y_um=at[1],
            n_points=len(edge.points_px),
            frames=frames,
            file=file,
        ),
        kept,
    )


def _save(
    walk: _Walk,
    frame: npt.NDArray[np.generic],
    well: str,
    side: WallSide,
    at: tuple[float, float],
    save_errors: list[str],
) -> str | None:
    """Write the centred frame whole (FM-38); the name, or ``None`` when it is not saved or could not be.

    A frame that cannot be written is not the registration's failure: the
    measurement is in the result already. The line says what the OS said,
    without the path: the result is attached to issues (FM-50).
    """
    if walk.folder is None:
        return None
    name = f"{well}_{side}.tif"
    meta = {"well": well, "side": side, "x_um": at[0], "y_um": at[1]}
    try:
        write_atomically(walk.folder / name, encode_wall_frame(frame, meta))
    except OSError as exc:
        why = exc.strerror or "could not be written"
        save_errors.append(" ".join(f"{type(exc).__name__}: {why}".split()))
        return None
    return name


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
            params.plate, measured, max_rotation_deg=params.max_rotation_deg
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


def _warn_left(ctx: RunContext, xy: XYStage, start: XY, where: _Where) -> None:
    """Say where the stage was left, reading it in a ``try`` of its own.

    A read that fails must not hide the error that is already on its way: the
    caller re-raises that one, whatever happens here. ``+ 0.0`` turns a ``-0.0``
    (the demo stage reads one at its origin) into ``0.0`` in the message.
    """
    try:
        here = xy.position_um()
    except Exception:
        ctx.logger.warning(
            "plate-walls stopped at %s: the stage is left at an unknown position "
            "(it started at (%.1f, %.1f) µm)",
            where.text,
            start.x_um + 0.0,
            start.y_um + 0.0,
        )
        return
    ctx.logger.warning(
        "plate-walls stopped at %s: the stage is left at (%.1f, %.1f) µm, not "
        "returned to (%.1f, %.1f) µm where it started",
        where.text,
        here.x_um + 0.0,
        here.y_um + 0.0,
        start.x_um + 0.0,
        start.y_um + 0.0,
    )
