"""The hardware loop that measures a well's centre from images of its four walls (m3 §17.4, §20.2).

``plate-walls`` registers a plate with it, and ``objective-offsets`` measures
the same wells again under another objective with it (m3 §20): one loop, so
that a refusal, a bound or a frame's framing cannot differ between the two
tools. Everything here that refuses names the **stage** (``PluginRunError``'s
first argument and the warning that says where the stage was left), the
plugin whose run it is.

The decisions that keep a measurement from lying live here, and they are the
ones ``plugin.py``'s docstring (``plate-walls``) lists: everything that can be
checked is checked before anything moves (FM-30); a frame is placed where the
stage says it was, not where it was sent (FM-142); the wall of each side is
measured in a frame centred on it, so that the threshold's bias is the same on
opposite walls and cancels in the centre; a fit that cannot be trusted is a
refusal that names the well; nothing moves to put something back (FM-88).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from smc.hardware.capabilities import XY, Camera, XYStage
from smc.hardware.errors import SafetyRefusedError
from smc.plugins.api import RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import Matrix2, fov_of
from smc.plugins.plate_registration.algorithm import DIAMETER_TOLERANCE
from smc.plugins.plate_walls.algorithm import (
    MAX_FRAMES,
    MAX_RMS_FRACTION,
    MIN_FRAME_SIDE_PX,
    OUTWARD,
    SIDES,
    along_um,
    encode_wall_frame,
    find_wall,
    fit_circle,
    wall_points_um,
)
from smc.plugins.slots import (
    CameraCalibration,
    WallFrame,
    WallSide,
    WellFit,
    WellRef,
)
from smc.run.files import write_atomically

__all__ = [
    "WallWalk",
    "Where",
    "check_calibration",
    "check_frame",
    "check_soft_limits",
    "check_walk_size",
    "checked_search",
    "measure_well",
    "move",
    "warn_left",
]

#: Below this a matrix is a camera that images nothing (``smc.plugins.geometry``'s
#: own bound): the walls' directions cannot be turned into pixels.
_MIN_DETERMINANT = 1e-12
_NO_OBJECTIVE = "an unknown objective"
_NO_LIMITS_LINE = (
    "the stage reports no soft limits, so the frames' moves were not checked "
    "against any travel"
)


@dataclass(frozen=True, slots=True)
class WallWalk:
    """What one run looks with: the camera, the nominal well and the walk, fixed before the first move.

    ``stage`` is the plugin whose refusals these are. ``no_wall_hint`` is the
    end of the "no wall found" refusal, after its first clause, with ``{well}``
    where the well's name goes: what to check differs by tool (a registration
    asks whether the stage was in the well when it was recorded, the offsets of
    another objective whether it images the wall as the first one did).
    """

    stage: str
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
    no_wall_hint: str


class Where:
    """What the loop is doing, for the warning that says where the stage was left.

    Text such as ``B3's +X wall``: a move's refusal names the same thing, so the
    operator reads one name in both.
    """

    __slots__ = ("text",)

    def __init__(self) -> None:
        self.text = "before the first move"


def check_calibration(
    stage: str,
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
            stage,
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
            stage,
            f"the camera calibration is not of this stand: it is of {was}, the camera is at {now}; {ask}",
        )
    if calibration.image_shape_px != shape:
        raise PluginRunError(
            stage,
            "the camera calibration is not of this stand: it is of a frame of "
            f"{calibration.image_shape_px}, the camera gives {shape}; {ask}",
        )
    (m00, m01), (m10, m11) = calibration.matrix_um_per_px
    determinant = m00 * m11 - m01 * m10
    if not abs(determinant) >= _MIN_DETERMINANT:
        raise PluginRunError(
            stage,
            "the camera calibration's matrix cannot be inverted (determinant "
            f"{determinant!r}); {ask}",
        )


def check_frame(
    stage: str, matrix: Matrix2, shape: tuple[int, int], diameter_um: float
) -> None:
    """Refuse a frame too small to tell a wall from its own corner, or a field that holds a whole well."""
    if min(shape) < MIN_FRAME_SIDE_PX:
        raise PluginRunError(
            stage,
            f"a frame of {shape[0]} \N{MULTIPLICATION SIGN} {shape[1]} px is too "
            "small to find a wall in",
        )
    field_um = max(fov_of(matrix, shape))
    if field_um >= diameter_um:
        raise PluginRunError(
            stage,
            f"the field of {field_um:.0f} µm holds a whole well of "
            f"{diameter_um:.0f} µm: use a higher magnification",
        )


def checked_search(
    stage: str, search_um: float | None, radius_um: float, *, default_um: float
) -> float:
    """The search distance: the parameter, or the caller's default; never more than the radius (FM-121).

    A search past the radius would look for a wall beyond the well's centre
    on the other side. The default is held to the same bound as the parameter,
    so that a caller's default cannot slip past it.
    """
    value = default_um if search_um is None else search_um
    if value > radius_um:
        raise PluginRunError(
            stage,
            f"search_um {value:g} is more than the well's radius {radius_um:g} µm",
        )
    return value


def check_walk_size(stage: str, n_wells: int, search_um: float, step_um: float) -> None:
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
            stage,
            f"up to {worst} frames: use a lower magnification or a smaller search_um",
        )


def check_soft_limits(
    stage: str, xy: XYStage, refs: tuple[WellRef, ...], offsets: tuple[float, ...]
) -> list[str]:
    """Refuse a walk that leaves the stage's soft limits; say when there are none to check.

    The test is the layer's own (``low <= v <= high`` on both axes), so a walk
    frame that passes here is one the move will not refuse. The frame centred on
    the wall is not one of them: it lies at the wall's measured distance, which can
    be a step past the walk's last offset, and the layer refuses it if it leaves the
    travel (the stage stays where it is). ``limits_um()`` is ``None`` when the
    stand does not say, and then the result says that no travel was checked: what
    is true of the layer, not what to do about it.
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
                        stage,
                        f"{ref.well}'s {side} wall would be looked for at "
                        f"({x + 0.0:.1f}, {y + 0.0:.1f}) µm, outside the stage's soft limits",
                    )
    return []


def move(
    stage: str, xy: XYStage, x_um: float, y_um: float, what: str
) -> tuple[float, float]:
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
    raise PluginRunError(stage, problem)


def _snap(camera: Camera, ctx: RunContext, walk: WallWalk) -> npt.NDArray[np.generic]:
    """Settle, then one frame, which must be the calibration's shape (a binning or region change is a refusal)."""
    ctx.wait_s(walk.settle_s)
    frame = np.asarray(camera.snap())
    if frame.shape != walk.shape:
        raise PluginRunError(
            walk.stage,
            f"the frame has shape {frame.shape}, the calibration {walk.shape}: "
            "check the camera's binning and region",
        )
    return frame


def measure_well(
    xy: XYStage,
    camera: Camera,
    ctx: RunContext,
    walk: WallWalk,
    ref: WellRef,
    offsets: tuple[float, ...],
    where: Where,
    save_errors: list[str],
    on_side: Callable[[str], None],
) -> tuple[WellFit, int]:
    """One well: its four walls, then the circle fitted to them and the refusals of that fit.

    The loop does not count progress: the caller knows how many sides the whole
    run has (a registration's wells, the offsets' wells under each target), so
    it gets ``on_side("<well> <side>")`` after each side and counts.
    """
    well = ref.well
    recorded = (ref.x_um, ref.y_um)
    points: list[npt.NDArray[np.float64]] = []
    wall_frames: list[WallFrame] = []
    taken = 0
    for side in SIDES:
        where.text = f"{well}'s {side} wall"
        wall, wall_points = _measure_side(
            xy, camera, ctx, walk, well, recorded, side, offsets, save_errors
        )
        wall_frames.append(wall)
        points.append(wall_points)
        taken += wall.frames
        on_side(f"{well} {side}")

    problem = ""
    try:
        fit = fit_circle(np.vstack(points), floor_um=2.0 * walk.pixel_size_um)
    except ValueError as exc:
        problem = f"{well}'s walls do not fit a circle: {exc}"
    if problem:
        # Raised outside the handler: nothing here keeps the fit's error (FM-98).
        raise PluginRunError(walk.stage, problem)

    nominal_um = 2.0 * walk.radius_um
    size_error = fit.diameter_um / nominal_um - 1.0
    seen_at = ", ".join(f"{w.side} ({w.x_um:.0f}, {w.y_um:.0f})" for w in wall_frames)
    if abs(size_error) > DIAMETER_TOLERANCE:
        raise PluginRunError(
            walk.stage,
            f"{well} measures {fit.diameter_um:.0f} µm across and the {walk.plate} "
            f"plate's wells {nominal_um:.0f} µm: is it the right plate type? Its "
            f"walls were imaged at {seen_at} µm",
        )
    bound_um = MAX_RMS_FRACTION * walk.radius_um
    if fit.rms_um > bound_um:
        raise PluginRunError(
            walk.stage,
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
    walk: WallWalk,
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
        at = move(
            walk.stage,
            xy,
            recorded[0] + outward[0] * d,
            recorded[1] + outward[1] * d,
            what,
        )
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
        # The hint's ``{well}`` is replaced, not formatted: a label in it may hold braces.
        hint = walk.no_wall_hint.replace("{well}", well)
        raise PluginRunError(
            walk.stage,
            f"no wall found on {well}'s {side} side in {frames} frames, from "
            f"{walk.radius_um - walk.search_um:.0f} to "
            f"{walk.radius_um + walk.search_um:.0f} µm out of the recorded centre "
            f"({recorded[0] + 0.0:.1f}, {recorded[1] + 0.0:.1f}) µm: {hint}",
        )

    ctx.raise_if_stopped()
    at = move(
        walk.stage,
        xy,
        recorded[0] + outward[0] * found,
        recorded[1] + outward[1] * found,
        what,
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
            walk.stage,
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
    walk: WallWalk,
    frame: npt.NDArray[np.generic],
    well: str,
    side: WallSide,
    at: tuple[float, float],
    save_errors: list[str],
) -> str | None:
    """Write the centred frame whole (FM-38); the name, or ``None`` when it is not saved or could not be.

    A frame that cannot be written is not the measurement's failure: the
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


def warn_left(
    stage: str, ctx: RunContext, xy: XYStage, start: XY, where: Where
) -> None:
    """Say where the stage was left, reading it in a ``try`` of its own.

    A read that fails must not hide the error that is already on its way: the
    caller re-raises that one, whatever happens here. ``+ 0.0`` turns a ``-0.0``
    (the demo stage reads one at its origin) into ``0.0`` in the message.
    """
    try:
        here = xy.position_um()
    except Exception:
        ctx.logger.warning(
            "%s stopped at %s: the stage is left at an unknown position "
            "(it started at (%.1f, %.1f) µm)",
            stage,
            where.text,
            start.x_um + 0.0,
            start.y_um + 0.0,
        )
        return
    ctx.logger.warning(
        "%s stopped at %s: the stage is left at (%.1f, %.1f) µm, not "
        "returned to (%.1f, %.1f) µm where it started",
        stage,
        where.text,
        here.x_um + 0.0,
        here.y_um + 0.0,
        start.x_um + 0.0,
        start.y_um + 0.0,
    )
