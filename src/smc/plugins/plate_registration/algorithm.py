"""Where a plate is on the stage, from wells the operator centred or walls they touched (m3 §5).

Calibration measures three numbers, A1's stage position and the plate's
rotation, and takes everything else from the plate's definition: spacing and
well size are a manufacturing property, not something to fit. The nominal
geometry is read back out of ``useq`` (its registry, its names, its forward
model), so this module cannot drift from ``useq``'s conventions: which way rows
run, millimetres against micrometres, which way a positive rotation turns.

Everything here is a pure function of numbers and of the plate's name; a
failure is a ``ValueError`` whose text says what to check, and the plugin that
calls it decides what a refusal means for the run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import useq

from smc.plugins.slots import (
    PlateResult,
    WallTouch,
    WellRef,
    normalise_well,
    well_geometry,
)

__all__ = [
    "DIAMETER_TOLERANCE",
    "GOOD_RESIDUAL_UM",
    "MAX_ROTATION_DEG",
    "SCALE_TOLERANCE",
    "EdgeCentre",
    "calibrate",
    "centre_from_walls",
    "nearest_well",
    "nominal_offsets",
    "suggested_refs",
    "wall_position",
]

#: A plate whose well-to-well distance is further than this from the plate
#: definition is the wrong plate type or a misnamed well: refused, not fitted.
SCALE_TOLERANCE = 0.02
#: A registration this good is enough for a well scan with a sensible margin: a
#: position error shifts the scan grid, it does not compound.
GOOD_RESIDUAL_UM = 300.0
#: Two opposite wall touches imply a well size; this far from the plate
#: definition is a touch of another well, or off the wall.
DIAMETER_TOLERANCE = 0.10
#: A plate is put on the stage by hand and is turned by degrees, not by tens of
#: degrees. Two references the same distance apart (A1 and A2, or A1 and B1)
#: fit a turn of 90 degrees with no residual, so the fit alone cannot say that
#: a well was misnamed (FM-121).
MAX_ROTATION_DEG = 10.0

_SIDES = ("left", "right", "top", "bottom")


@dataclass(frozen=True, slots=True)
class EdgeCentre:
    """A well's centre worked out from wall touches, and what it assumed.

    ``measured_width_um`` and ``measured_height_um`` are the sizes an opposite
    pair measured, ``None`` on an axis where only one wall was touched. That
    axis is then in ``assumed_axes`` (``"x"``, ``"y"``): its centre is one wall
    plus the nominal half-size.
    """

    well: str
    x_um: float
    y_um: float
    measured_width_um: float | None
    measured_height_um: float | None
    assumed_axes: tuple[str, ...]


def _plan_at_origin(plate: str) -> useq.WellPlatePlan:
    """The plate pinned at the origin and not turned: the nominal geometry, as ``useq`` has it."""
    return useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate), a1_center_xy=(0.0, 0.0), rotation=0.0
    )


def _xy(position: useq.Position) -> tuple[float, float]:
    """A position's x and y as floats: ``useq`` types them as optional, and a plate plan sets both."""
    return float(position.x), float(position.y)  # type: ignore[arg-type]


def nominal_offsets(plate: str) -> dict[str, tuple[float, float]]:
    """The offset of every well's centre from A1's, in µm, per ``useq``'s own geometry.

    Built from one call that asks ``useq`` for all the positions of a plate
    pinned at the origin (3 ms for 1536 wells): ``well_geometry`` per well
    would give the same numbers 100 times slower.

    Raises:
        ValueError: ``useq`` has no plate of this name (its error, unchanged).
    """
    return {str(p.name): _xy(p) for p in _plan_at_origin(plate).all_well_positions}


def suggested_refs(plate: str) -> tuple[str, ...]:
    """A1 and the two far corners: wells that make a good calibration.

    A long baseline is what pins the rotation down, because adjacent wells
    barely constrain it, so the corners are worth the extra stage travel. A
    plate with a single well (a coverslip) has one name, listed once.
    """
    names = _plan_at_origin(plate).all_well_names
    return tuple(
        dict.fromkeys(str(n) for n in (names[0][0], names[0][-1], names[-1][0]))
    )


def calibrate(
    plate: str,
    refs: Sequence[WellRef],
    *,
    max_rotation_deg: float = MAX_ROTATION_DEG,
) -> PlateResult:
    """Fit A1's stage position and the plate's rotation from wells centred under the objective.

    One reference sets A1 and assumes no rotation (a *seed*: enough to drive to
    every other well to within a few hundred µm). Two or more also fit the
    rotation, by a rigid 2-D fit (rotation and translation, Kabsch) with the
    scale fixed by the plate definition; three or more make the residual say
    something about the centring.

    The scale is checked before the fit and the rotation after it, because
    neither the scale nor the residual can see a mislabelled pair of equidistant
    wells: it fits at 90 degrees exactly (FM-121).

    Raises:
        ValueError: No reference; a well that is not on the plate, or given
            twice (after ``normalise_well``); a spacing off the plate
            definition by more than ``SCALE_TOLERANCE``; or a fitted rotation
            beyond ``max_rotation_deg``.
    """
    if not refs:
        raise ValueError("need at least one reference well")
    offsets = nominal_offsets(plate)
    names: list[str] = []
    for ref in refs:
        if normalise_well(ref.well) not in offsets:
            raise ValueError(f"well {ref.well!r} is not on the {plate} plate")
        names.append(normalise_well(ref.well))
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"well {name} given twice")
        seen.add(name)

    nominal = [offsets[name] for name in names]
    measured = [(ref.x_um, ref.y_um) for ref in refs]
    normalised = tuple(
        WellRef(well=name, x_um=x, y_um=y)
        for name, (x, y) in zip(names, measured, strict=True)
    )
    count = len(names)

    if count == 1:
        return PlateResult(
            plate=plate,
            a1_center_xy_um=(
                measured[0][0] - nominal[0][0],
                measured[0][1] - nominal[0][1],
            ),
            rotation_deg=0.0,
            method="seed",
            residual_um=None,
            scale_error=None,
            n_refs=1,
            refs=normalised,
            assumed=(
                f"one reference well ({names[0]}): rotation 0° assumed, not measured",
            ),
        )

    scale_error = _scale_error(nominal, measured)
    if abs(scale_error) > SCALE_TOLERANCE:
        raise ValueError(
            f"the measured well spacing is off the {plate} definition by "
            f"{scale_error:+.1%}: wrong plate type, or a misidentified well; "
            f"check the plate and the well names"
        )

    d_mean = (sum(p[0] for p in nominal) / count, sum(p[1] for p in nominal) / count)
    m_mean = (sum(p[0] for p in measured) / count, sum(p[1] for p in measured) / count)
    cross = 0.0
    dot = 0.0
    for (dx, dy), (mx, my) in zip(nominal, measured, strict=True):
        dx, dy = dx - d_mean[0], dy - d_mean[1]
        mx, my = mx - m_mean[0], my - m_mean[1]
        cross += dx * my - dy * mx
        dot += dx * mx + dy * my
    theta = math.atan2(cross, dot)
    cos, sin = math.cos(theta), math.sin(theta)
    a1 = (
        m_mean[0] - (cos * d_mean[0] - sin * d_mean[1]),
        m_mean[1] - (sin * d_mean[0] + cos * d_mean[1]),
    )
    rotation_deg = math.degrees(theta)
    if abs(rotation_deg) > max_rotation_deg:
        raise ValueError(
            f"the fit turns the plate by {rotation_deg:.1f}°, beyond "
            f"±{max_rotation_deg:g}°: a reference well is misidentified (two "
            f"references the same distance apart fit either way)"
        )

    # The fitted centres go through the slot's own geometry, so that the
    # residual is the distance to the centre that `well_center_um` will report.
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate), a1_center_xy=a1, rotation=rotation_deg
    )
    residual = max(
        math.dist(point, well_geometry(plan, name).center_um)
        for name, point in zip(names, measured, strict=True)
    )
    return PlateResult(
        plate=plate,
        a1_center_xy_um=a1,
        rotation_deg=rotation_deg,
        method="references",
        residual_um=residual,
        scale_error=scale_error,
        n_refs=count,
        refs=normalised,
    )


def _scale_error(
    nominal: Sequence[tuple[float, float]], measured: Sequence[tuple[float, float]]
) -> float:
    """The measured distance over the nominal one, minus 1, on the longest nominal baseline.

    A scale error shows most on the longest baseline. The first pair, in the
    order the references were given, wins a tie.
    """
    best_nominal = 0.0
    best_measured = 0.0
    for i in range(len(nominal)):
        for j in range(i + 1, len(nominal)):
            distance = math.dist(nominal[i], nominal[j])
            if distance > best_nominal:
                best_nominal = distance
                best_measured = math.dist(measured[i], measured[j])
    return best_measured / best_nominal - 1.0


def centre_from_walls(
    plate: str, well: str, touches: Sequence[WallTouch]
) -> EdgeCentre:
    """A well's centre from stage positions at its walls.

    At 40x a well is not visible (the field is a few hundred µm, a 96-well well
    6 400), so the centre is never judged by eye: touch one wall, then the
    opposite one, and take the midpoint. An opposite pair also measures the
    well, which checks that the right wall was touched; one wall gives the
    centre only with the nominal half-size, and the axis is reported as assumed.

    Each axis is read from its own coordinate: ``left`` and ``right`` use ``x_um``,
    ``top`` and ``bottom`` use ``y_um`` (``top`` is +Y, towards row A), and the
    other coordinate of a touch is never read (FM-73). The sizes are the
    plate's, width for X and height for Y.

    Raises:
        ValueError: The well is not on the plate; no touch, or a touch of
            another well; a side given twice; no wall for X or for Y; or an
            opposite pair whose size is off the plate's by more than
            ``DIAMETER_TOLERANCE``.
    """
    geometry = well_geometry(_plan_at_origin(plate), well)
    name = geometry.well
    if not touches:
        raise ValueError(f"give at least one wall touch for {name}")
    by_side: dict[str, WallTouch] = {}
    for touch in touches:
        other = normalise_well(touch.well)
        if other != name:
            raise ValueError(f"a touch of {other} was given for {name}")
        if touch.side in by_side:
            raise ValueError(f"{name}: the {touch.side} wall is given twice")
        by_side[touch.side] = touch
    if "left" not in by_side and "right" not in by_side:
        raise ValueError(f"no X wall for {name}: give left and/or right")
    if "top" not in by_side and "bottom" not in by_side:
        raise ValueError(f"no Y wall for {name}: give top and/or bottom")

    width_um, height_um = geometry.size_um
    x_walls = [by_side[side] for side in ("left", "right") if side in by_side]
    y_walls = [by_side[side] for side in ("bottom", "top") if side in by_side]
    assumed: list[str] = []

    measured_width: float | None = None
    if len(x_walls) == 2:
        left, right = x_walls
        measured_width = right.x_um - left.x_um
        _check_size(measured_width, width_um, plate, name, "left and right")
        x_um = (left.x_um + right.x_um) / 2.0
    else:
        assumed.append("x")
        (only_x,) = x_walls
        sign = 1.0 if only_x.side == "left" else -1.0
        x_um = only_x.x_um + sign * width_um / 2.0

    measured_height: float | None = None
    if len(y_walls) == 2:
        bottom, top = y_walls
        measured_height = top.y_um - bottom.y_um
        _check_size(measured_height, height_um, plate, name, "top and bottom")
        y_um = (top.y_um + bottom.y_um) / 2.0
    else:
        assumed.append("y")
        (only_y,) = y_walls
        sign = 1.0 if only_y.side == "bottom" else -1.0
        y_um = only_y.y_um + sign * height_um / 2.0

    return EdgeCentre(
        well=name,
        x_um=x_um,
        y_um=y_um,
        measured_width_um=measured_width,
        measured_height_um=measured_height,
        assumed_axes=tuple(assumed),
    )


def _check_size(
    size_um: float, nominal_um: float, plate: str, well: str, walls: str
) -> None:
    """Refuse an opposite pair whose distance is not the plate's well size, signed: a swapped pair is negative."""
    if abs(size_um / nominal_um - 1.0) > DIAMETER_TOLERANCE:
        raise ValueError(
            f"{well}'s {walls} walls are {size_um:.0f} µm apart, a {plate} well is "
            f"{nominal_um:.0f} µm ({size_um / nominal_um - 1:+.1%}): a touch of "
            f"another well, or off the wall; touch them again"
        )


def wall_position(result: PlateResult, well: str, side: str) -> tuple[float, float]:
    """The stage point at one wall of a well: the centre plus the half-size along the plate's own axes.

    The offset is turned by the plate's rotation, so that on a turned plate the
    point is on the wall and not on the stage axis. ``side`` is ``left``,
    ``right`` (along the plate's X) or ``top`` (+Y, towards row A), ``bottom``,
    and the half-size is the well's width for X and its height for Y (FM-73).

    Raises:
        ValueError: ``side`` is not one of the four; the well is not on the plate.
    """
    if side not in _SIDES:
        raise ValueError(f"side {side!r} is not one of left, right, top, bottom")
    geometry = well_geometry(result.to_plan(), well)
    half_w, half_h = geometry.size_um[0] / 2.0, geometry.size_um[1] / 2.0
    ox, oy = {
        "left": (-half_w, 0.0),
        "right": (half_w, 0.0),
        "top": (0.0, half_h),
        "bottom": (0.0, -half_h),
    }[side]
    angle = math.radians(result.rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    cx, cy = geometry.center_um
    return (cx + cos * ox - sin * oy, cy + sin * ox + cos * oy)


def nearest_well(result: PlateResult, x_um: float, y_um: float) -> tuple[str, float]:
    """The well whose centre is nearest a stage point, and the distance to it in µm.

    There is no cut-off: a point far from the plate still has a nearest well,
    and the distance says how far. On a tie the first well in ``useq``'s order
    wins.

    Raises:
        ValueError: A coordinate is not finite.
    """
    if not (math.isfinite(x_um) and math.isfinite(y_um)):
        raise ValueError(f"the point must be finite, got ({x_um!r}, {y_um!r})")
    best_name = ""
    best = math.inf
    for position in result.to_plan().all_well_positions:
        distance = math.dist((x_um, y_um), _xy(position))
        if distance < best:
            best_name, best = str(position.name), distance
    return best_name, best
