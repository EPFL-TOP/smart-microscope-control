"""The field grid of a scan plan: how many fields, which of them touch the well, what it costs (m3 §7.2).

Pure arithmetic over the camera's footprint, the plate's wells and the measured
timings: no stage and no camera, so it is unit-tested on synthetic numbers and
the plugin only supplies the camera's matrix.

Two things here are easy to get wrong and invisible on a square frame or a
round well (FM-73). The grid's **rows come from the heights** of the field and
the well and its **columns from the widths**. And **clipping to the well** keeps
a field when its *footprint* meets the well's shape, not when its centre is
inside: a camera's frame is a rectangle (turned by the camera's rotation, its
corners listed the other way round when the image is mirrored), a well is a disc
or a rectangle turned by the plate's rotation, and a field that misses the well
images only plastic.

The grid itself is ``useq``'s (``GridRowsColumns``, serpentine inside a well);
the grid is **axis-aligned in the stage frame** and so is never turned with the
camera, which is why ``gap_overlap`` exists.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import useq

from smc.plugins.scan_plan.wells import well_indices
from smc.plugins.slots import PlannedField, PlateResult, TimingsResult, WellGeometry
from smc.plugins.slots.plate import normalise_well, well_geometry

__all__ = [
    "estimate_assumed",
    "estimate_s",
    "gap_overlap",
    "grid_size",
    "plan_fields",
    "refocus_count",
    "turn_from_axes_deg",
]

Point = tuple[float, float]

#: A ratio of well to field beyond this is as good as infinite: it keeps ``math.ceil``
#: from an ``OverflowError`` on a field of 1e-300 µm, and any count it gives is far
#: above what the plugin accepts (100 per axis, FM-121).
_RATIO_CAP = 1e9


def grid_size(
    fov_um: tuple[float, float],
    well_size_um: tuple[float, float],
    *,
    coverage: float,
    overlap: float,
) -> tuple[int, int]:
    """``(rows, columns)`` of fields that span ``coverage`` of the well; rows from the heights.

    Per axis ``ceil(max(0, size * coverage - fov) / step) + 1`` fields, with
    ``step = fov * (1 - overlap)`` (``nikon-control``'s ``fields_for_coverage``):
    the first field covers one field of view, and each further one adds a step.
    A well that one field already covers takes one, not two.

    Raises:
        ValueError: A size is not finite and positive, ``coverage`` is not, or
            ``overlap`` is not in ``[0, 1)``. A size that is 0 would otherwise
            divide by it, and a pixel size of 0 means "not measured" (FM-104).
    """
    for label, sizes in (("a field of view", fov_um), ("a well size", well_size_um)):
        if not all(math.isfinite(size) and size > 0.0 for size in sizes):
            raise ValueError(
                f"grid_size needs {label} that is finite and positive, got {sizes!r}"
            )
    if not (math.isfinite(coverage) and coverage > 0.0):
        raise ValueError(
            f"grid_size needs a coverage that is finite and positive, got {coverage!r}"
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"grid_size needs an overlap in [0, 1), got {overlap!r}")

    def along(fov: float, size: float) -> int:
        step = fov * (1.0 - overlap)
        wanted = max(0.0, size * coverage - fov)
        ratio = wanted / step if step > 0.0 else math.inf
        return math.ceil(min(ratio, _RATIO_CAP)) + 1

    return along(fov_um[1], well_size_um[1]), along(fov_um[0], well_size_um[0])


def gap_overlap(rotation_deg: float, overlap: float) -> float | None:
    """The overlap that closes the gaps a turned camera leaves in an axis-aligned grid, or ``None``.

    The angle that matters is ``phi``, the distance from the camera's rotation to
    the nearest multiple of 90 degrees: a camera at 90 or 180 degrees fills its
    bounding box. Gaps open when ``sin(phi) > overlap / 2``. The suggestion is
    ``ceil(200 sin(phi)) / 100``, which is ``inf`` when it would be 0.9 or more:
    no overlap that the plugin accepts closes them, and the camera must be
    straightened. The check comes early (the first gap opens a little after the
    threshold), never late.

    Raises:
        ValueError: ``rotation_deg`` is not finite.
    """
    sine = math.sin(math.radians(turn_from_axes_deg(rotation_deg)))
    if sine <= overlap / 2.0:
        return None
    suggested = math.ceil(200.0 * sine) / 100.0
    return math.inf if suggested >= 0.9 else suggested


def turn_from_axes_deg(rotation_deg: float) -> float:
    """``phi`` in ``[0, 45]``: the distance from a camera's rotation to the nearest multiple of 90 degrees.

    Raises:
        ValueError: ``rotation_deg`` is not finite.
    """
    if not math.isfinite(rotation_deg):
        raise ValueError(f"rotation_deg must be finite, got {rotation_deg!r}")
    return abs(((rotation_deg + 45.0) % 90.0) - 45.0)


def plan_fields(
    plate: PlateResult,
    wells: Sequence[str],
    *,
    fov_um: tuple[float, float],
    footprint_of: Callable[[float, float], Sequence[Point]],
    rows: int,
    columns: int,
    overlap: float,
    clip_to_well: bool,
) -> tuple[useq.WellPlatePlan, tuple[PlannedField, ...]]:
    """A grid of fields in each well, in the order the wells are given, clipped to the well.

    The plan is ``useq``'s: the plate as registered, ``wells`` in the given order
    (the plan's own order is the scan's, never the typing order) and a
    ``GridRowsColumns``. ``fields`` are its image positions that survive clipping.
    ``index`` counts from 0 in each well over the fields that were kept, and
    ``name`` stays ``useq``'s (``A1_0003``, gaps included), so a name always says
    which grid position it is.

    Args:
        plate: Where the plate is.
        wells: Normalised well names in visiting order, without duplicates.
        fov_um: ``(width, height)`` of one field in stage axes, which is the
            grid's step before the overlap.
        footprint_of: The four stage corners of a frame taken at ``(x, y)``.
        rows: Fields along Y.
        columns: Fields along X.
        overlap: Fraction of a field shared with its neighbour, in ``[0, 1)``.
        clip_to_well: Keep only the fields whose footprint meets the well.

    Raises:
        ValueError: No wells, a duplicate or a well that is not on the plate, a
            grid or an overlap that is not one, a position that ``useq`` gives
            to another well or without coordinates (``useq`` changed, not the
            plate), or a well that no field touches (it would be left out of
            the scan without a word).
    """
    if not wells:
        raise ValueError("no wells to plan: give at least one well")
    if rows < 1 or columns < 1:
        raise ValueError("a grid needs at least one row and one column")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")
    if not all(math.isfinite(size) and size > 0.0 for size in fov_um):
        raise ValueError(
            f"plan_fields needs a field of view that is finite and positive, got {fov_um!r}"
        )
    names = [normalise_well(well) for well in wells]
    if len(set(names)) != len(names):
        raise ValueError(f"wells are listed twice: {', '.join(wells)}")

    selected_rows, selected_columns = well_indices(plate.plate, names)
    percent = overlap * 100.0  # useq takes a percentage
    plan = plate.to_plan(
        selected_wells=(selected_rows, selected_columns),
        well_points_plan=useq.GridRowsColumns(
            rows=rows,
            columns=columns,
            fov_width=fov_um[0],
            fov_height=fov_um[1],
            overlap=(percent, percent),
        ),
    )
    geometries = {well: well_geometry(plan, well) for well in names}

    kept_in: dict[str, int] = {}
    fields: list[PlannedField] = []
    for position in plan.image_positions:
        name = str(position.name)
        # `<well>_<index:04d>`, a 1 x 1 grid included: the well is up to the last `_`.
        well = name.rsplit("_", 1)[0]
        if well not in geometries:
            raise ValueError(
                f"position {name!r} is not of a well of the plan ({', '.join(names)}): "
                "this useq names its positions differently"
            )
        if position.x is None or position.y is None:
            raise ValueError(
                f"position {name!r} has no x or y: this useq gives positions "
                "without coordinates"
            )
        x_um, y_um = float(position.x), float(position.y)
        if clip_to_well and not _touches(geometries[well], footprint_of(x_um, y_um)):
            continue
        index = kept_in.get(well, 0)
        kept_in[well] = index + 1
        fields.append(
            PlannedField(well=well, index=index, name=name, x_um=x_um, y_um=y_um)
        )

    empty = [well for well in names if well not in kept_in]
    if empty:
        noun = "well" if len(empty) == 1 else "wells"
        raise ValueError(
            f"no field of the grid touches {noun} {', '.join(empty)}: give rows and "
            "columns, a larger overlap, or clip_to_well=false"
        )
    return plan, tuple(fields)


def estimate_s(
    fields: Sequence[PlannedField], timings: TimingsResult, *, refocuses: int
) -> float:
    """Seconds to visit ``fields`` in order: each move and snap, and ``refocuses`` locks of the autofocus.

    The first field costs no move (the stage is taken to be there), and a move
    is the straight line between two fields in µm. A lock is charged **per
    well, not per field** (m3 §8, #416): the scan switches the autofocus off at
    each well's wall and the first field locks it again, so a continuous
    autofocus costs one ``refocus_ms`` at a well's entry and nothing at the
    fields that follow. Charging it at every field made three scans of the
    Ti2-E's session 5 read 2.7 to 3.3 times slower than they ran (#549), which
    is why the caller says how many locks the scan makes (``refocus_count``)
    instead of this function guessing. A refocus whose cost the timings do not
    know (``refocus_ms`` is ``None``) is left out and not guessed.

    The locks are added once, after the loop, so that ``refocuses=0`` is
    exactly the figure the others add to and the figure without them is never
    above the figure with them.

    Raises:
        ValueError: ``refocuses`` is negative.
    """
    if refocuses < 0:
        raise ValueError(f"refocuses must be at least 0, got {refocuses}")
    total_ms = 0.0
    previous: PlannedField | None = None
    for field in fields:
        if previous is not None:
            total_ms += timings.move_ms(
                math.hypot(field.x_um - previous.x_um, field.y_um - previous.y_um)
            )
        total_ms += timings.snap_ms
        previous = field
    if timings.refocus_ms is not None:
        total_ms += refocuses * timings.refocus_ms
    return total_ms / 1000.0


def refocus_count(fields: Sequence[PlannedField]) -> int:
    """The locks a scan of ``fields`` makes: one in each distinct well (m3 §7.4 D1).

    The scan groups the fields by well and locks the autofocus again at each
    group's first field, so the count is the wells and not the runs of the
    order or the fields.
    """
    return len({field.well for field in fields})


def estimate_assumed(locks: int) -> tuple[str, ...]:
    """The ``estimate:`` lines of a plan that has an estimate (m3 §7.4 D5).

    The first, when ``locks`` is above 0, says what a lock assumes: the layer
    cannot tell an autofocus that holds its lock from one that refocuses at each
    field (D4), so the estimate takes the first and says so. The last says what
    the figure leaves out, which is about a fifth of the real time and is not
    guessed (*measured beats assumed*). The page puts ``assumed: `` before each
    and shows it whole up to ``ASSUMED_LINE_CHARS``; the longest forms are
    152 and 155 characters (a test holds them to the bound, FM-205).
    """
    if locks > 0:
        noun = "1 lock" if locks == 1 else f"{locks} locks"
        return (
            "estimate: the autofocus holds its lock between fields and locks again "
            f"once in each well ({noun}); one that refocuses at each field would "
            "take longer",
            "estimate: the timed moves, locks and snaps; it leaves out each frame's "
            "write, a lock lost in a well, a channel change, the cells, the report "
            "and the export",
        )
    return (
        "estimate: the timed moves and snaps; it leaves out each frame's write, a "
        "channel change, the cells, the report and the export",
    )


def _touches(well: WellGeometry, frame: Sequence[Point]) -> bool:
    """Whether a frame's footprint meets the well: the shape comes from ``WellGeometry`` only."""
    if well.circular:
        return _disc_meets(well.center_um, well.size_um[0] / 2.0, frame)
    return _polygons_overlap(frame, _rectangle_of(well))


def _rectangle_of(well: WellGeometry) -> list[Point]:
    """The corners of a square well in stage axes: its half-sizes turned by the plate's rotation."""
    half_width, half_height = well.size_um[0] / 2.0, well.size_um[1] / 2.0
    angle = math.radians(well.rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    return [
        (
            well.center_um[0] + cos * u - sin * v,
            well.center_um[1] + sin * u + cos * v,
        )
        for u, v in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]


def _disc_meets(center: Point, radius: float, polygon: Sequence[Point]) -> bool:
    """Whether the distance from ``center`` to the filled ``polygon`` is below ``radius``.

    The distance is 0 when the centre is inside. A mirrored camera lists the
    corners of its footprint in the opposite turning sense, so the inside test
    accepts either.
    """
    if _inside(center, polygon):
        return radius > 0.0
    nearest = min(_distance_to_segment(center, a, b) for a, b in _edges(polygon))
    return nearest < radius


def _polygons_overlap(a: Sequence[Point], b: Sequence[Point]) -> bool:
    """Whether two convex polygons overlap, by the separating axis theorem.

    A polygon pair is apart when some edge normal of either separates their
    projections; touching is not apart. Both orientations of corners work.
    """
    for polygon in (a, b):
        for (px, py), (qx, qy) in _edges(polygon):
            normal_x, normal_y = qy - py, px - qx
            low_a, high_a = _project(a, normal_x, normal_y)
            low_b, high_b = _project(b, normal_x, normal_y)
            if high_a < low_b or high_b < low_a:
                return False
    return True


def _edges(polygon: Sequence[Point]) -> Iterator[tuple[Point, Point]]:
    for index, corner in enumerate(polygon):
        yield corner, polygon[(index + 1) % len(polygon)]


def _project(
    polygon: Sequence[Point], axis_x: float, axis_y: float
) -> tuple[float, float]:
    values = [x * axis_x + y * axis_y for x, y in polygon]
    return min(values), max(values)


def _inside(point: Point, polygon: Sequence[Point]) -> bool:
    """Whether a point is inside a convex polygon, whichever way its corners turn."""
    crosses = [
        (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
        for (ax, ay), (bx, by) in _edges(polygon)
    ]
    return all(c >= 0.0 for c in crosses) or all(c <= 0.0 for c in crosses)


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.dist(point, start)
    along = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    along = max(0.0, min(1.0, along))
    return math.dist(point, (start[0] + along * dx, start[1] + along * dy))
