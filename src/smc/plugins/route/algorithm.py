"""Which fields to image and in what order: the densest fields first, then the shortest path (m3 §11).

Pure arithmetic on stage µm: no stage, no camera, no turret. The plugin hands in
the camera's matrix and the frame's shape, and the cells; what a field is, is
decided here once.

Two things are easy to get wrong and invisible on a square frame (FM-73): a
field is the camera's **footprint**, a rectangle turned by the camera and
possibly mirrored, not a square of the field of view around a centre; and its
width and height are exchanged in the pixel offsets of the overlap rule, in the
membership test and in the window that counts cells. Everything that tests
whether a cell or a corner is in a field goes through ``pixel_of_stage`` and
``footprint``, so the sign convention stays in ``geometry.py``.

A numeric choice made on measurements, not on taste:

* **No matrix product on coordinate arrays.** ``(N, 2) @ (2, 2)`` on finite
  data raises ``RuntimeWarning: divide by zero encountered in matmul`` from
  N = 2000 up on numpy 2.2.6 (macOS arm64), though not on 1.26 or 2.5. The four
  entries of the inverse are written out instead.
* **Counting is per cell, over the lattice points near it**, not cells times
  candidates: 300 000 cells in a 96-well well took 12 s the second way and
  0.46 s the first, and the second grows with the well (FM-74).
* **The bounds are on floats first.** ``3200.0 // 5e-324`` is ``inf`` and
  ``int(inf)`` is an ``OverflowError``, so a size is refused before it becomes
  an integer.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from smc.plugins.geometry import Matrix2, footprint, pixel_of_stage
from smc.plugins.slots import Cell, WellGeometry

__all__ = [
    "ChosenField",
    "LatticeTooLarge",
    "TooMuchWork",
    "allowed",
    "choose_fields",
    "fov_of",
    "order_route",
    "path_length_um",
]

Point = tuple[float, float]

#: The candidate lattice of a well never holds more points than this. The lattice
#: is derived from the camera's reported pixel size, which a wrong profile entry can
#: make 1000 times too small (FM-121), and from a parameter a person types.
_MAX_LATTICE_POINTS = 250_000
#: Cell-by-lattice-point tests one call may make. Measured at the bound (760 000
#: cells, 2 809 window points each, numpy 2.5.4, an Apple M-series laptop): about
#: 125 million tests a second, so 17 s and a peak of 1 GB, more on a slower PC. A
#: refusal threshold and not a promise: nothing looks at a stop inside one well.
_MAX_TESTS = 2**31
#: Elements of one chunk of the counting (cells x window points), at about 94
#: bytes each: 2**18 elements are 25 MB, 2**22 are 392 MB, and the time is the same.
_WORK_CHUNK = 2**18
#: Two fields whose centres differ by this many pixels less than a frame are
#: overlapping; at exactly a frame they touch, and touching is allowed.
_TOUCH_PX = 1e-6
#: A reversal that shortens the path by less than this is not made: it ends 2-opt,
#: and a float round-off is not an improvement.
_IMPROVEMENT_UM = 1e-9


class LatticeTooLarge(ValueError):  # noqa: N818 (the names are the plan's of #194: a refusal, read as one)
    """The candidate lattice of a well would hold more than ``_MAX_LATTICE_POINTS`` points."""


class TooMuchWork(ValueError):  # noqa: N818 (see LatticeTooLarge)
    """Counting the cells against the candidates would take more than ``_MAX_TESTS`` tests."""


@dataclass(frozen=True, slots=True)
class ChosenField:
    """A field the greedy chose: its centre in stage µm and the cells it holds.

    ``counts`` has one entry per wanted label, zeros included, over the cells
    that no earlier field held. It is built as a ``dict`` and not mutated after.
    """

    x_um: float
    y_um: float
    counts: Mapping[str, int]


def allowed(
    x_um: float,
    y_um: float,
    well: WellGeometry,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    exclude_centre_um: float,
    wall_margin_um: float,
) -> bool:
    """Whether a field centred at ``(x_um, y_um)`` keeps to the well's margin and away from its centre.

    The test is on the **footprint**, not on the centre: a camera's frame is a
    rectangle turned by the camera, and a field whose centre is inside the well
    can still image the wall. Every corner must be inside the well shrunk by
    ``wall_margin_um`` (both shapes of well are convex, so the corners are
    enough), and, when ``exclude_centre_um > 0``, the filled footprint must stay
    at least that far from the well's centre: the distance from the centre to
    the polygon, which is 0 when the centre is inside it. Measuring to the
    corners alone would let a field whose edge crosses the excluded disc pass
    whenever its corners are far away (the measured mutant that survived).

    ``WellGeometry.contains`` is the only place that knows a well's shape.

    Raises:
        ValueError: ``wall_margin_um`` is negative or not finite (``contains``'
            refusal), or the matrix is not finite.
    """
    corners = footprint(x_um, y_um, matrix=matrix, image_shape_px=image_shape_px)
    if not all(well.contains(x, y, margin_um=wall_margin_um) for x, y in corners):
        return False
    if exclude_centre_um <= 0.0:
        return True
    return _distance_to_polygon(well.center_um, corners) >= exclude_centre_um


def fov_of(matrix: Matrix2, image_shape_px: tuple[int, int]) -> tuple[float, float]:
    """``(width, height)`` of the frame's footprint in stage axes, in µm.

    The extent of the four corners at the origin: for a camera turned by the
    stage, the bounding box of the rotated frame. The candidate lattice's step
    and the window that counts cells use it; the fields' own footprints stay the
    rotated rectangle, so the overlap rule is on pixel offsets, not on this.
    """
    corners = footprint(0.0, 0.0, matrix=matrix, image_shape_px=image_shape_px)
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


def choose_fields(
    cells: Sequence[Cell],
    well: WellGeometry,
    *,
    labels: Sequence[str],
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    n: int,
    min_cells: int,
    exclude_centre_um: float,
    wall_margin_um: float,
    step_um: float,
) -> tuple[ChosenField, ...]:
    """The fields of one well that hold the most wanted cells, greedily; at most ``n``, never overlapping.

    Candidate centres are the points of a square lattice of ``step_um`` over the
    well, centred on it, kept when ``allowed``, and numbered row by row from +Y
    down and then from -X up. Each round takes the candidate with the most
    wanted cells **that no earlier field holds** (ties: the lowest number), stops
    when that is below ``min_cells``, and removes the candidates whose footprint
    would overlap the chosen one: with the camera's turn the same for both
    fields, that is ``|d_col| < W`` and ``|d_row| < H`` in pixels. Fields that
    touch are allowed, so a cell on the edge two of them share is held by the
    first and counted once. Every cell is counted once, so the counts and what
    is left uncovered always add up to the wanted cells.

    The counts are made once, per cell over the lattice points near it, and
    only go down afterwards, by the cells a chosen field took: a candidate's
    count is an upper bound, and the one at the top is recounted exactly before
    it is chosen. That is the greedy of recounting every candidate every round,
    and it never chooses a field with fewer than ``min_cells`` new cells.

    Args:
        cells: Cells of any well and label; only those of ``well`` and of
            ``labels`` count.
        well: The well to choose in.
        labels: The labels that count; each is a key of every field's counts.
        matrix: The camera's matrix, µm per pixel.
        image_shape_px: ``(height, width)`` of the frame.
        n: The most fields to choose.
        min_cells: A field must hold at least this many new cells.
        exclude_centre_um: Radius of the disc at the well's centre no field
            may touch, 0 for none.
        wall_margin_um: Distance from the wall every corner keeps.
        step_um: The lattice's step, ``candidate_step_fraction * min(fov)``.

    Returns:
        The chosen fields in the order chosen; empty when no candidate is
        allowed or the well has no wanted cell.

    Raises:
        LatticeTooLarge: The lattice would hold more than 250 000 points.
        TooMuchWork: The count would make more than 2**31 tests.
        ValueError: ``n`` or ``min_cells`` is below 1, ``labels`` is empty,
            ``step_um`` is not finite and positive, or the matrix cannot be
            inverted (a field of no area).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")
    if min_cells < 1:
        raise ValueError(f"min_cells must be at least 1, got {min_cells!r}")
    if len(labels) == 0:
        raise ValueError("labels must name at least one label")
    if not (math.isfinite(step_um) and step_um > 0.0):
        raise ValueError(f"step_um must be finite and positive, got {step_um!r}")
    inverse = _inverse_of(matrix)
    fov_um = fov_of(matrix, image_shape_px)
    position_of = {label: k for k, label in enumerate(dict.fromkeys(labels))}
    wanted = [
        cell for cell in cells if cell.well == well.well and cell.label in position_of
    ]
    index, centres, origin = _lattice(
        well,
        step_um,
        matrix=matrix,
        image_shape_px=image_shape_px,
        exclude_centre_um=exclude_centre_um,
        wall_margin_um=wall_margin_um,
    )
    if len(centres) == 0 or not wanted:
        return ()

    height, width = image_shape_px
    half_px = (width / 2.0, height / 2.0)
    xy = np.array([(cell.x_um, cell.y_um) for cell in wanted], dtype=np.float64)
    which = np.array([position_of[cell.label] for cell in wanted], dtype=np.int64)
    counts = _candidate_counts(
        xy,
        index,
        centres,
        origin=origin,
        step_um=step_um,
        fov_um=fov_um,
        inverse=inverse,
        half_px=half_px,
    )
    alive = np.ones(len(centres), dtype=bool)
    held = np.zeros(len(xy), dtype=bool)
    chosen: list[ChosenField] = []
    while len(chosen) < n:
        # `argmax` takes the first maximum, which is the lowest candidate number.
        scores = np.where(alive, counts, -1)
        best = int(scores.argmax())
        if scores[best] < min_cells:
            break
        x, y = float(centres[best, 0]), float(centres[best, 1])
        d_col, d_row = _pixel_offsets(xy[:, 0] - x, xy[:, 1] - y, inverse)
        inside = (np.abs(d_col) <= half_px[0]) & (np.abs(d_row) <= half_px[1])
        fresh = inside & ~held
        new = int(fresh.sum())
        if new < counts[best]:
            # A cell on the edge two touching fields share was held by the first:
            # the bound comes down and the round starts again, choosing nothing.
            counts[best] = new
            continue
        tallies = np.bincount(which[fresh], minlength=len(position_of))
        chosen.append(
            ChosenField(
                x, y, {label: int(tallies[k]) for label, k in position_of.items()}
            )
        )
        held |= fresh
        o_col, o_row = _pixel_offsets(centres[:, 0] - x, centres[:, 1] - y, inverse)
        alive &= ~(
            (np.abs(o_col) < width - _TOUCH_PX) & (np.abs(o_row) < height - _TOUCH_PX)
        )
    return tuple(chosen)


def order_route(points: Sequence[Point], *, start: int) -> list[int]:
    """The order in which to visit ``points``: nearest neighbour from ``start``, then 2-opt.

    An open path whose first point stays ``start``. The cost is the Euclidean
    distance: the timing model is an overhead per move plus a rate per µm, so
    the shortest path is the fastest under it. Whether the Ti2-E's stage is
    nearer to Chebyshev (both axes at once) is not measured (#198), so no
    optimality and no time is claimed here. 2-opt is a heuristic: it never
    leaves the path longer than nearest neighbour's, and it is not the optimum.

    Ties go to the lowest index, both in the nearest neighbour and in the
    reversal that 2-opt applies, so the same points give the same route.

    Returns:
        The indices into ``points``, each once; ``[]`` for no point.

    Raises:
        ValueError: ``start`` is not an index of ``points``, or a coordinate is
            not finite (``nan`` would make every distance compare false and
            end a path anywhere).
    """
    if len(points) == 0:
        return []
    coordinates = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(coordinates).all():
        raise ValueError("every point must have finite coordinates")
    n = len(coordinates)
    if not (0 <= start < n):
        raise ValueError(f"start must be an index of the {n} points, got {start!r}")

    order = _nearest_neighbour(coordinates, start)
    _two_opt(coordinates, order)
    return order


def path_length_um(points: Sequence[Point], order: Sequence[int]) -> float:
    """The length of the open path that visits ``points`` in ``order``, in µm.

    0 for fewer than two stops. A leg is a straight line, as ``estimate_s`` of
    the scan plan prices it.
    """
    return sum(math.dist(points[a], points[b]) for a, b in itertools.pairwise(order))


def _nearest_neighbour(coordinates: npt.NDArray[np.float64], start: int) -> list[int]:
    """Greedy path from ``start``; the points left stay in ascending order, so ``argmin`` takes the lowest index on a tie."""
    left = np.delete(np.arange(len(coordinates)), start)
    order = [start]
    here = coordinates[start]
    while len(left):
        distances = np.hypot(
            coordinates[left, 0] - here[0], coordinates[left, 1] - here[1]
        )
        position = int(distances.argmin())
        chosen = int(left[position])
        order.append(chosen)
        left = np.delete(left, position)
        here = coordinates[chosen]
    return order


def _two_opt(coordinates: npt.NDArray[np.float64], order: list[int]) -> None:
    """Uncross the open path in place, the first point fixed; repeated passes until one changes nothing.

    Reversing ``q[i..j]`` replaces the edges ``(q[i-1], q[i])`` and
    ``(q[j], q[j+1])`` by ``(q[i-1], q[j])`` and ``(q[i], q[j+1])``; at the end
    of an open path there is no second edge. For each ``i`` every ``j`` is
    priced at once and the most negative change is applied. Each applied
    reversal shortens the path by more than ``_IMPROVEMENT_UM``, so it ends.
    """
    n = len(order)
    if n < 3:
        return
    q = coordinates[order].copy()
    changed = True
    while changed:
        changed = False
        for i in range(1, n - 1):
            js = np.arange(i + 1, n)
            before = q[i - 1]
            delta = np.hypot(q[js, 0] - before[0], q[js, 1] - before[1]) - math.hypot(
                q[i, 0] - before[0], q[i, 1] - before[1]
            )
            # The second edge exists for every j but the last.
            inner = js[:-1]
            delta[:-1] += np.hypot(
                q[i, 0] - q[inner + 1, 0], q[i, 1] - q[inner + 1, 1]
            ) - np.hypot(q[inner, 0] - q[inner + 1, 0], q[inner, 1] - q[inner + 1, 1])
            best = int(delta.argmin())
            if delta[best] < -_IMPROVEMENT_UM:
                j = int(js[best])
                q[i : j + 1] = q[i : j + 1][::-1].copy()
                order[i : j + 1] = order[i : j + 1][::-1]
                changed = True


def _distance_to_polygon(point: Point, corners: Sequence[Point]) -> float:
    """The distance from ``point`` to the filled convex quadrilateral ``corners``; 0 when it is inside.

    A mirrored camera lists the corners of its footprint the other way round, so
    the inside test accepts all four cross products ``>= 0`` or all ``<= 0``.
    Outside, it is the smallest distance to the four edges.
    """
    edges = [(corners[k], corners[(k + 1) % len(corners)]) for k in range(len(corners))]
    crosses = [
        (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
        for (ax, ay), (bx, by) in edges
    ]
    if all(c >= 0.0 for c in crosses) or all(c <= 0.0 for c in crosses):
        return 0.0
    return min(_distance_to_segment(point, a, b) for a, b in edges)


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.dist(point, start)
    along = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    along = max(0.0, min(1.0, along))
    return math.dist(point, (start[0] + along * dx, start[1] + along * dy))


_Inverse = tuple[tuple[float, float], tuple[float, float]]


def _inverse_of(matrix: Matrix2) -> _Inverse:
    """``((a, b), (c, d))`` with ``d_col = a dx + b dy`` and ``d_row = c dx + d dy`` for a stage offset.

    Read from ``pixel_of_stage`` at one µm along each axis, and not re-derived
    from the matrix, so that the sign convention stays in ``geometry.py`` and a
    singular matrix is refused by its ``ValueError``. The frame asked about is
    **one pixel**, whose centre is ``(0, 0)``: with the real frame, its centre is
    added to the offset and taken off again, which costs a few ulps (``1 / 1.25``
    came back as 0.8000000000000007 on a 30-pixel frame, a pixel offset of
    15.000000000000014 for a cell exactly on a field's edge, and the cell was
    dropped), and the offset does not depend on the frame.
    """

    def pixel(dx: float, dy: float) -> tuple[float, float]:
        return pixel_of_stage(0.0, 0.0, dx, dy, matrix=matrix, image_shape_px=(1, 1))

    col_x, row_x = pixel(1.0, 0.0)
    col_y, row_y = pixel(0.0, 1.0)
    return ((col_x, col_y), (row_x, row_y))


def _pixel_offsets(
    dx: npt.NDArray[np.float64], dy: npt.NDArray[np.float64], inverse: _Inverse
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(d_col, d_row)`` of stage offsets, written out and not as a matrix product (see the module's notes)."""
    (a, b), (c, d) = inverse
    return a * dx + b * dy, c * dx + d * dy


def _lattice(
    well: WellGeometry,
    step_um: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    exclude_centre_um: float,
    wall_margin_um: float,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], Point]:
    """The candidate lattice of a well: ``(index, centres, origin)``.

    Centred on the well's centre: point ``(c, r)`` is at
    ``x = cx + (c - nx) * step``, ``y = cy + (ny - r) * step`` for ``c`` in
    ``0..2 nx`` and ``r`` in ``0..2 ny``, over the well's bounding box (for a
    square well, of the rectangle turned by the plate's rotation). ``centres``
    holds the points that are ``allowed``, numbered row by row from ``r = 0``
    (+Y) down and then from ``c = 0`` (-X) up; ``index`` has their number at
    their place and -1 elsewhere; ``origin`` is the point ``(0, 0)``.

    The size is refused on floats, before anything is allocated and before
    ``int()``: ``3200.0 // 5e-324`` is ``inf``, and ``nan`` fails the ``<=`` it
    is written as (FM-74).

    Raises:
        LatticeTooLarge: More than ``_MAX_LATTICE_POINTS`` points, or a step
            that is not positive (no step is as fine as one of 0).
    """
    width, height = well.size_um
    if well.circular:
        half_x = half_y = width / 2.0
    else:
        angle = math.radians(well.rotation_deg)
        cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
        half_x = cos * width / 2.0 + sin * height / 2.0
        half_y = sin * width / 2.0 + cos * height / 2.0
    nx_f = half_x / step_um if step_um > 0.0 else math.inf
    ny_f = half_y / step_um if step_um > 0.0 else math.inf
    # A message and not an exception kept in a local and raised twice: its
    # traceback would reach this frame, which holds it (FM-98).
    too_large = (
        f"the candidate lattice of {well.well} would hold more than "
        f"{_MAX_LATTICE_POINTS} fields at a step of {step_um:g} µm"
    )
    if not (nx_f <= _MAX_LATTICE_POINTS and ny_f <= _MAX_LATTICE_POINTS):
        raise LatticeTooLarge(too_large)
    nx, ny = int(nx_f), int(ny_f)
    if (2 * nx + 1) * (2 * ny + 1) > _MAX_LATTICE_POINTS:
        raise LatticeTooLarge(too_large)

    cx, cy = well.center_um
    index = np.full((2 * ny + 1, 2 * nx + 1), -1, dtype=np.int32)
    centres: list[Point] = []
    for r in range(2 * ny + 1):
        y = cy + (ny - r) * step_um
        for c in range(2 * nx + 1):
            x = cx + (c - nx) * step_um
            if allowed(
                x,
                y,
                well,
                matrix=matrix,
                image_shape_px=image_shape_px,
                exclude_centre_um=exclude_centre_um,
                wall_margin_um=wall_margin_um,
            ):
                index[r, c] = len(centres)
                centres.append((x, y))
    return (
        index,
        np.array(centres, dtype=np.float64).reshape(-1, 2),
        (cx - nx * step_um, cy + ny * step_um),
    )


def _candidate_counts(
    wanted_xy: npt.NDArray[np.float64],
    index: npt.NDArray[np.int32],
    centres: npt.NDArray[np.float64],
    *,
    origin: Point,
    step_um: float,
    fov_um: tuple[float, float],
    inverse: _Inverse,
    half_px: tuple[float, float],
) -> npt.NDArray[np.int64]:
    """How many wanted cells each candidate's frame holds, one count per candidate.

    For each cell, only the lattice points that can hold it are looked at: the
    window of ``kc x kr`` points whose centres are within the field's extent of
    it, found by arithmetic on the lattice, and each is then tested exactly in
    pixels (``|d_col| <= W/2`` and ``|d_row| <= H/2``), so a turned camera's
    bounding box over-selects and the test decides. A cell is tested against
    its own few candidates and not against all of them, so the work grows with
    the cells and not with the well. Cells go in chunks of ``_WORK_CHUNK``
    elements so that the temporary arrays stay at 25 MB.

    Raises:
        TooMuchWork: The tests would number more than ``_MAX_TESTS``. Checked on
            floats before any ``int()``, and before the first chunk.
    """
    counts = np.zeros(len(centres), dtype=np.int64)
    if len(wanted_xy) == 0 or len(centres) == 0:
        return counts
    # `//` on floats: inf for a tiny step, which the bound below refuses.
    kc_f = fov_um[0] // step_um + 3.0
    kr_f = fov_um[1] // step_um + 3.0
    per_cell = kc_f * kr_f
    if not len(wanted_xy) * per_cell <= _MAX_TESTS:
        raise TooMuchWork(
            f"{len(wanted_xy)} cells × {per_cell:.0f} lattice points per cell "  # noqa: RUF001
            f"would be tested, over {_MAX_TESTS}"
        )
    kc, kr = int(kc_f), int(kr_f)

    rows_n, cols_n = index.shape
    origin_x, origin_y = origin
    half_fov_x, half_fov_y = fov_um[0] / 2.0, fov_um[1] / 2.0
    # A cell that no field of the lattice can reach is not looked up (and its
    # window could not be turned into integers if it were far away). One step of
    # slack on each side keeps a rounding of ``fov_um`` from dropping an edge cell.
    reach = step_um + half_fov_x, step_um + half_fov_y
    near = (
        (wanted_xy[:, 0] >= origin_x - reach[0])
        & (wanted_xy[:, 0] <= origin_x + (cols_n - 1) * step_um + reach[0])
        & (wanted_xy[:, 1] <= origin_y + reach[1])
        & (wanted_xy[:, 1] >= origin_y - (rows_n - 1) * step_um - reach[1])
    )
    xs, ys = wanted_xy[near, 0], wanted_xy[near, 1]
    chunk = max(1, _WORK_CHUNK // (kc * kr))
    along_columns = np.arange(kc)
    along_rows = np.arange(kr)
    for first in range(0, len(xs), chunk):
        x = xs[first : first + chunk]
        y = ys[first : first + chunk]
        # The first lattice column and row that can hold the cell; the window is
        # one point wider than the field on each side, never narrower.
        col_lo = np.floor((x - origin_x - half_fov_x) / step_um).astype(np.int64)
        row_lo = np.floor((origin_y - (y + half_fov_y)) / step_um).astype(np.int64)
        cols = col_lo[:, None] + along_columns
        rows = row_lo[:, None] + along_rows
        inside = ((rows >= 0) & (rows < rows_n))[:, :, None] & (
            (cols >= 0) & (cols < cols_n)
        )[:, None, :]
        number = index[
            np.clip(rows, 0, rows_n - 1)[:, :, None],
            np.clip(cols, 0, cols_n - 1)[:, None, :],
        ]
        inside &= number >= 0
        # A window point without a candidate reads the last one, and is masked.
        d_col, d_row = _pixel_offsets(
            x[:, None, None] - centres[number, 0],
            y[:, None, None] - centres[number, 1],
            inverse,
        )
        inside &= (np.abs(d_col) <= half_px[0]) & (np.abs(d_row) <= half_px[1])
        counts += np.bincount(number[inside], minlength=len(centres))
    return counts
