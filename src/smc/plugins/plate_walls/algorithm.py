"""Where a well's wall is in a frame, and the circle of a well from its wall points (m3 §17.3).

Pure: numpy only, no device, no file. The wall is the fall from a well's inside
to the plastic that a line of the frame meets going outwards, at the half level
of that fall; the well's centre and size are the circle fitted to the wall
points of its four sides. What it decides, because a wall found in the wrong
place gives a plate registered a whole well off with no error to say so:

* **The wall is the fall to the plastic, not the first fall below the midpoint.**
  The meniscus darkens the inside of a well towards its wall: a shadow joined to
  the wall, or a dark ridge on the inner border of a grey band, 0.76 to 1.67 mm
  inside the plastic edge at the 4x, below the frame's midpoint before the
  plastic is (B5, #409). The plastic is the run below the midpoint that reaches
  the line's end or is long enough to be a gap between wells; a shadow or a
  ridge ends inside the frame and is shorter.
* **The camera's turn decides where to look.** The outward direction is a stage
  direction, turned into pixels by the camera matrix. A wrong matrix points the
  lines into the plastic, where every line starts dark and none gives a point:
  a refusal, not a centre one well-width off (#327's failure).
* **A line must start inside a well.** A neighbour's near wall is a rise going
  outwards and a frame on the plastic starts dark, so neither is read as ours.
* **Two classes or none.** A frame inside a well (texture, shading) has no dark
  class that is a quarter as bright as the inside, and finds no wall.

No matrix product anywhere (``@``, ``.dot``, ``.matmul``: FM-131): the 2 x 2
products and the inverse are written out.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile

from smc.plugins.geometry import Matrix2, pixel_of_stage
from smc.plugins.slots import WallSide

__all__ = [
    "BAND_PX",
    "DEFAULT_SEARCH_FRACTION",
    "MAX_DARK_RATIO",
    "MAX_FRAMES",
    "MAX_RMS_FRACTION",
    "MIN_CLASS_FRACTION",
    "MIN_FRAME_SIDE_PX",
    "MIN_LINE_FRACTION",
    "MIN_PLASTIC_UM",
    "OUTWARD",
    "SHOULDER_PX",
    "SIDES",
    "STEP_FRACTION",
    "WALL_LEVEL",
    "CircleFit",
    "WallEdge",
    "along_um",
    "encode_wall_frame",
    "find_wall",
    "fit_circle",
    "walk_offsets_um",
    "wall_points_um",
]

#: A line is the mean of this many rows (or columns) of the frame: noise falls
#: with the root of it, and a wall that is not parallel to the band's axis is
#: blurred over the band's width.
BAND_PX = 8
#: Each of the two classes holds at least this fraction of the pixels, or the
#: frame has one class only: all inside a well, or all plastic.
MIN_CLASS_FRACTION = 0.02
#: The dark class's mean over the bright class's, at most this: the plastic is
#: far darker than a well in brightfield (0.057-0.093 measured), shading inside
#: a well is not (0.72 on a texture, 0.82 on a ramp).
MAX_DARK_RATIO = 0.5
#: At least this fraction of the lines must give a point, or the edge is a
#: corner of the frame, a bubble or a scratch and not a wall.
MIN_LINE_FRACTION = 0.5
#: The walk's step as a fraction of the frame's smaller side: the inscribed
#: circle of the frame, so that it holds whatever the camera's turn, and
#: consecutive frames overlap by a quarter.
STEP_FRACTION = 0.75
#: How far the wall may lie from where the recorded centre puts it, as a
#: fraction of the well's radius.
DEFAULT_SEARCH_FRACTION = 0.5
#: The circle's rms may be this fraction of the radius: the well's own shape
#: is 0.5 % (41.9 µm of 7650, measured), a bubble or a label is more.
MAX_RMS_FRACTION = 0.02
#: A frame whose smaller side is under this holds too little of a wall to tell
#: it from the frame's own corner.
MIN_FRAME_SIDE_PX = 128
#: The walk's bound, on the frames of the whole run (FM-74): the same bound
#: applies to what is allocated, a walk's offsets, as to what is moved.
MAX_FRAMES = 600
#: The plastic is reached where a line of the plastic's run first falls below
#: this fraction of the way from the plastic's level to the inside's. The
#: meniscus's shadows reach 0.16 of the way at their darkest, next to the wall,
#: and the plastic sits at about 0.01 (B5 at the 4x, #409, §17.1).
WALL_LEVEL = 0.1
#: A run below the midpoint is the plastic when it reaches the line's end or is
#: at least this long, in µm. A shadow or a ridge of the meniscus that ends
#: inside the frame was 972 µm at most (B5's -X, 4x); the plastic between our
#: wall and the neighbour's is a gap between wells, 1 900 µm at least (48-well).
#: In µm, never in pixels: the samples are turned into pixels by the matrix.
MIN_PLASTIC_UM = 1400.0
#: How far before the plastic's first sample the wall's own shoulder is looked
#: for, in samples: a wall that crosses a band at up to 45 degrees spreads its
#: fall over BAND_PX samples, so twice that reaches the level it falls from.
SHOULDER_PX = 2 * BAND_PX
#: The sides of a well, as stage directions, in the order they are measured.
SIDES: tuple[WallSide, ...] = ("+X", "+Y", "-X", "-Y")
#: The unit vector from a well's centre to each side's wall, in stage axes.
OUTWARD: dict[WallSide, tuple[float, float]] = {
    "+X": (1.0, 0.0),
    "+Y": (0.0, 1.0),
    "-X": (-1.0, 0.0),
    "-Y": (0.0, -1.0),
}

#: The factor that makes a median absolute deviation an estimate of a
#: Gaussian's standard deviation.
_MAD_TO_SIGMA = 1.4826
#: A point is an outlier of the circle beyond this many robust sigmas.
_TRIM_SIGMAS = 5.0
#: The trims after the first fit: one is not enough when the first fit is
#: pulled towards a bad arc, and a third changed nothing in the measured cases.
_TRIMS = 2
#: Singular values below this fraction of the largest are zero: the points are
#: on a line (the points are scaled to the unit square first, so a real arc is
#: far above it).
_RANK_RCOND = 1e-9


@dataclass(frozen=True, slots=True)
class WallEdge:
    """The wall points of one frame, in pixels, and what they were found from.

    ``points_px`` is ``(n, 2)`` of ``(col, row)``, at most one point per line.
    ``n_lines`` counts the lines looked at, ``threshold`` is the level that
    separates the wall's two sides and ``dark_ratio`` the dark class's mean over
    the bright class's.
    """

    points_px: npt.NDArray[np.float64]
    n_lines: int
    threshold: float
    dark_ratio: float


@dataclass(frozen=True, slots=True)
class CircleFit:
    """A circle fitted to wall points, in the stage's µm.

    ``n_points`` is the points given and ``n_used`` the points kept after the
    trims; ``rms_um`` is over the kept points.
    """

    center_xy_um: tuple[float, float]
    diameter_um: float
    rms_um: float
    n_points: int
    n_used: int


def _outward_pixels(
    outward_xy: tuple[float, float], matrix: Matrix2
) -> tuple[float, float]:
    """The outward direction as ``(d_col, d_row)``: ``M`` inverted, written out.

    ``pixel_of_stage`` on a one-pixel frame, whose centre is ``(0, 0)``, is
    ``M`` inverted applied to the offset, with the geometry's own refusal of a
    singular matrix (FM-125).

    Raises:
        ValueError: ``outward_xy`` is not finite or is the zero vector; the
            matrix is singular or not finite.
    """
    ox, oy = outward_xy
    if not (math.isfinite(ox) and math.isfinite(oy) and (ox != 0.0 or oy != 0.0)):
        raise ValueError(
            f"outward_xy must be a finite direction that is not zero, got {outward_xy!r}"
        )
    return pixel_of_stage(0.0, 0.0, ox, oy, matrix=matrix, image_shape_px=(1, 1))


def _plastic_run(below: npt.NDArray[np.bool_], min_run: int) -> tuple[int, int] | None:
    """The line's plastic: ``[start, end)`` of the first run of ``below`` that is one, else ``None``.

    A run is the plastic when it reaches the line's end or is at least
    ``min_run`` samples long. A shadow or a ridge of the meniscus is a run that
    ends inside the frame and is shorter, so it is skipped, and the plastic
    behind it is found. Two runs that fewer than ``BAND_PX`` samples above the
    threshold separate are one: that is noise on the plastic's own edge.
    """
    padded = np.concatenate(([False], below, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1]).tolist()
    runs: list[list[int]] = []
    for start, end in zip(change[0::2], change[1::2], strict=True):
        if runs and start - runs[-1][1] < BAND_PX:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    for start, end in runs:
        if end == below.shape[0] or end - start >= min_run:
            return start, end
    return None


def _wall_in_line(
    line: npt.NDArray[np.float64],
    below: npt.NDArray[np.bool_],
    deep: float,
    plastic: float,
    min_run: int,
) -> float | None:
    """Where the wall is along one line, in samples, or ``None``: the half level of the fall into the plastic.

    ``at`` is the first sample of the plastic's run below ``deep``. The edge is
    not the frame's midpoint but the half level of the wall's own fall, between
    the plastic and the shoulder it falls from (the largest of the
    ``SHOULDER_PX`` samples before ``at``): behind a shadow the midpoint is
    crossed by the shadow, 0.76 to 1.67 mm early at the 4x (#409). On a clean
    step the shoulder is the inside and this is the half-intensity edge. The
    inward scan stops after ``SHOULDER_PX`` samples whatever the line holds, so
    the work of a line stays bounded by the frame (FM-74).
    """
    run = _plastic_run(below, min_run)
    if run is None:
        return None
    deeper = np.flatnonzero(line[run[0] : run[1]] < deep)
    if deeper.size == 0:
        return None
    at = run[0] + int(deeper[0])
    shoulder = float(line[max(0, at - SHOULDER_PX) : at].max())
    local = (plastic + shoulder) / 2.0
    k = at
    while k > 1 and at - k < SHOULDER_PX and line[k - 1] < local:
        k -= 1
    before = float(line[k - 1])
    span = before - float(line[k])
    # Between the sample at or above the half level and the next one; a span that
    # is not positive has no crossing to interpolate and is put on `k`.
    where = (before - local) / span if span > 0.0 else 1.0
    return (k - 1) + min(max(where, 0.0), 1.0)


def find_wall(
    frame: npt.NDArray[Any], *, outward_xy: tuple[float, float], matrix: Matrix2
) -> WallEdge | None:
    """A well's wall in one frame: the fall from the inside to the plastic, looking along ``outward_xy``.

    Brightfield, measured on the Ti2-E (§17.1): the inside of a well is
    brighter than the plastic, with no parameter for the other polarity. The
    wall is the fall into the plastic, a run below the frame's midpoint that
    reaches the line's end or is at least ``MIN_PLASTIC_UM`` long, at the half
    level of that fall. It is not the first fall below the midpoint: the
    meniscus of a well (a shadow joined to the wall, a dark ridge) crosses the
    midpoint 0.76 to 1.67 mm inside the plastic at the 4x, and the walk that
    took it fitted B5 a circle of 13 055 µm for 15 600 (#409). ``threshold`` is
    still the midpoint that splits the frame's two classes and the runs.
    ``None`` is a frame with no wall in it: one class only, a dark class that is
    not dark enough, or too few lines that start inside a well and fall to the
    plastic. Never a guess.

    The frame is read as ``float64`` and never cast to an integer (FM-75).

    Raises:
        ValueError: The frame is not 2-D or has no pixel; ``outward_xy`` is not
            a direction; the matrix cannot be inverted.
    """
    image = np.asarray(frame)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"a frame must be 2-D with pixels, got shape {image.shape}")
    height, width = image.shape
    v_col, v_row = _outward_pixels(outward_xy, matrix)
    (m00, m01), (m10, m11) = matrix

    pixels = image.astype(np.float64)
    low, high = np.percentile(pixels, (2.0, 98.0))
    threshold = float((low + high) / 2.0)
    if not math.isfinite(threshold):
        return None
    dark = pixels < threshold
    n_dark = int(dark.sum())
    fraction = n_dark / int(pixels.size)
    if fraction < MIN_CLASS_FRACTION or fraction > 1.0 - MIN_CLASS_FRACTION:
        return None
    bright_mean = float(pixels[~dark].mean())
    if not bright_mean > 0.0:
        return None
    dark_ratio = float(pixels[dark].mean()) / bright_mean
    if dark_ratio > MAX_DARK_RATIO:
        return None

    along_columns = abs(v_col) >= abs(v_row)
    n_lines = (height if along_columns else width) // BAND_PX
    length = width if along_columns else height
    if n_lines == 0 or length <= BAND_PX:
        return None
    if along_columns:
        forward = v_col > 0.0
        lines = (
            pixels[: n_lines * BAND_PX].reshape(n_lines, BAND_PX, width).mean(axis=1)
        )
    else:
        forward = v_row > 0.0
        lines = (
            pixels[:, : n_lines * BAND_PX]
            .reshape(height, n_lines, BAND_PX)
            .mean(axis=2)
            .T
        )
    if not forward:
        lines = lines[:, ::-1]

    # The plastic's and the inside's levels are read from the band means, whose
    # noise is lower than the pixels': the pixels' 2nd percentile fell below the
    # plastic under noise of 600 counts.
    plastic, inside = (float(v) for v in np.percentile(lines, (2.0, 98.0)))
    if not (math.isfinite(plastic) and math.isfinite(inside) and inside > plastic):
        return None
    deep = plastic + WALL_LEVEL * (inside - plastic)
    # The plastic's run is at least this many samples, from µm: a pixel count
    # would be right on one frame shape and one pixel size only (FM-126). A step
    # too small to count (an overflow, not a finite number) asks for a run that
    # only one reaching the line's end can be.
    step_um = math.hypot(m00, m10) if along_columns else math.hypot(m01, m11)
    needed = MIN_PLASTIC_UM / step_um if step_um > 0.0 else math.inf
    min_run = math.ceil(needed) if needed <= length else length + 1

    starts_inside = lines[:, :BAND_PX].mean(axis=1) >= threshold
    below = lines < threshold
    # The runs start after the first sample, as the first fall always did.
    below[:, 0] = False
    kept: list[int] = []
    positions: list[float] = []
    for index in np.nonzero(starts_inside)[0].tolist():
        position = _wall_in_line(lines[index], below[index], deep, plastic, min_run)
        if position is not None:
            kept.append(index)
            positions.append(position)
    if len(kept) < MIN_LINE_FRACTION * n_lines:
        return None

    along = np.asarray(positions, dtype=np.float64)
    if not forward:
        along = (length - 1) - along
    middle = np.asarray(kept, dtype=np.float64) * BAND_PX + (BAND_PX - 1) / 2.0
    points = (
        np.column_stack((along, middle))
        if along_columns
        else np.column_stack((middle, along))
    )
    return WallEdge(
        points_px=points.astype(np.float64),
        n_lines=int(n_lines),
        threshold=threshold,
        dark_ratio=dark_ratio,
    )


def wall_points_um(
    points_px: npt.NDArray[np.float64],
    x_um: float,
    y_um: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """The stage points of ``points_px``, a frame taken at ``(x_um, y_um)``: ``stage_of_pixel`` over arrays.

    The centre is ``((W - 1) / 2, (H - 1) / 2)``, as ``smc.plugins.geometry``
    has it, and the 2 x 2 product is written out (FM-131).

    Raises:
        ValueError: ``points_px`` is not ``(n, 2)``.
    """
    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (n, 2) of (col, row), got {points.shape}")
    height, width = image_shape_px
    d_col = points[:, 0] - (width - 1) / 2.0
    d_row = points[:, 1] - (height - 1) / 2.0
    (m00, m01), (m10, m11) = matrix
    return np.column_stack(
        (x_um + m00 * d_col + m01 * d_row, y_um + m10 * d_col + m11 * d_row)
    )


def along_um(
    points_um: npt.NDArray[np.float64],
    centre_xy_um: tuple[float, float],
    outward_xy: tuple[float, float],
) -> float:
    """How far out a wall is from the recorded centre on one side: the median projection of its points.

    The median, so that a stray point of a bubble or a scratch does not move the
    walk's next frame. ``outward_xy`` is a unit vector.

    Raises:
        ValueError: ``points_um`` is empty or not ``(n, 2)``.
    """
    points = np.asarray(points_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(f"points must be (n, 2) with n > 0, got {points.shape}")
    projection = (points[:, 0] - centre_xy_um[0]) * outward_xy[0] + (
        points[:, 1] - centre_xy_um[1]
    ) * outward_xy[1]
    return float(np.median(projection))


def walk_offsets_um(
    radius_um: float, search_um: float, step_um: float
) -> tuple[float, ...]:
    """The distances from the recorded centre at which frames are taken until a wall is found.

    ``r``, then ``r + s``, ``r - s``, ``r + 2s``, ``r - 2s`` and so on, ``k``
    up to ``floor(search_um / step_um)``: the nominal wall first, then the
    nearer and the further side by side, because a recorded centre is as likely
    to be off one way as the other and walking from the inside outwards took
    three times as many frames (§17.1). The length is ``1 + 2 * floor(search_um
    / step_um)``, bounded by ``MAX_FRAMES`` before anything is built (FM-74).

    Raises:
        ValueError: An argument is not finite and positive; the walk would be
            longer than ``MAX_FRAMES``.
    """
    for name, value in (
        ("radius_um", radius_um),
        ("search_um", search_um),
        ("step_um", step_um),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    ratio = search_um / step_um
    # `floor` of an infinite ratio raises OverflowError, so the ratio is judged first.
    if not math.isfinite(ratio) or 1 + 2 * math.floor(ratio) > MAX_FRAMES:
        raise ValueError(
            f"a search of {search_um:g} µm in steps of {step_um:g} µm is more than "
            f"{MAX_FRAMES} frames for one wall: use a smaller search or a larger step"
        )
    offsets = [radius_um]
    for k in range(1, math.floor(ratio) + 1):
        offsets.append(radius_um + k * step_um)
        offsets.append(radius_um - k * step_um)
    return tuple(offsets)


def _kasa(points: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """The algebraic circle through ``points``: ``(x, y, radius)``; ``x² + y² = 2ax + 2by + c``.

    The points are centred and scaled to the unit square first: a well's
    coordinates are 1e4 µm and their squares 1e8, so that the system is well
    conditioned for an arc and singular for a line, and ``_RANK_RCOND`` tells
    the two apart.
    """
    centre = points.mean(axis=0)
    shifted = points - centre
    scale = float(np.abs(shifted).max())
    if not scale > 0.0:
        raise ValueError("the points are all one point")
    u = shifted[:, 0] / scale
    v = shifted[:, 1] / scale
    design = np.column_stack((2.0 * u, 2.0 * v, np.ones_like(u)))
    solution, _, rank, _ = np.linalg.lstsq(design, u * u + v * v, rcond=_RANK_RCOND)
    if int(rank) < 3:
        raise ValueError("the points lie on a line, which is not a circle")
    a, b, c = (float(s) for s in solution)
    squared = c + a * a + b * b
    if not (math.isfinite(squared) and squared > 0.0):
        raise ValueError("the points fit no circle of a finite, positive radius")
    return (
        float(centre[0]) + scale * a,
        float(centre[1]) + scale * b,
        scale * math.sqrt(squared),
    )


def fit_circle(points_um: npt.NDArray[np.float64], *, floor_um: float) -> CircleFit:
    """A circle fitted to wall points: the well's centre and diameter, robust to a bad arc.

    The algebraic (Kåsa) fit, then twice: the residuals ``|p - centre| -
    radius`` of all points, the points within ``max(5 * 1.4826 * MAD,
    floor_um)`` of the circle kept (``MAD`` of the points kept until then), and
    the fit again on them. The floor is about a pixel or two: on a frame with
    little noise the MAD is tiny, and a bound of the MAD alone would drop the
    well's own shape. A bubble's arc or a label's edge is dropped; the number
    kept is in the result for the caller to judge.

    Raises:
        ValueError: Fewer than 3 points; points not ``(n, 2)`` or not finite;
            ``floor_um`` not finite or negative; points on a line; fewer than 3
            points kept; a radius that is not finite and positive.
    """
    points = np.asarray(points_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (n, 2), got {points.shape}")
    if not bool(np.isfinite(points).all()):
        raise ValueError("the points must be finite")
    if not (math.isfinite(floor_um) and floor_um >= 0.0):
        raise ValueError(f"floor_um must be finite and not negative, got {floor_um!r}")
    if points.shape[0] < 3:
        raise ValueError(f"a circle needs 3 points, got {points.shape[0]}")

    keep = np.ones(points.shape[0], dtype=np.bool_)
    x, y, radius = _kasa(points)
    for _ in range(_TRIMS):
        residual = np.hypot(points[:, 0] - x, points[:, 1] - y) - radius
        kept = residual[keep]
        mad = float(np.median(np.abs(kept - np.median(kept))))
        keep = np.abs(residual) <= max(_TRIM_SIGMAS * _MAD_TO_SIGMA * mad, floor_um)
        if int(keep.sum()) < 3:
            raise ValueError(
                f"only {int(keep.sum())} of {points.shape[0]} points lie near a circle"
            )
        x, y, radius = _kasa(points[keep])
    residual = np.hypot(points[keep, 0] - x, points[keep, 1] - y) - radius
    return CircleFit(
        center_xy_um=(x, y),
        diameter_um=2.0 * radius,
        rms_um=float(math.sqrt(float(np.mean(residual * residual)))),
        n_points=int(points.shape[0]),
        n_used=int(keep.sum()),
    )


def encode_wall_frame(frame: npt.NDArray[Any], meta: dict[str, Any]) -> bytes:
    """The frame as a TIFF in its own dtype, with ``meta`` as its JSON metadata.

    Encoded in memory so that the file is written whole by ``write_atomically``
    (FM-38): ``tifffile`` opens its target with ``"wb"``.
    """
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, frame, metadata=meta)
    return buffer.getvalue()
