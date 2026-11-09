"""Image sides and stage sides: the frames a wall is named in (m3 §19).

A direction means something in one frame only. The operator names a wall on the
image (``left``, ``right``, ``top``, ``bottom``); the layer names it on the stage
(``+X``, ``-X``, ``+Y``, ``-Y``, the X and Y of ``smc stage get``). Which stage
side an image side is depends on the camera's turn and mirror, because the
stand's optical path decides how the image moves with the stage: on the Ti2-E at
10x the image's left is the stage's +X (Ti2-E session 2, FM-189). The camera
calibration measures that turn, and this module is the one place that reads it
as words.

Pure arithmetic on the camera matrix of ``geometry`` (``(dx, dy) = M @ (d_col,
d_row)``); nothing here imports ``smc.hardware``. A direction is read per axis,
by the larger component of a matrix column, so a pixel size is never used: an
anisotropic matrix or a frame that is not square reads the same way (FM-73).
"""

from __future__ import annotations

import math
from typing import Literal

from smc.plugins.geometry import Matrix2
from smc.plugins.slots.plate import ImageSide, WallSide

__all__ = [
    "QUARTER_TURN_TOLERANCE_DEG",
    "ImageDirection",
    "image_direction_of_stage_step",
    "stage_side_of_image_side",
]

#: Where the content of the image goes when the stage steps.
ImageDirection = Literal["left", "right", "up", "down"]

#: How far a camera may be turned away from a quarter turn and still name a side.
#: Chosen, not measured: a wall touched ``t`` µm off the well's mid-line errs by
#: ``t * sin(5°)`` on the axis read, 87 µm for 1 mm, about what a hand-touched wall
#: is worth at 10x, and a mounted camera is within a degree or two of a quarter turn.
#: A camera turned further is for centring wells, not for naming walls.
QUARTER_TURN_TOLERANCE_DEG = 5.0

#: Below this a matrix images nothing (the same bound as ``geometry``).
_MIN_DETERMINANT = 1e-12

_STEPS: dict[WallSide, tuple[float, float]] = {
    "+X": (1.0, 0.0),
    "-X": (-1.0, 0.0),
    "+Y": (0.0, 1.0),
    "-Y": (0.0, -1.0),
}


def stage_side_of_image_side(side: ImageSide, matrix: Matrix2) -> WallSide:
    """The stage side that ``side`` of the image is, for a camera of this ``matrix``.

    ``right`` is where a pixel to the right of the frame's centre lies on the stage
    (the matrix's first column), ``bottom`` where a pixel below it does (the second
    column), and ``left`` and ``top`` their opposites.

    Raises:
        ValueError: The matrix cannot be inverted or is not finite, or the camera
            is turned more than ``QUARTER_TURN_TOLERANCE_DEG`` away from the
            stage's axes: no image side is then a stage side.
    """
    _check_invertible(matrix)
    (m00, m01), (m10, m11) = matrix
    vectors = {
        "right": (m00, m10),
        "bottom": (m01, m11),
        "left": (-m00, -m10),
        "top": (-m01, -m11),
    }
    axis, positive = _axis_of(vectors[side])
    return _STAGE_SIDES[axis, positive]


def image_direction_of_stage_step(step: WallSide, matrix: Matrix2) -> ImageDirection:
    """What a stage step does to the image: the direction its content moves.

    The camera looks at a fixed point of the stage and the stage moves, so the
    content moves by ``-M⁻¹ @ d`` pixels for a stage step ``d``: ``right`` for
    ``+col``, ``left``, ``down`` for ``+row`` and ``up``.

    Raises:
        ValueError: As ``stage_side_of_image_side``.
    """
    determinant = _check_invertible(matrix)
    (m00, m01), (m10, m11) = matrix
    dx, dy = _STEPS[step]
    d_col = -(m11 * dx - m01 * dy) / determinant
    d_row = -(-m10 * dx + m00 * dy) / determinant
    axis, positive = _axis_of((d_col, d_row))
    return _IMAGE_DIRECTIONS[axis, positive]


_STAGE_SIDES: dict[tuple[int, bool], WallSide] = {
    (0, True): "+X",
    (0, False): "-X",
    (1, True): "+Y",
    (1, False): "-Y",
}
_IMAGE_DIRECTIONS: dict[tuple[int, bool], ImageDirection] = {
    (0, True): "right",
    (0, False): "left",
    (1, True): "down",
    (1, False): "up",
}


def _check_invertible(matrix: Matrix2) -> float:
    """The determinant, or a refusal that says to run the calibration again."""
    (m00, m01), (m10, m11) = matrix
    determinant = m00 * m11 - m01 * m10
    # `not >=`, not `<`: a nan must be refused too (FM-104). An infinite entry can
    # leave a determinant that is finite or infinite, so it is refused on its own.
    finite = all(math.isfinite(v) for v in (m00, m01, m10, m11))
    if not (finite and abs(determinant) >= _MIN_DETERMINANT):
        raise ValueError(
            f"the camera matrix cannot be inverted (determinant {determinant!r}); "
            "run camera-calibration"
        )
    return determinant


def _axis_of(vector: tuple[float, float]) -> tuple[int, bool]:
    """The axis a vector lies nearest to (0 first, 1 second) and its sign.

    Raises:
        ValueError: The vector is more than ``QUARTER_TURN_TOLERANCE_DEG`` off both
            axes, so it names no side.
    """
    a, b = abs(vector[0]), abs(vector[1])
    major, minor = max(a, b), min(a, b)
    off = math.degrees(math.atan2(minor, major))
    # `not <=`, not `>`: a nan angle is refused too (FM-104).
    if not off <= QUARTER_TURN_TOLERANCE_DEG:
        raise ValueError(
            f"the camera is turned {off:.1f}° away from the stage's axes, a quarter "
            f"turn at most {QUARTER_TURN_TOLERANCE_DEG:g}° off: an arrow moves the "
            "image diagonally and an image side is not a stage side"
        )
    axis = 0 if a >= b else 1
    return axis, vector[axis] > 0.0
