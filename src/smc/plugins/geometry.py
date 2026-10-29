"""The camera matrix: pixels of a frame to µm of the stage and back (m3 §3.1).

Camera calibration measures it, the scan plan sizes its grid with it, the cells
stage turns a box in pixels into a stage position with it and the route turns a
stage position into a field with it. One convention, written once, so that no
stage re-derives a sign (FM-73: a swapped ``(height, width)`` or a half pixel
at the centre is invisible on a square frame and wrong on a camera's).

The stage position recorded with a frame is the point imaged at the frame
centre. The matrix ``M`` maps a pixel offset ``(d_col, d_row)`` from that
centre to a stage offset ``(dx_um, dy_um)``. Rows grow downwards and stage Y
upwards (M1 §2), so an upright camera without mirror has
``M = ps * R(theta) * diag(1, -1)``. This is the convention of
``smc.testing.synthetic`` (``camera_rotation_deg``, ``mirrored``), which the
tests render against; ``nikon-control`` called the upright camera "mirrored",
and here it is not.

Nothing here imports ``smc.hardware``: it is arithmetic on numbers.
"""

from __future__ import annotations

import math

__all__ = [
    "Matrix2",
    "footprint",
    "fov_of",
    "pixel_matrix",
    "pixel_of_stage",
    "stage_of_pixel",
]

#: ``((m00, m01), (m10, m11))`` in µm per pixel: ``(dx, dy) = M @ (d_col, d_row)``.
Matrix2 = tuple[tuple[float, float], tuple[float, float]]

#: Below this a matrix is a camera that images nothing: a pixel size of 1e-6 µm
#: is not an objective. ``abs(det)`` is in µm² per pixel².
_MIN_DETERMINANT = 1e-12


def pixel_matrix(pixel_size_um: float) -> Matrix2:
    """The matrix assumed when no camera calibration exists: upright, not mirrored.

    Rotation 0 and no mirror are an assumption, and a caller that uses this
    matrix says so in its result's ``assumed`` (measured beats assumed).

    Raises:
        ValueError: ``pixel_size_um`` is not finite and positive. A pixel size
            of 0 means "not measured" and is reported, never defaulted (FM-104).
    """
    if not (math.isfinite(pixel_size_um) and pixel_size_um > 0.0):
        raise ValueError(
            f"pixel_size_um must be finite and positive, got {pixel_size_um!r}"
        )
    return ((pixel_size_um, 0.0), (0.0, -pixel_size_um))


def stage_of_pixel(
    x_um: float,
    y_um: float,
    col_px: float,
    row_px: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> tuple[float, float]:
    """The stage point imaged at pixel ``(col_px, row_px)`` of a frame taken at ``(x_um, y_um)``.

    ``image_shape_px`` is ``(height, width)``. Pixels are fractional and may
    lie outside the frame: a detection box is a float, and so is a corner.
    """
    centre_col, centre_row = _centre(image_shape_px)
    d_col, d_row = col_px - centre_col, row_px - centre_row
    (m00, m01), (m10, m11) = matrix
    return (x_um + m00 * d_col + m01 * d_row, y_um + m10 * d_col + m11 * d_row)


def pixel_of_stage(
    x_um: float,
    y_um: float,
    px_x_um: float,
    px_y_um: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> tuple[float, float]:
    """The pixel ``(col, row)`` of a frame taken at ``(x_um, y_um)`` that images a stage point.

    ``px_x_um`` and ``px_y_um`` are the **stage** coordinates of the point, in
    µm, not pixels: the names are the design's (the inverse of
    ``stage_of_pixel``, whose ``x_um`` and ``y_um`` are the frame's).

    Raises:
        ValueError: The matrix is singular or not finite, so no pixel maps to
            a stage point and back.
    """
    (m00, m01), (m10, m11) = matrix
    determinant = m00 * m11 - m01 * m10
    # `not >=`, not `<`: a nan determinant must be refused too.
    if not abs(determinant) >= _MIN_DETERMINANT:
        raise ValueError(
            f"the camera matrix cannot be inverted (determinant {determinant!r}); "
            f"run camera-calibration, or give a pixel size"
        )
    centre_col, centre_row = _centre(image_shape_px)
    dx, dy = px_x_um - x_um, px_y_um - y_um
    d_col = (m11 * dx - m01 * dy) / determinant
    d_row = (m00 * dy - m10 * dx) / determinant
    return (centre_col + d_col, centre_row + d_row)


def footprint(
    x_um: float, y_um: float, *, matrix: Matrix2, image_shape_px: tuple[int, int]
) -> tuple[tuple[float, float], ...]:
    """The four stage corners of a frame taken at ``(x_um, y_um)``, a rotated rectangle.

    In the order of the pixel corners ``(0, 0)``, ``(W, 0)``, ``(W, H)``,
    ``(0, H)``: the outer edges of the edge pixels, ``-0.5`` and ``W - 0.5``
    in pixel-centre coordinates, so the footprint is the area the frame
    really covers and not the one its edge pixel *centres* span.
    """
    height, width = image_shape_px
    corners = (
        (-0.5, -0.5),
        (width - 0.5, -0.5),
        (width - 0.5, height - 0.5),
        (-0.5, height - 0.5),
    )
    return tuple(
        stage_of_pixel(
            x_um, y_um, col, row, matrix=matrix, image_shape_px=image_shape_px
        )
        for col, row in corners
    )


def fov_of(matrix: Matrix2, image_shape_px: tuple[int, int]) -> tuple[float, float]:
    """``(width, height)`` of the frame's footprint in stage axes, in µm.

    The extent of the four corners at the origin: for a camera turned by the
    stage, the bounding box of the rotated frame. The scan plan's axis-aligned
    grid steps by it, and so do the route's candidate lattice and the window that
    counts cells; the fields' own footprints stay the rotated rectangle, so the
    route's overlap rule is on pixel offsets, not on this.
    """
    corners = footprint(0.0, 0.0, matrix=matrix, image_shape_px=image_shape_px)
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _centre(image_shape_px: tuple[int, int]) -> tuple[float, float]:
    """The frame centre ``(col, row)``: ``((W - 1) / 2, (H - 1) / 2)``.

    As the synthetic sample, which puts the stage position at the middle of
    the frame's pixel centres. ``nikon-control`` used ``W / 2``, half a pixel
    off: 0.14 µm with a 0.275 µm pixel, and 0.7 px in the measurement of §2.1.
    """
    height, width = image_shape_px
    return ((width - 1) / 2.0, (height - 1) / 2.0)
