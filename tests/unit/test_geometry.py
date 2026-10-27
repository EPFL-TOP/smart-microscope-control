"""The camera matrix and the well shapes (m3 §3.1, §3.2): one convention, checked against the sample.

The synthetic plate (``smc.testing.synthetic``) is the independent oracle: it
renders frames from its own pose arithmetic, so a blob it draws at a stage
point must come back to that point through the matrix. Frames are not square
on purpose (FM-73): a swapped ``(height, width)`` is invisible on a square one.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from smc.plugins.geometry import (
    Matrix2,
    footprint,
    pixel_matrix,
    pixel_of_stage,
    stage_of_pixel,
)
from smc.testing.synthetic import Blob, PlateSample

SHAPE = (160, 256)  # (height, width): not square, on purpose (FM-73)


def _exactly(text: str) -> str:
    return rf"\A{re.escape(text)}\Z"


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)``, the formula of §3.1, not the code under test."""
    angle = math.radians(rotation_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    flip = np.diag([-1.0 if mirrored else 1.0, -1.0])
    m = pixel_size_um * (rotation @ flip)
    return ((float(m[0, 0]), float(m[0, 1])), (float(m[1, 0]), float(m[1, 1])))


# --- pixel_matrix ------------------------------------------------------------


def test_pixel_matrix_is_the_upright_camera() -> None:
    assert pixel_matrix(0.275) == ((0.275, 0.0), (0.0, -0.275))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_pixel_matrix_refuses_a_pixel_size_that_is_not_measured(bad: float) -> None:
    message = f"pixel_size_um must be finite and positive, got {bad!r}"
    with pytest.raises(ValueError, match=_exactly(message)):
        pixel_matrix(bad)


# --- against the synthetic sample ----------------------------------------------


@pytest.mark.parametrize(
    ("rotation_deg", "mirrored", "pixel_size_um"),
    [(0.0, False, 1.0), (30.0, False, 0.5), (30.0, True, 1.0), (-12.0, True, 0.5)],
)
@pytest.mark.parametrize(("x_um", "y_um"), [(30.0, 10.0), (-20.0, 25.0), (10.0, -15.0)])
def test_a_blob_the_synthetic_sample_rendered_maps_back_to_its_stage_point(
    rotation_deg: float,
    mirrored: bool,
    pixel_size_um: float,
    x_um: float,
    y_um: float,
) -> None:
    sample = PlateSample(
        noise_std=0.0,
        texture_std=0.0,
        well_level=0.0,
        plastic_level=0.0,
        blobs=[
            Blob(x_um, y_um, radius_um=4 * pixel_size_um, intensity=5000.0),
        ],
    )
    frame = sample.render(
        0.0,
        0.0,
        0.0,
        shape=SHAPE,
        pixel_size_um=pixel_size_um,
        camera_rotation_deg=rotation_deg,
        mirrored=mirrored,
    )
    rows, cols = np.indices(frame.shape)
    weight = frame.astype(np.float64)
    total = float(weight.sum())
    col, row = (
        float((cols * weight).sum() / total),
        float((rows * weight).sum() / total),
    )
    matrix = _matrix(pixel_size_um, rotation_deg, mirrored)

    stage = stage_of_pixel(0.0, 0.0, col, row, matrix=matrix, image_shape_px=SHAPE)
    assert math.dist(stage, (x_um, y_um)) / pixel_size_um == pytest.approx(
        0.0, abs=0.05
    )
    pixel = pixel_of_stage(0.0, 0.0, x_um, y_um, matrix=matrix, image_shape_px=SHAPE)
    assert math.dist(pixel, (col, row)) == pytest.approx(0.0, abs=0.05)


# --- stage_of_pixel, pixel_of_stage ------------------------------------------


def test_the_frame_centre_is_half_a_pixel_inside_the_edge_pixels() -> None:
    # A (3, 5) frame: columns 0..4, rows 0..2, so the centre is pixel (2, 1).
    matrix = pixel_matrix(0.5)
    assert stage_of_pixel(
        10.0, 20.0, 2.0, 1.0, matrix=matrix, image_shape_px=(3, 5)
    ) == (10.0, 20.0)
    # Columns grow with +X and rows with -Y: the top-left pixel is up and left.
    assert stage_of_pixel(
        10.0, 20.0, 0.0, 0.0, matrix=matrix, image_shape_px=(3, 5)
    ) == (9.0, 20.5)
    assert stage_of_pixel(
        10.0, 20.0, 4.0, 2.0, matrix=matrix, image_shape_px=(3, 5)
    ) == (11.0, 19.5)


@pytest.mark.parametrize(
    ("col", "row"), [(0.0, 0.0), (255.0, 159.0), (12.5, 100.25), (-40.0, 300.0)]
)
def test_pixel_of_stage_inverts_stage_of_pixel(col: float, row: float) -> None:
    matrix = _matrix(0.5, 30.0, True)
    x_um, y_um = stage_of_pixel(
        123.4, -56.7, col, row, matrix=matrix, image_shape_px=SHAPE
    )
    back = pixel_of_stage(123.4, -56.7, x_um, y_um, matrix=matrix, image_shape_px=SHAPE)
    assert back == pytest.approx((col, row), abs=1e-9)


@pytest.mark.parametrize(
    "matrix",
    [
        ((1.0, 2.0), (2.0, 4.0)),
        ((1e-7, 0.0), (0.0, 1e-7)),
        ((math.nan, 0.0), (0.0, 1.0)),
    ],
    ids=["rank-1", "tiny-pixel", "nan"],
)
def test_pixel_of_stage_refuses_a_singular_matrix(matrix: Matrix2) -> None:
    with pytest.raises(ValueError, match="the camera matrix cannot be inverted"):
        pixel_of_stage(0.0, 0.0, 1.0, 1.0, matrix=matrix, image_shape_px=SHAPE)


def test_pixel_of_stage_accepts_a_small_but_real_matrix() -> None:
    # 1 nm per pixel is absurd and still invertible; the bound is on the determinant.
    pixel = pixel_of_stage(
        0.0,
        0.0,
        0.002,
        0.0,
        matrix=((0.001, 0.0), (0.0, -0.001)),
        image_shape_px=SHAPE,
    )
    assert pixel == pytest.approx((127.5 + 2.0, 79.5))


# --- footprint ---------------------------------------------------------------


def test_footprint_of_an_upright_frame_is_the_outer_edges_of_its_edge_pixels() -> None:
    corners = footprint(100.0, 200.0, matrix=pixel_matrix(0.5), image_shape_px=SHAPE)
    flat = [value for corner in corners for value in corner]
    assert flat == pytest.approx([36.0, 240.0, 164.0, 240.0, 164.0, 160.0, 36.0, 160.0])


def test_footprint_of_a_rotated_frame_is_the_rotated_rectangle() -> None:
    matrix = _matrix(0.5, 30.0, False)
    top_left, top_right, bottom_right, bottom_left = footprint(
        0.0, 0.0, matrix=matrix, image_shape_px=SHAPE
    )
    angle = math.radians(30.0)
    assert (top_right[0] - top_left[0], top_right[1] - top_left[1]) == pytest.approx(
        (128 * math.cos(angle), 128 * math.sin(angle))
    )
    assert math.dist(bottom_left, top_left) == pytest.approx(80.0)
    # A rectangle: the fourth corner closes it, and the frame centre is its middle.
    assert (
        bottom_right[0] - bottom_left[0],
        bottom_right[1] - bottom_left[1],
    ) == pytest.approx((top_right[0] - top_left[0], top_right[1] - top_left[1]))
    xs = [corner[0] for corner in (top_left, top_right, bottom_right, bottom_left)]
    ys = [corner[1] for corner in (top_left, top_right, bottom_right, bottom_left)]
    assert (max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx(
        (150.851, 133.282), abs=1e-3
    )
    assert (sum(xs) / 4, sum(ys) / 4) == pytest.approx((0.0, 0.0), abs=1e-9)
