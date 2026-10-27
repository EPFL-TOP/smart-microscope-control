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
import useq

from smc.plugins.geometry import (
    Matrix2,
    footprint,
    pixel_matrix,
    pixel_of_stage,
    stage_of_pixel,
)
from smc.plugins.slots.plate import WellGeometry, normalise_well, well_geometry
from smc.testing.synthetic import Blob, PlateSample

SHAPE = (160, 256)  # (height, width): not square, on purpose (FM-73)
A1 = (1234.5, -987.6)


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


# --- wells -------------------------------------------------------------------


def _plan(plate: str, rotation: float | None = 13.0) -> useq.WellPlatePlan:
    return useq.WellPlatePlan(plate=plate, a1_center_xy=A1, rotation=rotation)


def _at(geometry: WellGeometry, u_um: float, v_um: float) -> tuple[float, float]:
    """The stage point at offset ``(u, v)`` in the plate's axes: turned by +rotation."""
    angle = math.radians(geometry.rotation_deg)
    cx, cy = geometry.center_um
    return (
        cx + math.cos(angle) * u_um - math.sin(angle) * v_um,
        cy + math.sin(angle) * u_um + math.cos(angle) * v_um,
    )


@pytest.mark.parametrize("plate", sorted(useq.registered_well_plate_keys()))
def test_well_centres_agree_with_useqs_positions_on_every_registered_plate(
    plate: str,
) -> None:
    plan = _plan(plate)
    positions = plan.all_well_positions
    assert positions
    for position in positions:
        geometry = well_geometry(plan, str(position.name))
        assert geometry.well == position.name
        assert geometry.center_um == pytest.approx((position.x, position.y), abs=1e-6)


def test_a_well_ends_where_the_synthetic_plate_says() -> None:
    # PlateSample.well_at is the independent oracle: it was written for the
    # renderer, from the plate's own pose arithmetic.
    round_plan = _plan("96-well")
    round_sample = PlateSample("96-well", A1, 13.0)
    disc = well_geometry(round_plan, "C5")
    polar = math.radians(40.0)
    for radius_um, inside in ((3200.0 - 1.0, True), (3200.0 + 1.0, False)):
        x_um = disc.center_um[0] + radius_um * math.cos(polar)
        y_um = disc.center_um[1] + radius_um * math.sin(polar)
        assert (round_sample.well_at(x_um, y_um) == "C5") is inside
        assert disc.contains(x_um, y_um) is inside

    square_plan = _plan("384-well")
    square_sample = PlateSample("384-well", A1, 13.0)
    square = well_geometry(square_plan, "F10")
    half = 1700.0
    for u_um, v_um, inside in (
        (half - 1.0, 0.0, True),
        (half + 1.0, 0.0, False),
        (0.95 * half, 0.95 * half, True),
    ):
        x_um, y_um = _at(square, u_um, v_um)
        assert (square_sample.well_at(x_um, y_um) == "F10") is inside
        assert square.contains(x_um, y_um) is inside


def test_a_round_well_is_a_disc_and_a_square_well_a_square() -> None:
    disc = well_geometry(_plan("96-well", 0.0), "A1")
    # Inside the bounding square of the well either way, but not inside the disc.
    assert disc.contains(*_at(disc, 2200.0, 2200.0)) is True
    assert disc.contains(*_at(disc, 2300.0, 2300.0)) is False
    square = well_geometry(_plan("384-well", 30.0), "A1")
    assert square.contains(*_at(square, 1600.0, 1600.0)) is True
    assert square.contains(*_at(square, 1600.0, 1800.0)) is False


def test_contains_shrinks_by_the_margin() -> None:
    disc = well_geometry(_plan("96-well"), "B2")
    assert disc.contains(*_at(disc, 2000.0, 2000.0), margin_um=200.0) is True
    assert disc.contains(*_at(disc, 2200.0, 2200.0), margin_um=200.0) is False
    square = well_geometry(_plan("384-well"), "B2")
    assert square.contains(*_at(square, 1600.0, 1600.0), margin_um=0.0) is True
    assert square.contains(*_at(square, 1600.0, 1600.0), margin_um=200.0) is False
    # A margin at or above the half-size leaves nothing, not even the centre.
    assert disc.contains(*disc.center_um, margin_um=3199.0) is True
    assert disc.contains(*disc.center_um, margin_um=3200.0) is False
    assert disc.contains(*disc.center_um, margin_um=3300.0) is False
    assert square.contains(*square.center_um, margin_um=1699.0) is True
    assert square.contains(*square.center_um, margin_um=1700.0) is False
    assert square.contains(*square.center_um, margin_um=1800.0) is False


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_a_negative_or_non_finite_margin_is_refused(bad: float) -> None:
    geometry = well_geometry(_plan("96-well"), "A1")
    message = f"margin_um must be finite and not negative, got {bad!r}"
    with pytest.raises(ValueError, match=_exactly(message)):
        geometry.contains(*geometry.center_um, margin_um=bad)


@pytest.mark.parametrize(
    ("x_um", "y_um"),
    [(math.nan, 0.0), (0.0, math.nan), (math.inf, 0.0), (0.0, -math.inf)],
)
def test_a_point_that_is_not_finite_is_outside(x_um: float, y_um: float) -> None:
    # Unknown is not inside: a missing detection must never count as a hit.
    geometry = well_geometry(_plan("96-well", 0.0), "A1")
    shifted = (geometry.center_um[0] + x_um, geometry.center_um[1] + y_um)
    assert geometry.contains(*shifted) is False


def test_to_plate_frame_undoes_the_plate_rotation() -> None:
    geometry = WellGeometry(
        well="A1",
        center_um=(100.0, 200.0),
        size_um=(3400.0, 3400.0),
        circular=False,
        rotation_deg=30.0,
    )
    # Anisotropic on purpose: a swapped axis or a wrong sign does not cancel.
    point = _at(geometry, 300.0, -1200.0)
    assert geometry.to_plate_frame(*point) == pytest.approx((300.0, -1200.0), abs=1e-9)


def test_a_plan_without_a_rotation_is_not_rotated() -> None:
    plan = useq.WellPlatePlan(plate="96-well", a1_center_xy=(0, 0))
    assert plan.rotation is None
    geometry = well_geometry(plan, "A2")
    assert geometry.rotation_deg == 0.0
    assert geometry.center_um == pytest.approx((9000.0, 0.0))


@pytest.mark.parametrize("well", ["Z9", "I13", "z9"])
def test_a_well_not_on_the_plate_is_refused_with_its_name(well: str) -> None:
    with pytest.raises(
        ValueError, match=_exactly(f"well {well!r} is not on the 96-well plate")
    ):
        well_geometry(_plan("96-well"), well)


def test_a_well_is_found_by_its_normalised_name() -> None:
    plan = _plan("96-well")
    assert well_geometry(plan, "a01") == well_geometry(plan, "A1")
    assert well_geometry(plan, "a01").well == "A1"
    big = _plan("1536-well")
    assert well_geometry(big, "aa3").well == "AA3"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("a1", "A1"),
        ("A01", "A1"),
        (" h 12 ", "H12"),
        ("aa03", "AA3"),
        ("glass", "GLASS"),
        ("", ""),
        ("A0", "A0"),
        ("A000", "A0"),
        ("B2-B5", "B2-B5"),
        ("A1_0003", "A1_0003"),
        ("A" + "0" * 5000 + "7", "A7"),
    ],
)
def test_normalise_well_gives_useqs_names(given: str, expected: str) -> None:
    assert normalise_well(given) == expected


# Every registered plate has the same pitch and well size along x and along y, so
# the tests above cannot see the two exchanged (FM-73). A custom ``useq.WellPlate``
# is a real input (``WellGeometry`` is public), and this one is lopsided on purpose.
LOPSIDED = useq.WellPlate(
    rows=2,
    columns=3,
    well_spacing=(9.0, 6.0),
    well_size=(6.0, 2.0),
    circular_wells=False,
    name="lopsided",
)


def test_pitch_and_size_of_a_plate_that_is_not_square_follow_useq() -> None:
    plan = useq.WellPlatePlan(
        plate=LOPSIDED, a1_center_xy=(100.0, 200.0), rotation=20.0
    )
    for position in plan.all_well_positions:
        geometry = well_geometry(plan, str(position.name))
        assert geometry.center_um == pytest.approx((position.x, position.y), abs=1e-6)
    geometry = well_geometry(plan, "B3")
    assert geometry.size_um == pytest.approx((6000.0, 2000.0))
    assert geometry.circular is False
    assert geometry.rotation_deg == 20.0


def test_a_wide_flat_well_is_a_rectangle_not_a_square() -> None:
    geometry = WellGeometry(
        well="A1",
        center_um=(0.0, 0.0),
        size_um=(6000.0, 2000.0),
        circular=False,
        rotation_deg=30.0,
    )
    # Off-axis, in the plate's axes: 2900 along the long side is inside, 1100
    # along the short side is outside, and neither is the other's mirror image.
    assert geometry.contains(*_at(geometry, 2900.0, 900.0)) is True
    assert geometry.contains(*_at(geometry, 2900.0, 1100.0)) is False
    assert geometry.contains(*_at(geometry, 3100.0, 0.0)) is False
    assert geometry.contains(*_at(geometry, 900.0, 2900.0)) is False
    # The margin shrinks each side by itself: 2000 wide is 200 µm from a 3000 µm half.
    assert geometry.contains(*_at(geometry, 2700.0, 700.0), margin_um=200.0) is True
    assert geometry.contains(*_at(geometry, 2700.0, 900.0), margin_um=200.0) is False
