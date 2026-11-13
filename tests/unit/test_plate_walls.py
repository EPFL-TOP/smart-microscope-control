"""The wall finder and the circle fit of ``plate-walls`` (m3 §17.3), on rendered frames and on real ones.

The oracle is the synthetic plate (``PlateSample.render``), which draws a well
from its own pose arithmetic and knows nothing of ``find_wall``: a centre fitted
from frames it rendered must come back to the well's own centre. The camera
matrices are written out from §3.1's formula (``ps * R(theta) * diag(s, -1)``),
not taken from the code, and frames are not square (FM-73): a swapped height
and width, or a row read for a column, is invisible on a square frame. A mirror
and a turn off the axes are in every case that can have them.

The B3 frames of Ti2-E session 2 and the frames of session 3 (B5 at the 4x, B3,
B5 and C3 at the 10x: #409) are private (FM-50) and are never in a worktree:
those tests are skipped, with the reason, when the folder is not there
(``SMC_SESSION2_DIR``, else ``local/surveys/nikon-ti2/session2`` of the
checkout; ``SMC_SESSION3_DIR``, else ``local/scans/2026-11-10/plate-walls``).
"""

from __future__ import annotations

import io
import json
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tifffile
import useq
from hypothesis import given, settings
from hypothesis import strategies as st

from smc.plugins.geometry import Matrix2, pixel_matrix, stage_of_pixel
from smc.plugins.plate_walls.algorithm import (
    BAND_PX,
    MAX_FRAMES,
    MAX_RMS_FRACTION,
    OUTWARD,
    SIDES,
    STEP_FRACTION,
    CircleFit,
    along_um,
    encode_wall_frame,
    find_wall,
    fit_circle,
    walk_offsets_um,
    wall_points_um,
)
from smc.plugins.slots import well_geometry
from smc.testing.synthetic import PlateSample

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)``: §3.1's formula, written out (no matrix product)."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    flip = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * flip, pixel_size_um * sin),
        (pixel_size_um * sin * flip, -pixel_size_um * cos),
    )


def _step_frame(
    shape: tuple[int, int] = (64, 192),
    *,
    last_bright: int = 100,
    bright: float = 10_000.0,
    dark: float = 1_000.0,
) -> npt.NDArray[np.uint16]:
    """Columns ``0..last_bright`` bright, the rest dark: a wall between two columns, no noise."""
    height, width = shape
    frame = np.full((height, width), dark)
    frame[:, : last_bright + 1] = bright
    return frame.astype(np.uint16)


# --- find_wall: where the wall is ------------------------------------------------


@pytest.mark.parametrize(
    ("last_bright", "flipped"),
    [(100, False), (37, False), (100, True)],
    ids=["forward", "near the start", "backward"],
)
def test_the_wall_is_the_half_intensity_edge_between_two_columns(
    last_bright: int, flipped: bool
) -> None:
    # The edge is where the line crosses the midpoint of the two levels: halfway
    # between the last bright column and the first dark one. An upright camera
    # walks +X along +col; a mirrored one walks it along -col, from the far end.
    frame = _step_frame(last_bright=last_bright)
    width = frame.shape[1]
    matrix = _matrix(2.0, 0.0, mirrored=flipped)
    if flipped:
        frame = frame[:, ::-1]
    edge = find_wall(frame, outward_xy=(1.0, 0.0), matrix=matrix)
    assert edge is not None
    expected_col = (width - 1) - (last_bright + 0.5) if flipped else last_bright + 0.5
    assert edge.points_px[:, 0] == pytest.approx(expected_col, abs=1e-9)
    # One point per band of 8 rows, on the band's middle.
    assert edge.n_lines == 8
    assert edge.points_px[:, 1] == pytest.approx(np.arange(8) * BAND_PX + 3.5)
    assert edge.threshold == pytest.approx(5_500.0)
    assert edge.dark_ratio == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("outward", "frame_of", "expected_row"),
    [
        ((0.0, -1.0), lambda f: f.T, 100.5),
        ((0.0, 1.0), lambda f: f.T[::-1], 191 - 100.5),
    ],
    ids=["-Y: rows grow with -Y", "+Y: walked against the rows"],
)
def test_a_wall_along_the_rows_is_read_in_rows_not_columns(
    outward: tuple[float, float], frame_of: object, expected_row: float
) -> None:
    # The same step turned on its side (FM-73: a transposed read gives the
    # wrong axis on a square frame and the right number on no other).
    frame = frame_of(_step_frame())  # type: ignore[operator]
    edge = find_wall(frame, outward_xy=outward, matrix=_matrix(2.0, 0.0, False))
    assert edge is not None
    assert edge.n_lines == 64 // BAND_PX
    assert edge.points_px[:, 1] == pytest.approx(expected_row, abs=1e-9)
    assert edge.points_px[:, 0] == pytest.approx(np.arange(8) * BAND_PX + 3.5)


def test_the_edge_is_interpolated_between_two_samples_not_put_on_one() -> None:
    # 10000, 7000, 1000 along a line: the threshold (the midpoint of the two
    # levels, 5500) is crossed a quarter of the way from the 7000 to the 1000, at
    # column 101.25. A constant half, or the sample itself, would put it elsewhere.
    frame = _step_frame(last_bright=100)
    frame[:, 101] = 7_000
    edge = find_wall(frame, outward_xy=(1.0, 0.0), matrix=_matrix(2.0, 0.0, False))
    assert edge is not None
    assert edge.threshold == pytest.approx(5_500.0)
    assert edge.points_px[:, 0] == pytest.approx(101.25, abs=1e-9)


def test_the_dark_class_must_be_dark_enough_to_be_the_plastic() -> None:
    # 0.4 of the bright class's mean is a wall; 0.6 is shading inside a well.
    matrix = _matrix(2.0, 0.0, False)
    assert find_wall(_step_frame(dark=4_000.0), outward_xy=(1, 0), matrix=matrix)
    assert (
        find_wall(_step_frame(dark=6_000.0), outward_xy=(1, 0), matrix=matrix) is None
    )


def test_half_the_lines_must_find_the_wall() -> None:
    # A wall in 60 % of the rows is a wall; in 40 % it is a corner of something
    # else. The other rows stay bright to the end of the frame.
    matrix = _matrix(2.0, 0.0, False)
    for rows_with_wall, found in ((38, True), (25, False)):
        frame = _step_frame((64, 192))
        frame[rows_with_wall:, :] = 10_000
        edge = find_wall(frame, outward_xy=(1, 0), matrix=matrix)
        assert (edge is not None) is found


def test_a_thin_dark_strip_is_a_class() -> None:
    # 3 % of the pixels, over the 2 % that a class needs.
    frame = _step_frame((64, 200), last_bright=193)
    edge = find_wall(frame, outward_xy=(1, 0), matrix=_matrix(2.0, 0.0, False))
    assert edge is not None
    assert edge.points_px[:, 0] == pytest.approx(193.5)


@pytest.mark.parametrize("level", [0, 7, 65_535])
def test_a_frame_of_one_level_has_no_wall(level: int) -> None:
    frame = np.full((64, 192), level, dtype=np.uint16)
    assert find_wall(frame, outward_xy=(1, 0), matrix=_matrix(2.0, 0.0, False)) is None


def test_a_frame_with_a_second_wall_takes_the_first_fall() -> None:
    # Going outwards: bright, a fall (ours), dark, a rise (the neighbour's near
    # wall), bright, a second fall (the neighbour's far wall). Only the first.
    # At 20 µm per pixel, so that the dark gap, 79 samples, is 1 580 µm: over
    # `MIN_PLASTIC_UM`, a gap between wells and not a band of the meniscus (at 2
    # µm per pixel it is 158 µm, and the next test refuses it).
    frame = np.full((64, 300), 10_000, dtype=np.uint16)
    frame[:, 101:180] = 1_000
    edge = find_wall(frame, outward_xy=(1, 0), matrix=_matrix(20.0, 0.0, False))
    assert edge is not None
    assert edge.points_px[:, 0] == pytest.approx(100.5)


def test_a_dark_band_narrower_than_the_plastic_is_not_a_wall() -> None:
    # The same frame at 2 µm per pixel: the dark band is 158 µm, a ridge of the
    # meniscus and not the plastic, which either reaches the frame's end or is a
    # gap between two wells. It ends inside the frame and is under 1 400 µm: no
    # wall, where the first fall below the midpoint took it (#409).
    frame = np.full((64, 300), 10_000, dtype=np.uint16)
    frame[:, 101:180] = 1_000
    assert find_wall(frame, outward_xy=(1, 0), matrix=_matrix(2.0, 0.0, False)) is None


@pytest.mark.parametrize(
    ("matrix", "outward", "along_rows", "found"),
    [
        (((20.0, 0.0), (0.0, -2.0)), (1.0, 0.0), False, True),
        (((2.0, 0.0), (0.0, -20.0)), (1.0, 0.0), False, False),
        (((2.0, 0.0), (0.0, -20.0)), (0.0, -1.0), True, True),
        (((20.0, 0.0), (0.0, -2.0)), (0.0, -1.0), True, False),
    ],
    ids=[
        "columns at 20 µm",
        "columns at 2 µm",
        "rows at 20 µm",
        "rows at 2 µm",
    ],
)
def test_the_plastics_minimum_length_is_in_um_along_the_line_not_across_it(
    matrix: Matrix2,
    outward: tuple[float, float],
    along_rows: bool,
    found: bool,
) -> None:
    # A dark band of 79 samples that ends inside the frame is 1 580 µm where a
    # sample is 20 µm along the line and 158 µm where it is 2 µm: the plastic of a
    # gap between wells, or a ridge (FM-126, FM-131). The pixel's size across the
    # line is the other number of the matrix, and must not be the one used.
    frame = np.full((64, 300), 10_000, dtype=np.uint16)
    frame[:, 101:180] = 1_000
    if along_rows:
        frame = frame.T
    edge = find_wall(frame, outward_xy=outward, matrix=matrix)
    assert (edge is not None) is found
    if edge is not None:
        assert edge.points_px[:, 1 if along_rows else 0] == pytest.approx(100.5)


@pytest.mark.parametrize(
    ("gap", "found"), [(1, True), (BAND_PX - 1, True), (BAND_PX, False), (40, False)]
)
def test_a_speck_in_the_plastic_splits_its_run_only_from_a_band_wide(
    gap: int, found: bool
) -> None:
    # The plastic of a gap between wells, 600 samples (1 650 µm at 2.75 µm per
    # pixel) that ends inside the frame, with bright specks of `gap` samples in the
    # middle. Under a band's width they are noise on the plastic and the run is
    # one; from a band's width they are the next well and each half is a shadow.
    frame = np.full((64, 800), 50_000, dtype=np.uint16)
    frame[:, 100:700] = 250
    frame[:, 400 : 400 + gap] = 50_000
    edge = find_wall(frame, outward_xy=(1, 0), matrix=_matrix(2.75, 0.0, False))
    assert (edge is not None) is found
    if edge is not None:
        assert edge.points_px[:, 0] == pytest.approx(99.5)


def test_a_shadow_joined_to_the_wall_is_not_taken_for_it() -> None:
    # B5's +X and -X frames at the 4x (#409, §17.1): going outwards the inside
    # (50 000) falls to a shadow joined to the wall (12 700, 0.25 of the way from
    # the plastic), which lies below the midpoint, and then to the plastic (250).
    # The midpoint is crossed at the shadow, 240 columns (660 µm) before the wall.
    # 64 x 600, not square (FM-73), at the 4x's 2.75 µm per pixel.
    frame = np.full((64, 600), 50_000, dtype=np.uint16)
    frame[:, 300:540] = 12_700
    frame[:, 540:] = 250
    edge = find_wall(frame, outward_xy=(1, 0), matrix=_matrix(2.75, 0.0, False))
    assert edge is not None
    # The wall's own half level, 6 475, is crossed halfway between the last
    # shadow column and the first plastic one.
    assert edge.points_px[:, 0] == pytest.approx(539.5)
    assert edge.points_px.shape == (8, 2)
    # The midpoint stays the one that splits the two classes.
    assert edge.threshold == pytest.approx(25_125.0)
    assert edge.dark_ratio == pytest.approx(0.2042, rel=1e-3)


def test_a_ridge_inside_the_well_is_not_taken_for_the_wall() -> None:
    # B5's -Y frames (#409): a dark ridge 247 µm wide whose middle lies 1.6 mm
    # inside the wall, on the inner border of a grey band (30 000). Going
    # outwards the midpoint is crossed at the ridge, column 150; the plastic
    # starts at 777. The ridge's run ends inside the frame and is not 1 400 µm.
    frame = np.full((64, 1_000), 50_000, dtype=np.uint16)
    frame[:, 150:240] = 1_000
    frame[:, 240:777] = 30_000
    frame[:, 777:] = 250
    edge = find_wall(frame, outward_xy=(1, 0), matrix=_matrix(2.75, 0.0, False))
    assert edge is not None
    assert edge.points_px[:, 0] == pytest.approx(776.5)


# --- find_wall on rendered wells ---------------------------------------------------

#: (plate, rotation_deg, pixel_um, (height, width), camera_rotation_deg, mirrored,
#:  rough-centre offset in µm, far well): §17.1's table, two wells each.
CASES = [
    ("96-well", 1.2, 0.65, (384, 512), 30.0, True, (300.0, -200.0), "H12"),
    ("96-well", -3.0, 1.625, (512, 384), -75.0, False, (-900.0, 700.0), "H12"),
    ("24-well", 2.0, 1.1, (1412, 1412), 180.0, False, (1500.0, -1200.0), "D6"),
    ("24-well", 0.0, 2.75, (1412, 1412), 90.0, True, (-2500.0, 2000.0), "D6"),
    ("6-well", 0.5, 1.625, (512, 512), 10.0, False, (3000.0, 0.0), "B3"),
    ("48-well", 0.5, 1.625, (512, 512), 10.0, False, (1000.0, -1000.0), "F8"),
]
CASE_IDS = [
    f"{c[0]}-{c[1]:+g}deg-{c[2]}um-{c[3][0]}x{c[3][1]}-cam{c[4]:g}{'m' if c[5] else ''}"
    for c in CASES
]
A1 = (2_000.0, -3_000.0)


def _render(
    sample: PlateSample,
    at: tuple[float, float],
    shape: tuple[int, int],
    pixel_um: float,
    camera_deg: float,
    mirrored: bool,
) -> npt.NDArray[np.uint16]:
    return sample.render(
        at[0],
        at[1],
        0.0,
        shape=shape,
        pixel_size_um=pixel_um,
        camera_rotation_deg=camera_deg,
        mirrored=mirrored,
    )


#: What `_walk` renders its frames with: `_render`'s signature.
Render = Callable[
    [PlateSample, tuple[float, float], tuple[int, int], float, float, bool],
    npt.NDArray[np.uint16],
]


def _centre_of(sample: PlateSample, well: str) -> tuple[float, float]:
    """The well's true centre from the sample's own plate: the oracle."""
    plan = useq.WellPlatePlan(
        plate=sample.plate_name,  # type: ignore[arg-type]
        a1_center_xy=sample.a1_center_xy_um,
        rotation=sample.rotation_deg,
    )
    return well_geometry(plan, well).center_um


def _radius_of(sample: PlateSample) -> float:
    """The well's nominal radius in µm: useq keeps ``well_size`` in mm."""
    return float(useq.WellPlate.from_str(sample.plate_name).well_size[0]) * 500.0


def _walk(
    sample: PlateSample,
    recorded: tuple[float, float],
    *,
    radius_um: float,
    shape: tuple[int, int],
    pixel_um: float,
    camera_deg: float,
    mirrored: bool,
    render: Render = _render,
) -> tuple[npt.NDArray[np.float64], int]:
    """§17.4's walk for one well, with no stage: the edge points of its four centred frames, and the frames taken."""
    matrix = _matrix(pixel_um, camera_deg, mirrored)
    search_um = 0.5 * radius_um
    step_um = STEP_FRACTION * min(shape) * pixel_um
    taken = 0
    kept = []
    for side in SIDES:
        outward = OUTWARD[side]
        found: float | None = None
        for d in walk_offsets_um(radius_um, search_um, step_um):
            at = (recorded[0] + outward[0] * d, recorded[1] + outward[1] * d)
            taken += 1
            edge = find_wall(
                render(sample, at, shape, pixel_um, camera_deg, mirrored),
                outward_xy=outward,
                matrix=matrix,
            )
            if edge is None:
                continue
            out = along_um(
                wall_points_um(
                    edge.points_px, *at, matrix=matrix, image_shape_px=shape
                ),
                recorded,
                outward,
            )
            if abs(out - radius_um) <= search_um:
                found = out
                break
        assert found is not None, f"no wall on the {side} side"
        at = (recorded[0] + outward[0] * found, recorded[1] + outward[1] * found)
        taken += 1
        edge = find_wall(
            render(sample, at, shape, pixel_um, camera_deg, mirrored),
            outward_xy=outward,
            matrix=matrix,
        )
        assert edge is not None, f"the wall of the {side} side was lost when centred"
        kept.append(
            wall_points_um(edge.points_px, *at, matrix=matrix, image_shape_px=shape)
        )
    return np.vstack(kept), taken


@pytest.mark.parametrize(
    ("plate", "rotation", "pixel_um", "shape", "camera_deg", "mirrored", "off", "far"),
    CASES,
    ids=CASE_IDS,
)
def test_the_centre_of_a_well_is_recovered_within_a_micrometre(
    plate: str,
    rotation: float,
    pixel_um: float,
    shape: tuple[int, int],
    camera_deg: float,
    mirrored: bool,
    off: tuple[float, float],
    far: str,
) -> None:
    sample = PlateSample(plate, A1, rotation, seed=3)
    for well, sign in (("A1", 1.0), (far, -1.0)):
        truth = _centre_of(sample, well)
        recorded = (truth[0] + sign * off[0], truth[1] + sign * off[1])
        radius_um = _radius_of(sample)
        points, _ = _walk(
            sample,
            recorded,
            radius_um=radius_um,
            shape=shape,
            pixel_um=pixel_um,
            camera_deg=camera_deg,
            mirrored=mirrored,
        )
        fit = fit_circle(points, floor_um=2.0 * pixel_um)
        assert math.dist(fit.center_xy_um, truth) < 1.0
        assert fit.diameter_um == pytest.approx(2.0 * radius_um, abs=2.0)
        assert fit.n_used >= 0.95 * fit.n_points


def _render_with_meniscus(
    sample: PlateSample,
    at: tuple[float, float],
    shape: tuple[int, int],
    pixel_um: float,
    camera_deg: float,
    mirrored: bool,
    *,
    centre: tuple[float, float],
    radius_um: float,
) -> npt.NDArray[np.uint16]:
    """``_render``, then the well's meniscus and an inner ring painted on, from each pixel's stage position (#409).

    ``d_in`` is how far a pixel lies inside the wall of the well centred at
    ``centre``. The meniscus is the 800 µm inside the wall, darkened to 0.3 of
    the way from the plastic's level to the well's; the ring is 250 µm wide,
    its middle 1.6 mm inside the wall, at 0.05 (B5's -Y ridge, §17.1). The
    pixel's position is written out from the matrix, as `wall_points_um` has it
    but without calling it: the test does not use what it tests.
    """
    frame = _render(sample, at, shape, pixel_um, camera_deg, mirrored)
    height, width = shape
    (m00, m01), (m10, m11) = _matrix(pixel_um, camera_deg, mirrored)
    d_col = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    d_row = np.arange(height, dtype=np.float64)[:, None] - (height - 1) / 2.0
    x = at[0] + m00 * d_col + m01 * d_row
    y = at[1] + m10 * d_col + m11 * d_row
    d_in = radius_um - np.hypot(x - centre[0], y - centre[1])
    factor = np.ones_like(d_in)
    factor[(d_in >= 0.0) & (d_in < 800.0)] = 0.3
    factor[np.abs(d_in - 1_600.0) < 125.0] = 0.05
    plastic = sample.plastic_level
    painted = plastic + (frame.astype(np.float64) - plastic) * factor
    return np.rint(np.clip(painted, 0.0, 65_535.0)).astype(np.uint16)


@pytest.mark.parametrize(
    "off",
    [(0.0, 1_900.0), (-1_200.0, 900.0)],
    ids=["B5's 1.9 mm in Y", "1.2 mm in X and 0.9 mm in Y"],
)
def test_a_well_with_a_meniscus_and_an_inner_ring_is_measured_at_its_wall(
    off: tuple[float, float],
) -> None:
    # The acceptance criterion of #409, on a 24-well at the Ti2-E's 4x
    # (2.739 µm per pixel, the camera at 177.437°, not mirrored) on a frame that
    # is not square (FM-73). The meniscus shades the inside and a ring lies 1.6 mm
    # inside the wall, as on B5; the recorded centre is off by B5's 1.9 mm or by
    # another vector. The first fall below the midpoint is the ring's or the
    # shadow's: the midpoint rule fitted this well 3 450 µm short of its diameter.
    sample = PlateSample("24-well", A1, 0.0, seed=3)
    truth = _centre_of(sample, "A1")
    radius_um = _radius_of(sample)
    pixel_um, shape, camera_deg = 2.739, (1412, 1600), 177.437

    def render(
        sample: PlateSample,
        at: tuple[float, float],
        shape: tuple[int, int],
        pixel_um: float,
        camera_deg: float,
        mirrored: bool,
    ) -> npt.NDArray[np.uint16]:
        return _render_with_meniscus(
            sample,
            at,
            shape,
            pixel_um,
            camera_deg,
            mirrored,
            centre=truth,
            radius_um=radius_um,
        )

    recorded = (truth[0] + off[0], truth[1] + off[1])
    points, _ = _walk(
        sample,
        recorded,
        radius_um=radius_um,
        shape=shape,
        pixel_um=pixel_um,
        camera_deg=camera_deg,
        mirrored=False,
        render=render,
    )
    fit = fit_circle(points, floor_um=2.0 * pixel_um)
    assert math.dist(fit.center_xy_um, truth) < 1.0
    assert fit.diameter_um == pytest.approx(2.0 * radius_um, abs=2.0)
    assert fit.n_used >= 0.95 * fit.n_points


def test_a_wall_is_found_on_each_side_with_a_turned_mirrored_camera() -> None:
    # A 384 x 512 frame under a camera turned 30° and mirrored (FM-73): +X and -X
    # are mostly along the columns (48 bands of rows), +Y and -Y mostly along the
    # rows (64 bands of columns), and each wall is where the plate has it.
    sample = PlateSample("96-well", A1, 1.2, seed=5)
    truth = _centre_of(sample, "B3")
    radius_um = 3200.0
    pixel_um, shape = 0.65, (384, 512)
    matrix = _matrix(pixel_um, 30.0, True)
    lines = {"+X": 48, "-X": 48, "+Y": 64, "-Y": 64}
    angle = math.radians(1.2)
    for side in SIDES:
        # Plate axes turned by the plate's 1.2°: the wall is r along the turned axis.
        ux, uy = OUTWARD[side]
        direction = (
            math.cos(angle) * ux - math.sin(angle) * uy,
            math.sin(angle) * ux + math.cos(angle) * uy,
        )
        at = (truth[0] + direction[0] * radius_um, truth[1] + direction[1] * radius_um)
        edge = find_wall(
            _render(sample, at, shape, pixel_um, 30.0, True),
            outward_xy=OUTWARD[side],
            matrix=matrix,
        )
        assert edge is not None, side
        assert edge.n_lines == lines[side]
        points = wall_points_um(
            edge.points_px, *at, matrix=matrix, image_shape_px=shape
        )
        assert abs(math.dist(tuple(np.median(points, axis=0)), at)) < 5.0


def test_a_wrong_camera_turn_finds_no_wall() -> None:
    # The Ti2-E's 10x camera is turned 180°. With the matrix of an upright camera
    # the outward direction points into the plastic, every line starts dark and
    # no line gives a point: a refusal, not a centre a well-width off.
    sample = PlateSample("24-well", A1, 0.0, seed=2)
    truth = _centre_of(sample, "B2")
    radius_um = 7800.0
    pixel_um, shape = 2.0, (384, 512)
    right = _matrix(pixel_um, 180.0, False)
    wrong = pixel_matrix(pixel_um)
    for side in SIDES:
        outward = OUTWARD[side]
        at = (truth[0] + outward[0] * radius_um, truth[1] + outward[1] * radius_um)
        frame = _render(sample, at, shape, pixel_um, 180.0, False)
        assert find_wall(frame, outward_xy=outward, matrix=right) is not None, side
        assert find_wall(frame, outward_xy=outward, matrix=wrong) is None, side


def test_a_frame_inside_a_well_has_no_wall() -> None:
    sample = PlateSample("96-well", A1, 0.0, seed=4)
    centre = _centre_of(sample, "C4")
    frame = _render(sample, centre, (384, 512), 0.65, 30.0, True)
    for side in SIDES:
        matrix = _matrix(0.65, 30.0, True)
        assert find_wall(frame, outward_xy=OUTWARD[side], matrix=matrix) is None


def test_shading_without_a_wall_is_not_a_wall() -> None:
    # The meniscus darkens the inside towards the wall: a ramp from 6000 to 9000
    # counts has a dark half at 0.82 of the bright half, over the bound of 0.5.
    rng = np.random.default_rng(7)
    ramp = np.linspace(6_000.0, 9_000.0, 512)[None, :] + np.zeros((384, 1))
    frame = np.clip(ramp + rng.normal(0.0, 20.0, ramp.shape), 0, 65_535).astype(
        np.uint16
    )
    matrix = _matrix(1.0, 0.0, False)
    for outward in ((1.0, 0.0), (-1.0, 0.0)):
        assert find_wall(frame, outward_xy=outward, matrix=matrix) is None


def test_a_frame_on_the_plastic_has_no_wall() -> None:
    # Between A1 and A2 of a 96-well plate, at 5 µm per pixel: 2560 µm of plastic
    # (the gap is 2600 µm), with the same texture as a well but no class.
    sample = PlateSample("96-well", A1, 0.0, seed=6)
    a1, a2 = _centre_of(sample, "A1"), _centre_of(sample, "A2")
    middle = ((a1[0] + a2[0]) / 2.0, (a1[1] + a2[1]) / 2.0)
    frame = _render(sample, middle, (384, 512), 5.0, 0.0, False)
    for side in SIDES:
        assert (
            find_wall(frame, outward_xy=OUTWARD[side], matrix=_matrix(5.0, 0.0, False))
            is None
        )


def test_the_neighbours_near_wall_is_not_taken() -> None:
    # A frame of plastic and the neighbouring well's near wall, none of ours: going
    # +X it starts dark and rises. The first sample of every line is the plastic.
    sample = PlateSample("96-well", A1, 0.0, seed=8)
    a1, a2 = _centre_of(sample, "A1"), _centre_of(sample, "A2")
    at = (a1[0] + 5_000.0, a1[1])  # spans x - 1280 .. x + 1280: 3720 to 6280 from A1
    assert a2[0] - a1[0] - 3200.0 < 6_280.0  # the neighbour's wall is in the frame
    frame = _render(sample, at, (384, 512), 5.0, 0.0, False)
    assert frame[:, :16].mean() < 1_200  # the frame begins on the plastic
    assert frame[:, -16:].mean() > 2_500  # and ends in the neighbour
    assert (
        find_wall(frame, outward_xy=(1.0, 0.0), matrix=_matrix(5.0, 0.0, False)) is None
    )


def test_of_two_walls_in_a_frame_ours_is_taken() -> None:
    # 48-well at 5 µm per pixel: the field (2560 µm) is wider than the gap
    # between wells (1900 µm), so a frame holds our wall and the neighbour's near
    # wall. Ours is the first fall going outwards.
    sample = PlateSample("48-well", A1, 0.0, seed=9)
    a1, a2 = _centre_of(sample, "A1"), _centre_of(sample, "A2")
    radius_um = _radius_of(sample)
    at = (a1[0] + radius_um + 950.0, a1[1])
    matrix = _matrix(5.0, 0.0, False)
    frame = _render(sample, at, (384, 512), 5.0, 0.0, False)
    # Both walls are in the frame, ours at col 66 and the neighbour's at col 446.
    assert a2[0] - radius_um - at[0] < 1_280.0
    edge = find_wall(frame, outward_xy=(1.0, 0.0), matrix=matrix)
    assert edge is not None
    points = wall_points_um(
        edge.points_px, *at, matrix=matrix, image_shape_px=(384, 512)
    )
    # On our circle (the neighbour's near wall is 1900 µm further out), not on the
    # line x = r: the wall curves away from it by 83 µm at the frame's top and bottom.
    distance = np.hypot(points[:, 0] - a1[0], points[:, 1] - a1[1])
    assert distance == pytest.approx(radius_um, abs=3.0)


def test_find_wall_refuses_what_is_not_a_frame_or_a_direction() -> None:
    matrix = _matrix(1.0, 0.0, False)
    with pytest.raises(ValueError, match="2-D"):
        find_wall(np.zeros((4, 4, 4)), outward_xy=(1.0, 0.0), matrix=matrix)
    with pytest.raises(ValueError, match="2-D"):
        find_wall(np.zeros(16), outward_xy=(1.0, 0.0), matrix=matrix)
    with pytest.raises(ValueError, match="2-D"):
        find_wall(np.zeros((0, 8)), outward_xy=(1.0, 0.0), matrix=matrix)
    frame = _step_frame()
    for bad in ((0.0, 0.0), (math.nan, 0.0), (math.inf, 1.0)):
        with pytest.raises(ValueError, match="direction"):
            find_wall(frame, outward_xy=bad, matrix=matrix)
    with pytest.raises(ValueError, match="cannot be inverted"):
        find_wall(frame, outward_xy=(1.0, 0.0), matrix=((1.0, 2.0), (2.0, 4.0)))


def test_find_wall_does_not_change_the_frame_it_is_given() -> None:
    frame = _step_frame()
    before = frame.copy()
    find_wall(frame, outward_xy=(1.0, 0.0), matrix=_matrix(2.0, 0.0, False))
    assert np.array_equal(frame, before)
    assert frame.dtype == np.uint16


# --- wall_points_um, along_um, walk_offsets_um -----------------------------------


def test_wall_points_agree_with_stage_of_pixel() -> None:
    # A rotated, mirrored matrix on a non-square frame, points off the axes.
    matrix = _matrix(0.7, 37.0, True)
    shape = (300, 450)
    rng = np.random.default_rng(1)
    points = rng.uniform((0.0, 0.0), (449.0, 299.0), size=(25, 2))
    mapped = wall_points_um(points, 1234.5, -987.6, matrix=matrix, image_shape_px=shape)
    assert mapped.shape == (25, 2)
    for (col, row), (x, y) in zip(points, mapped, strict=True):
        expected = stage_of_pixel(
            1234.5, -987.6, float(col), float(row), matrix=matrix, image_shape_px=shape
        )
        assert (x, y) == pytest.approx(expected, abs=1e-9)


def test_wall_points_refuse_points_that_are_not_a_column_pair() -> None:
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        wall_points_um(
            np.zeros((4, 3)),
            0.0,
            0.0,
            matrix=_matrix(1.0, 0.0, False),
            image_shape_px=(8, 8),
        )


def test_along_is_the_median_projection_on_the_outward_direction() -> None:
    points = np.array([[10.0, 5.0], [11.0, -7.0], [90.0, 0.0]])  # one stray point
    assert along_um(points, (4.0, 100.0), (1.0, 0.0)) == pytest.approx(7.0)
    assert along_um(points, (4.0, 100.0), (0.0, -1.0)) == pytest.approx(100.0)
    with pytest.raises(ValueError, match="n > 0"):
        along_um(np.zeros((0, 2)), (0.0, 0.0), (1.0, 0.0))


def test_walk_offsets_alternate_and_stay_within_the_search() -> None:
    offsets = walk_offsets_um(7800.0, 3900.0, 1165.0)
    assert offsets == (
        7800.0,
        7800.0 + 1165.0,
        7800.0 - 1165.0,
        7800.0 + 2 * 1165.0,
        7800.0 - 2 * 1165.0,
        7800.0 + 3 * 1165.0,
        7800.0 - 3 * 1165.0,
    )
    assert walk_offsets_um(100.0, 10.0, 25.0) == (100.0,)  # a step beyond the search


@given(
    radius=st.floats(1.0, 20_000.0),
    step=st.floats(1.0, 5_000.0),
    k=st.integers(0, 40),
    fraction=st.floats(0.0, 0.999),
)
@settings(deadline=None, max_examples=200)
def test_walk_offsets_hold_to_their_length_and_their_bound(
    radius: float, step: float, k: int, fraction: float
) -> None:
    search = (k + fraction) * step if (k or fraction) else step * 0.5
    offsets = walk_offsets_um(radius, search, step)
    assert len(offsets) == 1 + 2 * math.floor(search / step)
    assert offsets[0] == radius
    assert all(abs(d - radius) <= search * (1 + 1e-9) for d in offsets)
    assert len(set(offsets)) == len(offsets)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("which", [0, 1, 2])
def test_walk_offsets_refuse_what_is_not_finite_and_positive(
    bad: float, which: int
) -> None:
    args = [7800.0, 3900.0, 1165.0]
    args[which] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        walk_offsets_um(*args)


def test_a_walk_is_bounded_before_it_is_built() -> None:
    # FM-74: 20 000 µm in steps of 1e-6 µm is 4e10 offsets; the check is on the
    # count, in arithmetic, and no list is started.
    with pytest.raises(ValueError, match=str(MAX_FRAMES)):
        walk_offsets_um(7800.0, 20_000.0, 1e-6)
    assert len(walk_offsets_um(7800.0, 299.0, 1.0)) == 599  # under 600
    assert len(walk_offsets_um(7800.0, 299.9, 1.0)) == 599  # the floor, not 600.8
    with pytest.raises(ValueError, match=str(MAX_FRAMES)):
        walk_offsets_um(7800.0, 300.0, 1.0)  # 601


# --- fit_circle -------------------------------------------------------------------


def _arcs(
    centre: tuple[float, float],
    radius: float,
    *,
    half_width_deg: float = 5.0,
    per_arc: int = 100,
    noise_um: float = 0.0,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Four arcs of a circle, round 0°, 90°, 180° and 270°: what four walls give."""
    rng = np.random.default_rng(seed)
    angles = np.concatenate(
        [
            math.radians(c)
            + np.linspace(-1.0, 1.0, per_arc) * math.radians(half_width_deg)
            for c in (0.0, 90.0, 180.0, 270.0)
        ]
    )
    radii = radius + rng.normal(0.0, noise_um, angles.shape)
    return np.column_stack(
        (centre[0] + radii * np.cos(angles), centre[1] + radii * np.sin(angles))
    )


def test_fit_circle_recovers_a_circle_from_four_arcs() -> None:
    points = _arcs((10_182.0, -10_087.9), 7_650.0, noise_um=0.5, seed=1)
    fit = fit_circle(points, floor_um=2.2)
    assert fit.center_xy_um == pytest.approx((10_182.0, -10_087.9), abs=0.2)
    assert fit.diameter_um == pytest.approx(15_300.0, abs=0.5)
    assert fit.rms_um == pytest.approx(0.5, abs=0.1)
    assert (fit.n_points, fit.n_used) == (400, 400)


@given(
    cx=st.floats(-100_000.0, 100_000.0),
    cy=st.floats(-100_000.0, 100_000.0),
    radius=st.floats(1_000.0, 10_000.0),
    half_width=st.floats(3.0, 20.0),
)
@settings(deadline=None, max_examples=60)
def test_fit_circle_recovers_any_circle_from_its_four_arcs(
    cx: float, cy: float, radius: float, half_width: float
) -> None:
    fit = fit_circle(_arcs((cx, cy), radius, half_width_deg=half_width), floor_um=1.0)
    assert math.dist(fit.center_xy_um, (cx, cy)) < 1e-3
    assert fit.diameter_um == pytest.approx(2 * radius, abs=1e-3)
    assert fit.rms_um < 1e-3


def _rim(centre: tuple[float, float], radius: float, n: int) -> npt.NDArray[np.float64]:
    """``n`` points on a short arc at 45° of a circle: a bubble's or a label's edge."""
    angles = np.radians(np.linspace(40.0, 50.0, n))
    return np.column_stack(
        (centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles))
    )


def test_fit_circle_drops_an_outlying_arc() -> None:
    # A bubble's rim: 10 points 300 µm outside the circle, 2.4 % of the points.
    # They are dropped, the centre does not move by a tenth of a micrometre, and
    # the points kept are counted.
    good = _arcs((5_000.0, 6_000.0), 7_650.0, noise_um=0.5, seed=2)
    fit = fit_circle(
        np.vstack([good, _rim((5_000.0, 6_000.0), 7_950.0, 10)]), floor_um=2.2
    )
    assert math.dist(fit.center_xy_um, (5_000.0, 6_000.0)) < 0.1
    assert fit.diameter_um == pytest.approx(15_300.0, abs=0.5)
    assert (fit.n_points, fit.n_used) == (410, 400)


def test_fit_circle_does_not_hide_a_large_outlying_arc_in_its_rms() -> None:
    # 40 points of 440 (9 %) 1000 µm out pull the first fit enough that the MAD
    # trim keeps them: the fit is wrong by 178 µm. The plugin refuses it by the
    # rms (over 2 % of the radius, 153 µm here), which is what this pins: the
    # error is never a quiet one.
    good = _arcs((5_000.0, 6_000.0), 7_650.0, noise_um=0.5, seed=2)
    fit = fit_circle(
        np.vstack([good, _rim((5_000.0, 6_000.0), 8_650.0, 40)]), floor_um=2.2
    )
    assert math.dist(fit.center_xy_um, (5_000.0, 6_000.0)) > 100.0
    assert fit.rms_um > MAX_RMS_FRACTION * 7_650.0


def test_fit_circle_keeps_what_lies_within_the_floor_of_a_noiseless_circle() -> None:
    # Frames with no noise leave a MAD of almost nothing: a bound of the MAD alone
    # would trim every point that is a pixel off the circle. The floor, about a
    # pixel or two, is the bound then: 10 points 3 µm out are kept, 8 µm out are not.
    good = _arcs((0.0, 0.0), 7_650.0)
    for off_um, kept in ((3.0, 410), (8.0, 400)):
        fit = fit_circle(
            np.vstack([good, _rim((0.0, 0.0), 7_650.0 + off_um, 10)]), floor_um=4.0
        )
        assert (fit.n_points, fit.n_used) == (410, kept)


def test_fit_circle_refuses_collinear_points() -> None:
    line = np.column_stack((np.arange(10.0), 2.0 * np.arange(10.0) + 1.0))
    with pytest.raises(ValueError, match="line"):
        fit_circle(line, floor_um=1.0)
    with pytest.raises(ValueError, match="one point"):
        fit_circle(np.ones((5, 2)), floor_um=1.0)


@pytest.mark.parametrize(
    ("points", "floor", "message"),
    [
        (np.zeros((2, 2)), 1.0, "3 points"),
        (np.zeros((5, 3)), 1.0, r"\(n, 2\)"),
        (np.zeros(5), 1.0, r"\(n, 2\)"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, math.nan]]), 1.0, "finite"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, math.inf]]), 1.0, "finite"),
        (_arcs((0.0, 0.0), 100.0), math.nan, "floor_um"),
        (_arcs((0.0, 0.0), 100.0), -1.0, "floor_um"),
    ],
)
def test_fit_circle_refuses_what_it_cannot_fit(
    points: npt.NDArray[np.float64], floor: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_circle(points, floor_um=floor)


def test_fit_circle_refuses_when_the_trims_leave_fewer_than_three() -> None:
    # Five points scattered with no circle through them, and no floor: the bound
    # of the MAD alone leaves one point near the circle fitted to them. That is a
    # refusal that says so, never a circle through fewer than three points.
    scatter = np.array([[8.1, 18.3], [12.9, 15.4], [9.2, 7.7], [7.8, 6.5], [5.6, 19.2]])
    with pytest.raises(ValueError, match=r"only 1 of 5 points lie near a circle"):
        fit_circle(scatter, floor_um=0.0)


# --- the frame file ---------------------------------------------------------------


def test_a_wall_frame_is_a_tiff_with_its_metadata_in_json() -> None:
    frame = _step_frame()
    meta = {"well": "B3", "side": "+X", "x_um": 10226.7, "y_um": -9737.3}
    data = encode_wall_frame(frame, meta)
    with tifffile.TiffFile(io.BytesIO(data)) as handle:
        assert np.array_equal(handle.asarray(), frame)
        assert handle.asarray().dtype == np.uint16
        description = handle.pages[0].tags["ImageDescription"].value  # type: ignore[union-attr]
    assert json.loads(description) == {"shape": [64, 192], **meta}


# --- the B3 frames of Ti2-E session 2 (private; skipped when absent) --------------

SESSION2 = Path(
    os.environ.get("SMC_SESSION2_DIR")
    or Path(__file__).resolve().parents[2] / "local/surveys/nikon-ti2/session2"
)
#: Each frame's side by the stage's directions, and where the stage was (§17.1):
#: the file names are the image's sides, and the camera is turned 180°.
B3_FRAMES = {
    "+X": ("B3-left.tif", (18_067.4, -9_737.3)),
    "+Y": ("B3-bottom.tif", (10_226.7, -2_337.6)),
    "-X": ("B3-right.tif", (2_468.0, -9_737.3)),
    "-Y": ("B3-top.tif", (10_226.7, -17_738.5)),
}
B3_MATRIX: Matrix2 = ((-1.1, 0.0), (0.0, 1.1))
needs_b3 = pytest.mark.skipif(
    not all((SESSION2 / name).is_file() for name, _ in B3_FRAMES.values()),
    reason=(
        f"the B3 frames are private and not in this checkout: {SESSION2} has not "
        "B3-left, -right, -top and -bottom (set SMC_SESSION2_DIR)"
    ),
)


def _b3_frame(side: str) -> npt.NDArray[np.uint16]:
    return np.asarray(tifffile.imread(SESSION2 / B3_FRAMES[side][0]))


@needs_b3
def test_b3s_centre_agrees_with_the_midpoint_of_its_walls() -> None:
    # The session's own table (#334): the walls' half-intensity edges, taken by
    # eye on the frames, put B3's centre at (10 184.7, -10 091.4) µm. The eye read
    # the half-intensity edge on the meniscus's ramp, which is wider on some sides
    # than on others, and the fit takes the half level of the fall to the plastic
    # (#409): its centre is within 50 µm of the eye's (28.4 measured; the midpoint
    # rule was at 4.4) and its size within 1 % of the plate's 15 600 µm (15 561
    # measured; 15 298 before, when the edge lay on the ramp).
    kept = []
    for side in SIDES:
        frame = _b3_frame(side)
        edge = find_wall(frame, outward_xy=OUTWARD[side], matrix=B3_MATRIX)
        assert edge is not None, f"no wall on B3's {side} side"
        assert len(edge.points_px) >= 0.9 * edge.n_lines
        kept.append(
            wall_points_um(
                edge.points_px,
                *B3_FRAMES[side][1],
                matrix=B3_MATRIX,
                image_shape_px=(frame.shape[0], frame.shape[1]),
            )
        )
    fit = fit_circle(np.vstack(kept), floor_um=2.2)
    assert math.dist(fit.center_xy_um, (10_184.7, -10_091.4)) < 50.0
    assert fit.diameter_um == pytest.approx(15_600.0, rel=0.01)
    assert fit.n_used >= 0.95 * fit.n_points


@needs_b3
def test_b3_with_an_upright_camera_finds_no_wall() -> None:
    # The matrix of a camera that is not turned: every line points into the
    # plastic and starts dark. The frames are refused, not read a well-width off.
    upright = pixel_matrix(1.1)
    for side in SIDES:
        assert (
            find_wall(_b3_frame(side), outward_xy=OUTWARD[side], matrix=upright) is None
        )


# --- the frames of Ti2-E session 3 (private; skipped when absent) -----------------

SESSION3 = Path(
    os.environ.get("SMC_SESSION3_DIR")
    or Path(__file__).resolve().parents[2] / "local/scans/2026-11-10/plate-walls"
)
#: The 4x run that failed on B5 (its frames are the ones centred on what the walk
#: took for each wall) and the 10x run that found the walls of B3, B5 and C3.
RUN_4X, RUN_10X = "d5d32a2051ad", "cdde0e648acc"
#: The session's own calibrations, in µm per pixel (§17.1).
MATRIX_4X: Matrix2 = ((-2.7376, 0.1217), (0.1225, 2.7351))
MATRIX_10X: Matrix2 = ((-1.12426, 0.04985), (0.05022, 1.12245))
#: Half the plate's well diameter, for the bound on the circle's rms.
HALF_WELL_UM = 7_800.0


def _session3_frames(run: str, well: str) -> list[Path]:
    return [SESSION3 / run / f"{well}_{side}.tif" for side in SIDES]


def _skip_without_session3(run: str, well: str) -> None:
    """Skip, with the reason, when the well's four frames are not there (FM-50): a parameter reads its own frames."""
    if not all(path.is_file() for path in _session3_frames(run, well)):
        pytest.skip(
            "the frames of session 3 are private and not in this checkout: "
            f"{SESSION3 / run} has not {well}'s four frames (set SMC_SESSION3_DIR)"
        )


def _session3_fit(run: str, well: str, matrix: Matrix2) -> CircleFit:
    """A well's circle from its four saved frames, each at the stage position its own metadata records."""
    kept = []
    for side, path in zip(SIDES, _session3_frames(run, well), strict=True):
        with tifffile.TiffFile(path) as handle:
            frame = np.asarray(handle.asarray())
            shaped = handle.shaped_metadata
        assert shaped is not None, f"{path.name} has no metadata"
        edge = find_wall(frame, outward_xy=OUTWARD[side], matrix=matrix)
        assert edge is not None, f"no wall on {well}'s {side} side"
        assert len(edge.points_px) >= 0.9 * edge.n_lines, f"{well} {side}"
        kept.append(
            wall_points_um(
                edge.points_px,
                float(shaped[0]["x_um"]),
                float(shaped[0]["y_um"]),
                matrix=matrix,
                image_shape_px=(frame.shape[0], frame.shape[1]),
            )
        )
    # The floor of the trims is two pixels, as the plugin gives it.
    return fit_circle(
        np.vstack(kept), floor_um=2.0 * math.hypot(matrix[0][0], matrix[1][0])
    )


def test_b5_at_the_4x_is_measured_at_its_plastic_walls() -> None:
    # F7 of #405: the walk took B5's meniscus for its walls and fitted a circle of
    # 13 055 µm, rms 663 µm, which the size check refused. On the saved frames the
    # plastic's wall gives 15 436 µm (23.5 µm rms) and a centre 23 µm from B5's
    # centre at the 10x, (-29 004, -10 053) µm.
    _skip_without_session3(RUN_4X, "B5")
    fit = _session3_fit(RUN_4X, "B5", MATRIX_4X)
    assert fit.diameter_um == pytest.approx(15_600.0, rel=0.02)
    assert fit.rms_um < MAX_RMS_FRACTION * HALF_WELL_UM
    assert math.dist(fit.center_xy_um, (-29_004.0, -10_053.0)) < 50.0


@pytest.mark.parametrize(
    ("well", "midpoint_centre"),
    [
        ("B3", (9_981.0, -10_140.0)),
        ("B5", (-29_009.0, -10_057.0)),
        ("C3", (10_068.0, 9_394.0)),
    ],
)
def test_session_3s_10x_centres_stay_near_the_midpoint_rules(
    well: str, midpoint_centre: tuple[float, float]
) -> None:
    # The 10x run found these three wells with the midpoint rule, on the
    # meniscus's ramp. The plastic's wall moves each centre by 6 to 26 µm and the
    # diameter towards the plate's 15 600 µm: the 10x behaviour holds (#409).
    _skip_without_session3(RUN_10X, well)
    fit = _session3_fit(RUN_10X, well, MATRIX_10X)
    assert fit.diameter_um == pytest.approx(15_600.0, rel=0.01)
    assert fit.rms_um < MAX_RMS_FRACTION * HALF_WELL_UM
    assert math.dist(fit.center_xy_um, midpoint_centre) < 40.0
