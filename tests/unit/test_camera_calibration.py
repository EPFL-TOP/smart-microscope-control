"""Camera calibration's algorithm: shifts between frames, and the matrix they give (m3 §4.1).

Nothing here has a microscope. The frames are numpy noise or what the synthetic
plate renders at a pose, whose pixel size, rotation and mirror are known, so
the recovered values are checked against truth and not against themselves.
Frames are not square on purpose (FM-73): a swapped ``(height, width)`` is
invisible on a square one.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.camera_calibration import algorithm
from smc.plugins.camera_calibration.algorithm import (
    MIN_RESPONSE,
    UPSAMPLE,
    Fit,
    Shift,
    fit,
    measure_shift,
)
from smc.plugins.geometry import Matrix2
from smc.testing.synthetic import PlateSample

SHAPE = (384, 512)  # (height, width)


def _angle_difference_deg(a: float, b: float) -> float:
    """The smaller angle between two directions, so 179.9 and -179.9 are 0.2 apart."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


def _probes_of(matrix: Matrix2, step_um: float) -> tuple[Shift, Shift]:
    """The shifts a camera of this matrix measures for a +X and a +Y move: ``S = -inv(M) * step``."""
    shifts = -np.linalg.inv(np.array(matrix)) * step_um
    return (
        Shift(float(shifts[0][0]), float(shifts[1][0]), 1.0),
        Shift(float(shifts[0][1]), float(shifts[1][1]), 1.0),
    )


def _rendered_fit(
    sample: PlateSample,
    *,
    pixel_size_um: float,
    step_um: float,
    rotation_deg: float = 0.0,
    mirrored: bool = False,
) -> Fit:
    """What the calibration does, without a stage: three frames of the sample, and the fit."""

    def frame(x_um: float, y_um: float) -> np.ndarray:
        return sample.render(
            x_um,
            y_um,
            0.0,
            shape=SHAPE,
            pixel_size_um=pixel_size_um,
            camera_rotation_deg=rotation_deg,
            mirrored=mirrored,
        )

    reference = frame(0.0, 0.0)
    return fit(
        measure_shift(reference, frame(step_um, 0.0)),
        measure_shift(reference, frame(0.0, step_um)),
        step_um=step_um,
        image_shape_px=SHAPE,
    )


# ------------------------------------------------------------ the shift


def test_a_known_shift_is_recovered() -> None:
    a = np.random.default_rng(0).normal(size=(200, 300))
    b = np.roll(np.roll(a, 7, axis=0), -13, axis=1)

    shift = measure_shift(a, b)

    # The displacement of the content: rolled down 7 rows and left 13 columns.
    assert shift.d_row_px == pytest.approx(7.0, abs=0.05)
    assert shift.d_col_px == pytest.approx(-13.0, abs=0.05)
    assert shift.response > 0.99


def test_confidence_is_measured_after_alignment() -> None:
    """Regression of ``nikon-control``: correlating the raw frames scored a good measurement 0.

    Two views of one textured sample, offset, share nothing pixel for pixel, so
    an unaligned correlation would reject every real calibration.
    """
    image = np.random.default_rng(0).normal(size=(200, 300))
    a, b = image[:, :250], image[:, 40:290]

    raw = abs(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    shift = measure_shift(a, b)

    assert raw < 0.05  # the raw correlation says nothing: the test can fail
    assert shift.d_col_px == pytest.approx(-40.0, abs=0.05)
    assert shift.response > 0.9


def test_a_blank_field_scores_zero() -> None:
    blank = np.full((128, 160), 1234, dtype=np.uint16)

    assert measure_shift(blank, blank) == Shift(0.0, 0.0, 0.0)


def test_one_blank_frame_next_to_a_textured_one_has_no_shift_to_find() -> None:
    """A probe that lands on an empty margin: either side of the pair may be the blank one."""
    texture = np.random.default_rng(5).normal(size=(128, 160))
    blank = np.zeros_like(texture)

    assert measure_shift(texture, blank) == Shift(0.0, 0.0, 0.0)
    assert measure_shift(blank, texture) == Shift(0.0, 0.0, 0.0)


def _forced_shift(monkeypatch: pytest.MonkeyPatch, d_row: float, d_col: float) -> None:
    """Make the correlation answer ``(d_row, d_col)`` as the displacement of the content.

    skimage reports the shift that registers ``moved`` onto ``reference``, the
    negation of the displacement, so that is what the stand-in returns.
    """
    monkeypatch.setattr(
        algorithm,
        "_phase_cross_correlation",
        lambda reference, moved, **kwargs: (np.array([-d_row, -d_col]), 0.0, 0.0),
    )


def test_a_constant_overlap_scores_zero_not_perfect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``np.corrcoef`` of a constant overlap is ``nan``, and ``min(1.0, nan)`` is 1.0.

    Texture only in the first ten columns, and a shift of 20 columns: the
    overlap of the moved frame is blank, and would score a perfect response and
    defeat the refusal of a wrapped shift (FM-120) if it were not zeroed.
    """
    a = np.zeros((128, 160))
    a[:, :10] = np.random.default_rng(0).normal(size=(128, 10))
    _forced_shift(monkeypatch, d_row=0.0, d_col=20.0)

    shift = measure_shift(a, a.copy())

    assert shift.d_col_px == 20.0
    assert shift.response == 0.0


def test_an_inverted_contrast_agrees_as_much_as_the_same_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response is an absolute correlation: a frame and its negative are as alike."""
    a = np.random.default_rng(0).normal(size=(128, 160))
    _forced_shift(monkeypatch, d_row=0.0, d_col=0.0)

    assert measure_shift(a, -a).response > 0.99


def test_an_overlap_of_under_eight_columns_scores_zero() -> None:
    """The width side of the guard: seven columns agreeing say nothing."""
    a = np.random.default_rng(6).normal(size=(100, 12))
    b = np.roll(a, 5, axis=1)

    shift = measure_shift(a, b)

    assert shift.d_col_px == pytest.approx(5.0, abs=0.05)
    assert shift.response == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_the_frames_are_float64_when_they_reach_the_correlation(
    dtype: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-75: the cast comes first, so that no arithmetic runs in the camera's dtype."""
    seen: list[tuple[np.dtype, np.dtype]] = []

    def spy(reference: np.ndarray, moved: np.ndarray, **kwargs: object):
        seen.append((reference.dtype, moved.dtype))
        return np.array([0.0, 0.0]), 0.0, 0.0

    monkeypatch.setattr(algorithm, "_phase_cross_correlation", spy)
    frame = (np.random.default_rng(1).uniform(0.0, 1000.0, size=(64, 80))).astype(dtype)

    measure_shift(frame, frame.copy())

    assert seen == [(np.dtype(np.float64), np.dtype(np.float64))]


def test_an_overlap_of_under_eight_pixels_scores_zero() -> None:
    """Seven rows agreeing say nothing: the frames are 12 rows high and moved by 5."""
    a = np.random.default_rng(4).normal(size=(12, 100))
    b = np.roll(a, 5, axis=0)

    shift = measure_shift(a, b)

    assert shift.d_row_px == pytest.approx(5.0, abs=0.05)  # found, but not vouched for
    assert shift.response == 0.0


@pytest.mark.parametrize("level", [0.1, 1234.567])
def test_a_constant_float_frame_is_flat_whatever_its_rounding(level: float) -> None:
    """The std of a constant float frame is rounding noise, not 0 (2e-13 at 1234.567)."""
    blank = np.full((128, 160), level)
    assert blank.std() > 0.0  # the premise: an exact `std == 0` would not see it

    assert measure_shift(blank, blank) == Shift(0.0, 0.0, 0.0)


def test_a_frame_that_is_not_2d_or_of_another_shape_is_refused() -> None:
    flat = np.zeros((16, 16))
    with pytest.raises(ValueError, match="2-D"):
        measure_shift(np.zeros((4, 16, 16)), flat)
    with pytest.raises(ValueError, match="same shape"):
        measure_shift(np.zeros((16, 16)), np.zeros((16, 20)))


def test_a_frame_with_a_value_that_is_not_finite_is_refused() -> None:
    """A float frame with a NaN would otherwise score ``nan``, which no threshold catches."""
    good = np.random.default_rng(0).normal(size=(64, 80))
    bad = good.copy()
    bad[3, 5] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        measure_shift(good, bad)
    with pytest.raises(ValueError, match="not finite"):
        measure_shift(bad, good)


def test_a_float32_and_an_integer_frame_are_cast_before_any_arithmetic() -> None:
    """FM-75: an unsigned frame minus its mean wraps unless it is a float first."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4000, size=(120, 160), dtype=np.uint16)
    b = np.roll(a, (5, -9), axis=(0, 1))

    shift = measure_shift(a, b)

    assert shift.d_row_px == pytest.approx(5.0, abs=0.05)
    assert shift.d_col_px == pytest.approx(-9.0, abs=0.05)
    assert shift.response > 0.99


def test_the_upsample_is_what_gives_a_sub_pixel_answer() -> None:
    assert UPSAMPLE == 10
    a = np.random.default_rng(2).normal(size=(128, 160))
    # 2.5 px along columns, by Fourier shift: a known sub-pixel displacement.
    spectrum = np.fft.fft2(a) * np.exp(
        -2j * np.pi * 2.5 * np.fft.fftfreq(a.shape[1])[None, :]
    )
    b = np.fft.ifft2(spectrum).real

    shift = measure_shift(a, b)

    assert shift.d_col_px == pytest.approx(2.5, abs=0.15)
    assert shift.d_row_px == pytest.approx(0.0, abs=0.15)


# ------------------------------------------------------- the calibration


@pytest.mark.parametrize(
    ("pixel_size_um", "rotation_deg", "mirrored"),
    [
        (1.0, 0.0, False),
        (1.0, 30.0, True),
        (0.5, -12.0, True),
        (0.25, 179.5, False),
    ],
)
def test_pixel_size_and_handedness_are_recovered(
    pixel_size_um: float, rotation_deg: float, mirrored: bool
) -> None:
    step_um = 0.25 * SHAPE[0] * pixel_size_um

    result = _rendered_fit(
        PlateSample(),
        pixel_size_um=pixel_size_um,
        step_um=step_um,
        rotation_deg=rotation_deg,
        mirrored=mirrored,
    )

    assert result.pixel_size_um == pytest.approx(pixel_size_um, rel=0.01)
    assert _angle_difference_deg(result.rotation_deg, rotation_deg) < 0.2
    assert result.mirrored is mirrored
    assert result.trustworthy
    assert result.notes == ()


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("rotation_deg", [7.0, -170.0, 180.0, -180.0])
def test_rotation_is_recovered(rotation_deg: float, mirrored: bool) -> None:
    """Pure arithmetic: the matrix is built, turned into the shifts it implies, and fitted.

    ``-180.0`` is built from ``sin(-pi)``, a tiny negative number, so that
    ``atan2`` gives exactly ``-pi``: the one input on which the range's open
    end is decided.
    """
    matrix = _matrix(0.65, rotation_deg, mirrored)
    shift_x, shift_y = _probes_of(matrix, 96.0)

    result = fit(shift_x, shift_y, step_um=96.0, image_shape_px=SHAPE)

    # The range is (-180, 180]: the same direction is never written -180.
    expected = 180.0 if abs(rotation_deg) == 180.0 else rotation_deg
    assert result.rotation_deg == pytest.approx(expected, abs=1e-6)
    assert result.mirrored is mirrored
    assert result.pixel_size_um == pytest.approx(0.65, abs=1e-9)
    assert result.residual_um < 1e-9
    assert result.trustworthy
    if expected == 180.0:
        assert math.copysign(1.0, result.rotation_deg) == 1.0
        assert result.rotation_deg == 180.0


@given(
    pixel_size_um=st.floats(0.05, 5.0),
    rotation_deg=st.floats(-179.99, 179.99),
    mirrored=st.booleans(),
)
def test_fit_inverts_the_shifts_of_any_rigid_camera(
    pixel_size_um: float, rotation_deg: float, mirrored: bool
) -> None:
    matrix = _matrix(pixel_size_um, rotation_deg, mirrored)
    shift_x, shift_y = _probes_of(matrix, 40.0)

    result = fit(shift_x, shift_y, step_um=40.0, image_shape_px=SHAPE)

    assert -180.0 < result.rotation_deg <= 180.0
    assert _angle_difference_deg(result.rotation_deg, rotation_deg) < 1e-6
    assert result.mirrored is mirrored
    assert result.pixel_size_um == pytest.approx(pixel_size_um, rel=1e-9)
    for got_row, want_row in zip(result.matrix_um_per_px, matrix, strict=True):
        assert got_row == pytest.approx(want_row, abs=1e-9)


def test_the_residual_says_how_far_the_camera_is_from_rigid() -> None:
    """A camera whose X pixels are 10 % larger than its Y pixels is not a rotation."""
    stretched = ((0.55, 0.0), (0.0, -0.5))
    shift_x, shift_y = _probes_of(stretched, 50.0)

    result = fit(shift_x, shift_y, step_um=50.0, image_shape_px=(200, 300))

    # The fitted size is sqrt(|det|); the X entry is the one furthest from it.
    # Times half the larger side, so the residual is the edge's error in µm.
    off_um_per_px = 0.55 - math.sqrt(0.55 * 0.5)
    assert result.residual_um == pytest.approx(off_um_per_px * 300 / 2, rel=1e-6)
    assert result.notes == ()


def test_a_stage_that_does_not_move_is_caught() -> None:
    texture = np.random.default_rng(3).normal(size=(200, 300))

    still = measure_shift(texture, texture)
    result = fit(still, still, step_um=50.0, image_shape_px=(200, 300))

    assert still.response > 0.99  # the frames agree: the stage is what failed
    assert any("parallel shifts" in note for note in result.notes)
    assert result.matrix_um_per_px == ((0.0, 0.0), (0.0, 0.0))
    assert (result.pixel_size_um, result.rotation_deg, result.mirrored) == (
        0.0,
        0.0,
        False,
    )
    assert result.residual_um == 0.0
    assert result.trustworthy is False


def test_a_shift_beyond_half_the_frame_is_refused_not_returned() -> None:
    """FM-120: skimage reports each axis in [-N/2, N/2), so 200 rows of 384 read -184.

    The fit would then return a confident wrong matrix; the response, measured
    at the wrapped shift, is that of unrelated content, and refuses it.
    """
    result = _rendered_fit(PlateSample(), pixel_size_um=0.25, step_um=50.0)

    assert not result.trustworthy
    assert any("barely correlate" in note for note in result.notes)


@pytest.mark.parametrize("weak_probe", ["x", "y"])
def test_a_probe_that_barely_correlates_makes_the_fit_untrustworthy(
    weak_probe: str,
) -> None:
    """Either probe on its own: a camera turned 90 degrees wraps the +X probe first."""
    good_x, good_y = Shift(-10.0, 0.0, 0.9), Shift(0.0, 10.0, 0.9)
    shift_x = (
        replace(good_x, response=MIN_RESPONSE / 2) if weak_probe == "x" else good_x
    )
    shift_y = (
        replace(good_y, response=MIN_RESPONSE / 2) if weak_probe == "y" else good_y
    )

    result = fit(shift_x, shift_y, step_um=20.0, image_shape_px=SHAPE)

    assert any("barely correlate" in note for note in result.notes)
    assert not result.trustworthy


@pytest.mark.parametrize("unknown_probe", ["x", "y"])
def test_a_response_that_is_nan_is_not_trusted(unknown_probe: str) -> None:
    """``nan < MIN_RESPONSE`` is False: the check must be written so that nan fails it."""
    good_x, good_y = Shift(-10.0, 0.0, 0.9), Shift(0.0, 10.0, 0.9)
    shift_x = replace(good_x, response=math.nan) if unknown_probe == "x" else good_x
    shift_y = replace(good_y, response=math.nan) if unknown_probe == "y" else good_y

    result = fit(shift_x, shift_y, step_um=20.0, image_shape_px=SHAPE)

    assert not result.trustworthy


def _residual_of(matrix: Matrix2, image_shape_px: tuple[int, int]) -> float:
    """The residual of m3 §4.1, written as matrix products and not as the code's entries."""
    m = np.array(matrix)
    determinant = np.linalg.det(m)
    pixel_size = math.sqrt(abs(determinant))
    mirrored = determinant > 0.0
    a = m @ np.diag([1.0, -1.0])
    if mirrored:
        a[:, 0] *= -1.0
    theta = math.atan2(a[1][0], a[0][0])
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    ideal = pixel_size * rotation @ np.diag([-1.0 if mirrored else 1.0, -1.0])
    return float(np.abs(m - ideal).max()) * max(image_shape_px) / 2.0


@pytest.mark.parametrize(
    "entries",
    [
        # Each is a matrix on which a different entry of |M - ideal| is the largest
        # (the first four are upright, the last four mirrored), so that a residual
        # that ignored any one entry would be wrong on one of them.
        (0.79, -0.2, -0.46, -0.24),
        (0.7, -0.21, -0.04, -0.71),
        (-0.58, 0.09, -0.66, 0.23),
        (-0.14, 0.17, 0.48, 0.91),
        (0.93, 0.42, -0.57, 0.09),
        (-0.81, -0.13, -0.04, -0.68),
        (-0.22, -0.24, 0.82, -0.21),
        (0.4, -0.42, 0.74, -0.45),
    ],
)
def test_the_residual_counts_every_entry_of_the_matrix(
    entries: tuple[float, float, float, float],
) -> None:
    matrix = ((entries[0], entries[1]), (entries[2], entries[3]))
    shift_x, shift_y = _probes_of(matrix, 50.0)

    result = fit(shift_x, shift_y, step_um=50.0, image_shape_px=(200, 300))

    assert result.residual_um == pytest.approx(
        _residual_of(matrix, (200, 300)), rel=1e-6
    )
    assert result.residual_um > 1.0  # none of these is a rigid camera


def test_both_notes_can_be_present_at_once() -> None:
    """A blank field without noise gives zero shifts and zero responses: two reasons."""
    blank = Shift(0.0, 0.0, 0.0)

    result = fit(blank, blank, step_um=20.0, image_shape_px=SHAPE)

    assert len(result.notes) == 2
    assert any("barely correlate" in note for note in result.notes)
    assert any("parallel shifts" in note for note in result.notes)


@pytest.mark.parametrize("step_um", [0.0, -5.0, float("nan"), float("inf")])
def test_a_step_that_is_not_finite_and_positive_is_refused(step_um: float) -> None:
    shift = Shift(1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="step_um"):
        fit(shift, shift, step_um=step_um, image_shape_px=SHAPE)
