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
    MAX_RESIDUAL_SHARE,
    MIN_FEATURE_FRACTION,
    MIN_RESPONSE,
    MIN_SIGNAL_SPREAD_COUNTS,
    SMALL_STEP_FRACTION,
    UPSAMPLE,
    Fit,
    Shift,
    carries_signal,
    fit,
    measure_shift,
    precision_reasons,
    signal_floor_counts,
    signal_spread,
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


RIGIDITY_NOTE = (
    "the shifts do not fit a rigid camera: did each axis move its full command?"
)


@pytest.mark.parametrize(
    ("shift_x", "shift_y", "matrix"),
    [
        # The +X probe reads 10 % and 1 % of the 96 px that a full move gives,
        # the +Y probe is full: an X axis that executes a fraction of its command.
        (Shift(-9.6, 0.0, 1.0), Shift(0.0, 96.0, 1.0), ((10.0, 0.0), (0.0, -1.0))),
        (Shift(-0.96, 0.0, 1.0), Shift(0.0, 96.0, 1.0), ((100.0, 0.0), (0.0, -1.0))),
        # Y did not move at all and its probe reads 0.1 px of noise: the near
        # parallel case that the absolute determinant of 1e-12 lets through.
        (Shift(-96.0, 0.0, 1.0), Shift(0.0, 0.1, 1.0), ((1.0, 0.0), (0.0, -960.0))),
    ],
    ids=["x-at-10-percent", "x-at-1-percent", "y-stuck-with-noise"],
)
def test_an_axis_that_moves_a_fraction_of_its_command_is_refused(
    shift_x: Shift, shift_y: Shift, matrix: Matrix2
) -> None:
    """The matrix is not a rigid camera: its largest entry off a rotation reaches the pixel size.

    ps 1.0, upright, step 96 µm, a 384 x 512 frame. The response is high in
    every case (the frames do agree), which is why only the rigidity refuses.
    """
    result = fit(shift_x, shift_y, step_um=96.0, image_shape_px=SHAPE)

    assert RIGIDITY_NOTE in result.notes
    assert not any("parallel shifts" in note for note in result.notes)
    assert result.trustworthy is False
    # The matrix, the angle and the size are returned as computed, not zeroed.
    for got_row, want_row in zip(result.matrix_um_per_px, matrix, strict=True):
        assert got_row == pytest.approx(want_row, rel=1e-9, abs=1e-9)
    assert result.rotation_deg == 0.0
    assert result.mirrored is False
    assert result.pixel_size_um == pytest.approx(math.sqrt(abs(np.linalg.det(matrix))))
    assert result.residual_um >= result.pixel_size_um * max(SHAPE) / 2


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("rotation_deg", [0.0, 30.0, 90.0, 135.0])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_an_axis_at_ten_percent_is_refused_at_any_pose_of_the_camera(
    axis: str, rotation_deg: float, mirrored: bool
) -> None:
    """FM-73: off the axes too. A turned or mirrored camera puts both components in each probe."""
    shift_x, shift_y = _probes_of(_matrix(1.0, rotation_deg, mirrored), 96.0)
    slow = shift_x if axis == "x" else shift_y
    slow = replace(slow, d_col_px=0.1 * slow.d_col_px, d_row_px=0.1 * slow.d_row_px)
    if axis == "x":
        shift_x = slow
    else:
        shift_y = slow

    result = fit(shift_x, shift_y, step_um=96.0, image_shape_px=SHAPE)

    assert RIGIDITY_NOTE in result.notes
    assert result.trustworthy is False
    # Returned as computed, not zeroed: the handedness is the camera's, and a
    # turned camera's non-rigid matrix still has an angle.
    assert result.mirrored is mirrored
    if rotation_deg != 0.0:
        assert result.rotation_deg != 0.0


def test_a_fit_whose_matrix_is_nan_is_not_trusted() -> None:
    """The note is written ``not worst < pixel_size``, so that ``nan`` fails it.

    ``measure_shift`` cannot give an infinite shift, but ``fit`` is public: with
    ``worst >= pixel_size`` a nan matrix would carry no note and be trustworthy.
    """
    result = fit(
        Shift(-math.inf, 0.0, 1.0),
        Shift(0.0, math.inf, 1.0),
        step_um=96.0,
        image_shape_px=SHAPE,
    )

    assert math.isnan(result.pixel_size_um)
    assert result.trustworthy is False


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    ("fraction", "refused"),
    [(0.30, False), (0.26, False), (0.25, True), (0.24, True), (0.20, True)],
    ids=["30-percent", "26-percent", "25-percent", "24-percent", "20-percent"],
)
def test_the_rigidity_note_is_decided_at_the_pixel_size(
    axis: str, fraction: float, refused: bool
) -> None:
    """An axis at a fraction f of its command makes ``|M - ideal|`` reach the pixel size at f = 1/4.

    For a matrix ``diag(1/f, -1)`` the largest entry off a rotation is
    ``1/f - sqrt(1/f)`` and the pixel size ``sqrt(1/f)``: they meet at f = 0.25
    (exactly, in floating point, with a step of 96 and shifts of 24 and 96 px),
    so a tolerance looser or tighter than the pixel size changes one of these
    three cases, on either axis.
    """
    full = 96.0
    shift_x = Shift(-full * (fraction if axis == "x" else 1.0), 0.0, 1.0)
    shift_y = Shift(0.0, full * (fraction if axis == "y" else 1.0), 1.0)

    result = fit(shift_x, shift_y, step_um=96.0, image_shape_px=SHAPE)

    assert any("rigid camera" in note for note in result.notes) is refused
    assert result.trustworthy is (not refused)


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


# -- a frame with no signal (#408, F8): the light is off the camera's path


def _dark_frame(*, hot_pixel: bool = False) -> np.ndarray:
    """What the Prime 95B gave with no light on it (session 2's ``smc snap``).

    Measured on the two real frames: mean 100.5, standard deviation 2.3, so
    95 to 106 between the 1st and the 99th percentile; a column pattern is
    added, since the fixed pattern is what made the dark frames correlate well
    at a zero shift. One pixel at 65535 stands for a hot or stuck pixel.
    """
    rng = np.random.default_rng(408)
    frame = 100.5 + rng.normal(0.0, 2.3, SHAPE) + rng.normal(0.0, 1.0, SHAPE[1])
    frame = np.rint(frame).astype(np.uint16)
    if hot_pixel:
        frame[10, 20] = 65535
    return frame


@pytest.mark.parametrize(
    "frame",
    [
        _dark_frame(),
        _dark_frame(hot_pixel=True),
        np.full(SHAPE, 65535, dtype=np.uint16),
        np.zeros(SHAPE, dtype=np.uint16),
    ],
    ids=["dark", "dark-with-a-hot-pixel", "saturated", "zero"],
)
def test_a_measured_dark_frame_carries_no_signal(frame: np.ndarray) -> None:
    """The hot pixel makes ``max - min`` 65000: only the percentiles see a dark frame."""
    assert signal_spread(frame) < MIN_SIGNAL_SPREAD_COUNTS
    assert not carries_signal(frame)


def test_a_frame_with_a_sample_carries_signal() -> None:
    """The synthetic plate, and a frame spanning 177 counts: the dimmest lit one of 200 real frames."""
    rendered = PlateSample().render(0.0, 0.0, 0.0, shape=SHAPE, pixel_size_um=1.0)
    dimmest = np.linspace(1000.0, 1000.0 + 177.0 / 0.98, SHAPE[0] * SHAPE[1])

    assert carries_signal(rendered)
    assert carries_signal(dimmest.reshape(SHAPE))
    assert signal_spread(dimmest.reshape(SHAPE)) == pytest.approx(177.0, rel=1e-3)


def _step_frame(spread: float) -> np.ndarray:
    """Half the frame at 0 and half at ``spread``: its 1st to 99th percentile spans exactly that.

    The median is half of it, so no pixel lies a floor away from it: the
    percentiles alone decide.
    """
    frame = np.zeros(SHAPE, dtype=np.float64)
    frame[:, SHAPE[1] // 2 :] = spread
    return frame


@pytest.mark.parametrize(
    ("spread", "expected"), [(49.0, False), (50.0, True), (51.0, True)]
)
def test_the_signal_floor_is_decided_at_exactly_fifty_counts(
    spread: float, expected: bool
) -> None:
    """The boundary itself: ``>=`` and not ``>``, in the one function every caller asks."""
    frame = _step_frame(spread)

    assert signal_spread(frame) == spread
    assert carries_signal(frame) is expected


def test_the_constants_are_the_measured_numbers() -> None:
    """Real dark frames span 11 counts and the dimmest lit one 177: 50 sits between, 4 and 3.5 times away."""
    assert MIN_SIGNAL_SPREAD_COUNTS == 50.0
    assert 4.0 * 11.0 < MIN_SIGNAL_SPREAD_COUNTS < 177.0 / 3.0
    assert MIN_FEATURE_FRACTION == 1e-4
    assert SMALL_STEP_FRACTION == 0.1
    assert MAX_RESIDUAL_SHARE == 0.05


@pytest.mark.parametrize(
    ("bit_depth", "floor"), [(16, 50.0), (12, 3.125), (8, 0.1953125), (24, 50.0)]
)
def test_the_floor_scales_with_the_cameras_bit_depth_and_never_above_sixteen(
    bit_depth: int, floor: float
) -> None:
    assert signal_floor_counts(bit_depth) == pytest.approx(floor)


def test_a_lit_frame_of_a_12_bit_mode_carries_signal_and_its_dark_frame_does_not() -> (
    None
):
    """The same light reads 16 times fewer counts: against the 16-bit floor it would be refused."""
    rendered = PlateSample().render(0.0, 0.0, 0.0, shape=SHAPE, pixel_size_um=1.0)
    lit_12_bit = rendered // 16
    dark_12_bit = _dark_frame() // 16

    assert signal_spread(lit_12_bit) < MIN_SIGNAL_SPREAD_COUNTS
    assert not carries_signal(lit_12_bit)
    assert carries_signal(lit_12_bit, bit_depth=12)
    assert not carries_signal(dark_12_bit, bit_depth=12)


def _dark_with_features(count: int) -> np.ndarray:
    """The measured dark background with ``count`` pixels at 3000 counts: a sparse field."""
    frame = _dark_frame().astype(np.float64)
    flat = frame.reshape(-1)
    flat[np.random.default_rng(1).choice(flat.size, count, replace=False)] = 3000.0
    return frame


@pytest.mark.parametrize(
    ("count", "expected"),
    [(19, False), (20, True), (160, True)],
    ids=["19-hot-pixels", "20-pixels", "one-bead"],
)
def test_a_sparse_field_carries_signal_and_a_few_hot_pixels_do_not(
    count: int, expected: bool
) -> None:
    """A bead of 2.5 px is about 160 pixels, 0.08 % of the frame: under the percentiles' 1 %.

    ``MIN_FEATURE_FRACTION`` of this frame is 19.7 pixels; the percentiles read
    a dark frame in every case here, so it is the second rule that decides.
    """
    frame = _dark_with_features(count)

    assert signal_spread(frame) < MIN_SIGNAL_SPREAD_COUNTS
    assert carries_signal(frame) is expected


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_the_signal_does_not_depend_on_the_frames_dtype(dtype: type) -> None:
    frame = np.where(np.arange(SHAPE[0] * SHAPE[1]).reshape(SHAPE) % 2, 140, 100)

    assert signal_spread(frame.astype(dtype)) == pytest.approx(40.0)


def test_a_frame_that_is_not_2d_or_not_finite_has_no_signal_to_measure() -> None:
    with pytest.raises(ValueError, match="2-D"):
        signal_spread(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="not finite"):
        signal_spread(np.full(SHAPE, np.nan))


# -- a probe that fits poorly (#408, F6): 50 µm at the 4x is 18 px


def _fit_with(pixel_size_um: float, residual_um: float) -> Fit:
    """A trustworthy fit of a camera at 177.4 degrees, as session 3 measured, with this residual."""
    return Fit(
        _matrix(pixel_size_um, 177.437, False),
        pixel_size_um,
        177.437,
        False,
        residual_um,
        (),
    )


@pytest.mark.parametrize(
    ("step_um", "pixel_size_um", "residual_um", "reasons"),
    [
        # Session 3, the 4x with no pixel size known: 18 px, 21 % of the step.
        (
            50.0,
            2.74526,
            10.6551,
            (
                "18 px, under a tenth of the 1412 px frame",
                "residual 10.7 µm, 21 % of the step",
            ),
        ),
        # The three fits that were good: 329, 311 and 362 px; 0.1, 0.2 and 0.5 %.
        (900.0, 2.73909, 0.894, ()),
        (350.0, 1.12447, 0.642, ()),
        (100.0, 0.276136, 0.535, ()),
    ],
    ids=["4x-fallback", "4x-900um", "10x-350um", "40x-100um"],
)
def test_session_three_s_fits_are_judged_as_they_were_measured(
    step_um: float,
    pixel_size_um: float,
    residual_um: float,
    reasons: tuple[str, ...],
) -> None:
    found = precision_reasons(
        _fit_with(pixel_size_um, residual_um), step_um, (1412, 1412)
    )

    assert found == reasons


@pytest.mark.parametrize(
    ("pixel_size_um", "flagged"),
    [(2.74526, True), (1.12447, True), (0.276136, False)],
    ids=["4x-18-px", "10x-44-px", "40x-181-px"],
)
def test_the_fallback_step_is_a_few_pixels_at_the_4x_and_the_10x_and_not_at_the_40x(
    pixel_size_um: float, flagged: bool
) -> None:
    """The plan's own numbers, so that a bound that moves to 5 % or 20 % of the frame is seen."""
    found = precision_reasons(_fit_with(pixel_size_um, 0.0), 50.0, (1412, 1412))

    assert bool(found) is flagged


def test_a_residual_that_is_nan_is_a_reason_not_a_pass() -> None:
    found = precision_reasons(_fit_with(2.0, float("nan")), 1000.0, (1412, 1412))

    assert len(found) == 1
    assert found[0].startswith("residual nan")


@pytest.mark.parametrize(
    ("step_px", "residual_share", "expect_small", "expect_residual"),
    [
        (34.0, 0.0, True, False),  # a tenth of 384 rows is 38.4
        (42.0, 0.0, False, False),
        (100.0, 0.04, False, False),
        (100.0, 0.06, False, True),
        (34.0, 0.06, True, True),
    ],
    ids=["under-a-tenth", "over-a-tenth", "4-percent", "6-percent", "both"],
)
def test_the_precision_rules_are_decided_at_a_tenth_of_the_frame_and_five_percent_of_the_step(
    step_px: float, residual_share: float, expect_small: bool, expect_residual: bool
) -> None:
    """The smaller side of a 384 x 512 frame, not the larger one (FM-73): a tenth is 38.4 px."""
    pixel_size_um = 2.0
    step_um = step_px * pixel_size_um
    found = precision_reasons(
        _fit_with(pixel_size_um, residual_share * step_um), step_um, SHAPE
    )

    assert any("under a tenth of the 384 px frame" in r for r in found) is expect_small
    assert any("of the step" in r for r in found) is expect_residual
    assert len(found) == int(expect_small) + int(expect_residual)
