"""The second look at a probe the camera's own pattern won (m3 §4.4, #625).

At the Ti2-E (session 6, F14) *Calibrate the camera* at the 10x in phase contrast
was refused on every field of a plate with cells: the sample was almost absent,
the sensor's fixed pattern (vignette, column stripes, dust) won the phase
correlation at zero shift, and a stage that had moved a quarter field read as one
that had not. The frames of that refusal were never saved, so the field here is
built from what the stand's own scan showed (§4.4.1): the numbers are measured,
the frame is synthetic and seeded.

The premise of each test is asserted, not assumed: where ``measure_shift`` reads
the true shift on a seed, that seed is not a sparse field, and a test that ran on
it would pass without the second look. The frames are not square, on purpose
(FM-73).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from smc.plugins.camera_calibration.algorithm import (
    FIXED_PATTERN_EXCLUDE_PX,
    MIN_PEAK_RATIO,
    AwayShift,
    Shift,
    measure_shift,
    measure_shift_away_from_zero,
    reads_the_fixed_pattern,
    shifts_of_a_rigid_camera,
)

HEIGHT, WIDTH = 384, 512
STEP_PX = 96
#: The canvas of the sample is this much larger than the frame on every side,
#: so that a probe moved by up to this many pixels still sees sample.
MARGIN_PX = 120
#: The seeds on which ``measure_shift`` reads ``(0, 0)`` for both probes of the
#: sparse field (the premise of §4.4.4), on numpy 2.5.5: the generator is the
#: test's, and a seed that does not hold is left out rather than fixed up.
SEEDS = (0, 1, 2, 4, 5)
#: Where the content went for a +X and a +Y probe of the camera of the stand's
#: morning: (d_row, d_col). Content moves against the stage.
UPRIGHT = ((0, STEP_PX), (-STEP_PX, 0))

Frame = npt.NDArray[np.float64]


def _soften(image: Frame, sigma: float = 1.0) -> Frame:
    """A Gaussian blur of ``sigma`` pixels, in numpy (edge-padded, separable)."""
    radius = math.ceil(4.0 * sigma)
    taps = np.exp(-((np.arange(-radius, radius + 1)) ** 2) / (2.0 * sigma**2))
    taps /= taps.sum()
    height, width = image.shape
    padded = np.pad(image, radius, mode="edge")
    rows = sum(
        tap * padded[i : i + height, :] for i, tap in enumerate(taps)
    )  # (height, width + 2 * radius)
    return np.asarray(sum(tap * rows[:, i : i + width] for i, tap in enumerate(taps)))


def _discs(
    shape: tuple[int, int],
    count: int,
    radius_px: float,
    amplitude: float,
    rng: np.random.Generator,
) -> Frame:
    """``count`` round specks of ``amplitude`` (times 0.6 to 1.4), softened by one pixel."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    out = np.zeros(shape)
    for _ in range(count):
        centre_row = rng.uniform(0.0, shape[0])
        centre_col = rng.uniform(0.0, shape[1])
        height = amplitude * rng.uniform(0.6, 1.4)
        out[(rows - centre_row) ** 2 + (cols - centre_col) ** 2 <= radius_px**2] += (
            height
        )
    return _soften(out)


def _probes(
    seed: int,
    shifts: tuple[tuple[int, int], tuple[int, int]] = UPRIGHT,
    *,
    sample: bool = True,
) -> tuple[Frame, Frame, Frame]:
    """A reference and two probe frames of a sparse field over the camera's own pattern.

    The pattern (fixed to the sensor, so it does not move with the stage) is a
    vignette ``30000 - 3000 r^2``, a stripe of ``N(0, 150)`` counts per column
    and 60 specks of -1500 counts; the sample is a canvas with 45 specks of
    -2000 counts, of which a probe sees the window moved by ``shifts``
    (``d_row``, ``d_col`` of the content); each frame has ``N(0, 128)`` noise of
    its own. ``sample=False`` is a field with no sample: the same pattern and
    noise over an empty canvas. ``shifts=((0, 0), (0, 0))`` is a stage that did
    not move.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    r_squared = ((y - HEIGHT / 2) / (HEIGHT / 2)) ** 2 + (
        (x - WIDTH / 2) / (WIDTH / 2)
    ) ** 2
    pattern = (
        30000.0
        - 3000.0 * r_squared
        + rng.normal(0.0, 150.0, WIDTH)[None, :]
        + _discs((HEIGHT, WIDTH), 60, 2.0, -1500.0, rng)
    )
    canvas = _discs(
        (HEIGHT + 2 * MARGIN_PX, WIDTH + 2 * MARGIN_PX), 45, 3.0, -2000.0, rng
    )
    if not sample:
        canvas = np.zeros_like(canvas)
    noises = [rng.normal(0.0, 128.0, (HEIGHT, WIDTH)) for _ in range(3)]

    def seen(d_row: int, d_col: int, noise: Frame) -> Frame:
        top, left = MARGIN_PX - d_row, MARGIN_PX - d_col
        return pattern + canvas[top : top + HEIGHT, left : left + WIDTH] + noise

    (x_row, x_col), (y_row, y_col) = shifts
    return (
        seen(0, 0, noises[0]),
        seen(x_row, x_col, noises[1]),
        seen(y_row, y_col, noises[2]),
    )


def _fourier_shift(image: Frame, d_row: float, d_col: float) -> Frame:
    """``image`` with its content moved by a sub-pixel amount (periodic), as a phase ramp."""
    rows = np.fft.fftfreq(image.shape[0])[:, None]
    cols = np.fft.fftfreq(image.shape[1])[None, :]
    ramp = np.exp(-2j * np.pi * (rows * d_row + cols * d_col))
    return np.asarray(np.real(np.fft.ifft2(np.fft.fft2(image) * ramp)))


def _reads_zero(first: Shift) -> bool:
    """Whether the first look reads no shift: under a pixel (the sub-pixel step reads 0.1 to 0.5)."""
    return abs(first.d_col_px) < 1.0 and abs(first.d_row_px) < 1.0


# ------------------------------------------------------- a sparse field, read away from zero


@pytest.mark.parametrize("seed", SEEDS)
def test_a_sparse_field_over_the_cameras_pattern_reads_zero_and_the_second_look_reads_the_shift(
    seed: int,
) -> None:
    """F14: the camera's pattern wins at zero shift, and only the second look sees the sample."""
    reference, probe_x, probe_y = _probes(seed)

    # The premise: the first look reads both probes as a stage that did not move,
    # on frames that do agree (a response well above the refusal's 0.05).
    first_x, first_y = (
        measure_shift(reference, probe_x),
        measure_shift(reference, probe_y),
    )
    assert _reads_zero(first_x)
    assert _reads_zero(first_y)
    assert first_x.response > 0.3
    assert first_y.response > 0.3

    away_x = measure_shift_away_from_zero(reference, probe_x)
    away_y = measure_shift_away_from_zero(reference, probe_y)

    # The content moved 96 columns for +X and 96 rows up for +Y.
    assert away_x.shift.d_row_px == pytest.approx(0.0, abs=1.5)
    assert away_x.shift.d_col_px == pytest.approx(STEP_PX, abs=1.5)
    assert away_y.shift.d_row_px == pytest.approx(-STEP_PX, abs=1.5)
    assert away_y.shift.d_col_px == pytest.approx(0.0, abs=1.5)
    assert away_x.peak_ratio >= MIN_PEAK_RATIO
    assert away_y.peak_ratio >= MIN_PEAK_RATIO
    assert shifts_of_a_rigid_camera(away_x.shift, away_y.shift)
    # The response is measured at the shift found: the sample and the pattern agree there.
    assert away_x.shift.response > 0.3
    assert away_y.shift.response > 0.3


@pytest.mark.parametrize(
    ("shifts", "why"),
    [
        (((0, -STEP_PX), (STEP_PX, 0)), "aligned, the other way round"),
        (((48, 83), (-83, 48)), "turned by 30 degrees"),
        (((48, 83), (83, -48)), "turned by 30 degrees and mirrored"),
    ],
    ids=["aligned-reversed", "turned", "turned-and-mirrored"],
)
def test_a_camera_at_another_pose_is_read_by_the_second_look_as_well(
    shifts: tuple[tuple[int, int], tuple[int, int]], why: str
) -> None:
    """The second look takes no pose for granted: any rigid pair of shifts is read (``why``)."""
    reference, probe_x, probe_y = _probes(0, shifts)

    away_x = measure_shift_away_from_zero(reference, probe_x)
    away_y = measure_shift_away_from_zero(reference, probe_y)

    (x_row, x_col), (y_row, y_col) = shifts
    assert away_x.shift.d_row_px == pytest.approx(x_row, abs=1.5)
    assert away_x.shift.d_col_px == pytest.approx(x_col, abs=1.5)
    assert away_y.shift.d_row_px == pytest.approx(y_row, abs=1.5)
    assert away_y.shift.d_col_px == pytest.approx(y_col, abs=1.5)
    assert away_x.peak_ratio >= MIN_PEAK_RATIO, why
    assert away_y.peak_ratio >= MIN_PEAK_RATIO, why
    assert shifts_of_a_rigid_camera(away_x.shift, away_y.shift)


def test_a_camera_aligned_with_the_stage_is_read_on_the_axes() -> None:
    """The X probe is on the row axis and the Y probe on the column stripes' ridge.

    The stripes are one pattern per column, so they correlate along the whole
    column of zero shift: the Y probe's true peak sits on the camera's own ridge,
    and the exclusion removes only 12 rows of it. It must still stand out.
    """
    reference, probe_x, probe_y = _probes(1)

    away_x = measure_shift_away_from_zero(reference, probe_x)
    away_y = measure_shift_away_from_zero(reference, probe_y)

    assert away_x.shift.d_row_px == pytest.approx(0.0, abs=1.5)  # on the row axis
    assert away_y.shift.d_col_px == pytest.approx(0.0, abs=1.5)  # on the ridge
    assert away_x.peak_ratio >= MIN_PEAK_RATIO
    assert away_y.peak_ratio >= MIN_PEAK_RATIO


# ------------------------------------------------- what it must not read: no peak away from zero


@pytest.mark.parametrize("seed", SEEDS)
def test_a_stage_that_did_not_move_gives_no_peak_away_from_zero(seed: int) -> None:
    """The content where it was: the camera's pattern is the only thing the frames share.

    The first look reads zero, as it should, and the second look must not turn
    the noise elsewhere on the surface into a shift: its best peak is the best of
    a crowd (1.0 to 1.2 times the next, measured), far under the 1.5 asked.
    """
    reference, probe_x, probe_y = _probes(seed, ((0, 0), (0, 0)))
    assert _reads_zero(measure_shift(reference, probe_x))
    assert _reads_zero(measure_shift(reference, probe_y))

    for probe in (probe_x, probe_y):
        assert (
            measure_shift_away_from_zero(reference, probe).peak_ratio < MIN_PEAK_RATIO
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_a_field_with_no_sample_gives_no_peak_away_from_zero(seed: int) -> None:
    """The stage moved over a field with nothing on it: the frames share only the pattern."""
    reference, probe_x, probe_y = _probes(seed, sample=False)
    assert _reads_zero(measure_shift(reference, probe_x))
    assert _reads_zero(measure_shift(reference, probe_y))

    for probe in (probe_x, probe_y):
        assert (
            measure_shift_away_from_zero(reference, probe).peak_ratio < MIN_PEAK_RATIO
        )


def test_a_peak_inside_the_exclusion_is_not_read() -> None:
    """A rich pair shifted by 4 and 3 pixels is the camera's zone: the second look looks away.

    The zero peak's skirt reaches 4 to 5 pixels, which is why 12 are excluded and
    not 3: a shift inside them is read by ``measure_shift`` and not by this.
    """
    texture = np.random.default_rng(5).normal(size=(200, 300))
    moved = _fourier_shift(texture, 4.0, -3.0)
    assert measure_shift(texture, moved).d_row_px == pytest.approx(4.0, abs=0.05)

    away = measure_shift_away_from_zero(texture, moved)

    assert away.peak_ratio < MIN_PEAK_RATIO
    assert (away.shift.d_row_px, away.shift.d_col_px) != pytest.approx(
        (4.0, -3.0), abs=1.0
    )


# ----------------------------------------------------------- agreeing with measure_shift


@pytest.mark.parametrize(
    ("d_row", "d_col"),
    [(20.3, -31.6), (-45.75, 17.25), (13.5, 60.1), (-27.2, -14.4)],
)
def test_the_second_look_agrees_with_measure_shift_on_a_rich_pair(
    d_row: float, d_col: float
) -> None:
    """On a textured pair the true peak is the highest one: both read it, to 0.05 px.

    The shifts are sub-pixel and further than the exclusion from zero, so a
    refinement written differently from scikit-image's (the window, the offset,
    the sign) shows as a difference here.
    """
    texture = np.random.default_rng(5).normal(size=(200, 300))
    moved = _fourier_shift(texture, d_row, d_col)

    first = measure_shift(texture, moved)
    away = measure_shift_away_from_zero(texture, moved)

    assert (first.d_row_px, first.d_col_px) == pytest.approx((d_row, d_col), abs=0.2)
    assert away.shift.d_row_px == pytest.approx(first.d_row_px, abs=0.05)
    assert away.shift.d_col_px == pytest.approx(first.d_col_px, abs=0.05)
    assert away.shift.response == pytest.approx(first.response, abs=0.01)
    assert away.peak_ratio > 5.0 * MIN_PEAK_RATIO


def test_the_second_look_reads_the_content_moving_the_way_measure_shift_does() -> None:
    """Rolled down 7 rows... as ``measure_shift``'s own test: the sign is the content's."""
    a = np.random.default_rng(0).normal(size=(200, 300))
    b = np.roll(np.roll(a, 20, axis=0), -35, axis=1)

    away = measure_shift_away_from_zero(a, b)

    assert away.shift.d_row_px == pytest.approx(20.0, abs=0.05)
    assert away.shift.d_col_px == pytest.approx(-35.0, abs=0.05)


# ------------------------------------------------------------------- degenerate frames


def test_a_flat_frame_has_no_second_look() -> None:
    """A frame without variation has no shift to find: no correlation is asked of it."""
    texture = np.random.default_rng(1).normal(size=(120, 160))
    flat = np.full((120, 160), 1234.567)

    for reference, moved in ((flat, texture), (texture, flat), (flat, flat)):
        assert measure_shift_away_from_zero(reference, moved) == AwayShift(
            Shift(0.0, 0.0, 0.0), 1.0
        )


@pytest.mark.parametrize(
    ("shape", "refused"),
    [
        ((4 * FIXED_PATTERN_EXCLUDE_PX, 200), True),
        ((200, 4 * FIXED_PATTERN_EXCLUDE_PX), True),
        ((4 * FIXED_PATTERN_EXCLUDE_PX + 1, 200), False),
        ((200, 4 * FIXED_PATTERN_EXCLUDE_PX + 1), False),
    ],
    ids=["rows-at-the-limit", "columns-at-the-limit", "rows-over", "columns-over"],
)
def test_a_frame_not_larger_than_the_exclusion_is_refused(
    shape: tuple[int, int], refused: bool
) -> None:
    """With a side of four exclusions or less no peak is left to compare: a ``ValueError``."""
    frame = np.random.default_rng(2).normal(size=shape)
    if refused:
        with pytest.raises(ValueError, match="too small"):
            measure_shift_away_from_zero(frame, frame)
    else:
        measure_shift_away_from_zero(frame, frame)


def test_the_exclusion_is_the_callers_to_choose() -> None:
    """``exclude_px`` is read: a rich pair shifted by 8 is found with 5 excluded and not with 12."""
    texture = np.random.default_rng(5).normal(size=(200, 300))
    moved = _fourier_shift(texture, 8.0, 0.0)

    narrow = measure_shift_away_from_zero(texture, moved, exclude_px=5)
    wide = measure_shift_away_from_zero(texture, moved)

    assert narrow.shift.d_row_px == pytest.approx(8.0, abs=0.05)
    assert narrow.peak_ratio > MIN_PEAK_RATIO
    assert wide.peak_ratio < MIN_PEAK_RATIO


def test_frames_the_first_look_refuses_are_refused_by_the_second() -> None:
    """The same ``ValueError`` for a shape mismatch, a 3-D frame and a value that is not finite."""
    frame = np.random.default_rng(3).normal(size=(120, 160))
    holed = frame.copy()
    holed[3, 4] = np.nan

    with pytest.raises(ValueError, match="same shape"):
        measure_shift_away_from_zero(frame, frame[:, :150])
    with pytest.raises(ValueError, match="2-D"):
        measure_shift_away_from_zero(frame[:, :, None], frame[:, :, None])
    with pytest.raises(ValueError, match="not finite"):
        measure_shift_away_from_zero(frame, holed)


# --------------------------------------------------------------------- the two predicates


def _rigid(
    size_px: float, angle_deg: float, mirrored: bool = False
) -> tuple[Shift, Shift]:
    """The shifts of a +X and a +Y probe of a rigid camera: ``size`` long, at a right angle."""
    angle = math.radians(angle_deg)
    x = (size_px * math.cos(angle), size_px * math.sin(angle))
    sign = 1.0 if mirrored else -1.0
    y = (sign * x[1], -sign * x[0])
    return Shift(x[0], x[1], 1.0), Shift(y[0], y[1], 1.0)


@pytest.mark.parametrize("mirrored", [False, True], ids=["upright", "mirrored"])
@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 90.0, 178.33, -135.0, 270.5])
def test_a_rigid_cameras_shifts_pass_at_any_pose_and_mirror(
    angle_deg: float, mirrored: bool
) -> None:
    shift_x, shift_y = _rigid(96.0, angle_deg, mirrored)

    assert shifts_of_a_rigid_camera(shift_x, shift_y)
    assert shifts_of_a_rigid_camera(shift_y, shift_x)


@pytest.mark.parametrize("mirrored", [False, True], ids=["upright", "mirrored"])
def test_a_rigid_cameras_shifts_pass_at_their_tolerance_and_inside_it(
    mirrored: bool,
) -> None:
    """5 % of length and 5 degrees off a right angle are a camera's: both senses, both probes."""
    base_x, base_y = _rigid(100.0, 20.0, mirrored)
    for factor in (0.95, 1.05):
        longer = Shift(base_y.d_col_px * factor, base_y.d_row_px * factor, 1.0)
        assert shifts_of_a_rigid_camera(base_x, longer)
        assert shifts_of_a_rigid_camera(longer, base_x)
    for degrees in (-5.0, 5.0):
        turned_x, turned_y = _rigid(100.0, 20.0 + degrees, mirrored)
        assert shifts_of_a_rigid_camera(base_x, turned_y)
        assert shifts_of_a_rigid_camera(turned_x, base_y)


@pytest.mark.parametrize("mirrored", [False, True], ids=["upright", "mirrored"])
def test_shifts_of_unequal_length_or_off_a_right_angle_are_refused(
    mirrored: bool,
) -> None:
    """15 % of length or 15 degrees off: two peaks of the sample's own periodicity, not a camera."""
    base_x, base_y = _rigid(100.0, 20.0, mirrored)
    for factor in (0.85, 1.15):
        longer = Shift(base_y.d_col_px * factor, base_y.d_row_px * factor, 1.0)
        assert not shifts_of_a_rigid_camera(base_x, longer)
        assert not shifts_of_a_rigid_camera(longer, base_x)
    for degrees in (-15.0, 15.0):
        turned_x, turned_y = _rigid(100.0, 20.0 + degrees, mirrored)
        assert not shifts_of_a_rigid_camera(base_x, turned_y)
        assert not shifts_of_a_rigid_camera(turned_x, base_y)
    # Parallel and opposite probes are the limit of a stage that moved one way twice.
    assert not shifts_of_a_rigid_camera(base_x, base_x)
    assert not shifts_of_a_rigid_camera(
        base_x, Shift(-base_x.d_col_px, -base_x.d_row_px, 1.0)
    )


def test_shifts_without_length_or_without_a_value_are_not_a_cameras() -> None:
    """A probe that did not move, or whose shift is not a number, cannot make a camera."""
    good = Shift(96.0, 0.0, 1.0)

    assert not shifts_of_a_rigid_camera(Shift(0.0, 0.0, 1.0), good)
    assert not shifts_of_a_rigid_camera(good, Shift(0.0, 0.0, 1.0))
    assert not shifts_of_a_rigid_camera(
        Shift(math.nan, 0.0, 1.0), Shift(0.0, 96.0, 1.0)
    )
    assert not shifts_of_a_rigid_camera(good, Shift(0.0, math.inf, 1.0))


@pytest.mark.parametrize(
    ("d_col", "d_row", "fixed"),
    [
        (1.5, 0.0, True),
        (0.0, 1.5, True),
        (-1.5, 1.5, True),
        (0.0, 0.0, True),
        (3.0, 0.0, False),
        (0.0, 3.0, False),
        (-3.0, 0.0, False),
        (1.5, -3.0, False),
        (96.0, 0.0, False),
        (math.nan, 0.0, False),
        (0.0, math.nan, False),
    ],
)
def test_reads_the_fixed_pattern_is_decided_at_two_pixels(
    d_col: float, d_row: float, fixed: bool
) -> None:
    """1.5 is the camera's pattern and 3.0 is a shift, on each axis; a nan is neither (no second look)."""
    assert reads_the_fixed_pattern(Shift(d_col, d_row, 1.0)) is fixed
