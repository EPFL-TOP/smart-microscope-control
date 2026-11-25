"""How the camera is mounted, from frames taken before and after a stage move (m3 §4.1).

Everything downstream needs one thing: given a pixel of a frame, where is that
on the stage? Asking an operator for the pixel size, the direction of each
camera axis, the mirror and the rotation gives four numbers that are each
silently wrong until a revisit misses. So the stage is the ruler: **move it a
known distance in X and in Y and measure how far the image moved.** The two
displacements, divided by the distance, are the matrix of
``smc.plugins.geometry`` (inverted and negated: the content moves against the
stage).

Two failures look like success, and each is refused here instead of returned:

* **A field with nothing to lock onto.** Correlating featureless frames gives a
  confident shift that means nothing, so a weak response ends the fit.
* **A shift beyond half the frame.** Phase correlation reports each axis in
  ``[-N/2, N/2)``, so content moved by 200 rows of a 384-row frame reads -184.
  The response measured at the wrapped shift is that of unrelated content, so
  the same weak response refuses it (FM-120).

A third looks like a failure and is not always one: **a sparse field over the
camera's own pattern** (m3 §4.4). The sensor's fixed pattern (vignette, column
stripes, dust) does not move with the stage, so on a field that holds almost
nothing of the sample it wins the phase correlation at zero shift, and a stage
that moved a quarter field reads as one that did not (F14, Ti2-E session 6,
#625). ``measure_shift_away_from_zero`` is the second look at the probes the
first one read as zero; ``measure_shift`` is not changed.

This is the port of ``nikon-control``'s ``camera_cal.py`` without its hardware
loop (that is ``plugin.py``'s): it imports nothing of ``smc.hardware``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from skimage import registration

# The signal measure moved to ``smc.plugins.frame_signal`` (m5 §17.12 D22: the
# guided page asks it too). Re-exported by name (``x as x``), so that the plugin
# and the tests import it from here as before.
from smc.plugins.frame_signal import (
    MIN_FEATURE_FRACTION as MIN_FEATURE_FRACTION,
)
from smc.plugins.frame_signal import (
    MIN_SIGNAL_SPREAD_COUNTS as MIN_SIGNAL_SPREAD_COUNTS,
)
from smc.plugins.frame_signal import as_frame as _as_frame
from smc.plugins.frame_signal import carries_signal as carries_signal
from smc.plugins.frame_signal import signal_floor_counts as signal_floor_counts
from smc.plugins.frame_signal import signal_spread as signal_spread
from smc.plugins.geometry import Matrix2

__all__ = [
    "BARELY_CORRELATE_NOTE",
    "DEFAULT_STEP_FRACTION",
    "FALLBACK_STEP_UM",
    "FIXED_PATTERN_EXCLUDE_PX",
    "FIXED_SHIFT_PX",
    "MAX_RESIDUAL_SHARE",
    "MIN_FEATURE_FRACTION",
    "MIN_PEAK_RATIO",
    "MIN_RESPONSE",
    "MIN_SIGNAL_SPREAD_COUNTS",
    "PARALLEL_SHIFTS_NOTE",
    "PEAK_SEPARATION_PX",
    "RIGID_ANGLE_TOLERANCE_DEG",
    "RIGID_MAGNITUDE_TOLERANCE",
    "SMALL_STEP_FRACTION",
    "UPSAMPLE",
    "AwayShift",
    "Fit",
    "Shift",
    "carries_signal",
    "fit",
    "measure_shift",
    "measure_shift_away_from_zero",
    "precision_reasons",
    "reads_the_fixed_pattern",
    "shifts_of_a_rigid_camera",
    "signal_floor_counts",
    "signal_spread",
]

#: Sub-pixel refinement of the correlation peak: 1 / 10 of a pixel, far below
#: the stage's own repeatability.
UPSAMPLE = 10
#: A response below this means the frames had nothing in common once aligned.
MIN_RESPONSE = 0.05
#: How far to move, as a fraction of the smaller side of the frame: far enough
#: for a precise shift, and well inside the half-frame that wraps (§4.1).
DEFAULT_STEP_FRACTION = 0.25
#: The probe distance when no pixel size is known (a probe, not a result).
FALLBACK_STEP_UM = 50.0
#: A probe under this fraction of the frame's smaller side is a few pixels, and
#: a few pixels fit poorly: 50 µm at the 4x is 18 px of 1412 (1.3 %), where the
#: good fits of the same session were 311 to 362 px (22 to 26 %).
SMALL_STEP_FRACTION = 0.1
#: A fit whose residual is more than this share of its probe is poorly measured:
#: 21 % at 18 px, against 0.1 to 0.5 % for the three good fits of session 3.
MAX_RESIDUAL_SHARE = 0.05

#: What the fit says when the two probes shifted the content the same way, most
#: often by zero. The frames come first, then the stage: at the 40x in water, on
#: a field of two beads over a still background (dust, shading), the stage moved
#: and the background won the correlation, and "did the stage move?" sent the
#: operator to the wrong device, as it had for a frame with no light (F8, F12,
#: #408, #545). No semicolon in it: the plugin joins the notes with one.
PARALLEL_SHIFTS_NOTE = (
    "the two moves gave parallel shifts: the frames probably hold too little "
    "texture (a sparse field, a dark or flat frame): try a field with more "
    "structure, then the light path, then the stage"
)

#: What the fit says when a probe's frames share nothing once aligned. A name for
#: the text ``fit`` has always written, so that the plugin can tell a refusal for
#: texture (this and ``PARALLEL_SHIFTS_NOTE``) from the others by the note itself.
BARELY_CORRELATE_NOTE = "the frames barely correlate: find a field with texture"

#: The second look sets to zero the correlation surface this far (rows and
#: columns, both ways) from zero shift (m3 §4.4.2 D2). 12 px is the zero peak's
#: skirt, which reaches 4 to 5 px: with 3 px excluded an empty field's best
#: "other" peak stood at 1.7 times the next, a peak that is only the skirt.
FIXED_PATTERN_EXCLUDE_PX = 12
#: A peak's "next peak" is the highest cell further than this from it.
PEAK_SEPARATION_PX = 10
#: The second look is believed when its best peak is this many times the next
#: one. A stage that did not move gave at most 1.20 and an empty field 1.19 on
#: emulated pairs of the stand's frames; the fields it reads right have a tenth
#: percentile of 2.4 (cells) and 1.4 (beads), and none read at 1.5 was another peak.
MIN_PEAK_RATIO = 1.5
#: A first shift within this many pixels of zero, on both axes, is the camera's
#: own pattern: the stage moved tens of pixels.
FIXED_SHIFT_PX = 2.0
#: The two probes of a rigid camera are as long as each other, to this share...
RIGID_MAGNITUDE_TOLERANCE = 0.10
#: ... and at a right angle, to this many degrees (a mirrored camera as well).
RIGID_ANGLE_TOLERANCE_DEG = 10.0

#: Fewer overlapping pixels than this on an axis say nothing about agreement.
_MIN_OVERLAP_PX = 8
#: Below this ``abs(det)`` the two probes shifted the content the same way: a
#: stage that did not move one of its axes. Same value as ``geometry``'s.
_MIN_DETERMINANT = 1e-12
#: A frame whose standard deviation is below this fraction of its level
#: (at least 1 count) is a flat one: the mean of a constant float frame is
#: off by a rounding error (2e-13 for 1234.567), and dividing by it would turn
#: that into unit-variance "texture".
_FLAT_RELATIVE_STD = 1e-12

# scikit-image's `phase_cross_correlation` is unannotated, which `mypy --strict`
# refuses at the call (`no-untyped-call`). Bound once, with the shape of what it
# returns (shift, error, phase difference) rather than a `type: ignore`.
_phase_cross_correlation: Callable[..., tuple[Any, Any, Any]] = (
    registration.phase_cross_correlation
)


@dataclass(frozen=True, slots=True)
class Shift:
    """How far the content moved between two frames, and how well they agree.

    ``d_col_px`` and ``d_row_px`` are the displacement of the **content** from
    the reference to the moved frame (rows grow downwards). ``response`` is
    0 to 1 and is measured after the frames are aligned by that shift.
    """

    d_col_px: float
    d_row_px: float
    response: float


@dataclass(frozen=True, slots=True)
class Fit:
    """The camera matrix the shifts give, and the reasons not to trust it.

    ``notes`` each make the fit unusable; none means it is trustworthy. With a
    "parallel shifts" note the matrix is zero and the size and rotation are 0.0
    (unknown, not a measurement), never the result of a division by nothing.
    """

    matrix_um_per_px: Matrix2
    pixel_size_um: float
    rotation_deg: float
    mirrored: bool
    residual_um: float
    notes: tuple[str, ...]

    @property
    def trustworthy(self) -> bool:
        """Whether nothing speaks against the fit (the issue's ``trustworthy``)."""
        return not self.notes


@dataclass(frozen=True, slots=True)
class AwayShift:
    """The strongest correlation peak away from zero shift, and how far it stands out.

    ``peak_ratio`` is that peak over the highest one more than
    ``PEAK_SEPARATION_PX`` from it: a peak that is the best of a crowd of equals
    is noise, and one several times the next is a shift (m3 §4.4.2 D2).
    """

    shift: Shift
    peak_ratio: float


def measure_shift(reference: npt.NDArray[Any], moved: npt.NDArray[Any]) -> Shift:
    """The displacement of the content from ``reference`` to ``moved``, with its response.

    Content rolled by +7 rows and -13 columns gives ``d_row_px=7`` and
    ``d_col_px=-13``: scikit-image reports the shift that registers ``moved``
    onto ``reference``, and this is its negation. A flat frame has no shift to
    find, and asking anyway returns a confident meaningless answer, so it is
    ``Shift(0.0, 0.0, 0.0)`` without calling the correlation.

    Raises:
        ValueError: A frame is not 2-D, holds a value that is not finite, or
            the two frames differ in shape (the layer's ``Camera.snap`` is 2-D).
    """
    a = _as_frame(reference, "reference")
    b = _as_frame(moved, "moved")
    if a.shape != b.shape:
        raise ValueError(
            f"the two frames must have the same shape, got {a.shape} and {b.shape}"
        )
    if _is_flat(a) or _is_flat(b):
        return Shift(0.0, 0.0, 0.0)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    shift, _error, _phase = _phase_cross_correlation(a, b, upsample_factor=UPSAMPLE)
    # `+ 0.0` turns a -0.0 (the negation of 0.0) into 0.0.
    d_row = -float(shift[0]) + 0.0
    d_col = -float(shift[1]) + 0.0
    return Shift(d_col, d_row, _response(a, b, d_row, d_col))


def measure_shift_away_from_zero(
    reference: npt.NDArray[Any],
    moved: npt.NDArray[Any],
    *,
    exclude_px: int = FIXED_PATTERN_EXCLUDE_PX,
) -> AwayShift:
    """The shift of the strongest correlation peak that is not at zero shift (m3 §4.4).

    The camera's own pattern (vignette, column stripes, dust) does not move with
    the stage, so on a field that holds almost nothing of the sample it is the
    highest peak of ``measure_shift`` and the stage reads as one that did not
    move (F14, #625). When the stage moved a probe of tens of pixels, a peak at
    zero shift is what did not move. This looks at the same correlation surface
    with the cells within ``exclude_px`` of zero shift (the zero peak's skirt
    reaches 4 to 5 px) set to 0, and says how far the best remaining peak stands
    out from the next one: the camera's pattern is not a peak that stands out
    away from zero, and a sample's shift is. It is a second look, asked only of
    a probe the first one read as zero; ``measure_shift`` stays the answer for
    every field that calibrates without it.

    The correlation is ``measure_shift``'s (normalised frames, the cross-power
    spectrum at unit magnitude), and so is the sub-pixel step (the matrix-product
    DFT around the integer peak at ``UPSAMPLE``), and the response, measured at
    the shift found. A flat frame has no shift to find: ``Shift(0.0, 0.0, 0.0)``
    at a ratio of 1.0, without a correlation.

    Raises:
        ValueError: A frame is not 2-D, holds a value that is not finite, the two
            frames differ in shape, or a side is ``4 * exclude_px`` or less (the
            exclusion and the separation would then leave no peak to compare).
    """
    a = _as_frame(reference, "reference")
    b = _as_frame(moved, "moved")
    if a.shape != b.shape:
        raise ValueError(
            f"the two frames must have the same shape, got {a.shape} and {b.shape}"
        )
    if min(a.shape) <= 4 * exclude_px:
        raise ValueError(
            f"a frame of {a.shape} is too small to exclude {exclude_px} px around "
            "zero shift: each side must be more than four times that"
        )
    if _is_flat(a) or _is_flat(b):
        return AwayShift(Shift(0.0, 0.0, 0.0), 1.0)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()

    # The cross-power spectrum at unit magnitude, floored as scikit-image does.
    product = np.fft.fft2(a) * np.fft.fft2(b).conj()
    product /= np.maximum(np.abs(product), 100.0 * np.finfo(np.float64).eps)
    surface = np.abs(np.fft.ifft2(product))
    height, width = surface.shape
    surface[
        np.ix_(_cells_near(height, 0, exclude_px), _cells_near(width, 0, exclude_px))
    ] = 0.0

    row, col = (
        int(index) for index in np.unravel_index(np.argmax(surface), surface.shape)
    )
    best = float(surface[row, col])
    rest = surface.copy()
    rest[
        np.ix_(
            _cells_near(height, row, PEAK_SEPARATION_PX),
            _cells_near(width, col, PEAK_SEPARATION_PX),
        )
    ] = 0.0
    next_best = float(rest.max())
    if next_best > 0.0:
        peak_ratio = best / next_best
    else:
        peak_ratio = math.inf if best > 0.0 else 1.0

    # The integer peak as a signed shift, then refined at 1 / UPSAMPLE of a pixel
    # in a window of a pixel and a half, exactly as `phase_cross_correlation` does.
    # `//` and `int`, never `np.fix`, which numpy 2.5 deprecates.
    shift_row = float(row - height if row > height // 2 else row)
    shift_col = float(col - width if col > width // 2 else col)
    region = math.ceil(UPSAMPLE * 1.5)
    centre = region // 2
    refined = np.abs(
        _upsampled_dft(
            product.conj(),
            region,
            (centre - shift_row * UPSAMPLE, centre - shift_col * UPSAMPLE),
        )
    )
    fine_row, fine_col = (
        int(index) for index in np.unravel_index(np.argmax(refined), refined.shape)
    )
    shift_row += (fine_row - centre) / UPSAMPLE
    shift_col += (fine_col - centre) / UPSAMPLE
    # Negated, as `measure_shift`'s; `+ 0.0` turns a -0.0 into 0.0.
    d_row, d_col = -shift_row + 0.0, -shift_col + 0.0
    return AwayShift(Shift(d_col, d_row, _response(a, b, d_row, d_col)), peak_ratio)


def reads_the_fixed_pattern(shift: Shift) -> bool:
    """Whether a probe's first shift is the camera's own pattern, not the stage's.

    The stage moved a probe of tens of pixels, so a shift within
    ``FIXED_SHIFT_PX`` of zero on both axes is what did not move with it (m3
    §4.4.2 D3). A positive test, so that a shift that is not a number is not read
    as the pattern: nothing then asks for the second look.
    """
    return (
        abs(shift.d_col_px) <= FIXED_SHIFT_PX and abs(shift.d_row_px) <= FIXED_SHIFT_PX
    )


def shifts_of_a_rigid_camera(shift_x: Shift, shift_y: Shift) -> bool:
    """Whether a +X and a +Y probe look like one camera's: as long, and at a right angle.

    A camera, mirrored or not, turns the two stage axes into two image vectors of
    one length that stand at a right angle (the sign of the angle is the mirror).
    Two peaks that are not, are two peaks of the sample's own periodicity or of
    noise. This guards the second look independently of its peak ratio, and
    ``fit`` then judges the matrix as ever (m3 §4.4.2 D4). A shift without
    length, or that is not a number, is not a camera's.
    """
    length_x = math.hypot(shift_x.d_col_px, shift_x.d_row_px)
    length_y = math.hypot(shift_y.d_col_px, shift_y.d_row_px)
    if not (length_x > 0.0 and length_y > 0.0):
        return False
    if not abs(length_x - length_y) <= RIGID_MAGNITUDE_TOLERANCE * max(
        length_x, length_y
    ):
        return False
    cosine = (
        shift_x.d_col_px * shift_y.d_col_px + shift_x.d_row_px * shift_y.d_row_px
    ) / (length_x * length_y)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return abs(angle_deg - 90.0) <= RIGID_ANGLE_TOLERANCE_DEG


def fit(
    shift_x: Shift,
    shift_y: Shift,
    *,
    step_um: float,
    image_shape_px: tuple[int, int],
) -> Fit:
    """The camera matrix from the shifts of a +X and a +Y move of ``step_um``.

    ``S`` has the columns ``(d_col, d_row)`` of the two probes, ``J = S / step``
    is the pixel displacement per µm of stage travel, and the matrix is
    ``M = -inv(J)``: the content moves against the stage. The pixel size is
    ``sqrt(|det M|)``, the camera is mirrored when ``det M > 0`` and the
    rotation, in ``(-180, 180]``, is that of ``M * diag(1, -1)`` once a
    mirrored camera's first column is flipped back. ``residual_um`` is how far
    the frame's edge lies from where a rigid camera of that size and rotation
    would put it: the largest entry of ``|M - ideal|`` times half the larger
    side. A fit whose largest entry of ``|M - ideal|`` reaches the pixel size
    is not a camera, and is refused with a note (an axis that did not execute
    its command): its matrix, size and residual are still returned as computed.

    Raises:
        ValueError: ``step_um`` is not finite and positive.
    """
    if not (math.isfinite(step_um) and step_um > 0.0):
        raise ValueError(f"step_um must be finite and positive, got {step_um!r}")
    notes: list[str] = []
    # `not >=`, not `<`, and each probe on its own: a response that is nan must
    # fail the check, and `min(0.9, nan)` is 0.9.
    if not (shift_x.response >= MIN_RESPONSE and shift_y.response >= MIN_RESPONSE):
        notes.append(BARELY_CORRELATE_NOTE)
    a, b = shift_x.d_col_px / step_um, shift_y.d_col_px / step_um
    c, d = shift_x.d_row_px / step_um, shift_y.d_row_px / step_um
    determinant = a * d - b * c
    if not abs(determinant) >= _MIN_DETERMINANT:
        notes.append(PARALLEL_SHIFTS_NOTE)
        return Fit(((0.0, 0.0), (0.0, 0.0)), 0.0, 0.0, False, 0.0, tuple(notes))

    # M = -inv(J) = (1 / det J) * [[-d, b], [c, -a]]; `+ 0.0` turns a -0.0 into 0.0.
    m00, m01 = -d / determinant + 0.0, b / determinant + 0.0
    m10, m11 = c / determinant + 0.0, -a / determinant + 0.0
    det_m = m00 * m11 - m01 * m10
    pixel_size_um = math.sqrt(abs(det_m))
    mirrored = det_m > 0.0

    # A = M * diag(1, -1); a mirrored camera's first column is flipped back, so
    # that A is a scaled rotation of the camera's angle in both cases.
    flip = -1.0 if mirrored else 1.0
    rotation_deg = math.degrees(math.atan2(flip * m10, flip * m00)) + 0.0
    if rotation_deg <= -180.0:
        rotation_deg += 360.0

    angle = math.radians(rotation_deg)
    cos, sin = pixel_size_um * math.cos(angle), pixel_size_um * math.sin(angle)
    # ps * R(theta) * diag(+1 or -1, -1), as in m3 §3.1.
    ideal = ((flip * cos, sin), (flip * sin, -cos))
    worst = max(
        abs(m00 - ideal[0][0]),
        abs(m01 - ideal[0][1]),
        abs(m10 - ideal[1][0]),
        abs(m11 - ideal[1][1]),
    )
    # An axis that executes a fraction of its command, or none plus noise, gives
    # frames that agree (a high response) and a matrix that is no camera: one
    # entry then lies further from a rotation than the pixel size itself, where
    # every good fit sits at a thousandth of it. `not <`, so that a nan fails.
    if not worst < pixel_size_um:
        notes.append(
            "the shifts do not fit a rigid camera: did each axis move its full command?"
        )
    height, width = image_shape_px
    return Fit(
        ((m00, m01), (m10, m11)),
        pixel_size_um,
        rotation_deg,
        mirrored,
        worst * max(height, width) / 2.0,
        tuple(notes),
    )


def precision_reasons(
    fitted: Fit, step_um: float, image_shape_px: tuple[int, int]
) -> tuple[str, ...]:
    """Why the probe was too short for its fit, one text per reason; none for a good fit.

    A probe of a few pixels gives a result of poor precision, not an error: 50 µm
    at the 4x is 18 px, and the fit had a residual of 10.7 µm (21 % of the step)
    where the 900 µm probe, 329 px, had 0.9 µm. The step is judged in pixels at
    the **measured** pixel size (the fallback step is a guess, so the stand's own
    value cannot size it), against the smaller side of the frame (a rotated
    camera puts the X probe on either axis, §4.1), and the residual against the
    step. The fit is never refused for these: they are reasons to measure again.
    """
    reasons: list[str] = []
    side = min(image_shape_px)
    step_px = step_um / fitted.pixel_size_um
    if step_px < SMALL_STEP_FRACTION * side:
        reasons.append(f"{step_px:.0f} px, under a tenth of the {side} px frame")
    # `not <=`: a residual that is nan is a reason, not a pass.
    if not fitted.residual_um <= MAX_RESIDUAL_SHARE * step_um:
        share = 100.0 * fitted.residual_um / step_um
        reasons.append(
            f"residual {fitted.residual_um:.1f} µm, {share:.0f} % of the step"
        )
    return tuple(reasons)


def _cells_near(size: int, centre: int, radius: int) -> npt.NDArray[np.bool_]:
    """The indices of an axis of ``size`` cells within ``radius`` of ``centre``, wrapped.

    A correlation surface is periodic: a peak at -3 is at ``size - 3``.
    """
    distance = np.abs(np.arange(size) - centre)
    return np.asarray(np.minimum(distance, size - distance) <= radius)


def _upsampled_dft(
    data: npt.NDArray[np.complex128], region: int, offsets: tuple[float, float]
) -> npt.NDArray[np.complex128]:
    """The DFT of ``data`` sampled at 1 / UPSAMPLE of a pixel in a ``region`` square, by matrix products.

    ``phase_cross_correlation``'s own refinement (Guizar-Sicairos 2008), written
    here because scikit-image keeps it private: it is the same arithmetic, so the
    second look and ``measure_shift`` agree on a rich pair to a hundredth of a
    pixel. ``offsets`` is where, in samples, the window starts on each axis.
    """
    out = data
    for n_items, offset in zip(data.shape[::-1], offsets[::-1], strict=True):
        kernel = (np.arange(region) - offset)[:, None] * np.fft.fftfreq(
            n_items, UPSAMPLE
        )
        out = np.tensordot(np.exp(-2j * np.pi * kernel), out, axes=(1, -1))
    return out


def _is_flat(frame: npt.NDArray[np.float64]) -> bool:
    """Whether the frame has no variation to correlate, to within a rounding error."""
    return bool(frame.std() <= _FLAT_RELATIVE_STD * max(abs(frame.mean()), 1.0))


def _response(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    d_row: float,
    d_col: float,
) -> float:
    """How well the frames agree once aligned by the measured shift, 0 to 1.

    It has to be measured after alignment: correlating the raw frames scores a
    good measurement at zero, since two views of one textured sample, offset,
    share nothing pixel for pixel. The overlap is sliced, never rolled (a roll
    wraps the far edge into the near one). Fewer than 8 pixels of overlap on an
    axis, or a constant overlap, scores 0.
    """
    row, col = round(d_row), round(d_col)
    height, width = a.shape
    if height - abs(row) < _MIN_OVERLAP_PX or width - abs(col) < _MIN_OVERLAP_PX:
        return 0.0
    fixed = a[
        max(-row, 0) : height - max(row, 0),
        max(-col, 0) : width - max(col, 0),
    ]
    moved = b[
        max(row, 0) : height + min(row, 0),
        max(col, 0) : width + min(col, 0),
    ]
    if _is_flat(fixed) or _is_flat(moved):
        return 0.0
    return min(1.0, abs(float(np.corrcoef(fixed.ravel(), moved.ravel())[0, 1])))
