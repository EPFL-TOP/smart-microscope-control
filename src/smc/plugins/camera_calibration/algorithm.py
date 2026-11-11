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

from smc.plugins.geometry import Matrix2

__all__ = [
    "DEFAULT_STEP_FRACTION",
    "FALLBACK_STEP_UM",
    "MAX_RESIDUAL_SHARE",
    "MIN_RESPONSE",
    "MIN_SIGNAL_SPREAD_COUNTS",
    "SMALL_STEP_FRACTION",
    "UPSAMPLE",
    "Fit",
    "Shift",
    "carries_signal",
    "fit",
    "measure_shift",
    "precision_reasons",
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
#: A frame whose 1st-to-99th percentile spans fewer counts than this carries no
#: light: it is dark or flat, and no probe of it can be judged. Measured on the
#: Prime 95B (16-bit): its two dark frames of session 2 span 11 counts (std 2.3,
#: mean 100.5), the dimmest of 200 lit frames spans 177, and one stuck pixel
#: makes a dark frame's ``max - min`` 65000, which is why it is the percentiles.
#: A ratio to the noise cannot tell a dark frame from a lit blank one (both read
#: about 5), so the floor is in counts, for a 16-bit camera.
MIN_SIGNAL_SPREAD_COUNTS = 50.0
#: A probe under this fraction of the frame's smaller side is a few pixels, and
#: a few pixels fit poorly: 50 µm at the 4x is 18 px of 1412 (1.3 %), where the
#: good fits of the same session were 311 to 362 px (22 to 26 %).
SMALL_STEP_FRACTION = 0.1
#: A fit whose residual is more than this share of its probe is poorly measured:
#: 21 % at 18 px, against 0.1 to 0.5 % for the three good fits of session 3.
MAX_RESIDUAL_SHARE = 0.05

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
        notes.append("the frames barely correlate: find a field with texture")
    a, b = shift_x.d_col_px / step_um, shift_y.d_col_px / step_um
    c, d = shift_x.d_row_px / step_um, shift_y.d_row_px / step_um
    determinant = a * d - b * c
    if not abs(determinant) >= _MIN_DETERMINANT:
        notes.append("the two moves gave parallel shifts: did the stage move?")
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


def signal_spread(frame: npt.NDArray[Any]) -> float:
    """How many counts the middle 98 % of the frame spans: the 99th percentile minus the 1st.

    A frame with no light on the camera (the light path on the eyepiece, a
    closed shutter) is the camera's black level and its fixed pattern, and that
    pattern correlates well at a zero shift: the calibration then measures two
    shifts of zero and blames the stage (F8, Ti2-E session 3). The spread tells
    such a frame from one with a sample before anything moves. Percentiles, not
    ``max - min``: a hot or stuck pixel makes the range of a dark frame as wide
    as a lit one's.

    Raises:
        ValueError: The frame is not 2-D or holds a value that is not finite.
    """
    array = _as_frame(frame, "camera")
    low, high = np.percentile(array, (1.0, 99.0))
    return float(high - low)


def carries_signal(frame: npt.NDArray[Any]) -> bool:
    """Whether the frame spans at least ``MIN_SIGNAL_SPREAD_COUNTS``: it is not dark or flat.

    A positive test (``>=``), so that a spread that is not a number is not
    taken for signal.

    Raises:
        ValueError: The frame is not 2-D or holds a value that is not finite.
    """
    return signal_spread(frame) >= MIN_SIGNAL_SPREAD_COUNTS


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


def _as_frame(frame: npt.NDArray[Any], name: str) -> npt.NDArray[np.float64]:
    """``frame`` as float64, before any arithmetic (an unsigned frame minus its mean wraps, FM-75)."""
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"the {name} frame must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"the {name} frame has a value that is not finite")
    return array


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
