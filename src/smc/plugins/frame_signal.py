"""Whether a frame has light on it, and whether it is blown out: pure, numpy only.

Two callers ask the same question about a frame: ``camera-calibration`` before it
moves anything (a calibration of a dark frame blames the stage, F8, Ti2-E session
3) and the guided page, which tells the operator that the frame in Live is dark
(m5 §17.12 D22). The measure lives here so that the page does not import a tool's
algorithm module (and, through it, scikit-image) to read a percentile, and so that
the two callers cannot disagree on what "dark" is. ``camera_calibration/algorithm``
imports these names back, so that nothing that imported them there changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "MIN_FEATURE_FRACTION",
    "MIN_SIGNAL_SPREAD_COUNTS",
    "SATURATED_FRACTION",
    "as_frame",
    "carries_signal",
    "saturated",
    "signal_floor_counts",
    "signal_spread",
]

#: A frame whose 1st-to-99th percentile spans fewer counts than this carries no
#: light: it is dark or flat, and no probe of it can be judged. Measured on the
#: Prime 95B (16-bit): its two dark frames of session 2 span 11 counts (std 2.3,
#: mean 100.5), the dimmest of 200 lit frames spans 177, and one stuck pixel
#: makes a dark frame's ``max - min`` 65000, which is why it is the percentiles.
#: A ratio to the noise cannot tell a dark frame from a lit blank one (both read
#: about 5), so the floor is in counts, for a 16-bit camera.
MIN_SIGNAL_SPREAD_COUNTS = 50.0
#: A frame also carries signal when this fraction of its pixels lies at least the
#: floor away from its median: a few beads on a dark background cover 0.08 to
#: 0.4 % of the frame, where a dark frame has none (the two real ones: 0 pixels
#: of 2 million) and a hot pixel is one in a million.
MIN_FEATURE_FRACTION = 1e-4
#: A frame is saturated when at least this fraction of its pixels sits at the
#: camera's full scale. Assumed (one percent), not measured on a stand: the docs
#: say so, and a stand's own frames decide it later (m5 §17.12 D22).
SATURATED_FRACTION = 0.01


def as_frame(frame: npt.NDArray[Any], name: str) -> npt.NDArray[np.float64]:
    """``frame`` as float64, before any arithmetic (an unsigned frame minus its mean wraps, FM-75)."""
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"the {name} frame must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"the {name} frame has a value that is not finite")
    return array


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
    array = as_frame(frame, "camera")
    low, high = np.percentile(array, (1.0, 99.0))
    return float(high - low)


def signal_floor_counts(bit_depth: int = 16) -> float:
    """``MIN_SIGNAL_SPREAD_COUNTS`` for a camera of this bit depth: 50 at 16 bits, 3.125 at 12.

    The floor was measured on a 16-bit camera. A shallower one (a 12-bit mode of
    the same Prime 95B) gives counts 16 times smaller for the same light, so a
    lit frame would read under the 16-bit floor and be refused as dark: the
    floor is scaled down with the depth, which can only let a dark frame through
    (the old behaviour), never refuse a lit one. A depth above 16 is 16.
    """
    bits = min(max(int(bit_depth), 1), 16)
    return MIN_SIGNAL_SPREAD_COUNTS * 2.0 ** (bits - 16)


def carries_signal(frame: npt.NDArray[Any], bit_depth: int = 16) -> bool:
    """Whether the frame has light on it: it is not dark or flat.

    Two ways to carry signal, the single decision of every caller (a boundary
    tested here is the boundary of the plugin):

    * the 1st to 99th percentile spans at least ``signal_floor_counts``, which is
      any frame with a sample that fills a few percent of the field;
    * at least ``MIN_FEATURE_FRACTION`` of the pixels lie that far from the
      median: a **sparse** field (a few fluorescent beads or cells on a dark
      background) covers under 1 % of the pixels, so its percentiles read as
      dark, and a handful of isolated hot pixels is far below that fraction.

    Both are positive tests (``>=``), so that a number that is not one is not
    taken for signal.

    Raises:
        ValueError: The frame is not 2-D or holds a value that is not finite.
    """
    floor = signal_floor_counts(bit_depth)
    if signal_spread(frame) >= floor:
        return True
    array = as_frame(frame, "camera")
    far = int(np.count_nonzero(np.abs(array - np.median(array)) >= floor))
    return far >= MIN_FEATURE_FRACTION * array.size


def saturated(frame: npt.NDArray[Any], bit_depth: int) -> bool:
    """Whether at least ``SATURATED_FRACTION`` of the pixels sit at the camera's full scale.

    A frame at full scale is blown out, and the light path is not the remedy a
    dark frame has: the exposure is (FM-222 gave a frame at full scale its own
    remedy). The full scale is ``2 ** bit_depth - 1`` of the **camera's** depth,
    not the array's dtype: a 12-bit camera in a uint16 array tops out at 4095. A
    depth that is not known (zero or less) says nothing, because a full scale
    guessed from the dtype would call every bright frame of a 12-bit camera
    unsaturated and every dark one of an 8-bit camera saturated.
    """
    if bit_depth <= 0:
        return False
    array = np.asarray(frame)
    if array.size == 0:
        return False
    at_full_scale = int(np.count_nonzero(array >= float(2**bit_depth - 1)))
    return at_full_scale >= SATURATED_FRACTION * array.size
