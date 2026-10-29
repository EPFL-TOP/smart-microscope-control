"""A camera frame as the page keeps and shows it (m5 §2.2): numpy only, no Panel, no device.

The frame is kept exactly as the camera gave it, and what reaches the screen is
a copy: ``display_image`` stretches and strides a copy, so the measurement is
never rescaled and no session's drawing changes what another session sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

__all__ = ["Frame", "display_image"]

#: The percentiles of the stretch: a hot pixel or a dead one must not decide the contrast.
_LOW_PERCENTILE = 0.5
_HIGH_PERCENTILE = 99.5


@dataclass(frozen=True, slots=True)
class Frame:
    """One snapped frame and what the page needs to say of it.

    ``image`` is the camera's own array, never rescaled and read-only, so that
    the sessions that draw it cannot change it. ``number`` counts the
    controller's frames from 1: a view redraws when it changes, never on the
    tick alone (FM-145, FM-151). ``pixel_size_um`` is ``0.0`` when the camera
    does not know it, as ``smc snap`` writes it, and ``xy_um`` is ``None`` when
    the stand has no XY stage or cannot read it.
    """

    number: int
    image: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
    bit_depth: int
    exposure_ms: float
    pixel_size_um: float
    xy_um: tuple[float, float] | None
    taken_utc: datetime
    summary: str


def display_image(
    image: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
    *,
    bit_depth: int,
    max_side_px: int = 1024,
) -> npt.NDArray[np.uint8]:
    """The frame as 8-bit grey for the screen, in the camera's orientation (row 0 is the top row).

    A frame larger than ``max_side_px`` is strided, not averaged: the screen
    needs the picture, not a measurement, and a stride is cheap enough for live
    (m5 §5). The stretch is the 0.5 to 99.5 percentile of what is shown. A
    **flat** frame has no range to stretch, and stretching it would show noise,
    or a black one white: it is shown against the camera's range (``bit_depth``
    bits, or the dtype's when ``bit_depth`` cannot be that), so a saturated frame
    is white and a dark one black. The function never flips: the page does,
    because Bokeh draws an image's first row at the bottom.

    Raises:
        ValueError: The frame is not 2-D ``uint8`` or ``uint16``; the message
            names its dtype and shape.
    """
    if not (image.ndim == 2 and image.dtype in (np.uint8, np.uint16)):
        raise ValueError(f"a {image.dtype} frame of shape {image.shape} is not shown")
    height, width = image.shape
    stride = max(1, math.ceil(max(height, width) / max_side_px))
    seen = image[::stride, ::stride].astype(np.float64)
    low, high = (
        float(v) for v in np.percentile(seen, (_LOW_PERCENTILE, _HIGH_PERCENTILE))
    )
    if high <= low:
        dtype_bits = 8 * image.dtype.itemsize
        bits = bit_depth if 1 <= bit_depth <= dtype_bits else dtype_bits
        scaled = seen * 255.0 / (2**bits - 1)
    else:
        # Multiplied before it is divided: 255 / (high - low) rounds, and the
        # brightest pixel would show as 254.
        scaled = (seen - low) * 255.0 / (high - low)
    return np.asarray(np.clip(scaled, 0.0, 255.0).astype(np.uint8), dtype=np.uint8)
