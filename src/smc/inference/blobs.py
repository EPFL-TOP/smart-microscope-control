"""The blob detector: bright round spots, for the synthetic plate of the demo-sample profile (m3 §9.1).

numpy and scikit-image only: it needs neither torch nor weights, and it says in
every result that what it counts are spots, not cells. Not imported by
``smc.inference``'s ``__init__``, so importing the package does not load
scikit-image.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from skimage import measure

from smc.inference.detector import Detection

__all__ = ["BLOB_DETECTOR", "BLOB_DETECTOR_IDENTITY", "BlobDetector"]

#: ``CellsResult.model`` when the blob detector ran: there is no file to name.
BLOB_DETECTOR = "blobs"
#: What the blob detector is, rule by rule. Its SHA-256 is ``CellsResult.model_sha256``
#: and the resume's check that two runs found cells the same way (FM-111): a change
#: of a rule below changes the version in this string.
BLOB_DETECTOR_IDENTITY = (
    "smc BlobDetector 1: p75 + 8 sigma, 9 px to 5 % of the frame, "
    "axis ratio >= 0.3, single from 0.75"
)
# scikit-image's `label` and `regionprops` are unannotated, which `mypy --strict`
# refuses at the call (`no-untyped-call`). Bound once, with what they return
# (a label image, and region objects that have the properties used below).
_label_image: Callable[..., npt.NDArray[np.int_]] = measure.label
_region_properties: Callable[..., Sequence[Any]] = measure.regionprops
_BLOB_SIGMAS = 8.0
_BLOB_MIN_AREA_PX = 9
_BLOB_MAX_AREA_FRACTION = 0.05
_BLOB_MIN_AXIS_RATIO = 0.3
_BLOB_SINGLE_AXIS_RATIO = 0.75


class BlobDetector:
    """Bright round spots as ``single`` and ``doublet``: the detector for the synthetic plate.

    The trained model does not see the cells of the demo-sample stand (at most 2
    ``single`` among 14 objects, m1 §8), so the workflow could not be followed on
    images past the scan. This finds what that sample draws and nothing else: it
    knows nothing of cells, and the result says so.

    The background is the frame's 75th percentile, so that the cells, a few
    percent of the frame, do not lift it. The noise is the spread of the
    differences of neighbouring pixels along a row, not of the pixels around the
    median: a well wall is a step, which a spread around the median counts as
    noise (a frame centred on a wall gave five false cells at 1 µm per pixel).
    A component is a cell between 9 pixels and 5 % of the frame: a larger one is
    a sliver of well seen past a wall, and at 2 % the doublets vanished at
    0.1625 µm per pixel. A strip is not a cell. A round component is a
    ``single``, an elongated one a ``doublet``.

    ``score`` is always 1.0: a threshold on it would mean nothing here.
    """

    classes: tuple[str, ...] = ("single", "doublet")
    device: str = "cpu"

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The spots of one frame, in no particular order; none on a flat or noiseless one.

        Raises:
            ValueError: ``frame`` is not 2-D.
        """
        array = np.asarray(frame)
        if array.ndim != 2:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        values = array.astype(np.float64)
        if values.shape[0] < 1 or values.shape[1] < 2:
            return []
        sigma = (
            1.4826 * float(np.median(np.abs(np.diff(values, axis=1)))) / math.sqrt(2)
        )
        if not (math.isfinite(sigma) and sigma > 0.0):
            return []
        threshold = float(np.percentile(values, 75)) + _BLOB_SIGMAS * sigma
        largest_px = _BLOB_MAX_AREA_FRACTION * values.size
        found: list[Detection] = []
        for region in _region_properties(_label_image(values > threshold)):
            major = float(region.axis_major_length)
            if (
                region.area < _BLOB_MIN_AREA_PX
                or region.area > largest_px
                or major <= 0
            ):
                continue
            ratio = float(region.axis_minor_length) / major
            if ratio < _BLOB_MIN_AXIS_RATIO:
                continue
            row0, col0, row1, col1 = region.bbox
            found.append(
                Detection(
                    float(row0),
                    float(col0),
                    float(row1),
                    float(col1),
                    "single" if ratio >= _BLOB_SINGLE_AXIS_RATIO else "doublet",
                    1.0,
                )
            )
        return found
