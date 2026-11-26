"""The blob detector, for the synthetic plate of the demo-sample profile (m3 §9.1, #260).

numpy and scikit-image only: no torch and no weights, so CI runs all of it.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pytest

from smc.inference.blobs import BLOB_DETECTOR, BLOB_DETECTOR_IDENTITY, BlobDetector
from smc.inference.detector import Detection, Detector
from smc.testing import Blob, PlateSample

PIXEL_SIZES_UM = [0.1625, 0.25, 0.5, 1.0, 2.0]
#: The well's radius on the 96-well plate A1 sits at the stage origin of.
WELL_RADIUS_UM = 3200.0


def _spots(centres: list[tuple[float, float]]) -> list[Blob]:
    """Radius 5 um, 3000 counts: what ``smc.testing.sample_stand`` draws for a cell."""
    return [Blob(x, y, 5.0, 3000.0) for x, y in centres]


def _frame_with(blobs: list[Blob], pixel_size_um: float) -> np.ndarray[Any, Any]:
    sample = PlateSample(blobs=blobs, seed=3)
    return sample.render(0.0, 0.0, 0.0, shape=(512, 512), pixel_size_um=pixel_size_um)


def _labels(found: list[Detection]) -> Counter[str]:
    return Counter(d.label for d in found)


@pytest.mark.parametrize("pixel_size_um", PIXEL_SIZES_UM)
def test_the_blob_detector_finds_singles_and_doublets_at_every_pixel_size(
    pixel_size_um: float,
) -> None:
    w = 512 * pixel_size_um
    singles = [(-0.3 * w, -0.3 * w), (0.3 * w, -0.3 * w), (-0.3 * w, 0.3 * w)]
    singles += [(0.3 * w, 0.3 * w), (0.0, 0.0)]
    blobs = _spots(singles)
    # Three pairs along three angles, the two spots 4 um either side of a centre.
    for (cx, cy), angle in zip(
        [(0.0, 0.3 * w), (0.0, -0.3 * w), (-0.3 * w, 0.0)], [0.0, 1.0, 2.0], strict=True
    ):
        dx, dy = 4.0 * math.cos(angle), 4.0 * math.sin(angle)
        blobs += _spots([(cx - dx, cy - dy), (cx + dx, cy + dy)])

    found = BlobDetector().detect(_frame_with(blobs, pixel_size_um))

    assert _labels(found) == {"single": 5, "doublet": 3}
    assert {d.score for d in found} == {1.0}


@pytest.mark.parametrize("pixel_size_um", PIXEL_SIZES_UM)
def test_the_blob_detector_sees_nothing_across_a_well_wall(
    pixel_size_um: float,
) -> None:
    # A well wall is a step the background must not be mistaken for a cell by:
    # the frame is centred from half a frame inside the well to half outside, and
    # at the diagonal where the wall meets a corner.
    w = 512 * pixel_size_um
    centres = [
        (x, 0.0)
        for x in np.linspace(
            WELL_RADIUS_UM - 256 * pixel_size_um,
            WELL_RADIUS_UM + 256 * pixel_size_um,
            21,
        )
    ]
    centres.append((0.707 * WELL_RADIUS_UM + 0.2 * w, 0.707 * WELL_RADIUS_UM))
    sample = PlateSample(seed=3)

    for x, y in centres:
        frame = sample.render(x, y, 0.0, shape=(512, 512), pixel_size_um=pixel_size_um)
        assert BlobDetector().detect(frame) == [], (x, y)


@pytest.mark.parametrize("pixel_size_um", PIXEL_SIZES_UM)
def test_a_cell_in_the_sliver_of_well_beside_a_wall_is_still_found(
    pixel_size_um: float,
) -> None:
    # 40 % of the frame is well and 60 % plastic. The 75th percentile is then a
    # well pixel, so the sliver is background and the cell stands out of it; a
    # median would sit on the plastic, the whole sliver would be one bright
    # component (over 5 % of the frame, dropped) and the cell in it with it.
    w = 512 * pixel_size_um
    frame_x = WELL_RADIUS_UM + 0.1 * w
    cell = Blob(WELL_RADIUS_UM - 0.2 * w, 0.0, 5.0, 3000.0)
    sample = PlateSample(blobs=[cell], seed=3)
    frame = sample.render(
        frame_x, 0.0, 0.0, shape=(512, 512), pixel_size_um=pixel_size_um
    )

    found = BlobDetector().detect(frame)

    assert _labels(found) == {"single": 1}


def test_a_strip_is_not_a_cell() -> None:
    # A 3 x 60 px bar is bright, 180 px (inside the area limits), and not round:
    # only the axis ratio drops it.
    frame = np.random.default_rng(0).normal(1000, 20, (200, 200))
    frame[100:103, 70:130] += 3000

    assert BlobDetector().detect(frame) == []


def test_a_round_spot_is_a_single_and_an_elongated_one_a_doublet() -> None:
    rows, cols = np.mgrid[0:200, 0:300]
    frame = np.random.default_rng(1).normal(1000, 20, (200, 300))
    frame += 3000 * (((rows - 60) / 8.0) ** 2 + ((cols - 70) / 8.0) ** 2 <= 1)
    # Semi-axes 14 and 7: the minor/major ratio is 0.5, between 0.3 and 0.75.
    frame += 3000 * (((rows - 140) / 7.0) ** 2 + ((cols - 200) / 14.0) ** 2 <= 1)

    found = BlobDetector().detect(frame)

    assert sorted((round((d.row0_px + d.row1_px) / 2), d.label) for d in found) == [
        (60, "single"),
        (140, "doublet"),
    ]


def test_the_box_is_rows_then_columns_of_the_component() -> None:
    # A frame that is not square, and a blob that is not a square: a box with
    # the axes exchanged is wrong here and invisible on a square one (FM-73).
    frame = np.random.default_rng(2).normal(1000, 20, (100, 160))
    frame[20:30, 40:52] += 3000

    (box,) = BlobDetector().detect(frame)

    assert (box.row0_px, box.col0_px, box.row1_px, box.col1_px) == (20, 40, 30, 52)
    assert box.label == "single"


@pytest.mark.parametrize(
    ("side", "kept"), [(2, False), (3, True), (4, True)], ids=["4px", "9px", "16px"]
)
def test_a_component_under_nine_pixels_is_dropped(side: int, kept: bool) -> None:
    frame = np.random.default_rng(3).normal(1000, 20, (100, 100))
    frame[40 : 40 + side, 40 : 40 + side] += 3000

    assert (len(BlobDetector().detect(frame)) == 1) is kept


def test_a_component_over_five_percent_of_the_frame_is_dropped() -> None:
    # 100 x 100 is 10 000 px: a 22 x 22 blob (484 px) is under 5 % (500 px), a
    # 23 x 23 one (529 px) over it.
    frame = np.random.default_rng(4).normal(1000, 20, (100, 100))
    small = frame.copy()
    small[10:32, 10:32] += 3000
    large = frame.copy()
    large[10:33, 10:33] += 3000

    assert len(BlobDetector().detect(small)) == 1
    assert BlobDetector().detect(large) == []


def test_the_blob_detector_refuses_a_frame_that_is_not_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        BlobDetector().detect(np.zeros((40, 50, 3), np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        BlobDetector().detect(np.zeros(40, np.uint8))


@pytest.mark.parametrize("shape", [(0, 0), (5, 0), (0, 5), (5, 1), (1, 1)])
def test_a_frame_too_small_to_measure_gives_no_detection(
    shape: tuple[int, int],
) -> None:
    assert BlobDetector().detect(np.zeros(shape, np.uint16)) == []


@pytest.mark.parametrize("level", [0, 800, 65535])
def test_a_constant_frame_gives_no_detection(level: int) -> None:
    assert BlobDetector().detect(np.full((64, 80), level, np.uint16)) == []


def test_the_blob_detector_declares_its_classes_and_its_device() -> None:
    detector = BlobDetector()

    assert isinstance(detector, Detector)
    assert detector.classes == ("single", "doublet")
    assert detector.device == "cpu"


def test_the_blob_detectors_identity_is_the_one_the_design_names() -> None:
    # The resume compares the SHA-256 of this string: a change of a rule must
    # change it, and the string is part of the record, so it is pinned.
    assert BLOB_DETECTOR == "blobs"
    assert BLOB_DETECTOR_IDENTITY == (
        "smc BlobDetector 1: p75 + 8 sigma, 9 px to 5 % of the frame, "
        "axis ratio >= 0.3, single from 0.75"
    )
