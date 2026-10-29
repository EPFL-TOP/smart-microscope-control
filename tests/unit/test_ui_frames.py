"""The conversion of a camera frame to the 8-bit grey the page shows (m5 §2.2, #28).

Every frame is non-square, so that a transposed or swapped axis cannot pass (FM-73),
and the expected values are worked out here from the stretch's definition, not read
from the code.
"""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest

from smc.ui.frames import display_image


def _rows_ramp(rows: int, columns: int, step: int) -> npt.NDArray[np.uint16]:
    """Row ``r`` is ``r * step`` everywhere: darkest at the top, so a flip shows."""
    column = (np.arange(rows, dtype=np.uint16) * step).astype(np.uint16)
    return np.repeat(column[:, None], columns, axis=1)


def test_display_keeps_the_cameras_orientation() -> None:
    shown = display_image(_rows_ramp(6, 10, 1000), bit_depth=16)
    assert shown.shape == (6, 10)
    assert shown.dtype == np.uint8
    assert (shown[0] == 0).all()
    assert (shown[-1] == 255).all()
    # Monotonic from the top row down: no flip, and no transposition on 6 x 10.
    assert [int(row[0]) for row in shown] == sorted(int(row[0]) for row in shown)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((512, 2048), (256, 1024)), ((1200, 1200), (600, 600)), ((300, 700), (300, 700))],
)
def test_display_strides_a_large_frame_to_at_most_max_side(
    shape: tuple[int, int], expected: tuple[int, int]
) -> None:
    # Rows and columns differ, so a stride applied to one axis only is seen.
    image = np.arange(shape[0] * shape[1], dtype=np.uint32).astype(np.uint16)
    image = image.reshape(shape)
    shown = display_image(image, bit_depth=16, max_side_px=1024)
    assert shown.shape == expected


def test_the_stride_takes_every_nth_pixel_and_not_an_average() -> None:
    # Odd columns carry the high value: a stride of 2 shows only the even ones, an
    # average would show a grey between them.
    image = np.zeros((6, 8), dtype=np.uint16)
    image[:, 1::2] = 4000
    image[0, 0] = 1000
    shown = display_image(image, bit_depth=16, max_side_px=4)
    assert shown.shape == (3, 4)
    assert (shown[1:] == 0).all()


@pytest.mark.parametrize(
    ("value", "bit_depth", "expected"),
    [
        (65535, 16, 255),
        (0, 16, 0),
        (4095, 12, 255),
        (2048, 12, 127),
        (255, 8, 255),
    ],
)
def test_a_flat_frame_is_shown_against_the_cameras_range(
    value: int, bit_depth: int, expected: int
) -> None:
    flat = np.full((6, 10), value, dtype=np.uint16)
    shown = display_image(flat, bit_depth=bit_depth)
    assert shown.shape == (6, 10)
    assert (shown == expected).all()


@pytest.mark.parametrize("bit_depth", [0, 20, -3])
def test_a_bit_depth_the_dtype_cannot_hold_falls_back_to_its_maximum(
    bit_depth: int,
) -> None:
    flat = np.full((6, 10), 65535, dtype=np.uint16)
    assert (display_image(flat, bit_depth=bit_depth) == 255).all()
    half = np.full((6, 10), 32768, dtype=np.uint16)
    assert (display_image(half, bit_depth=bit_depth) == 127).all()
    # An 8-bit frame's maximum is 255, whatever the camera claims.
    eight = np.full((6, 10), 255, dtype=np.uint8)
    assert (display_image(eight, bit_depth=16) == 255).all()


def test_a_frame_with_a_range_is_stretched_between_its_percentiles() -> None:
    # One hot pixel (60000) and one dead one (0) in 2000, over a body between 1000
    # and 2000: both are beyond the 0.5 and 99.5 percentiles, so the body spans the
    # grey range and the two outliers only clip.
    rng = np.random.default_rng(0)
    image = rng.integers(1000, 2001, size=(40, 50)).astype(np.uint16)
    image[0, 0] = 60000
    image[39, 49] = 0
    shown = display_image(image, bit_depth=16)
    assert shown[0, 0] == 255
    assert shown[39, 49] == 0
    body = shown[(image >= 1000) & (image <= 2000)]
    assert body.min() < 20
    assert body.max() > 235


@pytest.mark.parametrize(
    ("image", "said"),
    [
        (np.zeros((4, 6, 3), dtype=np.uint8), "a uint8 frame of shape (4, 6, 3)"),
        (np.zeros((4, 6), dtype=np.int16), "a int16 frame of shape (4, 6)"),
        (np.zeros((4, 6), dtype=np.float32), "a float32 frame of shape (4, 6)"),
        (np.zeros((4, 6), dtype=np.uint32), "a uint32 frame of shape (4, 6)"),
        (np.zeros(6, dtype=np.uint16), "a uint16 frame of shape (6,)"),
    ],
    ids=["3d-uint8", "int16", "float32", "uint32", "1d"],
)
def test_a_frame_that_is_not_2d_unsigned_is_refused(
    image: npt.NDArray[np.generic], said: str
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(said)} is not shown$"):
        display_image(image, bit_depth=16)  # type: ignore[arg-type]  # the refusal is the point


def test_the_frame_is_never_changed() -> None:
    image = _rows_ramp(6, 10, 1000)
    image.flags.writeable = False
    before = image.copy()
    display_image(image, bit_depth=16)
    assert (image == before).all()
