"""The signal measure in its own module (m5 §17.12 D22): moved unchanged, and ``saturated``.

``camera-calibration`` and the guided page ask whether a frame carries light; the
measure moved to ``smc.plugins.frame_signal`` so that the page does not import a
tool's algorithm. The calibration's own tests still import the names from
``camera_calibration.algorithm`` and keep their results; this module proves that the
two are one function, and decides ``saturated`` at its bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from smc.plugins import frame_signal
from smc.plugins.camera_calibration import algorithm
from smc.testing.synthetic import PlateSample

SHAPE = (384, 512)  # (height, width): not square, so a swapped pair shows (FM-73)


def _dark_frame() -> np.ndarray:
    """The Prime 95B's dark frame of session 2: mean 100.5, std 2.3, a column pattern."""
    rng = np.random.default_rng(408)
    frame = 100.5 + rng.normal(0.0, 2.3, SHAPE) + rng.normal(0.0, 1.0, SHAPE[1])
    return np.rint(frame).astype(np.uint16)


def test_the_moved_measure_gives_the_camera_calibrations_results() -> None:
    """The calibration's dark-frame and lit-frame values, through the new module."""
    dark = _dark_frame()
    lit = PlateSample().render(0.0, 0.0, 0.0, shape=SHAPE, pixel_size_um=1.0)

    assert algorithm.carries_signal is frame_signal.carries_signal
    assert algorithm.signal_spread is frame_signal.signal_spread
    assert algorithm.signal_floor_counts is frame_signal.signal_floor_counts
    assert algorithm.MIN_SIGNAL_SPREAD_COUNTS == frame_signal.MIN_SIGNAL_SPREAD_COUNTS
    assert algorithm.MIN_FEATURE_FRACTION == frame_signal.MIN_FEATURE_FRACTION
    assert frame_signal.MIN_SIGNAL_SPREAD_COUNTS == 50.0
    assert frame_signal.MIN_FEATURE_FRACTION == 1e-4
    assert not frame_signal.carries_signal(dark)
    assert frame_signal.carries_signal(lit)
    assert frame_signal.signal_spread(dark) < frame_signal.MIN_SIGNAL_SPREAD_COUNTS
    assert frame_signal.signal_floor_counts(12) == pytest.approx(3.125)
    with pytest.raises(ValueError, match="2-D"):
        frame_signal.signal_spread(np.zeros((4, 4, 3)))


def _with_full_scale_pixels(count: int, *, bit_depth: int) -> np.ndarray:
    """A mid-grey frame of 20 000 pixels with ``count`` at the camera's full scale."""
    frame = np.full((100, 200), 100, dtype=np.uint16)
    frame.reshape(-1)[:count] = 2**bit_depth - 1
    return frame


@pytest.mark.parametrize("bit_depth", [16, 12, 8])
def test_a_frame_with_one_percent_at_full_scale_is_saturated_and_one_pixel_fewer_is_not(
    bit_depth: int,
) -> None:
    """The bound is ``>=`` one percent of the pixels, at the camera's depth and not the dtype's."""
    bound = 200  # exactly one percent of the pixels

    assert frame_signal.SATURATED_FRACTION == 0.01
    assert frame_signal.saturated(
        _with_full_scale_pixels(bound, bit_depth=bit_depth), bit_depth
    )
    assert not frame_signal.saturated(
        _with_full_scale_pixels(bound - 1, bit_depth=bit_depth), bit_depth
    )


def test_a_16_bit_frame_is_not_saturated_for_a_12_bit_camera_unless_it_reaches_4095() -> (
    None
):
    """4095 is the 12-bit full scale: a frame of 4094 everywhere is lit, not blown out."""
    assert not frame_signal.saturated(np.full(SHAPE, 4094, dtype=np.uint16), 12)
    assert frame_signal.saturated(np.full(SHAPE, 4095, dtype=np.uint16), 12)


def test_saturation_is_not_known_without_a_bit_depth() -> None:
    """A depth of 0 is "not known": the full scale is not guessed from the dtype."""
    everything_at_full_scale = np.full(SHAPE, 65535, dtype=np.uint16)

    assert not frame_signal.saturated(everything_at_full_scale, 0)
    assert not frame_signal.saturated(everything_at_full_scale, -1)
    assert frame_signal.saturated(everything_at_full_scale, 16)


def test_saturated_reads_an_empty_frame_as_not_saturated() -> None:
    assert not frame_signal.saturated(np.zeros((0, 0), dtype=np.uint16), 16)
