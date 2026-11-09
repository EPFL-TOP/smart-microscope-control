"""The focus on the cells' pure part: the scores, the planes and the peak (m3 §21.3)."""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest

from smc.plugins.cell_focus.algorithm import (
    MAX_PLANES,
    MIN_RISE,
    SECOND_PEAK_FRACTION,
    ScoreName,
    find_peak,
    first_outside,
    plan_planes,
    score_frame,
)
from smc.testing.synthetic import PlateSample

#: Inside well A1 of a plate whose A1 centre is the origin: the texture of the
#: well, at a pixel of 1 µm, is what the demo-sample's 10x shows.
_SHAPE = (128, 128)
_SCORES: tuple[ScoreName, ...] = get_args(ScoreName)


def _stack(
    planes_um: list[float], *, noise_std: float = 20.0, seed: int = 0
) -> list[np.ndarray]:
    sample = PlateSample(noise_std=noise_std, seed=seed)
    return [
        sample.render(0.0, 0.0, z, shape=_SHAPE, pixel_size_um=1.0) for z in planes_um
    ]


@pytest.mark.parametrize("noise_std", [20.0, 0.0])
@pytest.mark.parametrize("name", _SCORES)
def test_each_score_peaks_at_the_sharp_plane(name: ScoreName, noise_std: float) -> None:
    planes = [float(z) for z in range(-20, 21, 2)]
    scores = [score_frame(frame, name) for frame in _stack(planes, noise_std=noise_std)]
    assert planes[int(np.argmax(scores))] == 0.0


def test_the_peak_is_refined_between_planes() -> None:
    planes = list(plan_planes(1.3, 5.0, 1.0))
    scores = [score_frame(frame, "brenner") for frame in _stack(planes)]
    peak = find_peak(planes, scores)
    assert peak.outcome == "peak"
    assert peak.z_um is not None
    assert abs(peak.z_um) <= 0.3
    assert all(peak.z_um != plane for plane in planes)


def _rising(rise: float) -> list[float]:
    """Five scores whose (max - min) / max is ``rise``, with a peak in the middle."""
    return [1.0 - rise, 1.0 - rise / 2, 1.0, 1.0 - rise / 2, 1.0 - rise]


_Z5 = [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_flat_sweep_keeps_no_peak() -> None:
    assert 0.04 < MIN_RISE < 0.06
    flat = find_peak(_Z5, _rising(0.04))
    assert (flat.outcome, flat.best_index, flat.z_um) == ("flat", 2, None)
    assert find_peak(_Z5, _rising(0.06)).outcome == "peak"
    # Nothing above zero is no signal at all, whatever its rise.
    assert find_peak(_Z5, [0.0, 0.0, 0.0, 0.0, 0.0]).outcome == "flat"
    assert find_peak(_Z5, [-3.0, -2.0, -1.0, -2.0, -3.0]).outcome == "flat"


def test_a_best_at_the_end_is_at_edge() -> None:
    first = find_peak(_Z5, [1.0, 0.8, 0.6, 0.5, 0.4])
    assert (first.outcome, first.best_index, first.z_um) == ("at-edge", 0, None)
    last = find_peak(_Z5, [0.4, 0.5, 0.6, 0.8, 1.0])
    assert (last.outcome, last.best_index) == ("at-edge", 4)


def _two(second: float) -> list[float]:
    """A best of 1.0 at index 1 and a second maximum whose rise is ``second`` of the best's."""
    low = 0.5
    return [low, 1.0, low, low + second * (1.0 - low), low, low]


def test_two_maxima_are_two_peaks() -> None:
    assert 0.4 < SECOND_PEAK_FRACTION < 0.6
    z = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    two = find_peak(z, _two(0.6))
    assert (two.outcome, two.best_index, two.z_um) == ("two-peaks", 1, None)
    one = find_peak(z, _two(0.4))
    assert (one.outcome, one.best_index) == ("peak", 1)
    # An equal maximum later on is a second peak, not a tie to break.
    assert find_peak(z, [0.5, 1.0, 0.5, 1.0, 0.5, 0.5]).outcome == "two-peaks"


def test_unequal_spacing_is_refined_through_the_readbacks() -> None:
    peak = find_peak([0.0, 1.0, 3.0], [1.0, 2.0, 1.0])
    assert peak.outcome == "peak"
    assert peak.z_um == pytest.approx(1.5)
    # The same three points on an even spacing put the vertex on the best plane.
    assert find_peak([0.0, 1.0, 2.0], [1.0, 2.0, 1.0]).z_um == pytest.approx(1.0)
    # A neighbour almost as high as the best pulls the vertex toward it, and
    # it stays between the best's neighbours.
    lean = find_peak([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0, 1.0 - 1e-12, 0.0])
    assert lean.outcome == "peak"
    assert lean.z_um is not None
    assert 2.0 < lean.z_um <= 3.0


def test_plan_planes_is_symmetric_and_bounded() -> None:
    planes = plan_planes(10.0, 5.0, 1.0)
    assert planes == tuple(float(z) for z in range(5, 16))
    # One step on each side is the smallest sweep the parameters allow.
    assert plan_planes(0.0, 1.0, 1.0) == (-1.0, 0.0, 1.0)
    # The half-range is cut down to whole steps, never stretched past it.
    assert plan_planes(0.0, 2.5, 1.0) == (-2.0, -1.0, 0.0, 1.0, 2.0)
    # A range that is a whole number of steps in decimal keeps its ends.
    assert plan_planes(0.0, 0.3, 0.1)[0] == pytest.approx(-0.3)
    assert len(plan_planes(0.0, 0.3, 0.1)) == 7
    for bad in (math.nan, math.inf, -1.0, 0.0):
        with pytest.raises(ValueError, match="range_um"):
            plan_planes(0.0, bad, 1.0)
        with pytest.raises(ValueError, match="step_um"):
            plan_planes(0.0, 5.0, bad)
    with pytest.raises(ValueError, match="centre_um"):
        plan_planes(math.nan, 5.0, 1.0)
    # Less than one step on each side is not a sweep.
    with pytest.raises(ValueError, match=r"range_um=0\.5 holds no step"):
        plan_planes(0.0, 0.5, 1.0)


def test_too_many_planes_is_refused() -> None:
    half = (MAX_PLANES - 1) // 2
    assert len(plan_planes(0.0, float(half), 1.0)) == MAX_PLANES
    with pytest.raises(ValueError, match=f"{MAX_PLANES + 2}"):
        plan_planes(0.0, float(half + 1), 1.0)


def test_first_outside_names_the_first_plane_past_a_limit() -> None:
    planes = (980.0, 990.0, 1000.0, 1010.0, 1020.0)
    assert first_outside(planes, -1000.0, 1000.0) == 1010.0
    assert first_outside(planes, 985.0, 2000.0) == 980.0
    assert first_outside(planes, 980.0, 1020.0) is None
    assert first_outside((0.0, math.nan), -1.0, 1.0) is not None


def test_a_dark_frame_scores_zero() -> None:
    dark = np.zeros((16, 16), dtype=np.uint16)
    for name in _SCORES:
        assert score_frame(dark, name) == 0.0
    nan = np.full((16, 16), np.nan)
    assert score_frame(nan, "brenner") == 0.0
    with pytest.raises(ValueError, match="2-D"):
        score_frame(np.ones((4, 4, 3)), "brenner")
    with pytest.raises(ValueError, match="3 x 3"):
        score_frame(np.ones((2, 8)), "brenner")


def test_find_peak_refuses_nan_and_unsorted_z() -> None:
    with pytest.raises(ValueError, match="finite"):
        find_peak([0.0, 1.0, 2.0], [1.0, math.nan, 1.0])
    with pytest.raises(ValueError, match="finite"):
        find_peak([0.0, math.nan, 2.0], [1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="increasing"):
        find_peak([0.0, 2.0, 1.0], [1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="increasing"):
        find_peak([0.0, 1.0, 1.0], [1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        find_peak([0.0, 1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 3"):
        find_peak([0.0, 1.0], [1.0, 2.0])
