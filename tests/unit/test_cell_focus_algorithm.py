"""The focus on the cells' pure part: the scores, the planes and the peak (m3 §21.3)."""

from __future__ import annotations

import itertools
import math
from typing import get_args

import numpy as np
import pytest

from smc.plugins.cell_focus.algorithm import (
    MAX_PLANES,
    MIN_PLANES,
    NOISE_FACTOR,
    SECOND_PEAK_FRACTION,
    ScoreName,
    find_peak,
    first_outside,
    plan_planes,
    score_frame,
    sweep_size_problem,
)
from smc.plugins.cell_focus.plugin import FocusOnCellsParams
from smc.plugins.scan.plugin import ScanParams
from smc.testing.synthetic import PlateSample

#: Inside well A1 of a plate whose A1 centre is the origin: the texture of the
#: well, at a pixel of 1 µm, is what the demo-sample's 10x shows.
_SHAPE = (128, 128)
_SCORES: tuple[ScoreName, ...] = get_args(ScoreName)

# Sweeps recorded at the Ti2-E's 40x in session 6 (#620; 21 planes of 1 um around
# the well's centre, PFS off), copied from the run document as numbers: no frame.
# B4_0004, centre 4193.3: peak at plane 9, refined 4192.5217, prominence 33.6677
_B4_0004_Z_UM = (
    4183.3,
    4184.3,
    4185.3,
    4186.3,
    4187.28,
    4188.28,
    4189.32,
    4190.3,
    4191.3,
    4192.3,
    4193.3,
    4194.3,
    4195.3,
    4196.3,
    4197.3,
    4198.3,
    4199.3,
    4200.3,
    4201.3,
    4202.3,
    4203.3,
)
_B4_0004_SCORES = (
    0.00139743698,
    0.00139911296,
    0.00139704442,
    0.00139837237,
    0.00140177954,
    0.0014018067,
    0.0014152674,
    0.0014276851,
    0.00144334738,
    0.00145248545,
    0.0014430839,
    0.00143740949,
    0.00143486179,
    0.00143996228,
    0.00144094549,
    0.00142674549,
    0.00142093785,
    0.00141542296,
    0.00140231121,
    0.00140453686,
    0.00139540492,
)
# B4_0014, centre 4193.3: peak at plane 8, refined 4191.5151, prominence 10.7391
_B4_0014_Z_UM = (
    4183.3,
    4184.3,
    4185.3,
    4186.3,
    4187.3,
    4188.3,
    4189.3,
    4190.3,
    4191.3,
    4192.3,
    4193.3,
    4194.3,
    4195.3,
    4196.3,
    4197.3,
    4198.3,
    4199.3,
    4200.3,
    4201.3,
    4202.3,
    4203.3,
)
_B4_0014_SCORES = (
    0.0013935776,
    0.00139915938,
    0.0013933132,
    0.00139682212,
    0.00139730956,
    0.00139582828,
    0.00139880175,
    0.00140306453,
    0.00140148021,
    0.00140196352,
    0.00140180481,
    0.00139816479,
    0.00139322092,
    0.00139373373,
    0.00139756622,
    0.00139678306,
    0.00139299282,
    0.00139593415,
    0.00138899764,
    0.00139007461,
    0.0013882421,
)
# B4_0020, centre 4193.3, an empty field: flat, smoothed best at plane 9,
# prominence 8.8511
_B4_0020_Z_UM = (
    4183.3,
    4184.3,
    4185.3,
    4186.28,
    4187.3,
    4188.3,
    4189.3,
    4190.3,
    4191.3,
    4192.3,
    4193.3,
    4194.3,
    4195.3,
    4196.3,
    4197.3,
    4198.3,
    4199.3,
    4200.3,
    4201.3,
    4202.28,
    4203.3,
)
_B4_0020_SCORES = (
    0.00127949706,
    0.00128168104,
    0.00127683047,
    0.00127981936,
    0.00127777004,
    0.00127467709,
    0.00127803269,
    0.00127866502,
    0.00128566147,
    0.00128459992,
    0.00128104201,
    0.00127869753,
    0.00127891373,
    0.00127638123,
    0.00128000095,
    0.00127954869,
    0.00127727358,
    0.0012764879,
    0.00127622865,
    0.00127550829,
    0.0012742968,
)
# np.random.default_rng(0).normal(1.0, 0.003, 21), rounded to 6 decimals, on
# z = 0..20: flat, prominence 5.0737. Copied as numbers, so that no numpy
# release can change the draw.
_NOISE_ONLY = (
    1.000377,
    0.999604,
    1.001921,
    1.000315,
    0.998393,
    1.001085,
    1.003912,
    1.002841,
    0.997889,
    0.996204,
    0.99813,
    1.000124,
    0.993025,
    0.999344,
    0.996262,
    0.997803,
    0.998367,
    0.999051,
    1.001235,
    1.003128,
    0.999614,
)


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


_Z9 = [float(k) for k in range(9)]
_Z21 = [float(k) for k in range(21)]


def test_a_recorded_sweep_on_cells_peaks_below_the_by_eye_focus() -> None:
    # Session 6's B4_0004 (#622): a large cell, a broad maximum 9 to 11 planes
    # wide at half its rise, with the noise of the 40x's exposure on it. MIN_RISE
    # called it flat; without the dip a second maximum would make it two-peaks.
    peak = find_peak(_B4_0004_Z_UM, _B4_0004_SCORES)
    assert peak.outcome == "peak"
    assert peak.best_index == 9
    assert peak.z_um == pytest.approx(4192.52, abs=0.01)
    assert peak.prominence == pytest.approx(33.67, abs=0.01)


def test_the_noise_factor_lies_between_an_empty_field_and_a_cell() -> None:
    # The two sides of NOISE_FACTOR, with values that differ from it (FM-73): an
    # empty field at 8.85, a field with one cell at the frame's edge at 10.74.
    assert 8.85 < NOISE_FACTOR < 10.74
    empty = find_peak(_B4_0020_Z_UM, _B4_0020_SCORES)
    assert (empty.outcome, empty.best_index, empty.z_um) == ("flat", 9, None)
    assert empty.prominence == pytest.approx(8.85, abs=0.01)
    cell = find_peak(_B4_0014_Z_UM, _B4_0014_SCORES)
    assert (cell.outcome, cell.best_index) == ("peak", 8)
    assert cell.z_um == pytest.approx(4191.52, abs=0.01)
    assert cell.prominence == pytest.approx(10.74, abs=0.01)


def test_the_default_sweep_finds_the_weakest_recorded_cell_peak() -> None:
    # The two defaults (D11): the scan and the tool sweep the same 21 planes, the
    # sweep the rule was fitted to. B4_0014 is the weakest of session 6's 17 cell
    # peaks (B4_0004 is a peak at 11 planes too, so it could not fail on the old
    # defaults): cut to the default planes it is a peak at 10.74, and cut to the
    # old +-5 um it is flat at 6.32.
    tool = FocusOnCellsParams()
    scan = ScanParams()
    assert (tool.range_um, tool.step_um) == (scan.focus_range_um, scan.focus_step_um)
    planes = plan_planes(0.0, tool.range_um, tool.step_um)
    assert len(planes) == 21
    assert all(b - a == pytest.approx(1.0) for a, b in itertools.pairwise(planes))
    k = len(planes) // 2
    z = _B4_0014_Z_UM[10 - k : 11 + k]
    scores = _B4_0014_SCORES[10 - k : 11 + k]
    default = find_peak(z, scores)
    assert (default.outcome, default.best_index) == ("peak", 8)
    assert default.z_um == pytest.approx(4191.52, abs=0.01)
    assert default.prominence == pytest.approx(10.74, abs=0.01)
    old = find_peak(_B4_0014_Z_UM[5:16], _B4_0014_SCORES[5:16])
    assert (old.outcome, old.z_um) == ("flat", None)
    assert old.prominence == pytest.approx(6.32, abs=0.01)


def test_a_peak_narrower_than_a_step_refines_to_its_own_plane() -> None:
    # One plane far above neighbours that do not move: the smoothed scores of
    # planes 3, 4 and 5 are equal, and the first of them put Z half a step low (D12).
    narrow = find_peak(_Z9, [0.0, 1.0, 0.0, 1.0, 40.0, 0.0, 1.0, 0.0, 1.0])
    assert (narrow.outcome, narrow.best_index) == ("peak", 4)
    assert narrow.z_um == pytest.approx(4.0)
    assert narrow.prominence == pytest.approx(19.08, abs=0.01)


def test_a_sweep_whose_second_differences_are_mostly_zero_is_flat() -> None:
    # More than half the second differences are exactly 0, so their median is 0
    # and no robust noise is left (D13): the noise is read from their mean, and
    # the rise of a bare blip is not 10 of them. Without the guard each of these
    # was a peak of prominence inf. 1.97 and not 2.47 pins the sqrt(pi / 2).
    for scores in (
        [5.0, 5.0, 5.0, 5.0, 5.0001, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
    ):
        blip = find_peak(_Z9, scores)
        assert (blip.outcome, blip.z_um) == ("flat", None)
        assert blip.prominence == pytest.approx(1.97, abs=0.01)
    # A straight line has every second difference 0: no noise at all, and the
    # rise stays unbounded.
    line = find_peak(_Z9, [float(k) for k in range(9)])
    assert (line.outcome, line.z_um) == ("at-edge", None)
    assert line.prominence == math.inf


def test_a_noise_only_sweep_is_flat() -> None:
    noise = find_peak(_Z21, _NOISE_ONLY)
    assert (noise.outcome, noise.z_um) == ("flat", None)
    assert noise.prominence == pytest.approx(5.07, abs=0.01)


def test_a_flat_sweep_keeps_no_peak() -> None:
    for scores in ([0.0] * 9, [1.0] * 9):
        flat = find_peak(_Z9, scores)
        assert (flat.outcome, flat.z_um, flat.prominence) == ("flat", None, 0.0)
    # Nothing above zero is no signal at all, whatever its rise.
    below = find_peak(_Z9, [-3.0, -2.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])
    assert (below.outcome, below.z_um) == ("flat", None)


def test_a_best_at_the_end_is_at_edge() -> None:
    # A smoothed curve cannot tell a peak on the second plane from a rise that
    # goes on past the sweep's end: the best of the smoothed scores is the
    # sweep's second or second-to-last plane.
    rising = [1.0 + 0.001 * k for k in range(9)]
    last = find_peak(_Z9, rising)
    assert (last.outcome, last.best_index, last.z_um) == ("at-edge", 7, None)
    first = find_peak(_Z9, rising[::-1])
    assert (first.outcome, first.best_index, first.z_um) == ("at-edge", 1, None)


def test_a_best_one_plane_inside_the_edge_is_a_peak() -> None:
    # The zone's other side (FM-73): a smoothed best on the third plane, or the
    # third-to-last, is bracketed by a smoothed score on each side, and a field
    # whose focus lies that close to the sweep's end must still move Z to it.
    tent = [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    near_first = find_peak(_Z9, tent)
    assert (near_first.outcome, near_first.best_index) == ("peak", 2)
    assert near_first.z_um == pytest.approx(2.0)
    near_last = find_peak(_Z9, tent[::-1])
    assert (near_last.outcome, near_last.best_index) == ("peak", 6)
    assert near_last.z_um == pytest.approx(6.0)


def test_a_falling_shoulder_at_the_start_is_not_a_second_peak() -> None:
    # A second maximum is an interior one of the smoothed scores: an object
    # focused below the sweep leaves a shoulder that falls from its first
    # plane, which is no maximum and must not turn a clear peak into two.
    shoulder = [
        1.0
        + 0.04 * math.exp(-((k - 12) ** 2) / 4.5)
        + 0.03 * math.exp(-k / 2)
        + 0.0005 * (-1) ** k
        for k in range(21)
    ]
    peak = find_peak(_Z21, shoulder)
    assert (peak.outcome, peak.best_index) == ("peak", 12)
    assert peak.z_um == pytest.approx(12.0, abs=0.05)


def _two(second: float) -> list[float]:
    """A best at plane 6 and a second bump at plane 14 of ``second`` times its height, on 21 planes."""
    return [
        1.0
        + 0.04
        * (math.exp(-((k - 6) ** 2) / 4.5) + second * math.exp(-((k - 14) ** 2) / 4.5))
        + 0.0005 * (-1) ** k
        for k in range(21)
    ]


def test_two_maxima_are_two_peaks() -> None:
    # The two sides of SECOND_PEAK_FRACTION (FM-73); 0.4 is the case that fails
    # when the fraction condition is dropped: its second bump stands far above
    # the dip between the two.
    assert 0.4 < SECOND_PEAK_FRACTION < 0.6
    two = find_peak(_Z21, _two(0.6))
    assert (two.outcome, two.best_index, two.z_um) == ("two-peaks", 6, None)
    one = find_peak(_Z21, _two(0.4))
    assert (one.outcome, one.best_index) == ("peak", 6)
    assert one.z_um == pytest.approx(6.0, abs=0.01)


def test_unequal_spacing_is_refined_through_the_readbacks() -> None:
    tent = [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    peak = find_peak([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0], tent)
    assert peak.outcome == "peak"
    assert peak.z_um == pytest.approx(4.5)
    # A tent's median second difference is 0, so its noise is read from their
    # mean instead (D13): the prominence is finite, not unbounded.
    assert peak.prominence == pytest.approx(27.65, abs=0.01)
    # The same scores on an even spacing put the vertex on the best plane.
    assert find_peak(_Z9, tent).z_um == pytest.approx(4.0)
    # A parabola whose top lies just short of 4.5, halfway between two planes:
    # the smoothed best is plane 4, and the vertex leans toward plane 5.
    parabola = [100.0 - (k - 4.4999) ** 2 for k in range(9)]
    lean = find_peak(_Z9, parabola)
    assert lean.outcome == "peak"
    assert lean.best_index == 4
    assert lean.z_um == pytest.approx(4.4999)


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
    # Nine planes each, so that the message is the reason and not the count.
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        find_peak(_Z9, [*scores[:4], math.nan, *scores[5:]])
    with pytest.raises(ValueError, match="finite"):
        find_peak([*_Z9[:4], math.nan, *_Z9[5:]], scores)
    with pytest.raises(ValueError, match="increasing"):
        find_peak([*_Z9[:3], 4.0, 3.0, *_Z9[5:]], scores)
    with pytest.raises(ValueError, match="increasing"):
        find_peak([*_Z9[:4], 3.0, *_Z9[5:]], scores)
    with pytest.raises(ValueError, match="same length"):
        find_peak(_Z9, scores[:8])


def test_a_sweep_of_fewer_than_nine_planes_is_refused() -> None:
    # A sweep measures its noise between neighbouring planes: below nine, a
    # peak's own curvature passes for it (D9). Both sides of the bound (FM-73).
    assert MIN_PLANES == 9
    z8 = _Z9[:8]
    scores8 = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
    with pytest.raises(ValueError, match="at least 9 planes"):
        find_peak(z8, scores8)
    assert find_peak(_Z9, [*scores8, 1.0]).outcome == "peak"
    # The same bound in the parameters' validators: 7 planes, then 9, then 5.
    problem = sweep_size_problem(3.0, 1.0, "range_um", "step_um")
    assert problem == (
        "range_um=3 with step_um=1 makes 7 planes, fewer than 9: the sweep "
        "measures its noise between neighbouring planes; take a smaller step or "
        "a larger range"
    )
    assert sweep_size_problem(4.0, 1.0, "range_um", "step_um") is None
    assert sweep_size_problem(2.0, 1.0, "range_um", "step_um") is not None
    # The step above the range is still named as that, not as a count.
    above = sweep_size_problem(1.0, 2.0, "range_um", "step_um")
    assert above is not None
    assert "is above range_um" in above
    # The upper bound is unchanged.
    assert sweep_size_problem(float(MAX_PLANES), 2.0, "r", "s") is None
    assert "more than" in (sweep_size_problem(51.0, 1.0, "r", "s") or "")
