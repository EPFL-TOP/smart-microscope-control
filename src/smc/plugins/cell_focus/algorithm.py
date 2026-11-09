"""The focus on the cells' pure part: a focus score, the planes of a sweep, and its peak (m3 §21.3).

PFS holds the plate's bottom, and the cells sit a few micrometres above it,
by an amount that differs from well to well. The sweep takes a frame at each
of a few planes around a centre; this module scores each frame and finds the
plane of the best score. It is numpy only, so the tool, the scan and the
offline script (``scripts/dev/focus_scores.py``) score alike, and every
decision here is tested on synthetic stacks.

A sweep with no single clear maximum keeps its centre (D4): an empty field is
flat, a best at an end of the sweep says the focus is outside it, and two
maxima are what brightfield of unstained cells can show on either side of
focus. Unknown beats guessed: which of these the real 40x stacks show is the
owner's to measure (m3 §21.10).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_SCORE",
    "LANDING_FRACTION",
    "MAX_PLANES",
    "MAX_RANGE_UM",
    "MAX_STEP_UM",
    "MIN_RISE",
    "MIN_STEP_UM",
    "SECOND_PEAK_FRACTION",
    "Peak",
    "PeakOutcome",
    "ScoreName",
    "find_peak",
    "first_outside",
    "plan_planes",
    "score_frame",
    "sweep_size_problem",
]

ScoreName = Literal[
    "brenner", "normalized-variance", "gradient-energy", "laplacian-variance"
]
PeakOutcome = Literal["peak", "flat", "at-edge", "two-peaks"]

#: Chosen from the literature, not measured on cells: session 3's stacks
#: decide (m3 §21.10). On the demo-sample all four peak at Z = 0.
DEFAULT_SCORE: ScoreName = "brenner"
#: The widest half-range of a sweep. Chosen: the 40x's working distance is
#: 170-250 µm (catalogue), and a sweep never needs more than a few µm.
MAX_RANGE_UM = 50.0
#: Chosen: below the Ti2-E's Z steps that were read (0.01 µm) times ten.
MIN_STEP_UM = 0.1
#: Chosen: a step wider than this skips past a 40x's depth of field.
MAX_STEP_UM = 10.0
#: Bounds the frames of one sweep (FM-74). Chosen.
MAX_PLANES = 101
#: A sweep whose (max - min) / max is below this is flat. Chosen: the demo
#: gives 0.17 at its narrowest sweep; no real flat sweep was read.
MIN_RISE = 0.05
#: A second interior maximum whose rise above the sweep's minimum reaches
#: this fraction of the best's makes the sweep "two-peaks". Chosen.
SECOND_PEAK_FRACTION = 0.5
#: A Z readback farther from its command than this fraction of the step
#: fails the sweep (FM-167). Chosen.
LANDING_FRACTION = 0.25

#: Absorbs the decimal error of ``range / step`` (0.3 / 0.1 is 2.9999...), so
#: that a range of a whole number of steps keeps its ends.
_WHOLE_STEPS_SLACK = 1e-9


@dataclass(frozen=True)
class Peak:
    """What a sweep's scores say: its outcome, the best plane's index and the refined Z.

    ``best_index`` is the first index of the maximum in every outcome, so a
    caller can always name the best frame; ``z_um`` is set only for a
    ``peak``, the one outcome whose plane is taken.
    """

    outcome: PeakOutcome
    best_index: int
    #: The refined plane; ``None`` unless ``outcome == "peak"``.
    z_um: float | None


def score_frame(frame: npt.NDArray[Any], name: ScoreName) -> float:
    """The focus score of one frame, higher when sharper; only compared within one sweep.

    Each score but the normalised variance is divided by the mean squared, so
    an exposure or a lamp that drifts during a sweep does not move its peak.

    Raises:
        ValueError: The frame is not 2-D, smaller than 3 x 3, or the name is unknown.
    """
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"a focus score needs a 2-D frame, got shape {f.shape}")
    if f.shape[0] < 3 or f.shape[1] < 3:
        raise ValueError(
            f"a focus score needs a frame of at least 3 x 3 pixels, got {f.shape}"
        )
    m = float(f.mean())
    if not (math.isfinite(m) and m > 0.0):
        # A dark frame (or one with a NaN) has no contrast to compare: the
        # sweep reads it as the bottom of the curve, never as its peak.
        return 0.0
    if name == "brenner":
        value = float(np.mean((f[:, 2:] - f[:, :-2]) ** 2)) / m**2
    elif name == "gradient-energy":
        gx = f[:, 1:] - f[:, :-1]
        gy = f[1:, :] - f[:-1, :]
        value = float(np.mean(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)) / m**2
    elif name == "laplacian-variance":
        lap = (
            4.0 * f[1:-1, 1:-1]
            - f[:-2, 1:-1]
            - f[2:, 1:-1]
            - f[1:-1, :-2]
            - f[1:-1, 2:]
        )
        value = float(np.var(lap)) / m**2
    elif name == "normalized-variance":
        value = float(np.var(f)) / m
    else:
        raise ValueError(f"unknown focus score {name!r}")
    return value if math.isfinite(value) else 0.0


def plan_planes(centre_um: float, range_um: float, step_um: float) -> tuple[float, ...]:
    """The planes of a sweep: symmetric around the centre, ``step_um`` apart, increasing (D2).

    The half-range is cut down to a whole number of steps, never stretched
    past ``range_um``: the jog and soft-limit checks are made on ``range_um``.

    Raises:
        ValueError: A number is not finite, the step is not positive, the
            range holds less than one step on each side, or the sweep would
            have more than ``MAX_PLANES`` planes.
    """
    if not math.isfinite(centre_um):
        raise ValueError(f"centre_um must be finite, got {centre_um!r}")
    if not (math.isfinite(range_um) and range_um > 0.0):
        raise ValueError(f"range_um must be finite and positive, got {range_um!r}")
    if not (math.isfinite(step_um) and step_um > 0.0):
        raise ValueError(f"step_um must be finite and positive, got {step_um!r}")
    k = math.floor(range_um / step_um + _WHOLE_STEPS_SLACK)
    if k < 1:
        raise ValueError(
            f"range_um={range_um:g} holds no step of step_um={step_um:g} on each "
            "side of the centre: a sweep needs at least 3 planes"
        )
    if 2 * k + 1 > MAX_PLANES:
        raise ValueError(
            f"range_um={range_um:g} with step_um={step_um:g} makes {2 * k + 1} "
            f"planes, more than {MAX_PLANES}"
        )
    return tuple(centre_um + i * step_um for i in range(-k, k + 1))


def sweep_size_problem(
    range_um: float, step_um: float, range_name: str, step_name: str
) -> str | None:
    """Why a range and a step make no sweep, naming both parameters; ``None`` when they do.

    The parameters' model validators call it (the tool's and the scan's), so
    a bad pair is refused when the parameters are read, before the stand
    opens, rather than at the first field.
    """
    if step_um > range_um:
        return (
            f"{step_name}={step_um:g} is above {range_name}={range_um:g}: a sweep "
            "needs at least one step on each side of its centre"
        )
    planes = 2 * math.floor(range_um / step_um + _WHOLE_STEPS_SLACK) + 1
    if planes > MAX_PLANES:
        return (
            f"{range_name}={range_um:g} with {step_name}={step_um:g} makes {planes} "
            f"planes, more than {MAX_PLANES}: take a larger step or a smaller range"
        )
    return None


def first_outside(
    planes_um: Sequence[float], low_um: float, high_um: float
) -> float | None:
    """The first plane outside ``[low_um, high_um]`` (a NaN is outside), or ``None``."""
    for plane in planes_um:
        if not (low_um <= plane <= high_um):
            return plane
    return None


def find_peak(z_um: Sequence[float], scores: Sequence[float]) -> Peak:
    """The sweep's outcome, and the refined plane when it has one clear interior maximum (D4).

    The refinement is the vertex of the parabola through the best plane and
    its two neighbours, at their own Z (the readbacks, so unequal spacing is
    allowed), clamped between the neighbours: a sweep never claims a plane it
    did not bracket.

    Raises:
        ValueError: The lists differ in length, hold fewer than 3 values, a
            value is not finite, or ``z_um`` is not strictly increasing.
    """
    z = [float(v) for v in z_um]
    s = [float(v) for v in scores]
    if len(z) != len(s):
        raise ValueError(
            f"z_um and scores must have the same length, got {len(z)} and {len(s)}"
        )
    if len(z) < 3:
        raise ValueError(f"a sweep needs at least 3 planes, got {len(z)}")
    if not all(math.isfinite(v) for v in (*z, *s)):
        raise ValueError("every Z and every score of a sweep must be finite")
    if any(b <= a for a, b in itertools.pairwise(z)):
        raise ValueError(f"z_um must be strictly increasing, got {z}")

    top = max(s)
    bottom = min(s)
    i = s.index(top)
    last = len(s) - 1
    if top <= 0.0 or (top - bottom) / top < MIN_RISE:
        return Peak("flat", i, None)
    if i in (0, last):
        return Peak("at-edge", i, None)
    for j in range(1, last):
        if (
            j != i
            and s[j] > s[j - 1]
            and s[j] > s[j + 1]
            and s[j] - bottom >= SECOND_PEAK_FRACTION * (top - bottom)
        ):
            return Peak("two-peaks", i, None)
    return Peak("peak", i, _vertex(z[i - 1 : i + 2], s[i - 1 : i + 2]))


def _vertex(z: Sequence[float], s: Sequence[float]) -> float:
    """The vertex of the parabola through three points, clamped to their Z; the middle Z when it opens up or is a line."""
    z0, z1, z2 = z
    s0, s1, s2 = s
    denom = (z0 - z1) * (z0 - z2) * (z1 - z2)
    a = (z2 * (s1 - s0) + z1 * (s0 - s2) + z0 * (s2 - s1)) / denom
    b = (z2 * z2 * (s0 - s1) + z1 * z1 * (s2 - s0) + z0 * z0 * (s1 - s2)) / denom
    if not a < 0.0:
        return z1
    vertex = -b / (2.0 * a)
    if not math.isfinite(vertex):
        return z1
    return min(max(vertex, z0), z2)
