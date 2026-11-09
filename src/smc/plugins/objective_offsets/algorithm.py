"""The arithmetic of ``objective-offsets``: shifts, their spread, and the rebase to a reference (m3 §20.5).

Pure: no device, no file, no numpy. Under each target objective the plugin
measures again the centres of wells the registration measured, and a well's
**difference** is where the target puts its centre minus where the
registration's objective put it. That is the target's offset from the
registration's objective if the shift is one translation over the plate
(m2 §13.8), so the wells' differences are the evidence of it: their mean is the
shift, and how far apart they lie is the spread the plugin holds to a bound.

The profile's offsets are measured from a *reference* objective (m2 §13.2,
``p_O = p_R + offset(O)``), which need not be the one the plate was registered
with: ``offsets_from_reference`` moves the shifts to it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

__all__ = [
    "MAX_TARGETS",
    "SPREAD_PIXELS",
    "default_max_spread_um",
    "mean_shift_um",
    "offsets_from_reference",
    "spread_um",
]

#: The wells may disagree by this many pixels of the coarser of the two
#: objectives (m2 §13.6: 5.5 µm at the Ti2-E's 4x). Chosen, not measured.
SPREAD_PIXELS = 2.0
#: At most this many targets in one run, so that the walks' bound holds in all
#: (FM-140). Chosen: a turret holds six positions.
MAX_TARGETS = 5


def mean_shift_um(deltas: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The mean of the wells' differences, per axis: the target's shift.

    Raises:
        ValueError: There is no difference to average.
    """
    if not deltas:
        raise ValueError("no difference to average: no well was measured")
    n = len(deltas)
    return (
        math.fsum(dx for dx, _ in deltas) / n,
        math.fsum(dy for _, dy in deltas) / n,
    )


def spread_um(deltas: Sequence[tuple[float, float]]) -> float:
    """The largest distance between two wells' differences; ``0.0`` for one well.

    It is a distance between two wells and not a deviation from the mean, so
    that it does not shrink as wells are added, and it carries the stage's
    repeatability as well as a shift that is not one translation.

    Raises:
        ValueError: There is no difference.
    """
    if not deltas:
        raise ValueError("no difference to compare: no well was measured")
    return max(
        (math.dist(a, b) for a, b in itertools.combinations(deltas, 2)),
        default=0.0,
    )


def default_max_spread_um(origin_pixel_um: float, target_pixel_um: float) -> float:
    """The default bound on the spread: ``SPREAD_PIXELS`` pixels of the coarser objective.

    The coarser objective's pixel is the finest position either measurement
    can tell apart, so the wells cannot be asked to agree more closely than
    that.

    Raises:
        ValueError: A pixel size that is not finite and positive: a size of 0
            means "not measured", and a bound made of it would be no bound.
    """
    for size in (origin_pixel_um, target_pixel_um):
        if not (math.isfinite(size) and size > 0.0):
            raise ValueError(f"a pixel size must be finite and positive, got {size!r}")
    return SPREAD_PIXELS * max(origin_pixel_um, target_pixel_um)


def offsets_from_reference(
    origin: str,
    shifts: Sequence[tuple[str, tuple[float, float]]],
    reference: str | None,
) -> tuple[str, tuple[tuple[str, tuple[float, float]], ...]]:
    """The offsets of every objective from the reference: ``(reference, ((label, (dx, dy)), ...))``.

    ``origin`` is the objective the plate was registered with, whose own
    shift is ``(0, 0)``; ``shifts`` are the targets' shifts from it, in the
    order they were measured. ``reference`` is the objective the profile's
    offsets are from (``None``: the origin). Each offset is the objective's
    shift minus the reference's, per axis, so that ``p_O = p_R + offset(O)``
    (m2 §13.2). The result lists the origin first and then the targets in order,
    leaving out the reference, whose own offset is zero by definition. ``+ 0.0``
    turns a ``-0.0`` into ``0.0``: it would print as ``-0.0`` in a profile.

    Raises:
        ValueError: An objective is given twice (the origin among the
            targets included), or the reference is neither the origin nor a
            target: the plugin refuses that before anything turns, because
            the run cannot know where the reference is.
    """
    shift_of: dict[str, tuple[float, float]] = {origin: (0.0, 0.0)}
    for label, shift in shifts:
        if label in shift_of:
            raise ValueError(f"objective {label!r} is given twice")
        shift_of[label] = shift
    chosen = origin if reference is None else reference
    if chosen not in shift_of:
        raise ValueError(
            f"the reference {chosen!r} is neither the objective the plate was "
            f"registered with ({origin!r}) nor one of the targets"
        )
    rx, ry = shift_of[chosen]
    return chosen, tuple(
        (label, (sx - rx + 0.0, sy - ry + 0.0))
        for label, (sx, sy) in shift_of.items()
        if label != chosen
    )
