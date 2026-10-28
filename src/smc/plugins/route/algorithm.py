"""Which fields to image and in what order: the densest fields first, then the shortest path (m3 §11).

Pure arithmetic on stage µm: no stage, no camera, no turret. The plugin hands in
the camera's matrix and the frame's shape, and the cells; what a field is, is
decided here once.

Two things are easy to get wrong and invisible on a square frame (FM-73): a
field is the camera's **footprint**, a rectangle turned by the camera and
possibly mirrored, not a square of the field of view around a centre; and its
width and height are exchanged in the pixel offsets of the overlap rule, in the
membership test and in the window that counts cells. Everything that tests
whether a cell or a corner is in a field goes through ``pixel_of_stage`` and
``footprint``, so the sign convention stays in ``geometry.py``.

A numeric choice made on measurements, not on taste:

* **No matrix product on coordinate arrays.** ``(N, 2) @ (2, 2)`` on finite
  data raises ``RuntimeWarning: divide by zero encountered in matmul`` from
  N = 2000 up on numpy 2.2.6 (macOS arm64), though not on 1.26 or 2.5. The four
  entries of the inverse are written out instead.
* **Counting is per cell, over the lattice points near it**, not cells times
  candidates: 300 000 cells in a 96-well well took 12 s the second way and
  0.46 s the first, and the second grows with the well (FM-74).
* **The bounds are on floats first.** ``3200.0 // 5e-324`` is ``inf`` and
  ``int(inf)`` is an ``OverflowError``, so a size is refused before it becomes
  an integer.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "ChosenField",
    "order_route",
    "path_length_um",
]

Point = tuple[float, float]

#: A reversal that shortens the path by less than this is not made: it ends 2-opt,
#: and a float round-off is not an improvement.
_IMPROVEMENT_UM = 1e-9


@dataclass(frozen=True, slots=True)
class ChosenField:
    """A field the greedy chose: its centre in stage µm and the cells it holds.

    ``counts`` has one entry per wanted label, zeros included, over the cells
    that no earlier field held. It is built as a ``dict`` and not mutated after.
    """

    x_um: float
    y_um: float
    counts: Mapping[str, int]


def order_route(points: Sequence[Point], *, start: int) -> list[int]:
    """The order in which to visit ``points``: nearest neighbour from ``start``, then 2-opt.

    An open path whose first point stays ``start``. The cost is the Euclidean
    distance: the timing model is an overhead per move plus a rate per µm, so
    the shortest path is the fastest under it. Whether the Ti2-E's stage is
    nearer to Chebyshev (both axes at once) is not measured (#198), so no
    optimality and no time is claimed here. 2-opt is a heuristic: it never
    leaves the path longer than nearest neighbour's, and it is not the optimum.

    Ties go to the lowest index, both in the nearest neighbour and in the
    reversal that 2-opt applies, so the same points give the same route.

    Returns:
        The indices into ``points``, each once; ``[]`` for no point.

    Raises:
        ValueError: ``start`` is not an index of ``points``, or a coordinate is
            not finite (``nan`` would make every distance compare false and
            end a path anywhere).
    """
    if len(points) == 0:
        return []
    coordinates = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(coordinates).all():
        raise ValueError("every point must have finite coordinates")
    n = len(coordinates)
    if not (0 <= start < n):
        raise ValueError(f"start must be an index of the {n} points, got {start!r}")

    order = _nearest_neighbour(coordinates, start)
    _two_opt(coordinates, order)
    return order


def path_length_um(points: Sequence[Point], order: Sequence[int]) -> float:
    """The length of the open path that visits ``points`` in ``order``, in µm.

    0 for fewer than two stops. A leg is a straight line, as ``estimate_s`` of
    the scan plan prices it.
    """
    return sum(math.dist(points[a], points[b]) for a, b in itertools.pairwise(order))


def _nearest_neighbour(coordinates: npt.NDArray[np.float64], start: int) -> list[int]:
    """Greedy path from ``start``; the points left stay in ascending order, so ``argmin`` takes the lowest index on a tie."""
    left = np.delete(np.arange(len(coordinates)), start)
    order = [start]
    here = coordinates[start]
    while len(left):
        distances = np.hypot(
            coordinates[left, 0] - here[0], coordinates[left, 1] - here[1]
        )
        position = int(distances.argmin())
        chosen = int(left[position])
        order.append(chosen)
        left = np.delete(left, position)
        here = coordinates[chosen]
    return order


def _two_opt(coordinates: npt.NDArray[np.float64], order: list[int]) -> None:
    """Uncross the open path in place, the first point fixed; repeated passes until one changes nothing.

    Reversing ``q[i..j]`` replaces the edges ``(q[i-1], q[i])`` and
    ``(q[j], q[j+1])`` by ``(q[i-1], q[j])`` and ``(q[i], q[j+1])``; at the end
    of an open path there is no second edge. For each ``i`` every ``j`` is
    priced at once and the most negative change is applied. Each applied
    reversal shortens the path by more than ``_IMPROVEMENT_UM``, so it ends.
    """
    n = len(order)
    if n < 3:
        return
    q = coordinates[order].copy()
    changed = True
    while changed:
        changed = False
        for i in range(1, n - 1):
            js = np.arange(i + 1, n)
            before = q[i - 1]
            delta = np.hypot(q[js, 0] - before[0], q[js, 1] - before[1]) - math.hypot(
                q[i, 0] - before[0], q[i, 1] - before[1]
            )
            # The second edge exists for every j but the last.
            inner = js[:-1]
            delta[:-1] += np.hypot(
                q[i, 0] - q[inner + 1, 0], q[i, 1] - q[inner + 1, 1]
            ) - np.hypot(q[inner, 0] - q[inner + 1, 0], q[inner, 1] - q[inner + 1, 1])
            best = int(delta.argmin())
            if delta[best] < -_IMPROVEMENT_UM:
                j = int(js[best])
                q[i : j + 1] = q[i : j + 1][::-1].copy()
                order[i : j + 1] = order[i : j + 1][::-1]
                changed = True
