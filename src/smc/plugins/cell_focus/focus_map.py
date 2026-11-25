"""The 10x lock map of the plate's bottom, and the rules of a well's anchor (m3 §21.12.3).

Until PFS locks at the Ti2-E's 40x (#539), a 40x scan runs with the autofocus
off, and a sweep centred on the scan's start Z cannot follow a bottom that
varies by 9 to 36 µm across a well (#623). A scan taken with PFS locked at the
10x has measured that bottom at every one of its frames: this module reads a
40x field's Z from the 4 locked frames of its own well nearest it, weighted by
1/d (measured on session 6: 1.38 µm median error, against 2.26 for the nearest
alone). A wall lies between two wells, so another well's frames are never read.

The 10x -> 40x difference is not a constant (59.8, 81.3 and 129.5 µm on two
days, m3 §21.12.1): it is measured in each well by a few sweeps near its
centre, and ``anchor_of`` is their median. Pure: no capability, no slot model.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ANCHOR_PEAKS",
    "ANCHOR_TRIES",
    "MAP_NEIGHBOURS",
    "MAP_SNAP_UM",
    "FocusMap",
    "MapPoint",
    "anchor_of",
    "nearest_first",
]

#: Measured (m3 §21.12.1): leave one out on session 6's 10x scan, the 4 nearest by
#: 1/d err by 1.38 um (median) against 2.26 for the nearest and 1.65 for 6.
MAP_NEIGHBOURS = 4
#: A map frame this close gives its own Z. Chosen: under a stage's landing.
MAP_SNAP_UM = 1.0
#: Chosen: 17 of B4's 24 cell sweeps found a peak (m3 §21.11.3); C3 holds a
#: tenth of C4's cells (#636), so a sparse well needs more tries.
ANCHOR_TRIES = 8
ANCHOR_PEAKS = 3


@dataclass(frozen=True)
class MapPoint:
    """One locked frame of the map's scan: its well, its XY readback and its Z, in µm."""

    well: str
    x_um: float
    y_um: float
    z_um: float


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


class FocusMap:
    """The locked frames of a scan, read well by well (m3 §21.12.2 D17)."""

    def __init__(self, points: Sequence[MapPoint]) -> None:
        """Hold ``points`` in their order: a tie is broken by it.

        Raises:
            ValueError: A point's coordinate or Z is not finite (FM-32); the
                message names its well. An empty map is allowed: ``z_at`` and
                ``nearest`` refuse it.
        """
        by_well: dict[str, list[MapPoint]] = {}
        for point in points:
            if not _finite(point.x_um, point.y_um, point.z_um):
                raise ValueError(
                    f"a focus map point of well {point.well} is not finite: "
                    f"({point.x_um!r}, {point.y_um!r}, {point.z_um!r})"
                )
            by_well.setdefault(point.well, []).append(point)
        self._points = tuple(points)
        self._by_well = {well: tuple(found) for well, found in by_well.items()}

    def count(self, well: str) -> int:
        """How many locked frames the map holds in ``well``."""
        return len(self._by_well.get(well, ()))

    def z_at(self, well: str, x_um: float, y_um: float) -> float:
        """The map's Z at ``(x_um, y_um)`` from the 4 nearest frames of ``well``, by 1/d.

        A frame within ``MAP_SNAP_UM`` gives its own Z: no division by about 0.
        The sort is stable, so a tie keeps the input order.

        Raises:
            ValueError: ``x_um`` or ``y_um`` is not finite, or ``well`` holds
                fewer than ``MAP_NEIGHBOURS`` frames (named in the message).
        """
        if not _finite(x_um, y_um):
            raise ValueError(
                f"the focus map is read at a point that is not finite: ({x_um!r}, {y_um!r})"
            )
        points = self._by_well.get(well, ())
        if len(points) < MAP_NEIGHBOURS:
            raise ValueError(
                f"the focus map holds {len(points)} locked frames in {well}, fewer "
                f"than {MAP_NEIGHBOURS}"
            )
        ranked = sorted(
            ((math.hypot(p.x_um - x_um, p.y_um - y_um), p) for p in points),
            key=lambda pair: pair[0],
        )
        first_d, first = ranked[0]
        if first_d <= MAP_SNAP_UM:
            return first.z_um
        near = ranked[:MAP_NEIGHBOURS]
        weights = [1.0 / d for d, _ in near]
        return sum(w * p.z_um for w, (_, p) in zip(weights, near, strict=True)) / sum(
            weights
        )

    def nearest(self, x_um: float, y_um: float) -> tuple[MapPoint, float]:
        """The frame of any well nearest ``(x_um, y_um)``, and its distance; the first on a tie.

        Raises:
            ValueError: The map is empty, or a coordinate is not finite.
        """
        if not _finite(x_um, y_um):
            raise ValueError(
                f"the focus map is read at a point that is not finite: ({x_um!r}, {y_um!r})"
            )
        if not self._points:
            raise ValueError("the focus map holds no locked frame")
        best = min(self._points, key=lambda p: math.hypot(p.x_um - x_um, p.y_um - y_um))
        return best, math.hypot(best.x_um - x_um, best.y_um - y_um)


def anchor_of(offsets_um: Sequence[float]) -> float | None:
    """A well's anchor: the median of its peaks' offsets from the map; ``None`` without one.

    The median of three is less pulled by one field's cell or debris than one
    peak (m3 §21.12.2 D18).

    Raises:
        ValueError: An offset is not finite.
    """
    for offset in offsets_um:
        if not math.isfinite(offset):
            raise ValueError(f"an anchor offset is not finite: {offset!r}")
    if not offsets_um:
        return None
    return float(statistics.median(offsets_um))


def nearest_first(
    points_xy: Sequence[tuple[float, float]], centre_xy: tuple[float, float], n: int
) -> tuple[int, ...]:
    """The indices of the ``n`` points nearest ``centre_xy``, nearest first; the lower index on a tie.

    All of them when ``n`` is larger. The anchor sweeps near the well's
    centre, not in the scan's order: the scan starts at the wall, where the
    10x sees dried droplets and the meniscus and the map is least sure.

    Raises:
        ValueError: ``n`` is below 1.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    cx, cy = centre_xy
    order = sorted(
        range(len(points_xy)),
        key=lambda i: (math.hypot(points_xy[i][0] - cx, points_xy[i][1] - cy), i),
    )
    return tuple(order[:n])
