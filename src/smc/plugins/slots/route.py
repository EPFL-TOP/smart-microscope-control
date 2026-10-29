"""The route slot: the fields to visit, in order, to image the cells that matter (m3 §3.5)."""

from __future__ import annotations

import useq
from pydantic import Field

from smc.plugins.api import PluginResult
from smc.plugins.geometry import Matrix2
from smc.plugins.slots.base import SlotValue

__all__ = ["RoutePosition", "RouteResult"]


class RoutePosition(SlotValue):
    """One stop of the route: a field, its stage position and the cells it holds.

    ``z_um`` and ``pfs_offset`` are what the scan recorded, both from one frame,
    at the nearest frame of the well that was not ``unlocked`` (the autofocus had
    not lost the sample on it, FM-118): ``None`` when the well has no such frame,
    and ``pfs_offset`` ``None`` also when that frame did not use the autofocus.
    ``counts`` are the wanted labels' cells in the field.
    """

    order: int = Field(ge=0)
    well: str
    name: str
    x_um: float
    y_um: float
    z_um: float | None
    pfs_offset: float | None
    counts: dict[str, int]


class RouteResult(PluginResult):
    """The ordered stops, with what the route does not cover and what it costs.

    ``uncovered`` counts, per well, the cells of the wanted labels that no
    chosen field holds. ``travel_s`` is ``None`` when no timing model was
    available: an unknown is not a zero.
    """

    objective: str | None
    matrix_um_per_px: Matrix2
    image_shape_px: tuple[int, int]
    fov_um: tuple[float, float]
    labels: tuple[str, ...]
    positions: tuple[RoutePosition, ...]
    uncovered: dict[str, int]
    distance_um: float = Field(ge=0.0)
    travel_s: float | None = Field(ge=0.0)

    def to_useq(self) -> tuple[useq.AbsolutePosition, ...]:
        """The stops as ``useq`` positions, in route order.

        ADR-0004: where to image is a ``useq`` object, derived here from the
        one stored list rather than stored twice. An unknown ``z_um`` stays
        ``None`` (measured), never 0.
        """
        return tuple(
            useq.AbsolutePosition(x=stop.x_um, y=stop.y_um, z=stop.z_um, name=stop.name)
            for stop in self.positions
        )
