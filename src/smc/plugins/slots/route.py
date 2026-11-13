"""The route slot: the fields to visit, in order, to image the cells that matter (m3 §3.5)."""

from __future__ import annotations

from typing import Any

import useq
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from smc.plugins.api import PluginResult
from smc.plugins.geometry import Matrix2
from smc.plugins.slots.base import SlotValue

__all__ = ["OffsetFromFocus", "RoutePosition", "RouteResult"]


class OffsetFromFocus(SlotValue):
    """How a stop's ``pfs_offset`` was derived for the cells' plane, not read (m3 §26.3).

    ``lock_offset`` is the offset the scan read locked at the frame's field,
    before the focus step switched the autofocus off (the frame's
    ``pfs_offset``); ``dz_um`` is the frame's ``cell_focus.dz_um``, its plane
    minus the plane the autofocus held; ``offset_per_um`` is the profile's
    factor for the scan's objective, sign included. A factor of zero would make
    every distance worth nothing, so it is refused here and not only by the
    rule that builds the record.
    """

    lock_offset: float
    dz_um: float
    offset_per_um: float

    @model_validator(mode="after")
    def _the_factor_is_not_zero(self) -> OffsetFromFocus:
        if self.offset_per_um == 0.0:
            raise ValueError(
                "offset_per_um must not be 0: a factor of zero turns no distance "
                "into an offset"
            )
        return self


class RoutePosition(SlotValue):
    """One stop of the route: a field, its stage position and the cells it holds.

    ``z_um`` and ``pfs_offset`` are what the scan recorded, both from one frame,
    at the nearest frame of the well that was not ``unlocked`` (the autofocus had
    not lost the sample on it, FM-118): ``None`` when the well has no such frame,
    and ``pfs_offset`` ``None`` also when that frame did not use the autofocus.
    ``counts`` are the wanted labels' cells in the field.

    ``pfs_offset`` is the scan's **reading** except for a stop that has
    ``offset_from_focus``: that stop's frame was taken at the cells' plane with
    the autofocus off, and its ``pfs_offset`` is the offset of that plane,
    ``round(lock_offset + dz_um * offset_per_um)``, derived and never read. The
    validator checks that there is an offset to name and does not recompute the
    sum: the rule that built it (``route/focus.py``) owns the arithmetic.
    """

    order: int = Field(ge=0)
    well: str
    name: str
    x_um: float
    y_um: float
    z_um: float | None
    pfs_offset: float | None
    counts: dict[str, int]
    offset_from_focus: OffsetFromFocus | None = None

    @model_validator(mode="after")
    def _a_derived_offset_is_an_offset(self) -> RoutePosition:
        if self.offset_from_focus is not None and self.pfs_offset is None:
            raise ValueError(
                "offset_from_focus says how pfs_offset was derived, so a stop "
                "that has it must have a pfs_offset"
            )
        return self

    @model_serializer(mode="wrap")
    def _leave_out_what_is_missing(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # A stop that was not derived is written exactly as before the field
        # existed, so a route from a scan without the focus step still reads in
        # an older smc, whose model refuses a key it does not know (FM-107),
        # and SCHEMA_VERSION stays 1. A route with a derived stop does not read
        # there, as intended: that smc would drop what says the offset is not a
        # reading.
        data: dict[str, Any] = handler(self)
        if self.offset_from_focus is None:
            data.pop("offset_from_focus", None)
        return data


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
