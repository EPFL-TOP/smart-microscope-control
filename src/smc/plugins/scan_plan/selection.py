"""The fields a selection chose, as a scan plan's fields (m3 §28.4).

Pure: the stops of a ``route`` read from another run document, the wells the
plan names, the shift between the selection's objective and the plan's, each
well's shape and the stage's soft limits come in; the plan's fields come out,
or a ``ValueError`` that names the first stop that cannot be visited. No stage,
no camera, so it is unit-tested on stops placed by hand.

What it decides, because the 40x stage goes where it says:

* **A well is visited in one piece.** The route's orderer may interleave wells
  (algorithms §5.3), and the scan switches the autofocus off, checks the plate
  and anchors the focus map well by well: the wells come in the plan's order and
  a well's stops in the route's ``order`` within it.
* **A well with no stop is not planned** (m3 §28.2 D2): the scan and its focus
  map's anchor pass index every planned well, so a planned well with no field
  would raise there; it is listed instead. A stop of a well not named is counted.
* **The centre is checked after the shift**, in the frame the stage will move
  in: a stop 20 µm inside a wall, moved 37.5 µm towards it by the offset, is a
  frame of plastic. The test is ``WellGeometry.contains`` with no margin: a
  footprint may reach past the wall, as ``route``'s own ``wall_margin_um=0``
  allows, but its centre may not.
* **The limits are the layer's test**, ``lo <= v <= hi`` on both axes, so a
  field this keeps is one the stage would not refuse.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from smc.plugins.slots import PlannedField, RoutePosition, WellGeometry

__all__ = ["MAX_FIELDS_PER_WELL", "ChosenFields", "chosen_fields"]

#: The most fields a plan from a selection holds in one well: the grid's own most
#: (100 x 100, m3 §7.3). ``route`` chooses at most 2000 a well; a document edited
#: by hand, or another tool's route, could hold any number.
MAX_FIELDS_PER_WELL = 10_000


@dataclass(frozen=True, slots=True)
class ChosenFields:
    """What a selection gives a plan: the wells and their fields, and what was left out."""

    #: The planned wells, in the given order: those that hold at least one stop.
    wells: tuple[str, ...]
    #: In visiting order: well by well, each well's in the route's order.
    fields: tuple[PlannedField, ...]
    #: Given wells with no stop, in the given order.
    wells_without_fields: tuple[str, ...]
    #: Stops of wells not given.
    fields_left_out: int
    #: Their wells, in the order first met in the route's order.
    wells_left_out: tuple[str, ...]


def chosen_fields(
    positions: Sequence[RoutePosition],
    wells: Sequence[str],
    *,
    shift_um: tuple[float, float],
    geometry_of: Callable[[str], WellGeometry],
    xy_limits_um: tuple[tuple[float, float], tuple[float, float]] | None,
) -> ChosenFields:
    """The plan's fields from a route's stops, each moved by ``shift_um`` and checked.

    ``shift_um`` is added to every stop's ``x_um`` and ``y_um``; ``(0.0, 0.0)``
    when no shift could be applied, which the caller says in ``assumed``. The
    stops' ``z_um`` and ``pfs_offset`` are never read: they are the selection's
    scan's, not a focus at the plan's objective (m3 §28.2 D3). ``geometry_of``
    gives a well's shape in the plan's frame; ``xy_limits_um`` is ``((x_lo,
    x_hi), (y_lo, y_hi))``, or ``None`` when the stage reports none.

    Returns empty ``wells`` and ``fields`` when no stop lies in a given well:
    the caller's refusal names the document.

    Raises:
        ValueError: A given well holds more than ``MAX_FIELDS_PER_WELL`` stops,
            or a centre, moved, lies outside its well or the soft limits. The
            first offending stop in visiting order is named.
    """
    given = tuple(wells)
    named = set(given)
    by_well: dict[str, list[RoutePosition]] = {}
    left_out = 0
    wells_left_out: dict[str, None] = {}
    # A stable sort: a route's positions are already in their order, and a record
    # edited by hand is visited by the order it states.
    for stop in sorted(positions, key=lambda p: p.order):
        if stop.well in named:
            by_well.setdefault(stop.well, []).append(stop)
        else:
            left_out += 1
            wells_left_out.setdefault(stop.well, None)

    dx_um, dy_um = shift_um
    planned: list[str] = []
    without: list[str] = []
    fields: list[PlannedField] = []
    for well in given:
        stops = by_well.get(well, [])
        if not stops:
            without.append(well)
            continue
        if len(stops) > MAX_FIELDS_PER_WELL:
            raise ValueError(
                f"the selection holds {len(stops)} fields in {well}, more than "
                f"{MAX_FIELDS_PER_WELL} a well"
            )
        geometry = geometry_of(well)
        for index, stop in enumerate(stops):
            x_um = stop.x_um + dx_um
            y_um = stop.y_um + dy_um
            if not geometry.contains(x_um, y_um):
                raise ValueError(
                    f"field {stop.name} of the selection lies at ({x_um:.1f}, "
                    f"{y_um:.1f}) µm in the plan's frame, outside well {well}: check "
                    "the offsets of the profile's [objectives], or run route again"
                )
            if xy_limits_um is not None:
                (x_lo, x_hi), (y_lo, y_hi) = xy_limits_um
                if not (x_lo <= x_um <= x_hi and y_lo <= y_um <= y_hi):
                    raise ValueError(
                        f"field {stop.name} of the selection lies at ({x_um:.1f}, "
                        f"{y_um:.1f}) µm, outside the stage's soft limits "
                        f"x [{x_lo:g}, {x_hi:g}], y [{y_lo:g}, {y_hi:g}] µm"
                    )
            fields.append(
                PlannedField(
                    well=well,
                    index=index,
                    name=f"{well}_{index:04d}",
                    x_um=x_um,
                    y_um=y_um,
                    stop=stop.name,
                )
            )
        planned.append(well)
    return ChosenFields(
        wells=tuple(planned),
        fields=tuple(fields),
        wells_without_fields=tuple(without),
        fields_left_out=left_out,
        wells_left_out=tuple(wells_left_out),
    )
