"""The cells' plane's offset in a route stop: derive it from the scan's frame, or say why not (m3 §26).

A scan with the focus on the cells takes each frame at the cells' plane with the
autofocus off. The frame's Z is that plane, and its ``pfs_offset`` is the offset
of the lock the sweep was centred on: it names the plane the autofocus held,
``dz_um`` below or above the cells. The export writes that offset with the
autofocus enabled, and NIS-Elements would engage it there, with none of smc's
guards, and image the plane the autofocus held and not the cells'. So the offset
a stop carries is either the cells' plane's, derived here from the profile's
factor, or none, and the route says which and why.

Pure: it reads the frame and two numbers of the profile and moves nothing. What
it decides:

* **Only when there is a distance to convert** (D3): a ``locked`` frame with an
  offset and a focus step whose ``dz_um`` is not 0. Any other frame is as the scan
  recorded it, with no factor needed.
* **Each bound ends in no offset, never in a clamp** (D5, FM-121): a clamped
  offset would be a plane nobody chose, written where the autofocus engages. The
  factor is the scan's objective's by exact label (FM-214), finite and not zero;
  the distance is within the largest sweep the scan can make (FM-111); the offset
  is inside the profile's offset limits.
* **The factor's sign is kept** and the offset is rounded to a whole unit, the
  stand holds whole ones. ``z_um`` is never changed: it is the readback at the
  cells' plane.
* **A frame that found no peak is counted, not derived** (D8): its plane is the
  sweep's centre, which a number cannot say.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from smc.plugins.cell_focus.algorithm import MAX_RANGE_UM
from smc.plugins.slots import OffsetFromFocus, ScanFrame

__all__ = ["StopFocus", "StopOutcome", "focus_assumed", "stop_focus"]

#: ``recorded``: the frame's Z and offset as the scan recorded them (no distance to
#: convert, or no frame); ``derived``: the offset is the cells' plane's; the others
#: are the reason there is none (D3 to D5).
StopOutcome = Literal[
    "recorded", "derived", "no-objective", "no-factor", "outside-limits", "implausible"
]


@dataclass(frozen=True, slots=True)
class StopFocus:
    """What a stop takes from its frame: Z, the offset, and what became of the offset.

    ``refused_offset`` is the offset that ``outside-limits`` refused, for the line
    that says so. ``no_peak`` is true when the frame's own focus sweep found no
    peak, whatever became of the offset (D8).
    """

    z_um: float | None
    pfs_offset: float | None
    derived: OffsetFromFocus | None
    outcome: StopOutcome
    refused_offset: float | None
    no_peak: bool


def stop_focus(
    frame: ScanFrame | None,
    *,
    objective: str | None,
    factors: Mapping[str, float],
    offset_limits: tuple[float, float] | None,
) -> StopFocus:
    """The Z and the offset a stop takes from ``frame``, the offset derived for the cells' plane when it can be.

    ``factors`` are the profile's offset units per µm by objective label and
    ``offset_limits`` its autofocus offset limits, ``None`` when it has none.
    ``objective`` is the scan's. The steps are in the order of the design, and
    the first that applies decides.
    """
    if frame is None:
        return StopFocus(None, None, None, "recorded", None, no_peak=False)
    focus = frame.cell_focus
    no_peak = (
        focus is not None
        and focus.source == "swept"
        and focus.sweep is not None
        and focus.sweep.outcome != "peak"
    )
    z_um = frame.z_um
    if (
        frame.focus != "locked"
        or frame.pfs_offset is None
        or focus is None
        or focus.dz_um == 0.0
    ):
        return StopFocus(z_um, frame.pfs_offset, None, "recorded", None, no_peak)

    dz_um = focus.dz_um
    if not math.isfinite(dz_um) or abs(dz_um) > MAX_RANGE_UM:
        return StopFocus(z_um, None, None, "implausible", None, no_peak)
    if objective is None:
        return StopFocus(z_um, None, None, "no-objective", None, no_peak)
    factor = factors.get(objective)
    if factor is None or not math.isfinite(factor) or factor == 0.0:
        return StopFocus(z_um, None, None, "no-factor", None, no_peak)
    summed = frame.pfs_offset + dz_um * factor
    # A factor so large that the sum overflows is not a factor: `round` raises
    # on an infinity, and a record must end as a stop without an offset.
    if not math.isfinite(summed):
        return StopFocus(z_um, None, None, "no-factor", None, no_peak)
    offset = float(round(summed))
    if offset_limits is not None and not offset_limits[0] <= offset <= offset_limits[1]:
        return StopFocus(z_um, None, None, "outside-limits", offset, no_peak)
    derived = OffsetFromFocus(
        lock_offset=frame.pfs_offset, dz_um=dz_um, offset_per_um=factor
    )
    return StopFocus(z_um, offset, derived, "derived", None, no_peak)


def focus_assumed(
    stops: Sequence[StopFocus],
    *,
    total: int,
    objective: str | None,
    offset_limits: tuple[float, float] | None,
) -> tuple[str, ...]:
    """The ``assumed`` lines for the stops' offsets, each only when its count is above 0.

    ``total`` is all the route's stops. The words that matter come first and a
    free text, the objective's label, is last, so that the plate page's card,
    which cuts a line at 600 characters, never cuts what a line says (FM-205,
    FM-139): the longest forms are at most 426 characters with a 60-character
    label.
    """
    count = {
        outcome: [stop for stop in stops if stop.outcome == outcome]
        for outcome in (
            "derived",
            "no-objective",
            "no-factor",
            "outside-limits",
            "implausible",
        )
    }
    lines: list[str] = []
    derived = count["derived"]
    if derived:
        distances = [d.derived.dz_um for d in derived if d.derived is not None]
        factor = next(d.derived.offset_per_um for d in derived if d.derived is not None)
        lines.append(
            f"{len(derived)} of {total} stops carry a pfs_offset derived for the "
            f"cells' plane, not read with PFS locked there: their frame was taken "
            f"{min(distances):+.2f}…{max(distances):+.2f} µm from the plane PFS "
            f"held, and the offset is the lock's plus that distance times the "
            f"profile's factor (one number for a factor that varies with the "
            f"offset), rounded to a whole unit; {factor:g} offset units per µm "
            f"for {_named(objective)}"
        )
    if count["no-objective"]:
        lines.append(
            f"{len(count['no-objective'])} of {total} stops have no pfs_offset: "
            f"their frame was taken away from the plane PFS held (the focus on "
            f"the cells) and the scan recorded no objective, so the profile's "
            f"offset_per_um factor cannot be looked up; they keep z_um, the "
            f"cells' plane"
        )
    if count["no-factor"]:
        lines.append(
            f"{len(count['no-factor'])} of {total} stops have no pfs_offset: "
            f"their frame was taken away from the plane PFS held (the focus on "
            f"the cells) and the profile has no [autofocus.offset_per_um] factor "
            f"for the scan's objective; they keep z_um, the cells' plane "
            f"(objective: {_named(objective)})"
        )
    refused = count["outside-limits"]
    if refused:
        first = next(r.refused_offset for r in refused if r.refused_offset is not None)
        low, high = offset_limits if offset_limits is not None else (math.nan, math.nan)
        lines.append(
            f"{len(refused)} of {total} stops have no pfs_offset: the offset "
            f"derived for the cells' plane ({first:.0f} for the first) lies "
            f"outside the profile's autofocus_offset_soft_limits "
            f"[{low:g}, {high:g}]; they keep z_um, the cells' plane"
        )
    if count["implausible"]:
        lines.append(
            f"{len(count['implausible'])} of {total} stops have no pfs_offset: "
            f"their frame's plane is not within {MAX_RANGE_UM:g} µm of the plane "
            f"PFS held (the focus sweep's largest range), so the record cannot "
            f"be right; run scan again; they keep z_um"
        )
    flat = sum(stop.no_peak for stop in stops)
    if flat:
        lines.append(
            f"{flat} of {total} stops took their frame from a field whose focus "
            f"sweep found no peak (flat, at an edge or two peaks): its plane is "
            f"the sweep's centre, not the cells' plane"
        )
    return tuple(lines)


def _named(objective: str | None) -> str:
    """The objective's label, or the words for its absence: a line never says ``None``."""
    return objective if objective is not None else "no objective recorded"
