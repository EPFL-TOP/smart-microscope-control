"""The objective-offsets slot: where each objective centres the same wells, measured on the plate (m3 §20.4).

``objective-offsets`` turns the turret to each target objective and measures
again, with ``plate-walls``' loop, the centres of wells the plate's registration
measured. The slot keeps what was measured under each target and the offsets
that follow from it, which ``smc objective offset --write`` puts into the
profile's ``[objectives]`` (m2 §13.3).

Every position here is a stage position (``XYStage.position_um()``), never an
image side (FM-189).
"""

from __future__ import annotations

from pydantic import Field

from smc.plugins.api import PluginResult
from smc.plugins.slots.base import SlotValue
from smc.plugins.slots.plate import WellFit

__all__ = [
    "MeasuredOffset",
    "ObjectiveOffsetsResult",
    "OffsetWell",
    "TargetMeasure",
]


class OffsetWell(SlotValue):
    """One well, measured under the registration's objective and again under a target.

    ``origin_xy_um`` is the centre ``plate-walls`` measured with the
    registration's objective (``plate.walls``); ``fit`` is the well under the
    target, as ``plate-walls`` records one (its ``recorded_xy_um`` is
    ``origin_xy_um``); ``delta_xy_um`` is ``fit.center_xy_um`` minus
    ``origin_xy_um``: what the target's offset is, as this one well says it.
    """

    well: str
    origin_xy_um: tuple[float, float]
    delta_xy_um: tuple[float, float]
    fit: WellFit


class TargetMeasure(SlotValue):
    """What one target objective measured, and the bound its wells were held to.

    ``calibration`` is the run document given for this target, as typed
    (``as_posix()``, never resolved: FM-24, FM-50) and ``calibration_measured``
    when it was measured; ``pixel_size_um`` is that calibration's.
    ``z_after_turn_um`` is Z where the stand left it after the turn, once it
    was still, and ``None`` on a stand with no focus drive: unknown, not zero.
    ``wells`` are in the order run; ``shift_xy_um`` is the mean of their
    ``delta_xy_um``, the target's offset from the registration's objective;
    ``spread_um`` is the largest distance between two wells' ``delta_xy_um`` and
    ``max_spread_um`` the bound it was held to. ``frames`` counts the frames
    taken under this target; ``folder`` is run-relative
    (``objective-offsets/<token>/<n>``, ``n`` the target's place in ``targets``
    from 1, so that no label is in a path) or ``None`` when frames are not
    saved. ``folder`` and ``z_after_turn_um`` are required fields that may hold
    ``None``: a producer must say that it does not know.
    """

    objective: str
    calibration: str
    calibration_measured: str
    pixel_size_um: float = Field(gt=0.0)
    z_after_turn_um: float | None
    wells: tuple[OffsetWell, ...]
    shift_xy_um: tuple[float, float]
    spread_um: float = Field(ge=0.0)
    max_spread_um: float = Field(gt=0.0)
    frames: int = Field(ge=0)
    folder: str | None


class MeasuredOffset(SlotValue):
    """One objective's XY offset from the reference, as ``--write`` writes it (m2 §13.2)."""

    objective: str
    xy_offset_um: tuple[float, float]


class ObjectiveOffsetsResult(PluginResult):
    """The offsets between objectives, measured from the plate.

    ``plate_objective`` is ``plate.objective``. ``reference`` is the objective
    the offsets are from: the profile's (``reference_in_profile`` is ``True``)
    or, when the profile names none, ``plate_objective`` (``False``).
    ``measures`` are in the order of ``targets``. ``offsets`` is what ``--write``
    writes: every objective the run measured or registered with, except
    ``reference``, the registration's objective first and then the targets in
    their order. ``objective_in_place`` is the last target, where the turret is
    left (nothing turns it back, FM-88).
    """

    plate_objective: str
    reference: str
    reference_in_profile: bool
    measures: tuple[TargetMeasure, ...]
    offsets: tuple[MeasuredOffset, ...]
    objective_in_place: str
    elapsed_s: float = Field(ge=0.0)
