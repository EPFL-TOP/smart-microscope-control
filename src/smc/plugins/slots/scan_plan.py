"""The scan-plan slot: which fields of which wells will be imaged, and in what order (m3 §3.4).

The plan is a ``useq`` plan (ADR-0004: where to image is a useq object) plus
the list of fields that survived clipping to the wells, so that the scan, the
cells, the density and the route all read the same fields.
"""

from __future__ import annotations

from typing import Any, Literal

import useq
from pydantic import (
    Field,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from smc.plugins.api import PluginResult
from smc.plugins.geometry import Matrix2
from smc.plugins.slots.base import SlotValue, refuse_dropped_keys, refuse_non_finite

__all__ = ["PlannedField", "ScanPlanResult"]


class PlannedField(SlotValue):
    """One image position to take: a field of a well, by useq's name (``A1_0003``)."""

    well: str
    index: int = Field(ge=0)
    name: str
    x_um: float
    y_um: float


class ScanPlanResult(PluginResult):
    """The plan of a scan: the plate, the wells, the camera and the fields.

    ``plate_plan`` holds the plate, its registration, the selected wells in
    visiting order and the ``GridRowsColumns``; ``fields`` are its image
    positions that survived clipping (§7), in visiting order, ``index``
    counting from 0 in each well and ``name`` useq's (gaps included).
    ``fov_um`` is ``(width, height)``, the footprint's extent in stage axes.
    ``matrix_source`` says where the camera matrix came from: a calibration
    that was measured, the profile's pixel size, or a parameter.

    ``estimate_s`` is the time of the scan with the autofocus used: each timed
    move and snap, and one lock of the autofocus in each well when the stand has
    one and the timings know its cost (m3 §7.4). ``estimate_without_autofocus_s``
    is the same with no lock, the time of ``scan`` with ``autofocus="off"``; it is
    equal to ``estimate_s`` when nothing was charged to leave out, and never above
    it. Both are ``None`` without usable timings: an unknown is not a zero. A plan
    written before #549 has no second figure (``None``) and an ``estimate_s`` that
    charged a refocus at every field; it loads as it is, and running ``scan-plan``
    again replaces it.

    ``plate_plan`` and every field are in the frame of ``objective`` (m3
    §18.3). ``plate_objective`` is the objective the registration was made
    with, and ``objective_shift_um`` the XY shift the plan applied to move it
    from that frame to ``objective``'s: ``(0.0, 0.0)`` for the same objective,
    ``None`` when it could not be applied (an unknown objective or an offset
    that was not measured), in which case the fields are the registration's
    unshifted. Both default to unknown, so that older run documents load.

    ``calibration_from`` is the run document the camera matrix was read from
    when the plan was given one (``scan-plan``'s ``calibration``, m3 §23), as
    typed with forward slashes and never resolved, and ``calibration_measured``
    when that calibration was measured (its record's ``finished_utc`` to the
    second, or ``at an unknown time``). A plate registered with one objective
    and scanned with another cannot keep the second's calibration in the
    registration's document, which holds one record per plugin, so the plan
    says which document it rests on and how old that is. Both are ``None``
    when the matrix is not from another document (the document's own
    calibration, the profile's size or a parameter), and in every document
    written before they existed.
    """

    plate_plan: useq.WellPlatePlan
    wells: tuple[str, ...]
    objective: str | None
    pixel_size_um: float = Field(gt=0.0)
    matrix_um_per_px: Matrix2
    matrix_source: Literal["camera-calibration", "profile", "parameter"]
    image_shape_px: tuple[int, int]
    fov_um: tuple[float, float]
    overlap: float = Field(ge=0.0, lt=1.0)
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    fields: tuple[PlannedField, ...]
    estimate_s: float | None = Field(ge=0.0)
    estimate_without_autofocus_s: float | None = Field(None, ge=0.0)
    plate_objective: str | None = None
    objective_shift_um: tuple[float, float] | None = None
    calibration_from: str | None = None
    calibration_measured: str | None = None

    @model_validator(mode="after")
    def _the_figure_without_the_autofocus_is_not_above_the_figure_with_it(
        self,
    ) -> ScanPlanResult:
        """Refuse a second figure that cannot be the first less its locks (FM-125).

        The page reads ``estimate_without_autofocus_s`` as the first figure with the
        locks taken out, so a hand-edited record with it above ``estimate_s``, or
        with it and no ``estimate_s``, would be drawn as a time that gets longer
        when the autofocus is switched off.
        """
        without = self.estimate_without_autofocus_s
        if without is None:
            return self
        if self.estimate_s is None:
            raise ValueError(
                "estimate_without_autofocus_s is given but estimate_s is None"
            )
        if without > self.estimate_s:
            raise ValueError(
                f"estimate_without_autofocus_s ({without:g}) is above "
                f"estimate_s ({self.estimate_s:g})"
            )
        return self

    @field_validator("plate_plan", mode="wrap")
    @classmethod
    def _plate_plan_is_as_strict_as_the_result(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> useq.WellPlatePlan:
        """Hold the ``useq`` plan to the rules the result holds its own fields to.

        ``useq``'s models accept ``nan`` and ``inf`` and ignore keys they do not
        know, and the result's ``allow_inf_nan=False`` and ``extra="forbid"``
        do not reach into another library's fields (measured, #199). A
        registration of ``nan`` would otherwise be a plan, and a key written by
        a newer smc would load, be dropped, and be written back without it
        (FM-107).
        """
        plan: useq.WellPlatePlan = handler(value)
        dumped = plan.model_dump(mode="python")
        refuse_non_finite(dumped, "plate_plan")
        if isinstance(value, dict):
            refuse_dropped_keys(value, dumped, "plate_plan")
        return plan
