"""The scan-plan slot: which fields of which wells will be imaged, and in what order (m3 §3.4).

The plan is a ``useq`` plan (ADR-0004: where to image is a useq object) plus
the list of fields that survived clipping to the wells, so that the scan, the
cells, the density and the route all read the same fields.
"""

from __future__ import annotations

from typing import Any, Literal

import useq
from pydantic import Field, ValidatorFunctionWrapHandler, field_validator

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
