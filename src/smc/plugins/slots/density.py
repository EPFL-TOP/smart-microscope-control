"""The density slot: cells per mm² of each well (m3 §3.5)."""

from __future__ import annotations

from pydantic import Field

from smc.plugins.api import PluginResult
from smc.plugins.slots.base import SlotValue

__all__ = ["DensityResult", "DensityWell"]


class DensityWell(SlotValue):
    """The cells of one well, counted per label, and the area that was imaged.

    ``per_mm2`` is ``None`` when ``area_mm2`` is 0, because no frame of the
    well was taken: the density is unknown, not zero (unknown beats guessed).
    ``outside_well`` counts detections the well's shape rejected.
    """

    well: str
    fields: int = Field(ge=0)
    failed_fields: int = Field(ge=0)
    area_mm2: float = Field(ge=0.0)
    counts: dict[str, int]
    outside_well: int = Field(ge=0)
    per_mm2: dict[str, float] | None


class DensityResult(PluginResult):
    """Density per well, over the labels the cells stage detected.

    ``area_grid_um`` is the step of the grid on which the imaged area was
    computed, so that overlapping fields are not counted twice.
    """

    labels: tuple[str, ...]
    area_grid_um: float = Field(gt=0.0)
    wells: tuple[DensityWell, ...]
