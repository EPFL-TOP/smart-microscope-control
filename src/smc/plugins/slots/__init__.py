"""The slot models of the plate-scan workflow (m3 §3).

Every stage of the workflow fills a slot of the run document with a result,
and the models of those results live here, in one package written first
(m3 §2.3): plugins built in parallel import their ``Result`` from here instead
of from modules that do not exist yet. ``smc.plugins`` does not import this
package, so importing a plugin never loads it by accident.
"""

from __future__ import annotations

from smc.plugins.slots.base import SlotValue
from smc.plugins.slots.camera_calibration import CameraCalibration
from smc.plugins.slots.plate import (
    LatticeFit,
    PlateResult,
    WallTouch,
    WellGeometry,
    WellRef,
    normalise_well,
    well_geometry,
)
from smc.plugins.slots.scan_plan import PlannedField, ScanPlanResult

__all__ = [
    "CameraCalibration",
    "LatticeFit",
    "PlannedField",
    "PlateResult",
    "ScanPlanResult",
    "SlotValue",
    "WallTouch",
    "WellGeometry",
    "WellRef",
    "normalise_well",
    "well_geometry",
]
