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
from smc.plugins.slots.cells import Cell, CellsFile, CellsResult, load_cells
from smc.plugins.slots.density import DensityResult, DensityWell
from smc.plugins.slots.nis_export import NisExportResult
from smc.plugins.slots.plate import (
    LatticeFit,
    PlateResult,
    WallTouch,
    WellGeometry,
    WellRef,
    normalise_well,
    well_geometry,
)
from smc.plugins.slots.report import ReportResult
from smc.plugins.slots.route import RoutePosition, RouteResult
from smc.plugins.slots.scan import (
    FailedField,
    FocusState,
    ScanFrame,
    ScanResult,
    load_frame,
)
from smc.plugins.slots.scan_plan import PlannedField, ScanPlanResult
from smc.plugins.slots.timings import TimingsResult

__all__ = [
    "CameraCalibration",
    "Cell",
    "CellsFile",
    "CellsResult",
    "DensityResult",
    "DensityWell",
    "FailedField",
    "FocusState",
    "LatticeFit",
    "NisExportResult",
    "PlannedField",
    "PlateResult",
    "ReportResult",
    "RoutePosition",
    "RouteResult",
    "ScanFrame",
    "ScanPlanResult",
    "ScanResult",
    "SlotValue",
    "TimingsResult",
    "WallTouch",
    "WellGeometry",
    "WellRef",
    "load_cells",
    "load_frame",
    "normalise_well",
    "well_geometry",
]
