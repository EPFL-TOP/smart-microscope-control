"""``density`` counts the cells of each well and class and divides by the area that was imaged (m3 §10).

The pure part (the raster of the imaged area, the counting per label) is
``algorithm.py``. This reads the three records upstream and moves nothing, so a
dry run is no different from any other. What it decides, because the numbers
that a biologist reads rest on it:

* **The scan's camera and the stage's readback are the field.** The footprints are
  the scan's frames at the position the stage read back, with the scan's matrix
  and frame shape: the plan's copy is not read, because a cell is in a field
  exactly as the scan saw it (FM-124), and the demo rounds a command.
* **The density is per imaged area.** A field that failed is in neither the
  numerator (its cells were not seen) nor the denominator (its area was not).
  A well whose every field failed is reported with 0 fields, its failed count and
  a density of ``None``: unknown, not zero.
* **A record that no longer fits the others is refused, not averaged over.** The
  scan was run again after the cells (another number of frames, other wells), the
  matrix cannot be inverted (FM-125), the cells file was replaced behind its
  result: each says which stage to run again (FM-111).
* **Nothing is written and nothing is kept.** A stop ends it between two wells
  and a re-run starts over, with the same result: the work is arithmetic on
  records already on disk. The raster is not interruptible inside one well.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import Field

from smc.plugins.api import PluginParams, RunContext
from smc.plugins.density.algorithm import GridTooFineError, density, scanned_area_mm2
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import Matrix2, footprint, pixel_of_stage
from smc.plugins.slots import (
    Cell,
    CellsResult,
    DensityResult,
    DensityWell,
    ScanPlanResult,
    ScanResult,
    WellGeometry,
    load_cells,
    well_geometry,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["DensityParams", "DensityPlugin"]

_NAME = "density"


class DensityParams(PluginParams):
    """The step of the grid the imaged area is counted on."""

    area_grid_um: float = Field(
        5.0,
        ge=0.5,
        le=100.0,
        description=(
            "Step of the grid on which the imaged area is counted; smaller is "
            "finer and slower."
        ),
    )


class DensityPlugin:
    """Counts the cells of each well per class, and divides by the area the scan imaged."""

    name = _NAME
    summary = "Count the cells of each well and class, and divide by the area that was imaged."
    Params = DensityParams
    Result = DensityResult
    requires: tuple[type, ...] = ()
    provides = "density"
    reads: tuple[str, ...] = ("scan-plan", "scan", "cells")
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: DensityParams,
        ctx: RunContext,
    ) -> DensityResult:
        """The density of every well the scan attempted, in the scan plan's order.

        Raises:
            PluginRunError: The scan took no frame; its camera matrix cannot be
                inverted; the cells were counted on other frames or other wells
                than the scan holds, or the cells file no longer matches; a well
                is not in the scan plan's list or not on the plate; or the grid
                is too fine for a well.
            StopRequested: The operator stopped it, between two wells.
        """
        plan = ctx.upstream("scan-plan", ScanPlanResult)
        scan = ctx.upstream("scan", ScanResult)
        cells_result = ctx.upstream("cells", CellsResult)

        if not scan.frames:
            raise PluginRunError(
                _NAME,
                "the scan took no frame (a dry run, or every field failed), so no "
                "area was imaged; run scan again",
            )
        _require_invertible(scan.matrix_um_per_px, scan.image_shape_px)
        if cells_result.frames != len(scan.frames):
            raise PluginRunError(
                _NAME,
                f"the cells stage counted {cells_result.frames} frames and the scan "
                f"holds {len(scan.frames)}; run cells again",
            )
        frames_of = scan.frame_index()
        failed_in = Counter(failed.well for failed in scan.failed)
        _require_the_same_wells(plan, frames_of, failed_in, cells_result)

        problem = ""
        cells: tuple[Cell, ...] = ()
        try:
            cells = load_cells(ctx.run_dir, cells_result)
        except ValueError as exc:
            problem = f"{exc}; run cells again"
        if problem:
            # Raised outside the handler: nothing here keeps the file's error (FM-98).
            raise PluginRunError(_NAME, problem)

        # A well the scan did not attempt is not reported, as the route leaves it out.
        reported = [
            well for well in plan.wells if well in frames_of or well in failed_in
        ]
        geometries: list[WellGeometry] = []
        areas: dict[str, float] = {}
        fields: dict[str, int] = {}
        failed_fields: dict[str, int] = {}
        for done, well in enumerate(reported):
            ctx.raise_if_stopped()
            geometry = _geometry_of(plan, well)
            frames = [scan.frames[position] for position in frames_of.get(well, [])]
            corners = [
                footprint(
                    frame.x_um,
                    frame.y_um,
                    matrix=scan.matrix_um_per_px,
                    image_shape_px=scan.image_shape_px,
                )
                for frame in frames
            ]
            geometries.append(geometry)
            areas[geometry.well] = _area_of(corners, geometry, well, params)
            fields[geometry.well] = len(frames)
            failed_fields[geometry.well] = failed_in.get(well, 0)
            ctx.progress(done + 1, len(reported), well)

        problem = ""
        counted: tuple[DensityWell, ...] = ()
        try:
            counted = density(
                cells,
                wells=geometries,
                areas_mm2=areas,
                fields=fields,
                failed_fields=failed_fields,
                labels=cells_result.classes,
            )
        except ValueError as exc:
            problem = f"{exc}; run cells again"
        if problem:
            raise PluginRunError(_NAME, problem)

        return DensityResult(
            labels=cells_result.classes,
            area_grid_um=params.area_grid_um,
            wells=counted,
            assumed=(
                "the imaged area is the union of the scan's fields at the positions "
                f"the stage read back, counted on a grid of {params.area_grid_um:g} "
                "µm: an edge of it is known to within half a step",
                "the wells' shapes and positions are the plate registration's and "
                "the plate's nominal sizes: the wall was not measured, so the area "
                "and the cells near it are only as exact as the registration",
            ),
        )


def _require_invertible(matrix: Matrix2, shape: tuple[int, int]) -> None:
    """Refuse a matrix no pixel maps back through, with the refusal ``pixel_of_stage`` gives.

    The scan's matrix validates field by field, so a singular one (a run
    document edited by hand) would otherwise be a field of no area, refused as
    "footprint 0 has no area" far from its cause (FM-125).
    """
    problem = ""
    try:
        pixel_of_stage(0.0, 0.0, 0.0, 0.0, matrix=matrix, image_shape_px=shape)
    except ValueError as exc:
        problem = str(exc)
    if problem:
        # Raised outside the handler: nothing here keeps the geometry's error (FM-98).
        raise PluginRunError(_NAME, problem)


def _require_the_same_wells(
    plan: ScanPlanResult,
    frames_of: dict[str, list[int]],
    failed_in: Counter[str],
    cells_result: CellsResult,
) -> None:
    """Refuse cells and a scan that are not of the same wells, and a scan that holds a well the plan does not list.

    The scan run again after the cells (FM-111) leaves records that each
    validate: counting over them would divide one run's cells by another's area.
    The wells are named in sorted order so that the refusal is the same on every
    run.
    """
    for well in sorted(frames_of.keys() - cells_result.counts.keys()):
        raise PluginRunError(
            _NAME,
            f"the scan imaged well {well!r}, which the cells stage did not count; "
            f"run cells again",
        )
    for well in sorted(cells_result.counts.keys() - frames_of.keys()):
        raise PluginRunError(
            _NAME,
            f"the cells stage counted well {well!r}, which the scan has no frame in; "
            f"run cells again",
        )
    # The loop over the plan's wells would leave out a well it does not list.
    for well in sorted((frames_of.keys() | failed_in.keys()) - set(plan.wells)):
        raise PluginRunError(
            _NAME,
            f"the scan holds well {well!r}, which the scan plan does not list among "
            f"its wells; run scan-plan, scan and cells again",
        )


def _geometry_of(plan: ScanPlanResult, well: str) -> WellGeometry:
    """The shape and place of a well on the plan's plate, or the refusal that says to plan again."""
    problem = ""
    geometry: WellGeometry | None = None
    try:
        geometry = well_geometry(plan.plate_plan, well)
    except ValueError as exc:
        problem = f"{exc}; run scan-plan again"
    if geometry is None:
        # Raised outside the handler: nothing here keeps the plate's error (FM-98).
        raise PluginRunError(_NAME, problem)
    return geometry


def _area_of(
    corners: list[tuple[tuple[float, float], ...]],
    geometry: WellGeometry,
    well: str,
    params: DensityParams,
) -> float:
    """The imaged area of one well, with the refusals of the raster said in the stage's words."""
    problem = ""
    area = 0.0
    try:
        area = scanned_area_mm2(corners, geometry, grid_um=params.area_grid_um)
    except GridTooFineError as exc:
        problem = f"{exc}; raise area_grid_um"
    except ValueError as exc:
        problem = f"well {well}: {exc}"
    if problem:
        # Raised outside the handlers: nothing here keeps the raster's error (FM-98).
        raise PluginRunError(_NAME, problem)
    return area
