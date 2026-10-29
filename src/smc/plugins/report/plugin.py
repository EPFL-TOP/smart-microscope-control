"""``report`` writes one static HTML page from the records of a scan (m3 §13).

The page itself is ``render.py`` and is pure. This reads the records upstream
and moves nothing, so a dry run writes a report like any other (nothing in it
needs the stand). What it decides, because the page is attached to issues and
read by people who were not at the stand:

* **Records that do not fit each other are refused, not drawn.** The runner's
  staleness covers a stage run again; this covers a record edited or replaced
  behind it (FM-111): a density that counted other labels, other fields or
  other cells than the files hold would be a page whose numbers and dots
  disagree. Each refusal names the stage to run again.
* **Only the run document's name goes into the page**, never a path (FM-50):
  a report is attached to issues, and a path names the person's account.
* **The time lives here, not in ``render``**, so that the same records give the
  same bytes. It is the time of the report, not of the experiment: no record
  carries the date of the imaging, and nothing here invents one.
* **A new file for every run.** A ``done`` result's file is never rewritten
  (m3 §2.3), so old reports stay on disk and can be deleted by hand. Nothing is
  checkpointed: a stop ends it before the write and leaves no file, and a write
  that fails says what is kept and to run ``report`` again.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import Field

from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.report.render import render
from smc.plugins.slots import (
    Cell,
    CellsResult,
    DensityResult,
    ReportResult,
    RouteResult,
    ScanPlanResult,
    ScanResult,
    load_cells,
)
from smc.run.files import (
    AtomicWriteError,
    new_token,
    relative_to_run,
    write_atomically,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from smc.hardware.microscope import Microscope

__all__ = ["ReportParams", "ReportPlugin"]

_NAME = "report"


class ReportParams(PluginParams):
    """How many cells one well's panel draws."""

    max_cells_drawn_per_well: int = Field(
        2000,
        ge=0,
        le=20_000,
        description=(
            "Most cells drawn in one well's panel (a regular subsample); the "
            "whole page draws at most 300 000."
        ),
    )


class ReportPlugin:
    """Writes the plate's density per class, the cells and fields of each well, and the route."""

    name = _NAME
    summary = (
        "Write one HTML file: the plate's density per class, the cells and "
        "fields of each well, and the route."
    )
    Params = ReportParams
    Result = ReportResult
    requires: tuple[type, ...] = ()
    provides = "report"
    reads: tuple[str, ...] = ("scan-plan", "scan", "cells", "density")
    reads_optional: tuple[str, ...] = ("route",)

    def run(
        self,
        microscope: Microscope,
        params: ReportParams,
        ctx: RunContext,
    ) -> ReportResult:
        """Write ``report/<token>.html`` and name it, relative to the run folder.

        Raises:
            PluginRunError: The records do not fit each other (the message names
                the stage to run again), the cells file no longer matches its
                result, a coordinate in a record is not a finite number, or
                the file cannot be written (the message says what is kept).
            StopRequested: The operator stopped it, before the write.
        """
        plan = ctx.upstream("scan-plan", ScanPlanResult)
        scan = ctx.upstream("scan", ScanResult)
        cells_result = ctx.upstream("cells", CellsResult)
        density = ctx.upstream("density", DensityResult)
        route = ctx.upstream_or_none("route", RouteResult)

        cells = _read_cells(ctx, cells_result)
        _require_records_that_fit(scan, cells_result, density, cells, route)
        ctx.raise_if_stopped()

        facts = (
            ("Run document", ctx.run_doc.name),
            ("Profile", microscope.profile.microscope.name),
            ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            (
                "Detector",
                f"{cells_result.model} (sha256 {cells_result.model_sha256[:12]}), "
                f"score ≥ {cells_result.score_threshold:g}, "
                f"merge radius {cells_result.merge_radius_um:g} µm, "
                f"{cells_result.device}",
            ),
        )
        assumed = [
            f"{stage}: {line}"
            for stage, result in (
                ("scan-plan", plan),
                ("scan", scan),
                ("cells", cells_result),
                ("density", density),
                ("route", route),
            )
            if result is not None
            for line in result.assumed
        ]
        problem = ""
        page = ""
        try:
            page = render(
                plan=plan,
                scan=scan,
                cells=cells,
                density=density,
                route=route,
                title=f"Plate report — {ctx.run_doc.name}",
                facts=facts,
                assumed=assumed,
                max_cells_drawn_per_well=params.max_cells_drawn_per_well,
            )
        except ValueError as exc:
            problem = str(exc)
        if problem:
            # Raised outside the handler: nothing here keeps the render's error (FM-98).
            raise PluginRunError(_NAME, problem)

        ctx.raise_if_stopped()
        file = _write(ctx, page)
        ctx.logger.info("report written: %s", file)
        return ReportResult(file=file)


def _read_cells(ctx: RunContext, cells_result: CellsResult) -> tuple[Cell, ...]:
    """The cells the file holds, or the refusal that says to run the cells stage again."""
    problem = ""
    cells: tuple[Cell, ...] = ()
    try:
        cells = load_cells(ctx.run_dir, cells_result)
    except ValueError as exc:
        problem = f"{exc}; run cells again"
    if problem:
        # Raised outside the handler: nothing here keeps the file's error (FM-98).
        raise PluginRunError(_NAME, problem)
    return cells


def _require_records_that_fit(
    scan: ScanResult,
    cells_result: CellsResult,
    density: DensityResult,
    cells: Sequence[Cell],
    route: RouteResult | None,
) -> None:
    """Refuse records that each validate and do not describe one scan, naming the first.

    The runner refuses a stage that is stale; it cannot see a record edited or
    replaced behind it (FM-111). A report over them would draw dots that
    density did not count, or a route for cells that were never detected. The
    order is the order of the page's story: the labels, then each well's fields,
    failed fields and cells, then what the cells or the route hold that
    density does not report.

    Raises:
        PluginRunError: The first difference, with the stage to run again.
    """
    if density.labels != cells_result.classes:
        raise PluginRunError(
            _NAME,
            f"density counted the labels {list(density.labels)} and the cells "
            f"stage detected {list(cells_result.classes)}; run density again",
        )
    frames_of = scan.frame_index()
    failed_in = Counter(failed.well for failed in scan.failed)
    held_in = Counter(cell.well for cell in cells)
    for reported in density.wells:
        well = reported.well
        if reported.fields != len(frames_of.get(well, [])):
            raise PluginRunError(
                _NAME,
                f"density reports {reported.fields} fields for well {well!r} and "
                f"the scan holds {len(frames_of.get(well, []))}; run density again",
            )
        if reported.failed_fields != failed_in.get(well, 0):
            raise PluginRunError(
                _NAME,
                f"density reports {reported.failed_fields} failed fields for well "
                f"{well!r} and the scan holds {failed_in.get(well, 0)}; "
                f"run density again",
            )
        counted = sum(reported.counts.values()) + reported.outside_well
        if counted != held_in.get(well, 0):
            raise PluginRunError(
                _NAME,
                f"density counted {counted} cells in well {well!r} and the cells "
                f"file holds {held_in.get(well, 0)}; run density again",
            )
    reported_wells = {reported.well for reported in density.wells}
    for well in sorted(held_in.keys() - reported_wells):
        raise PluginRunError(
            _NAME,
            f"the cells file holds a cell in well {well!r}, which density does "
            f"not report; run density again",
        )
    if route is None:
        return
    for label in route.labels:
        if label not in cells_result.classes:
            raise PluginRunError(
                _NAME,
                f"the route has the label {label!r}, which the cells stage did "
                f"not detect; run route again",
            )
    for stop in route.positions:
        if stop.well not in reported_wells:
            raise PluginRunError(
                _NAME,
                f"the route has a stop in well {stop.well!r}, which density does "
                f"not report; run route again",
            )


def _write(ctx: RunContext, page: str) -> str:
    """Write the page under a new token and return its POSIX path relative to the run folder.

    The write is all or nothing (FM-38). Only the rename failing, which on
    Windows is a program holding the file, leaves the whole new page in a
    hidden file that the message names.
    """
    path = ctx.run_dir / "report" / f"{new_token()}.html"
    where = relative_to_run(ctx.run_dir, path)
    problem = ""
    try:
        write_atomically(path, page.encode("utf-8"))
    except AtomicWriteError as exc:
        if exc.kept is not None:
            advice = (
                f"the whole report is kept as {relative_to_run(ctx.run_dir, exc.kept)}; "
                f"close any program that holds {where}, then run report again"
            )
        else:
            advice = "free some space or check the run folder, then run report again"
        problem = f"cannot write {where} ({exc.strerror or exc}); {advice}"
    if problem:
        # Raised outside the handler: nothing here keeps the write's error (FM-98).
        raise PluginRunError(_NAME, problem)
    return where
