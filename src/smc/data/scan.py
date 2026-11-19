"""A scan's run folder, loaded with no hardware (algorithms design §3).

A collaborator develops a selection algorithm on a laptop that has no
Micro-Manager and no stand: this module reads what a scan wrote (the plan, the
frames and their records, the detected cells) from a run folder and hands it
over as models, arrays and columns. It imports no hardware layer, and
``tests/test_kit_imports.py`` keeps it so.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np
import tifffile
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from smc.plugins.geometry import Matrix2
from smc.plugins.slots import (
    Cell,
    CellsResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
    WellGeometry,
    load_cells,
    load_frame,
    normalise_well,
    well_geometry,
)
from smc.run import RunDocument, RunDocumentError, StageRecord

__all__ = ["DatasetError", "Scan", "cells_table", "load_scan"]

M = TypeVar("M", bound=BaseModel)

#: What ``Scan.cells_absent`` says when no ``cells`` record was given or found.
_NO_CELLS_RECORD = "no cells record"


class DatasetError(ValueError):
    """A run folder or dataset that cannot be read; the message names the file and why.

    A ``ValueError`` so that a caller that already handles a bad value handles
    this too, and one class for every refusal of this package, so that a tool
    built on it catches one thing and prints the message as it is.
    """


@dataclass(frozen=True, slots=True)
class Scan:
    """A finished scan, with everything an algorithm reads of it, and no stand.

    ``plan`` is the scan-plan's result and ``result`` the scan's; ``wells`` is
    the plan's order and ``geometry`` the geometry of each (a read-only
    mapping). The camera is the scan's (``result``), never the plan's copy: a
    plan laid for one matrix and a scan taken with another are two cameras,
    and the frames were taken with the second (FM-124).

    ``cells`` are the detected cells in the file's order, or ``None`` when
    there is no usable ``cells`` record, in which case ``cells_absent`` says
    why and ``cells_result`` is ``None`` too. ``cells == ()`` is "detected,
    found none", a different answer from ``None``, "never detected".
    ``profile`` is the profile the scan was taken on, ``""`` when it is not
    known (a scan built in memory, with no scan record to read it from).
    """

    run_dir: Path
    document: RunDocument
    profile: str
    plan: ScanPlanResult
    result: ScanResult
    wells: tuple[str, ...]
    geometry: Mapping[str, WellGeometry]
    cells: tuple[Cell, ...] | None
    cells_result: CellsResult | None
    cells_absent: str | None

    @classmethod
    def from_results(
        cls,
        run_dir: Path,
        document: RunDocument,
        *,
        plan: ScanPlanResult,
        result: ScanResult,
        cells_result: CellsResult | None,
        cells: Sequence[Cell] | None,
    ) -> Scan:
        """Build a scan from results that were read already; it checks none of them.

        ``route`` builds the input of a selector this way from what
        ``ctx.upstream`` has checked, so nothing here re-reads a file or
        re-judges a record (algorithms design §5.6). The profile is the
        document's ``scan`` record's, or ``""`` when it has none: a document
        that does not say is unknown, and the name of a stand is never
        guessed. ``cells=None`` is "no cells record" and ``cells=[]`` is "found
        none".
        """
        record = document.provider("scan")
        return _make_scan(
            run_dir,
            document,
            profile="" if record is None else record.profile,
            plan=plan,
            result=result,
            cells_result=cells_result,
            cells=None if cells is None else tuple(cells),
            cells_absent=_NO_CELLS_RECORD if cells is None else None,
        )

    @property
    def pixel_size_um(self) -> float:
        """The scan's pixel size, in µm per pixel."""
        return self.result.pixel_size_um

    @property
    def matrix_um_per_px(self) -> Matrix2:
        """The scan's camera matrix: pixels of a frame to stage µm (``smc.plugins.geometry``)."""
        return self.result.matrix_um_per_px

    @property
    def image_shape_px(self) -> tuple[int, int]:
        """The frame's shape as an array has it: ``(height, width)`` in pixels."""
        return self.result.image_shape_px

    def frames(self, well: str | None = None) -> tuple[ScanFrame, ...]:
        """The scan's frames in the scan's order: all of them, or those of one well.

        ``well`` is taken as a person types it (``b2`` and ``B02`` are ``B2``).
        A well that is not in this scan raises, naming the scan's wells: a
        typo must not read as a well with no frame. A well of the plan with no
        frame (every field of it failed) gives ``()``.

        Raises:
            DatasetError: ``well`` is not one of ``wells``.
        """
        if well is None:
            return self.result.frames
        name = self._well_named(well)
        return tuple(frame for frame in self.result.frames if frame.well == name)

    def image(self, frame: ScanFrame) -> NDArray[Any]:
        """The pixels of ``frame``, read from the run folder with the dtype and shape they were written with.

        Raises:
            DatasetError: The file is missing, empty, truncated or corrupt, or
                its path leads out of the run folder, or what it holds is not
                the scan's ``image_shape_px`` (a file that a power cut cut short
                reads as an empty array, and one written transposed is the right
                size and the wrong shape, FM-73). The message names the frame
                and its file, and the cause is chained when there is one.
                ``tifffile.TiffFileError`` is named: it is a ``ValueError`` only
                from tifffile 2026, and the floor is 2024.1; a file cut inside
                its header raises ``struct.error`` from the same read.
        """
        try:
            pixels = load_frame(self.run_dir, frame)
        except (OSError, ValueError, tifffile.TiffFileError, struct.error) as exc:
            raise DatasetError(f"frame {frame.name} ({frame.file}): {exc}") from exc
        if tuple(pixels.shape) != tuple(self.image_shape_px):
            raise DatasetError(
                f"frame {frame.name} ({frame.file}): the file holds "
                f"{tuple(pixels.shape)} pixels and the scan's frames are "
                f"{tuple(self.image_shape_px)}; it is cut short or is not this "
                f"scan's frame"
            )
        return pixels

    def cells_of(
        self, *, well: str | None = None, frame: int | None = None
    ) -> tuple[Cell, ...]:
        """The cells of one well, of one frame, or of both, in the file's order.

        ``frame`` is an index into ``result.frames``, as ``Cell.frame`` is. A
        scan with no cells raises instead of answering "none found": zero
        cells found and cells never detected are different answers.

        Raises:
            DatasetError: There are no cells (the message holds the reason),
                ``well`` is not one of ``wells``, or ``frame`` is out of range.
        """
        if self.cells is None:
            raise DatasetError(
                f"this scan has no cells: {self.cells_absent or _NO_CELLS_RECORD}"
            )
        chosen = self.cells
        if well is not None:
            name = self._well_named(well)
            chosen = tuple(cell for cell in chosen if cell.well == name)
        if frame is not None:
            count = len(self.result.frames)
            if not 0 <= frame < count:
                raise DatasetError(
                    f"frame {frame} is not in this scan (it has {count}, "
                    f"numbered from 0)"
                )
            chosen = tuple(cell for cell in chosen if cell.frame == frame)
        return chosen

    def _well_named(self, well: str) -> str:
        """The scan's own name of a well typed as a person types it, or a refusal naming the scan's wells."""
        names = {normalise_well(name): name for name in self.wells}
        found = names.get(normalise_well(well))
        if found is None:
            raise DatasetError(
                f"well {well!r} is not in this scan ({', '.join(self.wells)})"
            )
        return found


def _make_scan(
    run_dir: Path,
    document: RunDocument,
    *,
    profile: str,
    plan: ScanPlanResult,
    result: ScanResult,
    cells_result: CellsResult | None,
    cells: tuple[Cell, ...] | None,
    cells_absent: str | None,
) -> Scan:
    """The one place a ``Scan`` is built, so that ``load_scan`` and ``from_results`` cannot differ.

    The geometry is the plan's plate for each well of the plan. A well the
    plate does not have is a plan that does not belong to its plate: it is
    refused naming the well, and never dropped.
    """
    geometry: dict[str, WellGeometry] = {}
    for well in plan.wells:
        try:
            geometry[well] = well_geometry(plan.plate_plan, well)
        except ValueError as exc:
            raise DatasetError(f"the plan's well {well}: {exc}") from exc
    return Scan(
        run_dir=run_dir,
        document=document,
        profile=profile,
        plan=plan,
        result=result,
        wells=plan.wells,
        geometry=MappingProxyType(geometry),
        cells=cells,
        cells_result=cells_result,
        cells_absent=cells_absent,
    )


def load_scan(run_doc: Path | str) -> Scan:
    """Load the scan a run document holds, with its cells when there are usable ones.

    ``run_doc`` is the run document's file (not its folder): the frames and the
    cells file are read from its folder, at the paths the records name. Nothing
    is read of a frame here: ``Scan.image`` reads one when it is asked.

    A document that does not hold a usable scan is refused, because an
    algorithm that runs on a scan that is half there decides on nothing: the
    document is unreadable (never read as an empty scan, FM-229), has no
    ``scan`` or ``scan-plan`` record, or its last run did not end ``done``
    (a ``stopped``, ``running`` or ``failed`` scan is never read as finished,
    FM-106), is stale, or was a dry run. The ``cells`` record never refuses:
    when it is missing, stale or unreadable, ``Scan.cells`` is ``None`` and
    ``cells_absent`` says why, since a scan is worth loading before its cells
    are detected.

    Every refusal names the file, ``path: reason``, with the reason the run
    document itself gives where it has one (and the fix: which stage to run
    again).

    Raises:
        DatasetError: The document cannot be read or does not hold a usable
            scan and plan.
    """
    path = Path(run_doc)
    if _probe(path.is_dir):
        hint = "is a folder, not a run document; name a file"
        inside = path / "run.json"
        if _probe(inside.is_file):
            hint += f" (did you mean {inside}?)"
        raise DatasetError(f"{path}: {hint}")
    try:
        document = RunDocument.load(path)
    except RunDocumentError as exc:
        # Its text already starts with the path.
        raise DatasetError(str(exc)) from exc
    scan_record = document.provider("scan")
    profile = "" if scan_record is None else scan_record.profile
    for slot in ("scan", "scan-plan"):
        reason = document.unusable(slot, profile=profile, dry_run=False)
        if reason is not None:
            raise DatasetError(f"{path}: {reason}")
    plan_record = document.provider("scan-plan")
    assert scan_record is not None  # `unusable` refused a missing provider
    assert plan_record is not None
    result = _slot_result(path, scan_record, ScanResult, "scan")
    plan = _slot_result(path, plan_record, ScanPlanResult, "scan-plan")
    run_dir = path.parent
    cells_result, cells, absent = _read_cells(document, profile, run_dir)
    return _make_scan(
        run_dir,
        document,
        profile=profile,
        plan=plan,
        result=result,
        cells_result=cells_result,
        cells=cells,
        cells_absent=absent,
    )


def _probe(look: Callable[[], bool]) -> bool:
    """``look()``, or ``False`` when the file system refuses to answer (FM-24).

    ``Path.is_dir`` ignores only a missing path before Python 3.14, and "Access
    is denied" (a share, another user's folder) is an ``OSError`` that would
    end the load with a traceback. A probe that cannot tell says no, and
    ``RunDocument.load`` reads the path next and gives the reason in its own
    words, with the path.
    """
    try:
        return look()
    except OSError:
        return False


def _slot_result(path: Path, record: StageRecord, model: type[M], slot: str) -> M:
    """The record's result as ``model``, or a refusal that names the file, the slot and what to run again."""
    problem = "it holds no result"
    if record.result is not None:
        try:
            return model.model_validate(record.result)
        except ValidationError as exc:
            problem = _first_problem(exc)
    # Raised outside the handler: the validation error's text is the problem, not a cause.
    raise DatasetError(f"{path}: {_not_a_result(slot, problem, record.plugin)}")


def _not_a_result(slot: str, problem: str, plugin: str) -> str:
    return (
        f"the {slot} record is not a {slot} result this smc reads ({problem}); "
        f"run {plugin} again"
    )


def _first_problem(error: ValidationError) -> str:
    """``<where>: <what>`` of the first validation error, on one line."""
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "result"
    return f"{where}: {first['msg']}"


def _read_cells(
    document: RunDocument, profile: str, run_dir: Path
) -> tuple[CellsResult | None, tuple[Cell, ...] | None, str | None]:
    """The cells result and the cells, or ``(None, None, why not)``: cells never refuse a load.

    A ``cells`` record is read only when it is usable on the scan's profile, as
    any consumer of the slot reads it, and its file is the one the result
    counted (``load_cells`` checks). Every way it is not is a sentence, so that
    ``Scan.cells_absent`` can say it.
    """
    record = document.provider("cells")
    if record is None:
        return None, None, _NO_CELLS_RECORD
    reason = document.unusable("cells", profile=profile, dry_run=False)
    if reason is not None:
        return None, None, reason
    problem = "it holds no result"
    if record.result is not None:
        try:
            result = CellsResult.model_validate(record.result)
        except ValidationError as exc:
            problem = _first_problem(exc)
        else:
            try:
                return result, load_cells(run_dir, result), None
            except ValueError as exc:
                # "cells file <file> is missing ...", "... differs from the result ...".
                return None, None, str(exc)
    return None, None, _not_a_result("cells", problem, record.plugin)


def cells_table(cells: Sequence[Cell]) -> dict[str, NDArray[Any]]:
    """The cells as columns, one entry per cell, in the input's order.

    The keys are, in this order: ``well`` and ``label`` (``str``), ``frame``
    (``int64``, the index into the scan's frames), ``score``, ``x_um``,
    ``y_um``, ``row_px`` and ``col_px`` (the box's centre, in the frame's
    pixels), then ``row0_px``, ``col0_px``, ``row1_px`` and ``col1_px`` (the
    box, as ``Cell.box_px`` holds it: ``(row0, col0, row1, col1)``); all
    ``float64`` but the first three. ``pandas.DataFrame(cells_table(cells))``
    is the same table, and pandas is not a dependency of smc.

    An empty input gives twelve empty arrays of the same dtypes (``well`` and
    ``label`` as ``<U1``): a table that has columns and no row, never ``{}``.
    """
    boxes = np.array([cell.box_px for cell in cells], dtype=np.float64).reshape(
        len(cells), 4
    )
    row0, col0, row1, col1 = (np.ascontiguousarray(boxes[:, k]) for k in range(4))
    return {
        "well": np.array([cell.well for cell in cells], dtype=str),
        "label": np.array([cell.label for cell in cells], dtype=str),
        "frame": np.array([cell.frame for cell in cells], dtype=np.int64),
        "score": np.array([cell.score for cell in cells], dtype=np.float64),
        "x_um": np.array([cell.x_um for cell in cells], dtype=np.float64),
        "y_um": np.array([cell.y_um for cell in cells], dtype=np.float64),
        "row_px": (row0 + row1) / 2.0,
        "col_px": (col0 + col1) / 2.0,
        "row0_px": row0,
        "col0_px": col0,
        "row1_px": row1,
        "col1_px": col1,
    }
