"""A scan's run folder, loaded with no hardware (algorithms design §3).

A collaborator develops a selection algorithm on a laptop that has no
Micro-Manager and no stand: this module reads what a scan wrote (the plan, the
frames and their records, the detected cells) from a run folder and hands it
over as models, arrays and columns. It imports no hardware layer, and
``tests/test_kit_imports.py`` keeps it so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import tifffile
from numpy.typing import NDArray

from smc.plugins.geometry import Matrix2
from smc.plugins.slots import (
    Cell,
    CellsResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
    WellGeometry,
    load_frame,
    normalise_well,
    well_geometry,
)
from smc.run import RunDocument

__all__ = ["DatasetError", "Scan", "cells_table"]

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
            DatasetError: The file is missing, empty or corrupt, or its path
                leads out of the run folder. The message names the frame and
                its file, and the cause is chained. ``tifffile.TiffFileError``
                is named: it is a ``ValueError`` only from tifffile 2026, and
                the floor is 2024.1.
        """
        try:
            return load_frame(self.run_dir, frame)
        except (OSError, ValueError, tifffile.TiffFileError) as exc:
            raise DatasetError(f"frame {frame.name} ({frame.file}): {exc}") from exc

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
