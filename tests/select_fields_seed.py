"""The run folder the tests of ``scripts/dev/select_fields.py`` share (#343).

Not a test module: ``seed_run`` writes ``run.json`` with the four records the
prototype reads upstream (``plate-registration``, ``scan-plan``, ``scan`` and
``cells``) and the TIFFs of the scan, in a folder the caller owns. It copies the
builders of ``tests/test_route_demo.py`` (``_seed``, ``_record``, ``_matrix``)
so that the prototype reads what the real stages write and not a shortcut.

The camera is 60 x 80 pixels (not square, FM-73) at 5 um, so a field is
400 x 300 um on a 96-well plate whose A1 is at (1000, -500) um, 9 mm apart, round
wells of 6400 um. The caller turns or mirrors the camera with ``matrix_of``.

What it does not model: a scan whose frames are not at the planned fields (the
readback is the plan's field, 0.3 um and -0.2 um off, for every frame), a stand
(no microscope is opened: nothing here is a ``demo`` test), a cells file that a
detector wrote (the cells are the caller's, with whatever box they are given),
and a second attempt of any stage.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tifffile
from pydantic import BaseModel

from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    Cell,
    CellsFile,
    CellsResult,
    FocusState,
    PlateResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)
from smc.run import StageRecord, StageStatus, write_stage

SHAPE = (60, 80)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 5.0
FOV = (SHAPE[1] * PIXEL_UM, SHAPE[0] * PIXEL_UM)  # (400, 300): width, height
OBJECTIVE = "Nikon 10X S Fluor"
CLASSES = ("single", "doublet", "debris")
#: Where the 96-well plate's wells are, with A1 at (1000, -500): 9 mm apart, 6.4 mm wide.
CENTRES = {
    "A1": (1000.0, -500.0),
    "A2": (10000.0, -500.0),
    "B2": (10000.0, -9500.0),
    "B1": (1000.0, -9500.0),
}
RADIUS_UM = 3200.0
#: What every frame's readback differs from the planned field by.
READBACK_OFFSET_UM = (0.3, -0.2)


def matrix_of(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


UPRIGHT = matrix_of(PIXEL_UM, 0.0, False)


def cell_at(
    well: str,
    dx: float,
    dy: float,
    label: str = "single",
    *,
    box_px: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
) -> Cell:
    """A cell ``(dx, dy)`` um from the centre of ``well``, with the box the caller gives (a 10 x 10 pixel one by default)."""
    return Cell(
        well=well,
        x_um=CENTRES[well][0] + dx,
        y_um=CENTRES[well][1] + dy,
        label=label,
        score=0.9,
        frame=0,
        box_px=box_px,
    )


@dataclass(frozen=True)
class Seed:
    """What the records upstream say, kept for the expected values."""

    path: Path
    plan: ScanPlanResult
    scan: ScanResult
    cells: tuple[Cell, ...]
    result: CellsResult
    matrix: Matrix2
    shape: tuple[int, int]


def record(
    path: Path,
    plugin: str,
    provides: str,
    result: BaseModel,
    *,
    attempt: str = "a1",
    day: int = 28,
) -> None:
    """A stage record as an earlier run of ``plugin`` would have left it, ``done``.

    A second attempt of a stage is ``attempt="a2"`` on a later ``day`` (the latest start
    decides which record fills a slot).
    """
    then = datetime(2026, 10, day, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt=attempt,
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def extent_of(matrix: Matrix2, shape: tuple[int, int]) -> tuple[float, float]:
    """The oracle's field of view: the extent of ``footprint()`` corners at the origin."""
    corners = footprint(0.0, 0.0, matrix=matrix, image_shape_px=shape)
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


def flat_frame(shape: tuple[int, int]) -> npt.NDArray[np.uint16]:
    """The frame a scan with no pixels of interest holds: one value everywhere."""
    return np.full(shape, 1000, dtype=np.uint16)


def seed_run(
    tmp_path: Path,
    *,
    cells: Sequence[Cell] | Callable[[ScanResult], Sequence[Cell]],
    wells: tuple[str, ...] = ("A1",),
    matrix: Matrix2 = UPRIGHT,
    shape: tuple[int, int] = SHAPE,
    rows: int = 2,
    columns: int = 2,
    overlap: float = 0.1,
    focus_of: Mapping[tuple[str, int], FocusState] | None = None,
    frame_pixels: Callable[[ScanFrame], npt.NDArray[np.generic]] | None = None,
    write_frames: bool = True,
    scan_objective: str | None = OBJECTIVE,
    classes: tuple[str, ...] = CLASSES,
    file_token: str = "t1",
    profile: str = "demo",
    dry_run: bool = False,
) -> Seed:
    """The plan, the scan and the cells of ``wells``, in ``tmp_path/run.json``.

    ``cells`` is a list, or a function of the scan when the cells are placed
    relative to its frames (their positions are the plan's, which the test cannot
    know before the plan exists). ``overlap=0.0`` makes neighbouring fields touch
    exactly. Frames are at the planned fields, their readback ``READBACK_OFFSET_UM``
    off the target, and their Z and autofocus offset distinct (``z_um = 1000 +
    10 k``, ``pfs_offset = 2000 + 10 k`` for the k-th frame). ``focus_of`` gives
    the autofocus state of the frames it names, keyed by ``(well, field index)``;
    the others are ``locked``, and a frame that is ``off`` or ``none`` has no
    autofocus offset. ``frame_pixels`` gives a frame's pixels (flat by default);
    ``write_frames=False`` leaves the TIFFs unwritten, so that every read fails.
    """
    path = tmp_path / "run.json"
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=CENTRES["A1"],
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    fov = extent_of(matrix, shape)
    plate_plan, fields = plan_fields(
        plate,
        wells,
        fov_um=fov,
        footprint_of=lambda x, y: footprint(x, y, matrix=matrix, image_shape_px=shape),
        rows=rows,
        columns=columns,
        overlap=overlap,
        clip_to_well=True,
    )
    plan = ScanPlanResult(
        plate_plan=plate_plan,
        wells=wells,
        objective=scan_objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        matrix_source="parameter",
        image_shape_px=shape,
        fov_um=fov,
        overlap=overlap,
        rows=rows,
        columns=columns,
        fields=fields,
        estimate_s=None,
    )
    frames: list[ScanFrame] = []
    for k, field in enumerate(fields):
        focus: FocusState = (focus_of or {}).get((field.well, field.index), "locked")
        frames.append(
            ScanFrame(
                well=field.well,
                index=field.index,
                name=field.name,
                file=f"scan/t/{field.name}.tif",
                x_um=field.x_um + READBACK_OFFSET_UM[0],
                y_um=field.y_um + READBACK_OFFSET_UM[1],
                z_um=1000.0 + 10.0 * k,
                focus=focus,
                pfs_offset=None if focus in ("off", "none") else 2000.0 + 10.0 * k,
                t_s=float(k),
            )
        )
    scan = ScanResult(
        folder="scan/t",
        objective=scan_objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
        exposure_ms=10.0,
        frames=tuple(frames),
        failed=(),
    )
    if write_frames:
        (tmp_path / "scan" / "t").mkdir(parents=True, exist_ok=True)
        for frame in frames:
            pixels = (
                frame_pixels(frame) if frame_pixels is not None else flat_frame(shape)
            )
            tifffile.imwrite(tmp_path / frame.file, pixels)
    cell_list = tuple(cells(scan) if callable(cells) else cells)
    (tmp_path / "cells").mkdir(exist_ok=True)
    name = f"cells/{file_token}.json"
    (tmp_path / name).write_text(
        CellsFile(cells=cell_list).model_dump_json(), encoding="utf-8"
    )
    result = CellsResult(
        file=name,
        model="m.pth",
        model_sha256="0" * 64,
        classes=classes,
        device="cpu",
        score_threshold=0.5,
        merge_radius_um=5.0,
        frames=len(frames),
        counts={
            well: {
                label: sum(1 for c in cell_list if c.well == well and c.label == label)
                for label in classes
            }
            for well in wells
        },
        merged=0,
        seconds_per_frame=None,
    )
    record(path, "plate-registration", "plate", plate)
    record(path, "scan-plan", "scan-plan", plan)
    record(path, "scan", "scan", scan)
    record(path, "cells", "cells", result)
    return Seed(path, plan, scan, cell_list, result, matrix, shape)
