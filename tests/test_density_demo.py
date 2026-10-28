"""``density`` through ``run_plugin``: the three records upstream in, counts, area and density out (m3 §10, #193).

The plugin reads three records and touches no device (``requires = ()``): the
demo stand is here because ``run_plugin`` needs a ``Microscope``.

The upstream records are written by the test: the scan plan's plate from
``plan_fields``, the scan with a frame at every position the test gives and no
TIFF (density reads none), and the cells as a cells file whose counts are
computed from the cells. The camera is 60 x 80 pixels (not square, FM-73) at
5 µm: a field of 400 x 300 µm, 0.12 mm². Every expected value is computed here
from the numbers the test seeds, never read from the result's own fields
(FM-124).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from smc.hardware import Microscope
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.density import algorithm
from smc.plugins.density.plugin import DensityParams, DensityPlugin
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    Cell,
    CellsFile,
    CellsResult,
    DensityResult,
    FailedField,
    PlannedField,
    PlateResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

pytestmark = pytest.mark.demo

SHAPE = (60, 80)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 5.0
FIELD_MM2 = 400.0 * 300.0 / 1e6  # 0.12
CLASSES = ("single", "doublet", "debris")
#: Where the 96-well plate's wells are, with A1 at (1000, -500): 9 mm apart, 6.4 mm wide.
CENTRES = {
    "A1": (1000.0, -500.0),
    "A2": (10000.0, -500.0),
    "B1": (1000.0, -9500.0),
}
RADIUS_UM = 3200.0

#: ``(well, x_um, y_um)``: a frame's readback, or a cell's position.
Point = tuple[str, float, float]


def _matrix(pixel_size_um: float) -> Matrix2:
    """The upright camera of m3 §3.1, written here and not taken from the code."""
    return ((pixel_size_um, 0.0), (0.0, -pixel_size_um))


UPRIGHT = _matrix(PIXEL_UM)


def _at(well: str, dx: float, dy: float) -> Point:
    """The point ``(dx, dy)`` µm from the centre of a well."""
    return (well, CENTRES[well][0] + dx, CENTRES[well][1] + dy)


@dataclass(frozen=True)
class Seed:
    """What the three upstream records say, kept for the expected values."""

    plan: ScanPlanResult
    scan: ScanResult
    cells: tuple[Cell, ...]
    result: CellsResult


def _record(path: Path, plugin: str, provides: str, result: BaseModel) -> None:
    """A stage record as an earlier run of ``plugin`` would have left it, ``done``."""
    then = datetime(2026, 10, 28, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt="a1",
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


def _extent(matrix: Matrix2, shape: tuple[int, int]) -> tuple[float, float]:
    """The oracle's field of view: the extent of ``footprint()`` corners at the origin."""
    corners = footprint(0.0, 0.0, matrix=matrix, image_shape_px=shape)
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _seed(
    tmp_path: Path,
    *,
    frames: Sequence[Point],
    failed: Sequence[Point] = (),
    cells: Sequence[tuple[Point, str]] = (),
    wells: tuple[str, ...] | None = None,
    listed: tuple[str, ...] | None = None,
    planned: Sequence[Point] | None = None,
    matrix: Matrix2 = UPRIGHT,
    plan_matrix: Matrix2 | None = None,
    shape: tuple[int, int] = SHAPE,
    counted: Sequence[str] | None = None,
    cells_frames: int | None = None,
    classes: tuple[str, ...] = CLASSES,
    file_token: str = "t1",
) -> Seed:
    """The plan, the scan and the cells, in ``tmp_path/run.json``.

    ``wells`` is the plan's order (default: the order wells first appear in
    ``frames`` and ``failed``); ``listed`` is what the plan's own ``wells`` says
    when it is not that (a record that disagrees with the scan). ``planned`` are
    the positions the plan asked for, when they are not the frames' readback;
    ``plan_matrix`` the camera the plan was laid for, when it is not the scan's:
    density must take the scan's (FM-124). ``counted`` are the wells the cells
    stage has counts for (default: the wells with a frame) and ``cells_frames``
    the number of frames it says it counted (default: the scan's).
    """
    path = tmp_path / "run.json"
    in_order = list(dict.fromkeys(p[0] for p in [*frames, *failed]))
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=CENTRES["A1"],
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    laid = plan_matrix if plan_matrix is not None else matrix
    fov = _extent(laid, shape)
    plan_wells = wells if wells is not None else tuple(in_order)
    plate_plan, _ = plan_fields(
        plate,
        [w for w in plan_wells if w in CENTRES],
        fov_um=fov,
        footprint_of=lambda x, y: footprint(x, y, matrix=laid, image_shape_px=shape),
        rows=1,
        columns=1,
        overlap=0.0,
        clip_to_well=False,
    )
    targets = list(planned) if planned is not None else [*frames, *failed]
    index: dict[str, int] = {}
    fields = []
    for well, x, y in targets:
        k = index.get(well, 0)
        index[well] = k + 1
        fields.append(
            PlannedField(well=well, index=k, name=f"{well}_{k:04d}", x_um=x, y_um=y)
        )
    plan = ScanPlanResult(
        plate_plan=plate_plan,
        wells=plan_wells if listed is None else listed,
        objective=None,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=laid,
        matrix_source="parameter",
        image_shape_px=shape,
        fov_um=fov,
        overlap=0.0,
        rows=1,
        columns=1,
        fields=tuple(fields),
        estimate_s=None,
    )
    taken = []
    seen: dict[str, int] = {}
    for k, (well, x, y) in enumerate(frames):
        i = seen.get(well, 0)
        seen[well] = i + 1
        taken.append(
            ScanFrame(
                well=well,
                index=i,
                name=f"{well}_{i:04d}",
                file=f"scan/t/{well}_{i:04d}.tif",
                x_um=x,
                y_um=y,
                z_um=1000.0 + k,
                focus="locked",
                pfs_offset=2000.0 + k,
                t_s=float(k),
            )
        )
    lost = [
        FailedField(
            well=well,
            index=k,
            name=f"{well}_x{k:04d}",
            x_um=x,
            y_um=y,
            error="the stage did not reach it",
        )
        for k, (well, x, y) in enumerate(failed)
    ]
    scan = ScanResult(
        folder="scan/t",
        objective=None,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
        exposure_ms=10.0,
        frames=tuple(taken),
        failed=tuple(lost),
    )
    detected = tuple(
        Cell(
            well=well,
            x_um=x,
            y_um=y,
            label=label,
            score=0.9,
            frame=0,
            box_px=(0.0, 0.0, 1.0, 1.0),
        )
        for (well, x, y), label in cells
    )
    (tmp_path / "cells").mkdir(exist_ok=True)
    name = f"cells/{file_token}.json"
    (tmp_path / name).write_text(
        CellsFile(cells=detected).model_dump_json(), encoding="utf-8"
    )
    with_counts = counted if counted is not None else [w for w in in_order if w in seen]
    result = CellsResult(
        file=name,
        model="m.pth",
        model_sha256="0" * 64,
        classes=classes,
        device="cpu",
        score_threshold=0.5,
        merge_radius_um=5.0,
        frames=len(taken) if cells_frames is None else cells_frames,
        counts={
            well: {
                label: sum(1 for c in detected if c.well == well and c.label == label)
                for label in classes
            }
            for well in with_counts
        },
        merged=0,
        seconds_per_frame=None,
    )
    _record(path, "plate-registration", "plate", plate)
    _record(path, "scan-plan", "scan-plan", plan)
    _record(path, "scan", "scan", scan)
    _record(path, "cells", "cells", result)
    return Seed(plan, scan, detected, result)


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    on_progress: object = None,
    **params: object,
) -> StageRecord:
    return run_plugin(
        DensityPlugin(),
        microscope,
        DensityParams(**params),  # type: ignore[arg-type]
        tmp_path / "run.json",
        stop=stop,
        on_progress=on_progress,  # type: ignore[arg-type]
    )


def _result(record: StageRecord) -> DensityResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return DensityResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["density"]


def _refused(
    microscope: Microscope, tmp_path: Path, **params: object
) -> PluginRunError:
    """Run and expect a ``PluginRunError``; the stored record is ``failed`` and has no result."""
    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, **params)
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    return info.value


def _files(tmp_path: Path) -> set[str]:
    return {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}


# ------------------------------------------------------------------ oracles


def _clipped_mm2(
    centre: tuple[float, float], well_centre: tuple[float, float], w: float, h: float
) -> float:
    """A ``w`` by ``h`` µm rectangle clipped to the disc of the plate, sampled at 1 µm: the oracle of the seam.

    The disc is written out here (``x² + y² <= r²``), not asked of the code.
    """
    xs = np.arange(centre[0] - w / 2 + 0.5, centre[0] + w / 2, 1.0)
    ys = np.arange(centre[1] - h / 2 + 0.5, centre[1] + h / 2, 1.0)
    px, py = np.meshgrid(xs - well_centre[0], ys - well_centre[1])
    return float(np.count_nonzero(px * px + py * py <= RADIUS_UM**2)) / 1e6


# ------------------------------------------------------------- registration


def test_the_entry_point_loads_density() -> None:
    entry = load("density")

    assert entry.plugin.provides == "density"
    assert entry.plugin.requires == ()
    assert entry.plugin.reads == ("scan-plan", "scan", "cells")
    assert entry.plugin.reads_optional == ()
    assert entry.target == "smc.plugins.density.plugin:DensityPlugin"


# --------------------------------------------------------------- the result


def test_density_through_the_runner_gives_counts_area_and_density_per_well(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A1: four fields side by side (800 x 600 µm). B1: one field across the wall, where width and height tell apart.

    A field of 400 x 300 µm centred (3000, 1300) µm from B1's centre is 21 %
    smaller than the same field with its sides swapped once the disc has
    clipped it, so the area says which side is which.
    """
    straddling = _at("B1", 3000.0, 1300.0)
    inside = [_at("A1", dx, dy) for dx in (-200.0, 200.0) for dy in (-150.0, 150.0)]
    cells = (
        [(_at("A1", -300.0 + 7.0 * i, 100.0), "single") for i in range(5)]
        + [(_at("A1", 250.0, -100.0 + 6.0 * i), "doublet") for i in range(3)]
        + [
            (_at("B1", 2900.0, 1200.0), "single"),  # in the field and in the well
            (_at("B1", 3190.0, 1400.0), "doublet"),  # in the field, on the plastic
        ]
    )
    _seed(tmp_path, frames=[*inside, straddling], cells=cells)
    clipped = _clipped_mm2((3000.0, 1300.0), (0.0, 0.0), 400.0, 300.0)
    swapped = _clipped_mm2((3000.0, 1300.0), (0.0, 0.0), 300.0, 400.0)
    assert abs(swapped - clipped) / clipped > 0.05  # the guard: the seam is real

    result = _result(_run(demo_microscope, tmp_path))

    a1, b1 = result.wells
    assert (a1.well, b1.well) == ("A1", "B1")
    assert a1.area_mm2 == pytest.approx(0.48, abs=1e-9)
    assert a1.counts == {"single": 5, "doublet": 3, "debris": 0}
    assert a1.per_mm2 == pytest.approx(
        {"single": 5 / 0.48, "doublet": 3 / 0.48, "debris": 0.0}
    )
    assert (a1.fields, a1.failed_fields, a1.outside_well) == (4, 0, 0)
    assert b1.area_mm2 == pytest.approx(clipped, rel=0.015)
    assert b1.counts == {"single": 1, "doublet": 0, "debris": 0}
    assert b1.outside_well == 1
    assert (b1.fields, b1.failed_fields) == (1, 0)
    assert b1.per_mm2 is not None
    assert b1.per_mm2["single"] == pytest.approx(1 / clipped, rel=0.015)


def test_failed_fields_take_no_area_and_are_counted(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Four fields side by side, one failed: three frames of 0.12 mm², and the density is per imaged area."""
    row = [_at("A1", dx, 0.0) for dx in (-600.0, -200.0, 200.0, 600.0)]
    _seed(
        tmp_path,
        frames=[row[0], row[1], row[3]],
        failed=[row[2]],
        cells=[(_at("A1", -590.0, 10.0), "single"), (_at("A1", 590.0, 10.0), "single")],
    )

    (a1,) = _result(_run(demo_microscope, tmp_path)).wells

    assert (a1.fields, a1.failed_fields) == (3, 1)
    assert a1.area_mm2 == pytest.approx(0.36, abs=1e-9)
    assert a1.per_mm2 == pytest.approx(
        {"single": 2 / 0.36, "doublet": 0.0, "debris": 0.0}
    )


def test_the_readback_position_decides_the_area_not_the_plan(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Planned touching (400 µm apart), read back 300 µm apart: the union is 700 x 300 µm, not 800 x 300."""
    _seed(
        tmp_path,
        frames=[_at("A1", -150.0, 0.0), _at("A1", 150.0, 0.0)],
        planned=[_at("A1", -200.0, 0.0), _at("A1", 200.0, 0.0)],
    )

    (a1,) = _result(_run(demo_microscope, tmp_path)).wells

    assert a1.area_mm2 == pytest.approx(0.21, abs=1e-9)


def test_the_scans_camera_decides_the_area_not_the_plans(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The plan was laid for 5 µm and the scan took its frames at 4 µm: a field is 320 x 240 µm."""
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0)],
        matrix=_matrix(4.0),
        plan_matrix=UPRIGHT,
    )

    (a1,) = _result(_run(demo_microscope, tmp_path)).wells

    assert a1.area_mm2 == pytest.approx(0.0768, abs=1e-9)


def test_the_scans_frame_is_height_by_width_and_not_the_other_way_round(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Two fields 350 µm apart along X overlap by 50 µm if a frame is 400 wide, and leave a gap of 50 if it is 300."""
    _seed(tmp_path, frames=[_at("A1", -175.0, 0.0), _at("A1", 175.0, 0.0)])

    (a1,) = _result(_run(demo_microscope, tmp_path)).wells

    assert a1.area_mm2 == pytest.approx(0.225, abs=1e-9)  # 750 x 300; swapped: 0.18


def test_a_well_whose_every_field_failed_is_reported_with_unknown_density(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The plan lists B1 first: the result follows the plan, not the scan's frames."""
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0)],
        failed=[_at("B1", 0.0, 0.0), _at("B1", 400.0, 0.0)],
        cells=[(_at("A1", 10.0, 10.0), "single")],
        wells=("B1", "A1"),
    )

    b1, a1 = _result(_run(demo_microscope, tmp_path)).wells

    assert (b1.well, a1.well) == ("B1", "A1")
    assert (b1.fields, b1.failed_fields, b1.area_mm2) == (0, 2, 0.0)
    assert b1.counts == {"single": 0, "doublet": 0, "debris": 0}
    assert b1.per_mm2 is None
    assert a1.per_mm2 == pytest.approx(
        {"single": 1 / FIELD_MM2, "doublet": 0.0, "debris": 0.0}
    )


def test_a_well_the_scan_did_not_attempt_is_not_reported_and_failures_stay_in_their_well(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A1 failed one field and A2 two: neither count is the other's; B1 was never attempted."""
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0), _at("A2", 0.0, 0.0)],
        failed=[_at("A1", 400.0, 0.0), _at("A2", 400.0, 0.0), _at("A2", 800.0, 0.0)],
        wells=("A1", "A2", "B1"),
    )

    result = _result(_run(demo_microscope, tmp_path))

    assert [w.well for w in result.wells] == ["A1", "A2"]
    assert [w.failed_fields for w in result.wells] == [1, 2]
    assert [w.fields for w in result.wells] == [1, 1]


# ----------------------------------------------------------------- refusals


def _one_row(tmp_path: Path, **seed: object) -> Seed:
    """A1 with four fields side by side (an area of 0.48 mm²) and one cell, plus what ``seed`` changes."""
    options: dict[str, object] = {
        "frames": [_at("A1", dx, 0.0) for dx in (-600.0, -200.0, 200.0, 600.0)],
        "cells": [(_at("A1", 0.0, 0.0), "single")],
    }
    options.update(seed)
    return _seed(tmp_path, **options)  # type: ignore[arg-type]


def test_a_scan_with_no_frame_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, frames=[], failed=[_at("A1", 0.0, 0.0)])

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the scan took no frame (a dry run, or every field failed), so no area was "
        "imaged; run scan again"
    )


def test_a_matrix_that_cannot_be_inverted_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _one_row(tmp_path, matrix=((1.0, 2.0), (0.5, 1.0)), plan_matrix=UPRIGHT)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the camera matrix cannot be inverted (determinant 0.0); run "
        "camera-calibration, or give a pixel size"
    )


def test_cells_counted_on_another_number_of_frames_are_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-111: the scan was run again after the cells, and both records still validate."""
    _one_row(tmp_path, cells_frames=7)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the cells stage counted 7 frames and the scan holds 4; run cells again"
    )


def test_a_well_the_scan_imaged_and_the_cells_did_not_count_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0), _at("B1", 0.0, 0.0)],
        counted=["A1"],
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the scan imaged well 'B1', which the cells stage did not count; run cells again"
    )


def test_a_well_the_cells_counted_and_the_scan_has_no_frame_in_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0)],
        failed=[_at("B1", 0.0, 0.0)],
        counted=["A1", "B1"],
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the cells stage counted well 'B1', which the scan has no frame in; run "
        "cells again"
    )


def test_a_well_the_plan_does_not_list_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The scan holds B1, the plan lists A1 only: B1 would be left out of the loop, and the stage ``done``."""
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0), _at("B1", 0.0, 0.0)],
        listed=("A1",),
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the scan holds well 'B1', which the scan plan does not list among its "
        "wells; run scan-plan, scan and cells again"
    )


def test_a_well_that_only_failed_and_the_plan_does_not_list_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0)],
        failed=[_at("A2", 0.0, 0.0)],
        listed=("A1",),
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the scan holds well 'A2', which the scan plan does not list among its "
        "wells; run scan-plan, scan and cells again"
    )


def test_a_cells_file_that_no_longer_matches_says_to_run_cells_again(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _one_row(
        tmp_path,
        cells=[(_at("A1", 10.0 * i, 0.0), "single") for i in range(5)],
    )
    (tmp_path / seed.result.file).write_text(
        CellsFile(cells=seed.cells[1:]).model_dump_json(), encoding="utf-8"
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cells file cells/t1.json differs from the result at A1/single (the file "
        "holds 4, the result counted 5); run cells again"
    )


def test_a_well_that_is_not_on_the_plate_says_to_run_scan_plan_again(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0), ("Z9", 0.0, 0.0)],
        wells=("A1",),
        listed=("A1", "Z9"),
    )

    refused = _refused(demo_microscope, tmp_path)

    assert (
        refused.reason == "well 'Z9' is not on the 96-well plate; run scan-plan again"
    )


def test_a_grid_too_fine_for_the_well_names_the_well_and_says_to_raise_the_step(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1600 x 300 µm at 5 µm is 320 by 60 points: over a bound of 100 on a side."""
    _one_row(tmp_path)
    monkeypatch.setattr(algorithm, "_MAX_SIDE_POINTS", 100)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "well A1: a grid of 5 µm over its imaged area would hold 320 by 60 points, "
        "more than 100 on a side; raise area_grid_um"
    )


def test_a_frame_the_raster_refuses_names_its_well(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A readback of 1e13 µm is a corrupt record, and the refusal says which well's it is."""
    _seed(tmp_path, frames=[_at("A1", 0.0, 0.0), ("A1", 1e13, 0.0)])

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == "well A1: footprint 1 has a corner beyond 1e12 µm"


def test_a_label_the_cells_hold_that_their_classes_do_not_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The file and the counts agree on a doublet, the classes say there are only singles: no count may drop it."""
    _one_row(
        tmp_path,
        cells=[(_at("A1", 0.0, 0.0), "doublet")],
        classes=("single",),
    )
    result_path = tmp_path / "run.json"
    document = RunDocument.load(result_path)
    stored = document.stages["cells"]
    assert stored.result is not None
    stored.result["counts"] = {"A1": {"single": 0, "doublet": 1}}
    write_stage(result_path, stored)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "a cell of well 'A1' has label 'doublet', which is not among the labels "
        "['single']; run cells again"
    )


# ------------------------------------------------------ stop, the result


def test_a_stop_between_two_wells_ends_the_stage_and_writes_nothing(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, frames=[_at("A1", 0.0, 0.0), _at("A2", 0.0, 0.0)])
    before = _files(tmp_path)
    stop = threading.Event()
    seen: list[Progress] = []

    def on_progress(progress: Progress) -> None:
        seen.append(progress)
        stop.set()  # asked for during the first well, looked at before the second

    record = _run(demo_microscope, tmp_path, stop=stop, on_progress=on_progress)

    assert record.status is StageStatus.stopped
    assert record.result is None
    assert [(p.done, p.total, p.message) for p in seen] == [(1, 2, "A1")]
    assert _stored(tmp_path).status is StageStatus.stopped
    assert _files(tmp_path) == before


def test_progress_names_each_well_in_the_plans_order(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(
        tmp_path,
        frames=[_at("A1", 0.0, 0.0), _at("A2", 0.0, 0.0), _at("B1", 0.0, 0.0)],
        wells=("B1", "A1", "A2"),
    )
    seen: list[Progress] = []

    _run(demo_microscope, tmp_path, on_progress=seen.append)

    assert [(p.done, p.total, p.message) for p in seen] == [
        (1, 3, "B1"),
        (2, 3, "A1"),
        (3, 3, "A2"),
    ]


def test_the_result_names_what_it_assumed_and_round_trips(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _one_row(tmp_path)

    record = _run(demo_microscope, tmp_path, area_grid_um=2.5)
    result = _result(record)

    assert result.assumed == (
        "the imaged area is the union of the scan's fields at the positions the "
        "stage read back, counted on a grid of 2.5 µm: an edge of it is known to "
        "within half a step",
        "the wells' shapes and positions are the plate registration's and the "
        "plate's nominal sizes: the wall was not measured, so the area and the "
        "cells near it are only as exact as the registration",
    )
    assert result.labels == CLASSES
    assert result.area_grid_um == 2.5
    stored = _stored(tmp_path)
    assert stored.result is not None
    assert DensityResult.model_validate(stored.result) == result


def test_the_same_inputs_give_the_same_result(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _one_row(tmp_path)

    first = _result(_run(demo_microscope, tmp_path))
    second = _result(_run(demo_microscope, tmp_path))

    assert first.wells == second.wells
    assert math.isclose(first.wells[0].area_mm2, 0.48, abs_tol=1e-9)
