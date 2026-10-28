"""``route`` through ``run_plugin``: the three records upstream in, positions to image out (m3 §11, #194).

The plugin touches the stand's turret and nothing else (``requires = ()``): the
demo stand is here because ``run_plugin`` needs a ``Microscope``, and its turret
has the labels and the profile's pixel sizes that the objective tests use.

The upstream records are written by the test: the scan plan by calling
``plan_fields`` (so the route reads what ``scan-plan`` writes, for wells in a
non-raster order), the scan with a frame at every planned field and no TIFF (the
route reads none), and the cells as a cells file whose counts are computed from
the cells. The camera is 60 x 80 pixels (not square, FM-73) at 5 µm: a field of
400 x 300 µm in a well of 6 400 µm, so a lattice of a quarter of the smaller
side is 7 225 candidates. Every expected value is computed here from the seed
and from ``footprint()``, never read from the result's own fields (FM-124).
"""

from __future__ import annotations

import itertools
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.route import algorithm
from smc.plugins.route import plugin as plugin_module
from smc.plugins.route.plugin import RouteParams, RoutePlugin
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    Cell,
    CellsFile,
    CellsResult,
    FailedField,
    PlateResult,
    RoutePosition,
    RouteResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
    TimingsResult,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

pytestmark = pytest.mark.demo

SHAPE = (60, 80)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 5.0
FOV = (SHAPE[1] * PIXEL_UM, SHAPE[0] * PIXEL_UM)  # (400, 300): width, height
STEP_UM = 0.25 * min(FOV)  # 75: a quarter of the smaller side
IN_PLACE = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
TWENTY = "Nikon 20X Plan Fluor ELWD"  # no pixel size in the profile
#: The labels of the demo's turret, in position order.
LABELS = (
    "Nikon 40X Plan Fluor ELWD, Nikon 10X S Fluor, Objective-2, "
    "Nikon 20X Plan Fluor ELWD, Objective-4, Objective-5"
)
CLASSES = ("single", "doublet", "debris")
WELLS = ("A1", "A2", "B2", "B1")  # a non-raster order: the scan's
#: Where the 96-well plate's wells are, with A1 at (1000, -500): 9 mm apart, 6.4 mm wide.
CENTRES = {
    "A1": (1000.0, -500.0),
    "A2": (10000.0, -500.0),
    "B2": (10000.0, -9500.0),
    "B1": (1000.0, -9500.0),
}
RADIUS_UM = 3200.0

#: ``(well, dx, dy, label)``, the offset from the well's centre. Clusters are a few
#: µm across and over a field apart, so each is held whole by one field and the
#: counts are the clusters' sizes. Decimals keep a cell off the edge of any field.
LAYOUT: list[tuple[str, float, float, str]] = (
    [("A1", -1500.37 + 7.3 * i, 1200.11, "single") for i in range(5)]
    + [("A1", 1400.21, -900.4 + 6.1 * i, "doublet") for i in range(3)]
    + [("A1", 2.9 * i + 0.7, 2200.8, "single") for i in range(2)]
    + [("A1", -1000.5, -2000.3, "single")]
    + [("A2", 900.3 + 5.5 * i, 800.7, "single") for i in range(4)]
    + [("A2", 20.3 + 4.1 * i, -15.2, "single") for i in range(3)]
    + [("A2", -1500.6, -1500.2 + 3.3 * i, "debris") for i in range(2)]
    + [("B2", -2000.1 + 5.2 * i, -1000.9, "single") for i in range(6)]
    + [("B2", 2000.2 + 5.2 * i, 1500.3, "single") for i in range(2)]
    + [("B1", 0.4 + 3.1 * i, 1800.6, "doublet") for i in range(2)]
    + [("B1", -2500.2, 0.7, "single")]
)


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


UPRIGHT = _matrix(PIXEL_UM, 0.0, False)


@dataclass(frozen=True)
class Seed:
    """What the three upstream records say, kept for the expected values."""

    plan: ScanPlanResult
    scan: ScanResult
    cells: tuple[Cell, ...]
    result: CellsResult
    matrix: Matrix2
    shape: tuple[int, int]


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


def _cells_of(layout: Sequence[tuple[str, float, float, str]]) -> tuple[Cell, ...]:
    return tuple(
        Cell(
            well=well,
            x_um=CENTRES[well][0] + dx,
            y_um=CENTRES[well][1] + dy,
            label=label,
            score=0.9,
            frame=0,
            box_px=(0.0, 0.0, 1.0, 1.0),
        )
        for well, dx, dy, label in layout
    )


def _seed(
    tmp_path: Path,
    *,
    layout: Sequence[tuple[str, float, float, str]] = LAYOUT,
    wells: tuple[str, ...] = WELLS,
    listed: tuple[str, ...] | None = None,
    scanned: tuple[str, ...] = WELLS,
    matrix: Matrix2 = UPRIGHT,
    plan_matrix: Matrix2 | None = None,
    plan_shape: tuple[int, int] | None = None,
    plan_objective: str | None = "the scan's",
    shape: tuple[int, int] = SHAPE,
    scan_objective: str | None = IN_PLACE,
    failed_wells: tuple[str, ...] = (),
    file_token: str = "t1",
) -> Seed:
    """The plan, the scan and the cells of ``wells``, in ``tmp_path/run.json``.

    ``listed`` is what the plan's own ``wells`` says when it is not ``wells`` (a record
    that disagrees with its fields). ``plan_matrix``, ``plan_shape`` and
    ``plan_objective`` are the camera, the frame and the objective the plan was laid
    for when they are not the scan's: the route must take the scan's (FM-124), so a
    test that wants to know says them different. Frames are at the planned fields,
    their readback a little off the target and their Z and autofocus offset
    distinct; the wells in ``failed_wells`` have no frame, only failed fields.
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
    laid = plan_matrix if plan_matrix is not None else matrix
    laid_shape = plan_shape if plan_shape is not None else shape
    fov = _extent(laid, laid_shape)
    plate_plan, fields = plan_fields(
        plate,
        wells,
        fov_um=fov,
        footprint_of=lambda x, y: footprint(
            x, y, matrix=laid, image_shape_px=laid_shape
        ),
        rows=2,
        columns=2,
        overlap=0.1,
        clip_to_well=True,
    )
    plan = ScanPlanResult(
        plate_plan=plate_plan,
        wells=wells if listed is None else listed,
        objective=scan_objective if plan_objective == "the scan's" else plan_objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=laid,
        matrix_source="parameter",
        image_shape_px=laid_shape,
        fov_um=fov,
        overlap=0.1,
        rows=2,
        columns=2,
        fields=fields,
        estimate_s=None,
    )
    frames: list[ScanFrame] = []
    failed: list[FailedField] = []
    for k, field in enumerate(fields):
        if field.well in failed_wells:
            failed.append(
                FailedField(
                    well=field.well,
                    index=field.index,
                    name=field.name,
                    x_um=field.x_um,
                    y_um=field.y_um,
                    error="the stage did not reach it",
                )
            )
            continue
        frames.append(
            ScanFrame(
                well=field.well,
                index=field.index,
                name=field.name,
                file=f"scan/t/{field.name}.tif",
                x_um=field.x_um + 0.3,
                y_um=field.y_um - 0.2,
                z_um=1000.0 + 10.0 * k,
                focus="locked",
                pfs_offset=2000.0 + 10.0 * k,
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
        failed=tuple(failed),
    )
    cells = _cells_of(layout)
    folder = tmp_path / "cells"
    folder.mkdir(exist_ok=True)
    name = f"cells/{file_token}.json"
    (tmp_path / name).write_text(
        CellsFile(cells=cells).model_dump_json(), encoding="utf-8"
    )
    counts = {
        well: {
            label: sum(1 for c in cells if c.well == well and c.label == label)
            for label in CLASSES
        }
        for well in scanned
    }
    result = CellsResult(
        file=name,
        model="m.pth",
        model_sha256="0" * 64,
        classes=CLASSES,
        device="cpu",
        score_threshold=0.5,
        merge_radius_um=5.0,
        frames=len(frames),
        counts=counts,
        merged=0,
        seconds_per_frame=None,
    )
    _record(path, "plate-registration", "plate", plate)
    _record(path, "scan-plan", "scan-plan", plan)
    _record(path, "scan", "scan", scan)
    _record(path, "cells", "cells", result)
    return Seed(plan, scan, cells, result, matrix, shape)


def _extent(matrix: Matrix2, shape: tuple[int, int]) -> tuple[float, float]:
    """The oracle's field of view: the extent of ``footprint()`` corners at the origin."""
    corners = footprint(0.0, 0.0, matrix=matrix, image_shape_px=shape)
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _timings(tmp_path: Path) -> TimingsResult:
    timings = TimingsResult(
        move_overhead_ms=100.0,
        move_ms_per_mm=50.0,
        refocus_ms=400.0,
        snap_ms=20.0,
        samples=5,
    )
    _record(tmp_path / "run.json", "timing", "timings", timings)
    return timings


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    on_progress: object = None,
    **params: object,
) -> StageRecord:
    return run_plugin(
        RoutePlugin(),
        microscope,
        RouteParams(**params),  # type: ignore[arg-type]
        tmp_path / "run.json",
        stop=stop,
        on_progress=on_progress,  # type: ignore[arg-type]
    )


def _result(record: StageRecord) -> RouteResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return RouteResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["route"]


def _refused(
    microscope: Microscope, tmp_path: Path, **params: object
) -> PluginRunError:
    """Run and expect a ``PluginRunError``; the stored record is ``failed`` and has no result."""
    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, **params)
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    return info.value


def _without(
    microscope: Microscope, monkeypatch: pytest.MonkeyPatch, capability: type
) -> None:
    """The stand as if no device filled this capability's role."""
    real_get = microscope.get
    monkeypatch.setattr(
        microscope,
        "get",
        lambda wanted: None if wanted is capability else real_get(wanted),
    )


# ------------------------------------------------------------------- oracles


def _polygon(
    seed: Seed, position: RoutePosition, matrix: Matrix2 | None = None
) -> list[tuple[float, float]]:
    return list(
        footprint(
            position.x_um,
            position.y_um,
            matrix=matrix if matrix is not None else seed.matrix,
            image_shape_px=seed.shape,
        )
    )


def _inside(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """All four cross products of one sign, whichever way the corners turn."""
    crosses = [
        (polygon[(k + 1) % 4][0] - polygon[k][0]) * (point[1] - polygon[k][1])
        - (polygon[(k + 1) % 4][1] - polygon[k][1]) * (point[0] - polygon[k][0])
        for k in range(4)
    ]
    return all(c >= 0.0 for c in crosses) or all(c <= 0.0 for c in crosses)


def _distance_to_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> float:
    """0 inside; else the smallest distance to 400 points along each edge (an over-estimate by under a µm)."""
    if _inside(point, polygon):
        return 0.0
    best = math.inf
    for k in range(4):
        (ax, ay), (bx, by) = polygon[k], polygon[(k + 1) % 4]
        for step in range(401):
            t = step / 400
            best = min(best, math.dist(point, (ax + t * (bx - ax), ay + t * (by - ay))))
    return best


def _overlap_has_area(
    a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]
) -> bool:
    """Two convex polygons share area unless an edge normal separates them (touching is not an overlap)."""
    for polygon in (a, b):
        for k in range(4):
            (px, py), (qx, qy) = polygon[k], polygon[(k + 1) % 4]
            nx, ny = qy - py, px - qx
            norm = math.hypot(nx, ny)
            proj_a = [(x * nx + y * ny) / norm for x, y in a]
            proj_b = [(x * nx + y * ny) / norm for x, y in b]
            if max(proj_a) <= min(proj_b) + 1e-4 or max(proj_b) <= min(proj_a) + 1e-4:
                return False
    return True


def _positions_of(result: RouteResult, well: str) -> list[RoutePosition]:
    return [p for p in result.positions if p.well == well]


def _singles(well: str) -> int:
    return sum(1 for w, _, _, label in LAYOUT if w == well and label == "single")


# ------------------------------------------------------------- registration


def test_the_entry_point_loads_route() -> None:
    entry = load("route")

    assert entry.plugin.provides == "route"
    assert entry.plugin.requires == ()
    assert entry.plugin.reads == ("scan-plan", "scan", "cells")
    assert entry.plugin.reads_optional == ("timings",)
    assert entry.target == "smc.plugins.route.plugin:RoutePlugin"


def test_the_seeds_wells_are_where_the_layout_puts_them(tmp_path: Path) -> None:
    """The clusters of ``LAYOUT`` are placed from ``CENTRES``: the 2 x 2 grid of each well is centred on it."""
    seed = _seed(tmp_path)

    assert seed.plan.fields[0].well == "A1"
    for well, centre in CENTRES.items():
        planned = [f for f in seed.plan.fields if f.well == well]
        assert (
            sum(f.x_um for f in planned) / len(planned),
            sum(f.y_um for f in planned) / len(planned),
        ) == pytest.approx(centre)


# --------------------------------------------------------------- the result


def test_wells_follow_the_scan_order(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _seed(tmp_path)

    result = _result(_run(demo_microscope, tmp_path))

    assert [p.order for p in result.positions] == list(range(len(result.positions)))
    wells_in_route = [w for w, _ in _groups(result)]
    assert wells_in_route == list(WELLS)
    # Names count along the well's route from 0.
    for well in WELLS:
        assert [p.name for p in _positions_of(result, well)] == [
            f"{well}_r{k:03d}" for k in range(len(_positions_of(result, well)))
        ]
    # The first well starts at its field nearest the first planned field of that well...
    reference = (seed.plan.fields[0].x_um, seed.plan.fields[0].y_um)
    first = _positions_of(result, "A1")
    assert _distance(first[0], reference) == min(_distance(p, reference) for p in first)
    # ...and each later one at its field nearest the previous well's last position.
    for before, well in itertools.pairwise(WELLS):
        last = _positions_of(result, before)[-1]
        mine = _positions_of(result, well)
        assert _distance(mine[0], (last.x_um, last.y_um)) == min(
            _distance(p, (last.x_um, last.y_um)) for p in mine
        )


def _groups(result: RouteResult) -> list[tuple[str, int]]:
    """The wells of the route in order, each once, with how many stops it has; a well split in two would repeat."""
    groups: list[tuple[str, int]] = []
    for p in result.positions:
        if groups and groups[-1][0] == p.well:
            groups[-1] = (p.well, groups[-1][1] + 1)
        else:
            groups.append((p.well, 1))
    return groups


def _distance(position: RoutePosition, point: tuple[float, float]) -> float:
    return math.dist((position.x_um, position.y_um), point)


def test_the_first_well_starts_at_the_field_nearest_the_first_planned_field_not_the_wells_centre(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Two fields on either side of A1's centre: the sparser one is nearer the first planned field, the denser one the centre.

    The first planned field is (-180, +135) µm from the centre. Of the two fields
    below, the north-west one (-900, +675) is 900 µm from it and 1 125 µm from the
    centre, the south-east one (+825, -525) 1 202 and 978 µm: the two references
    pick different fields, by a margin far above the lattice's 75 µm. A cluster
    is put at the bottom-right of the field it should be held by, because the
    greedy takes the lowest-numbered candidate that holds it (the top-left-most),
    so the field is exactly that one.
    """
    layout = [("A1", -710.0 + 0.37 * i, 535.11, "single") for i in range(2)] + [
        ("A1", 1015.2 + 0.41 * i, -665.5, "single") for i in range(3)
    ]
    seed = _seed(tmp_path, layout=layout, wells=("A1",), scanned=("A1",))

    result = _result(_run(demo_microscope, tmp_path))

    centre = CENTRES["A1"]
    north_west = (centre[0] - 900.0, centre[1] + 675.0)
    south_east = (centre[0] + 825.0, centre[1] - 525.0)
    planned = (seed.plan.fields[0].x_um, seed.plan.fields[0].y_um)
    assert math.dist(north_west, planned) < math.dist(south_east, planned)
    assert math.dist(south_east, centre) < math.dist(north_west, centre)
    stops = _positions_of(result, "A1")
    assert [(p.x_um, p.y_um, p.counts) for p in stops] == [
        (*north_west, {"single": 2}),
        (*south_east, {"single": 3}),
    ]


def test_each_later_well_starts_at_the_field_nearest_the_previous_wells_last_stop_not_its_first(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A2 is entered from the west and left on the east side, and B2 has a field on each side.

    A1 is west of A2, so A2 starts at its west cluster and ends at its east one.
    The B2 field nearest A2's last stop is B2's east cluster (3 cells), the one
    nearest its first stop the west cluster (1 cell): a rule that took the previous
    well's first stop would start B2 at the other.
    """
    wells = ("A1", "A2", "B2")
    layout = (
        [("A1", -1500.3, 700.7, "single")]
        + [("A2", -2000.2 + 5.1 * i, 0.9, "single") for i in range(2)]
        + [("A2", 2000.4 + 5.1 * i, 0.6, "single") for i in range(2)]
        + [("B2", -2000.1, 0.8, "single")]
        + [("B2", 2000.3 + 5.1 * i, 0.7, "single") for i in range(3)]
    )
    _seed(tmp_path, layout=layout, wells=wells, scanned=wells)

    result = _result(_run(demo_microscope, tmp_path))

    a2, b2 = _positions_of(result, "A2"), _positions_of(result, "B2")
    nearest_to_last = min(b2, key=lambda p: _distance(p, (a2[-1].x_um, a2[-1].y_um)))
    nearest_to_first = min(b2, key=lambda p: _distance(p, (a2[0].x_um, a2[0].y_um)))
    assert nearest_to_last != nearest_to_first  # the test's own design
    assert [(p.counts["single"]) for p in a2] == [2, 2]
    assert a2[0].x_um < a2[-1].x_um  # in from the west, out on the east
    assert b2[0] == nearest_to_last
    assert [p.counts["single"] for p in b2] == [3, 1]


def test_every_cluster_is_held_whole_by_one_field_and_counted_once(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _seed(tmp_path)

    result = _result(_run(demo_microscope, tmp_path))

    # Singles of the four wells in clusters of 5, 2, 1 | 4, 3 | 6, 2 | 1 (A1, A2, B2, B1).
    assert {
        well: sorted(
            (p.counts["single"] for p in _positions_of(result, well)), reverse=True
        )
        for well in WELLS
    } == {"A1": [5, 2, 1], "A2": [4, 3], "B2": [6, 2], "B1": [1]}
    assert result.labels == ("single",)
    assert result.uncovered == {"A1": 0, "A2": 0, "B2": 0, "B1": 0}
    # Each position's count is what its footprint holds, and no cell is in two footprints.
    for p in result.positions:
        inside = [
            c
            for c in seed.cells
            if c.well == p.well
            and c.label == "single"
            and _inside((c.x_um, c.y_um), _polygon(seed, p))
        ]
        assert p.counts == {"single": len(inside)}


def test_fields_lie_on_the_lattice_of_a_quarter_of_the_smaller_field_side(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-124: the step is derived from the scan's matrix and shape (the smaller of 400 and 300 µm), not from the plan's or from the larger side."""
    _seed(tmp_path)

    def offsets_in(step: float, **params: object) -> list[float]:
        result = _result(_run(demo_microscope, tmp_path, **params))
        return [(p.x_um - CENTRES[p.well][0]) / step for p in result.positions] + [
            (p.y_um - CENTRES[p.well][1]) / step for p in result.positions
        ]

    fine = offsets_in(STEP_UM)
    coarse = offsets_in(STEP_UM, candidate_step_fraction=0.5)

    for value in fine + coarse:
        assert value == pytest.approx(round(value), abs=1e-9)
    # The fraction changes the multiple: in units of 75 µm the half-field lattice
    # (150 µm) is on even numbers only, and the quarter-field one is not.
    assert all(round(v) % 2 == 0 for v in coarse)
    assert any(round(v) % 2 == 1 for v in fine)


def test_the_scans_matrix_and_frame_decide_the_field(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-124: the field is the scan's, whatever the plan was laid with, and a turned, mirrored camera is a turned, mirrored footprint."""
    upright = _seed(tmp_path)
    result = _result(_run(demo_microscope, tmp_path))
    assert result.fov_um == pytest.approx((SHAPE[1] * PIXEL_UM, SHAPE[0] * PIXEL_UM))
    assert result.matrix_um_per_px == upright.matrix
    assert result.image_shape_px == SHAPE

    turned = _matrix(PIXEL_UM, 20.0, True)
    seed = _seed(
        tmp_path,
        matrix=turned,
        plan_matrix=UPRIGHT,
        plan_shape=(50, 70),
        plan_objective=FORTY,
    )
    result = _result(_run(demo_microscope, tmp_path))

    # The plan was laid for another camera, frame and objective than the scan's.
    assert seed.plan.matrix_um_per_px == UPRIGHT
    assert seed.plan.image_shape_px == (50, 70)
    assert seed.plan.objective == FORTY
    assert result.matrix_um_per_px == turned
    assert result.image_shape_px == SHAPE
    assert result.objective == IN_PLACE
    assert result.fov_um == pytest.approx(_extent(turned, SHAPE))
    polygons = [(p, _polygon(seed, p)) for p in result.positions]
    for (_, a), (_, b) in itertools.combinations(polygons, 2):
        assert not _overlap_has_area(a, b)
    for p, polygon in polygons:
        held = [
            c
            for c in seed.cells
            if c.well == p.well
            and c.label == "single"
            and _inside((c.x_um, c.y_um), polygon)
        ]
        assert p.counts == {"single": len(held)}
    assert sum(p.counts["single"] for p in result.positions) + sum(
        result.uncovered.values()
    ) == sum(_singles(w) for w in WELLS)


def test_labels_choose_which_cells_count(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    doublets = _result(_run(demo_microscope, tmp_path, labels=["doublet"]))

    assert doublets.labels == ("doublet",)
    # Three doublets in A1 and two in B1; A2 and B2 have none, and are keys all the same.
    assert [(p.well, p.counts) for p in doublets.positions] == [
        ("A1", {"doublet": 3}),
        ("B1", {"doublet": 2}),
    ]
    assert doublets.uncovered == {"A1": 0, "A2": 0, "B2": 0, "B1": 0}

    both = _result(_run(demo_microscope, tmp_path, labels=["single", "doublet"]))

    assert both.labels == ("single", "doublet")
    assert all(set(p.counts) == {"single", "doublet"} for p in both.positions)
    for well in WELLS:
        wanted = sum(1 for w, _, _, label in LAYOUT if w == well and label != "debris")
        held = sum(sum(p.counts.values()) for p in _positions_of(both, well))
        assert held + both.uncovered[well] == wanted


def test_fields_per_well_and_min_cells_limit_the_fields(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    two = _result(_run(demo_microscope, tmp_path, fields_per_well=2))

    assert {w: len(_positions_of(two, w)) for w in WELLS} == {
        "A1": 2,
        "A2": 2,
        "B2": 2,
        "B1": 1,
    }
    # The two densest singles' clusters of A1 are 5 and 2: one is left, and said so.
    assert sorted(
        (p.counts["single"] for p in _positions_of(two, "A1")), reverse=True
    ) == [5, 2]
    assert two.uncovered == {"A1": 1, "A2": 0, "B2": 0, "B1": 0}

    dense = _result(_run(demo_microscope, tmp_path, min_cells=3))

    assert all(p.counts["single"] >= 3 for p in dense.positions)
    # Clusters of 5 | 4, 3 | 6 hold 3 or more; B1's single and the small clusters do not.
    assert {w: len(_positions_of(dense, w)) for w in WELLS} == {
        "A1": 1,
        "A2": 2,
        "B2": 1,
        "B1": 0,
    }
    assert dense.uncovered == {"A1": 3, "A2": 0, "B2": 2, "B1": 1}


def test_wall_margin_and_centre_exclusion_keep_fields_away(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _seed(tmp_path)

    result = _result(
        _run(demo_microscope, tmp_path, wall_margin_um=700.0, exclude_centre_um=250.0)
    )

    assert result.positions
    for p in result.positions:
        centre = CENTRES[p.well]
        polygon = _polygon(seed, p)
        assert all(math.dist(corner, centre) <= RADIUS_UM - 700.0 for corner in polygon)
        assert _distance_to_polygon(centre, polygon) >= 250.0 - 1.0
    # The three singles within 30 µm of A2's centre cannot be held by a field 250 µm from it;
    # the one at 2 500 µm from B1's centre needs a corner past the shrunk wall at 2 500 µm.
    assert result.uncovered["A2"] == 3
    assert result.uncovered["B1"] == 1
    assert _positions_of(result, "B1") == []


# ---------------------------------------------------------- another objective


def test_another_objective_scales_the_field(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The scan's pixel size is 0.2 µm at the 40X and the profile's is 0.1625: only the profile's ratio, 0.65 / 0.1625, scales it."""
    shape = (240, 320)
    scan_matrix = _matrix(0.2, 12.0, True)
    # The plan was laid for the 10X with another pixel size: the scaling is of the scan's.
    seed = _seed(
        tmp_path,
        matrix=scan_matrix,
        shape=shape,
        scan_objective=FORTY,
        plan_matrix=_matrix(0.3, 0.0, False),
        plan_objective=IN_PLACE,
    )

    result = _result(_run(demo_microscope, tmp_path, objective=IN_PLACE))

    scan_fov = _extent(scan_matrix, shape)
    assert result.objective == IN_PLACE
    assert result.fov_um == pytest.approx((4 * scan_fov[0], 4 * scan_fov[1]))
    assert [v for row in result.matrix_um_per_px for v in row] == pytest.approx(
        [4.0 * v for row in scan_matrix for v in row]
    )
    assert result.positions
    assert all(p.z_um is None and p.pfs_offset is None for p in result.positions)
    assert result.assumed == (
        "the route minimises the Euclidean distance between fields; whether the "
        "stage's move time follows it is not measured",
        f"the field of {IN_PLACE} is the scan's scaled by the profile's pixel sizes",
        "no offset between the two objectives is applied",
        f"z_um and pfs_offset are unknown for {IN_PLACE}: the scan's were measured with {FORTY}",
    )
    # The scan's frames did measure a Z: the scan itself is not changed by this.
    assert all(f.z_um is not None for f in seed.scan.frames)


def test_the_scans_own_objective_given_by_name_changes_nothing_and_needs_no_turret(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path, scan_objective=FORTY)
    _without(demo_microscope, monkeypatch, ObjectiveTurret)

    result = _result(_run(demo_microscope, tmp_path, objective=FORTY))

    assert result.objective == FORTY
    assert result.matrix_um_per_px == seed.matrix
    assert result.fov_um == pytest.approx(FOV)
    assert all(
        p.z_um is not None and p.pfs_offset is not None for p in result.positions
    )
    assert result.assumed == (
        "the route minimises the Euclidean distance between fields; whether the "
        "stage's move time follows it is not measured",
    )


def test_an_objective_that_cannot_be_scaled_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path, scan_objective=FORTY)

    assert _refused(demo_microscope, tmp_path, objective="Nikon 60X").reason == (
        f"objective 'Nikon 60X' is not on the turret: {LABELS}; give one of them, "
        f"or leave objective unset"
    )
    # An empty position has the label "", which is no objective that can be asked for.
    assert (
        "objective '' is not on the turret"
        in _refused(demo_microscope, tmp_path, objective="").reason
    )
    # The profile says 0.0 for the 20X: unknown, which is not a size.
    assert _refused(demo_microscope, tmp_path, objective=TWENTY).reason == (
        f"the profile has no pixel size for {TWENTY!r}: add it to the profile's "
        f"[camera] pixel_size_um, or leave objective unset"
    )
    turret = demo_microscope.require(ObjectiveTurret)
    monkeypatch.setattr(
        turret, "pixel_size_um", lambda label=None: 0.0 if label == FORTY else 0.65
    )
    assert _refused(demo_microscope, tmp_path, objective=IN_PLACE).reason == (
        f"the profile has no pixel size for the scan's objective {FORTY!r}: add "
        f"it to the profile's [camera] pixel_size_um, or leave objective unset"
    )
    monkeypatch.undo()

    _without(demo_microscope, monkeypatch, ObjectiveTurret)
    assert _refused(demo_microscope, tmp_path, objective=IN_PLACE).reason == (
        f"the stand has no turret, so the size of {IN_PLACE!r} is unknown; leave "
        f"objective unset to image with the scan's objective"
    )
    monkeypatch.undo()

    _seed(tmp_path, scan_objective=None)
    assert _refused(demo_microscope, tmp_path, objective=IN_PLACE).reason == (
        "the scan did not record its objective, so the size of another one cannot "
        "be scaled from it; leave objective unset"
    )


# ------------------------------------------------------- travel and z, pfs


def test_travel_time_is_none_without_timings(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    assert _result(_run(demo_microscope, tmp_path)).travel_s is None


def test_travel_time_sums_the_move_model_over_every_leg(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """100 ms and 50 ms/mm, over the legs of the route, the ones between wells (9 mm and more) included."""
    _seed(tmp_path)
    _timings(tmp_path)

    result = _result(_run(demo_microscope, tmp_path))

    points = [(p.x_um, p.y_um) for p in result.positions]
    legs = [math.dist(a, b) for a, b in itertools.pairwise(points)]
    assert max(legs) > 8000.0  # a leg between two wells
    assert result.distance_um == pytest.approx(sum(legs))
    assert result.travel_s == pytest.approx(
        sum((100.0 + 50.0 * d / 1000.0) / 1000.0 for d in legs if d > 0.0)
    )


def test_z_and_pfs_come_from_the_nearest_taken_frame(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _seed(tmp_path, failed_wells=("B1",))

    result = _result(_run(demo_microscope, tmp_path))

    assert _positions_of(result, "B1")  # a well that has fields and no frame
    for p in result.positions:
        frames = [f for f in seed.scan.frames if f.well == p.well]
        if not frames:
            assert (p.z_um, p.pfs_offset) == (None, None)
            continue
        nearest = min(
            frames, key=lambda f: math.dist((f.x_um, f.y_um), (p.x_um, p.y_um))
        )
        assert (p.z_um, p.pfs_offset) == (nearest.z_um, nearest.pfs_offset)
    # More than one frame supplies a Z: it is not the well's first or last frame's.
    assert len({p.z_um for p in _positions_of(result, "A1")}) > 1


# --------------------------------------------------------------- uncovered


def test_a_well_the_cells_stage_did_not_scan_has_no_stop_and_no_key(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    unscanned = ("A1", "A2", "B2")
    _seed(
        tmp_path,
        layout=[row for row in LAYOUT if row[0] in unscanned],
        scanned=unscanned,
    )

    result = _result(_run(demo_microscope, tmp_path))

    assert list(result.uncovered) == ["A1", "A2", "B2"]
    assert _positions_of(result, "B1") == []


# ----------------------------------------------------------------- refusals


def test_no_route_is_a_refusal_with_its_reason(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, layout=[row for row in LAYOUT if row[3] != "single"])

    nothing = _refused(demo_microscope, tmp_path)

    assert nothing.reason == (
        "the cells stage found no single cell: there is nothing to route; choose "
        "other labels, or check the scan and the cells"
    )
    _seed(tmp_path)
    assert _refused(demo_microscope, tmp_path, min_cells=50).reason == (
        "no field of 400 by 300 µm holds 50 or more single cell and keeps to the "
        "wells' margins; lower min_cells, wall_margin_um or exclude_centre_um, or "
        "check the scan's pixel size"
    )
    assert _refused(
        demo_microscope, tmp_path, labels=["single", "doublet"], min_cells=50
    ).reason == (
        "no field of 400 by 300 µm holds 50 or more single or doublet cell and "
        "keeps to the wells' margins; lower min_cells, wall_margin_um or "
        "exclude_centre_um, or check the scan's pixel size"
    )


def test_a_label_the_cells_do_not_have_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    refused = _refused(demo_microscope, tmp_path, labels=["single", "mitotic"])

    assert refused.reason == (
        "label 'mitotic' is not one of the cells' classes ['single', 'doublet', "
        "'debris']; give labels among them"
    )


def test_a_cells_file_that_no_longer_matches_says_to_run_cells_again(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _seed(tmp_path)
    (tmp_path / seed.result.file).write_text(
        CellsFile(cells=seed.cells[1:]).model_dump_json(), encoding="utf-8"
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cells file cells/t1.json differs from the result at A1/single (the file "
        "holds 7, the result counted 8); run cells again"  # 5 + 2 + 1 singles in A1
    )


def test_cells_of_a_well_the_plan_has_no_field_for_are_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, wells=("A1", "A2", "B2"), scanned=WELLS)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the cells hold well 'B1', which the scan plan has no field for; run "
        "scan-plan, scan and cells again"
    )


def test_cells_of_a_well_the_plan_does_not_list_are_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The plan has fields for B2 and does not list it among its wells: its cells would be left out, and the stage ``done``."""
    _seed(tmp_path, listed=("A1", "A2", "B1"))

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the cells hold well 'B2', which the scan plan does not list among its "
        "wells; run scan-plan, scan and cells again"
    )


def test_a_matrix_that_cannot_be_inverted_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, matrix=((1.0, 2.0), (0.5, 1.0)), plan_matrix=UPRIGHT)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the camera matrix cannot be inverted (determinant 0.0); run "
        "camera-calibration, or give a pixel size"
    )


def test_a_lattice_too_fine_for_the_well_names_the_field_size(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-121: a scan matrix 1000 times too small would be a million candidates per well."""
    _seed(tmp_path, matrix=_matrix(PIXEL_UM / 1000.0, 0.0, False), plan_matrix=UPRIGHT)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the candidate lattice of A1 would hold more than 250000 fields at a step "
        "of 0.075 µm; the field is 0.4 by 0.3 µm: check the scan's pixel size, or "
        "raise candidate_step_fraction"
    )


def test_too_many_tests_are_refused_with_a_way_out(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1's eight singles, each against the lattice points around it: 8 x 56 tests, over a bound of 10."""
    _seed(tmp_path)
    monkeypatch.setattr(algorithm, "_MAX_TESTS", 10)
    per_cell = (FOV[0] // STEP_UM + 3) * (FOV[1] // STEP_UM + 3)  # 8 x 7 points

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        f"{_singles('A1')} cells × {per_cell:.0f} lattice points per cell would be "  # noqa: RUF001
        f"tested, over 10; raise candidate_step_fraction, or choose fewer labels"
    )


def test_a_well_that_is_not_on_the_plate_says_to_run_scan_plan_again(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)

    def not_on_the_plate(plan: object, well: str) -> object:
        raise ValueError(f"well {well!r} is not on the 96-well plate")

    monkeypatch.setattr(plugin_module, "well_geometry", not_on_the_plate)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "well 'A1' is not on the 96-well plate; run scan-plan again"
    )


def test_empty_turret_positions_are_not_listed_among_the_objectives(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path, scan_objective=FORTY)
    turret = demo_microscope.require(ObjectiveTurret)
    monkeypatch.setattr(turret, "labels", lambda: ["", IN_PLACE, ""])

    refused = _refused(demo_microscope, tmp_path, objective="Nikon 60X")

    assert refused.reason == (
        f"objective 'Nikon 60X' is not on the turret: {IN_PLACE}; give one of "
        f"them, or leave objective unset"
    )


# ------------------------------------------------------ stop, determinism


def test_a_stop_ends_the_stage_between_wells(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    stop = threading.Event()
    seen: list[Progress] = []

    def on_progress(progress: Progress) -> None:
        seen.append(progress)
        stop.set()  # asked for during the first well, looked at before the second

    record = _run(demo_microscope, tmp_path, stop=stop, on_progress=on_progress)

    assert record.status is StageStatus.stopped
    assert record.result is None
    assert [(p.done, p.total, p.message) for p in seen] == [(1, 4, "A1")]
    assert _stored(tmp_path).status is StageStatus.stopped


def test_progress_names_each_well_in_the_scans_order(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    seen: list[Progress] = []

    _run(demo_microscope, tmp_path, on_progress=seen.append)

    assert [(p.done, p.total, p.message) for p in seen] == [
        (1, 4, "A1"),
        (2, 4, "A2"),
        (3, 4, "B2"),
        (4, 4, "B1"),
    ]


def test_the_same_inputs_give_the_same_route(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    first = _result(_run(demo_microscope, tmp_path))
    second = _result(_run(demo_microscope, tmp_path))

    assert first.positions == second.positions
    assert first.distance_um == second.distance_um
