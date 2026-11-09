"""``nis-export`` reads the prototype's list as it reads the route's (#343, m3 §11 and §14).

The prototype writes a copy of the run document in which a record named
``route-prototype`` fills the ``route`` slot, and the acceptance of the issue is that
``smc run nis-export`` exports that list unchanged. The plugin needs a ``Microscope``
only because ``run_plugin`` does (it reads the ``route`` record and nothing else), so the
demo stand is here for the runner. Every expected value is computed from the seed and
from ``footprint``, never from the result's own fields (FM-124).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from select_fields_seed import (
    Seed,
    cell_at,
    load_script,
    seed_run,
)
from select_fields_seed import record as write_record
from smc.hardware import Microscope
from smc.plugins import run_plugin
from smc.plugins.nis_export.format import decode
from smc.plugins.nis_export.plugin import NisExportParams, NisExportPlugin
from smc.plugins.slots import Cell, NisExportResult, RouteResult, ScanResult
from smc.run import RunDocument, StageStatus

pytestmark = pytest.mark.demo


@pytest.fixture(scope="module")
def sf() -> ModuleType:
    return load_script()


def _cells(scan: ScanResult) -> list[Cell]:
    """Two single cells in every frame, so that every frame is worth choosing."""
    return [
        Cell(
            well=frame.well,
            x_um=frame.x_um + dx,
            y_um=frame.y_um,
            label="single",
            score=0.9,
            frame=k,
            box_px=(0.0, 0.0, 10.0, 10.0),
        )
        for k, frame in enumerate(scan.frames)
        for dx in (-20.0, 20.0)
    ]


def _seed(tmp_path: Path) -> Seed:
    # Two wells of three fields that touch (overlap 0), so that six stops are chosen.
    return seed_run(
        tmp_path,
        cells=_cells,
        wells=("A1", "A2"),
        rows=1,
        columns=3,
        overlap=0.0,
    )


def _export(microscope: Microscope, copy: Path) -> NisExportResult:
    record = run_plugin(NisExportPlugin(), microscope, NisExportParams(), copy)
    assert record.status is StageStatus.done
    assert record.result is not None
    return NisExportResult.model_validate(record.result)


def test_nis_export_reads_the_prototype_record_unchanged(
    demo_microscope: Microscope,
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = _seed(tmp_path)
    code = sf.main([str(seed.path), "--min-spacing-um", "30"])
    capsys.readouterr()
    assert code == 0
    copy = tmp_path / "run-selected.json"

    result = _export(demo_microscope, copy)

    exported = RunDocument.load(copy).stages["nis-export"]
    assert exported.reads["route"] is not None
    assert exported.reads["route"].plugin == "route-prototype"
    points = decode((tmp_path / result.file).read_bytes())
    stages = RunDocument.load(copy).stages
    chosen = RouteResult.model_validate(stages["route-prototype"].result).positions
    assert result.positions == len(chosen) == 6
    assert (
        result.z_mode == "pfs-offset"
    )  # every frame was locked: a Z and an offset each
    assert [p.name for p in points] == [s.name for s in chosen]
    # Each stop is its own frame's readback, Z and autofocus offset, exported as they are.
    frames = {(f.x_um, f.y_um): f for f in seed.scan.frames}
    for point in points:
        frame = frames[(point.x_um, point.y_um)]
        assert (point.z_um, point.pfs_offset) == (frame.z_um, frame.pfs_offset)


def test_an_older_route_record_stays_in_the_copy_and_the_export_is_the_prototypes(
    demo_microscope: Microscope,
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = _seed(tmp_path)
    # The route plugin's own answer, from the same document: one stop, from an older run.
    older_stops = RouteResult.model_validate(
        {
            "objective": seed.scan.objective,
            "matrix_um_per_px": seed.matrix,
            "image_shape_px": seed.shape,
            "fov_um": (400.0, 300.0),
            "labels": ("single",),
            "positions": [
                {
                    "order": 0,
                    "well": "A1",
                    "name": "A1_r000",
                    "x_um": seed.scan.frames[0].x_um,
                    "y_um": seed.scan.frames[0].y_um,
                    "z_um": 1.0,
                    "pfs_offset": 2.0,
                    "counts": {"single": 2},
                }
            ],
            "uncovered": {"A1": 0, "A2": 0},
            "distance_um": 0.0,
            "travel_s": None,
        }
    )
    write_record(seed.path, "route", "route", older_stops)
    older = RunDocument.load(seed.path).stages["route"]

    assert sf.main([str(seed.path), "--min-spacing-um", "30"]) == 0
    capsys.readouterr()
    copy = tmp_path / "run-selected.json"
    assert RunDocument.load(copy).stages["route"] == older  # kept, not edited

    result = _export(demo_microscope, copy)
    assert (
        result.positions == 6
    )  # the prototype's six stops, not the older one's single stop
    exported = RunDocument.load(copy).stages["nis-export"]
    assert exported.reads["route"] is not None
    assert exported.reads["route"].plugin == "route-prototype"


def test_a_stop_without_a_wanted_cell_is_never_exported(
    demo_microscope: Microscope,
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A well of debris only has no stop, so the export holds the other well's alone.
    seed = seed_run(
        tmp_path,
        cells=lambda scan: [
            *_cells(scan)[:6],
            cell_at("A2", 0.0, 0.0, "debris"),
        ],
        wells=("A1", "A2"),
        rows=1,
        columns=3,
        overlap=0.0,
    )
    assert sf.main([str(seed.path), "--min-spacing-um", "30"]) == 0
    capsys.readouterr()
    result = _export(demo_microscope, tmp_path / "run-selected.json")
    points = decode((tmp_path / result.file).read_bytes())
    assert [p.name for p in points] == ["A1_r000", "A1_r001", "A1_r002"]
