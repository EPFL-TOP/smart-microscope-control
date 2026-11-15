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
    OBJECTIVE,
    Seed,
    cell_at,
    focus_step,
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


#: The distance of frame k from the lock's plane, for k = 0 to 5 (a lock of 2000 + 10 k).
#: The offset of the cells' plane of each, by hand, with a factor of 112.6 offset units
#: per um: 2000 - 337.8, 2010 + 304.02, 2020 - 332.17, 2030 + 152.01, and the lock's own
#: where the frame is at the lock's plane (k = 4, 5).
_DZ = (-3.0, 2.7, -2.95, 1.35, 0.0, 0.0)
_DERIVED = (1662.0, 2314.0, 1688.0, 2182.0, 2040.0, 2050.0)


def _focused(tmp_path: Path) -> Seed:
    steps = {
        (well, i): focus_step(_DZ[3 * w + i])
        for w, well in enumerate(("A1", "A2"))
        for i in range(3)
    }
    return seed_run(
        tmp_path,
        cells=_cells,
        wells=("A1", "A2"),
        rows=1,
        columns=3,
        overlap=0.0,
        cell_focus_of=steps,
    )


def _profile(folder: Path, objective: str) -> Path:
    path = folder / "stand.toml"
    path.write_text(
        f'[microscope]\nname = "demo"\nvendor = "test"\n'
        f"[safety]\nautofocus_offset_soft_limits = [0, 20000]\n"
        f'[autofocus.offset_per_um]\n"{objective}" = 112.6\n',
        encoding="utf-8",
    )
    return path


def test_nis_export_says_and_writes_a_derived_offset(
    demo_microscope: Microscope,
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = _focused(tmp_path)
    profile = _profile(tmp_path, OBJECTIVE)
    code = sf.main(
        [str(seed.path), "--min-spacing-um", "30", "--profile", str(profile)]
    )
    capsys.readouterr()
    assert code == 0

    result = _export(demo_microscope, tmp_path / "run-selected.json")

    assert (result.positions, result.z_mode) == (6, "pfs-offset")
    points = decode((tmp_path / result.file).read_bytes())
    frames = {(f.x_um, f.y_um): k for k, f in enumerate(seed.scan.frames)}
    for point in points:
        k = frames[(point.x_um, point.y_um)]
        assert (point.z_um, point.pfs_offset) == (1000.0 + 10.0 * k, _DERIVED[k])
    # The export says that four of its offsets were derived and never read locked.
    assert any(
        line.startswith("4 of 6 positions have a dPFSOffset that route derived")
        for line in result.assumed
    )


def test_a_list_that_lost_an_offset_is_exported_in_z_mode(
    demo_microscope: Microscope,
    tmp_path: Path,
    sf: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No --profile: the four frames that have a distance to convert get no offset, and
    # one flag covers the whole list, so the export carries Z for every stop and the
    # autofocus for none: what the command's last warning says.
    seed = _focused(tmp_path)
    assert sf.main([str(seed.path), "--min-spacing-um", "30"]) == 0
    assert "nis-export writes the whole list without" in capsys.readouterr().err

    result = _export(demo_microscope, tmp_path / "run-selected.json")

    assert (result.positions, result.z_mode) == (6, "z")
    points = decode((tmp_path / result.file).read_bytes())
    assert [p.pfs_offset for p in points] == [None] * 6
    assert not any("derived" in line for line in result.assumed)


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
