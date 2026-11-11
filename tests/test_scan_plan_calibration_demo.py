"""``scan-plan`` for a plate registered at one objective, with another objective's calibration from a document of its own (m3 §23, #383).

The owner registers the plate with the 4x and scans with the 40x. A run document
holds one ``camera-calibration``, so the 40x's cannot be run into the
registration's document without replacing the 4x's and making ``plate-walls``
stale; and without it ``scan-plan`` plans with the profile's size, rotation 0 and
no mirror, for a camera that is turned. The stand here is the demo-sample's, as
§23.1 measured it: its camera turned 30° and mirrored, its plate turned 1.2°, no
``[objectives]``, so no offset between objectives is measured and the plan says so
(that line is the frame's, not the camera's).

A module fixture builds the documents once, through ``run_plugin`` (the seam, FM-124):
``camera-calibration`` under the 20X into ``cal-20x.json`` and under the 40X into
``cal-40x.json``, then ``camera-calibration`` and ``plate-walls`` (A1, H12 and A12,
recorded up to 300 µm off) under the 10X, the objective in place at open, into
``run.json``. Each test copies ``run.json`` and passes the calibration documents by
path. The demo stage moves at its ``Velocity`` property, 10 mm/s by default: the
stand here raises it to 100 000, as ``plate-walls``' tests do. Nothing here compares a
stage position (FM-142): the demo stage reads back slightly off its target, and the
registration passes at 300 µm off.

The expected matrix and the expected time are read from ``cal-40x.json``'s own
record, never from the code that wrote them (FM-203).
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from objective_offsets_stand import NOT_MEASURED
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile, SampleSection
from smc.plugins import PluginRunError, run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.geometry import pixel_matrix
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.scan_plan.plugin import ScanPlanParams, ScanPlanPlugin
from smc.plugins.slots import CameraCalibration, ScanPlanResult, WellRef
from smc.run import RunDocument, StageRecord, StageStatus

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
PLATE_DEG = 1.2
#: The demo-sample's A1 (a 96-well plate centred on the origin) and its pitch.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
#: The wells' (column, row) indices, written out here and not read from useq.
INDEX = {"A1": (0, 0), "H12": (11, 7), "A12": (11, 0)}
#: Where the operator "centred" each well: up to 300 µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "H12": (-250.0, 150.0), "A12": (-100.0, 200.0)}
#: A small explicit grid: the tests are not about the grid.
SMALL = {"rows": 2, "columns": 2}
NAME = "scan-plan"


def _truth(well: str) -> tuple[float, float]:
    """A well's true centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    col, row = INDEX[well]
    angle = math.radians(PLATE_DEG)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        A1_UM[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_UM[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _profile() -> Profile:
    """``demo-sample`` turned as §23.1 turned it: the camera 30° and mirrored, the plate 1.2°."""
    return Profile.load(DEMO_SAMPLE).model_copy(
        update={
            "sample": SampleSection(
                rotation_deg=PLATE_DEG, camera_rotation_deg=30.0, mirrored=True
            )
        }
    )


@dataclass(frozen=True)
class Documents:
    """The run documents the tests start from."""

    run: Path
    twenty: Path
    forty: Path


@pytest.fixture(scope="module")
def documents(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Documents:
    """``camera-calibration`` under the 20X and the 40X, and the registration under the 10X, through ``run_plugin``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("calibration-from-another-document")
    docs = Documents(
        folder / "run.json", folder / "cal-20x.json", folder / "cal-40x.json"
    )
    stand = Microscope.open(_profile())
    try:
        stand.core.setProperty("XY", "Velocity", "100000")
        turret = stand.require(ObjectiveTurret)
        assert turret.current() == TEN
        for label, path in ((TWENTY, docs.twenty), (FORTY, docs.forty)):
            turret.select(label, confirm=True)
            record = run_plugin(
                CameraCalibrationPlugin(), stand, CameraCalibrationParams(), path
            )
            assert record.status is StageStatus.done
        turret.select(TEN, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.run
        )
        assert record.status is StageStatus.done
        refs = tuple(
            WellRef(
                well=well,
                x_um=_truth(well)[0] + ROUGH_UM[well][0],
                y_um=_truth(well)[1] + ROUGH_UM[well][1],
            )
            for well in ("A1", "H12", "A12")
        )
        record = run_plugin(
            PlateWallsPlugin(),
            stand,
            PlateWallsParams(plate="96-well", refs=refs),
            docs.run,
        )
        assert record.status is StageStatus.done
    finally:
        stand.close()
    return docs


@pytest.fixture
def run_doc(documents: Documents, tmp_path: Path) -> Path:
    """A copy of the registration's ``run.json``: each test writes its own plan."""
    copy = tmp_path / "run.json"
    shutil.copy(documents.run, copy)
    return copy


@pytest.fixture
def open_stand(mm_available: bool) -> Iterator[Callable[..., Microscope]]:
    """Opens ``demo-sample`` turned as the documents were made, with the 10X in place; closes what it opened."""
    opened: list[Microscope] = []

    def make(*, dry_run: bool = False) -> Microscope:
        if not mm_available:
            pytest.skip("Micro-Manager demo adapters not installed")
        stand = Microscope.open(_profile(), dry_run=dry_run)
        opened.append(stand)
        return stand

    yield make
    for stand in opened:
        stand.close()


def _plan(
    stand: Microscope, run_doc: Path, calibration: Path, **params: Any
) -> StageRecord:
    return run_plugin(
        ScanPlanPlugin(),
        stand,
        ScanPlanParams(
            wells="A1", objective=FORTY, calibration=calibration, **SMALL, **params
        ),
        run_doc,
    )


def _calibration_in(path: Path) -> tuple[CameraCalibration, StageRecord]:
    """The calibration a document holds, as its record says: what the tests expect, never what ``scan-plan`` returns."""
    record = RunDocument.load(path).stages["camera-calibration"]
    assert record.result is not None
    return CameraCalibration.model_validate(record.result), record


def _determinant(matrix: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry"])
def test_a_plate_registered_at_one_objective_is_planned_with_another_objectives_calibration(
    dry_run: bool,
    documents: Documents,
    run_doc: Path,
    open_stand: Callable[..., Microscope],
) -> None:
    stand = open_stand(dry_run=dry_run)
    profile = stand.profile.microscope.name
    expected, record = _calibration_in(documents.forty)
    assert expected.objective == FORTY
    assert record.finished_utc is not None
    # The registration is of the 10X and usable before the plan: nothing has replaced it.
    assert (
        RunDocument.load(run_doc).unusable("plate", profile=profile, dry_run=dry_run)
        is None
    )

    planned = _plan(stand, run_doc, documents.forty)

    assert planned.status is StageStatus.done
    assert planned.result is not None
    result = ScanPlanResult.model_validate(planned.result)
    assert result.objective == FORTY
    assert result.plate_objective == TEN
    # The 40X's matrix and size, read from its document, not the profile's.
    assert result.matrix_source == "camera-calibration"
    assert result.matrix_um_per_px == expected.matrix_um_per_px
    assert result.pixel_size_um == expected.pixel_size_um
    # Turned and mirrored (FM-203, FM-73), not an upright camera of the same size.
    # The upright matrix, diag(ps, -ps), has the determinant -ps² (the image's rows
    # go down); a mirror flips its sign, so the mirrored camera's is +ps². The
    # size is the document's own, never taken from the plan.
    assert result.matrix_um_per_px != pixel_matrix(expected.pixel_size_um)
    assert _determinant(pixel_matrix(expected.pixel_size_um)) < 0.0
    assert _determinant(result.matrix_um_per_px) == pytest.approx(
        expected.pixel_size_um**2
    )
    assert result.calibration_from == documents.forty.as_posix()
    assert result.calibration_measured == record.finished_utc.isoformat(
        timespec="seconds"
    )
    # Nothing assumed the camera straight: the only lines are the frame's (no offset
    # between the objectives is declared on this stand) and the gaps a turned
    # camera leaves, which are the camera's measured turn and not an assumption.
    assert not [line for line in result.assumed if "rotation 0" in line]
    assert result.assumed[0] == NOT_MEASURED.format(origin=TEN, target=FORTY)
    assert len(result.assumed) == 2
    assert result.assumed[1] == (
        "the camera is turned 30.0° from the stage axes: no overlap below 0.9 "
        "closes the gaps between fields; straighten the camera"
    )
    # The registration is untouched, and the plan is usable.
    after = RunDocument.load(run_doc)
    assert after.unusable("plate", profile=profile, dry_run=dry_run) is None
    assert after.unusable("scan-plan", profile=profile, dry_run=dry_run) is None
    assert after.stages["camera-calibration"].result == (
        RunDocument.load(documents.run).stages["camera-calibration"].result
    )


def test_a_calibration_of_a_third_objective_is_refused_naming_both(
    documents: Documents, run_doc: Path, open_stand: Callable[..., Microscope]
) -> None:
    stand = open_stand()
    other, _ = _calibration_in(documents.twenty)
    assert other.objective == TWENTY

    with pytest.raises(PluginRunError) as raised:
        _plan(stand, run_doc, documents.twenty)

    assert raised.value.plugin == NAME
    assert raised.value.reason == (
        f"the calibration in {documents.twenty.as_posix()} is of {TWENTY}, the plan "
        f"is for {FORTY}: give calibration a run document whose camera-calibration "
        f"was made with {FORTY} in place"
    )
    # FM-98: raised outside the handler of the error it stands for.
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    stored = RunDocument.load(run_doc).stages[NAME]
    assert stored.status is StageStatus.failed
    assert stored.result is None
