"""The demo-sample stand and documents of the plate check's tests (m3 §27.1, §27.8, #542).

Not a test module: ``test_plate_check_demo`` (the tool) and
``test_scan_plate_check_demo`` (the scan) import it. The stand is
``profiles/demo-sample.toml`` turned as §20.1 turned it: the camera 30° and
mirrored, the plate 1.2°, and a shift per objective in ``[sample]`` (10X ``(5.0,
-2.5)``, 20X ``(0, 0)``, 40X ``(-32.5, 9.75)``) under which a feature is centred
at its stage position plus the shift. The profile's ``[objectives]`` holds the
same shifts as offsets from the 20X, so that the 10X -> 40X shift the layer
hands out, ``(-37.5, 12.25)``, is the one the sample makes; ``forty=False``
leaves the 40X's offset out (not measured).

``build_documents`` makes the documents once per module through ``run_plugin``
(the seam, FM-124): ``camera-calibration`` under the 40X into ``cal-40x.json``,
then ``camera-calibration`` and ``plate-walls`` (A1, H12 and A12, recorded up to
300 µm off) under the 10X, the objective in place at open, into ``run.json``.
``move_registration`` rewrites ``plate-walls``' record as if the plate had been
registered turned and slid (§27.1): the record stays usable, and the plate on
the stage stays where it is, so the check reads the move with its sign
reversed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile, SampleSection
from smc.plugins import run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.slots import WellRef
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
MEASURED = "2026-11-18, the demo-sample (#542)"
#: The shift of each objective, declared (§20.8): a feature is centred at its
#: stage position plus this.
SHIFT = {TEN: (5.0, -2.5), TWENTY: (0.0, 0.0), FORTY: (-32.5, 9.75)}
#: 10X -> 40X, written out: (-32.5 - 5.0, 9.75 + 2.5).
SHIFT_TEN_TO_FORTY = (-37.5, 12.25)
PLATE_DEG = 1.2
#: The demo-sample's A1 (a 96-well plate centred on the origin) and its pitch.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
#: The wells' (column, row) indices, written out here and not read from useq.
INDEX = {"A1": (0, 0), "H12": (11, 7), "A12": (11, 0)}
#: Where the operator "centred" each well: up to 300 µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "H12": (-250.0, 150.0), "A12": (-100.0, 200.0)}
WELLS = ("A1", "H12", "A12")


def truth(well: str) -> tuple[float, float]:
    """A well's true centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    col, row = INDEX[well]
    angle = math.radians(PLATE_DEG)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        A1_UM[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_UM[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def profile(*, forty: bool = True) -> Profile:
    """``demo-sample`` turned as §20.1, with the shifts in ``[sample]`` and as offsets from the 20X in ``[objectives]``."""
    base = Profile.load(DEMO_SAMPLE)
    offsets: dict[str, dict[str, Any]] = {
        TEN: {"xy_offset_um": [5.0, -2.5], "measured": MEASURED},
    }
    if forty:
        offsets[FORTY] = {"xy_offset_um": [-32.5, 9.75], "measured": MEASURED}
    dumped = base.model_dump()
    dumped["sample"] = SampleSection(
        rotation_deg=PLATE_DEG,
        camera_rotation_deg=30.0,
        mirrored=True,
        objective_shift_um=SHIFT,
    ).model_dump()
    dumped["objectives"] = {"reference": TWENTY, "offsets": offsets}
    return Profile.model_validate(dumped)


def open_stand(the_profile: Profile, *, dry_run: bool = False) -> Microscope:
    """The stand on ``the_profile`` with a stage fast enough to walk: the demo's moves at 10 mm/s by default."""
    stand = Microscope.open(the_profile, dry_run=dry_run)
    stand.core.setProperty("XY", "Velocity", "100000")
    return stand


@contextmanager
def opened(the_profile: Profile, *, dry_run: bool = False) -> Iterator[Microscope]:
    """``open_stand``, closed afterwards."""
    stand = open_stand(the_profile, dry_run=dry_run)
    try:
        yield stand
    finally:
        stand.close()


@dataclass(frozen=True)
class Documents:
    """The run documents the tests start from."""

    run: Path
    forty: Path


def build_documents(folder: Path) -> Documents:
    """``camera-calibration`` under the 40X, and the 10X's calibration and registration in ``run.json``."""
    docs = Documents(folder / "run.json", folder / "cal-40x.json")
    with opened(profile()) as stand:
        turret = stand.require(ObjectiveTurret)
        assert turret.current() == TEN
        turret.select(FORTY, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.forty
        )
        assert record.status is StageStatus.done, record.error
        turret.select(TEN, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.run
        )
        assert record.status is StageStatus.done, record.error
        refs = tuple(
            WellRef(
                well=well,
                x_um=truth(well)[0] + ROUGH_UM[well][0],
                y_um=truth(well)[1] + ROUGH_UM[well][1],
            )
            for well in WELLS
        )
        record = run_plugin(
            PlateWallsPlugin(),
            stand,
            PlateWallsParams(plate="96-well", refs=refs),
            docs.run,
        )
        assert record.status is StageStatus.done, record.error
    return docs


def _moved(
    point: list[float] | tuple[float, float],
    turn_deg: float,
    slide_um: tuple[float, float],
) -> list[float]:
    """``point`` turned about the stage's origin by ``turn_deg`` and slid by ``slide_um``."""
    angle = math.radians(turn_deg)
    x, y = point
    return [
        x * math.cos(angle) - y * math.sin(angle) + slide_um[0],
        x * math.sin(angle) + y * math.cos(angle) + slide_um[1],
    ]


def move_registration(
    path: Path, turn_deg: float = 0.0, slide_um: tuple[float, float] = (0.0, 0.0)
) -> None:
    """Rewrite ``plate-walls``' record with every position turned about the origin and slid (§27.1).

    A1, the rotation, the references and every measured well centre move; the
    record is validated again and stays usable (``unusable("plate", ...)`` is
    ``None``), as a plate registered where it now is not would be.
    """
    record = RunDocument.load(path).stages["plate-walls"]
    dumped = record.model_dump(mode="json")
    result = dumped["result"]
    result["a1_center_xy_um"] = _moved(result["a1_center_xy_um"], turn_deg, slide_um)
    result["rotation_deg"] = result["rotation_deg"] + turn_deg
    for ref in result["refs"]:
        ref["x_um"], ref["y_um"] = _moved(
            (ref["x_um"], ref["y_um"]), turn_deg, slide_um
        )
    for fit in result["walls"]["wells"]:
        fit["center_xy_um"] = _moved(fit["center_xy_um"], turn_deg, slide_um)
    write_stage(path, StageRecord.model_validate(dumped))
