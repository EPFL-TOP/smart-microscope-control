"""``plate-orientation`` through the runner, and the tools after it on a turned plate (m3 §24).

The plate is a 24-well (4 x 6, not square: FM-73) turned 180° in its holder and
tilted 0.3°, A1 at the stage's +X -Y corner. Every expected position is
``useq``'s own (``WellPlatePlan(rotation=180 + tilt)``), never ``smc``'s
geometry and never a hand-flipped sign: a mirror in place of a turn, or rows
swapped with columns, fails here.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import useq

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.plugins import PluginRunError, run_plugin
from smc.plugins.goto_well import GotoWellParams, GotoWellPlugin
from smc.plugins.plate_orientation import (
    PlateOrientationParams,
    PlateOrientationPlugin,
)
from smc.plugins.plate_registration.plugin import (
    PlateRegistrationParams,
    PlateRegistrationPlugin,
)
from smc.plugins.scan_plan.plugin import ScanPlanParams, ScanPlanPlugin
from smc.plugins.slots import PlateResult, ScanPlanResult
from smc.plugins.slots.plate import PlateOrientation, turned_well
from smc.run import RunDocument, StageRecord, StageStatus

pytestmark = pytest.mark.demo

PLATE = "24-well"
#: A1 of the plate as it lies, turned: on the stage's +X -Y side of the plate.
A1_UM = (48000.0, -29000.0)
TILT_DEG = 0.3
#: Half the width of a 24-well's well (useq: 15.6 mm).
HALF_UM = 7800.0


def _truth(well: str) -> tuple[float, float]:
    """Where the printed ``well`` is on the stage, from ``useq``'s plan of the turned plate."""
    plan = useq.WellPlatePlan(
        plate=PLATE, a1_center_xy=A1_UM, rotation=180.0 + TILT_DEG
    )
    (position,) = [p for p in plan.all_well_positions if p.name == well]
    return float(position.x), float(position.y)  # type: ignore[arg-type]


@pytest.fixture
def stand(demo_microscope: Microscope) -> Microscope:
    """The demo stand with a stage fast enough to cross a plate (13 ms for 30 mm)."""
    demo_microscope.core.setProperty("XY", "Velocity", "100000")
    return demo_microscope


def _register(
    microscope: Microscope, run_doc: Path, orientation: PlateOrientation
) -> StageRecord:
    """Three wells named from the plate, centred where they are."""
    refs = [
        {"well": well, "x_um": _truth(well)[0], "y_um": _truth(well)[1]}
        for well in ("A1", "A6", "D1")
    ]
    return run_plugin(
        PlateRegistrationPlugin(),
        microscope,
        PlateRegistrationParams.model_validate(
            {"plate": PLATE, "refs": refs, "orientation": orientation}
        ),
        run_doc,
    )


def _turn(
    microscope: Microscope, run_doc: Path, orientation: PlateOrientation
) -> StageRecord:
    return run_plugin(
        PlateOrientationPlugin(),
        microscope,
        PlateOrientationParams(orientation=orientation),
        run_doc,
    )


def _plate(record: StageRecord) -> PlateResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return PlateResult.model_validate(record.result)


def _unusable(microscope: Microscope, run_doc: Path) -> str | None:
    return RunDocument.load(run_doc).unusable(
        "plate",
        profile=microscope.profile.microscope.name,
        dry_run=microscope.dry_run,
    )


def _here(microscope: Microscope) -> tuple[float, float]:
    position = microscope.require(XYStage).position_um()
    return position.x_um, position.y_um


def test_plate_orientation_renames_the_registered_plate_through_the_runner(
    stand: Microscope, tmp_path: Path
) -> None:
    run_doc = tmp_path / "run.json"

    # Declared not turned, the wells named from the plate fit at 180°: refused.
    with pytest.raises(PluginRunError, match="it lies turned 180° from the declared"):
        _register(stand, run_doc, "not-turned")
    fit_record = _register(stand, run_doc, "turned-180")
    fit = _plate(fit_record)
    assert fit.orientation == "turned-180"
    assert fit.well_center_um("B3") == pytest.approx(_truth("B3"), abs=1e-6)

    renamed = _plate(_turn(stand, run_doc, "not-turned"))
    assert _unusable(stand, run_doc) is None
    doc = RunDocument.load(run_doc)
    assert not doc.is_stale("plate-orientation")
    assert doc.stages["plate-orientation"].reads["plate"] is not None
    assert doc.stages["plate-orientation"].reads["plate"].plugin == (
        "plate-registration"
    )
    assert renamed.orientation == "not-turned"
    assert renamed.method == fit.method
    assert renamed.residual_um == fit.residual_um
    # Positions kept, names swapped: the renamed B3 is the printed C4.
    for well in ("A1", "B3", "D6", "C2"):
        assert renamed.well_center_um(turned_well(PLATE, well)) == pytest.approx(
            fit.well_center_um(well), abs=1e-6
        )
    assert renamed.well_center_um("B3") == pytest.approx(_truth("C4"), abs=1e-6)

    # Back: it reads the fit again, never its own output, so the fit comes back as it is.
    back = _turn(stand, run_doc, "turned-180")
    assert _unusable(stand, run_doc) is None
    assert back.result == fit_record.result
    assert not RunDocument.load(run_doc).is_stale("plate-orientation")


def test_goto_well_on_a_turned_plate_reaches_the_printed_b3(
    stand: Microscope, tmp_path: Path
) -> None:
    run_doc = tmp_path / "run.json"
    _plate(_register(stand, run_doc, "turned-180"))

    record = run_plugin(GotoWellPlugin(), stand, GotoWellParams(well="B3"), run_doc)

    assert record.status is StageStatus.done
    here = _here(stand)
    assert math.dist(here, _truth("B3")) < 1.0
    # Not where B3 would be on the plate not turned: that is the printed C4.
    assert math.dist(here, _truth("C4")) > 10_000.0


@pytest.mark.parametrize("orientation", ["not-turned", "turned-180"])
def test_goto_well_side_plus_x_moves_towards_plus_x_on_a_turned_plate(
    stand: Microscope, tmp_path: Path, orientation: PlateOrientation
) -> None:
    """``side=+X`` is the stage's +X on either declaration (m3 §24 D9, FM-189)."""
    run_doc = tmp_path / "run.json"
    _plate(_register(stand, run_doc, "turned-180"))
    if orientation == "not-turned":
        _plate(_turn(stand, run_doc, "not-turned"))
    centre = _plate_of(run_doc).well_center_um("B3")

    for side, (ux, uy) in {"+X": (1.0, 0.0), "-Y": (0.0, -1.0)}.items():
        run_plugin(
            GotoWellPlugin(),
            stand,
            GotoWellParams.model_validate({"well": "B3", "side": side}),
            run_doc,
        )
        cos, sin = math.cos(math.radians(TILT_DEG)), math.sin(math.radians(TILT_DEG))
        expected = (
            centre[0] + HALF_UM * (cos * ux - sin * uy),
            centre[1] + HALF_UM * (sin * ux + cos * uy),
        )
        assert math.dist(_here(stand), expected) < 1.0, side


def _plate_of(run_doc: Path) -> PlateResult:
    provider = RunDocument.load(run_doc).provider("plate")
    assert provider is not None
    assert provider.result is not None
    return PlateResult.model_validate(provider.result)


def _plan(microscope: Microscope, run_doc: Path) -> ScanPlanResult:
    record = run_plugin(
        ScanPlanPlugin(),
        microscope,
        ScanPlanParams.model_validate(
            {"wells": "B3", "pixel_size_um": 0.3, "rows": 2, "columns": 2}
        ),
        run_doc,
    )
    assert record.status is StageStatus.done
    assert record.result is not None
    return ScanPlanResult.model_validate(record.result)


def _middle(plan: ScanPlanResult) -> tuple[float, float]:
    fields = [f for f in plan.fields if f.well == "B3"]
    assert fields
    return (
        sum(f.x_um for f in fields) / len(fields),
        sum(f.y_um for f in fields) / len(fields),
    )


def test_scan_plan_after_a_turn_is_stale_and_plans_the_turned_names(
    stand: Microscope, tmp_path: Path
) -> None:
    run_doc = tmp_path / "run.json"
    _plate(_register(stand, run_doc, "turned-180"))
    assert math.dist(_middle(_plan(stand, run_doc)), _truth("B3")) < 1.0

    _plate(_turn(stand, run_doc, "not-turned"))

    assert RunDocument.load(run_doc).is_stale("scan-plan")
    # Planned again, its B3 is the renamed plate's: the printed C4.
    replanned = _plan(stand, run_doc)
    assert not RunDocument.load(run_doc).is_stale("scan-plan")
    assert math.dist(_middle(replanned), _truth("C4")) < 1.0
