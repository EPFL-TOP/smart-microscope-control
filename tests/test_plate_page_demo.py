"""The plate page on Micro-Manager's demo devices (m3 §15, #197).

A click in drive mode is the one thing the page does that moves a stage, so it is
proved once on the demo stage: the click finds the well from the registered plate,
``goto-well`` runs through the layer, and the stage reads back within 0.5 µm of
where the forward model of m3 §2.1 puts the well's centre. The demo stage travels,
so the position is read after the facade's wait (FM-85).
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult
from smc.run import StageRecord, StageStatus, write_stage
from smc.ui.controller import PlatePage

pytestmark = pytest.mark.demo

GOTO_WELL = EntryPoint(
    name="goto-well", value="smc.plugins.goto_well:GotoWellPlugin", group=GROUP
)
PITCH_UM = 9000.0


def _seed(run_doc: Path, a1: tuple[float, float], rotation_deg: float) -> None:
    """The ``plate`` slot as an earlier ``plate-registration`` run left it, ``done``."""
    then = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=a1,
        rotation_deg=rotation_deg,
        method="references",
        residual_um=2.0,
        scale_error=0.001,
        n_refs=2,
    )
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=plate.model_dump(mode="json"),
        ),
    )


@pytest.mark.parametrize(
    ("well", "rotation_deg"),
    [("A1", 0.0), ("C5", 3.0)],
    ids=["a1-straight", "c5-turned"],
)
def test_a_drive_click_moves_the_demo_stage_to_the_well(
    demo_microscope: Microscope, tmp_path: Path, well: str, rotation_deg: float
) -> None:
    a1 = (1000.0, -500.0)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, a1, rotation_deg)
    row, col = ord(well[0]) - ord("A"), int(well[1:]) - 1
    angle = math.radians(rotation_deg)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    target = (
        a1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        a1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )
    page = PlatePage(demo_microscope, run_doc, entry_points=[GOTO_WELL])
    try:
        assert page.click(*target, drive=True) == well
        assert page.join(30.0)
        status = page.status()
        assert status.outcome is not None
        assert status.outcome.startswith("goto-well done in ")
        stage = demo_microscope.require(XYStage)
        stage.wait(10.0)
        position = stage.position_um()
        assert (position.x_um, position.y_um) == pytest.approx(target, abs=0.5)
        assert page.selection == ()
        assert page.current == well
    finally:
        page.close()


def test_the_page_builds_on_the_demo(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    if os.environ.get("SMC_REQUIRE_UI"):
        import panel  # noqa: F401  (CI must fail, not skip, when the extra is missing)
    else:
        pytest.importorskip("panel")
    from smc.ui.plate_page import PlatePageView, build

    page = PlatePage(demo_microscope, tmp_path / "run.json")
    try:
        assert build(page).get_root() is not None
        view = PlatePageView(page)
        assert view.status_text.object.startswith("XY (0.0, 0.0) µm | Z ")
        assert len(view.wells_source.data["name"]) == 96
        assert not tmp_path.joinpath(
            "run.json"
        ).exists()  # building the page writes nothing
    finally:
        page.close()
