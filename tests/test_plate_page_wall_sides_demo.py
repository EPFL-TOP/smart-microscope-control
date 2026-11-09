"""A wall named on the image, registered through the page, finds the centre where the walls are (#327, FM-189).

Ti2-E session 2 touched B3's walls at 10x and named them as Live showed them. On
that stand the image moves with the stage (camera at 180 degrees), so the image's
left is the stage's +X, and the layer, which read ``left`` as the stage's -X, put a
seed 15.5 mm from the well on both axes. These tests run the whole path on the demo
stand: the camera calibration's matrix in the run document, the stage moved to the
positions the session measured, each capture named on the image, and the real
``plate-registration`` run through the page. Every expected centre is written out
from the session's own numbers, never computed with ``frames.py`` (FM-124).

B3 of a 24-well plate is 15 600 µm across. The demo stage rounds a command by up to
0.005 µm, so a centre is compared within 0.05 µm and never with ``==`` (FM-85).
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.plugins import run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.registry import GROUP
from smc.plugins.slots import CameraCalibration, PlateResult
from smc.run import RunDocument, StageRecord, StageStatus
from smc.testing.fixtures import SampleMicroscopeFactory
from smc.ui.controller import PlatePage

pytestmark = pytest.mark.demo

REGISTRATION = EntryPoint(
    name="plate-registration",
    value="smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    group=GROUP,
)

#: The Ti2-E at 10x, a camera at 180 degrees: the image's left is the stage's +X.
TI2E = ((-1.1, 0.0), (0.0, 1.1))
#: An upright camera: the image's sides are the stage's.
UPRIGHT = ((1.1, 0.0), (0.0, -1.1))

# B3's four walls as session 2 measured them, by the stage direction that reaches each
# from the well's centre (M3 §17.1; the report's walls.txt excerpt of #334).
MINUS_X = (2468.0, -9737.3)
PLUS_X = (18067.4, -9737.3)
MINUS_Y = (10226.7, -17738.5)
PLUS_Y = (10226.7, -2337.6)
#: What those four give: the midpoints (the well's size is 15599.4 by 15400.9).
FOUR_WALLS_CENTRE = (10267.7, -10038.05)

# The incident: the two touches the operator named `left` and `bottom` on the image.
INCIDENT_LEFT = (17967.9, -10237.1)  # the stage's +X wall on the Ti2-E
INCIDENT_BOTTOM = (10368.3, -2334.1)  # the stage's +Y wall on the Ti2-E
INCIDENT_CENTRE = (10167.9, -10134.1)  # 14 and 46 µm from the measured centre
#: Where the layer put that seed when it read the image's words as the stage's.
INCIDENT_WRONG = (25767.9, 5465.9)


def _page(microscope: Microscope, run_doc: Path) -> PlatePage:
    page = PlatePage(microscope, run_doc, entry_points=[REGISTRATION])
    page.choose_plate("24-well")
    return page


def _capture(
    page: PlatePage, microscope: Microscope, side: str, where: tuple[float, float]
) -> None:
    """Move the stage to ``where`` and capture B3's wall at ``side`` of the image."""
    microscope.require(XYStage).move_to_um(*where)
    page.capture_wall("B3", side)


def _register(page: PlatePage, run_doc: Path) -> StageRecord:
    page.run("plate-registration", page.page_params("plate-registration"))
    assert page.join(30.0)
    return RunDocument.load(run_doc).stages["plate-registration"]


def _plate(record: StageRecord) -> PlateResult:
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    return PlateResult.model_validate(record.result)


def test_walls_named_on_the_image_give_the_centre_where_the_walls_are(
    demo_microscope: Microscope,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # The check of FM-189: the four walls, each named as Live shows it on a stand
    # whose image moves with the stage. A page that ignored the calibration would read
    # them in the stage's frame and refuse the pairs as swapped.
    run_doc = tmp_path / "run.json"
    seed_camera_calibration(run_doc, TI2E)
    page = _page(demo_microscope, run_doc)
    try:
        _capture(page, demo_microscope, "left", PLUS_X)
        _capture(page, demo_microscope, "right", MINUS_X)
        _capture(page, demo_microscope, "bottom", PLUS_Y)
        _capture(page, demo_microscope, "top", MINUS_Y)
        # What was recorded is the stage's, whatever the operator saw.
        recorded = {w.side: (w.x_um, w.y_um) for w in page.walls}
        assert recorded.keys() == {"+X", "-X", "+Y", "-Y"}
        assert recorded["+X"] == pytest.approx(PLUS_X, abs=0.05)
        assert recorded["-X"] == pytest.approx(MINUS_X, abs=0.05)
        assert recorded["+Y"] == pytest.approx(PLUS_Y, abs=0.05)
        assert recorded["-Y"] == pytest.approx(MINUS_Y, abs=0.05)
        record = _register(page, run_doc)
    finally:
        page.close()
    plate = _plate(record)
    assert plate.well_center_um("B3") == pytest.approx(FOUR_WALLS_CENTRE, abs=0.05)
    # Both axes were measured by a pair: nothing was assumed.
    assert not [line for line in plate.assumed if "nominal half-size" in line]


def test_the_incident_of_session_2_lands_on_the_well(
    demo_microscope: Microscope,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # The two touches of the incident, named as the operator saw them: `left` and
    # `bottom`. They gave B3's centre at (25767.9, 5465.9), a seed 15.5 mm away.
    run_doc = tmp_path / "run.json"
    seed_camera_calibration(run_doc, TI2E)
    page = _page(demo_microscope, run_doc)
    try:
        _capture(page, demo_microscope, "left", INCIDENT_LEFT)
        _capture(page, demo_microscope, "bottom", INCIDENT_BOTTOM)
        record = _register(page, run_doc)
    finally:
        page.close()
    plate = _plate(record)
    centre = plate.well_center_um("B3")
    assert centre == pytest.approx(INCIDENT_CENTRE, abs=0.05)
    assert abs(centre[0] - INCIDENT_WRONG[0]) > 15000.0
    assert abs(centre[1] - INCIDENT_WRONG[1]) > 15000.0
    # One wall per axis: the centre is a wall plus the nominal half-size, and says so.
    assert [line for line in plate.assumed if "nominal half-size" in line]


def test_on_an_upright_camera_the_image_names_are_the_stage_names(
    demo_microscope: Microscope,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # Without the calibration's turn the two frames are one: an upright camera's left
    # is the stage's -X. The same four walls, named for it, give the same centre; named
    # for the Ti2-E the same four captures are two swapped pairs, and refused as such.
    run_doc = tmp_path / "run.json"
    seed_camera_calibration(run_doc, UPRIGHT)
    page = _page(demo_microscope, run_doc)
    try:
        _capture(page, demo_microscope, "left", MINUS_X)
        _capture(page, demo_microscope, "right", PLUS_X)
        _capture(page, demo_microscope, "bottom", MINUS_Y)
        _capture(page, demo_microscope, "top", PLUS_Y)
        record = _register(page, run_doc)
        assert _plate(record).well_center_um("B3") == pytest.approx(
            FOUR_WALLS_CENTRE, abs=0.05
        )

        # The same captures on the Ti2-E: the page maps `left` to +X, so the pair
        # that was -X, +X is +X, -X: sizes of -15599.4 and -15400.9. A mutant that
        # ignores the matrix passes the first half and is killed by this one, and
        # one that always reads the Ti2-E's by the first test.
        seed_camera_calibration(run_doc, TI2E)
        page.forget()
        _capture(page, demo_microscope, "left", MINUS_X)
        _capture(page, demo_microscope, "right", PLUS_X)
        _capture(page, demo_microscope, "bottom", MINUS_Y)
        _capture(page, demo_microscope, "top", PLUS_Y)
        refused = _register(page, run_doc)
    finally:
        page.close()
    assert refused.status is StageStatus.failed
    assert refused.error is not None
    assert "B3's -X and +X walls are swapped" in refused.error
    assert "15599 µm" in refused.error
    assert "another well" not in refused.error


def test_a_real_calibration_of_a_camera_at_180_degrees_maps_left_to_plus_x(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    # The matrix is measured, not seeded: `camera-calibration` on a synthetic camera
    # turned by 180 degrees, run into the page's own run document, is what makes the
    # image's left the stage's +X.
    microscope = demo_microscope_with_sample(
        shape=(384, 512), camera_rotation_deg=180.0
    )
    run_doc = tmp_path / "run.json"
    record = run_plugin(
        CameraCalibrationPlugin(), microscope, CameraCalibrationParams(), run_doc
    )
    assert record.status is StageStatus.done
    assert record.result is not None
    assert CameraCalibration.model_validate(
        record.result
    ).rotation_deg == pytest.approx(180.0, abs=0.5)
    page = _page(microscope, run_doc)
    try:
        microscope.require(XYStage).move_to_um(*PLUS_X)
        touch = page.capture_wall("B3", "left")
        assert touch.side == "+X"
        assert page.capture_wall("B3", "right").side == "-X"
        assert page.capture_wall("B3", "top").side == "-Y"
        assert page.capture_wall("B3", "bottom").side == "+Y"
        # And the page says so, from the same document.
        matrix = page.view().camera_matrix
        assert matrix is not None
        assert matrix[0][0] == pytest.approx(-1.0, abs=0.05)
    finally:
        page.close()
