"""The plate page makes the camera calibration of each objective on the demo-sample stand, through the view's own handlers (m5 §18.6, #501).

The stand is ``demo-sample`` turned as m3 §20.1 turned it (camera 30° and mirrored, plate
1.2°) with a declared shift per objective: 10X ``(5.0, -2.5)``, 20X ``(0, 0)``, 40X
``(-32.5, 9.75)``. The plate is registered under the 10X, so the offsets to recover are written
out from the shifts and never read from the code (FM-124, FM-203): 40X from 10X
``(-37.5, 12.25)``, 20X from 10X ``(-5.0, 2.5)``, compared within 1.0 µm, a pixel of the 10X.

A module fixture builds ``run.json`` once through ``run_plugin`` (the seam): the 10X's
calibration and ``plate-walls`` on A1, H12 and A12, **and no calibration document**: the page
makes those. Each test copies ``run.json`` and a copy of the profile into its own folder, opens a
stand with the demo stage's ``Velocity`` raised on its own core before the page is built over it
(the demo moves at 10 mm/s, and a run would take a minute), and drives the view: a click is
``button.clicks += 1``. The turret is turned by ``select(label, confirm=True)``, which is the
operator's turn (the Stand screen's *Change objective*) and not under test. No assertion rests
on a duration.
"""

from __future__ import annotations

import math
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile
from smc.plugins import run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.registry import GROUP
from smc.plugins.slots import CameraCalibration, ObjectiveOffsetsResult, WellRef
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage
from smc.ui.guide import CALIBRATE_DRY_RUN
from smc.ui.plate_page import PlatePageView

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The demo turret's positions, from its labels in order (Micro-Manager's ``DObjective``): the 40X is
#: position 1, the 10X position 2, the 20X position 4.
POSITION = {FORTY: 1, TEN: 2, TWENTY: 4}
#: The shift of each objective, declared (m3 §20.8): a feature is centred at its stage
#: position plus this.
SHIFT = {TEN: [5.0, -2.5], TWENTY: [0.0, 0.0], FORTY: [-32.5, 9.75]}
PLATE_DEG = 1.2
#: The demo-sample's A1 (a 96-well plate centred on the origin) and its pitch.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
INDEX = {"A1": (0, 0), "H12": (11, 7), "A12": (11, 0)}
#: Where the operator "centred" each well: up to 300 µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "H12": (-250.0, 150.0), "A12": (-100.0, 200.0)}
TOOLS = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "objective-offsets": "smc.plugins.objective_offsets.plugin:ObjectiveOffsetsPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
}


def _truth(well: str) -> tuple[float, float]:
    """A well's true centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    col, row = INDEX[well]
    angle = math.radians(PLATE_DEG)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        A1_UM[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_UM[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _profile_text() -> str:
    """``demo-sample.toml`` with its ``[sample]`` turned as m3 §20.1 turned it: a profile file's text."""
    data = tomllib.loads(DEMO_SAMPLE.read_text(encoding="utf-8"))
    data["sample"].update(
        {
            "rotation_deg": PLATE_DEG,
            "camera_rotation_deg": 30.0,
            "mirrored": True,
            "objective_shift_um": SHIFT,
        }
    )
    return tomli_w.dumps(data)


def _entry_points() -> list[EntryPoint]:
    return [EntryPoint(name=n, value=v, group=GROUP) for n, v in TOOLS.items()]


@pytest.fixture(scope="module")
def run_json(mm_available: bool, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The registration under the 10X, through ``run_plugin``: its calibration and ``plate-walls``, and no other calibration."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("page-calibrations")
    profile = folder / "stand.toml"
    profile.write_text(_profile_text(), encoding="utf-8")
    run_doc = folder / "run.json"
    stand = Microscope.open(Profile.load(profile))
    try:
        stand.core.setProperty("XY", "Velocity", "100000")
        assert stand.require(ObjectiveTurret).current() == TEN
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), run_doc
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
            run_doc,
        )
        assert record.status is StageStatus.done
    finally:
        stand.close()
    return run_doc


@dataclass(frozen=True)
class Rig:
    """One test's page, its view, its stand and the run document it may write."""

    page: PlatePage
    view: PlatePageView
    stand: Microscope
    run_doc: Path

    @property
    def folder(self) -> Path:
        return self.run_doc.parent / "calibrations"

    def document(self, label: str) -> Path:
        return self.folder / f"position-{POSITION[label]}.json"

    def turn(self, label: str) -> None:
        """The operator's turn of the turret, then the tick that draws it."""
        self.stand.require(ObjectiveTurret).select(label, confirm=True)
        self.view.tick()

    def finish(self) -> None:
        """Let the run end as a session would: ticks, then the redraw the end of a run asks for."""
        started = time.monotonic()
        while not self.page.join(0.01):
            self.view.tick()
            assert time.monotonic() - started < 300, "the run did not end"
        self.view.tick()
        self.view.refresh()


@pytest.fixture
def make_rig(run_json: Path, tmp_path: Path) -> Iterator[Callable[..., Rig]]:
    """Opens the stand on a copy of the profile, a page over it on a copy of ``run.json``, and a view; closes what it opened."""
    opened: list[tuple[PlatePage, Microscope]] = []

    def make(*, dry_run: bool = False) -> Rig:
        run_doc = tmp_path / "run.json"
        shutil.copy(run_json, run_doc)
        profile = tmp_path / "stand.toml"
        profile.write_text(_profile_text(), encoding="utf-8")
        stand = Microscope.open(Profile.load(profile), dry_run=dry_run)
        # The demo stage moves at its `Velocity` property, 10 mm/s by default.
        stand.core.setProperty("XY", "Velocity", "100000")
        page = PlatePage(stand, run_doc, entry_points=_entry_points())
        opened.append((page, stand))
        view = PlatePageView(page)
        view.show_step("plate")
        view.refresh()
        view.tick()
        return Rig(page, view, stand, run_doc)

    try:
        yield make
    finally:
        for page, stand in opened:
            page.close()
            stand.close()


def _click(button: Any) -> None:
    button.clicks += 1


def _record(path: Path) -> Any:
    record = RunDocument.load(path).provider("camera-calibration")
    assert record is not None
    return record


def _calibration(path: Path) -> CameraCalibration:
    return CameraCalibration.model_validate(_record(path).result)


def _within(found: tuple[float, float], declared: tuple[float, float]) -> None:
    assert abs(found[0] - declared[0]) <= 1.0, (found, declared)
    assert abs(found[1] - declared[1]) <= 1.0, (found, declared)


def test_the_page_calibrates_two_objectives_and_measures_the_offsets_with_an_empty_field(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    view, stand = rig.view, rig.stand
    turret = stand.require(ObjectiveTurret)
    # No calibration document exists: the registration is the 10X's, in run.json alone.
    assert not rig.folder.exists()
    assert view.offsets_box.visible is True

    for label in (FORTY, TWENTY):
        rig.turn(label)
        assert view.offsets_calibrate_button.label == f"Calibrate the camera at {label}"
        assert view.offsets_calibrate_button.disabled is False
        _click(view.offsets_calibrate_button)
        rig.finish()
        assert rig.page.status().outcome is not None
        assert rig.page.status().outcome.startswith("Camera calibration done in ")  # type: ignore[union-attr]
        # The document of the objective's turret position holds that objective's calibration.
        assert rig.document(label).is_file()
        assert _calibration(rig.document(label)).objective == label
        assert _record(rig.document(label)).status is StageStatus.done

    # A 40X pixel is half a 20X pixel (40/20), read from the two documents: a swapped document,
    # or two calibrations of one objective, fails here.
    forty = _calibration(rig.document(FORTY)).pixel_size_um
    twenty = _calibration(rig.document(TWENTY)).pixel_size_um
    assert twenty / forty == pytest.approx(2.0, rel=0.02)
    # The list names both documents, and the objectives that have none (the turret's others).
    lines = view.calibrations_held_text.object.splitlines()
    assert any(line.startswith(f"{FORTY}: calibrated on ") for line in lines)
    assert any(line.startswith(f"{TWENTY}: calibrated on ") for line in lines)
    assert f"{TEN}: no calibration yet" not in lines
    # Nothing in the clicks turned the turret: it reads the objective the operator last chose.
    assert turret.current() == TWENTY

    # The field stays empty, and the measure arms the held documents, in position order.
    assert view.calibrations_input.value == ""
    _click(view.offsets_measure_button)
    assert view._offsets_armed == (FORTY, TWENTY)
    assert view.offsets_confirm_box.visible is True
    assert turret.current() == TWENTY
    _click(view.offsets_turn_button)
    rig.finish()

    record = RunDocument.load(rig.run_doc).stages["objective-offsets"]
    assert record.status is StageStatus.done, rig.page.status().outcome
    result = ObjectiveOffsetsResult.model_validate(record.result)
    shifts = {m.objective: m.shift_xy_um for m in result.measures}
    _within(shifts[FORTY], (-37.5, 12.25))
    _within(shifts[TWENTY], (-5.0, 2.5))
    assert [m.objective for m in result.measures] == [FORTY, TWENTY]
    # Nothing turned the turret back, and nothing wrote a document into the field.
    assert turret.current() == TWENTY
    assert view.calibrations_input.value == ""


def test_calibrating_another_objective_leaves_run_json_byte_identical_and_the_plate_registered(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.turn(FORTY)
    before = rig.run_doc.read_bytes()
    stages_before = sorted(RunDocument.load(rig.run_doc).stages)

    _click(rig.view.offsets_calibrate_button)
    rig.finish()
    view = rig.page.view()

    # FM-216: the registration's document is neither read nor written by the calibration, so the
    # plate stays usable, the block stays offered and the checklist's `plate` stays ok.
    assert rig.run_doc.read_bytes() == before
    assert sorted(RunDocument.load(rig.run_doc).stages) == stages_before
    assert view.registered is not None
    assert view.offsets is not None
    assert [item.state for item in view.checklist if item.slot == "plate"] == ["ok"]
    assert rig.document(FORTY).is_file()
    # The run document's own calibration is still the 10X's, as the Stand screen says.
    assert view.camera_calibration is not None
    assert view.camera_calibration.objective == TEN
    # The Stand screen's button says what it would replace, before it is pressed.
    assert TEN in rig.view.calibrate_note.object
    assert FORTY in rig.view.calibrate_note.object


def test_a_stop_during_a_calibration_says_stopped_and_leaves_the_other_documents_and_run_json_alone(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    view = rig.view
    # The 20X's document is made first and must come out of the stop as it went in.
    rig.turn(TWENTY)
    _click(view.offsets_calibrate_button)
    rig.finish()
    twenty_before = rig.document(TWENTY).read_bytes()
    run_before = rig.run_doc.read_bytes()
    rig.turn(FORTY)
    # A settle after each move makes the run long enough to stop in (FM-64, FM-88).
    view.tool_fields["camera-calibration"]["settle_ms"].value = "3000"

    _click(view.offsets_calibrate_button)
    started = time.monotonic()
    while rig.page.status().progress is None:
        assert time.monotonic() - started < 120, "the calibration made no report"
        time.sleep(0.02)
    # The first probe is in: the second probe's settle is where the stop lands.
    time.sleep(0.5)
    rig.page.stop()
    rig.finish()

    record = _record(rig.document(FORTY))
    assert record.status is StageStatus.stopped
    assert rig.page.status().outcome == "Camera calibration stopped"
    assert rig.document(TWENTY).read_bytes() == twenty_before
    assert rig.run_doc.read_bytes() == run_before
    assert rig.page.view().registered is not None
    # The stopped record leaves the position with no usable document, and the list says so.
    lines = view.calibrations_held_text.object.splitlines()
    assert any(
        line.startswith(f"position {POSITION[FORTY]} ({FORTY}): ") for line in lines
    )


def test_a_dry_run_page_on_a_walls_plate_refuses_to_calibrate_and_writes_nothing(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(dry_run=True)
    rig.turn(FORTY)
    before = rig.run_doc.read_bytes()

    assert rig.view.offsets_box.visible is True
    assert rig.view.offsets_calibrate_button.disabled is True
    assert rig.view.offsets_calibrate_wait.object == CALIBRATE_DRY_RUN
    # The click arrives all the same (FM-169): refused in the controller, nothing made.
    _click(rig.view.offsets_calibrate_button)

    assert rig.page.status().outcome == f"Camera calibration: {CALIBRATE_DRY_RUN}"
    assert not rig.folder.exists()
    assert rig.run_doc.read_bytes() == before


def test_a_second_calibration_of_the_same_objective_replaces_its_document(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.turn(FORTY)
    _click(rig.view.offsets_calibrate_button)
    rig.finish()
    first = _record(rig.document(FORTY))
    assert rig.view.offsets_calibrate_button.label == (
        f"Calibrate the camera again at {FORTY}"
    )

    _click(rig.view.offsets_calibrate_button)
    rig.finish()
    second = _record(rig.document(FORTY))

    # The same file, replaced by the new attempt, with its own date.
    assert [p.name for p in rig.folder.iterdir()] == [rig.document(FORTY).name]
    assert second.attempt != first.attempt
    assert second.finished_utc is not None
    assert first.finished_utc is not None
    assert second.finished_utc > first.finished_utc
    assert second.status is StageStatus.done
