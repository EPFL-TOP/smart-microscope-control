"""The Stand screen says the numbers a calibration measured, on the demo-sample stand, through the view's own handlers (m5 §23, #627).

At the Ti2-E (session 6, F7) the owner pressed *Calibrate the camera*, read `Camera calibration done in
2.5 s` and saw no number: the pixel size, the rotation, the residual and the file were in a document the
page never read. The stand is ``demo-sample`` turned as m3 §20.1 turned it (camera 30° and mirrored,
plate 1.2°), registered under the 10X from its walls: the click at the 40X goes into
``calibrations/position-1.json`` and the pane says that document's numbers; on a plate that was not
registered from its walls the click goes into ``run.json`` and the pane says the run document's.

Module fixtures build ``run.json`` once through ``run_plugin`` (the seam). Each test copies it into its
own folder, opens a stand with the demo stage's ``Velocity`` raised on its own core (the demo moves at
10 mm/s, and a calibration would take a minute), and drives the view: a click is ``button.clicks += 1``.
The turret is turned by ``select(label, confirm=True)``, the operator's turn and not under test. The
pane's text is built here from the numbers **read from the document the page wrote** (FM-203), in the
words of the design written out (FM-45), and never by the code under test.
"""

from __future__ import annotations

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
from smc.plugins.slots import CameraCalibration, WellRef
from smc.run import RunDocument, StageRecord, StageStatus
from smc.ui.controller import PlatePage
from smc.ui.guide import local_time
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
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The demo turret's positions, from its labels in order: the 40X is position 1, the 10X position 2.
POSITION = {FORTY: 1, TEN: 2}
PLATE_DEG = 1.2
#: The demo-sample's A1 (a 96-well plate centred on the origin) and its pitch.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
INDEX = {"A1": (0, 0), "H12": (11, 7), "A12": (11, 0)}
#: Where the operator "centred" each well: up to 300 µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "H12": (-250.0, 150.0), "A12": (-100.0, 200.0)}
TOOLS = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
}


def _truth(well: str) -> tuple[float, float]:
    """A well's true centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    import math

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
        }
    )
    return tomli_w.dumps(data)


def _entry_points() -> list[EntryPoint]:
    return [EntryPoint(name=n, value=v, group=GROUP) for n, v in TOOLS.items()]


def _open_stand(folder: Path, *, dry_run: bool = False) -> Microscope:
    profile = folder / "stand.toml"
    profile.write_text(_profile_text(), encoding="utf-8")
    stand = Microscope.open(Profile.load(profile), dry_run=dry_run)
    # The demo stage moves at its `Velocity` property, 10 mm/s by default.
    stand.core.setProperty("XY", "Velocity", "100000")
    return stand


@pytest.fixture(scope="module")
def registered_json(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The registration under the 10X through ``run_plugin``: its calibration and ``plate-walls``, and no other calibration."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("page-card-registered")
    run_doc = folder / "run.json"
    stand = _open_stand(folder)
    try:
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


@pytest.fixture(scope="module")
def unregistered_json(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The 10X's calibration alone: a plate nobody registered from its walls, so the page has no offer (m5 §18.2 D2)."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("page-card-unregistered")
    run_doc = folder / "run.json"
    stand = _open_stand(folder)
    try:
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), run_doc
        )
        assert record.status is StageStatus.done
    finally:
        stand.close()
    return run_doc


@dataclass(frozen=True)
class Rig:
    """One test's page, its view on the Stand screen, its stand and the run document it may write."""

    page: PlatePage
    view: PlatePageView
    stand: Microscope
    run_doc: Path

    def document(self, label: str) -> Path:
        return self.run_doc.parent / "calibrations" / f"position-{POSITION[label]}.json"

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
        self.view.tick()


@pytest.fixture
def make_rig(tmp_path: Path) -> Iterator[Callable[..., Rig]]:
    """Opens the stand on a copy of the profile, a page over it on a copy of the run document, and a view on the Stand screen; closes what it opened."""
    opened: list[tuple[PlatePage, Microscope]] = []

    def make(source: Path) -> Rig:
        run_doc = tmp_path / "run.json"
        shutil.copy(source, run_doc)
        stand = _open_stand(tmp_path)
        page = PlatePage(stand, run_doc, entry_points=_entry_points())
        opened.append((page, stand))
        view = PlatePageView(page)
        view.show_step("stand")
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


def _record(path: Path) -> StageRecord:
    record = RunDocument.load(path).provider("camera-calibration")
    assert record is not None
    return record


def _written(path: Path) -> tuple[CameraCalibration, StageRecord]:
    """The calibration the document holds and its record, read back from the file the page wrote."""
    record = _record(path)
    assert record.status is StageStatus.done
    return CameraCalibration.model_validate(record.result), record


def _pane_for(document: Path, objective: str) -> str:
    """What the pane must say of ``document``, built from the numbers it holds and the words of m5 §23.2 D3 (FM-203, FM-45)."""
    found, record = _written(document)
    assert found.objective == objective
    assert record.finished_utc is not None
    mirror = "mirrored" if found.mirrored else "not mirrored"
    return "\n".join(
        [
            f"Camera calibration at {objective}, measured {local_time(record.finished_utc)}.",
            f"Pixel size {found.pixel_size_um:.4f} µm, rotation {found.rotation_deg:.1f}°, "
            f"{mirror}, residual {found.residual_um:.2f} µm, probe step "
            f"{found.step_um:g} µm.",
            f"Written to {document}.",
            *(f"assumed: {line}" for line in found.assumed),
        ]
    )


def test_the_stand_screen_says_the_numbers_the_40x_click_wrote_to_its_document(
    make_rig: Callable[..., Rig], registered_json: Path
) -> None:
    # F7: the plate is registered under the 10X, the operator turns to the 40X and presses *Calibrate
    # the camera*; the strip says `done in 0.2 s` and, before this, no number.
    rig = make_rig(registered_json)
    view = rig.view
    rig.turn(FORTY)
    plate_calibration = RunDocument.load(rig.run_doc).stages["camera-calibration"]

    # Before the click: the 10X's numbers are not shown as the 40X's (FM-216).
    assert view.calibration_numbers.object == ""
    assert not view.calibration_numbers.visible

    _click(view.calibrate_button)
    rig.finish()

    document = rig.document(FORTY)
    measured, _ = _written(document)
    # The demo is turned 30° and mirrored (m3 §20.1): the numbers the page now says are the ones the
    # document holds, and they are the demo's.
    assert measured.mirrored is True
    assert measured.rotation_deg == pytest.approx(30.0, abs=1.0)
    assert measured.pixel_size_um == pytest.approx(0.25, abs=0.01)
    assert view.calibration_numbers.object == _pane_for(document, FORTY)
    assert view.calibration_numbers.visible
    assert view.calibration_numbers.object.split("\n")[1].endswith(" µm.")
    # The known block still speaks of the run document's calibration (D19, §23.5), and the pane under
    # it names the file it reads: the run document is untouched (FM-216).
    assert view.known_block.object.split("\n")[1].endswith(f", not at {FORTY}")
    assert view.known_block.object.split("\n")[1].startswith(
        f"Camera: calibrated at {TEN} on "
    )
    assert (
        RunDocument.load(rig.run_doc).stages["camera-calibration"] == plate_calibration
    )
    # Back at the plate's own objective the pane says the run document's calibration, and its path.
    rig.turn(TEN)
    assert view.calibration_numbers.object == _pane_for(rig.run_doc, TEN)


def test_at_the_plates_own_objective_the_pane_says_the_run_documents_numbers(
    make_rig: Callable[..., Rig], registered_json: Path
) -> None:
    rig = make_rig(registered_json)

    assert rig.stand.require(ObjectiveTurret).current() == TEN
    assert rig.view.calibration_numbers.object == _pane_for(rig.run_doc, TEN)
    assert rig.view.calibration_numbers.visible
    # Calibrated at the objective in place: the button's box is gone, the numbers are not.
    assert not rig.view.calibrate_box.visible


def test_a_click_into_the_run_document_says_its_numbers_and_the_run_documents_path(
    make_rig: Callable[..., Rig], unregistered_json: Path
) -> None:
    # A plate not registered from its walls: no offer, so the click is not routed and goes into
    # `run.json`, replacing the 10X's calibration there.
    rig = make_rig(unregistered_json)
    view = rig.view
    assert view.calibration_numbers.object == _pane_for(rig.run_doc, TEN)
    rig.turn(FORTY)
    # The run document's calibration is the 10X's: not the 40X's numbers (FM-216).
    assert view.calibration_numbers.object == ""
    assert view.calibrate_box.visible

    _click(view.calibrate_button)
    rig.finish()

    assert not rig.document(FORTY).exists()
    assert view.calibration_numbers.object == _pane_for(rig.run_doc, FORTY)
    assert (
        view.calibration_numbers.object.split("\n")[2] == f"Written to {rig.run_doc}."
    )
    assert not view.calibrate_box.visible
