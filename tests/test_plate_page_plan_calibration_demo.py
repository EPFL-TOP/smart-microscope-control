"""*Lay out the fields* at another objective plans with that objective's camera calibration, on the demo-sample stand, through the view's own handlers (m5 §18.6 D24 to D27, #617).

The stand is ``demo-sample`` turned as m3 §20.1 turned it (camera 30° and mirrored, plate 1.2°),
registered under the 10X from its walls, and **with no calibration document**: the page makes the
40X's on the Stand screen. Before #617 the plan at the 40X used the profile's size, rotation 0 and
no mirror (FM-216's second half): a camera turned 30° and mirrored planned as an upright one.

A module fixture builds ``run.json`` once through ``run_plugin`` (the seam): the 10X's calibration and
``plate-walls`` on A1, H12 and A12. Each test copies ``run.json`` and a copy of the profile into its
own folder, opens a stand with the demo stage's ``Velocity`` raised on its own core (the demo moves
at 10 mm/s, and a calibration would take a minute), and drives the view: a click is
``button.clicks += 1``. The turret is turned by ``select(label, confirm=True)``, the operator's turn
and not under test. The expected matrix is read from the document the page made, never from the
code that planned with it (FM-203), and every text is the design's, written out whole (FM-45).
"""

from __future__ import annotations

import math
import os
import shlex
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
from smc.plugins.geometry import pixel_matrix
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.registry import GROUP
from smc.plugins.slots import CameraCalibration, ScanPlanResult, WellRef
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage
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
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
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
    folder = tmp_path_factory.mktemp("page-plan-calibration")
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

    def lay_out(self) -> ScanPlanResult:
        """Select A1, press *Lay out the fields*, wait, and the plan it left in the run document."""
        self.page.select_text("A1")
        self.view.refresh()
        self.view.tick()
        _click(self.view.layout_button)
        self.finish()
        record = RunDocument.load(self.run_doc).stages["scan-plan"]
        assert record.status is StageStatus.done, self.page.status().outcome
        return ScanPlanResult.model_validate(record.result)


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
        view.show_step("wells")
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


def _determinant(matrix: tuple[tuple[float, float], tuple[float, float]]) -> float:
    (a, b), (c, d) = matrix
    return a * d - b * c


def _used(objective: str, position: int, document: Path) -> str:
    """The Wells screen's note of a usable document, as m5 §18.6.3 words it: the date is the one the document's record ends (UTC, as ``calibration_from`` says it)."""
    finished = _record(document).finished_utc
    assert finished is not None
    return (
        f"Lay out the fields uses the camera calibration of {objective} made on "
        f"{finished.date().isoformat()} (calibrations/position-{position}.json): the "
        "camera's turn and mirror are measured, not assumed."
    )


def _missing(objective: str, why: str) -> str:
    """The Wells screen's note of a document that cannot be used, as m5 §18.6.3 words it."""
    return (
        f"No camera calibration of {objective} can be used ({why}): Lay out the fields "
        "assumes the profile's pixel size, rotation 0 and no mirror, and the positions "
        "of cells in the fields will be wrong if the camera is turned or mirrored. To "
        f"get one: Stand screen, Calibrate the camera, with {objective} in place."
    )


def _given(view: PlatePageView) -> list[str]:
    """The ``calibration=`` values of the command under *Lay out the fields*."""
    words = shlex.split(view.layout_command.object)
    return [
        words[index + 1][len("calibration=") :]
        for index, word in enumerate(words[:-1])
        if word == "--param" and words[index + 1].startswith("calibration=")
    ]


def test_the_wells_screen_at_the_40x_plans_with_the_40x_calibration_the_page_made(
    make_rig: Callable[..., Rig],
) -> None:
    # Acceptance 1 on the demo stand: the plate is registered under the 10X, the operator turns
    # the turret to the 40X, presses the Stand screen's *Calibrate the camera* (which keeps the
    # plate's calibration) and then *Lay out the fields* on the Wells screen.
    rig = make_rig()
    view, turret = rig.view, rig.stand.require(ObjectiveTurret)
    rig.turn(FORTY)
    plate_calibration = (
        RunDocument.load(rig.run_doc).stages["camera-calibration"].result
    )
    _click(view.calibrate_button)
    rig.finish()
    document = rig.document(FORTY)
    assert _record(document).status is StageStatus.done
    measured = _calibration(document)
    assert measured.objective == FORTY

    # Before the click the Wells screen says which document the plan uses, and the command under
    # it is the call the click makes, with that document once.
    rig.page.select_text("A1")
    view.refresh()
    view.tick()
    assert view.layout_calibration_text.object == _used(FORTY, 1, document)
    (given,) = _given(view)
    assert os.path.samefile(given, document)

    plan = rig.lay_out()

    # The plan rests on the 40X's document: its matrix, turned and mirrored (FM-203, FM-73), and
    # not an upright camera of the same size. The upright matrix, diag(ps, -ps), has the
    # determinant -ps²; the mirror flips its sign, so a mirrored camera's is +ps².
    assert plan.objective == FORTY
    assert plan.plate_objective == TEN
    assert plan.matrix_source == "camera-calibration"
    assert plan.matrix_um_per_px == measured.matrix_um_per_px
    assert plan.matrix_um_per_px != pixel_matrix(measured.pixel_size_um)
    assert _determinant(pixel_matrix(measured.pixel_size_um)) < 0.0
    assert _determinant(plan.matrix_um_per_px) == pytest.approx(
        measured.pixel_size_um**2
    )
    assert plan.calibration_from is not None
    assert os.path.samefile(plan.calibration_from, document)
    # Nothing assumed the camera straight (the line that said so is the quiet failure this fixes).
    assert not [line for line in plan.assumed if "rotation 0" in line]
    assert not [line for line in plan.assumed if "camera calibration not used" in line]
    # The registration is the 10X's and still usable, and the page did not copy the 40X's
    # calibration into `run.json` (FM-216): its camera-calibration record is the one it was.
    view_after = rig.page.view()
    assert view_after.registered is not None
    assert view_after.camera_calibration is not None
    assert view_after.camera_calibration.objective == TEN
    assert (
        RunDocument.load(rig.run_doc).stages["camera-calibration"].result
        == plate_calibration
    )
    # Nothing here turned the turret (FM-14).
    assert turret.current() == FORTY


def test_without_a_document_the_plan_assumes_rotation_0_and_the_note_said_so_before_the_click(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    view = rig.view
    rig.turn(FORTY)
    rig.page.select_text("A1")
    view.refresh()
    view.tick()

    # Before the click: what the plan will assume, and how to get a document. Nothing is added.
    assert view.layout_calibration_text.object == _missing(FORTY, "none has been made")
    assert _given(view) == []
    assert view.layout_button.disabled is False

    plan = rig.lay_out()

    # The tool's own `assumed` lines say it after the plan, as before this page gave it anything.
    assert plan.calibration_from is None
    assert plan.matrix_source == "profile"
    assert (
        f"no camera calibration for {FORTY}: rotation 0 and no mirror assumed"
        in plan.assumed
    )
    assert plan.matrix_um_per_px == pixel_matrix(plan.pixel_size_um)
    assert rig.page.view().registered is not None


def test_a_dry_run_page_plans_with_a_real_document(
    make_rig: Callable[..., Rig], tmp_path: Path
) -> None:
    # FM-182: a page served with --dry-run reads a document a real run made, and plans with it.
    # The document is made by a real stand first (a dry run cannot measure the camera), into the
    # file of the 40X's position, and the dry-run page is opened over the same folder.
    profile = tmp_path / "stand.toml"
    profile.write_text(_profile_text(), encoding="utf-8")
    real = Microscope.open(Profile.load(profile))
    document = tmp_path / "calibrations" / "position-1.json"
    document.parent.mkdir()
    try:
        real.core.setProperty("XY", "Velocity", "100000")
        real.require(ObjectiveTurret).select(FORTY, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), real, CameraCalibrationParams(), document
        )
        assert record.status is StageStatus.done
    finally:
        real.close()

    rig = make_rig(dry_run=True)
    # A dry-run facade does not turn the turret: the operator's turn is made on the core itself.
    rig.stand.core.setStateLabel("Objective", FORTY)
    rig.view.tick()
    assert rig.stand.require(ObjectiveTurret).current() == FORTY
    rig.page.select_text("A1")
    rig.view.refresh()
    rig.view.tick()
    assert rig.view.layout_calibration_text.object == _used(FORTY, 1, document)

    plan = rig.lay_out()

    assert plan.matrix_source == "camera-calibration"
    assert plan.calibration_from is not None
    assert os.path.samefile(plan.calibration_from, document)
    assert plan.matrix_um_per_px == _calibration(document).matrix_um_per_px
