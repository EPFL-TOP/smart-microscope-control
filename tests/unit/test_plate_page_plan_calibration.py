"""What the page gives ``scan-plan`` as its camera calibration at an objective other than the plate's: the pure decision and the click (m5 §18.6 D24 to D27, #617).

No Panel here. The pure table is ``plan_calibration`` on the data it is given; the click is the real
controller over ``fake_microscope`` with a stub ``scan-plan`` (``Params = ScanPlanParams``, loaded
from its own entry-point value and never from the installed group: FM-112) that keeps what it was
given and ends in a ``PluginRunError``, so that no result is needed. The documents are seeded by
hand with the fake camera's frame (``(384, 512)``, non-square so that a transposed frame is
another frame: FM-73). Every expected text is the one of the design, written out whole and not
built by the code under test (FM-45). The turret is turned by the test, which is the operator's
turn; what a click does is asserted to leave the turret and the stage where they were (FM-14,
FM-88).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, ObjectiveTurret, XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.plugin import ScanPlanParams
from smc.plugins.slots import (
    CameraCalibration,
    PlateResult,
    ScanPlanResult,
    WallImages,
    WellFit,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import (
    CalibrationRead,
    HeldCalibration,
    PlanCalibration,
    PlatePage,
    plan_calibration,
)

T0 = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The fake turret's positions, read from its definition and not from the code under test: the
#: 40X is position 1, the 10X position 2, the 20X position 4 (the index plus one).
POSITION = {FORTY: 1, TEN: 2, TWENTY: 4}
LABELS = (FORTY, TEN, "Objective-2", TWENTY, "Objective-4", "Objective-5")
FRAME = (384, 512)
WHEN = "2026-11-25T10:20:30+00:00"


# --- the pure decision -----------------------------------------------------------------


def _read(
    objective: str | None = FORTY,
    *,
    path: str = "run/calibrations/position-1.json",
    measured: str = WHEN,
    problem: str | None = None,
    shape: tuple[int, int] | None = FRAME,
) -> CalibrationRead:
    return CalibrationRead(path, objective, measured, problem, image_shape_px=shape)


def _decide(**changes: Any) -> PlanCalibration:
    """``plan_calibration`` for a plan at the 40X after a registration under the 10X, with the 40X's document held."""
    arguments: dict[str, Any] = {
        "plate_objective": TEN,
        "objective": FORTY,
        "given": False,
        "labels": LABELS,
        "documents": (HeldCalibration(1, _read()),),
        "denied": None,
        "frame_px": FRAME,
    }
    arguments.update(changes)
    return plan_calibration(**arguments)


def _missing(objective: str, why: str) -> str:
    """The note of an unusable document, as m5 §18.6.3 words it, written out."""
    return (
        f"No camera calibration of {objective} can be used ({why}): Lay out the fields "
        "assumes the profile's pixel size, rotation 0 and no mirror, and the positions "
        "of cells in the fields will be wrong if the camera is turned or mirrored. To "
        f"get one: Stand screen, Calibrate the camera, with {objective} in place."
    )


def test_plan_calibration_gives_the_document_of_the_plans_objective() -> None:
    chosen = _decide()

    # Absolute, never resolved (FM-24, FM-50): the page's own path made absolute.
    assert chosen.path == Path("run/calibrations/position-1.json").absolute()
    assert chosen.path is not None
    assert chosen.path.is_absolute()
    assert chosen.note == (
        f"Lay out the fields uses the camera calibration of {FORTY} made on "
        "2026-11-25 (calibrations/position-1.json): the camera's turn and mirror are "
        "measured, not assumed."
    )
    assert chosen.refusal is None


def test_plan_calibration_takes_the_document_of_the_objectives_position_among_several() -> (
    None
):
    documents = (
        HeldCalibration(1, _read(FORTY, path="c/position-1.json")),
        HeldCalibration(2, _read(TEN, path="c/position-2.json")),
        HeldCalibration(4, _read(TWENTY, path="c/position-4.json")),
    )

    chosen = _decide(objective=TWENTY, documents=documents)

    assert chosen.path == Path("c/position-4.json").absolute()
    assert chosen.note.startswith(
        f"Lay out the fields uses the camera calibration of {TWENTY} "
    )
    assert "(calibrations/position-4.json)" in chosen.note


@pytest.mark.parametrize(
    "changes",
    [
        {"plate_objective": None},
        {"objective": TEN},
        {"objective": None},
        {"objective": ""},
        {"objective": "   "},
        {"given": True},
    ],
    ids=[
        "no offer (the plate was not registered from its walls)",
        "the plate's own objective",
        "no objective",
        "a blank objective",
        "a whitespace objective",
        "a calibration or a size typed",
    ],
)
def test_plan_calibration_gives_nothing_and_says_nothing_without_an_offer_for_the_plates_own_objective_for_no_objective_or_for_a_typed_choice(
    changes: dict[str, Any],
) -> None:
    assert _decide(**changes) == PlanCalibration()


@pytest.mark.parametrize(
    ("case", "why"),
    [
        ("no file", "none has been made"),
        (
            "a failed record",
            "the last calibration failed: the frame is dark or flat",
        ),
        (
            "the file of another objective",
            f"calibrations/position-1.json holds a calibration of {TEN}",
        ),
        (
            "a record that names no objective",
            "calibrations/position-1.json holds a calibration of an unknown objective",
        ),
        (
            "labels not read",
            f"the turret position of {FORTY} is not known: its labels could not be "
            "read, two positions carry that label, or it is past position 12",
        ),
        (
            "two positions with the label",
            f"the turret position of {FORTY} is not known: its labels could not be "
            "read, two positions carry that label, or it is past position 12",
        ),
        (
            "a position past 12",
            f"the turret position of {FORTY} is not known: its labels could not be "
            "read, two positions carry that label, or it is past position 12",
        ),
        (
            "a denied folder",
            "run/calibrations: the page cannot look into this folder "
            "(PermissionError: Permission denied); which objectives have a "
            "calibration is not known",
        ),
    ],
)
def test_plan_calibration_gives_nothing_and_says_why_for_each_document_it_cannot_use(
    case: str, why: str
) -> None:
    changes: dict[str, Any]
    if case == "no file":
        changes = {"documents": ()}
    elif case == "a failed record":
        changes = {
            "documents": (
                HeldCalibration(
                    1,
                    _read(
                        None,
                        measured="",
                        problem="the last calibration failed: the frame is dark or flat",
                        shape=None,
                    ),
                ),
            )
        }
    elif case == "the file of another objective":
        changes = {"documents": (HeldCalibration(1, _read(TEN)),)}
    elif case == "a record that names no objective":
        changes = {"documents": (HeldCalibration(1, _read(None)),)}
    elif case == "labels not read":
        changes = {"labels": ()}
    elif case == "two positions with the label":
        changes = {"labels": (FORTY, TEN, FORTY)}
    elif case == "a position past 12":
        changes = {"labels": (*(f"Objective-{n}" for n in range(12)), FORTY)}
    else:
        assert case == "a denied folder"
        changes = {
            "documents": (),
            "denied": (
                "run/calibrations: the page cannot look into this folder "
                "(PermissionError: Permission denied); which objectives have a "
                "calibration is not known"
            ),
        }

    chosen = _decide(**changes)

    assert chosen.path is None
    assert chosen.refusal is None
    assert chosen.note == _missing(FORTY, why)


def test_plan_calibration_uses_a_document_held_beside_a_denied_folder() -> None:
    # The denial says what the look could not reach: a document that was read is still the
    # position's (D22), and the denied line is only for a position with nothing held.
    chosen = _decide(denied="run/calibrations: the page cannot look into this folder")

    assert chosen.path is not None
    assert chosen.note.startswith("Lay out the fields uses the camera calibration of ")


@pytest.mark.parametrize(
    ("held_px", "camera_px"),
    [((1024, 1024), FRAME), ((512, 384), FRAME)],
    ids=["another size", "the transposed frame"],
)
def test_plan_calibration_refuses_a_document_of_another_frame_naming_both_frames(
    held_px: tuple[int, int], camera_px: tuple[int, int]
) -> None:
    chosen = _decide(
        documents=(HeldCalibration(1, _read(shape=held_px)),), frame_px=camera_px
    )

    assert chosen.path is None
    assert chosen.note == ""
    assert chosen.refusal == (
        f"the camera calibration of {FORTY} in calibrations/position-1.json is of a "
        f"frame of {held_px[0]} by {held_px[1]} px, the camera gives {camera_px[0]} by "
        f"{camera_px[1]} px (binning or the region changed): calibrate the camera at "
        f"{FORTY} again (Stand screen, Calibrate the camera)"
    )


def test_plan_calibration_compares_no_frame_when_the_frame_is_not_given() -> None:
    # The tick reads no camera (FM-145): a document of another frame is "used" in its note, and
    # only the click, which reads the frame, refuses it.
    other = (HeldCalibration(1, _read(shape=(1024, 1024))),)

    ticked = _decide(documents=other, frame_px=None)

    assert ticked.refusal is None
    assert ticked.path == Path("run/calibrations/position-1.json").absolute()
    # A read built without a frame compares nothing either.
    unknown = _decide(documents=(HeldCalibration(1, _read(shape=None)),))
    assert unknown.refusal is None
    assert unknown.path is not None


def test_plan_calibration_asks_its_questions_in_the_designs_order() -> None:
    # A typed choice wins over everything below it, the plate's own objective over the
    # documents, and a position that is not known over a document that is not there.
    broken = (HeldCalibration(1, _read(TEN, shape=(1, 1))),)
    assert _decide(given=True, documents=broken, labels=()) == PlanCalibration()
    assert _decide(objective=TEN, documents=broken, labels=()) == PlanCalibration()
    unknown = _decide(documents=(), labels=())
    assert unknown.note == _missing(
        FORTY,
        f"the turret position of {FORTY} is not known: its labels could not be read, "
        "two positions carry that label, or it is past position 12",
    )
    # The other objective's file is told before the frame is compared.
    other = _decide(documents=broken, frame_px=FRAME)
    assert other.refusal is None
    assert other.note == _missing(
        FORTY, f"calibrations/position-1.json holds a calibration of {TEN}"
    )


# --- the click, on a stub scan-plan --------------------------------------------------------


class StubScanPlan:
    """``scan-plan`` that keeps what it was given and ends the run with a refusal: the page's own parsing is what is under test."""

    name = provides = "scan-plan"
    summary = "A test tool."
    Params = ScanPlanParams
    Result = ScanPlanResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ("camera-calibration", "timings")
    seen: ClassVar[list[ScanPlanParams]] = []

    def run(
        self, microscope: Any, params: ScanPlanParams, ctx: RunContext
    ) -> ScanPlanResult:
        StubScanPlan.seen.append(params)
        raise PluginRunError("scan-plan", "the stub plans nothing")


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class StubTiming:
    """``timing`` that does nothing and succeeds: the first tool of *Measure the time*'s chain."""

    name = "timing"
    provides = "timings"
    summary = "A test tool."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        return NoResult()


class StubCalibration:
    """``camera-calibration`` that is never run: the page offers the tool by name, and a test that names it needs it loaded."""

    name = provides = "camera-calibration"
    summary = "A test tool."
    Params = NoParams
    Result = CameraCalibration
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    ran: ClassVar[list[str]] = []

    def run(
        self, microscope: Any, params: NoParams, ctx: RunContext
    ) -> CameraCalibration:
        StubCalibration.ran.append("ran")
        raise PluginRunError("camera-calibration", "the stub measures nothing")


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    StubScanPlan.seen = []
    StubCalibration.ran = []


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(name=c.name, value=f"{__name__}:{c.__name__}", group=GROUP)
        for c in (StubScanPlan, StubTiming, StubCalibration)
    ]


@pytest.fixture
def stand(fake_core: FakeCore, fake_microscope: Microscope) -> Microscope:
    """The demo-like stand whose camera gives the frame the seeded documents were made in."""
    fake_core.image_shape = FRAME
    return fake_microscope


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _write(
    path: Path,
    plugin: str,
    provides: str,
    result: dict[str, Any] | None,
    *,
    status: StageStatus = StageStatus.done,
    dry_run: bool = False,
    error: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt="a0",
            status=status,
            started_utc=T0,
            finished_utc=T0,
            profile="demo",
            dry_run=dry_run,
            smc_version="0",
            params=params if params is not None else {},
            result=result,
            error=error,
        ),
    )


def _walls_plate(*, by_eye: bool = False) -> PlateResult:
    """A plate as ``plate-walls`` leaves it, registered under the 10X; ``by_eye`` is one from references."""
    fit = WellFit(
        well="A1",
        recorded_xy_um=(0.0, 0.0),
        center_xy_um=(0.0, 0.0),
        diameter_um=6000.0,
        size_error=0.0,
        rms_um=0.1,
        n_points=100,
        walls=(),
    )
    walls = WallImages(
        objective=TEN,
        pixel_size_um=1.1,
        image_shape_px=(100, 100),
        search_um=1000.0,
        wells=(fit,),
        frames=8,
        folder=None,
        elapsed_s=10.0,
    )
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="references" if by_eye else "wall-images",
        residual_um=1.0,
        scale_error=None,
        n_refs=2,
        walls=None if by_eye else walls,
        objective=TEN,
    )


def _seed_plate(run_doc: Path, *, by_eye: bool = False) -> None:
    plate = _walls_plate(by_eye=by_eye)
    _write(
        run_doc,
        "plate-registration" if by_eye else "plate-walls",
        "plate",
        plate.model_dump(mode="json"),
        params={"plate": plate.plate},
    )


def _seed_document(
    run_doc: Path,
    label: str | None,
    *,
    position: int | None = None,
    shape: tuple[int, int] = FRAME,
    dry_run: bool = False,
    status: StageStatus = StageStatus.done,
    error: str | None = None,
) -> Path:
    """The calibration of ``label`` in its position's file, as a run of ``camera-calibration`` left it (the folder beside ``run.json``)."""
    where = POSITION[label] if position is None and label is not None else position
    assert where is not None
    path = run_doc.parent / "calibrations" / f"position-{where}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration = CameraCalibration(
        objective=label,
        image_shape_px=shape,
        matrix_um_per_px=((-0.2164, 0.1248), (-0.1246, -0.2164)),
        pixel_size_um=0.25,
        rotation_deg=29.9,
        mirrored=True,
        residual_um=0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=48.0,
        assumed=(),
    )
    _write(
        path,
        "camera-calibration",
        "camera-calibration",
        calibration.model_dump(mode="json") if status is StageStatus.done else None,
        dry_run=dry_run,
        status=status,
        error=error,
    )
    return path


def _page(microscope: Microscope, run_doc: Path) -> PlatePage:
    return PlatePage(microscope, run_doc, entry_points=_entry_points())


def _turn(microscope: Microscope, label: str) -> None:
    """The operator's turn of the turret: what the page does not do, and what these tests do not test."""
    microscope.require(ObjectiveTurret).select(label, confirm=True)


def _in_place(microscope: Microscope) -> str:
    return microscope.require(ObjectiveTurret).current()


def _lay_out(page: PlatePage, *pairs: str) -> str:
    """Press *Lay out the fields* with ``pairs`` and wait for the run; the outcome the strip would say."""
    page.run("scan-plan", ["wells=A1", *pairs])
    assert page.join(10.0)
    outcome = page.status().outcome
    assert outcome is not None
    return outcome


def _same(found: Path | None, expected: Path) -> bool:
    return found is not None and os.path.samefile(found, expected)


def test_lay_out_the_fields_at_another_objective_runs_scan_plan_with_that_objectives_document(
    stand: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    document = _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    page = _page(stand, run_doc)
    try:
        before = stand.require(XYStage).position_um()
        outcome = _lay_out(page)
        record = RunDocument.load(run_doc).provider("scan-plan")
    finally:
        page.close()

    (seen,) = StubScanPlan.seen
    assert _same(seen.calibration, document)
    assert seen.calibration is not None
    assert seen.calibration.is_absolute()
    # The plan's objective is the tool's to read: the page sent no `objective`, no size.
    assert seen.objective is None
    assert seen.pixel_size_um is None
    # The run is the record's: its params name the document, as a typed pair would.
    assert record is not None
    assert _same(Path(record.params["calibration"]), document)
    assert outcome == "Field layout failed: scan-plan: the stub plans nothing"
    # Nothing here moved the stage or turned the turret (FM-14, FM-88).
    assert _in_place(stand) == FORTY
    assert stand.require(XYStage).position_um() == before


def test_the_plates_own_registration_is_left_as_it_was_by_the_page_giving_a_document(
    stand: Microscope, run_doc: Path
) -> None:
    # FM-216: the position's document is given, never copied into `run.json`.
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    page = _page(stand, run_doc)
    try:
        before = RunDocument.load(run_doc)
        _lay_out(page)
        after = RunDocument.load(run_doc)
    finally:
        page.close()

    assert after.provider("camera-calibration") is None
    assert before.provider("camera-calibration") is None
    assert sorted(after.stages) == ["plate-walls", "scan-plan"]


def test_scan_plan_at_the_plates_own_objective_is_given_nothing_and_no_document_is_read(
    stand: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    _turn(stand, TEN)
    read: list[str] = []
    real = PlatePage._held_calibrations

    def spy(self: PlatePage) -> Any:
        read.append("read")
        return real(self)

    monkeypatch.setattr(PlatePage, "_held_calibrations", spy)
    page = _page(stand, run_doc)
    try:
        _lay_out(page)
    finally:
        page.close()

    (seen,) = StubScanPlan.seen
    assert seen.calibration is None
    assert read == []


@pytest.mark.parametrize(
    "case",
    [
        "a typed calibration",
        "a typed pixel size",
        "a typed objective of the 20X",
        "a typed objective equal to the plate's",
        "a typed null calibration",
    ],
)
def test_a_typed_calibration_pixel_size_or_objective_is_the_operators_and_wins(
    stand: Microscope, run_doc: Path, case: str, tmp_path: Path
) -> None:
    _seed_plate(run_doc)
    forty = _seed_document(run_doc, FORTY)
    twenty = _seed_document(run_doc, TWENTY)
    own = tmp_path / "mine.json"
    _turn(stand, FORTY)
    pairs = {
        "a typed calibration": [f"calibration={own}"],
        "a typed pixel size": ["pixel_size_um=0.3"],
        "a typed objective of the 20X": [f"objective={TWENTY}"],
        "a typed objective equal to the plate's": [f"objective={TEN}"],
        "a typed null calibration": ["calibration=null"],
    }[case]
    page = _page(stand, run_doc)
    try:
        outcome = _lay_out(page, *pairs)
    finally:
        page.close()

    # Never a refusal of the page's own, and never the tool's `given twice` or `not both`: a
    # pair the page makes is not met by one the operator typed.
    assert outcome == "Field layout failed: scan-plan: the stub plans nothing"
    (seen,) = StubScanPlan.seen
    if case == "a typed calibration":
        assert seen.calibration == own
        assert seen.pixel_size_um is None
    elif case == "a typed pixel size":
        assert seen.calibration is None
        assert seen.pixel_size_um == 0.3
    elif case == "a typed objective of the 20X":
        # The plan's objective is the typed one: its document, not the turret's.
        assert seen.objective == TWENTY
        assert _same(seen.calibration, twenty)
    elif case == "a typed objective equal to the plate's":
        # The plate's own objective: its calibration is `run.json`'s, nothing is added.
        assert seen.objective == TEN
        assert seen.calibration is None
    else:
        # `calibration=null` is none (the parser reads it so): the page's pair replaces it.
        assert seen.calibration is not None
        assert _same(seen.calibration, forty)


@pytest.mark.parametrize("document_at_the_new_one", [True, False])
def test_the_click_gives_the_document_of_the_objective_the_turret_reads_now_not_the_ticks(
    stand: Microscope, run_doc: Path, document_at_the_new_one: bool
) -> None:
    # FM-177, FM-151: the view and the tick saw the 40X; the operator turned the turret at the
    # handset and no tick came before the click.
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    twenty = _seed_document(run_doc, TWENTY) if document_at_the_new_one else None
    _turn(stand, FORTY)
    page = _page(stand, run_doc)
    try:
        page.view()
        assert page.status().objective == FORTY
        _turn(stand, TWENTY)
        _lay_out(page)
    finally:
        page.close()

    (seen,) = StubScanPlan.seen
    if twenty is None:
        assert seen.calibration is None
    else:
        assert _same(seen.calibration, twenty)
    assert _in_place(stand) == TWENTY


def test_a_document_of_another_frame_is_refused_before_anything_is_written(
    fake_core: FakeCore, stand: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    fake_core.image_shape = (1024, 1024)
    page = _page(stand, run_doc)
    try:
        before = run_doc.read_bytes()
        position = stand.require(XYStage).position_um()
        page.run("scan-plan", ["wells=A1"])
        status = page.status()
        after = run_doc.read_bytes()
        moved = stand.require(XYStage).position_um()
    finally:
        page.close()

    assert status.outcome == (
        f"Field layout: the camera calibration of {FORTY} in "
        "calibrations/position-1.json is of a frame of 384 by 512 px, the camera "
        "gives 1024 by 1024 px (binning or the region changed): calibrate the camera "
        f"at {FORTY} again (Stand screen, Calibrate the camera)"
    )
    # Nothing started and nothing written: the usable plan is not replaced by a failure (FM-206).
    assert status.running is None
    assert StubScanPlan.seen == []
    assert after == before
    assert moved == position
    assert _in_place(stand) == FORTY


@pytest.mark.parametrize("how", ["a frame of 0 by 0", "a read that raises"])
def test_a_frame_that_cannot_be_read_adds_nothing_and_refuses_nothing(
    fake_core: FakeCore,
    stand: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    if how == "a frame of 0 by 0":
        fake_core.image_shape = (0, 0)
    else:
        camera = stand.require(Camera)

        def broken() -> tuple[int, int]:
            raise RuntimeError("the camera does not answer\nat all")

        monkeypatch.setattr(camera, "image_shape", broken)
    page = _page(stand, run_doc)
    try:
        with caplog.at_level(logging.WARNING, logger="smc.ui"):
            outcome = _lay_out(page)
    finally:
        page.close()

    # Unknown beats guessed: the tool decides, as it did before this page gave it anything.
    assert outcome == "Field layout failed: scan-plan: the stub plans nothing"
    (seen,) = StubScanPlan.seen
    assert seen.calibration is None
    warnings = [r.getMessage() for r in caplog.records if r.name == "smc.ui"]
    if how == "a read that raises":
        assert warnings == [
            "camera: its frame could not be read before a scan plan: RuntimeError: "
            "the camera does not answer at all"
        ]
    else:
        assert warnings == []


@pytest.mark.parametrize(
    "case",
    [
        "no file",
        "a failed record",
        "the file of another objective",
        "a folder named like the file",
    ],
)
def test_an_unusable_document_adds_nothing_and_refuses_nothing(
    stand: Microscope, run_doc: Path, case: str
) -> None:
    _seed_plate(run_doc)
    _turn(stand, FORTY)
    if case == "a failed record":
        _seed_document(
            run_doc, FORTY, status=StageStatus.failed, error="the frame is dark"
        )
    elif case == "the file of another objective":
        _seed_document(run_doc, TEN, position=1)
    elif case == "a folder named like the file":
        (run_doc.parent / "calibrations" / "position-1.json").mkdir(parents=True)
    page = _page(stand, run_doc)
    try:
        before = RunDocument.load_or_new(run_doc).stages.keys()
        outcome = _lay_out(page)
    finally:
        page.close()

    # The tool's own `assumed` lines say the rest after the plan; the page refuses nothing.
    assert outcome == "Field layout failed: scan-plan: the stub plans nothing"
    (seen,) = StubScanPlan.seen
    assert seen.calibration is None
    assert "scan-plan" not in before


def test_a_dry_run_page_gives_a_real_document_and_a_dry_run_document_is_a_problem(
    fake_core: FakeCore,
    stand: Microscope,
    run_doc: Path,
    request: pytest.FixtureRequest,
) -> None:
    # FM-182: a dry-run page reads a real document without complaint; a document made in a dry
    # run is, for a real plan, the tool's own `problem` (`unusable`: a dry run read by a real
    # one), and nothing is added.
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    request.addfinalizer(dry.close)
    _seed_plate(run_doc)
    real = _seed_document(run_doc, FORTY)
    _seed_document(run_doc, TWENTY, dry_run=True)
    # A dry-run facade does not turn the turret: the operator's turn is made on the same core.
    _turn(stand, FORTY)
    dry_page = _page(dry, run_doc)
    try:
        _lay_out(dry_page)
    finally:
        dry_page.close()
    _turn(stand, TWENTY)
    real_page = _page(stand, run_doc)
    try:
        _lay_out(real_page)
    finally:
        real_page.close()

    first, second = StubScanPlan.seen
    assert _same(first.calibration, real)
    assert second.calibration is None


def test_the_measure_the_time_chain_gives_scan_plan_the_document_too(
    stand: Microscope, run_doc: Path
) -> None:
    # `scan-plan` is prepared on the worker after `timing`, by the same `_prepare` (D24).
    _seed_plate(run_doc)
    document = _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    page = _page(stand, run_doc)
    try:
        page.run_chain(("timing", "scan-plan"), {"scan-plan": ("wells=A1",)})
        assert page.join(10.0)
    finally:
        page.close()

    (seen,) = StubScanPlan.seen
    assert _same(seen.calibration, document)
    assert _in_place(stand) == FORTY


def test_the_next_steps_run_gets_the_document_at_the_click_and_its_command_is_unchanged(
    stand: Microscope, run_doc: Path
) -> None:
    # D27: the next step's pairs are a copy read at a view and run as drawn; the document is
    # given at the click, and the pairs and the command carry no `calibration=`.
    _seed_plate(run_doc)
    document = _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    page = _page(stand, run_doc)
    try:
        page.select_text("A1")
        step = page.view().next_step
        assert step.plugin == "scan-plan"
        assert step.params == ("wells=A1",)
        assert "calibration" not in step.command
        page.run(step.plugin, list(step.params))
        assert page.join(10.0)
    finally:
        page.close()

    (seen,) = StubScanPlan.seen
    assert _same(seen.calibration, document)


def test_no_other_tool_reads_the_documents_for_the_fill(
    stand: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[str] = []
    real = PlatePage._held_calibrations

    def spy(self: PlatePage) -> Any:
        read.append("read")
        return real(self)

    monkeypatch.setattr(PlatePage, "_held_calibrations", spy)
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    page = _page(stand, run_doc)
    try:
        # Another tool: the fill is `scan-plan`'s only.
        page.run("camera-calibration", [])
        assert page.join(10.0)
        assert StubCalibration.ran == ["ran"]
        assert read == []
    finally:
        page.close()
    # A plate registered by eye has no offer: its calibration is `run.json`'s, nothing is read.
    by_eye = run_doc.parent / "eye" / "run.json"
    by_eye.parent.mkdir()
    _seed_plate(by_eye, by_eye=True)
    _seed_document(by_eye, FORTY)
    other = _page(stand, by_eye)
    try:
        _lay_out(other)
    finally:
        other.close()

    assert read == []
    (seen,) = StubScanPlan.seen
    assert seen.calibration is None


def test_the_pair_is_made_only_for_scan_plan_run_into_the_pages_own_document(
    stand: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `doc_path` names the document of another run (the block's calibration into a position's
    # file): the page's own plan is not that run's, and the decision is not asked for it.
    _seed_plate(run_doc)
    _seed_document(run_doc, FORTY)
    _turn(stand, FORTY)
    asked: list[str] = []
    real = PlatePage._plan_calibration_for

    def spy(self: PlatePage, doc: RunDocument, parsed: ScanPlanParams) -> Any:
        asked.append("asked")
        return real(self, doc, parsed)

    monkeypatch.setattr(PlatePage, "_plan_calibration_for", spy)
    page = _page(stand, run_doc)
    try:
        page._prepare("scan-plan", ["wells=A1"], doc_path=run_doc.parent / "other.json")
        assert asked == []
        prepared = page._prepare("scan-plan", ["wells=A1"])
    finally:
        page.close()

    assert asked == ["asked"]
    assert not isinstance(prepared, str)
    assert prepared.parsed.calibration is not None  # type: ignore[attr-defined]
