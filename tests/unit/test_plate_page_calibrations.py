"""The plate page's calibrations of each objective, in the controller: the run into a document of its own, the refusals, the held documents and the empty field's default (m5 §18.6, #501).

No Panel here. The page is the real controller over ``fake_microscope`` with a stub
``camera-calibration`` (the ``CameraCalibration`` of the turret's current label, no stage),
loaded from its own entry-point value and never from the installed group (FM-112); the real
tool is used where its own words are pinned. Every expected text is the one of the design,
asserted whole (FM-45). Nothing here moves a real stage or turns a real turret: the turret is
turned by the test, which is the operator's turn and not under test, and what a click does is
asserted to leave it where it was (FM-14).
"""

from __future__ import annotations

import os
import shlex
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource, ObjectiveTurret, Shutter
from smc.hardware.errors import HardwareError, MicroscopeHaltedError
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.params import parse_params
from smc.plugins.registry import GROUP
from smc.plugins.runner import run_plugin
from smc.plugins.slots import (
    CameraCalibration,
    MeasuredOffset,
    ObjectiveOffsetsResult,
    OffsetWell,
    PlateResult,
    TargetMeasure,
    WallImages,
    WellFit,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import (
    NOT_CONNECTED,
    CalibrationRead,
    PlatePage,
)
from smc.ui.guide import (
    CALIBRATE_DRY_RUN,
    CALIBRATE_LIGHT_OFF,
    CALIBRATE_NAMES_NONE,
    CALIBRATE_NO_OFFER,
    CALIBRATE_NOT_KNOWN,
    CALIBRATE_REUSE_ROUTED,
    OFFSETS_NO_DOCUMENT,
    OFFSETS_TOO_MANY,
    OFFSETS_TURRET_UNREAD,
    SCAN_HALTED,
    busy_note,
    calibrate_changed,
    calibrate_is_the_plates,
    calibrate_measured_other,
    calibrate_position_unknown,
    calibration_failed,
)

T0 = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The fake turret's positions, read from its definition and not from the code under test: the
#: 40X is position 1, the 10X position 2, the 20X position 4 (the index plus one).
POSITION = {FORTY: 1, TEN: 2, TWENTY: 4}


# --- a stub tool, a slow one and a turret double -------------------------------------


class StubOffsetsParams(PluginParams):
    targets: tuple[str, ...]
    calibrations: tuple[Path, ...]
    confirm_turns: bool = False


def _measure(objective: str, shift: tuple[float, float]) -> TargetMeasure:
    fit = WellFit(
        well="A1",
        recorded_xy_um=shift,
        center_xy_um=shift,
        diameter_um=6000.0,
        size_error=0.0,
        rms_um=0.1,
        n_points=100,
        walls=(),
    )
    return TargetMeasure(
        objective=objective,
        calibration="cal.json",
        calibration_measured="2026-11-10T10:00:00+00:00",
        pixel_size_um=2.5,
        z_after_turn_um=None,
        wells=(
            OffsetWell(well="A1", origin_xy_um=(0.0, 0.0), delta_xy_um=shift, fit=fit),
        ),
        shift_xy_um=shift,
        spread_um=0.03,
        max_spread_um=2.0,
        frames=24,
        folder=None,
    )


class StubOffsets:
    """``objective-offsets`` without a stage: it keeps what it was given and returns a result of the targets."""

    name = provides = "objective-offsets"
    summary = "A test tool."
    Params = StubOffsetsParams
    Result = ObjectiveOffsetsResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[StubOffsetsParams]] = []

    def run(
        self, microscope: Any, params: StubOffsetsParams, ctx: RunContext
    ) -> ObjectiveOffsetsResult:
        StubOffsets.seen.append(params)
        return ObjectiveOffsetsResult(
            plate_objective=TEN,
            reference=TEN,
            reference_in_profile=False,
            measures=tuple(_measure(label, (1.0, 2.0)) for label in params.targets),
            offsets=tuple(
                MeasuredOffset(objective=label, xy_offset_um=(1.0, 2.0))
                for label in params.targets
            ),
            objective_in_place=params.targets[-1],
            elapsed_s=1.0,
        )


class StubCalibrationParams(PluginParams):
    """The tool's own parameters: what the block's form sends."""

    step_um: float | None = None
    settle_ms: float = 0.0


def _calibration(
    objective: str | None, assumed: tuple[str, ...] = ()
) -> CameraCalibration:
    return CameraCalibration(
        objective=objective,
        image_shape_px=(384, 512),
        matrix_um_per_px=((0.5, 0.0), (0.0, -0.5)),
        pixel_size_um=0.5,
        rotation_deg=0.0,
        mirrored=True,
        residual_um=0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=48.0,
        assumed=assumed,
    )


class StubCalibration:
    """``camera-calibration`` without a stage: the calibration of the turret's current label, or the failure the test sets."""

    name = provides = "camera-calibration"
    summary = "A test tool."
    Params = StubCalibrationParams
    Result = CameraCalibration
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[StubCalibrationParams]] = []
    fails: ClassVar[str | None] = None
    assumed: ClassVar[tuple[str, ...]] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event | None] = None
    #: The operator's turn at the handset **during the run**, between the click's read of the
    #: turret and the tool's own: the label the tool then reads is this one (m5 §18.6 D21).
    turns_to: ClassVar[str | None] = None

    def run(
        self, microscope: Any, params: StubCalibrationParams, ctx: RunContext
    ) -> CameraCalibration:
        StubCalibration.seen.append(params)
        StubCalibration.started.set()
        if StubCalibration.turns_to is not None:
            microscope.require(ObjectiveTurret).select(
                StubCalibration.turns_to, confirm=True
            )
        if StubCalibration.release is not None:
            # In the tool's own way: a STOP is seen between two steps, the stage stays put.
            deadline = time.monotonic() + 10.0
            while not StubCalibration.release.is_set() and time.monotonic() < deadline:
                ctx.raise_if_stopped()
                time.sleep(0.01)
        if StubCalibration.fails is not None:
            raise PluginRunError("camera-calibration", StubCalibration.fails)
        label = microscope.require(ObjectiveTurret).current()
        return _calibration(label, StubCalibration.assumed)


class SlowParams(PluginParams):
    pass


class SlowResult(PluginResult):
    pass


class Slow:
    """A tool that waits until the test lets it go: an activity that is going."""

    name = provides = "slow"
    summary = "A test tool that waits."
    Params = SlowParams
    Result = SlowResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> SlowResult:
        Slow.started.set()
        Slow.release.wait(10.0)
        return SlowResult()


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    StubOffsets.seen = []
    StubCalibration.seen = []
    StubCalibration.fails = None
    StubCalibration.assumed = ()
    StubCalibration.turns_to = None
    StubCalibration.started = threading.Event()
    StubCalibration.release = None
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()  # a worker a failed test left behind ends
        if StubCalibration.release is not None:
            StubCalibration.release.set()


class Turret:
    """A turret the test sets: the label in place and the positions' labels, or a read that fails."""

    def __init__(
        self, labels: list[str], now: str | None, *, labels_fail: bool = False
    ) -> None:
        self.labels_ = labels
        self.now = now
        self.labels_fail = labels_fail
        self.moves: list[str] = []

    def labels(self) -> list[str]:
        if self.labels_fail:
            raise HardwareError("the turret's positions cannot be read")
        return list(self.labels_)

    def current(self) -> str:
        if self.now is None:
            raise HardwareError("the turret is between two positions")
        return self.now

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.moves.append(label)
        self.now = label
        return label

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0

    def offset(self, label: str) -> Any:
        raise NotImplementedError


class AutoShutter:
    """The core's shutter and its auto-shutter switch, as the test sets them (m5 §18.6 D20).

    On the Ti2-E the lamp (``DiaLamp``) is the core's shutter and auto-shutter is on: the lamp
    reads closed at rest and Micro-Manager opens it around the camera's own snap.
    """

    def __init__(self, auto: bool | Exception) -> None:
        self.auto = auto

    def is_open(self) -> bool:
        return False

    def set_open(self, open_: bool) -> bool:
        return open_

    def auto_shutter(self) -> bool:
        if isinstance(self.auto, Exception):
            raise self.auto
        return self.auto

    def set_auto_shutter(self, on: bool) -> bool:
        self.auto = on
        return on


class Light:
    """A light source the test sets: its switch reads on, off, nothing smc knows, or raises (D20).

    The page reads ``is_on()`` at a tick and again at a click: ``reading`` is what the next read
    says, so that a test turns the light off **between** the two, with no tick in between.
    """

    def __init__(self, reading: bool | Exception | None) -> None:
        self.reading = reading

    def device(self) -> str:
        return "LED"

    def intensity_knob(self) -> Any:
        return None

    def intensity_native(self) -> float | None:
        return None

    def set_intensity_native(self, value: float) -> float:
        return value

    def is_on(self) -> bool | None:
        if isinstance(self.reading, Exception):
            raise self.reading
        return self.reading

    def set_on(self, on: bool) -> bool:
        self.reading = on
        return on


# --- records written out by hand -------------------------------------------------------


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(name=c.name, value=f"{__name__}:{c.__name__}", group=GROUP)
        for c in (StubCalibration, StubOffsets, Slow)
    ]


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _wall_fit(well: str, centre: tuple[float, float]) -> WellFit:
    return WellFit(
        well=well,
        recorded_xy_um=centre,
        center_xy_um=centre,
        diameter_um=6000.0,
        size_error=0.0,
        rms_um=0.1,
        n_points=100,
        walls=(),
    )


def _walls_plate(*, by_eye: bool = False) -> PlateResult:
    """A plate as ``plate-walls`` leaves it, registered under the 10X; ``by_eye`` is one from references."""
    walls = WallImages(
        objective=TEN,
        pixel_size_um=1.1,
        image_shape_px=(100, 100),
        search_um=1000.0,
        wells=(_wall_fit("A1", (0.0, 0.0)), _wall_fit("H12", (99000.0, -63000.0))),
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


def _write(
    path: Path,
    plugin: str,
    provides: str,
    result: dict[str, Any] | None,
    *,
    status: StageStatus = StageStatus.done,
    minute: int = 0,
    profile: str = "demo",
    dry_run: bool = False,
    error: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    then = T0 + timedelta(minutes=minute)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt=f"a{minute}",
            status=status,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params=params if params is not None else {},
            result=result,
            error=error,
        ),
    )


def _seed_calibration(
    path: Path,
    objective: str | None,
    *,
    assumed: tuple[str, ...] = (),
    profile: str = "demo",
    status: StageStatus = StageStatus.done,
    error: str | None = None,
    minute: int = 0,
) -> None:
    """A document that holds the camera calibration of ``objective``, as a run of the tool left it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(
        path,
        "camera-calibration",
        "camera-calibration",
        _calibration(objective, assumed).model_dump(mode="json")
        if status is StageStatus.done
        else None,
        profile=profile,
        status=status,
        error=error,
        minute=minute,
    )


def _folder(run_doc: Path) -> Path:
    return run_doc.parent / "calibrations"


def _page(microscope: Microscope, run_doc: Path) -> PlatePage:
    return PlatePage(microscope, run_doc, entry_points=_entry_points())


def _turn(microscope: Microscope, label: str) -> None:
    """The operator's turn of the turret: what the page does not do, and what these tests do not test."""
    microscope.require(ObjectiveTurret).select(label, confirm=True)


def _in_place(microscope: Microscope) -> str:
    return microscope.require(ObjectiveTurret).current()


def _dry_microscope(fake_core: FakeCore) -> Microscope:
    return Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )


#: The two buttons that calibrate into a document of the objective's own: the block's (always) and
#: the Stand screen's (for a plate registered under another objective, m5 §18.6 D19). What both
#: refuse is the same text from the same helpers (FM-169), and the tests of it run for each.
BUTTONS = ["the block's button", "the Stand screen's button"]


def _click(
    page: PlatePage, button: str, expected: str, pairs: tuple[str, ...] = ()
) -> None:
    """Press ``button`` once, with the label it showed, and do not wait."""
    if button == BUTTONS[0]:
        page.calibrate_objective(expected, pairs)
    else:
        page.calibrate_camera(expected, pairs)


def _calibrate(
    page: PlatePage,
    expected: str,
    pairs: tuple[str, ...] = (),
    button: str = BUTTONS[0],
) -> str:
    """Press the button (the block's by default) and wait for the run; the outcome the strip would say."""
    _click(page, button, expected, pairs)
    assert page.join(10.0)
    outcome = page.status().outcome
    assert outcome is not None
    return outcome


# --- D14: the run into a document of its own (FM-216) ---------------------------------


def test_calibrating_another_objective_leaves_run_json_byte_identical_and_the_plate_registered(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        stages_before = sorted(RunDocument.load(run_doc).stages)
        outcome = _calibrate(page, FORTY, ("step_um=40",))
        view = page.view()
        after = run_doc.read_bytes()
    finally:
        page.close()

    document = _folder(run_doc) / "position-1.json"
    record = RunDocument.load(document).provider("camera-calibration")
    assert outcome.startswith("Camera calibration done in ")
    # The page's own document is the registration's: not read for the run, not written by it.
    assert after == before
    assert sorted(RunDocument.load(run_doc).stages) == stages_before
    assert view.registered is not None
    assert view.offsets is not None
    assert [item.state for item in view.checklist if item.slot == "plate"] == ["ok"]
    # The calibration is in the file of the label's position, with the label inside it.
    assert record is not None
    assert record.status is StageStatus.done
    assert CameraCalibration.model_validate(record.result).objective == FORTY
    # The form's fields reach the tool, and nothing in the click turned the turret (FM-14).
    assert [p.step_um for p in StubCalibration.seen] == [40.0]
    assert _in_place(fake_microscope) == FORTY
    assert [h.position for h in view.calibrations] == [1]


@pytest.mark.parametrize("label", [FORTY, TWENTY], ids=["position-1", "position-4"])
def test_the_document_is_the_labels_index_plus_one_for_a_label_after_empty_positions(
    fake_microscope: Microscope, run_doc: Path, label: str
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, label)
    page = _page(fake_microscope, run_doc)
    try:
        _calibrate(page, label)
    finally:
        page.close()

    found = sorted(p.name for p in _folder(run_doc).iterdir())
    assert found == [f"position-{POSITION[label]}.json"]


def test_the_run_makes_the_calibrations_folder_and_runs_again_over_an_existing_one(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        assert not _folder(run_doc).exists()
        first = _calibrate(page, FORTY)
        assert _folder(run_doc).is_dir()
        first_attempt = (
            RunDocument.load(_folder(run_doc) / "position-1.json")
            .stages["camera-calibration"]
            .attempt
        )
        second = _calibrate(page, FORTY)
        second_attempt = (
            RunDocument.load(_folder(run_doc) / "position-1.json")
            .stages["camera-calibration"]
            .attempt
        )
    finally:
        page.close()

    assert first.startswith("Camera calibration done in ")
    assert second.startswith("Camera calibration done in ")
    # The same file, replaced by the new attempt.
    assert first_attempt != second_attempt
    assert sorted(p.name for p in _folder(run_doc).iterdir()) == ["position-1.json"]


def test_a_folder_that_cannot_be_made_is_the_outcome_and_no_record_is_written(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    folder = _folder(run_doc)
    made = Path.mkdir

    def refuse(self: Path, *args: Any, **kwargs: Any) -> None:
        # A folder the account may not make (a read-only share), and no other.
        if self == folder:
            raise PermissionError(13, "Permission denied", str(self))
        made(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refuse)
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        outcome = _calibrate(page, FORTY)
        running = page.status().running
        view = page.view()
    finally:
        page.close()

    assert outcome.startswith(f"Camera calibration failed: could not make {folder}: ")
    assert "Permission denied" in outcome
    assert "\n" not in outcome
    assert running is None
    # The tool never ran, nothing was made or written, and the plate is as it was (FM-98, FM-210).
    assert StubCalibration.seen == []
    assert not folder.exists()
    assert run_doc.read_bytes() == before
    assert view.calibrations == ()
    assert view.registered is not None


def test_a_file_named_like_the_folder_is_refused_in_one_line_and_nothing_is_written(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    blocker = _folder(run_doc)
    blocker.write_text("not a folder", encoding="utf-8")
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        page.calibrate_objective(FORTY)
        # Where it is refused depends on what the OS says for a file used as a folder: POSIX says
        # `Not a directory` and `_prepare` refuses it before a thread starts; Windows may say the
        # path was not found, `_prepare` then lets the run start and the worker's `mkdir` fails.
        # Either way the outcome is one line and nothing is written, so the test joins first.
        assert page.join(10.0)
        outcome = page.status().outcome
        running = page.status().running
        view = page.view()
    finally:
        page.close()

    assert outcome is not None
    assert outcome.startswith(
        (
            f"camera-calibration: RunDocumentError: {blocker / 'position-1.json'}: "
            "could not be read",
            f"Camera calibration failed: could not make {blocker}: ",
        )
    )
    assert "\n" not in outcome
    assert running is None
    assert StubCalibration.seen == []
    assert blocker.read_text(encoding="utf-8") == "not a folder"
    assert run_doc.read_bytes() == before
    assert view.calibrations == ()


def test_a_calibration_document_that_cannot_be_read_is_the_outcome_and_is_left_as_it_is(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    document = _folder(run_doc) / "position-1.json"
    document.parent.mkdir()
    document.write_text("{not json", encoding="utf-8")
    page = _page(fake_microscope, run_doc)
    try:
        page.calibrate_objective(FORTY)
        outcome = page.status().outcome
        running = page.status().running
    finally:
        page.close()

    # The document the run is into is the one `check_ready` reads, before anything starts.
    assert outcome is not None
    assert outcome.startswith(
        f"camera-calibration: RunDocumentError: {document}: is not valid JSON"
    )
    assert running is None
    assert StubCalibration.seen == []
    assert document.read_text(encoding="utf-8") == "{not json"


# --- D13: every refusal the page foresees, in the controller (FM-169) -----------------


def _foreseen(
    case: str,
    microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
    request: pytest.FixtureRequest,
) -> tuple[PlatePage, str, str]:
    """A page in the state of ``case``, the label its button would send and the outcome that refuses it."""
    _seed_plate(run_doc, by_eye=case == "no offer")
    _turn(microscope, FORTY)
    if case == "closed":
        page = PlatePage.closed(
            microscope.profile, run_doc, entry_points=_entry_points()
        )
        return page, FORTY, NOT_CONNECTED
    if case == "dry run":
        dry = _dry_microscope(fake_core)
        request.addfinalizer(dry.close)
        return _page(dry, run_doc), FORTY, CALIBRATE_DRY_RUN
    page = _page(microscope, run_doc)
    if case == "unreadable run document":
        run_doc.write_text("{not json", encoding="utf-8")
        return page, FORTY, "run document: "
    if case == "no offer":
        return page, FORTY, CALIBRATE_NO_OFFER
    if case == "not known":
        microscope.override(ObjectiveTurret, Turret(_labels(microscope), None))
        return page, FORTY, CALIBRATE_NOT_KNOWN
    if case == "the plate's own":
        _turn(microscope, TEN)
        return page, TEN, calibrate_is_the_plates(TEN)
    if case == "another objective than the button's":
        _turn(microscope, TWENTY)
        return page, FORTY, calibrate_changed(FORTY, TWENTY)
    if case == "a button that named none":
        return page, "", CALIBRATE_NAMES_NONE
    if case == "a label on two positions":
        labels = _labels(microscope)
        labels[4] = FORTY
        microscope.override(ObjectiveTurret, Turret(labels, FORTY))
        return page, FORTY, calibrate_position_unknown(FORTY)
    if case == "a typed from_run_doc":
        # The refusal is the pairs'; the page is a plain one at the 40X (the Stand screen's).
        return page, FORTY, CALIBRATE_REUSE_ROUTED
    assert case == "a halted stand"
    microscope.stop()
    return page, FORTY, SCAN_HALTED


def _labels(microscope: Microscope) -> list[str]:
    return microscope.require(ObjectiveTurret).labels()


@pytest.mark.parametrize(
    "case",
    [
        "closed",
        "unreadable run document",
        "no offer",
        "dry run",
        "not known",
        "the plate's own",
        "another objective than the button's",
        "a button that named none",
        "a label on two positions",
        "a halted stand",
    ],
)
def test_calibrate_objective_refuses_each_foreseen_state_and_writes_nothing(
    fake_microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
    request: pytest.FixtureRequest,
    case: str,
) -> None:
    page, expected, refusal = _foreseen(
        case, fake_microscope, fake_core, run_doc, request
    )
    seen_before = run_doc.read_bytes() if run_doc.exists() else None
    try:
        page.calibrate_objective(expected)
        outcome = page.status().outcome
        running = page.status().running
    finally:
        page.close()

    assert outcome is not None
    if case == "unreadable run document":
        # An unreadable document is not an empty one, and not a plate that is not registered.
        assert outcome.startswith(f"Camera calibration: {refusal}")
        assert CALIBRATE_NO_OFFER not in outcome
    else:
        assert outcome == f"Camera calibration: {refusal}"
    # Nothing is started, written or made: no thread, no folder, no file, the same run.json.
    assert running is None
    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()
    assert (run_doc.read_bytes() if run_doc.exists() else None) == seen_before


# --- D19: the Stand screen's button -----------------------------------------------------


def _press_stand(page: PlatePage, expected: str, pairs: tuple[str, ...] = ()) -> str:
    """Press the Stand screen's button and wait for the run; the outcome the strip would say."""
    page.calibrate_camera(expected, pairs)
    assert page.join(10.0)
    outcome = page.status().outcome
    assert outcome is not None
    return outcome


def test_the_stand_button_runs_into_the_position_document_and_leaves_run_json_byte_identical_and_the_plate_registered(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # Acceptance 1 (FM-216): the plate was registered under the 10X, the operator turned the
    # turret to the 40X on the Stand screen and pressed the button it showed. Into `run.json`
    # it made the plate `unusable`, with `registered` and `offsets` gone (§18.6.1, measured).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        stages_before = sorted(RunDocument.load(run_doc).stages)
        outcome = _press_stand(page, FORTY, ("step_um=40", "settle_ms=5"))
        view = page.view()
        after = run_doc.read_bytes()
    finally:
        page.close()

    record = RunDocument.load(_folder(run_doc) / "position-1.json").provider(
        "camera-calibration"
    )
    assert outcome.startswith("Camera calibration done in ")
    assert after == before
    assert sorted(RunDocument.load(run_doc).stages) == stages_before
    assert view.registered is not None
    assert view.offsets is not None
    assert [item.state for item in view.checklist if item.slot == "plate"] == ["ok"]
    assert record is not None
    assert record.status is StageStatus.done
    assert CameraCalibration.model_validate(record.result).objective == FORTY
    # The fields of the form reach the tool, and nothing turned the turret (FM-14).
    assert [(p.step_um, p.settle_ms) for p in StubCalibration.seen] == [(40.0, 5.0)]
    assert _in_place(fake_microscope) == FORTY
    assert [h.position for h in view.calibrations] == [1]


@pytest.mark.parametrize(
    "case",
    ["a plate by eye", "the plate's own objective", "no run document"],
)
def test_the_stand_button_runs_into_run_json_with_no_plate_registered_from_its_walls_and_at_the_plates_own_objective(
    fake_microscope: Microscope, run_doc: Path, case: str
) -> None:
    # As it was: the run document's calibration is the one `plate-walls` reads. With no offer
    # there is no registration from walls to protect, and at the plate's own objective the
    # calibration is the plate's. Every pair typed is sent, and the button's label need not name
    # one for a page that has no offer (the first calibration of a stand).
    expected = FORTY
    turned = FORTY
    if case == "a plate by eye":
        _seed_plate(run_doc, by_eye=True)
        expected = ""
    elif case == "the plate's own objective":
        _seed_plate(run_doc)
        turned = expected = TEN
    _turn(fake_microscope, turned)
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _press_stand(page, expected, ("step_um=40", "settle_ms=5"))
    finally:
        page.close()

    record = RunDocument.load(run_doc).provider("camera-calibration")
    assert outcome.startswith("Camera calibration done in ")
    assert record is not None
    assert CameraCalibration.model_validate(record.result).objective == turned
    assert not _folder(run_doc).exists()
    assert [(p.step_um, p.settle_ms) for p in StubCalibration.seen] == [(40.0, 5.0)]


def test_a_stand_click_with_no_offer_is_the_runs_own_call_and_its_own_refusals(
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no offer the click is `run("camera-calibration", pairs)` and nothing else: no read of
    # the turret, of the light or of the halt, and an unreadable run document is `run`'s refusal.
    calls: list[tuple[str, tuple[str, ...]]] = []
    page = _page(fake_microscope, run_doc)
    real_run = page.run

    def spy(plugin: str, params: Any, **kwargs: Any) -> None:
        calls.append((plugin, tuple(params)))
        real_run(plugin, params, **kwargs)

    monkeypatch.setattr(page, "run", spy)
    for name in ("_turret_reading", "_halted"):
        monkeypatch.setattr(
            page, name, lambda *a, _n=name: pytest.fail(f"{_n} was read")
        )
    try:
        page.calibrate_camera("", ("step_um=40", "from_run_doc=cal-10x.json"))
        refused = page.status().outcome
        assert page.join(10.0)
    finally:
        page.close()

    # Not the Stand's own refusals, but the call as it was: `run`'s, with every pair as typed.
    assert calls == [
        ("camera-calibration", ("step_um=40", "from_run_doc=cal-10x.json"))
    ]
    assert refused is not None
    assert not refused.startswith("Camera calibration: ")


def test_a_stand_click_on_an_unreadable_run_document_is_the_runs_refusal_and_writes_nothing(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _turn(fake_microscope, FORTY)
    run_doc.write_text("{not json", encoding="utf-8")
    page = _page(fake_microscope, run_doc)
    try:
        page.calibrate_camera(FORTY)
        outcome = page.status().outcome
        running = page.status().running
    finally:
        page.close()

    assert outcome is not None
    assert outcome.startswith(
        f"camera-calibration: RunDocumentError: {run_doc}: is not valid JSON"
    )
    assert CALIBRATE_NO_OFFER not in outcome
    assert running is None
    assert StubCalibration.seen == []
    assert run_doc.read_text(encoding="utf-8") == "{not json"
    assert not _folder(run_doc).exists()


@pytest.mark.parametrize(
    ("shown", "now"),
    [
        (TEN, TWENTY),
        (FORTY, TWENTY),
        (TWENTY, FORTY),
        (TEN, FORTY),
        # The turret now reads the plate's own objective: the destination would be the run
        # document, for a label that was drawn for a document of its own.
        (FORTY, TEN),
    ],
    ids=[
        "plates-own-then-20X",
        "40X-then-20X",
        "20X-then-40X",
        "plates-own-then-40X",
        "40X-then-plates-own",
    ],
)
def test_a_stand_click_that_showed_one_objective_is_refused_when_the_turret_now_reads_another(
    fake_microscope: Microscope, run_doc: Path, shown: str, now: str
) -> None:
    # FM-177 and its converse (D19): the destination was chosen for the label shown. A turn at
    # the handset since, with no tick in between, would send the 40X's calibration into
    # `run.json` (the button showed the 10X) and put the plate out of date. Refused, with the
    # page's own words and nothing written, started or turned.
    _seed_plate(run_doc)
    _turn(fake_microscope, now)
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        page.calibrate_camera(shown)
        outcome = page.status().outcome
        running = page.status().running
        after = run_doc.read_bytes()
    finally:
        page.close()

    assert outcome == f"Camera calibration: {calibrate_changed(shown, now)}"
    assert running is None
    assert after == before
    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()
    assert _in_place(fake_microscope) == now


@pytest.mark.parametrize("case", ["closed", "not known", "a button that named none"])
def test_a_stand_click_with_an_offer_is_refused_for_a_closed_stand_an_unknown_objective_and_no_label(
    fake_microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
    request: pytest.FixtureRequest,
    case: str,
) -> None:
    page, expected, refusal = _foreseen(
        case, fake_microscope, fake_core, run_doc, request
    )
    before = run_doc.read_bytes()
    try:
        page.calibrate_camera(expected)
        outcome = page.status().outcome
        running = page.status().running
    finally:
        page.close()

    # The tool's own run would have written the 40X's calibration into `run.json` here.
    assert outcome == f"Camera calibration: {refusal}"
    assert running is None
    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()
    assert run_doc.read_bytes() == before


@pytest.mark.parametrize(
    "case",
    [
        "dry run",
        "a halted stand",
        "a label on two positions",
        "a typed from_run_doc",
    ],
)
def test_a_stand_click_into_a_document_is_refused_for_a_dry_run_a_halt_a_position_not_known_and_a_typed_from_run_doc(
    fake_microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
    request: pytest.FixtureRequest,
    case: str,
) -> None:
    page, expected, refusal = _foreseen(
        case, fake_microscope, fake_core, run_doc, request
    )
    pairs = ("from_run_doc=cal-10x.json",) if case == "a typed from_run_doc" else ()
    before = run_doc.read_bytes()
    try:
        page.calibrate_camera(expected, pairs)
        outcome = page.status().outcome
        running = page.status().running
    finally:
        page.close()

    assert outcome == f"Camera calibration: {refusal}"
    assert running is None
    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()
    assert run_doc.read_bytes() == before


def test_a_stand_click_into_a_document_refuses_in_the_designs_order(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # D19's order for a document, after the label checks: the light, a typed from_run_doc, the
    # position, the halt. All four are wrong at once, then put right one at a time.
    _seed_plate(run_doc)
    labels = _labels(fake_microscope)
    twice = list(labels)
    twice[4] = FORTY  # the 40X on two positions: its position cannot be named
    light = Light(False)
    fake_microscope.override(LightSource, light)
    fake_microscope.override(Shutter, AutoShutter(False))
    fake_microscope.override(ObjectiveTurret, Turret(twice, FORTY))
    fake_microscope.stop()
    page = _page(fake_microscope, run_doc)
    typed = ("from_run_doc=cal-10x.json",)
    try:
        outcomes = [(_press(page, typed))]
        light.reading = True
        outcomes.append(_press(page, typed))
        outcomes.append(_press(page, ()))
        fake_microscope.override(ObjectiveTurret, Turret(labels, FORTY))
        outcomes.append(_press(page, ()))
    finally:
        page.close()

    assert outcomes == [
        f"Camera calibration: {CALIBRATE_LIGHT_OFF}",
        f"Camera calibration: {CALIBRATE_REUSE_ROUTED}",
        f"Camera calibration: {calibrate_position_unknown(FORTY)}",
        f"Camera calibration: {SCAN_HALTED}",
    ]
    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()


def _press(page: PlatePage, pairs: tuple[str, ...]) -> str | None:
    page.calibrate_camera(FORTY, pairs)
    return page.status().outcome


def test_a_stand_click_into_a_document_sends_only_step_um_and_settle_ms(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The form's other fields (a blank `from_run_doc`, whatever a script adds) are not sent into
    # the objective's own document: the tool is run with the two fields of the Stand's form.
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _press_stand(
            page, FORTY, ("step_um=40", "from_run_doc=", "unknown=1", "settle_ms=5")
        )
    finally:
        page.close()

    assert outcome.startswith("Camera calibration done in ")
    assert [(p.step_um, p.settle_ms) for p in StubCalibration.seen] == [(40.0, 5.0)]


def test_a_stand_click_while_an_activity_goes_raises_as_run_does(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        with pytest.raises(RuntimeError, match="slow is running"):
            page.calibrate_camera(FORTY)
        Slow.release.set()
        assert page.join(10.0)
    finally:
        page.close()
    with pytest.raises(RuntimeError, match="the page is closed"):
        page.calibrate_camera(FORTY)

    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()


def test_command_for_names_another_run_document_when_given_one(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR-0006, principle 5: the Stand screen's *Command* shows the call the click makes, which
    # for a document is the run into that position's file. Absolute, as the page's own is.
    monkeypatch.chdir(run_doc.parent)
    page = PlatePage(fake_microscope, Path("run.json"), entry_points=_entry_points())
    try:
        own = shlex.split(page.command_for("camera-calibration", ["step_um=40"]))
        routed = shlex.split(
            page.command_for(
                "camera-calibration",
                ["step_um=40"],
                run_doc=Path("calibrations") / "position-1.json",
            )
        )
    finally:
        page.close()

    def run_document(words: list[str]) -> str:
        return words[words.index("--run-doc") + 1]

    assert run_document(own) == str(run_doc.parent.absolute() / "run.json")
    assert run_document(routed) == str(
        run_doc.parent.absolute() / "calibrations" / "position-1.json"
    )
    # Nothing else of the call changed: the same tool, profile and pairs.
    own[own.index("--run-doc") + 1] = routed[routed.index("--run-doc") + 1]
    assert own == routed
    assert routed[:3] == ["smc", "run", "camera-calibration"]
    assert routed[-2:] == ["--param", "step_um=40"]


def test_the_stand_click_replaces_the_documents_calibration_and_a_failure_leaves_the_others(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The stand's click is the block's run: a failed one replaces its own position's document
    # and no other (FM-216: a document per objective, never two objectives' values in one).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    other = _folder(run_doc) / "position-4.json"
    _seed_calibration(other, TWENTY)
    other_before = other.read_bytes()
    StubCalibration.fails = "the frame is dark or flat"
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        outcome = _press_stand(page, FORTY)
        view = page.view()
        after = run_doc.read_bytes()
    finally:
        page.close()

    assert (
        outcome
        == "Camera calibration failed: camera-calibration: the frame is dark or flat"
    )
    assert after == before
    assert other.read_bytes() == other_before
    assert view.registered is not None
    assert [h.position for h in view.calibrations] == [1, 4]
    assert view.calibrations[0].read.problem == calibration_failed(
        "the frame is dark or flat"
    )


def test_calibrate_objective_while_an_activity_goes_raises_and_writes_nothing(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        before = run_doc.read_bytes()
        with pytest.raises(RuntimeError, match="slow is running"):
            page.calibrate_objective(FORTY)
        Slow.release.set()
        assert page.join(10.0)
    finally:
        page.close()
    with pytest.raises(RuntimeError, match="the page is closed"):
        page.calibrate_objective(FORTY)

    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()
    # The slow tool's own record is the only thing the page's document gained.
    assert sorted(RunDocument.load(run_doc).stages) == ["plate-walls", "slow"]
    assert before.startswith(b"{")


@pytest.mark.parametrize("button", BUTTONS)
def test_a_light_that_reads_off_refuses_the_calibration_and_leaves_the_usable_document_byte_identical(
    fake_microscope: Microscope, run_doc: Path, button: str
) -> None:
    # Acceptance 2, FM-206: the tool refuses a dark frame, and its refusal is a `failed` record
    # that replaces the position's usable document. The light is read at the click: a status read
    # (the tick's copy) says on, then it is switched off with no tick in between (FM-177).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    document = _folder(run_doc) / "position-1.json"
    _seed_calibration(document, FORTY)
    light = Light(True)
    fake_microscope.override(LightSource, light)
    # A stand whose exposure does not open the lamp (auto-shutter off): the light at rest is
    # what the frame sees.
    fake_microscope.override(Shutter, AutoShutter(False))
    page = _page(fake_microscope, run_doc)
    try:
        assert page.status().light_on is True
        light.reading = False
        document_before = document.read_bytes()
        run_before = run_doc.read_bytes()
        _click(page, button, FORTY)
        outcome = page.status().outcome
        running = page.status().running
        refused_document = document.read_bytes()
        refused_seen = list(StubCalibration.seen)
        held = page.view().calibrations
        # The light switched on again, nothing else changed: the same click runs, and it is the
        # one that replaces the document (so the check above could have seen a replacement).
        light.reading = True
        _click(page, button, FORTY)
        assert page.join(10.0)
        replaced_document = document.read_bytes()
    finally:
        page.close()

    assert outcome == f"Camera calibration: {CALIBRATE_LIGHT_OFF}"
    assert running is None
    # Nothing started and nothing written: the usable document is the one it was, and listed.
    assert refused_seen == []
    assert refused_document == document_before
    assert run_doc.read_bytes() == run_before
    assert [(h.position, h.read.objective, h.read.problem) for h in held] == [
        (1, FORTY, None)
    ]
    assert len(StubCalibration.seen) == 1
    assert replaced_document != document_before
    # Nothing in either click turned the turret (FM-14).
    assert _in_place(fake_microscope) == FORTY


@pytest.mark.parametrize("button", BUTTONS)
def test_a_light_that_reads_off_leaves_every_document_alone_for_a_click_that_ran_nothing(
    fake_microscope: Microscope, run_doc: Path, button: str
) -> None:
    # The refusal alone, with no second click: no file is made, none replaced, no folder.
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    fake_microscope.override(LightSource, Light(False))
    fake_microscope.override(Shutter, AutoShutter(False))
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        _click(page, button, FORTY, ("step_um=40",))
        outcome = page.status().outcome
        running = page.status().running
    finally:
        page.close()

    assert outcome == f"Camera calibration: {CALIBRATE_LIGHT_OFF}"
    assert running is None
    assert StubCalibration.seen == []
    assert not _folder(run_doc).exists()
    assert run_doc.read_bytes() == before


@pytest.mark.parametrize("button", BUTTONS)
@pytest.mark.parametrize(
    "reading",
    [None, RuntimeError("the lamp does not answer")],
    ids=["no switch smc knows", "a read that raises"],
)
def test_an_unknown_light_or_one_that_raises_refuses_nothing(
    fake_microscope: Microscope,
    run_doc: Path,
    reading: bool | Exception | None,
    button: str,
) -> None:
    # An unknown is not a refusal the page can justify: the tool decides (D20).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    fake_microscope.override(LightSource, Light(reading))
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _calibrate(page, FORTY, button=button)
    finally:
        page.close()

    assert outcome.startswith("Camera calibration done in ")
    assert (_folder(run_doc) / "position-1.json").is_file()
    assert len(StubCalibration.seen) == 1


@pytest.mark.parametrize("button", BUTTONS)
def test_a_light_that_reads_off_while_the_core_opens_the_shutter_for_the_snap_refuses_nothing(
    fake_microscope: Microscope, run_doc: Path, button: str
) -> None:
    # The review of #557 (probed on the Ti2-E's own layout, `profiles/nikon-ti2.cfg`): the lamp is
    # the core's shutter and AutoShutter is 1, so `is_on()` reads False at rest while the camera's
    # snap opens it. The refusal would be false, on the owner's main stand, for both buttons.
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    fake_microscope.override(LightSource, Light(False))
    fake_microscope.override(Shutter, AutoShutter(True))
    page = _page(fake_microscope, run_doc)
    try:
        status = page.status()
        outcome = _calibrate(page, FORTY, button=button)
    finally:
        page.close()

    assert status.light_on is False
    assert status.auto_shutter is True
    assert outcome.startswith("Camera calibration done in ")
    assert len(StubCalibration.seen) == 1


@pytest.mark.parametrize("button", BUTTONS)
@pytest.mark.parametrize(
    "auto",
    [False, RuntimeError("the core does not answer")],
    ids=["auto-shutter off", "a read that raises"],
)
def test_a_light_that_reads_off_still_refuses_unless_the_core_is_known_to_open_the_shutter(
    fake_microscope: Microscope,
    run_doc: Path,
    auto: bool | Exception,
    button: str,
) -> None:
    # Without auto-shutter, or when it cannot be read, the exposure does not open the lamp as far
    # as the page knows: the refusal D20 designed stands (unknown does not become `fine`).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    fake_microscope.override(LightSource, Light(False))
    fake_microscope.override(Shutter, AutoShutter(auto))
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        _click(page, button, FORTY)
        outcome = page.status().outcome
        running = page.status().running
        after = run_doc.read_bytes()
    finally:
        page.close()

    assert outcome == f"Camera calibration: {CALIBRATE_LIGHT_OFF}"
    assert running is None
    assert StubCalibration.seen == []
    assert after == before


def test_the_status_reads_the_auto_shutter_only_when_the_light_reads_off(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The flag matters only for a light that reads off: no extra read of the core otherwise.
    shutter = AutoShutter(True)
    reads: list[bool] = []
    real = shutter.auto_shutter

    def counted() -> bool:
        reads.append(True)
        return real()

    shutter.auto_shutter = counted  # type: ignore[method-assign]
    light = Light(True)
    fake_microscope.override(LightSource, light)
    fake_microscope.override(Shutter, shutter)
    page = _page(fake_microscope, run_doc)
    try:
        on = page.status()
        light.reading = None
        unknown = page.status()
        assert reads == []
        light.reading = False
        off = page.status()
    finally:
        page.close()

    assert (on.light_on, on.auto_shutter) == (True, None)
    assert (unknown.light_on, unknown.auto_shutter) == (None, None)
    assert (off.light_on, off.auto_shutter) == (False, True)
    assert reads == [True]


def test_a_stand_with_no_light_source_refuses_nothing(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    real = Microscope.get

    def get(self: Microscope, capability: type) -> Any:
        return None if capability is LightSource else real(self, capability)

    monkeypatch.setattr(Microscope, "get", get)
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _calibrate(page, FORTY)
    finally:
        page.close()

    assert outcome.startswith("Camera calibration done in ")


def test_the_dry_run_text_of_the_calibration_is_the_tools_own_refusal(
    fake_core: FakeCore, tmp_path: Path
) -> None:
    # The page says it before the click; the tool says it again and its refusal is a `failed`
    # record, which would replace the position's usable document: the page's copy is pinned to
    # the tool's own raise, with the real tool on a dry-run stand (FM-243).
    dry = _dry_microscope(fake_core)
    try:
        with pytest.raises(PluginRunError) as raised:
            run_plugin(
                CameraCalibrationPlugin(),
                dry,
                CameraCalibrationParams(),
                tmp_path / "position-1.json",
            )
    finally:
        dry.close()

    assert CALIBRATE_DRY_RUN in str(raised.value)
    assert raised.value.reason == CALIBRATE_DRY_RUN


def test_a_calibration_into_its_own_document_notes_no_attempt_of_the_run_document(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Scan screen's account of a record is `run.json`'s (FM-244): a calibration run into
    # another document must not make the page note an attempt of a record it did not write.
    noted: list[str] = []

    def spy(self: PlatePage, plugin: str) -> None:
        noted.append(plugin)

    monkeypatch.setattr(PlatePage, "_note_attempt", spy)
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        _calibrate(page, FORTY)
        assert noted == []
        # The spy can fail: a run into the page's own document notes its attempt.
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        Slow.release.set()
        assert page.join(10.0)
    finally:
        page.close()

    assert noted == ["slow"]


def test_a_stop_during_a_calibration_says_stopped_and_leaves_the_other_documents_alone(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    other = _folder(run_doc) / "position-4.json"
    _seed_calibration(other, TWENTY)
    other_before = other.read_bytes()
    StubCalibration.release = threading.Event()
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        page.calibrate_objective(FORTY)
        assert StubCalibration.started.wait(10.0)
        page.stop()
        assert page.join(10.0)
        outcome = page.status().outcome
        view = page.view()
    finally:
        page.close()

    record = RunDocument.load(_folder(run_doc) / "position-1.json").provider(
        "camera-calibration"
    )
    assert outcome == "Camera calibration stopped"
    assert record is not None
    assert record.status is StageStatus.stopped
    assert other.read_bytes() == other_before
    assert run_doc.read_bytes() == before
    assert view.registered is not None


# --- D21: a document is its file's position's -------------------------------------------


@pytest.mark.parametrize("button", BUTTONS)
def test_a_run_that_measured_another_objective_than_the_button_named_says_so_and_not_done(
    fake_microscope: Microscope, run_doc: Path, button: str
) -> None:
    # Acceptance 3 (FM-216, FM-177): the click read the 40X at position 1, and the operator turned
    # the turret to the 20X at the handset before the tool read its own label. What is on the
    # disk is the 20X's calibration in the 40X's file. It holds for both buttons (D21).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    StubCalibration.turns_to = TWENTY
    page = _page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        outcome = _calibrate(page, FORTY, button=button)
        view = page.view()
        # The turret stands at the 20X now: the default (the empty field) must not take the 20X's
        # calibration out of the 40X's file for the 20X's own (D16, D21).
        step = page.offsets_step("")
        after = run_doc.read_bytes()
    finally:
        page.close()

    document = _folder(run_doc) / "position-1.json"
    assert outcome == (
        f"Camera calibration: {calibrate_measured_other(FORTY, TWENTY, 'position-1.json')}"
    )
    assert "done" not in outcome
    # The run did write the record of what it measured, in the file of the label it started at.
    held = view.calibrations
    assert [(h.position, h.read.objective, h.read.problem) for h in held] == [
        (1, TWENTY, None)
    ]
    assert (
        CameraCalibration.model_validate(
            RunDocument.load(document).stages["camera-calibration"].result
        ).objective
        == TWENTY
    )
    # Left out of the default: nothing to measure with, and the wait is the page's own.
    assert step.targets == ()
    assert step.pairs == ()
    assert step.wait == OFFSETS_NO_DOCUMENT
    # The page's own document is the registration's, as ever, and the turret is where the
    # *operator* left it (the run did not turn it back, and nothing here turns it).
    assert after == before
    assert _in_place(fake_microscope) == TWENTY


def test_a_run_that_measured_the_named_objective_says_what_the_tool_says(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    StubCalibration.assumed = ("the stage was not turned",)
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _calibrate(page, FORTY)
    finally:
        page.close()

    # The tool's own end: its title, its time and its first assumed line, not the page's check.
    assert outcome.startswith("Camera calibration done in ")
    assert outcome.endswith("; assumed: the stage was not turned")
    assert "measured" not in outcome


def test_a_calibration_that_ended_failed_says_the_tools_failure_and_never_the_objective_check(
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The check runs after a run that ended `done` only. A document that reads as a usable one of
    # another objective whatever the end (here: the read is replaced) must not turn a failure
    # into the page's other words.
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    StubCalibration.fails = "the frame is dark or flat"
    monkeypatch.setattr(
        controller_module,
        "_read_calibration",
        lambda path, *, profile, dry_run: CalibrationRead(
            path, TWENTY, "2026-11-17T09:00:00+00:00", None
        ),
    )
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _calibrate(page, FORTY)
    finally:
        page.close()

    assert (
        outcome
        == "Camera calibration failed: camera-calibration: the frame is dark or flat"
    )


def test_a_record_that_cannot_be_read_after_the_run_leaves_the_tools_own_outcome(
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing the page can say of a document it cannot read: the tool's `done` stands (FM-98).
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    monkeypatch.setattr(
        controller_module,
        "_read_calibration",
        lambda path, *, profile, dry_run: CalibrationRead(
            path, None, "", "the share does not answer"
        ),
    )
    page = _page(fake_microscope, run_doc)
    try:
        outcome = _calibrate(page, FORTY)
    finally:
        page.close()

    assert outcome.startswith("Camera calibration done in ")


# --- D15: the held documents ------------------------------------------------------------


def test_a_failed_calibration_replaces_the_documents_usable_one_and_the_list_says_why(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    words = "the frame of Nikon 40X is dark or flat: check the light path"
    page = _page(fake_microscope, run_doc)
    try:
        _calibrate(page, FORTY)
        usable = page.view().calibrations
        StubCalibration.fails = words
        failed_outcome = _calibrate(page, FORTY)
        view = page.view()
    finally:
        page.close()

    assert [h.read.problem for h in usable] == [None]
    assert usable[0].read.objective == FORTY
    assert failed_outcome == f"Camera calibration failed: camera-calibration: {words}"
    # The latest attempt decides the slot: the usable calibration is gone, with its date, and
    # the list says the tool's words and not the generic `ended failed; run it again`.
    assert len(view.calibrations) == 1
    held = view.calibrations[0]
    assert held.position == 1
    assert held.read.objective is None
    assert held.read.measured == ""
    assert held.read.problem == calibration_failed(words)
    assert held.read.notes == ()


def test_the_held_calibrations_are_read_for_positions_1_to_12_in_order_and_for_no_other_file(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    folder = _folder(run_doc)
    _seed_calibration(folder / "position-12.json", "Objective-12")
    _seed_calibration(folder / "position-3.json", "Objective-2")
    _seed_calibration(folder / "position-13.json", "Objective-13")
    _seed_calibration(folder / "other.json", TEN)
    (folder / "position-9.json").mkdir()
    (folder / "position-5.json").write_text("not a document {", encoding="utf-8")
    _seed_calibration(folder / "position-7.json", TEN, profile="lightsheet")
    page = _page(fake_microscope, run_doc)
    try:
        held = page.view().calibrations
    finally:
        page.close()

    assert [h.position for h in held] == [3, 5, 7, 9, 12]
    by_position = {h.position: h.read for h in held}
    assert by_position[3] == CalibrationRead(
        str(folder / "position-3.json"),
        "Objective-2",
        "2026-11-17T09:00:00+00:00",
        None,
        image_shape_px=(384, 512),
    )
    assert by_position[12].problem is None
    # A page's path, as the page was given the run document, never resolved.
    assert by_position[9].path == str(folder / "position-9.json")
    assert by_position[9].problem == (
        f"{(folder / 'position-9.json').as_posix()}: is a folder, not a run document; "
        "name a file"
    )
    garbage = by_position[5].problem
    assert garbage is not None
    assert garbage.startswith(
        f"{(folder / 'position-5.json').as_posix()}: is not valid JSON"
    )
    assert by_position[7].problem == (
        f"in {(folder / 'position-7.json').as_posix()}: camera-calibration was run on "
        "lightsheet, this is demo; run camera-calibration again on this profile"
    )
    assert all(h.read.objective is None for h in held if h.read.problem is not None)


def test_a_held_document_carries_the_frame_it_was_made_in(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The scan plan refuses a document of another frame (m5 §18.6 D25, #617): the read holds it,
    # ``(height, width)`` as the tool keeps it, so that the click can compare it with the camera's.
    # A read with a problem has no frame.
    _seed_plate(run_doc)
    folder = _folder(run_doc)
    _seed_calibration(folder / "position-1.json", FORTY)
    _seed_calibration(
        folder / "position-4.json",
        TWENTY,
        status=StageStatus.failed,
        error="the frame is dark",
    )
    page = _page(fake_microscope, run_doc)
    try:
        held = {h.position: h.read for h in page.view().calibrations}
    finally:
        page.close()

    assert held[1].image_shape_px == (384, 512)
    assert held[4].problem is not None
    assert held[4].image_shape_px is None
    # A read built without a frame (every construction before #617) has none.
    assert CalibrationRead("p.json", FORTY, "", None).image_shape_px is None


def test_one_position_that_cannot_be_read_leaves_the_others(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Anything a read raises is a line for its position and never an exception (FM-98).
    _seed_plate(run_doc)
    folder = _folder(run_doc)
    _seed_calibration(folder / "position-1.json", FORTY)
    _seed_calibration(folder / "position-4.json", TWENTY)
    real = controller_module._read_calibration

    def broken(path: str, *, profile: str, dry_run: bool) -> CalibrationRead:
        if path.endswith("position-1.json"):
            raise OSError("the share does not answer")
        return real(path, profile=profile, dry_run=dry_run)

    monkeypatch.setattr(controller_module, "_read_calibration", broken)
    page = _page(fake_microscope, run_doc)
    try:
        held = page.view().calibrations
    finally:
        page.close()

    assert [h.position for h in held] == [1, 4]
    assert held[0].read.problem == (
        f"{folder / 'position-1.json'}: OSError: the share does not answer"
    )
    assert held[1].read.problem is None
    assert held[1].read.objective == TWENTY


def test_the_held_calibrations_carry_the_calibrations_notes_cut_and_flat(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-205: the longest form of a line that carries a tool's words, through the bound.
    long = "step of 50 µm\nno pixel size was known " + "x" * 700
    notes = tuple(f"note {n}" for n in range(12))
    _seed_plate(run_doc)
    _seed_calibration(_folder(run_doc) / "position-1.json", FORTY, assumed=(long,))
    _seed_calibration(_folder(run_doc) / "position-4.json", TWENTY, assumed=notes)
    page = _page(fake_microscope, run_doc)
    try:
        held = page.view().calibrations
    finally:
        page.close()

    (one,) = held[0].read.notes
    assert "\n" not in one
    assert one.startswith("step of 50 µm no pixel size was known xxx")
    assert len(one) == 601
    assert one.endswith("…")
    assert held[1].read.notes == notes[:8]


def test_the_view_reads_the_held_calibrations_only_when_the_block_is_offered(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[str] = []
    real = controller_module._read_calibration

    def spy(path: str, *, profile: str, dry_run: bool) -> CalibrationRead:
        read.append(path)
        return real(path, profile=profile, dry_run=dry_run)

    monkeypatch.setattr(controller_module, "_read_calibration", spy)
    _seed_calibration(_folder(run_doc) / "position-1.json", FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        # No plate at all, then a plate registered by eye: no offer, nothing read.
        nothing = page.view()
        _seed_plate(run_doc, by_eye=True)
        by_eye = page.view()
        assert read == []
        # A plate from its walls: the block is offered and its documents are read.
        _seed_plate(run_doc)
        offered = page.view()
    finally:
        page.close()

    assert nothing.calibrations == ()
    assert by_eye.offsets is None
    assert by_eye.calibrations == ()
    assert offered.offsets is not None
    assert [h.position for h in offered.calibrations] == [1]
    assert read == [str(_folder(run_doc) / "position-1.json")]


def test_the_held_calibrations_are_read_from_a_relative_run_document_as_given(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-24, FM-50: the page's path is the one it was given, never resolved.
    monkeypatch.chdir(tmp_path)
    _seed_plate(Path("run.json"))
    _seed_calibration(Path("calibrations") / "position-1.json", FORTY)
    page = PlatePage(fake_microscope, Path("run.json"), entry_points=_entry_points())
    try:
        held = page.view().calibrations
        folder = page.calibrations_folder
    finally:
        page.close()

    assert folder == Path("calibrations")
    assert not folder.is_absolute()
    assert [h.read.path for h in held] == [
        str(Path("calibrations") / "position-1.json")
    ]


# --- D16: the empty field measures with them -----------------------------------------


def _typed(*paths: Path | str) -> str:
    return "\n".join(str(path) for path in paths)


def _hold(run_doc: Path, **positions: str | None) -> dict[str, Path]:
    """One document per ``position_<n>=<objective>``; returns the paths by the same names."""
    paths: dict[str, Path] = {}
    for name, objective in positions.items():
        path = _folder(run_doc) / f"position-{name.removeprefix('position_')}.json"
        _seed_calibration(path, objective)
        paths[name] = path
    return paths


def test_offsets_step_with_an_empty_field_takes_the_held_documents_in_position_order(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    # The 20X is the turret's fourth position, the 40X its first: position order, not the
    # order the files were made in.
    held = _hold(run_doc, position_4=TWENTY, position_1=FORTY)
    page = _page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("")
        blank = page.offsets_step(" \n \n")
        page.measure_offsets(step.targets, "")
        assert page.join(10.0)
    finally:
        page.close()

    assert step.wait is None
    assert step.targets == (FORTY, TWENTY)
    assert blank.targets == (FORTY, TWENTY)
    assert [r.path for r in step.reads] == [
        str(held["position_1"]),
        str(held["position_4"]),
    ]
    # The pairs are what `smc run objective-offsets` is given, read back by the parameters'
    # own parser, never compared as strings (FM-45).
    parsed = parse_params(StubOffsetsParams, [*step.pairs, "confirm_turns=true"])
    assert parsed.targets == (FORTY, TWENTY)  # type: ignore[attr-defined]
    assert parsed.calibrations == (held["position_1"], held["position_4"])  # type: ignore[attr-defined]
    assert [(p.targets, p.confirm_turns) for p in StubOffsets.seen] == [
        ((FORTY, TWENTY), True)
    ]


def test_the_default_leaves_out_the_plates_objective_a_problem_and_an_objective_not_on_the_turret(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # D21: a document is kept only when its objective is the label of its own position, so the
    # 40X's calibration in the sixth file is left out although the 40X is on the turret.
    _seed_plate(run_doc)
    _hold(
        run_doc,
        position_1=FORTY,
        position_2=TEN,  # the plate's own objective
        position_3="Objective-2",  # a problem, below
        position_4=TWENTY,
        position_5="Nikon 60X Plan Apo",  # not on the turret now
        position_6=FORTY,  # on the turret, but at position 1, not here
    )
    (_folder(run_doc) / "position-3.json").write_text("{not json", encoding="utf-8")
    page = _page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("")
        held = page.view().calibrations
    finally:
        page.close()

    assert step.wait is None
    assert step.targets == (FORTY, TWENTY)
    assert [r.path for r in step.reads] == [
        str(_folder(run_doc) / "position-1.json"),
        str(_folder(run_doc) / "position-4.json"),
    ]
    # Nothing is dropped silently: the four left out are in the list the block draws.
    assert [h.position for h in held] == [1, 2, 3, 4, 5, 6]


def test_with_nothing_held_and_nothing_typed_the_wait_is_the_no_document_text(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = _page(fake_microscope, run_doc)
    try:
        nothing = page.offsets_step("")
        # Only the plate's own objective is held: nothing to measure.
        _hold(run_doc, position_2=TEN)
        only_the_plates = page.offsets_step("")
    finally:
        page.close()

    assert nothing.wait == OFFSETS_NO_DOCUMENT
    assert nothing.pairs == ()
    assert only_the_plates.wait == OFFSETS_NO_DOCUMENT
    assert only_the_plates.reads == ()


def test_a_typed_field_wins_over_the_held_documents_and_is_never_merged_with_them(
    fake_microscope: Microscope, run_doc: Path, tmp_path: Path
) -> None:
    _seed_plate(run_doc)
    held = _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    elsewhere = tmp_path / "elsewhere" / "cal-20x.json"
    _seed_calibration(elsewhere, TWENTY)
    page = _page(fake_microscope, run_doc)
    try:
        typed = page.offsets_step(_typed(elsewhere))
        # A typed path that is also a held document: used once, the other held one not added.
        same = page.offsets_step(_typed(held["position_4"]))
        # Typed twice with a held document: the tool's own refusal of two documents for one objective.
        twice = page.offsets_step(_typed(held["position_4"], elsewhere))
    finally:
        page.close()

    assert typed.wait is None
    assert typed.targets == (TWENTY,)
    assert [r.path for r in typed.reads] == [str(elsewhere)]
    assert same.targets == (TWENTY,)
    assert twice.wait == (
        f"{TWENTY} is in two documents: {held['position_4']} and {elsewhere}"
    )


def test_measure_offsets_with_the_default_compares_the_armed_objectives_again(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    held = _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    page = _page(fake_microscope, run_doc)
    try:
        armed = page.offsets_step("").targets
        assert armed == (FORTY, TWENTY)
        # Between the question and the click the 20X's document is taken away: the default is
        # read again and the turret would go to other objectives than the confirmed ones (D4).
        held["position_4"].unlink()
        before = run_doc.read_bytes()
        page.measure_offsets(armed, "")
        outcome = page.status().outcome
        running = page.status().running
        after = run_doc.read_bytes()
    finally:
        page.close()

    assert StubOffsets.seen == []
    assert running is None
    assert after == before
    assert outcome == (
        "Objective offsets: the objectives changed since you confirmed them "
        f"({FORTY}, {TWENTY} → {FORTY}): measure again"
    )


def test_the_default_with_more_usable_documents_than_the_limit_waits_for_the_limit(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    # The fake turret has six positions and the plate takes one: the limit is lowered.
    monkeypatch.setattr(controller_module, "MAX_TARGETS", 1)
    page = _page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("")
    finally:
        page.close()

    assert step.wait == OFFSETS_TOO_MANY
    assert step.pairs == ()
    # As for lines: the second document is not read.
    assert len(step.reads) == 1


def test_offsets_step_with_the_default_opens_no_file_while_an_activity_goes(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-238: a path on a share that does not answer must not hold the loop that handles STOP.
    opened: list[str] = []
    real = controller_module._read_calibration

    def spy(path: str, *, profile: str, dry_run: bool) -> CalibrationRead:
        opened.append(path)
        return real(path, profile=profile, dry_run=dry_run)

    monkeypatch.setattr(controller_module, "_read_calibration", spy)
    _seed_plate(run_doc)
    _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    page = _page(fake_microscope, run_doc)
    try:
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        going = page.offsets_step("")
        read_while_going = list(opened)
        Slow.release.set()
        assert page.join(10.0)
        idle = page.offsets_step("")
    finally:
        page.close()

    assert going.wait == busy_note("slow", live=False)
    assert going.reads == ()
    assert read_while_going == []
    # The spy can fail: the same call with nothing going reads both documents.
    assert idle.wait is None
    assert len(opened) == 2


def test_with_the_turrets_labels_unread_the_default_waits_for_them_and_does_not_say_nothing_is_held(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A turret whose positions cannot be read: no document can be checked against it, so none is
    # filtered out for it, and the step's own wait says the turret is the trouble (a list that
    # looked empty would send the operator to calibrate what the page already holds).
    _seed_plate(run_doc)
    _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    labels = _labels(fake_microscope)
    fake_microscope.override(ObjectiveTurret, Turret(labels, FORTY, labels_fail=True))
    page = _page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("")
    finally:
        page.close()

    assert step.wait == OFFSETS_TURRET_UNREAD
    assert step.pairs == ()
    assert [read.objective for read in step.reads] == [FORTY, TWENTY]


def test_the_tools_words_of_a_failed_calibration_are_one_line_and_cut_at_the_cards_bound(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-205: the longest form of a line that carries a tool's words, through the bound.
    words = "the frame is dark\nor flat " + "x" * 700
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    StubCalibration.fails = words
    page = _page(fake_microscope, run_doc)
    try:
        _calibrate(page, FORTY)
        held = page.view().calibrations
    finally:
        page.close()

    problem = held[0].read.problem
    assert problem is not None
    assert "\n" not in problem
    shown = problem.removeprefix(calibration_failed(""))
    assert shown.startswith("the frame is dark or flat xxx")
    assert len(shown) == 601
    assert shown.endswith("…")


def test_the_reason_of_a_calibration_the_halt_cut_says_press_resume_not_the_api_call(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-254: a calibration started on a halted stand ends failed with the layer's real
    # words, which end on a call of the API; the list says the page's.
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    StubCalibration.fails = " ".join(str(MicroscopeHaltedError()).split())
    page = _page(fake_microscope, run_doc)
    try:
        _calibrate(page, FORTY)
        held = page.view().calibrations
    finally:
        page.close()

    assert held[0].read.problem == calibration_failed(
        "the microscope was stopped; press Resume to continue"
    )
    assert "`" not in (held[0].read.problem or "")


def test_a_position_that_cannot_be_looked_at_is_a_line_and_the_others_are_read(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The look at a position is a `stat`, so that a folder the account may not list is said the
    # same way on every Python (`Path.exists` raised it up to 3.13 and hid it from 3.14). Any
    # failure of it but a denial (D22: that one is the folder's, said once) is its position's line:
    # here a name the OS refuses.
    _seed_plate(run_doc)
    held = _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    real = Path.stat

    def stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "position-2.json":
            raise OSError(22, "Invalid argument", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    page = _page(fake_microscope, run_doc)
    try:
        view = page.view()
        found = view.calibrations
    finally:
        page.close()

    assert [h.position for h in found] == [1, 2, 4]
    assert found[0].read.path == str(held["position_1"])
    problem = found[1].read.problem
    assert problem is not None
    assert problem.startswith(
        f"{_folder(run_doc) / 'position-2.json'}: OSError: [Errno 22] Invalid argument"
    )
    assert found[2].read.objective == TWENTY
    # It is not a denial of the folder.
    assert view.calibrations_denied is None


def _deny_the_folder(
    monkeypatch: pytest.MonkeyPatch, *names: str, reason: str = "Permission denied"
) -> None:
    """``Path.stat`` raises ``PermissionError`` for the files of the calibrations folder named in ``names`` (every one when none is): what a folder the account may not search does, on any platform (``chmod 0`` is POSIX only)."""
    real = Path.stat

    def stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.parent.name == "calibrations" and (not names or self.name in names):
            raise PermissionError(13, reason, str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_a_folder_the_account_cannot_look_into_is_one_line_and_no_position(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # D22: twelve near-identical `PermissionError` lines, the path twice in each, and `no
    # calibration yet` under them for every objective, for one folder the account may not search.
    _seed_plate(run_doc)
    _hold(run_doc, position_1=FORTY)
    _deny_the_folder(monkeypatch)
    page = _page(fake_microscope, run_doc)
    try:
        view = page.view()
    finally:
        page.close()

    folder = _folder(run_doc)
    assert view.calibrations == ()
    assert view.calibrations_denied == (
        f"{folder}: the page cannot look into this folder (PermissionError: "
        "Permission denied); which objectives have a calibration is not known"
    )
    # Plain text on one line, naming the folder and the reason and no file (FM-98, FM-139).
    assert "\n" not in (view.calibrations_denied or "")
    assert "position-" not in (view.calibrations_denied or "")
    assert view.offsets is not None


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a folder mode of 0 is POSIX's, and root searches it all the same",
)
def test_a_folder_with_no_right_to_search_it_is_one_line_for_the_real_os_too(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The patched `Path.stat` of the tests above is what runs everywhere; this is the same
    # claim made by the OS itself (macOS and Linux), for the platforms that can (§18.6.1).
    _seed_plate(run_doc)
    _hold(run_doc, position_1=FORTY)
    folder = _folder(run_doc)
    folder.chmod(0)
    page = _page(fake_microscope, run_doc)
    try:
        view = page.view()
    finally:
        folder.chmod(0o700)
        page.close()

    assert view.calibrations == ()
    assert view.calibrations_denied == (
        f"{folder}: the page cannot look into this folder (PermissionError: "
        "Permission denied); which objectives have a calibration is not known"
    )


def test_a_denial_is_said_once_and_the_positions_the_look_reached_are_listed(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    held = _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    # Positions 2, 3 and 5 are denied (three denials, one line); 1 and 4 are looked at.
    _deny_the_folder(
        monkeypatch,
        "position-2.json",
        "position-3.json",
        "position-5.json",
        reason="Access is denied",
    )
    page = _page(fake_microscope, run_doc)
    try:
        view = page.view()
    finally:
        page.close()

    assert [h.read.path for h in view.calibrations] == [
        str(held["position_1"]),
        str(held["position_4"]),
    ]
    assert view.calibrations_denied == (
        f"{_folder(run_doc)}: the page cannot look into this folder (PermissionError: "
        "Access is denied); which objectives have a calibration is not known"
    )


def test_a_denial_that_gives_no_reason_says_so_and_never_the_file(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `PermissionError` raised with no errno or message has no `strerror`: the line says the
    # class and that no reason was given (FM-98: text, and never the error or the file).
    _seed_plate(run_doc)
    real = Path.stat

    def stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.parent.name == "calibrations":
            raise PermissionError()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    page = _page(fake_microscope, run_doc)
    try:
        view = page.view()
    finally:
        page.close()

    assert view.calibrations_denied == (
        f"{_folder(run_doc)}: the page cannot look into this folder (PermissionError: "
        "no reason given); which objectives have a calibration is not known"
    )


def test_a_page_with_no_offer_reads_no_folder_and_says_nothing_of_a_denial(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc, by_eye=True)
    _deny_the_folder(monkeypatch)
    page = _page(fake_microscope, run_doc)
    try:
        view = page.view()
    finally:
        page.close()

    assert view.offsets is None
    assert view.calibrations == ()
    assert view.calibrations_denied is None


def test_the_offsets_wait_is_the_denied_line_when_nothing_else_was_read(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `OFFSETS_NO_DOCUMENT` would send the operator to a button that cannot work (D22).
    _seed_plate(run_doc)
    _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    elsewhere = run_doc.parent / "elsewhere" / "cal-20x.json"
    _seed_calibration(elsewhere, TWENTY)
    _deny_the_folder(monkeypatch)
    page = _page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("")
        # A typed field wins and is never merged with them: the denial is not its business.
        typed = page.offsets_step(_typed(elsewhere))
    finally:
        page.close()

    denied = (
        f"{_folder(run_doc)}: the page cannot look into this folder (PermissionError: "
        "Permission denied); which objectives have a calibration is not known"
    )
    assert step.wait == denied
    assert step.wait != OFFSETS_NO_DOCUMENT
    assert step.pairs == ()
    assert step.reads == ()
    assert typed.wait is None
    assert typed.targets == (TWENTY,)


def test_the_default_measures_with_the_documents_the_look_reached_when_one_position_was_denied(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With a usable document read, the empty field measures with it and the wait is none: the
    # denial is the list's line, said under its heading, and the step does not stop for it.
    _seed_plate(run_doc)
    held = _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    _deny_the_folder(monkeypatch, "position-4.json")
    page = _page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("")
        view = page.view()
    finally:
        page.close()

    assert step.wait is None
    assert step.targets == (FORTY,)
    assert [r.path for r in step.reads] == [str(held["position_1"])]
    assert view.calibrations_denied is not None


class StubPlateParams(PluginParams):
    pass


class StubPlate:
    """A tool that provides ``plate``, to see what a run into another document leaves alone."""

    name = "stub-plate"
    provides = "plate"
    summary = "A test tool."
    Params = StubPlateParams
    Result = PlateResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PlateResult:
        return _walls_plate()


def test_a_run_into_another_document_leaves_the_plate_follow_up_and_the_drives_text_alone(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No caller of `doc_path` provides `plate` today (a calibration provides its own slot), so the
    # gates are tried through the run machinery with a stub that does: the follow-up reads the
    # page's own document, and the drive's text belongs to the plate that document registers.
    followed: list[object] = []

    def follow(self: PlatePage, turned_from: object) -> str:
        followed.append(turned_from)
        return ""

    monkeypatch.setattr(PlatePage, "_follow_plate", follow)
    other = tmp_path / "other" / "document.json"
    other.parent.mkdir()
    page = PlatePage(
        fake_microscope,
        run_doc,
        entry_points=[
            EntryPoint(name="stub-plate", value=f"{__name__}:StubPlate", group=GROUP)
        ],
    )
    try:
        for target, expect_follow in ((other, False), (None, True)):
            prepared = page._prepare("stub-plate", [], doc_path=target)
            assert not isinstance(prepared, str), prepared
            entry, parsed, _planned = prepared
            page._goto = "the drive's text"
            followed.clear()
            ended = page._run_plugin(
                "stub-plate",
                entry,
                parsed,
                threading.Event(),
                None,
                doc_path=target,
            )
            assert ended.status == "done", ended
            assert bool(followed) is expect_follow
            assert (page._goto == "") is expect_follow
    finally:
        page.close()

    assert sorted(RunDocument.load(other).stages) == ["stub-plate"]
    assert sorted(RunDocument.load(run_doc).stages) == ["stub-plate"]
