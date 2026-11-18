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
from smc.hardware.capabilities import ObjectiveTurret
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
    CALIBRATE_NAMES_NONE,
    CALIBRATE_NO_OFFER,
    CALIBRATE_NOT_KNOWN,
    OFFSETS_NO_DOCUMENT,
    OFFSETS_TOO_MANY,
    OFFSETS_TURRET_UNREAD,
    SCAN_HALTED,
    busy_note,
    calibrate_changed,
    calibrate_is_the_plates,
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

    def run(
        self, microscope: Any, params: StubCalibrationParams, ctx: RunContext
    ) -> CameraCalibration:
        StubCalibration.seen.append(params)
        StubCalibration.started.set()
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


def _calibrate(page: PlatePage, expected: str, pairs: tuple[str, ...] = ()) -> str:
    """Press the button as the block does and wait for the run; the outcome the strip would say."""
    page.calibrate_objective(expected, pairs)
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
    _seed_plate(run_doc)
    _hold(
        run_doc,
        position_1=FORTY,
        position_2=TEN,  # the plate's own objective
        position_3="Objective-2",  # a problem, below
        position_4=TWENTY,
        position_5="Nikon 60X Plan Apo",  # not on the turret now
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
    # Nothing is dropped silently: the three left out are in the list the block draws.
    assert [h.position for h in held] == [1, 2, 3, 4, 5]


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
    # same way on every Python (`Path.exists` raised it up to 3.13 and hid it from 3.14).
    _seed_plate(run_doc)
    held = _hold(run_doc, position_1=FORTY, position_4=TWENTY)
    real = Path.stat

    def stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "position-2.json":
            raise PermissionError(13, "Access is denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    page = _page(fake_microscope, run_doc)
    try:
        found = page.view().calibrations
    finally:
        page.close()

    assert [h.position for h in found] == [1, 2, 4]
    assert found[0].read.path == str(held["position_1"])
    problem = found[1].read.problem
    assert problem is not None
    assert problem.startswith(
        f"{_folder(run_doc) / 'position-2.json'}: PermissionError: "
        "[Errno 13] Access is denied"
    )
    assert found[2].read.objective == TWENTY


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
