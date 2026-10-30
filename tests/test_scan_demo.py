"""``scan`` on the demo devices: what is refused before anything moves, what ends a scan, what is resumed (m3 §8, #213).

The plan is a real ``scan-plan`` run on a seeded 96-well plate: wells A1, A2
and B2, 2 x 2 fields each without clipping, so 12 fields, 4 per well, with
the 10x in place and the camera's own 1.0 µm per pixel. The camera is a
``SampleCamera`` with a frame that is not square (FM-73). The XY stage is the
demo's behind a wrapper that counts the moves, can add an offset to every
move (so the readback differs from the target) and can fail a given move;
the turret and the autofocus are fakes put in with ``Microscope.override``
when a test is about them. Every refusal before the scan asserts that
nothing moved and nothing was written.
"""

from __future__ import annotations

import contextlib
import errno
import gc
import math
import shutil
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import tifffile

import smc.plugins.scan.plugin as scan_plugin
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    FocusStep,
    Limits,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.errors import ResumeError
from smc.plugins.geometry import pixel_matrix
from smc.plugins.scan.plugin import ScanCheckpoint, ScanParams, ScanPlugin
from smc.plugins.scan_plan.plugin import ScanPlanParams, ScanPlanPlugin
from smc.plugins.slots import (
    CameraCalibration,
    PlannedField,
    PlateResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
    load_frame,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.run.files import AtomicWriteError
from smc.testing.synthetic import PlateSample, SampleCamera

pytestmark = pytest.mark.demo

SHAPE = (96, 128)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 1.0
IN_PLACE = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
PLAN = {"wells": "A1,A2,B2", "rows": 2, "columns": 2, "clip_to_well": False}
MUTATING = {
    "engage",
    "disengage",
    "set_offset_native",
    "offset_by_native",
    "focus_by_um",
}


# ------------------------------------------------------------------- the stand


class _Stage:
    """The demo's XY stage, counting moves; ``offset_um`` is added to every target.

    The dicts are keyed by the move's number, counted from 1 after it is
    incremented: ``fail`` raises before the inner move, ``landing_error_um``
    is added to that move's target only (a stage that lands short), and
    ``report`` is what that move returns, whatever the inner stage did.
    """

    def __init__(self, inner: XYStage) -> None:
        self.inner = inner
        self.offset_um = (0.0, 0.0)
        self.moves = 0
        self.readbacks: list[XY] = []
        self.fail: dict[int, BaseException] = {}
        self.landing_error_um: dict[int, tuple[float, float]] = {}
        self.report: dict[int, XY] = {}
        self.after_move: Callable[[int], None] | None = None

    def position_um(self) -> XY:
        return self.inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        if self.moves in self.fail:
            raise self.fail[self.moves]
        short_x_um, short_y_um = self.landing_error_um.get(self.moves, (0.0, 0.0))
        readback = self.inner.move_to_um(
            x_um + self.offset_um[0] + short_x_um,
            y_um + self.offset_um[1] + short_y_um,
        )
        if self.moves in self.report:
            readback = self.report[self.moves]
        self.readbacks.append(readback)
        if self.after_move is not None:
            self.after_move(self.moves)
        return readback

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("the scan makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self.inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self.inner.is_busy()

    def stop(self) -> None:
        self.inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self.inner.limits_um()


class _Camera(SampleCamera):
    """A sample camera that counts snaps and can report another size, shape or frame."""

    def __init__(self, microscope: Microscope, pixel_size_um: float = PIXEL_UM) -> None:
        super().__init__(
            microscope, PlateSample(), pixel_size_um=pixel_size_um, shape=SHAPE
        )
        self.snaps = 0
        self.reported_um: float | None = None
        self.reported_shape: tuple[int, int] | None = None
        self.transposed_snap: int | None = None
        self.fail: dict[int, BaseException] = {}

    def snap(self) -> npt.NDArray[np.uint16]:
        self.snaps += 1
        if self.snaps in self.fail:
            raise self.fail[self.snaps]
        if self.snaps == self.transposed_snap:
            return np.zeros((SHAPE[1], SHAPE[0]), dtype=np.uint16)
        return super().snap()

    def image_shape(self) -> tuple[int, int]:
        return self.reported_shape if self.reported_shape is not None else SHAPE

    def pixel_size_um(self) -> float:
        return (
            self.reported_um
            if self.reported_um is not None
            else super().pixel_size_um()
        )


class _Turret:
    """A turret whose objective a test changes by hand; ``select`` is recorded, never wanted."""

    def __init__(self, current: str = IN_PLACE) -> None:
        self.now = current
        self.selects: list[str] = []

    def labels(self) -> list[str]:
        return [FORTY, IN_PLACE]

    def current(self) -> str:
        return self.now

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.selects.append(label)
        self.now = label
        return label

    def pixel_size_um(self, label: str | None = None) -> float:
        return PIXEL_UM


class _Autofocus:
    """An autofocus that records every call; ``engage`` takes its results from a list."""

    def __init__(self, *, engaged: bool, locked: bool) -> None:
        self.engaged = engaged
        self.locked = locked
        self.offset = 4242.0
        self.calls: list[str] = []
        #: What successive ``engage`` calls return (and leave); empty: they lock.
        #: An exception is raised, after the fake switched itself off.
        self.engage_results: list[bool | BaseException] = []

    def lose_lock(self) -> None:
        self.locked = False

    def is_engaged(self) -> bool:
        self.calls.append("is_engaged")
        return self.engaged

    def is_locked(self) -> bool:
        self.calls.append("is_locked")
        return self.engaged and self.locked

    def is_in_range(self) -> bool | None:
        self.calls.append("is_in_range")
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        self.calls.append("engage")
        result = self.engage_results.pop(0) if self.engage_results else True
        # A failed engage leaves it off (the protocol's promise), one that
        # raises as well.
        if isinstance(result, BaseException):
            self.engaged = self.locked = False
            raise result
        self.engaged = self.locked = result
        return result

    def disengage(self) -> None:
        self.calls.append("disengage")
        self.engaged = self.locked = False

    def offset_native(self) -> float:
        self.calls.append("offset_native")
        return self.offset

    def set_offset_native(self, value: float) -> float:
        self.calls.append("set_offset_native")
        return value

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        self.calls.append("offset_by_native")
        return self.offset + delta

    def offset_unit(self) -> str:
        self.calls.append("offset_unit")
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        self.calls.append("offset_limits_native")
        return None

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        self.calls.append("focus_by_um")
        raise AssertionError("the scan never focuses")


@dataclass
class _Rig:
    microscope: Microscope
    stage: _Stage
    camera: _Camera
    run_doc: Path

    @property
    def run_dir(self) -> Path:
        return self.run_doc.parent


def _seed(path: Path, plugin: str, provides: str, result: Any) -> None:
    """A stage record as an earlier run of ``plugin`` would have left it, ``done``."""
    then = datetime(2026, 10, 28, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _plan(rig: _Rig, **params: object) -> ScanPlanResult:
    """Run ``scan-plan`` on the rig as it is; it moves nothing."""
    record = run_plugin(
        ScanPlanPlugin(),
        rig.microscope,
        ScanPlanParams(**{**PLAN, **params}),  # type: ignore[arg-type]
        rig.run_doc,
    )
    assert record.status is StageStatus.done
    assert record.result is not None
    return ScanPlanResult.model_validate(record.result)


def _build(
    microscope: Microscope, tmp_path: Path, *, calibrated_um: float | None = None
) -> _Rig:
    run_doc = tmp_path / "run.json"
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        PlateResult(
            plate="96-well",
            a1_center_xy_um=(0.0, 0.0),  # PlateSample puts A1 at the origin
            rotation_deg=0.0,
            method="seed",
            residual_um=None,
            scale_error=None,
            n_refs=0,
        ),
    )
    if calibrated_um is not None:
        _seed(
            run_doc,
            "camera-calibration",
            "camera-calibration",
            CameraCalibration(
                objective=IN_PLACE,
                image_shape_px=SHAPE,
                matrix_um_per_px=pixel_matrix(calibrated_um),
                pixel_size_um=calibrated_um,
                rotation_deg=0.0,
                mirrored=False,
                residual_um=0.01,
                response=0.97,
                return_error_um=0.4,
                step_um=24.0,
            ),
        )
    camera = _Camera(microscope)
    microscope.override(Camera, camera)
    stage = _Stage(microscope.require(XYStage))
    microscope.override(XYStage, stage)
    return _Rig(microscope, stage, camera, run_doc)


@pytest.fixture
def rig(demo_microscope: Microscope, tmp_path: Path) -> _Rig:
    """The demo stand with a plan of 12 fields in the run document."""
    built = _build(demo_microscope, tmp_path)
    plan = _plan(built)
    assert len(plan.fields) == 12
    assert plan.matrix_source == "profile"
    assert plan.objective == IN_PLACE
    return built


def _scan(
    rig: _Rig,
    *,
    resume: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
    stop: threading.Event | None = None,
    **params: object,
) -> StageRecord:
    return run_plugin(
        ScanPlugin(),
        rig.microscope,
        ScanParams(**params),  # type: ignore[arg-type]
        rig.run_doc,
        resume=resume,
        on_progress=on_progress,
        stop=stop,
    )


def _at(done: int, action: Callable[[], object]) -> Callable[[Progress], None]:
    """A progress callback that runs ``action`` once, when ``done`` fields are done."""

    def callback(progress: Progress) -> None:
        if progress.done == done:
            action()

    return callback


def _result(record: StageRecord) -> ScanResult:
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    return ScanResult.model_validate(record.result)


def _stored(rig: _Rig) -> StageRecord:
    return RunDocument.load(rig.run_doc).stages["scan"]


def _checkpoint(rig: _Rig) -> ScanCheckpoint:
    stored = _stored(rig)
    assert stored.checkpoint is not None
    return ScanCheckpoint.model_validate(stored.checkpoint)


def _refused(rig: _Rig, **params: object) -> str:
    """Run the scan, expect a refusal before anything moved, and return its reason."""
    with pytest.raises(PluginRunError) as raised:
        _scan(rig, **params)
    assert rig.stage.moves == 0
    assert rig.camera.snaps == 0
    assert not (rig.run_dir / "scan").exists()
    stored = _stored(rig)
    assert stored.status is StageStatus.failed
    assert stored.checkpoint is None
    return raised.value.reason


def _tiffs(rig: _Rig) -> list[Path]:
    return sorted((rig.run_dir / "scan").rglob("*.tif"))


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_scan() -> None:
    entry = load("scan")

    assert entry.plugin.provides == "scan"
    assert entry.plugin.requires == (XYStage, Camera)
    assert entry.plugin.reads == ("scan-plan",)
    assert entry.plugin.reads_optional == ()
    assert entry.target == "smc.plugins.scan.plugin:ScanPlugin"


# ---------------------------------------------------------------- the scan


def test_a_scan_writes_one_tiff_per_field_and_records_the_readback(rig: _Rig) -> None:
    plan = _plan(rig)
    rig.stage.offset_um = (3.0, -2.0)
    rig.camera.set_exposure_ms(37.5)
    z_stage = rig.microscope.require(ZStage)
    z0 = z_stage.position_um()

    result = _result(_scan(rig))

    assert result.folder.startswith("scan/")
    assert result.exposure_ms == 37.5
    assert [f.name for f in result.frames] == [f.name for f in plan.fields]
    assert result.failed == ()
    assert result.objective == IN_PLACE
    assert result.pixel_size_um == PIXEL_UM
    assert result.matrix_um_per_px == plan.matrix_um_per_px
    assert result.image_shape_px == SHAPE
    assert result.assumed == ()
    for frame, planned, readback in zip(
        result.frames, plan.fields, rig.stage.readbacks, strict=True
    ):
        assert frame.file == f"{result.folder}/{planned.well}/{planned.name}.tif"
        assert "\\" not in frame.file
        assert (frame.well, frame.index) == (planned.well, planned.index)
        # The readback, not the target: 3 µm right and 2 µm down of it.
        assert frame.x_um == readback.x_um
        assert frame.y_um == readback.y_um
        assert frame.x_um == pytest.approx(planned.x_um + 3.0, abs=0.05)
        assert frame.y_um == pytest.approx(planned.y_um - 2.0, abs=0.05)
        assert frame.z_um == pytest.approx(z0)
        assert frame.focus == "off"  # the demo's autofocus is off: not used
        assert frame.pfs_offset is None
        pixels = load_frame(rig.run_dir, frame)
        assert pixels.shape == SHAPE
        assert pixels.dtype == np.uint16
        with tifffile.TiffFile(rig.run_dir / frame.file) as tiff:
            stored = tiff.shaped_metadata
        assert stored is not None
        written = dict(stored[0])
        written.pop("shape")
        assert written == frame.model_dump(mode="json")
    assert [f.t_s for f in result.frames] == sorted(f.t_s for f in result.frames)
    assert result.frames[-1].t_s > result.frames[0].t_s
    assert len(_tiffs(rig)) == 12
    assert z_stage.position_um() == pytest.approx(z0)  # never moved


def test_progress_counts_every_field_of_the_attempt(rig: _Rig) -> None:
    seen: list[Progress] = []

    _result(_scan(rig, on_progress=seen.append))

    assert [(p.done, p.total) for p in seen] == [(n, 12) for n in range(1, 13)]
    assert seen[0].message == "A1_0000"


# ------------------------------------------------- refused before anything moves


def test_the_wrong_objective_is_refused_before_anything_moves(rig: _Rig) -> None:
    turret = _Turret(current=FORTY)
    rig.microscope.override(ObjectiveTurret, turret)

    reason = _refused(rig)

    assert reason == (
        f"the objective in place is {FORTY!r}; the scan was planned for {IN_PLACE!r}. "
        f"Turn the turret to {IN_PLACE!r} (smc never turns it during a scan), or "
        f"run scan-plan again with objective={FORTY}"
    )
    assert turret.selects == []


def test_a_changed_frame_shape_is_refused_before_anything_moves(rig: _Rig) -> None:
    rig.camera.reported_shape = (SHAPE[1], SHAPE[0])

    reason = _refused(rig)

    assert reason == (
        "the camera gives frames of (128, 96) pixels; the plan was made for "
        "(96, 128): set the binning and region of interest back, or run "
        "scan-plan again with the stand as it is now"
    )


def test_a_profile_plan_for_another_size_is_refused_before_anything_moves(
    rig: _Rig,
) -> None:
    rig.camera.reported_um = 1 / 1.5

    reason = _refused(rig)

    assert reason == (
        "the camera reports 0.666667 µm per pixel; the plan was made for 1 µm "
        "(profile), more than 1% away (a binning, the magnifier lever): run "
        "scan-plan again with the stand as it is now"
    )


def test_a_calibrated_plan_is_compared_within_ten_percent(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    rig = _build(demo_microscope, tmp_path, calibrated_um=1.03)
    plan = _plan(rig)
    assert plan.matrix_source == "camera-calibration"
    assert plan.pixel_size_um == 1.03

    # The camera's 1.0 is 3 % from the calibration's 1.03: a measurement, not a
    # change. Each well compares with the camera's 1.0 at the start, not the plan's.
    assert len(_result(_scan(rig)).frames) == 12

    rig.stage.moves = 0
    rig.camera.snaps = 0
    rig.camera.reported_um = 1.5
    with pytest.raises(PluginRunError) as raised:
        _scan(rig)
    assert raised.value.reason == (
        "the camera reports 1.5 µm per pixel; the plan was made for 1.03 µm "
        "(camera-calibration), more than 10% away (a binning, the magnifier "
        "lever): run camera-calibration, then scan-plan, again with the stand "
        "as it is now"
    )
    assert rig.stage.moves == 0
    assert rig.camera.snaps == 0


def test_a_parameter_pixel_size_is_not_compared_and_says_so(rig: _Rig) -> None:
    plan = _plan(rig, pixel_size_um=2.0)
    assert plan.matrix_source == "parameter"

    result = _result(_scan(rig))

    assert len(result.frames) == 12
    assert result.pixel_size_um == 2.0
    assert result.assumed == (
        "the plan's pixel size was given as a parameter: not compared with the "
        "camera's",
    )


@pytest.mark.parametrize("bad", [0.0, math.inf, math.nan])
def test_a_camera_without_a_pixel_size_is_not_compared_and_says_so(
    bad: float, rig: _Rig
) -> None:
    """FM-104: 0 is "not measured" and infinity is above 0; neither is compared, both are said."""
    rig.camera.reported_um = bad

    result = _result(_scan(rig))

    assert len(result.frames) == 12
    assert result.assumed == (
        "the camera reports no pixel size: the plan's size and a change during "
        "the scan were not checked",
    )


@pytest.mark.parametrize(
    ("engaged", "state"), [(False, "off"), (True, "on but not locked")]
)
def test_required_autofocus_that_is_not_locked_is_refused_before_anything_moves(
    engaged: bool, state: str, rig: _Rig
) -> None:
    af = _Autofocus(engaged=engaged, locked=False)
    rig.microscope.override(Autofocus, af)

    reason = _refused(rig, autofocus="required")

    assert reason == (
        f"autofocus=required, and the autofocus is {state}: engage PFS on the "
        "stand and let it lock on the sample before the scan"
    )
    assert not MUTATING & set(af.calls)


def test_required_autofocus_on_a_stand_without_one_is_refused(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = rig.microscope.get
    monkeypatch.setattr(
        rig.microscope,
        "get",
        lambda wanted: None if wanted is Autofocus else real_get(wanted),
    )

    reason = _refused(rig, autofocus="required")

    assert reason.startswith("autofocus=required, and the autofocus is missing: ")


def test_an_engaged_autofocus_that_is_not_locked_is_refused_before_anything_moves(
    rig: _Rig,
) -> None:
    af = _Autofocus(engaged=True, locked=False)
    rig.microscope.override(Autofocus, af)

    reason = _refused(rig)  # if-engaged

    assert reason == (
        "the autofocus is on but not locked: lock it on the sample, or switch it "
        "off and run with autofocus=off"
    )
    assert not MUTATING & set(af.calls)


def test_too_little_disk_is_refused_before_moving(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    need = math.ceil(1.1 * 12 * 96 * 128 * 2)  # 12 frames of 96 x 128 at 16 bits
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=need))

    reason = _refused(rig)

    assert reason == (
        f"the run folder's disk has {need / 1e6:,.1f} MB free and the 12 frames "
        f"need {need / 1e6:,.1f} MB: free some space, or put the run document on "
        "another disk"
    )

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: SimpleNamespace(free=need + 1)
    )
    assert len(_result(_scan(rig)).frames) == 12


def test_the_disk_counts_only_the_fields_left_on_a_resume(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = threading.Event()
    _scan(rig, stop=stop, on_progress=_at(4, stop.set))
    assert _checkpoint(rig).wells_done == ("A1",)

    need = math.ceil(1.1 * 8 * 96 * 128 * 2)  # A2 and B2 are left
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: SimpleNamespace(free=need + 1)
    )
    assert len(_result(_scan(rig, resume=True)).frames) == 12


def test_a_disk_that_cannot_be_read_is_refused_before_moving(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dropped(_path: object) -> object:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(shutil, "disk_usage", dropped)

    assert _refused(rig) == (
        "the run folder cannot be read (Input/output error): check the disk or the "
        "network share, then run scan --resume"
    )


def _edit_plan_result(rig: _Rig, edit: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite the stored plan as a copied or hand-edited run document would; its fields are copies."""
    record = RunDocument.load(rig.run_doc).stages["scan-plan"]
    assert record.result is not None
    result = {**record.result, "fields": [dict(f) for f in record.result["fields"]]}
    edit(result)
    write_stage(rig.run_doc, record.model_copy(update={"result": result}))


def _edit_plan(rig: _Rig, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    """Rewrite the stored plan's fields."""
    _edit_plan_result(rig, lambda result: edit(result["fields"]))


def _rename_second(fields: list[dict[str, Any]]) -> None:
    fields[1]["name"] = "../x"


def _duplicate_first(fields: list[dict[str, Any]]) -> None:
    fields[1]["name"] = fields[0]["name"]


def _foreign_well(fields: list[dict[str, Any]]) -> None:
    fields[1]["well"] = "Z9"
    fields[1]["name"] = "Z9_0000"


def _device_name(fields: list[dict[str, Any]]) -> None:
    fields[1]["name"] = "nul"


def _case_only(fields: list[dict[str, Any]]) -> None:
    fields[1]["name"] = fields[0]["name"].lower()


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (
            _rename_second,
            "field '../x' of well 'A1': '../x' cannot be a frame's field name in a "
            "file path: only letters, digits, '_' and '-' are",
        ),
        (
            _duplicate_first,
            "two fields of well A1 are named A1_0000, so they would write one "
            "file: run scan-plan again with the stand as it is now",
        ),
        (
            _foreign_well,
            "field 'Z9_0000' is of well 'Z9', which the scan plan does not list: "
            "run scan-plan again with the stand as it is now",
        ),
        (
            _device_name,
            "field 'nul' of well 'A1': 'nul' is a Windows device name and cannot be "
            "a frame's field name in a file path",
        ),
        (
            _case_only,
            "fields 'A1_0000' of well 'A1' and 'a1_0000' of well 'A1' would write "
            "one file, as the file system does not tell upper from lower case: run "
            "scan-plan again with the stand as it is now",
        ),
    ],
    ids=["dot-dot", "duplicate", "foreign-well", "device-name", "case-only"],
)
def test_a_plan_whose_paths_cannot_be_written_is_refused_before_anything_moves(
    edit: Callable[[list[dict[str, Any]]], None], reason: str, rig: _Rig
) -> None:
    """FM-24: the plan is a run document's, and may have been copied or edited."""
    _edit_plan(rig, edit)

    assert _refused(rig) == reason


def test_two_fields_of_wells_that_differ_only_in_case_would_write_one_file(
    rig: _Rig,
) -> None:
    """The well is folded too: ``a1`` and ``A1`` are one folder on NTFS and the default APFS."""

    def edit(result: dict[str, Any]) -> None:
        result["wells"] = [*result["wells"], "a1"]
        result["fields"][1]["well"] = "a1"
        result["fields"][1]["name"] = result["fields"][0]["name"]

    _edit_plan_result(rig, edit)

    assert _refused(rig) == (
        "fields 'A1_0000' of well 'A1' and 'A1_0000' of well 'a1' would write one "
        "file, as the file system does not tell upper from lower case: run "
        "scan-plan again with the stand as it is now"
    )


def test_the_paths_are_resolved_once_before_the_scan(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-116: a ``resolve`` at every frame is a share that can drop at every frame (24 calls before)."""
    real = scan_plugin.resolve_in_run
    calls: list[str] = []

    def counted(run_dir: Path, relative: str) -> Path:
        calls.append(relative)
        return real(run_dir, relative)

    monkeypatch.setattr(scan_plugin, "resolve_in_run", counted)

    assert len(_result(_scan(rig)).frames) == 12

    assert len(calls) == 12
    assert len(set(calls)) == 12


def test_a_share_that_drops_while_the_paths_are_checked_is_unreadable_not_a_plan_error(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-116: a ``ValueError`` whose cause is an ``OSError`` is the folder, not the plan's text."""

    def dropped(run_dir: Path, relative: str) -> Path:
        cause = OSError(64, "The specified network name is no longer available")
        raise ValueError(f"{relative!r} cannot be resolved") from cause

    monkeypatch.setattr(scan_plugin, "resolve_in_run", dropped)

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        "the run folder cannot be read (The specified network name is no longer "
        "available): check the disk or the network share, then run scan --resume"
    )
    assert raised.value.__context__ is None  # raised outside the handler (FM-98)
    assert rig.stage.moves == 0
    assert not (rig.run_dir / "scan").exists()


# The longest path the scan writes is the temporary file beside a frame, 18
# characters longer than its name. Every frame of the 12 has the same length.
_TEMPORARY_BESIDE_A_FRAME = len("/scan/" + "t" * 12 + "/A1/A1_0000.tif") + 18


def _longest_path_chars(rig: _Rig) -> int:
    return len(str(rig.run_dir.resolve())) + _TEMPORARY_BESIDE_A_FRAME


def test_a_path_that_windows_would_refuse_is_refused_before_anything_moves(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 4: a path of 260 characters or more fails at the first write on a PC without long paths."""
    n = _longest_path_chars(rig)
    monkeypatch.setattr(scan_plugin, "ON_WINDOWS", True)
    monkeypatch.setattr(scan_plugin, "MAX_PATH", n)

    assert _refused(rig) == (
        f"a frame's path would reach {n} characters while it is written, and "
        f"Windows takes {n - 1} at most unless long paths are enabled (not "
        "assumed): move the run document to a shorter folder, then run scan again"
    )


def test_a_path_one_short_of_the_limit_is_written(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scan_plugin, "ON_WINDOWS", True)
    monkeypatch.setattr(scan_plugin, "MAX_PATH", _longest_path_chars(rig) + 1)

    assert len(_result(_scan(rig)).frames) == 12


def test_a_long_path_is_not_refused_off_windows(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scan_plugin, "ON_WINDOWS", False)
    monkeypatch.setattr(scan_plugin, "MAX_PATH", 1)

    assert len(_result(_scan(rig)).frames) == 12


def _longer_last_name(fields: list[dict[str, Any]]) -> None:
    fields[-1]["name"] = "B2_0003_a_longer_name"


def test_the_longest_path_of_all_the_fields_is_the_one_checked(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the first field's: a well of a bigger plate, or a longer name, comes later."""
    _edit_plan(rig, _longer_last_name)
    longest = len(str(rig.run_dir.resolve())) + len(
        "/scan/" + "t" * 12 + "/B2/B2_0003_a_longer_name.tif"
    )
    n = longest + 18
    monkeypatch.setattr(scan_plugin, "ON_WINDOWS", True)
    monkeypatch.setattr(scan_plugin, "MAX_PATH", n)

    # The first eleven fields are shorter than the limit: only the last is refused.
    assert _refused(rig) == (
        f"a frame's path would reach {n} characters while it is written, and "
        f"Windows takes {n - 1} at most unless long paths are enabled (not "
        "assumed): move the run document to a shorter folder, then run scan again"
    )

    monkeypatch.setattr(scan_plugin, "MAX_PATH", n + 1)
    assert len(_result(_scan(rig)).frames) == 12


# ------------------------------------------------------------- at every well


def test_a_turret_turned_mid_well_stops_the_scan_at_the_end_of_that_well(
    rig: _Rig,
) -> None:
    """Turned after A2's second field: A2 is not marked done, A1 is kept (FM-103)."""
    turret = _Turret()
    rig.microscope.override(ObjectiveTurret, turret)

    def turn() -> None:
        turret.now = FORTY

    with pytest.raises(PluginRunError) as raised:
        _scan(rig, on_progress=_at(6, turn))

    assert raised.value.reason == (
        f"the objective in place is now {FORTY!r} at the end of well A2; the scan "
        f"was planned for {IN_PLACE!r}: turn the turret back to {IN_PLACE!r}, then "
        "run scan --resume"
    )
    assert _stored(rig).status is StageStatus.failed
    checkpoint = _checkpoint(rig)
    # A2's last two frames were taken under the 40x: A2 is not marked done.
    assert checkpoint.wells_done == ("A1",)
    assert len(checkpoint.frames) == 4
    assert rig.stage.moves == 8
    assert turret.selects == []


def test_a_turret_turned_after_a_wells_last_field_does_not_keep_that_well(
    rig: _Rig,
) -> None:
    """Turned after A1's last frame: refused before A2's first move, A1 not kept.

    The check at the end of A1 runs after A1's last progress report, so it sees
    the turn; A1's frames were all taken before it, but the check cannot tell
    when the turret turned, and redoing a well is cheaper than a wrong one.
    """
    turret = _Turret()
    rig.microscope.override(ObjectiveTurret, turret)

    def turn() -> None:
        turret.now = FORTY

    with pytest.raises(PluginRunError, match="at the end of well A1;"):
        _scan(rig, on_progress=_at(4, turn))

    assert rig.stage.moves == 4
    assert _stored(rig).checkpoint is None  # no well was marked done


@pytest.mark.parametrize(
    ("now_um", "text"),
    [(1 / 1.5, "0.666667"), (1.05, "1.05"), (math.nan, "nan")],
    ids=["lever", "five-percent", "nan"],
)
def test_a_changed_pixel_size_stops_the_scan(
    now_um: float, text: str, rig: _Rig
) -> None:
    """FM-103 (the lever), and a 5 % drift that a calibration's 10 % would let through; FM-104."""

    def lever() -> None:
        rig.camera.reported_um = now_um

    with pytest.raises(PluginRunError) as raised:
        _scan(rig, on_progress=_at(6, lever))

    assert raised.value.reason == (
        f"the camera reports {text} µm per pixel at the end of well A2, 1 at the "
        "start of the scan (the magnifier lever, a binning): set it back, then "
        "run scan --resume"
    )
    assert rig.stage.moves == 8
    assert _checkpoint(rig).wells_done == ("A1",)


def test_a_pixel_size_within_one_percent_does_not_stop_the_scan(rig: _Rig) -> None:
    def drift() -> None:
        rig.camera.reported_um = 1.005

    assert len(_result(_scan(rig, on_progress=_at(6, drift))).frames) == 12


def test_a_frame_shape_changed_mid_scan_stops_at_the_end_of_its_well(
    rig: _Rig,
) -> None:
    def binning() -> None:
        rig.camera.reported_shape = (48, 64)

    with pytest.raises(
        PluginRunError, match=r"\(48, 64\) pixels at the end of well A2"
    ):
        _scan(rig, on_progress=_at(6, binning))
    assert rig.stage.moves == 8
    assert _checkpoint(rig).wells_done == ("A1",)


def test_a_frame_of_another_shape_ends_the_scan(rig: _Rig) -> None:
    plan = _plan(rig)
    rig.camera.transposed_snap = 3

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        "the frame of A1_0002 has shape (128, 96); the plan's frames are (96, 128): "
        "check the camera's binning and region of interest, then run scan --resume"
    )
    third = plan.fields[2]
    assert not any(path.stem == third.name for path in _tiffs(rig))
    assert len(_tiffs(rig)) == 2


# -------------------------------------------------------------- failed fields


def test_one_failed_field_is_recorded_and_skipped(rig: _Rig) -> None:
    plan = _plan(rig)
    rig.stage.fail = {2: HardwareError("the stage did not answer\nsecond line")}

    result = _result(_scan(rig))

    second = plan.fields[1]
    assert len(result.failed) == 1
    failed = result.failed[0]
    assert (failed.well, failed.index, failed.name) == (
        second.well,
        second.index,
        second.name,
    )
    assert (failed.x_um, failed.y_um) == (second.x_um, second.y_um)  # the target
    assert failed.error == "HardwareError: the stage did not answer"
    assert len(result.frames) == 11
    assert second.name not in {f.name for f in result.frames}
    assert len(_tiffs(rig)) == 11


def test_a_refused_move_is_a_failed_field(rig: _Rig) -> None:
    rig.stage.fail = {2: SafetyRefusedError("the target is outside the soft limits")}

    result = _result(_scan(rig))

    assert len(result.failed) == 1
    assert result.failed[0].error.startswith("SafetyRefusedError: ")
    assert len(result.frames) == 11


def test_three_failed_fields_in_a_row_fail_the_scan_with_the_checkpoint_kept(
    rig: _Rig,
) -> None:
    rig.stage.fail = {n: HardwareError(f"no answer {n}") for n in (5, 6, 7)}

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        "3 fields failed in a row, the last A2_0002: HardwareError: no answer 7; "
        "the checkpoint keeps 1 wells: check the stage, then run scan --resume"
    )
    assert raised.value.__context__ is None  # raised outside the handler
    stored = _stored(rig)
    assert stored.status is StageStatus.failed
    checkpoint = _checkpoint(rig)
    assert checkpoint.wells_done == ("A1",)
    assert len(checkpoint.frames) == 4
    assert checkpoint.failed == ()


def test_failures_not_in_a_row_do_not_fail_the_scan(rig: _Rig) -> None:
    rig.stage.fail = {n: HardwareError("no answer") for n in (2, 4)}

    result = _result(_scan(rig, max_failed_in_row=2))

    assert [f.name for f in result.failed] == ["A1_0001", "A1_0003"]
    assert len(result.frames) == 10


def test_failures_in_a_row_are_counted_across_a_well(rig: _Rig) -> None:
    """A1's last field and A2's first two: three in a row, though in two wells."""
    rig.stage.fail = {n: HardwareError(f"no answer {n}") for n in (4, 5, 6)}

    with pytest.raises(
        PluginRunError, match="3 fields failed in a row, the last A2_0001: "
    ):
        _scan(rig)

    checkpoint = _checkpoint(rig)
    assert checkpoint.wells_done == ("A1",)
    # A1 was marked done with its failed field, which the checkpoint keeps.
    assert [f.name for f in checkpoint.failed] == ["A1_0003"]
    assert len(checkpoint.frames) == 3


def test_a_scan_where_every_field_failed_is_failed_with_its_checkpoint_kept(
    rig: _Rig,
) -> None:
    """A stage that refuses 12 moves is not a scan of 12 failed fields that is ``done``: ``cells`` would refuse it hours later."""
    timeout = DeviceTimeoutError("the stage did not answer in 60 s")
    rig.stage.fail = {n: timeout for n in range(1, 13)}
    reason = (
        "every field failed (12), the first A1_0000: DeviceTimeoutError: the stage "
        "did not answer in 60 s: check the stage and the camera, then run scan "
        "again without --resume (a resume skips the wells already walked)"
    )

    with pytest.raises(PluginRunError) as raised:
        _scan(rig, max_failed_in_row=100)

    assert raised.value.reason == reason
    assert _stored(rig).status is StageStatus.failed
    checkpoint = _checkpoint(rig)
    assert checkpoint.wells_done == ("A1", "A2", "B2")
    assert len(checkpoint.failed) == 12
    assert checkpoint.frames == ()
    assert rig.camera.snaps == 0
    assert _tiffs(rig) == []

    # The checkpoint keeps every well, so a resume meets the same refusal, with no move.
    with pytest.raises(PluginRunError) as again:
        _scan(rig, resume=True, max_failed_in_row=100)
    assert again.value.reason == reason
    assert rig.stage.moves == 12

    # The way out is the one the message says: without --resume, on a stage that answers.
    rig.stage.fail = {}
    assert len(_result(_scan(rig)).frames) == 12


def test_one_frame_among_failed_fields_keeps_the_scan_done(rig: _Rig) -> None:
    rig.stage.fail = {n: HardwareError("no answer") for n in range(1, 12)}

    result = _result(_scan(rig, max_failed_in_row=100))

    assert len(result.frames) == 1
    assert len(result.failed) == 11


def test_a_resume_whose_remaining_fields_all_failed_keeps_the_frames_and_is_done(
    rig: _Rig,
) -> None:
    """The checkpoint's four frames are a scan: the failed wells are listed, not the end of it.

    Stopped here and not by the ``stopped`` fixture: a resume refuses parameters
    that differ, and this one runs with ``max_failed_in_row=100``.
    """
    stop = threading.Event()
    _scan(rig, stop=stop, on_progress=_at(5, stop.set), max_failed_in_row=100)
    assert _checkpoint(rig).wells_done == ("A1",)
    rig.stage.moves = 0
    rig.stage.fail = {n: HardwareError("no answer") for n in range(1, 9)}

    result = _result(_scan(rig, resume=True, max_failed_in_row=100))

    assert len(result.frames) == 4
    assert len(result.failed) == 8
    assert rig.stage.moves == 8


def test_a_dry_run_where_every_move_failed_is_failed_too(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    """In a dry run the walked moves are the fields: none walked is a scan of nothing."""
    rig = _build(demo_microscope_dry, tmp_path)
    _plan(rig)
    rig.stage.fail = {n: HardwareError("no answer") for n in range(1, 13)}

    with pytest.raises(PluginRunError) as raised:
        _scan(rig, max_failed_in_row=100)

    assert raised.value.reason == (
        "every field failed (12), the first A1_0000: HardwareError: no answer: "
        "check the stage and the camera, then run scan again without --resume (a "
        "resume skips the wells already walked)"
    )
    assert _stored(rig).status is StageStatus.failed


def test_a_dry_run_with_one_failed_move_among_the_walked_ones_is_done(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    """The walked moves are a dry run's frames: eleven of them make a scan."""
    rig = _build(demo_microscope_dry, tmp_path)
    _plan(rig)
    rig.stage.fail = {3: HardwareError("no answer")}

    result = _result(_scan(rig, max_failed_in_row=100))

    assert [f.name for f in result.failed] == ["A1_0002"]
    assert rig.stage.moves == 12
    assert result.assumed == ("dry run: 12 fields walked, no frame taken",)


def test_a_snap_that_times_out_is_a_failed_field(rig: _Rig) -> None:
    plan = _plan(rig)
    rig.camera.fail = {3: DeviceTimeoutError("the camera did not answer in 5 s")}

    result = _result(_scan(rig))

    assert [f.name for f in result.failed] == [plan.fields[2].name]
    assert result.failed[0].error == (
        "DeviceTimeoutError: the camera did not answer in 5 s"
    )
    assert len(result.frames) == 11
    assert rig.stage.moves == 12


def test_a_halt_is_not_a_failed_field(rig: _Rig) -> None:
    """FM-64: the emergency stop ends the scan; it is not a field to skip."""
    try:
        with pytest.raises(MicroscopeHaltedError):
            _scan(rig, on_progress=_at(5, rig.microscope.stop))
    finally:
        rig.microscope.resume()

    assert _stored(rig).status is StageStatus.failed
    checkpoint = _checkpoint(rig)
    assert checkpoint.failed == ()
    assert checkpoint.wells_done == ("A1",)
    assert rig.stage.moves == 6
    assert len(_tiffs(rig)) == 5


def test_a_stopped_motion_is_not_a_failed_field(rig: _Rig) -> None:
    rig.stage.fail = {3: MotionStoppedError("xy_stage XY")}

    with pytest.raises(MotionStoppedError):
        _scan(rig)

    assert _stored(rig).status is StageStatus.failed
    assert _stored(rig).checkpoint is None
    assert rig.stage.moves == 3


def _fail_write(monkeypatch: pytest.MonkeyPatch, call: int, error: OSError) -> None:
    """Make the ``call``-th frame write of the scan raise ``error``; the others are real."""
    real = scan_plugin.write_atomically
    calls = 0

    def write(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        nonlocal calls
        calls += 1
        if calls == call:
            raise error
        real(path, data, make_parents=make_parents)

    monkeypatch.setattr(scan_plugin, "write_atomically", write)


def _token(rig: _Rig) -> str:
    """The folder of the frames under ``scan/``: the token the scan chose."""
    (folder,) = (rig.run_dir / "scan").iterdir()
    return folder.name


def test_a_full_disk_writing_a_frame_ends_the_scan_with_what_to_do(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_write(
        monkeypatch,
        2,
        AtomicWriteError(OSError(errno.ENOSPC, "No space left on device")),
    )

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        f"the frame scan/{_token(rig)}/A1/A1_0001.tif could not be written (No "
        "space left on device): free some space if the disk is full, or check the "
        "network share, then run scan --resume"
    )
    assert raised.value.__context__ is None  # raised outside the handler (FM-98)
    stored = _stored(rig)
    assert stored.status is StageStatus.failed
    assert stored.checkpoint is None  # no well was done: no failed field anywhere
    assert rig.stage.moves == 2


@pytest.mark.parametrize(
    "error",
    [
        AtomicWriteError(
            OSError(64, "The specified network name is no longer available")
        ),
        OSError(64, "The specified network name is no longer available"),
        # Only the final rename failed, so the temporary file is kept: a share
        # that dropped is still not a viewer holding the file open.
        AtomicWriteError(
            OSError(64, "The specified network name is no longer available"),
            kept=Path("kept.smc-tmp"),
        ),
    ],
    ids=["atomic-write-error", "plain-oserror", "rename-failed-with-kept"],
)
def test_a_share_that_drops_while_a_frame_is_written_says_check_the_share(
    error: OSError, rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-116: a share that drops is an ``OSError`` of the write, whoever raises it, and never a ``ValueError``."""
    _fail_write(monkeypatch, 3, error)

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        f"the frame scan/{_token(rig)}/A1/A1_0002.tif could not be written (The "
        "specified network name is no longer available): free some space if the "
        "disk is full, or check the network share, then run scan --resume"
    )
    assert rig.stage.moves == 3


def test_a_frame_held_open_is_a_plugin_run_error_that_says_close_it(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows refuses to replace a file a viewer holds open: ``kept`` is set, the cause is a ``PermissionError``."""
    held = AtomicWriteError(
        PermissionError(errno.EACCES, "Permission denied"), kept=Path("held.smc-tmp")
    )
    # An AtomicWriteError is an OSError, not a PermissionError: the cause says which.
    assert not isinstance(held, PermissionError)
    _fail_write(monkeypatch, 2, held)

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        f"the frame scan/{_token(rig)}/A1/A1_0001.tif cannot be replaced: another "
        "program holds it open (a viewer, a backup or antivirus agent): close it, "
        "then run scan --resume"
    )
    assert raised.value.__context__ is None  # raised outside the handler (FM-98)
    assert rig.stage.moves == 2
    assert _stored(rig).status is StageStatus.failed


def test_a_permission_error_with_nothing_kept_is_not_a_held_file(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only folder refuses the temporary file: nothing is kept, and no viewer holds anything."""
    _fail_write(
        monkeypatch,
        2,
        AtomicWriteError(PermissionError(errno.EACCES, "Permission denied")),
    )

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        f"the frame scan/{_token(rig)}/A1/A1_0001.tif could not be written "
        "(Permission denied): free some space if the disk is full, or check the "
        "network share, then run scan --resume"
    )


def test_an_interrupt_writing_a_frame_is_not_turned_into_a_run_error(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an ``OSError`` is the write's failure: the operator's Ctrl-C passes through."""

    def interrupted(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(scan_plugin, "write_atomically", interrupted)

    with pytest.raises(KeyboardInterrupt):
        _scan(rig)


class _Held:
    """Something only a callee's frame holds: alive exactly as long as that frame is."""

    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir = run_dir


def test_a_failed_write_leaves_no_reference_cycle(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FM-98: the caught ``OSError``'s traceback reaches ``_write_frame``'s frame, so no local may keep it.

    A frame kept alive keeps what it holds (the core, through ``run()``'s frame)
    until the cyclic collector, which an exception escaping the command does not
    wait for. The collector is off here, so only reference counts free it.
    """
    plan = _plan(rig)
    planned = plan.fields[0]
    target = scan_plugin._Target("scan/tok/A1/A1_0000.tif", tmp_path / "A1_0000.tif")
    meta = ScanFrame(
        well=planned.well,
        index=planned.index,
        name=planned.name,
        file=target.relative,
        x_um=planned.x_um,
        y_um=planned.y_um,
        z_um=None,
        focus="none",
        pfs_offset=None,
        t_s=0.0,
    )

    def full_disk(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        raise AtomicWriteError(OSError(errno.ENOSPC, "No space left on device"))

    monkeypatch.setattr(scan_plugin, "write_atomically", full_disk)
    pixels = np.zeros(SHAPE, dtype=np.uint16)
    alive = weakref.ref(pixels)
    gc.disable()
    try:
        with contextlib.suppress(PluginRunError):
            scan_plugin._write_frame(plan, planned, target, pixels, meta)
        del pixels
        assert alive() is None
    finally:
        gc.enable()


def test_a_refused_path_check_leaves_no_reference_cycle(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-98: the dropped share's ``OSError`` carries the resolver's frame, whose caller is ``_paths``."""
    plan = _plan(rig)

    def dropped(run_dir: Path, relative: str) -> Path:
        try:
            raise OSError(64, "The specified network name is no longer available")
        except OSError as exc:
            raise ValueError(f"{relative!r} cannot be resolved") from exc

    monkeypatch.setattr(scan_plugin, "resolve_in_run", dropped)
    ctx = _Held(rig.run_dir)
    alive = weakref.ref(ctx)
    gc.disable()
    try:
        with contextlib.suppress(PluginRunError):
            scan_plugin._paths(ctx, plan, "0123456789ab")  # type: ignore[arg-type]
        del ctx
        assert alive() is None
    finally:
        gc.enable()


# ---------------------------------------------------------------- the landing


def _off_target_line(planned: PlannedField, by_um: float) -> str:
    """A failed field's error for a landing ``by_um`` along x, against the 10x's 4.8 µm.

    Built from the plan's field and the error the test added, not read from
    the code: the demo rounds a command by up to 0.015 µm, which no ``.1f``
    digit of these numbers shows.
    """
    return (
        f"StageOffTarget: the readback is {by_um:.1f} µm from the target, more "
        f"than 4.8 µm (target ({planned.x_um:.1f}, {planned.y_um:.1f}), readback "
        f"({planned.x_um + by_um:.1f}, {planned.y_um:.1f}))"
    )


def test_a_field_that_lands_off_target_is_a_failed_field_and_nothing_else_is_done(
    rig: _Rig,
) -> None:
    """FM-104: a move that returned is not a move that arrived; no frame of the wrong place."""
    plan = _plan(rig)
    af = _Autofocus(engaged=True, locked=True)
    rig.microscope.override(Autofocus, af)
    rig.stage.landing_error_um = {2: (6.0, 0.0)}

    result = _result(_scan(rig))

    second = plan.fields[1]
    assert len(result.failed) == 1
    failed = result.failed[0]
    assert (failed.well, failed.index, failed.name) == (
        second.well,
        second.index,
        second.name,
    )
    assert (failed.x_um, failed.y_um) == (second.x_um, second.y_um)  # the target
    assert failed.error == _off_target_line(second, 6.0)
    assert len(result.frames) == 11
    assert second.name not in {f.name for f in result.frames}
    assert len(_tiffs(rig)) == 11
    assert rig.camera.snaps == 11
    # The failed field read nothing from the autofocus: the lock once before the
    # scan and once for each of the 11 frames, the offset for the frames only.
    assert af.calls.count("is_locked") == 12
    assert af.calls.count("offset_native") == 11
    assert "engage" not in af.calls


def test_a_field_off_target_does_not_engage_the_autofocus(rig: _Rig) -> None:
    """An ``engage()`` drives Z: it is not spent on a field that is not photographed."""
    af = _Autofocus(engaged=True, locked=True)
    rig.microscope.override(Autofocus, af)
    rig.stage.landing_error_um = {2: (6.0, 0.0)}
    rig.stage.after_move = lambda n: af.lose_lock() if n in (2, 3) else None

    result = _result(_scan(rig))

    assert len(result.failed) == 1
    assert af.calls.count("engage") == 1  # field 3's, not field 2's
    assert [f.focus for f in result.frames] == ["locked"] * 11


def test_a_landing_within_the_tolerance_is_a_frame(rig: _Rig) -> None:
    plan = _plan(rig)
    rig.stage.landing_error_um = {2: (4.0, 0.0)}

    result = _result(_scan(rig))

    assert len(result.frames) == 12
    assert result.failed == ()
    assert result.frames[1].x_um == pytest.approx(plan.fields[1].x_um + 4.0, abs=0.05)


@pytest.mark.parametrize(
    ("landing", "failed"),
    [((3.5, 3.5), 1), ((3.5, 0.0), 0), ((0.0, -3.5), 0), ((-3.5, -3.5), 1)],
    ids=["both-axes", "x-only", "y-only", "both-negative"],
)
def test_the_landing_error_is_the_distance_not_one_axis(
    landing: tuple[float, float], failed: int, rig: _Rig
) -> None:
    """3.5 µm on each axis is 4.95 µm off, past the 4.8 µm that neither axis reaches."""
    rig.stage.landing_error_um = {2: landing}

    result = _result(_scan(rig))

    assert len(result.failed) == failed
    assert len(result.frames) == 12 - failed


@pytest.mark.parametrize(
    "landing", [(5.5, 0.0), (0.0, 5.5)], ids=["along-x", "along-y"]
)
def test_the_tolerance_is_of_the_smaller_side_of_the_field(
    landing: tuple[float, float], rig: _Rig
) -> None:
    """FM-73: the field is 128 x 96 µm; 5 % of the larger side (6.4 µm) would take a 5.5 µm landing."""
    plan = _plan(rig)
    assert plan.fov_um == (128.0, 96.0)
    rig.stage.landing_error_um = {2: landing}

    result = _result(_scan(rig))

    assert [f.name for f in result.failed] == [plan.fields[1].name]
    assert len(result.frames) == 11


def test_the_tolerance_is_a_parameter(rig: _Rig) -> None:
    rig.stage.landing_error_um = {2: (6.0, 0.0)}

    # 10 % of 96 µm is 9.6 µm: the 6.0 µm landing is a frame.
    result = _result(_scan(rig, landing_tolerance=0.1))

    assert len(result.frames) == 12
    assert result.failed == ()
    for bad in (0.0, 0.6, math.nan):
        with pytest.raises(ValueError, match="landing_tolerance"):
            ScanParams(landing_tolerance=bad)


def test_the_tolerance_bound_is_open_below_and_closed_above() -> None:
    """FM-43: a fraction of nothing fails every field, and half a side is the most one may take."""
    assert ScanParams(landing_tolerance=0.5).landing_tolerance == 0.5
    assert ScanParams(landing_tolerance=1e-9).landing_tolerance == 1e-9
    with pytest.raises(ValueError, match=r"less than or equal to 0\.5"):
        ScanParams(landing_tolerance=0.5000001)
    with pytest.raises(ValueError, match="greater than 0"):
        ScanParams(landing_tolerance=0.0)


def test_failures_off_target_not_in_a_row_do_not_fail_the_scan(rig: _Rig) -> None:
    """The next frame resets the count: two off-target fields apart are not two in a row."""
    rig.stage.landing_error_um = {2: (6.0, 0.0), 4: (6.0, 0.0)}

    result = _result(_scan(rig, max_failed_in_row=2))

    assert [f.name for f in result.failed] == ["A1_0001", "A1_0003"]
    assert len(result.frames) == 10


def test_a_readback_that_is_not_a_number_is_a_failed_field(rig: _Rig) -> None:
    """FM-104: ``nan`` is not within any tolerance, and is not a pydantic error out of ``ScanFrame``."""
    plan = _plan(rig)
    rig.stage.report = {2: XY(math.nan, math.nan)}

    result = _result(_scan(rig))

    second = plan.fields[1]
    assert len(result.failed) == 1
    assert result.failed[0].error == (
        "StageOffTarget: the readback is nan µm from the target, more than 4.8 µm "
        f"(target ({second.x_um:.1f}, {second.y_um:.1f}), readback (nan, nan))"
    )
    assert len(result.frames) == 11


def test_three_fields_off_target_in_a_row_fail_the_scan(rig: _Rig) -> None:
    """A stage that is stuck lands off target at every field: ``max_failed_in_row`` ends it."""
    plan = _plan(rig)
    rig.stage.landing_error_um = {n: (6.0, 0.0) for n in (1, 2, 3)}

    with pytest.raises(PluginRunError) as raised:
        _scan(rig)

    assert raised.value.reason == (
        f"3 fields failed in a row, the last A1_0002: "
        f"{_off_target_line(plan.fields[2], 6.0)}; the checkpoint keeps 0 wells: "
        "check the stage, then run scan --resume"
    )
    assert rig.stage.moves == 3
    assert rig.camera.snaps == 0


def test_a_dry_run_does_not_compare_the_readback(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    """The layer's dry-run readback is the target: nothing is measured, so nothing is compared."""
    rig = _build(demo_microscope_dry, tmp_path)
    plan = _plan(rig)
    rig.stage.report = {2: XY(1e6, 1e6)}

    result = _result(_scan(rig))

    assert result.failed == ()
    assert rig.stage.moves == len(plan.fields)
    assert result.assumed == ("dry run: 12 fields walked, no frame taken",)


def test_the_same_readback_on_a_real_stand_is_a_failed_field(rig: _Rig) -> None:
    rig.stage.report = {2: XY(1e6, 1e6)}

    result = _result(_scan(rig))

    assert [f.name for f in result.failed] == ["A1_0001"]
    assert result.failed[0].error.endswith("readback (1000000.0, 1000000.0))")


@pytest.mark.parametrize(
    "fov", [(0.0, 96.0), (128.0, -1.0)], ids=["no-width", "negative"]
)
def test_a_plan_with_no_field_of_view_is_refused_before_anything_moves(
    fov: tuple[float, float], rig: _Rig
) -> None:
    """An edited run document: a tolerance of nothing would fail every field, or none."""

    def edit(result: dict[str, Any]) -> None:
        result["fov_um"] = list(fov)

    _edit_plan_result(rig, edit)

    assert _refused(rig) == (
        f"the plan's field of view {fov} µm is not positive: run scan-plan again "
        "with the stand as it is now"
    )


# ------------------------------------------------------------------- resume


@pytest.fixture
def stopped(rig: _Rig) -> _Rig:
    """The rig after a scan stopped by the operator after the first field of A2."""
    stop = threading.Event()
    record = _scan(rig, stop=stop, on_progress=_at(5, stop.set))
    assert record.status is StageStatus.stopped
    assert _checkpoint(rig).wells_done == ("A1",)
    # The stop is honoured before the next move, not after it.
    assert rig.stage.moves == 5
    assert rig.camera.snaps == 5
    rig.stage.moves = 0
    rig.camera.snaps = 0
    return rig


def test_a_stopped_scan_resumes_with_the_same_token(stopped: _Rig) -> None:
    checkpoint = _checkpoint(stopped)
    token = checkpoint.frames[0].file.split("/")[1]

    result = _result(_scan(stopped, resume=True))

    assert result.folder == f"scan/{token}"
    assert len(result.frames) == 12
    names = [f.name for f in result.frames]
    assert len(set(names)) == 12
    assert result.frames[:4] == checkpoint.frames
    # A2 was not done, so it is scanned whole again, and then B2.
    assert stopped.stage.moves == 8
    assert len(_tiffs(stopped)) == 12


def test_a_resume_keeps_the_failed_fields_of_the_wells_done(rig: _Rig) -> None:
    rig.stage.fail = {2: HardwareError("no answer")}
    stop = threading.Event()
    _scan(rig, stop=stop, on_progress=_at(5, stop.set))
    assert [f.name for f in _checkpoint(rig).failed] == ["A1_0001"]
    rig.stage.fail = {}

    result = _result(_scan(rig, resume=True))

    assert [f.name for f in result.failed] == ["A1_0001"]
    assert result.failed[0].error == "HardwareError: no answer"
    assert len(result.frames) == 11


def test_a_resume_whose_checkpoint_names_a_well_the_plan_lacks_is_refused(
    stopped: _Rig,
) -> None:
    record = _stored(stopped)
    assert record.checkpoint is not None
    write_stage(
        stopped.run_doc,
        record.model_copy(
            update={"checkpoint": {**record.checkpoint, "wells_done": ["A1", "Z9"]}}
        ),
    )

    with pytest.raises(PluginRunError) as raised:
        _scan(stopped, resume=True)

    assert raised.value.reason == (
        "the checkpoint names well Z9, which the scan plan does not have: run scan "
        "without --resume"
    )
    assert stopped.stage.moves == 0


def test_a_run_folder_that_cannot_be_read_is_not_a_missing_frame(
    stopped: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-116: Windows raises a dropped share out of ``resolve``; ``--resume`` must stay the advice."""

    def dropped(run_dir: Path, relative: str) -> Path:
        cause = OSError(64, "The specified network name is no longer available")
        raise ValueError(f"{relative!r} cannot be resolved") from cause

    monkeypatch.setattr(scan_plugin, "resolve_in_run", dropped)

    with pytest.raises(PluginRunError) as raised:
        _scan(stopped, resume=True)

    assert raised.value.reason == (
        "the run folder cannot be read (The specified network name is no longer "
        "available): check the disk or the network share, then run scan --resume"
    )
    assert stopped.stage.moves == 0
    assert _checkpoint(stopped).wells_done == ("A1",)  # kept for that --resume


def test_a_resume_after_a_frame_held_open_rewrites_the_well(
    stopped: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The well a viewer blocked is scanned again whole once the viewer is closed."""
    token = _checkpoint(stopped).token
    with monkeypatch.context() as patch:
        _fail_write(
            patch,
            1,
            AtomicWriteError(
                PermissionError(errno.EACCES, "Permission denied"),
                kept=Path("held.smc-tmp"),
            ),
        )
        with pytest.raises(PluginRunError) as raised:
            _scan(stopped, resume=True)

    assert raised.value.reason == (
        f"the frame scan/{token}/A2/A2_0000.tif cannot be replaced: another "
        "program holds it open (a viewer, a backup or antivirus agent): close it, "
        "then run scan --resume"
    )
    assert _stored(stopped).status is StageStatus.failed
    assert _checkpoint(stopped).wells_done == ("A1",)  # kept for that --resume

    result = _result(_scan(stopped, resume=True))

    assert len(result.frames) == 12
    assert result.folder == f"scan/{token}"
    assert len(_tiffs(stopped)) == 12


def test_a_resume_after_the_plan_was_made_again_is_refused(stopped: _Rig) -> None:
    """FM-111: the checkpoint's wells and frames are of the earlier plan."""
    _plan(stopped)

    with pytest.raises(ResumeError) as raised:
        _scan(stopped, resume=True)

    assert raised.value.reason == (
        "scan read its inputs from an earlier run, so its checkpoint no longer "
        "matches them; run it without --resume"
    )
    assert stopped.stage.moves == 0


def test_a_resume_whose_frames_are_missing_is_refused(stopped: _Rig) -> None:
    lost = _checkpoint(stopped).frames[1]
    (stopped.run_dir / lost.file).unlink()

    with pytest.raises(PluginRunError) as raised:
        _scan(stopped, resume=True)

    assert raised.value.reason == (
        f"1 of the checkpoint's 4 frame files are missing from the run folder, "
        f"the first {lost.file}: run scan without --resume"
    )
    assert stopped.stage.moves == 0


def test_a_checkpoint_token_that_is_not_a_token_is_refused(
    stopped: _Rig, tmp_path: Path
) -> None:
    record = _stored(stopped)
    assert record.checkpoint is not None
    write_stage(
        stopped.run_doc,
        record.model_copy(
            update={"checkpoint": {**record.checkpoint, "token": "../../evil"}}
        ),
    )

    with pytest.raises(ResumeError) as raised:
        _scan(stopped, resume=True)

    assert raised.value.reason.startswith(
        "the saved checkpoint is not what scan expects (token: String should match "
        "pattern"
    )
    assert stopped.stage.moves == 0
    assert not (tmp_path.parent / "evil").exists()
    assert not (tmp_path / "evil").exists()


# ---------------------------------------------------------------- autofocus


def test_an_unlocked_autofocus_is_engaged_once_per_field(rig: _Rig) -> None:
    af = _Autofocus(engaged=True, locked=True)
    rig.microscope.override(Autofocus, af)
    rig.stage.after_move = lambda n: af.lose_lock() if n in (2, 3) else None

    result = _result(_scan(rig))

    assert af.calls.count("engage") == 2
    assert [f.focus for f in result.frames] == ["locked"] * 12
    assert {f.pfs_offset for f in result.frames} == {af.offset}
    assert result.assumed == (
        "the autofocus was engaged again on 2 fields and locked on 2 of them",
    )
    assert not {"disengage", "set_offset_native", "offset_by_native"} & set(af.calls)


def test_a_failed_engage_records_unlocked_frames(rig: _Rig) -> None:
    af = _Autofocus(engaged=True, locked=True)
    af.engage_results = [False, True]
    rig.microscope.override(Autofocus, af)
    rig.stage.after_move = lambda n: af.lose_lock() if n == 2 else None

    result = _result(_scan(rig))

    assert [f.focus for f in result.frames] == ["locked", "unlocked"] + ["locked"] * 10
    assert af.calls.count("engage") == 2  # field 2 failed to lock, field 3 locked
    # Item 9: two attempts, one lock: the line counts both.
    assert result.assumed == (
        "the autofocus was engaged again on 2 fields and locked on 1 of them",
    )


def test_an_engage_that_raises_is_the_scans_own_failure_and_is_retried(
    rig: _Rig,
) -> None:
    """FM-129: the protocol leaves PFS off when it gives up; the next field must not read that as the operator's switch-off."""
    af = _Autofocus(engaged=True, locked=True)
    af.engage_results = [HardwareError("engage gave up")]
    rig.microscope.override(Autofocus, af)
    rig.stage.after_move = lambda n: af.lose_lock() if n == 2 else None

    result = _result(_scan(rig, max_unlocked_in_row=3))

    assert [f.error for f in result.failed] == ["HardwareError: engage gave up"]
    assert len(result.frames) == 11
    assert [f.focus for f in result.frames] == ["locked"] * 11
    assert af.calls.count("engage") == 2  # field 2's raised, field 3's locked
    assert result.assumed == (
        "the autofocus was engaged again on 2 fields and locked on 1 of them",
    )


def test_a_lock_the_operator_took_ends_the_scans_own_failure(rig: _Rig) -> None:
    """The flag is the scan's own failure only: a lock someone else took clears it.

    The scan's engage fails at field 2; the operator locks PFS by hand before
    field 3 and switches it off before field 4: that switch-off is theirs, and
    the scan must not engage it again (it did, measured while planning #228).
    """
    af = _Autofocus(engaged=True, locked=True)
    af.engage_results = [False]
    rig.microscope.override(Autofocus, af)

    def operator(n: int) -> None:
        if n == 2:
            af.lose_lock()
        elif n == 3:
            af.engaged = af.locked = True
        elif n == 4:
            af.engaged = af.locked = False

    rig.stage.after_move = operator

    result = _result(_scan(rig, max_unlocked_in_row=100))

    assert af.calls.count("engage") == 1
    assert [f.focus for f in result.frames] == ["locked", "unlocked", "locked"] + [
        "unlocked"
    ] * 9
    assert result.assumed == (
        "the autofocus was engaged again on 1 fields and locked on 0 of them",
    )


def test_unlocked_fields_in_a_row_fail_the_scan(rig: _Rig) -> None:
    af = _Autofocus(engaged=True, locked=True)
    af.engage_results = [False] * 12
    rig.microscope.override(Autofocus, af)
    rig.stage.after_move = lambda n: af.lose_lock() if n >= 2 else None

    with pytest.raises(PluginRunError) as raised:
        _scan(rig, max_unlocked_in_row=2)

    assert raised.value.reason == (
        "the autofocus did not lock on 2 fields in a row, the last A1_0002; the "
        "checkpoint keeps 0 wells: lock it on the sample, then run scan --resume"
    )
    assert rig.camera.snaps == 2


def test_unlocked_fields_not_in_a_row_do_not_fail_the_scan(rig: _Rig) -> None:
    """The count resets on a lock: two lost locks far apart are not two in a row."""
    af = _Autofocus(engaged=True, locked=True)
    af.engage_results = [False, True, False, True]
    rig.microscope.override(Autofocus, af)
    rig.stage.after_move = lambda n: af.lose_lock() if n in (2, 4) else None

    result = _result(_scan(rig, max_unlocked_in_row=2))

    focus = [f.focus for f in result.frames]
    assert focus == ["locked", "unlocked", "locked", "unlocked"] + ["locked"] * 8


def test_an_autofocus_switched_off_during_the_scan_is_not_engaged_again(
    rig: _Rig,
) -> None:
    """The operator's switch wins: engaging would drive Z to the stored offset's plane (#205)."""
    af = _Autofocus(engaged=True, locked=True)
    rig.microscope.override(Autofocus, af)

    def switch_off(n: int) -> None:
        if n == 3:
            af.engaged = af.locked = False

    rig.stage.after_move = switch_off

    result = _result(_scan(rig, max_unlocked_in_row=20))

    assert af.calls.count("engage") == 0
    assert [f.focus for f in result.frames] == ["locked"] * 2 + ["unlocked"] * 10
    assert result.assumed == ()

    # With the default, the unlocked fields end the scan before the 5th of them.
    rig.camera.snaps = 0
    rig.stage.moves = 0
    af.engaged = af.locked = True
    with pytest.raises(PluginRunError, match="did not lock on 5 fields in a row"):
        _scan(rig)
    assert af.calls.count("engage") == 0
    assert rig.camera.snaps == 6


def test_an_autofocus_left_off_is_never_engaged(rig: _Rig) -> None:
    af = _Autofocus(engaged=False, locked=False)
    rig.microscope.override(Autofocus, af)

    result = _result(_scan(rig))

    assert not MUTATING & set(af.calls)
    assert [f.focus for f in result.frames] == ["off"] * 12
    assert {f.pfs_offset for f in result.frames} == {None}


def test_autofocus_off_reads_nothing_from_it(rig: _Rig) -> None:
    af = _Autofocus(engaged=True, locked=True)
    rig.microscope.override(Autofocus, af)

    result = _result(_scan(rig, autofocus="off"))

    assert af.calls == []
    assert [f.focus for f in result.frames] == ["off"] * 12


def test_a_stand_without_an_autofocus_records_none(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = rig.microscope.get
    monkeypatch.setattr(
        rig.microscope,
        "get",
        lambda wanted: None if wanted is Autofocus else real_get(wanted),
    )

    result = _result(_scan(rig))

    assert [f.focus for f in result.frames] == ["none"] * 12


# ----------------------------------------------------------------- dry run


def test_a_dry_run_scan_writes_nothing(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    rig = _build(demo_microscope_dry, tmp_path)
    plan = _plan(rig)
    af = _Autofocus(engaged=True, locked=True)
    demo_microscope_dry.override(Autofocus, af)
    rig.stage.after_move = lambda _n: af.lose_lock()
    before = rig.stage.position_um()

    record = _scan(rig)

    result = _result(record)
    assert record.dry_run is True
    assert rig.camera.snaps == 0
    assert "engage" not in af.calls
    assert rig.stage.moves == len(plan.fields)
    after = rig.stage.position_um()
    assert (after.x_um, after.y_um) == pytest.approx((before.x_um, before.y_um))
    assert result.frames == ()
    assert result.failed == ()
    assert result.assumed == ("dry run: 12 fields walked, no frame taken",)
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["run.json"]
