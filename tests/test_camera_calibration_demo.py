"""``camera-calibration`` on the demo devices: the stage moves, and what it leaves (m3 §4.2, #22).

The demo camera ignores the stage, so these tests put the synthetic plate's
camera on the stand (``demo_microscope_with_sample``): frames follow the stage
the way a real camera's do, and the pose of the camera (30 degrees, mirrored,
on a 384 x 512 frame, FM-73) is known. What the tests are about is the
decision the plugin takes at each move: FM-88 says a stop, a halt or a fault
leaves the stage where it is and a return runs only after the measurement
returned normally.

The demo stage rounds a command (a 128 µm move reads back 127.995), so a
position is asserted within 0.05 µm and never with equality (FM-85).
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import smc.plugins.camera_calibration.plugin as plugin_module
from smc.hardware import Microscope
from smc.hardware.capabilities import XY, Camera, Limits, ObjectiveTurret, XYStage
from smc.hardware.errors import HardwareError, MicroscopeHaltedError
from smc.plugins import (
    PluginRunError,
    Progress,
    RunContext,
    StopRequested,
    load,
    run_plugin,
)
from smc.plugins.camera_calibration.algorithm import (
    BARELY_CORRELATE_NOTE,
    MIN_RESPONSE,
    MIN_SIGNAL_SPREAD_COUNTS,
    PARALLEL_SHIFTS_NOTE,
    AwayShift,
    Fit,
    Shift,
    carries_signal,
    measure_shift,
    measure_shift_away_from_zero,
    precision_reasons,
    signal_spread,
)
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
    _second_look_holds,
    _unusable_reason,
    _wants_a_second_look,
    precision_line,
    sparse_field_line,
    suggested_step_um,
)
from smc.plugins.slots import CameraCalibration
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.fixtures import SampleMicroscopeFactory
from smc.testing.synthetic import PlateSample, SampleCamera
from smc.ui.controller import _CARD_TEXT_CHARS

pytestmark = pytest.mark.demo

SHAPE = (384, 512)  # (height, width): not square, on purpose (FM-73)
STEP_UM = 96.0  # 0.25 * 384 rows * 1.0 µm per pixel
LOGGER = "smc.plugins.camera-calibration"
#: What the operator reads when the shifts are not a camera's (the design's text).
RIGIDITY_NOTE = (
    "the shifts do not fit a rigid camera: did each axis move its full command?"
)


class _FlakyStage:
    """The demo's stage, except that a numbered move fails or lands off target.

    It implements the seven methods of ``XYStage``: ``Microscope.override``
    refuses an implementation that lacks one. ``moves`` counts the absolute
    moves it was asked for and ``targets`` keeps them, so a test can say that no
    return was sent and how far a move was asked to go. With
    ``unreadable_after_fault`` the position cannot be read once a move failed.
    """

    def __init__(
        self,
        real: XYStage,
        *,
        fail_on_move: int | None = None,
        drift_um: dict[int, tuple[float, float]] | None = None,
        unreadable_after_fault: bool = False,
        x_fraction: float = 1.0,
    ) -> None:
        self._real = real
        self._fail_on_move = fail_on_move
        self._drift_um = drift_um or {}
        self._unreadable_after_fault = unreadable_after_fault
        # An X axis that executes this fraction of every command, counted from
        # where the stage was when the wrapper was made (the start).
        self._x_fraction = x_fraction
        self._origin_x_um = real.position_um().x_um
        self._faulted = False
        self.moves = 0
        self.targets: list[tuple[float, float]] = []

    def position_um(self) -> XY:
        if self._faulted and self._unreadable_after_fault:
            raise HardwareError("the position cannot be read")
        return self._real.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        self.targets.append((x_um, y_um))
        if self.moves == self._fail_on_move:
            self._faulted = True
            raise HardwareError("the stage did not answer")
        dx, dy = self._drift_um.get(self.moves, (0.0, 0.0))
        executed_x_um = self._origin_x_um + self._x_fraction * (
            x_um - self._origin_x_um
        )
        return self._real.move_to_um(executed_x_um + dx, y_um + dy)

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        return self._real.move_by_um(dx_um, dy_um, force=force)

    def wait(self, timeout_s: float | None = None) -> None:
        self._real.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._real.is_busy()

    def stop(self) -> None:
        self._real.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._real.limits_um()


class _ReportingCamera(SampleCamera):
    """A sample camera that reports another pixel size than the one it renders at."""

    def __init__(self, *args: object, reported_um: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._reported_um = reported_um

    def pixel_size_um(self) -> float:
        return self._reported_um


class _ForgetfulCamera(SampleCamera):
    """A sample camera whose fourth frame, the one after the return, is unrelated noise."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.snaps = 0

    def snap(self) -> np.ndarray:
        self.snaps += 1
        frame = super().snap()
        if self.snaps == 4:
            return np.random.default_rng(7).integers(
                0, 4000, size=frame.shape, dtype=np.uint16
            )
        return frame


class _NoisyCamera(SampleCamera):
    """A sample camera whose numbered frames carry extra noise: those probes correlate less.

    Snap 1 is the reference, 2 the +X probe, 3 the +Y probe, 4 the return.
    """

    def __init__(
        self, *args: object, noisy_snaps: tuple[int, ...], **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._noisy_snaps = noisy_snaps
        self.snaps = 0

    def snap(self) -> np.ndarray:
        self.snaps += 1
        frame = super().snap()
        if self.snaps not in self._noisy_snaps:
            return frame
        noise = np.random.default_rng(self.snaps).normal(0.0, 60.0, size=frame.shape)
        return np.rint(np.clip(frame + noise, 0.0, 65535.0)).astype(np.uint16)


class _BadFirstFrameCamera(SampleCamera):
    """A sample camera whose reference frame is something the correlation cannot use."""

    def __init__(
        self,
        *args: object,
        spoil: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._spoil = spoil
        self._first = True

    def snap(self) -> np.ndarray:
        frame = super().snap()
        if self._first:
            self._first = False
            return self._spoil(frame)
        return frame


def _sample_camera(microscope: Microscope, **kwargs: object) -> SampleCamera:
    return SampleCamera(
        microscope,
        PlateSample(),
        pixel_size_um=1.0,
        shape=SHAPE,
        **kwargs,  # type: ignore[arg-type]
    )


def _run(microscope: Microscope, tmp_path: Path, **params: object) -> StageRecord:
    return run_plugin(
        CameraCalibrationPlugin(),
        microscope,
        CameraCalibrationParams(**params),  # type: ignore[arg-type]
        tmp_path / "run.json",
    )


def _result(record: StageRecord) -> CameraCalibration:
    assert record.status is StageStatus.done
    assert record.result is not None
    return CameraCalibration.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["camera-calibration"]


def _here(microscope: Microscope) -> tuple[float, float]:
    position = microscope.require(XYStage).position_um()
    return position.x_um, position.y_um


def _write_calibration(
    path: Path,
    *,
    profile: str = "demo",
    objective: str | None,
    shape: tuple[int, int],
    status: StageStatus = StageStatus.done,
    dry_run: bool = False,
) -> CameraCalibration:
    """A ``camera-calibration`` record as an earlier run of the stage would have left it."""
    result = CameraCalibration(
        objective=objective,
        image_shape_px=shape,
        matrix_um_per_px=((0.5, 0.0), (0.0, -0.5)),
        pixel_size_um=0.5,
        rotation_deg=0.0,
        mirrored=False,
        residual_um=0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=48.0,
        assumed=("an earlier assumption",),
    )
    then = datetime(2026, 10, 27, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin="camera-calibration",
            provides="camera-calibration",
            attempt="a1",
            status=status,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )
    return result


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_camera_calibration() -> None:
    entry = load("camera-calibration")

    assert entry.plugin.provides == "camera-calibration"
    assert entry.plugin.requires == (XYStage, Camera)
    assert entry.target == (
        "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin"
    )


# ------------------------------------------------------------- the measurement


def test_calibration_recovers_a_rotated_mirrored_camera(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(
        shape=SHAPE, camera_rotation_deg=30.0, mirrored=True
    )
    start = _here(microscope)
    progress: list[Progress] = []

    record = run_plugin(
        CameraCalibrationPlugin(),
        microscope,
        CameraCalibrationParams(),
        tmp_path / "run.json",
        on_progress=progress.append,
    )

    result = _result(record)
    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    assert result.rotation_deg == pytest.approx(30.0, abs=0.5)
    assert result.mirrored is True
    assert result.objective == microscope.require(ObjectiveTurret).current()
    assert result.image_shape_px == SHAPE
    assert result.step_um == STEP_UM
    assert result.return_error_um is not None
    assert result.return_error_um < 2.0
    assert result.reused_from is None
    assert result.assumed == ()
    # The sample's noise (std 20 on a texture of 40) caps a response near 0.9;
    # what matters is that it is far above the refusal threshold (0.05).
    assert result.response > 0.5
    # The stage is back where the operator left it.
    assert _here(microscope) == pytest.approx(start, abs=0.05)
    assert [(p.done, p.total, p.message) for p in progress] == [
        (1, 3, "+X probe"),
        (2, 3, "+Y probe"),
        (3, 3, "return"),
    ]
    assert _stored(tmp_path) == record


def test_the_stage_is_waited_for_before_its_start_is_read(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """FM-85: the demo stage reports a fraction of a move that is still under way.

    A raw core move returns at once and leaves the stage travelling; a start read
    then is somewhere on the way, and the return would put the stage there.
    """
    microscope = demo_microscope_with_sample(shape=SHAPE)
    microscope.core.setXYPosition(300.0, 200.0)  # not waited for, on purpose

    _result(_run(microscope, tmp_path))

    assert _here(microscope) == pytest.approx((300.0, 200.0), abs=0.05)


def test_a_given_step_is_used_as_it_is(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)

    result = _result(_run(microscope, tmp_path, step_um=60.0))

    assert result.step_um == 60.0
    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    assert result.rotation_deg == pytest.approx(0.0, abs=0.5)
    assert result.mirrored is False


def test_a_featureless_field_fails_fills_no_slot_and_returns_the_stage(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(PlateSample(texture_std=0.0), shape=SHAPE)
    start = _here(microscope)

    with pytest.raises(PluginRunError, match="barely correlate") as raised:
        _run(microscope, tmp_path)

    assert f"step {STEP_UM:g} µm" in raised.value.reason
    # The default step is a quarter of the field: it did not wrap, so it is not blamed.
    assert "may wrap" not in raised.value.reason
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None
    # The return runs before the fit is judged: the operator gets the stage back.
    assert _here(microscope) == pytest.approx(start, abs=0.05)


def test_a_step_over_forty_percent_of_the_field_is_blamed_when_the_fit_fails(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """200 rows of a 384-row frame wrap to -184 (FM-120): the fit fails, and says why."""
    microscope = demo_microscope_with_sample(shape=SHAPE)

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, step_um=200.0)

    assert "barely correlate" in raised.value.reason
    assert "a step over 40 % of the field may wrap: give a smaller step_um" in (
        raised.value.reason
    )
    # The probe frames' contrast belongs to the parallel-shifts note only (#545).
    assert "parallel shifts" not in raised.value.reason
    assert "probe frames span" not in raised.value.reason
    assert _stored(tmp_path).result is None


def test_a_stage_that_moves_a_fraction_of_x_fails_and_returns(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """FM-122: an X axis that executes 10 % of its command, the frames agreeing all the while.

    The fit used to end ``done`` with a matrix of 10 µm per pixel on one axis and
    1 on the other (a residual of 1664 µm); the rigidity note refuses it, and the
    return, which runs before the fit is judged, gives the stage back. The camera
    is the one of the module docstring (30 degrees, mirrored), so that both
    components of each probe are non-zero, and the start is off the origin, so
    that a return to the wrong place or an origin that the wrapper forgets shows.
    """
    microscope = demo_microscope_with_sample(
        shape=SHAPE, camera_rotation_deg=30.0, mirrored=True
    )
    start = _start_off_the_origin(microscope)
    microscope.override(
        XYStage, _FlakyStage(microscope.require(XYStage), x_fraction=0.1)
    )

    with pytest.raises(PluginRunError, match=re.escape(RIGIDITY_NOTE)):
        _run(microscope, tmp_path)

    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None
    assert _here(microscope) == pytest.approx(start, abs=0.05)


# ------------------------------------------------- FM-88: stop, halt and fault


def test_a_stop_leaves_the_stage_where_it_is(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _here(microscope)
    stop = threading.Event()

    def after_the_first_probe(progress: Progress) -> None:
        if progress.done == 1:
            stop.set()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = run_plugin(
            CameraCalibrationPlugin(),
            microscope,
            CameraCalibrationParams(),
            tmp_path / "run.json",
            on_progress=after_the_first_probe,
            stop=stop,
        )

    assert record.status is StageStatus.stopped
    assert record.result is None
    assert _here(microscope) == pytest.approx((start[0] + STEP_UM, start[1]), abs=0.05)
    left = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any("left at" in line for line in left)


def test_a_halt_leaves_the_stage_where_it_is(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _here(microscope)

    def halt_after_the_first_probe(progress: Progress) -> None:
        if progress.done == 1:
            microscope.stop()

    try:
        with (
            caplog.at_level(logging.WARNING, logger=LOGGER),
            pytest.raises(MicroscopeHaltedError),
        ):
            run_plugin(
                CameraCalibrationPlugin(),
                microscope,
                CameraCalibrationParams(),
                tmp_path / "run.json",
                on_progress=halt_after_the_first_probe,
            )
        assert _stored(tmp_path).status is StageStatus.failed
        assert any(
            "left at" in r.getMessage() for r in caplog.records if r.name == LOGGER
        )
    finally:
        microscope.resume()
    assert _here(microscope) == pytest.approx((start[0] + STEP_UM, start[1]), abs=0.05)


def test_a_fault_leaves_the_stage_where_it_is(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _here(microscope)
    stage = _FlakyStage(microscope.require(XYStage), fail_on_move=2)
    microscope.override(XYStage, stage)

    with pytest.raises(HardwareError, match="did not answer"):
        _run(microscope, tmp_path)

    # Two moves were asked for, the +X probe and the +Y probe that failed: no
    # return was sent after a fault.
    assert stage.moves == 2
    assert _here(microscope) == pytest.approx((start[0] + STEP_UM, start[1]), abs=0.05)
    assert _stored(tmp_path).status is StageStatus.failed


def test_a_fault_on_the_return_leaves_the_stage_at_the_last_probe(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _here(microscope)
    stage = _FlakyStage(microscope.require(XYStage), fail_on_move=3)
    microscope.override(XYStage, stage)

    with (
        caplog.at_level(logging.WARNING, logger=LOGGER),
        pytest.raises(HardwareError, match="did not answer"),
    ):
        _run(microscope, tmp_path)

    assert stage.moves == 3  # the return was tried once and not repeated
    assert _here(microscope) == pytest.approx((start[0], start[1] + STEP_UM), abs=0.05)
    assert any("left at" in r.getMessage() for r in caplog.records if r.name == LOGGER)


def test_a_stop_before_the_return_leaves_the_stage_at_the_last_probe(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _here(microscope)
    stop = threading.Event()

    def after_the_second_probe(progress: Progress) -> None:
        if progress.done == 2:
            stop.set()

    record = run_plugin(
        CameraCalibrationPlugin(),
        microscope,
        CameraCalibrationParams(),
        tmp_path / "run.json",
        on_progress=after_the_second_probe,
        stop=stop,
    )

    assert record.status is StageStatus.stopped
    assert _here(microscope) == pytest.approx((start[0], start[1] + STEP_UM), abs=0.05)


# ------------------------------------------------------------------- dry run


def test_a_dry_run_refuses_to_measure(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    demo_microscope_dry.override(Camera, _sample_camera(demo_microscope_dry))
    start = _here(demo_microscope_dry)

    with pytest.raises(PluginRunError, match="dry run cannot measure"):
        _run(demo_microscope_dry, tmp_path)

    assert _here(demo_microscope_dry) == start
    assert _stored(tmp_path).status is StageStatus.failed


# --------------------------------------------------------------------- reuse


@pytest.mark.parametrize("stand", ["demo_microscope", "demo_microscope_dry"])
def test_reuse_copies_a_done_calibration_and_says_so(
    stand: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    microscope: Microscope = request.getfixturevalue(stand)
    source = tmp_path / "earlier.json"
    objective = microscope.require(ObjectiveTurret).current()
    shape = microscope.require(Camera).image_shape()
    assert shape == (512, 512)  # the demo's own camera, read in a dry run as well
    earlier = _write_calibration(source, objective=objective, shape=shape)
    start = _here(microscope)

    result = _result(_run(microscope, tmp_path, from_run_doc=source))

    assert result.reused_from == source.as_posix()
    assert result.return_error_um is None
    assert result.matrix_um_per_px == earlier.matrix_um_per_px
    assert (result.pixel_size_um, result.step_um, result.response) == (0.5, 48.0, 0.97)
    assert result.assumed[:-1] == earlier.assumed
    assert result.assumed[-1] == (
        f"camera calibration reused from {source.as_posix()}, "
        "measured 2026-10-27T09:30:00+00:00"
    )
    assert _here(microscope) == start  # nothing moved


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"profile": "other"}, ["was run on other, this is demo"]),
        ({"objective": "Objective-2"}, ["Objective-2", "{objective}"]),
        ({"shape": (384, 512)}, ["(384, 512)", "(512, 512)"]),
    ],
    ids=["profile", "objective", "shape"],
)
def test_reuse_refuses_another_profile_objective_or_shape(
    changed: dict[str, object],
    expected: list[str],
    demo_microscope: Microscope,
    tmp_path: Path,
) -> None:
    objective = demo_microscope.require(ObjectiveTurret).current()
    source = tmp_path / "earlier.json"
    _write_calibration(
        source,
        **{
            "profile": "demo",
            "objective": objective,
            "shape": (512, 512),
            **changed,
        },  # type: ignore[arg-type]
    )

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path, from_run_doc=source)

    for text in expected:
        assert text.format(objective=objective) in raised.value.reason
    assert source.as_posix() in raised.value.reason
    assert _stored(tmp_path).result is None


def test_reuse_refuses_a_document_that_is_missing_or_whose_run_did_not_end_done(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    objective = demo_microscope.require(ObjectiveTurret).current()
    missing = tmp_path / "missing.json"
    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path, from_run_doc=missing)
    assert missing.as_posix() in raised.value.reason

    failed = tmp_path / "failed.json"
    _write_calibration(
        failed, objective=objective, shape=(512, 512), status=StageStatus.failed
    )
    with pytest.raises(PluginRunError, match="ended failed"):
        _run(demo_microscope, tmp_path, from_run_doc=failed)


# ---------------------------------------- what the result says it assumed


def test_a_misreported_pixel_size_is_logged(
    demo_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    demo_microscope.override(
        Camera, _reporting_camera(demo_microscope, reported_um=0.5)
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = _result(_run(demo_microscope, tmp_path))

    # The step follows what the camera says, 0.25 * 384 * 0.5, and the answer is
    # what the stage and the pixels measured.
    assert result.step_um == 48.0
    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    warned = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any("reports 0.5" in line and "measured 1" in line for line in warned)
    assert any("update the profile's pixel size" in line for line in warned)


def test_a_pixel_size_that_agrees_within_two_percent_is_not_blamed(
    demo_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    demo_microscope.override(
        Camera, _reporting_camera(demo_microscope, reported_um=1.01)
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _result(_run(demo_microscope, tmp_path))

    assert [r for r in caplog.records if r.name == LOGGER] == []


@pytest.mark.parametrize("reported_um", [0.0, -1.5, float("nan"), float("inf")])
def test_a_pixel_size_that_is_not_finite_and_positive_gives_the_fallback_step(
    reported_um: float,
    demo_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-104: infinity is above 0, and 0 means "not measured"; neither sizes a move."""
    demo_microscope.override(
        Camera, _reporting_camera(demo_microscope, reported_um=reported_um)
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = _result(_run(demo_microscope, tmp_path))

    assert result.step_um == 50.0
    assert "step of 50 µm: no pixel size was known" in result.assumed
    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    # Nothing is reported, so nothing is contradicted.
    assert [r for r in caplog.records if r.name == LOGGER] == []


def test_a_failed_fit_blames_a_fallback_step_for_a_wrap(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """With no pixel size known the 50 µm step is a guess, so it is named when the fit fails."""
    demo_microscope.override(
        Camera,
        _reporting_camera(
            demo_microscope, reported_um=0.0, sample=PlateSample(texture_std=0.0)
        ),
    )

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path)

    assert "step 50 µm" in raised.value.reason
    assert "may wrap: give a smaller step_um" in raised.value.reason


def test_a_stage_that_comes_back_off_is_reported_not_refused(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    # The third move, the return, lands 10 µm to the right of the start.
    stage = _FlakyStage(microscope.require(XYStage), drift_um={3: (10.0, 0.0)})
    microscope.override(XYStage, stage)

    result = _result(_run(microscope, tmp_path))

    assert result.return_error_um == pytest.approx(10.0, abs=0.5)
    assert any(
        line.startswith("the stage came back ") and "µm off" in line
        for line in result.assumed
    )
    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)


def test_a_return_that_does_not_correlate_leaves_the_error_unknown(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _ForgetfulCamera(
        demo_microscope, PlateSample(), pixel_size_um=1.0, shape=SHAPE
    )
    demo_microscope.override(Camera, camera)

    result = _result(_run(demo_microscope, tmp_path))

    assert camera.snaps == 4
    assert result.return_error_um is None  # unknown, not zero
    assert any(
        line.startswith(
            "the return could not be checked: the frame after it did not correlate ("
        )
        for line in result.assumed
    )
    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)


# ----------------------------------------- found by the adversarial review of #22


def _fmt(value: float) -> str:
    """A coordinate as the WARNING writes it: one decimal, and no ``-0.0``."""
    return f"{value + 0.0:.1f}"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


def _start_off_the_origin(microscope: Microscope) -> tuple[float, float]:
    """Park the stage at (300, 200): a start with two different coordinates, so a swap shows."""
    microscope.require(XYStage).move_to_um(300.0, 200.0)
    return _here(microscope)


def test_a_step_derived_from_the_reported_pixel_size_is_capped(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-121: a reported size 100 times too large was sent as 9.6 mm of absolute moves."""
    demo_microscope.override(
        Camera, _reporting_camera(demo_microscope, reported_um=100.0)
    )
    start = _here(demo_microscope)
    stage = _FlakyStage(demo_microscope.require(XYStage))
    demo_microscope.override(XYStage, stage)

    with pytest.raises(PluginRunError, match="step 2000 µm"):
        _run(demo_microscope, tmp_path)  # 2000 µm is far beyond the field: no fit

    # One target at a time: pytest 9.2 refuses `approx` on a list of tuples.
    expected = [(start[0] + 2000.0, start[1]), (start[0], start[1] + 2000.0)]
    assert len(stage.targets) >= 2
    for target, wanted in zip(stage.targets[:2], expected, strict=True):
        assert target == pytest.approx(wanted)


def test_a_capped_step_says_so_in_the_result(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A camera that renders and reports 25 µm per pixel: a quarter of the field is 2400 µm."""
    demo_microscope.override(
        Camera,
        SampleCamera(demo_microscope, PlateSample(), pixel_size_um=25.0, shape=SHAPE),
    )

    result = _result(_run(demo_microscope, tmp_path))

    assert result.step_um == 2000.0
    assert result.assumed[0].startswith("step capped at 2000 µm")
    assert result.pixel_size_um == pytest.approx(25.0, rel=0.02)


@pytest.mark.parametrize("noisy_snap", [2, 3], ids=["x-probe", "y-probe"])
def test_the_response_is_the_smaller_of_the_two_probes(
    demo_microscope: Microscope, tmp_path: Path, noisy_snap: int
) -> None:
    """Whichever probe is weaker sets the result's response: neither is dropped or averaged.

    A clean run reads about 0.8, and a result that kept one probe's response, or
    the larger, would read that whichever probe was noisy.
    """
    demo_microscope.override(
        Camera,
        _NoisyCamera(
            demo_microscope,
            PlateSample(),
            pixel_size_um=1.0,
            shape=SHAPE,
            noisy_snaps=(noisy_snap,),
        ),
    )

    result = _result(_run(demo_microscope, tmp_path))

    assert MIN_RESPONSE < result.response < 0.6


@pytest.mark.parametrize(
    "rotation_deg", [0.0, 90.0], ids=["the-y-probe-wraps", "the-x-probe-wraps"]
)
def test_a_wrap_in_either_probe_is_refused(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    rotation_deg: float,
) -> None:
    """200 µm is 200 px: more than half of the 384 rows, on the axis that the camera puts Y or X on."""
    microscope = demo_microscope_with_sample(
        shape=SHAPE, camera_rotation_deg=rotation_deg
    )

    with pytest.raises(PluginRunError, match="barely correlate"):
        _run(microscope, tmp_path, step_um=200.0)


def test_the_warning_after_a_fault_names_where_the_stage_is_and_where_it_started(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _start_off_the_origin(microscope)
    microscope.override(
        XYStage, _FlakyStage(microscope.require(XYStage), fail_on_move=2)
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HardwareError):
        _run(microscope, tmp_path)

    assert _warnings(caplog) == [
        f"the stage is left at ({_fmt(start[0] + STEP_UM)}, {_fmt(start[1])}) µm, "
        f"not returned to ({_fmt(start[0])}, {_fmt(start[1])}) µm"
    ]


def test_a_position_that_cannot_be_read_does_not_hide_the_fault(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _start_off_the_origin(microscope)
    stage = _FlakyStage(
        microscope.require(XYStage), fail_on_move=2, unreadable_after_fault=True
    )
    microscope.override(XYStage, stage)

    with (
        caplog.at_level(logging.WARNING, logger=LOGGER),
        pytest.raises(HardwareError, match="did not answer"),
    ):
        _run(microscope, tmp_path)

    assert _warnings(caplog) == [
        "the stage is left at an unknown position, not returned to "
        f"({_fmt(start[0])}, {_fmt(start[1])}) µm"
    ]


def test_a_stop_before_any_move_leaves_the_stage_and_logs_nothing(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing moved, so there is nothing left anywhere to warn about."""
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _start_off_the_origin(microscope)
    stop = threading.Event()
    stop.set()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = run_plugin(
            CameraCalibrationPlugin(),
            microscope,
            CameraCalibrationParams(),
            tmp_path / "run.json",
            stop=stop,
        )

    assert record.status is StageStatus.stopped
    assert _here(microscope) == pytest.approx(start, abs=0.05)
    assert _warnings(caplog) == []


def test_a_stop_in_the_settle_after_the_return_logs_nothing(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage is home when the stop arrives: the measurement is dropped, no WARNING.

    It also pins that ``settle_ms`` is waited after each of the three moves, in
    seconds: the context's own wait, recorded, as the ``stage-report`` test does.
    """
    microscope = demo_microscope_with_sample(shape=SHAPE)
    start = _start_off_the_origin(microscope)
    waits: list[float] = []

    def wait_then_stop(_ctx: RunContext, seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 3:  # the settle after the return move
            raise StopRequested

    monkeypatch.setattr(RunContext, "wait_s", wait_then_stop)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(microscope, tmp_path, settle_ms=50.0)

    assert record.status is StageStatus.stopped
    assert waits == [0.05, 0.05, 0.05]
    assert _here(microscope) == pytest.approx(start, abs=0.05)
    assert _warnings(caplog) == []


def _nan_in(frame: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float64)
    out[3, 5] = np.nan
    return out


@pytest.mark.parametrize(
    ("spoil", "message"),
    [
        (_nan_in, "not finite"),
        (lambda frame: np.stack([frame] * 3, axis=-1), "(384, 512, 3)"),
        (lambda frame: frame.T.copy(), "(512, 384)"),
    ],
    ids=["a-nan", "a-colour-frame", "a-transposed-frame"],
)
def test_a_reference_frame_that_cannot_be_judged_is_refused_before_any_move(
    demo_microscope: Microscope,
    tmp_path: Path,
    spoil: Callable[[np.ndarray], np.ndarray],
    message: str,
) -> None:
    """Known at the first frame, so the stage is not left a quarter of a field away."""
    demo_microscope.override(
        Camera,
        _BadFirstFrameCamera(
            demo_microscope,
            PlateSample(),
            pixel_size_um=1.0,
            shape=SHAPE,
            spoil=spoil,
        ),
    )
    stage = _FlakyStage(demo_microscope.require(XYStage))
    demo_microscope.override(XYStage, stage)

    with pytest.raises(PluginRunError, match=re.escape(message)):
        _run(demo_microscope, tmp_path)

    assert stage.moves == 0
    assert _stored(tmp_path).status is StageStatus.failed


# -- reuse: the shape is compared as (height, width), and as typed


def test_reuse_accepts_a_non_square_shape_of_the_camera(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    source = tmp_path / "same.json"
    _write_calibration(
        source, objective=microscope.require(ObjectiveTurret).current(), shape=SHAPE
    )

    result = _result(_run(microscope, tmp_path, from_run_doc=source))

    assert result.image_shape_px == SHAPE


@pytest.mark.parametrize(
    "other_shape",
    [(512, 384), (384, 640), (256, 512)],
    ids=["transposed", "same-height", "same-width"],
)
def test_reuse_refuses_a_shape_that_differs_in_either_side(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    other_shape: tuple[int, int],
) -> None:
    """(height, width): a comparison of one side, or of the sorted sides, accepts one of these."""
    microscope = demo_microscope_with_sample(shape=SHAPE)
    source = tmp_path / "other.json"
    _write_calibration(
        source,
        objective=microscope.require(ObjectiveTurret).current(),
        shape=other_shape,
    )

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, from_run_doc=source)

    assert str(other_shape) in raised.value.reason
    assert str(SHAPE) in raised.value.reason


def test_reuse_keeps_a_relative_path_relative_and_never_resolves_it(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-50: an absolute path names a user's folder, and the run document is attached to issues."""
    monkeypatch.chdir(tmp_path)
    objective = demo_microscope.require(ObjectiveTurret).current()
    shape = demo_microscope.require(Camera).image_shape()
    _write_calibration(Path("earlier.json"), objective=objective, shape=shape)

    result = _result(_run(demo_microscope, tmp_path, from_run_doc=Path("earlier.json")))

    assert result.reused_from == "earlier.json"
    assert result.assumed[-1].startswith("camera calibration reused from earlier.json,")


@pytest.mark.parametrize(
    ("stand", "recorded_in_a_dry_run", "accepted"),
    [
        ("demo_microscope", False, True),
        ("demo_microscope_dry", False, True),
        ("demo_microscope_dry", True, True),
        ("demo_microscope", True, False),
    ],
    ids=["real-reads-real", "dry-reads-real", "dry-reads-dry", "real-reads-dry"],
)
def test_a_calibration_from_a_dry_run_is_read_only_by_a_dry_run(
    stand: str,
    recorded_in_a_dry_run: bool,
    accepted: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """A dry run's result stands for a stage that never moved: a real run does not read it."""
    microscope: Microscope = request.getfixturevalue(stand)
    source = tmp_path / "earlier.json"
    _write_calibration(
        source,
        objective=microscope.require(ObjectiveTurret).current(),
        shape=microscope.require(Camera).image_shape(),
        dry_run=recorded_in_a_dry_run,
    )

    if accepted:
        assert _result(_run(microscope, tmp_path, from_run_doc=source)).reused_from
    else:
        with pytest.raises(PluginRunError, match="was a dry run"):
            _run(microscope, tmp_path, from_run_doc=source)


def test_without_a_turret_the_objective_is_unknown_and_is_not_compared(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)
    real_get = microscope.get
    monkeypatch.setattr(
        microscope,
        "get",
        lambda capability: (
            None if capability is ObjectiveTurret else real_get(capability)
        ),
    )
    source = tmp_path / "earlier.json"
    _write_calibration(source, objective="Objective-1", shape=SHAPE)

    reused = _result(_run(microscope, tmp_path, from_run_doc=source))
    measured = _result(_run(microscope, tmp_path))

    assert reused.objective == "Objective-1"  # copied as it was measured
    assert measured.objective is None  # no turret: unknown, not guessed


# -- what the result says about the return, and when the camera is contradicted


@pytest.mark.parametrize(
    ("pose", "drift_um", "expected_um"),
    [
        ({}, (0.0, 10.0), 10.0),
        ({"camera_rotation_deg": 30.0, "mirrored": True}, (10.0, 0.0), 10.0),
        ({"camera_rotation_deg": 30.0, "mirrored": True}, (0.0, 10.0), 10.0),
    ],
    ids=["y-drift", "x-drift-rotated-mirrored", "y-drift-rotated-mirrored"],
)
def test_the_return_error_is_the_stage_length_through_the_matrix(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    pose: dict[str, object],
    drift_um: tuple[float, float],
    expected_um: float,
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE, **pose)  # type: ignore[arg-type]
    microscope.override(
        XYStage,
        _FlakyStage(microscope.require(XYStage), drift_um={3: drift_um}),
    )

    result = _result(_run(microscope, tmp_path))

    assert result.return_error_um == pytest.approx(expected_um, abs=0.7)
    assert any("came back" in line for line in result.assumed)


@pytest.mark.parametrize(
    ("pixel_size_um", "drift_um", "reported"),
    [
        # Under four pixels at 1 µm per pixel, and over 2 µm: the pixels decide.
        (1.0, 3.0, False),
        # Over four pixels (0.4 µm each: 1.6 µm), under 2 µm: the floor decides.
        (0.4, 1.9, False),
        (0.4, 2.5, True),
    ],
    ids=["three-pixels", "under-the-floor", "over-the-floor"],
)
def test_the_return_error_limit_is_two_um_or_four_pixels_whichever_is_larger(
    demo_microscope: Microscope,
    tmp_path: Path,
    pixel_size_um: float,
    drift_um: float,
    reported: bool,
) -> None:
    demo_microscope.override(
        Camera,
        SampleCamera(
            demo_microscope, PlateSample(), pixel_size_um=pixel_size_um, shape=SHAPE
        ),
    )
    demo_microscope.override(
        XYStage,
        _FlakyStage(demo_microscope.require(XYStage), drift_um={3: (drift_um, 0.0)}),
    )

    result = _result(_run(demo_microscope, tmp_path))

    assert result.return_error_um == pytest.approx(drift_um, abs=0.3)
    assert any("came back" in line for line in result.assumed) is reported


def test_a_pixel_size_that_is_five_percent_off_is_blamed(
    demo_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    demo_microscope.override(
        Camera, _reporting_camera(demo_microscope, reported_um=1.05)
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _result(_run(demo_microscope, tmp_path))

    assert any("reports 1.05" in line for line in _warnings(caplog))


def _reporting_camera(
    microscope: Microscope, *, reported_um: float, sample: PlateSample | None = None
) -> SampleCamera:
    return _ReportingCamera(
        microscope,
        sample if sample is not None else PlateSample(),
        pixel_size_um=1.0,
        shape=SHAPE,
        reported_um=reported_um,
    )


# -- a frame with no light on the camera (#408, F8) and a probe of a few pixels (F6)


def _dark(shape: tuple[int, ...]) -> np.ndarray:
    """The Prime 95B's dark frame (session 2): mean 100.5, std 2.3, a column pattern."""
    rng = np.random.default_rng(408)
    frame = 100.5 + rng.normal(0.0, 2.3, shape) + rng.normal(0.0, 1.0, shape[1])
    return np.rint(frame).astype(np.uint16)


def _step_frame(shape: tuple[int, ...], spread: float) -> np.ndarray:
    """Half the frame at 0 and half at ``spread``: its 1st to 99th percentile spans exactly that."""
    frame = np.zeros(shape, dtype=np.uint16)
    frame[:, shape[1] // 2 :] = int(spread)
    return frame


class _DarkSnapCamera(SampleCamera):
    """A sample camera whose numbered frames carry no light, as with the light path on the eyepiece.

    Snap 1 is the reference, 2 the +X probe, 3 the +Y probe, 4 the return. The
    numbered frames are ``frame_of(shape)`` (the Prime's dark frame by default).
    """

    def __init__(
        self,
        *args: object,
        dark_snaps: tuple[int, ...],
        frame_of: Callable[[tuple[int, ...]], np.ndarray] = _dark,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._dark_snaps = dark_snaps
        self._frame_of = frame_of
        self.snaps = 0

    def snap(self) -> np.ndarray:
        self.snaps += 1
        frame = super().snap()
        return self._frame_of(frame.shape) if self.snaps in self._dark_snaps else frame


class _ShallowCamera(_DarkSnapCamera):
    """A camera in a 12-bit mode: the same light gives counts 16 times smaller."""

    def bit_depth(self) -> int:
        return 12

    def snap(self) -> np.ndarray:
        return super().snap() // 16


class _BeadCamera(SampleCamera):
    """A dark field with a few bright beads fixed to the plate: lit, on under 1 % of the pixels.

    ``beads_px`` are the beads' pixels at the stage's first position; the content
    moves against the stage (an upright camera at 1 µm per pixel: +X moves it
    left, +Y down). Each bead is 2.5 px wide at 3000 counts over the measured
    dark background, about 160 pixels or 0.08 % of the frame.
    """

    def __init__(
        self,
        *args: object,
        beads_px: tuple[tuple[int, int], ...],
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._beads_px = beads_px
        self._noise = np.random.default_rng(7)
        self._origin: XY | None = None

    def snap(self) -> np.ndarray:
        microscope = self._microscope()
        assert microscope is not None
        here = microscope.require(XYStage).position_um()
        if self._origin is None:
            self._origin = here
        rows, cols = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
        frame = 100.0 + self._noise.normal(0.0, 2.3, SHAPE)
        for row, col in self._beads_px:
            d_col = col - (here.x_um - self._origin.x_um)
            d_row = row + (here.y_um - self._origin.y_um)
            squared = (rows - d_row) ** 2 + (cols - d_col) ** 2
            frame += 3000.0 * np.exp(-squared / (2.0 * 2.5**2))
        return np.rint(frame).astype(np.uint16)


class _StillBackgroundCamera(_BeadCamera):
    """Beads that follow the stage, over a background that does not: dust and shading on the sensor.

    At the 40x in water (Ti2-E session 5, F12, #545) the still background won the
    correlation on a field of two beads: the stage moved, the image said it had not.
    The background here is a gradient of 600 counts and a fixed pattern of 30; the
    beads of ``_BeadCamera`` are 3000 over it. ``taken`` keeps every frame snapped.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        rows, cols = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
        self._still = (
            400.0 * cols / SHAPE[1]
            + 200.0 * rows / SHAPE[0]
            + np.random.default_rng(12).normal(0.0, 30.0, SHAPE)
        )
        self.taken: list[np.ndarray] = []

    def snap(self) -> np.ndarray:
        frame = np.rint(super().snap() + self._still).astype(np.uint16)
        self.taken.append(frame)
        return frame


@pytest.mark.parametrize(
    ("spoil", "head", "advice"),
    [
        (
            lambda frame: _dark(frame.shape),
            "the camera's frame is dark or flat",
            "is the light on and the light path on the camera?",
        ),
        (
            # A frame at full scale: the opposite remedy, so the opposite message.
            lambda frame: np.full_like(frame, 65535),
            "the camera's frame is saturated",
            "lower the exposure or the light",
        ),
        (
            lambda frame: np.zeros_like(frame),
            "the camera's frame is dark or flat",
            "is the light on and the light path on the camera?",
        ),
    ],
    ids=["dark", "saturated", "zero"],
)
def test_a_frame_with_no_signal_is_refused_before_any_move_and_the_light_path_is_named(
    demo_microscope: Microscope,
    tmp_path: Path,
    spoil: Callable[[np.ndarray], np.ndarray],
    head: str,
    advice: str,
) -> None:
    """F8: the light path was on the eyepiece, and the message pointed at the stage."""
    demo_microscope.override(
        Camera,
        _BadFirstFrameCamera(
            demo_microscope,
            PlateSample(),
            pixel_size_um=1.0,
            shape=SHAPE,
            spoil=spoil,
        ),
    )
    stage = _FlakyStage(demo_microscope.require(XYStage))
    demo_microscope.override(XYStage, stage)

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path)

    reason = raised.value.reason
    assert head in reason
    assert advice in reason
    assert reason.endswith("nothing was moved")
    assert "parallel shifts" not in reason
    # What was measured and the floor are in the message, so that a surprise at a
    # stand is a number to bring back.
    measured = round(signal_spread(spoil(np.zeros(SHAPE, dtype=np.uint16))))
    assert f"spans {measured} counts, under 50)" in reason
    assert stage.moves == 0
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None


@pytest.mark.parametrize(
    ("dark_snaps", "label"),
    [((2,), "+X probe"), ((3,), "+Y probe"), ((2, 3), "+X probe")],
    ids=["x-probe", "y-probe", "both-name-the-first"],
)
def test_a_probe_frame_with_no_signal_is_refused_after_the_return_not_blamed_on_the_stage(
    demo_microscope: Microscope,
    tmp_path: Path,
    dark_snaps: tuple[int, ...],
    label: str,
) -> None:
    """A dark probe gave a shift of zero, and "parallel shifts: did the stage move?" blamed the stage.

    The stage did move, so the return runs first (FM-88) and the refusal says it
    was sent back (a move that returns is not a move that arrived, FM-167: where
    it landed is not read); the frame after the return is not taken.
    """
    camera = _DarkSnapCamera(
        demo_microscope,
        PlateSample(),
        pixel_size_um=1.0,
        shape=SHAPE,
        dark_snaps=dark_snaps,
    )
    demo_microscope.override(Camera, camera)
    stage = _FlakyStage(demo_microscope.require(XYStage))
    demo_microscope.override(XYStage, stage)
    start = _here(demo_microscope)

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path)

    reason = raised.value.reason
    assert f"the frame after the {label} is dark or flat" in reason
    assert "did the light go off" in reason
    assert f"step of {STEP_UM:g} µm" in reason
    assert reason.endswith("the stage was sent back to its start")
    assert f"spans {round(signal_spread(_dark(SHAPE)))} counts, under 50)" in reason
    assert "parallel shifts" not in reason
    assert "barely correlate" not in reason
    # Two probes and the return; the frame after the return was not taken.
    assert stage.moves == 3
    assert camera.snaps == 3
    assert _here(demo_microscope) == pytest.approx(start, abs=0.05)
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None


def test_a_lit_featureless_field_is_not_taken_for_a_dark_one(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """Noise and no texture spans 94 counts against a floor of 50: it is lit, and it moves."""
    microscope = demo_microscope_with_sample(PlateSample(texture_std=0.0), shape=SHAPE)
    assert signal_spread(microscope.require(Camera).snap()) > MIN_SIGNAL_SPREAD_COUNTS
    stage = _FlakyStage(microscope.require(XYStage))
    microscope.override(XYStage, stage)

    with pytest.raises(PluginRunError, match="barely correlate"):
        _run(microscope, tmp_path)

    assert stage.moves == 3


@pytest.mark.parametrize(
    ("spread", "refused"), [(49.0, True), (50.0, False)], ids=["49", "exactly-50"]
)
def test_a_reference_frame_of_exactly_fifty_counts_has_light_and_49_has_none(
    demo_microscope: Microscope, tmp_path: Path, spread: float, refused: bool
) -> None:
    """The boundary through the plugin: the reference check asks ``carries_signal``, ``>=``."""
    demo_microscope.override(
        Camera,
        _BadFirstFrameCamera(
            demo_microscope,
            PlateSample(),
            pixel_size_um=1.0,
            shape=SHAPE,
            spoil=lambda frame: _step_frame(frame.shape, spread),
        ),
    )
    stage = _FlakyStage(demo_microscope.require(XYStage))
    demo_microscope.override(XYStage, stage)

    reason = _refusal(demo_microscope, tmp_path)

    assert ("dark or flat" in reason) is refused
    # Past the check the probes are of another scene, and the fit says so.
    assert stage.moves == (0 if refused else 3)


@pytest.mark.parametrize(
    ("spread", "refused"), [(49.0, True), (50.0, False)], ids=["49", "exactly-50"]
)
def test_a_probe_frame_of_exactly_fifty_counts_has_light_and_49_has_none(
    demo_microscope: Microscope, tmp_path: Path, spread: float, refused: bool
) -> None:
    demo_microscope.override(
        Camera,
        _DarkSnapCamera(
            demo_microscope,
            PlateSample(),
            pixel_size_um=1.0,
            shape=SHAPE,
            dark_snaps=(2,),
            frame_of=lambda shape: _step_frame(shape, spread),
        ),
    )
    stage = _FlakyStage(demo_microscope.require(XYStage))
    demo_microscope.override(XYStage, stage)

    reason = _refusal(demo_microscope, tmp_path)

    assert ("the frame after the +X probe is dark or flat" in reason) is refused
    assert stage.moves == 3


def _refusal(microscope: Microscope, tmp_path: Path) -> str:
    """The reason the run was refused for, or an empty text when it was not."""
    try:
        _run(microscope, tmp_path)
    except PluginRunError as refused:
        return refused.reason
    return ""


def test_a_field_of_a_few_beads_on_a_dark_background_is_calibrated(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Three beads are lit on 0.24 % of the pixels, which the percentiles read as dark.

    The base code calibrates it, and a false refusal has no override: the second
    rule of ``carries_signal`` (pixels far from the median) is what lets it pass.
    """
    camera = _BeadCamera(
        demo_microscope,
        PlateSample(),
        pixel_size_um=1.0,
        shape=SHAPE,
        beads_px=((60, 150), (120, 300), (200, 200)),
    )
    demo_microscope.override(Camera, camera)
    assert signal_spread(camera.snap()) < MIN_SIGNAL_SPREAD_COUNTS

    result = _result(_run(demo_microscope, tmp_path))

    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    assert result.assumed == ()


# -- a sparse field over the camera's own pattern: the second look (#625, F14, m3 §4.4)

#: D6's last sentence, character for character: the conditional way out of a refusal for texture.
REUSE_SENTENCE = (
    "; if this camera was calibrated at this objective before, reuse that "
    "calibration with from_run_doc=PATH_TO_ITS_RUN_DOCUMENT (nothing moves)"
)
#: The line of a calibration the second look read: its return check is blind, so it says so.
BLIND_RETURN = (
    "the return could not be checked: on this field the camera's own pattern "
    "wins the correlation at zero shift"
)
#: The page's card keeps this many characters of a failure's words (``_CARD_TEXT_CHARS``).
CARD_TEXT_CHARS = _CARD_TEXT_CHARS


class _DeafCamera(_StillBackgroundCamera):
    """A camera whose beads do not follow the stage either: the whole frame is fixed to the sensor.

    ``_BeadCamera`` places its beads from the stage's first position. Resetting
    that origin at every snap puts them where they were at the first frame
    whatever the stage did: the stage moved a quarter field and the frames
    changed by their noise only. A stage that did not move, over the camera's own
    pattern: the second look must find no shift in that.
    """

    def snap(self) -> np.ndarray:
        self._origin = None
        return super().snap()


def _over_a_still_background(
    microscope: Microscope,
    beads_px: tuple[tuple[int, int], ...],
    camera_class: type[_StillBackgroundCamera] = _StillBackgroundCamera,
    drift_um: dict[int, tuple[float, float]] | None = None,
) -> tuple[_StillBackgroundCamera, _FlakyStage, tuple[float, float]]:
    """The stand with a camera that has a still background, a stage that counts, and where it starts.

    The first frame is taken and forgotten: it must be a lit one (it reaches the
    fit, which is where the note is), and ``taken`` then holds the run's own
    frames from its reference on.
    """
    camera = camera_class(
        microscope,
        PlateSample(),
        pixel_size_um=1.0,
        shape=SHAPE,
        beads_px=beads_px,
    )
    microscope.override(Camera, camera)
    stage = _FlakyStage(microscope.require(XYStage), drift_um=drift_um)
    microscope.override(XYStage, stage)
    start = _here(microscope)
    assert carries_signal(camera.snap())
    camera.taken.clear()
    return camera, stage, start


def _first_look_reads_zero(reference: np.ndarray, probe: np.ndarray) -> None:
    """The premise of every test of the second look: the first one read a stage that did not move.

    Exactly ``(0, 0)``, as F12's test asserted, on lit frames that agree (a
    response over the refusal's floor, not a blank).
    """
    first = measure_shift(reference, probe)
    assert (first.d_col_px, first.d_row_px) == (0.0, 0.0)
    assert first.response >= MIN_RESPONSE


def test_two_beads_over_a_still_background_are_calibrated_by_the_second_look_and_it_says_so(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """F12's field (#545) is calibrated now: the stage moved, the background won, and the beads say so.

    The first look reads both probes as zero (dust and shading fixed to the
    sensor out-correlate two beads); the second look, away from zero shift, reads
    the beads' own movement, at ratios that stand out (4.7 and 4.9 times the
    next, measured), and the fit accepts it. The result is a calibration like any
    other and says in ``assumed`` how it was read, because its ``response``
    counts the camera's pattern too.
    """
    camera, stage, start = _over_a_still_background(
        demo_microscope, ((120, 180), (250, 330))
    )

    result = _result(_run(demo_microscope, tmp_path))

    reference, probe_x, probe_y = camera.taken[:3]
    # The premise: the stage moved (two probes and the return), the frames did not.
    assert stage.moves == 3
    assert stage.targets[0] == (start[0] + STEP_UM, start[1])
    assert stage.targets[1] == (start[0], start[1] + STEP_UM)
    _first_look_reads_zero(reference, probe_x)
    _first_look_reads_zero(reference, probe_y)
    away_x = measure_shift_away_from_zero(reference, probe_x)
    away_y = measure_shift_away_from_zero(reference, probe_y)
    # Well over the 1.5 asked (4.7 to 4.9 measured): the test is about a field
    # the second look reads, not one that sits at its threshold.
    assert away_x.peak_ratio > 3.0
    assert away_y.peak_ratio > 3.0

    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    assert result.rotation_deg == pytest.approx(0.0, abs=1.0)
    assert result.mirrored is False
    assert result.step_um == STEP_UM
    assert result.reused_from is None
    # The response is the smaller of the two second-look responses.
    assert result.response == pytest.approx(
        min(away_x.shift.response, away_y.shift.response)
    )
    line = (
        "sparse field: the camera's own pattern won at zero shift, so both shifts "
        f"are the strongest peaks away from it ({away_x.peak_ratio:.1f} and "
        f"{away_y.peak_ratio:.1f} times the next, 1.5 asked)"
    )
    # The sparse-field line, then the return's: its check is blind on this field.
    assert result.assumed == (line, BLIND_RETURN)
    assert len(line) <= 160  # the strip keeps 160 characters of the first line (FM-205)
    assert len(BLIND_RETURN) <= 160
    # The return runs before the fit is judged, and the second look asks the
    # camera for nothing: the three moves, then the return's frame, no more (FM-88).
    assert stage.moves == 3
    assert len(camera.taken) == 4
    assert result.return_error_um is None  # unknown, not the 0 the blind check reads
    assert _here(demo_microscope) == pytest.approx(start, abs=0.05)
    assert _stored(tmp_path).status is StageStatus.done


def test_a_return_that_landed_far_off_is_not_reported_as_zero_on_a_field_the_second_look_read(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The camera's own pattern wins at zero shift on this field, so the return check cannot see a drift.

    The return move lands 70 µm off in X. On a rich field that is reported (``came back
    <n> µm off``); here the check would read zero and store it as a measurement, which
    it is not: the result says it could not be checked and leaves the error unknown.
    """
    camera, stage, start = _over_a_still_background(
        demo_microscope, ((120, 180), (250, 330)), drift_um={3: (-70.0, 0.0)}
    )

    result = _result(_run(demo_microscope, tmp_path))

    reference, _probe_x, _probe_y = camera.taken[:3]
    # The premise: the stage did not come back, and the frame after the return is the
    # reference's to the first look (the pattern wins), which is what would read zero.
    assert _here(demo_microscope)[0] == pytest.approx(start[0] - 70.0, abs=0.05)
    assert stage.moves == 3
    assert measure_shift(reference, camera.taken[3]).response >= MIN_RESPONSE
    assert result.return_error_um is None
    assert result.assumed[-1] == BLIND_RETURN
    assert not any("came back" in line for line in result.assumed)


def test_a_second_look_the_fit_refuses_is_a_refusal_with_the_fits_notes_and_the_ratios(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D4: a believed second look is judged by ``fit`` like any pair of shifts (FM-120, FM-122).

    Both peaks stand out (4.0 times the next) and are rigid (96 px long, at a right
    angle), but their response is 0.01: ``fit`` refuses it as a weak response. The
    run must end ``failed`` with that note and what the second look saw, never ``done``
    on the strength of the ratio and the shape alone, and the stage must be back.
    """
    camera, stage, start = _over_a_still_background(
        demo_microscope, ((120, 180), (250, 330))
    )
    looks = iter(
        [
            AwayShift(Shift(-96.0, 0.0, 0.01), 4.0),
            AwayShift(Shift(0.0, 96.0, 0.01), 4.0),
        ]
    )
    monkeypatch.setattr(
        plugin_module,
        "measure_shift_away_from_zero",
        lambda reference, probe: next(looks),
    )

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path)

    assert raised.value.reason == (
        f"{BARELY_CORRELATE_NOTE}; away from the camera's own pattern the best peaks "
        f"are 4.0 and 4.0 times the next (1.5 asked); step {STEP_UM:g} µm"
        + REUSE_SENTENCE
    )
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None
    # Two probes and the return; the frame after the return is not taken.
    assert stage.moves == 3
    assert len(camera.taken) == 3
    assert _here(demo_microscope) == pytest.approx(start, abs=0.05)


class _NoPixelSizeCamera(_StillBackgroundCamera):
    """The still-background camera of a stand whose profile has no pixel size: the probe is the 50 µm fallback."""

    def pixel_size_um(self) -> float:
        return 0.0


def test_the_sparse_field_line_follows_the_step_line_and_precedes_the_return_line(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """D5's order with a step line in front: the step's, the sparse field's, the return's.

    The strip keeps the first ``assumed`` line only (FM-205), so what comes first is
    what the operator reads: the step is a probe distance and not a result, but it
    stays where the other tools put it.
    """
    camera, _stage, _start = _over_a_still_background(
        demo_microscope, ((120, 180), (250, 330)), camera_class=_NoPixelSizeCamera
    )

    result = _result(_run(demo_microscope, tmp_path))

    reference, probe_x, probe_y = camera.taken[:3]
    _first_look_reads_zero(reference, probe_x)
    _first_look_reads_zero(reference, probe_y)
    assert result.step_um == 50.0
    assert len(result.assumed) == 3
    assert result.assumed[0] == "step of 50 µm: no pixel size was known"
    assert result.assumed[1].startswith("sparse field: the camera's own pattern won")
    assert result.assumed[2] == BLIND_RETURN


def test_the_sparse_field_line_comes_before_the_precision_line(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """D5's order: the step lines, the sparse-field line, the precision line, the return's lines.

    A probe of 30 px on a 384-row frame is under a tenth of it and fits poorly,
    and it is further than the exclusion from zero, so the second look reads it.
    """
    camera, _stage, _start = _over_a_still_background(
        demo_microscope, ((120, 180), (250, 330))
    )

    result = _result(_run(demo_microscope, tmp_path, step_um=30.0))

    reference, probe_x, probe_y = camera.taken[:3]
    _first_look_reads_zero(reference, probe_x)
    _first_look_reads_zero(reference, probe_y)
    assert len(result.assumed) == 3
    assert result.assumed[0].startswith("sparse field: the camera's own pattern won")
    assert result.assumed[1].startswith(
        "step of 30 µm fits poorly (30 px, under a tenth"
    )
    assert result.assumed[2] == BLIND_RETURN


def test_the_sparse_field_line_is_one_short_line_at_the_largest_ratios() -> None:
    """99.9 times the next is the widest the line is built for: it stays at 160 characters or less."""
    line = sparse_field_line((99.9, 99.9))

    assert line == (
        "sparse field: the camera's own pattern won at zero shift, so both shifts "
        "are the strongest peaks away from it (99.9 and 99.9 times the next, 1.5 asked)"
    )
    assert len(line) <= 160


def test_a_still_background_with_no_bead_is_refused_with_what_the_second_look_saw(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """F12 (#545) with no bead to read: the second look has nothing, and the refusal says so in numbers.

    The stage moved a quarter field in X and in Y and came back; the frames shifted
    by zero, because dust and shading fixed to the sensor are all there is. The
    message names the frames first, then the light path, then the stage, gives the
    contrast of the probe frames as a number to bring back, then what the second
    look saw (1.01 and 1.06 times the next, measured: nothing stands out), and the
    way out that needs no texture.
    """
    camera, stage, start = _over_a_still_background(demo_microscope, ())

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path)

    reference, probe_x, probe_y = camera.taken[:3]
    assert stage.moves == 3
    assert stage.targets[0] == (start[0] + STEP_UM, start[1])
    assert stage.targets[1] == (start[0], start[1] + STEP_UM)
    _first_look_reads_zero(reference, probe_x)
    _first_look_reads_zero(reference, probe_y)
    ratio_x = measure_shift_away_from_zero(reference, probe_x).peak_ratio
    ratio_y = measure_shift_away_from_zero(reference, probe_y).peak_ratio
    assert ratio_x < 1.5
    assert ratio_y < 1.5
    reason = raised.value.reason
    assert reason.startswith(PARALLEL_SHIFTS_NOTE)
    assert "did the stage move" not in reason
    # The frames, then the light path, then the stage.
    assert (
        reason.index("too little texture")
        < reason.index("light path")
        < reason.index("then the stage")
    )
    assert reason == (
        f"{PARALLEL_SHIFTS_NOTE}; the probe frames span "
        f"{round(signal_spread(probe_x))} and {round(signal_spread(probe_y))} counts "
        "(1st to 99th percentile); away from the camera's own pattern the best "
        f"peaks are {ratio_x:.1f} and {ratio_y:.1f} times the next (1.5 asked); "
        f"step {STEP_UM:g} µm" + REUSE_SENTENCE
    )
    # The reference, the two probes; the frame after the return is not taken.
    assert len(camera.taken) == 3
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None
    assert _here(demo_microscope) == pytest.approx(start, abs=0.05)


def test_a_stage_that_did_not_move_over_the_cameras_pattern_is_still_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Beads that stay where they were: the second look must not make a shift of the noise.

    The camera is fixed to the sensor, beads included, so a stage that moved a
    quarter field gives the frames of one that did not. Both probes read zero,
    nothing stands out away from zero (both ratios under 1.5), and the refusal is
    D5's, with the stage back where it started and nothing filled.
    """
    camera, stage, start = _over_a_still_background(
        demo_microscope, ((120, 180), (250, 330)), camera_class=_DeafCamera
    )

    with pytest.raises(PluginRunError) as raised:
        _run(demo_microscope, tmp_path)

    reference, probe_x, probe_y = camera.taken[:3]
    assert stage.moves == 3
    _first_look_reads_zero(reference, probe_x)
    _first_look_reads_zero(reference, probe_y)
    ratio_x = measure_shift_away_from_zero(reference, probe_x).peak_ratio
    ratio_y = measure_shift_away_from_zero(reference, probe_y).peak_ratio
    assert ratio_x < 1.5
    assert ratio_y < 1.5
    reason = raised.value.reason
    assert reason.startswith(PARALLEL_SHIFTS_NOTE)
    assert (
        "; away from the camera's own pattern the best peaks are "
        f"{ratio_x:.1f} and {ratio_y:.1f} times the next (1.5 asked); "
    ) in reason
    assert reason.endswith(REUSE_SENTENCE)
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None
    assert _here(demo_microscope) == pytest.approx(start, abs=0.05)


def test_a_refusal_for_parallel_shifts_ends_with_the_reuse_sentence_after_what_the_second_look_saw(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """D6: a refusal for texture offers the earlier calibration, after everything else it says."""
    _over_a_still_background(demo_microscope, ())

    reason = _refusal(demo_microscope, tmp_path)

    assert PARALLEL_SHIFTS_NOTE in reason
    assert reason.endswith(REUSE_SENTENCE)
    assert reason.count("from_run_doc") == 1
    assert reason.index("away from the camera's own pattern") < reason.index("step ")
    assert reason.index("step ") < reason.index("if this camera was calibrated")


def test_a_refusal_for_a_weak_response_ends_with_the_reuse_sentence_and_names_no_second_look(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """A lit field with no texture: the other texture note, the same way out, no second look."""
    microscope = demo_microscope_with_sample(PlateSample(texture_std=0.0), shape=SHAPE)

    reason = _refusal(microscope, tmp_path)

    assert BARELY_CORRELATE_NOTE in reason
    assert "may wrap" not in reason
    assert "away from the camera's own pattern" not in reason
    assert reason.endswith(REUSE_SENTENCE)
    assert reason.count("from_run_doc") == 1


def test_the_reuse_sentence_comes_after_the_wrap_hint(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """200 rows of a 384-row frame wrap to -184 (FM-120): the fit fails, and the sentence ends it."""
    microscope = demo_microscope_with_sample(shape=SHAPE)

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, step_um=200.0)

    reason = raised.value.reason
    wrap = "; a step over 40 % of the field may wrap: give a smaller step_um"
    assert BARELY_CORRELATE_NOTE in reason
    assert reason.endswith(wrap + REUSE_SENTENCE)
    assert "away from the camera's own pattern" not in reason


def test_a_refusal_that_is_not_for_texture_does_not_offer_the_reuse(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """An axis that moves a fraction of its command is the stage's fault: a reuse is no way out.

    The second look does not run either (the note is not "parallel shifts"), so
    the refusal says nothing of the camera's pattern.
    """
    microscope = demo_microscope_with_sample(
        shape=SHAPE, camera_rotation_deg=30.0, mirrored=True
    )
    microscope.override(
        XYStage, _FlakyStage(microscope.require(XYStage), x_fraction=0.1)
    )

    reason = _refusal(microscope, tmp_path)

    assert reason == f"{RIGIDITY_NOTE}; step {STEP_UM:g} µm"
    assert "from_run_doc" not in reason
    assert "away from the camera's own pattern" not in reason


def test_the_longest_parallel_shifts_refusal_stays_within_the_cards_600_characters() -> (
    None
):
    """Five-digit spreads, the 2000 µm step, the wrap hint, the widest ratios: 572 characters.

    The page keeps 600 characters of a failure's words, and a cut would take the
    end of the reuse sentence, the part that says nothing moves. This is the
    longest refusal of the design's measure, with ``parallel shifts`` as its only
    note: a fit that has both texture notes (a weak response and parallel shifts)
    is 628 characters in the same case, and the card clips it (FM-269, *Open*).
    """
    parallel = Fit(
        ((0.0, 0.0), (0.0, 0.0)), 0.0, 0.0, False, 0.0, (PARALLEL_SHIFTS_NOTE,)
    )

    reason = _unusable_reason(
        parallel, 2000.0, (1412, 1412), 1.0, False, (12345.0, 12345.0), (99.9, 99.9)
    )

    assert "12345 and 12345 counts" in reason
    assert "99.9 and 99.9 times the next" in reason
    assert "step 2000 µm" in reason
    assert "may wrap" in reason
    assert reason.endswith(REUSE_SENTENCE)
    assert len(reason) == 572
    assert len(reason) <= CARD_TEXT_CHARS


def test_what_a_refusal_and_the_assumed_line_say_has_no_angle_bracket() -> None:
    """The plate page shows these texts in a pane that a browser parses as HTML (#625).

    ``from_run_doc=<its run document>`` read ``from_run_doc= (nothing moves)`` on the
    Stand screen: words between angle brackets are a tag there. No text of the tool
    that the page can carry has one, whatever the note.
    """
    parallel = Fit(
        ((0.0, 0.0), (0.0, 0.0)), 0.0, 0.0, False, 0.0, (PARALLEL_SHIFTS_NOTE,)
    )
    weak = Fit(((0.0, 0.0), (0.0, 0.0)), 0.0, 0.0, False, 0.0, (BARELY_CORRELATE_NOTE,))
    for fitted in (parallel, weak):
        reason = _unusable_reason(
            fitted, 96.0, SHAPE, 1.0, True, (5138.0, 5138.0), (1.2, 1.1)
        )
        assert "<" not in reason
        assert ">" not in reason
    line = sparse_field_line((4.7, 4.9))
    assert "<" not in line
    assert ">" not in line


@pytest.mark.parametrize(
    ("notes", "first_x", "first_y", "shape", "asked"),
    [
        ((PARALLEL_SHIFTS_NOTE,), (0.0, 0.0), (0.0, 0.0), SHAPE, True),
        # One probe at zero is enough: the other may be right or wrapped.
        ((PARALLEL_SHIFTS_NOTE,), (0.4, -0.3), (30.0, 0.0), SHAPE, True),
        ((PARALLEL_SHIFTS_NOTE,), (30.0, 0.0), (0.0, 1.5), SHAPE, True),
        # Neither probe is at zero: not the camera's pattern.
        ((PARALLEL_SHIFTS_NOTE,), (30.0, 0.0), (30.0, 0.0), SHAPE, False),
        ((PARALLEL_SHIFTS_NOTE,), (3.0, 0.0), (0.0, 3.0), SHAPE, False),
        # Not "parallel shifts": a weak response, a fit that is no camera, a good one.
        ((BARELY_CORRELATE_NOTE,), (0.0, 0.0), (0.0, 0.0), SHAPE, False),
        ((RIGIDITY_NOTE,), (0.0, 0.0), (0.0, 0.0), SHAPE, False),
        ((), (0.0, 0.0), (0.0, 0.0), SHAPE, False),
        # A frame with no room outside the exclusion has no second look.
        ((PARALLEL_SHIFTS_NOTE,), (0.0, 0.0), (0.0, 0.0), (48, 512), False),
        ((PARALLEL_SHIFTS_NOTE,), (0.0, 0.0), (0.0, 0.0), (384, 48), False),
        ((PARALLEL_SHIFTS_NOTE,), (0.0, 0.0), (0.0, 0.0), (49, 49), True),
    ],
    ids=[
        "both-zero",
        "x-zero",
        "y-near-zero",
        "neither-zero",
        "both-over-two-px",
        "barely-correlate-only",
        "rigidity-only",
        "trustworthy",
        "rows-at-the-limit",
        "columns-at-the-limit",
        "just-over-the-limit",
    ],
)
def test_the_second_look_is_asked_only_where_the_design_says(
    notes: tuple[str, ...],
    first_x: tuple[float, float],
    first_y: tuple[float, float],
    shape: tuple[int, int],
    asked: bool,
) -> None:
    """D3: "parallel shifts" and a probe at zero, on a frame large enough to look away from zero."""
    fitted = Fit(((0.0, 0.0), (0.0, 0.0)), 0.0, 0.0, False, 0.0, notes)
    shift_x = Shift(first_x[0], first_x[1], 0.9)
    shift_y = Shift(first_y[0], first_y[1], 0.9)

    assert _wants_a_second_look(fitted, (shift_x, shift_y), shape) is asked


def test_a_second_look_that_does_not_stand_out_or_is_no_camera_is_not_believed() -> (
    None
):
    """The ratio and the rigid check each refuse on their own; both are asked, both must pass."""
    good_x, good_y = Shift(96.0, 0.0, 0.9), Shift(0.0, -96.0, 0.9)
    stands = (AwayShift(good_x, 4.0), AwayShift(good_y, 4.0))
    assert _second_look_holds(stands)
    # Either probe's ratio under 1.5, however good its shift.
    assert not _second_look_holds((AwayShift(good_x, 1.4), AwayShift(good_y, 4.0)))
    assert not _second_look_holds((AwayShift(good_x, 4.0), AwayShift(good_y, 1.4)))
    assert not _second_look_holds((AwayShift(good_x, math.nan), AwayShift(good_y, 4.0)))
    # Ratios that stand out for two shifts that are no camera's: not at a right angle.
    slanted = Shift(60.0, -75.0, 0.9)
    assert not _second_look_holds((AwayShift(good_x, 4.0), AwayShift(slanted, 4.0)))
    # And not as long as each other.
    short = Shift(0.0, -60.0, 0.9)
    assert not _second_look_holds((AwayShift(good_x, 4.0), AwayShift(short, 4.0)))


def test_a_lit_frame_of_a_12_bit_mode_is_calibrated_and_its_dark_frame_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The floor follows the camera's bit depth: the same light reads 16 times fewer counts."""
    lit = _ShallowCamera(
        demo_microscope, PlateSample(), pixel_size_um=1.0, shape=SHAPE, dark_snaps=()
    )
    demo_microscope.override(Camera, lit)
    assert signal_spread(lit.snap()) < MIN_SIGNAL_SPREAD_COUNTS

    result = _result(_run(demo_microscope, tmp_path))

    assert result.pixel_size_um == pytest.approx(1.0, rel=0.02)
    assert result.assumed == ()

    dark = _ShallowCamera(
        demo_microscope, PlateSample(), pixel_size_um=1.0, shape=SHAPE, dark_snaps=(1,)
    )
    demo_microscope.override(Camera, dark)

    reason = _refusal(demo_microscope, tmp_path)

    assert "the camera's frame is dark or flat" in reason
    assert "counts, under 3.125)" in reason


def test_the_precision_line_comes_before_the_returns_lines(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """A step of a few pixels and a stage that comes back 8 µm off: both are in ``assumed``, in this order."""
    microscope = demo_microscope_with_sample(shape=SHAPE)
    microscope.override(
        XYStage,
        _FlakyStage(microscope.require(XYStage), drift_um={3: (8.0, 0.0)}),
    )

    result = _result(_run(microscope, tmp_path, step_um=20.0))

    assert len(result.assumed) == 2
    assert result.assumed[0].startswith("step of 20 µm fits poorly")
    assert result.assumed[1].startswith("the stage came back")


def test_a_fallback_step_of_a_few_pixels_is_named_with_a_suggested_step(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """F6: with no pixel size known, 50 µm at 2.745 µm per pixel is 18 px of the frame's 384 rows."""
    demo_microscope.override(
        Camera,
        _ReportingCamera(
            demo_microscope,
            PlateSample(),
            pixel_size_um=2.745,
            shape=SHAPE,
            reported_um=0.0,
        ),
    )

    result = _result(_run(demo_microscope, tmp_path))

    assert result.step_um == 50.0
    assert result.assumed == (
        "step of 50 µm: no pixel size was known",
        "step of 50 µm fits poorly (18 px, under a tenth of the 384 px frame): "
        "run again with step_um=260",
    )
    # The suggestion is a step the plugin accepts, and it fits without a note.
    again = _result(_run(demo_microscope, tmp_path, step_um=260.0))
    assert again.assumed == ()
    assert again.pixel_size_um == pytest.approx(2.745, rel=0.01)


def test_a_given_step_of_a_few_pixels_is_named_too(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = demo_microscope_with_sample(shape=SHAPE)

    result = _result(_run(microscope, tmp_path, step_um=20.0))

    assert result.assumed == (
        "step of 20 µm fits poorly (20 px, under a tenth of the 384 px frame): "
        "run again with step_um=96",
    )


def test_a_poor_fit_at_the_default_step_is_named_without_a_step_to_suggest(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic plate has no residual to speak of, so the reason is given to the plugin.

    A residual that is large at a quarter of the field is not cured by a probe
    of the same size: the line names it and says nothing to run again with.
    """
    microscope = demo_microscope_with_sample(shape=SHAPE)
    monkeypatch.setattr(
        "smc.plugins.camera_calibration.plugin.precision_reasons",
        lambda fitted, step_um, image_shape_px: ("residual 60.0 µm, 6 % of the step",),
    )

    result = _result(_run(microscope, tmp_path))

    assert result.step_um == STEP_UM
    assert result.assumed == (
        f"step of {STEP_UM:g} µm fits poorly (residual 60.0 µm, 6 % of the step)",
    )


@pytest.mark.parametrize(
    ("pixel_size_um", "shape", "expected_um"),
    [
        (2.74526, (1412, 1412), 970.0),  # session 3's 4x: 969.0, two figures
        (0.276136, (1412, 1412), 97.0),  # the 40x: 97.47
        (2.7473, SHAPE, 260.0),  # a quarter of 384 rows: 263.7
        (30.0, SHAPE, 2000.0),  # 2880 µm, capped at the largest probe
    ],
)
def test_the_suggested_step_is_a_quarter_of_the_field_to_two_figures_and_capped(
    pixel_size_um: float, shape: tuple[int, int], expected_um: float
) -> None:
    assert suggested_step_um(pixel_size_um, shape) == expected_um


def test_the_longest_precision_line_fits_the_strip() -> None:
    """FM-205: the strip of the page keeps the first ``assumed`` line to 160 characters.

    Built from what ``precision_reasons`` says, not from typed texts: both
    reasons, a four-figure step, a residual of four figures and a 4096 px frame.
    """
    fitted = Fit(((0.0, 0.0), (0.0, 0.0)), 10.0, 0.0, False, 1999.9, ())
    reasons = precision_reasons(fitted, 1234.56, (4096, 4096))
    assert len(reasons) == 2

    line = precision_line(1234.56, reasons, suggested_step_um(10.0, (4096, 4096)))

    assert line == (
        "step of 1234.56 µm fits poorly (123 px, under a tenth of the 4096 px frame; "
        "residual 1999.9 µm, 162 % of the step): run again with step_um=2000"
    )
    assert len(line) <= 160


def test_a_precision_line_without_a_larger_step_to_suggest_ends_at_the_reasons() -> (
    None
):
    line = precision_line(970.0, ("residual 60.0 µm, 6 % of the step",), None)

    assert line == "step of 970 µm fits poorly (residual 60.0 µm, 6 % of the step)"
