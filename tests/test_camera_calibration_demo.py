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
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XY, Camera, Limits, ObjectiveTurret, XYStage
from smc.hardware.errors import HardwareError, MicroscopeHaltedError
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.slots import CameraCalibration
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.fixtures import SampleMicroscopeFactory
from smc.testing.synthetic import PlateSample, SampleCamera

pytestmark = pytest.mark.demo

SHAPE = (384, 512)  # (height, width): not square, on purpose (FM-73)
STEP_UM = 96.0  # 0.25 * 384 rows * 1.0 µm per pixel
LOGGER = "smc.plugins.camera-calibration"


class _FlakyStage:
    """The demo's stage, except that a numbered move fails or lands off target.

    It implements the seven methods of ``XYStage``: ``Microscope.override``
    refuses an implementation that lacks one. ``moves`` counts the absolute
    moves it was asked for, so a test can say that no return was sent.
    """

    def __init__(
        self,
        real: XYStage,
        *,
        fail_on_move: int | None = None,
        drift_um: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        self._real = real
        self._fail_on_move = fail_on_move
        self._drift_um = drift_um or {}
        self.moves = 0

    def position_um(self) -> XY:
        return self._real.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        if self.moves == self._fail_on_move:
            raise HardwareError("the stage did not answer")
        dx, dy = self._drift_um.get(self.moves, (0.0, 0.0))
        return self._real.move_to_um(x_um + dx, y_um + dy)

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
            dry_run=False,
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
    assert _stored(tmp_path).result is None


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
