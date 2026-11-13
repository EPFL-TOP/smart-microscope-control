"""``plate-walls`` on the demo devices and the fake core: the walk, the fit and what it leaves (m3 §17, #328).

The demo camera ignores the stage, so a ``PlateSample`` is put on the stand
through a camera whose frames follow it (``_Camera``, a ``SampleCamera`` that
also records where each frame was taken). The sample is a 96-well plate turned
1.2° with A1 at the origin, seen by a camera turned 30° and mirrored on a 384 x
512 frame (FM-73) at 0.65 µm per pixel. ``camera-calibration`` runs first on the
same run document through ``run_plugin`` (the seam, FM-124): the matrix the
plugin reads is a measured one, 2 % and a fraction of a degree off the truth.
The demo stage moves at its ``Velocity`` property, 10 mm/s by default: the demo
tests raise it to 100 000 on the demo's own core. The tests that are about the
loop's decisions and not about the stage run on ``fake_microscope``, whose
stage is instant, with a calibration written as the stage would leave it.

Expected values are written here, from the plate's geometry and the arithmetic
of the design, never read from the result's own fields (FM-124). The module runs
with ``RuntimeWarning`` as an error (FM-131).
"""

from __future__ import annotations

import io
import json
import logging
import math
import threading
import weakref
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import tifffile
import useq

from recording_autofocus import RecordingAutofocus
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Limits,
    ObjectiveOffset,
    ObjectiveTurret,
    XYStage,
)
from smc.hardware.errors import (
    CoreError,
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, Progress, RunContext, load, run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.slots import CameraCalibration, PlateResult, WellRef
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.fixtures import SampleMicroscopeFactory
from smc.testing.synthetic import PlateSample, SampleCamera
from smc.ui.guide import walls_square_wells

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

SHAPE = (384, 512)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 0.65
CAMERA_DEG = 30.0
MIRRORED = True
PLATE_DEG = 1.2
PITCH_UM = 9000.0  # the 96-well plate's
RADIUS_UM = 3200.0  # ... and its wells'
#: 0.75 of the 384-row side at 0.65 µm per pixel: the walk's step, by hand.
STEP_UM = 0.75 * 384 * PIXEL_UM
#: Half the radius, the default search: 1600 µm, which is 8 steps and 17 frames a walk.
FULL_WALK = 1 + 2 * math.floor(0.5 * RADIUS_UM / STEP_UM)
LOGGER = "smc.plugins.plate-walls"
#: Where each well is off from where the operator "centred" it: up to 300 µm.
ROUGH_UM = {
    "A1": (300.0, -200.0),
    "H12": (-250.0, 150.0),
    "H1": (200.0, 250.0),
    "A12": (-100.0, 200.0),
}
#: The wells' (column, row) indices, written out here and not read from useq.
INDEX = {"A1": (0, 0), "H1": (0, 7), "H12": (11, 7), "A12": (11, 0), "D7": (6, 3)}


def _truth(
    well: str,
    *,
    pitch_um: float = PITCH_UM,
    rotation_deg: float = PLATE_DEG,
    a1: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """A well's centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    col, row = INDEX[well]
    angle = math.radians(rotation_deg)
    dx, dy = col * pitch_um, -row * pitch_um
    return (
        a1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        a1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _matrix(
    pixel_um: float = PIXEL_UM,
    rotation_deg: float = CAMERA_DEG,
    mirrored: bool = MIRRORED,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``ps * R(theta) * diag(s, -1)``: §3.1's formula, written out."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    flip = -1.0 if mirrored else 1.0
    return (
        (pixel_um * cos * flip, pixel_um * sin),
        (pixel_um * sin * flip, -pixel_um * cos),
    )


def _refs(
    *wells: str, plate_deg: float = PLATE_DEG, pitch_um: float = PITCH_UM
) -> tuple[WellRef, ...]:
    """The wells as the operator "centred" them: the true centre and a rough error."""
    refs = []
    for well in wells:
        x, y = _truth(well, pitch_um=pitch_um, rotation_deg=plate_deg)
        off_x, off_y = ROUGH_UM[well]
        refs.append(WellRef(well=well, x_um=x + off_x, y_um=y + off_y))
    return tuple(refs)


# ------------------------------------------------------------------ the stand


class _Camera(SampleCamera):
    """A sample camera that records where each frame was taken and can act at a numbered snap.

    ``before(n)`` runs at snap ``n`` before the frame is rendered, and ``spoil(n,
    frame)`` may replace it. ``reset()`` forgets the snaps of the calibration run
    that came first, so that a test counts the registration's own.
    """

    def __init__(
        self,
        microscope: Microscope,
        sample: PlateSample,
        *,
        shape: tuple[int, int] = SHAPE,
        before: Callable[[int], None] | None = None,
        spoil: Callable[[int, npt.NDArray[np.uint16]], npt.NDArray[Any]] | None = None,
    ) -> None:
        super().__init__(
            microscope,
            sample,
            pixel_size_um=PIXEL_UM,
            shape=shape,
            camera_rotation_deg=CAMERA_DEG,
            mirrored=MIRRORED,
        )
        self._stand = weakref.ref(microscope)
        self.snaps = 0
        self.positions: list[tuple[float, float]] = []
        self._before = before
        self._spoil = spoil

    def reset(self) -> None:
        self.snaps = 0
        self.positions.clear()

    def snap(self) -> npt.NDArray[Any]:
        self.snaps += 1
        stand = self._stand()
        assert stand is not None
        here = stand.require(XYStage).position_um()
        self.positions.append((here.x_um, here.y_um))
        if self._before is not None:
            self._before(self.snaps)
        frame = super().snap()
        return frame if self._spoil is None else self._spoil(self.snaps, frame)


class _EmptyTurret:
    """A turret at an empty position: the protocol lists them, and their label is ``""`` (FM-125)."""

    def labels(self) -> list[str]:
        return ["", "Nikon 10X S Fluor"]

    def current(self) -> str:
        return ""

    def select(self, label: str, *, confirm: bool = False) -> str:
        raise AssertionError("plate-walls never turns the turret")

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0

    def offset(self, label: str) -> ObjectiveOffset:
        return ObjectiveOffset(label, None, None, None, "")


class _Stage:
    """The stand's XY stage, except where a test makes a move misbehave; the others delegate.

    ``raises`` maps a move number to what it raises; ``drift_um`` is added to
    every target, so the stage lands off it and reads back where it really is;
    ``nan_readback_at`` is a move that arrives and reads back a position that is
    not finite; ``limits`` is what ``limits_um()`` says (the demo has none);
    ``unreadable_after_fault`` makes the position unreadable once a move raised;
    ``travelling`` is the position a stage still on its way from an earlier move
    reads until ``wait()`` is called (FM-85). It implements the seven methods of
    ``XYStage``: ``Microscope.override`` refuses an implementation that lacks
    one, and it bypasses the ``Executor``.
    """

    def __init__(
        self,
        inner: XYStage,
        *,
        raises: dict[int, BaseException] | None = None,
        drift_um: tuple[float, float] = (0.0, 0.0),
        nan_readback_at: int | None = None,
        limits: tuple[Limits, Limits] | None = None,
        unreadable_after_fault: bool = False,
        travelling: tuple[float, float] | None = None,
    ) -> None:
        self._inner = inner
        self._raises = raises or {}
        self._drift_um = drift_um
        self._nan_readback_at = nan_readback_at
        self._limits = limits
        self._unreadable_after_fault = unreadable_after_fault
        self._travelling = travelling
        self._faulted = False
        self.moves = 0
        self.targets: list[tuple[float, float]] = []

    def position_um(self) -> XY:
        if self._faulted and self._unreadable_after_fault:
            raise HardwareError("the position cannot be read")
        if self._travelling is not None:
            return XY(*self._travelling)
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        self.targets.append((x_um, y_um))
        if self.moves in self._raises:
            self._faulted = True
            raise self._raises[self.moves]
        reached = self._inner.move_to_um(
            x_um + self._drift_um[0], y_um + self._drift_um[1]
        )
        if self.moves == self._nan_readback_at:
            return XY(math.nan, math.nan)
        return reached

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("plate-walls makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._travelling = None
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._limits


def _plate(
    *, pitch_um: float = PITCH_UM, rotation_deg: float = PLATE_DEG
) -> PlateSample:
    """A 96-well plate with A1 at the origin; another pitch builds a plate of its own."""
    if pitch_um == PITCH_UM:
        return PlateSample("96-well", (0.0, 0.0), rotation_deg, seed=11)
    return PlateSample(
        useq.WellPlate(
            name="96-well-of-another-pitch",
            rows=8,
            columns=12,
            well_spacing=(pitch_um / 1000.0, pitch_um / 1000.0),
            well_size=(6.4, 6.4),
            circular_wells=True,
        ),
        (0.0, 0.0),
        rotation_deg,
        seed=11,
    )


def _on_demo(
    factory: SampleMicroscopeFactory, sample: PlateSample | None = None, **hooks: Any
) -> tuple[Microscope, _Camera]:
    """The demo stand with a camera that follows the stage, and a stage fast enough to walk.

    The demo's XY stage moves at its ``Velocity`` property, 10 mm/s by default:
    a 100 mm move takes 10 s. At 100 000 it takes 13 ms (measured). The property
    is the demo adapter's own, set on this stand's core and nowhere else.
    """
    sample = sample if sample is not None else _plate()
    stand = factory(
        sample,
        pixel_size_um=PIXEL_UM,
        shape=SHAPE,
        camera_rotation_deg=CAMERA_DEG,
        mirrored=MIRRORED,
    )
    stand.core.setProperty("XY", "Velocity", "100000")
    camera = _Camera(stand, sample, **hooks)
    stand.override(Camera, camera)
    return stand, camera


def _on_fake(
    fake_microscope: Microscope, sample: PlateSample | None = None, **hooks: Any
) -> _Camera:
    camera = _Camera(
        fake_microscope, sample if sample is not None else _plate(), **hooks
    )
    fake_microscope.override(Camera, camera)
    return camera


@pytest.fixture
def open_limited(mm_available: bool) -> Iterator[Callable[..., Microscope]]:
    """A demo stand whose XY soft limits are given (the layer's own refusal), closed afterwards."""
    opened: list[Microscope] = []

    def make(limits: tuple[tuple[float, float], tuple[float, float]]) -> Microscope:
        if not mm_available:
            pytest.skip("Micro-Manager demo adapters not installed")
        profile = Profile.demo()
        limited = profile.model_copy(
            update={
                "safety": profile.safety.model_copy(
                    update={"xy_soft_limits_um": limits}
                )
            }
        )
        microscope = Microscope.open(limited, dry_run=False)
        opened.append(microscope)
        return microscope

    yield make
    for microscope in opened:
        microscope.close()


# ----------------------------------------------------------------- the helpers


def _calibrate(stand: Microscope, camera: _Camera, run_doc: Path) -> CameraCalibration:
    """Measure the camera on the stand, on the run document the registration reads; forget its snaps."""
    record = run_plugin(
        CameraCalibrationPlugin(), stand, CameraCalibrationParams(), run_doc
    )
    assert record.status is StageStatus.done
    assert record.result is not None
    camera.reset()
    return CameraCalibration.model_validate(record.result)


def _seed(
    run_doc: Path,
    stand: Microscope,
    *,
    objective: str | object | None = "in place",
    shape: tuple[int, int] = SHAPE,
    matrix: tuple[tuple[float, float], tuple[float, float]] | None = None,
    dry_run: bool = False,
) -> None:
    """A ``camera-calibration`` record as an earlier run would have left it: the true matrix, no measuring."""
    if objective == "in place":
        objective = stand.require(ObjectiveTurret).current()
    result = CameraCalibration(
        objective=cast("str | None", objective),
        image_shape_px=shape,
        matrix_um_per_px=matrix if matrix is not None else _matrix(),
        pixel_size_um=PIXEL_UM,
        rotation_deg=CAMERA_DEG,
        mirrored=MIRRORED,
        residual_um=0.01,
        response=0.9,
        return_error_um=0.2,
        step_um=62.4,
    )
    then = datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc)
    write_stage(
        run_doc,
        StageRecord(
            plugin="camera-calibration",
            provides="camera-calibration",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _run(
    stand: Microscope,
    run_doc: Path,
    refs: tuple[WellRef, ...],
    *,
    on_progress: Callable[[Progress], None] | None = None,
    stop: threading.Event | None = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        PlateWallsPlugin(),
        stand,
        PlateWallsParams(**{"plate": "96-well", "refs": refs, **params}),
        run_doc,
        on_progress=on_progress,
        stop=stop,
    )


def _result(record: StageRecord) -> PlateResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return PlateResult.model_validate(record.result)


def _stored(run_doc: Path) -> StageRecord:
    return RunDocument.load(run_doc).stages["plate-walls"]


def _here(stand: Microscope) -> tuple[float, float]:
    position = stand.require(XYStage).position_um()
    return position.x_um, position.y_um


def _refused(
    stand: Microscope, run_doc: Path, refs: tuple[WellRef, ...], **params: Any
) -> PluginRunError:
    with pytest.raises(PluginRunError) as raised:
        _run(stand, run_doc, refs, **params)
    assert raised.value.plugin == "plate-walls"
    # FM-98: a refusal is raised outside the handler of the error it stands for
    # (the layer's `SafetyRefusedError`, the fit's `ValueError`), so that error is
    # not on its traceback, where it would keep the core's frames alive.
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    return raised.value


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_plate_walls() -> None:
    entry = load("plate-walls")

    assert entry.plugin.provides == "plate"
    assert entry.plugin.requires == (XYStage, Camera)
    assert entry.plugin.reads == ("camera-calibration",)
    assert entry.plugin.reads_optional == ()
    assert entry.target == "smc.plugins.plate_walls.plugin:PlateWallsPlugin"


def test_an_unknown_plate_is_a_parameter_error_before_the_stand_opens() -> None:
    with pytest.raises(ValueError, match="96-WELL"):
        PlateWallsParams(plate="96-WELL")


# ----------------------------------------------------------- the registration


def test_two_wells_register_the_plate_of_the_sample(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    progress: list[Progress] = []

    record = _run(stand, run_doc, _refs("A1", "H12"), on_progress=progress.append)

    result = _result(record)
    assert result.method == "wall-images"
    # A1 at the origin and the plate turned 1.2°, from two wells 117 mm apart.
    assert result.a1_center_xy_um == pytest.approx((0.0, 0.0), abs=2.0)
    assert result.rotation_deg == pytest.approx(PLATE_DEG, abs=0.01)
    assert result.residual_um is not None
    assert result.residual_um < 2.0
    assert result.scale_error == pytest.approx(0.0, abs=2e-4)
    assert result.fit is None
    # The references are the fitted centres, in the order given, to a micrometre.
    assert [r.well for r in result.refs] == ["A1", "H12"]
    assert result.n_refs == 2
    for ref in result.refs:
        assert math.dist((ref.x_um, ref.y_um), _truth(ref.well)) < 1.0
    # The plate fitted is the sample's: where `goto-well` would drive to A12.
    assert result.well_center_um("A12") == pytest.approx(_truth("A12"), abs=2.0)
    assert result.assumed == (
        "the stage reports no soft limits, so the frames' moves were not checked "
        "against any travel",
        "two wells: the residual checks their distance only; a third well checks the fit",
    )
    assert result.walls is not None
    assert result.walls.objective == stand.require(ObjectiveTurret).current()
    assert result.walls.image_shape_px == SHAPE
    assert result.walls.search_um == pytest.approx(0.5 * RADIUS_UM)
    assert result.walls.pixel_size_um == pytest.approx(PIXEL_UM, rel=0.03)
    assert [w.well for w in result.walls.wells] == ["A1", "H12"]
    assert result.walls.elapsed_s > 0.0
    # Every frame the stand took is counted, and none was taken for nothing.
    assert result.walls.frames == camera.snaps
    assert 2 * 4 * 2 <= camera.snaps <= 2 * 4 * (FULL_WALK + 1)
    assert [(p.done, p.total, p.message) for p in progress] == [
        (i + 1, 8, f"{well} {side}")
        for i, (well, side) in enumerate(
            (w, s) for w in ("A1", "H12") for s in ("+X", "+Y", "-X", "-Y")
        )
    ]
    assert _stored(run_doc) == record


def test_the_wall_registration_records_its_objective(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """The plate carries the objective of its frames (m3 §18.1), for the scan that may use another."""
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)

    record = _run(stand, run_doc, _refs("A1", "H12"))

    result = _result(record)
    assert result.objective == "Nikon 10X S Fluor"
    assert result.walls is not None
    assert result.walls.objective == result.objective
    stored = _stored(run_doc).result
    assert stored is not None
    assert stored["objective"] == "Nikon 10X S Fluor"


def test_three_wells_fill_the_plate_slot_with_their_walls(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)

    result = _result(_run(stand, run_doc, _refs("A1", "H1", "H12")))

    assert result.method == "wall-images"
    assert result.n_refs == 3
    assert [r.well for r in result.refs] == ["A1", "H1", "H12"]
    assert result.rotation_deg == pytest.approx(PLATE_DEG, abs=0.01)
    assert result.residual_um is not None
    assert result.residual_um < 2.0
    # Three wells check the fit: no line about two.
    assert not [line for line in result.assumed if line.startswith("two wells")]
    walls = result.walls
    assert walls is not None
    assert walls.frames == camera.snaps
    assert walls.folder is not None
    assert walls.folder.startswith("plate-walls/")
    folder = tmp_path / walls.folder
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"{well}_{side}.tif"
        for well in ("A1", "H1", "H12")
        for side in ("+X", "+Y", "-X", "-Y")
    )
    first = 0
    for fitted in walls.wells:
        truth = _truth(fitted.well)
        assert [w.side for w in fitted.walls] == ["+X", "+Y", "-X", "-Y"]
        assert fitted.recorded_xy_um == pytest.approx(
            (truth[0] + ROUGH_UM[fitted.well][0], truth[1] + ROUGH_UM[fitted.well][1])
        )
        assert math.dist(fitted.center_xy_um, truth) < 1.0
        assert fitted.diameter_um == pytest.approx(2 * RADIUS_UM, abs=2.0)
        assert fitted.size_error == pytest.approx(0.0, abs=1e-3)
        # The well's points are the walls' points, none trimmed on a clean circle,
        # and they lie a pixel or less from it: not 0 (the pixels are quantised).
        assert 0.0 < fitted.rms_um < 2.0
        assert fitted.n_points == sum(w.n_points for w in fitted.walls)
        for wall in fitted.walls:
            # Under a camera turned 30° and mirrored, +X and -X run along the 384
            # rows' bands (48 lines) and +Y and -Y along the 512 columns' (64); a
            # wall is found by at least half of them, and by no more than all.
            lines = 48 if wall.side in ("+X", "-X") else 64
            assert lines / 2 <= wall.n_points <= lines
            first += wall.frames
            # The frame was taken where the stage says it was (the camera read it
            # there), and it is centred on the wall: its distance from the well's
            # centre is the radius, to a few µm (the wall curves over the frame).
            assert (wall.x_um, wall.y_um) == pytest.approx(
                camera.positions[first - 1], abs=0.05
            )
            assert math.dist((wall.x_um, wall.y_um), truth) == pytest.approx(
                RADIUS_UM, abs=8.0
            )
            assert wall.file == f"{fitted.well}_{wall.side}.tif"
            saved = folder / wall.file
            with tifffile.TiffFile(io.BytesIO(saved.read_bytes())) as handle:
                assert handle.asarray().shape == SHAPE
                meta = json.loads(handle.pages[0].tags["ImageDescription"].value)  # type: ignore[union-attr]
            assert (meta["well"], meta["side"]) == (fitted.well, wall.side)
            assert (meta["x_um"], meta["y_um"]) == pytest.approx((wall.x_um, wall.y_um))
    assert first == camera.snaps


def test_a_frame_is_placed_where_the_stage_says_it_was(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """FM-142: a stage that lands 20 µm off every target and says where it is must not move the centres.

    Placed at the target, each frame's wall would be 20 µm off and so would the
    centre; placed at the readback the centre is the sample's to a micrometre.
    """
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    stage = _Stage(stand.require(XYStage), drift_um=(17.0, -11.0))
    stand.override(XYStage, stage)

    result = _result(_run(stand, run_doc, _refs("A1", "H12")))

    for ref in result.refs:
        assert math.dist((ref.x_um, ref.y_um), _truth(ref.well)) < 1.0
    # The drift is real: every frame was taken 20 µm from where it was sent.
    assert stage.moves == camera.snaps + 1  # the end move has no frame
    for (sent_x, sent_y), here in zip(stage.targets, camera.positions, strict=False):
        assert math.dist((sent_x, sent_y), here) == pytest.approx(20.2, abs=0.2)


def test_the_stage_ends_at_the_last_wells_centre(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)

    result = _result(_run(stand, run_doc, _refs("A1", "H12")))

    # One move to the last well's fitted centre, where the operator sees on Live
    # that the stage sits in the middle of a well.
    assert _here(stand) == pytest.approx(
        (result.refs[-1].x_um, result.refs[-1].y_um), abs=0.1
    )
    assert math.dist(_here(stand), _truth("H12")) < 1.0
    assert _here(stand) != pytest.approx(camera.positions[-1], abs=1.0)


def test_a_plate_of_another_pitch_is_refused_by_the_residual(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """A 9.03 mm pitch fitted as 9.0: 0.33 % over 117 mm is under the scale tolerance, over the residual bound.

    Each of two wells ends 196 µm from where the other puts it (the measured
    spread, §17.1), against the bound of 100.
    """
    sample = _plate(pitch_um=9030.0)
    stand, camera = _on_demo(demo_microscope_with_sample, sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)

    refused = _refused(
        stand, run_doc, _refs("A1", "H12", pitch_um=9030.0), max_residual_um=100.0
    )

    assert refused.reason.startswith("H12 is ") or refused.reason.startswith("A1 is ")
    assert refused.reason.endswith(
        " µm from where the other wells put it (bound 100 µm): record it again, "
        "or check the plate type"
    )
    distance = float(refused.reason.split(" is ")[1].split(" µm")[0])
    assert 150.0 <= distance <= 250.0
    assert _stored(run_doc).status is StageStatus.failed
    assert _stored(run_doc).result is None


def test_a_recorded_centre_outside_its_well_is_refused_and_named(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """H12 recorded 4.5 mm out of its well: the +X walk looks at plastic, in the 17 frames of the walk."""
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    truth = _truth("H12")
    refs = (
        _refs("A1")[0],
        WellRef(well="H12", x_um=truth[0] + 4500.0, y_um=truth[1]),
    )
    here = _here(stand)

    refused = _refused(stand, run_doc, refs)

    recorded = (truth[0] + 4500.0, truth[1])
    assert refused.reason == (
        f"no wall found on H12's +X side in {FULL_WALK} frames, from 1600 to 4800 µm "
        f"out of the recorded centre ({recorded[0]:.1f}, {recorded[1]:.1f}) µm: check "
        "that the light is on (the inside of a well is brighter than the plastic in "
        "brightfield) and that the stage was in H12 when it was recorded, then "
        "record it again"
    )
    # The stage stays where it is: at the last frame of the walk, not at the start.
    assert _here(stand) == pytest.approx(camera.positions[-1], abs=0.1)
    assert _here(stand) != pytest.approx(here, abs=100.0)
    assert _stored(run_doc).status is StageStatus.failed


# ------------------------------------------------- stop, halt and faults: FM-88


def test_a_stop_leaves_the_stage_where_it_is(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stop = threading.Event()
    stand, camera = _on_demo(
        demo_microscope_with_sample,
        before=lambda n: stop.set() if n == 7 else None,
    )
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    stage = _Stage(stand.require(XYStage))
    stand.override(XYStage, stage)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(stand, run_doc, _refs("A1", "H12"), stop=stop)

    assert record.status is StageStatus.stopped
    # The stop came at the 7th frame; nothing moved after it, and no end move.
    assert camera.snaps == 7
    assert stage.moves == 7
    assert _here(stand) == pytest.approx(camera.positions[-1], abs=0.1)
    (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert line.startswith("plate-walls stopped at ")
    assert "the stage is left at (" in line
    assert "where it started" in line


def test_a_stop_after_the_last_frame_makes_no_end_move(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stop asked for as the last side is reported: the fit is fine, and the stage still does not go to the well's centre."""
    stop = threading.Event()
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    stage = _Stage(stand.require(XYStage))
    stand.override(XYStage, stage)

    def stop_on_the_last_side(progress: Progress) -> None:
        if progress.done == progress.total:
            stop.set()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(
            stand,
            run_doc,
            _refs("A1", "H12"),
            on_progress=stop_on_the_last_side,
            stop=stop,
        )

    assert record.status is StageStatus.stopped
    # Every move was a frame's: the move to the last well's centre was not made.
    assert stage.moves == camera.snaps
    assert _here(stand) == pytest.approx(camera.positions[-1], abs=0.1)
    (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert line.startswith("plate-walls stopped at H12's -Y wall: ")


def test_a_halt_leaves_the_stage_where_it_is(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-64, FM-88: the halt is the layer's, so the stage is the real one, and nothing moves to put anything back."""
    holder: list[Microscope] = []

    def halt_at_5(snap: int) -> None:
        if snap == 5:
            holder[0].stop()

    stand, camera = _on_demo(demo_microscope_with_sample, before=halt_at_5)
    holder.append(stand)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    where = _here(stand)

    try:
        with (
            caplog.at_level(logging.WARNING, logger=LOGGER),
            pytest.raises(MicroscopeHaltedError),
        ):
            _run(stand, run_doc, _refs("A1", "H12"))
    finally:
        stand.resume()

    assert _stored(run_doc).status is StageStatus.failed
    assert _stored(run_doc).result is None
    assert camera.snaps == 5
    assert _here(stand) == pytest.approx(camera.positions[-1], abs=0.1)
    assert _here(stand) != pytest.approx(where, abs=50.0)
    (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert "the stage is left at (" in line


def test_a_move_the_stage_stopped_is_not_a_refusal_of_the_layer(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-64: ``MotionStoppedError`` is not a ``SafetyRefusedError``, and passes unchanged."""
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    stage = _Stage(
        fake_microscope.require(XYStage),
        raises={3: MotionStoppedError("xy_stage XY")},
    )
    fake_microscope.override(XYStage, stage)

    with pytest.raises(MotionStoppedError):
        _run(fake_microscope, run_doc, _refs("A1", "H12"))

    assert stage.moves == 3
    assert camera.snaps == 2


# ------------------------------------------------------ what is refused first


def test_a_refused_move_names_the_well_and_the_side(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    stage = _Stage(
        fake_microscope.require(XYStage),
        raises={3: SafetyRefusedError("the target is outside the soft limits")},
    )
    fake_microscope.override(XYStage, stage)
    refs = _refs("A1", "H12")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        refused = _refused(fake_microscope, run_doc, refs)

    # A1 is recorded 300 µm off: the wall is further from the first frame (the
    # nominal radius, +X) than half of the 330 µm field, and from the second
    # (a step further out) more so. The third move is the walk's third offset,
    # r - s, a step in from the nominal wall along +X of the recorded centre.
    target = (refs[0].x_um + RADIUS_UM - STEP_UM, refs[0].y_um)
    assert stage.targets[2] == pytest.approx(target)
    assert refused.reason == (
        f"A1's +X wall: the move to ({target[0]:.1f}, {target[1]:.1f}) µm was "
        "refused: the target is outside the soft limits"
    )
    assert camera.snaps == 2
    # The stage is where the second frame left it, and a line says so.
    assert _here(fake_microscope) == pytest.approx(camera.positions[-1], abs=0.1)
    (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert line.startswith(
        "plate-walls stopped at A1's +X wall: the stage is left at ("
    )


def test_a_walk_outside_the_soft_limits_is_refused_before_anything_moves(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    refs = _refs("A1", "H12")
    # H12's +X walk reaches 3200 + 1600 µm along +X of the recorded centre at its
    # furthest offset; the stage is limited to 1000 µm past the nominal wall.
    limit_x = refs[1].x_um + RADIUS_UM + 1000.0
    stage = _Stage(
        fake_microscope.require(XYStage),
        limits=(Limits(-1e6, limit_x), Limits(-1e6, 1e6)),
    )
    fake_microscope.override(XYStage, stage)

    refused = _refused(fake_microscope, run_doc, refs)

    # A1's walks are well inside. H12's goes r, r+s, r-s, r+2s, ...: the first
    # offset past the limit is r + 6s (5 steps are 936 µm, 6 are 1123 µm).
    k = math.floor(1000.0 / STEP_UM) + 1
    assert k == 6
    x = refs[1].x_um + RADIUS_UM + k * STEP_UM
    assert refused.reason == (
        f"H12's +X wall would be looked for at ({x:.1f}, {refs[1].y_um:.1f}) µm, "
        "outside the stage's soft limits"
    )
    assert stage.moves == 0
    assert camera.snaps == 0


@pytest.mark.parametrize(
    ("limit", "well", "side", "direction"),
    [
        ("y_high", "A1", "+Y", (0.0, 1.0)),
        ("y_low", "H12", "-Y", (0.0, -1.0)),
        ("x_low", "A1", "-X", (-1.0, 0.0)),
    ],
)
def test_a_walk_that_leaves_the_travel_on_any_axis_is_refused_before_anything_moves(
    fake_microscope: Microscope,
    tmp_path: Path,
    limit: str,
    well: str,
    side: str,
    direction: tuple[float, float],
) -> None:
    """The check is on both axes and both ends (FM-73): one limit short each time, 1000 µm past the nominal wall."""
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    refs = _refs("A1", "H12")
    ref = next(r for r in refs if r.well == well)
    big = 1e6
    beyond = RADIUS_UM + 1000.0
    limits = {
        "y_high": (Limits(-big, big), Limits(-big, ref.y_um + beyond)),
        "y_low": (Limits(-big, big), Limits(ref.y_um - beyond, big)),
        "x_low": (Limits(ref.x_um - beyond, big), Limits(-big, big)),
    }[limit]
    stage = _Stage(fake_microscope.require(XYStage), limits=limits)
    fake_microscope.override(XYStage, stage)

    refused = _refused(fake_microscope, run_doc, refs)

    # The walk goes r, r+s, r-s, r+2s, ...: the first offset past 1000 µm beyond
    # the nominal wall is r + 6s, on the other axis the recorded coordinate.
    d = RADIUS_UM + 6 * STEP_UM
    x = ref.x_um + direction[0] * d
    y = ref.y_um + direction[1] * d
    assert refused.reason == (
        f"{well}'s {side} wall would be looked for at ({x + 0.0:.1f}, {y + 0.0:.1f}) "
        "µm, outside the stage's soft limits"
    )
    assert stage.moves == 0
    assert camera.snaps == 0


def test_a_frame_exactly_on_a_limit_is_inside_it_and_a_run_with_limits_says_nothing_of_none(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The layer's own test is ``low <= v <= high``: the walk's furthest frame on each of four limits passes."""
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    refs = _refs("A1", "H12")
    a1, h12 = refs
    # 1600 µm is 8 steps of 187.2: the furthest offset of the walk is r + 8s.
    d = RADIUS_UM + 8 * STEP_UM
    stage = _Stage(
        fake_microscope.require(XYStage),
        limits=(
            Limits(a1.x_um + -1.0 * d, h12.x_um + 1.0 * d),
            Limits(h12.y_um + -1.0 * d, a1.y_um + 1.0 * d),
        ),
    )
    fake_microscope.override(XYStage, stage)

    result = _result(_run(fake_microscope, run_doc, refs))

    # The stage said where it can go, so there is no line about the travel that
    # was not checked: only the two-wells line.
    assert result.assumed == (
        "two wells: the residual checks their distance only; a third well checks the fit",
    )
    assert result.walls is not None
    assert result.walls.frames == camera.snaps


def test_a_calibration_of_another_frame_is_refused_before_anything_moves(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope)
    stage = _Stage(fake_microscope.require(XYStage))
    fake_microscope.override(XYStage, stage)
    # A refusal that comes before anything moves comes before the autofocus is
    # touched too: a PFS switched off for a run that never moves is a loss (#478).
    af = RecordingAutofocus(engaged=True)
    fake_microscope.override(Autofocus, af)
    run_doc = tmp_path / "run.json"
    now = fake_microscope.require(ObjectiveTurret).current()
    ask = (
        f"run camera-calibration with {now} in place, on a field with texture: a "
        "well's wall will do"
    )
    cases: list[tuple[dict[str, Any], str]] = [
        (
            {"shape": (512, 384)},
            "the camera calibration is not of this stand: it is of a frame of "
            f"(512, 384), the camera gives (384, 512); {ask}",
        ),
        (
            {"objective": "Nikon 40X Plan Fluor ELWD"},
            "the camera calibration is not of this stand: it is of Nikon 40X Plan "
            f"Fluor ELWD, the camera is at {now}; {ask}",
        ),
        (
            {"objective": None},
            "the camera calibration is not of this stand: it is of an unknown "
            f"objective, the camera is at {now}; {ask}",
        ),
        (
            {"matrix": ((1.0, 2.0), (2.0, 4.0))},
            "the camera calibration's matrix cannot be inverted (determinant "
            f"0.0); {ask}",
        ),
    ]
    for seeded, reason in cases:
        run_doc.unlink(missing_ok=True)
        _seed(run_doc, fake_microscope, **seeded)

        refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"))

        assert refused.reason == reason
    assert stage.moves == 0
    assert camera.snaps == 0
    assert "disengage" not in af.calls
    assert af.engaged is True


def test_a_dry_run_is_refused(demo_microscope_dry: Microscope, tmp_path: Path) -> None:
    run_doc = tmp_path / "run.json"
    _seed(run_doc, demo_microscope_dry, dry_run=True)

    refused = _refused(demo_microscope_dry, run_doc, _refs("A1", "H12"))

    assert refused.reason == (
        "a dry run cannot image the walls; run it without --dry-run"
    )


def test_square_wells_are_refused(fake_microscope: Microscope, tmp_path: Path) -> None:
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    # No references at all: the plate is judged before them.
    refused = _refused(fake_microscope, run_doc, (), plate="384-well")

    assert refused.reason == (
        "plate-walls fits round wells, and the 384-well plate's are square: "
        "register it with plate-registration"
    )
    # The plate page says the same words beside the button and refuses by them (m5
    # §12.3, #375): whichever side changes its sentence, this fails.
    assert refused.reason == walls_square_wells("384-well")


@pytest.mark.parametrize(
    ("refs", "reason"),
    [
        (
            (WellRef(well="A1", x_um=0.0, y_um=0.0),),
            "give at least two wells in refs, each with the stage position where "
            "it was centred by eye",
        ),
        (
            tuple(
                WellRef(well=f"A{i}", x_um=9000.0 * i, y_um=0.0) for i in range(1, 8)
            ),
            "give at most 6 wells in refs, got 7",
        ),
        (
            (
                WellRef(well="A1", x_um=0.0, y_um=0.0),
                WellRef(well="I13", x_um=9000.0, y_um=0.0),
            ),
            "I13 is not a well of the 96-well plate",
        ),
        (
            (
                WellRef(well="a1", x_um=0.0, y_um=0.0),
                WellRef(well="A01", x_um=9000.0, y_um=0.0),
            ),
            "A1 is given twice in refs",
        ),
        (
            (
                WellRef(well="A1", x_um=0.0, y_um=0.0),
                WellRef(well="A2", x_um=9.0e9, y_um=0.0),
            ),
            "A2's recorded position (9e+09, 0) µm is beyond ±1e+06 µm: is it in µm?",
        ),
        (
            (
                WellRef(well="A1", x_um=0.0, y_um=0.0),
                WellRef(well="A2", x_um=0.0, y_um=-9.0e9),
            ),
            "A2's recorded position (0, -9e+09) µm is beyond ±1e+06 µm: is it in µm?",
        ),
    ],
    ids=["one", "seven", "off the plate", "twice", "a unit slip", "a unit slip in y"],
)
def test_the_references_are_checked_before_anything_moves(
    fake_microscope: Microscope,
    tmp_path: Path,
    refs: tuple[WellRef, ...],
    reason: str,
) -> None:
    camera = _on_fake(fake_microscope)
    stage = _Stage(fake_microscope.require(XYStage))
    fake_microscope.override(XYStage, stage)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    refused = _refused(fake_microscope, run_doc, refs)

    assert refused.reason == reason
    assert stage.moves == 0
    assert camera.snaps == 0


@pytest.mark.parametrize(
    ("shape", "params", "reason"),
    [
        (
            (100, 120),
            {},
            "a frame of 100 \N{MULTIPLICATION SIGN} 120 px is too small to find a wall in",
        ),
        # Short on one axis only, each way (FM-73): the smaller side decides.
        (
            (100, 512),
            {},
            "a frame of 100 \N{MULTIPLICATION SIGN} 512 px is too small to find a wall in",
        ),
        (
            (512, 100),
            {},
            "a frame of 512 \N{MULTIPLICATION SIGN} 100 px is too small to find a wall in",
        ),
        (
            SHAPE,
            {"search_um": 4000.0},
            "search_um 4000 is more than the well's radius 3200 µm",
        ),
    ],
    ids=[
        "small frame",
        "short in rows",
        "short in columns",
        "search beyond the radius",
    ],
)
def test_the_frame_and_the_search_are_checked_before_anything_moves(
    fake_microscope: Microscope,
    tmp_path: Path,
    shape: tuple[int, int],
    params: dict[str, Any],
    reason: str,
) -> None:
    camera = _on_fake(fake_microscope, shape=shape)
    stage = _Stage(fake_microscope.require(XYStage))
    fake_microscope.override(XYStage, stage)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope, shape=shape)

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"), **params)

    assert refused.reason == reason
    assert stage.moves == 0
    assert camera.snaps == 0


@pytest.mark.parametrize(
    ("shape", "field_um"),
    [((1412, 1412), 7060), ((200, 2000), 10000), ((2000, 200), 10000)],
    ids=["square", "long in columns", "long in rows"],
)
def test_a_field_that_holds_a_whole_well_is_refused(
    fake_microscope: Microscope,
    tmp_path: Path,
    shape: tuple[int, int],
    field_um: int,
) -> None:
    """At 5 µm per pixel a 1412-px frame is a 7 mm field; a 200 x 2000 one is 1 mm by 10 mm: the larger extent is wider than a 6.4 mm well."""
    camera = _on_fake(fake_microscope, shape=shape)
    stage = _Stage(fake_microscope.require(XYStage))
    fake_microscope.override(XYStage, stage)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope, shape=shape, matrix=_matrix(5.0, 0.0, False))

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"))

    assert refused.reason == (
        f"the field of {field_um} µm holds a whole well of 6400 µm: use a higher "
        "magnification"
    )
    assert stage.moves == 0
    assert camera.snaps == 0


def test_a_walk_of_too_many_frames_is_refused_before_anything_moves(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Six wells at the widest search: 36 offsets and the centred frame, four sides, six wells: 864 > 600 (FM-74)."""
    camera = _on_fake(fake_microscope)
    stage = _Stage(fake_microscope.require(XYStage))
    fake_microscope.override(XYStage, stage)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    refs = tuple(
        WellRef(well=f"{row}1", x_um=0.0, y_um=-9000.0 * i)
        for i, row in enumerate("ABCDEF")
    )

    refused = _refused(fake_microscope, run_doc, refs, search_um=3200.0)

    walk = 1 + 2 * math.floor(3200.0 / STEP_UM)
    assert refused.reason == (
        f"up to {6 * 4 * (walk + 1)} frames: use a lower magnification or a "
        "smaller search_um"
    )
    assert stage.moves == 0
    assert camera.snaps == 0


# ---------------------------------------------------- what a frame can be wrong


def test_a_frame_of_another_shape_than_the_calibrations_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    def binned(snap: int, frame: npt.NDArray[np.uint16]) -> npt.NDArray[Any]:
        return frame[::2, ::2] if snap == 2 else frame

    camera = _on_fake(fake_microscope, spoil=binned)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"))

    assert refused.reason == (
        "the frame has shape (192, 256), the calibration (384, 512): check the "
        "camera's binning and region"
    )
    assert camera.snaps == 2


def test_a_wall_lost_in_the_frame_centred_on_it_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A recorded centre that is exact finds the wall at the first frame; the second is the centred one."""

    def blank(snap: int, frame: npt.NDArray[np.uint16]) -> npt.NDArray[Any]:
        return np.full_like(frame, 2000) if snap == 2 else frame

    camera = _on_fake(fake_microscope, spoil=blank)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    truth = _truth("A1")
    refs = (
        WellRef(well="A1", x_um=truth[0], y_um=truth[1]),
        _refs("H12")[0],
    )

    refused = _refused(fake_microscope, run_doc, refs)

    x, y = camera.positions[1]
    assert refused.reason == (
        f"the wall of A1's +X side was lost in the frame centred on it at "
        f"({x:.1f}, {y:.1f}) µm"
    )
    assert camera.snaps == 2


def test_a_readback_that_is_not_a_position_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    stage = _Stage(fake_microscope.require(XYStage), nan_readback_at=2)
    fake_microscope.override(XYStage, stage)

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"))

    assert refused.reason == "the stage's readback is not a finite position"
    assert camera.snaps == 1


def test_a_well_of_another_size_is_refused_and_named(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The sample's wells are 4.8 mm and the plate is called a 96-well plate: 25 % under 6.4 mm."""
    sample = PlateSample(
        useq.WellPlate(
            name="narrow-wells",
            rows=8,
            columns=12,
            well_spacing=(9.0, 9.0),
            well_size=(4.8, 4.8),
            circular_wells=True,
        ),
        (0.0, 0.0),
        PLATE_DEG,
        seed=11,
    )
    camera = _on_fake(fake_microscope, sample)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    # Recorded exactly, and the wall is 800 µm in from where the 6.4 mm well puts it.
    refs = (
        WellRef(well="A1", x_um=_truth("A1")[0], y_um=_truth("A1")[1]),
        WellRef(well="H12", x_um=_truth("H12")[0], y_um=_truth("H12")[1]),
    )

    refused = _refused(fake_microscope, run_doc, refs)

    assert refused.reason.startswith(
        "A1 measures 4800 µm across and the 96-well plate's wells 6400 µm: is it the right plate type?"
    )
    assert "walls were imaged at +X (" in refused.reason
    assert camera.snaps > 0


def test_a_frame_that_cannot_be_saved_does_not_fail_the_registration(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    # A file where the folder should be: every write fails, whatever the OS.
    (tmp_path / "plate-walls").write_text("in the way", encoding="utf-8")

    result = _result(_run(stand, run_doc, _refs("A1", "H12")))

    assert result.rotation_deg == pytest.approx(PLATE_DEG, abs=0.01)
    assert result.walls is not None
    assert result.walls.folder is not None
    assert [w.file for f in result.walls.wells for w in f.walls] == [None] * 8
    saving = [line for line in result.assumed if "could not be saved" in line]
    assert len(saving) == 1
    assert saving[0].startswith("8 frames could not be saved: ")
    # The line says what the OS said and never where (FM-50).
    assert str(tmp_path) not in saving[0]
    assert "\n" not in saving[0]


def test_frames_are_not_saved_when_asked_not_to(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    result = _result(
        _run(fake_microscope, run_doc, _refs("A1", "H12"), save_frames=False)
    )

    assert result.walls is not None
    assert result.walls.folder is None
    assert [w.file for f in result.walls.wells for w in f.walls] == [None] * 8
    assert not (tmp_path / "plate-walls").exists()
    assert [line for line in result.assumed if "could not be saved" in line] == []


def test_the_soft_limits_of_the_layer_refuse_a_move_and_name_it(
    open_limited: Callable[..., Microscope], tmp_path: Path
) -> None:
    """The demo has none, so the stand is opened with them: the layer's own refusal reaches the plugin."""
    stand = open_limited(((-500.0, 500.0), (-500.0, 500.0)))
    stand.core.setProperty("XY", "Velocity", "100000")
    sample = _plate()
    camera = _Camera(stand, sample)
    stand.override(Camera, camera)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, stand)
    refs = (
        WellRef(well="A1", x_um=0.0, y_um=0.0),
        WellRef(well="A2", x_um=100.0, y_um=0.0),
    )

    refused = _refused(stand, run_doc, refs)

    # The check before anything moves is the same test the layer makes: the
    # walk of A1 reaches +3200 µm in x, past 500.
    assert refused.reason == (
        "A1's +X wall would be looked for at (3200.0, 0.0) µm, outside the stage's "
        "soft limits"
    )
    assert camera.snaps == 0


def test_walls_that_make_no_circle_are_refused_and_named(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The X walls are seen 90 µm further out than they are and the Y walls 90 µm further in: four arcs and no circle."""
    sample = _plate()
    camera: _Camera | None = None
    #: Where each centred frame is really rendered, from where the stage is:
    #: A1 is recorded exactly, so each side's walk ends at its first frame and
    #: the centred frame is the second, snaps 2 (+X), 4 (+Y), 6 (-X) and 8 (-Y).
    #: The frame rendered 90 µm in from the wall shows it 90 µm further out of
    #: the stage's position than it is; rendered 90 µm out, 90 µm further in.
    shifts = {2: (-90.0, 0.0), 4: (0.0, 90.0), 6: (90.0, 0.0), 8: (0.0, -90.0)}

    def oval(snap: int, frame: npt.NDArray[np.uint16]) -> npt.NDArray[Any]:
        if snap not in shifts:
            return frame
        assert camera is not None
        x, y = camera.positions[-1]
        dx, dy = shifts[snap]
        return sample.render(
            x + dx,
            y + dy,
            0.0,
            shape=SHAPE,
            pixel_size_um=PIXEL_UM,
            camera_rotation_deg=CAMERA_DEG,
            mirrored=MIRRORED,
        )

    camera = _on_fake(fake_microscope, sample, spoil=oval)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    truth = _truth("A1")
    refs = (WellRef(well="A1", x_um=truth[0], y_um=truth[1]), _refs("H12")[0])

    refused = _refused(fake_microscope, run_doc, refs)

    # Four tight arcs that no circle passes through: the trim of the fit keeps
    # none of them, which is the fit's own refusal (the oval is 90 µm from the
    # circle, and the MAD of the points is a fraction of a µm).
    assert refused.reason.startswith("A1's walls do not fit a circle: only 0 of ")
    assert refused.reason.endswith(" points lie near a circle")
    assert camera.snaps == 8


def test_a_circle_that_does_not_hold_is_refused_by_its_rms_and_named(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound of 0.02 µm on the circle's rms: the well's own pixel noise is over it, and the first well is refused."""
    _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    monkeypatch.setattr("smc.plugins.plate_walls.loop.MAX_RMS_FRACTION", 0.02 / 3200)

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"))

    assert refused.reason.startswith("A1's wall points lie ")
    rms = float(refused.reason.split("lie ")[1].split(" µm")[0])
    assert 0.02 < rms < 3.0
    assert (
        " µm from a circle on average, more than 0.02 µm: a frame may show something "
        "else than the wall (a bubble, a scratch, a label); look at its frames, "
        "imaged at +X ("
    ) in refused.reason
    assert refused.reason.endswith(" µm")


def test_a_turret_at_an_empty_position_is_no_objective(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-125: the label ``""`` is no objective, and is not one called nothing that a calibration can be of."""
    camera = _on_fake(fake_microscope)
    stage = _Stage(fake_microscope.require(XYStage))
    fake_microscope.override(XYStage, stage)
    fake_microscope.override(ObjectiveTurret, _EmptyTurret())
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope, objective="")

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"))

    assert refused.reason == (
        "no objective is in place: the turret is at an empty position; turn it to "
        "an objective and run camera-calibration there"
    )
    assert stage.moves == 0
    assert camera.snaps == 0


def test_a_plate_turned_the_other_way_is_refused_by_the_rotation_bound(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The Ti2-E's open question (#327): columns towards -X fit a rotation near 180°, which is refused.

    The wells are imaged as they are: the walls are found and the centres fitted.
    It is the plate that is refused, with the fit's own words, and the stage is
    left in the last well. The words say that the plate lies turned, or two wells
    are swapped (m3 §24.4 (1)).
    """
    sample = _plate(rotation_deg=179.0)
    camera = _on_fake(fake_microscope, sample)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12", plate_deg=179.0))

    assert refused.reason == (
        "the fit turns the plate by 179.0°: it lies turned 180° from the declared "
        "orientation (not-turned), or two reference wells are swapped; if A1 is at "
        "the other corner, declare orientation=turned-180, else check the well names"
    )
    assert _here(fake_microscope) == pytest.approx(camera.positions[-1], abs=0.1)
    assert _stored(run_doc).status is StageStatus.failed


def test_a_plate_declared_turned_is_registered_from_its_walls(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The same plate declared ``turned-180``: registered, its wells named from the plate (m3 §24 D3)."""
    _on_fake(fake_microscope, _plate(rotation_deg=179.0))
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    result = _result(
        _run(
            fake_microscope,
            run_doc,
            _refs("A1", "H12", plate_deg=179.0),
            orientation="turned-180",
        )
    )

    assert result.orientation == "turned-180"
    assert result.rotation_deg == pytest.approx(179.0, abs=0.1)
    assert [r.well for r in result.refs] == ["A1", "H12"]
    assert result.walls is not None
    assert [w.well for w in result.walls.wells] == ["A1", "H12"]
    assert result.well_center_um("H12") == pytest.approx(
        _truth("H12", rotation_deg=179.0), abs=5.0
    )


def test_a_narrower_search_is_the_one_used_and_reported(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    result = _result(
        _run(fake_microscope, run_doc, _refs("A1", "H12"), search_um=500.0)
    )

    assert result.walls is not None
    assert result.walls.search_um == 500.0
    for ref in result.refs:
        assert math.dist((ref.x_um, ref.y_um), _truth(ref.well)) < 1.0


def test_a_settle_time_is_waited_after_every_frames_move(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    waited: list[float] = []
    monkeypatch.setattr(RunContext, "wait_s", lambda _self, s: waited.append(s))

    _result(_run(fake_microscope, run_doc, _refs("A1", "H12"), settle_ms=250.0))

    # One wait before each frame, of the parameter in seconds.
    assert waited == [0.25] * camera.snaps


def test_the_warning_survives_a_stage_that_cannot_be_read(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A read that fails must not hide the error on its way: the stop is raised, and the line says the position is unknown."""
    _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    stage = _Stage(
        fake_microscope.require(XYStage),
        raises={3: MotionStoppedError("xy_stage XY")},
        unreadable_after_fault=True,
    )
    fake_microscope.override(XYStage, stage)

    with (
        caplog.at_level(logging.WARNING, logger=LOGGER),
        pytest.raises(MotionStoppedError),
    ):
        _run(fake_microscope, run_doc, _refs("A1", "H12"))

    (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert line.startswith(
        "plate-walls stopped at A1's +X wall: the stage is left at an unknown position"
    )


# ------------------------------------------- the operator's parameters reach the code


def test_a_search_narrower_than_the_error_finds_no_wall_and_says_so(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A1 is recorded 300 µm off and the search is 200: its walk is 3 frames (the nominal wall and a step either side), and the wall is not in one of them."""
    camera = _on_fake(fake_microscope)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)

    refused = _refused(fake_microscope, run_doc, _refs("A1", "H12"), search_um=200.0)

    assert refused.reason == (
        "no wall found on A1's +X side in 3 frames, from 3000 to 3400 µm out of the "
        "recorded centre (300.0, -200.0) µm: check that the light is on (the inside "
        "of a well is brighter than the plastic in brightfield) and that the stage "
        "was in A1 when it was recorded, then record it again"
    )
    assert camera.snaps == 3


def test_a_wall_seen_beyond_the_search_in_the_centred_frame_is_lost(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The centred frame shows the wall 90 µm further out than the walk's did, and the search is 60: it is not the wall that was found."""
    sample = _plate()
    camera: _Camera | None = None

    def further_out(snap: int, frame: npt.NDArray[np.uint16]) -> npt.NDArray[Any]:
        # A1 is recorded exactly and the search (60 µm) is under a step, so the walk
        # is its first frame alone and the centred frame is snap 2: rendered 90 µm
        # in from the wall, which then appears 90 µm further out of the stage's
        # position than it is.
        if snap != 2:
            return frame
        assert camera is not None
        x, y = camera.positions[-1]
        return sample.render(
            x - 90.0,
            y,
            0.0,
            shape=SHAPE,
            pixel_size_um=PIXEL_UM,
            camera_rotation_deg=CAMERA_DEG,
            mirrored=MIRRORED,
        )

    camera = _on_fake(fake_microscope, sample, spoil=further_out)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    truth = _truth("A1")
    refs = (WellRef(well="A1", x_um=truth[0], y_um=truth[1]), _refs("H12")[0])

    refused = _refused(fake_microscope, run_doc, refs, search_um=60.0)

    x, y = camera.positions[1]
    assert refused.reason == (
        f"the wall of A1's +X side was lost in the frame centred on it at "
        f"({x:.1f}, {y:.1f}) µm"
    )
    assert camera.snaps == 2


def test_the_residual_bound_is_the_parameter_and_the_residual_is_the_worst_well(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A plate of pitch 9.03 mm fitted as 9.0, three wells: 231 µm from where the others put it (§17.1), over 150 and under 300."""
    _on_fake(fake_microscope, _plate(pitch_um=9030.0))
    refs = _refs("A1", "A12", "H1", pitch_um=9030.0)
    below = tmp_path / "below.json"
    _seed(below, fake_microscope)

    refused = _refused(fake_microscope, below, refs, max_residual_um=150.0)

    assert refused.reason.endswith(
        " µm from where the other wells put it (bound 150 µm): record it again, "
        "or check the plate type"
    )
    shown = float(refused.reason.split(" is ")[1].split(" µm")[0])
    assert shown == pytest.approx(231.0, abs=6.0)

    above = tmp_path / "above.json"
    _seed(above, fake_microscope)

    result = _result(_run(fake_microscope, above, refs, max_residual_um=300.0))

    # The worst of the three, 231 µm, is what the result holds, and the scale
    # error is the 0.33 % of the plate's pitch, under the tolerance of 2 %.
    assert result.residual_um == pytest.approx(231.0, abs=6.0)
    assert result.scale_error == pytest.approx(0.0033, abs=3e-4)


def test_a_wider_rotation_bound_registers_a_plate_turned_30_degrees(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The default bound is 10°; a plate really turned by 30° is registered when the operator says so (up to 45°)."""
    _on_fake(fake_microscope, _plate(rotation_deg=30.0))
    refs = _refs("A1", "H12", plate_deg=30.0)
    narrow = tmp_path / "narrow.json"
    _seed(narrow, fake_microscope)

    refused = _refused(fake_microscope, narrow, refs)

    assert refused.reason == (
        "the fit turns the plate by 30.0°, beyond ±10°: a reference well is "
        "misidentified (two references the same distance apart fit either way)"
    )

    wide = tmp_path / "wide.json"
    _seed(wide, fake_microscope)

    result = _result(_run(fake_microscope, wide, refs, max_rotation_deg=45.0))

    assert result.rotation_deg == pytest.approx(30.0, abs=0.01)


def test_a_well_a_little_smaller_than_the_plate_says_is_measured_and_reported(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Wells of 6.0 mm on a plate that is called 6.4: 6.25 % under, which the tolerance of 10 % accepts."""
    sample = PlateSample(
        useq.WellPlate(
            name="small-wells",
            rows=8,
            columns=12,
            well_spacing=(9.0, 9.0),
            well_size=(6.0, 6.0),
            circular_wells=True,
        ),
        (0.0, 0.0),
        PLATE_DEG,
        seed=11,
    )
    _on_fake(fake_microscope, sample)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    refs = tuple(
        WellRef(well=well, x_um=_truth(well)[0], y_um=_truth(well)[1])
        for well in ("A1", "H12")
    )

    result = _result(_run(fake_microscope, run_doc, refs))

    assert result.walls is not None
    for fitted in result.walls.wells:
        assert fitted.diameter_um == pytest.approx(6000.0, abs=2.0)
        # The diameter over the plate's, minus 1: under, so negative.
        assert fitted.size_error == pytest.approx(6000.0 / 6400.0 - 1.0, abs=1e-3)
        assert math.dist(fitted.center_xy_um, _truth(fitted.well)) < 1.0


def test_the_start_is_read_after_the_stage_has_arrived(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-85: a stage still on its way from an earlier move reads a fraction of it; the start is read after ``wait()``."""
    stop = threading.Event()
    _on_fake(
        fake_microscope,
        before=lambda n: stop.set() if n == 3 else None,
    )
    run_doc = tmp_path / "run.json"
    _seed(run_doc, fake_microscope)
    stage = _Stage(fake_microscope.require(XYStage), travelling=(12345.0, -6789.0))
    fake_microscope.override(XYStage, stage)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(fake_microscope, run_doc, _refs("A1", "H12"), stop=stop)

    assert record.status is StageStatus.stopped
    (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert line.endswith("not returned to (0.0, 0.0) µm where it started")


# ------------------------- PFS across a well's wall: the switch-off (m3 §5.7, #478)

AUTOFOCUS_OFF_LINE = (
    "the autofocus was switched off before anything moved, so that no lock crossed "
    "a well's wall: engage it again on the sample"
)
AUTOFOCUS_OFF_WARNING = (
    "plate-walls: the autofocus was switched off before anything moved, so that no "
    "lock crosses a well's wall (m2 §12.2 D2): engage it again after the run"
)
#: What a registration of two wells says today, with an autofocus that is off.
TODAY_ASSUMED = (
    "the stage reports no soft limits, so the frames' moves were not checked "
    "against any travel",
    "two wells: the residual checks their distance only; a third well checks the fit",
)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The plugin's own warnings: ``override`` warns too, on the layer's logger."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == LOGGER
    ]


def test_plate_walls_switches_an_engaged_autofocus_off_before_anything_moves_and_says_so_first(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True)
    stand.override(Autofocus, af)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(stand, run_doc, _refs("A1", "H12")))

    # The whole list: one switch-off for the run, however many walls and wells.
    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    assert af.engaged is False
    assert result.assumed == (AUTOFOCUS_OFF_LINE, *TODAY_ASSUMED)
    assert _warnings(caplog) == [AUTOFOCUS_OFF_WARNING]


def test_plate_walls_switches_the_autofocus_off_before_its_first_move(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order: the switch-off and its read back, then the first move of the walk."""
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True)
    stand.override(Autofocus, af)
    stage = stand.require(XYStage)
    real_move = stage.move_to_um

    def move(x_um: float, y_um: float) -> XY:
        af.calls.append("move_to_um")
        return real_move(x_um, y_um)

    monkeypatch.setattr(stage, "move_to_um", move)

    _result(_run(stand, run_doc, _refs("A1", "H12")))

    assert af.calls[:4] == ["is_engaged", "disengage", "is_engaged", "move_to_um"]
    assert af.calls.count("disengage") == 1


def test_plate_walls_leaves_an_autofocus_that_is_off_alone(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=False)
    stand.override(Autofocus, af)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(stand, run_doc, _refs("A1", "H12")))

    assert af.calls == ["is_engaged"]
    assert result.assumed == TODAY_ASSUMED
    assert _warnings(caplog) == []


def test_plate_walls_switches_a_searching_autofocus_off_too(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """PFS on and searching drives Z the moment it finds its reference (m2 §12.11 D9)."""
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True, locked=False)
    stand.override(Autofocus, af)

    result = _result(_run(stand, run_doc, _refs("A1", "H12")))

    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    assert result.assumed[0] == AUTOFOCUS_OFF_LINE


def test_a_stop_before_plate_walls_moves_keeps_the_autofocus_on(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True)
    stand.override(Autofocus, af)
    before = _here(stand)
    stop = threading.Event()
    stop.set()

    record = _run(stand, run_doc, _refs("A1", "H12"), stop=stop)

    assert record.status is StageStatus.stopped
    assert "disengage" not in af.calls
    assert af.engaged is True
    assert _here(stand) == before
    assert camera.snaps == 0


_NOT_SWITCHED_OFF = (
    "the autofocus could not be switched off before anything moved ({cause}): "
    "nothing moved, and PFS may still be on: switch it off at the stand, then run "
    "plate-walls again"
)
_STILL_ON = (
    "the autofocus still reads on after it was switched off: nothing moved: switch "
    "it off at the stand, then run plate-walls again"
)


@pytest.mark.parametrize(
    ("options", "reason", "calls"),
    [
        (
            {"fails": "disengage"},
            _NOT_SWITCHED_OFF.format(cause="CoreError: disengage failed on the fake"),
            ["is_engaged", "disengage"],
        ),
        (
            {"fails": "is_engaged"},
            _NOT_SWITCHED_OFF.format(cause="CoreError: is_engaged failed on the fake"),
            ["is_engaged"],
        ),
        (
            {"read_fails_after_off": True},
            _NOT_SWITCHED_OFF.format(
                cause="CoreError: is_engaged failed on the fake after the switch-off"
            ),
            ["is_engaged", "disengage", "is_engaged"],
        ),
        ({"stays_on": True}, _STILL_ON, ["is_engaged", "disengage", "is_engaged"]),
    ],
    ids=["disengage-raises", "read-raises", "read-back-raises", "still-on"],
)
def test_a_switch_off_that_fails_ends_plate_walls_before_anything_moves(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    options: dict[str, Any],
    reason: str,
    calls: list[str],
) -> None:
    """Unknown counts as on (FM-19): the run fails with §5.7's text whole, and nothing moved."""
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True, **options)
    stand.override(Autofocus, af)
    before = _here(stand)

    refused = _refused(stand, run_doc, _refs("A1", "H12"))

    assert refused.reason == reason
    assert af.calls == calls
    assert _here(stand) == before
    assert camera.snaps == 0
    stored = _stored(run_doc)
    assert stored.status is StageStatus.failed
    assert stored.error == f"PluginRunError: plate-walls: {reason}"


def test_a_switch_off_error_of_several_lines_is_said_by_its_first_in_plate_walls(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A vendor's message can run over lines: the run's text keeps one.
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True)

    def fails() -> None:
        raise CoreError("PFS did not answer\nError in device PFS: 10")

    monkeypatch.setattr(af, "disengage", fails)
    stand.override(Autofocus, af)

    refused = _refused(stand, run_doc, _refs("A1", "H12"))

    assert refused.reason == _NOT_SWITCHED_OFF.format(
        cause="CoreError: PFS did not answer"
    )


@pytest.mark.parametrize(
    "fault",
    [MicroscopeHaltedError(), MotionStoppedError("the stage was stopped")],
    ids=["halt", "motion-stopped"],
)
def test_a_halt_from_the_switch_off_reaches_the_caller_unchanged_in_plate_walls(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    """A halt or a stop is not a switch-off that failed: it passes first (FM-64)."""
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True)

    def fails() -> None:
        raise fault

    monkeypatch.setattr(af, "disengage", fails)
    stand.override(Autofocus, af)
    before = _here(stand)

    with pytest.raises(type(fault)) as info:
        _run(stand, run_doc, _refs("A1", "H12"))

    assert info.value is fault
    assert _here(stand) == before
    assert camera.snaps == 0


def test_a_halted_stand_switches_the_autofocus_off_and_plate_walls_ends_on_the_halt(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """PFS off is the safe side; the first move then raises the halt (FM-64).

    The stand's stop switches the autofocus off itself; one switched on again
    while halted (at the stand) is what the tool's own switch-off meets.
    """
    stand, camera = _on_demo(demo_microscope_with_sample)
    run_doc = tmp_path / "run.json"
    _calibrate(stand, camera, run_doc)
    af = RecordingAutofocus(engaged=True)
    stand.override(Autofocus, af)
    before = _here(stand)
    stand.stop()
    assert af.engaged is False  # the stop's own switch-off
    af.engaged = True
    af.calls.clear()
    try:
        with pytest.raises(MicroscopeHaltedError):
            _run(stand, run_doc, _refs("A1", "H12"))
    finally:
        stand.resume()

    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    assert _stored(run_doc).status is StageStatus.failed
    assert _here(stand) == before
    assert camera.snaps == 0
