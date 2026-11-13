"""``objective-offsets`` on the demo-sample stand: a declared shift per objective, recovered from the plate (m3 §20, #380).

The stand is ``profiles/demo-sample.toml`` with its ``[sample]`` turned as §20.1
did: the camera turned 30° and mirrored, the plate turned 1.2°, and a shift per
objective (10X ``(5.0, -2.5)``, 20X ``(0, 0)``, 40X ``(-32.5, 9.75)``) under
which a feature is centred at its position plus the shift (§20.8). The 10X is in
place at open, so it is the objective the plate is registered with, and the
offsets to recover are written out from the shifts: 20X from 10X ``(-5.0,
2.5)``, 40X from 10X ``(-37.5, 12.25)``, never read from the code (FM-124,
FM-203).

A module fixture builds the documents once, through ``run_plugin`` (the seam):
``camera-calibration`` under the 20X and under the 40X, one document each, and
``camera-calibration`` and ``plate-walls`` (A1, H12 and A12, recorded 300 µm off)
under the 10X into ``run.json``. Each test copies ``run.json`` and opens a stand
of its own. The demo stage moves at its ``Velocity`` property, 10 mm/s by
default: the stands here raise it to 100 000 on their own core, as ``plate-walls``'
tests do.

The doubles write out every method of the capability they stand for: the
protocol check refuses a ``__getattr__`` delegate (§20.1). Positions are compared
with the stage's readback and never with the target (FM-142).
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

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
    ZStage,
)
from smc.hardware.errors import (
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.profile import ObjectivesSection, Profile, SampleSection
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.objective_offsets.plugin import (
    ObjectiveOffsetsParams,
    ObjectiveOffsetsPlugin,
)
from smc.plugins.plate_registration.plugin import (
    PlateRegistrationParams,
    PlateRegistrationPlugin,
)
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.slots import ObjectiveOffsetsResult, WellRef
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.sample_stand import sample_refs

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The shift of each objective, declared (§20.8, measured in §20.1): a feature is
#: centred at its stage position plus this.
SHIFT = {TEN: (5.0, -2.5), TWENTY: (0.0, 0.0), FORTY: (-32.5, 9.75)}
PLATE_DEG = 1.2
#: The demo-sample's A1 (a 96-well plate centred on the origin) and its pitch.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
#: The wells' (column, row) indices, written out here and not read from useq.
INDEX = {"A1": (0, 0), "H12": (11, 7), "A12": (11, 0)}
#: Where the operator "centred" each well: up to 300 µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "H12": (-250.0, 150.0), "A12": (-100.0, 200.0)}
LOGGER = "smc.plugins.objective-offsets"
NAME = "objective-offsets"


def _truth(well: str) -> tuple[float, float]:
    """A well's true centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    col, row = INDEX[well]
    angle = math.radians(PLATE_DEG)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        A1_UM[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_UM[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _centred(well: str, objective: str) -> tuple[float, float]:
    """Where ``objective`` centres a well: its true centre plus the objective's shift."""
    x, y = _truth(well)
    return (x + SHIFT[objective][0], y + SHIFT[objective][1])


def _profile(*, reference: str = "") -> Profile:
    """``demo-sample`` turned as §20.1 turned it, with the shifts declared; the profile's reference when given."""
    base = Profile.load(DEMO_SAMPLE)
    update: dict[str, Any] = {
        "sample": SampleSection(
            rotation_deg=PLATE_DEG,
            camera_rotation_deg=30.0,
            mirrored=True,
            objective_shift_um=SHIFT,
        )
    }
    if reference:
        update["objectives"] = ObjectivesSection(reference=reference)
    return base.model_copy(update=update)


# ------------------------------------------------------------------ the doubles


class _Turret:
    """The stand's turret, counting the turns; each method is written out."""

    def __init__(self, inner: ObjectiveTurret) -> None:
        self._inner = inner
        self.turns: list[str] = []

    def labels(self) -> list[str]:
        return self._inner.labels()

    def current(self) -> str:
        return self._inner.current()

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.turns.append(label)
        return self._inner.select(label, confirm=confirm)

    def pixel_size_um(self, label: str | None = None) -> float:
        return self._inner.pixel_size_um(label)

    def offset(self, label: str) -> ObjectiveOffset:
        return self._inner.offset(label)


class _LyingTurret(_Turret):
    """A turret that turns and then reads the objective it was at: a label that lags (m2 §8.11 D3)."""

    def __init__(self, inner: ObjectiveTurret) -> None:
        super().__init__(inner)
        self._was = inner.current()
        self._turned = False

    def select(self, label: str, *, confirm: bool = False) -> str:
        result = super().select(label, confirm=confirm)
        self._turned = True
        return result

    def current(self) -> str:
        return self._was if self._turned else super().current()


class _DriftingTurret(_Turret):
    """A turret whose ``select`` calls the real one and only then sets Z drifting.

    The layer's own wait for a still Z (m2 §8.11 D2) has returned by then, so
    the layer sees a still Z and the plugin a moving one (§20.1).
    """

    def __init__(self, inner: ObjectiveTurret, z: _Z) -> None:
        super().__init__(inner)
        self._z = z

    def select(self, label: str, *, confirm: bool = False) -> str:
        result = super().select(label, confirm=confirm)
        self._z.drift()
        return result


class _RefusingTurret(_Turret):
    """A turret whose layer refuses the turn, or a stand that was halted."""

    def __init__(self, inner: ObjectiveTurret, raises: BaseException) -> None:
        super().__init__(inner)
        self._raises = raises

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.turns.append(label)
        raise self._raises


class _UnreadableTurret(_Turret):
    """A turret that answers until ``stop`` is set, and then cannot be read.

    The sample camera reads the turret at every snap (the shift of the objective
    in place, §20.8), so the turret has to answer through the run and fail only
    when the run is already ending.
    """

    def __init__(self, inner: ObjectiveTurret, stop: threading.Event) -> None:
        super().__init__(inner)
        self._stop = stop

    def current(self) -> str:
        if self._stop.is_set():
            raise HardwareError("the turret does not answer")
        return super().current()


class _Z:
    """The stand's focus drive, drifting 1 µm per read once ``drift()`` is called; each method is written out."""

    def __init__(self, inner: ZStage) -> None:
        self._inner = inner
        self._drifting = False
        self._extra_um = 0.0

    def drift(self) -> None:
        self._drifting = True

    def position_um(self) -> float:
        if self._drifting:
            self._extra_um += 1.0
        return self._inner.position_um() + self._extra_um

    def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float:
        return self._inner.move_to_um(z_um, keep_autofocus=keep_autofocus)

    def move_by_um(
        self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
    ) -> float:
        return self._inner.move_by_um(dz_um, force=force, keep_autofocus=keep_autofocus)

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> Limits | None:
        return self._inner.limits_um()


class _ClockZ(_Z):
    """A focus drive that, once ``drift()`` is called, creeps with the clock: 10 µm per second.

    ``_Z`` drifts per read, so it is refused however close together the two reads are.
    This one is refused only if they are ``Z_STILL_S`` apart: a plugin that drops the
    wait reads it twice in a few microseconds and sees it still (FM-195).
    """

    def __init__(self, inner: ZStage) -> None:
        super().__init__(inner)
        self._armed_at: float | None = None

    def drift(self) -> None:
        self._armed_at = time.monotonic()

    def position_um(self) -> float:
        if self._armed_at is None:
            return self._inner.position_um()
        return self._inner.position_um() + 10.0 * (time.monotonic() - self._armed_at)


class _ScriptedZ(_Z):
    """A focus drive that, once ``drift()`` is called, answers the reads of ``script`` in order, then the last one again."""

    def __init__(self, inner: ZStage, script: Sequence[float]) -> None:
        super().__init__(inner)
        self._script = list(script)
        self._reads = 0

    def position_um(self) -> float:
        if not self._drifting:
            return self._inner.position_um()
        value = self._script[min(self._reads, len(self._script) - 1)]
        self._reads += 1
        return value


class _Stage:
    """The stand's XY stage, counting the moves; each method is written out."""

    def __init__(self, inner: XYStage) -> None:
        self._inner = inner
        self.moves = 0

    def position_um(self) -> XY:
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        return self._inner.move_to_um(x_um, y_um)

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("objective-offsets makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._inner.limits_um()


class _Camera:
    """The stand's camera, counting the snaps and calling ``before(n)`` and ``after(n)`` around snap ``n``; each method is written out."""

    def __init__(
        self,
        inner: Camera,
        before: Callable[[int], None] | None = None,
        after: Callable[[int], None] | None = None,
    ) -> None:
        self._inner = inner
        self._before = before
        self._after = after
        self.snaps = 0

    def snap(self) -> npt.NDArray[np.generic]:
        self.snaps += 1
        if self._before is not None:
            self._before(self.snaps)
        frame = self._inner.snap()
        if self._after is not None:
            self._after(self.snaps)
        return frame

    def exposure_ms(self) -> float:
        return self._inner.exposure_ms()

    def set_exposure_ms(self, value_ms: float) -> float:
        return self._inner.set_exposure_ms(value_ms)

    def image_shape(self) -> tuple[int, int]:
        return self._inner.image_shape()

    def bit_depth(self) -> int:
        return self._inner.bit_depth()

    def pixel_size_um(self) -> float:
        return self._inner.pixel_size_um()


@dataclass
class _Rig:
    """A stand with the doubles that count what it does."""

    stand: Microscope
    turret: _Turret
    stage: _Stage
    camera: _Camera


# ------------------------------------------------------------------ the stands


@pytest.fixture
def open_stand(mm_available: bool) -> Iterator[Callable[..., Microscope]]:
    """Opens ``demo-sample`` with its shifts and a stage fast enough to walk; closes what it opened."""
    opened: list[Microscope] = []

    def make(profile: Profile | None = None, *, dry_run: bool = False) -> Microscope:
        if not mm_available:
            pytest.skip("Micro-Manager demo adapters not installed")
        stand = Microscope.open(
            profile if profile is not None else _profile(), dry_run=dry_run
        )
        opened.append(stand)
        # The demo stage moves at its `Velocity` property, 10 mm/s by default.
        stand.core.setProperty("XY", "Velocity", "100000")
        return stand

    yield make
    for stand in opened:
        stand.close()


def _rig(
    stand: Microscope,
    *,
    turret: Callable[[ObjectiveTurret], _Turret] = _Turret,
    before: Callable[[int], None] | None = None,
    after: Callable[[int], None] | None = None,
) -> _Rig:
    """Put the counting doubles on the stand; ``turret`` builds the one that stands for the turret."""
    turret_double = turret(stand.require(ObjectiveTurret))
    stage = _Stage(stand.require(XYStage))
    camera = _Camera(stand.require(Camera), before, after)
    stand.override(ObjectiveTurret, turret_double)
    stand.override(XYStage, stage)
    stand.override(Camera, camera)
    return _Rig(stand, turret_double, stage, camera)


@dataclass(frozen=True)
class Documents:
    """The run documents the tests start from."""

    run: Path
    twenty: Path
    forty: Path


@pytest.fixture(scope="module")
def documents(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Documents:
    """``camera-calibration`` under the 20X and the 40X, and the registration under the 10X, through ``run_plugin``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("offsets")
    docs = Documents(
        folder / "run.json", folder / "cal-20x.json", folder / "cal-40x.json"
    )
    stand = Microscope.open(_profile())
    try:
        stand.core.setProperty("XY", "Velocity", "100000")
        turret = stand.require(ObjectiveTurret)
        assert turret.current() == TEN
        for label, path in ((TWENTY, docs.twenty), (FORTY, docs.forty)):
            turret.select(label, confirm=True)
            record = run_plugin(
                CameraCalibrationPlugin(), stand, CameraCalibrationParams(), path
            )
            assert record.status is StageStatus.done
        turret.select(TEN, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.run
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
            docs.run,
        )
        assert record.status is StageStatus.done
    finally:
        stand.close()
    return docs


@pytest.fixture
def run_doc(documents: Documents, tmp_path: Path) -> Path:
    """A copy of the registration's ``run.json``: each test writes its own result."""
    copy = tmp_path / "run.json"
    shutil.copy(documents.run, copy)
    return copy


def _params(
    targets: tuple[str, ...], calibrations: tuple[Path, ...], **params: Any
) -> ObjectiveOffsetsParams:
    return ObjectiveOffsetsParams(
        **{
            "targets": targets,
            "calibrations": calibrations,
            "confirm_turns": True,
            **params,
        }
    )


def _run(
    stand: Microscope,
    run_doc: Path,
    targets: tuple[str, ...],
    calibrations: tuple[Path, ...],
    *,
    on_progress: Callable[[Progress], None] | None = None,
    stop: threading.Event | None = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        ObjectiveOffsetsPlugin(),
        stand,
        _params(targets, calibrations, **params),
        run_doc,
        on_progress=on_progress,
        stop=stop,
    )


def _result(record: StageRecord) -> ObjectiveOffsetsResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return ObjectiveOffsetsResult.model_validate(record.result)


def _refused(
    stand: Microscope,
    run_doc: Path,
    targets: tuple[str, ...],
    calibrations: tuple[Path, ...],
    **params: Any,
) -> PluginRunError:
    with pytest.raises(PluginRunError) as raised:
        _run(stand, run_doc, targets, calibrations, **params)
    assert raised.value.plugin == NAME
    # FM-98: a refusal is raised outside the handler of the error it stands for, so
    # that error is not on its traceback, where it would keep the core's frames alive.
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    return raised.value


def _stored(run_doc: Path) -> StageRecord:
    return RunDocument.load(run_doc).stages[NAME]


def _here(stand: Microscope) -> tuple[float, float]:
    position = stand.require(XYStage).position_um()
    return (position.x_um, position.y_um)


def _offsets(result: ObjectiveOffsetsResult) -> dict[str, tuple[float, float]]:
    return {o.objective: o.xy_offset_um for o in result.offsets}


# ---------------------------------------------------------------- registration


def test_the_entry_point_loads_objective_offsets() -> None:
    entry = load(NAME)

    assert entry.plugin.provides == "objective-offsets"
    assert entry.plugin.requires == (XYStage, Camera, ObjectiveTurret)
    assert entry.plugin.reads == ("plate",)
    assert entry.plugin.reads_optional == ()
    assert entry.target == "smc.plugins.objective_offsets.plugin:ObjectiveOffsetsPlugin"


def test_the_parameters_are_checked_before_the_stand_opens() -> None:
    with pytest.raises(ValueError, match="targets"):
        ObjectiveOffsetsParams(targets=(), calibrations=(Path("a.json"),))
    with pytest.raises(ValueError, match="targets"):
        ObjectiveOffsetsParams(
            targets=tuple(f"objective {i}" for i in range(6)),
            calibrations=(Path("a.json"),),
        )
    with pytest.raises(ValueError, match="search_um"):
        _params(("a",), (Path("a.json"),), search_um=1000.5)
    with pytest.raises(ValueError, match="max_spread_um"):
        _params(("a",), (Path("a.json"),), max_spread_um=100.5)
    with pytest.raises(ValueError, match="wells"):
        _params(("a",), (Path("a.json"),), wells=tuple("ABCDEFG"))
    # Nothing turns unless the operator says so.
    assert (
        ObjectiveOffsetsParams(
            targets=("a",), calibrations=(Path("a.json"),)
        ).confirm_turns
        is False
    )


# -------------------------------------------------------------------- recovered


def test_the_samples_offsets_are_recovered_within_a_pixel(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()

    record = _run(stand, run_doc, (FORTY, TWENTY), (documents.forty, documents.twenty))

    result = _result(record)
    assert result.plate_objective == TEN
    # The profile names no reference: the offsets are from the plate's objective.
    assert (result.reference, result.reference_in_profile) == (TEN, False)
    offsets = _offsets(result)
    assert list(offsets) == [FORTY, TWENTY]
    # One pixel of the 10X is 1.0 µm; the values are the declared shifts' differences
    # (-32.5 - 5.0, 9.75 + 2.5) and (0 - 5.0, 0 + 2.5), written out.
    assert offsets[FORTY] == pytest.approx((-37.5, 12.25), abs=1.0)
    assert offsets[TWENTY] == pytest.approx((-5.0, 2.5), abs=1.0)
    # Each target's wells agree, far inside the bound of 2 pixels of the 10X.
    assert [m.objective for m in result.measures] == [FORTY, TWENTY]
    for measure in result.measures:
        assert measure.spread_um < 0.5
        # Two pixels of the coarser objective, the 10X's as the calibration measured
        # it (0.9997 µm, not the nominal 1.0).
        assert measure.max_spread_um == pytest.approx(2.0, abs=0.01)
        assert [w.well for w in measure.wells] == ["A1", "H12", "A12"]
    # The whole result says what it rests on.
    assert any("no soft limits" in line for line in result.assumed)
    assert any("magnifier lever" in line for line in result.assumed)
    assert any("no z_offset_um is measured" in line for line in result.assumed)
    assert any("names no reference objective" in line for line in result.assumed)
    # What the offsets rest on and the run cannot tell: a plate that moved as a whole
    # since plate-walls registered it shifts every well alike, the wells agree, and the
    # shift reads as an offset between objectives.
    assert any("has not moved since plate-walls" in line for line in result.assumed)
    assert result.objective_in_place == TWENTY


def test_each_well_is_measured_under_the_target_where_the_sample_puts_it(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()

    result = _result(_run(stand, run_doc, (FORTY,), (documents.forty,)))

    (measure,) = result.measures
    for well in measure.wells:
        # The centre under the 40X is the true centre plus the 40X's shift, and the
        # origin is the one under the 10X: both from the declared geometry.
        assert well.fit.center_xy_um == pytest.approx(
            _centred(well.well, FORTY), abs=0.5
        )
        assert well.origin_xy_um == pytest.approx(_centred(well.well, TEN), abs=0.5)
        assert well.delta_xy_um == pytest.approx((-37.5, 12.25), abs=0.5)
        assert well.fit.recorded_xy_um == well.origin_xy_um
    assert measure.shift_xy_um == pytest.approx((-37.5, 12.25), abs=0.5)
    assert measure.pixel_size_um == pytest.approx(0.25, abs=0.01)
    assert measure.calibration_measured != ""
    # The calibration's path is the one typed (FM-24, FM-50): not resolved, not absolute
    # by the tool's doing.
    assert measure.calibration == documents.forty.as_posix()
    # The frames taken are the sides' own, counted: every side of every well took at
    # least its walk and its centred frame.
    assert measure.frames == sum(
        wall.frames for well in measure.wells for wall in well.fit.walls
    )
    assert measure.frames >= 2 * 4 * len(measure.wells)
    # Z reads where the demo left it, still.
    assert measure.z_after_turn_um == pytest.approx(0.0, abs=0.5)


def test_offsets_are_given_from_the_profiles_reference(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand(_profile(reference=TWENTY))

    # The calibrations are matched to the targets by their objective, not by place.
    record = _run(stand, run_doc, (FORTY, TWENTY), (documents.twenty, documents.forty))

    result = _result(record)
    assert (result.reference, result.reference_in_profile) == (TWENTY, True)
    offsets = _offsets(result)
    # From the 20X: the 10X sits at (5.0, -2.5) and the 40X at (-32.5, 9.75), the
    # declared shifts, since the 20X's own is zero.
    assert list(offsets) == [TEN, FORTY]
    assert offsets[TEN] == pytest.approx((5.0, -2.5), abs=1.0)
    assert offsets[FORTY] == pytest.approx((-32.5, 9.75), abs=1.0)
    assert not any("names no reference objective" in line for line in result.assumed)


def test_a_reference_that_is_the_plates_objective_gives_the_shifts(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand(_profile(reference=TEN))

    result = _result(_run(stand, run_doc, (TWENTY,), (documents.twenty,)))

    assert (result.reference, result.reference_in_profile) == (TEN, True)
    assert _offsets(result)[TWENTY] == pytest.approx((-5.0, 2.5), abs=1.0)


def test_a_larger_bound_lets_wells_that_disagree_through(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    _move_well_in_the_record(run_doc, "H12", (20.0, 0.0))
    stand = open_stand()

    result = _result(
        _run(stand, run_doc, (TWENTY,), (documents.twenty,), max_spread_um=25.0)
    )

    (measure,) = result.measures
    assert measure.max_spread_um == 25.0
    assert measure.spread_um == pytest.approx(20.0, abs=0.5)


# ------------------------------------------------- refused before anything turns


def _nothing_moved(rig: _Rig) -> None:
    """The turret was not turned, the stage not moved and no frame taken, and the objective in place is the 10X's."""
    assert rig.turret.turns == []
    assert rig.stage.moves == 0
    assert rig.camera.snaps == 0
    assert rig.turret.current() == TEN


def test_without_confirm_turns_nothing_turns(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    with pytest.raises(PluginRunError) as raised:
        run_plugin(
            ObjectiveOffsetsPlugin(),
            rig.stand,
            ObjectiveOffsetsParams(
                targets=(FORTY, TWENTY),
                calibrations=(documents.forty, documents.twenty),
            ),
            run_doc,
        )

    assert raised.value.reason == (
        f"objective-offsets turns the turret to {FORTY}, {TWENTY} with the plate on "
        "the stage: check that each objective clears the plate (its working "
        "distance, the plate's height), then give confirm_turns=true"
    )
    _nothing_moved(rig)
    assert _stored(run_doc).status is StageStatus.failed


def test_the_profile_that_asks_for_no_confirmation_is_still_asked_to_confirm(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # `turret_requires_confirm` decides what the layer refuses, not what this tool
    # asks before it turns several times over a plate (§20.6).
    base = _profile()
    profile = base.model_copy(
        update={
            "safety": base.safety.model_copy(update={"turret_requires_confirm": False})
        }
    )
    rig = _rig(open_stand(profile))

    with pytest.raises(PluginRunError, match="confirm_turns=true"):
        run_plugin(
            ObjectiveOffsetsPlugin(),
            rig.stand,
            ObjectiveOffsetsParams(targets=(FORTY,), calibrations=(documents.forty,)),
            run_doc,
        )

    _nothing_moved(rig)


def test_a_dry_run_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand(dry_run=True)

    refused = _refused(stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == (
        "a dry run cannot turn the turret or image the walls; run it without --dry-run"
    )


def test_a_plate_registered_by_eye_is_refused(
    open_stand: Callable[..., Microscope], tmp_path: Path, documents: Documents
) -> None:
    eye = tmp_path / "eye.json"
    section = _profile().sample
    assert section is not None
    stand = open_stand()
    run_plugin(
        PlateRegistrationPlugin(),
        stand,
        PlateRegistrationParams(plate="96-well", refs=sample_refs(section)),
        eye,
    )
    rig = _rig(stand)

    refused = _refused(rig.stand, eye, (FORTY,), (documents.forty,))

    assert refused.reason == (
        "the plate was registered with references: objective-offsets compares the "
        "centres plate-walls measured, so register it with plate-walls"
    )
    _nothing_moved(rig)


def test_a_target_that_is_not_on_the_turret_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, ("Nikon 99X",), (documents.forty,))

    assert refused.reason.startswith("objective Nikon 99X is not on the turret: ")
    # The labels of the turret are named, the empty positions are not.
    assert TEN in refused.reason
    assert FORTY in refused.reason
    _nothing_moved(rig)


def test_the_plates_own_objective_is_not_a_target(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (TEN,), (documents.forty,))

    assert refused.reason == (
        f"{TEN} is the objective the plate was registered with: its offset from "
        "itself is zero; leave it out of targets"
    )
    _nothing_moved(rig)


def test_a_target_given_twice_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY, FORTY), (documents.forty,))

    assert refused.reason == f"{FORTY} is given twice in targets"
    _nothing_moved(rig)


def test_a_well_the_registration_did_not_measure_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    refused = _refused(
        rig.stand, run_doc, (FORTY,), (documents.forty,), wells=("a1", "B2")
    )

    assert refused.reason == (
        "B2 is not a well the registration measured: choose among A1, H12, A12"
    )
    _nothing_moved(rig)


def test_one_well_is_not_enough_and_a_well_given_twice_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    one = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,), wells=("A1",))
    twice = _refused(
        rig.stand, run_doc, (FORTY,), (documents.forty,), wells=("A1", "a01")
    )

    assert one.reason == (
        "give at least two wells: the offset is a translation only if they agree"
    )
    assert twice.reason == "A1 is given twice in wells"
    _nothing_moved(rig)


def test_a_target_without_a_calibration_is_refused_before_the_turret_turns(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY, TWENTY), (documents.forty,))

    assert refused.reason == (
        f"no camera calibration of {TWENTY} in calibrations: turn to it (smc "
        f'objective set "{TWENTY}" --confirm), run smc run camera-calibration '
        "--run-doc <a document of its own> on a well's wall, and add that document "
        "to calibrations"
    )
    _nothing_moved(rig)


def test_a_calibration_of_another_objective_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.twenty,))

    assert refused.reason == (
        f"the calibration in {documents.twenty.as_posix()} is of {TWENTY}, which is "
        "not in targets: give one calibration per target"
    )
    _nothing_moved(rig)


def test_two_calibrations_of_one_target_are_refused(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    tmp_path: Path,
) -> None:
    again = tmp_path / "cal-40x-again.json"
    shutil.copy(documents.forty, again)
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty, again))

    assert refused.reason == (
        f"{documents.forty.as_posix()} and {again.as_posix()} are both calibrations "
        f"of {FORTY}: give one per target"
    )
    _nothing_moved(rig)


def test_a_calibration_that_cannot_be_read_is_refused_with_its_path(
    open_stand: Callable[..., Microscope], run_doc: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "none.json"
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (missing,))

    assert refused.reason.startswith(f"{missing.as_posix()}: ")
    _nothing_moved(rig)


def test_a_reference_the_run_does_not_measure_is_refused_before_the_turret_turns(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand(_profile(reference=TWENTY)))

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == (
        f"the profile's offsets are measured from {TWENTY} ([objectives] reference), "
        f"which this run does not measure: add {TWENTY} to targets, or clear the "
        "reference"
    )
    _nothing_moved(rig)


def test_a_walk_that_leaves_the_soft_limits_is_refused_before_the_turret_turns(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    base = _profile()
    profile = base.model_copy(
        update={
            "safety": base.safety.model_copy(
                update={"xy_soft_limits_um": ((-100.0, 100.0), (-100.0, 100.0))}
            )
        }
    )
    rig = _rig(open_stand(profile))
    # A refusal that comes before anything moves comes before the autofocus is
    # touched too: a PFS switched off for a run that never moves is a loss (#478).
    af = RecordingAutofocus(engaged=True)
    rig.stand.override(Autofocus, af)

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert "wall would be looked for at" in refused.reason
    assert refused.reason.endswith("outside the stage's soft limits")
    _nothing_moved(rig)
    assert "disengage" not in af.calls
    assert af.engaged is True


def _edited_calibration(
    documents: Documents, tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> Path:
    """A copy of the 40X's calibration document whose result went through ``edit``; the record stays a usable one."""
    path = tmp_path / "cal-40x-edited.json"
    shutil.copy(documents.forty, path)
    record = RunDocument.load(path).stages["camera-calibration"]
    dumped = record.model_dump(mode="json")
    edit(dumped["result"])
    write_stage(path, StageRecord.model_validate(dumped))
    return path


def test_a_calibration_that_makes_too_many_frames_is_refused_before_the_turret_turns(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    tmp_path: Path,
) -> None:
    # A pixel size of 0.01 µm makes the walk's step 3.84 µm: 260 steps either way over
    # the default search, 3 wells and 4 walls, thousands of frames (FM-74, FM-140). The
    # demo's own objectives are all coarse enough to stay under the bound of 600.
    fine = _edited_calibration(
        documents, tmp_path, lambda result: result.update(pixel_size_um=0.01)
    )
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (fine,))

    match = re.fullmatch(
        r"up to (\d+) frames: use a lower magnification or a smaller search_um",
        refused.reason,
    )
    assert match is not None, refused.reason
    assert int(match[1]) > 600
    _nothing_moved(rig)


@pytest.mark.parametrize(
    ("edit", "start"),
    [
        (
            lambda result: result.update(image_shape_px=[256, 256]),
            "the camera calibration is not of this stand: it is of a frame of "
            "(256, 256), the camera gives (512, 512); ",
        ),
        (
            lambda result: result.update(matrix_um_per_px=[[0.0, 0.0], [0.0, 0.0]]),
            "the camera calibration's matrix cannot be inverted (determinant 0.0); ",
        ),
    ],
    ids=["another-frame", "singular-matrix"],
)
def test_a_calibration_that_does_not_fit_the_camera_is_refused_before_the_turret_turns(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    start: str,
) -> None:
    # Only the objective's match is by construction: the frame and the matrix are the
    # tool's own check of each calibration (FM-125, FM-43).
    bad = _edited_calibration(documents, tmp_path, edit)
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (bad,))

    assert refused.reason == (
        f"{start}run camera-calibration with {FORTY} in place, on a field with "
        "texture: a well's wall will do"
    )
    _nothing_moved(rig)


@pytest.mark.parametrize(
    ("offset", "shown"),
    [((-1500.0, 0.0), "(-1500.0, 0.0)"), ((0.0, 1000.5), "(0.0, 1000.5)")],
    ids=["x-beyond", "y-beyond"],
)
def test_an_offset_beyond_the_profiles_bound_is_refused_and_leaves_nothing_to_write(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    monkeypatch: pytest.MonkeyPatch,
    offset: tuple[float, float],
    shown: str,
) -> None:
    # Two parcentric objectives are tens of µm apart. A measured shift cannot pass 1000
    # (the search stops there), but an offset moved to another reference can: the
    # arithmetic is tested on its own, this is that the tool refuses what it returns.
    monkeypatch.setattr(
        "smc.plugins.objective_offsets.plugin.offsets_from_reference",
        lambda origin, shifts, reference: (origin, ((FORTY, offset),)),
    )
    stand = open_stand()

    refused = _refused(
        stand, run_doc, (FORTY,), (documents.forty,), wells=("A1", "H12")
    )

    assert refused.reason == (
        f"the offset of {FORTY} would be {shown} µm, beyond the profile's ±1000 µm: "
        "a wrong well or a wrong calibration, not two parcentric objectives"
    )
    record = _stored(run_doc)
    assert record.status is StageStatus.failed
    assert record.result is None


def test_an_offset_on_the_profiles_bound_is_accepted(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "smc.plugins.objective_offsets.plugin.offsets_from_reference",
        lambda origin, shifts, reference: (origin, ((FORTY, (1000.0, -1000.0)),)),
    )
    stand = open_stand()

    result = _result(
        _run(stand, run_doc, (FORTY,), (documents.forty,), wells=("A1", "H12"))
    )

    assert _offsets(result) == {FORTY: (1000.0, -1000.0)}


def test_a_calibration_path_is_recorded_as_typed_and_not_resolved(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A relative path is relative to the working directory, and the result names it so:
    # an absolute one is a user's folder in a file that is attached to issues (FM-24,
    # FM-50). The temporary folders here are absolute already, so the test types it
    # relative, from the folder that holds it.
    monkeypatch.chdir(documents.forty.parent)
    stand = open_stand()

    result = _result(
        _run(
            stand,
            run_doc,
            (FORTY,),
            (Path(documents.forty.name),),
            wells=("A1", "H12"),
        )
    )

    assert result.measures[0].calibration == "cal-40x.json"


def test_search_um_reaches_the_walk_and_its_refusal_names_this_tool(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # Under the 40X the walls lie 37.5 µm (x) and 12.25 µm (y) from where the 10X's
    # centres put them: a search of 5 µm does not reach them. The refusal comes from
    # plate-walls' loop, and says whose run it is and what to check for this tool.
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,), search_um=5.0)

    assert refused.reason.startswith("no wall found on A1's +X side in ")
    assert refused.reason.endswith(
        f"check that the light is on, that {FORTY} images the wall as the "
        "registration's objective did, and that search_um (5 µm) covers the offset"
    )
    # The turret was turned, and is left where it is.
    assert rig.turret.turns == [FORTY]


# --------------------------------------------------- refused once it has started


def _edit_plate(run_doc: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite the plate record's result through ``edit``; the record stays a usable one (§20.1)."""
    record = RunDocument.load(run_doc).stages["plate-walls"]
    dumped = record.model_dump(mode="json")
    edit(dumped["result"])
    write_stage(run_doc, StageRecord.model_validate(dumped))


def _move_well_in_the_record(
    run_doc: Path, well: str, by_um: tuple[float, float]
) -> None:
    """Move a well's measured centre in the plate record, which stays a usable record of a valid result."""

    def edit(result: dict[str, Any]) -> None:
        for fit in result["walls"]["wells"]:
            if fit["well"] == well:
                fit["center_xy_um"] = [
                    fit["center_xy_um"][0] + by_um[0],
                    fit["center_xy_um"][1] + by_um[1],
                ]

    _edit_plate(run_doc, edit)


def test_a_plate_whose_method_is_not_wall_images_is_refused_even_with_wall_images(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # A record edited by hand: the method says the centres were fitted, the walls
    # are still there. The method is what says who measured them.
    _edit_plate(run_doc, lambda result: result.update(method="lattice-fit"))
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == (
        "the plate was registered with lattice-fit: objective-offsets compares the "
        "centres plate-walls measured, so register it with plate-walls"
    )
    _nothing_moved(rig)


def test_a_plate_that_does_not_record_its_objective_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    _edit_plate(run_doc, lambda result: result.update(objective=None))
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == (
        "the plate does not record the objective it was registered with: register "
        "it with plate-walls on a stand with a turret"
    )
    _nothing_moved(rig)


def test_a_plate_whose_wall_images_name_another_objective_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    _edit_plate(run_doc, lambda result: result["walls"].update(objective=TWENTY))
    rig = _rig(open_stand())

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == (
        f"the plate's record names {TEN} and its wall images {TWENTY}: register it "
        "again with plate-walls"
    )
    _nothing_moved(rig)


def test_wells_that_disagree_are_refused_with_their_numbers(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # H12's recorded centre is 20 µm east of where the 10X measured it: under the
    # 20X it differs by (-5, 2.5) - (20, 0) there, the other wells by (-5, 2.5).
    _move_well_in_the_record(run_doc, "H12", (20.0, 0.0))
    stand = open_stand()

    refused = _refused(stand, run_doc, (TWENTY,), (documents.twenty,))

    reason = refused.reason
    assert reason.startswith(
        "the wells do not agree on the offset, so it is not a translation or not "
        f"measured well enough: {TWENTY}: A1 ("
    )
    numbers = [
        float(n) for n in re.findall(r"[+-]?\d+\.\d\d", reason.split(f"{TWENTY}: ")[1])
    ]
    # A1 (-5.00, +2.50), H12 (-25.00, +2.50), A12 (-5.00, +2.50), spread 20, bound 2.
    assert numbers[0:2] == pytest.approx([-5.0, 2.5], abs=0.5)
    assert numbers[2:4] == pytest.approx([-25.0, 2.5], abs=0.5)
    assert numbers[4:6] == pytest.approx([-5.0, 2.5], abs=0.5)
    assert numbers[6] == pytest.approx(20.0, abs=0.5)
    assert numbers[7] == 2.0
    assert re.search(
        r"look at the frames in objective-offsets/[0-9a-f]{12}/1, ", reason
    )
    assert reason.endswith(
        "or give a larger max_spread_um if the stage's repeatability is larger"
    )
    # A refusal is a failed record with no result: there is nothing to write from.
    record = _stored(run_doc)
    assert record.status is StageStatus.failed
    assert record.result is None


def test_the_frames_were_not_saved_is_said_in_place_of_the_folder(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    _move_well_in_the_record(run_doc, "H12", (20.0, 0.0))
    stand = open_stand()

    refused = _refused(
        stand, run_doc, (TWENTY,), (documents.twenty,), save_frames=False
    )

    assert (
        "; the frames were not saved, or give a larger max_spread_um" in refused.reason
    )
    assert "look at the frames" not in refused.reason


def test_a_turret_that_reads_another_label_after_the_turn_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(open_stand(), turret=_LyingTurret)

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == (
        f"the turret reads {TEN} after the turn to {FORTY}: it did not turn, or its "
        f"label lags (m2 §8.11 D3); nothing was measured with {FORTY}"
    )
    # The layer turned it, once; nothing was imaged under the target.
    assert rig.turret.turns == [FORTY]
    assert rig.camera.snaps == 0
    assert rig.stage.moves == 0


def test_z_still_moving_after_the_turn_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()
    z = _Z(stand.require(ZStage))
    stand.override(ZStage, z)
    rig = _rig(stand, turret=lambda inner: _DriftingTurret(inner, z))

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert re.fullmatch(
        r"Z moved from -?\d+\.\d\d to -?\d+\.\d\d µm within 0\.1 s after the turn to "
        rf"{FORTY}: the stand was still moving it; nothing was measured with {FORTY}",
        refused.reason,
    ), refused.reason
    assert rig.turret.turns == [FORTY]
    assert rig.camera.snaps == 0
    assert rig.stage.moves == 0


def _rig_with_z(stand: Microscope, make_z: Callable[[ZStage], _Z]) -> _Rig:
    """The counting doubles, with a focus drive of this test's own that the turret arms once it has turned."""
    z = make_z(stand.require(ZStage))
    stand.override(ZStage, z)
    return _rig(stand, turret=lambda inner: _DriftingTurret(inner, z))


def test_a_z_that_creeps_with_time_is_refused_because_the_two_reads_are_apart(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # `_Z` above drifts per read, which no wait can change. This one drifts 10 µm per
    # second: reads 0.1 s apart differ by 1 µm (over the 0.5 µm bound), and reads a few
    # microseconds apart by nothing, so only a plugin that waits between them refuses it.
    rig = _rig_with_z(open_stand(), _ClockZ)

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert re.fullmatch(
        r"Z moved from -?\d+\.\d\d to -?\d+\.\d\d µm within 0\.1 s after the turn to "
        rf"{FORTY}: the stand was still moving it; nothing was measured with {FORTY}",
        refused.reason,
    ), refused.reason
    assert rig.camera.snaps == 0
    assert rig.stage.moves == 0


def test_z_after_the_turn_is_the_second_read_of_a_z_that_is_still(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # Reads of 0.0 and then 0.3 µm: within the 0.5 µm bound, so the run goes on, and
    # what is recorded is where the stand was once it was still: the second read.
    rig = _rig_with_z(open_stand(), lambda inner: _ScriptedZ(inner, [0.0, 0.3]))

    result = _result(
        _run(
            rig.stand,
            run_doc,
            (TWENTY,),
            (documents.twenty,),
            wells=("A1", "H12"),
        )
    )

    assert result.measures[0].z_after_turn_um == pytest.approx(0.3)


@pytest.mark.parametrize(
    "script",
    [
        [math.nan, math.nan],
        [math.nan, 0.0],
        [0.0, math.nan],
        [math.inf, math.inf],
        [-math.inf, math.inf],
    ],
    ids=["nan-nan", "nan-number", "number-nan", "inf-inf", "minus-inf-inf"],
)
def test_a_z_that_is_not_a_number_is_refused_before_anything_is_imaged(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    script: list[float],
) -> None:
    # A NaN compares false with everything, so `abs(nan) > 0.5` called it still and the
    # walk started at an unknown Z (FM-104, FM-195).
    rig = _rig_with_z(open_stand(), lambda inner: _ScriptedZ(inner, script))

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert re.fullmatch(
        r"Z reads (-?inf|nan|0\.0) and (-?inf|nan|0\.0) µm after the turn to "
        rf"{FORTY}: a Z that is not a number is not a still Z; nothing was measured "
        rf"with {FORTY}",
        refused.reason,
    ), refused.reason
    assert rig.turret.turns == [FORTY]
    assert rig.camera.snaps == 0
    assert rig.stage.moves == 0


def test_a_stop_between_two_targets_turns_nothing_more(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    # The stop comes after the last side of the first target's last well (3 wells, 4
    # sides: the 12th of 24) and before the second target's turn: only the check at the
    # top of the loop can honour it, and a turn over a loaded plate after a stop is the
    # failure (FM-14, FM-64, FM-88).
    stop = threading.Event()
    rig = _rig(open_stand())

    def after_the_first_target(progress: Progress) -> None:
        if progress.done == 12:
            stop.set()

    record = _run(
        rig.stand,
        run_doc,
        (FORTY, TWENTY),
        (documents.forty, documents.twenty),
        stop=stop,
        on_progress=after_the_first_target,
    )

    assert record.status is StageStatus.stopped
    assert rig.turret.turns == [FORTY]
    assert rig.turret.current() == FORTY


def test_a_stop_asked_before_the_first_turn_turns_nothing(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stop = threading.Event()
    stop.set()
    rig = _rig(open_stand())

    record = _run(rig.stand, run_doc, (FORTY,), (documents.forty,), stop=stop)

    assert record.status is StageStatus.stopped
    _nothing_moved(rig)


def test_a_turn_the_layer_refuses_is_refused_with_its_text(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    refusal = SafetyRefusedError("the stage is not clear")
    rig = _rig(open_stand(), turret=lambda inner: _RefusingTurret(inner, refusal))

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == f"the turn to {FORTY} was refused: the stage is not clear"
    assert rig.camera.snaps == 0


def test_a_halt_during_a_turn_is_not_a_refusal(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    halted = MicroscopeHaltedError()
    rig = _rig(open_stand(), turret=lambda inner: _RefusingTurret(inner, halted))

    with pytest.raises(MicroscopeHaltedError):
        _run(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert _stored(run_doc).status is StageStatus.failed


def test_a_stop_leaves_the_turret_and_the_stage_where_they_are(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stop = threading.Event()
    rig = _rig(open_stand(), before=lambda n: stop.set() if n == 5 else None)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(
            rig.stand,
            run_doc,
            (FORTY, TWENTY),
            (documents.forty, documents.twenty),
            stop=stop,
        )

    assert record.status is StageStatus.stopped
    # Neither was put back: the turret stays at the 40X, the stage where it stopped.
    assert rig.turret.turns == [FORTY]
    assert rig.turret.current() == FORTY
    where = _here(rig.stand)
    # The plugin's two WARNINGs, in this order: the stage, then the turret.
    warnings = [r for r in caplog.records if r.name == LOGGER]
    assert [r.levelno for r in warnings] == [logging.WARNING, logging.WARNING]
    stage_line, turret_line = (r.getMessage() for r in warnings)
    assert stage_line.startswith("objective-offsets stopped at ")
    assert "not returned to" in stage_line
    match = re.search(r"left at \((-?\d+\.\d), (-?\d+\.\d)\) µm", stage_line)
    assert match is not None
    assert (float(match[1]), float(match[2])) == pytest.approx(where, abs=0.1)
    assert turret_line == (
        f"objective-offsets: the turret is left at {FORTY}, not turned back"
    )


def test_the_turret_warning_says_unknown_when_the_turret_cannot_be_read(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stop = threading.Event()
    rig = _rig(
        open_stand(),
        turret=lambda inner: _UnreadableTurret(inner, stop),
        after=lambda n: stop.set() if n == 3 else None,
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(rig.stand, run_doc, (FORTY,), (documents.forty,), stop=stop)

    assert record.status is StageStatus.stopped
    warnings = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert warnings[-1] == (
        "objective-offsets: the turret is left at unknown, not turned back"
    )


# ------------------------------------------------------------ what a run leaves


def test_the_stage_ends_at_the_last_wells_centre_under_the_last_target(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()

    result = _result(
        _run(stand, run_doc, (FORTY, TWENTY), (documents.forty, documents.twenty))
    )

    # The last target is the 20X, the last well A12: its centre under the 20X is the
    # true one plus the 20X's shift (zero), where the operator sees it on Live.
    assert _here(stand) == pytest.approx(_centred("A12", TWENTY), abs=0.5)
    assert result.measures[-1].wells[-1].fit.well == "A12"
    assert stand.require(ObjectiveTurret).current() == TWENTY
    assert result.objective_in_place == TWENTY


def test_frames_are_saved_per_target_without_a_label_in_the_path(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()

    result = _result(
        _run(stand, run_doc, (FORTY, TWENTY), (documents.forty, documents.twenty))
    )

    folders = [m.folder for m in result.measures]
    assert folders[0] is not None
    token = folders[0].split("/")[1]
    assert re.fullmatch(r"[0-9a-f]{12}", token)
    assert folders == [f"objective-offsets/{token}/1", f"objective-offsets/{token}/2"]
    saved = sorted((run_doc.parent / "objective-offsets").rglob("*.tif"))
    # Two targets, three wells, four walls: a frame centred on each.
    assert len(saved) == 24
    for path in saved:
        text = path.relative_to(run_doc.parent).as_posix()
        assert "Nikon" not in text
        assert " " not in text
    first = {p.name for p in (run_doc.parent / folders[0]).glob("*.tif")}
    assert first == {
        f"{w}_{s}.tif" for w in ("A1", "H12", "A12") for s in ("+X", "+Y", "-X", "-Y")
    }
    for measure in result.measures:
        for well in measure.wells:
            assert [w.file for w in well.fit.walls] == [
                f"{well.well}_{s}.tif" for s in ("+X", "+Y", "-X", "-Y")
            ]


def test_no_frame_is_written_without_save_frames(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()

    result = _result(
        _run(stand, run_doc, (TWENTY,), (documents.twenty,), save_frames=False)
    )

    assert result.measures[0].folder is None
    assert not (run_doc.parent / "objective-offsets").exists()


def test_progress_counts_every_side_of_every_well_under_every_target(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()
    seen: list[Progress] = []

    _run(
        stand,
        run_doc,
        (FORTY, TWENTY),
        (documents.forty, documents.twenty),
        on_progress=seen.append,
    )

    # Two targets, three wells, four sides: 24, counted from 1 over the whole run.
    assert [p.done for p in seen] == list(range(1, 25))
    assert {p.total for p in seen} == {24}
    assert seen[0].message == f"{FORTY}: A1 +X"
    assert seen[-1].message == f"{TWENTY}: A12 -Y"


def test_the_wells_given_are_the_wells_measured(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stand = open_stand()

    result = _result(
        _run(stand, run_doc, (TWENTY,), (documents.twenty,), wells=("a12", "A1"))
    )

    (measure,) = result.measures
    assert [w.well for w in measure.wells] == ["A12", "A1"]
    # Its frames: two wells, four sides.
    assert measure.folder is not None
    assert len(list((run_doc.parent / measure.folder).glob("*.tif"))) == 8


# ------------------------- PFS across a well's wall: the switch-off (m3 §5.7, #478)

AUTOFOCUS_OFF_LINE = (
    "the autofocus was switched off before anything moved, so that no lock crossed "
    "a well's wall: engage it again on the sample"
)
AUTOFOCUS_OFF_WARNING = (
    "objective-offsets: the autofocus was switched off before anything moved, so "
    "that no lock crosses a well's wall (m2 §12.2 D2): engage it again after the run"
)
NO_LIMITS_LINE = (
    "the stage reports no soft limits, so the frames' moves were not checked "
    "against any travel"
)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The plugin's own warnings: ``override`` and the layer's turret warn too, on their own loggers."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == LOGGER
    ]


def test_objective_offsets_switches_an_engaged_autofocus_off_before_anything_moves_at_the_first_target(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap that was measured: the turret already reads the first target.

    ``select`` then returns at once, so the layer's own switch-off before a turn
    never runs, and the first move to a wall was made with PFS engaged. An operator
    who ran ``camera-calibration`` at the 40x and left it in place meets it.
    """
    stand = open_stand()
    stand.require(ObjectiveTurret).select(FORTY, confirm=True)
    rig = _rig(stand)
    af = RecordingAutofocus(engaged=True)
    stand.override(Autofocus, af)
    real_move = rig.stage.move_to_um

    def move(x_um: float, y_um: float) -> XY:
        af.calls.append("move_to_um")
        return real_move(x_um, y_um)

    monkeypatch.setattr(rig.stage, "move_to_um", move)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(stand, run_doc, (FORTY,), (documents.forty,)))

    # The switch-off and its read back come before any move, and are made once. The
    # layer's turns read the autofocus too, so the rest of the list is not asserted.
    assert af.calls[:3] == ["is_engaged", "disengage", "is_engaged"]
    assert af.calls.index("move_to_um") > 2
    assert af.calls.count("disengage") == 1
    assert af.engaged is False
    assert result.assumed[:2] == (AUTOFOCUS_OFF_LINE, NO_LIMITS_LINE)
    assert _warnings(caplog) == [AUTOFOCUS_OFF_WARNING]


def test_objective_offsets_leaves_an_autofocus_that_is_off_alone(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stand = open_stand()
    af = RecordingAutofocus(engaged=False)
    stand.override(Autofocus, af)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(stand, run_doc, (FORTY,), (documents.forty,)))

    # The layer's turns read the autofocus too: the list is not asserted whole.
    assert "disengage" not in af.calls
    assert af.calls[0] == "is_engaged"
    assert AUTOFOCUS_OFF_LINE not in result.assumed
    assert result.assumed[0] == NO_LIMITS_LINE
    assert _warnings(caplog) == []


def test_a_stop_before_objective_offsets_turns_keeps_the_autofocus_on(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    stop = threading.Event()
    stop.set()
    rig = _rig(open_stand())
    af = RecordingAutofocus(engaged=True)
    rig.stand.override(Autofocus, af)

    record = _run(rig.stand, run_doc, (FORTY,), (documents.forty,), stop=stop)

    assert record.status is StageStatus.stopped
    _nothing_moved(rig)
    assert "disengage" not in af.calls
    assert af.engaged is True


_NOT_SWITCHED_OFF = (
    "the autofocus could not be switched off before anything moved ({cause}): "
    "nothing moved, and PFS may still be on: switch it off at the stand, then run "
    "objective-offsets again"
)
_STILL_ON = (
    "the autofocus still reads on after it was switched off: nothing moved: switch "
    "it off at the stand, then run objective-offsets again"
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
def test_a_switch_off_that_fails_ends_objective_offsets_before_anything_turns_or_moves(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    options: dict[str, Any],
    reason: str,
    calls: list[str],
) -> None:
    """Unknown counts as on (FM-19): the run fails with §5.7's text whole; the turret and the stage stay."""
    rig = _rig(open_stand())
    af = RecordingAutofocus(engaged=True, **options)
    rig.stand.override(Autofocus, af)
    before = _here(rig.stand)

    refused = _refused(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert refused.reason == reason
    assert af.calls == calls
    _nothing_moved(rig)
    assert _here(rig.stand) == before
    stored = _stored(run_doc)
    assert stored.status is StageStatus.failed
    assert stored.error == f"PluginRunError: objective-offsets: {reason}"


@pytest.mark.parametrize(
    "fault",
    [MicroscopeHaltedError(), MotionStoppedError("the stage was stopped")],
    ids=["halt", "motion-stopped"],
)
def test_a_halt_from_the_switch_off_reaches_the_caller_unchanged_in_objective_offsets(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    documents: Documents,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    """A halt or a stop is not a switch-off that failed: it passes first (FM-64)."""
    rig = _rig(open_stand())
    af = RecordingAutofocus(engaged=True)

    def fails() -> None:
        raise fault

    monkeypatch.setattr(af, "disengage", fails)
    rig.stand.override(Autofocus, af)

    with pytest.raises(type(fault)) as info:
        _run(rig.stand, run_doc, (FORTY,), (documents.forty,))

    assert info.value is fault
    _nothing_moved(rig)


def test_a_halted_stand_switches_the_autofocus_off_and_objective_offsets_ends_on_the_halt(
    open_stand: Callable[..., Microscope], run_doc: Path, documents: Documents
) -> None:
    """PFS off is the safe side; the turn then raises the halt (FM-64).

    The stand's stop switches the autofocus off itself; one switched on again
    while halted (at the stand) is what the tool's own switch-off meets.
    """
    rig = _rig(open_stand())
    af = RecordingAutofocus(engaged=True)
    rig.stand.override(Autofocus, af)
    rig.stand.stop()
    assert af.engaged is False  # the stop's own switch-off
    af.engaged = True
    af.calls.clear()
    try:
        with pytest.raises(MicroscopeHaltedError):
            _run(rig.stand, run_doc, (FORTY,), (documents.forty,))
    finally:
        rig.stand.resume()

    assert af.calls[:3] == ["is_engaged", "disengage", "is_engaged"]
    assert _stored(run_doc).status is StageStatus.failed
    assert rig.stage.moves == 0
    assert rig.camera.snaps == 0
