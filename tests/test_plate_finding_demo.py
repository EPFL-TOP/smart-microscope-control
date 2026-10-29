"""``plate-finding`` on the demo devices and the fake core: the survey loop and what it leaves (m3 §6, #20).

The demo camera ignores the stage, so a ``PlateSample`` is put on the stand
through a camera whose frames follow it (``_Camera``, a ``SampleCamera`` that
also records where each frame was taken). The demo stage moves at its
``Velocity`` property, 10 mm/s by default, which would make one 208-point survey
350 s: the demo tests raise it to 100 000 on the demo's own core. The tests that
are about the loop's decisions and not about the stage run on ``fake_microscope``,
whose stage is instant.

Expected values are written here, from the plate's geometry and the survey's
arithmetic, never read from the result's own fields (FM-124). The frames are 24
x 32 pixels, not square (FM-73), at 20 µm per pixel. The module runs with
``RuntimeWarning`` as an error (FM-131).

A 12-well plate is 3 rows by 4 columns at a pitch of 26 mm with wells of 22 mm:
at the default step (a third of the pitch) and one pitch of margin the survey
spans 130 x 104 mm in 16 columns by 13 rows, 208 points.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from smc.hardware import Microscope
from smc.hardware.capabilities import XY, Autofocus, Camera, FocusStep, Limits, XYStage
from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
)
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, Progress, ResumeError, load, run_plugin
from smc.plugins.plate_finding.algorithm import (
    DECISIVE_MARGIN,
    survey_extent_note,
    survey_points,
)
from smc.plugins.plate_finding.plugin import PlateFindingParams, PlateFindingPlugin
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.fakes import FakeCore
from smc.testing.synthetic import PlateSample, SampleCamera

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

SHAPE = (24, 32)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 20.0
PITCH_UM = 26_000.0  # the 12-well plate's
RADIUS_UM = 11_000.0  # ... and its wells'
LOGGER = "smc.plugins.plate-finding"
#: The WARNING logged when the survey switches the autofocus off.
AUTOFOCUS_LINE = (
    "the autofocus was switched off for the survey; engage it again before the scan"
)
#: Written out here, not imported: a test that reads the plugin's own text cannot
#: notice that the text changed.
NO_LIMITS_LINE = (
    "the stage reports no soft limits, so the survey's moves were not checked "
    "against any travel"
)
AUTOFOCUS_OFF_LINE = (
    "the autofocus is off after the survey; the scan never switches it on: lock it "
    "on the sample before the scan"
)
#: What an autofocus does that the survey must never do.
MUTATING = {
    "engage",
    "disengage",
    "set_offset_native",
    "offset_by_native",
    "focus_by_um",
}


# ------------------------------------------------------------------ the stand


class _Camera(SampleCamera):
    """A sample camera that records where each frame was taken and can act at a numbered snap.

    ``before(n)`` runs at snap ``n`` before the frame is rendered, ``spoil(n,
    frame)`` may replace it, and ``fail`` maps a snap to what it raises.
    """

    def __init__(
        self,
        microscope: Microscope,
        sample: PlateSample,
        *,
        before: Callable[[int], None] | None = None,
        spoil: Callable[[int, npt.NDArray[np.uint16]], npt.NDArray[Any]] | None = None,
    ) -> None:
        super().__init__(microscope, sample, pixel_size_um=PIXEL_UM, shape=SHAPE)
        self._stand = weakref.ref(microscope)
        self.snaps = 0
        self.positions: list[tuple[float, float]] = []
        self._before = before
        self._spoil = spoil

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


class _Stage:
    """The stand's XY stage, except where a test makes a move misbehave; the others delegate.

    Every ``every``-th move times out; ``raises`` maps a move number to what it
    raises; ``clamp_x_um`` sends the stage no further than that in x and returns
    where it really is (a stall, a hard stop); ``nan_readback_at`` is a move that
    arrives and reads back a position that is not finite.

    It implements the seven methods of ``XYStage``: ``Microscope.override``
    refuses an implementation that lacks one. It bypasses the ``Executor``, so it
    is never used where a halt is the point.
    """

    def __init__(
        self,
        inner: XYStage,
        *,
        every: int = 10**9,
        raises: dict[int, BaseException] | None = None,
        clamp_x_um: float | None = None,
        nan_readback_at: int | None = None,
    ) -> None:
        self._inner = inner
        self._every = every
        self._raises = raises or {}
        self._clamp_x_um = clamp_x_um
        self._nan_readback_at = nan_readback_at
        self.moves = 0
        self.calls: list[str] = []

    def position_um(self) -> XY:
        self.calls.append("position_um")
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        if self.moves in self._raises:
            raise self._raises[self.moves]
        if self.moves % self._every == 0:
            raise DeviceTimeoutError(
                f"xy_stage did not arrive within 30 s (move {self.moves})\nretry"
            )
        if self._clamp_x_um is not None:
            limit = self._clamp_x_um
            x_um = max(-limit, min(limit, x_um))
        reached = self._inner.move_to_um(x_um, y_um)
        if self.moves == self._nan_readback_at:
            return XY(math.nan, math.nan)
        return reached

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("plate-finding makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self.calls.append("wait")
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._inner.limits_um()


class _Autofocus:
    """An autofocus that records every call; the survey may read it and may only switch it off."""

    def __init__(self, *, engaged: bool) -> None:
        self.engaged = engaged
        self.calls: list[str] = []

    def is_engaged(self) -> bool:
        self.calls.append("is_engaged")
        return self.engaged

    def is_locked(self) -> bool:
        self.calls.append("is_locked")
        return self.engaged

    def is_in_range(self) -> bool | None:
        self.calls.append("is_in_range")
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        self.calls.append("engage")
        self.engaged = True
        return True

    def disengage(self) -> None:
        self.calls.append("disengage")
        self.engaged = False

    def offset_native(self) -> float:
        self.calls.append("offset_native")
        return 0.0

    def set_offset_native(self, value: float) -> float:
        self.calls.append("set_offset_native")
        return value

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        self.calls.append("offset_by_native")
        return delta

    def offset_unit(self) -> str:
        self.calls.append("offset_unit")
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        self.calls.append("offset_limits_native")
        return None

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        self.calls.append("focus_by_um")
        raise AssertionError("plate-finding never focuses")


class _AutofocusReadAtTheEnd(_Autofocus):
    """An autofocus whose first ``is_engaged()`` is the stand's and whose later ones answer as the test says.

    The survey reads the autofocus twice on a fresh run: at the start, to
    switch it off, and at the end, to say what the scan will find. The second
    read is ``at_the_end``: what it answers, or what it raises.
    """

    def __init__(self, *, engaged: bool, at_the_end: bool | BaseException) -> None:
        super().__init__(engaged=engaged)
        self._at_the_end = at_the_end
        self._reads = 0

    def is_engaged(self) -> bool:
        self._reads += 1
        if self._reads == 1:
            return super().is_engaged()
        self.calls.append("is_engaged")
        if isinstance(self._at_the_end, BaseException):
            raise self._at_the_end
        return self._at_the_end


def _plate_at(
    plate_centre: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
    *,
    well_level: float = 3000.0,
    plastic_level: float = 800.0,
    texture_std: float = 40.0,
    noise_std: float = 20.0,
) -> PlateSample:
    """A 12-well plate whose centre, not A1, is at ``plate_centre``: A1 is 1.5 pitches left and 1 up of it.

    ``A1 + R(theta) * (1.5 * pitch, -1 * pitch)`` is the centre, so with no
    rotation A1 is at ``centre + (-39 000, +26 000)``.
    """
    angle = math.radians(rotation_deg)
    lx, ly = 1.5 * PITCH_UM, -1.0 * PITCH_UM
    a1 = (
        plate_centre[0] - (math.cos(angle) * lx - math.sin(angle) * ly),
        plate_centre[1] - (math.sin(angle) * lx + math.cos(angle) * ly),
    )
    return PlateSample(
        "12-well",
        a1,
        rotation_deg,
        well_level=well_level,
        plastic_level=plastic_level,
        texture_std=texture_std,
        noise_std=noise_std,
    )


def _on_demo(demo_microscope: Microscope, sample: PlateSample, **hooks: Any) -> _Camera:
    """The demo stand with a camera that follows the stage, and a stage fast enough to survey.

    The demo's XY stage moves at its ``Velocity`` property, 10 mm/s by default:
    a 30 mm move takes 3 s. At 100 000 it takes 13 ms (measured), which makes a
    survey of 208 points 3 s. The property is the demo adapter's own, set on
    this stand's core and nowhere else.
    """
    demo_microscope.core.setProperty("XY", "Velocity", "100000")
    camera = _Camera(demo_microscope, sample, **hooks)
    demo_microscope.override(Camera, camera)
    return camera


def _on_fake(fake_microscope: Microscope, sample: PlateSample, **hooks: Any) -> _Camera:
    camera = _Camera(fake_microscope, sample, **hooks)
    fake_microscope.override(Camera, camera)
    return camera


@pytest.fixture
def open_limited(mm_available: bool) -> Iterator[Callable[..., Microscope]]:
    """A demo stand whose XY soft limits are given, closed afterwards; as ``goto-well``'s test opens one."""
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


@pytest.fixture
def open_limited_fake(fake_core: FakeCore) -> Iterator[Callable[..., Microscope]]:
    """The fake core's stand with XY soft limits given: instant moves, no Micro-Manager."""
    opened: list[Microscope] = []

    def make(limits: tuple[tuple[float, float], tuple[float, float]]) -> Microscope:
        profile = Profile.demo()
        limited = profile.model_copy(
            update={
                "safety": profile.safety.model_copy(
                    update={"xy_soft_limits_um": limits}
                )
            }
        )
        # The facade types its core as CMMCorePlus; the fake is structural.
        microscope = Microscope.from_core(cast("Any", fake_core), limited)
        opened.append(microscope)
        return microscope

    yield make
    for microscope in opened:
        microscope.close()


# ----------------------------------------------------------------- the helpers


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    resume: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
    stop: threading.Event | None = None,
    name: str = "run.json",
    **params: Any,
) -> StageRecord:
    """Run the survey; the plate surveyed is a 12-well one unless a test says another."""
    return run_plugin(
        PlateFindingPlugin(),
        microscope,
        PlateFindingParams(**{"plate": "12-well", **params}),
        tmp_path / name,
        resume=resume,
        on_progress=on_progress,
        stop=stop,
    )


def _result(record: StageRecord) -> PlateResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return PlateResult.model_validate(record.result)


def _stored(tmp_path: Path, name: str = "run.json") -> StageRecord:
    return RunDocument.load(tmp_path / name).stages["plate-finding"]


def _here(microscope: Microscope) -> tuple[float, float]:
    position = microscope.require(XYStage).position_um()
    return position.x_um, position.y_um


def _refusal(microscope: Microscope, tmp_path: Path, **params: Any) -> str:
    """What the survey refused with, after checking it is recorded ``failed`` with no result."""
    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, **params)
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    return str(info.value)


def _well_centre(
    a1: tuple[float, float], rotation_deg: float, row: float, col: float
) -> tuple[float, float]:
    """The forward model of M1 §8: ``A1 + R(theta) * (col * pitch, -row * pitch)``, for the 12-well plate."""
    angle = math.radians(rotation_deg)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        a1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        a1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _flat(points: list[tuple[float, float]]) -> list[float]:
    """The coordinates one after the other: ``pytest.approx`` takes no nested sequences."""
    return [value for point in points for value in point]


def _snake(columns: list[float], rows: list[float]) -> list[tuple[float, float]]:
    """Rows from the lowest y, each walked the other way from the one before."""
    return [
        (x, y)
        for index, y in enumerate(rows)
        for x in (columns if index % 2 == 0 else columns[::-1])
    ]


def _tamper(tmp_path: Path, *, drop: tuple[str, ...] = (), **checkpoint: Any) -> None:
    """Replace or drop fields of the stored checkpoint, as a hand-edited or foreign run document would."""
    path = tmp_path / "run.json"
    record = RunDocument.load(path).stages["plate-finding"]
    assert record.checkpoint is not None
    kept = {k: v for k, v in record.checkpoint.items() if k not in drop}
    write_stage(
        path,
        record.model_copy(update={"checkpoint": {**kept, **checkpoint}}),
    )


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_plate_finding() -> None:
    entry = load("plate-finding")

    assert entry.plugin.provides == "plate"
    assert entry.plugin.requires == (XYStage, Camera)
    assert entry.plugin.reads == ()
    assert entry.target == "smc.plugins.plate_finding.plugin:PlateFindingPlugin"


# ------------------------------------------------------------- the survey


def test_the_survey_fills_the_plate_slot_on_the_demo(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    sample = _plate_at((0.0, 0.0), 1.2)
    camera = _on_demo(demo_microscope, sample)
    centre = (sample.a1_center_xy_um[0], sample.a1_center_xy_um[1])
    plate_centre = _well_centre(centre, 1.2, 1, 1.5)
    # The operator parked the stage near the plate's middle, not on it.
    demo_microscope.require(XYStage).move_to_um(
        plate_centre[0] + 700.0, plate_centre[1] - 500.0
    )
    progress: list[Progress] = []

    record = _run(demo_microscope, tmp_path, on_progress=progress.append)

    result = _result(record)
    assert result.plate == "12-well"
    assert result.method == "lattice-fit"
    assert result.n_refs == 0
    assert result.fit is not None
    assert result.fit.runner_up is not None
    assert result.fit.margin is not None
    assert result.fit.margin > DECISIVE_MARGIN
    # 16 columns (floor(130 000 / 8666.67) + 1) by 13 rows (floor(104 000 / 8666.67) + 1).
    assert result.fit.n_samples == 16 * 13 == len(camera.positions) == 208
    worst = max(
        math.dist(
            result.well_center_um(f"{'ABC'[row]}{col + 1}"),
            _well_centre(centre, 1.2, row, col),
        )
        for row in range(3)
        for col in range(4)
    )
    assert worst < 0.30 * RADIUS_UM, f"{worst:.0f} µm off"
    # A1 is what the scan plan is built from. The turn of a plate this small is
    # coarse (-0.3 degrees for 1.2: the original does the same, the unit test pins it).
    assert math.dist(result.a1_center_xy_um, centre) < 0.30 * RADIUS_UM
    # A decision, not a coin toss: another plate came second, well behind, and
    # nothing is said about a close call.
    assert result.fit.runner_up != result.plate
    assert 0.0 < result.fit.margin < result.fit.score
    assert result.fit.notes == ()
    # The demo's stage says nothing about its travel, and its autofocus is off.
    assert result.assumed == (NO_LIMITS_LINE, AUTOFOCUS_OFF_LINE)
    assert (progress[-2].done, progress[-2].total, progress[-2].message) == (
        208,
        208,
        "survey",
    )
    assert (progress[-1].done, progress[-1].total, progress[-1].message) == (
        208,
        208,
        "fitting",
    )
    # FM-88: nothing moves after the survey; the stage stays at its last point.
    assert _here(demo_microscope) == pytest.approx(camera.positions[-1], abs=0.05)
    assert _stored(tmp_path) == record


def test_the_survey_visits_the_hand_computed_points_around_the_given_centre(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A 12-well survey 0.5 pitch past the wells: 104 x 78 mm, in steps of 10 mm, snake order.

    x runs from 5000 - 52 000 = -47 000 to 57 000 in 11 columns 10 400 µm apart
    (floor(104 000 / 10 000) = 10 steps); y from -3000 - 39 000 = -42 000 to
    36 000 in 8 rows 78 000 / 7 µm apart. The first row is walked +x, the second
    -x, so the second row starts at x = 57 000.
    """
    camera = _on_demo(demo_microscope, _plate_at())
    columns = [-47_000.0 + k * 10_400.0 for k in range(11)]
    rows = [-42_000.0 + j * 78_000.0 / 7.0 for j in range(8)]

    # Whether so coarse a survey fits is not what this test is about.
    with contextlib.suppress(PluginRunError):
        _run(
            demo_microscope,
            tmp_path,
            plate="12-well",
            plates=["12-well"],
            centre_x_um=5000.0,
            centre_y_um=-3000.0,
            step_um=10_000.0,
            margin_pitches=0.5,
            jitter=0.0,
        )

    expected = _snake(columns, rows)
    assert len(expected) == 88
    assert _flat(camera.positions) == pytest.approx(_flat(expected), abs=0.05)
    assert camera.positions[11][0] == pytest.approx(57_000.0, abs=0.05)
    assert camera.positions[11][1] == pytest.approx(
        -42_000.0 + 78_000.0 / 7.0, abs=0.05
    )


def test_the_seed_and_the_jitter_shape_the_survey(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The points are the algorithm's for the parameters given: a seed makes another survey, the same seed the same."""
    camera = _on_fake(fake_microscope, _plate_at())
    unseeded = survey_points("12-well", (0.0, 0.0), seed=0)
    seeded = survey_points("12-well", (0.0, 0.0), seed=5, jitter=0.25)
    assert seeded != unseeded

    _run(fake_microscope, tmp_path, plates=["12-well"], seed=5, jitter=0.25)

    assert _flat(camera.positions) == pytest.approx(_flat(seeded), abs=0.05)


def test_the_survey_is_centred_on_the_stage_after_it_has_stopped_when_no_centre_is_given(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A stage still travelling from an earlier move reads a fraction of it (FM-85): wait, then read."""
    camera = _on_fake(fake_microscope, _plate_at((3000.0, -2000.0)))
    fake_microscope.require(XYStage).move_to_um(3000.0, -2000.0)
    stage = _Stage(fake_microscope.require(XYStage), every=10**9)
    fake_microscope.override(XYStage, stage)

    _run(fake_microscope, tmp_path, plates=["12-well"], jitter=0.0)

    assert stage.calls[:2] == ["wait", "position_um"]

    # Jitter 0: the first point is the lowest-left corner, half the span from the centre.
    assert camera.positions[0] == pytest.approx(
        (3000.0 - 65_000.0, -2000.0 - 52_000.0), abs=0.05
    )
    assert len(camera.positions) == 208


# --------------------------------------------- the autofocus: FM-12, FM-88


def test_an_engaged_autofocus_is_switched_off_and_not_relocked(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _on_fake(fake_microscope, _plate_at())
    autofocus = fake_microscope.require(Autofocus)
    assert autofocus.engage() is True
    fake_core.log.clear()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(fake_microscope, tmp_path, plates=["12-well"])

    log = fake_core.log
    first_move = next(
        i for i, line in enumerate(log) if line.startswith("setXYPosition")
    )
    assert log.index("enableContinuousFocus(False)") < first_move
    assert "enableContinuousFocus(True)" not in log
    assert autofocus.is_engaged() is False
    # The result says what the scan will find, not what the survey did.
    assert _result(record).assumed == (NO_LIMITS_LINE, AUTOFOCUS_OFF_LINE)
    assert [r.getMessage() for r in caplog.records if r.name == LOGGER] == [
        AUTOFOCUS_LINE
    ]


def test_an_autofocus_that_is_off_is_left_alone(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Off before the survey is reported too: the scan finds it off, whoever switched it off."""
    spy = _Autofocus(engaged=False)
    fake_microscope.override(Autofocus, spy)
    _on_fake(fake_microscope, _plate_at())

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    assert not MUTATING & set(spy.calls)
    assert _result(record).assumed == (NO_LIMITS_LINE, AUTOFOCUS_OFF_LINE)


def test_a_resumed_survey_still_says_the_autofocus_is_off(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first attempt switched it off and was stopped; the resume finds it off and must still say so.

    The line is the autofocus's state at the end, read, so it does not depend on
    which attempt switched it off (#248: the resume returned ``done`` with
    nothing in ``assumed`` and the scan then ran without a lock).
    """
    _on_fake(fake_microscope, _plate_at())
    autofocus = fake_microscope.require(Autofocus)
    assert autofocus.engage() is True
    fake_core.log.clear()
    stop = threading.Event()

    def stop_at_120(progress: Progress) -> None:
        if progress.done == 120:
            stop.set()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        first = _run(
            fake_microscope,
            tmp_path,
            plates=["12-well"],
            on_progress=stop_at_120,
            stop=stop,
        )
        assert first.status is StageStatus.stopped
        assert autofocus.is_engaged() is False
        record = _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    assert _result(record).assumed == (NO_LIMITS_LINE, AUTOFOCUS_OFF_LINE)
    assert "enableContinuousFocus(True)" not in fake_core.log
    # Logged once, by the attempt that switched it off.
    assert [r.getMessage() for r in caplog.records if r.name == LOGGER] == [
        AUTOFOCUS_LINE
    ]


def test_a_stand_without_an_autofocus_says_nothing_about_it(
    fake_core: FakeCore, tmp_path: Path
) -> None:
    fake_core.unloadDevice("Autofocus")
    microscope = Microscope.from_core(cast("Any", fake_core), Profile.demo())
    try:
        assert microscope.get(Autofocus) is None
        _on_fake(microscope, _plate_at())

        record = _run(microscope, tmp_path, plates=["12-well"])

        assert _result(record).assumed == (NO_LIMITS_LINE,)
    finally:
        microscope.close()


def test_an_autofocus_engaged_again_before_the_end_is_not_reported_off(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """What is said is what the autofocus is at the end, read: a flag set at the switch-off would say off here.

    The operator locked it again between the switch-off and the end of the
    survey: the scan will find it on, and a line saying it is off would send the
    operator to lock a PFS that is already locked.
    """
    spy = _AutofocusReadAtTheEnd(engaged=True, at_the_end=True)
    fake_microscope.override(Autofocus, spy)
    _on_fake(fake_microscope, _plate_at())

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    assert spy.calls.count("disengage") == 1
    assert spy.calls.count("is_engaged") == 2
    assert _result(record).assumed == (NO_LIMITS_LINE,)


def test_an_autofocus_that_cannot_be_read_at_the_end_is_a_finding_not_a_lost_survey(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A survey of 208 points is not thrown away for a state that could not be read: it is said, in one line."""
    spy = _AutofocusReadAtTheEnd(
        engaged=True, at_the_end=HardwareError("pfs did not answer\nretry")
    )
    fake_microscope.override(Autofocus, spy)
    _on_fake(fake_microscope, _plate_at())

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    assert _result(record).assumed == (
        NO_LIMITS_LINE,
        "the autofocus's state could not be read after the survey (HardwareError: "
        "pfs did not answer); check it before the scan",
    )


def test_a_halt_while_the_autofocus_is_read_at_the_end_is_not_swallowed(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-64: the halt is a ``HardwareError``, and an end read that turned it into a line would end ``done`` on a halted stand.

    The survey is complete and checkpointed before the read, so ``--resume``
    fits it again without moving.
    """
    spy = _AutofocusReadAtTheEnd(engaged=True, at_the_end=MicroscopeHaltedError())
    fake_microscope.override(Autofocus, spy)
    camera = _on_fake(fake_microscope, _plate_at())

    with pytest.raises(MicroscopeHaltedError):
        _run(fake_microscope, tmp_path, plates=["12-well"])

    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 208
    assert len(stored.checkpoint["points"]) == 208
    assert camera.snaps == 208


# --------------------------------------------------- the checkpoint and a resume


def test_a_failed_fit_keeps_the_survey_for_a_resume(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A featureless sample: with noise, the best of thousands of hypotheses on 208 points can pass 0.25 by chance."""
    camera = _on_fake(
        fake_microscope,
        _plate_at(
            well_level=800.0, plastic_level=800.0, texture_std=0.0, noise_std=0.0
        ),
    )

    with pytest.raises(PluginRunError) as info:
        _run(fake_microscope, tmp_path, plates=["12-well", "6-well"])

    assert "too low to trust" in str(info.value)
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.checkpoint is not None
    assert stored.checkpoint["total"] == 208
    assert stored.checkpoint["next_index"] == 208
    assert stored.checkpoint["skipped"] == 0
    assert len(stored.checkpoint["points"]) == 208
    assert camera.snaps == 208

    # The resume refits without moving: not a snap, not a move, not an autofocus
    # call. The autofocus is read after the fit's checks, so a refused fit reads
    # nothing either.
    spy = _Autofocus(engaged=True)
    fake_microscope.override(Autofocus, spy)
    fake_core.log.clear()
    with pytest.raises(PluginRunError) as again:
        _run(fake_microscope, tmp_path, resume=True, plates=["12-well", "6-well"])

    assert "too low to trust" in str(again.value)
    assert camera.snaps == 208
    assert fake_core.log == []
    assert spy.calls == []
    assert spy.engaged is True


def test_a_stopped_survey_resumes_where_its_checkpoint_left_it(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    sample = _plate_at((0.0, 0.0))
    camera = _on_fake(fake_microscope, sample)
    stop = threading.Event()

    def stop_at_120(progress: Progress) -> None:
        if progress.done == 120:
            stop.set()

    first = _run(
        fake_microscope,
        tmp_path,
        plates=["12-well"],
        on_progress=stop_at_120,
        stop=stop,
    )

    assert first.status is StageStatus.stopped
    assert camera.snaps == 120
    assert first.checkpoint is not None
    # Saved every 50 points: 100 handled, the 20 since are done again.
    assert first.checkpoint["next_index"] == 100
    assert len(first.checkpoint["points"]) == 100
    planned = survey_points("12-well", (0.0, 0.0))

    camera.positions.clear()
    snaps_before = camera.snaps
    record = _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    result = _result(record)
    assert camera.snaps - snaps_before == 108
    # The centre is the first run's, not where the stage stood when it was stopped.
    assert camera.positions[0] == pytest.approx(planned[100], abs=0.05)
    assert result.fit is not None
    assert result.fit.n_samples == 208
    assert result.plate == "12-well"


def test_a_survey_stopped_before_its_first_periodic_checkpoint_resumes_around_its_own_centre(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The checkpoint is saved before the first move, so a stop at point 30 keeps the centre (FM-111).

    Without it a resume would survey again around wherever the stage stopped:
    here 208 points from -112 mm to +21 mm in x, up to 46 mm past the plate's edge.
    """
    camera = _on_fake(fake_microscope, _plate_at((0.0, 0.0)))
    stop = threading.Event()

    def stop_at_30(progress: Progress) -> None:
        if progress.done == 30:
            stop.set()

    first = _run(
        fake_microscope,
        tmp_path,
        plates=["12-well"],
        on_progress=stop_at_30,
        stop=stop,
    )

    assert first.status is StageStatus.stopped
    assert camera.snaps == 30
    assert first.checkpoint is not None
    assert first.checkpoint["centre_um"] == [0.0, 0.0]
    # The save before the first move holds the survey's size too.
    assert first.checkpoint["total"] == 208
    assert first.checkpoint["next_index"] == 0
    assert _here(fake_microscope) != pytest.approx((0.0, 0.0), abs=1000.0)
    planned = survey_points("12-well", (0.0, 0.0))

    camera.positions.clear()
    record = _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    result = _result(record)
    assert camera.positions[0] == pytest.approx(planned[0], abs=0.05)
    assert _flat(camera.positions) == pytest.approx(_flat(planned), abs=0.05)
    assert result.fit is not None
    assert result.fit.n_samples == 208


def test_a_checkpoint_that_is_not_this_surveys_is_refused_before_anything_moves(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(
        fake_microscope,
        _plate_at(
            well_level=800.0, plastic_level=800.0, texture_std=0.0, noise_std=0.0
        ),
    )
    with pytest.raises(PluginRunError):
        _run(fake_microscope, tmp_path, plates=["12-well"])
    snaps = camera.snaps
    ending = "run plate-finding without --resume"

    _tamper(tmp_path, next_index=300)
    with pytest.raises(PluginRunError) as beyond:
        _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])
    _tamper(tmp_path, next_index=207)
    with pytest.raises(PluginRunError) as miscounted:
        _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    assert str(beyond.value) == (
        "plate-finding: the checkpoint has handled 300 points and this survey "
        f"has 208: {ending}"
    )
    assert str(miscounted.value) == (
        "plate-finding: the checkpoint says 207 points were handled but holds "
        f"208 samples and 0 skipped: {ending}"
    )
    assert camera.snaps == snaps


def _inside(
    planned: list[tuple[float, float]],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> int:
    """How many of ``planned`` lie on or inside ``limits``: the test's own count, not the plugin's."""
    (x_low, x_high), (y_low, y_high) = limits
    return sum(1 for x, y in planned if x_low <= x <= x_high and y_low <= y <= y_high)


NARROW = ((-60_000.0, 60_000.0), (-48_000.0, 48_000.0))  # keeps 154 of 208
WIDE = ((-64_000.0, 64_000.0), (-52_000.0, 52_000.0))  # keeps 173


@pytest.mark.parametrize(
    ("first", "second"),
    [(NARROW, WIDE), (WIDE, NARROW)],
    ids=["narrow-then-wide", "wide-then-narrow"],
)
def test_a_resume_on_other_soft_limits_is_refused_before_anything_moves(
    open_limited_fake: Callable[..., Microscope],
    tmp_path: Path,
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """``next_index`` indexes the points left after the limits: other limits, other indices, and nothing said (#248).

    In both directions ``next_index`` (50) is below both totals, so the check
    that the checkpoint is not beyond the survey cannot refuse: only the saved
    size can.
    """
    planned = survey_points("12-well", (0.0, 0.0))
    counts = {NARROW: 154, WIDE: 173}
    assert _inside(planned, first) == counts[first]
    assert _inside(planned, second) == counts[second]
    stand_one = open_limited_fake(first)
    camera_one = _on_fake(stand_one, _plate_at())
    stand_two = open_limited_fake(second)
    camera_two = _on_fake(stand_two, _plate_at())
    stop = threading.Event()

    def stop_at_60(progress: Progress) -> None:
        if progress.done == 60:
            stop.set()

    stopped = _run(
        stand_one,
        tmp_path,
        plates=["12-well"],
        centre_x_um=0.0,
        centre_y_um=0.0,
        on_progress=stop_at_60,
        stop=stop,
    )
    assert stopped.status is StageStatus.stopped
    assert stopped.checkpoint is not None
    assert stopped.checkpoint["next_index"] == 50
    before = _here(stand_two)

    with pytest.raises(PluginRunError) as refused:
        _run(
            stand_two,
            tmp_path,
            resume=True,
            plates=["12-well"],
            centre_x_um=0.0,
            centre_y_um=0.0,
        )

    assert str(refused.value) == (
        f"plate-finding: the checkpoint was made for a survey of {counts[first]} "
        f"points inside the soft limits and this one has {counts[second]}: the "
        "stage's soft limits changed; run plate-finding without --resume"
    )
    assert camera_two.snaps == 0
    assert _here(stand_two) == before

    # The refusal keeps the checkpoint: on the first limits again it resumes.
    kept = _stored(tmp_path)
    assert kept.status is StageStatus.failed
    assert kept.checkpoint is not None
    assert kept.checkpoint["total"] == counts[first]
    assert kept.checkpoint["next_index"] == 50
    record = _run(
        stand_one,
        tmp_path,
        resume=True,
        plates=["12-well"],
        centre_x_um=0.0,
        centre_y_um=0.0,
    )

    result = _result(record)
    assert result.fit is not None
    assert result.fit.n_samples == counts[first]
    assert camera_one.snaps == 60 + (counts[first] - 50)


def test_a_checkpoint_without_the_surveys_size_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A checkpoint from before the size was kept cannot say what its indices index; the runner refuses it."""
    camera = _on_fake(
        fake_microscope,
        _plate_at(
            well_level=800.0, plastic_level=800.0, texture_std=0.0, noise_std=0.0
        ),
    )
    with pytest.raises(PluginRunError):
        _run(fake_microscope, tmp_path, plates=["12-well"])
    snaps = camera.snaps
    _tamper(tmp_path, drop=("total",))

    with pytest.raises(ResumeError, match="total"):
        _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    assert camera.snaps == snaps


# ---------------------------------------------------- stop, halt and faults: FM-64


def test_a_halt_ends_the_survey_and_is_not_a_skipped_point(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-64: the operator's emergency stop is the stage's, not a field that failed.

    The halt is the layer's, so the stage is the real one: a wrapper installed
    with ``override`` would bypass the ``Executor`` that checks for it.
    """

    def halt_at_10(snap: int) -> None:
        if snap == 10:
            demo_microscope.stop()

    camera = _on_demo(demo_microscope, _plate_at(), before=halt_at_10)

    try:
        with pytest.raises(MicroscopeHaltedError):
            _run(demo_microscope, tmp_path, plates=["12-well"])
    finally:
        demo_microscope.resume()

    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    assert camera.snaps == 10


def test_a_stopped_move_ends_the_survey_and_is_not_a_skipped_point(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-64: ``MotionStoppedError`` is a ``HardwareError``, and the stage is not where the loop believes it is."""
    camera = _on_fake(fake_microscope, _plate_at())
    stage = _Stage(
        fake_microscope.require(XYStage),
        raises={20: MotionStoppedError("xy_stage XY")},
    )
    fake_microscope.override(XYStage, stage)

    with pytest.raises(MotionStoppedError):
        _run(fake_microscope, tmp_path, plates=["12-well"])

    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    # Nothing carried on to the next point.
    assert stage.moves == 20
    assert camera.snaps == 19


def test_an_error_that_is_not_the_stands_propagates_and_is_not_a_skipped_point(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Only a ``HardwareError`` is a point that failed: a programming error ends the run."""

    def broken_at_5(snap: int) -> None:
        if snap == 5:
            raise ValueError("a frame of the wrong kind")

    camera = _on_fake(fake_microscope, _plate_at(), before=broken_at_5)

    with pytest.raises(ValueError, match="a frame of the wrong kind"):
        _run(fake_microscope, tmp_path, plates=["12-well"])

    assert _stored(tmp_path).status is StageStatus.failed
    assert camera.snaps == 5


def test_a_stage_that_stops_short_without_an_error_is_not_believed(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A frame is placed where the stage says it was, not where it was sent.

    The stage reaches no further than 45 mm in x and reports where it is. Placed
    at the commanded positions the frames gave a fit of the 12-well plate 26 mm
    (a whole pitch) off, ``done``, with nothing in ``assumed``. Placed where they
    were taken the survey spans 90 mm in x, 14 mm short of what the plate needs,
    and is refused.
    """
    camera = _on_fake(fake_microscope, _plate_at())
    fake_microscope.override(
        XYStage, _Stage(fake_microscope.require(XYStage), clamp_x_um=45_000.0)
    )

    reason = _refusal(fake_microscope, tmp_path, plates=["12-well"])

    assert "A1 cannot be pinned down" in reason
    assert camera.snaps == 208
    assert max(x for x, _ in camera.positions) == pytest.approx(45_000.0, abs=0.05)


def test_a_readback_that_is_not_finite_is_a_skipped_point(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _on_fake(fake_microscope, _plate_at())
    fake_microscope.override(
        XYStage, _Stage(fake_microscope.require(XYStage), nan_readback_at=40)
    )

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    result = _result(record)
    assert result.assumed == (
        NO_LIMITS_LINE,
        "1 of 208 survey points could not be sampled and were skipped",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == 207


def test_a_few_failed_points_are_skipped_and_counted(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope, _plate_at())
    stage = _Stage(fake_microscope.require(XYStage), every=20)
    fake_microscope.override(XYStage, stage)

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    result = _result(record)
    # Moves 20, 40, ..., 200 of 208 time out.
    assert result.assumed == (
        NO_LIMITS_LINE,
        "10 of 208 survey points could not be sampled and were skipped",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == 198
    assert camera.snaps == 198
    assert stage.moves == 208


def test_exactly_a_tenth_of_the_points_failing_is_not_too_many(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """The 10 % is in integers: 18 of 180 is exactly a tenth, and 19 is more (``skipped * 10 > total``)."""
    _on_fake(fake_microscope, _plate_at())
    fake_microscope.override(
        XYStage, _Stage(fake_microscope.require(XYStage), every=10)
    )

    record = _run(fake_microscope, tmp_path, plates=["12-well"], step_um=9000.0)

    result = _result(record)
    assert result.assumed == (
        NO_LIMITS_LINE,
        "18 of 180 survey points could not be sampled and were skipped",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == 162


def test_failed_points_beyond_ten_percent_fail_the_survey(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope, _plate_at())
    fake_microscope.override(XYStage, _Stage(fake_microscope.require(XYStage), every=5))

    reason = _refusal(fake_microscope, tmp_path, plates=["12-well"])

    # Move 105 is the 21st failure, and 21 * 10 = 210 is the first count above 208.
    # It was sent to planned point 104, and that is the one named.
    x_um, y_um = survey_points("12-well", (0.0, 0.0))[104]
    assert reason == (
        "plate-finding: 21 of 208 survey points could not be sampled, the last at "
        f"({x_um:.1f}, {y_um:.1f}) µm: DeviceTimeoutError: xy_stage did not "
        "arrive within 30 s (move 105); check the stage, then run plate-finding "
        "--resume"
    )
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 105
    assert stored.checkpoint["skipped"] == 21
    assert len(stored.checkpoint["points"]) == 84
    assert camera.snaps == 84


def _timeout() -> DeviceTimeoutError:
    return DeviceTimeoutError("xy_stage did not arrive within 60 s\nretry")


def test_three_failed_moves_in_a_row_end_the_survey(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A stage that times out at every move failed at a tenth of the points: 1.9 h on a 96-well survey (#248)."""
    camera = _on_fake(fake_microscope, _plate_at())
    timeout = _timeout()
    stage = _Stage(
        fake_microscope.require(XYStage), raises={5: timeout, 6: timeout, 7: timeout}
    )
    fake_microscope.override(XYStage, stage)

    reason = _refusal(fake_microscope, tmp_path, plates=["12-well"])

    # Move 7 was sent to planned point 6.
    x_um, y_um = survey_points("12-well", (0.0, 0.0))[6]
    assert reason == (
        "plate-finding: 3 survey points in a row could not be sampled, the last at "
        f"({x_um:.1f}, {y_um:.1f}) µm: DeviceTimeoutError: xy_stage did not arrive "
        "within 60 s; check the stage, then run plate-finding --resume"
    )
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 7
    assert stored.checkpoint["skipped"] == 3
    assert len(stored.checkpoint["points"]) == 4
    assert stage.moves == 7
    assert camera.snaps == 4

    # The resume starts its count at 0: a count carried over from the first
    # attempt (3) would end it at its first failure. Points 7 and 8 fail, and 9
    # is the first one sampled.
    again = _Stage(fake_microscope.require(XYStage), raises={1: timeout, 2: timeout})
    fake_microscope.override(XYStage, again)
    camera.positions.clear()
    record = _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    result = _result(record)
    assert result.assumed == (
        NO_LIMITS_LINE,
        "5 of 208 survey points could not be sampled and were skipped",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == 203
    assert camera.positions[0] == pytest.approx(
        survey_points("12-well", (0.0, 0.0))[9], abs=0.05
    )


def test_a_dead_stage_ends_the_survey_at_the_bound_not_at_the_tenth(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Every move times out: 3 moves, where the tenth of the points took 21 (and 113 on a 96-well plate)."""
    camera = _on_fake(fake_microscope, _plate_at())
    stage = _Stage(fake_microscope.require(XYStage), every=1)
    fake_microscope.override(XYStage, stage)

    reason = _refusal(fake_microscope, tmp_path, plates=["12-well"])

    assert "3 survey points in a row could not be sampled" in reason
    assert stage.moves == 3
    assert camera.snaps == 0
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 3
    assert stored.checkpoint["skipped"] == 3
    assert stored.checkpoint["points"] == []


def test_a_point_that_is_sampled_resets_the_count(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Two in a row twice is not three in a row."""
    _on_fake(fake_microscope, _plate_at())
    timeout = _timeout()
    stage = _Stage(
        fake_microscope.require(XYStage),
        raises={5: timeout, 6: timeout, 9: timeout, 10: timeout},
    )
    fake_microscope.override(XYStage, stage)

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    result = _result(record)
    assert result.assumed == (
        NO_LIMITS_LINE,
        "4 of 208 survey points could not be sampled and were skipped",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == 204


def test_max_failed_in_row_is_the_parameter_not_a_constant(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _on_fake(fake_microscope, _plate_at())
    timeout = _timeout()
    stage = _Stage(fake_microscope.require(XYStage), raises={5: timeout})
    fake_microscope.override(XYStage, stage)

    reason = _refusal(
        fake_microscope, tmp_path, plates=["12-well"], max_failed_in_row=1
    )

    assert "1 survey points in a row could not be sampled" in reason
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 5

    # Four in a row is under 5, and 4 of 208 is under a tenth.
    four = {5: timeout, 6: timeout, 7: timeout, 8: timeout}
    fake_microscope.override(
        XYStage, _Stage(fake_microscope.require(XYStage), raises=four)
    )
    record = _run(
        fake_microscope,
        tmp_path,
        name="five.json",
        plates=["12-well"],
        max_failed_in_row=5,
    )

    result = _result(record)
    assert result.fit is not None
    assert result.fit.n_samples == 204


def test_every_kind_of_failed_point_counts_in_a_row(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A move that raises, a frame that is ``nan`` and a readback that is ``nan``: three in a row, whichever kind.

    A failed move has no snap, so move 21's is snap 20.
    """

    def nan_at_20(snap: int, frame: npt.NDArray[np.uint16]) -> npt.NDArray[Any]:
        if snap == 20:
            return np.full(frame.shape, np.nan, dtype=np.float32)
        return frame

    _on_fake(fake_microscope, _plate_at(), spoil=nan_at_20)
    fake_microscope.override(
        XYStage,
        _Stage(
            fake_microscope.require(XYStage),
            raises={20: _timeout()},
            nan_readback_at=22,
        ),
    )

    reason = _refusal(fake_microscope, tmp_path, plates=["12-well"])

    assert "3 survey points in a row could not be sampled" in reason
    assert reason.endswith(
        ": the stage's readback is not a finite position; check the stage, then "
        "run plate-finding --resume"
    )
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 22
    assert stored.checkpoint["skipped"] == 3


def test_a_frame_value_that_is_not_finite_is_a_skipped_point(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A ``nan`` point would be accepted by the checkpoint model when built and refused when read back (m3 §6.2)."""

    def nan_at_30(snap: int, frame: npt.NDArray[np.uint16]) -> npt.NDArray[Any]:
        if snap == 30:
            return np.full(frame.shape, np.nan, dtype=np.float32)
        return frame

    camera = _on_fake(fake_microscope, _plate_at(), spoil=nan_at_30)
    stop = threading.Event()

    def stop_at_60(progress: Progress) -> None:
        if progress.done == 60:
            stop.set()

    first = _run(
        fake_microscope,
        tmp_path,
        plates=["12-well"],
        on_progress=stop_at_60,
        stop=stop,
    )

    assert first.status is StageStatus.stopped
    # The document loads back: the checkpoint holds the skip and not a nan.
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 50
    assert stored.checkpoint["skipped"] == 1
    assert len(stored.checkpoint["points"]) == 49
    assert all(math.isfinite(p["value"]) for p in stored.checkpoint["points"])

    record = _run(fake_microscope, tmp_path, resume=True, plates=["12-well"])

    result = _result(record)
    assert result.assumed == (
        NO_LIMITS_LINE,
        "1 of 208 survey points could not be sampled and were skipped",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == 207
    assert camera.snaps == 60 + 158


# ------------------------------------------------------------- the soft limits


def test_points_outside_the_soft_limits_are_not_visited(
    open_limited: Callable[..., Microscope], tmp_path: Path
) -> None:
    limits = ((-60_000.0, 60_000.0), (-48_000.0, 48_000.0))
    microscope = open_limited(limits)
    camera = _on_demo(microscope, _plate_at())
    planned = survey_points("12-well", (0.0, 0.0))
    inside = [
        p
        for p in planned
        if -60_000.0 <= p[0] <= 60_000.0 and -48_000.0 <= p[1] <= 48_000.0
    ]
    assert survey_extent_note(inside, "12-well") is None
    assert len(planned) - len(inside) > 0

    record = _run(
        microscope,
        tmp_path,
        plates=["12-well"],
        centre_x_um=0.0,
        centre_y_um=0.0,
    )

    result = _result(record)
    assert len(camera.positions) == len(inside)
    assert all(
        -60_000.0 <= x <= 60_000.0 and -48_000.0 <= y <= 48_000.0
        for x, y in camera.positions
    )
    assert result.assumed == (
        f"{len(planned) - len(inside)} of {len(planned)} survey points are "
        "outside the stage's soft limits and were not sampled",
        AUTOFOCUS_OFF_LINE,
    )
    assert result.fit is not None
    assert result.fit.n_samples == len(inside)


def test_a_survey_the_soft_limits_cut_is_refused_before_moving(
    open_limited: Callable[..., Microscope], tmp_path: Path
) -> None:
    microscope = open_limited(((-40_000.0, 40_000.0), (-30_000.0, 30_000.0)))
    camera = _on_demo(microscope, _plate_at())
    spy = _Autofocus(engaged=True)
    microscope.override(Autofocus, spy)
    before = _here(microscope)

    reason = _refusal(microscope, tmp_path, plates=["12-well"])

    assert reason.startswith("plate-finding: the survey spans ")
    assert reason.endswith(
        "; the stage's soft limits x [-40000, 40000], y [-30000, 30000] µm cut the "
        "survey: put the plate nearer the middle of the travel, or change "
        "[safety] if the limits are wrong"
    )
    assert camera.snaps == 0
    assert _here(microscope) == before
    assert not MUTATING & set(spy.calls)
    assert spy.engaged is True


@pytest.mark.parametrize(
    ("limits", "limits_text"),
    [
        # Wide enough in x for the 12-well plate (130 mm, it needs 104), cut in y.
        (
            ((-70_000.0, 70_000.0), (-30_000.0, 30_000.0)),
            "x [-70000, 70000], y [-30000, 30000]",
        ),
        # The reverse: long enough in y (104 mm, it needs 78), cut in x.
        (
            ((-40_000.0, 40_000.0), (-60_000.0, 60_000.0)),
            "x [-40000, 40000], y [-60000, 60000]",
        ),
    ],
    ids=["cut-in-y", "cut-in-x"],
)
def test_a_survey_the_limits_cut_on_one_axis_only_is_refused_before_moving(
    open_limited: Callable[..., Microscope],
    tmp_path: Path,
    limits: tuple[tuple[float, float], tuple[float, float]],
    limits_text: str,
) -> None:
    microscope = open_limited(limits)
    camera = _on_demo(microscope, _plate_at())
    before = _here(microscope)

    reason = _refusal(microscope, tmp_path, plates=["12-well"])

    assert reason.startswith("plate-finding: the survey spans ")
    assert f"; the stage's soft limits {limits_text} µm cut the survey: " in reason
    assert camera.snaps == 0
    assert _here(microscope) == before


def test_limits_that_cut_nothing_say_nothing_and_a_point_on_a_limit_is_visited(
    open_limited_fake: Callable[..., Microscope], tmp_path: Path
) -> None:
    """The layer's own test is ``low <= v <= high``: a point exactly on a limit is inside.

    The limits are the survey's own extremes, so every planned point is on or
    inside them; a strict comparison would drop the points on the edge, and a
    survey that cuts nothing would still claim that the limits cut some.
    """
    planned = survey_points("12-well", (0.0, 0.0))
    xs = [p[0] for p in planned]
    ys = [p[1] for p in planned]
    microscope = open_limited_fake(((min(xs), max(xs)), (min(ys), max(ys))))
    camera = _on_fake(microscope, _plate_at())

    record = _run(microscope, tmp_path, plates=["12-well"])

    result = _result(record)
    assert result.assumed == (AUTOFOCUS_OFF_LINE,)
    assert camera.snaps == 208
    assert result.fit is not None
    assert result.fit.n_samples == 208


def test_a_survey_with_no_point_inside_the_limits_is_refused(
    open_limited: Callable[..., Microscope], tmp_path: Path
) -> None:
    microscope = open_limited(((-1000.0, 1000.0), (-1000.0, 1000.0)))
    camera = _on_demo(microscope, _plate_at())

    reason = _refusal(
        microscope,
        tmp_path,
        plates=["12-well"],
        centre_x_um=200_000.0,
        centre_y_um=0.0,
    )

    assert reason.startswith("plate-finding: only 0 of 208 survey points are inside")
    assert "soft limits x [-1000, 1000]" in reason
    assert camera.snaps == 0


# ---------------------------------------------- refused before anything moves


def test_a_survey_without_margin_is_refused_before_moving(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope, _plate_at())

    reason = _refusal(
        fake_microscope, tmp_path, plate="12-well", margin_pitches=0.0, jitter=0.0
    )

    # Without margin the survey spans 78 x 52 mm, and the plate needs 104 x 78.
    assert reason == (
        "plate-finding: the survey spans 78x52 mm; this plate needs 104x78 mm — "
        "one pitch of margin past the outer wells — or which well is A1 cannot "
        "be pinned down"
    )
    assert camera.snaps == 0


def test_a_survey_of_too_many_points_is_refused_before_moving(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    camera = _on_fake(fake_microscope, _plate_at())
    before = _here(fake_microscope)

    reason = _refusal(fake_microscope, tmp_path, plate="12-well", step_um=1.0)

    count = (130_000 + 1) * (104_000 + 1)
    assert reason == (
        f"plate-finding: the survey would take {count} points, more than 25000: "
        "give a larger step_um or a smaller margin_pitches"
    )
    assert camera.snaps == 0
    assert _here(fake_microscope) == before


def test_a_dry_run_refuses_to_survey(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    before = _here(demo_microscope_dry)

    reason = _refusal(demo_microscope_dry, tmp_path)

    assert reason == (
        "plate-finding: a dry run cannot survey the plate; run it without --dry-run"
    )
    assert _here(demo_microscope_dry) == before


# --------------------------------------------------- the fit is judged after it


def test_a_winner_larger_than_the_survey_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A 24-well survey 0.52 pitch past its wells passes the 24-well plate's extent and not the 6-well plate's.

    The sample is a 6-well plate: its fit is usable (0.76 in the prototype), and
    A1 of a plate whose last wells the survey does not pass is exactly what the
    fit cannot tell. The mutant that drops the check after the fit must fail here.
    """
    centre = (1000.0, -2000.0)
    sample = PlateSample(
        "6-well",
        (centre[0] - 39_120.0 + 300.0, centre[1] + 19_560.0 - 200.0),
        0.8,
    )
    camera = _on_fake(fake_microscope, sample)
    planned = survey_points("24-well", centre, margin_pitches=0.52, jitter=0.0)
    assert survey_extent_note(planned, "24-well") is None
    assert survey_extent_note(planned, "6-well") is not None
    params: dict[str, Any] = {
        "plate": "24-well",
        "plates": ["6-well"],
        "centre_x_um": centre[0],
        "centre_y_um": centre[1],
        "jitter": 0.0,
    }

    reason = _refusal(fake_microscope, tmp_path, margin_pitches=0.52, **params)

    assert "A1 cannot be pinned down" in reason
    assert reason.startswith("plate-finding: 6-well fits best, but the survey spans ")
    assert reason.endswith(": run plate-finding with plate=6-well")
    assert camera.snaps == len(planned) == 247

    # Control: with a full pitch of margin the same plate is found.
    control = _run(fake_microscope, tmp_path, name="control.json", **params)

    result = _result(control)
    assert result.plate == "6-well"
    assert result.fit is not None
    assert result.fit.runner_up is None
    assert result.fit.margin is None


# ------------------------------------------------------- the statistic


def test_the_std_statistic_reaches_the_fit(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    """A well is a checkerboard of 600 and 1400 (mean exactly 1000, std 400), plastic a flat 1000.

    The mean cannot tell them apart and the standard deviation can. Exact means
    keep the ``mean`` case deterministic: random noise could correlate with the
    lattice by chance (FM-124).
    """
    sample = _plate_at()
    checker = ((np.indices(SHAPE).sum(axis=0) % 2) * 800 + 600).astype(np.uint16)

    class _Textured(_Camera):
        def snap(self) -> npt.NDArray[Any]:
            frame = super().snap()
            x_um, y_um = self.positions[-1]
            if sample.well_at(x_um, y_um) is None:
                return np.full_like(frame, 1000)
            return checker

    fake_microscope.override(Camera, _Textured(fake_microscope, sample))

    with_std = _run(
        fake_microscope, tmp_path, plates=["12-well", "6-well"], statistic="std"
    )
    result = _result(with_std)
    with pytest.raises(PluginRunError) as with_mean:
        _run(
            fake_microscope,
            tmp_path,
            name="mean.json",
            plates=["12-well", "6-well"],
            statistic="mean",
        )

    assert result.plate == "12-well"
    assert "too low to trust" in str(with_mean.value)


# ------------------------------------------------------------- the parameters


def test_a_centre_given_by_one_coordinate_is_refused() -> None:
    with pytest.raises(
        ValidationError, match="give centre_x_um and centre_y_um together"
    ):
        PlateFindingParams(centre_x_um=1000.0)
    with pytest.raises(
        ValidationError, match="give centre_x_um and centre_y_um together"
    ):
        PlateFindingParams(centre_y_um=1000.0)
    assert PlateFindingParams(centre_x_um=1.0, centre_y_um=2.0).centre_x_um == 1.0


def test_an_unknown_or_non_well_plate_is_refused() -> None:
    wells = "12-well, 1536-well, 24-well, 384-well, 48-well, 6-well, 96-well"

    with pytest.raises(ValidationError) as non_well:
        PlateFindingParams(plates=["coverslip-18mm-square"])
    with pytest.raises(ValidationError) as unknown:
        PlateFindingParams(plate="glass")
    with pytest.raises(ValidationError) as twice:
        PlateFindingParams(plates=["96-well", "384-well", "96-well"])

    assert (
        f"plate-finding fits well plates: coverslip-18mm-square is not one of {wells}"
        in str(non_well.value)
    )
    assert f"plate-finding fits well plates: glass is not one of {wells}" in str(
        unknown.value
    )
    assert "plate-finding fits each plate once: 96-well is given twice" in str(
        twice.value
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"step_um": 0.0},
        {"step_um": 20_001.0},
        {"margin_pitches": -0.1},
        {"margin_pitches": 5.1},
        {"jitter": 1.1},
        {"seed": -1},
        {"centre_x_um": 2e6, "centre_y_um": 0.0},
        {"statistic": "median"},
        {"max_failed_in_row": 0},
        {"max_failed_in_row": 101},
    ],
)
def test_parameters_out_of_bounds_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PlateFindingParams(**bad)
