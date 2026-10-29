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
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from smc.hardware import Microscope
from smc.hardware.capabilities import XY, Autofocus, Camera, FocusStep, Limits, XYStage
from smc.hardware.errors import DeviceTimeoutError, MicroscopeHaltedError
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, Progress, load, run_plugin
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
AUTOFOCUS_LINE = (
    "the autofocus was switched off for the survey; engage it again before the scan"
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
    """The stand's XY stage, except that every ``every``-th move times out; the others delegate.

    It implements the seven methods of ``XYStage``: ``Microscope.override``
    refuses an implementation that lacks one. It bypasses the ``Executor``, so it
    is never used where a halt is the point.
    """

    def __init__(self, inner: XYStage, *, every: int) -> None:
        self._inner = inner
        self._every = every
        self.moves = 0
        self.calls: list[str] = []

    def position_um(self) -> XY:
        self.calls.append("position_um")
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        if self.moves % self._every == 0:
            raise DeviceTimeoutError(
                f"xy_stage did not arrive within 30 s (move {self.moves})\nretry"
            )
        return self._inner.move_to_um(x_um, y_um)

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


def _tamper(tmp_path: Path, **checkpoint: Any) -> None:
    """Replace fields of the stored checkpoint, as a hand-edited or foreign run document would."""
    path = tmp_path / "run.json"
    record = RunDocument.load(path).stages["plate-finding"]
    assert record.checkpoint is not None
    write_stage(
        path,
        record.model_copy(update={"checkpoint": {**record.checkpoint, **checkpoint}}),
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
    assert result.assumed == ()
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
    assert AUTOFOCUS_LINE in _result(record).assumed
    assert [r.getMessage() for r in caplog.records if r.name == LOGGER] == [
        AUTOFOCUS_LINE
    ]


def test_an_autofocus_that_is_off_is_left_alone(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    spy = _Autofocus(engaged=False)
    fake_microscope.override(Autofocus, spy)
    _on_fake(fake_microscope, _plate_at())

    record = _run(fake_microscope, tmp_path, plates=["12-well"])

    assert not MUTATING & set(spy.calls)
    assert _result(record).assumed == ()


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
    assert stored.checkpoint["next_index"] == 208
    assert stored.checkpoint["skipped"] == 0
    assert len(stored.checkpoint["points"]) == 208
    assert camera.snaps == 208

    # The resume refits without moving: not a snap, not a move, not an autofocus call.
    spy = _Autofocus(engaged=True)
    fake_microscope.override(Autofocus, spy)
    fake_core.log.clear()
    with pytest.raises(PluginRunError) as again:
        _run(fake_microscope, tmp_path, resume=True, plates=["12-well", "6-well"])

    assert "too low to trust" in str(again.value)
    assert camera.snaps == 208
    assert fake_core.log == []
    assert not MUTATING & set(spy.calls)
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
        "10 of 208 survey points could not be sampled and were skipped",
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
        "18 of 180 survey points could not be sampled and were skipped",
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
    assert reason.startswith(
        "plate-finding: 21 of 208 survey points could not be sampled, the last at ("
    )
    assert reason.endswith(
        "µm: DeviceTimeoutError: xy_stage did not arrive within 30 s (move 105); "
        "check the stage, then run plate-finding --resume"
    )
    stored = _stored(tmp_path)
    assert stored.checkpoint is not None
    assert stored.checkpoint["next_index"] == 105
    assert stored.checkpoint["skipped"] == 21
    assert len(stored.checkpoint["points"]) == 84
    assert camera.snaps == 84


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
        "1 of 208 survey points could not be sampled and were skipped",
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
    ],
)
def test_parameters_out_of_bounds_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PlateFindingParams(**bad)
