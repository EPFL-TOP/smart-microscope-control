"""``timing`` on the demo devices, on a clock of the test's own, and on ``FakeCore`` (m3 §12, #24).

Three kinds of test, because a timing has three things to get wrong:

* **The demo** runs the whole loop on Micro-Manager's devices. What a demo move
  costs depends on the runner (a 100 and a 400 µm move read the same on the macOS
  job), so no duration is asserted there: order and presence only (FM-44).
* **A fake clock** (``_clock`` is read at call time) and a stage that advances it
  by a known cost make the fit exact, and the targets of the moves are asserted
  against a list written here and not read from the code (FM-124). These run on
  ``fake_microscope``, whose stage arrives at once, so a 9 mm probe costs
  nothing.
* **``FakeCore``** drives the autofocus: a PFS that never locks, one that locks
  and then does not, one switched off from outside (FM-129), a halt during the
  engage (FM-64).

What the plugin must never do after a stop, a halt or a fault (FM-88) is a
count of calls in a log: no move after the stop, no engage, no select back.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Channels,
    Limits,
    XYStage,
)
from smc.hardware.errors import (
    HardwareError,
    MicroscopeHaltedError,
    SafetyRefusedError,
)
from smc.hardware.profile import AutofocusSection, Profile
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.slots import TimingsResult
from smc.plugins.timing import plugin as timing_plugin
from smc.plugins.timing.algorithm import ASSUMED_REFOCUS_MS, describe_timings
from smc.plugins.timing.plugin import TimingParams, TimingPlugin
from smc.run import RunDocument, StageRecord, StageStatus
from smc.testing import FakeCore

pytestmark = pytest.mark.demo

LOGGER = "smc.plugins.timing"
#: Short probes keep the demo's 10 mm/s stage quick: 100 and 400 µm take 50 ms in all.
SHORT = (100.0, 400.0)
NO_LIMITS = (
    "the stage reports no soft limits: the probe moves were not checked "
    "against its travel"
)
LEFT_OFF = (
    "the autofocus was switched off for the measurement and did not lock "
    "again; engage it again before the scan"
)


class _Clock:
    """A clock the test owns: it moves only when a stand-in says so."""

    def __init__(self) -> None:
        self.now_s = 0.0

    def __call__(self) -> float:
        return self.now_s

    def advance_ms(self, ms: float) -> None:
        self.now_s += ms / 1000.0


class _CostedStage:
    """The stand's stage, except that every move advances a clock by a known cost.

    The cost of a move is ``overhead_ms + ms_per_mm * distance / 1000``, the
    distance being the one the stage really goes (from where it stands to where
    it lands), and ``extra_ms[n]`` more for the n-th move. It implements the
    seven methods of ``XYStage`` (``Microscope.override`` refuses an
    implementation that lacks one) and keeps ``moves`` and ``targets`` so that a
    test can say what was asked and that nothing was asked after a stop.

    * ``land``: where the stage really goes for a commanded X (a stage that
      stops short, FM-142); the readback is where it went.
    * ``land_y``: the Y it lands at for a commanded ``(x, y)``.
    * ``on_move``: called with the move's number before the move is made.
    * ``fault_after``: the n-th move is made and then raises ``HardwareError``.
    * ``limits``: what ``limits_um`` answers.
    * ``start_reads``: what ``position_um`` answers instead of the stage's.
    * ``travelling``: what ``position_um`` answers until ``wait()`` is called
      (a stage still moving, FM-85); ``waits`` counts the calls.
    * ``unreadable_after``: ``position_um`` raises once this many moves were made.
    """

    def __init__(
        self,
        real: XYStage,
        clock: _Clock,
        *,
        overhead_ms: float = 100.0,
        ms_per_mm: float = 50.0,
        extra_ms: dict[int, float] | None = None,
        limits: tuple[Limits, Limits] | None = None,
        land: Callable[[float], float] | None = None,
        land_y: Callable[[float, float], float] | None = None,
        on_move: Callable[[int], None] | None = None,
        fault_after: int | None = None,
        start_reads: XY | None = None,
        travelling: XY | None = None,
        unreadable_after: int | None = None,
    ) -> None:
        self._real = real
        self._clock = clock
        self._overhead_ms = overhead_ms
        self._ms_per_mm = ms_per_mm
        self._extra_ms = extra_ms or {}
        self._limits = limits
        self._land = land
        self._land_y = land_y
        self._on_move = on_move
        self._fault_after = fault_after
        self._start_reads = start_reads
        self._travelling = travelling
        self._unreadable_after = unreadable_after
        self.moves = 0
        self.waits = 0
        self.targets: list[tuple[float, float]] = []

    def position_um(self) -> XY:
        if self._unreadable_after is not None and self.moves >= self._unreadable_after:
            raise HardwareError("the position cannot be read")
        if self._start_reads is not None:
            return self._start_reads
        if self._travelling is not None and self.waits == 0:
            return self._travelling
        return self._real.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        self.targets.append((x_um, y_um))
        here = self._real.position_um()
        landed_x_um = x_um if self._land is None else self._land(x_um)
        landed_y_um = y_um if self._land_y is None else self._land_y(x_um, y_um)
        distance_mm = (
            math.hypot(landed_x_um - here.x_um, landed_y_um - here.y_um) / 1000.0
        )
        self._clock.advance_ms(
            self._overhead_ms
            + self._ms_per_mm * distance_mm
            + self._extra_ms.get(self.moves, 0.0)
        )
        if self._on_move is not None:
            self._on_move(self.moves)
        arrived = self._real.move_to_um(landed_x_um, landed_y_um)
        if self.moves == self._fault_after:
            raise HardwareError("the stage answered and then faulted")
        return arrived

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        return self._real.move_by_um(dx_um, dy_um, force=force)

    def wait(self, timeout_s: float | None = None) -> None:
        self.waits += 1
        self._real.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._real.is_busy()

    def stop(self) -> None:
        self._real.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._limits


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        TimingPlugin(),
        microscope,
        TimingParams.model_validate(params),
        tmp_path / "run.json",
        stop=stop,
        on_progress=on_progress,
    )


def _result(record: StageRecord) -> TimingsResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return TimingsResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["timing"]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno == logging.WARNING
    ]


def _autofocus_calls(core: FakeCore) -> list[str]:
    return [line for line in core.log if line.startswith("enableContinuousFocus")]


def _engaged(core: FakeCore) -> None:
    """The autofocus on and locked, and the log cleared: what an operator does before a timing."""
    core.enableContinuousFocus(True)
    core.log.clear()


def _with_clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(timing_plugin, "_clock", clock)
    return clock


def _costed(microscope: Microscope, clock: _Clock, **kwargs: Any) -> _CostedStage:
    stage = _CostedStage(microscope.require(XYStage), clock, **kwargs)
    microscope.override(XYStage, stage)
    return stage


def _recording_channels(microscope: Microscope) -> list[str]:
    """Replace ``Channels`` by a copy that records the names it is asked to select."""
    real = microscope.require(Channels)
    seen: list[str] = []

    def select(name: str) -> str | None:
        seen.append(name)
        return real.select(name)

    patched = copy.copy(real)
    patched.select = select  # type: ignore[method-assign]
    microscope.override(Channels, patched)
    return seen


@contextmanager
def _opened(core: FakeCore, profile: Profile | None = None) -> Iterator[Microscope]:
    microscope = Microscope.from_core(
        cast("Any", core), profile if profile is not None else Profile.demo()
    )
    try:
        yield microscope
    finally:
        microscope.close()


def _without(
    microscope: Microscope, monkeypatch: pytest.MonkeyPatch, capability: type
) -> None:
    """The stand as if no device filled this capability's role."""
    real_get = microscope.get
    monkeypatch.setattr(
        microscope,
        "get",
        lambda wanted: None if wanted is capability else real_get(wanted),
    )


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_timing() -> None:
    entry = load("timing")

    assert entry.plugin.provides == "timings"
    assert entry.plugin.requires == (XYStage, Camera)
    assert entry.plugin.reads == ()
    assert entry.plugin.reads_optional == ()
    assert entry.plugin.Result is TimingsResult
    assert entry.target == "smc.plugins.timing.plugin:TimingPlugin"


# ------------------------------------------------------------------ the demo


def test_the_stage_is_left_where_it_was_found(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A measurement that moved the sample would be worse than useless (FM-85: ``move_to_um`` waits)."""
    stage = demo_microscope.require(XYStage)
    stage.move_to_um(1234.0, -567.0)
    before = stage.position_um()

    _run(demo_microscope, tmp_path, repeats=1, distances_um=SHORT)

    after = stage.position_um()
    assert after.x_um == pytest.approx(before.x_um, abs=0.1)
    assert after.y_um == pytest.approx(before.y_um, abs=0.1)


def test_a_measured_run_fills_the_slot_on_the_demo(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    autofocus = demo_microscope.require(Autofocus)
    assert autofocus.engage()

    record = _run(demo_microscope, tmp_path, repeats=1, distances_um=SHORT)

    result = _result(record)
    # A lock that was timed, not the 400 ms assumed: no `refocus` line, and the
    # autofocus sample is counted (4 stage + 1 autofocus + 1 bare snap). No
    # duration is asserted, not even `move_ms_per_mm > 0`: the macOS runner read
    # 0.0 for it, a 100 and a 400 µm move of the demo stage costing the same there
    # (FM-44). The fit is exact in the fake-clock tests below.
    assert result.refocus_ms is not None
    assert not any("refocus" in line for line in result.assumed)
    assert result.samples == 6
    assert NO_LIMITS in result.assumed
    # The slot, as the scan plan reads it.
    filling = RunDocument.load(tmp_path / "run.json").provider("timings")
    assert filling is not None
    assert filling.plugin == "timing"
    assert filling.result == record.result
    # Locked again at the end: the stand is as it was found.
    assert autofocus.is_engaged()


def test_every_given_channel_is_timed(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    assert demo_microscope.require(Autofocus).engage()

    record = _run(
        demo_microscope,
        tmp_path,
        repeats=1,
        distances_um=SHORT,
        channels=("FITC", "Cy5"),
    )

    result = _result(record)
    assert set(result.channel_ms) == {"FITC", "Cy5"}
    assert result.channel_switch_ms is not None
    # 4 stage + 1 autofocus + 1 bare snap + 2 channels x 1 snap + 2 switches.
    assert result.samples == 10


@pytest.mark.parametrize(
    ("found", "channels", "selected"),
    [
        # The found channel is not asked for: both are switched to, then it is restored.
        ("Rhodamine", ("Cy5", "DAPI"), ["Cy5", "DAPI", "Rhodamine"]),
        # The found channel is asked for: it goes last and nothing is selected after.
        ("DAPI", ("DAPI", "FITC"), ["FITC", "DAPI"]),
    ],
)
def test_the_channel_found_is_selected_back(
    demo_microscope: Microscope,
    tmp_path: Path,
    found: str,
    channels: tuple[str, ...],
    selected: list[str],
) -> None:
    demo_microscope.require(Channels).select(found)
    seen = _recording_channels(demo_microscope)

    _run(demo_microscope, tmp_path, repeats=1, distances_um=SHORT, channels=channels)

    assert seen == selected
    assert demo_microscope.require(Channels).current() == found


def test_a_single_channel_that_is_the_one_found_selects_nothing(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    demo_microscope.require(Channels).select("DAPI")
    seen = _recording_channels(demo_microscope)

    record = _run(
        demo_microscope,
        tmp_path,
        repeats=1,
        distances_um=SHORT,
        channels=("DAPI",),
    )

    result = _result(record)
    assert seen == []
    assert set(result.channel_ms) == {"DAPI"}
    assert result.channel_switch_ms is None
    assert (
        "no switch was timed: the one channel given is the one the stand was on"
        in result.assumed
    )


def test_without_channels_a_bare_snap_is_timed_and_no_switch(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seen = _recording_channels(demo_microscope)

    result = _result(_run(demo_microscope, tmp_path, repeats=1, distances_um=SHORT))

    assert result.snap_ms > 0.0
    assert result.channel_ms == {}
    assert result.channel_switch_ms is None
    assert seen == []
    assert not any("no switch was timed" in line for line in result.assumed)


def test_a_dry_run_is_refused(demo_microscope_dry: Microscope, tmp_path: Path) -> None:
    stage = _costed(demo_microscope_dry, _Clock())

    with pytest.raises(PluginRunError, match="--dry-run"):
        _run(demo_microscope_dry, tmp_path, repeats=1, distances_um=SHORT)

    assert stage.moves == 0
    assert _stored(tmp_path).status is StageStatus.failed


def test_progress_reports_each_step(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    assert demo_microscope.require(Autofocus).engage()
    seen: list[Progress] = []

    _run(
        demo_microscope,
        tmp_path,
        on_progress=seen.append,
        repeats=1,
        distances_um=SHORT,
        channels=("FITC", "Cy5"),
    )

    # Two distances, the autofocus (measured), the bare snap, two channels.
    assert [p.total for p in seen] == [6] * 6
    assert [p.done for p in seen] == [1, 2, 3, 4, 5, 6]
    assert [p.message for p in seen] == [
        "stage 100 µm",
        "stage 400 µm",
        "autofocus lock",
        "snap",
        "channel FITC",
        "channel Cy5",
    ]


def test_progress_counts_no_autofocus_step_when_it_is_not_measured(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The demo's autofocus is off, so it is assumed and not a step."""
    seen: list[Progress] = []

    _run(
        demo_microscope,
        tmp_path,
        on_progress=seen.append,
        repeats=1,
        distances_um=SHORT,
    )

    assert [p.total for p in seen] == [3, 3, 3]
    assert [p.done for p in seen] == [1, 2, 3]


def test_the_model_is_logged_at_info(
    demo_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = _result(_run(demo_microscope, tmp_path, repeats=1, distances_um=SHORT))

    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert logged[0] == describe_timings(result)[0]
    assert logged == describe_timings(result)


# ------------------------------------------------------------- a fake clock


def test_the_fit_recovers_a_stage_whose_costs_are_known(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overhead 100 ms and 50 ms/mm: a 300 µm move costs 115 ms, a 7000 µm one 450 ms (FM-124)."""
    clock = _with_clock(monkeypatch)
    fake_microscope.require(XYStage).move_to_um(1000.0, -250.0)
    stage = _costed(fake_microscope, clock)

    result = _result(
        _run(
            fake_microscope,
            tmp_path,
            repeats=2,
            distances_um=(300.0, 7000.0),
        )
    )

    assert result.move_overhead_ms == pytest.approx(100.0)
    assert result.move_ms_per_mm == pytest.approx(50.0)
    # Out and back, twice per distance, along X from the start, Y never changes.
    assert stage.targets == [
        (1300.0, -250.0),
        (1000.0, -250.0),
        (1300.0, -250.0),
        (1000.0, -250.0),
        (8000.0, -250.0),
        (1000.0, -250.0),
        (8000.0, -250.0),
        (1000.0, -250.0),
    ]
    # 8 stage + 2 bare snaps; the autofocus is off, and no channel was given.
    assert result.samples == 10
    assert result.snap_ms == 0.0  # the clock does not move during a snap


def test_one_slow_move_does_not_move_the_median(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first long outbound move (the 7th) is 5 s late: [450 x 5, 5450], the median stays 450."""
    clock = _with_clock(monkeypatch)
    _costed(fake_microscope, clock, extra_ms={7: 5000.0})

    result = _result(
        _run(
            fake_microscope,
            tmp_path,
            repeats=3,
            distances_um=(300.0, 7000.0),
        )
    )

    assert result.move_overhead_ms == pytest.approx(100.0)
    assert result.move_ms_per_mm == pytest.approx(50.0)


def test_the_probe_goes_the_other_way_when_the_soft_limits_say_so(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _with_clock(monkeypatch)
    stage = _costed(
        fake_microscope,
        clock,
        limits=(Limits(-10000.0, 2000.0), Limits(-5000.0, 5000.0)),
    )

    result = _result(
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))
    )

    assert stage.targets == [(-200.0, 0.0), (0.0, 0.0), (-9000.0, 0.0), (0.0, 0.0)]
    assert not any("soft limits" in line for line in result.assumed)
    assert result.move_ms_per_mm == pytest.approx(50.0)


def test_the_probe_goes_plus_x_when_both_ways_fit(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _with_clock(monkeypatch)
    stage = _costed(
        fake_microscope,
        clock,
        limits=(Limits(-10000.0, 10000.0), Limits(-5000.0, 5000.0)),
    )

    _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    assert stage.targets[0] == (200.0, 0.0)
    assert stage.targets[2] == (9000.0, 0.0)


def test_limits_that_fit_neither_way_refuse_before_anything_moves(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _with_clock(monkeypatch)
    stage = _costed(
        fake_microscope,
        clock,
        limits=(Limits(-5000.0, 5000.0), Limits(-5000.0, 5000.0)),
    )

    with pytest.raises(PluginRunError) as caught:
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    reason = str(caught.value)
    assert "9000" in reason
    assert "-5000" in reason
    assert "5000" in reason
    assert stage.moves == 0


@pytest.mark.parametrize(
    ("short_by_um", "accepted"),
    # 4.5 is above ``min(5 µm, 2 %)`` and 5.0 is on the edge of ``max``: the
    # tolerance is a closed bound; 6.0 is over it.
    [(1.0, True), (4.5, True), (5.0, True), (6.0, False)],
)
def test_a_stage_that_lands_a_little_short_is_judged_by_its_tolerance(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    short_by_um: float,
    accepted: bool,
) -> None:
    """The tolerance is ``max(5 µm, 2 %)``: 5 µm at 200, 180 at 9000, so 6 µm short fails only the hop."""
    clock = _with_clock(monkeypatch)
    _costed(
        fake_microscope,
        clock,
        land=lambda x: x - short_by_um if x > 0.0 else x,
    )

    if accepted:
        _result(
            _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))
        )
    else:
        with pytest.raises(PluginRunError, match="asked for 200 µm"):
            _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))


def test_a_stage_that_falls_short_in_proportion_is_judged_by_the_percentage(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1.5 % short is 3 µm at 200 and 135 µm at 9000: inside ``max(5 µm, 2 %)`` both times.

    The stage's cost follows the distance it really went, 197 and 8865 µm, so the
    fit through the *readback* distances recovers 100 ms and 50 ms/mm; one through
    the commanded 200 and 9000 gives another line (FM-142).
    """
    clock = _with_clock(monkeypatch)
    _costed(fake_microscope, clock, land=lambda x: x * 0.985)

    result = _result(
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))
    )

    assert result.move_overhead_ms == pytest.approx(100.0)
    assert result.move_ms_per_mm == pytest.approx(50.0)


def test_a_stage_that_goes_too_far_fails_the_run(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overshoot is no more a measurement of the distance asked than a short move is."""
    clock = _with_clock(monkeypatch)
    _costed(fake_microscope, clock, land=lambda x: x * 2.0)

    with pytest.raises(
        PluginRunError, match=r"the stage went 400\.0 µm when it was asked for 200 µm"
    ):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))


def test_a_stage_that_lands_off_in_y_fails_the_run(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distance the stage went is the straight line between readbacks, Y included."""
    clock = _with_clock(monkeypatch)
    _costed(
        fake_microscope,
        clock,
        land_y=lambda x, y: y + 200.0 if x > 0.0 else y,
    )

    with pytest.raises(
        PluginRunError,
        match=r"the stage went 282\.8 µm when it was asked for 200 µm",
    ):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))


def test_a_return_that_comes_back_short_is_warned_about(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """150 µm is inside 2 % of 9000 µm, so the probe's tolerance does not see it."""
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock)
    real = fake_microscope.require(XYStage)
    original = stage.move_to_um

    def move_to_um(x_um: float, y_um: float) -> XY:
        # Every return that starts more than 1 mm out stops 150 µm after the start.
        here = real.position_um()
        if x_um == 0.0 and here.x_um > 1000.0:
            return original(150.0, y_um)
        return original(x_um, y_um)

    stage.move_to_um = move_to_um  # type: ignore[method-assign]

    result = _result(
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))
    )

    assert result.move_ms_per_mm > 0.0
    assert _warnings(caplog) == [
        "the stage reads (150.0, 0.0) µm after its return, 150.0 µm from where it "
        "started (0.0, 0.0) µm: it came back short"
    ]


def test_a_return_that_comes_back_far_short_says_where_the_stage_is(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Beyond the tolerance the run fails, and its text no longer says the stage is at the start."""
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock)
    real = fake_microscope.require(XYStage)
    original = stage.move_to_um

    def move_to_um(x_um: float, y_um: float) -> XY:
        here = real.position_um()
        if x_um == 0.0 and here.x_um > 1000.0:
            return original(300.0, y_um)
        return original(x_um, y_um)

    stage.move_to_um = move_to_um  # type: ignore[method-assign]

    with pytest.raises(PluginRunError) as caught:
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    assert (
        "the stage was sent back to where it started and reads 300.0 µm from it"
        in str(caught.value)
    )
    assert _warnings(caplog) == [
        "the stage reads (300.0, 0.0) µm after its return, 300.0 µm from where it "
        "started (0.0, 0.0) µm: it came back short"
    ]


def test_a_return_within_the_tolerance_says_nothing(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """5 µm from the start, on the edge: the bound is closed and nothing is warned."""
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock)
    real = fake_microscope.require(XYStage)
    original = stage.move_to_um

    def move_to_um(x_um: float, y_um: float) -> XY:
        here = real.position_um()
        if x_um == 0.0 and here.x_um > 1000.0:
            return original(5.0, y_um)
        return original(x_um, y_um)

    stage.move_to_um = move_to_um  # type: ignore[method-assign]

    _result(_run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0)))

    assert _warnings(caplog) == []


def test_a_probe_that_ends_exactly_on_the_limit_still_fits_that_way(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The soft limits are a closed range: a target on the limit is inside it."""
    clock = _with_clock(monkeypatch)
    stage = _costed(
        fake_microscope,
        clock,
        limits=(Limits(-10000.0, 9000.0), Limits(-5000.0, 5000.0)),
    )

    _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    assert stage.targets[2] == (9000.0, 0.0)


def test_a_stage_that_stops_short_fails_the_run_after_it_is_back(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-142: a clamp at +500 µm read back as 500 is not a measurement of 9000.

    The run fails with how far the stage went, but only after the return, so the
    stage reads the start and the operator keeps their field.
    """
    clock = _with_clock(monkeypatch)
    real = fake_microscope.require(XYStage)
    real.move_to_um(1000.0, 0.0)
    stage = _costed(fake_microscope, clock, land=lambda x: min(x, 1500.0))

    with pytest.raises(PluginRunError) as caught:
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    assert "the stage went 500.0 µm when it was asked for 9000 µm" in str(caught.value)
    assert stage.targets[-1] == (1000.0, 0.0)
    assert stage.moves == 4
    assert real.position_um() == XY(1000.0, 0.0)
    record = _stored(tmp_path)
    assert record.status is StageStatus.failed
    assert record.result is None


def test_a_readback_that_is_not_a_number_fails_the_run(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _with_clock(monkeypatch)
    real = fake_microscope.require(XYStage)

    class _NanStage(_CostedStage):
        def move_to_um(self, x_um: float, y_um: float) -> XY:
            super().move_to_um(x_um, y_um)
            return XY(math.nan, y_um)

    microscope_stage = _NanStage(real, clock)
    fake_microscope.override(XYStage, microscope_stage)

    with pytest.raises(PluginRunError, match="not a number after a move of 200 µm"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    assert microscope_stage.moves == 4


def test_a_start_that_is_not_finite_refuses_before_anything_moves(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-125: a position read as ``nan`` is not a place to come back to."""
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock, start_reads=XY(math.nan, 0.0))

    with pytest.raises(PluginRunError, match="not a position"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=SHORT)

    assert stage.moves == 0


def test_a_stop_before_the_first_move_leaves_nothing_to_say(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock)
    stop = threading.Event()
    stop.set()

    record = _run(fake_microscope, tmp_path, stop=stop, repeats=1, distances_um=SHORT)

    assert record.status is StageStatus.stopped
    assert stage.moves == 0
    assert _warnings(caplog) == []


def test_a_stop_between_two_moves_leaves_the_stage_and_says_where(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-88: the stop comes during the long outbound move; the return is not sent."""
    clock = _with_clock(monkeypatch)
    stop = threading.Event()
    real = fake_microscope.require(XYStage)
    real.move_to_um(500.0, -100.0)
    stage = _costed(
        fake_microscope,
        clock,
        on_move=lambda n: stop.set() if n == 3 else None,
    )

    record = _run(
        fake_microscope,
        tmp_path,
        stop=stop,
        repeats=1,
        distances_um=(200.0, 3000.0),
    )

    assert record.status is StageStatus.stopped
    assert record.result is None
    assert stage.moves == 3
    assert real.position_um() == XY(3500.0, -100.0)
    assert _warnings(caplog) == [
        "the stage is left at (3500.0, -100.0) µm, not returned to (500.0, -100.0) µm"
    ]
    assert RunDocument.load(tmp_path / "run.json").provider("timings") is not None
    assert _stored(tmp_path).result is None


def test_a_halt_ends_the_measurement_and_is_not_a_failed_sample(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-64: the emergency stop at the third move reaches the caller; no later move is sent."""
    clock = _with_clock(monkeypatch)
    stage = _costed(
        fake_microscope,
        clock,
        on_move=lambda n: fake_microscope.stop() if n == 3 else None,
    )

    with pytest.raises(MicroscopeHaltedError):
        _run(fake_microscope, tmp_path, repeats=2, distances_um=(200.0, 3000.0))

    assert stage.moves == 3
    assert _stored(tmp_path).status is StageStatus.failed
    assert len(_warnings(caplog)) == 1
    assert "the stage is left at" in _warnings(caplog)[0]


def test_a_position_that_cannot_be_read_does_not_hide_the_stop(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning's own read is in its own ``try``: the stop is still the error that arrives."""
    clock = _with_clock(monkeypatch)
    stop = threading.Event()
    _costed(
        fake_microscope,
        clock,
        on_move=lambda n: stop.set() if n == 1 else None,
        unreadable_after=1,
    )

    record = _run(fake_microscope, tmp_path, stop=stop, repeats=1, distances_um=SHORT)

    assert record.status is StageStatus.stopped
    assert _warnings(caplog) == [
        "the stage is left at an unknown position, not returned to (0.0, 0.0) µm"
    ]


# ---------------------------------------------------------------- the channels


def test_the_channels_are_checked_before_anything_moves(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _costed(fake_microscope, _Clock())
    fake_core.log.clear()

    with pytest.raises(PluginRunError) as unknown:
        _run(fake_microscope, tmp_path, channels=("DAPI", "GFP"))
    assert "GFP is not a channel of this stand" in str(unknown.value)
    assert "Cy5, DAPI, FITC, Rhodamine" in str(unknown.value)
    assert "DAPI is not" not in str(unknown.value)

    _without(fake_microscope, monkeypatch, Channels)
    with pytest.raises(PluginRunError, match="no channel group"):
        _run(fake_microscope, tmp_path, channels=("DAPI",))

    assert stage.moves == 0
    assert fake_core.log == []


def test_a_stop_after_a_channel_was_selected_leaves_it_and_says_so(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-88: nothing selects the channel found back after a stop; a WARNING names both."""
    stop = threading.Event()
    real = fake_microscope.require(Channels)
    seen: list[str] = []

    def select(name: str) -> str | None:
        seen.append(name)
        stop.set()
        return real.select(name)

    patched = copy.copy(real)
    patched.select = select  # type: ignore[method-assign]
    fake_microscope.override(Channels, patched)

    record = _run(
        fake_microscope,
        tmp_path,
        stop=stop,
        repeats=1,
        distances_um=SHORT,
        channels=("FITC", "Cy5"),
    )

    assert record.status is StageStatus.stopped
    assert seen == ["FITC"]
    assert real.current() == "FITC"
    assert _warnings(caplog) == [
        "the stand is left on channel FITC, not selected back to DAPI"
    ]


def test_a_stand_that_matches_no_channel_is_not_restored_and_says_so(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real = fake_microscope.require(Channels)
    seen: list[str] = []

    def select(name: str) -> str | None:
        seen.append(name)
        return real.select(name)

    patched = copy.copy(real)
    patched.select = select  # type: ignore[method-assign]
    patched.current = lambda: None  # type: ignore[method-assign]
    fake_microscope.override(Channels, patched)

    record = _run(
        fake_microscope,
        tmp_path,
        repeats=1,
        distances_um=SHORT,
        channels=("FITC",),
    )

    assert _result(record).channel_switch_ms is not None
    assert seen == ["FITC"]
    assert _warnings(caplog) == [
        "the stand matched no channel when the run began, so the channel is not "
        "selected back: it is left on the last one timed"
    ]


# --------------------------------------------------------------- the autofocus


def test_each_autofocus_sample_starts_from_disengaged(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    """Engaging while locked returns at once: a timed no-op, and a median of nothing."""
    _engaged(fake_core)

    result = _result(_run(fake_microscope, tmp_path, repeats=3, distances_um=SHORT))

    assert (
        _autofocus_calls(fake_core)
        == [
            "enableContinuousFocus(False)",
            "enableContinuousFocus(True)",
        ]
        * 3
    )
    assert result.refocus_ms is not None
    # The fake's stage reports no soft limits, and nothing else was assumed.
    assert result.assumed == (NO_LIMITS,)
    # 12 stage + 3 autofocus + 3 bare snaps.
    assert result.samples == 18
    # Locked again: the stand is as it was found.
    assert fake_microscope.require(Autofocus).is_engaged()


class _LosesTheLock(FakeCore):
    """A PFS that locks at the start and not again once it has been switched off."""

    def enableContinuousFocus(self, enable: bool) -> None:  # noqa: N802 (MMCore's name)
        super().enableContinuousFocus(enable)
        if not enable:
            self.autofocus_lockable = False


def test_a_pfs_that_never_locks_is_assumed_not_measured(tmp_path: Path) -> None:
    """The port's regression test, FM-88 and FM-118: a 5 s timeout is not the refocus cost.

    The first engage that does not lock ends the section (another would wait
    the whole timeout again, and a search drives Z); the plugin does not engage
    again, and the result says the autofocus was left off.
    """
    core = _LosesTheLock.demo_like()
    profile = Profile.demo().model_copy(
        update={"autofocus": AutofocusSection.model_validate({"lock_timeout_s": 0.2})}
    )
    with _opened(core, profile) as microscope:
        _engaged(core)

        result = _result(_run(microscope, tmp_path, repeats=3, distances_um=SHORT))

    assert result.refocus_ms == ASSUMED_REFOCUS_MS
    assert (
        f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: the autofocus did not "
        "lock; no sample locked"
    ) in result.assumed
    assert LEFT_OFF in result.assumed
    # One disengage, one engage that found no lock, then the layer's own switch-off.
    assert _autofocus_calls(core) == [
        "enableContinuousFocus(False)",
        "enableContinuousFocus(True)",
        "enableContinuousFocus(False)",
    ]
    # 12 stage + 3 bare snaps; none of the autofocus samples is in the result.
    assert result.samples == 15


def test_a_normal_end_that_leaves_the_autofocus_off_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    core = _LosesTheLock.demo_like()
    profile = Profile.demo().model_copy(
        update={"autofocus": AutofocusSection.model_validate({"lock_timeout_s": 0.2})}
    )
    with _opened(core, profile) as microscope:
        _engaged(core)
        _run(microscope, tmp_path, repeats=1, distances_um=SHORT)

    assert _warnings(caplog) == [LEFT_OFF]


def test_an_autofocus_that_is_off_is_not_switched_on(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-129: engaging drives Z to the stored offset's plane on the Ti2-E, so it is the operator's call."""
    fake_core.log.clear()

    result = _result(_run(fake_microscope, tmp_path, repeats=2, distances_um=SHORT))

    assert _autofocus_calls(fake_core) == []
    assert result.refocus_ms == ASSUMED_REFOCUS_MS
    assert (
        f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: the autofocus was off, "
        "and the plugin does not switch it on: engaging drives Z"
    ) in result.assumed


def test_an_autofocus_on_but_not_locked_is_not_timed(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    fake_core.enableContinuousFocus(True)
    fake_core.autofocus_lockable = False
    fake_core.log.clear()

    result = _result(_run(fake_microscope, tmp_path, repeats=2, distances_um=SHORT))

    assert _autofocus_calls(fake_core) == []
    assert result.refocus_ms == ASSUMED_REFOCUS_MS
    assert (
        f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: the autofocus was on "
        "but not locked on the sample"
    ) in result.assumed
    assert fake_microscope.require(Autofocus).is_engaged()


def test_with_autofocus_false_touches_nothing(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    _engaged(fake_core)

    result = _result(
        _run(
            fake_microscope,
            tmp_path,
            repeats=2,
            distances_um=SHORT,
            with_autofocus=False,
        )
    )

    assert _autofocus_calls(fake_core) == []
    assert result.refocus_ms == ASSUMED_REFOCUS_MS
    assert (
        f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: with_autofocus is false"
        in result.assumed
    )
    assert fake_microscope.require(Autofocus).is_engaged()


def test_a_stand_without_autofocus_has_no_refocus(
    fake_core: FakeCore, tmp_path: Path
) -> None:
    fake_core.unloadDevice("Autofocus")
    with _opened(fake_core) as microscope:
        assert microscope.get(Autofocus) is None
        seen: list[Progress] = []

        result = _result(
            _run(
                microscope,
                tmp_path,
                on_progress=seen.append,
                repeats=1,
                distances_um=SHORT,
            )
        )

    assert result.refocus_ms is None
    assert not any("refocus" in line for line in result.assumed)
    assert [p.total for p in seen] == [3, 3, 3]


def test_an_autofocus_switched_off_from_outside_is_not_switched_on_again(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-129: the operator's switch-off, after the first sample, is theirs and not ours to undo."""
    real = fake_microscope.require(Autofocus)
    _engaged(fake_core)

    def engage(timeout_s: float | None = None) -> bool:
        locked = real.engage(timeout_s)
        fake_core.enableContinuousFocus(False)  # someone else, right after the lock
        return locked

    patched = copy.copy(real)
    patched.engage = engage  # type: ignore[method-assign]
    fake_microscope.override(Autofocus, patched)
    fake_core.log.clear()

    result = _result(_run(fake_microscope, tmp_path, repeats=3, distances_um=SHORT))

    assert result.refocus_ms == ASSUMED_REFOCUS_MS
    assert (
        f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: something else switched "
        "the autofocus off during the measurement; 1 of 3 samples locked, median "
    ) in " ".join(result.assumed)
    assert LEFT_OFF not in result.assumed
    # One disengage and one engage; then the outside switch-off. No second engage.
    assert _autofocus_calls(fake_core) == [
        "enableContinuousFocus(False)",
        "enableContinuousFocus(True)",
        "enableContinuousFocus(False)",
    ]


def test_a_refused_engage_ends_the_section_with_the_reason(
    fake_core: FakeCore, fake_microscope: Microscope, tmp_path: Path
) -> None:
    real = fake_microscope.require(Autofocus)
    _engaged(fake_core)
    engages = 0

    def engage(timeout_s: float | None = None) -> bool:
        nonlocal engages
        engages += 1
        raise SafetyRefusedError("the offset was not seen locked with this objective")

    patched = copy.copy(real)
    patched.engage = engage  # type: ignore[method-assign]
    fake_microscope.override(Autofocus, patched)

    result = _result(_run(fake_microscope, tmp_path, repeats=3, distances_um=SHORT))

    assert engages == 1
    assert result.refocus_ms == ASSUMED_REFOCUS_MS
    assert (
        "the autofocus refused to engage (the offset was not seen locked with "
        "this objective); no sample locked"
    ) in " ".join(result.assumed)
    # It was switched off for the measurement and never came back on.
    assert LEFT_OFF in result.assumed


def test_a_halt_during_the_engage_propagates(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-64: a halt is not "it did not lock"; it ends the run, and the autofocus is left off."""
    real = fake_microscope.require(Autofocus)
    _engaged(fake_core)
    engages = 0

    def engage(timeout_s: float | None = None) -> bool:
        nonlocal engages
        engages += 1
        raise MicroscopeHaltedError

    patched = copy.copy(real)
    patched.engage = engage  # type: ignore[method-assign]
    fake_microscope.override(Autofocus, patched)

    with pytest.raises(MicroscopeHaltedError):
        _run(fake_microscope, tmp_path, repeats=3, distances_um=SHORT)

    assert engages == 1
    assert _stored(tmp_path).status is StageStatus.failed
    assert _warnings(caplog) == [
        "the autofocus was switched off by this run and is left off, not engaged "
        "again: engaging drives Z"
    ]


def test_a_stop_before_a_sample_does_not_switch_the_autofocus_off(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stop that arrives between the stage and the first lock leaves the autofocus on."""
    _engaged(fake_core)
    stop = threading.Event()
    real = fake_microscope.require(XYStage)
    seen = 0

    def on_move(n: int) -> None:
        nonlocal seen
        seen = n
        if n == 4:  # the last move of a repeats=1 stage section
            stop.set()

    stage = _CostedStage(real, _Clock(), on_move=on_move)
    fake_microscope.override(XYStage, stage)

    record = _run(fake_microscope, tmp_path, stop=stop, repeats=1, distances_um=SHORT)

    assert record.status is StageStatus.stopped
    assert seen == 4
    assert _autofocus_calls(fake_core) == []
    assert _warnings(caplog) == []
    assert fake_microscope.require(Autofocus).is_engaged()


def _patch_autofocus(
    microscope: Microscope, **methods: Callable[..., Any]
) -> Autofocus:
    """Replace methods of the stand's autofocus on a copy of it; return the real one."""
    real = microscope.require(Autofocus)
    patched = copy.copy(real)
    for name, method in methods.items():
        setattr(patched, name, method)
    microscope.override(Autofocus, patched)
    return real


STILL_ON = (
    "the autofocus is still ON: this run switched it off or tried to engage it and "
    "the call failed. It may be searching and driving Z: switch it off before "
    "anything else moves Z"
)


def test_a_disengage_that_faults_says_the_autofocus_is_still_on(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-127: the flag says it was switched off for the run; the device says it was not."""
    _engaged(fake_core)

    def disengage() -> None:
        raise HardwareError("the switch-off faulted")

    _patch_autofocus(fake_microscope, disengage=disengage)

    with pytest.raises(HardwareError, match="the switch-off faulted"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=SHORT)

    assert fake_core.isContinuousFocusEnabled()  # the premise: it is still on
    assert _warnings(caplog) == [STILL_ON]


def test_an_engage_that_raises_after_it_switched_on_says_the_autofocus_is_still_on(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A search that was stopped may leave the autofocus on and Z moving (FM-118)."""
    real = fake_microscope.require(Autofocus)
    _engaged(fake_core)

    def engage(timeout_s: float | None = None) -> bool:
        real.engage(timeout_s)
        raise HardwareError("search stopped; Z may still be moving")

    _patch_autofocus(fake_microscope, engage=engage)

    with pytest.raises(HardwareError, match="Z may still be moving"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=SHORT)

    assert fake_core.isContinuousFocusEnabled()
    assert _warnings(caplog) == [STILL_ON]


def test_an_autofocus_whose_state_cannot_be_read_after_a_fault_is_said_unknown(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning's own read is in a ``try``: it says what it does not know, and the fault arrives."""
    real = fake_microscope.require(Autofocus)
    _engaged(fake_core)
    faulted = False

    def disengage() -> None:
        nonlocal faulted
        faulted = True
        raise HardwareError("the switch-off faulted")

    def is_engaged() -> bool:
        if faulted:
            raise HardwareError("the autofocus does not answer")
        return real.is_engaged()

    _patch_autofocus(fake_microscope, disengage=disengage, is_engaged=is_engaged)

    with pytest.raises(HardwareError, match="the switch-off faulted"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=SHORT)

    assert _warnings(caplog) == [
        "the autofocus was switched off by this run and its state cannot be read "
        "now: it may be on and driving Z, so check it before anything else moves Z"
    ]


def test_an_engage_that_fails_with_the_autofocus_off_says_it_is_left_off(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plain fault of the engage, with the real switch-off done: the old, true, message."""
    _engaged(fake_core)

    def engage(timeout_s: float | None = None) -> bool:
        raise HardwareError("the autofocus did not answer")

    _patch_autofocus(fake_microscope, engage=engage)

    with pytest.raises(HardwareError, match="did not answer"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=SHORT)

    assert not fake_core.isContinuousFocusEnabled()
    assert _warnings(caplog) == [
        "the autofocus was switched off by this run and is left off, not engaged "
        "again: engaging drives Z"
    ]


# ------------------------------------------------ faults and what a stop leaves


def test_a_fault_ends_the_measurement_and_says_where_the_stage_is(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FM-88 asks for a stop, a halt and a fault: a plain ``HardwareError`` leaves the stage too."""
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock, fault_after=3)

    with pytest.raises(HardwareError, match="answered and then faulted"):
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 3000.0))

    assert stage.moves == 3  # no return was sent
    assert _stored(tmp_path).status is StageStatus.failed
    assert _warnings(caplog) == [
        "the stage is left at (3000.0, 0.0) µm, not returned to (0.0, 0.0) µm"
    ]


def test_a_channel_that_cannot_be_read_after_a_stop_is_said_unknown(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stop = threading.Event()
    real = fake_microscope.require(Channels)
    selected: list[str] = []

    def select(name: str) -> str | None:
        selected.append(name)
        stop.set()
        return real.select(name)

    def current() -> str | None:
        if selected:
            raise HardwareError("the channel group does not answer")
        return real.current()

    patched = copy.copy(real)
    patched.select = select  # type: ignore[method-assign]
    patched.current = current  # type: ignore[method-assign]
    fake_microscope.override(Channels, patched)

    record = _run(
        fake_microscope,
        tmp_path,
        stop=stop,
        repeats=1,
        distances_um=SHORT,
        channels=("FITC",),
    )

    assert record.status is StageStatus.stopped
    assert selected == ["FITC"]
    assert _warnings(caplog) == [
        "the stand's channel is unknown, not selected back to DAPI"
    ]


def test_a_stop_while_the_stand_is_on_the_channel_found_says_nothing(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The channel found is visited last: a stop during its snaps leaves the stand where it was found."""
    stop = threading.Event()
    real = fake_microscope.require(Camera)
    snaps = 0

    def snap() -> Any:
        nonlocal snaps
        snaps += 1
        if snaps == 5:  # 2 bare + 2 FITC come first; this is the first DAPI snap
            stop.set()
        return real.snap()

    patched = copy.copy(real)
    patched.snap = snap  # type: ignore[method-assign]
    fake_microscope.override(Camera, patched)

    record = _run(
        fake_microscope,
        tmp_path,
        stop=stop,
        repeats=2,
        distances_um=SHORT,
        channels=("FITC", "DAPI"),
    )

    assert record.status is StageStatus.stopped
    assert snaps == 5
    assert fake_microscope.require(Channels).current() == "DAPI"
    assert _warnings(caplog) == []


# ------------------------------------------------------- the start of the probe


@pytest.mark.parametrize(
    ("limits", "stands"),
    [
        # X: the start is below the low limit; the out move is allowed, the return is not.
        ((Limits(100.0, 20000.0), Limits(-1000.0, 1000.0)), (0.0, 0.0)),
        # Y: the start is outside the Y limits, which nothing else looks at.
        ((Limits(-20000.0, 20000.0), Limits(100.0, 1000.0)), (0.0, 0.0)),
    ],
)
def test_a_start_outside_the_soft_limits_is_refused_before_anything_moves(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limits: tuple[Limits, Limits],
    stands: tuple[float, float],
) -> None:
    clock = _with_clock(monkeypatch)
    stage = _costed(fake_microscope, clock, limits=limits)

    with pytest.raises(PluginRunError) as caught:
        _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    x, y = limits
    assert (
        f"the stage stands at ({stands[0]:g}, {stands[1]:g}) µm, outside its soft "
        f"limits x [{x.low_um:g}, {x.high_um:g}], y [{y.low_um:g}, {y.high_um:g}] µm, "
        "so the return to it would be refused; nothing was moved. Move the stage "
        "inside its limits first"
    ) in str(caught.value)
    assert stage.moves == 0


def test_a_start_on_the_soft_limits_is_inside_them(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limits are a closed range, in X and in Y."""
    clock = _with_clock(monkeypatch)
    stage = _costed(
        fake_microscope,
        clock,
        limits=(Limits(0.0, 20000.0), Limits(0.0, 1000.0)),
    )

    _run(fake_microscope, tmp_path, repeats=1, distances_um=(200.0, 9000.0))

    assert stage.targets[0] == (200.0, 0.0)


def test_the_start_is_read_after_the_stage_has_settled(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-85: a stage still travelling reads a fraction of its move until ``wait()`` returns."""
    clock = _with_clock(monkeypatch)
    real = fake_microscope.require(XYStage)
    real.move_to_um(1000.0, -250.0)
    stage = _costed(fake_microscope, clock, travelling=XY(1004.0, -250.0))

    _run(fake_microscope, tmp_path, repeats=1, distances_um=(300.0, 7000.0))

    assert stage.waits >= 1
    assert stage.targets[:2] == [(1300.0, -250.0), (1000.0, -250.0)]


def test_the_autofocus_is_read_before_the_stage_moves(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m2 §12: the lock read before any move decides; the operator's switch-off during the probe is theirs."""
    clock = _with_clock(monkeypatch)
    _engaged(fake_core)
    _costed(
        fake_microscope,
        clock,
        on_move=lambda n: fake_core.enableContinuousFocus(False) if n == 1 else None,
    )

    result = _result(_run(fake_microscope, tmp_path, repeats=2, distances_um=SHORT))

    assert (
        f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: something else switched "
        "the autofocus off during the measurement; no sample locked"
    ) in result.assumed
    # Only the operator's call: the plugin sent nothing.
    assert _autofocus_calls(fake_core) == ["enableContinuousFocus(False)"]


def test_the_clock_is_perf_counter() -> None:
    """``time.monotonic`` ticks every 15.6 ms on Windows and ``time.time`` can step back."""
    assert timing_plugin._clock is time.perf_counter


# ------------------------------------------------ what each timed call costs


def _advancing(
    clock: _Clock, costs_ms: list[float], call: Callable[..., Any]
) -> Callable[..., Any]:
    """``call``, after the clock has moved by the next scripted cost."""

    def timed(*args: Any, **kwargs: Any) -> Any:
        clock.advance_ms(costs_ms.pop(0))
        return call(*args, **kwargs)

    return timed


def test_the_medians_of_the_lock_the_snaps_and_the_switches_are_the_costs_given(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each timed call advances the clock by a scripted cost, none in order: a median is not the first sample.

    The costs are out of order on purpose (300, 100, 200 has the median 200, the
    first 300 and the largest 300), and the lists are empty at the end, so a loop
    that takes fewer samples than ``repeats`` leaves a cost unread (FM-124).
    """
    clock = _with_clock(monkeypatch)
    _costed(fake_microscope, clock)
    _engaged(fake_core)
    engage_ms = [300.0, 100.0, 200.0]
    # Three bare snaps, then three in FITC, then three in Cy5.
    snap_ms = [10.0, 30.0, 20.0, 5.0, 25.0, 15.0, 40.0, 60.0, 50.0]
    # FITC, Cy5, then the untimed select back to the channel found (DAPI).
    select_ms = [7.0, 9.0, 0.0]

    real_autofocus = fake_microscope.require(Autofocus)
    autofocus = copy.copy(real_autofocus)
    autofocus.engage = _advancing(clock, engage_ms, real_autofocus.engage)  # type: ignore[method-assign]
    fake_microscope.override(Autofocus, autofocus)
    real_camera = fake_microscope.require(Camera)
    camera = copy.copy(real_camera)
    camera.snap = _advancing(clock, snap_ms, real_camera.snap)  # type: ignore[method-assign]
    fake_microscope.override(Camera, camera)
    real_channels = fake_microscope.require(Channels)
    channels = copy.copy(real_channels)
    channels.select = _advancing(clock, select_ms, real_channels.select)  # type: ignore[method-assign]
    fake_microscope.override(Channels, channels)

    result = _result(
        _run(
            fake_microscope,
            tmp_path,
            repeats=3,
            distances_um=SHORT,
            channels=("FITC", "Cy5"),
        )
    )

    assert result.refocus_ms == pytest.approx(200.0)
    assert result.snap_ms == pytest.approx(20.0)
    assert result.channel_ms == pytest.approx({"FITC": 15.0, "Cy5": 50.0})
    assert result.channel_switch_ms == pytest.approx(8.0)
    # 12 stage + 3 autofocus + 3 bare snaps + 2 channels x 3 snaps + 2 switches.
    assert result.samples == 26
    assert (engage_ms, snap_ms, select_ms) == ([], [], [])
