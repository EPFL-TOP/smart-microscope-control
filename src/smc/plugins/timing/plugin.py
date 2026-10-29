"""``timing`` measures what a stage move, an autofocus lock, a channel switch and a snap cost (m3 §12).

The pure part (the line fit, the model, ``plan``) is ``algorithm.py``. This is
the hardware loop around it, and the decisions in it are about what a
measurement may do to the stand:

* **A restore is a move, and it runs only after a normal end** (FM-88). After a
  stop, a halt or a fault the stage is left where it is, the channel is not
  selected back and the autofocus is not engaged again; a WARNING says what
  was left, and nothing is caught (FM-64), so the operator's stop ends the
  measurement and is not a failed sample.
* **The autofocus is switched off and on again, and a lock that does not come
  ends the section** (FM-118, FM-129). Engaging drives Z on the Ti2-E, so the
  plugin engages only what it switched off itself, never after a lock failed,
  and never what something else switched off. A refocus it could not measure
  is assumed, and says so (``assumed``), rather than recorded as the five
  seconds a timeout took.
* **A distance is the one the stage read back** (FM-142): a stage that stops
  short of a probe is not a measurement of that distance, and the run fails
  after it is back, naming how far it went.
* **A cost is timed on a clock the test can own** (FM-124): ``_clock`` is read
  at call time, so a test advances it by a known cost and the fit is exact.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from pydantic import Field, field_validator

from smc.hardware.capabilities import XY, Autofocus, Camera, Channels, XYStage
from smc.hardware.errors import SafetyRefusedError
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.slots import TimingsResult
from smc.plugins.timing.algorithm import (
    ASSUMED_REFOCUS_MS,
    describe_timings,
    fit_line,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["TimingParams", "TimingPlugin"]

_NAME = "timing"
#: The longest probe: an absolute move that no jog guard covers, so it is bounded here.
_MAX_DISTANCE_UM = 50_000.0
#: A move that went further or shorter than it was asked by more than this (or this
#: fraction of the distance) is not a measurement of that distance (FM-142).
_REACH_FLOOR_UM = 5.0
_REACH_FRACTION = 0.02
#: The stage reading further than this from where it started, after the return, is
#: reported: the probe's own tolerance is between two readbacks and does not see it.
_RETURN_TOLERANCE_UM = 5.0

T = TypeVar("T")

#: The clock the helper reads at call time. ``time.perf_counter``, not
#: ``time.monotonic``, which ticks every 15.6 ms on Windows.
_clock = time.perf_counter


class TimingParams(PluginParams):
    """How often, how far, and in which channels to time the stand."""

    repeats: int = Field(
        3,
        ge=1,
        le=20,
        description="Samples per distance, autofocus lock, snap and channel; the median is kept.",
    )
    distances_um: tuple[float, float] = Field(
        (200.0, 9000.0),
        description=(
            "The short and the long probe, in µm along X from where the stage "
            "stands, each in (0, 50000] and the second longer: the stage moves "
            "that far and back. A field-to-field hop and a well-to-well move."
        ),
    )
    with_autofocus: bool = Field(
        True,
        description=(
            "Time the autofocus lock: it is switched off and on again `repeats` "
            "times, so it must be on and locked on the sample."
        ),
    )
    channels: tuple[str, ...] = Field(
        (),
        max_length=32,
        description=(
            "Channels to time, by name; each is selected, timed and the channel "
            "found is selected back. None: a bare snap at the channel in place."
        ),
    )

    @field_validator("distances_um")
    @classmethod
    def _short_then_long(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Both in ``(0, 50000]`` and strictly increasing: two equal distances fit no line.

        An ``after`` validator: pydantic has already made every entry a float,
        so a nested list or a string is a ``ValidationError`` before this runs
        (FM-128), and the only thing raised here is a ``ValueError``.
        """
        for distance in value:
            if not 0.0 < distance <= _MAX_DISTANCE_UM:
                raise ValueError(
                    f"each distance must be in (0, {_MAX_DISTANCE_UM:g}] µm, got {distance:g}"
                )
        if not value[0] < value[1]:
            raise ValueError(
                f"the distances are (short, long) and must increase, got {value[0]:g}, "
                f"{value[1]:g}: two equal distances fit no line"
            )
        return value

    @field_validator("channels")
    @classmethod
    def _names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """No empty name and no repeat: a repeat would overwrite its own entry in ``channel_ms``."""
        if any(not name for name in value):
            raise ValueError("a channel name is empty")
        repeated = sorted({name for name in value if value.count(name) > 1})
        if repeated:
            raise ValueError(f"a channel is given twice: {', '.join(repeated)}")
        return value


@dataclass
class _Left:
    """What this run changed and has not put back: the warnings after a stop read it."""

    #: A move away from the start was sent and the return has not arrived.
    stage_away: bool = False
    #: The plugin switched the autofocus off and it has not locked again.
    autofocus_off: bool = False
    #: A channel was selected and the one found is not back.
    channel_changed: bool = False


@dataclass(frozen=True, slots=True)
class _Move:
    """One timed stage move: the distance it was asked, the distance it read back, its cost."""

    asked_um: float
    went_um: float
    ms: float


@dataclass(frozen=True, slots=True)
class _Refocus:
    """The autofocus section: the median lock time when every sample locked, else why not."""

    ms: float | None
    samples: int
    note: str | None


class _Steps:
    """The run's progress: ``total`` is known before the first move."""

    def __init__(self, ctx: RunContext, total: int) -> None:
        self._ctx = ctx
        self._total = total
        self._done = 0

    def __call__(self, message: str) -> None:
        self._done += 1
        self._ctx.progress(self._done, self._total, message)


class TimingPlugin:
    """Times the stand's elementary steps and fills the ``timings`` slot."""

    name = _NAME
    summary = (
        "Measure what a stage move, an autofocus lock, a channel switch and a "
        "snap cost on this stand."
    )
    Params = TimingParams
    Result = TimingsResult
    requires: tuple[type, ...] = (XYStage, Camera)
    provides = "timings"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: TimingParams,
        ctx: RunContext,
    ) -> TimingsResult:
        """Move the stage, time the autofocus and the camera; raise when the stage did not go.

        The stage moves by up to ``distances_um[1]`` along X from where it
        stands and comes back; the autofocus is switched off and on again
        ``repeats`` times; a channel is selected and the one found selected
        back. After a stop, a halt or a fault none of that is undone.

        Raises:
            PluginRunError: A dry run, a channel the stand does not have, a start
                that is not a position, soft limits that fit no probe, or a stage
                that did not go the distance asked (after it is back).
            StopRequested: The operator stopped it; the stand is as it was left.
        """
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        autofocus = microscope.get(Autofocus)
        channels = microscope.get(Channels)
        if ctx.dry_run:
            raise PluginRunError(
                _NAME,
                "a dry run cannot time the stand: the calls that would be timed "
                "are skipped, so their durations are not the stand's; run it "
                "without --dry-run",
            )
        order, found = _check_channels(channels, params.channels)

        # A stage still travelling from an earlier move reads a fraction of it (FM-85).
        xy.wait()
        start = xy.position_um()
        if not (math.isfinite(start.x_um) and math.isfinite(start.y_um)):
            raise PluginRunError(
                _NAME,
                f"the stage reports a start of ({start.x_um}, {start.y_um}) µm, which "
                "is not a position; nothing was moved",
            )
        sign, limits_note = _direction(xy, start, params.distances_um[1])
        assumed: list[str] = [] if limits_note is None else [limits_note]

        # Read before any move: the `is_locked()` read is what makes the offset a
        # seen one on a stand where engaging drives Z (m2 §12).
        skipped = (
            None
            if autofocus is None
            else _why_not_measured(autofocus, params.with_autofocus)
        )
        timed_autofocus = autofocus is not None and skipped is None
        step = _Steps(ctx, 2 + (1 if timed_autofocus else 0) + 1 + len(order))

        left = _Left()
        refocus_ms: float | None = None
        samples = 0
        try:
            points, stage_samples = _probe_stage(
                xy, start, sign, params, ctx, left, step
            )
            samples += stage_samples
            if autofocus is None:
                pass
            elif skipped is not None:
                refocus_ms = ASSUMED_REFOCUS_MS
                assumed.append(_assumed_refocus(skipped))
            else:
                measured = _time_autofocus(autofocus, params.repeats, ctx, left)
                step("autofocus lock")
                samples += measured.samples
                if measured.ms is None:
                    refocus_ms = ASSUMED_REFOCUS_MS
                    assumed.append(measured.note or _assumed_refocus("it did not lock"))
                else:
                    refocus_ms = measured.ms
                if left.autofocus_off:
                    assumed.append(_LEFT_OFF)

            snap_ms = _median_snap_ms(camera, params.repeats, ctx)
            samples += params.repeats
            step("snap")

            channel_ms: dict[str, float] = {}
            switches: list[float] = []
            if channels is not None and order:
                switches = _time_channels(
                    camera,
                    channels,
                    order,
                    found,
                    params.repeats,
                    ctx,
                    left,
                    step,
                    channel_ms,
                )
                samples += params.repeats * len(order) + len(switches)
                if not switches:
                    assumed.append(
                        "no switch was timed: the one channel given is the one "
                        "the stand was on"
                    )
                _select_found_back(ctx, channels, found, left)
        except BaseException:
            # A stop, a halt, a fault or a refused move: nothing is restored and
            # nothing is caught (FM-88, FM-64); a WARNING says what was left.
            _warn_left(ctx, left, xy, start, autofocus, channels, found)
            raise

        if left.autofocus_off:
            # A normal end that leaves the autofocus off: the result says so too.
            ctx.logger.warning(_LEFT_OFF)
        overhead_ms, ms_per_mm = fit_line(points)
        result = TimingsResult(
            move_overhead_ms=overhead_ms,
            move_ms_per_mm=ms_per_mm,
            refocus_ms=refocus_ms,
            snap_ms=snap_ms,
            channel_ms=channel_ms,
            channel_switch_ms=statistics.median(switches) if switches else None,
            samples=samples,
            assumed=tuple(assumed),
        )
        for line in describe_timings(result):
            ctx.logger.info("%s", line)
        return result


_LEFT_OFF = (
    "the autofocus was switched off for the measurement and did not lock again; "
    "engage it again before the scan"
)


def _time_ms(fn: Callable[[], T]) -> tuple[T, float]:
    """What ``fn`` returned and how long it took, in ms, on the module's ``_clock``."""
    began = _clock()
    value = fn()
    return value, (_clock() - began) * 1000.0


def _check_channels(
    channels: Channels | None, wanted: tuple[str, ...]
) -> tuple[tuple[str, ...], str | None]:
    """The names in the order they are timed, and the channel the stand is on; nothing is selected.

    The channel found goes **last**, so every timed switch changes channel: a
    switch to the channel already selected can cost nothing and would pull the
    median down. ``None`` for the channel found means no channel matches.
    """
    if not wanted:
        return (), None
    if channels is None:
        raise PluginRunError(
            _NAME,
            "channels were given but this stand has no channel group; nothing "
            "was moved or switched. Run it without channels to time a bare snap",
        )
    known = channels.names()
    missing = [name for name in wanted if name not in known]
    if missing:
        raise PluginRunError(
            _NAME,
            f"{', '.join(missing)} is not a channel of this stand (its channels: "
            f"{', '.join(known) or 'none'}); nothing was moved or switched",
        )
    found = channels.current()
    others = tuple(name for name in wanted if name != found)
    if found is not None and found in wanted:
        return (*others, found), found
    return others, found


def _direction(xy: XYStage, start: XY, long_um: float) -> tuple[float, str | None]:
    """``+1`` or ``-1`` along X so that the long probe stays inside the soft limits, and a note.

    ``+1`` unless the soft limits put the long probe outside the X travel on
    that side and inside it on the other. Soft limits of ``None`` mean unknown,
    not unlimited: the probe goes ``+X`` and the result says it was unchecked.
    A start outside the limits is refused before anything moves.
    """
    limits = xy.limits_um()
    if limits is None:
        return 1.0, (
            "the stage reports no soft limits: the probe moves were not checked "
            "against its travel"
        )
    x_limits, y_limits = limits
    low_um, high_um = x_limits.low_um, x_limits.high_um
    # The start is where the stage must come back to: one outside the limits has
    # a return that the layer refuses, after the stage has gone out.
    if not (
        low_um <= start.x_um <= high_um
        and y_limits.low_um <= start.y_um <= y_limits.high_um
    ):
        raise PluginRunError(
            _NAME,
            f"the stage stands at ({start.x_um:g}, {start.y_um:g}) µm, outside its "
            f"soft limits x [{low_um:g}, {high_um:g}], y [{y_limits.low_um:g}, "
            f"{y_limits.high_um:g}] µm, so the return to it would be refused; "
            "nothing was moved. Move the stage inside its limits first",
        )
    if low_um <= start.x_um + long_um <= high_um:
        return 1.0, None
    if low_um <= start.x_um - long_um <= high_um:
        return -1.0, None
    raise PluginRunError(
        _NAME,
        f"a probe of {long_um:g} µm fits neither way: the stage stands at X = "
        f"{start.x_um:g} µm and its soft limits are [{low_um:g}, {high_um:g}] µm; "
        "nothing was moved. Park the stage nearer the middle of its travel or "
        "give a shorter distances_um",
    )


def _why_not_measured(autofocus: Autofocus, with_autofocus: bool) -> str | None:
    """Why the lock will not be timed, or ``None`` when it will be; read before any move.

    The plugin times a lock only on an autofocus that is on and locked on the
    sample: switching one on that was off drives Z to the plane of its stored
    offset on the Ti2-E (FM-118), which is the operator's to ask for.
    """
    if not with_autofocus:
        return "with_autofocus is false"
    if not autofocus.is_engaged():
        return "the autofocus was off, and the plugin does not switch it on: engaging drives Z"
    if not autofocus.is_locked():
        return "the autofocus was on but not locked on the sample"
    return None


def _assumed_refocus(why: str) -> str:
    return f"refocus {ASSUMED_REFOCUS_MS:.0f} ms is assumed: {why}"


def _probe_stage(
    xy: XYStage,
    start: XY,
    sign: float,
    params: TimingParams,
    ctx: RunContext,
    left: _Left,
    step: _Steps,
) -> tuple[list[tuple[float, float]], int]:
    """The ``(distance_um, ms)`` points of the line fit, and how many moves were timed.

    For each distance, ``repeats`` times: out along X and back, each move
    timed. A point is the median of the distances the stage *read back* (FM-142)
    and of its ``2 * repeats`` times. The loop always completes before a stage
    that fell short is reported: the stage is back where it started first.
    """
    here = start
    points: list[tuple[float, float]] = []
    moves: list[_Move] = []
    for distance_um in params.distances_um:
        timed: list[_Move] = []
        for _ in range(params.repeats):
            for target_x_um in (start.x_um + sign * distance_um, start.x_um):
                ctx.raise_if_stopped()
                # Set before the move, since one that fails may have gone part of
                # the way; cleared once the return has arrived.
                left.stage_away = True
                arrived, ms = _time_ms(partial(xy.move_to_um, target_x_um, start.y_um))
                timed.append(
                    _Move(
                        asked_um=distance_um,
                        went_um=math.hypot(
                            arrived.x_um - here.x_um, arrived.y_um - here.y_um
                        ),
                        ms=ms,
                    )
                )
                here = arrived
            left.stage_away = False
        moves.extend(timed)
        points.append(
            (
                statistics.median(move.went_um for move in timed),
                statistics.median(move.ms for move in timed),
            )
        )
        step(f"stage {distance_um:g} µm")
    # The tolerance of a probe is between two readbacks: a return that stopped
    # short of the start, by up to 2 % of the distance, would pass it unseen.
    off_um = math.hypot(here.x_um - start.x_um, here.y_um - start.y_um)
    back_short = not off_um <= _RETURN_TOLERANCE_UM
    if back_short:
        ctx.logger.warning(
            "the stage reads (%.1f, %.1f) µm after its return, %.1f µm from where "
            "it started (%.1f, %.1f) µm: it came back short",
            here.x_um + 0.0,
            here.y_um + 0.0,
            off_um,
            start.x_um + 0.0,
            start.y_um + 0.0,
        )
    problem = _reach_problem(moves)
    if problem is not None:
        raise PluginRunError(
            _NAME,
            f"{problem}; the stage was sent back to where it started"
            + (f" and reads {off_um:.1f} µm from it" if back_short else "")
            + " and nothing was fitted. A stage that stops short (a soft limit, a "
            "stop, a stall) did not time the distance it was asked",
        )
    return points, len(moves)


def _reach_problem(moves: list[_Move]) -> str | None:
    """The first move that did not go the distance it was asked (FM-142), or ``None``."""
    for move in moves:
        if not math.isfinite(move.went_um):
            return (
                "the stage read back a position that is not a number after a move "
                f"of {move.asked_um:g} µm"
            )
        tolerance_um = max(_REACH_FLOOR_UM, _REACH_FRACTION * move.asked_um)
        if abs(move.went_um - move.asked_um) > tolerance_um:
            return (
                f"the stage went {move.went_um:.1f} µm when it was asked for "
                f"{move.asked_um:g} µm"
            )
    return None


def _time_autofocus(
    autofocus: Autofocus, repeats: int, ctx: RunContext, left: _Left
) -> _Refocus:
    """``repeats`` lock times, each from disengaged; the first lock that does not come ends it.

    Each sample starts from disengaged, since engaging while locked returns
    at once and would time a no-op. One that is not engaged when its turn
    comes was switched off by something else (the operator, a lost lock the
    protocol answered): it is not switched on again (FM-129). A ``False`` from
    ``engage`` or a refusal ends the section, since another sample would wait
    the whole lock timeout again, and a search drives Z (FM-118). Nothing else
    is caught: a halt or an excursion is not "it did not lock" (FM-64).
    """
    times: list[float] = []
    why: str | None = None
    for _ in range(repeats):
        ctx.raise_if_stopped()
        if not autofocus.is_engaged():
            why = "something else switched the autofocus off during the measurement"
            break
        left.autofocus_off = True
        autofocus.disengage()
        try:
            locked, ms = _time_ms(autofocus.engage)
        except SafetyRefusedError as exc:
            why = f"the autofocus refused to engage ({exc})"
            break
        if not locked:
            why = "the autofocus did not lock"
            break
        left.autofocus_off = False
        times.append(ms)
    if why is None:
        return _Refocus(ms=statistics.median(times), samples=len(times), note=None)
    if times:
        tail = (
            f"; {len(times)} of {repeats} samples locked, median "
            f"{statistics.median(times):.0f} ms"
        )
    else:
        tail = "; no sample locked"
    return _Refocus(ms=None, samples=0, note=_assumed_refocus(why + tail))


def _median_snap_ms(camera: Camera, repeats: int, ctx: RunContext) -> float:
    """The median of ``repeats`` bare snaps, at the channel and exposure the stand has."""
    times: list[float] = []
    for _ in range(repeats):
        ctx.raise_if_stopped()
        times.append(_time_ms(camera.snap)[1])
    return statistics.median(times)


def _time_channels(
    camera: Camera,
    channels: Channels,
    order: tuple[str, ...],
    found: str | None,
    repeats: int,
    ctx: RunContext,
    left: _Left,
    step: _Steps,
    channel_ms: dict[str, float],
) -> list[float]:
    """Fill ``channel_ms`` and return the switch times; one name that is the channel found selects nothing."""
    switches: list[float] = []
    select_each = not (len(order) == 1 and order[0] == found)
    for name in order:
        ctx.raise_if_stopped()
        if select_each:
            # Set before the call: a select that fails may have changed the stand.
            left.channel_changed = True
            switches.append(_time_ms(partial(channels.select, name))[1])
        times: list[float] = []
        for _ in range(repeats):
            ctx.raise_if_stopped()
            times.append(_time_ms(camera.snap)[1])
        channel_ms[name] = statistics.median(times)
        step(f"channel {name}")
    return switches


def _select_found_back(
    ctx: RunContext, channels: Channels, found: str | None, left: _Left
) -> None:
    """After a normal end, select the channel found again if the stand is not on it.

    A stand that matched no channel when the run began cannot be put back: the
    WARNING says so and the run goes on. A restore that raises propagates.
    """
    if not left.channel_changed:
        return
    if found is None:
        ctx.logger.warning(
            "the stand matched no channel when the run began, so the channel is not "
            "selected back: it is left on the last one timed"
        )
    elif channels.current() != found:
        channels.select(found)
    left.channel_changed = False


def _warn_left(
    ctx: RunContext,
    left: _Left,
    xy: XYStage,
    start: XY,
    autofocus: Autofocus | None,
    channels: Channels | None,
    found: str | None,
) -> None:
    """One WARNING per thing left, each read in a ``try`` of its own.

    A read that fails must not hide the error already on its way: the caller
    re-raises that one, whatever happens here. ``+ 0.0`` turns a ``-0.0`` (the
    demo stage reads one at its origin) into ``0.0`` in the message.
    """
    if left.stage_away:
        try:
            here = xy.position_um()
        except Exception:
            ctx.logger.warning(
                "the stage is left at an unknown position, not returned to "
                "(%.1f, %.1f) µm",
                start.x_um + 0.0,
                start.y_um + 0.0,
            )
        else:
            ctx.logger.warning(
                "the stage is left at (%.1f, %.1f) µm, not returned to (%.1f, %.1f) µm",
                here.x_um + 0.0,
                here.y_um + 0.0,
                start.x_um + 0.0,
                start.y_um + 0.0,
            )
    if left.autofocus_off and autofocus is not None:
        # The flag says the run switched it off and it did not lock again, not
        # that it is off: a `disengage()` that faulted, or an `engage()` that
        # raised after it switched on, leaves it on, and possibly driving Z.
        engaged: bool | None
        try:
            engaged = autofocus.is_engaged()
        except Exception:
            engaged = None
        if engaged is None:
            ctx.logger.warning(
                "the autofocus was switched off by this run and its state cannot be "
                "read now: it may be on and driving Z, so check it before anything "
                "else moves Z"
            )
        elif engaged:
            ctx.logger.warning(
                "the autofocus is still ON: this run switched it off or tried to "
                "engage it and the call failed. It may be searching and driving Z: "
                "switch it off before anything else moves Z"
            )
        else:
            ctx.logger.warning(
                "the autofocus was switched off by this run and is left off, not "
                "engaged again: engaging drives Z"
            )
    if left.channel_changed and channels is not None:
        try:
            current = channels.current()
        except Exception:
            ctx.logger.warning(
                "the stand's channel is unknown, not selected back to %s", found
            )
        else:
            if current != found:
                ctx.logger.warning(
                    "the stand is left on channel %s, not selected back to %s",
                    current,
                    found,
                )
