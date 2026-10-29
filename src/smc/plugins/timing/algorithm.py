"""How long does a timepoint take, and how many positions fit in one? (m3 §12, #24).

A ten-hour movie at a five-minute interval is 120 timepoints. Whether it can
also visit 200 positions in three channels depends on numbers nobody can
guess: how fast this stage settles, how long this camera takes to read out, how
long the autofocus takes to lock, how long the filter turret takes to turn. The
plugin measures them on the stand; this module does the arithmetic on what it
measured, and is honest about what was assumed, because a throughput estimate
that silently uses a plausible-looking default is worse than no estimate.

The model per timepoint is::

    per position = move + refocus + sum over channels(switch + expose + read)
    timepoint    = positions x per position
    it fits      <=> timepoint < interval

Stage travel is ``overhead + distance x rate``, fitted from two measured
distances, because a 200 µm hop between fields inside a well and a 40 mm hop
between wells are not the same cost.

This is a port of ``nikon-control``'s ``timing.py``. Its ``Timings`` is the
``TimingsResult`` of the run document (m3 §3.5), so its methods became the
functions here; the other changes are the ones its tests asked for: a channel
nobody measured costs a bare snap and not zero, and nothing is clamped
silently (a headroom outside ``(0, 1]`` is an error).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from smc.plugins.slots import TimingsResult

__all__ = [
    "ASSUMED_CHANNEL_SWITCH_MS",
    "ASSUMED_REFOCUS_MS",
    "Budget",
    "channel_cost_ms",
    "describe_timings",
    "fit_line",
    "plan",
    "position_ms",
]

#: Used only when a measurement was not possible, and always reported as such.
ASSUMED_CHANNEL_SWITCH_MS = 100.0
ASSUMED_REFOCUS_MS = 400.0


@dataclass(frozen=True, slots=True)
class Budget:
    """What fits in one timepoint, and what it means over a whole movie.

    ``assumed`` is what the number rests on that nobody measured: the timings'
    own assumptions, and the costs ``plan`` had to fill in itself.
    """

    per_position_ms: float
    interval_s: float
    #: How many positions fit, with the headroom held back.
    positions: int
    #: How many were asked for.
    requested: int = 0
    duration_h: float = 0.0
    assumed: tuple[str, ...] = ()

    @property
    def timepoint_s(self) -> float:
        """Seconds one timepoint of the requested positions takes."""
        return self.per_position_ms * max(self.requested, 0) / 1000.0

    @property
    def fits(self) -> bool:
        """Whether the requested positions are no more than the ones that fit."""
        return self.requested <= self.positions

    @property
    def timepoints(self) -> int:
        """How many timepoints the duration holds at this interval."""
        return int(self.duration_h * 3600 / self.interval_s) if self.interval_s else 0

    @property
    def duty(self) -> float:
        """Fraction of the interval spent acquiring, for the requested count."""
        return self.timepoint_s / self.interval_s if self.interval_s else 0.0

    def describe(self) -> list[str]:
        """The budget in lines for an operator, ending with what it assumed."""
        out = [
            f"{self.per_position_ms / 1000:.2f} s per position",
            f"{self.positions} positions fit in a {self.interval_s / 60:g} min interval",
        ]
        if self.requested:
            verdict = "fits" if self.fits else "DOES NOT FIT"
            out.append(
                f"{self.requested} requested -> {self.timepoint_s:.0f} s per "
                f"timepoint ({self.duty * 100:.0f}% of the interval) — {verdict}"
            )
        if self.duration_h:
            out.append(f"{self.timepoints} timepoints over {self.duration_h:g} h")
        if self.assumed:
            out.append("assumed (not measured): " + "; ".join(self.assumed))
        return out


def fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares ``ms = overhead + rate * mm`` over ``(distance_um, ms)`` points.

    Returns ``(overhead_ms, ms_per_mm)``, each clamped at 0: a negative fit is
    noise, not physics. One point fits a rate through the origin; two at the
    same distance fit no rate and the mean time as the overhead.
    """
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        dist_mm = points[0][0] / 1000.0
        return 0.0, (points[0][1] / dist_mm if dist_mm else 0.0)
    xs = [d / 1000.0 for d, _ in points]
    ys = [ms for _, ms in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    rate = (
        sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom
        if denom
        else 0.0
    )
    overhead = my - rate * mx
    return max(0.0, overhead), max(0.0, rate)


def channel_cost_ms(timings: TimingsResult, channel: str) -> float:
    """The cost of exposing and reading one channel, in ms.

    The channel's own measured cost; else the mean of the measured ones; else
    a bare snap. The port returned 0 for the last case, which counted an
    exposure as free.
    """
    if channel in timings.channel_ms:
        return timings.channel_ms[channel]
    if timings.channel_ms:
        return statistics.fmean(timings.channel_ms.values())
    return timings.snap_ms


def position_ms(
    timings: TimingsResult,
    channels: Sequence[str],
    move_um: float = 0.0,
    *,
    refocus: bool = True,
) -> float:
    """What one position costs, in ms: a move, a refocus, then each channel.

    A refocus that the timings do not know (``refocus_ms`` is ``None``, a
    stand without autofocus) adds nothing, whatever ``refocus`` says. Each
    channel adds its switch (``ASSUMED_CHANNEL_SWITCH_MS`` when none was
    measured) and its own cost; with no channel the position is one bare snap
    and no switch, since nothing is selected.
    """
    total = timings.move_ms(move_um)
    if refocus and timings.refocus_ms is not None:
        total += timings.refocus_ms
    if not channels:
        return total + timings.snap_ms
    switch_ms = _switch_ms(timings)
    for name in channels:
        total += switch_ms + channel_cost_ms(timings, name)
    return total


def _switch_ms(timings: TimingsResult) -> float:
    if timings.channel_switch_ms is None:
        return ASSUMED_CHANNEL_SWITCH_MS
    return timings.channel_switch_ms


def describe_timings(timings: TimingsResult) -> list[str]:
    """The measured model in lines, for ``smc run -v``: the costs, then what was assumed."""
    refocus = (
        "not applicable: no autofocus"
        if timings.refocus_ms is None
        else f"{timings.refocus_ms:.0f} ms"
    )
    switch = (
        "not measured"
        if timings.channel_switch_ms is None
        else f"{timings.channel_switch_ms:.0f} ms"
    )
    out = [
        f"stage move   {timings.move_overhead_ms:.0f} ms + "
        f"{timings.move_ms_per_mm:.1f} ms/mm",
        f"refocus      {refocus}",
        f"channel swap {switch}",
    ]
    out += [f"  {name:<10} {ms:.0f} ms" for name, ms in timings.channel_ms.items()]
    out.append(f"snap         {timings.snap_ms:.0f} ms")
    if timings.assumed:
        out.append("assumed (not measured): " + "; ".join(timings.assumed))
    return out


def plan(
    timings: TimingsResult,
    channels: Sequence[str],
    interval_s: float,
    *,
    requested: int = 0,
    move_um: float = 2000.0,
    refocus: bool = True,
    headroom: float = 0.8,
    duration_h: float = 0.0,
) -> Budget:
    """How many positions fit in one interval.

    ``headroom`` is the share of the interval that may be used: a timelapse
    scheduled to 100 % of its interval has no slack for a slow refocus or a
    retry, and drifts later at every timepoint until it is minutes behind. A
    position that costs nothing fits none, as in the port: a cost of 0 means
    "nothing was measured", not "unlimited".

    Raises:
        ValueError: ``interval_s`` is not finite and positive, ``headroom`` is
            outside ``(0, 1]``, ``requested``, ``duration_h`` or ``move_um`` is
            negative or not finite (FM-32; the port clamped a headroom, which
            hid a typo), or the interval and duration are so large or so small
            that a count overflows.
    """
    if not (math.isfinite(interval_s) and interval_s > 0.0):
        raise ValueError(f"interval_s must be finite and positive, got {interval_s!r}")
    if not (math.isfinite(headroom) and 0.0 < headroom <= 1.0):
        raise ValueError(f"headroom must be in (0, 1], got {headroom!r}")
    for name, value in (
        ("requested", requested),
        ("duration_h", duration_h),
        ("move_um", move_um),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and 0 or more, got {value!r}")
    usable_ms = interval_s * 1000.0 * headroom
    timepoints = duration_h * 3600.0 / interval_s
    if not (math.isfinite(usable_ms) and math.isfinite(timepoints)):
        raise ValueError(
            f"interval_s {interval_s!r} and duration_h {duration_h!r} are too large "
            "or too small to count positions and timepoints"
        )
    names = list(channels)
    per = position_ms(timings, names, move_um, refocus=refocus)
    fit = int(usable_ms // per) if per > 0 else 0
    return Budget(
        per_position_ms=per,
        interval_s=interval_s,
        positions=fit,
        requested=requested,
        duration_h=duration_h,
        assumed=(*timings.assumed, *_assumed_by_plan(timings, names)),
    )


def _assumed_by_plan(timings: TimingsResult, channels: Sequence[str]) -> list[str]:
    """What ``position_ms`` filled in for the channels asked: a switch, a mean, a bare snap."""
    if not channels:
        return []
    lines: list[str] = []
    if timings.channel_switch_ms is None:
        lines.append(
            f"channel switch {ASSUMED_CHANNEL_SWITCH_MS:.0f} ms is assumed: "
            "the timings hold none"
        )
    for name in dict.fromkeys(channels):
        if name in timings.channel_ms:
            continue
        cost = channel_cost_ms(timings, name)
        if timings.channel_ms:
            why = "the mean of the measured channels: it was not measured"
        else:
            why = "a bare snap: no channel was measured"
        lines.append(f"channel {name} costs {cost:.0f} ms, {why}")
    return lines
