"""``_reread``: a readback that is read again until it is fresh (m2 §8.11 D1, #330).

The Ti2-E's ``DiaLamp`` read its previous value 9 ms after a set (FM-188). The
helper is pure: it takes the read and the test of freshness, so these tests
script the reads, and a clock of their own (``FakeClock``) stands in for
``mm.time``: how often and how long the helper sleeps is observed, and no
outcome depends on a stall of the real clock (FM-44). ``FRESH_READ_WINDOW_S`` is
read at the call, which is how a test shortens it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_mm_channels import fake_clock

import smc.hardware.backends.mm as mm
from smc.hardware.safety import POLL_INTERVAL_S


def _scripted(*values: int) -> tuple[Callable[[], int], list[int]]:
    """A read that returns ``values`` in turn, then the last one for ever."""
    reads: list[int] = []

    def read() -> int:
        reads.append(values[min(len(reads), len(values) - 1)])
        return reads[-1]

    return read, reads


def test_a_read_that_is_fresh_at_once_is_read_once_and_never_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = fake_clock(monkeypatch)
    read, reads = _scripted(210)
    result = mm._reread(read, lambda value: value == 210)
    assert (result.value, result.fresh, result.waited_s) == (210, True, 0.0)
    assert reads == [210]
    assert clock.sleeps == []


def test_a_stale_read_is_read_again_until_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = fake_clock(monkeypatch)
    read, reads = _scripted(501, 501, 210)
    result = mm._reread(read, lambda value: value == 210)
    assert (result.value, result.fresh) == (210, True)
    assert result.waited_s == pytest.approx(2 * POLL_INTERVAL_S)
    assert reads == [501, 501, 210]
    # One poll interval before each read after the first: a loop without the
    # sleep would hammer the device for the whole window.
    assert clock.sleeps == [POLL_INTERVAL_S, POLL_INTERVAL_S]


def test_a_read_that_never_turns_fresh_returns_the_last_read_when_the_window_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = fake_clock(monkeypatch)
    reads: list[int] = []

    def read() -> int:
        reads.append(len(reads))
        return reads[-1]

    result = mm._reread(read, lambda value: False)
    # The last read, not the first: a caller reports what the device held at the end.
    assert result.value == len(reads) - 1
    assert len(reads) > 1
    assert result.fresh is False
    assert result.waited_s >= mm.FRESH_READ_WINDOW_S
    # The window is paid in poll intervals, one sleep between two reads.
    assert clock.sleeps == [POLL_INTERVAL_S] * (len(reads) - 1)


def test_the_window_is_read_at_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # A test, or one day a profile, sets the window after the module was imported.
    clock = fake_clock(monkeypatch)
    monkeypatch.setattr(mm, "FRESH_READ_WINDOW_S", 0.0)
    read, reads = _scripted(501)
    result = mm._reread(read, lambda value: value == 210)
    assert (result.value, result.fresh) == (501, False)
    assert reads == [501]
    assert clock.sleeps == []


def test_a_read_that_raises_propagates() -> None:
    calls: list[int] = []

    def read() -> int:
        calls.append(0)
        if len(calls) == 2:
            raise RuntimeError("DiaLamp: no answer")
        return 501

    with pytest.raises(RuntimeError, match=r"^DiaLamp: no answer$"):
        mm._reread(read, lambda value: value == 210)
    assert len(calls) == 2


def test_a_read_again_is_a_wait_even_on_a_clock_too_coarse_to_see_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows below Python 3.13: time.monotonic() ticks every 15.625 ms, so two
    # reads 10 ms apart can measure 0.0. A caller tells a read that waited (an INFO
    # line) from one that did not by waited_s, so a re-read is never 0.0.
    fake_clock(monkeypatch, tick_s=0.015625)
    read, reads = _scripted(501, 210)
    result = mm._reread(read, lambda value: value == 210)
    assert (result.value, result.fresh, reads) == (210, True, [501, 210])
    assert result.waited_s >= POLL_INTERVAL_S


def test_the_formatting_tolerance_of_the_demo_is_fresh_and_a_real_difference_is_not() -> (
    None
):
    # The demo reads a number with four decimals and MMCore writes six.
    assert mm._reads_as_set(50.1234567, 50.1235)  # a gap of 4.3e-05
    assert not mm._reads_as_set(50.0, 50.001)  # a gap of 1e-3
    # At a small value the relative tolerance is nothing: only the absolute one
    # covers four decimals there, and a lamp's low settings are small values.
    assert mm._reads_as_set(0.1234567, 0.1235)
    assert not mm._reads_as_set(0.1234567, 0.1245)
    # At a large value the absolute tolerance is nothing: only the relative one
    # covers a device that formats a big number short.
    assert mm._reads_as_set(1_000_000.0, 1_000_000.5)
    assert not mm._reads_as_set(1_000_000.0, 1_000_002.0)
    # What the device did not take, whole numbers for 210.4, is no match.
    assert not mm._reads_as_set(210.4, 210.0)


@given(
    st.floats(min_value=-1e9, max_value=1e9, allow_nan=False),
    st.floats(min_value=-1e9, max_value=1e9, allow_nan=False),
)
def test_a_value_always_reads_as_itself_and_the_match_is_symmetric(
    value: float, other: float
) -> None:
    assert mm._reads_as_set(value, value)
    assert mm._reads_as_set(value, other) == mm._reads_as_set(other, value)
