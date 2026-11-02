"""``Microscope.state()`` logs a failed read when it starts to fail, not on every call (#288).

The live worker reads ``state()`` for every frame and each open tab's tick reads
it once a second, so a line per failed read per call is ten lines a second to a
console that may block (FM-62, FM-148). ``errors`` still lists every failure on
every call: the status line shows the present, the log says when it began.

Every record of every logger is read, with the root at DEBUG as pymmcore-plus
leaves it once a core is open: each of them is a console line.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.microscope import MicroscopeState
from smc.hardware.profile import Profile
from smc.testing import FakeCore


@pytest.fixture
def stand() -> Iterator[tuple[FakeCore, Microscope]]:
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    try:
        yield core, microscope
    finally:
        microscope.close()


def _read_failed(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The message of every record, of any logger, that tells of a failed read.

    Every one is a DEBUG line: the record of when a read began to fail must not
    become a console line at the default level, once per failure or not.
    """
    records = [r for r in caplog.records if " read failed (" in r.getMessage()]
    assert {r.levelno for r in records} <= {logging.DEBUG}
    return [r.getMessage() for r in records]


def _line(state: MicroscopeState, name: str) -> str:
    """The line the read ``name`` writes: the detail is the one ``errors`` carries.

    The test fails if ``state`` has no entry for the read, or more than one.
    """
    (entry,) = [e for e in state.errors if e.startswith(f"{name}: ")]
    return f"state: {name} read failed ({entry.split(': ', 1)[1]})"


def test_a_read_that_keeps_failing_is_logged_once(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    core, microscope = stand
    # Set after the facade exists: before it, the constructor's own reads raise.
    # A fresh error per test: `failing` raises the same object on every call (FM-92).
    core.failing["Z"] = RuntimeError("Z: no answer")
    with caplog.at_level(logging.DEBUG):
        states = [microscope.state() for _ in range(5)]
    # `errors` is unchanged: every call lists the failed read, in the same words.
    assert len({_line(state, "z") for state in states}) == 1
    assert _read_failed(caplog) == [_line(states[0], "z")]


def test_a_read_that_succeeds_again_and_fails_again_is_logged_again(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    core, microscope = stand
    with caplog.at_level(logging.DEBUG):
        core.failing["Z"] = RuntimeError("Z: no answer")
        first = microscope.state()
        microscope.state()
        core.failing.clear()
        assert microscope.state().errors == []
        core.failing["Z"] = RuntimeError("Z: no answer, again")
        second = microscope.state()
        microscope.state()
    assert _line(first, "z") != _line(second, "z")
    assert _read_failed(caplog) == [_line(first, "z"), _line(second, "z")]


def test_a_failing_read_does_not_silence_another_that_starts_to_fail(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    core, microscope = stand
    with caplog.at_level(logging.DEBUG):
        core.failing["Z"] = RuntimeError("Z: no answer")
        z_only = microscope.state()
        microscope.state()
        core.failing["XY"] = RuntimeError("XY: no answer")
        both = microscope.state()
        microscope.state()
    assert _read_failed(caplog) == [_line(z_only, "z"), _line(both, "xy")]


def test_a_second_facade_logs_a_failed_read_of_its_own(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    # The memory is the facade's, not a module's: two stands that both lose
    # their Z drive are two failures.
    core, microscope = stand
    other_core = FakeCore.demo_like()
    with Microscope.from_core(cast("CMMCorePlus", other_core), Profile.demo()) as other:
        core.failing["Z"] = RuntimeError("Z: no answer")
        other_core.failing["Z"] = RuntimeError("Z: no answer")
        with caplog.at_level(logging.DEBUG):
            states = [(microscope.state(), other.state()) for _ in range(2)]
    line = _line(states[0][0], "z")
    assert _line(states[0][1], "z") == line
    assert _read_failed(caplog) == [line, line]


def test_a_channel_read_that_fails_after_a_failed_lookup_is_logged_too(
    stand: tuple[FakeCore, Microscope],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The two reads named `channel` never fail in one call (the second needs the
    # first), so only across calls can their keys collide: the lookup of the channel
    # group fails and is retried at the next call, which then finds the presets
    # unreadable. The second failure is a new one and has its line.
    core, microscope = stand
    core.config_groups = {"Channel": {"BF": [("Camera", "Exposure", "10")]}}
    real_groups = core.getAvailableConfigGroups
    calls = 0

    def groups() -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("config: no answer")
        return real_groups()

    def presets(group: str) -> tuple[str, ...]:
        raise RuntimeError("presets: no answer")

    monkeypatch.setattr(core, "getAvailableConfigGroups", groups)
    with caplog.at_level(logging.DEBUG):
        lookup = microscope.state()
        monkeypatch.setattr(core, "getAvailableConfigs", presets)
        current = microscope.state()
    assert _read_failed(caplog) == [_line(lookup, "channel"), _line(current, "channel")]
    assert "getAvailableConfigGroups" in _line(lookup, "channel")
    assert "getAvailableConfigs" in _line(current, "channel")


class _InterruptOnce(logging.Handler):
    """Raises ``KeyboardInterrupt`` from ``emit`` on the first line of a failed read.

    A Ctrl-C that lands while the console is written, FM-62.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.attempts = 0

    def emit(self, record: logging.LogRecord) -> None:
        if " read failed (" not in record.getMessage():
            return
        self.attempts += 1
        if self.attempts == 1:
            raise KeyboardInterrupt


def test_a_line_cut_by_an_interrupt_is_not_written_again_by_the_next_call(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    # The memory is rebound before the lines are written: a write that is cut short
    # (a blocked console, a second Ctrl-C) leaves it updated, so the next state(), the
    # stop path's describe() for one, does not write the same line to the same console.
    core, microscope = stand
    core.failing["Z"] = RuntimeError("Z: no answer")
    handler = _InterruptOnce()
    root = logging.getLogger()
    with caplog.at_level(logging.DEBUG):
        root.addHandler(handler)
        try:
            with pytest.raises(KeyboardInterrupt):
                microscope.state()
        finally:
            root.removeHandler(handler)
        assert handler.attempts == 1
        before = len(caplog.records)
        microscope.state()
    again = [r.getMessage() for r in caplog.records[before:]]
    assert [m for m in again if " read failed (" in m] == []
