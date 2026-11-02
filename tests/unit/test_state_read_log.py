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
    """The message of every record, of any logger, that tells of a failed read."""
    return [m for m in caplog.messages if " read failed (" in m]


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
