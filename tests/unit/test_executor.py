"""The executor: one lock around every hardware call, and dry-run that reads but never moves."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable

import pytest

from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeBusyError,
    MicroscopeHaltedError,
    MotionInProgressError,
    MotionStoppedError,
)
from smc.hardware.safety import Executor, Motion

LOGGER = logging.getLogger("smc.hardware.test")


def test_dry_run_logs_and_does_not_call_the_action(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = Executor(dry_run=True, lock=threading.RLock(), logger=LOGGER)
    called: list[bool] = []

    def action() -> str:
        called.append(True)
        return "real"

    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        result = executor.do(
            "xy_stage: move_to (1.0, 2.0) µm", action, dry_result="commanded"
        )
    assert result == "commanded"
    assert called == []
    assert "[dry-run] xy_stage: move_to (1.0, 2.0) µm" in caplog.messages


def test_live_run_logs_and_returns_the_action_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = Executor(dry_run=False, lock=threading.RLock(), logger=LOGGER)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        result = executor.do("z: move_to 5.0 µm", lambda: 4.9, dry_result=5.0)
    assert result == 4.9
    assert caplog.messages == ["z: move_to 5.0 µm"]


def test_reads_reach_the_hardware_in_dry_run() -> None:
    executor = Executor(dry_run=True, lock=threading.RLock(), logger=LOGGER)
    assert executor.read(lambda: 42) == 42


def _held_by_another_thread(lock: threading.RLock) -> bool:
    # RLock is re-entrant, so the owning thread would always succeed; ask from
    # another thread whether the lock is free.
    result: list[bool] = []

    def probe() -> None:
        got = lock.acquire(blocking=False)
        if got:
            lock.release()
        result.append(not got)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return result[0]


def test_executor_holds_the_lock_during_action() -> None:
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER)
    assert not _held_by_another_thread(lock)
    assert executor.do(
        "mutation", lambda: _held_by_another_thread(lock), dry_result=False
    )
    assert executor.read(lambda: _held_by_another_thread(lock))
    assert not _held_by_another_thread(lock)


def test_dry_run_is_fixed_after_construction() -> None:
    executor = Executor(dry_run=True, lock=threading.RLock(), logger=LOGGER)
    with pytest.raises(AttributeError):
        executor.dry_run = False  # type: ignore[misc]


# --- Motion, stop and the lock (design §13, #54) -----------------------------
#
# The motions here are stubs whose busy flag the test sets. Threads meet on
# ``threading.Event``; every join is bounded and followed by an ``is_alive``
# check, so a deadlock fails the test instead of hanging the suite (FM-44).

JOIN_S = 5.0


class _Device:
    """A stub stage: the test sets ``busy``; ``stop`` is counted."""

    def __init__(
        self, label: str = "XY", *, stoppable: bool = True, stop_clears: bool = True
    ) -> None:
        self.label = label
        self.busy = False
        self.busy_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.stop_clears = stop_clears
        self.stops = 0
        self.polls = 0
        self.polled = threading.Event()
        self.on_poll: Callable[[int], None] | None = None
        self.motion = Motion(
            device=label,
            name=f"xy_stage {label}",
            is_busy=self.is_busy,
            stop=self.stop if stoppable else None,
        )

    def is_busy(self) -> bool:
        self.polls += 1
        self.polled.set()
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if self.busy_error is not None:
            raise self.busy_error
        return self.busy

    def stop(self) -> None:
        self.stops += 1
        if self.stop_error is not None:
            raise self.stop_error
        if self.stop_clears:
            self.busy = False


def _ex(*, dry_run: bool = False, lock_timeout_s: float = 60.0) -> Executor:
    return Executor(
        dry_run=dry_run,
        lock=threading.RLock(),
        logger=LOGGER,
        lock_timeout_s=lock_timeout_s,
    )


def _start(executor: Executor, device: _Device) -> None:
    """Move ``device`` through the executor, leaving it busy and registered."""

    def command() -> None:
        device.busy = True

    executor.do(
        "xy_stage: move_to (1.0, 2.0) µm",
        command,
        dry_result=None,
        motion=device.motion,
    )


class _Thread:
    """Run ``target`` in a thread and keep what it returned or raised."""

    def __init__(self, target: Callable[[], object]) -> None:
        self.result: object = None
        self.error: BaseException | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self._target()
        except BaseException as exc:
            self.error = exc

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def finished_within(self, seconds: float) -> bool:
        self._thread.join(timeout=seconds)
        return not self._thread.is_alive()

    def join(self) -> None:
        self._thread.join(timeout=JOIN_S)
        assert not self._thread.is_alive(), "thread did not finish: deadlock?"


def _never(*_: object) -> str:
    raise AssertionError("the refused action ran")


def test_mutation_is_refused_while_a_motion_is_busy() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    with pytest.raises(MotionInProgressError) as info:
        executor.do("shutter: open", _never, dry_result="")
    assert info.value.moving == ("xy_stage XY",)
    assert "`xy_stage XY` is still moving" in str(info.value)
    assert "wait()" in str(info.value)
    assert "stop()" in str(info.value)
    assert info.value.how_to_force == ""
    # A second move of the same stage is refused too: refuse, never queue.
    with pytest.raises(MotionInProgressError):
        executor.do("xy_stage: move_to", _never, dry_result="", motion=xy.motion)


def test_at_rest_read_is_refused_while_moving() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    with pytest.raises(MotionInProgressError):
        executor.read(_never, at_rest=True, description="camera: snap")


def test_plain_read_runs_while_moving() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    assert executor.read(lambda: 42) == 42
    assert executor.moving() == ("xy_stage XY",)


def test_idle_motion_is_dropped_and_the_next_mutation_runs() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    xy.busy = False
    assert executor.do("shutter: open", lambda: "ran", dry_result="") == "ran"
    assert executor.moving() == ()


def test_wait_does_not_hold_the_lock_between_polls() -> None:
    # A short lock timeout turns a wait that holds the lock into a failure
    # (MicroscopeBusyError) rather than a hang.
    executor, xy = _ex(lock_timeout_s=2.0), _Device()
    _start(executor, xy)
    waiter = _Thread(lambda: executor.wait(xy.motion, 30.0))
    assert xy.polled.wait(JOIN_S)
    assert executor.read(lambda: "read", description="xy_stage: position") == "read"
    assert waiter.is_alive()  # the read completed while the wait was polling
    xy.busy = False
    waiter.join()
    assert waiter.error is None
    assert executor.moving() == ()


def test_timed_out_wait_sends_stop_then_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    _start(executor, xy)
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(DeviceTimeoutError) as info,
    ):
        executor.wait(xy.motion, 0.05)
    assert xy.stops == 1
    assert "xy_stage XY is still busy after 0.05 s" in str(info.value)
    assert "the stop was sent" in str(info.value)
    assert "device_timeout_ms" in str(info.value)
    assert "xy_stage XY: stop after 0.05 s timeout" in caplog.messages


def test_timed_out_wait_on_an_unstoppable_device_says_so() -> None:
    executor, dichroic = _ex(), _Device("Dichroic", stoppable=False)
    _start(executor, dichroic)
    with pytest.raises(DeviceTimeoutError, match="cannot be stopped from smc"):
        executor.wait(dichroic.motion, 0.05)
    assert dichroic.stops == 0


def test_interrupted_wait_sends_stop_and_reraises() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)

    def interrupt(poll: int) -> None:
        if poll == 3:
            raise KeyboardInterrupt  # Ctrl-C, raised from the stub: no real signal

    xy.polls = 0
    xy.on_poll = interrupt
    with pytest.raises(KeyboardInterrupt):
        executor.wait(xy.motion, 30.0)
    assert xy.stops == 1


def test_stop_during_wait_makes_the_waiter_raise_motion_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    waiter = _Thread(lambda: executor.wait(xy.motion, 30.0))
    assert xy.polled.wait(JOIN_S)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(xy.motion)
    waiter.join()
    assert isinstance(waiter.error, MotionStoppedError)
    assert waiter.error.device == "xy_stage XY"
    assert xy.stops == 1
    assert "xy_stage XY: stop" in caplog.messages


def test_stopped_motion_dropped_by_another_guard_still_fails_its_waiter() -> None:
    # The waiter must remember the motion it waits for: once the stopped
    # stage is idle, another thread's guard may drop it from the registry
    # before the waiter polls again.
    executor, xy = _ex(), _Device()
    _start(executor, xy)

    def stop_then_another_mutation(poll: int) -> None:
        if poll == 1:
            executor.stop(xy.motion)
            # Its guard sees the stage idle (poll 2) and drops it.
            assert executor.do("shutter: open", lambda: 1, dry_result=0) == 1
            assert executor.moving() == ()

    xy.polls = 0
    xy.on_poll = stop_then_another_mutation
    with pytest.raises(MotionStoppedError):
        executor.wait(xy.motion, 30.0)


def test_stop_does_not_wait_for_the_lock() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    entered, release = threading.Event(), threading.Event()

    def hung_snap() -> None:
        entered.set()
        release.wait(JOIN_S)

    holder = _Thread(lambda: executor.read(hung_snap, description="camera: snap"))
    assert entered.wait(JOIN_S)
    started = time.monotonic()
    stopper = _Thread(lambda: executor.stop(xy.motion))
    alive = not stopper.finished_within(1.0)
    stopped_in_s = time.monotonic() - started
    release.set()
    holder.join()
    stopper.join()
    assert not alive, "stop() waited for the microscope lock"
    assert stopped_in_s < 1.0
    assert stopper.error is None
    assert xy.stops == 1


def test_stop_racing_a_command_is_resent_after_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device(stop_clears=False)

    def command_overtaken_by_a_stop() -> None:
        # The stop reaches the device first, then the command starts it.
        executor.stop(xy.motion)
        xy.busy = True

    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(MotionStoppedError),
    ):
        executor.do(
            "xy_stage: move_to",
            command_overtaken_by_a_stop,
            dry_result=None,
            motion=xy.motion,
        )
    assert xy.stops == 2
    assert "xy_stage XY: stop resent after the command" in caplog.messages


def test_failed_command_sends_stop_and_keeps_the_motion_until_idle() -> None:
    executor, xy = _ex(), _Device(stop_clears=False)

    def command_that_fails_after_reaching_the_device() -> None:
        xy.busy = True
        raise RuntimeError("serial timeout")

    with pytest.raises(RuntimeError, match="serial timeout"):
        executor.do(
            "xy_stage: move_to",
            command_that_fails_after_reaching_the_device,
            dry_result=None,
            motion=xy.motion,
        )
    assert xy.stops == 1
    assert executor.moving() == ("xy_stage XY",)
    with pytest.raises(MotionInProgressError):
        executor.do("shutter: open", _never, dry_result="")
    xy.busy = False
    assert executor.do("shutter: open", lambda: "ran", dry_result="") == "ran"
    assert executor.moving() == ()


def test_failed_stop_on_a_failure_path_does_not_mask_the_failure() -> None:
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")

    def failing_command() -> None:
        raise RuntimeError("serial timeout")

    with pytest.raises(RuntimeError, match="serial timeout"):
        executor.do(
            "xy_stage: move_to", failing_command, dry_result=None, motion=xy.motion
        )
    assert xy.stops == 1


def test_stop_that_raises_propagates() -> None:
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")
    with pytest.raises(OSError, match="port closed"):
        executor.stop(xy.motion)


def test_unreadable_busy_state_refuses_until_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    xy.busy_error = OSError("port closed")
    with pytest.raises(MotionInProgressError) as info:
        executor.do("shutter: open", _never, dry_result="")
    assert "could not be read" in str(info.value)
    assert "port closed" in str(info.value)
    executor.stop(xy.motion)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        assert executor.do("shutter: open", lambda: "ran", dry_result="") == "ran"
    assert "unreadable after a stop" in caplog.text
    assert executor.moving() == ()


@pytest.mark.parametrize("dry_run", [False, True])
def test_halt_refuses_mutations_and_acquisitions_until_resume(dry_run: bool) -> None:
    executor = _ex(dry_run=dry_run)
    executor.halt()
    halted_before_resume = executor.halted
    with pytest.raises(MicroscopeHaltedError, match=r"call `resume\(\)`"):
        executor.do("shutter: open", _never, dry_result="")
    with pytest.raises(MicroscopeHaltedError):
        executor.read(_never, at_rest=True, description="camera: snap")
    executor.resume()
    assert (halted_before_resume, executor.halted) == (True, False)
    expected = "dry" if dry_run else "ran"
    assert executor.do("shutter: open", lambda: "ran", dry_result="dry") == expected
    assert executor.read(lambda: "frame", at_rest=True) == "frame"


def test_halt_still_allows_reads_and_stop() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    executor.halt()
    assert executor.read(lambda: 42) == 42
    xy.stop_clears = False
    executor.stop(xy.motion)
    assert xy.stops == 1
    # A halt alone does not fail a waiter; a stopped motion does.
    other = _Device("Z")
    executor.wait(other.motion, 1.0)


def test_halt_during_the_guard_refuses_before_the_command() -> None:
    # Microscope.stop() halts, then stops the stages. A halt landing after
    # do() checked it must still keep the command from reaching the device.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    _start(executor, z)
    z.busy = False
    z.on_poll = lambda _: executor.halt()
    with pytest.raises(MicroscopeHaltedError):
        executor.do("xy_stage: move_to", _never, dry_result="", motion=xy.motion)
    assert executor.moving() == ()


def test_lock_timeout_names_the_holder() -> None:
    executor = _ex(lock_timeout_s=0.2)
    entered, release = threading.Event(), threading.Event()

    def hung_snap() -> None:
        entered.set()
        release.wait(JOIN_S)

    holder = _Thread(lambda: executor.read(hung_snap, description="camera: snap"))
    assert entered.wait(JOIN_S)
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.do("shutter: open", _never, dry_result="")
    finally:
        release.set()
        holder.join()
    assert info.value.holder == "camera: snap"
    assert 0.2 <= info.value.held_s < JOIN_S
    assert "camera: snap has held the microscope for" in str(info.value)


def test_nested_call_keeps_the_outer_holder() -> None:
    executor = _ex(lock_timeout_s=0.2)
    entered, release = threading.Event(), threading.Event()

    def hung() -> None:
        entered.set()
        release.wait(JOIN_S)

    holder = _Thread(
        lambda: executor.read(
            lambda: executor.do("inner", hung, dry_result=None), description="outer"
        )
    )
    assert entered.wait(JOIN_S)
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.read(_never)
    finally:
        release.set()
        holder.join()
    assert info.value.holder == "outer"
    # Released at the outermost exit: the lock is free again.
    assert executor.read(lambda: 1, description="after") == 1


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_lock_timeout_must_be_finite_and_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="lock_timeout_s must be finite"):
        _ex(lock_timeout_s=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_wait_timeout_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        _ex().wait(_Device().motion, bad)


def test_dry_run_registers_no_motion_and_stop_still_runs() -> None:
    executor, xy = _ex(dry_run=True), _Device()
    xy.busy = True  # moved by hand while a dry-run session watches
    assert (
        executor.do("xy_stage: move_to", _never, dry_result="dry", motion=xy.motion)
        == "dry"
    )
    assert executor.moving() == ()
    assert executor.read(lambda: "frame", at_rest=True) == "frame"
    executor.wait(xy.motion, 30.0)  # nothing moved through the layer: returns
    executor.stop(xy.motion)
    assert xy.stops == 1


def test_stop_without_a_stop_callable_says_it_cannot() -> None:
    executor, dichroic = _ex(), _Device("Dichroic", stoppable=False)
    with pytest.raises(HardwareError, match="cannot be stopped from smc"):
        executor.stop(dichroic.motion)
