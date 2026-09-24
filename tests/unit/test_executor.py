"""The executor: one lock section per action, dry-run that reads but never moves (§13)."""

from __future__ import annotations

import _thread
import logging
import math
import pickle
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
    SafetyRefusedError,
)
from smc.hardware.safety import Executor, Motion, Step

LOGGER = logging.getLogger("smc.hardware.test")

# Threads meet on ``threading.Event``; every join is bounded and followed by
# an ``is_alive`` check, so a deadlock fails the test instead of hanging the
# suite (FM-44). Interrupts are raised from stubs, except in the one test
# that is about delivering a real pending Ctrl-C.
JOIN_S = 5.0


def _ex(*, dry_run: bool = False, lock_timeout_s: float = 60.0) -> Executor:
    return Executor(
        dry_run=dry_run,
        lock=threading.RLock(),
        logger=LOGGER,
        lock_timeout_s=lock_timeout_s,
    )


def _step(
    log: str, send: Callable[[], object], result: object = "readback"
) -> Step[object]:
    return Step(log=log, send=send, readback=lambda: result, dry_result="dry")


def _never(*_: object) -> str:
    raise AssertionError("the refused action ran")


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
    thread.join(JOIN_S)
    assert not thread.is_alive()
    return result[0]


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
        self.commands = 0
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

    def command(self) -> None:
        self.commands += 1
        self.busy = True

    def step(self) -> Step[str]:
        return Step(
            log=f"xy_stage: move {self.label}",
            send=self.command,
            readback=lambda: "arrived",
            dry_result="dry",
        )


def _move(
    executor: Executor, device: _Device, *, wait: bool = True, timeout_s: float = 30.0
) -> str:
    return executor.do(
        "xy_stage: move_to (1.0, 2.0) µm",
        device.step(),
        motion=device.motion,
        wait=wait,
        timeout_s=timeout_s,
    )


def _start(executor: Executor, device: _Device) -> None:
    """A ``wait=False`` move: ``device`` is left busy and registered."""
    _move(executor, device, wait=False)


def _hold_the_lock(executor: Executor) -> tuple[_Thread, threading.Event]:
    """A hung snap in another thread; set the returned event to end it."""
    entered, release = threading.Event(), threading.Event()

    def hung_snap() -> None:
        entered.set()
        release.wait(JOIN_S)

    holder = _Thread(lambda: executor.acquire("camera: snap", hung_snap))
    assert entered.wait(JOIN_S)
    return holder, release


# --- the basics --------------------------------------------------------------


def test_dry_run_logs_and_does_not_send(caplog: pytest.LogCaptureFixture) -> None:
    executor = _ex(dry_run=True)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        result = executor.do(
            "xy_stage: move_to", _step("xy_stage: move_to (1.0, 2.0) µm", _never)
        )
    assert result == "dry"
    assert "[dry-run] xy_stage: move_to (1.0, 2.0) µm" in caplog.messages


def test_live_run_logs_sends_and_returns_the_readback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sent: list[bool] = []
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        result = _ex().do(
            "z: move_to", _step("z: move_to 5.0 µm", lambda: sent.append(True), 4.9)
        )
    assert (result, sent) == (4.9, [True])
    assert caplog.messages == ["z: move_to 5.0 µm"]


def test_an_action_holds_the_lock_and_a_read_does_not_take_it() -> None:
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER)
    seen: list[bool] = []
    executor.do(
        "mutation",
        Step(
            log="mutation",
            send=lambda: seen.append(_held_by_another_thread(lock)),
            readback=lambda: seen.append(_held_by_another_thread(lock)),
            dry_result=None,
        ),
    )
    assert seen == [True, True]  # command and readback in one section
    assert executor.acquire("snap", lambda: _held_by_another_thread(lock))
    assert not executor.read(lambda: _held_by_another_thread(lock))
    assert not _held_by_another_thread(lock)


def test_reads_reach_the_hardware_in_dry_run() -> None:
    assert _ex(dry_run=True).read(lambda: 42) == 42


def test_dry_run_is_fixed_after_construction() -> None:
    executor = _ex(dry_run=True)
    with pytest.raises(AttributeError):
        executor.dry_run = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, 0.0, -1.0, threading.TIMEOUT_MAX * 2]
)
def test_lock_timeout_must_be_finite_positive_and_at_most_timeout_max(
    bad: float,
) -> None:
    with pytest.raises(ValueError, match="lock_timeout_s must be finite"):
        _ex(lock_timeout_s=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_wait_timeout_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        _ex().wait(_Device().motion, bad)
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        _ex().wait(_Device().motion, lambda: bad)


def test_errors_survive_pickling() -> None:
    # A plugin may run in a worker process and send its failure back.
    for error in (
        MotionInProgressError(("xy_stage XY",), "detail"),
        MicroscopeHaltedError(),
        MotionStoppedError("xy_stage XY", "it was stopped before it started"),
        MicroscopeBusyError("camera: snap", 61.0),
    ):
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is type(error)
        assert str(copy) == str(error)


# --- the guard and wait=False ------------------------------------------------


def test_every_action_is_refused_while_a_motion_is_busy() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    with pytest.raises(MotionInProgressError) as info:
        executor.do("shutter: open", _step("shutter: open", _never))
    assert info.value.moving == ("xy_stage XY",)
    assert "`xy_stage XY` is still moving" in str(info.value)
    assert "wait()" in str(info.value)
    assert "stop()" in str(info.value)
    assert info.value.how_to_force == ""
    with pytest.raises(MotionInProgressError):
        executor.acquire("camera: snap", _never)
    # A second move of the same stage is refused too: refuse, never queue.
    with pytest.raises(MotionInProgressError):
        _move(executor, xy)
    assert xy.commands == 1


def test_reads_and_safe_calls_run_while_moving() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    assert executor.read(lambda: 42) == 42
    assert executor.safe("shutter: close", lambda: "closed") == "closed"
    assert executor.moving() == ("xy_stage XY",)


def test_idle_motion_is_dropped_and_the_next_action_runs() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    xy.busy = False
    assert (
        executor.do("shutter: open", _step("shutter: open", lambda: None)) == "readback"
    )
    assert executor.moving() == ()


def test_owner_wait_that_times_out_stops_first_then_raises(
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
    # Still busy after the stop: it is moving, so it stays registered.
    assert executor.moving() == ("xy_stage XY",)


def test_a_bystander_wait_never_stops_another_callers_move() -> None:
    # #59 finding 5/6: another thread's short wait(0.2) stopped a traverse.
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    bystander = _Thread(lambda: executor.wait(xy.motion, 0.2))
    bystander.join()
    assert isinstance(bystander.error, DeviceTimeoutError)
    assert "belongs to another caller, so nothing was stopped" in str(bystander.error)
    assert xy.stops == 0
    assert executor.moving() == ("xy_stage XY",)


def test_a_bystander_waits_for_the_lock_within_its_own_timeout() -> None:
    # #59 finding 12: each poll waited lock_timeout_s whatever the deadline.
    executor, xy = _ex(lock_timeout_s=60.0), _Device()
    holder, release = _hold_the_lock(executor)
    try:
        started = time.monotonic()
        with pytest.raises(DeviceTimeoutError, match="held the microscope"):
            executor.wait(xy.motion, 0.3)
        waited_s = time.monotonic() - started
    finally:
        release.set()
        holder.join()
    assert waited_s < JOIN_S
    assert xy.stops == 0


def test_timed_out_wait_on_an_unstoppable_device_says_so() -> None:
    executor, dichroic = _ex(), _Device("Dichroic", stoppable=False)
    _start(executor, dichroic)
    with pytest.raises(DeviceTimeoutError, match="cannot be stopped from smc"):
        executor.wait(dichroic.motion, 0.05)
    assert dichroic.stops == 0


def test_interrupted_wait_stops_and_reraises() -> None:
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


def test_stopped_motion_dropped_by_the_guard_still_fails_its_owners_wait() -> None:
    # #59 finding 4: the guard dropped a stopped, idle registration, so the
    # owner's later wait() reported an arrival.
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    executor.stop(xy.motion)
    assert executor.do("shutter: open", _step("shutter: open", lambda: None))
    assert executor.moving() == ()
    with pytest.raises(MotionStoppedError):
        executor.wait(xy.motion, 1.0)
    # Its owner has learnt it; the next wait on the idle stage returns.
    executor.wait(xy.motion, 1.0)


def test_a_bystander_that_sees_a_stopped_move_idle_leaves_it_for_the_owner() -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    executor.stop(xy.motion)
    bystander = _Thread(lambda: executor.wait(xy.motion, 1.0))
    bystander.join()
    assert isinstance(bystander.error, MotionStoppedError)
    with pytest.raises(MotionStoppedError):
        executor.wait(xy.motion, 1.0)


def test_failed_command_stops_and_keeps_the_motion_until_idle() -> None:
    executor, xy = _ex(), _Device(stop_clears=False)

    def command_that_fails_after_reaching_the_device() -> None:
        xy.busy = True
        raise RuntimeError("serial timeout")

    with pytest.raises(RuntimeError, match="serial timeout"):
        executor.do(
            "xy_stage: move_to",
            Step("move", command_that_fails_after_reaching_the_device, lambda: 0, 0),
            motion=xy.motion,
            timeout_s=30.0,
        )
    assert xy.stops == 1
    assert executor.moving() == ("xy_stage XY",)
    with pytest.raises(MotionInProgressError):
        executor.do("shutter: open", _step("shutter: open", _never))
    xy.busy = False
    assert executor.do("shutter: open", _step("shutter: open", lambda: None))
    assert executor.moving() == ()


def test_failed_stop_on_a_failure_path_does_not_mask_the_failure() -> None:
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")

    def failing_command() -> None:
        raise RuntimeError("serial timeout")

    with pytest.raises(RuntimeError, match="serial timeout"):
        executor.do(
            "move",
            Step("move", failing_command, lambda: 0, 0),
            motion=xy.motion,
            timeout_s=30.0,
        )
    assert xy.stops == 1


def test_timeout_is_resolved_before_the_command() -> None:
    # #59 finding 2: the core timeout was read after the command, outside the
    # wait's try, so a failure there left the stage moving with no stop.
    executor, xy = _ex(), _Device()

    def core_timeout() -> float:
        raise RuntimeError("core gone")

    with pytest.raises(RuntimeError, match="core gone"):
        executor.do("move", xy.step(), motion=xy.motion, timeout_s=core_timeout)
    assert xy.commands == 0
    assert executor.moving() == ()


# --- the lock held through the action ---------------------------------------


def test_move_holds_the_lock_through_its_wait_and_refuses_others_at_once() -> None:
    # #59 findings 12/13: the readback ran in its own lock section, and other
    # callers could act between the wait and the readback.
    executor, xy = _ex(lock_timeout_s=60.0), _Device()
    mover = _Thread(lambda: _move(executor, xy))
    try:
        assert xy.polled.wait(JOIN_S)
        started = time.monotonic()
        with pytest.raises(MotionInProgressError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
        refused_in_s = time.monotonic() - started
        with pytest.raises(MotionInProgressError):
            executor.acquire("camera: snap", _never)
        assert executor.read(lambda: "read") == "read"
        assert executor.moving() == ("xy_stage XY",)
        assert mover.is_alive()
    finally:
        xy.busy = False
        mover.join()
    assert refused_in_s < JOIN_S  # refused at once, not after lock_timeout_s
    assert info.value.moving == ("xy_stage XY",)
    assert "is waiting for it" in str(info.value)
    assert mover.error is None
    assert mover.result == "arrived"
    assert executor.moving() == ()


def test_timed_out_move_stops_first_and_stays_registered_while_busy() -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    with pytest.raises(DeviceTimeoutError, match="the stop was sent"):
        _move(executor, xy, timeout_s=0.05)
    assert xy.stops == 1
    assert executor.moving() == ("xy_stage XY",)


def test_stop_during_a_move_makes_the_mover_raise_motion_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device()
    mover = _Thread(lambda: _move(executor, xy))
    assert xy.polled.wait(JOIN_S)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(xy.motion)
    mover.join()
    assert isinstance(mover.error, MotionStoppedError)
    assert mover.error.device == "xy_stage XY"
    assert xy.stops == 1
    assert "xy_stage XY: stop" in caplog.messages
    assert executor.moving() == ()


def test_stop_does_not_wait_for_the_lock() -> None:
    executor, xy = _ex(), _Device()
    holder, release = _hold_the_lock(executor)
    started = time.monotonic()
    stopper = _Thread(lambda: executor.stop(xy.motion))
    alive = not stopper.finished_within(JOIN_S)
    stopped_in_s = time.monotonic() - started
    release.set()
    holder.join()
    stopper.join()
    assert not alive, "stop() waited for the microscope lock"
    assert stopped_in_s < JOIN_S
    assert stopper.error is None
    assert xy.stops == 1


def test_a_stop_while_a_move_waits_for_the_lock_cancels_it() -> None:
    # #59 finding 3: a stop that landed before the move registered was lost,
    # and the queued move ran after the user's stop.
    executor, xy = _ex(), _Device()
    holder, release = _hold_the_lock(executor)
    try:
        mover = _Thread(lambda: _move(executor, xy))
        assert not mover.finished_within(0.2)  # queued behind the snap
        executor.stop(xy.motion)
    finally:
        release.set()
        holder.join()
    mover.join()
    assert isinstance(mover.error, MotionStoppedError)
    assert "stopped before it started" in str(mover.error)
    assert xy.commands == 0
    assert executor.moving() == ()


def test_a_stop_before_the_call_does_not_cancel_the_next_move() -> None:
    executor, xy = _ex(), _Device()
    executor.stop(xy.motion)
    assert _move(executor, xy, wait=False) == "arrived"
    assert xy.commands == 1


@pytest.mark.parametrize("wait", [True, False])
def test_stop_racing_the_command_is_resent_after_it(
    wait: bool, caplog: pytest.LogCaptureFixture
) -> None:
    executor, xy = _ex(), _Device()

    def command_overtaken_by_a_stop() -> None:
        # The stop reaches the device first, then the command starts it.
        executor.stop(xy.motion)
        xy.busy = True

    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(MotionStoppedError),
    ):
        executor.do(
            "move",
            Step("move", command_overtaken_by_a_stop, lambda: 0, 0),
            motion=xy.motion,
            wait=wait,
            timeout_s=30.0,
        )
    assert xy.stops == 2
    assert "xy_stage XY: stop resent after the command" in caplog.messages


def test_a_failing_resend_still_raises_motion_stopped() -> None:
    # #59 finding A: a failing FM-18 resend hid the MotionStoppedError.
    executor, xy = _ex(), _Device()

    def command_overtaken_by_a_stop() -> None:
        executor.stop(xy.motion)
        xy.stop_error = OSError("port closed")
        xy.busy = True

    with pytest.raises(MotionStoppedError):
        executor.do(
            "move",
            Step("move", command_overtaken_by_a_stop, lambda: 0, 0),
            motion=xy.motion,
            wait=False,
            timeout_s=30.0,
        )


def test_a_failed_stop_still_ends_the_move_as_stopped() -> None:
    # #59 finding A: the caller asked for a stop, so the move did not end as
    # planned, even though the stop itself failed.
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")
    mover = _Thread(lambda: _move(executor, xy))
    assert xy.polled.wait(JOIN_S)
    with pytest.raises(OSError, match="port closed"):
        executor.stop(xy.motion)
    xy.busy = False  # it arrived anyway
    mover.join()
    assert isinstance(mover.error, MotionStoppedError)


class _OrderHandler(logging.Handler):
    """Records, for each WARNING, how many stops the device had seen by then."""

    def __init__(self, device: _Device) -> None:
        super().__init__(logging.WARNING)
        self.device = device
        self.seen: list[tuple[str, int]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.seen.append((record.getMessage(), self.device.stops))


@pytest.mark.parametrize("path", ["stop", "timeout", "interrupt", "resend"])
def test_every_stop_path_sends_the_stop_before_the_log_line(path: str) -> None:
    # #59 finding 9: the WARNING came first, so a blocked console or a second
    # Ctrl-C delayed or skipped the stop.
    executor, xy = _ex(), _Device(stop_clears=False)
    handler = _OrderHandler(xy)
    LOGGER.addHandler(handler)
    try:
        if path == "stop":
            executor.stop(xy.motion)
        elif path == "timeout":
            with pytest.raises(DeviceTimeoutError):
                _move(executor, xy, timeout_s=0.05)
        elif path == "interrupt":
            xy.on_poll = lambda poll: (_ for _ in ()).throw(KeyboardInterrupt)
            with pytest.raises(KeyboardInterrupt):
                _move(executor, xy)
        else:

            def overtaken() -> None:
                executor.stop(xy.motion)
                xy.busy = True

            with pytest.raises(MotionStoppedError):
                executor.do(
                    "move",
                    Step("move", overtaken, lambda: 0, 0),
                    motion=xy.motion,
                    wait=False,
                )
    finally:
        LOGGER.removeHandler(handler)
    stop_lines = [(msg, stops) for msg, stops in handler.seen if "stop" in msg]
    assert stop_lines
    for message, stops_by_then in stop_lines:
        assert stops_by_then >= 1, f"logged before the stop: {message!r}"
    if path == "resend":
        assert ("xy_stage XY: stop resent after the command", 2) in stop_lines


# --- unreadable, unstoppable, halted -----------------------------------------


def test_unreadable_busy_state_refuses_until_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    xy.busy_error = OSError("port closed")
    with pytest.raises(MotionInProgressError) as info:
        executor.do("shutter: open", _step("shutter: open", _never))
    assert "could not be read" in str(info.value)
    assert "port closed" in str(info.value)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(xy.motion)
    assert "busy state unreadable" in caplog.text
    assert executor.moving() == ()
    assert executor.do("shutter: open", _step("shutter: open", lambda: None))


def test_a_stage_whose_stop_raises_and_whose_state_is_unreadable_is_released() -> None:
    # #59 finding 8: only a restart freed the stand.
    executor, xy = _ex(), _Device()
    _start(executor, xy)
    xy.busy_error = OSError("port closed")
    xy.stop_error = OSError("port closed")
    with pytest.raises(OSError, match="port closed"):
        executor.stop(xy.motion)
    assert executor.moving() == ()


def test_resume_releases_an_unreadable_motion_that_cannot_be_stopped() -> None:
    # #59 finding 8: a State device (no stop) with an unreadable busy state.
    executor, turret = _ex(), _Device("Turret", stoppable=False)
    _start(executor, turret)
    turret.busy_error = OSError("port closed")
    with pytest.raises(MotionInProgressError):
        executor.acquire("camera: snap", _never)
    executor.resume()
    assert executor.moving() == ()
    assert executor.acquire("camera: snap", lambda: "frame") == "frame"


def test_a_stage_still_busy_after_a_stop_stays_registered() -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    _start(executor, xy)
    executor.stop(xy.motion)
    executor.resume()
    assert executor.moving() == ("xy_stage XY",)


def test_stop_without_a_stop_callable_says_it_cannot() -> None:
    executor, dichroic = _ex(), _Device("Dichroic", stoppable=False)
    with pytest.raises(HardwareError, match="cannot be stopped from smc"):
        executor.stop(dichroic.motion)


def test_stop_that_raises_propagates() -> None:
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")
    with pytest.raises(OSError, match="port closed"):
        executor.stop(xy.motion)


@pytest.mark.parametrize("dry_run", [False, True])
def test_halt_refuses_every_action_until_resume(dry_run: bool) -> None:
    executor, xy = _ex(dry_run=dry_run), _Device()
    executor.halt()
    halted_before_resume = executor.halted
    with pytest.raises(MicroscopeHaltedError, match=r"call `resume\(\)`"):
        executor.do("shutter: open", _step("shutter: open", _never))
    with pytest.raises(MicroscopeHaltedError):
        executor.acquire("camera: snap", _never)
    with pytest.raises(MicroscopeHaltedError):
        _move(executor, xy)
    executor.resume()
    assert (halted_before_resume, executor.halted) == (True, False)
    expected = "dry" if dry_run else "readback"
    assert (
        executor.do("shutter: open", _step("shutter: open", lambda: None)) == expected
    )
    assert executor.acquire("camera: snap", lambda: "frame") == "frame"


def test_halted_is_not_a_safety_refusal() -> None:
    # #59 finding 14: `except SafetyRefusedError: skip the target` swallowed
    # the emergency stop.
    assert issubclass(MicroscopeHaltedError, HardwareError)
    assert not issubclass(MicroscopeHaltedError, SafetyRefusedError)


def test_halt_still_allows_reads_stops_safe_calls_and_waits() -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    _start(executor, xy)
    executor.halt()
    assert executor.read(lambda: 42) == 42
    assert executor.safe("shutter: close", lambda: False) is False
    executor.stop(xy.motion)
    assert xy.stops == 1
    xy.busy = False
    # A halt alone does not fail a waiter; a stopped motion does.
    with pytest.raises(MotionStoppedError):
        executor.wait(xy.motion, 1.0)
    executor.wait(_Device("Z").motion, 1.0)


@pytest.mark.parametrize("kind", ["mutation", "acquisition", "motion"])
def test_a_halt_during_the_guard_refuses_before_the_command(kind: str) -> None:
    # #59 finding 7: a halt that landed while the guard polled let a
    # set_open(True) or a snap run after it.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    _start(executor, z)
    z.busy = False
    z.on_poll = lambda _: executor.halt()
    actions: dict[str, Callable[[], object]] = {
        "mutation": lambda: executor.do(
            "shutter: open", _step("shutter: open", _never)
        ),
        "acquisition": lambda: executor.acquire("camera: snap", _never),
        "motion": lambda: _move(executor, xy),
    }
    with pytest.raises(MicroscopeHaltedError):
        actions[kind]()
    assert xy.commands == 0
    assert executor.moving() == ()


def test_safe_call_runs_while_a_snap_holds_the_lock() -> None:
    # #59 finding 11: closing the shutter was refused, so the light could not
    # be cut.
    executor = _ex(lock_timeout_s=60.0)
    holder, release = _hold_the_lock(executor)
    try:
        closer = _Thread(lambda: executor.safe("shutter: close", lambda: False))
        closed = closer.finished_within(JOIN_S)
    finally:
        release.set()
        holder.join()
    assert closed
    assert closer.error is None


# --- lock timeout and holder --------------------------------------------------


def test_lock_timeout_names_the_holder() -> None:
    executor = _ex(lock_timeout_s=0.5)
    holder, release = _hold_the_lock(executor)
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
    finally:
        release.set()
        holder.join()
    assert info.value.holder == "camera: snap"
    # Timer steps on Windows are about 15 ms: keep a margin (FM-44).
    assert 0.4 <= info.value.held_s < JOIN_S + 1.0
    assert "camera: snap has held the microscope for" in str(info.value)


def test_a_nested_action_keeps_the_outer_holder_and_its_lock() -> None:
    executor, xy = _ex(lock_timeout_s=0.5), _Device()
    entered, release = threading.Event(), threading.Event()

    def snap_with_a_callback_that_moves() -> None:
        # A callback inside the snap moves the stage: part of the outer action.
        _move(executor, xy, wait=False)
        xy.busy = False
        entered.set()
        release.wait(JOIN_S)

    holder = _Thread(
        lambda: executor.acquire("camera: snap", snap_with_a_callback_that_moves)
    )
    assert entered.wait(JOIN_S)
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
    finally:
        release.set()
        holder.join()
    assert holder.error is None
    assert info.value.holder == "camera: snap"
    assert xy.commands == 1
    assert executor.do("after", _step("after", lambda: None)) == "readback"


def _run_until_interrupted(executor: Executor) -> None:
    """Call actions until the pending Ctrl-C is delivered inside one of them."""
    deadline = time.monotonic() + JOIN_S
    try:
        while time.monotonic() < deadline:
            executor.do("shutter: open", _step("shutter: open", lambda: None))
    except KeyboardInterrupt:
        return
    raise AssertionError("the interrupt was never delivered")


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="interrupt_main() interrupts the main thread",
)
def test_a_ctrl_c_as_the_lock_is_taken_does_not_leave_it_held() -> None:
    # #59 finding 1: a KeyboardInterrupt delivered between acquire() and the
    # try of a @contextmanager left the RLock held for good. Here the holder
    # sets a Ctrl-C pending while the main thread waits for the lock, then
    # releases it: the interrupt lands as acquire() returns (or just after).
    for _ in range(5):
        lock = threading.RLock()
        executor = Executor(dry_run=False, lock=lock, logger=LOGGER, lock_timeout_s=5.0)
        ready = threading.Event()

        def hold(lock: threading.RLock = lock, ready: threading.Event = ready) -> None:
            with lock:
                ready.set()
                time.sleep(0.2)  # the main thread now waits in acquire()
                _thread.interrupt_main()

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        assert ready.wait(JOIN_S)
        _run_until_interrupted(executor)
        holder.join(JOIN_S)
        assert not holder.is_alive()
        assert not _held_by_another_thread(lock), "the Ctrl-C left the lock held"
        assert executor.do("after", _step("after", lambda: None)) == "readback"


# --- dry-run -----------------------------------------------------------------


def test_dry_run_registers_no_motion_and_stop_still_runs() -> None:
    executor, xy = _ex(dry_run=True), _Device()
    xy.busy = True  # moved by hand while a dry-run session watches
    assert _move(executor, xy) == "dry"
    assert xy.commands == 0
    assert executor.moving() == ()
    assert executor.acquire("camera: snap", lambda: "frame") == "frame"
    executor.wait(xy.motion, 30.0)  # nothing moved through the layer: returns
    executor.stop(xy.motion)
    assert xy.stops == 1
    assert executor.safe("shutter: close", lambda: False) is False
