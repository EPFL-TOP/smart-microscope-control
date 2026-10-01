"""The executor: one lock section per action; reads, waits and stops never take it (§13)."""

from __future__ import annotations

import _thread
import logging
import math
import pickle
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

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
from smc.hardware.safety import Executor, Motion, SafeCall, Step

LOGGER = logging.getLogger("smc.hardware.test")

# Threads meet on ``threading.Event``; every join is bounded and followed by
# an ``is_alive`` check, so a deadlock fails the test instead of hanging the
# suite (FM-44). Interrupts are raised from stubs, except in the tests that
# are about delivering a real pending Ctrl-C.
JOIN_S = 5.0

main_thread_only = pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="interrupt_main() interrupts the main thread",
)


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


_CLOSE = SafeCall("Shutter", "shutter: close", lambda: None)


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
    """A stub stage: the test sets ``busy``; ``stop`` and commands are counted.

    ``arrive_after`` makes it report idle by itself after that many busy
    polls following a command; ``None`` keeps it busy until the test (or a
    stop that clears it) says otherwise.
    """

    def __init__(
        self,
        label: str = "XY",
        *,
        stoppable: bool = True,
        stop_clears: bool = True,
        arrive_after: int | None = None,
    ) -> None:
        self.label = label
        self.busy = False
        self.busy_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.stop_clears = stop_clears
        self.arrive_after = arrive_after
        self.stops = 0
        self.commands = 0
        self.polls = 0
        self._busy_polls = 0
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
        if self.busy and self.arrive_after is not None:
            if self._busy_polls >= self.arrive_after:
                self.busy = False
            self._busy_polls += 1
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
        self._busy_polls = 0

    def step(self) -> Step[str]:
        return Step(
            log=f"xy_stage: move {self.label}",
            send=self.command,
            readback=lambda: "arrived",
            dry_result="dry",
        )


def _move(executor: Executor, device: _Device, *, timeout_s: float = 30.0) -> str:
    return executor.do(
        "xy_stage: move_to (1.0, 2.0) µm",
        device.step(),
        motion=device.motion,
        timeout_s=timeout_s,
    )


def _outlive(executor: Executor, device: _Device) -> None:
    """Leave ``device`` moving after its action gave up on it (§13, "outlives").

    The move times out at once and the stage ignores the stop, so it still
    reads busy and stays registered. It costs one command and one stop.
    """
    stop_clears, device.stop_clears = device.stop_clears, False
    try:
        with pytest.raises(DeviceTimeoutError):
            _move(executor, device, timeout_s=0.0)
    finally:
        device.stop_clears = stop_clears
    assert device.busy
    assert device.motion.name in executor.moving()


def _moving(executor: Executor, device: _Device) -> _Thread:
    """A move in another thread, waiting inside its action; set ``device.busy = False`` to end it."""
    device.polled.clear()
    mover = _Thread(lambda: _move(executor, device, timeout_s=JOIN_S))
    assert device.polled.wait(JOIN_S)
    return mover


def _blocking_move(
    executor: Executor, device: _Device
) -> tuple[_Thread, threading.Event]:
    """A move whose command blocks for the traverse, in another thread; set the event to end it.

    Some adapters return from ``setXYPosition`` only once the stage has
    arrived; the wait that follows then finds it idle at once.
    """
    inside, release = threading.Event(), threading.Event()

    def traverse() -> None:
        device.command()
        inside.set()
        release.wait(JOIN_S)
        device.busy = False

    mover = _Thread(
        lambda: executor.do(
            "xy_stage: move_to (1.0, 2.0) µm",
            Step("xy_stage: move XY", traverse, lambda: "arrived", "dry"),
            motion=device.motion,
            timeout_s=JOIN_S,
        )
    )
    assert inside.wait(JOIN_S)
    return mover, release


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


def test_an_action_holds_the_lock_and_reads_and_waits_do_not_take_it() -> None:
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
    xy = _Device()
    xy.on_poll = lambda _: seen.append(_held_by_another_thread(lock))
    executor.wait(xy.motion, 1.0)
    assert seen[-1] is False
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


def test_lock_timeout_accepts_timeout_max_itself() -> None:
    assert (
        _ex(lock_timeout_s=threading.TIMEOUT_MAX).do(
            "shutter: open", _step("shutter: open", lambda: None)
        )
        == "readback"
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_wait_timeout_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        _ex().wait(_Device().motion, bad)
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        _ex().wait(_Device().motion, lambda: bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_move_timeout_must_be_finite_and_non_negative(bad: float) -> None:
    xy = _Device()
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        _move(_ex(), xy, timeout_s=bad)
    assert xy.commands == 0


def test_an_action_with_a_motion_needs_a_timeout() -> None:
    xy = _Device()
    with pytest.raises(ValueError, match="needs timeout_s"):
        _ex().do("move", xy.step(), motion=xy.motion)
    assert xy.commands == 0


def test_errors_survive_pickling() -> None:
    # A plugin may run in a worker process and send its failure back.
    for error in (
        MotionInProgressError(("xy_stage XY",), "detail"),
        MicroscopeHaltedError(),
        MotionStoppedError("xy_stage XY", "it was stopped before it started"),
        MicroscopeBusyError("camera: snap", 61.0),
        MicroscopeBusyError("an unknown caller", 0.5, at_least=True),
    ):
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is type(error)
        assert str(copy) == str(error)
        assert copy.__dict__ == error.__dict__


# --- the table of §13: every kind of call under every condition ---------------


@contextmanager
def _condition(name: str) -> Iterator[Executor]:
    """An executor in one of the columns of §13's "Kinds of call" table."""
    executor = _ex(dry_run=name == "dry-run", lock_timeout_s=0.5)
    if name == "held":  # another thread's action (not a motion) holds the lock
        holder, release = _hold_the_lock(executor)
        try:
            yield executor
        finally:
            release.set()
            holder.join()
    elif name == "waiting":  # another thread's action waits for a motion
        xy = _Device()
        mover = _moving(executor, xy)
        try:
            yield executor
        finally:
            xy.busy = False
            mover.join()
    elif name == "outlived":  # a motion outlived its action
        _outlive(executor, _Device())
        yield executor
    elif name == "halted":
        executor.halt()
        yield executor
    else:
        yield executor


_REFUSAL: dict[str, type[BaseException]] = {
    "held": MicroscopeBusyError,  # after lock_timeout_s
    "waiting": MotionInProgressError,  # at once
    "outlived": MotionInProgressError,
    "halted": MicroscopeHaltedError,
}


@pytest.mark.parametrize(
    "condition", ["held", "waiting", "outlived", "halted", "dry-run"]
)
@pytest.mark.parametrize(
    "call", ["read", "wait", "mutation", "acquisition", "move", "stop", "close"]
)
def test_every_kind_of_call_under_every_condition(call: str, condition: str) -> None:
    # The Definition of done of #54: each row of the table (read, action,
    # safe) against "another action holds the lock", "a motion outlived its
    # action", "halted" and "dry-run". The calls act on Z, never on the XY
    # stage that sets up the condition.
    z = _Device("Z")
    sent: list[str] = []
    calls: dict[str, Callable[[Executor], object]] = {
        "read": lambda ex: ex.read(lambda: "read"),
        "wait": lambda ex: ex.wait(z.motion, 1.0),
        "mutation": lambda ex: ex.do(
            "shutter: open", _step("shutter: open", lambda: sent.append("open"))
        ),
        "acquisition": lambda ex: ex.acquire(
            "camera: snap", lambda: sent.append("snap") or "frame"
        ),
        "move": lambda ex: _move(ex, z),
        "stop": lambda ex: ex.stop(z.motion),
        "close": lambda ex: ex.safe(
            SafeCall("Shutter", "shutter: close", lambda: sent.append("close")),
            lambda: False,
        ),
    }
    with _condition(condition) as executor:
        started = time.monotonic()
        if call in ("mutation", "acquisition", "move") and condition in _REFUSAL:
            with pytest.raises(_REFUSAL[condition]):
                calls[call](executor)
            result: object = "refused"
        else:
            result = calls[call](executor)
        took_s = time.monotonic() - started
    # A read, a wait or a safe call that waited for the lock would take until
    # the holder gave up (JOIN_S), not a fraction of it.
    assert took_s < JOIN_S / 2
    expected: dict[str, tuple[object, list[str]]] = {
        "read": ("read", []),
        "wait": (None, []),
        "mutation": ("dry", []),
        "acquisition": ("frame", ["snap"]),
        "move": ("dry", []),
        "stop": (None, []),
        "close": (False, ["close"]),
    }
    if result == "refused":
        assert (sent, z.commands) == ([], 0)
    else:
        assert (result, sent) == expected[call]
        assert z.commands == 0
    assert z.stops == (1 if call == "stop" else 0)


# --- actions and the lock -----------------------------------------------------


def test_move_holds_the_lock_through_its_wait_and_refuses_others_at_once() -> None:
    # #59 findings 12/13: the readback ran in its own lock section, and other
    # callers could act between the wait and the readback.
    executor, xy = _ex(lock_timeout_s=60.0), _Device()
    mover = _moving(executor, xy)
    try:
        started = time.monotonic()
        with pytest.raises(MotionInProgressError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
        refused_in_s = time.monotonic() - started
        with pytest.raises(MotionInProgressError):
            executor.acquire("camera: snap", _never)
        assert executor.read(lambda: "read") == "read"
        # moving() names the motion an action is waiting for (#8's state()).
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


def test_a_nested_action_keeps_the_outer_holder_and_its_lock() -> None:
    executor, xy = _ex(lock_timeout_s=0.5), _Device(arrive_after=0)
    entered, release = threading.Event(), threading.Event()

    def snap_with_a_callback_that_moves() -> None:
        # A callback inside the snap moves the stage: part of the outer action.
        assert _move(executor, xy) == "arrived"
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


def test_an_action_inside_a_moves_command_is_refused() -> None:
    # #68, finding 1: pymmcore-plus runs a propertyChanged handler
    # synchronously inside setProperty, in the mover's own thread. Such a
    # callback re-enters the lock, and the guard looked only at motions that
    # outlived their action, so a snap ran while the turret moved.
    executor, turret, z = _ex(), _Device("Turret", arrive_after=0), _Device("Z")
    refused: list[MotionInProgressError] = []

    def command_with_a_synchronous_callback() -> None:
        turret.command()
        callbacks: list[Callable[[], object]] = [
            lambda: executor.acquire("camera: snap", _never),
            lambda: executor.do("shutter: open", _step("shutter: open", _never)),
            lambda: _move(executor, z),
        ]
        for callback in callbacks:
            try:
                callback()
            except MotionInProgressError as exc:
                refused.append(exc)

    result = executor.do(
        "turret: set",
        Step("turret: set", command_with_a_synchronous_callback, lambda: "2", "dry"),
        motion=turret.motion,
        timeout_s=30.0,
    )
    assert result == "2"  # the set itself carries on
    assert [exc.moving for exc in refused] == [("xy_stage Turret",)] * 3
    assert "`turret: set` is waiting for it" in str(refused[0])
    assert z.commands == 0


def test_an_action_halted_inside_a_moves_command_raises_halted() -> None:
    # Adversarial review of #80 (FM-64): a callback inside a move's command
    # that acted after a halt was refused with MotionInProgressError, a
    # SafetyRefusedError, before its halt was looked at.
    executor, turret, z = _ex(), _Device("Turret", arrive_after=0), _Device("Z")
    shutter = _Shutter()
    raised: list[BaseException] = []

    def command_with_a_synchronous_callback() -> None:
        turret.command()
        executor.halt()  # the operator's emergency stop, during the command
        for callback in _actions(executor, z, shutter).values():
            try:
                callback()
            except HardwareError as exc:
                raised.append(exc)

    result = executor.do(
        "turret: set",
        Step("turret: set", command_with_a_synchronous_callback, lambda: "2", "dry"),
        motion=turret.motion,
        timeout_s=30.0,
    )
    assert result == "2"  # the set was sent before the halt, and carries on
    assert [type(exc) for exc in raised] == [MicroscopeHaltedError] * 3
    assert (shutter.sent, z.commands) == ([], 0)


def test_moving_names_a_traverse_during_its_command() -> None:
    # #68, finding 4: "waiting for" was recorded only once the command had
    # returned, so a move whose command blocks for the traverse was absent
    # from moving() (#8's state()) for its whole length.
    executor, xy = _ex(), _Device()
    mover, release = _blocking_move(executor, xy)
    try:
        during = executor.moving()
    finally:
        release.set()
        mover.join()
    assert during == ("xy_stage XY",)
    assert mover.error is None
    assert mover.result == "arrived"
    assert executor.moving() == ()


def test_contender_is_refused_during_a_blocking_command() -> None:
    # #68, finding 4: during such a command another caller queued for the
    # lock, up to lock_timeout_s, instead of being refused at once.
    executor, xy, z = _ex(lock_timeout_s=60.0), _Device(), _Device("Z", arrive_after=0)
    mover, release = _blocking_move(executor, xy)
    try:
        contender = _Thread(lambda: _move(executor, z))
        refused = contender.finished_within(JOIN_S / 2)
    finally:
        release.set()
        mover.join()
    assert refused, "the contender queued behind the command"
    contender.join()
    assert isinstance(contender.error, MotionInProgressError)
    assert contender.error.moving == ("xy_stage XY",)
    assert z.commands == 0


def test_snap_right_after_the_move_is_idle_is_not_refused() -> None:
    # #68, finding 5: "waiting for" stayed recorded through the readback, so
    # the documented pattern "wait for the move, then snap" was refused
    # although the stage had arrived. A real readback takes time: two serial
    # reads of X and Y are about 120 ms; here it lasts until the test says.
    executor, xy = _ex(lock_timeout_s=60.0), _Device()
    reading_back, release = threading.Event(), threading.Event()

    def slow_readback() -> str:
        reading_back.set()
        release.wait(JOIN_S)
        return "arrived"

    mover = _Thread(
        lambda: executor.do(
            "xy_stage: move_to (1.0, 2.0) µm",
            Step("xy_stage: move XY", xy.command, slow_readback, "dry"),
            motion=xy.motion,
            timeout_s=JOIN_S,
        )
    )
    try:
        assert xy.polled.wait(JOIN_S)
        xy.busy = False  # the stage arrives, and the mover reads it back
        assert reading_back.wait(JOIN_S)
        executor.wait(xy.motion, JOIN_S)  # the UI waits for the move ...
        snapper = _Thread(lambda: executor.acquire("camera: snap", lambda: "frame"))
        done_early = snapper.finished_within(0.3)  # ... then snaps
    finally:
        release.set()
        mover.join()
    assert not done_early, f"the snap did not wait for the readback: {snapper.error!r}"
    snapper.join()
    assert snapper.error is None
    assert snapper.result == "frame"
    assert mover.result == "arrived"


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
    assert info.value.at_least is False
    assert "at least" not in str(info.value)


def test_unknown_holder_time_is_labelled_as_assumed() -> None:
    # #59 second review, finding 13: the "unknown caller" time was this
    # caller's wait, reported as if it were measured.
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER, lock_timeout_s=0.5)
    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        with lock:
            entered.set()
            release.wait(JOIN_S)

    holder = _Thread(hold)
    assert entered.wait(JOIN_S)
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
    finally:
        release.set()
        holder.join()
    assert info.value.holder == "an unknown caller"
    assert info.value.at_least is True
    assert 0.4 <= info.value.held_s < JOIN_S + 1.0  # this caller's own wait
    assert "has held the microscope for at least" in str(info.value)


def _outside_the_executor(
    lock: threading.RLock, action: Callable[[], object]
) -> _Thread:
    """``action`` in another thread, under ``lock`` taken outside the Executor.

    A facade helper that holds the lock for a sequence of actions does this.
    """

    def sequence() -> object:
        with lock:
            return action()

    return _Thread(sequence)


def test_reentrant_move_under_an_external_lock_is_named_by_moving() -> None:
    # #68, finding 7: a move made inside a lock taken outside the Executor
    # found no holder record to write "waiting for" into, so moving() was
    # empty during the traverse and other callers queued for lock_timeout_s.
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER, lock_timeout_s=60.0)
    xy = _Device()
    mover = _outside_the_executor(lock, lambda: _move(executor, xy, timeout_s=JOIN_S))
    try:
        assert xy.polled.wait(JOIN_S)
        during = executor.moving()
        contender = _Thread(lambda: executor.acquire("camera: snap", _never))
        refused = contender.finished_within(JOIN_S / 2)
    finally:
        xy.busy = False
        mover.join()
    assert during == ("xy_stage XY",)
    assert refused, "the contender queued behind the move"
    contender.join()
    assert isinstance(contender.error, MotionInProgressError)
    assert mover.result == "arrived"
    assert executor.moving() == ()


def test_an_action_under_an_external_lock_is_named_with_an_assumed_time() -> None:
    # The action's record starts when it does, but the lock was taken before
    # it, outside the Executor: the time is a lower bound, and says so.
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER, lock_timeout_s=0.5)
    entered, release = threading.Event(), threading.Event()
    holder = _outside_the_executor(
        lock,
        lambda: executor.acquire(
            "camera: snap", lambda: (entered.set(), release.wait(JOIN_S))
        ),
    )
    assert entered.wait(JOIN_S)
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
    finally:
        release.set()
        holder.join()
    assert info.value.holder == "camera: snap"
    assert info.value.at_least is True
    assert "camera: snap has held the microscope for at least" in str(info.value)


def test_a_move_under_an_external_lock_keeps_its_record_through_a_late_clean_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lock is released before the holder record is cleared, so another
    # thread can take it outside the Executor and move in between. On the
    # previous holder's record, that clean-up would erase the move's
    # "waiting for"; the move keeps a record of its own.
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER, lock_timeout_s=60.0)
    xy = _Device()
    clear = executor._clear_holder
    movers: list[_Thread] = []

    def an_external_move_starts_first(record: object) -> None:
        monkeypatch.undo()
        movers.append(
            _outside_the_executor(lock, lambda: _move(executor, xy, timeout_s=JOIN_S))
        )
        assert xy.polled.wait(JOIN_S)
        clear(record)  # type: ignore[arg-type]

    monkeypatch.setattr(executor, "_clear_holder", an_external_move_starts_first)
    executor.do("first", _step("first", lambda: None))
    try:
        during = executor.moving()
    finally:
        xy.busy = False
        movers[0].join()
    assert during == ("xy_stage XY",)
    assert movers[0].result == "arrived"


def _run_until_interrupted(executor: Executor) -> None:
    """Call actions until the pending Ctrl-C is delivered inside one of them."""
    deadline = time.monotonic() + JOIN_S
    try:
        while time.monotonic() < deadline:
            executor.do("shutter: open", _step("shutter: open", lambda: None))
    except KeyboardInterrupt:
        return
    raise AssertionError("the interrupt was never delivered")


@main_thread_only
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


def test_interrupt_in_release_does_not_keep_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #59 second review, finding 10: the holder record was cleared before the
    # release, so a second Ctrl-C in that clean-up kept the lock for good.
    lock = threading.RLock()
    executor = Executor(dry_run=False, lock=lock, logger=LOGGER)

    def interrupted(record: object) -> None:
        raise KeyboardInterrupt  # a second Ctrl-C, raised from a stub

    monkeypatch.setattr(executor, "_clear_holder", interrupted)
    with pytest.raises(KeyboardInterrupt):
        executor.do("shutter: open", _step("shutter: open", lambda: None))
    assert not _held_by_another_thread(lock), "the Ctrl-C kept the lock"
    monkeypatch.undo()
    # The stale holder record does not let the next action skip the lock.
    seen: list[bool] = []
    executor.do(
        "after",
        Step("after", lambda: seen.append(_held_by_another_thread(lock)), lambda: 0, 0),
    )
    assert seen == [True]
    assert not _held_by_another_thread(lock)


def test_releasing_the_lock_never_clears_the_next_holders_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lock is released before the holder record is cleared, so another
    # thread may take the lock in between; the clean-up must leave that
    # thread's record alone, or its callers are told "an unknown caller".
    executor = _ex(lock_timeout_s=0.5)
    clear = executor._clear_holder
    inside, release = threading.Event(), threading.Event()
    second: list[_Thread] = []

    def next_holder_takes_over_first(record: object) -> None:
        monkeypatch.undo()
        second.append(
            _Thread(
                lambda: executor.acquire(
                    "camera: snap 2", lambda: (inside.set(), release.wait(JOIN_S))
                )
            )
        )
        assert inside.wait(JOIN_S)
        clear(record)  # type: ignore[arg-type]

    monkeypatch.setattr(executor, "_clear_holder", next_holder_takes_over_first)
    executor.do("first", _step("first", lambda: None))
    try:
        with pytest.raises(MicroscopeBusyError) as info:
            executor.do("shutter: open", _step("shutter: open", _never))
    finally:
        release.set()
        second[0].join()
    assert info.value.holder == "camera: snap 2"


@pytest.mark.parametrize(
    "why", ["stopped while queued", "halted during the guard", "refused by the guard"]
)
def test_cancelled_move_is_not_logged_as_sent(
    why: str, caplog: pytest.LogCaptureFixture
) -> None:
    # #59 second review, finding 11: the INFO line was written before the
    # checks, so the log showed moves that were never sent.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        if why == "stopped while queued":
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
        elif why == "halted during the guard":
            _outlive(executor, z)
            z.busy = False
            z.on_poll = lambda _: executor.halt()
            with pytest.raises(MicroscopeHaltedError):
                _move(executor, xy)
        else:
            _outlive(executor, z)
            with pytest.raises(MotionInProgressError):
                _move(executor, xy)
    assert xy.commands == 0
    assert "xy_stage: move XY" not in caplog.messages


class _OnLog(logging.Handler):
    """Runs ``action`` when ``message`` is logged, as a slow console write would allow."""

    def __init__(self, message: str, action: Callable[[], object]) -> None:
        super().__init__()
        self.message = message
        self.action = action

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == self.message:
            self.action()


@pytest.mark.parametrize("what", ["halt", "stop"])
def test_a_halt_or_a_stop_while_the_line_is_logged_keeps_the_command_away(
    what: str, caplog: pytest.LogCaptureFixture
) -> None:
    # FM-66: the check that matters is the last one before the command. A log
    # line can block (a QuickEdit selection on Windows), and a halt or a stop
    # that lands meanwhile must still keep the command from the stage.
    executor, xy = _ex(), _Device()
    action: Callable[[], object] = (
        executor.halt if what == "halt" else lambda: executor.stop(xy.motion)
    )
    handler = _OnLog("xy_stage: move XY", action)
    LOGGER.addHandler(handler)
    expected = MicroscopeHaltedError if what == "halt" else MotionStoppedError
    try:
        with (
            caplog.at_level(logging.INFO, logger=LOGGER.name),
            pytest.raises(expected),
        ):
            _move(executor, xy)
    finally:
        LOGGER.removeHandler(handler)
    assert xy.commands == 0
    assert "xy_stage: move XY: not sent" in caplog.messages


def _landing_as_waiting_is_recorded(
    monkeypatch: pytest.MonkeyPatch, executor: Executor, event: Callable[[], object]
) -> None:
    """Run ``event`` as the move records "waiting for", just before its command."""
    set_waiting = executor._set_waiting

    def landing(name: str | None) -> str | None:
        if name is not None:  # the recording, not the clear
            event()
        return set_waiting(name)

    monkeypatch.setattr(executor, "_set_waiting", landing)


@pytest.mark.parametrize("what", ["halt", "stop"])
def test_a_halt_or_a_stop_while_waiting_for_is_recorded_keeps_the_command_away(
    what: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Adversarial review of #68 (FM-66): recording "waiting for" before the
    # command put a step after the last check, so a halt landing there was
    # missed and the command reached the stage.
    executor, xy = _ex(), _Device(arrive_after=0)
    event: Callable[[], object] = (
        executor.halt if what == "halt" else lambda: executor.stop(xy.motion)
    )
    _landing_as_waiting_is_recorded(monkeypatch, executor, event)
    expected = MicroscopeHaltedError if what == "halt" else MotionStoppedError
    with (
        caplog.at_level(logging.INFO, logger=LOGGER.name),
        pytest.raises(expected),
    ):
        _move(executor, xy)
    assert xy.commands == 0
    assert xy.stops == (1 if what == "stop" else 0)  # only the caller's own stop
    assert "xy_stage: move XY: not sent" in caplog.messages
    assert executor.moving() == ()


@pytest.mark.parametrize("what", ["halt", "stop"])
def test_a_move_refused_at_its_last_check_is_not_named_while_not_sent_is_logged(
    what: str, caplog: pytest.LogCaptureFixture
) -> None:
    # #80: the "not sent" WARNING was written while "waiting for" was still
    # recorded, so for as long as a blocked console held that line (FM-62),
    # moving() named a motion that was never commanded and contenders were
    # refused for it.
    executor, xy = _ex(), _Device()
    event: Callable[[], object] = (
        executor.halt if what == "halt" else lambda: executor.stop(xy.motion)
    )
    seen: list[tuple[str, ...]] = []
    handlers = [
        _OnLog("xy_stage: move XY", event),
        _OnLog("xy_stage: move XY: not sent", lambda: seen.append(executor.moving())),
    ]
    for handler in handlers:
        LOGGER.addHandler(handler)
    expected = MicroscopeHaltedError if what == "halt" else MotionStoppedError
    try:
        with (
            caplog.at_level(logging.INFO, logger=LOGGER.name),
            pytest.raises(expected),
        ):
            _move(executor, xy)
    finally:
        for handler in handlers:
            LOGGER.removeHandler(handler)
    assert seen == [()]
    assert xy.commands == 0


@pytest.mark.parametrize("stoppable", [True, False])
def test_an_interrupt_before_the_command_stops_nothing(
    stoppable: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Adversarial review of #68: a Ctrl-C that landed before the command gave
    # up on the motion, so it stopped a stage that had been told nothing, or
    # logged that a turret "cannot be stopped from smc".
    executor, turret = _ex(), _Device("Turret", stoppable=stoppable)

    def ctrl_c() -> None:
        raise KeyboardInterrupt

    _landing_as_waiting_is_recorded(monkeypatch, executor, ctrl_c)
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(KeyboardInterrupt),
    ):
        _move(executor, turret)
    assert (turret.commands, turret.stops) == (0, 0)
    assert "gave up" not in caplog.text
    assert "stop after" not in caplog.text
    assert executor.moving() == ()


# --- waiting, and giving up ---------------------------------------------------


def test_wait_is_a_lock_free_poll_that_never_stops() -> None:
    # §13 as revised by #67: wait() is a read. It runs while another thread's
    # action holds the lock, returns once the device is idle, and at its
    # deadline raises without stopping anything. It replaces the owner and
    # bystander rules of the previous round (#59 second review, findings 1,
    # 2, 3, 5, 6 and 8).
    executor, xy = _ex(), _Device()
    xy.busy = True  # moved by another caller, or by hand
    holder, release = _hold_the_lock(executor)
    try:
        with pytest.raises(DeviceTimeoutError) as info:
            executor.wait(xy.motion, 0.05)
        xy.busy = False
        waiter = _Thread(lambda: executor.wait(xy.motion, JOIN_S))
        returned = waiter.finished_within(JOIN_S / 2)
    finally:
        release.set()
        holder.join()
    assert returned, "wait() waited for the microscope lock"
    assert waiter.error is None
    assert "xy_stage XY is still busy after 0.05 s" in str(info.value)
    assert "`wait()` stops nothing" in str(info.value)
    assert xy.stops == 0


def test_wait_on_another_threads_move_neither_stops_nor_blocks_it() -> None:
    executor, xy = _ex(), _Device()
    mover = _moving(executor, xy)
    try:
        with pytest.raises(DeviceTimeoutError):
            executor.wait(xy.motion, 0.05)
        assert mover.is_alive()
        xy.busy = False
        executor.wait(xy.motion, JOIN_S)
    finally:
        xy.busy = False
        mover.join()
    assert xy.stops == 0
    assert mover.error is None
    assert mover.result == "arrived"


def test_wait_drops_the_registration_it_sees_idle() -> None:
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.busy = False
    executor.wait(xy.motion, 1.0)
    assert executor.moving() == ()


def test_wait_keeps_a_registration_made_after_it_looked() -> None:
    # A drop compares the registration read before the poll: one made since,
    # by a newer give-up, is not the one this wait saw idle.
    executor, xy = _ex(), _Device()

    def a_give_up_lands_during_the_poll(poll: int) -> None:
        xy.on_poll = None
        _outlive(executor, xy)
        xy.busy = False  # this poll then answers idle

    xy.on_poll = a_give_up_lands_during_the_poll
    executor.wait(xy.motion, 1.0)
    assert executor.moving() == ("xy_stage XY",)


def test_wait_raises_what_the_busy_state_raised_and_keeps_the_motion() -> None:
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.busy_error = OSError("port closed")
    with pytest.raises(OSError, match="port closed"):
        executor.wait(xy.motion, 1.0)
    assert executor.moving() == ("xy_stage XY",)


def test_timed_out_move_stops_first_and_stays_registered_while_busy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(DeviceTimeoutError) as info,
    ):
        _move(executor, xy, timeout_s=0.05)
    assert xy.stops == 1
    assert "xy_stage XY is still busy after 0.05 s" in str(info.value)
    assert "the stop was sent" in str(info.value)
    assert "device_timeout_ms" in str(info.value)
    assert any(
        m.startswith("xy_stage XY: stop after 0.05 s timeout; it still reads busy")
        for m in caplog.messages
    )
    assert executor.moving() == ("xy_stage XY",)
    with pytest.raises(MotionInProgressError):
        executor.do("shutter: open", _step("shutter: open", _never))


def test_timed_out_move_that_the_stop_settles_is_not_registered() -> None:
    executor, xy = _ex(), _Device()  # the stop clears it
    with pytest.raises(DeviceTimeoutError, match="the stop was sent"):
        _move(executor, xy, timeout_s=0.05)
    assert executor.moving() == ()
    assert executor.do("shutter: open", _step("shutter: open", lambda: None))


def test_timed_out_move_of_an_unstoppable_device_says_so_and_stays_registered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, turret = _ex(), _Device("Turret", stoppable=False)
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(DeviceTimeoutError, match="cannot be stopped from smc"),
    ):
        _move(executor, turret, timeout_s=0.05)
    assert turret.stops == 0
    assert executor.moving() == ("xy_stage Turret",)
    assert "gave up after 0.05 s timeout; it cannot be stopped from smc" in caplog.text


def test_interrupted_move_stops_and_reraises() -> None:
    executor, xy = _ex(), _Device()

    def interrupt(poll: int) -> None:
        if poll == 3:
            raise KeyboardInterrupt  # Ctrl-C, raised from the stub: no real signal

    xy.on_poll = interrupt
    with pytest.raises(KeyboardInterrupt):
        _move(executor, xy)
    assert xy.stops == 1
    assert executor.moving() == ()  # the stop settled it


@main_thread_only
def test_interrupt_between_command_and_wait_sends_the_stop() -> None:
    # #59 second review, finding 9: calls ran unprotected between the command
    # returning and the wait's try, so a Ctrl-C there left the stage moving.
    # The command is interrupt_main itself: the Ctrl-C is pending as the
    # command returns, and lands in the executor's frame, not in the stub.
    for _ in range(5):
        executor, xy = _ex(), _Device()
        xy.busy = True  # what the command did to the stage
        with pytest.raises(KeyboardInterrupt):
            executor.do(
                "move",
                Step("move", _thread.interrupt_main, lambda: "arrived", "dry"),
                motion=xy.motion,
                timeout_s=JOIN_S,
            )
        assert xy.stops == 1
        assert executor.moving() == ()


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


def test_a_readback_that_fails_after_arrival_sends_no_stop() -> None:
    executor, xy = _ex(), _Device(arrive_after=0)

    def readback() -> str:
        raise OSError("port closed")

    with pytest.raises(OSError, match="port closed"):
        executor.do(
            "move",
            Step("move", xy.command, readback, "dry"),
            motion=xy.motion,
            timeout_s=30.0,
        )
    assert xy.stops == 0
    assert executor.moving() == ()


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


def test_give_up_does_not_cancel_another_threads_queued_move() -> None:
    # #59 second review, finding 7: a failed command's give-up advanced the
    # stop generation, so another thread's move of the same stage, queued
    # behind it, was cancelled as "stopped before it started".
    executor, xy = _ex(), _Device(arrive_after=0)
    building, go = threading.Event(), threading.Event()

    def command_that_fails() -> None:
        raise RuntimeError("serial timeout")

    def failing_step() -> Step[str]:
        building.set()
        assert go.wait(JOIN_S)
        return Step("xy_stage: move A", command_that_fails, lambda: "A", "dry")

    first = _Thread(
        lambda: executor.do("move A", failing_step, motion=xy.motion, timeout_s=30.0)
    )
    assert building.wait(JOIN_S)  # the first move holds the lock
    second = _Thread(lambda: _move(executor, xy))  # records the generation, queues
    queued = not second.finished_within(0.2)
    go.set()
    first.join()
    second.join()
    assert queued
    assert isinstance(first.error, RuntimeError)
    assert xy.stops == 1  # the give-up stopped the stage
    assert second.error is None
    assert second.result == "arrived"
    assert xy.commands == 1


# --- stop ---------------------------------------------------------------------


def test_stop_during_a_move_makes_the_mover_raise_motion_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device()
    mover = _moving(executor, xy)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(xy.motion)
    mover.join()
    assert isinstance(mover.error, MotionStoppedError)
    assert mover.error.device == "xy_stage XY"
    assert xy.stops == 1
    assert "xy_stage XY: stop" in caplog.messages
    assert executor.moving() == ()


def test_stopped_move_that_times_out_raises_motion_stopped() -> None:
    # #68, finding 6: a move stopped during its wait whose stage still read
    # busy at the deadline (it decelerates slowly, or ignored the stop) raised
    # DeviceTimeoutError, which tells the operator to raise device_timeout_ms.
    # The caller asked for the stop, so the move ended as stopped.
    executor, xy = _ex(), _Device(stop_clears=False)

    def stop_during_the_wait(poll: int) -> None:
        xy.on_poll = None
        executor.stop(xy.motion)  # lock-free, as from another thread

    xy.on_poll = stop_during_the_wait
    with pytest.raises(MotionStoppedError) as info:
        _move(executor, xy, timeout_s=0.05)
    assert info.value.device == "xy_stage XY"
    assert info.value.detail == (
        "it was stopped, but it still read busy at the move's 0.05 s timeout, "
        "and on the timeout the stop was sent"
    )
    assert xy.stops == 2  # the caller's stop, then the give-up's
    assert executor.moving() == ("xy_stage XY",)  # still busy: it counts as moving


def test_stop_does_not_wait_for_the_lock() -> None:
    executor, xy = _ex(), _Device()
    holder, release = _hold_the_lock(executor)
    try:
        stopper = _Thread(lambda: executor.stop(xy.motion))
        stopped = stopper.finished_within(JOIN_S / 2)
    finally:
        release.set()
        holder.join()
    stopper.join()
    assert stopped, "stop() waited for the microscope lock"
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
    executor, xy = _ex(), _Device(arrive_after=0)
    executor.stop(xy.motion)
    assert _move(executor, xy) == "arrived"
    assert xy.commands == 1


def test_stop_racing_the_command_is_resent_after_it(
    caplog: pytest.LogCaptureFixture,
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
            timeout_s=30.0,
        )
    assert xy.stops == 2
    assert "xy_stage XY: stop resent after the command" in caplog.messages


def test_a_failing_resend_still_raises_motion_stopped() -> None:
    # #59 finding A: a failing FM-18 resend hid the MotionStoppedError. The
    # caller asked for a stop, so the move did not end as planned.
    executor, xy = _ex(), _Device(arrive_after=2)  # it arrives anyway

    def command_overtaken_by_a_stop() -> None:
        executor.stop(xy.motion)
        xy.stop_error = OSError("port closed")
        xy.command()

    with pytest.raises(MotionStoppedError):
        executor.do(
            "move",
            Step("move", command_overtaken_by_a_stop, lambda: 0, 0),
            motion=xy.motion,
            timeout_s=30.0,
        )
    assert xy.stops == 2


def test_a_failed_stop_still_ends_the_move_as_stopped() -> None:
    # #59 finding A: the caller asked for a stop, so the move did not end as
    # planned, even though the stop itself failed.
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")
    mover = _moving(executor, xy)
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
    executor, xy = _ex(), _Device(stop_clears=path != "timeout")
    handler = _OrderHandler(xy)
    LOGGER.addHandler(handler)
    try:
        if path == "stop":
            executor.stop(xy.motion)
        elif path == "timeout":
            with pytest.raises(DeviceTimeoutError):
                _move(executor, xy, timeout_s=0.05)
        elif path == "interrupt":

            def interrupt(poll: int) -> None:
                if poll == 1:
                    raise KeyboardInterrupt

            xy.on_poll = interrupt
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
                    timeout_s=30.0,
                )
    finally:
        LOGGER.removeHandler(handler)
    stop_lines = [(msg, stops) for msg, stops in handler.seen if "stop" in msg]
    assert stop_lines
    for message, stops_by_then in stop_lines:
        assert stops_by_then >= 1, f"logged before the stop: {message!r}"
    if path == "resend":
        assert ("xy_stage XY: stop resent after the command", 2) in stop_lines


class _InterruptOn(logging.Handler):
    """A second Ctrl-C while ``fragment`` is being logged, raised from the handler."""

    def __init__(self, fragment: str) -> None:
        super().__init__(logging.WARNING)
        self.fragment = fragment

    def emit(self, record: logging.LogRecord) -> None:
        if self.fragment in record.getMessage():
            raise KeyboardInterrupt


def test_a_second_ctrl_c_in_the_give_up_log_still_leaves_the_motion_tracked() -> None:
    # §13: a give-up sends the stop, registers the motion, and only then logs,
    # so an interrupt in the log line cannot drop the registration.
    executor, xy = _ex(), _Device(stop_clears=False)
    handler = _InterruptOn("stop after")
    LOGGER.addHandler(handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            _move(executor, xy, timeout_s=0.05)
    finally:
        LOGGER.removeHandler(handler)
    # The interrupted give-up runs again from the action's except: a second
    # stop is harmless, a missing one is not.
    assert xy.stops >= 1
    assert executor.moving() == ("xy_stage XY",)


def test_second_interrupt_during_give_up_stop_leaves_the_motion_registered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # #68, finding 2: the give-up registered the motion only after its stop
    # call had returned, so a second Ctrl-C inside that call left the stand
    # accepting actions while the stage still moved.
    executor, xy = _ex(), _Device(stop_clears=False)

    def first_ctrl_c(poll: int) -> None:
        if poll == 1:
            raise KeyboardInterrupt  # in the wait: the give-up sends the stop

    xy.on_poll = first_ctrl_c
    xy.stop_error = KeyboardInterrupt()  # the second Ctrl-C, inside that stop
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(KeyboardInterrupt),
    ):
        _move(executor, xy)
    xy.on_poll, xy.stop_error = None, None
    assert (xy.stops, xy.busy) == (1, True)
    assert executor.moving() == ("xy_stage XY",)
    assert (
        "xy_stage XY: stop after KeyboardInterrupt was interrupted before it "
        "returned, so it may not have reached the device" in caplog.messages
    )
    with pytest.raises(MotionInProgressError):
        executor.do("shutter: open", _step("shutter: open", _never))
    xy.busy = False  # once it reads idle, the stand is free again
    assert executor.do("shutter: open", _step("shutter: open", lambda: None))
    assert executor.moving() == ()


def test_an_interrupt_in_the_give_up_busy_check_counts_as_moving() -> None:
    # The busy check after a give-up's stop is interrupted before it answers:
    # unknown counts as moving, so the next action is refused.
    executor, xy = _ex(), _Device()

    def interrupt(poll: int) -> None:
        if poll <= 2:  # the wait's poll, then the give-up's check
            raise KeyboardInterrupt

    xy.on_poll = interrupt
    with pytest.raises(KeyboardInterrupt):
        _move(executor, xy)
    assert xy.stops == 1
    assert executor.moving() == ("xy_stage XY",)


def test_an_interrupt_in_the_post_stop_check_still_logs_the_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Adversarial review of round 3: a Ctrl-C in the busy check that follows a
    # sent stop cut the log line, so nothing recorded that the stop happened.
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.stops = 0

    def interrupt_after_the_stop(poll: int) -> None:
        if xy.stops:
            raise KeyboardInterrupt

    xy.on_poll = interrupt_after_the_stop
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(KeyboardInterrupt),
    ):
        executor.stop(xy.motion)
    assert xy.stops == 1
    assert "xy_stage XY: stop" in caplog.messages
    assert executor.moving() == ("xy_stage XY",)  # not known to be idle


def test_a_second_ctrl_c_in_the_stop_log_still_releases_the_motion() -> None:
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.stops = 0
    handler = _InterruptOn("xy_stage XY: stop")
    LOGGER.addHandler(handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            executor.stop(xy.motion)
    finally:
        LOGGER.removeHandler(handler)
    assert xy.stops == 1
    assert executor.moving() == ()


# --- motions that outlived their action ---------------------------------------


def test_every_action_is_refused_while_a_motion_outlived_its_action() -> None:
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    with pytest.raises(MotionInProgressError) as info:
        executor.do("shutter: open", _step("shutter: open", _never))
    assert info.value.moving == ("xy_stage XY",)
    assert "`xy_stage XY` is still moving" in str(info.value)
    assert "stop()" in str(info.value)
    assert "resume()" in str(info.value)
    assert info.value.how_to_force == ""
    with pytest.raises(MotionInProgressError):
        executor.acquire("camera: snap", _never)
    # A second move of the same stage is refused too: refuse, never queue.
    with pytest.raises(MotionInProgressError):
        _move(executor, xy)
    assert xy.commands == 1


def test_idle_motion_is_dropped_by_the_guard_and_the_next_action_runs() -> None:
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.busy = False
    assert (
        executor.do("shutter: open", _step("shutter: open", lambda: None)) == "readback"
    )
    assert executor.moving() == ()


def test_unreadable_busy_state_refuses_until_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.busy_error = OSError("port closed")
    with pytest.raises(MotionInProgressError) as info:
        executor.do("shutter: open", _step("shutter: open", _never))
    assert "could not be read" in str(info.value)
    assert "port closed" in str(info.value)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(xy.motion)
    assert "busy state cannot be read" in caplog.text
    assert "no longer tracked as moving" in caplog.text
    assert executor.moving() == ()
    assert executor.do("shutter: open", _step("shutter: open", lambda: None))


def test_a_move_whose_busy_state_breaks_mid_wait_is_stopped_and_tracked() -> None:
    executor, xy = _ex(), _Device()

    def port_closes(poll: int) -> None:
        if poll == 2:
            xy.busy_error = OSError("port closed")

    xy.on_poll = port_closes
    with pytest.raises(OSError, match="port closed"):
        _move(executor, xy)
    assert xy.stops == 1
    assert executor.moving() == ("xy_stage XY",)  # unknown counts as moving


def test_a_stage_whose_stop_raises_stays_registered_until_resume() -> None:
    # #59 finding 8: only a restart freed the stand. A failed stop leaves the
    # motion to the guard; resume() is the operator's way out.
    executor, xy = _ex(), _Device()
    _outlive(executor, xy)
    xy.busy_error = OSError("port closed")
    xy.stop_error = OSError("port closed")
    with pytest.raises(OSError, match="port closed"):
        executor.stop(xy.motion)
    assert executor.moving() == ("xy_stage XY",)
    executor.resume()
    assert executor.moving() == ()


def test_a_stage_still_busy_after_a_stop_stays_registered_until_resume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    _outlive(executor, xy)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(xy.motion)
    assert "it still reads busy, so it stays tracked as moving" in caplog.text
    assert executor.moving() == ("xy_stage XY",)
    executor.resume()
    assert executor.moving() == ()


@pytest.mark.parametrize("stoppable", [False, True])
def test_resume_drops_a_motion_that_never_settles_with_a_warning(
    stoppable: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # #59 second review, finding 4: a State device still busy after its
    # deadline stayed registered for good; neither stop() nor resume()
    # released it. A stage that keeps answering busy is the same case. (One
    # registration at most: while one exists, the guard refuses every action.)
    executor = _ex()
    device = _Device("Turret", stoppable=stoppable, stop_clears=False)
    _outlive(executor, device)
    with pytest.raises(MotionInProgressError):
        executor.acquire("camera: snap", _never)
    stops = device.stops
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.resume()
    assert executor.moving() == ()
    assert "xy_stage Turret: no longer tracked as moving (resume)" in caplog.text
    assert executor.acquire("camera: snap", lambda: "frame") == "frame"
    assert device.busy  # resume() starts and stops nothing
    assert device.stops == stops


def test_resume_releases_an_unreadable_motion_that_cannot_be_stopped() -> None:
    # #59 finding 8: a State device (no stop) with an unreadable busy state.
    executor, turret = _ex(), _Device("Turret", stoppable=False)
    _outlive(executor, turret)
    turret.busy_error = OSError("port closed")
    with pytest.raises(MotionInProgressError):
        executor.acquire("camera: snap", _never)
    with pytest.raises(HardwareError, match="cannot be stopped from smc"):
        executor.stop(turret.motion)
    executor.resume()
    assert executor.moving() == ()
    assert executor.acquire("camera: snap", lambda: "frame") == "frame"


def test_stop_without_a_stop_callable_says_it_cannot_and_cancels_nothing() -> None:
    executor, turret = _ex(), _Device("Turret", stoppable=False)
    mover = _moving(executor, turret)
    try:
        with pytest.raises(HardwareError, match="cannot be stopped from smc"):
            executor.stop(turret.motion)
    finally:
        turret.busy = False
    mover.join()
    # Nothing was stopped, so the set in flight ends as it was asked.
    assert mover.error is None
    assert mover.result == "arrived"


def test_stop_that_raises_propagates() -> None:
    executor, xy = _ex(), _Device()
    xy.stop_error = OSError("port closed")
    with pytest.raises(OSError, match="port closed"):
        executor.stop(xy.motion)


# --- halt ---------------------------------------------------------------------


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


def _actions(
    executor: Executor, device: _Device, shutter: _Shutter
) -> dict[str, Callable[[], object]]:
    """One action of each kind, for the tests that queue one: an open, a move, a snap."""
    return {
        "open": lambda: _open(executor, shutter),
        "move": lambda: _move(executor, device),
        "snap": lambda: executor.acquire("camera: snap", _never),
    }


@pytest.mark.parametrize("when", ["called while halted", "halted while queued"])
@pytest.mark.parametrize("kind", ["open", "move", "snap"])
def test_action_queued_while_halted_does_not_run_after_resume(
    kind: str, when: str
) -> None:
    # #68, finding 3: an action waiting for the lock ran as soon as the
    # operator resumed, although it was called while halted, or was waiting
    # when the halt landed. resume() starts nothing.
    executor, xy, shutter = _ex(), _Device(arrive_after=0), _Shutter()
    holder, release = _hold_the_lock(executor)
    try:
        if when == "called while halted":
            executor.halt()
        queued = _Thread(_actions(executor, xy, shutter)[kind])
        if when == "called while halted":
            # Refused at its first lock poll (#80), so before the resume.
            assert queued.finished_within(JOIN_S / 2)
        else:
            assert not queued.finished_within(0.2)  # queued behind the snap
            executor.halt()
        executor.resume()
    finally:
        release.set()
        holder.join()
    queued.join()
    assert isinstance(queued.error, MicroscopeHaltedError)
    assert (shutter.sent, xy.commands) == ([], 0)
    assert _move(executor, xy) == "arrived"  # a call made after resume() runs
    assert xy.commands == 1


@pytest.mark.parametrize("when", ["called while halted", "halted after the call"])
@pytest.mark.parametrize("kind", ["open", "move", "snap"])
def test_a_halt_and_resume_before_the_lock_is_taken_still_refuse(
    kind: str, when: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adversarial review of #80: a queued action is now refused at its lock
    # polls, so the test above no longer reaches the check made once the
    # lock is taken. That check alone sees a halt and a resume() that land
    # between the call and the lock (within one poll, or before a free lock
    # is taken), and it must compare the halt mark, not the flag (#68).
    executor, xy, shutter = _ex(), _Device(arrive_after=0), _Shutter()
    halt_mark = executor._halt_mark

    def halt_and_resume_around_the_call() -> int | None:
        if when == "called while halted":
            executor.halt()
        mark = halt_mark()
        if when == "halted after the call":
            executor.halt()
        executor.resume()
        return mark

    monkeypatch.setattr(executor, "_halt_mark", halt_and_resume_around_the_call)
    with pytest.raises(MicroscopeHaltedError):
        _actions(executor, xy, shutter)[kind]()
    monkeypatch.undo()
    assert (shutter.sent, xy.commands) == ([], 0)
    assert _move(executor, xy) == "arrived"  # a call made after resume() runs


@pytest.mark.parametrize("kind", ["open", "move", "snap"])
def test_action_called_while_halted_behind_a_moving_holder_raises_halted(
    kind: str,
) -> None:
    # #80 (FM-64): a caller that found the lock held by a move was refused
    # with MotionInProgressError before its halt was looked at. That is a
    # SafetyRefusedError, so `except SafetyRefusedError: skip` carried on
    # through the emergency stop.
    executor, xy = _ex(), _Device()
    z, shutter = _Device("Z", arrive_after=0), _Shutter()
    mover = _moving(executor, xy)
    try:
        executor.halt()
        with pytest.raises(MicroscopeHaltedError):
            _actions(executor, z, shutter)[kind]()
    finally:
        xy.busy = False
        mover.join()
    assert (shutter.sent, z.commands) == ([], 0)
    assert mover.result == "arrived"  # the halt itself stops nothing


@pytest.mark.parametrize("when", ["halted while queued", "called while halted"])
@pytest.mark.parametrize("kind", ["open", "move", "snap"])
def test_action_halted_while_queued_behind_a_slow_holder_raises_halted_not_busy(
    kind: str, when: str
) -> None:
    # #80: a caller queued behind a slow holder (a snap) never looked at its
    # halt while it waited, so it waited the whole lock_timeout_s and raised
    # MicroscopeBusyError, "the device driver may be hung". "called while
    # halted" is the issue's reproduction.
    executor = _ex(lock_timeout_s=5.0)
    z, shutter = _Device("Z", arrive_after=0), _Shutter()
    holder, release = _hold_the_lock(executor)
    try:
        if when == "called while halted":
            executor.halt()
        queued = _Thread(_actions(executor, z, shutter)[kind])
        if when == "halted while queued":
            assert not queued.finished_within(0.2)  # queued behind the snap
            executor.halt()
        # One lock poll is 50 ms; 1 s is well short of lock_timeout_s (FM-44).
        refused = queued.finished_within(1.0)
    finally:
        release.set()
        holder.join()
    queued.join()
    assert refused, f"the halted action kept waiting for the lock: {queued.error!r}"
    assert isinstance(queued.error, MicroscopeHaltedError)
    assert (shutter.sent, z.commands) == ([], 0)


@pytest.mark.parametrize("kind", ["open", "move", "snap"])
def test_halt_and_resume_while_queued_still_raise_halted(kind: str) -> None:
    # #80: the check at each lock poll compares the halt epoch, not the flag:
    # a resume() right after the halt does not let the queued action through,
    # nor leave it waiting for the lock, since it was halted after its call.
    executor = _ex(lock_timeout_s=5.0)
    z, shutter = _Device("Z", arrive_after=0), _Shutter()
    holder, release = _hold_the_lock(executor)
    try:
        queued = _Thread(_actions(executor, z, shutter)[kind])
        assert not queued.finished_within(0.2)  # queued behind the snap
        executor.halt()
        executor.resume()
        refused = queued.finished_within(1.0)
    finally:
        release.set()
        holder.join()
    queued.join()
    assert refused, f"the halted action kept waiting for the lock: {queued.error!r}"
    assert isinstance(queued.error, MicroscopeHaltedError)
    assert (shutter.sent, z.commands) == ([], 0)


def test_halted_is_not_a_safety_refusal() -> None:
    # #59 finding 14: `except SafetyRefusedError: skip the target` swallowed
    # the emergency stop.
    assert issubclass(MicroscopeHaltedError, HardwareError)
    assert not issubclass(MicroscopeHaltedError, SafetyRefusedError)


def test_halt_still_allows_reads_stops_safe_calls_and_waits() -> None:
    executor, xy = _ex(), _Device(stop_clears=False)
    _outlive(executor, xy)
    executor.halt()
    assert executor.read(lambda: 42) == 42
    assert executor.safe(_CLOSE, lambda: False) is False
    executor.stop(xy.motion)
    assert xy.stops == 2  # the give-up's, then this one
    xy.busy = False
    executor.wait(xy.motion, 1.0)
    assert executor.moving() == ()
    assert executor.halted


@pytest.mark.parametrize("kind", ["mutation", "acquisition", "motion"])
def test_a_halt_during_the_guard_refuses_before_the_command(kind: str) -> None:
    # #59 finding 7: a halt that landed while the guard polled let a
    # set_open(True) or a snap run after it.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    _outlive(executor, z)
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
        closer = _Thread(lambda: executor.safe(_CLOSE, lambda: False))
        closed = closer.finished_within(JOIN_S / 2)
    finally:
        release.set()
        holder.join()
    assert closed
    assert closer.error is None


# --- a close wins over an open ----------------------------------------------


class _Shutter:
    """A stub shutter whose commands are recorded in the order they reach it."""

    def __init__(self) -> None:
        self.is_open = False
        self.sent: list[bool] = []
        self.close_error: BaseException | None = None
        self.close = SafeCall("Shutter", "shutter: close", self._close)

    def _close(self) -> None:
        self.sent.append(False)
        if self.close_error is not None:
            raise self.close_error
        self.is_open = False

    def open_step(self, before: Callable[[], None] | None = None) -> Step[bool]:
        def send() -> None:
            if before is not None:
                before()  # lands on the device first
            self.sent.append(True)
            self.is_open = True

        return Step("shutter: open", send, lambda: self.is_open, True)


def _open(
    executor: Executor, shutter: _Shutter, before: Callable[[], None] | None = None
) -> bool:
    return executor.do(
        "shutter: open", shutter.open_step(before), overridden_by=shutter.close
    )


@pytest.mark.parametrize("halted", [False, True])
def test_close_while_an_open_is_sent_is_resent_and_the_open_raises(
    halted: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # #59 second review, finding 12: an open past its halt check could land
    # after an emergency close, and the shutter ended open.
    executor, shutter = _ex(), _Shutter()

    def emergency_close() -> None:
        if halted:
            executor.halt()
        executor.safe(shutter.close, lambda: shutter.is_open)

    expected = MicroscopeHaltedError if halted else HardwareError
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(expected) as info,
    ):
        _open(executor, shutter, before=emergency_close)
    assert shutter.sent == [False, True, False]
    assert shutter.is_open is False
    assert "shutter: close resent after `shutter: open`" in caplog.messages
    if not halted:
        assert type(info.value) is HardwareError
        assert "was sent again after it" in str(info.value)


@pytest.mark.parametrize(
    "order", ["close halt resume", "halt close resume", "halt resume close"]
)
def test_halt_and_resume_during_an_overridable_send_raise_halted(order: str) -> None:
    # #80 (#75 item 6): the open chose between MicroscopeHaltedError and
    # HardwareError from the halt flag, so a halt and a resume() that both
    # landed while it was sent raised HardwareError, not the
    # MicroscopeHaltedError that do() documents for a halt since the call.
    executor, shutter = _ex(), _Shutter()
    calls: dict[str, Callable[[], object]] = {
        "close": lambda: executor.safe(shutter.close, lambda: shutter.is_open),
        "halt": executor.halt,
        "resume": executor.resume,
    }

    def during_the_send() -> None:
        for name in order.split():
            calls[name]()

    with pytest.raises(MicroscopeHaltedError):
        _open(executor, shutter, before=during_the_send)
    assert shutter.sent == [False, True, False]  # the close was sent again
    assert shutter.is_open is False


def test_a_close_while_an_open_waits_for_the_lock_cancels_it() -> None:
    executor, shutter = _ex(), _Shutter()
    holder, release = _hold_the_lock(executor)
    try:
        opener = _Thread(lambda: _open(executor, shutter))
        assert not opener.finished_within(0.2)  # queued behind the snap
        executor.safe(shutter.close, lambda: shutter.is_open)
    finally:
        release.set()
        holder.join()
    opener.join()
    assert isinstance(opener.error, HardwareError)
    assert "was not sent" in str(opener.error)
    assert shutter.sent == [False]


def test_a_close_before_the_open_is_called_does_not_cancel_it() -> None:
    executor, shutter = _ex(), _Shutter()
    executor.safe(shutter.close, lambda: shutter.is_open)
    assert _open(executor, shutter) is True
    assert shutter.sent == [False, True]


def test_a_close_racing_an_open_that_raises_is_still_resent() -> None:
    executor, shutter = _ex(), _Shutter()

    def close_then_fail() -> None:
        executor.safe(shutter.close, lambda: shutter.is_open)
        shutter.is_open = True  # the open reached the device ...
        raise OSError("serial timeout")  # ... and then the driver raised

    with pytest.raises(OSError, match="serial timeout"):
        _open(executor, shutter, before=close_then_fail)
    assert shutter.sent == [False, False]
    assert shutter.is_open is False


def test_a_close_that_fails_raises_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, shutter = _ex(), _Shutter()
    shutter.close_error = OSError("port closed")
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER.name),
        pytest.raises(OSError, match="port closed"),
    ):
        executor.safe(shutter.close, lambda: shutter.is_open)
    assert "shutter: close failed (OSError('port closed'))" in caplog.messages


def test_a_close_is_logged_after_it_is_sent(caplog: pytest.LogCaptureFixture) -> None:
    executor, shutter = _ex(), _Shutter()
    order: list[str] = []
    handler = _OnLog("shutter: close", lambda: order.append(f"log {shutter.sent}"))
    LOGGER.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER.name):
            executor.safe(shutter.close, lambda: shutter.is_open)
    finally:
        LOGGER.removeHandler(handler)
    assert order == ["log [False]"]


def test_an_action_cannot_have_both_a_motion_and_a_safe_call() -> None:
    xy = _Device()
    with pytest.raises(ValueError, match="overridden by its own stop"):
        _ex().do(
            "move", xy.step(), motion=xy.motion, timeout_s=1.0, overridden_by=_CLOSE
        )
    assert xy.commands == 0


# --- dry-run -----------------------------------------------------------------


def test_dry_run_registers_no_motion_and_waits_and_stops_still_run() -> None:
    executor, xy = _ex(dry_run=True), _Device()
    xy.busy = True  # moved by hand while a dry-run session watches
    assert _move(executor, xy) == "dry"
    assert xy.commands == 0
    assert executor.moving() == ()
    assert executor.acquire("camera: snap", lambda: "frame") == "frame"
    # wait() is a read: it watches the real device, in dry-run too.
    with pytest.raises(DeviceTimeoutError):
        executor.wait(xy.motion, 0.05)
    executor.stop(xy.motion)
    assert xy.stops == 1
    executor.wait(xy.motion, 1.0)
    assert executor.safe(_CLOSE, lambda: False) is False


def test_dry_run_move_stopped_while_queued_is_cancelled_too() -> None:
    executor, xy = _ex(dry_run=True), _Device()
    holder, release = _hold_the_lock(executor)
    try:
        mover = _Thread(lambda: _move(executor, xy))
        assert not mover.finished_within(0.2)
        executor.stop(xy.motion)
    finally:
        release.set()
        holder.join()
    mover.join()
    assert isinstance(mover.error, MotionStoppedError)


# --- a stop that re-enters the registry lock (FM-70) ---------------------------


def test_halt_and_stop_inside_a_registry_section_return() -> None:
    # #76 (FM-70): Python runs a Ctrl-C handler on the main thread, between
    # two of its bytecodes. With a plain Lock, a handler that called halt()
    # or stop() while its thread was inside a registry section (every move
    # enters one) hung there for ever, and no stop went out.
    executor, xy = _ex(), _Device()

    def handler_inside_a_section() -> tuple[str, ...]:
        with executor._registry_lock:
            executor.halt()
            executor.stop(xy.motion)
            return executor.moving()

    handler = _Thread(handler_inside_a_section)
    assert handler.finished_within(JOIN_S), "the stop deadlocked on the registry lock"
    assert handler.error is None
    assert handler.result == ()
    assert xy.stops == 1
    assert executor.halted


class _StopLandsInTheDrop(dict[str, object]):
    """The registry, with a stop landing inside the first lookup made of it.

    It stands for a Ctrl-C handler that runs between a drop's identity check
    and its delete: the entry is read first, then the handler's stop drops
    that same entry, and the check goes on with what it read.
    """

    def __init__(self, registry: dict[str, object], stop: Callable[[], None]) -> None:
        super().__init__(registry)
        self._stop: Callable[[], None] | None = stop

    def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
        value = super().get(key, default)
        stop, self._stop = self._stop, None
        if stop is not None:
            stop()
        return value


def test_a_stop_landing_inside_a_drop_does_not_raise() -> None:
    # #76 (FM-70, re-entry): once the lock is re-entrant, a handler can run
    # inside a section and change the registry under it. A drop that checked
    # the entry and then deleted it raised KeyError when the handler's stop
    # had dropped the same entry in between.
    executor, xy, z = _ex(), _Device(), _Device("Z", arrive_after=0)
    _outlive(executor, xy)
    xy.busy = False  # the next action's guard sees it idle and drops it
    executor._registry = _StopLandsInTheDrop(  # type: ignore[assignment]
        executor._registry,  # type: ignore[arg-type]
        lambda: executor.stop(xy.motion),
    )
    mover = _Thread(lambda: _move(executor, z))
    assert mover.finished_within(JOIN_S), "the stop deadlocked inside the drop"
    assert mover.error is None
    assert mover.result == "arrived"
    assert xy.stops == 2  # the give-up's, then the handler's
    assert executor.moving() == ()


# --- several stops, then their lines (FM-62) ----------------------------------


class _StopsSeen(logging.Handler):
    """Records each line with how many stops each device had seen by then."""

    def __init__(self, *devices: _Device) -> None:
        super().__init__()
        self.devices = devices
        self.seen: list[tuple[str, tuple[int, ...]]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.seen.append((record.getMessage(), tuple(d.stops for d in self.devices)))


def test_hold_log_writes_the_halt_and_every_stop_line_after_the_last_stop() -> None:
    # #76 (FM-62): the emergency stop wrote each line as it went, so a blocked
    # console held the Z stop behind the halt's line and the XY stop's.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    handler = _StopsSeen(xy, z)
    LOGGER.addHandler(handler)

    def emergency_stop() -> str:
        executor.halt()
        executor.stop(xy.motion)
        executor.stop(z.motion)
        return "stopped"

    try:
        result = executor.hold_log(emergency_stop)
    finally:
        LOGGER.removeHandler(handler)
    assert result == "stopped"
    assert handler.seen == [
        ("microscope: halted", (1, 1)),
        ("xy_stage XY: stop", (1, 1)),
        ("xy_stage Z: stop", (1, 1)),
    ]


def test_hold_log_writes_its_lines_when_the_stops_raise_and_holds_nothing_after(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A second Ctrl-C during the stops: the lines held so far are still
    # written, and the thread does not go on holding the ones that follow.
    executor, xy, z = _ex(), _Device(), _Device("Z")

    def interrupted() -> None:
        executor.stop(xy.motion)
        raise KeyboardInterrupt  # raised from the stub, not sent as a signal

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        with pytest.raises(KeyboardInterrupt):
            executor.hold_log(interrupted)
        assert caplog.messages == ["xy_stage XY: stop"]
        executor.stop(z.motion)
        assert caplog.messages == ["xy_stage XY: stop", "xy_stage Z: stop"]


def test_a_second_ctrl_c_while_the_held_lines_are_written_leaves_nothing_held(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Adversarial review of #76 (FM-65): the hold is cleared before the first
    # held line is written. Cleared after them, an interrupt in that writing
    # left the thread holding, and every later line on it was kept for good.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    handler = _InterruptOn("xy_stage XY: stop")
    LOGGER.addHandler(handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            executor.hold_log(lambda: executor.stop(xy.motion))
    finally:
        LOGGER.removeHandler(handler)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.stop(z.motion)
    assert caplog.messages == ["xy_stage Z: stop"]
    assert (xy.stops, z.stops) == (1, 1)


def test_a_nested_hold_log_leaves_the_writing_to_the_outer_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A handler's emergency stop inside close()'s stops: its lines wait for
    # the outer stops too, and nothing is written twice.
    executor, xy, z = _ex(), _Device(), _Device("Z")
    after_inner: list[str] = []

    def outer() -> None:
        executor.hold_log(lambda: executor.stop(xy.motion))
        after_inner.extend(caplog.messages)
        executor.stop(z.motion)

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        executor.hold_log(outer)
    assert after_inner == []
    assert caplog.messages == ["xy_stage XY: stop", "xy_stage Z: stop"]


def test_hold_log_does_not_hold_another_threads_lines() -> None:
    # Held per thread: a stop on another thread (a UI) is not kept waiting
    # for this one's lines, nor are its own lines kept back.
    executor, z = _ex(), _Device("Z")
    written = threading.Event()
    handler = _OnLog("xy_stage Z: stop", written.set)
    LOGGER.addHandler(handler)
    stoppers: list[_Thread] = []

    def run() -> bool:
        stoppers.append(_Thread(lambda: executor.stop(z.motion)))
        return written.wait(JOIN_S)

    try:
        written_inside = executor.hold_log(run)
    finally:
        LOGGER.removeHandler(handler)
    stoppers[0].join()
    assert written_inside, "another thread's line was held"
    assert stoppers[0].error is None
    assert z.stops == 1
