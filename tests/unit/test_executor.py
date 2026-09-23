"""The executor: one lock around every hardware call, and dry-run that reads but never moves."""

from __future__ import annotations

import logging
import threading

import pytest

from smc.hardware.safety import Executor

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
