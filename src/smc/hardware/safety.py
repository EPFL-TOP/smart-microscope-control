"""Safety guards and the executor every hardware call goes through (design §7).

The guards live in the layer, not in the tools, so a tool cannot opt out of
them (CLAUDE.md, architecture rules):

* **Jog guard.** A relative XY move larger than ``max_jog_um`` is refused
  unless the call site passes ``force=True``. A mistyped relative move
  (``1000`` for ``100``) is the classic way to drive a stage out of the well
  or into the plate holder; an absolute move is not guarded, because
  crossing a plate is legitimate travel.
* **Soft limits.** A target outside the profile's travel range is refused
  and cannot be forced: the limits describe where the objective or the
  holder would collide, and no tool knows better at run time. No limits
  configured means no check, not a default range.
* **Non-finite targets** (``nan``, ``inf``) are refused whatever the limits:
  ``nan`` compares false against every bound and would otherwise slip
  through each check and reach the stage driver.

``Executor`` serialises hardware access behind one re-entrant lock and
implements dry-run. In dry-run, mutations are logged and skipped, but
*reads still reach the hardware*: a dry-run session shows the real stage
position and never moves it.

* **No action while something moves** (design §13). The ``Executor`` keeps
  a registry of the motions the layer started; a mutation or an acquisition
  is refused until each of them reports idle. Waits take the lock for each
  poll only, stops never take it, and every wait that gives up sends the
  stop first.

This module is pure: it imports no Micro-Manager code.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeBusyError,
    MicroscopeHaltedError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)

__all__ = ["POLL_INTERVAL_S", "Executor", "Motion", "Safety"]

T = TypeVar("T")


def _check_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"{name} must be finite with low <= high, got {bounds!r}")
    return (float(low), float(high))


class Safety:
    """The guards for one microscope, built from plain numbers.

    Plain arguments keep this module independent of the profile model; the
    ``Microscope`` facade builds a ``Safety`` from ``profile.safety``.
    """

    def __init__(
        self,
        *,
        max_jog_um: float,
        z_soft_limits_um: tuple[float, float] | None = None,
        xy_soft_limits_um: tuple[tuple[float, float], tuple[float, float]]
        | None = None,
    ) -> None:
        if not math.isfinite(max_jog_um) or max_jog_um < 0:
            raise ValueError(f"max_jog_um must be finite and >= 0, got {max_jog_um!r}")
        self.max_jog_um = float(max_jog_um)
        self.z_soft_limits_um = (
            None
            if z_soft_limits_um is None
            else _check_range("z_soft_limits_um", z_soft_limits_um)
        )
        self.xy_soft_limits_um = (
            None
            if xy_soft_limits_um is None
            else (
                _check_range("xy_soft_limits_um[x]", xy_soft_limits_um[0]),
                _check_range("xy_soft_limits_um[y]", xy_soft_limits_um[1]),
            )
        )

    def check_jog_um(self, dx_um: float, dy_um: float, *, force: bool) -> None:
        """Refuse a relative XY move above ``max_jog_um`` unless forced.

        Raises:
            SafetyRefusedError: The jog is too large (forceable), or not finite
                (never forceable).
        """
        if not (math.isfinite(dx_um) and math.isfinite(dy_um)):
            raise SafetyRefusedError(
                f"jog ({dx_um}, {dy_um}) µm is not a finite distance"
            )
        size = max(abs(dx_um), abs(dy_um))
        if size > self.max_jog_um and not force:
            raise SafetyRefusedError(
                f"jog ({dx_um}, {dy_um}) µm exceeds the jog limit of {self.max_jog_um} µm",
                how_to_force="pass force=True",
            )

    def check_xy_target_um(self, x_um: float, y_um: float) -> None:
        """Refuse an XY target outside the soft limits; never forceable.

        Raises:
            SafetyRefusedError: The target is outside the limits or not finite.
        """
        if not (math.isfinite(x_um) and math.isfinite(y_um)):
            raise SafetyRefusedError(
                f"XY target ({x_um}, {y_um}) µm is not a finite position"
            )
        if self.xy_soft_limits_um is None:
            return
        (x_low, x_high), (y_low, y_high) = self.xy_soft_limits_um
        if not (x_low <= x_um <= x_high and y_low <= y_um <= y_high):
            raise SafetyRefusedError(
                f"XY target ({x_um}, {y_um}) µm is outside the soft limits "
                f"x [{x_low}, {x_high}], y [{y_low}, {y_high}] µm; change "
                f"[safety] in the profile if the limits are wrong"
            )

    def check_z_target_um(self, z_um: float) -> None:
        """Refuse a Z target outside the soft limits; never forceable.

        Raises:
            SafetyRefusedError: The target is outside the limits or not finite.
        """
        if not math.isfinite(z_um):
            raise SafetyRefusedError(f"Z target {z_um} µm is not a finite position")
        if self.z_soft_limits_um is None:
            return
        low, high = self.z_soft_limits_um
        if not low <= z_um <= high:
            raise SafetyRefusedError(
                f"Z target {z_um} µm is outside the soft limits [{low}, {high}] µm; "
                f"change [safety] in the profile if the limits are wrong"
            )


#: How often a wait asks the device whether it is still busy.
POLL_INTERVAL_S = 0.01


@dataclass(frozen=True, slots=True)
class Motion:
    """A device the layer sets moving, as the ``Executor`` sees it (design §13).

    The backend builds it from its device calls; the ``Executor`` needs only
    these callables, which keeps this module free of Micro-Manager code.
    """

    device: str  # registry key: the device label
    name: str  # for messages: "xy_stage XY", "z Z", "Dichroic"
    is_busy: Callable[[], bool]
    stop: Callable[[], None] | None  # None: cannot be stopped (a State device)


@dataclass
class _MotionState:
    """One registered motion. Its flags change only under the registry lock."""

    motion: Motion
    #: A stop was issued while it was registered: its waiter must not report
    #: an arrival, because the move did not complete.
    stopped: bool = False
    #: A stop reached the device without raising: an unreadable busy state no
    #: longer locks the stand (FM-19).
    stop_sent: bool = False


class Executor:
    """Runs every backend call under the microscope's lock, honouring dry-run.

    One ``RLock`` per microscope makes the capabilities safe to call from a
    UI thread and a worker at once (they stay sequential); re-entrant, so a
    mutation can check, command and read its own state in one critical
    section. The lock is held for checks and commands, never for the wait
    that follows a move: a plate traverse must not block readers or a stop
    (FM-16).

    The motion registry has its own small lock. Lock order: a thread holding
    the microscope lock may take the registry lock, never the other way
    round, and no device call is made while the registry lock is held.
    """

    def __init__(
        self,
        *,
        dry_run: bool,
        lock: threading.RLock,
        logger: logging.Logger,
        lock_timeout_s: float = 60.0,
    ) -> None:
        if not math.isfinite(lock_timeout_s) or lock_timeout_s <= 0:
            # nan would make acquire() raise, inf would wait for ever (FM-32).
            raise ValueError(
                f"lock_timeout_s must be finite and > 0, got {lock_timeout_s!r}"
            )
        self._dry_run = dry_run
        self._lock = lock
        self._logger = logger
        self._lock_timeout_s = float(lock_timeout_s)
        self._registry: dict[str, _MotionState] = {}
        self._registry_lock = threading.Lock()
        self._halted = False
        # Who holds the microscope lock, for MicroscopeBusyError. Written only
        # by the thread that owns the lock, at its outermost acquisition.
        self._holder: tuple[str, float] | None = None
        self._depth = 0

    @property
    def dry_run(self) -> bool:
        """Whether mutations are skipped; fixed at construction."""
        return self._dry_run

    @property
    def halted(self) -> bool:
        """Whether mutations and acquisitions are refused until ``resume()``."""
        return self._halted

    @contextmanager
    def _locked(self, description: str) -> Iterator[None]:
        """Hold the microscope lock, giving up after ``lock_timeout_s`` (FM-15)."""
        if not self._lock.acquire(timeout=self._lock_timeout_s):
            holder = self._holder
            if holder is None:
                # Released just now, or held by a caller outside the Executor.
                raise MicroscopeBusyError("an unknown caller", self._lock_timeout_s)
            what, since = holder
            raise MicroscopeBusyError(what, time.monotonic() - since)
        try:
            self._depth += 1
            if self._depth == 1:
                self._holder = (description, time.monotonic())
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._holder = None
            self._lock.release()

    def do(
        self,
        description: str,
        action: Callable[[], T],
        *,
        dry_result: T,
        motion: Motion | None = None,
    ) -> T:
        """Log and perform a mutation; in dry-run, log it and return ``dry_result``.

        With ``motion``, the device is registered as moving before the
        command is sent, so the guard knows about it even if the command is
        interrupted, and a command that raises sends the stop (FM-17).

        Raises:
            MicroscopeHaltedError: The microscope is halted.
            MotionInProgressError: A device the layer moved is still moving.
            MotionStoppedError: A ``stop()`` arrived while the command was being
                sent; the stop was sent again after it (FM-18).
            MicroscopeBusyError: The lock was not free within ``lock_timeout_s``.
        """
        with self._locked(description):
            self._refuse_if_halted()
            self._refuse_if_moving()
            if self._dry_run:
                self._logger.info("[dry-run] %s", description)
                return dry_result
            self._logger.info("%s", description)
            if motion is None:
                return action()
            state = self._register(motion)
            try:
                result = action()
            except BaseException as exc:
                # The command may have reached the device before it raised.
                self._stop_after(motion, state, f"{type(exc).__name__} in the command")
                raise
            with self._registry_lock:
                stopped = state.stopped
            if stopped:
                # The stop did not wait for the lock and may have reached the
                # device before this command did.
                self._logger.warning("%s: stop resent after the command", motion.name)
                self._send_stop(motion, state)
                raise MotionStoppedError(motion.name)
            return result

    def read(
        self, action: Callable[[], T], *, at_rest: bool = False, description: str = ""
    ) -> T:
        """Perform a read under the lock; reads reach the hardware even in dry-run.

        ``at_rest=True`` marks an acquisition: it is refused while halted or
        while anything moves, in dry-run too, because a frame taken during a
        move is a wrong result, not a skipped mutation.

        Raises:
            MicroscopeHaltedError: ``at_rest`` and the microscope is halted.
            MotionInProgressError: ``at_rest`` and a device is still moving.
            MicroscopeBusyError: The lock was not free within ``lock_timeout_s``.
        """
        with self._locked(description or "a read"):
            if at_rest:
                self._refuse_if_halted()
                self._refuse_if_moving()
            return action()

    def wait(self, motion: Motion, timeout_s: float) -> None:
        """Poll ``motion`` until idle, taking the lock for each poll only (FM-16).

        Every exit other than "idle" sends the stop first (FM-17): a stage never
        keeps moving because the program that moved it gave up.

        Raises:
            ValueError: ``timeout_s`` is negative or not finite.
            MotionStoppedError: The motion went idle because it was stopped.
            DeviceTimeoutError: Still busy at the deadline; the stop was sent,
                or the message says why not.
        """
        if not math.isfinite(timeout_s) or timeout_s < 0:
            # nan would make the deadline comparison always false: an endless wait.
            raise ValueError(f"timeout_s must be finite and >= 0, got {timeout_s!r}")
        if self._dry_run:
            return
        with self._registry_lock:
            # Kept for the whole wait: if another thread's guard drops the
            # motion once it is idle, its "stopped" flag must still be seen.
            state = self._registry.get(motion.device)
        deadline = time.monotonic() + timeout_s
        description = f"{motion.name}: wait"
        try:
            while True:
                with self._locked(description):
                    busy = bool(motion.is_busy())
                if not busy or time.monotonic() >= deadline:
                    break
                time.sleep(POLL_INTERVAL_S)
        except BaseException as exc:
            self._stop_after(motion, state, f"{type(exc).__name__} during the wait")
            raise
        if busy:
            outcome = self._stop_after(motion, state, f"{timeout_s:g} s timeout")
            raise DeviceTimeoutError(
                f"{motion.name} is still busy after {timeout_s:g} s; {outcome}. "
                f"If the move is legitimately long, raise [micromanager] "
                f"device_timeout_ms in the profile; otherwise check the device "
                f"and its cabling."
            )
        if state is not None:
            self._drop(state)
            with self._registry_lock:
                stopped = state.stopped
            if stopped:
                raise MotionStoppedError(motion.name)

    def stop(self, motion: Motion) -> None:
        """Stop ``motion`` now, without the microscope lock; never refused.

        It runs in dry-run too: a stop cannot create motion. A stop that
        raises propagates, because a failed stop is a finding.

        Raises:
            HardwareError: The device cannot be stopped from smc.
        """
        if motion.stop is None:
            raise HardwareError(f"{motion.name} cannot be stopped from smc")
        self._logger.warning("%s: stop", motion.name)
        with self._registry_lock:
            state = self._registry.get(motion.device)
        self._send_stop(motion, state)

    def halt(self) -> None:
        """Refuse every mutation and acquisition until ``resume()``; reads still run."""
        with self._registry_lock:
            self._halted = True
        self._logger.warning("microscope: halted")

    def resume(self) -> None:
        """Clear the halt; starts nothing."""
        with self._registry_lock:
            self._halted = False
        self._logger.warning("microscope: resumed")

    def moving(self) -> tuple[str, ...]:
        """Names of the motions not yet seen idle; asks no device, never blocks."""
        with self._registry_lock:
            return tuple(state.motion.name for state in self._registry.values())

    def _refuse_if_halted(self) -> None:
        if self._halted:
            raise MicroscopeHaltedError()

    def _refuse_if_moving(self) -> None:
        """The guard: poll every registered motion, drop the idle ones, refuse the rest.

        Called under the microscope lock. An unreadable busy state counts as
        moving (unknown beats guessed), unless a stop was already sent to it:
        a broken device must not lock the stand for good (FM-19).
        """
        with self._registry_lock:
            states = list(self._registry.values())
        moving: list[str] = []
        details: list[str] = []
        for state in states:
            motion = state.motion
            try:
                busy = bool(motion.is_busy())
            except Exception as exc:
                with self._registry_lock:
                    stop_sent = state.stop_sent
                if stop_sent:
                    self._logger.warning(
                        "%s: busy state unreadable after a stop (%r); no longer "
                        "tracked as moving",
                        motion.name,
                        exc,
                    )
                    self._drop(state)
                    continue
                moving.append(motion.name)
                details.append(
                    f"the busy state of `{motion.name}` could not be read ({exc!r}), "
                    f"so it counts as moving"
                )
                continue
            if busy:
                moving.append(motion.name)
            else:
                self._drop(state)
        if moving:
            raise MotionInProgressError(tuple(moving), "; ".join(details))

    def _register(self, motion: Motion) -> _MotionState:
        with self._registry_lock:
            # Checked again here: Microscope.stop() halts, then stops every
            # stage. A halt that lands after the check at the top of do() but
            # before this line would otherwise let the command start a stage
            # that the stop has already missed.
            if self._halted:
                raise MicroscopeHaltedError()
            state = _MotionState(motion)
            self._registry[motion.device] = state
            return state

    def _drop(self, state: _MotionState) -> None:
        with self._registry_lock:
            if self._registry.get(state.motion.device) is state:
                del self._registry[state.motion.device]

    def _send_stop(self, motion: Motion, state: _MotionState | None) -> None:
        """Mark ``state`` stopped, call the device's stop, then record that it was sent."""
        if motion.stop is None:
            raise HardwareError(f"{motion.name} cannot be stopped from smc")
        if state is not None:
            with self._registry_lock:
                state.stopped = True
        motion.stop()
        if state is not None:
            with self._registry_lock:
                state.stop_sent = True

    def _stop_after(self, motion: Motion, state: _MotionState | None, why: str) -> str:
        """Stop on a failure path without masking the failure; return what happened."""
        if motion.stop is None:
            self._logger.warning(
                "%s: %s; it cannot be stopped from smc", motion.name, why
            )
            return "it cannot be stopped from smc, so it may still be moving"
        self._logger.warning("%s: stop after %s", motion.name, why)
        try:
            self._send_stop(motion, state)
        except Exception as exc:
            self._logger.exception("%s: the stop failed", motion.name)
            return f"the stop failed ({exc!r}), so it may still be moving"
        return "the stop was sent"
