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

``Executor`` runs every action (a mutation, an acquisition) under the
microscope's one re-entrant lock and implements dry-run. In dry-run,
mutations are logged and skipped, but *reads still reach the hardware*: a
dry-run session shows the real stage position and never moves it.

* **No action while something moves** (design §13). An action holds the
  lock from its first check to its readback, including the wait of a move,
  so only the thread inside it sends commands and waits for them. Reads
  (``wait()`` included), stops and closing a shutter never take the lock:
  MMCore, not smc, keeps concurrent calls to one device safe (measured,
  §13). There is no ``wait=False``: a motion outlives its action only when
  the action gave up on it, and then it refuses every action until the
  device reports idle, a stop gets through, or the operator resumes.

This module is pure: it imports no Micro-Manager code.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeBusyError,
    MicroscopeHaltedError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)

__all__ = ["POLL_INTERVAL_S", "Executor", "Motion", "SafeCall", "Safety", "Step"]

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
#: How often a caller waiting for the lock looks at the halt and at what the
#: holder is doing, so that it is refused as soon as the stand is halted or
#: the holder starts waiting for a motion.
_LOCK_POLL_S = 0.05


@dataclass(frozen=True, slots=True)
class Motion:
    """A device the layer sets moving, as the ``Executor`` sees it (design §13).

    The backend builds it from its device calls; the ``Executor`` needs only
    these callables, which keeps this module free of Micro-Manager code.
    """

    device: str  # registry and stop-generation key: the device label
    name: str  # for messages: "xy_stage XY", "z Z", "Dichroic"
    is_busy: Callable[[], bool]
    stop: Callable[[], None] | None  # None: cannot be stopped (a State device)


@dataclass(frozen=True)
class Step(Generic[T]):
    """What an action sends and reads back, decided inside its lock section.

    An action builds its ``Step`` under the lock, after the halt check and the
    guard, so a relative move computes its target from a position no other
    caller can change before the command (design §13).
    """

    log: str  # the INFO line: "xy_stage: move_to (1.0, 2.0) µm"
    send: Callable[[], object]
    readback: Callable[[], T]
    dry_result: T  # returned in dry-run instead of sending


@dataclass(frozen=True, slots=True)
class SafeCall:
    """A call that always gets through and wins over an action it races (design §13).

    Closing a shutter: it never waits for the microscope lock and is never
    refused, so the light can always be cut. An action on the same device
    that was called before it (opening that shutter) is cancelled if it has
    not been sent yet, and followed by this call again if it was being sent,
    so the shutter never ends open after a close that reported success.
    """

    device: str  # the key it shares with the actions it wins over: the label
    log: str  # the INFO line: "shutter: close"
    send: Callable[[], object]


@dataclass(eq=False)
class _Registration:
    """A motion that outlived its action: its action gave up while it still moved.

    One object per registration, compared by identity: a caller that saw the
    device idle drops only the registration it read before it looked, never
    one made since for a newer give-up.
    """

    motion: Motion


@dataclass
class _Holder:
    """The action holding the microscope lock. Read and written under the registry lock."""

    description: str
    since: float
    #: The motion the action is waiting for, from just before its command
    #: until the device reads idle: a command can block for the whole
    #: traverse, and callbacks run inside it (#68).
    waiting_for: str | None = None
    #: The thread that holds the lock: a record left by another thread (its
    #: clean-up has not run yet) is not this thread's.
    thread: int = field(default_factory=threading.get_ident)
    #: The lock was taken outside the ``Executor`` before this action, so it
    #: has been held for longer than ``since`` says.
    external: bool = False


def _lock_is_owned(lock: threading.RLock) -> bool:
    """Whether the calling thread holds ``lock``.

    ``_is_owned`` is what ``threading.Condition`` uses; both the C and the
    Python ``RLock`` have it.
    """
    is_owned = getattr(lock, "_is_owned", None)
    return bool(is_owned()) if is_owned is not None else False


class Executor:
    """Runs every action under the microscope's lock, honouring dry-run (design §13).

    One ``RLock`` per microscope keeps actions sequential across a UI thread
    and a worker. An action holds it from its first check to its readback,
    wait included; a caller that finds it held by an action waiting for a
    motion is refused at once, any other waits up to ``lock_timeout_s``.
    Reads, ``wait()``, stops and closing a shutter never take it.

    The registry lock guards the small shared state (motions that outlived
    their action, stop generations, the halt, who holds the lock). It is
    never held across a device call or a log call, and a thread holding it
    never takes the microscope lock.
    """

    def __init__(
        self,
        *,
        dry_run: bool,
        lock: threading.RLock,
        logger: logging.Logger,
        lock_timeout_s: float = 60.0,
    ) -> None:
        if (
            not math.isfinite(lock_timeout_s)
            or lock_timeout_s <= 0
            or lock_timeout_s > threading.TIMEOUT_MAX
        ):
            # nan would make acquire() raise, inf would wait for ever (FM-32),
            # and above TIMEOUT_MAX every acquire() raises OverflowError.
            raise ValueError(
                f"lock_timeout_s must be finite, > 0 and <= threading.TIMEOUT_MAX, "
                f"got {lock_timeout_s!r}"
            )
        self._dry_run = dry_run
        self._lock = lock
        self._logger = logger
        self._lock_timeout_s = float(lock_timeout_s)
        self._registry_lock = threading.Lock()
        self._registry: dict[str, _Registration] = {}
        #: Per device: how many explicit stops (or, for a shutter, closes) it
        #: has had. Only ``stop()`` and ``safe()`` advance it, so a give-up
        #: never cancels another caller's action.
        self._generations: dict[str, int] = {}
        self._halted = False
        #: How many halts there have been. An action records it when it is
        #: called, as it records the stop generation, so a halt that lands
        #: while it waits for the lock refuses it even after ``resume()``.
        self._halt_epoch = 0
        self._holder: _Holder | None = None

    @property
    def dry_run(self) -> bool:
        """Whether mutations are skipped; fixed at construction."""
        return self._dry_run

    @property
    def halted(self) -> bool:
        """Whether every action is refused until ``resume()``."""
        return self._halted

    # --- the lock section --------------------------------------------------

    def _section(
        self, description: str, body: Callable[[], T], halt_mark: int | None
    ) -> T:
        """Run ``body`` holding the microscope lock (design §13, "The lock").

        A thread that already holds the lock runs ``body`` inside it
        (``_nested``). Ownership is asked of the lock itself, not of the holder
        record, which a second Ctrl-C can leave stale.

        Otherwise ``acquire()`` is called here, in the frame whose ``finally``
        releases the lock, and nothing else runs between it and the ``try``:
        a ``KeyboardInterrupt`` delivered as ``acquire()`` returns would
        otherwise leave the lock held for good (#59). It can still land before
        ``acquired`` is set, so the ``finally`` also asks the lock whether this
        thread took it. The lock is released before the holder record is
        cleared, so an interrupt in that clean-up cannot keep it.

        ``halt_mark`` is what the action recorded of the halt when it was
        called. ``body`` checks it once the lock is taken; while the lock is
        held by another thread, ``_contention`` checks it at every poll, and
        inside this thread's own lock ``_nested`` checks it first.
        """
        if _lock_is_owned(self._lock):
            return self._nested(description, body, halt_mark)
        started = time.monotonic()
        deadline = started + self._lock_timeout_s
        acquired = False
        record: _Holder | None = None
        try:
            while not acquired:
                remaining = deadline - time.monotonic()
                acquired = self._lock.acquire(
                    timeout=min(_LOCK_POLL_S, max(remaining, 0.0))
                )
                if not acquired:
                    # Expired only once an attempt made at the deadline failed.
                    self._contention(started, halt_mark, expired=remaining <= 0)
            record = _Holder(description, time.monotonic())
            with self._registry_lock:
                self._holder = record
            return body()
        finally:
            if acquired or _lock_is_owned(self._lock):
                self._lock.release()
                self._clear_holder(record)

    def _nested(
        self, description: str, body: Callable[[], T], halt_mark: int | None
    ) -> T:
        """Run ``body`` inside the lock this thread already holds (design §13, "Re-entry").

        A callback inside ``snap`` that moves the stage is part of the snap.
        It is refused while this thread's action waits for a motion, as
        another thread would be: pymmcore-plus emits ``propertyChanged``
        synchronously from ``setProperty``, so a UI handler that snaps runs
        inside the command of a turret change (#68). A halt since the call
        comes before that refusal, as in ``_contention`` (FM-64): the
        refusal is a ``SafetyRefusedError``, and an emergency stop pressed
        during the command must not be caught as one.

        With no record of this thread's, the lock was taken outside the
        ``Executor`` (a facade helper holding it for a sequence), or the
        previous holder has not cleared its record yet. The section then
        records itself, so that ``moving()`` names its motion and contenders
        are refused at once, and a contender that times out reads its time
        as a lower bound (#68).
        """
        record: _Holder | None = None
        try:
            with self._registry_lock:
                halted = self._halted_since(halt_mark)
                holder = self._holder
                if holder is None or holder.thread != threading.get_ident():
                    record = self._holder = _Holder(
                        description, time.monotonic(), external=True
                    )
                    waiting_for, outer = None, description
                else:
                    waiting_for, outer = holder.waiting_for, holder.description
            if halted:
                raise MicroscopeHaltedError()
            if waiting_for is not None:
                raise MotionInProgressError(
                    (waiting_for,),
                    f"`{outer}` is waiting for it, and this call runs inside it",
                )
            return body()
        finally:
            self._clear_holder(record)

    def _clear_holder(self, record: _Holder | None) -> None:
        """Forget ``record`` if it is still the holder: the lock may have changed hands."""
        with self._registry_lock:
            if record is not None and self._holder is record:
                self._holder = None

    def _contention(
        self, started: float, halt_mark: int | None, *, expired: bool
    ) -> None:
        """An action facing a lock held by another thread: halted, refused, waiting, or out of time.

        A halt since the call comes first (FM-64). Behind a holder waiting for
        a motion, the refusal would be a ``SafetyRefusedError``, which a tool
        may catch and carry on after, through the emergency stop. Behind a
        slow holder, the caller would wait ``lock_timeout_s`` for a stand the
        operator has stopped, then report a hung driver. The halt mark, not
        the flag, is compared, so a ``resume()`` since does not let it wait on.
        """
        with self._registry_lock:
            halted = self._halted_since(halt_mark)
            holder = None if self._holder is None else replace(self._holder)
        if halted:
            raise MicroscopeHaltedError()
        if holder is not None and holder.waiting_for is not None:
            raise MotionInProgressError(
                (holder.waiting_for,), f"`{holder.description}` is waiting for it"
            )
        if not expired:
            return
        now = time.monotonic()
        if holder is None:
            # Taken outside the Executor: nothing recorded when, so this
            # caller's own wait is all that is known, and it is a lower bound.
            raise MicroscopeBusyError("an unknown caller", now - started, at_least=True)
        raise MicroscopeBusyError(
            holder.description, now - holder.since, at_least=holder.external
        )

    def _set_waiting(self, name: str | None) -> str | None:
        """Record what the holder waits for; return what it waited for before."""
        with self._registry_lock:
            holder = self._holder
            if holder is None:
                return None
            previous, holder.waiting_for = holder.waiting_for, name
            return previous

    # --- actions -----------------------------------------------------------

    def do(
        self,
        description: str,
        step: Step[T] | Callable[[], Step[T]],
        *,
        motion: Motion | None = None,
        timeout_s: float | Callable[[], float] | None = None,
        overridden_by: SafeCall | None = None,
    ) -> T:
        """Perform a mutation as one lock section; in dry-run, log it and skip it.

        Under the lock and in this order (design §13): the halt check, the
        guard, ``step`` (a callable is built now, so a relative move reads its
        anchor here), the timeout, the generation and halt checks, the INFO
        line, the same two checks again (the line may have blocked on a
        console), the command, for a ``motion`` the wait, and the readback.

        Args:
            description: Names the action for ``MicroscopeBusyError`` and
                ``MotionInProgressError`` while it holds the lock.
            step: What to send and read back.
            motion: The device the command sets moving, if any; the action
                waits for it before reading back.
            timeout_s: The wait's deadline, or a callable that resolves it (the
                core timeout); required with ``motion``.
            overridden_by: The safe call that wins over this action (the close
                of the shutter it opens); not with ``motion``, whose stop is
                the one that wins.

        Raises:
            MicroscopeHaltedError: The microscope is halted, or was halted
                since this action was called, even if ``resume()`` came
                since: it starts nothing.
            MotionInProgressError: A device is still moving, or the lock holder
                is waiting for one (this thread's own action, for a callback
                that runs inside its command).
            MotionStoppedError: A stop for ``motion``'s device was called after
                this action: before the command (not sent), or during the
                command or the wait (the stop is sent again after the command),
                including a move still busy at its deadline after that stop.
            HardwareError: ``overridden_by`` was called after this action: it
                was not sent, or it was and the safe call was sent again.
            DeviceTimeoutError: Still busy at the deadline, and not stopped by
                a caller; the stop was sent, or the message says why not.
            MicroscopeBusyError: The lock was not free within ``lock_timeout_s``.
        """
        if motion is not None and timeout_s is None:
            raise ValueError("an action with a motion needs timeout_s")
        if motion is not None and overridden_by is not None:
            raise ValueError("a motion is overridden by its own stop, not a SafeCall")
        limit: float | Callable[[], float] = 0.0 if timeout_s is None else timeout_s
        cancelled: HardwareError
        if motion is not None:
            key: str | None = motion.device
            cancelled = MotionStoppedError(
                motion.name, "it was stopped before it started"
            )
        elif overridden_by is not None:
            key = overridden_by.device
            cancelled = HardwareError(
                f"`{description}` was not sent: `{overridden_by.log}` was "
                f"requested after it was called; call it again if it is still "
                f"wanted"
            )
        else:
            key, cancelled = None, HardwareError(description)  # never raised
        # Recorded before waiting for the lock: a stop (or a close) or a halt
        # pressed while this call waits for it cancels the call.
        generation = self._generation(key)
        halt_mark = self._halt_mark()

        def body() -> T:
            self._refuse_if_halted(halt_mark)
            self._guard()
            prepared = step() if callable(step) else step
            limit_s = 0.0
            if motion is not None and not self._dry_run:
                limit_s = _resolve_timeout(limit)
            self._refuse_if_cancelled(key, generation, halt_mark, cancelled)
            if self._dry_run:
                self._logger.info("[dry-run] %s", prepared.log)
                return prepared.dry_result
            self._logger.info("%s", prepared.log)

            def last_check() -> None:
                # FM-66: the halt or a stop may have landed while the line
                # above was written; the check that matters is the last one.
                # Its caller writes the "not sent" line (#80).
                self._refuse_if_cancelled(key, generation, halt_mark, cancelled)

            if motion is not None:
                return self._move(prepared, motion, generation, limit_s, last_check)
            try:
                last_check()
            except HardwareError:
                self._logger.warning("%s: not sent", prepared.log)
                raise
            if overridden_by is not None:
                return self._send_overridable(
                    description, prepared, overridden_by, generation, halt_mark
                )
            prepared.send()
            return prepared.readback()

        return self._section(description, body, halt_mark)

    def _send_overridable(
        self,
        description: str,
        step: Step[T],
        call: SafeCall,
        generation: int,
        halt_mark: int | None,
    ) -> T:
        """The command and readback of an action ``call`` wins over, under the lock.

        ``call`` (the close) does not wait for the lock, so it may have reached
        the device before this command (the open) did. If it was called during
        the command, it is sent again after it, whether the command returned or
        raised, and the action raises: the shutter never ends open after a
        close that reported success (design §13). It raises
        ``MicroscopeHaltedError`` if the stand was halted since the action was
        called, as ``do()`` documents, even if ``resume()`` came since: the
        halt mark decides, not the flag (#80).
        """
        resent = False
        try:
            step.send()
            if self._generation(call.device) != generation:
                error = self._send_safe(
                    call.send, f"{call.log} resent after `{step.log}`"
                )
                resent = True
                self._refuse_if_halted(halt_mark)
                outcome = (
                    "it was sent again after it"
                    if error is None
                    else f"sending it again failed ({error!r}), so check the device"
                )
                raise HardwareError(
                    f"`{description}`: `{call.log}` was requested while it was "
                    f"being sent, and {outcome}; call it again if it is still "
                    f"wanted"
                )
            return step.readback()
        except BaseException:
            if not resent and self._generation(call.device) != generation:
                self._send_safe(call.send, f"{call.log} resent after `{step.log}`")
            raise

    def _move(
        self,
        step: Step[T],
        motion: Motion,
        generation: int,
        limit_s: float,
        last_check: Callable[[], None],
    ) -> T:
        """The command, the wait and the readback of a motion, under the lock.

        One ``try`` covers all three (design §13): nothing unprotected runs
        between the command returning and the wait, so a Ctrl-C, a command
        that raises or an unreadable busy state anywhere before the device is
        seen idle gives up on the motion, which sends the stop. Once the device
        is idle there is nothing left to stop, and before the command there is
        nothing to stop either.

        The holder waits for the motion from just before the command, which
        can block for the traverse or run a callback, until the device reads
        idle. From then on the readback is an ordinary lock holder: a caller
        that saw the stage idle and acts next waits for the lock instead of
        being refused (#68). ``last_check`` runs after "waiting for" is
        recorded, so nothing lies between it and the command (FM-66). When it
        refuses, "waiting for" is cleared before the "not sent" line: that
        line can block on a console (FM-62), and meanwhile ``moving()`` would
        name, and contenders be refused for, a motion never commanded (#80).
        """
        deadline = time.monotonic() + limit_s
        sending = idle = gave_up = False
        previous: str | None = None
        try:
            previous = self._set_waiting(motion.name)
            try:
                last_check()
            except HardwareError:
                self._set_waiting(previous)
                self._logger.warning("%s: not sent", step.log)
                raise
            sending = True  # from here on the command may reach the device
            step.send()
            if (
                self._generation(motion.device) != generation
                and motion.stop is not None
            ):
                # The stop did not wait for the lock and may have reached the
                # device before this command did (FM-18).
                self._send_safe(
                    motion.stop, f"{motion.name}: stop resent after the command"
                )
            idle = self._poll_until_idle(motion, deadline)
            if not idle:
                outcome = self._give_up(motion, f"{limit_s:g} s timeout")
                gave_up = True
                if self._generation(motion.device) != generation:
                    # The caller asked for the stop: the move ended as stopped,
                    # and the timeout is no reason to raise device_timeout_ms.
                    raise MotionStoppedError(
                        motion.name,
                        f"it was stopped, but it still read busy at the move's "
                        f"{limit_s:g} s timeout, and on the timeout {outcome}",
                    )
                raise DeviceTimeoutError(
                    f"{motion.name} is still busy after {limit_s:g} s; {outcome}. "
                    f"If the move is legitimately long, raise [micromanager] "
                    f"device_timeout_ms in the profile; otherwise check the "
                    f"device and its cabling."
                )
            self._set_waiting(previous)
            if self._generation(motion.device) != generation:
                raise MotionStoppedError(motion.name)
            return step.readback()
        except BaseException as exc:
            if sending and not (idle or gave_up):
                self._give_up(motion, type(exc).__name__)
            raise
        finally:
            self._set_waiting(previous)

    def acquire(self, description: str, action: Callable[[], T]) -> T:
        """Perform an acquisition (``snap``) as one lock section; it runs in dry-run too.

        A frame taken during a move is a wrong result, not a skipped mutation:
        it is refused while halted or while anything moves, in dry-run too.

        Raises:
            MicroscopeHaltedError: The microscope is halted, or was halted
                since this call, even if ``resume()`` came since.
            MotionInProgressError: A device is still moving, or the lock holder
                is waiting for one (this thread's own action, for a callback
                that runs inside its command).
            MicroscopeBusyError: The lock was not free within ``lock_timeout_s``.
        """
        halt_mark = self._halt_mark()

        def body() -> T:
            self._refuse_if_halted(halt_mark)
            self._guard()
            self._refuse_if_halted(halt_mark)
            return action()

        return self._section(description, body, halt_mark)

    # --- reads ---------------------------------------------------------------

    def read(self, action: Callable[[], T]) -> T:
        """A read: never takes the lock, never refused, reaches the hardware in dry-run."""
        return action()

    def wait(self, motion: Motion, timeout_s: float | Callable[[], float]) -> None:
        """Block until ``motion``'s device reports idle; a read (design §13).

        It polls without the microscope lock, so it runs during another
        thread's action, while halted and in dry-run, and it never stops
        anything: a caller that stops waiting did not start the move and has
        no business ending it. Seeing the device idle drops its registration,
        if it outlived its action, so the next action runs.

        Raises:
            ValueError: ``timeout_s`` is negative or not finite.
            DeviceTimeoutError: Still busy at the deadline; nothing was stopped.
            Exception: Whatever reading the busy state raised.
        """
        limit_s = _resolve_timeout(timeout_s)
        deadline = time.monotonic() + limit_s
        while True:
            # Read before the poll, so that a registration made after the
            # device was seen idle (a newer give-up) is not the one dropped.
            registration = self._registered(motion.device)
            if not motion.is_busy():
                if registration is not None:
                    self._drop(motion.device, registration)
                return
            if time.monotonic() >= deadline:
                raise DeviceTimeoutError(
                    f"{motion.name} is still busy after {limit_s:g} s; `wait()` "
                    f"stops nothing, so it may still be moving. Stop it "
                    f"(`stop()`) if it should not be, or wait longer."
                )
            time.sleep(POLL_INTERVAL_S)

    def moving(self) -> tuple[str, ...]:
        """What moves: the motion an action is waiting for, then those that outlived theirs.

        It asks no device and never blocks.
        """
        with self._registry_lock:
            names = [r.motion.name for r in self._registry.values()]
            waiting = None if self._holder is None else self._holder.waiting_for
        if waiting is not None and waiting not in names:
            names.insert(0, waiting)
        return tuple(names)

    # --- the safe calls ----------------------------------------------------

    def stop(self, motion: Motion) -> None:
        """Stop ``motion`` now, without the microscope lock; never refused (§13).

        The device's stop generation moves first, so every action on it that
        was called before raises ``MotionStoppedError``, even when the stop
        itself fails. The stop is sent before the log line. It runs in dry-run
        too: a stop cannot create motion. A motion that outlived its action
        is dropped once the stop is sent, unless the device still reads busy
        (it is moving); after a failed stop it stays, for the guard or for
        ``resume()``.

        Raises:
            HardwareError: The device cannot be stopped from smc.
            Exception: Whatever the device's stop raised; a failed stop is a
                finding.
        """
        send = motion.stop
        if send is None:
            self._logger.warning(
                "%s: stop requested; it cannot be stopped from smc", motion.name
            )
            raise HardwareError(
                f"{motion.name} cannot be stopped from smc; wait for it, or "
                f"check the stand and call `resume()`"
            )
        with self._registry_lock:
            self._generations[motion.device] = (
                self._generations.get(motion.device, 0) + 1
            )
        error = self._send_safe(
            send,
            f"{motion.name}: stop",
            then=lambda failed: "" if failed else self._release_after_stop(motion),
        )
        if error is not None:
            raise error

    def safe(self, call: SafeCall, readback: Callable[[], T]) -> T:
        """Send ``call`` (closing a shutter) now, then read back; design §13.

        It never takes the microscope lock and is never refused, halted or
        not, so the light can always be cut. It runs in dry-run too. The
        device's generation moves first, so an action ``call`` wins over that
        was called before this (opening that shutter) is cancelled, or, if it
        is being sent, followed by ``call`` again. The call comes before the
        log line, as for a stop.

        Raises:
            Exception: Whatever the call raised; a failed close is a finding.
        """
        with self._registry_lock:
            self._generations[call.device] = self._generations.get(call.device, 0) + 1
        error = self._send_safe(call.send, call.log, level=logging.INFO)
        if error is not None:
            raise error
        return readback()

    def halt(self) -> None:
        """Refuse every action until ``resume()``; reads and the safe calls still run.

        An action already called and waiting for the lock is refused too,
        within one lock poll and even once resumed.
        """
        with self._registry_lock:
            self._halted = True
            self._halt_epoch += 1
        self._logger.warning("microscope: halted")

    def resume(self) -> None:
        """Clear the halt and drop every motion that outlived its action; starts nothing.

        It is the operator's way out after checking the stand (design §13): a
        ``State`` device that cannot be stopped, a stage that keeps answering
        busy, or one whose busy state cannot be read would otherwise refuse
        every action for good. An action called before it, while halted or
        before the halt, and still waiting for the lock, is refused: only a
        call made after ``resume()`` runs (#68).
        """
        with self._registry_lock:
            self._halted = False
            dropped = [r.motion.name for r in self._registry.values()]
            self._registry.clear()
        self._logger.warning("microscope: resumed")
        for name in dropped:
            self._logger.warning(
                "%s: no longer tracked as moving (resume); check it before "
                "moving it again",
                name,
            )

    # --- helpers -----------------------------------------------------------

    def _generation(self, device: str | None) -> int:
        if device is None:
            return 0
        with self._registry_lock:
            return self._generations.get(device, 0)

    def _halt_mark(self) -> int | None:
        """What an action records of the halt when it is called; ``None`` if halted then."""
        with self._registry_lock:
            return None if self._halted else self._halt_epoch

    def _halted_since(self, mark: int | None) -> bool:
        """Whether the stand is halted, or was halted after ``mark`` was taken. Registry lock held."""
        return self._halted or mark is None or self._halt_epoch != mark

    def _refuse_if_halted(self, mark: int | None) -> None:
        with self._registry_lock:
            halted = self._halted_since(mark)
        if halted:
            raise MicroscopeHaltedError()

    def _refuse_if_cancelled(
        self,
        device: str | None,
        generation: int,
        halt_mark: int | None,
        cancelled: HardwareError,
    ) -> None:
        """The checks before a command: a halt, then a stop or close, since the call."""
        with self._registry_lock:
            halted = self._halted_since(halt_mark)
            moved = (
                device is not None and self._generations.get(device, 0) != generation
            )
        if halted:
            raise MicroscopeHaltedError()
        if moved:
            raise cancelled

    def _poll_until_idle(self, motion: Motion, deadline: float) -> bool:
        """Poll ``motion`` until idle (``True``) or the deadline (``False``)."""
        while True:
            if not motion.is_busy():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(POLL_INTERVAL_S)

    def _guard(self) -> None:
        """Poll every motion that outlived its action, drop the idle ones, refuse the rest.

        Called under the microscope lock. An unreadable busy state counts as
        moving (unknown beats guessed); ``stop()`` or ``resume()`` releases it.
        """
        with self._registry_lock:
            registrations = list(self._registry.items())
        moving: list[str] = []
        details: list[str] = []
        for device, registration in registrations:
            motion = registration.motion
            try:
                busy = bool(motion.is_busy())
            except Exception as exc:
                moving.append(motion.name)
                details.append(
                    f"the busy state of `{motion.name}` could not be read ({exc!r}), "
                    f"so it counts as moving"
                )
                continue
            if busy:
                moving.append(motion.name)
            else:
                self._drop(device, registration)
        if moving:
            raise MotionInProgressError(tuple(moving), "; ".join(details))

    def _registered(self, device: str) -> _Registration | None:
        with self._registry_lock:
            return self._registry.get(device)

    def _drop(self, device: str, registration: _Registration) -> None:
        """Drop ``registration`` if it is still the device's; a newer one stays."""
        with self._registry_lock:
            if self._registry.get(device) is registration:
                del self._registry[device]

    def _keep_while_moving(self, motion: Motion, registration: _Registration) -> str:
        """Drop a give-up's ``registration`` if the device reads idle; return a log note.

        A busy state that cannot be read, or a check interrupted before it
        answers, keeps it: unknown counts as moving.
        """
        try:
            busy = bool(motion.is_busy())
        except Exception as exc:
            note = f"; its busy state cannot be read ({exc!r})"
        else:
            if not busy:
                self._drop(motion.device, registration)
                return ""
            note = "; it still reads busy"
        return f"{note}, so it counts as moving until it is idle"

    def _track(self, motion: Motion) -> _Registration:
        registration = _Registration(motion)
        with self._registry_lock:
            self._registry[motion.device] = registration
        return registration

    def _release_after_stop(self, motion: Motion) -> str:
        """After a sent stop: drop the device's registration unless it still reads busy."""
        registration = self._registered(motion.device)
        if registration is None:
            return ""
        try:
            busy = bool(motion.is_busy())
        except Exception as exc:
            # FM-19: an unreadable device must not lock the stand for good.
            self._drop(motion.device, registration)
            return (
                f"; its busy state cannot be read ({exc!r}), so it is no longer "
                f"tracked as moving: check the device before moving it again"
            )
        if busy:
            return "; it still reads busy, so it stays tracked as moving"
        self._drop(motion.device, registration)
        return ""

    def _give_up(self, motion: Motion, why: str) -> str:
        """Stop a motion its action stops watching, keep it registered while it moves, then log.

        The motion is registered before the stop is sent, since unknown counts
        as moving, and dropped only once the device reads idle: a second
        Ctrl-C inside the stop call then leaves the stand refusing actions,
        not free with a moving stage (#68). A give-up does not advance the
        stop generation: it cancels nobody else's action (design §13). It
        never raises on its own account, so a failed stop never masks the
        failure that caused the give-up. Returns what happened, for the error
        message.
        """
        registration = self._track(motion)
        send = motion.stop
        if send is None:
            note = self._keep_while_moving(motion, registration)
            self._logger.warning(
                "%s: gave up after %s; it cannot be stopped from smc%s",
                motion.name,
                why,
                note,
            )
            return "it cannot be stopped from smc, so it may still be moving"
        error = self._send_safe(
            send,
            f"{motion.name}: stop after {why}",
            then=lambda _: self._keep_while_moving(motion, registration),
        )
        if error is None:
            return "the stop was sent"
        return f"the stop failed ({error!r}), so it may still be moving"

    def _send_safe(
        self,
        send: Callable[[], object],
        what: str,
        *,
        then: Callable[[bool], str] | None = None,
        level: int = logging.WARNING,
    ) -> Exception | None:
        """Send a safe call, update the registry (``then``), and only then log (FM-62).

        Every stop and every close goes through here (``stop()``, a give-up,
        the resend after a command, ``safe()``), so the order lives in one
        place: a blocked console or a second Ctrl-C can delay or cut the log
        line, never the call or the registry update. ``then`` is told whether
        the call failed and returns a note for the log line; a failure is
        logged at WARNING whatever ``level``. Returns what the call raised, if
        anything; it never raises it. A call interrupted before it returned (a
        second Ctrl-C) is logged as such, since it may not have got through,
        and the interrupt goes on.
        """
        error: Exception | None = None
        try:
            send()
        except Exception as exc:
            error = exc
        except BaseException:
            self._logger.warning(
                "%s was interrupted before it returned, so it may not have "
                "reached the device",
                what,
            )
            raise
        note = ""
        try:
            if then is not None:
                note = then(error is not None)
        finally:
            # Written even when ``then`` was interrupted (a Ctrl-C in its busy
            # check): the call was sent, and the log must say so.
            if error is None:
                self._logger.log(level, "%s%s", what, note)
            else:
                self._logger.warning("%s failed (%r)%s", what, error, note)
        return error


def _resolve_timeout(timeout_s: float | Callable[[], float]) -> float:
    value = float(timeout_s() if callable(timeout_s) else timeout_s)
    if not math.isfinite(value) or value < 0:
        # nan would make the deadline comparison always false: an endless wait.
        raise ValueError(f"timeout_s must be finite and >= 0, got {value!r}")
    return value
