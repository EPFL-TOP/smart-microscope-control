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

``Executor`` runs every action (a mutation, an acquisition, a wait) under
the microscope's one re-entrant lock and implements dry-run. In dry-run,
mutations are logged and skipped, but *reads still reach the hardware*: a
dry-run session shows the real stage position and never moves it.

* **No action while something moves** (design §13). An action holds the
  lock from its first check to its readback, including the wait of a move,
  so only the thread inside it sends commands and waits. Reads, stops and
  closing a shutter never take the lock: MMCore, not smc, keeps concurrent
  calls to one device safe (measured, §13). A ``wait=False`` motion stays
  registered, and refuses every action, until the device reports idle.

This module is pure: it imports no Micro-Manager code.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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

__all__ = ["POLL_INTERVAL_S", "Executor", "Motion", "Safety", "Step"]

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
#: How often a caller waiting for the lock looks at what the holder is doing,
#: so that it is refused as soon as the holder starts waiting for a motion.
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


@dataclass
class _Registration:
    """A motion the layer started and has not yet seen idle."""

    motion: Motion
    #: The thread that started it: the only one that stops it on giving up.
    #: ``None`` for a device nobody moved through the layer.
    owner: int | None
    #: The device's stop generation when the action was called. If the device
    #: counter differs once the motion is idle, it was stopped.
    generation: int


@dataclass
class _Holder:
    """The action holding the microscope lock. Read and written under the registry lock."""

    description: str
    since: float
    thread: int
    waiting_for: str | None = None


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
    Reads, stops and closing a shutter never take it.

    The registry lock guards the small shared state (registered motions,
    stop generations, the halt, who holds the lock). It is never held across
    a device call or a log call, and a thread holding it never takes the
    microscope lock.
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
        #: Registrations the guard or a bystander's wait saw idle, kept until
        #: their owner's ``wait()`` so that it still learns of a stop. Keyed by
        #: (device, owner thread): another caller's new move on the same
        #: device must not erase how this owner's move ended.
        self._finished: dict[tuple[str, int], _Registration] = {}
        self._generations: dict[str, int] = {}
        self._halted = False
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
        self,
        description: str,
        body: Callable[[], T],
        *,
        deadline: float,
        on_contention: Callable[[_Holder | None, bool], None],
    ) -> T:
        """Run ``body`` holding the microscope lock.

        ``acquire()`` is called here, in the frame whose ``finally`` releases
        the lock, and nothing else runs between it and the ``try``: a
        ``KeyboardInterrupt`` delivered as ``acquire()`` returns would
        otherwise leave the lock held for good (#59). The interrupt can still
        land between ``acquire()`` returning and ``acquired`` being set, so
        the ``finally`` also asks the lock whether this thread took it.

        ``on_contention(holder, expired)`` runs while the lock is held by
        another thread; it raises to give up.
        """
        me = threading.get_ident()
        with self._registry_lock:
            holder = self._holder
            reentrant = holder is not None and holder.thread == me
        if reentrant:
            # A callback inside an outer action (a move inside a snap) is part
            # of that action: it keeps the outer holder and its lock.
            return body()
        owned_before = _lock_is_owned(self._lock)
        acquired = False
        try:
            while not acquired:
                remaining = deadline - time.monotonic()
                acquired = self._lock.acquire(
                    timeout=min(_LOCK_POLL_S, max(remaining, 0.0))
                )
                if not acquired:
                    with self._registry_lock:
                        holder = self._holder
                    on_contention(holder, time.monotonic() >= deadline)
            with self._registry_lock:
                self._holder = _Holder(description, time.monotonic(), me)
            return body()
        finally:
            if acquired or (not owned_before and _lock_is_owned(self._lock)):
                with self._registry_lock:
                    self._holder = None
                self._lock.release()

    def _action_contention(self, holder: _Holder | None, expired: bool) -> None:
        """An action facing a held lock: refused at once if the holder waits for a motion."""
        if holder is not None and holder.waiting_for is not None:
            raise MotionInProgressError(
                (holder.waiting_for,), f"`{holder.description}` is waiting for it"
            )
        if expired:
            if holder is None:
                # Released just now, or held by a caller outside the Executor.
                raise MicroscopeBusyError("an unknown caller", self._lock_timeout_s)
            raise MicroscopeBusyError(
                holder.description, time.monotonic() - holder.since
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
        wait: bool = True,
        timeout_s: float | Callable[[], float] | None = None,
    ) -> T:
        """Perform a mutation as one lock section; in dry-run, log it and skip it.

        Under the lock and in this order (design §13): the halt check, the
        guard, ``step`` (a callable is built now, so a relative move reads its
        anchor here), the timeout, the stop-generation check, the halt again,
        the command, for a ``motion`` with ``wait`` the wait, and the readback.

        Args:
            description: Names the action for ``MicroscopeBusyError`` and
                ``MotionInProgressError`` while it holds the lock.
            step: What to send and read back.
            motion: The device the command sets moving, if any.
            wait: With ``motion``: wait for it inside the action (the default),
                or register it and return after the readback.
            timeout_s: The wait's deadline, or a callable that resolves it (the
                core timeout); required with ``motion`` and ``wait``.

        Raises:
            MicroscopeHaltedError: The microscope is halted.
            MotionInProgressError: A device is still moving, or the lock holder
                is waiting for one.
            MotionStoppedError: A stop for ``motion``'s device arrived after the
                call: before the command (not sent) or during it (the stop is
                sent again) or during the wait.
            DeviceTimeoutError: Still busy at the deadline; the stop was sent,
                or the message says why not.
            MicroscopeBusyError: The lock was not free within ``lock_timeout_s``.
        """
        generation = self._generation(motion) if motion is not None else 0

        def body() -> T:
            self._refuse_if_halted()
            self._guard()
            prepared = step() if callable(step) else step
            if self._dry_run:
                self._logger.info("[dry-run] %s", prepared.log)
                return prepared.dry_result
            if motion is None:
                self._logger.info("%s", prepared.log)
                self._refuse_if_halted()
                prepared.send()
                return prepared.readback()
            return self._move(prepared, motion, generation, wait, timeout_s)

        return self._section(
            description,
            body,
            deadline=time.monotonic() + self._lock_timeout_s,
            on_contention=self._action_contention,
        )

    def _move(
        self,
        step: Step[T],
        motion: Motion,
        generation: int,
        wait: bool,
        timeout_s: float | Callable[[], float] | None,
    ) -> T:
        """The command, wait and readback of a motion; called under the lock."""
        limit_s = 0.0
        if wait:
            if timeout_s is None:
                raise ValueError("a motion with wait=True needs timeout_s")
            limit_s = _resolve_timeout(timeout_s)
        self._logger.info("%s", step.log)
        me = threading.get_ident()
        with self._registry_lock:
            if self._generations.get(motion.device, 0) != generation:
                raise MotionStoppedError(
                    motion.name, "it was stopped before it started"
                )
            if self._halted:
                # Microscope.stop() halts, then stops every stage: a halt that
                # lands after the first check must still keep the command away.
                raise MicroscopeHaltedError()
            registration = _Registration(motion, me, generation)
            self._registry[motion.device] = registration
            # This owner's earlier move on the device is superseded.
            self._finished.pop((motion.device, me), None)
        try:
            step.send()
        except BaseException as exc:
            # The command may have reached the device before it raised; the
            # motion stays registered until the guard sees it idle.
            self._give_up(motion, f"{type(exc).__name__} in the command")
            raise
        if self._generation(motion) != generation:
            # The stop did not wait for the lock and may have reached the
            # device before this command did (FM-18).
            self._resend_stop(motion)
            if not wait:
                raise MotionStoppedError(motion.name)
        if wait:
            self._wait_idle(
                registration, time.monotonic() + limit_s, limit_s, owner=True
            )
        return step.readback()

    def acquire(self, description: str, action: Callable[[], T]) -> T:
        """Perform an acquisition (``snap``) as one lock section; it runs in dry-run too.

        A frame taken during a move is a wrong result, not a skipped mutation:
        it is refused while halted or while anything moves, in dry-run too.

        Raises:
            MicroscopeHaltedError: The microscope is halted.
            MotionInProgressError: A device is still moving.
            MicroscopeBusyError: The lock was not free within ``lock_timeout_s``.
        """

        def body() -> T:
            self._refuse_if_halted()
            self._guard()
            self._refuse_if_halted()
            return action()

        return self._section(
            description,
            body,
            deadline=time.monotonic() + self._lock_timeout_s,
            on_contention=self._action_contention,
        )

    def wait(self, motion: Motion, timeout_s: float | Callable[[], float]) -> None:
        """Block until ``motion`` is idle, holding the lock while it polls.

        It waits for the lock within its own ``timeout_s`` too. Only the thread
        that started the motion stops it when it gives up; any other waiter
        raises and leaves the move alone, so a bystander's short wait cannot
        stop a traverse (design §13).

        Raises:
            ValueError: ``timeout_s`` is negative or not finite.
            MotionStoppedError: The motion went idle because it was stopped.
            DeviceTimeoutError: Still busy at the deadline, or the lock was not
                free by then. The message says whether the stop was sent.
        """
        limit_s = _resolve_timeout(timeout_s)
        if self._dry_run:
            return
        me = threading.get_ident()
        deadline = time.monotonic() + limit_s

        def contention(holder: _Holder | None, expired: bool) -> None:
            if not expired:
                return
            what = "an unknown caller" if holder is None else f"`{holder.description}`"
            with self._registry_lock:
                registration = self._registry.get(motion.device)
            if registration is not None and registration.owner == me:
                outcome = self._give_up(motion, f"{limit_s:g} s wait for the lock")
            else:
                outcome = _not_stopped(registration)
            raise DeviceTimeoutError(
                f"{motion.name}: {what} held the microscope for the whole "
                f"{limit_s:g} s wait; {outcome}."
            )

        def body() -> None:
            with self._registry_lock:
                current = self._generations.get(motion.device, 0)
                # This caller's own move, seen idle already by the guard or by
                # a bystander's wait: its owner now learns how it ended, even
                # if another caller has moved the device since.
                finished = self._finished.pop((motion.device, me), None)
                registration = self._registry.get(motion.device)
            if finished is not None:
                if current != finished.generation:
                    raise MotionStoppedError(motion.name)
                return
            if registration is None:
                # Nobody moved it through the layer: poll it, never stop it.
                registration = _Registration(motion, None, current)
            self._wait_idle(
                registration, deadline, limit_s, owner=registration.owner == me
            )

        self._section(
            f"{motion.name}: wait", body, deadline=deadline, on_contention=contention
        )

    def _wait_idle(
        self,
        registration: _Registration,
        deadline: float,
        limit_s: float,
        *,
        owner: bool,
    ) -> None:
        """Poll until idle under the lock; the owner stops the motion on giving up."""
        motion = registration.motion
        previous = self._set_waiting(motion.name)
        idle = False
        try:
            while True:
                if not motion.is_busy():
                    idle = True
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(POLL_INTERVAL_S)
        except BaseException as exc:
            # Ctrl-C, an unreadable busy state: nobody is left watching the move.
            if owner:
                self._give_up(motion, f"{type(exc).__name__} during the wait")
            raise
        finally:
            self._set_waiting(previous)
        if not idle:
            if owner:
                outcome = self._give_up(motion, f"{limit_s:g} s timeout")
            else:
                outcome = _not_stopped(registration)
            raise DeviceTimeoutError(
                f"{motion.name} is still busy after {limit_s:g} s; {outcome}. "
                f"If the move is legitimately long, raise [micromanager] "
                f"device_timeout_ms in the profile; otherwise check the device "
                f"and its cabling."
            )
        with self._registry_lock:
            if self._registry.get(motion.device) is registration:
                del self._registry[motion.device]
                if not owner and registration.owner is not None:
                    self._finished[(motion.device, registration.owner)] = registration
            stopped = self._generations.get(motion.device, 0) != registration.generation
        if stopped:
            raise MotionStoppedError(motion.name)

    # --- the safe calls ----------------------------------------------------

    def stop(self, motion: Motion) -> None:
        """Stop ``motion`` now, without the microscope lock; never refused.

        The device's stop generation moves first, so every action on it that
        was called before raises ``MotionStoppedError``, even when the stop
        itself fails. The stop is sent before the log line. It runs in dry-run
        too: a stop cannot create motion. A registered motion whose busy state
        cannot be read is dropped afterwards, so a broken device never locks
        the stand for good.

        Raises:
            HardwareError: The device cannot be stopped from smc.
            Exception: Whatever the device's stop raised; a failed stop is a
                finding.
        """
        with self._registry_lock:
            self._generations[motion.device] = (
                self._generations.get(motion.device, 0) + 1
            )
        error: Exception | None = None
        if motion.stop is not None:
            try:
                motion.stop()
            except Exception as exc:
                error = exc
        if motion.stop is None:
            self._logger.warning(
                "%s: stop requested; it cannot be stopped", motion.name
            )
        elif error is not None:
            self._logger.warning("%s: stop failed (%r)", motion.name, error)
        else:
            self._logger.warning("%s: stop", motion.name)
        with self._registry_lock:
            registration = self._registry.get(motion.device)
        if registration is not None:
            self._drop_if_unreadable(registration)
        if motion.stop is None:
            raise HardwareError(f"{motion.name} cannot be stopped from smc")
        if error is not None:
            raise error

    def safe(self, description: str, action: Callable[[], T]) -> T:
        """A call that must always get through (closing a shutter), design §13.

        It never takes the microscope lock and is never refused, halted or
        not, so the light can always be cut. It runs in dry-run too. The call
        comes before the log line, as for a stop.
        """
        result = action()
        self._logger.info("%s", description)
        return result

    def read(self, action: Callable[[], T]) -> T:
        """A read: never takes the lock, never refused, reaches the hardware in dry-run."""
        return action()

    def halt(self) -> None:
        """Refuse every action until ``resume()``; reads and the safe calls still run."""
        with self._registry_lock:
            self._halted = True
        self._logger.warning("microscope: halted")

    def resume(self) -> None:
        """Clear the halt and drop unreadable motions; starts nothing."""
        with self._registry_lock:
            self._halted = False
            registrations = list(self._registry.values())
        self._logger.warning("microscope: resumed")
        for registration in registrations:
            self._drop_if_unreadable(registration)

    def moving(self) -> tuple[str, ...]:
        """Names of the motions not yet seen idle; asks no device, never blocks."""
        with self._registry_lock:
            return tuple(r.motion.name for r in self._registry.values())

    # --- helpers -----------------------------------------------------------

    def _generation(self, motion: Motion) -> int:
        with self._registry_lock:
            return self._generations.get(motion.device, 0)

    def _refuse_if_halted(self) -> None:
        if self._halted:
            raise MicroscopeHaltedError()

    def _guard(self) -> None:
        """Poll every registered motion, drop the idle ones, refuse the rest.

        Called under the microscope lock. An unreadable busy state counts as
        moving (unknown beats guessed); ``stop()`` or ``resume()`` releases it.
        """
        with self._registry_lock:
            registrations = list(self._registry.values())
        moving: list[str] = []
        details: list[str] = []
        for registration in registrations:
            motion = registration.motion
            try:
                busy = bool(motion.is_busy())
            except Exception as exc:
                moving.append(motion.name)
                details.append(
                    f"the busy state of `{motion.name}` could not be read ({exc!r}), "
                    f"so it counts as moving; `stop()` or `resume()` releases it"
                )
                continue
            if busy:
                moving.append(motion.name)
                continue
            with self._registry_lock:
                if self._registry.get(motion.device) is registration:
                    del self._registry[motion.device]
                    if registration.owner is not None:
                        # Kept so that its owner's wait() still learns of a stop.
                        self._finished[(motion.device, registration.owner)] = (
                            registration
                        )
        if moving:
            raise MotionInProgressError(tuple(moving), "; ".join(details))

    def _drop_if_unreadable(self, registration: _Registration) -> None:
        """Drop a registered motion whose busy state raises (FM-19)."""
        motion = registration.motion
        try:
            motion.is_busy()
        except Exception as exc:
            with self._registry_lock:
                dropped = self._registry.get(motion.device) is registration
                if dropped:
                    del self._registry[motion.device]
            if dropped:
                self._logger.warning(
                    "%s: busy state unreadable (%r); no longer tracked as moving, "
                    "check the device before moving it again",
                    motion.name,
                    exc,
                )

    def _give_up(self, motion: Motion, why: str) -> str:
        """Stop on a failure path, then log; never masks the failure. Returns what happened."""
        if motion.stop is None:
            self._logger.warning(
                "%s: %s; it cannot be stopped from smc", motion.name, why
            )
            return "it cannot be stopped from smc, so it may still be moving"
        with self._registry_lock:
            self._generations[motion.device] = (
                self._generations.get(motion.device, 0) + 1
            )
        try:
            motion.stop()
        except Exception as exc:
            self._logger.warning("%s: stop after %s failed (%r)", motion.name, why, exc)
            return f"the stop failed ({exc!r}), so it may still be moving"
        self._logger.warning("%s: stop after %s", motion.name, why)
        return "the stop was sent"

    def _resend_stop(self, motion: Motion) -> None:
        """Send a stop again after a command it may have overtaken (FM-18); stop, then log."""
        if motion.stop is None:
            return
        try:
            motion.stop()
        except Exception as exc:
            # The caller still gets MotionStoppedError: a stop was asked for,
            # so the move did not end as planned.
            self._logger.warning(
                "%s: stop resent after the command failed (%r)", motion.name, exc
            )
            return
        self._logger.warning("%s: stop resent after the command", motion.name)


def _not_stopped(registration: _Registration | None) -> str:
    """Why a waiter that gives up leaves the motion alone (design §13)."""
    if registration is None or registration.owner is None:
        return "nothing moved it through smc, so nothing was stopped"
    return (
        "the move belongs to another caller, so nothing was stopped; that "
        "caller stops it if it gives up"
    )


def _resolve_timeout(timeout_s: float | Callable[[], float]) -> float:
    value = float(timeout_s() if callable(timeout_s) else timeout_s)
    if not math.isfinite(value) or value < 0:
        # nan would make the deadline comparison always false: an endless wait.
        raise ValueError(f"timeout_s must be finite and >= 0, got {value!r}")
    return value
