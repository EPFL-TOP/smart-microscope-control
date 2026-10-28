"""Micro-Manager implementations of the capabilities (design §6).

Each class wraps one device label of a ``CMMCorePlus`` and nothing else: it
does not choose the device (roles do, #6) and does not own the core (the
facade does, #8). Every action goes through the microscope's ``Executor``,
so all of them share one lock and honour dry-run.

Hardware behaviour encoded here:

* **Explicit labels.** Stage and shutter calls pass the device label to
  MMCore instead of relying on the core's "current" XY/focus/shutter slot, so
  a stand with several stages (the Ti2 has four ``XYStage`` devices, FM-13)
  moves the one the role names.
* **Readback.** A mutation returns what the device reports afterwards, not
  what was asked: stages round to their step size, cameras clamp exposure.
* **One lock section per action** (design §13). A move checks, commands,
  waits and reads back while it holds the microscope lock, so its target
  and its readback are its own; there is no ``wait=False``. Reads (``wait()``
  included), ``stop()`` and closing a shutter skip the lock: MMCore
  serialises calls to one adapter, so they are safe during a move, and a
  stop gets through a plate traverse. A move's deadline is the core
  timeout, which the facade raises to 60 s because a plate traverse
  outlasts MMCore's 5 s default (FM-10); a move that gives up stops the
  stage first (FM-17).
* **A device fault is a ``HardwareError``** (#93, FM-79). MMCore raises a bare
  ``RuntimeError`` for it; each class reaches the core through
  ``as_hardware_errors``, which names the role and the call. A
  ``HardwareError`` raised inside a core call (a halt, a refusal) and any
  exception that is not a ``RuntimeError`` pass untouched. What
  ``Microscope.__init__`` reads from the core is not translated.
* **Pixel size is measured or unknown.** MMCore's calibrated value when it
  has one, else the profile's value for the current objective, else ``0.0``.
* **Dry-run reads real state.** Safety checks and the position a relative
  move starts from use live reads; only the mutation is skipped.
* **Moving Z switches PFS off** on the Nikon stands (micro-manager #1815,
  FM-12). On a stand with an autofocus, a Z move suspends it, moves and
  engages it again, as one compound that no other action can come between
  (m2 §3.4), and engages it again only after a move that completed (FM-88).
  The operator's Z stop wins over the relock: while PFS locks again, or a
  locked ``focus_by_um`` moves its offset, ``ZStage.stop()`` switches it off
  first, since PFS searching or following its offset drives Z.
* **The continuous-focus calls act on a core slot** (FM-87): they take no
  label, so every call first checks that the AutoFocus slot still names the
  autofocus role, as the camera calls do for the camera slot.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, cast

from smc.hardware.capabilities import (
    XY,
    ChannelPreset,
    FocusStep,
    Limits,
    PropertyInfo,
)
from smc.hardware.channels import (
    check_preset_name,
    check_value,
    forbidden_settings,
    replace_preset,
    setting_matches,
)
from smc.hardware.errors import (
    AutofocusExcursionError,
    AutofocusLostError,
    CapabilityMissingError,
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.roles import Role, device_type_name
from smc.hardware.safety import POLL_INTERVAL_S, Motion, SafeCall, Step, StopLink

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from pymmcore_plus import CMMCorePlus

    from smc.hardware.capabilities import Autofocus, LightSource, ZStage
    from smc.hardware.safety import Executor, Safety

__all__ = [
    "DEFAULT_LOCK_TIMEOUT_S",
    "INTENSITY_HINTS",
    "MAX_LOCK_TIMEOUT_S",
    "Z_STOP_OUTCOME_WAIT_S",
    "MMAutofocus",
    "MMCamera",
    "MMChannels",
    "MMLightSource",
    "MMObjectiveTurret",
    "MMProperties",
    "MMShutter",
    "MMXYStage",
    "MMZStage",
    "ZInterlock",
    "as_hardware_errors",
    "autofocus_offset_motion",
]

#: How long a lock is waited for when the profile sets no ``[autofocus]
#: lock_timeout_s``. It is nikon-control's figure, never measured, so it is
#: labelled as assumed wherever it shows (m2 §3.2). Read at call time.
DEFAULT_LOCK_TIMEOUT_S = 5.0
#: How long a call that a Z stop reached waits for that stop's switch-off to
#: return before it reports (FM-97, m2 §3.3.1). Assumed: MMCore's default
#: device timeout, never measured on a PFS. Read at call time, so that a test
#: can shorten it.
Z_STOP_OUTCOME_WAIT_S = 5.0
#: The longest lock wait ``engage`` accepts: the wait holds the microscope
#: lock, so every other action waits behind it (FM-32).
MAX_LOCK_TIMEOUT_S = 600.0
#: The longest settle window ``[autofocus] settle_s`` may set (m2 §12.3): it
#: holds the microscope lock, as the lock wait does.
_MAX_SETTLE_S = 60.0

#: The note a message carries when the lock wait is the default.
_ASSUMED_TIMEOUT = " (assumed: the profile sets no [autofocus] lock_timeout_s)"
_NO_FACTORS: Mapping[str, float] = MappingProxyType({})

logger = logging.getLogger("smc.hardware.backends.mm")


class _FaultTranslator:
    """Forwards every call to the core; a ``RuntimeError`` becomes a ``HardwareError``.

    Stateless and lock-free on purpose: a stop runs through it from a signal
    handler (FM-70), so it adds nothing to ``__getattr__``.
    """

    __slots__ = ("_core", "_role")

    def __init__(self, core: CMMCorePlus, role: str) -> None:
        self._core = core
        self._role = role

    def __getattr__(self, name: str) -> Any:
        # copy and pickle probe dunders on a wrapper whose slots are not filled
        # yet, and ``hasattr(x, "__len__")`` must not find one.
        if name.startswith("__"):
            raise AttributeError(name)
        attr = getattr(self._core, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            except HardwareError:
                # First: HardwareError is a RuntimeError (FM-79), and a callback
                # inside a command may raise the halt or a refusal.
                raise
            except RuntimeError as exc:
                raise HardwareError(self._message(name, args, kwargs, exc)) from exc

        return call

    def _message(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], exc: Exception
    ) -> str:
        shown = ", ".join(
            [*map(repr, args), *(f"{k}={v!r}" for k, v in kwargs.items())]
        )
        message = (
            f"{self._role}: {name}({shown}) failed ({_named(exc)}); check the device: "
            "power, cable, another program holding it"
        )
        # An adapter's bytes may not be UTF-8 (FM-05): one lone surrogate and
        # the message can be neither logged nor serialised.
        return message.encode("utf-8", "replace").decode("utf-8")


def as_hardware_errors(core: CMMCorePlus, role: str) -> CMMCorePlus:
    """``core``, with a device fault raised as a ``HardwareError`` naming ``role`` and the call.

    MMCore reports a fault (a serial timeout, a camera that does not answer, an
    unplugged cable) as a bare ``RuntimeError``, which the CLI can only call a
    bug. ``HardwareError`` is itself a ``RuntimeError``, so it, and every
    other exception, passes untouched (design §2, §6; FM-79). The cast keeps
    mypy checking every call on the core; the proxy is stateless and holds no
    lock, so a stop can run through it from a signal handler (FM-70).
    """
    return cast("CMMCorePlus", _FaultTranslator(core, role))


def _motion(
    core: CMMCorePlus,
    label: str,
    name: str,
    *,
    stoppable: bool,
    stop_refused: bool = False,
) -> Motion:
    def stop() -> None:
        core.stop(label)

    return Motion(
        device=label,
        name=name,
        is_busy=lambda: bool(core.deviceBusy(label)),
        stop=stop if stoppable else None,
        stop_refused=stop_refused,
    )


def autofocus_offset_motion(
    core: CMMCorePlus, label: str, *, stop_refused: bool = False
) -> Motion:
    """The ``autofocus_offset`` device's move, as the ``Executor`` sees it (m2 §3.6).

    One definition for ``MMAutofocus``, which waits for it, and for the
    emergency stop, which stops that device by its role (m2 §3.11): the same
    key, so that the stop cancels the move under way, and the same name in
    the lines. Both pass the profile's declaration (``stop_refused``, m2
    §11.4), so both build the same ``Motion``.
    """
    role = Role.autofocus_offset.value
    return _motion(
        as_hardware_errors(core, role),
        label,
        f"{role} {label}",
        stoppable=True,
        stop_refused=stop_refused,
    )


def _core_timeout(core: CMMCorePlus) -> Callable[[], float]:
    """The core timeout in seconds, resolved inside the action (design §2, §13)."""
    return lambda: core.getTimeoutMs() / 1000.0


class MMXYStage:
    """``XYStage`` over one Micro-Manager XY stage device."""

    def __init__(
        self, core: CMMCorePlus, label: str, executor: Executor, safety: Safety
    ) -> None:
        self._core = as_hardware_errors(core, Role.xy_stage.value)
        self._label = label
        self._executor = executor
        self._safety = safety
        # From the proxied core: a busy poll or a stop that faults is a
        # HardwareError too, and the Executor treats it as a failed wait (FM-19).
        self._motion = _motion(
            self._core,
            label,
            f"xy_stage {label}",
            stoppable=True,
            stop_refused=Role.xy_stage.value in safety.stop_refused_by,
        )

    def _read_position(self) -> XY:
        return XY(
            float(self._core.getXPosition(self._label)),
            float(self._core.getYPosition(self._label)),
        )

    def position_um(self) -> XY:
        """Where the stage reports it is."""
        return self._executor.read(self._read_position)

    def _step(self, log: str, x_um: float, y_um: float) -> Step[XY]:
        return Step(
            log=log,
            send=lambda: self._core.setXYPosition(self._label, x_um, y_um),
            readback=self._read_position,
            dry_result=XY(x_um, y_um),
        )

    def _move(self, description: str, step: Callable[[], Step[XY]]) -> XY:
        return self._executor.do(
            description, step, motion=self._motion, timeout_s=_core_timeout(self._core)
        )

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        """Move to an absolute position (soft limits apply, no jog guard).

        Returns where the stage landed, once it reports idle (§13). A UI that
        must stay responsive moves from a worker thread: reads and ``stop()``
        never wait for the microscope lock.
        """
        x_um, y_um = float(x_um), float(y_um)
        self._safety.check_xy_target_um(x_um, y_um)
        description = f"xy_stage: move_to ({x_um}, {y_um}) µm"
        return self._move(description, lambda: self._step(description, x_um, y_um))

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        """Move relative to the live position; jog-guarded unless ``force``.

        The target is computed from a position read inside the action, after
        the guard, and moved to absolutely: the soft-limit check sees the real
        destination, and no other caller can move the stage in between.
        """
        dx_um, dy_um = float(dx_um), float(dy_um)
        self._safety.check_jog_um(dx_um, dy_um, force=force)

        def relative() -> Step[XY]:
            here = self._read_position()
            x_um, y_um = here.x_um + dx_um, here.y_um + dy_um
            self._safety.check_xy_target_um(x_um, y_um)
            return self._step(
                f"xy_stage: move_by ({dx_um}, {dy_um}) µm to ({x_um}, {y_um}) µm",
                x_um,
                y_um,
            )

        return self._move(f"xy_stage: move_by ({dx_um}, {dy_um}) µm", relative)

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until idle; ``DeviceTimeoutError`` after ``timeout_s`` (default: core timeout).

        A read (§13): it never takes the microscope lock and never stops the
        stage, so it can watch a move made by another thread, or by hand.
        """
        self._executor.wait(
            self._motion, _core_timeout(self._core) if timeout_s is None else timeout_s
        )

    def is_busy(self) -> bool:
        """Whether the stage reports it is still moving."""
        return self._executor.read(lambda: bool(self._core.deviceBusy(self._label)))

    def stop(self) -> None:
        """Stop the stage now, without waiting for the microscope lock (§13)."""
        self._executor.stop(self._motion)

    def limits_um(self) -> tuple[Limits, Limits] | None:
        """The configured soft limits as ``(x, y)``; ``None`` when none are configured."""
        limits = self._safety.xy_soft_limits_um
        if limits is None:
            return None
        (x_low, x_high), (y_low, y_high) = limits
        return (Limits(x_low, x_high), Limits(y_low, y_high))


class ZInterlock(Protocol):
    """What ``MMZStage`` needs of an autofocus to move Z under it (m2 §3.4); internal.

    ``MMAutofocus`` implements it. Both calls run only inside the Z move's
    compound, so no other action comes between the suspend, the move and
    the relock. ``str()`` of an interlock names it in the Z stage's log lines
    (``autofocus PFS``).
    """

    def suspend(self, reason: str) -> bool:
        """Switch it off if it is engaged; ``True``: it was engaged and is now off."""
        ...

    def relock(self, reason: str, *, z_um: float | None) -> None:
        """Engage it with the profile's timeout; no lock: ``AutofocusLostError``."""
        ...


class MMZStage:
    """``ZStage`` over one Micro-Manager stage (focus) device.

    With an autofocus (``autofocus`` returns its interlock), a move runs as
    one compound: the guards, the suspend, the move and, only once the move
    has returned normally, the relock (m2 §3.4). The interlock is asked for at
    each move, never cached: the facade builds the autofocus lazily, and an
    autofocus overridden since is no longer interlocked.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        safety: Safety,
        *,
        autofocus: Callable[[], ZInterlock | None] = lambda: None,
    ) -> None:
        self._core = as_hardware_errors(core, Role.focus.value)
        self._label = label
        self._executor = executor
        self._safety = safety
        self._motion = _motion(
            self._core,
            label,
            f"z {label}",
            stoppable=True,
            stop_refused=Role.focus.value in safety.stop_refused_by,
        )
        self._autofocus = autofocus

    def _read_position(self) -> float:
        return float(self._core.getPosition(self._label))

    def position_um(self) -> float:
        """Where the drive reports it is."""
        return self._executor.read(self._read_position)

    def _step(self, log: str, z_um: float) -> Step[float]:
        return Step(
            log=log,
            send=lambda: self._core.setPosition(self._label, z_um),
            readback=self._read_position,
            dry_result=z_um,
        )

    def _move(
        self, description: str, step: Step[float] | Callable[[], Step[float]]
    ) -> float:
        return self._executor.do(
            description, step, motion=self._motion, timeout_s=_core_timeout(self._core)
        )

    def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float:
        """Move to an absolute position (soft limits apply).

        Returns where the drive landed, once it reports idle (§13), as for
        ``XYStage.move_to_um``. Under an engaged autofocus, the readback is
        the one taken before the relock: once it holds focus again, the drive
        reads where the autofocus holds it.
        """
        z_um = float(z_um)
        self._safety.check_z_target_um(z_um)
        description = f"z: move_to {z_um} µm"
        return self._interlocked(
            description, lambda: self._step(description, z_um), keep_autofocus
        )

    def move_by_um(
        self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
    ) -> float:
        """Move relative to the live position; jog-guarded unless ``force``.

        The anchor is read inside the action, as for ``XYStage.move_by_um``.
        ``force`` passes the jog guard only: the soft limits apply to the
        target, read inside the action, and cannot be forced. Under an
        autofocus, the anchor is read before the suspend and the move sent is
        absolute, so a drive that settles after PFS lets go cannot shift it.
        """
        dz_um = float(dz_um)
        self._safety.check_z_jog_um(dz_um, force=force)

        def relative() -> Step[float]:
            z_um = self._read_position() + dz_um
            self._safety.check_z_target_um(z_um)
            return self._step(f"z: move_by {dz_um} µm to {z_um} µm", z_um)

        return self._interlocked(f"z: move_by {dz_um} µm", relative, keep_autofocus)

    def _interlocked(
        self,
        description: str,
        step: Callable[[], Step[float]],
        keep_autofocus: bool,
    ) -> float:
        """The move, inside the autofocus's suspend and relock when there is one (m2 §3.4).

        Without an autofocus it is M1's move: no compound, no extra read. With
        one, ``step`` runs inside the compound before the suspend, so a target
        the soft limits refuse never touches the autofocus. The relock runs
        only when the move returned: after a stop, a halt, a timeout, a fault
        or a Ctrl-C, engaging PFS would move Z after the operator's stop
        (FM-88), so it stays off and a WARNING says so.
        """
        interlock = self._autofocus()
        if interlock is None:
            return self._move(description, step)

        def run() -> float:
            prepared = step()
            # A Z stop pressed while this move waited for the lock cancels it
            # before the autofocus is touched (design §13, m2 §3.7).
            self._executor.refuse_if_stopped(
                self._label,
                MotionStoppedError(
                    self._motion.name, "it was stopped before it started"
                ),
            )
            was_on = interlock.suspend(description)
            try:
                landed = self._move(description, prepared)
            except BaseException as exc:
                if was_on:
                    logger.warning(
                        "%s: left off, because %s did not complete (%s)",
                        interlock,
                        description,
                        type(exc).__name__,
                    )
                raise
            if was_on:
                if keep_autofocus:
                    interlock.relock(description, z_um=landed)
                else:
                    logger.info("%s: left off (keep_autofocus=False)", interlock)
            return landed

        return self._executor.compound(description, run)

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until idle; ``DeviceTimeoutError`` after ``timeout_s`` (default: core timeout).

        Same contract as ``XYStage.wait``.
        """
        self._executor.wait(
            self._motion, _core_timeout(self._core) if timeout_s is None else timeout_s
        )

    def is_busy(self) -> bool:
        """Whether the drive reports it is still moving."""
        return self._executor.read(lambda: bool(self._core.deviceBusy(self._label)))

    def stop(self) -> None:
        """Stop the drive now, without waiting for the microscope lock (§13).

        While PFS can drive Z because of an action, this switches the
        autofocus off first, and stops its offset device while that moves
        (m2 §3.3.1, ``MMAutofocus._ended_by_a_z_stop``): the relock after a
        Z move, a locked ``focus_by_um``, a direct ``engage()``, and a direct
        ``set_offset_native`` or ``offset_by_native`` with it on. Otherwise
        it leaves the autofocus alone: PFS holding focus is not a motion, and
        with it off, an offset move moves no Z.
        """
        self._executor.stop(self._motion)

    def limits_um(self) -> Limits | None:
        """The configured soft limits; ``None`` when none are configured."""
        limits = self._safety.z_soft_limits_um
        return None if limits is None else Limits(*limits)


def _checked_lock_timeout(timeout_s: float, name: str) -> float:
    """A lock wait in (0, ``MAX_LOCK_TIMEOUT_S``] s, or ``ValueError`` before anything is sent.

    The wait holds the microscope lock: ``nan`` would never end, nor ``inf``,
    and an hour would block every other action for that long (FM-32).
    """
    value = float(timeout_s)
    if not (math.isfinite(value) and 0 < value <= MAX_LOCK_TIMEOUT_S):
        raise ValueError(
            f"{name} must be finite and in (0, {MAX_LOCK_TIMEOUT_S:g}] s, got "
            f"{timeout_s!r}"
        )
    return value


def _yes_no(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _named(failure: BaseException) -> str:
    """``failure`` for a message: an interrupt's ``str()`` is empty, so its type names it.

    Also when ``str()`` itself fails: a log line or a read that is unknown
    "never an error" must not become one (#166).
    """
    try:
        return str(failure) or type(failure).__name__
    except Exception:
        return type(failure).__name__


def _check_autofocus_slot(core: CMMCorePlus, label: str) -> None:
    """Refuse unless the core's AutoFocus slot names ``label`` (FM-87).

    MMCore's continuous-focus calls take no label: they act on the slot's
    device, so a slot that something moved would switch another device.
    """
    current = core.getAutoFocusDevice()
    if current != label:
        raise HardwareError(
            f"autofocus {label!r} is not the device the core's AutoFocus "
            f"slot names ({current!r}), and the continuous-focus calls act on "
            f"the slot's device; something changed the slot after the stand "
            f"was opened. Open it again, or set Core-AutoFocus in the "
            f"configuration and the autofocus role under [roles.assign] to the "
            f"same device."
        )


def _send_continuous_focus_off(core: CMMCorePlus, label: str) -> None:
    """Switch the autofocus off, after the slot check; the switch-off's ``SafeCall`` sends it.

    A function of the core and the label, not a method: the ``SafeCall`` is
    stored on the ``MMAutofocus``, and one that held ``self._send_off`` would
    be a reference cycle through the core (#142).
    """
    _check_autofocus_slot(core, label)
    core.enableContinuousFocus(False)


@dataclass(frozen=True, slots=True)
class _Lock:
    """How a lock wait ended: locked, or not, with what the stand said of its range."""

    locked: bool
    in_range: bool | None = None


#: How a lock wait's readback found the autofocus once the wait ended idle.
_AfterWait = Literal["locked", "switched itself off", "lost the lock at once"]


@dataclass
class _ZStop:
    """What the operator's Z stop did to a compound of the autofocus.

    Written by the stopping thread and read by the compound's, one
    attribute at a time, which the GIL keeps whole.
    """

    #: The compound's link to the focus drive's stop (``Executor.linked_stop``).
    link: StopLink | None = None
    #: The offset move is about to be sent or under way: the Z stop stops
    #: the offset device too, which also cancels a move not sent yet.
    offset_pending: bool = False
    #: What the Z stop's switch-off raised: it may still be on. An interrupt
    #: counts, since whether the switch-off went out is then unknown.
    switch_off_failure: BaseException | None = None
    #: Set by the stopping thread once its switch-off has returned or raised:
    #: the link is marked reached before the switch-off is sent (FM-97). A
    #: plain store, not an ``Event``: setting one takes a lock, which a
    #: second stop re-entering that thread would wait on for ever (FM-70).
    finished: bool = False
    #: Set by the compound once it reads no more; the last of the two threads
    #: to finish clears ``switch_off_failure``. The failure's traceback holds
    #: the stopping thread's frames, which hold this object: kept, it would
    #: keep the autofocus and the core until the cyclic collector (FM-92,
    #: FM-98). The compound sets this, then clears; the stopping thread sets
    #: the failure, then ``finished``, then reads this. Plain stores (FM-70).
    closed: bool = False

    @property
    def fired(self) -> bool:
        """Whether a Z stop reached the link: set as that stop took its snapshot."""
        return self.link is not None and self.link.reached


class _LostLockError(DeviceTimeoutError):
    """Given up on once its wait ended, and it could not be switched off.

    It locked and lost the lock at once, or the Z guard ended the wait (m2
    §12.4). A ``DeviceTimeoutError`` for every caller; its own class only so
    that ``engage`` does not take it for its lock wait's deadline.
    """


class _EngageRefusedError(SafetyRefusedError):
    """The engage check refused before the switch-on (m2 §12.4); never forceable.

    Its own class so that a relock can turn it into ``AutofocusLostError``,
    with ``detail``, the reason after "not engaged: ", in that message.
    """

    def __init__(self, detail: str, label: str) -> None:
        self.detail = detail
        self.label = label
        super().__init__(f"autofocus {label}: not engaged: {detail}")

    def __reduce__(self) -> tuple[type[_EngageRefusedError], tuple[str, str]]:
        return (type(self), (self.detail, self.label))


class _EngageGuard:
    """The Z guard and the settle window around one engage (m2 §12.2 D2, D3).

    Polled as the lock wait's ``Motion.is_busy``, in the action's thread and
    under the microscope lock. Each poll reads engaged, locked and Z: busy
    while it searches or settles. Z outside the soft limits or the band, or
    a Z read that fails, ends the wait at once (idle), and the outcome is kept
    as text, never as an exception (FM-98): the readback gives up on the
    motion, which switches it off, then raises. From then on a poll reads
    only whether it is on, so that a switch-off that failed keeps the motion
    registered (FM-127).

    The band is ``[min(z_start, z_held) - E, max(z_start, z_held) + E]``: a
    relock brings Z back to where PFS held it, so the way back is inside it.
    The window is settled once every Z read of the last ``settle_s`` lies
    within ``tolerance_um`` of each other, still locked; a lock lost meanwhile
    starts it again at the next locked read.

    It limits a jump, it does not prevent one: Z is read every
    ``POLL_INTERVAL_S`` plus the read's own time, and run 1's jump took less
    than 100 ms (#205).
    """

    def __init__(
        self,
        autofocus: MMAutofocus,
        focus: ZStage,
        *,
        z_start_um: float,
        z_held_um: float | None,
        limits_um: tuple[float, float] | None,
        excursion_um: float | None,
        settle_s: float,
        tolerance_um: float,
    ) -> None:
        self._autofocus = autofocus
        self._focus = focus
        self.z_start_um = z_start_um
        self._limits = limits_um
        self._excursion_um = excursion_um
        self._band: tuple[float, float] | None = None
        if excursion_um is not None:
            held = z_start_um if z_held_um is None else z_held_um
            self._band = (
                min(z_start_um, held) - excursion_um,
                max(z_start_um, held) + excursion_um,
            )
        self._settle_s = settle_s
        self._tolerance_um = tolerance_um
        #: The (objective, offset) it engages at, on a declared stand: what a
        #: trip forgets.
        self.seen: tuple[str | None, float] | None = None
        #: Why Z tripped it, and the read that did; or why Z could not be read.
        self.why = ""
        self.tripped_z_um: float | None = None
        self.failure = ""
        #: The last poll read locked but not settled, and how far Z moved then.
        self.settling = False
        self.spread_um = 0.0
        self._window: list[tuple[float, float]] = []
        self._since = 0.0

    @property
    def ended(self) -> bool:
        """Whether it ended the wait: Z out of bounds, or unreadable."""
        return bool(self.why or self.failure)

    @property
    def given_up_for(self) -> str:
        """The give-up's reason, for its log line."""
        return "a Z read that failed" if self.failure else "Z out of its bound"

    def what_happened(self) -> str:
        """The first clause of the error (m2 §12.5)."""
        if self.tripped_z_um is None:
            return f"Z could not be read while it engaged ({self.failure})"
        return (
            f"Z went from {self.z_start_um:.2f} µm to {self.tripped_z_um:.2f} µm "
            f"while it engaged, {self.why}"
        )

    def is_busy(self) -> bool:
        autofocus = self._autofocus
        engaged = autofocus._read_engaged()
        if self.ended:
            return engaged
        locked = engaged and autofocus._read_locked(record=False)
        z_um = self._read_z()
        if z_um is None or not engaged:
            return False
        if not locked:
            self._window = []
            self.settling = False
            return True
        if self._settle_s <= 0:
            return False
        now = time.monotonic()
        if not self._window:
            self._since = now
        window = [(t, z) for t, z in self._window if t >= now - self._settle_s]
        window.append((now, z_um))
        self._window = window
        values = [z for _, z in window]
        self.spread_um = max(values) - min(values)
        if now - self._since >= self._settle_s and self.spread_um <= self._tolerance_um:
            self.settling = False
            return False
        self.settling = True
        return True

    def _read_z(self) -> float | None:
        """Z, or ``None`` once it ended the wait: a failed read, or Z out of bounds."""
        try:
            z_um = float(self._focus.position_um())
        except Exception as exc:
            self.failure = _named(exc)
            return None
        limits, band = self._limits, self._band
        if limits is not None and not limits[0] <= z_um <= limits[1]:
            self.why = f"outside the soft limits [{limits[0]:g}, {limits[1]:g}] µm"
        elif band is not None and not band[0] <= z_um <= band[1]:
            self.why = (
                f"more than [safety] max_engage_excursion_um ({self._excursion_um:g} "
                f"µm) beyond where it started"
            )
        else:
            return z_um
        self.tripped_z_um = z_um
        return None


class _EndedByAZStop:
    """Inside a compound: from here on, the operator's Z stop wins (FM-88; review of #140).

    While the block runs, a ``ZStage.stop()`` switches the autofocus off,
    and stops the offset device while its move is pending or under way,
    before Z's own stop (``Executor.linked_stop``): PFS locking again, or
    following its offset, drives Z after the operator stopped it. The
    block then ends with ``MotionStoppedError`` and the "left off"
    WARNING, whatever it was doing; a stop that lands once it has locked
    switches it off as well, and is reported the same way. A Z stop that
    came since the compound was called, before this block, cancels it
    with nothing sent. ``suspended`` says the autofocus is off already
    (the relock), so that it is then left off, with the WARNING, as after
    a halt that refuses the relock before it sends anything; otherwise
    (a locked ``focus_by_um``, a direct ``engage()`` or offset move, m2
    §3.3.1) it was not touched, so a Z stop before the block leaves it as
    it was and claims nothing. ``offset_pending`` is
    set before the link is, so that no Z stop falls between the two;
    the caller clears it once the offset move has returned.

    Once the block has ended, a call that a Z stop reached waits for
    that stop's switch-off to return, at most ``Z_STOP_OUTCOME_WAIT_S``,
    still under the microscope lock, in short sleeps that a Ctrl-C
    interrupts (FM-97, FM-83). The link is marked reached before the
    stopping thread sends the switch-off, so a block that its lock or its
    landed move ended meanwhile would otherwise report "left off" while
    the switch-off is still on its way, and may yet fail. A block that a
    halt or a Ctrl-C ended does not wait: it goes on at once, with what
    is known by then. Nor does one that comes during the wait: a halt is
    read from the stand's halted state at each poll, since the Z stop has
    already ended the block with a plain ``MotionStoppedError`` by the time
    the emergency stop comes (the call then reports the switch-off as not
    returned yet, and cannot know whether the halt's own worked); a Ctrl-C
    is decided as one that ended the block, so that its WARNING is logged
    (#166). When the switch-off failed, or had not returned in time, it may
    still be on: no "left off" is claimed, the "may still move Z" WARNING
    is logged, and the call raises ``MotionStoppedError`` naming Z and
    the failure, chained from whatever ended the block (m2 §3.3.1).
    Three endings go on as they are: a ``DeviceTimeoutError`` (a give-up
    whose own error says it may still move Z, with its motion still
    registered), a halt and an interrupt.

    While halted, the emergency stop switches the autofocus off and stops
    the offset device itself, in its own order (m2 §3.11): the Z stop
    then reaches no link, and the block ends on what the halt did.
    Without a focus drive of this module, no Z stop is linked.

    A class, not a ``@contextmanager`` generator: a generator that raises a
    different exception from the one thrown into it keeps contextlib's
    frame, and through it the autofocus and the core, in a reference cycle
    on Python 3.11+ (#163, FM-92, FM-98). For the same reason it keeps no
    exception on ``self``, and every frame that raises an exception it holds
    in a local deletes that local first.
    """

    def __init__(
        self,
        autofocus: MMAutofocus,
        description: str,
        *,
        suspended: bool,
        offset_pending: bool,
    ) -> None:
        self._autofocus = autofocus
        self._description = description
        self._suspended = suspended
        self._stopped = _ZStop(offset_pending=offset_pending)
        self._name = ""
        self._link: AbstractContextManager[StopLink] | None = None

    def __enter__(self) -> _ZStop:
        autofocus, stopped = self._autofocus, self._stopped
        key = autofocus._focus_key()
        if key is None:
            return stopped
        self._name = f"z {key}"
        before = MotionStoppedError(self._name, "it was stopped before it started")
        link = self._link = autofocus._executor.linked_stop(
            key, lambda: autofocus._on_z_stop(stopped)
        )
        try:
            stopped.link = link.__enter__()
            autofocus._executor.refuse_if_stopped(key, before)
        except BaseException as exc:
            # ``__exit__`` does not run when ``__enter__`` raises: the refusal
            # (this check, or a halt, #141) ends the block here. The link is
            # left in every case, as a ``with`` statement would: an interrupt
            # once its entry has returned would otherwise leave it live; one
            # whose own entry failed is gone, and leaving it does nothing.
            self._link = None
            try:
                self._end(link, exc, refused=exc is before)
            finally:
                self._close()
            raise
        finally:
            # Raised through this frame, it is on its own traceback (FM-98).
            del before
        return stopped

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Never swallows: the block's exception goes on, or another is raised.
        link, self._link = self._link, None
        if link is None:
            return
        try:
            self._end(link, exc, refused=False)
        finally:
            self._close()

    def _end(
        self,
        link: AbstractContextManager[StopLink],
        exc: BaseException | None,
        *,
        refused: bool,
    ) -> None:
        """Leave the link, then decide; returns when the block's ending goes on as it is."""
        try:
            if exc is None:
                link.__exit__(None, None, None)
            else:
                tb = exc.__traceback__
                try:
                    link.__exit__(type(exc), exc, tb)
                finally:
                    # Python 3.10's contextlib leaves its own frames on the
                    # traceback of the exception it re-raised, and its frame
                    # holds that exception: a cycle (FM-98). 3.11+ restores
                    # the traceback itself; a ``with`` statement did too.
                    exc.__traceback__ = tb
        except BaseException as raised:
            # An interrupt in the link's own exit ends the block in place of
            # ``exc``, as the generator's outer ``try`` caught it.
            self._decide(raised, refused=False)
            raise
        self._decide(exc, refused=refused)

    def _decide(self, exc: BaseException | None, *, refused: bool) -> None:
        """The decision table (m2 §3.3.1); returns when the block's ending goes on as it is."""
        autofocus, stopped = self._autofocus, self._stopped
        name, description = self._name, self._description
        failure: BaseException | None = None
        error: MotionStoppedError | None = None
        try:
            if exc is not None:
                # Suspended, a refusal before the engage was sent (the check
                # on entry, or a halt, #141) leaves it off as surely as a Z
                # stop does.
                refused = refused or isinstance(exc, MicroscopeHaltedError)
                if not (stopped.fired or (self._suspended and refused)):
                    return
                # A halt and a Ctrl-C go on at once: they wait for no outcome.
                urgent = isinstance(exc, MicroscopeHaltedError) or not isinstance(
                    exc, Exception
                )
                failure = self._outcome(wait=not urgent) if stopped.fired else None
                if failure is not None:
                    autofocus._log_not_switched_off(description, failure)
                    if urgent or isinstance(exc, DeviceTimeoutError):
                        return
                    error = autofocus._not_switched_off(name, description, failure)
                    raise error from exc
                if urgent:
                    autofocus._log_left_off(description, exc)
                    return  # a halt stays a halt, a Ctrl-C a Ctrl-C
                error = autofocus._z_stopped(name, description)
                autofocus._log_left_off(description, error)
                raise error from exc
            if stopped.fired:
                # It locked, then the stop switched it off: not a success.
                failure = self._outcome(wait=True)
                if failure is not None:
                    autofocus._log_not_switched_off(description, failure)
                    error = autofocus._not_switched_off(name, description, failure)
                    raise error from failure
                error = autofocus._z_stopped(name, description)
                autofocus._log_left_off(description, error)
                raise error
        finally:
            # `error` is raised through this frame: kept here, it would be on
            # its own traceback (FM-98). `failure`, chained to it, goes with it.
            del failure, error

    def _outcome(self, *, wait: bool) -> BaseException | None:
        """The Z stop's switch-off outcome; a Ctrl-C in the wait is decided as one that ended the block.

        The wait is the one place where a Ctrl-C can come after the block
        ended, and the exception would otherwise leave before any WARNING
        said that the switch-off had not returned (#166). The nested decision
        sees an interrupt: no wait, the WARNING of what is known, and it
        returns; the interrupt goes on. Python deletes ``interrupt`` at the
        end of the clause, on the bare ``raise`` too: bound to no other
        local (FM-98).
        """
        try:
            return self._autofocus._z_stop_failure(self._stopped, wait=wait)
        except BaseException as interrupt:
            self._decide(interrupt, refused=False)
            raise

    def _close(self) -> None:
        """The compound reads no more: drop what would keep the core in a cycle.

        The link's callback holds ``stopped``, and ``stopped`` held the link:
        a cycle that kept this autofocus and the core alive until the cyclic
        collector (FM-92). The callback never reads ``link``, a stop that is
        still on its way needs only ``stopped``, and nothing reads ``fired``
        after this. The switch-off's failure is cleared after ``closed`` is
        set, so that whichever thread finishes last clears it (``_ZStop``).
        """
        stopped = self._stopped
        stopped.link = None
        stopped.closed = True
        stopped.switch_off_failure = None


class MMAutofocus:
    """``Autofocus`` over the core's continuous-focus API (m2 §3.2 to §3.6); the ``ZInterlock``.

    Hardware behaviour encoded here:

    * **The slot, not the label** (FM-87). ``enableContinuousFocus``,
      ``isContinuousFocus*`` and ``get``/``setAutoFocusOffset`` act on the
      device the core's AutoFocus slot names. Every call that reaches them
      first checks that the slot still names ``label``, and refuses otherwise.
    * **The lock wait is a motion** (m2 §3.3): until it locks, PFS searches
      and may move Z, so no other action may run, a Ctrl-C or a fault gives
      up on it, and giving up switches it off (FM-17). Once locked, its
      corrections are not a motion: the stand would otherwise refuse every
      action while PFS holds focus.
    * **The offset is in the device's own units** and is never converted
      implicitly. With an ``autofocus_offset`` device (the Ti2-E's
      ``PFSOffset``) it is that stage's position; else the autofocus device's
      own (the demo's). ``setAutoFocusOffset`` is never used when an offset
      device is resolved: pymmcore-plus redirects it to the offset stage for
      ``NikonTI`` only, not for ``NikonTi2`` (m2 §1).
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        safety: Safety,
        *,
        offset_label: str | None,
        offset_unit: str = "",
        lock_timeout_s: float | None = None,
        offset_per_um: Mapping[str, float] = _NO_FACTORS,
        in_range_property: str = "",
        in_range_values: Sequence[str] = (),
        objective_label: Callable[[], str | None] = lambda: None,
        focus: Callable[[], ZStage | None] = lambda: None,
        engage_drives_z_to_offset: bool = False,
        settle_s: float = 0.0,
        settle_tolerance_um: float = 0.5,
    ) -> None:
        if lock_timeout_s is not None:
            lock_timeout_s = _checked_lock_timeout(lock_timeout_s, "lock_timeout_s")
        # As the profile validates them (m2 §12.3), for code that builds this.
        settle_s = float(settle_s)
        if not (math.isfinite(settle_s) and 0 <= settle_s <= _MAX_SETTLE_S):
            raise ValueError(
                f"settle_s must be finite and in [0, {_MAX_SETTLE_S:g}] s, got "
                f"{settle_s!r}"
            )
        bound = (
            float(DEFAULT_LOCK_TIMEOUT_S) if lock_timeout_s is None else lock_timeout_s
        )
        if settle_s >= bound:
            raise ValueError(
                f"settle_s must be below the lock timeout ({bound:g} s), which bounds "
                f"the lock and the settling together; got {settle_s!r}"
            )
        settle_tolerance_um = float(settle_tolerance_um)
        if not (math.isfinite(settle_tolerance_um) and settle_tolerance_um > 0):
            raise ValueError(
                f"settle_tolerance_um must be finite and > 0, got "
                f"{settle_tolerance_um!r}"
            )
        for objective, factor in offset_per_um.items():
            # A zero factor would make focus_by_um a move of nothing, reported
            # as done; the profile refuses it too, but code may build this.
            if not (math.isfinite(factor) and factor != 0):
                raise ValueError(
                    f"offset_per_um for objective {objective!r} must be finite and "
                    f"non-zero, got {factor!r}"
                )
        # Rebound: the switch-off below captures the proxied core, and a
        # switch-off that faults must be translated as well (as MMShutter).
        # The offset device's proxy wraps the raw core: wrapping the autofocus
        # proxy would name every offset fault "autofocus".
        self._offset_core = as_hardware_errors(core, Role.autofocus_offset.value)
        offset_motion = (
            None
            if offset_label is None
            else autofocus_offset_motion(
                core,
                offset_label,
                stop_refused=Role.autofocus_offset.value in safety.stop_refused_by,
            )
        )
        core = self._core = as_hardware_errors(core, Role.autofocus.value)
        self._label = label
        self._executor = executor
        self._safety = safety
        self._offset_label = offset_label
        self._offset_unit = offset_unit
        self._lock_timeout_s = lock_timeout_s
        self._offset_per_um = dict(offset_per_um)
        self._in_range_property = in_range_property
        self._in_range_values = tuple(in_range_values)
        self._objective_label = objective_label
        self._focus = focus
        self._declared = bool(engage_drives_z_to_offset)
        self._settle_s = settle_s
        self._settle_tolerance_um = settle_tolerance_um
        #: Per objective label (``None`` without a turret): the offsets read
        #: locked since this was built (m2 §12.2 D1). Recorded on a declared
        #: stand only, and replaced whole at each change: one store, no lock
        #: (FM-70).
        self._seen: dict[str | None, frozenset[float]] = {}
        #: Where Z was when ``suspend()`` found it locked, for the relock that
        #: follows in the same compound (the band, m2 §12.2 D2).
        self._z_held: float | None = None
        # Not `self._send_off`: a bound method stored on self is a cycle that
        # keeps the core alive until the cyclic collector (#142).
        self._off = SafeCall(
            device=label,
            log="autofocus: disengage",
            send=functools.partial(_send_continuous_focus_off, core, label),
        )
        if offset_motion is not None:
            self._offset_motion = offset_motion
        else:
            # The autofocus device's own offset: no stop of its own (m2 §3.6).
            self._offset_motion = Motion(
                device=label,
                name=f"autofocus {label} offset",
                is_busy=lambda: bool(core.deviceBusy(label)),
                stop=None,
            )

    def __str__(self) -> str:
        """How the Z stage's log lines name it (``ZInterlock``)."""
        return f"autofocus {self._label}"

    # --- the slot and the reads ----------------------------------------------

    def _check_slot(self) -> None:
        _check_autofocus_slot(self._core, self._label)

    def _read_engaged(self) -> bool:
        self._check_slot()
        return bool(self._core.isContinuousFocusEnabled())

    def _read_locked(self, *, record: bool = True) -> bool:
        """The one place a lock is read; on a declared stand, a lock read makes its offset seen.

        m2 §12.2 D1: every read the layer makes that finds it locked counts,
        so the operator's lock taken at the stand is seen at the first
        ``is_locked()``. ``record=False`` is the guard's poll only: it engaged
        a seen offset, and two more reads per poll would slow its Z reads.
        """
        self._check_slot()
        locked = bool(self._core.isContinuousFocusLocked())
        if locked and record and self._declared:
            self._record_seen()
        return locked

    def _record_seen(self) -> None:
        """Add the offset now set, with the objective now in place, to the seen offsets.

        A failure of either read records nothing: the absence of a seen
        offset is the safe side (m2 §12.4). Compared as read back, exactly:
        one readback of a device against another.
        """
        try:
            objective = self._objective_label()
            offset = self._read_offset()
        except Exception as exc:
            # Text, never the error: a record that keeps it keeps its traceback (FM-92).
            logger.debug(
                "autofocus %s: read locked, but its offset or objective could not "
                "be read (%s); not recorded as seen",
                self._label,
                _named(exc),
            )
            return
        seen = self._seen
        known = seen.get(objective, frozenset())
        if offset not in known:
            self._seen = {**seen, objective: known | {offset}}

    def _forget(self, objective: str | None, offset: float) -> None:
        """Drop ``offset`` from the seen offsets of ``objective``: it drove Z out of bounds."""
        seen = self._seen
        self._seen = {**seen, objective: seen.get(objective, frozenset()) - {offset}}

    def _searching(self) -> bool:
        """The lock wait's busy state: on, and not holding focus yet."""
        return self._read_engaged() and not self._read_locked()

    def _read_in_range(self) -> bool | None:
        if not self._in_range_property:
            return None
        value = str(self._core.getProperty(self._label, self._in_range_property))
        return value in self._in_range_values

    def _in_range_for_message(self) -> bool | None:
        """``is_in_range()`` for a message: a failed read is unknown here, never an error."""
        try:
            return self._read_in_range()
        except Exception as exc:
            # Text, never the error: a record that keeps it keeps its traceback (FM-92).
            logger.debug(
                "autofocus %s: in-range read failed (%s)", self._label, _named(exc)
            )
            return None

    def _read_offset(self) -> float:
        if self._offset_label is not None:
            return float(self._offset_core.getPosition(self._offset_label))
        self._check_slot()
        return float(self._core.getAutoFocusOffset())

    def _unit(self) -> str:
        return self._offset_unit or "device units"

    def is_engaged(self) -> bool:
        """Whether continuous focus is switched on (``isContinuousFocusEnabled``)."""
        return self._executor.read(self._read_engaged)

    def is_locked(self) -> bool:
        """Whether it holds focus now (``isContinuousFocusLocked``)."""
        return self._executor.read(self._read_locked)

    def is_in_range(self) -> bool | None:
        """Whether the profile's in-range property reads one of its in-range values.

        ``None`` without that profile entry: the layer never guesses which
        property or value means "in range".
        """
        return self._executor.read(self._read_in_range)

    def offset_native(self) -> float:
        """The offset device's position, else the autofocus device's own offset."""
        return self._executor.read(self._read_offset)

    def offset_unit(self) -> str:
        """The profile's ``[autofocus] offset_unit``; ``""``: unknown."""
        return self._offset_unit

    def offset_limits_native(self) -> tuple[float, float] | None:
        """The profile's ``[safety] autofocus_offset_soft_limits``, or ``None``."""
        return self._safety.autofocus_offset_soft_limits

    # --- switching ---------------------------------------------------------

    def _send_on(self) -> None:
        self._check_slot()
        self._core.enableContinuousFocus(True)

    def _send_off(self) -> None:
        _send_continuous_focus_off(self._core, self._label)

    def _lock_timeout(self, timeout_s: float | None) -> tuple[float, bool]:
        """The lock wait's limit, and whether it is the assumed default.

        It bounds the settle window too (m2 §12.2 D3): a limit not above
        ``settle_s`` could never end settled, so it is a ``ValueError``
        before anything is sent.
        """
        if timeout_s is not None:
            limit, assumed = _checked_lock_timeout(timeout_s, "timeout_s"), False
        elif self._lock_timeout_s is not None:
            limit, assumed = self._lock_timeout_s, False
        else:
            # The module's value at call time, so that a test can shorten it.
            limit, assumed = float(DEFAULT_LOCK_TIMEOUT_S), True
        if self._settle_s > 0 and limit <= self._settle_s:
            raise ValueError(
                f"the lock timeout ({limit:g} s{_ASSUMED_TIMEOUT if assumed else ''}) "
                f"must be above [autofocus] settle_s ({self._settle_s:g} s), since it "
                f"bounds the settling too"
            )
        return limit, assumed

    def engage(self, timeout_s: float | None = None) -> bool:
        """Switch continuous focus on and wait for the lock; ``False``: no lock, and it is off.

        The wait is a motion (m2 §3.3): another thread's action is refused at
        once while it lasts, and ``disengage()`` cancels it with
        ``MotionStoppedError``. With no lock by ``timeout_s`` (default: the
        profile's ``lock_timeout_s``, else ``DEFAULT_LOCK_TIMEOUT_S``,
        assumed), the wait gives up like a move, which switches it off, and a
        WARNING names the timeout and the range. ``nikon-control`` left PFS
        on after a failed lock; one left searching can move Z with no action
        holding the lock, so here ``False`` always means "off". In dry-run it
        returns ``True``, the commanded outcome, and sends nothing.

        PFS searching drives Z, so the operator's ``ZStage.stop()`` ends the
        wait too (m2 §3.3.1): it switches the autofocus off before Z's stop,
        and the call raises. A Z stop is never a ``False``: that would read
        as "no lock found", not "stopped".

        Raises:
            ValueError: ``timeout_s`` is not finite or not in (0, 600] s;
                nothing was sent.
            DeviceTimeoutError: No lock, and it could not be switched off (or
                still reads on). The wait stays registered as a motion, so the
                stand refuses actions until ``stop()`` or ``resume()``.
            MotionStoppedError: ``disengage()`` or a stop cancelled the wait;
                a ``ZStage.stop()`` left it off (named ``z <label>``), or
                could not switch it off, or not in time, and then says it may
                still move Z; or it cancelled the call before anything was
                sent.
            HardwareError: Its state could not be read once the wait ended. It
                was given up on as at the deadline: switched off, or still
                registered as a motion if that failed too. Or the Microscope
                it belongs to no longer exists (FM-92); nothing was read or
                sent.
        """
        limit_s, assumed = self._lock_timeout(timeout_s)
        description = "autofocus: engage"

        def run() -> bool:
            with self._ended_by_a_z_stop(description, suspended=False):
                return self._engage(description, limit_s, assumed).locked

        # Its own compound, so that the link and the check for a Z stop made
        # while it waited for the lock have a record to compare against.
        return self._executor.compound(description, run)

    def _engage(
        self,
        description: str,
        limit_s: float,
        assumed: bool,
        *,
        z_held: float | None = None,
    ) -> _Lock:
        """``engage``'s action, also run nested by the relocks; how the wait ended.

        Its step runs the engage check before the switch-on (m2 §12.4), and
        the wait polls the Z guard when there is one: ``z_held`` is where PFS
        held Z before a relock's Z move, which widens the band.
        """
        label = self._label
        switch_off_failures: list[Exception] = []
        guard: _EngageGuard | None = None

        def is_busy() -> bool:
            return self._searching() if guard is None else guard.is_busy()

        def switch_off() -> None:
            # The lock wait's stop: kept, so that a failure can be named.
            try:
                self._send_off()
            except Exception as exc:
                switch_off_failures.append(exc)
                raise

        def after_wait() -> _AfterWait:
            # The wait ended idle: locked, or no longer on, or the guard ended it.
            if guard is not None and guard.ended:
                self._executor.give_up(motion, guard.given_up_for)
                self._guard_ended(guard, switch_off_failures)
            try:
                if self._read_locked():
                    return "locked"
                engaged = self._read_engaged()
            except Exception:
                # Unknown counts as searching (FM-19): left on, it could move
                # Z with no action holding the lock. Given up on as at a
                # deadline, then the read's failure is raised (#141).
                self._executor.give_up(motion, "a state unreadable after the wait")
                raise
            if not engaged:
                return "switched itself off"
            # It locked, then lost the lock before this read: left on, it would
            # search, and move Z, with no action holding the lock. Given up on
            # as at a deadline: switched off, and registered while it searches.
            outcome = self._executor.give_up(motion, "a lock lost at once")
            if switch_off_failures:
                raise _LostLockError(
                    f"autofocus {label}: it locked, then lost the lock at once, and "
                    f"it could not be switched off ({outcome}); it may still move "
                    f"Z: stop() and check the stand, then resume()"
                ) from switch_off_failures[0]
            return "lost the lock at once"

        def step() -> Step[_AfterWait]:
            nonlocal guard
            self._check_slot()  # in dry-run too
            guard = self._check_engage(z_held)  # in dry-run too
            return Step(
                log=description,
                send=self._send_on,
                readback=after_wait,
                dry_result="locked",
            )

        motion = Motion(
            device=label,
            name=f"autofocus {label}",
            is_busy=is_busy,
            stop=switch_off,
        )
        try:
            ended = self._executor.do(
                description, step, motion=motion, timeout_s=limit_s
            )
        except _LostLockError:
            raise
        except DeviceTimeoutError as exc:
            # Otherwise only the lock wait raises this here: the Executor gave
            # up at the deadline and sent the stop, which switches it off. Its
            # busy check after the stop may still have read Z out of bounds.
            if guard is not None and guard.ended:
                self._guard_ended(guard, switch_off_failures)
            return self._no_lock(limit_s, assumed, exc, switch_off_failures, guard)
        finally:
            # A failure's traceback holds `switch_off`'s frame, whose closure
            # holds this list: emptied on every exit, the normal return too
            # (a Z stop's safe call ends the wait through `switch_off`), once
            # read (#163, FM-98). Nothing calls the motion's stop after this.
            switch_off_failures.clear()
        if ended == "locked":
            return _Lock(True)
        in_range = self._in_range_for_message()
        how = (
            "switched itself off before it locked"
            if ended == "switched itself off"
            else "lost the lock at once, so it was switched off"
        )
        logger.warning("autofocus %s: %s (in range: %s)", label, how, _yes_no(in_range))
        return _Lock(False, in_range)

    def _no_lock(
        self,
        limit_s: float,
        assumed: bool,
        timeout: DeviceTimeoutError,
        switch_off_failures: list[Exception],
        guard: _EngageGuard | None = None,
    ) -> _Lock:
        """After the lock wait gave up: ``False`` if it is off, else the timeout, re-raised.

        A lock that had come but not settled is named as such (m2 §12.2 D3):
        "no lock" would send the operator looking for the sample.
        """
        label = self._label
        try:
            still_on = self._read_engaged()
        except Exception as exc:
            outcome = f"its state cannot be read: {exc}"
        else:
            if not still_on and not switch_off_failures:
                in_range = self._in_range_for_message()
                if guard is not None and guard.settling:
                    logger.warning(
                        "autofocus %s: locked, but Z did not settle within %g s (it "
                        "moved %.2f µm in the last %g s); switched off",
                        label,
                        limit_s,
                        guard.spread_um,
                        self._settle_s,
                    )
                else:
                    logger.warning(
                        "autofocus %s: no lock within %g s%s; in range: %s; switched "
                        "off",
                        label,
                        limit_s,
                        _ASSUMED_TIMEOUT if assumed else "",
                        _yes_no(in_range),
                    )
                return _Lock(False, in_range)
            outcome = (
                f"the switch-off failed: {switch_off_failures[0]}"
                if switch_off_failures
                else "it still reads on"
            )
        raise DeviceTimeoutError(
            f"autofocus {label}: no lock within {limit_s:g} s, and it could not be "
            f"switched off ({outcome}); it may still move Z: stop() and check the "
            f"stand, then resume()"
        ) from timeout

    # --- the engage check and the guard (m2 §12) -------------------------------

    def _guards_z(self) -> bool:
        """Whether an engage has a bound to read Z against: a soft limit, a band or a window."""
        return (
            self._safety.z_soft_limits_um is not None
            or self._safety.max_engage_excursion_um is not None
            or self._settle_s > 0
        )

    def _check_engage(self, z_held: float | None) -> _EngageGuard | None:
        """Refuse an engage before the switch-on, and build its guard (m2 §12.4).

        Under the lock, inside the action's step, in dry-run too, as the
        implied-Z check is (FM-90). First Z against the soft limits: the guard
        could not tell a jump from where Z already is. Then, on a stand that
        declares that engaging drives Z to the offset's plane, the offset now
        set must be one seen locked with this objective; a failed read there
        propagates with nothing sent, since it decides whether Z moves
        (FM-19). Neither refusal can be forced: a forced engage of an unknown
        offset at 40x is the crash itself (m2 §12.2 D1).
        """
        label = self._label
        focus = self._focus()
        guard: _EngageGuard | None = None
        if focus is None:
            logger.warning(
                "autofocus %s: no focus drive to read, so Z is not guarded while it "
                "engages (m2 §12)",
                label,
            )
        elif self._guards_z():
            z_start = float(focus.position_um())
            limits = self._safety.z_soft_limits_um
            if limits is not None and not limits[0] <= z_start <= limits[1]:
                raise _EngageRefusedError(
                    f"Z is at {z_start:.2f} µm, outside the soft limits "
                    f"[{limits[0]:g}, {limits[1]:g}] µm",
                    label=label,
                )
            guard = _EngageGuard(
                self,
                focus,
                z_start_um=z_start,
                z_held_um=z_held,
                limits_um=limits,
                excursion_um=self._safety.max_engage_excursion_um,
                settle_s=self._settle_s,
                tolerance_um=self._settle_tolerance_um,
            )
        if not self._declared:
            return guard
        self._read_locked()  # the operator's lock, if it holds now, is seen
        objective = self._objective_label()
        offset = self._read_offset()
        seen = self._seen.get(objective, frozenset())
        if offset in seen:
            if guard is not None:
                guard.seen = (objective, offset)
            return guard
        # Left searching, it would drive Z to that offset's plane the moment it
        # found the reference. Unknown counts as on (FM-19).
        try:
            on = self._read_engaged()
        except Exception as exc:
            logger.debug(
                "autofocus %s: its state could not be read (%s); switched off",
                label,
                _named(exc),
            )
            on = True
        if on:
            self._executor.safe(self._off, lambda: None)
        unit = f" {self._offset_unit}" if self._offset_unit else ""
        with_objective = "" if objective is None else f" with objective {objective!r}"
        known = ", ".join(f"{value:g}" for value in sorted(seen)) or "none"
        switched = "; it was on and searching, so it was switched off" if on else ""
        raise _EngageRefusedError(
            f"on this stand engaging drives Z to the plane the offset names "
            f"([autofocus] engage_drives_z_to_offset, m2 §12), and offset "
            f"{offset:g}{unit} has not been seen locked{with_objective} since the "
            f"stand was opened (seen: {known}){switched}. Focus on the sample, "
            f"switch {label} on with the stand's own controls and set the offset "
            f"until the sample is sharp; smc then reads the lock",
            label=label,
        )

    def _guard_ended(
        self, guard: _EngageGuard, switch_off_failures: list[Exception]
    ) -> NoReturn:
        """After the guard ended the wait and it was given up on: forget, warn, raise (m2 §12.4).

        The order is decided: the switch-off (the give-up, done by the
        caller), the offset forgotten, the WARNING, the error. A switch-off
        that failed never reads "switched off" (FM-127): it may still move Z,
        and the motion stays registered, as for a lock lost at once.
        """
        label = self._label
        if guard.seen is not None:
            self._forget(*guard.seen)
        what = guard.what_happened()
        if switch_off_failures:
            raise _LostLockError(
                f"autofocus {label}: {what}, and it could not be switched off (the "
                f"switch-off failed: {_named(switch_off_failures[0])}); it may still "
                f"move Z: stop() and check the stand, then resume()"
            ) from switch_off_failures[0]
        z_um = guard.tripped_z_um
        if z_um is None:
            logger.warning(
                "autofocus %s: switched off: Z could not be read while it engaged (%s)",
                label,
                guard.failure,
            )
            raise HardwareError(
                f"autofocus {label}: {what}; it was switched off, since unread it "
                f"could drive Z past any bound (m2 §12)"
            )
        logger.warning(
            "autofocus %s: switched off: Z left its bound while it engaged (%.2f µm)",
            label,
            z_um,
        )
        again = (
            f"; on this stand, lock {label} on the sample at the stand before "
            f"engaging again"
            if self._declared
            else ""
        )
        raise AutofocusExcursionError(
            f"autofocus {label}: {what}; it was switched off. The guard reads Z every "
            f"{POLL_INTERVAL_S * 1000:g} ms: it stops {label} once a jump is read, it "
            f"does not prevent one. Check the sample and the objective{again}",
            guard.z_start_um,
            z_um,
        )

    def disengage(self) -> None:
        """Switch continuous focus off now: a safe call, like closing a shutter (m2 §3.2).

        It never takes the microscope lock and is never refused; it runs while
        halted and in dry-run, since switching off cannot create motion. An
        ``engage`` waiting for its lock is cancelled with ``MotionStoppedError``.

        Raises:
            HardwareError: The slot names another device, or the switch-off
                faulted; a failed switch-off is a finding.
        """
        self._executor.safe(self._off, lambda: None)

    # --- the Z interlock (ZInterlock) --------------------------------------

    def suspend(self, reason: str) -> bool:
        """Switch it off for ``reason`` (a Z move) if it is engaged; whether it was.

        A nested action of the move's compound: skipped in dry-run, unlike
        ``disengage()``, since the move it protects is skipped too. Before the
        switch-off, on a declared stand, it reads the lock, so that the
        operator's lock is seen before the move ends it; and with a band, it
        reads where PFS holds Z, for the relock that follows (m2 §12.4).
        """
        self._z_held = None
        if not self._read_engaged():
            return False
        excursion = self._safety.max_engage_excursion_um
        if (self._declared or excursion is not None) and self._read_locked():
            focus = self._focus()
            if excursion is not None and focus is not None:
                self._z_held = float(focus.position_um())
        description = f"autofocus: suspend for {reason}"
        self._executor.do(
            description,
            Step(
                log=description,
                send=self._send_off,
                readback=lambda: None,
                dry_result=None,
            ),
        )
        return True

    def relock(self, reason: str, *, z_um: float | None) -> None:
        """Engage it again after ``reason``, with the profile's timeout.

        The operator's Z stop wins over it (``_ended_by_a_z_stop``): PFS
        searching for its lock drives Z.

        It runs the engage check and the Z guard, as ``engage()`` does (m2
        §12.4), with the band widened to where PFS held Z before the move.

        Raises:
            AutofocusLostError: No lock, or the engage check refused it (an
                unseen offset on a declared stand): it is off, and Z is at
                ``z_um``.
            AutofocusExcursionError: Z left its bound while it engaged; it was
                switched off.
            MotionStoppedError: ``disengage()`` was called during ``reason``,
                or the focus drive was stopped since ``reason`` was called; it
                is left off, as a close wins over an open.
        """
        z_held, self._z_held = self._z_held, None
        where = "z: moved" if z_um is None else f"z: moved to {z_um:.2f} µm"
        refused = ""
        with self._ended_by_a_z_stop(reason, suspended=True) as stopped:
            self._refuse_relock_after_disengage(reason, stopped)
            limit_s, assumed = self._lock_timeout(None)
            try:
                ended = self._engage(
                    f"autofocus: engage after {reason}",
                    limit_s,
                    assumed,
                    z_held=z_held,
                )
            except _EngageRefusedError as exc:
                refused = exc.detail
                ended = _Lock(False)
        if refused:
            raise AutofocusLostError(
                f"{where}, but autofocus {self._label} was not engaged again: "
                f"{refused}; it is off, or move Z with keep_autofocus=False",
                z_um,
                self._in_range_for_message(),
            )
        if ended.locked:
            return
        raise AutofocusLostError(
            f"{where}, {self._not_locked_again(limit_s, assumed, ended)}, or move Z "
            f"with keep_autofocus=False",
            z_um,
            ended.in_range,
        )

    def _refuse_relock_after_disengage(self, reason: str, stopped: _ZStop) -> None:
        """No relock after a ``disengage()`` made since the compound was called.

        ``disengage()`` is a safe call: it wins over the actions called before
        it, as a close wins over an open (design §13). Engaging again after it
        would end with PFS on after a ``disengage()`` that reported success.
        The relock's own action would refuse too; this names the reason. A Z
        stop switches it off through the same key: its WARNING is
        ``_ended_by_a_z_stop``'s, not this one.
        """
        label = self._label
        try:
            self._executor.refuse_if_stopped(
                label,
                MotionStoppedError(
                    f"autofocus {label}",
                    f"disengage() was called during {reason}, so it was left off",
                ),
            )
        except MotionStoppedError:
            if not stopped.fired:
                logger.warning(
                    "autofocus %s: left off, because disengage() was called during %s",
                    label,
                    reason,
                )
            raise

    def _focus_key(self) -> str | None:
        """The focus drive's key in the ``Executor``, when it is this module's ``MMZStage``.

        A focus drive that ``Microscope.override`` replaced does not stop
        through this ``Executor``, so there is no stop of it to link.
        """
        focus = self._focus()
        return focus._label if isinstance(focus, MMZStage) else None

    def _ended_by_a_z_stop(
        self, description: str, *, suspended: bool, offset_pending: bool = False
    ) -> _EndedByAZStop:
        """Inside a compound: from here on, the operator's Z stop wins (``_EndedByAZStop``)."""
        return _EndedByAZStop(
            self, description, suspended=suspended, offset_pending=offset_pending
        )

    def _on_z_stop(self, stopped: _ZStop) -> None:
        """A ``ZStage.stop()`` during a compound of the autofocus; the stopping thread's.

        It is switched off first, then its offset device is stopped while its
        move is pending or under way, as the emergency stop orders them (m2
        §3.11): a stop of an offset move not sent yet cancels it. Without an
        offset device, the offset move is the autofocus device's own, keyed
        by its label: the switch-off cancels it. Each failure is raised to
        the stopping caller, the switch-off's first. ``finished`` is set as
        soon as the switch-off has returned or raised, before the offset
        stop: the compound reports the switch-off's outcome, and the offset
        stop's goes to the stopping caller. An interrupt in the switch-off
        is recorded as its failure, since whether it went out is then
        unknown, and is re-raised once the offset stop has been sent, as
        ``Executor._stop_linked`` does: the offset device, which PFS follows,
        is stopped past it (FM-62, #166). It takes no lock that a handler
        re-entering this thread could already hold (FM-70), and never the
        microscope lock: the compound waits for it while holding that lock.
        """
        failure: Exception | None = None
        interrupted: BaseException | None = None
        try:
            try:
                self._executor.safe(self._off, lambda: None)
            except Exception as exc:
                stopped.switch_off_failure = failure = exc
            except BaseException as exc:
                stopped.switch_off_failure = interrupted = exc
            finally:
                stopped.finished = True
                # The compound reads it no more: this thread finished last, so
                # it clears the failure, whose traceback holds this frame (#163).
                if stopped.closed:
                    stopped.switch_off_failure = None
            if self._offset_label is not None and stopped.offset_pending:
                try:
                    self._executor.stop(self._offset_motion)
                except Exception as exc:
                    failure = failure or exc
            if interrupted is not None:
                raise interrupted
            if failure is not None:
                raise failure
        finally:
            # Raised through this frame, they are on its own traceback (FM-98).
            del failure, interrupted

    def _z_stop_failure(self, stopped: _ZStop, *, wait: bool) -> BaseException | None:
        """What the Z stop's switch-off raised, once it has returned; waited for, bounded (FM-97).

        A switch-off that has not returned by ``Z_STOP_OUTCOME_WAIT_S`` counts
        as failed: it may never get through, and "left off" would then be a
        claim nobody checked. The wait polls in short sleeps, which a Ctrl-C
        interrupts on Windows too (FM-83). Without ``wait`` (a halt or an
        interrupt ended the call, and goes on at once), the outcome is what
        is known now. A halt that comes during the wait cuts it as well: the
        stand's halted state is read at each poll, because the Z stop has
        already ended the block with a plain ``MotionStoppedError`` by the
        time the emergency stop comes, and the halt switches PFS off itself
        (#166). The read takes no lock (FM-70).
        """
        wait_s = float(Z_STOP_OUTCOME_WAIT_S)  # at call time, so a test can shorten it
        deadline = time.monotonic() + (wait_s if wait else 0.0)
        while not stopped.finished:
            if not wait or self._executor.halted:
                return DeviceTimeoutError("its switch-off had not returned yet")
            if time.monotonic() >= deadline:
                return DeviceTimeoutError(
                    f"its switch-off had not returned after {wait_s:g} s"
                )
            time.sleep(POLL_INTERVAL_S)
        return stopped.switch_off_failure

    def _not_switched_off(
        self, name: str, description: str, failure: BaseException
    ) -> MotionStoppedError:
        return MotionStoppedError(
            name,
            f"it was stopped during {description}, but autofocus {self._label} "
            f"could not be switched off ({_named(failure)}); it may still move Z: "
            f"stop() and check the stand, then resume()",
        )

    def _z_stopped(self, name: str, description: str) -> MotionStoppedError:
        return MotionStoppedError(
            name,
            f"it was stopped during {description}, so autofocus {self._label} was "
            f"left off",
        )

    def _log_left_off(self, description: str, raised: BaseException) -> None:
        """The decided WARNING (m2 §3.4), naming what the caller gets."""
        logger.warning(
            "autofocus %s: left off, because %s did not complete (%s)",
            self._label,
            description,
            type(raised).__name__,
        )

    def _log_not_switched_off(self, description: str, failure: BaseException) -> None:
        logger.warning(
            "autofocus %s: the Z stop during %s could not switch it off (%s); it "
            "may still move Z",
            self._label,
            description,
            _named(failure),
        )

    def _not_locked_again(self, limit_s: float, assumed: bool, ended: _Lock) -> str:
        return (
            f"but autofocus {self._label} did not lock again within {limit_s:g} s"
            f"{_ASSUMED_TIMEOUT if assumed else ''} and is off (in range: "
            f"{_yes_no(ended.in_range)}); engage it (engage()) once the sample is "
            f"in range"
        )

    # --- the offset --------------------------------------------------------

    def _offset_step(self, log: str, value: float) -> Step[float]:
        if self._offset_label is not None:
            label = self._offset_label

            def send() -> None:
                self._offset_core.setPosition(label, value)

        else:
            # Built under the lock, just before the command: in dry-run too.
            self._check_slot()

            def send() -> None:
                self._core.setAutoFocusOffset(value)

        return Step(log=log, send=send, readback=self._read_offset, dry_result=value)

    def _offset_move(self, description: str, step: Callable[[], Step[float]]) -> float:
        timed = self._core if self._offset_label is None else self._offset_core
        return self._executor.do(
            description,
            step,
            motion=self._offset_motion,
            timeout_s=_core_timeout(timed),
        )

    def _check_implied_z(self, target: float, current: float | None = None) -> None:
        """While locked, refuse an offset ``target`` that would drive Z past its soft limits (FM-90).

        Locked, the autofocus follows its offset and drives Z by ``(target -
        current) / factor`` µm, with the profile's measured factor for the
        current objective: that Z gets ``check_z_target_um``'s refusal, which
        no Z move may pass even forced (the review of #140). Without a factor,
        an objective turret or a focus drive, nothing converts the change, so
        the offset's own guards are the only bound: unknown, not guessed. Not
        locked, the offset moves no Z. Run inside the action, before anything
        is sent, and in dry-run too. ``current`` is read here when not given.
        """
        if self._safety.z_soft_limits_um is None:
            return
        focus = self._focus()
        if focus is None or not self._read_locked():
            return
        objective = self._objective_label()
        factor = None if objective is None else self._offset_per_um.get(objective)
        if factor is None:
            return
        here = self._read_offset() if current is None else current
        self._safety.check_z_target_um(focus.position_um() + (target - here) / factor)

    def set_offset_native(self, value: float) -> float:
        """Move the offset to ``value``, in its own units; return the readback.

        Waited for like a stage move. Not jog-guarded, as an absolute Z move is
        not; the soft limits apply and cannot be forced. With the autofocus
        locked, it follows the new offset, which moves Z: with a factor for
        the current objective, the Z it implies is held to the Z soft limits
        (``_check_implied_z``); without one, only the offset's own guards
        bound it. With it on, a ``ZStage.stop()`` ends the move too
        (``_linked_offset_move``).

        Raises:
            SafetyRefusedError: Outside the offset's soft limits, or, locked,
                past the Z soft limits; nothing was sent.
            MotionStoppedError: The offset device was stopped; or, with the
                autofocus on, ``ZStage.stop()`` was called (named ``z
                <label>``): during the move, it was switched off and the
                offset device stopped; while the call waited for the
                microscope lock, nothing was sent and it was left as it was.
                When the Z stop's switch-off failed or had not returned, the
                error says it may still move Z.
            HardwareError: The Microscope it belongs to no longer exists
                (FM-92), or the core's AutoFocus slot names another device
                (FM-87); nothing was read or sent.
        """
        value = float(value)
        # Before the compound: a refused change takes no lock and reads nothing.
        self._safety.check_offset_target_native(value)
        description = self._describe_move_to(value)
        return self._linked_offset_move(
            description, lambda: self._set_offset(description, value)
        )

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        """Move the offset by ``delta`` from where it is read inside the action.

        Jog-guarded unless ``force``; with no limit in the profile, any non-zero
        ``delta`` needs ``force`` (m2 §3.2). The target is checked against the
        soft limits, which cannot be forced, and, locked, the Z it implies
        against the Z soft limits, as for ``set_offset_native``. With the
        autofocus on, a ``ZStage.stop()`` ends the move too, as for
        ``set_offset_native``.

        Raises:
            SafetyRefusedError: The jog is too large (forceable); the target
                is outside the offset's soft limits or, locked, past the Z soft
                limits (not forceable); nothing was sent.
            MotionStoppedError: As for ``set_offset_native``.
            HardwareError: As for ``set_offset_native``.
        """
        delta = float(delta)
        self._safety.check_offset_jog_native(delta, force=force)
        description = self._describe_move_by(delta)
        return self._linked_offset_move(
            description, lambda: self._offset_by(description, delta)
        )

    def _describe_move_to(self, value: float) -> str:
        return f"autofocus_offset: move_to {value} {self._unit()}"

    def _describe_move_by(self, delta: float) -> str:
        return f"autofocus_offset: move_by {delta} {self._unit()}"

    def _set_offset(self, description: str, value: float) -> float:
        """``set_offset_native``'s action, not linked to the Z stop; its caller guarded ``value``."""

        def step() -> Step[float]:
            self._check_implied_z(value)
            return self._offset_step(description, value)

        return self._offset_move(description, step)

    def _offset_by(self, description: str, delta: float) -> float:
        """``offset_by_native``'s action, not linked: ``focus_by_um`` holds its own link.

        A second link, nested in ``focus_by_um``'s, would send the switch-off
        and the offset device's stop twice on one Z stop. Its caller ran the
        jog guard; the target's guards run here, on the anchor read inside
        the action.
        """
        unit = self._unit()

        def relative() -> Step[float]:
            here = self._read_offset()
            value = here + delta
            self._safety.check_offset_target_native(value)
            self._check_implied_z(value, here)
            return self._offset_step(
                f"autofocus_offset: move_by {delta} {unit} to {value} {unit}", value
            )

        return self._offset_move(description, relative)

    def _linked_offset_move(self, description: str, move: Callable[[], float]) -> float:
        """``move``, as one compound, ended by a ``ZStage.stop()`` while PFS is on (m2 §3.3.1).

        On, PFS follows its offset, locked or still searching, and drives Z:
        the operator's Z stop switches it off and stops the offset device. Off,
        the offset moves a lens and no Z (assumed until the PFS session, m2 §4
        item 5), so the Z stop leaves the move alone and nothing claims PFS
        was "left off". The state is read under the microscope lock, where no
        other action can switch PFS on before the move. The focus drive is
        looked up first: without one of this module there is no stop to
        link, so no state is read, and a ``Microscope`` that no longer
        exists refuses the move whatever PFS's state (FM-92).
        """

        def run() -> float:
            if self._focus_key() is None:
                return move()
            if not self._engaged_or_unknown(description):
                return move()
            with self._ended_by_a_z_stop(
                description, suspended=False, offset_pending=True
            ) as stopped:
                landed = move()
                stopped.offset_pending = False
                return landed

        return self._executor.compound(description, run)

    def _engaged_or_unknown(self, description: str) -> bool:
        """Whether it is on before ``description``; a failed read counts as on (FM-19).

        The slot is checked first and refuses (FM-87): a slot that names
        another device is a configuration error, not a transient read
        failure. Only a failed ``isContinuousFocusEnabled()`` is counted as
        on, and labelled as assumed at WARNING. The policy differs from
        ``focus_by_um``'s and ``suspend``'s, which let the same failed read
        raise (m2 §3.3.1): they choose by it what moves (Z or the offset;
        whether PFS is engaged again after a Z move, which drives Z), so a
        guess either way can move the wrong axis. The offset move does the
        same thing whatever the read says; the read only decides whether
        the operator's Z stop also reaches it, and counted as on, the worst
        outcome is a switch-off nobody needed after a stop was asked for.
        """
        self._check_slot()
        try:
            return bool(self._core.isContinuousFocusEnabled())
        except HardwareError as exc:
            logger.warning(
                "autofocus %s: its state could not be read before %s (%s); counted "
                "as on, so a Z stop during it switches it off",
                self._label,
                description,
                # Not the exception: a kept record would keep its traceback,
                # and through it the core (FM-92).
                _named(exc),
            )
            return True

    # --- focus_by_um -------------------------------------------------------

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        """Change focus by ``dz_um`` µm through whatever holds focus now (m2 §3.5).

        One compound. Z's jog guard applies to ``dz_um`` first. Off: a Z move,
        with the autofocus left off. On but not locked: refused. Locked: the
        change goes to the offset, since a Z move would be undone by the
        relock, converted by the profile's factor for the current objective,
        then the lock is waited for again. The unit is always µm: where no
        measured factor exists the call refuses rather than guesses, and
        ``offset_by_native()`` remains for a tool that works in the device's
        units.

        Raises:
            SafetyRefusedError: The jog is too large (forceable); on but not
                locked, or locked with no factor or no objective turret (not
                forceable); the offset's own guards.
            CapabilityMissingError: Off, and the stand has no focus drive.
            AutofocusLostError: The offset moved, but no lock came again.
            MotionStoppedError: Locked, and ``ZStage.stop()`` was called
                during the offset move or the relock: it was switched off, and
                the offset device stopped if it was moving. Before either of
                them, nothing was sent.
        """
        dz_um = float(dz_um)
        self._safety.check_z_jog_um(dz_um, force=force)
        description = f"autofocus: focus_by {dz_um} µm"
        return self._executor.compound(
            description, lambda: self._focus_by(description, dz_um, force)
        )

    def _focus_by(self, description: str, dz_um: float, force: bool) -> FocusStep:
        if not self._read_engaged():
            focus = self._focus()
            if focus is None:
                raise CapabilityMissingError("ZStage", Role.focus)
            landed = focus.move_by_um(dz_um, force=force, keep_autofocus=False)
            return FocusStep("z", landed, None)
        if not self._read_locked():
            raise SafetyRefusedError(
                f"autofocus {self._label} is on but not locked (in range: "
                f"{_yes_no(self._in_range_for_message())}); wait for the lock "
                f"(engage()), or disengage() it to focus with Z"
            )
        factor = self._factor()
        focus = self._focus()
        if focus is not None and self._safety.z_soft_limits_um is not None:
            # PFS follows the offset and drives Z by about dz_um: the Z soft
            # limits, which no Z move may pass even forced, bound it too.
            self._safety.check_z_target_um(focus.position_um() + dz_um)
        delta = dz_um * factor
        # Before the link: a refused jog never touches the Z stop's link.
        self._safety.check_offset_jog_native(delta, force=force)
        with self._ended_by_a_z_stop(
            description, suspended=False, offset_pending=True
        ) as stopped:
            offset = self._offset_by(self._describe_move_by(delta), delta)
            stopped.offset_pending = False
            self._refuse_relock_after_disengage(description, stopped)
            limit_s, assumed = self._lock_timeout(None)
            refused = ""
            try:
                ended = self._engage(
                    f"autofocus: engage after {description}", limit_s, assumed
                )
            except _EngageRefusedError as exc:
                refused = exc.detail
                ended = _Lock(False)
        if refused:
            raise AutofocusLostError(
                f"{Role.autofocus_offset.value}: moved to {offset} {self._unit()}, "
                f"but autofocus {self._label} was not engaged again: {refused}; it "
                f"is off, and while it is off, focus_by_um() moves Z",
                self._z_or_none(),
                self._in_range_for_message(),
            )
        if not ended.locked:
            raise AutofocusLostError(
                f"{Role.autofocus_offset.value}: moved to {offset} {self._unit()}, "
                f"{self._not_locked_again(limit_s, assumed, ended)}; while it is "
                f"off, focus_by_um() moves Z",
                self._z_or_none(),
                ended.in_range,
            )
        if focus is None:
            return FocusStep("offset", None, offset)
        try:
            z_um = focus.position_um()
        except HardwareError as exc:
            # The change was made and locked: a caller that took this for a
            # failure and tried again would apply it twice.
            raise HardwareError(
                f"{description}: the offset moved to {offset} {self._unit()} and "
                f"autofocus {self._label} locked again, but Z could not be read "
                f"afterwards ({exc}); the change is done, do not repeat it"
            ) from exc
        return FocusStep("offset", z_um, offset)

    def _factor(self) -> float:
        """The profile's measured offset change per µm for the current objective, or a refusal."""
        objective = self._objective_label()
        locked = (
            f"autofocus {self._label} is locked, so focus_by_um() would move its "
            f"offset, but"
        )
        instead = "or use offset_by_native() in the offset's own units"
        if objective is None:
            raise SafetyRefusedError(
                f"{locked} no objective turret is resolved, so no [autofocus] "
                f"offset_per_um factor can be chosen; assign the objective_turret "
                f"role under [roles.assign], {instead}"
            )
        factor = self._offset_per_um.get(objective)
        if factor is None:
            raise SafetyRefusedError(
                f"{locked} the profile has no [autofocus] offset_per_um factor for "
                f"objective {objective!r}; measure it with the autofocus locked and "
                f"add it, {instead}"
            )
        return factor

    def _z_or_none(self) -> float | None:
        """The focus drive's position for an error being raised: ``None`` if none, or unreadable."""
        focus = self._focus()
        if focus is None:
            return None
        try:
            return focus.position_um()
        except Exception as exc:
            logger.debug("autofocus %s: Z read failed (%s)", self._label, _named(exc))
            return None


def _magnification_factor(core: CMMCorePlus, for_what: str) -> float | None:
    """The core's magnification factor, or ``None`` (with a warning) if it is not usable.

    The factor is the product of every ``Magnifier`` device (the Ti2-E's
    intermediate-magnification lever is one; FM-13), and MMCore divides its own
    pixel size by it. It passes an invalid factor on (measured: 0, -1.5, NaN and
    infinity all come back as read, and the size reads infinity, -0.667, NaN and
    0.0), so a profile size divided by it would be wrong without a word: such a
    factor is unknown, like a bad profile entry. ``core`` should be the role's
    fault-translating core, so that a device that does not answer is a
    ``HardwareError`` naming the role (FM-79).
    """
    factor = float(core.getMagnificationFactor())
    if math.isfinite(factor) and factor > 0:
        return factor
    logger.warning(
        "the magnification factor (the product of the Magnifier devices) reads "
        "%r, not a positive number; reporting 0.0 (unknown) for %s",
        factor,
        for_what,
    )
    return None


class MMCamera:
    """``Camera`` over the core's current camera.

    MMCore's camera calls act on its *current* camera, not on a label.
    Switching cameras is not a capability in M1, so every call first checks
    that the current camera is still ``label`` and refuses otherwise, rather
    than silently snapping another camera.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        pixel_sizes_um: Mapping[str, float],
        objective_label: Callable[[], str | None],
    ) -> None:
        self._core = as_hardware_errors(core, Role.camera.value)
        self._label = label
        self._executor = executor
        self._pixel_sizes_um = pixel_sizes_um
        self._objective_label = objective_label

    def _check_current(self) -> None:
        current = self._core.getCameraDevice()
        if current != self._label:
            raise HardwareError(
                f"camera {self._label!r} is not the core's current camera "
                f"({current!r}); switching cameras is not supported in M1 — "
                f"set Core-Camera in the configuration or assign the camera "
                f"role in the profile under [roles.assign]."
            )

    def snap(self) -> np.ndarray:
        """One frame, exactly as MMCore returns it (no copy, no flip).

        An acquisition (§13): refused while halted or while a device the layer
        moved is still moving, because a frame taken mid-move is a wrong
        result.

        Unbounded (FM-15): ``snapImage`` has no timeout of its own, and a thread
        inside a driver call cannot be cancelled. It runs under the microscope
        lock, so a hung camera driver holds it; other actions then fail after
        the lock timeout with ``MicroscopeBusyError`` naming "camera: snap"
        instead of hanging, while reads, ``stop()`` and closing the shutter
        still get through.
        """

        def action() -> np.ndarray:
            self._check_current()
            self._core.snapImage()
            return self._core.getImage()

        return self._executor.acquire("camera: snap", action)

    def exposure_ms(self) -> float:
        """The current exposure time."""

        def action() -> float:
            self._check_current()
            return float(self._core.getExposure())

        return self._executor.read(action)

    def set_exposure_ms(self, value_ms: float) -> float:
        """Set the exposure; return the camera's readback (it may clamp).

        Raises:
            ValueError: ``value_ms`` is negative or not finite; it never reaches
                the driver, whose handling of such values is undefined.
        """
        value_ms = float(value_ms)
        if not math.isfinite(value_ms) or value_ms < 0:
            raise ValueError(f"exposure must be finite and >= 0 ms, got {value_ms!r}")
        description = f"camera: exposure {value_ms} ms"

        def step() -> Step[float]:
            # Checked inside the action, in dry-run too.
            self._check_current()
            return Step(
                log=description,
                send=lambda: self._core.setExposure(value_ms),
                readback=lambda: float(self._core.getExposure()),
                dry_result=value_ms,
            )

        return self._executor.do(description, step)

    def image_shape(self) -> tuple[int, int]:
        """``(height, width)`` of the frames ``snap`` returns."""

        def action() -> tuple[int, int]:
            self._check_current()
            return (int(self._core.getImageHeight()), int(self._core.getImageWidth()))

        return self._executor.read(action)

    def bit_depth(self) -> int:
        """Significant bits per pixel."""

        def action() -> int:
            self._check_current()
            return int(self._core.getImageBitDepth())

        return self._executor.read(action)

    def pixel_size_um(self) -> float:
        """MMCore's calibrated value, else the profile's for the objective, else ``0.0``.

        MMCore's value already includes binning and the magnification factor,
        so it is never divided again. The profile map is for binning 1 and a
        factor of 1.0 (m2 §8.10), so the fallback is multiplied by the current
        binning and divided by the factor. A binning that cannot be read, a
        factor that is not a positive number, or a map entry that is not a
        positive finite number, gives ``0.0`` (unknown) with a warning rather
        than a pixel size that looks measured but is wrong. A device that fails
        to answer raises, as every reader does (design §3); the facade's
        ``state()`` is the tolerant layer.
        """

        def action() -> float:
            # MMCore's value is for its current camera and binning; for another
            # camera it would be a measurement of the wrong device. It reads
            # infinity for a factor of 0, which is above 0 and not a size.
            self._check_current()
            measured = float(self._core.getPixelSizeUm())
            if math.isfinite(measured) and measured > 0:
                return measured
            objective = self._objective_label()
            if objective is None or objective not in self._pixel_sizes_um:
                return 0.0
            at_bin_1 = float(self._pixel_sizes_um[objective])
            if not (math.isfinite(at_bin_1) and at_bin_1 > 0):
                logger.warning(
                    "pixel size for objective %r in the profile is %r, not a "
                    "positive number; reporting 0.0 (unknown). Fix it under "
                    "[camera] pixel_size_um in the profile.",
                    objective,
                    at_bin_1,
                )
                return 0.0
            binning = self._binning()
            if binning is None:
                return 0.0
            factor = _magnification_factor(self._core, f"objective {objective!r}")
            if factor is None:
                return 0.0
            return at_bin_1 * binning / factor

        return self._executor.read(action)

    def _binning(self) -> int | None:
        """The camera's square binning factor, or ``None`` (with a warning) if unreadable.

        Adapters spell it ``"2"`` or ``"2x2"``; a non-square binning has no
        single pixel size and counts as unreadable.
        """
        raw = ""
        if self._core.hasProperty(self._label, "Binning"):
            raw = str(self._core.getProperty(self._label, "Binning"))
        parts = raw.lower().replace(" ", "").split("x")
        square = len(parts) <= 2 and len(set(parts)) == 1
        if square and parts[0].isdigit() and int(parts[0]) >= 1:
            return int(parts[0])
        logger.warning(
            "camera %r binning %r cannot be read as one integer factor; the "
            "profile pixel size cannot be scaled, reporting 0.0 (unknown).",
            self._label,
            raw,
        )
        return None


class MMShutter:
    """``Shutter`` over one Micro-Manager shutter device, plus the core's auto-shutter."""

    def __init__(self, core: CMMCorePlus, label: str, executor: Executor) -> None:
        # Rebound, not only stored: the close below captures ``core``, and a
        # close that faults must be translated as well.
        core = self._core = as_hardware_errors(core, Role.shutter.value)
        self._label = label
        self._executor = executor
        self._close = SafeCall(
            device=label,
            log="shutter: close",
            send=lambda: core.setShutterOpen(label, False),
        )

    def _read_open(self) -> bool:
        return bool(self._core.getShutterOpen(self._label))

    def is_open(self) -> bool:
        """Whether the shutter reports open."""
        return self._executor.read(self._read_open)

    def set_open(self, open_: bool) -> bool:
        """Open or close; return the readback.

        Closing is a safe call (§13): it never waits for the microscope lock
        and is never refused, halted, moving or in dry-run, so the light can
        always be cut. Opening is an action, and a close wins over it: a
        close requested while the open waits for the lock cancels it, and one
        requested while the open is being sent is sent again after it. The
        open then raises (``MicroscopeHaltedError`` if halted meanwhile), so
        the shutter never ends open after a close that reported success.
        """
        if not open_:
            return self._executor.safe(self._close, self._read_open)
        return self._executor.do(
            "shutter: open",
            Step(
                log="shutter: open",
                send=lambda: self._core.setShutterOpen(self._label, True),
                readback=self._read_open,
                dry_result=True,
            ),
            overridden_by=self._close,
        )

    def auto_shutter(self) -> bool:
        """Whether the core opens the shutter around each exposure."""
        return self._executor.read(lambda: bool(self._core.getAutoShutter()))

    def set_auto_shutter(self, on: bool) -> bool:
        """Switch auto-shutter; return the readback."""
        on = bool(on)
        description = f"shutter: auto_shutter {'on' if on else 'off'}"
        return self._executor.do(
            description,
            Step(
                log=description,
                send=lambda: self._core.setAutoShutter(on),
                readback=lambda: bool(self._core.getAutoShutter()),
                dry_result=on,
            ),
        )


class MMObjectiveTurret:
    """``ObjectiveTurret`` over the ``objective_turret`` role's State device (m2 §8.3).

    Hardware behaviour encoded here:

    * **Confirmed turns** (FM-14). Turning the nosepiece can drive an
      objective into a loaded plate, so ``select`` refuses without
      ``confirm=True`` while the profile's ``[safety] turret_requires_confirm``
      is true, before it reads or sends anything, in dry-run too.
    * **The autofocus is switched off before a turn, and left off** (FM-88).
      A PFS that follows through a rotation drives Z while no action holds
      the lock, and the new objective's offset differs, so engaging it again
      could drive Z anywhere. The autofocus is looked up through the facade at
      each turn (FM-92), never stored.
    * **A turret cannot be stopped** (a ``State`` device): one still busy at
      the deadline stays registered as moving until it reads idle or
      ``resume()`` is called.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        *,
        requires_confirm: bool,
        pixel_sizes_um: Mapping[str, float],
        autofocus: Callable[[], Autofocus | None] = lambda: None,
    ) -> None:
        self._core = as_hardware_errors(core, Role.objective_turret.value)
        self._label = label
        self._executor = executor
        self._requires_confirm = requires_confirm
        self._pixel_sizes_um = pixel_sizes_um
        self._autofocus = autofocus
        self._motion = _motion(
            self._core, label, f"objective_turret {label}", stoppable=False
        )

    def _read_labels(self) -> list[str]:
        return [str(text) for text in self._core.getStateLabels(self._label)]

    def _read_current(self) -> str:
        return str(self._core.getStateLabel(self._label))

    def labels(self) -> list[str]:
        """Every position's label, in position order, as the configuration names them."""
        return self._executor.read(self._read_labels)

    def current(self) -> str:
        """The label of the objective in the light path (``getStateLabel``)."""
        return self._executor.read(self._read_current)

    def select(self, label: str, *, confirm: bool = False) -> str:
        """Turn the turret to ``label``; return the label it reads afterwards.

        In this order: the confirmation, before anything is read; the label,
        checked against ``labels()``; then one compound, which sends nothing
        when the turret is already at ``label``, switches an engaged autofocus
        off (a state that cannot be read counts as engaged, FM-19), and turns
        the turret, waiting for it like a move. The autofocus is never engaged
        again, and a WARNING says so. In dry-run the compound logs the turn
        and skips it, and switches nothing off.

        Raises:
            SafetyRefusedError: ``confirm`` is false while the profile requires
                it; nothing was read or sent.
            ValueError: ``label`` is not one of the turret's labels; nothing
                was sent.
            HardwareError: The autofocus could not be switched off, so the
                turret was not turned; or the turret faulted; or the
                Microscope it belongs to no longer exists (FM-92).
            DeviceTimeoutError: Still busy at the core timeout; it stays
                registered as moving until it reads idle or ``resume()``.
        """
        if self._requires_confirm and not confirm:
            raise SafetyRefusedError(
                f"turning the objective turret can drive an objective into a "
                f"loaded plate (objective_turret {self._label!r})",
                how_to_force="pass confirm=True once the stage is clear",
            )
        if not self._requires_confirm and not confirm:
            logger.info(
                "objective_turret: no confirmation required ([safety] "
                "turret_requires_confirm = false)"
            )
        labels = self.labels()
        if label not in labels:
            raise ValueError(
                f"objective_turret {self._label!r} has no position {label!r}; its "
                f"labels are: {', '.join(repr(text) for text in labels)}"
            )
        description = f"objective_turret: select {label}"
        return self._executor.compound(
            description, lambda: self._turn(description, label)
        )

    def _turn(self, description: str, label: str) -> str:
        """The compound: nothing if already there, else the switch-off, then the turn."""
        if self._read_current() == label:
            logger.info("objective_turret: already at %s", label)
            return label
        autofocus = self._autofocus()
        if autofocus is not None and not self._executor.dry_run:
            self._switch_off_first(autofocus, label)
        core, turret = self._core, self._label
        return self._executor.do(
            description,
            Step(
                log=description,
                send=lambda: core.setStateLabel(turret, label),
                readback=self._read_current,
                dry_result=label,
            ),
            motion=self._motion,
            timeout_s=_core_timeout(core),
        )

    def _switch_off_first(self, autofocus: Autofocus, label: str) -> None:
        """Switch ``autofocus`` off if it is engaged, or if that cannot be read (FM-19)."""
        name = autofocus._label if isinstance(autofocus, MMAutofocus) else "autofocus"
        try:
            engaged = autofocus.is_engaged()
        except Exception as exc:
            logger.warning(
                "objective_turret: autofocus %r: its state could not be read before "
                "the turret turns (%s); counted as engaged, so it is switched off",
                name,
                _named(exc),
            )
            engaged = True
        if not engaged:
            return
        try:
            autofocus.disengage()
        except Exception as exc:
            raise HardwareError(
                f"objective_turret: not turned to {label!r}: autofocus {name!r} "
                f"could not be switched off first ({_named(exc)}); a turn under an "
                f"engaged autofocus drives Z"
            ) from exc
        logger.warning(
            "objective_turret: autofocus %r switched off before the turret turns; it "
            "is not engaged again (the new objective's offset is not known); "
            "engage() it once in focus",
            name,
        )

    def pixel_size_um(self, label: str | None = None) -> float:
        """The profile's ``[camera] pixel_size_um`` for ``label`` (default: the current one).

        The profile's size is for binning 1 and a magnification factor of 1.0,
        so it is divided by the core's factor (m2 §8.10, #167), for any label:
        the Ti2-E's intermediate-magnification lever does not depend on the
        objective. ``0.0`` (unknown, never guessed) when the profile has no
        size for ``label`` (the factor is not read then), or one that is not a
        positive number, or when the factor is not a positive number. A
        factor that cannot be read raises, as every reader does (design §3).
        """
        objective = self.current() if label is None else label
        size = self._pixel_sizes_um.get(objective)
        if size is None:
            return 0.0
        if not (math.isfinite(size) and size > 0):
            logger.warning(
                "pixel size for objective %r in the profile is %r, not a positive "
                "number; reporting 0.0 (unknown). Fix it under [camera] "
                "pixel_size_um in the profile.",
                objective,
                size,
            )
            return 0.0
        factor = self._executor.read(
            lambda: _magnification_factor(self._core, f"objective {objective!r}")
        )
        if factor is None:
            return 0.0
        return float(size) / factor


#: What a property that sets how bright is called, best first (m2 §8.4).
INTENSITY_HINTS = ("intensity", "brightness", "power", "level", "voltage")


def _normalised(text: str) -> str:
    """Lower-case, ``[a-z0-9]`` only, as role names are matched (design §4)."""
    return "".join(c for c in text.lower() if c.isascii() and c.isalnum())


def _property_info(core: CMMCorePlus, device: str, name: str) -> PropertyInfo:
    """One property with its constraints, read now."""
    has_limits = bool(core.hasPropertyLimits(device, name))
    return PropertyInfo(
        device=device,
        name=name,
        value=str(core.getProperty(device, name)),
        read_only=bool(core.isPropertyReadOnly(device, name)),
        allowed=tuple(str(v) for v in core.getAllowedPropertyValues(device, name)),
        lower=float(core.getPropertyLowerLimit(device, name)) if has_limits else None,
        upper=float(core.getPropertyUpperLimit(device, name)) if has_limits else None,
    )


def _is_shutter(core: CMMCorePlus, label: str) -> bool:
    return device_type_name(core.getDeviceType(label)) == "Shutter"


class MMLightSource:
    """``LightSource`` over the ``light_source`` role's device (m2 §8.4).

    Hardware behaviour encoded here:

    * **The knob is found, never assumed.** The profile's ``[light_source]
      intensity_property`` first, then the device's writable properties named
      like one (``INTENSITY_HINTS``). Only this device is searched: a camera's
      ``BeadBrightness`` is no lamp. The knob's *name* is kept once found; its
      value is read at every call.
    * **Native units, never clamped.** The Ti2-E's ``DiaLamp`` ``Intensity`` is
      a free number in units nobody measured; a value outside a property's
      limits is refused, since a clamp reads as success.
    * **On/off only for a ``Shutter`` device** (the ``DiaLamp``), with the
      kinds of call of ``MMShutter``: off is a safe call keyed by the label,
      so an off from this capability or a close from ``Shutter`` cancels an
      on from either. The demo's ``LED`` is a ``State`` device and has no
      switch that smc knows.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        *,
        intensity_property: str = "",
    ) -> None:
        # Rebound: the switch-off captures the proxied core (as MMShutter).
        core = self._core = as_hardware_errors(core, Role.light_source.value)
        self._label = label
        self._executor = executor
        self._intensity_property = intensity_property
        #: The knob, once resolved: ``(name,)`` or ``(None,)``; ``None`` until then.
        self._knob: tuple[str | None] | None = None
        self._off = SafeCall(
            device=label,
            log="light_source: off",
            send=lambda: core.setShutterOpen(label, False),
        )

    def device(self) -> str:
        """The light source's device label."""
        return self._label

    # --- the knob -------------------------------------------------------------

    def _knob_name(self) -> str | None:
        """The knob's property name, resolved on the first call and kept; ``None``: none."""
        if self._knob is None:
            self._knob = (self._resolve_knob(),)
        return self._knob[0]

    def _resolve_knob(self) -> str | None:
        core, label = self._core, self._label
        names = [str(n) for n in core.getDevicePropertyNames(label)]
        writable = [n for n in names if not core.isPropertyReadOnly(label, n)]
        wanted = self._intensity_property
        if wanted:
            if wanted in writable:
                return wanted
            logger.warning(
                "light_source %r has no writable property %r ([light_source] "
                "intensity_property in the profile); its writable properties are: "
                "%s. Searching by name instead",
                label,
                wanted,
                ", ".join(writable) or "none",
            )
        ranked: list[str] = []
        for hint in INTENSITY_HINTS:
            ranked += [
                n for n in writable if hint in _normalised(n) and n not in ranked
            ]
        with_limits = [n for n in ranked if core.hasPropertyLimits(label, n)]
        if with_limits:
            return with_limits[0]
        for name in ranked:
            if _finite_number(str(core.getProperty(label, name))) is not None:
                return name
        logger.info(
            "light_source %r: no intensity property (searched for %s); set "
            "[light_source] intensity_property if the device has one",
            label,
            ", ".join(INTENSITY_HINTS),
        )
        return None

    def _read_intensity(self, knob: str) -> float:
        value = str(self._core.getProperty(self._label, knob))
        number = _finite_number(value)
        if number is None:
            raise HardwareError(
                f"light_source {self._label!r}: {knob} reads {value!r}, which is not "
                f"a finite number; check the device, or set [light_source] "
                f"intensity_property in the profile"
            )
        return number

    def intensity_knob(self) -> PropertyInfo | None:
        """The property that sets how bright, read now; ``None``: none that smc knows."""

        def read() -> PropertyInfo | None:
            knob = self._knob_name()
            return (
                None if knob is None else _property_info(self._core, self._label, knob)
            )

        return self._executor.read(read)

    def intensity_native(self) -> float | None:
        """The knob's value in the device's own units; ``None`` without a knob.

        Raises:
            HardwareError: The value does not read as a finite number.
        """

        def read() -> float | None:
            knob = self._knob_name()
            return None if knob is None else self._read_intensity(knob)

        return self._executor.read(read)

    def set_intensity_native(self, value: float) -> float:
        """Set the knob to ``value``, in its own units; return the readback.

        Raises, before anything is sent:
            HardwareError: The device has no knob that smc knows.
            ValueError: ``value`` is not finite, or lies outside the knob's
                limits; it is never clamped.
        """
        value = float(value)
        label = self._label
        knob = self._executor.read(self._knob_name)
        if knob is None:
            raise HardwareError(
                f"light_source {label!r} has no intensity property that smc knows; "
                f"set [light_source] intensity_property in the profile, or use "
                f"Properties"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"light_source {label!r}: intensity must be finite, got {value!r}"
            )
        core = self._core
        if core.hasPropertyLimits(label, knob):
            low = float(core.getPropertyLowerLimit(label, knob))
            high = float(core.getPropertyUpperLimit(label, knob))
            if not low <= value <= high:
                raise ValueError(
                    f"light_source {label!r}: intensity {value:g} is outside {knob}'s "
                    f"limits [{low:g}, {high:g}]; it is never clamped"
                )
        description = f"light_source: intensity {value}"
        return self._executor.do(
            description,
            Step(
                log=description,
                send=lambda: core.setProperty(label, knob, value),
                readback=lambda: self._read_intensity(knob),
                dry_result=value,
            ),
        )

    # --- on and off -----------------------------------------------------------

    def _no_switch(self) -> HardwareError:
        kind = device_type_name(self._core.getDeviceType(self._label))
        return HardwareError(
            f"light_source {self._label!r} (a {kind} device) has no on/off that smc "
            f"knows; switch the light with its shutter (Shutter)"
        )

    def _read_on(self) -> bool:
        return bool(self._core.getShutterOpen(self._label))

    def is_on(self) -> bool | None:
        """Whether a ``Shutter``-typed light source reads open; ``None`` for any other type."""

        def read() -> bool | None:
            return self._read_on() if _is_shutter(self._core, self._label) else None

        return self._executor.read(read)

    def set_on(self, on: bool) -> bool:
        """Switch a ``Shutter``-typed light source; return the readback.

        Off is a safe call (design §13): never refused, halted, moving or in
        dry-run. On is an action that an off wins over, as for
        ``Shutter.set_open``.

        Raises:
            HardwareError: The device is not a ``Shutter``; nothing was sent.
        """
        if not _is_shutter(self._core, self._label):
            raise self._no_switch()
        if not on:
            return self._executor.safe(self._off, self._read_on)
        core, label = self._core, self._label
        return self._executor.do(
            "light_source: on",
            Step(
                log="light_source: on",
                send=lambda: core.setShutterOpen(label, True),
                readback=self._read_on,
                dry_result=True,
            ),
            overridden_by=self._off,
        )


def _finite_number(text: str) -> float | None:
    """``text`` as a finite float, or ``None``: ``nan`` is no intensity."""
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` by ``data`` so that it is either the old file or the new one (FM-38).

    A temporary file in the same folder, created with ``O_EXCL`` and the mode
    a plain write gives (``0o666`` less the umask; ``mkstemp``'s 0600 would
    survive the rename), then ``os.replace``, in one ``try``: a Ctrl-C between
    two blocks would leave a file nothing names. On a failure the temporary
    file is removed, except when only the rename failed (Windows refuses to
    replace a file another program holds open): it then holds the whole new
    text, so it is kept and the error names it. Bytes in, bytes out, so the
    CRLF of a Windows ``.cfg`` is written as it was (FM-24).

    Raises:
        HardwareError: The file could not be written; it is as it was.
    """
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.smc-tmp")
    renaming = False
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o666,
        )
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        renaming = True
        os.replace(temporary, path)
    except BaseException as exc:
        if renaming and isinstance(exc, OSError):
            raise HardwareError(
                f"channels: {path} could not be replaced ({exc.strerror or exc}); it "
                f"is as it was, and the new text is kept as {temporary}. Close any "
                f"program that holds {path.name} open, then capture again"
            ) from exc
        with suppress(OSError):
            temporary.unlink()
        if isinstance(exc, OSError):
            raise HardwareError(
                f"channels: {path} could not be written ({exc.strerror or exc}); it "
                f"is as it was"
            ) from exc
        raise


class MMChannels:
    """``Channels`` over one config group of the core (m2 §8.5).

    The pure rules are in ``smc.hardware.channels``; this class reads the
    stand and sends. Hardware behaviour encoded here:

    * **A preset is checked under the lock, at every select** (FM-43): a
      ``setConfig`` that would move a stage, an autofocus or the objective
      turret, or change a core slot, goes around their guards, so it is
      refused and cannot be forced. The settings are read in the step the
      action builds under the lock, so a preset redefined while the select
      waited is the one checked.
    * **One command moves several devices**, and no stop reaches a ``State``
      device: the motion has a key of its own, ``config group <group>``, and
      is busy while any of the preset's devices reads busy.
    * **``current()`` compares numbers as numbers**, not with
      ``getCurrentConfig``, which reads ``''`` once ``Exposure`` reads
      ``10.0000`` for a preset's ``10``.
    * **A capture writes the file first** (the record), as a text edit that
      keeps every other line, then deletes and redefines the preset in the
      running core, since ``defineConfig`` merges into an existing one and
      would keep stale settings (``nikon-control``'s capture did).
      ``saveSystemConfiguration`` is never called: it rewrites the whole file
      from the core's state, comments and the serial ports included.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        group: str,
        executor: Executor,
        *,
        objective_turret: str | None,
        exclude: Sequence[str] = (),
        config_path: Path | None = None,
        light: Callable[[], LightSource | None] = lambda: None,
    ) -> None:
        self._core = as_hardware_errors(core, "channels")
        self._group = group
        self._executor = executor
        self._turret = objective_turret
        self._exclude = frozenset(exclude)
        self._config_path = config_path
        self._light = light

    def group(self) -> str:
        """The config group whose presets are the channels."""
        return self._group

    # --- reads ------------------------------------------------------------------

    def _names(self) -> list[str]:
        return [str(n) for n in self._core.getAvailableConfigs(self._group)]

    def _settings(self, name: str) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (str(device), str(prop), str(value))
            for device, prop, value in self._core.getConfigData(self._group, name)
        )

    def _read_setting(self, device: str, prop: str) -> str:
        """What the stand reads now for one setting; ``Core`` ones through the core's slots."""
        if device == "Core" and prop == "Shutter":
            return str(self._core.getShutterDevice())
        if device == "Core" and prop == "AutoShutter":
            return "1" if self._core.getAutoShutter() else "0"
        return str(self._core.getProperty(device, prop))

    def _mismatches(
        self, settings: Sequence[tuple[str, str, str]], *, first_only: bool = False
    ) -> list[str]:
        """Each setting that reads otherwise, as ``device,property: value read (expected)``."""
        found: list[str] = []
        for device, prop, expected in settings:
            read = self._read_setting(device, prop)
            if not setting_matches(expected, read):
                found.append(f"{device},{prop} reads {read!r} (preset: {expected!r})")
                if first_only:
                    break
        return found

    def _current(self) -> str | None:
        loaded = {str(label) for label in self._core.getLoadedDevices()}
        for name in self._names():
            settings = self._settings(name)
            # A preset that names a device that did not load matches nothing:
            # read, it would fail every current() and every state() (FM-19).
            if any(d != "Core" and d not in loaded for d, _, _ in settings):
                continue
            if not self._mismatches(settings, first_only=True):
                return name
        return None

    def _known(self, name: str) -> list[str]:
        names = self._names()
        if name not in names:
            raise ValueError(
                f"channels: group {self._group!r} has no preset {name!r}; its "
                f"presets are: {', '.join(repr(n) for n in names) or 'none'}"
            )
        return names

    def names(self) -> list[str]:
        """The group's presets as the core lists them; ``[]`` while the group does not exist."""
        return self._executor.read(self._names)

    def current(self) -> str | None:
        """The first preset, in ``names()`` order, whose every setting matches the stand.

        Numbers match as numbers (``10`` and ``10.0000``); ``Core,Shutter`` is
        compared with the core's shutter slot. ``None``: the stand matches no
        preset. A read that fails raises ``HardwareError``.
        """
        return self._executor.read(self._current)

    def preset(self, name: str) -> ChannelPreset:
        """One preset's settings, as the running core holds them.

        Raises:
            ValueError: ``name`` is not one of ``names()``.
        """

        def read() -> ChannelPreset:
            self._known(name)
            return ChannelPreset(self._group, name, self._settings(name))

        return self._executor.read(read)

    # --- select -----------------------------------------------------------------

    def _device_types(self) -> dict[str, str]:
        core = self._core
        return {
            str(label): device_type_name(core.getDeviceType(label))
            for label in core.getLoadedDevices()
            if label != "Core"
        }

    def select(self, name: str) -> str | None:
        """Apply preset ``name``; return the preset the stand matches afterwards.

        One action, waited for like a move until none of the preset's devices
        reads busy, with the core timeout as the deadline; it cannot be
        stopped. The preset is read and checked under the lock. The readback
        is ``name`` when every setting matches what the stand reads, else
        ``current()``, with a WARNING naming each setting that reads otherwise.
        In dry-run it returns ``name`` and sends nothing.

        Raises:
            ValueError: ``name`` is not one of ``names()``; nothing was sent.
            SafetyRefusedError: The preset would move a stage, an autofocus
                or the objective turret, change a core slot, or set a device
                that is not loaded; not forceable, nothing was sent.
            DeviceTimeoutError: A device still read busy at the core timeout;
                the change stays registered as moving until it reads idle or
                ``resume()``.
        """
        # A read: an unknown name takes no lock and sends nothing.
        self._executor.read(lambda: self._known(name))
        group, core = self._group, self._core
        description = f"channels: select {group} {name}"
        # Filled by the step, under the lock: what the motion polls and what the
        # readback found otherwise.
        devices: list[str] = []
        otherwise: list[str] = []

        def step() -> Step[str | None]:
            settings = self._settings(name)
            reasons = forbidden_settings(
                settings,
                device_types=self._device_types(),
                objective_turret=self._turret,
            )
            if reasons:
                raise SafetyRefusedError(
                    f"channel {name!r} of group {group!r} is refused: "
                    f"{'; '.join(reasons)}. Edit the preset in the configuration, "
                    f"or move these through their capability (ZStage, XYStage, "
                    f"Autofocus, ObjectiveTurret)"
                )
            devices[:] = list(dict.fromkeys(d for d, _, _ in settings if d != "Core"))

            def readback() -> str | None:
                otherwise[:] = self._mismatches(settings)
                return name if not otherwise else self._current()

            return Step(
                log=description,
                send=lambda: core.setConfig(group, name),
                readback=readback,
                dry_result=name,
            )

        motion = Motion(
            device=f"config group {group}",
            name=f"channels {group} {name}",
            is_busy=lambda: any(bool(core.deviceBusy(d)) for d in devices),
            stop=None,
        )
        landed = self._executor.do(
            description, step, motion=motion, timeout_s=_core_timeout(core)
        )
        if landed != name:
            logger.warning(
                "channels: %s %s selected, but the stand reads otherwise: %s; it "
                "matches %s",
                group,
                name,
                "; ".join(otherwise),
                "no preset" if landed is None else repr(landed),
            )
        return landed

    # --- capture ----------------------------------------------------------------

    def _capture_settings(self) -> list[tuple[str, str, str]]:
        """What a capture records, in order: State labels, the light's knob, the shutter slot."""
        core = self._core
        settings: list[tuple[str, str, str]] = []
        for label in core.getLoadedDevices():
            if label == "Core" or label == self._turret or label in self._exclude:
                continue
            if device_type_name(core.getDeviceType(label)) != "State":
                continue
            if not core.hasProperty(label, "Label") or core.isPropertyReadOnly(
                label, "Label"
            ):
                continue
            settings.append((str(label), "Label", str(core.getStateLabel(label))))
        light = self._light()
        knob = None if light is None else light.intensity_knob()
        if knob is not None and knob.device not in self._exclude:
            settings.append((knob.device, knob.name, knob.value))
        shutter = str(core.getShutterDevice())
        if shutter:
            settings.append(("Core", "Shutter", shutter))
        return settings

    def capture(self, name: str) -> ChannelPreset:
        """Record the stand's illumination as preset ``name``, replacing one of that name.

        It records each ``State`` device's ``Label`` (not the objective
        turret's, not a ``[channels] exclude`` device's, not a read-only one),
        then the light source's knob, then the core's shutter: which light path
        and how bright, never whether the light is on, the exposure, or a
        stage. The profile's configuration file is written first, atomically
        and as UTF-8 with its line endings kept; then the running core
        (delete, then define). Without a file (the demo's configuration), the
        core only, and an INFO line says so. In dry-run nothing is written.

        Raises:
            ValueError: ``name`` cannot be written to a ``.cfg``.
            HardwareError: A value cannot be written to a ``.cfg``, nothing
                was found to capture, the file is not UTF-8 or cannot be
                written (both unchanged), or the core refused the preset after
                the file was written (the next open loads it from the file).
        """
        check_preset_name(name)
        group = self._group
        description = f"channels: capture {group} {name}"

        def step() -> Step[ChannelPreset]:
            settings = self._capture_settings()
            for setting in settings:
                check_value(*setting)
            if not settings:
                raise HardwareError(
                    f"channels: nothing to capture for {name!r}: no State device, "
                    f"light-source knob or shutter to record"
                )
            return Step(
                log=description,
                send=lambda: self._record(name, settings),
                readback=lambda: self._saved(name),
                dry_result=ChannelPreset(group, name, tuple(settings)),
            )

        return self._executor.do(description, step)

    def _saved(self, name: str) -> ChannelPreset:
        return ChannelPreset(
            self._group, name, self._settings(name), saved_to=self._config_path
        )

    def _record(self, name: str, settings: Sequence[tuple[str, str, str]]) -> None:
        """The capture's send: the file, then the core."""
        path = self._config_path
        if path is None:
            logger.info(
                "channels: %s not saved: the profile loads the demo configuration, "
                "which has no file",
                name,
            )
        else:
            self._write_file(path, name, settings)
        try:
            self._define(name, settings)
        except Exception as exc:
            if path is None:
                raise
            raise HardwareError(
                f"channels: {name!r} was written to {path}, but the running core did "
                f"not take it ({_named(exc)}), and may hold part of it; the file "
                f"holds the preset, and the next open loads it"
            ) from exc

    def _write_file(
        self, path: Path, name: str, settings: Sequence[tuple[str, str, str]]
    ) -> None:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise HardwareError(
                f"channels: {path} could not be read ({exc.strerror or exc}); "
                f"nothing was written"
            ) from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            line = raw.count(b"\n", 0, exc.start) + 1
            raise HardwareError(
                f"channels: {path} is not UTF-8 text (line {line}: {exc.reason}); "
                f"nothing was written. Save it as UTF-8, then capture again"
            ) from exc
        new = replace_preset(text, self._group, name, settings)
        _write_atomically(path, new.encode("utf-8"))

    def _define(self, name: str, settings: Sequence[tuple[str, str, str]]) -> None:
        group, core = self._group, self._core
        old: tuple[tuple[str, str, str], ...] = ()
        if name in self._names():
            old = self._settings(name)
            # defineConfig merges into an existing preset (m2 §8.1).
            core.deleteConfig(group, name)
        for device, prop, value in settings:
            core.defineConfig(group, name, device, prop, value)
        dropped = [s for s in old if s not in settings]
        if dropped:
            # A recapture records only what a channel is made of, so what the
            # preset held besides (shutter states, an exposure) is gone.
            added = [s for s in settings if s not in old]
            logger.warning(
                "channels: %s %s replaced; dropped: %s; added: %s",
                group,
                name,
                "; ".join(",".join(s) for s in dropped),
                "; ".join(",".join(s) for s in added) or "nothing",
            )


class MMProperties:
    """``Properties`` over every loaded device of a core."""

    def __init__(self, core: CMMCorePlus, executor: Executor) -> None:
        # The one class with no role: the message names "properties".
        self._core = as_hardware_errors(core, "properties")
        self._executor = executor

    def devices(self) -> list[str]:
        """Loaded device labels, without the ``Core`` pseudo-device."""
        return self._executor.read(
            lambda: [d for d in self._core.getLoadedDevices() if d != "Core"]
        )

    def describe(self, device: str) -> list[PropertyInfo]:
        """Every property of one device, with its constraints."""
        return self._executor.read(
            lambda: [
                _property_info(self._core, device, n)
                for n in self._core.getDevicePropertyNames(device)
            ]
        )

    def get(self, device: str, name: str) -> str:
        """One property value, as the adapter reports it."""
        return self._executor.read(lambda: str(self._core.getProperty(device, name)))

    def set(self, device: str, name: str, value: str | float | int) -> str:
        """Set one property; return the readback.

        On a ``State`` device (a turret, a filter wheel, a light path) the set
        can start a movement, so it is a motion (§13): it waits inside the
        action for the device to report idle before reading back, with the
        core timeout as the deadline. Such a device cannot be stopped from
        smc, so one still busy at the deadline refuses every action until it
        reports idle or the operator calls ``resume()``. A property of a stage
        is not a motion: moving a stage through ``Properties`` bypasses every
        guard.

        Raises:
            HardwareError: The property is read-only (checked before the call,
                so dry-run refuses it too).
        """
        if self._core.isPropertyReadOnly(device, name):
            raise HardwareError(
                f"{device}.{name} is read-only; it reports state, it cannot be "
                f"set. See `describe({device!r})` for the settable properties."
            )
        description = f"property: {device}.{name}={value}"
        motion = self._motion_for(device)
        return self._executor.do(
            description,
            Step(
                log=description,
                send=lambda: self._core.setProperty(device, name, value),
                readback=lambda: str(self._core.getProperty(device, name)),
                dry_result=str(value),
            ),
            motion=motion,
            timeout_s=_core_timeout(self._core),
        )

    def _motion_for(self, device: str) -> Motion | None:
        """The ``Motion`` a set on ``device`` starts, or ``None`` if it moves nothing we track."""
        from pymmcore_plus import DeviceType

        try:
            kind = DeviceType(self._core.getDeviceType(device))
        except ValueError:
            return None  # a type this pymmcore-plus does not know
        if kind != DeviceType.State:
            return None
        return _motion(self._core, device, device, stoppable=False)
