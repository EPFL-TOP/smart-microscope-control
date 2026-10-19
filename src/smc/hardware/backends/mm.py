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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from smc.hardware.capabilities import XY, FocusStep, Limits, PropertyInfo
from smc.hardware.errors import (
    AutofocusLostError,
    CapabilityMissingError,
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.roles import Role
from smc.hardware.safety import Motion, SafeCall, Step, StopLink

if TYPE_CHECKING:
    import numpy as np
    from pymmcore_plus import CMMCorePlus

    from smc.hardware.capabilities import ZStage
    from smc.hardware.safety import Executor, Safety

__all__ = [
    "DEFAULT_LOCK_TIMEOUT_S",
    "MAX_LOCK_TIMEOUT_S",
    "MMAutofocus",
    "MMCamera",
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
#: The longest lock wait ``engage`` accepts: the wait holds the microscope
#: lock, so every other action waits behind it (FM-32).
MAX_LOCK_TIMEOUT_S = 600.0

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
        detail = str(exc) or type(exc).__name__
        message = (
            f"{self._role}: {name}({shown}) failed ({detail}); check the device: "
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


def _motion(core: CMMCorePlus, label: str, name: str, *, stoppable: bool) -> Motion:
    def stop() -> None:
        core.stop(label)

    return Motion(
        device=label,
        name=name,
        is_busy=lambda: bool(core.deviceBusy(label)),
        stop=stop if stoppable else None,
    )


def autofocus_offset_motion(core: CMMCorePlus, label: str) -> Motion:
    """The ``autofocus_offset`` device's move, as the ``Executor`` sees it (m2 §3.6).

    One definition for ``MMAutofocus``, which waits for it, and for the
    emergency stop, which stops that device by its role (m2 §3.11): the same
    key, so that the stop cancels the move under way, and the same name in
    the lines.
    """
    role = Role.autofocus_offset.value
    return _motion(
        as_hardware_errors(core, role), label, f"{role} {label}", stoppable=True
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
        self._motion = _motion(self._core, label, f"xy_stage {label}", stoppable=True)

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
        self._motion = _motion(self._core, label, f"z {label}", stoppable=True)
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
    #: What the Z stop's switch-off raised: it may still be on.
    switch_off_failure: Exception | None = None

    @property
    def fired(self) -> bool:
        """Whether a Z stop reached the link: set as that stop took its snapshot."""
        return self.link is not None and self.link.reached


class _LostLockError(DeviceTimeoutError):
    """It locked, lost the lock at once, and could not be switched off.

    A ``DeviceTimeoutError`` for every caller; its own class only so that
    ``engage`` does not take it for its lock wait's deadline.
    """


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
    ) -> None:
        if lock_timeout_s is not None:
            lock_timeout_s = _checked_lock_timeout(lock_timeout_s, "lock_timeout_s")
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
            else autofocus_offset_motion(core, offset_label)
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

    def _read_locked(self) -> bool:
        self._check_slot()
        return bool(self._core.isContinuousFocusLocked())

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
        except Exception:
            logger.debug(
                "autofocus %s: in-range read failed", self._label, exc_info=True
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
        """The lock wait's limit, and whether it is the assumed default."""
        if timeout_s is not None:
            return _checked_lock_timeout(timeout_s, "timeout_s"), False
        if self._lock_timeout_s is not None:
            return self._lock_timeout_s, False
        # The module's value at call time, so that a test can shorten it.
        return float(DEFAULT_LOCK_TIMEOUT_S), True

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
                cancelled the call before anything was sent.
            HardwareError: Its state could not be read once the wait ended. It
                was given up on as at the deadline: switched off, or still
                registered as a motion if that failed too.
        """
        limit_s, assumed = self._lock_timeout(timeout_s)
        description = "autofocus: engage"

        def run() -> bool:
            with self._ended_by_a_z_stop(description, suspended=False):
                return self._engage(description, limit_s, assumed).locked

        # Its own compound, so that the link and the check for a Z stop made
        # while it waited for the lock have a record to compare against.
        return self._executor.compound(description, run)

    def _engage(self, description: str, limit_s: float, assumed: bool) -> _Lock:
        """``engage``'s action, also run nested by the relocks; how the wait ended."""
        label = self._label
        switch_off_failures: list[Exception] = []

        def switch_off() -> None:
            # The lock wait's stop: kept, so that a failure can be named.
            try:
                self._send_off()
            except Exception as exc:
                switch_off_failures.append(exc)
                raise

        def after_wait() -> _AfterWait:
            # The wait ended idle: locked, or no longer on.
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
            self._check_slot()  # in dry-run too
            return Step(
                log=description,
                send=self._send_on,
                readback=after_wait,
                dry_result="locked",
            )

        motion = Motion(
            device=label,
            name=f"autofocus {label}",
            is_busy=self._searching,
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
            # up at the deadline and sent the stop, which switches it off.
            return self._no_lock(limit_s, assumed, exc, switch_off_failures)
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
    ) -> _Lock:
        """After the lock wait gave up: ``False`` if it is off, else the timeout, re-raised."""
        label = self._label
        try:
            still_on = self._read_engaged()
        except Exception as exc:
            outcome = f"its state cannot be read: {exc}"
        else:
            if not still_on and not switch_off_failures:
                in_range = self._in_range_for_message()
                logger.warning(
                    "autofocus %s: no lock within %g s%s; in range: %s; switched off",
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
        ``disengage()``, since the move it protects is skipped too.
        """
        if not self._read_engaged():
            return False
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

        Raises:
            AutofocusLostError: No lock: it is off, and Z is at ``z_um``.
            MotionStoppedError: ``disengage()`` was called during ``reason``,
                or the focus drive was stopped since ``reason`` was called; it
                is left off, as a close wins over an open.
        """
        with self._ended_by_a_z_stop(reason, suspended=True) as stopped:
            self._refuse_relock_after_disengage(reason, stopped)
            limit_s, assumed = self._lock_timeout(None)
            ended = self._engage(f"autofocus: engage after {reason}", limit_s, assumed)
        if ended.locked:
            return
        where = "z: moved" if z_um is None else f"z: moved to {z_um:.2f} µm"
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

    @contextmanager
    def _ended_by_a_z_stop(
        self, description: str, *, suspended: bool, offset_pending: bool = False
    ) -> Iterator[_ZStop]:
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

        When the Z stop's switch-off failed, it may still be on: no "left
        off" is claimed. The block's own error goes on (a lock wait that
        then gave up says what it found), or one naming the failure.

        While halted, the emergency stop switches the autofocus off and stops
        the offset device itself, in its own order (m2 §3.11): the Z stop
        then reaches no link, and the block ends on what the halt did.
        Without a focus drive of this module, no Z stop is linked.
        """
        stopped = _ZStop(offset_pending=offset_pending)
        key = self._focus_key()
        if key is None:
            yield stopped
            return
        name = f"z {key}"
        before = MotionStoppedError(name, "it was stopped before it started")
        try:
            try:
                with self._executor.linked_stop(
                    key, lambda: self._on_z_stop(stopped)
                ) as link:
                    stopped.link = link
                    self._executor.refuse_if_stopped(key, before)
                    yield stopped
            except BaseException as exc:
                # Suspended, a refusal before the engage was sent (this check, or a
                # halt, #141) leaves it off as surely as a Z stop does.
                refused = exc is before or isinstance(exc, MicroscopeHaltedError)
                if not (stopped.fired or (suspended and refused)):
                    raise
                if stopped.switch_off_failure is not None:
                    self._log_not_switched_off(description, stopped.switch_off_failure)
                    raise
                if isinstance(exc, MicroscopeHaltedError) or not isinstance(
                    exc, Exception
                ):
                    self._log_left_off(description, exc)
                    raise  # a halt stays a halt, a Ctrl-C a Ctrl-C
                error = self._z_stopped(name, description)
                self._log_left_off(description, error)
                raise error from exc
            if stopped.fired:
                # It locked, then the stop switched it off: not a success.
                failure = stopped.switch_off_failure
                if failure is not None:
                    self._log_not_switched_off(description, failure)
                    raise MotionStoppedError(
                        name,
                        f"it was stopped during {description}, but autofocus "
                        f"{self._label} could not be switched off ({failure}); it "
                        f"may still move Z: stop() and check the stand, then "
                        f"resume()",
                    ) from failure
                error = self._z_stopped(name, description)
                self._log_left_off(description, error)
                raise error
        finally:
            # The link's callback holds `stopped`, and `stopped` held the link: a
            # cycle that kept this autofocus and the core alive until the cyclic
            # collector (FM-92). The callback never reads `link`, a stop that is
            # still on its way needs only `stopped`, and nothing reads `fired`
            # after this block.
            stopped.link = None

    def _on_z_stop(self, stopped: _ZStop) -> None:
        """A ``ZStage.stop()`` during a compound of the autofocus; the stopping thread's.

        It is switched off first, then its offset device is stopped while its
        move is pending or under way, as the emergency stop orders them (m2
        §3.11): a stop of an offset move not sent yet cancels it. Without an
        offset device, the offset move is the autofocus device's own, keyed
        by its label: the switch-off cancels it. Each failure is raised to
        the stopping caller, the switch-off's first.
        """
        failure: Exception | None = None
        try:
            self._executor.safe(self._off, lambda: None)
        except Exception as exc:
            stopped.switch_off_failure = failure = exc
        if self._offset_label is not None and stopped.offset_pending:
            try:
                self._executor.stop(self._offset_motion)
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

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

    def _log_not_switched_off(self, description: str, failure: Exception) -> None:
        logger.warning(
            "autofocus %s: the Z stop during %s could not switch it off (%s); it "
            "may still move Z",
            self._label,
            description,
            failure,
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
        """
        value = float(value)
        # Before the compound: a refused change takes no lock and reads nothing.
        self._safety.check_offset_target_native(value)
        return self._linked_offset_move(
            f"autofocus_offset: move_to {value} {self._unit()}",
            lambda: self._set_offset(value),
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
        """
        delta = float(delta)
        self._safety.check_offset_jog_native(delta, force=force)
        return self._linked_offset_move(
            f"autofocus_offset: move_by {delta} {self._unit()}",
            lambda: self._offset_by(delta, force=force),
        )

    def _set_offset(self, value: float) -> float:
        """``set_offset_native``'s action, not linked to the Z stop."""
        self._safety.check_offset_target_native(value)
        description = f"autofocus_offset: move_to {value} {self._unit()}"

        def step() -> Step[float]:
            self._check_implied_z(value)
            return self._offset_step(description, value)

        return self._offset_move(description, step)

    def _offset_by(self, delta: float, *, force: bool) -> float:
        """``offset_by_native``'s action, not linked: ``focus_by_um`` holds its own link.

        A second link, nested in ``focus_by_um``'s, would send the switch-off
        and the offset device's stop twice on one Z stop.
        """
        self._safety.check_offset_jog_native(delta, force=force)
        unit = self._unit()

        def relative() -> Step[float]:
            here = self._read_offset()
            value = here + delta
            self._safety.check_offset_target_native(value)
            self._check_implied_z(value, here)
            return self._offset_step(
                f"autofocus_offset: move_by {delta} {unit} to {value} {unit}", value
            )

        return self._offset_move(f"autofocus_offset: move_by {delta} {unit}", relative)

    def _linked_offset_move(self, description: str, move: Callable[[], float]) -> float:
        """``move``, as one compound, ended by a ``ZStage.stop()`` while PFS is on (m2 §3.3.1).

        On, PFS follows its offset, locked or still searching, and drives Z:
        the operator's Z stop switches it off and stops the offset device. Off,
        the offset moves a lens and no Z (assumed until the PFS session, m2 §4
        item 5), so the Z stop leaves the move alone and nothing claims PFS
        was "left off". The state is read under the microscope lock, where no
        other action can switch PFS on before the move.
        """

        def run() -> float:
            if not self._engaged_or_unknown():
                return move()
            with self._ended_by_a_z_stop(
                description, suspended=False, offset_pending=True
            ) as stopped:
                landed = move()
                stopped.offset_pending = False
                return landed

        return self._executor.compound(description, run)

    def _engaged_or_unknown(self) -> bool:
        """Whether it is on; unknown counts as on (FM-19), since on, it may drive Z."""
        try:
            return self._read_engaged()
        except HardwareError:
            logger.debug(
                "autofocus %s: state unreadable before an offset move; counted as on",
                self._label,
                exc_info=True,
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
        with self._ended_by_a_z_stop(
            description, suspended=False, offset_pending=True
        ) as stopped:
            offset = self._offset_by(dz_um * factor, force=force)
            stopped.offset_pending = False
            self._refuse_relock_after_disengage(description, stopped)
            limit_s, assumed = self._lock_timeout(None)
            ended = self._engage(
                f"autofocus: engage after {description}", limit_s, assumed
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
        except Exception:
            logger.debug("autofocus %s: Z read failed", self._label, exc_info=True)
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

        MMCore's value already includes binning; the profile map is for
        binning 1, so the fallback is multiplied by the current binning. A
        binning that cannot be read, or a map entry that is not a positive
        finite number, gives ``0.0`` (unknown) with a warning rather than a
        pixel size that looks measured but is wrong. A device that fails to
        answer raises, as every reader does (design §3); the facade's
        ``state()`` is the tolerant layer.
        """

        def action() -> float:
            # MMCore's value is for its current camera and binning; for another
            # camera it would be a measurement of the wrong device.
            self._check_current()
            measured = float(self._core.getPixelSizeUm())
            if measured > 0:
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
            return at_bin_1 * binning

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

    def _info(self, device: str, name: str) -> PropertyInfo:
        core = self._core
        has_limits = bool(core.hasPropertyLimits(device, name))
        return PropertyInfo(
            device=device,
            name=name,
            value=str(core.getProperty(device, name)),
            read_only=bool(core.isPropertyReadOnly(device, name)),
            allowed=tuple(str(v) for v in core.getAllowedPropertyValues(device, name)),
            lower=float(core.getPropertyLowerLimit(device, name))
            if has_limits
            else None,
            upper=float(core.getPropertyUpperLimit(device, name))
            if has_limits
            else None,
        )

    def describe(self, device: str) -> list[PropertyInfo]:
        """Every property of one device, with its constraints."""
        return self._executor.read(
            lambda: [
                self._info(device, n) for n in self._core.getDevicePropertyNames(device)
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
