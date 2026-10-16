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
* **The continuous-focus calls act on a core slot** (FM-87): they take no
  label, so every call first checks that the AutoFocus slot still names the
  autofocus role, as the camera calls do for the camera slot.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from smc.hardware.capabilities import XY, FocusStep, Limits, PropertyInfo
from smc.hardware.errors import (
    AutofocusLostError,
    CapabilityMissingError,
    DeviceTimeoutError,
    HardwareError,
    SafetyRefusedError,
)
from smc.hardware.roles import Role
from smc.hardware.safety import Motion, SafeCall, Step

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
        """Stop the drive now, without waiting for the microscope lock (§13)."""
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


@dataclass(frozen=True, slots=True)
class _Lock:
    """How a lock wait ended: locked, or not, with what the stand said of its range."""

    locked: bool
    in_range: bool | None = None


#: How a lock wait's readback found the autofocus once the wait ended idle.
_AfterWait = Literal["locked", "switched itself off", "lost the lock at once"]


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
        core = self._core = as_hardware_errors(core, Role.autofocus.value)
        self._offset_core = as_hardware_errors(core, Role.autofocus_offset.value)
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
        self._off = SafeCall(
            device=label, log="autofocus: disengage", send=self._send_off
        )
        if offset_label is not None:
            self._offset_motion = _motion(
                self._offset_core,
                offset_label,
                f"{Role.autofocus_offset.value} {offset_label}",
                stoppable=True,
            )
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
        current = self._core.getAutoFocusDevice()
        if current != self._label:
            raise HardwareError(
                f"autofocus {self._label!r} is not the device the core's AutoFocus "
                f"slot names ({current!r}), and the continuous-focus calls act on "
                f"the slot's device; something changed the slot after the stand "
                f"was opened. Open it again, or set Core-AutoFocus in the "
                f"configuration and the autofocus role under [roles.assign] to the "
                f"same device."
            )

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
        self._check_slot()
        self._core.enableContinuousFocus(False)

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

        Raises:
            ValueError: ``timeout_s`` is not finite or not in (0, 600] s;
                nothing was sent.
            DeviceTimeoutError: No lock, and it could not be switched off (or
                still reads on). The wait stays registered as a motion, so the
                stand refuses actions until ``stop()`` or ``resume()``.
            MotionStoppedError: ``disengage()`` or a stop cancelled the wait.
        """
        limit_s, assumed = self._lock_timeout(timeout_s)
        return self._engage("autofocus: engage", limit_s, assumed).locked

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
            if self._read_locked():
                return "locked"
            if not self._read_engaged():
                return "switched itself off"
            # It locked, then lost the lock before this read: left on, it would
            # search, and move Z, with no action holding the lock.
            switch_off()
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
        except DeviceTimeoutError as exc:
            # Only the lock wait raises this here: the Executor gave up at the
            # deadline and sent the stop, which switches it off.
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

        Raises:
            AutofocusLostError: No lock: it is off, and Z is at ``z_um``.
        """
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

    def set_offset_native(self, value: float) -> float:
        """Move the offset to ``value``, in its own units; return the readback.

        Waited for like a stage move. Not jog-guarded, as an absolute Z move is
        not; the soft limits apply and cannot be forced. With the autofocus
        locked, it follows the new offset, which moves Z.
        """
        value = float(value)
        self._safety.check_offset_target_native(value)
        description = f"autofocus_offset: move_to {value} {self._unit()}"
        return self._offset_move(
            description, lambda: self._offset_step(description, value)
        )

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        """Move the offset by ``delta`` from where it is read inside the action.

        Jog-guarded unless ``force``; with no limit in the profile, any non-zero
        ``delta`` needs ``force`` (m2 §3.2). The target is checked against the
        soft limits, which cannot be forced.
        """
        delta = float(delta)
        self._safety.check_offset_jog_native(delta, force=force)
        unit = self._unit()

        def relative() -> Step[float]:
            value = self._read_offset() + delta
            self._safety.check_offset_target_native(value)
            return self._offset_step(
                f"autofocus_offset: move_by {delta} {unit} to {value} {unit}", value
            )

        return self._offset_move(f"autofocus_offset: move_by {delta} {unit}", relative)

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
        offset = self.offset_by_native(dz_um * self._factor(), force=force)
        limit_s, assumed = self._lock_timeout(None)
        ended = self._engage(f"autofocus: engage after {description}", limit_s, assumed)
        if not ended.locked:
            raise AutofocusLostError(
                f"{Role.autofocus_offset.value}: moved to {offset} {self._unit()}, "
                f"{self._not_locked_again(limit_s, assumed, ended)}; while it is "
                f"off, focus_by_um() moves Z",
                self._z_or_none(),
                ended.in_range,
            )
        focus = self._focus()
        return FocusStep(
            "offset", None if focus is None else focus.position_um(), offset
        )

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
