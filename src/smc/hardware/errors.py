"""Error hierarchy for the hardware layer (design: docs/design/m1-hardware-layer.md §2).

Every error carries what to do next — the CLI flag, the profile key, the
issue to read — because a microscope error at 2 a.m. is read by someone who
did not write this code.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smc.hardware.roles import Role


def join_names(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``: names in a message."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


class HardwareError(RuntimeError):
    """Base class: something in `smc.hardware` could not do what was asked."""


class CoreError(HardwareError):
    """Micro-Manager could not be found, or a configuration could not be loaded."""


class ProfileError(HardwareError):
    """A profile file is missing, unreadable or invalid; the message names file and key."""


class DeviceTimeoutError(HardwareError):
    """A device stayed busy past its deadline."""


class CapabilityMissingError(HardwareError):
    """No loaded device fills the role a capability needs.

    ``hint``, when set, replaces the ``[roles.assign]`` advice: ``Channels``
    needs a config group, not a role (m2 §8.6), and ``available`` then lists
    the core's groups.
    """

    def __init__(
        self,
        capability: str,
        role: Role | None,
        available: tuple[str, ...] = (),
        *,
        hint: str = "",
    ) -> None:
        self.capability = capability
        self.role = role
        self.available = available
        self.hint = hint
        have = ", ".join(available) or "nothing"
        if hint:
            message = f"{capability} is not available: {hint} (available: {have})."
        else:
            where = f"the {role.value!r} role" if role is not None else "any role"
            message = (
                f"{capability} needs {where}, which no loaded device fills "
                f"(available roles: {have}). Assign one in the profile under "
                f"[roles.assign], or check `smc devices` for candidates."
            )
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuilt from the arguments, not the message, so the error survives
        # a worker process (pickle); the hint is a keyword.
        return (
            functools.partial(type(self), hint=self.hint),
            (self.capability, self.role, self.available),
        )


class SafetyRefusedError(HardwareError):
    """A guard refused a move. ``how_to_force`` is empty when it cannot be forced."""

    def __init__(self, reason: str, how_to_force: str = "") -> None:
        self.reason = reason
        self.how_to_force = how_to_force
        tail = f" — {how_to_force}" if how_to_force else ""
        super().__init__(f"{reason}{tail}")


class MotionInProgressError(SafetyRefusedError):
    """A mutation or an acquisition was refused because a device is still moving (§13).

    It cannot be forced: acting on a microscope whose movement has not
    finished is what the rule forbids. ``detail`` says who is moving it, or
    what could not be read for a device whose busy state raised.
    """

    def __init__(self, moving: tuple[str, ...], detail: str = "") -> None:
        self.moving = moving
        self.detail = detail
        names = ", ".join(f"`{name}`" for name in moving)
        verb, it = ("is", "it") if len(moving) == 1 else ("are", "them")
        # resume() is the way out for a device that keeps answering busy or
        # cannot be read, including a State device that cannot be stopped.
        reason = (
            f"{names} {verb} still moving; wait for {it} (`wait()`) or stop "
            f"{it} (`stop()`), or, if {it} never settles, check the stand and "
            f"call `resume()`"
        )
        if detail:
            reason += f" ({detail})"
        super().__init__(reason)

    def __reduce__(self) -> tuple[type[MotionInProgressError], tuple[object, ...]]:
        # Rebuilt from the constructor's arguments, not the message, so the
        # error survives a worker process (pickle).
        return (type(self), (self.moving, self.detail))


class MicroscopeHaltedError(HardwareError):
    """An action was refused because the microscope is halted (§13).

    Not a ``SafetyRefusedError``: a plugin that skips a target on ``except
    SafetyRefusedError`` (the soft limits) must not swallow the emergency
    stop.
    """

    def __init__(self) -> None:
        super().__init__(
            "the microscope was stopped (`Microscope.stop()`); call `resume()` "
            "to continue"
        )

    def __reduce__(self) -> tuple[type[MicroscopeHaltedError], tuple[object, ...]]:
        return (type(self), ())


class MotionStoppedError(HardwareError):
    """A move ended because it was stopped, not because it arrived (§13).

    A plugin loop that sees it must end instead of carrying on to the next
    well: the stage is not where the loop believes it is. ``device`` is the
    motion's name (``"xy_stage XY"``).
    """

    def __init__(
        self, device: str, detail: str = "the move was stopped before it completed"
    ) -> None:
        self.device = device
        self.detail = detail
        super().__init__(f"`{device}`: {detail}; read the position before moving on")

    def __reduce__(self) -> tuple[type[MotionStoppedError], tuple[object, ...]]:
        return (type(self), (self.device, self.detail))


class StopRefusedError(HardwareError):
    """A stop the device refused, as the profile declares ([safety] stop_refused_by).

    Not a fault: nothing is broken, and a move under way runs to its target.
    The Ti2-E answers every ``stop()`` of its XY stage, Z drive and PFS offset
    with ``Unsupported device command (11)``, still or moving, and a move that
    was sent went all the way (m2 §11.1). Not a ``SafetyRefusedError`` either
    (FM-64): a tool that skips a target on a refusal must not read a move that
    is running as one that was never sent. ``refused`` holds the motions'
    names (``"xy_stage XYStage"``).
    """

    def __init__(self, refused: tuple[str, ...], message: str = "") -> None:
        self.refused = refused
        super().__init__(
            message
            or (
                f"{join_names([f'`{name}`' for name in refused])} refused the stop, "
                f"as the profile declares ([safety] stop_refused_by): a move under "
                f"way runs to its target; stop it with the stand's own controls if "
                f"it must not finish"
            )
        )

    def __reduce__(self) -> tuple[type[StopRefusedError], tuple[object, ...]]:
        # Rebuilt from the constructor's arguments, so the error survives a
        # worker process (pickle).
        return (type(self), (self.refused, str(self)))


class StopOutcome(str, Enum):
    """What a stop sent on giving up on a motion did (m2 §11.4)."""

    accepted = "accepted"  # the device took the stop
    refused = "refused"  # the device raised: a move under way may run to its target
    impossible = "impossible"  # no stop from smc (a State device)
    interrupted = "interrupted"  # a second Ctrl-C cut the stop: it may not have reached the device


@dataclass(frozen=True, slots=True)
class StopNote:
    """One give-up's stop, noted on the exception that ended the wait (m2 §11.4).

    Strings, an enum and a bool: a note holds no exception, so it cannot keep
    a traceback, and the core with it (FM-98).
    """

    motion: str  # "xy_stage XYStage"
    outcome: StopOutcome
    declared: bool  # the motion's stop_refused


#: The attribute the notes ride on. A tuple, replaced whole by ``note_stop``.
_NOTES = "_smc_stop_notes"


def note_stop(exc: BaseException, note: StopNote) -> None:
    """Append ``note`` to the notes ``exc`` carries.

    It runs on the path a Ctrl-C re-enters (``Executor._move``'s ``except``),
    so it takes no lock and does no I/O: it reads one attribute, builds one
    tuple and stores it (FM-70). An exception that takes no attribute keeps
    no note, and it still propagates: the note never replaces it.
    """
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, _NOTES, (*getattr(exc, _NOTES, ()), note))


def stop_notes(exc: BaseException) -> tuple[StopNote, ...]:
    """The notes of ``exc`` and of its ``__context__`` chain, oldest first.

    ``__context__``, not ``__cause__``: a second Ctrl-C inside a give-up's
    stop raises a new ``KeyboardInterrupt`` whose context is the first, and
    the two notes are on the two exceptions. It stops at an exception it has
    already seen, so a chain that loops ends.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__context__
    return tuple(note for link in reversed(chain) for note in getattr(link, _NOTES, ()))


class AutofocusLostError(HardwareError):
    """A move under the autofocus completed, but the autofocus did not lock again (m2 §3.10).

    It is then off: a Z move on a Nikon stand switches PFS off (FM-12), and a
    relock that gives up switches it off too, so that it cannot drive Z with
    no action holding the lock. Raising is the point: a Z move that silently
    leaves PFS off is the "PFS off overnight" ADR-0003 names. ``z_um`` is
    where the focus drive was read after the move (``None`` without one) and
    ``in_range`` what the stand said of its reference (``None``: unknown).

    Not a ``SafetyRefusedError`` (FM-64): the move was made, so a tool that
    skips a target on a refusal must not carry on as if nothing moved.
    """

    def __init__(self, message: str, z_um: float | None, in_range: bool | None) -> None:
        self.z_um = z_um
        self.in_range = in_range
        super().__init__(message)

    def __reduce__(self) -> tuple[type[AutofocusLostError], tuple[object, ...]]:
        return (type(self), (str(self), self.z_um, self.in_range))


class AutofocusExcursionError(HardwareError):
    """Z left its bound while the autofocus engaged, and it was switched off (m2 §12.5).

    On the Ti2-E, engaging PFS drives Z to the plane its stored offset names,
    and run 1 jumped +235 µm in under 100 ms (#205). The guard reads Z while
    the lock wait and the settle window last: it stops PFS once a jump is
    read, it does not prevent one. ``z_start_um`` is the Z read before the
    switch-on, ``z_um`` the read that tripped the guard.

    Not a ``SafetyRefusedError`` (FM-64): something moved, so a tool that
    skips a target on a refusal must not carry on as if nothing had.
    """

    def __init__(self, message: str, z_start_um: float, z_um: float) -> None:
        self.z_start_um = z_start_um
        self.z_um = z_um
        super().__init__(message)

    def __reduce__(self) -> tuple[type[AutofocusExcursionError], tuple[object, ...]]:
        return (type(self), (str(self), self.z_start_um, self.z_um))


class MicroscopeBusyError(HardwareError):
    """The microscope lock could not be taken in time (§13, FM-15).

    ``holder`` is the description of the call holding the lock and
    ``held_s`` how long it had held it when this caller gave up. When nothing
    recorded when the holder took the lock (it was taken outside the
    ``Executor``), ``held_s`` is this caller's own wait and ``at_least`` is
    true: an assumed figure is labelled, not reported as a measurement.
    """

    def __init__(self, holder: str, held_s: float, at_least: bool = False) -> None:
        self.holder = holder
        self.held_s = held_s
        self.at_least = at_least
        bound = "at least " if at_least else ""
        super().__init__(
            f"{holder} has held the microscope for {bound}{held_s:.1f} s; the "
            f"device driver may be hung. If the call is legitimately long, "
            f"raise [micromanager] device_timeout_ms in the profile; otherwise "
            f"restart the device or the program."
        )

    def __reduce__(self) -> tuple[type[MicroscopeBusyError], tuple[object, ...]]:
        return (type(self), (self.holder, self.held_s, self.at_least))
