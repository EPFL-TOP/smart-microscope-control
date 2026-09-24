"""Error hierarchy for the hardware layer (design: docs/design/m1-hardware-layer.md §2).

Every error carries what to do next — the CLI flag, the profile key, the
issue to read — because a microscope error at 2 a.m. is read by someone who
did not write this code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smc.hardware.roles import Role


class HardwareError(RuntimeError):
    """Base class: something in `smc.hardware` could not do what was asked."""


class CoreError(HardwareError):
    """Micro-Manager could not be found, or a configuration could not be loaded."""


class ProfileError(HardwareError):
    """A profile file is missing, unreadable or invalid; the message names file and key."""


class DeviceTimeoutError(HardwareError):
    """A device stayed busy past its deadline."""


class CapabilityMissingError(HardwareError):
    """No loaded device fills the role a capability needs."""

    def __init__(
        self,
        capability: str,
        role: Role | None,
        available: tuple[str, ...] = (),
    ) -> None:
        self.capability = capability
        self.role = role
        self.available = available
        have = ", ".join(available) or "nothing"
        where = f"the {role.value!r} role" if role is not None else "any role"
        super().__init__(
            f"{capability} needs {where}, which no loaded device fills "
            f"(available roles: {have}). Assign one in the profile under "
            f"[roles.assign], or check `smc devices` for candidates."
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
    finished is what the rule forbids. ``detail`` carries what could not be
    read, for a device whose busy state raised.
    """

    def __init__(self, moving: tuple[str, ...], detail: str = "") -> None:
        self.moving = moving
        self.detail = detail
        names = ", ".join(f"`{name}`" for name in moving)
        verb, it = ("is", "it") if len(moving) == 1 else ("are", "them")
        reason = (
            f"{names} {verb} still moving; wait for {it} (`wait()`) or stop "
            f"{it} (`stop()`)"
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


class MicroscopeBusyError(HardwareError):
    """The microscope lock could not be taken in time (§13, FM-15).

    ``holder`` is the description of the call holding the lock and
    ``held_s`` how long it had held it when this caller gave up.
    """

    def __reduce__(self) -> tuple[type[MicroscopeBusyError], tuple[object, ...]]:
        return (type(self), (self.holder, self.held_s))

    def __init__(self, holder: str, held_s: float) -> None:
        self.holder = holder
        self.held_s = held_s
        super().__init__(
            f"{holder} has held the microscope for {held_s:.1f} s; the device "
            f"driver may be hung. If the call is legitimately long, raise "
            f"[micromanager] device_timeout_ms in the profile; otherwise "
            f"restart the device or the program."
        )
