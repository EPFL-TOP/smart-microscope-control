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
