"""The drive actions, written once for the command line and the page (m5 §6.1, ADR-0006).

Principle 5 of ADR-0006: nothing exists only in the UI. ``smc stage jog`` and
the page's ``→`` are the same call, so each action is here, and both the CLI
and the plate page's controller call it. Each one is ``require`` and **one**
capability call, through the facade's guards (the jog limit, the Z↔autofocus
interlock, the turret's confirmation, dry run); nothing here adds or skips a
guard, and the stand is reached only through ``smc.hardware``.

Every text is read back from the stand after the call, never built from what was
asked (FM-155, FM-167): a turret that stops short of its label, an autofocus that
does not lock and a light whose readback disagrees say so, with ``reached`` false.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from smc.hardware.capabilities import (
    XY,
    Autofocus,
    LightSource,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.microscope import describe_autofocus

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "DriveResult",
    "disengage_autofocus",
    "engage_autofocus",
    "jog_xy",
    "jog_z",
    "set_objective",
    "switch_light",
    "um_text",
    "xy_text",
    "z_text",
]

#: What a switch-off says in dry run: the facade sends it all the same (design §13),
#: and the operator must not believe the stand was left as it was.
_OFF_IN_DRY_RUN = " (dry run: switching off is sent all the same)"


@dataclass(frozen=True, slots=True)
class DriveResult:
    """What a drive action did, as the stand reads now.

    ``reached`` is ``False`` when the call returned but the stand is not where it
    was asked to be: the command prints ``text`` as an error and exits 1, the page
    shows it as the outcome.
    """

    text: str
    reached: bool = True


def um_text(value: float) -> str:
    """Two decimals; a stage that reads ``-0.0`` at the origin shows ``0.00``."""
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def xy_text(position: XY) -> str:
    """``XY (x, y) µm``, as ``smc stage`` prints it."""
    return f"XY ({um_text(position.x_um)}, {um_text(position.y_um)}) µm"


def z_text(z_um: float) -> str:
    """``Z z µm``, as ``smc z`` prints it."""
    return f"Z {um_text(z_um)} µm"


def jog_xy(
    microscope: Microscope, dx_um: float, dy_um: float, *, force: bool = False
) -> DriveResult:
    """Move the XY stage by ``(dx_um, dy_um)``; the position it returns, or what dry run would do.

    Raises:
        CapabilityMissingError: The stand has no XY stage.
        SafetyRefusedError: The jog exceeds ``max_jog_um`` (unless ``force``) or
            leaves the soft limits.
        MicroscopeHaltedError: The stand is halted.
    """
    stage = microscope.require(XYStage)
    landed = stage.move_by_um(dx_um, dy_um, force=force)
    if microscope.dry_run:
        # The move returned the target it would command; the stage is read again
        # to show that it did not move.
        here = stage.position_um()
        return DriveResult(
            f"[dry-run] would move to {xy_text(landed)}; the stage is at {xy_text(here)}"
        )
    return DriveResult(xy_text(landed))


def jog_z(microscope: Microscope, dz_um: float, *, force: bool = False) -> DriveResult:
    """Move the focus drive by ``dz_um``; the interlock suspends and relocks an engaged autofocus (FM-12).

    Raises:
        CapabilityMissingError: The stand has no focus drive.
        SafetyRefusedError: The jog exceeds ``max_z_jog_um`` (unless ``force``) or
            leaves the soft limits.
        MicroscopeHaltedError: The stand is halted.
    """
    drive = microscope.require(ZStage)
    landed = drive.move_by_um(dz_um, force=force)
    if microscope.dry_run:
        here = drive.position_um()
        return DriveResult(
            f"[dry-run] would move to {z_text(landed)}; the drive is at {z_text(here)}"
        )
    return DriveResult(z_text(landed))


def set_objective(microscope: Microscope, label: str, *, confirm: bool) -> DriveResult:
    """Turn the turret to ``label``; the label it reads afterwards (FM-14, FM-125).

    The facade checks the confirmation first, then the label, so a turn without
    ``confirm`` reads nothing and sends nothing.

    Raises:
        CapabilityMissingError: The stand has no objective turret.
        SafetyRefusedError: ``confirm`` is false while the profile requires it.
        ValueError: ``label`` is not one of the turret's labels.
        MicroscopeHaltedError: The stand is halted.
    """
    turret = microscope.require(ObjectiveTurret)
    turret.select(label, confirm=confirm)
    current = turret.current()
    if microscope.dry_run:
        return DriveResult(
            f"[dry-run] would turn to {label!r}; the turret is at {current!r}"
        )
    if current != label:
        return DriveResult(
            f"objective {current!r}, although {label!r} was asked", reached=False
        )
    return DriveResult(f"objective {current!r}")


def engage_autofocus(microscope: Microscope) -> DriveResult:
    """Engage the autofocus with the profile's lock timeout; ``reached`` only when it locked.

    Raises:
        CapabilityMissingError: The stand has no autofocus.
        SafetyRefusedError: The stand engages only at an offset seen locked with
            this objective, and none was.
        MicroscopeHaltedError: The stand is halted.
    """
    locked = microscope.require(Autofocus).engage()
    if microscope.dry_run:
        return DriveResult(
            f"[dry-run] would engage the autofocus; "
            f"{describe_autofocus(microscope.state())}"
        )
    if locked:
        return DriveResult("autofocus locked")
    return DriveResult("autofocus did not lock; it is off", reached=False)


def disengage_autofocus(microscope: Microscope) -> DriveResult:
    """Switch the autofocus off: never refused, and sent in dry run too; what the stand reads after.

    Raises:
        CapabilityMissingError: The stand has no autofocus.
    """
    microscope.require(Autofocus).disengage()
    words = describe_autofocus(microscope.state())
    # Switching off is sent in dry run, so what the stand reads after it is a
    # readback in both modes, and an autofocus still on is not reached in either.
    reached = words == "autofocus off"
    if microscope.dry_run:
        return DriveResult(words + _OFF_IN_DRY_RUN, reached=reached)
    return DriveResult(words, reached=reached)


def switch_light(microscope: Microscope, on: bool) -> DriveResult:
    """Switch the light source on or off; what it reads after the call (FM-47).

    Off is never refused and is sent in dry run; on, in dry run, is not sent.

    Raises:
        CapabilityMissingError: The stand has no light source.
        HardwareError: The light source has no on/off that smc knows (a
            ``State`` device); the message says to use its shutter.
        MicroscopeHaltedError: ``on`` while the stand is halted.
    """
    light = microscope.require(LightSource)
    light.set_on(on)
    now = _light_word(light.is_on())
    if on and microscope.dry_run:
        return DriveResult(f"[dry-run] would switch the light on; it is {now}")
    asked = "on" if on else "off"
    reached = now == asked
    text = f"light {now}" if reached else f"light {now}, although {asked} was sent"
    if microscope.dry_run:
        return DriveResult(text + _OFF_IN_DRY_RUN, reached=reached)
    return DriveResult(text, reached=reached)


def _light_word(on: bool | None) -> str:
    """``on``, ``off`` or ``?`` (``None``: no switch that smc knows)."""
    if on is None:
        return "?"
    return "on" if on else "off"
