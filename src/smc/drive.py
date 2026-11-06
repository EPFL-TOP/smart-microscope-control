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

from collections.abc import Callable
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
from smc.hardware.errors import HardwareError
from smc.hardware.microscope import autofocus_known, describe_autofocus

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "AutofocusReading",
    "DriveResult",
    "disengage_autofocus",
    "engage_autofocus",
    "jog_xy",
    "jog_z",
    "move_xy",
    "move_z",
    "read_autofocus",
    "read_autofocus_offset",
    "set_autofocus_offset",
    "set_objective",
    "switch_light",
    "um_text",
    "xy_text",
    "z_text",
]

#: What a switch-off says in dry run: the facade sends it all the same (design §13),
#: and the operator must not believe the stand was left as it was.
_OFF_IN_DRY_RUN = " (dry run: switching off is sent all the same)"

#: How far Z must differ across a turn before the text says the stand moved it.
#: Chosen: the read noise at rest is 0.04 µm and the corrections measured are
#: 8.7 µm (Ti2-E session 2, m2 §8.11 D7).
_Z_MOVED_UM = 0.5


@dataclass(frozen=True, slots=True)
class DriveResult:
    """What a drive action did, as the stand reads now.

    ``reached`` is ``False`` when the call returned but the stand is not where it
    was asked to be: the command prints ``text`` as an error and exits 1, the page
    shows it as the outcome.
    """

    text: str
    reached: bool = True


@dataclass(frozen=True, slots=True)
class AutofocusReading:
    """What one read of the autofocus says, and what a decision may be made from.

    ``engaged`` is the state's field (``None`` when the read failed) and ``known``
    is ``autofocus_known`` of the same state; ``text`` is for the operator and is
    never compared (FM-155).
    """

    text: str
    engaged: bool | None
    known: bool


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
    return _moved(
        microscope, xy_text(landed), lambda: xy_text(stage.position_um()), "stage"
    )


def move_xy(microscope: Microscope, x_um: float, y_um: float) -> DriveResult:
    """Move the XY stage to ``(x_um, y_um)``; the position it returns, or what dry run would do.

    ``smc stage move``'s call (#295). Only the profile's soft limits apply: an
    absolute target has no jog limit to refuse, and the page has no such control
    (its own absolute move is ``goto-well``).

    Raises:
        CapabilityMissingError: The stand has no XY stage.
        SafetyRefusedError: The target is outside the soft limits.
        MicroscopeHaltedError: The stand is halted.
    """
    stage = microscope.require(XYStage)
    landed = stage.move_to_um(x_um, y_um)
    return _moved(
        microscope, xy_text(landed), lambda: xy_text(stage.position_um()), "stage"
    )


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
    return _moved(
        microscope, z_text(landed), lambda: z_text(drive.position_um()), "drive"
    )


def move_z(microscope: Microscope, z_um: float) -> DriveResult:
    """Move the focus drive to ``z_um``; the position it returns, or what dry run would do.

    ``smc z move``'s call (#295); the interlock and the soft limits are the
    facade's, as for a jog.

    Raises:
        CapabilityMissingError: The stand has no focus drive.
        SafetyRefusedError: The target is outside the soft limits.
        MicroscopeHaltedError: The stand is halted.
    """
    drive = microscope.require(ZStage)
    landed = drive.move_to_um(z_um)
    return _moved(
        microscope, z_text(landed), lambda: z_text(drive.position_um()), "drive"
    )


def _moved(
    microscope: Microscope, landed: str, here: Callable[[], str], mover: str
) -> DriveResult:
    """The text of the four moves (``jog_*``, ``move_*``): where it landed, or what dry run would do.

    In dry run the move returned the target it would command, so the drive or the
    stage (``mover``) is read again, through ``here``, to show that it did not move;
    outside dry run nothing is read twice.
    """
    if microscope.dry_run:
        return DriveResult(
            f"[dry-run] would move to {landed}; the {mover} is at {here()}"
        )
    return DriveResult(landed)


def _read_z(zstage: ZStage) -> float | None:
    """Z now, or ``None`` when the read fails: a turn that was sent is not undone by it.

    The turret's own WARNING has the cause, and nothing here keeps the
    exception (FM-98).
    """
    try:
        return float(zstage.position_um())
    except HardwareError:
        return None


def _z_clause(before: float | None, after: float | None) -> str:
    """What the text says about Z across a turn: a move, nothing, or that it was not read."""
    if before is None or after is None:
        return "; Z could not be read around the turn"
    if abs(after - before) > _Z_MOVED_UM:
        return (
            f"; the stand moved Z from {um_text(before)} to {um_text(after)} µm "
            f"({after - before:+.2f} µm)"
        )
    return ""


def set_objective(microscope: Microscope, label: str, *, confirm: bool) -> DriveResult:
    """Turn the turret to ``label``; the label it reads afterwards (FM-14, FM-125).

    The facade checks the confirmation first, then the label, so a refused turn
    sends nothing. Z is read before the turn, as a read, and again after it (the
    turret waits until Z is still): the Ti2-E moves Z during a turn, so the text
    says what the stand did to it (m2 §8.11 D7). A stand with no focus drive, and
    a dry run, say nothing about Z.

    Raises:
        CapabilityMissingError: The stand has no objective turret.
        SafetyRefusedError: ``confirm`` is false while the profile requires it.
        ValueError: ``label`` is not one of the turret's labels.
        MicroscopeHaltedError: The stand is halted.
    """
    turret = microscope.require(ObjectiveTurret)
    zstage = None if microscope.dry_run else microscope.get(ZStage)
    before = None if zstage is None else _read_z(zstage)
    turret.select(label, confirm=confirm)
    current = turret.current()
    if microscope.dry_run:
        return DriveResult(
            f"[dry-run] would turn to {label!r}; the turret is at {current!r}"
        )
    z_clause = "" if zstage is None else _z_clause(before, _read_z(zstage))
    if current != label:
        return DriveResult(
            f"objective {current!r}, although {label!r} was asked{z_clause}",
            reached=False,
        )
    return DriveResult(f"objective {current!r}{z_clause}")


def read_autofocus(microscope: Microscope) -> AutofocusReading:
    """One fresh read of the autofocus: its words, and the fields a decision is made from.

    ``state()`` keeps a failed read as an error and the words say only ``autofocus
    ?``: when the state is not ``known`` the causes follow in brackets, once each and
    each on one line (a driver's message may span several, and an outcome is one
    line, FM-98), so the operator gets a finding and not a question mark (CLAUDE.md:
    failures are findings). A caller decides from ``engaged`` and ``known``, never
    from ``text`` (FM-155).
    """
    state = microscope.state()
    known = autofocus_known(state)
    text = describe_autofocus(state)
    if not known:
        causes = [
            " ".join(error.split())
            for error in state.errors
            if error.startswith("autofocus:")
        ]
        if causes:
            text += f" ({'; '.join(dict.fromkeys(causes))})"
    return AutofocusReading(text, state.autofocus_engaged, known)


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
            f"[dry-run] would engage the autofocus; {read_autofocus(microscope).text}"
        )
    if locked:
        return DriveResult("autofocus locked")
    # ``False`` means off by the capability's contract, but a search left running
    # drives Z: what the stand reads is said, never what the contract promises
    # (FM-155).
    now = read_autofocus(microscope)
    if now.engaged is False:
        return DriveResult("autofocus did not lock; it is off", reached=False)
    return DriveResult(f"autofocus did not lock; {now.text}", reached=False)


def disengage_autofocus(microscope: Microscope) -> DriveResult:
    """Switch the autofocus off: never refused, and sent in dry run too; what the stand reads after.

    Raises:
        CapabilityMissingError: The stand has no autofocus.
    """
    microscope.require(Autofocus).disengage()
    now = read_autofocus(microscope)
    # Switching off is sent in dry run, so what the stand reads after it is a
    # readback in both modes, and an autofocus still on is not reached in either.
    reached = now.engaged is False
    if microscope.dry_run:
        return DriveResult(now.text + _OFF_IN_DRY_RUN, reached=reached)
    return DriveResult(now.text, reached=reached)


def _offset_text(autofocus: Autofocus, value: float) -> str:
    """``5000 PFSOffset units``: the offset in its own units, never converted."""
    return f"{value:g} {autofocus.offset_unit() or 'device units'}"


def read_autofocus_offset(microscope: Microscope) -> DriveResult:
    """The autofocus offset and its soft limits, whose lower bound is a first lock's floor (m2 §12.10).

    Reads only: ``smc autofocus offset`` with no option.

    Raises:
        CapabilityMissingError: The stand has no autofocus.
    """
    autofocus = microscope.require(Autofocus)
    here = _offset_text(autofocus, autofocus.offset_native())
    limits = autofocus.offset_limits_native()
    if limits is None:
        return DriveResult(
            f"offset {here}, no soft limits in the profile (no floor: no first lock)"
        )
    return DriveResult(f"offset {here}, soft limits [{limits[0]:g}, {limits[1]:g}]")


def set_autofocus_offset(microscope: Microscope, value: float | None) -> DriveResult:
    """Set the autofocus offset; ``None``: the floor, the lower bound of the offset's soft limits.

    Step 2 of a first lock from below (m2 §12.10 D7): with the autofocus off the
    write moves no Z. One mutating call (m5 §6.1): the soft limits and the
    offset are reads. The text is the readback, and one that differs from the
    value asked says so with ``reached`` false (FM-155): the Ti2-E holds whole
    units, so 7927.5 honestly reads 7927.

    Raises:
        CapabilityMissingError: The stand has no autofocus.
        SafetyRefusedError: Outside the offset's soft limits; or, on a stand that
            declares ``engage_drives_z_to_offset``, the autofocus is on and not
            locked with a factor (m2 §12.10 D6).
        MicroscopeHaltedError: The stand is halted.
    """
    autofocus = microscope.require(Autofocus)
    if value is None:
        limits = autofocus.offset_limits_native()
        if limits is None:
            return DriveResult(
                "no floor: [safety] autofocus_offset_soft_limits is not set (m2 "
                "§12.10)",
                reached=False,
            )
        value = limits[0]
    landed = autofocus.set_offset_native(value)
    if microscope.dry_run:
        return DriveResult(
            f"[dry-run] would set the offset to {_offset_text(autofocus, value)}; it "
            f"is at {_offset_text(autofocus, autofocus.offset_native())}"
        )
    if landed != value:
        return DriveResult(
            f"offset {_offset_text(autofocus, landed)}, although {value:g} was asked",
            reached=False,
        )
    return DriveResult(f"offset {_offset_text(autofocus, landed)}")


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
