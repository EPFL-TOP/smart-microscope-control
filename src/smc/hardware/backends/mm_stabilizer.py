"""``MMFocusStabilizer``: a one-shot focus stabilizer as an ``Autofocus`` (m4 §11.6).

Vendor-neutral: it knows the five properties of m4 §11.4 (``Action``,
``Reference``, ``OperationMode``, ``PiezoFocusUsage``, ``Periodic``) and
nothing of MTB, so any Micro-Manager ``GenericDevice`` that offers them is a
focus stabilizer. Zeiss Definite Focus 2 (``MTBDefiniteFocus``) is the first.

Hardware behaviour encoded here:

* **One shot only.** DF2 holds a distance from the coverslip and corrects
  through the focus drive on the Observer 7 (m4 §11.1). smc never switches
  its periodic mode on: ``engage()`` and a non-zero offset are
  ``AutofocusUnsupportedError``, with nothing sent.
* **Refused before anything is sent**: no reference, a reference of another
  device or through another objective (FM-223), an empty one outside a dry
  run, and DF2 set to correct through the piezo, which no role reads: the Z
  readback would not see the correction.
* **The guard reads Z after a stabilize, it cannot stop one.** A Z change
  beyond ``[safety] max_engage_excursion_um`` or outside the Z soft limits is
  an ``AutofocusExcursionError``, and Z is left where DF2 put it. How far one
  ``StabilizeNow`` moves Z is not measured (m4 §12).
* **What DF2 holds is what the device reports.** A stabilize restores the
  reference first unless the device's ``Reference`` already reads it: several
  facades may share one core, and one's memory is not what DF2 holds. A
  learn or a restore that fails leaves no reference held.
* **A switch-off is a safe call** (``executor.safe``): never refused, run
  while halted and in dry-run, outside the microscope lock, so a ``Stop``
  never waits behind the stabilize it ends.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from smc.hardware.backends.mm import _core_timeout, _named, as_hardware_errors
from smc.hardware.capabilities import FocusReference, FocusStabilized, FocusStep
from smc.hardware.errors import (
    AutofocusExcursionError,
    AutofocusUnsupportedError,
    CapabilityMissingError,
    HardwareError,
    SafetyRefusedError,
)
from smc.hardware.roles import Role
from smc.hardware.safety import Motion, SafeCall, Step

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from smc.hardware.capabilities import ZStage
    from smc.hardware.safety import Executor, Safety

__all__ = ["MMFocusStabilizer"]

logger = logging.getLogger("smc.hardware.backends.mm_stabilizer")

#: The properties of a focus stabilizer, and their values (m4 §11.4).
ACTION, REFERENCE = "Action", "Reference"
PIEZO_FOCUS_USAGE, PERIODIC = "PiezoFocusUsage", "Periodic"
IDLE, LEARN, STABILIZE_NOW, STOP = "Idle", "Learn", "StabilizeNow", "Stop"
ON, OFF = "On", "Off"


def _busy(core: CMMCorePlus, label: str) -> bool:
    return bool(core.deviceBusy(label))


def _send_stop(core: CMMCorePlus, label: str) -> None:
    """The motion's stop: ``Action`` set to ``Stop``, which the device sends in the set."""
    core.setProperty(label, ACTION, STOP)


def _switch_off(core: CMMCorePlus, label: str) -> None:
    """Stop what it does, then its periodic mode off; both tried, failures raised together.

    A read that fails counts as running, or as on (FM-19): a ``Stop`` or an
    ``Off`` that was not needed is harmless, one that was skipped is not. A
    function of the core and the label, not a method: the ``SafeCall`` is
    stored on the stabilizer, and one that held a bound method would be a
    reference cycle through the core (#142).
    """
    failures: list[str] = []
    for prop, idle, value in ((ACTION, IDLE, STOP), (PERIODIC, OFF, OFF)):
        try:
            try:
                needed = str(core.getProperty(label, prop)) != idle
            except Exception:
                needed = True
            if needed:
                core.setProperty(label, prop, value)
        except Exception as exc:
            failures.append(f"{prop} {value}: {_named(exc)}")
    if failures:
        raise HardwareError(
            f"autofocus {label}: the switch-off failed ({'; '.join(failures)}); it "
            f"may still be acting on the focus drive: check it at the stand"
        )


class MMFocusStabilizer:
    """``Autofocus`` over a focus stabilizer's properties, one shot at a time; the ``ZInterlock``.

    ``objective_label`` and ``focus`` are the facade's, as ``MMAutofocus``
    takes them: the objective is read live (``getStateLabel``, never the
    cached ``Label``, pymmcore-plus#629), and the focus drive is looked up at
    the call.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        safety: Safety,
        *,
        objective_label: Callable[[], str | None],
        focus: Callable[[], ZStage | None],
    ) -> None:
        core = self._core = as_hardware_errors(core, Role.autofocus.value)
        self._label = label
        self._executor = executor
        self._safety = safety
        self._objective_label = objective_label
        self._focus = focus
        #: The reference last learned or restored, for ``stabilize_now()``
        #: without one; ``None`` after one that failed. Replaced whole.
        self._held: FocusReference | None = None
        # Partials of module functions, never bound methods (#142).
        self._motion = Motion(
            device=label,
            name=f"autofocus {label}",
            is_busy=functools.partial(_busy, core, label),
            stop=functools.partial(_send_stop, core, label),
            stop_refused=False,
        )
        self._off = SafeCall(
            device=label,
            log="autofocus: disengage",
            send=functools.partial(_switch_off, core, label),
        )

    def __str__(self) -> str:
        """How the Z stage's log lines name it (``ZInterlock``)."""
        return f"autofocus {self._label}"

    def _read(self, prop: str) -> str:
        return str(self._core.getProperty(self._label, prop))

    def _unsupported(self, call: str) -> AutofocusUnsupportedError:
        return AutofocusUnsupportedError(
            f"autofocus {self._label}: {call} is not supported: it holds focus one "
            f"shot at a time: learn_reference(), stabilize_now(); nothing was sent"
        )

    # --- reads -----------------------------------------------------------------

    def is_engaged(self) -> bool:
        """Whether its periodic mode is on: smc never switches it on (m4 §11.2)."""
        return self._executor.read(lambda: self._read(PERIODIC) == ON)

    def is_locked(self) -> bool:
        """``is_engaged()``: smc reads no lock flag of a one-shot stabilizer."""
        return self.is_engaged()

    def is_in_range(self) -> bool | None:
        """``None``: the stand does not say (``SignalQuality`` reads 0 while DF2 works)."""
        return None

    # --- no offset: zero by construction -----------------------------------------

    def offset_native(self) -> float:
        """``0.0``: smc sets no offset on a reference, so none is assumed."""
        return 0.0

    def offset_unit(self) -> str:
        """``"µm"``: the offset is zero, in any unit."""
        return "µm"

    def offset_limits_native(self) -> tuple[float, float] | None:
        """``(0.0, 0.0)``: the only offset it holds."""
        return (0.0, 0.0)

    def set_offset_native(self, value: float) -> float:
        """``0.0`` sends nothing; any other value is refused (m4 §11.6)."""
        if float(value) == 0.0:
            return 0.0
        raise self._unsupported(f"set_offset_native({value!r})")

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        """``0.0`` sends nothing; any other change is refused (m4 §11.6)."""
        if float(delta) == 0.0:
            return 0.0
        raise self._unsupported(f"offset_by_native({delta!r})")

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        """A Z move, as on an autofocus that is off; the Z jog guard applies.

        Raises:
            CapabilityMissingError: The stand has no focus drive.
        """
        focus = self._focus()
        if focus is None:
            raise CapabilityMissingError("ZStage", Role.focus)
        return FocusStep("z", focus.move_by_um(dz_um, force=force), None)

    # --- switching ---------------------------------------------------------------

    def engage(self, timeout_s: float | None = None) -> bool:
        """Refused: it holds focus one shot at a time (m4 §11.6).

        Raises:
            AutofocusUnsupportedError: Always, before anything is sent.
        """
        raise self._unsupported("engage()")

    def disengage(self) -> None:
        """Stop what it does, then its periodic mode off: a safe call, never refused.

        It runs while halted and in dry-run, outside the microscope lock, so
        a stabilize that holds the lock is stopped, not waited for; that
        action then raises.

        Raises:
            HardwareError: The ``Stop`` or the switch-off failed; both were tried.
        """
        self._executor.safe(self._off, lambda: None)

    # --- one shot ------------------------------------------------------------------

    def learn_reference(self) -> FocusReference:
        """Record the plane in focus now, with the objective in place; nothing moves.

        One action whose motion is the learn. The reference is held from then
        on; a learn that fails leaves none held, since what the device holds
        is then unknown. In dry-run nothing is sent and its ``data`` is ``""``.

        Raises:
            HardwareError: The learn failed, or ended with no reference.
        """
        label = self._label
        description = f"autofocus: learn the reference of {label}"

        def step() -> Step[FocusReference]:
            objective = self._objective_label()
            return Step(
                log=description,
                send=functools.partial(self._core.setProperty, label, ACTION, LEARN),
                readback=lambda: self._learned(objective),
                dry_result=FocusReference(label, "", objective),
            )

        self._held = None
        reference = self._executor.do(
            description,
            step,
            motion=self._motion,
            timeout_s=_core_timeout(self._core),
        )
        self._held = reference
        return reference

    def _learned(self, objective: str | None) -> FocusReference:
        data = self._read(REFERENCE)
        if not data:
            raise HardwareError(
                f"autofocus {self._label}: the learn ended without an error, but no "
                f"reference reads back; none is held. Learn again, and check that a "
                f"coverslip is in the light path"
            )
        return FocusReference(self._label, data, objective)

    def stabilize_now(self, reference: FocusReference | None = None) -> FocusStabilized:
        """Bring focus to ``reference`` (else the one held) once, and leave Z there.

        One compound: the refusals, Z read, a restore when the device does not
        hold ``reference`` already, the stabilize, Z read back, then the Z
        guard. A ``disengage()`` during it sends ``Stop`` and the call raises.

        Raises:
            SafetyRefusedError: No reference given or held, one of another
                device, an empty one outside dry-run, one learned through
                another objective, or ``PiezoFocusUsage`` on; nothing was sent,
                and no ``force`` passes it.
            AutofocusExcursionError: Z moved beyond the profile's bound; it is
                left there.
            HardwareError: The restore or the stabilize failed (the device
                answered ``False``, or a ``Stop`` ended it); a failed restore
                leaves no reference held.
        """
        description = f"autofocus: stabilize_now {self._label}"
        return self._executor.compound(
            description, lambda: self._stabilize(description, reference)
        )

    def _stabilize(
        self, description: str, given: FocusReference | None
    ) -> FocusStabilized:
        reference = self._checked(given)
        focus = self._focus()
        z_before = None if focus is None else float(focus.position_um())
        if self._read(REFERENCE) != reference.data:
            self._restore(reference)
        self._held = reference
        z_um = self._executor.do(
            description,
            Step(
                log=description,
                send=functools.partial(
                    self._core.setProperty, self._label, ACTION, STABILIZE_NOW
                ),
                readback=lambda: None if focus is None else float(focus.position_um()),
                dry_result=z_before,
            ),
            motion=self._motion,
            timeout_s=_core_timeout(self._core),
        )
        if not self._executor.dry_run:
            self._judge(z_before, z_um)
        return FocusStabilized(reference, z_before, z_um)

    def _checked(self, given: FocusReference | None) -> FocusReference:
        """The reference to bring focus to, or a refusal before anything is sent (m4 §11.6)."""
        label = self._label
        reference = given if given is not None else self._held
        if reference is None:
            raise SafetyRefusedError(
                f"autofocus {label}: no reference was given and none is held; "
                f"learn_reference() first, or pass one; nothing was sent"
            )
        if reference.device != label:
            raise SafetyRefusedError(
                f"autofocus {label}: the reference was learned by "
                f"{reference.device!r}, not by this autofocus; nothing was sent"
            )
        if not reference.data and not self._executor.dry_run:
            raise SafetyRefusedError(
                f"autofocus {label}: the reference is empty, as a dry run learns it; "
                f"learn it outside the dry run; nothing was sent"
            )
        objective = self._objective_label()
        if (
            reference.objective is not None
            and objective is not None
            and reference.objective != objective
        ):
            # FM-223: the plane was learned through another objective.
            raise SafetyRefusedError(
                f"autofocus {label}: the reference was learned through "
                f"{reference.objective!r}, and {objective!r} is in place, so it "
                f"would bring focus to another objective's plane; turn back to "
                f"{reference.objective!r}, or learn a reference with {objective!r}; "
                f"nothing was sent"
            )
        if self._read(PIEZO_FOCUS_USAGE) == ON:
            raise SafetyRefusedError(
                f"autofocus {label}: PiezoFocusUsage reads On, so it would correct "
                f"through the piezo, which no role reads, and the focus drive's "
                f"readback would not see the correction (m4 §11.1); switch it off at "
                f"the stand, then stabilize again; nothing was sent"
            )
        return reference

    def _restore(self, reference: FocusReference) -> None:
        """Make ``reference`` the device's: one action; a failure leaves none held."""
        label = self._label
        description = f"autofocus: restore the reference of {label}"
        self._held = None
        self._executor.do(
            description,
            Step(
                log=description,
                send=functools.partial(
                    self._core.setProperty, label, REFERENCE, reference.data
                ),
                readback=lambda: None,
                dry_result=None,
            ),
            motion=self._motion,
            timeout_s=_core_timeout(self._core),
        )

    def _judge(self, z_before: float | None, z_um: float | None) -> None:
        """Log the Z change, then the guard: it reads after, it cannot stop during."""
        label = self._label
        if z_before is None or z_um is None:
            logger.info("autofocus %s: stabilized; no focus drive to read", label)
            return
        moved = z_um - z_before
        logger.info(
            "autofocus %s: stabilize_now moved Z from %.3f to %.3f µm (%+.3f µm)",
            label,
            z_before,
            z_um,
            moved,
        )
        beyond: list[str] = []
        excursion = self._safety.max_engage_excursion_um
        # `not <=`: a Z that reads NaN is beyond any bound.
        if excursion is not None and not abs(moved) <= excursion:
            beyond.append(
                f"by {abs(moved):.3f} µm, beyond [safety] max_engage_excursion_um "
                f"({excursion:g} µm)"
            )
        limits = self._safety.z_soft_limits_um
        if limits is not None and not limits[0] <= z_um <= limits[1]:
            beyond.append(
                f"outside the Z soft limits [{limits[0]:g}, {limits[1]:g}] µm"
            )
        if beyond:
            raise AutofocusExcursionError(
                f"autofocus {label}: stabilize_now moved Z from {z_before:.3f} to "
                f"{z_um:.3f} µm, {' and '.join(beyond)}. It has already moved Z "
                f"there, and the layer left it there: the guard reads Z after a "
                f"stabilize and cannot stop one. Check the sample and the reference "
                f"before going on",
                z_start_um=z_before,
                z_um=z_um,
            )

    # --- the Z interlock (ZInterlock) ------------------------------------------------

    def suspend(self, reason: str) -> bool:
        """``False``: smc never has it on; periodic left on by another client is switched off.

        A nested action of the Z move's compound, skipped in dry-run as the
        move is. It is not switched on again: smc did not switch it on.
        """
        if self._read(PERIODIC) != ON:
            return False
        description = f"autofocus: periodic stabilisation off for {reason}"
        self._executor.do(
            description,
            Step(
                log=description,
                send=functools.partial(
                    self._core.setProperty, self._label, PERIODIC, OFF
                ),
                readback=lambda: None,
                dry_result=None,
            ),
        )
        logger.warning(
            "autofocus %s: periodic stabilisation was on (another client left it on) "
            "and was switched off for %s; smc does not switch it on again",
            self._label,
            reason,
        )
        return False

    def relock(self, reason: str, *, z_um: float | None) -> None:
        """Nothing: ``suspend()`` always returns ``False``, so the Z move never asks."""
        return None
