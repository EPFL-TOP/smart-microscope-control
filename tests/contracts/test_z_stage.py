"""The ``ZStage`` contract, on every backend (design §3, §8).

Every target sent is at or below where the drive started (``envelope.z_um``),
by at most 2 µm: away from the sample on an inverted stand. The envelope's Z
jog limit (``max_z_jog_um``, 1 µm) is below that, so a 1.5 µm jog exercises the
jog guard without leaving the envelope (#84).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smc.hardware.capabilities import Limits, ZStage
from smc.hardware.errors import SafetyRefusedError, StopRefusedError
from smc.hardware.roles import Role

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from conftest import Envelope
    from smc.hardware import Microscope


def _origin(envelope: Envelope) -> float:
    assert envelope.z_um is not None
    return envelope.z_um


def test_move_to_lands_within_tolerance(z: ZStage, envelope: Envelope) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    assert abs(z.move_to_um(z0 - 2) - (z0 - 2)) <= tol
    assert abs(z.position_um() - (z0 - 2)) <= tol


def test_move_to_returns_once_the_stage_is_idle(z: ZStage, envelope: Envelope) -> None:
    z.move_to_um(_origin(envelope) - 2)
    assert z.is_busy() is False


def test_move_by_adds_to_the_position(z: ZStage, envelope: Envelope) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    assert abs(z.move_by_um(-1) - (z0 - 1)) <= tol
    assert abs(z.position_um() - (z0 - 1)) <= tol


def test_jog_above_the_limit_is_refused_and_sends_nothing(
    z: ZStage, envelope: Envelope
) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    assert envelope.max_z_jog_um < 1.5
    with pytest.raises(SafetyRefusedError) as refused:
        z.move_by_um(-1.5)
    assert refused.value.how_to_force != ""
    assert abs(z.position_um() - z0) <= tol


def test_forced_jog_above_the_limit_passes(z: ZStage, envelope: Envelope) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    landed = z.move_by_um(-1.5, force=True)
    assert abs(landed - (z0 - 1.5)) <= tol


def test_target_outside_soft_limits_is_refused_and_not_forceable(
    z: ZStage, envelope: Envelope
) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    assert envelope.z_span_um < 5
    with pytest.raises(SafetyRefusedError) as refused:
        z.move_to_um(z0 - 5)
    assert refused.value.how_to_force == ""
    assert abs(z.position_um() - z0) <= tol


def test_relative_target_outside_soft_limits_is_refused_even_forced(
    z: ZStage, envelope: Envelope
) -> None:
    # Forced, so that the soft limit and not the jog guard is what refuses it:
    # unforced, the 5 µm jog is above max_z_jog_um and is refused first.
    z0, tol = _origin(envelope), envelope.tolerance_um
    with pytest.raises(SafetyRefusedError) as refused:
        z.move_by_um(-5, force=True)
    assert refused.value.how_to_force == ""
    assert abs(z.position_um() - z0) <= tol


def test_wait_returns_on_an_idle_stage(z: ZStage) -> None:
    assert z.wait(timeout_s=10.0) is None
    assert z.is_busy() is False


def _stop_refused(microscope: Microscope) -> bool:
    """Whether the profile declares that this stand refuses a stop of its focus drive (m2 §11)."""
    return Role.focus in microscope.profile.safety.stop_refused_by


def test_stop_on_an_idle_stage_is_harmless(
    z: ZStage, envelope: Envelope, microscope: Microscope
) -> None:
    if _stop_refused(microscope):
        pytest.skip(
            "the profile declares that this stand refuses a stop of focus "
            "([safety] stop_refused_by, m2 §11)"
        )
    z0, tol = _origin(envelope), envelope.tolerance_um
    z.stop()
    assert abs(z.position_um() - z0) <= tol
    # The stop cancels nothing called after it: the next move runs.
    assert abs(z.move_to_um(z0 - 2) - (z0 - 2)) <= tol


def test_a_declared_stop_refusal_still_holds(
    z: ZStage, envelope: Envelope, microscope: Microscope
) -> None:
    # A firmware that starts to accept the stop fails this: correct the profile.
    if not _stop_refused(microscope):
        pytest.skip(
            "the profile does not declare that this stand refuses a stop of focus "
            "([safety] stop_refused_by)"
        )
    z0, tol = _origin(envelope), envelope.tolerance_um
    with pytest.raises(StopRefusedError):
        z.stop()
    assert abs(z.position_um() - z0) <= tol
    # The refused stop cancels nothing called after it: the next move runs.
    assert abs(z.move_to_um(z0 - 2) - (z0 - 2)) <= tol


def test_limits_um_reports_the_envelope(z: ZStage, envelope: Envelope) -> None:
    z0, span = _origin(envelope), envelope.z_span_um
    assert z.limits_um() == Limits(z0 - span, z0 + span)


def test_dry_run_move_returns_the_commanded_position_and_does_not_move(
    dry_z: ZStage, envelope: Envelope
) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    assert dry_z.move_to_um(z0 - 2) == z0 - 2
    assert abs(dry_z.position_um() - z0) <= tol
