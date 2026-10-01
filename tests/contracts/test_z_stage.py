"""The ``ZStage`` contract, on every backend (design §3, §8).

Every target sent is at or below where the drive started (``envelope.z_um``),
by at most 2 µm: away from the sample on an inverted stand. The jog guard
for Z is #84's, which adds its case here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smc.hardware.capabilities import Limits, ZStage
from smc.hardware.errors import SafetyRefusedError

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from conftest import Envelope


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
    # ZStage.move_by_um takes no force: the refusal says it cannot be forced.
    z0, tol = _origin(envelope), envelope.tolerance_um
    with pytest.raises(SafetyRefusedError) as refused:
        z.move_by_um(-5)
    assert refused.value.how_to_force == ""
    assert abs(z.position_um() - z0) <= tol


def test_wait_returns_on_an_idle_stage(z: ZStage) -> None:
    assert z.wait(timeout_s=10.0) is None
    assert z.is_busy() is False


def test_stop_on_an_idle_stage_is_harmless(z: ZStage, envelope: Envelope) -> None:
    z0, tol = _origin(envelope), envelope.tolerance_um
    z.stop()
    assert abs(z.position_um() - z0) <= tol
    # The stop cancels nothing called after it: the next move runs.
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
