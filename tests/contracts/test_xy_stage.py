"""The ``XYStage`` contract, on every backend (design §3, §8).

Targets are relative to where the stage started (``envelope.xy``) and stay
within 60 µm of it; the ones outside the envelope are those the layer must
refuse, so they are never sent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smc.hardware.capabilities import XY, Limits, XYStage
from smc.hardware.errors import SafetyRefusedError

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from conftest import Envelope


def _origin(envelope: Envelope) -> XY:
    assert envelope.xy is not None
    return envelope.xy


def _assert_near(got: XY, x_um: float, y_um: float, tolerance_um: float) -> None:
    assert abs(got.x_um - x_um) <= tolerance_um, (got, x_um, y_um)
    assert abs(got.y_um - y_um) <= tolerance_um, (got, x_um, y_um)


def test_move_to_lands_within_tolerance(xy: XYStage, envelope: Envelope) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    landed = xy.move_to_um(o.x_um + 20, o.y_um - 10)
    _assert_near(landed, o.x_um + 20, o.y_um - 10, tol)
    _assert_near(xy.position_um(), o.x_um + 20, o.y_um - 10, tol)


def test_move_to_returns_once_the_stage_is_idle(
    xy: XYStage, envelope: Envelope
) -> None:
    # The demo's XY stage reads busy right after a move (about 1 s per 10 mm).
    o = _origin(envelope)
    xy.move_to_um(o.x_um + 20, o.y_um - 10)
    assert xy.is_busy() is False


def test_move_by_adds_to_the_position(xy: XYStage, envelope: Envelope) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    landed = xy.move_by_um(10, 5)
    _assert_near(landed, o.x_um + 10, o.y_um + 5, tol)
    _assert_near(xy.position_um(), o.x_um + 10, o.y_um + 5, tol)


def test_jog_above_the_limit_is_refused_and_sends_nothing(
    xy: XYStage, envelope: Envelope
) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    assert envelope.max_jog_um < 60
    with pytest.raises(SafetyRefusedError) as refused:
        xy.move_by_um(60, 0)
    assert refused.value.how_to_force != ""
    _assert_near(xy.position_um(), o.x_um, o.y_um, tol)


def test_forced_jog_above_the_limit_passes(xy: XYStage, envelope: Envelope) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    landed = xy.move_by_um(60, 0, force=True)
    _assert_near(landed, o.x_um + 60, o.y_um, tol)


def test_target_outside_soft_limits_is_refused_and_not_forceable(
    xy: XYStage, envelope: Envelope
) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    assert envelope.xy_span_um < 150
    with pytest.raises(SafetyRefusedError) as refused:
        xy.move_to_um(o.x_um + 150, o.y_um)
    assert refused.value.how_to_force == ""
    _assert_near(xy.position_um(), o.x_um, o.y_um, tol)


def test_relative_target_outside_soft_limits_is_refused_even_forced(
    xy: XYStage, envelope: Envelope
) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    with pytest.raises(SafetyRefusedError) as refused:
        xy.move_by_um(150, 0, force=True)
    assert refused.value.how_to_force == ""
    _assert_near(xy.position_um(), o.x_um, o.y_um, tol)


def test_wait_returns_on_an_idle_stage(xy: XYStage) -> None:
    assert xy.wait(timeout_s=10.0) is None
    assert xy.is_busy() is False


def test_stop_on_an_idle_stage_is_harmless(xy: XYStage, envelope: Envelope) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    xy.stop()
    _assert_near(xy.position_um(), o.x_um, o.y_um, tol)
    # The stop cancels nothing called after it: the next move runs.
    _assert_near(xy.move_to_um(o.x_um + 20, o.y_um - 10), o.x_um + 20, o.y_um - 10, tol)


def test_limits_um_reports_the_envelope(xy: XYStage, envelope: Envelope) -> None:
    o, span = _origin(envelope), envelope.xy_span_um
    assert xy.limits_um() == (
        Limits(o.x_um - span, o.x_um + span),
        Limits(o.y_um - span, o.y_um + span),
    )


def test_dry_run_move_returns_the_commanded_position_and_does_not_move(
    dry_xy: XYStage, envelope: Envelope
) -> None:
    o, tol = _origin(envelope), envelope.tolerance_um
    assert dry_xy.move_to_um(o.x_um + 20, o.y_um - 10) == XY(o.x_um + 20, o.y_um - 10)
    _assert_near(dry_xy.position_um(), o.x_um, o.y_um, tol)
