"""The facade's contract, and the contract fixtures' own put-back (design §7, §8).

A missing role and a failing device can only be staged on the fake, so
those two cases run there alone. The put-back is safety code at a stand:
``back_where_it_started`` checks it from a teardown that runs right after
``microscope``'s, and before the stand is closed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, Shutter, XYStage, ZStage
from smc.hardware.errors import CapabilityMissingError
from smc.hardware.profile import Profile
from smc.hardware.roles import Role

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from pymmcore_plus import CMMCorePlus

    from conftest import Envelope
    from smc.testing import FakeCore


def test_state_of_a_healthy_stand_lists_no_errors(microscope: Microscope) -> None:
    assert microscope.state().errors == []


def test_require_missing_role_names_the_role(fake_core: FakeCore) -> None:
    # Both shutters: LED Shutter would fill the role once the slot is empty.
    fake_core.unloadDevice("White Light Shutter")
    fake_core.unloadDevice("LED Shutter")
    facade = Microscope.from_core(cast("CMMCorePlus", fake_core), Profile.demo())
    with facade, pytest.raises(CapabilityMissingError) as missing:
        facade.require(Shutter)
    assert missing.value.role is Role.shutter
    assert "[roles.assign]" in str(missing.value)


def test_state_fills_what_it_can_when_z_fails(
    fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    fake_core.failing["Z"] = RuntimeError("Z: no answer")
    state = fake_microscope.state()
    assert state.z_um is None
    assert "z: RuntimeError: Z: no answer" in state.errors
    assert state.xy is not None
    assert state.exposure_ms is not None
    assert state.shutter_open is not None


@pytest.fixture
def back_where_it_started(stand: Microscope, envelope: Envelope) -> Iterator[None]:
    """After ``microscope``'s teardown, the stand reads as ``envelope`` again.

    Requested before ``microscope``, so its teardown runs after the put-back;
    it depends on ``stand``, so it runs before the stand is closed.
    """
    yield
    tol = envelope.tolerance_um
    xy, z = stand.get(XYStage), stand.get(ZStage)
    camera, shutter = stand.get(Camera), stand.get(Shutter)
    if xy is not None and envelope.xy is not None:
        here = xy.position_um()
        assert abs(here.x_um - envelope.xy.x_um) <= tol, here
        assert abs(here.y_um - envelope.xy.y_um) <= tol, here
    if z is not None and envelope.z_um is not None:
        assert abs(z.position_um() - envelope.z_um) <= tol
    if camera is not None and envelope.exposure_ms is not None:
        assert camera.exposure_ms() == pytest.approx(envelope.exposure_ms, rel=0.01)
    if shutter is not None:
        assert shutter.is_open() is envelope.shutter_open
        assert shutter.auto_shutter() is envelope.auto_shutter


def test_teardown_puts_the_stand_back_where_it_started(
    back_where_it_started: None, microscope: Microscope, envelope: Envelope
) -> None:
    xy, z = microscope.get(XYStage), microscope.get(ZStage)
    camera, shutter = microscope.get(Camera), microscope.get(Shutter)
    if xy is not None and envelope.xy is not None:
        xy.move_to_um(envelope.xy.x_um + 20, envelope.xy.y_um - 10)
    if z is not None and envelope.z_um is not None:
        z.move_to_um(envelope.z_um - 2)
    if camera is not None and envelope.exposure_ms is not None:
        start = envelope.exposure_ms
        camera.set_exposure_ms(30.0 if abs(start - 20.0) <= 1.0 else 20.0)
    if shutter is not None:
        shutter.set_open(not envelope.shutter_open)
        shutter.set_auto_shutter(not envelope.auto_shutter)


@pytest.fixture
def stopped_not_put_back(
    stand: Microscope, fake_core: FakeCore, envelope: Envelope
) -> Iterator[None]:
    """After ``microscope``'s teardown: the stages were stopped and the light cut, nothing moved back."""
    yield
    assert fake_core.log[-3:] == [
        "stop('XY')",
        "stop('Z')",
        "setShutterOpen('White Light Shutter', False)",
    ]
    assert envelope.xy is not None
    assert stand.require(XYStage).position_um().x_um == envelope.xy.x_um + 20


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_teardown_stops_a_halted_stand_instead_of_moving_it(
    stopped_not_put_back: None,
    microscope: Microscope,
    envelope: Envelope,
    fake_core: FakeCore,
) -> None:
    assert envelope.xy is not None
    assert envelope.shutter_open is False
    microscope.require(XYStage).move_to_um(envelope.xy.x_um + 20, envelope.xy.y_um)
    microscope.require(Shutter).set_open(True)
    microscope.stop()
    fake_core.log.clear()
