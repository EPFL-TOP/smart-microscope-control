"""The facade's contract, and the contract fixtures' own put-back (design §7, §8).

A missing role and a failing device can only be staged on the fake, so
those two cases run there alone. The put-back is safety code at a stand:
``back_where_it_started`` checks it from a teardown that runs right after
``microscope``'s, and before the stand is closed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, Shutter, XYStage, ZStage
from smc.hardware.errors import CapabilityMissingError, HardwareError
from smc.hardware.profile import Profile, SafetySection
from smc.hardware.roles import Role

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from pymmcore_plus import CMMCorePlus

    from conftest import Envelope, PutBack
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


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_envelope_refuses_a_start_too_close_to_the_profile_limits(
    stand: Microscope, envelope: Envelope
) -> None:
    # The fake starts at (0, 0) and Z 0; the window is XY +/- 100, Z +/- 3.
    def limited(**limits: object) -> Profile:
        return stand.profile.model_copy(
            update={"safety": SafetySection.model_validate(limits)}
        )

    near_x = limited(xy_soft_limits_um=((-1000, 30), (-1000, 1000)))
    with pytest.raises(
        pytest.fail.Exception, match=r"too close to its X soft limits \[-1000, 30\] µm"
    ):
        envelope.profile(near_x)
    near_z = limited(z_soft_limits_um=(-1000, 1))
    with pytest.raises(
        pytest.fail.Exception, match=r"too close to its Z soft limits \[-1000, 1\] µm"
    ):
        envelope.profile(near_z)
    roomy = limited(
        xy_soft_limits_um=((-1000, 1000), (-1000, 1000)), z_soft_limits_um=(-10, 10)
    )
    safety = envelope.profile(roomy).safety
    assert safety.xy_soft_limits_um == ((-100, 100), (-100, 100))
    assert safety.z_soft_limits_um == (-3, 3)


#: An inner run that moves the fake stand, opens the light, then is
#: interrupted as an operator's Ctrl-C would be. ``snapshot`` is requested
#: before ``microscope``, and depends on ``stand``, so it records the log
#: right after the put-back and before the stand is closed.
INTERRUPTED_RUN = """
from pathlib import Path

import pytest

from smc.hardware.capabilities import Shutter, XYStage

LOG = Path(__file__).with_name("teardown.log")


@pytest.fixture
def snapshot(stand, fake_core):
    yield
    LOG.write_text("\\n".join(fake_core.log), encoding="utf-8")


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_operator_presses_ctrl_c(snapshot, microscope, envelope, fake_core):
    microscope.require(XYStage).move_to_um(envelope.xy.x_um + 20, envelope.xy.y_um)
    microscope.require(Shutter).set_open(True)
    fake_core.log.clear()
    raise KeyboardInterrupt
"""

#: Two tests on the fake: a Ctrl-C lands in test one's put-back, while the
#: stage travels back. ``snapshot`` depends on ``fake_core`` only, so it
#: records the log after the stand's close().
CTRL_C_IN_THE_PUT_BACK = """
from pathlib import Path

import pytest

from smc.hardware.capabilities import XYStage

LOG = Path(__file__).with_name("events.log")


def note(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\\n")


@pytest.fixture
def snapshot(fake_core):
    yield
    note("core: " + " | ".join(fake_core.log))


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_one(snapshot, microscope, envelope, fake_core):
    microscope.require(XYStage).move_to_um(envelope.xy.x_um + 20, envelope.xy.y_um)

    def interrupted_move(label, x_um, y_um):
        raise KeyboardInterrupt

    fake_core.setXYPosition = interrupted_move
    fake_core.log.clear()


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_two(microscope):
    note("test two ran")
"""

INI = """
[pytest]
markers =
    demo: needs the Micro-Manager demo adapters
    hardware: needs a real microscope
"""


def test_teardown_after_ctrl_c_stops_the_stand_instead_of_moving_it(
    pytester: pytest.Pytester,
) -> None:
    # The layer's Ctrl-C handling stops the stage and leaves the facade
    # neither halted nor moving; only the interrupted session tells the
    # put-back that the operator meant "stop", not "go back".
    pytester.makeini(INI)
    pytester.makeconftest(
        (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(INTERRUPTED_RUN)
    result = pytester.runpytest_subprocess("-p", "smc.testing.fixtures", timeout=120.0)
    assert result.ret == pytest.ExitCode.INTERRUPTED, result.stdout.str()
    log = (pytester.path / "teardown.log").read_text(encoding="utf-8").splitlines()
    assert log == [
        "stop('XY')",
        "stop('Z')",
        "setShutterOpen('White Light Shutter', False)",
    ]


def test_ctrl_c_during_the_put_back_ends_the_run_and_still_closes_the_stand(
    pytester: pytest.Pytester,
) -> None:
    # Reported as an ordinary teardown error, the Ctrl-C would let test two
    # reopen the stand and move it.
    pytester.makeini(INI)
    pytester.makeconftest(
        (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(CTRL_C_IN_THE_PUT_BACK)
    result = pytester.runpytest_subprocess("-p", "smc.testing.fixtures", timeout=120.0)
    output = result.stdout.str()
    assert result.ret == pytest.ExitCode.INTERRUPTED, output
    log = pytester.path / "events.log"
    # No file at all: not even the snapshot after the stand's close() ran.
    events = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    assert "test two ran" not in events, events
    # The stand's close() still ran: re-raising the KeyboardInterrupt from the
    # put-back would also end the run, but would skip it.
    assert len(events) == 1, events
    assert "unloadAllDevices()" in events[0]
    assert "Ctrl-C during the put-back" in output


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_put_back_cuts_the_light_before_moving_and_restores_it_last(
    stand: Microscope, envelope: Envelope, fake_core: FakeCore, put_back: PutBack
) -> None:
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    shutter = facade.require(Shutter)
    assert envelope.xy is not None
    assert envelope.z_um is not None
    assert envelope.shutter_open is False
    assert envelope.auto_shutter is True
    x0, y0, z0 = envelope.xy.x_um, envelope.xy.y_um, envelope.z_um

    # Started closed: the light goes off before anything moves.
    facade.require(XYStage).move_to_um(x0 + 20, y0 - 10)
    facade.require(ZStage).move_to_um(z0 - 2)
    facade.require(Camera).set_exposure_ms(20.0)
    shutter.set_auto_shutter(False)
    shutter.set_open(True)
    fake_core.log.clear()
    put_back(facade, envelope)
    assert fake_core.log == [
        "setShutterOpen('White Light Shutter', False)",
        f"setXYPosition('XY', {x0!r}, {y0!r})",
        f"setPosition('Z', {z0!r})",
        f"setExposure({envelope.exposure_ms!r})",
        "setAutoShutter(True)",
    ]

    # Started open: the light comes back on only after every other item.
    started_open = replace(envelope, shutter_open=True)
    facade.require(XYStage).move_to_um(x0 + 20, y0 - 10)
    fake_core.log.clear()
    put_back(facade, started_open)
    assert fake_core.log == [
        f"setXYPosition('XY', {x0!r}, {y0!r})",
        "setShutterOpen('White Light Shutter', True)",
    ]


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_put_back_closes_the_shutter_even_when_its_read_fails(
    stand: Microscope,
    envelope: Envelope,
    fake_core: FakeCore,
    put_back: PutBack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    facade.require(Shutter).set_open(True)

    def no_answer(label: str | None = None) -> bool:
        raise RuntimeError("shutter: no answer")

    monkeypatch.setattr(fake_core, "getShutterOpen", no_answer)
    fake_core.log.clear()
    # The readback after the close fails too, so the put-back still reports it.
    with pytest.raises(RuntimeError, match="was not put back"):
        put_back(facade, envelope)
    monkeypatch.undo()
    assert fake_core.log[:1] == ["setShutterOpen('White Light Shutter', False)"]
    assert facade.require(Shutter).is_open() is False


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_second_ctrl_c_during_the_stop_still_cuts_the_light(
    stand: Microscope,
    envelope: Envelope,
    fake_core: FakeCore,
    put_back: PutBack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    facade.require(Shutter).set_open(True)
    stop = fake_core.stop

    def interrupted_stop(label: str) -> None:
        stop(label)
        if label == "Z":
            raise KeyboardInterrupt

    monkeypatch.setattr(fake_core, "stop", interrupted_stop)
    fake_core.log.clear()
    with pytest.raises(RuntimeError, match="was not put back") as caught:
        put_back(facade, envelope, interrupted=True)
    assert type(caught.value).__name__ == "PutBackInterruptedError"
    assert "stop: KeyboardInterrupt()" in str(caught.value)
    assert fake_core.log[-1] == "setShutterOpen('White Light Shutter', False)"


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_ctrl_c_during_the_put_back_skips_the_moves_left_and_names_them(
    stand: Microscope,
    envelope: Envelope,
    fake_core: FakeCore,
    put_back: PutBack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert envelope.z_um is not None
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    facade.require(XYStage).move_to_um(20, -10)
    facade.require(ZStage).move_to_um(envelope.z_um - 2)
    started_open = replace(envelope, shutter_open=True)

    def interrupted_move(label: str, x_um: float, y_um: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_core, "setXYPosition", interrupted_move)
    fake_core.log.clear()
    with pytest.raises(RuntimeError, match="was not put back") as caught:
        put_back(facade, started_open)
    assert type(caught.value).__name__ == "PutBackInterruptedError"
    message = str(caught.value)
    assert "xy: KeyboardInterrupt()" in message
    assert (
        "not attempted after Ctrl-C: z, exposure, auto-shutter, shutter open" in message
    )
    assert not any(entry.startswith("setPosition(") for entry in fake_core.log)
    assert "setShutterOpen('White Light Shutter', True)" not in fake_core.log


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_put_back_names_every_item_it_could_not_restore(
    stand: Microscope,
    envelope: Envelope,
    fake_core: FakeCore,
    put_back: PutBack,
) -> None:
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    facade.require(XYStage).move_to_um(20, -10)
    facade.require(ZStage).move_to_um(-2)
    fake_core.failing["XY"] = RuntimeError("xy: no answer")
    fake_core.failing["Z"] = RuntimeError("z: no answer")
    with pytest.raises(RuntimeError, match="was not put back") as caught:
        put_back(facade, envelope)
    fake_core.failing.clear()
    # An ordinary failure: the run carries on, so it must not end the session.
    assert type(caught.value) is RuntimeError
    message = str(caught.value)
    assert "xy: RuntimeError('xy: no answer')" in message
    assert "z: RuntimeError('z: no answer')" in message
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("backend", ["fake"], indirect=True)
def test_put_back_of_a_halted_stand_reports_a_failed_stop_and_a_failed_close(
    stand: Microscope,
    envelope: Envelope,
    fake_core: FakeCore,
    put_back: PutBack,
) -> None:
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    facade.require(Shutter).set_open(True)
    fake_core.failing["XY"] = RuntimeError("xy: no stop")
    with pytest.raises(HardwareError):
        facade.stop()  # halted, and the XY stop failed
    fake_core.failing["White Light Shutter"] = RuntimeError("shutter: stuck")
    with pytest.raises(RuntimeError, match="was not put back") as caught:
        put_back(facade, envelope)
    fake_core.failing.clear()
    message = str(caught.value)
    assert "stop: HardwareError(" in message
    assert "xy: no stop" in message
    assert "shutter close: RuntimeError('shutter: stuck')" in message
