"""A device fault is a ``HardwareError`` that names the role and the call (#93, FM-79).

MMCore reports a fault as a bare ``RuntimeError``; ``as_hardware_errors`` turns
it into the ``HardwareError`` the CLI prints as ``✗ <message>``. The first
group runs the proxy on a stub core, with no Micro-Manager; the second makes
the *core* of ``FakeCore`` raise, because a test that patches a capability's
own method skips the layer it is meant to prove.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import pytest

from smc.hardware import Microscope
from smc.hardware.backends.mm import as_hardware_errors
from smc.hardware.capabilities import Camera, Properties, Shutter, XYStage, ZStage
from smc.hardware.errors import (
    HardwareError,
    MicroscopeHaltedError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.testing import FakeCore

TAIL = "check the device: power, cable, another program holding it"


class _Core:
    """Stands for the core: every call raises what the test gave it."""

    label = "XY"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def getXPosition(self, label: str) -> float:  # noqa: N802 - MMCore's name
        raise self.error

    def configure(self, *args: object, **kwargs: object) -> None:
        raise self.error


def _proxy(error: BaseException, role: str = "xy_stage") -> Any:
    return as_hardware_errors(cast("Any", _Core(error)), role)


# --- the proxy, on a stub ---------------------------------------------------------


def test_a_runtime_error_from_a_core_call_is_a_hardware_error_chained_from_it() -> None:
    boom = RuntimeError("Serial port timed out")
    with pytest.raises(HardwareError) as caught:
        _proxy(boom).getXPosition("XY")
    err = caught.value
    assert str(err) == (
        f"xy_stage: getXPosition('XY') failed (Serial port timed out); {TAIL}"
    )
    assert type(err) is HardwareError
    assert err.__cause__ is boom


def test_arguments_are_shown_as_repr_positional_then_keyword() -> None:
    long_argument = "x" * 500
    with pytest.raises(HardwareError) as caught:
        _proxy(RuntimeError("no"), "camera").configure(
            "A", 1.5, long_argument, mode="fast", n=2
        )
    start = "camera: configure('A', 1.5, "
    end = ", mode='fast', n=2) failed (no); "
    message = str(caught.value)
    assert message.startswith(start)
    # No truncation: the operator reads what was asked of the device.
    assert message[len(start) :].startswith(repr(long_argument) + end)


@pytest.mark.parametrize(
    "raised",
    [
        SafetyRefusedError("r", "h"),
        MicroscopeHaltedError(),
        MotionStoppedError("xy_stage XY"),
        MotionInProgressError(("xy_stage XY",)),
        HardwareError("plain"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_hardware_error_raised_inside_a_core_call_passes_through_untouched(
    raised: HardwareError,
) -> None:
    # HardwareError is a RuntimeError (FM-79): a halt raised by a callback
    # inside a command must still be the halt, not a "fault" of the device.
    with pytest.raises(HardwareError) as caught:
        _proxy(raised).getXPosition("XY")
    assert caught.value is raised
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "raised",
    [TypeError("bad"), ValueError("bad"), OSError("bad"), KeyboardInterrupt()],
    ids=lambda e: type(e).__name__,
)
def test_exceptions_that_are_not_runtime_errors_pass_through_untouched(
    raised: BaseException,
) -> None:
    with pytest.raises(type(raised)) as caught:
        _proxy(raised).getXPosition("XY")
    assert caught.value is raised


def test_a_fault_message_that_is_not_utf8_can_be_encoded() -> None:
    # pymmcore decodes adapter text with surrogateescape (FM-05).
    with pytest.raises(HardwareError) as caught:
        _proxy(RuntimeError("caf\udce9")).getXPosition("XY")
    message = str(caught.value)
    assert message.encode("utf-8")
    assert "caf" in message
    assert message.endswith(f"; {TAIL}")


def test_an_empty_fault_message_names_the_exception_type() -> None:
    with pytest.raises(HardwareError) as caught:
        _proxy(RuntimeError()).getXPosition("XY")
    assert (
        str(caught.value)
        == f"xy_stage: getXPosition('XY') failed (RuntimeError); {TAIL}"
    )


def test_attributes_that_are_not_calls_are_returned_and_dunders_are_refused() -> None:
    proxy = _proxy(RuntimeError("no"))
    assert proxy.label == "XY"
    assert not hasattr(proxy, "__len__")
    with pytest.raises(AttributeError):
        getattr(proxy, "__len__")  # noqa: B009 - the dunder is the point
    # copy looks for __setstate__ on a wrapper that has no slots filled yet.
    assert copy.copy(proxy).label == "XY"


# --- every capability, on FakeCore --------------------------------------------------


def boom(*args: object, **kwargs: object) -> Any:
    raise RuntimeError("device did not answer")


_FAULTS: list[tuple[str, Callable[[Microscope], object], str]] = [
    (
        "XY",
        lambda m: m.require(XYStage).position_um(),
        "xy_stage: getXPosition('XY') failed (",
    ),
    (
        "Z",
        lambda m: m.require(ZStage).position_um(),
        "focus: getPosition('Z') failed (",
    ),
    (
        "Camera",
        lambda m: m.require(Camera).exposure_ms(),
        "camera: getExposure() failed (",
    ),
    ("Camera", lambda m: m.require(Camera).snap(), "camera: snapImage() failed ("),
    (
        "White Light Shutter",
        lambda m: m.require(Shutter).is_open(),
        "shutter: getShutterOpen('White Light Shutter') failed (",
    ),
    (
        "Z",
        lambda m: m.require(Properties).get("Z", "Name"),
        "properties: getProperty('Z', 'Name') failed (",
    ),
    (
        "XY",
        lambda m: m.require(XYStage).move_to_um(5.0, 6.0),
        "xy_stage: setXYPosition('XY', 5.0, 6.0) failed (",
    ),
    (
        "Z",
        lambda m: m.require(ZStage).move_by_um(1.0),
        "focus: getPosition('Z') failed (",
    ),
    (
        "Camera",
        lambda m: m.require(Camera).set_exposure_ms(5.0),
        "camera: setExposure(5.0) failed (",
    ),
    (
        "White Light Shutter",
        lambda m: m.require(Shutter).set_open(True),
        "shutter: setShutterOpen('White Light Shutter', True) failed (",
    ),
    (
        "White Light Shutter",
        lambda m: m.require(Shutter).set_open(False),
        "shutter: setShutterOpen('White Light Shutter', False) failed (",
    ),
]


@pytest.mark.parametrize(
    ("label", "call", "expected_start"),
    _FAULTS,
    ids=[start.split(" failed")[0] for _, _, start in _FAULTS],
)
def test_a_device_fault_names_the_role_and_the_call(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    label: str,
    call: Callable[[Microscope], object],
    expected_start: str,
) -> None:
    original = RuntimeError("device did not answer")
    fake_core.failing[label] = original
    with pytest.raises(HardwareError) as caught:
        call(fake_microscope)
    err = caught.value
    assert str(err).startswith(expected_start)
    assert err.__cause__ is original
    assert type(err) is HardwareError


def test_a_faulting_objective_turret_is_a_hardware_error_naming_the_role(
    fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    # Pixel size comes from the profile's map by objective when the core has none.
    fake_core.pixel_size_um = 0.0
    fake_core.failing["Objective"] = RuntimeError("turret did not answer")
    with pytest.raises(HardwareError) as caught:
        fake_microscope.require(Camera).pixel_size_um()
    assert str(caught.value).startswith(
        "objective_turret: getStateLabel('Objective') failed ("
    )
    assert type(caught.value) is HardwareError


def test_a_move_whose_command_faults_is_stopped(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-17: the layer's stop on a failed command still goes out, through the proxy.
    xy = fake_microscope.require(XYStage)
    fake_core.log.clear()
    monkeypatch.setattr(fake_core, "setXYPosition", boom)
    with pytest.raises(HardwareError) as caught:
        xy.move_to_um(10.0, 20.0)
    assert str(caught.value).startswith(
        "xy_stage: setXYPosition('XY', 10.0, 20.0) failed ("
    )
    assert fake_core.log == ["stop('XY')"]


def test_a_move_whose_busy_state_faults_is_stopped_and_stays_tracked(
    fake_core: FakeCore,
    fake_microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-19 and FM-17: the wait cannot read the busy state, so the move is
    # stopped, and the next action finds the stage counted as moving.
    xy = fake_microscope.require(XYStage)
    fake_core.log.clear()
    monkeypatch.setattr(fake_core, "deviceBusy", boom)
    with pytest.raises(HardwareError) as caught:
        xy.move_to_um(10.0, 20.0)
    assert str(caught.value).startswith("xy_stage: deviceBusy('XY') failed (")
    assert fake_core.log == ["setXYPosition('XY', 10.0, 20.0)", "stop('XY')"]
    with pytest.raises(MotionInProgressError) as refused:
        xy.move_to_um(1.0, 2.0)
    assert "could not be read" in str(refused.value)
