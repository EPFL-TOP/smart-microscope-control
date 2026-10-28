"""A stand whose stages refuse ``stop()``: what the layer says, and what it still sends (m2 §11).

The Ti2-E answers every ``stop()`` of its XY stage, Z drive and PFS offset
with ``Unsupported device command (11)``, and a move that was sent runs to
its target (#208). A ``FakeCore`` with ``stop_refused`` reproduces it; the
profile declares it with ``[safety] stop_refused_by``. Nothing about what is
sent, or when, changes: only what the errors, the notes and the lines say.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.errors import (
    HardwareError,
    MotionStoppedError,
    SafetyRefusedError,
    StopNote,
    StopOutcome,
    StopRefusedError,
    stop_notes,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.testing import FakeCore

LOGGER = "smc.hardware.microscope"

#: The Ti2-E's refusal, as the layer's translator words it (m2 §11.8).
_XY_REFUSED = (
    "xy_stage: stop('XY') failed (Error in device \"XY\": Unsupported device "
    "command (11)); check the device: power, cable, another program holding it"
)


def _declared(*roles: Role) -> Profile:
    """The demo profile, declaring ``roles`` in ``[safety] stop_refused_by``."""
    demo = Profile.demo()
    safety = demo.safety.model_copy(update={"stop_refused_by": roles})
    return demo.model_copy(update={"safety": safety})


@contextmanager
def _stand(
    profile: Profile | None = None, *, refused: tuple[str, ...] = ("XY", "Z")
) -> Iterator[tuple[FakeCore, Microscope]]:
    """A fake demo stand whose ``refused`` stages refuse their stop; the log cleared."""
    core = FakeCore.demo_like()
    core.stop_refused = set(refused)
    microscope = Microscope.from_core(
        cast("CMMCorePlus", core),
        profile if profile is not None else _declared(Role.xy_stage, Role.focus),
    )
    core.log.clear()
    try:
        yield core, microscope
    finally:
        microscope.close()


def _interrupt_the_first_busy_poll(core: FakeCore) -> Callable[[], None]:
    """Ctrl-C at the move's first busy poll; later polls answer as the fake does.

    An instance attribute, removed by the returned call (FM-98's test hazards).
    Measured on a prototype (2026-10-28): this path reaches the give-up.
    """
    real = core.deviceBusy
    polled: list[str] = []

    def busy(label: str) -> bool:
        if not polled:
            polled.append(label)
            raise KeyboardInterrupt
        return real(label)

    core.deviceBusy = busy  # type: ignore[method-assign]

    def undo() -> None:
        del core.deviceBusy

    return undo


# --- Executor.stop ------------------------------------------------------------


def test_a_declared_refused_stop_raises_stop_refused_error_and_was_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _stand() as (core, microscope):
        with (
            caplog.at_level(logging.WARNING, logger=LOGGER),
            pytest.raises(StopRefusedError) as refused,
        ):
            microscope.require(XYStage).stop()
        # The declaration changes what is said, never what is sent (m2 §11.2).
        assert core.log == ["stop('XY')"]
    error = refused.value
    assert error.refused == ("xy_stage XY",)
    assert type(error.__cause__) is HardwareError
    assert str(error.__cause__) == _XY_REFUSED
    assert str(error) == (
        "`xy_stage XY` refused the stop, as the profile declares ([safety] "
        "stop_refused_by): a move under way runs to its target; stop it with the "
        "stand's own controls if it must not finish"
    )
    assert caplog.messages[0] == (
        f"xy_stage XY: stop failed (HardwareError({_XY_REFUSED!r})); the profile "
        "declares this refusal ([safety] stop_refused_by), so a move under way runs "
        "to its target"
    )


def test_an_undeclared_refused_stop_raises_the_device_error_as_before(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The Executor never parses the device's text: undeclared, it is a fault.
    with _stand(Profile.demo()) as (core, microscope):
        with (
            caplog.at_level(logging.WARNING, logger=LOGGER),
            pytest.raises(HardwareError) as failed,
        ):
            microscope.require(XYStage).stop()
        assert core.log == ["stop('XY')"]
    assert type(failed.value) is HardwareError
    assert str(failed.value) == _XY_REFUSED
    assert caplog.messages[0] == (
        f"xy_stage XY: stop failed (HardwareError({_XY_REFUSED!r}))"
    )


def test_stop_refused_error_is_not_a_safety_refusal() -> None:
    # FM-64: a tool that skips a target on a refusal must not read a move
    # that runs on as one that was never sent.
    assert not issubclass(StopRefusedError, SafetyRefusedError)
    caught: list[str] = []
    with _stand() as (_, microscope):
        try:
            microscope.require(XYStage).stop()
        except SafetyRefusedError:
            caught.append("a refusal")
        except StopRefusedError:
            caught.append("a declared refusal of stop()")
    assert caught == ["a declared refusal of stop()"]


# --- the give-up and its notes -------------------------------------------------


def test_ctrl_c_during_a_move_notes_the_declared_refusal() -> None:
    with _stand() as (core, microscope):
        core.busy_devices = {"XY"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            undo()
        assert core.log == ["setXYPosition('XY', 10.0, 0.0)", "stop('XY')"]
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.refused, True),
    )


def test_ctrl_c_during_a_move_on_an_undeclared_stand_notes_a_refusal_not_declared() -> (
    None
):
    with _stand(Profile.demo()) as (core, microscope):
        core.busy_devices = {"XY"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            undo()
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.refused, False),
    )


def test_ctrl_c_during_a_move_notes_an_accepted_stop() -> None:
    with _stand(Profile.demo(), refused=()) as (core, microscope):
        core.busy_devices = {"XY"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            undo()
        assert core.log == ["setXYPosition('XY', 10.0, 0.0)", "stop('XY')"]
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.accepted, False),
    )


def test_ctrl_c_outside_a_move_leaves_no_note() -> None:
    # Interrupted while the move reads its anchor: nothing was sent, so
    # nothing was given up on.
    with _stand() as (core, microscope):

        def interrupted_read(label: str | None = None) -> list[float]:
            raise KeyboardInterrupt

        core.getXYPosition = interrupted_read  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_by_um(10.0, 0.0)
        finally:
            del core.getXYPosition
        assert core.log == []
    assert stop_notes(interrupted.value) == ()


def test_a_second_ctrl_c_inside_the_stop_is_noted_as_interrupted() -> None:
    with _stand() as (core, microscope):
        core.busy_devices = {"XY"}
        undo = _interrupt_the_first_busy_poll(core)

        def cut(label: str) -> None:
            raise KeyboardInterrupt("second Ctrl-C")

        core.stop = cut  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            undo()
            del core.stop
    second = interrupted.value
    assert str(second) == "second Ctrl-C"
    # The first Ctrl-C is the context, and it carries no note of its own: its
    # give-up never returned.
    assert isinstance(second.__context__, KeyboardInterrupt)
    assert stop_notes(second) == (
        StopNote("xy_stage XY", StopOutcome.interrupted, True),
    )


def test_a_stop_during_a_move_on_a_declared_stand_says_it_may_have_run_to_its_target() -> (
    None
):
    # The generation moved, then the device read idle: the caller asked for a
    # stop, so the plan's next step must not run (M1 §13).
    with _stand() as (core, microscope):
        stage = microscope.require(XYStage)
        polls: list[str] = []

        def busy(label: str) -> bool:
            polls.append(label)
            if len(polls) == 1:
                with pytest.raises(StopRefusedError):
                    stage.stop()
                return True
            return False

        core.deviceBusy = busy  # type: ignore[method-assign]
        try:
            with pytest.raises(MotionStoppedError) as stopped:
                stage.move_to_um(10.0, 0.0)
        finally:
            del core.deviceBusy
    assert str(stopped.value) == (
        "`xy_stage XY`: a stop was requested during the move, but the profile "
        "declares that it refuses its stop ([safety] stop_refused_by), so it may "
        "have run to its target; read the position before moving on"
    )


def test_a_timeout_on_a_declared_stand_says_the_move_runs_to_its_target() -> None:
    profile = _declared(Role.xy_stage)
    micromanager = profile.micromanager.model_copy(update={"device_timeout_ms": 50})
    with _stand(profile.model_copy(update={"micromanager": micromanager})) as (
        core,
        microscope,
    ):
        core.busy_devices = {"XY"}
        with pytest.raises(HardwareError) as timed_out:
            microscope.require(XYStage).move_to_um(10.0, 0.0)
    assert str(timed_out.value).startswith(
        "xy_stage XY is still busy after 0.05 s; the stand refuses its stop, as the "
        "profile declares ([safety] stop_refused_by), so it runs to its target. "
    )
