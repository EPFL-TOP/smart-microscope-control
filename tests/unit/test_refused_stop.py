"""A stand whose stages refuse ``stop()``: what the layer says, and what it still sends (m2 §11).

The Ti2-E answers every ``stop()`` of its XY stage, Z drive and PFS offset
with ``Unsupported device command (11)``, and a move that was sent runs to
its target (#208). A ``FakeCore`` with ``stop_refused`` reproduces it; the
profile declares it with ``[safety] stop_refused_by``. Nothing about what is
sent, or when, changes: only what the errors, the notes and the lines say.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus, DeviceType

from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, ObjectiveTurret, XYStage, ZStage
from smc.hardware.errors import (
    HardwareError,
    MotionStoppedError,
    SafetyRefusedError,
    StopNote,
    StopOutcome,
    StopRefusedError,
    stop_notes,
)
from smc.hardware.profile import (
    MicroscopeSection,
    Profile,
    RolesSection,
    SafetySection,
)
from smc.hardware.roles import Role
from smc.hardware.safety import Executor, Motion
from smc.testing import FakeCore, FakeDevice

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


def test_a_failed_linked_call_wins_over_a_declared_refusal() -> None:
    # A declared refusal is no fault, but a PFS switch-off that failed is: the
    # stopping caller must hear of it (adversarial review of #204).
    executor = Executor(
        dry_run=False, lock=threading.RLock(), logger=logging.getLogger(LOGGER)
    )
    sent: list[str] = []

    def refuse() -> None:
        sent.append("Z")
        raise RuntimeError('Error in device "Z": Unsupported device command (11)')

    def switch_off() -> None:
        sent.append("PFS off")
        raise ConnectionError("PFS: stuck on")

    z = Motion("Z", "z Z", lambda: False, refuse, stop_refused=True)
    with (
        executor.linked_stop("Z", switch_off),
        pytest.raises(ConnectionError, match=r"^PFS: stuck on$"),
    ):
        executor.stop(z)
    assert sent == ["PFS off", "Z"]


_LINK_NAME = "autofocus_offset PFSOffset"


def _refusal_of_the_link() -> Exception:
    return StopRefusedError((_LINK_NAME,))


def _refusal_of_the_device() -> Exception:
    return RuntimeError('Error in device "Z": Unsupported device command (11)')


def _fault_of_the_link() -> Exception:
    return ConnectionError("PFS: stuck on")


def _fault_of_the_device() -> Exception:
    return ConnectionError("Z: stuck")


@pytest.mark.parametrize(
    ("link", "device", "declared", "raises", "text", "cause"),
    [
        pytest.param(
            _refusal_of_the_link,
            _refusal_of_the_device,
            True,
            StopRefusedError,
            str(StopRefusedError((_LINK_NAME, "z Z"))),
            RuntimeError,
            id="refusal-refusal-merged",
        ),
        pytest.param(
            _refusal_of_the_link,
            None,
            True,
            StopRefusedError,
            str(StopRefusedError((_LINK_NAME,))),
            None,
            id="refusal-accepted-the-links",
        ),
        pytest.param(
            None,
            _refusal_of_the_device,
            True,
            StopRefusedError,
            str(StopRefusedError(("z Z",))),
            RuntimeError,
            id="accepted-refusal-the-devices",
        ),
        pytest.param(
            _fault_of_the_link,
            _refusal_of_the_device,
            True,
            ConnectionError,
            "PFS: stuck on",
            None,
            id="fault-refusal-the-fault",
        ),
        pytest.param(
            _refusal_of_the_link,
            _fault_of_the_device,
            False,
            ConnectionError,
            "Z: stuck",
            None,
            id="refusal-fault-the-devices-fault",
        ),
        pytest.param(
            _fault_of_the_link,
            _fault_of_the_device,
            False,
            ConnectionError,
            "Z: stuck",
            None,
            id="fault-fault-the-devices-fault",
        ),
    ],
)
def test_a_linked_stop_raises_one_error_by_the_table_of_m2_11_11(
    link: Callable[[], Exception] | None,
    device: Callable[[], Exception] | None,
    declared: bool,
    raises: type[Exception],
    text: str,
    cause: type[Exception] | None,
) -> None:
    # Of the linked call and the device's own stop, one error is raised. Two
    # refusals used to leave Z's own out of `.refused` (m2 §11.11 E1, #224), and
    # a fault still wins over a refusal (FM-127).
    executor = Executor(
        dry_run=False, lock=threading.RLock(), logger=logging.getLogger(LOGGER)
    )
    sent: list[str] = []

    def linked_call() -> None:
        sent.append("link")
        if link is not None:
            raise link()

    def stop() -> None:
        sent.append("Z")
        if device is not None:
            raise device()

    z = Motion("Z", "z Z", lambda: False, stop, stop_refused=declared)
    with (
        executor.linked_stop("Z", linked_call),
        pytest.raises(Exception) as raised,  # noqa: PT011 - the table decides the type
    ):
        executor.stop(z)
    assert sent == ["link", "Z"]
    assert type(raised.value) is raises
    assert str(raised.value) == text
    assert type(raised.value.__cause__) is (cause or type(None))


def test_the_merged_refusal_names_each_stop_once_link_first() -> None:
    # `refused` is the link's names, then the device's, each once, whichever
    # of them the two lists repeat (m2 §11.11 E1, `_Stops.refused` alike).
    executor = Executor(
        dry_run=False, lock=threading.RLock(), logger=logging.getLogger(LOGGER)
    )

    def linked_call() -> None:
        raise StopRefusedError((_LINK_NAME, "z Z"))

    def stop() -> None:
        raise RuntimeError("Unsupported device command (11)")

    z = Motion("Z", "z Z", lambda: False, stop, stop_refused=True)
    with (
        executor.linked_stop("Z", linked_call),
        pytest.raises(StopRefusedError) as raised,
    ):
        executor.stop(z)
    assert raised.value.refused == (_LINK_NAME, "z Z")


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
        StopNote("xy_stage XY", StopOutcome.refused),
    )


def test_ctrl_c_during_a_move_on_an_undeclared_stand_notes_a_failure_not_a_refusal() -> (
    None
):
    # The layer never parses the device's text: a stop that raised on a role the
    # profile does not declare is a fault, and the line must not call it a
    # refusal (m2 §11.11 E4).
    with _stand(Profile.demo()) as (core, microscope):
        core.busy_devices = {"XY"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            undo()
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.failed),
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
        StopNote("xy_stage XY", StopOutcome.accepted),
    )


def test_ctrl_c_during_a_turret_move_notes_that_it_cannot_be_stopped() -> None:
    # A State device has no stop: the note must never read "accepted", or the
    # Ctrl-C line would say a turret was stopped that was sent no stop.
    with _stand(Profile.demo()) as (core, microscope):
        turret = microscope.require(ObjectiveTurret)
        target = next(label for label in turret.labels() if label != turret.current())
        core.log.clear()
        core.busy_devices = {"Objective"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                turret.select(target, confirm=True)
        finally:
            undo()
            core.busy_devices.clear()
        assert "stop('Objective')" not in core.log
    assert stop_notes(interrupted.value) == (
        StopNote("objective_turret Objective", StopOutcome.impossible),
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


_STILL_BUSY = "; it still reads busy, so it counts as moving until it is idle"
_DECLARES = (
    "; the profile declares this refusal ([safety] stop_refused_by), so a move "
    "under way runs to its target"
)


_CAUSES = [
    pytest.param("ctrl-c", "KeyboardInterrupt", id="ctrl-c"),
    pytest.param("timeout", "0.05 s timeout", id="timeout"),
]


def _give_up_line(
    caplog: pytest.LogCaptureFixture,
    profile: Profile,
    *,
    ctrl_c: bool,
    refused: tuple[str, ...] = ("XY", "Z"),
) -> str:
    """The WARNING a give-up writes for XY's stop, after a move left busy.

    Ended by a Ctrl-C at the first busy poll, or by the move's 50 ms timeout.
    """
    micromanager = profile.micromanager.model_copy(update={"device_timeout_ms": 50})
    profile = profile.model_copy(update={"micromanager": micromanager})
    with _stand(profile, refused=refused) as (core, microscope):
        core.busy_devices = {"XY"}
        undo = _interrupt_the_first_busy_poll(core) if ctrl_c else None
        try:
            with (
                caplog.at_level(logging.WARNING, logger=LOGGER),
                pytest.raises((KeyboardInterrupt, HardwareError)),
            ):
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            if undo is not None:
                undo()
    (line,) = [m for m in caplog.messages if m.startswith("xy_stage XY: stop after")]
    return line


@pytest.mark.parametrize(("cause", "why"), _CAUSES)
def test_the_give_up_line_on_a_declared_role_says_the_move_runs_to_its_target(
    cause: str, why: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The CLI line said it; the log contradicted it by naming the failure alone
    # (#224 item 2). Order: the failure, the declaration, the busy note.
    line = _give_up_line(
        caplog, _declared(Role.xy_stage, Role.focus), ctrl_c=cause == "ctrl-c"
    )
    assert line == (
        f"xy_stage XY: stop after {why} failed (HardwareError({_XY_REFUSED!r}))"
        f"{_DECLARES}{_STILL_BUSY}"
    )


@pytest.mark.parametrize(("cause", "why"), _CAUSES)
def test_the_give_up_line_on_an_undeclared_role_has_no_declaration_note(
    cause: str, why: str, caplog: pytest.LogCaptureFixture
) -> None:
    line = _give_up_line(caplog, Profile.demo(), ctrl_c=cause == "ctrl-c")
    assert line == (
        f"xy_stage XY: stop after {why} failed (HardwareError({_XY_REFUSED!r}))"
        f"{_STILL_BUSY}"
    )


@pytest.mark.parametrize(("cause", "why"), _CAUSES)
def test_the_give_up_line_of_an_accepted_stop_on_a_declared_role_has_no_refusal_note(
    cause: str, why: str, caplog: pytest.LogCaptureFixture
) -> None:
    # A firmware that took the stop of a role the profile declares refused: the
    # declaration is a statement about a stop that raised, and this one did not.
    line = _give_up_line(
        caplog,
        _declared(Role.xy_stage, Role.focus),
        ctrl_c=cause == "ctrl-c",
        refused=(),
    )
    assert line == f"xy_stage XY: stop after {why}{_STILL_BUSY}"


def test_a_ctrl_c_inside_the_timeouts_give_up_sends_one_stop_and_is_noted_interrupted() -> (
    None
):
    # #224 item 4: the Ctrl-C cut the timeout's give-up, and `_move`'s outer
    # `except` gave up again: two stops, the second one's `accepted` noted. The
    # Ctrl-C is the first here, so nothing says "second".
    profile = _declared(Role.xy_stage)
    micromanager = profile.micromanager.model_copy(update={"device_timeout_ms": 50})
    with _stand(profile.model_copy(update={"micromanager": micromanager})) as (
        core,
        microscope,
    ):
        core.busy_devices = {"XY"}
        stops: list[str] = []

        def cut(label: str) -> None:
            stops.append(label)
            raise KeyboardInterrupt("Ctrl-C in the stop")

        core.stop = cut  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            del core.stop
        # Sent once, before the stand closes; `close()` sends the retry.
        assert stops == ["XY"]
    assert str(interrupted.value) == "Ctrl-C in the stop"
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.interrupted),
    )


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
    assert stop_notes(second) == (StopNote("xy_stage XY", StopOutcome.interrupted),)


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


def test_a_stop_then_a_timeout_on_a_declared_stand_never_says_it_was_stopped() -> None:
    # FM-117: the stop was refused, so "it was stopped" would be false
    # (adversarial review of #204).
    profile = _declared(Role.xy_stage)
    micromanager = profile.micromanager.model_copy(update={"device_timeout_ms": 50})
    with _stand(profile.model_copy(update={"micromanager": micromanager})) as (
        core,
        microscope,
    ):
        stage = microscope.require(XYStage)
        polls: list[str] = []

        def busy(label: str) -> bool:
            polls.append(label)
            if len(polls) == 1:
                with pytest.raises(StopRefusedError):
                    stage.stop()
            return True

        core.deviceBusy = busy  # type: ignore[method-assign]
        try:
            with pytest.raises(MotionStoppedError) as stopped:
                stage.move_to_um(10.0, 0.0)
        finally:
            del core.deviceBusy
    assert str(stopped.value) == (
        "`xy_stage XY`: a stop was requested during the move, but the profile "
        "declares that it refuses its stop ([safety] stop_refused_by), so it may "
        "have run to its target; it still read busy at the move's 0.05 s timeout, "
        "and on the timeout the stand refuses its stop, as the profile declares "
        "([safety] stop_refused_by), so it runs to its target; read the position "
        "before moving on"
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


# --- the emergency stop and close() (m2 §11.5) ---------------------------------


def test_the_emergency_stop_on_a_declared_stand_raises_stop_refused_error_naming_what_it_did() -> (
    None
):
    with _stand() as (_, microscope):
        with pytest.raises(StopRefusedError) as refused:
            microscope.stop()
        assert microscope.state().halted
    assert refused.value.refused == ("xy_stage XY", "z Z")
    assert str(refused.value) == (
        "the emergency stop halted the microscope and switched autofocus Autofocus "
        "off, but xy_stage XY and z Z refused the stop, as the profile declares "
        "([safety] stop_refused_by): a move under way runs to its target. The "
        "microscope stays halted: stop the stage with the stand's own controls if "
        "it must not finish its move, then call `resume()`"
    )


def test_the_emergency_stop_with_a_refusal_and_a_failure_raises_hardware_error_listing_both() -> (
    None
):
    # Any failure keeps today's text, with the declared refusal added.
    with _stand(_declared(Role.xy_stage), refused=("XY",)) as (core, microscope):
        core.failing["Z"] = RuntimeError("device fault")
        try:
            with pytest.raises(HardwareError) as failed:
                microscope.stop()
        finally:
            core.failing.clear()
    assert type(failed.value) is HardwareError
    message = str(failed.value)
    assert message.startswith(
        "the emergency stop failed for ZStage 'Z' (HardwareError(\"focus: stop('Z') "
        "failed (device fault); check the device"
    )
    assert message.endswith(
        "; xy_stage XY refused the stop, as the profile declares ([safety] "
        "stop_refused_by), so a move under way runs to its target. The microscope "
        "stays halted: stop the stage at the stand if it still moves, then call "
        "`resume()`"
    )


def test_every_stop_is_still_sent_in_order_on_a_declared_stand() -> None:
    # FM-17, FM-62: the declaration changes no send and no order (m2 §11.2).
    with _stand() as (core, microscope):
        with pytest.raises(StopRefusedError):
            microscope.stop()
        assert core.log == ["enableContinuousFocus(False)", "stop('XY')", "stop('Z')"]


def test_the_emergency_stop_names_only_what_it_switched_off_beside_a_refusal() -> None:
    # Z took its stop: it is neither "switched off" nor refused.
    with (
        _stand(_declared(Role.xy_stage), refused=("XY",)) as (_, microscope),
        pytest.raises(StopRefusedError) as refused,
    ):
        microscope.stop()
    assert refused.value.refused == ("xy_stage XY",)
    assert str(refused.value).startswith(
        "the emergency stop halted the microscope and switched autofocus Autofocus "
        "off, but xy_stage XY refused the stop, as the profile declares"
    )


def _left_moving(core: FakeCore, microscope: Microscope) -> None:
    """A Ctrl-C during an XY move that stays busy: the motion outlives its action."""
    core.busy_devices = {"XY"}
    undo = _interrupt_the_first_busy_poll(core)
    try:
        with pytest.raises(KeyboardInterrupt):
            microscope.require(XYStage).move_to_um(10.0, 0.0)
    finally:
        undo()
    assert microscope.state().moving == ("xy_stage XY",)
    core.log.clear()


def test_close_logs_stopped_only_for_the_stops_that_were_accepted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # #206: "the stages were stopped" was written whatever each stop did.
    with _stand(_declared(Role.xy_stage), refused=("XY",)) as (core, microscope):
        _left_moving(core, microscope)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            microscope.close()
        assert core.log == [
            "enableContinuousFocus(False)",
            "stop('XY')",
            "stop('Z')",
            "unloadAllDevices()",
        ]
    assert caplog.messages[-1] == (
        "close: xy_stage XY still moving; stopped: autofocus Autofocus off, z Z; "
        "refused, as the profile declares ([safety] stop_refused_by), so a move "
        "under way runs to its target: xy_stage XY"
    )
    assert not any("the stages were stopped" in line for line in caplog.messages)


def test_close_on_a_declared_stand_says_the_move_runs_to_its_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Unloading does not wait for the refused move: the stand finishes it.
    with _stand() as (core, microscope):
        _left_moving(core, microscope)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            microscope.close()
        assert core.log[-1] == "unloadAllDevices()"
    assert caplog.messages[-1] == (
        "close: xy_stage XY still moving; stopped: autofocus Autofocus off; "
        "refused, as the profile declares ([safety] stop_refused_by), so a move "
        "under way runs to its target: xy_stage XY, z Z"
    )


def _ti2_stand() -> tuple[FakeCore, Microscope]:
    """The Ti2-E's layout, refusing every stop as measured (#208), and declaring it."""
    core = FakeCore(
        [
            FakeDevice("ZDrive", DeviceType.Stage),
            FakeDevice("PFS", DeviceType.AutoFocus),
            FakeDevice("PFSOffset", DeviceType.Stage),
            FakeDevice("XYStage", DeviceType.XYStage),
        ],
        xy_stage="XYStage",
        focus="ZDrive",
        autofocus="PFS",
    )
    core.stop_refused = {"XYStage", "ZDrive", "PFSOffset"}
    profile = Profile(
        microscope=MicroscopeSection(name="ti2e-fake"),
        roles=RolesSection.model_validate(
            {"assign": {"autofocus": "PFS", "autofocus_offset": "PFSOffset"}}
        ),
        safety=SafetySection.model_validate(
            {"stop_refused_by": ["xy_stage", "focus", "autofocus_offset"]}
        ),
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    core.log.clear()
    return core, microscope


def test_the_emergency_stop_on_the_ti2_names_the_pfs_offset_too() -> None:
    # The offset device is stopped by its role, with the profile's declaration.
    core, microscope = _ti2_stand()
    try:
        with pytest.raises(StopRefusedError) as refused:
            microscope.stop()
        assert core.log == [
            "enableContinuousFocus(False)",
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
    finally:
        microscope.close()
    assert refused.value.refused == (
        "autofocus_offset PFSOffset",
        "xy_stage XYStage",
        "z ZDrive",
    )
    assert str(refused.value).startswith(
        "the emergency stop halted the microscope and switched autofocus PFS off, "
        "but autofocus_offset PFSOffset, xy_stage XYStage and z ZDrive refused the "
        "stop, as the profile declares"
    )


def test_close_on_the_ti2_names_an_accepted_offset_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A firmware whose PFS offset took its stop: close() says it was stopped.
    core, microscope = _ti2_stand()
    core.stop_refused = {"XYStage", "ZDrive"}
    try:
        core.busy_devices = {"XYStage"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt):
                microscope.require(XYStage).move_to_um(10.0, 0.0)
        finally:
            undo()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            microscope.close()
    finally:
        microscope.close()
    assert caplog.messages[-1] == (
        "close: xy_stage XYStage still moving; stopped: autofocus PFS off, "
        "autofocus_offset PFSOffset; refused, as the profile declares ([safety] "
        "stop_refused_by), so a move under way runs to its target: xy_stage "
        "XYStage, z ZDrive"
    )


class _Mover:
    """``target`` in a thread: whether it ended, and the type of what it raised."""

    def __init__(self, target: Callable[[], object]) -> None:
        self.raised: str | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._target()
        except BaseException as exc:
            # The type's name, not the error: its traceback holds this frame.
            self.raised = type(exc).__name__

    def join(self) -> None:
        self._thread.join(timeout=5.0)
        assert not self._thread.is_alive(), "thread did not finish: deadlock?"


def _wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + 5.0
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def _offset_move_with_pfs_on(core: FakeCore, microscope: Microscope) -> _Mover:
    """A direct offset move with PFS on, waiting for the offset device to go idle.

    PFS on, the autofocus follows its offset and drives Z, so the compound
    links Z's stop to the offset's (m2 §3.3.1). The offset device reads busy
    until the test clears it, since the Ti2-E refuses its stop (#208).
    """
    core.enableContinuousFocus(True)
    core.busy_devices = {"PFSOffset"}
    mover = _Mover(lambda: microscope.require(Autofocus).set_offset_native(10.0))
    _wait_until(
        lambda: _LINK_NAME in microscope.state().moving, "the offset move to wait"
    )
    return mover


def test_z_stop_during_an_offset_move_on_the_ti2_names_z_and_the_offset() -> None:
    # #224 item 1: the link's refusal (the offset's) won over Z's own, so
    # `.refused` missed `z ZDrive`.
    core, microscope = _ti2_stand()
    try:
        mover = _offset_move_with_pfs_on(core, microscope)
        try:
            with pytest.raises(StopRefusedError) as refused:
                microscope.require(ZStage).stop()
        finally:
            core.busy_devices.clear()  # the move ends; the stage was never stopped
            mover.join()
    finally:
        microscope.close()
    assert refused.value.refused == (_LINK_NAME, "z ZDrive")
    assert mover.raised == "MotionStoppedError"


def test_close_from_another_thread_lists_each_refused_stage_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # `close()` sends the offset's stop, then Z's, whose link stops the offset
    # again: the line named the offset twice and, by the link winning, Z not at
    # all (#224 item 1, measured).
    core, microscope = _ti2_stand()
    real_stop = core.stop

    def stop(label: str) -> None:
        # Z's stop is the last one `close()` sends; the offset device, which
        # refuses its own, reads idle after it (the fake raises for a refused
        # label, hence the `finally`), or `close()` would wait 60 s for the lock.
        try:
            real_stop(label)
        finally:
            if label == "ZDrive":
                core.busy_devices.discard("PFSOffset")

    try:
        mover = _offset_move_with_pfs_on(core, microscope)
        core.stop = stop  # type: ignore[method-assign]
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                microscope.close()
        finally:
            del core.stop
            core.busy_devices.clear()
            mover.join()
    finally:
        microscope.close()
    lines = [line for line in caplog.messages if line.startswith("close: ")]
    assert len(lines) == 1, lines
    assert lines[0].endswith(
        "so a move under way runs to its target: "
        "autofocus_offset PFSOffset, xy_stage XYStage, z ZDrive"
    )


def test_ctrl_c_during_an_offset_move_on_the_ti2_notes_the_declared_refusal() -> None:
    # The autofocus waits for its offset device's move with the same Motion
    # the emergency stop builds, declaration included (m2 §11.4).
    core, microscope = _ti2_stand()
    try:
        core.busy_devices = {"PFSOffset"}
        undo = _interrupt_the_first_busy_poll(core)
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(Autofocus).set_offset_native(10.0)
        finally:
            undo()
        assert "stop('PFSOffset')" in core.log
    finally:
        microscope.close()
    assert stop_notes(interrupted.value) == (
        StopNote("autofocus_offset PFSOffset", StopOutcome.refused),
    )


def test_close_counts_an_undeclared_refusal_as_a_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _stand(Profile.demo(), refused=("XY",)) as (core, microscope):
        _left_moving(core, microscope)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            microscope.close()
    warning = (
        "close: xy_stage XY still moving; stopped: autofocus Autofocus off, z Z; "
        "not stopped: 1 failed"
    )
    assert warning in caplog.messages
    after = caplog.messages[caplog.messages.index(warning) + 1 :]
    assert after[0].startswith("close: the stop failed for XYStage 'XY' (")


# --- what the registry still tracks when the stand is released (m2 §11.11 E6) -----


def _xy_stand_that_gives_up_after_50_ms(
    *, refuses_stop: bool = True
) -> tuple[FakeCore, Microscope]:
    """The demo stand, its moves giving up after 50 ms; XY refuses its stop, declared, by default."""
    core = FakeCore.demo_like()
    if refuses_stop:
        core.stop_refused = {"XY"}
    profile = _declared(Role.xy_stage) if refuses_stop else Profile.demo()
    micromanager = profile.micromanager.model_copy(update={"device_timeout_ms": 50})
    stand = Microscope.from_core(
        cast("CMMCorePlus", core),
        profile.model_copy(update={"micromanager": micromanager}),
    )
    return core, stand


def _ctrl_c_in_a_wait_on_a_move_left_tracked(
    core: FakeCore, microscope: Microscope
) -> None:
    """Inside the ``with``: a move times out and stays tracked, then a Ctrl-C lands in ``wait()``.

    The earlier give-up was refused (declared), so the stage reads busy and
    stays tracked. The wait is no move of its own: it gives up on nothing and
    notes nothing, which is what #224 item 6 measured (notes ``()``,
    ``moving()`` ``("xy_stage XY",)``).
    """
    stage = microscope.require(XYStage)
    core.busy_devices = {"XY"}
    with pytest.raises(HardwareError):
        stage.move_to_um(10.0, 0.0)
    assert microscope.state().moving == ("xy_stage XY",)
    real, polled = core.deviceBusy, list[str]()

    def busy(label: str) -> bool:
        polled.append(label)
        if len(polled) == 3:
            raise KeyboardInterrupt
        return real(label)

    core.deviceBusy = busy  # type: ignore[method-assign]
    try:
        stage.wait(5.0)
    finally:
        del core.deviceBusy


def test_a_ctrl_c_in_a_wait_on_a_stage_still_tracked_is_noted_when_the_stand_is_released() -> (
    None
):
    core, microscope = _xy_stand_that_gives_up_after_50_ms()
    with pytest.raises(KeyboardInterrupt) as interrupted, microscope:
        _ctrl_c_in_a_wait_on_a_move_left_tracked(core, microscope)
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.still_moving),
    )


def _ctrl_c_in_a_move_left_tracked(core: FakeCore, microscope: Microscope) -> None:
    """A Ctrl-C in a move on the declared XY: its refused give-up leaves it tracked."""
    core.busy_devices = {"XY"}
    undo = _interrupt_the_first_busy_poll(core)
    try:
        microscope.require(XYStage).move_to_um(10.0, 0.0)
    finally:
        undo()


def _a_tool_error_after_a_caught_ctrl_c(core: FakeCore, microscope: Microscope) -> None:
    with pytest.raises(KeyboardInterrupt):  # a tool that caught an earlier Ctrl-C
        _ctrl_c_in_a_move_left_tracked(core, microscope)
    assert microscope.state().moving == ("xy_stage XY",)
    raise ValueError("a tool's own error")


def test_a_move_that_noted_its_own_give_up_is_not_noted_again_at_release() -> None:
    # The stage reads busy and its stop was refused, so it is still tracked at
    # the release: only its own note keeps `still_moving` from being added.
    core, microscope = _xy_stand_that_gives_up_after_50_ms()
    with pytest.raises(KeyboardInterrupt) as interrupted, microscope:
        _ctrl_c_in_a_move_left_tracked(core, microscope)
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.refused),
    )


def test_an_error_that_is_not_an_interrupt_leaves_the_block_with_no_note() -> None:
    # `still_moving` is for a Ctrl-C only: any other error has its own message,
    # and the Ctrl-C line is the one that reads the notes.
    core, microscope = _xy_stand_that_gives_up_after_50_ms()
    with pytest.raises(ValueError, match="a tool's own error") as failed, microscope:
        _a_tool_error_after_a_caught_ctrl_c(core, microscope)
    assert stop_notes(failed.value) == ()


def test_a_close_that_raises_still_leaves_the_still_moving_note() -> None:
    # The note is added in a `finally`: a close() that raises (a busy lock, a
    # call from inside an action) loses none of it.
    core, microscope = _xy_stand_that_gives_up_after_50_ms()

    def failing_close() -> None:
        raise RuntimeError("close failed")

    microscope.close = failing_close  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="close failed") as failed, microscope:
            _ctrl_c_in_a_wait_on_a_move_left_tracked(core, microscope)
    finally:
        del microscope.close
        microscope.close()
    # The Ctrl-C is the context of close()'s error, and `stop_notes` reads it.
    assert isinstance(failed.value.__context__, KeyboardInterrupt)
    assert stop_notes(failed.value) == (
        StopNote("xy_stage XY", StopOutcome.still_moving),
    )


def test_the_still_moving_note_is_read_before_close_stops_the_stage() -> None:
    # close() sends the stop and, the device then reading idle, drops the
    # registration: read after it, the registry would say nothing was tracked.
    core, microscope = _xy_stand_that_gives_up_after_50_ms(refuses_stop=False)
    real_stop, xy_stops = core.stop, list[str]()

    def stop(label: str) -> None:
        real_stop(label)
        if label == "XY":
            xy_stops.append(label)
            if len(xy_stops) > 1:  # close()'s, after the timeout give-up's
                core.busy_devices.discard("XY")

    core.stop = stop  # type: ignore[method-assign]
    try:
        with pytest.raises(KeyboardInterrupt) as interrupted, microscope:
            _ctrl_c_in_a_wait_on_a_move_left_tracked(core, microscope)
    finally:
        del core.stop
    assert len(xy_stops) == 2  # the give-up's, and close()'s that released it
    assert core.busy_devices == set()
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.still_moving),
    )
