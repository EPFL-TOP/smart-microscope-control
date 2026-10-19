"""``ObjectiveTurret`` on the Ti2-E's layout (m2 §8.3, #14).

The layout is ``test_mm_channels.ti2_core()``: ``Nosepiece`` at
``2-Plan Fluor 10x``, ``PFS`` in the AutoFocus slot. A turn under PFS
switches it off first and leaves it off (FM-88); a turn needs
``confirm=True`` (FM-14). Messages are compared whole (FM-45).
"""

from __future__ import annotations

import logging
import re
import weakref
from typing import TYPE_CHECKING, cast

import pytest
from test_mm_channels import (
    JOIN_S,
    Thread,
    lock_held_by_a_snap,
    ti2,
    ti2_core,
    ti2_profile,
    wait_for,
)

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret, XYStage
from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionInProgressError,
    SafetyRefusedError,
)
from smc.hardware.profile import CameraSection

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

OFF = "enableContinuousFocus(False)"
TEN, FORTY = "2-Plan Fluor 10x", "3-Plan Apo LmbdD0.25 40x"
TURN_40 = f"setStateLabel('Nosepiece', {FORTY!r})"
REFUSED = (
    "turning the objective turret can drive an objective into a loaded plate "
    "(objective_turret 'Nosepiece') — pass confirm=True once the stage is clear"
)
LEFT_OFF = (
    "objective_turret: autofocus 'PFS' switched off before the turret turns; it is "
    "not engaged again (the new objective's offset is not known); engage() it once "
    "in focus"
)


def test_rotating_the_turret_needs_confirmation() -> None:
    for dry_run in (False, True):
        core = ti2_core()
        with ti2(core=core, dry_run=dry_run) as (_, microscope):
            turret = microscope.require(ObjectiveTurret)
            # The refusal comes first: it reads nothing, so a failing turret
            # still gets the refusal, not a fault.
            core.failing["Nosepiece"] = RuntimeError("must not be read")
            with pytest.raises(SafetyRefusedError) as refused:
                turret.select(FORTY)
            core.failing.clear()
            assert str(refused.value) == REFUSED
            assert (
                refused.value.how_to_force
                == "pass confirm=True once the stage is clear"
            )
            assert core.log == []
            assert turret.current() == TEN


def test_without_the_profile_key_no_confirmation_is_needed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = ti2_profile(safety={"turret_requires_confirm": False})
    with ti2(profile) as (core, microscope):
        with caplog.at_level(logging.INFO):
            assert microscope.require(ObjectiveTurret).select(FORTY) == FORTY
        assert core.log == [TURN_40]
        assert (
            "objective_turret: no confirmation required ([safety] "
            "turret_requires_confirm = false)"
        ) in caplog.messages


def test_an_unknown_objective_is_refused_before_anything_is_sent() -> None:
    with ti2() as (core, microscope):
        message = (
            "objective_turret 'Nosepiece' has no position '60x'; its labels are: "
            "'1-Plan Fluor 4x', '2-Plan Fluor 10x', '3-Plan Apo LmbdD0.25 40x', "
            "'4-', '5-', '6-'"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            microscope.require(ObjectiveTurret).select("60x", confirm=True)
        assert core.log == []


def test_labels_and_current_read_the_configurations_names() -> None:
    with ti2() as (_, microscope):
        turret = microscope.require(ObjectiveTurret)
        assert turret.labels() == [
            "1-Plan Fluor 4x",
            TEN,
            FORTY,
            "4-",
            "5-",
            "6-",
        ]
        assert turret.current() == TEN
        assert microscope.state().objective == TEN


def test_selecting_the_current_objective_sends_nothing_and_leaves_pfs_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with ti2() as (core, microscope):
        core.enableContinuousFocus(True)
        core.log.clear()
        with caplog.at_level(logging.INFO):
            assert microscope.require(ObjectiveTurret).select(TEN, confirm=True) == TEN
        assert core.log == []
        assert core.isContinuousFocusEnabled() is True
        assert f"objective_turret: already at {TEN}" in caplog.messages


def test_turning_the_turret_switches_pfs_off_first_and_never_back_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with ti2() as (core, microscope):
        core.enableContinuousFocus(True)
        core.log.clear()
        with caplog.at_level(logging.WARNING):
            assert microscope.require(ObjectiveTurret).select(FORTY, confirm=True) == (
                FORTY
            )
        assert core.log == [OFF, TURN_40]
        assert core.isContinuousFocusEnabled() is False
        assert caplog.messages == [LEFT_OFF]


def test_turning_with_pfs_off_sends_the_turn_alone() -> None:
    with ti2() as (core, microscope):
        assert microscope.require(ObjectiveTurret).select(FORTY, confirm=True) == FORTY
        assert core.log == [TURN_40]


def test_an_unreadable_autofocus_counts_as_engaged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # FM-19: read as off, a PFS that is on would follow the turn and drive Z.
    with ti2() as (core, microscope):
        core.enableContinuousFocus(True)
        core.log.clear()

        def unreadable() -> bool:
            raise RuntimeError("PFS: no answer")

        core.isContinuousFocusEnabled = unreadable  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING):
            microscope.require(ObjectiveTurret).select(FORTY, confirm=True)
        del core.isContinuousFocusEnabled
        assert core.log == [OFF, TURN_40]
        assert caplog.messages == [
            "objective_turret: autofocus 'PFS': its state could not be read before the "
            "turret turns (autofocus: isContinuousFocusEnabled() failed (PFS: no "
            "answer); check the device: power, cable, another program holding it); "
            "counted as engaged, so it is switched off",
            LEFT_OFF,
        ]


def test_a_failed_switch_off_leaves_the_turret_where_it_is() -> None:
    with ti2() as (core, microscope):
        core.enableContinuousFocus(True)
        core.log.clear()

        def stuck(enable: bool) -> None:
            raise RuntimeError("PFS: stuck")

        core.enableContinuousFocus = stuck  # type: ignore[method-assign]
        with pytest.raises(HardwareError) as failed:
            microscope.require(ObjectiveTurret).select(FORTY, confirm=True)
        del core.enableContinuousFocus
        assert str(failed.value).startswith(
            f"objective_turret: not turned to {FORTY!r}: autofocus 'PFS' could not be "
            f"switched off first (autofocus: enableContinuousFocus(False) failed"
        )
        assert core.getStateLabel("Nosepiece") == TEN
        assert TURN_40 not in core.log


def test_a_turret_still_busy_at_the_deadline_stays_registered_until_resume() -> None:
    with ti2(ti2_profile(device_timeout_ms=100)) as (core, microscope):
        core.busy_devices.add("Nosepiece")
        turret = microscope.require(ObjectiveTurret)
        with pytest.raises(DeviceTimeoutError) as timed_out:
            turret.select(FORTY, confirm=True)
        assert str(timed_out.value).startswith(
            "objective_turret Nosepiece is still busy after 0.1 s; "
        )
        assert core.log == [TURN_40]  # no stop: a turret cannot be stopped
        assert microscope.state().moving == ("objective_turret Nosepiece",)
        with pytest.raises(MotionInProgressError):
            microscope.require(XYStage).move_to_um(1.0, 1.0)
        microscope.resume()
        assert microscope.state().moving == ()
        core.busy_devices.clear()


def test_another_action_is_refused_while_the_turret_turns() -> None:
    with ti2() as (core, microscope):
        core.busy_devices.add("Nosepiece")
        turret = microscope.require(ObjectiveTurret)
        turner = Thread(lambda: turret.select(FORTY, confirm=True))
        wait_for(
            lambda: microscope.state().moving == ("objective_turret Nosepiece",),
            "the turn",
        )
        with pytest.raises(MotionInProgressError):
            microscope.require(XYStage).move_to_um(1.0, 1.0)
        # Reads still run (FM-16).
        assert turret.current() == FORTY
        core.busy_devices.clear()
        turner.join()
        assert (turner.error, turner.result) == (None, FORTY)


def test_a_halt_while_the_turn_waits_for_the_lock_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-89: the compound records the halt when it is called; a halt while
    # it waits for the lock must not let the switch-off or the turn out.
    with ti2() as (core, microscope):
        turret = microscope.require(ObjectiveTurret)
        core.enableContinuousFocus(True)
        core.log.clear()
        with lock_held_by_a_snap(core, microscope, monkeypatch) as queued:
            turner = Thread(lambda: turret.select(FORTY, confirm=True))
            assert queued.wait(JOIN_S)
            microscope._executor.halt()
        turner.join()
        assert isinstance(turner.error, MicroscopeHaltedError)
        assert core.log == ["snapImage()"]
        assert core.getStateLabel("Nosepiece") == TEN
        assert core.isContinuousFocusEnabled() is True
        microscope.resume()


def test_dry_run_turns_nothing_and_switches_nothing_off() -> None:
    with ti2(dry_run=True) as (core, microscope):
        core.enableContinuousFocus(True)
        core.log.clear()
        assert microscope.require(ObjectiveTurret).select(FORTY, confirm=True) == FORTY
        assert core.log == []
        assert core.isContinuousFocusEnabled() is True
        assert core.getStateLabel("Nosepiece") == TEN


def test_a_turret_fault_is_a_hardware_error_naming_the_role() -> None:
    # FM-79: MMCore's bare RuntimeError names neither the role nor the call.
    with ti2() as (core, microscope):
        turret = microscope.require(ObjectiveTurret)

        def jammed(label: str, text: str) -> None:
            raise RuntimeError("Nosepiece: jammed")

        core.setStateLabel = jammed  # type: ignore[method-assign]
        with pytest.raises(HardwareError) as failed:
            turret.select(FORTY, confirm=True)
        del core.setStateLabel
        assert str(failed.value) == (
            f"objective_turret: setStateLabel('Nosepiece', {FORTY!r}) failed "
            f"(Nosepiece: jammed); check the device: power, cable, another program "
            f"holding it"
        )
        core.failing["Nosepiece"] = RuntimeError("Nosepiece: no answer")
        with pytest.raises(HardwareError, match=r"^objective_turret: getStateLabel"):
            turret.current()
        core.failing.clear()


def test_pixel_size_comes_from_the_profile_per_objective() -> None:
    profile = ti2_profile().model_copy(
        update={"camera": CameraSection(pixel_size_um={TEN: 0.65, FORTY: 0.1625})}
    )
    with ti2(profile) as (_, microscope):
        turret = microscope.require(ObjectiveTurret)
        assert turret.pixel_size_um() == 0.65  # the current objective
        assert turret.pixel_size_um(FORTY) == 0.1625
        assert turret.pixel_size_um("1-Plan Fluor 4x") == 0.0  # unknown, not guessed
    with ti2() as (_, microscope):
        assert microscope.require(ObjectiveTurret).pixel_size_um() == 0.0


def test_a_turret_outliving_its_microscope_refuses_the_turn_and_sends_nothing(
    no_cyclic_gc: None,
) -> None:
    # FM-92: the autofocus is looked up through the facade; a facade that is
    # gone must not read as "no autofocus" and turn under PFS.
    core = ti2_core()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), ti2_profile())
    turret = microscope.require(ObjectiveTurret)
    core.enableContinuousFocus(True)
    core.log.clear()
    alive = weakref.ref(microscope)
    del microscope
    assert alive() is None
    with pytest.raises(
        HardwareError, match=r"^ObjectiveTurret belongs to a Microscope"
    ):
        turret.select(FORTY, confirm=True)
    assert core.log == []
