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
    fake_clock,
    lock_held_by_a_snap,
    ti2,
    ti2_core,
    ti2_profile,
    wait_for,
)

import smc.hardware.backends.mm as mm
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret, XYStage, ZStage
from smc.hardware.errors import (
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionInProgressError,
    SafetyRefusedError,
)
from smc.hardware.profile import CameraSection, ObjectivesSection, Profile
from smc.hardware.safety import POLL_INTERVAL_S
from smc.testing import FakeCore

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
        calls = _count_positions(core)
        assert microscope.require(ObjectiveTurret).select(FORTY, confirm=True) == FORTY
        del core.getPosition
        assert core.log == []
        assert core.isContinuousFocusEnabled() is True
        assert core.getStateLabel("Nosepiece") == TEN
        assert calls == []  # nothing turned, so Z is not read around it


def test_a_turret_that_does_not_follow_reports_where_it_is(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The readback, not the request: a tool that took the requested label
    # would scale its images with the wrong objective's pixel size. The label
    # is read again until the window ends (m2 §8.11 D3), shortened here.
    monkeypatch.setattr(mm, "FRESH_READ_WINDOW_S", 0.05)
    with ti2() as (core, microscope):

        def ignored(label: str, text: str) -> None:
            core.log.append(f"setStateLabel({label!r}, {text!r})")

        core.setStateLabel = ignored  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING):
            landed = microscope.require(ObjectiveTurret).select(FORTY, confirm=True)
        del core.setStateLabel
        assert landed == TEN
        assert core.log == [TURN_40]
        assert caplog.messages == [
            f"objective_turret 'Nosepiece': reads {TEN!r} after 0.05 s, although "
            f"{FORTY!r} was set; the read may predate the turn, or the turret did "
            f"not turn"
        ]


def test_the_turns_label_is_read_again_until_it_is_the_label_asked_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # m2 §8.11 D3: whether getStateLabel reads the new label at the return of
    # a turn is not verified; a label that lags must not come back as the result.
    core = ti2_core()
    core.make_reads_stale("Nosepiece", "Label", reads=3)
    with ti2(core=core) as (_, microscope):
        with caplog.at_level(logging.INFO):
            assert (
                microscope.require(ObjectiveTurret).select(FORTY, confirm=True) == FORTY
            )
        assert [m for m in caplog.messages if "FM-188" in m] == [
            f"objective_turret 'Nosepiece': read {TEN!r} right after the turn, then "
            f"{FORTY!r}: a read right after a set may be the previous value (FM-188)"
        ]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_label_that_is_fresh_at_once_is_read_once_and_logs_nothing_about_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The turn sleeps for Z's settle (D2), so what a label read fresh at once
    # costs is counted in reads: the check that the turret is not there yet, and
    # the readback.
    with ti2() as (core, microscope):
        reads: list[str] = []
        real = core.getStateLabel

        def counted(label: str) -> str:
            reads.append(label)
            return real(label)

        core.getStateLabel = counted  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            assert (
                microscope.require(ObjectiveTurret).select(FORTY, confirm=True) == FORTY
            )
        del core.getStateLabel
        assert reads == ["Nosepiece", "Nosepiece"]
        assert [m for m in caplog.messages if "FM-188" in m] == []


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


def _profile_with_offsets() -> Profile:
    # The 10x is the reference; the 4x was measured, the 40x was not (m2 §13.3).
    return ti2_profile().model_copy(
        update={
            "objectives": ObjectivesSection.model_validate(
                {
                    "reference": TEN,
                    "offsets": {
                        "1-Plan Fluor 4x": {
                            "xy_offset_um": [12.5, -4.0],
                            "z_offset_um": -8.7,
                            "measured": "2026-11-12, session 3 (#400)",
                        }
                    },
                }
            )
        }
    )


def test_offset_comes_from_the_profile_and_sends_nothing() -> None:
    for dry_run in (False, True):
        with ti2(_profile_with_offsets(), dry_run=dry_run) as (core, microscope):
            turret = microscope.require(ObjectiveTurret)
            four = turret.offset("1-Plan Fluor 4x")
            assert (four.objective, four.reference) == ("1-Plan Fluor 4x", TEN)
            assert (four.xy_um, four.z_um) == ((12.5, -4.0), -8.7)
            assert four.measured == "2026-11-12, session 3 (#400)"
            ten = turret.offset(TEN)
            assert (ten.xy_um, ten.z_um) == ((0.0, 0.0), 0.0)
            forty = turret.offset(FORTY)  # on the turret, not measured
            assert (forty.xy_um, forty.z_um, forty.measured) == (None, None, "")
            assert core.log == []


def test_offset_without_a_reference_in_the_profile_is_unknown_for_every_label() -> None:
    with ti2() as (_, microscope):
        turret = microscope.require(ObjectiveTurret)
        for label in turret.labels():
            offset = turret.offset(label)
            assert (offset.reference, offset.xy_um, offset.z_um) == (None, None, None)


def test_offset_answers_while_another_action_holds_the_lock_and_after_a_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # m2 §13.4: no lock, no executor. A snap holds the lock; a read through the
    # executor would queue behind it.
    with ti2(_profile_with_offsets()) as (core, microscope):
        turret = microscope.require(ObjectiveTurret)
        with lock_held_by_a_snap(core, microscope, monkeypatch) as queued:
            asker = Thread(lambda: turret.offset("1-Plan Fluor 4x"))
            asker.join()
            assert asker.error is None
            assert getattr(asker.result, "xy_um", None) == (12.5, -4.0)
            assert not queued.is_set()
        microscope.stop()
        core.log.clear()
        assert turret.offset("1-Plan Fluor 4x").xy_um == (12.5, -4.0)
        assert core.log == []
        microscope.resume()


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


# --- Z around a turn (m2 §8.11 D2, #330) ------------------------------------------

FOUR = "1-Plan Fluor 4x"
#: A profile whose core timeout bounds the settle: a settle that never ends does
#: not wait for the default 60 s, in a test or in a mutant of it.
BOUNDED = ti2_profile(device_timeout_ms=2000)


def _core_at(z_um: float, delta_um: float, *, escape: bool = True) -> FakeCore:
    """The Ti2-E layout at ``z_um`` whose turn to the 4x moves Z (session 2, m2 §8.9).

    With ``escape``, Z reads 500.0 for the first three reads after the turn, as
    the call returned while the stand was still moving it.
    """
    core = ti2_core()
    core.setPosition("ZDrive", z_um)
    if escape:
        core.move_z_on_turn(FOUR, delta_um, escape_um=500.0, escape_reads=3)
    else:
        core.move_z_on_turn(FOUR, delta_um)
    return core


def _turned(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [m for m in caplog.messages if m.startswith("objective_turret: turned to")]


def _z_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every line the turn wrote about Z, whatever it said."""
    return [
        m
        for m in caplog.messages
        if m.startswith("objective_turret: turned to") or "Z " in m
    ]


def _count_positions(core: FakeCore) -> list[str]:
    """Count the ``getPosition`` calls from now on; the caller ``del``s the hook (FM-92)."""
    calls: list[str] = []
    real = core.getPosition

    def counted(label: str | None = None) -> float:
        calls.append(str(label))
        return real(label)

    core.getPosition = counted  # type: ignore[method-assign]
    return calls


def test_a_turn_reads_z_before_and_once_still_after_and_logs_what_the_stand_did(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Session 2: Z read 561.52 µm before, 500.0 µm when the call returned, and
    # settled at 570.22 µm. A report that read Z once at the return would say 500.00.
    # The clock is the test's: the escape lasts three reads whatever the real one does.
    clock = fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70)
    with ti2(BOUNDED, core=core) as (_, microscope):
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        assert landed == FOUR
        # Z is read every poll interval until it has been still for the window: a
        # settle that did not sleep would read as fast as it can, holding the lock.
        assert clock.sleeps
        assert set(clock.sleeps) == {POLL_INTERVAL_S}
        assert _z_lines(caplog) == [
            "objective_turret: turned to '1-Plan Fluor 4x'; the stand moved Z from "
            "561.52 µm to 570.22 µm (+8.70 µm)"
        ]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        # No Z move of the layer's: the turn is all it sent.
        assert core.log == [f"setStateLabel('Nosepiece', {FOUR!r})"]


def test_a_turn_that_leaves_z_within_the_noise_says_z_is_as_before(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Z at rest reads 0.04 µm apart (session 2): that is no move.
    fake_clock(monkeypatch)
    core = _core_at(561.52, 0.04, escape=False)
    with ti2(BOUNDED, core=core) as (_, microscope):
        with caplog.at_level(logging.INFO):
            microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        assert _z_lines(caplog) == [
            "objective_turret: turned to '1-Plan Fluor 4x'; Z is 561.52 µm, as before"
        ]


def test_the_settle_reads_its_constants_at_the_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Both are set after the import, as a test or one day a profile does. Z wobbles
    # by 1 µm between two reads after the turn: still, to a tolerance of 10 µm, but
    # not to the default 0.5 µm, which would never let the settle end (the INFO
    # line, whose own threshold is the same constant, is the other place it is read).
    monkeypatch.setattr(mm, "TURN_Z_STILL_UM", 10.0)
    monkeypatch.setattr(mm, "TURN_Z_SETTLE_S", 0.05)
    clock = fake_clock(monkeypatch)
    core = _core_at(561.52, 0.0, escape=False)
    with ti2(BOUNDED, core=core) as (_, microscope):
        reads: list[int] = []

        def wobbling(label: str | None = None) -> float:
            reads.append(0)
            return (
                561.52 + (len(reads) % 2 == 0) * 1.0
            )  # the first read is the 'before'

        core.getPosition = wobbling  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert _z_lines(caplog) == [
            "objective_turret: turned to '1-Plan Fluor 4x'; Z is 561.52 µm, as before"
        ]
        # The window of 0.05 s, not the default 0.1 s: five or six polls, not ten.
        assert 5 <= len(clock.sleeps) <= 6


def test_a_settle_of_zero_seconds_reports_the_first_read_after_the_turn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The window is read at the call, and 0 s is no wait: the escape position the
    # stand reports at the return of the turn is then what the line says.
    monkeypatch.setattr(mm, "TURN_Z_SETTLE_S", 0.0)
    clock = fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70)
    with ti2(BOUNDED, core=core) as (_, microscope):
        with caplog.at_level(logging.INFO):
            microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        assert _z_lines(caplog) == [
            "objective_turret: turned to '1-Plan Fluor 4x'; the stand moved Z from "
            "561.52 µm to 500.00 µm (-61.52 µm)"
        ]
        assert clock.sleeps == []


def test_a_z_that_never_settles_is_reported_and_the_turn_still_returns_its_label(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70, escape=False)
    with ti2(ti2_profile(device_timeout_ms=100), core=core) as (_, microscope):
        reads: list[int] = []

        def drifting(label: str | None = None) -> float:
            reads.append(0)
            return 561.52 + 5.0 * len(reads)

        core.getPosition = drifting  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert landed == FOUR
        lines = _z_lines(caplog)
        assert len(lines) == 1
        assert re.fullmatch(
            r"objective_turret: Z was still changing 0\.1 s after the turn "
            r"\(\d+\.\d\d µm at the last read\); the stand may still be moving it",
            lines[0],
        )
        assert [r.levelname for r in caplog.records if "Z was" in r.getMessage()] == [
            "WARNING"
        ]


def test_a_z_that_drifts_slowly_is_not_still(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Each read lies 0.2 µm from the one before, within the 0.5 µm tolerance, but
    # the stretch is anchored at its first read: ten reads (0.1 s) drift by 2 µm.
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70, escape=False)
    with ti2(ti2_profile(device_timeout_ms=300), core=core) as (_, microscope):
        reads: list[int] = []

        def drifting(label: str | None = None) -> float:
            reads.append(0)
            return 561.52 + 0.2 * len(reads)

        core.getPosition = drifting  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        lines = _z_lines(caplog)
        assert len(lines) == 1
        assert re.fullmatch(
            r"objective_turret: Z was still changing 0\.3 s after the turn "
            r"\(\d+\.\d\d µm at the last read\); the stand may still be moving it",
            lines[0],
        )


def test_a_halt_during_the_settle_stops_the_wait_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70, escape=False)
    with ti2(BOUNDED, core=core) as (_, microscope):
        real = core.getPosition
        calls: list[int] = []

        def stopping(label: str | None = None) -> float:
            calls.append(0)
            if len(calls) == 2:  # the first read of the settle
                microscope.stop()
            return real(label)

        core.getPosition = stopping  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert landed == FOUR
        assert _z_lines(caplog) == [
            "objective_turret: Z was not waited for: the stand was halted"
        ]
        assert len(calls) == 2  # the settle read no further
        microscope.resume()


def test_a_z_read_that_fails_does_not_fail_the_turn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70)
    with ti2(BOUNDED, core=core) as (_, microscope):
        core.failing["ZDrive"] = RuntimeError("ZDrive: no answer")
        with pytest.raises(HardwareError) as unreadable:
            microscope.require(ZStage).position_um()
        cause = str(unreadable.value)
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        core.failing.clear()
        assert landed == FOUR  # it was sent: a read that informs a report fails nothing
        assert core.getStateLabel("Nosepiece") == FOUR
        assert _z_lines(caplog) == [
            f"objective_turret: Z could not be read around the turn ({cause}); the "
            f"stand moves Z during a turn (m2 §8.9)"
        ]
        assert [r.levelname for r in caplog.records if "Z could" in r.getMessage()] == [
            "WARNING"
        ]


def test_a_z_read_that_fails_after_the_turn_says_so_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70, escape=False)
    with ti2(BOUNDED, core=core) as (_, microscope):
        real = core.getPosition
        calls: list[int] = []

        def breaking(label: str | None = None) -> float:
            calls.append(0)
            if len(calls) == 3:  # the second read of the settle
                raise RuntimeError("ZDrive: no answer")
            return real(label)

        core.getPosition = breaking  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert landed == FOUR
        lines = _z_lines(caplog)
        assert len(lines) == 1
        assert lines[0].startswith(
            "objective_turret: Z could not be read around the turn ("
        )
        assert lines[0].endswith("); the stand moves Z during a turn (m2 §8.9)")


def test_a_z_read_that_fails_only_before_the_turn_still_waits_for_z_and_says_so_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The wait is for the next action, which must not start while the stand moves Z:
    # it comes whether or not Z could be read before the turn. Only the line about
    # what the stand did is lost, and a read that worked is not warned about again.
    clock = fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70)
    with ti2(BOUNDED, core=core) as (_, microscope):
        real = core.getPosition
        calls: list[int] = []

        def failing_once(label: str | None = None) -> float:
            calls.append(0)
            if len(calls) == 1:  # the read before the turn
                raise RuntimeError("ZDrive: no answer")
            return real(label)

        core.getPosition = failing_once  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert landed == FOUR
        assert _z_lines(caplog) == [
            "objective_turret: Z could not be read around the turn (focus: "
            "getPosition('ZDrive') failed (ZDrive: no answer); check the device: "
            "power, cable, another program holding it); the stand moves Z during a "
            "turn (m2 §8.9)"
        ]
        # The settle read past the escape position: Z is where the stand left it.
        assert core.getPosition("ZDrive") == pytest.approx(570.22)
        assert clock.sleeps


def test_a_turn_the_turret_did_not_follow_says_nothing_of_having_turned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The label stays stale for the whole window: the WARNING of the re-read says
    # the turret may not have turned, and "turned to" would contradict it (FM-155).
    # Z is still waited for: the stand may have moved it all the same.
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70, escape=False)
    core.make_reads_stale("Nosepiece", "Label", reads=100_000)
    with ti2(BOUNDED, core=core) as (_, microscope):
        calls = _count_positions(core)
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert landed == TEN
        assert _turned(caplog) == []
        assert [r.levelname for r in caplog.records if "reads" in r.getMessage()] == [
            "WARNING"
        ]
        assert len(calls) > 1  # the read before, and the settle's reads


def test_z_is_read_after_the_switch_off_not_before_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D2: the switch-off comes first, so that what it does to Z (FM-12) is not
    # blamed on the stand's turn in the line that says what the stand did.
    fake_clock(monkeypatch)
    core = _core_at(561.52, 8.70)
    with ti2(BOUNDED, core=core) as (_, microscope):
        core.enableContinuousFocus(True)
        engaged: list[bool] = []
        real = core.getPosition

        def watching(label: str | None = None) -> float:
            engaged.append(core.isContinuousFocusEnabled())
            return real(label)

        core.getPosition = watching  # type: ignore[method-assign]
        microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert engaged
        assert not any(engaged)  # every read, the first included, saw PFS off


def test_a_drive_that_reads_minus_zero_shows_zero() -> None:
    # A drive at the origin may read -0.0; the line shows 0.00, as drive.um_text does.
    assert mm._um(-0.001) == "0.00"
    assert mm._um(-0.004) == "0.00"
    assert mm._um(-0.006) == "-0.01"
    assert mm._um(561.5249) == "561.52"


def test_a_stand_with_no_focus_drive_turns_and_says_nothing_about_z(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with ti2(BOUNDED) as (core, microscope):
        turret = microscope.require(ObjectiveTurret)
        real = microscope._lookup

        def lookup(capability: type) -> object | None:
            return None if capability is ZStage else real(capability)

        monkeypatch.setattr(microscope, "_lookup", lookup)
        calls = _count_positions(core)
        with caplog.at_level(logging.INFO):
            assert turret.select(FORTY, confirm=True) == FORTY
        del core.getPosition
        assert calls == []
        assert _z_lines(caplog) == []


def test_a_turn_that_raises_logs_nothing_about_z_and_settles_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    core = _core_at(561.52, 8.70)
    with ti2(ti2_profile(device_timeout_ms=100), core=core) as (_, microscope):
        core.busy_devices.add("Nosepiece")
        calls = _count_positions(core)
        with caplog.at_level(logging.INFO), pytest.raises(DeviceTimeoutError):
            microscope.require(ObjectiveTurret).select(FOUR, confirm=True)
        del core.getPosition
        assert len(calls) == 1  # the read before the turn; no settle after it
        assert _z_lines(caplog) == []
        microscope.resume()
        core.busy_devices.clear()


def test_a_zstage_lookup_that_raises_comes_before_the_switch_off() -> None:
    # D2: both lookups come before the switch-off, so a facade that is gone raises
    # with PFS still on and nothing sent. The autofocus's own lookup cannot show
    # the order (it raises first when the facade is gone), so this one raises for
    # the focus drive only.
    with ti2() as (core, microscope):
        turret = microscope.require(ObjectiveTurret)
        core.enableContinuousFocus(True)
        core.log.clear()
        real = microscope._lookup

        def lookup(capability: type) -> object | None:
            if capability is ZStage:
                raise HardwareError("ZStage: the focus drive cannot be looked up")
            return real(capability)

        microscope._lookup = lookup  # type: ignore[method-assign]
        with pytest.raises(
            HardwareError, match=r"^ZStage: the focus drive cannot be looked up$"
        ):
            turret.select(FORTY, confirm=True)
        del microscope._lookup
        assert core.log == []
        assert core.isContinuousFocusEnabled() is True
        assert core.getStateLabel("Nosepiece") == TEN
