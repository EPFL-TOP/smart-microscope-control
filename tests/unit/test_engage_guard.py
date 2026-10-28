"""Engaging an autofocus that drives Z to its offset's plane, on the Ti2-E's layout (m2 §12, #205).

Run 1 of session 1 (#208): PFS engaged at a stale offset read locked at
once, then drove Z +235 µm, past the 4350 µm soft limit, in under 100 ms.
The fake reproduces it with ``autofocus_target_um`` and
``autofocus_drive_after_reads`` (m2 §12.7). Unless a test says otherwise:
``z_soft_limits_um = [10, 4350]``, Z at 4269.44 µm (the dish at 10x),
``max_engage_excursion_um = 50``, ``settle_s = 0.05``, objective ``10x``,
and the default lock wait shortened to ``LOCK_S``. "Declared" is
``[autofocus] engage_drives_z_to_offset = true``.

A test whose point is that something happens *inside* the settle window
widens the window, so that a slow poll (Windows sleeps in 15.6 ms steps)
cannot let the window end first. Threads meet on events (FM-44); messages
are compared whole (FM-45).
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from pymmcore_plus import DeviceType
from test_mm_autofocus import (
    JOIN_S,
    LOCK_S,
    OFF,
    ON,
    Stand,
    _profile,
    _Thread,
    _ti2,
    _ti2_core,
)

import smc.hardware.backends.mm as mm
from smc.hardware.capabilities import XYStage
from smc.hardware.errors import (
    AutofocusExcursionError,
    AutofocusLostError,
    DeviceTimeoutError,
    HardwareError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.profile import Profile
from smc.testing import FakeCore, FakeDevice

Z0 = 4269.44
TEN_X = 1  # the Nosepiece position of "10x"
SEEN_OFFSET = 7927.0  # run 2's offset, set on the sample (#208)
STALE_OFFSET = 9370.0  # run 1's, left from an earlier use
LIMITS = {"z_soft_limits_um": (10.0, 4350.0), "max_engage_excursion_um": 50.0}
UNSEEN = (
    "autofocus PFS: not engaged: on this stand engaging drives Z to the plane the "
    "offset names ([autofocus] engage_drives_z_to_offset, m2 §12), and offset "
    "{offset} has not been seen locked with objective {objective!r} since the "
    "stand was opened (seen: {seen}){switched}. Focus on the sample, switch PFS on "
    "with the stand's own controls and set the offset until the sample is sharp; "
    "smc then reads the lock"
)
SWITCHED = "; it was on and searching, so it was switched off"
OUTSIDE = "outside the soft limits [10, 4350] µm"
BEYOND = "more than [safety] max_engage_excursion_um (50 µm) beyond where it started"


@pytest.fixture(autouse=True)
def short_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Read at call time (m2 §3.6), as test_mm_autofocus.py shortens it.
    monkeypatch.setattr(mm, "DEFAULT_LOCK_TIMEOUT_S", LOCK_S)


def _unseen(
    offset: float = STALE_OFFSET,
    objective: str = "10x",
    seen: str = "none",
    switched: str = "",
) -> str:
    return UNSEEN.format(
        offset=f"{offset:g}", objective=objective, seen=seen, switched=switched
    )


def _guarded(*, declared: bool = False, **autofocus: Any) -> Profile:
    return _profile(
        autofocus={
            "settle_s": 0.05,
            "settle_tolerance_um": 0.5,
            "engage_drives_z_to_offset": declared,
            **autofocus,
        },
        safety=LIMITS,
    )


@contextmanager
def _stand(
    profile: Profile | None = None,
    *,
    offset: float = SEEN_OFFSET,
    z_um: float = Z0,
    dry_run: bool = False,
    core: FakeCore | None = None,
) -> Iterator[Stand]:
    """The Ti2-E fake at ``z_um``, ``offset`` and 10x, PFS off; the log cleared."""
    fake = core or _ti2_core()
    fake.setPosition("ZDrive", z_um)
    fake.setPosition("PFSOffset", offset)
    fake.setState("Nosepiece", TEN_X)
    with _ti2(profile or _guarded(), dry_run=dry_run, core=fake) as stand:
        yield stand


def _see(stand: Stand) -> None:
    """The operator's lock at the stand, read by ``is_locked()``, then PFS off."""
    stand.engage_directly()
    assert stand.af.is_locked() is True
    stand.af.disengage()
    stand.core.log.clear()


def _jumps_to(stand: Stand, target_um: float, *, after_reads: int = 0) -> None:
    stand.core.autofocus_target_um = lambda: target_um
    stand.core.autofocus_drive_after_reads = after_reads


# --- D1: only at a seen offset, on a declared stand ---------------------------


def test_an_unseen_offset_is_refused_with_nothing_sent() -> None:
    with _stand(_guarded(declared=True), offset=STALE_OFFSET) as stand:
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == _unseen()
        assert refused.value.how_to_force == ""
        assert stand.core.log == []
        assert stand.core.getPosition("ZDrive") == Z0


def test_the_operators_lock_read_by_is_locked_makes_its_offset_seen() -> None:
    with _stand(_guarded(declared=True)) as stand:
        _see(stand)
        assert stand.af.engage() is True
        assert stand.core.log == [ON]


def test_a_seen_offset_with_another_objective_in_place_is_refused() -> None:
    with _stand(_guarded(declared=True)) as stand:
        _see(stand)
        stand.core.setState("Nosepiece", 2)  # 40x, turned at the stand
        stand.core.log.clear()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == _unseen(SEEN_OFFSET, objective="40x")
        assert stand.core.log == []


def test_an_unseen_offset_while_searching_is_switched_off_then_refused() -> None:
    with _stand(_guarded(declared=True), offset=STALE_OFFSET) as stand:
        stand.engage_directly(lockable=False)
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == _unseen(switched=SWITCHED)
        assert stand.core.log == [OFF]
        assert stand.af.is_engaged() is False


def test_an_offset_moved_with_pfs_off_is_unseen_until_put_back() -> None:
    with _stand(_guarded(declared=True)) as stand:
        _see(stand)
        stand.core.setPosition("PFSOffset", 7937.0)
        stand.core.log.clear()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == _unseen(7937.0, seen="7927")
        stand.core.setPosition("PFSOffset", SEEN_OFFSET)
        stand.core.log.clear()
        assert stand.af.engage() is True
        assert stand.core.log == [ON]


def test_an_undeclared_stand_engages_an_unseen_offset_as_before() -> None:
    with _stand(offset=STALE_OFFSET) as stand:
        assert stand.af.engage() is True
        assert stand.core.log == [ON]


def test_dry_run_engage_at_an_unseen_offset_is_refused_too() -> None:
    with _stand(_guarded(declared=True), offset=STALE_OFFSET, dry_run=True) as stand:
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == _unseen()
        assert stand.core.log == []


def test_the_relock_at_an_unseen_offset_raises_autofocus_lost_error_with_pfs_off() -> (
    None
):
    with _stand(_guarded(declared=True), offset=STALE_OFFSET) as stand:
        stand.engage_directly(lockable=False)
        with pytest.raises(AutofocusLostError) as lost:
            stand.z.move_to_um(4260.0)
        reason = _unseen().removeprefix("autofocus PFS: not engaged: ")
        assert str(lost.value) == (
            f"z: moved to 4260.00 µm, but autofocus PFS was not engaged again: "
            f"{reason}; it is off, or move Z with keep_autofocus=False"
        )
        assert lost.value.z_um == 4260.0
        assert stand.core.log == [OFF, "setPosition('ZDrive', 4260.0)"]
        assert stand.af.is_engaged() is False


def test_a_z_move_under_the_operators_lock_relocks_without_a_prior_read() -> None:
    # suspend() reads the lock before the move ends it (m2 §12.4).
    with _stand(_guarded(declared=True)) as stand:
        stand.engage_directly()
        assert stand.z.move_to_um(4260.0) == 4260.0
        assert stand.core.log == [OFF, "setPosition('ZDrive', 4260.0)", ON]
        assert stand.af.is_locked() is True


# --- D2: the Z guard -----------------------------------------------------------


def test_z_outside_the_soft_limits_before_the_switch_on_refuses_with_nothing_sent() -> (
    None
):
    with _stand(z_um=4400.0) as stand:
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == (
            "autofocus PFS: not engaged: Z is at 4400.00 µm, outside the soft limits "
            "[10, 4350] µm"
        )
        assert refused.value.how_to_force == ""
        assert stand.core.log == []


def _excursion(z_start: float, z_um: float, why: str, *, declared: bool) -> str:
    again = (
        "; on this stand, lock PFS on the sample at the stand before engaging again"
        if declared
        else ""
    )
    return (
        f"autofocus PFS: Z went from {z_start:.2f} µm to {z_um:.2f} µm while it "
        f"engaged, {why}; it was switched off. The guard reads Z every 10 ms: it "
        f"stops PFS once a jump is read, it does not prevent one. Check the sample "
        f"and the objective{again}"
    )


def test_the_jump_of_run_1_trips_the_guard_inside_the_settle_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Locked read True three times, then Z at 4504.76 µm: without the window,
    # the first locked read would have ended the wait (the control below).
    profile = _guarded(settle_s=0.5, lock_timeout_s=2.0)
    with _stand(profile, offset=STALE_OFFSET) as stand:
        _jumps_to(stand, 4504.76, after_reads=3)
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(AutofocusExcursionError) as jumped,
        ):
            stand.af.engage()
        assert str(jumped.value) == _excursion(Z0, 4504.76, OUTSIDE, declared=False)
        assert (jumped.value.z_start_um, jumped.value.z_um) == (Z0, 4504.76)
        assert stand.core.log == [ON, OFF]
        assert stand.af.is_engaged() is False
        assert (
            "autofocus PFS: switched off: Z left its bound while it engaged "
            "(4504.76 µm)"
        ) in caplog.messages
        # Given up and switched off: nothing stays registered as moving.
        assert stand.microscope.state().moving == ()


def test_a_tripped_guard_forgets_the_offset_so_the_next_engage_is_refused() -> None:
    with _stand(_guarded(declared=True)) as stand:
        _see(stand)
        # Out of the band, inside the soft limits: the next engage meets the
        # seen-offset check, not the limits.
        _jumps_to(stand, Z0 + 60.0)
        with pytest.raises(AutofocusExcursionError) as jumped:
            stand.af.engage()
        assert str(jumped.value) == _excursion(Z0, Z0 + 60.0, BEYOND, declared=True)
        assert stand.core.log == [ON, OFF]
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.engage()
        assert str(refused.value) == _unseen(SEEN_OFFSET)
        assert stand.core.log == [ON, OFF]


def _relock_drives_to(stand: Stand, target_um: float) -> None:
    """PFS holds Z where it is until the relock's switch-on, then drives it to ``target_um``."""
    held = stand.core.getPosition("ZDrive")
    log = stand.core.log
    stand.core.autofocus_target_um = lambda: target_um if ON in log else held


def test_a_relock_back_to_where_pfs_held_is_inside_the_band() -> None:
    # The band holds where PFS held Z before the move: without it, it would be
    # [4139.44, 4239.44] around the landing, and the way back would trip it.
    with _stand() as stand:
        stand.engage_directly()
        _relock_drives_to(stand, Z0)
        assert stand.z.move_to_um(Z0 - 80.0) == Z0 - 80.0
        assert stand.core.log == [OFF, f"setPosition('ZDrive', {Z0 - 80.0})", ON]
        assert stand.z.position_um() == Z0
        assert stand.af.is_locked() is True


def test_a_relock_that_overshoots_where_pfs_held_trips_the_guard() -> None:
    with _stand() as stand:
        stand.engage_directly()
        _relock_drives_to(stand, Z0 + 60.0)
        with pytest.raises(AutofocusExcursionError) as jumped:
            stand.z.move_to_um(Z0 - 80.0)
        assert str(jumped.value) == _excursion(
            Z0 - 80.0, Z0 + 60.0, BEYOND, declared=False
        )
        assert stand.core.log == [OFF, f"setPosition('ZDrive', {Z0 - 80.0})", ON, OFF]
        assert stand.af.is_engaged() is False


def test_a_z_read_that_fails_during_the_engage_switches_off_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _stand() as stand:
        core = stand.core
        read = core.getPosition

        def unreadable_once_on(label: str | None = None) -> float:
            if label == "ZDrive" and core.isContinuousFocusEnabled():
                raise RuntimeError("ZDrive: no answer")
            return read(label)

        # An instance attribute: the facade's proxy looks it up at each call.
        monkeypatch.setattr(core, "getPosition", unreadable_once_on)
        with pytest.raises(HardwareError) as failed:
            stand.af.engage()
        assert type(failed.value) is HardwareError
        assert str(failed.value) == (
            "autofocus PFS: Z could not be read while it engaged (focus: "
            "getPosition('ZDrive') failed (ZDrive: no answer); check the device: "
            "power, cable, another program holding it); it was switched off, since "
            "unread it could drive Z past any bound (m2 §12)"
        )
        assert core.log == [ON, OFF]
        assert core.isContinuousFocusEnabled() is False


def test_a_trip_whose_switch_off_fails_says_pfs_may_still_move_z(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-127: a trip and a failed switch-off never read "switched off".
    with _stand() as stand:
        _jumps_to(stand, 4504.76)
        switch = stand.core.enableContinuousFocus

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
        with pytest.raises(DeviceTimeoutError) as stuck:
            stand.af.engage()
        assert not isinstance(stuck.value, AutofocusExcursionError)
        assert str(stuck.value) == (
            f"autofocus PFS: Z went from {Z0:.2f} µm to 4504.76 µm while it engaged, "
            f"{OUTSIDE}, and it could not be switched off (the switch-off failed: "
            "autofocus: enableContinuousFocus(False) failed (PFS: stuck on); check "
            "the device: power, cable, another program holding it); it may still "
            "move Z: stop() and check the stand, then resume()"
        )
        # Registered while it may move Z: the stand refuses actions.
        assert stand.microscope.state().moving == ("autofocus PFS",)
        with pytest.raises(MotionInProgressError):
            stand.microscope.require(XYStage).move_to_um(1.0, 0.0)
        monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
        stand.microscope.resume()


def test_no_focus_drive_engages_without_a_guard_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    devices = [
        FakeDevice("PFS", DeviceType.AutoFocus),
        FakeDevice("PFSOffset", DeviceType.Stage),
        FakeDevice("XYStage", DeviceType.XYStage),
        FakeDevice("Camera-1", DeviceType.Camera),
        FakeDevice("Nosepiece", DeviceType.State, state_labels=("4x", "10x", "40x")),
    ]
    core = FakeCore(devices, camera="Camera-1", xy_stage="XYStage", autofocus="PFS")
    core.setPosition("PFSOffset", SEEN_OFFSET)
    with _ti2(_guarded(), core=core) as stand:
        with caplog.at_level(logging.WARNING):
            assert stand.af.engage() is True
        assert stand.core.log == [ON]
        assert caplog.messages == [
            "autofocus PFS: no focus drive to read, so Z is not guarded while it "
            "engages (m2 §12)"
        ]


# --- D3: settled, not only locked ------------------------------------------------


def test_engage_returns_true_only_once_z_has_settled() -> None:
    # Inside the band and the limits: the jump is allowed, but True waits for it.
    profile = _guarded(settle_s=0.3, lock_timeout_s=2.0)
    with _stand(profile) as stand:
        _jumps_to(stand, Z0 + 20.0, after_reads=3)
        assert stand.af.engage() is True
        assert stand.z.position_um() == Z0 + 20.0


def test_without_a_settle_window_engage_returns_at_the_first_locked_read() -> None:
    # The control of the test above: the same fake, settle_s = 0.
    with _stand(_guarded(settle_s=0.0)) as stand:
        _jumps_to(stand, Z0 + 20.0, after_reads=3)
        assert stand.af.engage() is True
        assert stand.z.position_um() == Z0


def test_a_lock_whose_z_never_settles_is_given_up_switched_off_and_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _stand() as stand:
        reads: list[int] = []

        def creeping() -> float:
            reads.append(1)
            return Z0 + 1.0 * len(reads)

        stand.core.autofocus_target_um = creeping
        with caplog.at_level(logging.WARNING):
            assert stand.af.engage() is False
        assert stand.core.log == [ON, OFF]
        assert stand.af.is_engaged() is False
        pattern = (
            r"autofocus PFS: locked, but Z did not settle within 0\.2 s \(it moved "
            r"\d+\.\d\d µm in the last 0\.05 s\); switched off"
        )
        assert [m for m in caplog.messages if re.fullmatch(pattern, m)], caplog.messages
        assert not [m for m in caplog.messages if "no lock within" in m]


def test_a_lock_timeout_not_above_settle_s_is_refused_before_anything_is_sent() -> None:
    with _stand() as stand:
        with pytest.raises(ValueError, match=re.escape("must be above")) as refused:
            stand.af.engage(timeout_s=0.05)
        assert str(refused.value) == (
            "the lock timeout (0.05 s) must be above [autofocus] settle_s (0.05 s), "
            "since it bounds the settling too"
        )
        assert stand.core.log == []


def test_disengage_during_the_settle_window_cancels_the_engage() -> None:
    profile = _guarded(settle_s=1.0, lock_timeout_s=3.0)
    with _stand(profile) as stand:
        locked = threading.Event()

        def holding() -> float:
            locked.set()
            return Z0

        stand.core.autofocus_target_um = holding
        engage: Callable[[], object] = stand.af.engage
        caller = _Thread(engage)
        assert locked.wait(JOIN_S), "the lock never came"
        stand.af.disengage()
        caller.join()
        assert isinstance(caller.error, MotionStoppedError), caller.error
        assert stand.af.is_engaged() is False
