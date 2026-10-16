"""The Autofocus capability and the Z interlock, on the Ti2-E's layout (m2 §3, #13).

A ``FakeCore`` built from the session-0 labels (m2 §3.12): ``ZDrive`` in the
focus slot, ``PFS`` in the AutoFocus slot with its read-only ``PFS in Range``
property, ``PFSOffset`` the offset stage, and a ``Nosepiece`` with three
objectives. Everything goes through ``Microscope.from_core``, as a tool's
calls would. The fake's Nikon quirk is on: a Z move switches PFS off (FM-12).

Threads meet on the facade's ``state().moving`` or on a join with a timeout,
followed by an ``is_alive`` check (FM-44); messages are compared whole (FM-45).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest
from pymmcore_plus import DeviceType

import smc.hardware.backends.mm as mm
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, FocusStep, XYStage, ZStage
from smc.hardware.errors import (
    AutofocusLostError,
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.profile import (
    AutofocusSection,
    MicroscopeSection,
    Profile,
    RolesSection,
    SafetySection,
)
from smc.hardware.roles import Role
from smc.testing import FakeCore, FakeDevice, FakeProperty

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

JOIN_S = 5.0
#: The default lock wait, shortened for every test here: 5 s is the design's.
LOCK_S = 0.2
ASSUMED = "(assumed: the profile sets no [autofocus] lock_timeout_s)"
IN_RANGE = {"in_range_property": "PFS in Range", "in_range_values": ["Within Range"]}
OFF, ON = "enableContinuousFocus(False)", "enableContinuousFocus(True)"


@pytest.fixture(autouse=True)
def short_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Read at call time (m2 §3.6, decided in the plan): a profile without
    # lock_timeout_s waits this long.
    monkeypatch.setattr(mm, "DEFAULT_LOCK_TIMEOUT_S", LOCK_S)


def _pfs() -> FakeDevice:
    """PFS as session 0 read it; a test keeps it to change ``PFS in Range``."""
    return FakeDevice(
        "PFS",
        DeviceType.AutoFocus,
        properties={"PFS in Range": FakeProperty("Out of Range", read_only=True)},
    )


def _ti2_core(
    *,
    pfs: FakeDevice | None = None,
    turret: bool = True,
    extra: Sequence[FakeDevice] = (),
) -> FakeCore:
    """The Ti2-E's layout; the core holds ``pfs`` itself, so a change to it shows."""
    devices = [
        FakeDevice("ZDrive", DeviceType.Stage),
        pfs or _pfs(),
        FakeDevice("PFSOffset", DeviceType.Stage),
        FakeDevice("XYStage", DeviceType.XYStage),
        FakeDevice("Camera-1", DeviceType.Camera),
    ]
    if turret:
        devices.append(
            FakeDevice("Nosepiece", DeviceType.State, state_labels=("4x", "10x", "40x"))
        )
    devices.extend(extra)
    return FakeCore(
        devices,
        camera="Camera-1",
        xy_stage="XYStage",
        focus="ZDrive",
        autofocus="PFS",
    )


def _profile(
    *,
    autofocus: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
    roles: dict[str, Any] | None = None,
) -> Profile:
    return Profile(
        microscope=MicroscopeSection(name="ti2e-fake"),
        safety=SafetySection.model_validate({"max_z_jog_um": 50.0, **(safety or {})}),
        autofocus=AutofocusSection.model_validate(autofocus or {}),
        roles=RolesSection.model_validate(roles or {}),
    )


@dataclass
class Stand:
    core: FakeCore
    microscope: Microscope

    @property
    def af(self) -> Autofocus:
        return self.microscope.require(Autofocus)

    @property
    def z(self) -> ZStage:
        return self.microscope.require(ZStage)

    def engage_directly(self, *, lockable: bool = True) -> None:
        """PFS on, as if switched on at the stand, and the log cleared."""
        self.core.autofocus_lockable = lockable
        self.core.enableContinuousFocus(True)
        self.core.log.clear()


@contextmanager
def _ti2(
    profile: Profile | None = None,
    *,
    dry_run: bool = False,
    core: FakeCore | None = None,
) -> Iterator[Stand]:
    """A facade over ``core`` (the Ti2-E layout), with the log cleared after building it."""
    fake = core or _ti2_core()
    microscope = Microscope.from_core(
        cast("CMMCorePlus", fake), profile or _profile(), dry_run=dry_run
    )
    fake.log.clear()
    try:
        yield Stand(fake, microscope)
    finally:
        microscope.close()


class _Thread:
    """Run ``target`` in a thread and keep what it returned or raised."""

    def __init__(self, target: Callable[[], object]) -> None:
        self.result: object = None
        self.error: BaseException | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self._target()
        except BaseException as exc:
            self.error = exc

    def join(self) -> None:
        self._thread.join(timeout=JOIN_S)
        assert not self._thread.is_alive(), "thread did not finish: deadlock?"


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + JOIN_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def _moving(stand: Stand, *names: str) -> Callable[[], bool]:
    return lambda: stand.microscope.state().moving == names


def _no_relock_after_the_move(log: list[str]) -> None:
    """Nothing after the Z command switched PFS on again (FM-88)."""
    sent = log.index("setPosition('ZDrive', 12.0)")
    assert ON not in log[sent:], log


# --- the interlock (m2 §3.4) -------------------------------------------------


def test_moving_z_re_engages_pfs(caplog: pytest.LogCaptureFixture) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        with caplog.at_level(logging.INFO):
            landed = stand.z.move_to_um(12.0)
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", ON]
        assert landed == 12.0
        assert stand.core.isContinuousFocusEnabled() is True
        assert stand.core.isContinuousFocusLocked() is True
        assert [m for m in caplog.messages if m.startswith(("autofocus", "z:"))] == [
            "autofocus: suspend for z: move_to 12.0 µm",
            "z: move_to 12.0 µm",
            "autofocus: engage after z: move_to 12.0 µm",
        ]


def test_moving_z_by_re_engages_pfs() -> None:
    with _ti2() as stand:
        stand.core.setPosition("ZDrive", 10.0)
        stand.engage_directly()
        assert stand.z.move_by_um(2.0) == 12.0
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", ON]
        assert stand.core.isContinuousFocusLocked() is True


def test_the_z_readback_is_taken_before_the_relock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Once PFS holds focus again, the drive reads where PFS holds it: the
    # move returns where it landed (m2 §3.4 step 5).
    with _ti2() as stand:
        stand.engage_directly()
        switch = stand.core.enableContinuousFocus

        def correcting(enable: bool) -> None:
            switch(enable)
            if enable:
                # PFS drives Z to its focal plane; the fake's setPosition would
                # switch it off again (the quirk), so the store is set directly.
                stand.core._z["ZDrive"] = 12.3

        monkeypatch.setattr(stand.core, "enableContinuousFocus", correcting)
        assert stand.z.move_to_um(12.0) == 12.0
        assert stand.z.position_um() == 12.3


def test_moving_z_with_pfs_off_sends_no_continuous_focus_call() -> None:
    with _ti2() as stand:
        assert stand.z.move_to_um(12.0) == 12.0
        assert stand.z.move_by_um(1.0) == 13.0
        assert stand.core.log == [
            "setPosition('ZDrive', 12.0)",
            "setPosition('ZDrive', 13.0)",
        ]


def test_keep_autofocus_false_leaves_pfs_off(caplog: pytest.LogCaptureFixture) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        with caplog.at_level(logging.INFO):
            assert stand.z.move_to_um(12.0, keep_autofocus=False) == 12.0
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)"]
        assert stand.core.isContinuousFocusEnabled() is False
        assert "autofocus PFS: left off (keep_autofocus=False)" in caplog.messages
        stand.engage_directly()
        assert stand.z.move_by_um(-2.0, keep_autofocus=False) == 10.0
        assert stand.core.log == [OFF, "setPosition('ZDrive', 10.0)"]


@pytest.mark.parametrize("move", ["absolute", "relative"])
def test_z_move_refused_by_the_soft_limits_never_touches_pfs(move: str) -> None:
    profile = _profile(safety={"z_soft_limits_um": (-100.0, 100.0)})
    with _ti2(profile) as stand:
        stand.engage_directly()
        z = stand.z
        # Forced: the soft limit, not the jog guard, refuses the relative one.
        attempt = (
            (lambda: z.move_to_um(500.0))
            if move == "absolute"
            else (lambda: z.move_by_um(150.0, force=True))
        )
        with pytest.raises(SafetyRefusedError) as refused:
            attempt()
        assert refused.value.how_to_force == ""
        assert "is outside the soft limits [-100.0, 100.0] µm" in refused.value.reason
        assert stand.core.log == []
        assert stand.core.isContinuousFocusLocked() is True


def test_z_jog_refused_never_touches_pfs() -> None:
    with _ti2() as stand:
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.z.move_by_um(60.0)
        assert refused.value.reason == (
            "Z jog 60.0 µm exceeds the Z jog limit of 50.0 µm"
        )
        assert stand.core.log == []
        assert stand.core.isContinuousFocusLocked() is True


def test_z_move_that_cannot_relock_raises_autofocus_lost_error_with_pfs_off() -> None:
    with _ti2(_profile(autofocus=IN_RANGE)) as stand:
        stand.engage_directly(lockable=False)
        with pytest.raises(AutofocusLostError) as lost:
            stand.z.move_to_um(12.0)
        assert str(lost.value) == (
            f"z: moved to 12.00 µm, but autofocus PFS did not lock again within "
            f"0.2 s {ASSUMED} and is off (in range: no); engage it (engage()) once "
            f"the sample is in range, or move Z with keep_autofocus=False"
        )
        assert (lost.value.z_um, lost.value.in_range) == (12.0, False)
        assert not isinstance(lost.value, SafetyRefusedError)
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", ON, OFF]
        assert stand.core.isContinuousFocusEnabled() is False
        # Nothing is left registered: the next action runs.
        assert stand.microscope.state().moving == ()
        assert stand.z.move_to_um(11.0) == 11.0


# --- no relock after a failure (FM-88) ---------------------------------------


def test_failed_z_move_does_not_relock(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        send = stand.core.setPosition

        def faulty(label: str, z_um: float) -> None:
            send(label, z_um)
            if label == "ZDrive":
                raise RuntimeError("ZDrive: no answer")

        monkeypatch.setattr(stand.core, "setPosition", faulty)
        with caplog.at_level(logging.WARNING), pytest.raises(HardwareError) as failed:
            stand.z.move_to_um(12.0)
        assert str(failed.value).startswith(
            "focus: setPosition('ZDrive', 12.0) failed (ZDrive: no answer)"
        )
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", "stop('ZDrive')"]
        _no_relock_after_the_move(stand.core.log)
        assert stand.core.isContinuousFocusEnabled() is False
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(HardwareError)"
        ) in caplog.messages


def test_stopped_z_move_does_not_relock(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        z = stand.z
        stand.core.busy_devices.add("ZDrive")
        stop = stand.core.stop

        def stop_arrives(label: str) -> None:
            stop(label)
            stand.core.busy_devices.discard(label)

        monkeypatch.setattr(stand.core, "stop", stop_arrives)
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: z.move_to_um(12.0))
            _wait_for(_moving(stand, "z ZDrive"), "the Z move to wait")
            z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", "stop('ZDrive')"]
        _no_relock_after_the_move(stand.core.log)
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(MotionStoppedError)"
        ) in caplog.messages


def test_halted_z_move_does_not_relock(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        z = stand.z
        stand.core.busy_devices.add("ZDrive")
        stop = stand.core.stop

        def stop_arrives(label: str) -> None:
            stop(label)
            stand.core.busy_devices.discard(label)

        monkeypatch.setattr(stand.core, "stop", stop_arrives)
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: z.move_to_um(12.0))
            _wait_for(_moving(stand, "z ZDrive"), "the Z move to wait")
            stand.microscope.stop()
            mover.join()
        assert isinstance(mover.error, HardwareError)
        assert stand.core.log == [
            OFF,
            "setPosition('ZDrive', 12.0)",
            OFF,  # the emergency stop switches PFS off first
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        _no_relock_after_the_move(stand.core.log)
        assert any(
            m.startswith(
                "autofocus PFS: left off, because z: move_to 12.0 µm did not complete"
            )
            for m in caplog.messages
        )
        stand.microscope.resume()


def test_interrupted_z_move_does_not_relock(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        busy = stand.core.deviceBusy
        interrupted: list[str] = []

        def ctrl_c(label: str) -> bool:
            if label == "ZDrive" and not interrupted:
                interrupted.append(label)
                raise KeyboardInterrupt
            return busy(label)

        monkeypatch.setattr(stand.core, "deviceBusy", ctrl_c)
        with caplog.at_level(logging.WARNING), pytest.raises(KeyboardInterrupt):
            stand.z.move_to_um(12.0)
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", "stop('ZDrive')"]
        _no_relock_after_the_move(stand.core.log)
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(KeyboardInterrupt)"
        ) in caplog.messages


def test_dry_run_z_move_with_pfs_engaged_sends_nothing() -> None:
    with _ti2(dry_run=True) as stand:
        stand.engage_directly()
        assert stand.z.move_to_um(12.0) == 12.0
        assert stand.z.move_by_um(1.0) == 1.0
        assert stand.core.log == []
        assert stand.core.isContinuousFocusLocked() is True


# --- engage (m2 §3.2, §3.3) --------------------------------------------------


def test_engage_returns_true_once_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    with _ti2() as stand:
        locked = stand.core.isContinuousFocusLocked
        polls: list[int] = []

        def locks_on_the_third_read() -> bool:
            polls.append(1)
            return len(polls) >= 3 and locked()

        monkeypatch.setattr(
            stand.core, "isContinuousFocusLocked", locks_on_the_third_read
        )
        # A success path waits as long as it must, not the shortened default.
        assert stand.af.engage(timeout_s=JOIN_S) is True
        assert len(polls) >= 3  # it waited for the lock
        assert stand.core.log == [ON]
        assert stand.af.is_locked() is True


def test_engage_returns_false_on_no_lock_and_switches_pfs_off() -> None:
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        assert stand.af.engage() is False
        assert stand.core.log == [ON, OFF]
        assert stand.af.is_engaged() is False
        # Nothing is left registered: the next action runs.
        assert stand.microscope.state().moving == ()
        assert stand.z.move_to_um(1.0) == 1.0


def test_no_lock_warning_labels_the_default_timeout_as_assumed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        with _ti2(_profile(autofocus=IN_RANGE)) as stand:
            stand.core.autofocus_lockable = False
            assert stand.af.engage() is False
            # An explicit timeout is not assumed, whatever the profile says.
            assert stand.af.engage(timeout_s=0.25) is False
        with _ti2(_profile(autofocus={"lock_timeout_s": 0.3})) as stand:
            stand.core.autofocus_lockable = False
            assert stand.af.engage() is False
    assert [m for m in caplog.messages if m.startswith("autofocus PFS: no lock")] == [
        f"autofocus PFS: no lock within 0.2 s {ASSUMED}; in range: no; switched off",
        "autofocus PFS: no lock within 0.25 s; in range: no; switched off",
        "autofocus PFS: no lock within 0.3 s; in range: unknown; switched off",
    ]


def test_engage_raises_when_pfs_cannot_be_switched_off_after_no_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        switch = stand.core.enableContinuousFocus

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
        with pytest.raises(DeviceTimeoutError) as timed_out:
            stand.af.engage()
        message = str(timed_out.value)
        assert message.startswith(
            "autofocus PFS: no lock within 0.2 s, and it could not be switched off "
            "(the switch-off failed: autofocus: enableContinuousFocus(False) failed "
            "(PFS: stuck on);"
        )
        assert message.endswith(
            "); it may still move Z: stop() and check the stand, then resume()"
        )
        # The Executor's own message (device_timeout_ms) does not reach the caller.
        assert "device_timeout_ms" not in message
        assert isinstance(timed_out.value.__cause__, DeviceTimeoutError)
        assert stand.core.log == [ON, OFF]
        # The wait stays registered: the stand refuses actions until resume().
        assert stand.microscope.state().moving == ("autofocus PFS",)
        with pytest.raises(MotionInProgressError) as refused:
            stand.z.move_to_um(1.0)
        assert refused.value.moving == ("autofocus PFS",)
        monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
        stand.microscope.resume()


def test_engage_raises_when_the_switch_off_failed_even_if_it_reads_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A switch-off that raised may or may not have taken: m2 §3.2 raises
    # rather than return False on the strength of one read.
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        switch = stand.core.enableContinuousFocus

        def off_but_raises(enable: bool) -> None:
            switch(enable)
            if not enable:
                raise RuntimeError("PFS: no acknowledgement")

        monkeypatch.setattr(stand.core, "enableContinuousFocus", off_but_raises)
        with pytest.raises(DeviceTimeoutError) as timed_out:
            stand.af.engage()
        assert (
            "and it could not be switched off (the switch-off failed: autofocus: "
            "enableContinuousFocus(False) failed (PFS: no acknowledgement);"
        ) in str(timed_out.value)
        assert stand.core.isContinuousFocusEnabled() is False
        # It reads off, so nothing stays registered and the next action runs.
        monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
        assert stand.microscope.state().moving == ()
        assert stand.z.move_to_um(1.0) == 1.0


def test_engage_that_loses_the_lock_at_once_switches_it_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Locked at the wait's last poll, lost by the readback: left on, it would
    # search and move Z with no action holding the lock (beyond the plan).
    with _ti2() as stand:
        locked = stand.core.isContinuousFocusLocked
        reads: list[int] = []

        def lost_after_the_first_read() -> bool:
            reads.append(1)
            return len(reads) == 1 and locked()

        monkeypatch.setattr(
            stand.core, "isContinuousFocusLocked", lost_after_the_first_read
        )
        with caplog.at_level(logging.WARNING):
            assert stand.af.engage() is False
        assert stand.core.log == [ON, OFF]
        assert stand.core.isContinuousFocusEnabled() is False
        assert (
            "autofocus PFS: lost the lock at once, so it was switched off (in range: "
            "unknown)"
        ) in caplog.messages


def test_engage_that_finds_it_switched_itself_off_returns_false(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        monkeypatch.setattr(stand.core, "isContinuousFocusEnabled", lambda: False)
        with caplog.at_level(logging.WARNING):
            assert stand.af.engage() is False
        assert stand.core.log == [ON]
        assert (
            "autofocus PFS: switched itself off before it locked (in range: unknown)"
        ) in caplog.messages


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, 600.5])
def test_engage_timeout_is_bounded(bad: float) -> None:
    # The wait holds the microscope lock (FM-32): refused before anything is sent.
    with _ti2() as stand:
        with pytest.raises(
            ValueError, match=r"^timeout_s must be finite and in \(0, 600\] s, got "
        ):
            stand.af.engage(timeout_s=bad)
        assert stand.core.log == []
        assert stand.af.engage(timeout_s=600.0) is True  # the bound itself


def test_disengage_during_engage_cancels_it() -> None:
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af = stand.af
        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_moving(stand, "autofocus PFS"), "the lock wait")
        af.disengage()
        waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert waiter.error.device == "autofocus PFS"
        assert stand.core.log == [ON, OFF]
        assert stand.core.isContinuousFocusEnabled() is False
        assert stand.microscope.state().moving == ()


def test_another_action_during_the_lock_wait_is_refused_at_once() -> None:
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, xy = stand.af, stand.z, stand.microscope.require(XYStage)
        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_moving(stand, "autofocus PFS"), "the lock wait")
        # Refused, not queued behind the wait for the lock (which would raise
        # MicroscopeBusyError after 60 s).
        for action in (lambda: z.move_to_um(1.0), lambda: xy.move_to_um(1.0, 0.0)):
            with pytest.raises(MotionInProgressError) as refused:
                action()
            assert refused.value.moving == ("autofocus PFS",)
            assert refused.value.detail == "`autofocus: engage` is waiting for it"
        af.disengage()
        waiter.join()
        assert stand.core.log == [ON, OFF]


def test_dry_run_engage_returns_true_and_sends_nothing() -> None:
    with _ti2(dry_run=True) as stand:
        stand.core.autofocus_lockable = False
        assert stand.af.engage() is True  # the commanded outcome
        assert stand.core.log == []


def test_disengage_runs_while_halted_and_in_dry_run() -> None:
    with _ti2() as stand:
        stand.engage_directly()
        stand.microscope.stop()
        stand.engage_directly()
        stand.af.disengage()
        assert stand.core.log == [OFF]
        assert stand.core.isContinuousFocusEnabled() is False
        stand.microscope.resume()
    with _ti2(dry_run=True) as stand:
        stand.engage_directly()
        stand.af.disengage()
        assert stand.core.log == [OFF]
        assert stand.core.isContinuousFocusEnabled() is False


# --- the offset (m2 §3.2, §3.8) ----------------------------------------------


def _at_offset(value: float, **kwargs: Any) -> FakeCore:
    core = _ti2_core(**kwargs)
    core.setPosition("PFSOffset", value)
    return core


def test_offset_goes_to_the_offset_device_when_one_is_resolved() -> None:
    with _ti2(core=_at_offset(7803.0)) as stand:
        assert stand.microscope.roles.get(Role.autofocus_offset) == "PFSOffset"
        assert stand.af.offset_native() == 7803.0
        assert stand.af.set_offset_native(7805.0) == 7805.0
        # Never setAutoFocusOffset: pymmcore-plus does not redirect it for NikonTi2.
        assert stand.core.log == ["setPosition('PFSOffset', 7805.0)"]
        assert stand.core.getAutoFocusOffset() == 0.0


def test_offset_goes_through_the_autofocus_device_without_an_offset_role() -> None:
    core = FakeCore.demo_like()
    with _ti2(Profile.demo(), core=core) as stand:
        assert stand.microscope.roles.get(Role.autofocus_offset) is None
        assert stand.af.set_offset_native(12.5) == 12.5
        assert stand.af.offset_native() == 12.5
        assert stand.af.offset_by_native(1.0, force=True) == 13.5
        assert core.log == ["setAutoFocusOffset(12.5)", "setAutoFocusOffset(13.5)"]
        assert core.getPosition("Z") == 0.0


def test_offset_jog_without_a_profile_limit_needs_force() -> None:
    with _ti2(core=_at_offset(7803.0)) as stand:
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.offset_by_native(1.0)
        assert refused.value.reason == (
            "autofocus offset jog 1.0 device units: the profile sets no [safety] "
            "max_autofocus_offset_jog, so every relative offset change needs "
            "force=True"
        )
        assert stand.core.log == []
        assert stand.af.offset_by_native(0.0) == 7803.0
        assert stand.af.offset_by_native(-3.0, force=True) == 7800.0
        assert stand.core.log == [
            "setPosition('PFSOffset', 7803.0)",
            "setPosition('PFSOffset', 7800.0)",
        ]
    limited = _profile(
        safety={"max_autofocus_offset_jog": 5.0},
        autofocus={"offset_unit": "PFS offset steps"},
    )
    with _ti2(limited, core=_at_offset(7803.0)) as stand:
        assert stand.af.offset_by_native(5.0) == 7808.0
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.offset_by_native(5.5)
        assert refused.value.reason == (
            "autofocus offset jog 5.5 PFS offset steps exceeds the offset jog limit "
            "of 5.0 PFS offset steps"
        )
        assert stand.af.offset_unit() == "PFS offset steps"


def test_offset_is_refused_outside_the_soft_limits_even_forced() -> None:
    profile = _profile(
        safety={
            "max_autofocus_offset_jog": 5000.0,
            "autofocus_offset_soft_limits": (7000.0, 8000.0),
        }
    )
    with _ti2(profile, core=_at_offset(7803.0)) as stand:
        assert stand.af.offset_limits_native() == (7000.0, 8000.0)
        for change in (
            lambda: stand.af.set_offset_native(9000.0),
            lambda: stand.af.offset_by_native(1000.0, force=True),
            lambda: stand.af.set_offset_native(math.nan),
        ):
            with pytest.raises(SafetyRefusedError) as refused:
                change()
            assert refused.value.how_to_force == ""
        assert stand.core.log == []
        assert stand.af.offset_native() == 7803.0


_Z_LIMITS = {"z_soft_limits_um": (-10.0, 10.0), "max_autofocus_offset_jog": 1000.0}


def _at_offset_and_z(**kwargs: Any) -> FakeCore:
    """The offset at 7800 and Z at 9 µm, 1 µm inside the Z soft limits of ``_Z_LIMITS``."""
    core = _at_offset(7800.0, **kwargs)
    core.setPosition("ZDrive", 9.0)
    return core


def test_offset_change_while_locked_is_refused_past_the_z_soft_limits() -> None:
    # The review of #140 (FM-90): locked, PFS follows its offset and drives Z
    # by (target - offset) / factor, here -2.0 per µm.
    profile = _profile(autofocus={"offset_per_um": {"4x": -2.0}}, safety=_Z_LIMITS)
    refusal = (
        "Z target 11.0 µm is outside the soft limits [-10.0, 10.0] µm; change "
        "[safety] in the profile if the limits are wrong"
    )
    with _ti2(profile, core=_at_offset_and_z()) as stand:
        stand.engage_directly()
        af = stand.af
        for change in (
            lambda: af.set_offset_native(7796.0),
            lambda: af.offset_by_native(-4.0),
            lambda: af.offset_by_native(-4.0, force=True),
        ):
            with pytest.raises(SafetyRefusedError) as refused:
                change()
            assert (refused.value.reason, refused.value.how_to_force) == (refusal, "")
        assert stand.core.log == []
        # Inside them: +4.0 is Z 7 µm, then 7803.0 is Z 9.5 µm.
        assert af.offset_by_native(4.0) == 7804.0
        assert af.set_offset_native(7803.0) == 7803.0
        assert stand.core.log == [
            "setPosition('PFSOffset', 7804.0)",
            "setPosition('PFSOffset', 7803.0)",
        ]
        # Off, the offset moves no Z.
        af.disengage()
        assert af.set_offset_native(7796.0) == 7796.0
    with _ti2(profile, dry_run=True, core=_at_offset_and_z()) as stand:
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError, match=re.escape(refusal)):
            stand.af.set_offset_native(7796.0)


def test_offset_change_while_locked_without_a_factor_has_only_the_offset_guards() -> (
    None
):
    # Nothing converts the change, so the Z limits are not guessed at.
    no_factor = _profile(autofocus={"offset_per_um": {"40x": -2.0}}, safety=_Z_LIMITS)
    no_turret = _profile(autofocus={"offset_per_um": {"4x": -2.0}}, safety=_Z_LIMITS)
    for profile, core in (
        (no_factor, _at_offset_and_z()),
        (no_turret, _at_offset_and_z(turret=False)),
    ):
        with _ti2(profile, core=core) as stand:
            stand.engage_directly()
            assert stand.af.set_offset_native(7796.0) == 7796.0
            assert stand.af.offset_by_native(-4.0) == 7792.0
            assert stand.core.log == [
                "setPosition('PFSOffset', 7796.0)",
                "setPosition('PFSOffset', 7792.0)",
            ]


# --- focus_by_um (m2 §3.5) ---------------------------------------------------


def _factor(factors: dict[str, float], **more: Any) -> Profile:
    return _profile(
        autofocus={"offset_per_um": factors, **more},
        safety={"max_autofocus_offset_jog": 10.0},
    )


def test_focus_by_with_pfs_off_moves_z() -> None:
    with _ti2(_factor({"4x": -3.0})) as stand:
        assert stand.af.focus_by_um(2.0) == FocusStep("z", 2.0, None)
        assert stand.core.log == ["setPosition('ZDrive', 2.0)"]
        assert stand.core.isContinuousFocusEnabled() is False


def test_focus_by_while_locked_converts_with_the_measured_factor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        with caplog.at_level(logging.INFO):
            step = stand.af.focus_by_um(2.0)
        # 2 µm times -3.0 per µm (the Nosepiece is at "4x"), then the lock again.
        assert stand.core.log == ["setPosition('PFSOffset', 7794.0)", ON]
        assert step == FocusStep("offset", 0.0, 7794.0)
        assert "autofocus: engage after autofocus: focus_by 2.0 µm" in caplog.messages
        assert stand.core.isContinuousFocusLocked() is True


def test_focus_by_while_locked_without_a_factor_for_this_objective_is_refused() -> None:
    with _ti2(_factor({"40x": 1.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(2.0, force=True)
        assert refused.value.reason == (
            "autofocus PFS is locked, so focus_by_um() would move its offset, but "
            "the profile has no [autofocus] offset_per_um factor for objective "
            "'4x'; measure it with the autofocus locked and add it, or use "
            "offset_by_native() in the offset's own units"
        )
        assert refused.value.how_to_force == ""
        assert stand.core.log == []


def test_focus_by_without_an_objective_turret_is_refused_while_locked() -> None:
    with _ti2(_factor({"4x": -3.0}), core=_ti2_core(turret=False)) as stand:
        assert stand.microscope.roles.get(Role.objective_turret) is None
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(2.0)
        assert refused.value.reason == (
            "autofocus PFS is locked, so focus_by_um() would move its offset, but "
            "no objective turret is resolved, so no [autofocus] offset_per_um factor "
            "can be chosen; assign the objective_turret role under [roles.assign], "
            "or use offset_by_native() in the offset's own units"
        )
        assert refused.value.how_to_force == ""
        assert stand.core.log == []


def test_focus_by_while_on_but_not_locked_is_refused() -> None:
    with _ti2(_factor({"4x": -3.0}, **IN_RANGE)) as stand:
        stand.engage_directly(lockable=False)
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(1.0, force=True)
        assert refused.value.reason == (
            "autofocus PFS is on but not locked (in range: no); wait for the lock "
            "(engage()), or disengage() it to focus with Z"
        )
        assert refused.value.how_to_force == ""
        assert stand.core.log == []


def test_focus_by_jog_guard_applies_to_dz_um() -> None:
    # An offset jog limit wide enough for the converted change (60 µm times
    # -3.0): only Z's guard on dz_um stands between the call and the offset.
    profile = _profile(
        autofocus={"offset_per_um": {"4x": -3.0}},
        safety={"max_autofocus_offset_jog": 1000.0},
    )
    with _ti2(profile, core=_at_offset(7800.0)) as stand:
        refusal = "Z jog 60.0 µm exceeds the Z jog limit of 50.0 µm"
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(60.0)
        assert refused.value.reason == refusal
        assert refused.value.how_to_force == "pass force=True"
        stand.engage_directly()
        # Locked too: the guard is on dz_um, before the offset is converted.
        with pytest.raises(SafetyRefusedError) as locked:
            stand.af.focus_by_um(60.0)
        assert locked.value.reason == refusal
        assert stand.core.log == []
        stand.af.disengage()
        assert stand.af.focus_by_um(60.0, force=True) == FocusStep("z", 60.0, None)


def test_focus_by_that_cannot_relock_raises_autofocus_lost_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _ti2(_factor({"4x": -3.0}, **IN_RANGE), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        move = stand.core.setPosition

        def loses_the_reference(label: str, value: float) -> None:
            move(label, value)
            if label == "PFSOffset":
                stand.core.autofocus_lockable = False

        monkeypatch.setattr(stand.core, "setPosition", loses_the_reference)
        with pytest.raises(AutofocusLostError) as lost:
            stand.af.focus_by_um(2.0)
        assert str(lost.value) == (
            f"autofocus_offset: moved to 7794.0 device units, but autofocus PFS did "
            f"not lock again within 0.2 s {ASSUMED} and is off (in range: no); "
            f"engage it (engage()) once the sample is in range; while it is off, "
            f"focus_by_um() moves Z"
        )
        assert (lost.value.z_um, lost.value.in_range) == (0.0, False)
        assert stand.core.log == ["setPosition('PFSOffset', 7794.0)", ON, OFF]
        assert stand.core.isContinuousFocusEnabled() is False


def test_focus_by_while_locked_passes_force_to_the_offset_jog_guard() -> None:
    # m2 §3.5: the offset's own guards apply to the converted change. 40 µm is
    # inside Z's 50 µm guard; times -3.0 it is 120, above the offset's 10.
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(40.0)
        assert refused.value.reason == (
            "autofocus offset jog -120.0 device units exceeds the offset jog limit "
            "of 10.0 device units"
        )
        assert refused.value.how_to_force == "pass force=True"
        assert stand.core.log == []
        step = stand.af.focus_by_um(40.0, force=True)
        assert stand.core.log == ["setPosition('PFSOffset', 7680.0)", ON]
        assert step == FocusStep("offset", 0.0, 7680.0)
    no_limit = _profile(autofocus={"offset_per_um": {"4x": -3.0}})
    with _ti2(no_limit, core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(1.0)
        assert "the profile sets no [safety] max_autofocus_offset_jog" in (
            refused.value.reason
        )
        assert stand.core.log == []


def test_focus_by_while_locked_respects_the_z_soft_limits() -> None:
    # PFS follows the offset and drives Z: the Z soft limits, which no Z move
    # may pass even forced, bound focus_by_um too (adversarial review of #13).
    profile = _profile(
        autofocus={"offset_per_um": {"4x": 1.0}},
        safety={
            "z_soft_limits_um": (0.0, 100.0),
            "max_autofocus_offset_jog": 1000.0,
        },
    )
    core = _at_offset(0.0)
    core.setPosition("ZDrive", 99.0)
    with _ti2(profile, core=core) as stand:
        stand.engage_directly()
        for force in (False, True):
            with pytest.raises(SafetyRefusedError) as refused:
                stand.af.focus_by_um(30.0, force=force)
            assert refused.value.reason == (
                "Z target 129.0 µm is outside the soft limits [0.0, 100.0] µm; "
                "change [safety] in the profile if the limits are wrong"
            )
            assert refused.value.how_to_force == ""
        assert stand.core.log == []
        assert stand.af.focus_by_um(1.0) == FocusStep("offset", 99.0, 1.0)


def test_focus_by_that_cannot_read_z_afterwards_says_the_change_was_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reported as a plain failure, the change would be made twice by a caller
    # that tries again (adversarial review of #13).
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        move, read = stand.core.setPosition, stand.core.getPosition
        moved: list[str] = []

        def move_then_note(label: str, value: float) -> None:
            move(label, value)
            moved.append(label)

        def unreadable_once_moved(label: str | None = None) -> float:
            if label == "ZDrive" and moved:
                raise RuntimeError("ZDrive: no answer")
            return read(label)

        monkeypatch.setattr(stand.core, "setPosition", move_then_note)
        monkeypatch.setattr(stand.core, "getPosition", unreadable_once_moved)
        with pytest.raises(HardwareError) as failed:
            stand.af.focus_by_um(2.0)
        assert str(failed.value).startswith(
            "autofocus: focus_by 2.0 µm: the offset moved to 7794.0 device units "
            "and autofocus PFS locked again, but Z could not be read afterwards ("
        )
        assert str(failed.value).endswith("); the change is done, do not repeat it")
        assert stand.core.log == ["setPosition('PFSOffset', 7794.0)", ON]


def test_relock_and_focus_by_wait_for_the_profiles_lock_timeout() -> None:
    profile = _factor({"4x": -3.0}, lock_timeout_s=0.3)
    not_again = (
        "but autofocus PFS did not lock again within 0.3 s and is off (in range: "
        "unknown)"
    )
    with _ti2(profile, core=_at_offset(7800.0)) as stand:
        stand.engage_directly(lockable=False)
        with pytest.raises(AutofocusLostError) as lost:
            stand.z.move_to_um(12.0)
        assert f"z: moved to 12.00 µm, {not_again};" in str(lost.value)
        assert "assumed" not in str(lost.value)
        stand.engage_directly()
        stand.core.autofocus_lockable = True
        move = stand.core.setPosition

        def loses_the_reference(label: str, value: float) -> None:
            move(label, value)
            stand.core.autofocus_lockable = False

        stand.core.setPosition = loses_the_reference  # type: ignore[method-assign]
        with pytest.raises(AutofocusLostError) as lost:
            stand.af.focus_by_um(2.0)
        assert f"device units, {not_again};" in str(lost.value)


# --- stops and disengage() during a compound (m2 §3.7) -----------------------


def test_z_stop_while_the_move_waits_for_the_lock_cancels_it_before_pfs_is_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Design §13: a stop pressed while a move waits for the lock cancels it.
    # The compound records the stop generations when it is called, so the Z
    # move's nested action is cancelled, and nothing is sent at all.
    with _ti2() as stand:
        stand.engage_directly()
        z, camera = stand.z, stand.microscope.require(Camera)
        executor = stand.microscope._executor
        queued, held, release = threading.Event(), threading.Event(), threading.Event()
        contention, snap = executor._contention, stand.core.snapImage

        def noted(*args: Any, **kwargs: Any) -> None:
            queued.set()
            contention(*args, **kwargs)

        def hung_snap() -> None:
            held.set()
            release.wait(JOIN_S)
            snap()

        monkeypatch.setattr(executor, "_contention", noted)
        monkeypatch.setattr(stand.core, "snapImage", hung_snap)
        snapper = _Thread(camera.snap)
        assert held.wait(JOIN_S)
        mover = _Thread(lambda: z.move_to_um(7.0))
        assert queued.wait(JOIN_S)
        z.stop()
        release.set()
        snapper.join()
        mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error).startswith(
            "`z ZDrive`: it was stopped before it started"
        )
        assert stand.core.log == ["stop('ZDrive')", "snapImage()"]
        assert stand.core.isContinuousFocusLocked() is True  # untouched


def test_z_stop_during_the_suspend_cancels_the_move(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.engage_directly()
        z, switch = stand.z, stand.core.enableContinuousFocus

        def stopped_meanwhile(enable: bool) -> None:
            switch(enable)
            if not enable:
                z.stop()  # the operator's stop, while the suspend is sent

        monkeypatch.setattr(stand.core, "enableContinuousFocus", stopped_meanwhile)
        with caplog.at_level(logging.WARNING), pytest.raises(MotionStoppedError):
            z.move_to_um(12.0)
        assert stand.core.log == [OFF, "stop('ZDrive')"]
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(MotionStoppedError)"
        ) in caplog.messages


def test_disengage_during_a_z_move_is_not_undone_by_the_relock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # disengage() is a safe call: it wins over the actions called before it,
    # as a close wins over an open (design §13).
    with _ti2() as stand:
        stand.engage_directly()
        z, af = stand.z, stand.af
        stand.core.busy_devices.add("ZDrive")
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: z.move_to_um(12.0))
            _wait_for(_moving(stand, "z ZDrive"), "the Z move to wait")
            af.disengage()
            stand.core.busy_devices.discard("ZDrive")
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == (
            "`autofocus PFS`: disengage() was called during z: move_to 12.0 µm, so "
            "it was left off; read the position before moving on"
        )
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", OFF]
        assert stand.core.isContinuousFocusEnabled() is False
        assert (
            "autofocus PFS: left off, because disengage() was called during z: "
            "move_to 12.0 µm"
        ) in caplog.messages


def test_disengage_during_focus_by_is_not_undone_by_the_relock() -> None:
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af = stand.af
        stand.core.busy_devices.add("PFSOffset")
        mover = _Thread(lambda: af.focus_by_um(2.0))
        _wait_for(_moving(stand, "autofocus_offset PFSOffset"), "the offset move")
        af.disengage()
        stand.core.busy_devices.discard("PFSOffset")
        mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert "disengage() was called during autofocus: focus_by 2.0 µm" in str(
            mover.error
        )
        assert stand.core.log == ["setPosition('PFSOffset', 7794.0)", OFF]
        assert stand.core.isContinuousFocusEnabled() is False


# --- the operator's Z stop wins over the relock (the review of #140) ---------


def _sent_and_waiting(stand: Stand, sent: list[str], waits: str) -> Callable[[], bool]:
    """``sent`` is the whole log, and the action waits for ``waits``: no command is half-sent."""
    return lambda: stand.core.log == sent and _moving(stand, waits)()


def test_z_stop_during_the_relock_switches_pfs_off_and_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # PFS searching for its lock drives Z (FM-88): the operator's Z stop
    # switches it off before Z's own stop, and the move ends as stopped.
    with _ti2(_profile(autofocus={"lock_timeout_s": 2 * JOIN_S})) as stand:
        stand.engage_directly(lockable=False)
        z = stand.z
        relocking = [OFF, "setPosition('ZDrive', 12.0)", ON]
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: z.move_to_um(12.0))
            _wait_for(_sent_and_waiting(stand, relocking, "autofocus PFS"), "relock")
            z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == (
            "`z ZDrive`: it was stopped during z: move_to 12.0 µm, so autofocus PFS "
            "was left off; read the position before moving on"
        )
        # Switched off before Z's stop, and nothing switches it on again.
        assert stand.core.log == [*relocking, OFF, "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(MotionStoppedError)"
        ) in caplog.messages
        assert stand.microscope.state().moving == ()
        # With no compound running, a Z stop leaves PFS alone (m2 §3.3).
        stand.engage_directly()
        z.stop()
        assert stand.core.log == ["stop('ZDrive')"]
        assert stand.core.isContinuousFocusLocked() is True


@pytest.mark.parametrize("phase", ["the offset move", "the relock"])
def test_z_stop_during_focus_by_switches_pfs_off_and_raises(
    phase: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _factor({"4x": -3.0}, lock_timeout_s=2 * JOIN_S)
    with _ti2(profile, core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z = stand.af, stand.z
        move, stop = stand.core.setPosition, stand.core.stop

        def moved(label: str, value: float) -> None:
            move(label, value)
            if phase == "the relock":
                stand.core.autofocus_lockable = False  # PFS searches again

        def stop_arrives(label: str) -> None:
            stop(label)
            stand.core.busy_devices.discard(label)

        monkeypatch.setattr(stand.core, "setPosition", moved)
        monkeypatch.setattr(stand.core, "stop", stop_arrives)
        if phase == "the offset move":
            stand.core.busy_devices.add("PFSOffset")
            sent, waits = (
                ["setPosition('PFSOffset', 7794.0)"],
                "autofocus_offset PFSOffset",
            )
            stops = [OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        else:
            sent, waits = ["setPosition('PFSOffset', 7794.0)", ON], "autofocus PFS"
            stops = [OFF, "stop('ZDrive')"]  # the offset device is idle
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.focus_by_um(2.0))
            _wait_for(_sent_and_waiting(stand, sent, waits), phase)
            z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == (
            "`z ZDrive`: it was stopped during autofocus: focus_by 2.0 µm, so "
            "autofocus PFS was left off; read the position before moving on"
        )
        assert stand.core.log == [*sent, *stops]
        assert stand.core.isContinuousFocusEnabled() is False
        assert (
            "autofocus PFS: left off, because autofocus: focus_by 2.0 µm did not "
            "complete (MotionStoppedError)"
        ) in caplog.messages
        assert stand.microscope.state().moving == ()


def test_z_stop_after_the_move_and_before_the_relock_leaves_pfs_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stop lands in the move's readback, once the move has passed its own
    # check: the relock, which would move Z after it, never starts.
    with _ti2() as stand:
        stand.engage_directly()
        z, read = stand.z, stand.core.getPosition
        stopped: list[str] = []

        def stop_on_arrival(label: str | None = None) -> float:
            if label == "ZDrive" and not stopped:
                stopped.append(label)
                z.stop()
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", stop_on_arrival)
        with caplog.at_level(logging.WARNING), pytest.raises(MotionStoppedError) as e:
            z.move_to_um(12.0)
        assert str(e.value) == (
            "`z ZDrive`: it was stopped during z: move_to 12.0 µm, so autofocus PFS "
            "was left off; read the position before moving on"
        )
        assert stand.core.log == [OFF, "setPosition('ZDrive', 12.0)", "stop('ZDrive')"]
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(MotionStoppedError)"
        ) in caplog.messages


def test_z_stop_once_it_has_locked_again_still_switches_it_off_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stop lands in the lock wait's readback: the relock has not returned,
    # so the call ends stopped, with PFS off, not as a success.
    with _ti2() as stand:
        stand.engage_directly()
        z, locked = stand.z, stand.core.isContinuousFocusLocked
        reads: list[int] = []

        def stopped_on_the_readback() -> bool:
            reads.append(1)
            if len(reads) == 2:  # the poll's read, then the readback's
                z.stop()
                return True
            return locked()

        monkeypatch.setattr(
            stand.core, "isContinuousFocusLocked", stopped_on_the_readback
        )
        with pytest.raises(MotionStoppedError) as stopped:
            z.move_to_um(12.0)
        assert str(stopped.value).startswith("`z ZDrive`: it was stopped during z:")
        assert stand.core.log == [
            OFF,
            "setPosition('ZDrive', 12.0)",
            ON,
            OFF,
            "stop('ZDrive')",
        ]
        assert stand.core.isContinuousFocusEnabled() is False


def test_z_stop_after_focus_bys_offset_move_is_not_reported_as_a_disengage(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Z stop switches PFS off through the key disengage() uses: the
    # WARNING names the stop, not a disengage() nobody called.
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, read = stand.af, stand.z, stand.core.getPosition
        stopped: list[str] = []

        def stop_on_arrival(label: str | None = None) -> float:
            if label == "PFSOffset" and stand.core.log and not stopped:
                stopped.append(label)
                z.stop()
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", stop_on_arrival)
        with caplog.at_level(logging.WARNING), pytest.raises(MotionStoppedError):
            af.focus_by_um(2.0)
        assert stand.core.log == [
            "setPosition('PFSOffset', 7794.0)",
            OFF,
            "stop('ZDrive')",
        ]
        assert [m for m in caplog.messages if "left off" in m] == [
            "autofocus PFS: left off, because autofocus: focus_by 2.0 µm did not "
            "complete (MotionStoppedError)"
        ]


@pytest.mark.parametrize("then", ["halt", "ctrl-c"])
def test_a_halt_or_a_ctrl_c_after_a_z_stop_is_not_turned_into_a_stop(
    then: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The emergency stop and a Ctrl-C mean more than the Z stop before them:
    # what the caller gets is theirs, and the WARNING names it.
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, read = stand.af, stand.z, stand.core.getPosition
        z_stopped: list[str] = []

        def stop_on_arrival(label: str | None = None) -> float:
            if label == "PFSOffset" and stand.core.log and not z_stopped:
                z_stopped.append(label)
                z.stop()
                if then == "halt":
                    stand.microscope.stop()
                else:
                    raise KeyboardInterrupt  # raised from the stub
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", stop_on_arrival)
        raised = MicroscopeHaltedError if then == "halt" else KeyboardInterrupt
        with caplog.at_level(logging.WARNING), pytest.raises(raised):
            af.focus_by_um(2.0)
        assert (
            f"autofocus PFS: left off, because autofocus: focus_by 2.0 µm did not "
            f"complete ({raised.__name__})"
        ) in caplog.messages
        stand.microscope.resume()


def test_z_stop_while_focus_by_waits_for_the_lock_cancels_it_and_leaves_pfs_locked(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As for a Z move (design §13): cancelled before anything is sent, so PFS,
    # which still holds focus, is left alone (beyond the plan).
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, camera = stand.af, stand.z, stand.microscope.require(Camera)
        executor = stand.microscope._executor
        queued, held, release = threading.Event(), threading.Event(), threading.Event()
        contention, snap = executor._contention, stand.core.snapImage

        def noted(*args: Any, **kwargs: Any) -> None:
            queued.set()
            contention(*args, **kwargs)

        def hung_snap() -> None:
            held.set()
            release.wait(JOIN_S)
            snap()

        monkeypatch.setattr(executor, "_contention", noted)
        monkeypatch.setattr(stand.core, "snapImage", hung_snap)
        snapper = _Thread(camera.snap)
        assert held.wait(JOIN_S)
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.focus_by_um(2.0))
            assert queued.wait(JOIN_S)
            z.stop()
            release.set()
            snapper.join()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == (
            "`z ZDrive`: it was stopped before it started; read the position before "
            "moving on"
        )
        assert stand.core.log == ["stop('ZDrive')", "snapImage()"]
        assert stand.core.isContinuousFocusLocked() is True
        assert not any("left off" in m for m in caplog.messages)


def test_emergency_stop_during_the_relock_switches_pfs_off_once() -> None:
    # Halted, the Z stop adds nothing to the emergency stop's own switch-off
    # (m2 §3.11): sent twice, it would delay the stages' stops.
    with _ti2(_profile(autofocus={"lock_timeout_s": 2 * JOIN_S})) as stand:
        stand.engage_directly(lockable=False)
        z = stand.z
        relocking = [OFF, "setPosition('ZDrive', 12.0)", ON]
        mover = _Thread(lambda: z.move_to_um(12.0))
        _wait_for(_sent_and_waiting(stand, relocking, "autofocus PFS"), "relock")
        stand.microscope.stop()
        mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert stand.core.log == [
            *relocking,
            OFF,
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        assert stand.core.isContinuousFocusEnabled() is False
        stand.microscope.resume()


def test_engage_that_loses_the_lock_and_cannot_switch_off_stays_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As after a failed switch-off at the deadline (m2 §3.2): the stand
    # refuses actions while PFS may still search (adversarial review of #13).
    with _ti2() as stand:
        reads: list[int] = []

        def lost_after_the_first_read() -> bool:
            reads.append(1)
            return len(reads) == 1

        switch = stand.core.enableContinuousFocus

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        monkeypatch.setattr(
            stand.core, "isContinuousFocusLocked", lost_after_the_first_read
        )
        monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
        with pytest.raises(DeviceTimeoutError) as timed_out:
            stand.af.engage()
        assert str(timed_out.value).startswith(
            "autofocus PFS: it locked, then lost the lock at once, and it could not "
            "be switched off (the stop failed ("
        )
        assert str(timed_out.value).endswith(
            "; it may still move Z: stop() and check the stand, then resume()"
        )
        assert stand.microscope.state().moving == ("autofocus PFS",)
        with pytest.raises(MotionInProgressError):
            stand.microscope.require(XYStage).move_to_um(1.0, 0.0)
        monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
        stand.microscope.resume()


# --- range, slot and faults --------------------------------------------------


def test_in_range_reads_the_profile_property() -> None:
    pfs = _pfs()
    with _ti2(_profile(autofocus=IN_RANGE), core=_ti2_core(pfs=pfs)) as stand:
        assert stand.af.is_in_range() is False
        pfs.properties["PFS in Range"].value = "Within Range"
        assert stand.af.is_in_range() is True
    with _ti2() as stand:
        # Without the profile entry the layer does not guess: unknown.
        assert stand.af.is_in_range() is None


def test_in_range_with_a_missing_property_is_a_hardware_error() -> None:
    profile = _profile(
        autofocus={"in_range_property": "PFS Range", "in_range_values": ["In"]}
    )
    with _ti2(profile) as stand, pytest.raises(HardwareError) as failed:
        stand.af.is_in_range()
    assert str(failed.value).startswith(
        "autofocus: getProperty('PFS', 'PFS Range') failed (Cannot get value of "
        'property "PFS Range"); check the device'
    )


def test_facade_points_the_autofocus_slot_at_the_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    core = _ti2_core(extra=[FakeDevice("PFS2", DeviceType.AutoFocus)])
    profile = _profile(roles={"assign": {"autofocus": "PFS2"}})
    with caplog.at_level(logging.INFO):
        microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    try:
        assert core.getAutoFocusDevice() == "PFS2"
        assert "setAutoFocusDevice('PFS2')" in core.log
        assert (
            "autofocus: the core's AutoFocus slot now names 'PFS2' (was 'PFS'), as "
            "the roles say"
        ) in caplog.messages
        assert microscope.require(Autofocus).engage() is True
        core.setAutoFocusDevice("PFS")
        assert core.isContinuousFocusEnabled() is False  # PFS was never switched
        core.setAutoFocusDevice("PFS2")
    finally:
        microscope.close()
    # When the slot already names the role, nothing is sent.
    core = _ti2_core()
    with _ti2(core=core):
        pass
    assert not any(entry.startswith("setAutoFocusDevice") for entry in core.log)


def test_a_slot_that_cannot_take_the_role_is_a_warning_not_a_failed_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A profile that assigns the role to a device of another type: the open
    # goes on, and every autofocus call and Z move refuses (beyond the plan).
    profile = _profile(roles={"assign": {"autofocus": "PFSOffset"}})
    with caplog.at_level(logging.WARNING), _ti2(profile) as stand:
        assert stand.core.getAutoFocusDevice() == "PFS"
        with pytest.raises(HardwareError, match="is not the device the core's"):
            stand.af.is_engaged()
        with pytest.raises(HardwareError, match="is not the device the core's"):
            stand.z.move_to_um(1.0)
        assert stand.core.log == []
    assert any(
        m.startswith(
            "autofocus: the core's AutoFocus slot names 'PFS' and cannot be pointed "
            "at 'PFSOffset'"
        )
        for m in caplog.messages
    )


def test_autofocus_call_refused_when_the_slot_names_another_device() -> None:
    core = _ti2_core(extra=[FakeDevice("PFS2", DeviceType.AutoFocus)])
    with _ti2(_factor({"4x": -3.0}), core=core) as stand:
        af = stand.af
        stand.core.setAutoFocusDevice("PFS2")  # behind the layer
        stand.core.log.clear()
        for call in (
            af.is_engaged,
            af.is_locked,
            af.engage,
            af.disengage,
            lambda: af.focus_by_um(1.0),
            lambda: stand.z.move_to_um(1.0),
        ):
            with pytest.raises(HardwareError) as refused:
                call()
            assert str(refused.value).startswith(
                "autofocus 'PFS' is not the device the core's AutoFocus slot names "
                "('PFS2'), and the continuous-focus calls act on the slot's device;"
            )
        assert stand.core.log == []
        stand.core.setAutoFocusDevice("PFS")


def test_a_continuous_focus_fault_is_a_hardware_error_naming_autofocus() -> None:
    with _ti2() as stand:
        boom = RuntimeError("PFS: no answer")
        stand.core.failing["PFS"] = boom
        with pytest.raises(HardwareError) as failed:
            stand.af.is_engaged()
        assert str(failed.value) == (
            "autofocus: isContinuousFocusEnabled() failed (PFS: no answer); check "
            "the device: power, cable, another program holding it"
        )
        assert failed.value.__cause__ is boom
        with pytest.raises(HardwareError) as engage_failed:
            stand.af.engage()
        assert str(engage_failed.value).startswith(
            "autofocus: enableContinuousFocus(True) failed (PFS: no answer)"
        )
        stand.core.failing.clear()
        stand.microscope.resume()


def test_an_offset_device_fault_names_autofocus_offset() -> None:
    with _ti2() as stand:
        stand.core.failing["PFSOffset"] = RuntimeError("PFSOffset: no answer")
        with pytest.raises(HardwareError) as failed:
            stand.af.offset_native()
        assert str(failed.value).startswith(
            "autofocus_offset: getPosition('PFSOffset') failed (PFSOffset: no answer)"
        )
        with pytest.raises(HardwareError) as set_failed:
            stand.af.set_offset_native(1.0)
        assert str(set_failed.value).startswith(
            "autofocus_offset: setPosition('PFSOffset', 1.0) failed"
        )
        stand.core.failing.clear()
        stand.microscope.resume()


@pytest.mark.parametrize("dry_run", [False, True])
def test_the_autofocus_devices_own_offset_is_refused_when_the_slot_names_another(
    dry_run: bool,
) -> None:
    # FM-87: without an offset role, the offset is the slot's device's;
    # setAutoFocusOffset would move another device's.
    core = FakeCore.demo_like()
    with _ti2(Profile.demo(), dry_run=dry_run, core=core) as stand:
        af = stand.af
        core.unloadDevice("Autofocus")  # the slot is emptied behind the layer
        core.log.clear()
        for call in (
            af.offset_native,
            lambda: af.set_offset_native(1.0),
            lambda: af.offset_by_native(1.0, force=True),
        ):
            with pytest.raises(HardwareError) as refused:
                call()
            assert str(refused.value).startswith(
                "autofocus 'Autofocus' is not the device the core's AutoFocus slot "
                "names (''),"
            )
        assert core.log == []


def test_a_stuck_offset_device_is_stopped_at_the_core_timeout() -> None:
    # FM-17: the offset device's move gives up like a stage move, stop first.
    profile = _profile().model_copy(
        update={
            "micromanager": _profile().micromanager.model_copy(
                update={"device_timeout_ms": 200}
            )
        }
    )
    with _ti2(profile) as stand:
        stand.core.busy_devices.add("PFSOffset")
        with pytest.raises(DeviceTimeoutError, match="the stop was sent"):
            stand.af.set_offset_native(1.0)
        assert stand.core.log == ["setPosition('PFSOffset', 1.0)", "stop('PFSOffset')"]
        assert stand.microscope.state().moving == ("autofocus_offset PFSOffset",)
        stand.core.busy_devices.clear()
        assert stand.microscope.state().moving == ("autofocus_offset PFSOffset",)
        stand.microscope.resume()


def test_a_failed_in_range_read_is_unknown_in_messages_never_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing = {"in_range_property": "PFS Range", "in_range_values": ["In"]}
    with caplog.at_level(logging.WARNING):
        with _ti2(_profile(autofocus=missing)) as stand:
            stand.core.autofocus_lockable = False
            assert stand.af.engage() is False  # no HardwareError from the read
        pfs = _pfs()
        pfs.properties["PFS in Range"].value = "Within Range"
        with _ti2(_profile(autofocus=IN_RANGE), core=_ti2_core(pfs=pfs)) as stand:
            stand.core.autofocus_lockable = False
            assert stand.af.engage() is False
    assert [m for m in caplog.messages if m.startswith("autofocus PFS: no lock")] == [
        f"autofocus PFS: no lock within 0.2 s {ASSUMED}; in range: unknown; "
        "switched off",
        f"autofocus PFS: no lock within 0.2 s {ASSUMED}; in range: yes; switched off",
    ]


@pytest.mark.parametrize(
    ("kwargs", "phrase"),
    [
        ({"lock_timeout_s": 0.0}, "lock_timeout_s must be finite and in (0, 600] s"),
        (
            {"lock_timeout_s": math.nan},
            "lock_timeout_s must be finite and in (0, 600] s",
        ),
        (
            {"offset_per_um": {"4x": 0.0}},
            "offset_per_um for objective '4x' must be finite and non-zero",
        ),
        (
            {"offset_per_um": {"4x": math.inf}},
            "offset_per_um for objective '4x' must be finite and non-zero",
        ),
    ],
)
def test_mm_autofocus_built_in_code_refuses_what_the_profile_would(
    kwargs: dict[str, Any], phrase: str
) -> None:
    # A tool may build the backend without a profile; the same bounds hold.
    core = _ti2_core()
    with _ti2(core=core) as stand:
        executor, safety = stand.microscope._executor, stand.microscope._safety
        with pytest.raises(ValueError, match=re.escape(phrase)):
            mm.MMAutofocus(
                cast("CMMCorePlus", core),
                "PFS",
                executor,
                safety,
                offset_label="PFSOffset",
                **kwargs,
            )


# --- the facade (m2 §3.11) ---------------------------------------------------


def test_emergency_stop_switches_pfs_off_before_the_stages(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.require_stages = (stand.microscope.require(XYStage), stand.z)
        lines_before: dict[str, int] = {}
        stop, switch = stand.core.stop, stand.core.enableContinuousFocus

        def counted_stop(label: str) -> None:
            lines_before[label] = len(caplog.records)
            stop(label)

        def counted_switch(enable: bool) -> None:
            lines_before["PFS"] = len(caplog.records)
            switch(enable)

        monkeypatch.setattr(stand.core, "stop", counted_stop)
        monkeypatch.setattr(stand.core, "enableContinuousFocus", counted_switch)
        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            stand.microscope.stop()
        assert stand.core.log == [
            OFF,
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        # Every call before any line (FM-62).
        assert lines_before == {"PFS": 0, "PFSOffset": 0, "XYStage": 0, "ZDrive": 0}
        assert caplog.messages[:5] == [
            "microscope: halted",
            "autofocus: disengage",
            "autofocus_offset PFSOffset: stop",
            "xy_stage XYStage: stop",
            "z ZDrive: stop",
        ]
        assert stand.core.isContinuousFocusEnabled() is False
        stand.microscope.resume()


def test_emergency_stop_reports_a_failed_switch_off_and_stops_the_stages() -> None:
    with _ti2() as stand:
        stand.core.failing["PFS"] = RuntimeError("PFS: stuck")
        with pytest.raises(HardwareError) as failed:
            stand.microscope.stop()
        message = str(failed.value)
        assert message.startswith(
            "the emergency stop failed for Autofocus 'PFS' (HardwareError("
        )
        assert "failed (PFS: stuck)" in message
        assert stand.core.log == [
            OFF,
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        stand.core.failing.clear()
        stand.microscope.resume()


@pytest.mark.parametrize("how", ["stop()", "close()"])
def test_emergency_stop_stops_the_offset_device_after_the_switch_off(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review of #140: a move of the offset device is a motion moving()
    # reports; left alone, it would hold the lock until its timeout.
    with _ti2(core=_at_offset(7803.0)) as stand:
        af, stop = stand.af, stand.core.stop

        def stop_arrives(label: str) -> None:
            stop(label)
            stand.core.busy_devices.discard(label)

        monkeypatch.setattr(stand.core, "stop", stop_arrives)
        stand.core.busy_devices.add("PFSOffset")
        mover = _Thread(lambda: af.set_offset_native(7805.0))
        _wait_for(_moving(stand, "autofocus_offset PFSOffset"), "the offset move")
        if how == "stop()":
            stand.microscope.stop()
        else:
            stand.microscope.close()
        mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert mover.error.device == "autofocus_offset PFSOffset"
        assert stand.core.log[:5] == [
            "setPosition('PFSOffset', 7805.0)",
            OFF,
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        assert stand.microscope.state().moving == ()
        stand.microscope.resume()


def test_emergency_stop_reports_a_failed_offset_stop_and_still_stops_the_stages() -> (
    None
):
    with _ti2() as stand:
        stand.core.failing["PFSOffset"] = RuntimeError("PFSOffset: no answer")
        with pytest.raises(HardwareError) as failed:
            stand.microscope.stop()
        assert str(failed.value).startswith(
            "the emergency stop failed for autofocus_offset 'PFSOffset' "
            "(HardwareError(\"autofocus_offset: stop('PFSOffset') failed (PFSOffset: "
            "no answer)"
        )
        assert stand.core.log == [
            OFF,
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        stand.core.failing.clear()
        stand.microscope.resume()


def test_close_with_nothing_moving_leaves_pfs_as_it_is() -> None:
    core = _ti2_core()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    core.enableContinuousFocus(True)
    core.log.clear()
    microscope.close()
    # PFS holding focus after smc exits is a legitimate state.
    assert core.log == ["unloadAllDevices()"]


def test_state_reports_the_autofocus() -> None:
    with _ti2(core=_at_offset(7803.0)) as stand:
        stand.engage_directly()
        state = stand.microscope.state()
        assert (
            state.autofocus_engaged,
            state.autofocus_locked,
            state.autofocus_offset_native,
        ) == (True, True, 7803.0)
        assert state.errors == []
        stand.engage_directly(lockable=False)
        state = stand.microscope.state()
        assert (state.autofocus_engaged, state.autofocus_locked) == (True, False)
    without = FakeCore([FakeDevice("Z", DeviceType.Stage)], focus="Z")
    with Microscope.from_core(cast("CMMCorePlus", without), _profile()) as microscope:
        state = microscope.state()
        assert (
            state.autofocus_engaged,
            state.autofocus_locked,
            state.autofocus_offset_native,
        ) == (None, None, None)
        assert Autofocus not in microscope.available()


def test_state_lists_a_failed_autofocus_read_as_an_error() -> None:
    with _ti2() as stand:
        stand.core.failing["PFS"] = RuntimeError("PFS: no answer")
        state = stand.microscope.state()
        stand.core.failing.clear()
    assert (state.autofocus_engaged, state.autofocus_locked) == (None, None)
    assert state.autofocus_offset_native == 0.0  # PFSOffset still answers
    tail = "() failed (PFS: no answer); check the device: power, cable, another program holding it"
    assert state.errors == [
        f"autofocus: HardwareError: autofocus: isContinuousFocusEnabled{tail}",
        f"autofocus: HardwareError: autofocus: isContinuousFocusLocked{tail}",
    ]
    assert state.z_um == 0.0  # one failing device hides no other


def test_describe_shows_the_autofocus() -> None:
    with _ti2() as stand:
        line = "XY (0.0, 0.0) µm | Z 0.00 µm | 10 ms | autofocus"
        assert stand.microscope.describe() == f"{line} off"
        stand.engage_directly()
        assert stand.microscope.describe() == f"{line} locked"
        stand.engage_directly(lockable=False)
        assert stand.microscope.describe() == f"{line} on, not locked"
        stand.core.failing["PFS"] = RuntimeError("PFS: no answer")
        assert stand.microscope.describe() == f"{line} ?"
        stand.core.failing.clear()
    core = FakeCore.demo_like()
    with Microscope.from_core(cast("CMMCorePlus", core), Profile.demo()) as demo_like:
        # After the shutter's part.
        assert demo_like.describe() == (
            "XY (0.0, 0.0) µm | Z 0.00 µm | 10 ms | shutter closed | autofocus off"
        )


class _StubAutofocus:
    """An ``Autofocus`` that records what it is asked: a test double, not interlocked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _note(self, name: str) -> None:
        self.calls.append(name)

    def is_engaged(self) -> bool:
        self._note("is_engaged")
        return True

    def is_locked(self) -> bool:
        self._note("is_locked")
        return True

    def is_in_range(self) -> bool | None:
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        self._note("engage")
        return True

    def disengage(self) -> None:
        self._note("disengage")

    def offset_native(self) -> float:
        return 0.0

    def set_offset_native(self, value: float) -> float:
        return value

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        return delta

    def offset_unit(self) -> str:
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        return None

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        return FocusStep("z", dz_um, None)


def test_override_of_autofocus_warns_that_z_is_no_longer_interlocked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _ti2() as stand:
        z = stand.z  # built before the override: the interlock is asked per move
        stub = _StubAutofocus()
        with caplog.at_level(logging.WARNING):
            stand.microscope.override(Autofocus, stub)
        assert (
            "Autofocus overridden: Z moves no longer suspend and re-engage it"
        ) in caplog.messages
        stand.engage_directly()
        assert z.move_to_um(12.0) == 12.0
        assert stand.core.log == ["setPosition('ZDrive', 12.0)"]
        assert stub.calls == []
        # The quirk switched PFS off, and nothing switched it on again.
        assert stand.core.isContinuousFocusEnabled() is False
