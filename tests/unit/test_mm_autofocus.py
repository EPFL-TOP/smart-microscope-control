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
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from pymmcore_plus import DeviceType

import smc.hardware.backends.mm as mm
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Channels,
    FocusStep,
    LightSource,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import (
    AutofocusLostError,
    CapabilityMissingError,
    DeviceTimeoutError,
    HardwareError,
    MicroscopeHaltedError,
    MotionInProgressError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.microscope import CAPABILITY_ROLES
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
#: How long a slow fake switch-off gives the caller to finish before it
#: returns or fails (FM-96): well below ``Z_STOP_OUTCOME_WAIT_S``, so a caller
#: that waits for the switch-off's outcome is still waiting by then.
SETTLE_S = 0.5
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

    def finished_within(self, timeout_s: float) -> bool:
        """Join for up to ``timeout_s``; whether the thread has ended."""
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()


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


def test_halt_between_the_move_and_the_relock_says_pfs_was_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #141 item 2: the emergency stop lands once the move has passed its own
    # check; the relock is refused before it sends anything, and says so.
    with _ti2() as stand:
        stand.engage_directly()
        read = stand.core.getPosition
        halted: list[str] = []

        def halt_on_arrival(label: str | None = None) -> float:
            if label == "ZDrive" and not halted:
                halted.append(label)
                stand.microscope.stop()
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", halt_on_arrival)
        with caplog.at_level(logging.WARNING), pytest.raises(MicroscopeHaltedError):
            stand.z.move_to_um(12.0)
        _no_relock_after_the_move(stand.core.log)
        assert (
            "autofocus PFS: left off, because z: move_to 12.0 µm did not complete "
            "(MicroscopeHaltedError)"
        ) in caplog.messages
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


def test_engage_whose_state_cannot_be_read_after_the_wait_switches_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #141 item 1: unknown counts as searching (FM-19); left on, PFS could move
    # Z with no action holding the lock, so it is given up on as at a deadline.
    with _ti2() as stand:
        locked = stand.core.isContinuousFocusLocked
        reads: list[int] = []

        def unreadable_on_the_readback() -> bool:
            reads.append(1)
            if len(reads) == 2:  # the poll's read, then the readback's
                raise RuntimeError("PFS: no answer")
            return locked()

        monkeypatch.setattr(
            stand.core, "isContinuousFocusLocked", unreadable_on_the_readback
        )
        with pytest.raises(HardwareError) as failed:
            stand.af.engage()
        assert str(failed.value).startswith(
            "autofocus: isContinuousFocusLocked() failed (PFS: no answer)"
        )
        assert stand.core.log == [ON, OFF]
        assert stand.core.isContinuousFocusEnabled() is False
        assert stand.microscope.state().moving == ()


def test_engage_whose_enabled_state_cannot_be_read_after_the_wait_switches_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The twin of the test above, for the readback's second read.
    with _ti2() as stand:
        enabled = stand.core.isContinuousFocusEnabled
        reads: list[int] = []

        def idle_then_unreadable() -> bool:
            reads.append(1)
            if len(reads) == 1:
                return False  # the poll: not searching, so the wait ends
            if len(reads) == 2:  # the readback
                raise RuntimeError("PFS: no answer")
            return enabled()

        monkeypatch.setattr(stand.core, "isContinuousFocusLocked", lambda: False)
        monkeypatch.setattr(
            stand.core, "isContinuousFocusEnabled", idle_then_unreadable
        )
        with pytest.raises(HardwareError) as failed:
            stand.af.engage()
        assert str(failed.value).startswith(
            "autofocus: isContinuousFocusEnabled() failed (PFS: no answer)"
        )
        assert stand.core.log == [ON, OFF]
        assert stand.microscope.state().moving == ()


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


def test_offset_change_while_locked_without_a_focus_drive_has_only_the_offset_guards() -> (
    None
):
    devices = [
        FakeDevice("PFS", DeviceType.AutoFocus),
        FakeDevice("PFSOffset", DeviceType.Stage),
        FakeDevice("XYStage", DeviceType.XYStage),
        FakeDevice("Camera-1", DeviceType.Camera),
        FakeDevice("Nosepiece", DeviceType.State, state_labels=("4x", "10x", "40x")),
    ]
    core = FakeCore(devices, camera="Camera-1", xy_stage="XYStage", autofocus="PFS")
    core.setPosition("PFSOffset", 7800.0)
    profile = _profile(autofocus={"offset_per_um": {"4x": -2.0}}, safety=_Z_LIMITS)
    with _ti2(profile, core=core) as stand:
        assert stand.microscope.roles.get(Role.focus) is None
        stand.engage_directly()
        assert stand.af.set_offset_native(7796.0) == 7796.0


def test_offset_change_with_pfs_on_but_not_locked_is_not_held_to_the_z_limits() -> None:
    # Searching, PFS follows no offset yet: the change implies no Z.
    profile = _profile(autofocus={"offset_per_um": {"4x": -2.0}}, safety=_Z_LIMITS)
    with _ti2(profile, core=_at_offset_and_z()) as stand:
        stand.engage_directly(lockable=False)
        assert stand.af.set_offset_native(7796.0) == 7796.0


# --- focus_by_um (m2 §3.5) ---------------------------------------------------


class _StubZ:
    """A focus drive that ``Microscope.override`` put in: not interlocked, not linked."""

    def position_um(self) -> float:
        return 0.0

    def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float:
        return z_um

    def move_by_um(
        self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
    ) -> float:
        return dz_um

    def wait(self, timeout_s: float | None = None) -> None:
        return None

    def is_busy(self) -> bool:
        return False

    def stop(self) -> None:
        return None

    def limits_um(self) -> None:
        return None


def test_locked_focus_by_with_an_overridden_focus_drive_links_no_z_stop() -> None:
    # Its stop does not go through the Executor: there is nothing to link.
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.microscope.override(ZStage, _StubZ())
        stand.engage_directly()
        assert stand.af.focus_by_um(2.0) == FocusStep("offset", 0.0, 7794.0)
        assert stand.core.log == ["setPosition('PFSOffset', 7794.0)", ON]


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


def test_focus_by_past_the_z_soft_limits_and_the_offset_jog_says_the_z_limit_first() -> (
    None
):
    # Adversarial review of #158: the Z soft limits cannot be forced, the
    # offset jog can. Named first, the jog would invite a force=True that the
    # Z limit then refuses anyway.
    profile = _profile(
        autofocus={"offset_per_um": {"4x": -3.0}},
        safety={"z_soft_limits_um": (-10.0, 10.0), "max_autofocus_offset_jog": 10.0},
    )
    with _ti2(profile, core=_at_offset_and_z()) as stand:
        stand.engage_directly()
        with pytest.raises(SafetyRefusedError) as refused:
            stand.af.focus_by_um(5.0)  # Z 9 -> 14 µm; the offset by -15
        assert refused.value.reason == (
            "Z target 14.0 µm is outside the soft limits [-10.0, 10.0] µm; change "
            "[safety] in the profile if the limits are wrong"
        )
        assert refused.value.how_to_force == ""
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
            "stop('PFSOffset')",  # its move has not returned yet: still pending
            "stop('ZDrive')",
        ]
        assert [m for m in caplog.messages if "left off" in m] == [
            "autofocus PFS: left off, because autofocus: focus_by 2.0 µm did not "
            "complete (MotionStoppedError)"
        ]


def test_z_stop_before_focus_bys_offset_move_is_sent_cancels_it(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adversarial review of the fix round: the stop lands while the offset
    # action reads its anchor. Sent after it, the offset change would make
    # the next engage() drive Z by the whole step.
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, read = stand.af, stand.z, stand.core.getPosition
        stopped: list[str] = []

        def stop_on_the_anchor(label: str | None = None) -> float:
            if label == "PFSOffset" and not stopped:
                stopped.append(label)
                z.stop()
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", stop_on_the_anchor)
        with caplog.at_level(logging.WARNING), pytest.raises(MotionStoppedError) as e:
            af.focus_by_um(2.0)
        assert str(e.value) == (
            "`z ZDrive`: it was stopped during autofocus: focus_by 2.0 µm, so "
            "autofocus PFS was left off; read the position before moving on"
        )
        assert stand.core.log == [OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert stand.af.offset_native() == 7800.0
        assert (
            "autofocus PFS: left off, because autofocus: focus_by 2.0 µm did not "
            "complete (MotionStoppedError)"
        ) in caplog.messages


#: ``str()`` of the ``HardwareError`` that a stuck switch-off raises.
_STUCK = (
    "autofocus: enableContinuousFocus(False) failed (PFS: stuck on); check the "
    "device: power, cable, another program holding it"
)
_STUCK_ON = f"autofocus PFS: the Z stop during {{}} could not switch it off ({_STUCK}); it may still move Z"


def _not_switched_off(description: str, failure: str = _STUCK) -> str:
    """What the caller gets when the Z stop's switch-off failed (m2 §3.3.1, #158)."""
    return (
        f"`z ZDrive`: it was stopped during {description}, but autofocus PFS could "
        f"not be switched off ({failure}); it may still move Z: stop() and check "
        f"the stand, then resume(); read the position before moving on"
    )


def test_z_stop_whose_switch_off_fails_during_the_relock_does_not_say_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adversarial review of the fix round: the stopper gets the failure, and
    # the mover is not told that PFS was left off while it may still search.
    with _ti2(_profile(autofocus={"lock_timeout_s": 2 * JOIN_S})) as stand:
        stand.engage_directly(lockable=False)
        z, switch = stand.z, stand.core.enableContinuousFocus
        relocking = [OFF, "setPosition('ZDrive', 12.0)", ON]

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: z.move_to_um(12.0))
            _wait_for(_sent_and_waiting(stand, relocking, "autofocus PFS"), "relock")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
            with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                z.stop()
            assert stand.core.log == [*relocking, OFF, "stop('ZDrive')"]
            assert stand.core.isContinuousFocusEnabled() is True
            # The emergency stop, with a switch-off that works, ends the wait.
            monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
            stand.microscope.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _not_switched_off("z: move_to 12.0 µm")
        # Chained from the wait's own error, which the emergency stop ended.
        assert isinstance(mover.error.__cause__, MotionStoppedError)
        assert mover.error.__cause__.device == "autofocus PFS"
        assert not any("left off" in m for m in caplog.messages)
        assert _STUCK_ON.format("z: move_to 12.0 µm") in caplog.messages
        stand.microscope.resume()


@pytest.mark.parametrize("failing", ["the switch-off", "the offset stop", "both"])
def test_a_z_stop_that_fails_during_focus_bys_offset_move_still_stops_z(
    failing: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every step is tried past a failure, the switch-off's is the one raised,
    # and "left off" is claimed only when the switch-off went through.
    profile = _factor({"4x": -3.0}, lock_timeout_s=2 * JOIN_S)
    with _ti2(profile, core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z = stand.af, stand.z
        switch, stop = stand.core.enableContinuousFocus, stand.core.stop

        def switch_stuck(enable: bool) -> None:
            if not enable and failing != "the offset stop":
                stand.core.log.append(OFF)
                raise RuntimeError("PFS: stuck on")
            switch(enable)

        def stop_stuck(label: str) -> None:
            if label == "PFSOffset" and failing != "the switch-off":
                stand.core.log.append(f"stop({label!r})")
                raise RuntimeError("PFSOffset: no answer")
            stop(label)

        sent = ["setPosition('PFSOffset', 7794.0)"]
        stand.core.busy_devices.add("PFSOffset")
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.focus_by_um(2.0))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            monkeypatch.setattr(stand.core, "enableContinuousFocus", switch_stuck)
            monkeypatch.setattr(stand.core, "stop", stop_stuck)
            raised = (
                "PFSOffset: no answer"
                if failing == "the offset stop"
                else ("PFS: stuck on")
            )
            with pytest.raises(HardwareError, match=re.escape(f"({raised})")):
                z.stop()
            assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
            monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
            monkeypatch.setattr(stand.core, "stop", stop)
            stand.core.busy_devices.discard("PFSOffset")
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        switched_off = failing == "the offset stop"
        description = "autofocus: focus_by 2.0 µm"
        assert str(mover.error) == (
            _z_stopped_during(description)
            if switched_off
            else _not_switched_off(description)
        )
        assert any("left off" in m for m in caplog.messages) is switched_off
        assert (
            _STUCK_ON.format("autofocus: focus_by 2.0 µm") in caplog.messages
        ) is not switched_off


def test_z_stop_during_focus_by_without_an_offset_device_is_the_switch_off_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The demo's layout: the offset is the autofocus device's own and cannot
    # be stopped; the switch-off, keyed by the same label, cancels its move.
    devices = [
        FakeDevice("ZDrive", DeviceType.Stage),
        FakeDevice("PFS", DeviceType.AutoFocus),
        FakeDevice("XYStage", DeviceType.XYStage),
        FakeDevice("Camera-1", DeviceType.Camera),
        FakeDevice("Nosepiece", DeviceType.State, state_labels=("4x", "10x", "40x")),
    ]
    core = FakeCore(
        devices, camera="Camera-1", xy_stage="XYStage", focus="ZDrive", autofocus="PFS"
    )
    with _ti2(_factor({"4x": -3.0}, lock_timeout_s=2 * JOIN_S), core=core) as stand:
        assert stand.microscope.roles.get(Role.autofocus_offset) is None
        stand.engage_directly()
        af, z = stand.af, stand.z
        core.busy_devices.add("PFS")
        mover = _Thread(lambda: af.focus_by_um(2.0))
        sent = ["setAutoFocusOffset(-6.0)"]
        _wait_for(_sent_and_waiting(stand, sent, "autofocus PFS offset"), "offset")
        z.stop()  # no stop of an offset device that cannot be stopped
        core.busy_devices.discard("PFS")
        mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert mover.error.device == "z ZDrive"
        assert core.log == [*sent, OFF, "stop('ZDrive')"]


@pytest.mark.parametrize("switch_off", ["works", "fails"])
@pytest.mark.parametrize("then", ["halt", "ctrl-c"])
def test_a_halt_or_a_ctrl_c_after_a_z_stop_is_not_turned_into_a_stop(
    then: str,
    switch_off: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The emergency stop and a Ctrl-C mean more than the Z stop before them:
    # what the caller gets is theirs, and the WARNING names it. Also when the
    # Z stop's switch-off failed (adversarial review of #158).
    with _ti2(_factor({"4x": -3.0}), core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, read = stand.af, stand.z, stand.core.getPosition
        switch = stand.core.enableContinuousFocus
        z_stopped: list[str] = []

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        def stop_on_arrival(label: str | None = None) -> float:
            if label == "PFSOffset" and stand.core.log and not z_stopped:
                z_stopped.append(label)
                if switch_off == "works":
                    z.stop()
                else:
                    monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
                    with pytest.raises(HardwareError):
                        z.stop()
                    monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
                if then == "halt":
                    stand.microscope.stop()
                else:
                    raise KeyboardInterrupt  # raised from the stub
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", stop_on_arrival)
        raised = MicroscopeHaltedError if then == "halt" else KeyboardInterrupt
        with caplog.at_level(logging.WARNING), pytest.raises(raised):
            af.focus_by_um(2.0)
        left_off = (
            f"autofocus PFS: left off, because autofocus: focus_by 2.0 µm did not "
            f"complete ({raised.__name__})"
        )
        if switch_off == "works":
            assert left_off in caplog.messages
        else:
            assert _STUCK_ON.format("autofocus: focus_by 2.0 µm") in caplog.messages
            assert left_off not in caplog.messages
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


# --- the operator's Z stop reaches a direct call (m2 §3.3.1, #144) -----------

_LEFT_OFF = "autofocus PFS: left off, because {} did not complete (MotionStoppedError)"


def _z_stopped_during(description: str) -> str:
    return (
        f"`z ZDrive`: it was stopped during {description}, so autofocus PFS was left "
        f"off; read the position before moving on"
    )


@contextmanager
def _lock_held_by_a_snap(
    stand: Stand, monkeypatch: pytest.MonkeyPatch
) -> Iterator[threading.Event]:
    """A snap holds the microscope lock until the block ends.

    Yields the event set once another action waits for the lock behind it.
    The snap is logged (``snapImage()``) when the block ends.
    """
    camera, executor = stand.microscope.require(Camera), stand.microscope._executor
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
    try:
        yield queued
    finally:
        release.set()
        snapper.join()


def _z_stop_while_queued(
    stand: Stand, monkeypatch: pytest.MonkeyPatch, action: Callable[[], object]
) -> _Thread:
    """``action`` queued behind a hung snap, then a Z stop, then the snap released.

    The thread that ran ``action``, joined.
    """
    z = stand.z
    with _lock_held_by_a_snap(stand, monkeypatch) as queued:
        mover = _Thread(action)
        assert queued.wait(JOIN_S)
        z.stop()
    mover.join()
    return mover


def test_z_stop_during_a_direct_engage_switches_pfs_off_and_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # PFS searching for its lock drives Z: before #144, this stop was ignored
    # and the search went on until the lock timeout.
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z = stand.af, stand.z
        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            z.stop()
            waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _z_stopped_during("autofocus: engage")
        assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False
        assert _LEFT_OFF.format("autofocus: engage") in caplog.messages
        assert stand.microscope.state().moving == ()


def test_z_stop_while_a_direct_engage_waits_for_the_lock_cancels_it_with_nothing_sent(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # M1 §13: a stop cancels the actions on its device called before it. PFS
    # is left as it was, still holding focus, so nothing says "left off".
    with _ti2() as stand:
        stand.engage_directly()
        af = stand.af
        with caplog.at_level(logging.WARNING):
            mover = _z_stop_while_queued(stand, monkeypatch, af.engage)
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == (
            "`z ZDrive`: it was stopped before it started; read the position before "
            "moving on"
        )
        assert stand.core.log == ["stop('ZDrive')", "snapImage()"]
        assert stand.core.isContinuousFocusLocked() is True
        assert not any("left off" in m for m in caplog.messages)


def test_z_stop_whose_switch_off_fails_during_a_direct_engage_does_not_say_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, switch = stand.af, stand.z, stand.core.enableContinuousFocus

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
            with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                z.stop()
            assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
            assert stand.core.isContinuousFocusEnabled() is True
            # The emergency stop, with a switch-off that works, ends the wait.
            monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
            stand.microscope.stop()
            waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _not_switched_off("autofocus: engage")
        # Chained from the wait's own error, which the emergency stop ended.
        assert isinstance(waiter.error.__cause__, MotionStoppedError)
        assert waiter.error.__cause__.device == "autofocus PFS"
        assert not any("left off" in m for m in caplog.messages)
        assert _STUCK_ON.format("autofocus: engage") in caplog.messages
        stand.microscope.resume()


def test_emergency_stop_during_a_direct_engage_switches_pfs_off_once() -> None:
    # Halted, no link is called (m2 §3.11): one switch-off, then the stages.
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af = stand.af
        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
        stand.microscope.stop()
        waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert stand.core.log == [
            ON,
            OFF,
            "stop('PFSOffset')",
            "stop('XYStage')",
            "stop('ZDrive')",
        ]
        assert stand.core.isContinuousFocusEnabled() is False
        stand.microscope.resume()


def test_direct_engage_with_an_overridden_focus_drive_links_no_z_stop() -> None:
    # Its stop does not go through the Executor: there is nothing to link.
    with _ti2() as stand:
        stand.microscope.override(ZStage, _StubZ())
        stand.core.autofocus_lockable = False
        af, executor = stand.af, stand.microscope._executor
        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
        assert executor._links == {}
        af.disengage()
        waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert waiter.error.device == "autofocus PFS"
        assert stand.core.log == [ON, OFF]


_OFFSET_CALLS: dict[str, tuple[Callable[[Autofocus], float], str]] = {
    "absolute": (
        lambda af: af.set_offset_native(7801.0),
        "autofocus_offset: move_to 7801.0 device units",
    ),
    "relative": (
        lambda af: af.offset_by_native(1.0, force=True),
        "autofocus_offset: move_by 1.0 device units",
    ),
}


def _offset_busy_until_stopped(stand: Stand, monkeypatch: pytest.MonkeyPatch) -> None:
    """``PFSOffset`` reads busy from its move until a stop of it arrives."""
    stop = stand.core.stop

    def stop_arrives(label: str) -> None:
        stop(label)
        stand.core.busy_devices.discard(label)

    monkeypatch.setattr(stand.core, "stop", stop_arrives)
    stand.core.busy_devices.add("PFSOffset")


@pytest.mark.parametrize("pfs", ["locked", "on, not locked"])
@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_z_stop_during_a_direct_offset_move_with_pfs_on_switches_it_off_and_stops_the_offset_device(
    call: str,
    pfs: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PFS follows its offset, and drives Z, as soon as it is on: locked or
    # still searching (m2 §3.3.1). The link condition is "engaged".
    move, description = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly(lockable=pfs == "locked")
        af, z = stand.af, stand.z
        _offset_busy_until_stopped(stand, monkeypatch)
        sent = ["setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: move(af))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _z_stopped_during(description)
        assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False
        assert _LEFT_OFF.format(description) in caplog.messages
        assert stand.microscope.state().moving == ()


def test_z_stop_during_a_direct_offset_move_with_pfs_off_leaves_the_move_alone(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Off, the offset moves a lens, not Z (assumed, m2 §4 item 5).
    with _ti2(core=_at_offset(7800.0)) as stand:
        af, z = stand.af, stand.z
        stand.core.busy_devices.add("PFSOffset")
        sent = ["setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.set_offset_native(7801.0))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            z.stop()
            stand.core.busy_devices.discard("PFSOffset")
            mover.join()
        assert mover.error is None
        assert mover.result == 7801.0
        assert stand.core.log == [*sent, "stop('ZDrive')"]
        assert not any("left off" in m for m in caplog.messages)


def test_z_stop_during_a_direct_offset_move_with_pfs_state_unreadable_still_switches_it_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unknown counts as on (FM-19): left on, PFS would follow the offset.
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, enabled = stand.af, stand.z, stand.core.isContinuousFocusEnabled
        reads: list[int] = []

        def unreadable_once() -> bool:
            reads.append(1)
            if len(reads) == 1:
                raise RuntimeError("PFS: no answer")
            return enabled()

        monkeypatch.setattr(stand.core, "isContinuousFocusEnabled", unreadable_once)
        _offset_busy_until_stopped(stand, monkeypatch)
        sent = ["setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.set_offset_native(7801.0))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            z.stop()
            mover.join()
        assert reads, "the state was never read"
        description = "autofocus_offset: move_to 7801.0 device units"
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _z_stopped_during(description)
        assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert _LEFT_OFF.format(description) in caplog.messages
        # An assumption, labelled as one (#158).
        assumed = (
            f"autofocus PFS: its state could not be read before {description} "
            f"(autofocus: isContinuousFocusEnabled() failed (PFS: no answer); check "
            f"the device: power, cable, another program holding it); counted as on, "
            f"so a Z stop during it switches it off"
        )
        [record] = [r for r in caplog.records if r.getMessage() == assumed]
        # A kept record must not keep the exception, its traceback, the core
        # (FM-92; adversarial review of #158).
        assert not any(isinstance(arg, BaseException) for arg in record.args or ())


def test_z_stop_before_a_direct_offset_move_is_sent_cancels_it(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stop lands while the relative move reads its anchor: sent after it,
    # the offset change would drive Z after the operator's stop.
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, read = stand.af, stand.z, stand.core.getPosition
        stopped: list[str] = []

        def stop_on_the_anchor(label: str | None = None) -> float:
            if label == "PFSOffset" and not stopped:
                stopped.append(label)
                z.stop()
            return read(label)

        monkeypatch.setattr(stand.core, "getPosition", stop_on_the_anchor)
        description = "autofocus_offset: move_by 1.0 device units"
        with caplog.at_level(logging.WARNING), pytest.raises(MotionStoppedError) as e:
            af.offset_by_native(1.0, force=True)
        assert str(e.value) == _z_stopped_during(description)
        assert stand.core.log == [OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert stand.af.offset_native() == 7800.0
        assert _LEFT_OFF.format(description) in caplog.messages


@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_z_stop_while_a_direct_offset_move_waits_for_the_lock_cancels_it_with_nothing_sent(
    call: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    move, _ = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af = stand.af
        with caplog.at_level(logging.WARNING):
            mover = _z_stop_while_queued(stand, monkeypatch, lambda: move(af))
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == (
            "`z ZDrive`: it was stopped before it started; read the position before "
            "moving on"
        )
        assert stand.core.log == ["stop('ZDrive')", "snapImage()"]
        assert stand.af.offset_native() == 7800.0
        assert stand.core.isContinuousFocusLocked() is True
        assert not any("left off" in m for m in caplog.messages)


@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_pfs_switched_on_while_a_direct_offset_move_waits_for_the_lock_is_still_linked(
    call: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adversarial review of #144: the state is read under the microscope
    # lock, not when the call is made. PFS switched on meanwhile follows the
    # offset, so the Z stop must reach the move.
    move, description = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        af, z = stand.af, stand.z
        _offset_busy_until_stopped(stand, monkeypatch)
        with _lock_held_by_a_snap(stand, monkeypatch) as queued:
            mover = _Thread(lambda: move(af))
            assert queued.wait(JOIN_S)
            stand.core.enableContinuousFocus(True)  # at the stand, meanwhile
        sent = [ON, "snapImage()", "setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _z_stopped_during(description)
        assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False
        assert _LEFT_OFF.format(description) in caplog.messages


def test_a_refused_direct_offset_move_neither_waits_for_the_lock_nor_reads_pfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The guards run before the compound (plan of #144): behind a long lock
    # holder, an engage() waiting up to 10 min, a refusal comes at once.
    profile = _profile(
        safety={
            "max_autofocus_offset_jog": 5.0,
            "autofocus_offset_soft_limits": (7000.0, 8000.0),
        }
    )
    with _ti2(profile, core=_at_offset(7800.0)) as stand:
        af, enabled = stand.af, stand.core.isContinuousFocusEnabled
        reads: list[int] = []

        def counted() -> bool:
            reads.append(1)
            return enabled()

        monkeypatch.setattr(stand.core, "isContinuousFocusEnabled", counted)
        with _lock_held_by_a_snap(stand, monkeypatch):
            for change in (
                lambda: af.set_offset_native(9000.0),
                lambda: af.offset_by_native(6.0),
            ):
                refused = _Thread(change)
                refused.join()  # while the snap still holds the lock
                assert isinstance(refused.error, SafetyRefusedError)
        assert reads == []
        assert stand.core.log == ["snapImage()"]


@pytest.mark.parametrize("how", ["stop()", "close()"])
def test_emergency_stop_or_close_during_a_direct_offset_move_with_pfs_on(
    how: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # m2 §3.3.1's last two columns. Halted, no link is called: one switch-off,
    # and the move's own error. close() does not halt, so the Z stop it sends
    # last still reaches the link (m2 §3.11). The offset device reads busy
    # until Z's stop, so the move cannot end between close()'s two stops.
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, stop = stand.af, stand.core.stop

        def stop_arrives(label: str) -> None:
            stop(label)
            if label == "ZDrive":
                stand.core.busy_devices.discard("PFSOffset")

        monkeypatch.setattr(stand.core, "stop", stop_arrives)
        stand.core.busy_devices.add("PFSOffset")
        sent = ["setPosition('PFSOffset', 7801.0)"]
        description = "autofocus_offset: move_to 7801.0 device units"
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.set_offset_native(7801.0))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            if how == "stop()":
                stand.microscope.stop()
            else:
                stand.microscope.close()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert stand.core.isContinuousFocusEnabled() is False
        stages = [OFF, "stop('PFSOffset')", "stop('XYStage')"]
        if how == "stop()":
            assert mover.error.device == "autofocus_offset PFSOffset"
            assert stand.core.log == [*sent, *stages, "stop('ZDrive')"]
            assert not any("left off" in m for m in caplog.messages)
            stand.microscope.resume()
        else:
            assert str(mover.error) == _z_stopped_during(description)
            assert stand.core.log == [
                *sent,
                *stages,
                OFF,
                "stop('PFSOffset')",
                "stop('ZDrive')",
                "unloadAllDevices()",
            ]
            assert _LEFT_OFF.format(description) in caplog.messages


def test_z_stop_during_a_direct_offset_move_without_an_offset_device_is_the_switch_off_alone() -> (
    None
):
    # The demo's layout: the offset is the autofocus device's own, keyed by
    # its label, and cannot be stopped: the switch-off cancels its move.
    devices = [
        FakeDevice("ZDrive", DeviceType.Stage),
        FakeDevice("PFS", DeviceType.AutoFocus),
        FakeDevice("XYStage", DeviceType.XYStage),
        FakeDevice("Camera-1", DeviceType.Camera),
    ]
    core = FakeCore(
        devices, camera="Camera-1", xy_stage="XYStage", focus="ZDrive", autofocus="PFS"
    )
    with _ti2(core=core) as stand:
        assert stand.microscope.roles.get(Role.autofocus_offset) is None
        stand.engage_directly()
        af, z = stand.af, stand.z
        core.busy_devices.add("PFS")
        mover = _Thread(lambda: af.set_offset_native(-6.0))
        sent = ["setAutoFocusOffset(-6.0)"]
        _wait_for(_sent_and_waiting(stand, sent, "autofocus PFS offset"), "offset")
        z.stop()  # no stop of an offset device that cannot be stopped
        core.busy_devices.discard("PFS")
        mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _z_stopped_during(
            "autofocus_offset: move_to -6.0 device units"
        )
        assert core.log == [*sent, OFF, "stop('ZDrive')"]


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


# --- the Z stop's outcome is waited for (FM-97, #158) ------------------------


class _SlowSwitchOff:
    """The Z stop's switch-off, slow: it ends the caller's block, then lets it finish.

    FM-96: it first does what ends the caller's block (``ends_the_block``: PFS
    locks, or the offset move lands), then gives the caller up to
    ``SETTLE_S`` to finish, and only then fails or switches PFS off. A caller
    that reports before the switch-off has returned has finished by then; one
    that waits for it is still waiting, whatever the fake's speed. A
    successful one logs ``switch-off returned`` at WARNING as it returns.
    """

    def __init__(
        self, stand: Stand, ends_the_block: Callable[[], None], *, fails: bool
    ) -> None:
        self.caller: _Thread | None = None
        self._stand = stand
        self._switch = stand.core.enableContinuousFocus
        self._ends_the_block = ends_the_block
        self._fails = fails

    def __call__(self, enable: bool) -> None:
        if enable:
            self._switch(True)
            return
        if self._fails:
            self._stand.core.log.append(OFF)
        self._ends_the_block()
        assert self.caller is not None
        self.caller.finished_within(SETTLE_S)
        if self._fails:
            raise RuntimeError("PFS: stuck on")
        self._switch(False)
        logging.getLogger("tests.fake").warning("switch-off returned")


def _lock_comes(stand: Stand) -> Callable[[], None]:
    def lock() -> None:
        stand.core.autofocus_lockable = True

    return lock


def test_a_slow_failing_z_stop_switch_off_during_a_direct_engage_is_waited_for_and_not_called_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-97: the link is marked reached before the switch-off is sent; the
    # lock that comes meanwhile ended the block, and on main the caller was
    # told "left off" while the switch-off was still on its way, then failed.
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z = stand.af, stand.z
        slow = _SlowSwitchOff(stand, _lock_comes(stand), fails=True)
        with caplog.at_level(logging.WARNING):
            waiter = slow.caller = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", slow)
            with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                z.stop()
            waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _not_switched_off("autofocus: engage")
        # The switch-off moved PFS's stop generation: the wait that saw the
        # lock ended as stopped, and the error is chained from it.
        assert isinstance(waiter.error.__cause__, MotionStoppedError)
        assert waiter.error.__cause__.device == "autofocus PFS"
        assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
        assert _STUCK_ON.format("autofocus: engage") in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stand.core.isContinuousFocusEnabled() is True
        assert stand.microscope.state().moving == ()


@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_a_slow_failing_z_stop_switch_off_during_a_direct_offset_move_is_waited_for_and_not_called_left_off(
    call: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The offset move lands during the switch-off: the move had returned, so
    # no offset stop is sent, and the error is chained from the failure.
    move, description = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z = stand.af, stand.z
        stand.core.busy_devices.add("PFSOffset")
        slow = _SlowSwitchOff(
            stand, lambda: stand.core.busy_devices.discard("PFSOffset"), fails=True
        )
        sent = ["setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            mover = slow.caller = _Thread(lambda: move(af))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            monkeypatch.setattr(stand.core, "enableContinuousFocus", slow)
            with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _not_switched_off(description)
        assert isinstance(mover.error.__cause__, HardwareError)
        assert str(mover.error.__cause__) == _STUCK
        assert stand.core.log == [*sent, OFF, "stop('ZDrive')"]
        assert _STUCK_ON.format(description) in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stand.core.isContinuousFocusEnabled() is True


def test_a_slow_failing_z_stop_switch_off_during_the_relock_is_waited_for_and_not_called_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _ti2(_profile(autofocus={"lock_timeout_s": 2 * JOIN_S})) as stand:
        stand.engage_directly(lockable=False)
        z = stand.z
        slow = _SlowSwitchOff(stand, _lock_comes(stand), fails=True)
        relocking = [OFF, "setPosition('ZDrive', 12.0)", ON]
        with caplog.at_level(logging.WARNING):
            mover = slow.caller = _Thread(lambda: z.move_to_um(12.0))
            _wait_for(_sent_and_waiting(stand, relocking, "autofocus PFS"), "relock")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", slow)
            with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _not_switched_off("z: move_to 12.0 µm")
        assert stand.core.log == [*relocking, OFF, "stop('ZDrive')"]
        assert _STUCK_ON.format("z: move_to 12.0 µm") in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stand.core.isContinuousFocusEnabled() is True


def test_a_z_stop_switch_off_that_has_not_returned_in_time_says_pfs_may_still_move_z(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bounded: the caller holds the microscope lock while it waits, so a
    # switch-off that hangs must not hang the stand; unreturned is unknown,
    # so it is reported as a failure, never as "left off".
    monkeypatch.setattr(mm, "Z_STOP_OUTCOME_WAIT_S", 0.2)
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, switch = stand.af, stand.z, stand.core.enableContinuousFocus
        release = threading.Event()

        def hangs(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.autofocus_lockable = True  # the lock ends the block
            release.wait(JOIN_S)
            switch(False)

        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", hangs)
            stopper = _Thread(z.stop)
            waiter.join()
            release.set()
            stopper.join()
        failure = "its switch-off had not returned after 0.2 s"
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _not_switched_off("autofocus: engage", failure)
        assert (
            f"autofocus PFS: the Z stop during autofocus: engage could not switch it "
            f"off ({failure}); it may still move Z"
        ) in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stopper.error is None
        assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False


def test_a_slow_z_stop_switch_off_that_succeeds_is_waited_for_and_reported_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The normal path: "left off" is claimed once the switch-off has returned.
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z = stand.af, stand.z
        slow = _SlowSwitchOff(stand, _lock_comes(stand), fails=False)
        with caplog.at_level(logging.WARNING):
            waiter = slow.caller = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", slow)
            z.stop()
            waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _z_stopped_during("autofocus: engage")
        left_off = _LEFT_OFF.format("autofocus: engage")
        ours = [m for m in caplog.messages if m in (left_off, "switch-off returned")]
        assert ours == ["switch-off returned", left_off]
        assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False


@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_a_failed_z_stop_switch_off_during_a_direct_offset_move_says_pfs_may_still_move_z(
    call: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # m2 §3.3.1: the offset device's own "stopped" error would hide that PFS,
    # still on, may follow the offset and move Z.
    move, description = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, switch = stand.af, stand.z, stand.core.enableContinuousFocus

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        _offset_busy_until_stopped(stand, monkeypatch)
        sent = ["setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: move(af))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
            with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                z.stop()
            mover.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _not_switched_off(description)
        assert isinstance(mover.error.__cause__, MotionStoppedError)
        assert str(mover.error.__cause__) == (
            "`autofocus_offset PFSOffset`: the move was stopped before it completed; "
            "read the position before moving on"
        )
        assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert _STUCK_ON.format(description) in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stand.core.isContinuousFocusEnabled() is True


def test_a_failed_z_stop_switch_off_keeps_a_lost_lock_give_up_as_it_is(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # m2 §3.3.1: a give-up whose own error already says PFS may still move Z
    # goes on as it is, with its motion registered. The Z stop is made on the
    # caller's own thread, inside a core call (FM-70): its switch-off has
    # returned before the block ends, so nothing waits.
    with _ti2() as stand:
        af, z = stand.af, stand.z
        switch, locked = (
            stand.core.enableContinuousFocus,
            stand.core.isContinuousFocusLocked,
        )
        reads: list[int] = []

        def stuck_on(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")

        def lost_on_the_readback() -> bool:
            # Locked at the poll; lost from the readback on, so it searches.
            reads.append(1)
            if len(reads) == 2:
                with pytest.raises(HardwareError, match=re.escape("(PFS: stuck on)")):
                    z.stop()
            return len(reads) == 1 and locked()

        monkeypatch.setattr(stand.core, "enableContinuousFocus", stuck_on)
        monkeypatch.setattr(stand.core, "isContinuousFocusLocked", lost_on_the_readback)
        with caplog.at_level(logging.WARNING), pytest.raises(DeviceTimeoutError) as e:
            af.engage(timeout_s=2 * JOIN_S)
        assert str(e.value).startswith(
            "autofocus PFS: it locked, then lost the lock at once, and it could not "
            "be switched off (the stop failed ("
        )
        assert _STUCK_ON.format("autofocus: engage") in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stand.core.log == [ON, OFF, "stop('ZDrive')", OFF]
        assert stand.microscope.state().moving == ("autofocus PFS",)
        monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
        stand.microscope.resume()


def test_an_interrupted_z_stop_switch_off_is_reported_as_one_that_may_have_failed(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A second Ctrl-C in the switch-off: whether it went out is unknown, so
    # the caller is not told "left off" (#158, beyond the plan: named).
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, switch = stand.af, stand.z, stand.core.enableContinuousFocus

        def interrupted(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            stand.core.autofocus_lockable = True  # the lock ends the block
            raise KeyboardInterrupt

        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", interrupted)
            with pytest.raises(KeyboardInterrupt):
                z.stop()
            waiter.join()
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _not_switched_off(
            "autofocus: engage", "KeyboardInterrupt"
        )
        assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
        assert (
            "autofocus PFS: the Z stop during autofocus: engage could not switch it "
            "off (KeyboardInterrupt); it may still move Z"
        ) in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)


@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_an_interrupted_z_stop_switch_off_still_stops_the_offset_device(
    call: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-62, #166: `Executor._stop_linked` holds an interrupt until the later
    # stops are sent; this link's own stop raised it at once, and PFSOffset,
    # which PFS follows, was left moving.
    move, description = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, switch = stand.af, stand.z, stand.core.enableContinuousFocus

        def interrupted(enable: bool) -> None:
            if enable:
                switch(True)
                return
            stand.core.log.append(OFF)
            raise KeyboardInterrupt

        _offset_busy_until_stopped(stand, monkeypatch)
        sent = ["setPosition('PFSOffset', 7801.0)"]
        mover = _Thread(lambda: move(af))
        _wait_for(
            _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
        )
        monkeypatch.setattr(stand.core, "enableContinuousFocus", interrupted)
        try:
            with pytest.raises(KeyboardInterrupt):
                z.stop()
        finally:
            # Restored before `close()`, and the move released: without the
            # fix nothing stops the offset device, which stays busy.
            monkeypatch.setattr(stand.core, "enableContinuousFocus", switch)
            stand.core.busy_devices.discard("PFSOffset")
            mover.join()
        assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _not_switched_off(description, "KeyboardInterrupt")


def _hung_switch_off(stand: Stand, release: threading.Event) -> Callable[[bool], None]:
    """The Z stop's switch-off: PFS locks (the caller's block ends), then it hangs until ``release``.

    Only the first switch-off hangs; any later one (a give-up's) goes through.
    """
    switch, calls = stand.core.enableContinuousFocus, []

    def hangs(enable: bool) -> None:
        if not enable:
            calls.append(1)
            if len(calls) == 1:
                stand.core.autofocus_lockable = True
                release.wait(JOIN_S)
        switch(enable)

    return hangs


def test_a_ctrl_c_during_the_wait_for_the_z_stops_outcome_ends_it_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-83: one long Event.wait is not interrupted by a Ctrl-C on Windows;
    # short sleeps are. The interrupt goes on, the link is removed and the
    # microscope lock released (adversarial review of #158).
    sleeps: list[float] = []

    def interrupted(seconds: float) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, release = stand.af, stand.z, threading.Event()
        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
        monkeypatch.setattr(
            stand.core, "enableContinuousFocus", _hung_switch_off(stand, release)
        )
        monkeypatch.setattr(
            mm, "time", SimpleNamespace(monotonic=time.monotonic, sleep=interrupted)
        )
        stopper = _Thread(z.stop)
        waiter.join()
        assert isinstance(waiter.error, KeyboardInterrupt)
        assert sleeps == [mm.POLL_INTERVAL_S]
        assert stand.microscope._executor._links == {}
        assert stand.microscope.require(XYStage).move_to_um(1.0, 0.0) == XY(1.0, 0.0)
        release.set()
        stopper.join()
        assert stopper.error is None
        assert stand.core.isContinuousFocusEnabled() is False


def test_a_ctrl_c_that_ends_the_call_after_a_z_stop_does_not_wait_for_its_outcome(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adversarial review of #158: the interrupt was held for the whole wait,
    # under the microscope lock. It goes on at once, with what is known.
    monkeypatch.setattr(mm, "Z_STOP_OUTCOME_WAIT_S", 2 * JOIN_S)
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, locked = stand.af, stand.z, stand.core.isContinuousFocusLocked
        release, reached = threading.Event(), threading.Event()
        hangs = _hung_switch_off(stand, release)

        def noted(enable: bool) -> None:
            if not enable:
                reached.set()
            hangs(enable)

        def interrupted_once_stopped() -> bool:
            if reached.is_set():
                raise KeyboardInterrupt  # raised from the stub, on the caller
            return locked()

        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", noted)
            monkeypatch.setattr(
                stand.core, "isContinuousFocusLocked", interrupted_once_stopped
            )
            stopper = _Thread(z.stop)
            # Ended while the Z stop's switch-off still hangs.
            assert waiter.finished_within(JOIN_S)
            release.set()
            stopper.join()
        assert isinstance(waiter.error, KeyboardInterrupt)
        assert (
            "autofocus PFS: the Z stop during autofocus: engage could not switch it "
            "off (its switch-off had not returned yet); it may still move Z"
        ) in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stopper.error is None
        # The give-up's switch-off, then the Z stop's once released, then Z.
        assert stand.core.log == [ON, OFF, OFF, "stop('ZDrive')"]


_NOT_RETURNED_YET = (
    "autofocus PFS: the Z stop during autofocus: engage could not switch it off "
    "(its switch-off had not returned yet); it may still move Z"
)


def test_a_halt_after_a_z_stop_does_not_wait_for_the_z_stops_outcome(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #166: the Z stop moves PFS's generation before it sends its switch-off,
    # so the lock wait ends with a plain `MotionStoppedError` and the
    # emergency stop comes during the outcome wait. The halt returned at
    # once, but the caller waited the whole of `Z_STOP_OUTCOME_WAIT_S`, under
    # the microscope lock, for a switch-off that the halt does itself.
    monkeypatch.setattr(mm, "Z_STOP_OUTCOME_WAIT_S", 2 * JOIN_S)
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z = stand.af, stand.z
        release, reached = threading.Event(), threading.Event()
        hangs = _hung_switch_off(stand, release)

        def noted(enable: bool) -> None:
            if not enable:
                reached.set()
            hangs(enable)

        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            monkeypatch.setattr(stand.core, "enableContinuousFocus", noted)
            stopper = _Thread(z.stop)
            _wait_for(reached.is_set, "the Z stop's switch-off")
            stand.microscope.stop()
            # Against a wait of 10 s, ended while the Z stop's switch-off hangs.
            ended = waiter.finished_within(1.0)
            release.set()
            stopper.join()
            waiter.join()
        stand.microscope.resume()
        assert ended, "the call waited for the Z stop's outcome after the halt"
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _not_switched_off(
            "autofocus: engage", "its switch-off had not returned yet"
        )
        assert _NOT_RETURNED_YET in caplog.messages
        assert not any("left off" in m for m in caplog.messages)


def test_a_halt_during_the_wait_for_the_z_stops_outcome_ends_it_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same, without a timing margin: the halt comes in the first poll's
    # sleep, so the next poll must read it. The Z stop's generation move
    # ended the lock wait, which the error's cause shows (measured, #166).
    monkeypatch.setattr(mm, "Z_STOP_OUTCOME_WAIT_S", 2 * JOIN_S)
    sleeps: list[float] = []
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z = stand.af, stand.z
        release, reached = threading.Event(), threading.Event()
        hangs = _hung_switch_off(stand, release)

        def noted(enable: bool) -> None:
            if not enable:
                reached.set()
            hangs(enable)

        def halted_in_the_first_poll(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) == 1:
                # The halt's own switch-off must not be the one that hangs.
                _wait_for(reached.is_set, "the Z stop's switch-off")
                stand.microscope.stop()
            time.sleep(seconds)

        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
        monkeypatch.setattr(stand.core, "enableContinuousFocus", noted)
        monkeypatch.setattr(
            mm,
            "time",
            SimpleNamespace(monotonic=time.monotonic, sleep=halted_in_the_first_poll),
        )
        stopper = _Thread(z.stop)
        ended = waiter.finished_within(1.0)
        release.set()
        stopper.join()
        waiter.join()
        stand.microscope.resume()
        assert ended, "the call kept polling after the halt"
        assert sleeps == [mm.POLL_INTERVAL_S]
        assert isinstance(waiter.error, MotionStoppedError)
        assert str(waiter.error) == _not_switched_off(
            "autofocus: engage", "its switch-off had not returned yet"
        )
        assert isinstance(waiter.error.__cause__, MotionStoppedError)
        assert waiter.error.__cause__.device == "autofocus PFS"


def test_a_ctrl_c_during_the_wait_for_the_z_stops_outcome_logs_that_pfs_may_still_move_z(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #166: the interrupt came out of the wait before any WARNING was written,
    # while the switch-off had still not returned. It goes on at once, and is
    # decided as an interrupt that ended the block (stubbed sleep: FM-83).
    sleeps: list[float] = []

    def interrupted(seconds: float) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z, release = stand.af, stand.z, threading.Event()
        waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
        _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
        monkeypatch.setattr(
            stand.core, "enableContinuousFocus", _hung_switch_off(stand, release)
        )
        monkeypatch.setattr(
            mm, "time", SimpleNamespace(monotonic=time.monotonic, sleep=interrupted)
        )
        with caplog.at_level(logging.WARNING):
            stopper = _Thread(z.stop)
            waiter.join()
            release.set()
            stopper.join()
        assert isinstance(waiter.error, KeyboardInterrupt)
        assert sleeps == [mm.POLL_INTERVAL_S]
        assert _NOT_RETURNED_YET in caplog.messages
        assert not any("left off" in m for m in caplog.messages)
        assert stand.microscope._executor._links == {}


def test_a_slow_offset_stop_after_a_z_stop_switch_off_that_worked_says_left_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adversarial review of #158: the caller reports the switch-off, and PFS
    # is off; the offset stop's own outcome goes to the stopping caller.
    monkeypatch.setattr(mm, "Z_STOP_OUTCOME_WAIT_S", 0.2)
    description = "autofocus_offset: move_to 7801.0 device units"
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, z, stop, release = stand.af, stand.z, stand.core.stop, threading.Event()

        def slow_offset_stop(label: str) -> None:
            stop(label)
            if label == "PFSOffset":
                stand.core.busy_devices.discard(label)  # the move ends, stopped
                release.wait(JOIN_S)

        monkeypatch.setattr(stand.core, "stop", slow_offset_stop)
        stand.core.busy_devices.add("PFSOffset")
        sent = ["setPosition('PFSOffset', 7801.0)"]
        with caplog.at_level(logging.WARNING):
            mover = _Thread(lambda: af.set_offset_native(7801.0))
            _wait_for(
                _sent_and_waiting(stand, sent, "autofocus_offset PFSOffset"), "offset"
            )
            stopper = _Thread(z.stop)
            assert mover.finished_within(JOIN_S)
            release.set()
            stopper.join()
        assert isinstance(mover.error, MotionStoppedError)
        assert str(mover.error) == _z_stopped_during(description)
        assert _LEFT_OFF.format(description) in caplog.messages
        assert stopper.error is None
        assert stand.core.log == [*sent, OFF, "stop('PFSOffset')", "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False


# --- the direct offset moves' reads and guards (#158) ------------------------


@pytest.mark.parametrize("call", sorted(_OFFSET_CALLS))
def test_a_direct_offset_move_is_refused_when_the_slot_names_another_device(
    call: str,
) -> None:
    # FM-87: a slot that names another device is a configuration error, not
    # a state that cannot be read; on main it was counted as "on" and moved.
    move, _ = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af = stand.af
        stand.core.setAutoFocusDevice("")  # behind the layer
        stand.core.log.clear()
        with pytest.raises(HardwareError) as refused:
            move(af)
        assert str(refused.value) == (
            "autofocus 'PFS' is not the device the core's AutoFocus slot names "
            "(''), and the continuous-focus calls act on the slot's device; "
            "something changed the slot after the stand was opened. Open it "
            "again, or set Core-AutoFocus in the configuration and the autofocus "
            "role under [roles.assign] to the same device."
        )
        assert stand.core.log == []
        assert stand.core.getPosition("PFSOffset") == 7800.0
        stand.core.setAutoFocusDevice("PFS")


def test_a_direct_offset_move_with_an_overridden_focus_drive_reads_no_pfs_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No focus drive of this module: no Z stop can be linked, so the state
    # that would decide the link is not read.
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.microscope.override(ZStage, _StubZ())
        stand.engage_directly()
        af, enabled = stand.af, stand.core.isContinuousFocusEnabled
        reads: list[int] = []

        def counted() -> bool:
            reads.append(1)
            return enabled()

        monkeypatch.setattr(stand.core, "isContinuousFocusEnabled", counted)
        assert af.set_offset_native(7801.0) == 7801.0
        assert reads == []
        assert stand.core.log == ["setPosition('PFSOffset', 7801.0)"]


@pytest.mark.parametrize(
    ("call", "expected"),
    [("absolute", {"target": 1, "jog": 0}), ("relative", {"target": 1, "jog": 1})],
)
def test_a_direct_offset_move_runs_each_guard_once(
    call: str, expected: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    move, _ = _OFFSET_CALLS[call]
    with _ti2(core=_at_offset(7800.0)) as stand:
        stand.engage_directly()
        af, safety = stand.af, stand.microscope._safety
        target, jog = safety.check_offset_target_native, safety.check_offset_jog_native
        counts = {"target": 0, "jog": 0}

        def counted_target(value: float) -> None:
            counts["target"] += 1
            target(value)

        def counted_jog(delta: float, *, force: bool = False) -> None:
            counts["jog"] += 1
            jog(delta, force=force)

        monkeypatch.setattr(safety, "check_offset_target_native", counted_target)
        monkeypatch.setattr(safety, "check_offset_jog_native", counted_jog)
        assert move(af) == 7801.0
        assert counts == expected


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


def test_a_failed_in_range_read_logs_no_exception_object(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-92, #166: the DEBUG line passed `exc_info=True`, and the record kept
    # the error and its traceback on the stop and engage paths.
    def unreadable() -> bool | None:
        raise HardwareError("PFS: no answer")

    with _ti2() as stand:
        af = stand.microscope.require(Autofocus)
        assert isinstance(af, mm.MMAutofocus)
        monkeypatch.setattr(af, "_read_in_range", unreadable)
        with caplog.at_level(logging.DEBUG, logger="smc"):
            assert af._in_range_for_message() is None
        assert [m for m in caplog.messages if "in-range read failed" in m] == [
            "autofocus PFS: in-range read failed (PFS: no answer)"
        ]
        assert _logged_exception_objects(caplog) == []


def test_a_failed_z_read_for_an_error_message_logs_no_exception_object(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The same rule, at the other place that read a value for an error (#166).
    with _ti2() as stand:
        af = stand.microscope.require(Autofocus)
        assert isinstance(af, mm.MMAutofocus)
        stand.core.failing["ZDrive"] = RuntimeError("ZDrive: no answer")
        with caplog.at_level(logging.DEBUG, logger="smc"):
            assert af._z_or_none() is None
        stand.core.failing.clear()
        (read,) = (m for m in caplog.messages if "Z read failed" in m)
        assert read.startswith("autofocus PFS: Z read failed (")
        assert "ZDrive: no answer" in read
        assert _logged_exception_objects(caplog) == []


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


def test_close_during_the_relock_keeps_the_link(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Decided in m2 §3.11 (#144): close() does not halt, so the Z stop it
    # sends still reaches the relock's link, which switches PFS off once more
    # before Z's stop. Harmless, and the mover is told it was left off.
    core = _ti2_core()
    profile = _profile(autofocus={"lock_timeout_s": 2 * JOIN_S})
    microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    stand = Stand(core, microscope)
    stand.engage_directly(lockable=False)
    z, enabled = stand.z, core.isContinuousFocusEnabled

    def reports_off_late() -> bool:
        # The link is reached only while the relock still waits when Z's
        # stop goes out. Here PFS reads on until then; on a stand, a slow
        # stage stop can let the relock end first, with PFS off all the same
        # (adversarial review of #144).
        return "stop('ZDrive')" not in core.log or enabled()

    relocking = [OFF, "setPosition('ZDrive', 12.0)", ON]
    with caplog.at_level(logging.WARNING):
        mover = _Thread(lambda: z.move_to_um(12.0))
        _wait_for(_sent_and_waiting(stand, relocking, "autofocus PFS"), "relock")
        monkeypatch.setattr(core, "isContinuousFocusEnabled", reports_off_late)
        microscope.close()
        mover.join()
    assert isinstance(mover.error, MotionStoppedError)
    assert str(mover.error) == _z_stopped_during("z: move_to 12.0 µm")
    assert core.log == [
        *relocking,
        OFF,
        "stop('PFSOffset')",
        "stop('XYStage')",
        OFF,
        "stop('ZDrive')",
        "unloadAllDevices()",
    ]
    assert _LEFT_OFF.format("z: move_to 12.0 µm") in caplog.messages


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


def _logged_exception_objects(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The records of ``smc`` that keep an exception: ``exc_info``, or one among the arguments.

    A record that keeps an error keeps its traceback, whose frames hold the
    stopping thread's locals and, through them, the core (FM-92).
    """
    kept = []
    for record in caplog.records:
        if not record.name.startswith("smc"):
            continue
        args = record.args if isinstance(record.args, tuple) else ()
        if record.exc_info or any(isinstance(a, BaseException) for a in args):
            kept.append(record.getMessage())
    return kept


def test_a_failed_state_read_logs_no_exception_object(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # #166: the DEBUG line of a failed read kept the error (`exc_info`).
    with _ti2() as stand:
        stand.core.failing["PFS"] = RuntimeError("PFS: no answer")
        with caplog.at_level(logging.DEBUG, logger="smc"):
            state = stand.microscope.state()
        stand.core.failing.clear()
    assert len(state.errors) == 2
    read_failed = [m for m in caplog.messages if " read failed (" in m]
    # The same text as `errors`, which is unchanged.
    assert read_failed == [
        f"state: {name} read failed ({detail})"
        for name, detail in (e.split(": ", 1) for e in state.errors)
    ]
    assert _logged_exception_objects(caplog) == []


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


# --- the facade's lifetime (#142) -------------------------------------------

_DEAD_OWNER = (
    "{asked_by} belongs to a Microscope that no longer exists, and looks up its "
    "sibling capabilities (the Z interlock, the focus drive) through it; keep the "
    "Microscope alive for as long as its capabilities are in use"
)


def _use_the_stand(core: FakeCore, microscope: Microscope) -> None:
    """What a session that used the whole stand has built and done, ``state()`` included.

    Every capability the facade knows is built, so that a new one is under the
    test, and then each thing the stand does runs once: a cycle can be made by
    a call as well as by building (the Z stage's compound linked a stop to the
    autofocus: a cycle that only a Z move with PFS on showed).
    """
    for capability in CAPABILITY_ROLES:
        # The Ti2-E has no shutter role; the others it has.
        with suppress(CapabilityMissingError):
            microscope.require(capability)
    microscope.require(XYStage).move_to_um(10.0, 5.0)
    microscope.require(Camera).snap()
    autofocus, z = microscope.require(Autofocus), microscope.require(ZStage)
    core.enableContinuousFocus(True)
    z.move_to_um(3.0)  # suspend, move, relock
    z.move_by_um(1.0)
    autofocus.focus_by_um(1.0)  # locked: moves the offset
    autofocus.disengage()
    autofocus.engage()  # direct: linked to the Z stop, locks at once
    autofocus.set_offset_native(7798.0)  # PFS on: linked too
    autofocus.offset_by_native(1.0, force=True)
    autofocus.disengage()
    z.move_to_um(0.0)  # PFS off: a plain move
    # m2 §8: the turret looks the autofocus up through the facade, Channels
    # the light source; each does one thing.
    autofocus.engage()
    microscope.require(ObjectiveTurret).select("10x", confirm=True)  # PFS off first
    channels = microscope.require(Channels)
    channels.select("BF")
    channels.capture("smc")
    microscope.require(LightSource).set_intensity_native(100.0)
    microscope.state()


def _with_lamp_and_channel(core: FakeCore) -> FakeCore:
    """``core`` with the Ti2-E's ``DiaLamp`` and a ``Channel`` group of one preset."""
    core.config_groups = {"Channel": {"BF": [("DiaLamp", "Intensity", "210")]}}
    return core


_DIA_LAMP = (
    FakeDevice(
        "DiaLamp", DeviceType.Shutter, properties={"Intensity": FakeProperty("501")}
    ),
)


@pytest.mark.parametrize("closed", [True, False])
def test_a_dropped_facade_frees_its_core_without_the_cyclic_collector(
    no_cyclic_gc: None, closed: bool
) -> None:
    # #142: a capability that held the facade (the Z stage's interlock, the
    # autofocus's focus drive), itself (the switch-off's SafeCall) or the
    # callback of its own stop link (a Z move with PFS on) kept the core alive
    # until the cyclic collector, which at the exit of `smc doctor` runs during
    # interpreter finalization: the child aborted there.
    core = _with_lamp_and_channel(_at_offset(7800.0, extra=_DIA_LAMP))
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _factor({"4x": -3.0}))
    _use_the_stand(core, microscope)
    if closed:
        microscope.close()
    alive = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}
    del microscope, core
    assert {name: ref() is not None for name, ref in alive.items()} == {
        "facade": False,
        "core": False,
    }


def test_a_z_stage_outliving_its_microscope_refuses_the_move_and_sends_nothing(
    no_cyclic_gc: None,
) -> None:
    # The weak link must not read as "no autofocus": that would move Z with
    # PFS locked and switch it off for good (FM-12).
    core = _ti2_core()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    z = microscope.require(ZStage)
    microscope.require(Autofocus)
    core.enableContinuousFocus(True)
    core.log.clear()
    del microscope
    with pytest.raises(HardwareError) as refused:
        z.move_to_um(5.0)
    assert str(refused.value) == _DEAD_OWNER.format(asked_by="ZStage")
    assert core.log == []
    assert core.isContinuousFocusEnabled() is True


def test_an_autofocus_outliving_its_microscope_refuses_focus_by_um_and_sends_nothing(
    no_cyclic_gc: None,
) -> None:
    # No focus drive found would skip the Z soft-limit check on a locked
    # focus_by_um; a facade that is gone is not "no focus drive".
    core = _at_offset(7800.0)
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _factor({"4x": -3.0}))
    autofocus = microscope.require(Autofocus)
    core.enableContinuousFocus(True)
    core.log.clear()
    del microscope
    with pytest.raises(HardwareError) as refused:
        autofocus.focus_by_um(2.0)
    assert str(refused.value) == _DEAD_OWNER.format(asked_by="Autofocus")
    assert core.log == []


_DIRECT_CALLS: dict[str, Callable[[Autofocus], object]] = {
    "engage": lambda af: af.engage(),
    "absolute": lambda af: af.set_offset_native(7801.0),
    "relative": lambda af: af.offset_by_native(1.0, force=True),
}


@pytest.mark.parametrize("pfs", ["on", "off"])
@pytest.mark.parametrize("call", sorted(_DIRECT_CALLS))
def test_an_autofocus_outliving_its_microscope_refuses_engage_and_the_offset_moves_and_sends_nothing(
    no_cyclic_gc: None, call: str, pfs: str
) -> None:
    # The focus drive is looked up first (#158): on main, the offset moves
    # with PFS off read no focus drive and moved the offset.
    core = _at_offset(7800.0)
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    autofocus = microscope.require(Autofocus)
    core.enableContinuousFocus(pfs == "on")
    core.log.clear()
    del microscope
    with pytest.raises(HardwareError) as refused:
        _DIRECT_CALLS[call](autofocus)
    assert str(refused.value) == _DEAD_OWNER.format(asked_by="Autofocus")
    assert core.log == []
    assert core.getPosition("PFSOffset") == 7800.0
    assert core.isContinuousFocusEnabled() is (pfs == "on")


@pytest.fixture
def no_kept_log_records() -> Iterator[None]:
    """Logging off: pytest keeps every ``LogRecord``, and a failed stop's holds the error.

    That error's traceback holds the fake: a retention by the test's handler,
    not a cycle of the code (FM-98 test hazards).
    """
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _fails_once(
    core: FakeCore, switch: Callable[[bool], None]
) -> Callable[[bool], None]:
    """``switch``, whose first switch-off raises a fresh error; any later one goes through.

    Fresh, not ``FakeCore.failing``, which keeps the instance it raised; only
    the Z stop's switch-off fails, so a give-up after it leaves nothing
    registered (FM-98 test hazards).
    """
    failed: list[bool] = []

    def switch_once(enable: bool) -> None:
        if not enable and not failed:
            failed.append(True)
            core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")
        switch(enable)

    return switch_once


def _described(error: BaseException | None) -> tuple[str, str, str]:
    """``error``'s type name, message and cause's type name: what a freeing test keeps of it."""
    if error is None:
        return "", "", ""
    return type(error).__name__, str(error), type(error.__cause__).__name__


def _z_stop_a_compound(
    core: FakeCore, microscope: Microscope, compound: str, switch_off: str
) -> tuple[str, str, str]:
    """``compound`` with PFS on, ended by a Z stop; what it raised (``_described``).

    Its own frame, so that the thread and the capabilities are gone once it
    returns, and the mover's error is dropped: ``_Thread`` keeps it, and its
    traceback holds ``_Thread._run``'s frame, whose ``self`` holds the error
    (FM-98 test hazards). The caller holds nothing but the two it drops.
    """
    autofocus, z = microscope.require(Autofocus), microscope.require(ZStage)
    switch = core.enableContinuousFocus
    if compound == "locked":
        stopped: list[bool] = []

        def stop_once_locked(enable: bool) -> None:
            switch(enable)
            if enable and not stopped:
                stopped.append(True)
                if switch_off == "fails once":
                    core.enableContinuousFocus = _fails_once(core, switch)  # type: ignore[method-assign]
                with suppress(HardwareError):  # a failed switch-off reaches the stopper
                    z.stop()

        core.enableContinuousFocus(True)
        core.log.clear()
        core.enableContinuousFocus = stop_once_locked  # type: ignore[method-assign]
        try:
            z.move_to_um(12.0)
        except BaseException as exc:
            return _described(exc)
        finally:
            # Removed, not restored: the bound method assigned back is a cycle.
            del core.enableContinuousFocus
        return _described(None)
    runs: dict[str, Callable[[], object]] = {
        "engage": lambda: autofocus.engage(timeout_s=2 * JOIN_S),
        "absolute": lambda: autofocus.set_offset_native(7801.0),
        "relative": lambda: autofocus.offset_by_native(1.0, force=True),
        "focus_by": lambda: autofocus.focus_by_um(2.0),
        "relock": lambda: z.move_to_um(12.0),
    }
    if compound in ("engage", "relock"):
        core.autofocus_lockable = False
        waits = "autofocus PFS"
        if compound == "relock":
            core.enableContinuousFocus(True)
            sent = [OFF, "setPosition('ZDrive', 12.0)", ON]
        else:
            sent = [ON]
    else:
        core.enableContinuousFocus(True)
        core.busy_devices.add("PFSOffset")
        waits = "autofocus_offset PFSOffset"
        target = 7794.0 if compound == "focus_by" else 7801.0
        sent = [f"setPosition('PFSOffset', {target})"]
    core.log.clear()
    mover = _Thread(runs[compound])
    _wait_for(lambda: core.log == sent and microscope.state().moving == (waits,), waits)
    if switch_off == "fails once":
        core.enableContinuousFocus = _fails_once(core, switch)  # type: ignore[method-assign]
    with suppress(HardwareError):  # a failed switch-off reaches the stopper
        z.stop()
    core.busy_devices.discard("PFSOffset")
    # A switch-off that failed left PFS on, searching: it locks, which ends a
    # lock wait at once instead of at its deadline.
    core.autofocus_lockable = True
    mover.join()
    if switch_off == "fails once":
        # Removed, not restored: the bound method assigned back is a cycle.
        del core.enableContinuousFocus
    raised = _described(mover.error)
    mover.error = None
    return raised


_Z_STOPPED_COMPOUNDS = (
    "engage",
    "absolute",
    "relative",
    "focus_by",
    "relock",
    "locked",
)


@pytest.mark.parametrize("switch_off", ["succeeds", "fails once"])
@pytest.mark.parametrize("compound", _Z_STOPPED_COMPOUNDS)
def test_a_z_stopped_autofocus_compound_frees_its_core_without_the_cyclic_collector(
    no_cyclic_gc: None, no_kept_log_records: None, compound: str, switch_off: str
) -> None:
    # FM-92, #163: the stop path's exceptions were kept where their own
    # tracebacks reached them (the `@contextmanager`'s conversion on 3.11+,
    # the generator's locals, `_ZStop.switch_off_failure`, `_on_z_stop`'s
    # `failure`, `_engage`'s list, `_give_up`'s `error`), so the core stayed
    # alive. On 3.10 some cases pass without the fix: the conversion's edge
    # exists from 3.11 on.
    core = _at_offset(7800.0)
    microscope = Microscope.from_core(
        cast("CMMCorePlus", core), _factor({"4x": -3.0}, lock_timeout_s=2 * JOIN_S)
    )
    kind, message, cause = _z_stop_a_compound(core, microscope, compound, switch_off)
    # The row of the decision table the case meant to take: "left off" only
    # when the switch-off went through, chained from what ended the block.
    assert (kind, cause) == ("MotionStoppedError", "MotionStoppedError")
    if switch_off == "succeeds":
        assert "so autofocus PFS was left off" in message, message
    else:
        assert f"could not be switched off ({_STUCK})" in message, message
    # A motion left registered keeps the core by design (FM-92): it must not
    # pass for a cycle.
    assert microscope.state().moving == ()
    alive = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}
    del microscope, core
    assert {name: ref() is not None for name, ref in alive.items()} == {
        "facade": False,
        "core": False,
    }


def _z_stop_once_the_move_landed(
    core: FakeCore, microscope: Microscope, switch_off: str
) -> tuple[str, str, str]:
    """A Z stop whose switch-off returns, or fails, once the offset move has landed.

    The block ends normally while the switch-off is on its way (FM-97), so the
    call takes the rows of a block that a Z stop reached without ending it.
    """
    autofocus, z = microscope.require(Autofocus), microscope.require(ZStage)
    switch, movers = core.enableContinuousFocus, list[_Thread]()

    def lands_then_switches(enable: bool) -> None:
        if enable:
            switch(True)
            return
        core.busy_devices.discard("PFSOffset")  # the move lands: the block ends
        movers[0].finished_within(SETTLE_S)  # FM-96: it waits for this outcome
        if switch_off == "fails":
            core.log.append(OFF)
            raise RuntimeError("PFS: stuck on")
        switch(False)

    core.enableContinuousFocus(True)
    core.busy_devices.add("PFSOffset")
    core.log.clear()
    movers.append(_Thread(lambda: autofocus.set_offset_native(7801.0)))
    _wait_for(
        lambda: (
            core.log == ["setPosition('PFSOffset', 7801.0)"]
            and microscope.state().moving == ("autofocus_offset PFSOffset",)
        ),
        "the offset move",
    )
    core.enableContinuousFocus = lands_then_switches  # type: ignore[method-assign]
    with suppress(HardwareError):  # a failed switch-off reaches the stopper
        z.stop()
    mover = movers.pop()
    mover.join()
    del core.enableContinuousFocus
    raised = _described(mover.error)
    mover.error = None
    return raised


@pytest.mark.parametrize("switch_off", ["succeeds", "fails"])
def test_a_z_stop_that_reached_a_compound_which_ended_meanwhile_frees_its_core(
    no_cyclic_gc: None, no_kept_log_records: None, switch_off: str
) -> None:
    # The rows of the decision table for a block that ended normally: "left
    # off" raised plain, or the failure raised from the switch-off's error
    # (adversarial review of #163: no freeing test took them).
    core = _at_offset(7800.0)
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    description = "autofocus_offset: move_to 7801.0 device units"
    expected = (
        ("MotionStoppedError", _z_stopped_during(description), "NoneType")
        if switch_off == "succeeds"
        else ("MotionStoppedError", _not_switched_off(description), "HardwareError")
    )
    assert _z_stop_once_the_move_landed(core, microscope, switch_off) == expected
    assert microscope.state().moving == ()
    alive = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}
    del microscope, core
    assert {name: ref() is not None for name, ref in alive.items()} == {
        "facade": False,
        "core": False,
    }


class _LinkExitError(RuntimeError):
    """What the link's own exit raised, in place of what ended the block."""


@pytest.mark.parametrize("raised", [_LinkExitError, KeyboardInterrupt])
def test_what_the_z_stop_links_own_exit_raises_still_goes_through_the_decision_table(
    raised: type[BaseException],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An interrupt in `linked_stop`'s `finally` ends the block in place of
    # what did, as the generator's outer `try` caught it (#163, a risk the
    # plan named): a Z stop reached the link, so "left off" is still said.
    with _ti2() as stand:
        stand.core.autofocus_lockable = False
        af, z = stand.af, stand.z
        executor = stand.microscope._executor
        real = executor.linked_stop

        @contextmanager
        def exit_raises(device: str, stop: Callable[[], None]) -> Iterator[Any]:
            try:
                with real(device, stop) as link:
                    yield link
            finally:
                raise raised("link exit")

        monkeypatch.setattr(executor, "linked_stop", exit_raises)
        with caplog.at_level(logging.WARNING):
            waiter = _Thread(lambda: af.engage(timeout_s=2 * JOIN_S))
            _wait_for(_sent_and_waiting(stand, [ON], "autofocus PFS"), "the lock wait")
            z.stop()
            waiter.join()
        monkeypatch.undo()
        if raised is KeyboardInterrupt:
            # A Ctrl-C stays a Ctrl-C, and the WARNING names it.
            assert isinstance(waiter.error, KeyboardInterrupt)
            assert (
                "autofocus PFS: left off, because autofocus: engage did not complete "
                "(KeyboardInterrupt)"
            ) in caplog.messages
        else:
            assert isinstance(waiter.error, MotionStoppedError)
            assert str(waiter.error) == _z_stopped_during("autofocus: engage")
            assert isinstance(waiter.error.__cause__, _LinkExitError)
            assert _LEFT_OFF.format("autofocus: engage") in caplog.messages
        assert stand.core.log == [ON, OFF, "stop('ZDrive')"]
        assert stand.core.isContinuousFocusEnabled() is False


def test_a_relock_refused_on_entry_leaves_no_z_stop_link_while_its_error_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `__exit__` does not run when `__enter__` raises: the refusal leaves the
    # link itself. Left to the generator's finalization, it stayed live for as
    # long as the error was held, and a Z stop then switched PFS off for a
    # block that had ended (adversarial review of #163).
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
        with pytest.raises(MotionStoppedError) as refused:
            z.move_to_um(12.0)
        assert refused.value.__traceback__ is not None  # held, frames and all
        assert stand.microscope._executor._links == {}


def _z_stop_before_the_relock(core: FakeCore, microscope: Microscope) -> str:
    """A Z stop in the move's readback: the relock's block refuses on entry; what it raised."""
    z = microscope.require(ZStage)
    read, stopped = core.getPosition, list[bool]()

    def stop_on_arrival(label: str | None = None) -> float:
        if label == "ZDrive" and not stopped:
            stopped.append(True)
            z.stop()
        return read(label)

    core.enableContinuousFocus(True)
    core.getPosition = stop_on_arrival  # type: ignore[method-assign]
    try:
        z.move_to_um(12.0)
    except BaseException as exc:
        return type(exc).__name__
    finally:
        del core.getPosition
    return ""


def test_a_relock_refused_by_an_earlier_z_stop_frees_its_core_without_the_cyclic_collector(
    no_cyclic_gc: None, no_kept_log_records: None
) -> None:
    # The refusal on entry is raised through `__enter__`'s frame, which held
    # it in a local (#163, beyond the plan: the one path the compound test
    # does not take).
    core = _ti2_core()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    assert _z_stop_before_the_relock(core, microscope) == "MotionStoppedError"
    assert core.log[-2:] == ["setPosition('ZDrive', 12.0)", "stop('ZDrive')"]
    alive = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}
    del microscope, core
    assert {name: ref() is not None for name, ref in alive.items()} == {
        "facade": False,
        "core": False,
    }


def _late_failing_switch_off(
    core: FakeCore, microscope: Microscope, release: threading.Event
) -> tuple[str, str]:
    """A Z stop whose switch-off fails once the engage stopped waiting for it; both errors' text.

    Its own frame, for the same reason as ``_z_stop_a_compound``.
    """
    autofocus, z = microscope.require(Autofocus), microscope.require(ZStage)
    switch = core.enableContinuousFocus
    core.autofocus_lockable = False

    def hangs_then_fails(enable: bool) -> None:
        if enable:
            switch(True)
            return
        core.autofocus_lockable = True  # the lock ends the block
        release.wait(JOIN_S)
        raise RuntimeError("PFS: stuck on")

    core.log.clear()
    waiter = _Thread(lambda: autofocus.engage(timeout_s=2 * JOIN_S))
    _wait_for(
        lambda: core.log == [ON] and microscope.state().moving == ("autofocus PFS",),
        "the lock wait",
    )
    core.enableContinuousFocus = hangs_then_fails  # type: ignore[method-assign]
    stopper = _Thread(z.stop)
    waiter.join()  # FM-96: released only once the waiter has stopped waiting
    release.set()
    stopper.join()
    del core.enableContinuousFocus
    outcome = (str(waiter.error), str(stopper.error))
    waiter.error = stopper.error = None
    return outcome


def test_a_z_stop_switch_off_that_fails_after_the_compound_stopped_waiting_frees_the_core(
    no_cyclic_gc: None,
    no_kept_log_records: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The compound reads `switch_off_failure` no more once it has given up
    # waiting; the stopping thread, which sets it later, clears it (#163).
    monkeypatch.setattr(mm, "Z_STOP_OUTCOME_WAIT_S", 0.2)
    core = _ti2_core()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    waiter, stopper = _late_failing_switch_off(core, microscope, threading.Event())
    assert waiter == _not_switched_off(
        "autofocus: engage", "its switch-off had not returned after 0.2 s"
    )
    assert stopper == _STUCK
    alive = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}
    del microscope, core
    assert {name: ref() is not None for name, ref in alive.items()} == {
        "facade": False,
        "core": False,
    }


def _z_stop_interrupted_in_its_switch_off(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str, str]:
    """A direct offset move whose Z stop's switch-off is interrupted; what the move raised.

    Its own frame, for the same reason as ``_z_stop_a_compound``.
    """
    autofocus, z = microscope.require(Autofocus), microscope.require(ZStage)
    switch = core.enableContinuousFocus
    core.enableContinuousFocus(True)
    core.busy_devices.add("PFSOffset")
    core.log.clear()
    mover = _Thread(lambda: autofocus.set_offset_native(7801.0))
    _wait_for(
        lambda: (
            core.log == ["setPosition('PFSOffset', 7801.0)"]
            and microscope.state().moving == ("autofocus_offset PFSOffset",)
        ),
        "the offset move",
    )

    def interrupted(enable: bool) -> None:
        if enable:
            switch(True)
            return
        core.log.append(OFF)
        raise KeyboardInterrupt

    core.enableContinuousFocus = interrupted  # type: ignore[method-assign]
    with suppress(KeyboardInterrupt):
        z.stop()
    core.busy_devices.discard("PFSOffset")  # the fake's stop does not end the move
    mover.join()
    # Removed, not restored: the bound method assigned back is a cycle.
    del core.enableContinuousFocus
    raised = _described(mover.error)
    mover.error = None
    return raised


def _z_stop_whose_outcome_wait_is_cut_short(
    core: FakeCore,
    microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> tuple[str, str, str]:
    """An engage ended by a Z stop whose switch-off hangs, then a halt or a Ctrl-C in the wait.

    Its own frame, as ``_z_stop_a_compound``. The stub that ``mm.time`` is
    given holds the facade, so it is put back here, not at the test's end.
    """
    autofocus, z = microscope.require(Autofocus), microscope.require(ZStage)
    release, reached = threading.Event(), threading.Event()
    hangs = _hung_switch_off(Stand(core, microscope), release)

    def noted(enable: bool) -> None:
        if not enable:
            reached.set()
        hangs(enable)

    def cut_in_the_first_poll(seconds: float) -> None:
        # The halt's own switch-off must not be the one that hangs.
        _wait_for(reached.is_set, "the Z stop's switch-off")
        if case == "ctrl-c in the wait":
            raise KeyboardInterrupt
        microscope.stop()
        time.sleep(seconds)

    core.autofocus_lockable = False
    core.log.clear()
    waiter = _Thread(lambda: autofocus.engage(timeout_s=2 * JOIN_S))
    _wait_for(
        lambda: core.log == [ON] and microscope.state().moving == ("autofocus PFS",),
        "the lock wait",
    )
    core.enableContinuousFocus = noted  # type: ignore[method-assign]
    with monkeypatch.context() as patched:
        patched.setattr(
            mm,
            "time",
            SimpleNamespace(monotonic=time.monotonic, sleep=cut_in_the_first_poll),
        )
        stopper = _Thread(z.stop)
        waiter.join()
        release.set()
        stopper.join()
    # Removed, not restored: the bound method assigned back is a cycle.
    del core.enableContinuousFocus
    if case == "halt in the wait":
        microscope.resume()
    raised = _described(waiter.error)
    waiter.error = stopper.error = None
    return raised


@pytest.mark.parametrize(
    "case", ["interrupted switch-off", "halt in the wait", "ctrl-c in the wait"]
)
def test_a_z_stop_interrupted_or_halted_frees_its_core_without_the_cyclic_collector(
    no_cyclic_gc: None,
    no_kept_log_records: None,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    # FM-92, #166: the stop paths this fix added (an interrupt held until the
    # offset stop was sent, `_outcome`'s nested decision, a halt read in the
    # wait) raise errors through frames of their own, as #163's did.
    core = _at_offset(7800.0)
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile())
    if case == "interrupted switch-off":
        raised = _z_stop_interrupted_in_its_switch_off(core, microscope)
        # The row the case meant to take: the offset device was stopped past it.
        assert core.log == [
            "setPosition('PFSOffset', 7801.0)",
            OFF,
            "stop('PFSOffset')",
            "stop('ZDrive')",
        ]
        expected = (
            "MotionStoppedError",
            _not_switched_off(
                "autofocus_offset: move_to 7801.0 device units", "KeyboardInterrupt"
            ),
            "MotionStoppedError",
        )
    else:
        raised = _z_stop_whose_outcome_wait_is_cut_short(
            core, microscope, monkeypatch, case
        )
        expected = (
            ("KeyboardInterrupt", "", "NoneType")
            if case == "ctrl-c in the wait"
            else (
                "MotionStoppedError",
                _not_switched_off(
                    "autofocus: engage", "its switch-off had not returned yet"
                ),
                "MotionStoppedError",
            )
        )
    assert raised == expected
    assert microscope.state().moving == ()
    alive = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}
    del microscope, core
    assert {name: ref() is not None for name, ref in alive.items()} == {
        "facade": False,
        "core": False,
    }
