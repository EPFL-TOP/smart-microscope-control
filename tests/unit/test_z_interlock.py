"""The Z interlock holds any autofocus that is a ``ZInterlock`` (m4 §11.7).

Until #26 the facade interlocked ``MMAutofocus`` alone, so an autofocus of
another class (DF2's ``MMFocusStabilizer``) would have been moved under with
no suspend: another MTB client's periodic stabilisation would have fought the
Z move. Now the Protocol decides, checked at run time, and the override's
WARNING is for an implementation that is not one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pytest
from pymmcore_plus import DeviceType

from recording_autofocus import RecordingAutofocus
from smc.hardware import Microscope
from smc.hardware.backends.mm import ZInterlock
from smc.hardware.capabilities import Autofocus, ZStage
from smc.hardware.profile import MicroscopeSection, Profile, RolesSection
from smc.testing import FakeCore, FakeDevice

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

#: The override's WARNING for an autofocus that Z moves no longer suspend.
NOT_INTERLOCKED = "Autofocus overridden: Z moves no longer suspend and re-engage it"


class _Interlocked(RecordingAutofocus):
    """A recording autofocus that is also a ``ZInterlock``: it records the suspend and the relock."""

    def suspend(self, reason: str) -> bool:
        self.calls.append(f"suspend for {reason}")
        was_on, self.engaged = self.engaged, False
        return was_on

    def relock(self, reason: str, *, z_um: float | None) -> None:
        self.calls.append(f"relock after {reason}")
        self.engaged = True


def test_any_autofocus_that_is_a_z_interlock_is_interlocked(
    fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    autofocus = _Interlocked(engaged=True)
    with caplog.at_level(logging.WARNING):
        fake_microscope.override(Autofocus, autofocus)
    assert NOT_INTERLOCKED not in caplog.messages
    z = fake_microscope.require(ZStage)
    target = z.position_um() - 1.0
    z.move_to_um(target)
    assert autofocus.calls == [
        f"suspend for z: move_to {target} µm",
        f"relock after z: move_to {target} µm",
    ]
    assert autofocus.engaged is True


def test_an_override_that_is_no_interlock_warns_and_is_not_interlocked(
    fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    autofocus = RecordingAutofocus(engaged=True)
    with caplog.at_level(logging.WARNING):
        fake_microscope.override(Autofocus, autofocus)
    assert NOT_INTERLOCKED in caplog.messages
    z = fake_microscope.require(ZStage)
    z.move_to_um(z.position_um() - 1.0)
    # Not asked whether it is engaged, not switched off: it is left alone.
    assert autofocus.calls == []


def test_point_autofocus_slot_leaves_a_generic_device_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The core's AutoFocus slot refuses a Generic device (ValueError, measured on
    # 0.18.1), and a focus stabilizer makes no continuous-focus call (m4 §11.7).
    core = FakeCore(
        [
            FakeDevice("ZDrive", DeviceType.Stage),
            FakeDevice("PFS", DeviceType.AutoFocus),
            FakeDevice("DefiniteFocus", DeviceType.GenericDevice),
        ],
        focus="ZDrive",
        autofocus="PFS",
    )
    profile = Profile(
        microscope=MicroscopeSection(name="df2-fake"),
        roles=RolesSection.model_validate({"assign": {"autofocus": "DefiniteFocus"}}),
    )
    with caplog.at_level(logging.INFO):
        microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    try:
        assert not any(entry.startswith("setAutoFocusDevice") for entry in core.log)
        assert core.getAutoFocusDevice() == "PFS"
        assert not any("AutoFocus slot" in m for m in caplog.messages)
        autofocus = microscope.require(Autofocus)
        assert type(autofocus).__name__ == "MMFocusStabilizer"
        assert isinstance(autofocus, ZInterlock)
        assert microscope._interlock() is autofocus
    finally:
        microscope.close()
