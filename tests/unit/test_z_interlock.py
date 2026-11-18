"""The Z interlock holds any autofocus that is a ``ZInterlock`` (m4 §11.7).

Until #26 the facade interlocked ``MMAutofocus`` alone, so an autofocus of
another class (DF2's ``MMFocusStabilizer``) would have been moved under with
no suspend: another MTB client's periodic stabilisation would have fought the
Z move. Now the Protocol decides, checked at run time, and the override's
WARNING is for an implementation that is not one.
"""

from __future__ import annotations

import logging

import pytest

from recording_autofocus import RecordingAutofocus
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, ZStage

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
