"""What only the demo shows of the autofocus (m2 §1, §3.13).

The contracts run on the demo too; these are the demo's own facts, measured
2026-10-16 (pymmcore-plus 0.18.1): ``DAutoFocus`` fills the role from the
core's slot, no device fills ``autofocus_offset``, so the offset goes through
the autofocus device itself, and enabling continuous focus locks at once.
"""

from __future__ import annotations

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, ZStage
from smc.hardware.roles import Role

pytestmark = pytest.mark.demo


def test_demo_autofocus_comes_from_the_core_slot_and_has_no_offset_device(
    demo_microscope: Microscope,
) -> None:
    roles = demo_microscope.roles
    assert roles.get(Role.autofocus) == "Autofocus"
    assert roles.sources[Role.autofocus] == "core"
    assert roles.get(Role.autofocus_offset) is None
    assert demo_microscope.core.getAutoFocusDevice() == "Autofocus"


def test_demo_offset_goes_through_dautofocus(demo_microscope: Microscope) -> None:
    autofocus = demo_microscope.require(Autofocus)
    z_before = demo_microscope.require(ZStage).position_um()
    assert autofocus.set_offset_native(12.5) == 12.5  # measured: reads back 12.5
    assert demo_microscope.core.getAutoFocusOffset() == 12.5
    assert autofocus.offset_native() == 12.5
    # The demo's offset moves nothing.
    assert demo_microscope.require(ZStage).position_um() == z_before


def test_demo_engage_locks_at_once(demo_microscope: Microscope) -> None:
    autofocus = demo_microscope.require(Autofocus)
    assert autofocus.is_engaged() is False  # the demo starts off
    assert autofocus.engage() is True
    assert autofocus.is_locked() is True
    # A Z move under it ends engaged: on the demo, as on a Nikon stand.
    demo_microscope.require(ZStage).move_by_um(-1.0)
    assert autofocus.is_locked() is True
    autofocus.disengage()
    assert autofocus.is_engaged() is False


def test_demo_state_reports_the_autofocus(demo_microscope: Microscope) -> None:
    state = demo_microscope.state()
    assert (
        state.autofocus_engaged,
        state.autofocus_locked,
        state.autofocus_offset_native,
    ) == (False, False, 0.0)
    assert state.errors == []
    assert demo_microscope.describe().endswith("| autofocus off")
