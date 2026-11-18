"""After *Measure the time* the page locks the autofocus again at the timed offset (m5 §20, #546).

At the Ti2-E (session 5, F10) *Measure the time* timed the PFS lock at offset 7808 and then
switched PFS off before its probe. The checklist had the operator Disconnect and run ``smc
autofocus engage``, which is refused by design: the offsets seen locked live in the process that
opened the stand (m2 §12.2 D1). The page's own *Autofocus on*, in the connection that had read the
lock, was never tried. These tests pin that it works, on the declared stand of §20.1: a
``FakeCore.demo_like()`` whose autofocus drives Z to its offset's plane
(``engage_drives_z_to_offset``), where a lock from the floor would be refused and a seen offset is
engaged with one switch-on and nothing else.

Texts are asserted whole (FM-45), threads meet on a bounded ``join`` (FM-44).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, ZStage
from smc.hardware.errors import SafetyRefusedError
from smc.hardware.profile import AutofocusSection, Profile
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.guide import MEASURE_THE_TIME

#: A bound for a slow CI runner, never a wait for something that must happen (FM-44).
WAIT_S = 30.0
#: The offset session 5's timing read locked (F10).
TIMED_OFFSET = 7808.0
#: What a focus call of the page may send; anything else is the camera's or the lamp's.
FOCUS_CALLS = ("enableContinuousFocus", "setAutoFocusOffset", "setPosition")

#: m2 §12.5's unseen-offset text, written out and not read from the code (FM-45): the stand declares
#: ``engage_drives_z_to_offset`` and ``max_engage_excursion_um`` and nothing for a first lock from
#: below, so the advice is the one without it.
UNSEEN_SECOND_OPENING = (
    "autofocus Autofocus: not engaged: on this stand engaging drives Z to the plane the "
    "offset names ([autofocus] engage_drives_z_to_offset, m2 §12), and offset 7808 has not "
    "been seen locked with objective 'Nikon 10X S Fluor' since the stand was opened (seen: "
    "none); each smc command and each Connect of the page opens the stand anew, so a lock "
    "read by another one does not count. A first lock from below (m2 §12.10) needs [safety] "
    "autofocus_offset_soft_limits and [safety] z_away_from_sample, which this stand does not "
    "have; until then smc engages only at an offset it has read locked"
)


def _declared_profile() -> Profile:
    """``Profile.demo()`` that declares what the Ti2-E declares: the autofocus drives Z to its offset."""
    profile = Profile.demo().model_copy(deep=True)
    profile.autofocus = AutofocusSection.model_validate(
        {"engage_drives_z_to_offset": True}
    )
    profile.safety = profile.safety.model_copy(update={"max_engage_excursion_um": 50.0})
    return profile


@pytest.fixture
def stand(tmp_path: Path) -> Iterator[tuple[FakeCore, Microscope, PlatePage]]:
    """The declared fake stand and the page over it, the autofocus locked at the timed offset by the operator."""
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _declared_profile())
    page = PlatePage(microscope, tmp_path / "run.json")
    try:
        core.setAutoFocusOffset(TIMED_OFFSET)
        core.enableContinuousFocus(True)  # the operator locked it at the stand
        assert (
            microscope.require(Autofocus).is_locked() is True
        )  # the page read it locked
        yield core, microscope, page
    finally:
        page.close()
        microscope.close()


def _measure_the_time(page: PlatePage) -> None:
    """*Measure the time*'s chain: the one ``timing`` run, one repeat, labelled as the button does."""
    page.run_chain(("timing",), {"timing": ["repeats=1"]}, label=MEASURE_THE_TIME)
    assert page.join(WAIT_S), "the chain did not end"


def _focus_calls(core: FakeCore) -> list[str]:
    return [line for line in core.log if line.startswith(FOCUS_CALLS)]


def test_autofocus_on_after_measure_the_time_locks_again_at_the_timed_offset(
    stand: tuple[FakeCore, Microscope, PlatePage],
) -> None:
    """The page's *Autofocus on* after the timing: one switch-on, no offset written, no floor step, Z where it was."""
    core, microscope, page = stand
    _measure_the_time(page)
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith(f"{MEASURE_THE_TIME}: done in "), outcome
    assert (
        microscope.require(Autofocus).is_engaged() is False
    )  # the timing switched it off
    z_before = microscope.require(ZStage).position_um()
    core.log.clear()

    page.autofocus(True)
    assert page.join(WAIT_S), "Autofocus on did not end"

    assert page.status().outcome == "autofocus engage: autofocus locked"
    assert microscope.require(Autofocus).is_locked() is True
    assert core.getAutoFocusOffset() == TIMED_OFFSET
    assert _focus_calls(core) == ["enableContinuousFocus(True)"]
    assert microscope.require(ZStage).position_um() == z_before


def test_a_second_opening_of_the_stand_refuses_the_lock_the_first_read(
    stand: tuple[FakeCore, Microscope, PlatePage],
) -> None:
    """What ``smc autofocus engage`` is after the page read the lock: another process's memory does not count (D5, FM-45)."""
    core, _first, page = stand
    page.autofocus(False)
    assert page.join(WAIT_S), "Autofocus off did not end"
    assert core.isContinuousFocusEnabled() is False
    core.log.clear()

    second = Microscope.from_core(cast("CMMCorePlus", core), _declared_profile())
    try:
        with pytest.raises(
            SafetyRefusedError, match=f"^{re.escape(UNSEEN_SECOND_OPENING)}$"
        ):
            second.require(Autofocus).engage()
    finally:
        second.close()

    assert core.isContinuousFocusEnabled() is False
    assert _focus_calls(core) == []
