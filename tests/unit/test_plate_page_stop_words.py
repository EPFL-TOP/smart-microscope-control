"""What the page says after a STOP that the stand refuses in part, and the layer's words on the page (m3 §15, #544, FM-254).

The Ti2-E refuses the stop of its XY stage, Z drive and PFS offset, and the page showed the
facade's sentence behind its own ``halted, but``, ending on ``then call `resume()```: an API
call, on a page whose way out is the **Resume** button. These tests run the real facade on a
``FakeCore`` whose stop is refused (or fails), never a copy of its words, so a rewording in the
layer fails here and not at the stand. The CLI keeps the layer's text (its tests are
``test_refused_stop.py`` and ``test_cli.py``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera
from smc.hardware.errors import MicroscopeHaltedError
from smc.hardware.profile import Profile, SafetySection
from smc.hardware.roles import Role
from smc.plugins.api import PluginParams, PluginResult, RunContext, StopRequested
from smc.plugins.registry import GROUP
from smc.run import RunDocument, StageStatus
from smc.testing import FakeCore
from smc.ui.controller import PlatePage

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

#: Seconds a test waits for the other thread before it gives up: a bound, never a timing (FM-44).
BOUND_S = 10.0

#: The demo-like fake's XY and Z both refuse, and its profile cuts their moves: what the
#: strip says after a STOP on it, from the device labels the fake gives (``XY``, ``Z``,
#: ``Autofocus``) and the profile's sizes.
REFUSED = (
    "the stand is halted (Resume clears the halt) and the autofocus (Autofocus) is "
    "switched off. XY and Z refuse a stop on this stand: a move already sent finishes "
    "first; a long one ends at the end of its current segment (at most 500 µm in XY, "
    "100 µm in Z). Use the stand's own controls to stop a move that must not finish."
)


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class HaltedScan:
    """``scan`` as a real one ends when STOP lands in a move: the halt refuses the next move."""

    name = provides = "scan"
    summary = "A test scan that meets the halt."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        HaltedScan.started.set()
        try:
            while True:
                ctx.wait_s(0.05)
        except StopRequested:
            raise MicroscopeHaltedError() from None


class MeetsTheHalt(HaltedScan):
    """``scan`` started on a stand that was halted before: the halt, with no STOP of this run."""

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        raise MicroscopeHaltedError()


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    HaltedScan.started = threading.Event()


def _stand(
    *,
    refused: set[str],
    declared: tuple[Role, ...],
    segments: bool = True,
) -> Microscope:
    """A fake demo stand whose ``refused`` devices refuse their stop, as the Ti2-E's do."""
    core = FakeCore.demo_like()
    core.stop_refused = refused
    demo = Profile.demo()
    update: dict[str, Any] = {"stop_refused_by": [role.value for role in declared]}
    if segments:
        update |= {"max_segment_um": 500.0, "max_z_segment_um": 100.0}
    safety = SafetySection.model_validate({**demo.safety.model_dump(), **update})
    return Microscope.from_core(
        cast("CMMCorePlus", core), demo.model_copy(update={"safety": safety})
    )


@pytest.fixture
def refusing(tmp_path: Path) -> Iterator[PlatePage]:
    """A page on a stand whose XY and Z refuse the stop and whose moves are cut into segments."""
    microscope = _stand(refused={"XY", "Z"}, declared=(Role.xy_stage, Role.focus))
    scan = EntryPoint(name="scan", value=f"{__name__}:HaltedScan", group=GROUP)
    page = PlatePage(microscope, tmp_path / "run.json", entry_points=[scan])
    try:
        yield page
    finally:
        page.close()
        microscope.close()


def _says_no_api_call(text: str | None) -> None:
    assert text is not None
    for word in ("resume()", "halted, but", "Microscope.stop()", "`"):
        assert word not in text, text


def test_idle_stop_on_a_refusing_stand_says_it_in_the_pages_words(
    refusing: PlatePage,
) -> None:
    refusing.stop()
    status = refusing.status()
    assert status.stop_note == REFUSED
    _says_no_api_call(status.stop_note)
    # FM-117: a stop the stand refuses is never told as "stopped".
    assert not (status.stop_note or "").startswith("stopped")
    assert status.halted


def test_stop_during_live_on_a_refusing_stand_says_the_same_after_its_prefix(
    refusing: PlatePage,
) -> None:
    refusing.live(True)
    deadline = time.monotonic() + BOUND_S
    while refusing.frame() is None:
        assert time.monotonic() < deadline, "Live took no frame"
        time.sleep(0.02)
    refusing.stop()
    assert refusing.join(BOUND_S)
    status = refusing.status()
    assert status.stop_note == f"live told to stop; {REFUSED}"
    _says_no_api_call(status.stop_note)
    assert (status.outcome or "").startswith("live ended: ")


def test_stop_during_a_scan_on_a_refusing_stand_says_the_same_and_the_console_says_no_api_name(
    refusing: PlatePage, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(logging.INFO, logger="smc.ui"):
        refusing.run("scan", [])
        assert HaltedScan.started.wait(BOUND_S)
        refusing.stop()
        assert refusing.join(BOUND_S)
    status = refusing.status()
    assert status.stop_note == f"Scan told to stop; {REFUSED}"
    _says_no_api_call(status.stop_note)
    # The strip's own line for a scan that the halt cut is right, and stays.
    assert status.outcome == "Scan stopped by STOP"
    # The server console: the type stays (it is for the log), the API call goes.
    assert (
        "scan failed: MicroscopeHaltedError: the microscope was stopped; "
        "press Resume to continue"
    ) in caplog.messages
    for message in caplog.messages:
        _says_no_api_call(message)
    # And the record the cards draw says it the same way, with the layer's text kept in
    # the run document.
    record = RunDocument.load(tmp_path / "run.json").stages["scan"]
    assert record.status is StageStatus.failed
    assert record.error is not None
    assert "call `resume()`" in record.error
    view = refusing.view()
    _says_no_api_call(view.cards["scan"].error)
    assert view.scan_end is not None
    _says_no_api_call(view.scan_end.text)


def test_a_failed_stop_says_press_resume_not_call_resume(tmp_path: Path) -> None:
    # The fake's XY refuses, and the profile does not declare it: a fault, not a refusal,
    # so the facade raises its own HardwareError and its text ends on the API call.
    microscope = _stand(refused={"XY"}, declared=(), segments=False)
    page = PlatePage(microscope, tmp_path / "run.json", entry_points=[])
    try:
        page.stop()
        note = page.status().stop_note
    finally:
        page.close()
        microscope.close()
    assert note is not None
    assert note.startswith("the stop failed: the emergency stop failed for XYStage")
    assert note.endswith(
        "The microscope stays halted: stop the stage at the stand if it still moves, "
        "then press Resume"
    )
    _says_no_api_call(note)


def test_a_run_that_meets_a_halt_it_was_not_stopped_by_says_press_resume(
    tmp_path: Path,
) -> None:
    # The `failed` branch of the outcome, not the `stopped by STOP` one: the line carries the
    # layer's text, so it is the page's words.
    microscope = _stand(refused=set(), declared=(), segments=False)
    scan = EntryPoint(name="scan", value=f"{__name__}:MeetsTheHalt", group=GROUP)
    page = PlatePage(microscope, tmp_path / "run.json", entry_points=[scan])
    try:
        page.run("scan", [])
        assert page.join(BOUND_S)
        outcome = page.status().outcome
    finally:
        page.close()
        microscope.close()
    assert (
        outcome == "Scan failed: the microscope was stopped; press Resume to continue"
    )
    _says_no_api_call(outcome)


def test_a_live_that_meets_the_halt_ends_without_the_api_call(
    refusing: PlatePage,
) -> None:
    # A camera that raises the halt's real error (no STOP is pressed here: that the frame
    # fails on the halt is the stub's doing). The outcome says the layer's cause in the
    # page's words; the type stays, it is the Live's existing line.
    microscope = refusing._stand()
    camera = microscope.require(Camera)

    def halted() -> Any:
        raise MicroscopeHaltedError()

    camera.snap = halted  # type: ignore[method-assign]
    try:
        refusing.live(True)
        assert refusing.join(BOUND_S)
    finally:
        del camera.snap
    outcome = refusing.status().outcome
    assert outcome == (
        "live ended after 0 frames: MicroscopeHaltedError: "
        "the microscope was stopped; press Resume to continue"
    )
    _says_no_api_call(outcome)


def test_a_stop_the_stand_accepts_is_told_as_before_whatever_the_profile_declares(
    tmp_path: Path,
) -> None:
    # The profile declares XY refused, but this stand's XY takes the stop: the text follows
    # what happened, not what was declared.
    microscope = _stand(refused=set(), declared=(Role.xy_stage,), segments=False)
    page = PlatePage(microscope, tmp_path / "run.json", entry_points=[])
    try:
        page.stop()
        note = page.status().stop_note
    finally:
        page.close()
        microscope.close()
    assert note == "stopped: the stand is halted (Resume clears the halt)"


def test_a_refusal_on_a_stand_whose_moves_are_not_cut_says_nothing_of_segments(
    tmp_path: Path,
) -> None:
    microscope = _stand(refused={"XY"}, declared=(Role.xy_stage,), segments=False)
    page = PlatePage(microscope, tmp_path / "run.json", entry_points=[])
    try:
        page.stop()
        note = page.status().stop_note
    finally:
        page.close()
        microscope.close()
    assert note == (
        "the stand is halted (Resume clears the halt) and the autofocus (Autofocus) is "
        "switched off. XY refuses a stop on this stand: a move already sent finishes "
        "first. Use the stand's own controls to stop a move that must not finish."
    )
