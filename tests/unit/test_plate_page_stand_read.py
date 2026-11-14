"""While an activity goes, the page's status reads no device on its caller's thread (m3 §15.4, #333).

Measured at the Ti2-E (#334): ``Microscope.state()`` takes 0.4 ms idle and 133 ms
median, 464 ms at most, while another thread snaps, and the tick that calls it is
on the loop that also takes the click on STOP. So during an activity ``status()``
serves the latest stand read and reads the stand on one daemon thread
(``smc-stand-read``); between activities it reads directly, as before.

No Panel except in the acceptance test, which builds the view the way the server
does. The held reads are held by the test's own events, never by a sleep, and each
test releases them in a ``finally`` and joins the page, so that no daemon thread
outlives its test with a held fake (FM-44, FM-96, FM-98). Texts are asserted whole
or by the text around a number (FM-45).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.testing import FakeCore
from smc.ui.controller import PlatePage, _StandRead

READ_THREAD = "smc-stand-read"
LATE_TEXT = re.compile(
    r"^the stand has not answered a read for \d+ s; the strip shows its last answer$"
)


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class Slow:
    """Runs until released or stopped: the activity that holds the device."""

    name = provides = "slow"
    summary = "A test plugin that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        Slow.started.set()
        while not Slow.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


ENTRY_POINTS = [EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP)]


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()


@pytest.fixture
def page(fake_microscope: Microscope, tmp_path: Path) -> Iterator[PlatePage]:
    controller = PlatePage(
        fake_microscope, tmp_path / "run.json", entry_points=ENTRY_POINTS
    )
    try:
        yield controller
    finally:
        Slow.release.set()
        controller.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _eventually(condition: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    """Poll ``condition`` for at most ``timeout_s`` (FM-44); whether it came true."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def _no_read_thread_left() -> bool:
    return not any(
        t.name == READ_THREAD and t.is_alive() for t in threading.enumerate()
    )


def _hold_a_run(page: PlatePage) -> None:
    """Start ``slow`` and wait until it holds the device."""
    page.run("slow", [])
    assert Slow.started.wait(5)


def _end_the_run(page: PlatePage) -> None:
    """End ``slow`` and wait for it; no status here, which would read the stand at once."""
    Slow.release.set()
    assert page.join(5)


class _Held:
    """A device call held by the test's events: ``inside`` says it was entered, ``go`` lets it return."""

    def __init__(self, real: Callable[..., Any]) -> None:
        self.real = real
        self.inside = threading.Event()
        self.go = threading.Event()
        self.threads: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.threads.append(threading.current_thread().name)
        self.inside.set()
        assert self.go.wait(10), "the test never let the read go"
        return self.real(*args, **kwargs)


class _Spy:
    """A device call that answers at once and notes the thread it was made on."""

    def __init__(self, real: Callable[..., Any]) -> None:
        self.real = real
        self.threads: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.threads.append(threading.current_thread().name)
        return self.real(*args, **kwargs)


def _view_class() -> Any:
    """``PlatePageView``; CI (``SMC_REQUIRE_UI``) fails, not skips, when Panel is missing."""
    if os.environ.get("SMC_REQUIRE_UI"):
        import panel  # noqa: F401
    else:
        pytest.importorskip("panel")
    from smc.ui.plate_page import PlatePageView

    return PlatePageView


# --- the acceptance: STOP is not held by a read that waits on the snap -----------------------


@pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)
def test_stop_during_live_is_not_held_by_a_status_read_that_waits_on_the_snap(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # The Ti2-E's case, #333: ``state()`` reads the camera, and MMCore holds the
    # camera's lock for the whole snap. The fake blocks ``state()`` for as long as a
    # snap is held, and the tick and the click on STOP run on one thread, as the
    # server's loop does. Without the fix both calls wait for ``go``; 5 s is a join
    # timeout against an unbounded wait, not a measured duration.
    view = _view_class()(page)
    view.tick()
    camera = fake_microscope.require(Camera)
    real_snap = camera.snap
    real_state = fake_microscope.state
    snapping = threading.Event()
    go = threading.Event()

    def held_snap() -> Any:
        snapping.set()
        assert go.wait(10), "the test never let the snap go"
        try:
            return real_snap()
        finally:
            snapping.clear()

    def state_behind_the_snap() -> Any:
        if snapping.is_set():
            assert go.wait(10), "the test never let the read go"
        return real_state()

    done = threading.Event()
    failures: list[str] = []

    def the_loop() -> None:
        try:
            view.tick()
            view.on_stop()
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            done.set()

    camera.snap = held_snap  # type: ignore[method-assign]
    fake_microscope.state = state_behind_the_snap  # type: ignore[method-assign]
    try:
        page.live(True)
        assert snapping.wait(5)
        loop = threading.Thread(target=the_loop, daemon=True)
        loop.start()
        # While the snap is still held.
        assert done.wait(5), "the tick or STOP waited on a device read during Live"
        assert snapping.is_set()
        assert failures == []
        stop_note = page.status().stop_note
        assert stop_note is not None
        assert stop_note.startswith("live told to stop;"), stop_note
    finally:
        go.set()
        del camera.snap
        del fake_microscope.state
        page.live(False)
        assert page.join(5)


# --- what status() does during an activity ---------------------------------------------------


def test_status_during_an_activity_calls_no_device_on_the_callers_thread(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    page.status()
    spy = _Spy(fake_microscope.state)
    with _replaced(fake_microscope, "state", spy):
        _hold_a_run(page)
        try:
            caller = threading.current_thread().name
            assert caller != READ_THREAD
            deadline = time.monotonic() + 5
            while not spy.threads and time.monotonic() < deadline:
                page.status()
                time.sleep(0.005)
            assert spy.threads, "no stand read ran off the loop"
            for _ in range(20):
                page.status()
            assert caller not in spy.threads
            assert set(spy.threads) == {READ_THREAD}
        finally:
            _end_the_run(page)
        assert _eventually(_no_read_thread_left)


def test_one_stand_read_in_flight_whatever_the_number_of_status_calls(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    held = _Held(fake_microscope.state)
    with _replaced(fake_microscope, "state", held):
        _hold_a_run(page)
        try:
            for _ in range(10):
                page.status()
            assert held.inside.wait(5)
            for _ in range(10):
                page.status()
            assert len(held.threads) == 1
            # A daemon, so that a camera that never answers cannot hold the process
            # at its exit.
            readers = [t for t in threading.enumerate() if t.name == READ_THREAD]
            assert [t.daemon for t in readers] == [True]
            held.go.set()
            # Once the read is released the next call starts a second one.
            assert _eventually(
                lambda: page.status() is not None and len(held.threads) >= 2
            )
        finally:
            held.go.set()
            _end_the_run(page)
        assert _eventually(_no_read_thread_left)


def test_a_read_that_began_before_a_newer_one_is_not_kept(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    real = fake_microscope.state
    inside = threading.Event()
    go = threading.Event()
    calls: list[str] = []

    def first_call_held() -> Any:
        calls.append(threading.current_thread().name)
        if len(calls) > 1:
            return real()
        snapshot = real()  # the stand as it is when the read begins
        inside.set()
        assert go.wait(10), "the test never let the read go"
        return snapshot

    xy = fake_microscope.require(XYStage)
    before = xy.position_um()
    after = (before.x_um + 250.0, before.y_um - 125.0)
    with _replaced(fake_microscope, "state", first_call_held):
        try:
            _hold_a_run(page)
            page.status()
            assert inside.wait(5)
            _end_the_run(page)
            xy.move_to_um(*after)
            # A direct read, after the activity: the newer read.
            assert page.status().xy_um == after
            go.set()
            assert _eventually(_no_read_thread_left)
            Slow.started = threading.Event()
            Slow.release = threading.Event()
            _hold_a_run(page)
            # The first call of the new activity is served from the store: the old
            # read, which began first, did not replace the direct one.
            assert page.status().xy_um == after
        finally:
            go.set()
            _end_the_run(page)
    assert (before.x_um, before.y_um) != after


def test_status_during_an_activity_before_any_read_says_the_stand_is_unknown(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    held = _Held(fake_microscope.state)
    with _replaced(fake_microscope, "state", held):
        _hold_a_run(page)
        try:
            status = page.status()
            assert status.running == "slow"
            assert status.xy_um is None
            assert status.z_um is None
            assert status.objective is None
            assert status.autofocus == "?"
            assert status.light_on is None
            assert status.moving == ()
            assert status.halted is False
            assert status.problems == ()
            assert status.connected is True
            assert held.inside.wait(5)
            assert threading.current_thread().name not in held.threads
            # The read is in flight and has not been going for the bound: nothing is
            # said yet (D5). A read said late at once would put a line on every strip.
            assert page.status().problems == ()
        finally:
            held.go.set()
            _end_the_run(page)
        assert _eventually(_no_read_thread_left)


def test_a_stand_read_late_by_the_bound_is_said_in_problems(
    page: PlatePage, fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_STAND_READ_LATE_S", 0.2)
    held = _Held(fake_microscope.state)

    def told_late() -> bool:
        return any(LATE_TEXT.match(problem) for problem in page.status().problems)

    with _replaced(fake_microscope, "state", held):
        _hold_a_run(page)
        try:
            page.status()  # starts the read, which the test then holds
            assert held.inside.wait(5)
            assert _eventually(told_late)
            held.go.set()
            # The read is stored and the next one is new: nothing is late any more.
            assert _eventually(lambda: not told_late())
        finally:
            held.go.set()
            _end_the_run(page)
        assert _eventually(_no_read_thread_left)


def test_a_background_read_that_raises_is_said_once_and_the_next_one_clears_it(
    page: PlatePage, fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    real = fake_microscope.state
    raised = 0
    text = "stand read: RuntimeError: the camera did not answer"

    def failing_three_times() -> Any:
        nonlocal raised
        if raised < 3:
            raised += 1
            raise RuntimeError("the camera did not answer")  # a fresh error each time
        return real()

    def said() -> bool:
        return text in page.status().problems

    caplog.set_level(logging.WARNING, logger="smc.ui")
    with _replaced(fake_microscope, "state", failing_three_times):
        _hold_a_run(page)
        try:
            assert _eventually(said)
            assert _eventually(lambda: page.status() is not None and raised == 3)
            assert _eventually(lambda: not said())
            assert _eventually(lambda: page.status().xy_um is not None)
            assert page.status().problems == ()
        finally:
            _end_the_run(page)
        assert _eventually(_no_read_thread_left)
    warnings = [r.getMessage() for r in caplog.records if r.getMessage() == text]
    assert warnings == [text]


def test_a_disconnect_waits_for_a_background_stand_read(
    tmp_path: Path,
) -> None:
    # The connect helpers of ``test_plate_page_connect.py``, copied: test modules do
    # not import each other.
    stand = Microscope.from_core(
        cast("CMMCorePlus", FakeCore.demo_like()), Profile.demo()
    )
    page = PlatePage(
        stand, tmp_path / "run.json", opener=lambda: stand, entry_points=ENTRY_POINTS
    )
    held = _Held(stand.state)
    closing = threading.Event()
    real_close = stand.close

    def watched_close() -> None:
        closing.set()
        real_close()

    stand.state = held  # type: ignore[method-assign]
    stand.close = watched_close  # type: ignore[method-assign]
    try:
        _hold_a_run(page)
        page.status()
        assert held.inside.wait(5)
        _end_the_run(page)
        page.disconnect()
        # Past the drop: from here a missing wait reaches close() within
        # milliseconds, so what follows cannot pass before it could have failed (FM-96).
        assert _eventually(lambda: not page.connected)
        assert not closing.wait(0.3)
        assert list(stand.core.getLoadedDevices()) != ["Core"]
        held.go.set()
        assert page.join(5)
        assert closing.is_set()
        assert list(stand.core.getLoadedDevices()) == ["Core"]
        assert page.status().outcome == "disconnect: demo released"
    finally:
        held.go.set()
        del stand.state
        del stand.close
        page.close()
        stand.close()


def test_the_problems_of_the_stored_read_stay_on_the_strip_during_an_activity(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A device fault the last read found must not disappear when a scan starts: the
    # strip is then the only thing in view. Order: the stored read's own problems,
    # then the text of the background read that raised.
    real = fake_microscope.state
    fault = "z: RuntimeError: the focus did not answer"
    error = "stand read: RuntimeError: the camera did not answer"
    go = threading.Event()
    calls: list[int] = []

    def state() -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the camera did not answer")
        if len(calls) == 3:
            assert go.wait(10), "the test never let the read go"
        found = real()
        found.errors.append(fault)
        return found

    with _replaced(fake_microscope, "state", state):
        assert page.status().problems == (fault,)
        _hold_a_run(page)
        try:
            assert page.status().problems == (fault,)
            assert _eventually(lambda: page.status().problems == (fault, error))
            go.set()
            assert _eventually(lambda: page.status().problems == (fault,))
        finally:
            go.set()
            _end_the_run(page)
        assert _eventually(_no_read_thread_left)


def test_a_thread_that_cannot_start_is_a_read_that_raised_and_the_next_status_tries_again(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_start = threading.Thread.start
    refused: list[str] = []

    def start(self: threading.Thread) -> None:
        if self.name == READ_THREAD and not refused:
            refused.append(self.name)
            raise RuntimeError("can't start new thread")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    _hold_a_run(page)
    try:
        assert page.status().problems == ()
        assert refused == [READ_THREAD]
        # The failure is said, and the in-flight mark was cleared: this status starts
        # the read again, and when it lands the strip knows the stand.
        assert page.status().problems == (
            "stand read: RuntimeError: can't start new thread",
        )
        assert _eventually(lambda: page.status().xy_um is not None)
        assert _eventually(lambda: page.status().problems == ())
    finally:
        _end_the_run(page)
    assert _eventually(_no_read_thread_left)


def test_a_closed_read_kept_before_a_connect_is_not_served_as_the_stand(
    tmp_path: Path,
) -> None:
    # The Connect is an activity, and the activity after it must not say ``not
    # connected`` of a page that holds a stand, from the read kept before it, until
    # the first background read lands (adversarial review of #333).
    stands: list[Microscope] = []

    def opener() -> Microscope:
        stand = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()), Profile.demo()
        )
        stands.append(stand)
        return stand

    page = PlatePage.closed(
        Profile.demo(), tmp_path / "run.json", opener=opener, entry_points=ENTRY_POINTS
    )
    try:
        assert page.status().connected is False
        page.connect()
        assert page.join(5)
        assert page.connected
        _hold_a_run(page)
        status = page.status()
        assert status.running == "slow"
        assert status.connected is True
        assert status.xy_um is None
        assert status.autofocus == "?"
        # And it is the page's stand that the background read finds.
        assert _eventually(lambda: page.status().xy_um is not None)
    finally:
        Slow.release.set()
        page.close()
        page.release()
        for stand in stands:
            stand.close()
    assert _eventually(_no_read_thread_left)


def test_two_reads_that_began_at_the_same_clock_value_keep_the_one_stored_last(
    page: PlatePage,
) -> None:
    # ``time.monotonic()`` ticks every 15.6 ms on Windows before Python 3.13 (FM-197):
    # a read that began in the same tick as the one kept, after the stage moved, is
    # the newer one. One that began before the kept one is not.
    first = _StandRead(
        began_s=100.0,
        connected=True,
        xy_um=(0.0, 0.0),
        z_um=0.0,
        objective=None,
        autofocus="no autofocus",
        light_on=None,
        moving=(),
        halted=False,
        problems=(),
    )
    same_tick = replace(first, xy_um=(400.0, 300.0))
    earlier = replace(first, began_s=99.0, xy_um=(-5.0, -5.0))
    page._keep_stand_read(first)
    page._keep_stand_read(same_tick)
    assert page._stand_read is same_tick
    page._keep_stand_read(earlier)
    assert page._stand_read is same_tick
