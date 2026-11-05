"""The plate page with the stand closed: Connect, Disconnect and release (m5 §7.2, #264).

No Panel here. The stand is a facade over ``FakeCore.demo_like()``, opened by an
opener the test hands the page (measured twice in one process, m5 §7.5); a
release is seen as a core that lists only ``Core``. Texts are asserted whole
(FM-45); threads meet on ``Event`` and every wait is bounded and asserted
(FM-44); no test pins an order that the fake's speed decides (FM-96).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, suppress
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource, ObjectiveTurret, XYStage
from smc.hardware.errors import (
    CoreError,
    MicroscopeBusyError,
    StopOutcome,
    stop_notes,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, PlatePage
from smc.ui.frames import Frame


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class Slow:
    """Runs until released or stopped."""

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


class Plate:
    """A stand-in registered as ``plate-registration``, so the view has a card for it."""

    name = "plate-registration"
    provides = "plate"
    summary = "A test registration."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = (XYStage,)
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        return NoResult()


ENTRY_POINTS = [
    EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP),
    EntryPoint(name="plate-registration", value=f"{__name__}:Plate", group=GROUP),
]


def _fake_stand(profile: Profile | None = None, *, dry_run: bool = False) -> Microscope:
    return Microscope.from_core(
        cast("CMMCorePlus", FakeCore.demo_like()),
        profile or Profile.demo(),
        dry_run=dry_run,
    )


def _released(microscope: Microscope) -> bool:
    """What a closed ``FakeCore`` facade lists (measured, m5 §7.5)."""
    return list(microscope.core.getLoadedDevices()) == ["Core"]


class Opener:
    """Opens a fake stand; ``gate`` holds it inside the call, as a slow driver does."""

    def __init__(
        self,
        *,
        gate: threading.Event | None = None,
        fail_with: Callable[[], Exception] | None = None,
    ) -> None:
        self.gate = gate
        self.fail_with = fail_with
        self.entered = threading.Event()
        self.opened: list[Microscope] = []
        self.calls = 0

    def __call__(self) -> Microscope:
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never let the opener return"
        if self.fail_with is not None:
            raise self.fail_with()
        microscope = _fake_stand()
        self.opened.append(microscope)
        return microscope


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def opener() -> Iterator[Opener]:
    found = Opener()
    try:
        yield found
    finally:
        for microscope in found.opened:
            microscope.close()


def _closed(
    run_doc: Path,
    opener: Callable[[], Microscope] | None = None,
    *,
    profile: Profile | None = None,
    dry_run: bool = False,
) -> PlatePage:
    return PlatePage.closed(
        profile or Profile.demo(),
        run_doc,
        dry_run=dry_run,
        opener=opener,
        entry_points=ENTRY_POINTS,
    )


@pytest.fixture
def page(run_doc: Path, opener: Opener) -> Iterator[PlatePage]:
    controller = _closed(run_doc, opener)
    try:
        yield controller
    finally:
        controller.close()
        controller.release()


def _connected(page: PlatePage) -> None:
    page.connect()
    assert page.join(5)
    assert page.connected, page.status().outcome


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _refuses(error: type[Exception], text: str) -> AbstractContextManager[Any]:
    """``pytest.raises`` that insists on the whole message, not on a word of it (FM-45)."""
    return pytest.raises(error, match=f"^{re.escape(text)}$")


def _outcome(page: PlatePage) -> str | None:
    return page.status().outcome


# --- the closed page --------------------------------------------------------------


def test_a_closed_page_reads_the_profile_and_not_the_stand(
    run_doc: Path, opener: Opener
) -> None:
    demo = Profile.demo()
    declared = demo.model_copy(
        update={
            "microscope": demo.microscope.model_copy(update={"name": "ti2e-bench"}),
            "safety": demo.safety.model_copy(
                update={"stop_refused_by": (Role.xy_stage,)}
            ),
        }
    )
    page = _closed(run_doc, opener, profile=declared, dry_run=True)
    status = page.status()
    assert not page.connected
    assert not status.connected
    assert status.stand == "ti2e-bench"
    assert status.dry_run
    assert status.stop_refused == ("xy_stage",)
    assert status.xy_um is None
    assert status.z_um is None
    assert status.objective is None
    assert status.light_on is None
    assert status.autofocus == "?"
    assert status.moving == ()
    assert not status.halted
    assert status.problems == ()
    assert page.jog_limits_um()[0] == declared.safety.max_jog_um
    assert opener.calls == 0


def test_every_call_that_needs_the_stand_says_it_is_not_connected(
    page: PlatePage, opener: Opener
) -> None:
    assert page.lacks(XYStage) == NOT_CONNECTED
    assert page.light_lacks() == NOT_CONNECTED
    assert page.objective_labels() == ()
    for act, name in (
        (lambda: page.run("slow", []), "slow"),
        (page.snap, "snap"),
        (lambda: page.live(True), "live"),
        (lambda: page.jog_xy(10.0, 0.0), "stage jog"),
    ):
        act()
        assert _outcome(page) == f"{name}: the stand is not connected"
    assert page.view().blocked == {
        "slow": NOT_CONNECTED,
        "plate-registration": NOT_CONNECTED,
    }
    with _refuses(RuntimeError, NOT_CONNECTED):
        page.capture_centre("A1")
    with _refuses(RuntimeError, NOT_CONNECTED):
        page.resume()
    assert page.join(5)
    assert page.activities_ended == 0
    assert opener.calls == 0


def test_stop_with_the_stand_closed_says_so_and_never_raises(page: PlatePage) -> None:
    page.stop()
    assert page.status().stop_note == "the stand is not connected: nothing to stop"


def test_the_map_the_checklist_and_the_cards_work_with_the_stand_closed(
    run_doc: Path, opener: Opener
) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=datetime(2026, 10, 30, tzinfo=timezone.utc),
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={"plate": "96-well"},
            result={
                "plate": "96-well",
                "a1_center_xy_um": [1000.0, -500.0],
                "rotation_deg": 0.0,
                "method": "references",
                "residual_um": 2.0,
                "scale_error": 0.001,
                "n_refs": 2,
            },
        ),
    )
    closed = _closed(run_doc, opener)
    stand = _fake_stand()
    try:
        connected = PlatePage(stand, run_doc, entry_points=ENTRY_POINTS)
        shut, held = closed.view(), connected.view()
    finally:
        stand.close()
    assert not shut.nominal
    assert shut.wells == held.wells
    assert shut.checklist == held.checklist
    assert shut.cards == held.cards
    assert "plate-registration" in shut.cards
    assert held.blocked == {}
    assert shut.blocked == {
        "slow": NOT_CONNECTED,
        "plate-registration": NOT_CONNECTED,
    }


# --- Connect ----------------------------------------------------------------------


def test_connect_opens_the_stand_on_the_worker_and_is_an_activity(
    page: PlatePage, opener: Opener
) -> None:
    on_worker: list[str] = []

    def opening() -> Microscope:
        on_worker.append(threading.current_thread().name)
        return opener()

    with _replaced(page, "_opener", opening):
        page.connect()
        assert page.join(5)
    devices = len(opener.opened[0].devices)
    assert _outcome(page) == f"connect: demo connected, {devices} devices"
    assert page.connected
    assert page.status().connected
    assert page.activities_ended == 1
    assert on_worker == ["smc-ui-connect"]


def _fake_open(opened: list[tuple[Profile, bool, Microscope]]) -> Callable[..., Any]:
    def open_(profile: Profile, *, dry_run: bool = False) -> Microscope:
        microscope = _fake_stand(profile, dry_run=dry_run)
        opened.append((profile, dry_run, microscope))
        return microscope

    return open_


def test_the_default_opener_is_microscope_open_looked_up_at_connect(
    run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = Profile.demo()
    page = PlatePage.closed(profile, run_doc, entry_points=ENTRY_POINTS)
    opened: list[tuple[Profile, bool, Microscope]] = []
    monkeypatch.setattr(Microscope, "open", _fake_open(opened))
    try:
        _connected(page)
        assert [(p, d) for p, d, _m in opened] == [(profile, False)]
    finally:
        page.close()
        page.release()
    assert _released(opened[0][2])


def test_a_dry_run_connect_says_so(
    run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = PlatePage.closed(
        Profile.demo(), run_doc, dry_run=True, entry_points=ENTRY_POINTS
    )
    opened: list[tuple[Profile, bool, Microscope]] = []
    monkeypatch.setattr(Microscope, "open", _fake_open(opened))
    try:
        _connected(page)
        microscope = opened[0][2]
        assert microscope.dry_run
        assert _outcome(page) == (
            f"connect: demo connected, {len(microscope.devices)} devices (dry run)"
        )
    finally:
        page.close()
        page.release()


def test_a_connect_that_fails_is_an_outcome_and_the_stand_stays_closed(
    run_doc: Path,
) -> None:
    opener = Opener(fail_with=lambda: CoreError("the adapter\ndid not load"))
    page = _closed(run_doc, opener)
    page.connect()
    assert page.join(5)
    assert _outcome(page) == "connect: CoreError: the adapter did not load"
    assert not page.connected
    assert page.activities_ended == 1


def test_connect_when_connected_is_an_outcome_and_starts_no_thread(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    page.connect()
    assert page.join(5)
    assert _outcome(page) == "connect: demo is already connected"
    assert page.activities_ended == 1
    assert opener.calls == 1


def test_stop_while_connecting_halts_the_stand_once_it_is_open(run_doc: Path) -> None:
    gate = threading.Event()
    opener = Opener(gate=gate)
    page = _closed(run_doc, opener)
    try:
        page.connect()
        assert opener.entered.wait(5)
        page.stop()
        gate.set()
        assert page.join(5)
        status = page.status()
        assert status.halted
        outcome = status.outcome or ""
        assert outcome.endswith(", and halted: STOP was pressed while it connected")
        assert status.stop_note == (
            "connect told to stop; stopped: the stand is halted (Resume clears the halt)"
        )
    finally:
        gate.set()
        page.close()
        page.release()


def test_a_connect_that_ends_after_close_releases_what_it_opened(
    run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 0.2)
    gate = threading.Event()
    opener = Opener(gate=gate)
    page = _closed(run_doc, opener)
    try:
        page.connect()
        assert opener.entered.wait(5)
        page.close()
        assert not page.close_ended
    finally:
        gate.set()
    assert page.join(5)
    assert (
        _outcome(page) == "connect: demo was opened after the page closed, and released"
    )
    assert not page.connected
    assert _released(opener.opened[0])
    page.release()


def test_a_late_connect_whose_release_fails_says_so_and_the_page_stays_closed(
    run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 0.2)
    gate = threading.Event()
    opener = Opener(gate=gate)
    real_open = opener.__call__

    def opening() -> Microscope:
        microscope = real_open()
        microscope.close = _busy  # type: ignore[method-assign]
        return microscope

    page = _closed(run_doc, opening)
    try:
        page.connect()
        assert opener.entered.wait(5)
        page.close()
    finally:
        gate.set()
    assert page.join(5)
    assert _outcome(page) == (
        "connect: demo was opened after the page closed, and its release failed: "
        f"MicroscopeBusyError: {BUSY_TEXT}"
    )
    assert not page.connected
    del opener.opened[0].close  # the fixture-less opener's stand, released for real
    opener.opened[0].close()


BUSY_TEXT = (
    "another call has held the microscope for at least 2.0 s; the device driver may "
    "be hung. If the call is legitimately long, raise [micromanager] "
    "device_timeout_ms in the profile; otherwise restart the device or the program."
)


def _busy() -> None:
    raise MicroscopeBusyError("another call", 2.0, at_least=True)


def _boom() -> None:
    raise RuntimeError("boom")


def test_a_connect_clears_the_stop_note_and_the_light_answer(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    first = page.light_lacks()
    assert first == "the light source LED has no on/off that smc knows"
    page.stop()
    assert page.status().stop_note is not None
    page.disconnect()
    assert page.join(5)
    assert page.status().stop_note is not None  # the note outlives the facade
    with _replaced(page, "_opener", _with_light(opener, Light("Lamp", False))):
        _connected(page)
    try:
        assert page.status().stop_note is None
        assert page.light_lacks() is None
    finally:
        del opener.opened[1].get


class Light:
    """A light source's two reads that ``light_lacks`` makes."""

    def __init__(self, name: str, on: bool | None) -> None:
        self.name = name
        self.on = on

    def device(self) -> str:
        return self.name

    def is_on(self) -> bool | None:
        return self.on


def _with_light(opener: Opener, light: Light) -> Callable[[], Microscope]:
    """An opener whose stand's ``get(LightSource)`` is ``light`` (an instance attribute)."""

    def opening() -> Microscope:
        microscope = opener()
        real_get = microscope.get
        microscope.get = (  # type: ignore[method-assign]
            lambda capability: (
                light if capability is LightSource else real_get(capability)
            )
        )
        return microscope

    return opening


def test_a_light_answer_read_across_a_reconnect_is_not_kept_for_the_new_stand(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read of the first stand's light that outlasts a Disconnect and a Connect: its
    # answer belongs to a stand the page no longer holds (m5 §7.2). A Disconnect waits
    # for a read in flight, so this is the case where the wait gave up (#305).
    monkeypatch.setattr(controller_module, "_READS_WAIT_S", 0.05)
    with _replaced(page, "_opener", _with_light(opener, Light("LED", None))):
        _connected(page)
    first = opener.opened[0]

    class Across(Light):
        def is_on(self) -> bool | None:
            page.disconnect()
            assert page.join(5)
            with _replaced(page, "_opener", _with_light(opener, Light("Lamp", False))):
                _connected(page)
            return None

    with _replaced(first, "get", lambda capability: Across("LED", None)):
        assert page.light_lacks() == "the light source LED has no on/off that smc knows"
    try:
        assert page.light_lacks() is None
    finally:
        del opener.opened[1].get


def test_a_status_light_read_across_a_reconnect_is_not_kept_for_the_new_stand(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tick's own read of the first stand's light outlasts a Disconnect and a
    # Connect (#295, `_note_light_read`): its answer belongs to a stand the page no
    # longer holds, and must not become the new stand's. A Disconnect waits for a
    # read in flight, so this is the case where the wait gave up (#305).
    monkeypatch.setattr(controller_module, "_READS_WAIT_S", 0.05)
    with _replaced(page, "_opener", _with_light(opener, Light("LED", None))):
        _connected(page)
    first = opener.opened[0]
    real_get = first.get

    class Across(Light):
        def is_on(self) -> bool | None:
            page.disconnect()
            assert page.join(5)
            with _replaced(page, "_opener", _with_light(opener, Light("Lamp", False))):
                _connected(page)
            return None

    def get(capability: type) -> Any:
        return (
            Across("LED", None) if capability is LightSource else real_get(capability)
        )

    with _replaced(first, "get", get):
        page.status()
    try:
        assert page.light_lacks() is None
    finally:
        del opener.opened[1].get


def test_a_labels_failure_is_logged_again_for_the_next_stand(
    page: PlatePage, opener: Opener, caplog: pytest.LogCaptureFixture
) -> None:
    # A failing streak is one line (FM-148); a Disconnect ends it, so a second stand
    # that fails the same way is news again.
    def failing() -> list[str]:
        raise CoreError("the turret did not answer")

    def lines() -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.name == "smc.ui"
            and "its positions could not be read" in r.getMessage()
        ]

    _connected(page)
    with (
        caplog.at_level("WARNING", logger="smc.ui"),
        _replaced(opener.opened[0].require(ObjectiveTurret), "labels", failing),
    ):
        assert page.objective_labels() == ()
        assert page.objective_labels() == ()
    assert len(lines()) == 1
    page.disconnect()
    assert page.join(5)
    _connected(page)
    with (
        caplog.at_level("WARNING", logger="smc.ui"),
        _replaced(opener.opened[1].require(ObjectiveTurret), "labels", failing),
    ):
        assert page.objective_labels() == ()
    assert len(lines()) == 2


# --- Disconnect -------------------------------------------------------------------


def test_disconnect_releases_the_stand_and_is_an_activity(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    page.disconnect()
    assert page.join(5)
    assert _outcome(page) == "disconnect: demo released"
    assert not page.connected
    assert _released(opener.opened[0])
    assert page.activities_ended == 2


def test_disconnect_drops_the_stand_before_closing_it_even_when_the_close_fails(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    held_during_close: list[bool] = []

    def failing() -> None:
        held_during_close.append(page.connected)
        raise RuntimeError("boom")

    with _replaced(opener.opened[0], "close", failing):
        page.disconnect()
        assert page.join(5)
    # No read that starts once the close has begun reaches a closing facade (§7.2).
    assert held_during_close == [False]
    # Any error but a busy one may have left a half-unloaded facade: let go for good.
    assert _outcome(page) == "disconnect: RuntimeError: boom"
    assert not page.connected


@pytest.mark.parametrize(
    ("fail", "error", "text"),
    [
        (_busy, MicroscopeBusyError, BUSY_TEXT),
        (_boom, RuntimeError, "boom"),
    ],
    ids=["busy", "other"],
)
def test_release_lets_go_of_the_stand_even_when_its_close_fails(
    page: PlatePage,
    opener: Opener,
    fail: Callable[[], None],
    error: type[Exception],
    text: str,
) -> None:
    # Unlike Disconnect, which keeps the facade of a busy close to retry: the
    # command is ending, and release() runs once.
    _connected(page)
    page.close()
    held_during_close: list[bool] = []

    def failing() -> None:
        held_during_close.append(page.connected)
        fail()

    with _replaced(opener.opened[0], "close", failing):
        with _refuses(error, text):
            page.release()
        assert not page.connected
        page.release()  # idempotent: the facade is not closed a second time
    assert held_during_close == [False]


@pytest.mark.parametrize("activity", ["run", "live"])
def test_disconnect_is_refused_while_a_run_or_live_goes(
    page: PlatePage, opener: Opener, activity: str
) -> None:
    _connected(page)
    try:
        if activity == "run":
            page.run("slow", [])
            assert Slow.started.wait(5)
            name = "slow"
        else:
            page.live(True)
            name = "live"
        with _refuses(RuntimeError, f"{name} is running"):
            page.disconnect()
        assert page.connected
        assert not _released(opener.opened[0])
    finally:
        Slow.release.set()
        page.live(False)
        assert page.join(5)


def test_disconnect_when_closed_is_an_outcome_and_starts_no_thread(
    page: PlatePage,
) -> None:
    page.disconnect()
    assert page.join(5)
    assert _outcome(page) == "disconnect: the stand is not connected"
    assert page.activities_ended == 0


def test_a_status_read_across_a_disconnect_is_the_closed_stands(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The wait for a read in flight gave up (#305): the read ends on a released stand.
    monkeypatch.setattr(controller_module, "_READS_WAIT_S", 0.05)
    _connected(page)
    microscope = opener.opened[0]
    real_state = microscope.state

    def across() -> Any:
        page.disconnect()
        assert page.join(5)
        # The released facade's read: XY and Z fail as "another program holding it".
        return real_state()

    with _replaced(microscope, "state", across):
        status = page.status()
    assert not status.connected
    assert status.xy_um is None
    assert status.problems == ()
    assert _released(microscope)


def test_the_last_frame_stays_after_a_disconnect(page: PlatePage) -> None:
    _connected(page)
    page.snap()
    assert page.join(5)
    taken = page.frame()
    assert isinstance(taken, Frame)
    page.disconnect()
    assert page.join(5)
    assert page.frame() is taken


# --- release ------------------------------------------------------------------------


def test_release_comes_after_close_and_is_idempotent(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    with _refuses(RuntimeError, "release() comes after close()"):
        page.release()
    assert not _released(opener.opened[0])
    page.close()
    page.release()
    assert _released(opener.opened[0])
    assert not page.connected
    page.release()


def test_release_lets_go_of_a_facade_handed_to_the_constructor(run_doc: Path) -> None:
    stand = _fake_stand()
    page = PlatePage(stand, run_doc, entry_points=ENTRY_POINTS)
    assert page.connected
    page.close()
    page.release()
    assert _released(stand)
    assert not page.connected


# --- reads in flight (#305, m5 §7.2) -----------------------------------------------


class _Holder:
    """A page's slots as ``_Reading`` sees them: it logs how many readers were in when the facade was taken."""

    def __init__(self) -> None:
        self._readers: set[object] = set()
        self.seen: list[int] = []

    @property
    def _microscope(self) -> None:
        self.seen.append(len(self._readers))
        return None


def test_a_reading_registers_before_it_takes_the_facade() -> None:
    # The order is the whole fix: a reader that took the facade before a drop must
    # be in the set when the Disconnect looks, which only holds if it registered
    # first. A fast fake never shows it, so the order is read here.
    holder = _Holder()
    reading = controller_module._Reading(cast("PlatePage", holder))
    with reading as microscope:
        assert microscope is None
        assert len(holder._readers) == 1
    assert holder.seen == [1]
    assert holder._readers == set()


def test_a_reading_unregisters_when_its_block_raises_and_never_swallows() -> None:
    holder = _Holder()
    with (
        pytest.raises(KeyError, match="boom"),
        controller_module._Reading(cast("PlatePage", holder)),
    ):
        raise KeyError("boom")
    assert holder._readers == set()


def _probe(
    page: PlatePage, seen: list[int], real: Callable[..., Any], raises: bool
) -> Callable[..., Any]:
    """A wrapper for a device call: how many reads are registered inside it, then the call or an error."""

    def probing(*args: Any, **kwargs: Any) -> Any:
        seen.append(len(page._readers))
        if raises:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    return probing


@pytest.mark.parametrize("raises", [False, True], ids=["answers", "raises"])
@pytest.mark.parametrize(
    ("method", "device_call"),
    [
        ("status", "state"),
        ("status", "is_on"),
        ("light_lacks", "is_on"),
        ("objective_labels", "labels"),
        ("capture_centre", "position_um"),
        ("capture_wall", "position_um"),
        ("stop", "stop"),
    ],
)
def test_every_read_of_the_page_registers_itself(
    page: PlatePage, opener: Opener, method: str, device_call: str, raises: bool
) -> None:
    # Registered at the call that reaches the device (the core's read), not at the
    # lookup of the capability before it: a read moved out of the `with` while its
    # lookup stays in must fail here, since that read is what an unload would meet.
    _connected(page)
    microscope = opener.opened[0]
    call: dict[str, Callable[[], object]] = {
        "status": page.status,
        "light_lacks": page.light_lacks,
        "objective_labels": page.objective_labels,
        "capture_centre": lambda: page.capture_centre("A1"),
        "capture_wall": lambda: page.capture_wall("A1", "left"),
        "stop": page.stop,
    }
    owners: dict[str, object] = {
        "state": microscope,
        "stop": microscope,
        "is_on": Light("LED", None),
        "labels": microscope.require(ObjectiveTurret),
        "position_um": microscope.require(XYStage),
    }
    owner = owners[device_call]
    seen: list[int] = []
    probe = _probe(page, seen, getattr(owner, device_call), raises)
    assert page._readers == set()
    real_get = microscope.get
    held: list[AbstractContextManager[Any]] = [_replaced(owner, device_call, probe)]
    if isinstance(owner, Light):
        # The stand's light is this stub, whose `is_on` is the probe.
        held.append(
            _replaced(
                microscope,
                "get",
                lambda capability: (
                    owner if capability is LightSource else real_get(capability)
                ),
            )
        )
    # A read that fails, as asked: what counts is that it let go.
    with suppress(RuntimeError), ExitStack() as stack:
        for context in held:
            stack.enter_context(context)
        call[method]()
    assert seen, f"{method} never reached the device's {device_call}"
    assert min(seen) >= 1
    assert page._readers == set()


def test_no_info_line_is_written_by_a_read_or_a_tick(
    page: PlatePage, opener: Opener, caplog: pytest.LogCaptureFixture
) -> None:
    # `smc ui` puts `smc.ui` at INFO (m5 §7.1): a line a tick wrote would fill the
    # service's log and block the loop that handles STOP (FM-148, FM-62).
    _connected(page)
    with caplog.at_level("INFO", logger="smc.ui"):
        caplog.clear()
        for _ in range(5):
            page.status()
        page.light_lacks()
        page.objective_labels()
        page.lacks(XYStage)
    assert [r.getMessage() for r in caplog.records if r.name == "smc.ui"] == []


def _eventually(condition: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    """Poll ``condition`` for at most ``timeout_s`` (FM-44); whether it came true."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class _HeldRead:
    """A device call held inside the facade by the test's own events, never by a sleep (FM-96)."""

    def __init__(self, real: Callable[..., Any]) -> None:
        self.real = real
        self.inside = threading.Event()
        self.go = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.inside.set()
        assert self.go.wait(10), "the test never let the read go"
        return self.real(*args, **kwargs)


def _watch_close(microscope: Microscope) -> threading.Event:
    """An event set the moment ``close()`` is reached, with the real close kept (put back by the caller)."""
    reached = threading.Event()
    real = microscope.close

    def closing() -> None:
        reached.set()
        real()

    microscope.close = closing  # type: ignore[method-assign]
    return reached


def _threaded(target: Callable[[], object]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize("way", ["disconnect", "release"])
def test_disconnect_and_release_wait_for_a_read_in_flight_before_they_unload(
    page: PlatePage, opener: Opener, way: str
) -> None:
    _connected(page)
    microscope = opener.opened[0]
    held = _HeldRead(microscope.state)
    statuses: list[Any] = []
    closing = _watch_close(microscope)
    microscope.state = held  # type: ignore[method-assign]
    try:
        reader = _threaded(lambda: statuses.append(page.status()))
        assert held.inside.wait(5)
        if way == "disconnect":
            page.disconnect()
            finisher = None
        else:
            page.close()
            finisher = _threaded(page.release)
        # Past the drop: from here a missing wait reaches close() within
        # milliseconds, so what follows cannot pass before it could have failed (FM-96).
        assert _eventually(lambda: not page.connected)
        assert not closing.wait(0.3)
        assert not _released(microscope)
        held.go.set()
        reader.join(5)
        if finisher is None:
            assert page.join(5)
        else:
            finisher.join(5)
            assert not finisher.is_alive()
    finally:
        held.go.set()
        del microscope.state
        del microscope.close
    assert _released(microscope)
    assert closing.is_set()
    if way == "disconnect":
        assert _outcome(page) == "disconnect: demo released"
    assert [s.connected for s in statuses] == [False]


def test_disconnect_drops_the_stand_before_it_looks_at_the_reads(
    page: PlatePage, opener: Opener
) -> None:
    # The other half of the order (``_Reading`` registers first): a Disconnect that
    # looked before it dropped would miss a reader that took the facade in between.
    _connected(page)
    held_when_looking: list[bool] = []
    real = page._wait_for_reads

    def looking() -> int:
        held_when_looking.append(page.connected)
        return real()

    with _replaced(page, "_wait_for_reads", looking):
        page.disconnect()
        assert page.join(5)
    assert held_when_looking == [False]


def test_disconnect_stops_waiting_for_a_read_after_the_bound_and_says_so(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_READS_WAIT_S", 0.2)
    _connected(page)
    microscope = opener.opened[0]
    held = _HeldRead(microscope.state)
    microscope.state = held  # type: ignore[method-assign]
    try:
        reader = _threaded(page.status)
        assert held.inside.wait(5)
        page.disconnect()
        assert page.join(5)
        assert _outcome(page) == (
            "disconnect: demo released, with 1 read(s) still in flight after 0.2 s"
        )
        assert _released(microscope)
    finally:
        held.go.set()
        del microscope.state
    reader.join(5)
    assert not reader.is_alive()


def test_a_stop_is_a_read_that_a_disconnect_waits_for(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    microscope = opener.opened[0]
    held = _HeldRead(microscope.stop)
    closing = _watch_close(microscope)
    microscope.stop = held  # type: ignore[method-assign]
    try:
        stopper = _threaded(page.stop)
        assert held.inside.wait(5)
        page.disconnect()
        assert _eventually(lambda: not page.connected)
        assert not closing.wait(0.3)
        assert not _released(microscope)  # the stop reaches loaded devices
        held.go.set()
        stopper.join(5)
        assert page.join(5)
    finally:
        held.go.set()
        del microscope.stop
        del microscope.close
    assert _released(microscope)
    assert _outcome(page) == "disconnect: demo released"
    assert page.status().stop_note == (
        "stopped: the stand is halted (Resume clears the halt)"
    )


def test_stop_takes_no_lock_the_worker_could_hold(page: PlatePage) -> None:
    # FM-70: STOP registers as a read and never waits for a lock the page holds.
    _connected(page)
    done = threading.Event()

    def stopping() -> None:
        page.stop()
        done.set()

    with page._lock:
        stopper = _threaded(stopping)
        assert done.wait(5)
    stopper.join(5)
    assert not stopper.is_alive()


def test_a_disconnect_whose_close_is_busy_keeps_the_stand_and_a_second_one_releases(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    microscope = opener.opened[0]
    real_close = microscope.close
    calls: list[int] = []

    def busy_once() -> None:
        calls.append(1)
        if len(calls) == 1:
            _busy()
        real_close()

    with _replaced(microscope, "close", busy_once):
        page.disconnect()
        assert page.join(5)
        assert _outcome(page) == (
            f"disconnect: MicroscopeBusyError: {BUSY_TEXT}; the stand is still connected"
        )
        assert page.connected
        assert page._microscope is microscope  # the same facade, to retry on
        assert not _released(microscope)
        page.disconnect()
        assert page.join(5)
    assert _outcome(page) == "disconnect: demo released"
    assert not page.connected
    assert _released(microscope)
    assert len(calls) == 2


def test_a_busy_close_does_not_put_the_facade_back_on_a_closed_page(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The command's release() has looked by then; a facade put back would have no owner.
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 0.2)
    _connected(page)
    microscope = opener.opened[0]
    entered, gate = threading.Event(), threading.Event()

    def busy_after_the_gate() -> None:
        entered.set()
        gate.wait(10)
        _busy()

    with _replaced(microscope, "close", busy_after_the_gate):
        try:
            page.disconnect()
            assert entered.wait(5)
            page.close()
            assert not page.close_ended
        finally:
            gate.set()
        assert page.join(5)
    assert not page.connected
    page.release()  # finds nothing to release
    microscope.close()  # the real one, for the fake's sake


def test_a_disconnect_that_could_not_unload_says_the_stand_may_still_be_held(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    microscope = opener.opened[0]

    def no_answer() -> None:
        raise RuntimeError("unload: no answer")

    with _replaced(microscope.core, "unloadAllDevices", no_answer):
        page.disconnect()
        assert page.join(5)
    assert _outcome(page) == (
        "disconnect: demo may still be held: its devices did not unload "
        "(RuntimeError('unload: no answer')); restart smc ui before another "
        "program takes it"
    )
    assert not page.connected


@pytest.mark.parametrize(
    "order", ["ended before release", "ended while release waits", "never ends"]
)
def test_release_waits_for_a_connect_close_gave_up_on_and_says_what_became_of_it(
    run_doc: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 0.2)
    # A connect that ends while release waits is not made to race the wait (FM-96).
    monkeypatch.setattr(
        controller_module, "_RELEASE_WAIT_S", 0.2 if order == "never ends" else 5.0
    )
    gate = threading.Event()
    opener = Opener(gate=gate)
    page = _closed(run_doc, opener)
    try:
        page.connect()
        assert opener.entered.wait(5)
        page.close()
        assert not page.close_ended
        if order == "ended before release":
            gate.set()
            assert page.join(5)
            page.release()
        elif order == "ended while release waits":
            opening = threading.Timer(0.05, gate.set)
            opening.start()
            page.release()
            opening.join(5)
        else:
            page.release()
    finally:
        gate.set()
    assert page.join(5)
    if order == "never ends":
        assert page.close_lines[-1] == (
            "connect: demo is still opening after 0.4 s: the process ends now, and "
            "whether the stand is left connected is not known"
        )
        assert not page.close_ended
    else:
        assert page.close_lines[-1] == (
            "connect: demo was opened after the page closed, and released"
        )
        assert page.close_ended
    # Whichever the order, the facade the connect opened is not left held: the late
    # release of the worker cleans up after a release() that did not wait for it.
    assert _released(opener.opened[0])


def test_close_does_not_overwrite_the_outcome_of_a_connect_that_just_ended(
    run_doc: Path,
) -> None:
    # The connect ends in the instant between the end of close()'s wait and the
    # line it writes: its own outcome is what release() must tell, never "did not end".
    gate = threading.Event()
    opener = Opener(gate=gate)
    page = _closed(run_doc, opener)
    page.connect()
    assert opener.entered.wait(5)
    real_join = page.join

    def ends_after_the_wait(timeout_s: float | None = None) -> bool:
        gate.set()
        assert real_join(5)
        return False  # close() has already given up on it

    with _replaced(page, "join", ends_after_the_wait):
        page.close()
    assert _outcome(page) == (
        "connect: demo was opened after the page closed, and released"
    )
    page.release()
    assert page.close_lines[-1] == (
        "connect: demo was opened after the page closed, and released"
    )
    assert page.close_ended
    assert _released(opener.opened[0])


@pytest.mark.wallclock
def test_release_does_not_wait_for_an_activity_other_than_a_connect(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 0.2)
    monkeypatch.setattr(controller_module, "_RELEASE_WAIT_S", 5.0)
    _connected(page)
    page.run("slow", [])
    assert Slow.started.wait(5)
    # A stopped run that ignores its stop: close() gives up on it.
    with _replaced(page, "stop", lambda: None):
        page.close()
    assert not page.close_ended
    lines = page.close_lines
    started = time.monotonic()
    page.release()
    assert time.monotonic() - started < 2.0  # no 5 s wait: only a connect is waited for
    assert page.close_lines == lines
    assert not page.close_ended
    assert _released(opener.opened[0])


def test_release_ends_the_facade_through_its_exit_with_the_exception_it_is_given(
    page: PlatePage, opener: Opener
) -> None:
    # A second Ctrl-C during close() lost the note of the motions still tracked when
    # release() called close() directly; the facade's own exit writes it (m2 §11.11 E6).
    _connected(page)
    microscope = opener.opened[0]
    exc = KeyboardInterrupt()
    with _replaced(microscope._executor, "moving", lambda: ("xy",)):
        page.close()
        page.release(exc)
    assert [(n.motion, n.outcome) for n in stop_notes(exc)] == [
        ("xy", StopOutcome.still_moving)
    ]
    assert _released(microscope)


@pytest.mark.parametrize("wait", ["the reads", "the connect"])
def test_a_ctrl_c_in_a_wait_of_release_still_unloads_the_stand(
    page: PlatePage, opener: Opener, wait: str
) -> None:
    # A second Ctrl-C lands in a wait of `release()`: the facade, dropped from the page
    # and held by a local only, was never unloaded (FM-71, FM-49). The Ctrl-C itself is
    # not swallowed. After the reads' wait the page has let go of the facade already;
    # a Ctrl-C in the connect's wait comes before the drop, and the page still holds it.
    _connected(page)
    microscope = opener.opened[0]
    page.close()

    def interrupted(*args: object) -> None:
        raise KeyboardInterrupt

    if wait == "the connect":
        page._gave_up = "connect"  # as if close() had given up on a connect
    name = "_wait_for_reads" if wait == "the reads" else "_wait_for_the_connect"
    with _replaced(page, name, interrupted), pytest.raises(KeyboardInterrupt):
        page.release()
    assert _released(microscope)
    assert not page.connected
    page.release()  # idempotent after it


def test_stop_keeps_its_note_before_it_logs_it(
    page: PlatePage, caplog: pytest.LogCaptureFixture
) -> None:
    # `smc ui` puts `smc.ui` at INFO, so the line reaches a console: a handler that
    # blocks there (Windows QuickEdit) must not keep the page from its note (FM-148).
    _connected(page)
    seen: list[str | None] = []

    class Spy(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage().startswith("stop: "):
                seen.append(page.status().stop_note)

    logger = logging.getLogger("smc.ui")
    spy = Spy()
    logger.addHandler(spy)
    try:
        with caplog.at_level("INFO", logger="smc.ui"):
            page.stop()
    finally:
        logger.removeHandler(spy)
    assert seen == ["stopped: the stand is halted (Resume clears the halt)"]


@pytest.mark.wallclock
def test_a_read_that_never_ends_does_not_slow_the_next_disconnect(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The token of a read stuck in a device call belongs to the facade that was
    # released: kept, it made every later Disconnect wait the bound again and report a
    # read that no stand had.
    monkeypatch.setattr(controller_module, "_READS_WAIT_S", 0.2)
    _connected(page)
    first = opener.opened[0]
    held = _HeldRead(first.state)
    first.state = held  # type: ignore[method-assign]
    try:
        reader = _threaded(page.status)
        assert held.inside.wait(5)
        page.disconnect()
        assert page.join(5)
        assert _outcome(page) == (
            "disconnect: demo released, with 1 read(s) still in flight after 0.2 s"
        )
        assert reader.is_alive()  # the read is still stuck, and its facade is gone
        assert page._readers == set()
        _connected(page)
        started = time.monotonic()
        page.disconnect()
        assert page.join(5)
        assert _outcome(page) == "disconnect: demo released"
        assert time.monotonic() - started < 2.0
    finally:
        held.go.set()
        del first.state
    reader.join(5)
    assert not reader.is_alive()
    assert page._readers == set()


def test_a_busy_close_keeps_the_token_of_a_read_that_never_ends(
    page: PlatePage, opener: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The page gets the facade back on a busy close, with the stuck read still inside
    # it: the retry must wait for it, and say so, as the first Disconnect did.
    monkeypatch.setattr(controller_module, "_READS_WAIT_S", 0.2)
    _connected(page)
    microscope = opener.opened[0]
    held = _HeldRead(microscope.state)
    real_close = microscope.close
    calls: list[int] = []

    def busy_once() -> None:
        calls.append(1)
        if len(calls) == 1:
            _busy()
        real_close()

    microscope.state = held  # type: ignore[method-assign]
    try:
        with _replaced(microscope, "close", busy_once):
            reader = _threaded(page.status)
            assert held.inside.wait(5)
            page.disconnect()
            assert page.join(5)
            assert page.connected
            assert len(page._readers) == 1
            page.disconnect()
            assert page.join(5)
            assert _outcome(page) == (
                "disconnect: demo released, with 1 read(s) still in flight after 0.2 s"
            )
    finally:
        held.go.set()
        del microscope.state
    reader.join(5)
    assert not reader.is_alive()


def test_release_adds_no_line_for_a_connect_that_ended_inside_closes_wait(
    run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The usual Ctrl-C during a slow Connect: it ends while close() still waits. Only
    # an activity close() gave up on is waited for again, and told once.
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 5.0)
    gate = threading.Event()
    opener = Opener(gate=gate)
    page = _closed(run_doc, opener)
    page.connect()
    assert opener.entered.wait(5)
    opening = threading.Timer(0.05, gate.set)
    opening.start()
    try:
        page.close()
    finally:
        gate.set()
        opening.join(5)
    assert page.close_ended
    lines = page.close_lines
    page.release()
    assert page.close_lines == lines
    assert page.close_ended
    assert _released(opener.opened[0])
