"""The plate page with the stand closed: Connect, Disconnect and release (m5 §7.2, #264).

No Panel here. The stand is a facade over ``FakeCore.demo_like()``, opened by an
opener the test hands the page (measured twice in one process, m5 §7.5); a
release is seen as a core that lists only ``Core``. Texts are asserted whole
(FM-45); threads meet on ``Event`` and every wait is bounded and asserted
(FM-44); no test pins an order that the fake's speed decides (FM-96).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource, ObjectiveTurret, XYStage
from smc.hardware.errors import CoreError, MicroscopeBusyError
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
    page: PlatePage, opener: Opener
) -> None:
    # A read of the first stand's light that outlasts a Disconnect and a Connect: its
    # answer belongs to a stand the page no longer holds (m5 §7.2).
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
    page: PlatePage, opener: Opener
) -> None:
    # The tick's own read of the first stand's light outlasts a Disconnect and a
    # Connect (#295, `_note_light_read`): its answer belongs to a stand the page no
    # longer holds, and must not become the new stand's.
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
        _busy()

    with _replaced(opener.opened[0], "close", failing):
        page.disconnect()
        assert page.join(5)
    # No read that starts once the close has begun reaches a closing facade (§7.2).
    assert held_during_close == [False]
    assert _outcome(page) == f"disconnect: MicroscopeBusyError: {BUSY_TEXT}"
    assert not page.connected


def test_release_lets_go_of_the_stand_even_when_its_close_fails(
    page: PlatePage, opener: Opener
) -> None:
    _connected(page)
    page.close()
    held_during_close: list[bool] = []

    def failing() -> None:
        held_during_close.append(page.connected)
        _busy()

    with _replaced(opener.opened[0], "close", failing):
        with _refuses(MicroscopeBusyError, BUSY_TEXT):
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
    page: PlatePage, opener: Opener
) -> None:
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
