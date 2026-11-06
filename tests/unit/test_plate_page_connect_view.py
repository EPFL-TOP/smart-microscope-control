"""The plate page's *Connect* row, without a browser (m5 §7.3, #264).

``button.clicks += 1`` is what a click raises, as in ``test_plate_page.py``; a
click on a *disabled* Panel button still calls its handler on the server
(measured, m5 §6.4), so the controller's refusal is the guard and a test clicks a
disabled button to prove it (FM-169). Whether a click works in a real Bokeh
session (FM-161) is ``/ui-check``'s, not this module's. Panel's deprecation
warnings are errors here (FM-147).
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel  # noqa: F401  # CI must fail, not skip, when the extra is missing
else:
    pytest.importorskip("panel")

from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
import smc.ui.plate_page as plate_page_module
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, ObjectiveTurret
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, PlatePage
from smc.ui.plate_page import NO_LABEL, PlatePageView

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

CLOSED_TEXT = (
    "demo is not connected: Connect opens it, and the controls that need the "
    "stand are disabled until then"
)
CONNECTED_TEXT = "Disconnect releases demo for another program (NIS-Elements, ZEN)"


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


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()


@pytest.fixture
def opened() -> Iterator[list[Microscope]]:
    """Every fake stand the page's opener opened; each closed after the test."""
    found: list[Microscope] = []
    try:
        yield found
    finally:
        for microscope in found:
            microscope.close()


def _page(
    run_doc: Path, opened: list[Microscope], *, dry_run: bool = False
) -> PlatePage:
    def opener() -> Microscope:
        microscope = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()), Profile.demo(), dry_run=dry_run
        )
        opened.append(microscope)
        return microscope

    slow = EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP)
    return PlatePage.closed(
        Profile.demo(),
        run_doc,
        dry_run=dry_run,
        opener=opener,
        entry_points=[slow],
    )


@pytest.fixture
def page(tmp_path: Path, opened: list[Microscope]) -> Iterator[PlatePage]:
    controller = _page(tmp_path / "run.json", opened)
    try:
        yield controller
    finally:
        controller.close()
        controller.release()


def _drive_controls(view: PlatePageView) -> list[Any]:
    return [
        *view.jog_buttons.values(),
        *view.z_buttons.values(),
        view.autofocus_on_button,
        view.autofocus_off_button,
        view.objective_button,
        view.light_on_button,
        view.light_off_button,
    ]


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _connect(view: PlatePageView) -> None:
    view.connect_button.clicks += 1
    assert view.page.join(5)
    view.tick()


def test_a_closed_stand_disables_what_needs_it_and_says_why_once(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    assert not view.connect_button.disabled
    assert view.disconnect_button.disabled
    assert view.connection_text.object == CLOSED_TEXT
    assert view.capture_centre_button.disabled
    assert view.capture_wall_button.disabled
    assert view.snap_button.disabled
    assert all(control.disabled for control in _drive_controls(view))
    assert view.drive_note.object == "the stand is not connected"
    assert view.status_text.object == "demo | not connected"
    assert view.tool_runs["slow"].disabled
    assert view.tool_notes["slow"].object == NOT_CONNECTED


def test_a_dry_run_page_says_so_on_its_connect_row(
    tmp_path: Path, opened: list[Microscope]
) -> None:
    page = _page(tmp_path / "run.json", opened, dry_run=True)
    view = PlatePageView(page)
    assert view.connection_text.object == f"{CLOSED_TEXT} (dry run)"
    _connect(view)
    assert view.connection_text.object == f"{CONNECTED_TEXT} (dry run)"
    page.close()
    page.release()


def test_connect_through_a_click_enables_the_controls_and_reads_the_objectives(
    page: PlatePage, opened: list[Microscope]
) -> None:
    view = PlatePageView(page)
    assert view.objective_select.options == {}
    _connect(view)
    [stand] = opened
    labels = stand.require(ObjectiveTurret).labels()
    assert len(labels) > 1
    assert list(view.objective_select.options.values()) == labels
    assert list(view.objective_select.options) == [
        label or NO_LABEL for label in labels
    ]
    assert view.objective_select.value == page.status().objective
    # The pad, the focus and the objective: what the demo stand has. Its light and
    # its autofocus are its own lacks, one line each, as before #264.
    held = [
        *view.jog_buttons.values(),
        *view.z_buttons.values(),
        view.objective_button,
    ]
    assert not any(control.disabled for control in held)
    assert NOT_CONNECTED not in view.drive_note.object
    assert view.drive_note.object == "\n".join(
        f"{name}: {reason}"
        for name, reason in (
            ("autofocus", page.lacks(Autofocus)),
            ("light", page.light_lacks()),
        )
        if reason
    )
    assert view.connect_button.disabled
    assert not view.disconnect_button.disabled
    assert view.connection_text.object == CONNECTED_TEXT
    # Connected, but no well is selected: the captures wait for one (m5 §11.1), and
    # are usable as soon as a click on the map names it.
    assert view.capture_centre_button.disabled
    assert view.capture_wall_button.disabled
    assert page.click(0.0, 0.0, drive=False) == "A1"
    view.tick()
    assert not view.capture_centre_button.disabled
    assert not view.capture_wall_button.disabled
    assert view.status_text.object.startswith("demo | XY (")


def test_disconnect_through_a_click_closes_the_page_again(
    page: PlatePage, opened: list[Microscope]
) -> None:
    view = PlatePageView(page)
    _connect(view)
    view.disconnect_button.clicks += 1
    assert page.join(5)
    view.tick()
    assert not page.connected
    assert view.connection_text.object == CLOSED_TEXT
    assert view.objective_select.options == {}
    assert view.drive_note.object == "the stand is not connected"
    assert list(opened[0].core.getLoadedDevices()) == ["Core"]


def test_the_objective_select_follows_the_turret_of_a_second_connection(
    page: PlatePage,
) -> None:
    # `_offer_objectives` starts the once-only sync over for the stand now held: the
    # first connection's sync has been done, and the second stand's turret is where
    # *its* objective is, not where the select was left (#295).
    view = PlatePageView(page)
    _connect(view)
    assert view.objective_select.value == "Nikon 10X S Fluor"
    view.objective_select.value = "Nikon 40X Plan Fluor ELWD"
    view.disconnect_button.clicks += 1
    assert page.join(5)
    view.tick()
    assert view.objective_select.options == {}

    _connect(view)

    assert view.objective_select.value == "Nikon 10X S Fluor"


def test_connect_is_disabled_while_a_connect_goes(
    tmp_path: Path, opened: list[Microscope]
) -> None:
    gate = threading.Event()
    entered = threading.Event()
    page = _page(tmp_path / "run.json", opened)
    real_opener = page._opener

    def slow() -> Microscope:
        entered.set()
        assert gate.wait(5)
        return real_opener()

    page._opener = slow
    view = PlatePageView(page)
    try:
        view.connect_button.clicks += 1
        assert entered.wait(5)
        view.tick()
        assert view.connect_button.disabled
        assert view.disconnect_button.disabled
    finally:
        gate.set()
        assert page.join(5)
        page.close()
        page.release()


@pytest.mark.parametrize("dry_run", [False, True], ids=["live-stand", "dry-run"])
def test_the_connection_row_says_connecting_and_disconnecting_while_the_activity_goes(
    tmp_path: Path, opened: list[Microscope], dry_run: bool
) -> None:
    # On a slow stand the row read `Connect opens it` beside a greyed-out button, and
    # the page looked stuck (#305, m5 §7.3). Texts whole (FM-45), the ellipsis U+2026.
    suffix = " (dry run)" if dry_run else ""
    gate, entered = threading.Event(), threading.Event()
    page = _page(tmp_path / "run.json", opened, dry_run=dry_run)
    real_opener = page._opener

    def slow() -> Microscope:
        entered.set()
        assert gate.wait(5)
        return real_opener()

    view = PlatePageView(page)
    try:
        with _replaced(page, "_opener", slow):
            view.connect_button.clicks += 1
            assert entered.wait(5)
            view.tick()
            assert (
                view.connection_text.object
                == f"connecting demo\N{HORIZONTAL ELLIPSIS}{suffix}"
            )
            assert view.connect_button.disabled
            assert view.disconnect_button.disabled
            gate.set()
            assert page.join(5)
        view.tick()
        assert view.connection_text.object == f"{CONNECTED_TEXT}{suffix}"

        closing, held = threading.Event(), threading.Event()
        stand = opened[0]
        real_close = stand.close

        def slow_close() -> None:
            closing.set()
            assert held.wait(5)
            real_close()

        with _replaced(stand, "close", slow_close):
            view.disconnect_button.clicks += 1
            assert closing.wait(5)
            view.tick()
            assert view.connection_text.object == (
                f"disconnecting demo\N{HORIZONTAL ELLIPSIS}{suffix}"
            )
            assert view.connect_button.disabled
            assert view.disconnect_button.disabled
            held.set()
            assert page.join(5)
        view.tick()
        assert view.connection_text.object == f"{CLOSED_TEXT}{suffix}"
    finally:
        gate.set()
        page.close()
        page.release()


def test_the_capture_buttons_are_disabled_while_disconnect_runs(
    page: PlatePage, opened: list[Microscope]
) -> None:
    # While a Disconnect closes the stand the facade is already dropped, so `not
    # connected` alone switches the buttons off: this does not pin #317's rule for an
    # activity (its test does). What it adds is the controller's own refusal, which a
    # click on a disabled button still reaches (FM-169).
    view = PlatePageView(page)
    _connect(view)
    page.current = "A1"
    view.tick()
    assert not view.capture_centre_button.disabled
    closing, held = threading.Event(), threading.Event()
    stand = opened[0]
    real_close = stand.close

    def slow_close() -> None:
        closing.set()
        assert held.wait(5)
        real_close()

    try:
        with _replaced(stand, "close", slow_close):
            view.disconnect_button.clicks += 1
            assert closing.wait(5)
            view.tick()
            assert view.capture_centre_button.disabled
            assert view.capture_wall_button.disabled
            # Panel still calls the handler of a disabled button: the controller
            # refuses, and says why (FM-169).
            view.capture_centre_button.clicks += 1
            assert view.message.object == "disconnect is running"
            assert page.refs == ()
            held.set()
            assert page.join(5)
    finally:
        held.set()
    assert not page.connected


def test_the_stop_budget_in_the_service_doc_covers_the_pages_waits() -> None:
    # `close()`'s wait, the server's stop and one wait of `release()` run one after the
    # other inside NSSM's console stop (m5 §7.4): if one of them grows past the number
    # the document gives NSSM, the service is ended by `TerminateProcess`, which
    # releases nothing. `release()` pays one of its two waits: for a connect that
    # `close()` gave up on, which never keeps a facade, or for the reads on a facade
    # the page holds. So the longer of them counts, and a change to either is seen.
    doc = Path(__file__).resolve().parents[2] / "docs" / "ui" / "service.md"
    found = re.search(r"AppStopMethodConsole (\d+)", doc.read_text(encoding="utf-8"))
    assert found is not None, "docs/ui/service.md no longer sets AppStopMethodConsole"
    budget_s = int(found.group(1)) / 1000
    waits_s = (
        controller_module._CLOSE_WAIT_S
        + plate_page_module.SERVER_STOP_WAIT_S
        + max(controller_module._RELEASE_WAIT_S, controller_module._READS_WAIT_S)
    )
    assert waits_s <= budget_s, (waits_s, budget_s)


def test_a_confirmation_armed_for_one_stand_is_dropped_when_the_connection_changes(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    _connect(view)
    view.objective_select.value = list(view.objective_select.options.values())[1]
    view.on_objective()
    assert view.objective_confirm.visible
    # Disconnect and Connect between two ticks: the drive controls never read as
    # disabled, so only the change of connection can drop the confirmation.
    page.disconnect()
    assert page.join(5)
    page.connect()
    assert page.join(5)
    view._seen_connected = False
    view.tick()
    assert not view.objective_confirm.visible
    assert view._armed is None


@pytest.mark.parametrize("activity", ["run", "live"])
def test_disconnect_is_disabled_while_an_activity_goes(
    page: PlatePage, activity: str
) -> None:
    view = PlatePageView(page)
    _connect(view)
    try:
        if activity == "run":
            page.run("slow", [])
            assert Slow.started.wait(5)
        else:
            page.live(True)
        view.tick()
        assert view.disconnect_button.disabled
        assert view.connect_button.disabled
    finally:
        Slow.release.set()
        page.live(False)
        assert page.join(5)
    view.tick()
    assert not view.disconnect_button.disabled


def test_a_click_on_a_disabled_disconnect_is_refused_by_the_controller(
    page: PlatePage, opened: list[Microscope]
) -> None:
    view = PlatePageView(page)
    _connect(view)
    try:
        page.run("slow", [])
        assert Slow.started.wait(5)
        view.tick()
        assert view.disconnect_button.disabled
        view.disconnect_button.clicks += 1
        assert view.message.object == "slow is running"
        assert page.connected
        assert list(opened[0].core.getLoadedDevices()) != ["Core"]
    finally:
        Slow.release.set()
        assert page.join(5)
