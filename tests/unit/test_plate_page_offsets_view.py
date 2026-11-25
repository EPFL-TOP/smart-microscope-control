"""The offsets block of the plate page, without a browser (m5 §18, #382).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1``, a
typed field is its ``.value`` (a ``TextAreaInput`` sends its value on blur or Ctrl-Enter, not at
each key). A click on a *disabled* button still calls its handler on the server (measured on
Panel 1.10.0 and 1.9.4), so the controller's refusals are the guard (FM-169) and a test clicks
a disabled button to prove it. What this cannot show is a browser: the layout, and whether a
newcomer finds the two clicks, are for the session of ``/ui-check`` (FM-161). The tool is a
stub that records what it was given and turns nothing; the controller's tests
(``test_plate_page_offsets.py``) hold the comparison itself.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import threading
from collections.abc import Callable, Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from test_plate_page_offsets import (
    DEMO_PROFILE,
    FORTY,
    TEN,
    TWENTY,
    Slow,
    StubOffsetsParams,
    _document,
    _dry_microscope,
    _plan_page,
    _result,
    _seed_offsets,
    _seed_plate,
    _typed,
    _walls_plate,
)

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile
from smc.plugins.api import RunContext
from smc.plugins.registry import GROUP
from smc.plugins.slots import ObjectiveOffsetsResult
from smc.run import RunDocument, StageStatus
from smc.run.files import AtomicWriteError
from smc.testing import FakeCore
from smc.ui import plate_page
from smc.ui.controller import PlatePage
from smc.ui.guide import (
    CALIBRATION_HOW_WINDOWS_NOTE,
    OFFSETS_DRY_RUN,
    OFFSETS_NO_DOCUMENT,
    OFFSETS_NOT_TURNED_BACK,
    OFFSETS_NOTHING_ARMED,
    OFFSETS_PLAN_CHANGED,
    OFFSETS_STALE,
    busy_note,
    offsets_confirm_text,
    offsets_lead,
    offsets_written_line,
)
from smc.ui.plate_page import PlatePageView, TextPane

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]


class Measuring:
    """``objective-offsets`` without a stage: records its parameters, reports once, and waits while ``hold`` is set."""

    name = provides = "objective-offsets"
    summary = "A test tool."
    Params = StubOffsetsParams
    Result = ObjectiveOffsetsResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[StubOffsetsParams]] = []
    hold: ClassVar[bool] = False
    started: ClassVar[threading.Event] = threading.Event()
    reported: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(
        self, microscope: Any, params: StubOffsetsParams, ctx: RunContext
    ) -> ObjectiveOffsetsResult:
        Measuring.seen.append(params)
        Measuring.started.set()
        if Measuring.hold:
            # The first report comes after the turn and the walk to the first well.
            while not Measuring.reported.is_set() and not ctx.should_stop():
                ctx.wait_s(0.02)
            ctx.progress(3, 24, f"{params.targets[0]}: A12 -Y")
            while Measuring.hold and not Measuring.release.is_set():
                ctx.wait_s(0.02)
        return _result(*((label, (1.0, 2.0)) for label in params.targets))


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    Measuring.seen = []
    Measuring.hold = False
    Measuring.started = threading.Event()
    Measuring.reported = threading.Event()
    Measuring.release = threading.Event()
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Measuring.release.set()
        Measuring.reported.set()
        Slow.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_view(
    fake_microscope: Microscope, run_doc: Path
) -> Iterator[Callable[..., PlatePageView]]:
    """A view over the fake stand (or the one given) with the stub tool and ``Slow``; every page it built is closed."""
    pages: list[PlatePage] = []

    def make(microscope: Microscope | None = None) -> PlatePageView:
        page = PlatePage(
            microscope or fake_microscope,
            run_doc,
            entry_points=[
                EntryPoint(
                    name="objective-offsets", value=f"{__name__}:Measuring", group=GROUP
                ),
                EntryPoint(
                    name="slow", value="test_plate_page_offsets:Slow", group=GROUP
                ),
            ],
        )
        pages.append(page)
        return PlatePageView(page)

    try:
        yield make
    finally:
        Measuring.release.set()
        Measuring.reported.set()
        Slow.release.set()
        for page in pages:
            page.close()


def _click(button: Any) -> None:
    button.clicks += 1


def _objects(items: list[Any]) -> list[Any]:
    """Every object under ``items``, the Accordion's panels too."""
    found: list[Any] = []
    for item in items:
        found.extend(item.select())
        for accordion in item.select(pn.Accordion):
            for panel in accordion.objects:
                found.extend(panel.select())
    return found


def _ready(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_calibration: Callable[..., Any],
) -> tuple[PlatePageView, str]:
    """A view on a plate registered from its walls with the 10X, and two calibration documents (40X then 20X) typed in its field."""
    _seed_plate(run_doc)
    forty = _document(tmp_path, FORTY, seed_calibration)
    twenty = _document(tmp_path, TWENTY, seed_calibration)
    view = make_view()
    view.show_step("plate")
    view.refresh()
    view.tick()
    text = _typed(forty, twenty)
    view.calibrations_input.value = text
    return view, text


# --- where the block is, and when -----------------------------------------------------


def test_the_block_is_hidden_until_the_plate_is_registered_from_its_walls(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    view = make_view()
    view.show_step("plate")
    # No plate at all, then one registered by eye, then one with no objective: no block, and
    # none of its panes says anything a screen's count of texts would meet.
    for plate in (
        None,
        _walls_plate(method="references", with_walls=False, walls_objective=None),
        _walls_plate(objective=None, walls_objective=None),
    ):
        if plate is not None:
            _seed_plate(run_doc, plate, plugin="plate-registration")
        view.refresh()
        view.tick()
        assert view.offsets_box.visible is False
        assert view.page.view().offsets is None
        for pane in (
            view.offsets_lead_text,
            view.objective_in_place_text,
            view.offsets_wait_text,
            view.offsets_result_text,
            view.offsets_plan_text,
        ):
            assert pane.object == ""
    _seed_plate(run_doc, _walls_plate(), minute=1)
    view.refresh()
    view.tick()
    assert view.offsets_box.visible is True
    assert view.offsets_box.collapsed is False
    assert view.offsets_box.title == "Measure the offsets between objectives"
    labels = [label for label in view._objective_labels if label]
    assert view.offsets_lead_text.object == offsets_lead(
        TEN, [label for label in labels if label != TEN]
    )


def test_the_block_is_on_the_plate_screen_and_in_the_expert_view_and_drawn_once(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    view = make_view()
    box = view.offsets_box

    for step in ("stand", "wells", "scan", "results"):
        view.show_step(step)  # type: ignore[arg-type]
        assert box not in _objects(list(view.step_panel.objects))

    view.show_step("plate")
    plate = list(view.step_panel.objects)
    assert [o for o in _objects(plate) if o is box] == [box]
    # After the way to register and before *Continue*.
    assert plate.index(box) == plate.index(view.continue_button) - 1
    assert plate[plate.index(box) - 1] is view.other_ways_text

    view.show_expert(True)
    expert = list(view.step_panel.objects)
    assert [o for o in _objects(expert) if o is box] == [box]
    # After the next-step region, ahead of the plate's controls.
    assert expert.index(box) == 2
    assert view.next_text in _objects([expert[1]])
    # One instance of each of the block's widgets in the layout (FM-174: containers, no figure).
    for widget in (
        view.offsets_measure_button,
        view.offsets_turn_button,
        view.calibrations_input,
        view.offsets_write_button,
        view.offsets_session_input,
    ):
        assert _objects(expert).count(widget) == 1
    assert not any(isinstance(o, plate_page.figure) for o in box.select())
    # Its how-to is closed, and its labels are the design's.
    assert view.calibration_how_card.collapsed is True
    assert view.calibrations_input.rows == 3
    assert view.calibrations_input.label == (
        "Other calibration documents (optional), one path per line, in the order the "
        "turret turns"
    )
    assert view.calibrations_input.placeholder == (
        "leave empty to measure with the calibrations listed above"
    )
    assert view.calibration_how_card.title == "Or make a document in a terminal"
    assert view.offsets_session_input.label == (
        "Session (goes into each offset's measured text; optional)"
    )
    assert view.offsets_session_input.placeholder == "session 3 (#NNN)"
    assert view.offsets_measure_button.label == "Measure the offsets"
    assert view.offsets_turn_button.label == "Turn and measure"
    assert view.offsets_cancel_button.label == "Cancel"


# --- the two clicks (D4) --------------------------------------------------------------


def test_measure_arms_the_confirmation_and_turns_nothing(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    fake_microscope: Microscope,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, _text = _ready(make_view, run_doc, tmp_path, seed_camera_calibration)
    before = run_doc.read_bytes()
    assert view.offsets_confirm_box.visible is False

    _click(view.offsets_measure_button)

    assert view._offsets_armed == (FORTY, TWENTY)
    assert view.offsets_confirm_box.visible is True
    assert view.offsets_confirm_text.object == offsets_confirm_text(
        TEN, (FORTY, TWENTY)
    )
    assert view.offsets_confirm_text.visible is True
    # Nothing turned and nothing ran: the turret still reads the 10X, the document is as it was.
    assert fake_microscope.require(ObjectiveTurret).current() == TEN
    assert Measuring.seen == []
    assert run_doc.read_bytes() == before
    # One line under the field for each document, in the tool's own words.
    assert view.calibration_reads_text.object.splitlines() == [
        f"{path}: calibrated at {objective} on 2026-11-09"
        for path, objective in zip(
            view.calibrations_input.value.splitlines(), (FORTY, TWENTY), strict=True
        )
    ]

    _click(view.offsets_turn_button)

    assert view.page.join(10.0)
    assert [(p.targets, p.confirm_turns) for p in Measuring.seen] == [
        ((FORTY, TWENTY), True)
    ]
    # Disarmed by the click, whatever it came to.
    assert view._offsets_armed is None
    assert view.offsets_confirm_box.visible is False


def test_a_click_on_turn_and_measure_with_nothing_armed_turns_nothing(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, _text = _ready(make_view, run_doc, tmp_path, seed_camera_calibration)

    # Idle, with nothing armed: the controller says to choose the documents again.
    _click(view.offsets_turn_button)
    outcome = view.page.status().outcome
    assert outcome == f"Objective offsets: {OFFSETS_NOTHING_ARMED}"

    # An activity going disables the button; the click still arrives (FM-169) and the
    # controller refuses it: nothing started, the message says what runs.
    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()
    assert view.offsets_turn_button.disabled is True
    _click(view.offsets_turn_button)
    assert view.message.object == "slow is running"

    # Nothing of the tool ran, and the document holds no record of it: `slow` wrote its own.
    assert Measuring.seen == []
    assert RunDocument.load_or_new(run_doc).provider("objective-offsets") is None
    Slow.release.set()
    assert view.page.join(10.0)


def test_cancel_an_edit_of_the_field_and_a_busy_page_each_drop_the_confirmation(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, text = _ready(make_view, run_doc, tmp_path, seed_camera_calibration)

    _click(view.offsets_measure_button)
    assert view._offsets_armed is not None
    _click(view.offsets_cancel_button)
    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)

    _click(view.offsets_measure_button)
    assert view._offsets_armed is not None
    # An edit of the field, even to the same objectives in another order, is a new question.
    view.calibrations_input.value = "\n".join(reversed(text.splitlines()))
    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)

    view.calibrations_input.value = text
    _click(view.offsets_measure_button)
    assert view._offsets_armed == (FORTY, TWENTY)
    # An activity that goes drops it at the next tick.
    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()
    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)
    assert view.offsets_confirm_text.visible is False
    Slow.release.set()
    assert view.page.join(10.0)


def test_the_blocks_handlers_are_guarded_and_listed_where_the_guard_test_walks(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    # FM-171: every handler through `_handler`, and every widget of the block reachable from
    # the lists the guard test walks (`view.screens`), so that a bare registration would fail it.
    _seed_plate(run_doc)
    view = make_view()
    walked = [
        *_objects(view.screens["plate"]),
        *_objects(view.screens["expert"]),
    ]
    widgets = (
        view.offsets_measure_button,
        view.offsets_turn_button,
        view.offsets_cancel_button,
        view.offsets_write_button,
        view.calibrations_input,
        view.offsets_session_input,
    )
    for widget in widgets:
        assert widget in walked
        ours = [
            watcher.fn
            for by_name in widget.param.watchers.values()
            for watchers in by_name.values()
            for watcher in watchers
            if getattr(watcher.fn, "__module__", "") == plate_page.__name__
        ]
        assert ours, f"{widget!r} has no handler of the page"
        assert all(hasattr(fn, "__wrapped__") for fn in ours)


# --- what follows the stand (D5) -------------------------------------------------------


def test_the_in_place_line_follows_the_status_at_every_tick(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    fake_microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_plate(run_doc)
    view = make_view()
    view.show_step("plate")
    view.tick()
    assert view.objective_in_place_text.object == f"Objective in place: {TEN}"
    turret = fake_microscope.require(ObjectiveTurret)

    # The tick reads the stand and the document's cards already drawn, and no file the block
    # reads at a click (FM-145): a spy on each finds no call.
    calls: list[str] = []
    monkeypatch.setattr(
        view.page, "offsets_step", lambda *a: calls.append("offsets_step")
    )
    monkeypatch.setattr(
        view.page, "offsets_plan", lambda *a: calls.append("offsets_plan")
    )
    seen: list[str] = []
    for label in (FORTY, TWENTY, TEN):
        turret.select(label, confirm=True)
        view.tick()
        seen.append(view.objective_in_place_text.object)
    assert seen == [f"Objective in place: {label}" for label in (FORTY, TWENTY, TEN)]
    assert calls == []

    # After a run (a card of the tool in the document) the line says nothing turns it back.
    monkeypatch.undo()
    _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25))))
    view.refresh()
    view.tick()
    assert view.objective_in_place_text.object == (
        f"Objective in place: {TEN} {OFFSETS_NOT_TURNED_BACK}"
    )


def test_the_run_line_says_no_report_yet_then_the_last_report(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    Measuring.hold = True
    view, _text = _ready(make_view, run_doc, tmp_path, seed_camera_calibration)
    assert view.offsets_run_text.visible is False

    _click(view.offsets_measure_button)
    _click(view.offsets_turn_button)
    assert Measuring.started.wait(5)
    view.tick()
    first = view.offsets_run_text.object
    visible_before = view.offsets_run_text.visible
    Measuring.reported.set()
    deadline = threading.Event()
    for _ in range(200):
        view.tick()
        if view.offsets_run_text.object != first:
            break
        deadline.wait(0.02)
    reported = view.offsets_run_text.object
    # The in-place line says the turret may be turning only when it does not know it.
    Measuring.release.set()
    assert view.page.join(10.0)
    view.tick()

    assert first == "no report yet: the turn and the walk to the first well come first"
    assert visible_before is True
    assert reported == f"{FORTY}: A12 -Y (3 of 24)"
    assert view.offsets_run_text.visible is False


# --- text, never markup (FM-139) ------------------------------------------------------


def test_the_card_and_the_preview_are_text_and_not_markup(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    browser_text: Callable[[Any], str],
) -> None:
    label = "40X [bold]*lens*\nnext <b>x</b>"
    _seed_plate(run_doc)
    _seed_offsets(run_doc, _result((label, (-37.5, 12.25))))
    view = make_view()
    view.show_step("plate")
    view.refresh()
    view.tick()

    panes = (
        view.offsets_lead_text,
        view.objective_in_place_text,
        view.calibration_reads_text,
        view.offsets_wait_text,
        view.offsets_confirm_text,
        view.offsets_run_text,
        view.offsets_result_text,
        view.offsets_plan_text,
        view.offsets_write_note,
        view.calibration_how_text,
    )
    assert all(type(pane) is TextPane for pane in panes)
    # And what the browser is sent for each is one `pre` that says the pane's text (FM-271).
    for pane in panes:
        assert browser_text(pane) == (pane.object or " ")
    text = view.offsets_result_text.object
    # The label is in the card whole, on one line, and the markup is the words it is.
    assert "40X [bold]*lens* next <b>x</b>: shift from " in text
    assert text.count("\n") == len(text.splitlines()) - 1
    # No profile file on this stand: the preview says so, as a text and with no button.
    assert view.offsets_plan_text.object == (
        "this stand's profile is built into smc, with no file to write"
    )
    assert view.offsets_write_button.visible is False


# --- per tab (FM-151) ----------------------------------------------------------------


def test_a_second_tab_has_its_own_armed_objectives(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    first, text = _ready(make_view, run_doc, tmp_path, seed_camera_calibration)
    second = make_view()
    second.show_step("plate")
    second.refresh()
    second.tick()
    second.calibrations_input.value = text

    _click(first.offsets_measure_button)

    assert first._offsets_armed == (FORTY, TWENTY)
    assert second._offsets_armed is None
    assert second.offsets_confirm_box.visible is False
    # The second tab's *Turn and measure* turns nothing: it armed nothing, whatever the first did.
    _click(second.offsets_turn_button)
    assert Measuring.seen == []
    assert first._offsets_armed == (FORTY, TWENTY)


# --- a dry run (FM-182) --------------------------------------------------------------


def test_a_dry_run_page_shows_the_wait_beside_the_button(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    fake_core: FakeCore,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # A plate registered from its walls by a real run is read by a dry run page: the block is
    # offered, and says at once why it cannot run, as the tool's own refusal would.
    _seed_plate(run_doc)
    forty = _document(tmp_path, FORTY, seed_camera_calibration)
    dry = _dry_microscope(fake_core)
    try:
        view = make_view(dry)
        view.show_step("plate")
        view.refresh()
        view.tick()
        view.calibrations_input.value = _typed(forty)
        before = run_doc.read_bytes()

        assert view.offsets_box.visible is True
        assert view.offsets_measure_button.disabled is True
        assert view.offsets_turn_button.disabled is True
        assert view.offsets_wait_text.object == OFFSETS_DRY_RUN
        # The click arrives all the same: nothing is armed, nothing is asked, nothing runs.
        _click(view.offsets_measure_button)
        assert view._offsets_armed is None
        assert view.offsets_confirm_box.visible is False
        assert view.offsets_wait_text.object == OFFSETS_DRY_RUN
        assert Measuring.seen == []
        assert run_doc.read_bytes() == before
    finally:
        dry.close()


def test_with_no_document_typed_the_wait_says_to_give_one(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    view = make_view()
    view.show_step("plate")
    view.refresh()
    view.tick()

    _click(view.offsets_measure_button)

    assert view._offsets_armed is None
    assert view.offsets_wait_text.object == OFFSETS_NO_DOCUMENT
    assert view.calibration_reads_text.visible is False


def test_a_busy_page_says_what_runs_beside_the_button(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    view = make_view()
    view.show_step("plate")
    view.refresh()
    view.tick()
    # Idle, and no document read yet: nothing waits beside the button.
    assert view.offsets_wait_text.object == ""
    assert view.offsets_wait_text.visible is False

    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()

    assert view.offsets_wait_text.object == busy_note("slow", live=False)
    assert view.offsets_measure_button.disabled is True
    Slow.release.set()
    assert view.page.join(10.0)


# --- the write, as the view sends it (D7, FM-172) -----------------------------------


@pytest.fixture
def stand_file(tmp_path: Path) -> Path:
    """A copy of ``demo.toml`` in a folder with a space in its name: the one file a write may replace."""
    folder = tmp_path / "my profiles"
    folder.mkdir()
    path = folder / "stand.toml"
    path.write_bytes(DEMO_PROFILE.read_bytes())
    return path


@pytest.fixture
def make_plan_view(
    fake_core: FakeCore, run_doc: Path, stand_file: Path
) -> Iterator[Callable[..., PlatePageView]]:
    """A view over a stand on the profile file, its document holding the plate and a result of the 40X; every page it built is closed."""
    stands: list[Microscope] = []
    pages: list[PlatePage] = []

    def make_stand(extra: str = "") -> Microscope:
        if extra:
            stand_file.write_bytes(stand_file.read_bytes() + extra.encode("utf-8"))
        stand = Microscope.from_core(
            cast("CMMCorePlus", fake_core), Profile.load(stand_file)
        )
        stands.append(stand)
        return stand

    def make(*, held: bool = True) -> PlatePageView:
        page = _plan_page(make_stand, run_doc, held=held)
        pages.append(page)
        view = PlatePageView(page)
        view.show_step("plate")
        view.refresh()
        view.tick()
        return view

    try:
        yield make
    finally:
        for page in pages:
            page.close()
        for stand in stands:
            stand.close()


def _edit_the_profile(stand_file: Path) -> bytes:
    """What an editor or a ``git pull`` does between the preview and the click: another offset for the 40X."""
    stand_file.write_text(
        stand_file.read_text(encoding="utf-8").replace(
            "xy_offset_um = [1.0, 2.0]", "xy_offset_um = [3.0, 4.0]"
        ),
        encoding="utf-8",
    )
    return stand_file.read_bytes()


def test_write_sends_the_lines_that_were_drawn_and_a_profile_edited_since_is_refused(
    make_plan_view: Callable[..., PlatePageView], stand_file: Path
) -> None:
    view = make_plan_view()
    assert view.offsets_write_button.visible is True
    assert view.offsets_write_button.label == "Write these offsets into stand.toml"
    assert view.offsets_write_note.object == ""
    edited = _edit_the_profile(stand_file)

    # The preview on screen is the one drawn before the edit: the controller computes the
    # plan again, finds other lines and writes nothing (a page that recomputed the lines at
    # the click would write what nobody was shown).
    _click(view.offsets_write_button)
    assert stand_file.read_bytes() == edited
    assert view.page.status().outcome == f"Objective offsets: {OFFSETS_PLAN_CHANGED}"
    assert view.offsets_write_note.object == ""

    # Drawn again from the file as it is now, the same click writes.
    view.refresh()
    _click(view.offsets_write_button)
    assert stand_file.read_bytes() != edited
    offsets = Profile.load(stand_file).objectives.offsets[FORTY]
    assert offsets.xy_offset_um == (-37.5, 12.25)
    assert view.page.status().outcome == "Offsets written into stand.toml"
    assert view.offsets_write_note.object == offsets_written_line(
        view.page.profile_argument
    )


def test_the_session_typed_goes_into_the_text_written(
    make_plan_view: Callable[..., PlatePageView], stand_file: Path
) -> None:
    view = make_plan_view()
    view.offsets_session_input.value = "session 3 (#412)"
    assert view.offsets_plan_text.object.endswith(
        'measured = "2026-11-14, session 3 (#412), objective-offsets in run.json"'
    )

    _click(view.offsets_write_button)

    written = Profile.load(stand_file).objectives.offsets[FORTY].measured
    assert written == "2026-11-14, session 3 (#412), objective-offsets in run.json"


def test_a_write_that_fails_is_an_outcome_and_the_block_says_nothing_was_written(
    make_plan_view: Callable[..., PlatePageView],
    stand_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows refuses the rename while an editor or a sync client holds the profile (FM-24,
    # FM-38): the file is as it was, the outcome says why, and nothing claims a write.
    view = make_plan_view()
    before = stand_file.read_bytes()

    def refuse(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        raise AtomicWriteError(PermissionError(13, "Access is denied"), None)

    monkeypatch.setattr("smc.run.files.write_atomically", refuse)
    _click(view.offsets_write_button)

    assert stand_file.read_bytes() == before
    assert view.page.status().outcome == (
        f"Objective offsets: {stand_file.resolve()} was not changed: Access is denied"
    )
    assert view.offsets_write_note.object == ""
    assert view.page.offsets_plan("").written is False


def test_the_session_field_and_the_write_button_wait_for_a_result(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    view = make_view()
    view.show_step("plate")
    view.refresh()
    view.tick()

    assert view.offsets_box.visible is True
    assert view.offsets_session_input.visible is False
    assert view.offsets_write_button.visible is False
    assert view.offsets_write_note.object == ""

    _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25))))
    view.refresh()

    assert view.offsets_session_input.visible is True


# --- what drops the confirmation (D4) ------------------------------------------------


def _armed(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_calibration: Callable[..., Any],
) -> tuple[PlatePageView, list[str]]:
    view, text = _ready(make_view, run_doc, tmp_path, seed_calibration)
    _click(view.offsets_measure_button)
    assert view._offsets_armed == (FORTY, TWENTY)
    return view, text.splitlines()


def test_a_stand_released_since_the_question_drops_it_at_the_next_tick(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, _paths = _armed(make_view, run_doc, tmp_path, seed_camera_calibration)
    status = view.page.status()

    # No activity goes: only the connection changed (the how-to's second step is Disconnect).
    view._draw_offsets(dataclasses.replace(status, connected=False))

    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)
    assert view.offsets_confirm_text.visible is False


def test_a_turn_the_controller_refuses_still_drops_what_was_armed(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, paths = _armed(make_view, run_doc, tmp_path, seed_camera_calibration)
    # The second document is now the calibration of another objective: the click is refused
    # and no tool starts, so nothing running can be what disarms the view.
    seed_camera_calibration(Path(paths[1]), objective="Objective-2")

    _click(view.offsets_turn_button)

    assert Measuring.seen == []
    assert view.page.status().running is None
    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)
    assert "the objectives changed since you confirmed them" in (
        view.page.status().outcome or ""
    )


def test_a_second_measure_that_now_waits_closes_the_question(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, paths = _armed(make_view, run_doc, tmp_path, seed_camera_calibration)
    Path(paths[1]).unlink()

    _click(view.offsets_measure_button)

    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)
    assert "does not exist" in view.offsets_wait_text.object


def test_a_plate_that_is_no_longer_from_the_walls_hides_the_block_and_drops_the_question(
    make_view: Callable[..., PlatePageView],
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    view, _paths = _armed(make_view, run_doc, tmp_path, seed_camera_calibration)
    by_eye = _walls_plate(method="references", with_walls=False, walls_objective=None)
    _seed_plate(run_doc, by_eye, plugin="plate-registration", minute=2)

    view.refresh()

    assert view.offsets_box.visible is False
    assert (view._offsets_armed, view.offsets_confirm_box.visible) == (None, False)


# --- a refusal in the tool's words, and a result gone stale (D6) --------------------


def test_the_block_draws_a_refusal_as_the_tool_words_it_and_says_when_a_result_is_stale(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _seed_offsets(
        run_doc,
        status=StageStatus.failed,
        error="PluginRunError: objective-offsets: 40X is given twice in targets",
    )
    view = make_view()
    view.show_step("plate")
    view.refresh()
    failed = view.offsets_result_text.object.splitlines()

    assert failed[1] == "error: 40X is given twice in targets"
    assert "PluginRunError" not in view.offsets_result_text.object

    # The plate is registered again after a good result: it says so before anything else.
    _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25))), minute=2)
    _seed_plate(run_doc, plugin="plate-walls", minute=3)
    view.refresh()
    stale = view.offsets_result_text.object.splitlines()

    assert stale[1] == OFFSETS_STALE


# --- the how-to (D9) ----------------------------------------------------------------


def test_the_how_to_quotes_the_profile_and_a_windows_page_gets_its_own_note(
    make_plan_view: Callable[..., PlatePageView],
) -> None:
    view = make_plan_view()
    profile = view.page.profile_argument
    assert " " in profile  # the folder of the test's profile has a space
    command = f"-p {shlex.quote(profile)} --run-doc cal-<objective>.json"

    view._windows = False
    view.refresh()
    assert command in view.calibration_how_text.object
    assert CALIBRATION_HOW_WINDOWS_NOTE not in view.calibration_how_text.object

    view._windows = True
    view.refresh()
    assert view.calibration_how_text.object.endswith(CALIBRATION_HOW_WINDOWS_NOTE)
    # The page-wide note points at a button this card does not have: not drawn here.
    assert "press the button" not in view.calibration_how_text.object
