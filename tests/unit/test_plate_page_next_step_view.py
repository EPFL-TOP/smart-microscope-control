"""The plate page's next step, its Wells box and its wrapped text, without a browser (m5 §8.5, #293).

The widgets are driven the way a session drives them: a tap is ``figure._trigger_event``,
a click is ``button.clicks += 1`` (which reaches a *disabled* button's handler too, FM-169),
a typed range is ``select_input.value``. What this cannot show is a browser: the look of the
region, a wrapped path and the box after a click are for ``/ui-check`` (FM-153, FM-161), and
the Wells box changed while the operator types in it is for the owner. Panel's warnings are
errors here (FM-147), and a handler that failed fails the test that did not expect it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from bokeh.events import Tap
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, PlatePage
from smc.ui.guide import NextStep
from smc.ui.plate_page import _WARNING, PlatePageView, next_step_text

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)
STAND = "demo-sample"
PITCH_96_UM = 9000.0


class NoParams(PluginParams):
    pass


class RegistrationParams(PluginParams):
    plate: str = "96-well"
    refs: str = "[]"
    walls: str = "[]"


class ScanPlanParams(PluginParams):
    wells: str
    coverage: float = 1.0
    overlap: float = 0.1


class NoResult(PluginResult):
    pass


class Registration:
    name = "plate-registration"
    provides = "plate"
    summary = "A test registration."
    Params = RegistrationParams
    Result = PlateResult
    requires: tuple[type, ...] = (XYStage,)
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PlateResult:
        return _plate_result(params.plate)


class ScanPlan:
    name = provides = "scan-plan"
    summary = "A test scan plan."
    Params = ScanPlanParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[Any]] = []

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> NoResult:
        ScanPlan.seen.append(params)
        return NoResult()


class Slow:
    """Runs until released or stopped."""

    name = provides = "slow"
    summary = "A test tool that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> NoResult:
        Slow.started.set()
        while not Slow.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


def _ep(cls: type) -> EntryPoint:
    return EntryPoint(
        name=cast("Any", cls).name, value=f"{__name__}:{cls.__name__}", group=GROUP
    )


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    ScanPlan.seen = []
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()  # a worker a failed test left behind ends


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _plate_result(plate: str = "96-well") -> PlateResult:
    """A plate whose A1 is at the origin, unturned: the nominal map's wells, registered."""
    return PlateResult(
        plate=plate,
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=1,
    )


def _seed_plate(run_doc: Path) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=T0,
            finished_utc=T0 + timedelta(minutes=1),
            profile=STAND,
            dry_run=False,
            smc_version="0",
            params={},
            result=_plate_result().model_dump(mode="json"),
        ),
    )


def _seed_raw(run_doc: Path, plugin: str, result: dict[str, Any]) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=plugin,
            attempt="a1",
            status=StageStatus.done,
            started_utc=T0,
            finished_utc=T0 + timedelta(minutes=3),
            profile=STAND,
            dry_run=False,
            smc_version="0",
            params={},
            result=result,
        ),
    )


def _profile(*, sample: bool = True, suggest: dict[str, Any] | None = None) -> Profile:
    """The ``demo-sample`` stand: with its ``[sample]``, the plate step is one click."""
    data: dict[str, Any] = {"microscope": {"name": STAND}}
    if sample:
        data["sample"] = {}
    if suggest is not None:
        data["suggest"] = suggest
    return Profile.model_validate(data)


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    """A closed page over fake stands, connected unless asked not to; all closed afterwards."""
    pages: list[PlatePage] = []
    stands: list[Microscope] = []

    def open_stand() -> Microscope:
        stand = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()),
            Profile.model_validate({"microscope": {"name": STAND}}),
        )
        stands.append(stand)
        return stand

    def make(
        profile: Profile | None = None,
        *,
        tools: tuple[type, ...] = (Registration, ScanPlan),
        connect: bool = True,
    ) -> PlatePage:
        page = PlatePage.closed(
            profile or _profile(),
            run_doc,
            opener=open_stand,
            entry_points=[_ep(cls) for cls in tools],
        )
        pages.append(page)
        if connect:
            page.connect()
            assert page.join(5), "connect did not end"
            assert page.connected, page.status().outcome
        return page

    try:
        yield make
    finally:
        for page in pages:
            page.close()
        for stand in stands:
            stand.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _click(button: Any) -> None:
    button.clicks += 1


def _centre(well: str) -> tuple[float, float]:
    """A well of the 96-well plate with A1 at the origin and no turn (the registered map)."""
    row = "ABCDEFGH".index(well[0])
    return (int(well[1:]) - 1) * PITCH_96_UM, -row * PITCH_96_UM


def _tap(view: PlatePageView, well: str) -> None:
    x, y = _centre(well)
    view.figure._trigger_event(Tap(view.figure, x=x, y=y, sx=0, sy=0))


# --- the text of a step --------------------------------------------------------------------


def test_next_step_text_is_pure_text_in_three_shapes() -> None:
    assert next_step_text(None) == ""
    with_tool = NextStep(
        plugin="scan-plan",
        params=("wells=B2",),
        why=("the run document has no scan-plan yet", "wells: the 1 wells selected"),
        command="smc run scan-plan --param wells=B2",
        blocked=None,
    )
    assert next_step_text(with_tool) == (
        "scan-plan\n"
        "  the run document has no scan-plan yet\n"
        "  wells: the 1 wells selected\n"
        "  command: smc run scan-plan --param wells=B2"
    )
    # Markup in a line is the words it is, never structure (FM-139).
    done = NextStep(
        None, (), ("every stage of the workflow is done:", "<b>[x]</b>"), "", None
    )
    assert next_step_text(done) == "every stage of the workflow is done:\n<b>[x]</b>"


# --- the region, its button, and what it runs ------------------------------------------------


def test_the_next_step_region_heads_the_right_column(
    make_page: Callable[..., PlatePage],
) -> None:
    page = make_page()
    view = PlatePageView(page)
    _left, right = view.layout[-1]
    region, drive = right[0], right[1]
    # Above Drive, and the first thing the column shows (ADR-0006's picture).
    assert region[0].object == "Next step"
    assert region[1] is view.next_text
    assert region[2] is view.next_button
    assert region[3] is view.next_note
    assert drive[0].object == "Drive"
    assert view.next_text.object == next_step_text(page.view().next_step)
    assert view.next_text.object.startswith(
        "plate-registration\n  the run document has no plate yet\n"
    )
    assert view.next_button.label == "Run plate-registration"
    assert view.next_button.color == "primary"
    assert view.next_button.visible
    assert not view.next_button.disabled
    assert view.next_note.object == ""
    # The reason a step cannot run is in the warning colour, as a tool's own note is.
    assert view.next_note.styles["color"] == _WARNING


class SlowScanPlan(ScanPlan):
    """The scan plan, held until released: the run the Next button starts goes on."""

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> NoResult:
        Slow.started.set()
        while not Slow.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


def test_a_click_on_the_next_button_draws_the_run_at_once(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # As every Run handler, `on_next` ends with a tick: the strip, the buttons and the
    # activity follow the click, not the periodic callback up to a second later, so a
    # second click is met by a disabled button and not only by the controller.
    _seed_plate(run_doc)
    page = make_page(tools=(Registration, SlowScanPlan))
    view = PlatePageView(page)
    page.select_text("B2")
    view.refresh()
    assert not view.next_button.disabled
    assert view.run_text.object == "connect: demo-sample connected, 13 devices"
    _click(view.next_button)
    try:
        assert Slow.started.wait(5)
        assert view.run_text.object == "scan-plan is running"
        assert view.next_button.disabled
    finally:
        Slow.release.set()
        assert page.join(10.0)


def test_the_refresh_button_forgets_the_last_outcome_and_a_redraw_does_not(
    make_page: Callable[..., PlatePage],
) -> None:
    # The walk's strip kept `cells: ... run scan again` after the chain was run in a
    # terminal and the page refreshed (m5 §8.8.3). Only the button forgets; the redraw
    # that the end of a run makes must show the outcome it has just got.
    page = make_page(tools=(Registration, ScanPlan, Slow))
    view = PlatePageView(page)
    page.run("slow", [])
    try:
        assert Slow.started.wait(5)
        view.tick()
    finally:
        Slow.release.set()
    assert page.join(10.0)
    view.tick()  # the tick that sees the run end refreshes the view
    assert view.run_text.object.startswith("slow done in ")
    view.refresh()
    view.tick()
    assert view.run_text.object.startswith("slow done in ")

    _click(view.refresh_button)

    assert page.status().outcome is None
    assert view.run_text.object == ""


def test_the_next_button_runs_the_pairs_it_drew(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    page.select_text("B2")
    view.refresh()
    assert "wells=B2" in view.next_text.object
    # The selection moves with no redraw (another tab, a click not yet seen): the
    # operator runs what they read, not what the page would compute now.
    page.select_text("C3")
    _click(view.next_button)
    assert page.join(10.0)
    assert [p.wells for p in ScanPlan.seen] == ["B2"]
    record = RunDocument.load(run_doc).stages["scan-plan"]
    assert record.params["wells"] == "B2"


def test_the_next_button_is_disabled_with_the_reason_and_hidden_without_a_tool(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page(connect=False)
    view = PlatePageView(page)
    assert view.next_button.disabled
    assert view.next_note.object == NOT_CONNECTED

    page.connect()
    assert page.join(5)
    view.tick()
    assert not view.next_button.disabled
    assert view.next_note.object == ""

    # While an activity goes, no button of a run is offered.
    view._running = "something"
    view._draw_buttons()
    assert view.next_button.disabled
    view._running = None

    # No plugin fills the slot: nothing to run, and the lines are the reason.
    _seed_plate(run_doc)
    bare = make_page(tools=(Registration,))
    bare_view = PlatePageView(bare)
    assert not bare_view.next_button.visible
    assert bare_view.next_text.object == (
        "no registered plugin provides scan-plan; smc run --list shows the plugins"
    )


def test_a_click_on_the_disabled_next_button_is_refused_by_the_controller(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # FM-169: `disabled` is drawn by the browser; a click can still arrive.
    page = make_page(connect=False)
    view = PlatePageView(page)
    assert view.next_button.disabled
    _click(view.next_button)
    assert page.status().outcome == f"plate-registration: {NOT_CONNECTED}"
    assert not run_doc.exists()
    assert page.status().running is None


def test_a_click_with_no_step_drawn_runs_nothing_and_says_to_refresh(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page()
    view = PlatePageView(page)
    view._next_drawn = None
    view.on_next()
    assert view.message.object == "nothing to run: press Refresh"
    done = NextStep(None, (), ("every stage of the workflow is done:",), "", None)
    view._next_drawn = done
    view.on_next()
    assert view.message.object == "nothing to run: press Refresh"
    assert not run_doc.exists()


def test_a_run_that_is_refused_is_the_message_and_not_an_exception(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The step of a page with a run in flight: the controller says `<plugin> is running`.
    _seed_plate(run_doc)
    page = make_page(tools=(Registration, ScanPlan, Slow))
    view = PlatePageView(page)
    page.select_text("B2")
    view.refresh()
    page.run("slow", [])
    assert Slow.started.wait(5)
    try:
        # The button is disabled while it runs, and a click still arrives (FM-169).
        assert view.next_button.disabled is False  # the tick has not drawn it yet
        _click(view.next_button)
        assert view.message.object == "slow is running"
    finally:
        Slow.release.set()
        assert page.join(10.0)
    assert [p.wells for p in ScanPlan.seen] == []


def test_the_next_steps_stage_opens_and_others_stay_open(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page()
    view = PlatePageView(page)
    stages = list(dict.fromkeys(tool.stage for tool in page.tools()))
    assert stages == ["Set up", "Choose"]
    # The operator folded everything away; the next step moves to the scan plan.
    view.accordion.active = []
    _seed_plate(run_doc)
    view.refresh()
    assert "scan-plan" in view.next_text.object.splitlines()[0]
    assert view.accordion.active == [stages.index("Choose")]
    # The operator opens the other card: a redraw with the same tool keeps both.
    view.accordion.active = [0, 1]
    view.refresh()
    assert view.accordion.active == [0, 1]
    # A step that moves to a stage already open leaves the open cards as they are.
    view.accordion.active = [0, 1]
    write_stage(run_doc, _failed("plate-registration", "plate"))
    view.refresh()
    assert view.accordion.active == [0, 1]


def test_a_redraw_with_the_same_tool_does_not_open_a_card_the_operator_folded(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The card opens when the step *moves* to a tool: a click on a well, Refresh and the
    # end of a run redraw with the same tool, and must not undo the operator's fold.
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    assert "scan-plan" in view.next_text.object.splitlines()[0]
    view.accordion.active = [0]  # the scan plan's card (1) is folded
    page.select_text("B2")
    view.refresh()
    assert view.accordion.active == [0]
    view.tick()
    assert view.accordion.active == [0]


def test_a_step_that_comes_back_to_a_tool_opens_its_card_again(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # Tool, then no tool (the scan has no plugin here), then the tool again: the card
    # opens at the return, which is a move, as it did the first time.
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    view.accordion.active = [0]
    _seed_raw(run_doc, "scan-plan", {})
    view.refresh()
    assert not view.next_button.visible  # no registered plugin provides the scan
    assert view.accordion.active == [0]
    write_stage(run_doc, _failed("scan-plan", "scan-plan"))
    view.refresh()
    assert view.next_button.label == "Run scan-plan"
    assert view.accordion.active == [0, 1]


def _failed(plugin: str, provides: str) -> StageRecord:
    return StageRecord(
        plugin=plugin,
        provides=provides,
        attempt="a2",
        status=StageStatus.failed,
        started_utc=T0,
        finished_utc=T0,
        profile=STAND,
        dry_run=False,
        smc_version="0",
        params={},
        result=None,
        error="the stage failed",
    )


# --- a suggested field -------------------------------------------------------------------------


def test_a_suggested_field_starts_with_its_value_and_its_run_sends_it(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    profile = _profile(
        suggest={
            "scan-plan": {"params": ["coverage=0.4"], "why": "a few wells take seconds"}
        }
    )
    page = make_page(profile)
    view = PlatePageView(page)
    fields = view.tool_fields["scan-plan"]
    assert fields["coverage"].value == "0.4"
    assert "suggested by the profile: a few wells take seconds" in (
        fields["coverage"].description
    )
    # Only the suggested key starts filled.
    assert fields["overlap"].value == ""
    assert "suggested" not in fields["overlap"].description

    page.select_text("B2")
    view.on_run("scan-plan")
    assert page.join(10.0)
    assert [(p.wells, p.coverage) for p in ScanPlan.seen] == [("B2", 0.4)]


def test_a_profile_without_a_suggestion_leaves_every_field_empty(
    make_page: Callable[..., PlatePage],
) -> None:
    view = PlatePageView(make_page())
    assert all(
        widget.value == ""
        for fields in view.tool_fields.values()
        for widget in fields.values()
    )


# --- the run document and the wrap -----------------------------------------------------------


def test_the_run_document_header_names_the_absolute_path(
    make_page: Callable[..., PlatePage],
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = make_page()
    view = PlatePageView(page)
    left, _right = view.layout[-1]
    assert view.run_document_text.object == f"Run document  {run_doc}"
    # Heads the checklist.
    items = list(left)
    assert items.index(view.checklist_text) == items.index(view.run_document_text) + 1
    assert view.run_document_text.styles["font-weight"] == "bold"

    # A relative `--run-doc` is shown as the server resolves it.
    monkeypatch.chdir(run_doc.parent)
    relative = PlatePage.closed(
        _profile(), Path("run.json"), entry_points=[_ep(Registration)]
    )
    try:
        shown = PlatePageView(relative).run_document_text.object
    finally:
        relative.close()
    assert shown == f"Run document  {Path.cwd() / 'run.json'}"


def test_cards_and_the_step_wrap(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    long_path = "report/" + "d" * 120 + ".html"
    _seed_raw(run_doc, "scan-plan", {"file": long_path, "n": 3})
    page = make_page()
    view = PlatePageView(page)
    wrapped = [
        *view.tool_cards.values(),
        view.next_text,
        view.next_note,
        view.run_document_text,
    ]
    assert len(wrapped) == len(view.tool_cards) + 3
    for pane in wrapped:
        # Plain text, never Markdown: a result's words are not markup (FM-139).
        assert type(pane) is pn.pane.Str
        assert pane.styles["white-space"] == "pre-wrap"
        assert pane.styles["overflow-wrap"] == "anywhere"
        # The `<pre>` inside the pane's shadow root keeps its own rule, so the stylesheet
        # carries it too (measured in Chromium, #252); a test sees that it is there.
        assert pane.stylesheets == [
            "pre { white-space: pre-wrap; overflow-wrap: anywhere; }"
        ]
    for pane in [*view.tool_cards.values(), view.next_text, view.next_note]:
        assert pane.width == 380
    # A path longer than the old 80-character cut reaches the card whole.
    text = view.tool_cards["scan-plan"].object
    assert f"  file: {run_doc.parent / long_path}" in text.splitlines()


# --- the Wells box follows the selection (the owner's fourth point) ----------------------------


def test_a_click_on_a_well_writes_the_selection_in_wells(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    view = PlatePageView(make_page())
    assert view.select_input.value == ""
    _tap(view, "B2")
    assert view.select_input.value == "B2"
    _tap(view, "B3")
    assert view.select_input.value == "B2,B3"
    _tap(view, "B2")  # a second click takes it off again
    assert view.select_input.value == "B3"
    _tap(view, "B3")
    assert view.select_input.value == ""


def test_a_typed_range_stays_as_typed(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    view.select_input.value = "A1:A3"
    assert page.selection == ("A1", "A2", "A3")
    assert view.select_input.value == "A1:A3"
    view.refresh()
    assert view.select_input.value == "A1:A3"
    # A later click is another change: the box then says what is selected.
    _tap(view, "B2")
    assert view.select_input.value == "A1,A2,A3,B2"


def test_an_emptied_wells_box_clears_the_selection_and_the_box_follows_again(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The owner's sixth point: an emptied box changed nothing, and then stood for a
    # selection it no longer showed until the selection changed (m5 §8.8.5). The box
    # is set by `value=`, the way a session's change arrives (FM-161).
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    view.select_input.value = "A1:A3"
    assert page.selection == ("A1", "A2", "A3")

    view.select_input.value = ""

    assert page.selection == ()
    assert view.select_input.value == ""
    assert view.message.object == ""
    # The next click is a change again: the box follows it.
    _tap(view, "B2")
    assert page.selection == ("B2",)
    assert view.select_input.value == "B2"


def test_a_box_of_blanks_clears_the_selection_too(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    view.select_input.value = "A1:A3"
    view.select_input.value = "  "
    assert page.selection == ()
    assert view.message.object == ""
    # As a typed range does, the box keeps what was typed: it stands for no wells.
    assert view.select_input.value == "  "


def test_the_echo_of_an_emptied_selection_does_not_select_again(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The click that takes the last well off empties the box by the echo; the guard
    # alone stops that assignment from calling `select_text("")` (it no longer returns
    # at a blank text, m5 §8.8.5).
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    calls: list[str] = []
    real = page.select_text

    def spy(text: str) -> None:
        calls.append(text)
        real(text)

    with _replaced(page, "select_text", spy):
        _tap(view, "B2")
        _tap(view, "B2")
        assert view.select_input.value == ""
        assert page.selection == ()
        assert calls == []


def test_a_range_that_is_refused_stays_in_the_box_with_the_reason(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page()
    view = PlatePageView(page)
    page.select_text("A1")
    view.refresh()
    view.select_input.value = "Z99"
    assert page.selection == ("A1",)
    assert view.message.object != ""
    assert view.select_input.value == "Z99"


def test_a_selection_another_tab_made_or_a_plate_chosen_reaches_the_box(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page()
    view = PlatePageView(page)
    other = PlatePageView(page)  # another tab of the same server
    other.select_input.value = "A1:A2"
    view.tick()
    assert view.select_input.value == "A1,A2"
    page.choose_plate("384-well")
    view.tick()
    assert view.select_input.value == ""


def test_a_selection_the_run_document_holds_is_in_the_box_at_the_start(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin="scan-plan",
            provides="scan-plan",
            attempt="a0",
            status=StageStatus.done,
            started_utc=T0,
            finished_utc=T0,
            profile=STAND,
            dry_run=False,
            smc_version="0",
            params={"wells": "B2,B3"},
            result={},
        ),
    )
    view = PlatePageView(make_page())
    assert view.select_input.value == "B2,B3"


def test_the_wells_echo_does_not_select_again(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The echo assigns the box, and a watcher on its value is called by the assignment
    # at once (measured, m5 §8.5): without the guard it would select the same wells
    # again, through the handler a session registered (FM-161: the box is set by
    # `value=`, the way a session's change arrives, not by calling the handler).
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    calls: list[str] = []
    real = page.select_text

    def spy(text: str) -> None:
        calls.append(text)
        real(text)

    with _replaced(page, "select_text", spy):
        _tap(view, "B2")
        _tap(view, "B3")
        assert view.select_input.value == "B2,B3"
        assert calls == []
        assert view._echoing is False
        # A range typed after the echo is a change, and selects.
        view.select_input.value = "C1:C2"
        assert calls == ["C1:C2"]
    assert page.selection == ("C1", "C2")


def test_a_failure_of_the_echo_does_not_leave_the_guard_up(
    make_page: Callable[..., PlatePage], run_doc: Path, handler_failures: list[Any]
) -> None:
    _seed_plate(run_doc)
    page = make_page()
    view = PlatePageView(page)
    seen: list[str] = []

    def boom(event: Any) -> None:
        seen.append("watcher")
        raise RuntimeError("the watcher failed")

    watcher = view.select_input.param.watch(boom, "value")
    _tap(view, "B2")
    # The tap's handler is guarded: the failure is a line of the page, and the
    # guard of the echo is down again, so that a range typed later is applied.
    assert seen == ["watcher"]
    assert view.message.object.startswith(
        "that action failed on the page: RuntimeError: the watcher failed"
    )
    assert len(handler_failures) == 1
    handler_failures.clear()  # expected here: the fixture fails a test that leaves one
    assert view._echoing is False
    view.select_input.param.unwatch(watcher)
    view.select_input.value = "C1"
    assert page.selection == ("C1",)
