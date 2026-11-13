"""The controller's chain: *Run all that are left* is one activity over the Results tools (m5 §17.11.2, #446).

Five stubs, each named and provided as the real tool and with the real tool's ``reads``, run over
a seeded scan plan and scan; no Panel and no browser. How each ends is the test's: ``done``,
``fail`` (raises ``PluginRunError``), ``wait`` (waits for its ``Event``, then looks at the stop)
and ``wait_ignoring_stop`` (the same, without the look). Threads meet on ``threading.Event`` with
a bound, never on a sleep (FM-44), and a stub does not look at a stop by itself: the test releases
its ``Event`` after the stop.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.errors import MotionStoppedError
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.run import (
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, ChainEnd, PlatePage

#: Bounded waits (FM-44): long enough for a slow CI runner, never forever.
WAIT_S = 10.0
T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
TOOLS = ("cells", "density", "route", "report", "nis-export")


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class _Stub:
    """A tool of the Results screen that ends as ``Stub.behaviour`` says for its name."""

    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads_optional: tuple[str, ...] = ()
    name: str
    provides: str
    summary = "A test stand-in for a Results tool."
    reads: tuple[str, ...]

    #: How each tool ends (default ``done``), and what each did.
    behaviour: ClassVar[dict[str, str]] = {}
    ran: ClassVar[list[str]] = []
    reports: ClassVar[dict[str, int]] = {}
    started: ClassVar[dict[str, threading.Event]] = {}
    release: ClassVar[dict[str, threading.Event]] = {}

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        _Stub.ran.append(self.name)
        if self.name in _Stub.reports:
            ctx.progress(_Stub.reports[self.name], 3, "")
        how = _Stub.behaviour.get(self.name, "done")
        if how == "fail":
            raise PluginRunError(self.name, "it broke")
        if how == "move_stopped":
            # Inside a move when the stop lands: the move raises, and the run ends `failed`.
            _Stub.started[self.name].set()
            _Stub.release[self.name].wait(WAIT_S)
            raise MotionStoppedError("xy_stage XY")
        if how in ("wait", "wait_ignoring_stop"):
            _Stub.started[self.name].set()
            _Stub.release[self.name].wait(WAIT_S)
            if how == "wait":
                ctx.raise_if_stopped()
        return NoResult()


class Cells(_Stub):
    name = provides = "cells"
    reads = ("scan",)


class Density(_Stub):
    name = provides = "density"
    reads = ("scan-plan", "scan", "cells")


class Route(_Stub):
    name = provides = "route"
    reads = ("scan-plan", "scan", "cells")


class Report(_Stub):
    name = provides = "report"
    reads = ("scan-plan", "scan", "cells", "density")
    # As the real tool: it reads the route when there is one, so a new route makes it out of date.
    reads_optional = ("route",)


class NisExport(_Stub):
    name = provides = "nis-export"
    reads = ("route",)


STUBS: tuple[type[_Stub], ...] = (Cells, Density, Route, Report, NisExport)


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)
        for cls in STUBS
    ]


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    _Stub.behaviour = {}
    _Stub.ran = []
    _Stub.reports = {}
    _Stub.started = {name: threading.Event() for name in TOOLS}
    _Stub.release = {name: threading.Event() for name in TOOLS}
    try:
        yield
    finally:
        for event in _Stub.release.values():  # a worker a failed test left behind ends
            event.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    """A page over a fake stand, connected unless told not to; every page and stand is closed afterwards."""
    pages: list[PlatePage] = []
    stands: list[Microscope] = []

    def make(*, connect: bool = True) -> PlatePage:
        profile = Profile.demo()

        def open_stand() -> Microscope:
            stand = Microscope.from_core(
                cast("Any", FakeCore.demo_like()), profile, dry_run=False
            )
            stands.append(stand)
            return stand

        page = PlatePage.closed(
            profile, run_doc, opener=open_stand, entry_points=_entry_points()
        )
        pages.append(page)
        if connect:
            page.connect()
            assert page.join(WAIT_S), "connect did not end"
            assert page.connected, page.status().outcome
        return page

    try:
        yield make
    finally:
        for event in _Stub.release.values():
            event.set()
        for page in pages:
            page.close()
        for stand in stands:
            stand.close()


@pytest.fixture
def page(make_page: Callable[..., PlatePage], run_doc: Path) -> PlatePage:
    _seed_scan(run_doc)
    return make_page()


# --- the run document, as the scan left it -------------------------------------------------


def _seed(
    run_doc: Path,
    plugin: str,
    *,
    minute: int,
    reads: dict[str, ReadRecord | None] | None = None,
    status: StageStatus = StageStatus.done,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=plugin,
            attempt=f"a{minute}",
            status=status,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result={} if status is StageStatus.done else None,
            reads=reads or {},
            checkpoint=checkpoint,
        ),
    )


def _seed_scan(run_doc: Path) -> None:
    """A scan plan and the scan that read it: what a chain finds when the scan has just ended."""
    _seed(run_doc, "scan-plan", minute=1)
    _seed(
        run_doc,
        "scan",
        minute=2,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )


def _read(run_doc: Path, *slots: str) -> dict[str, ReadRecord | None]:
    """The reads of a record made now: each slot's latest provider, as the runner records them."""
    doc = RunDocument.load(run_doc)
    reads: dict[str, ReadRecord | None] = {}
    for slot in slots:
        provider = doc.provider(slot)
        assert provider is not None, slot
        reads[slot] = ReadRecord(plugin=provider.plugin, attempt=provider.attempt)
    return reads


def _ended(page: PlatePage) -> None:
    assert page.join(WAIT_S), "the activity did not end"
    assert page.status().running is None


def _outcome(page: PlatePage) -> str:
    outcome = page.status().outcome
    assert outcome is not None
    return outcome


def _records(run_doc: Path) -> dict[str, StageStatus]:
    return {name: r.status for name, r in RunDocument.load(run_doc).stages.items()}


# --- the chain ---------------------------------------------------------------------------


def test_run_all_runs_each_tool_left_in_order_when_the_one_before_ended_done(
    page: PlatePage, run_doc: Path
) -> None:
    page.run_chain(TOOLS, {})
    _ended(page)

    assert _Stub.ran == list(TOOLS)
    outcome = _outcome(page)
    assert outcome.startswith("Run all that are left: done in ")
    assert outcome.endswith(
        " s (Count the cells, Density per well, Route, Report, Export for NIS-Elements)"
    )
    chain = page.status().chain
    assert chain is not None
    assert chain.plugins == TOOLS
    assert [(end.plugin, end.how) for end in chain.ended] == [
        (tool, "done") for tool in TOOLS
    ]
    assert chain.current is None
    # The records are the runner's own: five done records that read what the one before wrote.
    records = _records(run_doc)
    assert all(records[tool] is StageStatus.done for tool in TOOLS)
    # The checklist says the same: nothing is left to run.
    assert page.view().checklist[-1].state == "ok"


def test_a_tool_that_is_ok_is_never_run_by_the_chain(
    page: PlatePage, run_doc: Path
) -> None:
    page.run("cells", [])
    _ended(page)
    first_attempt = RunDocument.load(run_doc).stages["cells"].attempt
    _Stub.ran.clear()

    page.run_chain(TOOLS, {})
    _ended(page)

    # Running `cells` again would make everything after it out of date (m5 §17.11.1).
    assert _Stub.ran == ["density", "route", "report", "nis-export"]
    assert RunDocument.load(run_doc).stages["cells"].attempt == first_attempt
    chain = page.status().chain
    assert chain is not None
    assert chain.plugins == ("density", "route", "report", "nis-export")
    assert _outcome(page).endswith(
        " s (Density per well, Route, Report, Export for NIS-Elements)"
    )


def test_a_tool_the_chain_would_make_out_of_date_is_left_too(
    page: PlatePage, run_doc: Path
) -> None:
    """#446's `## Plan deviation`: a `report` that read the route as absent, with the route not run.

    The report is `ok`, so a chain that left it would run the route and end with a report that
    is out of date and that it did not run. The rule (m5 §17.12 D14): an `ok` tool that a tool
    left before it feeds, through `reads` or `reads_optional`, is left too.
    """
    _seed(run_doc, "cells", minute=3, reads=_read(run_doc, "scan"))
    _seed(
        run_doc,
        "density",
        minute=4,
        reads=_read(run_doc, "scan-plan", "scan", "cells"),
    )
    _seed(
        run_doc,
        "report",
        minute=5,
        # The route was not there when the report was made: an optional read of nothing.
        reads={
            **_read(run_doc, "scan-plan", "scan", "cells", "density"),
            "route": None,
        },
    )
    states = {item.slot: item.state for item in page.view().checklist}
    assert (states["cells"], states["density"], states["report"]) == ("ok",) * 3
    assert states["route"] == "not run"

    assert page._chain_left(TOOLS, RunDocument.load(run_doc)) == (
        "route",
        "report",
        "nis-export",
    )
    page.run_chain(TOOLS, {})
    _ended(page)

    # `cells` and `density` read nothing the chain makes again: they are never run.
    assert _Stub.ran == ["route", "report", "nis-export"]
    chain = page.status().chain
    assert chain is not None
    assert chain.plugins == ("route", "report", "nis-export")
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("route", "done"),
        ("report", "done"),
        ("nis-export", "done"),
    ]
    # What fails today: the report was out of date after the route and the chain left it.
    assert {
        item.slot: item.state for item in page.view().checklist if item.slot in TOOLS
    } == {tool: "ok" for tool in TOOLS}


def test_what_a_tool_left_feeds_is_left_transitively_and_in_the_chains_order(
    page: PlatePage, run_doc: Path
) -> None:
    # `cells` failed, so it is made again; the route reads it and the export reads the route.
    _seed(run_doc, "cells", minute=3, status=StageStatus.failed)
    _seed(run_doc, "route", minute=5, reads=_read(run_doc, "scan-plan", "scan"))
    _seed(run_doc, "nis-export", minute=6, reads=_read(run_doc, "route"))
    doc = RunDocument.load(run_doc)
    # The export reads only the route: it is left because the route is, which is left because
    # of `cells` (transitively), and the chain's order says each is looked at after its feeder.
    assert page._chain_left(("cells", "route", "nis-export"), doc) == (
        "cells",
        "route",
        "nis-export",
    )
    # A tool the chain was not asked to run is not run for it: `nis-export` is a feeder's
    # reader only through the route, which is not in this chain.
    assert page._chain_left(("cells", "nis-export"), doc) == ("cells",)
    # In the chain's order: a tool before its feeder was looked at before the feeder was left.
    assert page._chain_left(("route", "cells"), doc) == ("cells",)
    # A name no plugin has provides itself and reads nothing; it is left only when not ok.
    assert page._chain_left(("their-tool",), doc) == ("their-tool",)


def test_a_failed_tool_ends_the_chain_and_the_outcome_names_it_with_what_was_done_and_not_run(
    page: PlatePage, run_doc: Path
) -> None:
    _Stub.behaviour["route"] = "fail"

    page.run_chain(TOOLS, {})
    _ended(page)

    assert _Stub.ran == ["cells", "density", "route"]
    # The tool's own words, its plain title and its one line, with no exception type.
    assert _outcome(page) == (
        "Run all that are left: Route failed: route: it broke; "
        "done: Count the cells, Density per well; not run: Report, Export for NIS-Elements"
    )
    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("cells", "done"),
        ("density", "done"),
        ("route", "failed"),
    ]
    assert chain.ended[-1].outcome == "Route failed: route: it broke"
    assert chain.current is None
    records = _records(run_doc)
    assert records["route"] is StageStatus.failed
    assert "report" not in records
    assert "nis-export" not in records


def test_a_refused_tool_ends_the_chain_with_the_text_run_gives_and_writes_nothing(
    page: PlatePage, run_doc: Path, make_page: Callable[..., PlatePage]
) -> None:
    # A parameter the tool refuses: the text is the one a single run of it leaves.
    page.run("route", ["bogus=1"])
    single = _outcome(page)
    assert single.startswith("route: ParamsError: ")

    page.run_chain(TOOLS, {"route": ["bogus=1"]})
    _ended(page)

    assert _Stub.ran == ["cells", "density"]
    assert _outcome(page) == (
        f"Run all that are left: {single}; done: Count the cells, Density per well; "
        "not run: Report, Export for NIS-Elements"
    )
    chain = page.status().chain
    assert chain is not None
    assert chain.ended[-1] == ChainEnd("route", "refused", single)
    # Nothing was written for the refused tool or the ones after it.
    assert set(_records(run_doc)) == {"scan-plan", "scan", "cells", "density"}


def test_a_stopped_record_with_a_checkpoint_is_refused_by_the_chain_as_run_refuses_it(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-172: a stage stopped with a checkpoint is never started afresh over it.
    page.run_chain(("cells", "density"), {})
    _ended(page)
    _seed(
        run_doc,
        "route",
        minute=10,
        status=StageStatus.stopped,
        checkpoint={"done": 1},
        reads=_read(run_doc, "scan-plan", "scan", "cells"),
    )
    page.run("route", [])
    single = _outcome(page)
    assert "with a checkpoint" in single
    _Stub.ran.clear()

    page.run_chain(TOOLS, {})
    _ended(page)

    assert _Stub.ran == []
    # `cells` and `density` are ok, so the chain was `route`, `report` and `nis-export`.
    assert _outcome(page) == (
        f"Run all that are left: {single}; not run: Report, Export for NIS-Elements"
    )
    chain = page.status().chain
    assert chain is not None
    assert chain.ended == (ChainEnd("route", "refused", single),)
    record = RunDocument.load(run_doc).stages["route"]
    assert record.status is StageStatus.stopped
    assert record.checkpoint == {"done": 1}


def test_stop_during_a_tool_ends_the_chain_stopped_at_that_tool(
    page: PlatePage, run_doc: Path
) -> None:
    _Stub.behaviour["route"] = "wait"

    page.run_chain(TOOLS, {})
    assert _Stub.started["route"].wait(WAIT_S)
    page.stop()
    _Stub.release["route"].set()
    _ended(page)

    assert _Stub.ran == ["cells", "density", "route"]
    assert _outcome(page) == (
        "Run all that are left: Route stopped; done: Count the cells, Density per well; "
        "not run: Report, Export for NIS-Elements"
    )
    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("cells", "done"),
        ("density", "done"),
        ("route", "stopped"),
    ]
    assert _records(run_doc)["route"] is StageStatus.stopped
    # The stop note names the tool that was going, never "run all".
    assert page.status().stop_note is not None
    assert page.status().stop_note.startswith("Route told to stop; ")  # type: ignore[union-attr]


def test_a_stop_that_lands_in_a_move_ends_the_chain_stopped_while_the_record_stays_failed(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-64: the stage stopped, so the words say stopped (the Results screen's line says `by STOP`,
    # not `which failed`); the record is `failed`, because the run raised.
    _Stub.behaviour["route"] = "move_stopped"

    page.run_chain(TOOLS, {})
    assert _Stub.started["route"].wait(WAIT_S)
    page.stop()
    _Stub.release["route"].set()
    _ended(page)

    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("cells", "done"),
        ("density", "done"),
        ("route", "stopped"),
    ]
    assert _outcome(page) == (
        "Run all that are left: Route stopped by STOP; done: Count the cells, "
        "Density per well; not run: Report, Export for NIS-Elements"
    )
    assert _records(run_doc)["route"] is StageStatus.failed


def test_stop_between_two_tools_ends_the_chain_before_the_next(
    page: PlatePage, run_doc: Path
) -> None:
    # `density` ignores the stop and ends done: without a look at the stop before the next
    # tool, `route` would start (FM-226's new entry).
    _Stub.behaviour["density"] = "wait_ignoring_stop"

    page.run_chain(TOOLS, {})
    assert _Stub.started["density"].wait(WAIT_S)
    page.stop()
    _Stub.release["density"].set()
    _ended(page)

    assert _Stub.ran == ["cells", "density"]
    assert _outcome(page) == (
        "Run all that are left: stopped before Route; done: Count the cells, "
        "Density per well; not run: Report, Export for NIS-Elements"
    )
    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("cells", "done"),
        ("density", "done"),
    ]
    assert chain.current is None
    records = _records(run_doc)
    assert records["density"] is StageStatus.done
    assert "route" not in records


def test_the_chain_is_one_activity_a_second_run_is_refused_between_two_tools(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused: list[str] = []
    current_in_the_gap: list[str | None] = []
    real = page._begin_tool

    def spy(tool: str, event: threading.Event) -> None:
        # The gap between two tools, on the worker: the activity still holds the tool before.
        if tool == "route":
            try:
                page.run("report", [])
            except RuntimeError as exc:
                refused.append(str(exc))
            # A tick here reads the tool about to start, not a chain that stopped.
            chain = page.status().chain
            assert chain is not None
            current_in_the_gap.append(chain.current)
        real(tool, event)

    monkeypatch.setattr(page, "_begin_tool", spy)
    _Stub.behaviour["cells"] = "wait"

    page.run_chain(TOOLS, {})
    assert _Stub.started["cells"].wait(WAIT_S)
    # From another thread, while a tool goes: a run and a second chain are both refused.
    with pytest.raises(RuntimeError, match="cells is running"):
        page.run("density", [])
    with pytest.raises(RuntimeError, match="cells is running"):
        page.run_chain(TOOLS, {})
    _Stub.release["cells"].set()
    _ended(page)

    assert refused == ["density is running"]
    assert current_in_the_gap == ["route"]
    assert page.activities_ended == 1 + 1  # connect, then one chain


def test_run_all_with_the_stand_closed_or_nothing_left_starts_no_activity(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_scan(run_doc)
    closed = make_page(connect=False)
    closed.run_chain(TOOLS, {})
    assert closed.status().outcome == f"Run all that are left: {NOT_CONNECTED}"
    assert closed.status().running is None
    assert closed.status().chain is None
    assert closed.activities_ended == 0
    assert _Stub.ran == []

    connected = make_page()
    connected.run_chain(TOOLS, {})
    _ended(connected)
    ended = connected.activities_ended
    _Stub.ran.clear()

    connected.run_chain(TOOLS, {})
    assert connected.status().outcome == (
        "Run all that are left: nothing is left: every step is done"
    )
    assert connected.status().running is None
    assert connected.activities_ended == ended
    assert _Stub.ran == []


def test_status_names_the_tool_going_and_the_tools_the_chain_has_ended(
    page: PlatePage,
) -> None:
    _Stub.behaviour["route"] = "wait"
    before = page.activities_ended

    page.run_chain(TOOLS, {})
    assert _Stub.started["route"].wait(WAIT_S)
    during = page.status()
    assert during.running == "route"
    assert during.chain is not None
    assert during.chain.plugins == TOOLS
    assert during.chain.current == "route"
    assert [end.plugin for end in during.chain.ended] == ["cells", "density"]
    # Two tools ended and no activity was counted for them: the chain is one (m5 §2.1).
    assert page.activities_ended == before
    _Stub.release["route"].set()
    _ended(page)

    after = page.status()
    assert after.chain is not None
    assert after.chain.current is None
    assert [end.plugin for end in after.chain.ended] == list(TOOLS)
    assert page.activities_ended == before + 1


@pytest.mark.parametrize("read_during_the_first", [True, False])
def test_a_tools_last_report_does_not_show_as_the_next_tools_first(
    page: PlatePage, read_during_the_first: bool
) -> None:
    # `cells` reports and `density` does not: what `status()` shows while `density` goes.
    _Stub.reports["cells"] = 2
    _Stub.behaviour["density"] = "wait"
    if read_during_the_first:
        _Stub.behaviour["cells"] = "wait"

    page.run_chain(TOOLS, {})
    if read_during_the_first:
        assert _Stub.started["cells"].wait(WAIT_S)
        # A tick drained the report into the page's kept progress.
        progress = page.status().progress
        assert progress is not None
        assert progress.done == 2
        _Stub.release["cells"].set()
    assert _Stub.started["density"].wait(WAIT_S)

    during = page.status()
    assert during.running == "density"
    assert during.progress is None
    assert during.clock is not None
    assert during.clock.first is None
    assert during.clock.last is None
    _Stub.release["density"].set()
    _ended(page)


def test_every_record_of_a_chain_names_the_preset_loaded_at_the_click(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = {"name": "assay"}
    monkeypatch.setattr(
        PlatePage,
        "preset",
        property(
            lambda self: SimpleNamespace(file=SimpleNamespace(name=loaded["name"]))
        ),
    )
    _Stub.behaviour["cells"] = "wait"

    page.run_chain(TOOLS, {})
    assert _Stub.started["cells"].wait(WAIT_S)
    loaded["name"] = "another"  # a preset loaded in a second tab, mid-chain
    _Stub.release["cells"].set()
    _ended(page)

    stages = RunDocument.load(run_doc).stages
    assert [stages[tool].preset for tool in TOOLS] == ["assay"] * 5


def test_close_during_a_chain_stops_it_and_says_so(
    page: PlatePage, run_doc: Path
) -> None:
    _Stub.behaviour["density"] = "wait"

    page.run_chain(TOOLS, {})
    assert _Stub.started["density"].wait(WAIT_S)
    current = page._current_run
    assert current is not None
    closer = threading.Thread(target=page.close, name="test-close")
    closer.start()
    # `close()` takes the stop path (FM-117, FM-70): the event is set before it waits.
    assert current[1].wait(WAIT_S)
    _Stub.release["density"].set()
    closer.join(WAIT_S)
    assert not closer.is_alive()

    assert page.close_ended
    assert len(page.close_lines) == 2
    assert page.close_lines[0].startswith("Density per well told to stop; ")
    assert page.close_lines[1] == (
        "Run all that are left: Density per well stopped; done: Count the cells; "
        "not run: Route, Report, Export for NIS-Elements"
    )
    assert _records(run_doc)["density"] is StageStatus.stopped


def test_a_bug_in_the_chain_is_an_outcome_and_not_an_exception(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(plugin: str, params: Any) -> Any:
        raise ValueError("a bug\nin the chain")

    monkeypatch.setattr(page, "_prepare", broken)

    page.run_chain(TOOLS, {})
    _ended(page)

    # The text, on one line, and never the exception (FM-98); the page is free again.
    assert _outcome(page) == "Run all that are left: ValueError: a bug in the chain"
    chain = page.status().chain
    assert chain is not None
    # The tool whose turn it was is the one that failed: the Results screen then says
    # which, and never reads the chain as a STOP nobody pressed.
    assert chain.ended == (
        ChainEnd(
            "cells", "failed", "Count the cells failed: ValueError: a bug in the chain"
        ),
    )
    assert chain.current is None
    assert _Stub.ran == []


def test_a_bug_after_one_tool_ends_that_tool_failed_and_is_never_read_as_a_stop(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = page._prepare
    calls: list[str] = []

    def breaks_on_the_second(plugin: str, params: Any) -> Any:
        calls.append(plugin)
        if len(calls) == 2:
            raise ValueError("boom")
        return real(plugin, params)

    monkeypatch.setattr(page, "_prepare", breaks_on_the_second)

    page.run_chain(TOOLS, {})
    _ended(page)

    assert _outcome(page) == "Run all that are left: ValueError: boom"
    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("cells", "done"),
        ("density", "failed"),
    ]
    assert chain.ended[-1].outcome == "Density per well failed: ValueError: boom"
    assert chain.current is None
    assert _Stub.ran == ["cells"]


# --- the call itself: what it reads, what it refuses ---------------------------------------


def test_a_run_document_that_cannot_be_read_ends_the_call_with_its_error_and_runs_nothing(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreadable document is an empty one to `_document()`: every tool would read as `not
    # run`, and a tool that is `ok` would be run again (FM-24: a read refused for a moment on
    # Windows, where `_prepare`'s own read a few milliseconds later succeeds).
    page.run("cells", [])
    _ended(page)
    _Stub.ran.clear()
    real = controller_module.RunDocument.load_or_new
    calls: list[int] = []

    def refused_once(path: Path) -> RunDocument:
        calls.append(1)
        if len(calls) == 1:
            raise RunDocumentError(path, "the file is locked by another program")
        return real(path)

    monkeypatch.setattr(
        controller_module.RunDocument, "load_or_new", staticmethod(refused_once)
    )

    page.run_chain(TOOLS, {})

    outcome = _outcome(page)
    assert outcome.startswith("Run all that are left: ")
    assert "the file is locked by another program" in outcome
    assert "\n" not in outcome
    # Nothing started: no activity, no chain, no tool run, and the `ok` one was not run again.
    assert page.status().running is None
    assert page.status().chain is None
    assert _Stub.ran == []


def test_run_all_with_no_step_asked_for_says_so_and_starts_no_activity(
    page: PlatePage,
) -> None:
    # The view passes an empty list when no Results tool loaded; the controller refuses on its
    # own (FM-169), and says what is true: nothing is done, nothing was asked for.
    before = page.activities_ended
    page.run_chain([], {})
    assert _outcome(page) == "Run all that are left: no step was asked for"
    assert page.status().running is None
    assert page.status().chain is None
    assert page.activities_ended == before
    assert _Stub.ran == []


def test_run_all_while_the_stand_is_connecting_is_refused_as_busy_not_as_closed(
    run_doc: Path,
) -> None:
    # The guard is the first line of the call: with it gone, a call during `connect` (the stand
    # is still None) would answer `the stand is not connected` and not `connect is running`.
    gate = threading.Event()
    opened = threading.Event()
    stands: list[Microscope] = []

    def slow_open() -> Microscope:
        opened.set()
        gate.wait(WAIT_S)
        stand = Microscope.from_core(
            cast("Any", FakeCore.demo_like()), Profile.demo(), dry_run=False
        )
        stands.append(stand)
        return stand

    page = PlatePage.closed(
        Profile.demo(), run_doc, opener=slow_open, entry_points=_entry_points()
    )
    try:
        page.connect()
        assert opened.wait(WAIT_S)
        with pytest.raises(RuntimeError, match="connect is running"):
            page.run_chain(TOOLS, {})
        assert page.status().outcome is None
    finally:
        gate.set()
        page.join(WAIT_S)
        page.close()
        for stand in stands:
            stand.close()


def test_a_name_no_plugin_has_is_refused_at_its_turn_as_text(
    page: PlatePage, run_doc: Path
) -> None:
    page.run_chain(["their-tool"], {})
    _ended(page)

    outcome = _outcome(page)
    assert outcome.startswith("Run all that are left: their-tool: ")
    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [("their-tool", "refused")]
    assert "their-tool" not in RunDocument.load(run_doc).stages


@pytest.mark.parametrize("with_result", [False, True])
def test_a_record_that_could_not_be_written_ends_the_tool_failed_and_stops_the_chain(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch, with_result: bool
) -> None:
    # `_run_plugin` has two branches for a run document that refuses a write (FM-34): the stage
    # ran and its result is lost, or it was never recorded. Both are `failed`, and a chain
    # that carried on over either would end `run all done` with no record of the last tool.
    def refuse(*args: Any, **kwargs: Any) -> Any:
        record = None
        if with_result:
            record = StageRecord(
                plugin="cells",
                provides="cells",
                attempt="x",
                status=StageStatus.done,
                started_utc=T0,
                finished_utc=T0,
                profile="demo",
                dry_run=False,
                smc_version="0",
                params={},
                result={},
                reads={},
            )
        raise RunDocumentError(run_doc, "the disk is full", record=record)

    monkeypatch.setattr(controller_module, "run_plugin", refuse)

    page.run_chain(TOOLS, {})
    _ended(page)

    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [("cells", "failed")]
    outcome = _outcome(page)
    assert outcome.startswith("Run all that are left: Count the cells ")
    assert "the disk is full" in outcome
    assert outcome.endswith(
        "; not run: Density per well, Route, Report, Export for NIS-Elements"
    )
    assert ("ran, but its record could not be written" in outcome) is with_result


# --- what the status says from the first tool, and after the chain -----------------------


def test_the_chain_is_visible_from_its_first_tool(page: PlatePage) -> None:
    # `cells` is the slow one on real frames: its row and the chain's line say it goes.
    _Stub.behaviour["cells"] = "wait"

    page.run_chain(TOOLS, {})
    assert _Stub.started["cells"].wait(WAIT_S)
    during = page.status()
    assert during.running == "cells"
    assert during.chain is not None
    assert during.chain.plugins == TOOLS
    assert during.chain.ended == ()
    assert during.chain.current == "cells"
    _Stub.release["cells"].set()
    _ended(page)


def test_another_activity_clears_the_chain_when_it_starts(page: PlatePage) -> None:
    page.run_chain(TOOLS, {})
    _ended(page)
    assert page.status().chain is not None  # kept after its end, as the progress is

    _Stub.behaviour["cells"] = "wait"
    page.run("cells", [])  # *Again* on a tool that is done
    assert _Stub.started["cells"].wait(WAIT_S)
    assert page.status().chain is None  # the chain's line goes with the next activity
    _Stub.release["cells"].set()
    _ended(page)
    assert page.status().chain is None
