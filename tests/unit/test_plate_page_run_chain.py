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

from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
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
    assert outcome.startswith("run all done in ")
    assert outcome.endswith(" s: cells, density, route, report, nis-export")
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
    assert _outcome(page).endswith(" s: density, route, report, nis-export")


def test_a_failed_tool_ends_the_chain_and_the_outcome_names_it_with_what_was_done_and_not_run(
    page: PlatePage, run_doc: Path
) -> None:
    _Stub.behaviour["route"] = "fail"

    page.run_chain(TOOLS, {})
    _ended(page)

    assert _Stub.ran == ["cells", "density", "route"]
    assert _outcome(page) == (
        "run all stopped at route: route: PluginRunError: route: it broke; "
        "done: cells, density; not run: report, nis-export"
    )
    chain = page.status().chain
    assert chain is not None
    assert [(end.plugin, end.how) for end in chain.ended] == [
        ("cells", "done"),
        ("density", "done"),
        ("route", "failed"),
    ]
    assert chain.ended[-1].outcome == "route: PluginRunError: route: it broke"
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
        f"run all stopped at route: {single}; done: cells, density; "
        "not run: report, nis-export"
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
        f"run all stopped at route: {single}; not run: report, nis-export"
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
        "run all stopped at route: route stopped; done: cells, density; "
        "not run: report, nis-export"
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
    assert page.status().stop_note.startswith("route told to stop; ")  # type: ignore[union-attr]


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
        "run all stopped before route; done: cells, density; not run: report, nis-export"
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
    assert closed.status().outcome == f"run all: {NOT_CONNECTED}"
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
    assert connected.status().outcome == "run all: nothing is left: every step is done"
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
    assert page.close_lines[0].startswith("density told to stop; ")
    assert page.close_lines[1] == (
        "run all stopped at density: density stopped; done: cells; "
        "not run: route, report, nis-export"
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
    assert _outcome(page) == "run all: ValueError: a bug in the chain"
    chain = page.status().chain
    assert chain is not None
    assert chain.ended == ()
    assert chain.current is None
    assert _Stub.ran == []
