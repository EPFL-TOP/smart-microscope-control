"""The Results screen of the guided page, without a browser (m5 §17.4.5, §17.11, #446).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1``, a tick is
``view.tick()``. A click on a *disabled* button still calls its handler on the server (measured
on Panel 1.10.0 and 1.9.4), so the controller's refusals are the guard (FM-169) and a test clicks
a disabled button to prove it. What this cannot show is a browser: that the five rows and the
button read at 1280 x 800, and what is below the fold, are for ``/ui-check`` (FM-161). Panel's
warnings are errors (FM-147), and a handler that failed fails the test that did not expect it.

The five tools are stubs named and ``provides``d as the real ones, with the real ``reads``, over a
seeded plan and scan; each ends as the test says (``done``, ``fail``, ``wait``). The tools whose
fields and commands a test reads are the real ones. Threads meet on ``Event``s with a bound, never
on a sleep (FM-44).
"""

from __future__ import annotations

import html
import os
import re
import shlex
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import unquote

import pytest
from pymmcore_plus import CMMCorePlus

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

import smc.ui.plate_page as plate_page
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    DensityResult,
    DensityWell,
    PlateResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, PlatePage
from smc.ui.guide import (
    DRY_RUN_NO_FRAME,
    FILES_HEADING,
    LAY_OUT_FIRST,
    NO_FILES_YET,
    NOTHING_LEFT,
    RESULT_TOOLS,
    RUN_ALL_LABEL,
    SCAN_FIRST,
    SCREEN_LEADS,
    SCREEN_TITLES,
    again_note,
    local_time,
)
from smc.ui.plate_page import PlatePageView, posix_quoting_note

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
#: Seconds a test waits for the other thread before it gives up: a bound, never a timing (FM-44).
BOUND_S = 10.0
HOSTILE = '<img src=x onerror="alert(1)"> **bold** [x](javascript:y) &amp;'
DEMO_SAMPLE = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"

#: The real tools, by the entry points the page loads them from.
REAL = {
    "cells": "smc.plugins.cells.plugin:CellsPlugin",
    "density": "smc.plugins.density.plugin:DensityPlugin",
    "nis-export": "smc.plugins.nis_export.plugin:NisExportPlugin",
    "report": "smc.plugins.report.plugin:ReportPlugin",
    "route": "smc.plugins.route.plugin:RoutePlugin",
    "scan": "smc.plugins.scan.plugin:ScanPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
}


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class FileResult(PluginResult):
    """A result that names a file, as ``cells``, ``report`` and ``nis-export`` do (``_FILE_KEYS``)."""

    file: str


class _Stub:
    """A tool of the Results screen that ends as ``_Stub.behaviour`` says for its name."""

    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads_optional: tuple[str, ...] = ()
    summary = "A test stand-in for a Results tool."
    name: str
    provides: str
    reads: tuple[str, ...]

    behaviour: ClassVar[dict[str, str]] = {}
    #: The count a tool reports (of 5) before it ends or waits: the row says it while it goes.
    reports: ClassVar[dict[str, int]] = {}
    ran: ClassVar[list[str]] = []
    started: ClassVar[dict[str, threading.Event]] = {}
    release: ClassVar[dict[str, threading.Event]] = {}

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        _Stub.ran.append(self.name)
        if self.name in _Stub.reports:
            ctx.progress(_Stub.reports[self.name], 5, "")
        how = _Stub.behaviour.get(self.name, "done")
        if how == "fail":
            raise PluginRunError(self.name, HOSTILE)
        if how == "wait":
            _Stub.started[self.name].set()
            _Stub.release[self.name].wait(BOUND_S)
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


STUBS: dict[str, type[_Stub]] = {
    cls.name: cls for cls in (Cells, Density, Route, Report, NisExport)
}


def _entry_points(real: tuple[str, ...] = ()) -> list[EntryPoint]:
    """The five tools as stubs, except those named in ``real``; and the real ``scan-plan`` and ``scan``."""
    found = [
        EntryPoint(name=name, value=REAL[name], group=GROUP)
        for name in ("scan-plan", "scan", *real)
    ]
    for name, cls in STUBS.items():
        if name not in real:
            found.append(
                EntryPoint(name=name, value=f"{__name__}:{cls.__name__}", group=GROUP)
            )
    return found


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _Stub.behaviour = {}
    _Stub.reports = {}
    _Stub.ran = []
    _Stub.started = {name: threading.Event() for name in RESULT_TOOLS}
    _Stub.release = {name: threading.Event() for name in RESULT_TOOLS}
    # Presets are read from the working directory's `profiles/presets`.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMC_PROFILES", raising=False)
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

    def make(
        *,
        profile: Profile | None = None,
        connect: bool = True,
        dry_run: bool = False,
        real: tuple[str, ...] = (),
    ) -> PlatePage:
        chosen = profile or Profile.demo()

        def open_stand() -> Microscope:
            stand = Microscope.from_core(
                cast("CMMCorePlus", FakeCore.demo_like()), chosen, dry_run=dry_run
            )
            stands.append(stand)
            return stand

        page = PlatePage.closed(
            chosen,
            run_doc,
            dry_run=dry_run,
            opener=open_stand,
            entry_points=_entry_points(real),
        )
        pages.append(page)
        if connect:
            page.connect()
            assert page.join(BOUND_S), "connect did not end"
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
def page(make_page: Callable[..., PlatePage]) -> PlatePage:
    return make_page()


# --- the run document, as earlier runs left it --------------------------------------------


def _plate_result() -> PlateResult:
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=1,
    )


def _scan_plan_result(wells: tuple[str, ...] = ("B11", "B2")) -> ScanPlanResult:
    """A plan of 2 x 2 fields in each well, in visiting order."""
    fov_um = (512.0, 512.0)
    half_w, half_h = fov_um[0] / 2, fov_um[1] / 2
    plate_plan, fields = plan_fields(
        _plate_result(),
        list(wells),
        fov_um=fov_um,
        footprint_of=lambda x, y: [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ],
        rows=2,
        columns=2,
        overlap=0.1,
        clip_to_well=True,
    )
    return ScanPlanResult(
        plate_plan=plate_plan,
        wells=wells,
        objective="Nikon 10X S Fluor",
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        matrix_source="parameter",
        image_shape_px=(512, 512),
        fov_um=fov_um,
        overlap=0.1,
        rows=2,
        columns=2,
        fields=fields,
        estimate_s=42.16,
    )


def _scan_result(plan: ScanPlanResult) -> ScanResult:
    return ScanResult(
        assumed=(),
        folder="scan",
        objective=plan.objective,
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        image_shape_px=(512, 512),
        exposure_ms=10.0,
        frames=tuple(
            ScanFrame(
                well=field.well,
                index=field.index,
                name=field.name,
                file=f"scan/{field.name}.tif",
                x_um=field.x_um,
                y_um=field.y_um,
                z_um=None,
                focus="none",
                pfs_offset=None,
                t_s=0.0,
            )
            for field in plan.fields
        ),
        failed=(),
    )


def _seed(
    run_doc: Path,
    plugin: str,
    result: PluginResult,
    *,
    minute: int,
    dry_run: bool = False,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=plugin if plugin != "plate-registration" else "plate",
            attempt=f"a{minute}",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
            reads=reads or {},
        ),
    )


def _seed_scan(run_doc: Path, *, dry_run: bool = False) -> ScanPlanResult:
    """A registered plate, a plan and the scan that read it: what the Results screen finds after a scan."""
    plan = _scan_plan_result()
    _seed(
        run_doc,
        "plate-registration",
        _plate_result(),
        minute=0,
        dry_run=dry_run,
    )
    _seed(run_doc, "scan-plan", plan, minute=1, dry_run=dry_run)
    _seed(
        run_doc,
        "scan",
        _scan_result(plan),
        minute=2,
        dry_run=dry_run,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )
    return plan


def _seed_done(
    run_doc: Path, tool: str, *, minute: int, result: PluginResult | None = None
) -> None:
    """A done record of ``tool`` that read nothing, so that nothing makes it out of date."""
    _seed(run_doc, tool, result or NoResult(), minute=minute)


def _objects(items: list[Any]) -> list[Any]:
    """Every object under ``items``, the Accordion's panels too."""
    found: list[Any] = []
    for item in items:
        found.extend(item.select())
        for accordion in item.select(pn.Accordion):
            for panel in accordion.objects:
                found.extend(panel.select())
    return found


def _click(button: Any) -> None:
    button.clicks += 1


def _words(command: str) -> list[str]:
    """A command as a POSIX shell splits it, never compared as the string the code builds (FM-45)."""
    return shlex.split(command)


def _ended(page: PlatePage) -> None:
    assert page.join(BOUND_S), "the activity did not end"


def _states(view: PlatePageView) -> dict[str, str]:
    return {tool: row.state.object for tool, row in view.result_rows.items()}


def _waits(view: PlatePageView) -> dict[str, str]:
    return {tool: row.wait.object for tool, row in view.result_rows.items()}


def _count_views(page: PlatePage) -> list[int]:
    """Replace ``page.view`` by a counting one; the list gets a ``1`` for each read."""
    reads: list[int] = []
    original = page.view

    def counting() -> Any:
        reads.append(1)
        return original()

    page.view = counting  # type: ignore[method-assign]
    return reads


# --- the screen ---------------------------------------------------------------------------


def test_the_results_screen_lists_run_all_the_five_rows_and_the_files_and_no_next_step_section_or_checklist(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    shown = list(view.screens["results"])
    title, lead, *rest = shown
    assert title.object == SCREEN_TITLES["results"] == "Results"
    assert lead.object == SCREEN_LEADS["results"]
    # The action first, so that it is on the page without a scroll (m5 §17.4.5), then the five
    # rows in the checklist's order, the files, the run document's path and *Continue*.
    assert rest == [
        view.run_all_button,
        view.run_all_wait_text,
        view.chain_text,
        *(view.result_rows[tool].box for tool in RESULT_TOOLS),
        view.files_heading,
        view.files_text,
        view.run_document_text,
        view.continue_button,
    ]
    assert list(view.result_rows) == list(RESULT_TOOLS)
    assert view.run_all_button.label == RUN_ALL_LABEL == "Run all that are left"
    assert view.run_all_button.button_type == "primary"
    assert view.files_heading.object == FILES_HEADING == "Where the files are"
    # What the screen no longer carries, on purpose (FM-226): the next step, the tools' own
    # sections and the checklist are the expert view's. The tools' own *Run* are not here.
    objects = _objects(shown)
    for gone in (view.next_text, view.next_button, view.next_note, view.checklist_text):
        assert gone not in objects
    assert not [o for o in objects if isinstance(o, pn.Accordion)]
    assert not [b for b in view.tool_runs.values() if b in objects]
    assert not [o for o in objects if isinstance(o, pn.Card) and o.title == "Advanced"]


def test_the_checklist_and_the_next_step_are_in_the_expert_view_only(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    view.show_expert(True)
    objects = _objects(list(view.step_panel.objects))
    for kept in (
        view.next_text,
        view.next_button,
        view.checklist_text,
        view.run_document_text,
    ):
        assert kept in objects
    # Nor the screen's own buttons: the expert view keeps each tool's own *Run*.
    assert view.run_all_button not in objects
    assert not [
        row.button for row in view.result_rows.values() if row.button in objects
    ]
    assert all(button in objects for button in view.tool_runs.values())
    view.show_step("results")
    objects = _objects(list(view.step_panel.objects))
    assert view.checklist_text not in objects
    assert view.next_text not in objects
    assert view.run_document_text in objects


# --- a row ----------------------------------------------------------------------------------


def test_a_row_per_tool_in_checklist_order_with_the_plain_names_and_the_states_of_the_run_document(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page()
    view = PlatePageView(page)
    assert [row.box.objects[0].object for row in view.result_rows.values()] == [
        "Count the cells",
        "Density per well",
        "Route through the densest fields",
        "Report",
        "Export for NIS-Elements",
    ]
    assert all(
        row.box.objects[0].styles["font-weight"] == "bold"
        for row in view.result_rows.values()
    )
    # Nothing done: every row says which step makes its input, in the operator's words.
    assert _states(view) == {
        "cells": SCAN_FIRST,
        "density": LAY_OUT_FIRST,
        "route": LAY_OUT_FIRST,
        "report": LAY_OUT_FIRST,
        "nis-export": "after Route through the densest fields",
    }
    # After the scan: `cells` is ready and the others wait for what they read.
    _seed_scan(run_doc)
    view.refresh()
    assert _states(view) == {
        "cells": "Ready",
        "density": "after Count the cells",
        "route": "after Count the cells",
        "report": "after Count the cells",
        "nis-export": "after Route through the densest fields",
    }
    # `cells` and `density` done: the time of each record, the next ones ready.
    _seed_done(run_doc, "cells", minute=3)
    _seed_done(run_doc, "density", minute=4)
    view.refresh()
    assert _states(view) == {
        # The PC's own time to the minute (m5 §17.12 D20): the test asks the same clock.
        "cells": f"Done {local_time(T0 + timedelta(minutes=3))}",
        "density": f"Done {local_time(T0 + timedelta(minutes=4))}",
        "route": "Ready",
        "report": "Ready",
        "nis-export": "after Route through the densest fields",
    }
    assert {tool: row.button.label for tool, row in view.result_rows.items()} == {
        "cells": "Again",
        "density": "Again",
        "route": "Run",
        "report": "Run",
        "nis-export": "Run",
    }
    # The document's own words, for a terminal, are on no row.
    for text in (*_states(view).values(), *_waits(view).values()):
        assert "run document" not in text
        assert "smc run" not in text


def test_a_result_that_is_not_usable_says_why_in_its_row_and_offers_run(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_scan(run_doc)
    view = PlatePageView(page)
    for tool in ("cells", "density"):
        view.on_run(tool)
        _ended(page)
    view.tick()
    assert view.result_rows["density"].button.label == "Again"
    # `cells` again makes what read it out of date: the document's reason is the row's state,
    # the row offers *Run*, and the row that was waiting on nothing still says `Done`.
    view.on_run("cells")
    _ended(page)
    view.tick()
    assert view.result_rows["cells"].button.label == "Again"
    assert (
        view.result_rows["density"].state.object
        == "Not usable: density read its inputs from an earlier run; run it again"
    )
    assert view.result_rows["density"].button.label == "Run"
    assert view.result_rows["density"].button.disabled is False


# --- what a disabled button says, and who refuses ------------------------------------------


def test_a_row_that_cannot_run_says_why_and_a_click_on_it_is_refused_by_the_controller(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # FM-169: a click on a disabled button still arrives at the controller, which refuses on
    # its own and writes nothing.
    _seed_scan(run_doc)
    closed = make_page(connect=False)
    view = PlatePageView(closed)
    for tool in RESULT_TOOLS:
        row = view.result_rows[tool]
        assert row.button.disabled is True, tool
    assert view.result_rows["cells"].wait.object == NOT_CONNECTED
    assert view.result_rows["cells"].wait.visible is True
    assert view.run_all_button.disabled is True
    assert view.run_all_wait_text.object == NOT_CONNECTED
    _click(view.result_rows["cells"].button)
    assert closed.status().outcome == f"cells: {NOT_CONNECTED}"
    _click(view.run_all_button)
    assert closed.status().outcome == f"Run all that are left: {NOT_CONNECTED}"
    assert _Stub.ran == []

    page = make_page()
    view = PlatePageView(page)
    # Connected: `cells` can run, `density` waits for it and says so, with no wait of its own.
    assert view.result_rows["cells"].button.disabled is False
    assert view.result_rows["density"].button.disabled is True
    assert view.result_rows["density"].state.object == "after Count the cells"
    assert view.result_rows["density"].wait.visible is False
    _click(view.result_rows["density"].button)
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith("density: UpstreamError: ")
    assert page.status().running is None
    assert _Stub.ran == []
    assert "density" not in RunDocument.load(run_doc).stages


def test_run_all_waits_for_the_first_tool_left_and_a_click_on_it_is_refused_by_the_controller(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # Nothing was scanned: the chain would stop at once at `cells`, and the button says why.
    page = make_page()
    view = PlatePageView(page)
    assert view.run_all_button.disabled is True
    assert view.run_all_wait_text.object == SCAN_FIRST
    assert view.run_all_wait_text.visible is True
    _click(view.run_all_button)
    # The chain starts and refuses at its first tool's turn, with the text `run` gives it.
    _ended(page)
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith("Run all that are left: cells: UpstreamError: ")
    assert _Stub.ran == []
    # After the scan it is enabled and says nothing (the tick that follows a run's end).
    _seed_scan(run_doc)
    view.tick()
    view.refresh()
    assert view.run_all_button.disabled is False
    assert view.run_all_wait_text.visible is False


def test_a_tool_that_did_not_load_says_so_in_its_row_and_is_not_passed_to_the_chain(
    make_page: Callable[..., PlatePage], run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(run_doc)
    page = make_page()
    # `route` fails to load: its section has no `Run`, so it has no entry in `tool_runs`.
    original = page.tools
    monkeypatch.setattr(
        page,
        "tools",
        lambda: tuple(tool for tool in original() if tool.name != "route"),
    )
    view = PlatePageView(page)
    assert "route" not in view.tool_runs
    row = view.result_rows["route"]
    assert row.button.disabled is True
    assert row.wait.object == "route did not load: the problems line says why"
    assert row.card.visible is False
    passed: list[Any] = []
    real = page.run_chain
    monkeypatch.setattr(
        page,
        "run_chain",
        lambda tools, params: (passed.append(list(tools)), real(tools, params)),
    )
    _click(view.run_all_button)
    _ended(page)
    assert passed == [["cells", "density", "report", "nis-export"]]
    assert "route" not in _Stub.ran


# --- the choices and the command -----------------------------------------------------------


def test_a_rows_run_is_the_tools_own_handler_with_the_typed_choices(
    make_page: Callable[..., PlatePage], run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(run_doc)
    # The real `cells` and `route`, on the stand whose profile suggests a detector (m5 §8.1).
    page = make_page(profile=Profile.load(DEMO_SAMPLE), real=("cells", "route"))
    view = PlatePageView(page)

    cells = view.result_rows["cells"]
    detector = view.tool_fields["cells"]["detector"]
    # One instance: the field of the tool's own section (the expert view's) is the row's.
    assert detector in cells.card.objects
    assert detector in _objects(view.screens["expert"])
    assert detector.value == "blobs"
    assert cells.from_lines["detector"] in cells.card.objects
    assert cells.from_lines["detector"].object == "from the profile's suggestions"
    assert cells.card.title == "Choices and command"
    assert cells.card.collapsed is True

    route = view.result_rows["route"]
    labels = view.tool_fields["route"]["labels"]
    per_well = view.tool_fields["route"]["fields_per_well"]
    assert [o for o in route.card.objects if isinstance(o, pn.widgets.TextInput)] == [
        labels,
        per_well,
    ]
    assert per_well in _objects(view.screens["expert"])
    assert route.from_lines["fields_per_well"].object.startswith("the tool's default")
    # A typed choice is in the row's command, and its line says where it comes from.
    per_well.value = "3"
    assert route.from_lines["fields_per_well"].object == "typed on this page"
    assert _words(route.command.object)[:3] == ["smc", "run", "route"]
    assert _words(route.command.object)[-2:] == ["--param", "fields_per_well=3"]

    # The row's button makes the very call of the tool's own *Run* (the expert view's).
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        page, "run", lambda plugin, params: calls.append((plugin, list(params)))
    )
    _click(route.button)
    _click(view.tool_runs["route"])
    _click(cells.button)
    _click(view.tool_runs["cells"])
    assert calls == [
        ("route", ["fields_per_well=3"]),
        ("route", ["fields_per_well=3"]),
        ("cells", ["detector=blobs"]),
        ("cells", ["detector=blobs"]),
    ]


def test_run_all_sends_each_tools_typed_choices_to_the_chain(
    make_page: Callable[..., PlatePage], run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(run_doc)
    page = make_page(profile=Profile.load(DEMO_SAMPLE), real=("cells", "route"))
    view = PlatePageView(page)
    view.tool_fields["route"]["fields_per_well"].value = "3"
    sent: list[Any] = []
    monkeypatch.setattr(
        page,
        "run_chain",
        lambda tools, params: sent.append((list(tools), dict(params))),
    )
    _click(view.run_all_button)
    assert sent == [
        (
            ["cells", "density", "route", "report", "nis-export"],
            {
                "cells": ["detector=blobs"],
                "density": [],
                "route": ["fields_per_well=3"],
                "report": [],
                "nis-export": [],
            },
        )
    ]


def test_every_command_card_of_the_rows_carries_the_windows_note(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-175: a page served on Windows says its commands are quoted for a POSIX shell, under
    # each *Command* of a row, which replaced the tool's section (FM-226).
    _seed_scan(run_doc)
    view = PlatePageView(page)
    view._windows = True
    view.refresh()
    note = posix_quoting_note(True)
    assert note
    for tool, row in view.result_rows.items():
        assert row.card.visible is True, tool
        assert row.command.object.startswith("smc run "), tool
        assert _words(row.command.object)[:3] == ["smc", "run", tool]
        assert row.command_note.object == note
        assert row.command_note.visible is True
        assert row.command_note in row.card.objects
        assert row.command in row.card.objects
    # Not a Windows page: no note, and no height for it.
    view._windows = False
    view.refresh()
    assert all(not row.command_note.visible for row in view.result_rows.values())


# --- the chain, as the page draws it --------------------------------------------------------


def test_run_all_redraws_the_rows_at_each_tools_end_and_reads_the_view_once_per_end(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-145, FM-151: a tool's end inside a chain changes the run document with no activity
    # ending, so the tick reads the view again for it, once; the chain's last end is also the
    # run's end, and one read covers both.
    _seed_scan(run_doc)
    _Stub.behaviour["density"] = "wait"
    view = PlatePageView(page)
    view.show_step("results")
    reads = _count_views(page)

    _click(view.run_all_button)
    assert _Stub.started["density"].wait(BOUND_S)
    view.tick()
    assert reads == [1]  # `cells` ended: its row is drawn from a new read
    states = _states(view)
    assert states["cells"].startswith("Done 2026-")
    assert states["density"] == "Running now"
    assert view.chain_text.object == "Running Density per well (2 of 5)."
    assert view.chain_text.visible is True
    assert view.run_all_button.disabled is True
    assert (
        view.run_all_wait_text.object
        == "density is running: wait for it to end, or press STOP."
    )
    assert all(row.button.disabled for row in view.result_rows.values())
    # What runs holds for every button: it is said once, under *Run all that are left*, and
    # not under each row (m5 §17.12 D19). A row's wait is its own reason only.
    assert all(row.wait.object == "" for row in view.result_rows.values())
    assert all(row.wait.visible is False for row in view.result_rows.values())
    view.tick()
    view.tick()
    assert reads == [1]  # nothing ended since: no read

    _Stub.release["density"].set()
    _ended(page)
    view.tick()
    assert reads == [1, 1]  # four tools ended between the two ticks: one read, not two
    assert [state.startswith("Done ") for state in _states(view).values()] == [True] * 5
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith("Run all that are left: done in ")
    assert view.chain_text.object == (
        "Done: Count the cells, Density per well, Route through the densest fields, "
        "Report, Export for NIS-Elements."
    )
    assert view.step_buttons["results"].label == "5. Results ✓"
    # Nothing is left: the button says so, and nothing runs when it is clicked.
    assert view.run_all_button.disabled is True
    assert view.run_all_wait_text.object == NOTHING_LEFT
    _Stub.ran.clear()
    _click(view.run_all_button)
    assert page.status().outcome == (
        "Run all that are left: nothing is left: every step is done"
    )
    assert _Stub.ran == []


def test_a_second_tab_draws_the_chain_the_first_started(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-151: the chain is the controller's; a tab opened mid-chain draws it from the shared
    # status at once, and its rows follow the end.
    _seed_scan(run_doc)
    _Stub.behaviour["route"] = "wait"
    first = PlatePageView(page)
    first.show_step("results")
    _click(first.run_all_button)
    assert _Stub.started["route"].wait(BOUND_S)
    second = PlatePageView(page)
    second.show_step("results")
    first.tick()
    second.tick()
    for view in (first, second):
        assert (
            view.chain_text.object
            == "Running Route through the densest fields (3 of 5)."
        )
        assert view.result_rows["route"].state.object == "Running now"
        assert view.result_rows["cells"].state.object.startswith("Done ")
        assert view.run_all_button.disabled is True
    _Stub.release["route"].set()
    _ended(page)
    first.tick()
    second.tick()
    for view in (first, second):
        assert view.chain_text.object.startswith("Done: Count the cells")
        assert all(
            row.state.object.startswith("Done ") for row in view.result_rows.values()
        )
        assert view.step_buttons["results"].label == "5. Results ✓"


@pytest.mark.parametrize(
    ("how", "line"),
    [
        (
            "stop",
            "Stopped at Route through the densest fields by STOP. Done: Count the cells, "
            "Density per well. Not run: Report, Export for NIS-Elements.",
        ),
        (
            "fail",
            # The tool's own outcome is the head: it says which tool and that it failed.
            f"Route failed: route: {HOSTILE}. Done: Count the cells, Density per well. "
            "Not run: Report, Export for NIS-Elements.",
        ),
    ],
)
def test_the_chain_line_names_the_tool_that_stopped_it_and_stop_from_the_page_ends_the_chain(
    page: PlatePage, run_doc: Path, how: str, line: str
) -> None:
    _seed_scan(run_doc)
    _Stub.behaviour["route"] = "wait" if how == "stop" else "fail"
    view = PlatePageView(page)
    view.show_step("results")
    _click(view.run_all_button)
    if how == "stop":
        assert _Stub.started["route"].wait(BOUND_S)
        view.on_stop()  # the strip's own STOP
        _Stub.release["route"].set()
    _ended(page)
    view.tick()
    assert view.chain_text.object == line
    assert view.chain_text.visible is True
    # The tools after it never ran, and their rows say what they wait for.
    assert _Stub.ran == ["cells", "density", "route"]
    states = _states(view)
    assert states["cells"].startswith("Done ")
    assert states["density"].startswith("Done ")
    assert states["route"].startswith("Not usable: ")
    assert states["report"] == "Ready"
    assert states["nis-export"] == "after Route through the densest fields"
    # The route's row offers *Run* again, and *Run all* is enabled: it runs what is left.
    assert view.result_rows["route"].button.label == "Run"
    assert view.result_rows["route"].button.disabled is False
    assert view.run_all_button.disabled is False
    # A STOP stopped the stand: the strip says it, and the chain's line does not need to.
    status = page.status()
    assert status.outcome is not None
    assert status.outcome.startswith("Run all that are left: Route ")


def test_again_says_what_it_replaces_and_the_note_goes_with_the_tool(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_scan(run_doc)
    view = PlatePageView(page)
    view.show_step("results")
    _click(view.run_all_button)
    _ended(page)
    view.tick()
    cells = view.result_rows["cells"]
    assert cells.button.label == "Again"
    assert cells.replaces.visible is True
    expected = again_note(
        "cells",
        [item for item in view._view.checklist],  # type: ignore[union-attr]
    )
    assert cells.replaces.object == expected
    assert "Density, Route, Report, Export" in expected
    # What no later tool reads says nothing, and takes no height.
    for tool in ("report", "nis-export"):
        assert view.result_rows[tool].replaces.object == ""
        assert view.result_rows[tool].replaces.visible is False
    assert "makes Report out of date" in view.result_rows["density"].replaces.object

    # *Again* on `cells` runs the tool, and what read it is out of date: the note is gone (no
    # tool after it is done), and the chain button runs the four again.
    _click(cells.button)
    _ended(page)
    view.tick()
    assert cells.replaces.visible is False
    # The chain's line goes with the next activity: it said `Done: ...` about a result that is
    # no longer the document's (the rows say Run for four tools).
    assert page.status().chain is None
    assert view.chain_text.visible is False
    assert [view.result_rows[t].button.label for t in RESULT_TOOLS] == [
        "Again",
        "Run",
        "Run",
        "Run",
        "Run",
    ]
    assert view.run_all_button.disabled is False
    _Stub.ran.clear()
    _click(view.run_all_button)
    _ended(page)
    assert _Stub.ran == ["density", "route", "report", "nis-export"]
    view.tick()
    assert [view.result_rows[t].button.label for t in RESULT_TOOLS] == ["Again"] * 5


def test_a_dry_run_page_over_a_dry_scan_holds_cells_back_with_the_dry_run_text(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # FM-226: what the next step said (m5 §8.8.1) is the row's wait, and the chain's.
    _seed_scan(run_doc, dry_run=True)
    page = make_page(dry_run=True)
    view = PlatePageView(page)
    view.show_step("results")
    cells = view.result_rows["cells"]
    assert cells.state.object == "Ready"
    assert cells.wait.object == DRY_RUN_NO_FRAME
    assert cells.wait.visible is True
    assert cells.button.disabled is True
    assert view.run_all_button.disabled is True
    assert view.run_all_wait_text.object == DRY_RUN_NO_FRAME
    # A tool whose input is missing says that, and not the dry run's text as well.
    assert view.result_rows["density"].state.object == "after Count the cells"
    assert view.result_rows["density"].wait.visible is False
    # `nis-export` reads the route and no scan: it has nothing to hold back.
    assert DRY_RUN_NO_FRAME not in view.result_rows["nis-export"].wait.object
    # The controller's `_blocked` is unchanged: the expert view's own *Run cells* stays enabled.
    assert page.view().blocked.get("cells") is None
    assert view.tool_runs["cells"].disabled is False
    # And a page that is not a dry run says none of it.
    real = make_page()
    real_view = PlatePageView(real)
    assert all(DRY_RUN_NO_FRAME not in wait for wait in _waits(real_view).values())


def test_a_dry_run_page_over_a_real_scan_holds_nothing_back(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # What `/ui-check` serves: a page with --dry-run over the records of a real chain. The
    # frames are there, so the dry-run text is not true of this page and is not said.
    _seed_scan(run_doc)
    page = make_page(dry_run=True)
    view = PlatePageView(page)
    view.show_step("results")
    cells = view.result_rows["cells"]
    assert cells.state.object == "Ready"
    assert cells.wait.visible is False
    assert cells.button.disabled is False
    assert view.run_all_button.disabled is False
    assert view.run_all_wait_text.visible is False
    assert all(DRY_RUN_NO_FRAME not in wait for wait in _waits(view).values())


def test_a_running_row_says_the_tools_own_count_and_no_other_row_does(
    page: PlatePage, run_doc: Path
) -> None:
    # `Running now: <done> of <total>` is the tool's own report (m5 §17.11.3): the only live
    # word of a `cells` over real frames, which takes minutes.
    _seed_scan(run_doc)
    _Stub.reports["cells"] = 2
    _Stub.behaviour["cells"] = "wait"
    view = PlatePageView(page)
    view.show_step("results")

    _click(view.run_all_button)
    assert _Stub.started["cells"].wait(BOUND_S)
    view.tick()
    assert _states(view)["cells"] == "Running now: 2 of 5"
    assert _states(view)["density"] == "after Count the cells"
    assert view.chain_text.object == "Running Count the cells (1 of 5)."
    _Stub.release["cells"].set()
    _ended(page)


def test_a_bug_in_the_chain_is_said_as_the_tools_failure_and_never_as_a_stop(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(run_doc)
    real = page._prepare
    calls: list[str] = []

    def breaks_on_the_second(plugin: str, params: Any) -> Any:
        calls.append(plugin)
        if len(calls) == 2:
            raise ValueError("boom")
        return real(plugin, params)

    monkeypatch.setattr(page, "_prepare", breaks_on_the_second)
    view = PlatePageView(page)
    view.show_step("results")
    _click(view.run_all_button)
    _ended(page)
    view.tick()

    assert view.chain_text.object == (
        "Density per well failed: ValueError: boom. "
        "Done: Count the cells. Not run: Route through the densest fields, Report, "
        "Export for NIS-Elements."
    )
    assert "STOP" not in view.chain_text.object
    outcome = page.status().outcome
    assert outcome == "Run all that are left: ValueError: boom"


def test_no_results_tool_loaded_says_so_under_run_all_and_disables_it(
    make_page: Callable[..., PlatePage], monkeypatch: pytest.MonkeyPatch
) -> None:
    page = make_page()
    monkeypatch.setattr(page, "tools", lambda: ())
    view = PlatePageView(page)
    view.show_step("results")
    assert view.run_all_button.disabled is True
    assert (
        view.run_all_wait_text.object
        == "the results tools did not load: the problems line says why"
    )
    assert all(row.button.disabled for row in view.result_rows.values())
    # A click that arrives anyway is the controller's to answer, with what is true.
    _click(view.run_all_button)
    assert page.status().outcome == "Run all that are left: no step was asked for"


def test_the_colour_bar_follows_the_density_select(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    assert view.colour_bar.title is None  # no class: no title
    density = DensityResult(
        labels=("live", "dead"),
        area_grid_um=10.0,
        wells=tuple(
            DensityWell(
                well=well,
                fields=1,
                failed_fields=0,
                area_mm2=2.0,
                counts={"live": n, "dead": 1},
                outside_well=0,
                per_mm2={"live": n / 2.0, "dead": 0.5},
            )
            for well, n in (("A1", 2), ("A2", 8))
        ),
    )
    _seed_scan(run_doc)
    _seed(run_doc, "density", density, minute=5)
    view.refresh()
    assert view.label_select.value == "live"
    assert view.colour_bar.title == "live cells per mm²"
    view.label_select.value = "dead"
    assert view.colour_bar.title == "dead cells per mm²"


def test_the_tick_draws_the_results_screen_without_reading_the_view(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-145: a tick reads the stand once, never the run document.
    _seed_scan(run_doc)
    view = PlatePageView(page)
    view.show_step("results")
    reads = _count_views(page)
    try:
        for _ in range(3):
            view.tick()
    finally:
        del page.view
    assert reads == []
    assert view.result_rows["cells"].state.object == "Ready"


# --- guards and plain text -----------------------------------------------------------------


def test_every_handler_the_results_buttons_register_is_guarded(
    page: PlatePage, handler_failures: list[Any]
) -> None:
    # FM-171: a handler runs under the document lock and never raises into Panel. The guard
    # test of test_plate_page.py walks the layout and `screens`; the new buttons are walked
    # here by name too, so that a button added without `_handler` is found by this module.
    view = PlatePageView(page)
    buttons = [view.run_all_button, *(row.button for row in view.result_rows.values())]
    for button in buttons:
        registered = [
            watcher.fn
            for by_name in button.param.watchers.values()
            for watchers in by_name.values()
            for watcher in watchers
        ]
        ours = [
            fn
            for fn in registered
            if getattr(fn, "__module__", "") == plate_page.__name__
        ]
        assert ours, button.label
        assert [fn for fn in ours if not hasattr(fn, "__wrapped__")] == [], button.label
    # A handler that raises is the page's message and no more.
    view.on_run_all = lambda: (_ for _ in ()).throw(ValueError("boom"))  # type: ignore[method-assign]
    _click(view.run_all_button)
    assert view.message.object.startswith(
        "that action failed on the page: ValueError: boom"
    )
    # The one failure this test expects, and no other.
    assert len(handler_failures) == 1
    handler_failures.clear()


def test_every_text_pane_of_the_results_screen_is_plain_text_whatever_a_record_says(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-139. Free text reaches this screen from a result (a file's path), from a tool's
    # failure (its message) and from a class (the density select): it is shown as it is,
    # never as markup, or a path of `<img onerror=...>` runs on the stand PC.
    _seed_scan(run_doc)
    _seed_done(run_doc, "cells", minute=3, result=FileResult(file=HOSTILE))
    _Stub.behaviour["density"] = "fail"
    view = PlatePageView(page)
    view.show_step("results")
    _click(view.run_all_button)
    _ended(page)
    view.tick()
    # The one HTML pane of the page (m5 §17.12 D21): every text in it is escaped, so the file
    # name that is markup is shown as text and is never the pane's own.
    assert type(view.files_text) is pn.pane.HTML
    assert html.escape(HOSTILE) in view.files_text.object
    assert HOSTILE not in view.files_text.object
    assert "<img" not in view.files_text.object
    assert HOSTILE in view.chain_text.object
    assert view.chain_text.object.startswith("Density per well failed: density: ")
    shown = [
        o
        for o in _objects(list(view.step_panel.objects))
        if isinstance(o, pn.pane.PaneBase)
        and o is not view.files_text
        # A card draws its title as an HTML pane of its own, from a constant of the page's.
        and not (
            type(o) is pn.pane.HTML
            and o.object
            in {
                f"<h3>{c.title}</h3>"
                for c in _objects(list(view.step_panel.objects))
                if isinstance(c, pn.Card)
            }
        )
    ]
    # Counted, so that a pane that becomes Markdown is a different count and not a pass.
    assert len(shown) >= 7 + 4 * len(RESULT_TOOLS)
    assert [type(o).__name__ for o in shown if type(o) is not pn.pane.Str] == []


def test_no_file_is_named_before_a_step_wrote_one(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.files_text.object == f"<div>{NO_FILES_YET}</div>"


def test_each_files_line_names_its_step_and_its_path(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_scan(run_doc)
    _seed_done(run_doc, "cells", minute=3, result=FileResult(file="cells/cells.json"))
    page = make_page()
    view = PlatePageView(page)
    # The scan's folder stays a path; a file of the run folder is a link that opens it in a new
    # tab, through the page's own route (m5 §17.12 D21).
    assert view.files_text.object == (
        f"<div>Scan frames: {html.escape(str(run_doc.parent / 'scan'))}</div>"
        '<div>Count the cells: <a href="/run-files/cells/cells.json" '
        'target="_blank" rel="noopener">cells/cells.json</a></div>'
    )


@pytest.mark.demo
def test_a_scan_runs_through_to_the_export_from_one_click(
    mm_available: bool, tmp_path: Path
) -> None:
    # `demo-sample` served without --dry-run: its stage is simulated, the scan takes real frames
    # of the synthetic plate, and the five tools are the real ones. The steps before the scan
    # are the screens' own buttons (the walk of the Wells and Scan screens); the results are
    # ONE click on *Run all that are left*, with the profile's `detector=blobs` already in the
    # field. No tool's own button is pressed.
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), run_doc)
    try:
        view = PlatePageView(page)
        _click(view.connect_button)
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome
        view.tick()
        view.show_step("plate")
        _click(view.register_button)
        assert page.join(60.0), "plate-registration did not end"
        view.tick()
        view.show_step("wells")
        view.select_input.value = "B2,B11"
        _click(view.layout_button)
        assert page.join(60.0), "scan-plan did not end"
        view.tick()
        view.show_step("scan")
        _click(view.scan_button)
        assert page.join(120.0), "scan did not end"
        view.tick()
        assert view.step_buttons["scan"].label == "4. Scan ✓"

        # 5. Results: nothing ran yet, the first row is ready and the button is on.
        view.show_step("results")
        assert view.tool_fields["cells"]["detector"].value == "blobs"
        assert view.result_rows["cells"].state.object == "Ready"
        assert view.run_all_button.disabled is False, view.run_all_wait_text.object
        assert view.files_text.object.startswith("<div>Scan frames: ")
        _click(view.run_all_button)
        assert page.join(180.0), "the chain did not end"
        view.tick()
        outcome = page.status().outcome
        assert outcome is not None
        # No number asserted: it is a duration (FM-197).
        assert outcome.startswith("Run all that are left: done in "), outcome
        assert outcome.endswith(
            " s (Count the cells, Density per well, Route, Report, Export for NIS-Elements)"
        )
        assert [
            row.state.object.startswith("Done ") for row in view.result_rows.values()
        ] == [True] * 5
        assert view.step_buttons["results"].label == "5. Results ✓"
        assert view.chain_text.object.startswith("Done: Count the cells, ")
        # The report and the export were written, and the files block names them.
        # ... as links, and what each link asks the route for is served (D21).
        links = re.findall(
            r'<div>([^<:]+): <a href="/run-files/([^"]+)"', view.files_text.object
        )
        named = {step: unquote(href) for step, href in links}
        for step in ("Report", "Export for NIS-Elements"):
            assert page.run_file(named[step]) is not None, (step, named)
            assert (run_doc.parent / named[step]).exists(), (step, named[step])
        # Nothing is left: the button says so and is off.
        assert view.run_all_button.disabled is True
        assert view.run_all_wait_text.object == NOTHING_LEFT
        doc = RunDocument.load(run_doc)
        assert all(doc.provider(slot) is not None for slot in RESULT_TOOLS)
    finally:
        page.close()
        page.release()


# --- round 2: a reason said once, the chain's line after its rows, the files that open (m5 §17.12, #475) ---


def test_the_busy_note_is_drawn_under_run_all_only(
    page: PlatePage, run_doc: Path
) -> None:
    # #446's Report: `density is running: wait for it to end, or press STOP.` six times while a
    # chain went. A reason common to every button of a screen is said once, under the first
    # (D19); a row says only what is its own.
    _seed_scan(run_doc)
    _Stub.behaviour["density"] = "wait"
    view = PlatePageView(page)
    view.show_step("results")
    _click(view.run_all_button)
    assert _Stub.started["density"].wait(BOUND_S)
    view.tick()
    busy = "density is running: wait for it to end, or press STOP."
    try:
        texts = [
            o.object
            for o in _objects(list(view.step_panel.objects))
            if isinstance(o, pn.pane.Str) and o.visible
        ]
        assert view.run_all_wait_text.object == busy
        assert sum(busy in text for text in texts) == 1
        assert all(row.wait.object == "" for row in view.result_rows.values())
    finally:
        _Stub.release["density"].set()
        _ended(page)
    # A row's own reason is still its own: an input that is missing is said by its state line,
    # and a tool the stand cannot run by its wait.
    view.tick()
    assert all(row.wait.object == "" for row in view.result_rows.values())


def test_the_chain_line_and_the_rows_change_in_the_same_tick(
    page: PlatePage, run_doc: Path
) -> None:
    # The chain's line is the status's and the rows are the view's: drawn first, the line said
    # `Running Density per well (2 of 5)` over a `cells` row that still said `Ready` (#446's
    # Report). Within one tick the rows are redrawn first and the line after them.
    _seed_scan(run_doc)
    _Stub.behaviour["density"] = "wait"
    view = PlatePageView(page)
    view.show_step("results")
    events: list[str] = []
    view.result_rows["cells"].state.param.watch(
        lambda event: events.append(f"row:{event.new}"), "object"
    )
    view.chain_text.param.watch(
        lambda event: events.append(f"line:{event.new}"), "object"
    )
    _click(view.run_all_button)
    assert _Stub.started["density"].wait(BOUND_S)
    events.clear()
    try:
        view.tick()
        row = next(i for i, e in enumerate(events) if e.startswith("row:Done "))
        line = next(
            i
            for i, e in enumerate(events)
            if e == "line:Running Density per well (2 of 5)."
        )
        assert row < line, events
    finally:
        _Stub.release["density"].set()
        _ended(page)


def test_where_the_files_are_links_the_report_and_the_export(
    page: PlatePage, run_doc: Path
) -> None:
    run = run_doc.parent
    (run / "report").mkdir()
    (run / "report" / "abc.html").write_text("<p>report</p>", encoding="utf-8")
    (run / "export").mkdir()
    (run / "export" / "list.xml").write_text("<x/>", encoding="utf-8")
    _seed_scan(run_doc)
    _seed_done(run_doc, "cells", minute=3)
    _seed_done(run_doc, "density", minute=4)
    _seed_done(run_doc, "route", minute=5)
    _seed_done(run_doc, "report", minute=6, result=FileResult(file="report/abc.html"))
    _seed_done(
        run_doc, "nis-export", minute=7, result=FileResult(file="export/list.xml")
    )
    view = PlatePageView(page)
    view.show_step("results")

    links = re.findall(
        r'<a href="([^"]+)" target="_blank" rel="noopener">', view.files_text.object
    )

    assert links == ["/run-files/report/abc.html", "/run-files/export/list.xml"]
    # Each link asks the route for a file the controller serves, and nothing else is a link.
    for href in links:
        served = page.run_file(unquote(href.removeprefix("/run-files/")))
        assert served is not None
        assert served.is_file()
    assert "Scan frames: " in view.files_text.object
    assert view.files_text.object.count("<a ") == 2
