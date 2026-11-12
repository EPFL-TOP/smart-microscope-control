"""The Wells and Scan screens of the guided page, without a browser (m5 §17.4.3, §17.4.4, §17.10, #445).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1``, a typed
range is ``select_input.value``, a tick is ``view.tick()``. A click on a *disabled* button still
calls its handler on the server (measured on Panel 1.10.0 and 1.9.4), so the controller's
refusals are the guard (FM-169) and a test clicks a disabled button to prove it. What this cannot
show is a browser: that the green fields read at the plate's scale, the layout at 1280 x 800 and
that the title and the first action are inside the page are for ``/ui-check`` (FM-161). Panel's
warnings are errors (FM-147), and a handler that failed fails the test that did not expect it.

The scan is a stub that reports each field of the plan and waits on an ``Event``; the clock the
controller stamps its reports with is the test's own (FM-44: the name patched in the module,
inside ``monkeypatch.context()``), so that *Left* is asserted as text and not as a timing.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, cast

import pytest
from pymmcore_plus import CMMCorePlus

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from bokeh.models import Line, Rect

import smc.ui.controller as controller_module
import smc.ui.plate_page as plate_page
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, Progress, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    FailedField,
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
    LAY_OUT_FIRST,
    PLATE_NOT_CHOSEN,
    REGISTER_FIRST,
    SCREEN_LEADS,
    SCREEN_TITLES,
    SELECT_WELLS,
    busy_note,
    plain_default,
)
from smc.ui.plate_page import PlatePageView, posix_quoting_note

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
TEN_X = "Nikon 10X S Fluor"
#: Seconds a test waits for the other thread before it gives up: a bound, never a timing (FM-44).
BOUND_S = 10.0

#: The tools the page loads from their own ``value``, so that every section and field exists.
REAL = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "cells": "smc.plugins.cells.plugin:CellsPlugin",
    "density": "smc.plugins.density.plugin:DensityPlugin",
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "nis-export": "smc.plugins.nis_export.plugin:NisExportPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
    "report": "smc.plugins.report.plugin:ReportPlugin",
    "route": "smc.plugins.route.plugin:RoutePlugin",
    "scan": "smc.plugins.scan.plugin:ScanPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
    "timing": "smc.plugins.timing.plugin:TimingPlugin",
}


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class StubScanParams(PluginParams):
    autofocus: Literal["if-engaged", "required", "off"] = "if-engaged"


class StubScan:
    """A ``scan`` that reports each field of the plan it reads, and can wait after one of them.

    ``pause_at`` is the number of fields it reports before it waits on ``release``; ``0`` is no
    wait. ``before_report`` is called with the field's position before each report, so that a
    test moves its own clock where the worker would have read it.
    """

    name = provides = "scan"
    summary = "A test scan that reports every field of the plan."
    Params = StubScanParams
    Result = ScanResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("scan-plan",)
    reads_optional: tuple[str, ...] = ()
    ran: ClassVar[int] = 0
    pause_at: ClassVar[int] = 0
    paused: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()
    before_report: ClassVar[Callable[[int], None] | None] = None

    def run(
        self, microscope: Any, params: StubScanParams, ctx: RunContext
    ) -> ScanResult:
        StubScan.ran += 1
        plan = ctx.upstream("scan-plan", ScanPlanResult)
        total = len(plan.fields)
        for position, field in enumerate(plan.fields, start=1):
            hook = StubScan.before_report
            if hook is not None:
                hook(position)
            ctx.progress(position, total, field.name)
            if position == StubScan.pause_at:
                StubScan.paused.set()
                StubScan.release.wait(BOUND_S)
            # A STOP ends the run `stopped`, as the real scan's does.
            ctx.raise_if_stopped()
        return _scan_result(plan)


class Slow:
    """A tool that runs until released: an activity that is going."""

    name = provides = "slow"
    summary = "A test tool that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        Slow.started.set()
        Slow.release.wait(BOUND_S)
        return NoResult()


class Clock:
    """The clock the controller reads in these tests: it moves when the test says so."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def namespace(self) -> SimpleNamespace:
        # Only `monotonic` is the test's: the controller's waits sleep for real.
        import time

        return SimpleNamespace(monotonic=lambda: self.now, sleep=time.sleep)


def _entry_points(
    *stubs: type, real: tuple[str, ...] = ("scan-plan",)
) -> list[EntryPoint]:
    found = [EntryPoint(name=n, value=REAL[n], group=GROUP) for n in real]
    for cls in stubs:
        found.append(
            EntryPoint(
                name=cast("Any", cls).name,
                value=f"{__name__}:{cls.__name__}",
                group=GROUP,
            )
        )
    return found


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    StubScan.ran = 0
    StubScan.pause_at = 0
    StubScan.paused = threading.Event()
    StubScan.release = threading.Event()
    StubScan.before_report = None
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    # Presets are read from the working directory's `profiles/presets`.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    try:
        yield
    finally:
        StubScan.release.set()  # a worker a failed test left behind ends
        Slow.release.set()


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
        stubs: tuple[type, ...] = (StubScan, Slow),
        real: tuple[str, ...] = ("scan-plan",),
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
            entry_points=_entry_points(*stubs, real=real),
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
        StubScan.release.set()
        Slow.release.set()
        for page in pages:
            page.close()
        for stand in stands:
            stand.close()


@pytest.fixture
def page(make_page: Callable[..., PlatePage]) -> PlatePage:
    return make_page()


# --- the run document, as an earlier run left it ------------------------------------------


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


def _scan_plan_result(
    wells: tuple[str, ...] = ("B11", "B2", "A1"), *, estimate_s: float | None = 42.16
) -> ScanPlanResult:
    """A plan of 2 x 2 fields in each well, in visiting order: B11, then B2, then A1 (FM-204)."""
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
        objective=TEN_X,
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        matrix_source="parameter",
        image_shape_px=(512, 512),
        fov_um=fov_um,
        overlap=0.1,
        rows=2,
        columns=2,
        fields=fields,
        estimate_s=estimate_s,
    )


def _scan_result(
    plan: ScanPlanResult,
    *,
    failed: tuple[str, ...] = (),
    assumed: tuple[str, ...] = (),
) -> ScanResult:
    """The scan of ``plan``: a frame for each field but the ``failed`` ones, resting on ``assumed``."""
    return ScanResult(
        assumed=assumed,
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
            if field.name not in failed
        ),
        failed=tuple(
            FailedField(
                well=field.well,
                index=field.index,
                name=field.name,
                x_um=field.x_um,
                y_um=field.y_um,
                error="the stage did not arrive",
            )
            for field in plan.fields
            if field.name in failed
        ),
    )


def _seed(
    run_doc: Path,
    plugin: str,
    result: PluginResult,
    *,
    minute: int,
    dry_run: bool = False,
    reads: dict[str, ReadRecord | None] | None = None,
    params: dict[str, Any] | None = None,
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
            params=params or {},
            result=result.model_dump(mode="json"),
            reads=reads or {},
        ),
    )


def _seed_plate(run_doc: Path, *, dry_run: bool = False) -> None:
    _seed(
        run_doc,
        "plate-registration",
        _plate_result(),
        minute=0,
        dry_run=dry_run,
        params={"plate": "96-well"},
    )


def _seed_plan(
    run_doc: Path, plan: ScanPlanResult | None = None, *, dry_run: bool = False
) -> ScanPlanResult:
    plan = plan or _scan_plan_result()
    _seed(run_doc, "scan-plan", plan, minute=1, dry_run=dry_run)
    return plan


def _seed_scan(
    run_doc: Path,
    plan: ScanPlanResult,
    *,
    failed: tuple[str, ...] = (),
    assumed: tuple[str, ...] = (),
    minute: int = 2,
) -> None:
    _seed(
        run_doc,
        "scan",
        _scan_result(plan, failed=failed, assumed=assumed),
        minute=minute,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )


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


def _worker(page: PlatePage) -> threading.Thread:
    worker = page._worker
    assert worker is not None
    return worker


# --- the Wells screen -----------------------------------------------------------------------


def test_the_wells_screen_lists_the_wells_box_the_two_fields_and_the_action_before_what_is_known(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    coverage = view.tool_fields["scan-plan"]["coverage"]
    # Only `scan-plan` is real here; the stub `scan` has the same key field.
    autofocus = view.tool_fields["scan"]["autofocus"]
    shown = view.screens["wells"]
    assert [shown[0].object, shown[1].object] == [
        SCREEN_TITLES["wells"],
        SCREEN_LEADS["wells"],
    ]
    # The order of §17.4.3: the action is above what is known, so that it is on the page
    # without a scroll; the box and the two fields come before it.
    assert shown[2:] == [
        view.select_input,
        view.wells_from_text,
        coverage,
        view.coverage_from_text,
        autofocus,
        view.autofocus_from_text,
        view.layout_button,
        view.layout_wait,
        view.wells_known_block,
        view.wells_note,
        view.layout_card,
        view.continue_button,
    ]
    assert view.layout_button.label == "Lay out the fields"
    assert view.layout_button.button_type == "primary"
    # The Scan screen: the action, its wait and note, the bar and its lines, the result.
    scan = view.screens["scan"]
    assert scan[0] is view.scan_title_text
    assert scan[1] is view.scan_lead_text
    assert scan[2:] == [
        view.scan_button,
        view.scan_wait_text,
        view.scan_dry_note,
        view.scan_progress,
        view.scan_progress_text,
        view.scanned_block,
        view.scan_card,
        view.continue_button,
    ]
    assert view.scan_button.label == "Start the scan"
    assert view.scan_button.button_type == "primary"


def test_neither_screen_lists_the_next_step_or_a_tool_section(page: PlatePage) -> None:
    view = PlatePageView(page)
    for step in ("wells", "scan"):
        view.show_step(step)
        shown = _objects(list(view.step_panel.objects))
        assert view.next_text not in shown, step
        assert view.next_button not in shown, step
        assert not [b for b in view.tool_runs.values() if b in shown], step
        # No section's own *Run <tool>* and no *Advanced* fold: the tools' sections are the
        # expert view's.
        assert not [
            o
            for o in shown
            if isinstance(o, pn.widgets.Button) and o.label.startswith("Run ")
        ]
        assert not [
            o for o in shown if isinstance(o, pn.Card) and o.title == "Advanced"
        ]
        assert not [o for o in shown if isinstance(o, pn.Accordion)]
    # The next step is still where B4 has not replaced it.
    view.show_step("results")
    assert view.next_text in _objects(list(view.step_panel.objects))


def test_a_field_on_two_screens_is_the_tools_own_widget_with_its_from_line(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    for tool, key, line in (
        ("scan-plan", "coverage", view.coverage_from_text),
        ("scan", "autofocus", view.autofocus_from_text),
    ):
        widget = view.tool_fields[tool][key]
        # One instance, on the screen and in the expert view's section (D5), once on each.
        for screen in ("wells", "expert"):
            assert sum(o is widget for o in _objects(view.screens[screen])) == 1
        assert line in view.screens["wells"]
    coverage = view.tool_fields["scan-plan"]["coverage"]
    # An empty field is the tool's default, said without a string's quotes.
    assert view.coverage_from_text.object == "the tool's default: 1.0"
    assert view.autofocus_from_text.object == "the tool's default: if-engaged"
    # Typed: the line follows the field at once, without a redraw of the screen.
    coverage.value = "0.7"
    assert view.coverage_from_text.object == "typed on this page"
    coverage.value = ""
    assert view.coverage_from_text.object == "the tool's default: 1.0"
    # The *Wells* box says where its wells come from.
    assert view.wells_from_text.object == "no well chosen yet"
    view.select_input.value = "B2,B11"
    assert view.wells_from_text.object == "chosen on this page"
    # The widgets are the same ones after a swap, with the value the operator typed.
    coverage.value = "0.4"
    view.show_expert(True)
    view.show_step("wells")
    assert view.tool_fields["scan-plan"]["coverage"] is coverage
    assert coverage.value == "0.4"


def test_a_field_says_it_comes_from_the_profile_or_the_preset(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    profile = Profile.model_validate(
        {
            "microscope": {"name": "demo"},
            "suggest": {
                "scan-plan": {
                    "params": ["coverage=0.4"],
                    "why": "fast on the demo stage",
                }
            },
        }
    )
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    (folder / "weekly.toml").write_text(
        'stand = "demo"\nplate = "96-well"\nwells = "B2,B11"\n\n'
        '[params]\nscan-plan = ["coverage=0.7"]\n',
        encoding="utf-8",
    )
    view = PlatePageView(make_page(profile=profile))
    coverage = view.tool_fields["scan-plan"]["coverage"]
    # The field starts with the profile's suggestion, and says so.
    assert coverage.value == "0.4"
    assert view.coverage_from_text.object == "from the profile's suggestions"
    # The preset's value wins over the suggestion, and so does its selection of wells.
    view.preset_select.value = "weekly"
    _click(view.preset_load_button)
    assert coverage.value == "0.7"
    assert view.coverage_from_text.object == "from the preset weekly"
    assert view.wells_from_text.object == "from the preset weekly"
    # An edit leaves both: the line says the operator's.
    coverage.value = "0.9"
    assert view.coverage_from_text.object == "typed on this page"
    view.select_input.value = "B2"
    assert view.wells_from_text.object == "chosen on this page"


def test_the_autofocus_placeholder_has_no_quotes_and_neither_has_any_string_default(
    make_page: Callable[..., PlatePage],
) -> None:
    # Every tool, so that no field shows a quote the operator would type (finding 13).
    view = PlatePageView(make_page(real=tuple(REAL), stubs=()))
    assert view.tool_fields["scan"]["autofocus"].placeholder == "if-engaged"
    quoted = 0
    for (tool, key), default in view._field_defaults.items():
        placeholder = view.tool_fields[tool][key].placeholder
        assert placeholder == plain_default(default), (tool, key)
        if default[:1] in ("'", '"'):
            quoted += 1
            assert placeholder[:1] != default[0], (tool, key)
            assert placeholder[-1:] != default[0], (tool, key)
    assert quoted >= 3, "the tools have string defaults, and this checked them"
    # A number or a literal that is not a string keeps its text.
    assert view.tool_fields["scan-plan"]["coverage"].placeholder == "1.0"


def test_lay_out_the_fields_says_what_it_waits_for_and_runs_the_tools_handler(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page(connect=False)
    view = PlatePageView(page)
    view.show_step("wells")

    def waits() -> tuple[bool, str]:
        return view.layout_button.disabled, view.layout_wait.object

    # The stand is closed: the controller's reason for every tool.
    assert waits() == (True, NOT_CONNECTED)
    page.connect()
    assert page.join(BOUND_S)
    view.tick()
    # Connected, and nothing named the plate: the placeholder's wait, the screen's words.
    assert waits() == (True, PLATE_NOT_CHOSEN)
    page.choose_plate("96-well")
    view.refresh()
    assert waits() == (True, REGISTER_FIRST)
    _seed_plate(run_doc)
    view.refresh()
    assert waits() == (True, SELECT_WELLS)
    # Typed as the operator types: the selection is the page's, and the wait goes.
    view.select_input.value = "B2,B11"
    coverage = view.tool_fields["scan-plan"]["coverage"]
    coverage.value = "0.4"
    assert waits() == (False, "")
    assert view.layout_wait.visible is False
    # The command under the button is the call it makes, read back as a shell reads it.
    words = _words(view.layout_command.object)
    assert words[:3] == ["smc", "run", "scan-plan"]
    pairs = [w for i, w in enumerate(words) if i and words[i - 1] == "--param"]
    assert pairs == ["wells=B2,B11", "coverage=0.4"]
    assert view.layout_card.visible is True
    assert view.layout_card.collapsed is True
    # The click is the tool's own *Run*: its pairs, and the plan it wrote.
    _click(view.layout_button)
    assert page.join(BOUND_S)
    view.tick()
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("scan-plan done in ")
    doc = RunDocument.load(run_doc)
    record = doc.provider("scan-plan")
    assert record is not None
    assert record.params["coverage"] == 0.4
    plan = ScanPlanResult.model_validate(record.result)
    assert set(plan.wells) == {"B2", "B11"}
    # Nothing runs now, and a second click is allowed: the button is enabled again.
    assert view.layout_button.disabled is False


def test_a_disabled_lay_out_or_start_is_refused_by_the_controller(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # FM-169: a click on a disabled button reaches the handler, and the controller refuses.
    closed = make_page(connect=False)
    closed_view = PlatePageView(closed)
    assert closed_view.layout_button.disabled is True
    assert closed_view.scan_button.disabled is True
    # A selection first: the page fills the wells before the controller sees the stand.
    closed_view.select_input.value = "B2"
    _click(closed_view.layout_button)
    assert closed.status().outcome == f"scan-plan: {NOT_CONNECTED}"
    _click(closed_view.scan_button)
    assert closed.status().outcome == f"scan: {NOT_CONNECTED}"
    assert closed.status().running is None
    assert not run_doc.exists()

    # Connected, no selection, nothing registered: a Lay out that the page cannot fill.
    connected = make_page()
    view = PlatePageView(connected)
    assert view.layout_button.disabled is True
    _click(view.layout_button)
    assert view.message.object == "select at least one well first"
    assert connected.status().running is None
    # With wells selected but no registered plate, the tool refuses and nothing is written.
    view.select_input.value = "B2"
    assert view.layout_button.disabled is True
    _click(view.layout_button)
    outcome = connected.status().outcome
    assert outcome is not None
    assert outcome.startswith("scan-plan: ")
    assert "plate" in outcome
    # Start the scan with no plan: the same.
    assert view.scan_button.disabled is True
    _click(view.scan_button)
    outcome = connected.status().outcome
    assert outcome is not None
    assert outcome.startswith("scan: ")
    assert "scan-plan" in outcome
    assert connected.status().running is None
    assert StubScan.ran == 0
    assert not run_doc.exists()

    # A run going: both are refused by the controller as for every Run. Every other wait is
    # clear first (the plate chosen and registered, wells selected, a plan laid out), so that
    # the buttons are disabled by the run and by nothing else.
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    connected.choose_plate("96-well")
    view.refresh()
    assert (view.layout_button.disabled, view.layout_wait.object) == (False, "")
    assert (view.scan_button.disabled, view.scan_wait_text.object) == (False, "")
    before = run_doc.read_bytes()
    connected.run("slow", [])
    assert Slow.started.wait(BOUND_S)
    view.tick()
    assert view.layout_button.disabled is True
    assert view.scan_button.disabled is True
    assert view.layout_wait.object == busy_note("slow", live=False)
    assert view.scan_wait_text.object == busy_note("slow", live=False)
    _click(view.scan_button)
    assert view.message.object == "slow is running"
    _click(view.layout_button)
    assert view.message.object == "slow is running"
    assert StubScan.ran == 0
    Slow.release.set()
    assert connected.join(BOUND_S)
    assert plan.wells == ("B11", "B2", "A1")
    assert (
        run_doc.read_bytes() != before
    )  # the slow tool's own record, and nothing else
    doc = RunDocument.load(run_doc)
    assert doc.provider("scan") is None


def test_laying_out_two_wells_draws_the_fields_the_time_and_the_assumed_lines(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    view.show_step("wells")
    _seed_plate(run_doc)
    view.select_input.value = "B2,B11"
    view.tool_fields["scan-plan"]["coverage"].value = "0.4"
    view.refresh()
    assert view.wells_known_block.object == "\n".join(
        (
            f"Objective: {TEN_X}, in place",
            "Fields: not laid out yet",
        )
    )
    _click(view.layout_button)
    assert page.join(BOUND_S)
    view.tick()
    doc = RunDocument.load(run_doc)
    record = doc.provider("scan-plan")
    assert record is not None
    plan = ScanPlanResult.model_validate(record.result)
    # The map draws the plan's fields, and none is imaged yet.
    assert len(view.fields_source.data["x"]) == len(plan.fields) > 0
    assert view.imaged_source.data["x"] == []
    lines = view.wells_known_block.object.split("\n")
    assert lines[:3] == [
        f"Objective: {TEN_X}, in place",
        f"Fields: {len(plan.fields)} in 2 wells",
        # No timing was measured on this page: an unknown, never `0 s` (FM-170).
        "Time: not known (no timing measured; the expert view's timing tool measures it, "
        "then lay out the fields again)",
    ]
    # The plan card's own assumptions, one line each, after the time.
    card = page.view().cards["scan-plan"]
    assert card.assumed
    assert lines[3:] == [f"assumed: {text}" for text in card.assumed]
    assert view.wells_note.visible is False
    # The step bar says the same count.
    assert view.step_summaries["wells"].object == f"2 wells, {len(plan.fields)} fields"


def test_a_plan_with_an_estimate_says_the_time_and_the_singular(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view.refresh()
    assert view.wells_known_block.object.split("\n")[:3] == [
        f"Objective: {TEN_X}, in place",
        "Fields: 12 in 3 wells",
        "Time: about 42 s",
    ]


def test_a_selection_changed_after_the_plan_is_said_on_both_screens(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    _seed_plate(run_doc)
    _seed_plan(run_doc)  # for B11, B2 and A1
    view.select_input.value = "A1,B2,B11"
    view.refresh()
    assert view.wells_note.visible is False
    assert view.scan_title_text.object == "Scan A1, B2 and B11"
    # The operator selects another well: the plan has not changed, and says whose it is.
    view.select_input.value = "B2"
    assert view.wells_note.visible is True
    assert view.wells_note.object == (
        "The fields laid out are for A1, B2 and B11; the wells selected now are B2. "
        "Lay out the fields again to scan these."
    )
    # The Scan screen names the plan's wells, not the selection's: it is what `scan` does.
    assert view.scan_title_text.object == "Scan A1, B2 and B11"
    view.select_input.value = ""
    assert view.wells_note.object == (
        "The fields laid out are for A1, B2 and B11; no well is selected now."
    )
    view.select_input.value = "B11,A1,B2"
    assert view.wells_note.visible is False


def test_a_stale_plan_is_said_on_the_wells_screen_with_the_checklists_reason(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-226: the tool's section carried the card and its stale mark; the screen says why.
    view = PlatePageView(page)
    _seed_plate(run_doc)
    _seed(
        run_doc,
        "scan-plan",
        _scan_plan_result(),
        minute=1,
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
    )
    # The plate was registered again after the plan read it.
    _seed(
        run_doc,
        "plate-registration",
        _plate_result(),
        minute=3,
        params={"plate": "96-well"},
    )
    view.refresh()
    assert view.wells_known_block.object.split("\n")[1] == (
        "Fields: out of date (scan-plan read its inputs from an earlier run; run it again)"
    )
    # Nothing is drawn as the plan's, nor claimed: no estimate, no assumption.
    assert "Time:" not in view.wells_known_block.object
    assert view.fields_source.data["x"] == []
    # And the Scan screen waits for the fields to be laid out again.
    assert view.scan_button.disabled is True
    assert view.scan_wait_text.object == LAY_OUT_FIRST


# --- the Scan screen ------------------------------------------------------------------------


def test_the_scan_screens_title_names_the_plans_wells_and_its_lead_the_fields(
    page: PlatePage, run_doc: Path
) -> None:
    view = PlatePageView(page)
    # No plan and no selection: the static title and lead.
    assert view.scan_title_text.object == SCREEN_TITLES["scan"]
    assert view.scan_lead_text.object == SCREEN_LEADS["scan"]
    # A selection and no plan: the selection's wells, in plate order.
    view.select_input.value = "B11,B2"
    assert view.scan_title_text.object == "Scan B2 and B11"
    assert view.scan_lead_text.object == SCREEN_LEADS["scan"]
    # A plan: its wells in plate order (the demo visits B11 first), and its fields.
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view.refresh()
    assert view.scan_title_text.object == "Scan A1, B2 and B11"
    assert view.scan_lead_text.object == (
        f"12 fields at {TEN_X}. STOP, at the top right, stops the scan."
    )
    # Twelve wells are counted.
    twelve = tuple(f"{row}{col}" for row in "AB" for col in range(1, 7))
    _seed(run_doc, "scan-plan", _scan_plan_result(twelve), minute=4)
    view.refresh()
    assert view.scan_title_text.object == "Scan 12 wells"


def test_the_scan_lead_says_a_move_in_flight_finishes_first_on_a_stand_that_refuses_a_stop(
    make_page: Callable[..., PlatePage],
) -> None:
    profile = Profile.model_validate(
        {
            "microscope": {"name": "demo"},
            "safety": {"stop_refused_by": ["xy_stage"]},
        }
    )
    view = PlatePageView(make_page(profile=profile))
    assert view.scan_lead_text.object == (
        f"{SCREEN_LEADS['scan']} A move already sent finishes first on this stand."
    )


def test_start_the_scan_waits_for_the_plan_and_says_so(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page(connect=False)
    view = PlatePageView(page)
    view.show_step("scan")
    assert (view.scan_button.disabled, view.scan_wait_text.object) == (
        True,
        NOT_CONNECTED,
    )
    page.connect()
    assert page.join(BOUND_S)
    view.tick()
    # Connected and no plan: the first thing to do, in the screen's own words.
    assert (view.scan_button.disabled, view.scan_wait_text.object) == (
        True,
        LAY_OUT_FIRST,
    )
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view.refresh()
    assert (view.scan_button.disabled, view.scan_wait_text.visible) == (False, False)
    # A run going: what runs and how to end it, in place of a wait.
    page.run("slow", [])
    assert Slow.started.wait(BOUND_S)
    view.tick()
    assert view.scan_button.disabled is True
    assert view.scan_wait_text.object == busy_note("slow", live=False)
    Slow.release.set()
    assert page.join(BOUND_S)
    view.tick()
    assert view.scan_button.disabled is False


def test_a_dry_run_page_says_what_a_scan_leaves_out_and_still_starts_it(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # A dry run walks the fields: it is the mode the page was asked for, and /ui-check serves
    # it so. The note is a note, not a wait (m5 §17.4.4).
    dry = make_page(dry_run=True)
    view = PlatePageView(dry)
    _seed_plate(run_doc, dry_run=True)
    _seed_plan(run_doc, dry_run=True)
    view.refresh()
    assert view.scan_dry_note.object == DRY_RUN_NO_FRAME
    assert view.scan_dry_note.visible is True
    assert view.scan_button.disabled is False
    assert view.scan_wait_text.visible is False
    _click(view.scan_button)
    assert dry.join(BOUND_S)
    assert StubScan.ran == 1
    view.tick()
    assert dry.status().outcome is not None
    assert dry.status().outcome.startswith("scan done in ")


def test_a_page_that_is_not_a_dry_run_has_no_dry_run_note(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.scan_dry_note.visible is False
    assert view.scan_dry_note.object == ""


def _run_a_scan_to(
    page: PlatePage,
    view: PlatePageView,
    clock: Clock,
    *,
    pause_at: int,
) -> None:
    """Start the stub scan on the page, with the clock the test owns, and wait until it reported ``pause_at`` fields."""
    StubScan.pause_at = pause_at
    start = clock.now
    # The first field is reported 4 s after the click, then every 2 s: the start-up of a
    # scan is not its pace (m5 §17.10.1).
    StubScan.before_report = lambda position: setattr(
        clock, "now", start + 2.0 * (position + 1)
    )
    _click(view.scan_button)
    assert StubScan.paused.wait(BOUND_S)


def test_the_progress_bar_and_its_lines_follow_the_scan_and_stay_after_it(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view = PlatePageView(page)
    view.show_step("scan")
    view.refresh()
    # Before any scan: no bar, no lines.
    assert view.scan_progress.visible is False
    assert view.scan_progress_text.visible is False
    clock = Clock(100.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        _run_a_scan_to(page, view, clock, pause_at=5)
        view.tick()
        # Five of twelve; reports at 4, 6, 8, 10 and 12 s after the click: 2 s a field from the
        # first, so the seven that are left take 14 s, where a rate from the click (12 s for
        # five) would say 16.8 s.
        assert view.scan_progress.visible is True
        assert (view.scan_progress.value, view.scan_progress.max) == (5, 12)
        assert view.scan_progress.active is True
        assert view.scan_progress_text.object == "\n".join(
            (
                "Field 5 of 12",
                "Elapsed 12 s",
                "Left about 14 s (estimated from the fields so far)",
            )
        )
        assert view.scan_button.disabled is True
        assert view.scan_wait_text.object == busy_note("scan", live=False)
        # It ends: 40 s after the click.
        clock.now = 140.0
        StubScan.before_report = lambda position: setattr(clock, "now", 140.0)
        StubScan.release.set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        view.tick()
        # After the end the bar and its lines stay, with no *Left*, the stripes stopped and
        # the length of the run fixed.
        assert (view.scan_progress.value, view.scan_progress.max) == (12, 12)
        assert view.scan_progress.active is False
        assert view.scan_progress.visible is True
        assert view.scan_progress_text.object == "Field 12 of 12\nElapsed 40 s"
        clock.now = 500.0
        view.tick()
        assert view.scan_progress_text.object == "Field 12 of 12\nElapsed 40 s"
    assert page.join(BOUND_S)
    assert view.scan_button.disabled is False


def test_the_progress_block_goes_with_the_next_activity_and_is_not_lives(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view = PlatePageView(page)
    view.show_step("scan")
    view.refresh()
    _click(view.scan_button)
    assert page.join(BOUND_S)
    view.tick()
    assert view.scan_progress.visible is True
    # Another activity starts: its progress is its own, and the block goes (a snap resets it).
    view.on_snap()
    assert page.join(BOUND_S)
    view.tick()
    assert view.scan_progress.visible is False
    assert view.scan_progress_text.visible is False
    # Live reports its own progress (plugin `live`): never the scan's bar.
    view.on_live()
    try:
        status = page.status()
        for _ in range(100):
            if status.progress is not None:
                break
            threading.Event().wait(0.05)
            status = page.status()
        assert status.progress is not None
        assert status.progress.plugin == "live"
        view.tick()
        assert view.scan_progress.visible is False
    finally:
        view.on_live()
        assert page.join(BOUND_S)


def test_the_map_marks_the_first_done_fields_while_the_scan_runs_and_the_record_after(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    centres = [(f.x_um, f.y_um) for f in plan.fields]
    view = PlatePageView(page)
    view.show_step("scan")
    view.refresh()
    assert view.imaged_source.data["x"] == []
    clock = Clock(100.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        _run_a_scan_to(page, view, clock, pause_at=5)
        view.tick()
        # The plan's first five fields, in the plan's order, each fov_um wide and high: the
        # orange fields stay under them.
        data = view.imaged_source.data
        assert list(zip(data["x"], data["y"], strict=True)) == centres[:5]
        assert data["w"] == [512.0] * 5
        assert data["h"] == [512.0] * 5
        assert len(view.fields_source.data["x"]) == 12
        # A tick that sees the same number draws nothing again (FM-145): a marker put in the
        # source is still there after it, where a redraw would have replaced it.
        marker = {"x": [1.0], "y": [2.0], "w": [3.0], "h": [4.0]}
        view.imaged_source.data = marker
        view.tick()
        assert dict(view.imaged_source.data) == marker
        StubScan.before_report = None
        StubScan.release.set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        # The end of the run redraws the record: every field has a frame, and each is the
        # field of view wide and high (a rectangle of no size is a layer nobody sees).
        view.tick()
        data = view.imaged_source.data
        assert list(zip(data["x"], data["y"], strict=True)) == centres
        assert data["w"] == [512.0] * 12
        assert data["h"] == [512.0] * 12
    assert page.join(BOUND_S)


def test_a_scan_that_stops_clears_the_map_and_a_new_scan_starts_from_none(
    page: PlatePage, run_doc: Path
) -> None:
    # The map claims only what a usable `scan` holds: a stopped scan's frames are on disk and
    # its checkpoint continues it, but its record is not a result (m5 §17.10.2).
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    _seed_scan(run_doc, plan)
    view = PlatePageView(page)
    view.refresh()
    assert len(view.imaged_source.data["x"]) == 12
    StubScan.pause_at = 3
    _click(view.scan_button)
    assert StubScan.paused.wait(BOUND_S)
    view.tick()
    # The previous scan's twelve are gone at once, and three of the new one are drawn.
    assert len(view.imaged_source.data["x"]) == 3
    page.stop()
    StubScan.release.set()
    assert page.join(BOUND_S)
    view.tick()
    doc = RunDocument.load(run_doc)
    stopped = doc.provider("scan")
    assert stopped is not None
    assert stopped.status is StageStatus.stopped
    # The end of the run redraws the record, and a stopped scan holds no result: the layer
    # is empty, and the result line says nothing, rather than three fields or an older twelve.
    assert view.imaged_source.data["x"] == []
    assert view.scanned_block.visible is False
    # A second scan reaches the same three fields: its first tick draws them, though the
    # count is the one the view drew for the first (a count is not what says "already drawn").
    page.resume()
    StubScan.paused = threading.Event()
    StubScan.release = threading.Event()
    # Started through the controller, so that no tick of this view runs between the click
    # and the third report: the first tick it makes is the one that sees `done == 3`.
    page.run("scan", [])
    assert StubScan.paused.wait(BOUND_S)
    view.tick()
    assert len(view.imaged_source.data["x"]) == 3
    StubScan.release.set()
    assert page.join(BOUND_S)


def test_a_refresh_during_a_scan_does_not_draw_an_older_scans_fields(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    # An older scan: all twelve fields have a frame.
    _seed_scan(run_doc, plan)
    view = PlatePageView(page)
    view.refresh()
    assert len(view.imaged_source.data["x"]) == 12
    StubScan.pause_at = 4
    _click(view.scan_button)
    assert StubScan.paused.wait(BOUND_S)
    view.tick()
    assert len(view.imaged_source.data["x"]) == 4
    # Another tab's redraw, a click on the map or the end of another run: this view's
    # refresh reads the document, whose latest scan is not this one's progress.
    view.refresh()
    assert len(view.imaged_source.data["x"]) == 4
    view.tick()
    assert len(view.imaged_source.data["x"]) == 4
    StubScan.release.set()
    assert page.join(BOUND_S)
    view.tick()
    assert len(view.imaged_source.data["x"]) == 12


def test_a_second_tab_draws_the_same_progress_from_the_shared_status(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-151: the progress, the clock and the imaged fields are the controller's; each tab's
    # view draws them from its own tick, and a tab opened mid-scan draws them at once.
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    first = PlatePageView(page)
    first.show_step("scan")
    StubScan.pause_at = 6
    _click(first.scan_button)
    assert StubScan.paused.wait(BOUND_S)
    second = PlatePageView(page)
    second.show_step("scan")
    first.tick()
    second.tick()
    for view in (first, second):
        assert view.scan_progress.visible is True
        assert (view.scan_progress.value, view.scan_progress.max) == (6, 12)
        assert view.scan_progress_text.object.split("\n")[0] == "Field 6 of 12"
        assert view.scan_progress.active is True
        assert len(view.imaged_source.data["x"]) == 6
        assert view.scan_button.disabled is True
    assert second.imaged_source.data["x"] == [f.x_um for f in plan.fields[:6]]
    StubScan.release.set()
    assert page.join(BOUND_S)
    first.tick()
    second.tick()
    for view in (first, second):
        assert view.scan_progress.active is False
        assert view.scan_progress_text.object.startswith("Field 12 of 12")
        assert len(view.imaged_source.data["x"]) == 12
        assert (
            view.scanned_block.object.split("\n")[0] == "Scanned: 12 frames, 0 failed"
        )


def test_the_result_line_follows_the_run_document_not_the_progress(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    failed = plan.fields[5].name
    assumption = "the autofocus was left as it was: it is off on this stand"
    _seed_scan(run_doc, plan, failed=(failed,), assumed=(assumption,))
    # A page started after the scan, as after a restart: no progress and no clock, and the
    # result is there all the same, with the card's `assumed` lines and its folder (the scan's
    # own section, which carried them, is the expert view's: FM-226).
    page = make_page()
    view = PlatePageView(page)
    view.show_step("scan")
    assert page.status().progress is None
    assert view.scan_progress.visible is False
    lines = view.scanned_block.object.split("\n")
    assert lines == [
        "Scanned: 11 frames, 1 failed",
        f"assumed: {assumption}",
        f"folder: {run_doc.parent / 'scan'}",
    ]
    assert view.scanned_block.visible is True
    # The frame that failed is not marked imaged; the others are, in the plan's order, each
    # the field of view wide and high.
    data = view.imaged_source.data
    assert len(data["x"]) == 11
    assert data["w"] == [512.0] * 11
    assert data["h"] == [512.0] * 11
    assert (plan.fields[5].x_um, plan.fields[5].y_um) not in list(
        zip(data["x"], data["y"], strict=True)
    )


def test_a_dry_runs_result_line_says_no_frame_was_taken(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    dry = make_page(dry_run=True)
    view = PlatePageView(dry)
    _seed_plate(run_doc, dry_run=True)
    plan = _seed_plan(run_doc, dry_run=True)
    walked = "dry run: 12 fields walked, no frame taken"
    _seed(
        run_doc,
        "scan",
        _scan_result(plan, assumed=(walked,)).model_copy(update={"frames": ()}),
        minute=2,
        dry_run=True,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )
    view.refresh()
    # The result line, then what the run assumed (it says what it left out), then the folder.
    assert view.scanned_block.object.split("\n")[:2] == [
        "Scanned in a dry run: no frame was taken",
        f"assumed: {walked}",
    ]
    assert view.imaged_source.data["x"] == []


def test_the_scan_screen_says_nothing_of_a_result_that_is_not_the_views(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    _seed_scan(run_doc, plan)
    view = PlatePageView(page)
    view.refresh()
    assert view.scanned_block.visible is True
    # The fields were laid out again (a4): the scan read the first attempt, so it is of an
    # earlier plan, and says nothing.
    _seed(run_doc, "scan-plan", plan, minute=4)
    view.refresh()
    assert view.scanned_block.visible is False
    assert view.scanned_block.object == ""


def test_the_tick_draws_the_new_screens_without_reading_the_view(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-145: a tick reads the stand once, never the run document.
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view = PlatePageView(page)
    reads: list[int] = []
    original = page.view

    def counting() -> Any:
        reads.append(1)
        return original()

    page.view = counting  # type: ignore[method-assign]
    try:
        for _ in range(3):
            view.tick()
    finally:
        del page.view
    assert reads == []


def test_every_command_card_of_the_new_buttons_carries_the_windows_note(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-175: a page served on Windows says its commands are quoted for a POSIX shell, under
    # each *Command* of a button that replaced a tool's section (FM-226).
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view = PlatePageView(page)
    view.select_input.value = "B2"
    view._windows = True
    view.refresh()
    note = posix_quoting_note(True)
    assert note
    for card, command, quoted in (
        (view.layout_card, view.layout_command, view.layout_command_note),
        (view.scan_card, view.scan_command, view.scan_command_note),
    ):
        assert card.title == "Command"
        assert card.collapsed is True
        assert card.visible is True
        assert command.object.startswith("smc run ")
        assert quoted.object == note
        assert quoted.visible is True
        assert quoted in card.objects
    assert _words(view.scan_command.object)[:3] == ["smc", "run", "scan"]
    # Not a Windows page: no note, and no height for it.
    view._windows = False
    view.refresh()
    assert view.layout_command_note.visible is False
    assert view.scan_command_note.visible is False


def test_a_typed_field_is_in_both_commands_and_a_selection_the_page_refuses_draws_none(
    page: PlatePage, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    _seed_plan(run_doc)
    view = PlatePageView(page)
    view.refresh()
    # No wells selected: no command for *Lay out the fields*, and the card is hidden.
    assert view.layout_command.object == ""
    assert view.layout_card.visible is False
    # *Start the scan* has its command whatever the selection is.
    assert view.scan_card.visible is True
    view.tool_fields["scan"]["autofocus"].value = "off"
    words = _words(view.scan_command.object)
    assert words[-2:] == ["--param", "autofocus=off"]
    view.select_input.value = "B2,B11"
    view.tool_fields["scan-plan"]["coverage"].value = "0.4"
    words = _words(view.layout_command.object)
    assert words[-4:] == ["--param", "wells=B2,B11", "--param", "coverage=0.4"]
    assert view.layout_card.visible is True


def test_every_handler_the_new_buttons_register_is_guarded(page: PlatePage) -> None:
    # FM-171: a handler runs under the document lock and never raises into Panel. The guard
    # test of test_plate_page.py walks the layout and `screens`; the new buttons are walked
    # here by name too, so that a button added without `_handler` is found by this module.
    view = PlatePageView(page)
    for button in (view.layout_button, view.scan_button):
        registered = [
            watcher.fn
            for by_name in button.param.watchers.values()
            for watchers in by_name.values()
            for watcher in watchers
        ]
        # Panel registers watchers of its own on the same widget: ours are the functions of
        # the page's module, which `functools.wraps` leaves a `__wrapped__` on.
        ours = [
            fn
            for fn in registered
            if getattr(fn, "__module__", "") == plate_page.__name__
        ]
        assert ours, button.label
        assert [fn for fn in ours if not hasattr(fn, "__wrapped__")] == [], button.label


HOSTILE = '<img src=x onerror="alert(1)"> **bold** [x](javascript:y) &amp;'


def test_every_text_pane_of_the_wells_and_scan_screens_is_plain_text_whatever_the_stand_says(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-139. Free text reaches these screens from the stand (the turret's label), from a plan
    # (its objective and its assumptions) and from a scan (its assumptions): it is shown as it
    # is, never as markup, or a label of `<img onerror=...>` runs on the stand PC.
    _seed_plate(run_doc)
    plan = _scan_plan_result().model_copy(
        update={"objective": HOSTILE, "assumed": (HOSTILE,)}
    )
    _seed(run_doc, "scan-plan", plan, minute=1)
    _seed(
        run_doc,
        "scan",
        _scan_result(plan, assumed=(HOSTILE,)),
        minute=2,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )
    real_status = page.status
    monkeypatch.setattr(
        page, "status", lambda: dataclasses.replace(real_status(), objective=HOSTILE)
    )
    view = PlatePageView(page)
    view.select_input.value = "B2,B11"
    view.refresh()
    view.tick()
    # Said as it was given, in each of the places that hold a stand's or a result's words.
    assert (
        view.wells_known_block.object.split("\n")[0]
        == f"Objective: {HOSTILE}, in place"
    )
    assert f"assumed: {HOSTILE}" in view.wells_known_block.object.split("\n")
    assert view.scan_lead_text.object.startswith(f"12 fields at {HOSTILE}. ")
    assert f"assumed: {HOSTILE}" in view.scanned_block.object.split("\n")
    # And every text pane of both screens is a `Str`, counted together so that a pane that
    # becomes Markdown is a different count and not a pass.
    shown: list[Any] = []
    for step in ("wells", "scan"):
        view.show_step(step)
        objects = _objects(list(view.step_panel.objects))
        # A card draws its title as an HTML pane of its own, from a constant of the page's.
        titles = {f"<h3>{o.title}</h3>" for o in objects if isinstance(o, pn.Card)}
        shown += [
            o
            for o in objects
            if isinstance(o, pn.pane.PaneBase)
            and not (type(o) is pn.pane.HTML and o.object in titles)
        ]
    assert len(shown) > 14
    assert [type(o).__name__ for o in shown if type(o) is not pn.pane.Str] == []


def test_the_imaged_layer_is_a_green_rect_over_the_fields_under_the_route_and_not_hoverable(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)

    def layer() -> Any:
        renderers = view.figure.renderers
        imaged = [r for r in renderers if r.data_source is view.imaged_source]
        fields = [r for r in renderers if r.data_source is view.fields_source]
        route = [
            r
            for r in renderers
            if r.data_source is view.route_source and isinstance(r.glyph, Line)
        ]
        assert len(imaged) == len(fields) == len(route) == 1
        assert isinstance(imaged[0].glyph, Rect)
        assert imaged[0] not in view.hover.renderers
        # The green is over the plan's orange, and the route over both.
        assert (
            renderers.index(fields[0])
            < renderers.index(imaged[0])
            < renderers.index(route[0])
        )
        return imaged[0].glyph, fields[0].glyph

    for _plate in ("96-well", "384-well"):
        # A plate with square wells swaps the wells' renderer: the layer stays where it was.
        view.plate_select.value = _plate
        green, orange = layer()
        assert green.fill_color == green.line_color == "#2ca02c"
        assert green.fill_color != orange.fill_color
        # Visible over the orange: not clear, and more opaque than the plan's fields.
        assert 0.0 < orange.fill_alpha < green.fill_alpha < 1.0
        assert view.hover.renderers == [view.wells_renderer]


def test_a_dry_run_scan_marks_no_field_while_it_runs(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # A dry run walks the fields and takes no frame, and says so (the note, the result line):
    # a green field is a field with a frame, so the layer does not grow with the progress while
    # the bar does. The stub takes frames, a real dry-run scan does not; only the layer while
    # it runs is asserted here (m5 §17.10.2).
    dry = make_page(dry_run=True)
    view = PlatePageView(dry)
    _seed_plate(run_doc, dry_run=True)
    _seed_plan(run_doc, dry_run=True)
    view.refresh()
    StubScan.pause_at = 5
    _click(view.scan_button)
    assert StubScan.paused.wait(BOUND_S)
    view.tick()
    assert (view.scan_progress.value, view.scan_progress.max) == (5, 12)
    assert view.imaged_source.data["x"] == []
    StubScan.release.set()
    assert dry.join(BOUND_S)


def test_the_bar_never_holds_a_value_above_its_maximum_nor_a_maximum_below_one(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Panel refuses a `value` above the `max` it holds at that moment, and not a `max` below
    # the `value` it holds: a shorter scan after a longer one, a total of 0, a report with no
    # total and a done above the total are all drawn as a bar that holds.
    view = PlatePageView(page)
    real_status = page.status
    for done, total, expected in (
        (50, 72, (50, 72)),
        (
            6,
            3,
            (3, 3),
        ),  # `max` shrinks below the value the bar held, then `value` follows
        (0, 0, (0, 1)),
        (4, None, (4, 4)),
        (-1, 12, (0, 12)),
    ):
        progress = Progress("scan", done, total, "B2_0000")
        monkeypatch.setattr(
            page,
            "status",
            lambda progress=progress: dataclasses.replace(
                real_status(), progress=progress
            ),
        )
        view.tick()
        assert (view.scan_progress.value, view.scan_progress.max) == expected, (
            done,
            total,
        )
        assert view.scan_progress.visible is True


def test_a_second_scan_that_starts_before_the_view_redrew_is_drawn_from_its_own_count(
    page: PlatePage, run_doc: Path
) -> None:
    # The view saw five fields of the first run; that run ended and a second began between two
    # ticks, so no end-of-run redraw separated them: the second's two fields replace the five
    # (m5 §17.10.2: a `done` lower than what the view drew is a new run).
    _seed_plate(run_doc)
    plan = _seed_plan(run_doc)
    view = PlatePageView(page)
    StubScan.pause_at = 5
    _click(view.scan_button)
    assert StubScan.paused.wait(BOUND_S)
    view.tick()
    assert len(view.imaged_source.data["x"]) == 5
    StubScan.release.set()
    assert page.join(BOUND_S)
    StubScan.paused = threading.Event()
    StubScan.release = threading.Event()
    StubScan.pause_at = 2
    # Through the controller and with no tick of this view in between.
    page.run("scan", [])
    assert StubScan.paused.wait(BOUND_S)
    view.tick()
    data = view.imaged_source.data
    assert list(zip(data["x"], data["y"], strict=True)) == [
        (f.x_um, f.y_um) for f in plan.fields[:2]
    ]
    StubScan.release.set()
    assert page.join(BOUND_S)


# --- the whole path, on the demo's sample (the issue's second criterion) ---------------------

DEMO_SAMPLE = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"


@pytest.mark.demo
def test_two_wells_laid_out_and_scanned_from_these_screens_alone(
    mm_available: bool, tmp_path: Path
) -> None:
    # `demo-sample` served without --dry-run: its stage is simulated, and the scan takes real
    # frames of the synthetic plate. Every step is a click on a button of the screens, in the
    # order a newcomer takes them; none runs a tool's own section.
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), run_doc)
    try:
        view = PlatePageView(page)
        # 1. Connect: the strip's own button.
        _click(view.connect_button)
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome
        view.tick()
        # 2. Plate: *Register the plate*, from the sample's references.
        view.show_step("plate")
        assert view.register_button.disabled is False, view.register_wait.object
        _click(view.register_button)
        assert page.join(60.0), "plate-registration did not end"
        view.tick()
        assert view.step_buttons["plate"].label == "2. Plate ✓"
        # 3. Wells: two wells typed, the profile's coverage in the field, *Lay out the fields*.
        view.show_step("wells")
        view.select_input.value = "B2,B11"
        assert view.tool_fields["scan-plan"]["coverage"].value == "0.4"
        assert view.coverage_from_text.object == "from the profile's suggestions"
        assert view.layout_button.disabled is False, view.layout_wait.object
        _click(view.layout_button)
        assert page.join(60.0), "scan-plan did not end"
        view.tick()
        lines = view.wells_known_block.object.split("\n")
        assert lines[1] == "Fields: 72 in 2 wells"
        assert len(view.fields_source.data["x"]) == 72
        assert view.imaged_source.data["x"] == []
        # 4. Scan: the title and lead name the plan, *Start the scan* runs it.
        view.show_step("scan")
        assert view.scan_title_text.object == "Scan B2 and B11"
        assert view.scan_lead_text.object.startswith("72 fields at ")
        assert view.scan_dry_note.visible is False
        assert view.scan_button.disabled is False, view.scan_wait_text.object
        _click(view.scan_button)
        assert page.join(120.0), "scan did not end"
        view.tick()
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith("scan done in "), outcome
        assert (
            view.scanned_block.object.split("\n")[0] == "Scanned: 72 frames, 0 failed"
        )
        assert len(view.imaged_source.data["x"]) == 72
        assert view.scan_progress.value == view.scan_progress.max == 72
        assert view.scan_progress_text.object.startswith("Field 72 of 72\nElapsed ")
        # Nothing of a tool's own section was needed, and the run document says so: the
        # plan and the scan are there, laid out and made by these screens' buttons.
        doc = RunDocument.load(run_doc)
        assert {
            slot for slot in ("plate", "scan-plan", "scan") if doc.provider(slot)
        } == {
            "plate",
            "scan-plan",
            "scan",
        }
    finally:
        page.close()
        page.release()
