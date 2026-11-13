"""What the controller keeps for the Scan screen: a clock for the progress, and the scan in the view (m5 §17.10, #445).

No Panel here. The clock is stamped where the worker reports, never where a tick drains the
queue, so these tests own the clock the controller reads (FM-44: the name patched in the
module, inside ``monkeypatch.context()``, so that a thread of the code under test reads the
real one afterwards) and a gate between two reports, and ask for the status long after. The
plan layer's tests seed a plan and a scan the way an earlier run left them, with wells in
visiting order and not in alphabetical order (FM-204), and read ``view()``.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import Camera
from smc.hardware.errors import MotionStoppedError
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
from smc.ui.controller import PlatePage, RunClock, ScanSummary
from smc.ui.guide import scan_progress_lines

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
TEN_X = "Nikon 10X S Fluor"
#: Seconds a test waits for the other thread before it gives up: a bound, never a timing (FM-44).
BOUND_S = 10.0


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class Reporting:
    """A tool that reports two fields, each after a gate the test opens, and ends at a third gate."""

    name = provides = "reporting"
    summary = "A test tool that reports its progress."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    gates: ClassVar[list[threading.Event]] = []
    reported: ClassVar[list[threading.Event]] = []

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        Reporting.gates[0].wait(BOUND_S)
        ctx.progress(1, 3, "field-a")
        Reporting.reported[0].set()
        Reporting.gates[1].wait(BOUND_S)
        ctx.progress(2, 3, "field-b")
        Reporting.reported[1].set()
        Reporting.gates[2].wait(BOUND_S)
        return NoResult()


class Quick:
    """A tool that reports nothing and ends at once: the activity that follows a scan."""

    name = provides = "quick"
    summary = "A test tool that ends at once."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        return NoResult()


class LiveCamera:
    """The six methods of ``Camera``, giving one 6 x 10 frame at once: enough for a Live."""

    def snap(self) -> Any:
        import numpy as np

        return np.arange(60, dtype=np.uint16).reshape(6, 10)

    def exposure_ms(self) -> float:
        return 5.0

    def set_exposure_ms(self, value_ms: float) -> float:
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (6, 10)

    def bit_depth(self) -> int:
        return 16

    def pixel_size_um(self) -> float:
        return 0.65


class Clock:
    """The clock the controller reads in these tests: it moves when the test says so."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def namespace(self) -> SimpleNamespace:
        # Only `monotonic` is the test's: the controller's `join` and a read's wait sleep.
        return SimpleNamespace(monotonic=lambda: self.now, sleep=time.sleep)


@pytest.fixture(autouse=True)
def _fresh_gates() -> Iterator[None]:
    Reporting.gates = [threading.Event() for _ in range(3)]
    Reporting.reported = [threading.Event() for _ in range(2)]
    try:
        yield
    finally:
        for gate in Reporting.gates:
            gate.set()  # a worker a failed test left behind ends


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)
        for cls in (Reporting, Quick)
    ]


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = PlatePage(fake_microscope, run_doc, entry_points=_entry_points())
    try:
        yield controller
    finally:
        for gate in Reporting.gates:
            gate.set()
        controller.close()


def _worker(page: PlatePage) -> threading.Thread:
    worker = page._worker
    assert worker is not None
    return worker


# --- the clock (m5 §17.10.2) ----------------------------------------------------------


def test_a_run_stamps_its_progress_reports_at_the_worker_not_at_the_drain(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(100.0)
    # The name, not `time.monotonic`: the worker reads the real clock after the block.
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        page.run("reporting", [])
        # 4 s after the click the first field is reported, 10 s after it the second:
        # nobody asks for the status in between, as a tick one second coarse would not.
        clock.now = 104.0
        Reporting.gates[0].set()
        assert Reporting.reported[0].wait(BOUND_S)
        clock.now = 110.0
        Reporting.gates[1].set()
        assert Reporting.reported[1].wait(BOUND_S)
        # The drain comes 20 s later. The stamps are the worker's, not this moment's.
        clock.now = 130.0
        running = page.status()
        assert running.running == "reporting"
        assert running.progress == Progress("reporting", 2, 3, "field-b")
        assert running.clock == RunClock(elapsed_s=30.0, first=(4.0, 1), last=(10.0, 2))
        # The end is stamped when the worker ends, and the length is fixed from then on.
        clock.now = 135.0
        Reporting.gates[2].set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        clock.now = 500.0
        ended = page.status()
        assert ended.running is None
        assert ended.clock == RunClock(elapsed_s=35.0, first=(4.0, 1), last=(10.0, 2))
        # The report the tick drains late is the one it held, and so is the second drain.
        assert page.status().clock == ended.clock
    assert page.join(BOUND_S)


def test_the_clock_is_cleared_when_another_activity_starts_and_fixed_when_one_ends(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(10.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        # Before any activity there is no clock.
        assert page.status().clock is None
        page.run("reporting", [])
        Reporting.gates[0].set()
        Reporting.gates[1].set()
        assert Reporting.reported[1].wait(BOUND_S)
        clock.now = 17.0
        Reporting.gates[2].set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        first_run = page.status()
        assert first_run.progress == Progress("reporting", 2, 3, "field-b")
        assert first_run.clock == RunClock(elapsed_s=7.0, first=(0.0, 1), last=(0.0, 2))

        # Another activity starts: its reports and its clock are its own, not the scan's.
        clock.now = 40.0
        page.run("quick", [])
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        clock.now = 99.0
        second_run = page.status()
        assert second_run.progress is None
        assert second_run.clock == RunClock(elapsed_s=0.0, first=None, last=None)
    assert page.join(BOUND_S)


def test_a_tick_that_drains_the_queue_leaves_the_clock_as_the_worker_made_it(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first report is the first one the worker made, whichever status call sees it, and
    # the latest is the latest: two status calls between two reports change nothing of
    # what the first one held.
    clock = Clock(0.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        page.run("reporting", [])
        clock.now = 3.0
        Reporting.gates[0].set()
        assert Reporting.reported[0].wait(BOUND_S)
        early = page.status()
        assert early.clock == RunClock(elapsed_s=3.0, first=(3.0, 1), last=(3.0, 1))
        clock.now = 9.0
        Reporting.gates[1].set()
        assert Reporting.reported[1].wait(BOUND_S)
        later = page.status()
        assert later.clock == RunClock(elapsed_s=9.0, first=(3.0, 1), last=(9.0, 2))
        Reporting.gates[2].set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
    assert page.join(BOUND_S)


class CoarseClock(Clock):
    """A clock that ticks every 15.625 ms, as ``time.monotonic()`` does on Windows below Python 3.13 (FM-197)."""

    TICK_S = 0.015625

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            monotonic=lambda: math.floor(self.now / self.TICK_S) * self.TICK_S,
            sleep=time.sleep,
        )


def test_two_reports_in_one_clock_tick_leave_the_time_unknown_and_never_divide_by_zero(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 100.0 is a multiple of the tick, so the start reads exactly 100.0; the two reports come 8
    # ms apart inside one tick and carry the same stamp. The controller keeps both, and *Left*
    # says it is not known rather than dividing by a pace of zero.
    clock = CoarseClock(100.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        page.run("reporting", [])
        clock.now = 100.001
        Reporting.gates[0].set()
        assert Reporting.reported[0].wait(BOUND_S)
        clock.now = 100.009
        Reporting.gates[1].set()
        assert Reporting.reported[1].wait(BOUND_S)
        status = page.status()
        assert status.clock == RunClock(elapsed_s=0.0, first=(0.0, 1), last=(0.0, 2))
        assert status.clock is not None
        assert scan_progress_lines(2, 3, status.clock, running=True) == (
            "Field 2 of 3",
            "Elapsed 0 s",
            "Left: not known yet (it needs two fields)",
        )
        Reporting.gates[2].set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
    assert page.join(BOUND_S)


def test_two_reports_in_two_clock_ticks_give_a_finite_time_that_is_never_negative(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = CoarseClock(100.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        page.run("reporting", [])
        clock.now = 100.001
        Reporting.gates[0].set()
        assert Reporting.reported[0].wait(BOUND_S)
        # 0.2 s later: twelve ticks on, and the clock reads 100.1875.
        clock.now = 100.2
        Reporting.gates[1].set()
        assert Reporting.reported[1].wait(BOUND_S)
        status = page.status()
        assert status.clock == RunClock(
            elapsed_s=0.1875, first=(0.0, 1), last=(0.1875, 2)
        )
        assert status.clock is not None
        # One field left at 0.1875 s a field: under half a second, said as 0 s, and said.
        assert scan_progress_lines(2, 3, status.clock, running=True)[-1] == (
            "Left about 0 s (estimated from the fields so far)"
        )
        Reporting.gates[2].set()
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
    assert page.join(BOUND_S)


def test_live_has_no_clock(
    page: PlatePage, fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_microscope.override(Camera, LiveCamera())
    clock = Clock(10.0)
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", clock.namespace())
        # A run before it has a clock, which Live's start clears with its progress.
        page.run("quick", [])
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        assert page.status().clock == RunClock(0.0, None, None)
        page.live(True)
        try:
            # It sets its own progress, from the worker, and no clock: a bounded count of
            # polls, never a deadline on the clock this test owns (FM-44).
            seen: Progress | None = None
            for _ in range(200):
                status = page.status()
                assert status.clock is None
                seen = status.progress
                if seen is not None:
                    break
                threading.Event().wait(0.05)
            assert seen is not None
            assert seen.plugin == "live"
        finally:
            page.live(False)
        # The thread, not `page.join`: that reads the patched clock, which only this test moves.
        _worker(page).join(BOUND_S)
        assert not _worker(page).is_alive()
        assert page.status().clock is None
    assert page.join(BOUND_S)


# --- the plan and the scan in the view (m5 §17.10.2) -----------------------------------


def _plate() -> PlateResult:
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )


def _scan_plan_result(
    wells: tuple[str, ...] = ("B11", "B2", "A1"),
    *,
    estimate_s: float | None = 42.16,
    objective: str | None = TEN_X,
) -> ScanPlanResult:
    """A scan plan built by ``plan_fields`` over a registered plate: B11, then B2, then A1 (FM-204)."""
    fov_um = (512.0, 512.0)
    half_w, half_h = fov_um[0] / 2, fov_um[1] / 2
    plate_plan, fields = plan_fields(
        _plate(),
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
        objective=objective,
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


def _scan_result(plan: ScanPlanResult, *, failed: Sequence[str] = ()) -> ScanResult:
    """The scan of ``plan``: a frame for each field but the ``failed`` ones, in the reverse of the plan's order.

    The frames come reversed so that a view which matches a frame to a field by its place in
    the list, and not by its name, is wrong on every field.
    """
    frames = tuple(
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
        for field in reversed(plan.fields)
        if field.name not in failed
    )
    return ScanResult(
        folder="scan",
        objective=plan.objective,
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        image_shape_px=(512, 512),
        exposure_ms=10.0,
        frames=frames,
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
    result: dict[str, Any],
    *,
    minute: int,
    status: StageStatus = StageStatus.done,
    profile: str = "demo",
    dry_run: bool = False,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A record as an earlier run of ``plugin`` left it; ``minute`` orders the starts, the attempt is ``a<minute>``."""
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
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=result,
            reads=reads or {},
        ),
    )


READS_THE_PLAN = {"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")}


def _seed_plan_and_scan(
    run_doc: Path,
    plan: ScanPlanResult,
    scan: ScanResult,
    *,
    dry_run: bool = False,
    profile: str = "demo",
    status: StageStatus = StageStatus.done,
) -> None:
    _seed(
        run_doc,
        "scan-plan",
        plan.model_dump(mode="json"),
        minute=1,
        dry_run=dry_run,
        profile=profile,
    )
    _seed(
        run_doc,
        "scan",
        scan.model_dump(mode="json"),
        minute=2,
        dry_run=dry_run,
        profile=profile,
        status=status,
        reads=READS_THE_PLAN,
    )


def _plan_page(microscope: Microscope, run_doc: Path) -> PlatePage:
    return PlatePage(microscope, run_doc, entry_points=_entry_points())


def test_the_plan_layer_carries_the_wells_the_objective_and_the_estimate(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    plan = _scan_plan_result()
    _seed(run_doc, "scan-plan", plan.model_dump(mode="json"), minute=1)
    layer = _plan_page(fake_microscope, run_doc).view().plan
    assert layer is not None
    # In the plan's visiting order, which is neither the alphabetical nor the natural one.
    assert layer.well_names == ("B11", "B2", "A1")
    assert (layer.objective, layer.estimate_s) == (TEN_X, 42.16)
    assert (layer.count, layer.wells) == (12, 3)
    # No scan yet: nothing is imaged and nothing is summarised.
    assert layer.imaged == ()


def test_a_plan_made_with_no_timing_has_no_estimate_and_no_objective_when_it_has_none(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    plan = _scan_plan_result(estimate_s=None, objective=None)
    _seed(run_doc, "scan-plan", plan.model_dump(mode="json"), minute=1)
    layer = _plan_page(fake_microscope, run_doc).view().plan
    assert layer is not None
    # An unknown is None, never 0.0 (FM-170).
    assert (layer.objective, layer.estimate_s) == (None, None)


def test_the_plan_layer_marks_the_fields_the_scan_has_a_frame_for(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    plan = _scan_plan_result()
    # The first field of B2 failed: it has no frame, so it is not marked.
    failed = plan.fields[4].name
    assert plan.fields[4].well == "B2"
    _seed_plan_and_scan(run_doc, plan, _scan_result(plan, failed=[failed]))
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is not None
    # The plan's order, B11's fields first (the frames are in the reverse order), and
    # without the failed field's centre.
    assert view.plan.imaged == tuple(
        (field.x_um, field.y_um) for field in plan.fields if field.name != failed
    )
    assert len(view.plan.imaged) == 11
    assert (plan.fields[4].x_um, plan.fields[4].y_um) not in view.plan.imaged
    assert view.plan.imaged[0] == (plan.fields[0].x_um, plan.fields[0].y_um)
    assert plan.fields[0].well == "B11"
    # The result line's numbers are the record's own.
    assert view.scanned == ScanSummary(frames=11, failed=1, dry_run=False)
    # The plan's own fields stay all drawn: the imaged ones are over them.
    assert len(view.plan.fields) == 12


def test_a_scan_with_every_frame_marks_every_field(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    plan = _scan_plan_result()
    _seed_plan_and_scan(run_doc, plan, _scan_result(plan))
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is not None
    assert view.plan.imaged == view.plan.fields
    assert view.scanned == ScanSummary(frames=12, failed=0, dry_run=False)


def test_a_dry_runs_scan_has_a_summary_and_no_imaged_field(
    fake_core: FakeCore, run_doc: Path
) -> None:
    # A dry run walks the fields and takes no frame: the record says so, and the map
    # claims nothing (the page is a dry-run one, so the record is usable).
    plan = _scan_plan_result()
    scan = _scan_result(plan, failed=[field.name for field in plan.fields])
    scan = scan.model_copy(update={"failed": ()})
    _seed_plan_and_scan(run_doc, plan, scan, dry_run=True)
    microscope = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        view = _plan_page(microscope, run_doc).view()
    finally:
        microscope.close()
    assert view.plan is not None
    assert view.plan.imaged == ()
    assert view.scanned == ScanSummary(frames=0, failed=0, dry_run=True)


def test_a_scan_made_before_the_plan_was_laid_out_again_has_no_summary_and_no_imaged_field(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    plan = _scan_plan_result()
    _seed_plan_and_scan(run_doc, plan, _scan_result(plan))
    # The fields were laid out again (a3) after the scan read the plan's first attempt (a1).
    _seed(run_doc, "scan-plan", plan.model_dump(mode="json"), minute=3)
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is not None
    assert view.plan.imaged == ()
    assert view.scanned is None
    # Stale is not a problem of the document: the checklist and the card say it (FM-154).
    assert view.problems == ()
    scan = {item.slot: item for item in view.checklist}["scan"]
    assert (scan.state, scan.reason) == (
        "unusable",
        "scan read its inputs from an earlier run; run it again",
    )


@pytest.mark.parametrize(
    ("profile", "dry_run"),
    [("another-stand", False), ("demo", True)],
    ids=["another-profile", "dry-run-on-a-live-page"],
)
def test_a_scan_of_another_profile_or_a_dry_run_on_a_live_page_is_not_the_views(
    fake_microscope: Microscope, run_doc: Path, profile: str, dry_run: bool
) -> None:
    plan = _scan_plan_result()
    _seed(run_doc, "scan-plan", plan.model_dump(mode="json"), minute=1)
    _seed(
        run_doc,
        "scan",
        _scan_result(plan).model_dump(mode="json"),
        minute=2,
        profile=profile,
        dry_run=dry_run,
        reads=READS_THE_PLAN,
    )
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.scanned is None
    assert view.plan is not None
    assert view.plan.imaged == ()


@pytest.mark.parametrize("status", [StageStatus.stopped, StageStatus.failed])
def test_a_scan_that_did_not_end_done_has_no_summary_and_marks_no_field(
    fake_microscope: Microscope, run_doc: Path, status: StageStatus
) -> None:
    # Its frames are on disk and its checkpoint continues it, but the map claims only what a
    # usable record holds (m5 §17.10.2): the result is not a result.
    plan = _scan_plan_result()
    _seed_plan_and_scan(run_doc, plan, _scan_result(plan), status=status)
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.scanned is None
    assert view.plan is not None
    assert view.plan.imaged == ()


def test_a_scan_that_does_not_validate_is_a_problem_and_no_summary(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-125: a hand-edited document is data; view() raises nothing for it.
    plan = _scan_plan_result()
    broken = _scan_result(plan).model_dump(mode="json")
    del broken["folder"]
    _seed(run_doc, "scan-plan", plan.model_dump(mode="json"), minute=1)
    _seed(run_doc, "scan", broken, minute=2, reads=READS_THE_PLAN)
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.scanned is None
    assert view.problems == ("scan: result does not validate: folder: Field required",)
    # The plan is the plan's own, and marks nothing.
    assert view.plan is not None
    assert view.plan.imaged == ()


def test_the_imaged_fields_are_bounded_like_the_plan(
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-140: the layer holds a bounded payload whatever the document holds. The bound is
    # patched down (12 fields in this plan): on it everything is drawn, one over it nothing.
    plan = _scan_plan_result()
    _seed_plan_and_scan(run_doc, plan, _scan_result(plan))
    page = _plan_page(fake_microscope, run_doc)
    for bound, drawn in ((12, True), (11, False)):
        with monkeypatch.context() as patch:
            patch.setattr(controller_module, "MAP_FIELDS_UP_TO", bound)
            layer = page.view().plan
        assert layer is not None
        assert layer.count == 12
        assert (len(layer.fields) == 12) is drawn
        assert (len(layer.imaged) == 12) is drawn
        assert (layer.imaged == ()) is (not drawn)
        # The numbers of the result line are not drawn payload: the record's own.
        assert page.view().scanned == ScanSummary(frames=12, failed=0, dry_run=False)


def test_the_view_of_a_page_with_no_scan_has_none_and_reads_nothing_else_for_it(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.scanned is None
    assert view.plan is None
    assert view.problems == ()


def test_a_report_with_no_activity_is_queued_and_stamps_nothing_it_could_not_time(
    page: PlatePage,
) -> None:
    # `_on_progress` is what the runner calls; a report that arrives with no activity (a
    # test, a late plugin) is queued as before and stamps nothing it could not time.
    page._on_progress(Progress("scan", 1, 2, "B2_0000"))
    status = page.status()
    assert status.progress == Progress("scan", 1, 2, "B2_0000")
    assert status.clock is None


# --- part B: the scan after a STOP or a failure (m5 §17.12 D17, #470) ------------------------


class CheckpointScanParams(PluginParams):
    step: int = 1


class CheckpointScanResult(PluginResult):
    resumed_from: int | None


class _Kept(PluginParams):
    """The checkpoint the stub keeps: how many fields it took."""

    taken: int


class CheckpointScan:
    """A ``scan`` of three fields that checkpoints after the first and waits at a gate there.

    ``ending`` says how it ends at the gate: ``stop`` raises ``StopRequested`` once STOP was
    pressed (a ``stopped`` record), ``move`` raises ``MotionStoppedError`` as a stage stopped in
    a move does (a ``failed`` record, FM-64), ``fail`` raises a plain error whether or not STOP
    was pressed. ``resumed`` holds what each run read of the checkpoint (``None``: afresh).
    """

    name = provides = "scan"
    summary = "A test scan that keeps a checkpoint."
    Params = CheckpointScanParams
    Result = CheckpointScanResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    ending: ClassVar[str] = "stop"
    paused: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()
    resumed: ClassVar[list[int | None]] = []

    def run(
        self, microscope: Any, params: CheckpointScanParams, ctx: RunContext
    ) -> CheckpointScanResult:
        kept = ctx.checkpoint(_Kept)
        start = 0 if kept is None else kept.taken
        CheckpointScan.resumed.append(None if kept is None else kept.taken)
        for done in range(start + 1, 4):
            ctx.progress(done, 3, f"F{done}")
            if done == 1 and kept is None:
                ctx.save_checkpoint(_Kept(taken=1))
                CheckpointScan.paused.set()
                CheckpointScan.release.wait(BOUND_S)
                if CheckpointScan.ending == "fail":
                    raise RuntimeError("the camera timed out")
                if CheckpointScan.ending == "move" and ctx.should_stop():
                    raise MotionStoppedError("xy_stage XY")
                ctx.raise_if_stopped()
        return CheckpointScanResult(resumed_from=None if kept is None else kept.taken)


def _scan_entry_points(*, with_others: bool = True) -> list[EntryPoint]:
    scan = EntryPoint(name="scan", value=f"{__name__}:CheckpointScan", group=GROUP)
    return [scan, *(_entry_points() if with_others else [])]


@pytest.fixture
def scan_page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    CheckpointScan.ending = "stop"
    CheckpointScan.paused = threading.Event()
    CheckpointScan.release = threading.Event()
    CheckpointScan.resumed = []
    controller = PlatePage(fake_microscope, run_doc, entry_points=_scan_entry_points())
    try:
        yield controller
    finally:
        CheckpointScan.release.set()
        controller.close()


def _stop_the_scan(page: PlatePage, params: Sequence[str] = ()) -> None:
    """Start the stub scan, STOP it at its gate, wait for its end and clear the halt."""
    page.run("scan", list(params))
    assert CheckpointScan.paused.wait(BOUND_S)
    page.stop()
    CheckpointScan.release.set()
    assert page.join(BOUND_S)
    page.resume()


def test_continue_runs_the_scan_with_resume_and_the_runners_checks(
    scan_page: PlatePage, run_doc: Path
) -> None:
    _stop_the_scan(scan_page)
    end = scan_page.view().scan_end
    assert end is not None
    assert (end.kind, end.done, end.total, end.resumable) == ("stopped", 1, 3, True)
    # A plain click over the checkpoint starts nothing: the page's words, not "close this page".
    before = run_doc.read_bytes()
    scan_page.run("scan", [])
    outcome = scan_page.status().outcome
    assert outcome is not None
    assert "Continue the scan, on the Scan screen" in outcome
    assert "close this page first" not in outcome
    assert run_doc.read_bytes() == before
    assert CheckpointScan.resumed == [None]
    # Continue: the runner's resume, which hands the plugin its checkpoint.
    scan_page.run("scan", [], resume=True)
    assert scan_page.join(BOUND_S)
    assert CheckpointScan.resumed == [None, 1]
    record = RunDocument.load_or_new(run_doc).stages["scan"]
    assert record.status is StageStatus.done
    assert record.result is not None
    assert record.result["resumed_from"] == 1
    assert scan_page.view().scan_end is None
    # A resume of a stage that finished is the runner's refusal, said as the outcome.
    scan_page.run("scan", [], resume=True)
    outcome = scan_page.status().outcome
    assert outcome is not None
    assert "cannot resume scan: scan finished" in outcome
    assert CheckpointScan.resumed == [None, 1]


def test_start_again_afresh_drops_the_checkpoint_for_that_call_only(
    scan_page: PlatePage, run_doc: Path
) -> None:
    _stop_the_scan(scan_page)
    CheckpointScan.paused = threading.Event()
    CheckpointScan.release = threading.Event()
    scan_page.run("scan", [], afresh=True)
    assert CheckpointScan.paused.wait(BOUND_S)
    # Afresh: the plugin was not handed the checkpoint.
    assert CheckpointScan.resumed == [None, None]
    scan_page.stop()
    CheckpointScan.release.set()
    assert scan_page.join(BOUND_S)
    scan_page.resume()
    # The next plain click is refused again: `afresh` held for one call.
    before = run_doc.read_bytes()
    scan_page.run("scan", [])
    assert run_doc.read_bytes() == before
    assert CheckpointScan.resumed == [None, None]


def test_resume_and_afresh_together_are_refused_before_anything_starts(
    scan_page: PlatePage, run_doc: Path
) -> None:
    with pytest.raises(ValueError, match="not both"):
        scan_page.run("scan", [], resume=True, afresh=True)
    assert scan_page._worker is None
    assert not run_doc.exists()
    assert scan_page.status().outcome is None


def test_a_stop_that_lands_in_a_move_makes_a_stopped_scan_end(
    scan_page: PlatePage, run_doc: Path
) -> None:
    CheckpointScan.ending = "move"
    _stop_the_scan(scan_page)
    record = RunDocument.load_or_new(run_doc).stages["scan"]
    # The record stays `failed` (FM-64); the words follow the STOP.
    assert record.status is StageStatus.failed
    assert (record.error or "").startswith("MotionStoppedError: ")
    end = scan_page.view().scan_end
    assert end is not None
    assert (end.kind, end.done, end.total, end.resumable) == ("stopped", 1, 3, True)
    # Another activity since (a quick run) neither makes the stop a failure nor loses the count.
    scan_page.run("quick", [])
    assert scan_page.join(BOUND_S)
    end = scan_page.view().scan_end
    assert end is not None
    assert (end.kind, end.done, end.total) == ("stopped", 1, 3)


def test_a_failure_with_no_stop_is_a_failed_scan_end_with_the_records_error(
    scan_page: PlatePage,
) -> None:
    CheckpointScan.ending = "fail"
    scan_page.run("scan", [])
    assert CheckpointScan.paused.wait(BOUND_S)
    CheckpointScan.release.set()
    assert scan_page.join(BOUND_S)
    end = scan_page.view().scan_end
    assert end is not None
    assert (end.kind, end.done, end.total, end.resumable) == ("failed", 1, 3, True)
    assert end.text == "RuntimeError: the camera timed out"


def test_a_scan_end_seen_by_a_new_page_has_no_field_count(
    scan_page: PlatePage, fake_microscope: Microscope, run_doc: Path
) -> None:
    CheckpointScan.ending = "move"
    _stop_the_scan(scan_page)
    scan_page.close()
    fresh = PlatePage(
        fake_microscope, run_doc, entry_points=_scan_entry_points(with_others=False)
    )
    try:
        end = fresh.view().scan_end
    finally:
        fresh.close()
    assert end is not None
    # This page kept no progress and saw no STOP: the record's words, without a count.
    assert (end.kind, end.done, end.total, end.resumable) == (
        "failed",
        None,
        None,
        True,
    )
    assert end.text.startswith("MotionStoppedError: ")


def test_continue_with_fields_changed_since_the_stop_says_the_runners_refusal(
    scan_page: PlatePage, run_doc: Path
) -> None:
    _stop_the_scan(scan_page, ["step=1"])
    before = run_doc.read_bytes()
    scan_page.run("scan", ["step=2"], resume=True)
    assert scan_page.status().outcome == (
        "scan: ResumeError: cannot resume scan: the parameters differ from the run "
        "being resumed: step"
    )
    # Nothing ran and nothing was written: the checkpoint still stands.
    assert run_doc.read_bytes() == before
    assert CheckpointScan.resumed == [None]
    end = scan_page.view().scan_end
    assert end is not None
    assert end.resumable


def test_a_scan_with_no_record_or_a_done_record_has_no_scan_end(
    scan_page: PlatePage,
) -> None:
    assert scan_page.view().scan_end is None
    CheckpointScan.release.set()
    scan_page.run("scan", [])
    assert scan_page.join(BOUND_S)
    assert scan_page.view().scan_end is None
