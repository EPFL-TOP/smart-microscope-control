"""The plate page's controller on ``fake_microscope`` and stub plugins (m3 §15, #197).

No Panel here: the controller decides everything the page shows or does. Stub
plugins are classes of this module, loaded through ``EntryPoint`` and handed to
the page as ``entry_points=``; the real ``plate-registration``, ``scan-plan``,
``scan``... are loaded the same way from their own ``value``, never from the
installed group, so the tests do not depend on the checkout's install (FM-112).

Expected values come from the forward model of m3 §2.1, written here (``world =
A1 + R(+theta) * (col * pitch_x, -row * pitch_y)``, a 96-well pitch of 9 mm and
wells 6.4 mm wide), never read from the code (FM-124); a plate that is turned
(7°) and off the origin is used wherever the axes matter (FM-73). Texts are
asserted whole (FM-45), threads meet on ``Event`` and every ``join`` is checked
(FM-44), and a hook that runs inside a swallowing call stores what it saw and the
test asserts after (FM-94).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import string
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest
import useq
from pydantic import Field, field_validator
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, XYStage, ZStage
from smc.hardware.errors import HardwareError, StopRefusedError
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.plugins import runner as runner_mod
from smc.plugins.api import (
    PluginParams,
    PluginResult,
    Progress,
    RunContext,
    StopRequested,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.params import parse_params
from smc.plugins.registry import GROUP, LoadFailure, Registry
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    DensityResult,
    DensityWell,
    PlateResult,
    RouteResult,
    ScanPlanResult,
    WallFrame,
    WallImages,
    WallTouch,
    WellFit,
    WellRef,
)
from smc.run import (
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)
from smc.testing import FakeCore
from smc.ui.controller import (
    KEY_PARAMS,
    LIVE_PERIOD_S,
    PAGE_PARAMS,
    CheckItem,
    PlatePage,
)
from smc.ui.frames import Frame
from smc.ui.guide import (
    NO_CALIBRATION_FOR_SIDES,
    WALLS_DRY_RUN,
    WALLS_NEED_CALIBRATION,
    scan_plan_card_lines,
    walls_centres_problem,
    walls_square_wells,
)

T0 = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
PITCH_96_UM = 9000.0
PITCH_384_UM = 4500.0
A1_OFF = (1000.0, -500.0)
TURN_DEG = 7.0

#: The real plugins, by the ``value`` of their own entry points.
REAL = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "cells": "smc.plugins.cells.plugin:CellsPlugin",
    "density": "smc.plugins.density.plugin:DensityPlugin",
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "plate-finding": "smc.plugins.plate_finding.plugin:PlateFindingPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
    "route": "smc.plugins.route.plugin:RoutePlugin",
    "scan": "smc.plugins.scan.plugin:ScanPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
}


# --- stub plugins -------------------------------------------------------------


class StubParams(PluginParams):
    n: int = 1
    names: tuple[str, ...] = ()


class GotoParams(PluginParams):
    well: str
    side: str = "centre"


class NoParams(PluginParams):
    pass


class StubResult(PluginResult):
    n: int = 0


class Stub:
    """Returns ``StubResult(n=n)``; remembers the parameters it was given."""

    name = "stub"
    summary = "A test plugin."
    Params = StubParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    provides = "stub"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[StubParams]] = []

    def run(self, microscope: Any, params: StubParams, ctx: RunContext) -> StubResult:
        Stub.seen.append(params)
        return StubResult(n=params.n)


class Reader(Stub):
    name = provides = "reader"
    reads = ("up",)


class StubGoto:
    """A ``goto-well`` that records the well and the side and moves nothing."""

    name = provides = "goto-well"
    summary = "A test goto-well."
    Params = GotoParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()
    ran: ClassVar[list[tuple[str, str]]] = []

    def run(self, microscope: Any, params: GotoParams, ctx: RunContext) -> StubResult:
        StubGoto.ran.append((params.well, params.side))
        return StubResult()


class Slow:
    """Runs until released or stopped; ``ended`` is set in a ``finally``."""

    name = provides = "slow"
    summary = "A test plugin that waits."
    Params = NoParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()
    ended: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        Slow.started.set()
        try:
            while not Slow.release.is_set():
                ctx.wait_s(0.05)
        finally:
            Slow.ended.set()
        return StubResult()


class Failing:
    """Raises a ``PluginRunError`` whose message has a newline, and keeps a weak reference to it."""

    name = provides = "failing"
    summary = "A test plugin that fails."
    Params = NoParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    raised: ClassVar[list[weakref.ref[BaseException]]] = []

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        # No local holds the error: a local would be a cycle through its traceback.
        raise _track(PluginRunError("failing", "line one\nline two"))


def _track(error: PluginRunError) -> PluginRunError:
    Failing.raised.append(weakref.ref(error))
    return error


class Progressing(Stub):
    name = provides = "progressing"

    def run(self, microscope: Any, params: StubParams, ctx: RunContext) -> StubResult:
        ctx.progress(1, 3, "one of three")
        return StubResult()


class Stubborn:
    """Ignores the stop: it never calls ``ctx.wait_s``, and ends only when released."""

    name = provides = "stubborn"
    summary = "A test plugin that ignores the stop."
    Params = NoParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        Stubborn.started.set()
        Stubborn.release.wait(30)
        return StubResult()


class FailsOnStop(Slow):
    """Inside a move when the stop lands: the move fails, and so does the run."""

    name = provides = "fails"

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        Slow.started.set()
        try:
            while True:
                ctx.wait_s(0.05)
        except StopRequested:
            raise PluginRunError("fails", "the move was stopped") from None


class Saves(Slow):
    """Saves a checkpoint, then waits for the stop: what a scan does after its first well."""

    name = provides = "saves"

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        ctx.save_checkpoint(StubResult(n=1))
        Slow.started.set()
        while True:
            ctx.wait_s(0.05)


class Assuming(Stub):
    """Ends ``done`` with the ``assumed`` lines a test sets in ``assumed``, as ``scan-plan`` does."""

    name = provides = "assuming"
    assumed: ClassVar[tuple[str, ...]] = ()

    def run(self, microscope: Any, params: StubParams, ctx: RunContext) -> StubResult:
        return StubResult(assumed=Assuming.assumed)


class SavesAssuming(Saves):
    """Saves a checkpoint that holds an assumption, then waits for the stop."""

    name = provides = "saves-assuming"

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        ctx.save_checkpoint(StubResult(n=1, assumed=("a guess",)))
        Slow.started.set()
        while True:
            ctx.wait_s(0.05)


class StubScan(Stub):
    """Registered as ``scan``, and needs an XY stage."""

    name = provides = "scan"
    requires = (XYStage,)


STUBS = (Stub, Reader, StubGoto, Slow, Failing, Progressing)


def _ep(cls: type) -> EntryPoint:
    return EntryPoint(
        name=cast("Any", cls).name, value=f"{__name__}:{cls.__name__}", group=GROUP
    )


def _real(*names: str) -> list[EntryPoint]:
    return [EntryPoint(name=n, value=REAL[n], group=GROUP) for n in names]


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    """The stubs' class-level Events and logs are new for every test, and released after it."""
    Stub.seen = []
    StubGoto.ran = []
    Assuming.assumed = ()
    Failing.raised = []
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    Slow.ended = threading.Event()
    Stubborn.started = threading.Event()
    Stubborn.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()  # a worker a failed test left behind ends
        Stubborn.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _page(
    microscope: Microscope, run_doc: Path, extra: list[EntryPoint] | None = None
) -> PlatePage:
    return PlatePage(
        microscope, run_doc, entry_points=[*(_ep(c) for c in STUBS), *(extra or [])]
    )


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = _page(fake_microscope, run_doc)
    try:
        yield controller
    finally:
        controller.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _seed(
    run_doc: Path,
    plugin: str,
    provides: str,
    result: PluginResult | None = None,
    *,
    status: StageStatus = StageStatus.done,
    params: dict[str, Any] | None = None,
    minute: int = 0,
    profile: str = "demo",
    dry_run: bool = False,
    checkpoint: dict[str, Any] | None = None,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A record as an earlier run of ``plugin`` left it; ``minute`` orders the starts.

    ``reads`` names the attempt of each slot the record read (``a<minute>`` of
    the provider): a record that read an older attempt than the current one is
    stale.

    A plate result with no ``params`` gets the plate it holds, as the runner writes
    it: the page starts on the plate a record's params name (m5 §11.3), and a real
    record always names one.
    """
    if params is None and isinstance(result, PlateResult):
        params = {"plate": result.plate}
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt=f"a{minute}",
            status=status,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params=params or {},
            result=None if result is None else result.model_dump(mode="json"),
            checkpoint=checkpoint,
            reads=reads or {},
        ),
    )


def _plate(
    plate: str = "96-well",
    a1: tuple[float, float] = A1_OFF,
    rotation_deg: float = TURN_DEG,
) -> PlateResult:
    return PlateResult(
        plate=plate,
        a1_center_xy_um=a1,
        rotation_deg=rotation_deg,
        method="references",
        residual_um=2.0,
        scale_error=0.001,
        n_refs=2,
    )


def _centre(
    well: str,
    *,
    a1: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
    pitch_um: float = PITCH_96_UM,
) -> tuple[float, float]:
    """The forward model of m3 §2.1, written out: ``A1 + R(+theta) * (col * px, -row * py)``."""
    row = string.ascii_uppercase.index(well[0])
    col = int(well[1:]) - 1
    angle = math.radians(rotation_deg)
    dx, dy = col * pitch_um, -row * pitch_um
    return (
        a1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        a1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _refuses(error: type[Exception], text: str) -> AbstractContextManager[Any]:
    """``pytest.raises`` that insists on the whole message, not on a word of it (FM-45)."""
    return pytest.raises(error, match=f"^{re.escape(text)}$")


def _run_and_join(
    page: PlatePage, plugin: str, params: list[str] | None = None
) -> None:
    page.run(plugin, params or [])
    assert page.join(10.0)


# --- the selection: a click and a typed range ------------------------------------


def test_a_click_and_a_typed_range_build_the_same_selection(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    typed = _page(fake_microscope, run_doc)
    clicked = _page(fake_microscope, run_doc)
    typed.select_text("A1:B2")
    for well in ("A1", "A2", "B1", "B2"):
        x, y = _centre(well)
        assert clicked.click(x, y, drive=False) == well
    assert typed.selection == ("A1", "A2", "B1", "B2")
    assert clicked.selection == typed.selection
    x, y = _centre("A1")
    assert clicked.click(x, y, drive=False) == "A1"
    assert clicked.selection == ("A2", "B1", "B2")
    # A1 was deselected and was not the well the buttons act on: B2, the last well
    # selected, still is (m5 §11.1).
    assert clicked.current == "B2"


def test_a_click_finds_the_well_on_a_rotated_registered_plate(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    x, y = _centre("H12", a1=A1_OFF, rotation_deg=TURN_DEG)
    assert page.click(x, y, drive=False) == "H12"
    assert page.selection == ("H12",)
    # Half a pitch from the centre, along the plate's own X axis: between wells.
    angle = math.radians(TURN_DEG)
    off = (x - math.cos(angle) * PITCH_96_UM / 2, y - math.sin(angle) * PITCH_96_UM / 2)
    assert page.click(*off, drive=False) is None
    assert page.selection == ("H12",)
    # The picture is in stage axes: the same numbers with x and y exchanged are no well.
    assert page.click(y, x, drive=False) is None


def test_a_click_on_no_well_or_a_point_that_is_not_finite_does_nothing(
    page: PlatePage,
) -> None:
    assert page.click(-50000.0, 50000.0, drive=False) is None
    assert page.click(math.nan, 0.0, drive=True) is None
    assert page.click(0.0, math.inf, drive=False) is None
    assert page.selection == ()
    assert page.current is None


def test_a_click_in_select_mode_moves_nothing(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A registered plate and the real `goto-well`: a click that also drove would move
    # the stage, and the run document would hold its record.
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = PlatePage(fake_microscope, run_doc, entry_points=_real("goto-well"))
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    for well in ("A1", "C5", "H12"):
        x, y = _centre(well, a1=A1_OFF, rotation_deg=TURN_DEG)
        assert page.click(x, y, drive=False) == well
    assert page.join(10.0)
    assert stage.position_um() == before
    assert page.status().running is None
    assert page.status().outcome is None
    assert "goto-well" not in RunDocument.load(run_doc).stages
    assert page.selection == ("A1", "C5", "H12")


def test_a_drive_click_on_another_plates_map_moves_nothing(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate("384-well"))
    page = _page(fake_microscope, run_doc)
    assert page.plate == "384-well"
    page.choose_plate("96-well")
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    refusal = (
        "the registered plate is a 384-well; calibrate the 96-well plate again, "
        "or choose 384-well"
    )
    page.select_text("A1")
    with _refuses(ValueError, refusal):
        page.page_params("scan-plan")
    with _refuses(ValueError, refusal):
        page.click(*_centre("C5"), drive=True)
    with _refuses(ValueError, refusal):
        page.goto("C5")
    assert StubGoto.ran == []
    assert stage.position_um() == before
    assert page.status().running is None
    assert page.current is None
    page.choose_plate("384-well")
    x, y = _centre("C5", a1=A1_OFF, rotation_deg=TURN_DEG, pitch_um=PITCH_384_UM)
    assert page.click(x, y, drive=True) == "C5"
    assert page.join(10.0)
    assert StubGoto.ran == [("C5", "centre")]


def test_a_click_in_drive_mode_runs_goto_well_with_the_wells_name_and_side(
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    seed_camera_calibration(run_doc)  # the Ti2-E's: the image's left is stage +X
    page = _page(fake_microscope, run_doc)
    x, y = _centre("B3", a1=A1_OFF, rotation_deg=TURN_DEG)
    assert page.click(x, y, drive=True, side="left") == "B3"
    assert page.join(10.0)
    assert StubGoto.ran == [("B3", "+X")]
    assert page.current == "B3"
    assert page.selection == ()
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("goto-well done in ")


# --- `current` is a well the operator can see (m5 §11.1, #332) ----------------------


def _tap(page: PlatePage, well: str) -> None:
    """A select-mode click on the centre of ``well`` of the nominal map."""
    assert page.click(*_centre(well), drive=False) == well


def test_a_click_that_deselects_the_current_well_drops_it(page: PlatePage) -> None:
    _tap(page, "B3")
    assert (page.selection, page.current) == (("B3",), "B3")
    # The measured case of the session at the stand: the well the buttons act on
    # was one the map no longer showed as selected.
    _tap(page, "B3")
    assert (page.selection, page.current) == ((), None)


def test_a_click_that_deselects_another_well_leaves_current(page: PlatePage) -> None:
    _tap(page, "B3")
    _tap(page, "B4")
    assert page.current == "B4"
    _tap(page, "B3")
    assert (page.selection, page.current) == (("B4",), "B4")


def test_a_typed_range_drops_a_current_well_it_does_not_hold(page: PlatePage) -> None:
    # The measured case: B3 selected and current, then `C1:C2` typed.
    _tap(page, "B3")
    page.select_text("C1:C2")
    assert (page.selection, page.current) == (("C1", "C2"), None)
    # A box the operator emptied replaces the selection by none: the same rule.
    _tap(page, "C3")
    assert page.current == "C3"
    page.select_text("  ")
    assert (page.selection, page.current) == ((), None)


def test_a_typed_range_keeps_a_current_well_it_holds(page: PlatePage) -> None:
    _tap(page, "B3")
    page.select_text("B1:B4")
    assert (page.selection, page.current) == (("B1", "B2", "B3", "B4"), "B3")


def test_a_typed_range_that_is_refused_leaves_current_as_it_was(
    page: PlatePage,
) -> None:
    _tap(page, "B3")
    with _refuses(ValueError, "wells not understood on a 96-well plate: 'Z99'"):
        page.select_text("Z99")
    assert (page.selection, page.current) == (("B3",), "B3")


def test_goto_sets_current_without_selecting_and_a_typed_range_drops_it(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    try:
        page.goto("B3")
        assert page.join(10.0)
        assert (page.selection, page.current) == ((), "B3")
        # A driven-to well is not selected; a typed range replaces the selection, and
        # one that does not hold it drops it too.
        page.select_text("A1")
        assert (page.selection, page.current) == (("A1",), None)
    finally:
        page.close()


def test_a_deselect_of_another_well_leaves_a_driven_to_well(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    try:
        x, y = _centre("A1", a1=A1_OFF, rotation_deg=TURN_DEG)
        page.click(x, y, drive=False)
        page.goto("B3")
        assert page.join(10.0)
        # A1 is selected, B3 is current without being selected: deselecting A1
        # must not take B3 away.
        page.toggle("A1")
        assert (page.selection, page.current) == ((), "B3")
    finally:
        page.close()


# --- running a plugin ----------------------------------------------------------------


def test_run_goes_through_the_same_parameters_as_the_cli(
    page: PlatePage, run_doc: Path
) -> None:
    strings = ["n=3", 'names=["a","b"]']
    _run_and_join(page, "stub", strings)
    assert Stub.seen == [parse_params(StubParams, strings)]
    record = RunDocument.load(run_doc).stages["stub"]
    assert record.status is StageStatus.done
    assert record.params == parse_params(StubParams, strings).model_dump(mode="json")
    assert record.params == {"n": 3, "names": ["a", "b"]}


def test_run_refuses_before_a_thread_starts(page: PlatePage, run_doc: Path) -> None:
    threads = threading.active_count()
    page.run("nope", [])
    assert page.status().outcome == (
        "nope: PluginNotFoundError: no plugin named 'nope' (available: "
        "failing, goto-well, progressing, reader, slow, stub); "
        "smc run --list shows them"
    )
    page.run("stub", ["n=abc"])
    assert page.status().outcome == (
        "stub: ParamsError: parameters of stub: n: Input should be a valid "
        "integer, unable to parse string as an integer. smc run stub "
        "--show-params lists the keys"
    )
    page.run("reader", [])
    assert page.status().outcome == (
        "reader: UpstreamError: reader reads up, but the run document has no up; "
        "smc run --list shows which plugin provides it"
    )
    assert page.status().running is None
    assert not run_doc.exists()
    assert threading.active_count() == threads
    assert Stub.seen == []


def test_run_refuses_an_unreadable_run_document_and_leaves_it_alone(
    page: PlatePage, run_doc: Path
) -> None:
    run_doc.write_bytes(b"not json")
    page.run("stub", [])
    assert page.status().outcome == (
        f"stub: RunDocumentError: {run_doc}: is not valid JSON: Expecting value "
        f"(line 1, column 1); it was left as it is, restore it from a copy"
    )
    assert run_doc.read_bytes() == b"not json"
    assert page.status().running is None


def _checkpoint_text(name: str, status: str) -> str:
    """What the page says of a stage it will not start afresh (m3 §15.3), written out."""
    return (
        f"{status} with a checkpoint: continue it with `smc run {name} --resume` "
        f"(close this page first); `smc run {name}` without --resume starts afresh "
        f"and drops it"
    )


def test_run_refuses_a_stage_stopped_with_a_checkpoint_and_says_how_to_resume(
    page: PlatePage, run_doc: Path
) -> None:
    # #252: one click after STOP must not drop the work the stop saved, and no
    # command brings it back.
    _seed(run_doc, "stub", "stub", status=StageStatus.stopped, checkpoint={"done": 1})
    before = run_doc.read_bytes()
    threads = threading.active_count()
    page.run("stub", [])
    assert page.status().outcome == f"stub: {_checkpoint_text('stub', 'stopped')}"
    assert run_doc.read_bytes() == before
    assert Stub.seen == []
    assert page.status().running is None
    assert page.activities_ended == 0
    assert threading.active_count() == threads


def test_a_failed_stage_with_a_checkpoint_is_refused_too(
    page: PlatePage, run_doc: Path
) -> None:
    _seed(run_doc, "stub", "stub", status=StageStatus.failed, checkpoint={"done": 1})
    before = run_doc.read_bytes()
    page.run("stub", [])
    assert page.status().outcome == f"stub: {_checkpoint_text('stub', 'failed')}"
    assert run_doc.read_bytes() == before
    assert Stub.seen == []


@pytest.mark.parametrize("case", ["another-profile", "a-dry-run", "stale"])
def test_a_checkpoint_resume_would_refuse_does_not_block_a_run(
    page: PlatePage, run_doc: Path, case: str
) -> None:
    # Nothing is lost by replacing what no command can reach: the run goes on, as
    # `smc run` without --resume does.
    name = "stub"
    checkpoint = {"done": 1}
    if case == "stale":
        name = "reader"
        _seed(run_doc, "up", "up", minute=0)
        _seed(
            run_doc,
            "reader",
            "reader",
            status=StageStatus.stopped,
            checkpoint=checkpoint,
            minute=1,
            reads={"up": ReadRecord(plugin="up", attempt="a0")},
        )
        _seed(run_doc, "up", "up", minute=2)  # produced again since the checkpoint
    else:
        _seed(
            run_doc,
            "stub",
            "stub",
            status=StageStatus.stopped,
            checkpoint=checkpoint,
            profile="another" if case == "another-profile" else "demo",
            dry_run=case == "a-dry-run",
        )
    _run_and_join(page, name)
    record = RunDocument.load(run_doc).stages[name]
    assert record.status is StageStatus.done
    assert record.checkpoint is None
    assert page.status().outcome is not None
    assert page.status().outcome.startswith(f"{name} done in ")


def test_a_stopped_stage_without_a_checkpoint_runs_afresh(
    page: PlatePage, run_doc: Path
) -> None:
    _seed(run_doc, "stub", "stub", status=StageStatus.stopped)
    _run_and_join(page, "stub")
    assert RunDocument.load(run_doc).stages["stub"].status is StageStatus.done
    assert len(Stub.seen) == 1


def test_changed_parameters_do_not_lift_the_refusal(
    page: PlatePage, run_doc: Path
) -> None:
    # Decided in m3 §15.3: the form changes between clicks, and the loss is the same.
    _seed(
        run_doc,
        "stub",
        "stub",
        status=StageStatus.stopped,
        params={"n": 1, "names": []},
        checkpoint={"done": 1},
    )
    before = run_doc.read_bytes()
    page.run("stub", ["n=2"])
    assert page.status().outcome == f"stub: {_checkpoint_text('stub', 'stopped')}"
    assert run_doc.read_bytes() == before
    assert Stub.seen == []


def test_run_says_what_the_stage_reads_before_the_checkpoint(
    page: PlatePage, run_doc: Path
) -> None:
    # The order of `view().blocked` is the order of `run()`: a stage that cannot run
    # at all says so before it offers a way to continue.
    _seed(run_doc, "reader", "reader", status=StageStatus.stopped, checkpoint={"d": 1})
    page.run("reader", [])
    assert page.status().outcome == (
        "reader: UpstreamError: reader reads up, but the run document has no up; "
        "smc run --list shows which plugin provides it"
    )


def test_a_run_that_stops_on_its_own_page_blocks_the_next_click(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # End to end: the stop is the page's own, and so is the click that follows it.
    page = _page(fake_microscope, run_doc, [_ep(Saves)])
    try:
        page.run("saves", [])
        assert Slow.started.wait(5)
        page.stop()
        assert page.join(10.0)
        record = RunDocument.load(run_doc).stages["saves"]
        assert record.status is StageStatus.stopped
        assert record.checkpoint == StubResult(n=1).model_dump(mode="json")
        # The end of a run redraws from `view()`: the button is blocked then, not
        # only after a Refresh.
        assert page.view().blocked["saves"] == _checkpoint_text("saves", "stopped")
        stopped = run_doc.read_bytes()
        page.run("saves", [])
        assert page.status().outcome == f"saves: {_checkpoint_text('saves', 'stopped')}"
        assert run_doc.read_bytes() == stopped
        assert page.activities_ended == 1
    finally:
        page.close()


# --- the strip says what a run assumed (m5 §8.8.2) ---------------------------------------


def test_a_done_run_says_its_first_assumption_on_the_strip(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    Assuming.assumed = ("first line", "second line")
    page = _page(fake_microscope, run_doc, [_ep(Assuming)])
    try:
        _run_and_join(page, "assuming")
        outcome = page.status().outcome
        assert outcome is not None
        assert re.fullmatch(
            r"assuming done in \d+\.\d s; assumed: first line \(\+1 more, on the card\)",
            outcome,
        ), outcome
        # The card holds all of them: the strip is a pointer, not a copy.
        assert page.view().cards["assuming"].assumed == ("first line", "second line")
    finally:
        page.close()


def test_a_done_run_with_one_assumption_does_not_count_more(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    Assuming.assumed = ("only line",)
    page = _page(fake_microscope, run_doc, [_ep(Assuming)])
    try:
        _run_and_join(page, "assuming")
        outcome = page.status().outcome
        assert outcome is not None
        assert re.fullmatch(
            r"assuming done in \d+\.\d s; assumed: only line", outcome
        ), outcome
    finally:
        page.close()


def test_a_done_run_with_no_assumption_has_the_outcome_it_had(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(Assuming)])
    try:
        _run_and_join(page, "assuming")
        outcome = page.status().outcome
        assert outcome is not None
        assert re.fullmatch(r"assuming done in \d+\.\d s", outcome), outcome
    finally:
        page.close()


def test_a_long_assumption_is_cut_on_the_strip(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    Assuming.assumed = ("a" * 400,)
    page = _page(fake_microscope, run_doc, [_ep(Assuming)])
    try:
        _run_and_join(page, "assuming")
        outcome = page.status().outcome
        assert outcome is not None
        assert re.fullmatch(
            rf"assuming done in \d+\.\d s; assumed: {'a' * 160}…", outcome
        ), outcome
        # The card is cut later: the strip's cut loses nothing the card holds.
        assert page.view().cards["assuming"].assumed == ("a" * 400,)
    finally:
        page.close()


def test_an_assumption_with_a_line_break_is_one_line_on_the_strip(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    Assuming.assumed = ("pixel size\ntaken as given",)
    page = _page(fake_microscope, run_doc, [_ep(Assuming)])
    try:
        _run_and_join(page, "assuming")
        outcome = page.status().outcome
        assert outcome is not None
        assert "\n" not in outcome
        assert outcome.endswith("; assumed: pixel size taken as given"), outcome
    finally:
        page.close()


def test_a_stopped_or_failed_run_never_carries_an_assumption(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A stopped run's checkpoint holds an assumption, and a failed run's error is its
    # own text: neither is a result, so the strip says no assumption (FM-154).
    page = _page(fake_microscope, run_doc, [_ep(SavesAssuming)])
    try:
        page.run("saves-assuming", [])
        assert Slow.started.wait(5)
        page.stop()
        assert page.join(10.0)
        assert page.status().outcome == "saves-assuming stopped"
        _run_and_join(page, "failing")
        outcome = page.status().outcome
        assert outcome is not None
        assert "assumed" not in outcome, outcome
    finally:
        page.close()


# --- Refresh forgets the outcome (m5 §8.8.3) ----------------------------------------------


def test_clear_outcome_forgets_the_last_outcome_when_nothing_runs(
    page: PlatePage,
) -> None:
    _run_and_join(page, "stub")
    assert page.status().outcome is not None
    page.clear_outcome()
    assert page.status().outcome is None
    # A second call has nothing to forget and raises nothing.
    page.clear_outcome()
    assert page.status().outcome is None


def test_clear_outcome_leaves_the_stop_note_alone(page: PlatePage) -> None:
    page.run("slow", [])
    assert Slow.started.wait(5)
    page.stop()
    assert page.join(10.0)
    status = page.status()
    assert status.outcome == "slow stopped"
    assert status.stop_note is not None
    page.clear_outcome()
    after = page.status()
    assert after.outcome is None
    assert after.stop_note == status.stop_note


def test_clear_outcome_does_nothing_while_an_activity_runs(page: PlatePage) -> None:
    page.run("slow", [])
    try:
        assert Slow.started.wait(5)
        # A text the running activity's page holds: it is not for a button to drop.
        page._set_outcome("kept")
        page.clear_outcome()
        assert page.status().outcome == "kept"
        assert page.status().running == "slow"
    finally:
        Slow.release.set()
    assert page.join(10.0)
    # The run's own end replaces the text, as it always does.
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("slow done in ")


def test_a_second_run_while_one_runs_is_refused(page: PlatePage) -> None:
    page.run("slow", [])
    assert Slow.started.wait(5)
    assert page.status().running == "slow"
    with _refuses(RuntimeError, "slow is running"):
        page.run("stub", [])
    Slow.release.set()
    assert page.join(10.0)
    assert page.status().running is None
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("slow done in ")


def test_stop_sets_the_event_then_stops_the_stand(
    page: PlatePage, fake_microscope: Microscope, run_doc: Path
) -> None:
    original = fake_microscope.stop
    saw: list[bool] = []

    def spy() -> None:
        # What the stop saw when it was called: the run must already have been told
        # to stop and have ended, or the order is the wrong way round (FM-94: the
        # test asserts after, not here).
        saw.append(Slow.ended.wait(timeout=5))
        original()

    page.run("slow", [])
    assert Slow.started.wait(5)
    with _replaced(fake_microscope, "stop", spy):
        page.stop()
    assert page.join(10.0)
    assert saw == [True]
    assert RunDocument.load(run_doc).stages["slow"].status is StageStatus.stopped
    status = page.status()
    assert status.stop_note == (
        "slow told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert status.halted
    assert status.outcome == "slow stopped"
    assert status.running is None


@contextmanager
def _refusing_stand(
    refused: tuple[str, ...], roles: tuple[Role, ...]
) -> Iterator[Microscope]:
    """A fake demo stand whose ``refused`` stages refuse their stop, as the Ti2-E's do."""
    core = FakeCore.demo_like()
    core.stop_refused = set(refused)
    demo = Profile.demo()
    declared = demo.model_copy(
        update={"safety": demo.safety.model_copy(update={"stop_refused_by": roles})}
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", core), declared)
    try:
        yield microscope
    finally:
        microscope.close()


def test_stop_says_what_the_stand_did_when_it_refuses_the_stop(
    run_doc: Path,
) -> None:
    with _refusing_stand(("XY",), (Role.xy_stage,)) as microscope:
        # The facade's own words for the refusal, taken once so the test does not
        # restate them; the stand is then resumed.
        with pytest.raises(StopRefusedError) as refusal:
            microscope.stop()
        said = " ".join(str(refusal.value).split())
        microscope.resume()
        page = _page(microscope, run_doc)
        page.stop()
        status = page.status()
    assert "refused the stop" in said
    assert status.stop_note == f"halted, but {said}"
    assert not (status.stop_note or "").startswith("stopped")
    assert status.halted
    assert status.stop_refused == ("xy_stage",)


def test_stop_with_nothing_running_still_halts_the_stand(page: PlatePage) -> None:
    page.stop()
    status = page.status()
    assert status.halted
    assert status.stop_note == "stopped: the stand is halted (Resume clears the halt)"


def test_a_stop_that_fails_is_a_note_and_not_an_error(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    def failing() -> None:
        raise HardwareError("the emergency stop failed for xy_stage: cable\nunplugged")

    with _replaced(fake_microscope, "stop", failing):
        page.stop()
    assert page.status().stop_note == (
        "the stop failed: the emergency stop failed for xy_stage: cable unplugged"
    )


def test_a_stop_that_is_a_bug_is_not_swallowed(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    def broken() -> None:
        raise KeyError("a bug")

    with _replaced(fake_microscope, "stop", broken), pytest.raises(KeyError):
        page.stop()


def test_resume_is_refused_while_a_run_is_going_and_clears_the_halt_after(
    page: PlatePage,
) -> None:
    page.run("slow", [])
    assert Slow.started.wait(5)
    with _refuses(RuntimeError, "slow is running"):
        page.resume()
    page.stop()
    assert page.join(10.0)
    assert page.status().halted
    page.resume()
    status = page.status()
    assert not status.halted
    assert status.stop_note is None


def test_a_failed_run_keeps_no_exception(page: PlatePage) -> None:
    gc.disable()
    try:
        page.run("failing", [])
        assert page.join(10.0)
        assert len(Failing.raised) == 1
        # Dead without the cyclic collector: nothing the page holds reaches it (FM-98).
        assert Failing.raised[0]() is None
        assert page.status().outcome == (
            "failing: PluginRunError: failing: line one line two"
        )
    finally:
        gc.enable()


class PickyParams(PluginParams):
    n: int = 1

    @field_validator("n")
    @classmethod
    def _known(cls, value: int) -> int:
        # Raises a KeyError, not a ValueError: pydantic does not turn it into a
        # validation error, so it reaches whoever called `parse_params`.
        return {1: 1}[value]


class Picky:
    """A third party's plugin whose validator raises something other than a ValueError (FM-113)."""

    name = provides = "picky"
    summary = "A test plugin with a validator that raises a KeyError."
    Params = PickyParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: PickyParams, ctx: RunContext) -> StubResult:
        return StubResult()


def test_a_validator_that_raises_something_else_is_an_outcome_and_starts_no_thread(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(Picky)])
    threads = threading.active_count()
    # `n=3` is an int, so the type check passes and the plugin's own validator runs.
    page.run("picky", ["n=3"])
    assert page.status().outcome == "picky: KeyError: 3"
    assert page.status().running is None
    assert threading.active_count() == threads
    assert not run_doc.exists()
    # The same plugin with a value it accepts still runs.
    _run_and_join(page, "picky", ["n=1"])
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("picky done in ")


def test_activities_ended_counts_runs_and_snaps_and_not_refusals(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    assert page.activities_ended == 0
    _run_and_join(page, "stub")
    assert page.activities_ended == 1
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.snap()
    assert page.join(10.0)
    assert page.activities_ended == 2
    # A refusal before a thread starts is no activity.
    page.run("nope", [])
    page.run("stub", ["n=abc"])
    assert page.activities_ended == 2
    # A run that fails ends too, and a second look at the count changes nothing.
    _run_and_join(page, "failing")
    assert page.activities_ended == 3
    assert page.activities_ended == 3


def test_a_run_record_that_could_not_be_written_still_says_so(
    page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = runner_mod.write_stage

    def refuses_the_done_record(path: Path, record: StageRecord) -> RunDocument:
        if record.status is StageStatus.done:
            raise RunDocumentError(
                path, "could not be written (disk full); it is as it was", record=record
            )
        return real_write(path, record)

    monkeypatch.setattr(runner_mod, "write_stage", refuses_the_done_record)
    _run_and_join(page, "stub")
    assert page.status().outcome == (
        f"stub ran, but its record could not be written: {run_doc}: could not be "
        f"written (disk full); it is as it was"
    )


def test_a_closed_page_takes_no_run_and_close_is_harmless_twice(
    page: PlatePage,
) -> None:
    page.close()
    page.close()
    with _refuses(RuntimeError, "the page is closed"):
        page.run("stub", [])
    with pytest.raises(RuntimeError):
        page.goto("A1")


def test_close_stops_a_run_and_joins_it(page: PlatePage) -> None:
    page.run("slow", [])
    assert Slow.started.wait(5)
    page.close()
    assert page.join(0.0)
    status = page.status()
    assert status.running is None
    assert status.outcome == "slow stopped"
    assert status.stop_note == (
        "slow told to stop; stopped: the stand is halted (Resume clears the halt)"
    )


def test_close_when_idle_does_not_touch_the_stand(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    calls: list[str] = []
    with _replaced(fake_microscope, "stop", lambda: calls.append("stop")):
        page.close()
    assert calls == []
    assert not page.status().halted


def test_close_says_when_a_run_does_not_end(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc)
    page.run("slow", [])
    assert Slow.started.wait(5)
    # A worker that ignores the stop: the page stops waiting for it.
    with _replaced(page, "join", lambda timeout_s=None: False):
        page.close()
    assert page.status().outcome == "slow did not end within 10 s"
    Slow.release.set()
    assert page.join(10.0)


def test_join_says_that_a_worker_is_still_alive(page: PlatePage) -> None:
    page.run("slow", [])
    assert Slow.started.wait(5)
    assert not page.join(0.0)
    Slow.release.set()
    assert page.join(10.0)


def test_join_waits_in_short_slices(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows one long wait on a thread is not interrupted by Ctrl-C (FM-83).
    # The deadline runs on a clock of the test's own: on the macOS runner two joins of
    # 0.1 s used up 0.35 s of wall clock and left two slices, not three (#255, FM-44).
    page.run("slow", [])
    assert Slow.started.wait(5)
    worker = page._worker
    assert worker is not None
    now = [0.0]
    reads = [0]
    waits: list[float | None] = []

    def clock() -> float:
        # Only a wait moves it, so a loop that spins on the clock without waiting
        # fails here instead of hanging.
        reads[0] += 1
        if reads[0] > 200:
            raise AssertionError("join read the clock 200 times without waiting")
        return now[0]

    def spy(timeout: float | None = None) -> None:
        # The worker stays alive; a slice takes exactly as long as it was given. A wait
        # with no limit, or one wait too many, fails here instead of hanging.
        waits.append(timeout)
        if timeout is None or len(waits) > 50:
            raise AssertionError(f"join not sliced against the deadline: {waits[-5:]}")
        now[0] += timeout

    # The name, not `time.monotonic`: the worker reads the real clock when it ends.
    with monkeypatch.context() as patch:
        patch.setattr(controller_module, "time", SimpleNamespace(monotonic=clock))
        with _replaced(worker, "join", spy):
            assert not page.join(0.35)
    assert waits == pytest.approx([0.1, 0.1, 0.1, 0.05])


_ACCEPTED = "stopped: the stand is halted (Resume clears the halt)"


def test_close_says_was_stopped_when_the_stand_took_the_stop_and_the_run_ended(
    page: PlatePage,
) -> None:
    page.run("slow", [])
    assert Slow.started.wait(5)
    page.close()
    assert page.close_lines == ("slow was stopped",)
    assert page.close_ended


def test_close_says_what_a_refused_stop_did_and_never_that_the_run_was_stopped(
    run_doc: Path,
) -> None:
    with _refusing_stand(("XY",), (Role.xy_stage,)) as microscope:
        # The facade's own words for the refusal, taken once; the stand is resumed.
        with pytest.raises(StopRefusedError) as refusal:
            microscope.stop()
        said = " ".join(str(refusal.value).split())
        microscope.resume()
        page = _page(microscope, run_doc)
        page.run("slow", [])
        assert Slow.started.wait(5)
        page.close()
    assert page.close_lines == (
        f"slow told to stop; halted, but {said}",
        "slow stopped",
    )
    assert page.close_ended


def test_close_says_how_a_run_that_failed_while_it_was_stopped_ended(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(FailsOnStop)])
    page.run("fails", [])
    assert Slow.started.wait(5)
    page.close()
    assert page.close_lines == (
        f"fails told to stop; {_ACCEPTED}",
        "fails: PluginRunError: fails: the move was stopped",
    )
    assert page.close_ended


def test_close_does_not_say_stopped_for_a_run_that_ended_done(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(Stubborn)])
    page.run("stubborn", [])
    assert Stubborn.started.wait(5)
    timer = threading.Timer(0.3, Stubborn.release.set)
    timer.start()
    try:
        page.close()
    finally:
        timer.cancel()
        Stubborn.release.set()
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith("stubborn done in ")
    assert page.close_lines == (f"stubborn told to stop; {_ACCEPTED}", outcome)
    assert page.close_ended


def test_close_says_a_run_that_does_not_end_and_a_second_close_does_not_wait_again(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_CLOSE_WAIT_S", 1.0)
    page = _page(fake_microscope, run_doc, [_ep(Stubborn)])
    page.run("stubborn", [])
    assert Stubborn.started.wait(5)
    stops: list[str] = []
    real_stop = page.stop

    def counting_stop() -> None:
        stops.append("stop")
        real_stop()

    with _replaced(page, "stop", counting_stop):
        page.close()
        page.close()  # the command's `finally` after `serve` already closed it
    Stubborn.release.set()
    assert page.join(10.0)
    assert stops == ["stop"]
    assert not page.close_ended
    assert page.close_lines == (
        f"stubborn told to stop; {_ACCEPTED}",
        "stubborn did not end within 1 s",
    )


@pytest.mark.parametrize(
    "read_between", [True, False], ids=["status-between", "no-status-between"]
)
def test_a_new_run_starts_with_no_outcome_and_no_progress(
    page: PlatePage, read_between: bool
) -> None:
    _run_and_join(page, "progressing")
    if read_between:
        assert page.status().progress == Progress("progressing", 1, 3, "one of three")
    page.run("slow", [])
    assert Slow.started.wait(5)
    status = page.status()
    assert (status.outcome, status.progress) == (None, None)


# --- records of another stand, or of a dry run ----------------------------------------


@pytest.mark.parametrize(
    ("profile", "dry_run", "why"),
    [
        pytest.param(
            "nikon-ti2",
            False,
            "plate-registration was run on nikon-ti2, this is demo; run "
            "plate-registration again on this profile",
            id="another-stand",
        ),
        pytest.param(
            "demo",
            True,
            "plate-registration was a dry run; run it without --dry-run",
            id="dry-run",
        ),
    ],
)
def test_a_plate_of_another_stand_or_a_dry_run_is_neither_drawn_nor_driven_to(
    fake_microscope: Microscope,
    run_doc: Path,
    profile: str,
    dry_run: bool,
    why: str,
) -> None:
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        profile=profile,
        dry_run=dry_run,
    )
    page = _page(fake_microscope, run_doc)
    view = page.view()
    assert view.nominal
    assert {i.slot: (i.state, i.reason) for i in view.checklist}["plate"] == (
        "unusable",
        why,
    )
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    page.goto("C5")
    assert page.status().outcome == (
        f"goto-well: UpstreamError: goto-well reads plate, but {why}"
    )
    assert StubGoto.ran == []
    assert stage.position_um() == before
    assert page.status().running is None


@pytest.mark.parametrize(
    "recorded_dry", [False, True], ids=["real-record", "dry-record"]
)
def test_a_dry_run_page_reads_the_plate_a_real_or_a_dry_run_registered(
    run_doc: Path, fake_core: FakeCore, recorded_dry: bool
) -> None:
    # A dry run may read a real result, and a dry run's own: only a real run refuses
    # a dry run's.
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        _seed(run_doc, "plate-registration", "plate", _plate(), dry_run=recorded_dry)
        page = _page(dry, run_doc)
        assert not page.view().nominal
        page.goto("C5")
        assert page.join(10.0)
        assert StubGoto.ran == [("C5", "centre")]
    finally:
        dry.close()


# --- status ------------------------------------------------------------------------


def test_status_names_the_stand(run_doc: Path) -> None:
    core = FakeCore.demo_like()
    demo = Profile.demo()
    named = demo.model_copy(
        update={"microscope": demo.microscope.model_copy(update={"name": "ti2e-bench"})}
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", core), named)
    try:
        assert _page(microscope, run_doc).status().stand == "ti2e-bench"
    finally:
        microscope.close()


def test_status_reads_the_stand(page: PlatePage, fake_microscope: Microscope) -> None:
    fake_microscope.require(XYStage).move_to_um(1000.0, -500.0)
    fake_microscope.require(ZStage).move_to_um(12.5)
    status = page.status()
    assert status.xy_um == (1000.0, -500.0)
    assert status.z_um == 12.5
    assert status.objective == "Nikon 10X S Fluor"
    assert status.autofocus == "autofocus off"
    assert status.light_on is None
    assert status.moving == ()
    assert not status.halted
    assert not status.dry_run
    assert status.running is None
    assert status.progress is None
    assert status.outcome is None
    assert status.stop_note is None
    assert status.stop_refused == ()
    assert status.problems == ()


def test_status_says_no_autofocus_without_the_capability(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    with _replaced(
        fake_microscope, "has", lambda capability: capability is not Autofocus
    ):
        assert page.status().autofocus == "no autofocus"


def test_a_light_that_cannot_be_read_is_a_problem_and_not_a_crash(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    class Broken:
        def is_on(self) -> bool | None:
            raise RuntimeError("the lamp\ndoes not answer")

    with _replaced(fake_microscope, "get", lambda capability: Broken()):
        status = page.status()
    assert status.light_on is None
    assert status.problems == ("light: RuntimeError: the lamp does not answer",)


def test_status_reads_progress_once_for_every_session(page: PlatePage) -> None:
    _run_and_join(page, "progressing")
    first = page.status().progress
    assert first == Progress("progressing", 1, 3, "one of three")
    assert page.status().progress == first


# --- captures and what the page fills --------------------------------------------------


def test_captured_references_reach_plate_registration(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, _real("plate-registration"))
    page.choose_plate("96-well")
    stage = fake_microscope.require(XYStage)
    for well in ("A1", "A12"):
        stage.move_to_um(*_centre(well, a1=A1_OFF, rotation_deg=TURN_DEG))
        page.capture_centre(well)
    _run_and_join(page, "plate-registration", page.page_params("plate-registration"))
    record = RunDocument.load(run_doc).stages["plate-registration"]
    assert record.status is StageStatus.done
    assert record.result is not None
    assert record.result["a1_center_xy_um"] == pytest.approx(A1_OFF, abs=1e-6)
    assert record.result["rotation_deg"] == pytest.approx(TURN_DEG, abs=1e-6)
    assert record.params["plate"] == "96-well"
    assert record.params["refs"] == [r.model_dump(mode="json") for r in page.refs]
    assert [r["well"] for r in record.params["refs"]] == ["A1", "A12"]


def test_a_capture_replaces_the_same_well_and_side(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)  # the Ti2-E's: left is +X, top is -Y
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(10.0, 20.0)
    page.capture_centre("A1")
    stage.move_to_um(30.0, 40.0)
    page.capture_centre("b1")
    stage.move_to_um(50.0, 60.0)
    page.capture_centre("A1")
    assert page.refs == (
        WellRef(well="B1", x_um=30.0, y_um=40.0),
        WellRef(well="A1", x_um=50.0, y_um=60.0),
    )
    stage.move_to_um(1.0, 2.0)
    page.capture_wall("A1", "left")
    page.capture_wall("A1", "top")
    stage.move_to_um(3.0, 4.0)
    assert page.capture_wall("A1", "left") == WallTouch(
        well="A1", side="+X", x_um=3.0, y_um=4.0
    )
    assert page.walls == (
        WallTouch(well="A1", side="-Y", x_um=1.0, y_um=2.0),
        WallTouch(well="A1", side="+X", x_um=3.0, y_um=4.0),
    )


def test_a_capture_of_a_well_that_is_not_on_the_plate_is_refused(
    page: PlatePage,
) -> None:
    with _refuses(ValueError, "Z9 is not a well of the 96-well plate"):
        page.capture_centre("Z9")
    page.choose_plate("6-well")
    with _refuses(ValueError, "A12 is not a well of the 6-well plate"):
        page.capture_wall("A12", "left")
    with _refuses(
        ValueError, "side must be one of left, right, top, bottom, got 'centre'"
    ):
        page.capture_wall("A1", "centre")
    assert page.refs == ()
    assert page.walls == ()


# --- walls named on the image (m5 §13, #327) -------------------------------------------

#: The Ti2-E at 10x: the image's left is the stage's +X and its bottom the stage's +Y.
TI2E: tuple[tuple[float, float], tuple[float, float]] = ((-1.1, 0.0), (0.0, 1.1))
#: An upright camera, not mirrored: the image's sides are the stage's.
UPRIGHT: tuple[tuple[float, float], tuple[float, float]] = ((1.1, 0.0), (0.0, -1.1))
#: A camera at 90 degrees, mirrored: the one matrix of the table that is neither
#: diagonal nor symmetric (M3 §19.1: left +Y, right -Y, top -X, bottom +X). The two
#: above equal their own transpose, so a page that handed the matrix over transposed
#: passed every test (FM-73: the symmetric case hides the exchange of two axes);
#: this one's transpose is the -90 degree mirrored camera, opposite on all four sides.
NINETY_MIRRORED: tuple[tuple[float, float], tuple[float, float]] = (
    (0.0, 1.1),
    (-1.1, 0.0),
)


def _turned(deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """``1.1 * R(deg) * diag(1, -1)``: an upright camera turned by ``deg``."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((1.1 * c, 1.1 * s), (1.1 * s, -1.1 * c))


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (TI2E, {"left": "+X", "right": "-X", "top": "-Y", "bottom": "+Y"}),
        (UPRIGHT, {"left": "-X", "right": "+X", "top": "+Y", "bottom": "-Y"}),
        (
            NINETY_MIRRORED,
            {"left": "+Y", "right": "-Y", "top": "-X", "bottom": "+X"},
        ),
    ],
    ids=["ti2e", "upright", "ninety-mirrored"],
)
def test_capture_wall_records_the_stage_side_of_the_image_side(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
    matrix: tuple[tuple[float, float], tuple[float, float]],
    expected: dict[str, str],
) -> None:
    seed_camera_calibration(run_doc, matrix)
    stage = fake_microscope.require(XYStage)
    for index, image_side in enumerate(("left", "right", "top", "bottom")):
        x, y = 100.0 * (index + 1), -50.0 * (index + 1)
        stage.move_to_um(x, y)
        touch = page.capture_wall("B3", image_side)
        # The side is the stage's, the position the stage's, and the well normalised.
        assert touch == WallTouch(
            well="B3",
            side=expected[image_side],
            x_um=x,
            y_um=y,  # type: ignore[arg-type]
        )
    assert [w.side for w in page.walls] == [
        expected[s] for s in ("left", "right", "top", "bottom")
    ]


def test_capture_wall_without_a_calibration_is_refused_and_records_nothing(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    stage = fake_microscope.require(XYStage)
    reads: list[None] = []
    real_position = stage.position_um

    def counting() -> Any:
        reads.append(None)
        return real_position()

    with (
        _replaced(stage, "position_um", counting),
        _refuses(ValueError, NO_CALIBRATION_FOR_SIDES),
    ):
        page.capture_wall("B3", "left")
    assert page.walls == ()
    # A wall that cannot be mapped is refused before the stage is read.
    assert reads == []


def test_capture_wall_with_a_calibration_off_a_quarter_turn_is_refused(
    page: PlatePage, run_doc: Path, seed_camera_calibration: Callable[..., Any]
) -> None:
    seed_camera_calibration(run_doc, _turned(6.0))
    with pytest.raises(ValueError, match=r"the camera is turned 6\.0° away") as raised:
        page.capture_wall("B3", "left")
    # `frames`' own text goes through, taken where it is caught (FM-98).
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert page.walls == ()
    seed_camera_calibration(run_doc, _turned(4.0))  # within the bound: a side again
    assert page.capture_wall("B3", "left").side == "-X"


def test_a_capture_of_the_same_stage_side_replaces_the_first(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    stage = fake_microscope.require(XYStage)
    seed_camera_calibration(run_doc, TI2E)
    stage.move_to_um(1.0, 2.0)
    page.capture_wall("B3", "left")  # stage +X
    # The camera was turned and calibrated again: the image's right is +X now, so a
    # capture of it is the same stage side, and replaces the first (m5 §13.1).
    seed_camera_calibration(run_doc, UPRIGHT)
    stage.move_to_um(3.0, 4.0)
    page.capture_wall("B3", "right")
    assert page.walls == (WallTouch(well="B3", side="+X", x_um=3.0, y_um=4.0),)


def test_goto_an_image_side_runs_goto_well_with_the_stage_side(
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    try:
        seed_camera_calibration(run_doc, TI2E)
        page.goto("B3", "left")
        assert page.join(10.0)
        seed_camera_calibration(run_doc, UPRIGHT)
        page.goto("B3", "left")
        assert page.join(10.0)
        page.goto("B3", "centre")  # no mapping: it needs no calibration to be read
        assert page.join(10.0)
        assert StubGoto.ran == [("B3", "+X"), ("B3", "-X"), ("B3", "centre")]
    finally:
        page.close()


def test_goto_an_image_side_without_a_calibration_is_refused_before_current_changes(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    try:
        with _refuses(ValueError, NO_CALIBRATION_FOR_SIDES):
            page.goto("B3", "left")
        x, y = _centre("B3", a1=A1_OFF, rotation_deg=TURN_DEG)
        with _refuses(ValueError, NO_CALIBRATION_FOR_SIDES):
            page.click(x, y, drive=True, side="bottom")
        # Before `current` changes and before a run starts: the page shows no well
        # that was not driven to.
        assert page.current is None
        assert StubGoto.ran == []
        assert page.status().running is None
        assert page.status().outcome is None
    finally:
        page.close()


def test_goto_refuses_a_word_that_is_no_image_side(
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # A stage direction is not a side of the image: it would skip the mapping.
    _seed(run_doc, "plate-registration", "plate", _plate())
    seed_camera_calibration(run_doc)
    page = _page(fake_microscope, run_doc)
    try:
        for word in ("+X", "middle"):
            with _refuses(
                ValueError,
                f"side must be one of centre, left, right, top, bottom, got {word!r}",
            ):
                page.goto("B3", word)
        assert StubGoto.ran == []
        assert page.current is None
    finally:
        page.close()


def test_view_carries_the_matrix_of_the_latest_usable_calibration(
    page: PlatePage, run_doc: Path, seed_camera_calibration: Callable[..., Any]
) -> None:
    assert page.view().camera_matrix is None
    seed_camera_calibration(run_doc, TI2E)
    assert page.view().camera_matrix == TI2E
    # Rotation and mirror belong to the optical path: another objective's is used.
    seed_camera_calibration(run_doc, UPRIGHT, objective="Nikon 40X Plan Fluor ELWD")
    assert page.view().camera_matrix == UPRIGHT
    # A dry run's record is not usable on a real stand, nor another stand's.
    seed_camera_calibration(run_doc, TI2E, dry_run=True)
    assert page.view().camera_matrix is None
    seed_camera_calibration(run_doc, TI2E, profile="another-stand")
    assert page.view().camera_matrix is None


def test_a_calibration_whose_result_does_not_validate_is_a_problem_not_a_crash(
    page: PlatePage, run_doc: Path
) -> None:
    # The record is `done` and usable, but its result is not a calibration (FM-125):
    # the view draws it as absent and says why; a capture is refused as without one.
    _seed(run_doc, "camera-calibration", "camera-calibration", StubResult())
    view = page.view()
    assert view.camera_matrix is None
    assert any(
        p.startswith("camera-calibration: result does not validate")
        for p in view.problems
    )
    with _refuses(ValueError, NO_CALIBRATION_FOR_SIDES):
        page.capture_wall("B3", "left")


def test_a_run_document_that_names_walls_on_the_image_starts_the_page_with_none(
    fake_microscope: Microscope, run_doc: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Ti2-E session 2's document holds `left` and `bottom` in its params (smc 0.0.1).
    # They are not translated, because a translation is a guess of the frame (M3
    # §19.2): the page shows no capture and says so in its log.
    walls = [{"well": "B3", "side": "left", "x_um": 5.0, "y_um": 6.0}]
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": [], "walls": walls},
    )
    with caplog.at_level(logging.INFO, logger="smc.ui"):
        page = _page(fake_microscope, run_doc)
    try:
        assert page.walls == ()
        lines = [r.getMessage() for r in caplog.records if r.name == "smc.ui"]
        assert any("walls are not read" in line for line in lines)
    finally:
        page.close()


def test_an_unreadable_run_document_is_the_reason_and_not_a_missing_calibration(
    page: PlatePage, run_doc: Path
) -> None:
    # `run camera-calibration` is advice that cannot work here: the run document is
    # what cannot be read, and a write would refuse it too. The page says so, in the
    # words `view().problems` already has, and records nothing.
    run_doc.write_text("{not json", encoding="utf-8")
    reason = r"^run document: .*is not valid JSON"
    with pytest.raises(ValueError, match=reason) as captured:
        page.capture_wall("B3", "left")
    with pytest.raises(ValueError, match=reason) as driven:
        page.goto("B3", "left")
    for raised in (captured, driven):
        assert NO_CALIBRATION_FOR_SIDES not in str(raised.value)
        assert raised.value.__context__ is None  # taken where caught (FM-98)
    assert page.walls == ()
    assert StubGoto.ran == []
    assert page.current is None


def test_a_drive_action_that_starts_while_the_side_is_mapped_refuses_the_capture(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # The stage-at-rest guard is not atomic with the read (its docstring says the
    # window is one `position_um()`). The mapping reads the run document in between,
    # so a jog that another tab starts during it must meet a second guard: without it
    # the wall is recorded where the stage was before the jog moved it.
    seed_camera_calibration(run_doc)
    stage = fake_microscope.require(XYStage)
    reads: list[None] = []
    real_position = stage.position_um
    real_map = page._stage_side

    def counting() -> Any:
        reads.append(None)
        return real_position()

    def maps_while_a_jog_starts(side: Any) -> Any:
        mapped = real_map(side)
        with page._lock:
            page._drive_running = "stage jog"
        return mapped

    try:
        with (
            _replaced(page, "_stage_side", maps_while_a_jog_starts),
            _replaced(stage, "position_um", counting),
            _refuses(RuntimeError, "stage jog is running"),
        ):
            page.capture_wall("B3", "left")
    finally:
        with page._lock:
            page._drive_running = None
    assert page.walls == ()
    assert reads == []


@pytest.mark.parametrize("record", ["dry-run", "another-stand", "failed"])
def test_an_unusable_calibration_is_no_calibration_at_the_press(
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
    record: str,
) -> None:
    # The press is the guard (FM-169), and it reads the document itself: a dry run's
    # synthetic matrix on a real stand, another stand's, or the matrix of a run that
    # failed since, would map walls the stage then drives to. `view()` was tested for
    # it; this is the press.
    _seed(run_doc, "plate-registration", "plate", _plate())
    if record == "failed":
        _seed(
            run_doc,
            "camera-calibration",
            "camera-calibration",
            status=StageStatus.failed,
        )
    else:
        seed_camera_calibration(
            run_doc,
            TI2E,
            dry_run=record == "dry-run",
            profile="another-stand" if record == "another-stand" else "demo",
        )
    page = _page(fake_microscope, run_doc)
    try:
        with _refuses(ValueError, NO_CALIBRATION_FOR_SIDES):
            page.capture_wall("B3", "left")
        with _refuses(ValueError, NO_CALIBRATION_FOR_SIDES):
            page.goto("B3", "left")
        assert page.walls == ()
        assert StubGoto.ran == []
        assert page.current is None
    finally:
        page.close()


def test_a_capture_of_the_same_stage_side_on_another_well_keeps_the_first(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # The replace rule is the same well *and* the same stage side: the operator
    # captures B3's +X wall and then B4's, and the page must keep both.
    seed_camera_calibration(run_doc)
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(1.0, 2.0)
    page.capture_wall("A1", "left")
    stage.move_to_um(3.0, 4.0)
    page.capture_wall("A2", "left")
    assert page.walls == (
        WallTouch(well="A1", side="+X", x_um=1.0, y_um=2.0),
        WallTouch(well="A2", side="+X", x_um=3.0, y_um=4.0),
    )


def test_goto_an_image_side_on_another_plates_map_says_the_plate_not_the_calibration(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # Two refusals could apply and neither is a courtesy: the plate mismatch comes
    # first (m5 §13.1), because with it the same-named well of another plate is the
    # target, which no calibration fixes.
    _seed(run_doc, "plate-registration", "plate", _plate("384-well"))
    page = _page(fake_microscope, run_doc)
    try:
        page.choose_plate("96-well")
        with _refuses(
            ValueError,
            "the registered plate is a 384-well; calibrate the 96-well plate again, "
            "or choose 384-well",
        ):
            page.goto("C5", "left")
        assert StubGoto.ran == []
        assert page.current is None
    finally:
        page.close()


def test_forget_drops_one_wells_captures_or_all(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(1.0, 2.0)
    for well in ("A1", "A2"):
        page.capture_centre(well)
        page.capture_wall(well, "left")
    page.forget("a1")
    assert [r.well for r in page.refs] == ["A2"]
    assert [w.well for w in page.walls] == ["A2"]
    page.forget()
    assert page.refs == ()
    assert page.walls == ()


def test_references_and_selection_start_from_the_latest_records(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    refs = [
        {"well": "A1", "x_um": 1000.0, "y_um": -500.0},
        {"well": "A12", "x_um": 100000.0, "y_um": 4000.0},
    ]
    walls = [{"well": "B2", "side": "-X", "x_um": 5.0, "y_um": 6.0}]
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate("384-well"),
        params={"plate": "384-well", "refs": refs, "walls": walls},
    )
    _seed(
        run_doc,
        "scan-plan",
        "scan-plan",
        status=StageStatus.failed,
        params={"wells": "B1,A1"},
        minute=1,
    )
    page = _page(fake_microscope, run_doc)
    assert page.plate == "384-well"
    assert page.selection == ("B1", "A1")
    assert page.refs == tuple(WellRef.model_validate(r) for r in refs)
    assert page.walls == (WallTouch.model_validate(walls[0]),)


def test_records_that_do_not_parse_start_the_page_empty_and_say_so(
    fake_microscope: Microscope, run_doc: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        status=StageStatus.failed,
        params={"plate": "96-well", "refs": [{"well": "A1"}], "walls": "nonsense"},
    )
    _seed(run_doc, "scan-plan", "scan-plan", params={"wells": "Z99"}, minute=1)
    with caplog.at_level(logging.INFO, logger="smc.ui"):
        page = _page(fake_microscope, run_doc)
    assert (page.selection, page.refs, page.walls) == ((), (), ())
    lines = [r.getMessage() for r in caplog.records if r.name == "smc.ui"]
    assert len(lines) == 3
    assert all("not read" in line for line in lines)


def test_choose_plate_clears_the_selection_and_the_captures(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)
    page.select_text("A1:A3")
    page.capture_centre("A1")
    page.capture_wall("A1", "top")
    page.click(*_centre("A1"), drive=False)
    page.choose_plate("96-well")  # the same plate: nothing is cleared
    assert page.selection == ("A2", "A3")
    assert page.refs != ()
    page.choose_plate("384-well")
    assert page.plate == "384-well"
    assert (page.selection, page.refs, page.walls, page.current) == ((), (), (), None)
    plates = ", ".join(sorted(useq.registered_well_plate_keys()))
    with _refuses(ValueError, f"'nope' is not a plate; the plates are {plates}"):
        page.choose_plate("nope")
    assert page.plate == "384-well"


def test_the_lab_plates_can_be_chosen(page: PlatePage) -> None:
    """A plate of ``src/smc/plates.toml`` is a ``useq`` key like any other (m3 §22)."""
    page.choose_plate("greiner-24-well")
    assert page.plate == "greiner-24-well"


def test_scan_plan_params_are_the_selection_in_order(page: PlatePage) -> None:
    page.select_text("B3,A1")
    assert page.page_params("scan-plan") == ["wells=B3,A1"]
    assert page.page_params("stub") == []


def test_scan_plan_needs_a_selection(page: PlatePage) -> None:
    with _refuses(ValueError, "select at least one well first"):
        page.page_params("scan-plan")


def test_wells_chosen_on_another_plate_than_the_registered_one_are_refused(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate("96-well"))
    page = _page(fake_microscope, run_doc)
    page.choose_plate("384-well")
    page.select_text("A1")
    refusal = (
        "the registered plate is a 96-well; calibrate the 384-well plate again, "
        "or choose 96-well"
    )
    with _refuses(ValueError, refusal):
        page.page_params("scan-plan")


def test_select_text_and_toggle_name_what_is_not_on_the_plate_and_keep_the_selection(
    page: PlatePage,
) -> None:
    page.select_text("A1,A2")
    with _refuses(ValueError, "wells not understood on a 96-well plate: 'Z99'"):
        page.select_text("Z99")
    with _refuses(ValueError, "Z99 is not a well of the 96-well plate"):
        page.toggle("Z99")
    assert page.selection == ("A1", "A2")


@pytest.mark.parametrize("blank", ["", "   ", "\t", " \n "])
def test_select_text_with_a_blank_text_clears_the_selection(
    page: PlatePage, blank: str
) -> None:
    # An emptied Wells box means "no wells" (m5 §8.8.5); `parse_wells` itself refuses
    # an empty text, which is right for `smc run scan-plan --param wells=`.
    page.select_text("A1:A3")
    assert page.selection == ("A1", "A2", "A3")
    page.select_text(blank)
    assert page.selection == ()
    # And again: nothing selected stays nothing selected, without a refusal.
    page.select_text(blank)
    assert page.selection == ()


def test_a_text_that_names_no_well_but_is_not_blank_keeps_its_refusal_and_the_selection(
    page: PlatePage,
) -> None:
    # Only a blank text means "no wells": a stray comma is a mistake, and says so.
    page.select_text("A1,A2")
    with _refuses(ValueError, "no wells selected"):
        page.select_text(",")
    assert page.selection == ("A1", "A2")


def test_plate_registration_params_are_the_plate_and_the_captures(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)  # the Ti2-E's: the image's right is stage -X
    page.choose_plate("96-well")
    fake_microscope.require(XYStage).move_to_um(1.5, 2.5)
    page.capture_centre("A1")
    page.capture_wall("A1", "right")
    refs = json.dumps([{"well": "A1", "x_um": 1.5, "y_um": 2.5}])
    # The pairs carry the stage side the page mapped, never the word the operator saw.
    walls = json.dumps([{"well": "A1", "side": "-X", "x_um": 1.5, "y_um": 2.5}])
    assert page.page_params("plate-registration") == [
        "plate=96-well",
        f"refs={refs}",
        f"walls={walls}",
    ]


# --- plate-walls on the page (m5 §12, #346) ------------------------------------------


def test_plate_walls_is_in_set_up_right_after_plate_registration(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _stood_page(
        fake_microscope,
        run_doc,
        "scan-plan",
        "plate-walls",
        "plate-registration",
        "camera-calibration",
    )
    try:
        # The page's own stub tools come along (`_page`): the real ones are those named.
        named = {"camera-calibration", "plate-registration", "plate-walls", "scan-plan"}
        tools = [t for t in page.tools() if t.name in named]
        # The workflow's order, not the entry points' (§12.2): the owner's path sits
        # beside the registration it is an alternative to.
        assert [t.name for t in tools] == [
            "camera-calibration",
            "plate-registration",
            "plate-walls",
            "scan-plan",
        ]
        assert {t.name: t.stage for t in tools}["plate-walls"] == "Set up"
    finally:
        page.close()


def test_the_form_of_plate_walls_offers_neither_plate_nor_refs(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _stood_page(fake_microscope, run_doc, "plate-walls")
    try:
        walls = {t.name: t for t in page.tools()}["plate-walls"]
        # `plate` and `refs` are the page's (its captures); every other field is under
        # *Advanced*, closed: a newcomer sees buttons (§12.2), and KEY_PARAMS has no line.
        assert "plate-walls" not in KEY_PARAMS
        assert walls.key == ()
        assert {row[0] for row in walls.advanced} == {
            "search_um",
            "max_residual_um",
            "max_rotation_deg",
            "settle_ms",
            "save_frames",
        }
        assert PAGE_PARAMS["plate-walls"] == ("plate", "refs")
    finally:
        page.close()


def test_plate_walls_params_are_the_plate_and_the_captured_centres(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)
    page.choose_plate("96-well")
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(1.5, 2.5)
    page.capture_centre("A1")
    stage.move_to_um(9001.5, -9000.5)
    page.capture_centre("B2")
    page.capture_wall("A1", "right")  # a wall touch is registration's, never the tool's
    refs = json.dumps(
        [
            {"well": "A1", "x_um": 1.5, "y_um": 2.5},
            {"well": "B2", "x_um": 9001.5, "y_um": -9000.5},
        ]
    )
    assert page.page_params("plate-walls") == ["plate=96-well", f"refs={refs}"]
    # The one dump for both tools: a step and a click agree, byte for byte.
    registration = page.page_params("plate-registration")
    assert registration[:2] == page.page_params("plate-walls")


def test_plate_walls_params_refuse_fewer_than_two_centres_and_more_than_six(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    page.choose_plate("96-well")
    stage = fake_microscope.require(XYStage)
    with _refuses(ValueError, walls_centres_problem(0) or ""):
        page.page_params("plate-walls")
    for index, well in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
        stage.move_to_um(9000.0 * index, 0.0)
        page.capture_centre(well)
        if index == 1:
            with _refuses(ValueError, walls_centres_problem(1) or ""):
                page.page_params("plate-walls")
        elif index <= 6:
            # 2 to 6 centres: the pairs, whatever they are.
            assert page.page_params("plate-walls")[0] == "plate=96-well"
    with _refuses(ValueError, walls_centres_problem(7) or ""):
        page.page_params("plate-walls")
    # The step's pairs (`_page_pairs`) never raise for the count: a preset's check of
    # the tool's other fields is not refused for it (§12.2).
    assert page._page_pairs("plate-walls")[0] == "plate=96-well"


def test_the_plates_gate_comes_before_the_count_for_plate_walls(
    page: PlatePage,
) -> None:
    # Unchosen and uncaptured: the plate is the first thing the operator is told.
    with _refuses(ValueError, PLACEHOLDER_REFUSAL):
        page.page_params("plate-walls")


def _walls_page(
    microscope: Microscope,
    run_doc: Path,
    *,
    centres: int = 2,
    calibrated: bool = True,
    chosen: bool = True,
    plate: str = "96-well",
) -> PlatePage:
    """A page with the real ``plate-walls`` and ``camera-calibration``, in one state of §12.3's list.

    The calibration is a record the document calls usable (the page reads its state,
    never its matrix); the centres are captured on the stage at distinct places.
    """
    if calibrated:
        _seed(run_doc, "camera-calibration", "camera-calibration")
    page = _stood_page(microscope, run_doc, "plate-walls", "camera-calibration")
    if chosen:
        page.choose_plate(plate)
    stage = microscope.require(XYStage)
    for index, well in enumerate(["B3", "D5", "F7", "H9", "A1", "C2"][:centres]):
        stage.move_to_um(9000.0 * index, -9000.0 * index)
        page.capture_centre(well)
    return page


def _walls_reason(page: PlatePage) -> str | None:
    return page.view().blocked.get("plate-walls")


def test_plate_walls_is_blocked_in_the_order_dry_run_square_wells_calibration_stand_centres(
    fake_microscope: Microscope, fake_core: FakeCore, run_doc: Path
) -> None:
    # §12.3: the stand closed, the placeholder plate, then a dry run, square wells, the
    # calibration, the stand's capabilities, the centres. Each state alone, then two at
    # once: the earlier reason wins, so the operator is told the one that holds longest.
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        # Each reason alone.
        ok = _walls_page(fake_microscope, run_doc)
        assert _walls_reason(ok) is None
        ok.close()
        run_doc.unlink()

        none = _walls_page(fake_microscope, run_doc, centres=0)
        assert _walls_reason(none) == walls_centres_problem(0)
        none.close()
        one = _walls_page(fake_microscope, run_doc, centres=1)
        assert _walls_reason(one) == walls_centres_problem(1)
        one.close()
        seven_page = _walls_page(fake_microscope, run_doc, centres=2)
        stage = fake_microscope.require(XYStage)
        for index, well in enumerate(["A2", "A3", "A4", "A5", "A6"], start=2):
            stage.move_to_um(9000.0 * index, 100.0)
            seven_page.capture_centre(well)
        assert _walls_reason(seven_page) == walls_centres_problem(7)
        seven_page.close()
        run_doc.unlink()

        no_cal = _walls_page(fake_microscope, run_doc, calibrated=False)
        assert _walls_reason(no_cal) == WALLS_NEED_CALIBRATION
        with _replaced(fake_microscope, "has", _without(Camera)):
            # No calibration and no camera: the calibration is asked for first.
            assert _walls_reason(no_cal) == WALLS_NEED_CALIBRATION
        no_cal.close()

        no_camera = _walls_page(fake_microscope, run_doc, centres=0)
        with _replaced(fake_microscope, "has", _without(Camera)):
            # A stand that cannot image is told before the centres it lacks.
            assert _walls_reason(no_camera) == "this stand has no Camera"
        no_camera.close()
        run_doc.unlink()

        # A dry run: the plate's reason first, then the dry run ahead of the rest.
        unchosen = _stood_page(dry, run_doc, "plate-walls", "camera-calibration")
        assert _walls_reason(unchosen) == PLACEHOLDER_REFUSAL
        unchosen.close()
        dry_page = _walls_page(dry, run_doc, centres=0, calibrated=False)
        assert _walls_reason(dry_page) == WALLS_DRY_RUN
        dry_page.close()

        # Square wells (#375): ahead of the calibration, the stand and the centres, and
        # behind the dry run. No calibration, no centre and no camera: the wells' reason.
        square = _walls_page(
            fake_microscope, run_doc, centres=0, calibrated=False, plate="384-well"
        )
        assert _walls_reason(square) == walls_square_wells("384-well")
        with _replaced(fake_microscope, "has", _without(Camera)):
            assert _walls_reason(square) == walls_square_wells("384-well")
        square.close()
        # A round plate is not told it: the same page on a 24-well one has the next reason.
        round_page = _walls_page(
            fake_microscope, run_doc, centres=0, calibrated=False, plate="24-well"
        )
        assert _walls_reason(round_page) == WALLS_NEED_CALIBRATION
        round_page.close()
        dry_square = _walls_page(
            dry, run_doc, centres=0, calibrated=False, plate="1536-well"
        )
        assert _walls_reason(dry_square) == WALLS_DRY_RUN
        dry_square.close()
    finally:
        dry.close()

    # The stand closed is the reason ahead of all of them.
    closed = PlatePage.closed(
        fake_microscope.profile, run_doc, entry_points=_real("plate-walls")
    )
    try:
        assert _walls_reason(closed) == "the stand is not connected"
    finally:
        closed.close()


def test_with_no_calibration_the_reason_says_to_run_camera_calibration(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _walls_page(fake_microscope, run_doc, calibrated=False)
    try:
        # The page's own sentence, not the run document's, which is for a terminal.
        assert _walls_reason(page) == WALLS_NEED_CALIBRATION
        assert "smc run --list" not in WALLS_NEED_CALIBRATION
    finally:
        page.close()


def test_an_unusable_calibration_keeps_the_run_documents_words(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A calibration that exists and is of another stand is not "run camera-calibration
    # first": the document says what is wrong with it, as it does for every tool.
    _seed(run_doc, "camera-calibration", "camera-calibration", profile="lightsheet")
    page = _walls_page(fake_microscope, run_doc, calibrated=False)
    try:
        expected = RunDocument.load_or_new(run_doc).unusable(
            "camera-calibration", profile="demo", dry_run=False
        )
        assert expected is not None
        assert _walls_reason(page) == expected
        assert _walls_reason(page) != WALLS_NEED_CALIBRATION
    finally:
        page.close()


def test_a_dry_run_page_refuses_plate_walls_in_the_controller_and_writes_nothing(
    fake_core: FakeCore, run_doc: Path
) -> None:
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        # A usable plate, as a real registration left it: the plugin's own refusal of a
        # dry run would be a `failed` record that replaces it (§11.4).
        _seed(run_doc, "plate-registration", "plate", _plate())
        _seed(run_doc, "camera-calibration", "camera-calibration", minute=1)
        page = _walls_page(dry, run_doc, centres=2, calibrated=False, chosen=False)
        try:
            before = run_doc.read_bytes()
            # Pairs that the parameters would refuse: the refusal is the dry run's,
            # which comes before anything is read (FM-169).
            page.run("plate-walls", ["bogus=1"])
            assert page.status().running is None
            assert page.status().outcome == f"plate-walls: {WALLS_DRY_RUN}"
            assert run_doc.read_bytes() == before
            provider = RunDocument.load_or_new(run_doc).provider("plate")
            assert provider is not None
            assert provider.plugin == "plate-registration"
            assert provider.status is StageStatus.done
        finally:
            page.close()
    finally:
        dry.close()


@pytest.mark.parametrize("plate", ["384-well", "1536-well"])
def test_a_square_welled_plate_refuses_plate_walls_in_the_controller_and_writes_nothing(
    fake_microscope: Microscope, run_doc: Path, plate: str
) -> None:
    # #375: the plugin fits round wells and refuses a square-welled plate before it
    # moves, and its refusal is a `failed` record that replaces the usable plate that
    # plate-registration left (§11.4). The button's state is a courtesy (FM-169): the
    # controller refuses by itself, before the parameters are read, and writes nothing.
    _seed(run_doc, "plate-registration", "plate", _plate(plate))
    _seed(run_doc, "camera-calibration", "camera-calibration", minute=1)
    page = _walls_page(
        fake_microscope, run_doc, centres=2, calibrated=False, plate=plate
    )
    try:
        before = run_doc.read_bytes()
        # Pairs that the parameters would refuse: the refusal is the plate's.
        page.run("plate-walls", ["bogus=1"])
        assert page.status().running is None
        assert page.status().outcome == f"plate-walls: {walls_square_wells(plate)}"
        assert run_doc.read_bytes() == before
        provider = RunDocument.load_or_new(run_doc).provider("plate")
        assert provider is not None
        assert provider.plugin == "plate-registration"
        assert provider.status is StageStatus.done
        # A round plate is not refused by it: the pairs are read, and `bogus` is the
        # parameters' own refusal (the control that shows the guard is the plate's).
        page.choose_plate("96-well")
        page.run("plate-walls", ["bogus=1"])
        assert "bogus" in (page.status().outcome or "")
        assert walls_square_wells("96-well") not in (page.status().outcome or "")
    finally:
        page.close()


def test_the_dry_run_comes_before_square_wells_when_plate_walls_is_run(
    fake_core: FakeCore, run_doc: Path
) -> None:
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        _seed(run_doc, "plate-registration", "plate", _plate("384-well"))
        page = _walls_page(dry, run_doc, centres=0, calibrated=False, chosen=False)
        try:
            assert page.plate == "384-well"
            page.run("plate-walls", [])
            assert page.status().outcome == f"plate-walls: {WALLS_DRY_RUN}"
        finally:
            page.close()
    finally:
        dry.close()


def test_plate_walls_params_refuse_the_dry_run_then_square_wells_then_the_count(
    fake_microscope: Microscope, fake_core: FakeCore, run_doc: Path
) -> None:
    # The page's click builds the pairs first, so the text a click shows is this one:
    # the dry run's ahead of the count's, as the button's reason is (§12.3, #375).
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        dry_page = _walls_page(dry, run_doc, centres=0, calibrated=False)
        with _refuses(ValueError, WALLS_DRY_RUN):
            dry_page.page_params("plate-walls")
        dry_page.close()
        # The plate's gate is ahead of both: a dry-run page that nothing named a plate
        # for says to choose it, as its button does (adversarial review of #375).
        unchosen = _walls_page(dry, run_doc, centres=0, calibrated=False, chosen=False)
        assert unchosen.plate_chosen is False
        assert _walls_reason(unchosen) == PLACEHOLDER_REFUSAL
        with _refuses(ValueError, PLACEHOLDER_REFUSAL):
            unchosen.page_params("plate-walls")
        unchosen.close()
    finally:
        dry.close()

    square = _walls_page(
        fake_microscope, run_doc, centres=0, calibrated=False, plate="384-well"
    )
    with _refuses(ValueError, walls_square_wells("384-well")):
        square.page_params("plate-walls")
    square.close()

    round_page = _walls_page(fake_microscope, run_doc, centres=0, calibrated=False)
    with _refuses(ValueError, walls_centres_problem(0) or ""):
        round_page.page_params("plate-walls")
    round_page.close()


def test_the_plates_refusal_comes_before_a_dry_runs_for_plate_walls(
    fake_core: FakeCore, run_doc: Path
) -> None:
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        page = _stood_page(dry, run_doc, "plate-walls")
        try:
            page.run("plate-walls", [])
            assert page.status().outcome == f"plate-walls: {PLACEHOLDER_REFUSAL}"
        finally:
            page.close()
    finally:
        dry.close()


def _walls_fit(well: str, centre: tuple[float, float], frames: int) -> WellFit:
    return WellFit(
        well=well,
        recorded_xy_um=(centre[0] + 4.0, centre[1] - 4.0),
        center_xy_um=centre,
        diameter_um=15298.4,
        size_error=0.0,
        rms_um=3.0,
        n_points=40,
        walls=tuple(
            WallFrame(side=side, x_um=0.0, y_um=0.0, n_points=10, frames=frames)
            for side in ("+X", "+Y", "-X", "-Y")
        ),
    )


def _walls_result(folder: str | None = "plate-walls/tok") -> PlateResult:
    """What ``plate-walls`` writes for two wells: four frames on each side of the first, two of the second."""
    return PlateResult(
        plate="24-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.1234,
        method="wall-images",
        residual_um=3.24,
        scale_error=0.0,
        n_refs=2,
        assumed=(
            "two wells: the residual checks their distance only; a third well "
            "checks the fit",
        ),
        walls=WallImages(
            objective="10x",
            pixel_size_um=1.0,
            image_shape_px=(100, 100),
            search_um=3900.0,
            wells=(
                # In the order the wells were measured, which is not alphabetical.
                _walls_fit("D5", (1.0, 2.0), 2),
                _walls_fit("B3", (10182.04, -10087.93), 4),
            ),
            frames=48,
            folder=folder,
            elapsed_s=16.5,
        ),
    )


def test_a_plate_walls_card_shows_the_wells_the_plate_and_the_frames_folder(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(
        run_doc,
        "plate-walls",
        "plate",
        _walls_result(),
        params={"plate": "24-well"},
    )
    page = _stood_page(fake_microscope, run_doc, "plate-walls")
    try:
        card = page.view().cards["plate-walls"]
        # §12.6: one line per well (the frames of its four sides summed), then the
        # plate; not the result's first eight keys with the walls cut away.
        assert card.lines == (
            "D5: centre (1.0, 2.0) µm, 15298 µm across, 8 frames",
            "B3: centre (10182.0, -10087.9) µm, 15298 µm across, 16 frames",
            "plate 24-well: residual 3.2 µm, rotation 0.12°",
        )
        # The folder is the frames the operator may open: under the run folder, built
        # the way the page builds every path (FM-24).
        assert card.files == (f"folder: {run_doc.parent / 'plate-walls' / 'tok'}",)
        assert card.assumed == (
            "two wells: the residual checks their distance only; a third well "
            "checks the fit",
        )
    finally:
        page.close()


@pytest.mark.parametrize("folder", [None, "", "   "], ids=["none", "empty", "blank"])
def test_a_walls_card_with_no_folder_names_no_file(
    fake_microscope: Microscope, run_doc: Path, folder: str | None
) -> None:
    _seed(run_doc, "plate-walls", "plate", _walls_result(folder))
    page = _stood_page(fake_microscope, run_doc, "plate-walls")
    try:
        card = page.view().cards["plate-walls"]
        assert card.files == ()
        assert len(card.lines) == 3
    finally:
        page.close()


@pytest.mark.parametrize(
    "walls",
    [
        {"wells": "none", "folder": "plate-walls/tok"},
        {"folder": 7, "wells": []},
        "not an object",
        [],
    ],
    ids=["wells-not-a-list", "a-folder-that-is-a-number", "text", "a-list"],
)
def test_a_walls_result_that_does_not_validate_keeps_the_generic_card(
    fake_microscope: Microscope, run_doc: Path, walls: object
) -> None:
    # The document is data (FM-125) and `view()` raises nothing for it: a `walls` the
    # model refuses is the generic card of the result's keys, never an exception.
    _record(
        run_doc,
        "plate-walls",
        provides="plate",
        result={
            "plate": "24-well",
            "a1_center_xy_um": [0.0, 0.0],
            "rotation_deg": 0.0,
            "method": "wall-images",
            "residual_um": 1.0,
            "scale_error": None,
            "n_refs": 2,
            "walls": walls,
        },
    )
    page = _stood_page(fake_microscope, run_doc, "plate-walls")
    try:
        card = page.view().cards["plate-walls"]
        assert "plate: 24-well" in card.lines
        assert not any(line.startswith("plate 24-well") for line in card.lines)
    finally:
        page.close()


def test_a_plate_registration_card_keeps_its_generic_lines(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _stood_page(fake_microscope, run_doc, "plate-registration")
    try:
        card = page.view().cards["plate-registration"]
        assert "method: references" in card.lines
        assert "plate: 96-well" in card.lines
        assert card.files == ()
    finally:
        page.close()


# --- the plate the page starts with (m5 §11.3, #332) --------------------------------

PLACEHOLDER_REFUSAL = (
    "choose the plate on the stage first: the page shows a 96-well placeholder"
)
REF_A1 = {"well": "A1", "x_um": 1000.0, "y_um": -500.0}
REF_B2 = {"well": "B2", "x_um": 10000.0, "y_um": -9500.0}
WALL_A1 = {"well": "A1", "side": "-X", "x_um": 5.0, "y_um": 6.0}


def test_a_fresh_document_names_no_plate_so_the_page_is_unchosen(
    page: PlatePage,
) -> None:
    assert (page.plate, page.plate_chosen) == ("96-well", False)


def test_the_plate_of_the_latest_provider_wins_over_an_older_registration(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # PR #351's note on #346: `plate-walls` also provides `plate`, and a document
    # holding an older registration and a newer wall registration opened on the older.
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate("24-well"),
        params={"plate": "24-well", "refs": [REF_A1]},
        minute=0,
    )
    _seed(
        run_doc,
        "plate-walls",
        "plate",
        _plate("6-well"),
        params={"plate": "6-well", "refs": [REF_A1]},
        minute=1,
    )
    page = _page(fake_microscope, run_doc)
    assert (page.plate, page.plate_chosen) == ("6-well", True)


def test_a_failed_latest_provider_still_names_the_plate(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The latest attempt decides the slot even when it failed (`RunDocument.provider`);
    # what it was asked to register is still what the operator is working on.
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate("96-well"),
        params={"plate": "96-well"},
        minute=0,
    )
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        status=StageStatus.failed,
        params={"plate": "24-well"},
        minute=1,
    )
    page = _page(fake_microscope, run_doc)
    assert (page.plate, page.plate_chosen) == ("24-well", True)


@pytest.mark.parametrize("named", ["nope", "96-WELL", 96, None, ["96-well"]])
def test_a_record_whose_plate_is_not_a_known_plate_is_ignored(
    fake_microscope: Microscope, run_doc: Path, named: object
) -> None:
    # A record is data (FM-125): nothing raises, the page asks for the plate, and the
    # captures of a record that names no plate it can build are not taken over.
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        status=StageStatus.failed,
        params={"plate": named, "refs": [REF_A1]},
    )
    page = _page(fake_microscope, run_doc)
    assert (page.plate, page.plate_chosen) == ("96-well", False)
    assert page.refs == ()


def test_a_plate_finding_record_names_the_plate_it_found_not_the_one_it_was_sized_for(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # `plate-finding`'s `plate` parameter sizes its survey (default 96-well); its result
    # is the plate it fitted. Read as the plate on the stage, the parameter opened the
    # page on a 96-well map and said it had been chosen (adversarial review of #332).
    _seed(
        run_doc,
        "plate-finding",
        "plate",
        _plate("24-well"),
        params={"plate": "96-well", "plates": []},
    )
    page = _page(fake_microscope, run_doc)
    assert (page.plate, page.plate_chosen) == ("24-well", True)


def test_a_failed_plate_finding_names_no_plate(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A survey that failed found no plate, and its parameter is not one: the page asks.
    _seed(
        run_doc,
        "plate-finding",
        "plate",
        status=StageStatus.failed,
        params={"plate": "24-well"},
    )
    page = _page(fake_microscope, run_doc)
    assert (page.plate, page.plate_chosen) == ("96-well", False)


def test_a_usable_result_names_the_plate_even_when_its_params_do_not(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate("24-well"),
        params={"plate": "nope"},
    )
    page = _page(fake_microscope, run_doc)
    assert (page.plate, page.plate_chosen) == ("24-well", True)


def test_the_profiles_sample_plate_is_chosen(run_doc: Path) -> None:
    sampled = Profile.model_validate(
        {"microscope": {"name": "demo-sample"}, "sample": {"plate": "24-well"}}
    )
    page = PlatePage.closed(sampled, run_doc, entry_points=[_ep(c) for c in STUBS])
    assert (page.plate, page.plate_chosen) == ("24-well", True)
    # The stand knows what its camera images: a sample whose plate is the default is
    # chosen too, and a run document that names a plate wins over the sample's.
    plain = Profile.model_validate(
        {"microscope": {"name": "demo-sample"}, "sample": {}}
    )
    page = PlatePage.closed(plain, run_doc, entry_points=[_ep(c) for c in STUBS])
    assert (page.plate, page.plate_chosen) == ("96-well", True)
    _seed(run_doc, "plate-registration", "plate", params={"plate": "6-well"})
    page = PlatePage.closed(sampled, run_doc, entry_points=[_ep(c) for c in STUBS])
    assert (page.plate, page.plate_chosen) == ("6-well", True)


def test_choose_plate_with_the_plate_shown_still_confirms_it(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    assert not page.plate_chosen
    fake_microscope.require(XYStage).move_to_um(1.0, 2.0)
    page.capture_centre("A1")
    page.choose_plate("96-well")
    # The early return for the same plate skips the reset, not the confirmation.
    assert (page.plate, page.plate_chosen) == ("96-well", True)
    assert [r.well for r in page.refs] == ["A1"]
    with _refuses(
        ValueError,
        "'nope' is not a plate; the plates are "
        + ", ".join(sorted(useq.registered_well_plate_keys())),
    ):
        page.choose_plate("nope")


def test_choosing_another_plate_chooses_it(page: PlatePage) -> None:
    page.choose_plate("24-well")
    assert (page.plate, page.plate_chosen) == ("24-well", True)


def test_a_refused_plate_name_does_not_confirm_the_placeholder(
    page: PlatePage,
) -> None:
    plates = ", ".join(sorted(useq.registered_well_plate_keys()))
    with _refuses(ValueError, f"'nope' is not a plate; the plates are {plates}"):
        page.choose_plate("nope")
    # The name is checked before anything changes: a typo does not open the gate.
    assert (page.plate, page.plate_chosen) == ("96-well", False)


def test_restore_reads_the_captures_of_the_latest_provider(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    refs = [REF_A1, REF_B2]
    # A registration after a wall registration: its centres and its walls.
    _seed(
        run_doc,
        "plate-walls",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": [REF_A1]},
    )
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": refs, "walls": [WALL_A1]},
        minute=1,
    )
    page = _page(fake_microscope, run_doc)
    assert page.refs == tuple(WellRef.model_validate(r) for r in refs)
    assert page.walls == (WallTouch.model_validate(WALL_A1),)


def test_restore_reads_only_the_centres_of_a_newer_wall_registration(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": [REF_A1], "walls": [WALL_A1]},
    )
    _seed(
        run_doc,
        "plate-walls",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": [REF_B2]},
        minute=1,
    )
    page = _page(fake_microscope, run_doc)
    assert page.refs == (WellRef.model_validate(REF_B2),)
    # `plate-walls` takes no walls: the older registration's are not the page's.
    assert page.walls == ()


def test_restore_takes_no_captures_from_another_provider_of_the_plate(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": [REF_A1], "walls": [WALL_A1]},
    )
    # A third party's `plate-finding` is the latest provider: whatever it holds under
    # `refs` is not a capture of this page.
    _seed(
        run_doc,
        "plate-finding",
        "plate",
        _plate(),
        params={"plate": "96-well", "refs": [REF_B2]},
        minute=1,
    )
    page = _page(fake_microscope, run_doc)
    assert (page.refs, page.walls) == ((), ())
    assert page.plate == "96-well"


def _stood_page(fake_microscope: Microscope, run_doc: Path, *names: str) -> PlatePage:
    return _page(fake_microscope, run_doc, _real(*names))


def test_an_unchosen_plate_refuses_plate_registration_in_the_controller(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _stood_page(fake_microscope, run_doc, "plate-registration")
    try:
        # A disabled button is a courtesy (FM-169): the controller itself refuses,
        # before anything is read, and nothing is written.
        page.run("plate-registration", ["plate=96-well", "refs=[]", "walls=[]"])
        assert page.status().running is None
        assert page.status().outcome == f"plate-registration: {PLACEHOLDER_REFUSAL}"
        assert not run_doc.exists()
        with _refuses(ValueError, PLACEHOLDER_REFUSAL):
            page.page_params("plate-registration")
        # The other pages' tools are not gated: only a tool that takes the plate is.
        assert page.page_params("stub") == []
        page.choose_plate("96-well")
        assert page.page_params("plate-registration")[0] == "plate=96-well"
    finally:
        page.close()


def test_an_unchosen_plate_is_the_tools_reason_and_the_next_steps_stuck_with_no_pairs(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _stood_page(fake_microscope, run_doc, "plate-registration")
    try:
        # With captures made, so that the pairs would not be empty if the step ignored
        # the plate: the plugin's own default must never be what a click runs.
        fake_microscope.require(XYStage).move_to_um(1.0, 2.0)
        page.capture_centre("A1")
        view = page.view()
        assert view.blocked["plate-registration"] == PLACEHOLDER_REFUSAL
        step = view.next_step
        assert step is not None
        assert step.plugin == "plate-registration"
        assert step.params == ()
        assert step.blocked == PLACEHOLDER_REFUSAL
        page.choose_plate("96-well")
        chosen = page.view()
        assert "plate-registration" not in chosen.blocked
        assert chosen.next_step is not None
        assert chosen.next_step.blocked is None
        assert chosen.next_step.params[0] == "plate=96-well"
    finally:
        page.close()


def test_a_closed_stand_is_still_the_reason_ahead_of_an_unchosen_plate(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = PlatePage.closed(
        fake_microscope.profile, run_doc, entry_points=_real("plate-registration")
    )
    try:
        assert page.view().blocked["plate-registration"] == "the stand is not connected"
        page.run("plate-registration", [])
        assert page.status().outcome == (
            "plate-registration: the stand is not connected"
        )
    finally:
        page.close()


def test_an_unchosen_plate_refuses_plate_walls_in_the_controller(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # `PAGE_PARAMS["plate-walls"]` names `plate` (#346): the gate is the table's, so
    # that tool is refused on a placeholder with no code of its own (m5 §12.2).
    page = _stood_page(fake_microscope, run_doc, "plate-walls")
    try:
        assert page.view().blocked["plate-walls"] == PLACEHOLDER_REFUSAL
        page.run("plate-walls", [])
        assert page.status().outcome == f"plate-walls: {PLACEHOLDER_REFUSAL}"
        page.choose_plate("96-well")
        assert page.view().blocked.get("plate-walls") != PLACEHOLDER_REFUSAL
    finally:
        page.close()


def test_a_usable_plate_gives_each_plate_tool_a_replace_note(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        params={"plate": "96-well"},
    )
    page = _stood_page(fake_microscope, run_doc, "plate-registration", "plate-walls")
    try:
        when = T0.astimezone().strftime("%Y-%m-%d %H:%M")
        note = (
            f"Run replaces the plate registered {when} (references, 2 wells). "
            "A failed run leaves no plate; the captures stay, so Forget the wrong "
            "one and Run again to get a plate back."
        )
        assert page.view().replaces == {
            "plate-registration": note,
            "plate-walls": note,
        }
    finally:
        page.close()


def test_the_replace_note_under_plate_finding_does_not_mention_captures(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # plate-finding fills the plate from the image: it has no capture to forget, so the
    # sentence about them is false under it (m5 §12.7). The tools that run from the
    # page's captures keep it.
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        _plate(),
        params={"plate": "96-well"},
    )
    page = _stood_page(
        fake_microscope,
        run_doc,
        "plate-registration",
        "plate-walls",
        "plate-finding",
    )
    try:
        when = T0.astimezone().strftime("%Y-%m-%d %H:%M")
        head = f"Run replaces the plate registered {when} (references, 2 wells). "
        replaces = page.view().replaces
        assert replaces["plate-finding"] == f"{head}A failed run leaves no plate."
        keeps = (
            f"{head}A failed run leaves no plate; the captures stay, so Forget the "
            "wrong one and Run again to get a plate back."
        )
        assert replaces["plate-registration"] == keeps
        assert replaces["plate-walls"] == keeps
    finally:
        page.close()


def test_a_failed_provider_gives_no_replace_note(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate(), params={"plate": "96-well"})
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        status=StageStatus.failed,
        params={"plate": "96-well"},
        minute=1,
    )
    page = _stood_page(fake_microscope, run_doc, "plate-registration")
    try:
        # Nothing is replaced: the slot is not usable now. A fresh document has none.
        assert page.view().replaces == {}
    finally:
        page.close()
    assert _stood_page_replaces(fake_microscope, run_doc.with_name("none.json")) == {}


def _stood_page_replaces(microscope: Microscope, run_doc: Path) -> dict[str, str]:
    page = _stood_page(microscope, run_doc, "plate-registration")
    try:
        return dict(page.view().replaces)
    finally:
        page.close()


def test_no_other_slot_gives_a_replace_note(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A card already says the last result; a note under every tool would teach the
    # operator to read none of them (§11.4): only the `plate` slot has one.
    _seed(run_doc, "stub", "stub", StubResult())
    _seed(run_doc, "scan-plan", "scan-plan", minute=1)
    page = _stood_page(fake_microscope, run_doc, "plate-registration")
    try:
        assert page.view().replaces == {}
        _seed(
            run_doc,
            "plate-registration",
            "plate",
            _plate(),
            params={"plate": "96-well"},
            minute=2,
        )
        assert set(page.view().replaces) == {"plate-registration"}
    finally:
        page.close()


# --- the view ---------------------------------------------------------------------------


def test_the_view_of_a_new_document_is_the_nominal_plate(page: PlatePage) -> None:
    view = page.view()
    assert view.plate == "96-well"
    assert view.nominal
    assert len(view.wells) == 96
    a1, h12 = view.wells[0], view.wells[-1]
    assert (a1.well, a1.center_um, a1.circular, a1.rotation_deg) == (
        "A1",
        (0.0, 0.0),
        True,
        0.0,
    )
    assert a1.size_um == pytest.approx((6400.0, 6400.0))
    assert h12.well == "H12"
    assert h12.center_um == pytest.approx((11 * PITCH_96_UM, -7 * PITCH_96_UM))
    assert (view.labels, view.route, view.problems) == ((), (), ())
    assert not any(w.selected for w in view.wells)
    assert all(w.counts is None and w.per_mm2 is None for w in view.wells)


def test_the_view_draws_the_registered_plate_and_marks_the_selection(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    page.select_text("A1,H12")
    view = page.view()
    assert not view.nominal
    by_well = {w.well: w for w in view.wells}
    assert by_well["H12"].center_um == pytest.approx(
        _centre("H12", a1=A1_OFF, rotation_deg=TURN_DEG)
    )
    assert by_well["H12"].rotation_deg == TURN_DEG
    assert [w.well for w in view.wells if w.selected] == ["A1", "H12"]


def test_the_map_of_a_plate_that_is_not_the_registered_one_is_nominal(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate("384-well"))
    page = _page(fake_microscope, run_doc)
    assert not page.view().nominal
    page.choose_plate("96-well")
    view = page.view()
    assert view.nominal
    assert (view.plate, len(view.wells)) == ("96-well", 96)


def test_the_view_shows_why_a_stage_is_unusable(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    _seed(run_doc, "scan-plan", "scan-plan", params={"wells": "A1"}, minute=1)
    _seed(run_doc, "scan", "scan", status=StageStatus.failed, minute=2)
    page = _page(fake_microscope, run_doc, _real("scan", "cells"))
    view = page.view()
    failed = "the last run of scan ended failed; run it again"
    assert view.checklist == (
        CheckItem("camera-calibration", True, "not run", None),
        CheckItem("plate", False, "ok", None),
        CheckItem("timings", True, "not run", None),
        CheckItem("scan-plan", False, "ok", None),
        CheckItem("scan", False, "unusable", failed),
        CheckItem("cells", False, "not run", None),
        CheckItem("density", False, "not run", None),
        CheckItem("route", False, "not run", None),
        CheckItem("report", False, "not run", None),
        CheckItem("nis-export", False, "not run", None),
    )
    # Every registered plugin is on the page: the stub `reader` is blocked as well.
    assert view.blocked == {
        "cells": failed,
        "reader": (
            "the run document has no up; smc run --list shows which plugin provides it"
        ),
    }


def test_a_blocked_tool_says_what_the_stand_lacks(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = PlatePage(fake_microscope, run_doc, entry_points=[_ep(StubScan)])
    assert page.view().blocked == {}

    def has(capability: type) -> bool:
        return capability is not XYStage

    with _replaced(fake_microscope, "has", has):
        assert page.view().blocked == {"scan": "this stand has no XYStage"}
    assert page.view().blocked == {}


def test_view_blocks_a_stage_with_a_reachable_checkpoint_after_its_other_reasons(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # A tool that cannot run at all says why first; the checkpoint is the last reason.
    page = _page(fake_microscope, run_doc, [_ep(StubScan)])
    # `stub` failed and the others were stopped: the text says which, per stage.
    statuses = {
        "stub": StageStatus.failed,
        "scan": StageStatus.stopped,
        "reader": StageStatus.stopped,
    }
    for name, status in statuses.items():
        _seed(run_doc, name, name, status=status, checkpoint={"done": 1})
    blocked = page.view().blocked
    assert blocked["stub"] == _checkpoint_text("stub", "failed")
    assert blocked["scan"] == _checkpoint_text("scan", "stopped")
    # What it reads comes before the checkpoint: `reader` reads `up`, which is absent.
    assert blocked["reader"] == (
        "the run document has no up; smc run --list shows which plugin provides it"
    )

    def has(capability: type) -> bool:
        return capability is not XYStage

    # What the stand lacks too: the capability is the reason, not the checkpoint.
    with _replaced(fake_microscope, "has", has):
        assert page.view().blocked["scan"] == "this stand has no XYStage"
    assert page.view().blocked["scan"] == _checkpoint_text("scan", "stopped")


@pytest.mark.parametrize("fault", ["another-profile", "a-dry-run"])
def test_view_does_not_block_a_stage_whose_checkpoint_resume_would_refuse(
    fake_microscope: Microscope, run_doc: Path, fault: str
) -> None:
    # A real page and a record `--resume` would refuse: nothing is protected, and a
    # blocked button here would trap the operator with no way out on the page.
    _seed(
        run_doc,
        "stub",
        "stub",
        status=StageStatus.stopped,
        checkpoint={"done": 1},
        profile="another" if fault == "another-profile" else "demo",
        dry_run=fault == "a-dry-run",
    )
    _seed(run_doc, "failing", "failing", status=StageStatus.done)
    page = _page(fake_microscope, run_doc)
    assert "stub" not in page.view().blocked
    assert "failing" not in page.view().blocked


@pytest.mark.parametrize(
    "recorded_dry", [False, True], ids=["real-record", "dry-record"]
)
def test_a_dry_run_page_protects_a_checkpoint_resume_can_reach(
    run_doc: Path, fake_core: FakeCore, recorded_dry: bool
) -> None:
    # `smc run --resume --dry-run` accepts a real record's checkpoint and a dry run's
    # own: only a real run refuses a dry run's. The page asks the same question, from
    # the stand it is on, in `view()` and in `run()`.
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        _seed(
            run_doc,
            "stub",
            "stub",
            status=StageStatus.stopped,
            checkpoint={"done": 1},
            dry_run=recorded_dry,
        )
        page = _page(dry, run_doc)
        assert page.view().blocked["stub"] == _checkpoint_text("stub", "stopped")
        before = run_doc.read_bytes()
        page.run("stub", [])
        assert page.status().outcome == f"stub: {_checkpoint_text('stub', 'stopped')}"
        assert run_doc.read_bytes() == before
        assert Stub.seen == []
    finally:
        dry.close()


def test_the_view_survives_an_unreadable_run_document(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    run_doc.write_bytes(b"not json")
    page = _page(fake_microscope, run_doc)
    view = page.view()
    assert view.problems == (
        f"run document: {run_doc}: is not valid JSON: Expecting value (line 1, "
        f"column 1); it was left as it is, restore it from a copy",
    )
    assert view.nominal
    assert len(view.wells) == 96
    assert {item.state for item in view.checklist} == {"not run"}
    assert run_doc.read_bytes() == b"not json"


def test_a_result_the_model_refuses_is_a_problem_and_not_a_crash(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin="density",
            provides="density",
            attempt="a1",
            status=StageStatus.done,
            started_utc=T0,
            finished_utc=T0,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result={"nope": 1},
        ),
    )
    page = _page(fake_microscope, run_doc)
    view = page.view()
    assert view.problems == (
        "density: result does not validate: labels: Field required",
    )
    assert view.labels == ()
    assert all(w.per_mm2 is None for w in view.wells)
    # Usable as far as the checklist goes: the document says ok, the model refuses.
    assert {i.slot: i.state for i in view.checklist}["density"] == "ok"


def test_the_view_carries_the_density_and_the_route(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    _seed(
        run_doc,
        "density",
        "density",
        DensityResult(
            labels=("live", "dead"),
            area_grid_um=10.0,
            wells=(
                DensityWell(
                    well="A1",
                    fields=2,
                    failed_fields=0,
                    area_mm2=2.0,
                    counts={"live": 4, "dead": 1},
                    outside_well=0,
                    per_mm2={"live": 2.0, "dead": 0.5},
                ),
                DensityWell(
                    well="A2",
                    fields=0,
                    failed_fields=1,
                    area_mm2=0.0,
                    counts={"live": 0, "dead": 0},
                    outside_well=0,
                    per_mm2=None,
                ),
            ),
        ),
        minute=1,
    )
    _seed(
        run_doc,
        "route",
        "route",
        RouteResult.model_validate(
            {
                "objective": None,
                "matrix_um_per_px": [[1.0, 0.0], [0.0, 1.0]],
                "image_shape_px": [10, 10],
                "fov_um": [10.0, 10.0],
                "labels": ["live"],
                "positions": [
                    {
                        "order": order,
                        "well": well,
                        "name": f"{well}_{order}",
                        "x_um": x,
                        "y_um": y,
                        "z_um": None,
                        "pfs_offset": None,
                        "counts": {"live": 1},
                    }
                    for order, well, x, y in ((1, "A2", 5.0, 6.0), (0, "A1", 1.0, 2.0))
                ],
                "uncovered": {},
                "distance_um": 5.0,
                "travel_s": None,
            }
        ),
        minute=2,
    )
    view = _page(fake_microscope, run_doc).view()
    by_well = {w.well: w for w in view.wells}
    assert view.labels == ("live", "dead")
    assert by_well["A1"].counts == {"live": 4, "dead": 1}
    assert by_well["A1"].per_mm2 == {"live": 2.0, "dead": 0.5}
    # Counted, but the area is unknown: the density is unknown, not zero.
    assert by_well["A2"].counts == {"live": 0, "dead": 0}
    assert by_well["A2"].per_mm2 is None
    # The density does not list it.
    assert by_well["B1"].counts is None
    assert by_well["B1"].per_mm2 is None
    assert [(s.order, s.well, s.x_um, s.y_um) for s in view.route] == [
        (0, "A1", 1.0, 2.0),
        (1, "A2", 5.0, 6.0),
    ]
    assert view.problems == ()


def test_wells_are_built_once_per_registered_plate(
    fake_microscope: Microscope, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    real = controller_module.well_geometry

    def counting(plan: useq.WellPlatePlan, well: str) -> Any:
        built.append(well)
        return real(plan, well)

    monkeypatch.setattr(controller_module, "well_geometry", counting)
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    page.view()
    page.view()
    page.click(*_centre("A1", a1=A1_OFF, rotation_deg=TURN_DEG), drive=False)
    assert len(built) == 96
    # A new registration of the same plate (another rotation) is a new map.
    _seed(run_doc, "plate-registration", "plate", _plate(rotation_deg=8.0), minute=1)
    page.view()
    assert len(built) == 192
    page.view()
    assert len(built) == 192


# --- a snap and its frame ------------------------------------------------------------------


class FrameCamera:
    """The six methods of ``Camera``, giving the frame it is handed; every one is bounded (FM-44).

    ``entered`` is set when a snap starts and, when ``block`` is set, the snap waits
    for ``release``: a snap in flight, as a driver that takes its time is.
    """

    def __init__(
        self,
        frame: np.ndarray,
        *,
        exposure_ms: float = 5.0,
        bit_depth: int = 16,
        pixel_size_um: float | None = 0.65,
        block: bool = False,
        fail_with: Callable[[], Exception] | None = None,
    ) -> None:
        self.frame = frame
        self._exposure_ms = exposure_ms
        self._bit_depth = bit_depth
        self._pixel_size_um = pixel_size_um
        self._block = block
        self._fail_with = fail_with
        self.entered = threading.Event()
        self.release = threading.Event()

    def snap(self) -> np.ndarray:
        self.entered.set()
        if self._block:
            assert self.release.wait(10), "the test never released the snap"
        if self._fail_with is not None:
            # Built here, so that nothing the camera holds keeps the error (FM-98).
            raise self._fail_with()
        return self.frame

    def exposure_ms(self) -> float:
        return self._exposure_ms

    def set_exposure_ms(self, value_ms: float) -> float:
        self._exposure_ms = value_ms
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (int(self.frame.shape[0]), int(self.frame.shape[1]))

    def bit_depth(self) -> int:
        return self._bit_depth

    def pixel_size_um(self) -> float:
        # ``None`` is a camera whose read fails: the facade's state() then says unknown.
        if self._pixel_size_um is None:
            raise HardwareError("the pixel size cannot be read")
        return self._pixel_size_um


def _ramp_frame() -> np.ndarray:
    """A 6 x 10 ``uint16`` frame, non-square and not symmetric, whose min is 7 and max 5907."""
    return (np.arange(60, dtype=np.uint16).reshape(6, 10) * 100 + 7).astype(np.uint16)


def _without(missing: type) -> Any:
    def has(capability: type) -> bool:
        return capability is not missing

    return has


def test_snap_keeps_the_frame_and_says_its_summary(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.require(XYStage).move_to_um(1000.0, -500.0)
    fake_microscope.override(
        Camera, FrameCamera(_ramp_frame(), exposure_ms=5.0, bit_depth=12)
    )
    assert page.frame() is None
    page.snap()
    assert page.join(10.0)
    frame = page.frame()
    assert frame is not None
    assert frame.image.shape == (6, 10)
    assert frame.image.dtype == np.uint16
    assert (frame.image == _ramp_frame()).all()
    assert frame.number == 1
    assert frame.bit_depth == 12
    assert frame.exposure_ms == 5.0
    assert frame.pixel_size_um == 0.65
    assert frame.xy_um == (1000.0, -500.0)
    assert frame.taken_utc.tzinfo is timezone.utc
    assert abs((datetime.now(timezone.utc) - frame.taken_utc).total_seconds()) < 60
    summary = "6x10 uint16, exposure 5 ms, min 7, max 5907"
    assert frame.summary == summary
    status = page.status()
    assert status.outcome == f"snap: {summary}"
    assert status.running is None


def test_a_snap_of_an_unknown_pixel_size_is_zero_and_never_a_guess(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_ramp_frame(), pixel_size_um=0.0))
    page.snap()
    assert page.join(10.0)
    frame = page.frame()
    assert frame is not None
    assert frame.pixel_size_um == 0.0


def test_a_snap_whose_pixel_size_cannot_be_read_still_keeps_the_frame_as_unknown(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # state() says None for a read that failed: the frame says 0.0, unknown, and
    # the snap is not lost for it.
    fake_microscope.override(Camera, FrameCamera(_ramp_frame(), pixel_size_um=None))
    assert fake_microscope.state().pixel_size_um is None
    page.snap()
    assert page.join(10.0)
    frame = page.frame()
    assert frame is not None, page.status().outcome
    assert frame.pixel_size_um == 0.0


def test_a_snap_on_a_stand_whose_stage_cannot_be_read_keeps_the_frame_without_a_position(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    def unreadable() -> None:
        raise HardwareError("the XY stage did not answer")

    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    stage = fake_microscope.require(XYStage)
    with _replaced(stage, "position_um", unreadable):
        assert fake_microscope.state().xy is None
        page.snap()
        assert page.join(10.0)
    frame = page.frame()
    assert frame is not None, page.status().outcome
    assert frame.xy_um is None
    assert page.status().outcome == f"snap: {frame.summary}"


def test_each_snap_is_a_new_number_and_a_read_only_frame(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame())
    fake_microscope.override(Camera, camera)
    page.snap()
    assert page.join(10.0)
    first = page.frame()
    assert first is not None
    page.snap()
    assert page.join(10.0)
    second = page.frame()
    assert second is not None
    assert (first.number, second.number) == (1, 2)
    assert first is not second
    assert not second.image.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        second.image[0, 0] = 1
    # The camera hands out its own buffer again and writes to it: what the page
    # keeps is a copy, so no session draws the camera's next frame.
    camera.frame[0, 0] = 9999
    assert second.image[0, 0] == 7


def test_snap_on_a_stand_without_a_camera_is_an_outcome_and_starts_no_thread(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    threads = threading.active_count()
    with _replaced(fake_microscope, "has", _without(Camera)):
        page.snap()
        assert page.lacks(Camera) == "this stand has no Camera"
    assert page.status().outcome == "snap: this stand has no Camera"
    assert page.status().running is None
    assert page.frame() is None
    assert threading.active_count() == threads


def test_snap_refuses_a_busy_or_a_closed_page_even_when_the_stand_has_no_camera(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # The refusal comes before the missing camera: a click that cannot start is told
    # why, as a run's is, and is not an outcome that hides the run that is going.
    with _replaced(fake_microscope, "has", _without(Camera)):
        page.run("slow", [])
        try:
            assert Slow.started.wait(5)
            with _refuses(RuntimeError, "slow is running"):
                page.snap()
        finally:
            Slow.release.set()
        assert page.join(10.0)
        page.close()
        with _refuses(RuntimeError, "the page is closed"):
            page.snap()
    assert page.frame() is None


def test_lacks_names_the_missing_capability(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    assert page.lacks(Camera) is None
    assert page.lacks(XYStage) is None
    with _replaced(fake_microscope, "has", _without(XYStage)):
        assert page.lacks(XYStage) == "this stand has no XYStage"
        assert page.lacks(Camera) is None


def test_a_frame_the_page_cannot_show_is_an_outcome_and_not_kept(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame())
    fake_microscope.override(Camera, camera)
    page.snap()
    assert page.join(10.0)
    kept = page.frame()
    assert kept is not None
    camera.frame = np.zeros((3, 4, 3), dtype=np.uint8)
    page.snap()
    assert page.join(10.0)
    assert page.status().outcome == (
        "snap: the camera returned a uint8 frame of shape (3, 4, 3); the page "
        "shows 2-D uint8 or uint16 frames"
    )
    assert page.frame() is kept
    # int16 has two bytes like a uint16, but is signed, and uint32 is unsigned but
    # wide: only the kind and the size together tell them from a frame we show.
    for dtype in (np.float32, np.int16, np.uint32):
        camera.frame = np.zeros((3, 4), dtype=dtype)
        page.snap()
        assert page.join(10.0)
        assert page.status().outcome == (
            f"snap: the camera returned a {np.dtype(dtype)} frame of shape (3, 4); "
            f"the page shows 2-D uint8 or uint16 frames"
        )
        assert page.frame() is kept


def test_a_snap_that_fails_keeps_no_exception(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    raised: list[weakref.ref[BaseException]] = []

    def failing() -> HardwareError:
        error = HardwareError("the camera timed out\nafter 5 s")
        raised.append(weakref.ref(error))
        return error

    fake_microscope.override(Camera, FrameCamera(_ramp_frame(), fail_with=failing))
    gc.disable()
    try:
        page.snap()
        assert page.join(10.0)
        assert len(raised) == 1
        # Dead without the cyclic collector: nothing the page holds reaches it (FM-98).
        assert raised[0]() is None
        assert page.status().outcome == (
            "snap: HardwareError: the camera timed out after 5 s"
        )
        assert page.frame() is None
        assert page.status().running is None
    finally:
        gc.enable()


def test_a_snap_and_a_run_refuse_each_other(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame(), block=True)
    fake_microscope.override(Camera, camera)
    try:
        page.snap()
        assert camera.entered.wait(5)
        assert page.status().running == "snap"
        with _refuses(RuntimeError, "snap is running"):
            page.run("stub", [])
        with _refuses(RuntimeError, "snap is running"):
            page.snap()
        with _refuses(RuntimeError, "snap is running"):
            page.resume()
    finally:
        camera.release.set()
    assert page.join(10.0)
    assert page.status().running is None
    assert Stub.seen == []
    page.run("slow", [])
    try:
        assert Slow.started.wait(5)
        with _refuses(RuntimeError, "slow is running"):
            page.snap()
    finally:
        Slow.release.set()
    assert page.join(10.0)
    assert page.frame() is not None
    assert page.frame().number == 1  # type: ignore[union-attr]


def test_a_snap_leaves_the_run_documents_progress_and_the_next_run_alone(
    page: PlatePage, fake_microscope: Microscope, run_doc: Path
) -> None:
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.snap()
    assert page.join(10.0)
    assert not run_doc.exists()
    _run_and_join(page, "stub")
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("stub done in ")
    assert page.frame() is not None


def test_stop_during_a_snap_tells_it_and_halts_the_stand(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame(), block=True)
    fake_microscope.override(Camera, camera)
    try:
        page.snap()
        assert camera.entered.wait(5)
        page.stop()
    finally:
        camera.release.set()
    assert page.join(10.0)
    status = page.status()
    assert status.stop_note == (
        "snap told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert status.halted
    assert status.running is None


def test_close_waits_for_a_snap_and_says_what_the_stop_did(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame(), block=True)
    fake_microscope.override(Camera, camera)
    try:
        page.snap()
        assert camera.entered.wait(5)
        # A snap is one device call: it ends when the driver returns, and the close
        # waits for it, then says what the stop did and how the snap ended.
        threading.Timer(0.3, camera.release.set).start()
        page.close()
    finally:
        camera.release.set()
    assert page.close_ended
    assert page.close_lines == (
        "snap told to stop; stopped: the stand is halted (Resume clears the halt)",
        "snap: 6x10 uint16, exposure 5 ms, min 7, max 5907",
    )
    with _refuses(RuntimeError, "the page is closed"):
        page.snap()


def test_a_snap_while_the_stand_is_halted_is_the_facades_refusal(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # The fake core's own camera, not an override: an override bypasses the halt.
    page.stop()
    with pytest.raises(HardwareError) as refusal:
        fake_microscope.require(Camera).snap()
    said = " ".join(str(refusal.value).split())
    name = type(refusal.value).__name__
    del refusal
    page.snap()
    assert page.join(10.0)
    assert page.status().outcome == f"snap: {name}: {said}"
    assert page.frame() is None


def test_a_snap_on_the_fake_stand_reads_its_own_camera(page: PlatePage) -> None:
    page.snap()
    assert page.join(10.0)
    frame = page.frame()
    assert frame is not None, page.status().outcome
    assert frame.image.ndim == 2
    assert page.status().outcome == f"snap: {frame.summary}"


# --- live ------------------------------------------------------------------------------------


class LivePlugin(Slow):
    """A plugin that is merely *named* ``live``: ending the live view must not end it."""

    name = provides = "live"


def _wait_for_frames(page: PlatePage, count: int, *, timeout_s: float = 5.0) -> Frame:
    """The frame numbered ``count`` or later; bounded, with the outcome in the failure (FM-44)."""
    deadline = time.monotonic() + timeout_s
    while True:
        frame = page.frame()
        if frame is not None and frame.number >= count:
            return frame
        assert time.monotonic() < deadline, (
            f"{count} frames not seen within {timeout_s} s: {page.status().outcome}"
        )
        time.sleep(0.005)


def test_live_keeps_frames_until_ended_and_says_how_many(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.live(True)
    _wait_for_frames(page, 3)
    status = page.status()
    assert status.running == "live"
    assert page.is_live
    assert status.progress is not None
    # A frame is kept before its progress is put: the progress may be one behind.
    assert status.progress.done >= 2
    assert status.progress == Progress("live", status.progress.done, None, "frames")
    page.live(False)
    assert page.join(10.0)
    status = page.status()
    kept = page.frame()
    assert kept is not None
    assert status.outcome is not None
    found = re.fullmatch(r"live ended: (\d+) frames in \d+\.\d s", status.outcome)
    assert found is not None, status.outcome
    assert int(found.group(1)) == kept.number
    assert status.running is None
    assert not page.is_live
    # Ending a live view is not an emergency: the stand is not halted.
    assert not status.halted


def test_live_logs_its_start_and_its_end_and_not_a_line_per_frame(
    page: PlatePage, fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    # FM-62, FM-148: a console that blocks must not hold the worker ten times a second.
    # The root at DEBUG, as pymmcore-plus leaves it once a core is open: every record
    # of every logger is a console line, so all of them are counted, not one logger's.
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        page.live(True)
        # The first frames may light up what the stand reads lazily, once (a backend
        # says that a light has no intensity property); after them, five more frames
        # are taken and no record, of any logger, is written for any of them.
        _wait_for_frames(page, 3)
        before = [(r.name, r.getMessage()) for r in caplog.records]
        _wait_for_frames(page, 8)
        after = [(r.name, r.getMessage()) for r in caplog.records]
        page.live(False)
        assert page.join(10.0)
    assert after == before
    lines = [(r.name, r.getMessage()) for r in caplog.records]
    assert lines[0] == ("smc.ui", "live: started from the plate page")
    assert lines[-1][0] == "smc.ui"
    assert re.fullmatch(r"live ended: \d+ frames in \d+\.\d s", lines[-1][1]), lines


def test_live_whose_pixel_size_cannot_be_read_logs_the_failed_read_once(
    page: PlatePage, fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    # #288, FM-62, FM-148: a read of state() that fails was a DEBUG line per frame
    # (13 lines for 10 frames and 3 status ticks, measured). Both writers of the line,
    # the live worker and the tab's tick, are in the window; every logger is counted.
    fake_microscope.override(Camera, FrameCamera(_ramp_frame(), pixel_size_um=None))
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        page.live(True)
        _wait_for_frames(page, 3)
        before = [(r.name, r.getMessage()) for r in caplog.records]
        _wait_for_frames(page, 8)
        for _ in range(3):
            page.status()
        after = [(r.name, r.getMessage()) for r in caplog.records]
        page.live(False)
        assert page.join(10.0)
    assert after == before
    assert [
        r.getMessage() for r in caplog.records if r.name == "smc.hardware.microscope"
    ] == [
        "state: pixel_size read failed (HardwareError: the pixel size cannot be read)"
    ]


def test_live_leaves_no_backlog_of_progress_when_no_session_reads_the_status(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A live view outlives every tab, and the tick of a tab is the only reader of the
    # progress queue: nothing a frame writes may wait there for a status() call.
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.live(True)
    try:
        _wait_for_frames(page, 5)
        assert page._queue.empty()
    finally:
        page.live(False)
    assert page.join(10.0)
    # And the count is still what the next status() reads.
    progress = page.status().progress
    assert progress is not None
    assert progress == Progress("live", progress.done, None, "frames")
    assert progress.done >= 5


def test_live_frames_do_not_count_as_ended_activities(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    before = page.activities_ended
    page.live(True)
    _wait_for_frames(page, 3)
    assert page.activities_ended == before
    page.live(False)
    assert page.join(10.0)
    assert page.activities_ended == before + 1


@pytest.mark.wallclock
def test_live_keeps_at_most_ten_frames_a_second(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A camera that answers at once: only the period holds the rate down.
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.live(True)
    first = _wait_for_frames(page, 1)
    began = time.monotonic()
    time.sleep(1.0)
    last = page.frame()
    elapsed = time.monotonic() - began
    page.live(False)
    assert page.join(10.0)
    assert last is not None
    delta = last.number - first.number
    assert delta <= elapsed / LIVE_PERIOD_S + 1, f"{delta} frames in {elapsed:.2f} s"
    # The cap is not a stall: frames do come.
    assert delta >= 3, f"{delta} frames in {elapsed:.2f} s"
    # The seconds the outcome tells are the live's own, from its start, not the last
    # frame's: the rate the owner reads at a stand is frames over these seconds.
    found = re.fullmatch(
        r"live ended: (\d+) frames in (\d+\.\d) s", page.status().outcome or ""
    )
    assert found is not None, page.status().outcome
    assert float(found.group(2)) >= 0.9, found.group(0)


class ClockedCamera(FrameCamera):
    """A camera whose snap costs ``snap_s`` on a clock the test owns; ``began`` is when each started.

    The first snap waits for ``go``, so that the test can put its spy on the live
    wait before the loop reaches it (FM-44, FM-157).
    """

    def __init__(self, frame: np.ndarray, *, now: list[float], snap_s: float) -> None:
        super().__init__(frame)
        self._now = now
        self._snap_s = snap_s
        self.began: list[float] = []
        self.go = threading.Event()

    def snap(self) -> np.ndarray:
        assert self.go.wait(10), "the test never let the first snap go"
        self.began.append(self._now[0])
        self._now[0] += self._snap_s
        return super().snap()


@pytest.mark.parametrize(
    ("snap_s", "wait_s"),
    [(0.2, 0.0), (0.05, 0.15), (0.3, 0.0)],
    ids=["snap-equals-period", "snap-shorter", "snap-longer"],
)
def test_the_live_wait_is_what_is_left_of_the_period_since_the_frame_began(
    page: PlatePage,
    fake_microscope: Microscope,
    monkeypatch: pytest.MonkeyPatch,
    snap_s: float,
    wait_s: float,
) -> None:
    # The cap is a period: the wait is what is left of it since the frame began, so a
    # frame that takes a whole period is followed by the next at once, and one that
    # takes less waits for the rest. A wait of the whole period after each frame would
    # put 0.4 s between two snaps in the first case. A clock the test owns, moved
    # only by the snap and the wait: a mean of wall-clock gaps failed on a stalled runner
    # (#311).
    monkeypatch.setattr(controller_module, "LIVE_PERIOD_S", 0.2)
    now = [100.0]
    camera = ClockedCamera(_ramp_frame(), now=now, snap_s=snap_s)
    fake_microscope.override(Camera, camera)
    waits: list[float | None] = []

    def spy(timeout: float | None = None) -> bool:
        # A wait with no limit, or one wait too many, fails here instead of hanging.
        waits.append(timeout)
        if timeout is None or len(waits) > 50:
            raise AssertionError(f"the live wait is not bounded: {waits[-5:]}")
        now[0] += timeout
        if len(waits) == 5:
            event.set()
        return event.is_set()

    # The name, not `time.monotonic`: the worker reads the real clock afterwards.
    with monkeypatch.context() as patch:
        patch.setattr(
            controller_module, "time", SimpleNamespace(monotonic=lambda: now[0])
        )
        page.live(True)
        current = page._current_run
        assert current is not None
        event = current[1]
        try:
            with _replaced(event, "wait", spy):
                camera.go.set()
                # The thread, not `page.join`: that reads the patched clock, which
                # only a snap and the spy move, so a worker blocked where the spy
                # does not see (a wait on another event) would hang the test and
                # not fail it (FM-44, #311).
                worker = page._worker
                assert worker is not None
                worker.join(10.0)
                assert not worker.is_alive(), f"the live loop never ended: {waits}"
        finally:
            camera.go.set()
            event.set()
    assert page.join(10.0)
    assert waits == pytest.approx([wait_s] * 5)
    assert np.diff(camera.began).tolist() == pytest.approx([max(snap_s, 0.2)] * 4)


def test_the_live_period_leaves_room_for_five_frames_a_second() -> None:
    # #262's five a second is two bounds: the page's own work per frame (the demo test,
    # with the period set to 0) and the period itself, which nothing else pins: the
    # tests above set it to 0.2 or read it, and the tie in test_plate_page.py is
    # skipped without the `ui` extra. The wait is proved on an owned clock above (#311).
    assert LIVE_PERIOD_S <= 0.2


def test_end_live_and_is_live_return_at_once_and_take_no_lock(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # FM-70: ending live is not an emergency but it runs on the loop that handles STOP,
    # so it never waits for the worker (a frame in flight may last for seconds) and
    # takes no lock the signal handler could re-enter. The lock is held by this thread
    # while another asks, under a join timeout (FM-44).
    camera = FrameCamera(_ramp_frame(), block=True)
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        assert camera.entered.wait(5)
        answers: list[bool] = []

        def end_and_ask() -> None:
            page.live(False)
            answers.append(page.is_live)

        asker = threading.Thread(target=end_and_ask, daemon=True)
        with page._lock:
            asker.start()
            asker.join(2.0)
            answered = not asker.is_alive()
        assert answered, "live(False) or is_live waited for the worker or for the lock"
        # The frame in flight is still the activity going: it ends at its next check.
        assert answers == [True]
    finally:
        camera.release.set()
    assert page.join(10.0)
    assert not page.is_live


def test_live_and_every_other_activity_refuse_each_other(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.live(True)
    try:
        assert page.status().running == "live"
        with _refuses(RuntimeError, "live is running"):
            page.run("stub", [])
        with _refuses(RuntimeError, "live is running"):
            page.snap()
        with _refuses(RuntimeError, "live is running"):
            page.live(True)
        with _refuses(RuntimeError, "live is running"):
            page.resume()
        with _refuses(RuntimeError, "live is running"):
            page.goto("A1")
    finally:
        page.live(False)
    assert page.join(10.0)
    assert Stub.seen == []
    page.run("slow", [])
    try:
        assert Slow.started.wait(5)
        with _refuses(RuntimeError, "slow is running"):
            page.live(True)
    finally:
        Slow.release.set()
    assert page.join(10.0)
    assert not page.is_live


def test_end_live_stops_only_the_live_it_started(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(LivePlugin)])
    try:
        # Nothing runs: ending a live view that is not there changes nothing.
        page.live(False)
        assert page.status().outcome is None
        assert not page.status().halted
        page.run("live", [])
        assert Slow.started.wait(5)
        page.live(False)
        # A run of a plugin called "live" is not the live view: it is still going.
        assert not Slow.ended.wait(0.3)
        status = page.status()
        assert status.running == "live"
        assert not page.is_live
        assert not status.halted
    finally:
        page.close()
    assert Slow.ended.is_set()


def test_live_on_a_stand_without_a_camera_is_an_outcome_and_starts_no_thread(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    threads = threading.active_count()
    before = page.activities_ended
    with _replaced(fake_microscope, "has", _without(Camera)):
        page.live(True)
    status = page.status()
    assert status.outcome == "live: this stand has no Camera"
    assert status.running is None
    assert not page.is_live
    assert page.activities_ended == before
    assert threading.active_count() == threads


def test_stop_during_live_ends_it_and_halts_the_stand(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame(), block=True)
    fake_microscope.override(Camera, camera)
    try:
        page.live(True)
        assert camera.entered.wait(5)
        page.stop()
    finally:
        camera.release.set()
    assert page.join(10.0)
    status = page.status()
    assert status.stop_note == (
        "live told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert status.halted
    assert status.running is None
    # The frame in flight completes and is kept; the loop ends at its next check.
    assert status.outcome is not None
    assert status.outcome.startswith("live ended: ")


def test_close_ends_live_and_says_what_the_stop_did(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_ramp_frame()))
    page.live(True)
    _wait_for_frames(page, 2)
    page.close()
    assert page.close_ended
    assert len(page.close_lines) == 2
    assert page.close_lines[0] == (
        "live told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert re.fullmatch(r"live ended: \d+ frames in \d+\.\d s", page.close_lines[1])
    with _refuses(RuntimeError, "the page is closed"):
        page.live(True)
    page.live(False)  # ends nothing, raises nothing


def test_a_live_frame_of_the_wrong_shape_ends_live_and_keeps_the_last_good_frame(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_ramp_frame())
    fake_microscope.override(Camera, camera)
    before = page.activities_ended
    page.live(True)
    _wait_for_frames(page, 2)
    camera.frame = np.zeros((2, 3, 4), dtype=np.uint16)
    assert page.join(10.0)
    kept = page.frame()
    assert kept is not None
    assert page.status().outcome == (
        f"live ended after {kept.number} frames: the camera returned a uint16 "
        f"frame of shape (2, 3, 4); the page shows 2-D uint8 or uint16 frames"
    )
    assert kept.image.shape == (6, 10)
    assert page.activities_ended == before + 1


def test_a_live_whose_camera_fails_keeps_no_exception(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    raised: list[weakref.ref[BaseException]] = []

    def failing() -> HardwareError:
        error = HardwareError("the camera timed out\nafter 5 s")
        raised.append(weakref.ref(error))
        return error

    fake_microscope.override(Camera, FrameCamera(_ramp_frame(), fail_with=failing))
    gc.disable()
    try:
        page.live(True)
        assert page.join(10.0)
        assert len(raised) == 1
        # Dead without the cyclic collector: nothing the page holds reaches it (FM-98).
        assert raised[0]() is None
        assert page.status().outcome == (
            "live ended after 0 frames: HardwareError: the camera timed out after 5 s"
        )
        assert page.frame() is None
        assert page.status().running is None
    finally:
        gc.enable()


def test_live_while_the_stand_is_halted_ends_at_its_first_frame_with_the_facades_refusal(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # The fake core's own camera, not an override: an override bypasses the halt.
    page.stop()
    with pytest.raises(HardwareError) as refusal:
        fake_microscope.require(Camera).snap()
    said = " ".join(str(refusal.value).split())
    name = type(refusal.value).__name__
    del refusal
    page.live(True)
    assert page.join(10.0)
    assert page.status().outcome == f"live ended after 0 frames: {name}: {said}"
    assert page.frame() is None


# --- the result card and the checklist -------------------------------------------------------


def _record(
    run_doc: Path,
    plugin: str,
    *,
    provides: str | None = None,
    status: StageStatus = StageStatus.done,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    started_minute: int = 0,
    finished_minute: int | None = 0,
    profile: str = "demo",
    dry_run: bool = False,
    attempt: str = "a0",
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A record whose result is the dict given as it stands, for values no model would dump."""
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=provides or plugin,
            attempt=attempt,
            reads=reads or {},
            status=status,
            started_utc=T0 + timedelta(minutes=started_minute),
            finished_utc=(
                None
                if finished_minute is None
                else T0 + timedelta(minutes=finished_minute)
            ),
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=result,
            error=error,
        ),
    )


def test_a_done_record_is_a_card_with_its_result_lines(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(
        run_doc,
        "stub",
        started_minute=1,
        finished_minute=3,
        result={
            "assumed": ["pixel size 0.65 um from the profile", "no stage calibration"],
            "text": "x" * 90,
            "note": "line one\nline two",
            "ratio": 12.3456789,
            "count": 3,
            "flag": True,
            "nothing": None,
            "wells": ["A1", "A2", "A3"],
            "stats": {"a": 1, "b": 2},
        },
    )
    cards = _page(fake_microscope, run_doc).view().cards
    assert list(cards) == ["stub"]
    card = cards["stub"]
    assert card.plugin == "stub"
    assert card.status == "done"
    # The finish, not the start.
    assert card.when_utc == "2026-10-29 09:03:00"
    assert card.profile == "demo"
    assert card.dry_run is False
    assert card.error is None
    assert card.lines == (
        f"text: {'x' * 80}…",
        "note: line one line two",
        "ratio: 12.3457",
        "count: 3",
        "flag: True",
        "nothing: None",
        "wells: 3 items",
        "stats: 2 entries",
    )
    assert card.assumed == (
        "pixel size 0.65 um from the profile",
        "no stage calibration",
    )


def test_a_string_of_exactly_eighty_characters_is_not_cut(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"text": "y" * 80})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.lines == (f"text: {'y' * 80}",)


def test_a_card_shows_at_most_eight_values_and_says_how_many_more(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    ten = {f"k{i}": i for i in range(10)}
    _record(run_doc, "stub", result={"assumed": ["a"], **ten})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.lines == (*(f"k{i}: {i}" for i in range(8)), "… 2 more")
    eight = {f"k{i}": i for i in range(8)}
    _record(run_doc, "stub", result=eight)
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.lines == tuple(f"k{i}: {i}" for i in range(8))


def test_a_card_says_when_it_was_a_dry_run_on_another_profile_and_without_a_finish(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(
        run_doc,
        "stub",
        status=StageStatus.running,
        started_minute=7,
        finished_minute=None,
        profile="lightsheet",
        dry_run=True,
    )
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert (card.status, card.when_utc) == ("running", "2026-10-29 09:07:00")
    assert (card.profile, card.dry_run) == ("lightsheet", True)


@pytest.mark.parametrize(
    "status", [StageStatus.stopped, StageStatus.failed, StageStatus.running]
)
def test_a_stopped_or_failed_record_shows_no_result_values(
    fake_microscope: Microscope, run_doc: Path, status: StageStatus
) -> None:
    # FM-154: the values of a record that did not end done are not a result.
    _record(
        run_doc,
        "stub",
        status=status,
        result={"assumed": ["a guess"], "n": 7, "text": "partial"},
        error="the camera\nfailed",
    )
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.status == status.value
    assert card.lines == ()
    assert card.assumed == ()
    assert card.error == "the camera failed"


def test_a_card_is_made_for_a_tool_with_a_record_and_for_no_other(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"n": 1})
    _record(run_doc, "slow", result={"n": 2})
    # A record of a plugin this registry does not hold is no tool of the page.
    _record(run_doc, "gone", result={"n": 3})
    cards = _page(fake_microscope, run_doc).view().cards
    assert sorted(cards) == ["slow", "stub"]
    assert _page(fake_microscope, run_doc.with_name("none.json")).view().cards == {}


def test_a_record_with_no_result_is_a_card_with_no_lines(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result=None)
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert (card.status, card.lines, card.assumed) == ("done", (), ())
    assert card.files == ()


# --- where the files are: a card names them (m5 §8.4, #293) ---------------------------------


def test_a_card_names_its_files_under_the_run_folder(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(
        run_doc,
        "stub",
        result={"folder": "scan/t1", "n": 3, "file": "report/t1.html"},
    )
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    # A file key is a `files` entry, in the result's order, and not a value line.
    assert card.files == (
        f"folder: {run_doc.parent / 'scan' / 't1'}",
        f"file: {run_doc.parent / 'report' / 't1.html'}",
    )
    assert card.lines == ("n: 3",)


def test_an_absolute_file_is_shown_as_given(
    fake_microscope: Microscope, run_doc: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere" / "positions.xml"
    _record(run_doc, "stub", result={"file": str(elsewhere)})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.files == (f"file: {elsewhere}",)


@pytest.mark.parametrize(
    "status", [StageStatus.stopped, StageStatus.failed, StageStatus.running]
)
def test_a_failed_card_names_no_file(
    fake_microscope: Microscope, run_doc: Path, status: StageStatus
) -> None:
    # FM-154: a record that did not end done wrote no result to point at.
    _record(
        run_doc,
        "stub",
        status=status,
        result={"file": "report/partial.html", "folder": "scan/t1"},
        error="the camera failed",
    )
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.files == ()
    assert card.lines == ()


def test_a_files_entry_is_cut_at_the_text_bound(
    fake_microscope: Microscope, run_doc: Path, tmp_path: Path
) -> None:
    long = tmp_path / ("d" * 700)
    _record(run_doc, "stub", result={"file": str(long)})
    entry = _page(fake_microscope, run_doc).view().cards["stub"].files[0]
    assert len(entry) == 601
    assert entry.startswith("file: ")
    assert entry.endswith("d…")
    # Exactly at the bound it is whole: a relative value, joined to the run folder.
    room = 600 - len("file: ") - len(str(run_doc.parent / "x")) + 1
    whole = "w" * room
    _record(run_doc, "stub", result={"file": whole})
    entry = _page(fake_microscope, run_doc).view().cards["stub"].files[0]
    assert entry == f"file: {run_doc.parent / whole}"
    assert len(entry) == 600


def test_a_path_with_a_line_break_is_one_line(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"file": "a\nb.html"})
    entry = _page(fake_microscope, run_doc).view().cards["stub"].files[0]
    assert "\n" not in entry
    assert entry.endswith("a b.html")


@pytest.mark.parametrize("value", [3, None, True, ["a"], {"a": 1}, "", "   "])
def test_a_file_key_with_no_path_in_it_is_a_value_line_like_any_other(
    fake_microscope: Microscope, run_doc: Path, value: object
) -> None:
    # `files` claim a path: a number, a list or a blank text is no path (FM-125).
    _record(run_doc, "stub", result={"file": value})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.files == ()
    assert len(card.lines) == 1
    assert card.lines[0].startswith("file:")


def test_file_keys_do_not_count_against_the_eight_values_of_a_card(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    eight = {f"k{i}": i for i in range(8)}
    _record(run_doc, "stub", result={"file": "f.txt", "folder": "d", **eight})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.lines == tuple(f"k{i}: {i}" for i in range(8))
    assert len(card.files) == 2


def test_a_card_made_without_a_run_folder_shows_the_file_as_written(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"file": "report/t1.html"})
    record = RunDocument.load(run_doc).stages["stub"]
    assert controller_module._card(record).files == ("file: report/t1.html",)
    folder = run_doc.parent / "elsewhere"
    assert controller_module._card(record, run_dir=folder).files == (
        f"file: {folder / 'report' / 't1.html'}",
    )


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        # Absolute in the flavour of the PC that wrote it: shown as written, never
        # joined to this run folder (on Windows `/data/p.xml` is rooted, not absolute,
        # and the run folder's drive would invent `C:\data\p.xml`).
        ("/data/exports/p.xml", "/data/exports/p.xml"),
        ("D:/export/p.xml", "D:/export/p.xml"),
        ("//srv/share/p.xml", "//srv/share/p.xml"),
        # Relative: under the run folder, in the separators of this PC.
        ("report/t1.html", None),
    ],
)
@pytest.mark.parametrize("flavour", [PureWindowsPath, PurePosixPath])
def test_a_file_in_either_flavour_is_shown_as_written_and_a_relative_one_is_joined(
    flavour: type[PurePath], value: str, shown: str | None
) -> None:
    # By flavour, not by this OS: every runner proves both (FM-24).
    run_dir = flavour("C:/runs/exp1" if flavour is PureWindowsPath else "/runs/exp1")
    expected = str(run_dir / value) if shown is None else shown
    assert controller_module._file_path(value, run_dir) == expected
    if shown is None:
        assert expected.startswith(str(run_dir))


def test_the_run_documents_path_is_absolute_not_resolved(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the operator copies is the path as the command was given it: a link in the
    # run folder's path stays a link (m5 §8.4).
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this PC cannot make a symbolic link")
    monkeypatch.chdir(tmp_path)
    page = _page(fake_microscope, Path("link") / "run.json")
    assert page.run_doc == tmp_path.absolute() / "link" / "run.json"
    assert page.run_doc != real.resolve() / "run.json"


def test_the_run_documents_path_is_absolute_and_the_cards_files_follow_it(
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(run_doc.parent)
    relative = Path("run.json")
    _record(relative, "stub", result={"file": "report/t1.html"})
    page = _page(fake_microscope, relative)
    # The path as the command was given it, made absolute where the server runs.
    assert page.run_doc == Path.cwd() / "run.json"
    assert page.run_doc.is_absolute()
    entry = page.view().cards["stub"].files[0]
    assert entry == f"file: {Path.cwd() / 'report' / 't1.html'}"


def test_the_checklist_lists_a_slot_the_workflow_does_not_name(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "zeta-tool", status=StageStatus.done, result={})
    _record(run_doc, "alpha-tool", status=StageStatus.failed)
    _record(run_doc, "scan", result={})
    items = _page(fake_microscope, run_doc).view().checklist
    named = [item.slot for item in items[:10]]
    assert named == [
        "camera-calibration",
        "plate",
        "timings",
        "scan-plan",
        "scan",
        "cells",
        "density",
        "route",
        "report",
        "nis-export",
    ]
    # M3's slots keep their order and their state; the others follow by name, optional.
    assert items[4] == CheckItem("scan", False, "ok", None)
    assert items[10:] == (
        CheckItem(
            "alpha-tool",
            True,
            "unusable",
            "the last run of alpha-tool ended failed; run it again",
        ),
        CheckItem("zeta-tool", True, "ok", None),
    )


def _record_at(
    run_doc: Path,
    plugin: str,
    started: datetime,
    finished: datetime | None = None,
    *,
    result: dict[str, Any] | None = None,
) -> None:
    """A done record with the dates given as they are, offsets included."""
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=plugin,
            attempt="a0",
            status=StageStatus.done,
            started_utc=started,
            finished_utc=finished,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result,
        ),
    )


def test_a_card_time_is_utc_whatever_the_offset_of_the_record(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # 11:00 at +02:00 is 09:00 UTC: the card says UTC, so it must be.
    zone = timezone(timedelta(hours=2))
    _record_at(run_doc, "stub", datetime(2026, 10, 29, 11, 0, tzinfo=zone))
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.when_utc == "2026-10-29 09:00:00"


@pytest.mark.parametrize(
    "when",
    [
        datetime.fromisoformat("0001-01-01T00:00:00+05:00"),
        datetime.fromisoformat("9999-12-31T23:00:00-05:00"),
    ],
    ids=["year-1-ahead-of-utc", "year-9999-behind-utc"],
)
def test_a_date_utc_cannot_hold_is_an_unknown_time_and_the_view_still_builds(
    fake_microscope: Microscope, run_doc: Path, when: datetime
) -> None:
    # A document nobody should edit by hand, edited: `view()` raises nothing for it.
    _record_at(run_doc, "stub", when)
    view = _page(fake_microscope, run_doc).view()
    assert view.cards["stub"].when_utc == "????-??-?? ??:??:??"
    assert view.cards["stub"].status == "done"


def test_an_assumed_that_is_a_text_or_has_line_breaks_is_one_line_each(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"assumed": ["pixel size\ntaken as given", "x"]})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == ("pixel size taken as given", "x")
    _record(run_doc, "stub", result={"assumed": "one\ntwo"})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == ("one two",)


def test_a_key_of_a_result_is_cut_and_flattened_like_a_value(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"k" * 100: 1, "two\nlines": 2})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.lines == (f"{'k' * 80}…: 1", "two lines: 2")


def test_a_bare_assumption_and_a_profile_with_a_line_break_are_bounded_and_one_line(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The `assumed` that is one text or an object, not a list, and the profile that a
    # hand-edited document gives a line break: each is bounded like the rest of the card.
    _record(run_doc, "stub", result={"assumed": "a" * 5000}, profile="x\ny")
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == (f"{'a' * 600}…",)
    assert card.profile == "x y"
    _record(run_doc, "stub", result={"assumed": {"note": "v" * 1000}})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert len(card.assumed) == 1
    assert len(card.assumed[0]) == 601


def _stale_reader(run_doc: Path, *, read_attempt: str, **kwargs: Any) -> None:
    """``reader`` read the ``up`` slot at ``read_attempt``; ``up``'s provider is attempt ``a2``."""
    _record(
        run_doc,
        "upstream",
        provides="up",
        attempt="a2",
        started_minute=2,
        finished_minute=2,
        result={},
    )
    _record(
        run_doc,
        "reader",
        started_minute=3,
        finished_minute=3,
        reads={"up": ReadRecord(plugin="upstream", attempt=read_attempt)},
        **kwargs,
    )


def test_a_done_record_that_read_an_earlier_input_has_a_stale_card(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _stale_reader(run_doc, read_attempt="a1", result={"n": 4, "assumed": ["a guess"]})
    assert RunDocument.load(run_doc).is_stale("reader")
    card = _page(fake_microscope, run_doc).view().cards["reader"]
    assert card.stale is True
    # Marked, not hidden: a stale result is still a result, of inputs that moved.
    assert card.lines == ("n: 4",)
    assert card.assumed == ("a guess",)


def test_a_fresh_record_and_a_failed_stale_one_have_no_stale_mark(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _stale_reader(run_doc, read_attempt="a2", result={"n": 1})
    assert not RunDocument.load(run_doc).is_stale("reader")
    assert _page(fake_microscope, run_doc).view().cards["reader"].stale is False
    for status in (StageStatus.failed, StageStatus.stopped):
        _stale_reader(run_doc, read_attempt="a1", status=status, error="it failed")
        # The document calls it stale whatever its status; the card marks a result.
        assert RunDocument.load(run_doc).is_stale("reader")
        card = _page(fake_microscope, run_doc).view().cards["reader"]
        assert (card.status, card.stale) == (status.value, False)


def test_a_card_is_bounded_whatever_its_record_holds(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    result: dict[str, Any] = {"assumed": ["a" * 700] * 2000}
    result.update({f"key{i}": i for i in range(2000)})
    _record(run_doc, "stub", result=result, error="e" * 100_000, profile="p" * 5000)
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert len(card.assumed) == 9
    assert card.assumed[:8] == (f"{'a' * 600}…",) * 8
    assert card.assumed[8] == "… 1992 more"
    assert card.lines == (*(f"key{i}: {i}" for i in range(8)), "… 1992 more")
    assert card.error == f"{'e' * 600}…"
    assert card.profile == f"{'p' * 80}…"


def test_eight_assumptions_are_shown_whole_and_nine_are_cut_to_eight_and_a_count(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    items = [f"assumption {i}" for i in range(9)]
    _record(run_doc, "stub", result={"assumed": items[:8]})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == tuple(items[:8])
    _record(run_doc, "stub", result={"assumed": items})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == (*items[:8], "… 1 more")


def test_an_assumption_a_plugin_writes_today_is_shown_whole(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # 236 characters is the longest a built-in plugin writes (the route's
    # unlocked-frame line for 1536 stops); 600 is the bound, and 601 is cut.
    _record(
        run_doc,
        "stub",
        result={"assumed": ["z" * 236, "y" * 600, "x" * 601]},
    )
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == ("z" * 236, "y" * 600, f"{'x' * 600}…")


def test_a_non_string_assumption_and_a_stopped_records_error_are_cut_too(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _record(run_doc, "stub", result={"assumed": [{"note": "v" * 1000}, 7]})
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert len(card.assumed) == 2
    assert card.assumed[0].startswith("{'note': 'vvv")
    assert len(card.assumed[0]) == 601
    assert card.assumed[1] == "7"
    _record(
        run_doc,
        "stub",
        status=StageStatus.stopped,
        result={"assumed": ["hidden"]},
        error="line one\n" + "s" * 5000,
    )
    card = _page(fake_microscope, run_doc).view().cards["stub"]
    assert card.assumed == ()
    assert card.error is not None
    assert len(card.error) == 601
    assert card.error.startswith("line one sss")


def test_a_checklist_without_other_slots_is_m3s(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    assert len(_page(fake_microscope, run_doc).view().checklist) == 10


# --- the tools ----------------------------------------------------------------------------


def test_every_key_and_page_param_is_a_field_of_its_plugin(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    names = (
        "scan-plan",
        "plate-registration",
        "plate-walls",
        "camera-calibration",
        "scan",
        "cells",
        "route",
    )
    page = PlatePage(fake_microscope, run_doc, entry_points=_real(*names))
    plugins = page._registry().plugins
    assert set(names) <= set(
        plugins
    )  # a plugin that failed to load must not hide a row
    for table in (KEY_PARAMS, PAGE_PARAMS):
        assert table
        for plugin, fields in table.items():
            assert plugin in plugins
            for field in fields:
                assert field in plugins[plugin].plugin.Params.model_fields, (
                    plugin,
                    field,
                )


def test_tools_follow_the_workflow_order_and_hide_what_the_page_fills(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = PlatePage(
        fake_microscope,
        run_doc,
        entry_points=[
            *_real("scan", "cells", "scan-plan", "plate-registration", "goto-well"),
            _ep(Stub),
        ],
    )
    tools = page.tools()
    # The plugins `STAGES` does not name follow under `More`, by name.
    assert [t.name for t in tools] == [
        "plate-registration",
        "scan-plan",
        "scan",
        "cells",
        "goto-well",
        "stub",
    ]
    assert [t.stage for t in tools] == [
        "Set up",
        "Choose",
        "Acquire",
        "Find",
        "More",
        "More",
    ]
    by_name = {t.name: t for t in tools}
    plan = by_name["scan-plan"]
    plan_rows = [row[0] for row in (*plan.key, *plan.advanced)]
    assert "wells" not in plan_rows
    assert [row[0] for row in plan.key] == ["objective", "overlap", "coverage"]
    assert plan_rows[len(plan.key) :] == [
        "serpentine",
        "rows",
        "columns",
        "clip_to_well",
        "pixel_size_um",
    ]
    registration = by_name["plate-registration"]
    registration_rows = {row[0] for row in (*registration.key, *registration.advanced)}
    assert registration_rows == {"max_rotation_deg"}
    assert registration.summary.startswith("Locate a well plate on the stage")
    assert by_name["scan"].key[0][0] == "autofocus"


def test_cells_offers_the_detector_as_a_key_parameter(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # On the demo-sample stand the operator picks `blobs` here, not under the fold
    # with the model path and the device, which `blobs` does not take (#260).
    page = PlatePage(fake_microscope, run_doc, entry_points=_real("cells"))

    (cells,) = page.tools()

    assert cells.name == "cells"
    assert [row[0] for row in cells.key] == ["detector", "score_threshold"]
    assert {row[0] for row in cells.advanced} == {
        "model_path",
        "device",
        "max_overlap",
        "merge_radius_um",
        # The model's pre-processing (#342): the page lists every parameter of a plugin.
        "model_um_per_px",
        "low_percentile",
        "high_percentile",
        "checked_um_per_px",
    }
    detector = cells.key[0]
    assert "blobs" in " ".join(str(part) for part in detector)


class Alpha(Stub):
    name = provides = "alpha-tool"


class Zeta(Stub):
    name = provides = "zeta-tool"


class NeedyParams(PluginParams):
    zebra: int
    apple: int = 2
    mango: str


class KeyedParams(PluginParams):
    labels: str = ""
    fields_per_well: int = 1
    zebra: int
    apple: int = 2
    mango: str


class PlanParams(PluginParams):
    objective: str = ""
    wells: str
    must: int


class Needy:
    """A plugin no table names, with two required fields around an optional one."""

    name = provides = "needy"
    summary = "A test plugin with required fields."
    Params = NeedyParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: NeedyParams, ctx: RunContext) -> StubResult:
        return StubResult()


class NeedyRoute(Needy):
    """Registered as ``route``: ``KEY_PARAMS`` names two of its fields, and two more are required."""

    name = provides = "route"
    Params = KeyedParams  # type: ignore[assignment]


class NeedyPlan(Needy):
    """Registered as ``scan-plan``: its required ``wells`` is the page's, and ``must`` is not."""

    name = provides = "scan-plan"
    Params = PlanParams  # type: ignore[assignment]


def _unbuildable_default() -> int:
    raise RuntimeError("cannot build the default\nof this field")


class BrokenParams(PluginParams):
    n: int = Field(default_factory=_unbuildable_default)


class Broken(Needy):
    """Third-party parameters that ``describe_params`` cannot read (FM-113)."""

    name = provides = "broken"
    Params = BrokenParams  # type: ignore[assignment]


def test_every_registered_plugin_is_a_tool_and_the_unnamed_are_under_more(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(
        fake_microscope,
        run_doc,
        [_ep(Zeta), _ep(Alpha), *_real("scan", "plate-registration")],
    )
    tools = page.tools()
    # What `STAGES` names, in its order; then every other plugin, by name.
    assert [t.name for t in tools] == [
        "plate-registration",
        "scan",
        "alpha-tool",
        "failing",
        "goto-well",
        "progressing",
        "reader",
        "slow",
        "stub",
        "zeta-tool",
    ]
    assert [t.stage for t in tools] == ["Set up", "Acquire", *["More"] * 8]
    assert tools[-1].name == "zeta-tool"
    assert tools[-1].summary == "A test plugin."


def test_a_required_field_is_never_folded(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(
        fake_microscope,
        run_doc,
        [_ep(Needy), _ep(NeedyRoute), _ep(NeedyPlan)],
    )
    by_name = {t.name: t for t in page.tools()}
    needy = by_name["needy"]
    assert [row[0] for row in needy.key] == ["zebra", "mango"]
    assert [row[0] for row in needy.advanced] == ["apple"]
    assert [row[2] for row in needy.key] == ["(required)", "(required)"]
    # `KEY_PARAMS` first, in its order; then the required ones, in declaration order.
    route = by_name["route"]
    assert [row[0] for row in route.key] == [
        "labels",
        "fields_per_well",
        "zebra",
        "mango",
    ]
    assert [row[0] for row in route.advanced] == ["apple"]
    # A field the page fills is not offered, required or not.
    plan = by_name["scan-plan"]
    assert [row[0] for row in plan.key] == ["objective", "must"]
    assert plan.advanced == ()


class RequiredKeyParams(PluginParams):
    objective: str
    wells: str


class KeyedPlan(Needy):
    """Registered as ``scan-plan``: its required ``objective`` is also named by ``KEY_PARAMS``."""

    name = provides = "scan-plan"
    Params = RequiredKeyParams  # type: ignore[assignment]


def test_a_required_field_that_key_params_names_is_one_row_and_not_two(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(KeyedPlan)])
    (plan,) = [t for t in page.tools() if t.name == "scan-plan"]
    assert [row[0] for row in plan.key] == ["objective"]
    assert plan.advanced == ()


def test_the_tools_are_described_once_and_not_at_every_view(
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tap rebuilds the view: describing every plugin's parameters again each time
    # is the work FM-112 keeps out of it.
    described: list[object] = []
    real = controller_module.describe_params

    def counting(model: Any) -> Any:
        described.append(model)
        return real(model)

    monkeypatch.setattr(controller_module, "describe_params", counting)
    page = _page(fake_microscope, run_doc)
    for _ in range(3):
        page.tools()
        page.view()
    assert len(described) == len(STUBS)


def test_a_reason_a_plugin_failed_to_load_is_one_line(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc)
    page._registry_found = Registry(
        plugins=page._registry().plugins,
        failures=(
            LoadFailure(
                "ghost", "ghost:Plugin", None, "could not be imported:\n  boom"
            ),
        ),
    )
    assert page.view().problems == ("plugin ghost: could not be imported: boom",)


def test_a_plugin_that_failed_to_load_is_a_problem(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    ghost = EntryPoint(name="ghost", value="no_such_module_xyz:Plugin", group=GROUP)
    page = _page(fake_microscope, run_doc, [ghost])
    assert "ghost" not in [t.name for t in page.tools()]
    assert page.view().problems == (
        "plugin ghost: could not be imported: ModuleNotFoundError: No module "
        "named 'no_such_module_xyz'",
    )


def test_a_plugin_whose_parameters_cannot_be_described_is_a_problem_and_no_crash(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    page = _page(fake_microscope, run_doc, [_ep(Broken)])
    names = [t.name for t in page.tools()]
    assert "broken" not in names
    assert "stub" in names
    said = "plugin broken: RuntimeError: cannot build the default of this field"
    assert page.view().problems == (said,)
    # Described once: the same answer, and not a second line, the next time.
    assert [t.name for t in page.tools()] == names
    assert page.view().problems == (said,)


def test_the_controller_imports_without_panel() -> None:
    # A child process, so that the blocked modules do not outlive the test: `None`
    # in `sys.modules` makes `import panel` raise, as on a PC without the extra.
    code = (
        "import sys\n"
        "sys.modules['panel'] = None\n"
        "sys.modules['bokeh'] = None\n"
        "import smc.ui.controller\n"
    )
    # No `-I`: the interpreter keeps PYTHONPATH, so it imports the code under test
    # (`scripts/dev/mutant.py` puts a copy's `src` there) and not another checkout's.
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr


# --- the scan plan on the map and its card (m5 §14, #320) ----------------------------


def _scan_plan_result(
    wells: tuple[str, ...] = ("B11", "B2", "A1"),
    *,
    rows: int = 2,
    columns: int = 2,
    fov_um: tuple[float, float] = (512.0, 512.0),
    estimate_s: float | None = 42.16,
) -> ScanPlanResult:
    """A scan plan built as test_report.py builds one: ``plan_fields`` over a registered plate.

    The default wells are in visiting order, which is neither the alphabetical one
    (A1, B11, B2) nor the natural one (A1, B2, B11): a sort in the code fails the
    tests that read the order (FM-204).
    """
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    half_w, half_h = fov_um[0] / 2, fov_um[1] / 2
    plate_plan, fields = plan_fields(
        plate,
        list(wells),
        fov_um=fov_um,
        footprint_of=lambda x, y: [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ],
        rows=rows,
        columns=columns,
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
        rows=rows,
        columns=columns,
        fields=fields,
        estimate_s=estimate_s,
    )


def _with_fields(result: ScanPlanResult, count: int) -> ScanPlanResult:
    """``result`` with ``count`` fields (the planned ones, repeated): for the bound, not for the geometry."""
    return result.model_copy(
        update={
            "fields": tuple(result.fields[i % len(result.fields)] for i in range(count))
        }
    )


def _plan_page(microscope: Microscope, run_doc: Path) -> PlatePage:
    return _page(microscope, run_doc, _real("scan-plan"))


def test_a_usable_scan_plan_gives_the_view_its_fields_in_visiting_order(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    result = _scan_plan_result()
    assert [f.well for f in result.fields[:4]] == ["B11"] * 4
    assert [f.well for f in result.fields[4:8]] == ["B2"] * 4
    _seed(run_doc, "scan-plan", "scan-plan", result)
    plan = _plan_page(fake_microscope, run_doc).view().plan
    assert plan is not None
    assert (plan.count, plan.wells, plan.fov_um) == (12, 3, (512.0, 512.0))
    # The plan's order, B11's fields before B2's before A1's: not the wells' order on the plate.
    assert plan.fields == tuple((f.x_um, f.y_um) for f in result.fields)


def test_no_scan_plan_gives_no_layer(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    assert _plan_page(fake_microscope, run_doc).view().plan is None


def test_a_stale_scan_plan_gives_no_layer_and_keeps_its_card_marked_stale(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    result = _scan_plan_result()
    # The plate was registered again (a2) after the plan read its attempt a1.
    _seed(run_doc, "plate-registration", "plate", _plate(), minute=2)
    _seed(
        run_doc,
        "scan-plan",
        "scan-plan",
        result,
        minute=3,
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a1")},
    )
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is None
    scan_plan = {i.slot: i for i in view.checklist}["scan-plan"]
    assert (scan_plan.state, scan_plan.reason) == (
        "unusable",
        "scan-plan read its inputs from an earlier run; run it again",
    )
    # Marked, not hidden: the card is of the inputs the plan read.
    card = view.cards["scan-plan"]
    assert card.stale is True
    assert card.lines == scan_plan_card_lines(result)
    assert view.problems == ()


@pytest.mark.parametrize(
    ("profile", "dry_run"),
    [("another-stand", False), ("demo", True)],
    ids=["another-profile", "dry-run-on-a-live-page"],
)
def test_a_scan_plan_of_another_profile_or_a_dry_run_on_a_live_page_gives_no_layer(
    fake_microscope: Microscope, run_doc: Path, profile: str, dry_run: bool
) -> None:
    # The density layer's rule: a result the document calls unusable here is not drawn.
    _seed(
        run_doc,
        "scan-plan",
        "scan-plan",
        _scan_plan_result(),
        profile=profile,
        dry_run=dry_run,
    )
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is None
    assert {i.slot: i.state for i in view.checklist}["scan-plan"] == "unusable"
    assert view.cards["scan-plan"].lines[0] == "wells: 3 (B11, B2, A1)"


def test_a_plan_over_the_bound_gives_a_layer_with_no_fields_and_the_count(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # One under, on and one over the bound: it is the boundary the test is about.
    for count, drawn in ((4999, True), (5000, True), (5001, False)):
        _seed(
            run_doc,
            "scan-plan",
            "scan-plan",
            _with_fields(_scan_plan_result(), count),
        )
        plan = _plan_page(fake_microscope, run_doc).view().plan
        assert plan is not None
        assert plan.count == count
        assert (len(plan.fields) == count) is drawn
        assert (plan.fields == ()) is (not drawn)
        assert plan.wells == 3


@pytest.mark.parametrize("fov_um", [(0.0, 512.0), (-512.0, 512.0), (512.0, 0.0)])
def test_a_plan_whose_field_of_view_is_not_positive_is_a_problem_not_a_layer(
    fake_microscope: Microscope, run_doc: Path, fov_um: tuple[float, float]
) -> None:
    result = _scan_plan_result()
    # The model does not bound fov_um (measured, #320): this result is valid.
    _seed(
        run_doc,
        "scan-plan",
        "scan-plan",
        result.model_copy(update={"fov_um": fov_um}),
    )
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is None
    assert view.problems == (
        f"scan-plan: fov_um {fov_um} is not positive; its fields are not drawn",
    )
    # The card still says what the plan holds.
    assert view.cards["scan-plan"].lines[0] == "wells: 3 (B11, B2, A1)"


def test_a_scan_plan_that_does_not_validate_is_a_problem_and_the_generic_card(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-125: a hand-edited document is data; view() raises nothing for it.
    broken = _scan_plan_result().model_dump(mode="json")
    del broken["estimate_s"]
    _record(run_doc, "scan-plan", result=broken)
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.plan is None
    assert view.problems == (
        "scan-plan: result does not validate: estimate_s: Field required",
    )
    # The generic card: the first keys as counted values, not the scan plan's lines.
    lines = view.cards["scan-plan"].lines
    assert "wells: 3 items" in lines
    assert not any(line.startswith("estimate: ") for line in lines)


def test_the_card_of_a_done_scan_plan_holds_the_estimate_line_the_generic_card_cut(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    result = _scan_plan_result()
    _seed(run_doc, "scan-plan", "scan-plan", result)
    card = _plan_page(fake_microscope, run_doc).view().cards["scan-plan"]
    # Four lines, none of them cut: `estimate_s` was the 14th key of the generic card.
    assert card.lines == scan_plan_card_lines(result)
    assert card.lines[3] == (
        "estimate: about 42 s (the moves, refocus and snaps of the timings)"
    )
    assert card.stale is False


@pytest.mark.parametrize("status", [StageStatus.stopped, StageStatus.failed])
def test_a_scan_plan_that_did_not_end_done_has_no_lines(
    fake_microscope: Microscope, run_doc: Path, status: StageStatus
) -> None:
    # FM-154: a stopped or failed record's values are not a result, even when it holds one.
    _seed(run_doc, "scan-plan", "scan-plan", _scan_plan_result(), status=status)
    view = _plan_page(fake_microscope, run_doc).view()
    assert view.cards["scan-plan"].lines == ()
    assert view.plan is None


def test_a_record_that_is_not_a_scan_plan_keeps_its_generic_card(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The same keys under another `provides`: the tool is not scan-plan, so the card is generic.
    result = _scan_plan_result().model_dump(mode="json")
    _record(run_doc, "stub", result=result)
    lines = _page(fake_microscope, run_doc).view().cards["stub"].lines
    assert lines[0] == "plate_plan: 5 entries"
    assert "wells: 3 items" in lines
    assert not any(line.startswith("estimate: ") for line in lines)


def test_the_card_of_a_clipped_plan_says_why_the_two_counts_differ(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The plan's own measure (coverage 1.0 on B2, B11): 344 fields of 14 x 14 x 2 = 392.
    # The clipped form of the line is 90 characters, longer than a value's 80: the
    # sentence that says why the counts differ is the part a cut would drop (§14.3).
    base = _scan_plan_result(("B11", "B2"))
    clipped = base.model_copy(
        update={
            "rows": 14,
            "columns": 14,
            "fields": tuple(base.fields[i % len(base.fields)] for i in range(344)),
        }
    )
    _seed(run_doc, "scan-plan", "scan-plan", clipped)
    card = _plan_page(fake_microscope, run_doc).view().cards["scan-plan"]
    assert card.lines[1] == (
        "fields: 344 of 392, a grid of 14 × 14 in each well, "  # noqa: RUF001
        "the fields that miss the well left out"
    )
    assert len(card.lines[1]) == 90


def test_the_cards_scan_plan_lines_are_one_line_each_and_bounded(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-139: the objective is the stand's label and may hold anything. It is the only
    # free text of the four lines, and it sits before the overlap, which a cut drops.
    hostile = _scan_plan_result().model_copy(
        update={"objective": "10X\nPlan Apo" + "x" * 700}
    )
    _seed(run_doc, "scan-plan", "scan-plan", hostile)
    lines = _plan_page(fake_microscope, run_doc).view().cards["scan-plan"].lines
    assert len(lines) == 4
    assert all("\n" not in line and "\r" not in line for line in lines)
    # Cut at the limit of a diagnosis (600) and a `…`: bounded, whatever the label holds.
    assert all(len(line) <= 601 for line in lines)
    assert lines[2].startswith("field: 512 × 512 µm at 10X Plan Apoxxx")  # noqa: RUF001
    assert lines[2].endswith("x…")
    assert len(lines[2]) == 601
