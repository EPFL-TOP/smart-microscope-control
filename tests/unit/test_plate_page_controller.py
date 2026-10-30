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
from pathlib import Path
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
from smc.plugins.slots import (
    DensityResult,
    DensityWell,
    PlateResult,
    RouteResult,
    WallTouch,
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
    _autofocus_words,
)
from smc.ui.frames import Frame

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
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
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
) -> None:
    """A record as an earlier run of ``plugin`` left it; ``minute`` orders the starts."""
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
    assert clicked.current == "A1"


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
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    page = _page(fake_microscope, run_doc)
    x, y = _centre("B3", a1=A1_OFF, rotation_deg=TURN_DEG)
    assert page.click(x, y, drive=True, side="left") == "B3"
    assert page.join(10.0)
    assert StubGoto.ran == [("B3", "left")]
    assert page.current == "B3"
    assert page.selection == ()
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("goto-well done in ")


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


@pytest.mark.parametrize(
    ("engaged", "locked", "words"),
    [
        (None, None, "autofocus ?"),
        (None, True, "autofocus ?"),
        (True, None, "autofocus ?"),
        (False, None, "autofocus off"),
        (False, True, "autofocus off"),
        (True, False, "autofocus on, not locked"),
        (True, True, "autofocus locked"),
    ],
)
def test_autofocus_words(engaged: bool | None, locked: bool | None, words: str) -> None:
    assert _autofocus_words(engaged, locked) == words


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
    page: PlatePage, fake_microscope: Microscope
) -> None:
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
        well="A1", side="left", x_um=3.0, y_um=4.0
    )
    assert page.walls == (
        WallTouch(well="A1", side="top", x_um=1.0, y_um=2.0),
        WallTouch(well="A1", side="left", x_um=3.0, y_um=4.0),
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


def test_forget_drops_one_wells_captures_or_all(
    page: PlatePage, fake_microscope: Microscope
) -> None:
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
    walls = [{"well": "B2", "side": "left", "x_um": 5.0, "y_um": 6.0}]
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
    page: PlatePage, fake_microscope: Microscope
) -> None:
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


def test_plate_registration_params_are_the_plate_and_the_captures(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.require(XYStage).move_to_um(1.5, 2.5)
    page.capture_centre("A1")
    page.capture_wall("A1", "right")
    refs = json.dumps([{"well": "A1", "x_um": 1.5, "y_um": 2.5}])
    walls = json.dumps([{"well": "A1", "side": "right", "x_um": 1.5, "y_um": 2.5}])
    assert page.page_params("plate-registration") == [
        "plate=96-well",
        f"refs={refs}",
        f"walls={walls}",
    ]


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


class SlowCamera(FrameCamera):
    """A camera whose snap takes ``snap_s``; ``began`` is when each snap started (FM-157)."""

    def __init__(self, frame: np.ndarray, *, snap_s: float) -> None:
        super().__init__(frame)
        self._snap_s = snap_s
        self.began: list[float] = []

    def snap(self) -> np.ndarray:
        self.began.append(time.monotonic())
        time.sleep(self._snap_s)
        return super().snap()


def test_a_camera_as_slow_as_the_period_runs_at_its_own_rate_and_not_slower(
    page: PlatePage, fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cap is a period: the wait is what is left of it since the frame began, so a
    # frame that takes a whole period is followed by the next at once. A wait of the
    # whole period after each frame would give 0.4 s between two snaps, not 0.2 s.
    monkeypatch.setattr(controller_module, "LIVE_PERIOD_S", 0.2)
    camera = SlowCamera(_ramp_frame(), snap_s=0.2)
    fake_microscope.override(Camera, camera)
    page.live(True)
    try:
        _wait_for_frames(page, 6, timeout_s=10.0)
    finally:
        page.live(False)
    assert page.join(10.0)
    gaps = len(camera.began) - 1
    mean_gap_s = (camera.began[-1] - camera.began[0]) / gaps
    assert 0.18 <= mean_gap_s < 0.3, f"{gaps} gaps of {mean_gap_s:.3f} s on average"


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
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
