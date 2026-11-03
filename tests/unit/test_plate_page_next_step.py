"""The plate page's next step, and what the profile suggests, on stub plugins and a fake stand (m5 §8.3, #293).

No Panel here. The stand is ``fake_microscope``, handed to the page by an opener, and
the tools are stubs that carry the names and slots of the workflow's, so the step's
walk over the checklist is tested without a scan. The profile is built in code with a
``[sample]`` and a ``[suggest]``, as a ``profiles/*.toml`` would hold them. Texts are
asserted whole (FM-45), and the pairs a step shows are read back through
``parse_params`` and ``shlex``, the way the page's button and a shell read them.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.params import parse_params
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult, WallTouch, WellRef
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.testing.sample_stand import sample_refs
from smc.ui.controller import NOT_CONNECTED, PlatePage
from smc.ui.guide import CAPTURE_WELLS, DRY_RUN_NO_FRAME, SELECT_WELLS

T0 = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)
STAND = "demo-sample"


# --- stub plugins: the workflow's names and slots, with the parameters that matter ---


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class RegistrationParams(PluginParams):
    plate: str = "96-well"
    refs: tuple[WellRef, ...] = ()
    walls: tuple[WallTouch, ...] = ()


class ScanPlanParams(PluginParams):
    wells: str
    coverage: float = 1.0
    overlap: float = 0.1


class ScanParams(PluginParams):
    autofocus: bool = False
    retries: int = 3
    label: str


class CellsParams(PluginParams):
    detector: Literal["model", "blobs"] = "model"
    score_threshold: float = 0.5


class Tool:
    """Returns an empty result; ``seen`` keeps the parameters of every run, by plugin."""

    name = provides = "tool"
    summary = "A test tool."
    Params: type[PluginParams] = NoParams
    Result: type[PluginResult] = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[tuple[str, PluginParams]]] = []

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PluginResult:
        Tool.seen.append((self.name, params))
        return self.Result()


class Registration(Tool):
    name = "plate-registration"
    provides = "plate"
    summary = "A test registration."
    Params = RegistrationParams
    Result = PlateResult
    requires = (XYStage,)

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PlateResult:
        Tool.seen.append((self.name, params))
        return PlateResult(
            plate=params.plate,
            a1_center_xy_um=(0.0, 0.0),
            rotation_deg=0.0,
            method="seed",
            residual_um=None,
            scale_error=None,
            n_refs=len(params.refs),
        )


class Finding(Tool):
    name = "plate-finding"
    provides = "plate"
    # Two lines, as a third party's summary may be: a step's line holds one.
    summary = "A test finder.\nIt surveys the plate."


class Calibration(Tool):
    name = provides = "camera-calibration"


class ScanPlan(Tool):
    name = provides = "scan-plan"
    Params = ScanPlanParams
    reads = ("plate",)


class CameraScanPlan(ScanPlan):
    """The scan plan as the real one is: it requires the camera, for the pixel size."""

    requires = (Camera,)


class Scan(Tool):
    name = provides = "scan"
    Params = ScanParams
    reads = ("scan-plan",)


class Cells(Tool):
    name = provides = "cells"
    Params = CellsParams
    reads = ("scan",)


class Density(Tool):
    name = provides = "density"
    reads = ("cells",)


class Route(Tool):
    name = provides = "route"
    reads = ("density",)


class Report(Tool):
    name = provides = "report"


class NisExport(Tool):
    name = provides = "nis-export"


WORKFLOW = (
    Registration,
    ScanPlan,
    Scan,
    Cells,
    Density,
    Route,
    Report,
    NisExport,
)


def _ep(cls: type[Tool]) -> EntryPoint:
    return EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    Tool.seen = []


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _profile(
    *,
    sample: dict[str, Any] | None = None,
    suggest: dict[str, Any] | None = None,
) -> Profile:
    """A profile named ``demo-sample``, the way a ``profiles/*.toml`` would read."""
    data: dict[str, Any] = {"microscope": {"name": STAND}}
    if sample is not None:
        data["sample"] = sample
    if suggest is not None:
        data["suggest"] = suggest
    return Profile.model_validate(data)


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    """Builds a closed page over a fake stand; every page and stand it built is closed afterwards.

    The stand carries the page's profile name, as the one ``Microscope.open`` opens
    does: a record is made on the *stand's* profile, and the checklist reads it as
    the *page's*.
    """
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
        tools: tuple[type[Tool], ...] = WORKFLOW,
        dry_run: bool = False,
        connect: bool = True,
    ) -> PlatePage:
        page = PlatePage.closed(
            profile or _profile(),
            run_doc,
            dry_run=dry_run,
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


def _seed(
    run_doc: Path,
    plugin: str,
    provides: str | None = None,
    *,
    status: StageStatus = StageStatus.done,
    minute: int = 0,
    result: dict[str, Any] | None = None,
    reads: dict[str, ReadRecord | None] | None = None,
    checkpoint: dict[str, Any] | None = None,
    profile: str = STAND,
    dry_run: bool = False,
    params: dict[str, Any] | None = None,
) -> None:
    """A record an earlier run of ``plugin`` left; ``minute`` orders the starts and names the attempt."""
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=provides or plugin,
            attempt=f"a{minute}",
            reads=reads or {},
            status=status,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params=params or {},
            result=result,
            checkpoint=checkpoint,
        ),
    )


def _plate_done(
    run_doc: Path,
    *,
    minute: int = 0,
    plate: str = "96-well",
    profile: str = STAND,
    params: dict[str, Any] | None = None,
    status: StageStatus = StageStatus.done,
) -> None:
    """The ``plate`` slot filled, as a registration of an earlier session left it."""
    result = PlateResult(
        plate=plate,
        a1_center_xy_um=(-49500.0, 31500.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=1,
    ).model_dump(mode="json")
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        minute=minute,
        result=result if status is StageStatus.done else None,
        profile=profile,
        params=params,
        status=status,
    )


def _chain_to_scan(run_doc: Path, *, scan_dry_run: bool | None = None) -> None:
    """The plate and the scan plan done, and a ``scan`` too when ``scan_dry_run`` is not ``None``.

    Every record read its upstream's current attempt, so none is stale. The scan
    plan was a dry run when the scan was: a dry run reads what a dry run made.
    """
    _plate_done(run_doc)
    _seed(
        run_doc,
        "scan-plan",
        minute=1,
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
        dry_run=bool(scan_dry_run),
    )
    if scan_dry_run is not None:
        _seed(
            run_doc,
            "scan",
            minute=2,
            reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
            dry_run=scan_dry_run,
        )


def _pairs_of(command: str) -> list[str]:
    """The ``--param`` values of a command line, as a shell hands them to ``smc run``."""
    words = shlex.split(command)
    return [words[i + 1] for i, word in enumerate(words) if word == "--param"]


# --- the walk over the checklist -----------------------------------------------------


def test_the_next_step_is_the_first_required_slot_not_done(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    fresh = make_page().view().next_step
    assert fresh is not None
    assert fresh.plugin == "plate-registration"
    assert fresh.why[0] == "the run document has no plate yet"

    _plate_done(run_doc)
    after = make_page().view().next_step
    assert after is not None
    assert after.plugin == "scan-plan"
    assert after.why[0] == "the run document has no scan-plan yet"


def test_optional_slots_are_never_the_next_step(
    make_page: Callable[..., PlatePage],
) -> None:
    # `camera-calibration` has a registered tool and comes first in the checklist:
    # were it a step, the walk would start there.
    page = make_page(tools=(Calibration, *WORKFLOW))
    assert page.view().checklist[0].slot == "camera-calibration"
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"


@pytest.mark.parametrize("how", ["failed", "stale"])
def test_an_unusable_slot_is_the_next_step_with_the_run_documents_reason(
    make_page: Callable[..., PlatePage], run_doc: Path, how: str
) -> None:
    if how == "failed":
        _seed(run_doc, "plate-registration", "plate", status=StageStatus.failed)
        plugin = "plate-registration"
        reason = "the last run of plate-registration ended failed; run it again"
    else:
        # The scan plan read the plate's first attempt, and the plate was made again.
        _seed(
            run_doc,
            "scan-plan",
            minute=1,
            reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
        )
        _plate_done(run_doc, minute=2)
        plugin = "scan-plan"
        reason = "scan-plan read its inputs from an earlier run; run it again"
    step = make_page().view().next_step
    assert step is not None
    assert step.plugin == plugin
    assert step.why[0] == reason


def test_a_slot_without_a_registered_tool_has_no_tool(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc)
    page = make_page(tools=tuple(t for t in WORKFLOW if t is not ScanPlan))
    step = page.view().next_step
    assert step is not None
    assert step.plugin is None
    assert step.params == ()
    assert step.command == ""
    assert step.blocked is None
    assert step.why == (
        "no registered plugin provides scan-plan; smc run --list shows the plugins",
    )


# --- the plate step: captures, the sample, or the operator ---------------------------


def test_the_plate_step_registers_from_the_sample_when_nothing_is_captured(
    make_page: Callable[..., PlatePage],
) -> None:
    profile = _profile(
        sample={"a1_center_xy_um": [-40000.0, 25000.0], "rotation_deg": 3.0}
    )
    assert profile.sample is not None
    step = make_page(profile).view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"
    assert step.blocked is None
    parsed = parse_params(RegistrationParams, step.params)
    assert parsed.plate == "96-well"
    assert parsed.refs == sample_refs(profile.sample)
    assert parsed.walls == ()
    assert [ref.well for ref in parsed.refs] == ["A1", "A12", "H1"]
    assert step.why == (
        "the run document has no plate yet",
        "refs: the centres of A1, A12, H1 where the profile's [sample] puts the "
        "plate; this stand's plate is synthetic, so they are known, not measured",
    )


# --- the plate card says where its references came from (m5 §8.8.4) -------------------

#: The step's own line, whole: the card says it too, so that one text says it twice.
SAMPLE_REFS_LINE = (
    "refs: the centres of A1, A12, H1 where the profile's [sample] puts the plate; "
    "this stand's plate is synthetic, so they are known, not measured"
)


def _the_samples_refs(profile: Profile) -> list[dict[str, Any]]:
    """``params["refs"]`` of a registration made from the step: the sample's centres, as JSON."""
    assert profile.sample is not None
    return [ref.model_dump(mode="json") for ref in sample_refs(profile.sample)]


def test_the_plate_card_says_the_refs_are_the_samples(
    make_page: Callable[..., PlatePage],
) -> None:
    page = make_page(_profile(sample={"rotation_deg": 3.0}))
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"
    page.run(step.plugin, list(step.params))
    assert page.join(10.0)

    card = page.view().cards["plate-registration"]
    assert card.status == "done"
    assert card.assumed[-1] == SAMPLE_REFS_LINE
    # The step said it before the click, with the same text.
    assert step.why[1] == SAMPLE_REFS_LINE


def test_the_plate_card_keeps_what_the_result_assumed_and_adds_the_source_last(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    profile = _profile(sample={})
    result = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="references",
        residual_um=0.0,
        scale_error=None,
        n_refs=3,
        assumed=("pitch taken from the plate type",),
    ).model_dump(mode="json")
    _seed(
        run_doc,
        "plate-registration",
        "plate",
        result=result,
        params={"refs": _the_samples_refs(profile)},
    )
    card = make_page(profile).view().cards["plate-registration"]
    assert card.assumed == ("pitch taken from the plate type", SAMPLE_REFS_LINE)


def test_a_ref_within_the_tolerance_is_still_the_samples(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    profile = _profile(sample={})
    refs = _the_samples_refs(profile)
    refs[1]["x_um"] += 0.005  # the tolerance is 0.01 µm: a rounded JSON is the same ref
    _plate_done(run_doc, params={"refs": refs})
    card = make_page(profile).view().cards["plate-registration"]
    assert card.assumed == (SAMPLE_REFS_LINE,)


def _moved(refs: list[dict[str, Any]], by_um: float) -> list[dict[str, Any]]:
    refs[1]["y_um"] += by_um
    return refs


def _moved_x(refs: list[dict[str, Any]], by_um: float) -> list[dict[str, Any]]:
    # Each axis is compared on its own: a ref moved along x only, with y as it was.
    refs[1]["x_um"] += by_um
    return refs


def _without_a_key(refs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    del refs[0][key]
    return refs


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda refs: _moved(refs, 100.0), id="moved-100-um"),
        pytest.param(lambda refs: _moved(refs, 0.05), id="moved-past-the-tolerance"),
        pytest.param(lambda refs: _moved_x(refs, 100.0), id="x-moved-100-um"),
        pytest.param(lambda refs: _moved_x(refs, 0.05), id="x-moved-past-tolerance"),
        pytest.param(lambda refs: refs[:2], id="one-ref-missing"),
        pytest.param(lambda refs: [*refs, refs[0]], id="one-ref-more"),
        pytest.param(lambda refs: [refs[0], refs[0], refs[1]], id="a-well-twice"),
        pytest.param(lambda refs: [*refs[:2], {**refs[2], "well": "H2"}], id="a-well"),
        pytest.param(lambda refs: [*refs[:2], 7], id="an-item-that-is-a-number"),
        pytest.param(lambda refs: [*refs[:2], "H1"], id="an-item-that-is-a-text"),
        pytest.param(lambda refs: _without_a_key(refs, "x_um"), id="no-x"),
        pytest.param(lambda refs: _without_a_key(refs, "well"), id="no-well"),
        pytest.param(lambda refs: [{**refs[0], "y_um": "0"}, *refs[1:]], id="y-text"),
        pytest.param(lambda refs: [{**refs[0], "y_um": False}, *refs[1:]], id="y-bool"),
        pytest.param(lambda refs: [{**refs[0], "well": 1}, *refs[1:]], id="well-int"),
        # Unhashable: used as a dict key, they would raise and `view()` must not.
        pytest.param(
            lambda refs: [{**refs[0], "well": ["A1"]}, *refs[1:]], id="well-list"
        ),
        pytest.param(
            lambda refs: [{**refs[0], "well": {"A1": 1}}, *refs[1:]], id="well-obj"
        ),
        pytest.param(lambda refs: json.dumps(refs), id="a-text"),
        pytest.param(lambda refs: {"A1": refs[0]}, id="an-object"),
        pytest.param(lambda refs: None, id="none"),
        pytest.param(lambda refs: [], id="empty"),
    ],
)
def test_the_plate_card_says_nothing_of_refs_that_are_not_the_samples(
    make_page: Callable[..., PlatePage],
    run_doc: Path,
    tamper: Callable[[list[dict[str, Any]]], object],
) -> None:
    # A1 at the origin: a flag in place of a 0 would equal it, were it taken for a number.
    profile = _profile(sample={"a1_center_xy_um": [0.0, 0.0]})
    _plate_done(run_doc, params={"refs": tamper(_the_samples_refs(profile))})
    view = make_page(profile).view()  # never an exception: the record is read as it is
    assert view.cards["plate-registration"].assumed == ()


def test_the_plate_card_says_nothing_of_a_record_with_no_refs_key(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc, params={})
    card = make_page(_profile(sample={})).view().cards["plate-registration"]
    assert card.assumed == ()


def test_the_plate_card_says_nothing_of_a_record_made_on_another_stand(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    profile = _profile(sample={})
    _plate_done(run_doc, profile="demo", params={"refs": _the_samples_refs(profile)})
    card = make_page(profile).view().cards["plate-registration"]
    assert card.assumed == ()
    assert card.profile == "demo"


def test_the_plate_card_says_nothing_of_a_registration_that_did_not_finish(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    profile = _profile(sample={})
    _plate_done(
        run_doc,
        params={"refs": _the_samples_refs(profile)},
        status=StageStatus.failed,
    )
    card = make_page(profile).view().cards["plate-registration"]
    assert card.status == "failed"
    assert card.assumed == ()


def test_the_plate_card_says_nothing_on_a_profile_with_no_sample(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The refs are the sample's, but this profile has none: captured by hand, they
    # are measured, and a stand that is not the demo's never claims otherwise.
    _plate_done(run_doc, params={"refs": _the_samples_refs(_profile(sample={}))})
    card = make_page(_profile()).view().cards["plate-registration"]
    assert card.assumed == ()


def test_captures_win_over_the_sample(make_page: Callable[..., PlatePage]) -> None:
    page = make_page(_profile(sample={}))
    page.refs = (WellRef(well="B2", x_um=-31500.0, y_um=22500.0),)
    page.walls = (
        WallTouch(well="B2", side="left", x_um=-34000.0, y_um=22500.0),
        WallTouch(well="B2", side="top", x_um=-31500.0, y_um=25000.0),
    )
    step = page.view().next_step
    assert step is not None
    assert list(step.params) == page.page_params("plate-registration")
    parsed = parse_params(RegistrationParams, step.params)
    assert [ref.well for ref in parsed.refs] == ["B2"]
    assert len(parsed.walls) == 2
    assert step.why == (
        "the run document has no plate yet",
        "plate, refs, walls: the 1 centres and 2 walls captured on this page",
    )
    assert step.blocked is None


def test_walls_alone_are_captures_too(make_page: Callable[..., PlatePage]) -> None:
    page = make_page(_profile(sample={}))
    page.walls = (WallTouch(well="B2", side="left", x_um=1.0, y_um=2.0),)
    step = page.view().next_step
    assert step is not None
    assert step.why[1] == (
        "plate, refs, walls: the 0 centres and 1 walls captured on this page"
    )


def test_without_sample_or_captures_the_plate_step_says_how_to_capture(
    make_page: Callable[..., PlatePage],
) -> None:
    step = make_page(_profile()).view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"
    assert step.blocked == CAPTURE_WELLS
    # Nothing is invented: no references, so none are offered.
    assert step.params == ()


def test_a_sample_of_another_plate_asks_to_choose_it(
    make_page: Callable[..., PlatePage],
) -> None:
    page = make_page(_profile(sample={"plate": "384-well"}))
    step = page.view().next_step
    assert step is not None
    assert step.blocked == (
        "the profile's sample is a 384-well plate: choose 384-well under "
        "plate-registration"
    )
    assert step.params == ()

    page.choose_plate("384-well")
    chosen = page.view().next_step
    assert chosen is not None
    assert chosen.blocked is None
    parsed = parse_params(RegistrationParams, chosen.params)
    assert parsed.plate == "384-well"
    assert [ref.well for ref in parsed.refs] == ["A1", "A24", "P1"]


def test_the_step_names_the_other_tools_of_its_slot(
    make_page: Callable[..., PlatePage],
) -> None:
    step = make_page(tools=(*WORKFLOW, Finding)).view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"
    assert step.why[-1] == "or plate-finding: A test finder. It surveys the plate."
    # Only the plate has another way: the scan plan has none.
    assert not any(line.startswith("or ") for line in make_page().view().next_step.why)  # type: ignore[union-attr]


# --- the scan plan waits for wells ------------------------------------------------------


def test_the_scan_plan_step_waits_for_wells_then_carries_them(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc)
    page = make_page()
    waiting = page.view().next_step
    assert waiting is not None
    assert waiting.plugin == "scan-plan"
    assert waiting.blocked == SELECT_WELLS
    assert waiting.params == ()

    page.select_text("B2,B3")
    step = page.view().next_step
    assert step is not None
    assert step.blocked is None
    assert step.params == ("wells=B2,B3",)
    assert step.why == (
        "the run document has no scan-plan yet",
        "wells: the 2 wells selected on the map",
    )


def test_a_selection_on_another_plate_than_the_registered_one_blocks_the_step_with_why(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc, plate="96-well")
    page = make_page()
    page.choose_plate("384-well")
    page.selection = ("A1",)
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "scan-plan"
    assert step.blocked == (
        "the registered plate is a 96-well; calibrate the 384-well plate again, "
        "or choose 96-well"
    )


# --- a dry-run page does not lead to a dead end (m5 §8.8.1) -----------------------------


def test_a_dry_run_page_stops_the_step_at_scan_and_says_why(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _chain_to_scan(run_doc)
    profile = _profile(suggest={"scan": {"params": ["label=t1"], "why": "a name"}})
    step = make_page(profile, dry_run=True).view().next_step
    assert step is not None
    assert step.plugin == "scan"
    assert step.blocked == DRY_RUN_NO_FRAME
    # The step stops advising; the call stays as the page makes it.
    assert step.params == ("label=t1",)
    words = shlex.split(step.command)
    assert words[:3] == ["smc", "run", "scan"]
    assert "--dry-run" in words
    assert _pairs_of(step.command) == ["label=t1"]


def test_a_dry_run_page_stops_the_step_at_a_tool_that_reads_a_dry_run_scan(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _chain_to_scan(run_doc, scan_dry_run=True)
    page = make_page(dry_run=True)
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "cells"
    assert step.blocked == DRY_RUN_NO_FRAME

    # The walk's loop: `cells` ran, failed on the missing frame, and the step is
    # `cells` again. It says why instead of advising a run that cannot work.
    _seed(
        run_doc,
        "cells",
        minute=3,
        status=StageStatus.failed,
        reads={"scan": ReadRecord(plugin="scan", attempt="a2")},
        dry_run=True,
    )
    again = page.view().next_step
    assert again is not None
    assert again.plugin == "cells"
    assert again.why[0] == "the last run of cells ended failed; run it again"
    assert again.blocked == DRY_RUN_NO_FRAME


def test_a_dry_run_page_that_reads_a_real_scan_is_not_stopped(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # A dry run may read a real result: the scan was made in a terminal.
    _chain_to_scan(run_doc, scan_dry_run=False)
    step = make_page(dry_run=True).view().next_step
    assert step is not None
    assert step.plugin == "cells"
    assert step.blocked is None


def test_a_page_that_is_not_a_dry_run_is_never_stopped_at_scan(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # The newcomer's path: the page served without --dry-run on a simulated stand.
    _chain_to_scan(run_doc)
    at_scan = make_page().view().next_step
    assert at_scan is not None
    assert at_scan.plugin == "scan"
    assert at_scan.blocked is None

    _seed(
        run_doc,
        "scan",
        minute=2,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )
    at_cells = make_page().view().next_step
    assert at_cells is not None
    assert at_cells.plugin == "cells"
    assert at_cells.blocked is None


def test_a_dry_run_page_does_not_stop_a_tool_that_merely_requires_the_camera(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # `scan-plan` requires a camera (for the pixel size) and computes the same in a
    # dry run: the rule is by the plugin's name, not by a capability (§8.8.1).
    _plate_done(run_doc)
    page = make_page(tools=(Registration, CameraScanPlan, Scan, Cells), dry_run=True)
    page.select_text("B2")
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "scan-plan"
    assert step.blocked is None


def test_a_dry_run_page_does_not_stop_a_tool_that_does_not_read_a_dry_run_scan(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # A dry-run `scan` is in the run document, and the plate was registered again after
    # it: the step is `scan-plan`, which reads the plate and not the scan. A rule that
    # stopped every tool while a dry-run scan is there would stop it (§8.8.1).
    _chain_to_scan(run_doc, scan_dry_run=True)
    _plate_done(run_doc, minute=10)
    page = make_page(dry_run=True)
    page.select_text("B2")
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "scan-plan"
    assert step.blocked is None


def test_a_dry_run_page_with_a_scan_record_and_no_scan_plugin_raises_nothing(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # A `scan` record is in the run document and no plugin named `scan` is registered
    # (it failed to load, or the document came from another install): there is no card
    # for it, and `view()` raises nothing (FM-125). `cells` reads a scan it cannot show
    # was a dry run, so it is not stopped.
    _chain_to_scan(run_doc, scan_dry_run=True)
    page = make_page(tools=(Registration, ScanPlan, Cells), dry_run=True)
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "cells"
    assert step.blocked is None


def test_the_stand_not_connected_wins_over_the_dry_run_note(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _chain_to_scan(run_doc)
    step = make_page(dry_run=True, connect=False).view().next_step
    assert step is not None
    assert step.plugin == "scan"
    assert step.blocked == NOT_CONNECTED


# --- what the profile suggests ----------------------------------------------------------


def test_the_profiles_suggestion_joins_the_step_with_its_why(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc)
    profile = _profile(
        suggest={
            "scan-plan": {
                "params": ["coverage=0.4", "overlap=0.2"],
                "why": "a few wells take seconds",
            }
        }
    )
    page = make_page(profile)
    page.select_text("B2")
    step = page.view().next_step
    assert step is not None
    assert step.params == ("wells=B2", "coverage=0.4", "overlap=0.2")
    assert step.why == (
        "the run document has no scan-plan yet",
        "wells: the 1 wells selected on the map",
        "coverage=0.4 overlap=0.2: suggested by the profile demo-sample: a few "
        "wells take seconds",
    )
    # What the step shows is what the command passes.
    assert _pairs_of(step.command) == list(step.params)


def test_a_suggestion_alone_is_the_steps_pairs_for_a_tool_the_page_fills_nothing_for(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc)
    _seed(
        run_doc,
        "scan-plan",
        minute=1,
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
    )
    _seed(
        run_doc,
        "scan",
        minute=2,
        reads={"scan-plan": ReadRecord(plugin="scan-plan", attempt="a1")},
    )
    profile = _profile(
        suggest={"cells": {"params": ["detector=blobs"], "why": "spots"}}
    )
    step = make_page(profile).view().next_step
    assert step is not None
    assert step.plugin == "cells"
    assert step.params == ("detector=blobs",)
    assert step.why == (
        "the run document has no cells yet",
        "detector=blobs: suggested by the profile demo-sample: spots",
    )


def test_a_suggestion_for_an_unknown_plugin_or_key_is_a_problem_and_ignored(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc)
    profile = _profile(
        suggest={
            "ghost": {"params": ["a=1"], "why": "x"},
            "scan-plan": {"params": ["coverage=0.4", "nope=1"], "why": "y"},
        }
    )
    page = make_page(profile)
    page.select_text("B2")
    view = page.view()
    assert "profile [suggest.ghost]: no plugin named ghost" in view.problems
    assert (
        "profile [suggest.scan-plan]: nope is not a parameter of scan-plan "
        "(keys: wells, coverage, overlap)"
    ) in view.problems
    # The entry is ignored whole: not even its good key is offered.
    assert view.next_step is not None
    assert view.next_step.params == ("wells=B2",)
    (scan_plan,) = [t for t in page.tools() if t.name == "scan-plan"]
    assert scan_plan.suggested == ()
    assert scan_plan.suggested_why is None


def test_a_suggestion_of_a_page_key_is_a_problem_and_ignored(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    profile = _profile(
        sample={},
        suggest={
            "scan-plan": {"params": ["wells=A1"], "why": "x"},
            "plate-registration": {"params": ["refs=[]"], "why": "y"},
        },
    )
    page = make_page(profile)
    view = page.view()
    assert "profile [suggest.scan-plan]: wells is filled by the page" in view.problems
    assert (
        "profile [suggest.plate-registration]: refs is filled by the page"
        in view.problems
    )
    assert view.next_step is not None
    assert all("suggested by" not in line for line in view.next_step.why)
    assert [p for p in page.tools() if p.suggested] == []


def test_a_profile_without_a_suggestion_has_no_problem_and_no_suggested_field(
    make_page: Callable[..., PlatePage],
) -> None:
    page = make_page(_profile())
    assert page.view().problems == ()
    assert all(tool.suggested == () for tool in page.tools())


def test_a_suggested_key_is_a_key_row_with_its_value(
    make_page: Callable[..., PlatePage],
) -> None:
    plain = {t.name: t for t in make_page().tools()}
    # `retries` is no line of KEY_PARAMS: it is folded under *Advanced*...
    assert [row[0] for row in plain["scan"].key] == ["autofocus", "label"]
    assert "retries" in [row[0] for row in plain["scan"].advanced]

    profile = _profile(
        suggest={"scan": {"params": ["retries=5"], "why": "a flaky stage"}}
    )
    suggested = {t.name: t for t in make_page(profile).tools()}
    scan = suggested["scan"]
    # ...unless the stand suggests it: after KEY_PARAMS, before the required fields.
    assert [row[0] for row in scan.key] == ["autofocus", "retries", "label"]
    assert "retries" not in [row[0] for row in scan.advanced]
    assert scan.suggested == (("retries", "5"),)
    assert scan.suggested_why == "a flaky stage"
    # No other tool changed.
    assert suggested["cells"] == plain["cells"]


def test_a_suggestion_of_a_key_that_is_already_a_key_row_is_one_row(
    make_page: Callable[..., PlatePage],
) -> None:
    # What the shipped demo-sample suggests: `coverage` of scan-plan and `detector`
    # of cells are lines of KEY_PARAMS already. A second row would be the same field
    # twice in the column.
    profile = _profile(
        suggest={
            "scan-plan": {"params": ["coverage=0.4"], "why": "fast"},
            "cells": {"params": ["detector=blobs"], "why": "spots"},
        }
    )
    tools = {t.name: t for t in make_page(profile).tools()}
    assert [row[0] for row in tools["scan-plan"].key] == ["overlap", "coverage"]
    assert [row[0] for row in tools["cells"].key] == ["detector", "score_threshold"]
    assert tools["scan-plan"].suggested == (("coverage", "0.4"),)
    assert tools["cells"].suggested == (("detector", "blobs"),)


def test_a_suggested_value_may_hold_an_equals_sign(
    make_page: Callable[..., PlatePage],
) -> None:
    # `parse_params` splits at the first `=`: the value is whatever follows, so the
    # controller reads the key the same way, and does not call `a=b` a parameter.
    profile = _profile(suggest={"scan": {"params": ["label=a=b"], "why": "x"}})
    page = make_page(profile)
    assert page.view().problems == ()
    (scan,) = [t for t in page.tools() if t.name == "scan"]
    assert scan.suggested == (("label", "a=b"),)


# --- blocked: the reason that holds first -----------------------------------------------


def test_the_step_carries_the_tools_blocked_reason_first(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page(_profile(sample={}), connect=False)
    # The step works with the stand closed, and says so: it also has what to run.
    closed = page.view().next_step
    assert closed is not None
    assert closed.plugin == "plate-registration"
    assert closed.blocked == NOT_CONNECTED
    assert closed.params != ()

    page.connect()
    assert page.join(5)
    assert page.view().next_step is not None
    assert page.view().next_step.blocked is None  # type: ignore[union-attr]

    # A stopped scan plan with a checkpoint: the run document's refusal wins over
    # the step's own wait for wells, which is the weaker reason.
    _plate_done(run_doc)
    _seed(
        run_doc,
        "scan-plan",
        minute=1,
        status=StageStatus.stopped,
        checkpoint={"done": 1},
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
    )
    held = page.view().next_step
    assert held is not None
    assert held.plugin == "scan-plan"
    assert not page.selection  # SELECT_WELLS would be the step's own reason
    assert held.blocked == (
        "stopped with a checkpoint: continue it with `smc run scan-plan --resume` "
        "(close this page first); `smc run scan-plan` without --resume starts "
        "afresh and drops it"
    )


# --- the end ------------------------------------------------------------------------------


def test_the_last_step_lists_what_the_workflow_wrote(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _plate_done(run_doc)
    reads: dict[str, ReadRecord | None] = {}
    chain = [
        ("scan-plan", "plate-registration", {}),
        ("scan", "scan-plan", {"folder": "scan/t1"}),
        ("cells", "scan", {"file": "cells/t1.json"}),
        ("density", "cells", {}),
        ("route", "density", {}),
        ("report", "route", {"file": "report/t1.html"}),
        (
            "nis-export",
            "route",
            {"file": str(run_doc.parent / "out" / "positions.xml")},
        ),
    ]
    for minute, (name, upstream, result) in enumerate(chain, start=1):
        reads = {upstream: ReadRecord(plugin=upstream, attempt=f"a{minute - 1}")}
        # Every record read its upstream's current attempt: none is stale.
        if name == "scan-plan":
            reads = {"plate": ReadRecord(plugin="plate-registration", attempt="a0")}
        elif name == "report" or name == "nis-export":
            reads = {"route": ReadRecord(plugin="route", attempt="a5")}
        _seed(run_doc, name, minute=minute, result=result, reads=reads)
    step = make_page().view().next_step
    assert step is not None
    assert step.plugin is None
    assert step.blocked is None
    assert step.command == ""
    assert step.params == ()
    assert step.why == (
        "every stage of the workflow is done; what it wrote:",
        f"folder: {run_doc.parent / 'scan' / 't1'}",
        f"file: {run_doc.parent / 'cells' / 't1.json'}",
        f"file: {run_doc.parent / 'report' / 't1.html'}",
        f"file: {run_doc.parent / 'out' / 'positions.xml'}",
    )


# --- the command and the click -------------------------------------------------------------


def test_the_command_names_the_profile_the_dry_run_and_the_run_document(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page(_profile(sample={}), dry_run=True)
    step = page.view().next_step
    assert step is not None
    words = shlex.split(step.command)
    assert words[:3] == ["smc", "run", "plate-registration"]
    assert words[words.index("-p") + 1] == STAND
    assert words[words.index("--run-doc") + 1] == str(page.run_doc)
    assert Path(words[words.index("--run-doc") + 1]).is_absolute()
    assert "--dry-run" in words
    assert "--dry-run" not in shlex.split(
        make_page(_profile(sample={})).view().next_step.command  # type: ignore[union-attr]
    )


def test_a_profile_loaded_from_a_file_is_named_by_its_path(
    make_page: Callable[..., PlatePage],
) -> None:
    shipped = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
    profile = Profile.load(shipped)
    step = make_page(profile).view().next_step
    assert step is not None
    words = shlex.split(step.command)
    assert words[words.index("-p") + 1] == str(shipped)


def test_running_the_step_records_the_params_its_command_names(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    page = make_page(_profile(sample={"rotation_deg": 3.0}))
    step = page.view().next_step
    assert step is not None
    assert step.plugin == "plate-registration"
    page.run(step.plugin, list(step.params))
    assert page.join(10.0)

    record = RunDocument.load(run_doc).stages["plate-registration"]
    assert record.status is StageStatus.done
    # What `smc run` would record for the command the step shows, read as a shell does.
    expected = parse_params(RegistrationParams, _pairs_of(step.command))
    assert record.params == expected.model_dump(mode="json")
    assert len(record.params["refs"]) == 3
    # And the checklist moves on: the next step is the scan plan's.
    after = page.view().next_step
    assert after is not None
    assert after.plugin == "scan-plan"


def test_a_page_loads_smc_testing_only_for_a_profile_with_a_sample() -> None:
    # `smc.testing` is a stand-in for a stand (FakeCore, a synthetic plate): a real
    # stand's page must not load it, so the controller imports `sample_stand` inside the
    # step that reads a `[sample]`, and in the card that says where a registration's
    # references came from (m5 §8.8.4). In a fresh interpreter, since another test may
    # have imported it, and with a positive control: the same page does load it for a
    # sample.
    code = (
        "import sys, tempfile\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "from smc.hardware.profile import Profile\n"
        "from smc.plugins.slots import PlateResult\n"
        "from smc.run import StageRecord, StageStatus, write_stage\n"
        "from smc.ui.controller import PlatePage\n"
        "with tempfile.TemporaryDirectory() as tmp:\n"
        "    run_doc = Path(tmp) / 'run.json'\n"
        "    bare = PlatePage.closed(Profile.demo(), run_doc)\n"
        "    step = bare.view().next_step\n"
        "    assert step is not None and step.plugin == 'plate-registration', step\n"
        "    assert 'smc.testing' not in sys.modules, 'loaded without a sample'\n"
        "    # A registration done on this stand: its card is built, and loads nothing.\n"
        "    when = datetime(2026, 11, 2, tzinfo=timezone.utc)\n"
        "    plate = PlateResult(plate='96-well', a1_center_xy_um=(0.0, 0.0),\n"
        "        rotation_deg=0.0, method='references', residual_um=0.0,\n"
        "        scale_error=None, n_refs=3)\n"
        "    write_stage(Path(tmp) / 'cards.json', StageRecord(\n"
        "        plugin='plate-registration', provides='plate', attempt='a0',\n"
        "        status=StageStatus.done, started_utc=when, finished_utc=when,\n"
        "        profile='demo', dry_run=False, smc_version='0', params={'refs': []},\n"
        "        result=plate.model_dump(mode='json')))\n"
        "    cards = PlatePage.closed(Profile.demo(), Path(tmp) / 'cards.json').view().cards\n"
        "    assert cards['plate-registration'].status == 'done', cards\n"
        "    assert 'smc.testing' not in sys.modules, 'loaded by a plate card'\n"
        "    named = {'microscope': {'name': 's'}, 'sample': {}}\n"
        "    sampled = PlatePage.closed(Profile.model_validate(named), run_doc)\n"
        "    assert sampled.view().next_step is not None\n"
        "    assert 'smc.testing.sample_stand' in sys.modules, 'not loaded for a sample'\n"
    )
    # No `-I`: the interpreter keeps PYTHONPATH, so it imports the code under test
    # (`scripts/dev/mutant.py` puts a copy's `src` there) and not another checkout's.
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # Without `-I` the child honours PYTHONIOENCODING, which a mutant run sets: the
        # parent must read what the child writes, not the PC's code page.
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
