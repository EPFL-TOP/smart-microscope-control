"""Which way the plate lies, in the page's controller: the choice, the switch and the names (m5 §16, #422).

The operator says it once next to the plate menu; every plate tool is given it as a
parameter, the estimate turns about the holder's centre, and on a registered plate the
switch is a run of ``plate-orientation`` that renames the wells without measuring.
These tests drive the controller over ``FakeCore`` (no Micro-Manager), with the real
plate tools where a run is the point and a stub that waits where an activity must be
going. The 96-well plate is the test bed of the estimate: its grid is not square (12 by
8), so swapped rows and columns, or a sign, show (FM-73).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret, XYStage
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult, WallTouch, WellRef
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, PAGE_PARAMS, PlatePage, PlateView
from smc.ui.guide import SELECTION_KEPT_ITS_NAMES

T0 = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
HOLDER_CENTRE = (400.0, -325.0)
#: The 96-well's grid centre from A1, written out: 5.5 pitches right, 3.5 down, of 9 mm.
OFFSET_96 = (5.5 * 9000.0, -3.5 * 9000.0)

REAL = {
    "plate-finding": "smc.plugins.plate_finding.plugin:PlateFindingPlugin",
    "plate-orientation": "smc.plugins.plate_orientation:PlateOrientationPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
}
PLATE_TOOLS = tuple(REAL)


class _NoParams(PluginParams):
    pass


class _NoResult(PluginResult):
    pass


class Slow:
    """Runs until released: an activity that is going while the operator switches."""

    name = provides = "slow"
    summary = "A test plugin that waits."
    Params = _NoParams
    Result = _NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: _NoParams, ctx: RunContext) -> _NoResult:
        Slow.started.set()
        while not Slow.release.is_set():
            ctx.wait_s(0.05)
        return _NoResult()


@pytest.fixture(autouse=True)
def _fresh_slow() -> Iterator[None]:
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()


def _entry_points(*names: str, slow: bool = False) -> list[EntryPoint]:
    points = [EntryPoint(name=n, value=REAL[n], group=GROUP) for n in names]
    if slow:
        points.append(EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP))
    return points


def _profile(*, sides: bool = True, back: str = "+Y", right: str = "+X") -> Profile:
    """The demo profile with a ``[holder]``: its centre, and its sides unless ``sides`` is false."""
    holder = HolderSection(
        plate_centre_xy_um=HOLDER_CENTRE,
        measured="2026-11-10, sessions 2 and 3 (#334, #405)",
        **(
            {
                "back": back,
                "right": right,
                "directions_measured": "2026-11-11, a test",
            }
            if sides
            else {}
        ),
    )
    return Profile.demo().model_copy(update={"holder": holder})


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def stand(fake_core: FakeCore) -> Iterator[Microscope]:
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), _profile())
    try:
        yield microscope
    finally:
        microscope.close()


def _page(
    stand: Microscope, run_doc: Path, *names: str, slow: bool = False
) -> PlatePage:
    return PlatePage(stand, run_doc, entry_points=_entry_points(*names, slow=slow))


def _fit(
    plate: str = "96-well",
    *,
    a1: tuple[float, float] = (1000.0, -500.0),
    rotation_deg: float = 0.3,
    orientation: str = "not-turned",
) -> PlateResult:
    return PlateResult(
        plate=plate,
        a1_center_xy_um=a1,
        rotation_deg=rotation_deg,
        method="references",
        residual_um=2.0,
        scale_error=0.001,
        n_refs=2,
        orientation=orientation,  # type: ignore[arg-type]
    )


def _record(
    run_doc: Path,
    plugin: str,
    result: PlateResult | None,
    *,
    params: dict[str, Any],
    minute: int = 0,
    status: StageStatus = StageStatus.done,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A record of a provider of ``plate`` as an earlier run left it; ``minute`` orders the starts."""
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides="plate",
            attempt=f"a{minute}",
            status=status,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params=params,
            reads=reads or {},
            result=None if result is None else result.model_dump(mode="json"),
        ),
    )


def _registered(
    run_doc: Path,
    result: PlateResult,
    *,
    refs: tuple[WellRef, ...] = (),
    walls: tuple[WallTouch, ...] = (),
    minute: int = 0,
) -> None:
    """A ``plate-registration`` record with the captures it was run with."""
    _record(
        run_doc,
        "plate-registration",
        result,
        minute=minute,
        params={
            "plate": result.plate,
            "refs": [r.model_dump(mode="json") for r in refs],
            "walls": [w.model_dump(mode="json") for w in walls],
            "orientation": result.orientation,
        },
    )


def _centres(view: PlateView) -> dict[str, tuple[float, float]]:
    return {w.well: w.center_um for w in view.wells}


def _close(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6


# --- every plate tool is given the choice (D5) ------------------------------------------


def test_the_orientation_reaches_every_plate_tool_as_a_parameter(
    stand: Microscope, run_doc: Path
) -> None:
    page = _page(stand, run_doc, *PLATE_TOOLS)
    try:
        page.choose_plate("96-well")
        xy = stand.require(XYStage)
        for well, (x_um, y_um) in (("A1", (0.0, 0.0)), ("B2", (9000.0, -9000.0))):
            xy.move_to_um(x_um, y_um)
            page.capture_centre(well)
        for tool in PLATE_TOOLS:
            assert PAGE_PARAMS[tool][-1] == "orientation", tool
            assert page.page_params(tool)[-1] == "orientation=not-turned", tool
        page.choose_orientation("turned-180")
        for tool in PLATE_TOOLS:
            assert page.page_params(tool)[-1] == "orientation=turned-180", tool
        # No form offers it a second time: the select is the one place to say it.
        forms = {t.name: t for t in page.tools()}
        for tool in PLATE_TOOLS:
            rows = {row[0] for row in (*forms[tool].key, *forms[tool].advanced)}
            assert "orientation" not in rows, tool
    finally:
        page.close()


def test_an_unknown_orientation_is_refused_and_changes_nothing(
    stand: Microscope, run_doc: Path
) -> None:
    page = _page(stand, run_doc)
    try:
        with pytest.raises(ValueError, match="not-turned, turned-180"):
            page.choose_orientation(cast("Any", "90"))
        assert page.orientation == "not-turned"
    finally:
        page.close()


# --- on a registered plate, the switch is a run (D4) ------------------------------------


def test_switching_on_a_registered_plate_runs_plate_orientation(
    stand: Microscope, run_doc: Path
) -> None:
    _registered(run_doc, _fit())
    page = _page(stand, run_doc, "plate-orientation", "plate-registration")
    try:
        before = page.view()
        assert page.orientation == "not-turned"
        page.choose_orientation("turned-180")
        assert page.join(10.0)
        assert page.status().outcome is not None
        assert page.status().outcome.startswith("plate-orientation done in ")

        doc = RunDocument.load(run_doc)
        turn = doc.stages["plate-orientation"]
        assert turn.status is StageStatus.done
        assert turn.params == {"orientation": "turned-180"}
        assert turn.reads["plate"] == ReadRecord(
            plugin="plate-registration", attempt="a0"
        )
        # Nothing measured again: the registration is the one written above.
        assert doc.stages["plate-registration"].attempt == "a0"

        after = page.view()
        assert page.orientation == after.orientation == "turned-180"
        assert after.pose.kind == "registered"
        assert after.pose.a1_corner == "the front right"
        # The same places, other names: the well at the old A1 is H12, and B3 is G10.
        old, new = _centres(before), _centres(after)
        assert _close(new["H12"], old["A1"])
        assert _close(new["G10"], old["B3"])
        assert sorted(round(c[0], 3) for c in new.values()) == sorted(
            round(c[0], 3) for c in old.values()
        )

        # The orientation the plate already has does nothing: no second run.
        ended = page.activities_ended
        page.choose_orientation("turned-180")
        assert page.join(10.0)
        assert page.activities_ended == ended
    finally:
        page.close()


def test_switching_without_a_stand_changes_nothing_and_says_why(
    stand: Microscope, run_doc: Path
) -> None:
    _registered(run_doc, _fit())
    closed = PlatePage.closed(
        _profile(), run_doc, entry_points=_entry_points("plate-orientation")
    )
    try:
        closed.choose_orientation("turned-180")
        assert closed.status().outcome == f"plate-orientation: {NOT_CONNECTED}"
        assert closed.orientation == "not-turned"
        assert "plate-orientation" not in RunDocument.load(run_doc).stages
    finally:
        closed.close()

    # An activity going: the controller refuses it itself, whatever the select showed
    # (FM-169), and says why in the outcome.
    page = _page(stand, run_doc, "plate-orientation", slow=True)
    try:
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        page.choose_orientation("turned-180")
        assert page.status().outcome == "plate-orientation: slow is running"
        Slow.release.set()
        assert page.join(10.0)
        assert page.orientation == "not-turned"
        assert "plate-orientation" not in RunDocument.load(run_doc).stages
    finally:
        page.close()


def test_the_orientation_shown_is_the_registered_plates(
    stand: Microscope, run_doc: Path
) -> None:
    # A plate registered turned, in a terminal or an earlier session: the page starts
    # on it, and it wins over what was declared for the estimate.
    _registered(run_doc, _fit(rotation_deg=-179.7, orientation="turned-180"))
    page = _page(stand, run_doc)
    try:
        assert page.orientation == "turned-180"
        view = page.view()
        assert view.orientation == view.pose.orientation == "turned-180"
        # A registered plate of another plate than the page's is no registration of it:
        # the page shows what it declared, which started as the slot's.
        page.choose_plate("24-well")
        assert page.view().pose.kind == "estimated"
        assert page.orientation == "turned-180"
        page.choose_orientation("not-turned")
        assert page.orientation == "not-turned"
        # Back on the registered plate, the registration's own wins again.
        page.choose_plate("96-well")
        assert page.orientation == "turned-180"
    finally:
        page.close()


def test_a_turn_made_in_a_terminal_shows_at_the_next_view(
    stand: Microscope, run_doc: Path
) -> None:
    # `smc run plate-orientation` from a terminal writes the run document only: the
    # page learns it at its next read, and the tick (which compares `orientation`
    # without reading the file, FM-145) then redraws every tab.
    _registered(run_doc, _fit())
    page = _page(stand, run_doc)
    try:
        assert page.orientation == "not-turned"
        _record(
            run_doc,
            "plate-orientation",
            _fit(rotation_deg=-179.7, orientation="turned-180"),
            minute=1,
            params={"orientation": "turned-180"},
            reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
        )
        assert page.orientation == "not-turned"  # nothing read the file yet
        assert page.view().orientation == "turned-180"
        assert page.orientation == "turned-180"
    finally:
        page.close()


# --- on an estimate, the plate turns about the holder's centre (D7) -----------------------


def test_switching_on_an_estimate_turns_it_about_the_holder_centre(
    stand: Microscope, run_doc: Path
) -> None:
    page = _page(stand, run_doc)
    try:
        page.choose_plate("96-well")
        before = page.view()
        assert before.pose.kind == "estimated"
        cx, cy = HOLDER_CENTRE
        dx, dy = OFFSET_96
        assert _close(_centres(before)["A1"], (cx - dx, cy - dy))
        assert before.pose.a1_corner == "the back left"

        page.choose_orientation("turned-180")
        after = page.view()
        assert after.pose.kind == "estimated"
        assert after.pose.orientation == after.orientation == "turned-180"
        assert after.pose.a1_corner == "the front right"
        # A1 is the holder's centre plus the offset; H12 is where A1 was; A12 (twelve
        # columns along X) and H1 (eight rows along Y) are swapped corners, not swapped
        # axes: 99 000 µm in X and 63 000 µm in Y from A1 (FM-73).
        new = _centres(after)
        assert _close(new["A1"], (cx + dx, cy + dy))
        assert _close(new["H12"], _centres(before)["A1"])
        assert _close(new["A12"], (cx + dx - 99_000.0, cy + dy))
        assert _close(new["H1"], (cx + dx, cy + dy + 63_000.0))
        assert all(w.rotation_deg == 180.0 for w in after.wells)
        # The grid's centre stays on the holder.
        mean_x = sum(c[0] for c in new.values()) / len(new)
        mean_y = sum(c[1] for c in new.values()) / len(new)
        assert abs(mean_x - cx) < 1e-6
        assert abs(mean_y - cy) < 1e-6
        # Nothing is run or written for an estimate.
        assert not run_doc.exists()
    finally:
        page.close()


def test_choosing_another_plate_keeps_the_orientation(
    stand: Microscope, run_doc: Path
) -> None:
    page = _page(stand, run_doc)
    try:
        page.choose_plate("96-well")
        page.choose_orientation("turned-180")
        page.choose_plate("greiner-24-well")
        assert page.orientation == "turned-180"
        assert page.view().pose.orientation == "turned-180"
    finally:
        page.close()


# --- the captures follow a turn; the selection keeps its names (D6) -----------------------

REFS = (
    WellRef(well="B3", x_um=19_000.0, y_um=-9_500.0),
    WellRef(well="H12", x_um=100_000.0, y_um=-63_500.0),
)
WALLS = (WallTouch(well="B3", side="+X", x_um=22_000.0, y_um=-9_500.0),)


def test_a_turn_renames_the_captures_and_keeps_the_selection_by_name(
    stand: Microscope, run_doc: Path
) -> None:
    _registered(run_doc, _fit(), refs=REFS, walls=WALLS)
    page = _page(stand, run_doc, "plate-orientation", "plate-registration")
    try:
        assert [r.well for r in page.refs] == ["B3", "H12"]
        page.select_text("B3,C5")
        page.current = "B3"  # the well the capture buttons name

        page.choose_orientation("turned-180")
        assert page.join(10.0)

        # The same positions under the turned plate's names (B3 is G10, H12 is A1).
        assert page.refs == (
            WellRef(well="G10", x_um=19_000.0, y_um=-9_500.0),
            WellRef(well="A1", x_um=100_000.0, y_um=-63_500.0),
        )
        # A wall keeps its stage side: the stage did not turn, the plate's names did.
        assert page.walls == (
            WallTouch(well="G10", side="+X", x_um=22_000.0, y_um=-9_500.0),
        )
        assert page.current is None
        assert set(page.selection) == {"B3", "C5"}
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.endswith(f"; {SELECTION_KEPT_ITS_NAMES}")
        # The next fit is given the captures by their new names and the new orientation.
        pairs = page.page_params("plate-registration")
        assert json.loads(pairs[1].removeprefix("refs="))[0]["well"] == "G10"
        assert pairs[-1] == "orientation=turned-180"

        # Turned back, the names come back; no selection, no note.
        page.select_text("")
        page.choose_orientation("not-turned")
        assert page.join(10.0)
        assert [r.well for r in page.refs] == ["B3", "H12"]
        assert page.walls[0].well == "B3"
        outcome = page.status().outcome
        assert outcome is not None
        assert SELECTION_KEPT_ITS_NAMES not in outcome
    finally:
        page.close()


def test_a_turn_renames_the_capture_objectives_with_the_captures(
    stand: Microscope, run_doc: Path
) -> None:
    # The turret label a capture was taken with is a third piece of state beside `refs`
    # (#444): the positions stay and the names follow the turned plate, so the label must
    # follow its centre, or the Plate screen would say "the objective not known" for it.
    _registered(run_doc, _fit(), refs=REFS)
    page = _page(stand, run_doc, "plate-orientation", "plate-registration")
    try:
        stand.require(XYStage).move_to_um(19_000.0, -9_500.0)
        page.capture_centre("B3")
        label = stand.require(ObjectiveTurret).current()
        assert page.capture_objectives == {"B3": label}

        page.choose_orientation("turned-180")
        assert page.join(10.0)

        # B3 is G10 on the turned plate: the same centre, the same objective.
        assert [r.well for r in page.refs] == ["A1", "G10"]
        assert page.capture_objectives == {"G10": label}
        assert page._capture_objectives == {"G10": label}
        page.choose_orientation("not-turned")
        assert page.join(10.0)
        assert page.capture_objectives == {"B3": label}
    finally:
        page.close()


def test_captures_are_restored_through_a_turn_after_a_restart(
    stand: Microscope, run_doc: Path
) -> None:
    _registered(run_doc, _fit(), refs=REFS, walls=WALLS)
    turned = _fit(rotation_deg=-179.7, orientation="turned-180")
    read = {"plate": ReadRecord(plugin="plate-registration", attempt="a0")}
    _record(
        run_doc,
        "plate-orientation",
        turned,
        minute=1,
        params={"orientation": "turned-180"},
        reads=read,
    )
    page = _page(stand, run_doc)
    try:
        # The turn has no captures: they are the fit's, renamed for the turned plate.
        assert [r.well for r in page.refs] == ["G10", "A1"]
        assert [w.well for w in page.walls] == ["G10"]
        assert page.orientation == "turned-180"
    finally:
        page.close()

    # A turn that failed renamed nothing: the fit's names stand.
    _record(
        run_doc,
        "plate-orientation",
        None,
        minute=2,
        status=StageStatus.failed,
        params={"orientation": "turned-180"},
        reads=read,
    )
    page = _page(stand, run_doc)
    try:
        assert [r.well for r in page.refs] == ["B3", "H12"]
    finally:
        page.close()

    # Turned back: the latest turn read the same fit and says its orientation, so the
    # fit's names stand (a rename here would name every capture 180° off).
    _record(
        run_doc,
        "plate-orientation",
        _fit(),
        minute=2,
        params={"orientation": "not-turned"},
        reads=read,
    )
    page = _page(stand, run_doc)
    try:
        assert [r.well for r in page.refs] == ["B3", "H12"]
        assert [w.well for w in page.walls] == ["B3"]
        assert page.orientation == "not-turned"
    finally:
        page.close()

    # A fit run again since the turn read it is no source: no captures, as today.
    _registered(run_doc, _fit(), refs=REFS[:1], minute=3)
    _record(
        run_doc,
        "plate-orientation",
        turned,
        minute=4,
        params={"orientation": "turned-180"},
        reads=read,
    )
    page = _page(stand, run_doc)
    try:
        assert page.refs == ()
        assert page.walls == ()
    finally:
        page.close()


def test_a_failed_turn_keeps_the_plate_and_the_captures_of_the_fit_it_read(
    stand: Microscope, run_doc: Path
) -> None:
    # A plate that is not the page's placeholder (96-well): a failed turn has no
    # result and no `plate` parameter, and the page must not fall back to the
    # placeholder and drop the captures with it.
    refs = (
        WellRef(well="B3", x_um=19_000.0, y_um=-19_000.0),
        WellRef(well="C4", x_um=38_000.0, y_um=-38_000.0),
    )
    _registered(run_doc, _fit("24-well"), refs=refs)
    _record(
        run_doc,
        "plate-orientation",
        None,
        minute=1,
        status=StageStatus.failed,
        params={"orientation": "turned-180"},
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
    )
    page = _page(stand, run_doc)
    try:
        assert page.plate == "24-well"
        assert page.plate_chosen
        assert [r.well for r in page.refs] == ["B3", "C4"]
    finally:
        page.close()


def test_a_seed_declared_turned_shows_its_orientation(
    stand: Microscope, run_doc: Path
) -> None:
    # One centre, registered turned (a seed): the pose and the select say turned, and
    # A1's corner is the stand's front right (m5 §16 D4, D8).
    seed = PlateResult(
        plate="96-well",
        a1_center_xy_um=(1000.0, -500.0),
        rotation_deg=180.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=1,
        refs=(WellRef(well="G10", x_um=1000.0, y_um=-500.0),),
        orientation="turned-180",
    )
    _registered(run_doc, seed)
    page = _page(stand, run_doc)
    try:
        view = page.view()
        assert view.pose.kind == "seed"
        assert view.orientation == view.pose.orientation == "turned-180"
        assert view.pose.a1_corner == "the front right"
    finally:
        page.close()


def test_an_unplaced_map_keeps_the_declared_orientation(
    fake_core: FakeCore, run_doc: Path
) -> None:
    # A stand that does not say where its plate lies (no holder centre): the map stays
    # unplaced, but the choice stands, for the select and for the next fit.
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), Profile.demo())
    page = PlatePage(microscope, run_doc, entry_points=_entry_points(*PLATE_TOOLS))
    try:
        page.choose_plate("96-well")
        page.choose_orientation("turned-180")
        view = page.view()
        assert view.pose.kind == "unplaced"
        assert view.orientation == "turned-180"
        assert page.page_params("plate-finding") == ["orientation=turned-180"]
    finally:
        page.close()
        microscope.close()


def test_a_tool_run_after_a_terminal_turn_is_given_the_turned_orientation(
    stand: Microscope, run_doc: Path
) -> None:
    # The click reads the run document (`goto`'s rule): a turn written since the last
    # view is the one the next fit is given, with no view in between.
    _registered(run_doc, _fit())
    page = _page(stand, run_doc, *PLATE_TOOLS)
    try:
        page.view()
        _record(
            run_doc,
            "plate-orientation",
            _fit(rotation_deg=-179.7, orientation="turned-180"),
            minute=1,
            params={"orientation": "turned-180"},
            reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
        )
        assert page.page_params("plate-finding") == ["orientation=turned-180"]
    finally:
        page.close()


# --- another tab (FM-151) -------------------------------------------------------------------


@pytest.mark.usefixtures("handler_failures")
def test_a_second_tab_draws_the_new_orientation(
    stand: Microscope, run_doc: Path
) -> None:
    from smc.ui.plate_page import PlatePageView

    page = _page(stand, run_doc)
    try:
        page.choose_plate("96-well")
        first, second = PlatePageView(page), PlatePageView(page)
        first.refresh()
        second.refresh()
        assert second.orientation_select.value == "not-turned"
        before = second.figure.title.text

        first.orientation_select.value = "turned-180"
        assert page.orientation == "turned-180"
        refreshed = 0
        original = second.refresh

        def counted() -> None:
            nonlocal refreshed
            refreshed += 1
            original()

        second.refresh = counted  # type: ignore[method-assign]
        second.tick()
        # The tick saw the change and redrew; nothing else told it.
        assert refreshed == 1
        assert second.orientation_select.value == "turned-180"
        assert second.figure.title.text != before
        assert "declared turned 180" in second.figure.title.text
    finally:
        page.close()
