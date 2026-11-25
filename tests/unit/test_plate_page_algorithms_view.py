"""The Results screen chooses how ``route`` selects and orders the fields, without a browser (m5 §21, #569).

The widgets are driven the way a session drives them: a value is ``widget.value = ...``, a click is
``button.clicks += 1``. A second selector (``density``, with a float, an int and a tuple) and a second
orderer are put in the registry by replacing ``smc.selection.registry.entry_points`` as
``test_route_selection.py`` does, next to the built-ins, and ``monkeypatch`` puts it back. ``route`` is
the real plugin (its ``Params`` carry the two names and their two JSON texts); the other four tools of
the screen are stubs. What this cannot show is a browser: how the long option labels read at 1280 px,
and the hover itself, are for ``/ui-check`` and the stand (FM-161); here the sources, the lists and the
text of every pane are asserted. Panel's warnings are errors (FM-147), and a handler that failed fails
the test that did not expect it.
"""

from __future__ import annotations

import html
import json
import os
import shlex
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
from markdown_it import MarkdownIt
from pydantic import Field, create_model, field_validator
from pymmcore_plus import CMMCorePlus

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

import smc.ui.plate_page as plate_page
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    PlateResult,
    RoutePosition,
    RouteResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)
from smc.run import StageRecord, StageStatus, write_stage
from smc.selection import registry
from smc.selection.api import OrdererParams, SelectorParams
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.plate_page import PlatePageView, TextPane

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

THIS = __name__
HOSTILE = '<img src=x onerror="alert(1)"> **bold** [x](javascript:y) &amp;'
DEMO_SAMPLE = Path(__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
BOUND_S = 10.0

#: The real tools the page loads from their own entry values; the other four of the screen are stubs.
REAL = {
    "route": "smc.plugins.route.plugin:RoutePlugin",
    "scan": "smc.plugins.scan.plugin:ScanPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
}


# --- the algorithms the registry holds besides the built-ins --------------------------------


class DensityParams(SelectorParams):
    """A selector's own parameters of the kinds a collaborator writes: a bounded float, a bounded int, a tuple."""

    min_density_per_mm2: float = Field(10.0, gt=0, description="Cells per mm2.")
    keep: int = Field(3, ge=1)
    tags: tuple[str, ...] = ()


class DensitySelector:
    name = "density"
    summary = "Fields above a density,  best first."
    Params = DensityParams

    def select(self, data: Any, params: DensityParams) -> list[Any]:
        return []


class ByScoreParams(OrdererParams):
    descending: bool = True


class ByScore:
    name = "by-score"
    summary = "Highest score first."
    Params = ByScoreParams

    def order(self, choices: Any, data: Any, params: ByScoreParams) -> list[int]:
        return list(range(len(choices)))


class ModedParams(SelectorParams):
    """A string and a tuple with defaults: what ``describe_params`` prints with quotes and a repr."""

    mode: str = "fast"
    tags: tuple[str, ...] = ("a", "b")


class ModedSelector:
    name = "moded"
    summary = "Has a string default and a tuple default."
    Params = ModedParams

    def select(self, data: Any, params: ModedParams) -> list[Any]:
        return []


class StrictParams(SelectorParams):
    """A validator that expects text and raises what pydantic does not wrap when it is given a list."""

    tags: tuple[str, ...] = ()

    @field_validator("tags", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        return value.split(",")  # type: ignore[attr-defined]


class StrictSelector:
    name = "strict"
    summary = "Its validator raises an AttributeError on a list."
    Params = StrictParams

    def select(self, data: Any, params: StrictParams) -> list[Any]:
        return []


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class _Stub:
    """A tool of the Results screen that does nothing: this module reads widgets, it runs no tool."""

    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads_optional: tuple[str, ...] = ()
    summary = "A test stand-in for a Results tool."
    name: str
    provides: str
    reads: tuple[str, ...]

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        return NoResult()


class Cells(_Stub):
    name = provides = "cells"
    reads = ("scan",)


class Density(_Stub):
    name = provides = "density"
    reads = ("scan-plan", "scan", "cells")


class Report(_Stub):
    name = provides = "report"
    reads = ("scan-plan", "scan", "cells", "density")


class NisExport(_Stub):
    name = provides = "nis-export"
    reads = ("route",)


STUBS = (Cells, Density, Report, NisExport)


def _entry_points() -> list[EntryPoint]:
    found = [EntryPoint(name=n, value=v, group=GROUP) for n, v in REAL.items()]
    for cls in STUBS:
        found.append(
            EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)
        )
    return found


def _algorithm(name: str, target: str, group: str) -> EntryPoint:
    return EntryPoint(name, target, group)


MOST_CELLS = _algorithm(
    "most-cells", "smc.selection.builtin:MostCells", registry.SELECTOR_GROUP
)
NEAREST = _algorithm("nearest", "smc.selection.builtin:Nearest", registry.ORDERER_GROUP)
AS_CHOSEN = _algorithm(
    "as-chosen", "smc.selection.builtin:AsChosen", registry.ORDERER_GROUP
)
DENSITY = _algorithm("density", f"{THIS}:DensitySelector", registry.SELECTOR_GROUP)
BY_SCORE = _algorithm("by-score", f"{THIS}:ByScore", registry.ORDERER_GROUP)
MODED = _algorithm("moded", f"{THIS}:ModedSelector", registry.SELECTOR_GROUP)
STRICT = _algorithm("strict", f"{THIS}:StrictSelector", registry.SELECTOR_GROUP)


@pytest.fixture(autouse=True)
def registered(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """The built-ins and ``density`` and ``by-score`` in place of the installed algorithms; a test may put others first."""

    def put(
        selectors: list[EntryPoint] | None = None,
        orderers: list[EntryPoint] | None = None,
    ) -> None:
        groups = {
            registry.SELECTOR_GROUP: list(
                [MOST_CELLS, DENSITY] if selectors is None else selectors
            ),
            registry.ORDERER_GROUP: list(
                [NEAREST, AS_CHOSEN, BY_SCORE] if orderers is None else orderers
            ),
        }
        monkeypatch.setattr(
            registry, "entry_points", lambda *, group: list(groups[group])
        )

    put()
    return put


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Presets are read from, and written to, the working directory's `profiles/presets`.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMC_PROFILES", raising=False)


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    """A page over a fake stand, not connected unless told to; every page and stand is closed afterwards."""
    pages: list[PlatePage] = []
    stands: list[Microscope] = []

    def make(
        *, profile: Profile | None = None, connect: bool = False, dry_run: bool = False
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
            entry_points=_entry_points(),
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
        for page in pages:
            page.close()
        for stand in stands:
            stand.close()


@pytest.fixture
def page(make_page: Callable[..., PlatePage]) -> PlatePage:
    return make_page()


# --- helpers -------------------------------------------------------------------------------


def _seed_route(run_doc: Path, positions: list[RoutePosition]) -> None:
    then = datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc)
    result = RouteResult(
        objective=None,
        matrix_um_per_px=((1.0, 0.0), (0.0, 1.0)),
        image_shape_px=(10, 10),
        fov_um=(10.0, 10.0),
        labels=("live",),
        positions=tuple(positions),
        uncovered={},
        distance_um=5.0,
        travel_s=None,
    )
    write_stage(
        run_doc,
        StageRecord(
            plugin="route",
            provides="route",
            attempt="a0",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _stop(order: int, well: str, name: str, **extra: Any) -> RoutePosition:
    return RoutePosition(
        order=order,
        well=well,
        name=name,
        x_um=float(order),
        y_um=2.0 * order,
        z_um=None,
        pfs_offset=None,
        counts={"live": 1},
        **extra,
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


def _words(command: str) -> list[str]:
    """A command as a POSIX shell splits it, never compared as the string the code builds (FM-45)."""
    return shlex.split(command)


# --- the map --------------------------------------------------------------------------------


def test_the_map_tip_names_the_stops_and_the_hover_reads_the_route_points_and_not_the_frame(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    _seed_route(
        run_doc,
        [
            _stop(0, "A1", "A1_r000"),
            _stop(
                1,
                "A1",
                "A1_r001",
                score=0.823456,
                measures={"nn_median_um": 35.0, "density_per_mm2": 41.2},
            ),
            _stop(2, "A2", "A2_r000", score=0.0),
        ],
    )
    view = PlatePageView(make_page())
    data = view.route_source.data
    # One tip per stop, in visiting order; a stop of the default route has its name and visit
    # (the route was not hoverable before), a score of 0.0 is a score.
    assert list(data["tip"]) == [
        "A1_r000 (visit 1)",
        "A1_r001 (visit 2), score 0.823; density_per_mm2 41.2, nn_median_um 35",
        "A2_r000 (visit 3), score 0",
    ]
    # Every column has the length of the stops: a source whose columns differ is a map that
    # stops redrawing.
    assert {len(column) for column in data.values()} == {3}
    assert set(data) == {"x", "y", "order", "well", "tip"}
    # The hover reads the wells and the route's circles, never the map's own frame (FM-174).
    assert view.hover.renderers == [view.wells_renderer, view.route_points]
    assert view.route_points.data_source is view.route_source
    assert view.frame_renderer not in view.hover.renderers
    # A plate that swaps the wells' glyph keeps the route's circles in the list.
    view._ensure_renderer(not view._circular)
    assert view.hover.renderers == [view.wells_renderer, view.route_points]
    assert view.frame_renderer not in view.hover.renderers
    # No route: every column is empty and still of one length.
    run_doc.unlink()
    view.refresh()
    assert {len(column) for column in view.route_source.data.values()} == {0}
    assert set(view.route_source.data) == {"x", "y", "order", "well", "tip"}


def test_a_hostile_stop_name_is_in_the_tip_as_text(
    make_page: Callable[..., PlatePage], run_doc: Path
) -> None:
    # FM-139: the tip is a column of plain strings; Bokeh escapes `@tip` when it draws it.
    _seed_route(run_doc, [_stop(0, "A1", HOSTILE, measures={HOSTILE: 1.0})])
    view = PlatePageView(make_page())
    assert list(view.route_source.data["tip"]) == [f"{HOSTILE} (visit 1); {HOSTILE} 1"]
    assert view.hover.tooltips == "@tip"


# --- the run document of a scan, for Run all ------------------------------------------------


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


def _seed(run_doc: Path, plugin: str, result: PluginResult, *, minute: int) -> None:
    then = datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
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
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _seed_scan(run_doc: Path) -> None:
    """A registered plate, a plan and the scan: what the Results screen finds after a scan."""
    plan = _scan_plan_result()
    _seed(run_doc, "plate-registration", _plate_result(), minute=0)
    _seed(run_doc, "scan-plan", plan, minute=1)
    _seed(run_doc, "scan", _scan_result(plan), minute=2)


# --- the selector and the orderer as the row's choices (D1, D2) -----------------------------

MOST_CELLS_SUMMARY = (
    "In each well, the fields that hold the most cells of the wanted classes, "
    "greedily, never overlapping."
)
NEAREST_SUMMARY = (
    "Wells in the plan's order; in each, nearest neighbour then 2-opt, "
    "starting nearest the previous well's last stop."
)
DASH = "\N{EM DASH}"
ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"


def _selector(view: PlatePageView) -> Any:
    return view.tool_fields["route"]["selector"]


def _orderer(view: PlatePageView) -> Any:
    return view.tool_fields["route"]["orderer"]


def _carrier(view: PlatePageView, kind: str = "selector") -> Any:
    return view.tool_fields["route"][f"{kind}_params"]


def _density(view: PlatePageView) -> dict[str, Any]:
    """The three fields of ``density``'s group, by key."""
    return view._algorithms["selector"].fields["density"]


def test_the_route_row_offers_the_installed_algorithms_with_the_default_chosen(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    route = view.result_rows["route"]
    selector, orderer = _selector(view), _orderer(view)
    assert isinstance(selector, pn.widgets.Select)
    assert isinstance(orderer, pn.widgets.Select)
    assert (selector.label, orderer.label) == (
        "Select the fields with",
        "Visit the fields in this order",
    )
    # The default first, its value "" (an empty field is the tool's default everywhere on the
    # page), then the others by name; a long summary is cut and the pane below says it whole.
    assert list(selector.options.items()) == [
        (
            f"most-cells (default) {DASH} In each well, the fields that hold the most "
            f"cells of the wanted classes, greedi{ELLIPSIS}",
            "",
        ),
        (f"density {DASH} Fields above a density, best first.", "density"),
    ]
    assert list(orderer.options.items()) == [
        (
            f"nearest (default) {DASH} Wells in the plan's order; in each, nearest "
            f"neighbour then 2-opt, starting near{ELLIPSIS}",
            "",
        ),
        (
            f"as-chosen {DASH} Visit the fields in the order the selector listed them.",
            "as-chosen",
        ),
        (f"by-score {DASH} Highest score first.", "by-score"),
    ]
    assert (selector.value, orderer.value) == ("", "")
    algorithms = view._algorithms
    assert algorithms["selector"].summary.object == MOST_CELLS_SUMMARY
    assert algorithms["orderer"].summary.object == NEAREST_SUMMARY
    assert algorithms["selector"].summary.visible is True
    # Each choice has the *from ...* line of the other choices, and its panes follow it.
    assert route.from_lines["selector"].object == "the tool's default: most-cells"
    assert route.from_lines["orderer"].object == "the tool's default: nearest"
    inside = list(route.card.objects)
    for kind, field in (("selector", selector), ("orderer", orderer)):
        widgets = algorithms[kind]
        order = [
            inside.index(field),
            inside.index(route.from_lines[kind]),
            inside.index(widgets.summary),
            inside.index(widgets.note),
            *(inside.index(group) for group in widgets.groups.values()),
        ]
        assert order == sorted(order), kind
    # One instance in the card and in the expert view's *Advanced*, as every choice is (D5 of §17.2).
    expert = _objects(view.screens["expert"])
    assert selector in expert
    assert orderer in expert
    assert selector in inside
    assert orderer in inside
    # The tool's own `labels` and `fields_per_well` are still fields of the card, before them.
    assert inside.index(view.tool_fields["route"]["fields_per_well"]) < inside.index(
        selector
    )
    assert algorithms["selector"].note.visible is False


def test_the_most_cells_fields_are_shown_for_it_and_hidden_for_another_selector(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    selector = view._algorithms["selector"]
    own = view.tool_fields["route"]
    assert set(selector.groups) == {"density", "most-cells"}
    # The default is chosen: its group is route's own two fields, the instances of *Advanced*.
    assert selector.groups["most-cells"].visible is True
    assert selector.groups["density"].visible is False
    assert list(selector.groups["most-cells"].objects) == [
        own["min_cells"],
        own["candidate_step_fraction"],
    ]
    _selector(view).value = "density"
    assert selector.groups["most-cells"].visible is False
    assert selector.groups["density"].visible is True
    # An orderer that takes nothing has no group: a Select and a summary, nothing else.
    orderer = view._algorithms["orderer"]
    assert set(orderer.groups) == {"by-score"}
    assert all(group.visible is False for group in orderer.groups.values())
    _orderer(view).value = "by-score"
    assert orderer.groups["by-score"].visible is True
    _orderer(view).value = "as-chosen"
    assert orderer.groups["by-score"].visible is False
    assert orderer.summary.object == (
        "Visit the fields in the order the selector listed them."
    )


def test_the_default_choice_sends_no_selector_pair_and_another_sends_its_name(
    make_page: Callable[..., PlatePage], monkeypatch: pytest.MonkeyPatch
) -> None:
    page = make_page(connect=True)
    view = PlatePageView(page)
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        page, "run", lambda plugin, params: calls.append((plugin, list(params)))
    )
    route = view.result_rows["route"]
    route.button.clicks += 1
    assert calls == [("route", [])]
    assert "selector" not in route.command.object
    _selector(view).value = "density"
    _orderer(view).value = "by-score"
    assert route.from_lines["selector"].object == "typed on this page"
    assert _words(route.command.object)[-4:] == [
        "--param",
        "selector=density",
        "--param",
        "orderer=by-score",
    ]
    # The row's *Run* and the expert view's are the tool's one handler: the same call.
    route.button.clicks += 1
    view.tool_runs["route"].clicks += 1
    assert calls[1:] == [("route", ["selector=density", "orderer=by-score"])] * 2
    # The default chosen again is no pair.
    _selector(view).value = ""
    assert route.from_lines["selector"].object == "the tool's default: most-cells"
    route.button.clicks += 1
    assert calls[-1] == ("route", ["orderer=by-score"])


# --- the chosen algorithm's fields and the JSON text (D3 to D5) -----------------------------


def test_a_selectors_fields_write_selector_params_as_typed_json(
    page: PlatePage,
) -> None:
    from smc.plugins.params import parse_params
    from smc.plugins.route.plugin import RouteParams

    view = PlatePageView(page)
    route = view.result_rows["route"]
    _selector(view).value = "density"
    fields = _density(view)
    assert list(fields) == ["min_density_per_mm2", "keep", "tags"]
    # Label, placeholder and description are the key, the default and the type and bounds: an
    # empty field is the algorithm's default.
    assert fields["min_density_per_mm2"].label == "min_density_per_mm2"
    assert fields["min_density_per_mm2"].placeholder == "10.0"
    assert fields["min_density_per_mm2"].description == "float  gt=0  Cells per mm2."
    assert fields["keep"].placeholder == "3"
    assert fields["keep"].description == "int  ge=1"
    assert _carrier(view).value == ""
    fields["min_density_per_mm2"].value = "20"
    assert _carrier(view).value == '{"min_density_per_mm2": 20.0}'
    # The page's own write of the text is not an edit of the text: the fields keep what the
    # operator typed (``20``, not the ``20.0`` the text holds), and nothing echoes back.
    assert fields["min_density_per_mm2"].value == "20"
    fields["keep"].value = "4"
    assert _carrier(view).value == '{"min_density_per_mm2": 20.0, "keep": 4}'
    assert (fields["min_density_per_mm2"].value, fields["keep"].value) == ("20", "4")
    fields["tags"].value = '["a", "b"]'
    assert json.loads(_carrier(view).value) == {
        "min_density_per_mm2": 20.0,
        "keep": 4,
        "tags": ["a", "b"],
    }
    fields["tags"].value = ""
    assert _carrier(view).value == '{"min_density_per_mm2": 20.0, "keep": 4}'
    assert view._algorithms["selector"].note.object == ""
    assert view._algorithms["selector"].note.visible is False
    # The command is the pair the CLI takes, quoted for a POSIX shell, and `route` accepts it.
    pairs = [
        "selector=density",
        'selector_params={"min_density_per_mm2": 20.0, "keep": 4}',
    ]
    assert _words(route.command.object)[-4:] == [
        "--param",
        pairs[0],
        "--param",
        pairs[1],
    ]
    assert parse_params(RouteParams, pairs).selector_params == {
        "min_density_per_mm2": 20.0,
        "keep": 4,
    }
    # An emptied field leaves the key out, and the last one leaves no text at all.
    fields["keep"].value = ""
    fields["min_density_per_mm2"].value = ""
    assert _carrier(view).value == ""


def test_a_value_the_selector_refuses_is_sent_as_typed_and_named_in_the_note(
    page: PlatePage,
) -> None:
    from smc.plugins.errors import ParamsError
    from smc.plugins.params import parse_params
    from smc.plugins.route.plugin import RouteParams

    view = PlatePageView(page)
    _selector(view).value = "density"
    _density(view)["keep"].value = "abc"
    note = view._algorithms["selector"].note
    assert _carrier(view).value == '{"keep": "abc"}'
    assert note.object == (
        "keep: Input should be a valid integer, unable to parse string as an integer"
    )
    assert note.visible is True
    # The run's own refusal names the same key, in `route`'s words.
    with pytest.raises(ParamsError) as refused:
        parse_params(
            RouteParams, ["selector=density", f"selector_params={_carrier(view).value}"]
        )
    assert refused.value.problems == (
        "parameters: Value error, selector_params: keep: Input should be a valid "
        "integer, unable to parse string as an integer",
    )
    # A correction clears the note and writes the typed value.
    _density(view)["keep"].value = "4"
    assert _carrier(view).value == '{"keep": 4}'
    assert note.object == ""
    assert note.visible is False


def test_a_text_typed_in_the_expert_view_fills_the_fields(page: PlatePage) -> None:
    view = PlatePageView(page)
    _selector(view).value = "density"
    text = '{"min_density_per_mm2": 20, "tags": ["a", "b"]}'
    _carrier(view).value = text
    fields = _density(view)
    assert {key: widget.value for key, widget in fields.items()} == {
        "min_density_per_mm2": "20",
        "keep": "",
        "tags": '["a", "b"]',
    }
    # The text is the truth and was not rewritten (the user's own spacing stays).
    assert _carrier(view).value == text
    assert view._algorithms["selector"].note.visible is False
    # A new text empties the fields it does not hold.
    _carrier(view).value = '{"keep": 2}'
    assert {key: widget.value for key, widget in fields.items()} == {
        "min_density_per_mm2": "",
        "keep": "2",
        "tags": "",
    }
    _carrier(view).value = ""
    assert {widget.value for widget in fields.values()} == {""}


def test_a_key_the_selector_lacks_is_named_and_stays_in_the_text(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    _selector(view).value = "density"
    note = view._algorithms["selector"].note
    text = '{"keep": 4, "zzz": 1}'
    _carrier(view).value = text
    assert _density(view)["keep"].value == "4"
    assert note.object == (
        "selector_params also holds zzz, which density does not take: "
        "Run will refuse it."
    )
    assert note.visible is True
    # Left in the text: `route` refuses it in its own words, which a rewrite would hide.
    assert _carrier(view).value == text
    # A text that is not an object is named too, and shows no field.
    _carrier(view).value = "[1]"
    assert note.object == (
        "selector_params is not a JSON object (a JSON object is expected): "
        "Run will refuse it."
    )
    assert _density(view)["keep"].value == ""
    _carrier(view).value = "{nope"
    assert note.object == (
        "selector_params is not a JSON object (Expecting property name enclosed in "
        "double quotes: line 1 column 2 (char 1)): Run will refuse it."
    )
    assert _carrier(view).value == "{nope"
    _carrier(view).value = ""
    assert note.object == ""
    assert note.visible is False
    # Any key of `most-cells` in the text is one it does not take here (its parameters are
    # route's own fields, and `route` refuses the pair).
    _selector(view).value = ""
    _carrier(view).value = '{"min_cells": 2}'
    assert note.object == (
        "selector_params also holds min_cells, which most-cells does not take: "
        "Run will refuse it."
    )


def test_an_edit_of_a_field_after_that_rewrites_the_text_whole_and_the_note_goes(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    _selector(view).value = "density"
    note = view._algorithms["selector"].note
    _carrier(view).value = '{"keep": 4, "zzz": 1}'
    assert note.visible is True
    _density(view)["keep"].value = "5"
    # The whole text from the fields: the key the algorithm lacks is gone, and so is the note.
    assert _carrier(view).value == '{"keep": 5}'
    assert note.object == ""
    assert note.visible is False


def test_an_orderers_fields_write_orderer_params(page: PlatePage) -> None:
    view = PlatePageView(page)
    _orderer(view).value = "by-score"
    fields = view._algorithms["orderer"].fields["by-score"]
    assert list(fields) == ["descending"]
    assert fields["descending"].placeholder == "True"
    fields["descending"].value = "no"
    assert _carrier(view, "orderer").value == '{"descending": false}'
    assert _carrier(view).value == ""
    _carrier(view, "orderer").value = '{"descending": true}'
    assert fields["descending"].value == "true"
    # `nearest` takes nothing: any key of the text is one it does not take.
    _orderer(view).value = ""
    assert _carrier(view, "orderer").value == ""
    _carrier(view, "orderer").value = '{"x": 1}'
    assert view._algorithms["orderer"].note.object == (
        "orderer_params also holds x, which nearest does not take: Run will refuse it."
    )


def test_the_most_cells_fields_typed_on_the_default_selector_are_route_parameters_and_never_selector_params(
    page: PlatePage,
) -> None:
    from smc.plugins.params import parse_params
    from smc.plugins.route.plugin import RouteParams

    view = PlatePageView(page)
    own = view.tool_fields["route"]
    own["min_cells"].value = "3"
    own["candidate_step_fraction"].value = "0.5"
    # `route` refuses these two inside `selector_params` ("give them as route parameters"): they
    # are fields of the tool, sent as pairs of their own, and the JSON text stays empty.
    assert _carrier(view).value == ""
    assert view._algorithms["selector"].note.visible is False
    route = view.result_rows["route"]
    assert _words(route.command.object)[-4:] == [
        "--param",
        "min_cells=3",
        "--param",
        "candidate_step_fraction=0.5",
    ]
    parsed = parse_params(RouteParams, ["min_cells=3", "candidate_step_fraction=0.5"])
    assert (parsed.min_cells, parsed.candidate_step_fraction) == (3, 0.5)


def test_choosing_an_orderer_keeps_what_was_typed_for_the_selector(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    own = view.tool_fields["route"]
    own["min_cells"].value = "3"
    own["candidate_step_fraction"].value = "0.5"
    _orderer(view).value = "by-score"
    # D5 empties the old algorithm's own text and fields: the selector's, when the selector
    # changes. An orderer's change is none of the selector's business.
    assert (own["min_cells"].value, own["candidate_step_fraction"].value) == (
        "3",
        "0.5",
    )
    _orderer(view).value = ""
    assert (own["min_cells"].value, own["candidate_step_fraction"].value) == (
        "3",
        "0.5",
    )
    # And a selector's text is left alone by the orderer's choice.
    _selector(view).value = "density"
    _density(view)["keep"].value = "4"
    _orderer(view).value = "by-score"
    assert _carrier(view).value == '{"keep": 4}'
    assert _carrier(view, "orderer").value == ""
    assert _words(view.result_rows["route"].command.object)[-6:] == [
        "--param",
        "selector=density",
        "--param",
        'selector_params={"keep": 4}',
        "--param",
        "orderer=by-score",
    ]


def test_a_group_fields_placeholder_is_the_default_as_the_operator_reads_it(
    make_page: Callable[..., PlatePage], registered: Callable[..., None]
) -> None:
    registered(selectors=[MOST_CELLS, MODED])
    view = PlatePageView(make_page())
    fields = view._algorithms["selector"].fields["moded"]
    # `describe_params` prints a string with its quotes and a tuple as its repr: neither is
    # what the operator would type (the same rule as every tool's field).
    assert fields["mode"].placeholder == "fast"
    assert fields["tags"].placeholder == "a, b"


def test_a_validator_that_raises_what_pydantic_does_not_wrap_is_a_note_and_the_text_is_written(
    make_page: Callable[..., PlatePage],
    registered: Callable[..., None],
    handler_failures: list[Any],
) -> None:
    registered(selectors=[MOST_CELLS, STRICT])
    view = PlatePageView(make_page())
    _selector(view).value = "strict"
    fields = view._algorithms["selector"].fields["strict"]
    fields["tags"].value = '["a", "b"]'
    # No handler failed: the text is what was typed, and the note says what `route` will say.
    assert handler_failures == []
    assert _carrier(view).value == '{"tags": "[\\"a\\", \\"b\\"]"}'
    note = view._algorithms["selector"].note
    assert note.object == "AttributeError: 'list' object has no attribute 'split'"
    assert note.visible is True
    assert view.message.object == ""


# --- a change of algorithm (D5) -------------------------------------------------------------


def test_choosing_another_selector_empties_the_old_text_and_the_most_cells_fields(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    own = view.tool_fields["route"]
    own["min_cells"].value = "3"
    own["candidate_step_fraction"].value = "0.5"
    selector = _selector(view)
    selector.value = "density"
    # `route` refuses `min_cells` other than its default beside another selector: it must not
    # travel hidden, and the old text is gone.
    assert (own["min_cells"].value, own["candidate_step_fraction"].value) == ("", "")
    assert _carrier(view).value == ""
    _density(view)["keep"].value = "4"
    assert _carrier(view).value == '{"keep": 4}'
    # Most-cells again: the density text goes with the algorithm, and its fields show their
    # placeholders (an empty field is the default).
    selector.value = ""
    assert _carrier(view).value == ""
    assert (own["min_cells"].value, own["candidate_step_fraction"].value) == ("", "")
    assert (
        own["min_cells"].placeholder,
        own["candidate_step_fraction"].placeholder,
    ) == (
        "1",
        "0.25",
    )
    assert view._algorithms["selector"].note.visible is False
    # The density fields are emptied the next time they are chosen.
    selector.value = "density"
    assert {w.value for w in _density(view).values()} == {""}


def test_a_note_is_cleared_when_another_algorithm_is_chosen(page: PlatePage) -> None:
    view = PlatePageView(page)
    _selector(view).value = "density"
    _density(view)["keep"].value = "abc"
    note = view._algorithms["selector"].note
    assert note.visible is True
    _selector(view).value = ""
    assert note.object == ""
    assert note.visible is False


# --- presets (D6) ---------------------------------------------------------------------------


def _load(view: PlatePageView, name: str) -> None:
    view.preset_select.value = name
    view.preset_load_button.clicks += 1


def _stored(tmp_path: Path, name: str) -> dict[str, Any]:
    path = tmp_path / "profiles" / "presets" / f"{name}.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_a_preset_with_a_selector_and_its_parameters_round_trips_through_save_and_load(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    first = PlatePageView(make_page())
    _selector(first).value = "density"
    _density(first)["min_density_per_mm2"].value = "20"
    _density(first)["keep"].value = "4"
    first.preset_name.value = "mine"
    first.preset_save_button.clicks += 1
    path = tmp_path / "profiles" / "presets" / "mine.toml"
    # The one pair `selector_params=` the command takes, beside `selector=`.
    assert _stored(tmp_path, "mine")["params"]["route"] == [
        "selector=density",
        'selector_params={"min_density_per_mm2": 20.0, "keep": 4}',
    ]
    assert first.message.object == f"saved mine to {path}"

    second = PlatePageView(make_page())
    _load(second, "mine")
    assert _selector(second).value == "density"
    assert _carrier(second).value == '{"min_density_per_mm2": 20.0, "keep": 4}'
    assert {key: w.value for key, w in _density(second).items()} == {
        "min_density_per_mm2": "20.0",
        "keep": "4",
        "tags": "",
    }
    assert second.result_rows["route"].from_lines["selector"].object == (
        "from the preset mine"
    )
    assert view_group_visible(second) == "density"
    # The load is not an edit: nothing differs from the preset, and the summary is drawn.
    assert second.preset_note.object == (
        f"loaded: mine  ({path})\nthe page holds the preset's values"
    )
    assert second._algorithms["selector"].summary.object == (
        "Fields above a density, best first."
    )


def view_group_visible(view: PlatePageView, kind: str = "selector") -> str | None:
    """The name of the one visible group of ``kind``, or ``None``."""
    shown = [n for n, g in view._algorithms[kind].groups.items() if g.visible]
    assert len(shown) <= 1
    return shown[0] if shown else None


def test_a_load_does_not_rewrite_the_text_it_put_in_the_field(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    # Written by hand, with a spacing and an order the page would not write.
    pair = 'selector_params={"keep":4,  "min_density_per_mm2":20}'
    (folder / "hand.toml").write_text(
        f"[params]\nroute = [\"selector=density\", '{pair}']\n", encoding="utf-8"
    )
    view = PlatePageView(make_page())
    _load(view, "hand")
    assert _carrier(view).value == '{"keep":4,  "min_density_per_mm2":20}'
    assert {key: w.value for key, w in _density(view).items()} == {
        "min_density_per_mm2": "20",
        "keep": "4",
        "tags": "",
    }
    # Nothing differs from the preset: the text was not counted as an edit.
    assert view.preset_note.object.endswith("the page holds the preset's values")
    assert view.result_rows["route"].from_lines["selector"].object == (
        "from the preset hand"
    )


def test_a_load_over_another_selector_keeps_what_the_preset_holds_for_the_default(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    (folder / "own.toml").write_text(
        '[params]\nroute = ["min_cells=3"]\n', encoding="utf-8"
    )
    view = PlatePageView(make_page())
    _selector(view).value = "density"
    _density(view)["keep"].value = "4"
    assert _carrier(view).value == '{"keep": 4}'
    _load(view, "own")
    # The load assigns `min_cells` before the Select, and the Select going back to the default
    # is the preset's value, not the operator's choice: it must not empty what the preset put.
    assert _selector(view).value == ""
    assert view.tool_fields["route"]["min_cells"].value == "3"
    assert _carrier(view).value == ""
    assert view_group_visible(view) == "most-cells"
    assert view.result_rows["route"].from_lines["selector"].object == (
        "the tool's default: most-cells"
    )
    assert view.preset_note.object.endswith("the page holds the preset's values")
    assert _words(view.result_rows["route"].command.object)[-2:] == [
        "--param",
        "min_cells=3",
    ]


def test_a_preset_that_names_a_selector_that_is_not_installed_is_shown_and_run_is_refused_with_the_installed_ones(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    (folder / "gone.toml").write_text(
        '[params]\nroute = ["selector=my-selector"]\n', encoding="utf-8"
    )
    page = make_page(connect=True)
    view = PlatePageView(page)
    _load(view, "gone")
    selector = _selector(view)
    # Shown, not lost: the option is added before the assignment (D6).
    assert selector.value == "my-selector"
    assert selector.options["my-selector (not installed)"] == "my-selector"
    assert view._algorithms["selector"].summary.object == (
        "my-selector is not installed on this PC; installed: density, most-cells."
    )
    assert view_group_visible(view) is None
    assert view.result_rows["route"].from_lines["selector"].object == (
        "from the preset gone"
    )
    assert view.preset_note.object.endswith("the page holds the preset's values")
    # `route` refuses it with the names it has.
    page.run("route", ["selector=my-selector"])
    outcome = page.status().outcome
    assert outcome is not None
    assert "my-selector" in outcome
    assert "known: density, most-cells" in outcome


def test_ensure_option_adds_a_name_once(page: PlatePage) -> None:
    view = PlatePageView(page)
    selector = _selector(view)
    before = dict(selector.options)
    view._ensure_option(selector, "density")  # an option: nothing is added
    assert selector.options == before
    view._ensure_option(selector, "my-selector")
    view._ensure_option(selector, "my-selector")
    assert selector.options == {**before, "my-selector (not installed)": "my-selector"}
    selector.value = "my-selector"
    assert selector.value == "my-selector"


def test_a_profile_suggestion_that_names_a_selector_that_is_not_installed_is_shown(
    make_page: Callable[..., PlatePage],
) -> None:
    profile = Profile.model_validate(
        {
            "microscope": {"name": "demo"},
            "suggest": {
                "route": {
                    "params": ["selector=my-selector", 'selector_params={"x": 1}'],
                    "why": "the stand's own selector",
                }
            },
        }
    )
    view = PlatePageView(make_page(profile=profile))
    selector = _selector(view)
    assert selector.value == "my-selector"
    assert "my-selector (not installed)" in selector.options
    assert view.result_rows["route"].from_lines["selector"].object == (
        "from the profile's suggestions"
    )
    assert view._algorithms["selector"].summary.object.startswith(
        "my-selector is not installed on this PC"
    )


def test_the_default_named_explicitly_by_a_suggestion_or_a_preset_is_installed_and_says_so(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    # `selector=most-cells` and `orderer=nearest` are legal pairs: the algorithms are installed,
    # and their entries are `""` (the default's), so the value names them as an entry of its own.
    profile = Profile.model_validate(
        {
            "microscope": {"name": "demo"},
            "suggest": {
                "route": {
                    "params": ["selector=most-cells", "orderer=nearest"],
                    "why": "the stand's own choice",
                }
            },
        }
    )
    view = PlatePageView(make_page(profile=profile))
    for field in (_selector(view), _orderer(view)):
        assert "not installed" not in " ".join(field.options)
    assert (_selector(view).value, _orderer(view).value) == ("most-cells", "nearest")
    assert _selector(view).options["most-cells (default, named)"] == "most-cells"
    assert _orderer(view).options["nearest (default, named)"] == "nearest"
    assert view._algorithms["selector"].summary.object == MOST_CELLS_SUMMARY
    assert view._algorithms["orderer"].summary.object == NEAREST_SUMMARY
    assert view_group_visible(view) == "most-cells"
    # The same through a load of a preset that names them.
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True)
    (folder / "named.toml").write_text(
        '[params]\nroute = ["selector=most-cells", "orderer=nearest"]\n',
        encoding="utf-8",
    )
    other = PlatePageView(make_page())
    _load(other, "named")
    assert "not installed" not in " ".join(_selector(other).options)
    assert (_selector(other).value, _orderer(other).value) == ("most-cells", "nearest")
    assert other.result_rows["route"].from_lines["selector"].object == (
        "from the preset named"
    )
    assert other.preset_note.object.endswith("the page holds the preset's values")
    assert view_group_visible(other) == "most-cells"


def test_a_profile_suggestion_of_a_selector_and_its_text_fills_the_group_at_the_build(
    make_page: Callable[..., PlatePage],
) -> None:
    profile = Profile.model_validate(
        {
            "microscope": {"name": "demo"},
            "suggest": {
                "route": {
                    "params": [
                        "selector=density",
                        'selector_params={"keep": 7, "zzz": 1}',
                    ],
                    "why": "the stand's own selector",
                }
            },
        }
    )
    view = PlatePageView(make_page(profile=profile))
    assert _selector(view).value == "density"
    assert _density(view)["keep"].value == "7"
    assert view_group_visible(view) == "density"
    assert view._algorithms["selector"].note.object == (
        "selector_params also holds zzz, which density does not take: Run will refuse it."
    )


# --- no algorithm, or the default is not installed (D7) -------------------------------------


def test_no_selector_registered_leaves_the_text_fields_and_says_why(
    make_page: Callable[..., PlatePage], registered: Callable[..., None]
) -> None:
    registered(selectors=[])
    view = PlatePageView(make_page())
    # The generic text field stays, as before this section; the orderer is a Select.
    assert type(_selector(view)) is pn.widgets.TextInput
    assert isinstance(_orderer(view), pn.widgets.Select)
    assert set(view._algorithms) == {"orderer"}
    assert "selector" in view.result_rows["route"].from_lines
    view.refresh()
    assert (
        "no selector is registered: run pip install -e . in the smc folder "
        "(entry points are read when smc is installed)"
    ) in view.problems_text.object
    # The text field still sends what it holds.
    _selector(view).value = "anything"
    assert _words(view.result_rows["route"].command.object)[-2:] == [
        "--param",
        "selector=anything",
    ]


def test_a_default_that_is_not_installed_is_an_entry_that_says_so(
    make_page: Callable[..., PlatePage], registered: Callable[..., None]
) -> None:
    registered(selectors=[DENSITY])
    view = PlatePageView(make_page())
    selector = _selector(view)
    # Never an algorithm nobody chose: the value is still "" (the tool's default).
    assert selector.value == ""
    assert next(iter(selector.options.items())) == (
        "most-cells (default, not installed)",
        "",
    )
    assert view._algorithms["selector"].summary.object == (
        "most-cells is not installed on this PC; installed: density."
    )
    assert view_group_visible(view) is None


def test_an_algorithm_that_does_not_import_is_no_option_and_the_problem_is_on_the_page(
    make_page: Callable[..., PlatePage], registered: Callable[..., None]
) -> None:
    broken = _algorithm("broken", "no_such_module_569:Thing", registry.SELECTOR_GROUP)
    registered(selectors=[MOST_CELLS, broken, DENSITY])
    view = PlatePageView(make_page())
    assert sorted(_selector(view).options.values()) == ["", "density"]
    view.refresh()
    assert (
        "selector 'broken' could not be imported: "
        "ModuleNotFoundError: No module named 'no_such_module_569'"
    ) in view.problems_text.object


# --- the whole results screen ---------------------------------------------------------------


def test_run_all_sends_the_selector_pairs_to_the_chain(
    make_page: Callable[..., PlatePage], run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(run_doc)
    page = make_page(connect=True)
    view = PlatePageView(page)
    _selector(view).value = "density"
    _density(view)["keep"].value = "4"
    sent: list[Any] = []
    monkeypatch.setattr(
        page,
        "run_chain",
        lambda tools, params: sent.append((list(tools), dict(params))),
    )
    view.run_all_button.clicks += 1
    assert len(sent) == 1
    tools, params = sent[0]
    assert tools == ["cells", "density", "route", "report", "nis-export"]
    assert params["route"] == ["selector=density", 'selector_params={"keep": 4}']
    assert params["cells"] == []


def test_the_rows_command_carries_the_selector_pairs_and_the_windows_note(
    page: PlatePage,
) -> None:
    from smc.ui.plate_page import posix_quoting_note

    view = PlatePageView(page)
    view._windows = True
    _selector(view).value = "density"
    _density(view)["keep"].value = "4"
    route = view.result_rows["route"]
    assert _words(route.command.object)[-4:] == [
        "--param",
        "selector=density",
        "--param",
        'selector_params={"keep": 4}',
    ]
    # The braces and quotes of the JSON are quoted as a POSIX shell wants them (FM-175), and a
    # page served on Windows says so under the command: the saved preset is the way in there.
    assert "'selector_params={\"keep\": 4}'" in route.command.object
    assert route.command_note.object == posix_quoting_note(True)
    assert route.command_note.visible is True


def test_the_tick_and_a_refresh_read_no_registry_and_leave_the_choices_as_they_are(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = PlatePageView(page)
    _selector(view).value = "density"
    _density(view)["keep"].value = "4"
    asked: list[str] = []
    real = registry.entry_points

    def counting(*, group: str) -> Any:
        asked.append(group)
        return real(group=group)

    monkeypatch.setattr(registry, "entry_points", counting)
    view.tick()
    view.refresh()
    view.tick()
    # FM-145: the algorithms were read with the tools, and the tick reads nothing of them.
    assert asked == []
    assert _selector(view).value == "density"
    assert _carrier(view).value == '{"keep": 4}'
    assert view._algorithms["selector"].summary.object == (
        "Fields above a density, best first."
    )


def test_a_second_tab_draws_the_shared_preset_and_the_choice_of_its_own(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    # FM-151: a preset loaded through the first tab is the page's; the second tab keeps its own
    # widgets and says the preset differs from what it holds, as it does for every field.
    page = make_page()
    first = PlatePageView(page)
    second = PlatePageView(page)
    _selector(first).value = "density"
    first.preset_name.value = "mine"
    first.preset_save_button.clicks += 1
    second.tick()
    assert page.preset is not None
    assert _selector(second).value == ""
    assert second.preset_note.object != first.preset_note.object


# --- plain text, whatever a third party wrote (FM-139) ---------------------------------------

HostileParams = create_model(
    "HostileParams",
    __base__=SelectorParams,
    **{"<b>key</b>": (int, Field(1, description=HOSTILE))},  # type: ignore[call-overload]
)


class HostileSelector:
    name = "hostile"
    summary = HOSTILE
    Params = HostileParams

    def select(self, data: Any, params: Any) -> list[Any]:
        return []


HOSTILE_ENTRY = _algorithm(
    "hostile", f"{THIS}:HostileSelector", registry.SELECTOR_GROUP
)


def test_a_hostile_summary_key_and_description_stay_plain_text(
    make_page: Callable[..., PlatePage],
    registered: Callable[..., None],
    browser_text: Callable[[Any], str],
) -> None:
    registered(selectors=[MOST_CELLS, HOSTILE_ENTRY])
    view = PlatePageView(make_page())
    route = view.result_rows["route"]
    selector = _selector(view)
    selector.value = "hostile"
    widgets = view._algorithms["selector"]
    # The summary pane is the page's escaping text pane and holds the string as it is, and
    # the browser is sent one `pre` that says it (FM-271).
    assert type(widgets.summary) is TextPane
    assert widgets.summary.object == HOSTILE
    assert browser_text(widgets.summary) == HOSTILE
    label = next(
        label for label, value in selector.options.items() if value == "hostile"
    )
    assert label.startswith(f"hostile {DASH} <img src=x onerror=")
    (field,) = widgets.fields["hostile"].values()
    assert field.label == "<b>key</b>"
    # Panel renders a description as Markdown into HTML in a tooltip, so the page hands it the
    # characters, not the markup: entities and backslashes that read back as the original string.
    assert field.description == (
        'int  &lt;img src=x onerror="alert(1)"&gt; \\*\\*bold\\*\\* '
        "\\[x\\](javascript:y) &amp;amp;"
    )
    rendered = MarkdownIt("commonmark").render(field.description)
    assert "<img" not in rendered
    assert "<strong>" not in rendered
    assert html.unescape(rendered).strip() == f"<p>int  {HOSTILE}</p>"
    # A note that names a hostile key is plain text too.
    _carrier(view).value = json.dumps({HOSTILE: 1})
    assert type(widgets.note) is TextPane
    assert HOSTILE in widgets.note.object
    assert browser_text(widgets.note) == widgets.note.object
    # Every pane of the row is a text pane, but the card's own HTML title.
    titles = {"<h3>Choices and command</h3>"}
    panes = [
        o
        for o in _objects([route.box])
        if isinstance(o, pn.pane.PaneBase)
        and not (type(o) is pn.pane.HTML and o.object in titles)
    ]
    assert panes
    assert [type(o).__name__ for o in panes if type(o) is not TextPane] == []
    for pane in panes:
        assert browser_text(pane) == (pane.object or " ")


def test_every_handler_the_algorithm_widgets_register_is_guarded(
    page: PlatePage, handler_failures: list[Any]
) -> None:
    # FM-171: a handler runs under the document lock and never raises into Panel, the
    # Select's included; each widget of the groups is walked by name here.
    view = PlatePageView(page)
    widgets: list[Any] = [_selector(view), _orderer(view)]
    for built in view._algorithms.values():
        for fields in built.fields.values():
            widgets.extend(fields.values())
    assert len(widgets) >= 2 + 2 + 3 + 1
    for widget in widgets:
        registered = [
            watcher.fn
            for by_name in widget.param.watchers.values()
            for watchers in by_name.values()
            for watcher in watchers
        ]
        ours = [
            fn
            for fn in registered
            if getattr(fn, "__module__", "") == plate_page.__name__
        ]
        assert ours, widget.label
        assert [fn for fn in ours if not hasattr(fn, "__wrapped__")] == [], widget.label
    assert handler_failures == []
