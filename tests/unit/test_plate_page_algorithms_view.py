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

import json
import os
import shlex
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import Field
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
from smc.plugins.slots import RoutePosition, RouteResult
from smc.run import StageRecord, StageStatus, write_stage
from smc.selection import registry
from smc.selection.api import OrdererParams, SelectorParams
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.plate_page import PlatePageView

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


# Referenced by later tests of this module; kept here so that the imports above stay used.
_ = (json, plate_page)
