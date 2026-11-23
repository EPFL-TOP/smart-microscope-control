"""The page's list of the installed selectors and orderers, and a stop's selector score, in the controller (m5 §21, #569).

No Panel and no stand. The registry is the real one with the entry points replaced as
``test_route_selection.py`` does, so a second selector, a broken one, or none at all
stands for what an installation holds. ``route`` is the real plugin (its ``Params`` have the
two names); the page reads the algorithms once, with the tools (FM-145), and what it could
not read is a line of the problems in the registry's own words (FM-113, FM-98).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.params import describe_params
from smc.plugins.registry import GROUP
from smc.plugins.slots import RoutePosition, RouteResult
from smc.run import StageRecord, StageStatus, write_stage
from smc.selection import registry
from smc.selection.api import OrdererParams, SelectorParams
from smc.selection.builtin import MostCellsParams
from smc.ui.controller import PlatePage

THIS = __name__
ROUTE = EntryPoint("route", "smc.plugins.route.plugin:RoutePlugin", GROUP)


class DensityParams(SelectorParams):
    """A selector's own parameters: a bounded float, a bounded int and a tuple."""

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


class NoRouteParams(PluginParams):
    """The parameters of a ``route`` that has no selector and no orderer."""


class NoResult(PluginResult):
    pass


class StubRoute:
    """A ``route`` of another kind: its parameters have neither name, so the page offers nothing."""

    name = provides = "route"
    summary = "A stand-in route."
    Params = NoRouteParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> NoResult:
        return NoResult()


def _entry(name: str, target: str, group: str) -> EntryPoint:
    return EntryPoint(name, target, group)


BUILTIN_SELECTORS = [
    _entry("most-cells", "smc.selection.builtin:MostCells", registry.SELECTOR_GROUP)
]
BUILTIN_ORDERERS = [
    _entry("nearest", "smc.selection.builtin:Nearest", registry.ORDERER_GROUP),
    _entry("as-chosen", "smc.selection.builtin:AsChosen", registry.ORDERER_GROUP),
]


@pytest.fixture
def register(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[str]]:
    """Put these entry points in the registry in place of the installed ones; each call asked of it is noted.

    Returns the list of groups the registry was asked for, so that a test can say how
    often the page reads it.
    """

    def put(
        selectors: list[EntryPoint] | None = None,
        orderers: list[EntryPoint] | None = None,
    ) -> list[str]:
        groups = {
            registry.SELECTOR_GROUP: list(
                BUILTIN_SELECTORS if selectors is None else selectors
            ),
            registry.ORDERER_GROUP: list(
                BUILTIN_ORDERERS if orderers is None else orderers
            ),
        }
        asked: list[str] = []

        def entry_points(*, group: str) -> list[EntryPoint]:
            asked.append(group)
            return list(groups[group])

        monkeypatch.setattr(registry, "entry_points", entry_points)
        return asked

    return put


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    pages: list[PlatePage] = []

    def make(tools: tuple[EntryPoint, ...] = (ROUTE,)) -> PlatePage:
        page = PlatePage.closed(Profile.demo(), run_doc, entry_points=list(tools))
        pages.append(page)
        return page

    try:
        yield make
    finally:
        for page in pages:
            page.close()


DENSITY = _entry("density", f"{THIS}:DensitySelector", registry.SELECTOR_GROUP)
BY_SCORE = _entry("by-score", f"{THIS}:ByScore", registry.ORDERER_GROUP)


def test_the_page_lists_the_installed_selectors_and_orderers_with_their_summaries_and_parameter_rows(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    register([*BUILTIN_SELECTORS, DENSITY], [*BUILTIN_ORDERERS, BY_SCORE])
    page = make_page()
    selectors = page.algorithms("selector")
    # By name, the default among them: the page puts it first, not the controller.
    assert [choice.name for choice in selectors] == ["density", "most-cells"]
    density, most_cells = selectors
    # The summary as the algorithm wrote it, whole: the page collapses it where it shows it.
    assert density.summary == "Fields above a density,  best first."
    assert density.model is DensityParams
    assert density.params == tuple(describe_params(DensityParams))  # type: ignore[arg-type]
    assert density.params[0] == (
        "min_density_per_mm2",
        "float",
        "10.0",
        "gt=0  Cells per mm2.",
    )
    assert most_cells.model is MostCellsParams
    assert [row[0] for row in most_cells.params] == [
        "min_cells",
        "candidate_step_fraction",
    ]
    orderers = page.algorithms("orderer")
    assert [choice.name for choice in orderers] == ["as-chosen", "by-score", "nearest"]
    by_score = orderers[1]
    assert (by_score.summary, by_score.params) == (
        "Highest score first.",
        (("descending", "bool", "True", ""),),
    )
    assert page.view().problems == ()


def test_an_algorithm_that_does_not_import_is_no_option_and_a_line_of_the_problems(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    broken = _entry("broken", "no_such_module_569:Thing", registry.SELECTOR_GROUP)
    register([*BUILTIN_SELECTORS, broken, DENSITY])
    page = make_page()
    assert [c.name for c in page.algorithms("selector")] == ["density", "most-cells"]
    # The registry's own words, one line, and no exception kept (FM-98).
    assert page.view().problems == (
        "selector 'broken' could not be imported: "
        "ModuleNotFoundError: No module named 'no_such_module_569'",
    )


def test_an_algorithm_whose_parameters_cannot_be_described_is_no_option_and_a_line(
    register: Callable[..., list[str]],
    make_page: Callable[..., PlatePage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import smc.ui.controller as controller_module

    register([*BUILTIN_SELECTORS, DENSITY])
    real = controller_module.describe_params

    def describe(model: Any) -> Any:
        if model is DensityParams:
            raise RuntimeError("a default that cannot\nbe printed")
        return real(model)

    monkeypatch.setattr(controller_module, "describe_params", describe)
    page = make_page()
    assert [c.name for c in page.algorithms("selector")] == ["most-cells"]
    assert page.view().problems == (
        "selector density: RuntimeError: a default that cannot be printed",
    )


def test_a_kind_with_no_algorithm_is_a_problem_in_the_registrys_words_and_lists_none(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    register(selectors=[])
    page = make_page()
    assert page.algorithms("selector") == ()
    # The orderers are there: one kind's absence is not the other's.
    assert [c.name for c in page.algorithms("orderer")] == ["as-chosen", "nearest"]
    assert page.view().problems == (
        "no selector is registered: run pip install -e . in the smc folder "
        "(entry points are read when smc is installed)",
    )


def test_the_algorithms_are_read_once_and_the_tick_reads_no_registry(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    asked = register([*BUILTIN_SELECTORS, DENSITY])
    page = make_page()
    assert asked == []  # building the page reads nothing
    page.algorithms("selector")
    after_the_first = len(asked)
    assert after_the_first > 0
    # Both kinds were read together, then never again (FM-145): a second ask, a view
    # and a status, the page's tick, add nothing.
    page.algorithms("selector")
    page.algorithms("orderer")
    page.view()
    page.status()
    page.view()
    assert len(asked) == after_the_first


def test_a_view_alone_reads_the_algorithms_once(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    asked = register()
    page = make_page()
    page.view()
    after_the_first = len(asked)
    assert after_the_first > 0
    page.view()
    page.status()
    assert len(asked) == after_the_first


def test_a_route_that_did_not_load_has_no_algorithms_and_no_problem(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    asked = register(selectors=[])  # a problem, if anything read it
    page = make_page(())
    assert page.algorithms("selector") == ()
    assert page.algorithms("orderer") == ()
    assert page.view().problems == ()
    assert asked == []


def test_a_route_with_neither_name_among_its_parameters_offers_none_and_says_nothing(
    register: Callable[..., list[str]], make_page: Callable[..., PlatePage]
) -> None:
    asked = register(selectors=[])
    page = make_page((_entry("route", f"{THIS}:StubRoute", GROUP),))
    assert page.algorithms("selector") == ()
    assert page.algorithms("orderer") == ()
    assert page.view().problems == ()
    assert asked == []


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


def test_a_route_stop_carries_its_name_score_and_measures_and_a_default_route_has_none(
    register: Callable[..., list[str]],
    make_page: Callable[..., PlatePage],
    run_doc: Path,
) -> None:
    register()
    _seed_route(
        run_doc,
        [
            _stop(1, "A2", "A2_r000", score=0.0, measures={"b_um": 2.0, "a": 1.0}),
            _stop(0, "A1", "A1_r000"),
        ],
    )
    stops = make_page().view().route
    # In visiting order; a score of 0.0 is a score and a stop without one has None.
    assert [(s.order, s.well, s.name) for s in stops] == [
        (0, "A1", "A1_r000"),
        (1, "A2", "A2_r000"),
    ]
    assert (stops[0].score, stops[0].measures) == (None, ())
    assert (stops[1].score, stops[1].measures) == (0.0, (("a", 1.0), ("b_um", 2.0)))
    # The positions of the first four fields are what they were: a stop built by position still reads.
    assert (stops[1].x_um, stops[1].y_um) == (1.0, 2.0)
