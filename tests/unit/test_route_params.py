"""The route's parameters: what ``RouteParams`` refuses, and what ``smc run`` can give it (m3 §11, #194).

Every bound here is one that ``docs/plugins/route.md`` states, and that a value
which is not a number of fields or a distance must not get past (FM-32).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from typing import Any

import pytest
from pydantic import ConfigDict, Field, ValidationError, model_validator

from smc.plugins.errors import ParamsError
from smc.plugins.params import describe_params, format_param, parse_params
from smc.plugins.route.plugin import (
    RouteParams,
    orderer_params_of,
    selector_params_of,
)
from smc.selection import registry
from smc.selection.api import (
    Choice,
    OrdererParams,
    SelectionInput,
    SelectorParams,
)
from smc.selection.builtin import MostCellsParams

THIS = __name__


class BoundedParams(SelectorParams):
    min_density_per_mm2: float = Field(0.0, ge=0.0)


class Bounded:
    name = "bounded"
    summary = "A selector with one bounded parameter."
    Params = BoundedParams

    def select(self, data: SelectionInput, params: BoundedParams) -> Sequence[Choice]:
        return []


class AnythingParams(SelectorParams):
    model_config = ConfigDict(extra="allow")


class Anything:
    name = "anything"
    summary = "A selector that takes any key."
    Params = AnythingParams

    def select(self, data: SelectionInput, params: AnythingParams) -> Sequence[Choice]:
        return []


class RaisingParams(SelectorParams):
    @model_validator(mode="before")
    @classmethod
    def _boom(cls, data: Any) -> Any:
        raise TypeError("boom: not a ValueError")


class Raising:
    name = "raising"
    summary = "A selector whose parameters raise a TypeError."
    Params = RaisingParams

    def select(self, data: SelectionInput, params: RaisingParams) -> Sequence[Choice]:
        return []


class UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("str fails")


class UnprintableParams(SelectorParams):
    @model_validator(mode="before")
    @classmethod
    def _boom(cls, data: Any) -> Any:
        raise UnprintableError


class UnprintableSelector:
    name = "unprintable"
    summary = "A selector whose parameters raise an error that cannot be printed."
    Params = UnprintableParams

    def select(
        self, data: SelectionInput, params: UnprintableParams
    ) -> Sequence[Choice]:
        return []


class WindowParams(OrdererParams):
    window: int = Field(3, ge=1)


class Windowed:
    name = "windowed"
    summary = "An orderer with one bounded parameter."
    Params = WindowParams

    def order(
        self, choices: Sequence[Choice], data: SelectionInput, params: WindowParams
    ) -> Sequence[int]:
        return list(range(len(choices)))


@pytest.fixture
def algorithms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-ins and four algorithms of this module in place of the installed ones."""

    def builtin(name: str, target: str, group: str) -> EntryPoint:
        return EntryPoint(name, f"smc.selection.builtin:{target}", group)

    def own(name: str, target: str, group: str) -> EntryPoint:
        return EntryPoint(name, f"{THIS}:{target}", group)

    groups = {
        registry.SELECTOR_GROUP: [
            builtin("most-cells", "MostCells", registry.SELECTOR_GROUP),
            own("bounded", "Bounded", registry.SELECTOR_GROUP),
            own("anything", "Anything", registry.SELECTOR_GROUP),
            own("raising", "Raising", registry.SELECTOR_GROUP),
            own("unprintable", "UnprintableSelector", registry.SELECTOR_GROUP),
        ],
        registry.ORDERER_GROUP: [
            builtin("nearest", "Nearest", registry.ORDERER_GROUP),
            builtin("as-chosen", "AsChosen", registry.ORDERER_GROUP),
            own("windowed", "Windowed", registry.ORDERER_GROUP),
        ],
    }
    monkeypatch.setattr(registry, "entry_points", lambda *, group: list(groups[group]))


def test_the_defaults_are_the_designs() -> None:
    params = RouteParams()

    assert params.labels == ("single",)
    assert params.fields_per_well == 10
    assert params.min_cells == 1
    assert params.exclude_centre_um == 0.0
    assert params.wall_margin_um == 0.0
    assert params.objective is None
    assert params.candidate_step_fraction == 0.25
    # The algorithms are the ones m3 §11 always used, with no parameter of their own.
    assert params.selector == "most-cells"
    assert params.selector_params == {}
    assert params.orderer == "nearest"
    assert params.orderer_params == {}


def test_a_label_given_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="labels repeat 'single'"):
        RouteParams(labels=("single", "doublet", "single"))


def test_at_least_one_label_is_needed() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        RouteParams(labels=())


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("fields_per_well", 0),
        ("fields_per_well", 501),
        ("min_cells", 0),
        ("exclude_centre_um", -1.0),
        ("wall_margin_um", -1.0),
        ("candidate_step_fraction", 0.0),
        ("candidate_step_fraction", 1.0001),
        ("exclude_centre_um", math.nan),
        ("wall_margin_um", math.inf),
        ("candidate_step_fraction", math.nan),
    ],
)
def test_a_value_out_of_its_bound_is_refused(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        RouteParams(**{field: bad})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "edge"),
    [
        ("fields_per_well", 1),
        ("fields_per_well", 500),
        ("min_cells", 1),
        ("exclude_centre_um", 0.0),
        ("wall_margin_um", 0.0),
        ("candidate_step_fraction", 1.0),
        ("candidate_step_fraction", 1e-9),
    ],
)
def test_the_bound_itself_is_allowed_where_it_is_inclusive(
    field: str, edge: float
) -> None:
    assert getattr(RouteParams(**{field: edge}), field) == edge  # type: ignore[arg-type]


def test_labels_come_from_text_as_json_and_a_bare_word_is_not_json() -> None:
    assert parse_params(RouteParams, ['labels=["single","doublet"]']).labels == (
        "single",
        "doublet",
    )
    with pytest.raises(ParamsError, match="'labels' takes JSON"):
        parse_params(RouteParams, ["labels=single"])
    with pytest.raises(ParamsError, match="labels repeat 'single'"):
        parse_params(RouteParams, ['labels=["single","single"]'])


def test_every_parameter_is_described_with_its_bound() -> None:
    rows = {
        name: (default, notes)
        for name, _, default, notes in describe_params(RouteParams)
    }

    assert list(rows) == [
        "labels",
        "fields_per_well",
        "min_cells",
        "exclude_centre_um",
        "wall_margin_um",
        "objective",
        "candidate_step_fraction",
        "selector",
        "selector_params",
        "orderer",
        "orderer_params",
    ]
    assert "le=500" in rows["fields_per_well"][1]
    assert "gt=0.0 le=1.0" in rows["candidate_step_fraction"][1]
    assert all(notes.strip() for _, notes in rows.values())
    assert rows["selector"][0] == "'most-cells'"
    assert rows["selector_params"][0] == "{}"
    assert rows["orderer"][0] == "'nearest'"
    assert rows["orderer_params"][0] == "{}"
    assert "registered name" in rows["selector"][1]
    assert "registered name" in rows["orderer"][1]


def test_an_unknown_selector_or_orderer_is_a_params_error_that_lists_the_known_names(
    algorithms: None,
) -> None:
    with pytest.raises(ParamsError) as selector:
        parse_params(RouteParams, ["selector=nope"])
    with pytest.raises(ParamsError) as orderer:
        parse_params(RouteParams, ["orderer=nope"])

    assert (
        "selector: Value error, selector 'nope' is not registered; known: "
        "anything, bounded, most-cells, raising, unprintable"
    ) in str(selector.value)
    assert (
        "orderer: Value error, orderer 'nope' is not registered; known: "
        "as-chosen, nearest, windowed"
    ) in str(orderer.value)
    # The registered ones are accepted, by name, and the defaults need none.
    assert RouteParams(selector="bounded", orderer="windowed").selector == "bounded"
    assert RouteParams().orderer == "nearest"


def test_no_registered_algorithm_at_all_says_to_reinstall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A venv that was not installed again after the entry points were added (FM-112).
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [])

    with pytest.raises(ValidationError, match=r"pip install -e \."):
        RouteParams()


def test_a_bad_name_and_a_bad_bound_are_listed_together(algorithms: None) -> None:
    with pytest.raises(ParamsError) as raised:
        parse_params(RouteParams, ["selector=nope", "min_cells=0"])

    problems = raised.value.problems
    assert any(p.startswith("selector: ") and "'nope'" in p for p in problems)
    assert any(p.startswith("min_cells: ") for p in problems)


def test_selector_params_are_validated_against_the_selectors_own_model_and_name_the_field(
    algorithms: None,
) -> None:
    def problem_of(**kwargs: Any) -> str:
        with pytest.raises(ValidationError) as raised:
            RouteParams(**kwargs)
        (error,) = raised.value.errors()
        assert error["type"] == "value_error"
        return str(error["msg"])

    assert problem_of(
        selector="bounded", selector_params={"min_density_per_mm2": "dense"}
    ).startswith("Value error, selector_params: min_density_per_mm2: ")
    assert problem_of(
        selector="bounded", selector_params={"min_density_per_mm2": -1.0}
    ).startswith("Value error, selector_params: min_density_per_mm2: ")
    assert problem_of(selector="bounded", selector_params={"nope": 1}) == (
        "Value error, selector_params: nope: Extra inputs are not permitted"
    )
    # The same for the orderer, with its own prefix.
    assert problem_of(orderer="windowed", orderer_params={"window": 0}).startswith(
        "Value error, orderer_params: window: "
    )
    assert problem_of(orderer="windowed", orderer_params={"nope": 1}) == (
        "Value error, orderer_params: nope: Extra inputs are not permitted"
    )
    # An orderer that has no parameter refuses any.
    assert problem_of(orderer="nearest", orderer_params={"window": 3}) == (
        "Value error, orderer_params: window: Extra inputs are not permitted"
    )
    # And what the model accepts is accepted.
    assert RouteParams(
        selector="bounded", selector_params={"min_density_per_mm2": 20}
    ).selector_params == {"min_density_per_mm2": 20}


def test_selector_params_come_from_text_as_json_and_a_bare_word_is_not_json(
    algorithms: None,
) -> None:
    params = parse_params(
        RouteParams,
        ["selector=bounded", 'selector_params={"min_density_per_mm2": 20}'],
    )

    assert params.selector_params == {"min_density_per_mm2": 20}
    with pytest.raises(ParamsError, match="'selector_params' takes JSON"):
        parse_params(RouteParams, ["selector=bounded", "selector_params=dense"])
    with pytest.raises(ParamsError, match="'orderer_params' takes JSON"):
        parse_params(RouteParams, ["orderer_params=window"])
    # JSON that is not an object is the model's refusal, not a decoding one.
    with pytest.raises(
        ParamsError, match="orderer_params: Input should be a valid dictionary"
    ):
        parse_params(RouteParams, ["orderer_params=3"])


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nan_in_selector_params_is_refused(algorithms: None, bad: str) -> None:
    # FM-32: JSON that Python reads as a float that is not a number never reaches a selector.
    # Both name a finite number: `anything` takes any key and `windowed` takes `window`,
    # so it is the value that is refused and not an unknown key.
    with pytest.raises(ParamsError, match=r"selector_params.*finite number"):
        parse_params(
            RouteParams,
            ["selector=anything", f'selector_params={{"min_density_per_mm2": {bad}}}'],
        )
    with pytest.raises(ParamsError, match=r"orderer_params.*finite number"):
        parse_params(
            RouteParams, ["orderer=windowed", f'orderer_params={{"window": {bad}}}']
        )


def test_most_cells_keeps_min_cells_and_the_step_on_route_and_refuses_them_in_selector_params(
    algorithms: None,
) -> None:
    params = RouteParams(min_cells=3, candidate_step_fraction=0.5)

    effective = selector_params_of(params, registry.selector("most-cells"))

    assert effective == MostCellsParams(min_cells=3, candidate_step_fraction=0.5)
    for key in ("min_cells", "candidate_step_fraction"):
        with pytest.raises(
            ValidationError,
            match=r"selector_params: give min_cells and candidate_step_fraction "
            r"as route parameters: they are most-cells' parameters",
        ):
            RouteParams(selector_params={key: 0.5})
    # Any other key is not most-cells' either.
    with pytest.raises(ValidationError, match="selector_params: nope: Extra inputs"):
        RouteParams(selector_params={"nope": 1})
    # The defaults and the parameters of the page and the presets are accepted with it.
    assert RouteParams(min_cells=1, candidate_step_fraction=0.25).selector_params == {}


def test_another_selector_refuses_a_min_cells_or_a_step_that_is_not_the_default_naming_most_cells(
    algorithms: None,
) -> None:
    with pytest.raises(
        ValidationError,
        match="min_cells is a parameter of most-cells; selector 'bounded' has its "
        "own: give them in selector_params",
    ):
        RouteParams(selector="bounded", min_cells=2)
    with pytest.raises(
        ValidationError,
        match="candidate_step_fraction is a parameter of most-cells; selector "
        "'bounded' has its own",
    ):
        RouteParams(selector="bounded", candidate_step_fraction=0.5)
    # Their defaults are what the page and the presets send, and are accepted.
    ok = RouteParams(selector="bounded", min_cells=1, candidate_step_fraction=0.25)
    assert selector_params_of(ok, registry.selector("bounded")) == BoundedParams()
    # The effective parameters are the selector's own, built from selector_params.
    own = RouteParams(selector="bounded", selector_params={"min_density_per_mm2": 7})
    assert selector_params_of(own, registry.selector("bounded")) == BoundedParams(
        min_density_per_mm2=7.0
    )
    assert orderer_params_of(
        RouteParams(orderer="windowed", orderer_params={"window": 5}),
        registry.orderer("windowed"),
    ) == WindowParams(window=5)


def test_a_selectors_params_that_raise_something_else_than_a_value_error_are_still_a_params_error(
    algorithms: None,
) -> None:
    # FM-128: a `mode="before"` validator that raises a TypeError is not wrapped
    # by pydantic and would end the construction of RouteParams with a traceback.
    with pytest.raises(
        ParamsError,
        match="selector_params: TypeError: boom: not a ValueError",
    ):
        parse_params(RouteParams, ["selector=raising"])
    # FM-98: the refusal is raised outside the handler that caught the TypeError,
    # so it keeps neither that error nor its traceback (the helper every refusal goes through).
    # The helper is called directly, on parameters made without validation: through
    # `RouteParams()` the validator raises a fresh error of its own outside its handler.
    unvalidated = RouteParams.model_construct(selector="raising", selector_params={})
    with pytest.raises(ValueError, match="selector_params: TypeError: boom") as raised:
        selector_params_of(unvalidated, registry.selector("raising"))
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_params_that_raise_an_error_that_cannot_be_printed_are_still_a_params_error(
    algorithms: None,
) -> None:
    # FM-113: the text of a third party's error is built inside a handler; one whose
    # `__str__` raises must not end the construction of RouteParams.
    with pytest.raises(
        ParamsError,
        match=r"selector_params: UnprintableError: \(the message cannot be printed\)",
    ):
        parse_params(RouteParams, ["selector=unprintable"])


def test_selector_params_survive_format_param_and_parse_params(
    algorithms: None,
) -> None:
    # FM-190: a preset is the text `format_param` gives, and reads back as it was.
    given = {"window": 5, "names": ["a", "é"], "nested": {"on": True, "none": None}}
    params = RouteParams(selector="anything", selector_params=given)
    fields = RouteParams.model_fields

    text = format_param(fields["selector_params"], params.selector_params)
    again = parse_params(RouteParams, ["selector=anything", f"selector_params={text}"])

    assert again.selector_params == given
    assert format_param(fields["selector_params"], {}) == "{}"
    assert format_param(fields["orderer_params"], {"window": 3}) == '{"window": 3}'
    assert text.isascii()
