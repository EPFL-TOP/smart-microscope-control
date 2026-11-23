"""The page's pure helpers for choosing the algorithms of ``route`` (m5 §21.3).

Whole texts (FM-45): every sentence the page shows an operator is compared in full, so that
a rule changed in one word fails here and not on a stand. The module holds no Panel, no
Bokeh and no registry, so none of this opens a widget.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import Field

import smc.ui.selection as selection
from smc.plugins.route.plugin import RouteParams
from smc.selection.api import SelectorParams
from smc.selection.builtin import MostCellsParams
from smc.ui.selection import (
    LABEL_SUMMARY_CHARS,
    NOT_A_JSON_OBJECT,
    NOT_INSTALLED,
    ROUTE_OWN_PARAMS,
    UNKNOWN_KEYS,
    AlgorithmChoice,
    algorithm_params_text,
    option_label,
    options,
    params_from_text,
    stop_tip,
    summary_text,
)


class DensityParams(SelectorParams):
    """A selector's parameters of the kinds a collaborator writes: a bounded float and int, a tuple, a bool, a str."""

    min_density_per_mm2: float = Field(10.0, gt=0, description="Cells per mm2.")
    keep: int = Field(3, ge=1)
    tags: tuple[str, ...] = ()
    on: bool = False
    mode: str = "fast"


class NoParams(SelectorParams):
    """A selector that takes nothing."""


def _choice(
    name: str, summary: str, model: type[SelectorParams] = NoParams
) -> AlgorithmChoice:
    return AlgorithmChoice(name=name, summary=summary, params=(), model=model)


CHOICES = (
    _choice("most-cells", "Most cells first."),
    _choice("density", "Fields above a density,\n  best first.", DensityParams),
    _choice("alpha", "The first."),
)


def test_a_label_names_the_algorithm_marks_the_default_and_cuts_a_long_summary() -> (
    None
):
    assert (
        option_label("most-cells", "Most  cells\nfirst.", default=True)
        == "most-cells (default) \N{EM DASH} Most cells first."
    )
    assert (
        option_label("density", "Fields above a density.", default=False)
        == "density \N{EM DASH} Fields above a density."
    )
    long = "x" * 120
    cut = option_label("density", long, default=False)
    assert LABEL_SUMMARY_CHARS == 80
    assert cut == "density \N{EM DASH} " + "x" * 79 + "\N{HORIZONTAL ELLIPSIS}"
    # A summary of exactly the limit is not cut.
    exact = option_label("a", "y" * 80, default=False)
    assert exact == "a \N{EM DASH} " + "y" * 80


def test_options_put_the_default_first_with_an_empty_value_and_the_rest_by_name() -> (
    None
):
    shown = options(CHOICES, "most-cells")
    assert list(shown.items()) == [
        ("most-cells (default) \N{EM DASH} Most cells first.", ""),
        ("alpha \N{EM DASH} The first.", "alpha"),
        ("density \N{EM DASH} Fields above a density, best first.", "density"),
    ]


def test_options_have_no_default_entry_when_the_default_is_not_installed() -> None:
    shown = options(CHOICES[1:], "most-cells")
    assert list(shown.values()) == ["alpha", "density"]
    assert "" not in shown.values()


def test_the_summary_of_the_chosen_algorithm_is_whole_and_an_empty_value_is_the_default() -> (
    None
):
    assert summary_text("", CHOICES, "most-cells") == "Most cells first."
    assert (
        summary_text("density", CHOICES, "most-cells")
        == "Fields above a density, best first."
    )
    # Whole: the label cuts a long summary, the pane under the Select does not.
    long = _choice("long", "word " * 40)
    assert summary_text("long", (long,), "long") == ("word " * 40).strip()


def test_the_summary_of_a_name_that_is_not_installed_lists_the_installed_ones() -> None:
    assert (
        summary_text("my-selector", CHOICES, "most-cells")
        == "my-selector is not installed on this PC; installed: alpha, density, most-cells."
    )
    assert NOT_INSTALLED.format(name="x", names="a, b") == (
        "x is not installed on this PC; installed: a, b."
    )
    # An empty value whose default is not installed says the default's name.
    assert summary_text("", CHOICES[1:], "most-cells") == (
        "most-cells is not installed on this PC; installed: alpha, density."
    )


def test_algorithm_params_text_is_typed_json_of_the_filled_fields_only() -> None:
    assert algorithm_params_text(
        DensityParams, {"keep": "4", "min_density_per_mm2": "20", "mode": ""}
    ) == ('{"min_density_per_mm2": 20.0, "keep": 4}', None)
    # The five kinds of field of §21.1, in the model's own field order.
    assert algorithm_params_text(
        DensityParams,
        {
            "min_density_per_mm2": "20",
            "keep": "4",
            "tags": '["a","b"]',
            "on": "yes",
            "mode": "20",
        },
    ) == (
        '{"min_density_per_mm2": 20.0, "keep": 4, "tags": ["a", "b"], "on": true, "mode": "20"}',
        None,
    )
    # A text is stripped: a trailing space typed after a number is not a problem.
    assert algorithm_params_text(DensityParams, {"keep": " 4 "}) == (
        '{"keep": 4}',
        None,
    )


def test_algorithm_params_text_of_blank_fields_is_empty_and_has_no_problem() -> None:
    assert algorithm_params_text(DensityParams, {}) == ("", None)
    assert algorithm_params_text(
        DensityParams, {"keep": "", "mode": "   ", "tags": "\t"}
    ) == ("", None)


def test_a_value_the_model_refuses_is_kept_as_typed_with_the_problem_named() -> None:
    text, problem = algorithm_params_text(
        DensityParams, {"keep": "abc", "min_density_per_mm2": "20"}
    )
    assert text == '{"keep": "abc", "min_density_per_mm2": "20"}'
    assert (
        problem
        == "keep: Input should be a valid integer, unable to parse string as an integer"
    )
    # Every problem at once, the way the command line reports them.
    _, both = algorithm_params_text(
        DensityParams, {"keep": "0", "min_density_per_mm2": "-1"}
    )
    assert both == (
        "min_density_per_mm2: Input should be greater than 0; "
        "keep: Input should be greater than or equal to 1"
    )
    # A list that is not JSON is the model's refusal too, and stays as typed.
    text, problem = algorithm_params_text(DensityParams, {"tags": "a, b"})
    assert text == '{"tags": "a, b"}'
    assert problem == '\'tags\' takes JSON, for example ["A1", "B2"]'


def test_params_from_text_gives_each_key_as_the_text_the_cli_takes() -> None:
    fields, unknown, problem = params_from_text(
        '{"mode": "fast", "on": true, "tags": ["a", "b"], "min_density_per_mm2": 20}',
        DensityParams,
    )
    assert fields == {
        "mode": "fast",
        "on": "true",
        "tags": '["a", "b"]',
        "min_density_per_mm2": "20",
    }
    assert (unknown, problem) == ((), None)
    # A float keeps its point and a null is the word the command takes.
    fields, _, _ = params_from_text(
        '{"min_density_per_mm2": 20.5, "mode": null}', DensityParams
    )
    assert fields == {"min_density_per_mm2": "20.5", "mode": "null"}


def test_params_from_text_is_the_inverse_of_algorithm_params_text() -> None:
    typed = {"min_density_per_mm2": "20", "keep": "4", "tags": '["a", "b"]'}
    text, problem = algorithm_params_text(DensityParams, typed)
    assert problem is None
    fields, unknown, problem = params_from_text(text, DensityParams)
    assert fields == {"min_density_per_mm2": "20.0", "keep": "4", "tags": '["a", "b"]'}
    assert (unknown, problem) == ((), None)
    # A text the page wrote from a refused value reads back as it was typed.
    refused, _ = algorithm_params_text(DensityParams, {"keep": "abc"})
    assert params_from_text(refused, DensityParams)[0] == {"keep": "abc"}


def test_params_from_text_names_the_keys_the_model_lacks_and_a_text_that_is_not_an_object() -> (
    None
):
    assert params_from_text('{"keep": 4, "zzz": 1, "aaa": 2}', DensityParams) == (
        {"keep": "4"},
        ("zzz", "aaa"),
        None,
    )
    assert params_from_text("[1]", DensityParams) == (
        {},
        (),
        "a JSON object is expected",
    )
    assert params_from_text("3", DensityParams) == (
        {},
        (),
        "a JSON object is expected",
    )
    assert params_from_text("{nope", DensityParams) == (
        {},
        (),
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
    )
    assert params_from_text("", DensityParams) == ({}, (), None)
    assert params_from_text("  \n", DensityParams) == ({}, (), None)
    # Nested too deep for the parser is a text that is not JSON, not a crash.
    fields, unknown, problem = params_from_text("[" * 200_000, DensityParams)
    assert (fields, unknown) == ({}, ())
    assert problem is not None


def test_the_notes_are_plain_sentences_with_their_blanks_named() -> None:
    assert UNKNOWN_KEYS.format(
        carrier="selector_params", keys="zzz, aaa", name="density"
    ) == (
        "selector_params also holds zzz, aaa, which density does not take: "
        "Run will refuse it."
    )
    assert NOT_A_JSON_OBJECT.format(
        carrier="orderer_params", problem="a JSON object is expected"
    ) == (
        "orderer_params is not a JSON object (a JSON object is expected): "
        "Run will refuse it."
    )


def test_route_own_params_are_most_cells_parameters_and_fields_of_route_params() -> (
    None
):
    assert set(ROUTE_OWN_PARAMS) == {"most-cells"}
    own = ROUTE_OWN_PARAMS["most-cells"]
    # Every parameter of most-cells is one of route's own, and none is missing.
    assert set(own) == set(MostCellsParams.model_fields)
    # `route` refuses a value other than its default beside another selector, so the
    # defaults of the two models are the same number.
    for key in own:
        assert key in RouteParams.model_fields
        assert (
            RouteParams.model_fields[key].default
            == MostCellsParams.model_fields[key].default
        )


def test_stop_tip_says_the_name_the_visit_number_the_score_and_the_measures() -> None:
    assert (
        stop_tip(name="A1_r002", order=2, score=None, measures={})
        == "A1_r002 (visit 3)"
    )
    assert (
        stop_tip(name="A1_r002", order=2, score=0.0, measures={})
        == "A1_r002 (visit 3), score 0"
    )
    assert (
        stop_tip(name="A1_r002", order=2, score=0.823456, measures={})
        == "A1_r002 (visit 3), score 0.823"
    )
    assert (
        stop_tip(name="A1_r002", order=2, score=None, measures={"a": 1.0})
        == "A1_r002 (visit 3); a 1"
    )
    assert stop_tip(
        name="E8_r000",
        order=0,
        score=0.5,
        measures={"nn_median_um": 35.0, "density_per_mm2": 41.2},
    ) == ("E8_r000 (visit 1), score 0.5; density_per_mm2 41.2, nn_median_um 35")


def test_the_module_imports_no_panel_bokeh_or_hardware() -> None:
    source = Path(selection.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    forbidden = (
        "panel",
        "bokeh",
        "pymmcore",
        "pymmcore_plus",
        "smc.hardware",
        "smc.testing",
        "smc.selection",
    )
    offenders = sorted(
        name
        for name in imported
        if any(name == bad or name.startswith(bad + ".") for bad in forbidden)
    )
    assert offenders == []
