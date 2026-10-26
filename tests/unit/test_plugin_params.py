"""Parameters from text (m3 §1.6): what ``smc run --param`` and a UI form both post.

Every refusal is asserted by its exact text (FM-45): the model's own name and the
keys appear in the messages, so a bare word would match the wrong thing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import pytest
from pydantic import Field

from smc.plugins.api import PluginParams
from smc.plugins.errors import ParamsError
from smc.plugins.params import describe_params, parse_params


class Mode(str, Enum):
    fast = "fast"
    slow = "slow"


class Demo(PluginParams):
    must: int
    samples: int = Field(3, ge=1, le=100, description="How many readings.")
    include_z: bool = True
    interval_ms: float = Field(0.0, ge=0.0, le=10_000.0)
    label: str = "x"
    where: Path | None = None
    mode: Mode = Mode.fast
    kind: Literal["a", "b"] = "a"
    wells: tuple[str, ...] = ()
    shape: tuple[float, float] | None = None
    note: str | None = None


def _parse(*pairs: str) -> Demo:
    return parse_params(Demo, ["must=1", *pairs])


def _problems(*pairs: str) -> tuple[str, ...]:
    with pytest.raises(ParamsError) as raised:
        parse_params(Demo, list(pairs))
    return raised.value.problems


def test_scalars_are_parsed_from_text() -> None:
    params = _parse(
        "samples=3", "include_z=no", "interval_ms=1e3", "label=3", "kind=b",
        "mode=slow", "where=/tmp/x",
    )  # fmt: skip
    assert params.samples == 3
    assert params.include_z is False
    assert params.interval_ms == 1000.0
    assert params.label == "3"  # a str field keeps "3" as text
    assert params.kind == "b"
    assert params.mode is Mode.slow
    assert params.where == Path("/tmp/x")


def test_no_pairs_give_the_defaults() -> None:
    params = _parse()
    assert params == Demo(must=1)
    assert isinstance(params, Demo)


def test_a_tuple_field_takes_json() -> None:
    params = _parse('wells=["A1", "B2"]', "shape=[1.5, 2]")
    assert params.wells == ("A1", "B2")
    assert params.shape == (1.5, 2.0)


def test_a_container_given_as_plain_text_says_it_takes_json() -> None:
    assert _problems("must=1", "wells=A1,B2") == (
        """'wells' takes JSON, for example ["A1", "B2"]""",
    )


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "NaN"])
def test_nan_and_inf_are_refused(text: str) -> None:
    # FM-32: the base config makes every float finite, a bound or not.
    assert _problems("must=1", f"interval_ms={text}") == (
        "interval_ms: Input should be a finite number",
    )


def test_nan_inside_json_is_refused_too() -> None:
    assert _problems("must=1", "shape=[NaN, 1]") == (
        "shape.0: Input should be a finite number",
    )


def test_null_sets_an_optional_field_to_none() -> None:
    params = _parse("where=null", "note=null", "shape=null")
    assert (params.where, params.note, params.shape) == (None, None, None)


def test_null_is_text_for_a_field_that_does_not_admit_none() -> None:
    # `label` is a str: "null" is a label, not an absent value.
    assert _parse("label=null").label == "null"
    assert _problems("must=null") == (
        "must: Input should be a valid integer, unable to parse string as an integer",
    )


def test_a_value_may_contain_an_equals_sign_and_may_be_empty() -> None:
    assert _parse("label=a=b=c").label == "a=b=c"
    assert _parse("label=").label == ""


def test_an_item_without_a_key_is_refused() -> None:
    assert _problems("must=1", "samples", "=3") == (
        "expected key=value, got 'samples'",
        "expected key=value, got '=3'",
    )


def test_every_problem_is_reported_at_once() -> None:
    # An unknown key, a duplicate and a bad value, together.
    assert _problems(
        "must=1", "nope=1", "samples=3", "samples=4", "include_z=maybe"
    ) == (
        "unknown key 'nope' (keys: must, samples, include_z, interval_ms, label, "
        "where, mode, kind, wells, shape, note)",
        "'samples' given twice",
        "include_z: Input should be a valid boolean, unable to interpret input",
    )


def test_an_unknown_key_alone_is_refused_though_every_value_is_valid() -> None:
    # Nothing for pydantic to refuse: the key problem alone must end the parse.
    assert _problems("must=1", "smaples=3") == (
        "unknown key 'smaples' (keys: must, samples, include_z, interval_ms, "
        "label, where, mode, kind, wells, shape, note)",
    )


def test_a_duplicate_key_alone_is_refused_though_every_value_is_valid() -> None:
    assert _problems("must=1", "samples=3", "samples=4") == ("'samples' given twice",)


def test_a_required_field_that_is_missing_is_reported() -> None:
    assert _problems("samples=3") == ("must: Field required",)


def test_a_value_out_of_its_bounds_says_which() -> None:
    assert _problems("must=1", "samples=0") == (
        "samples: Input should be greater than or equal to 1",
    )


def test_a_key_with_a_json_error_is_not_also_reported_missing() -> None:
    class NeedsWells(PluginParams):
        wells: tuple[str, ...]

    with pytest.raises(ParamsError) as raised:
        parse_params(NeedsWells, ["wells=A1"])
    assert raised.value.problems == (
        """'wells' takes JSON, for example ["A1", "B2"]""",
    )


def test_a_params_error_names_the_model_and_points_at_show_params() -> None:
    with pytest.raises(ParamsError) as raised:
        parse_params(Demo, ["must=x"])
    assert raised.value.plugin == "Demo"
    assert str(raised.value) == (
        "parameters of Demo: must: Input should be a valid integer, unable to parse "
        "string as an integer. smc run Demo --show-params lists the keys"
    )


def test_describe_params_shows_type_default_constraints_and_description() -> None:
    rows = describe_params(Demo)
    assert [row[0] for row in rows] == list(Demo.model_fields)  # declaration order
    by_name = {row[0]: row for row in rows}
    assert by_name["must"] == ("must", "int", "(required)", "")
    assert by_name["samples"] == (
        "samples",
        "int",
        "3",
        "ge=1 le=100  How many readings.",
    )
    assert by_name["interval_ms"] == (
        "interval_ms",
        "float",
        "0.0",
        "ge=0.0 le=10000.0",
    )
    assert by_name["label"] == ("label", "str", "'x'", "")
    assert by_name["where"] == ("where", "Path | None", "None", "")
    assert by_name["mode"] == ("mode", "Mode", "<Mode.fast: 'fast'>", "")
    assert by_name["kind"] == ("kind", "Literal['a', 'b']", "'a'", "")
    assert by_name["wells"] == ("wells", "tuple[str, ...]", "()", "")
    assert by_name["shape"] == ("shape", "tuple[float, float] | None", "None", "")


def test_describe_params_of_a_model_without_fields_is_empty() -> None:
    class Nothing(PluginParams):
        pass

    assert describe_params(Nothing) == []
