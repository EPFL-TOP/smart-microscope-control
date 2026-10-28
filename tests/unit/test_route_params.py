"""The route's parameters: what ``RouteParams`` refuses, and what ``smc run`` can give it (m3 §11, #194).

Every bound here is one that ``docs/plugins/route.md`` states, and that a value
which is not a number of fields or a distance must not get past (FM-32).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from smc.plugins.errors import ParamsError
from smc.plugins.params import describe_params, parse_params
from smc.plugins.route.plugin import RouteParams


def test_the_defaults_are_the_designs() -> None:
    params = RouteParams()

    assert params.labels == ("single",)
    assert params.fields_per_well == 10
    assert params.min_cells == 1
    assert params.exclude_centre_um == 0.0
    assert params.wall_margin_um == 0.0
    assert params.objective is None
    assert params.candidate_step_fraction == 0.25


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
    ]
    assert "le=500" in rows["fields_per_well"][1]
    assert "gt=0.0 le=1.0" in rows["candidate_step_fraction"][1]
    assert all(notes.strip() for _, notes in rows.values())
