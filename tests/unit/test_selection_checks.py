"""What the layer refuses of a selector's choices and an orderer's order (algorithms design §5.5, #567).

Each check has its own test and its own sentence (the plan's *Decided here* 6).
The wells are ``useq``'s: B2 and B3 of a 96-well plate, round, 6400 µm, at
(9000, -9000) and (18000, -9000) µm. The camera is 4 by 3 µm a pixel, so a field
is 400 by 300 µm, and two fields exactly 400 µm apart along x touch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from smc.selection.api import Choice, SelectionError, SelectionInput
from smc.selection.checks import check_choices, check_order
from smc.selection.testing import make_input

B2 = (9000.0, -9000.0)
B3 = (18000.0, -9000.0)


def _input(**overrides: Any) -> SelectionInput:
    arguments: dict[str, Any] = {
        "wells": ("B2", "B3"),
        "field_um": (400.0, 300.0),
        "fields_per_well": 3,
    }
    arguments.update(overrides)
    return make_input([], **arguments)


def _at(well: tuple[float, float], dx: float, dy: float = 0.0) -> tuple[float, float]:
    return (well[0] + dx, well[1] + dy)


def _choice(well: str, centre: tuple[float, float], **kwargs: Any) -> Choice:
    return Choice(well, centre[0], centre[1], **kwargs)


def test_a_valid_choice_list_and_order_pass() -> None:
    data = _input()
    choices = [
        _choice("B2", _at(B2, 0.0), score=3.0, measures={"density_per_mm2": 20.0}),
        _choice("B2", _at(B2, 400.0)),  # touches the first: allowed
        _choice("B3", _at(B3, 0.0, 300.0)),
    ]

    check_choices(choices, data)
    check_choices([], data)
    check_order([2, 0, 1], 3)
    check_order([], 0)


def test_a_choice_in_a_well_the_scan_does_not_have_is_refused() -> None:
    data = _input()
    choices = [_choice("B2", B2), _choice("C4", B2)]

    with pytest.raises(
        SelectionError,
        match=r"choice 1 \(C4 at x=9000\.0, y=-9000\.0 µm\): its well is not a "
        r"well of this scan \(B2, B3\)",
    ):
        check_choices(choices, data)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["x_um", "y_um", "score", "measure nn_median_um"])
def test_a_non_finite_position_score_or_measure_is_refused(
    field: str, bad: float
) -> None:
    data = _input()
    values: dict[str, Any] = {
        "x_um": B2[0],
        "y_um": B2[1],
        "score": 1.0,
        "measures": {"nn_median_um": 10.0},
    }
    if field.startswith("measure "):
        values["measures"] = {"nn_median_um": bad}
    else:
        values[field] = bad
    choice = Choice("B2", **values)

    with pytest.raises(
        SelectionError,
        match=rf"choice 0 \(B2 at x=.*\): {field} is not finite \({bad!r}\)",
    ):
        check_choices([choice], data)


@pytest.mark.parametrize(
    ("kwargs", "field", "kind"),
    [
        ({"x_um": "9000"}, "x_um", "str"),
        ({"y_um": None}, "y_um", "NoneType"),
        ({"score": "high"}, "score", "str"),
        ({"x_um": True}, "x_um", "bool"),
        ({"measures": {"density_per_mm2": "many"}}, "measure density_per_mm2", "str"),
    ],
)
def test_a_value_that_is_not_a_number_is_refused_as_such(
    kwargs: dict[str, Any], field: str, kind: str
) -> None:
    values: dict[str, Any] = {"x_um": B2[0], "y_um": B2[1]}
    values.update(kwargs)

    with pytest.raises(
        SelectionError, match=rf"choice 0 .*: {field} is a {kind}, not a number"
    ):
        check_choices([Choice("B2", **values)], _input())


def test_measures_that_are_not_a_mapping_are_refused_in_words() -> None:
    choice = Choice("B2", *B2, measures=[("density_per_mm2", 1.0)])  # type: ignore[arg-type]

    with pytest.raises(SelectionError, match="measures is a list, not a mapping"):
        check_choices([choice], _input())


@pytest.mark.parametrize(
    "key", ["Density", "1st", "nn median", "density-per-mm2", "", "_x", "écart", 3]
)
def test_a_measure_key_that_is_not_snake_case_is_refused(key: object) -> None:
    choice = Choice("B2", *B2, measures={key: 1.0})  # type: ignore[dict-item]

    with pytest.raises(
        SelectionError,
        match=r"choice 0 .*: measure key .* is not snake_case \(a lowercase "
        r"letter, then lowercase letters, digits and underscores\)",
    ):
        check_choices([choice], _input())


def test_a_snake_case_measure_key_with_digits_and_underscores_is_accepted() -> None:
    choice = Choice("B2", *B2, measures={"a": 1.0, "nn_median_2_um": 2.0, "d3": 0.0})

    check_choices([choice], _input())


def test_a_choice_that_leaves_the_well_or_touches_the_excluded_centre_is_refused() -> (
    None
):
    # The well is round, 3200 µm in radius: a field of 400 x 300 whose centre is 3100 µm out has corners at 3300.
    outside = _choice("B2", _at(B2, 3100.0))
    on_the_centre = _choice("B2", B2)
    sentence = (
        r"the field leaves the well shrunk by wall_margin_um={m} µm or touches "
        r"the disc of exclude_centre_um={e} µm"
    )

    with pytest.raises(
        SelectionError,
        match=r"choice 0 \(B2 at x=12100\.0, y=-9000\.0 µm\): "
        + sentence.format(m="0", e="0"),
    ):
        check_choices([outside], _input())
    # The margin is part of the sentence, in `:g` form: corners at 3103.6 µm are
    # inside the wall and outside the wall shrunk by 150 µm.
    near_the_wall = _choice("B2", _at(B2, 2900.0))
    check_choices([near_the_wall], _input())
    with pytest.raises(SelectionError, match=sentence.format(m="150", e="0")):
        check_choices([near_the_wall], _input(wall_margin_um=150.0))
    with pytest.raises(SelectionError, match=sentence.format(m="0", e="50")):
        check_choices([on_the_centre], _input(exclude_centre_um=50.0))
    # The same fields, in a well they fit and with no margin and no disc, pass.
    check_choices([on_the_centre, _choice("B2", _at(B2, 2000.0))], _input())


def test_the_first_problem_of_a_choice_is_the_one_in_the_order_of_the_checks() -> None:
    # A well it is not in, a nan and a place outside: the well is read first.
    choice = Choice("C4", math.nan, 1e9, measures={"Bad": math.inf})

    with pytest.raises(SelectionError, match="its well is not a well"):
        check_choices([choice], _input())
    # The same choice in a well: the number before the key before the place.
    choice = Choice("B2", math.nan, 1e9, measures={"Bad": 1.0})
    with pytest.raises(SelectionError, match="x_um is not finite"):
        check_choices([choice], _input())
    choice = Choice("B2", 1e9, 1e9, measures={"Bad": 1.0})
    with pytest.raises(SelectionError, match="measure key 'Bad' is not snake_case"):
        check_choices([choice], _input())
    with pytest.raises(SelectionError, match="the field leaves the well"):
        check_choices([Choice("B2", 1e9, 1e9)], _input())


def test_more_choices_than_fields_per_well_in_a_well_are_refused() -> None:
    data = _input(fields_per_well=2)
    # Three in B3 (two fields a width apart and a third beside them), one in B2:
    # four in all, which fits 2 x 2, and B3 has one too many.
    three_in_b3 = [
        _choice("B3", _at(B3, 0.0, 0.0)),
        _choice("B3", _at(B3, 400.0, 0.0)),
        _choice("B3", _at(B3, 800.0, 0.0)),
    ]

    with pytest.raises(
        SelectionError, match="well B3 has 3 choices, more than fields_per_well=2"
    ):
        check_choices([_choice("B2", B2), *three_in_b3], data)
    # Two in each of the wells are what the limit allows.
    check_choices(
        [*three_in_b3[:2], _choice("B2", B2), _choice("B2", _at(B2, 400.0))], data
    )


def test_the_count_of_a_well_is_checked_before_the_overlaps_of_its_fields() -> None:
    data = _input(fields_per_well=2)
    # Three fields on top of each other: too many, and overlapping. The count is the sentence.
    stacked = [_choice("B2", _at(B2, 10.0 * k)) for k in range(3)]

    with pytest.raises(SelectionError, match="well B2 has 3 choices"):
        check_choices(stacked, data)


def test_the_total_is_checked_against_the_limit_of_every_well_before_any_choice_is_read() -> (
    None
):
    data = _input(fields_per_well=2)  # two wells: at most four choices in all

    with pytest.raises(
        SelectionError,
        match=r"5 choices in all, more than fields_per_well=2 in each of the "
        r"scan's 2 wells",
    ):
        check_choices([_choice("B2", B2)] * 5, data)
    with pytest.raises(SelectionError, match=r"scan's 1 well$"):
        check_choices([_choice("B2", B2)] * 3, make_input([], fields_per_well=2))


class _Booby:
    """An item that fails the test if anything looks at it: its attributes, its class, its repr."""

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"the item was looked at ({name})")


def test_a_huge_list_is_refused_before_any_choice_is_examined() -> None:
    data = _input(fields_per_well=3)
    huge: list[Any] = [_Booby()] * 1_000_000

    with pytest.raises(SelectionError, match="1000000 choices in all"):
        check_choices(huge, data)


@pytest.mark.parametrize(
    ("item", "kind"),
    [
        ((("B2", *B2)), "tuple"),
        ({"well": "B2", "x_um": B2[0], "y_um": B2[1]}, "dict"),
        (None, "NoneType"),
        ("B2", "str"),
        (7, "int"),
    ],
)
def test_an_item_that_is_not_a_choice_is_refused(item: object, kind: str) -> None:
    choices: list[Any] = [_choice("B2", B2), item]

    with pytest.raises(SelectionError, match=f"item 1 is a {kind}, not a Choice"):
        check_choices(choices, _input())


def test_an_object_that_looks_like_a_choice_is_not_one() -> None:
    class Lookalike:
        well, x_um, y_um, score, measures = "B2", B2[0], B2[1], None, {}

    with pytest.raises(SelectionError, match="item 0 is a Lookalike, not a Choice"):
        check_choices([Lookalike()], _input())  # type: ignore[list-item]


def test_the_types_of_all_items_are_read_before_the_first_choice_is_judged() -> None:
    # The first choice is in no well, the second is not a Choice: the type comes first.
    choices: list[Any] = [_choice("C4", B2), None]

    with pytest.raises(SelectionError, match="item 1 is a NoneType"):
        check_choices(choices, _input())


def test_two_overlapping_choices_of_a_well_are_refused_and_touching_ones_are_not() -> (
    None
):
    data = _input()
    # 400 µm wide: a width apart touches, a hair less overlaps.
    touching = [_choice("B2", _at(B2, 0.0)), _choice("B2", _at(B2, 400.0))]
    overlapping = [_choice("B2", _at(B2, 0.0)), _choice("B2", _at(B2, 399.99))]

    check_choices(touching, data)
    with pytest.raises(
        SelectionError,
        match=r"choice 0 \(B2 at x=9000\.0, y=-9000\.0 µm\): it overlaps choice 1: "
        r"two fields of one well must not overlap \(touching is allowed\)",
    ):
        check_choices(overlapping, data)
    # Along the height (300 µm) the same, and the width is not the height.
    check_choices(
        [_choice("B2", _at(B2, 0.0)), _choice("B2", _at(B2, 0.0, 300.0))], data
    )
    with pytest.raises(SelectionError, match="it overlaps choice 1"):
        check_choices(
            [_choice("B2", _at(B2, 0.0)), _choice("B2", _at(B2, 0.0, 299.99))], data
        )
    with pytest.raises(SelectionError, match="it overlaps choice 1"):
        check_choices(
            [_choice("B2", _at(B2, 0.0)), _choice("B2", _at(B2, 350.0, 0.0))], data
        )


def test_an_overlap_names_the_choices_by_their_index_in_the_whole_list() -> None:
    data = _input()
    choices = [
        _choice("B3", B3),
        _choice("B2", _at(B2, 0.0)),
        _choice("B3", _at(B3, 1000.0)),
        _choice("B2", _at(B2, 100.0)),
    ]

    with pytest.raises(
        SelectionError,
        match=r"choice 1 \(B2 at .*\): it overlaps choice 3",
    ):
        check_choices(choices, data)


def test_fields_of_two_wells_never_overlap_each_other() -> None:
    # The same offsets in two wells: one well's fields are not another's neighbours.
    data = _input()

    check_choices(
        [_choice("B2", _at(B2, 0.0)), _choice("B3", _at(B3, 0.0))],
        data,
    )


@pytest.mark.parametrize(
    ("order", "n", "message"),
    [
        ([0, 1], 3, "has 2 items, expected 3"),
        ([0, 1, 2, 3], 3, "has 4 items, expected 3"),
        ([0, 1, 1], 3, "repeats index 1"),
        ([0, 1, 5], 3, "holds 5, out of range"),
        ([0, 1, 3], 3, "holds 3, out of range"),
        ([0, -1, 2], 3, "holds -1, out of range"),
        ([0, 1.0, 2], 3, r"holds 1\.0, which is not an integer"),
        ([0, True, 2], 3, "holds True, which is not an integer"),
        ([0, None, 2], 3, "holds None, which is not an integer"),
        ([0, "1", 2], 3, "holds '1', which is not an integer"),
        ([0], 0, "has 1 items, expected 0"),
    ],
)
def test_an_order_that_is_not_a_permutation_is_refused(
    order: Sequence[object], n: int, message: str
) -> None:
    shown = f"0..{n - 1}" if n else "no index at all"

    with pytest.raises(
        SelectionError, match=f"the order is not a permutation of {shown}: {message}"
    ):
        check_order(order, n)  # type: ignore[arg-type]


def test_numpy_integers_are_indices_and_an_order_that_is_not_a_list_is_refused() -> (
    None
):
    check_order([np.int64(2), np.int32(0), np.uint8(1)], 3)
    check_order(np.array([1, 0, 2]), 3)
    check_order((2, 1, 0), 3)

    with pytest.raises(SelectionError, match="it is a NoneType, not a list of indices"):
        check_order(None, 3)  # type: ignore[arg-type]
    # An array of floats is not a list of indices, whatever its values.
    with pytest.raises(SelectionError, match="which is not an integer"):
        check_order(np.array([0.0, 1.0, 2.0]), 3)


def test_an_order_too_long_is_refused_before_its_items_are_read() -> None:
    huge: list[Any] = [_Booby()] * 1_000_000

    with pytest.raises(SelectionError, match="has 1000000 items, expected 3"):
        check_order(huge, 3)
