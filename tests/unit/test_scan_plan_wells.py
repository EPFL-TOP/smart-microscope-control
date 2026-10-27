"""Choosing wells: the grammar and the visiting order, pure (m3 §7.1, #21).

Ported from ``nikon-control``'s ``test_wells.py``. A wrong well set is only
discovered hours into an overnight run, so the range logic is pinned here and
not trusted to a page. Unlike the port, text that cannot be read is an error
that names every item, never a silent skip.
"""

from __future__ import annotations

import itertools
import math
import re

import pytest
import useq

from smc.plugins.scan_plan.wells import (
    _layout,
    order_wells,
    parse_wells,
    well_indices,
)
from smc.plugins.slots import well_geometry

PLATE = "96-well"


def _exactly(message: str) -> str:
    """A ``match`` for the whole message: a word the plate's name could also supply is no proof (FM-45)."""
    return f"^{re.escape(message)}$"


def _centres(plate: str) -> dict[str, tuple[float, float]]:
    """Where each well is on a plate registered at the origin, from the shared geometry."""
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate), a1_center_xy=(0.0, 0.0)
    )
    return {
        well: well_geometry(plan, well).center_um
        for row in _layout(plate)
        for well in row
    }


# ------------------------------------------------------------------ names


def test_names_come_from_useq_not_from_us() -> None:
    layout = _layout(PLATE)

    assert (len(layout), len(layout[0])) == (8, 12)
    assert layout[0][0] == "A1"
    assert layout[-1][-1] == "H12"
    # `numpy.str_` is a str, but it must not reach a model or a comparison as one.
    assert all(type(name) is str for row in layout for name in row)


def test_big_plate_rows_run_past_z() -> None:
    """A 1536-well plate has 32 rows: A..Z then AA..AF. Worth not reinventing."""
    layout = _layout("1536-well")

    assert len(layout) == 32
    assert layout[25][0] == "Z1"
    assert layout[26][0] == "AA1"

    row = parse_wells("1536-well", "AA*")
    assert len(row) == 48
    assert all(well.startswith("AA") for well in row)
    # `A*` is the row A, not the 48 + 48 wells whose name starts with that letter.
    assert len(parse_wells("1536-well", "A*")) == 48
    assert parse_wells("1536-well", "AF48") == ("AF48",)


# ---------------------------------------------------------------- parsing


def test_parse_understands_the_shorthands() -> None:
    got = parse_wells(PLATE, "A1, B2-B5, D*, *12")

    assert got[0] == "A1"
    assert {"B2", "B3", "B4", "B5"} <= set(got)
    assert len([well for well in got if well.startswith("D")]) == 12
    assert "H12" in got
    assert len(got) == len(set(got)), "parse returned duplicates"


def test_parse_rectangle_with_a_colon() -> None:
    assert parse_wells(PLATE, "A1:B3") == ("A1", "A2", "A3", "B1", "B2", "B3")


def test_a_dash_and_a_colon_are_both_rectangles_in_any_corner_order() -> None:
    assert parse_wells(PLATE, "B5-B2") == parse_wells(PLATE, "B2:B5")
    assert parse_wells(PLATE, "B2:B5") == ("B2", "B3", "B4", "B5")
    # The port's behaviour: the dash is a rectangle too, so this is a column run.
    assert parse_wells(PLATE, "B2-D2") == ("B2", "C2", "D2")
    assert parse_wells(PLATE, "C3:A1") == parse_wells(PLATE, "A1-C3")
    assert len(parse_wells(PLATE, "C3:A1")) == 9


def test_parse_all_in_any_case() -> None:
    assert len(parse_wells(PLATE, "all")) == 96
    assert parse_wells(PLATE, "ALL") == parse_wells(PLATE, "All")
    assert parse_wells(PLATE, "ALL")[0] == "A1"


def test_what_a_person_types_is_normalised_to_useqs_names() -> None:
    assert parse_wells(PLATE, "a01") == ("A1",)
    assert parse_wells(PLATE, "d*") == parse_wells(PLATE, "D*")
    # A column label is compared as a number, so `*03` is column 3.
    assert parse_wells(PLATE, "*03") == parse_wells(PLATE, "*3")
    assert parse_wells(PLATE, "*3") == tuple(f"{row}3" for row in "ABCDEFGH")


def test_parse_rejects_and_lists_every_item_it_cannot_read() -> None:
    """One error that names both items, in the order typed (FM-45: the whole text)."""
    message = "wells not understood on a 96-well plate: 'Q99', 'zzz'"

    with pytest.raises(ValueError, match=_exactly(message)):
        parse_wells(PLATE, "A1, Q99, zzz, B2")


@pytest.mark.parametrize(
    "item", ["-B2", "B2-", "A1:", ":A1", "*", "**", "*x", "A1:B3:C4", "A1.5", "0"]
)
def test_malformed_items_are_refused_not_guessed(item: str) -> None:
    message = f"wells not understood on a 96-well plate: {item!r}"

    with pytest.raises(ValueError, match=_exactly(message)):
        parse_wells(PLATE, f"A1, {item}")


def test_parse_rejects_a_well_that_is_not_on_this_plate() -> None:
    message = "wells not understood on a 6-well plate: 'H12'"
    with pytest.raises(ValueError, match=_exactly(message)):
        parse_wells("6-well", "H12")

    assert parse_wells("6-well", "B3") == ("B3",)
    # A row, a column or a corner that is not there is the same refusal.
    for text in ("Z*", "*9", "A1:H12"):
        with pytest.raises(ValueError, match="wells not understood on a 6-well plate"):
            parse_wells("6-well", text)


def test_one_bad_item_among_good_ones_refuses_the_whole_selection() -> None:
    """Unlike the port, which kept `A1` and `B2` and said nothing about the rest."""
    with pytest.raises(ValueError, match="'Q99'"):
        parse_wells(PLATE, "A1, Q99, B2")


@pytest.mark.parametrize("text", ["", "   ", " , ", ",,"])
def test_an_empty_selection_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="no wells selected"):
        parse_wells(PLATE, text)


def test_duplicates_are_removed_keeping_the_first() -> None:
    assert parse_wells(PLATE, "B2, A1, B2, A1:B2") == ("B2", "A1", "A2", "B1")


# --------------------------------------------------------------- ordering


def test_ordered_follows_plate_order_not_typing_order() -> None:
    assert order_wells(PLATE, ["C3", "A1", "B2"], serpentine=False) == (
        "A1",
        "B2",
        "C3",
    )


def test_serpentine_reverses_alternate_rows() -> None:
    """A raster drives back across the whole plate at every row end."""
    wells = parse_wells(PLATE, "A1:B4")

    assert order_wells(PLATE, wells, serpentine=False) == (
        "A1",
        "A2",
        "A3",
        "A4",
        "B1",
        "B2",
        "B3",
        "B4",
    )
    assert order_wells(PLATE, wells) == (
        "A1",
        "A2",
        "A3",
        "A4",
        "B4",
        "B3",
        "B2",
        "B1",
    )


def test_serpentine_goes_by_the_row_index_not_by_the_letter() -> None:
    """Row AB of a 1536-well plate is the 28th row: odd index, reversed."""
    got = order_wells("1536-well", ["AA1", "AA2", "AB1", "AB2", "AC1"])

    assert got == ("AA1", "AA2", "AB2", "AB1", "AC1")


def test_serpentine_travel_is_shorter_than_raster() -> None:
    """The reason serpentine exists, measured rather than asserted."""
    everything = parse_wells(PLATE, "all")
    centres = _centres(PLATE)

    def travel(order: tuple[str, ...]) -> float:
        return sum(
            math.dist(centres[a], centres[b]) for a, b in itertools.pairwise(order)
        )

    serpentine = order_wells(PLATE, everything)
    raster = order_wells(PLATE, everything, serpentine=False)

    assert travel(serpentine) < travel(raster) * 0.6


def test_ordering_removes_duplicates_and_normalises_the_names() -> None:
    assert order_wells(PLATE, ["b2", "B2", "A01"], serpentine=False) == ("A1", "B2")


def test_ordering_a_well_that_is_not_on_the_plate_is_refused() -> None:
    message = "wells not on the 6-well plate: 'H12', 'Z9'"

    with pytest.raises(ValueError, match=_exactly(message)):
        order_wells("6-well", ["A1", "H12", "Z9"])


# ---------------------------------------------------------------- indices


def test_well_indices_are_useqs_selected_wells_pair_in_the_given_order() -> None:
    rows, columns = well_indices(PLATE, ["B6", "A1", "H12"])

    assert (rows, columns) == ([1, 0, 7], [5, 0, 11])
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(PLATE),
        a1_center_xy=(0.0, 0.0),
        selected_wells=(rows, columns),
    )
    assert [str(name) for name in plan.selected_well_names] == ["B6", "A1", "H12"]


def test_well_indices_refuse_a_well_that_is_not_on_the_plate() -> None:
    with pytest.raises(ValueError, match="wells not on the 96-well plate: 'I1'"):
        well_indices(PLATE, ["A1", "I1"])
