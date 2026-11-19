"""What a ``SelectionInput`` answers to a selector (algorithms design §5.2, #567).

The wells are ``useq``'s (B2 of a 96-well plate with A1 at the origin is a round
well of 6400 µm at (9000, -9000) µm) and a field is the camera's footprint,
which these tests compute with ``footprint`` and never take from the input.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from smc.plugins.geometry import footprint
from smc.selection.api import Choice, SelectionError, SelectionInput
from smc.selection.testing import make_input

B2 = (9000.0, -9000.0)
RADIUS_UM = 3200.0


def _without_cells(data: SelectionInput, reason: str | None) -> SelectionInput:
    scan = dataclasses.replace(data.scan, cells=None, cells_absent=reason)
    return dataclasses.replace(data, scan=scan)


def test_cells_of_an_input_without_cells_raise_with_the_reason() -> None:
    data = make_input([("B2", *B2, "single")])

    with pytest.raises(SelectionError, match="this scan has no cells: no cells record"):
        assert _without_cells(data, None).cells
    with pytest.raises(SelectionError, match="this scan has no cells: stale, run it"):
        assert _without_cells(data, "stale, run it").cells
    # `cells_in` reads them the same way: "none found" and "never detected" differ.
    with pytest.raises(SelectionError, match="this scan has no cells"):
        _without_cells(data, "stale").cells_in(*B2, "B2")


def test_cells_of_an_input_are_the_scans_in_its_order() -> None:
    cells = [("B2", 9005.0, -9000.0, "single"), ("B2", 9001.0, -9001.0, "doublet")]

    data = make_input(cells)

    assert [(c.x_um, c.y_um, c.label) for c in data.cells] == [
        (9005.0, -9000.0, "single"),
        (9001.0, -9001.0, "doublet"),
    ]


def test_allowed_follows_the_wall_margin_and_the_excluded_centre() -> None:
    # A field of 100 x 100 µm, whose farthest corner is a little inside the wall.
    centre_x, centre_y = B2[0] + 3135.0, B2[1]
    kwargs = {"field_um": (100.0, 100.0)}
    open_input = make_input([], **kwargs)
    margin_input = make_input([], wall_margin_um=100.0, **kwargs)
    corners = footprint(
        centre_x,
        centre_y,
        matrix=open_input.matrix_um_per_px,
        image_shape_px=open_input.image_shape_px,
    )
    farthest = max(math.dist(corner, B2) for corner in corners)
    # The oracle: inside the wall by 14.6 µm, so the margin of 100 µm refuses it.
    assert RADIUS_UM - 100.0 < farthest < RADIUS_UM

    assert open_input.allowed(centre_x, centre_y, "B2") is True
    assert margin_input.allowed(centre_x, centre_y, "B2") is False
    # The same field 100 µm further in has its corners inside the margin too.
    assert margin_input.allowed(centre_x - 100.0, centre_y, "B2") is True

    # A field over the centre touches a disc of 50 µm; one a field away does not.
    excluded = make_input([], exclude_centre_um=50.0, **kwargs)
    assert excluded.allowed(*B2, "B2") is False
    assert excluded.allowed(B2[0] + 500.0, B2[1], "B2") is True
    assert open_input.allowed(*B2, "B2") is True


def test_allowed_follows_the_fields_own_extent_when_it_is_not_square() -> None:
    # 400 wide and 100 high: the same distance from the centre is inside along
    # the well's y axis and outside along its x axis (FM-73).
    data = make_input([], field_um=(400.0, 100.0), wall_margin_um=0.0)
    along_x = (B2[0] + 3100.0, B2[1])
    along_y = (B2[0], B2[1] + 3100.0)

    assert data.allowed(*along_x, "B2") is False
    assert data.allowed(*along_y, "B2") is True


def test_allowed_of_a_well_that_is_not_in_the_scan_names_the_wells() -> None:
    data = make_input([], wells=("B2", "B3"))

    with pytest.raises(
        SelectionError, match=r"well 'C4' is not a well of this scan \(B2, B3\)"
    ):
        data.allowed(0.0, 0.0, "C4")


def test_cells_in_filters_by_well_and_by_labels_and_defaults_to_the_inputs_labels() -> (
    None
):
    cells = [
        ("B2", 9000.0, -9000.0, "single"),
        ("B2", 9010.0, -9010.0, "doublet"),
        ("B2", 9020.0, -9020.0, "debris"),
        ("B3", 18000.0, -9000.0, "single"),
        ("B2", 9030.0, -9030.0, "single"),
    ]
    data = make_input(
        cells, wells=("B2", "B3"), labels=("single", "doublet"), field_um=(400.0, 300.0)
    )

    def seen(held: tuple[object, ...]) -> list[tuple[float, str]]:
        return [(c.x_um, c.label) for c in held]  # type: ignore[attr-defined]

    # By default the input's labels, so debris is left out, and so is the other well.
    assert seen(data.cells_in(*B2, "B2")) == [
        (9000.0, "single"),
        (9010.0, "doublet"),
        (9030.0, "single"),
    ]
    assert seen(data.cells_in(*B2, "B2", labels=("debris",))) == [(9020.0, "debris")]
    assert seen(data.cells_in(*B2, "B2", labels=("single", "debris"))) == [
        (9000.0, "single"),
        (9020.0, "debris"),
        (9030.0, "single"),
    ]
    assert seen(data.cells_in(18000.0, -9000.0, "B3")) == [(18000.0, "single")]
    # A field of another place holds none of them.
    assert data.cells_in(B2[0] + 1000.0, B2[1], "B2") == ()


def test_cells_in_holds_a_cell_on_the_edge_and_follows_the_width_and_the_height() -> (
    None
):
    # A field of 400 x 300 µm: half a width is 200 µm and half a height 150 µm.
    cells = [
        ("B2", B2[0] + 200.0, B2[1], "single"),  # on the right edge: held
        ("B2", B2[0] + 200.01, B2[1], "single"),  # a hair beyond: not held
        ("B2", B2[0], B2[1] + 150.0, "single"),  # on the top edge: held
        ("B2", B2[0], B2[1] + 150.01, "single"),  # beyond: not held
        (
            "B2",
            B2[0] - 190.0,
            B2[1],
            "single",
        ),  # inside a width, outside a height: held
        ("B2", B2[0], B2[1] - 190.0, "single"),  # the other way: not held
    ]
    data = make_input(cells, field_um=(400.0, 300.0))

    held = data.cells_in(*B2, "B2")

    assert [(c.x_um - B2[0], c.y_um - B2[1]) for c in held] == [
        (200.0, 0.0),
        (0.0, 150.0),
        (-190.0, 0.0),
    ]


def test_the_input_is_judged_by_the_imaging_objectives_field_and_not_the_scans() -> (
    None
):
    # The scan was taken at 4 x 3 µm a pixel (a field of 400 x 300 µm); the route
    # images at twice the magnification, so its field is 200 x 150 µm (FM-124).
    scanned = make_input(
        [("B2", B2[0] + 150.0, B2[1], "single")], field_um=(400.0, 300.0)
    )
    imaging = dataclasses.replace(scanned, matrix_um_per_px=((2.0, 0.0), (0.0, -1.5)))

    assert imaging.scan.matrix_um_per_px == ((4.0, 0.0), (0.0, -3.0))
    assert imaging.fov_um == (200.0, 150.0)
    assert len(scanned.cells_in(*B2, "B2")) == 1
    # 150 µm along x is inside half of 400, and outside half of 200.
    assert imaging.cells_in(*B2, "B2") == ()
    # The same field next to the wall: 200 x 150 fits where 400 x 300 does not.
    near_wall = (B2[0] + 3000.0, B2[1])
    assert scanned.allowed(*near_wall, "B2") is False
    assert imaging.allowed(*near_wall, "B2") is True


def test_cells_in_of_a_well_that_is_not_in_the_scan_is_refused_and_not_empty() -> None:
    data = make_input([("B2", *B2, "single")])

    with pytest.raises(
        SelectionError, match=r"well 'B3' is not a well of this scan \(B2\)"
    ):
        data.cells_in(0.0, 0.0, "B3")


def test_first_field_of_a_well_with_no_planned_field_is_refused_in_words() -> None:
    data = make_input([], wells=("B2", "B3"))
    fields = tuple(f for f in data.scan.plan.fields if f.well != "B3")
    plan = data.scan.plan.model_copy(update={"fields": fields})
    data = dataclasses.replace(data, scan=dataclasses.replace(data.scan, plan=plan))

    assert data.first_field("B2") == pytest.approx(B2)
    with pytest.raises(
        SelectionError,
        match="the scan plan has no field for well 'B3'; run scan-plan again",
    ):
        data.first_field("B3")


def test_first_field_is_the_first_of_the_wells_fields_in_the_plans_order() -> None:
    data = make_input([], wells=("B2",))
    plan = data.scan.plan
    first, *_ = plan.fields
    later = first.model_copy(update={"index": 1, "name": "B2_0001", "x_um": 1.0})
    plan = plan.model_copy(update={"fields": (first, later)})
    data = dataclasses.replace(data, scan=dataclasses.replace(data.scan, plan=plan))

    assert data.first_field("B2") == (first.x_um, first.y_um)


def test_a_choice_has_no_score_and_no_measures_unless_given() -> None:
    plain = Choice("B2", 1.0, 2.0)
    scored = Choice("B2", 1.0, 2.0, score=0.5, measures={"density_per_mm2": 12.0})

    assert (plain.score, dict(plain.measures)) == (None, {})
    assert (scored.score, dict(scored.measures)) == (0.5, {"density_per_mm2": 12.0})
    # Two default choices never share a measures dict.
    assert plain.measures is not Choice("B2", 1.0, 2.0).measures
