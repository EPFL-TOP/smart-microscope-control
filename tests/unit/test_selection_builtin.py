"""The three built-in algorithms behind the contract: route's own rule, unchanged (algorithms design §5.4, §5.8, #567).

``MostCells`` is checked against ``choose_fields`` called in the test with the
arguments the plugin always gave it, and ``Nearest`` against orders worked out
by hand on points whose distances are not ties. The cameras are turned,
mirrored and not square (FM-73) wherever a width and a height, or an axis, can
be swapped.
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest
from pydantic import ValidationError

from smc.plugins.geometry import Matrix2
from smc.plugins.route import algorithm
from smc.plugins.route.algorithm import choose_fields, order_route
from smc.plugins.slots import well_geometry
from smc.selection.api import Choice, SelectionError, SelectionInput, Selector
from smc.selection.builtin import (
    AsChosen,
    MostCells,
    MostCellsParams,
    Nearest,
    NearestParams,
)
from smc.selection.testing import make_input

# The contract's Protocol is fitted by the classes as written (mypy checks it).
_AS_A_SELECTOR: Selector[MostCellsParams] = MostCells()

CENTRES = {
    "A1": (0.0, 0.0),
    "A2": (9000.0, 0.0),
    "B1": (0.0, -9000.0),
    "B2": (9000.0, -9000.0),
    "B3": (18000.0, -9000.0),
}


def _camera(pixel_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_um * cos * sign, pixel_um * sin),
        (pixel_um * sin * sign, -pixel_um * cos),
    )


def _clusters(
    well: str, count: int, seed: int, labels: tuple[str, ...]
) -> list[tuple[str, float, float, str]]:
    """Cells in a few clusters a little wider than a field, so that more than one field is chosen."""
    generator = random.Random(seed)
    cx, cy = CENTRES[well]
    cells = []
    for _ in range(4):
        x0 = cx + generator.uniform(-1500.0, 1500.0)
        y0 = cy + generator.uniform(-1500.0, 1500.0)
        for _ in range(count):
            cells.append(
                (
                    well,
                    x0 + generator.uniform(-250.0, 250.0),
                    y0 + generator.uniform(-250.0, 250.0),
                    generator.choice(labels),
                )
            )
    return cells


def _imaging(
    data: SelectionInput, matrix: Matrix2, shape: tuple[int, int]
) -> SelectionInput:
    """The same scan imaged with another camera: the input's field is the imaging objective's, never the scan's."""
    return dataclasses.replace(data, matrix_um_per_px=matrix, image_shape_px=shape)


def test_most_cells_chooses_in_each_well_what_choose_fields_chooses() -> None:
    labels = ("single", "doublet")
    cells = [
        *_clusters("B3", 12, 1, ("single", "doublet", "debris")),
        *_clusters("B2", 12, 2, ("single", "doublet", "debris")),
    ]
    matrix = _camera(5.0, 17.0, True)
    shape = (60, 80)  # (height, width): not square
    data = _imaging(
        make_input(
            cells,
            wells=("B3", "B2"),  # the plan's order is not the alphabet's
            labels=labels,
            fields_per_well=4,
            wall_margin_um=50.0,
            exclude_centre_um=100.0,
        ),
        matrix,
        shape,
    )
    params = MostCellsParams(min_cells=2, candidate_step_fraction=0.5)

    chosen = MostCells().select(data, params)

    fov = algorithm.fov_of(matrix, shape)
    expected = []
    for well in ("B3", "B2"):
        geometry = well_geometry(data.scan.plan.plate_plan, well)
        fields = choose_fields(
            [cell for cell in data.cells if cell.well == well],
            geometry,
            labels=labels,
            matrix=matrix,
            image_shape_px=shape,
            n=4,
            min_cells=2,
            exclude_centre_um=100.0,
            wall_margin_um=50.0,
            step_um=0.5 * min(fov),
        )
        expected += [(well, f.x_um, f.y_um) for f in fields]
    # The oracle chose something in both wells, and not the same number in each.
    assert {well for well, _, _ in expected} == {"B3", "B2"}
    assert [(c.well, c.x_um, c.y_um) for c in chosen] == expected
    assert all(c.score is None and dict(c.measures) == {} for c in chosen)
    assert all(isinstance(c, Choice) for c in chosen)


def test_most_cells_counts_only_the_wanted_labels() -> None:
    # Debris alone in one cluster and singles in another: only the singles are chosen.
    cells = [
        ("B2", 9000.0, -9000.0, "debris"),
        ("B2", 9010.0, -9000.0, "debris"),
        ("B2", 9500.0, -9000.0, "single"),
    ]
    data = make_input(cells, field_um=(100.0, 100.0))

    chosen = MostCells().select(data, MostCellsParams())

    assert len(chosen) == 1
    assert chosen[0].x_um == pytest.approx(9500.0, abs=100.0)


def test_most_cells_skips_a_well_with_no_wanted_cell_even_when_its_lattice_is_too_large() -> (
    None
):
    # A lattice of a billionth of a field is far beyond the bound: a refusal for a
    # well with a cell, none for a well that has nothing to choose (and never was).
    params = MostCellsParams(candidate_step_fraction=1e-9)
    empty = make_input([("B2", 9000.0, -9000.0, "debris")], field_um=(400.0, 300.0))
    with_cell = make_input([("B2", 9000.0, -9000.0, "single")], field_um=(400.0, 300.0))

    assert MostCells().select(empty, params) == []
    with pytest.raises(
        SelectionError,
        match=r"the candidate lattice of B2 would hold more than .*; the field is "
        r"400 by 300 µm: check the scan's pixel size, or raise candidate_step_fraction",
    ):
        MostCells().select(with_cell, params)


def test_most_cells_turns_the_refusals_of_the_algorithm_into_selection_errors_in_the_plugins_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_input([("B2", 9000.0, -9000.0, "single")], field_um=(400.0, 300.0))
    monkeypatch.setattr(algorithm, "_MAX_TESTS", 1)

    with pytest.raises(
        SelectionError,
        match=r"1 cells . 56 lattice points per cell would be tested, over 1; raise "
        r"candidate_step_fraction, or choose fewer labels",
    ):
        MostCells().select(data, MostCellsParams())
    monkeypatch.undo()

    # A matrix that cannot be inverted is `choose_fields`' own ValueError, said as it is.
    singular = _imaging(data, ((1.0, 2.0), (2.0, 4.0)), (60, 80))
    with pytest.raises(SelectionError, match="the camera matrix cannot be inverted"):
        MostCells().select(singular, MostCellsParams())


def test_most_cells_of_a_scan_with_no_cells_says_so() -> None:
    data = make_input([("B2", 9000.0, -9000.0, "single")])
    scan = dataclasses.replace(data.scan, cells=None, cells_absent="no cells record")

    with pytest.raises(SelectionError, match="this scan has no cells: no cells record"):
        MostCells().select(dataclasses.replace(data, scan=scan), MostCellsParams())


@pytest.mark.parametrize(
    "bad",
    [
        {"min_cells": 0},
        {"candidate_step_fraction": 0.0},
        {"candidate_step_fraction": 1.0001},
        {"candidate_step_fraction": math.nan},
        {"fields": 3},
    ],
)
def test_most_cells_params_refuse_what_choose_fields_refuses(
    bad: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        MostCellsParams(**bad)  # type: ignore[arg-type]

    assert MostCellsParams().min_cells == 1
    assert MostCellsParams().candidate_step_fraction == 0.25
    assert MostCellsParams(candidate_step_fraction=1.0).candidate_step_fraction == 1.0


def _walk(*wells: str, **kwargs: object) -> SelectionInput:
    return make_input([], wells=wells, field_um=(100.0, 100.0), **kwargs)  # type: ignore[arg-type]


def _choices(well_and_points: list[tuple[str, float, float]]) -> list[Choice]:
    return [Choice(well, x, y) for well, x, y in well_and_points]


def test_nearest_walks_the_wells_in_the_plans_order_and_chains_from_each_wells_last_stop() -> (
    None
):
    # The plan's order is A1, A2, B2, B1: not the raster's, not the alphabet's.
    data = _walk("A1", "A2", "B2", "B1")
    choices = _choices(
        [
            ("A1", 500.0, 0.0),  # 0
            ("A1", 100.0, 0.0),  # 1
            ("A1", -200.0, 0.0),  # 2
            ("A2", 9400.0, 0.0),  # 3
            ("A2", 8800.0, 0.0),  # 4
            ("A2", 9100.0, 0.0),  # 5
            ("B2", 9000.0, -8000.0),  # 6
            ("B2", 9000.0, -9100.0),  # 7
            ("B1", -100.0, -9000.0),  # 8
            ("B1", 600.0, -9000.0),  # 9
        ]
    )

    order = Nearest().order(choices, data, NearestParams())

    # A1 from its own first field (0, 0): 1, 2, 0, ending at (500, 0). A2 from there:
    # 4 (8800) first, though its own first field, (9000, 0), is nearest to 5. B2 from
    # A2's last stop (9400, 0): 6 first, though its own centre is nearest to 7. B1
    # from B2's last stop (9000, -9100): 9 first, though its own centre is nearest to 8.
    assert order == [1, 2, 0, 4, 5, 3, 6, 7, 9, 8]


def test_a_well_with_no_choice_sets_no_reference_for_the_next_one() -> None:
    data = _walk("A1", "A2", "B1")
    two_wells = _choices(
        [
            ("A1", 400.0, 0.0),  # 0
            ("B1", 5000.0, -9000.0),  # 1
            ("B1", 0.0, -8000.0),  # 2
        ]
    )

    # B1 starts at the field nearest A1's last stop (400, 0): number 2, at 8 010 µm,
    # and not the one nearest A2's centre, which A2's empty well must not leave
    # behind as a reference: number 1.
    assert Nearest().order(two_wells, data, NearestParams()) == [0, 2, 1]

    # The first well that has a choice starts at its own first field, not at the
    # first well's: A2's choice nearest (9000, 0) is 0, and A1's centre would give 1.
    only_a2 = _choices([("A2", 8000.0, 0.0), ("A2", 9100.0, 0.0)])
    assert Nearest().order(only_a2, data, NearestParams()) == [1, 0]


def test_the_reference_of_a_well_is_the_previous_wells_last_stop_and_not_its_first_field() -> (
    None
):
    data = _walk("A1", "A2")
    # A1's two stops end at (3000, 0), far from its first field (0, 0). From (0, 0)
    # the nearest of A2's fields is (1000, 2000) (2 236 µm against 3 394); from
    # (3000, 0) it is (2400, -2400) (2 474 µm against 2 828).
    choices = _choices(
        [
            ("A1", 2000.0, 0.0),
            ("A1", 3000.0, 0.0),
            ("A2", 1000.0, 2000.0),
            ("A2", 2400.0, -2400.0),
        ]
    )

    assert Nearest().order(choices, data, NearestParams()) == [0, 1, 3, 2]


def test_the_nearest_of_two_equally_near_fields_is_the_one_listed_first() -> None:
    data = _walk("A1")

    # Both are 100 µm from the well's first field, (0, 0): the first in `choices` starts.
    assert Nearest().order(
        _choices([("A1", 100.0, 0.0), ("A1", -100.0, 0.0)]), data, NearestParams()
    ) == [0, 1]
    assert Nearest().order(
        _choices([("A1", -100.0, 0.0), ("A1", 100.0, 0.0)]), data, NearestParams()
    ) == [0, 1]
    assert (
        Nearest().order(
            _choices([("A1", 300.0, 0.0), ("A1", -100.0, 0.0), ("A1", 100.0, 0.0)]),
            data,
            NearestParams(),
        )[0]
        == 1
    )


def test_nearest_gives_a_permutation_and_keeps_each_wells_stops_together_when_the_selector_listed_them_apart() -> (
    None
):
    data = _walk("B2", "A1", "B1", "A2")
    generator = random.Random(5)
    wells = ["A1", "A2", "B1", "B2"]
    choices = []
    for _ in range(40):
        well = generator.choice(wells)
        cx, cy = CENTRES[well]
        choices.append(
            Choice(
                well,
                cx + generator.uniform(-1000.0, 1000.0),
                cy + generator.uniform(-1000.0, 1000.0),
            )
        )

    order = Nearest().order(choices, data, NearestParams())

    assert sorted(order) == list(range(40))
    # The wells come in the plan's order, each well's stops together.
    visited = [choices[i].well for i in order]
    assert [w for w, _ in _runs(visited)] == [
        w for w in data.scan.wells if w in visited
    ]
    # And each well's stops are `order_route`'s path from the field nearest the reference.
    reference = data.first_field(visited[0])
    for well, indices in _runs_with_indices(order, visited):
        listed = [i for i in range(40) if choices[i].well == well]
        points = [(choices[i].x_um, choices[i].y_um) for i in listed]
        # The nearest to the reference, written as a scan: strictly nearer replaces, so the first wins a tie.
        start = 0
        for k in range(1, len(points)):
            if math.dist(points[k], reference) < math.dist(points[start], reference):
                start = k
        assert indices == [listed[k] for k in order_route(points, start=start)]
        reference = (choices[indices[-1]].x_um, choices[indices[-1]].y_um)


def _runs(items: list[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for item in items:
        if runs and runs[-1][0] == item:
            runs[-1] = (item, runs[-1][1] + 1)
        else:
            runs.append((item, 1))
    return runs


def _runs_with_indices(
    order: list[int], visited: list[str]
) -> list[tuple[str, list[int]]]:
    grouped: list[tuple[str, list[int]]] = []
    for index, well in zip(order, visited, strict=True):
        if grouped and grouped[-1][0] == well:
            grouped[-1][1].append(index)
        else:
            grouped.append((well, [index]))
    return grouped


def test_nearest_of_a_choice_in_a_well_the_scan_does_not_have_is_told() -> None:
    data = _walk("A1")

    with pytest.raises(
        SelectionError,
        match=r"choice 1 is in well 'C3', which is not a well of this scan \(A1\)",
    ):
        Nearest().order(
            _choices([("A1", 0.0, 0.0), ("C3", 0.0, 0.0)]), data, NearestParams()
        )


def test_nearest_of_no_choice_is_the_empty_order() -> None:
    assert Nearest().order([], _walk("A1"), NearestParams()) == []


def test_as_chosen_is_the_identity() -> None:
    data = _walk("A1", "A2")
    choices = _choices([("A2", 9000.0, 0.0), ("A1", 0.0, 0.0), ("A2", 9100.0, 0.0)])

    assert AsChosen().order(choices, data, AsChosen.Params()) == [0, 1, 2]
    assert AsChosen().order([], data, AsChosen.Params()) == []


def test_the_built_ins_are_named_as_their_entry_points_are() -> None:
    assert (MostCells.name, Nearest.name, AsChosen.name) == (
        "most-cells",
        "nearest",
        "as-chosen",
    )
    assert all(
        "\n" not in cls.summary and cls.summary
        for cls in (MostCells, Nearest, AsChosen)
    )
