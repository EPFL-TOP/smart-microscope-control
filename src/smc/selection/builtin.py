"""The three algorithms ``route`` had before they were named: most cells, nearest first, as chosen (algorithms design §5.4, §5.8).

``route`` chose the fields that hold the most wanted cells in each well and then
the shortest path through them (m3 §11); this is that rule, moved behind the
contract so that another selector or orderer takes its place by name and the
default route is the same one it always was. The work is ``route.algorithm``'s
(``choose_fields``, ``order_route``) and is not repeated here: what these classes
add is the translation between the contract (``SelectionInput``, ``Choice``) and
that algorithm, and the refusals said in the words ``route`` always used.

They are loaded through their entry points like any other algorithm, so a
mistake here is found by the same registry as a collaborator's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field

from smc.plugins.route.algorithm import (
    ChosenField,
    LatticeTooLarge,
    TooMuchWork,
    choose_fields,
    order_route,
)
from smc.plugins.slots import Cell
from smc.selection.api import (
    Choice,
    OrdererParams,
    SelectionError,
    SelectionInput,
    SelectorParams,
)

__all__ = [
    "AsChosen",
    "AsChosenParams",
    "MostCells",
    "MostCellsParams",
    "Nearest",
    "NearestParams",
]


class MostCellsParams(SelectorParams):
    """How many cells a field must hold, and how fine the candidate fields are."""

    min_cells: int = Field(
        1,
        ge=1,
        description="A field must hold at least this many cells that no earlier field holds.",
    )
    candidate_step_fraction: float = Field(
        0.25,
        gt=0.0,
        le=1.0,
        description=(
            "The spacing of the candidate fields, as a fraction of the field's "
            "shorter side; smaller is finer and slower."
        ),
    )


class MostCells:
    """In each well, the fields that hold the most wanted cells that no earlier field holds, best first (m3 §11)."""

    name = "most-cells"
    summary = (
        "In each well, the fields that hold the most cells of the wanted classes, "
        "greedily, never overlapping."
    )
    Params = MostCellsParams

    def select(self, data: SelectionInput, params: MostCellsParams) -> list[Choice]:
        """Choose in the wells of the plan, in the plan's order, with ``choose_fields`` on route's candidate lattice.

        A well with no wanted cell is skipped before its lattice is built, so a
        lattice too large for the field is no refusal for a well that has
        nothing to choose, as it never was.

        Raises:
            SelectionError: The scan has no cells; the lattice is too large or
                the counting is too much work (``choose_fields``' refusals, in
                route's words); or the camera cannot be inverted.
        """
        wanted_labels = set(data.labels)
        by_well: dict[str, list[Cell]] = {well: [] for well in data.scan.wells}
        for cell in data.cells:
            if cell.label in wanted_labels and cell.well in by_well:
                by_well[cell.well].append(cell)
        fov_um = data.fov_um
        step_um = params.candidate_step_fraction * min(fov_um)
        chosen: list[Choice] = []
        for well in data.scan.wells:
            if not by_well[well]:
                continue
            fields = _fields_of(data, well, by_well[well], params, step_um, fov_um)
            chosen.extend(Choice(well, field.x_um, field.y_um) for field in fields)
        return chosen


def _fields_of(
    data: SelectionInput,
    well: str,
    cells: list[Cell],
    params: MostCellsParams,
    step_um: float,
    fov_um: tuple[float, float],
) -> tuple[ChosenField, ...]:
    """The fields ``choose_fields`` gives one well, with its refusals said in the words ``route`` used."""
    problem = ""
    fields: tuple[ChosenField, ...] = ()
    try:
        fields = choose_fields(
            cells,
            data.scan.geometry[well],
            labels=data.labels,
            matrix=data.matrix_um_per_px,
            image_shape_px=data.image_shape_px,
            n=data.fields_per_well,
            min_cells=params.min_cells,
            exclude_centre_um=data.exclude_centre_um,
            wall_margin_um=data.wall_margin_um,
            step_um=step_um,
        )
    except LatticeTooLarge as exc:
        problem = (
            f"{exc}; the field is {fov_um[0]:.3g} by {fov_um[1]:.3g} µm: check "
            f"the scan's pixel size, or raise candidate_step_fraction"
        )
    except TooMuchWork as exc:
        problem = f"{exc}; raise candidate_step_fraction, or choose fewer labels"
    except ValueError as exc:
        problem = str(exc)
    if problem:
        # Raised outside the handlers: nothing here keeps the algorithm's error (FM-98).
        raise SelectionError(problem)
    return fields


class NearestParams(OrdererParams):
    """``nearest`` takes no parameter."""


class Nearest:
    """Each well's fields by the shortest path, the wells in the plan's order, each starting where the last ended (m3 §11)."""

    name = "nearest"
    summary = (
        "Wells in the plan's order; in each, nearest neighbour then 2-opt, "
        "starting nearest the previous well's last stop."
    )
    Params = NearestParams

    def order(
        self, choices: Sequence[Choice], data: SelectionInput, params: NearestParams
    ) -> list[int]:
        """The wells of the plan in order, and in each the shortest path from the field nearest the previous well's last stop.

        The first well that has a choice starts from the field nearest its own
        first planned field, where the scan started in it; a well with no
        choice sets no reference. Ties for the nearest go to the choice that
        comes first in ``choices``, and the path is ``order_route``'s.

        Raises:
            SelectionError: A choice is in a well that is not a well of the scan
                (the layer refuses it before this is called; a test that calls
                it directly is told).
        """
        in_well: dict[str, list[int]] = {well: [] for well in data.scan.wells}
        for index, choice in enumerate(choices):
            if choice.well not in in_well:
                raise SelectionError(
                    f"choice {index} is in well {choice.well!r}, which is not a "
                    f"well of this scan ({', '.join(data.scan.wells)})"
                )
            in_well[choice.well].append(index)
        visiting: list[int] = []
        reference: tuple[float, float] | None = None
        for well in data.scan.wells:
            indices = in_well[well]
            if not indices:
                continue
            if reference is None:
                reference = data.first_field(well)
            points = [(choices[i].x_um, choices[i].y_um) for i in indices]
            start = _nearest_to(points, reference)
            visiting.extend(indices[k] for k in order_route(points, start=start))
            last = choices[visiting[-1]]
            reference = (last.x_um, last.y_um)
        return visiting


def _nearest_to(
    points: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> int:
    """The index of the point nearest ``reference``; ties go to the lowest index, the choice listed first."""
    return min(range(len(points)), key=lambda k: (math.dist(points[k], reference), k))


class AsChosenParams(OrdererParams):
    """``as-chosen`` takes no parameter."""


class AsChosen:
    """Keeps the selector's order: the identity."""

    name = "as-chosen"
    summary = "Visit the fields in the order the selector listed them."
    Params = AsChosenParams

    def order(
        self, choices: Sequence[Choice], data: SelectionInput, params: AsChosenParams
    ) -> list[int]:
        """``0..n-1``: for a selector whose own order is the route it wants."""
        return list(range(len(choices)))
