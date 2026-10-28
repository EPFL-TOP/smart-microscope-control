"""Choosing wells: the text a person types, and the order a scan visits them (m3 §7.1).

Pure selection logic over a plate's names: no stage, no camera. A wrong well
set is only found hours into an overnight run, and every convenient way of
writing one (a row, a column, a rectangle, ``all``) is easy to get subtly wrong
by hand, so the grammar is written once here, with tests.

Well names are the plate's own, taken from ``useq`` and never derived: ``A1``
to ``H12`` on a 96-well plate, and the rows of a 1536-well plate run past ``Z``
to ``AA``. Rows and columns are found in those names, never by counting
letters.

Ported from ``nikon-control``'s ``wells.py``, with one deliberate difference:
that ``parse`` skipped what it could not read ("the caller compares the count
it asked for with the count it got"), and nobody did. Here an item that cannot
be read, or a well that is not on the plate, is an error that names it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby

import numpy as np
import useq

from smc.plugins.slots.plate import normalise_well

__all__ = ["order_wells", "parse_wells", "well_indices"]

_NAME = re.compile(r"([A-Z]+)([0-9]+)")
_SEPARATORS = re.compile(r"[,\s]+")
_DIGITS = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class _Grid:
    """A plate's wells as ``useq`` names them: labels of rows and columns, and where each is."""

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    names: tuple[tuple[str, ...], ...]
    at: dict[str, tuple[int, int]]

    def row(self, label: str) -> list[str] | None:
        """The wells of the row labelled ``label`` (any case), or ``None`` when there is none."""
        want = label.upper()
        if want not in self.rows:
            return None
        return list(self.names[self.rows.index(want)])

    def column(self, label: str) -> list[str] | None:
        """The wells of the column labelled ``label`` (``03`` is ``3``), or ``None``."""
        if _DIGITS.fullmatch(label) is None:
            return None
        # By text, not `int()`, which refuses a very long run of digits.
        want = label.lstrip("0") or "0"
        if want not in self.columns:
            return None
        index = self.columns.index(want)
        return [row[index] for row in self.names]

    def rectangle(self, corner_a: str, corner_b: str) -> list[str] | None:
        """Every well of the block between two corners, in any order; ``None`` if one is not a well."""
        a, b = normalise_well(corner_a), normalise_well(corner_b)
        if a not in self.at or b not in self.at:
            return None
        row_lo, row_hi = sorted((self.at[a][0], self.at[b][0]))
        col_lo, col_hi = sorted((self.at[a][1], self.at[b][1]))
        return [
            self.names[row][col]
            for row in range(row_lo, row_hi + 1)
            for col in range(col_lo, col_hi + 1)
        ]


def _layout(plate: str) -> list[list[str]]:
    """The well names of a plate as rows of columns, from ``useq``'s own geometry.

    ``useq`` returns ``numpy.str_``, which is a ``str`` that a model or a
    comparison should not receive as one, so every name is made a plain ``str``.

    Raises:
        ValueError: ``useq`` does not know the plate.
    """
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate), a1_center_xy=(0.0, 0.0)
    )
    return [[str(name) for name in row] for row in np.asarray(plan.all_well_names)]


def _grid(plate: str) -> _Grid:
    layout = _layout(plate)
    labels: dict[str, tuple[str, str]] = {}
    for row in layout:
        for name in row:
            match = _NAME.fullmatch(name)
            if match is None:
                # `useq` changed, not the plate: say so rather than misread it.
                raise ValueError(
                    f"useq names a well {name!r} on the {plate} plate, which is "
                    "not letters then digits"
                )
            labels[name] = (match.group(1), match.group(2))
    return _Grid(
        rows=tuple(labels[row[0]][0] for row in layout),
        columns=tuple(labels[name][1] for name in layout[0]),
        names=tuple(tuple(row) for row in layout),
        at={name: (r, c) for r, row in enumerate(layout) for c, name in enumerate(row)},
    )


def parse_wells(plate: str, text: str) -> tuple[str, ...]:
    """Read a well list as a person types it: ``A1, B2-B5, A1:D6, C*, *3, all``.

    Items are separated by commas or spaces:

    - ``A1``: one well (``a01`` is ``A1``);
    - ``A1:D6`` and ``B2-B5``: the rectangle between two corners, in any order
      (so ``B2-D2`` is a run down a column; the dash is the port's spelling);
    - ``C*``: every well of row C; ``*3``: every well of column 3;
    - ``all`` (any case): every well.

    Duplicates are removed, keeping the first.

    Raises:
        ValueError: An item cannot be read, or names a well that is not on the
            plate (one error lists every such item); or nothing is selected.
    """
    grid = _grid(plate)
    picked: list[str] = []
    unreadable: list[str] = []
    for item in _SEPARATORS.split(text.strip()):
        if not item:
            continue
        found = _read(grid, item)
        if found is None:
            unreadable.append(item)
        else:
            picked.extend(found)
    if unreadable:
        shown = ", ".join(repr(item) for item in dict.fromkeys(unreadable))
        raise ValueError(f"wells not understood on a {plate} plate: {shown}")
    if not picked:
        raise ValueError("no wells selected")
    return tuple(dict.fromkeys(picked))


def _read(grid: _Grid, item: str) -> list[str] | None:
    """The wells an item names, or ``None`` when it is not one of the forms above."""
    if item.lower() == "all":
        return [name for row in grid.names for name in row]
    if ":" in item:
        corner_a, corner_b = item.split(":", 1)
        return grid.rectangle(corner_a, corner_b)
    if "-" in item and not item.startswith("-"):
        corner_a, corner_b = item.split("-", 1)
        return grid.rectangle(corner_a, corner_b)
    if item.endswith("*"):
        return grid.row(item[:-1])
    if item.startswith("*"):
        return grid.column(item[1:])
    name = normalise_well(item)
    return [name] if name in grid.at else None


def _on_plate(plate: str, grid: _Grid, wells: Iterable[str]) -> list[str]:
    """``wells`` as ``useq`` names them, or a ``ValueError`` naming those that are not on the plate."""
    names = [normalise_well(well) for well in wells]
    missing = [name for name in dict.fromkeys(names) if name not in grid.at]
    if missing:
        shown = ", ".join(repr(name) for name in missing)
        raise ValueError(f"wells not on the {plate} plate: {shown}")
    return names


def order_wells(
    plate: str, wells: Iterable[str], *, serpentine: bool = True
) -> tuple[str, ...]:
    """The order a scan visits ``wells``: plate order by rows, alternate rows reversed when ``serpentine``.

    The typing order never reaches the plan: a raster returns across the whole
    plate at the end of every row, a serpentine does not when the rows are next
    to each other. A row is reversed by its **plate index**, so row B, D, ... of
    any plate (and ``AB`` of a 1536-well one), and not by its place in the visit:
    ``A*, C*`` runs both rows left to right and drives back across the plate
    between them (the port's ``serpentine`` did the same; alternating by the
    count of rows visited is a change of the design, left to the owner).
    Duplicates are removed; the names are normalised.

    Raises:
        ValueError: A well is not on the plate; one error lists every such well.
    """
    grid = _grid(plate)
    ordered = sorted(
        dict.fromkeys(_on_plate(plate, grid, wells)), key=lambda name: grid.at[name]
    )
    if not serpentine:
        return tuple(ordered)
    visiting: list[str] = []
    for row, group in groupby(ordered, key=lambda name: grid.at[name][0]):
        names = list(group)
        visiting.extend(reversed(names) if row % 2 else names)
    return tuple(visiting)


def well_indices(plate: str, wells: Sequence[str]) -> tuple[list[int], list[int]]:
    """``useq``'s ``selected_wells`` pair for ``wells``: row indices and column indices, in the given order.

    Raises:
        ValueError: A well is not on the plate; one error lists every such well.
    """
    grid = _grid(plate)
    names = _on_plate(plate, grid, wells)
    return [grid.at[name][0] for name in names], [grid.at[name][1] for name in names]
