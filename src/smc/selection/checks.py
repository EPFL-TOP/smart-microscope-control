"""What the layer checks of a selector's choices and an orderer's order (algorithms design §5.5).

A selector and an orderer are code nobody here wrote, run inside ``route``, and
what they return becomes the positions that NIS-Elements will send the stage to.
A mistake is therefore a **refusal** that says what is wrong, never a silent
drop: a choice that is quietly left out is a field nobody imaged and nobody
knows why. ``route`` calls both checks on every result and puts ``selector
<name>:`` or ``orderer <name>:`` in front of the sentence.

Everything a choice carries is untrusted: it is a plain dataclass, so a
position can be a string, a measure's key a number, an item not a ``Choice`` at
all (FM-125), and the list can hold a million items (FM-74), so it is counted
before it is looked into. Text of a value is shown cut short and never raises.
"""

from __future__ import annotations

import math
import numbers
import operator
import re
import reprlib
from collections.abc import Sequence
from typing import SupportsIndex

from smc.plugins.route.algorithm import first_overlap
from smc.selection.api import Choice, SelectionError, SelectionInput

__all__ = ["check_choices", "check_order"]

#: A measure's key: a lowercase letter, then lowercase letters, digits and underscores.
_MEASURE_KEY = re.compile(r"[a-z][a-z0-9_]*")


def check_choices(choices: Sequence[Choice], data: SelectionInput) -> None:
    """Refuse the first thing wrong with ``choices``: the list, then each choice, then each well.

    The order of the checks is the order of the cost, and of what a person
    should read first: the number of choices (before any is looked at), the
    type of every item, then for each choice in order its well, its numbers,
    its measures' keys and its place (``data.allowed``), then the count in each
    well, and last the overlaps of each well's fields (touching is allowed).

    Raises:
        SelectionError: The first problem, in words that name the choice by its
            index, well and position.
    """
    wells = data.scan.wells
    limit = data.fields_per_well
    if len(choices) > limit * len(wells):
        raise SelectionError(
            f"{len(choices)} choices in all, more than fields_per_well={limit} "
            f"in each of the scan's {len(wells)} "
            f"well{'s' if len(wells) != 1 else ''}"
        )
    for index, item in enumerate(choices):
        if not isinstance(item, Choice):
            raise SelectionError(
                f"item {index} is a {type(item).__name__}, not a Choice"
            )

    in_well: dict[str, list[int]] = {}
    for index, choice in enumerate(choices):
        if choice.well not in wells:
            raise SelectionError(
                f"{_where(index, choice)}: its well is not a well of this scan "
                f"({', '.join(wells)})"
            )
        _check_numbers(index, choice)
        if not data.allowed(choice.x_um, choice.y_um, choice.well):
            raise SelectionError(
                f"{_where(index, choice)}: the field leaves the well shrunk by "
                f"wall_margin_um={data.wall_margin_um:g} µm or touches the disc "
                f"of exclude_centre_um={data.exclude_centre_um:g} µm"
            )
        in_well.setdefault(choice.well, []).append(index)

    for well in wells:
        held = len(in_well.get(well, ()))
        if held > limit:
            raise SelectionError(
                f"well {well} has {held} choices, more than fields_per_well={limit}"
            )
    for well in wells:
        indices = in_well.get(well, [])
        if len(indices) < 2:
            continue
        pair = first_overlap(
            [(choices[k].x_um, choices[k].y_um) for k in indices],
            matrix=data.matrix_um_per_px,
            image_shape_px=data.image_shape_px,
        )
        if pair is not None:
            first, second = indices[pair[0]], indices[pair[1]]
            raise SelectionError(
                f"{_where(first, choices[first])}: it overlaps choice {second}: "
                f"two fields of one well must not overlap (touching is allowed)"
            )


def check_order(order: Sequence[int], n: int) -> None:
    """Refuse an ``order`` that is not a permutation of ``range(n)``.

    The length is compared before any item is read; an item is an index when
    ``operator.index`` accepts it (numpy integers do) and is not a ``bool``,
    which is an integer to Python and a mistake here.

    Raises:
        SelectionError: The first problem, left to right.
    """
    shown = f"0..{n - 1}" if n > 0 else "no index at all"
    prefix = f"the order is not a permutation of {shown}: "
    try:
        length = len(order)
    except TypeError:
        raise SelectionError(
            f"{prefix}it is a {type(order).__name__}, not a list of indices"
        ) from None
    if length != n:
        raise SelectionError(f"{prefix}has {length} items, expected {n}")
    seen: set[int] = set()
    for value in order:
        index = _as_index(value)
        if index is None:
            raise SelectionError(
                f"{prefix}holds {_shown(value)}, which is not an integer"
            )
        if not 0 <= index < n:
            raise SelectionError(f"{prefix}holds {index}, out of range")
        if index in seen:
            raise SelectionError(f"{prefix}repeats index {index}")
        seen.add(index)


def _as_index(value: object) -> int | None:
    """``value`` as an integer index, or ``None``: a ``bool`` is not one, a numpy integer is."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        return None
    return operator.index(value)


def _check_numbers(index: int, choice: Choice) -> None:
    """The position, the score and every measure of ``choice`` are finite numbers, and the measures' keys are names.

    Raises:
        SelectionError: The first that is not; a value that is not a number at
            all says so, since "not finite" would send the reader looking for a
            ``nan``.
    """
    numbers_to_check: list[tuple[str, object]] = [
        ("x_um", choice.x_um),
        ("y_um", choice.y_um),
    ]
    if choice.score is not None:
        numbers_to_check.append(("score", choice.score))
    measures = choice.measures
    if not hasattr(measures, "items"):
        raise SelectionError(
            f"{_where(index, choice)}: measures is a {type(measures).__name__}, "
            f"not a mapping of names to numbers"
        )
    items = list(measures.items())
    numbers_to_check.extend((f"measure {key}", value) for key, value in items)
    for field, value in numbers_to_check:
        if not _is_number(value):
            raise SelectionError(
                f"{_where(index, choice)}: {field} is a {type(value).__name__}, "
                f"not a number ({_shown(value)})"
            )
        if not _is_finite(value):
            raise SelectionError(
                f"{_where(index, choice)}: {field} is not finite ({_shown(value)})"
            )
    for key, _ in items:
        if not (isinstance(key, str) and _MEASURE_KEY.fullmatch(key)):
            raise SelectionError(
                f"{_where(index, choice)}: measure key {_shown(key)} is not "
                f"snake_case (a lowercase letter, then lowercase letters, digits "
                f"and underscores)"
            )


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    """Whether a real number is finite; one too large for a float is not."""
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:
        return False


def _where(index: int, choice: Choice) -> str:
    """``choice 3 (B2 at x=9010.0, y=-9020.0 µm)``, or the same with the values as they are when they are no numbers."""
    try:
        return (
            f"choice {index} ({choice.well} at x={choice.x_um:.1f}, "
            f"y={choice.y_um:.1f} µm)"
        )
    except (TypeError, ValueError):
        return (
            f"choice {index} ({_shown(choice.well)} at x={_shown(choice.x_um)}, "
            f"y={_shown(choice.y_um)})"
        )


def _shown(value: object) -> str:
    """``repr(value)``, cut short, and never an exception: a third party's value may be huge or may not print."""
    try:
        return reprlib.repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
