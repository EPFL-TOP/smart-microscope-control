"""Selectors and orderers: how the fields of a scan are chosen, and in what order they are visited (algorithms design §5).

A selector turns a scan's cells into fields to image; an orderer turns those
fields into a visiting order. ``route`` chooses both by name from two
entry-point groups (``smc.selectors``, ``smc.orderers``), so an algorithm
written in another package, or in a collaborator's own file, is a registration
and no change to ``route``.

Nothing in this package imports ``smc.hardware``, ``smc.testing``, ``smc.ui``
or ``pymmcore_plus`` (``tests/test_kit_imports.py``, design §8): what is written
here must load on a laptop that has no Micro-Manager. ``builtin`` (the three
algorithms ``route`` had before they were named) is not imported here: it is
loaded through its entry points, like any other.
"""

from __future__ import annotations

from smc.selection.api import (
    Choice,
    Orderer,
    OrdererParams,
    SelectionError,
    SelectionInput,
    Selector,
    SelectorParams,
)
from smc.selection.checks import check_choices, check_order
from smc.selection.registry import (
    orderer,
    orderer_names,
    orderers,
    selector,
    selector_names,
    selectors,
)

__all__ = [
    "Choice",
    "Orderer",
    "OrdererParams",
    "SelectionError",
    "SelectionInput",
    "Selector",
    "SelectorParams",
    "check_choices",
    "check_order",
    "orderer",
    "orderer_names",
    "orderers",
    "selector",
    "selector_names",
    "selectors",
]
