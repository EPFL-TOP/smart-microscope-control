"""What a selector and an orderer are given and give back (algorithms design §5.1 to §5.3).

A selector chooses the fields to image from a scan's cells; an orderer says in
what order to visit them. Both are pure: the same input gives the same answer,
nothing is read but ``data.scan.image(frame)``, and the hardware layer is out
of reach, so an algorithm developed on a laptop with a run folder behaves the
same inside ``route`` on the stand. ``route`` checks what comes back (``checks``)
before any position is written, because a position is where NIS-Elements will
send the stage.

Nothing here imports ``smc.hardware``, ``smc.testing``, ``smc.ui`` or
``pymmcore_plus``: ``tests/test_kit_imports.py`` keeps it so (design §8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from smc.data import Scan
from smc.plugins.geometry import Matrix2, fov_of
from smc.plugins.route import algorithm as _route
from smc.plugins.slots import Cell, WellGeometry

__all__ = [
    "Choice",
    "Orderer",
    "OrdererParams",
    "SelectionError",
    "SelectionInput",
    "Selector",
    "SelectorParams",
]


class SelectorParams(BaseModel):
    """Base of a selector's parameters: frozen, no unknown key, every float finite.

    Same rules as a plugin's (``PluginParams``): a misspelt ``selector_params``
    key is an error and not a default, a run's parameters are the ones recorded,
    and ``allow_inf_nan=False`` keeps a bound nobody wrote from being ``nan``
    (FM-32).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class OrdererParams(BaseModel):
    """Base of an orderer's parameters; the rules of ``SelectorParams``."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SelectionError(ValueError):
    """Data a selector or an orderer cannot use, or a choice the layer refuses.

    The text is read by the person who runs ``route``, so it names what is
    wrong and what to do about it.
    """


@dataclass(frozen=True, slots=True)
class SelectionInput:
    """Everything a selector reads of a scan, and the limits ``route`` was given.

    ``matrix_um_per_px`` and ``image_shape_px`` are the **imaging objective's**
    field, which is the scan's unless ``route`` was asked to image with another
    objective: a field chosen at 40x from a 10x scan has the 40x footprint, and a
    selector that read the scan's camera would choose fields of the wrong size
    (FM-124). ``fov_um`` is derived from them and never stored.

    ``scan.cells`` may be ``None`` (a scan whose cells were never detected);
    ``cells`` then raises instead of answering "none found", two different things.
    """

    scan: Scan
    labels: tuple[str, ...]
    fields_per_well: int
    wall_margin_um: float
    exclude_centre_um: float
    objective: str | None
    matrix_um_per_px: Matrix2
    image_shape_px: tuple[int, int]

    @property
    def fov_um(self) -> tuple[float, float]:
        """``(width, height)`` of the imaging field's footprint in stage axes, in µm."""
        return fov_of(self.matrix_um_per_px, self.image_shape_px)

    @property
    def cells(self) -> tuple[Cell, ...]:
        """The scan's cells, in the file's order.

        Raises:
            SelectionError: The scan has no cells; the text holds the reason.
        """
        cells = self.scan.cells
        if cells is None:
            raise SelectionError(
                f"this scan has no cells: {self.scan.cells_absent or 'no cells record'}"
            )
        return cells

    def allowed(self, x_um: float, y_um: float, well: str) -> bool:
        """Whether a field centred at ``(x_um, y_um)`` keeps to ``well``'s margins: inside the wall margin, away from the excluded centre.

        ``route.algorithm.allowed`` with this input's field, margin and excluded
        disc: the layer's own check (``checks.check_choices``) asks the same
        question, so a selector that filters with it is never refused for the
        reason it already filtered on.

        Raises:
            SelectionError: ``well`` is not a well of the scan.
        """
        return _route.allowed(
            x_um,
            y_um,
            self._geometry_of(well),
            matrix=self.matrix_um_per_px,
            image_shape_px=self.image_shape_px,
            exclude_centre_um=self.exclude_centre_um,
            wall_margin_um=self.wall_margin_um,
        )

    def cells_in(
        self,
        x_um: float,
        y_um: float,
        well: str,
        *,
        labels: Sequence[str] | None = None,
    ) -> tuple[Cell, ...]:
        """The cells of ``well`` and of ``labels`` (default: this input's) that a field centred at ``(x_um, y_um)`` holds, in the cells' order.

        The membership is ``route``'s own (``route.algorithm.cells_in_field``): a
        cell on the edge two touching fields share is held by both, and ``route``
        will give it to whichever of them comes first in the selector's
        choices. A test of the cell's pixel against the half-extents would
        disagree on cells exactly on an edge (FM-132).

        Raises:
            SelectionError: ``well`` is not a well of the scan, or the scan has no cells.
        """
        self._geometry_of(well)
        wanted = set(self.labels if labels is None else labels)
        of_well = [
            cell for cell in self.cells if cell.well == well and cell.label in wanted
        ]
        return _route.cells_in_field(
            of_well,
            x_um,
            y_um,
            matrix=self.matrix_um_per_px,
            image_shape_px=self.image_shape_px,
        )

    def first_field(self, well: str) -> tuple[float, float]:
        """The scan plan's first field of ``well``: where the scan started in it, and where the route's order starts from.

        Raises:
            SelectionError: The plan has no field for ``well``.
        """
        for planned in self.scan.plan.fields:
            if planned.well == well:
                return (planned.x_um, planned.y_um)
        raise SelectionError(
            f"the scan plan has no field for well {well!r}; run scan-plan again"
        )

    def _geometry_of(self, well: str) -> WellGeometry:
        geometry = self.scan.geometry.get(well)
        if geometry is None:
            raise SelectionError(
                f"well {well!r} is not a well of this scan "
                f"({', '.join(self.scan.wells)})"
            )
        return geometry


@dataclass(frozen=True, slots=True)
class Choice:
    """A field a selector chose: its centre in stage µm, with the score and the numbers it decided on.

    ``score`` is the selector's own (larger is better, any scale) and
    ``measures`` are the numbers behind the decision, keys in ``snake_case``
    with their unit (``density_per_mm2``, ``nn_median_um``). ``route`` writes
    both next to the stop and ``nis-export`` ignores them. Both are optional so
    that the default selector's route is written exactly as before they existed.
    """

    well: str
    x_um: float
    y_um: float
    score: float | None = None
    measures: Mapping[str, float] = field(default_factory=dict)


SP = TypeVar("SP", bound=SelectorParams)
OP = TypeVar("OP", bound=OrdererParams)


class Selector(Protocol[SP]):
    """Chooses the fields to image; registered under the ``smc.selectors`` entry-point group.

    ``name`` is the entry point's name (kebab-case), ``summary`` one line, and
    ``Params`` the model its parameters are validated against. ``select``
    returns the chosen fields of every well, **in the order it ranks them**:
    that is the order in which ``route`` attributes the cells that two touching
    fields share. It may choose any centre; the layer checks them.

    A selector that cannot use its input raises ``SelectionError`` with a
    sentence the owner can act on. Its parameters are the only thing that
    varies a result: a randomised method takes a ``seed``.
    """

    name: str
    summary: str
    Params: type[SP]

    def select(self, data: SelectionInput, params: SP) -> Sequence[Choice]:
        """The chosen fields of every well, best first."""
        ...


class Orderer(Protocol[OP]):
    """Orders the chosen fields; registered under the ``smc.orderers`` entry-point group.

    ``order`` returns a permutation of ``range(len(choices))``: the visiting
    order of the whole plate, across wells (it may reorder the wells).
    """

    name: str
    summary: str
    Params: type[OP]

    def order(
        self, choices: Sequence[Choice], data: SelectionInput, params: OP
    ) -> Sequence[int]:
        """The visiting order, as indices into ``choices``."""
        ...
