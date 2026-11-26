"""``route`` chooses the fields to image and the order to visit them in: by default the fields that hold the most cells of the wanted classes, then the shortest path through them (m3 §11).

The choosing and the ordering are a **selector** and an **orderer**, registered
by name in ``smc.selection`` (algorithms design §5): ``most-cells`` and ``nearest``
are m3 §11's own rule, so a route with the defaults is the route this plugin always
made, and another algorithm is a parameter and no change here. What stays in the
plugin is what the positions that NIS-Elements will visit rest on, and the layer
that checks an algorithm's answer before any position is written. The pure part of
the default rule (the candidate lattice, the greedy that counts each cell once, the
nearest-neighbour-then-2-opt ordering) is ``algorithm.py``, and the stop's Z and
autofocus offset are ``focus.py``'s. This reads the three records upstream and
moves nothing, so a dry run is no different from any other, and it reads the
stand only to ask the turret which objectives it has and what the profile says
their pixel sizes are, and, for the scan's own objective, the profile's autofocus
offset factors and offset limits (a profile read, not a device's). What it
decides:

* **The scan's camera is the camera.** The matrix and the frame's shape are the
  scan's, so a cell is in a field exactly as the scan saw it (FM-124); the plan's
  copy is not read, and the field of view is derived from the matrix and the shape
  and not taken from a stored number.
* **Another objective is a scaling, not a measurement.** The field of an objective
  that is not the scan's is the scan's matrix times the ratio of the profile's
  pixel sizes (binning 1 for both, so the ratio is free of it, FM-103). Offline,
  with no turret, the objectives are the keys of that profile map and the sizes
  its entries: the same ratio, since the core's factor divides both (algorithms
  §4.4). Nothing here knows the offset between two objectives, and the Z and the
  autofocus offset of the scan are for the scan's objective, so they are dropped
  for another one (FM-118) and the result says so in ``assumed``. A selector is
  given that objective's field, not the scan's.
* **An algorithm's answer is checked, never trusted.** A selector and an orderer
  are code nobody here wrote: what they return is refused with a sentence that
  names them and the choice when it is not a list of fields that keep to the wells
  and do not overlap, or not a permutation (``smc.selection.checks``), and an
  exception in them ends the stage ``failed`` naming them, its type and where it
  was raised (a file's base name and a line, never a path, FM-50).
* **The cells of a field are counted once, as ``choose_fields`` counts them.** Each
  wanted cell goes to the first of the selector's choices that holds it
  (``held_counts``), whoever chose them, so ``counts`` and ``uncovered`` add up to
  the wanted cells for any selector.
* **A stop's Z is a frame's the autofocus had not lost.** ``z_um`` and
  ``pfs_offset`` come together from the nearest frame of the well that is not
  ``unlocked``: on an unlocked frame the Z is wherever the search had got to and
  the offset is one that nobody read locked on this sample (FM-118), which the
  export would hand on as the operator's. A well whose every frame is unlocked
  gives ``None`` for both, and ``assumed`` counts the stops the rule changed.
* **The offset names the plane the frame was taken at** (m3 §26). A scan with the focus
  on the cells takes its frames at the cells' plane with the autofocus off, and records
  the offset of the lock it swept from. Handed on, that offset would have NIS-Elements
  engage the autofocus at the plane the autofocus held, not the cells'. So the stop's
  ``pfs_offset`` is the lock's plus the frame's distance from it times the profile's
  factor for the scan's objective, with ``offset_from_focus`` saying so; or none, with
  ``assumed`` saying why (no objective recorded, no factor, an offset outside the
  profile's limits, a distance the scan could not have swept), and the Z kept. Never a
  clamp: ``focus.py`` has each bound.
* **A size that is not a number is not a size** (FM-104): a pixel size of 0 means
  "not measured", infinity is above 0, and either is refused, not defaulted.
* **The lattice is derived from a value that may be wrong** (FM-121): a scan matrix
  1000 times too small would make a million candidates, which the algorithm
  refuses, and this names the size of the field it was derived from.
* **No route is a refusal.** A route with no stop would be a ``done`` result that
  every reader must remember to check; the stage ends ``failed`` with the reason.
* **A stop ends it between two wells.** The selector is one step for every well and
  nothing in its contract lets it see the stop, so it is looked at before the
  selector and at each well after it.
* **Nothing is written and nothing is kept.** A stop ends it and a re-run starts
  over: the work is arithmetic on records already on disk.
"""

from __future__ import annotations

import itertools
import math
import operator
import os
import sys
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from smc.data import DatasetError, Scan
from smc.hardware.capabilities import ObjectiveTurret
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import Matrix2, fov_of, pixel_of_stage
from smc.plugins.route.algorithm import held_counts, path_length_um
from smc.plugins.route.focus import StopFocus, focus_assumed, stop_focus
from smc.plugins.slots import (
    Cell,
    CellsResult,
    RoutePosition,
    RouteResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
    TimingsResult,
    load_cells,
    well_geometry,
)
from smc.selection import (
    Choice,
    Orderer,
    OrdererParams,
    SelectionError,
    SelectionInput,
    Selector,
    SelectorParams,
    check_choices,
    check_order,
    orderer_names,
    selector_names,
)
from smc.selection import orderer as load_orderer
from smc.selection import selector as load_selector
from smc.selection.builtin import MostCellsParams

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "RouteParams",
    "RoutePlugin",
    "orderer_params_of",
    "selector_params_of",
]

_NAME = "route"
#: The names of the built-ins that ``route``'s own parameters and lines of ``assumed`` speak of.
_MOST_CELLS = "most-cells"
_NEAREST = "nearest"
#: A message that comes from an algorithm is cut here: a third party's error may carry a whole
#: table, and the stage's record keeps it. The layer's own sentences are far shorter (the
#: longest lists the plan's wells).
_MAX_MESSAGE = 1000

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class RouteParams(PluginParams):
    """Which cells count, how many fields per well, what keeps a field away from the wall and the centre, and which algorithms choose and order the fields."""

    labels: tuple[str, ...] = Field(
        ("single",),
        min_length=1,
        description=(
            "The cell classes that count, JSON for more than one "
            '(labels=["single","doublet"]); each is one of the cells stage\'s classes.'
        ),
    )
    fields_per_well: int = Field(
        10, ge=1, le=2000, description="The most fields to choose in a well."
    )
    min_cells: int = Field(
        1,
        ge=1,
        description=(
            "A field must hold at least this many cells that no earlier field "
            "holds. A parameter of most-cells, kept here for the presets and the "
            "page; another selector has its own, in selector_params."
        ),
    )
    exclude_centre_um: float = Field(
        0.0,
        ge=0.0,
        description="Radius of the disc at each well's centre that no field may touch; 0 for none.",
    )
    wall_margin_um: float = Field(
        0.0,
        ge=0.0,
        description="Distance from the wall that every corner of a field keeps; 0 for none.",
    )
    objective: str | None = Field(
        None,
        description=(
            "Image with this objective, by its label on the turret; default the "
            "scan's. Its field is the scan's scaled by the profile's pixel sizes."
        ),
    )
    candidate_step_fraction: float = Field(
        0.25,
        gt=0.0,
        le=1.0,
        description=(
            "The spacing of the candidate fields, as a fraction of the field's "
            "shorter side; smaller is finer and slower. A parameter of most-cells, "
            "kept here for the presets and the page."
        ),
    )
    selector: str = Field(
        _MOST_CELLS,
        description=(
            "The algorithm that chooses the fields, by its registered name; an "
            "unknown name lists the known ones."
        ),
    )
    selector_params: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "The selector's own parameters, as JSON "
            '(selector_params={"min_density_per_mm2": 20}), validated against its '
            "model; most-cells takes none here."
        ),
    )
    orderer: str = Field(
        _NEAREST,
        description=(
            "The algorithm that orders the fields, by its registered name; an "
            "unknown name lists the known ones."
        ),
    )
    orderer_params: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "The orderer's own parameters, as JSON, validated against its model."
        ),
    )

    @field_validator("labels")
    @classmethod
    def _no_label_twice(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A label given twice would be a count that two keys share: ask for it once."""
        for label in value:
            if value.count(label) > 1:
                raise ValueError(f"labels repeat {label!r}")
        return value

    @field_validator("selector")
    @classmethod
    def _the_selector_is_registered(cls, value: str) -> str:
        """The name is one of the registered ones, read without importing any of them, so that a bad name is listed with a bad bound."""
        return _registered(value, "selector", selector_names(), load_selector)

    @field_validator("orderer")
    @classmethod
    def _the_orderer_is_registered(cls, value: str) -> str:
        """The same for the orderer's name."""
        return _registered(value, "orderer", orderer_names(), load_orderer)

    @model_validator(mode="after")
    def _the_algorithms_parameters_are_theirs(self) -> RouteParams:
        """Load the two algorithms and validate ``selector_params`` and ``orderer_params`` against their own models.

        An ``after`` validator, so that ``--param`` reports it as a
        ``ParamsError``; it does not run when a field already failed, which is
        why the names are checked by field validators and only the parameters
        are checked here. Whatever a selector's ``Params`` raises is a problem
        that starts ``selector_params:`` (FM-128).
        """
        problem = ""
        try:
            selector = load_selector(self.selector)
            orderer = load_orderer(self.orderer)
        except SelectionError as exc:
            problem = str(exc)
        else:
            try:
                selector_params_of(self, selector)
                orderer_params_of(self, orderer)
            except ValueError as exc:
                problem = str(exc)
        if problem:
            # Raised outside the handlers: nothing here keeps the algorithm's error (FM-98).
            raise ValueError(problem)
        return self


def _registered(
    name: str,
    kind: str,
    names: tuple[str, ...],
    load: Callable[[str], object],
) -> str:
    """``name`` when it is registered; otherwise a refusal that lists the known names, or says to reinstall when there are none."""
    if name in names:
        return name
    problem = ""
    if names:
        problem = f"{kind} {name!r} is not registered; known: {', '.join(names)}"
    else:
        try:
            load(name)
        except SelectionError as exc:
            problem = str(exc)
    raise ValueError(problem or f"{kind} {name!r} is not registered")


def selector_params_of(params: RouteParams, selector: Selector[Any]) -> SelectorParams:
    """The parameters ``selector`` is called with: its own model validated, with the two ``most-cells`` bounds ``route`` keeps.

    ``min_cells`` and ``candidate_step_fraction`` are ``most-cells``' parameters,
    kept on ``RouteParams`` for the presets and the page: with ``most-cells`` they
    are its parameters, and giving either in ``selector_params`` too is refused;
    with another selector a value other than their default is refused, naming
    ``most-cells``, since it would be silently ignored. The defaults are always
    accepted, so a page that sends all the fields reads right.

    Raises:
        ValueError: A problem that starts ``selector_params:`` (or names the
            bound); the first, and the same text whatever the selector.
    """
    if selector.name == _MOST_CELLS:
        taken = [
            key
            for key in ("min_cells", "candidate_step_fraction")
            if key in params.selector_params
        ]
        if taken:
            raise ValueError(
                "selector_params: give min_cells and candidate_step_fraction as "
                "route parameters: they are most-cells' parameters"
            )
        _validated(selector.Params, params.selector_params, "selector_params")
        return MostCellsParams(
            min_cells=params.min_cells,
            candidate_step_fraction=params.candidate_step_fraction,
        )
    for key in ("min_cells", "candidate_step_fraction"):
        if getattr(params, key) != RouteParams.model_fields[key].default:
            raise ValueError(
                f"{key} is a parameter of most-cells; selector "
                f"{selector.name!r} has its own: give them in selector_params"
            )
    own: type[SelectorParams] = selector.Params
    return _validated(own, params.selector_params, "selector_params")


def orderer_params_of(params: RouteParams, orderer: Orderer[Any]) -> OrdererParams:
    """The parameters ``orderer`` is called with: its own model, validated.

    Raises:
        ValueError: A problem that starts ``orderer_params:``.
    """
    own: type[OrdererParams] = orderer.Params
    return _validated(own, params.orderer_params, "orderer_params")


def _validated(model: type[M], raw: dict[str, JsonValue], field: str) -> M:
    """``model`` built from ``raw``, or a ``ValueError`` that starts ``field:`` and names the first problem.

    Whatever an algorithm's ``Params`` raises is the same refusal (FM-128): a
    ``mode="before"`` validator that raises ``TypeError`` is not wrapped by
    pydantic, and would otherwise end ``RouteParams()`` with a traceback. The
    problem is built as text and raised outside the handlers (FM-98).
    """
    problem = ""
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        problem = f"{field}: {where + ': ' if where else ''}{first['msg']}"
    except Exception as exc:
        problem = f"{field}: {type(exc).__name__}: {_one_line(exc)}"
    raise ValueError(problem)


def _one_line(error: BaseException) -> str:
    """An error's message on one line, cut short: a third party's may be a table, and may not print at all.

    The message of an algorithm's error is built inside a handler, so one whose
    ``__str__`` raises would end the stage (or a ``RouteParams``) with that
    error and not with a refusal (FM-113).
    """
    try:
        text = " ".join(str(error).split())
    except Exception:
        return "(the message cannot be printed)"
    return text if len(text) <= _MAX_MESSAGE else text[: _MAX_MESSAGE - 1] + "…"


@dataclass(frozen=True)
class _Imaging:
    """What the route images with: the objective, its matrix and what that assumed.

    ``other`` is true for an objective that is not the scan's, whose Z and
    autofocus offset are therefore not known.
    """

    objective: str | None
    matrix: Matrix2
    other: bool
    assumed: tuple[str, ...]


class RoutePlugin:
    """Chooses fields with a selector (by default, the ones that hold the most wanted cells in each well) and orders them with an orderer (by default, the shortest path)."""

    name = _NAME
    summary = (
        "Choose the fields that hold the most cells of the wanted classes in each "
        "well, then the shortest route through them; the algorithms are "
        "parameters (selector, orderer)."
    )
    Params = RouteParams
    Result = RouteResult
    requires: tuple[type, ...] = ()
    provides = "route"
    reads: tuple[str, ...] = ("scan-plan", "scan", "cells")
    reads_optional: tuple[str, ...] = ("timings",)
    #: Never drives the stand, so ``smc run route --offline`` runs it (algorithms §4).
    offline: ClassVar[bool] = True

    def run(
        self,
        microscope: Microscope,
        params: RouteParams,
        ctx: RunContext,
    ) -> RouteResult:
        """Choose the fields, then order them, then give each stop its Z and counts; raise when there is no route to give.

        Raises:
            PluginRunError: A label is not one of the cells' classes; the
                objective asked for cannot be scaled; the camera matrix cannot
                be inverted; the cells file no longer matches; the cells hold a
                well the plan has no field for; the selector or the orderer
                is not registered, raises, or gives an answer the layer
                refuses (the text names it and the choice); or no field holds a
                wanted cell.
            StopRequested: The operator stopped it, before the selector or
                between two wells after it.
        """
        plan = ctx.upstream("scan-plan", ScanPlanResult)
        scan = ctx.upstream("scan", ScanResult)
        cells_result = ctx.upstream("cells", CellsResult)

        unknown = [
            label for label in params.labels if label not in cells_result.classes
        ]
        if unknown:
            raise PluginRunError(
                _NAME,
                f"label {unknown[0]!r} is not one of the cells' classes "
                f"{list(cells_result.classes)}; give labels among them",
            )

        imaging = _imaging_with(params, scan, microscope)
        shape = scan.image_shape_px
        _require_invertible(imaging.matrix, shape)
        fov_um = fov_of(imaging.matrix, shape)

        problem = ""
        cells: tuple[Cell, ...] = ()
        try:
            cells = load_cells(ctx.run_dir, cells_result)
        except ValueError as exc:
            problem = f"{exc}; run cells again"
        if problem:
            # Raised outside the handler: nothing here keeps the file's error (FM-98).
            raise PluginRunError(_NAME, problem)

        planned = {field.well for field in plan.fields}
        listed = set(plan.wells)
        for well in cells_result.counts:
            if well not in planned:
                raise PluginRunError(
                    _NAME,
                    f"the cells hold well {well!r}, which the scan plan has no field "
                    f"for; run scan-plan, scan and cells again",
                )
            # The loop below walks `plan.wells`: a well the plan has fields for and
            # does not list would be left out with its cells, and the stage `done`.
            if well not in listed:
                raise PluginRunError(
                    _NAME,
                    f"the cells hold well {well!r}, which the scan plan does not "
                    f"list among its wells; run scan-plan, scan and cells again",
                )

        scanned = [well for well in plan.wells if well in cells_result.counts]
        _require_geometry(plan)
        dataset = _dataset_of(ctx, plan, scan, cells_result, cells)
        data = SelectionInput(
            scan=dataset,
            labels=params.labels,
            fields_per_well=params.fields_per_well,
            wall_margin_um=params.wall_margin_um,
            exclude_centre_um=params.exclude_centre_um,
            objective=imaging.objective,
            matrix_um_per_px=imaging.matrix,
            image_shape_px=shape,
        )
        selector, orderer, selector_params, orderer_params = _algorithms_of(params)

        ctx.raise_if_stopped()
        # No more than the plan can hold is read of a selector's answer: a lazy one
        # that never ends is refused at the bound and not drained (FM-74).
        most = params.fields_per_well * len(dataset.wells)
        answered = _guarded(
            "selector",
            selector.name,
            selector,
            lambda: _as_list(
                selector.select(data, selector_params),
                "select",
                most,
                f"more than {most} choices (fields_per_well={params.fields_per_well} "
                f"in each of the scan's {len(dataset.wells)} wells); its answer was "
                f"not read further",
            ),
        )
        _guarded(
            "selector", selector.name, selector, lambda: check_choices(answered, data)
        )
        # What is checked is what is used: the checked choices are copied (a
        # `measures` dict is a mapping the selector may still hold) and nothing
        # after this reads the list the selector gave (FM-125).
        choices = tuple(_frozen(choice) for choice in answered)
        by_well = _by_well(choices, scanned, selector.name)
        counts_of = _counts_of(choices, by_well, cells, params, imaging, shape)

        order: list[int] = []
        if choices:
            # The orderer is given a view of its own, immutable all the way down:
            # one that sorts, replaces or appends in place would otherwise change
            # what was checked, and the counts above were keyed on.
            view = tuple(_frozen(choice) for choice in choices)
            order = _guarded(
                "orderer",
                orderer.name,
                orderer,
                lambda: _as_list(
                    orderer.order(view, data, orderer_params),
                    "order",
                    len(choices),
                    f"the order is not a permutation of 0..{len(choices) - 1}: "
                    f"it has more than {len(choices)} items",
                ),
            )
            _guarded(
                "orderer",
                orderer.name,
                orderer,
                lambda: check_order(order, len(choices)),
            )
            # numpy integers are indices (the check accepted them): plain ones from here on.
            order = [operator.index(i) for i in order]
        place = {index: rank for rank, index in enumerate(order)}
        visiting = {
            well: [i for i in order if choices[i].well == well] for well in scanned
        }

        frames_of = scan.frame_index()
        made: dict[int, RoutePosition] = {}
        uncovered: dict[str, int] = {}
        total_wanted = 0
        nearest_unlocked = 0
        without_candidate = 0
        factors, offset_limits = _offset_profile(microscope, imaging)
        focuses: list[StopFocus] = []
        for done, well in enumerate(scanned):
            ctx.raise_if_stopped()
            wanted = sum(
                cells_result.counts[well].get(label, 0) for label in params.labels
            )
            total_wanted += wanted
            stops = visiting[well]
            uncovered[well] = wanted - sum(sum(counts_of[i].values()) for i in stops)
            taken = frames_of.get(well, [])
            candidates = _candidates(scan, taken)
            for k, index in enumerate(stops):
                choice = choices[index]
                frame = (
                    None if imaging.other else _nearest_frame(scan, candidates, choice)
                )
                focus = stop_focus(
                    frame,
                    objective=scan.objective,
                    factors=factors,
                    offset_limits=offset_limits,
                )
                focuses.append(focus)
                if not imaging.other and len(candidates) < len(taken):
                    # The rule changed this stop's answer only when the nearest
                    # of all the well's frames is the unlocked one.
                    nearest = _nearest_frame(scan, taken, choice)
                    if nearest is not None and nearest.focus == "unlocked":
                        nearest_unlocked += 1
                        if frame is None:
                            without_candidate += 1
                made[index] = RoutePosition(
                    order=place[index],
                    well=well,
                    name=f"{well}_r{k:03d}",
                    x_um=float(choice.x_um),
                    y_um=float(choice.y_um),
                    z_um=focus.z_um,
                    pfs_offset=focus.pfs_offset,
                    counts=dict(counts_of[index]),
                    offset_from_focus=focus.derived,
                    score=None if choice.score is None else float(choice.score),
                    measures={
                        key: float(value) for key, value in choice.measures.items()
                    },
                )
            ctx.progress(done + 1, len(scanned), well)

        positions = [made[index] for index in order]
        if not positions:
            raise PluginRunError(
                _NAME, _no_route(params, fov_um, total_wanted, selector.name)
            )

        points = [(p.x_um, p.y_um) for p in positions]
        timings = ctx.upstream_or_none("timings", TimingsResult)
        return RouteResult(
            objective=imaging.objective,
            matrix_um_per_px=imaging.matrix,
            image_shape_px=shape,
            fov_um=fov_um,
            labels=params.labels,
            positions=tuple(positions),
            uncovered=uncovered,
            distance_um=path_length_um(points, range(len(points))),
            travel_s=(
                sum(
                    timings.move_ms(math.dist(a, b))
                    for a, b in itertools.pairwise(points)
                )
                / 1000.0
                if timings is not None
                else None
            ),
            assumed=(
                _order_line(orderer.name),
                *imaging.assumed,
                *(
                    ()
                    if selector.name == _MOST_CELLS
                    else (_selector_line(selector.name),)
                ),
                *(
                    (
                        _unlocked_line(
                            nearest_unlocked, len(positions), without_candidate
                        ),
                    )
                    if nearest_unlocked
                    else ()
                ),
                *focus_assumed(
                    focuses,
                    total=len(positions),
                    objective=scan.objective,
                    offset_limits=offset_limits,
                ),
            ),
        )


def _order_line(orderer: str) -> str:
    """The ``assumed`` line about the visiting order: route's shortest path, or the orderer's."""
    if orderer == _NEAREST:
        return (
            "the route minimises the Euclidean distance between fields; whether "
            "the stage's move time follows it is not measured"
        )
    return (
        f"the visiting order is orderer '{orderer}''s, not route's shortest path: "
        f"distance_um is the length of that order's path, and whether the stage's "
        f"move time follows it is not measured"
    )


def _selector_line(selector: str) -> str:
    return (
        f"the fields were chosen by selector '{selector}', not by route's own rule "
        f"of most cells: its choice, its score and its measures are its own"
    )


def _no_route(
    params: RouteParams,
    fov_um: tuple[float, float],
    total_wanted: int,
    selector: str,
) -> str:
    """Why there is no stop at all: no wanted cell, or no field the algorithm chose, in the words that tell what to change."""
    labels = " or ".join(params.labels)
    if total_wanted == 0:
        return (
            f"the cells stage found no {labels} cell: there is nothing to "
            f"route; choose other labels, or check the scan and the cells"
        )
    if selector == _MOST_CELLS:
        return (
            f"no field of {fov_um[0]:.0f} by {fov_um[1]:.0f} µm holds "
            f"{params.min_cells} or more {labels} cell and keeps to the wells' "
            f"margins; lower min_cells, wall_margin_um or exclude_centre_um, or "
            f"check the scan's pixel size"
        )
    return (
        f"selector {selector} chose no field in any well ({total_wanted} {labels} "
        f"cells found); check its parameters, or choose another selector"
    )


def _require_geometry(plan: ScanPlanResult) -> None:
    """Refuse a plan with a well its own plate does not have, as a plan to run again.

    ``Scan.from_results`` takes the geometry of every well of the plan and says
    the same in its own words; this is the sentence this stage has always given
    (a well that is not on the plate is a plan that does not belong to its
    plate), asked once here so that it does not change with the loader.
    """
    problem = ""
    for well in plan.wells:
        try:
            well_geometry(plan.plate_plan, well)
        except ValueError as exc:
            problem = f"{exc}; run scan-plan again"
            break
    if problem:
        # Raised outside the handler: nothing here keeps the geometry's error (FM-98).
        raise PluginRunError(_NAME, problem)


def _dataset_of(
    ctx: RunContext,
    plan: ScanPlanResult,
    scan: ScanResult,
    cells_result: CellsResult,
    cells: tuple[Cell, ...],
) -> Scan:
    """The scan as a selection algorithm reads it, built from what ``ctx.upstream`` has already read and checked."""
    problem = ""
    try:
        return Scan.from_results(
            ctx.run_dir,
            ctx.document,
            plan=plan,
            result=scan,
            cells_result=cells_result,
            cells=cells,
        )
    except DatasetError as exc:
        problem = f"{exc}; run scan-plan, scan and cells again"
    # Raised outside the handler: nothing here keeps the loader's error (FM-98).
    raise PluginRunError(_NAME, problem)


def _algorithms_of(
    params: RouteParams,
) -> tuple[Selector[Any], Orderer[Any], SelectorParams, OrdererParams]:
    """The two algorithms by name, and the parameters each is called with.

    ``RouteParams`` has checked all of it when it was built, so this fails only
    for parameters made without validation, or an algorithm that changed
    between the two: said as a refusal of the stage all the same.
    """
    problem = ""
    try:
        selector = load_selector(params.selector)
        orderer = load_orderer(params.orderer)
        return (
            selector,
            orderer,
            selector_params_of(params, selector),
            orderer_params_of(params, orderer),
        )
    except (SelectionError, ValueError) as exc:
        problem = str(exc)
    # Raised outside the handler: nothing here keeps the registry's error (FM-98).
    raise PluginRunError(_NAME, problem)


def _guarded(kind: str, name: str, owner: object, call: Callable[[], T]) -> T:
    """``call()``, with an algorithm's failure said as a refusal of this stage that names it (``selector most-cells: ...``).

    A ``SelectionError`` is the algorithm's own sentence (or a check's) and is
    kept as it is. Any other ``Exception`` is the algorithm's bug, named by its
    type and by where it was raised: the base name of the file and the line of the
    innermost frame in ``owner``'s own module, and never a path, which says who
    is logged on (FM-50). ``KeyboardInterrupt`` and ``SystemExit`` pass. The
    sentence is built inside the handler, from the traceback, and raised outside
    it, so nothing keeps the algorithm's frames (FM-98).
    """
    problem = ""
    try:
        return call()
    except SelectionError as exc:
        problem = f"{kind} {name}: {_one_line(exc)}"
    except Exception as exc:
        problem = (
            f"{kind} {name}: {type(exc).__name__}: {_one_line(exc)}"
            f"{_raised_at(exc, owner)}"
        )
    raise PluginRunError(_NAME, problem)


def _raised_at(error: BaseException, owner: object) -> str:
    """`` (at plugin.py:42)``: where in ``owner``'s own module the error was raised, or nothing when it came from elsewhere."""
    module = sys.modules.get(type(owner).__module__)
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        return ""
    wanted = os.path.normcase(os.path.abspath(source))
    found: tuple[str, int] | None = None
    for frame, line in traceback.walk_tb(error.__traceback__):
        file = frame.f_code.co_filename
        if os.path.normcase(os.path.abspath(file)) == wanted:
            found = (os.path.basename(file), line)
    return "" if found is None else f" (at {found[0]}:{found[1]})"


def _as_list(raw: Iterable[T] | None, method: str, most: int, too_many: str) -> list[T]:
    """What an algorithm returned as a list, read up to ``most`` items and no further.

    ``None`` is refused in words, since ``list(None)`` would say "not iterable".
    A lazy answer (a generator over a far too fine lattice) is not drained to
    count it: one item past ``most`` is enough to refuse it with ``too_many``, and
    an endless one ends there too (FM-74).
    """
    if raw is None:
        raise SelectionError(f"{method} returned None, not a list")
    items = list(itertools.islice(raw, most + 1))
    if len(items) > most:
        raise SelectionError(too_many)
    return items


def _frozen(choice: Choice) -> Choice:
    """A copy of a checked choice whose ``measures`` is a read-only mapping of its own."""
    return Choice(
        choice.well,
        choice.x_um,
        choice.y_um,
        choice.score,
        MappingProxyType(dict(choice.measures)),
    )


def _by_well(
    choices: Sequence[Choice], scanned: list[str], selector: str
) -> dict[str, list[int]]:
    """The indices of the choices in each well, in the selector's order; a choice in a well the cells stage did not scan is refused, not dropped.

    A route covers the wells that have cells, and a choice in another would be a
    field that nobody asked for and nothing counts: left out without a word.
    """
    by_well: dict[str, list[int]] = {well: [] for well in scanned}
    for index, choice in enumerate(choices):
        if choice.well not in by_well:
            raise PluginRunError(
                _NAME,
                f"selector {selector}: choice {index} is in well {choice.well}, "
                f"which the cells stage did not scan ({', '.join(scanned) or 'none'}); "
                f"a route covers the wells that have cells",
            )
        by_well[choice.well].append(index)
    return by_well


def _counts_of(
    choices: Sequence[Choice],
    by_well: dict[str, list[int]],
    cells: tuple[Cell, ...],
    params: RouteParams,
    imaging: _Imaging,
    shape: tuple[int, int],
) -> dict[int, dict[str, int]]:
    """Each choice's ``counts``: the wanted cells it holds that no earlier choice of its well holds, in the selector's order."""
    cells_of: dict[str, list[Cell]] = {well: [] for well in by_well}
    for cell in cells:
        if cell.well in cells_of:
            cells_of[cell.well].append(cell)
    counts: dict[int, dict[str, int]] = {}
    for well, indices in by_well.items():
        held = held_counts(
            cells_of[well],
            [(choices[i].x_um, choices[i].y_um) for i in indices],
            labels=params.labels,
            matrix=imaging.matrix,
            image_shape_px=shape,
        )
        counts.update(zip(indices, held, strict=True))
    return counts


def _turret_sizes(
    microscope: Microscope, wanted: str
) -> tuple[list[str], Callable[[str], float]]:
    """The objectives that can be imaged with, and the pixel size of each: the turret's, or the profile's when no stand is open.

    On a stand it is the turret's non-empty labels and its ``pixel_size_um``,
    which divides the profile's size by the core's magnification factor. Offline
    there is no turret and no core: the labels are the keys of the profile's
    ``[camera] pixel_size_um`` and the size of each is its entry, ``0.0`` (unknown)
    for the others. The factor divides the size of every objective alike, so the
    ratio of two sizes, the only thing ``_imaging_with`` uses, is the same on both
    (algorithms §4.4).

    Raises:
        PluginRunError: The stand has no turret to ask (never offline).
    """
    if microscope.is_offline:
        sizes = microscope.profile.camera.pixel_size_um

        def profile_size_of(label: str) -> float:
            return sizes.get(label, 0.0)

        return [label for label in sizes if label], profile_size_of
    turret = microscope.get(ObjectiveTurret)
    if turret is None:
        raise PluginRunError(
            _NAME,
            f"the stand has no turret, so the size of {wanted!r} is unknown; "
            f"leave objective unset to image with the scan's objective",
        )

    def turret_size_of(label: str) -> float:
        return turret.pixel_size_um(label)

    return [label for label in turret.labels() if label], turret_size_of


def _imaging_with(
    params: RouteParams, scan: ScanResult, microscope: Microscope
) -> _Imaging:
    """The objective to image with and its matrix: the scan's, or the scan's scaled by the profile's sizes.

    The scan's own objective, given by name, is the scan's: no turret is asked
    and nothing is scaled or assumed. Any other needs the scan's objective (to
    scale from), the sizes of ``_turret_sizes`` (a turret, or offline the
    profile's map), a label among them and a profile size for both.
    """
    wanted = params.objective
    if wanted is None or wanted == scan.objective:
        return _Imaging(scan.objective, scan.matrix_um_per_px, False, ())
    if scan.objective is None:
        raise PluginRunError(
            _NAME,
            "the scan did not record its objective, so the size of another one "
            "cannot be scaled from it; leave objective unset",
        )
    known, size_of = _turret_sizes(microscope, wanted)
    if wanted not in known:
        place = (
            "in the profile's [camera] pixel_size_um"
            if microscope.is_offline
            else "on the turret"
        )
        names = ", ".join(known) or ("none" if microscope.is_offline else "")
        raise PluginRunError(
            _NAME,
            f"objective {wanted!r} is not {place}: {names}; give one of them, or "
            f"leave objective unset",
        )
    size = size_of(wanted)
    if not (math.isfinite(size) and size > 0.0):
        raise PluginRunError(
            _NAME,
            f"the profile has no pixel size for {wanted!r}: add it to the "
            f"profile's [camera] pixel_size_um, or leave objective unset",
        )
    scan_size = size_of(scan.objective)
    if not (math.isfinite(scan_size) and scan_size > 0.0):
        raise PluginRunError(
            _NAME,
            f"the profile has no pixel size for the scan's objective "
            f"{scan.objective!r}: add it to the profile's [camera] pixel_size_um, "
            f"or leave objective unset",
        )
    scale = size / scan_size
    (m00, m01), (m10, m11) = scan.matrix_um_per_px
    matrix: Matrix2 = (
        (m00 * scale, m01 * scale),
        (m10 * scale, m11 * scale),
    )
    return _Imaging(
        wanted,
        matrix,
        True,
        (
            f"the field of {wanted} is the scan's scaled by the profile's pixel sizes",
            "no offset between the two objectives is applied",
            f"z_um and pfs_offset are unknown for {wanted}: the scan's were "
            f"measured with {scan.objective}",
        ),
    )


def _offset_profile(
    microscope: Microscope, imaging: _Imaging
) -> tuple[dict[str, float], tuple[float, float] | None]:
    """The profile's autofocus offset factors and offset limits: what turns a frame's distance into an offset (m3 §26.4).

    Read once, and only when the route images with the scan's objective: for
    another there is no frame to read and nothing to convert. The factors are
    the profile's alone (the ``Autofocus`` capability has no read of them, and
    ``focus_by_um`` that uses them moves), read as ``goto-well`` reads the
    holder's centre. Nothing is asked of a device.
    """
    if imaging.other:
        return {}, None
    profile = microscope.profile
    return (
        dict(profile.autofocus.offset_per_um),
        profile.safety.autofocus_offset_soft_limits,
    )


def _require_invertible(matrix: Matrix2, shape: tuple[int, int]) -> None:
    """Refuse a matrix no pixel maps back through, with the refusal ``pixel_of_stage`` gives.

    The scan's matrix validates field by field, so a singular one (a run
    document edited by hand) would otherwise end as a refusal of the first well.
    """
    problem = ""
    try:
        pixel_of_stage(0.0, 0.0, 0.0, 0.0, matrix=matrix, image_shape_px=shape)
    except ValueError as exc:
        problem = str(exc)
    if problem:
        # Raised outside the handler: nothing here keeps the geometry's error (FM-98).
        raise PluginRunError(_NAME, problem)


def _candidates(scan: ScanResult, positions: list[int]) -> list[int]:
    """The frames of a well that a stop may take its Z and autofocus offset from: every one the autofocus had not lost the sample on.

    ``unlocked`` means the autofocus was on and searching, so the frame's Z is
    wherever the search had got to and its offset is the stored one that nobody
    read locked on this sample (FM-118): the export would hand it on as the
    operator's. ``off`` and ``none`` stay: the scan did not use the autofocus, so
    their Z is what the focus stage read under the operator's focus and their
    offset is ``None``, which is what the stop then says.
    """
    return [p for p in positions if scan.frames[p].focus != "unlocked"]


def _unlocked_line(n: int, total: int, without_candidate: int) -> str:
    """The ``assumed`` line for the stops whose nearest frame was ``unlocked``, said once for all of them.

    ``n`` of ``total`` stops had one as their nearest frame; ``without_candidate``
    of those have no Z and no offset because every frame of their well was
    ``unlocked``, the rest took both from the nearest frame that was not.
    """
    return (
        f"{n} of {total} stops had an unlocked frame as their nearest frame "
        f"(the autofocus had not locked on it): {n - without_candidate} took z_um "
        f"and pfs_offset from the nearest frame that was not unlocked, "
        f"{without_candidate} have none because every frame of their well was "
        f"unlocked"
    )


def _nearest_frame(
    scan: ScanResult, positions: list[int], field: Choice
) -> ScanFrame | None:
    """The frame, among ``positions``, that the stage reached nearest to ``field``; ``None`` when there are none.

    By the frame's readback, which is where the stage really was (the demo
    rounds a command); ties go to the lowest index in ``scan.frames``. It takes
    whatever indices it is given: which frames may answer is ``_candidates``'.
    """
    if not positions:
        return None
    best = min(
        positions,
        key=lambda p: (
            math.dist(
                (scan.frames[p].x_um, scan.frames[p].y_um), (field.x_um, field.y_um)
            ),
            p,
        ),
    )
    return scan.frames[best]
