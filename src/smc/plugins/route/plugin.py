"""``route`` chooses the fields that hold the most cells of the wanted classes, then the shortest path through them (m3 §11).

The pure part (the candidate lattice, the greedy that counts each cell once, the
nearest-neighbour-then-2-opt ordering) is ``algorithm.py``. This reads the three
records upstream and moves nothing, so a dry run is no different from any other,
and it reads the stand only to ask the turret which objectives it has and what
the profile says their pixel sizes are. What it decides, because the positions
that NIS-Elements will visit rest on it:

* **The scan's camera is the camera.** The matrix and the frame's shape are the
  scan's, so a cell is in a field exactly as the scan saw it (FM-124); the plan's
  copy is not read, and the field of view is derived from the matrix and the shape
  and not taken from a stored number.
* **Another objective is a scaling, not a measurement.** The field of an objective
  that is not the scan's is the scan's matrix times the ratio of the profile's
  pixel sizes (binning 1 for both, so the ratio is free of it, FM-103). Nothing
  here knows the offset between two objectives, and the Z and the autofocus offset
  of the scan are for the scan's objective, so they are dropped for another one
  (FM-118) and the result says so in ``assumed``.
* **A stop's Z is a frame's the autofocus had not lost.** ``z_um`` and
  ``pfs_offset`` come together from the nearest frame of the well that is not
  ``unlocked``: on an unlocked frame the Z is wherever the search had got to and
  the offset is one that nobody read locked on this sample (FM-118), which the
  export would hand on as the operator's. A well whose every frame is unlocked
  gives ``None`` for both, and ``assumed`` counts the stops the rule changed.
* **A size that is not a number is not a size** (FM-104): a pixel size of 0 means
  "not measured", infinity is above 0, and either is refused, not defaulted.
* **The lattice is derived from a value that may be wrong** (FM-121): a scan matrix
  1000 times too small would make a million candidates, which the algorithm
  refuses, and this names the size of the field it was derived from.
* **No route is a refusal.** A route with no stop would be a ``done`` result that
  every reader must remember to check; the stage ends ``failed`` with the reason.
* **Nothing is written and nothing is kept.** A stop ends it and a re-run starts
  over: the work is arithmetic on records already on disk.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from smc.hardware.capabilities import ObjectiveTurret
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import Matrix2, pixel_of_stage
from smc.plugins.route.algorithm import (
    ChosenField,
    LatticeTooLarge,
    TooMuchWork,
    choose_fields,
    fov_of,
    order_route,
    path_length_um,
)
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

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["RouteParams", "RoutePlugin"]

_NAME = "route"


class RouteParams(PluginParams):
    """Which cells count, how many fields per well, and what keeps a field away from the wall and the centre."""

    labels: tuple[str, ...] = Field(
        ("single",),
        min_length=1,
        description=(
            "The cell classes that count, JSON for more than one "
            '(labels=["single","doublet"]); each is one of the cells stage\'s classes.'
        ),
    )
    fields_per_well: int = Field(
        10, ge=1, le=500, description="The most fields to choose in a well."
    )
    min_cells: int = Field(
        1,
        ge=1,
        description="A field must hold at least this many cells that no earlier field holds.",
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
            "shorter side; smaller is finer and slower."
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
    """Chooses fields that hold the most wanted cells in each well, and the shortest path through them."""

    name = _NAME
    summary = (
        "Choose the fields that hold the most cells of the wanted classes in each "
        "well, then the shortest route through them."
    )
    Params = RouteParams
    Result = RouteResult
    requires: tuple[type, ...] = ()
    provides = "route"
    reads: tuple[str, ...] = ("scan-plan", "scan", "cells")
    reads_optional: tuple[str, ...] = ("timings",)

    def run(
        self,
        microscope: Microscope,
        params: RouteParams,
        ctx: RunContext,
    ) -> RouteResult:
        """Choose the fields well by well, then order them; raise when there is no route to give.

        Raises:
            PluginRunError: A label is not one of the cells' classes; the
                objective asked for cannot be scaled; the camera matrix cannot
                be inverted; the cells file no longer matches; the cells hold a
                well the plan has no field for; the candidate lattice or the
                counting is too large; or no field holds a wanted cell.
            StopRequested: The operator stopped it, between two wells.
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
        step_um = params.candidate_step_fraction * min(fov_um)

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
        frames_of = scan.frame_index()
        positions: list[RoutePosition] = []
        uncovered: dict[str, int] = {}
        previous: tuple[float, float] | None = None
        total_wanted = 0
        nearest_unlocked = 0
        without_candidate = 0
        for done, well in enumerate(scanned):
            ctx.raise_if_stopped()
            wanted = sum(
                cells_result.counts[well].get(label, 0) for label in params.labels
            )
            total_wanted += wanted
            chosen: tuple[ChosenField, ...] = ()
            if wanted:
                chosen = self._choose(
                    plan, well, cells, params, imaging, shape, fov_um, step_um
                )
            uncovered[well] = wanted - sum(sum(f.counts.values()) for f in chosen)
            if chosen:
                reference = (
                    previous if previous is not None else _first_field_of(plan, well)
                )
                stops = _stops(chosen, reference)
                taken = frames_of.get(well, [])
                candidates = _candidates(scan, taken)
                for k, index in enumerate(stops):
                    field = chosen[index]
                    frame = (
                        None
                        if imaging.other
                        else _nearest_frame(scan, candidates, field)
                    )
                    if not imaging.other and len(candidates) < len(taken):
                        # The rule changed this stop's answer only when the nearest
                        # of all the well's frames is the unlocked one.
                        nearest = _nearest_frame(scan, taken, field)
                        if nearest is not None and nearest.focus == "unlocked":
                            nearest_unlocked += 1
                            if frame is None:
                                without_candidate += 1
                    positions.append(
                        RoutePosition(
                            order=len(positions),
                            well=well,
                            name=f"{well}_r{k:03d}",
                            x_um=field.x_um,
                            y_um=field.y_um,
                            z_um=frame.z_um if frame is not None else None,
                            pfs_offset=frame.pfs_offset if frame is not None else None,
                            counts=dict(field.counts),
                        )
                    )
                previous = (positions[-1].x_um, positions[-1].y_um)
            ctx.progress(done + 1, len(scanned), well)

        labels = " or ".join(params.labels)
        if not positions:
            if total_wanted == 0:
                raise PluginRunError(
                    _NAME,
                    f"the cells stage found no {labels} cell: there is nothing to "
                    f"route; choose other labels, or check the scan and the cells",
                )
            raise PluginRunError(
                _NAME,
                f"no field of {fov_um[0]:.0f} by {fov_um[1]:.0f} µm holds "
                f"{params.min_cells} or more {labels} cell and keeps to the wells' "
                f"margins; lower min_cells, wall_margin_um or exclude_centre_um, or "
                f"check the scan's pixel size",
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
                "the route minimises the Euclidean distance between fields; whether "
                "the stage's move time follows it is not measured",
                *imaging.assumed,
                *(
                    (
                        _unlocked_line(
                            nearest_unlocked, len(positions), without_candidate
                        ),
                    )
                    if nearest_unlocked
                    else ()
                ),
            ),
        )

    def _choose(
        self,
        plan: ScanPlanResult,
        well: str,
        cells: tuple[Cell, ...],
        params: RouteParams,
        imaging: _Imaging,
        shape: tuple[int, int],
        fov_um: tuple[float, float],
        step_um: float,
    ) -> tuple[ChosenField, ...]:
        """The fields of one well, with the refusals of the algorithm said in the stage's words."""
        problem = ""
        chosen: tuple[ChosenField, ...] = ()
        try:
            geometry = well_geometry(plan.plate_plan, well)
        except ValueError as exc:
            problem = f"{exc}; run scan-plan again"
        else:
            try:
                chosen = choose_fields(
                    cells,
                    geometry,
                    labels=params.labels,
                    matrix=imaging.matrix,
                    image_shape_px=shape,
                    n=params.fields_per_well,
                    min_cells=params.min_cells,
                    exclude_centre_um=params.exclude_centre_um,
                    wall_margin_um=params.wall_margin_um,
                    step_um=step_um,
                )
            except LatticeTooLarge as exc:
                problem = (
                    f"{exc}; the field is {fov_um[0]:.3g} by {fov_um[1]:.3g} µm: "
                    f"check the scan's pixel size, or raise candidate_step_fraction"
                )
            except TooMuchWork as exc:
                problem = (
                    f"{exc}; raise candidate_step_fraction, or choose fewer labels"
                )
            except ValueError as exc:
                problem = str(exc)
        if problem:
            # Raised outside the handlers: nothing here keeps the algorithm's error (FM-98).
            raise PluginRunError(_NAME, problem)
        return chosen


def _imaging_with(
    params: RouteParams, scan: ScanResult, microscope: Microscope
) -> _Imaging:
    """The objective to image with and its matrix: the scan's, or the scan's scaled by the profile's sizes.

    The scan's own objective, given by name, is the scan's: no turret is asked
    and nothing is scaled or assumed. Any other needs the scan's objective (to
    scale from), a turret, a label that is on it and a profile size for both.
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
    turret = microscope.get(ObjectiveTurret)
    if turret is None:
        raise PluginRunError(
            _NAME,
            f"the stand has no turret, so the size of {wanted!r} is unknown; "
            f"leave objective unset to image with the scan's objective",
        )
    known = [label for label in turret.labels() if label]
    if wanted not in known:
        raise PluginRunError(
            _NAME,
            f"objective {wanted!r} is not on the turret: {', '.join(known)}; give "
            f"one of them, or leave objective unset",
        )
    size = turret.pixel_size_um(wanted)
    if not (math.isfinite(size) and size > 0.0):
        raise PluginRunError(
            _NAME,
            f"the profile has no pixel size for {wanted!r}: add it to the "
            f"profile's [camera] pixel_size_um, or leave objective unset",
        )
    scan_size = turret.pixel_size_um(scan.objective)
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


def _first_field_of(plan: ScanPlanResult, well: str) -> tuple[float, float]:
    """The first planned field of a well, in the scan's visiting order: where the scan starts in it."""
    for field in plan.fields:
        if field.well == well:
            return (field.x_um, field.y_um)
    raise PluginRunError(
        _NAME,
        f"the scan plan has no field for well {well!r}; run scan-plan again",
    )


def _stops(
    chosen: tuple[ChosenField, ...], reference: tuple[float, float]
) -> list[int]:
    """The visiting order of a well's fields: from the one nearest ``reference``, then the shortest path.

    Ties for the nearest go to the field chosen first.
    """
    start = min(
        range(len(chosen)),
        key=lambda i: (math.dist((chosen[i].x_um, chosen[i].y_um), reference), i),
    )
    return order_route([(f.x_um, f.y_um) for f in chosen], start=start)


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
    scan: ScanResult, positions: list[int], field: ChosenField
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
