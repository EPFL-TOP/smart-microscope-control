"""``scan-plan`` chooses the wells to image and lays a field grid in each, from the camera's real field (m3 §7.3).

The pure part (the well grammar, the grid, the clipping, the estimate) is
``wells.py`` and ``algorithm.py``. This reads the camera's frame and pixel size
and the turret, and moves nothing, so it runs in a dry run like at any other time.
What it decides is where the **camera matrix** comes from, because every field
position, every cell that is later turned into a stage position, and the route
rest on it:

* **Measured beats assumed.** A calibration of the same objective and frame shape
  is used as it is. A pixel size from the profile gives an upright camera with no
  mirror, and the result says so in ``assumed``; it never says the camera is
  straight when nothing measured it.
* **A size that is not a number is not a size** (FM-104): a pixel size of 0 means
  "not measured", infinity is above 0, and either is refused, not defaulted.
* **The profile's size is for binning 1** (FM-103). The camera's own size includes
  the binning and the magnifier, the turret's does not and nothing in ``Camera``
  reads the binning, so a plan for an objective that is not in place says that its
  size is for binning 1.
* **The default grid is derived from the very value that may be wrong** (FM-121),
  so a derived grid beyond 100 fields per axis is refused with the pixel size it
  was derived from; giving ``rows`` and ``columns`` is the way out.
* **The plan is in the frame of the objective it is for** (m3 §18.3). A plate
  registered with the 10x and planned for the 40x is moved by the offset between
  the two, which the profile holds (m2 §13), before any field is laid, so that
  ``plate_plan``, every field and the ``scan`` that visits them are the 40x's. An
  offset that is not measured, or a plate that does not record its objective, plans
  with the registration's positions and says so in ``assumed``, before the camera's
  lines, and in a WARNING. The result keeps the plate's objective and the shift.
* **A calibration from another document** (m3 §23). A run document holds one
  ``camera-calibration``, so a plate registered with the 4x cannot keep the 40x's
  calibration beside its registration: running one into it replaces the 4x's and
  makes the registration stale. ``calibration`` names a document of its own, whose
  matrix, rotation and mirror included, the plan then rests on. It was asked for,
  so one of another objective or frame is refused, never set aside for the
  profile's size; and without it the plan would assume rotation 0, which routes a
  cell of a camera turned 177 degrees to the other side of its field.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import useq
from pydantic import Field, ValidationError, model_validator

from smc.hardware.capabilities import Autofocus, Camera, ObjectiveTurret, XYStage
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.camera_calibration.plugin import calibration_from
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import (
    Matrix2,
    footprint,
    fov_of,
    pixel_matrix,
    pixel_of_stage,
)
from smc.plugins.objectives import (
    log_frame,
    objective_for,
    plate_for_objective,
    shift_between,
)
from smc.plugins.scan_plan.algorithm import (
    estimate_assumed,
    estimate_s,
    gap_overlap,
    grid_size,
    plan_fields,
    refocus_count,
    turn_from_axes_deg,
)
from smc.plugins.scan_plan.selection import ChosenFields, chosen_fields
from smc.plugins.scan_plan.wells import order_wells, parse_wells, well_indices
from smc.plugins.slots import (
    CameraCalibration,
    PlannedField,
    PlanSelection,
    PlateResult,
    RouteResult,
    ScanPlanResult,
    TimingsResult,
    well_geometry,
)
from smc.run.document import RunDocument, RunDocumentError, StageRecord

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["FOV_TOLERANCE", "ScanPlanParams", "ScanPlanPlugin"]

#: A selection's field size that differs from the plan's by more than this fraction
#: of the plan's, on either axis, is said in ``assumed``: the route chose fields of
#: one footprint and the scan takes another, so neighbours may overlap or leave a
#: gap (m3 §28.5). Chosen, not measured: 1 % is 4 µm of a 40x field.
FOV_TOLERANCE = 0.01
#: The selector a route that records none ran (algorithms §5.6).
_DEFAULT_SELECTOR = "most-cells"
#: The wells a line of ``assumed`` names before it says how many more (FM-205).
_WELLS_SHOWN = 8

#: The largest grid per axis, given or derived. The derived default comes from
#: the camera's pixel size, which a wrong profile entry can make 1000 times too
#: small, and that would be a million fields per well (FM-121).
_MAX_AXIS = 100
_WELLS_HELP = "give wells as A1, B2-B5, A1:D6, C*, *3 or all"
_NO_OBJECTIVE = "an unknown objective"


class ScanPlanParams(PluginParams):
    """The wells, the grid in each and the objective to plan for."""

    wells: str = Field(description="A1, B2-B5, A1:D6, C*, *3 or all.")
    serpentine: bool = Field(
        True, description="Visit alternate rows in reverse: less travel between wells."
    )
    objective: str | None = Field(
        None,
        description=(
            "The objective to plan for, by its label on the turret; default the "
            "one in place. The turret is never turned."
        ),
    )
    overlap: float = Field(
        0.10,
        ge=0.0,
        lt=0.9,
        description="Fraction of a field shared with its neighbour.",
    )
    coverage: float = Field(
        1.0,
        gt=0.0,
        le=1.5,
        description=(
            "Fraction of the well's width the grid spans; ignored when rows and "
            "columns are given. 1 is edge to edge, 0.33 the middle third."
        ),
    )
    rows: int | None = Field(
        None,
        ge=1,
        le=_MAX_AXIS,
        description="Fields along Y; give with columns, or neither.",
    )
    columns: int | None = Field(
        None,
        ge=1,
        le=_MAX_AXIS,
        description="Fields along X; give with rows, or neither.",
    )
    clip_to_well: bool = Field(
        True, description="Keep only the fields whose frame meets the well."
    )
    pixel_size_um: float | None = Field(
        None,
        gt=0.0,
        le=100.0,
        description=(
            "Plan with this pixel size (rotation 0, no mirror) instead of a "
            "calibration's or the profile's."
        ),
    )
    calibration: Path | None = Field(
        None,
        description=(
            "A run document holding a camera-calibration made with the plan's "
            "objective in place, for a plate registered with another objective "
            "(relative to the working directory, as from_run_doc). Its matrix "
            "is used, rotation and mirror included."
        ),
    )
    fields_from: Path | None = Field(
        None,
        description=(
            "A run document holding a route: plan the fields it chose instead of "
            "a grid (relative to the working directory, as from_run_doc). Copy "
            "the document after the survey scan and give the copy."
        ),
    )

    @model_validator(mode="after")
    def _rows_and_columns_come_together(self) -> ScanPlanParams:
        """A grid of 3 rows by however many columns the coverage gives is not a grid anyone asked for."""
        if (self.rows is None) != (self.columns is None):
            raise ValueError("rows and columns are given together or not at all")
        return self

    @model_validator(mode="after")
    def _a_size_and_a_calibration_are_two_answers(self) -> ScanPlanParams:
        """A bare size says rotation 0 and no mirror, a calibration says what was measured: asked for both, one would be dropped without a word."""
        if self.pixel_size_um is not None and self.calibration is not None:
            raise ValueError("give pixel_size_um or calibration, not both")
        return self

    @model_validator(mode="after")
    def _a_selection_lays_no_grid(self) -> ScanPlanParams:
        """The grid's parameters given with a selection would be dropped without a word (m3 §28.5).

        A value equal to its default is not refused: it says nothing a plan from a
        selection would ignore, and the page or a preset may send it.
        """
        if self.fields_from is None:
            return self
        defaults = type(self).model_fields
        names = [
            name
            for name in ("rows", "columns", "coverage", "overlap", "clip_to_well")
            if getattr(self, name) != defaults[name].default
        ]
        if names:
            raise ValueError(
                f"fields_from plans the fields a selection chose: {', '.join(names)} "
                "are not read with it; leave them unset"
            )
        return self


@dataclass(frozen=True)
class _Basis:
    """The camera matrix a plan rests on, where it came from, and what it assumed."""

    matrix: Matrix2
    pixel_size_um: float
    source: Literal["camera-calibration", "profile", "parameter"]
    assumed: tuple[str, ...]


class ScanPlanPlugin:
    """Chooses wells and lays a grid of fields in each, clipped to the well."""

    name = "scan-plan"
    summary = "Choose wells and lay a field grid in each from the camera's real field of view."
    Params = ScanPlanParams
    Result = ScanPlanResult
    requires: tuple[type, ...] = (Camera,)
    provides = "scan-plan"
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ("camera-calibration", "timings")
    #: ``fields_from`` read in the document this run writes would read a route made
    #: from the very plan the stage's ``running`` record replaces (FM-272, m3 §28.7).
    reuses_from: tuple[str, ...] = ("fields_from",)
    reuse_slots: ClassVar[Mapping[str, str]] = {"fields_from": "route"}

    def run(
        self,
        microscope: Microscope,
        params: ScanPlanParams,
        ctx: RunContext,
    ) -> ScanPlanResult:
        """Plan the fields; raise when there is no pixel size or no usable grid.

        Raises:
            PluginRunError: The wells cannot be read, the objective is not on the
                turret or none is in place, there is no pixel size, the camera
                matrix cannot be inverted, the derived grid is over 100 fields
                per axis, or a well has no field that touches it.
        """
        camera = microscope.require(Camera)
        turret = microscope.get(ObjectiveTurret)
        plate = ctx.upstream("plate", PlateResult)

        wells = _wells_in_order(self.name, plate, params)
        objective, in_place = objective_for(self.name, params.objective, turret)
        # The plan is for `objective`, which may not be the one the plate was
        # registered with: from here on the plate is the one in its frame (m3 §18.3).
        frame = plate_for_objective(plate, objective, turret)
        log_frame(ctx.logger, plate, frame, turret)
        framed = frame.plate
        height, width = camera.image_shape()
        shape = (int(height), int(width))
        if min(shape) < 1:
            raise PluginRunError(
                self.name,
                f"the camera reports a frame of {shape} pixels: check the "
                "camera's region of interest",
            )

        basis, given = _camera_basis(
            self.name,
            params,
            ctx,
            camera,
            turret,
            objective,
            in_place,
            shape,
            microscope.profile.microscope.name,
        )
        if given is not None:
            ctx.logger.info(
                "camera calibration of %s from %s, measured %s",
                _objective_name(objective),
                given[0],
                given[1],
            )
        _require_invertible(self.name, basis.matrix, shape)
        fov_um = fov_of(basis.matrix, shape)

        chosen: _Selection | None = None
        if params.fields_from is not None:
            # The fields a selection chose, at this plan's objective (m3 §28.5): every
            # check is made here, before the plan is written, so that `scan` never
            # starts on a field it should not visit.
            chosen = _selection_given(
                self.name,
                params.fields_from,
                ctx,
                microscope,
                framed,
                wells,
                objective,
                fov_um,
                params.wells,
            )
            rows, columns, overlap = 1, 1, 0.0
            plate_plan = framed.to_plan(
                selected_wells=well_indices(plate.plate, chosen.fields.wells),
                well_points_plan=useq.RelativePosition(x=0.0, y=0.0),
            )
            fields = chosen.fields.fields
            wells = chosen.fields.wells
        else:
            rows, columns, overlap, plate_plan, fields = self._grid(
                params, framed, wells, fov_um, basis, shape
            )

        timings = ctx.upstream_or_none("timings", TimingsResult)
        estimate: float | None = None
        without: float | None = None
        estimate_lines: tuple[str, ...] = ()
        if timings is not None:
            # A lock is the scan's, one in each well (m3 §7.4 D1): charged only on a
            # stand that has an autofocus and a lock time that was measured, so that
            # an unknown cost is left out and nothing is said about a lock.
            locks = (
                refocus_count(fields)
                if microscope.get(Autofocus) is not None
                and timings.refocus_ms is not None
                else 0
            )
            estimate = estimate_s(fields, timings, refocuses=locks)
            without = estimate_s(fields, timings, refocuses=0)
            estimate_lines = estimate_assumed(locks)
        # The frame's line first: the positions are what the rest rests on; then the
        # selection's, which the positions come from (m3 §28.5). An estimate whose
        # refocus was assumed must not read as measured, and its own lines, what it
        # counted and what it leaves out, come last.
        selection_lines = chosen.assumed if chosen is not None else ()
        assumed = (
            (*frame.assumed, *selection_lines, *basis.assumed)
            if timings is None
            else (
                *frame.assumed,
                *selection_lines,
                *basis.assumed,
                *(f"timings: {line}" for line in timings.assumed),
                *estimate_lines,
            )
        )
        return ScanPlanResult(
            plate_plan=plate_plan,
            wells=wells,
            objective=objective,
            pixel_size_um=basis.pixel_size_um,
            matrix_um_per_px=basis.matrix,
            matrix_source=basis.source,
            image_shape_px=shape,
            fov_um=fov_um,
            overlap=overlap,
            rows=rows,
            columns=columns,
            fields=fields,
            estimate_s=estimate,
            estimate_without_autofocus_s=without,
            plate_objective=plate.objective,
            objective_shift_um=frame.shift_um,
            calibration_from=given[0] if given is not None else None,
            calibration_measured=given[1] if given is not None else None,
            selection=chosen.selection if chosen is not None else None,
            assumed=assumed,
        )

    def _grid(
        self,
        params: ScanPlanParams,
        framed: PlateResult,
        wells: tuple[str, ...],
        fov_um: tuple[float, float],
        basis: _Basis,
        shape: tuple[int, int],
    ) -> tuple[int, int, float, useq.WellPlatePlan, tuple[PlannedField, ...]]:
        """The grid laid in each well: rows, columns, overlap, the useq plan and the fields (m3 §7.2).

        Raises:
            PluginRunError: The derived grid is over 100 fields per axis, or a
                well has no field that touches it.
        """
        if params.rows is not None and params.columns is not None:
            rows, columns = params.rows, params.columns
        else:
            # Every registered well has one size (m3 §3.2), so the first stands for all.
            well_um = well_geometry(framed.to_plan(), wells[0]).size_um
            rows, columns = grid_size(
                fov_um, well_um, coverage=params.coverage, overlap=params.overlap
            )
            if max(rows, columns) > _MAX_AXIS:
                raise PluginRunError(
                    self.name,
                    f"the grid would be {rows} × {columns} fields per well at "  # noqa: RUF001
                    f"{basis.pixel_size_um:g} µm per pixel: check the pixel size, "
                    "or give rows and columns",
                )

        built: tuple[useq.WellPlatePlan, tuple[PlannedField, ...]] | None = None
        problem = ""
        try:
            built = plan_fields(
                framed,
                wells,
                fov_um=fov_um,
                footprint_of=lambda x, y: footprint(
                    x, y, matrix=basis.matrix, image_shape_px=shape
                ),
                rows=rows,
                columns=columns,
                overlap=params.overlap,
                clip_to_well=params.clip_to_well,
            )
        except ValueError as exc:
            problem = str(exc)
        if built is None:
            # Raised outside the handler: nothing here keeps the grid's error.
            raise PluginRunError(self.name, problem)
        plate_plan, fields = built
        return rows, columns, params.overlap, plate_plan, fields


def _wells_in_order(
    name: str, plate: PlateResult, params: ScanPlanParams
) -> tuple[str, ...]:
    """The wells the text names, in the order a scan visits them; the typing order is dropped."""
    parsed: tuple[str, ...] = ()
    problem: str | None = None
    try:
        parsed = parse_wells(plate.plate, params.wells)
    except ValueError as exc:
        problem = f"{exc}; {_WELLS_HELP}"
    if problem is not None:
        # Raised outside the handler: nothing here keeps the grammar's error.
        raise PluginRunError(name, problem)
    return order_wells(plate.plate, parsed, serpentine=params.serpentine)


def _require_invertible(name: str, matrix: Matrix2, shape: tuple[int, int]) -> None:
    """Refuse a matrix that no pixel maps back through, with the refusal the later stages would give.

    ``pixel_of_stage`` is what the cells and the route turn a stage position into
    a pixel with, and it refuses a singular matrix. A calibration that validates
    field by field but is singular (a run document edited by hand, another tool
    filling the slot) would otherwise be planned as a field that is a line, and
    refused only when someone uses the plan.
    """
    problem = ""
    try:
        pixel_of_stage(0.0, 0.0, 0.0, 0.0, matrix=matrix, image_shape_px=shape)
    except ValueError as exc:
        problem = str(exc)
    if problem:
        # Raised outside the handler: nothing here keeps the geometry's error.
        raise PluginRunError(name, problem)


def _camera_basis(
    name: str,
    params: ScanPlanParams,
    ctx: RunContext,
    camera: Camera,
    turret: ObjectiveTurret | None,
    objective: str | None,
    in_place: bool,
    shape: tuple[int, int],
    profile: str,
) -> tuple[_Basis, tuple[str, str] | None]:
    """The matrix to plan with, and where and when a given calibration was measured (``None`` for any other source).

    In this order (m3 §23.3): a pixel size given as a parameter; a calibration
    document given as a parameter; the run document's own calibration of this
    objective and frame; the profile's size. A document that was asked for and
    does not fit is refused (the operator named it), where the document's own
    that does not fit is set aside with the reason, because nobody chose it.
    """
    if params.pixel_size_um is not None:
        return (
            _Basis(
                pixel_matrix(params.pixel_size_um),
                params.pixel_size_um,
                "parameter",
                ("pixel size given as a parameter; rotation 0 and no mirror assumed",),
            ),
            None,
        )
    if params.calibration is not None:
        # The document's own camera-calibration is not read when one is given, even
        # one of this objective: the parameter was asked for.
        basis, where, when = _calibration_given(
            name,
            params.calibration,
            ctx,
            profile,
            objective,
            shape,
            _grid_overlap(params),
        )
        return basis, (where, when)

    assumed: list[str] = []
    calibration = ctx.upstream_or_none("camera-calibration", CameraCalibration)
    if calibration is not None:
        why = _why_not_used(calibration, objective, shape)
        if why is None:
            return _calibrated(calibration, _grid_overlap(params)), None
        # A calibration of another objective is what the plate's own registration
        # leaves behind (a run document keeps one per plugin): say how to get the
        # one of this objective. A plan with no objective has none to ask for.
        hint = (
            f"; give calibration, a run document whose camera-calibration was "
            f"made with {objective} in place"
            if calibration.objective != objective and objective is not None
            else ""
        )
        assumed.append(f"camera calibration not used: {why}{hint}")

    label = objective if objective is not None else "this stand"
    if in_place or turret is None:
        size_um = camera.pixel_size_um()
    else:
        size_um = turret.pixel_size_um(objective)
        assumed.append(
            f"pixel size of {label} from the profile, for binning 1: "
            "the camera's binning is not applied"
        )
    if not (math.isfinite(size_um) and size_um > 0.0):
        raise PluginRunError(
            name,
            f"no pixel size for {label}: run camera-calibration with it in place, "
            "or add it to the profile's [camera] pixel_size_um",
        )
    assumed.append(
        f"no camera calibration for {label}: rotation 0 and no mirror assumed"
    )
    return _Basis(pixel_matrix(size_um), size_um, "profile", tuple(assumed)), None


def _grid_overlap(params: ScanPlanParams) -> float | None:
    """The overlap of the grid the plan lays, or ``None`` when it lays none (``fields_from``)."""
    return None if params.fields_from is not None else params.overlap


def _calibrated(calibration: CameraCalibration, overlap: float | None) -> _Basis:
    """The basis of a calibration that was measured: its matrix, and only the line about the gaps it leaves.

    What the calibration itself assumed is not copied: it is the calibration's
    to say, and the plan assumed nothing about the camera. ``overlap`` is
    ``None`` for a plan with no grid (a selection's, m3 §28): the gaps are
    between a grid's fields, and the advice to raise ``overlap`` is one
    ``fields_from`` refuses.
    """
    gaps = None if overlap is None else _gap_line(calibration.rotation_deg, overlap)
    return _Basis(
        calibration.matrix_um_per_px,
        calibration.pixel_size_um,
        "camera-calibration",
        (gaps,) if gaps is not None else (),
    )


def _calibration_given(
    name: str,
    path: Path,
    ctx: RunContext,
    profile: str,
    objective: str | None,
    shape: tuple[int, int],
    overlap: float | None,
) -> tuple[_Basis, str, str]:
    """The basis of the calibration in another run document, where it is (as typed) and when it was measured.

    A plate is registered with the 4x and scanned with the 40x, and one run
    document holds one ``camera-calibration`` (m3 §1.7): the 40x's cannot be run
    into the registration's without making the registration stale, so it lives in
    a document of its own, named here (m3 §23). The rule that loads it is
    ``objective-offsets``'. The operator named this document, so one that does
    not fit is refused and never set aside for the profile's size. ``path`` is
    named as typed, with forward slashes (FM-24) and never resolved (FM-50).

    Raises:
        PluginRunError: The document cannot be used (``calibration_from``'s
            refusals), or its calibration is of another objective or frame.
    """
    found, when = calibration_from(name, path, profile=profile, dry_run=ctx.dry_run)
    where = path.as_posix()
    if found.objective != objective:
        # Raised outside any handler (FM-98): nothing here keeps another error.
        raise PluginRunError(
            name,
            f"the calibration in {where} is of {_objective_name(found.objective)}, "
            f"the plan is for {_objective_name(objective)}: give calibration a run "
            f"document whose camera-calibration was made with "
            f"{_objective_name(objective)} in place",
        )
    if found.image_shape_px != shape:
        raise PluginRunError(
            name,
            f"the calibration in {where} is of a frame of {found.image_shape_px}, "
            f"the camera gives {shape} (binning or the region changed): "
            "run camera-calibration into it again with this camera setting",
        )
    return _calibrated(found, overlap), where, when


@dataclass(frozen=True)
class _Selection:
    """A selection's fields for the plan, the record of where they came from, and its lines."""

    fields: ChosenFields
    selection: PlanSelection
    assumed: tuple[str, ...]


def _selection_given(
    name: str,
    path: Path,
    ctx: RunContext,
    microscope: Microscope,
    framed: PlateResult,
    wells: tuple[str, ...],
    objective: str | None,
    fov_um: tuple[float, float],
    wells_text: str,
) -> _Selection:
    """The fields the route of another run document chose, in this plan's frame, checked (m3 §28.5).

    The route is read from a document of its own, never this one: a plan run
    into the selection's document replaces the plan the route rests on (m3
    §28.1). Each refusal is a ``PluginRunError`` of ``name`` raised outside any
    handler (FM-98), in §28.5's order: the route, its plan, the registration,
    the objective of the route's fields, a field outside its well or the soft
    limits, no field. ``path`` is named as typed, with forward slashes (FM-24),
    and never resolved (FM-50). The stops' Z and PFS offset are never read: they
    are the selection's scan's, not a focus at this objective (D3).

    Raises:
        PluginRunError: The selection cannot be planned; nothing was written.
    """
    where = path.as_posix()
    profile = microscope.profile.microscope.name
    problem: str | None = None
    other: RunDocument | None = None
    try:
        other = RunDocument.load(path)
    except RunDocumentError as exc:
        problem = f"{where}: {exc.reason}"
    if problem is not None or other is None:
        # Raised outside the handler: nothing here keeps the load's error.
        raise PluginRunError(name, problem or where)

    # 1. The route, as `calibration_from` reads a calibration (m3 §20.3).
    route_record = _usable_record(other, "route", where, profile, ctx.dry_run, name)
    route: RouteResult | None = None
    try:
        route = RouteResult.model_validate(route_record.result)
    except ValidationError:
        route = None
    selector = route_record.params.get("selector", _DEFAULT_SELECTOR)
    if route is None or not isinstance(selector, str):
        raise PluginRunError(
            name,
            f"the route in {where} is not one this smc reads; run route again there",
        )

    # 2. The plan the selection's scan was made with: its objective is the frame the
    # stops are in (m3 §28.1), and its registration is checked next.
    plan_record = _usable_record(other, "scan-plan", where, profile, ctx.dry_run, name)
    source_plan: ScanPlanResult | None = None
    try:
        source_plan = ScanPlanResult.model_validate(plan_record.result)
    except ValidationError:
        source_plan = None
    if source_plan is None:
        raise PluginRunError(
            name,
            f"the scan plan in {where} is not one this smc reads; run scan-plan "
            "again there",
        )

    # 3. The registration, as §21.12.5's check 2: both sides must name one.
    read = plan_record.reads.get("plate")
    theirs = read.attempt if read is not None else None
    provider = ctx.document.provider("plate")
    ours = provider.attempt if provider is not None else None
    if theirs is None or ours is None or theirs != ours:
        raise PluginRunError(
            name,
            f"the selection in {where} was chosen on another registration of the "
            f"plate ({theirs or 'none'} against this run's {ours or 'none'})",
        )

    # 4. The footprint the route chose its fields for.
    if route.objective != objective:
        raise PluginRunError(
            name,
            f"the selection in {where} chose fields of "
            f"{_objective_name(route.objective)}, the plan is for "
            f"{_objective_name(objective)}: run route there with "
            f"objective={_objective_name(objective)}, or plan for "
            f"{_objective_name(route.objective)}",
        )

    # 5. Each stop moved into this plan's frame, inside its well and the limits.
    turret = microscope.get(ObjectiveTurret)
    shift, shift_line = shift_between(source_plan.objective, objective, turret)
    xy = microscope.get(XYStage)
    travel = xy.limits_um() if xy is not None else None
    limits = (
        None
        if travel is None
        else (
            (travel[0].low_um, travel[0].high_um),
            (travel[1].low_um, travel[1].high_um),
        )
    )
    plan = framed.to_plan()
    chosen: ChosenFields | None = None
    try:
        chosen = chosen_fields(
            route.positions,
            wells,
            shift_um=shift if shift is not None else (0.0, 0.0),
            geometry_of=lambda well: well_geometry(plan, well),
            xy_limits_um=limits,
        )
    except ValueError as exc:
        problem = str(exc)
    if chosen is None:
        # Raised outside the handler: nothing here keeps the check's error.
        raise PluginRunError(name, problem or where)

    # 6. No field in the wells named.
    if not chosen.fields:
        theirs_wells = tuple(dict.fromkeys(stop.well for stop in route.positions))
        raise PluginRunError(
            name,
            f"the selection in {where} holds no field in the wells named "
            f"({wells_text}); it chose fields in "
            f"{_first_eight(theirs_wells) or 'no well'}",
        )

    measured = (
        route_record.finished_utc.isoformat(timespec="seconds")
        if route_record.finished_utc is not None
        else "at an unknown time"
    )
    selection = PlanSelection(
        run_doc=where,
        route_attempt=route_record.attempt,
        measured=measured,
        selector=selector,
        source_objective=source_plan.objective,
        shift_um=shift,
        wells_without_fields=chosen.wells_without_fields,
        fields_left_out=chosen.fields_left_out,
    )
    lines = [
        f"the fields are the {len(chosen.fields)} that selector '{selector}' chose "
        f"in {where} (measured {measured}), each well's in the selection's order"
    ]
    if shift_line is not None:
        ctx.logger.warning(shift_line)
        lines.append(shift_line)
    if chosen.wells_without_fields:
        lines.append(
            f"no field was chosen in {len(chosen.wells_without_fields)} of the wells "
            f"named: {_first_eight(chosen.wells_without_fields)}"
        )
    if chosen.fields_left_out:
        lines.append(
            f"{chosen.fields_left_out} fields of the selection lie in wells not named "
            f"and were left out: {_first_eight(chosen.wells_left_out)}"
        )
    if any(
        abs(theirs_um - ours_um) > FOV_TOLERANCE * ours_um
        for theirs_um, ours_um in zip(route.fov_um, fov_um, strict=True)
    ):
        lines.append(
            f"the selection sized its fields {route.fov_um[0]:.1f} × "  # noqa: RUF001
            f"{route.fov_um[1]:.1f} µm, the plan's are {fov_um[0]:.1f} × "  # noqa: RUF001
            f"{fov_um[1]:.1f} µm: neighbouring fields may overlap or leave a gap"
        )
    if limits is None:
        lines.append(
            "the stage reports no soft limits: the selection's fields were not "
            "checked against its travel"
        )
    ctx.logger.info(
        "%d fields in %d wells chosen by %s, from %s measured %s; shift %s",
        len(chosen.fields),
        len(chosen.wells),
        selector,
        where,
        measured,
        f"({shift[0] + 0.0:+.2f}, {shift[1] + 0.0:+.2f}) µm"
        if shift is not None
        else "not applied",
    )
    return _Selection(chosen, selection, tuple(lines))


def _usable_record(
    other: RunDocument,
    slot: str,
    where: str,
    profile: str,
    dry_run: bool,
    name: str,
) -> StageRecord:
    """The record of ``slot`` in another run document, if the runner's own rule lets it be read.

    Raises:
        PluginRunError: Not ``done``, stale, another profile, a dry run read by a
            real one, or no result.
    """
    reason = other.unusable(slot, profile=profile, dry_run=dry_run)
    record = other.provider(slot)
    if reason is not None or record is None or record.result is None:
        raise PluginRunError(name, f"in {where}: {reason or 'it holds no result'}")
    return record


def _first_eight(wells: Sequence[str]) -> str:
    """The first eight wells, ``, ``-joined, and how many more: a line stays short (FM-205)."""
    shown = ", ".join(wells[:_WELLS_SHOWN])
    more = len(wells) - _WELLS_SHOWN
    return f"{shown}, … {more} more" if more > 0 else shown


def _objective_name(objective: str | None) -> str:
    """The objective's label, or the words for no objective."""
    return objective if objective is not None else _NO_OBJECTIVE


def _why_not_used(
    calibration: CameraCalibration, objective: str | None, shape: tuple[int, int]
) -> str | None:
    """Why a calibration is not of this plan's objective and frame, or ``None`` when it is.

    A calibration of another objective or binning is silently wrong, not merely
    stale, so it is compared on both and the one that differs is named.
    """
    if calibration.objective != objective:
        return (
            f"it is of {_objective_name(calibration.objective)}, "
            f"the plan is for {_objective_name(objective)}"
        )
    if calibration.image_shape_px != shape:
        return f"it is of a frame of {calibration.image_shape_px}, the camera gives {shape}"
    return None


def _gap_line(rotation_deg: float, overlap: float) -> str | None:
    """What to tell the operator when the camera's turn leaves gaps in the axis-aligned grid."""
    suggested = gap_overlap(rotation_deg, overlap)
    if suggested is None:
        return None
    turned = f"the camera is turned {turn_from_axes_deg(rotation_deg):.1f}° from the stage axes: "
    if math.isinf(suggested):
        return (
            turned + "no overlap below 0.9 closes the gaps between fields; "
            "straighten the camera"
        )
    return (
        turned + f"at overlap {overlap:g} the grid leaves gaps between fields; "
        f"use overlap={suggested:.2f}"
    )
