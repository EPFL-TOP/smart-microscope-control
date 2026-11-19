"""A ``SelectionInput`` built in memory, for the test of a selector (algorithms design §5.7).

A collaborator's selector is tested on cells placed by hand: no run folder, no
stand, no file. This builds the scan around them from ``useq``'s plate (so a
well is where the plate registry says, not where this module guesses) and a
camera with no rotation and no mirror, over a 100 by 100 pixel frame.

It lives in ``smc.selection`` and not in ``smc.testing`` so that it never
imports ``pymmcore_plus``: ``tests/test_kit_imports.py`` keeps it so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import useq

from smc.data import Scan
from smc.plugins.scan_plan.wells import well_indices
from smc.plugins.slots import (
    Cell,
    CellsResult,
    PlannedField,
    PlateResult,
    ScanPlanResult,
    ScanResult,
    normalise_well,
    well_geometry,
)
from smc.run import RunDocument
from smc.selection.api import SelectionError, SelectionInput

__all__ = ["make_input"]

#: Pixels of the frame ``make_input``'s camera has, each way.
_SIDE_PX = 100


def make_input(
    cells: Sequence[tuple[str, float, float, str]],
    *,
    wells: Sequence[str] = ("B2",),
    plate: str = "96-well",
    field_um: tuple[float, float] = (100.0, 100.0),
    labels: Sequence[str] = ("single",),
    fields_per_well: int = 10,
    wall_margin_um: float = 0.0,
    exclude_centre_um: float = 0.0,
) -> SelectionInput:
    """A ``SelectionInput`` for ``cells`` given as ``(well, x_um, y_um, label)`` in stage µm.

    The plate is ``useq``'s ``plate`` with A1 at the origin and no rotation; the
    plan holds **one field per well, at the well's centre** (no grid), so
    ``first_field(well)`` is the well's centre. The camera is upright and not
    mirrored, ``field_um`` is ``(width, height)`` of its footprint, and the frame
    is 100 by 100 pixels, so ``pixel_size_um`` is the width over 100. ``labels``
    is what ``route`` would be asked to count (its default, ``("single",)``);
    the cells' classes are those, then any other label the cells carry, in the
    order they appear. A scan built here has no frames: ``scan.image`` of any
    frame raises ``DatasetError``, and a selector that reads pixels is tested on
    a run folder the test writes.

    Raises:
        SelectionError: A cell is of a well that was not given, or a coordinate
            is not finite; ``field_um`` is not finite and positive;
            ``fields_per_well`` is below 1; a margin is negative or not finite;
            ``labels`` is empty or repeats; or ``wells`` or ``plate`` is not
            one ``useq`` knows.
    """
    width_um, height_um = field_um
    if not all(math.isfinite(side) and side > 0.0 for side in field_um):
        raise SelectionError(
            f"field_um must be finite and positive, got {field_um!r}: "
            f"give (width, height) in µm"
        )
    if fields_per_well < 1:
        raise SelectionError(
            f"fields_per_well must be at least 1, got {fields_per_well!r}"
        )
    for name, value in (
        ("wall_margin_um", wall_margin_um),
        ("exclude_centre_um", exclude_centre_um),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise SelectionError(f"{name} must be finite and 0 or more, got {value!r}")
    wanted = tuple(labels)
    if not wanted:
        raise SelectionError("labels must name at least one label")
    for label in wanted:
        if wanted.count(label) > 1:
            raise SelectionError(f"labels repeat {label!r}")

    names = tuple(normalise_well(well) for well in wells)
    if len(set(names)) != len(names):
        raise SelectionError(f"wells are listed twice: {', '.join(wells)}")
    plan, planned = _plan(plate, names)
    listed = set(names)
    held: list[Cell] = []
    for k, (well, x_um, y_um, label) in enumerate(cells):
        if well not in listed:
            raise SelectionError(
                f"cell {k} is in well {well!r}, which make_input was not given "
                f"({', '.join(names)}): add it to wells"
            )
        if not (math.isfinite(x_um) and math.isfinite(y_um)):
            raise SelectionError(
                f"cell {k} ({well}, {label}) has a position that is not finite: "
                f"x_um={x_um!r}, y_um={y_um!r}"
            )
        held.append(
            Cell(
                well=well,
                x_um=x_um,
                y_um=y_um,
                label=label,
                score=1.0,
                frame=0,
                box_px=(0.0, 0.0, 0.0, 0.0),
            )
        )

    pixel_um = width_um / _SIDE_PX
    matrix = ((pixel_um, 0.0), (0.0, -height_um / _SIDE_PX))
    shape = (_SIDE_PX, _SIDE_PX)
    classes = tuple(dict.fromkeys([*wanted, *(cell.label for cell in held)]))
    counts = {well: dict.fromkeys(classes, 0) for well in names}
    for cell in held:
        counts[cell.well][cell.label] += 1

    scan_plan = ScanPlanResult(
        plate_plan=plan,
        wells=names,
        objective=None,
        pixel_size_um=pixel_um,
        matrix_um_per_px=matrix,
        matrix_source="parameter",
        image_shape_px=shape,
        fov_um=(width_um, height_um),
        overlap=0.0,
        rows=1,
        columns=1,
        fields=planned,
        estimate_s=None,
        estimate_without_autofocus_s=None,
    )
    scan_result = ScanResult(
        folder="in-memory",
        objective=None,
        pixel_size_um=pixel_um,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
        exposure_ms=0.0,
        frames=(),
        failed=(),
    )
    cells_result = CellsResult(
        file="in-memory",
        model="make_input",
        model_sha256="0" * 64,
        classes=classes,
        device="none",
        score_threshold=0.0,
        merge_radius_um=0.0,
        frames=0,
        counts=counts,
        merged=0,
        seconds_per_frame=None,
    )
    scan = Scan.from_results(
        Path("in-memory"),
        RunDocument.new(),
        plan=scan_plan,
        result=scan_result,
        cells_result=cells_result,
        cells=held,
    )
    return SelectionInput(
        scan=scan,
        labels=wanted,
        fields_per_well=fields_per_well,
        wall_margin_um=wall_margin_um,
        exclude_centre_um=exclude_centre_um,
        objective=None,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
    )


def _plan(
    plate: str, names: tuple[str, ...]
) -> tuple[useq.WellPlatePlan, tuple[PlannedField, ...]]:
    """``useq``'s plan of ``names`` on ``plate`` (A1 at the origin, no rotation), and one planned field at each well's centre.

    Built as ``scan_plan.algorithm.plan_fields`` builds its plan (the registry's
    ``well_indices`` and the plate result's ``to_plan``) so that the wells are
    named and placed as a real scan plan has them, minus the grid.
    """
    problem = ""
    try:
        registered = PlateResult(
            plate=plate,
            a1_center_xy_um=(0.0, 0.0),
            rotation_deg=0.0,
            method="seed",
            residual_um=None,
            scale_error=None,
            n_refs=0,
        )
        rows, columns = well_indices(plate, names)
        plan = registered.to_plan(selected_wells=(rows, columns))
        fields = []
        for well in names:
            x_um, y_um = well_geometry(plan, well).center_um
            fields.append(
                PlannedField(
                    well=well, index=0, name=f"{well}_0000", x_um=x_um, y_um=y_um
                )
            )
    except ValueError as exc:
        problem = f"plate {plate!r}, wells {', '.join(names)}: {exc}"
    else:
        return plan, tuple(fields)
    # Raised outside the handler: nothing here keeps the registry's error (FM-98).
    raise SelectionError(problem)
