"""``plate-registration`` fills the ``plate`` slot from wells the operator centred or walls they touched (m3 §5).

The pure part (the fit, the wall arithmetic) is ``algorithm.py``. This touches
no device, so a dry run is the same computation; it reads the references from
its parameters and keeps nothing between two runs, so the run document's
``params`` hold every reference that made the registration. What it decides,
because every later stage places its fields by the plate it writes:

* **A seed is a result, not a fit.** One reference sets A1 and assumes no
  rotation; it says so in ``assumed`` and has no residual, so nothing reads it
  as a measured registration.
* **A wall centre is a reference like any other, and says what it assumed.** A
  well given by walls goes into the fit with the others; an axis where only one
  wall was touched is reported (the centre is one wall plus the nominal
  half-size).
* **A bad fit is a refusal, not a ``done`` result to be checked.** A reference
  further than ``GOOD_RESIDUAL_UM`` from where the others put it fails the stage
  and names the well, so no unusable ``plate`` fills the slot.
* **A rotation beyond the bound is a refusal** (FM-121): two equidistant
  references fit a turn of 90 degrees with no residual, and a plate is put on the
  stage by hand.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import Field

from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.plate_registration.algorithm import (
    GOOD_RESIDUAL_UM,
    MAX_ROTATION_DEG,
    EdgeCentre,
    calibrate,
    centre_from_walls,
)
from smc.plugins.slots import (
    PlateResult,
    WallTouch,
    WellRef,
    normalise_well,
    well_geometry,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["PlateRegistrationParams", "PlateRegistrationPlugin"]

_NAME = "plate-registration"


class PlateRegistrationParams(PluginParams):
    """The plate, and the wells or walls that locate it on the stage."""

    plate: str = Field(
        "96-well",
        description="A useq plate name (96-well, 384-well); names are exact.",
    )
    refs: tuple[WellRef, ...] = Field(
        (),
        description=(
            "Wells centred under the objective, JSON: "
            'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}], the position '
            "from smc stage get. One is a seed (rotation 0 assumed); every "
            "reference so far goes in each run."
        ),
    )
    walls: tuple[WallTouch, ...] = Field(
        (),
        description=(
            "Stage positions at a well's walls, JSON: "
            'walls=[{"well": "B2", "side": "left", "x_um": 0, "y_um": 0}]. '
            "Each well needs one left or right and one top or bottom (top is +Y); "
            "an opposite pair gives the centre and measures the well."
        ),
    )
    max_rotation_deg: float = Field(
        MAX_ROTATION_DEG,
        gt=0.0,
        le=45.0,
        description=(
            "The largest rotation of the plate that the fit accepts; a rotation "
            "past it is read as a misnamed well. Raise it only for a plate that "
            "is really turned."
        ),
    )


class PlateRegistrationPlugin:
    """Locates a plate on the stage: A1's centre and the plate's rotation, from references and wall touches."""

    name = _NAME
    summary = (
        "Locate a well plate on the stage from wells you centred and walls you "
        "touched; one well is a seed."
    )
    Params = PlateRegistrationParams
    Result = PlateResult
    requires: tuple[type, ...] = ()
    provides = "plate"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: PlateRegistrationParams,
        ctx: RunContext,
    ) -> PlateResult:
        """Fit the plate from the references and the wall centres, or refuse.

        Raises:
            PluginRunError: No reference at all; a well given both as a reference
                and by walls; a plate, well, wall or spacing the algorithm
                refuses; a rotation beyond ``max_rotation_deg``; or a reference
                further than ``GOOD_RESIDUAL_UM`` from where the others put it.
        """
        walls_of: dict[str, list[WallTouch]] = {}
        for touch in params.walls:
            walls_of.setdefault(normalise_well(touch.well), []).append(touch)

        both = [
            name
            for name in dict.fromkeys(normalise_well(r.well) for r in params.refs)
            if name in walls_of
        ]
        if both:
            raise PluginRunError(
                _NAME,
                f"{both[0]} is given both as a reference and by walls: give it one way",
            )
        if not params.refs and not params.walls:
            raise PluginRunError(
                _NAME,
                "no reference: give refs (a centred well) or walls (wall touches); "
                "see docs/plugins/plate-registration.md",
            )

        problem = ""
        edges: dict[str, EdgeCentre] = {}
        fitted: PlateResult | None = None
        try:
            for well, touches in walls_of.items():
                edges[well] = centre_from_walls(params.plate, well, touches)
            references = [
                *params.refs,
                *(WellRef(well=w, x_um=e.x_um, y_um=e.y_um) for w, e in edges.items()),
            ]
            fitted = calibrate(
                params.plate, references, max_rotation_deg=params.max_rotation_deg
            )
        except ValueError as exc:
            problem = str(exc) or "the registration was refused"
        if problem or fitted is None:
            # Raised outside the handler: nothing here keeps the algorithm's error (FM-98).
            raise PluginRunError(_NAME, problem)

        if fitted.residual_um is not None and fitted.residual_um > GOOD_RESIDUAL_UM:
            worst, distance = max(
                (
                    (r.well, math.dist(fitted.well_center_um(r.well), (r.x_um, r.y_um)))
                    for r in fitted.refs
                ),
                key=lambda pair: pair[1],
            )
            raise PluginRunError(
                _NAME,
                f"{worst} is {distance:.0f} µm from where the others put it: "
                f"centre it again",
            )

        plan = fitted.to_plan()
        lines: list[str] = []
        for well, edge in edges.items():
            width_um, height_um = well_geometry(plan, well).size_um
            for axis in edge.assumed_axes:
                half_um = (width_um if axis == "x" else height_um) / 2.0
                lines.append(
                    f"{well}: the {axis.upper()} centre is one wall plus the "
                    f"nominal half-size, {half_um:.0f} µm"
                )
        return PlateResult.model_validate(
            {**fitted.model_dump(), "assumed": (*fitted.assumed, *lines)}
        )
