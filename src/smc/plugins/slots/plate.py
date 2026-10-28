"""The plate slot: where the plate is, and the shape of its wells (m3 §3.2).

Plate registration and plate finding both fill the ``plate`` slot with a
``PlateResult``, and everything downstream (the scan plan, density, the route,
the report) reads wells through it. The shape of a well is derived here, once,
from ``useq`` (its registry, its names, the forward model of M1 §8): nothing
else re-derives where a well is or whether a point is inside it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import useq
from pydantic import Field, field_validator

from smc.plugins.api import PluginResult
from smc.plugins.slots.base import SlotValue

__all__ = [
    "LatticeFit",
    "PlateResult",
    "WallTouch",
    "WellGeometry",
    "WellRef",
    "normalise_well",
    "well_geometry",
]

_WELL = re.compile(r"([A-Za-z]+)\s*([0-9]+)")


class WellRef(SlotValue):
    """A stage position the operator (or a finder) gave for a well's centre, in µm."""

    well: str
    x_um: float
    y_um: float


class WallTouch(SlotValue):
    """A stage position at which a well's wall was found, on one of its four sides."""

    well: str
    side: Literal["left", "right", "top", "bottom"]
    x_um: float
    y_um: float


class LatticeFit(SlotValue):
    """How well the lattice of wells fits what plate finding sampled.

    ``runner_up`` and ``margin`` are ``None`` when nothing else was tried: an
    unknown is not a zero.
    """

    score: float = Field(ge=0.0, le=1.0)
    polarity: Literal[1, -1]
    runner_up: str | None
    margin: float | None
    n_samples: int = Field(ge=0)
    n_inside: int = Field(ge=0)
    notes: tuple[str, ...] = ()


class PlateResult(PluginResult):
    """Where the plate is: A1's centre and the plate's rotation, and how that was found.

    ``residual_um`` is the worst distance between a reference and the fitted
    well centre; ``None`` when nothing measured it (a seed, a lattice fit).
    ``scale_error`` is the measured spacing over the nominal, minus 1; ``None``
    with fewer than two references. Both are required fields that may hold
    ``None``: a producer must say that it does not know.
    """

    plate: str
    a1_center_xy_um: tuple[float, float]
    rotation_deg: float = Field(ge=-180.0, le=180.0)
    method: Literal["seed", "references", "lattice-fit"]
    residual_um: float | None = Field(ge=0.0)
    scale_error: float | None
    n_refs: int = Field(ge=0)
    refs: tuple[WellRef, ...] = ()
    fit: LatticeFit | None = None

    @field_validator("plate")
    @classmethod
    def _plate_is_one_useq_knows(cls, value: str) -> str:
        """Refuse a plate name ``useq`` has no definition for, and keep the text as given.

        Names are exact (``96-WELL`` is refused, as ``useq`` refuses it), so
        the run document names a plate that every reader can build.
        """
        useq.WellPlate.from_str(value)
        return value

    def to_plan(self, **kwargs: Any) -> useq.WellPlatePlan:
        """The registration as a ``useq`` plan; ``kwargs`` add the wells and the grid.

        ``selected_wells`` and ``grid_plan`` are the scan plan's to give: the
        plate result says where the plate is, not which wells to image.
        """
        return useq.WellPlatePlan(
            plate=useq.WellPlate.from_str(self.plate),
            a1_center_xy=self.a1_center_xy_um,
            rotation=self.rotation_deg,
            **kwargs,
        )

    def well_center_um(self, well: str) -> tuple[float, float]:
        """The stage position of a well's centre, in µm.

        Raises:
            ValueError: The well is not on the plate.
        """
        return well_geometry(self.to_plan(), well).center_um


@dataclass(frozen=True, slots=True)
class WellGeometry:
    """Where a well is and what shape it has: the one place that knows.

    Density, the route, the report and the scan plan's clipping use ``contains``
    (density's raster, its vectorised twin ``contains_many``) and nothing else
    re-derives a well's shape. ``size_um`` is ``(width,
    height)`` from the plate's ``well_size``; ``rotation_deg`` is the plate's,
    in which a square well is axis-aligned.
    """

    well: str
    center_um: tuple[float, float]
    size_um: tuple[float, float]
    circular: bool
    rotation_deg: float

    def to_plate_frame(self, x_um: float, y_um: float) -> tuple[float, float]:
        """The offset of a stage point from the well's centre, in the plate's axes.

        The offset is turned by ``-rotation_deg``, so that the walls of a
        square well are parallel to the axes (as ``PlateSample._locate``).
        """
        dx, dy = x_um - self.center_um[0], y_um - self.center_um[1]
        angle = math.radians(self.rotation_deg)
        cos, sin = math.cos(angle), math.sin(angle)
        return (cos * dx + sin * dy, -sin * dx + cos * dy)

    def contains(self, x_um: float, y_um: float, *, margin_um: float = 0.0) -> bool:
        """Whether a stage point is inside the well shrunk by ``margin_um``; the edge counts.

        A disc of radius ``width / 2 - margin`` (the registry's round wells
        have equal sizes), or the rectangle of half-sizes minus the margin in
        the plate's axes. A margin at or above the half-size leaves nothing,
        and a point that is not finite is outside: unknown is not inside.

        Raises:
            ValueError: ``margin_um`` is negative or not finite.
        """
        if not (math.isfinite(margin_um) and margin_um >= 0.0):
            raise ValueError(
                f"margin_um must be finite and not negative, got {margin_um!r}"
            )
        if not (math.isfinite(x_um) and math.isfinite(y_um)):
            return False
        u, v = self.to_plate_frame(x_um, y_um)
        half_u = self.size_um[0] / 2.0 - margin_um
        if self.circular:
            return half_u > 0.0 and u * u + v * v <= half_u * half_u
        half_v = self.size_um[1] / 2.0 - margin_um
        return half_u > 0.0 and half_v > 0.0 and abs(u) <= half_u and abs(v) <= half_v

    def contains_many(
        self,
        x_um: npt.NDArray[np.float64],
        y_um: npt.NDArray[np.float64],
        *,
        margin_um: float = 0.0,
    ) -> npt.NDArray[np.bool_]:
        """``contains`` for many points at once: the same answer for each, the edge included.

        ``x_um`` and ``y_um`` broadcast against each other, so a row against a
        column is a grid and the result has the broadcast shape. The arithmetic
        is ``contains``' own, in the same order, because two correct definitions
        of "inside" differ by one ulp on the edge (FM-132); the test holds the
        two to each other, margins and points that are not finite included. No
        matrix product on the coordinates (FM-131).

        Raises:
            ValueError: ``margin_um`` is negative or not finite.
        """
        if not (math.isfinite(margin_um) and margin_um >= 0.0):
            raise ValueError(
                f"margin_um must be finite and not negative, got {margin_um!r}"
            )
        x = np.asarray(x_um, dtype=np.float64)
        y = np.asarray(y_um, dtype=np.float64)
        shape = np.broadcast_shapes(x.shape, y.shape)
        half_u = self.size_um[0] / 2.0 - margin_um
        half_v = self.size_um[1] / 2.0 - margin_um
        if half_u <= 0.0 or (not self.circular and half_v <= 0.0):
            return np.zeros(shape, dtype=np.bool_)
        angle = math.radians(self.rotation_deg)
        cos, sin = math.cos(angle), math.sin(angle)
        # numpy warns `overflow` and `invalid value` where Python's floats go to
        # `inf` and `nan` without a word: a point at 1e200 is outside, not a warning.
        with np.errstate(over="ignore", invalid="ignore"):
            dx = x - self.center_um[0]
            dy = y - self.center_um[1]
            u = cos * dx + sin * dy
            v = -sin * dx + cos * dy
            if self.circular:
                inside = u * u + v * v <= half_u * half_u
            else:
                inside = (np.abs(u) <= half_u) & (np.abs(v) <= half_v)
        return np.asarray(inside & np.isfinite(x) & np.isfinite(y), dtype=np.bool_)


def well_geometry(plan: useq.WellPlatePlan, well: str) -> WellGeometry:
    """The geometry of one well of a plan, whether or not the plan selects it.

    The centre is the forward model of M1 §8 from ``plan.a1_center_xy`` and the
    plate's pitch: ``A1 + R(+theta) * (col * pitch_x, -row * pitch_y)``. A plan
    built without a rotation has ``rotation None`` (measured), read as 0.

    Raises:
        ValueError: The well is not on the plate; the message names it as given.
    """
    plate = plan.plate
    names = np.asarray(plan.all_well_names)
    found = np.argwhere(names == normalise_well(well))
    if len(found) == 0:
        raise ValueError(f"well {well!r} is not on the {plate.name} plate")
    row, col = int(found[0][0]), int(found[0][1])
    rotation_deg = float(plan.rotation) if plan.rotation is not None else 0.0
    angle = math.radians(rotation_deg)
    # useq keeps well_size and well_spacing in mm, and a1_center_xy in µm.
    dx = col * plate.well_spacing[0] * 1000.0
    dy = -row * plate.well_spacing[1] * 1000.0
    a1_x, a1_y = plan.a1_center_xy
    centre = (
        a1_x + math.cos(angle) * dx - math.sin(angle) * dy,
        a1_y + math.sin(angle) * dx + math.cos(angle) * dy,
    )
    return WellGeometry(
        well=str(names[row, col]),
        center_um=centre,
        size_um=(plate.well_size[0] * 1000.0, plate.well_size[1] * 1000.0),
        circular=plate.circular_wells,
        rotation_deg=rotation_deg,
    )


def normalise_well(name: str) -> str:
    """``a1``, ``A 1`` and ``A01`` are ``A1``: what a person types, as ``useq`` names it.

    Ported from ``nikon-control``'s ``wells.normalise``. Anything that is not
    letters then digits (``glass``, ``B2-B5``, ``A1_0003``) is only stripped
    and upper-cased. It never raises and does not check that the well exists:
    ``well_geometry`` does. The digits are stripped of leading zeros by text,
    not by ``int()``, which refuses a very long run of them.
    """
    text = str(name).strip()
    match = _WELL.fullmatch(text)
    if match is None:
        return text.upper()
    return f"{match.group(1).upper()}{match.group(2).lstrip('0') or '0'}"
