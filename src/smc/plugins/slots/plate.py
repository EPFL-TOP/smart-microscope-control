"""The plate slot: where the plate is, and the shape of its wells (m3 §3.2).

Plate registration and plate finding both fill the ``plate`` slot with a
``PlateResult``, and everything downstream (the scan plan, density, the route,
the report) reads wells through it. The shape of a well is derived here, once,
from ``useq`` (its registry, its names, the forward model of M1 §8): nothing
else re-derives where a well is or whether a point is inside it.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import useq
from pydantic import Field, field_validator, model_validator

from smc.plugins.api import PluginResult
from smc.plugins.geometry import wrap_deg
from smc.plugins.slots.base import SlotValue

__all__ = [
    "IMAGE_SIDES",
    "ORIENTATION_DEG",
    "ImageSide",
    "LatticeFit",
    "PlateOrientation",
    "PlateResult",
    "WallFrame",
    "WallImages",
    "WallSide",
    "WallTouch",
    "WellFit",
    "WellGeometry",
    "WellRef",
    "holder_estimate",
    "normalise_well",
    "refuse_image_side",
    "turned_well",
    "well_geometry",
]

_WELL = re.compile(r"([A-Za-z]+)\s*([0-9]+)")

#: Which way the plate lies in its holder, in the stage's terms (m3 §24 D1).
#: ``not-turned`` is ``useq``'s layout on the stage: A1 at the -X +Y corner,
#: columns towards +X, rows towards -Y. ``turned-180`` is that plate turned 180°:
#: A1 at the +X -Y corner. Text, not ``0``/``180``: a number literal cannot be
#: given on the command line (pydantic does not turn "180" into an int literal).
PlateOrientation = Literal["not-turned", "turned-180"]

#: The rotation each orientation stands for; a fit's tilt is measured from it.
ORIENTATION_DEG: Mapping[str, float] = MappingProxyType(
    {"not-turned": 0.0, "turned-180": 180.0}
)

#: The bound between a plate's rotation and its orientation's (m3 §24 D2): a
#: result beyond it says one thing in ``rotation_deg`` and another in
#: ``orientation``, and is refused rather than read either way.
_MAX_FROM_ORIENTATION_DEG = 90.0


def _turned(offset: tuple[float, float], rotation_deg: float) -> tuple[float, float]:
    """``offset`` turned by ``rotation_deg`` counter-clockwise, X right and Y up: as ``useq`` turns a plan about A1.

    Measured on ``useq``: at 90° the 96-well's A12, (99 000, 0) µm from A1, lands at
    (0, 99 000). The two orientations are 0° and 180°, whose cosine and sine are
    exact once rounded: a turned estimate's A1 is then the holder's centre plus the
    offset to the micrometre, and the page's geometry cache's key stays one value.
    """
    angle = math.radians(rotation_deg)
    cos, sin = round(math.cos(angle), 12), round(math.sin(angle), 12)
    x, y = offset
    return (cos * x - sin * y, sin * x + cos * y)


@functools.lru_cache(maxsize=8)
def _grid_centre_offset(plate: str) -> tuple[float, float]:
    """The centre of ``plate``'s grid of wells, as an offset from A1 (the mean of the wells' positions).

    A holder's point is the centre of the grid and a plan is placed by A1, so the
    estimate needs this offset: ``(49 500, -31 500)`` µm for 96 wells, whose 12 by 8
    grid is not square, so a swapped axis or a sign shows (FM-73). Kept per plate:
    the positions of 1536 wells cost 5 ms, small but paid by every refresh.

    Raises:
        ValueError: ``useq`` does not know the plate, or one of its wells has no
            position.
    """
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate), a1_center_xy=(0.0, 0.0)
    )
    xs: list[float] = []
    ys: list[float] = []
    for position in plan.all_well_positions:
        # ``useq`` types a position's axes as optional; a well of a plate plan
        # always has both, and a missing one must not skew the mean unseen.
        if position.x is None or position.y is None:
            raise ValueError(f"a well of the {plate} plate has no position in useq")
        xs.append(position.x)
        ys.append(position.y)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def holder_estimate(
    plate: str,
    centre_xy_um: tuple[float, float],
    orientation: PlateOrientation,
) -> useq.WellPlatePlan:
    """The plate as the stand's holder puts it: the grid's centre on ``centre_xy_um`` (m3 §5.5 D2).

    The one estimate before a registration, which the plate page draws and
    ``goto-well`` drives to, so that the two never disagree. The plan's rotation is
    ``ORIENTATION_DEG[orientation]`` and its A1 the holder's centre minus the grid's
    centre offset from A1 turned by it (for ``turned-180``, plus the offset): the
    grid's centre stays on the holder, and a turn renames the wells in place. It is
    a place to draw and to drive to by eye, never a registration: it fills no slot.

    Raises:
        ValueError: ``useq`` does not know the plate (its own text passes), or a
            well of it has no position.
    """
    rotation = ORIENTATION_DEG[orientation]
    turned_x, turned_y = _turned(_grid_centre_offset(plate), rotation)
    return useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate),
        a1_center_xy=(centre_xy_um[0] - turned_x, centre_xy_um[1] - turned_y),
        rotation=rotation,
    )


def turned_well(plate: str, well: str) -> str:
    """The name of the well at ``well``'s place once the plate is turned 180°.

    Row ``r`` becomes ``R-1-r`` and column ``c`` becomes ``C-1-c`` (B3 is C4 on a
    24-well): a plate whose wells form a uniform grid lands on itself when turned,
    so a turn only renames (m3 §24.1). The row and column are found in ``useq``'s
    names, never by parsing letters (§3.2), because a 1536-well's rows run to AF.

    Raises:
        ValueError: The plate is unknown to ``useq`` or the well is not on it.
    """
    names = np.asarray(useq.WellPlate.from_str(plate).all_well_names)
    found = np.argwhere(names == normalise_well(well))
    if len(found) == 0:
        raise ValueError(f"well {well!r} is not on the {plate} plate")
    rows, cols = names.shape
    row, col = int(found[0][0]), int(found[0][1])
    return str(names[rows - 1 - row, cols - 1 - col])


#: A wall of a well by the stage direction that reaches it from the well's centre.
#: Never an image side: ``+X`` is the wall the stage reaches by going +X, whatever
#: the camera's turn, so no operator names a side on the image-wall path (m3 §17.2).
WallSide = Literal["+X", "+Y", "-X", "-Y"]

#: A side as the operator sees it on the image. These words exist on the plate page
#: only, which maps them to a ``WallSide`` with the camera calibration (m3 §19.3);
#: the layer never takes one, because the same word in two frames was FM-189.
ImageSide = Literal["left", "right", "top", "bottom"]
IMAGE_SIDES: tuple[ImageSide, ...] = ("left", "right", "top", "bottom")


def refuse_image_side(value: Any) -> Any:
    """Refuse an image word where a stage direction is wanted, and say why.

    Used as a ``mode="before"`` validator: a ``Literal`` alone would answer
    ``left`` with its generic "input should be '+X', ..." and hide that the word
    is a real side in the *other* frame. Anything else is returned unchanged, so
    the ``Literal``'s own error still stands for a word that is no side at all.
    """
    if isinstance(value, str) and value in IMAGE_SIDES:
        raise ValueError(
            f"side {value!r} is a side on the image: a wall is named by a stage "
            "direction, -X, +X, -Y or +Y (the plate page names walls on the image "
            "and maps them with the camera calibration)"
        )
    return value


class WellRef(SlotValue):
    """A stage position the operator (or a finder) gave for a well's centre, in µm."""

    well: str
    x_um: float
    y_um: float


class WallTouch(SlotValue):
    """A stage position at which a well's wall was found, on one of its four sides.

    ``side`` is a stage direction (``-X``, ``+X``, ``-Y``, ``+Y``): the wall the
    stage reaches by going that way from the well's centre. A side named on the
    image is refused, not translated (m3 §19.2).
    """

    well: str
    side: WallSide
    x_um: float
    y_um: float

    @field_validator("side", mode="before")
    @classmethod
    def _side_is_not_an_image_side(cls, value: Any) -> Any:
        return refuse_image_side(value)


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


class WallFrame(SlotValue):
    """The frame whose edge points were used for one wall of a well (m3 §17.2).

    It is the frame centred on the wall, the last of the side's walk: every
    side is framed the same way, so the threshold's bias is the same on
    opposite walls and cancels in the centre. ``x_um`` and ``y_um`` are the
    stage's readback where it was taken, never the target (FM-142). ``frames``
    counts every frame the side took, the walk and the centred one; ``file`` is
    its name in ``WallImages.folder``, ``None`` when frames are not saved or
    this one could not be.
    """

    side: WallSide
    x_um: float
    y_um: float
    n_points: int = Field(ge=0)
    frames: int = Field(ge=1)
    file: str | None = None


class WellFit(SlotValue):
    """One well as its walls measured it: where the operator put the stage, and the circle fitted.

    ``center_xy_um``, ``diameter_um``, ``rms_um`` and ``n_points`` (the points
    kept) are the circle fit's; ``size_error`` is the diameter over the
    nominal, minus 1. ``walls`` are in the order ``+X``, ``+Y``, ``-X``, ``-Y``.
    """

    well: str
    recorded_xy_um: tuple[float, float]
    center_xy_um: tuple[float, float]
    diameter_um: float = Field(gt=0.0)
    size_error: float
    rms_um: float = Field(ge=0.0)
    n_points: int = Field(ge=0)
    walls: tuple[WallFrame, ...]


class WallImages(SlotValue):
    """What ``plate-walls`` measured, kept beside the plate it fitted (m3 §17.2).

    ``objective`` is the one in place (``None`` without a turret);
    ``pixel_size_um`` and ``image_shape_px`` are the calibration's. ``folder`` is
    run-relative (``plate-walls/<token>``), ``None`` when frames are not saved;
    ``elapsed_s`` runs from the first move to the result. Both ``objective``
    and ``folder`` are required fields that may hold ``None``: a producer must
    say that it does not know.
    """

    objective: str | None
    pixel_size_um: float = Field(gt=0.0)
    image_shape_px: tuple[int, int]
    search_um: float = Field(gt=0.0)
    wells: tuple[WellFit, ...]
    frames: int = Field(ge=0)
    folder: str | None
    elapsed_s: float = Field(ge=0.0)


class PlateResult(PluginResult):
    """Where the plate is: A1's centre and the plate's rotation, and how that was found.

    ``residual_um`` is the worst distance between a reference and the fitted
    well centre; ``None`` when nothing measured it (a seed, a lattice fit).
    ``scale_error`` is the measured spacing over the nominal, minus 1; ``None``
    with fewer than two references. Both are required fields that may hold
    ``None``: a producer must say that it does not know. ``walls`` is filled
    by ``method="wall-images"`` only; the ``refs`` of that method are the
    fitted well centres, and its ``fit`` is ``None``. A result written before
    ``walls`` existed has none, and still validates.

    ``objective`` is the turret label of the objective the plate's positions
    were taken with (m3 §18.1): two objectives do not centre the same feature
    at the same stage position, so a tool that uses the plate with another
    one needs to know which it was. ``None`` is *unknown*, not "the one in
    place": a stand with no turret, and every result written before ``objective``
    existed. It is the plate's own objective for every method, and has a
    default so that older run documents still load.

    ``orientation`` is which way the plate lies in its holder, as the operator
    declared it (m3 §24 D2): the samples follow the names printed on the plate,
    and a plate turned 180° lands on itself, so nothing measured tells the two
    apart. A result written before it existed reads ``not-turned``, which is what
    every earlier fit was checked against. ``rotation_deg`` is the full rotation
    (about 180° on a turned plate); the validator refuses one more than 90° from
    the orientation's, so the two fields never contradict each other.
    """

    plate: str
    a1_center_xy_um: tuple[float, float]
    rotation_deg: float = Field(ge=-180.0, le=180.0)
    method: Literal["seed", "references", "lattice-fit", "wall-images"]
    residual_um: float | None = Field(ge=0.0)
    scale_error: float | None
    n_refs: int = Field(ge=0)
    refs: tuple[WellRef, ...] = ()
    fit: LatticeFit | None = None
    walls: WallImages | None = None
    objective: str | None = None
    orientation: PlateOrientation = "not-turned"

    @model_validator(mode="after")
    def _rotation_agrees_with_orientation(self) -> PlateResult:
        delta = wrap_deg(self.rotation_deg - ORIENTATION_DEG[self.orientation])
        # `not <=`: a nan never reaches here (allow_inf_nan=False), but the
        # comparison refuses one anyway rather than let it through.
        if not abs(delta) <= _MAX_FROM_ORIENTATION_DEG:
            raise ValueError(
                f"rotation_deg {self.rotation_deg:g} is {delta:.1f}° from "
                f"orientation {self.orientation} "
                f"({ORIENTATION_DEG[self.orientation]:g}°), beyond "
                f"±{_MAX_FROM_ORIENTATION_DEG:g}°: the two contradict each other"
            )
        return self

    def turned(self) -> PlateResult:
        """The same plate turned 180° about its grid's centre: every well renamed, nothing moved.

        A1 moves to the old centre of the last well and the rotation gains 180°,
        so each well position stays and only its name changes (m3 §24 D6). Every
        well name the result holds is renamed (``refs``, ``walls.wells``);
        ``residual_um``, ``scale_error``, ``fit``, ``objective`` and ``method`` are
        kept, because nothing is measured again. ``assumed`` keeps its lines and
        says that their well names are the fit's.
        """
        names = useq.WellPlate.from_str(self.plate).all_well_names
        last = str(names[-1][-1])
        new: PlateOrientation = (
            "turned-180" if self.orientation == "not-turned" else "not-turned"
        )
        refs = tuple(
            ref.model_copy(update={"well": turned_well(self.plate, ref.well)})
            for ref in self.refs
        )
        walls = self.walls
        if walls is not None:
            walls = walls.model_copy(
                update={
                    "wells": tuple(
                        fit.model_copy(
                            update={"well": turned_well(self.plate, fit.well)}
                        )
                        for fit in walls.wells
                    )
                }
            )
        line = (
            f"turned 180° to {new} after the fit: every well renamed (row r to "
            f"R-1-r, column c to C-1-c), nothing measured again; a well named in a "
            f"line above is the fit's name"
        )
        # Validated again, not `model_copy`: the new rotation and orientation are
        # checked against each other like any result's.
        return type(self).model_validate(
            {
                **{name: getattr(self, name) for name in type(self).model_fields},
                "a1_center_xy_um": self.well_center_um(last),
                "rotation_deg": wrap_deg(self.rotation_deg + 180.0),
                "orientation": new,
                "refs": refs,
                "walls": walls,
                "assumed": (*self.assumed, line),
            }
        )

    def with_orientation(self, orientation: PlateOrientation) -> PlateResult:
        """This result declared ``orientation``: itself when it already is, else ``turned()``.

        Absolute, not a toggle: asking for the fit's own orientation gives the
        fit's numbers back, so running ``plate-orientation`` twice with the same
        value changes nothing (m3 §24 D6).
        """
        return self if orientation == self.orientation else self.turned()

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
