"""A synthetic plate sample and a camera that images it (design §8, #10).

The demo camera ignores the stage: every frame is the same noise. Code that
looks at images to decide where the stage should go (plate finding, the
camera-to-stage calibration, detection and tracking) cannot be tested on it.
``PlateSample`` is a plate that exists at stage coordinates, and
``SampleCamera`` is a ``Camera`` that renders the part of it under the stage's
current XY position, so frames follow the stage the way a real camera's do:
the brightness changes across a well wall, a feature moves ``Δ / pixel_size``
pixels when the stage moves ``Δ``, and an object sits where its stage position
says.

What a frame is made of, in order: a level (well or plastic) with a hard edge,
a texture that is a function of the world position only (so two overlapping
frames correlate), Gaussian blobs, a Gaussian blur that grows with ``|z|``,
the exposure, Gaussian noise and the cast to ``uint16``. Well geometry is read
from ``useq.WellPlatePlan``, never re-derived.

Limits, so nobody mistakes it for a simulation of optics:

* The sample is **flat**: one plane at z = 0, plate coordinates equal stage
  coordinates. There is no PSF, shading or illumination, and the z blur is a
  stand-in for defocus, not a z-stack.
* There is no sub-pixel stage resolution, and the texture is built from 4 µm
  tiles: it is meant for pixel sizes up to 4 µm, and above that only
  whole-pixel shifts correlate.
* ``SampleCamera`` is installed with ``Microscope.override`` and so bypasses
  the ``Executor``: no lock, no dry-run, no halt and no motion guard, and it
  is not thread-safe (it owns a random generator).

Axes (what the camera-to-stage calibration will later measure): the stage
position is the plate point at the centre of the frame; columns grow with +X
and rows with -Y, so row 0 is the +Y side. A +10 µm X move shifts the content
``10 / pixel_size_um`` columns left, a +10 µm Y move shifts it as many rows
down.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import useq

__all__ = [
    "MAX_BLUR_PX",
    "REFERENCE_EXPOSURE_MS",
    "TEXTURE_TILE_UM",
    "Blob",
    "PlateSample",
]

#: The demo's exposure: ``well_level`` and ``plastic_level`` are the counts at it.
REFERENCE_EXPOSURE_MS = 10.0
#: Side of one texture tile, in µm. The texture is constant inside a tile.
TEXTURE_TILE_UM = 4.0
#: The blur of a frame never exceeds this many pixels (``|z_um| / 10`` is capped).
MAX_BLUR_PX = 32.0

# Bounds, so that a typo (1e30) is refused instead of overflowing an int64
# tile index, a float64 frame or a gigabyte allocation (FM-32).
_MAX_POSITION_UM = 1e9
_MAX_PIXEL_SIZE_UM = 1e4
_MAX_ROTATION_DEG = 1e6
_MAX_LEVEL = 1e9
_MAX_EXPOSURE_MS = 1e6
_MAX_PIXELS = 2**24


def _bounded(name: str, value: float, low: float, high: float) -> float:
    if not (math.isfinite(value) and low <= value <= high):
        raise ValueError(
            f"{name} must be finite and within [{low:g}, {high:g}], got {value!r}"
        )
    return float(value)


def _positive(name: str, value: float, high: float) -> float:
    if not (math.isfinite(value) and 0.0 < value <= high):
        raise ValueError(
            f"{name} must be finite and positive (at most {high:g}), got {value!r}"
        )
    return float(value)


def _pair(name: str, value: Sequence[float], bound: float) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be two numbers (x, y), got {tuple(value)!r}")
    return (
        _bounded(f"{name}[0]", value[0], -bound, bound),
        _bounded(f"{name}[1]", value[1], -bound, bound),
    )


@dataclass(frozen=True, slots=True)
class Blob:
    """A Gaussian spot at a stage position, ``intensity * exp(-d^2 / (2 sigma^2))``.

    ``sigma`` is ``radius_um / 2``, so the spot is negligible beyond twice its
    radius. Counts are at the reference exposure, like the sample's levels. A
    negative ``intensity`` is a dark spot.
    """

    x_um: float
    y_um: float
    radius_um: float
    intensity: float

    def __post_init__(self) -> None:
        _bounded("x_um", self.x_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        _bounded("y_um", self.y_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        _positive("radius_um", self.radius_um, _MAX_POSITION_UM)
        _bounded("intensity", self.intensity, -_MAX_LEVEL, _MAX_LEVEL)


class PlateSample:
    """A well plate at stage coordinates, rendered as camera frames.

    The plate is placed by the stage position of well A1's centre and a
    rotation, counter-clockwise with Y up, about A1: ``world = A1 + R(theta) *
    (col * pitch_x, -row * pitch_y)``. Rows run towards -Y and columns towards
    +X, as in ``useq``. ``well_level`` and ``plastic_level`` are counts at
    ``REFERENCE_EXPOSURE_MS``.

    Args:
        plate: A ``useq`` plate name (``"96-well"``) or a ``useq.WellPlate``.
        a1_center_xy_um: Stage position of A1's centre, in µm.
        rotation_deg: The plate's rotation about A1, in degrees.
        well_level: Counts inside a well.
        plastic_level: Counts between wells and outside the plate.
        blobs: Gaussian spots, in stage coordinates.
        texture_std: Standard deviation of the position-locked texture.
        noise_std: Standard deviation of the frame noise, which is *not*
            position-locked: successive frames differ.
        seed: Seeds the texture and the noise generator. Two samples with the
            same seed give the same texture and the same sequence of noise.

    Raises:
        ValueError: A level or deviation is negative or not finite, a blob is
            invalid, the plate is unknown, or its ``well_size`` exceeds its
            ``well_spacing`` (the nearest-well search assumes each well lies
            in its own pitch cell).
    """

    def __init__(
        self,
        plate: str | useq.WellPlate = "96-well",
        a1_center_xy_um: tuple[float, float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        *,
        well_level: float = 3000.0,
        plastic_level: float = 800.0,
        blobs: Sequence[Blob] = (),
        texture_std: float = 40.0,
        noise_std: float = 20.0,
        seed: int = 0,
    ) -> None:
        self.a1_center_xy_um = _pair(
            "a1_center_xy_um", a1_center_xy_um, _MAX_POSITION_UM
        )
        self.rotation_deg = _bounded(
            "rotation_deg", rotation_deg, -_MAX_ROTATION_DEG, _MAX_ROTATION_DEG
        )
        self.well_level = _bounded("well_level", well_level, 0.0, _MAX_LEVEL)
        self.plastic_level = _bounded("plastic_level", plastic_level, 0.0, _MAX_LEVEL)
        self.texture_std = _bounded("texture_std", texture_std, 0.0, _MAX_LEVEL)
        self.noise_std = _bounded("noise_std", noise_std, 0.0, _MAX_LEVEL)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = seed
        self.blobs: tuple[Blob, ...] = tuple(blobs)

        plan = useq.WellPlatePlan(
            # useq's validator turns a registered name into a WellPlate; its
            # field is typed for the result only.
            plate=plate,  # type: ignore[arg-type]
            a1_center_xy=self.a1_center_xy_um,
            rotation=self.rotation_deg,
        )
        well_plate = plan.plate
        size_mm, spacing_mm = well_plate.well_size, well_plate.well_spacing
        if size_mm[0] > spacing_mm[0] or size_mm[1] > spacing_mm[1]:
            raise ValueError(
                f"the plate's well_size {size_mm} mm exceeds its well_spacing "
                f"{spacing_mm} mm: a well must lie inside its own pitch cell"
            )
        self.plate_name: str = well_plate.name
        self._rows, self._columns = well_plate.rows, well_plate.columns
        self._circular = well_plate.circular_wells
        # useq keeps well_size and well_spacing in mm and a1_center_xy in µm.
        self._pitch_um = (spacing_mm[0] * 1000.0, spacing_mm[1] * 1000.0)
        self._half_size_um = (size_mm[0] * 500.0, size_mm[1] * 500.0)
        self._names = plan.all_well_names
        angle = math.radians(self.rotation_deg)
        self._cos, self._sin = math.cos(angle), math.sin(angle)

    def well_at(self, x_um: float, y_um: float) -> str | None:
        """The name of the well (``"B3"``) at a stage position, or ``None`` on plastic.

        The ground truth the geometry tests compare against.
        """
        x = _bounded("x_um", x_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        y = _bounded("y_um", y_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        inside, row, col = self._locate(np.array([x]), np.array([y]))
        if not inside[0]:
            return None
        return str(self._names[row[0], col[0]])

    def _locate(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Whether each stage position is inside a well, and the nearest well's row and column.

        Rotates into plate coordinates, rounds to the nearest well and compares
        the offset with the well's size. The row and column are only meaningful
        where ``inside`` is true (elsewhere they may lie outside the plate).
        """
        dx = x - self.a1_center_xy_um[0]
        dy = y - self.a1_center_xy_um[1]
        ux = self._cos * dx + self._sin * dy
        uy = -self._sin * dx + self._cos * dy
        col = np.rint(ux / self._pitch_um[0]).astype(np.int64)
        row = np.rint(-uy / self._pitch_um[1]).astype(np.int64)
        on_plate = (row >= 0) & (row < self._rows) & (col >= 0) & (col < self._columns)
        ox = ux - col * self._pitch_um[0]
        oy = uy + row * self._pitch_um[1]
        if self._circular:
            within = ox * ox + oy * oy <= self._half_size_um[0] ** 2
        else:
            within = (np.abs(ox) <= self._half_size_um[0]) & (
                np.abs(oy) <= self._half_size_um[1]
            )
        return on_plate & within, row, col
