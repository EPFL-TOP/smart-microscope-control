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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import useq

from smc.hardware.capabilities import XYStage, ZStage

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "MAX_BLUR_PX",
    "REFERENCE_EXPOSURE_MS",
    "TEXTURE_TILE_UM",
    "Blob",
    "PlateSample",
    "SampleCamera",
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
# A blur runs on the frame padded by 4 sigma on each side: a frame at the
# bound with the maximum blur (4096 x 4096, 4352 x 4352 padded) must fit, a
# single row of 2**24 pixels padded to 4e9 values must not.
_MAX_BLURRED_PIXELS = 2 * _MAX_PIXELS
# Below this, radius**2 underflows towards 0 and a pixel on the blob's centre
# gets 0 / 0 in the exponent (NaN, cast to uint16).
_MIN_RADIUS_UM = 1e-6


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


def _shape(shape: tuple[int, int]) -> tuple[int, int]:
    message = (
        "shape must be two positive integers (height, width) with at most "
        f"{_MAX_PIXELS} pixels, got {shape!r}"
    )
    try:
        height, width = shape
    except (TypeError, ValueError):
        raise ValueError(message) from None
    for side in (height, width):
        if isinstance(side, bool) or not isinstance(side, (int, np.integer)):
            raise ValueError(message)
        if side <= 0:
            raise ValueError(message)
    # Python ints: the product of two np.int32 (the default integer of numpy < 2
    # on Windows) wraps below the bound, and a small dtype's product warns.
    height, width = int(height), int(width)
    if height * width > _MAX_PIXELS:
        raise ValueError(message)
    return height, width


def _pair(name: str, value: Sequence[float], bound: float) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be two numbers (x, y), got {tuple(value)!r}")
    return (
        _bounded(f"{name}[0]", value[0], -bound, bound),
        _bounded(f"{name}[1]", value[1], -bound, bound),
    )


_MASK64 = (1 << 64) - 1


def _tile_values(
    ix: npt.NDArray[np.int64], iy: npt.NDArray[np.int64], seed: int
) -> npt.NDArray[np.float64]:
    """The texture of tiles ``(ix, iy)``: zero mean, unit variance, a function of the tile only.

    SplitMix64's finaliser over the tile indices and the seed. It is written on
    ``uint64`` *arrays*: a scalar ``np.uint64`` product warns on overflow, while
    the same product on an array wraps silently, and every dtype is explicit
    because numpy < 2 on Windows defaults to a 32-bit integer. The result is
    therefore bit-identical on every platform, and a test pins its values.
    """
    a = np.atleast_1d(ix).astype(np.int64).view(np.uint64)
    b = np.atleast_1d(iy).astype(np.int64).view(np.uint64)
    # The seed term is computed in Python ints, where nothing overflows.
    seed_term = np.uint64((seed * 0x165667B19E3779F9) & _MASK64)
    z = a * np.uint64(0x9E3779B97F4A7C15) + b * np.uint64(0xC2B2AE3D27D4EB4F)
    z += seed_term
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    # The top 53 bits as a uniform on [0, 1), centred, scaled to unit variance.
    uniform = (z >> np.uint64(11)).astype(np.float64) * 2.0**-53
    return (uniform - 0.5) * math.sqrt(12.0)


@dataclass(frozen=True, slots=True)
class _Pose:
    """Where a camera is and how its axes point: the map between pixels and the stage plane.

    The stage position is the plate point at the centre of the frame. Columns
    grow with +X and rows with -Y (row 0 is the +Y side). ``mirrored`` flips
    the columns, and the rotation, counter-clockwise, comes after it:
    ``world = (x, y) + R(rotation) * (+-dx, dy)``.
    """

    x_um: float
    y_um: float
    shape: tuple[int, int]  # (height, width)
    pixel_size_um: float
    rotation_deg: float
    mirrored: bool

    def _rotation(self) -> tuple[float, float]:
        angle = math.radians(self.rotation_deg)
        return math.cos(angle), math.sin(angle)

    def world(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The stage position of every pixel centre, as two ``shape`` arrays (X, Y)."""
        height, width = self.shape
        cos, sin = self._rotation()
        step = self.pixel_size_um
        cols = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * step
        rows = -(np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * step
        dx = (-cols if self.mirrored else cols)[None, :]
        dy = rows[:, None]
        return (
            self.x_um + (cos * dx - sin * dy),
            self.y_um + (sin * dx + cos * dy),
        )

    def pixel_of(self, x_um: float, y_um: float) -> tuple[float, float]:
        """The ``(col, row)`` a stage position falls on, fractional and possibly off the frame."""
        height, width = self.shape
        cos, sin = self._rotation()
        offset_x, offset_y = x_um - self.x_um, y_um - self.y_um
        # Undo the rotation, then the mirror.
        camera_x = cos * offset_x + sin * offset_y
        camera_y = -sin * offset_x + cos * offset_y
        col = (-camera_x if self.mirrored else camera_x) / self.pixel_size_um
        row = -camera_y / self.pixel_size_um
        return col + (width - 1) / 2.0, row + (height - 1) / 2.0


def _gaussian_blur(
    image: npt.NDArray[np.float64], sigma_px: float
) -> npt.NDArray[np.float64]:
    """Blur by multiplying the spectrum with a Gaussian: periodic, so the caller pads.

    The transfer function of a Gaussian of ``sigma_px`` is ``exp(-2 pi^2
    sigma^2 f^2)``. No ``scipy``: it is not a dependency of this package.
    """
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.rfftfreq(image.shape[1])[None, :]
    transfer = np.exp(-2.0 * math.pi**2 * sigma_px**2 * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(image) * transfer, s=image.shape)


def _add_blob(
    signal: npt.NDArray[np.float64],
    world_x: npt.NDArray[np.float64],
    world_y: npt.NDArray[np.float64],
    blob: Blob,
    pose: _Pose,
) -> None:
    """Add a Gaussian spot inside the box of +-2 radii around it, touching only that window."""
    height, width = signal.shape
    col, row = pose.pixel_of(blob.x_um, blob.y_um)
    if not (math.isfinite(col) and math.isfinite(row)):
        return  # absurdly far away: finite stage position over a tiny pixel
    # A window that holds the whole box whatever the camera rotation.
    reach = 2.0 * math.sqrt(2.0) * blob.radius_um / pose.pixel_size_um
    half = int(min(reach, height + width)) + 2
    c0, c1 = max(0, math.floor(col) - half), min(width, math.floor(col) + half + 1)
    r0, r1 = max(0, math.floor(row) - half), min(height, math.floor(row) + half + 1)
    if c0 >= c1 or r0 >= r1:
        return
    gx = world_x[r0:r1, c0:c1] - blob.x_um
    gy = world_y[r0:r1, c0:c1] - blob.y_um
    box = 2.0 * blob.radius_um
    # sigma = radius / 2, so 2 sigma^2 = radius^2 / 2.
    bump = blob.intensity * np.exp(-(gx * gx + gy * gy) / (blob.radius_um**2 / 2.0))
    signal[r0:r1, c0:c1] += np.where(
        (np.abs(gx) <= box) & (np.abs(gy) <= box), bump, 0.0
    )


@dataclass(frozen=True, slots=True)
class Blob:
    """A Gaussian spot at a stage position, ``intensity * exp(-d^2 / (2 sigma^2))``.

    ``sigma`` is ``radius_um / 2``, so the spot is negligible beyond twice its
    radius, and ``radius_um`` is at least 1e-6 (below that the exponent
    underflows). Counts are at the reference exposure, like the sample's
    levels. A negative ``intensity`` is a dark spot, clipped at 0 counts.
    """

    x_um: float
    y_um: float
    radius_um: float
    intensity: float

    def __post_init__(self) -> None:
        _bounded("x_um", self.x_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        _bounded("y_um", self.y_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        _bounded("radius_um", self.radius_um, _MIN_RADIUS_UM, _MAX_POSITION_UM)
        _bounded("intensity", self.intensity, -_MAX_LEVEL, _MAX_LEVEL)


class PlateSample:
    """A well plate at stage coordinates, rendered as camera frames.

    The plate is placed by the stage position of well A1's centre and a
    rotation, counter-clockwise with Y up, about A1: ``world = A1 + R(theta) *
    (col * pitch_x, -row * pitch_y)``. Rows run towards -Y and columns towards
    +X, as in ``useq``. ``well_level`` and ``plastic_level`` are counts at
    ``REFERENCE_EXPOSURE_MS``. A circular well takes its diameter from the
    plate's ``well_size[0]``; a square one uses both sizes.

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
        # The sample's own generator: successive frames get fresh noise, and a
        # new sample with the same seed repeats the sequence.
        self._rng = np.random.default_rng(self.seed)

    def render(
        self,
        x_um: float,
        y_um: float,
        z_um: float,
        *,
        shape: tuple[int, int],
        pixel_size_um: float,
        camera_rotation_deg: float = 0.0,
        mirrored: bool = False,
        exposure_ms: float = REFERENCE_EXPOSURE_MS,
    ) -> npt.NDArray[np.uint16]:
        """One frame of the plate as a camera at a stage position would see it.

        ``(x_um, y_um)`` is the plate point at the centre of the frame, so a
        test can ask for any pose without a stage. Every call draws fresh noise
        from the sample's generator; texture and blobs depend on the position
        only, so a test that compares frames sets ``noise_std=0``.

        Args:
            x_um: Stage X, the plate point at the centre of the frame.
            y_um: Stage Y.
            z_um: Stage Z; the focus plane is z = 0 and the blur has no sign.
            shape: ``(height, width)`` in pixels.
            pixel_size_um: Object-space pixel size. Never defaulted: a pixel
                size that is not finite and positive is refused.
            camera_rotation_deg: Rotation of the camera about the optical
                axis, counter-clockwise, applied after the mirror.
            mirrored: Flip the image left-right (before the rotation).
            exposure_ms: Scales the signal, not the noise; the levels are the
                counts at ``REFERENCE_EXPOSURE_MS``.

        Returns:
            A ``uint16`` array of ``shape``.

        Raises:
            ValueError: An argument is not finite or is out of range.
        """
        x = _bounded("x_um", x_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        y = _bounded("y_um", y_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        z = _bounded("z_um", z_um, -_MAX_POSITION_UM, _MAX_POSITION_UM)
        height, width = _shape(shape)
        scale_um = _positive("pixel_size_um", pixel_size_um, _MAX_PIXEL_SIZE_UM)
        rotation = _bounded(
            "camera_rotation_deg",
            camera_rotation_deg,
            -_MAX_ROTATION_DEG,
            _MAX_ROTATION_DEG,
        )
        exposure = _bounded("exposure_ms", exposure_ms, 0.0, _MAX_EXPOSURE_MS)

        pose = _Pose(x, y, (height, width), scale_um, rotation, mirrored)
        # A stand-in for defocus: sharp at z = 0, symmetric, no sign.
        blur_px = min(abs(z) / 10.0, MAX_BLUR_PX)
        if blur_px > 0.0:
            # The FFT blur is periodic, so it runs on a frame padded by 4 sigma
            # on every side, with the same centre, and the padding is cropped:
            # the far edge never leaks into the near one.
            pad = math.ceil(4.0 * blur_px)
            padded_pixels = (height + 2 * pad) * (width + 2 * pad)
            if padded_pixels > _MAX_BLURRED_PIXELS:
                raise ValueError(
                    f"a blur of {blur_px:g} px (z_um={z!r}) pads shape "
                    f"{(height, width)!r} to {padded_pixels} pixels, more than "
                    f"the {_MAX_BLURRED_PIXELS} allowed: use a smaller shape or "
                    "a smaller |z_um|"
                )
            padded = replace(pose, shape=(height + 2 * pad, width + 2 * pad))
            blurred = _gaussian_blur(self._signal(padded), blur_px)
            signal = np.ascontiguousarray(
                blurred[pad : pad + height, pad : pad + width]
            )
        else:
            signal = self._signal(pose)
        signal *= exposure / REFERENCE_EXPOSURE_MS
        if self.noise_std > 0.0:
            signal += self._rng.normal(0.0, self.noise_std, size=signal.shape)
        return np.rint(np.clip(signal, 0.0, 65535.0)).astype(np.uint16)

    def _signal(self, pose: _Pose) -> npt.NDArray[np.float64]:
        """Level, texture and blobs at the reference exposure, before blur, scale and noise."""
        world_x, world_y = pose.world()
        inside, _, _ = self._locate(world_x, world_y)
        signal = np.where(
            inside, np.float64(self.well_level), np.float64(self.plastic_level)
        )
        if self.texture_std > 0.0:
            tile_x = np.floor(world_x / TEXTURE_TILE_UM).astype(np.int64)
            tile_y = np.floor(world_y / TEXTURE_TILE_UM).astype(np.int64)
            signal += self.texture_std * _tile_values(tile_x, tile_y, self.seed)
        for blob in self.blobs:
            _add_blob(signal, world_x, world_y, blob, pose)
        return signal

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


class SampleCamera:
    """A ``Camera`` whose frames follow the stage: it renders a ``PlateSample``.

    ``snap()`` reads the XY position of the stand's stage, and the Z position
    when it has a Z drive (0.0 otherwise), and renders the sample there. So a
    tool that moves the stage and looks at the result sees what a real camera
    would show, with the demo devices doing the moving.

    It is installed with ``Microscope.override(Camera, ...)`` (the
    ``demo_microscope_with_sample`` fixture does it), and therefore **bypasses
    the** ``Executor``: it takes no lock, ignores dry-run, the halt and the
    motion guard, and is not thread-safe, because the sample owns a random
    generator. It works on any ``Microscope``, a ``FakeCore``'s included. A test
    that needs a lock or a halt on the camera does not belong here.

    The exposure starts at ``REFERENCE_EXPOSURE_MS`` and scales the frame's
    signal, not its noise. ``bit_depth()`` is 16.

    Args:
        microscope: The stand whose stages say where the camera is looking.
        sample: What it looks at.
        pixel_size_um: Object-space pixel size. Required and never defaulted:
            a pixel size of 0 is refused, as it would be reported by a real
            camera that nothing calibrated.
        shape: ``(height, width)`` of a frame.
        camera_rotation_deg: Rotation of the camera about the optical axis.
        mirrored: Whether the image is flipped left-right.

    Raises:
        ValueError: ``pixel_size_um``, ``shape`` or ``camera_rotation_deg`` is
            not a usable number.
    """

    def __init__(
        self,
        microscope: Microscope,
        sample: PlateSample,
        *,
        pixel_size_um: float,
        shape: tuple[int, int] = (512, 512),
        camera_rotation_deg: float = 0.0,
        mirrored: bool = False,
    ) -> None:
        self._microscope = microscope
        self._sample = sample
        self._pixel_size_um = _positive(
            "pixel_size_um", pixel_size_um, _MAX_PIXEL_SIZE_UM
        )
        self._shape = _shape(shape)
        self._rotation_deg = _bounded(
            "camera_rotation_deg",
            camera_rotation_deg,
            -_MAX_ROTATION_DEG,
            _MAX_ROTATION_DEG,
        )
        self._mirrored = bool(mirrored)
        self._exposure_ms = REFERENCE_EXPOSURE_MS

    def snap(self) -> npt.NDArray[np.uint16]:
        """One frame of the sample at the stage's current position."""
        position = self._microscope.require(XYStage).position_um()
        z_stage = self._microscope.get(ZStage)
        z_um = 0.0 if z_stage is None else z_stage.position_um()
        return self._sample.render(
            position.x_um,
            position.y_um,
            z_um,
            shape=self._shape,
            pixel_size_um=self._pixel_size_um,
            camera_rotation_deg=self._rotation_deg,
            mirrored=self._mirrored,
            exposure_ms=self._exposure_ms,
        )

    def exposure_ms(self) -> float:
        """The stored exposure."""
        return self._exposure_ms

    def set_exposure_ms(self, value_ms: float) -> float:
        """Store the exposure and return it; a value that is not usable keeps the old one.

        Raises:
            ValueError: ``value_ms`` is negative or not finite.
        """
        self._exposure_ms = _bounded("exposure_ms", value_ms, 0.0, _MAX_EXPOSURE_MS)
        return self._exposure_ms

    def image_shape(self) -> tuple[int, int]:
        """The frame shape as ``(height, width)``."""
        return self._shape

    def bit_depth(self) -> int:
        """Frames are 16-bit."""
        return 16

    def pixel_size_um(self) -> float:
        """The pixel size the camera was given."""
        return self._pixel_size_um
