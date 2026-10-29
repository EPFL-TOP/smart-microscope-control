"""The demo stand whose camera images a synthetic plate (m1 §8, "The demo-sample stand", #260).

The synthetic sample of ``smc.testing.synthetic`` is reached only through a
fixture, so the plate tools could not be seen working on images outside
``pytest``. A profile with a ``[sample]`` table (``profiles/demo-sample.toml``) is
the demo stand whose camera renders that sample at the stage's position:
``Microscope.from_core`` calls ``install_sample`` for it, so ``smc run <tool> -p
demo-sample``, ``smc ui plate -p demo-sample`` and a notebook all get the same
stand, and nothing else knows which stand it is on.

This module is imported **inside** ``from_core``, for a profile that has the
table, and never by a plugin: a real stand never loads it, and a test proves that
nothing in ``smc.plugins`` imports ``smc.testing``.

What it is not: the frames are rendered where the stage is, not imaged. The
sample is one flat plane at z = 0 (Z only blurs), there is no optics, and its
texture aliases above ``TEXTURE_TILE_UM`` per pixel. ``sample_lines`` says so to
the operator, and the run document records the profile's name on every stage.
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import useq

from smc.hardware.capabilities import Camera, ObjectiveTurret, XYStage, ZStage
from smc.hardware.errors import HardwareError, ProfileError
from smc.hardware.profile import SampleSection
from smc.hardware.roles import Role
from smc.testing.synthetic import TEXTURE_TILE_UM, Blob, PlateSample

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "CELL_INTENSITY",
    "CELL_REACH",
    "DOUBLET_OFFSET",
    "StandSampleCamera",
    "centred_a1_um",
    "install_sample",
    "sample_lines",
    "sample_of",
    "scatter_cells",
]

log = logging.getLogger("smc.testing.sample_stand")

#: Counts a cell adds above the well around it, at ``REFERENCE_EXPOSURE_MS``.
CELL_INTENSITY = 3000.0
#: A cell's centre lies within this fraction of the well's half-size, so that the
#: cell and a doublet's two spots stay inside the well.
CELL_REACH = 0.85
#: A doublet's two spots lie this many radii either side of its centre.
DOUBLET_OFFSET = 0.8
#: The only library whose camera this module replaces the pixels of.
_DEMO_CAMERA_LIBRARY = "DemoCamera"


def centred_a1_um(plate: str) -> tuple[float, float]:
    """Where A1's centre lies when the plate is centred on the stage origin, in µm.

    The demo stage starts at (0, 0), so a plate centred there puts a frame at
    the origin over plastic and the first move onto a well in reach. 96 wells
    give (-49 500, 31 500) and 384 wells (-51 750, 33 750), as measured.
    ``useq`` keeps the pitch in mm, hence the factor.
    """
    wells = useq.WellPlate.from_str(plate)
    pitch_x_um, pitch_y_um = (v * 1000.0 for v in wells.well_spacing)
    return (
        -(wells.columns - 1) / 2.0 * pitch_x_um,
        (wells.rows - 1) / 2.0 * pitch_y_um,
    )


def _a1_of(section: SampleSection) -> tuple[float, float]:
    if section.a1_center_xy_um is not None:
        return section.a1_center_xy_um
    return centred_a1_um(section.plate)


def scatter_cells(section: SampleSection) -> tuple[Blob, ...]:
    """The cells of the sample as bright round spots, one or two per cell, inside their wells.

    Deterministic in ``section.seed`` (``numpy.random.default_rng``). Wells are
    visited row by row (A1, A2, ..., B1, ...). A well in column ``c`` of ``C``
    holds ``round(cells_per_well * (0.5 + c / (C - 1)))`` cells (``cells_per_well``
    when there is one column): from half the mean in the first column to one and
    a half times it in the last, so density and route have something to tell
    apart while the mean over the columns stays ``cells_per_well``.

    A cell's centre is uniform over the disc of radius ``CELL_REACH`` times the
    well's radius (a circular well) or over the square of half-sides ``CELL_REACH``
    times the well's half-size (a square one), in the plate's own axes, turned
    with the plate. It is a doublet with probability ``doublet_fraction``: two
    spots ``DOUBLET_OFFSET * cell_radius_um`` either side of the centre along a
    uniform angle, else one spot.

    The numbers are drawn in a fixed order per cell (the radius, the angle, the
    doublet draw, then the doublet's angle when it is one), so the same seed
    gives the same cells; the tests check properties, not positions, which a
    numpy release that changed the stream would move.
    """
    rng = np.random.default_rng(section.seed)
    plan = useq.WellPlatePlan(
        plate=section.plate,  # type: ignore[arg-type]
        a1_center_xy=_a1_of(section),
        rotation=section.rotation_deg,
    )
    wells = plan.plate
    # `well_size` is in mm, as the plate's pitch.
    half_x_um, half_y_um = (v * 500.0 for v in wells.well_size)
    angle = math.radians(section.rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    radius_um = section.cell_radius_um
    offset_um = DOUBLET_OFFSET * radius_um
    blobs: list[Blob] = []
    for well in plan.all_well_positions:
        column = well.plate_col or 0
        # One column has no first and last to spread between: it takes the mean.
        share = 0.5 + column / (wells.columns - 1) if wells.columns > 1 else 1.0
        for _ in range(round(section.cells_per_well * share)):
            first, second = rng.random(), rng.random()
            if wells.circular_wells:
                reach = CELL_REACH * half_x_um * math.sqrt(first)
                around = 2.0 * math.pi * second
                local_x, local_y = reach * math.cos(around), reach * math.sin(around)
            else:
                local_x = (2.0 * first - 1.0) * CELL_REACH * half_x_um
                local_y = (2.0 * second - 1.0) * CELL_REACH * half_y_um
            x_um = (well.x or 0.0) + cos * local_x - sin * local_y
            y_um = (well.y or 0.0) + sin * local_x + cos * local_y
            spots: tuple[tuple[float, float], ...] = ((x_um, y_um),)
            if rng.random() < section.doublet_fraction:
                pair = 2.0 * math.pi * rng.random()
                dx, dy = offset_um * math.cos(pair), offset_um * math.sin(pair)
                spots = ((x_um - dx, y_um - dy), (x_um + dx, y_um + dy))
            blobs += [Blob(x, y, radius_um, CELL_INTENSITY) for x, y in spots]
    return tuple(blobs)


def sample_of(section: SampleSection) -> PlateSample:
    """The ``PlateSample`` a ``[sample]`` table describes, its cells scattered."""
    return PlateSample(
        section.plate,
        _a1_of(section),
        section.rotation_deg,
        blobs=scatter_cells(section),
        seed=section.seed,
    )


def sample_lines(section: SampleSection) -> tuple[str, ...]:
    """What ``smc doctor`` says about the sample: what it is and what it is not.

    Three lines, printed with ``_say`` and never as markup (FM-36). ASCII
    apart from ``µ`` and ``°``, which cp1252 has: a redirected doctor on Windows
    prints them (FM-20).
    """
    x_um, y_um = _a1_of(section)
    return (
        f"Sample: a synthetic {section.plate} plate, A1 at ({x_um:.1f}, {y_um:.1f}) µm, "
        f"turned {section.rotation_deg:g}°; {section.cells_per_well} cells per well on "
        f"average, {section.doublet_fraction * 100:g} % of them doublets "
        f"(seed {section.seed})",
        "  frames are rendered where the stage is, not imaged: one flat plane at "
        "z = 0 (Z only blurs), no optics",
        f"  pixel sizes up to {TEXTURE_TILE_UM:g} µm (the texture aliases above); the "
        "cells are bright round spots: run cells with detector=blobs",
    )


class StandSampleCamera:
    """The stand's own camera with its pixels replaced by a rendering of the sample.

    ``snap()`` calls the stand's camera first (``inner``), so the halt, the
    motion guard and the lock of the ``Executor`` apply as on any stand: a frame
    during a move is refused, and a dry run snaps. It then reads that camera's
    pixel size and exposure, the stage's XY and Z (0.0 without a Z stage), and
    renders the sample at the shape of the frame it got, so the objective, the
    binning and the exposure the stand has are the ones the frame shows. Every
    other method is the stand camera's.

    The render runs under a lock of its own, because the sample's random
    generator is not thread-safe; that lock is never held across ``inner.snap()``.
    A move by another thread between the snap and the position read is rendered
    at the new position: the page and ``smc run`` run one tool at a time, and
    that is the limit of an overlay over a demo device.

    Every refusal is a ``HardwareError``, never a ``ValueError``, because the
    plugins' loops catch ``HardwareError`` where the design says they may.

    Args:
        microscope: The stand whose stages say where the camera is looking.
            Held weakly: the facade holds this camera as its ``Camera``, and a
            strong reference back would make a cycle through the core, freed
            only by the cyclic collector (FM-92).
        sample: What it looks at.
        inner: The stand's own camera, which this one wraps.
        camera_rotation_deg: Rotation of the camera about the optical axis.
        mirrored: Whether the image is flipped left-right.
    """

    def __init__(
        self,
        microscope: Microscope,
        sample: PlateSample,
        inner: Camera,
        *,
        camera_rotation_deg: float = 0.0,
        mirrored: bool = False,
    ) -> None:
        self._microscope = weakref.ref(microscope)
        self._sample = sample
        self._inner = inner
        self._rotation_deg = camera_rotation_deg
        self._mirrored = mirrored
        self._render_lock = threading.Lock()
        self._warned_about_the_texture = False

    def snap(self) -> npt.NDArray[np.uint16]:
        """One frame of the sample at the stage's current position.

        Raises:
            HardwareError: The stand is gone; the frame is not 2-D; the camera's
                pixel size is not a finite, positive number; or the sample
                cannot be rendered (its message is kept). A halt or a move in
                progress is the inner camera's own refusal and passes through.
        """
        microscope = self._microscope()
        if microscope is None:
            raise HardwareError(
                "StandSampleCamera belongs to a Microscope that no longer exists; "
                "keep the Microscope alive for as long as the camera is in use"
            )
        frame = self._inner.snap()
        shape = np.shape(frame)
        if len(shape) != 2:
            raise HardwareError(
                f"the stand's camera returned a frame of shape {shape}, not a 2-D "
                "one: the synthetic plate is rendered at a 2-D frame's shape"
            )
        pixel_size_um = self._inner.pixel_size_um()
        # MMCore passes an invalid magnification through (FM-104): infinity is
        # above 0, so the value is checked here, not taken as it comes.
        if not (math.isfinite(pixel_size_um) and pixel_size_um > 0.0):
            raise HardwareError(
                f"the stand's camera reports a pixel size of {pixel_size_um!r} µm "
                f"for objective {_objective_name(microscope)!r}, and the synthetic "
                "plate is never rendered at a guessed scale: set [camera] "
                "pixel_size_um for this objective in the profile, or select one "
                "that has it"
            )
        position = microscope.require(XYStage).position_um()
        z_stage = microscope.get(ZStage)
        z_um = 0.0 if z_stage is None else z_stage.position_um()
        exposure_ms = self._inner.exposure_ms()
        self._warn_if_the_texture_aliases(pixel_size_um)
        with self._render_lock:
            try:
                return self._sample.render(
                    position.x_um,
                    position.y_um,
                    z_um,
                    shape=(int(shape[0]), int(shape[1])),
                    pixel_size_um=pixel_size_um,
                    camera_rotation_deg=self._rotation_deg,
                    mirrored=self._mirrored,
                    exposure_ms=exposure_ms,
                )
            except ValueError as exc:
                raise HardwareError(
                    f"the synthetic plate cannot be rendered for this frame: {exc}"
                ) from exc

    def _warn_if_the_texture_aliases(self, pixel_size_um: float) -> None:
        # Once per camera: a scan snaps thousands of frames.
        if pixel_size_um > TEXTURE_TILE_UM and not self._warned_about_the_texture:
            self._warned_about_the_texture = True
            log.warning(
                "the pixel size %g µm is above the %g µm of the sample's texture: "
                "the texture is aliased noise, and only whole-pixel shifts correlate",
                pixel_size_um,
                TEXTURE_TILE_UM,
            )

    def exposure_ms(self) -> float:
        """The stand camera's exposure."""
        return self._inner.exposure_ms()

    def set_exposure_ms(self, value_ms: float) -> float:
        """Set the stand camera's exposure; the frames scale with it."""
        return self._inner.set_exposure_ms(value_ms)

    def image_shape(self) -> tuple[int, int]:
        """The stand camera's frame shape, which is the shape of every frame."""
        return self._inner.image_shape()

    def bit_depth(self) -> int:
        """The stand camera's depth."""
        return self._inner.bit_depth()

    def pixel_size_um(self) -> float:
        """The stand camera's pixel size, for the objective in place: ``0.0`` is unknown."""
        return self._inner.pixel_size_um()


def _objective_name(microscope: Microscope) -> str:
    """The current objective's name for a message, or ``""`` when it cannot be read."""
    turret = microscope.get(ObjectiveTurret)
    if turret is None:
        return ""
    try:
        return turret.current()
    except HardwareError:
        return ""


def install_sample(microscope: Microscope, section: SampleSection) -> StandSampleCamera:
    """Put the sample's camera on a demo stand, in place of the demo camera's pixels.

    The camera it wraps must be the demo's: ``[sample]`` over a real camera
    would show synthetic frames as a working stand. The profile's ``config = ""``
    already says so, and this checks what was loaded.

    Raises:
        ProfileError: No device fills the camera role, or the camera is not from
            the ``DemoCamera`` library.
    """
    where = f"{microscope.profile.source}: " if microscope.profile.source else ""
    label = microscope.roles.get(Role.camera)
    if label is None:
        raise ProfileError(
            f"{where}[sample] needs a camera to replace the pixels of, and no "
            "device fills the camera role on this stand"
        )
    library = next((d.library for d in microscope.devices if d.label == label), "")
    if library != _DEMO_CAMERA_LIBRARY:
        raise ProfileError(
            f"{where}[sample] replaces the pixels of the demo camera, and camera "
            f"{label!r} comes from the library {library!r}, not "
            f"{_DEMO_CAMERA_LIBRARY!r}: use the demo configuration "
            '(micromanager.config = ""), or remove [sample]'
        )
    camera = StandSampleCamera(
        microscope,
        sample_of(section),
        microscope.require(Camera),
        camera_rotation_deg=section.camera_rotation_deg,
        mirrored=section.mirrored,
    )
    microscope.override(Camera, camera)
    log.info(
        "%s: frames are rendered from a synthetic %s plate, not imaged",
        microscope.profile.microscope.name,
        section.plate,
    )
    return camera
