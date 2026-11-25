"""Dark objects and rough background in a low-magnification frame: the image half of ``low-mag-survey`` (#636).

At 10x in phase contrast a cell is a dark round blob of 12-15 um with a bright
halo, on a bright background that the illumination vignettes (each frame is
brighter in its centre) and that shot noise makes grainy. The cell model was
trained on 40x frames and misses almost all of them, so these functions find
them by contrast: the frame divided by a wide blur of itself, which removes the
vignetting, then a threshold on the frame's own robust noise. Near a well's wall
the glass carries dried droplets and the meniscus, which look like many small
dark spots: ``tile_texture`` measures that roughness so that a field on it can be
banned.

Pure: arrays in, arrays out, no ``smc.data``, no stage, no I/O (ADR-0004,
algorithms design §5). The thresholds were measured on the 10x scan of
2026-11-24 (``docs/algorithms/low-mag-survey.md``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from skimage import filters, measure, morphology

__all__ = [
    "DarkObjects",
    "find_dark_objects",
    "relative_contrast",
    "tile_texture",
]

#: The robust standard deviation of a normal sample is its median absolute deviation times this.
_MAD_TO_SIGMA = 1.4826

# scikit-image is untyped: its functions are given a type once, here, as
# `smc.plugins.cells.detector` does, so that the rest of the module is checked.
# scipy (which these wrap) has no stubs at all and is not imported directly.
_gaussian: Callable[..., npt.NDArray[np.floating[Any]]] = filters.gaussian
_label: Callable[..., npt.NDArray[np.int_]] = measure.label
_regions: Callable[..., dict[str, npt.NDArray[Any]]] = measure.regionprops_table
_dilate: Callable[..., npt.NDArray[np.bool_]] = morphology.dilation
_diamond: Callable[[int], npt.NDArray[np.uint8]] = morphology.diamond


def _blur(
    pixels: npt.NDArray[np.float32], sigma_px: float, mode: str = "nearest"
) -> npt.NDArray[np.float32]:
    """A Gaussian blur that keeps the values and the float32 type."""
    return _gaussian(pixels, sigma=sigma_px, mode=mode, preserve_range=True).astype(
        np.float32
    )


@dataclass(frozen=True)
class DarkObjects:
    """The dark objects of one frame, one entry per object in each array, in label order.

    ``diameter_px`` is the diameter of the disc of the object's area, and
    ``min_contrast`` its darkest smoothed relative contrast (negative: -0.2 is
    20 % darker than the background).
    """

    row_px: npt.NDArray[np.float64]
    col_px: npt.NDArray[np.float64]
    diameter_px: npt.NDArray[np.float64]
    min_contrast: npt.NDArray[np.float64]


def relative_contrast(
    image: npt.NDArray[Any], *, background_sigma_px: float
) -> npt.NDArray[np.float32]:
    """``image / background - 1``, the background being a wide blur of the image with its dark objects filled in.

    A plain blur is pulled down around a large dark object (a bubble, a clump),
    and the cells next to it would then look less dark than they are; so the
    image is first raised to a first blur wherever it is darker than it, and the
    background is the blur of that. A background below 1 count (the black
    outside a well) is taken as 1, so the division never meets a zero.

    The blur is taken on the frame extended by an **odd reflection** (the edge
    continues its own slope), not by repeating the edge: the vignetting falls
    towards the frame's edges, a flat extension makes the background there too
    bright, and the edges then read as dark objects (a 40 % vignetting gave two
    false objects of 85 um at the edges of an empty frame, and the real 10x
    frames showed the same).

    Raises:
        ValueError: ``image`` is not two-dimensional, or the sigma is not positive.
    """
    if background_sigma_px <= 0.0:
        raise ValueError(
            f"background_sigma_px must be positive, got {background_sigma_px!r}"
        )
    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"a frame is two-dimensional, got shape {pixels.shape}")
    first = _blur_extended(pixels, background_sigma_px)
    background = _blur_extended(np.maximum(pixels, first), background_sigma_px)
    contrast: npt.NDArray[np.float32] = (
        pixels / np.maximum(background, 1.0) - 1.0
    ).astype(np.float32)
    return contrast


def find_dark_objects(
    contrast: npt.NDArray[np.float32],
    *,
    smooth_sigma_px: float,
    k_noise: float,
    min_contrast: float,
    min_area_px: int,
    extent_fraction: float,
    diameter_depth_fraction: float,
) -> tuple[DarkObjects, npt.NDArray[np.bool_]]:
    """The objects darker than the frame's noise and than ``min_contrast``, and the mask of their pixels.

    The contrast is smoothed against the shot noise, and the noise is measured
    on the smoothed frame itself (its median absolute deviation), so that a dim,
    noisy frame needs a darker object than a bright one. A pixel belongs to an
    object when it is ``k_noise`` noise levels dark **and** ``extent_fraction``
    of ``min_contrast`` dark (so the object's extent reaches past its darkest
    core); an object is kept when its darkest point reaches ``min_contrast`` and
    it has ``min_area_px`` pixels.

    The diameter is that of the object's pixels at least
    ``diameter_depth_fraction`` as dark as its own darkest point (at 0.5, its
    width at half depth), not of every pixel past the detection threshold: the
    smoothing spreads an object, and a fixed threshold makes a dark object
    larger than a faint one of the same size (a sharp 12 um disc read 15 um).

    Every number is the caller's: the selector's parameters hold the defaults
    and say which were measured and which were chosen.
    """
    smoothed = _blur(contrast.astype(np.float32), smooth_sigma_px, "reflect")
    noise = float(np.median(np.abs(smoothed - np.median(smoothed)))) * _MAD_TO_SIGMA
    dark = (smoothed < -k_noise * noise) & (smoothed < -min_contrast * extent_fraction)
    labels = _label(dark)
    table = _regions(
        labels,
        intensity_image=smoothed,
        properties=("label", "centroid", "area", "intensity_min"),
    )
    label_ids = np.asarray(table["label"], dtype=np.int64)
    area = np.asarray(table["area"], dtype=np.float64)
    darkest = np.asarray(table["intensity_min"], dtype=np.float64)
    keep = (area >= min_area_px) & (darkest <= -min_contrast)
    # A fraction of each object's own darkest contrast, spread over its pixels.
    depth = np.zeros(int(labels.max()) + 1, dtype=np.float64)
    depth[label_ids] = darkest * diameter_depth_fraction
    core = (labels > 0) & (smoothed <= depth[labels])
    core_area = np.asarray(
        np.bincount(labels.ravel(), weights=core.ravel())[label_ids], dtype=np.float64
    )
    kept_labels = label_ids[keep]
    objects = DarkObjects(
        row_px=np.asarray(table["centroid-0"], dtype=np.float64)[keep],
        col_px=np.asarray(table["centroid-1"], dtype=np.float64)[keep],
        diameter_px=np.sqrt(4.0 * core_area[keep] / np.pi),
        min_contrast=darkest[keep],
    )
    mask: npt.NDArray[np.bool_] = np.isin(labels, kept_labels) & (labels > 0)
    return objects, mask


def tile_texture(
    contrast: npt.NDArray[np.float32],
    object_mask: npt.NDArray[np.bool_],
    *,
    tile_px: int,
    band_sigmas_px: tuple[float, float],
    noise_sigma_px: float,
    dilate_px: int,
    min_kept_fraction: float,
) -> npt.NDArray[np.float64]:
    """The roughness of each ``tile_px`` square of the frame, objects left out: ``(rows // tile_px, cols // tile_px)``.

    The roughness is the spread of the contrast's structure at the scale of
    dried droplets (a difference of two blurs, ``band_sigmas_px``) divided by
    the spread of the tile's own pixel noise (the contrast less a blur of
    ``noise_sigma_px``). The plain spread of the contrast also grows where a frame is dim, as
    the shot noise is relatively larger there, and on the 10x scan of 2026-11-24
    it called 44 to 83 % of the clean glass 1.0 to 1.5 mm from the wall rough;
    the ratio does not depend on the brightness. Both spreads are medians of
    absolute deviations, robust to the cells that the mask leaves.

    The objects' pixels, grown by ``dilate_px`` to take their halos, are left
    out; a tile with less than ``min_kept_fraction`` of its pixels left is
    ``nan``. Pixels beyond the last whole tile are not read.
    """
    if tile_px < 1:
        raise ValueError(f"tile_px must be at least 1, got {tile_px!r}")
    low, high = band_sigmas_px
    image = contrast.astype(np.float32)
    band = _blur(image, low, "reflect") - _blur(image, high, "reflect")
    fine = image - _blur(image, noise_sigma_px, "reflect")
    left_out = (
        _dilate(object_mask, _diamond(dilate_px)) if dilate_px > 0 else object_mask
    )
    rows, cols = image.shape[0] // tile_px, image.shape[1] // tile_px
    texture = np.full((rows, cols), np.nan, dtype=np.float64)
    least = math.ceil(tile_px * tile_px * min_kept_fraction)
    for i in range(rows):
        for j in range(cols):
            window = (
                slice(i * tile_px, (i + 1) * tile_px),
                slice(j * tile_px, (j + 1) * tile_px),
            )
            kept = ~left_out[window]
            if int(kept.sum()) < least:
                continue
            structure = _spread(band[window][kept])
            noise = _spread(fine[window][kept])
            if noise > 0.0:
                texture[i, j] = structure / noise
    return texture


def _blur_extended(
    pixels: npt.NDArray[np.float32], sigma_px: float
) -> npt.NDArray[np.float32]:
    """A Gaussian blur of ``pixels`` extended by an odd reflection of four sigmas, cut back to its shape.

    The extension is at most the frame's own size on each side: a sigma larger
    than the frame would otherwise grow the array with it (FM-74).
    """
    pad = min(int(np.ceil(4.0 * sigma_px)), *pixels.shape)
    extended = np.pad(pixels, pad, mode="reflect", reflect_type="odd")
    blurred = _blur(extended, sigma_px)
    out: npt.NDArray[np.float32] = blurred[
        pad : pad + pixels.shape[0], pad : pad + pixels.shape[1]
    ]
    return out


def _spread(values: npt.NDArray[np.float32]) -> float:
    """The robust standard deviation of ``values``: their median absolute deviation, scaled."""
    return float(np.median(np.abs(values - np.median(values)))) * _MAD_TO_SIGMA
