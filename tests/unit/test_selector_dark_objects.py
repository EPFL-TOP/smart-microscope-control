"""The pure image functions of ``low-mag-survey`` on synthetic frames (#636).

At 10x a cell is a dark round blob of 12-15 um on a bright, vignetted, noisy
background; these frames are built to that description with a seeded noise,
and none is square (FM-73).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from smc.selection.dark_objects import (
    find_dark_objects,
    relative_contrast,
    tile_texture,
)

PIXEL_UM = 1.124
LEVEL = 20000.0
BACKGROUND_SIGMA_PX = 30.0


def _frame(
    shape: tuple[int, int] = (300, 420),
    *,
    vignetting: float = 0.0,
    noise: float = 0.01,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """A bright frame, darker towards its corners by ``vignetting``, with relative Gaussian noise."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    r = np.hypot((rows - shape[0] / 2) / shape[0], (cols - shape[1] / 2) / shape[1])
    level = LEVEL * (1.0 - vignetting * (r / r.max()) ** 2)
    rng = np.random.default_rng(seed)
    return level * (1.0 + noise * rng.standard_normal(shape))


def _disc(
    image: npt.NDArray[np.float64],
    row: float,
    col: float,
    diameter_um: float,
    depth: float,
) -> None:
    """Darken a disc of ``diameter_um`` centred at ``(row, col)`` by the fraction ``depth``."""
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    inside = np.hypot(rows - row, cols - col) <= diameter_um / PIXEL_UM / 2
    image[inside] *= 1.0 - depth


def _objects(image: npt.NDArray[np.float64], min_contrast: float = 0.06):  # type: ignore[no-untyped-def]
    contrast = relative_contrast(image, background_sigma_px=BACKGROUND_SIGMA_PX)
    return find_dark_objects(
        contrast,
        smooth_sigma_px=1.5,
        k_noise=5.0,
        min_contrast=min_contrast,
        min_area_px=12,
        extent_fraction=0.5,
        diameter_depth_fraction=0.5,
    )


def test_a_dark_disc_of_12_um_is_found_at_its_centre_with_its_diameter() -> None:
    image = _frame()
    _disc(image, 140.0, 251.0, 12.0, 0.2)
    objects, mask = _objects(image)
    assert len(objects.row_px) == 1
    assert abs(objects.row_px[0] - 140.0) < 0.5
    assert abs(objects.col_px[0] - 251.0) < 0.5
    assert abs(objects.diameter_px[0] * PIXEL_UM - 12.0) < 0.15 * 12.0
    assert objects.min_contrast[0] < -0.1
    assert mask[140, 251]
    assert not mask[20, 20]


def test_a_vignetted_flat_frame_has_no_object() -> None:
    image = _frame(vignetting=0.4)
    objects, mask = _objects(image)
    assert len(objects.row_px) == 0
    assert not mask.any()


def test_a_disc_fainter_than_min_contrast_is_not_an_object() -> None:
    image = _frame(noise=0.002)
    _disc(image, 150.0, 210.0, 14.0, 0.04)
    faint, _ = _objects(image, min_contrast=0.06)
    assert len(faint.row_px) == 0
    found, _ = _objects(image, min_contrast=0.03)
    assert len(found.row_px) == 1


def test_a_large_dark_disc_does_not_drag_the_background_down_around_it() -> None:
    # A clump of 150 um, half dark, 30 um from a cell: a plain blur of the frame
    # takes the clump into the cell's background, and the cell then reads paler.
    alone = _frame(noise=0.002)
    _disc(alone, 150.0, 260.0, 12.0, 0.2)
    beside = _frame(noise=0.002)
    _disc(beside, 150.0, 150.0, 150.0, 0.5)
    _disc(beside, 150.0, 260.0, 12.0, 0.2)
    cell_alone, _ = _objects(alone)
    found, _ = _objects(beside)
    diameters_um = found.diameter_px * PIXEL_UM
    cell = int(np.argmin(np.abs(diameters_um - 12.0)))
    assert len(found.row_px) == 2
    assert abs(diameters_um[cell] - 12.0) < 0.15 * 12.0
    assert abs(found.col_px[cell] - 260.0) < 0.5
    assert found.min_contrast[cell] < 0.9 * cell_alone.min_contrast[0]


def test_rows_and_columns_are_not_swapped_on_a_frame_that_is_not_square() -> None:
    image = _frame((200, 500))
    _disc(image, 40.0, 430.0, 13.0, 0.2)
    objects, mask = _objects(image)
    assert len(objects.row_px) == 1
    assert abs(objects.row_px[0] - 40.0) < 0.5
    assert abs(objects.col_px[0] - 430.0) < 0.5
    assert mask.shape == (200, 500)


def test_texture_is_higher_on_a_rough_tile_than_on_a_smooth_one_and_nan_where_masked() -> (
    None
):
    image = _frame((256, 320), noise=0.01, seed=3)
    rng = np.random.default_rng(4)
    # The rough tile (rows 0-63, cols 64-127): dried droplets, dark spots of a few um.
    for _ in range(60):
        _disc(image, rng.uniform(4, 60), rng.uniform(68, 124), rng.uniform(4, 8), 0.05)
    contrast = relative_contrast(image, background_sigma_px=BACKGROUND_SIGMA_PX)
    masked = np.zeros(contrast.shape, dtype=bool)
    masked[192:256, 0:64] = True  # one tile wholly under objects
    texture = tile_texture(
        contrast,
        masked,
        tile_px=64,
        band_sigmas_px=(1.5, 6.0),
        noise_sigma_px=1.0,
        dilate_px=0,
        min_kept_fraction=0.25,
    )
    assert texture.shape == (4, 5)
    assert np.isnan(texture[3, 0])
    smooth = texture[1:3, 1:4]
    assert texture[0, 1] > 2.0 * float(np.nanmax(smooth))


def test_texture_of_pure_noise_does_not_depend_on_the_brightness() -> None:
    bright = _frame((256, 256), noise=0.005, seed=5)
    dark = _frame(
        (256, 256), noise=0.05, seed=5
    )  # ten times noisier: a dim frame near the wall
    masked = np.zeros((256, 256), dtype=bool)
    textures = [
        tile_texture(
            relative_contrast(image, background_sigma_px=BACKGROUND_SIGMA_PX),
            masked,
            tile_px=64,
            band_sigmas_px=(1.5, 6.0),
            noise_sigma_px=1.0,
            dilate_px=0,
            min_kept_fraction=0.25,
        )
        for image in (bright, dark)
    ]
    low, high = sorted(float(np.median(t)) for t in textures)
    # Ten times the noise, the same texture: within 20 %.
    assert high < 1.2 * low
