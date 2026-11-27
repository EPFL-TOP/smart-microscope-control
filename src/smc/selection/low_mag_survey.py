"""``low-mag-survey``: the 40x fields worth scanning, chosen from a fast low-magnification scan's own frames (#636).

A whole well at 40x is about 1500 fields and a quarter of an hour; at 10x it is
109 fields and under a minute. The 10x scan says where the 40x is worth going:
not near the wall, where the meniscus and dried droplets cover the glass; not on
a bubble or a clump; not where the 10x scan did not look; and not where there is
nothing (or too much). This selector reads the 10x frames themselves (the cell
model was trained on 40x and misses almost every cell at 10x, so
``data.cells`` is not read and ``route``'s ``labels`` do not matter), finds the
dark objects and the rough background of every frame (``dark_objects``), places
them on the stage, and returns the 40x fields of each well that pass the bans,
the most cells first: placed freely and greedily by default (each the best that
does not overlap the ones before), or on a touching lattice.

What was measured to set the defaults, on the 10x scan of 2026-11-24, is on the
page ``docs/algorithms/low-mag-survey.md``. The choices are in **the 10x scan's
stage frame**, like every selector's: the shift between the 10x and the 40x of
that session was measured on the data, and it is the stand's to correct, not
this selector's (FM-207).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from smc.data import DatasetError
from smc.plugins.geometry import Matrix2, footprint, stage_of_pixel
from smc.plugins.route.algorithm import _inverse_of, _pixel_offsets
from smc.plugins.slots import WellGeometry
from smc.selection.api import Choice, SelectionError, SelectionInput, SelectorParams
from smc.selection.dark_objects import (
    find_dark_objects,
    relative_contrast,
    tile_texture,
)

__all__ = [
    "Candidate",
    "LowMagSurvey",
    "LowMagSurveyParams",
    "WellSurvey",
    "candidates_of",
    "frame_scale_um_per_px",
    "rank_candidates",
    "survey_well",
]

#: The session every "measured" default was measured on (``docs/algorithms/low-mag-survey.md``).
MEASURED_ON = (
    "the 10x scan of 2026-11-24 (B3, B4, C3, C4) against the 40x tilings of C3 and C4"
)


def _chosen(default: Any, description: str, **constraints: Any) -> Any:
    """A parameter whose default was **chosen**, not measured: said in its description and its schema."""
    return Field(
        default,
        description=f"Chosen, not measured: {description}",
        json_schema_extra={"basis": "chosen"},
        **constraints,
    )


def _measured(default: Any, description: str, **constraints: Any) -> Any:
    """A parameter whose default was **measured** on ``MEASURED_ON``: said in its description and its schema."""
    return Field(
        default,
        description=f"Measured on {MEASURED_ON}: {description}",
        json_schema_extra={"basis": "measured"},
        **constraints,
    )


class LowMagSurveyParams(SelectorParams):
    """Every number of the survey and of the choice, each saying whether its default was measured or chosen.

    Nothing the algorithm decides with is a constant of the module: a default
    that was chosen (a reasonable value, not optimised on data) is labelled so in
    its description and in its JSON schema (``basis``), so that ``route
    --show-params`` and the page say which numbers rest on a measurement. A test
    holds every field to it.
    """

    # --- finding dark objects in a frame (``dark_objects``) -----------------------
    background_sigma_um: float = _chosen(
        34.0,
        "the blur that estimates a frame's background (30 px at 10x): well above a "
        "cell (~10 um), well below the vignetting's slope (hundreds of um).",
        gt=0.0,
        le=500.0,
    )
    smooth_sigma_um: float = _chosen(
        1.7,
        "the blur against the shot noise (1.5 px at 10x): small next to a cell.",
        gt=0.0,
        le=20.0,
    )
    k_noise: float = _chosen(
        5.0,
        "how many noise levels (the frame's robust spread) dark an object's pixel is.",
        gt=0.0,
    )
    min_contrast: float = _chosen(
        0.06,
        "how much darker than its background an object's darkest point is; "
        "checked only through the per-field match with the 40x counts.",
        gt=0.0,
        lt=1.0,
    )
    extent_fraction: float = _chosen(
        0.5,
        "an object's pixels are at least this fraction of min_contrast dark, so "
        "its extent reaches past its darkest core.",
        gt=0.0,
        le=1.0,
    )
    diameter_depth_fraction: float = _chosen(
        0.5,
        "the diameter is that of the pixels at least this fraction of the object's "
        "own darkest contrast (0.5: the width at half depth).",
        gt=0.0,
        lt=1.0,
    )
    min_area_um2: float = _chosen(
        15.0, "the smallest object, in um2 (12 px at 10x): below is noise.", gt=0.0
    )
    merge_radius_um: float = _chosen(
        5.0,
        "two sightings of one object by two overlapping frames are this close or "
        "closer; two touching cells are ~10 um apart, and the cells stage merges at 5.",
        ge=0.0,
        le=50.0,
    )
    # --- classes by size ----------------------------------------------------------
    cell_min_diameter_um: float = _measured(
        8.0,
        "the smallest object counted as a cell; 6 um read 1.18 of the 40x count in "
        "C3, 8 um read 0.96 (and 0.89 in C4).",
        gt=0.0,
    )
    cell_max_diameter_um: float = _chosen(
        25.0, "the largest object counted as a cell (a doublet still counts).", gt=0.0
    )
    large_min_diameter_um: float = _chosen(
        30.0,
        "an object this large or larger is a clump, a bubble or debris.",
        gt=0.0,
    )
    # --- texture -----------------------------------------------------------------
    tile_um: float = _chosen(
        72.0,
        "the side of the squares whose texture is read (64 px at 10x; ~30 a 40x field).",
        gt=0.0,
        le=1000.0,
    )
    texture_low_um: float = _chosen(
        1.7,
        "the finer blur of the texture's band-pass: the scale of dried droplets.",
        gt=0.0,
    )
    texture_high_um: float = _chosen(
        6.7,
        "the coarser blur of the texture's band-pass: the scale of dried droplets.",
        gt=0.0,
    )
    texture_noise_sigma_px: float = _chosen(
        1.0,
        "the blur whose residual is a tile's pixel noise, which the band-pass is divided by.",
        gt=0.0,
        le=5.0,
    )
    halo_um: float = _chosen(
        3.4,
        "how far an object's halo reaches past its dark pixels (3 px at 10x); left "
        "out of the texture.",
        ge=0.0,
        le=50.0,
    )
    min_tile_kept_fraction: float = _chosen(
        0.25,
        "a tile with less than this share of its pixels left after the objects has "
        "no texture (nan).",
        gt=0.0,
        le=1.0,
    )
    max_texture: float | None = _measured(
        0.32,
        "a tile rougher than this is rough: in each well >= 90 % of the tiles within "
        "600 um of the wall and <= 2 % beyond 1.5 mm. None: no texture ban.",
        gt=0.0,
    )
    # --- the bans on a field ------------------------------------------------------
    max_rough_fraction: float = _chosen(
        0.1,
        "a field whose share of rough tiles is above this is banned; banning on one "
        "rough tile (0) emptied 22 % of C4's clean fields, 0.1 bans 2 % of them.",
        ge=0.0,
        lt=1.0,
    )
    min_cells: int = _chosen(
        1, "a field must hold at least this many cells: not an empty field.", ge=0
    )
    max_density_per_mm2: float | None = _chosen(
        None,
        "a field denser than this is banned (overgrown). None: no bound, as no "
        "overgrown well was seen.",
        gt=0.0,
    )
    max_large: int = _chosen(
        0, "a field may hold at most this many large objects.", ge=0
    )
    min_coverage: float = _chosen(
        0.9,
        "the share of a field that a frame that was read covers; tolerates a seam, "
        "not a missing frame.",
        gt=0.0,
        le=1.0,
    )
    coverage_probes_per_side: int = _chosen(
        8,
        "the coverage is read on this many by this many points of a field.",
        ge=2,
        le=64,
    )
    coverage_reach_tiles: float = _chosen(
        0.75,
        "a probe is covered by a tile whose centre is within this many tile sides "
        "(between the half side 0.5 and the half diagonal 0.71, and a little more).",
        gt=0.0,
        le=2.0,
    )
    # --- the lattice --------------------------------------------------------------
    lattice_offset_x_um: float = _chosen(
        0.0,
        "the lattice of fields is laid from the well's centre moved by this, in x.",
        ge=-10000.0,
        le=10000.0,
    )
    lattice_offset_y_um: float = _chosen(
        0.0,
        "the lattice of fields is laid from the well's centre moved by this, in y.",
        ge=-10000.0,
        le=10000.0,
    )
    # --- the placement ------------------------------------------------------------
    placement: Literal["greedy", "lattice"] = _measured(
        "greedy",
        "how the fields are placed. 'lattice': the touching lattice, each cell in one "
        "field. 'greedy': candidates on a lattice finer by greedy_subdivisions, and "
        "the best field that overlaps none taken before, again and again. Greedy put "
        "16 to 26 % more 40x cells in the first 100 to 200 fields of C3 and C4, 4 % "
        "more in C4's first 500, and needed a quarter to a third fewer fields for 90 "
        "% of the lattice's cells in B3, B4 and C3; it packs a dense well worse "
        "without a cap (C4: 486 fields where the lattice keeps 823).",
    )
    greedy_subdivisions: int = _chosen(
        4,
        "greedy placement's candidates step by a field's extent divided by this "
        "(4: a quarter field, 16 shifted lattices).",
        ge=1,
        le=8,
    )

    @model_validator(mode="after")
    def _bands_in_order(self) -> LowMagSurveyParams:
        if not self.cell_min_diameter_um < self.cell_max_diameter_um:
            raise ValueError(
                f"cell_min_diameter_um ({self.cell_min_diameter_um:g}) must be below "
                f"cell_max_diameter_um ({self.cell_max_diameter_um:g})"
            )
        if not self.cell_max_diameter_um <= self.large_min_diameter_um:
            raise ValueError(
                f"large_min_diameter_um ({self.large_min_diameter_um:g}) must not be "
                f"below cell_max_diameter_um ({self.cell_max_diameter_um:g})"
            )
        if not self.texture_low_um < self.texture_high_um:
            raise ValueError(
                f"texture_low_um ({self.texture_low_um:g}) must be below "
                f"texture_high_um ({self.texture_high_um:g})"
            )
        return self


@dataclass(frozen=True)
class WellSurvey:
    """What a well's low-magnification frames show, on the stage: its dark objects and its tiles.

    Each object and each tile is counted once (by the frame whose centre is
    nearest) and only inside the well. A tile's ``texture`` is ``nan`` when its
    objects covered most of it: it was read, its roughness is unknown.
    """

    well: str
    x_um: npt.NDArray[np.float64]
    y_um: npt.NDArray[np.float64]
    diameter_um: npt.NDArray[np.float64]
    min_contrast: npt.NDArray[np.float64]
    tile_x_um: npt.NDArray[np.float64]
    tile_y_um: npt.NDArray[np.float64]
    tile_texture: npt.NDArray[np.float64]
    tile_area_um2: float


@dataclass(frozen=True)
class Candidate:
    """A candidate 40x field, with what the survey found in it."""

    x_um: float
    y_um: float
    n_cells: int
    n_large: int
    rough_fraction: float
    coverage: float
    density_per_mm2: float
    wall_distance_um: float

    def choice(self, well: str) -> Choice:
        """This field as the contract's ``Choice``: the score is its cells, the measures its numbers."""
        return Choice(
            well=well,
            x_um=self.x_um,
            y_um=self.y_um,
            score=float(self.n_cells),
            measures={
                "n_cells": float(self.n_cells),
                "density_per_mm2": self.density_per_mm2,
                "n_large": float(self.n_large),
                "rough_fraction": self.rough_fraction,
                "coverage": self.coverage,
                "wall_distance_um": self.wall_distance_um,
            },
        )


class LowMagSurvey:
    """The 40x fields worth scanning, from a low-magnification scan's own frames."""

    name = "low-mag-survey"
    summary = (
        "From a fast low-magnification scan's frames: the fields away from the "
        "wall's rough band, bubbles and empty glass, the most cells first."
    )
    Params = LowMagSurveyParams

    def select(self, data: SelectionInput, params: LowMagSurveyParams) -> list[Choice]:
        """Survey each well of the scan, in its order, and keep its best ``fields_per_well`` fields.

        Raises:
            SelectionError: The scan's camera cannot place a pixel on the stage,
                a well is not circular, or a frame cannot be read.
        """
        chosen: list[Choice] = []
        for well in data.scan.wells:
            survey = survey_well(data, well, params)
            candidates = candidates_of(data, well, survey, params)
            chosen.extend(
                c.choice(well) for c in rank_candidates(candidates, params, data)
            )
        return chosen


def frame_scale_um_per_px(matrix: Matrix2) -> float:
    """The side of a pixel in um, ``sqrt(|det|)`` of the matrix that placed it on the stage.

    The scan's matrix, not its recorded ``pixel_size_um``: the matrix is what
    turns a pixel into a stage position, and the two can disagree in a record
    (the selection seed records 5.0 whatever matrix it is given).

    Raises:
        SelectionError: The scale is not finite and positive.
    """
    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    if not (math.isfinite(determinant) and determinant != 0.0):
        raise SelectionError(
            f"the scan's camera matrix {matrix!r} cannot place a pixel on the stage "
            f"(its determinant is {determinant!r}): run camera-calibration again"
        )
    return math.sqrt(abs(determinant))


def survey_well(
    data: SelectionInput, well: str, params: LowMagSurveyParams
) -> WellSurvey:
    """Every dark object and tile of ``well``'s frames, on the stage, once each, inside the well.

    A frame whose ``focus`` is ``unlocked`` is not read (FM-154): the autofocus
    lost the sample there, and what it shows is not the plane of the cells. Its
    region keeps no object and no tile, so the fields over it lose coverage.

    Raises:
        SelectionError: The camera matrix is singular, the well is not one of
            the scan's or not circular, or a frame cannot be read.
    """
    scan = data.scan
    geometry = _circular_geometry(data, well)
    matrix = scan.matrix_um_per_px
    shape = scan.image_shape_px
    scale = frame_scale_um_per_px(matrix)
    tile_px = max(1, round(params.tile_um / scale))
    min_area_px = max(1, math.ceil(params.min_area_um2 / scale**2))
    band_px = (params.texture_low_um / scale, params.texture_high_um / scale)
    halo_px = round(params.halo_um / scale)

    frames = scan.frames(well)
    centres = np.array([(f.x_um, f.y_um) for f in frames], dtype=np.float64).reshape(
        -1, 2
    )
    objects: list[tuple[float, ...]] = []
    tiles: list[tuple[float, ...]] = []
    for index, frame in enumerate(frames):
        if frame.focus == "unlocked":
            continue
        problem = ""
        try:
            image = scan.image(frame)
        except DatasetError as exc:
            problem = f"well {well}: {exc}"
        if problem:
            # Raised outside the handler: nothing keeps the loader's error (FM-98).
            raise SelectionError(problem)
        contrast = relative_contrast(
            image, background_sigma_px=params.background_sigma_um / scale
        )
        found, mask = find_dark_objects(
            contrast,
            smooth_sigma_px=params.smooth_sigma_um / scale,
            k_noise=params.k_noise,
            min_contrast=params.min_contrast,
            min_area_px=min_area_px,
            extent_fraction=params.extent_fraction,
            diameter_depth_fraction=params.diameter_depth_fraction,
        )
        texture = tile_texture(
            contrast,
            mask,
            tile_px=tile_px,
            band_sigmas_px=band_px,
            noise_sigma_px=params.texture_noise_sigma_px,
            dilate_px=halo_px,
            min_kept_fraction=params.min_tile_kept_fraction,
        )
        for row, col, diameter, darkest in zip(
            found.row_px,
            found.col_px,
            found.diameter_px,
            found.min_contrast,
            strict=True,
        ):
            x, y = stage_of_pixel(
                frame.x_um,
                frame.y_um,
                float(col),
                float(row),
                matrix=matrix,
                image_shape_px=shape,
            )
            reach = math.hypot(x - frame.x_um, y - frame.y_um)
            objects.append(
                (x, y, float(diameter) * scale, float(darkest), float(index), reach)
            )
        for i in range(texture.shape[0]):
            for j in range(texture.shape[1]):
                col = j * tile_px + (tile_px - 1) / 2.0
                row = i * tile_px + (tile_px - 1) / 2.0
                x, y = stage_of_pixel(
                    frame.x_um,
                    frame.y_um,
                    col,
                    row,
                    matrix=matrix,
                    image_shape_px=shape,
                )
                tiles.append((x, y, float(texture[i, j])))
        tiles = _keep_owned(
            tiles, centres, index, geometry, start=len(tiles) - texture.size
        )

    object_array = _merged(
        np.array(objects, dtype=np.float64).reshape(-1, 6), params.merge_radius_um
    )
    inside = geometry.contains_many(object_array[:, 0], object_array[:, 1])
    object_array = object_array[inside]
    tile_array = np.array(tiles, dtype=np.float64).reshape(-1, 3)
    return WellSurvey(
        well=well,
        x_um=object_array[:, 0],
        y_um=object_array[:, 1],
        diameter_um=object_array[:, 2],
        min_contrast=object_array[:, 3],
        tile_x_um=tile_array[:, 0],
        tile_y_um=tile_array[:, 1],
        tile_texture=tile_array[:, 2],
        tile_area_um2=(tile_px * scale) ** 2,
    )


def candidates_of(
    data: SelectionInput, well: str, survey: WellSurvey, params: LowMagSurveyParams
) -> list[Candidate]:
    """The candidate fields of the imaging objective over ``well`` that ``data.allowed`` keeps, with their numbers, in lattice order.

    The lattice steps by ``data.fov_um``, the extent of the imaging field in
    stage axes, from the well's centre: for a turned camera that is the bounding
    box of the turned footprint, so two neighbours never overlap and ``route``'s
    overlap check passes. For ``placement="greedy"`` the candidates are the
    ``greedy_subdivisions`` squared copies of that lattice shifted by a fraction of
    a step, which overlap: ``rank_candidates`` keeps the ones that do not. "In a
    field" is ``route``'s own test on the pixel offset (``|d_col| <= W/2``,
    ``|d_row| <= H/2``), never re-derived (FM-132). The shifted lattices come in
    order, each with its rows first, then its columns.

    Raises:
        SelectionError: The well is not one of the scan's or not circular, or
            the imaging camera cannot be inverted.
    """
    geometry = _circular_geometry(data, well)
    centre_x, centre_y = geometry.center_um
    radius = geometry.size_um[0] / 2.0
    step_x, step_y = data.fov_um
    height, width = data.image_shape_px
    try:
        inverse = _inverse_of(data.matrix_um_per_px)
    except ValueError as exc:
        raise SelectionError(f"the imaging camera cannot be inverted: {exc}") from None
    field_area_um2 = frame_scale_um_per_px(data.matrix_um_per_px) ** 2 * width * height

    diameter = survey.diameter_um
    is_cell = (diameter >= params.cell_min_diameter_um) & (
        diameter <= params.cell_max_diameter_um
    )
    is_large = diameter >= params.large_min_diameter_um
    texture = survey.tile_texture
    is_rough = (
        np.zeros(len(texture), dtype=bool)
        if params.max_texture is None
        else np.nan_to_num(texture, nan=0.0) > params.max_texture
    )
    is_known = ~np.isnan(texture)
    tile_side_um = math.sqrt(survey.tile_area_um2)
    tile_grid = _grid_of(survey.tile_x_um, survey.tile_y_um, tile_side_um)
    probe_dx, probe_dy = _probe_offsets(
        data.matrix_um_per_px, data.image_shape_px, params.coverage_probes_per_side
    )

    reach_x = math.ceil(radius / step_x) + 1
    reach_y = math.ceil(radius / step_y) + 1
    parts = params.greedy_subdivisions if params.placement == "greedy" else 1
    origins = [
        (
            centre_x + params.lattice_offset_x_um + a * step_x / parts,
            centre_y + params.lattice_offset_y_um + b * step_y / parts,
        )
        for b in range(parts)
        for a in range(parts)
    ]
    candidates: list[Candidate] = []
    for (origin_x, origin_y), j, i in itertools.product(
        origins, range(-reach_y, reach_y + 1), range(-reach_x, reach_x + 1)
    ):
        x = origin_x + i * step_x
        y = origin_y + j * step_y
        if not data.allowed(x, y, well):
            continue
        held = _held(survey.x_um - x, survey.y_um - y, inverse, width, height)
        held_tiles = _held(
            survey.tile_x_um - x, survey.tile_y_um - y, inverse, width, height
        )
        n_cells = int(np.sum(held & is_cell))
        known = held_tiles & is_known
        corners = footprint(
            x, y, matrix=data.matrix_um_per_px, image_shape_px=data.image_shape_px
        )
        farthest = max(math.hypot(cx - centre_x, cy - centre_y) for cx, cy in corners)
        candidates.append(
            Candidate(
                x_um=x,
                y_um=y,
                n_cells=n_cells,
                n_large=int(np.sum(held & is_large)),
                rough_fraction=float(
                    np.sum(known & is_rough) / max(int(np.sum(known)), 1)
                ),
                coverage=_covered_share(
                    x + probe_dx,
                    y + probe_dy,
                    survey,
                    tile_grid,
                    tile_side_um,
                    params.coverage_reach_tiles,
                ),
                density_per_mm2=n_cells / (field_area_um2 / 1e6),
                wall_distance_um=radius - farthest,
            )
        )
    return candidates


def rank_candidates(
    candidates: Sequence[Candidate], params: LowMagSurveyParams, data: SelectionInput
) -> list[Candidate]:
    """The candidates that pass every ban, the most cells first, then the farthest from the wall, at most ``data.fields_per_well``.

    A field is banned when it holds more large objects than ``max_large``, a
    share of rough tiles above ``max_rough_fraction`` (when ``max_texture`` is
    set), less than ``min_coverage`` of it was read, it holds fewer cells than
    ``min_cells``, or it is denser than ``max_density_per_mm2``. Ties keep the
    lattice order (a stable sort). With ``placement="greedy"`` the candidates
    overlap, and a field is taken in that order only when it overlaps none taken
    before (their extents ``data.fov_um`` apart or more on one axis, as on the
    lattice): the greedy choice, which is not the best packing but put more cells
    in the first fields on 2026-11-24 (``docs/algorithms/low-mag-survey.md``).

    The rough share, not one rough tile: a field holds about 30 tiles, and the
    clean glass of C4 on 2026-11-24 has 1.6 % of its tiles above the texture
    threshold, so "any rough tile" banned 22 % of the fields more than 1.5 mm
    from the wall; a share above 10 % bans 2 % of those and 16 % of the fields
    0.8 to 1.5 mm from it (``docs/algorithms/low-mag-survey.md``).
    """
    kept = [
        c
        for c in candidates
        if c.n_large <= params.max_large
        and c.rough_fraction <= params.max_rough_fraction
        and c.coverage >= params.min_coverage
        and c.n_cells >= params.min_cells
        and (
            params.max_density_per_mm2 is None
            or c.density_per_mm2 <= params.max_density_per_mm2
        )
    ]
    kept.sort(key=lambda c: (-c.n_cells, -c.wall_distance_um))
    if params.placement == "greedy":
        kept = _without_overlap(kept, data.fov_um, data.fields_per_well)
    return kept[: data.fields_per_well]


def _without_overlap(
    ranked: Sequence[Candidate], extent_um: tuple[float, float], limit: int
) -> list[Candidate]:
    """The candidates in their order, each kept when it overlaps none kept before, at most ``limit``.

    Two fields overlap when their centres are closer than the extent on both
    axes. The extent is the bounding box of the turned footprint, so fields kept
    here never overlap and ``route``'s check passes (a little room is lost on a
    turned camera, as on the lattice). The kept centres are bucketed by extent,
    so each candidate is compared with the nine buckets around it only.
    """
    width, height = extent_um
    buckets: dict[tuple[int, int], list[Candidate]] = {}
    kept: list[Candidate] = []
    for c in ranked:
        if len(kept) >= limit:
            break
        bx, by = math.floor(c.x_um / width), math.floor(c.y_um / height)
        if any(
            abs(k.x_um - c.x_um) < width and abs(k.y_um - c.y_um) < height
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for k in buckets.get((bx + dx, by + dy), ())
        ):
            continue
        kept.append(c)
        buckets.setdefault((bx, by), []).append(c)
    return kept


def _circular_geometry(data: SelectionInput, well: str) -> WellGeometry:
    geometry = data.scan.geometry.get(well)
    if geometry is None:
        raise SelectionError(
            f"well {well!r} is not a well of this scan ({', '.join(data.scan.wells)})"
        )
    if not geometry.circular:
        raise SelectionError(
            f"well {well} is not circular: low-mag-survey measures the distance to a "
            f"round wall only"
        )
    return geometry


def _merged(
    objects: npt.NDArray[np.float64], radius_um: float
) -> npt.NDArray[np.float64]:
    """One row per object seen by several frames: the sighting nearest its own frame's centre.

    Rows are ``(x, y, diameter, contrast, frame, distance to the frame's
    centre)``. Two sightings from **different** frames closer than
    ``radius_um`` are one object. Owning an object by the frame whose
    centre is nearest its measured position is not enough: an object on the
    line between two frames' centres is measured about 1 um apart in each, and
    each frame then owns its own sighting (a disc at a well's centre, in all
    four frames of a 2 x 2 plan, was counted twice).

    The neighbours are found on a grid of ``radius_um`` squares (the
    sighting's square and the eight around it), so the work grows with the
    number of objects, not its square: a dense 10x well holds about 9000.
    """
    if len(objects) == 0 or radius_um <= 0.0:
        return objects
    order = np.argsort(objects[:, 5], kind="stable")
    cell_of = np.floor(objects[:, :2] / radius_um).astype(np.int64)
    grid: dict[tuple[int, int], list[int]] = {}
    for k, (gx, gy) in enumerate(cell_of.tolist()):
        grid.setdefault((gx, gy), []).append(k)
    taken = np.zeros(len(objects), dtype=bool)
    kept: list[int] = []
    for k in order.tolist():
        if taken[k]:
            continue
        kept.append(k)
        gx, gy = cell_of[k].tolist()
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for other in grid.get((nx, ny), ()):
                    near = math.hypot(
                        objects[other, 0] - objects[k, 0],
                        objects[other, 1] - objects[k, 1],
                    )
                    if near <= radius_um and objects[other, 4] != objects[k, 4]:
                        taken[other] = True
    merged: npt.NDArray[np.float64] = objects[np.sort(np.array(kept, dtype=np.int64))]
    return merged


def _keep_owned(
    points: list[tuple[float, ...]],
    centres: npt.NDArray[np.float64],
    index: int,
    geometry: WellGeometry,
    *,
    start: int,
) -> list[tuple[float, ...]]:
    """Drop, among ``points[start:]`` (this frame's), those another frame's centre is nearer to, or outside the well.

    The scan's frames overlap: a point in the overlap is kept by the frame whose
    centre is nearest, so it is counted once (ties: the first frame).
    """
    new = np.array(points[start:], dtype=np.float64).reshape(
        -1, len(points[0]) if points else 2
    )
    if len(new) == 0:
        return points
    distance = (new[:, 0:1] - centres[None, :, 0]) ** 2 + (
        new[:, 1:2] - centres[None, :, 1]
    ) ** 2
    owned = np.argmin(distance, axis=1) == index
    inside = geometry.contains_many(new[:, 0], new[:, 1])
    kept = [points[start + k] for k in np.flatnonzero(owned & inside)]
    return points[:start] + kept


def _held(
    dx: npt.NDArray[np.float64],
    dy: npt.NDArray[np.float64],
    inverse: tuple[tuple[float, float], tuple[float, float]],
    width: int,
    height: int,
) -> npt.NDArray[np.bool_]:
    """Which stage offsets from a field's centre the field holds: ``route``'s rule on the pixel offset."""
    d_col, d_row = _pixel_offsets(dx, dy, inverse)
    held: npt.NDArray[np.bool_] = (np.abs(d_col) <= width / 2.0) & (
        np.abs(d_row) <= height / 2.0
    )
    return held


def _probe_offsets(
    matrix: Matrix2, image_shape_px: tuple[int, int], per_side: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Stage offsets from a field's centre of an even ``per_side`` x ``per_side`` grid of points over its footprint."""
    height, width = image_shape_px
    fractions = (np.arange(per_side) + 0.5) / per_side - 0.5
    offsets = [
        stage_of_pixel(
            0.0,
            0.0,
            (width - 1) / 2.0 + fc * width,
            (height - 1) / 2.0 + fr * height,
            matrix=matrix,
            image_shape_px=image_shape_px,
        )
        for fr in fractions
        for fc in fractions
    ]
    array = np.array(offsets, dtype=np.float64)
    return array[:, 0], array[:, 1]


def _grid_of(
    x_um: npt.NDArray[np.float64], y_um: npt.NDArray[np.float64], side_um: float
) -> dict[tuple[int, int], list[int]]:
    """The indices of points by the ``side_um`` square they fall in."""
    grid: dict[tuple[int, int], list[int]] = {}
    cells = np.floor(np.column_stack([x_um, y_um]) / side_um).astype(np.int64)
    for k, (gx, gy) in enumerate(cells.tolist()):
        grid.setdefault((gx, gy), []).append(k)
    return grid


def _covered_share(
    probe_x: npt.NDArray[np.float64],
    probe_y: npt.NDArray[np.float64],
    survey: WellSurvey,
    grid: dict[tuple[int, int], list[int]],
    side_um: float,
    reach_tiles: float,
) -> float:
    """The share of a field's probe points that lie on a tile that was read.

    A probe is covered when a read tile's centre is within ``reach_tiles``
    tile sides of it (0.75: between the half side and the half diagonal of the
    square). Counting the tiles whose centre falls in the field instead was
    off by about 15 % either way, as the tiles and the fields are not aligned,
    and the 0.9 threshold then banned 315 of the 885 fields of C4 as unread.
    """
    reach = reach_tiles * side_um
    span = math.ceil(reach_tiles)
    covered = 0
    for px, py in zip(probe_x.tolist(), probe_y.tolist(), strict=True):
        gx, gy = math.floor(px / side_um), math.floor(py / side_um)
        found = False
        for nx in range(gx - span, gx + span + 1):
            for ny in range(gy - span, gy + span + 1):
                for k in grid.get((nx, ny), ()):
                    if (
                        math.hypot(survey.tile_x_um[k] - px, survey.tile_y_um[k] - py)
                        <= reach
                    ):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        covered += found
    return covered / len(probe_x)
