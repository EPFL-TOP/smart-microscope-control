"""Choose the fields to image again from a scan's own frames, scored by weights the owner tunes (#343, m3 §11).

A prototype, not the ``route`` plugin. ``route`` picks the fields that hold the most
wanted cells; the owner wants to find out, on real 40x scans, what else makes a
position good for the experiment (isolated cells, a clean background, a field away
from the wall), and that is a matter of looking at the chosen fields with different
weights, offline, before anything changes in ``src/smc/``. So one command reads a
run document that has ``scan-plan``, ``scan`` and ``cells`` and, for each well:

1. takes the scan's own frames as the candidate fields (every frame the autofocus
   had not lost the sample on), and computes the features of each from the cells in
   its footprint and from its pixels;
2. scores them with the weights it is given, picks fields greedily by score (no two
   overlapping) and orders them with ``route``'s ``order_route``;
3. prints the ranked fields with their features, draws a PNG per well and writes
   ``selection.json``;
4. writes a **copy** of the run document in which a record named
   ``route-prototype`` fills the ``route`` slot, so that
   ``smc run nis-export --run-doc <copy>`` exports the list unchanged.

Why the scan's frames and not ``route``'s lattice: every feature, and the Z and the
autofocus offset of a stop, are then measured on the very frame that will be imaged
again, no candidate lies on an area the scan did not see, and what the owner looks
at on the PNG is what was scored. The features are written for any field centre
(``cells_in_field(x, y)``), so the follow-up can move them to a lattice.

"Inside a field" and "overlap" are *the* arithmetic of ``route.algorithm``, which
this imports, private helpers included, and does not re-derive (FM-132): two correct
definitions of "inside" differ by an ulp on an edge. The follow-up issue makes those
helpers part of ``choose_fields``, and the imports go.

Decisions that were made on measurements, not on taste:

* **The scan's camera is the camera.** Every footprint, overlap and mosaic uses the
  scan's matrix and frame shape, which are not square, turned and mirrored (FM-73);
  the plan's copy is never read (FM-124).
* **A position is the stage's readback** (FM-167), and an ``unlocked`` frame is never a
  candidate (FM-154): its Z is wherever the autofocus search had got to.
* **No matrix product on coordinate arrays** (FM-131): the entries of the inverse are
  written out, as in ``route.algorithm``.
* **The PNG's canvas is cut to the well**, never to the frames' bounding box
  (FM-136), and every float is clipped before it becomes an integer (FM-75).
* **Output is ASCII** (FM-20) and every file is UTF-8 (FM-21): a Windows console
  decodes with a code page, and an objective's name may hold a Greek letter.
* **The table is printed before any file is written** (FM-30, FM-34): a write that
  fails must not hide minutes of work.

The score is the owner's, not the programmer's: nothing here is tuned, and the only
default is the one that reproduces ``route``'s own preference (the density).

Not in this script, on purpose: any change in ``src/smc/``, lattice candidates,
another objective, and any default that the owner has not chosen on a real scan.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ValidationError

import smc

# The route's own arithmetic for "inside a field" and for "overlap" (FM-132): private
# helpers, imported on purpose so that the prototype and the plugin cannot disagree.
from smc.plugins.geometry import Matrix2, footprint, fov_of
from smc.plugins.route.algorithm import (
    _TOUCH_PX,
    _inverse_of,
    _pixel_offsets,
    allowed,
    order_route,
    path_length_um,
)
from smc.plugins.slots import (
    Cell,
    CellsResult,
    RoutePosition,
    RouteResult,
    ScanPlanResult,
    ScanResult,
    WellGeometry,
    load_cells,
    load_frame,
    normalise_well,
    well_geometry,
)
from smc.run import (
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)

__all__ = ["Candidate", "Refusal", "Skipped", "main"]

_M = TypeVar("_M", bound=BaseModel)

#: The weights, in the order the table and the files list them.
WEIGHT_NAMES = ("density", "isolation", "purity", "wall", "sharpness", "background")

Floats = npt.NDArray[np.float64]


class Refusal(ValueError):  # noqa: N818 (a refusal is read as one: the command exits 2 with its text)
    """The command cannot do what it was asked; the text says what to change."""


# ======================================================================= Features


def cells_in_field(
    xy: Floats,
    tree: Any,
    x_um: float,
    y_um: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> npt.NDArray[np.int64]:
    """Indices into ``xy`` of the cells inside the footprint of a frame centred at ``(x_um, y_um)``, the edge included.

    A ball query at the farthest corner narrows the cells (a well holds up to
    300 000), and the exact test is the route's: ``|d_col| <= W / 2`` and
    ``|d_row| <= H / 2`` on the pixel offsets of the stage offset, so a camera that
    is turned or mirrored, and not square, is counted as the route counts it.

    Args:
        xy: ``(N, 2)`` stage positions of every cell of the well, any label.
        tree: A ``scipy.spatial.cKDTree`` of ``xy`` (unused when ``xy`` is empty).
        x_um: Stage X of the frame's centre, in um.
        y_um: Stage Y of the frame's centre, in um.
        matrix: The camera's matrix, um per pixel.
        image_shape_px: ``(height, width)`` of the frame.

    Returns:
        The indices, ascending.
    """
    if len(xy) == 0:
        return np.empty(0, dtype=np.int64)
    corners = footprint(x_um, y_um, matrix=matrix, image_shape_px=image_shape_px)
    # One um of slack: the ball only narrows, the exact test below decides.
    reach = max(math.dist((x_um, y_um), corner) for corner in corners) + 1.0
    near = np.asarray(tree.query_ball_point((x_um, y_um), reach), dtype=np.int64)
    near.sort()
    height, width = image_shape_px
    d_col, d_row = _pixel_offsets(
        xy[near, 0] - x_um, xy[near, 1] - y_um, _inverse_of(matrix)
    )
    inside = (np.abs(d_col) <= width / 2.0) & (np.abs(d_row) <= height / 2.0)
    return near[inside]


def neighbour_distances_um(xy: Floats, tree: Any = None) -> Floats:
    """The distance of each cell to its nearest *other* cell, in um; ``inf`` when it has none.

    Two cells at one position are at distance 0 from each other, and a cell alone in
    its well has no neighbour and so no distance: "unknown" is not a number, and the
    caller counts it as isolated.
    """
    if len(xy) == 0:
        return np.empty(0, dtype=np.float64)
    if tree is None:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
    # k=2: the nearest neighbour of a point is itself, so the second is the other one.
    distances, _ = tree.query(xy, k=2)
    return np.asarray(distances[:, 1], dtype=np.float64)


def wall_distance_um(
    well: WellGeometry, corners: Sequence[tuple[float, float]]
) -> float:
    """The distance from a footprint's corners to the well's wall, in um; negative when a corner is outside.

    A round well: the radius less the farthest corner. A square one: the smaller of
    the two margins in the plate's axes (``to_plate_frame``, so a plate that is turned
    is measured along its own walls). Both wells are convex, so the corners are enough.
    """
    offsets = [well.to_plate_frame(x, y) for x, y in corners]
    half_u = well.size_um[0] / 2.0
    if well.circular:
        return half_u - max(math.hypot(u, v) for u, v in offsets)
    half_v = well.size_um[1] / 2.0
    return min(
        half_u - max(abs(u) for u, _ in offsets),
        half_v - max(abs(v) for _, v in offsets),
    )


def default_min_spacing_um(cells: Sequence[Cell], matrix: Matrix2) -> float | None:
    """Three times the median length of ``cells``, in um; ``None`` for no cell.

    A cell's length is the longer side of its box, in pixels, times the pixel's
    size, which for a camera that is not square is the geometric mean
    ``sqrt(|det|)``. The factor 3 is **chosen, not measured**: it is the first value
    to try at the stand, and the output and the record say so.
    """
    if not cells:
        return None
    (m00, m01), (m10, m11) = matrix
    pixel_um = math.sqrt(abs(m00 * m11 - m01 * m10))
    lengths = [
        max(c.box_px[2] - c.box_px[0], c.box_px[3] - c.box_px[1]) * pixel_um
        for c in cells
    ]
    return 3.0 * float(np.median(lengths))


def frame_stats(frame: npt.NDArray[Any]) -> tuple[float | None, float]:
    """``(sharpness, background)`` of a frame.

    ``background`` is the median. ``sharpness`` is the variance of the 4-neighbour
    Laplacian of the interior, divided by the median squared, so that it does not
    depend on the exposure; ``None`` when the median is not above 0 (nothing to
    divide by). It is **not a focus measure**: cells add contrast, so a frame full
    of cells reads sharper than an empty one in focus, and the output calls the
    column ``sharp``. The frame is a ``float64`` first, because a Laplacian in
    ``uint16`` wraps.

    Raises:
        ValueError: The frame is not two-dimensional, is empty, or holds a value that
            is not finite.
    """
    values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"the frame is not two-dimensional (shape {values.shape})")
    if values.size == 0:
        raise ValueError("the frame is empty")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("the frame holds values that are not finite")
    background = float(np.median(values))
    if min(values.shape) < 3 or not background > 0.0:
        return None, background
    laplacian = (
        4.0 * values[1:-1, 1:-1]
        - values[:-2, 1:-1]
        - values[2:, 1:-1]
        - values[1:-1, :-2]
        - values[1:-1, 2:]
    )
    with np.errstate(all="ignore"):
        sharpness = float(np.var(laplacian)) / background**2
    return (sharpness if math.isfinite(sharpness) else None), background


@dataclass(frozen=True, slots=True)
class Candidate:
    """A frame of the scan as a field that could be imaged again, with its features.

    ``frame`` is the frame's position in ``ScanResult.frames`` (the index a cell's
    ``frame`` uses), at its **readback** position. ``counts`` has every class of the
    cells stage, zeros included; ``n_all`` counts every class, ``n_wanted`` the wanted
    labels. ``isolated`` is the number of wanted cells whose nearest other cell (any
    label, anywhere in the well) is at least ``min_spacing_um`` away. ``sharpness``
    and ``background`` are ``None`` when the frame was not read or could not be.
    """

    frame: int
    name: str
    x_um: float
    y_um: float
    z_um: float | None
    pfs_offset: float | None
    focus: str
    counts: dict[str, int]
    n_all: int
    n_wanted: int
    density_per_mm2: float
    isolated: int
    nn_median_um: float | None
    nn_min_um: float | None
    wall_um: float
    sharpness: float | None
    background: float | None


@dataclass(slots=True)
class Skipped:
    """The frames of a well that were not candidates, by reason."""

    unlocked: int = 0
    refused: int = 0


def candidates_of(
    well: str,
    scan: ScanResult,
    cells: Sequence[Cell],
    geometry: WellGeometry,
    *,
    labels: Sequence[str],
    classes: Sequence[str],
    min_spacing_um: float,
    margin_um: float,
    exclude_centre_um: float,
    stats: Mapping[int, tuple[float | None, float] | None],
) -> tuple[tuple[Candidate, ...], Skipped]:
    """The candidates of one well, and how many of its frames were left out and why.

    The well's frames at the scan's readback, minus the ``unlocked`` ones (FM-154:
    the autofocus had not locked, so the Z and the offset of the frame are not
    the sample's; ``off`` and ``none`` stay, with no offset, as in ``route``), minus
    the footprints that ``route.algorithm.allowed`` refuses for the wall margin and
    the excluded centre. The camera is the scan's (FM-124).

    ``stats`` maps a frame's position in ``scan.frames`` to ``(sharpness,
    background)``, or to ``None`` for a frame that could not be read; a frame it does
    not name was not asked for (``--no-pixels``). Both give a candidate with no pixel
    features.
    """
    matrix, shape = scan.matrix_um_per_px, scan.image_shape_px
    height, width = shape
    (m00, m01), (m10, m11) = matrix
    area_mm2 = width * height * abs(m00 * m11 - m01 * m10) / 1e6
    wanted_labels = set(labels)
    in_well = [c for c in cells if c.well == well]
    xy = np.array([(c.x_um, c.y_um) for c in in_well], dtype=np.float64).reshape(-1, 2)
    wanted = np.array([c.label in wanted_labels for c in in_well], dtype=bool)
    tree: Any = None
    if len(xy):
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
    nearest = neighbour_distances_um(xy, tree)

    skipped = Skipped()
    found: list[Candidate] = []
    for position in scan.frame_index().get(well, []):
        frame = scan.frames[position]
        if frame.focus == "unlocked":
            skipped.unlocked += 1
            continue
        if not allowed(
            frame.x_um,
            frame.y_um,
            geometry,
            matrix=matrix,
            image_shape_px=shape,
            exclude_centre_um=exclude_centre_um,
            wall_margin_um=margin_um,
        ):
            skipped.refused += 1
            continue
        inside = cells_in_field(
            xy, tree, frame.x_um, frame.y_um, matrix=matrix, image_shape_px=shape
        )
        tally = Counter(in_well[i].label for i in inside)
        mine = inside[wanted[inside]]
        distances = nearest[mine]
        finite = distances[np.isfinite(distances)]
        pixels = stats.get(position)
        found.append(
            Candidate(
                frame=position,
                name=frame.name,
                x_um=frame.x_um,
                y_um=frame.y_um,
                z_um=frame.z_um,
                pfs_offset=frame.pfs_offset,
                focus=frame.focus,
                counts={label: tally.get(label, 0) for label in classes},
                n_all=len(inside),
                n_wanted=len(mine),
                density_per_mm2=len(mine) / area_mm2,
                isolated=int((distances >= min_spacing_um).sum()),
                nn_median_um=float(np.median(finite)) if finite.size else None,
                nn_min_um=float(finite.min()) if finite.size else None,
                wall_um=wall_distance_um(
                    geometry,
                    footprint(
                        frame.x_um, frame.y_um, matrix=matrix, image_shape_px=shape
                    ),
                ),
                sharpness=pixels[0] if pixels is not None else None,
                background=pixels[1] if pixels is not None else None,
            )
        )
    return tuple(found), skipped


# ========================================================================= Score

#: What each weight is when it is not given: the density alone, which is what ``route``
#: prefers, so that every other criterion is a deliberate step of the owner.
DEFAULT_WEIGHTS: dict[str, float] = {
    name: (1.0 if name == "density" else 0.0) for name in WEIGHT_NAMES
}
_MAX_WEIGHT = 1e6


def parse_weights(pairs: Sequence[str]) -> dict[str, float]:
    """The six weights from ``NAME=VALUE`` pairs: the defaults, each replaced by the pair that names it.

    A weight is finite and in [0, 1e6], at least one is above 0, a name is one of
    ``WEIGHT_NAMES`` and is given once (a value typed twice is a typo, and the last
    one silently winning would hide it).

    Raises:
        Refusal: A pair is malformed, a name unknown or repeated, a value not a
            number in range, or every weight is 0.
    """
    weights = dict(DEFAULT_WEIGHTS)
    given: set[str] = set()
    for pair in pairs:
        name, equals, text = pair.partition("=")
        if not equals:
            raise Refusal(f"--weight expects NAME=VALUE, got {pair!r}")
        if name not in WEIGHT_NAMES:
            raise Refusal(
                f"--weight name {name!r} is not one of {', '.join(WEIGHT_NAMES)}"
            )
        if name in given:
            raise Refusal(f"--weight {name} is given twice")
        given.add(name)
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not (math.isfinite(value) and 0.0 <= value <= _MAX_WEIGHT):
            raise Refusal(
                f"--weight {name} must be a number from 0 to {_MAX_WEIGHT:.0f}, "
                f"got {text!r}"
            )
        weights[name] = value
    if not any(weight > 0.0 for weight in weights.values()):
        raise Refusal("at least one weight must be above 0")
    return weights


def percentile_rank(value: float, values: Sequence[float]) -> float:
    """The share of ``values`` below ``value``, a tie counting for half: in (0, 1), never 0 or 1 for a member.

    The three values 1, 2, 3 rank 1/6, 3/6, 5/6, and a value shared by all of them
    ranks 1/2, which says "no better than the others" and not "the best".
    """
    if len(values) == 0:
        raise ValueError("a rank needs at least one value")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    below = int(np.searchsorted(ordered, value, side="left"))
    up_to = int(np.searchsorted(ordered, value, side="right"))
    return (below + 0.5 * (up_to - below)) / len(ordered)


@dataclass(frozen=True, slots=True)
class Scored:
    """A candidate, its score in [0, 1] and the six parts that made it, each in [0, 1]."""

    candidate: Candidate
    score: float
    parts: dict[str, float]


def _unit(value: float) -> float:
    """``value`` clipped to [0, 1]: a ratio that a rounding put a hair outside is not a different score."""
    return max(0.0, min(1.0, value))


def score_candidates(
    candidates: Sequence[Candidate], weights: Mapping[str, float]
) -> list[Scored]:
    """Score a well's candidates: the weighted mean of six parts, each in [0, 1], in the candidates' order.

    The *pool* is the candidates given, less those that have no ``sharpness`` or
    ``background`` while that weight is above 0 (the caller counts the difference).
    Every part is taken over the pool, before ``min_cells``, so that a rank does not
    depend on how many cells a field holds:

    * ``density``: the density over the pool's largest (0 when that is 0);
    * ``isolation``: the share of the field's wanted cells that are isolated;
    * ``purity``: wanted cells over all cells (doublets and debris lower it);
    * ``wall``: the distance to the wall over the pool's largest (0 when that is 0);
    * ``sharpness``: its ``percentile_rank`` among the pool;
    * ``background``: one less the ``percentile_rank`` of the background (darker is
      better).

    The score divides by the sum of the weights, so that scaling all of them changes
    nothing and a weight is read as a share.
    """
    pool = [
        c
        for c in candidates
        if not (
            (weights["sharpness"] > 0.0 and c.sharpness is None)
            or (weights["background"] > 0.0 and c.background is None)
        )
    ]
    if not pool:
        return []
    densest = max(c.density_per_mm2 for c in pool)
    widest = max(c.wall_um for c in pool)
    sharpnesses = [c.sharpness for c in pool if c.sharpness is not None]
    backgrounds = [c.background for c in pool if c.background is not None]
    total = sum(weights[name] for name in WEIGHT_NAMES)
    scored: list[Scored] = []
    for c in pool:
        parts = {
            "density": _unit(c.density_per_mm2 / densest) if densest > 0.0 else 0.0,
            "isolation": c.isolated / c.n_wanted if c.n_wanted else 0.0,
            "purity": c.n_wanted / c.n_all if c.n_all else 0.0,
            "wall": _unit(c.wall_um / widest) if widest > 0.0 else 0.0,
            "sharpness": (
                percentile_rank(c.sharpness, sharpnesses)
                if c.sharpness is not None
                else 0.0
            ),
            "background": (
                1.0 - percentile_rank(c.background, backgrounds)
                if c.background is not None
                else 0.0
            ),
        }
        score = sum(weights[name] * parts[name] for name in WEIGHT_NAMES) / total
        scored.append(Scored(c, _unit(score), parts))
    return scored


# ========================================================================= Choice


def choose(
    scored: Sequence[Scored],
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    n: int,
    min_cells: int,
    min_score: float,
) -> list[Scored]:
    """The fields to image, best score first, none overlapping another; at most ``n``.

    Greedy by ``(-score, frame)``, so equal scores go to the lowest frame and the same
    scan gives the same fields. A candidate with fewer than ``min_cells`` wanted
    cells is left out, a candidate that overlaps a chosen one is left out, and the
    choice stops at the first score below ``min_score``. Two fields overlap by the
    route's rule: with one camera turn for both, the centres differ by less than a frame
    in both pixel offsets; at exactly a frame they touch, and touching is allowed. A
    candidate's score is static (it does not depend on what else is chosen), so there
    is no recount.
    """
    inverse = _inverse_of(matrix)
    height, width = image_shape_px
    chosen: list[Scored] = []
    for s in sorted(scored, key=lambda s: (-s.score, s.candidate.frame)):
        if len(chosen) >= n or s.score < min_score:
            break
        if s.candidate.n_wanted < min_cells:
            continue
        if chosen:
            d_col, d_row = _pixel_offsets(
                np.array([c.candidate.x_um for c in chosen]) - s.candidate.x_um,
                np.array([c.candidate.y_um for c in chosen]) - s.candidate.y_um,
                inverse,
            )
            overlapping = (np.abs(d_col) < width - _TOUCH_PX) & (
                np.abs(d_row) < height - _TOUCH_PX
            )
            if bool(overlapping.any()):
                continue
        chosen.append(s)
    return chosen


def order_stops(
    points: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> list[int]:
    """The visiting order of a well's fields: from the one nearest ``reference``, then the route's shortest path.

    ``reference`` is where the stage is coming from: the previous well's last stop,
    or the first planned field of the well for the first one. Ties for the nearest
    go to the field chosen first. The indices are into ``points``.
    """
    if not points:
        return []
    start = min(range(len(points)), key=lambda i: (math.dist(points[i], reference), i))
    return order_route(points, start=start)


def tally(
    fields: Sequence[tuple[float, float]],
    cells: Sequence[Cell],
    *,
    labels: Sequence[str],
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> tuple[list[dict[str, int]], int]:
    """Each field's counts of the wanted cells **no earlier field holds**, and the wanted cells in no field.

    ``fields`` are the centres in the order they were chosen, which is the order the
    cells are given away in: a cell on the edge two touching fields share goes to the
    first, so every cell is counted once and the counts and the uncovered add up to
    the wanted cells, as ``route``'s do. Each count has every wanted label, zeros
    included. "Inside" is ``cells_in_field``'s.
    """
    keys = list(dict.fromkeys(labels))
    wanted = [c for c in cells if c.label in keys]
    xy = np.array([(c.x_um, c.y_um) for c in wanted], dtype=np.float64).reshape(-1, 2)
    which = np.array([keys.index(c.label) for c in wanted], dtype=np.int64)
    tree: Any = None
    if len(xy):
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
    held = np.zeros(len(xy), dtype=bool)
    counts: list[dict[str, int]] = []
    for x, y in fields:
        inside = cells_in_field(
            xy, tree, x, y, matrix=matrix, image_shape_px=image_shape_px
        )
        fresh = inside[~held[inside]]
        tallies = np.bincount(which[fresh], minlength=len(keys))
        counts.append({label: int(tallies[k]) for k, label in enumerate(keys)})
        held[fresh] = True
    return counts, int((~held).sum())


# ============================================================== Pixels and the PNG
#
# The mosaic is for *looking*, not for measuring: a frame is reduced by block means,
# placed on the canvas by the nearest pixel, and a later frame overwrites an earlier
# one where they overlap. The placement error (up to ``k - 1`` source pixels of a
# cropped remainder) is below one canvas pixel.

#: Dark grey of a canvas pixel no frame reached, so that a hole is not a black pixel.
_UNPLACED_GREY = 40
#: A canvas is the well's bounding box plus this much on each side.
_CANVAS_MARGIN = 0.02
_CIRCLE_SEGMENTS = 128
_BLUE = (90, 90, 255)
_GREEN = (0, 255, 0)
_ROUTE = (255, 255, 160)
_WHITE = (255, 255, 255)
#: The classes, in the cells stage's order: gold, cyan, red, light green, magenta, white.
_CLASS_COLOURS = (
    (255, 215, 0),
    (0, 200, 255),
    (255, 80, 80),
    (120, 255, 120),
    (255, 120, 255),
    _WHITE,
)

if TYPE_CHECKING:
    from PIL import Image


def reduction_factor(
    um_per_canvas_px: float, pixel_size_um: float, shape: tuple[int, int]
) -> int:
    """How many source pixels make one tile pixel: the canvas's pixel over the camera's, at least 1.

    Bounded by the frame's smaller side on floats before it becomes an integer, so a
    pixel size that is far too small (a wrong profile entry, FM-121) gives a one-pixel
    tile and not an ``OverflowError`` (FM-74).

    Raises:
        ValueError: The pixel size is not finite and positive (0 means "not measured"
            and is reported, not defaulted: FM-104).
    """
    if not (math.isfinite(pixel_size_um) and pixel_size_um > 0.0):
        raise ValueError(
            f"the pixel size must be finite and positive, got {pixel_size_um!r}"
        )
    ratio = min(um_per_canvas_px / pixel_size_um, float(min(shape)))
    return max(1, math.floor(ratio + 0.5))


def block_mean(frame: npt.NDArray[Any], k: int) -> npt.NDArray[np.float32]:
    """The mean of each ``k`` x ``k`` block of a frame, as ``float32``; the remainder of an edge is cropped.

    Raises:
        ValueError: ``k`` is below 1 or larger than the frame's smaller side.
    """
    values = np.asarray(frame)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k!r}")
    rows, cols = values.shape
    if k > min(rows, cols):
        raise ValueError(f"k={k} is larger than the frame ({rows} x {cols} pixels)")
    if k == 1:
        return values.astype(np.float32)
    kept_rows, kept_cols = rows // k, cols // k
    blocks = values[: kept_rows * k, : kept_cols * k].astype(np.float64)
    return (
        blocks.reshape(kept_rows, k, kept_cols, k).mean(axis=(1, 3)).astype(np.float32)
    )


@dataclass(frozen=True, slots=True)
class Canvas:
    """Where a well's mosaic is: canvas pixel ``(c, r)`` is centred at ``x = x_min + (c + 0.5) s``, ``y = y_max - (r + 0.5) s``.

    Stage +X is to the right and +Y up, so the picture is the plate seen from above
    as the stage's axes have it, whatever the camera's turn.
    """

    x_min_um: float
    y_max_um: float
    um_per_px: float
    width_px: int
    height_px: int

    def blank(self) -> npt.NDArray[np.float32]:
        """A canvas of ``nan``: nothing is placed yet."""
        return np.full((self.height_px, self.width_px), np.nan, dtype=np.float32)

    def px_of(self, x_um: float, y_um: float) -> tuple[float, float]:
        """The canvas pixel ``(col, row)``, fractional, whose centre is a stage point."""
        return (
            (x_um - self.x_min_um) / self.um_per_px - 0.5,
            (self.y_max_um - y_um) / self.um_per_px - 0.5,
        )


def canvas_of(well: WellGeometry, png_max_px: int) -> Canvas:
    """The canvas of a well: its bounding box plus 2 % on each side, the longer side ``png_max_px`` pixels.

    The extent is the **well's**, never the frames': a frame read back at a wrong
    position (a unit slip, a corrupt record) is clipped, and does not grow the canvas
    to the size of the mistake (FM-136). A disc is its centre plus and minus the
    radius; a square well, the half-extents of its rectangle turned by the plate.
    """
    width, height = well.size_um
    if well.circular:
        half_x = half_y = width / 2.0
    else:
        angle = math.radians(well.rotation_deg)
        cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
        half_x = cos * width / 2.0 + sin * height / 2.0
        half_y = sin * width / 2.0 + cos * height / 2.0
    half_x *= 1.0 + _CANVAS_MARGIN
    half_y *= 1.0 + _CANVAS_MARGIN
    um_per_px = 2.0 * max(half_x, half_y) / png_max_px
    return Canvas(
        x_min_um=well.center_um[0] - half_x,
        y_max_um=well.center_um[1] + half_y,
        um_per_px=um_per_px,
        width_px=min(png_max_px, max(1, math.ceil(2.0 * half_x / um_per_px))),
        height_px=min(png_max_px, max(1, math.ceil(2.0 * half_y / um_per_px))),
    )


def _clamped(value: float, size: int) -> int:
    """``value`` as an index in ``[0, size]``, clamped as a float before it becomes an integer (FM-75, FM-74)."""
    return int(max(0.0, min(float(size), value)))


def place_frame(
    canvas: npt.NDArray[np.float32],
    tile: npt.NDArray[np.float32],
    x_um: float,
    y_um: float,
    *,
    matrix: Matrix2,
    x_min_um: float,
    y_max_um: float,
    um_per_px: float,
) -> None:
    """Write a tile into a canvas, by inverse mapping: each canvas pixel in the footprint's box looks up the tile pixel at its stage point.

    ``matrix`` is the **tile's** (the camera's times the reduction factor) and
    ``(x_um, y_um)`` the stage position of the tile's centre. For a canvas pixel at
    stage ``(x, y)`` the tile pixel is ``col = (W - 1) / 2 + d_col`` and ``row = (H -
    1) / 2 + d_row`` with ``d_col`` and ``d_row`` the inverse of the matrix written
    out, and not as a product on arrays (FM-131); the nearest tile pixel is taken when
    the point is inside the tile. The box is clamped to the canvas on floats first, so
    a frame far away places nothing and raises nothing.

    Raises:
        ValueError: The matrix cannot be inverted.
    """
    tile_h, tile_w = tile.shape
    (m00, m01), (m10, m11) = matrix
    det = m00 * m11 - m01 * m10
    if not abs(det) > 0.0:
        raise ValueError(f"the tile's matrix cannot be inverted (determinant {det!r})")
    n_rows, n_cols = canvas.shape
    corners = footprint(x_um, y_um, matrix=matrix, image_shape_px=(tile_h, tile_w))
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    col_lo = _clamped((min(xs) - x_min_um) / um_per_px - 1.0, n_cols)
    col_hi = _clamped((max(xs) - x_min_um) / um_per_px + 2.0, n_cols)
    row_lo = _clamped((y_max_um - max(ys)) / um_per_px - 1.0, n_rows)
    row_hi = _clamped((y_max_um - min(ys)) / um_per_px + 2.0, n_rows)
    if col_lo >= col_hi or row_lo >= row_hi:
        return
    cols = np.arange(col_lo, col_hi)
    rows = np.arange(row_lo, row_hi)
    dx = (x_min_um + (cols + 0.5) * um_per_px)[None, :] - x_um
    dy = (y_max_um - (rows + 0.5) * um_per_px)[:, None] - y_um
    col = (tile_w - 1) / 2.0 + (m11 * dx - m01 * dy) / det
    row = (tile_h - 1) / 2.0 + (m00 * dy - m10 * dx) / det
    inside = (col >= -0.5) & (col < tile_w - 0.5) & (row >= -0.5) & (row < tile_h - 0.5)
    source_col = np.clip(np.floor(col + 0.5), 0, tile_w - 1).astype(np.int64)
    source_row = np.clip(np.floor(row + 0.5), 0, tile_h - 1).astype(np.int64)
    target_rows = np.broadcast_to(rows[:, None], inside.shape)[inside]
    target_cols = np.broadcast_to(cols[None, :], inside.shape)[inside]
    canvas[target_rows, target_cols] = tile[source_row[inside], source_col[inside]]


def stretch_to_uint8(pixels: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Stretch a canvas to 8 bits between the 1st and the 99.5th percentile of the placed pixels.

    A pixel outside the two percentiles is clipped to 0 or 255 **before** the cast
    (FM-75: a float outside the type's range, or a ``nan``, is not defined by C and
    differs by platform), and a pixel no frame reached is dark grey.
    """
    grey = np.full(pixels.shape, _UNPLACED_GREY, dtype=np.uint8)
    placed = np.isfinite(pixels)
    if not placed.any():
        return grey
    values = pixels[placed].astype(np.float64)
    low, high = np.percentile(values, (1.0, 99.5))
    if not high > low:
        high = low + 1.0
    grey[placed] = np.clip(
        np.rint((values - low) / (high - low) * 255.0), 0.0, 255.0
    ).astype(np.uint8)
    return grey


def _outline_of(well: WellGeometry) -> list[tuple[float, float]]:
    """The well's wall as a closed polyline in stage um: a disc as 128 segments, a square well as its turned rectangle."""
    cx, cy = well.center_um
    if well.circular:
        radius = well.size_um[0] / 2.0
        points = [
            (
                cx + radius * math.cos(2.0 * math.pi * k / _CIRCLE_SEGMENTS),
                cy + radius * math.sin(2.0 * math.pi * k / _CIRCLE_SEGMENTS),
            )
            for k in range(_CIRCLE_SEGMENTS)
        ]
    else:
        angle = math.radians(well.rotation_deg)
        cos, sin = math.cos(angle), math.sin(angle)
        half_u, half_v = well.size_um[0] / 2.0, well.size_um[1] / 2.0
        points = [
            (cx + cos * u - sin * v, cy + sin * u + cos * v)
            for u, v in (
                (-half_u, -half_v),
                (half_u, -half_v),
                (half_u, half_v),
                (-half_u, half_v),
            )
        ]
    return [*points, points[0]]


def render_png(
    grey: npt.NDArray[np.uint8],
    canvas: Canvas,
    well: WellGeometry,
    *,
    cells: Sequence[Cell],
    classes: Sequence[str],
    wanted: Sequence[str],
    stops: Sequence[tuple[int, float, float]],
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    caption: str,
) -> Image.Image:
    """The picture of a well: the mosaic, the wall, the cells, the chosen fields numbered in route order.

    Blue is the wall; each cell is a 3-pixel dot coloured by its class (the classes in
    the cells stage's order), with a second, larger ring when its label is wanted; a
    chosen field is its footprint, a closed 2-pixel green line (turned or mirrored as
    the camera is) with its number at the centre; the route is a thin yellow line
    through the stops. ``stops`` are ``(number, x_um, y_um)`` in route order.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.stack([grey, grey, grey], axis=-1))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def at(x_um: float, y_um: float) -> tuple[float, float]:
        return canvas.px_of(x_um, y_um)

    draw.line([at(x, y) for x, y in _outline_of(well)], fill=_BLUE, width=1)
    colours = {
        label: _CLASS_COLOURS[k % len(_CLASS_COLOURS)]
        for k, label in enumerate(classes)
    }
    wanted_labels = set(wanted)
    for cell in cells:
        col, row = at(cell.x_um, cell.y_um)
        # A cell far outside the canvas (a corrupt record) is not drawn.
        if not (-10.0 <= col <= width + 10.0 and -10.0 <= row <= height + 10.0):
            continue
        c, r = round(col), round(row)
        colour = colours.get(cell.label, _WHITE)
        draw.ellipse((c - 1, r - 1, c + 1, r + 1), fill=colour)
        if cell.label in wanted_labels:
            draw.ellipse((c - 5, r - 5, c + 5, r + 5), outline=colour)
    if len(stops) > 1:
        draw.line([at(x, y) for _, x, y in stops], fill=_ROUTE, width=1)
    for _, x, y in stops:
        corners = [
            at(*corner)
            for corner in footprint(x, y, matrix=matrix, image_shape_px=image_shape_px)
        ]
        draw.line([*corners, corners[0]], fill=_GREEN, width=2)
    font = _font(max(12, round(max(width, height) / 80)))
    for number, x, y in stops:
        col, row = at(x, y)
        left, top, right, bottom = draw.textbbox((0, 0), str(number), font=font)
        draw.text(
            (col - (right - left) / 2.0 - left, row - (bottom - top) / 2.0 - top),
            str(number),
            fill=_GREEN,
            font=font,
        )
    draw.text((4, 4), caption, fill=_WHITE, font=font)
    return image


def _font(size: int) -> Any:
    """Pillow's default font at ``size`` pixels, so that a stop's number can be read on a picture of 1600 pixels.

    A size is accepted from Pillow 10.1 and needs FreeType; an older Pillow, or a build
    without it, gives the default 10-pixel font, which is small and still correct.
    """
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, ImportError):
        return ImageFont.load_default()


# ========================================================================= Record

#: The record's plugin name. Not ``route``: the document says who chose the fields, the
#: original ``route`` record (if any) stays in the copy, and ``provider("route")`` takes
#: the later start, which is this one.
RECORD_NAME = "route-prototype"
_READS = ("scan-plan", "scan", "cells")


@dataclass(frozen=True, slots=True)
class Options:
    """What a run was asked: the labels, the weights, the limits and the spacing that was used.

    ``spacing_given`` says whether ``min_spacing_um`` was typed or measured, and
    ``spacing_cells`` is the number of wanted cells it was measured over.
    """

    labels: tuple[str, ...]
    weights: Mapping[str, float]
    fields_per_well: int
    min_cells: int
    min_score: float
    wall_margin_um: float
    exclude_centre_um: float
    min_spacing_um: float
    spacing_given: bool
    spacing_cells: int


@dataclass(frozen=True, slots=True)
class WellChoice:
    """What was decided for one well: every candidate, its score, the fields chosen and what they leave.

    ``scored`` is in the candidates' order and ``chosen`` in the order chosen, which
    is the order the cells were given away in; ``counts`` is each chosen field's, as
    ``tally`` gives it. ``dropped`` counts the candidates that a pixel weight left out
    because their frame could not be read.
    """

    well: str
    candidates: tuple[Candidate, ...]
    skipped: Skipped
    scored: list[Scored]
    chosen: list[Scored]
    counts: list[dict[str, int]]
    uncovered: int
    dropped: int
    n_wanted: int


def select_well(
    well: str,
    scan: ScanResult,
    cells: Sequence[Cell],
    geometry: WellGeometry,
    *,
    classes: Sequence[str],
    options: Options,
    stats: Mapping[int, tuple[float | None, float] | None],
) -> WellChoice:
    """The candidates of a well, scored, chosen and counted: what one well of the command does, without the files."""
    candidates, skipped = candidates_of(
        well,
        scan,
        cells,
        geometry,
        labels=options.labels,
        classes=classes,
        min_spacing_um=options.min_spacing_um,
        margin_um=options.wall_margin_um,
        exclude_centre_um=options.exclude_centre_um,
        stats=stats,
    )
    scored = score_candidates(candidates, options.weights)
    chosen = choose(
        scored,
        matrix=scan.matrix_um_per_px,
        image_shape_px=scan.image_shape_px,
        n=options.fields_per_well,
        min_cells=options.min_cells,
        min_score=options.min_score,
    )
    wanted = [c for c in cells if c.well == well and c.label in options.labels]
    counts, uncovered = tally(
        [(s.candidate.x_um, s.candidate.y_um) for s in chosen],
        wanted,
        labels=options.labels,
        matrix=scan.matrix_um_per_px,
        image_shape_px=scan.image_shape_px,
    )
    return WellChoice(
        well=well,
        candidates=candidates,
        skipped=skipped,
        scored=scored,
        chosen=chosen,
        counts=counts,
        uncovered=uncovered,
        dropped=len(candidates) - len(scored),
        n_wanted=len(wanted),
    )


def first_field_of(plan: ScanPlanResult, well: str) -> tuple[float, float]:
    """The first planned field of a well, in the scan's visiting order: where the scan started in it.

    Raises:
        Refusal: The plan has no field for the well.
    """
    for field in plan.fields:
        if field.well == well:
            return (field.x_um, field.y_um)
    raise Refusal(f"the scan plan has no field for well {well!r}; run scan-plan again")


def assumed_lines(
    options: Options, *, unlocked: int, unreadable: int
) -> tuple[str, ...]:
    """What the result rests on and did not measure: the texts the plan fixed, in its order."""
    weights = ", ".join(f"{name}={options.weights[name]:g}" for name in WEIGHT_NAMES)
    if options.spacing_given:
        spacing = f"min_spacing_um={options.min_spacing_um:g}, given"
    else:
        spacing = (
            f"min_spacing_um={options.min_spacing_um:g}, measured as 3 times the "
            f"median cell length of {options.spacing_cells} wanted cells (the factor 3 "
            f"is chosen, not measured)"
        )
    lines = [
        "positions chosen by scripts/dev/select_fields.py, a prototype and not the "
        f"route plugin; score weights {weights}",
        "the candidate fields are the scan's own frames at their readback positions, "
        "not a lattice",
        spacing,
        "the route minimises the Euclidean distance between fields; whether the "
        "stage's move time follows it is not measured",
        "travel_s is not computed by the prototype",
    ]
    if unlocked:
        lines.append(
            f"{unlocked} frames were not candidates because the autofocus had not "
            f"locked on them (unlocked)"
        )
    if unreadable:
        lines.append(
            f"{unreadable} frames could not be read and were left out of the mosaic "
            f"and of the pixel features"
        )
    return tuple(lines)


def route_orders(
    plan: ScanPlanResult, choices: Sequence[WellChoice]
) -> list[list[int]]:
    """For each well, the order its chosen fields are visited in: indices into ``choice.chosen``.

    Each well starts at the field nearest where the stage is coming from, which is
    the last stop of the previous well that has one, or, for the first, the well's
    first planned field (where the scan started in it); then ``route``'s
    ``order_route``. A well with no chosen field has an empty order.
    """
    orders: list[list[int]] = []
    previous: tuple[float, float] | None = None
    for choice in choices:
        if not choice.chosen:
            orders.append([])
            continue
        reference = (
            previous if previous is not None else first_field_of(plan, choice.well)
        )
        order = order_stops(
            [(s.candidate.x_um, s.candidate.y_um) for s in choice.chosen], reference
        )
        orders.append(order)
        last = choice.chosen[order[-1]].candidate
        previous = (last.x_um, last.y_um)
    return orders


def route_result(
    scan: ScanResult,
    plan: ScanPlanResult,
    choices: Sequence[WellChoice],
    *,
    options: Options,
    unreadable: int = 0,
) -> RouteResult:
    """The chosen fields as a ``RouteResult``, in the shape ``nis-export`` reads.

    Each well's fields are ordered from the one nearest where the stage is coming
    from (the previous well's last stop, or the well's first planned field), then by
    ``route``'s ``order_route``. A stop is the **frame's** readback position, Z and
    autofocus offset, taken together from the one frame that was scored and will be
    imaged again (FM-167, FM-154), and its counts have every wanted label, zeros
    included. ``uncovered`` has every well, in the plan's order, zeros included.
    ``travel_s`` is ``None``: no timings are read, and unknown is not zero.
    """
    positions: list[RoutePosition] = []
    uncovered: dict[str, int] = {}
    for choice, order in zip(choices, route_orders(plan, choices), strict=True):
        uncovered[choice.well] = choice.uncovered
        for k, index in enumerate(order):
            candidate = choice.chosen[index].candidate
            positions.append(
                RoutePosition(
                    order=len(positions),
                    well=choice.well,
                    name=f"{choice.well}_r{k:03d}",
                    x_um=candidate.x_um,
                    y_um=candidate.y_um,
                    z_um=candidate.z_um,
                    pfs_offset=candidate.pfs_offset,
                    counts=dict(choice.counts[index]),
                )
            )
    points = [(p.x_um, p.y_um) for p in positions]
    return RouteResult(
        objective=scan.objective,
        matrix_um_per_px=scan.matrix_um_per_px,
        image_shape_px=scan.image_shape_px,
        fov_um=fov_of(scan.matrix_um_per_px, scan.image_shape_px),
        labels=options.labels,
        positions=tuple(positions),
        uncovered=uncovered,
        distance_um=path_length_um(points, range(len(points))),
        travel_s=None,
        assumed=assumed_lines(
            options,
            unlocked=sum(choice.skipped.unlocked for choice in choices),
            unreadable=unreadable,
        ),
    )


def record_params(
    options: Options, wells: Sequence[str], *, pixels: bool
) -> dict[str, Any]:
    """What the record says it was asked, as JSON values and never a path (FM-50)."""
    return {
        "labels": list(options.labels),
        "weights": {name: options.weights[name] for name in WEIGHT_NAMES},
        "fields_per_well": options.fields_per_well,
        "min_cells": options.min_cells,
        "min_score": options.min_score,
        "wall_margin_um": options.wall_margin_um,
        "exclude_centre_um": options.exclude_centre_um,
        "min_spacing_um": options.min_spacing_um,
        "min_spacing_source": "given" if options.spacing_given else "measured",
        "wells": list(wells),
        "pixels": pixels,
    }


def route_record(
    doc: RunDocument, result: RouteResult, *, params: dict[str, Any]
) -> StageRecord:
    """The ``route-prototype`` record that fills the ``route`` slot of a copy of ``doc``.

    It reads the current providers of ``scan-plan``, ``scan`` and ``cells``, so that
    it goes stale, as ``route``'s own record does, when one of them runs again; its
    profile and dry-run flag are the scan's, which is what ``nis-export`` compares
    with its own run.

    Raises:
        Refusal: The document has no provider of one of the three slots.
    """
    reads: dict[str, ReadRecord | None] = {}
    for slot in _READS:
        provider = doc.provider(slot)
        if provider is None:
            raise Refusal(
                f"the run document has no {slot}; smc run --list shows which plugin "
                f"provides it"
            )
        reads[slot] = ReadRecord(plugin=provider.plugin, attempt=provider.attempt)
    scan_record = doc.provider("scan")
    assert scan_record is not None  # `scan` is one of _READS: checked above
    now = datetime.now(timezone.utc)
    return StageRecord(
        plugin=RECORD_NAME,
        provides="route",
        attempt=uuid.uuid4().hex,
        status=StageStatus.done,
        started_utc=now,
        finished_utc=now,
        profile=scan_record.profile,
        dry_run=scan_record.dry_run,
        smc_version=smc.__version__,
        plugin_version=None,
        params=params,
        result=result.model_dump(mode="json"),
        reads=reads,
    )


def check_copy_target(source: Path, target: Path) -> None:
    """Refuse a copy that would be the original, a folder, or in a folder that is not there.

    Checked before any frame is read (FM-30): minutes of work must not end at a
    destination that was wrong from the start. ``samefile`` follows links, so a hard
    or symbolic link to the original is the original.

    Raises:
        Refusal: The target is a folder, the input itself, or has no folder to go in.
    """
    if target.is_dir():
        raise Refusal(f"--run-doc-out {target} is a folder, not a file; name a file")
    if target.exists() and source.samefile(target):
        raise Refusal(
            "--run-doc-out is the input run document: the original is never "
            "written, name another file"
        )
    if not target.parent.is_dir():
        raise Refusal(
            f"the folder of --run-doc-out ({target.parent}) does not exist; create "
            f"it or name another file"
        )


def write_copy(source: Path, target: Path, record: StageRecord) -> None:
    """Copy the run document and add ``record`` to the copy, which is replaced if it is there.

    The copy is derived (rebuilt from the original at each run), so replacing it
    loses nothing. Every other stage is kept, byte for byte in its content. A copy
    that could not be completed is removed: the original with no route nobody chose
    would otherwise read as a finished copy.

    Raises:
        OSError: The copy could not be made (a file another program holds open is a
            ``PermissionError`` on Windows).
        RunDocumentError: The record could not be written into the copy.
    """
    shutil.copyfile(source, target)
    try:
        write_stage(target, record)
    except RunDocumentError:
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
        raise


# ================================================================== Command line

_SELECTION_FILE = "selection.json"
_SCHEMA_VERSION = 1


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose errors are the command's: ``select_fields: <reason>`` and exit 2, not argparse's usage."""

    def error(self, message: str) -> NoReturn:
        raise Refusal(message)


def build_parser() -> argparse.ArgumentParser:
    """The command's options (ASCII only: a Windows console prints help in its code page)."""
    parser = _Parser(
        prog="select_fields.py",
        description=(
            "Choose the fields to image again from a scan's own frames, scored by "
            "weights you tune. A prototype (issue 343), not the route plugin."
        ),
    )
    parser.add_argument("run_doc", help="a run document with scan-plan, scan and cells")
    parser.add_argument(
        "--wells", help="comma separated wells to run (default: every scanned well)"
    )
    parser.add_argument(
        "--labels",
        default="single",
        help="comma separated wanted classes of the cells stage (default: single)",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            f"a weight in 0..{_MAX_WEIGHT:.0f}; NAME is one of "
            f"{', '.join(WEIGHT_NAMES)}; the default is density=1 and 0 for the rest"
        ),
    )
    parser.add_argument("--fields-per-well", type=int, default=10)
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--wall-margin-um", type=float, default=0.0)
    parser.add_argument("--exclude-centre-um", type=float, default=0.0)
    parser.add_argument(
        "--min-spacing-um",
        type=float,
        default=None,
        help="cells closer than this are not isolated (default: 3 median cell lengths)",
    )
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="do not read the frames: no sharpness, no background, no mosaic",
    )
    parser.add_argument(
        "--out-dir", default=None, help="default: selection/ next to the run document"
    )
    parser.add_argument(
        "--run-doc-out",
        default=None,
        help="default: <run document stem>-selected.json next to it",
    )
    parser.add_argument("--no-png", action="store_true", help="write no picture")
    parser.add_argument("--png-max-px", type=int, default=1600)
    parser.add_argument(
        "--show", type=int, default=15, help="candidates listed per well"
    )
    return parser


@dataclass(frozen=True, slots=True)
class Request:
    """What the command line asked, bounded and with its paths made: nothing is read yet."""

    run_doc: Path
    out_dir: Path
    copy_path: Path
    wells: tuple[str, ...] | None
    labels: tuple[str, ...]
    weights: dict[str, float]
    fields_per_well: int
    min_cells: int
    min_score: float
    wall_margin_um: float
    exclude_centre_um: float
    min_spacing_um: float | None
    pixels: bool
    png: bool
    png_max_px: int
    show: int


def _number(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:g}"


def _require_range(
    option: str,
    value: float,
    low: float,
    high: float,
    *,
    integer: bool = False,
    above: bool = False,
) -> None:
    """Refuse a value that is not finite or not in the range, in the words ``select_fields: <option> must be ...``."""
    in_range = (value > low if above else value >= low) and value <= high
    if not (math.isfinite(value) and in_range):
        kind = "an integer" if integer else "a number"
        where = (
            f"above {_number(low)} and at most {_number(high)}"
            if above
            else f"from {_number(low)} to {_number(high)}"
        )
        raise Refusal(f"{option} must be {kind} {where}")


def _names(text: str, option: str) -> tuple[str, ...]:
    """The comma separated names of an option, without blanks and in order."""
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names:
        raise Refusal(f"{option} must name at least one value")
    return names


def parse_request(argv: Sequence[str] | None) -> Request:
    """Read and bound the command line (FM-32): ``nan``, ``inf``, negative and huge values never reach a file.

    Raises:
        Refusal: An option is malformed or out of bounds, a weight is refused, or a
            pixel weight is asked of a run that reads no pixel.
    """
    args = build_parser().parse_args(argv)
    _require_range("--fields-per-well", args.fields_per_well, 1, 500, integer=True)
    _require_range("--min-cells", args.min_cells, 1, 1e9, integer=True)
    _require_range("--min-score", args.min_score, 0.0, 1.0)
    _require_range("--wall-margin-um", args.wall_margin_um, 0.0, 1e6)
    _require_range("--exclude-centre-um", args.exclude_centre_um, 0.0, 1e6)
    _require_range("--png-max-px", args.png_max_px, 256, 8000, integer=True)
    _require_range("--show", args.show, 1, 500, integer=True)
    if args.min_spacing_um is not None:
        _require_range("--min-spacing-um", args.min_spacing_um, 0.0, 1e5, above=True)
    weights = parse_weights(args.weight)
    if args.no_pixels:
        for name in ("sharpness", "background"):
            if weights[name] > 0.0:
                raise Refusal(
                    f"--weight {name} needs the frames' pixels, and --no-pixels reads "
                    f"none: drop --no-pixels or set {name}=0"
                )
    labels = _names(args.labels, "--labels")
    for label in labels:
        if labels.count(label) > 1:
            raise Refusal(f"--labels repeats {label!r}")
    wells = (
        tuple(normalise_well(name) for name in _names(args.wells, "--wells"))
        if args.wells is not None
        else None
    )
    run_doc = Path(args.run_doc)
    return Request(
        run_doc=run_doc,
        out_dir=(
            Path(args.out_dir)
            if args.out_dir is not None
            else run_doc.parent / "selection"
        ),
        copy_path=(
            Path(args.run_doc_out)
            if args.run_doc_out is not None
            else run_doc.with_name(f"{run_doc.stem}-selected.json")
        ),
        wells=wells,
        labels=labels,
        weights=weights,
        fields_per_well=args.fields_per_well,
        min_cells=args.min_cells,
        min_score=args.min_score,
        wall_margin_um=args.wall_margin_um,
        exclude_centre_um=args.exclude_centre_um,
        min_spacing_um=args.min_spacing_um,
        pixels=not args.no_pixels,
        png=not args.no_png,
        png_max_px=args.png_max_px,
        show=args.show,
    )


@dataclass(frozen=True, slots=True)
class Upstream:
    """The three records the command reads, validated, and the cells they name."""

    plan: ScanPlanResult
    scan: ScanResult
    cells_result: CellsResult
    cells: tuple[Cell, ...]
    profile: str


def _result_of(doc: RunDocument, slot: str, model: type[_M]) -> _M:
    """The result of the record that fills ``slot``, validated; a record that does not validate says which to run again."""
    record = doc.provider(slot)
    assert record is not None  # `unusable` has been asked first
    problem = ""
    parsed: _M | None = None
    try:
        parsed = model.model_validate(record.result)
    except ValidationError as exc:
        first = exc.errors()[0]
        place = ".".join(str(part) for part in first["loc"]) or slot
        problem = (
            f"the {slot} record does not validate ({place}: {first['msg']}); run "
            f"{record.plugin} again"
        )
    if parsed is None:
        raise Refusal(problem)
    return parsed


def load_upstream(doc: RunDocument, run_doc: Path, labels: Sequence[str]) -> Upstream:
    """``scan-plan``, ``scan`` and ``cells`` of a run document, each usable for the scan's profile, and the cells file.

    Raises:
        Refusal: A slot is missing, did not end ``done``, is stale or was run on another
            profile (the reason of ``RunDocument.unusable``), does not validate, the
            cells file no longer matches, a label is not one of the cells' classes, or
            the camera cannot be inverted.
    """
    scan_record = doc.provider("scan")
    if scan_record is None:
        raise Refusal(
            "the run document has no scan; smc run --list shows which plugin "
            "provides it"
        )
    for slot in _READS:
        reason = doc.unusable(
            slot, profile=scan_record.profile, dry_run=scan_record.dry_run
        )
        if reason is not None:
            raise Refusal(reason)
    plan = _result_of(doc, "scan-plan", ScanPlanResult)
    scan = _result_of(doc, "scan", ScanResult)
    cells_result = _result_of(doc, "cells", CellsResult)
    unknown = [label for label in labels if label not in cells_result.classes]
    if unknown:
        raise Refusal(
            f"label {unknown[0]!r} is not one of the cells' classes "
            f"{list(cells_result.classes)}; give --labels among them"
        )
    try:
        _inverse_of(scan.matrix_um_per_px)
    except ValueError as exc:
        raise Refusal(str(exc)) from exc
    if min(scan.image_shape_px) < 1:
        raise Refusal(f"the scan's frame shape {scan.image_shape_px} is empty")
    try:
        cells = load_cells(run_doc.parent, cells_result)
    except ValueError as exc:
        raise Refusal(f"{exc}; run cells again") from exc
    return Upstream(plan, scan, cells_result, cells, scan_record.profile)


def wells_to_run(
    plan: ScanPlanResult, cells_result: CellsResult, requested: Sequence[str] | None
) -> list[str]:
    """The wells the cells stage scanned, in the plan's order, or the ones asked for among them.

    Raises:
        Refusal: The cells hold a well the plan has no field for or does not list (as
            ``route`` refuses it), a well asked for was not scanned, or none is left.
    """
    planned = {field.well for field in plan.fields}
    listed = set(plan.wells)
    for well in cells_result.counts:
        if well not in planned:
            raise Refusal(
                f"the cells hold well {well!r}, which the scan plan has no field for; "
                f"run scan-plan, scan and cells again"
            )
        if well not in listed:
            raise Refusal(
                f"the cells hold well {well!r}, which the scan plan does not list "
                f"among its wells; run scan-plan, scan and cells again"
            )
    scanned = [well for well in plan.wells if well in cells_result.counts]
    if requested is not None:
        missing = [well for well in requested if well not in scanned]
        if missing:
            raise Refusal(
                f"well {missing[0]!r} was not scanned by the cells stage; the wells "
                f"are {', '.join(scanned) or 'none'}"
            )
        scanned = [well for well in scanned if well in requested]
    if not scanned:
        raise Refusal("the cells stage scanned no well: nothing to select")
    return scanned


@dataclass(slots=True)
class Pixels:
    """What reading the frames gave: each frame's statistics, each well's mosaic, and what could not be read."""

    stats: dict[str, dict[int, tuple[float | None, float] | None]]
    grey: dict[str, npt.NDArray[np.uint8]]
    unreadable: int = 0
    first_problem: str | None = None


def read_pixels(
    run_dir: Path,
    scan: ScanResult,
    geometries: Mapping[str, WellGeometry],
    *,
    mosaic: bool,
    png_max_px: int,
) -> Pixels:
    """Read each frame of each well **once**: its statistics and, when ``mosaic``, its tile, in the same pass.

    A frame that cannot be read (missing, corrupt, outside the run folder, another
    shape than the scan says, not finite) is counted and skipped, and never ends the
    run (failures are findings). A frame is read, reduced and dropped: only the
    canvas of the well being read is held.

    Raises:
        Refusal: The scan's pixel size is not a size (FM-104).
    """
    frames_of = scan.frame_index()
    height, width = scan.image_shape_px
    matrix = scan.matrix_um_per_px
    found = Pixels(stats={}, grey={})
    for well, geometry in geometries.items():
        stats: dict[int, tuple[float | None, float] | None] = {}
        canvas = canvas_of(geometry, png_max_px) if mosaic else None
        pixels = canvas.blank() if canvas is not None else None
        reduction = 1
        if canvas is not None:
            try:
                reduction = reduction_factor(
                    canvas.um_per_px, scan.pixel_size_um, scan.image_shape_px
                )
            except ValueError as exc:
                raise Refusal(f"the scan's {exc}") from exc
        tile_matrix: Matrix2 = (
            (matrix[0][0] * reduction, matrix[0][1] * reduction),
            (matrix[1][0] * reduction, matrix[1][1] * reduction),
        )
        for position in frames_of.get(well, []):
            frame = scan.frames[position]
            tile = None
            problem = ""
            try:
                raw = load_frame(run_dir, frame)
                if tuple(raw.shape) != (height, width):
                    raise ValueError(
                        f"{frame.name} is {raw.shape[-1]} x {raw.shape[0]} pixels, "
                        f"the scan says {width} x {height}"
                    )
                stats[position] = frame_stats(raw)
                if pixels is not None:
                    tile = block_mean(raw, reduction)
            except Exception as exc:
                problem = f"{frame.file}: {exc}"
            if problem:
                stats[position] = None
                found.unreadable += 1
                found.first_problem = found.first_problem or problem
                continue
            if pixels is not None and tile is not None and canvas is not None:
                place_frame(
                    pixels,
                    tile,
                    frame.x_um,
                    frame.y_um,
                    matrix=tile_matrix,
                    x_min_um=canvas.x_min_um,
                    y_max_um=canvas.y_max_um,
                    um_per_px=canvas.um_per_px,
                )
        found.stats[well] = stats
        if pixels is not None:
            found.grey[well] = stretch_to_uint8(pixels)
    return found


def _ascii(text: str) -> str:
    """``text`` with everything that is not ASCII written as an escape: a Windows console decodes with a code page (FM-20)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _say(line: str = "") -> None:
    print(_ascii(line))


def _finite(value: float | None) -> float | None:
    """``value``, or ``None`` when it is unknown or not a finite number: JSON has no ``nan``."""
    return value if value is not None and math.isfinite(value) else None


def _cell(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def table_lines(
    *,
    run_name: str,
    upstream: Upstream,
    request: Request,
    options: Options,
    choices: Sequence[WellChoice],
    orders: Sequence[Sequence[int]],
    result: RouteResult,
    pixels: Pixels | None,
) -> list[str]:
    """The table: the settings, then for each well its best candidates by score, its chosen fields in route order and what they leave."""
    scan = upstream.scan
    height, width = scan.image_shape_px
    fov = result.fov_um
    skipped_unlocked = sum(c.skipped.unlocked for c in choices)
    skipped_refused = sum(c.skipped.refused for c in choices)
    candidates = sum(len(c.candidates) for c in choices)
    dropped = sum(c.dropped for c in choices)
    weights = " ".join(f"{name}={options.weights[name]:g}" for name in WEIGHT_NAMES)
    lines = [
        f"select_fields: {run_name} (a prototype, not the route plugin)",
        f"  profile {upstream.profile}, objective {scan.objective or 'unknown'}, "
        f"pixel {scan.pixel_size_um:g} um, frame {width} x {height} px = "
        f"{fov[0]:.0f} x {fov[1]:.0f} um (field)",
        f"  labels {', '.join(options.labels)}; weights {weights}",
        f"  min_spacing_um {options.min_spacing_um:g} "
        f"({'given' if options.spacing_given else 'measured, 3 times the median cell length of ' + str(options.spacing_cells) + ' wanted cells; the factor 3 is chosen, not measured'})",
        f"  at most {options.fields_per_well} fields per well, min cells "
        f"{options.min_cells}, min score {options.min_score:g}, wall margin "
        f"{options.wall_margin_um:g} um, excluded centre {options.exclude_centre_um:g} um",
        f"  {candidates} candidates ({skipped_unlocked} frames unlocked and "
        f"{skipped_refused} over the wall or the excluded centre left out"
        + (f", {dropped} dropped for a pixel weight" if dropped else "")
        + ")"
        + (
            f"; {pixels.unreadable} frames could not be read"
            + (
                f" (first: {pixels.first_problem})"
                if pixels.first_problem is not None
                else ""
            )
            if pixels is not None and pixels.unreadable
            else ""
        ),
    ]
    columns = (
        "rank",
        "frame",
        "x_um",
        "y_um",
        "score",
        "n",
        "dens/mm2",
        "iso",
        "purity",
        "nn_um",
        "wall_um",
        "sharp",
        "bg",
        "focus",
        "stop",
    )
    for choice, order in zip(choices, orders, strict=True):
        stop_of = {choice.chosen[i].candidate.frame: k for k, i in enumerate(order)}
        lines.append("")
        lines.append(
            f"== {choice.well}: {len(choice.chosen)} fields chosen of "
            f"{len(choice.candidates)} candidates, {choice.n_wanted} wanted cells"
        )
        if choice.n_wanted == 0:
            lines.append("   no wanted cell in this well: no field chosen")
            continue
        if not choice.scored:
            lines.append("   no candidate to rank (see the counts above)")
            continue
        ranked = sorted(choice.scored, key=lambda s: (-s.score, s.candidate.frame))
        rows = [list(columns)]
        for rank, s in enumerate(ranked[: request.show], start=1):
            c = s.candidate
            rows.append(
                [
                    str(rank),
                    str(c.frame),
                    f"{c.x_um:.1f}",
                    f"{c.y_um:.1f}",
                    f"{s.score:.3f}",
                    str(c.n_wanted),
                    f"{c.density_per_mm2:.1f}",
                    str(c.isolated),
                    f"{c.n_wanted / c.n_all:.2f}" if c.n_all else "-",
                    _cell(c.nn_median_um, ".1f"),
                    f"{c.wall_um:.0f}",
                    _cell(c.sharpness, ".3g"),
                    _cell(c.background, ".1f"),
                    c.focus,
                    str(stop_of[c.frame]) if c.frame in stop_of else "-",
                ]
            )
        widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
        lines.extend(
            "   "
            + "  ".join(
                cell.rjust(width) for cell, width in zip(row, widths, strict=True)
            )
            for row in rows
        )
        if len(ranked) > request.show:
            lines.append(f"   ... {len(ranked) - request.show} more in selection.json")
        lines.append("   chosen, in route order:")
        for k, i in enumerate(order):
            c = choice.chosen[i].candidate
            counts = " ".join(f"{label}={n}" for label, n in choice.counts[i].items())
            lines.append(
                f"     r{k:03d} frame {c.frame} {c.name}  x={c.x_um:.1f} y={c.y_um:.1f} "
                f"z={_cell(c.z_um, '.1f')} pfs={_cell(c.pfs_offset, 'g')}  {counts}"
            )
        lines.append(
            f"   uncovered: {choice.uncovered} of {choice.n_wanted} wanted cells"
        )
    lines.append("")
    lines.append(
        f"route: {len(result.positions)} stops, {result.distance_um:.0f} um between "
        f"fields (travel time not computed)"
    )
    return lines


def selection_document(
    *,
    run_name: str,
    request: Request,
    options: Options,
    choices: Sequence[WellChoice],
    orders: Sequence[Sequence[int]],
    pixels: bool,
) -> dict[str, Any]:
    """``selection.json``: every candidate of every well with its features, score, parts and stop, for the owner's notebook.

    Nothing in smc reads it. The run document is named by its file name, never its
    path (FM-50), and no value is ``nan``: an unknown is ``null``.
    """
    wells: dict[str, Any] = {}
    for choice, order in zip(choices, orders, strict=True):
        stop_of = {choice.chosen[i].candidate.frame: k for k, i in enumerate(order)}
        scored = {s.candidate.frame: s for s in choice.scored}
        entries = []
        for c in choice.candidates:
            s = scored.get(c.frame)
            entries.append(
                {
                    "frame": c.frame,
                    "name": c.name,
                    "x_um": c.x_um,
                    "y_um": c.y_um,
                    "z_um": c.z_um,
                    "pfs_offset": c.pfs_offset,
                    "focus": c.focus,
                    "counts": c.counts,
                    "n_all": c.n_all,
                    "n_wanted": c.n_wanted,
                    "density_per_mm2": _finite(c.density_per_mm2),
                    "isolated": c.isolated,
                    "nn_median_um": _finite(c.nn_median_um),
                    "nn_min_um": _finite(c.nn_min_um),
                    "wall_um": _finite(c.wall_um),
                    "sharpness": _finite(c.sharpness),
                    "background": _finite(c.background),
                    "score": s.score if s is not None else None,
                    "parts": s.parts if s is not None else None,
                    "stop": stop_of.get(c.frame),
                }
            )
        wells[choice.well] = {"candidates": entries, "uncovered": choice.uncovered}
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_doc": run_name,
        "labels": list(options.labels),
        "weights": {name: options.weights[name] for name in WEIGHT_NAMES},
        "options": {
            "fields_per_well": options.fields_per_well,
            "min_cells": options.min_cells,
            "min_score": options.min_score,
            "wall_margin_um": options.wall_margin_um,
            "exclude_centre_um": options.exclude_centre_um,
            "pixels": pixels,
        },
        "min_spacing_um": options.min_spacing_um,
        "min_spacing_source": "given" if options.spacing_given else "measured",
        "wells": wells,
    }


def write_selection(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``selection.json``: ASCII JSON (``\\uXXXX`` escapes), UTF-8, and never a ``nan``."""
    text = json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_picture(image: Image.Image, path: Path) -> None:
    """Write one well's picture; an existing one is replaced, it is derived."""
    image.save(path, "PNG")


def run(argv: Sequence[str] | None) -> int:
    """The command: validate, read, choose, print the table, then write the files.

    The order is the plan's (decided in #343): options and destinations are checked
    before any file is read, the frames are read once, the table is printed **before**
    any file is written (a write that fails must not hide minutes of work: FM-30,
    FM-34), and a file that cannot be written is a line on stderr and an exit of 1,
    with the other files still written.

    Returns:
        0 when everything was done, 1 when a file could not be written, 2 on a refusal
        (raised as ``Refusal`` for ``main`` to print).
    """
    request = parse_request(argv)
    run_doc = request.run_doc
    try:
        doc = RunDocument.load(run_doc)
    except RunDocumentError as exc:
        raise Refusal(f"{run_doc.name}: {exc.reason}") from exc
    check_copy_target(run_doc, request.copy_path)
    if request.out_dir.exists() and not request.out_dir.is_dir():
        raise Refusal(f"--out-dir {request.out_dir} is a file, not a folder")
    upstream = load_upstream(doc, run_doc, request.labels)
    plan, scan = upstream.plan, upstream.scan
    wells = wells_to_run(plan, upstream.cells_result, request.wells)
    geometries: dict[str, WellGeometry] = {}
    for well in wells:
        try:
            geometries[well] = well_geometry(plan.plate_plan, well)
        except ValueError as exc:
            raise Refusal(f"{exc}; run scan-plan again") from exc

    wanted_cells = [
        c for c in upstream.cells if c.well in geometries and c.label in request.labels
    ]
    if request.min_spacing_um is not None:
        spacing, given = request.min_spacing_um, True
    else:
        measured = default_min_spacing_um(wanted_cells, scan.matrix_um_per_px)
        if measured is None:
            raise Refusal(
                f"no wanted cell ({', '.join(request.labels)}) in the wells run: the "
                f"spacing cannot be measured; give --min-spacing-um or other --labels"
            )
        if not 0.0 < measured <= 1e5:
            raise Refusal(
                f"the measured min_spacing_um is {measured:g}, which is not above 0 "
                f"and at most 100000; give --min-spacing-um"
            )
        spacing, given = measured, False
    options = Options(
        labels=request.labels,
        weights=request.weights,
        fields_per_well=request.fields_per_well,
        min_cells=request.min_cells,
        min_score=request.min_score,
        wall_margin_um=request.wall_margin_um,
        exclude_centre_um=request.exclude_centre_um,
        min_spacing_um=spacing,
        spacing_given=given,
        spacing_cells=len(wanted_cells),
    )

    pixels: Pixels | None = None
    if request.pixels:
        pixels = read_pixels(
            run_doc.parent,
            scan,
            geometries,
            mosaic=request.png,
            png_max_px=request.png_max_px,
        )
    choices = [
        select_well(
            well,
            scan,
            upstream.cells,
            geometries[well],
            classes=upstream.cells_result.classes,
            options=options,
            stats=pixels.stats.get(well, {}) if pixels is not None else {},
        )
        for well in wells
    ]
    orders = route_orders(plan, choices)
    result = route_result(
        scan,
        plan,
        choices,
        options=options,
        unreadable=pixels.unreadable if pixels is not None else 0,
    )
    for line in table_lines(
        run_name=run_doc.name,
        upstream=upstream,
        request=request,
        options=options,
        choices=choices,
        orders=orders,
        result=result,
        pixels=pixels,
    ):
        _say(line)

    failed = _write_files(
        request, upstream, doc, options, choices, orders, result, pixels, wells
    )
    return 1 if failed else 0


def _write_files(
    request: Request,
    upstream: Upstream,
    doc: RunDocument,
    options: Options,
    choices: Sequence[WellChoice],
    orders: Sequence[Sequence[int]],
    result: RouteResult,
    pixels: Pixels | None,
    wells: Sequence[str],
) -> bool:
    """Write ``selection.json``, one picture per well and the copy; say each and return whether any failed.

    A failure is one line on stderr and the others are still written (FM-34).
    """
    failed = False

    def fail(what: Path, exc: Exception) -> None:
        nonlocal failed
        failed = True
        reason = getattr(exc, "strerror", None) or getattr(exc, "reason", None) or exc
        print(
            _ascii(f"select_fields: could not write {what}: {reason}"), file=sys.stderr
        )

    try:
        request.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail(request.out_dir, exc)
        return True
    target = request.out_dir / _SELECTION_FILE
    try:
        write_selection(
            target,
            selection_document(
                run_name=request.run_doc.name,
                request=request,
                options=options,
                choices=choices,
                orders=orders,
                pixels=request.pixels,
            ),
        )
        _say(f"wrote {target}")
    except OSError as exc:
        fail(target, exc)

    scan = upstream.scan
    if request.png:
        weights = " ".join(
            f"{name}={options.weights[name]:g}"
            for name in WEIGHT_NAMES
            if options.weights[name] > 0.0
        )
        for choice, order in zip(choices, orders, strict=True):
            geometry = well_geometry(upstream.plan.plate_plan, choice.well)
            canvas = canvas_of(geometry, request.png_max_px)
            grey = pixels.grey.get(choice.well) if pixels is not None else None
            if grey is None:
                grey = np.full(
                    (canvas.height_px, canvas.width_px), _UNPLACED_GREY, dtype=np.uint8
                )
            image = render_png(
                grey,
                canvas,
                geometry,
                cells=[c for c in upstream.cells if c.well == choice.well],
                classes=upstream.cells_result.classes,
                wanted=options.labels,
                stops=[
                    (
                        k,
                        choice.chosen[i].candidate.x_um,
                        choice.chosen[i].candidate.y_um,
                    )
                    for k, i in enumerate(order)
                ],
                matrix=scan.matrix_um_per_px,
                image_shape_px=scan.image_shape_px,
                caption=f"{choice.well}: {len(order)} fields, {weights}",
            )
            path = request.out_dir / f"{choice.well}.png"
            try:
                write_picture(image, path)
                _say(f"wrote {path}")
            except OSError as exc:
                fail(path, exc)

    if result.positions:
        try:
            record = route_record(
                doc,
                result,
                params=record_params(options, wells, pixels=request.pixels),
            )
            write_copy(request.run_doc, request.copy_path, record)
            _say(
                f"wrote {request.copy_path} (export it with: smc run nis-export "
                f"--run-doc {request.copy_path})"
            )
        except (OSError, RunDocumentError) as exc:
            fail(request.copy_path, exc)
    else:
        _say("no field was chosen in any well: no run document copy written")
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; a refusal is one line on stderr, ``select_fields: <reason>``, and the exit code 2."""
    try:
        return run(argv)
    except Refusal as exc:
        print(_ascii(f"select_fields: {exc}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
