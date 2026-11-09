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

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# The route's own arithmetic for "inside a field" and for "overlap" (FM-132): private
# helpers, imported on purpose so that the prototype and the plugin cannot disagree.
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.route.algorithm import _inverse_of, _pixel_offsets, allowed
from smc.plugins.slots import Cell, ScanResult, WellGeometry

__all__ = ["Candidate", "Refusal", "Skipped"]

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
