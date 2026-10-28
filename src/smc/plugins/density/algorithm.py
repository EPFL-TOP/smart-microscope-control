"""The area that was imaged, and the cells per label and mm² of each well (m3 §10).

Pure arithmetic on stage µm: no stage, no camera, no file. The plugin hands in
the footprints of the frames the scan took, as corners, and the wells' shapes;
what a well's imaged area is, is decided here once.

The area is the union of the footprints clipped to the well, counted on a grid
of cell centres ``(x_lo + (i + 0.5) g, y_lo + (j + 0.5) g)``: a point counts
when it is inside at least one footprint and inside the well. Measured on a
prototype (numpy 2.5.3 on Python 3.13, 1.26.0 and 2.2.6 on Python 3.10, the
same areas on the three): a 96-well disc covered by 305 fields at 5 µm gives
32.1703 mm² for π r² = 32.1699 (+0.001 %) in 0.03 s. A single field has the
error of its edges, up to ``g / (2 w)`` per axis (+1.3 % for a 333 µm field at
5 µm), so the grid is fine for a well and rough for one field.

The counting is the other half: a cell is in a well when the well's shape says
so (the one place that knows it, ``WellGeometry``), and a cell on the plastic is
``outside_well`` and in no label, so the counts of a well always add up to its
cells. The density divides by the imaged area only: what a failed field would
have shown is in neither the numerator nor the denominator.

Numeric choices made on measurements, not on taste:

* **The grid is cut to the well, not to the data.** A frame read back at a wrong
  stage position (1e9 µm) would make a grid as large as the data. The extent is
  the footprints' bounding box cut to the well's circumscribed square, so a
  bound on the grid is a bound on the well, and a footprint far away leaves the
  grid and the area alone.
* **The bounds are on floats first.** ``400.0 / 5e-324`` is ``inf`` and
  ``math.ceil(inf)`` is an ``OverflowError``, so a size is refused before it
  becomes an integer, written ``not x <= bound`` so that ``nan`` is refused too
  (FM-74). The sizes in a message are formatted with ``:.3g``: a step of 1e-300
  would print a number of 300 digits.
* **A window index is clipped on the floats before it is cast.** A float far
  outside the integer range is an undefined conversion (FM-75), and a footprint
  1e11 µm away is legal.
* **No matrix product on coordinate arrays** (FM-131): the entries of the
  inverse are written out.
* **Bands of 2**20 elements.** A band of 2**24 peaked at 689 MB (a 96-well well
  at 0.5 µm), one of 2**20 at 95 MB, in the same time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from smc.plugins.slots import Cell, DensityWell, WellGeometry

__all__ = ["GridTooFineError", "density", "scanned_area_mm2"]

#: A grid never holds more points than this on a side: the bound that keeps
#: ``math.ceil`` of the float count a Python int of a size that can be allocated.
_MAX_SIDE_POINTS = 2**17
#: ...nor in all. Memory does not grow with it (a band is at most 2**20
#: elements); the time does, and 2**30 points are a 6.4 mm well at 0.2 µm.
_MAX_GRID_POINTS = 2**30
#: Footprint-window by grid-point tests one area may make. Measured 440 to 650
#: million a second on the three numpys, so 3 to 5 s at the bound on a laptop,
#: more on a slower PC. A refusal threshold, not a promise: nothing looks at a
#: stop inside one well.
_MAX_TESTS = 2**31
#: Elements of one band of the raster (rows by columns): 2**20 peaked at 95 MB.
_BAND_ELEMENTS = 2**20
#: Corner 2 of a footprint is ``c1 + c3 - c0`` to this, plus a few ulps of the
#: coordinates. Each corner of ``geometry.footprint`` is the position plus one
#: rounded term per axis of the frame, so within one binade the four are a
#: parallelogram to the last bit (60 000 random frames far from the well, none
#: beyond 1e-6 µm). A footprint that straddles a power of two is the exception:
#: the spacing of the floats doubles inside it, and corner 2 is off by up to
#: 2 ulps (3.8e-6 µm at 2**33 µm, 6e-5 µm at 2**39, measured), which a frame read
#: back far away must survive, since a far frame is ignored and not refused.
_PARALLELOGRAM_TOL_UM = 1e-6
#: The ulps of the largest coordinate that are added to the tolerance: the four
#: roundings of the corners and the two of the sum that checks them come to at most
#: about 4, and 8 is the margin.
_PARALLELOGRAM_ULPS = 8
#: A corner beyond this (1000 km of stage) is a corrupt record, and below it the
#: arithmetic of the raster cannot overflow.
_MAX_COORDINATE_UM = 1e12


class GridTooFineError(ValueError):
    """The grid, or the counting on it, would be larger than the bounds allow.

    The plugin turns it into a refusal that names the well and says to raise
    ``area_grid_um``: a step of 1e-9 µm is a legal parameter, and what it would
    allocate is not.
    """


@dataclass(frozen=True, slots=True)
class _Parallelogram:
    """A footprint by its corner 0 and the edges to corners 1 and 3, and its bounding box."""

    c0x: float
    c0y: float
    e1x: float
    e1y: float
    e3x: float
    e3y: float
    det: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def scanned_area_mm2(
    footprints: Sequence[Sequence[tuple[float, float]]],
    well: WellGeometry,
    *,
    grid_um: float,
) -> float:
    """The area of the well that the footprints cover, in mm², each point once.

    A footprint is four corners in ``geometry.footprint``'s order, a
    parallelogram given by its corners 0, 1 and 3. The area is the number of
    grid points inside at least one footprint **and** inside the well, times
    ``grid_um`` squared: overlapping fields are counted once, and a field that
    straddles the wall counts only what is inside it. ``well.contains_many``
    decides membership, so the raster agrees with ``contains`` on the edge.

    Args:
        footprints: The corners of every frame the scan took, in stage µm.
        well: The well whose area it is; its walls clip the footprints.
        grid_um: The step of the grid, in µm; an edge is known to half of it.

    Returns:
        The covered area in mm²; ``0.0`` for no footprint, or none that meets
        the well.

    Raises:
        GridTooFineError: The grid over the well's imaged area, or the counting
            on it, would be larger than the bounds allow.
        ValueError: ``grid_um`` is not finite and positive, or a footprint is
            not four finite corners of a parallelogram of some area within
            1e12 µm of the origin. The message names the first one, from 0.
    """
    if not (math.isfinite(grid_um) and grid_um > 0.0):
        raise ValueError(f"grid_um must be finite and positive, got {grid_um!r}")
    if len(footprints) == 0:
        return 0.0
    shapes = [_parallelogram(k, corners) for k, corners in enumerate(footprints)]

    # The well's circumscribed square, whatever the plate's turn: half the width
    # of a disc, the half-diagonal of a rectangle. An extent, not a membership:
    # `contains_many` decides that.
    centre_x, centre_y = well.center_um
    half_um = (
        well.size_um[0] / 2.0
        if well.circular
        else math.hypot(well.size_um[0] / 2.0, well.size_um[1] / 2.0)
    )
    x_lo = max(min(p.xmin for p in shapes), centre_x - half_um)
    x_hi = min(max(p.xmax for p in shapes), centre_x + half_um)
    y_lo = max(min(p.ymin for p in shapes), centre_y - half_um)
    y_hi = min(max(p.ymax for p in shapes), centre_y + half_um)
    if not (x_hi > x_lo and y_hi > y_lo):
        return 0.0

    nx, ny = _grid_size(well, x_hi - x_lo, y_hi - y_lo, grid_um)
    xmin = np.array([p.xmin for p in shapes], dtype=np.float64)
    xmax = np.array([p.xmax for p in shapes], dtype=np.float64)
    ymin = np.array([p.ymin for p in shapes], dtype=np.float64)
    ymax = np.array([p.ymax for p in shapes], dtype=np.float64)
    i0 = _index(np.floor((xmin - x_lo) / grid_um - 0.5), nx)
    i1 = _index(np.floor((xmax - x_lo) / grid_um - 0.5) + 2.0, nx)
    j0 = _index(np.floor((ymin - y_lo) / grid_um - 0.5), ny)
    j1 = _index(np.floor((ymax - y_lo) / grid_um - 0.5) + 2.0, ny)

    tests = int(np.sum((i1 - i0) * (j1 - j0)))
    if tests > _MAX_TESTS:
        raise GridTooFineError(
            f"well {well.well}: counting its fields on a grid of {grid_um:g} µm "
            f"would make {tests} tests, more than {_MAX_TESTS}"
        )

    xs = x_lo + (np.arange(nx, dtype=np.float64) + 0.5) * grid_um
    rows = max(1, _BAND_ELEMENTS // nx)
    covered = 0
    for start in range(0, ny, rows):
        stop = min(ny, start + rows)
        ys = y_lo + (np.arange(start, stop, dtype=np.float64) + 0.5) * grid_um
        band = np.zeros((stop - start, nx), dtype=np.bool_)
        meets = (i1 > i0) & (j1 > j0) & (j0 < stop) & (j1 > start)
        for k in np.nonzero(meets)[0]:
            a, b = max(int(j0[k]), start), min(int(j1[k]), stop)
            window = band[a - start : b - start, int(i0[k]) : int(i1[k])]
            window |= _inside(
                shapes[k], xs[int(i0[k]) : int(i1[k])], ys[a - start : b - start]
            )
        band &= well.contains_many(xs[None, :], ys[:, None])
        covered += int(band.sum())
    return covered * grid_um * grid_um / 1e6


def density(
    cells: Sequence[Cell],
    *,
    wells: Sequence[WellGeometry],
    areas_mm2: Mapping[str, float],
    fields: Mapping[str, int],
    failed_fields: Mapping[str, int],
    labels: Sequence[str],
) -> tuple[DensityWell, ...]:
    """The cells of each well counted per label, and divided by the area that was imaged.

    A cell is counted when ``well.contains_many`` says it is inside its well's
    shape, the edge included, as ``contains`` does: one outside (a field's
    corner on the plastic) is ``outside_well`` and in no label. So the counts
    and ``outside_well`` of a well always add up to its cells. Every label is a
    key of ``counts``, zeros included, in ``labels`` order. ``per_mm2`` divides
    by the imaged area only, and is ``None`` for an area of 0: with no frame the
    density is unknown, not zero.

    Everything that would make those sums wrong is refused up front, so that a
    cell is never silently dropped: a label or a well that is repeated, a well
    that a mapping lacks, an area that is not a size, a cell of a well or a label
    that the caller did not list. The checks come in the order of the ``Raises``
    and name the first offender.

    Args:
        cells: The cells of every well, as the cells stage detected them.
        wells: The wells to report, in the order to report them.
        areas_mm2: The imaged area of each well, by name.
        fields: The frames taken in each well, by name.
        failed_fields: The fields that failed in each well, by name.
        labels: The classes, in the order the counts keep.

    Returns:
        One ``DensityWell`` per entry of ``wells``, in that order.

    Raises:
        ValueError: No label, or a label twice; a well twice; a well missing
            from ``areas_mm2``, ``fields`` or ``failed_fields``; an area not
            finite or negative, or a count below 0; a cell in a well that is
            not among ``wells``, or with a label that is not among ``labels``.
    """
    if len(labels) == 0:
        raise ValueError("labels must name at least one label")
    for k, label in enumerate(labels):
        if label in labels[:k]:
            raise ValueError(f"label {label!r} is repeated")
    for k, well in enumerate(wells):
        if any(well.well == earlier.well for earlier in wells[:k]):
            raise ValueError(f"well {well.well!r} is listed twice")
    mappings: tuple[tuple[str, Mapping[str, float]], ...] = (
        ("areas_mm2", areas_mm2),
        ("fields", fields),
        ("failed_fields", failed_fields),
    )
    for well in wells:
        for mapping_name, mapping in mappings:
            if well.well not in mapping:
                raise ValueError(f"{mapping_name} has no entry for well {well.well!r}")
    for well in wells:
        area = areas_mm2[well.well]
        if not (math.isfinite(area) and area >= 0.0):
            raise ValueError(
                f"areas_mm2 of well {well.well!r} must be finite and not "
                f"negative, got {area!r}"
            )
        for mapping_name, mapping in mappings[1:]:
            if mapping[well.well] < 0:
                raise ValueError(
                    f"{mapping_name} of well {well.well!r} must not be negative, "
                    f"got {mapping[well.well]!r}"
                )

    of_well: dict[str, list[Cell]] = {well.well: [] for well in wells}
    for cell in cells:
        if cell.well not in of_well:
            raise ValueError(
                f"a cell is in well {cell.well!r}, which is not among the wells"
            )
        if cell.label not in labels:
            raise ValueError(
                f"a cell of well {cell.well!r} has label {cell.label!r}, which is "
                f"not among the labels {list(labels)}"
            )
        of_well[cell.well].append(cell)

    result = []
    for well in wells:
        counts = dict.fromkeys(labels, 0)
        outside = 0
        mine = of_well[well.well]
        if mine:
            inside = well.contains_many(
                np.array([cell.x_um for cell in mine], dtype=np.float64),
                np.array([cell.y_um for cell in mine], dtype=np.float64),
            )
            for cell, is_inside in zip(mine, inside.tolist(), strict=True):
                if is_inside:
                    counts[cell.label] += 1
                else:
                    outside += 1
        area = areas_mm2[well.well]
        result.append(
            DensityWell(
                well=well.well,
                fields=fields[well.well],
                failed_fields=failed_fields[well.well],
                area_mm2=area,
                counts=counts,
                outside_well=outside,
                per_mm2=(
                    {label: counts[label] / area for label in labels}
                    if area > 0.0
                    else None
                ),
            )
        )
    return tuple(result)


def _parallelogram(k: int, corners: Sequence[tuple[float, float]]) -> _Parallelogram:
    """Footprint ``k`` as a parallelogram, or the ``ValueError`` that says what is wrong with it.

    The checks come in this order so that the first problem is the one named.
    Corner 2 is checked against ``c1 + c3 - c0``: a rectangle turned by the
    camera is one, and anything else (corners in z-order, a trapezoid) would be
    rasterised as a different shape without a word.
    """
    if len(corners) != 4:
        raise ValueError(f"footprint {k} must have four corners, got {len(corners)}")
    if not all(math.isfinite(c) for corner in corners for c in corner):
        raise ValueError(f"footprint {k} has a corner that is not finite")
    if any(abs(c) > _MAX_COORDINATE_UM for corner in corners for c in corner):
        raise ValueError(f"footprint {k} has a corner beyond 1e12 µm")
    (c0x, c0y), (c1x, c1y), (c2x, c2y), (c3x, c3y) = corners
    e1x, e1y = c1x - c0x, c1y - c0y
    e3x, e3y = c3x - c0x, c3y - c0y
    det = e1x * e3y - e1y * e3x
    if not abs(det) > 0.0:
        raise ValueError(f"footprint {k} has no area")
    off = math.hypot(c2x - (c1x + c3x - c0x), c2y - (c1y + c3y - c0y))
    reach = max(abs(c) for corner in corners for c in corner)
    if off > _PARALLELOGRAM_TOL_UM + _PARALLELOGRAM_ULPS * math.ulp(reach):
        raise ValueError(
            f"footprint {k} is not a parallelogram: corner 2 is {off:.3g} µm from "
            f"where corners 0, 1 and 3 put it"
        )
    xs = (c0x, c1x, c2x, c3x)
    ys = (c0y, c1y, c2y, c3y)
    return _Parallelogram(
        c0x, c0y, e1x, e1y, e3x, e3y, det, min(xs), max(xs), min(ys), max(ys)
    )


def _grid_size(
    well: WellGeometry, width_um: float, height_um: float, grid_um: float
) -> tuple[int, int]:
    """The points per side of the grid, refused before anything is allocated (FM-74).

    The side bound is tested on the floats, so ``inf`` and ``nan`` are refused
    and ``math.ceil`` is only ever given a number that is at most 2**17; the
    product of the two sides is then a product of Python ints.
    """
    fx, fy = width_um / grid_um, height_um / grid_um
    prefix = (
        f"well {well.well}: a grid of {grid_um:g} µm over its imaged area would hold "
    )
    if not (fx <= _MAX_SIDE_POINTS and fy <= _MAX_SIDE_POINTS):
        raise GridTooFineError(
            f"{prefix}{fx:.3g} by {fy:.3g} points, more than {_MAX_SIDE_POINTS} on a side"
        )
    nx, ny = math.ceil(fx), math.ceil(fy)
    if nx * ny > _MAX_GRID_POINTS:
        raise GridTooFineError(
            f"{prefix}{nx} by {ny} points, more than {_MAX_GRID_POINTS} in all"
        )
    return nx, ny


def _index(values: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.int64]:
    """Grid indices clipped to ``[0, size]`` on the floats, then cast (FM-75)."""
    return np.clip(values, 0.0, float(size)).astype(np.int64)


def _inside(
    shape: _Parallelogram,
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Which points of the ``ys`` by ``xs`` window are in the parallelogram, the edge included.

    ``p = c0 + s e1 + t e3`` with the inverse of ``[e1 e3]`` written out: a
    point is inside when ``0 <= s, t <= 1``.
    """
    dx = xs[None, :] - shape.c0x
    dy = ys[:, None] - shape.c0y
    s = (dx * shape.e3y - dy * shape.e3x) / shape.det
    t = (dy * shape.e1x - dx * shape.e1y) / shape.det
    return (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
