"""``render`` turns the records of a scan into one static HTML page (m3 §13).

Pure: the same records give the same bytes (the plugin passes the time in
``facts``), and nothing here imports ``smc.hardware``. What it decides, because
the page is opened by double-click on a Windows PC, offline, and attached to
issues:

* **One file, no script, no URL.** Inline SVG and one static ``<style>``; the
  only ``href`` are ``#`` fragments to the well panels. Free text from a record
  goes through ``escape`` and nothing from a record is an ``id``, a class or a
  style (markup built from a result's text).
* **A frame is what the scan saw.** A footprint is the scan's matrix and frame
  shape at the stage's readback; a chosen field of the route uses the route's
  (another objective is another field), never the plan's copy (FM-124).
* **Y is flipped by arithmetic, never by a transform.** Stage Y is up and the
  page's is down; a ``scale(1, -1)`` on a group would mirror the text in it.
* **A number that is not finite is refused, not drawn.** ``num`` names the stage
  to run again (FM-114); a ``nan`` in an attribute would silently vanish.
* **Unknown is not zero** (FM-138): a well with no frame, a well whose imaged
  area is below the grid, and a measured 0 are three different texts and three
  different ``data-state``/fills.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.report.svg import (
    MAX_CELLS_DRAWN_TOTAL,
    MAX_COORDINATE_UM,
    density_text,
    escape,
    label_colour,
    num,
    sample_indices,
    shade,
    text_colour_on,
)
from smc.plugins.slots import (
    Cell,
    DensityResult,
    DensityWell,
    RoutePosition,
    RouteResult,
    ScanPlanResult,
    ScanResult,
    WellGeometry,
    normalise_well,
    well_geometry,
)

__all__ = ["render"]

#: A plate of more than this many wells has no value written in its wells: the
#: text would be a speck. The table and each well's tooltip hold it always.
_MAX_WELLS_WITH_TEXT = 384

#: Grey of a scanned well whose density is unknown, never the fill of a zero.
_UNKNOWN_FILL = "#e3e3e3"

_NO_FRAME = "no frame: unknown"

#: Side of a class swatch in the legend, in px. The stylesheet's ``svg { height:
#: auto }`` lets an svg that has a viewBox and no size fill its container (1252
#: px in Chrome), so the swatch carries a size of its own, and ``svg.dot-key`` in
#: ``_CSS`` makes it follow the text.
_SWATCH_PX = 16

Point = tuple[float, float]

_CSS = """
:root { color-scheme: light; }
body { font-family: system-ui, "Segoe UI", sans-serif; color: #1b1b1b; background: #ffffff; margin: 0 auto; max-width: 1280px; padding: 1rem 1.5rem 3rem; line-height: 1.4; }
h1 { margin-bottom: 0.4rem; }
h2 { margin-top: 2rem; border-bottom: 1px solid #d0d0d0; padding-bottom: 0.2rem; }
svg { max-width: 100%; height: auto; }
svg.dot-key { width: 1em; height: 1em; vertical-align: -0.15em; }
table { border-collapse: collapse; }
th, td { border: 1px solid #cfcfcf; padding: 0.2rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
table.facts th { text-align: left; background: #f3f3f3; font-weight: 600; }
table.facts td { text-align: left; }
table.density th { background: #f3f3f3; }
table.density tr:nth-child(even) td { background: #fafafa; }
td.unknown { color: #5a5a5a; font-style: italic; }
.warn { background: #fff3d1; border-left: 4px solid #e69f00; padding: 0.5rem 0.8rem; }
.note { color: #4a4a4a; }
.assumed li { margin-bottom: 0.25rem; }
.maps { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; }
.wells { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
section.well-detail { border: 1px solid #d8d8d8; border-radius: 4px; padding: 0.4rem 0.8rem 0.8rem; break-inside: avoid; }
section.well-detail h3 { margin: 0.2rem 0; }
p.counts, p.caption { margin: 0.2rem 0; font-size: 0.9rem; }
p.caption { color: #4a4a4a; }
.well circle, .well path { stroke: #707070; stroke-width: 1; vector-effect: non-scaling-stroke; }
.outline { fill: none; stroke: #707070; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
path.frames { fill: none; stroke: #4c78a8; stroke-width: 1; vector-effect: non-scaling-stroke; }
path.failed { fill: none; stroke: #c0392b; stroke-width: 1; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; }
path.field { fill: #ffd54f; fill-opacity: 0.35; stroke: #1b1b1b; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
polyline.route, polyline.route-well { fill: none; stroke: #1b1b1b; stroke-width: 1.2; stroke-dasharray: 5 3; vector-effect: non-scaling-stroke; }
circle.start { fill: #009e73; stroke: #ffffff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
circle.end { fill: #d55e00; stroke: #ffffff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.route-map .visited { stroke: #1b1b1b; stroke-width: 2; }
text { font-family: system-ui, "Segoe UI", sans-serif; }
text.order, text.rank { paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; font-weight: 600; }
.swatch-label { font-size: 0.9rem; }
.key { display: inline-block; width: 1.4em; height: 0.8em; vertical-align: baseline; margin-right: 0.2em; }
.key.frames { border: 1px solid #4c78a8; }
.key.failed { border: 1px dashed #c0392b; }
.key.field { border: 1.5px solid #1b1b1b; background: #fff0b8; }
""".strip()


@dataclass(frozen=True)
class _Plate:
    """The plate drawn in one frame: every well's geometry, one origin and one flip for all.

    ``(x0, y1)`` is the top-left of the extent of all the wells plus a margin of
    a tenth of a well, so that stage ``(x, y)`` is drawn at ``(x - x0, y1 - y)``
    and A1 is at the top left when the stage's Y of row A is the larger. A well
    is the unit every size on the page is a fraction of: ``well_um`` is the
    widest.
    """

    geometries: dict[str, WellGeometry]
    x0: float
    y1: float
    width_um: float
    height_um: float
    well_um: float

    def at(self, x_um: float, y_um: float) -> Point:
        return (x_um - self.x0, self.y1 - y_um)


def render(
    *,
    plan: ScanPlanResult,
    scan: ScanResult,
    cells: Sequence[Cell],
    density: DensityResult,
    route: RouteResult | None,
    title: str,
    facts: Sequence[tuple[str, str]],
    assumed: Sequence[str],
    max_cells_drawn_per_well: int,
) -> str:
    """The report as one HTML page.

    Args:
        plan: The scan plan; its plate gives every well's shape and place.
        scan: The scan; its matrix and frame shape draw the footprints.
        cells: Every detected cell of the cells file.
        density: The density per well; the wells it reports are the ones drawn.
        route: The route, or ``None``: the page then says there is none.
        title: The page's title and ``<h1>``.
        facts: ``(name, text)`` rows printed after the page's own, escaped.
        assumed: Every assumption of the stages upstream, already labelled by
            stage by the caller.
        max_cells_drawn_per_well: The most cells drawn in one well's panel.

    Raises:
        ValueError: A well is not on the plate, or a coordinate or size is not
            finite; the message ends ``; run <stage> again``.
    """
    plate = _plate_of(plan)
    scanned = _scanned_index(plate, density)
    _require_matrix(scan.matrix_um_per_px, stage="scan")
    if route is not None:
        _require_matrix(route.matrix_um_per_px, stage="route")
    body = [
        _header(
            title=title,
            plan=plan,
            scan=scan,
            density=density,
            route=route,
            facts=facts,
            assumed=assumed,
        ),
        _plate_maps(plate, density, scanned),
        _density_table(density, route),
        _route(plate, route),
        _panels(
            plate=plate,
            scan=scan,
            cells=cells,
            density=density,
            route=route,
            max_cells_drawn_per_well=max_cells_drawn_per_well,
        ),
    ]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<style>{_CSS}</style></head><body>" + "".join(body) + "</body></html>"
    )


# ------------------------------------------------------------------- the plate


def _plate_of(plan: ScanPlanResult) -> _Plate:
    """Every well of the plan's plate, in its order, and the one frame they are drawn in.

    Raises:
        ValueError: The plate's registration is not finite (``scan-plan``).
    """
    plate_plan = plan.plate_plan
    stage = "scan-plan"
    for value, what in (
        (plate_plan.a1_center_xy[0], "the plate's A1 centre"),
        (plate_plan.a1_center_xy[1], "the plate's A1 centre"),
        (
            plate_plan.rotation if plate_plan.rotation is not None else 0.0,
            "the plate's rotation",
        ),
    ):
        num(value, what=what, stage=stage)
    geometries: dict[str, WellGeometry] = {}
    for name in plate_plan.all_well_names.ravel():
        geometry = _geometry(plan, str(name))
        geometries[geometry.well] = geometry
    xs: list[float] = []
    ys: list[float] = []
    for geometry in geometries.values():
        for x, y in _extent_points(geometry):
            xs.append(x)
            ys.append(y)
    well_um = max(geometry.size_um[0] for geometry in geometries.values())
    margin = 0.1 * well_um
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    return _Plate(
        geometries=geometries,
        x0=x0,
        y1=y1,
        width_um=x1 - x0,
        height_um=y1 - y0,
        well_um=well_um,
    )


def _geometry(plan: ScanPlanResult, well: str) -> WellGeometry:
    """The geometry of a well, or the refusal that says to plan again."""
    try:
        return well_geometry(plan.plate_plan, well)
    except ValueError as exc:
        raise ValueError(f"{exc}; run scan-plan again") from exc


def _scanned_index(plate: _Plate, density: DensityResult) -> dict[str, int]:
    """The index in ``density.wells`` of each scanned well, by the plate's name for it.

    Raises:
        ValueError: A reported well is not on the plate (``scan-plan``).
    """
    index: dict[str, int] = {}
    for k, reported in enumerate(density.wells):
        name = _plate_name(plate, reported.well)
        index[name] = k
    return index


def _plate_name(plate: _Plate, well: str) -> str:
    """The plate's own name of a well (``a1`` is ``A1``), or the refusal."""
    name = normalise_well(well)
    if name not in plate.geometries:
        raise ValueError(
            f"well {well!r} is not on the {len(plate.geometries)}-well plate; "
            f"run scan-plan again"
        )
    return name


def _extent_points(geometry: WellGeometry) -> list[Point]:
    """The points whose bounding box is the well: the corners, or the circle's extremes."""
    cx, cy = geometry.center_um
    if geometry.circular:
        radius = geometry.size_um[0] / 2.0
        return [(cx - radius, cy - radius), (cx + radius, cy + radius)]
    return _corners(geometry)


def _corners(geometry: WellGeometry) -> list[Point]:
    """The four corners of a square well in stage µm: the plate's axes turned by its rotation."""
    half_u, half_v = geometry.size_um[0] / 2.0, geometry.size_um[1] / 2.0
    angle = math.radians(geometry.rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    cx, cy = geometry.center_um
    return [
        (cx + cos * u - sin * v, cy + sin * u + cos * v)
        for u, v in (
            (-half_u, -half_v),
            (half_u, -half_v),
            (half_u, half_v),
            (-half_u, half_v),
        )
    ]


def _plate_num(value: float) -> str:
    """A coordinate of the plate's layout, which the scan plan holds."""
    return num(value, what="the plate's well position", stage="scan-plan")


# ------------------------------------------------------------------ the header


def _header(
    *,
    title: str,
    plan: ScanPlanResult,
    scan: ScanResult,
    density: DensityResult,
    route: RouteResult | None,
    facts: Sequence[tuple[str, str]],
    assumed: Sequence[str],
) -> str:
    """The title, the table of facts, the warning for failed fields, and what was assumed."""
    taken = sum(well.fields for well in density.wells)
    failed = sum(well.failed_fields for well in density.wells)
    wells_failed = sum(1 for well in density.wells if well.failed_fields > 0)
    height, width = scan.image_shape_px
    rows = [
        ("Plate", plan.plate_plan.plate.name),
        ("Objective", scan.objective or "not recorded"),
    ]
    if route is not None and route.objective != scan.objective:
        rows.append(("Route imaged with", route.objective or "not recorded"))
    rows += [
        # RUF001: the multiplication sign is meant, as in the scan plan's refusals.
        ("Frame", f"{height} × {width} px at {scan.pixel_size_um:g} µm per pixel"),  # noqa: RUF001
        ("Wells scanned", str(len(density.wells))),
        ("Fields", f"{taken} taken, {failed} failed"),
        ("Cells", _cells_found(density)),
        *facts,
    ]
    table = "".join(
        f"<tr><th>{escape(name)}</th><td>{escape(text)}</td></tr>"
        for name, text in rows
    )
    parts = [f"<h1>{escape(title)}</h1>", f'<table class="facts">{table}</table>']
    if failed:
        # A partial scan must not read as a whole one (FM-106).
        parts.append(
            f'<p class="warn">{_count(failed, "field", "fields")} failed in '
            f"{_count(wells_failed, 'well', 'wells')}: "
            f"{'its' if failed == 1 else 'their'} area is not in the density</p>"
        )
    parts.append("<h2>What was assumed</h2>")
    if assumed:
        items = "".join(f"<li>{escape(line)}</li>" for line in assumed)
        parts.append(f'<ul class="assumed">{items}</ul>')
    else:
        parts.append('<p class="note">The stages recorded no assumption.</p>')
    return "".join(parts)


def _cells_found(density: DensityResult) -> str:
    """The cells of all the scanned wells per label: ``single 1 234 · doublet 56``."""
    if not density.labels:
        return "none: no class was detected"
    return " · ".join(
        f"{label} {_grouped(sum(well.counts.get(label, 0) for well in density.wells))}"
        for label in density.labels
    )


def _grouped(count: int) -> str:
    """A count with its thousands set apart by a no-break space: ``1 234``, never ``1,234``."""
    return f"{count:,}".replace(",", "\N{NO-BREAK SPACE}")


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# ----------------------------------------------------------------- plate maps


def _value(reported: DensityWell, label: str) -> float | None:
    """The density of a label in a well: ``None`` is unknown, a measured 0 is 0.0."""
    return None if reported.per_mm2 is None else reported.per_mm2.get(label)


def _unknown_text(reported: DensityWell, area_grid_um: float) -> str:
    """Why a density is unknown, in the two words that are not one zero (FM-138)."""
    if reported.fields == 0:
        return _NO_FRAME
    return f"area below the grid's resolution ({area_grid_um:g} µm)"


def _plate_maps(plate: _Plate, density: DensityResult, scanned: dict[str, int]) -> str:
    """One map per label: every well of the plate, filled by density on that label's own scale."""
    if not density.labels:
        return '<h2>Plate maps</h2><p class="note">No class was detected.</p>'
    maps = "".join(
        _plate_map(plate, density, scanned, index, label)
        for index, label in enumerate(density.labels)
    )
    return f'<h2>Plate maps</h2><div class="maps">{maps}</div>'


def _plate_map(
    plate: _Plate,
    density: DensityResult,
    scanned: dict[str, int],
    index: int,
    label: str,
) -> str:
    """The plate drawn once for ``label``: the largest density has the full colour.

    The scale is linear from 0 to the largest density of this label among the
    scanned wells that have one, so a map says where the cells are, not how
    many the other labels hold. The unit of the legend is no larger than a
    fourteenth of the plate: a six-well plate's wells are 35 mm wide and its
    legend must still fit.
    """
    colour = label_colour(index)
    values = [v for w in density.wells if (v := _value(w, label)) is not None]
    top = max(values, default=0.0)
    unit = min(plate.well_um, plate.width_um / 14.0)
    legend_um = 2.0 * unit
    wells = [
        _map_well(plate, density, scanned, label, colour, top, name, geometry)
        for name, geometry in plate.geometries.items()
    ]
    width = _plate_num(plate.width_um)
    height = _plate_num(plate.height_um + legend_um)
    text = escape(label)
    return (
        f'<section class="plate-map-section"><h3>{text}: cells per mm²</h3>'
        f'<svg class="plate-map" data-label="{text}" role="img" '
        f'aria-label="Plate map of {text}: cells per mm² in each well" '
        f'viewBox="0 0 {width} {height}">'
        f"<title>Plate map of {text}</title>"
        + "".join(wells)
        + _legend(plate, colour, top, unit)
        + "</svg></section>"
    )


def _map_well(
    plate: _Plate,
    density: DensityResult,
    scanned: dict[str, int],
    label: str,
    colour: str,
    top: float,
    name: str,
    geometry: WellGeometry,
) -> str:
    """One well of a map: scanned with a density, scanned without one, or only outlined."""
    k = scanned.get(name)
    value_text: str | None = None
    attributes = ""
    if k is None:
        state, fill, tip = "unscanned", "none", f"{name} — not scanned"
    else:
        reported = density.wells[k]
        value = _value(reported, label)
        if value is None:
            state, fill, value_text = "unknown", _UNKNOWN_FILL, "n/a"
            tip = f"{name} — {label}: {_unknown_text(reported, density.area_grid_um)}"
        else:
            state, value_text = "density", density_text(value)
            fill = shade(colour, value / top if top > 0.0 else 0.0)
            attributes = f' data-value="{_plate_num(value)}"'
            tip = (
                f"{name} — {label} {value_text} per mm² "
                f"({reported.counts.get(label, 0)} cells in {reported.area_mm2:.1f} mm², "
                f"{reported.fields} fields)"
            )
    cx, cy = plate.at(*geometry.center_um)
    parts = [
        f'<g class="well" data-well="{escape(name)}" data-state="{state}"{attributes}>',
        f"<title>{escape(tip)}</title>",
        _well_shape(plate, geometry, fill),
    ]
    if value_text is not None and len(plate.geometries) <= _MAX_WELLS_WITH_TEXT:
        parts.append(
            f'<text x="{_plate_num(cx)}" y="{_plate_num(cy)}" '
            f'font-size="{_plate_num(0.28 * geometry.size_um[0])}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'fill="{text_colour_on(fill) if fill != "none" else "#000000"}">'
            f"{escape(value_text)}</text>"
        )
    parts.append("</g>")
    well = "".join(parts)
    return f'<a href="#w{k}">{well}</a>' if k is not None else well


def _well_shape(plate: _Plate, geometry: WellGeometry, fill: str) -> str:
    """A well of a map, filled."""
    return _shape(geometry, plate.at, f'fill="{fill}"')


def _shape(
    geometry: WellGeometry, at: Callable[[float, float], Point], attributes: str
) -> str:
    """A round well is a circle, a square one a path of its four corners turned by the plate.

    ``at`` puts a stage point on the page (the plate's frame or one well's), so
    a map and a panel draw the same shape.
    """
    if geometry.circular:
        cx, cy = at(*geometry.center_um)
        return (
            f'<circle cx="{_plate_num(cx)}" cy="{_plate_num(cy)}" '
            f'r="{_plate_num(geometry.size_um[0] / 2.0)}" {attributes}/>'
        )
    points = [at(x, y) for x, y in _corners(geometry)]
    return f'<path d="{_polygon(points, _plate_num)}" {attributes}/>'


def _polygon(points: Sequence[Point], printed: Callable[[float], str]) -> str:
    """A closed polygon as the ``d`` of a path; ``printed`` writes each coordinate."""
    return "M" + "L".join(f"{printed(x)} {printed(y)}" for x, y in points) + "Z"


def _legend(plate: _Plate, colour: str, top: float, unit: float) -> str:
    """Five swatches of the scale (0, ¼, ½, ¾, 1 of the largest density) and the unit."""
    side, pitch = 0.6 * unit, 1.3 * unit
    y = plate.height_um + 0.2 * unit
    font = _plate_num(0.3 * unit)
    parts = []
    for step in range(5):
        t = step / 4.0
        x = 0.5 * unit + step * pitch
        parts.append(
            f'<rect x="{_plate_num(x)}" y="{_plate_num(y)}" '
            f'width="{_plate_num(side)}" height="{_plate_num(side)}" '
            f'fill="{shade(colour, t)}" stroke="#707070"/>'
            f'<text x="{_plate_num(x + side / 2.0)}" y="{_plate_num(y + side + 0.45 * unit)}" '
            f'font-size="{font}" text-anchor="middle" fill="#1b1b1b">'
            f"{density_text(t * top)}</text>"
        )
    parts.append(
        f'<text x="{_plate_num(0.5 * unit + 5 * pitch)}" y="{_plate_num(y + side / 2.0)}" '
        f'font-size="{font}" dominant-baseline="central" fill="#1b1b1b">per mm²</text>'
    )
    return "".join(parts)


# ----------------------------------------------------------------- the table


def _density_table(density: DensityResult, route: RouteResult | None) -> str:
    """Well by label: fields, failed fields, area, count and density, and what lies outside."""
    head = ["<th>Well</th><th>Fields</th><th>Failed fields</th><th>Area (mm²)</th>"]
    for label in density.labels:
        text = escape(label)
        head.append(f"<th>{text}: cells</th><th>{text}: per mm²</th>")
    head.append("<th>Outside the well shape</th>")
    if route is not None:
        wanted = escape(", ".join(route.labels))
        head.append(f"<th>Not in a chosen field ({wanted})</th>")
    rows = []
    for k, reported in enumerate(density.wells):
        cells = [
            f'<td><a href="#w{k}">{escape(reported.well)}</a></td>',
            f"<td>{reported.fields}</td>",
            f"<td>{reported.failed_fields}</td>",
            f"<td>{reported.area_mm2:.2f}</td>",
        ]
        for label in density.labels:
            cells.append(f"<td>{reported.counts.get(label, 0)}</td>")
            value = _value(reported, label)
            if value is None:
                unknown = escape(_unknown_text(reported, density.area_grid_um))
                cells.append(f'<td class="unknown">{unknown}</td>')
            else:
                cells.append(f"<td>{density_text(value)}</td>")
        cells.append(f"<td>{reported.outside_well}</td>")
        if route is not None:
            cells.append(f"<td>{route.uncovered.get(reported.well, 0)}</td>")
        rows.append(f'<tr data-well="{escape(reported.well)}">{"".join(cells)}</tr>')
    return (
        "<h2>Density per well</h2>"
        f'<table class="density"><thead><tr>{"".join(head)}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# ------------------------------------------------------------------ the route


def _require_matrix(matrix: Matrix2, *, stage: str) -> None:
    """Refuse a camera matrix that is not finite: every footprint drawn from it would be ``nan``.

    Named here so that the refusal says "a camera matrix" and not "a frame's
    position", which the frames would show a step later.
    """
    for row in matrix:
        for entry in row:
            num(entry, what="a camera matrix", stage=stage)


def _stop_num(value: float) -> str:
    """A coordinate of a stop, which the route holds."""
    return num(value, what="a stop's position", stage="route")


def _route(plate: _Plate, route: RouteResult | None) -> str:
    """The route across the plate, its length and its time; or the sentence that there is none."""
    if route is None:
        return (
            '<h2>Route</h2><p class="note">'
            "no route: run route to add the fields to image</p>"
        )
    stops = sorted(route.positions, key=lambda stop: stop.order)
    # The rank of a well is the order of its first stop; the stop's well must be on the plate.
    rank: dict[str, int] = {}
    for stop in stops:
        name = normalise_well(stop.well)
        if name not in plate.geometries:
            raise ValueError(
                f"the route has a stop in well {stop.well!r}, which is not on the "
                f"{len(plate.geometries)}-well plate; run route again"
            )
        rank.setdefault(name, len(rank) + 1)
    outlines = "".join(
        _shape(
            geometry,
            plate.at,
            'class="outline visited"' if name in rank else 'class="outline"',
        )
        for name, geometry in plate.geometries.items()
    )
    parts = [outlines]
    if stops:
        points = [plate.at(stop.x_um, stop.y_um) for stop in stops]
        line = " ".join(f"{_stop_num(x)},{_stop_num(y)}" for x, y in points)
        parts.append(
            f'<polyline class="route" data-stops="{len(stops)}" points="{line}"/>'
        )
        radius = _plate_num(0.12 * plate.well_um)
        sx, sy = points[0]
        parts.append(
            f'<circle class="start" cx="{_stop_num(sx)}" cy="{_stop_num(sy)}" r="{radius}"/>'
        )
        if len(points) > 1:
            ex, ey = points[-1]
            parts.append(
                f'<circle class="end" cx="{_stop_num(ex)}" cy="{_stop_num(ey)}" r="{radius}"/>'
            )
    if len(plate.geometries) <= _MAX_WELLS_WITH_TEXT:
        font = _plate_num(0.35 * plate.well_um)
        for name, order in rank.items():
            x, y = plate.at(*plate.geometries[name].center_um)
            parts.append(
                f'<text class="rank" x="{_plate_num(x)}" y="{_plate_num(y)}" '
                f'font-size="{font}" text-anchor="middle" '
                f'dominant-baseline="central" fill="#1b1b1b">{order}</text>'
            )
    travel = (
        "not estimated: no timing model"
        if route.travel_s is None
        else "{} min {} s".format(*divmod(round(route.travel_s), 60))
    )
    uncovered = sum(route.uncovered.values())
    wanted = escape(", ".join(route.labels))
    return (
        "<h2>Route</h2>"
        '<svg class="route-map" role="img" '
        'aria-label="The route across the plate, from the green start to the red end" '
        f'viewBox="0 0 {_plate_num(plate.width_um)} {_plate_num(plate.height_um)}">'
        "<title>The route across the plate</title>" + "".join(parts) + "</svg>"
        f'<p class="route-summary">{_count(len(stops), "stop", "stops")} in '
        f"{_count(len(rank), 'well', 'wells')} · {route.distance_um / 1000.0:.1f} mm · "
        f"{travel}</p>"
        f'<p class="note">{_count(uncovered, "cell", "cells")} of the wanted labels '
        f"({wanted}) {'is' if uncovered == 1 else 'are'} in no chosen field.</p>"
    )


# ------------------------------------------------------------------ the panels


@dataclass(frozen=True)
class _Scene:
    """What every panel draws from: the scan, the route, the labels and the cap of dots."""

    scan: ScanResult
    density: DensityResult
    route: RouteResult | None
    label_index: dict[str, int]
    frames_of: dict[str, list[int]]
    cap: int


def _panels(
    *,
    plate: _Plate,
    scan: ScanResult,
    cells: Sequence[Cell],
    density: DensityResult,
    route: RouteResult | None,
    max_cells_drawn_per_well: int,
) -> str:
    """One panel per scanned well, in the density's order, linked from the maps by index.

    The dots of the whole page are bounded: a cap per well times 96 wells is no
    bound (FM-74), so each well draws at most its share of ``MAX_CELLS_DRAWN_TOTAL``.
    """
    if not density.wells:
        return '<h2>Wells</h2><p class="note">No well was scanned.</p>'
    by_well: dict[str, list[Cell]] = {}
    for cell in cells:
        by_well.setdefault(cell.well, []).append(cell)
    stops_of: dict[str, list[RoutePosition]] = {}
    if route is not None:
        for stop in sorted(route.positions, key=lambda stop: stop.order):
            stops_of.setdefault(stop.well, []).append(stop)
    scene = _Scene(
        scan=scan,
        density=density,
        route=route,
        label_index={label: i for i, label in enumerate(density.labels)},
        frames_of=scan.frame_index(),
        cap=min(
            max_cells_drawn_per_well,
            MAX_CELLS_DRAWN_TOTAL // max(1, len(density.wells)),
        ),
    )
    sections = [
        _panel(
            k=k,
            reported=reported,
            geometry=plate.geometries[_plate_name(plate, reported.well)],
            scene=scene,
            cells_here=by_well.get(reported.well, []),
            stops_here=stops_of.get(reported.well, []),
        )
        for k, reported in enumerate(density.wells)
    ]
    return f'<h2>Wells</h2>{_panel_legend(density)}<div class="wells">{"".join(sections)}</div>'


def _panel_legend(density: DensityResult) -> str:
    """What the outlines, the dashes, the dots and the numbered rectangles are, once."""
    dots = "".join(
        f'<svg class="dot-key" role="img" aria-label="{escape(label)}" '
        f'width="{_SWATCH_PX}" height="{_SWATCH_PX}" viewBox="0 0 10 10">'
        f'<circle cx="5" cy="5" r="4" fill="{label_colour(i)}"/></svg>'
        f" {escape(label)} "
        for i, label in enumerate(density.labels)
    )
    return (
        '<p class="legend">'
        '<span class="key frames"></span> fields the scan imaged · '
        '<span class="key failed"></span> fields that failed · '
        f"dots: detected cells, {dots}· "
        '<span class="key field"></span> numbered: the fields of the route, in order'
        "</p>"
    )


def _panel(
    *,
    k: int,
    reported: DensityWell,
    geometry: WellGeometry,
    scene: _Scene,
    cells_here: list[Cell],
    stops_here: list[RoutePosition],
) -> str:
    """One well: its outline, the footprints of its frames, its cells, and the route's fields.

    The centre of the well is the origin and a stage point ``(x, y)`` is drawn
    at ``(x - cx, cy - y)``. The frames are the scan's footprints at the stage's
    readback, with the scan's matrix and frame shape; a failed field has only
    its target; a chosen field of the route has the route's own matrix and shape
    (FM-124). The viewBox is the well's circumscribed square plus a tenth, and
    grows to hold a footprint that reaches past the wall (fields are clipped by
    their footprint meeting the well, so the outer ones stick out).
    """
    scan = scene.scan
    well = reported.well
    cx, cy = geometry.center_um
    reach = max(max(abs(x - cx), abs(y - cy)) for x, y in _extent_points(geometry))

    def local(corners: Sequence[Point]) -> list[Point]:
        return [(x - cx, cy - y) for x, y in corners]

    def farthest(points: Sequence[Point]) -> float:
        return max(max(abs(px), abs(py)) for px, py in points)

    positions = scene.frames_of.get(well, [])
    frame_paths: list[str] = []
    for position in positions:
        frame = scan.frames[position]
        corners = local(
            footprint(
                frame.x_um,
                frame.y_um,
                matrix=scan.matrix_um_per_px,
                image_shape_px=scan.image_shape_px,
            )
        )
        reach = max(reach, farthest(corners))
        frame_paths.append(_polygon(corners, _frame_num))
    failed_here = [failed for failed in scan.failed if failed.well == well]
    failed_paths: list[str] = []
    for failed in failed_here:
        corners = local(
            footprint(
                failed.x_um,
                failed.y_um,
                matrix=scan.matrix_um_per_px,
                image_shape_px=scan.image_shape_px,
            )
        )
        reach = max(reach, farthest(corners))
        failed_paths.append(_polygon(corners, _failed_num))

    outside, drawn, total, dots = _dots(geometry, cells_here, scene)
    diameter = _plate_num(max(0.012 * geometry.size_um[0], 12.0))

    body = [
        _shape(geometry, lambda x, y: (x - cx, cy - y), 'class="outline"'),
        f'<path class="frames" data-n="{len(positions)}" d="{"".join(frame_paths)}"/>',
    ]
    if failed_paths:
        body.append(
            f'<path class="failed" data-n="{len(failed_paths)}" '
            f'd="{"".join(failed_paths)}"/>'
        )
    for label, label_dots in dots.items():
        index = scene.label_index[label]
        body.append(
            f'<path class="cells" data-label="{escape(label)}" '
            f'data-drawn="{len(label_dots)}" data-total="{total[label]}" '
            f'd="{"".join(label_dots)}" fill="none" stroke="{label_colour(index)}" '
            f'stroke-width="{diameter}" stroke-linecap="round"/>'
        )
    route = scene.route
    if route is not None and stops_here:
        fields, numbers, reach = _route_fields(
            stops_here, route, cx, cy, geometry.size_um[0], reach
        )
        points = " ".join(
            f"{_stop_num(stop.x_um - cx)},{_stop_num(cy - stop.y_um)}"
            for stop in stops_here
        )
        body.extend(fields)
        if len(stops_here) > 1:
            body.append(f'<polyline class="route-well" points="{points}"/>')
        body.extend(numbers)

    view = 1.1 * reach
    return (
        f'<section class="well-detail" id="w{k}" data-well="{escape(well)}">'
        f"<h3>{escape(well)}</h3>"
        f'<p class="counts">{_counts_line(reported, scene.density)}</p>'
        f'<svg class="well-panel" role="img" '
        f'aria-label="Well {escape(well)}: its fields, its cells and the route" '
        f'viewBox="{_plate_num(-view)} {_plate_num(-view)} '
        f'{_plate_num(2.0 * view)} {_plate_num(2.0 * view)}">'
        f"<title>Well {escape(well)}</title>" + "".join(body) + "</svg>"
        f'<p class="caption">{_caption(reported, len(failed_here), drawn, sum(total.values()), outside, scene.cap)}</p>'
        "</section>"
    )


def _frame_num(value: float) -> str:
    return num(value, what="a frame's position", stage="scan")


def _failed_num(value: float) -> str:
    return num(value, what="a failed field's position", stage="scan")


def _counts_line(reported: DensityWell, density: DensityResult) -> str:
    """Each label's count with its density: ``single 5 (12.5 per mm²) · doublet 2 (5.00 per mm²)``."""
    parts = []
    for label in density.labels:
        value = _value(reported, label)
        shown = (
            _unknown_text(reported, density.area_grid_um)
            if value is None
            else f"{density_text(value)} per mm²"
        )
        parts.append(
            f"{escape(label)} {reported.counts.get(label, 0)} ({escape(shown)})"
        )
    return " · ".join(parts)


def _caption(
    reported: DensityWell,
    failed: int,
    drawn: int,
    total: int,
    outside: int,
    cap: int,
) -> str:
    """How many fields, how many cells are drawn of how many, and what is not drawn and why."""
    text = (
        f"{reported.fields} fields ({failed} failed) · {drawn} of {total} cells drawn"
    )
    if total and drawn < total:
        text += (
            " (none: the cap per well is 0)" if cap <= 0 else " (a regular subsample)"
        )
    if outside:
        text += (
            f" · {_count(outside, 'detection', 'detections')} outside the well "
            f"shape {'is' if outside == 1 else 'are'} not drawn"
        )
    return text


def _require_finite_cells(
    xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64]
) -> None:
    """Refuse a cell whose position is not finite or is beyond 1e12 µm, naming the cells stage.

    ``contains_many`` calls a point that is not finite *outside*, which would
    leave it uncounted and undrawn without a word (FM-114); this says so.
    """
    for values in (xs, ys):
        finite = np.isfinite(values)
        bad = ~finite | (np.abs(np.where(finite, values, 0.0)) > MAX_COORDINATE_UM)
        if bad.any():
            num(float(values[bad][0]), what="a cell's position", stage="cells")


def _dots(
    geometry: WellGeometry, cells_here: list[Cell], scene: _Scene
) -> tuple[int, int, dict[str, int], dict[str, list[str]]]:
    """The dots of a well's cells: how many lie outside the shape, how many are drawn, per label.

    Only the cells inside the well's shape (``contains_many``, the set density
    counted, FM-132) are candidates; they are thinned once across labels,
    evenly and in the order of the cells file, to the well's cap. A dot is
    ``M x y h0.1`` stroked with a round cap: a path per label and well, never
    an element per cell, is the difference between 4.7 MB and 12.2 MB on a
    plate of 96. Returns the number of cells outside the shape, the number
    drawn, the number of candidates per label and the subpaths per label.
    """
    cx, cy = geometry.center_um
    xs = np.array([cell.x_um for cell in cells_here], dtype=np.float64)
    ys = np.array([cell.y_um for cell in cells_here], dtype=np.float64)
    _require_finite_cells(xs, ys)
    inside = [bool(flag) for flag in geometry.contains_many(xs, ys)]
    candidates = [
        cell
        for cell, held in zip(cells_here, inside, strict=True)
        if held and cell.label in scene.label_index
    ]
    total = {label: 0 for label in scene.label_index}
    for cell in candidates:
        total[cell.label] += 1
    dots: dict[str, list[str]] = {label: [] for label in scene.label_index}
    chosen = sample_indices(len(candidates), scene.cap)
    for position in chosen:
        cell = candidates[position]
        x = num(cell.x_um - cx, what="a cell's position", stage="cells")
        y = num(cy - cell.y_um, what="a cell's position", stage="cells")
        dots[cell.label].append(f"M{x} {y}h0.1")
    return len(cells_here) - sum(inside), len(chosen), total, dots


def _route_fields(
    stops: list[RoutePosition],
    route: RouteResult,
    cx: float,
    cy: float,
    well_um: float,
    reach: float,
) -> tuple[list[str], list[str], float]:
    """The chosen fields of one well as numbered rectangles, in the route's matrix and frame shape.

    Another objective is another field, so the footprint is the route's and not
    the scan's (FM-124). Returns the rectangles, their numbers (``order + 1``,
    drawn after the line so that nothing hides them) and the grown reach.
    """
    fields: list[str] = []
    numbers: list[str] = []
    font = _plate_num(0.045 * well_um)
    for stop in stops:
        corners = [
            (x - cx, cy - y)
            for x, y in footprint(
                stop.x_um,
                stop.y_um,
                matrix=route.matrix_um_per_px,
                image_shape_px=route.image_shape_px,
            )
        ]
        reach = max(reach, *(max(abs(x), abs(y)) for x, y in corners))
        held = (
            ", ".join(f"{n} {label}" for label, n in stop.counts.items())
            or "no wanted cell"
        )
        fields.append(
            f'<path class="field" data-order="{stop.order}" data-name="{escape(stop.name)}" '
            f'd="{_polygon(corners, _stop_num)}">'
            f"<title>{escape(stop.name)}: {escape(held)}</title></path>"
        )
        numbers.append(
            f'<text class="order" x="{_stop_num(stop.x_um - cx)}" '
            f'y="{_stop_num(cy - stop.y_um)}" font-size="{font}" text-anchor="middle" '
            f'dominant-baseline="central" fill="#1b1b1b">{stop.order + 1}</text>'
        )
    return fields, numbers, reach
