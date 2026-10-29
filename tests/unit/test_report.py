"""``render``, the page of the report, on records built here (m3 §13, #195).

Pure: the records are built by the test (the plan by ``plan_fields``, the scan
with a frame at every planned field, the cells and their density from the seed)
and the page is read back with ``html.parser``, no browser. The camera is
60 x 80 pixels (not square, FM-73) at 5 µm: a field of 400 x 300 µm. Every
expected value is written from the seed and from ``footprint()``, never read
from the report's own numbers (FM-124); a coordinate is checked within 0.06 µm,
because the page is rounded to a tenth, and a swapped axis or the plan's matrix
moves it by hundreds. ``RuntimeWarning`` is an error, as numpy 2.2 on Python
3.10 warns where 2.5 does not (FM-131).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser

import pytest

from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.report.render import render
from smc.plugins.report.svg import density_text, label_colour, shade
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    Cell,
    DensityResult,
    DensityWell,
    FailedField,
    PlateResult,
    RoutePosition,
    RouteResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

SHAPE = (60, 80)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 5.0
CLASSES = ("single", "doublet", "debris")
PITCH_UM = 9000.0  # the 96-well plate's well spacing
RADIUS_UM = 3200.0  # and its wells' radius
A1 = (1000.0, -500.0)
UPRIGHT: Matrix2 = ((PIXEL_UM, 0.0), (0.0, -PIXEL_UM))
_VOID = {"meta", "br", "hr", "link", "img", "input"}
_SVG_EMPTY = {"circle", "path", "polyline", "rect", "line", "ellipse"}
_TITLE = "Plate report — run.json"


def _turned(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


def _centre(well: str, rotation_deg: float = 0.0) -> tuple[float, float]:
    """A well of the 96-well plate: A1 plus the pitch, turned by the plate (M1 §8), by hand."""
    row, col = ord(well[0]) - ord("A"), int(well[1:]) - 1
    angle = math.radians(rotation_deg)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        A1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


# ------------------------------------------------------------------- the audit


class Audit(HTMLParser):
    """What a reader can verify of the page without a browser: its tags, and whether they balance."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.unbalanced: list[str] = []
        self._open: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, {k: v or "" for k, v in attrs}))
        if tag not in _VOID and tag not in _SVG_EMPTY:
            self._open.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, {k: v or "" for k, v in attrs}))

    def handle_endtag(self, tag: str) -> None:
        if tag in _SVG_EMPTY:
            return
        if self._open and self._open[-1] == tag:
            self._open.pop()
        else:
            self.unbalanced.append(tag)


def audit(text: str) -> Audit:
    parser = Audit()
    parser.feed(text)
    parser.close()
    parser.unbalanced.extend(f"unclosed {tag}" for tag in parser._open)
    return parser


def _find(
    page: Audit, tag: str, cls: str | None = None, **attrs: str
) -> list[dict[str, str]]:
    """The attributes of every ``tag`` (with class ``cls``) whose ``data-*`` attributes match."""
    found = []
    for name, have in page.tags:
        if name != tag:
            continue
        if cls is not None and cls not in have.get("class", "").split():
            continue
        if all(have.get(k.replace("_", "-")) == v for k, v in attrs.items()):
            found.append(have)
    return found


def _numbers(d: str) -> list[float]:
    return [float(token) for token in re.findall(r"-?\d+(?:\.\d+)?", d)]


def _points(d: str) -> list[tuple[float, float]]:
    values = _numbers(d)
    return list(zip(values[0::2], values[1::2], strict=True))


# ----------------------------------------------------------------------- seed


@dataclass(frozen=True)
class Records:
    """What the upstream stages say, kept for the expected values."""

    plan: ScanPlanResult
    scan: ScanResult
    cells: tuple[Cell, ...]
    density: DensityResult
    wells: tuple[str, ...]
    rotation_deg: float


def _cell(
    well: str, dx: float, dy: float, label: str, rotation_deg: float = 0.0
) -> Cell:
    cx, cy = _centre(well, rotation_deg)
    # Built without validation: a plate of 96 wells with 4 000 cells each is
    # 384 000 of them, and the seed is trusted (the tests of ``nan`` do the same).
    return Cell.model_construct(
        well=well,
        x_um=cx + dx,
        y_um=cy + dy,
        label=label,
        score=0.9,
        frame=0,
        box_px=(0.0, 0.0, 1.0, 1.0),
    )


#: ``(well, dx, dy, label)``: the offset from the well's centre. A1 holds seven
#: cells inside (5 single, 2 doublet) and one on the plastic (3 500 µm out).
LAYOUT: list[tuple[str, float, float, str]] = (
    [("A1", -1500.4 + 7.3 * i, 1200.1, "single") for i in range(5)]
    + [("A1", 1400.2, -900.4 + 6.1 * i, "doublet") for i in range(2)]
    + [("A1", 3500.0, 0.0, "single")]
    + [("A2", 900.3 + 5.5 * i, 800.7, "single") for i in range(4)]
    + [("B1", 0.4 + 3.1 * i, 1800.6, "doublet") for i in range(3)]
)


def _inside(dx: float, dy: float) -> bool:
    return math.hypot(dx, dy) <= RADIUS_UM


def _records(
    *,
    wells: Sequence[str] = ("A1", "A2", "B1"),
    layout: Sequence[tuple[str, float, float, str]] = LAYOUT,
    rotation_deg: float = 0.0,
    plate: str = "96-well",
    matrix: Matrix2 = UPRIGHT,
    shape: tuple[int, int] = SHAPE,
    plan_matrix: Matrix2 | None = None,
    plan_shape: tuple[int, int] | None = None,
    area_mm2: float = 0.4,
    failed: Mapping[str, int] | None = None,
    below_grid: Sequence[str] = (),
    rows: int = 2,
    columns: int = 2,
    objective: str | None = "Nikon 10X S Fluor",
    labels: Sequence[str] = CLASSES,
) -> Records:
    """The plan, the scan, the cells and the density of ``wells``; the density is built from the seed.

    ``failed`` maps a well to the number of its first fields that failed (all of
    them: no frame, a density of ``None`` and 0 fields). A well in ``below_grid``
    keeps one frame and has no density either: the imaged area is under the
    grid's resolution (FM-138).
    """
    laid = plan_matrix if plan_matrix is not None else matrix
    laid_shape = plan_shape if plan_shape is not None else shape
    result = PlateResult(
        plate=plate,
        a1_center_xy_um=A1,
        rotation_deg=rotation_deg,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    corners = footprint(0.0, 0.0, matrix=laid, image_shape_px=laid_shape)
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    fov = (max(xs) - min(xs), max(ys) - min(ys))
    plate_plan, fields = plan_fields(
        result,
        list(wells),
        fov_um=fov,
        footprint_of=lambda x, y: footprint(
            x, y, matrix=laid, image_shape_px=laid_shape
        ),
        rows=rows,
        columns=columns,
        overlap=0.1,
        clip_to_well=True,
    )
    plan = ScanPlanResult(
        plate_plan=plate_plan,
        wells=tuple(wells),
        objective=objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=laid,
        matrix_source="parameter",
        image_shape_px=laid_shape,
        fov_um=fov,
        overlap=0.1,
        rows=rows,
        columns=columns,
        fields=fields,
        estimate_s=None,
    )
    frames: list[ScanFrame] = []
    failures: list[FailedField] = []
    to_fail = dict(failed or {})
    kept: dict[str, int] = {}
    for k, planned in enumerate(fields):
        if to_fail.get(planned.well, 0) > 0:
            to_fail[planned.well] -= 1
            failures.append(
                FailedField(
                    well=planned.well,
                    index=planned.index,
                    name=planned.name,
                    x_um=planned.x_um,
                    y_um=planned.y_um,
                    error="the stage did not reach it",
                )
            )
            continue
        if planned.well in below_grid and kept.get(planned.well, 0) >= 1:
            continue
        kept[planned.well] = kept.get(planned.well, 0) + 1
        frames.append(
            ScanFrame(
                well=planned.well,
                index=planned.index,
                name=planned.name,
                file=f"scan/t/{planned.name}.tif",
                # The readback is not the target (the demo rounds a command).
                x_um=planned.x_um + 0.3,
                y_um=planned.y_um - 0.2,
                z_um=1000.0 + 10.0 * k,
                focus="locked",
                pfs_offset=None,
                t_s=float(k),
            )
        )
    scan = ScanResult(
        folder="scan/t",
        objective=objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
        exposure_ms=10.0,
        frames=tuple(frames),
        failed=tuple(failures),
    )
    cells = tuple(
        _cell(well, dx, dy, label, rotation_deg) for well, dx, dy, label in layout
    )
    inside_counts = Counter(
        (name, label) for name, dx, dy, label in layout if _inside(dx, dy)
    )
    outside_counts = Counter(name for name, dx, dy, _ in layout if not _inside(dx, dy))
    reported = []
    for well in wells:
        counts = {label: inside_counts[(well, label)] for label in labels}
        n_fields = kept.get(well, 0)
        unknown = n_fields == 0 or well in below_grid
        reported.append(
            DensityWell(
                well=well,
                fields=n_fields,
                failed_fields=sum(1 for f in failures if f.well == well),
                area_mm2=0.0 if unknown else area_mm2,
                counts=counts,
                outside_well=outside_counts[well],
                per_mm2=None
                if unknown
                else {label: counts[label] / area_mm2 for label in labels},
            )
        )
    density = DensityResult(
        labels=tuple(labels),
        area_grid_um=100.0,
        wells=tuple(reported),
        assumed=("the imaged area is a union of fields on a grid of 100 µm",),
    )
    return Records(plan, scan, cells, density, tuple(wells), rotation_deg)


def _render(records: Records, **overrides: object) -> str:
    """The page, with the arguments ``render`` needs and the ones a test changes."""
    arguments: dict[str, object] = {
        "plan": records.plan,
        "scan": records.scan,
        "cells": records.cells,
        "density": records.density,
        "route": None,
        "title": _TITLE,
        "facts": (("Run document", "run.json"),),
        "assumed": ("scan-plan: the plate is where the registration says",),
        "max_cells_drawn_per_well": 2000,
    }
    arguments.update(overrides)
    return render(**arguments)  # type: ignore[arg-type]


def _page(records: Records, **overrides: object) -> tuple[str, Audit]:
    text = _render(records, **overrides)
    return text, audit(text)


def _wells_of_map(page: Audit, label: str) -> dict[str, dict[str, str]]:
    """The ``g.well`` of the map of ``label``, by well; the map's own ``g`` come between its ``svg`` tags."""
    wells: dict[str, dict[str, str]] = {}
    inside = False
    for tag, attrs in page.tags:
        if tag == "svg":
            inside = (
                "plate-map" in attrs.get("class", "").split()
                and attrs.get("data-label") == label
            )
        elif inside and tag == "g" and "well" in attrs.get("class", "").split():
            wells[attrs["data-well"]] = attrs
    return wells


# ------------------------------------------------------------- the document


def test_the_report_is_one_file_with_no_script_and_no_url() -> None:
    text, page = _page(_records())
    names = {tag for tag, _ in page.tags}
    assert names.isdisjoint({"script", "link", "img", "iframe", "object", "embed"})
    assert "xmlns" not in text
    assert "@import" not in text
    for tag, attrs in page.tags:
        assert "xmlns" not in attrs
        for name, value in attrs.items():
            assert "://" not in value, (tag, name)
            assert not value.startswith("//"), (tag, name)
            assert not value.startswith("data:"), (tag, name)
            assert "url(" not in value, (tag, name)
        if "href" in attrs:
            assert attrs["href"].startswith("#"), attrs
    assert "url(" not in text
    assert page.unbalanced == []
    svgs = [attrs for tag, attrs in page.tags if tag == "svg"]
    assert svgs
    for attrs in svgs:
        assert attrs.get("role") == "img"
        assert attrs.get("aria-label")
        assert attrs.get("viewbox")
        assert "width" not in attrs
        assert "height" not in attrs


def test_the_page_is_utf8_with_the_charset_first() -> None:
    data = _render(_records()).encode("utf-8")
    assert b'<meta charset="utf-8">' in data[:1024]
    assert not data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8")  # strictly: no replacement character, no BOM
    assert "µ" in text
    assert "—" in text
    assert data.startswith(b'<!doctype html><html lang="en"><head><meta charset=')


def test_the_same_records_give_the_same_bytes() -> None:
    records = _records()
    assert _render(records) == _render(records)


# --------------------------------------------------------------- the header


def test_the_header_names_the_plate_the_frame_of_the_scan_and_the_facts() -> None:
    records = _records(plan_matrix=_turned(2.0, 0.0, False), plan_shape=(90, 120))
    text, _ = _page(
        records, facts=(("Profile", "demo"), ("Generated", "2026-10-28 14:05 UTC"))
    )
    assert f"<h1>{_TITLE}</h1>" in text
    assert "<tr><th>Plate</th><td>96-well</td></tr>" in text
    assert "<tr><th>Objective</th><td>Nikon 10X S Fluor</td></tr>" in text
    # The scan's frame (60 x 80 at 5 µm), not the plan's copy (90 x 120 at 2 µm).
    assert "<tr><th>Frame</th><td>60 × 80 px at 5 µm per pixel</td></tr>" in text  # noqa: RUF001
    assert "<tr><th>Wells scanned</th><td>3</td></tr>" in text
    assert "<tr><th>Fields</th><td>12 taken, 0 failed</td></tr>" in text
    # A1 holds 5 single and 2 doublet inside, A2 4 single, B1 3 doublet.
    assert "<tr><th>Cells</th><td>single 9 · doublet 5 · debris 0</td></tr>" in text
    # Then the caller's rows, in order, after the page's own.
    assert text.index("<th>Cells</th>") < text.index("<th>Profile</th>")
    assert text.index("<th>Profile</th>") < text.index("<th>Generated</th>")
    assert "<tr><th>Profile</th><td>demo</td></tr>" in text


def test_the_objective_is_not_recorded_rather_than_guessed() -> None:
    text, _ = _page(_records(objective=None))
    assert "<tr><th>Objective</th><td>not recorded</td></tr>" in text


def test_every_assumed_line_is_listed() -> None:
    text, page = _page(_records(), assumed=("scan-plan: a", "density: b", "route: c"))
    assert (
        '<ul class="assumed"><li>scan-plan: a</li><li>density: b</li>'
        "<li>route: c</li></ul>"
    ) in text
    assert _find(page, "ul", "assumed")


def test_a_scan_that_has_failed_fields_says_so_in_words() -> None:
    text, _ = _page(_records(failed={"A1": 1, "B1": 1}))
    assert (
        '<p class="warn">2 fields failed in 2 wells: their area is not in the '
        "density</p>"
    ) in text
    clean, _ = _page(_records())
    assert 'class="warn"' not in clean


def test_one_failed_field_reads_in_the_singular() -> None:
    text, _ = _page(_records(failed={"A1": 1}))
    assert (
        '<p class="warn">1 field failed in 1 well: its area is not in the density</p>'
    ) in text


# ----------------------------------------------------------------- plate maps


def test_every_scanned_well_appears_with_its_density() -> None:
    records = _records()
    text, page = _page(records)
    rows = _find(page, "tr")
    assert [row["data-well"] for row in rows if "data-well" in row] == [
        "A1",
        "A2",
        "B1",
    ]
    # Hand-computed: the cells inside / the area of 0.4 mm².
    expected = {
        "A1": {"single": 5 / 0.4, "doublet": 2 / 0.4, "debris": 0.0},
        "A2": {"single": 4 / 0.4, "doublet": 0.0, "debris": 0.0},
        "B1": {"single": 0.0, "doublet": 3 / 0.4, "debris": 0.0},
    }
    for label in CLASSES:
        wells = _wells_of_map(page, label)
        assert set(wells) >= {"A1", "A2", "B1"}
        for well in ("A1", "A2", "B1"):
            assert wells[well]["data-state"] == "density"
            assert float(wells[well]["data-value"]) == pytest.approx(
                expected[well][label], abs=0.06
            )
    # The table says the same, whole: A1's single is 12.5 per mm².
    assert "<td>5</td><td>12.5</td><td>2</td><td>5.00</td>" in text


def test_the_plate_map_has_every_well_and_fills_only_the_scanned() -> None:
    _, page = _page(_records())
    wells = _wells_of_map(page, "single")
    assert len(wells) == 96
    scanned = {
        name for name, attrs in wells.items() if attrs["data-state"] != "unscanned"
    }
    assert scanned == {"A1", "A2", "B1"}
    assert [name for name in wells][:3] == [
        "A1",
        "A2",
        "A3",
    ]  # the plate's order, row by row
    assert all("data-value" not in wells[name] for name in wells if name not in scanned)


def _circles(page: Audit, label: str) -> dict[str, tuple[float, float]]:
    """The centre of each well's circle on the map of ``label``, by the ``g`` it is in."""
    centres: dict[str, tuple[float, float]] = {}
    inside, well = False, ""
    for tag, attrs in page.tags:
        if tag == "svg":
            inside = (
                "plate-map" in attrs.get("class", "").split()
                and attrs.get("data-label") == label
            )
        elif inside and tag == "g" and "well" in attrs.get("class", "").split():
            well = attrs["data-well"]
        elif inside and tag == "circle":
            centres[well] = (float(attrs["cx"]), float(attrs["cy"]))
    return centres


def test_the_plate_map_puts_a1_at_the_top_left_whatever_the_plate_is_turned_by() -> (
    None
):
    # The stage's Y is up: row B is below row A in the stage and on the page.
    _, page = _page(_records(rotation_deg=12.0))
    centres = _circles(page, "single")
    assert centres["A1"][0] < centres["A2"][0]  # A2 is to the right of A1
    assert centres["A1"][1] < centres["B1"][1]  # B1 is below A1 (a larger y is lower)
    # By hand: A1 to A2 is 9000 µm along the plate's x, turned by 12 degrees.
    dx = centres["A2"][0] - centres["A1"][0]
    dy = centres["A2"][1] - centres["A1"][1]
    assert dx == pytest.approx(PITCH_UM * math.cos(math.radians(12.0)), abs=0.06)
    # A2 is higher in the stage (+y), which is a smaller y on the page.
    assert dy == pytest.approx(-PITCH_UM * math.sin(math.radians(12.0)), abs=0.06)


def test_one_origin_serves_every_well_of_the_plate() -> None:
    _, page = _page(_records())
    centres = _circles(page, "single")
    # A1 to A2 and A1 to B1: the pitch, to the rounding, in the page's µm.
    assert centres["A2"][0] - centres["A1"][0] == pytest.approx(PITCH_UM, abs=0.06)
    assert centres["B1"][1] - centres["A1"][1] == pytest.approx(PITCH_UM, abs=0.06)
    assert centres["A2"][1] == pytest.approx(centres["A1"][1], abs=0.06)


def test_a_384_well_plate_draws_square_wells_as_paths_turned_with_the_plate() -> None:
    records = _records(plate="384-well", wells=("A1",), rotation_deg=10.0, layout=())
    _, page = _page(records)
    wells = _wells_of_map(page, "single")
    assert len(wells) == 384
    # A square well is a path of four corners, not a circle.
    inside, name, corners = False, "", None
    for tag, attrs in page.tags:
        if tag == "svg":
            inside = (
                "plate-map" in attrs.get("class", "").split()
                and attrs.get("data-label") == "single"
            )
        elif inside and tag == "g" and "well" in attrs.get("class", "").split():
            name = attrs["data-well"]
        elif inside and name == "A1" and tag == "path":
            corners = _points(attrs["d"])
            break
    assert corners is not None
    assert len(corners) == 4
    # The 384-well plate's wells are 3.4 mm wide and square (useq): the side of the
    # polygon is the well's, and its first edge is turned by the plate's 10 degrees.
    (x0, y0), (x1, y1) = corners[0], corners[1]
    assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(3400.0, abs=0.2)
    # Drawn with Y down, so a turn of +10 degrees (counter-clockwise on the stage) is -10.
    assert math.degrees(math.atan2(y1 - y0, x1 - x0)) == pytest.approx(-10.0, abs=0.1)


def test_a_value_is_written_in_a_well_up_to_384_wells_and_not_beyond() -> None:
    small, _ = _page(_records(plate="384-well", wells=("A1", "A2"), layout=()))
    assert small.count("<text") > 0
    assert re.search(r">0</text>", small) is not None
    big, _ = _page(_records(plate="1536-well", wells=("A1", "A2"), layout=()))
    # No text per well on a plate of 1 536: only the legend's own texts (5 + 1 per label).
    assert big.count("<text") == 3 * 6
    assert "A1 — single 0 per mm²" in big  # the tooltip still holds the value


def test_the_density_scale_is_per_label_and_the_largest_well_has_the_full_colour() -> (
    None
):
    text = _render(_records())
    # `single`: the largest is A1 at 12.5 per mm²; `doublet`: B1 at 7.5. Each is
    # the full colour of its label, whatever the other label's numbers.
    # In the well's own markup: the legend's last swatch has that colour too.
    single = shade(label_colour(0), 1.0)
    doublet = shade(label_colour(1), 1.0)
    assert f'fill="{single}"' in _well_markup(_map_section(text, "single"), "A1")
    assert f'fill="{doublet}"' in _well_markup(_map_section(text, "doublet"), "B1")
    # A2's single is 10.0 of 12.5: t = 0.8; the same well's doublet is 0, the paler end.
    a2_single = _well_markup(_map_section(text, "single"), "A2")
    assert f'fill="{shade(label_colour(0), 10.0 / 12.5)}"' in a2_single
    a2_doublet = _well_markup(_map_section(text, "doublet"), "A2")
    assert f'fill="{shade(label_colour(1), 0.0)}"' in a2_doublet
    assert shade(label_colour(1), 0.0) != "#ffffff"
    # A label with no cell anywhere (top == 0): every scanned well is the pale end.
    debris = _map_section(text, "debris")
    assert f'fill="{shade(label_colour(2), 0.0)}"' in _well_markup(debris, "A1")


def _map_section(text: str, label: str) -> str:
    start = text.index(f'<svg class="plate-map" data-label="{label}"')
    return text[start : text.index("</svg>", start)]


def _well_markup(section: str, well: str) -> str:
    start = section.index(f'data-well="{well}"')
    return section[start : section.index("</g>", start)]


def test_the_legend_has_five_swatches_and_the_unit() -> None:
    text, _ = _page(_records())
    section = _map_section(text, "single")
    legend = section[section.rindex("</g>") :]
    assert legend.count("<rect") == 5
    for step in range(5):
        assert f'fill="{shade(label_colour(0), step / 4)}"' in legend
    # 0, 1/4, 1/2, 3/4 and 1 of the largest density, 12.5 per mm².
    for value in (0.0, 12.5 / 4, 12.5 / 2, 12.5 * 3 / 4, 12.5):
        assert f">{density_text(value)}</text>" in legend
    assert ">per mm²</text>" in legend


def test_unknown_density_is_not_zero_and_a_sampled_zero_is_not_unknown() -> None:
    # A1: every field failed, so no frame; A2: one field of an area below the grid;
    # B1: measured.
    records = _records(failed={"A1": 4}, below_grid=("A2",))
    text, page = _page(records)
    wells = _wells_of_map(page, "single")
    assert wells["A1"]["data-state"] == "unknown"
    assert wells["A2"]["data-state"] == "unknown"
    assert wells["B1"]["data-state"] == "density"
    assert wells["B1"]["data-value"] == "0"  # a measured 0
    # The three texts differ, each said whole.
    assert '<td class="unknown">no frame: unknown</td>' in text
    # The apostrophe is escaped, as every text is (it reads as one in a browser).
    assert (
        '<td class="unknown">area below the grid&#x27;s resolution (100 µm)</td>'
    ) in text
    assert "<td>0</td></tr>" in text or "<td>0</td><td>" in text
    row_b1 = text[text.index('<tr data-well="B1">') :]
    assert "<td>0</td>" in row_b1[: row_b1.index("</tr>")]
    assert "unknown" not in row_b1[: row_b1.index("</tr>")]
    # In the map: the unknown well is grey with n/a, the zero is the pale end with its 0.
    section = _map_section(text, "single")
    a1 = _well_markup(section, "A1")
    assert 'fill="#e3e3e3"' in a1
    assert ">n/a</text>" in a1
    assert "A1 — single: no frame: unknown" in a1
    assert ">0</text>" in _well_markup(section, "B1")


def test_a_density_with_one_label_only_still_maps_and_tabulates() -> None:
    records = _records(labels=("single",))
    _, page = _page(records)
    assert len(_find(page, "svg", "plate-map")) == 1


def test_a_well_not_on_the_plate_is_refused_naming_the_stage() -> None:
    records = _records()
    odd = DensityWell(
        well="Z99",
        fields=1,
        failed_fields=0,
        area_mm2=0.1,
        counts={label: 0 for label in CLASSES},
        outside_well=0,
        per_mm2={label: 0.0 for label in CLASSES},
    )
    density = DensityResult(
        labels=records.density.labels,
        area_grid_um=100.0,
        wells=(*records.density.wells, odd),
    )
    with pytest.raises(
        ValueError,
        match=r"^well 'Z99' is not on the 96-well plate; run scan-plan again$",
    ):
        _render(records, density=density)


# ------------------------------------------------------------------- tooltips


def test_a_well_tooltip_holds_the_value_the_counts_the_area_and_the_fields() -> None:
    text, _ = _page(_records())
    assert "A1 — single 12.5 per mm² (5 cells in 0.4 mm², 4 fields)" in text
    assert "A1 — not scanned" not in text
    assert "A4 — not scanned" in text


# ------------------------------------------------------------------ the panels


def _section(page: Audit, well: str) -> list[tuple[str, dict[str, str]]]:
    """The tags of the well's panel: from its ``section`` to the next one."""
    out: list[tuple[str, dict[str, str]]] = []
    inside = False
    for tag, attrs in page.tags:
        if tag == "section" and "well-detail" in attrs.get("class", "").split():
            if inside:
                break
            inside = attrs.get("data-well") == well
        if inside:
            out.append((tag, attrs))
    return out


def _paths(section: list[tuple[str, dict[str, str]]], cls: str) -> list[dict[str, str]]:
    return [
        a for tag, a in section if tag == "path" and cls in a.get("class", "").split()
    ]


def _polygons(d: str) -> list[list[tuple[float, float]]]:
    """Each closed subpath of a ``d``, as its corners."""
    return [_points(sub) for sub in d.split("M") if sub]


def _dots(d: str) -> list[tuple[float, float]]:
    """The dots of a ``path.cells``: the point of each ``M x y h0.1``."""
    return [
        (float(x), float(y)) for x, y in re.findall(r"M(-?[\d.]+) (-?[\d.]+)h0\.1", d)
    ]


Stop = tuple[int, str, float, float, str, dict[str, int]]

#: ``(order, well, dx, dy, name, counts)``: A1 is visited twice, then A2, then B1.
STOPS: list[Stop] = [
    (0, "A1", -1500.0, 1200.0, "A1_r000", {"single": 5}),
    (1, "A1", 1400.0, -900.0, "A1_r001", {"single": 0, "doublet": 2}),
    (2, "A2", 900.0, 800.0, "A2_r000", {"single": 4}),
    (3, "B1", 0.0, 1800.0, "B1_r000", {"doublet": 3}),
]


def _stops(
    rotation_deg: float = 0.0, stops: Sequence[Stop] = STOPS
) -> tuple[RoutePosition, ...]:
    positions = []
    for order, well, dx, dy, name, counts in stops:
        cx, cy = _centre(well, rotation_deg)
        positions.append(
            RoutePosition(
                order=order,
                well=well,
                name=name,
                x_um=cx + dx,
                y_um=cy + dy,
                z_um=None,
                pfs_offset=None,
                counts=counts,
            )
        )
    return tuple(positions)


def _route(
    records: Records,
    *,
    matrix: Matrix2 = ((10.0, 0.0), (0.0, -10.0)),
    shape: tuple[int, int] = SHAPE,
    objective: str | None = "Nikon 20X Plan Fluor ELWD",
    travel_s: float | None = 247.0,
    uncovered: dict[str, int] | None = None,
) -> RouteResult:
    """A route imaged by another objective: the scan's camera at twice the pixel size."""
    return RouteResult(
        objective=objective,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
        fov_um=(800.0, 600.0),
        labels=("single", "doublet"),
        positions=_stops(records.rotation_deg),
        uncovered={"A1": 1} if uncovered is None else uncovered,
        distance_um=12_345.0,
        travel_s=travel_s,
        assumed=("the route minimises the Euclidean distance",),
    )


def test_every_scanned_well_has_a_panel_linked_from_its_map_and_its_row_by_index() -> (
    None
):
    text, page = _page(_records())
    sections = _find(page, "section", "well-detail")
    assert [(s["id"], s["data-well"]) for s in sections] == [
        ("w0", "A1"),
        ("w1", "A2"),
        ("w2", "B1"),
    ]
    links = [a["href"] for a in _find(page, "a") if "href" in a]
    # Each well is linked from its row of the table and from its entry in each of the
    # three maps; an unscanned well is not linked at all.
    assert links.count("#w0") == 1 + 3
    assert links.count("#w2") == 1 + 3
    assert len(links) == 3 * 4
    assert '<td><a href="#w1">A2</a></td>' in text


def test_footprints_use_the_scans_matrix_and_frame_not_the_plans() -> None:
    # FM-124: the plan was laid for another camera and frame (2 µm, 90 x 120);
    # the scan turned by 30 degrees, mirrored, on a frame of 60 x 80.
    matrix = _turned(5.0, 30.0, True)
    records = _records(
        matrix=matrix, plan_matrix=_turned(2.0, 0.0, False), plan_shape=(90, 120)
    )
    _, page = _page(records)
    frames = _paths(_section(page, "A1"), "frames")
    assert len(frames) == 1
    in_a1 = [f for f in records.scan.frames if f.well == "A1"]
    assert frames[0]["data-n"] == str(len(in_a1))
    drawn = _polygons(frames[0]["d"])
    assert len(drawn) == len(in_a1)
    cx, cy = _centre("A1")
    for polygon, frame in zip(drawn, in_a1, strict=True):
        # The scan's matrix and shape at the *readback* position, from the well's centre.
        expected = footprint(
            frame.x_um, frame.y_um, matrix=matrix, image_shape_px=SHAPE
        )
        assert len(polygon) == 4
        for (px, py), (x, y) in zip(polygon, expected, strict=True):
            assert px == pytest.approx(x - cx, abs=0.06)
            assert py == pytest.approx(cy - y, abs=0.06)
    # The readback is not the plan's field: a corner drawn at the target would be 0.3 off.
    planned = records.plan.fields[0]
    assert (in_a1[0].x_um, in_a1[0].y_um) != (planned.x_um, planned.y_um)


def test_the_cells_drawn_are_the_cells_density_counted() -> None:
    text, page = _page(_records())
    section = _section(page, "A1")
    cells = {a["data-label"]: a for a in _paths(section, "cells")}
    assert set(cells) == set(CLASSES)
    # A1 holds 5 single and 2 doublet inside; one single is on the plastic: not drawn.
    assert (cells["single"]["data-drawn"], cells["single"]["data-total"]) == ("5", "5")
    assert (cells["doublet"]["data-drawn"], cells["doublet"]["data-total"]) == (
        "2",
        "2",
    )
    assert (cells["debris"]["data-drawn"], cells["debris"]["data-total"]) == ("0", "0")
    expected = sorted(
        (dx, -dy)
        for well, dx, dy, label in LAYOUT
        if well == "A1" and label == "single" and _inside(dx, dy)
    )
    drawn = sorted(_dots(cells["single"]["d"]))
    assert len(drawn) == len(expected) == 5
    for (px, py), (x, y) in zip(drawn, expected, strict=True):
        assert px == pytest.approx(x, abs=0.06)
        assert py == pytest.approx(y, abs=0.06)
    # The cell on the plastic is not drawn, and the caption gives its number, whole.
    assert not any(abs(px - 3500.0) < 1.0 for px, _ in _dots(cells["single"]["d"]))
    assert (
        '<p class="caption">4 fields (0 failed) · 7 of 7 cells drawn · '
        "1 detection outside the well shape is not drawn</p>"
    ) in text
    # A2 has nothing on the plastic: no sentence about it.
    assert '<p class="caption">4 fields (0 failed) · 4 of 4 cells drawn</p>' in text


def test_a_dot_is_a_round_capped_subpath_of_the_diameter_the_well_asks_for() -> None:
    _, page = _page(_records())
    single = _paths(_section(page, "A1"), "cells")[0]
    assert single["stroke-linecap"] == "round"
    assert single["fill"] == "none"
    assert single["stroke"] == label_colour(0)
    # max(0.012 x 6400, 12) = 76.8, to the tenth.
    assert single["stroke-width"] == "76.8"
    assert re.fullmatch(r"(M-?[\d.]+ -?[\d.]+h0\.1)+", single["d"])


def _grid(n: int, spacing: float = 70.0) -> list[tuple[float, float]]:
    """The first ``n`` points of a lattice that lie well inside a well of 3 200 µm radius."""
    points = []
    for i in range(90):
        for j in range(90):
            dx, dy = -3100.0 + spacing * i + 0.37, -3100.0 + spacing * j + 0.37
            if math.hypot(dx, dy) <= 3100.0:
                points.append((dx, dy))
    return points[:n]


def test_the_cells_drawn_are_capped_and_the_caption_says_so() -> None:
    points = _grid(5000)
    assert len(points) == 5000
    layout = [
        ("A1", dx, dy, "single" if k % 2 == 0 else "doublet")
        for k, (dx, dy) in enumerate(points)
    ]
    records = _records(wells=("A1",), layout=layout)
    text, page = _page(records, max_cells_drawn_per_well=1000)
    cells = {a["data-label"]: a for a in _paths(_section(page, "A1"), "cells")}
    drawn = int(cells["single"]["data-drawn"]) + int(cells["doublet"]["data-drawn"])
    total = int(cells["single"]["data-total"]) + int(cells["doublet"]["data-total"])
    assert (drawn, total) == (1000, 5000)
    assert (
        '<p class="caption">4 fields (0 failed) · 1000 of 5000 cells drawn '
        "(a regular subsample)</p>"
    ) in text
    # The cells at (i * 5000) // 1000 = 5 i, in the order of the file: written from the spec.
    chosen = [layout[(i * 5000) // 1000] for i in range(1000)]
    for label in ("single", "doublet"):
        want = sorted((dx, -dy) for _, dx, dy, lab in chosen if lab == label)
        have = sorted(_dots(cells[label]["d"]))
        assert len(have) == len(want)
        for (px, py), (x, y) in zip(have, want, strict=True):
            assert px == pytest.approx(x, abs=0.06)
            assert py == pytest.approx(y, abs=0.06)
    # The same records give the same bytes.
    assert _render(records, max_cells_drawn_per_well=1000) == text


def test_a_cap_of_zero_draws_no_cell_and_says_so() -> None:
    text, page = _page(_records(), max_cells_drawn_per_well=0)
    for path in _paths(_section(page, "A1"), "cells"):
        assert path["data-drawn"] == "0"
        assert path["d"] == ""
    assert (
        '<p class="caption">4 fields (0 failed) · 0 of 7 cells drawn '
        "(none: the cap per well is 0) · 1 detection outside the well shape "
        "is not drawn</p>"
    ) in text


def test_the_total_of_cells_drawn_is_bounded_whatever_the_cap() -> None:
    # FM-74: a cap per well times 96 wells is no bound. 4 000 cells in each of the
    # 96 wells, at the largest cap the parameter allows.
    names = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]
    points = _grid(4000, spacing=60.0)
    assert len(points) == 4000
    layout = [(well, dx, dy, "single") for well in names for dx, dy in points]
    records = _records(
        wells=names, layout=layout, rows=1, columns=1, labels=("single",)
    )
    text, page = _page(records, max_cells_drawn_per_well=20_000)
    paths = _find(page, "path", "cells")
    dots = [int(a["data-drawn"]) for a in paths]
    assert len(dots) == 96
    assert sum(dots) <= 300_000
    # Each well draws its share: 300 000 // 96 = 3 125.
    assert set(dots) == {3125}
    assert sum(len(_dots(a["d"])) for a in paths) == 96 * 3125
    assert len(text.encode("utf-8")) < 8_000_000


# ---------------------------------------------------------------------- failed


def test_a_failed_field_is_drawn_dashed_and_named_in_the_header() -> None:
    # The scan's camera is turned and mirrored, the plan's is not (FM-124).
    matrix = _turned(5.0, 30.0, True)
    records = _records(failed={"A1": 1}, matrix=matrix, plan_matrix=UPRIGHT)
    text, page = _page(records)
    failed = _paths(_section(page, "A1"), "failed")
    assert len(failed) == 1
    assert failed[0]["data-n"] == "1"
    # At the target, with the scan's matrix and frame: the field the scan could not take.
    target = records.scan.failed[0]
    cx, cy = _centre("A1")
    (polygon,) = _polygons(failed[0]["d"])
    expected = footprint(target.x_um, target.y_um, matrix=matrix, image_shape_px=SHAPE)
    for (px, py), (x, y) in zip(polygon, expected, strict=True):
        assert px == pytest.approx(x - cx, abs=0.06)
        assert py == pytest.approx(cy - y, abs=0.06)
    assert "1 field failed in 1 well: its area is not in the density" in text
    assert '<p class="caption">3 fields (1 failed) · 7 of 7 cells drawn' in text
    # A well that lost none has no such path.
    assert _paths(_section(page, "A2"), "failed") == []


def test_a_well_whose_every_field_failed_has_no_frames_and_says_unknown() -> None:
    text, page = _page(_records(failed={"A1": 4}))
    (frames,) = _paths(_section(page, "A1"), "frames")
    assert frames["data-n"] == "0"
    assert frames["d"] == ""
    assert '<p class="counts">single 5 (no frame: unknown)' in text
    assert _wells_of_map(page, "single")["A1"]["data-state"] == "unknown"
    assert "4 fields failed in 1 well: their area is not in the density" in text


def test_the_panel_is_the_wells_circumscribed_square_plus_a_tenth() -> None:
    # The 96-well plate's wells are 6 400 µm: half 3 200, and the fields of the
    # seed stay inside it, so the view is 3 200 x 1.1 on each side.
    _, page = _page(_records())
    (panel,) = [a for t, a in _section(page, "A1") if t == "svg"]
    assert panel["viewbox"] == "-3520 -3520 7040 7040"


def test_the_panel_grows_to_hold_a_footprint_that_reaches_past_the_wall() -> None:
    # One field per well, centred. At 120 µm per pixel a frame of 60 x 80 px is
    # 9 600 x 7 200 µm: its corner is 4 800 µm from the centre (plus the 0.3 µm
    # by which the stage's readback differs from the target), past the 3 200 µm wall.
    huge = ((120.0, 0.0), (0.0, -120.0))
    records = _records(
        matrix=huge, plan_matrix=UPRIGHT, plan_shape=SHAPE, rows=1, columns=1
    )
    _, page = _page(records)
    (panel,) = [a for t, a in _section(page, "A1") if t == "svg"]
    _, _, w, h = _numbers(panel["viewbox"])
    assert w == h == pytest.approx(2.2 * 4800.3, abs=0.2)
    # A footprint inside the wall does not shrink the view below the well's square.
    small = ((50.0, 0.0), (0.0, -50.0))
    records = _records(
        matrix=small, plan_matrix=UPRIGHT, plan_shape=SHAPE, rows=1, columns=1
    )
    _, page = _page(records)
    (panel,) = [a for t, a in _section(page, "A1") if t == "svg"]
    assert panel["viewbox"] == "-3520 -3520 7040 7040"


# ----------------------------------------------------------------------- route


def test_route_fields_are_drawn_in_the_routes_frame() -> None:
    records = _records()
    route = _route(
        records
    )  # the scan's camera at twice the pixel size: another objective
    text, page = _page(records, route=route)
    section = _section(page, "A1")
    fields = _paths(section, "field")
    assert [f["data-order"] for f in fields] == ["0", "1"]
    assert [f["data-name"] for f in fields] == ["A1_r000", "A1_r001"]
    cx, cy = _centre("A1")
    for drawn, stop in zip(fields, route.positions[:2], strict=True):
        (polygon,) = _polygons(drawn["d"])
        expected = footprint(
            stop.x_um, stop.y_um, matrix=route.matrix_um_per_px, image_shape_px=SHAPE
        )
        for (px, py), (x, y) in zip(polygon, expected, strict=True):
            assert px == pytest.approx(x - cx, abs=0.06)
            assert py == pytest.approx(cy - y, abs=0.06)
        # The route's frame is 800 µm wide; the scan's would be 400.
        assert max(px for px, _ in polygon) - min(px for px, _ in polygon) == (
            pytest.approx(800.0, abs=0.2)
        )
    assert "<title>A1_r000: 5 single</title>" in text
    assert "<title>A1_r001: 0 single, 2 doublet</title>" in text
    # The numbers are order + 1, across the whole page, in the order of the route.
    assert re.findall(r'class="order"[^>]*>(\d+)</text>', text) == ["1", "2", "3", "4"]
    (line,) = [a for t, a in section if t == "polyline" and a["class"] == "route-well"]
    pairs = _points(line["points"].replace(",", " "))
    assert len(pairs) == 2
    for (px, py), stop in zip(pairs, route.positions[:2], strict=True):
        assert px == pytest.approx(stop.x_um - cx, abs=0.06)
        assert py == pytest.approx(cy - stop.y_um, abs=0.06)
    # A well with one stop has its field and no line of its own.
    assert [a for t, a in _section(page, "A2") if t == "polyline"] == []
    assert len(_paths(_section(page, "A2"), "field")) == 1


def test_the_route_is_drawn_in_the_order_of_its_stops_whatever_the_list_order() -> None:
    records = _records()
    route = _route(records)
    shuffled = route.model_copy(update={"positions": tuple(reversed(route.positions))})
    text, page = _page(records, route=shuffled)
    cx, cy = _centre("A1")
    (line,) = [
        a
        for t, a in _section(page, "A1")
        if t == "polyline" and a["class"] == "route-well"
    ]
    pairs = _points(line["points"].replace(",", " "))
    # A1_r000 first, then A1_r001, as the stops' own order says.
    for (px, py), stop in zip(pairs, _stops()[:2], strict=True):
        assert px == pytest.approx(stop.x_um - cx, abs=0.06)
        assert py == pytest.approx(cy - stop.y_um, abs=0.06)
    assert [f["data-order"] for f in _paths(_section(page, "A1"), "field")] == [
        "0",
        "1",
    ]
    # The whole route: the start is the first stop (green), whatever the list says.
    (route_line,) = _find(page, "polyline", "route")
    pairs = _points(route_line["points"].replace(",", " "))
    first, second = _stops()[:2]
    assert pairs[1][0] - pairs[0][0] == pytest.approx(
        second.x_um - first.x_um, abs=0.11
    )
    assert re.findall(r'class="order"[^>]*>(\d+)</text>', text) == ["1", "2", "3", "4"]


def test_the_report_without_a_route_says_so() -> None:
    text, page = _page(_records())
    assert (
        '<h2>Route</h2><p class="note">no route: run route to add the fields '
        "to image</p>"
    ) in text
    assert _find(page, "svg", "route-map") == []
    assert _find(page, "path", "field") == []
    assert _find(page, "polyline") == []
    # And no column for what a route leaves out.
    assert "Not in a chosen field" not in text


def test_the_route_map_gives_distance_and_travel_time_or_says_there_is_none() -> None:
    records = _records()
    text, page = _page(records, route=_route(records))
    (line,) = _find(page, "polyline", "route")
    assert line["data-stops"] == "4"
    # 4 stops in 3 wells; 12 345 µm is 12.3 mm; 247 s is 4 min 7 s.
    assert (
        '<p class="route-summary">4 stops in 3 wells · 12.3 mm · 4 min 7 s</p>' in text
    )
    assert (
        '<p class="note">1 cell of the wanted labels (single, doublet) is in no '
        "chosen field.</p>"
    ) in text
    # The ranks of the visited wells, in the order of their first stop.
    assert re.findall(r'class="rank"[^>]*>(\d+)</text>', text) == ["1", "2", "3"]
    markers = [a["class"] for t, a in page.tags if t == "circle" and "class" in a]
    assert "start" in markers
    assert "end" in markers
    # The line goes through the stops in order, in the plate's one frame.
    pairs = _points(line["points"].replace(",", " "))
    assert len(pairs) == 4
    first, second = _stops()[:2]
    assert pairs[1][0] - pairs[0][0] == pytest.approx(
        second.x_um - first.x_um, abs=0.11
    )
    assert pairs[1][1] - pairs[0][1] == pytest.approx(
        first.y_um - second.y_um, abs=0.11
    )
    # The table has the column the route adds.
    assert "<th>Not in a chosen field (single, doublet)</th>" in text
    unknown, _ = _page(records, route=_route(records, travel_s=None))
    assert (
        '<p class="route-summary">4 stops in 3 wells · 12.3 mm · not estimated: '
        "no timing model</p>"
    ) in unknown


def test_the_header_names_the_objective_the_route_was_imaged_with_when_it_differs() -> (
    None
):
    records = _records()
    text, _ = _page(records, route=_route(records))
    assert (
        "<tr><th>Route imaged with</th><td>Nikon 20X Plan Fluor ELWD</td></tr>" in text
    )
    same, _ = _page(records, route=_route(records, objective="Nikon 10X S Fluor"))
    assert "Route imaged with" not in same
    unrecorded, _ = _page(records, route=_route(records, objective=None))
    assert "<tr><th>Route imaged with</th><td>not recorded</td></tr>" in unrecorded


# --------------------------------------------------------------------- escaping

_HOSTILE = "<img src=x onerror=alert(1)>\"'"


def _classes_and_ids(page: Audit) -> tuple[set[str], set[str]]:
    classes: set[str] = set()
    ids: set[str] = set()
    for _, attrs in page.tags:
        classes.update(attrs.get("class", "").split())
        if "id" in attrs:
            ids.add(attrs["id"])
    return classes, ids


def test_text_with_markup_is_escaped_wherever_it_is_shown() -> None:
    label = 'single"><b>'
    layout = [
        (w, dx, dy, label if lab == "single" else lab) for w, dx, dy, lab in LAYOUT
    ]
    records = _records(
        layout=layout, labels=(label, "doublet", "debris"), objective=_HOSTILE
    )
    route = _route(records, objective=_HOSTILE + "!")  # differs: it is shown
    renamed = route.positions[0].model_copy(update={"name": _HOSTILE})
    route = route.model_copy(
        update={
            "positions": (renamed, *route.positions[1:]),
            "labels": (label, "doublet"),
        }
    )
    text, page = _page(
        records,
        route=route,
        title=_HOSTILE,
        facts=((_HOSTILE, _HOSTILE),),
        assumed=(_HOSTILE,),
    )
    escaped = "&lt;img src=x onerror=alert(1)&gt;&quot;&#x27;"
    # <title> and <h1>, both sides of a fact, an assumed line, the objective and the
    # route's, a stop's name (in its attribute and its tooltip).
    assert text.count(escaped) == 9
    assert "<img" not in text
    assert [t for t, _ in page.tags if t in {"img", "b"}] == []
    assert not any("onerror" in attrs for _, attrs in page.tags)
    assert "single&quot;&gt;&lt;b&gt;" in text
    assert page.unbalanced == []
    # Nothing from a record became an id or a class: the same sets as a benign page.
    benign = _records()
    _, clean = _page(benign, route=_route(benign))
    assert _classes_and_ids(page) == _classes_and_ids(clean)
    # The data attributes carry the label, escaped on the way out and whole on the way back.
    assert {a["data-label"] for a in _find(page, "path", "cells")} == {
        label,
        "doublet",
        "debris",
    }


# --------------------------------------------------------------------- refusals


def _refused(records: Records, message: str, **overrides: object) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _render(records, **overrides)


def _cell_at(x_um: float) -> Cell:
    return Cell.model_construct(
        well="A1",
        x_um=x_um,
        y_um=0.0,
        label="single",
        score=0.9,
        frame=0,
        box_px=(0.0, 0.0, 1.0, 1.0),
    )


def test_a_non_finite_coordinate_is_refused_naming_the_stage() -> None:
    nan = float("nan")
    records = _records()
    # The plate's registration: useq accepts nan (FM-114); a copy skips validation.
    bad_plate = records.plan.plate_plan.model_copy(update={"a1_center_xy": (nan, 0.0)})
    _refused(
        records,
        "the plate's A1 centre is not a finite number; run scan-plan again",
        plan=records.plan.model_copy(update={"plate_plan": bad_plate}),
    )
    # A frame, a failed field, a cell and a stop, built without validation.
    frame = records.scan.frames[0].model_copy(update={"x_um": nan})
    _refused(
        records,
        "a frame's position is not a finite number; run scan again",
        scan=records.scan.model_copy(
            update={"frames": (frame, *records.scan.frames[1:])}
        ),
    )
    failing = _records(failed={"A1": 1})
    lost = failing.scan.failed[0].model_copy(update={"y_um": nan})
    _refused(
        failing,
        "a failed field's position is not a finite number; run scan again",
        scan=failing.scan.model_copy(update={"failed": (lost,)}),
    )
    _refused(
        records,
        "a cell's position is not a finite number; run cells again",
        cells=(_cell_at(nan), *records.cells),
    )
    route = _route(records)
    stop = route.positions[0].model_copy(update={"x_um": nan})
    _refused(
        records,
        "a stop's position is not a finite number; run route again",
        route=route.model_copy(update={"positions": (stop, *route.positions[1:])}),
    )


def test_a_camera_matrix_that_is_not_finite_names_the_stage_that_holds_it() -> None:
    inf = float("inf")
    records = _records()
    _refused(
        records,
        "a camera matrix is not a finite number; run scan again",
        scan=records.scan.model_copy(
            update={"matrix_um_per_px": ((inf, 0.0), (0.0, -5.0))}
        ),
    )
    _refused(
        records,
        "a camera matrix is not a finite number; run route again",
        route=_route(records).model_copy(
            update={"matrix_um_per_px": ((10.0, 0.0), (0.0, inf))}
        ),
    )


def test_a_coordinate_beyond_1e12_um_is_refused_not_printed() -> None:
    # f"{1e300:.1f}" is 303 characters: a position that far is a corrupt record.
    records = _records()
    frame = records.scan.frames[0].model_copy(update={"x_um": 1e300})
    _refused(
        records,
        "a frame's position is beyond 1e12 µm; run scan again",
        scan=records.scan.model_copy(
            update={"frames": (frame, *records.scan.frames[1:])}
        ),
    )
    _refused(
        records,
        "a cell's position is beyond 1e12 µm; run cells again",
        cells=(_cell_at(1e13), *records.cells),
    )


def test_a_route_stop_in_a_well_that_is_not_on_the_plate_is_refused() -> None:
    records = _records()
    route = _route(records)
    stop = route.positions[0].model_copy(update={"well": "Z99"})
    _refused(
        records,
        "the route has a stop in well 'Z99', which is not on the 96-well plate; "
        "run route again",
        route=route.model_copy(update={"positions": (stop, *route.positions[1:])}),
    )
