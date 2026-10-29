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
from collections.abc import Sequence
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
    return Cell(
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
    failed_wells: Sequence[str] = (),
    unknown_wells: dict[str, int] | None = None,
    rows: int = 2,
    columns: int = 2,
    objective: str | None = "Nikon 10X S Fluor",
    labels: Sequence[str] = CLASSES,
) -> Records:
    """The plan, the scan, the cells and the density of ``wells``; the density is built from the seed.

    A well in ``unknown_wells`` has no density: the number is its ``fields``
    (0: no frame at all, 1: one field of an area below the grid). A well in
    ``failed_wells`` has one failed field besides its frames.
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
    failed: list[FailedField] = []
    seen_failed: set[str] = set()
    for k, planned in enumerate(fields):
        if planned.well in failed_wells and planned.well not in seen_failed:
            seen_failed.add(planned.well)
            failed.append(
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
        if unknown_wells is not None and planned.well in unknown_wells:
            continue
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
        failed=tuple(failed),
    )
    cells = tuple(
        _cell(well, dx, dy, label, rotation_deg) for well, dx, dy, label in layout
    )
    in_frames: dict[str, int] = {}
    for frame in frames:
        in_frames[frame.well] = in_frames.get(frame.well, 0) + 1
    reported = []
    for well in wells:
        counts = {
            label: sum(
                1
                for name, dx, dy, lab in layout
                if name == well and lab == label and _inside(dx, dy)
            )
            for label in labels
        }
        outside = sum(
            1 for name, dx, dy, _ in layout if name == well and not _inside(dx, dy)
        )
        n_fields = in_frames.get(well, 0)
        failed_in = sum(1 for f in failed if f.well == well)
        if unknown_wells is not None and well in unknown_wells:
            reported.append(
                DensityWell(
                    well=well,
                    fields=unknown_wells[well],
                    failed_fields=failed_in,
                    area_mm2=0.0,
                    counts=counts,
                    outside_well=outside,
                    per_mm2=None,
                )
            )
            continue
        reported.append(
            DensityWell(
                well=well,
                fields=n_fields,
                failed_fields=failed_in,
                area_mm2=area_mm2,
                counts=counts,
                outside_well=outside,
                per_mm2={label: counts[label] / area_mm2 for label in labels},
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
    text, _ = _page(_records(failed_wells=("A1", "B1")))
    assert (
        '<p class="warn">2 fields failed in 2 wells: their area is not in the '
        "density</p>"
    ) in text
    clean, _ = _page(_records())
    assert 'class="warn"' not in clean


def test_one_failed_field_reads_in_the_singular() -> None:
    text, _ = _page(_records(failed_wells=("A1",)))
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
    # A1: no frame at all; A2: one field of an area below the grid; B1: measured.
    records = _records(unknown_wells={"A1": 0, "A2": 1})
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
