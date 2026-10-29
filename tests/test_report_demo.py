"""``report`` through ``run_plugin``: the records of a scan in, one HTML file out (m3 §13, #195).

The chain is the real one: the test seeds the plate, the scan plan, the scan and
the cells as an earlier run would have left them (as ``tests/test_route_demo.py``
does: 60 x 80 pixels, not square, FM-73, at 5 µm), then ``density`` and ``route``
run through ``run_plugin`` on the demo stand, so the report reads genuine
results and not hand-written ones. The page is read back with ``html.parser``.
Every expected value is computed here from the seed and from ``footprint()``,
never from the report's own numbers (FM-124).
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from smc.hardware import Microscope
from smc.plugins import PluginRunError, load, run_plugin
from smc.plugins.density.plugin import DensityParams, DensityPlugin
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.report import plugin as plugin_module
from smc.plugins.report.plugin import ReportParams, ReportPlugin
from smc.plugins.route.plugin import RouteParams, RoutePlugin
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    Cell,
    CellsFile,
    CellsResult,
    DensityResult,
    PlateResult,
    ReportResult,
    RouteResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.run.files import AtomicWriteError

pytestmark = pytest.mark.demo

SHAPE = (60, 80)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 5.0
IN_PLACE = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
CLASSES = ("single", "doublet", "debris")
WELLS = ("A1", "A2", "B1")
#: Where the 96-well plate's wells are, with A1 at (1000, -500): 9 mm apart, 6.4 mm wide.
CENTRES = {"A1": (1000.0, -500.0), "A2": (10000.0, -500.0), "B1": (1000.0, -9500.0)}
SHA = "abcdef0123456789" * 4
UPRIGHT: Matrix2 = ((PIXEL_UM, 0.0), (0.0, -PIXEL_UM))
_VOID = {"meta", "br", "hr", "link", "img", "input"}
_SVG_EMPTY = {"circle", "path", "polyline", "rect", "line", "ellipse"}

#: ``(well, dx, dy, label)``, the offset from the well's centre. Clusters are a few µm
#: across and over a field apart, so each is held whole by one field.
LAYOUT: list[tuple[str, float, float, str]] = (
    [("A1", -1500.37 + 7.3 * i, 1200.11, "single") for i in range(5)]
    + [("A1", 1400.21, -900.4 + 6.1 * i, "doublet") for i in range(3)]
    + [("A1", 2.9 * i + 0.7, 2200.8, "single") for i in range(2)]
    + [("A1", -1000.5, -2000.3, "single")]
    + [("A2", 900.3 + 5.5 * i, 800.7, "single") for i in range(4)]
    + [("A2", 20.3 + 4.1 * i, -15.2, "single") for i in range(3)]
    + [("A2", -1500.6, -1500.2 + 3.3 * i, "debris") for i in range(2)]
    + [("B1", 0.4 + 3.1 * i, 1800.6, "doublet") for i in range(2)]
    + [("B1", -2500.2, 0.7, "single")]
)


def _turned(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


# ----------------------------------------------------------------------- audit


class Audit(HTMLParser):
    """The tags of the page and whether they balance: what a reader can verify without a browser."""

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


def _audit(text: str) -> Audit:
    parser = Audit()
    parser.feed(text)
    parser.close()
    parser.unbalanced.extend(f"unclosed {tag}" for tag in parser._open)
    return parser


def _find(page: Audit, tag: str, cls: str | None = None) -> list[dict[str, str]]:
    return [
        attrs
        for name, attrs in page.tags
        if name == tag and (cls is None or cls in attrs.get("class", "").split())
    ]


def _points(d: str) -> list[tuple[float, float]]:
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
    return list(zip(values[0::2], values[1::2], strict=True))


# ------------------------------------------------------------------------ seed


@dataclass(frozen=True)
class Seed:
    """What the upstream records say, kept for the expected values."""

    plan: ScanPlanResult
    scan: ScanResult
    cells: tuple[Cell, ...]
    result: CellsResult
    matrix: Matrix2
    cells_file: Path


def _record(path: Path, plugin: str, provides: str, result: BaseModel) -> None:
    """A stage record as an earlier run of ``plugin`` would have left it, ``done``."""
    then = datetime(2026, 10, 28, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _extent(matrix: Matrix2, shape: tuple[int, int]) -> tuple[float, float]:
    corners = footprint(0.0, 0.0, matrix=matrix, image_shape_px=shape)
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _seed(
    tmp_path: Path,
    *,
    matrix: Matrix2 = UPRIGHT,
    plan_matrix: Matrix2 | None = None,
    plan_shape: tuple[int, int] | None = None,
    scan_objective: str | None = IN_PLACE,
) -> Seed:
    """The plate, the plan, the scan and the cells of ``WELLS``, in ``tmp_path/run.json``.

    ``plan_matrix`` and ``plan_shape`` are the camera the plan was laid for when
    it is not the scan's: the report must draw the scan's (FM-124). The frames are
    at the planned fields, their readback a little off the target.
    """
    path = tmp_path / "run.json"
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=CENTRES["A1"],
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    laid = plan_matrix if plan_matrix is not None else matrix
    laid_shape = plan_shape if plan_shape is not None else SHAPE
    fov = _extent(laid, laid_shape)
    plate_plan, fields = plan_fields(
        plate,
        WELLS,
        fov_um=fov,
        footprint_of=lambda x, y: footprint(
            x, y, matrix=laid, image_shape_px=laid_shape
        ),
        rows=2,
        columns=2,
        overlap=0.1,
        clip_to_well=True,
    )
    plan = ScanPlanResult(
        plate_plan=plate_plan,
        wells=WELLS,
        objective=scan_objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=laid,
        matrix_source="parameter",
        image_shape_px=laid_shape,
        fov_um=fov,
        overlap=0.1,
        rows=2,
        columns=2,
        fields=fields,
        estimate_s=None,
        assumed=("the plate is where the registration says",),
    )
    frames = tuple(
        ScanFrame(
            well=f.well,
            index=f.index,
            name=f.name,
            file=f"scan/t/{f.name}.tif",
            x_um=f.x_um + 0.3,
            y_um=f.y_um - 0.2,
            z_um=1000.0 + 10.0 * k,
            focus="locked",
            pfs_offset=2000.0 + 10.0 * k,
            t_s=float(k),
        )
        for k, f in enumerate(fields)
    )
    scan = ScanResult(
        folder="scan/t",
        objective=scan_objective,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        image_shape_px=SHAPE,
        exposure_ms=10.0,
        frames=frames,
        failed=(),
        assumed=("the frames' Z is the focus drive's reading",),
    )
    cells = tuple(
        Cell(
            well=well,
            x_um=CENTRES[well][0] + dx,
            y_um=CENTRES[well][1] + dy,
            label=label,
            score=0.9,
            frame=0,
            box_px=(0.0, 0.0, 1.0, 1.0),
        )
        for well, dx, dy, label in LAYOUT
    )
    (tmp_path / "cells").mkdir(exist_ok=True)
    name = "cells/t1.json"
    (tmp_path / name).write_text(
        CellsFile(cells=cells).model_dump_json(), encoding="utf-8"
    )
    counts = {
        well: {
            label: sum(1 for c in cells if c.well == well and c.label == label)
            for label in CLASSES
        }
        for well in WELLS
    }
    result = CellsResult(
        file=name,
        model="m.pth",
        model_sha256=SHA,
        classes=CLASSES,
        device="cpu",
        score_threshold=0.5,
        merge_radius_um=5.0,
        frames=len(frames),
        counts=counts,
        merged=0,
        seconds_per_frame=None,
        assumed=("cells within 5 µm seen by two fields are counted once",),
    )
    _record(path, "plate-registration", "plate", plate)
    _record(path, "scan-plan", "scan-plan", plan)
    _record(path, "scan", "scan", scan)
    _record(path, "cells", "cells", result)
    return Seed(plan, scan, cells, result, matrix, tmp_path / name)


def _chain(
    microscope: Microscope,
    tmp_path: Path,
    *,
    route: bool = True,
    route_objective: str | None = None,
    **seed: Any,
) -> Seed:
    """Seed the records, then run ``density`` and ``route`` as the operator would."""
    seeded = _seed(tmp_path, **seed)
    run_doc = tmp_path / "run.json"
    run_plugin(DensityPlugin(), microscope, DensityParams(), run_doc)
    if route:
        run_plugin(
            RoutePlugin(),
            microscope,
            RouteParams(labels=("single", "doublet"), objective=route_objective),
            run_doc,
        )
    return seeded


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        ReportPlugin(),
        microscope,
        ReportParams(**params),
        tmp_path / "run.json",
        stop=stop,
    )


def _result(record: StageRecord) -> ReportResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return ReportResult.model_validate(record.result)


def _stored(tmp_path: Path, stage: str = "report") -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages[stage]


def _page_of(tmp_path: Path, record: StageRecord) -> str:
    return (tmp_path / _result(record).file).read_bytes().decode("utf-8")


def _refused(microscope: Microscope, tmp_path: Path, **params: Any) -> PluginRunError:
    """Run and expect a ``PluginRunError``; the stored record is ``failed`` and has no result.

    Raised outside the handler that caught its cause (FM-98), so that it holds
    neither the file's error nor the microscope's frames.
    """
    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, **params)
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    assert not list((tmp_path / "report").glob("*.html"))
    return info.value


def _edit(tmp_path: Path, stage: str, **changes: Any) -> None:
    """Change fields of a stored result behind the runner's back, as a hand edit or a replaced file would."""
    document = RunDocument.load(tmp_path / "run.json")
    stored = document.stages[stage]
    assert stored.result is not None
    stored.result.update(changes)
    write_stage(tmp_path / "run.json", stored)


# ---------------------------------------------------------------- registration


def test_the_entry_point_loads_report() -> None:
    entry = load("report")

    assert entry.plugin.provides == "report"
    assert entry.plugin.requires == ()
    assert entry.plugin.reads == ("scan-plan", "scan", "cells", "density")
    assert entry.plugin.reads_optional == ("route",)
    assert entry.target == "smc.plugins.report.plugin:ReportPlugin"


# ------------------------------------------------------------------ the result


def test_report_through_the_runner_writes_one_file_and_names_it(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _chain(demo_microscope, tmp_path)

    first = _run(demo_microscope, tmp_path)

    name = _result(first).file
    assert re.fullmatch(r"report/[0-9a-f]{12}\.html", name)
    assert "\\" not in name  # POSIX on every OS
    page = tmp_path / name
    assert page.is_file()
    text = page.read_bytes().decode("utf-8")  # strictly: no BOM, no cp1252
    assert text.startswith("<!doctype html>")
    assert _stored(tmp_path) == first
    before = page.read_bytes()

    second = _run(demo_microscope, tmp_path)

    # A `done` result's file is never rewritten: the next run has its own.
    assert _result(second).file != name
    assert (tmp_path / _result(second).file).is_file()
    assert page.read_bytes() == before
    assert len(list((tmp_path / "report").glob("*.html"))) == 2


def test_a_dry_run_writes_the_report_like_any_other(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    # Nothing in the report needs the stand.
    _chain(demo_microscope_dry, tmp_path)

    record = _run(demo_microscope_dry, tmp_path)

    assert (tmp_path / _result(record).file).is_file()
    assert record.dry_run is True


def test_the_report_holds_the_numbers_of_the_stages_it_read(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _chain(demo_microscope, tmp_path, route_objective=FORTY)
    density = DensityResult.model_validate(_stored(tmp_path, "density").result)
    route = RouteResult.model_validate(_stored(tmp_path, "route").result)
    assert route.matrix_um_per_px != seed.matrix  # another objective, another field

    text = _page_of(tmp_path, _run(demo_microscope, tmp_path))
    page = _audit(text)

    # The table: one row per well, with the counts and densities of the record.
    for reported in density.wells:
        row = re.search(rf'<tr data-well="{reported.well}">(.*?)</tr>', text)
        assert row is not None
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row.group(1))
        assert cells[1] == str(reported.fields)
        assert cells[2] == str(reported.failed_fields)
        assert float(cells[3]) == pytest.approx(reported.area_mm2, abs=0.006)
        assert reported.per_mm2 is not None
        for i, label in enumerate(density.labels):
            assert cells[4 + 2 * i] == str(reported.counts[label])
            assert float(cells[5 + 2 * i]) == pytest.approx(
                reported.per_mm2[label], rel=0.006, abs=0.006
            )
    # The map: the densities again, to the rounding of the hook.
    first = next(
        a
        for a in _find(page, "g", "well")
        if a["data-well"] == "A1" and a["data-state"] == "density"
    )
    assert density.wells[0].per_mm2 is not None
    assert float(first["data-value"]) == pytest.approx(
        density.wells[0].per_mm2["single"], abs=0.06
    )
    # The route: one rectangle per stop, in the route's matrix and frame, not the scan's.
    fields = _find(page, "path", "field")
    assert len(fields) == len(route.positions) > 0
    stop = route.positions[0]
    drawn = next(f for f in fields if f["data-name"] == stop.name)
    cx, cy = CENTRES[stop.well]
    corners = footprint(
        stop.x_um,
        stop.y_um,
        matrix=route.matrix_um_per_px,
        image_shape_px=route.image_shape_px,
    )
    # The first path of the section of that well: the polygon is relative to its centre.
    for (px, py), (x, y) in zip(_points(drawn["d"]), corners, strict=True):
        assert px == pytest.approx(x - cx, abs=0.06)
        assert py == pytest.approx(cy - y, abs=0.06)
    assert page.unbalanced == []


def test_the_header_names_the_run_document_the_profile_the_detector_and_every_assumed_line_by_stage(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _chain(demo_microscope, tmp_path)

    text = _page_of(tmp_path, _run(demo_microscope, tmp_path))

    assert "<title>Plate report — run.json</title>" in text
    assert "<tr><th>Run document</th><td>run.json</td></tr>" in text
    assert "<tr><th>Profile</th><td>demo</td></tr>" in text
    assert re.search(
        r"<tr><th>Generated</th><td>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC</td></tr>", text
    )
    assert (
        "<tr><th>Detector</th><td>m.pth (sha256 abcdef012345), score ≥ 0.5, "
        "merge radius 5 µm, cpu</td></tr>"
    ) in text
    # Every line of every stage, with its stage in front, in the stage's order.
    lines = [
        "<li>scan-plan: the plate is where the registration says</li>",
        "<li>scan: the frames&#x27; Z is the focus drive&#x27;s reading</li>",
        "<li>cells: cells within 5 µm seen by two fields are counted once</li>",
    ]
    for line in lines:
        assert line in text
    positions = [text.index(line) for line in lines]
    assert positions == sorted(positions)
    assert "<li>density: the imaged area is the union of the scan" in text
    assert (
        "<li>route: the route minimises the Euclidean distance between fields" in text
    )
    assert text.index("<li>density:") < text.index("<li>route:")


def test_no_path_is_in_the_report(demo_microscope: Microscope, tmp_path: Path) -> None:
    # FM-50: a report is attached to issues; a path names the person's account.
    _chain(demo_microscope, tmp_path)

    text = _page_of(tmp_path, _run(demo_microscope, tmp_path))

    assert str(tmp_path) not in text
    assert tmp_path.as_posix() not in text
    assert tmp_path.name not in text
    assert tmp_path.parent.name not in text
    assert "run.json" in text


def test_the_scans_matrix_and_shape_reach_the_page(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-124 at the seam: the plan was laid for another camera and frame; the scan's
    # is turned by 30 degrees and mirrored. The footprint the page shows is the scan's.
    matrix = _turned(PIXEL_UM, 30.0, True)
    seed = _chain(
        demo_microscope,
        tmp_path,
        route=False,
        matrix=matrix,
        plan_matrix=_turned(2.0, 0.0, False),
        plan_shape=(90, 120),
    )
    assert seed.plan.image_shape_px != seed.scan.image_shape_px

    text = _page_of(tmp_path, _run(demo_microscope, tmp_path))
    page = _audit(text)

    section = text[text.index('<section class="well-detail" id="w0"') :]
    frames = re.search(r'<path class="frames" data-n="(\d+)" d="([^"]*)"', section)
    assert frames is not None
    in_a1 = [f for f in seed.scan.frames if f.well == "A1"]
    assert int(frames.group(1)) == len(in_a1)
    cx, cy = CENTRES["A1"]
    drawn = [_points(sub) for sub in frames.group(2).split("M") if sub]
    for polygon, frame in zip(drawn, in_a1, strict=True):
        expected = footprint(
            frame.x_um, frame.y_um, matrix=matrix, image_shape_px=SHAPE
        )
        for (px, py), (x, y) in zip(polygon, expected, strict=True):
            assert px == pytest.approx(x - cx, abs=0.06)
            assert py == pytest.approx(cy - y, abs=0.06)
    assert page.unbalanced == []


def test_the_cap_reaches_the_page(demo_microscope: Microscope, tmp_path: Path) -> None:
    _chain(demo_microscope, tmp_path, route=False)

    whole = _audit(_page_of(tmp_path, _run(demo_microscope, tmp_path)))
    capped = _audit(
        _page_of(tmp_path, _run(demo_microscope, tmp_path, max_cells_drawn_per_well=2))
    )

    def drawn(page: Audit, well: int) -> int:
        paths = _find(page, "path", "cells")
        per_well = len(CLASSES)  # one path per label, in the order of the wells
        return sum(
            int(p["data-drawn"]) for p in paths[well * per_well : (well + 1) * per_well]
        )

    # A1 holds 11 cells; the cap of 2 leaves 2 (A2 holds 9, B1 3: 2 as well).
    assert [drawn(whole, k) for k in range(3)] == [11, 9, 3]
    assert [drawn(capped, k) for k in range(3)] == [2, 2, 2]


def test_the_route_is_optional(demo_microscope: Microscope, tmp_path: Path) -> None:
    _chain(demo_microscope, tmp_path, route=False)

    record = _run(demo_microscope, tmp_path)

    text = _page_of(tmp_path, record)
    assert '<p class="note">no route: run route to add the fields to image</p>' in text
    page = _audit(text)
    assert _find(page, "svg", "route-map") == []
    assert _find(page, "path", "field") == []


def test_a_route_that_was_run_on_other_labels_than_the_cells_hold_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _chain(demo_microscope, tmp_path)
    _edit(tmp_path, "route", labels=["x"])

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the route has the label 'x', which the cells stage did not detect; "
        "run route again"
    )


def test_a_route_stop_in_a_well_density_does_not_report_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _chain(demo_microscope, tmp_path)
    route = _stored(tmp_path, "route").result
    assert route is not None
    stops = [dict(stop) for stop in route["positions"]]
    stops[0]["well"] = "H12"
    _edit(tmp_path, "route", positions=stops)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the route has a stop in well 'H12', which density does not report; "
        "run route again"
    )


# --------------------------------------------------------- records that disagree


def test_a_density_that_does_not_match_the_cells_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-111: each record validates; together they describe two scans.
    _chain(demo_microscope, tmp_path, route=False)
    stored = _stored(tmp_path, "density").result
    assert stored is not None
    wells = [dict(w) for w in stored["wells"]]

    # One count more than the file holds: A1 has 11 cells.
    changed = [dict(w) for w in wells]
    changed[0]["counts"] = {**changed[0]["counts"], "single": 9}
    _edit(tmp_path, "density", wells=changed)
    assert _refused(demo_microscope, tmp_path).reason == (
        "density counted 12 cells in well 'A1' and the cells file holds 11; "
        "run density again"
    )

    # One field more than the scan holds: A1 has 4 frames.
    changed = [dict(w) for w in wells]
    changed[0]["fields"] = 5
    _edit(tmp_path, "density", wells=changed)
    assert _refused(demo_microscope, tmp_path).reason == (
        "density reports 5 fields for well 'A1' and the scan holds 4; run density again"
    )

    # A failed field the scan does not have.
    changed = [dict(w) for w in wells]
    changed[0]["failed_fields"] = 1
    _edit(tmp_path, "density", wells=changed)
    assert _refused(demo_microscope, tmp_path).reason == (
        "density reports 1 failed fields for well 'A1' and the scan holds 0; "
        "run density again"
    )

    # Other labels than the cells stage detected.
    _edit(tmp_path, "density", wells=wells, labels=["doublet", "single", "debris"])
    assert _refused(demo_microscope, tmp_path).reason == (
        "density counted the labels ['doublet', 'single', 'debris'] and the cells "
        "stage detected ['single', 'doublet', 'debris']; run density again"
    )


def test_a_cell_in_a_well_density_does_not_report_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _chain(demo_microscope, tmp_path, route=False)
    stored = _stored(tmp_path, "density").result
    assert stored is not None
    # B1 dropped from the density, and the cells file and its result still hold B1's.
    _edit(tmp_path, "density", wells=stored["wells"][:2])

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the cells file holds a cell in well 'B1', which density does not report; "
        "run density again"
    )


def test_a_cells_file_that_no_longer_matches_says_to_run_cells_again(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seed = _chain(demo_microscope, tmp_path, route=False)
    # The first of A1's 8 single cells removed from the file behind the result.
    first = next(c for c in seed.cells if c.well == "A1" and c.label == "single")
    fewer = tuple(c for c in seed.cells if c is not first)
    seed.cells_file.write_text(
        CellsFile(cells=fewer).model_dump_json(), encoding="utf-8"
    )

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cells file cells/t1.json differs from the result at A1/single "
        "(the file holds 7, the result counted 8); run cells again"
    )


# ------------------------------------------------------------------ the write


def test_a_write_that_fails_names_the_file_and_says_to_run_report_again(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _chain(demo_microscope, tmp_path, route=False)
    monkeypatch.setattr(plugin_module, "new_token", lambda: "abc123abc123")

    def disk_full(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(OSError(28, "No space left on device"))

    monkeypatch.setattr(plugin_module, "write_atomically", disk_full)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cannot write report/abc123abc123.html (No space left on device); "
        "free some space or check the run folder, then run report again"
    )
    assert str(tmp_path) not in refused.reason


def test_a_write_that_only_failed_to_rename_names_the_kept_file(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the rename failed: the whole new page waits in a hidden file, which the error names (FM-38)."""
    _chain(demo_microscope, tmp_path, route=False)
    monkeypatch.setattr(plugin_module, "new_token", lambda: "abc123abc123")
    kept = tmp_path / "report" / ".abc123abc123.html.1234abcd.smc-tmp"

    def rename_refused(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(PermissionError(13, "Access is denied"), kept=kept)

    monkeypatch.setattr(plugin_module, "write_atomically", rename_refused)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cannot write report/abc123abc123.html (Access is denied); the whole report "
        "is kept as report/.abc123abc123.html.1234abcd.smc-tmp; close any program "
        "that holds report/abc123abc123.html, then run report again"
    )


def test_a_stop_set_before_the_run_leaves_no_file(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _chain(demo_microscope, tmp_path, route=False)
    stop = threading.Event()
    stop.set()

    record = _run(demo_microscope, tmp_path, stop=stop)

    assert record.status is StageStatus.stopped
    assert _stored(tmp_path).status is StageStatus.stopped
    assert not (tmp_path / "report").exists()


def test_a_stop_during_the_render_still_leaves_no_file(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's Stop arrives while the page is being built: the write is the
    # point of no return, so the stage ends before it.
    _chain(demo_microscope, tmp_path, route=False)
    stop = threading.Event()
    real = plugin_module.render

    def stopped_while_rendering(**kwargs: Any) -> str:
        page = real(**kwargs)
        stop.set()
        return page

    monkeypatch.setattr(plugin_module, "render", stopped_while_rendering)

    record = _run(demo_microscope, tmp_path, stop=stop)

    assert record.status is StageStatus.stopped
    assert not (tmp_path / "report").exists()


def test_a_coordinate_that_is_not_a_number_ends_the_stage_naming_the_stage_to_run_again(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-114: a position of `1e300` in the stored scan would print 303 characters.
    _chain(demo_microscope, tmp_path, route=False)
    stored = _stored(tmp_path, "scan").result
    assert stored is not None
    frames = [dict(f) for f in stored["frames"]]
    frames[0]["x_um"] = 1e300
    _edit(tmp_path, "scan", frames=frames)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == "a frame's position is beyond 1e12 µm; run scan again"


def test_the_parameter_is_bounded_where_it_enters() -> None:
    with pytest.raises(ValueError, match="less than or equal to 20000"):
        ReportParams(max_cells_drawn_per_well=20_001)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ReportParams(max_cells_drawn_per_well=-1)
    assert ReportParams().max_cells_drawn_per_well == 2000
