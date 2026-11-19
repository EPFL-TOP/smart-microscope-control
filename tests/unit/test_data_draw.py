"""``smc.data.draw``: the drawing of a scan's wells, with no stand (algorithms design §6, #568).

Pixels are read with Pillow's ``getpixel`` at positions computed here from
``Canvas.px_of`` and ``smc.plugins.geometry.footprint``, never from the code under
test (FM-124). Every colour check is at a footprint's edge, a wall point or a marker,
at least 5 pixels from any text, because a text has a black stroke.
"""

from __future__ import annotations

import errno
import itertools
import math
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageDraw, ImageFont

import smc.data.scan as scan_module
from select_fields_seed import (
    PIXEL_UM,
    SHAPE,
    UPRIGHT,
    Seed,
    cell_at,
    load_script,
    matrix_of,
    record,
    seed_run,
)
from smc.data import DatasetError, Scan, draw, load_scan
from smc.plugins.geometry import Matrix2, footprint, stage_of_pixel
from smc.plugins.slots import (
    Cell,
    RoutePosition,
    RouteResult,
    ScanFrame,
    WellGeometry,
    load_frame,
    well_geometry,
)
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
from smc.run.files import AtomicWriteError

HEIGHT, WIDTH = SHAPE
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
LAMBDA = "\N{GREEK SMALL LETTER LAMDA}"

SMALL_WELL = WellGeometry(
    well="A1",
    center_um=(1000.0, -500.0),
    size_um=(1000.0, 1000.0),
    circular=True,
    rotation_deg=0.0,
)


def _pixel_of(canvas: draw.Canvas, x_um: float, y_um: float) -> tuple[int, int]:
    """The canvas pixel of a stage point, from the design's formula ``x = x_min + (c + 0.5) s``."""
    return (
        round((x_um - canvas.x_min_um) / canvas.um_per_px - 0.5),
        round((canvas.y_max_um - y_um) / canvas.um_per_px - 0.5),
    )


def _render(
    *,
    stops: list[tuple[int | str, float, float]],
    caption: str = "A1: test",
    grey_level: int = draw.UNPLACED_GREY,
    cells: list[Cell] | None = None,
    classes: tuple[str, ...] = ("single", "doublet", "debris"),
    max_px: int = 800,
    matrix: Matrix2 = UPRIGHT,
) -> tuple[draw.Canvas, Image.Image]:
    canvas = draw.canvas_of(SMALL_WELL, max_px)
    grey = np.full((canvas.height_px, canvas.width_px), grey_level, dtype=np.uint8)
    image = draw.render_png(
        grey,
        canvas,
        SMALL_WELL,
        cells=cells or [],
        classes=classes,
        wanted=("single",),
        stops=stops,
        matrix=matrix,
        image_shape_px=SHAPE,
        caption=caption,
    )
    return canvas, image


def _cell(dx: float, label: str) -> Cell:
    return Cell(
        well="A1",
        x_um=1000.0 + dx,
        y_um=-500.0,
        label=label,
        score=0.9,
        frame=0,
        box_px=(0.0, 0.0, 10.0, 10.0),
    )


def _count(
    image: Image.Image, colour: tuple[int, int, int], box: tuple[int, ...]
) -> int:
    left, top, right, bottom = box
    return sum(
        image.getpixel((c, r)) == colour
        for c in range(left, right)
        for r in range(top, bottom)
    )


def _clear_green(image: Image.Image, box: tuple[int, ...]) -> int:
    """Pixels of ``box`` that are green-dominant: a glyph is anti-aliased, so it is not always pure green."""
    left, top, right, bottom = box
    found = 0
    for c in range(left, right):
        for r in range(top, bottom):
            red, green, blue = image.getpixel((c, r))
            found += green > 150 and red < 60 and blue < 60
    return found


# ------------------------------------------------------------------ the script


def test_the_script_draws_through_the_kit_module() -> None:
    script = load_script()
    for name in (
        "render_png",
        "canvas_of",
        "place_frame",
        "block_mean",
        "reduction_factor",
        "stretch_to_uint8",
    ):
        assert getattr(script, name) is getattr(draw, name), name
    assert script.frame_problem is draw.frame_problem
    assert script.UNPLACED_GREY == draw.UNPLACED_GREY


# --------------------------------------------------------------------- the text


def test_a_non_ascii_text_is_written_as_an_escape() -> None:
    assert draw.ascii_text(f"Plan Apo {LAMBDA} 20x") == "Plan Apo \\u03bb 20x"
    assert draw.ascii_text("plain") == "plain"
    # A lone surrogate (a name read from a damaged file) does not raise.
    assert draw.ascii_text("a\ud800b") == "a\\ud800b"


def _measure() -> tuple[ImageDraw.ImageDraw, ImageFont.FreeTypeFont]:
    image = Image.new("RGB", (10, 10))
    return ImageDraw.Draw(image), ImageFont.load_default(size=14)


def test_wrap_text_breaks_at_spaces_and_no_line_is_wider_than_the_width() -> None:
    pen, font = _measure()
    text = "the route chose five stops in this well and covered most of its cells"
    width = 160.0
    lines = draw.wrap_text(pen, text, font, width)
    assert len(lines) > 1
    assert " ".join(lines) == text, "words are neither lost nor cut"
    assert all(pen.textlength(line, font=font) <= width for line in lines)
    # Greedy: the first word of a line would not have fitted on the line before.
    for before, after in itertools.pairwise(lines):
        first = after.split(" ")[0]
        assert pen.textlength(f"{before} {first}", font=font) > width


def test_wrap_text_keeps_a_newline_as_a_break_and_an_empty_text_as_one_line() -> None:
    pen, font = _measure()
    assert draw.wrap_text(pen, "one\ntwo", font, 500.0) == ["one", "two"]
    assert draw.wrap_text(pen, "", font, 500.0) == [""]


def test_wrap_text_cuts_a_word_wider_than_the_picture_and_loses_nothing() -> None:
    pen, font = _measure()
    word = "x" * 400
    width = 120.0
    lines = draw.wrap_text(pen, f"a {word} b", font, width)
    assert "".join(lines).replace(" ", "") == f"a{word}b"
    assert all(pen.textlength(line, font=font) <= width for line in lines)
    assert len(lines) >= 400 * pen.textlength("x", font=font) / width
    # Each cut piece is as long as the width allows: one more character would not fit.
    for piece in lines[1:-1]:
        assert pen.textlength(piece + "x", font=font) > width


def test_wrap_text_ends_when_the_width_is_smaller_than_one_glyph() -> None:
    pen, font = _measure()
    lines = draw.wrap_text(pen, "abc", font, 0.5)
    assert lines == ["a", "b", "c"]


def test_wrap_text_of_a_word_of_a_hundred_thousand_characters_is_cut_not_hung() -> None:
    pen, font = _measure()
    lines = draw.wrap_text(pen, "y" * 100_000, font, 300.0)
    assert "".join(lines) == "y" * 100_000
    assert max(len(line) for line in lines) < 200


# ----------------------------------------------------------------- render_png


@pytest.mark.parametrize("number", ["3", 3, "2 (0.823)"])
def test_a_stop_number_may_be_text_and_is_drawn_at_the_footprints_centre(
    number: int | str,
) -> None:
    canvas, image = _render(stops=[(number, 1000.0, -500.0)])
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    # The number is green, with the glyphs' own anti-aliasing: some pixel near the
    # centre is a clear green and the picture without the stop has none.
    near = (column - 20, row - 8, column + 20, row + 8)
    _, bare = _render(stops=[])
    assert _clear_green(image, near) > 0
    assert _clear_green(bare, near) == 0


def test_a_longer_label_is_drawn_wider_than_a_short_one() -> None:
    canvas, short = _render(stops=[("1", 1000.0, -500.0)])
    _, long = _render(stops=[("1 (0.823)", 1000.0, -500.0)])
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    # Past the glyph of "1" and well inside the footprint, whose edges are 150 pixels away.
    beyond = (column + 12, row - 8, column + 40, row + 8)
    assert _clear_green(short, beyond) == 0
    assert _clear_green(long, beyond) > 0


def test_a_text_has_a_black_stroke_so_that_it_reads_on_a_white_mosaic() -> None:
    canvas, image = _render(stops=[("7", 1000.0, -500.0)], grey_level=255, caption="")
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    around = (column - 15, row - 15, column + 15, row + 15)
    _, bare = _render(stops=[], grey_level=255, caption="")
    assert _count(image, BLACK, around) > 0
    assert _count(bare, BLACK, around) == 0
    # The caption has it too: black pixels in the corner, none on the bare mosaic.
    _, captioned = _render(stops=[], grey_level=255, caption="A1: test")
    assert _count(captioned, BLACK, (0, 0, 90, 24)) > 0
    assert _count(bare, BLACK, (0, 0, 90, 24)) == 0


@pytest.mark.parametrize("far", [1e300, 1e9, -1e9, math.inf, math.nan])
def test_a_number_far_from_the_picture_is_not_drawn_and_raises_nothing(
    far: float,
) -> None:
    # Pillow raises SystemError for a text at 1e300 (measured); the other stop stays.
    canvas, image = _render(stops=[("1", far, -500.0), ("2", 1000.0, -500.0)])
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    assert _clear_green(image, (column - 20, row - 8, column + 20, row + 8)) > 0
    assert image.size == (canvas.width_px, canvas.height_px)


def test_the_colours_of_the_classes_are_named_pairs_in_the_cells_stages_order() -> None:
    names = [name for name, _ in draw.CLASS_COLOURS]
    assert names == ["gold", "cyan", "red", "light green", "magenta", "white"]
    assert draw.CLASS_COLOURS[0][1] == (255, 215, 0)
    canvas, image = _render(
        stops=[],
        cells=[_cell(-100.0, "single"), _cell(0.0, "doublet"), _cell(100.0, "debris")],
    )
    for dx, (_, rgb) in zip((-100.0, 0.0, 100.0), draw.CLASS_COLOURS[:3], strict=True):
        column, row = _pixel_of(canvas, 1000.0 + dx, -500.0)
        assert image.getpixel((column, row)) == rgb


def test_a_class_past_the_sixth_takes_the_first_colour_again() -> None:
    classes = ("a", "b", "c", "d", "e", "f", "g")
    canvas, image = _render(stops=[], classes=classes, cells=[_cell(0.0, "g")])
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    assert image.getpixel((column, row)) == draw.CLASS_COLOURS[0][1]


def test_a_non_ascii_caption_does_not_raise_on_the_bitmap_font_and_is_shown_as_an_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What a Pillow built without FreeType gives: its bitmap font refuses a Greek letter.
    def bitmap(size: object = None) -> object:
        return ImageFont.load_default_imagefont()

    monkeypatch.setattr(ImageFont, "load_default", bitmap)
    with pytest.raises(UnicodeEncodeError):
        ImageDraw.Draw(Image.new("RGB", (40, 20))).text(
            (0, 0),
            LAMBDA,
            font=ImageFont.load_default_imagefont(),
        )
    _, image = _render(stops=[], caption=f"Plan Apo {LAMBDA} 20x")
    assert image.size == (800, 800)


def test_a_long_caption_is_wrapped_to_the_picture_and_a_huge_one_is_cut() -> None:
    sentence = "the route chose stops and the selector kept these cells in view; "
    _, image = _render(stops=[], caption=sentence * 6, max_px=300)
    assert image.size[0] <= 300
    # Wrapped: the text goes down the picture and never reaches its right edge.
    right = image.size[0]
    assert _count(image, (255, 255, 255), (right - 6, 0, right, 120)) == 0
    assert _count(image, (255, 255, 255), (0, 40, 120, 80)) > 0
    # A caption of a million characters is cut and the picture is made.
    _, huge = _render(stops=[], caption="z" * 1_000_000, max_px=300)
    assert huge.size[0] <= 300


def test_the_stops_of_the_script_are_drawn_as_before() -> None:
    # A footprint's four edges are green at their midpoints, for a stop with an int number.
    canvas, image = _render(stops=[(0, 1000.0, -500.0)], caption="")
    corners = footprint(1000.0, -500.0, matrix=UPRIGHT, image_shape_px=SHAPE)
    for k in range(4):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
        column, row = _pixel_of(canvas, (ax + bx) / 2.0, (ay + by) / 2.0)
        around = {
            image.getpixel((column + dc, row + dr))
            for dc in (-1, 0, 1)
            for dr in (-1, 0, 1)
        }
        assert GREEN in around, k


# ---------------------------------------------------------------------- Mosaic


def _noise(frame: ScanFrame) -> np.ndarray:
    """A frame that is different at every pixel and from every other frame: nothing a placement error could hide in."""
    rng = np.random.default_rng(frame.index + 1)
    return rng.integers(100, 4000, size=SHAPE).astype(np.uint16)


def test_the_mosaic_is_what_the_scripts_read_pixels_makes(tmp_path: Path) -> None:
    matrix = matrix_of(PIXEL_UM, 30.0, True)
    seed = seed_run(
        tmp_path, cells=[], rows=2, columns=2, matrix=matrix, frame_pixels=_noise
    )
    geometry = well_geometry(seed.plan.plate_plan, "A1")
    expected = (
        load_script()
        .read_pixels(tmp_path, seed.scan, {"A1": geometry}, mosaic=True, png_max_px=300)
        .grey["A1"]
    )
    mosaic = draw.Mosaic(
        geometry,
        max_px=300,
        pixel_size_um=seed.scan.pixel_size_um,
        matrix_um_per_px=matrix,
        image_shape_px=SHAPE,
    )
    for frame in seed.scan.frames:
        mosaic.add(load_frame(tmp_path, frame), frame.x_um, frame.y_um)
    assert mosaic.reduction > 1, "the frames are reduced, so the tile's matrix matters"
    assert np.array_equal(mosaic.grey(), expected)
    # Not a blank canvas: four frames placed, the rest of the well dark grey.
    placed = mosaic.grey() != draw.UNPLACED_GREY
    assert 100 < placed.sum() < placed.size


def test_a_mosaics_tile_matrix_is_the_cameras_times_the_reduction() -> None:
    matrix = matrix_of(PIXEL_UM, 30.0, True)
    mosaic = draw.Mosaic(
        SMALL_WELL,
        max_px=100,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=matrix,
        image_shape_px=SHAPE,
    )
    k = mosaic.reduction
    assert k == round(mosaic.canvas.um_per_px / PIXEL_UM)
    for row in range(2):
        for col in range(2):
            assert mosaic.tile_matrix_um_per_px[row][col] == pytest.approx(
                matrix[row][col] * k
            )
    # The canvas is the well's: the frames never grow it (FM-136).
    assert mosaic.canvas == draw.canvas_of(SMALL_WELL, 100)


def test_a_frame_far_from_the_well_places_nothing_and_raises_nothing() -> None:
    mosaic = draw.Mosaic(
        SMALL_WELL,
        max_px=100,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=UPRIGHT,
        image_shape_px=SHAPE,
    )
    before = mosaic.grey().copy()
    frame = np.full(SHAPE, 3000, dtype=np.uint16)
    for far in (1e9, -1e9, 1e300):
        mosaic.add(frame, far, -500.0)
    assert np.array_equal(mosaic.grey(), before)
    assert mosaic.grey().shape == (mosaic.canvas.height_px, mosaic.canvas.width_px)


@pytest.mark.parametrize("pixel_size_um", [0.0, -1.0, math.nan, math.inf])
def test_a_pixel_size_that_is_not_a_size_is_refused_not_defaulted(
    pixel_size_um: float,
) -> None:
    with pytest.raises(ValueError, match="pixel size"):
        draw.Mosaic(
            SMALL_WELL,
            max_px=100,
            pixel_size_um=pixel_size_um,
            matrix_um_per_px=UPRIGHT,
            image_shape_px=SHAPE,
        )


def test_a_camera_matrix_that_cannot_be_inverted_is_refused_when_a_frame_is_added() -> (
    None
):
    mosaic = draw.Mosaic(
        SMALL_WELL,
        max_px=100,
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=((1.0, 1.0), (1.0, 1.0)),
        image_shape_px=SHAPE,
    )
    with pytest.raises(ValueError, match="cannot be inverted"):
        mosaic.add(np.zeros(SHAPE, dtype=np.uint16), 1000.0, -500.0)


# ------------------------------------------------------------------- the route


def _stop(
    order: int,
    well: str,
    index: int,
    x_um: float = 0.0,
    y_um: float = 0.0,
    *,
    score: float | None = None,
) -> RoutePosition:
    return RoutePosition(
        order=order,
        well=well,
        name=f"{well}_r{index:03d}",
        x_um=x_um,
        y_um=y_um,
        z_um=None,
        pfs_offset=None,
        counts={"single": 1},
        score=score,
    )


def _route(
    positions: tuple[RoutePosition, ...],
    *,
    labels: tuple[str, ...] = ("single", "doublet"),
    uncovered: dict[str, int] | None = None,
    objective: str | None = "2-Plan Fluor 10x",
    matrix: Matrix2 = UPRIGHT,
    shape: tuple[int, int] = SHAPE,
    fov_um: tuple[float, float] = (400.0, 300.0),
) -> RouteResult:
    return RouteResult(
        objective=objective,
        matrix_um_per_px=matrix,
        image_shape_px=shape,
        fov_um=fov_um,
        labels=labels,
        positions=positions,
        uncovered=uncovered if uncovered is not None else {},
        distance_um=0.0,
        travel_s=None,
    )


def _write_route(
    path: Path,
    result: RouteResult | dict[str, Any],
    *,
    plugin: str = "route",
    params: dict[str, Any] | None = None,
    day: int = 29,
    profile: str = "demo",
    dry_run: bool = False,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A ``route`` record as a run of ``plugin`` left it, ``done``, after the seed's stages (day 28)."""
    then = datetime(2026, 10, day, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides="route",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params=params if params is not None else {},
            result=(
                result if isinstance(result, dict) else result.model_dump(mode="json")
            ),
            reads=reads or {},
        ),
    )


def _scan_of_run(
    tmp_path: Path,
    *,
    route: RouteResult | dict[str, Any] | None = None,
    cells: list[Cell] | None = None,
    wells: tuple[str, ...] = ("A1", "B2"),
    **route_options: Any,
) -> Scan:
    """A scan seeded in ``tmp_path`` and loaded back by ``load_scan``, with a ``route`` record when ``route`` is given."""
    seed = seed_run(
        tmp_path,
        cells=cells if cells is not None else [cell_at("A1", 0.0, 0.0)],
        wells=wells,
        rows=1,
        columns=1,
    )
    if route is not None:
        _write_route(seed.path, route, **route_options)
    return load_scan(seed.path)


CHOSE = {"selector": "most-cells", "orderer": "nearest"}


def test_a_stop_is_numbered_by_its_rank_in_its_well_not_by_its_plate_wide_order(
    tmp_path: Path,
) -> None:
    # The demo-sample's shape: the order is plate-wide, so B2 starts at 3, not at 0.
    positions = (
        _stop(0, "A1", 0),
        _stop(1, "A1", 1),
        _stop(2, "A1", 2),
        _stop(3, "B2", 0),
        _stop(4, "B2", 1),
    )
    scan = _scan_of_run(tmp_path, route=_route(positions), params=CHOSE)
    view, why = draw.route_of(scan)
    assert why is None
    assert view is not None
    second = draw.stops_of_well(view, "B2")
    assert [(s.rank, s.order, s.name) for s in second] == [
        (1, 3, "B2_r000"),
        (2, 4, "B2_r001"),
    ]
    assert [s.rank for s in draw.stops_of_well(view, "A1")] == [1, 2, 3]
    # A well typed as a person types it is the same well.
    assert draw.stops_of_well(view, "b02") == second
    assert draw.stops_of_well(view, "A2") == ()


def test_the_rank_follows_the_visiting_order_and_equal_orders_keep_the_records() -> (
    None
):
    result = _route((_stop(7, "A1", 0), _stop(5, "A1", 1), _stop(6, "A1", 2)))
    view = draw.RouteView("route", None, None, result)
    by_order = draw.stops_of_well(view, "A1")
    assert [(s.name, s.rank) for s in by_order] == [
        ("A1_r001", 1),
        ("A1_r002", 2),
        ("A1_r000", 3),
    ]
    tied = _route((_stop(4, "A1", 0), _stop(4, "A1", 1), _stop(3, "A1", 2)))
    ranks = draw.stops_of_well(draw.RouteView("route", None, None, tied), "A1")
    assert [s.name for s in ranks] == ["A1_r002", "A1_r000", "A1_r001"]


def test_a_stop_keeps_the_score_the_selector_gave_it() -> None:
    result = _route((_stop(0, "A1", 0, score=0.5), _stop(1, "A1", 1)))
    stops = draw.stops_of_well(draw.RouteView("route", None, None, result), "A1")
    assert [s.score for s in stops] == [0.5, None]


def test_a_score_follows_the_number_in_three_significant_digits_and_no_score_adds_nothing() -> (
    None
):
    assert draw.label_of(2, 0.82345) == "2 (0.823)"
    assert draw.label_of(2, None) == "2"
    assert draw.label_of(2, 12345.6) == "2 (1.23e+04)"
    assert draw.label_of(2, 0.0) == "2 (0)"
    assert draw.label_of(2, -0.5) == "2 (-0.5)"


def _captions_scan(tmp_path: Path, route: RouteResult, **options: Any) -> Scan:
    cells = [
        cell_at("B2", 0.0, 0.0, "single"),
        cell_at("B2", 50.0, 0.0, "single"),
        cell_at("B2", 100.0, 0.0, "doublet"),
        cell_at("B2", 150.0, 0.0, "debris"),
        cell_at("B2", 200.0, 0.0, "debris"),
        cell_at("B2", 250.0, 0.0, "debris"),
        cell_at("A1", 0.0, 0.0, "single"),
    ]
    return _scan_of_run(tmp_path, route=route, cells=cells, **options)


def test_the_caption_names_the_well_the_route_the_selector_the_orderer_the_field_the_stops_and_the_uncovered(
    tmp_path: Path,
) -> None:
    route = _route((_stop(0, "B2", 0),), uncovered={"B2": 3})
    scan = _captions_scan(tmp_path, route, params=CHOSE)
    view, _ = draw.route_of(scan)
    assert draw.caption_lines(scan, "B2", view, None, stops=5, frames=1) == [
        "B2: 5 stops; uncovered single, doublet cells: 3",
        "route: selector most-cells, orderer nearest",
        "fields 400 x 300 um at 2-Plan Fluor 10x",
        "cells: single (gold) 2, doublet (cyan) 1, debris (red) 3",
    ]


def test_the_caption_counts_stops_in_words_and_leaves_out_what_it_does_not_know(
    tmp_path: Path,
) -> None:
    route = _route((_stop(0, "B2", 0),), objective=None, uncovered={"A1": 1})
    scan = _captions_scan(tmp_path, route, params=CHOSE)
    view, _ = draw.route_of(scan)
    one = draw.caption_lines(scan, "B2", view, None, stops=1, frames=1)
    none = draw.caption_lines(scan, "B2", view, None, stops=0, frames=1)
    # B2 is not in `uncovered`: not counted, which is not 0. No objective: no "at".
    assert one[0] == "B2: 1 stop; uncovered cells: not counted"
    assert none[0] == "B2: no stop; uncovered cells: not counted"
    assert one[2] == "fields 400 x 300 um"


def test_the_caption_says_the_frames_that_could_not_be_read_and_the_stops_left_out(
    tmp_path: Path,
) -> None:
    route = _route((_stop(0, "B2", 0),), uncovered={"B2": 0})
    scan = _captions_scan(tmp_path, route, params=CHOSE)
    view, _ = draw.route_of(scan)
    lines = draw.caption_lines(
        scan,
        "B2",
        view,
        None,
        stops=3,
        frames=109,
        unreadable=3,
        first_problem="scan/t1/B2/B2_0003.tif: the file is missing",
        off_picture=2,
    )
    assert lines[-2:] == [
        "3 of 109 frames could not be read; first: "
        "scan/t1/B2/B2_0003.tif: the file is missing",
        "2 stops outside the picture are not drawn",
    ]
    one = draw.caption_lines(scan, "B2", view, None, stops=3, frames=9, off_picture=1)
    assert one[-1] == "1 stop outside the picture is not drawn"
    clean = draw.caption_lines(scan, "B2", view, None, stops=3, frames=9)
    assert not any("outside" in line or "could not" in line for line in clean)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"selector": 5, "orderer": ["nearest"]},
        {"selector": "", "orderer": None},
    ],
)
def test_a_route_that_names_no_selector_says_it_is_not_recorded(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    # A route made before the selection contract (or by the prototype) says neither;
    # a value that is not a string counts as not recorded (FM-125).
    scan = _captions_scan(tmp_path, _route((_stop(0, "B2", 0),)), params=params)
    view, _ = draw.route_of(scan)
    assert view is not None
    assert (view.selector, view.orderer) == (None, None)
    lines = draw.caption_lines(scan, "B2", view, None, stops=1, frames=1)
    assert lines[1] == "route: selector and orderer not recorded"


def test_the_prototypes_record_is_drawn_as_the_route_and_named_as_such(
    tmp_path: Path,
) -> None:
    # `route-prototype` started later than `route`, so it fills the slot (provider).
    scan = _captions_scan(tmp_path, _route((_stop(0, "B2", 0),)), params=CHOSE)
    _write_route(
        scan.run_dir / "run.json",
        _route((_stop(0, "B2", 0), _stop(1, "B2", 1))),
        plugin="route-prototype",
        day=30,
    )
    scan = load_scan(scan.run_dir / "run.json")
    view, why = draw.route_of(scan)
    assert why is None
    assert view is not None
    assert view.plugin == "route-prototype"
    assert len(draw.stops_of_well(view, "B2")) == 2
    lines = draw.caption_lines(scan, "B2", view, None, stops=2, frames=1)
    assert lines[1] == "route-prototype: selector and orderer not recorded"


def test_only_one_name_recorded_says_which_is_missing(tmp_path: Path) -> None:
    scan = _captions_scan(
        tmp_path, _route((_stop(0, "B2", 0),)), params={"selector": "most-cells"}
    )
    view, _ = draw.route_of(scan)
    lines = draw.caption_lines(scan, "B2", view, None, stops=1, frames=1)
    assert lines[1] == "route: selector most-cells, orderer not recorded"


def _stale_route(tmp_path: Path) -> Scan:
    """A route that read ``scan`` attempt ``a1``, and a scan run again since (attempt ``a2``, later)."""
    seed = seed_run(tmp_path, cells=[cell_at("A1", 0.0, 0.0)], wells=("A1",))
    _write_route(
        seed.path,
        _route((_stop(0, "A1", 0),)),
        reads={"scan": ReadRecord(plugin="scan", attempt="a1")},
    )
    record(seed.path, "scan", "scan", seed.scan, attempt="a2", day=30)
    return load_scan(seed.path)


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("none", "the run document has no route; smc run --list shows"),
        ("stale", "route read its inputs from an earlier run; run it again"),
        ("other profile", "route was run on stand-b, this is demo; run route again"),
        ("dry run", "route was a dry run; run it without --dry-run"),
    ],
)
def test_a_route_that_is_not_usable_on_the_scan_is_a_reason_and_not_a_route(
    tmp_path: Path, case: str, reason: str
) -> None:
    if case == "none":
        scan = _scan_of_run(tmp_path)
    elif case == "stale":
        scan = _stale_route(tmp_path)
    elif case == "other profile":
        scan = _scan_of_run(
            tmp_path, route=_route((_stop(0, "A1", 0),)), profile="stand-b"
        )
    else:
        scan = _scan_of_run(tmp_path, route=_route((_stop(0, "A1", 0),)), dry_run=True)
    view, why = draw.route_of(scan)
    assert view is None
    assert why is not None
    assert why.startswith(reason), why


def test_a_route_record_that_is_not_a_route_result_is_a_reason_naming_the_first_problem(
    tmp_path: Path,
) -> None:
    good = _route((_stop(0, "A1", 0),)).model_dump(mode="json")
    good["positions"] = "not a list of stops"
    scan = _scan_of_run(tmp_path, route=good)
    view, why = draw.route_of(scan)
    assert view is None
    assert why is not None
    assert why.startswith(
        "the route record is not a route result this smc reads (positions: "
    )
    assert why.endswith("); run route again")


def test_a_route_record_with_no_result_is_a_reason(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[cell_at("A1", 0.0, 0.0)], wells=("A1",))
    then = datetime(2026, 10, 29, 9, 30, tzinfo=timezone.utc)
    write_stage(
        seed.path,
        StageRecord(
            plugin="route",
            provides="route",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=None,
        ),
    )
    view, why = draw.route_of(load_scan(seed.path))
    assert view is None
    assert why == "the route record has no result; run route again"


def test_without_a_route_the_caption_says_why_in_its_second_line(
    tmp_path: Path,
) -> None:
    scan = _scan_of_run(tmp_path)
    view, why = draw.route_of(scan)
    lines = draw.caption_lines(scan, "A1", view, why, stops=0, frames=1)
    assert lines[0] == "A1: no route drawn"
    assert lines[1] == why
    assert lines[2].startswith("cells: ")
    assert len(lines) == 3


def test_a_record_string_of_a_million_characters_is_clipped_in_the_caption(
    tmp_path: Path,
) -> None:
    huge = "s" * 1_000_000
    route = _route((_stop(0, "B2", 0),), objective="o" * 1_000_000)
    scan = _captions_scan(tmp_path, route, params={"selector": huge, "orderer": "n"})
    view, _ = draw.route_of(scan)
    lines = draw.caption_lines(scan, "B2", view, None, stops=1, frames=1)
    assert len(lines) <= 8
    assert all(len(line) <= 200 for line in lines)
    assert "s" * 117 + "..." in lines[1]
    assert "s" * 118 not in lines[1]


def test_a_record_string_with_a_newline_a_control_character_or_a_path_stays_on_one_line(
    tmp_path: Path,
) -> None:
    route = _route(
        (_stop(0, "B2", 0),), objective="a\nb\x00c\N{GREEK SMALL LETTER LAMDA}"
    )
    scan = _captions_scan(tmp_path, route, params=CHOSE)
    view, _ = draw.route_of(scan)
    lines = draw.caption_lines(scan, "B2", view, None, stops=1, frames=1)
    assert lines[2] == "fields 400 x 300 um at a b?c\\u03bb"
    assert all(line.isascii() and "\n" not in line for line in lines)


def test_the_cells_line_lists_a_label_the_classes_do_not_know_as_white(
    tmp_path: Path,
) -> None:
    # `load_scan` refuses cells the result did not count, so the scan is built in memory.
    seed = seed_run(tmp_path, cells=[], wells=("A1", "B2"), rows=1, columns=1)
    scan = Scan.from_results(
        tmp_path,
        RunDocument.load(seed.path),
        plan=seed.plan,
        result=seed.scan,
        cells_result=seed.result,
        cells=[cell_at("B2", 0.0, 0.0, "single"), cell_at("B2", 50.0, 0.0, "mystery")],
    )
    line = draw.caption_lines(scan, "B2", None, "none", stops=0, frames=1)[-1]
    assert line == (
        "cells: single (gold) 1, doublet (cyan) 0, debris (red) 0, mystery (white) 1"
    )


def test_a_scan_with_no_cells_says_why_in_the_cells_line_without_a_path(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",))
    cells_file = tmp_path / seed.result.file
    cells_file.unlink()
    scan = load_scan(seed.path)
    assert scan.cells is None
    line = draw.caption_lines(scan, "A1", None, "none", stops=0, frames=1)[-1]
    assert line.startswith("cells: none (cells file ")
    assert str(tmp_path) not in line
    assert "<run folder>" in line


def test_a_scan_that_found_no_cell_shows_zeros_and_not_none(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",))
    scan = load_scan(seed.path)
    line = draw.caption_lines(scan, "A1", None, "none", stops=0, frames=1)[-1]
    assert line == "cells: single (gold) 0, doublet (cyan) 0, debris (red) 0"


# -------------------------------------------------------------- draw_well / draw_scan


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What ``draw_well`` hands ``render_png``, call by call: the caption and the stops are not in the PNG as data."""
    real = draw.render_png
    calls: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Image.Image:
        calls.append({"grey": args[0], "canvas": args[1], **kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(draw, "render_png", spy)
    return calls


def _picture(drawing: draw.Drawing, well: str) -> Image.Image:
    """A well's PNG as it was written, read back with Pillow."""
    (found,) = [p for p in drawing.pictures if p.well == well]
    assert found.written, found.error
    image = Image.open(found.path)
    image.load()
    return image.convert("RGB")


def _anywhere(image: Image.Image, colour: tuple[int, int, int]) -> bool:
    """Whether ``colour`` is on the picture at all."""
    return bool((np.asarray(image) == colour).all(axis=-1).any())


def _has(image: Image.Image, colour: tuple[int, int, int], x: int, y: int) -> bool:
    """Whether ``colour`` is within one pixel of ``(x, y)``: a line is 1 to 2 pixels wide."""
    return any(
        image.getpixel((x + dx, y + dy)) == colour
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    )


def _edge_midpoints(
    x_um: float, y_um: float, matrix: Matrix2, shape: tuple[int, int]
) -> list[tuple[float, float]]:
    corners = footprint(x_um, y_um, matrix=matrix, image_shape_px=shape)
    return [
        ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True)
    ]


def _seed_with_route(
    tmp_path: Path, stops: tuple[RoutePosition, ...], **route_options: Any
) -> Scan:
    seed = seed_run(
        tmp_path,
        cells=[cell_at("A1", 1500.0, 1000.0)],
        wells=("A1", "B2"),
        rows=1,
        columns=1,
    )
    _write_route(seed.path, _route(stops, **route_options), params=CHOSE)
    return load_scan(seed.path)


def test_draw_scan_draws_the_stops_of_each_well_as_footprints_numbered_in_their_well(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    stops = (
        _stop(0, "A1", 0, 1000.0, -500.0),
        _stop(1, "A1", 1, 1600.0, -300.0, score=0.82345),
        _stop(2, "B2", 0, 10000.0, -9500.0),
    )
    scan = _seed_with_route(tmp_path, stops)
    drawing = draw.draw_scan(scan, tmp_path / "out", max_px=1600)
    assert [(p.well, p.stops, p.off_picture, p.written) for p in drawing.pictures] == [
        ("A1", 2, 0, True),
        ("B2", 1, 0, True),
    ]
    assert (drawing.route, drawing.route_absent) == ("route", None)
    # B2's only stop is rank 1, though its order in the route is 2; A1's second carries its score.
    texts = {c["caption"].split(":")[0]: [s[0] for s in c["stops"]] for c in renders}
    assert texts == {"A1": ["1", "2 (0.823)"], "B2": ["1"]}
    canvas = draw.canvas_of(scan.geometry["A1"], 1600)
    first = _picture(drawing, "A1")
    for x, y in _edge_midpoints(1000.0, -500.0, UPRIGHT, SHAPE):
        assert _has(first, GREEN, *_pixel_of(canvas, x, y))
    for x, y in _edge_midpoints(1600.0, -300.0, UPRIGHT, SHAPE):
        assert _has(first, GREEN, *_pixel_of(canvas, x, y))
    # The path is the yellow line between the two stops' centres, at its middle.
    middle = _pixel_of(canvas, 1300.0, -400.0)
    assert _has(first, (255, 255, 160), *middle)


def test_the_wanted_labels_of_the_route_are_ringed_and_the_others_are_not(
    tmp_path: Path,
) -> None:
    cells = [cell_at("A1", 0.0, 0.0, "single"), cell_at("A1", 400.0, 0.0, "doublet")]
    seed = seed_run(tmp_path, cells=cells, wells=("A1",), rows=1, columns=1)
    _write_route(
        seed.path,
        _route((_stop(0, "A1", 0, 1000.0, -2000.0),), labels=("single",)),
        params=CHOSE,
    )
    scan = load_scan(seed.path)
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=800), "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 800)
    gold, cyan = draw.CLASS_COLOURS[0][1], draw.CLASS_COLOURS[1][1]
    for dx, colour, ringed in ((0.0, gold, True), (400.0, cyan, False)):
        column, row = _pixel_of(canvas, 1000.0 + dx, -500.0)
        assert image.getpixel((column, row)) == colour
        assert (image.getpixel((column + 5, row)) == colour) is ringed


@pytest.mark.parametrize("case", ["none", "stale", "other profile", "not a result"])
def test_without_a_usable_route_the_picture_has_the_frames_and_the_cells_and_the_caption_says_why(
    tmp_path: Path, renders: list[dict[str, Any]], case: str
) -> None:
    if case == "none":
        scan = _scan_of_run(tmp_path, wells=("A1",))
    elif case == "stale":
        scan = _stale_route(tmp_path)
    elif case == "other profile":
        scan = _scan_of_run(
            tmp_path,
            wells=("A1",),
            route=_route((_stop(0, "A1", 0, 1000.0, -500.0),)),
            profile="stand-b",
        )
    else:
        broken = _route((_stop(0, "A1", 0, 1000.0, -500.0),)).model_dump(mode="json")
        broken["positions"] = "not a list of stops"
        scan = _scan_of_run(tmp_path, wells=("A1",), route=broken)
    drawing = draw.draw_scan(scan, tmp_path / "out", max_px=800)
    _, why = draw.route_of(scan)
    assert drawing.route is None
    assert drawing.route_absent == why
    assert why is not None
    # The caption's first line says no route was drawn and its second says why.
    caption = renders[0]["caption"].split("\n")
    assert caption[0] == "A1: no route drawn"
    assert caption[1] == why
    assert renders[0]["stops"] == []
    image = _picture(drawing, "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 800)
    # The mosaic and the cell are there: the frame's stretch puts a pixel of it above
    # the dark grey, the cell is a gold dot, and nothing is the route's green.
    assert not _anywhere(image, GREEN)
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    assert image.getpixel((column, row)) == draw.CLASS_COLOURS[0][1]
    assert drawing.pictures[0].stops == 0


def test_a_stop_is_drawn_with_the_routes_own_field_and_not_the_scans(
    tmp_path: Path,
) -> None:
    # FM-124: the route's camera is a quarter of the scan's area and its frame is
    # transposed; the footprint drawn is the route's.
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    matrix: Matrix2 = ((2.5, 0.0), (0.0, -2.5))
    shape = (80, 60)
    _write_route(
        seed.path,
        _route(
            (_stop(0, "A1", 0, 1000.0, -500.0),),
            matrix=matrix,
            shape=shape,
            fov_um=(150.0, 200.0),
        ),
        params=CHOSE,
    )
    scan = load_scan(seed.path)
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=1600), "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 1600)
    for x, y in _edge_midpoints(1000.0, -500.0, matrix, shape):
        assert _has(image, GREEN, *_pixel_of(canvas, x, y))
    for x, y in footprint(1000.0, -500.0, matrix=UPRIGHT, image_shape_px=SHAPE):
        assert not _has(image, GREEN, *_pixel_of(canvas, x, y))


@pytest.mark.parametrize("name", ["upright", "mirrored", "turned 30"])
def test_the_frames_the_cells_and_the_wall_are_where_the_stage_says(
    tmp_path: Path, name: str
) -> None:
    matrix = {
        "upright": UPRIGHT,
        "mirrored": matrix_of(PIXEL_UM, 0.0, True),
        "turned 30": matrix_of(PIXEL_UM, 30.0, False),
    }[name]

    def marked(frame: ScanFrame) -> np.ndarray:
        # A 15 x 15 pixel block (rows 5-19, columns 55-69) is the only thing the stretch
        # maps to white: off the axes and the frame's centre on purpose (FM-73).
        data = np.full(SHAPE, 100, dtype=np.uint16)
        data[5:20, 55:70] = 4000
        return data

    seed = seed_run(
        tmp_path,
        cells=[cell_at("A1", 1500.0, 1000.0)],
        wells=("A1",),
        rows=1,
        columns=1,
        matrix=matrix,
        frame_pixels=marked,
    )
    scan = load_scan(seed.path)
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=256), "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 256)
    frame = seed.scan.frames[0]
    marker = stage_of_pixel(
        frame.x_um, frame.y_um, 62.0, 12.0, matrix=matrix, image_shape_px=SHAPE
    )
    assert image.getpixel(_pixel_of(canvas, *marker)) == (255, 255, 255)
    # The point of the frame opposite through its centre is the dark background.
    opposite = stage_of_pixel(
        frame.x_um, frame.y_um, 17.0, 47.0, matrix=matrix, image_shape_px=SHAPE
    )
    assert image.getpixel(_pixel_of(canvas, *opposite)) == (0, 0, 0)
    assert image.getpixel(_pixel_of(canvas, 2500.0, 500.0)) == draw.CLASS_COLOURS[0][1]
    for dx, dy in ((3200.0, 0.0), (-3200.0, 0.0), (0.0, 3200.0), (0.0, -3200.0)):
        assert _has(image, (90, 90, 255), *_pixel_of(canvas, 1000.0 + dx, -500.0 + dy))


def test_a_cell_is_coloured_by_the_class_the_legend_names(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    labels = ("single", "doublet", "debris", "mystery")
    cells = [
        cell_at("A1", 400.0 * k - 600.0, 1000.0, label)
        for k, label in enumerate(labels)
    ]
    scan = Scan.from_results(
        tmp_path,
        RunDocument.load(seed.path),
        plan=seed.plan,
        result=seed.scan,
        cells_result=seed.result,
        cells=cells,
    )
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=800), "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 800)
    legend = renders[0]["caption"].split("\n")[-1]
    colours = [rgb for _, rgb in draw.CLASS_COLOURS[:3]] + [(255, 255, 255)]
    names = [name for name, _ in draw.CLASS_COLOURS[:3]] + ["white"]
    for cell, colour, name in zip(cells, colours, names, strict=True):
        column, row = _pixel_of(canvas, cell.x_um, cell.y_um)
        assert image.getpixel((column, row)) == colour, cell.label
        assert f"{cell.label} ({name})" in legend


def test_a_scan_with_no_cells_draws_the_frames_and_the_stops_and_says_why(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    (tmp_path / seed.result.file).unlink()
    _write_route(seed.path, _route((_stop(0, "A1", 0, 1000.0, -500.0),)), params=CHOSE)
    scan = load_scan(seed.path)
    assert scan.cells is None
    drawing = draw.draw_scan(scan, tmp_path / "out", max_px=800)
    assert drawing.cells is False
    assert drawing.pictures[0].cells == 0
    assert renders[0]["caption"].split("\n")[-1].startswith("cells: none (cells file ")
    image = _picture(drawing, "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 800)
    for x, y in _edge_midpoints(1000.0, -500.0, UPRIGHT, SHAPE):
        assert _has(image, GREEN, *_pixel_of(canvas, x, y))


def _first_frame_file(scan: Scan) -> Path:
    return scan.run_dir / scan.frames("A1")[0].file


def _rewrite(path: Path, how: str) -> None:
    if how == "missing":
        path.unlink()
    elif how == "garbage":
        path.write_bytes(b"this is not a tiff at all")
    elif how == "empty":
        path.write_bytes(b"")
    elif how == "wrong shape":
        tifffile.imwrite(path, np.full((40, 40), 1000, dtype=np.uint16))
    else:
        nan = np.full(SHAPE, 1000.0, dtype=np.float32)
        nan[3, 3] = np.nan
        tifffile.imwrite(path, nan)


@pytest.mark.parametrize("how", ["missing", "garbage", "empty", "wrong shape", "nan"])
def test_an_unreadable_frame_is_counted_and_never_ends_the_drawing(
    tmp_path: Path, renders: list[dict[str, Any]], how: str
) -> None:
    # The run folder has a space and a bracket in its name (FM-50, FM-36): none of it
    # may reach the caption.
    folder = tmp_path / "run [1] a b"
    folder.mkdir()
    seed = seed_run(folder, cells=[cell_at("A1", 0.0, 0.0)], wells=("A1",), rows=2)
    scan = load_scan(seed.path)
    first = scan.frames("A1")[0]
    _rewrite(_first_frame_file(scan), how)
    drawing = draw.draw_scan(scan, tmp_path / "out", max_px=800)
    (picture,) = drawing.pictures
    assert picture.written
    assert picture.frames == len(scan.frames("A1")) >= 2
    assert (picture.unreadable, picture.placed) == (1, picture.frames - 1)
    caption = renders[0]["caption"].split("\n")
    line = next(line for line in caption if "could not be read" in line)
    assert line.startswith(f"1 of {picture.frames} frames could not be read; first: ")
    assert f"first: {first.file}: " in line
    assert str(folder) not in line
    assert "run [1]" not in line
    assert "a b" not in line
    assert line.isascii()


def test_a_reason_that_quotes_the_run_folder_is_scrubbed_before_it_reaches_the_caption(
    tmp_path: Path, renders: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-50: a picture is shared, and a decoder's message may quote the PC's path.
    folder = tmp_path / "run [1] a b"
    folder.mkdir()
    seed = seed_run(folder, cells=[], wells=("A1",), rows=2)
    scan = load_scan(seed.path)
    first = scan.frames("A1")[0]
    real = scan_module.load_frame

    def quoting(run_dir: Path, frame: ScanFrame) -> np.ndarray:
        if frame.name == first.name:
            raise ValueError(f"cannot decode {run_dir / frame.file} in {run_dir}")
        return real(run_dir, frame)

    monkeypatch.setattr(scan_module, "load_frame", quoting)
    draw.draw_scan(scan, tmp_path / "out", max_px=800)
    lines = renders[0]["caption"].split("\n")
    line = next(text for text in lines if "could not" in text)
    assert str(folder) not in line
    assert folder.name not in line
    assert "cannot decode <run folder>" in line


def test_a_stop_far_outside_the_well_is_counted_and_left_out_and_raises_nothing(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    # FM-74, FM-75: Pillow raises SystemError for a text at 1e300. The ranks are the
    # well's whole list's, so the third stop stays 3 when the second is left out.
    stops = (
        _stop(0, "A1", 0, 1000.0, -500.0),
        _stop(1, "A1", 1, 1e300, -500.0),
        _stop(2, "A1", 2, 1600.0, -500.0),
        _stop(3, "A1", 3, 1e9, 1e9),
    )
    scan = _seed_with_route(tmp_path, stops)
    drawing = draw.draw_scan(scan, tmp_path / "out", wells=["A1"], max_px=800)
    (picture,) = drawing.pictures
    assert (picture.stops, picture.off_picture) == (4, 2)
    assert [s[0] for s in renders[0]["stops"]] == ["1", "3"]
    assert [s[1] for s in renders[0]["stops"]] == [1000.0, 1600.0]
    caption = renders[0]["caption"].split("\n")
    assert "2 stops outside the picture are not drawn" in caption
    assert caption[0].startswith("A1: 4 stops;")
    # No line is drawn to a left-out stop: the path's colour appears only between the
    # two stops on the picture.
    image = _picture(drawing, "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 800)
    row_of_path = _pixel_of(canvas, 1300.0, -500.0)
    assert _has(image, (255, 255, 160), *row_of_path)
    assert _anywhere(image, (255, 255, 160))
    far_right = [
        image.getpixel((c, r))
        for c in range(image.size[0] - 30, image.size[0])
        for r in range(image.size[1])
    ]
    assert (255, 255, 160) not in far_right


def test_a_matrix_that_cannot_be_inverted_is_a_dataset_error_naming_the_scan(
    tmp_path: Path,
) -> None:
    # FM-125: nothing checks that the matrix of a loaded scan can be inverted.
    seed = seed_run(
        tmp_path,
        cells=[],
        wells=("A1",),
        rows=1,
        columns=1,
        matrix=((1.0, 1.0), (1.0, 1.0)),
        plan_matrix=UPRIGHT,
    )
    scan = load_scan(seed.path)
    out = tmp_path / "out"
    with pytest.raises(DatasetError, match="the scan's camera cannot be drawn"):
        draw.draw_scan(scan, out)
    assert not out.exists()


def test_the_picture_is_the_wells_box_and_never_the_frames(tmp_path: Path) -> None:
    # FM-136: a frame whose readback is 1e9 um away places nothing and does not grow the picture.
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    far = seed.scan.frames[0].model_copy(update={"x_um": 1e9})
    result = seed.scan.model_copy(update={"frames": (far,)})
    scan = Scan.from_results(
        tmp_path,
        RunDocument.load(seed.path),
        plan=seed.plan,
        result=result,
        cells_result=seed.result,
        cells=[],
    )
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=512), "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 512)
    assert image.size == (canvas.width_px, canvas.height_px)
    # The frame is read and placed nowhere: the well's centre is still the dark grey
    # of a pixel no frame reached, not the (0, 0, 0) a flat frame would stretch to.
    assert image.getpixel(_pixel_of(canvas, 1000.0, -500.0)) == (40, 40, 40)


def test_max_px_is_the_longest_side(tmp_path: Path) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    scan = load_scan(seed.path)
    square = WellGeometry(
        well="A1",
        center_um=(1000.0, -500.0),
        size_um=(4000.0, 2500.0),
        circular=False,
        rotation_deg=17.0,
    )
    turned = replace(scan, geometry=MappingProxyType({"A1": square}))
    for max_px in (256, 1600):
        round_picture = _picture(
            draw.draw_scan(scan, tmp_path / "r", max_px=max_px), "A1"
        )
        assert max(round_picture.size) == max_px
        other = _picture(draw.draw_scan(turned, tmp_path / "s", max_px=max_px), "A1")
        assert max(other.size) == max_px
        assert min(other.size) < max_px


def _plain_scan(tmp_path: Path, wells: tuple[str, ...] = ("A1", "B2", "B1")) -> Scan:
    seed = seed_run(tmp_path, cells=[], wells=wells, rows=1, columns=1)
    return load_scan(seed.path)


def test_draw_scan_writes_one_picture_per_well_in_the_plans_order_and_touches_nothing_else(
    tmp_path: Path,
) -> None:
    scan = _plain_scan(tmp_path)
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    keep = {
        out / "notes.txt": b"my notes",
        out / "Z9.png": b"not ours",
        out / "sub" / "A1.png": b"in a folder",
    }
    for path, content in keep.items():
        path.write_bytes(content)
    (out / "A1.png").write_bytes(b"sentinel")
    drawing = draw.draw_scan(scan, out, wells=("b2", "A1", "B02"), max_px=256)
    assert [p.well for p in drawing.pictures] == ["A1", "B2"]
    assert all(p.path == out / f"{p.well}.png" and p.written for p in drawing.pictures)
    for path, content in keep.items():
        assert path.read_bytes() == content
    assert not (out / "B1.png").exists()
    with Image.open(out / "A1.png") as new:
        new.load()
        assert new.format == "PNG"
    assert sorted(p.name for p in out.iterdir()) == [
        "A1.png",
        "B2.png",
        "Z9.png",
        "notes.txt",
        "sub",
    ]


def test_every_well_is_drawn_when_none_are_named_and_progress_follows_each_well(
    tmp_path: Path,
) -> None:
    scan = _plain_scan(tmp_path)
    seen: list[tuple[int, int, str]] = []
    drawing = draw.draw_scan(
        scan, tmp_path / "out", max_px=256, progress=lambda *step: seen.append(step)
    )
    assert [p.well for p in drawing.pictures] == ["A1", "B2", "B1"]
    assert seen == [(1, 3, "A1"), (2, 3, "B2"), (3, 3, "B1")]
    assert drawing.route is None
    assert drawing.cells is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wells": ["Z9"]}, "well 'Z9' is not in this scan (A1, B2, B1)"),
        ({"wells": ["A1", "C3"]}, "well 'C3' is not in this scan"),
        ({"wells": []}, "no well was asked for"),
        ({"max_px": 255}, "max_px must be an integer from 256 to 4000"),
        ({"max_px": 4001}, "max_px must be an integer from 256 to 4000"),
        ({"max_px": True}, "max_px must be an integer from 256 to 4000"),
        ({"max_px": 800.0}, "max_px must be an integer from 256 to 4000"),
    ],
)
def test_an_unknown_well_or_a_bad_max_px_is_refused_before_any_file_is_written(
    tmp_path: Path, renders: list[dict[str, Any]], kwargs: dict[str, Any], message: str
) -> None:
    scan = _plain_scan(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(DatasetError, match=re.escape(message)):
        draw.draw_scan(scan, out, **kwargs)
    assert not out.exists()
    assert renders == [], "no frame was read: nothing was drawn"


def test_an_out_dir_that_is_a_file_is_refused_and_the_file_is_not_touched(
    tmp_path: Path,
) -> None:
    scan = _plain_scan(tmp_path)
    target = tmp_path / "out"
    target.write_bytes(b"a file")
    with pytest.raises(DatasetError, match="is a file, not a folder"):
        draw.draw_scan(scan, target)
    assert target.read_bytes() == b"a file"


def test_a_picture_that_cannot_be_written_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-34: a viewer that holds B2.png open is B2's error and not the scan's.
    scan = _plain_scan(tmp_path)
    real = draw.write_atomically

    def flaky(path: Path, data: bytes, **kwargs: Any) -> None:
        if path.name == "B2.png":
            raise AtomicWriteError(OSError(errno.EIO, "disk"))
        real(path, data, **kwargs)

    monkeypatch.setattr(draw, "write_atomically", flaky)
    out = tmp_path / "out"
    drawing = draw.draw_scan(scan, out, max_px=256)
    by_well = {p.well: p for p in drawing.pictures}
    assert (out / "A1.png").is_file()
    assert (out / "B1.png").is_file()
    assert not (out / "B2.png").exists()
    assert by_well["A1"].written
    assert by_well["B1"].written
    assert not by_well["B2"].written
    assert by_well["B2"].error == "disk"
    assert str(tmp_path) not in (by_well["B2"].error or "")


def test_a_picture_is_replaced_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-38: the old picture stays when the new one cannot be put in its place.
    scan = _plain_scan(tmp_path, wells=("A1",))
    out = tmp_path / "out"
    draw.draw_scan(scan, out, max_px=256)
    old = (out / "A1.png").read_bytes()

    def refuse(src: object, dst: object) -> None:
        raise PermissionError(errno.EACCES, "in use")

    monkeypatch.setattr(os, "replace", refuse)
    drawing = draw.draw_scan(scan, out, max_px=512)
    assert not drawing.pictures[0].written
    assert (out / "A1.png").read_bytes() == old


def test_stops_in_wells_the_scan_does_not_have_are_a_note_and_the_rest_is_drawn(
    tmp_path: Path,
) -> None:
    stops = (
        _stop(0, "A1", 0, 1000.0, -500.0),
        _stop(1, "C3", 0, 50.0, 50.0),
        _stop(2, "C3", 1, 60.0, 60.0),
    )
    scan = _seed_with_route(tmp_path, stops)
    drawing = draw.draw_scan(scan, tmp_path / "out", max_px=256)
    assert drawing.notes == (
        "the route holds 2 stops in wells the scan does not have (C3): not drawn",
    )
    assert drawing.pictures[0].stops == 1


def test_the_scan_is_the_one_the_picture_says_it_is_when_a_well_has_no_frame(
    tmp_path: Path,
) -> None:
    # A well of the plan with no frame (every field failed) is a picture of the wall
    # and the cells, not an error.
    seed = seed_run(tmp_path, cells=[cell_at("A1", 0.0, 0.0)], wells=("A1",), rows=1)
    result = seed.scan.model_copy(update={"frames": ()})
    scan = Scan.from_results(
        tmp_path,
        RunDocument.load(seed.path),
        plan=seed.plan,
        result=result,
        cells_result=seed.result,
        cells=seed.cells,
    )
    drawing = draw.draw_scan(scan, tmp_path / "out", max_px=256)
    (picture,) = drawing.pictures
    assert (picture.frames, picture.placed, picture.unreadable) == (0, 0, 0)
    assert picture.written


def _from_results(tmp_path: Path, seed: Seed, **changes: Any) -> Scan:
    """A scan built in memory from the seed's results, with the changes given (``Scan.from_results`` checks none of them)."""
    arguments: dict[str, Any] = {
        "plan": seed.plan,
        "result": seed.scan,
        "cells_result": seed.result,
        "cells": seed.cells,
    }
    arguments.update(changes)
    return Scan.from_results(tmp_path, RunDocument.load(seed.path), **arguments)


def test_the_longest_side_may_be_as_large_as_4000(tmp_path: Path) -> None:
    # A well with no frame: only the canvas is that large (64 MB), nothing is placed.
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    scan = _from_results(
        tmp_path, seed, result=seed.scan.model_copy(update={"frames": ()})
    )
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=4000), "A1")
    assert max(image.size) == 4000


def test_a_scan_with_no_well_is_refused_and_not_drawn_as_nothing(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    scan = replace(load_scan(seed.path), wells=())
    with pytest.raises(DatasetError, match="this scan has no well"):
        draw.draw_scan(scan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_pixel_size_that_is_not_a_size_is_a_dataset_error_naming_the_camera(
    tmp_path: Path,
) -> None:
    # `ScanResult.pixel_size_um` only has to be above 0: infinity gets through.
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    scan = _from_results(
        tmp_path, seed, result=seed.scan.model_copy(update={"pixel_size_um": math.inf})
    )
    with pytest.raises(
        DatasetError, match="the scan's camera cannot be drawn: the pixel size"
    ):
        draw.draw_scan(scan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_cell_far_outside_the_picture_is_not_drawn_and_raises_nothing(
    tmp_path: Path,
) -> None:
    # FM-74, FM-75: Pillow raises SystemError for an ellipse at 1e300.
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    cells = [
        cell_at("A1", 1e300, 0.0),
        cell_at("A1", 1e9, 0.0),
        cell_at("A1", -1e9, 1e9),
        cell_at("A1", 0.0, 0.0),
    ]
    scan = _from_results(tmp_path, seed, cells=cells)
    image = _picture(draw.draw_scan(scan, tmp_path / "out", max_px=800), "A1")
    canvas = draw.canvas_of(scan.geometry["A1"], 800)
    assert image.getpixel(_pixel_of(canvas, 1000.0, -500.0)) == draw.CLASS_COLOURS[0][1]


def test_the_cells_line_counts_the_labels_it_lists_past_eight_and_says_the_rest(
    tmp_path: Path,
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    cells = [cell_at("A1", 10.0 * k, 0.0, f"x{k}") for k in range(9)]
    scan = _from_results(tmp_path, seed, cells=cells)
    line = draw.caption_lines(scan, "A1", None, "none", stops=0, frames=1)[-1]
    assert line.endswith("x7 (white) 1, 1 more label (white)")
    assert "x8" not in line
    ten = _from_results(
        tmp_path, seed, cells=[cell_at("A1", 10.0 * k, 0.0, f"x{k}") for k in range(10)]
    )
    assert draw.caption_lines(ten, "A1", None, "none", stops=0, frames=1)[-1].endswith(
        "x7 (white) 1, 2 more labels (white)"
    )


def test_the_note_on_stops_in_other_wells_lists_eight_and_says_there_are_more(
    tmp_path: Path,
) -> None:
    stops = tuple(_stop(k, f"C{k + 1}", 0, 50.0, 50.0) for k in range(9))
    scan = _seed_with_route(tmp_path, stops)
    (note,) = draw.draw_scan(scan, tmp_path / "out", max_px=256).notes
    assert note == (
        "the route holds 9 stops in wells the scan does not have "
        "(C1, C2, C3, C4, C5, C6, C7, C8, ...): not drawn"
    )


def test_a_caption_is_cut_before_it_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-74: a record string of a million characters costs a million measurements
    # if it is not cut first; the length wrap_text is given is the proof, not a clock.
    seen: list[int] = []
    real = draw.wrap_text

    def spy(pen: Any, text: str, font: Any, width: float) -> list[str]:
        seen.append(len(text))
        return real(pen, text, font, width)

    monkeypatch.setattr(draw, "wrap_text", spy)
    _render(stops=[], caption="z" * 1_000_000, max_px=300)
    assert seen == [2000]


# ----------------------------------------------- what the picture is made of


def test_the_picture_draws_the_mosaic_the_scripts_read_pixels_makes(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    # Overlapping noise frames: the order they are placed in, and the pixel size the
    # reduction is made from, change the picture.
    matrix = matrix_of(PIXEL_UM, 30.0, True)
    seed = seed_run(
        tmp_path, cells=[], rows=2, columns=2, matrix=matrix, frame_pixels=_noise
    )
    scan = load_scan(seed.path)
    draw.draw_scan(scan, tmp_path / "out", max_px=800)
    expected = (
        load_script()
        .read_pixels(
            tmp_path,
            scan.result,
            {"A1": scan.geometry["A1"]},
            mosaic=True,
            png_max_px=800,
        )
        .grey["A1"]
    )
    assert np.array_equal(renders[0]["grey"], expected)
    assert (expected != draw.UNPLACED_GREY).sum() > 1000


def test_the_caption_names_the_first_frame_that_could_not_be_read(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=2)
    scan = load_scan(seed.path)
    first, second = scan.frames("A1")[:2]
    _rewrite(scan.run_dir / first.file, "missing")
    _rewrite(scan.run_dir / second.file, "garbage")
    draw.draw_scan(scan, tmp_path / "out", max_px=800)
    (line,) = [
        text
        for text in renders[0]["caption"].split("\n")
        if "could not be read" in text
    ]
    assert line.startswith(f"2 of {len(scan.frames('A1'))} frames could not be read; ")
    assert f"first: {first.file}: the file is missing" in line
    assert second.file not in line


def test_equal_orders_keep_the_records_order_whatever_the_names_say() -> None:
    result = _route((_stop(4, "A1", 1), _stop(4, "A1", 0), _stop(3, "A1", 2)))
    stops = draw.stops_of_well(draw.RouteView("route", None, None, result), "A1")
    assert [s.name for s in stops] == ["A1_r002", "A1_r001", "A1_r000"]


def test_a_stops_number_is_centred_on_the_footprints_centre() -> None:
    # The centre of a "0" (a symmetric glyph) is the centre of its ink.
    canvas, image = _render(stops=[("0", 1000.0, -500.0)], caption="")
    column, row = _pixel_of(canvas, 1000.0, -500.0)
    ink = [
        (c, r)
        for c in range(column - 10, column + 11)
        for r in range(row - 10, row + 11)
        if _clear_green(image, (c, r, c + 1, r + 1))
    ]
    assert len(ink) > 10
    assert sum(c for c, _ in ink) / len(ink) == pytest.approx(column, abs=2.0)
    assert sum(r for _, r in ink) / len(ink) == pytest.approx(row, abs=2.0)


def test_a_stop_five_pixels_off_the_picture_is_drawn_and_fifty_are_left_out(
    tmp_path: Path, renders: list[dict[str, Any]]
) -> None:
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    canvas = draw.canvas_of(load_scan(seed.path).geometry["A1"], 800)
    right = canvas.x_min_um + canvas.width_px * canvas.um_per_px

    def beyond(pixels: float) -> float:
        return right + pixels * canvas.um_per_px

    stops = (
        _stop(0, "A1", 0, 1000.0, -500.0),
        _stop(1, "A1", 1, beyond(5), -500.0),
        _stop(2, "A1", 2, beyond(50), -500.0),
    )
    _write_route(seed.path, _route(stops), params=CHOSE)
    scan = load_scan(seed.path)
    (picture,) = draw.draw_scan(scan, tmp_path / "out", max_px=800).pictures
    assert (picture.stops, picture.off_picture) == (3, 1)
    assert [text for text, _, _ in renders[0]["stops"]] == ["1", "2"]
    assert "1 stop outside the picture is not drawn" in renders[0]["caption"]


# ------------------------------------------------------------- the run folder


def test_the_run_folder_in_every_spelling_is_replaced_and_a_dot_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = Path("runs") / "a b"
    folder.mkdir(parents=True)
    resolved = str(folder.resolve())
    # One spelling at a time, so that dropping one is seen where it differs from the
    # others: ``as_posix`` and ``str`` are the same string on POSIX and differ on
    # Windows, where this test runs it (FM-43).
    for spelling in (str(folder), folder.as_posix(), resolved):
        assert draw.scrub_run_folder(f"x {spelling} y", folder) == "x <run folder> y"
    # A folder written as a dot or two is not a path to look for: every dot would go.
    for dots in (".", ".."):
        assert draw.scrub_run_folder("a.b ..c", Path(dots)) == "a.b ..c"
    # ... but the absolute path of the folder it stands for still is.
    here = str(tmp_path.resolve())
    assert draw.scrub_run_folder(f"in {here}", Path(".")) == "in <run folder>"


def test_a_run_document_in_the_current_folder_keeps_its_cells_line_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `cd <run folder>` then `smc dataset draw run.json`: the folder is Path(".").
    seed = seed_run(tmp_path, cells=[], wells=("A1",), rows=1, columns=1)
    (tmp_path / seed.result.file).unlink()
    monkeypatch.chdir(tmp_path)
    scan = load_scan(Path("run.json"))
    assert scan.run_dir == Path()
    line = draw.caption_lines(scan, "A1", None, "none", stops=0, frames=1)[-1]
    assert line == (
        "cells: none (cells file cells/t1.json is missing from the run folder .)"
    )
