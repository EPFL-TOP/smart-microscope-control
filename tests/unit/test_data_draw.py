"""``smc.data.draw``: the drawing of a scan's wells, with no stand (algorithms design §6, #568).

Pixels are read with Pillow's ``getpixel`` at positions computed here from
``Canvas.px_of`` and ``smc.plugins.geometry.footprint``, never from the code under
test (FM-124). Every colour check is at a footprint's edge, a wall point or a marker,
at least 5 pixels from any text, because a text has a black stroke.
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from select_fields_seed import (
    PIXEL_UM,
    SHAPE,
    UPRIGHT,
    cell_at,
    load_script,
    matrix_of,
    record,
    seed_run,
)
from smc.data import Scan, draw, load_scan
from smc.plugins.geometry import Matrix2, footprint
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
