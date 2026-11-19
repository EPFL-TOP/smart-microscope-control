"""``smc.data.draw``: the drawing of a scan's wells, with no stand (algorithms design §6, #568).

Pixels are read with Pillow's ``getpixel`` at positions computed here from
``Canvas.px_of`` and ``smc.plugins.geometry.footprint``, never from the code under
test (FM-124). Every colour check is at a footprint's edge, a wall point or a marker,
at least 5 pixels from any text, because a text has a black stroke.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from select_fields_seed import SHAPE, UPRIGHT, load_script
from smc.data import draw
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.slots import Cell, WellGeometry

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
