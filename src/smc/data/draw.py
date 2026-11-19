"""The drawing of a scan's wells: the frames as a mosaic, the wall, the cells, the chosen fields (algorithms design §6).

Moved from ``scripts/dev/select_fields.py``, which now imports it: the owner and a
collaborator judge a selection by looking at it, and both look at the same
picture. Pillow and numpy only; this module never imports the hardware layer, the
test helpers, the page or the CLI (``tests/test_kit_imports.py`` keeps it so), and
``smc.data`` does not import it, so that ``import smc.data`` stays free of Pillow.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.slots import Cell, ScanFrame, WellGeometry

# The mosaic is for *looking*, not for measuring: a frame is reduced by block means,
# placed on the canvas by the nearest pixel, and a later frame overwrites an earlier
# one where they overlap. The placement error (up to ``k - 1`` source pixels of a
# cropped remainder) is below one canvas pixel.

#: Dark grey of a canvas pixel no frame reached, so that a hole is not a black pixel.
UNPLACED_GREY = 40
#: A canvas is the well's bounding box plus this much on each side.
_CANVAS_MARGIN = 0.02
_CIRCLE_SEGMENTS = 128
_BLUE = (90, 90, 255)
_GREEN = (0, 255, 0)
_ROUTE = (255, 255, 160)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
#: The classes' colours as ``(name, rgb)`` in the cells stage's order, the one definition
#: of the picture and of its legend: a class past the sixth takes the first's again.
CLASS_COLOURS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("gold", (255, 215, 0)),
    ("cyan", (0, 200, 255)),
    ("red", (255, 80, 80)),
    ("light green", (120, 255, 120)),
    ("magenta", (255, 120, 255)),
    ("white", _WHITE),
)
#: A stop's number is drawn when its centre is within this many pixels of the picture.
#: Pillow's ``text`` raises ``SystemError`` at x = 1e300 (measured, Pillow 13), and a
#: number whose centre is that far away is not on the picture anyway.
_TEXT_REACH_PX = 10.0
#: A caption longer than this is cut: a string of a million characters is a record that
#: is not ours (FM-74), and measuring it character by character is not free.
_CAPTION_LIMIT = 2000


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
    grey = np.full(pixels.shape, UNPLACED_GREY, dtype=np.uint8)
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
    stops: Sequence[tuple[int | str, float, float]],
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    caption: str,
) -> Image.Image:
    """The picture of a well: the mosaic, the wall, the cells, the chosen fields numbered in route order.

    Blue is the wall; each cell is a 3-pixel dot coloured by its class (the classes in
    the cells stage's order, ``CLASS_COLOURS``), with a second, larger ring when its
    label is wanted; a chosen field is its footprint, a closed 2-pixel green line
    (turned or mirrored as the camera is) with its number at the centre; the route is
    a thin yellow line through the stops. ``stops`` are ``(number, x_um, y_um)`` in
    route order; the number is drawn as ``str(number)``, so a caller that has a score
    to show formats the text itself.

    Every text has a 1-pixel black stroke (a mosaic can be white), the caption is
    written as ASCII and wrapped to the picture's width (the bitmap font of a Pillow
    built without FreeType raises on a Greek letter), and a number whose centre is not
    within 10 pixels of the picture is not drawn (Pillow raises ``SystemError`` for a
    text at 1e300): the footprint's line clips and is harmless.
    """
    image = Image.fromarray(np.stack([grey, grey, grey], axis=-1))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def at(x_um: float, y_um: float) -> tuple[float, float]:
        return canvas.px_of(x_um, y_um)

    draw.line([at(x, y) for x, y in _outline_of(well)], fill=_BLUE, width=1)
    colours = {
        label: CLASS_COLOURS[k % len(CLASS_COLOURS)][1]
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
        # Written so that a ``nan`` is outside: every comparison with it is false.
        if not (
            -_TEXT_REACH_PX <= col <= width + _TEXT_REACH_PX
            and -_TEXT_REACH_PX <= row <= height + _TEXT_REACH_PX
        ):
            continue
        text = ascii_text(str(number))
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (col - (right - left) / 2.0 - left, row - (bottom - top) / 2.0 - top),
            text,
            fill=_GREEN,
            font=font,
            stroke_width=1,
            stroke_fill=_BLACK,
        )
    text = ascii_text(caption)
    if len(text) > _CAPTION_LIMIT:
        text = text[: _CAPTION_LIMIT - 3] + "..."
    draw.multiline_text(
        (4, 4),
        "\n".join(wrap_text(draw, text, font, max(1.0, width - 8.0))),
        fill=_WHITE,
        font=font,
        stroke_width=1,
        stroke_fill=_BLACK,
    )
    return image


def ascii_text(text: str) -> str:
    """``text`` with everything that is not ASCII written as a Python escape (a lambda is ``u03bb`` after a backslash).

    Pillow's bitmap default font, which is what a build without FreeType gives,
    raises ``UnicodeEncodeError`` on a Greek letter (measured), and an objective's
    name may hold one; the FreeType font draws it, and a picture that is read on
    three platforms says the same thing on each (FM-20).
    """
    return text.encode("ascii", "backslashreplace").decode("ascii")


def wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: Any, width_px: float
) -> list[str]:
    """``text`` as lines no wider than ``width_px`` when drawn in ``font``, broken at spaces and at newlines.

    Greedy by words. A word that is wider than the picture on its own is cut where it
    reaches the width, by accumulating characters, so that the work is linear in its
    length: a record that holds a string of a million characters costs a
    million measurements and not a million squared (FM-74). A line always takes at
    least one character, so a width smaller than one glyph still ends.
    """

    def fits(line: str) -> bool:
        return bool(draw.textlength(line, font=font) <= width_px)

    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            joined = f"{line} {word}" if line else word
            if fits(joined):
                line = joined
                continue
            if line:
                lines.append(line)
            start = 0
            while True:
                end = start + 1
                while end < len(word) and fits(word[start : end + 1]):
                    end += 1
                if end >= len(word):
                    break
                lines.append(word[start:end])
                start = end
            line = word[start:]
        lines.append(line)
    return lines


def _font(size: int) -> Any:
    """Pillow's default font at ``size`` pixels, so that a stop's number can be read on a picture of 1600 pixels.

    A size is accepted from Pillow 10.1 and needs FreeType; an older Pillow, or a build
    without it, gives the default 10-pixel font, which is small and still correct.
    """
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, ImportError):
        return ImageFont.load_default()


def frame_problem(frame: ScanFrame, exc: Exception, run_dir: Path) -> str:
    """Why a frame could not be read, without a path of this PC: the table is pasted into public reports (FM-50).

    A missing file and an ``OSError`` say what the system said and not its text, which
    quotes the file's absolute path; any other message is scrubbed of the run folder.
    """
    if isinstance(exc, FileNotFoundError):
        reason = "the file is missing"
    elif isinstance(exc, OSError):
        reason = exc.strerror or type(exc).__name__
    else:
        reason = str(exc)
    folders = {str(run_dir), run_dir.as_posix()}
    with contextlib.suppress(OSError):
        folders.add(str(run_dir.resolve()))
    for folder in sorted(folders, key=len, reverse=True):
        reason = reason.replace(folder, "<run folder>")
    return f"{frame.file}: {reason}"
