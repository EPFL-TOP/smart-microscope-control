"""The drawing of a scan's wells: the frames as a mosaic, the wall, the cells, the chosen fields (algorithms design §6).

Moved from ``scripts/dev/select_fields.py``, which now imports it: the owner and a
collaborator judge a selection by looking at it, and both look at the same
picture. Pillow and numpy only; this module never imports the hardware layer, the
test helpers, the page or the CLI (``tests/test_kit_imports.py`` keeps it so), and
``smc.data`` does not import it, so that ``import smc.data`` stays free of Pillow.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from smc.data.scan import DatasetError, Scan
from smc.plugins.geometry import Matrix2, footprint
from smc.plugins.slots import (
    Cell,
    RouteResult,
    ScanFrame,
    WellGeometry,
    normalise_well,
)
from smc.run.files import write_atomically

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


class Mosaic:
    """One well's mosaic: a canvas that frames are added to, one at a time, and read back as 8 bits.

    It is the canvas, the reduction and the tile's matrix of
    ``scripts/dev/select_fields.py``'s ``read_pixels``, kept together so that a
    frame is read, reduced and dropped and only the canvas of the well being drawn
    is held. The camera is the **scan's** (its matrix and its shape: FM-124), the
    tile's matrix is the camera's times the reduction factor, and the canvas is cut to
    the well and never to the frames (FM-136).

    Raises:
        ValueError: From the constructor, the pixel size is not finite and positive
            (``reduction_factor``); from ``add``, the camera's matrix cannot be
            inverted (``place_frame``). Both propagate: they are the scan's, not a
            frame's.
    """

    def __init__(
        self,
        geometry: WellGeometry,
        *,
        max_px: int,
        pixel_size_um: float,
        matrix_um_per_px: Matrix2,
        image_shape_px: tuple[int, int],
    ) -> None:
        self.canvas = canvas_of(geometry, max_px)
        self.reduction = reduction_factor(
            self.canvas.um_per_px, pixel_size_um, image_shape_px
        )
        (m00, m01), (m10, m11) = matrix_um_per_px
        k = self.reduction
        self.tile_matrix_um_per_px: Matrix2 = ((m00 * k, m01 * k), (m10 * k, m11 * k))
        self._pixels = self.canvas.blank()

    def add(self, frame: npt.NDArray[Any], x_um: float, y_um: float) -> None:
        """Reduce ``frame`` by block means and place it, centred at the stage point ``(x_um, y_um)``.

        A later frame overwrites an earlier one where they overlap, as the script's.
        A frame far away places nothing and raises nothing (``place_frame`` clamps
        its box on floats first).
        """
        place_frame(
            self._pixels,
            block_mean(frame, self.reduction),
            x_um,
            y_um,
            matrix=self.tile_matrix_um_per_px,
            x_min_um=self.canvas.x_min_um,
            y_max_um=self.canvas.y_max_um,
            um_per_px=self.canvas.um_per_px,
        )

    def grey(self) -> npt.NDArray[np.uint8]:
        """The canvas stretched to 8 bits, a pixel no frame reached dark grey (``stretch_to_uint8``)."""
        return stretch_to_uint8(self._pixels)


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
    return f"{frame.file}: {scrub_run_folder(reason, run_dir)}"


def scrub_run_folder(text: str, run_dir: Path) -> str:
    """``text`` with the run folder's path (as written, as POSIX and resolved) replaced by ``<run folder>``.

    A picture is shared and pasted into public reports, and a message that came from
    a file or from the file system may quote the absolute path of this PC (FM-50). A
    run folder written as ``.`` or ``..`` (the document is in the current folder) is
    not replaced as written, or every dot of the text would go: the resolved path is.
    """
    folders = {str(run_dir), run_dir.as_posix()}
    with contextlib.suppress(OSError):
        folders.add(str(run_dir.resolve()))
    for folder in sorted(folders, key=len, reverse=True):
        if folder.strip("./\\"):
            text = text.replace(folder, "<run folder>")
    return text


# ====================================================================== The route
#
# What is drawn of a selection is what the ``route`` slot holds, whichever plugin made
# it (``route``, or ``route-prototype`` of ``scripts/dev/select_fields.py``): the
# stops of a well, numbered from 1 in the order they are visited. The route's own
# camera is the one that draws them (FM-124), and its record is read as a result the
# collaborator may have edited: a record that is not a route result is a reason in the
# caption and never an exception.

#: Every string of a record is clipped to this many characters in a caption (FM-74).
_CLIP = 120
#: The classes' names and counts are listed without limit by the cells stage; the
#: caption lists at most this many labels that the cells stage does not know.
_LABELS_LISTED = 8


@dataclass(frozen=True, slots=True)
class Stop:
    """One stop of the route in one well: its rank there, from 1, and where it is.

    ``rank`` is the stop's place among the well's stops by the record's ``order``,
    never parsed from ``name``: it is the ``k + 1`` of ``<well>_r<k:03d>``, so "stop 3"
    on a picture is the ``E8_r002`` that the NIS list names. ``order`` is the route's
    own, plate-wide (the second well's first stop is not 0).
    """

    rank: int
    order: int
    name: str
    x_um: float
    y_um: float
    score: float | None


@dataclass(frozen=True, slots=True)
class RouteView:
    """The route that is drawn: the record's plugin, what it says it chose with, and its result.

    ``selector`` and ``orderer`` are the names the record's ``params`` hold, ``None``
    for a record made before the selection contract (or by the prototype script),
    which does not say: unknown is not guessed.
    """

    plugin: str
    selector: str | None
    orderer: str | None
    result: RouteResult


def _first_problem(error: ValidationError) -> str:
    """``<where>: <what>`` of the first validation error, on one line."""
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "result"
    return f"{where}: {first['msg']}"


def _name_in(params: Mapping[str, Any], key: str) -> str | None:
    """The record's name for ``key`` when it is a non-empty string; anything else is "not recorded" (FM-125)."""
    value = params.get(key)
    return value if isinstance(value, str) and value else None


def route_of(scan: Scan) -> tuple[RouteView | None, str | None]:
    """The route to draw, or ``(None, why not)``: the slot's latest record when it is usable on the scan.

    Usable is what ``RunDocument.unusable`` says for every reader of the slot (it
    exists, ended ``done``, is not stale, was run on the scan's profile and was not a
    dry run) **and** a record whose result this smc reads as a ``RouteResult``. Every
    way it is not is a sentence for the caption, with what to run again.
    """
    document = scan.document
    reason = document.unusable("route", profile=scan.profile, dry_run=False)
    if reason is not None:
        return None, reason
    record = document.provider("route")
    if record is None or record.result is None:
        return None, "the route record has no result; run route again"
    try:
        result = RouteResult.model_validate(record.result)
    except ValidationError as exc:
        return None, (
            f"the route record is not a route result this smc reads "
            f"({_first_problem(exc)}); run route again"
        )
    return (
        RouteView(
            plugin=record.plugin,
            selector=_name_in(record.params, "selector"),
            orderer=_name_in(record.params, "orderer"),
            result=result,
        ),
        None,
    )


def stops_of_well(view: RouteView, well: str) -> tuple[Stop, ...]:
    """The stops of ``well`` in visiting order, ranked from 1 there.

    The sort is stable: two stops with the same ``order`` keep the record's order.
    ``well`` is compared as a person types it (``b02`` is ``B2``).
    """
    name = normalise_well(well)
    mine = [p for p in view.result.positions if normalise_well(p.well) == name]
    mine.sort(key=lambda p: p.order)
    return tuple(
        Stop(
            rank=rank,
            order=p.order,
            name=p.name,
            x_um=p.x_um,
            y_um=p.y_um,
            score=p.score,
        )
        for rank, p in enumerate(mine, start=1)
    )


def label_of(rank: int, score: float | None) -> str:
    """The text of a stop's number: ``3``, or ``3 (0.823)`` with the selector's score in three significant digits.

    A score of ``0.0`` is a score (``is None`` and not falsy). The ``measures`` the
    selector gave are not drawn: they are in the record, and a picture holds one
    number per stop.
    """
    return str(rank) if score is None else f"{rank} ({score:.3g})"


def _record_text(value: str, run_dir: Path | None = None) -> str:
    """A string that came from a file, as one short ASCII line: scrubbed of the run folder, collapsed, escaped, clipped.

    A record can hold anything: a name of a million characters (FM-74), a newline
    that would add lines to the caption, a control character, a path of the PC it
    was made on (FM-50). The head is cut first so that the work is bounded.
    """
    head = value[: 4 * _CLIP]
    if run_dir is not None:
        head = scrub_run_folder(head, run_dir)
    text = ascii_text(" ".join(head.split()))
    text = "".join(c if " " <= c <= "~" else "?" for c in text)
    return text if len(text) <= _CLIP else text[: _CLIP - 3] + "..."


def _stops_text(count: int) -> str:
    if count == 0:
        return "no stop"
    return "1 stop" if count == 1 else f"{count} stops"


def _route_lines(
    well: str, view: RouteView | None, absent: str | None, stops: int
) -> list[str]:
    """The caption's lines about the route: the well and its stops, who chose them and with what, and the field."""
    if view is None:
        return [f"{well}: no route drawn", _record_text(absent or "no route record")]
    result = view.result
    labels = ", ".join(_record_text(label) for label in result.labels)
    key = normalise_well(well)
    count = next(
        (n for name, n in result.uncovered.items() if normalise_well(name) == key),
        None,
    )
    if count is None:
        uncovered = "uncovered cells: not counted"
    else:
        uncovered = (
            f"uncovered {labels} cells: {count}"
            if labels
            else (f"uncovered cells: {count}")
        )
    plugin = _record_text(view.plugin)
    if view.selector is None and view.orderer is None:
        chose = f"{plugin}: selector and orderer not recorded"
    else:
        selector = _record_text(view.selector) if view.selector else "not recorded"
        orderer = _record_text(view.orderer) if view.orderer else "not recorded"
        chose = f"{plugin}: selector {selector}, orderer {orderer}"
    width, height = result.fov_um
    fields = f"fields {width:.0f} x {height:.0f} um"
    if result.objective is not None:
        fields += f" at {_record_text(result.objective)}"
    return [f"{well}: {_stops_text(stops)}; {uncovered}", chose, fields]


def _cells_line(scan: Scan, well: str) -> str:
    """``cells: single (gold) 5, doublet (cyan) 3``: this well's cells per class with the colour each is drawn in, or why there are none."""
    if scan.cells is None:
        return f"cells: none ({_record_text(scan.cells_absent or 'no reason given', scan.run_dir)})"
    classes = _classes_of(scan, well)
    counts = Counter(cell.label for cell in scan.cells_of(well=well))
    parts = [
        f"{_record_text(label)} ({CLASS_COLOURS[k % len(CLASS_COLOURS)][0]}) "
        f"{counts.get(label, 0)}"
        for k, label in enumerate(classes)
    ]
    # A label the cells stage does not know is drawn white, and is listed so.
    strangers = sorted(set(counts) - set(classes))
    parts += [
        f"{_record_text(label)} (white) {counts[label]}"
        for label in strangers[:_LABELS_LISTED]
    ]
    if len(strangers) > _LABELS_LISTED:
        parts.append(f"{len(strangers) - _LABELS_LISTED} more labels (white)")
    return "cells: " + ", ".join(parts)


def _classes_of(scan: Scan, well: str) -> tuple[str, ...]:
    """The classes in the cells stage's order; a scan built with cells and no cells result has the labels its cells hold, sorted; one with no cells has none."""
    if scan.cells_result is not None:
        return tuple(scan.cells_result.classes)
    if scan.cells is None:
        return ()
    return tuple(sorted({cell.label for cell in scan.cells_of(well=well)}))


def caption_lines(
    scan: Scan,
    well: str,
    view: RouteView | None,
    absent: str | None,
    *,
    stops: int,
    frames: int,
    unreadable: int = 0,
    first_problem: str | None = None,
    off_picture: int = 0,
) -> list[str]:
    """The caption of a well's picture, one string per line, ASCII, each string of a record clipped to 120 characters.

    With a route: the well, its stops and its uncovered cells; the plugin, the
    selector and the orderer it names (or that it does not); the field. Without one:
    the well and why. Then, always, the cells per class with the colour each is drawn
    in, and only when it applies the frames that could not be read and the stops
    outside the picture. The lines are wrapped to the picture's width by
    ``render_png``.
    """
    lines = _route_lines(well, view, absent, stops)
    lines.append(_cells_line(scan, well))
    if unreadable:
        lines.append(
            f"{unreadable} of {frames} frames could not be read; first: "
            f"{_record_text(first_problem or 'no reason given')}"
        )
    if off_picture:
        lines.append(
            "1 stop outside the picture is not drawn"
            if off_picture == 1
            else f"{off_picture} stops outside the picture are not drawn"
        )
    return lines


# ====================================================================== The pictures

#: The longest side of a picture, in pixels: the default and the bounds of ``max_px``.
#: The canvas, its 8-bit stretch and the PNG's encoding grow with the square of it
#: (165 MB of RSS at 1600 px, 2.5 GB at 8000: measured), so it is bounded above (FM-74).
DEFAULT_MAX_PX = 1600
MIN_MAX_PX = 256
MAX_MAX_PX = 4000

#: What a well's name may hold to be the name of a file: a well of a plate is letters and
#: digits, and a name from a record never goes up a folder or into another one.
_FILE_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class WellPicture:
    """What was drawn of one well, and whether its file was written.

    ``frames`` is the well's frames and ``placed`` those that were read and put on the
    mosaic (``unreadable`` the others). ``stops`` is the route's stops in the well, as
    the caption counts them, of which ``off_picture`` are not drawn. ``cells`` is the
    well's cells of every class. ``error`` is what the file system said, never a path.
    """

    well: str
    path: Path
    written: bool
    error: str | None
    frames: int
    placed: int
    unreadable: int
    stops: int
    off_picture: int
    cells: int


@dataclass(frozen=True, slots=True)
class Drawing:
    """What ``draw_scan`` did: one ``WellPicture`` per well in the plan's order, and what it says about the whole.

    ``route`` is the plugin of the route that was drawn and ``route_absent`` why none
    was (exactly one of the two is set); ``cells`` says whether the scan has cells.
    ``notes`` are the sentences that belong to no well.
    """

    out_dir: Path
    pictures: tuple[WellPicture, ...]
    route: str | None
    route_absent: str | None
    cells: bool
    notes: tuple[str, ...]


def _on_picture(canvas: Canvas, x_um: float, y_um: float) -> bool:
    """Whether a stage point is within 10 pixels of the picture; a point that is not finite is not (FM-74, FM-75)."""
    column, row = canvas.px_of(x_um, y_um)
    return (
        -_TEXT_REACH_PX <= column <= canvas.width_px + _TEXT_REACH_PX
        and -_TEXT_REACH_PX <= row <= canvas.height_px + _TEXT_REACH_PX
    )


def _read_problem(frame: ScanFrame, exc: DatasetError, run_dir: Path) -> str:
    """Why ``Scan.image`` refused a frame, in the words of its cause and without the PC's paths (FM-50).

    ``Scan.image`` wraps the cause in a message that starts with the frame's name and
    file, and ``frame_problem`` names the file itself; the cause is what the system
    said (``strerror``, not a path), and a refusal with none is its own text less that
    prefix.
    """
    cause = exc.__cause__
    if isinstance(cause, Exception):
        return frame_problem(frame, cause, run_dir)
    prefix = f"frame {frame.name} ({frame.file}): "
    return frame_problem(frame, ValueError(str(exc).removeprefix(prefix)), run_dir)


def _camera_error(exc: ValueError) -> DatasetError:
    return DatasetError(
        f"the scan's camera cannot be drawn: {exc}; its record in the run document "
        f"holds a pixel size or a matrix that is not one"
    )


def draw_well(
    scan: Scan,
    well: str,
    view: RouteView | None,
    route_absent: str | None,
    *,
    max_px: int = DEFAULT_MAX_PX,
) -> tuple[Image.Image, WellPicture]:
    """The picture of one well and what it holds: the frames, the wall, the cells and the route's stops of that well.

    Each frame is read once, in the scan's order, and a later one overwrites an
    earlier where they overlap. A frame that cannot be read, is not the scan's shape
    or (a float) holds a value that is not finite is **counted and skipped**: a
    picture with a hole in it and a caption that says so is worth more than none
    (failures are findings). A stop whose centre is not within 10 pixels of the
    picture is left out altogether (no footprint, no number, no vertex of the path)
    and counted, and the stops keep the ranks of the well's whole list, so a third
    stop stays ``3`` when the second is left out.

    The picture's path is only the file's name; ``draw_scan`` puts it in its folder.

    Raises:
        DatasetError: The scan's camera cannot be drawn (a pixel size that is not a
            size, a matrix that cannot be inverted): every well would fail the same way.
    """
    geometry = scan.geometry[well]
    frames = scan.frames(well)
    try:
        mosaic = Mosaic(
            geometry,
            max_px=max_px,
            pixel_size_um=scan.pixel_size_um,
            matrix_um_per_px=scan.matrix_um_per_px,
            image_shape_px=scan.image_shape_px,
        )
    except ValueError as exc:
        raise _camera_error(exc) from exc
    unreadable = 0
    first_problem: str | None = None
    for frame in frames:
        problem: str | None = None
        try:
            pixels = scan.image(frame)
        except DatasetError as exc:
            problem = _read_problem(frame, exc, scan.run_dir)
        else:
            if (
                np.issubdtype(pixels.dtype, np.floating)
                and not np.isfinite(pixels).all()
            ):
                problem = frame_problem(
                    frame,
                    ValueError("the frame holds values that are not finite"),
                    scan.run_dir,
                )
        if problem is not None:
            unreadable += 1
            first_problem = first_problem or problem
            continue
        try:
            mosaic.add(pixels, frame.x_um, frame.y_um)
        except ValueError as exc:
            raise _camera_error(exc) from exc

    cells = scan.cells_of(well=well) if scan.cells is not None else ()
    every = stops_of_well(view, well) if view is not None else ()
    shown = [s for s in every if _on_picture(mosaic.canvas, s.x_um, s.y_um)]
    off_picture = len(every) - len(shown)
    caption = caption_lines(
        scan,
        well,
        view,
        route_absent,
        stops=len(every),
        frames=len(frames),
        unreadable=unreadable,
        first_problem=first_problem,
        off_picture=off_picture,
    )
    image = render_png(
        mosaic.grey(),
        mosaic.canvas,
        geometry,
        cells=cells,
        classes=_classes_of(scan, well),
        wanted=view.result.labels if view is not None else (),
        stops=[(label_of(s.rank, s.score), s.x_um, s.y_um) for s in shown],
        matrix=(
            view.result.matrix_um_per_px if view is not None else scan.matrix_um_per_px
        ),
        image_shape_px=(
            view.result.image_shape_px if view is not None else scan.image_shape_px
        ),
        caption="\n".join(caption),
    )
    picture = WellPicture(
        well=well,
        path=Path(f"{well}.png"),
        written=False,
        error=None,
        frames=len(frames),
        placed=len(frames) - unreadable,
        unreadable=unreadable,
        stops=len(every),
        off_picture=off_picture,
        cells=len(cells),
    )
    return image, picture


def _wells_to_draw(scan: Scan, wells: Sequence[str] | None) -> tuple[str, ...]:
    """The scan's own names of the wells asked for, in the plan's order, each once; a well the scan lacks is a refusal."""
    if wells is None:
        chosen = list(scan.wells)
    else:
        if not wells:
            raise DatasetError("no well was asked for: name wells, or none for all")
        known = {normalise_well(name): name for name in scan.wells}
        wanted: set[str] = set()
        for well in wells:
            key = normalise_well(well)
            if key not in known:
                raise DatasetError(
                    f"well {well!r} is not in this scan ({', '.join(scan.wells)})"
                )
            wanted.add(key)
        chosen = [name for name in scan.wells if normalise_well(name) in wanted]
    for name in chosen:
        if _FILE_NAME.fullmatch(name) is None:
            raise DatasetError(
                f"the scan's well {name!r} is not a name this command can write "
                f"a file for"
            )
    return tuple(chosen)


def _stops_in_other_wells(scan: Scan, view: RouteView) -> str | None:
    """A sentence for the route's stops in wells the scan does not have, or ``None``: they are not drawn, and not silently."""
    known = {normalise_well(name) for name in scan.wells}
    strangers = [
        p.well for p in view.result.positions if normalise_well(p.well) not in known
    ]
    if not strangers:
        return None
    names = sorted(set(strangers))
    listed = ", ".join(_record_text(name) for name in names[:_LABELS_LISTED])
    if len(names) > _LABELS_LISTED:
        listed += ", ..."
    return (
        f"the route holds {len(strangers)} stops in wells the scan does not have "
        f"({listed}): not drawn"
    )


def draw_scan(
    scan: Scan,
    out_dir: Path,
    *,
    wells: Sequence[str] | None = None,
    max_px: int = DEFAULT_MAX_PX,
    progress: Callable[[int, int, str], None] | None = None,
) -> Drawing:
    """Draw each well of a scan to ``<out_dir>/<well>.png``, with the route's stops when the scan has a usable one.

    The wells come in the plan's order whatever order ``wells`` named them. A well the
    scan lacks, a ``max_px`` outside ``MIN_MAX_PX`` to ``MAX_MAX_PX`` (a ``bool`` is
    not one) and an ``out_dir`` that is a file are refused **before any frame is read
    or any folder made**. ``out_dir`` is made, with its parents, when the first picture
    is written. Each PNG is encoded to bytes and written with ``write_atomically``:
    replaced whole or not at all (FM-38), the rename retried when a viewer holds the old
    picture on Windows (FM-24). Nothing else in ``out_dir`` is touched. An ``OSError``
    on one picture is its ``WellPicture.error`` and the others are still written
    (FM-34). ``progress(done, total, well)`` is called after each well.

    Raises:
        DatasetError: A refusal above, or the scan's camera cannot be drawn.
    """
    if (
        isinstance(max_px, bool)
        or not isinstance(max_px, int)
        or not MIN_MAX_PX <= max_px <= MAX_MAX_PX
    ):
        raise DatasetError(
            f"max_px must be an integer from {MIN_MAX_PX} to {MAX_MAX_PX}, "
            f"got {max_px!r}"
        )
    chosen = _wells_to_draw(scan, wells)
    try:
        is_file = out_dir.exists() and not out_dir.is_dir()
    except OSError:
        is_file = False  # the write says what is wrong, with its own reason
    if is_file:
        raise DatasetError(f"{out_dir} is a file, not a folder: name a folder")
    view, absent = route_of(scan)
    notes = [] if view is None else [_stops_in_other_wells(scan, view)]
    pictures: list[WellPicture] = []
    for done, well in enumerate(chosen, start=1):
        image, picture = draw_well(scan, well, view, absent, max_px=max_px)
        path = out_dir / f"{well}.png"
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        try:
            write_atomically(path, buffer.getvalue())
        except OSError as exc:
            picture = replace(
                picture, path=path, error=exc.strerror or type(exc).__name__
            )
        else:
            picture = replace(picture, path=path, written=True)
        pictures.append(picture)
        if progress is not None:
            progress(done, len(chosen), well)
    return Drawing(
        out_dir=out_dir,
        pictures=tuple(pictures),
        route=view.plugin if view is not None else None,
        route_absent=absent,
        cells=scan.cells is not None,
        notes=tuple(note for note in notes if note is not None),
    )
