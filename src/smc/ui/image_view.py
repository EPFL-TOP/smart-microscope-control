"""The image view: one frame with its zoom, its scale bar and its boxes (m5 §24, #631).

A component of its own, not a region of the plate page: the page builds one and
feeds it from the controller, and a notebook builds one and feeds it from a run
folder (``docs/ui/views-in-a-notebook.md``), so the frame looks the same in both.
Of ``smc`` it imports ``smc.ui.frames`` (numpy only) and nothing else: no
hardware, no controller, no page, no plugin registry, so that ``smc.analysis`` may
import it on a laptop with no Micro-Manager (``analysis.md`` §2.8). A test walks
its imports and loads it in a fresh interpreter to keep it so.

Three rules the view keeps, each measured (m5 §24.1):

* **The coordinates are micrometres when the pixel size is known**, pixels when it
  is not. Bokeh's scale bar reads the range it is given and has no factor from
  pixels, so a view in pixels has no scale bar and says so in its title: unknown
  beats guessed (D4).
* **Row 0 is on top by the range, not by the array.** ``y_range.flipped`` puts the
  image glyph's first row at the top, so the array shown is the frame's own and a
  box given in the frame's pixels falls on the pixels it names.
* **The figure is never moved.** *Enlarge* adds a CSS class to the view's root
  that turns it into an overlay of the whole window; a figure that left and came
  back would be rebuilt, and the ranges of a DataRange1d keep every bound they
  have had (FM-174).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import panel as pn
from bokeh.models import (
    ColumnDataSource,
    LabelSet,
    LinearColorMapper,
    MetricLength,
    ScaleBar,
)
from bokeh.palettes import Greys256
from bokeh.plotting import figure

from smc.ui.frames import Box, display_image

__all__ = [
    "BOX_TEXT_CHARS",
    "ENLARGE_LABEL",
    "LIVE_MAX_SIDE_PX",
    "NO_SCALE_TEXT",
    "REDUCE_LABEL",
    "STILL_MAX_SIDE_PX",
    "Box",
    "ImageView",
]

#: How many pixels of a frame's longer side reach the browser: Live's frames are strided to
#: 1024, as the page always has (m5 §5.3); a still frame (a snap, a plugin's frame, a scanned
#: field) to 2048, so that a zoom into it shows the camera's own pixels on every camera smc
#: drives (D5). Neither changes the coordinates: the extent drawn is the frame's own (FM-136).
LIVE_MAX_SIDE_PX = 1024
STILL_MAX_SIDE_PX = 2048

#: A box's text is cut here: a cells file is data (FM-125), and a label of a few hundred
#: characters would cover the field.
BOX_TEXT_CHARS = 60

ENLARGE_LABEL = "Enlarge"
REDUCE_LABEL = "Reduce"

#: What the title says when the pixel size is unknown and the view is in pixels (D4).
NO_SCALE_TEXT = "no scale bar: pixel size unknown"

#: The class the root gains while the view is enlarged, and the one the figure's box carries:
#: the stylesheet names both, and a test reads both from here.
_ENLARGED_CLASS = "smc-image-enlarged"
_BOX_CLASS = "smc-image-box"

#: The enlarged view (D3): the root over the whole window, the figure's box taking what the
#: bar above it and the controls below it leave. Measured on Panel 1.11.1 (m5 §24.1): a height
#: left ``auto`` without the flex rule collapsed the canvas to 35 px, and a fixed ``top`` below
#: the page's header cannot be written in CSS, because that header's height changes with its
#: outcome lines. The class is on the root and the rule for the box is under it, since the
#: root's own shadow root is the one this stylesheet is loaded into. ``min-height`` keeps a
#: canvas that a short window would squeeze to nothing, and the overlay scrolls instead.
_ENLARGED_CSS = f"""
:host(.{_ENLARGED_CLASS}) {{
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 100vh !important;
  width: auto !important;
  z-index: 1000;
  background-color: white;
  overflow-y: auto;
}}
:host(.{_ENLARGED_CLASS}) .{_BOX_CLASS} {{
  flex: 1 1 auto;
  height: auto !important;
  min-height: 200px;
}}
"""

_EMPTY_FRAME: dict[str, list[Any]] = {"image": [], "dw": [], "dh": []}
_EMPTY_BOXES: dict[str, list[Any]] = {
    "left": [],
    "right": [],
    "top": [],
    "bottom": [],
    "colour": [],
    "text": [],
}


class ImageView:
    """A frame as the page and a notebook show it, with its boxes over it (m5 §24.3).

    ``figure``, ``source``, ``boxes_source``, ``scale_bar`` and ``enlarge_toggle`` are
    public because the page's tests and ``/ui-check`` read them; the view is fed through
    ``show`` and ``clear`` only. Bokeh models are touched by the thread that owns the
    document: the page's callbacks, a notebook's cell.
    """

    figure: Any
    source: Any
    boxes_source: Any
    scale_bar: Any
    enlarge_toggle: pn.widgets.Toggle

    def __init__(
        self, *, height_px: int = 360, enlargeable: bool = False, title: str = ""
    ) -> None:
        """Build the models, with no frame in them and nothing on a server.

        Args:
            height_px: The height of the figure's box while the view is not enlarged.
            enlargeable: Whether the *Enlarge* toggle is offered. A notebook's cell is
                the whole of its output: only a page has a window to cover.
            title: The figure's title before the first frame.
        """
        self._height_px = height_px
        self._root: pn.viewable.Viewable | None = None
        self._callbacks: list[Callable[[bool], None]] = []
        self.source = ColumnDataSource(dict(_EMPTY_FRAME))
        self.boxes_source = ColumnDataSource(dict(_EMPTY_BOXES))
        self.figure = figure(
            match_aspect=True,
            sizing_mode="stretch_both",
            title=title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        # Row 0 on top without flipping the array (m5 §24.1): the glyph's first row is at
        # its ``y``, and the flipped range puts ``y = 0`` at the top.
        self.figure.y_range.flipped = True
        self.figure.xaxis.visible = False
        self.figure.yaxis.visible = False
        self.figure.xgrid.visible = False
        self.figure.ygrid.visible = False
        # 8-bit grey from 0 to 255 (``display_image``), whatever the camera's bit depth is.
        self.figure.image(
            image="image",
            x=0,
            y=0,
            dw="dw",
            dh="dh",
            source=self.source,
            color_mapper=LinearColorMapper(palette=Greys256, low=0, high=255),
        )
        # A box is an outline and its text at its top left, both in its own colour: over the
        # same source, so that they follow the zoom and the enlarged view together.
        self.figure.quad(
            left="left",
            right="right",
            top="top",
            bottom="bottom",
            source=self.boxes_source,
            fill_alpha=0,
            line_color="colour",
            line_width=2,
        )
        self.figure.add_layout(
            LabelSet(
                x="left",
                y="top",
                text="text",
                source=self.boxes_source,
                text_color="colour",
                text_baseline="bottom",
                text_font_size="11px",
            )
        )
        self.scale_bar = ScaleBar(
            range=self.figure.x_range,
            unit="µm",
            dimensional=MetricLength(),
            location="bottom_right",
            bar_length=0.2,
            visible=False,
        )
        self.figure.add_layout(self.scale_bar)
        self.enlarge_toggle = pn.widgets.Toggle(
            label=ENLARGE_LABEL,
            value=False,
            visible=enlargeable,
            align="start",
        )
        self.enlarge_toggle.param.watch(self._enlarge_changed, "value")

    def show(
        self,
        image: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
        *,
        bit_depth: int,
        pixel_size_um: float,
        boxes: Sequence[Box] = (),
        title: str = "",
        max_side_px: int = STILL_MAX_SIDE_PX,
    ) -> None:
        """Draw ``image`` and ``boxes`` over it, and say in the title what the view could not.

        The extent drawn is the frame's own: ``width x pixel_size_um`` micrometres when
        the pixel size is known, the frame's pixels when it is not, whatever ``max_side_px``
        strides the picture to (FM-136). ``bit_depth`` ``0`` is "unknown", as ``Frame``
        says, and the picture is stretched as ``display_image`` does. The title is
        ``title``, then ``NO_SCALE_TEXT`` when there is no scale, then how many boxes were
        not drawn, each after a ``"; "``. The view is left as it was when the frame is
        refused.

        Raises:
            ValueError: The frame is not 2-D ``uint8`` or ``uint16`` (from ``display_image``).
        """
        shown = display_image(image, bit_depth=bit_depth, max_side_px=max_side_px)
        height, width = image.shape
        known = math.isfinite(pixel_size_um) and pixel_size_um > 0.0
        scale = pixel_size_um if known else 1.0
        drawn = [box for box in boxes if _encloses_an_area(box)]
        notes = [title] if title else []
        if not known:
            notes.append(NO_SCALE_TEXT)
        skipped = len(boxes) - len(drawn)
        if skipped:
            notes.append(f"{skipped} box{'' if skipped == 1 else 'es'} not drawn")
        self.source.data = {
            "image": [np.ascontiguousarray(shown)],
            "dw": [width * scale],
            "dh": [height * scale],
        }
        self.boxes_source.data = {
            "left": [box.col0 * scale for box in drawn],
            "right": [box.col1 * scale for box in drawn],
            "top": [box.row0 * scale for box in drawn],
            "bottom": [box.row1 * scale for box in drawn],
            "colour": [box.colour for box in drawn],
            "text": [box.text[:BOX_TEXT_CHARS] for box in drawn],
        }
        self.scale_bar.visible = known
        self.figure.title.text = "; ".join(notes)

    def clear(self, title: str) -> None:
        """Show no frame and no box, with ``title`` saying why."""
        self.source.data = dict(_EMPTY_FRAME)
        self.boxes_source.data = dict(_EMPTY_BOXES)
        self.scale_bar.visible = False
        self.figure.title.text = title

    def on_enlarge(self, callback: Callable[[bool], None]) -> None:
        """Call ``callback(enlarged)`` after the toggle changes the view, in the order they were given.

        The page shows its second STOP from one (D3). A callback is called as it is: the
        page wraps its own in the guard that makes a failure a line of the page.
        """
        self._callbacks.append(callback)

    def panel(
        self,
        *,
        toolbar: Sequence[pn.viewable.Viewable] = (),
        below: Sequence[pn.viewable.Viewable] = (),
    ) -> pn.viewable.Viewable:
        """The view's root: ``toolbar`` and the *Enlarge* toggle, the figure, then ``below``.

        Built once. A second call returns the same object and **ignores its arguments**:
        a Panel object belongs to one place in one layout, so a view has one root, and
        building a second from other widgets would be a programming error that nothing
        would report. ``get_root()`` works with no server, as a notebook's cell calls it.
        """
        if self._root is None:
            figure_box = pn.Column(
                pn.pane.Bokeh(self.figure, sizing_mode="stretch_both"),
                height=self._height_px,
                sizing_mode="stretch_width",
                css_classes=[_BOX_CLASS],
            )
            self._root = pn.Column(
                pn.Row(*toolbar, self.enlarge_toggle),
                figure_box,
                *below,
                sizing_mode="stretch_width",
                stylesheets=[_ENLARGED_CSS],
            )
        return self._root

    def _enlarge_changed(self, event: Any) -> None:
        """Set the class and the label, then tell the callbacks (the order D3 and the page rely on)."""
        enlarged = bool(event.new)
        with pn.io.unlocked():
            self.enlarge_toggle.label = REDUCE_LABEL if enlarged else ENLARGE_LABEL
            if self._root is not None:
                self._root.css_classes = [_ENLARGED_CLASS] if enlarged else []
        for callback in self._callbacks:
            callback(enlarged)


def _encloses_an_area(box: Box) -> bool:
    """A box with four finite edges and a positive width and height: anything else is data gone wrong (D6, FM-125)."""
    edges = (box.row0, box.col0, box.row1, box.col1)
    return all(math.isfinite(e) for e in edges) and (
        box.row1 > box.row0 and box.col1 > box.col0
    )
