# Views in a notebook

The plate page's views are components of their own: a notebook shows the same object the page
does, fed from a run folder with **no stand and no microscope**. This page is the image view
([m5 §24](../design/m5-web-ui.md), `smc.ui.image_view`); the plate map comes with its own issue
and will be added here.

It needs the `ui` extra (Panel and Bokeh) and `smc.data`, which reads a run folder:

```text
pip install -e ".[ui]"
```

`smc.ui.image_view` imports of `smc` only `smc.ui.frames` (numpy), so it loads without
Micro-Manager and `smc.analysis` may import it. What it draws is a frame with its **zoom and
pan**, a **scale bar in micrometres**, and **boxes** over it, in the frame's pixels.

## A scanned field with its cells

Point `RUN_DOC` at a run document whose `scan` is done (and, for the boxes, whose `cells` is
done):

```python
from pathlib import Path

RUN_DOC = Path("runs/2026-11-24/run.json")  # your run document
```

then, in the next cell:

```python
import panel as pn

from smc.data import load_scan
from smc.data.draw import CLASS_COLOURS
from smc.ui.image_view import Box, ImageView

pn.extension()
scan = load_scan(RUN_DOC)
frame = scan.frames(scan.wells[0])[0]
index = scan.result.frames.index(frame)
classes = scan.cells_result.classes if scan.cells_result is not None else ()
colour = {
    label: "#%02x%02x%02x" % CLASS_COLOURS[k % len(CLASS_COLOURS)][1]
    for k, label in enumerate(classes)
}
cells = scan.cells_of(frame=index) if scan.cells is not None else ()
boxes = [
    Box(*cell.box_px, colour=colour[cell.label], text=f"{cell.label} {cell.score:.2f}")
    for cell in cells
]
view = ImageView()
view.show(
    scan.image(frame),
    bit_depth=0,
    pixel_size_um=scan.pixel_size_um,
    boxes=boxes,
    title=f"{frame.well} {frame.name}",
)
view.panel()
```

The last line is the cell's value: Jupyter shows a Panel object as its output. Scroll over the
image to zoom, drag to pan, and use the toolbar to box-zoom or reset; the scale bar follows the
zoom.

## What the example does, and what it does not

- **The colours are `smc dataset draw`'s.** `CLASS_COLOURS` is the one definition of the picture
  and of its legend, in the cells stage's order, so a class has the same colour here and in the
  PNG.
- **`bit_depth=0` means "unknown"**, as a `Frame` of the page says it, and the picture is stretched
  between the 0.5 and 99.5 percentiles of the frame, as the page's is. A flat frame is shown against
  the dtype's range.
- **A scan with no `cells` stage shows no box.** `scan.cells` is `None` then, and the example
  passes no box; `scan.cells_absent` says why.
- **A box is `(row0, col0, row1, col1)` in the frame's pixels**, the edges, as `Cell.box_px` is.
  A box that is not finite or encloses no area is **not drawn**, and the title says how many were
  not (`; 2 boxes not drawn`): a cells file is data, and one bad cell must not hide the field.
  Its text is drawn on the canvas, cut at 60 characters.
- **Micrometres when the pixel size is known.** A scan's `pixel_size_um` is the camera's, so the
  view is in micrometres and the bar reads a length. With an unknown pixel size (`0` or not
  finite) the view is in the frame's pixels, has **no scale bar**, and its title says
  `no scale bar: pixel size unknown`.
- **It is a view of a folder, not a second live client.** A stand is held by one process, the
  page's; a notebook shows what a run left.

The example's code is run by `tests/unit/test_image_view_notebook_example.py` on a seeded scan, so
that it cannot rot. **Not verified:** the example opened in Jupyter (the venv has none): the first
person to open it says what the output cell looks like.
