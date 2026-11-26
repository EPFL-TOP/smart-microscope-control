"""The image view as a component of its own (m5 §24.3, #631): no server, no stand.

``ImageView`` is built and driven the way a notebook drives it: its models are read
(``source.data``, ``figure.y_range``, ``scale_bar.visible``) and its toggle is set,
the way ``test_plate_page.py`` drives the page. What no test here does is open a
browser: the look of the enlarged view is ``/ui-check``'s ``enlarged`` state.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from bokeh.models import LabelSet, MetricLength
from bokeh.palettes import Greys256

from smc.ui.frames import Box as FramesBox
from smc.ui.image_view import (
    BOX_TEXT_CHARS,
    ENLARGE_LABEL,
    LIVE_MAX_SIDE_PX,
    NO_SCALE_TEXT,
    REDUCE_LABEL,
    STILL_MAX_SIDE_PX,
    Box,
    ImageView,
)

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

GOLD = "#ffd700"


def _frame(rows: int, columns: int) -> np.ndarray:
    """``row * 1000 + column``: not symmetric, so a flip or a transposition shows."""
    grid = np.arange(rows)[:, None] * 1000 + np.arange(columns)[None, :]
    return grid.astype(np.uint16)


def _shown(view: ImageView, **kwargs: Any) -> ImageView:
    """A 300 x 400 frame at 0.5 µm, unless a test says otherwise."""
    arguments: dict[str, Any] = {"bit_depth": 16, "pixel_size_um": 0.5}
    arguments.update(kwargs)
    view.show(_frame(300, 400), **arguments)
    return view


def test_the_box_is_the_one_of_the_frames_module() -> None:
    # One class, imported from either module: #677 adds `smc.ui.image_view` alone to its
    # allowed imports and still builds the page's boxes from `smc.ui.frames`.
    assert Box is FramesBox


def test_a_box_is_frozen_and_slotted_with_an_empty_text_by_default() -> None:
    # Plan step 1: a box is a value that is handed to a view and kept by the caller, so it
    # cannot change under the view, and it carries no `__dict__`.
    box = Box(1, 2, 3, 4, GOLD)
    assert box.text == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.row0 = 9.0  # type: ignore[misc]
    assert not hasattr(box, "__dict__")


def test_a_known_pixel_size_draws_micrometres_and_shows_the_scale_bar() -> None:
    view = ImageView()
    assert not view.scale_bar.visible
    _shown(view)
    # 400 columns of 0.5 µm and 300 rows of 0.5 µm: the view's coordinates are µm (D4).
    assert list(view.source.data["dw"]) == [200.0]
    assert list(view.source.data["dh"]) == [150.0]
    assert view.scale_bar.visible


@pytest.mark.parametrize("unknown", [0.0, float("nan"), -0.5, float("inf")])
def test_an_unknown_pixel_size_draws_pixels_and_says_there_is_no_scale_bar(
    unknown: float,
) -> None:
    view = ImageView()
    _shown(view, pixel_size_um=0.5)
    assert view.scale_bar.visible
    _shown(view, pixel_size_um=unknown, title="frame 3")
    # Unknown beats guessed: the pixels are the frame's own and nothing says a length.
    assert list(view.source.data["dw"]) == [400]
    assert list(view.source.data["dh"]) == [300]
    assert not view.scale_bar.visible
    assert view.figure.title.text == "frame 3; " + NO_SCALE_TEXT


def test_a_title_that_is_empty_is_not_started_by_a_semicolon() -> None:
    view = ImageView()
    _shown(view, pixel_size_um=0.0)
    assert view.figure.title.text == NO_SCALE_TEXT
    _shown(view, pixel_size_um=0.5)
    assert view.figure.title.text == ""


def test_boxes_are_scaled_from_the_frame_pixels() -> None:
    view = ImageView()
    _shown(view, boxes=[Box(100, 200, 150, 260, GOLD, "single 0.93")])
    data = view.boxes_source.data
    # (row0, col0, row1, col1) in the frame's pixels, drawn at 0.5 µm per pixel.
    assert list(data["left"]) == [100.0]
    assert list(data["right"]) == [130.0]
    assert list(data["top"]) == [50.0]
    assert list(data["bottom"]) == [75.0]
    assert list(data["colour"]) == [GOLD]
    assert list(data["text"]) == ["single 0.93"]


def test_boxes_are_in_pixels_when_the_pixel_size_is_unknown() -> None:
    view = ImageView()
    _shown(view, pixel_size_um=0.0, boxes=[Box(100, 200, 150, 260, GOLD)])
    data = view.boxes_source.data
    assert list(data["left"]) == [200.0]
    assert list(data["right"]) == [260.0]
    assert list(data["top"]) == [100.0]
    assert list(data["bottom"]) == [150.0]


def test_a_box_empty_or_not_finite_is_skipped_and_counted() -> None:
    view = ImageView()
    _shown(
        view,
        title="A1",
        boxes=[
            Box(10, 10, 20, 20, GOLD),
            Box(math.nan, 10, 20, 20, GOLD),
            Box(10, 30, 20, 30, GOLD),
        ],
    )
    # A cells file is data (FM-125): one bad cell never hides the field.
    assert len(view.boxes_source.data["left"]) == 1
    assert view.figure.title.text == "A1; 2 boxes not drawn"


@pytest.mark.parametrize(
    "bad",
    [
        Box(10, 10, 5, 20, GOLD),
        Box(10, 10, 20, 5, GOLD),
        Box(10, 10, 20, math.inf, GOLD),
        Box(10, -math.inf, 20, 20, GOLD),
        Box(10, 10, math.nan, 20, GOLD),
    ],
)
def test_a_box_that_does_not_enclose_an_area_is_not_drawn(bad: Box) -> None:
    view = ImageView()
    _shown(view, title="A1", boxes=[bad])
    assert len(view.boxes_source.data["left"]) == 0
    assert view.figure.title.text == "A1; 1 box not drawn"


def test_the_two_notes_of_a_title_come_in_a_fixed_order() -> None:
    view = ImageView()
    _shown(view, pixel_size_um=0.0, title="A1", boxes=[Box(1, 1, 1, 1, GOLD)])
    assert view.figure.title.text == f"A1; {NO_SCALE_TEXT}; 1 box not drawn"


def test_a_box_text_is_cut_at_box_text_chars() -> None:
    view = ImageView()
    _shown(view, boxes=[Box(10, 10, 20, 20, GOLD, "x" * 200)])
    assert list(view.boxes_source.data["text"]) == ["x" * BOX_TEXT_CHARS]
    assert BOX_TEXT_CHARS == 60


def test_a_box_text_is_kept_as_typed_for_the_canvas() -> None:
    # Canvas text, never markup (FM-139): nothing is escaped or interpreted on the way, and
    # the text is drawn by a LabelSet (Bokeh's canvas text), never by a pane or a tooltip.
    view = ImageView()
    text = "<b>single</b> & *0.9* [x]"
    _shown(view, boxes=[Box(10, 10, 20, 20, GOLD, text)])
    assert list(view.boxes_source.data["text"]) == [text]
    (labels,) = [o for o in view.figure.center if isinstance(o, LabelSet)]
    assert labels.source is view.boxes_source
    assert labels.text == "text"


def test_a_live_frame_is_strided_to_1024_and_a_still_one_is_not() -> None:
    assert (LIVE_MAX_SIDE_PX, STILL_MAX_SIDE_PX) == (1024, 2048)
    frame = _frame(1200, 1200)
    view = ImageView()
    view.show(
        frame,
        bit_depth=16,
        pixel_size_um=0.0,
        max_side_px=STILL_MAX_SIDE_PX,
    )
    assert view.source.data["image"][0].shape == (1200, 1200)
    view.show(frame, bit_depth=16, pixel_size_um=0.0, max_side_px=LIVE_MAX_SIDE_PX)
    # A stride of 2: 600 pixels shown, and the frame's own extent drawn (FM-136).
    assert view.source.data["image"][0].shape == (600, 600)
    assert list(view.source.data["dw"]) == [1200]
    assert list(view.source.data["dh"]) == [1200]


def test_the_default_is_a_still_frames_stride() -> None:
    view = ImageView()
    view.show(_frame(1200, 1200), bit_depth=16, pixel_size_um=0.0)
    assert view.source.data["image"][0].shape == (1200, 1200)


def test_row_zero_is_drawn_on_top_by_the_range_not_by_the_array() -> None:
    frame = np.zeros((6, 10), dtype=np.uint16)
    frame[0, :] = 4000
    view = ImageView()
    view.show(frame, bit_depth=12, pixel_size_um=0.0)
    assert view.figure.y_range.flipped is True
    # Bokeh draws an image's first row at its y, which the flipped range puts on top: the
    # array is the frame's own, row 0 first.
    shown = view.source.data["image"][0]
    assert (shown[0] == 255).all()
    assert (shown[1:] == 0).all()


def test_the_figure_has_no_axes_and_keeps_its_tools() -> None:
    view = ImageView()
    assert not any(axis.visible for axis in view.figure.axis)
    assert not any(grid.visible for grid in view.figure.grid)
    tools = {type(tool).__name__ for tool in view.figure.toolbar.tools}
    assert {"PanTool", "WheelZoomTool", "BoxZoomTool", "ResetTool", "SaveTool"} <= tools
    assert view.figure.sizing_mode == "stretch_both"


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.zeros((4, 5), dtype=np.float32),
        np.zeros(7, dtype=np.uint16),
    ],
)
def test_a_frame_that_is_not_two_d_uint_raises(bad: np.ndarray) -> None:
    view = ImageView()
    _shown(view, title="before", boxes=[Box(1, 1, 2, 2, GOLD)])
    with pytest.raises(ValueError, match="is not shown"):
        view.show(bad, bit_depth=8, pixel_size_um=1.0, title="after")
    # The view still shows the frame it had: a refused frame changes nothing.
    assert view.figure.title.text == "before"
    assert len(view.boxes_source.data["left"]) == 1
    assert list(view.source.data["dw"]) == [200.0]


def test_clear_empties_the_frame_and_the_boxes() -> None:
    view = ImageView()
    _shown(view, boxes=[Box(10, 10, 20, 20, GOLD, "cell")])
    assert view.scale_bar.visible
    view.clear("no frame yet")
    assert len(view.source.data["image"]) == 0
    assert len(view.source.data["dw"]) == 0
    assert len(view.boxes_source.data["left"]) == 0
    assert not view.scale_bar.visible
    assert view.figure.title.text == "no frame yet"


def test_the_title_given_at_construction_is_the_figures() -> None:
    assert ImageView(title="no frame yet").figure.title.text == "no frame yet"


def test_enlarge_sets_the_class_relabels_and_calls_back() -> None:
    view = ImageView(enlargeable=True)
    asked: list[bool] = []
    view.on_enlarge(asked.append)
    root = view.panel()
    assert list(root.css_classes) == []
    assert view.enlarge_toggle.label == ENLARGE_LABEL
    view.enlarge_toggle.value = True
    assert list(root.css_classes) == ["smc-image-enlarged"]
    assert view.enlarge_toggle.label == REDUCE_LABEL
    assert asked == [True]
    view.enlarge_toggle.value = False
    assert list(root.css_classes) == []
    assert view.enlarge_toggle.label == ENLARGE_LABEL
    assert asked == [True, False]


def test_a_callback_runs_after_the_class_is_set() -> None:
    view = ImageView(enlargeable=True)
    root = view.panel()
    seen: list[list[str]] = []
    view.on_enlarge(lambda enlarged: seen.append(list(root.css_classes)))
    view.enlarge_toggle.value = True
    assert seen == [["smc-image-enlarged"]]


def test_every_callback_of_the_view_is_called_in_the_order_it_was_given() -> None:
    view = ImageView(enlargeable=True)
    order: list[str] = []
    view.on_enlarge(lambda enlarged: order.append("first"))
    view.on_enlarge(lambda enlarged: order.append("second"))
    view.panel()
    view.enlarge_toggle.value = True
    assert order == ["first", "second"]


def test_the_enlarge_toggle_is_hidden_unless_enlargeable() -> None:
    assert not ImageView().enlarge_toggle.visible
    assert ImageView(enlargeable=True).enlarge_toggle.visible


def test_the_toggle_is_a_light_button_with_a_label_the_page_can_find() -> None:
    # `/ui-check` finds the toggle by its text and waits for its `bk-active` class.
    toggle = ImageView(enlargeable=True).enlarge_toggle
    assert isinstance(toggle, pn.widgets.Toggle)
    assert toggle.label == "Enlarge"
    assert REDUCE_LABEL == "Reduce"


def test_panel_builds_with_no_server_and_returns_one_root() -> None:
    view = _shown(ImageView(enlargeable=True, title="no frame yet"))
    root = view.panel()
    # What a notebook does: no document, no server.
    model = root.get_root()
    assert model is not None
    assert view.panel() is root
    # The second call ignores its arguments, as the docstring says: one root per view.
    assert view.panel(toolbar=[pn.widgets.Button(label="late")]) is root


def test_panel_places_the_toolbar_the_figure_and_what_is_below() -> None:
    view = ImageView(enlargeable=True)
    first = pn.widgets.Button(label="Snap")
    below = pn.widgets.Button(label="Drive")
    root = view.panel(toolbar=[first], below=[below])
    placed = {id(o) for o in root.select()}
    assert {id(first), id(view.enlarge_toggle), id(below)} <= placed
    assert any(
        getattr(o, "object", None) is view.figure for o in root.select(pn.pane.Bokeh)
    )
    # The toolbar's own widgets come before the toggle, then the figure, then `below`.
    objects = [id(o) for o in root.select()]
    assert objects.index(id(first)) < objects.index(id(view.enlarge_toggle))
    assert objects.index(id(view.enlarge_toggle)) < objects.index(id(below))


def test_the_figures_box_has_the_class_the_enlarged_rule_names() -> None:
    root = ImageView(height_px=360).panel()
    boxes = [o for o in root.select(pn.Column) if "smc-image-box" in o.css_classes]
    assert len(boxes) == 1
    assert boxes[0].height == 360
    # The rule that gives the box the window's height is in the root's stylesheet and
    # names the same class, so that renaming one without the other is seen here.
    stylesheet = "".join(str(s) for s in root.stylesheets)
    assert ".smc-image-box" in stylesheet
    assert ":host(.smc-image-enlarged)" in stylesheet


def test_the_enlarged_rule_covers_the_window_over_the_pages_sticky_header() -> None:
    # The browser is what shows the look (`/ui-check`'s `enlarged` state), but two words of the
    # rule are a promise the tests can keep: the overlay is fixed over the window, and it is
    # above the page's header, whose z-index the page states (`_STICKY_STRIP`, m5 §24, risk 2).
    from smc.ui.plate_page import _STICKY_STRIP

    stylesheet = "".join(str(s) for s in ImageView().panel().stylesheets)
    assert "position: fixed" in stylesheet
    assert "flex: 1 1 auto" in stylesheet
    z_index = re.search(r"z-index:\s*(\d+)", stylesheet)
    assert z_index is not None
    assert int(z_index.group(1)) > int(_STICKY_STRIP["z-index"])


def test_the_figure_is_drawn_as_the_design_says() -> None:
    # What a browser draws is read from the models: every one of these was a mutant that no
    # other test noticed (a bar in metres, boxes filled over the cells, a picture whitewashed,
    # a frame stretched out of square). `/ui-check` photographs the picture, not these fields.
    view = ImageView()
    renderers = {type(r.glyph).__name__: r for r in view.figure.renderers}
    image = renderers["Image"]
    assert image.data_source is view.source
    glyph = image.glyph
    assert (glyph.image, glyph.dw, glyph.dh) == ("image", "dw", "dh")
    assert (glyph.x, glyph.y) == (0, 0)
    mapper = glyph.color_mapper
    assert (mapper.low, mapper.high) == (0, 255)
    assert list(mapper.palette) == list(Greys256)
    quad = renderers["Quad"]
    assert quad.data_source is view.boxes_source
    glyph = quad.glyph
    assert (glyph.left, glyph.right, glyph.top, glyph.bottom) == (
        "left",
        "right",
        "top",
        "bottom",
    )
    # An outline, never a fill: a filled box would hide the cell it names.
    assert glyph.fill_alpha == 0
    assert glyph.line_color == "colour"
    (labels,) = [o for o in view.figure.center if isinstance(o, LabelSet)]
    assert (labels.x, labels.y) == ("left", "top")
    assert labels.text_color == "colour"
    assert labels.text_baseline == "bottom"
    bar = view.scale_bar
    assert bar.unit == "µm"
    assert bar.range is view.figure.x_range
    assert isinstance(bar.dimensional, MetricLength)
    assert bar.location == "bottom_right"
    assert view.figure.match_aspect is True


def _tree_ids(view: ImageView) -> set[str]:
    return {model.id for model in view.panel().get_root().references()}


def test_enlarging_never_takes_the_figure_out_of_the_model_tree() -> None:
    # FM-174: a figure that leaves the model tree has its view destroyed in the browser, and
    # a DataRange1d keeps the bounds of every view it has had. *Enlarge* is a class on the root
    # and nothing else: the figure, its pane and its box are the very objects they were.
    view = _shown(ImageView(enlargeable=True))
    root = view.panel()
    pane = next(o for o in root.select(pn.pane.Bokeh) if o.object is view.figure)
    box = next(o for o in root.select(pn.Column) if "smc-image-box" in o.css_classes)
    assert view.figure.id in _tree_ids(view)
    for enlarged in (True, False, True, False):
        view.enlarge_toggle.value = enlarged
        assert view.figure.id in _tree_ids(view)
        assert pane in root.select(pn.pane.Bokeh)
        assert box in root.select(pn.Column)
        assert [o.object for o in box.objects] == [view.figure]
        # A frame drawn while enlarged lands in the same figure.
        _shown(view, title=f"enlarged {enlarged}")
        assert view.figure.title.text == f"enlarged {enlarged}"


_SRC = Path(__file__).resolve().parents[2] / "src" / "smc" / "ui" / "image_view.py"


def _smc_imports(path: Path) -> list[str]:
    """Every module of smc that ``path`` imports, with a relative import refused (as `test_kit_imports.py`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(a.name for a in node.names if a.name.split(".")[0] == "smc")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append("." * node.level + (node.module or ""))
            elif (node.module or "").split(".")[0] == "smc":
                found.append(node.module or "")
                found.extend(f"{node.module}.{a.name}" for a in node.names)
    return found


def test_the_image_view_imports_nothing_of_the_hardware_or_the_page() -> None:
    # The only module of smc it reads is the frames' (numpy only): a `from smc.ui import x`
    # would be read as `smc.ui.x`, and `from smc import x` as `smc.x`, which is refused.
    found = _smc_imports(_SRC)
    allowed = {"smc.ui.frames", "smc.ui.frames.Box", "smc.ui.frames.display_image"}
    assert set(found) <= allowed, found
    assert "smc.ui.frames" in found


def test_the_import_walk_sees_a_relative_a_nested_and_a_from_import(
    tmp_path: Path,
) -> None:
    # The walk is the only thing that fails the test above if it is blind.
    sources = {
        "relative.py": "from . import frames\n",
        "nested.py": "def f():\n    from smc.ui.controller import PlatePage\n",
        "package.py": "from smc.ui import plate_page\n",
        "plain.py": "import smc.hardware\n",
        "clean.py": "import numpy\nfrom bokeh.models import Plot\n",
    }
    seen: dict[str, list[str]] = {}
    for name, text in sources.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        seen[name] = _smc_imports(path)
    assert seen["relative.py"] == ["."]
    assert "smc.ui.controller" in seen["nested.py"]
    assert "smc.ui.plate_page" in seen["package.py"]
    assert seen["plain.py"] == ["smc.hardware"]
    assert seen["clean.py"] == []


_NEVER_LOADED = [
    "smc.hardware",
    "smc.plugins",
    "smc.data",
    "smc.ui.controller",
    "smc.ui.guide",
    "smc.ui.plate_page",
    "pymmcore_plus",
    "pymmcore",
]


def test_importing_the_image_view_loads_none_of_the_hardware_the_registry_or_the_page(
    tmp_path: Path,
) -> None:
    # Measured while planning: `smc.ui.frames` with Panel and Bokeh loads `smc`, `smc.plates`,
    # `smc.ui` and itself. `smc.analysis` imports this module (analysis.md §2.8), and a
    # collaborator's laptop has no Micro-Manager.
    code = (
        "import json, sys\n"
        "import smc.ui.image_view\n"
        f"bad = {_NEVER_LOADED!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if any(m == b or m.startswith(b + '.') for b in bad))))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
