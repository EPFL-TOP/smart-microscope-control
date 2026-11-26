"""The notebook example of ``docs/ui/views-in-a-notebook.md`` runs on a seeded scan (m5 §24.2 D9, #631).

The page's ```python blocks are joined and run with ``exec``, with the one line that
names the user's run document replaced by a seeded folder's: a documented example
that a change to ``smc.data`` or to the view broke is a failure here, not a surprise
for the collaborator who opens it. No stand and no Jupyter: ``exec`` is a cell's own
way to run code, and the last expression's display is Panel's, which the test does
not need to see.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from select_fields_seed import PIXEL_UM, SHAPE, cell_at, seed_run
from smc.ui.image_view import NO_SCALE_TEXT

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

PAGE = Path(__file__).resolve().parents[2] / "docs" / "ui" / "views-in-a-notebook.md"

_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)
_RUN_DOC_LINE = re.compile(r"^RUN_DOC = Path\(.*$", re.MULTILINE)


def _example(run_doc: Path) -> str:
    """The page's code, as one script whose ``RUN_DOC`` is ``run_doc``."""
    text = PAGE.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(text)
    assert len(blocks) == 2, (
        "the page's example is two blocks: the document, then the view"
    )
    code = "\n".join(blocks)
    replaced, count = _RUN_DOC_LINE.subn(f"RUN_DOC = Path({str(run_doc)!r})", code)
    assert count == 1, "the first block has one `RUN_DOC = Path(...)` line to replace"
    return replaced


def _run(code: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "notebook"}
    exec(compile(code, str(PAGE), "exec"), namespace)
    return namespace


def test_the_notebook_example_runs_on_a_seeded_scan(tmp_path: Path) -> None:
    seed = seed_run(
        tmp_path,
        cells=[
            cell_at("A1", 0, 0, box_px=(10, 20, 40, 60)),
            cell_at("A1", 5, 5, "doublet", box_px=(50, 50, 70, 90)),
        ],
    )
    namespace = _run(_example(seed.path))
    view = namespace["view"]
    # The scan's first frame of its first well, 60 x 80 pixels at 5 um: drawn in micrometres.
    height, width = SHAPE
    assert list(view.source.data["dw"]) == pytest.approx([width * PIXEL_UM])
    assert list(view.source.data["dh"]) == pytest.approx([height * PIXEL_UM])
    assert view.scale_bar.visible
    # Both cells are on frame 0, and their boxes are scaled from pixels to micrometres.
    data = view.boxes_source.data
    assert list(data["left"]) == pytest.approx([20 * PIXEL_UM, 50 * PIXEL_UM])
    assert list(data["right"]) == pytest.approx([60 * PIXEL_UM, 90 * PIXEL_UM])
    assert list(data["top"]) == pytest.approx([10 * PIXEL_UM, 50 * PIXEL_UM])
    assert list(data["bottom"]) == pytest.approx([40 * PIXEL_UM, 70 * PIXEL_UM])
    assert list(data["text"]) == ["single 0.90", "doublet 0.90"]
    # The scan's colours are `smc dataset draw`'s: gold for the first class, cyan for the second.
    assert list(data["colour"]) == ["#ffd700", "#00c8ff"]
    assert (
        view.figure.title.text
        == f"{seed.scan.frames[0].well} {seed.scan.frames[0].name}"
    )
    # What the cell outputs is a Panel object that builds with no server.
    root = namespace["view"].panel()
    assert root.get_root() is not None
    assert isinstance(root, pn.viewable.Viewable)


def test_the_example_shows_no_box_for_a_scan_with_no_cells_stage(
    tmp_path: Path,
) -> None:
    # The page says so: `scan.cells` is `None` when there is no usable `cells` record, and
    # `cells_of` would raise for it, so the example asks `scan.cells` first.
    seed = seed_run(tmp_path, cells=[cell_at("A1", 3.0, 2.0)])
    data = json.loads(seed.path.read_text(encoding="utf-8"))
    del data["stages"]["cells"]
    seed.path.write_text(json.dumps(data), encoding="utf-8")
    namespace = _run(_example(seed.path))
    view = namespace["view"]
    assert namespace["scan"].cells is None
    assert len(view.boxes_source.data["left"]) == 0
    # The frame is still shown, and its title has no note about boxes.
    assert list(view.source.data["dw"]) == pytest.approx([SHAPE[1] * PIXEL_UM])
    assert NO_SCALE_TEXT not in view.figure.title.text


def test_the_pages_first_block_is_the_two_lines_the_test_replaces() -> None:
    text = PAGE.read_text(encoding="utf-8")
    first = _BLOCK.findall(text)[0]
    assert first.splitlines()[0] == "from pathlib import Path"
    assert sum(1 for line in first.splitlines() if line.startswith("RUN_DOC = ")) == 1
