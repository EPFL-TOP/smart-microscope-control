"""``smc.data.load_scan`` on a scan the demo-sample stand wrote (algorithms design §3, #564).

The synthetic run folders of ``tests/unit/test_data_scan.py`` are what the seed
writes; this one is what ``plate-registration``, ``scan-plan``, ``scan`` and
``cells`` write, through ``smc run``, so a loader that matched the seed and not
the stages would fail here. Every expected value is read from the run document's
raw JSON and the files, or written by hand from the plate's geometry, and never
taken from the loader (FM-124). It needs the Micro-Manager demo adapters: it
skips where they are not installed and fails under ``SMC_REQUIRE_MM=1``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import tifffile
from typer.testing import CliRunner

import smc
from smc.cli import app
from smc.data import cells_table, load_scan

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

DEMO_SAMPLE_TOML = (
    Path(smc.__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
)
#: A1 and H12 of the centred 96-well plate: the corners the registration is told.
CORNER_REFS = '[{"well":"A1","x_um":-49500,"y_um":31500},{"well":"H12","x_um":49500,"y_um":-31500}]'
#: Where E7 and E8 are, written by hand: A1 at (-49500, 31500), 9 mm apart, row E
#: is the fifth row (4 pitches down) and column 7 the seventh (6 pitches along).
E7 = (-49500.0 + 6 * 9000.0, 31500.0 - 4 * 9000.0)
E8 = (-49500.0 + 7 * 9000.0, 31500.0 - 4 * 9000.0)


@pytest.fixture(autouse=True)
def _restore_smc_logging() -> Iterator[None]:
    # The root callback replaces the smc logger's handler and level.
    logger = logging.getLogger("smc")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    logger.propagate = saved[2]


@pytest.fixture
def demo(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


def _dense_demo_sample(tmp_path: Path) -> Path:
    """A copy of ``demo-sample.toml`` with 400 cells per well: the fields of E7 and E8 then hold some."""
    text = DEMO_SAMPLE_TOML.read_text(encoding="utf-8")
    dense = re.sub(r"^cells_per_well = 40\b", "cells_per_well = 400", text, flags=re.M)
    assert dense != text  # the line this test changes is still there
    path = tmp_path / "demo-sample.toml"
    path.write_text(dense, encoding="utf-8")
    return path


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_demo_sample_scan_loads_with_no_stand_and_matches_what_its_stages_wrote(
    tmp_path: Path,
) -> None:
    profile = _dense_demo_sample(tmp_path)
    doc = tmp_path / "run.json"

    def stage(name: str, *params: str) -> None:
        args = ["run", name, "-p", str(profile), "--run-doc", str(doc)]
        for param in params:
            args += ["--param", param]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{name}: {result.output}"

    stage("plate-registration", f"refs={CORNER_REFS}")
    stage("scan-plan", "wells=E7,E8", "rows=2", "columns=2")
    stage("scan")
    stage("cells", "detector=blobs")

    raw = json.loads(doc.read_text(encoding="utf-8"))["stages"]
    scan_raw: dict[str, Any] = raw["scan"]["result"]
    cells_raw: dict[str, Any] = raw["cells"]["result"]

    scan = load_scan(doc)

    assert scan.profile == "demo-sample"
    assert scan.wells == ("E7", "E8")
    assert len(scan_raw["frames"]) == 8
    assert len(scan.frames()) == len(scan_raw["frames"])
    assert scan.pixel_size_um == scan_raw["pixel_size_um"]
    assert scan.image_shape_px == tuple(scan_raw["image_shape_px"])
    assert scan.geometry["E7"].center_um == pytest.approx(E7, abs=0.5)
    assert scan.geometry["E8"].center_um == pytest.approx(E8, abs=0.5)

    # The pixels: each frame's own file, as tifffile reads it.
    for frame, written in zip(scan.frames(), scan_raw["frames"], strict=True):
        on_disk = tifffile.imread(tmp_path / written["file"])
        assert scan.image(frame).shape == on_disk.shape
        assert scan.image(frame).shape == tuple(scan_raw["image_shape_px"])

    # The frames' place in their well: the stage's readback less the well's
    # centre, 230 um or so away in each frame, far from the tolerance.
    for frame in scan.frames("E7"):
        assert frame.well_dx_um == pytest.approx(frame.x_um - E7[0], abs=0.5)
        assert frame.well_dy_um == pytest.approx(frame.y_um - E7[1], abs=0.5)
        assert abs(frame.well_dx_um or 0.0) > 100.0

    # The cells: the file's, as many as the result counted, per well.
    counted = {
        well: sum(by_label.values()) for well, by_label in cells_raw["counts"].items()
    }
    assert scan.cells is not None
    assert len(scan.cells) == sum(counted.values())
    assert len(scan.cells) > 0
    assert scan.cells_absent is None
    table = cells_table(scan.cells)
    assert Counter(table["well"].tolist()) == Counter(
        {well: n for well, n in counted.items() if n > 0}
    )

    # And the command reads the same folder: every frame knows its place in its well.
    shown = runner.invoke(app, ["dataset", "show", str(doc)])
    assert shown.exit_code == 0, shown.output
    lines = [line.split() for line in shown.output.splitlines()]
    assert ["profile", "demo-sample"] in lines
    assert ["place", "in", "well", "8", "of", "8", "frames"] in lines
