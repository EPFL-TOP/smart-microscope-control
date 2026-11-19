"""``smc dataset draw`` on a route the demo-sample stand's stages wrote (algorithms design §6, #568).

The synthetic run folders of ``tests/unit/test_data_draw.py`` are what the seed writes;
this one is what ``plate-registration``, ``scan-plan``, ``scan``, ``cells`` and
``route`` write, through ``smc run``, so a drawing that matched the seed and not the
stages would fail here: above all the route's ``order`` is plate-wide, and E8's first
stop is not the route's first. Every expected value is read from the raw JSON of
``run.json`` and the geometry of ``smc.plugins.geometry``, never from the loader's
answers (FM-124). It needs the Micro-Manager demo adapters: it skips where they are
not installed and fails under ``SMC_REQUIRE_MM=1``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

import smc
from smc.cli import app
from smc.data import draw as draw_module
from smc.plugins.geometry import footprint

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

DEMO_SAMPLE_TOML = (
    Path(smc.__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
)
#: A1 and H12 of the centred 96-well plate: the corners the registration is told.
CORNER_REFS = '[{"well":"A1","x_um":-49500,"y_um":31500},{"well":"H12","x_um":49500,"y_um":-31500}]'
GREEN = (0, 255, 0)


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
def test_a_demo_sample_route_is_drawn_per_well_with_its_stops_numbered_from_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _dense_demo_sample(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    doc = run / "run.json"

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
    stage("route", 'labels=["single","doublet"]')

    # What the drawing hands to render_png, per well: the canvas and the stops' texts.
    real = draw_module.render_png
    drawn: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Image.Image:
        drawn.append({"canvas": args[1], **kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(draw_module, "render_png", spy)
    result = runner.invoke(app, ["dataset", "draw", str(doc)])
    assert result.exit_code == 0, result.output

    route: dict[str, Any] = json.loads(doc.read_text(encoding="utf-8"))["stages"][
        "route"
    ]["result"]
    positions: list[dict[str, Any]] = route["positions"]
    matrix = tuple(tuple(row) for row in route["matrix_um_per_px"])
    shape = tuple(route["image_shape_px"])
    by_well = {
        well: sorted(
            (p for p in positions if p["well"] == well), key=lambda p: p["order"]
        )
        for well in ("E7", "E8")
    }
    assert all(by_well.values()), "the demo route has stops in both wells"
    # The route's order is plate-wide: E8's first stop is not the route's first.
    assert by_well["E8"][0]["order"] == len(by_well["E7"]) > 0

    out = run / "draw"
    assert sorted(p.name for p in out.iterdir()) == ["E7.png", "E8.png"]
    for call in drawn:
        well = call["caption"].split(":")[0]
        stops = by_well[well]
        # Numbered from 1 to the well's count, in visiting order; no score (the
        # default selector gives none).
        assert [text for text, _, _ in call["stops"]] == [
            str(k) for k in range(1, len(stops) + 1)
        ]
        assert [(x, y) for _, x, y in call["stops"]] == [
            (p["x_um"], p["y_um"]) for p in stops
        ]
        with Image.open(out / f"{well}.png") as opened:
            image = opened.convert("RGB")
        canvas = call["canvas"]
        for stop in stops:
            corners = footprint(
                stop["x_um"], stop["y_um"], matrix=matrix, image_shape_px=shape
            )
            for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
                column = round(
                    ((a[0] + b[0]) / 2.0 - canvas.x_min_um) / canvas.um_per_px - 0.5
                )
                row = round(
                    (canvas.y_max_um - (a[1] + b[1]) / 2.0) / canvas.um_per_px - 0.5
                )
                around = {
                    image.getpixel((column + dc, row + dr))
                    for dc in (-1, 0, 1)
                    for dr in (-1, 0, 1)
                }
                assert GREEN in around, (stop["name"], column, row)
