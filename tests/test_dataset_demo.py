"""``smc dataset export`` and ``check`` on a scan the demo-sample stand wrote (algorithms design §2, #565).

The synthetic run folders of ``tests/unit/test_dataset_export.py`` are what the
seed writes; this one is what ``plate-registration``, ``scan-plan``, ``scan`` and
``cells`` write, through ``smc run``, in the layout a real scan has
(``scan/<token>/<well>/<name>.tif``), so an export that matched the seed and not
the stages would fail here. Every expected value is read from the raw JSON and the
files on disk, never from the export's or the loader's own answers (FM-124). It
needs the Micro-Manager demo adapters: it skips where they are not installed and
fails under ``SMC_REQUIRE_MM=1``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import smc
from smc.cli import app
from smc.data import load_scan

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

DEMO_SAMPLE_TOML = (
    Path(smc.__file__).resolve().parents[2] / "profiles" / "demo-sample.toml"
)
#: A1 and H12 of the centred 96-well plate: the corners the registration is told.
CORNER_REFS = '[{"well":"A1","x_um":-49500,"y_um":31500},{"well":"H12","x_um":49500,"y_um":-31500}]'

#: A frame's path as a real scan writes it: the stage's folder, one token per attempt, the well.
FRAME_PATH = re.compile(r"scan/[0-9a-f]{12}/(E7|E8)/[^/]+\.tif")


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
def test_a_demo_sample_scan_exports_checks_and_loads(tmp_path: Path) -> None:
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

    source: dict[str, Any] = json.loads(doc.read_text(encoding="utf-8"))["stages"]
    frames: list[dict[str, Any]] = source["scan"]["result"]["frames"]
    cells_file: str = source["cells"]["result"]["file"]
    counted = sum(
        count
        for by_label in source["cells"]["result"]["counts"].values()
        for count in by_label.values()
    )
    assert len(frames) == 8
    assert counted > 0

    out = tmp_path / "out"
    exported = runner.invoke(app, ["dataset", "export", str(doc), "--out", str(out)])
    assert exported.exit_code == 0, exported.output

    manifest: dict[str, Any] = json.loads(
        (out / "manifest.json").read_text(encoding="utf-8")
    )
    # The frames, the cells file, run.json and README.md: no other stage's file.
    assert len(manifest["files"]) == len(frames) + 1 + 2
    assert list(manifest["files"]) == [
        *[frame["file"] for frame in frames],
        cells_file,
        "run.json",
        "README.md",
    ]
    assert sorted(
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
    ) == sorted([*manifest["files"], "manifest.json"])
    # The real layout is kept, and every frame is the source's bytes.
    for frame in frames:
        name = frame["file"]
        assert FRAME_PATH.fullmatch(name), name
        assert (out / name).read_bytes() == (run / name).read_bytes(), name
    assert (out / cells_file).read_bytes() == (run / cells_file).read_bytes()
    assert manifest["scrubbed"] == []
    assert manifest["cells"] is True
    assert manifest["wells"] == ["E7", "E8"]

    checked = runner.invoke(app, ["dataset", "check", str(out)])
    assert checked.exit_code == 0, checked.output
    assert f"{len(manifest['files'])} files match manifest.json" in checked.output

    # The dataset loads as the run folder does: the source's wells and as many
    # cells as the cells stage counted.
    loaded = load_scan(out / "run.json")
    assert loaded.wells == ("E7", "E8")
    assert loaded.cells is not None
    assert len(loaded.cells) == counted
    assert len(loaded.frames()) == len(frames)
