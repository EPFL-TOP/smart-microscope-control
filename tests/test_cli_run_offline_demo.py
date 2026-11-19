"""The five offline plugins on a copied demo-sample run folder, end to end (algorithms §4, #566).

A stand run makes the folder: plate-registration, scan-plan and scan on the
demo-sample stand (a synthetic plate), through the same command a user types.
The folder is then copied, as a collaborator receives it, and the demo adapters are
hidden (``find_install`` answers ``None``, as on a laptop with no Micro-Manager) and
every way to open a stand fails: from then on a command that opened one, even to
swallow its ``CoreError``, fails the test. ``cells``, ``density``, ``route``,
``report`` and ``nis-export`` then run with ``--offline`` on the copy.

The stand's folder is built once for the module: it is the slow part (the stage
travels to the plate's corners). Each test copies it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

import smc.hardware.core as core_mod
import smc.hardware.microscope as microscope_mod
from smc.cli import app
from smc.hardware import Microscope
from smc.plugins.slots import CellsResult, ReportResult, RouteResult, ScanResult
from smc.run import RunDocument, StageRecord, StageStatus

pytestmark = pytest.mark.demo

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

REPO = Path(__file__).resolve().parents[1]
DEMO_SAMPLE_TOML = REPO / "profiles" / "demo-sample.toml"
#: A1 and H12 of the centred 96-well plate: the corners the registration is told
#: (copied from test_cli_run.py, which does not export it).
CORNER_REFS = '[{"well":"A1","x_um":-49500,"y_um":31500},{"well":"H12","x_um":49500,"y_um":-31500}]'
STAND_STAGES = ("plate-registration", "scan-plan", "scan")
OFFLINE_STAGES = ("cells", "density", "route", "report", "nis-export")


def _dense_demo_sample(folder: Path) -> Path:
    """A copy of ``demo-sample.toml`` with 400 cells per well: the four fields of E7 then hold some."""
    text = DEMO_SAMPLE_TOML.read_text(encoding="utf-8")
    dense = re.sub(r"^cells_per_well = 40\b", "cells_per_well = 400", text, flags=re.M)
    assert dense != text  # the line this changes is still there
    path = folder / "demo-sample.toml"
    path.write_text(dense, encoding="utf-8")
    return path


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


@pytest.fixture(scope="module")
def stand_folder(
    tmp_path_factory: pytest.TempPathFactory, mm_available: bool
) -> Iterator[Path]:
    """The run folder a stand made: registration, plan and scan of well E7, on demo-sample.

    A module fixture runs the CLI itself, so it puts back what the commands change
    for the whole process: the ``smc`` logger's handlers and ``PATH`` (FM-46).
    """
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("stand")
    profile = _dense_demo_sample(tmp_path_factory.mktemp("profile"))
    doc = folder / "run.json"
    logger = logging.getLogger("smc")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    path = os.environ["PATH"]
    try:
        for stage, params in (
            ("plate-registration", [f"refs={CORNER_REFS}"]),
            ("scan-plan", ["wells=E7", "rows=2", "columns=2"]),
            ("scan", []),
        ):
            args = ["run", stage, "-p", str(profile), "--run-doc", str(doc)]
            for param in params:
                args += ["--param", param]
            result = _invoke(*args)
            assert result.exit_code == 0, f"{stage}: {result.output}"
        yield folder
    finally:
        logger.setLevel(saved[0])
        logger.handlers[:] = saved[1]
        logger.propagate = saved[2]
        os.environ["PATH"] = path


@pytest.fixture(autouse=True)
def _restore_smc_logging() -> Iterator[None]:
    logger = logging.getLogger("smc")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    logger.propagate = saved[2]


@pytest.fixture
def copied(stand_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The folder copied as a collaborator gets it, with the stand hidden: the copy's run document.

    The profile is looked up by name, from the repository's ``profiles/``: the
    command runs from the repository folder, as the docs say.
    """
    monkeypatch.chdir(REPO)
    copy = tmp_path / "copy"
    shutil.copytree(stand_folder, copy)
    hide_the_stand(monkeypatch)
    return copy / "run.json"


def hide_the_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Micro-Manager install is found, and every way to open a stand fails the test."""

    def refuse(*args: object, **kwargs: object) -> None:
        pytest.fail("the command opened a stand")

    monkeypatch.setattr(core_mod, "find_install", lambda: None)
    monkeypatch.setattr(core_mod, "open_core", refuse)
    monkeypatch.setattr(microscope_mod, "open_core", refuse)
    monkeypatch.setattr(Microscope, "open", classmethod(refuse))
    monkeypatch.setattr(Microscope, "from_core", classmethod(refuse))


def _offline(doc: Path, stage: str, *params: str) -> Result:
    args = ["run", stage, "--offline", "--run-doc", str(doc)]
    for param in params:
        args += ["--param", param]
    return _invoke(*args)


def _stages(doc: Path) -> dict[str, StageRecord]:
    return RunDocument.load(doc).stages


def _text_of_stages(doc: Path) -> dict[str, Any]:
    stages: dict[str, Any] = json.loads(doc.read_text(encoding="utf-8"))["stages"]
    return stages


def test_the_five_plugins_run_offline_on_a_copied_demo_sample_run_folder(
    copied: Path,
) -> None:
    for stage, params in (
        ("cells", ["detector=blobs"]),
        ("density", []),
        ("route", ['labels=["single","doublet"]']),
        ("report", []),
        ("nis-export", []),
    ):
        result = _offline(copied, stage, *params)
        assert result.exit_code == 0, f"{stage}: {result.output}"

    stages = _stages(copied)
    for stage in OFFLINE_STAGES:
        record = stages[stage]
        assert record.status is StageStatus.done, stage
        # The stand's profile, recorded; offline says it was made with no stand.
        assert record.profile == "demo-sample", stage
        assert record.offline is True, stage
        assert record.dry_run is False, stage
    # What the copy's plugins made is real: cells were found, the report is a file.
    counts = CellsResult.model_validate(stages["cells"].result).counts["E7"]
    assert counts["single"] > 0
    assert len(RouteResult.model_validate(stages["route"].result).positions) >= 1
    report = ReportResult.model_validate(stages["report"].result)
    assert (copied.parent / report.file).is_file()
    # The three stages the stand made keep no `offline` key: as written there.
    text = _text_of_stages(copied)
    for stage in STAND_STAGES:
        assert "offline" not in text[stage], stage
        assert stages[stage].offline is False
    assert set(text) == {*STAND_STAGES, *OFFLINE_STAGES}


def test_route_offline_with_another_objective_equals_the_stand(
    stand_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    original = tmp_path / "original"
    shutil.copytree(stand_folder, original)
    copy = tmp_path / "copy"
    profile = str(_dense_demo_sample(tmp_path))
    # The cells and the density a route reads are made on the stand, as a user
    # does at the stand, and the folder is copied after them: the same records on
    # both, and only the route differs.
    for stage, params in (("cells", ["detector=blobs"]), ("density", [])):
        args = ["run", stage, "-p", profile, "--run-doc", str(original / "run.json")]
        made = _invoke(*args, *[a for param in params for a in ("--param", param)])
        assert made.exit_code == 0, f"{stage}: {made.output}"
    shutil.copytree(original, copy)
    scan = ScanResult.model_validate(_stages(copy / "run.json")["scan"].result)
    other = next(
        label
        for label in ("Nikon 40X Plan Fluor ELWD", "Nikon 10X S Fluor")
        if label != scan.objective
    )
    params = ['labels=["single","doublet"]', f"objective={other}"]

    # On the stand: the original folder, the turret and the demo adapters.
    on_the_stand = _invoke(
        "run", "route", "-p", profile, "--run-doc", str(original / "run.json"),
        *[arg for param in params for arg in ("--param", param)],
    )  # fmt: skip
    assert on_the_stand.exit_code == 0, on_the_stand.output
    stand_record = _stages(original / "run.json")["route"]
    assert stand_record.offline is False

    # Offline: the copy, with no stand.
    hide_the_stand(monkeypatch)
    at_home = _offline(copy / "run.json", "route", *params)
    assert at_home.exit_code == 0, at_home.output
    home_record = _stages(copy / "run.json")["route"]
    assert home_record.offline is True

    on = RouteResult.model_validate(stand_record.result)
    home = RouteResult.model_validate(home_record.result)
    assert on.objective == other
    assert on.positions
    # Every field, not only the matrix: the field's size, the stops, what is assumed.
    assert home == on


def test_an_offline_run_leaves_pymmcore_plus_out_of_the_process(
    copied: Path,
) -> None:
    # The cells density reads, made in this process on the copy.
    made = _offline(copied, "cells", "detector=blobs")
    assert made.exit_code == 0, made.output
    code = (
        "import sys\n"
        "from smc.cli import main\n"
        "sys.argv = ['smc', 'run', 'density', '--offline', '--run-doc', sys.argv[1]]\n"
        "try:\n"
        "    main()\n"
        "except SystemExit as stop:\n"
        "    code = stop.code\n"
        "else:\n"
        "    code = 0\n"
        "named = ('pymmcore_plus', 'pymmcore', 'smc.testing')\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in named or "
        "m.startswith('smc.testing'))\n"
        "print('EXIT', code)\n"
        "print('LOADED', loaded)\n"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(copied)],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
        cwd=REPO,
        # No operator's profile or folders reach the child (FM-40).
        env={
            key: value
            for key, value in os.environ.items()
            if key not in ("SMC_PROFILE", "SMC_PROFILES")
        },
    )

    lines = done.stdout.strip().splitlines()
    assert done.returncode == 0, done.stderr
    assert "EXIT 0" in lines, done.stdout + done.stderr
    assert lines[-1] == "LOADED []", done.stdout
    assert _stages(copied)["density"].offline is True
