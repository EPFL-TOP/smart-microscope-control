"""The ``smc`` command line (design §9): CliRunner on the demo devices.

The runner pins the width (FM-37: rich wraps at 80 columns when it is not
writing to a terminal) and clears ``SMC_PROFILE``, so an operator's
``SMC_PROFILE=nikon-ti2`` never makes ``pytest`` open a real stand (FM-40).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smc import __version__
from smc.cli import app
from smc.hardware import Microscope

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})


@pytest.fixture(autouse=True)
def _restore_smc_logging() -> Iterator[None]:
    # The root callback replaces the smc logger's handler and level; a later
    # caplog test would otherwise lose its INFO records.
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


@pytest.fixture
def no_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command opens a microscope."""

    def refuse(cls: type[Microscope], *args: object, **kwargs: object) -> Microscope:
        pytest.fail("the command opened a microscope")

    monkeypatch.setattr(Microscope, "open", classmethod(refuse))


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "doctor" in result.output


@pytest.mark.demo
def test_doctor_passes_on_the_demo_devices(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "loads and answers" in result.output


def test_doctor_fails_cleanly_on_a_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", "--config", str(tmp_path / "nope.cfg")])
    assert result.exit_code == 1
    assert "not found" in result.output
