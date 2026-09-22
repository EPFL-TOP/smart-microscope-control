import pytest
from typer.testing import CliRunner

from smc import __version__
from smc.cli import app

runner = CliRunner()


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


def test_doctor_fails_cleanly_on_a_missing_config(tmp_path) -> None:
    result = runner.invoke(app, ["doctor", "--config", str(tmp_path / "nope.cfg")])
    assert result.exit_code == 1
    assert "not found" in result.output
