"""scripts/dev/loc.py groups a cloc report by directory; pin that grouping."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REPORT = {
    "header": {"cloc_version": "2.10", "n_files": 4},
    "./README.md": {"blank": 3, "comment": 0, "code": 10, "language": "Markdown"},
    "./src/smc/hardware/safety.py": {
        "blank": 5,
        "comment": 7,
        "code": 100,
        "language": "Python",
    },
    "./src/smc/cli.py": {"blank": 1, "comment": 2, "code": 40, "language": "Python"},
    "./docs/adr/0001.md": {
        "blank": 2,
        "comment": 0,
        "code": 25,
        "language": "Markdown",
    },
    "SUM": {"blank": 11, "comment": 9, "code": 175, "nFiles": 4},
}


@pytest.fixture(scope="module")
def loc():
    spec = importlib.util.spec_from_file_location(
        "loc_script", ROOT / "scripts" / "dev" / "loc.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows(table: str) -> dict[str, list[str]]:
    """The table's rows keyed by their first cell, separator lines left out."""
    lines = [line.split() for line in table.splitlines() if not line.startswith("-")]
    return {cells[0]: cells[1:] for cells in lines}


def test_a_file_at_the_root_is_counted_under_dot(loc) -> None:
    assert loc.directory_of("./README.md", 1) == "."


def test_depth_cuts_the_directory_and_never_includes_the_file_name(loc) -> None:
    assert loc.directory_of("./src/smc/hardware/safety.py", 1) == "src"
    assert loc.directory_of("./src/smc/hardware/safety.py", 3) == "src/smc/hardware"
    assert loc.directory_of("./src/smc/cli.py", 3) == "src/smc"


def test_a_windows_separator_groups_like_a_slash(loc) -> None:
    assert loc.directory_of(".\\src\\smc\\cli.py", 2) == "src/smc"


def test_by_directory_sums_code_per_directory_and_language(loc) -> None:
    table = rows(loc.by_directory(REPORT, 1))
    assert table["directory"] == ["files", "Python", "Markdown", "code"]
    assert table["."] == ["1", "-", "10", "10"]
    assert table["docs"] == ["1", "-", "25", "25"]
    assert table["src"] == ["2", "140", "-", "140"]
    assert table["SUM"] == ["4", "140", "35", "175"]


def test_header_and_sum_entries_are_not_counted_as_files(loc) -> None:
    table = rows(loc.by_directory(REPORT, 1))
    assert "header" not in table
    assert table["SUM"][0] == "4"


def test_by_directory_of_an_empty_report_is_an_empty_table(loc) -> None:
    table = rows(loc.by_directory({"header": {}}, 1))
    assert table["SUM"] == ["0", "0"]


def test_without_cloc_the_script_says_how_to_install_it(
    loc, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(sys, "argv", ["loc.py"])
    monkeypatch.setattr(loc.shutil, "which", lambda _name: None)
    assert loc.main() == 1
    assert "brew install cloc" in capsys.readouterr().err
