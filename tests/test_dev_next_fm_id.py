"""scripts/dev/next_fm_id.py: the next checklist id, across branches."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dev" / "next_fm_id.py"


@pytest.fixture(scope="module")
def fm():
    spec = importlib.util.spec_from_file_location("next_fm_id_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_highest_id_on_any_branch_wins(fm) -> None:
    main = "- **FM-12 A** text\n- **FM-85 B** text"
    other_branch = "+- **FM-86 C** text, see FM-9"
    assert fm.next_id([main, other_branch]) == "FM-87"


def test_ids_are_compared_as_numbers_not_text(fm) -> None:
    assert fm.next_id(["FM-9", "FM-10"]) == "FM-11"


def test_no_entry_starts_at_one(fm) -> None:
    assert fm.next_id([]) == "FM-01"
    assert fm.next_id(["no ids here, PFM-3x"]) == "FM-01"


def test_the_script_prints_an_id_beyond_the_checklist_on_disk() -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()
    on_disk = (ROOT / "docs" / "design" / "failure-modes.md").read_text(
        encoding="utf-8"
    )
    highest = max(int(n) for n in re.findall(r"\bFM-(\d+)\b", on_disk))
    assert out.startswith("FM-")
    assert int(out[3:]) > highest
