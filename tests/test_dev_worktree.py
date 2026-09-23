"""scripts/dev/worktree.py decides which worktrees to delete; pin its rules."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wt():
    spec = importlib.util.spec_from_file_location(
        "worktree_script", ROOT / "scripts" / "dev" / "worktree.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _venv_cfg(checkout: Path, text: str) -> None:
    (checkout / ".venv").mkdir()
    (checkout / ".venv" / "pyvenv.cfg").write_text(text, encoding="utf-8")


def test_base_interpreter_prefers_the_recorded_executable(wt, tmp_path) -> None:
    exe = tmp_path / "bin" / "python3.13"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    _venv_cfg(tmp_path, f"home = /nowhere\nexecutable = {exe}\n")
    assert wt.base_interpreter(tmp_path) == str(exe)


def test_base_interpreter_falls_back_to_home(wt, tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    name = "python.exe" if os.name == "nt" else "python3"
    (home / name).write_text("", encoding="utf-8")
    _venv_cfg(tmp_path, f"home = {home}\n")
    assert wt.base_interpreter(tmp_path) == str(home / name)


def test_no_main_venv_means_no_base_interpreter(wt, tmp_path) -> None:
    assert wt.base_interpreter(tmp_path) is None


def test_worktree_porcelain_is_parsed(wt, monkeypatch) -> None:
    sample = (
        "worktree /repo\nHEAD abc\nbranch refs/heads/main\n\n"
        "worktree /repo/.claude/worktrees/issue-30\nHEAD def\nbranch refs/heads/feat/30-discover\n\n"
        "worktree /repo/.claude/worktrees/x\nHEAD 123\ndetached\n"
    )
    monkeypatch.setattr(wt, "run", lambda *a, **k: sample)
    assert wt.worktrees(Path("/repo")) == [
        (Path("/repo"), "main"),
        (Path("/repo/.claude/worktrees/issue-30"), "feat/30-discover"),
        (Path("/repo/.claude/worktrees/x"), ""),
    ]


@pytest.mark.parametrize(
    ("gh_states", "expected"),
    [
        ("MERGED", "merged"),
        ("CLOSED,MERGED", "merged"),
        ("OPEN", "open"),
        ("CLOSED", "closed"),
        ("", "no PR"),
    ],
)
def test_merged_state_comes_from_github(wt, monkeypatch, gh_states, expected) -> None:
    monkeypatch.setattr(wt.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        wt.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout=gh_states + "\n", stderr=""
        ),
    )
    assert wt.pr_state(Path("."), "feat/30-discover") == expected


def test_a_fresh_branch_without_a_pr_is_never_merged(wt, monkeypatch) -> None:
    # A just-created worktree's branch is an ancestor of main; ancestry must
    # not be mistaken for "done", or clean() would delete a live session.
    monkeypatch.setattr(wt.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        wt.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="\n", stderr=""),
    )
    assert wt.pr_state(Path("."), "feat/31-new") == "no PR"


def test_without_gh_nothing_counts_as_merged(wt, monkeypatch) -> None:
    monkeypatch.setattr(wt.shutil, "which", lambda name: None)
    assert wt.pr_state(Path("."), "feat/30-discover") == "unknown"
