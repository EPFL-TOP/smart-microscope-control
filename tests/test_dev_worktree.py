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
        "worktree /repo/.claude/worktrees/x\nHEAD 123\ndetached\n\n"
        "worktree /repo/.claude/worktrees/locked-bare\nHEAD 789\n"
        "branch refs/heads/feat/locked-bare\nlocked\n\n"
        "worktree /repo/.claude/worktrees/locked-reason\nHEAD 012\n"
        "branch refs/heads/feat/locked-reason\n"
        "locked claude session x (pid 1 start y)\n"
    )
    monkeypatch.setattr(wt, "run", lambda *a, **k: sample)
    assert wt.worktrees(Path("/repo")) == [
        (Path("/repo"), "main", None),
        (Path("/repo/.claude/worktrees/issue-30"), "feat/30-discover", None),
        (Path("/repo/.claude/worktrees/x"), "", None),
        (Path("/repo/.claude/worktrees/locked-bare"), "feat/locked-bare", ""),
        (
            Path("/repo/.claude/worktrees/locked-reason"),
            "feat/locked-reason",
            "claude session x (pid 1 start y)",
        ),
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


def test_lock_owner_parses_the_claude_lock_reason(wt) -> None:
    reason = "claude session issue-30-discover (pid 4242 start 2026-09-24T10:00:00)"
    assert wt.lock_owner(reason) == 4242
    assert wt.lock_owner("manual lock, no pid recorded") is None


# -- cmd_clean, on a temporary git repo -------------------------------------


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / "README.md").write_text("x", encoding="utf-8")
    _git("add", ".", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    return root


def _add_worktree(repo: Path, name: str, branch: str, *, new: bool = True) -> Path:
    path = repo / ".claude" / "worktrees" / name
    args = ["worktree", "add", "-q", str(path)]
    args += ["-b", branch] if new else [branch]
    _git(*args, cwd=repo)
    return path


def _lock(repo: Path, path: Path, reason: str) -> None:
    _git("worktree", "lock", str(path), "--reason", reason, cwd=repo)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _is_locked(wt, repo: Path, path: Path) -> bool:
    entries = {p: locked for p, _, locked in wt.worktrees(repo)}
    return entries[path] is not None


def test_clean_removes_a_merged_worktree_locked_by_a_dead_session(
    wt, repo, monkeypatch
) -> None:
    path = _add_worktree(repo, "issue-1", "feat/1")
    reason = f"claude session issue-1-demo (pid {_dead_pid()} start now)"
    _lock(repo, path, reason)
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "merged")
    removed = wt.clean_worktrees(repo, repo, dry_run=False)
    assert removed == 1
    assert not path.exists()


def test_clean_keeps_a_worktree_locked_by_a_live_process(wt, repo, monkeypatch) -> None:
    path = _add_worktree(repo, "issue-2", "feat/2")
    reason = f"claude session issue-2-demo (pid {os.getpid()} start now)"
    _lock(repo, path, reason)
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "merged")
    removed = wt.clean_worktrees(repo, repo, dry_run=False)
    assert removed == 0
    assert path.is_dir()


def test_clean_keeps_a_dirty_merged_worktree(wt, repo, monkeypatch) -> None:
    # Locked by a dead session so that, without the dirty check, the code
    # would proceed to unlock (mutating the worktree) before the native
    # `git worktree remove` guard against dirt finally refused it.
    path = _add_worktree(repo, "issue-3", "feat/3")
    reason = f"claude session issue-3-demo (pid {_dead_pid()} start now)"
    _lock(repo, path, reason)
    (path / "scratch.txt").write_text("wip", encoding="utf-8")
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "merged")
    removed = wt.clean_worktrees(repo, repo, dry_run=False)
    assert removed == 0
    assert path.is_dir()
    assert _is_locked(wt, repo, path)


def test_clean_deletes_merged_local_branches_only(
    wt, repo, monkeypatch, capsys
) -> None:
    for name in ("feat/merged", "feat/open", "feat/closed", "feat/nopr"):
        _git("branch", name, cwd=repo)
    checked_out = _add_worktree(repo, "issue-4", "feat/checked-out")
    states = {
        "feat/merged": "merged",
        "feat/open": "open",
        "feat/closed": "closed",
        "feat/nopr": "no PR",
        "feat/checked-out": "merged",  # merged, but still checked out: kept
    }
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: states[branch])
    deleted = wt.clean_branches(repo, repo, dry_run=False)
    # A branch checked out elsewhere must never even be *attempted*: git
    # itself would refuse `branch -D` on it, which would still leave
    # `deleted` and the branch list correct and hide a missing filter
    # (FM-43) behind git's own guard.
    out = capsys.readouterr().out
    assert "checked-out" not in out
    assert deleted == 1
    remaining = set(
        wt.run("git", "branch", "--format=%(refname:short)", cwd=repo).split()
    )
    assert remaining == {
        "main",
        "feat/open",
        "feat/closed",
        "feat/nopr",
        "feat/checked-out",
    }
    assert checked_out.is_dir()


def test_dry_run_changes_nothing(wt, repo, monkeypatch) -> None:
    wt_path = _add_worktree(repo, "issue-5", "feat/5")
    reason = f"claude session issue-5-demo (pid {_dead_pid()} start now)"
    _lock(repo, wt_path, reason)
    _git("branch", "feat/merged-branch", cwd=repo)
    monkeypatch.setattr(
        wt,
        "pr_state",
        lambda root, branch: "merged" if branch != "main" else "no PR",
    )
    removed = wt.clean_worktrees(repo, repo, dry_run=True)
    deleted = wt.clean_branches(repo, repo, dry_run=True)
    assert removed == 0
    assert deleted == 0
    assert wt_path.is_dir()
    assert _is_locked(wt, repo, wt_path)
    remaining = set(
        wt.run("git", "branch", "--format=%(refname:short)", cwd=repo).split()
    )
    assert "feat/merged-branch" in remaining
