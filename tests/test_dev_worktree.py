"""scripts/dev/worktree.py decides which worktrees to delete; pin its rules."""

import importlib.util
import json
import os
import shutil
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


@pytest.fixture
def dead_pid():
    # On Windows a pid is eligible for reuse the moment its last handle
    # closes, which can happen as soon as the Popen object is garbage
    # collected; a live process can then claim the same number before the
    # test gets to check it. Keeping `proc` referenced by this paused
    # fixture (not just its pid) holds that handle open for the whole test.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    yield proc.pid
    assert proc.returncode == 0  # teardown: keeps `proc` alive until here


def _is_locked(wt, repo: Path, path: Path) -> bool:
    entries = {p: locked for p, _, locked in wt.worktrees(repo)}
    return entries[path] is not None


def test_clean_removes_a_merged_worktree_locked_by_a_dead_session(
    wt, repo, monkeypatch, dead_pid
) -> None:
    path = _add_worktree(repo, "issue-1", "feat/1")
    reason = f"claude session issue-1-demo (pid {dead_pid} start now)"
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


def test_clean_keeps_a_dirty_merged_worktree(wt, repo, monkeypatch, dead_pid) -> None:
    # Locked by a dead session so that, without the dirty check, the code
    # would proceed to unlock (mutating the worktree) before the native
    # `git worktree remove` guard against dirt finally refused it.
    path = _add_worktree(repo, "issue-3", "feat/3")
    reason = f"claude session issue-3-demo (pid {dead_pid} start now)"
    _lock(repo, path, reason)
    (path / "scratch.txt").write_text("wip", encoding="utf-8")
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "merged")
    removed = wt.clean_worktrees(repo, repo, dry_run=False)
    assert removed == 0
    assert path.is_dir()
    assert _is_locked(wt, repo, path)


def test_clean_notes_a_worktree_whose_directory_is_gone(wt, repo, monkeypatch) -> None:
    # git still has both worktrees registered; only the directory of the
    # first was removed out of band (a manual rm -rf, a sync tool, a
    # partially-finished earlier clean). The status check on the missing
    # one must not crash the loop before the second, healthy worktree is
    # reached (FM-31).
    gone = _add_worktree(repo, "issue-6a", "feat/6a")
    healthy = _add_worktree(repo, "issue-6b", "feat/6b")
    shutil.rmtree(gone)
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "merged")
    removed = wt.clean_worktrees(repo, repo, dry_run=False)
    assert removed == 1
    assert not healthy.exists()


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


def test_dry_run_changes_nothing(wt, repo, monkeypatch, dead_pid) -> None:
    wt_path = _add_worktree(repo, "issue-5", "feat/5")
    reason = f"claude session issue-5-demo (pid {dead_pid} start now)"
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


def _answers(table):
    """A run_note stand-in: the first table key found in the command decides."""

    def fake(args, cwd):
        line = " ".join(args)
        for key, answer in table.items():
            if key in line:
                return answer
        raise AssertionError(line)

    return fake


def test_a_review_checkout_of_a_merged_pr_counts_as_merged(wt, monkeypatch) -> None:
    monkeypatch.setattr(wt.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        wt,
        "run_note",
        _answers(
            {
                "pr list --head": (True, "\n"),
                "pr view 109": (True, "MERGED abc123\n"),
                "rev-parse pr109": (True, "def456\n"),
                "compare/abc123...def456": (True, "behind\n"),
            }
        ),
    )
    assert wt.pr_state(Path("."), "pr109") == "merged"


def test_a_review_checkout_with_a_local_commit_is_kept(wt, monkeypatch) -> None:
    # GitHub does not know a commit made only here: compare fails, keep it.
    monkeypatch.setattr(wt.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        wt,
        "run_note",
        _answers(
            {
                "pr list --head": (True, "\n"),
                "pr view 81": (True, "MERGED abc123\n"),
                "rev-parse pr-81": (True, "local9\n"),
                "compare/abc123...local9": (False, "HTTP 404"),
            }
        ),
    )
    assert wt.pr_state(Path("."), "pr-81") == "no PR"


def test_a_review_checkout_of_an_open_pr_is_open(wt, monkeypatch) -> None:
    monkeypatch.setattr(wt.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        wt,
        "run_note",
        _answers({"pr list --head": (True, "\n"), "pr view 5": (True, "OPEN abc\n")}),
    )
    assert wt.pr_state(Path("."), "pr5") == "open"
    assert wt.review_checkout_state(Path("."), "print-fix") is None


# -- a review's detached worktree and the branches a subagent leaves (#591) ----


def _git_out(*args: str, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return proc.stdout.strip()


def _add_detached(repo: Path, name: str) -> tuple[Path, str]:
    """A worktree with no branch (a review's checkout), at the repository's HEAD."""
    path = repo / ".claude" / "worktrees" / name
    _git("worktree", "add", "-q", "--detach", str(path), "HEAD", cwd=repo)
    return path, _git_out("rev-parse", "HEAD", cwd=path)


def _commit_in(path: Path, name: str) -> str:
    """A commit made in the worktree itself: GitHub has never seen it."""
    (path / name).write_text(name, encoding="utf-8")
    _git("add", name, cwd=path)
    _git("commit", "-q", "-m", name, cwd=path)
    return _git_out("rev-parse", "HEAD", cwd=path)


def _pull(sha: str, state: str = "closed", merged: bool = True, head: str = "") -> dict:
    """One entry of `commits/<sha>/pulls`: the PR's own head is what counts."""
    return {
        "state": state,
        "merged_at": "2026-11-18T10:00:00Z" if merged else None,
        "head": {"sha": head or sha},
    }


def _github(wt, monkeypatch, table: dict[str, list[dict]] | None = None, fail=""):
    """A stubbed `gh api commits/<sha>/pulls`: the table's answer, else GitHub's 422."""
    monkeypatch.setattr(wt.shutil, "which", lambda name: "/usr/bin/gh")
    real = wt.run_note
    asked: list[str] = []

    def fake(args, cwd):
        if args[0] != "gh":
            return real(args, cwd)
        line = " ".join(args)
        asked.append(line)
        if fail:
            return False, fail
        sha = line.split("/commits/")[1].split("/")[0]
        if sha in (table or {}):
            return True, json.dumps((table or {})[sha])
        return False, f"gh: No commit found for SHA: {sha} (HTTP 422)"

    monkeypatch.setattr(wt, "run_note", fake)
    return asked


@pytest.mark.parametrize(
    ("pull", "expected"),
    [
        (_pull("x"), "merged"),
        (_pull("x", merged=False), "closed"),
        (_pull("x", state="open", merged=False), "open"),
    ],
)
def test_the_state_of_a_commit_is_that_of_the_pr_it_heads(
    wt, monkeypatch, pull, expected
) -> None:
    _github(wt, monkeypatch, {"x": [pull]})
    assert wt.commit_pr_state(Path("."), "x") == expected


def test_a_commit_that_is_only_inside_a_pr_is_not_its_head(wt, monkeypatch) -> None:
    # `commits/<sha>/pulls` also lists a PR that merely contains the commit, and the PR a
    # merge commit introduced: neither says the commit made in a worktree is done.
    _github(wt, monkeypatch, {"x": [_pull("x", head="someone-elses-head")]})
    assert wt.commit_pr_state(Path("."), "x") == "no PR"
    _github(wt, monkeypatch, {"x": []})
    assert wt.commit_pr_state(Path("."), "x") == "no PR"


def test_a_pr_that_shares_its_head_with_another_is_decided_open_then_merged(
    wt, monkeypatch
) -> None:
    both = [_pull("x"), _pull("x", merged=False), _pull("x", "open", merged=False)]
    _github(wt, monkeypatch, {"x": both})
    assert wt.commit_pr_state(Path("."), "x") == "open"
    _github(wt, monkeypatch, {"x": both[:2]})
    assert wt.commit_pr_state(Path("."), "x") == "merged"


def test_a_commit_github_does_not_know_is_no_pr_and_any_other_failure_is_unknown(
    wt, monkeypatch
) -> None:
    _github(wt, monkeypatch, {})
    assert wt.commit_pr_state(Path("."), "x") == "no PR"
    _github(wt, monkeypatch, fail="gh: Server Error (HTTP 502)")
    assert wt.commit_pr_state(Path("."), "x") == "unknown"
    monkeypatch.setattr(wt, "run_note", lambda args, cwd: (True, "not json"))
    assert wt.commit_pr_state(Path("."), "x") == "unknown"
    monkeypatch.setattr(wt, "run_note", lambda args, cwd: (True, '{"a": 1}'))
    assert wt.commit_pr_state(Path("."), "x") == "unknown"
    monkeypatch.setattr(wt.shutil, "which", lambda name: None)
    assert wt.commit_pr_state(Path("."), "x") == "unknown"


@pytest.mark.parametrize("merged", [True, False], ids=["merged", "closed"])
def test_clean_removes_a_detached_clean_worktree_at_the_head_of_a_merged_pr(
    wt, repo, monkeypatch, merged
) -> None:
    # (a PR closed without a merge counts as well: its review is over)
    path, sha = _add_detached(repo, "review-1")
    asked = _github(wt, monkeypatch, {sha: [_pull(sha, merged=merged)]})
    assert wt.clean_worktrees(repo, repo, dry_run=False) == 1
    assert not path.exists()
    assert f"commits/{sha}/pulls" in asked[0]


def test_a_branch_worktree_of_a_closed_pr_is_still_kept(wt, repo, monkeypatch) -> None:
    # only a detached worktree ends with a closed PR: a branch is done when merged
    path = _add_worktree(repo, "issue-7", "feat/7")
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "closed")
    assert wt.clean_worktrees(repo, repo, dry_run=False) == 0
    assert path.is_dir()


def test_clean_keeps_a_detached_worktree_that_is_dirty_locked_by_a_live_process_or_at_a_commit_github_does_not_know(
    wt, repo, monkeypatch, capsys
) -> None:
    dirty, dirty_sha = _add_detached(repo, "review-dirty")
    (dirty / "scratch.txt").write_text("wip", encoding="utf-8")
    locked, locked_sha = _add_detached(repo, "review-locked")
    _lock(repo, locked, f"claude session review (pid {os.getpid()} start now)")
    unknown, _ = _add_detached(repo, "review-local")
    local_sha = _commit_in(unknown, "mine.txt")
    # a merged PR has the first two commits as its head, so only dirt and the lock keep
    # them; the third commit is absent from the table: GitHub answers 422 for it
    table = {
        dirty_sha: [_pull(dirty_sha)],
        locked_sha: [_pull(locked_sha)],
    }
    assert local_sha not in table
    _github(wt, monkeypatch, table)
    assert wt.clean_worktrees(repo, repo, dry_run=False) == 0
    assert dirty.is_dir()
    assert locked.is_dir()
    assert unknown.is_dir()
    assert (unknown / "mine.txt").read_text(encoding="utf-8") == "mine.txt"
    out = capsys.readouterr().out
    assert "(dirty)" in out
    assert f"locked by live pid {os.getpid()}" in out  # a live lock is named
    assert "(no PR)" in out
    assert out.count("(detached)") == 3


def test_clean_keeps_a_detached_worktree_whose_pr_is_open_or_when_github_does_not_answer(
    wt, repo, monkeypatch, capsys
) -> None:
    path, sha = _add_detached(repo, "review-open")
    _github(wt, monkeypatch, {sha: [_pull(sha, "open", merged=False)]})
    assert wt.clean_worktrees(repo, repo, dry_run=False) == 0
    assert "(open)" in capsys.readouterr().out
    _github(wt, monkeypatch, fail="gh: Server Error (HTTP 502)")
    assert wt.clean_worktrees(repo, repo, dry_run=False) == 0
    assert "(unknown)" in capsys.readouterr().out
    assert path.is_dir()


def test_clean_notes_a_detached_worktree_whose_directory_is_gone(
    wt, repo, monkeypatch
) -> None:
    # as for a branch worktree (FM-31): git still lists it, the HEAD cannot be read,
    # and the healthy one after it is still reached
    gone, _ = _add_detached(repo, "review-gone")
    healthy, sha = _add_detached(repo, "review-healthy")
    shutil.rmtree(gone)
    _github(wt, monkeypatch, {sha: [_pull(sha)]})
    assert wt.clean_worktrees(repo, repo, dry_run=False) == 1
    assert not healthy.exists()


def test_a_dry_run_names_the_detached_worktree_it_would_remove(
    wt, repo, monkeypatch, capsys
) -> None:
    path, sha = _add_detached(repo, "review-2")
    _github(wt, monkeypatch, {sha: [_pull(sha)]})
    assert wt.clean_worktrees(repo, repo, dry_run=True) == 0
    assert path.is_dir()
    out = capsys.readouterr().out
    assert "would remove" in out
    assert "(detached)" in out


def test_list_says_what_a_detached_worktree_is_waiting_for(
    wt, repo, monkeypatch, capsys
) -> None:
    _, sha = _add_detached(repo, "review-3")
    _github(wt, monkeypatch, {sha: [_pull(sha)]})
    monkeypatch.chdir(repo)
    assert wt.cmd_list(repo) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("main ")
    assert lines[1].startswith("merged ")
    assert "(detached)" in lines[1]


def _agent_repo(repo: Path) -> str:
    """``origin/main`` one commit ahead of the init commit, whose sha is returned."""
    init = _git_out("rev-parse", "HEAD", cwd=repo)
    (repo / "later.txt").write_text("later", encoding="utf-8")
    _git("add", "later.txt", cwd=repo)
    _git("commit", "-q", "-m", "later", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", "main", cwd=repo)
    return init


def _branches(wt, repo: Path) -> set[str]:
    return set(wt.run("git", "branch", "--format=%(refname:short)", cwd=repo).split())


def test_clean_deletes_a_worktree_agent_branch_with_no_worktree_and_no_commit_of_its_own(
    wt, repo, monkeypatch, capsys
) -> None:
    init = _agent_repo(repo)
    _git("branch", "worktree-agent-aaa", init, cwd=repo)
    _git("branch", "worktree-agent-at-main", "main", cwd=repo)

    def never(root, branch):
        raise AssertionError(f"GitHub was asked about {branch}")

    monkeypatch.setattr(wt, "pr_state", never)
    assert wt.clean_branches(repo, repo, dry_run=True) == 0
    assert "worktree-agent-aaa" in _branches(wt, repo)  # a dry run changes nothing
    assert "would delete" in capsys.readouterr().out
    assert wt.clean_branches(repo, repo, dry_run=False) == 2
    assert _branches(wt, repo) == {"main"}


def test_clean_keeps_a_worktree_agent_branch_that_has_a_commit_or_a_worktree_or_no_origin_main(
    wt, repo, monkeypatch, capsys
) -> None:
    init = _agent_repo(repo)
    tree = _git_out("rev-parse", "HEAD^{tree}", cwd=repo)
    own = _git_out("commit-tree", tree, "-p", init, "-m", "mine", cwd=repo)
    _git("branch", "worktree-agent-own", own, cwd=repo)
    _git("branch", "worktree-agent-used", init, cwd=repo)
    _add_worktree(repo, "agent-used", "worktree-agent-used", new=False)
    # an old commit of main is not a reason to delete a branch the process does not name
    _git("branch", "feat/old", init, cwd=repo)
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "no PR")
    assert wt.clean_branches(repo, repo, dry_run=False) == 0
    out = capsys.readouterr().out
    assert "worktree-agent-own" in out
    assert "1 commit(s) that origin/main lacks" in out
    assert "worktree-agent-used" not in out  # never even attempted (FM-43)
    assert "feat/old" in out
    # without origin/main nothing can be compared: kept, with the reason
    _git("branch", "worktree-agent-aaa", init, cwd=repo)
    _git("update-ref", "-d", "refs/remotes/origin/main", cwd=repo)
    assert wt.clean_branches(repo, repo, dry_run=False) == 0
    assert "could not compare with origin/main" in capsys.readouterr().out
    assert _branches(wt, repo) >= {
        "worktree-agent-aaa",
        "worktree-agent-own",
        "worktree-agent-used",
        "feat/old",
    }


def test_a_worktree_agent_branch_with_a_merged_pr_goes_the_way_of_any_merged_branch(
    wt, repo, monkeypatch
) -> None:
    init = _agent_repo(repo)
    tree = _git_out("rev-parse", "HEAD^{tree}", cwd=repo)
    own = _git_out("commit-tree", tree, "-p", init, "-m", "mine", cwd=repo)
    _git("branch", "worktree-agent-own", own, cwd=repo)
    monkeypatch.setattr(wt, "pr_state", lambda root, branch: "merged")
    assert wt.clean_branches(repo, repo, dry_run=False) == 1
    assert _branches(wt, repo) == {"main"}
