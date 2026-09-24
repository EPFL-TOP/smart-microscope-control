"""Worktree helpers for execution sessions.

Every `/develop` session works in its own git worktree, created by Claude
Code's EnterWorktree under ``.claude/worktrees/``. A worktree has no virtual
environment, and the main checkout's ``.venv`` is an *editable* install that
imports the main checkout's ``src/`` - tests run from a worktree with that
venv would test the wrong code. ``setup`` gives the worktree its own
``.venv``, built from the same interpreter as the main checkout's, and
proves that ``smc`` imports from inside the worktree.

    python scripts/dev/worktree.py setup [--python PATH]   # in the worktree
    python scripts/dev/worktree.py list
    python scripts/dev/worktree.py clean [--dry-run]       # from the main checkout

"Merged" is asked of GitHub (``gh``), by head branch. Git ancestry cannot
answer it: a squash merge leaves no ancestry to find, and a fresh branch
with no commits yet *is* an ancestor of main without being done - removing
its worktree would destroy a session that has just started. Without ``gh``
nothing is removed.

Stdlib only, Windows included, ASCII output (a redirected stream on Windows
cannot encode check marks, #42).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
SUBPROCESS_TIMEOUT_S = 30
LOCK_PID_RE = re.compile(r"\bpid (\d+)\b")
WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WIN_STILL_ACTIVE = 259


def run(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"{' '.join(args[:3])}... failed ({proc.returncode})")
    return proc.stdout


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def repo_root(start: Path) -> Path:
    return Path(run("git", "rev-parse", "--show-toplevel", cwd=start).strip()).resolve()


def main_checkout(start: Path) -> Path:
    """The checkout that owns the shared ``.git`` directory."""
    common = Path(run("git", "rev-parse", "--git-common-dir", cwd=start).strip())
    if not common.is_absolute():
        common = start / common
    return common.resolve().parent


def base_interpreter(checkout: Path) -> str | None:
    """The interpreter the main checkout's ``.venv`` was built from, if any.

    Building every worktree's venv from the same interpreter keeps dependency
    versions (numpy, mypy's view of the stubs) identical to the design
    session's, instead of whatever ``python`` an agent shell happens to find.
    """
    cfg = checkout / ".venv" / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    values: dict[str, str] = {}
    for line in cfg.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip().lower()] = value.strip()
    executable = values.get("executable", "")
    if executable and Path(executable).is_file():
        return executable
    home = values.get("home", "")
    names = ("python.exe",) if os.name == "nt" else ("python3", "python")
    for name in names:
        candidate = Path(home) / name
        if home and candidate.is_file():
            return str(candidate)
    return None


def python_version(executable: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run(
            [executable, "-c", "import sys; print(*sys.version_info[:2])"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    major, minor = proc.stdout.split()
    return int(major), int(minor)


def cmd_setup(path: Path, python: str | None) -> int:
    root = repo_root(path)
    py = venv_python(root)
    if not py.exists():
        base = python or base_interpreter(main_checkout(root)) or sys.executable
        version = python_version(base)
        if version is None or version < MIN_PYTHON:
            found = (
                "not runnable"
                if version is None
                else f"Python {version[0]}.{version[1]}"
            )
            print(
                f"error: {base} is {found}; this project needs Python >= 3.10. "
                f"Pass --python PATH.",
                file=sys.stderr,
            )
            return 1
        print(
            f"creating {root / '.venv'} with {base} (Python {version[0]}.{version[1]})"
        )
        run(base, "-m", "venv", str(root / ".venv"))
    print("installing the project (editable) and dev tools ...")
    run(str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip")
    run(str(py), "-m", "pip", "install", "--quiet", "-e", ".[dev]", cwd=root)

    probe = "import smc, pathlib; print(pathlib.Path(smc.__file__).resolve())"
    where = Path(run(str(py), "-c", probe).strip())
    if not where.is_relative_to(root):
        print(
            f"error: smc imports from {where}, not from this worktree ({root})",
            file=sys.stderr,
        )
        return 1
    print(f"ok: smc imports from {where}")
    bindir = py.parent
    print("ok: worktree ready. Use its tools explicitly:")
    print(f"    {bindir / 'ruff'} check . && {bindir / 'mypy'} && {bindir / 'pytest'}")
    print("  and commit with the venv on PATH so the shared pre-commit hook finds it:")
    if os.name == "nt":
        print(f'    $env:PATH = "{bindir};" + $env:PATH; git commit ...')
    else:
        print(f'    PATH="{bindir}:$PATH" git commit ...')
    return 0


def worktrees(root: Path) -> list[tuple[Path, str, str | None]]:
    """``(path, branch, lock reason)`` for every worktree.

    ``branch`` is "" when detached. The lock reason is ``None`` when the
    worktree is not locked, and "" when it is locked with no reason given.
    """
    out = run("git", "worktree", "list", "--porcelain", cwd=root)
    found: list[tuple[Path, str, str | None]] = []
    path: Path | None = None
    branch = ""
    locked: str | None = None
    for line in [*out.splitlines(), ""]:
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :])
        elif line.startswith("branch "):
            branch = line[len("branch ") :].removeprefix("refs/heads/")
        elif line == "locked" or line.startswith("locked "):
            locked = line[len("locked") :].strip()
        elif not line and path is not None:
            found.append((path, branch, locked))
            path, branch, locked = None, "", None
    return found


def lock_owner(reason: str) -> int | None:
    """The pid in a lock reason such as ``claude session x (pid 123 start …)``.

    ``None`` when the reason has no ``pid N`` - the lock is then kept
    rather than guessed at (design decision: an unparseable reason counts
    as alive).
    """
    match = LOCK_PID_RE.search(reason)
    return int(match.group(1)) if match else None


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process. Unknown counts as alive (the safe side)."""
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError (owned by someone else) and anything else we
        # cannot interpret: assume alive rather than delete live work.
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return True
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return bool(exit_code.value == WIN_STILL_ACTIVE)
    finally:
        kernel32.CloseHandle(handle)


def pr_state(root: Path, branch: str) -> str:
    """The state of the pull requests whose head is ``branch``, asked of GitHub.

    One of ``merged``, ``open``, ``closed``, ``no PR``, ``detached`` or
    ``unknown`` (no ``gh``, or ``gh`` failed). Only ``merged`` allows removal.
    """
    if not branch:
        return "detached"
    if shutil.which("gh") is None:
        return "unknown"
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "all",
            "--json",
            "state",
            "--jq",
            '[.[].state] | join(",")',
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return "unknown"
    states = {s for s in proc.stdout.strip().split(",") if s}
    for state in ("MERGED", "OPEN", "CLOSED"):
        if state in states:
            return state.lower()
    return "no PR"


def cmd_list(root: Path) -> int:
    main = main_checkout(root)
    for path, branch, _ in worktrees(root):
        state = "main" if path.resolve() == main else pr_state(root, branch)
        print(f"{state:8s} {branch or '(detached)':40s} {path}")
    return 0


def git_note(args: list[str], cwd: Path) -> tuple[bool, str]:
    """Run a git command whose failure is a note, not a crash (FM-31).

    Returns ``(ok, detail)``; ``detail`` is empty on success, else the last
    stderr line or a timeout message.
    """
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {SUBPROCESS_TIMEOUT_S}s"
    if proc.returncode != 0:
        reason = (proc.stderr.strip().splitlines() or ["unknown reason"])[-1]
        return False, reason
    return True, ""


def clean_worktrees(root: Path, main: Path, dry_run: bool) -> int:
    """Remove managed worktrees that are merged, clean and not locked by a live pid."""
    managed = (main / ".claude" / "worktrees").resolve()
    main_resolved = main.resolve()
    here = Path.cwd().resolve()
    removed = 0
    for path, branch, locked in worktrees(root):
        resolved = path.resolve()
        if resolved == main_resolved or not resolved.is_relative_to(managed):
            continue
        if here == resolved or here.is_relative_to(resolved):
            print(f"skip     {branch:40s} {path}  (you are in it)")
            continue
        state = pr_state(root, branch)
        if state != "merged":
            print(f"keep     {branch or '(detached)':40s} {path}  ({state})")
            continue
        if run("git", "status", "--porcelain", cwd=resolved).strip():
            print(f"keep     {branch:40s} {path}  (dirty)")
            continue
        if locked is not None:
            owner = lock_owner(locked)
            if owner is None:
                print(f"keep     {branch:40s} {path}  (locked, no pid in: {locked!r})")
                continue
            if pid_alive(owner):
                print(f"keep     {branch:40s} {path}  (locked by live pid {owner})")
                continue
        print(f"{'would remove' if dry_run else 'remove':8s} {branch:40s} {path}")
        if dry_run:
            continue
        if locked is not None:
            ok, detail = git_note(["git", "worktree", "unlock", str(path)], main)
            if not ok:
                print(f"         kept: could not unlock ({detail})")
                continue
        ok, detail = git_note(["git", "worktree", "remove", str(path)], main)
        if not ok:
            print(f"         kept: git refused ({detail})")
            continue
        removed += 1
    if not dry_run:
        run("git", "worktree", "prune", cwd=main)
    return removed


def clean_branches(root: Path, main: Path, dry_run: bool) -> int:
    """Delete local branches, other than ``main``, whose PR is merged and unused."""
    checked_out = {branch for _, branch, _ in worktrees(root) if branch}
    deleted = 0
    for branch in run(
        "git", "branch", "--format=%(refname:short)", cwd=main
    ).splitlines():
        branch = branch.strip()
        if not branch or branch == "main" or branch in checked_out:
            continue
        state = pr_state(root, branch)
        if state != "merged":
            print(f"keep     {branch:40s} (branch, {state})")
            continue
        print(f"{'would delete' if dry_run else 'deleted':8s} {branch:40s} (branch)")
        if dry_run:
            continue
        ok, detail = git_note(["git", "branch", "-D", branch], main)
        if not ok:
            print(f"         kept: git refused ({detail})")
            continue
        deleted += 1
    return deleted


def cmd_clean(root: Path, dry_run: bool) -> int:
    main = main_checkout(root)
    removed = clean_worktrees(root, main, dry_run)
    deleted = clean_branches(root, main, dry_run)
    note = ""
    if not dry_run:
        ok, detail = git_note(
            [
                "git",
                "-c",
                "url.https://github.com/.insteadOf=git@github.com:",
                "fetch",
                "--prune",
                "origin",
            ],
            main,
        )
        if not ok:
            note = f" (fetch --prune failed: {detail})"
        print(f"{removed} worktree(s) removed, {deleted} branch(es) deleted{note}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "setup", help="create this worktree's .venv and install the project"
    )
    s.add_argument(
        "--path", default=".", help="the worktree (default: current directory)"
    )
    s.add_argument(
        "--python",
        default=None,
        help="interpreter for the new .venv (default: the main checkout's .venv base)",
    )
    sub.add_parser("list", help="worktrees with the state of their pull request")
    c = sub.add_parser(
        "clean", help="remove managed worktrees whose pull request is merged"
    )
    c.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    if ns.cmd == "setup":
        return cmd_setup(Path(ns.path).resolve(), ns.python)
    root = repo_root(Path.cwd())
    if ns.cmd == "list":
        return cmd_list(root)
    return cmd_clean(root, ns.dry_run)


if __name__ == "__main__":
    sys.exit(main())
