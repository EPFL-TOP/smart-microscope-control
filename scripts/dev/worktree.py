"""Worktree helpers for execution sessions.

Every `/develop` session works in its own git worktree (created by Claude
Code's EnterWorktree under ``.claude/worktrees/``). A worktree has no
virtual environment, and the main checkout's ``.venv`` is an *editable*
install that imports the main checkout's ``src/`` — so tests run from a
worktree with that venv would test the wrong code. ``setup`` gives the
worktree its own ``.venv`` and proves the import points inside it.

    python scripts/dev/worktree.py setup            # in the worktree
    python scripts/dev/worktree.py list
    python scripts/dev/worktree.py clean [--dry-run] # remove worktrees whose branch is merged into origin/main

Stdlib only, Windows included.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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
        raise SystemExit(f"{' '.join(args[:3])}… failed ({proc.returncode})")
    return proc.stdout


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def repo_root(start: Path) -> Path:
    return Path(run("git", "rev-parse", "--show-toplevel", cwd=start).strip())


def cmd_setup(path: Path) -> int:
    root = repo_root(path)
    py = venv_python(root)
    if not py.exists():
        print(f"creating {root / '.venv'} with {sys.executable}")
        run(sys.executable, "-m", "venv", str(root / ".venv"))
    print("installing the project (editable) and dev tools …")
    run(str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip")
    run(str(py), "-m", "pip", "install", "--quiet", "-e", ".[dev]", cwd=root)

    where = run(
        str(py),
        "-c",
        "import smc, pathlib; print(pathlib.Path(smc.__file__).resolve())",
    ).strip()
    if not where.startswith(str(root.resolve())):
        print(
            f"✗ smc imports from {where}, not from this worktree ({root})",
            file=sys.stderr,
        )
        return 1
    print(f"✓ smc imports from {where}")
    print("✓ worktree ready. Use its tools explicitly:")
    bindir = py.parent
    print(f"    {bindir / 'ruff'} check . && {bindir / 'mypy'} && {bindir / 'pytest'}")
    print("  and commit with the venv on PATH so the shared pre-commit hook finds it:")
    if os.name == "nt":
        print(f'    $env:PATH = "{bindir};" + $env:PATH; git commit …')
    else:
        print(f'    PATH="{bindir}:$PATH" git commit …')
    return 0


def worktrees(root: Path) -> list[tuple[Path, str]]:
    out = run("git", "worktree", "list", "--porcelain", cwd=root)
    found: list[tuple[Path, str]] = []
    path: Path | None = None
    branch = ""
    for line in [*out.splitlines(), ""]:
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :])
        elif line.startswith("branch "):
            branch = line[len("branch ") :].removeprefix("refs/heads/")
        elif not line and path is not None:
            found.append((path, branch))
            path, branch = None, ""
    return found


def is_merged(root: Path, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", branch, "origin/main"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return proc.returncode == 0


def cmd_list(root: Path) -> int:
    run("git", "fetch", "--quiet", "origin", "main", cwd=root, check=False)
    for path, branch in worktrees(root):
        state = "merged" if branch and is_merged(root, branch) else "open"
        print(f"{state:7s} {branch or '(detached)':40s} {path}")
    return 0


def cmd_clean(root: Path, dry_run: bool) -> int:
    run("git", "fetch", "--quiet", "origin", "main", cwd=root, check=False)
    here = Path.cwd().resolve()
    managed = (root / ".claude" / "worktrees").resolve()
    removed = 0
    for path, branch in worktrees(root):
        if not str(path.resolve()).startswith(str(managed)):
            continue
        if path.resolve() == here or path.resolve() in here.parents:
            print(f"skip    {branch:40s} {path}  (you are in it)")
            continue
        if not branch or not is_merged(root, branch):
            print(f"keep    {branch or '(detached)':40s} {path}  (not merged)")
            continue
        print(f"{'would remove' if dry_run else 'remove'}  {branch:40s} {path}")
        if not dry_run:
            run("git", "worktree", "remove", str(path), cwd=root)
            run("git", "branch", "-D", branch, cwd=root, check=False)
            removed += 1
    if not dry_run:
        run("git", "worktree", "prune", cwd=root)
        print(f"{removed} worktree(s) removed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "setup", help="create this worktree's .venv and install the project into it"
    )
    s.add_argument(
        "--path", default=".", help="the worktree (default: current directory)"
    )
    sub.add_parser("list", help="worktrees with their branch and merged state")
    c = sub.add_parser(
        "clean", help="remove managed worktrees whose branch is merged into origin/main"
    )
    c.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    if ns.cmd == "setup":
        return cmd_setup(Path(ns.path).resolve())
    root = repo_root(Path.cwd())
    if ns.cmd == "list":
        return cmd_list(root)
    return cmd_clean(root, ns.dry_run)


if __name__ == "__main__":
    sys.exit(main())
