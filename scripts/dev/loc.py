"""Count the lines of code in the repository with cloc.

Only the files git tracks are counted (``cloc --vcs=git``), so the
``.venv``, the worktrees under ``.claude/worktrees/``, ``local/`` and every
cache stay out without an exclude list to keep up to date. The other side
of that choice: a new file is counted once it is added (``git add``).

    python scripts/dev/loc.py                  # by language
    python scripts/dev/loc.py --by-dir         # code lines per top-level directory
    python scripts/dev/loc.py --by-dir 3       # three levels deep (src/smc/hardware)
    python scripts/dev/loc.py --include-lang=Python   # other options go to cloc

cloc counts Python docstrings as comments, so ``code`` is statements only;
``--docstring-as-code`` counts them as code.

cloc itself is installed apart: ``brew install cloc`` (macOS),
``winget install AlDanial.Cloc`` (Windows), ``sudo apt install cloc``
(Debian/Ubuntu).

Stdlib only, Windows included, ASCII output.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

SUBPROCESS_TIMEOUT_S = 120
# A child sharing the console can change its code page (FM-22); cloc runs
# without one and its output is written back from here.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
INSTALL_HINT = (
    "cloc is not on PATH. Install it: brew install cloc (macOS), "
    "winget install AlDanial.Cloc (Windows), sudo apt install cloc (Debian/Ubuntu)."
)


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SUBPROCESS_TIMEOUT_S,
        creationflags=_NO_WINDOW,
    )


def repo_root() -> Path:
    proc = run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())
    if proc.returncode != 0:
        raise SystemExit("not inside a git checkout")
    return Path(proc.stdout.strip()).resolve()


def directory_of(path: str, depth: int) -> str:
    """The directory a file is counted under: its first ``depth`` levels.

    cloc names files ``./src/...``; on Windows the separator may be a
    backslash. A file at the root is counted under ``.``.
    """
    parts = PurePosixPath(path.replace("\\", "/").removeprefix("./")).parts[:-1]
    return "/".join(parts[:depth]) or "."


def by_directory(report: dict[str, Any], depth: int) -> str:
    """A table of code lines, one row per directory, one column per language."""
    code: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    files: dict[str, int] = defaultdict(int)
    for path, entry in report.items():
        if path in ("header", "SUM"):
            continue
        directory = directory_of(path, depth)
        code[directory][entry["language"]] += int(entry["code"])
        files[directory] += 1

    per_language: dict[str, int] = defaultdict(int)
    for languages in code.values():
        for language, lines in languages.items():
            per_language[language] += lines
    columns = sorted(per_language, key=lambda lang: (-per_language[lang], lang))

    def cell(lines: int) -> str:
        return str(lines) if lines else "-"

    rows = [
        [
            directory,
            str(files[directory]),
            *(cell(code[directory].get(lang, 0)) for lang in columns),
            str(sum(code[directory].values())),
        ]
        for directory in sorted(code)
    ]
    total = [
        "SUM",
        str(sum(files.values())),
        *(cell(per_language[lang]) for lang in columns),
        str(sum(per_language.values())),
    ]
    header = ["directory", "files", *columns, "code"]
    widths = [
        max(len(row[i]) for row in [header, *rows, total]) for i in range(len(header))
    ]

    def line(row: list[str]) -> str:
        first = row[0].ljust(widths[0])
        rest = (
            value.rjust(width) for value, width in zip(row[1:], widths[1:], strict=True)
        )
        return "  ".join([first, *rest])

    rule = "-" * len(line(header))
    return "\n".join(
        [rule, line(header), rule, *map(line, rows), rule, line(total), rule]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count the lines of the files git tracks, with cloc.",
        epilog="Any other option is passed to cloc unchanged (see cloc --help).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--by-dir",
        nargs="?",
        const=1,
        type=int,
        metavar="DEPTH",
        help="code lines per directory, DEPTH levels deep (default 1)",
    )
    args, cloc_args = parser.parse_known_args()
    if args.by_dir is not None and args.by_dir < 1:
        parser.error("--by-dir DEPTH must be 1 or more")

    # Paths and file names may not encode on a Windows console (cp1252).
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")

    cloc = shutil.which("cloc")
    if cloc is None:
        sys.stderr.write(INSTALL_HINT + "\n")
        return 1

    command = [cloc, "--vcs=git", *cloc_args]
    if args.by_dir is not None:
        command += ["--by-file", "--json"]
    proc = run(command, cwd=repo_root())
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        return proc.returncode

    if args.by_dir is None:
        sys.stdout.write(proc.stdout)
    else:
        sys.stdout.write(by_directory(json.loads(proc.stdout), args.by_dir) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
