"""Print the next free failure-mode id, counted across every branch.

    git fetch origin
    python scripts/dev/next_fm_id.py

Parallel branches add entries to ``docs/design/failure-modes.md``, so the
highest id on ``main`` is not enough: ids collided between #84 and #10, and
#131's plan named an id that #130 had taken meanwhile. It is one plain
command because the develop worktree's sandbox refuses the pipe that did
this before. Ids are never reused, so removed lines count too.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = "docs/design/failure-modes.md"
_ID = re.compile(r"\bFM-(\d+)\b")


def next_id(texts: list[str]) -> str:
    """The id after the highest one named in ``texts``; ``FM-01`` for none."""
    ids = [int(n) for text in texts for n in _ID.findall(text)]
    return f"FM-{max(ids, default=0) + 1:02d}"


def main() -> int:
    proc = subprocess.run(
        ["git", "log", "--all", "-p", "--", CHECKLIST],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return 2
    on_disk = ROOT / CHECKLIST
    local = on_disk.read_text(encoding="utf-8") if on_disk.is_file() else ""
    print(next_id([proc.stdout, local]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
