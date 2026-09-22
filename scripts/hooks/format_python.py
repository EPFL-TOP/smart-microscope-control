"""Claude Code PostToolUse hook: format and lint-fix a Python file just edited.

Reads the tool payload on stdin, and if the edited file is a ``.py`` inside
this repository, runs ``ruff format`` and ``ruff check --fix`` on it. Never
fails the tool call: formatting is a convenience, CI is the gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    path = str(payload.get("tool_input", {}).get("file_path", ""))
    if not path.endswith(".py"):
        return 0
    file = Path(path)
    root = Path(
        os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2])
    )
    try:
        file.resolve().relative_to(root.resolve())
    except ValueError:
        return 0
    if not file.exists():
        return 0
    ruff = root / ".venv" / "bin" / "ruff"
    if not ruff.exists():
        ruff = root / ".venv" / "Scripts" / "ruff.exe"
    exe = str(ruff) if ruff.exists() else "ruff"
    for args in (["format", "--quiet"], ["check", "--fix", "--quiet"]):
        subprocess.run([exe, *args, str(file)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
