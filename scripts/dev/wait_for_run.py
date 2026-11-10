"""Wait for a GitHub Actions run to complete, quietly.

``gh run watch`` and ``gh pr checks --watch`` reprint the whole job tree on
every refresh (about 60 KB of context for one wait, #353), and a worktree's
sandbox refuses the ``until`` loop with ``$(...)`` that the develop skill
used instead (#342, #380). This script polls ``gh run view --json status``
and prints one line at the end.

Usage::

    python scripts/dev/wait_for_run.py <run-id> [--every 60] [--timeout 3600]

Exit code: 0 when the run's conclusion is ``success``, 1 for any other
conclusion, 2 on a timeout or a ``gh`` error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def _view(run_id: str) -> dict[str, str]:
    proc = subprocess.run(
        ["gh", "run", "view", run_id, "--json", "status,conclusion,url"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"gh exited {proc.returncode}")
    data: dict[str, str] = json.loads(proc.stdout)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "run_id", help="the run id, as `gh pr checks` or `gh run list` prints it"
    )
    ap.add_argument("--every", type=float, default=60.0, help="seconds between reads")
    ap.add_argument(
        "--timeout", type=float, default=3600.0, help="seconds before giving up"
    )
    ns = ap.parse_args()
    deadline = time.monotonic() + ns.timeout
    while True:
        try:
            data = _view(ns.run_id)
        except (RuntimeError, json.JSONDecodeError) as exc:
            print(f"wait_for_run: {exc}", file=sys.stderr)
            return 2
        if data.get("status") == "completed":
            conclusion = data.get("conclusion") or "unknown"
            print(f"run {ns.run_id}: {conclusion} {data.get('url', '')}")
            return 0 if conclusion == "success" else 1
        if time.monotonic() >= deadline:
            print(
                f"run {ns.run_id}: still {data.get('status')} after {ns.timeout:.0f} s",
                file=sys.stderr,
            )
            return 2
        time.sleep(ns.every)


if __name__ == "__main__":
    raise SystemExit(main())
