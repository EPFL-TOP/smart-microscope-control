"""Facts for /point: what is open, ready, blocked, reported and surveyed.

The supervising (design) session reads this summary instead of crawling
GitHub: one command, a short Markdown report, no judgement. The decisions
are the session's.

    python scripts/dev/point.py [--since YYYY-MM-DD]

"Since" defaults to the last comment on the open issue titled
"Point: next steps", or seven days ago when there is none. Stdlib only;
needs an authenticated ``gh``. ASCII output (#42).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRACKING_TITLE = "Point: next steps"
STATUS = "status: "
MAX_LINE = 220
_DEPENDS = re.compile(r"\*\*Depends on\*\*:?\s*(.*)", re.IGNORECASE)
_REF = re.compile(r"#(\d+)")
_ADR_STATUS = re.compile(r"\*\*Status\*\*:\s*(.+)")
_FAILING = {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}


# -- pure helpers (tested) ----------------------------------------------------


def clip(text: str, limit: int = MAX_LINE) -> str:
    """One line, at most ``limit`` characters."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def depends_on(plan: str) -> list[int]:
    """Issue or PR numbers on a plan's ``**Depends on**`` line; [] for none."""
    for line in plan.splitlines():
        m = _DEPENDS.search(line)
        if m:
            return [int(n) for n in _REF.findall(m.group(1))]
    return []


def latest(comments: list[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    """The newest comment whose body starts with ``prefix``."""
    hits = [c for c in comments if str(c.get("body", "")).lstrip().startswith(prefix)]
    return max(hits, key=lambda c: str(c.get("createdAt", "")), default=None)


def verdict(texts: list[dict[str, Any]]) -> str:
    """The design session's latest verdict on a PR, from its reviews and comments."""
    review = latest(texts, "## Review (")
    if review is None:
        return "no design review yet"
    body = str(review["body"])
    if "Verdict: ready to merge" in body:
        found = "ready to merge"
    elif "Verdict: changes requested" in body:
        found = "changes requested"
    elif "merged before review" in body:
        found = "reviewed after merge"
    else:
        found = "review without a verdict"
    addressed = latest(texts, "## Review addressed")
    if addressed and str(addressed["createdAt"]) > str(review["createdAt"]):
        return f"{found}, then addressed: needs a re-review"
    return found


def field(body: str, name: str) -> list[str]:
    """The text of a ``**Name**:`` field of a report: its line and its bullets."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"**{name}**"):
            head = line.split(":", 1)[1].strip() if ":" in line else ""
            out = [head] if head else []
            for nxt in lines[i + 1 :]:
                if nxt.strip().startswith("- "):
                    out.append(nxt.strip()[2:])
                else:
                    break
            return [x for x in out if x and x.lower() not in {"none", "none."}]
    return []


def checks(rollup: list[dict[str, Any]]) -> str:
    """``green``, ``running`` or ``failing: <names>`` for a PR's checks."""
    if not rollup:
        return "no checks"
    failing = [
        str(c.get("name") or c.get("context"))
        for c in rollup
        if str(c.get("conclusion") or c.get("state") or "") in _FAILING
    ]
    if failing:
        return "failing: " + ", ".join(failing)
    if any(str(c.get("status", "COMPLETED")) != "COMPLETED" for c in rollup):
        return "running"
    return "green"


def status_of(issue: dict[str, Any]) -> str:
    names = [str(label["name"]) for label in issue.get("labels", [])]
    found = [n[len(STATUS) :] for n in names if n.startswith(STATUS)]
    return found[0] if found else "no status"


# -- GitHub --------------------------------------------------------------------


def gh(*args: str) -> str:
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"gh {' '.join(args[:2])} failed ({proc.returncode})")
    return proc.stdout


def gh_json(*args: str) -> Any:
    return json.loads(gh(*args) or "null")


def ref_state(repo: str, number: int) -> str:
    data = gh_json("api", f"repos/{repo}/issues/{number}")
    pr = data.get("pull_request")
    if pr:
        return "merged" if pr.get("merged_at") else str(data.get("state"))
    return str(data.get("state"))


# -- the report ------------------------------------------------------------------


def since_default(tracking: dict[str, Any] | None) -> datetime:
    if tracking and tracking.get("comments"):
        last = max(str(c["createdAt"]) for c in tracking["comments"])
        return datetime.fromisoformat(last.replace("Z", "+00:00"))
    return datetime.now(timezone.utc) - timedelta(days=7)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--since", help="YYYY-MM-DD (default: the last point)")
    ns = ap.parse_args()

    repo = str(gh_json("repo", "view", "--json", "nameWithOwner")["nameWithOwner"])
    issues = gh_json(
        "issue",
        "list",
        "--state",
        "open",
        "--limit",
        "300",
        "--json",
        "number,title,labels,milestone,comments,updatedAt",
    )
    tracking = next((i for i in issues if i["title"] == TRACKING_TITLE), None)
    since = (
        datetime.fromisoformat(ns.since).replace(tzinfo=timezone.utc)
        if ns.since
        else since_default(tracking)
    )
    stamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    closed = gh_json(
        "issue",
        "list",
        "--state",
        "closed",
        "--limit",
        "100",
        "--search",
        f"closed:>={since:%Y-%m-%d}",
        "--json",
        "number,title,comments",
    )
    prs = gh_json(
        "pr",
        "list",
        "--state",
        "open",
        "--json",
        "number,title,mergeStateStatus,statusCheckRollup,reviews,comments,isDraft",
    )

    out = [
        f"# Point facts ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, since {stamp})",
        "",
    ]

    out.append("## Open pull requests")
    for pr in prs:
        texts = [
            {"body": r.get("body", ""), "createdAt": r.get("submittedAt", "")}
            for r in pr.get("reviews", [])
        ] + list(pr.get("comments", []))
        out.append(
            f"- #{pr['number']} {clip(pr['title'], 90)} | checks {checks(pr.get('statusCheckRollup') or [])}"
            f" | merge {pr['mergeStateStatus']} | {verdict(texts)}"
        )
    if not prs:
        out.append("- none")
    out.append("")

    work = [i for i in issues if i["title"] != TRACKING_TITLE]
    groups: dict[str, list[str]] = {}
    for issue in sorted(work, key=lambda i: i["number"]):
        groups.setdefault(status_of(issue), []).append(f"#{issue['number']}")
    out.append("## Open issues by status")
    for name in (
        "ready",
        "in-progress",
        "in-review",
        "blocked",
        "needs-decision",
        "needs-hardware",
        "no status",
    ):
        if name in groups:
            out.append(f"- {name}: {' '.join(groups[name])}")
    out.append("")

    out.append("## Blocked issues and their dependencies")
    for issue in (i for i in work if status_of(i) == "blocked"):
        plan = latest(issue.get("comments", []), "## Plan")
        deps = depends_on(str(plan["body"])) if plan else []
        states = {n: ref_state(repo, n) for n in deps}
        ready = bool(deps) and all(s in {"merged", "closed"} for s in states.values())
        listed = (
            ", ".join(f"#{n} {s}" for n, s in states.items()) or "no Depends-on line"
        )
        out.append(
            f"- #{issue['number']}: {listed}{'  -> ALL MERGED, can be unblocked' if ready else ''}"
        )
    out.append("")

    out.append("## Since the last point: reports, deviations, reviews, owner notes")
    seen = False
    for issue in [*work, *closed]:
        for c in issue.get("comments", []):
            if str(c.get("createdAt", "")) < stamp:
                continue
            body = str(c.get("body", "")).strip()
            head = body.splitlines()[0] if body else ""
            ref = f"#{issue['number']}"
            if head.startswith("## Report"):
                seen = True
                for name in (
                    "Follow-ups",
                    "Not verified",
                    "Deviations from plan",
                    "Beyond the plan",
                    "Friction",
                ):
                    for item in field(body, name):
                        out.append(f"- {ref} report, {name}: {clip(item)}")
            elif head.startswith("## Plan deviation"):
                seen = True
                out.append(f"- {ref} PLAN DEVIATION: {clip(body[len(head) :])}")
            elif head.startswith(
                ("## Plan", "## Review", "## Point", "## Applied", "Starting")
            ):
                continue
            elif not head.startswith("## "):
                seen = True
                out.append(f"- {ref} owner note: {clip(body, 160)}")
    if not seen:
        out.append("- nothing new")
    out.append("")

    out.append("## Survey folders (local/surveys)")
    surveys = ROOT / "local" / "surveys"
    folders = (
        sorted(p for p in surveys.iterdir() if p.is_dir()) if surveys.is_dir() else []
    )
    for folder in folders:
        curated = (ROOT / "docs" / "hardware" / f"{folder.name}.md").is_file()
        out.append(f"- {folder.name}: {'curated' if curated else 'NOT CURATED YET'}")
    if not folders:
        out.append("- none")
    out.append("")

    out.append("## ADRs not accepted")
    for path in sorted((ROOT / "docs" / "adr").glob("0*.md")):
        m = _ADR_STATUS.search(path.read_text(encoding="utf-8"))
        status = m.group(1).replace("*", "").strip() if m else "?"
        if not status.lower().startswith("accepted"):
            out.append(f"- {path.stem}: {clip(status, 120)}")
    out.append("")

    out.append("## Milestones")
    for ms in gh_json("api", f"repos/{repo}/milestones?state=open&per_page=100"):
        out.append(
            f"- {ms['title']}: {ms['closed_issues']} closed, {ms['open_issues']} open"
        )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
