"""Create the repository's labels, milestones and initial backlog on GitHub.

Idempotent: labels are upserted, milestones and issues are matched by exact
title and never duplicated. Needs an authenticated ``gh`` (``gh auth login``).

    python scripts/github/bootstrap.py --dry-run
    python scripts/github/bootstrap.py
    python scripts/github/bootstrap.py --only labels,milestones

Issue files live in ``issues/`` with a small front matter::

    ---
    title: feat(core): …
    labels: [type: feature, area: core]
    milestone: M1 — Stage on the simulator
    ---
    body in Markdown

Files are processed in name order, so number them to control creation order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gh(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"gh {' '.join(args[:3])}… failed ({proc.returncode})")
    return proc.stdout


def repo_slug(explicit: str | None) -> str:
    if explicit:
        return explicit
    data = json.loads(gh("repo", "view", "--json", "nameWithOwner"))
    return str(data["nameWithOwner"])


@dataclass
class Issue:
    path: Path
    title: str
    labels: list[str] = field(default_factory=list)
    milestone: str | None = None
    body: str = ""


def parse_issue(path: Path) -> Issue:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise SystemExit(f"{path}: missing front matter")
    _, fm, body = text.split("---", 2)
    issue = Issue(path=path, title="", body=body.strip() + "\n")
    for line in fm.strip().splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "title":
            issue.title = value.strip('"')
        elif key == "labels":
            inner = value.strip("[]")
            issue.labels = [lab.strip() for lab in inner.split(",") if lab.strip()]
        elif key == "milestone":
            issue.milestone = value.strip('"') or None
    if not issue.title:
        raise SystemExit(f"{path}: front matter has no title")
    return issue


def sync_labels(repo: str, dry: bool) -> None:
    wanted = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))
    for lab in wanted:
        print(f"label     {lab['name']}")
        if not dry:
            gh(
                "label",
                "create",
                lab["name"],
                "--repo",
                repo,
                "--force",
                "--color",
                lab["color"],
                "--description",
                lab["description"],
            )


def existing_milestones(repo: str) -> dict[str, int]:
    out = gh("api", f"repos/{repo}/milestones?state=all&per_page=100", "--paginate")
    found: dict[str, int] = {}
    # --paginate concatenates JSON arrays; split defensively.
    for chunk in out.replace("][", "]\n[").splitlines():
        if chunk.strip():
            for m in json.loads(chunk):
                found[m["title"]] = int(m["number"])
    return found


def sync_milestones(repo: str, dry: bool) -> None:
    wanted = json.loads((HERE / "milestones.json").read_text(encoding="utf-8"))
    have = existing_milestones(repo) if not dry else {}
    for ms in wanted:
        status = "exists" if ms["title"] in have else "create"
        print(f"milestone {ms['title']}  [{status}]")
        if not dry and status == "create":
            gh(
                "api",
                "-X",
                "POST",
                f"repos/{repo}/milestones",
                "-f",
                f"title={ms['title']}",
                "-f",
                f"description={ms['description']}",
            )


def existing_issue_titles(repo: str) -> set[str]:
    out = gh(
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "all",
        "--limit",
        "1000",
        "--json",
        "title",
    )
    return {str(i["title"]) for i in json.loads(out)}


def sync_issues(repo: str, dry: bool) -> None:
    issues = [parse_issue(p) for p in sorted((HERE / "issues").glob("*.md"))]
    have = existing_issue_titles(repo) if not dry else set()
    for issue in issues:
        status = "exists" if issue.title in have else "create"
        print(f"issue     {issue.title}  [{status}]")
        if dry or status == "exists":
            continue
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(issue.body)
            body_path = fh.name
        args = [
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            issue.title,
            "--body-file",
            body_path,
        ]
        for lab in issue.labels:
            args += ["--label", lab]
        if issue.milestone:
            args += ["--milestone", issue.milestone]
        url = gh(*args).strip()
        print(f"          → {url}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", help="owner/name (default: the current repository)")
    ap.add_argument(
        "--only", default="labels,milestones,issues", help="comma list of steps"
    )
    ap.add_argument("--dry-run", action="store_true", help="print what would be done")
    ns = ap.parse_args()

    steps = {s.strip() for s in ns.only.split(",")}
    repo = repo_slug(ns.repo) if not ns.dry_run or ns.repo else (ns.repo or "<repo>")
    print(f"repository: {repo}{'  (dry run)' if ns.dry_run else ''}")
    if "labels" in steps:
        sync_labels(repo, ns.dry_run)
    if "milestones" in steps:
        sync_milestones(repo, ns.dry_run)
    if "issues" in steps:
        sync_issues(repo, ns.dry_run)


if __name__ == "__main__":
    main()
