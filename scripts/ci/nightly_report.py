"""Report the nightly's full matrix in one GitHub issue (ADR-0009).

When the full matrix is red, the run opens an issue ``CI nightly red`` (or
``CI nightly red (<branch>)`` for a dispatch on a branch) that names the run,
the failed jobs and the failing tests; the next red run rewrites its body and
adds one line; the next green run closes it. ``scripts/dev/point.py`` lists an
open one, so nobody has to look at the Actions tab.

    python3 scripts/ci/nightly_report.py

reads ``LINT_RESULT`` and ``TEST_RESULT`` (the ``needs.*.result`` of the two
jobs) and the JUnit files that ``actions/download-artifact`` put under
``$JUNIT_DIR`` (default ``junit``), one folder per test job.

Two sources, because each misses something: the JUnit files name the failing
tests, and a job whose process aborted (FM-179: a macOS abort inside the
MMCore) leaves no file at all, so the jobs' own conclusions, from the API,
name every failed job. A run that was cancelled or skipped says nothing about
the code and touches no issue.

Every call to GitHub goes through one injected ``gh`` callable, so the tests
run without a network.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# The issue of the nightly of ``main``; a dispatch on another branch adds " (<branch>)".
TITLE = "CI nightly red"
# The issue is found among the open issues with this label, by its exact title: the
# search index lags behind a creation by minutes, a listing by label does not.
LABEL = "area: ci"
# Every body and comment of an agent starts with a header naming its kind (CLAUDE.md):
# ``point.py`` tells the owner's notes from the workflow's by it.
HEADER = "## CI nightly (workflow, {date})"
# More than this many failing tests are cut: the issue is a pointer to the run, not the log.
MAX_TESTS = 40

_GH_TIMEOUT_S = 120
# The conclusions of a job that did not fail: anything else is listed.
_NOT_FAILED = {None, "success", "skipped", "neutral"}
# The conclusions that say a job's tests ran and failed (a cancelled job did not run them).
_FAILED = {"failure", "timed_out"}

# ``gh`` takes the arguments after the program name and the text for its standard input.
Gh = Callable[[list[str], str | None], str]


class GhError(RuntimeError):
    """A call to ``gh`` that failed, with what ``gh`` said."""


@dataclass
class JunitFailures:
    """What the JUnit files of the test jobs say.

    ``by_job`` maps a job's artifact (``ubuntu-latest-py3.13``) to its failing
    tests; ``read`` lists every artifact that held a file; ``unreadable``
    those whose file did not parse to the end.
    """

    by_job: dict[str, list[str]] = field(default_factory=dict)
    read: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


def state(lint: str, test: str) -> str:
    """``red``, ``green`` or ``unknown`` from the results of the two jobs.

    A cancelled or skipped job proves nothing, so it is neither: closing the
    issue on a run that did not run the tests would hide a red nightly.
    """
    if "failure" in (lint, test):
        return "red"
    if lint == "success" and test == "success":
        return "green"
    return "unknown"


def issue_title(ref: str) -> str:
    """The title of the nightly's issue for ``ref`` (the plain one for ``main``)."""
    return TITLE if ref == "main" else f"{TITLE} ({ref})"


def find_issue(issues: Iterable[dict[str, object]], title: str) -> int | None:
    """The number of the open issue whose title is exactly ``title``, or ``None``.

    Two issues of one title (a dispatch on ``main`` while the nightly runs) are
    harmless: the oldest is the one that is updated.
    """
    numbers = sorted(int(str(i["number"])) for i in issues if i.get("title") == title)
    return numbers[0] if numbers else None


def _read_junit(path: Path) -> tuple[list[str], bool]:
    """The failing tests of one JUnit file, and whether it parsed to the end.

    The file is fed to the parser in blocks, and what was complete before a
    syntax error or a cut still counts: a truncated file must not hide the
    failure that it did record.
    """
    parser = ET.XMLPullParser(events=("end",))
    failed: list[str] = []

    def drain() -> None:
        for _, element in parser.read_events():
            if element.tag != "testcase":
                continue
            if element.find("failure") is not None or element.find("error") is not None:
                name = element.get("name", "")
                classname = element.get("classname", "")
                failed.append(f"{classname}::{name}" if classname else name)
            element.clear()

    complete = True
    try:
        with path.open("rb") as handle:
            while block := handle.read(1 << 16):
                parser.feed(block)
                drain()
        parser.close()
    except (ET.ParseError, OSError):
        # ``feed`` queues a syntax error and ``read_events`` raises it only after it
        # has yielded the events that came before it: those were collected already.
        complete = False
    return failed, complete


def failed_tests(junit_dir: Path) -> JunitFailures:
    """The failing tests of every ``<junit_dir>/junit-*/junit.xml``.

    A missing directory (nothing was downloaded) is no failure: the jobs'
    conclusions still name what failed.
    """
    found = JunitFailures()
    if not junit_dir.is_dir():
        return found
    for file in sorted(junit_dir.glob("junit-*/junit.xml")):
        job = file.parent.name.removeprefix("junit-")
        found.read.append(job)
        failed, complete = _read_junit(file)
        if failed:
            found.by_job[job] = failed
        if not complete:
            found.unreadable.append(job)
    return found


def _artifact_of(job_name: str) -> str | None:
    """``tests (ubuntu-latest, py3.13)`` -> ``ubuntu-latest-py3.13`` (the artifact's name)."""
    match = re.fullmatch(r"tests \((?P<os>[^,]+), py(?P<py>[^)]+)\)", job_name)
    return f"{match['os']}-py{match['py']}" if match else None


def has_failed_job(jobs: list[tuple[str, str | None]]) -> bool:
    """Whether a job of the run failed or timed out, from the jobs' own conclusions.

    A cancelled job is not one: it did not run the tests. The report's own job has
    no conclusion yet while it runs, and is left out by name besides.
    """
    return any(
        conclusion in _FAILED and name != "nightly report" for name, conclusion in jobs
    )


def _failed_jobs(jobs: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
    return [
        (name, conclusion)
        for name, conclusion in jobs
        if conclusion not in _NOT_FAILED and name != "nightly report"
    ]


def body(
    today: str,
    ref: str,
    sha: str,
    run_url: str,
    jobs: list[tuple[str, str | None]] | None,
    junit: JunitFailures,
) -> str:
    """The text of the issue: the run, the failed jobs, the failing tests."""
    lines = [
        HEADER.format(date=today),
        "",
        f"**Ref**: `{ref}` at `{sha[:7]}`",
        f"**Run**: {run_url}",
        "",
        "### Failed jobs",
        "",
    ]
    if jobs is None:
        lines.append("- the jobs of this run could not be listed (the API call failed)")
    else:
        failed = _failed_jobs(jobs)
        for name, conclusion in failed:
            note = ""
            artifact = _artifact_of(name)
            if artifact is not None and artifact not in junit.read:
                note = " (no JUnit file: the process may have aborted)"
            lines.append(f"- `{name}`: {conclusion}{note}")
        if not failed:
            lines.append("- none listed")
    lines += ["", "### Failing tests", ""]

    systems: dict[str, list[str]] = {}
    for job, tests in junit.by_job.items():
        for test in tests:
            systems.setdefault(test, [])
            if job not in systems[test]:
                systems[test].append(job)
    if not systems:
        lines.append("- none read from the JUnit files")
    for test, where in list(systems.items())[:MAX_TESTS]:
        lines.append(f"- {test} ({', '.join(where)})")
    if len(systems) > MAX_TESTS:
        lines.append(f"- and {len(systems) - MAX_TESTS} more")
    if junit.unreadable:
        lines += [""]
        for job in junit.unreadable:
            lines.append(
                f"The JUnit file of `{job}` could not be read to the end: the list above may be"
                " incomplete."
            )
    lines += ["", "Fixed before the next merge of the code it names (ADR-0009).", ""]
    return "\n".join(lines)


def act(
    status: str,
    title: str,
    existing: int | None,
    text: str,
    run_url: str,
    gh: Gh,
    *,
    today: str,
) -> str:
    """Open, update or close the issue for ``status``; return what was done."""
    header = HEADER.format(date=today)
    if status == "red" and existing is None:
        gh(
            ["issue", "create", "--title", title, "--label", LABEL, "--body-file", "-"],
            text,
        )
        return f"opened the issue {title!r}"
    if status == "red":
        gh(["issue", "edit", str(existing), "--body-file", "-"], text)
        gh(
            [
                "issue",
                "comment",
                str(existing),
                "--body",
                f"{header}\n\nStill red: {run_url}",
            ],
            None,
        )
        return f"updated the issue #{existing}"
    if status == "green" and existing is not None:
        gh(
            [
                "issue",
                "close",
                str(existing),
                "--comment",
                f"{header}\n\nGreen again: {run_url}",
            ],
            None,
        )
        return f"closed the issue #{existing}"
    return (
        f"nothing to do ({status}, {'an open issue' if existing else 'no open issue'})"
    )


def run_gh(args: list[str], stdin: str | None = None) -> str:
    """Run ``gh`` and return its standard output; raise ``GhError`` when it fails."""
    # GH_REPO makes every `gh issue ...` talk to this repository, checkout or not.
    env = dict(os.environ)
    if os.environ.get("GITHUB_REPOSITORY"):
        env.setdefault("GH_REPO", os.environ["GITHUB_REPOSITORY"])
    try:
        done = subprocess.run(
            ["gh", *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=_GH_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GhError(f"gh {' '.join(args[:2])}: {error}") from error
    if done.returncode != 0:
        raise GhError(
            f"gh {' '.join(args[:2])} failed: {done.stderr.strip() or done.returncode}"
        )
    return done.stdout


def _json_documents(text: str) -> list[object]:
    """The JSON documents of ``text``, one after another (``--paginate`` prints one per page)."""
    decoder = json.JSONDecoder()
    documents: list[object] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        document, position = decoder.raw_decode(text, position)
        documents.append(document)
    return documents


def list_jobs(gh: Gh, repo: str, run_id: str) -> list[tuple[str, str | None]]:
    """``(name, conclusion)`` of every job of the run, from the API."""
    out = gh(["api", f"repos/{repo}/actions/runs/{run_id}/jobs", "--paginate"], None)
    jobs: list[tuple[str, str | None]] = []
    for page in _json_documents(out):
        if not isinstance(page, dict):
            continue
        for job in page.get("jobs", []):
            jobs.append((str(job.get("name", "")), job.get("conclusion")))
    return jobs


def main(gh: Gh | None = None, today: str | None = None) -> int:
    gh = gh or run_gh
    today = today or datetime.now(timezone.utc).date().isoformat()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    ref = os.environ.get("GITHUB_REF_NAME", "main")
    sha = os.environ.get("GITHUB_SHA", "")
    run_url = f"{server}/{repo}/actions/runs/{run_id}"
    status = state(os.environ.get("LINT_RESULT", ""), os.environ.get("TEST_RESULT", ""))
    title = issue_title(ref)
    try:
        listing_args = ["issue", "list", "--state", "open", "--label", LABEL]
        listing = gh([*listing_args, "--json", "number,title", "--limit", "200"], None)
        raw = json.loads(listing or "[]")
        existing = find_issue(raw, title)
        text = ""
        jobs: list[tuple[str, str | None]] | None = None
        if status != "green":
            try:
                jobs = list_jobs(gh, repo, run_id)
            except GhError as error:
                print(f"::warning::{error}")
        # The matrix's own result is `abandoned` when one entry never got a runner (a
        # starved macOS job, measured on the demo run of #353), whatever the other
        # entries did: four failed jobs and one abandoned made `unknown`. A job that
        # failed is a failure whatever the aggregate says.
        if status == "unknown" and jobs is not None and has_failed_job(jobs):
            status = "red"
        if status == "red":
            junit = failed_tests(Path(os.environ.get("JUNIT_DIR", "junit")))
            text = body(today, ref, sha, run_url, jobs, junit)
        done = act(status, title, existing, text, run_url, gh, today=today)
    except GhError as error:
        print(f"::error::{error}")
        return 1
    print(done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
