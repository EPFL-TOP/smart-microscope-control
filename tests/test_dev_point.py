"""scripts/dev/point.py feeds /point; pin the parsing it relies on."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def point():
    spec = importlib.util.spec_from_file_location(
        "point_script", ROOT / "scripts" / "dev" / "point.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_depends_on_reads_the_plan_line(point) -> None:
    plan = "## Plan\n\n**Depends on**: #5, #6, #7 merged into `main`\n**Risk**: pure"
    assert point.depends_on(plan) == [5, 6, 7]


def test_depends_on_none_is_empty(point) -> None:
    assert (
        point.depends_on("**Depends on**: none — uses only `smc.hardware.core`") == []
    )
    assert point.depends_on("no such line") == []


def test_dependencies_prefer_the_plan_over_the_body(point) -> None:
    issue = {
        "body": "**Depends on**: #4 — ADR-0007 accepted.\n\n## Goal",
        "comments": [
            {"body": "## Plan (design session)\n\n**Depends on**: #8", "createdAt": "x"}
        ],
    }
    assert point.dependencies(issue) == [8]


def test_blocked_issue_without_a_plan_reads_its_body(point) -> None:
    issue = {
        "body": "**Depends on**: #3, #35 — ADR-0006 and the spike.",
        "comments": [],
    }
    assert point.dependencies(issue) == [3, 35]
    assert point.dependencies({"body": None, "comments": []}) == []


def _c(body: str, at: str) -> dict:
    return {"body": body, "createdAt": at}


def test_verdict_takes_the_latest_design_review(point) -> None:
    texts = [
        _c(
            "## Review (design session, 2026-09-23)\n\n**Verdict: changes requested**",
            "2026-09-23T08:00:00Z",
        ),
        _c(
            "## Review (design session, 2026-09-24)\n\n**Verdict: ready to merge**",
            "2026-09-24T08:00:00Z",
        ),
    ]
    assert point.verdict(texts) == "ready to merge"


def test_verdict_notices_an_addressed_review(point) -> None:
    texts = [
        _c(
            "## Review (design session, 2026-09-23)\n\n**Verdict: changes requested**",
            "2026-09-23T08:00:00Z",
        ),
        _c(
            "## Review addressed (develop session, 2026-09-23)\n- 1. fixed",
            "2026-09-23T12:00:00Z",
        ),
    ]
    assert (
        point.verdict(texts) == "changes requested, then addressed: needs a re-review"
    )


def test_verdict_without_review(point) -> None:
    assert (
        point.verdict([_c("some note", "2026-09-23T08:00:00Z")])
        == "no design review yet"
    )


def test_report_field_collects_its_bullets(point) -> None:
    body = (
        "## Report (develop session, 2026-09-23)\n\n"
        "**Not verified**: real Get-PnpDevice output\n"
        "**Follow-ups**:\n- macOS SPUSBHostDataType\n- setup guide paths\n\n"
        "**Friction**: none\n"
    )
    assert point.field(body, "Follow-ups") == [
        "macOS SPUSBHostDataType",
        "setup guide paths",
    ]
    assert point.field(body, "Not verified") == ["real Get-PnpDevice output"]
    assert point.field(body, "Friction") == []


def test_checks_summary(point) -> None:
    green = [{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}]
    failing = [
        *green,
        {"name": "tests (windows)", "status": "COMPLETED", "conclusion": "FAILURE"},
    ]
    running = [{"name": "lint", "status": "IN_PROGRESS", "conclusion": ""}]
    assert point.checks(green) == "green"
    assert point.checks(failing) == "failing: tests (windows)"
    assert point.checks(running) == "running"


def test_merge_with_changes_still_requested_is_flagged(point) -> None:
    pr = {
        "number": 72,
        "title": "fix(hardware): seven concurrency gaps",
        "reviews": [],
        "comments": [
            _c("## Review (design session)\n\n**Verdict: changes requested**", "t1")
        ],
    }
    assert point.merged_lines(pr) == [
        "- #72 fix(hardware): seven concurrency gaps | changes requested"
        "  -> MERGED WITH CHANGES REQUESTED"
    ]


def test_merged_pr_lists_the_follow_ups_its_fix_round_named(point) -> None:
    pr = {
        "number": 71,
        "title": "feat(core): Microscope facade",
        "reviews": [],
        "comments": [
            _c("## Review (design session)\n\n**Verdict: changes requested**", "t1"),
            _c(
                "## Review addressed (develop session)\n\n"
                "**Follow-ups to open** (not opened here):\n"
                "- deferred: `safety.py:286`, a plain lock that halt() takes",
                "t2",
            ),
            _c(
                "## Review (design session, fix round)\n\n**Verdict: ready to merge**",
                "t3",
            ),
        ],
    }
    assert point.merged_lines(pr) == [
        "- #71 feat(core): Microscope facade | ready to merge",
        "- #71 fix round, follow-up: deferred: `safety.py:286`, a plain lock that halt() takes",
    ]


def _run(at: str, conclusion: str, title: str = "feat: x") -> dict:
    return {
        "workflowName": "CI",
        "createdAt": at,
        "conclusion": conclusion,
        "status": "completed",
        "displayTitle": title,
        "url": f"u/{at}",
    }


def test_a_flake_that_failed_main_since_the_last_point_is_listed(point) -> None:
    runs = [
        _run("2026-10-01T09:00:00Z", "failure", "fix: a"),
        _run("2026-10-01T10:00:00Z", "success", "fix: b"),
        _run("2026-09-20T10:00:00Z", "failure", "old"),
        {**_run("2026-10-01T11:00:00Z", "failure"), "workflowName": "Dependabot"},
    ]
    assert point.main_ci(runs, "2026-10-01T00:00:00Z") == [
        "- latest: success (2026-10-01T10:00, fix: b)",
        "- FAILED 2026-10-01T09:00 fix: a u/2026-10-01T09:00:00Z",
    ]


def test_main_without_ci_runs_says_so(point) -> None:
    assert point.main_ci([], "2026-10-01T00:00:00Z") == ["- no CI run on main"]
