"""scripts/dev/point.py feeds /point; pin the parsing it relies on."""

import importlib.util
import io
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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


def _run(at: str, conclusion: str, title: str = "feat: x", event: str = "push") -> dict:
    # `gh run list` leaves the conclusion empty while a run is in progress.
    return {
        "workflowName": "CI",
        "createdAt": at,
        "conclusion": conclusion,
        "status": "completed" if conclusion else "in_progress",
        "displayTitle": title,
        "url": f"u/{at}",
        "event": event,
    }


def test_a_flake_that_failed_main_since_the_last_point_is_listed(point) -> None:
    runs = [
        _run("2026-10-01T09:00:00Z", "failure", "fix: a"),
        _run("2026-10-01T10:00:00Z", "success", "fix: b"),
        _run("2026-09-20T10:00:00Z", "failure", "old"),
        {**_run("2026-10-01T11:00:00Z", "failure"), "workflowName": "Dependabot"},
    ]
    assert point.main_ci(runs, "2026-10-01T00:00:00Z") == [
        "- latest: success, push (2026-10-01T10:00, fix: b)",
        "- FAILED 2026-10-01T09:00 push fix: a u/2026-10-01T09:00:00Z",
    ]


def test_a_run_cancelled_by_a_newer_push_is_listed_as_no_verdict(point) -> None:
    # Six superseded push runs read as six red mains on 2026-11-11.
    runs = [
        _run("2026-10-01T09:00:00Z", "cancelled", "feat: a"),
        _run("2026-10-01T10:00:00Z", "success", "feat: b"),
    ]
    assert point.main_ci(runs, "2026-10-01T00:00:00Z") == [
        "- latest: success, push (2026-10-01T10:00, feat: b)",
        "- cancelled, no verdict 2026-10-01T09:00 push feat: a u/2026-10-01T09:00:00Z",
    ]


def test_the_ci_lines_of_main_say_the_event(point) -> None:
    runs = [
        _run("2026-10-01T09:00:00Z", "failure", "feat: a", event="schedule"),
        _run("2026-10-01T10:00:00Z", "failure", "feat: b", event="workflow_dispatch"),
        _run("2026-10-01T11:00:00Z", "success", "feat: c", event="schedule"),
    ]
    assert point.main_ci(runs, "2026-10-01T00:00:00Z") == [
        "- latest: success, nightly (2026-10-01T11:00, feat: c)",
        "- FAILED 2026-10-01T09:00 nightly feat: a u/2026-10-01T09:00:00Z",
        "- FAILED 2026-10-01T10:00 manual feat: b u/2026-10-01T10:00:00Z",
    ]
    # an older `gh` or a run without the field is shown, not guessed
    bare = {
        k: v for k, v in _run("2026-10-01T09:00:00Z", "success").items() if k != "event"
    }
    assert point.main_ci([bare], "2026-10-01T00:00:00Z")[0].startswith(
        "- latest: success, ?"
    )


def test_main_without_ci_runs_says_so(point) -> None:
    assert point.main_ci([], "2026-10-01T00:00:00Z") == ["- no CI run on main"]


def test_an_open_pr_shows_the_issues_github_will_close(point) -> None:
    pr = {"closingIssuesReferences": [{"number": 16}, {"number": 13}]}
    assert point.closes(pr) == "closes #13 #16"
    assert point.closes({"closingIssuesReferences": []}) == "closes no issue"
    assert point.closes({}) == "closes no issue"


# --- the CI that stops the merges (ADR-0009) ---------------------------------


def test_a_red_main_is_the_latest_failed_push_run_not_a_nightly(point) -> None:
    red_push = _run("2026-10-01T09:00:00Z", "failure", "feat: a")
    runs = [
        red_push,
        _run("2026-10-01T10:00:00Z", "success", "nightly", event="schedule"),
    ]
    # a green nightly does not clear a red push: only a later push does
    assert point.red_main(runs) == red_push
    # and a red nightly does not make main red: it has its own issue
    runs = [
        _run("2026-10-01T09:00:00Z", "success", "feat: a"),
        _run("2026-10-01T10:00:00Z", "failure", "nightly", event="schedule"),
        _run("2026-10-01T11:00:00Z", "failure", "manual", event="workflow_dispatch"),
    ]
    assert point.red_main(runs) is None


def test_a_green_push_after_a_red_one_clears_main(point) -> None:
    runs = [
        _run("2026-10-01T09:00:00Z", "failure", "feat: a"),
        _run("2026-10-01T10:00:00Z", "success", "fix: a"),
    ]
    assert point.red_main(runs) is None


def test_a_push_run_still_in_progress_does_not_decide(point) -> None:
    red = _run("2026-10-01T09:00:00Z", "failure", "feat: a")
    assert point.red_main([red, _run("2026-10-01T10:00:00Z", "", "fix: a")]) == red
    green = _run("2026-10-01T09:00:00Z", "success", "feat: a")
    assert point.red_main([green, _run("2026-10-01T10:00:00Z", "", "feat: b")]) is None


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped", "neutral"])
def test_a_cancelled_push_run_does_not_make_main_red(point, conclusion) -> None:
    # A push to main cancels the older run of main (concurrency): a cancellation is
    # not a verdict on the code, and it is the usual state of the older of two merges.
    # A skipped or neutral run ran no test either: it neither makes main red nor clears it.
    green = _run("2026-10-01T09:00:00Z", "success", "feat: a")
    no_verdict = _run("2026-10-01T10:00:00Z", conclusion, "feat: b")
    assert point.red_main([green, no_verdict]) is None
    red = _run("2026-10-01T08:00:00Z", "failure", "feat: z")
    assert point.red_main([red, no_verdict]) == red


def test_only_the_ci_workflow_decides_whether_main_is_red(point) -> None:
    other = {
        **_run("2026-10-01T11:00:00Z", "failure"),
        "workflowName": "closing references",
    }
    assert point.red_main([other]) is None
    assert point.red_main([]) is None


def _nightly(number: int, title: str = "CI nightly red") -> dict:
    return {
        "number": number,
        "title": title,
        "labels": [{"name": "area: ci"}],
        "comments": [],
        "body": "## CI nightly (workflow, 2026-11-06)",
    }


def test_an_open_nightly_issue_is_listed_among_the_ci_that_stops_the_merges(
    point,
) -> None:
    issues = [
        _nightly(77),
        _nightly(80, "CI nightly red (chore/x)"),
        _issue(5, "feat: y", []),
    ]
    lines = point.merge_stoppers([_run("2026-10-01T09:00:00Z", "success")], issues)
    assert len(lines) == 2
    assert lines[0].startswith("- #77 ")
    assert "CI nightly red" in lines[0]
    assert lines[1].startswith("- #80 ")
    assert "(chore/x)" in lines[1]


def test_a_red_main_is_listed_with_its_run_and_nothing_else_is_none(point) -> None:
    red = _run("2026-10-01T09:00:00Z", "failure", "feat: a")
    lines = point.merge_stoppers([red], [])
    assert len(lines) == 1
    assert "main is RED" in lines[0]
    assert "feat: a" in lines[0]
    assert "u/2026-10-01T09:00:00Z" in lines[0]
    assert point.merge_stoppers([_run("2026-10-01T09:00:00Z", "success")], []) == [
        "- none"
    ]
    assert point.merge_stoppers([], []) == ["- none"]


def test_both_stoppers_are_listed_red_main_first(point) -> None:
    red = _run("2026-10-01T09:00:00Z", "failure", "feat: a")
    lines = point.merge_stoppers([red], [_nightly(77)])
    assert "main is RED" in lines[0]
    assert lines[1].startswith("- #77 ")


@pytest.mark.parametrize(
    ("event", "label"), [("schedule", "nightly"), ("workflow_dispatch", "manual")]
)
def test_a_failed_nightly_with_no_open_issue_stops_the_merges(
    point, event, label
) -> None:
    # The report job crashed or ended silent (#358): no issue, but the run itself is red.
    failed = _run("2026-10-01T09:00:00Z", "failure", "feat: a", event=event)
    lines = point.merge_stoppers([failed], [])
    assert len(lines) == 1
    assert "FAILED" in lines[0]
    assert label in lines[0]
    assert "u/2026-10-01T09:00:00Z" in lines[0]
    assert "no open" in lines[0]
    assert "CI nightly red" in lines[0]
    assert point.unreported_nightly([failed], []) == failed


def test_a_red_main_comes_before_a_failed_nightly_and_before_an_issue(point) -> None:
    red = _run("2026-10-01T09:00:00Z", "failure", "feat: a")
    failed = _run("2026-10-01T10:00:00Z", "failure", "feat: b", event="schedule")
    other = _nightly(80, "CI nightly red (chore/x)")
    lines = point.merge_stoppers([failed, red], [other])
    assert len(lines) == 3
    assert "main is RED" in lines[0]
    assert "the nightly run FAILED" in lines[1]
    assert lines[2].startswith("- #80 ")


def test_a_failed_nightly_with_its_open_issue_is_listed_once(point) -> None:
    failed = _run("2026-10-01T09:00:00Z", "failure", "feat: a", event="schedule")
    lines = point.merge_stoppers([failed], [_nightly(77)])
    assert len(lines) == 1
    assert lines[0].startswith("- #77 ")
    assert point.unreported_nightly([failed], [_nightly(77)]) is None


def test_a_nightly_issue_of_another_branch_does_not_cover_a_failed_nightly_of_main(
    point,
) -> None:
    failed = _run("2026-10-01T09:00:00Z", "failure", "feat: a", event="schedule")
    other = _nightly(80, "CI nightly red (chore/x)")
    lines = point.merge_stoppers([failed], [other])
    assert len(lines) == 2
    assert "FAILED" in lines[0]
    assert lines[1].startswith("- #80 ")


def test_a_later_green_nightly_clears_a_failed_one_and_a_cancelled_or_running_one_not(
    point,
) -> None:
    failed = _run("2026-10-01T09:00:00Z", "failure", "feat: a", event="schedule")
    green = _run("2026-10-02T09:00:00Z", "success", "feat: b", event="schedule")
    assert point.unreported_nightly([failed, green], []) is None
    # a green dispatch clears it as well: it ran the same matrix
    manual = _run(
        "2026-10-02T10:00:00Z", "success", "feat: b", event="workflow_dispatch"
    )
    assert point.unreported_nightly([failed, manual], []) is None
    # a cancelled, skipped or running nightly is no verdict: the failed one stands
    for conclusion in ("cancelled", "skipped", "neutral", ""):
        later = _run("2026-10-02T09:00:00Z", conclusion, "feat: b", event="schedule")
        assert point.unreported_nightly([failed, later], []) == failed
    # the latest verdict decides, whatever the order of the list
    assert point.unreported_nightly([green, failed], []) is None
    red_again = _run("2026-10-03T09:00:00Z", "failure", "feat: c", event="schedule")
    assert point.unreported_nightly([red_again, green, failed], []) == red_again


@pytest.mark.parametrize(
    "conclusion", ["failure", "timed_out", "action_required", "startup_failure"]
)
def test_every_failing_conclusion_of_a_nightly_counts(point, conclusion) -> None:
    run = _run("2026-10-01T09:00:00Z", conclusion, "feat: a", event="schedule")
    assert point.unreported_nightly([run], []) == run


def test_only_a_failed_run_of_the_ci_workflow_that_is_a_nightly_counts(point) -> None:
    push = _run("2026-10-01T09:00:00Z", "failure", "feat: a", event="push")
    other = {
        **_run("2026-10-01T10:00:00Z", "failure", "feat: b", event="schedule"),
        "workflowName": "closing references",
    }
    bare = {
        k: v
        for k, v in _run("2026-10-01T11:00:00Z", "failure", event="schedule").items()
        if k != "event"
    }
    assert point.unreported_nightly([push, other, bare], []) is None
    assert point.unreported_nightly([], []) is None


def test_a_title_that_only_resembles_the_nightly_is_not_a_nightly_issue(point) -> None:
    # A work issue about the nightly may start with its words: only the exact title, or the
    # title with a branch, is what nightly_report.py writes.
    issues = [
        _nightly(1),
        _issue(2, "Fix the CI nightly red flag in point.py", []),
        _issue(3, "CI nightly red: make the report name the failing job", []),
        _issue(4, "CI nightly redesign", []),
        _nightly(5, "CI nightly red (chore/x)"),
    ]
    assert [i["number"] for i in point.nightly_issues(issues)] == [1, 5]


def test_the_nightly_issue_title_is_the_one_the_report_writes(point) -> None:
    spec = importlib.util.spec_from_file_location(
        "nightly_report_for_point", ROOT / "scripts" / "ci" / "nightly_report.py"
    )
    assert spec is not None
    assert spec.loader is not None
    report = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = report
    spec.loader.exec_module(report)
    assert point.NIGHTLY_TITLE == report.TITLE
    assert point.nightly_issues([_nightly(1, report.issue_title("chore/x"))])


def _issue(number: int, title: str, labels: list[str]) -> dict:
    return {
        "number": number,
        "title": title,
        "labels": [{"name": n} for n in labels],
        "comments": [],
        "body": "",
    }


def test_the_nightly_issue_is_not_counted_among_the_issues_by_status(point) -> None:
    issues = [
        _issue(3, "feat: a", ["status: ready"]),
        _issue(4, "feat: b", ["status: ready"]),
        _nightly(77),
        _issue(9, point.TRACKING_TITLE, []),
    ]
    assert point.status_groups(point.work_issues(issues)) == {"ready": ["#3", "#4"]}
    # the nightly's issue has no status label, so it would have been "no status" otherwise
    assert "no status" not in point.status_groups(point.work_issues(issues))


# --- the report, end to end on canned data ------------------------------------


def _canned(
    runs: list[dict],
    issues: list[dict],
    seen: list[tuple[str, ...]],
    prs=(),
    merged=(),
    closed=(),
    refs: dict | None = None,
):
    """A stand-in for ``gh_json``: the listings, and the state of a referenced issue.

    ``refs`` maps an issue number to what ``gh api repos/<repo>/issues/<n>`` answers
    (open by default); any other ``api`` call answers ``[]`` (the milestones).
    """

    def fake(*args: str):
        seen.append(args)
        if args[:2] == ("repo", "view"):
            return {"nameWithOwner": "EPFL-TOP/smart-microscope-control"}
        if args[:2] == ("issue", "list"):
            return issues if "open" in args else list(closed)
        if args[:2] == ("pr", "list"):
            if "open" in args:
                return list(prs)
            return list(merged) if "merged" in args else []
        if args[0] == "run":
            return runs
        if args[0] == "api":
            found = re.fullmatch(r"repos/[^/]+/[^/]+/issues/(\d+)", args[1])
            if found:
                return (refs or {}).get(int(found.group(1)), {"state": "open"})
            return []
        raise AssertionError(f"unexpected gh call: {args}")

    return fake


def test_the_report_opens_with_the_ci_that_stops_the_merges(
    point, monkeypatch, capsys
) -> None:
    seen: list[tuple[str, ...]] = []
    red = _run("2026-11-05T09:00:00Z", "failure", "feat: a")
    issues = [_nightly(77), _issue(3, "feat: b", ["status: ready"])]
    monkeypatch.setattr(point, "gh_json", _canned([red], issues, seen))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])

    assert point.main() == 0

    out = capsys.readouterr().out
    # the sections this change writes stay ASCII (#42); the ADR section quotes the docs as they are
    assert out.split("## Merged since")[0].isascii()
    headings = [line for line in out.splitlines() if line.startswith("## ")]
    assert headings[0] == "## CI that stops the merges"
    assert headings[1] == "## Open pull requests"
    first = out.split("## Open pull requests")[0]
    assert "main is RED" in first
    assert "#77" in first
    # the nightly's issue is neither a ready, an in-progress nor a "no status" issue
    status = out.split("## Open issues by status")[1].split("## ")[0]
    assert "- ready: #3" in status
    assert "#77" not in status
    # the run list asks for the event
    run_call = next(a for a in seen if a[0] == "run")
    assert "event" in run_call[run_call.index("--json") + 1].split(",")


def test_the_report_lists_a_failed_nightly_with_no_issue_among_the_stoppers(
    point, monkeypatch, capsys
) -> None:
    # A nightly that is older than the 50 newest runs of main would be missed (measured on
    # 2026-11-13: 50 runs reach back four days): the listing asks for more.
    seen: list[tuple[str, ...]] = []
    failed = _run("2026-11-05T09:00:00Z", "failure", "feat: a", event="schedule")
    monkeypatch.setattr(point, "gh_json", _canned([failed], [], seen))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])

    assert point.main() == 0

    out = capsys.readouterr().out
    first = out.split("## Open pull requests")[0]
    assert "the nightly run FAILED" in first
    assert "u/2026-11-05T09:00:00Z" in first
    run_call = next(a for a in seen if a[0] == "run")
    assert int(run_call[run_call.index("--limit") + 1]) >= 100


def test_the_report_says_none_when_the_ci_is_clear(point, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        point, "gh_json", _canned([_run("2026-11-05T09:00:00Z", "success")], [], [])
    )
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    section = (
        capsys.readouterr().out.split("## CI that stops the merges")[1].split("## ")[0]
    )
    assert section.strip() == "- none"


def test_the_nightly_issue_is_left_out_of_the_owner_notes(
    point, monkeypatch, capsys
) -> None:
    # Its comments are the workflow's (`## CI nightly`), and a plain line on it would
    # read as an owner note on a piece of work: the nightly's issue is not one.
    nightly = {
        **_nightly(77),
        "comments": [
            {
                "body": "## CI nightly (workflow, 2026-11-06)\n\nStill red: u",
                "createdAt": "2026-11-06T03:00:00Z",
            },
            {
                "body": "looking at the macOS failure",
                "createdAt": "2026-11-06T04:00:00Z",
            },
        ],
    }
    monkeypatch.setattr(point, "gh_json", _canned([], [nightly], []))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    notes = (
        capsys.readouterr().out.split("## Since the last point")[1].split("\n## ")[0]
    )
    assert "nothing new" in notes
    assert "#77 note (" not in notes


# --- the window starts at the last Point; the lanes are listed (#432) ---------


def _tracking(comments: list[dict]) -> dict:
    return {**_issue(51, "Point: next steps", []), "comments": comments}


def _lane(lane: str, at: str, done: str, nxt: str) -> dict:
    return _c(
        f"## Lane ({lane}, 2026-11-12)\n\n**Done**: {done}\n**Next**: {nxt}\n", at
    )


def test_since_is_the_last_point_not_a_later_lane_note(point) -> None:
    # A note posted after the Point (a lane's, or the owner's) must not move the
    # window: the next point would skip every report written before it (FM-220).
    tracking = _tracking(
        [
            _c("## Point (design session, 2026-11-10)\n\nold", "2026-11-10T17:00:00Z"),
            _c("## Point (design session, 2026-11-11)\n\nnew", "2026-11-11T08:00:00Z"),
            _lane("S1", "2026-11-11T09:00:00Z", "S1.1 done", "S1.2 next"),
            _c("a plain owner note", "2026-11-11T10:00:00Z"),
        ]
    )
    assert point.last_point_time(tracking["comments"]) == "2026-11-11T08:00:00Z"
    assert point.since_default(tracking) == datetime(
        2026, 11, 11, 8, 0, tzinfo=timezone.utc
    )


def test_without_a_point_the_window_is_seven_days(point) -> None:
    notes = [_lane("S1", "2026-11-11T09:00:00Z", "x", "y"), _c("a note", "t")]
    assert point.last_point_time(notes) is None
    for tracking in (None, _tracking([]), _tracking(notes)):
        before = datetime.now(timezone.utc)
        since = point.since_default(tracking)
        after = datetime.now(timezone.utc)
        assert before - timedelta(days=7) <= since <= after - timedelta(days=7)


def test_the_latest_lane_note_of_each_lane_is_listed(point) -> None:
    comments = [
        _lane("S2", "2026-11-12T11:00:00Z", "S2.1 `/plan 430`", "wait for CI of #12"),
        _lane("S1", "2026-11-12T09:00:00Z", "S1.1 old", "old"),
        _lane(
            "S1",
            "2026-11-12T10:31:00Z",
            "S1.2 `/review 445` (sonnet): ready to merge",
            "you merge #445",
        ),
        _lane("S10", "2026-11-12T12:00:00Z", "tenth", "n10"),
        _lane("S3", "2026-11-11T23:59:59Z", "before the window", "x"),
        _c(
            "## Point (design session, 2026-11-12)\n\nnot a lane",
            "2026-11-12T12:00:00Z",
        ),
        _c("a plain note", "2026-11-12T12:30:00Z"),
    ]
    # sorted by lane number, one line each, the window's start included
    assert point.lane_notes(comments, "2026-11-12T09:00:00Z") == [
        "- S1 (2026-11-12T10:31:00Z): S1.2 `/review 445` (sonnet): ready to merge"
        " / you merge #445",
        "- S2 (2026-11-12T11:00:00Z): S2.1 `/plan 430` / wait for CI of #12",
        "- S10 (2026-11-12T12:00:00Z): tenth / n10",
    ]
    assert point.lane_notes(comments, "2026-11-13T00:00:00Z") == []


def test_a_lane_note_line_is_one_short_line(point) -> None:
    long = "word " * 80
    (line,) = point.lane_notes([_lane("S1", "2026-11-12T09:00:00Z", long, long)], "")
    assert "\n" not in line
    assert len(line) < 300
    assert line.count("...") == 2
    # a note that lacks a field is still listed, with a dash for it
    bare = _c("## Lane (S2, 2026-11-12)\n\nsomething else", "2026-11-12T09:00:00Z")
    assert point.lane_notes([bare], "") == ["- S2 (2026-11-12T09:00:00Z): - / -"]


def test_a_lane_note_line_is_ascii_whatever_the_subagent_wrote(point) -> None:
    # A subagent writes arrows and dashes freely, and a redirected stdout on Windows
    # is cp1252 with strict errors: one arrow would end the whole report (FM-20).
    note = _lane(
        "S1",
        "2026-11-12T09:00:00Z",
        "step \N{RIGHTWARDS ARROW} done",
        "you \N{EM DASH} merge #5 \N{GREEK SMALL LETTER MU}m",
    )
    (line,) = point.lane_notes([note], "")
    assert line == "- S1 (2026-11-12T09:00:00Z): step -> done / you - merge #5 ?m"
    assert line.isascii()


def test_the_report_lists_the_lanes_right_after_the_open_prs(
    point, monkeypatch, capsys
) -> None:
    tracking = _tracking(
        [
            _c("## Point (design session, 2026-11-11)\n\nnew", "2026-11-11T08:00:00Z"),
            _lane("S1", "2026-11-11T09:00:00Z", "S1.1 plan", "S1.2 develop"),
            _c("an owner note", "2026-11-11T10:00:00Z"),
        ]
    )
    monkeypatch.setattr(point, "gh_json", _canned([], [tracking], []))
    monkeypatch.setattr(sys, "argv", ["point.py"])
    assert point.main() == 0
    out = capsys.readouterr().out
    # the window starts at the last Point, whatever was posted after it
    assert "since 2026-11-11T08:00:00Z" in out.splitlines()[0]
    headings = [line for line in out.splitlines() if line.startswith("## ")]
    assert headings[1:3] == ["## Open pull requests", "## Lanes since the last point"]
    lanes = out.split("## Lanes since the last point")[1].split("\n## ")[0]
    assert "- S1 (2026-11-11T09:00:00Z): S1.1 plan / S1.2 develop" in lanes
    assert out.split("## Merged since")[0].isascii()


def test_the_lanes_section_says_none_without_a_note(point, monkeypatch, capsys) -> None:
    monkeypatch.setattr(point, "gh_json", _canned([], [_tracking([])], []))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    lanes = capsys.readouterr().out.split("## Lanes since the last point")[1]
    assert lanes.split("\n## ")[0].strip() == "- none"


def test_a_lane_note_is_not_an_owner_note(point, monkeypatch, capsys) -> None:
    # A lane's note on a piece of work is the machine's, not the owner's: only the plain
    # comment beside it is an owner note.
    work = {
        **_issue(5, "feat: y", ["status: in-progress"]),
        "comments": [
            _lane("S1", "2026-11-12T10:00:00Z", "S1.1 done", "S1.2 next"),
            _c("is the stand free on Friday?", "2026-11-12T11:00:00Z"),
        ],
    }
    monkeypatch.setattr(point, "gh_json", _canned([], [work], []))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    notes = (
        capsys.readouterr().out.split("## Since the last point")[1].split("\n## ")[0]
    )
    assert "#5 note (unknown): is the stand free on Friday?" in notes
    assert "Lane" not in notes


def test_an_unheaded_comment_is_a_note_named_by_its_author(
    point, monkeypatch, capsys
) -> None:
    # Two people write plain notes now (the owner and a collaborator): the point
    # prints who wrote each one instead of reading every note as the owner's.
    work = {
        **_issue(5, "feat: y", ["status: in-progress"]),
        "comments": [
            {
                "author": {"login": "someone"},
                "body": "a note",
                "createdAt": "2026-11-12T10:00:00Z",
            },
            {"body": "no author at all", "createdAt": "2026-11-12T11:00:00Z"},
            {
                "author": None,
                "body": "a null author",
                "createdAt": "2026-11-12T12:00:00Z",
            },
            {
                "author": {"login": ""},
                "body": "an empty login",
                "createdAt": "2026-11-12T13:00:00Z",
            },
        ],
    }
    monkeypatch.setattr(point, "gh_json", _canned([], [work], []))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    notes = (
        capsys.readouterr().out.split("## Since the last point")[1].split("\n## ")[0]
    )
    assert "- #5 note (someone): a note" in notes
    assert "- #5 note (unknown): no author at all" in notes
    assert "- #5 note (unknown): a null author" in notes
    assert "- #5 note (unknown): an empty login" in notes
    assert "owner note" not in notes


def test_a_comment_read_through_the_rest_api_keeps_its_author(point) -> None:
    # `gh api` names the author `user`; the point reads the listing's `author`. A
    # tracking issue past 100 comments is read through the REST API, and its notes
    # must not all become `note (unknown)`.
    rest = {**_rest("a note", "2026-11-12T10:00:00Z", 7), "user": {"login": "someone"}}
    assert point.author_of(point.normalise_comment(rest)) == "someone"
    assert point.author_of(point.normalise_comment(_rest("x", "t"))) == "unknown"
    assert point.normalise_comment({**rest, "user": None})["body"] == "a note"


# --- a listing cuts comments at 100; the point reads them all (#591) -----------------


def _rest(body: str, at: str, n: int = 0) -> dict:
    """A comment as `gh api` returns it: REST names, not the listing's."""
    return {
        "node_id": f"IC_{n}",
        "body": body,
        "created_at": at,
        "html_url": f"https://github.com/o/r/issues/51#issuecomment-{n}",
    }


def _old_comments(count: int) -> list[dict]:
    """``count`` comments of 2026-11-13, the last of them the Point of that day."""
    notes = [
        _c(f"note {i}", f"2026-11-13T10:{i % 60:02d}:00Z") for i in range(count - 1)
    ]
    point_of_the_day = _c(
        "## Point (design session, 2026-11-13)\n\nold", "2026-11-13T19:23:40Z"
    )
    return [*notes, point_of_the_day]


def _paged(listed: list[dict], newer: list[dict], seen: list[tuple[str, ...]]):
    """A stand-in for `gh_json` where `gh api --paginate` holds more than the listing."""
    full = [_rest(c["body"], c["createdAt"], n) for n, c in enumerate(listed)] + newer
    tracking = {**_issue(51, "Point: next steps", []), "comments": listed}
    base = _canned([], [tracking], [])

    def fake(*args: str):
        if args[:2] == ("api", "--paginate"):
            seen.append(args)
            assert args[2].startswith("repos/EPFL-TOP/smart-microscope-control/")
            return full
        return base(*args)

    return fake


def test_a_tracking_issue_with_more_than_100_comments_gives_its_since_from_the_newest_point(
    point, monkeypatch, capsys
) -> None:
    # The listing holds the first 100 comments of #51 (the newest Point among them is the
    # one of 11-13); the paginated read holds the Points of 11-17 and 11-18 and a lane note.
    listed = _old_comments(100)
    newer = [
        _rest(
            "## Point (design session, 2026-11-17)\n\nmid", "2026-11-17T20:00:00Z", 101
        ),
        _rest(
            "## Point (design session, 2026-11-18)\n\nnew", "2026-11-18T20:17:58Z", 102
        ),
        _rest(
            "## Lane (S1, 2026-11-18)\n\n**Done**: x\n**Next**: y\n**For /point**: z\n",
            "2026-11-18T21:00:00Z",
            103,
        ),
    ]
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(point, "gh_json", _paged(listed, newer, seen))
    monkeypatch.setattr(sys, "argv", ["point.py"])
    assert point.main() == 0
    out = capsys.readouterr().out
    assert "since 2026-11-18T20:17:58Z" in out.splitlines()[0]
    # the lane note after the newest Point is listed, in the lanes' and the For /point section
    assert "- S1 (2026-11-18T21:00:00Z): x / y" in out
    assert (
        "- S1 (2026-11-18T21:00:00Z): z"
        " https://github.com/o/r/issues/51#issuecomment-103"
    ) in out
    # one read of the tracking issue's comments, paginated, 100 to a page
    assert len(seen) == 1
    assert seen[0][2].endswith("/issues/51/comments?per_page=100")


def test_an_issue_under_the_listing_cap_is_not_read_again(
    point, monkeypatch, capsys
) -> None:
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(point, "gh_json", _paged(_old_comments(99), [], seen))
    monkeypatch.setattr(sys, "argv", ["point.py"])
    assert point.main() == 0
    assert seen == []
    assert "since 2026-11-13T19:23:40Z" in capsys.readouterr().out.splitlines()[0]


def test_every_listing_the_report_reads_is_read_again_where_it_is_cut(
    point, monkeypatch, capsys
) -> None:
    # open issues, closed issues, open PRs and merged PRs: each listing cuts at 100
    cut = [_c(f"x{i}", "2026-11-10T00:00:00Z") for i in range(100)]
    asked: list[str] = []

    def fake(*args: str):
        if args[:2] == ("repo", "view"):
            return {"nameWithOwner": "o/r"}
        if args[:2] == ("issue", "list"):
            number = 3 if "open" in args else 4
            return [{**_issue(number, "feat: x", []), "comments": cut}]
        if args[:2] == ("pr", "list"):
            if "open" in args:
                pr = {"number": 5, "title": "p", "mergeStateStatus": "CLEAN"}
                return [{**pr, "comments": cut, "reviews": []}]
            return [
                {
                    "number": 6,
                    "title": "q",
                    "mergedAt": "2026-11-10T00:00:00Z",
                    "comments": [],
                    "reviews": [{"body": "r", "submittedAt": "t"}] * 100,
                }
            ]
        if args[:2] == ("api", "--paginate"):
            asked.append(args[2])
            return []
        if args[0] == "run" or args[0] == "api":
            return []
        raise AssertionError(args)

    monkeypatch.setattr(point, "gh_json", fake)
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    assert sorted(asked) == [
        "repos/o/r/issues/3/comments?per_page=100",
        "repos/o/r/issues/4/comments?per_page=100",
        "repos/o/r/issues/5/comments?per_page=100",
        "repos/o/r/pulls/6/reviews?per_page=100",
    ]


def test_the_listing_cap_is_100_comments(point) -> None:
    # measured 2026-11-19 on gh 2.103.3: the listing of #51 returned 100 of 182
    assert point.LISTING_CAP == 100


def test_a_prs_reviews_and_comments_past_the_cap_are_read_again(point) -> None:
    pr = {
        "number": 7,
        "comments": [
            _c(f"c{i}", f"2026-11-01T00:{i % 60:02d}:00Z") for i in range(100)
        ],
        "reviews": [
            {"body": f"r{i}", "submittedAt": "2026-11-01T01:00:00Z"} for i in range(100)
        ],
    }
    calls: list[str] = []
    ready = "## Review (design session, 2026-11-02)\n\n**Verdict: ready to merge**"
    asked = "## Review (design session, 2026-11-01)\n\n**Verdict: changes requested**"

    def get(*args: str):
        assert args[:2] == ("api", "--paginate")
        calls.append(args[2])
        if "/pulls/7/reviews" in args[2]:
            return [
                *[
                    {"body": f"r{i}", "submitted_at": "2026-11-01T01:00:00Z"}
                    for i in range(100)
                ],
                # an older "changes requested" first: only the dates tell the newest
                {"body": asked, "submitted_at": "2026-11-01T02:00:00Z", "state": "X"},
                {"body": ready, "submitted_at": "2026-11-02T08:00:00Z", "state": "X"},
                {"body": "", "submitted_at": None, "state": "PENDING"},
            ]
        return [_rest("last comment", "2026-11-03T00:00:00Z", 1)]

    point.complete("o/r", [pr], "pr", get)
    assert calls == [
        "repos/o/r/issues/7/comments?per_page=100",
        "repos/o/r/pulls/7/reviews?per_page=100",
    ]
    # the verdict was the 101st review: the listing could not show it
    assert point.verdict(point.review_texts(pr)) == "ready to merge"
    assert pr["comments"][0]["createdAt"] == "2026-11-03T00:00:00Z"
    assert pr["reviews"][100]["submittedAt"] == "2026-11-01T02:00:00Z"
    assert pr["reviews"][-1]["submittedAt"] == ""  # a pending review has no date


def test_an_item_at_99_is_complete_and_one_at_100_is_not(point) -> None:
    calls: list[str] = []

    def get(*args: str):
        calls.append(args[2])
        return []

    at_99 = {"number": 1, "comments": [_c("x", "t")] * 99}
    at_100 = {"number": 2, "comments": [_c("x", "t")] * 100}
    point.complete("o/r", [at_99, at_100], "issue", get)
    assert calls == ["repos/o/r/issues/2/comments?per_page=100"]
    assert at_99["comments"] == [_c("x", "t")] * 99
    assert at_100["comments"] == []


def test_a_paginated_read_that_is_not_a_list_stops_the_report(point) -> None:
    item = {"number": 3, "comments": [_c("x", "t")] * 100}
    with pytest.raises(SystemExit, match="expected a list"):
        point.complete("o/r", [item], "issue", lambda *a: None)


# --- the For /point lines of every lane note (#591) ------------------------------------


def _note(lane: str, at: str, body: str, url: str = "") -> dict:
    return {
        "body": f"## Lane ({lane}, 2026-11-18)\n\n{body}\n",
        "createdAt": at,
        "url": url,
    }


def test_the_for_point_lines_of_every_lane_note_since_the_point_are_listed(
    point,
) -> None:
    comments = [
        _note(
            "S1", "2026-11-18T09:00:00Z", "**Done**: a\n**For /point**: first, S1", "u1"
        ),
        _note("S2", "2026-11-18T10:00:00Z", "**Done**: b\n**Next**: you merge"),
        # a later note of S1 has none: the first note's line must still be listed
        _note("S1", "2026-11-18T11:00:00Z", "**Done**: c\n**Next**: wait"),
        _note("S3", "2026-11-18T12:00:00Z", "**For /point**:\n- one\n- two", "u3"),
        _note("S2", "2026-11-17T23:59:59Z", "**For /point**: before the window", "u0"),
        _note("S4", "2026-11-18T00:00:00Z", "**For /point**: at the start", "u4"),
        _c("## Point (design session, 2026-11-18)\n\n**For /point**: not a lane", "t"),
        _c("**For /point**: a plain owner note", "2026-11-18T13:00:00Z"),
    ]
    assert point.for_point_lines(comments, "2026-11-18T00:00:00Z") == [
        "- S4 (2026-11-18T00:00:00Z): at the start u4",
        "- S1 (2026-11-18T09:00:00Z): first, S1 u1",
        "- S3 (2026-11-18T12:00:00Z): one u3",
        "- S3 (2026-11-18T12:00:00Z): two u3",
    ]
    assert point.for_point_lines(comments, "2026-11-19T00:00:00Z") == []


def test_a_for_point_line_with_the_colon_inside_the_bold_is_read(point) -> None:
    note = _note("S1", "2026-11-18T09:00:00Z", "**For /point:** a decision for #27")
    assert point.for_point_lines([note], "") == [
        "- S1 (2026-11-18T09:00:00Z): a decision for #27"
    ]
    assert point.field("**Done:** x\n**Next**: y", "Done") == ["x"]
    assert point.field("**Done:** x\n**Next**: y", "Next") == ["y"]


def test_a_for_point_line_is_one_short_line(point) -> None:
    long = "word " * 80
    note = _note("S1", "2026-11-18T09:00:00Z", f"**For /point**: {long}end")
    (line,) = point.for_point_lines([note], "")
    assert "\n" not in line
    assert len(line) < 300
    assert line.endswith("...")


def test_a_for_point_line_is_ascii_whatever_the_subagent_wrote(point) -> None:
    # The arrow and the dash sit inside the first characters: a long text would clip
    # them away before the transliteration is asked for (FM-20).
    text = "go \N{RIGHTWARDS ARROW} on \N{EM DASH} #5 \N{GREEK SMALL LETTER MU}m"
    note = _note("S1", "2026-11-18T09:00:00Z", f"**For /point**: {text}", "u1")
    assert point.for_point_lines([note], "") == [
        "- S1 (2026-11-18T09:00:00Z): go -> on - #5 ?m u1"
    ]


def test_the_report_lists_the_for_point_lines_under_their_own_heading(
    point, monkeypatch, capsys
) -> None:
    tracking = _tracking(
        [
            _c("## Point (design session, 2026-11-11)\n\nnew", "2026-11-11T08:00:00Z"),
            _note(
                "S1", "2026-11-11T09:00:00Z", "**Done**: a\n**For /point**: early one"
            ),
            _note("S1", "2026-11-11T10:00:00Z", "**Done**: b\n**Next**: c"),
        ]
    )
    monkeypatch.setattr(point, "gh_json", _canned([], [tracking], []))
    monkeypatch.setattr(sys, "argv", ["point.py"])
    assert point.main() == 0
    out = capsys.readouterr().out
    headings = [line for line in out.splitlines() if line.startswith("## ")]
    assert headings[2:4] == [
        "## Lanes since the last point",
        "## For /point (every lane note since the last point)",
    ]
    section = out.split("## For /point")[1].split("\n## ")[0]
    assert "- S1 (2026-11-11T09:00:00Z): early one" in section
    # the lanes' section keeps showing the newest note only
    lanes = out.split("## Lanes since the last point")[1].split("\n## ")[0]
    assert "early one" not in lanes
    assert out.split("## Merged since")[0].isascii()


def test_the_for_point_section_says_none_without_a_line(
    point, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(point, "gh_json", _canned([], [_tracking([])], []))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    section = capsys.readouterr().out.split("## For /point")[1]
    assert section.split("\n## ")[0].strip().splitlines()[1:] == ["- none"]


# --- the whole report is ASCII; every `gh` child has a timeout (#591) ----------------


def test_a_report_with_an_arrow_in_any_text_does_not_end_on_a_cp1252_stdout(
    point, monkeypatch
) -> None:
    # A Report, a fix round's follow-ups and an owner note are written by subagents or
    # typed by the owner, and a redirected stdout on Windows is cp1252 with strict errors:
    # the colon inside the bold is read since #591, and one arrow lost the whole report
    # (the older `**Follow-ups**:` form already did, FM-20).
    arrow = "\N{RIGHTWARDS ARROW}"
    work = {
        **_issue(5, "feat: y", ["status: in-progress"]),
        "comments": [
            _c(
                f"## Report (develop session)\n\n**Friction:** {arrow} slow",
                "2026-11-10",
            ),
            _c("is the \N{GREEK SMALL LETTER MU}m stage free?", "2026-11-11"),
        ],
    }
    merged = {
        "number": 9,
        "title": "feat: x",
        "mergedAt": "2026-11-10T00:00:00Z",
        "reviews": [],
        "comments": [
            _c(
                f"## Review addressed (develop session)\n\n**Follow-ups:** {arrow} track it",
                "2026-11-10T01:00:00Z",
            )
        ],
    }
    base = _canned([], [work], [])

    def fake(*args: str):
        if args[:2] == ("pr", "list") and "merged" in args:
            return [merged]
        return base(*args)

    monkeypatch.setattr(point, "gh_json", fake)
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    assert point.main() == 0  # a UnicodeEncodeError here is the bug
    stream.flush()
    out = raw.getvalue().decode("cp1252")
    assert out.isascii()
    assert "- #9 fix round, follow-up: -> track it" in out
    assert "- #5 report, Friction: -> slow" in out
    assert "- #5 note (unknown): is the ?m stage free?" in out


def test_every_gh_child_of_the_point_has_no_console_a_timeout_and_utf8(
    point, monkeypatch
) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="[1]", stderr="")

    monkeypatch.setattr(point.subprocess, "run", fake_run)
    assert point.gh_json("issue", "list") == [1]
    assert seen["cmd"] == ["gh", "issue", "list"]
    assert seen["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
    assert seen["timeout"] == point.GH_TIMEOUT_S
    assert seen["stdin"] == subprocess.DEVNULL

    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, point.GH_TIMEOUT_S)

    monkeypatch.setattr(point.subprocess, "run", hang)
    with pytest.raises(SystemExit, match="timed out"):
        point.gh("api", "--paginate", "repos/o/r/issues/51/comments")


# --- two tracks (ADR-0010, #653): one report per track --------------------------------


def _work(number: int, title: str, labels: list[str], body: str = "") -> dict:
    return {**_issue(number, title, labels), "body": body}


# The open issues of 2026-11-25, in `gh`'s shape (titles are invented, labels are the day's).
TODAY = [
    _work(
        636,
        "feat: the ten-to-forty plan",
        [
            "type: feature",
            "status: in-progress",
            "priority: p1",
            "area: algorithms",
            "track: analysis",
        ],
    ),
    _work(
        632,
        "adr: the notebook",
        [
            "type: adr",
            "area: ui",
            "status: needs-decision",
            "priority: p1",
            "track: analysis",
        ],
    ),
    _work(
        654,
        "chore: the analysis docs",
        ["type: chore", "area: docs", "priority: p1", "track: analysis"],
    ),
    _work(
        600,
        "feat: models by name",
        [
            "type: feature",
            "area: plugins",
            "area: ui",
            "status: blocked",
            "priority: p1",
            "track: analysis",
        ],
        "**Depends on**: #599, #567",
    ),
    _work(
        634,
        "feat: the focus screen",
        [
            "type: feature",
            "area: plugins",
            "area: ui",
            "scope: nikon-ti2",
            "priority: p0",
        ],
    ),
    _work(
        644,
        "feat: the focus sweep",
        [
            "type: feature",
            "area: plugins",
            "scope: nikon-ti2",
            "status: in-progress",
            "priority: p1",
        ],
    ),
    _work(653, "chore: the tracks", ["type: chore", "area: ci", "priority: p1"]),
    _work(
        599,
        "adr: the models",
        ["type: adr", "area: plugins", "status: needs-decision", "priority: p1"],
    ),
    _work(
        635,
        "feat: the stand side of the models",
        ["type: feature", "area: plugins", "status: blocked", "priority: p1"],
        "**Depends on**: #599",
    ),
    _issue(51, "Point: next steps", []),
    _issue(670, "Point (analysis): next steps", ["track: analysis"]),
]
CLOSED_REFS = {567: {"state": "closed"}}


def _report(
    point,
    monkeypatch,
    capsys,
    tmp_path,
    issues,
    *,
    track: str | None = None,
    since: str | None = "2026-11-01",
    **canned,
) -> str:
    """The report of a track on canned data, with ``point.ROOT`` in an empty folder.

    The real ADR files and survey folders must not make an assertion pass by accident.
    """
    monkeypatch.setattr(point, "ROOT", tmp_path)
    monkeypatch.setattr(point, "gh_json", _canned([], issues, [], **canned))
    argv = ["point.py"]
    if since:
        argv += ["--since", since]
    if track:
        argv += ["--track", track]
    monkeypatch.setattr(sys, "argv", argv)
    assert point.main() == 0
    return capsys.readouterr().out


def _section(out: str, heading: str) -> str:
    return out.split(f"## {heading}")[1].split("\n## ")[0]


def _mentions(out: str, number: int) -> bool:
    return re.search(rf"#{number}\b", out) is not None


def test_the_analysis_report_lists_the_analysis_issues_and_none_of_the_stands(
    point, monkeypatch, capsys, tmp_path
) -> None:
    out = _report(
        point,
        monkeypatch,
        capsys,
        tmp_path,
        TODAY,
        track="analysis",
        refs=CLOSED_REFS,
    )
    assert out.splitlines()[0].startswith("# Point facts (analysis track, ")
    status = _section(out, "Open issues by status").strip().splitlines()
    assert status == [
        "- in-progress: #636",
        "- blocked: #600",
        "- needs-decision: #632",
        "- no status: #654",
    ]
    for number in (634, 644, 653, 635, 51, 670):
        assert not _mentions(out, number), number
    # #599 is the stand's: it is named in the last section only
    before_asks, asks = out.split("## Asks to the other track")
    assert not _mentions(before_asks, 599)
    assert asks.strip().splitlines() == [
        "- #599 (stand track, needs-decision): awaited by #600"
    ]
    blocked = _section(out, "Blocked issues and their dependencies")
    assert "- #600: #567 closed; waits on the stand track (see Asks)" in blocked
    assert "ALL MERGED" not in blocked


def test_the_stand_report_names_the_analysis_issues_only_among_the_asks(
    point, monkeypatch, capsys, tmp_path
) -> None:
    waiting = _work(
        637, "feat: wires the plan", ["type: feature"], "**Depends on**: #636"
    )
    out = _report(
        point,
        monkeypatch,
        capsys,
        tmp_path,
        [*TODAY, waiting],
        refs=CLOSED_REFS,
    )
    assert out.splitlines()[0].startswith("# Point facts (stand track, ")
    for number in (632, 654, 600, 670):
        assert not _mentions(out, number), number
    before_asks, asks = out.split("## Asks to the other track")
    assert not _mentions(before_asks, 636)
    assert asks.strip().splitlines() == [
        "- #636 (analysis track, in-progress): awaited by #637"
    ]
    # the stand's own issues are all there
    status = _section(out, "Open issues by status")
    for number in (634, 644, 653, 599, 635, 637):
        assert _mentions(status, number), number
    # #635 waits on #599, an open issue of its own track: it is named, not an ask
    assert "- #635: #599 open" in _section(out, "Blocked issues")


def test_an_ask_is_an_open_issue_of_the_other_track_only(point) -> None:
    rules = point.tracks.read_rules()
    issues = [
        _work(10, "a", ["status: blocked"], "**Depends on**: #20, #11"),
        _work(11, "b", [], "**Depends on**: #99"),  # #99 is closed: not in the listing
        _work(12, "c", [], "**Depends on**: #20"),
        _work(20, "d", ["status: ready", "track: analysis"], "**Depends on**: #10"),
    ]
    assert point.asks("stand", issues, [], rules) == [
        "- #20 (analysis track, ready): awaited by #10, #12"
    ]
    # the same list read by the other track: #20 waits on the stand's #10
    assert point.asks("analysis", issues, [], rules) == [
        "- #10 (stand track, blocked): awaited by #20"
    ]
    assert point.asks("stand", issues[:3], [], rules) == ["- none"]


def test_the_asks_are_in_number_order_and_name_a_waiting_issue_once(point) -> None:
    rules = point.tracks.read_rules()
    issues = [
        # listed in the order that would give the wrong answer if nothing sorted them
        _work(12, "c", [], "**Depends on**: #20, #20"),
        _work(10, "a", [], "**Depends on**: #21, #20"),
        _work(21, "d", ["track: analysis", "status: ready"]),
        _work(20, "e", ["track: analysis", "status: blocked"]),
    ]
    assert point.asks("stand", issues, [], rules) == [
        "- #20 (analysis track, blocked): awaited by #10, #12",
        "- #21 (analysis track, ready): awaited by #10",
    ]
    crossings = [
        _pr_of(9, ["pyproject.toml"], (21,)),
        _pr_of(7, ["pyproject.toml"], (21,)),
    ]
    lines = point.asks("analysis", [issues[2]], crossings, rules)
    assert [ln.split(" of ")[0] for ln in lines] == ["- PR #7", "- PR #9"]


def test_the_first_issue_of_a_pr_is_the_one_github_says_it_closes(point) -> None:
    # closing #5 on a branch named for #9: both are open issues, #5 is the first
    pr = _pr_of(8, ["pyproject.toml"], (5,), branch="feat/9-x")
    by_number = {5: _work(5, "a", ["track: analysis"]), 9: _work(9, "b", [])}
    assert point.pr_open_issue(pr, by_number)["number"] == 5
    assert point.pr_open_issue(_pr_of(8, [], branch="feat/7-x"), by_number) is None


def test_an_ask_takes_a_plans_depends_on_line_before_the_body(point) -> None:
    rules = point.tracks.read_rules()
    planned = {
        **_work(10, "a", [], "**Depends on**: #11"),
        "comments": [_c("## Plan (design session)\n\n**Depends on**: #20", "t")],
    }
    issues = [planned, _work(11, "b", []), _work(20, "c", ["track: analysis"])]
    assert point.asks("stand", issues, [], rules) == [
        "- #20 (analysis track, no status): awaited by #10"
    ]


def _pr_of(
    number: int,
    paths: list[str],
    closing: tuple[int, ...] = (),
    branch: str = "",
    changed: int | None = None,
) -> dict:
    return {
        "number": number,
        "title": f"pr {number}",
        "mergeStateStatus": "CLEAN",
        "statusCheckRollup": [],
        "reviews": [],
        "comments": [],
        "isDraft": False,
        "headRefName": branch,
        "closingIssuesReferences": [{"number": n} for n in closing],
        "files": [{"path": p} for p in paths],
        "changedFiles": len(paths) if changed is None else changed,
    }


ANALYSIS_PATH = "src/smc/selection/builtin.py"


def test_a_pr_is_listed_in_the_report_of_the_track_of_its_paths(
    point, monkeypatch, capsys, tmp_path
) -> None:
    prs = [
        _pr_of(701, [ANALYSIS_PATH, "docs/algorithms/README.md"], (654,)),
        _pr_of(702, [ANALYSIS_PATH, "pyproject.toml"], (636,)),
        # two files listed of three: a cut listing is read as the stand's (FM-256)
        _pr_of(703, [ANALYSIS_PATH, "docs/algorithms/README.md"], changed=3),
        # a plain PR of the stand's own issue and paths: nobody is asked anything
        _pr_of(707, ["pyproject.toml"], branch="feat/653-x"),
    ]
    stand = _report(point, monkeypatch, capsys, tmp_path, TODAY, prs=prs)
    analysis = _report(
        point, monkeypatch, capsys, tmp_path, TODAY, prs=prs, track="analysis"
    )
    stand_prs = _section(stand, "Open pull requests")
    analysis_prs = _section(analysis, "Open pull requests")
    assert not _mentions(stand_prs, 701)
    assert [ln[:6] for ln in analysis_prs.strip().splitlines()] == ["- #701"]
    # a crossing: the issue is the analysis track's, a path is the stand's
    (crossing,) = (ln for ln in stand_prs.splitlines() if ln.startswith("- #702"))
    assert crossing.endswith("| crossing: its issue is the analysis track's")
    (cut,) = (ln for ln in stand_prs.splitlines() if ln.startswith("- #703"))
    assert "crossing" not in cut
    # the analysis lead is asked for the crossing, by its stand paths, and for nothing else:
    # #701 and #707 are each track's own, #703 has no issue (the whole section, both ways)
    assert (
        _section(analysis, "Asks to the other track").strip().splitlines()
        == [
            "- #599 (stand track, needs-decision): awaited by #600",  # TODAY's, before the PRs
            "- PR #702 of #636 is on the stand track's paths: pyproject.toml",
        ]
    )
    assert not _mentions(analysis_prs, 702)
    assert _section(stand, "Asks to the other track").strip().splitlines() == ["- none"]


def test_a_pr_of_a_stand_issue_on_analysis_paths_is_the_analysis_leads_and_an_ask(
    point, monkeypatch, capsys, tmp_path
) -> None:
    prs = [_pr_of(704, [ANALYSIS_PATH], branch="feat/653-x")]
    stand = _report(point, monkeypatch, capsys, tmp_path, TODAY, prs=prs)
    analysis = _report(
        point, monkeypatch, capsys, tmp_path, TODAY, prs=prs, track="analysis"
    )
    assert not _mentions(_section(stand, "Open pull requests"), 704)
    assert (
        "- PR #704 of #653 is on the analysis track's paths: " + ANALYSIS_PATH
    ) in stand.split("## Asks to the other track")[1]
    (line,) = (
        ln
        for ln in _section(analysis, "Open pull requests").splitlines()
        if ln.startswith("- #704")
    )
    assert line.endswith("| crossing: its issue is the stand track's")


def test_a_cut_listing_with_no_path_to_name_is_still_an_ask(point) -> None:
    rules = point.tracks.read_rules()
    issues = [_work(636, "x", ["track: analysis"])]
    cut = _pr_of(705, [ANALYSIS_PATH], (636,), changed=4)
    (line,) = point.asks("analysis", issues, [cut], rules)
    assert line == (
        "- PR #705 of #636 is on the stand track's paths: 1 of 4 files listed, the rest cut"
    )
    many = _pr_of(706, [f"src/smc/ui/p{i}.py" for i in range(5)], (636,))
    (line,) = point.asks("analysis", issues, [many], rules)
    assert line.endswith(
        "src/smc/ui/p0.py, src/smc/ui/p1.py, src/smc/ui/p2.py (+2 more)"
    )


def test_the_merged_prs_are_listed_in_the_report_of_the_track_of_their_paths(
    point, monkeypatch, capsys, tmp_path
) -> None:
    def merged(number: int, paths: list[str]) -> dict:
        return {
            "number": number,
            "title": f"merged {number}",
            "mergedAt": "2026-11-10T00:00:00Z",
            "reviews": [],
            "comments": [],
            "files": [{"path": p} for p in paths],
            "changedFiles": len(paths),
        }

    done = [
        merged(801, [ANALYSIS_PATH]),
        merged(802, ["pyproject.toml", ANALYSIS_PATH]),
    ]
    stand = _report(point, monkeypatch, capsys, tmp_path, TODAY, merged=done)
    analysis = _report(
        point, monkeypatch, capsys, tmp_path, TODAY, merged=done, track="analysis"
    )
    assert _mentions(_section(stand, "Merged since the last point"), 802)
    assert not _mentions(_section(stand, "Merged since the last point"), 801)
    assert _mentions(_section(analysis, "Merged since the last point"), 801)
    assert not _mentions(_section(analysis, "Merged since the last point"), 802)


def test_each_report_reads_the_lanes_of_its_own_tracking_issue(
    point, monkeypatch, capsys, tmp_path
) -> None:
    stand_issue = {
        **_issue(51, "Point: next steps", []),
        "comments": [
            _c("## Point (design session, 2026-11-11)\n\nS", "2026-11-11T08:00:00Z"),
            _lane("S1", "2026-11-11T09:00:00Z", "S1.1 stand", "S1.2 next"),
        ],
    }
    analysis_issue = {
        **_issue(670, "Point (analysis): next steps", ["track: analysis"]),
        "comments": [
            _c("## Point (design session, 2026-11-14)\n\nA", "2026-11-14T08:00:00Z"),
            _lane("A1", "2026-11-14T09:00:00Z", "A1.1 analysis", "A1.2 next"),
            _lane("S2", "2026-11-14T10:00:00Z", "a stray note", "x"),
        ],
    }
    issues = [stand_issue, analysis_issue]
    stand = _report(point, monkeypatch, capsys, tmp_path, issues, since=None)
    analysis = _report(
        point, monkeypatch, capsys, tmp_path, issues, since=None, track="analysis"
    )
    # each window opens at the last Point of its own tracking issue (FM-220)
    assert "since 2026-11-11T08:00:00Z" in stand.splitlines()[0]
    assert "since 2026-11-14T08:00:00Z" in analysis.splitlines()[0]
    stand_lanes = _section(stand, "Lanes since the last point")
    assert "- S1 (2026-11-11T09:00:00Z): S1.1 stand / S1.2 next" in stand_lanes
    assert "A1" not in stand_lanes
    analysis_lanes = _section(analysis, "Lanes since the last point")
    assert "- A1 (2026-11-14T09:00:00Z): A1.1 analysis / A1.2 next" in analysis_lanes
    assert "S1" not in analysis_lanes


def test_a_lane_note_of_an_analysis_lane_is_read(point) -> None:
    comments = [
        _lane("A2", "2026-11-12T11:00:00Z", "A2.1 `/plan 700`", "wait"),
        _lane("A1", "2026-11-12T09:00:00Z", "A1.1 done", "A1.2 develop"),
        _lane("S1", "2026-11-12T08:00:00Z", "stand", "x"),
        _lane("A10", "2026-11-12T12:00:00Z", "tenth", "n10"),
    ]
    assert point.lane_notes(comments, "") == [
        "- A1 (2026-11-12T09:00:00Z): A1.1 done / A1.2 develop",
        "- A2 (2026-11-12T11:00:00Z): A2.1 `/plan 700` / wait",
        "- A10 (2026-11-12T12:00:00Z): tenth / n10",
        "- S1 (2026-11-12T08:00:00Z): stand / x",
    ]
    note = _note("A1", "2026-11-18T09:00:00Z", "**For /point**: an A1 line", "u1")
    assert point.for_point_lines([note], "") == [
        "- A1 (2026-11-18T09:00:00Z): an A1 line u1"
    ]


def test_both_tracking_issues_are_left_out_of_the_work(point) -> None:
    issues = [
        _issue(51, "Point: next steps", []),
        _issue(670, "Point (analysis): next steps", ["track: analysis"]),
        _issue(5, "feat: x", []),
    ]
    assert [i["number"] for i in point.work_issues(issues)] == [5]


@pytest.mark.parametrize("track", [None, "analysis"])
def test_the_report_ends_with_the_asks_to_the_other_track(
    point, monkeypatch, capsys, tmp_path, track
) -> None:
    out = _report(point, monkeypatch, capsys, tmp_path, [], track=track)
    headings = [line for line in out.splitlines() if line.startswith("## ")]
    assert headings[-1] == "## Asks to the other track"
    assert out.split("## Asks to the other track")[1].strip() == "- none"


def test_the_survey_folders_are_listed_in_the_stand_report_only(
    point, monkeypatch, capsys, tmp_path
) -> None:
    (tmp_path / "local" / "surveys" / "stand-x").mkdir(parents=True)
    stand = _report(point, monkeypatch, capsys, tmp_path, [])
    analysis = _report(point, monkeypatch, capsys, tmp_path, [], track="analysis")
    assert "- stand-x: NOT CURATED YET" in stand
    assert "Survey folders" not in analysis
    assert "stand-x" not in analysis


def test_the_listings_ask_for_the_paths_and_the_labels_the_tracks_need(
    point, monkeypatch, capsys, tmp_path
) -> None:
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(point, "ROOT", tmp_path)
    monkeypatch.setattr(point, "gh_json", _canned([], [], seen))
    monkeypatch.setattr(sys, "argv", ["point.py", "--since", "2026-11-01"])
    assert point.main() == 0
    capsys.readouterr()

    def fields(kind: str, state: str) -> set[str]:
        call = next(a for a in seen if a[:2] == (kind, "list") and state in a)
        return set(call[call.index("--json") + 1].split(","))

    assert {"files", "changedFiles", "headRefName"} <= fields("pr", "open")
    assert {"files", "changedFiles"} <= fields("pr", "merged")
    assert "labels" in fields("issue", "closed")


def test_the_closed_issues_of_the_other_track_are_not_read_as_news(
    point, monkeypatch, capsys, tmp_path
) -> None:
    closed = [
        {
            **_issue(660, "feat: done", ["track: analysis"]),
            "comments": [_c("a note on the analysis side", "2026-11-12T10:00:00Z")],
        },
        {
            **_issue(661, "feat: done", []),
            "comments": [_c("a note on the stand side", "2026-11-12T10:00:00Z")],
        },
    ]
    stand = _report(point, monkeypatch, capsys, tmp_path, [], closed=closed)
    analysis = _report(
        point, monkeypatch, capsys, tmp_path, [], closed=closed, track="analysis"
    )
    assert "#661 note" in stand
    assert "#660" not in stand
    assert "#660 note" in analysis
    assert "#661" not in analysis


def test_an_unknown_track_is_a_usage_error(point, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["point.py", "--track", "lab"])
    with pytest.raises(SystemExit) as stop:
        point.main()
    assert stop.value.code == 2


def test_the_default_track_is_the_stand_and_the_stand_title_is_unchanged(
    point,
) -> None:
    assert point.TRACKING_TITLE == "Point: next steps"
    assert point.tracks.TRACKING_TITLES["stand"] == point.TRACKING_TITLE


def test_a_blocked_line_with_no_dependency_keeps_its_words(point) -> None:
    issue = {"number": 9}
    assert point.blocked_line(issue, {}, "stand", {}) == "- #9: no Depends-on line"
    states = {4: "merged", 5: "closed"}
    assert point.blocked_line(issue, states, "stand", {}) == (
        "- #9: #4 merged, #5 closed  -> ALL MERGED, can be unblocked"
    )
    # an open issue of the other track is the whole line after the colon when alone
    other = {7: _work(7, "x", ["track: analysis"])}
    assert point.blocked_line(issue, {7: "open"}, "stand", other) == (
        "- #9: waits on the analysis track (see Asks)"
    )
    # the same issue seen by its own track is an ordinary dependency
    assert point.blocked_line(issue, {7: "open"}, "analysis", other) == "- #9: #7 open"
