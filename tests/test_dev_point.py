"""scripts/dev/point.py feeds /point; pin the parsing it relies on."""

import importlib.util
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


def _canned(runs: list[dict], issues: list[dict], seen: list[tuple[str, ...]]):
    def fake(*args: str):
        seen.append(args)
        if args[:2] == ("repo", "view"):
            return {"nameWithOwner": "EPFL-TOP/smart-microscope-control"}
        if args[:2] == ("issue", "list"):
            return issues if "open" in args else []
        if args[:2] == ("pr", "list"):
            return []
        if args[0] == "run":
            return runs
        if args[0] == "api":
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
    assert "#77 owner note" not in notes


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
    assert "#5 owner note: is the stand free on Friday?" in notes
    assert "Lane" not in notes
