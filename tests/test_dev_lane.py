"""scripts/dev/lane.py drives /lane: pin the grammar of a lane and the state it derives.

Every fact is a literal dict (no `gh`). The Point is the text of the 2026-11-11
afternoon Point on #51, copied as a string (FM-101).
"""

import importlib.util
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lane_script", ROOT / "scripts" / "dev" / "lane.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lane = _load()

POINT = """## Point (design session, 2026-11-11, afternoon) — session 4 cancelled; the day goes to the UI and the scan, locally

**Where we are.** This morning's two lanes are merged: the plate's declared orientation (#422 as #438 and #440, design #435) and the searching autofocus at a seen floor (#421 as #436, design #433); #429 too. `main` is green, no PR is open. **Session 4 at the Ti2-E was cancelled** (note on #198); no date yet. Nothing below needs the stand.

**The next visit.** Yes: `/hardware-session nikon-ti2 before` (Opus) runs again, the evening before or the morning of the visit, **after the last merge**: it starts from today's checklist on #198 and updates it against what was merged since (the orientation choice on the page, D9, whatever the UI lane changes). A checklist older than the code it drives is how session 3 lost its registration.

**Today** (one session at a time inside a lane)

**S1 — the UI as one guided scan (#431)** — `docs/design/m5-web-ui.md`, then the page
S1.1 `/plan 431` (Opus), **with you**: it starts with a live walk on `demo-sample` (Claude in Chrome, `/ui-check` live mode) of the path as it is now, with the new orientation choice and the live cross; it folds in #373 (which objective a registration records) and #382 (the offsets step), and ends with a design PR: the screens, their texts, and the build cut into steps
S1.2 read the design PR with its snapshots or mock-ups; merge it: that is the decision on the UI
S1.3 `/develop 431` for the plan's first step (the plan's model) → `/review <pr>` (Sonnet; it runs `/ui-check`) → merge; the following steps the same way, one at a time

**S2 — the scan's data (#430), then the 40\N{MULTIPLICATION SIGN} across wells (#416)** — the `scan` plugin and m3 §8
S2.1 `/plan 430` (Sonnet; a frame's position in its well, in the record and the TIFF) → merge its design PR → `/develop 430` → `/review` (Sonnet) → merge
S2.2 `/plan 416` (Opus; a 40\N{MULTIPLICATION SIGN} scan of several wells that never lets PFS drive Z across a wall: PFS off and a first lock per well, from what session 3 measured) → read its design PR (a decision), merge it → `/develop 416` → `/review` (Opus) → merge

**S3 — the process (#432), then session 3's small fixes** — skills; then `camera_calibration`, the turret's errors
S3.1 `/plan 432` (Opus; `/lane`) → merge its design PR, if any → `/develop 432` → `/review` (Sonnet) → merge
S3.2 `/feature 408` (Sonnet; camera-calibration names a dark frame, and takes its default step from the profile's pixel size) → `/review` (Sonnet) → merge
S3.3 `/feature 406` (Sonnet; the error for a turret between positions) → `/review` (Sonnet) → merge

**End of the day**: `/point`.

**For you**: the walk of S1.1 · the design PRs of S1.2 and S2.2 · the date of the next visit · close #364, #404 and #414.

**Waiting on the stand**: #342 and #343 part 2 (frames with cells and a scan's run document) · #198 · #349 · #392 · #409, #411 · the objective offsets (#380's tool) · #333. **Backlog**: #221, #233, #358 and the p2 leftovers of the previous points.

**Housekeeping**: `status: in-review` removed from the closed #421, #422; `needs-decision` from #414 (decided and built; a note says what the stand must still confirm). `worktree.py clean` removed 5 worktrees and 7 merged branches.
"""

T0 = "2026-11-11T14:17:47Z"
AFTER = "2026-11-11T15:00:00Z"
BEFORE = "2026-11-11T09:00:00Z"


def _parse(*lines: str) -> dict:
    return lane.parse_point("\n".join(lines))


def _kinds(steps: list) -> list[list[str]]:
    return [[s.kind for s in step.segments] for step in steps]


# --- the grammar ------------------------------------------------------------------


def test_the_afternoon_point_of_2026_11_11_parses_into_three_lanes() -> None:
    lanes = lane.parse_point(POINT)
    assert list(lanes) == ["S1", "S2", "S3"]
    assert [len(lanes[k]) for k in lanes] == [3, 2, 3]
    s1 = lanes["S1"]
    assert [s.id for s in s1] == ["S1.1", "S1.2", "S1.3"]
    # "with you": the owner's, though it names `/plan 431`
    first = s1[0].segments[0]
    assert (first.kind, first.issue, first.model) == ("owner", 431, "opus")
    assert _kinds(s1) == [
        ["owner"],
        ["design-merge"],
        ["develop", "review", "merge"],
    ]
    assert [s.issue for step in s1 for s in step.segments] == [431] * 5
    # `/review <pr>` has no digits: no pr; (the plan's model) is no model
    develop, review, merge = s1[2].segments
    assert (develop.model, review.model, review.pr) == (None, "sonnet", None)
    assert merge.kind == "merge"
    s2 = lanes["S2"]
    assert _kinds(s2) == [
        ["plan", "design-merge", "develop", "review", "merge"],
        ["plan", "design-merge", "develop", "review", "merge"],
    ]
    assert [(s.kind, s.issue, s.model) for s in s2[0].segments] == [
        ("plan", 430, "sonnet"),
        ("design-merge", 430, None),
        ("develop", 430, None),
        ("review", 430, "sonnet"),
        ("merge", 430, None),
    ]
    assert [s.issue for s in s2[1].segments] == [416] * 5
    assert [s.model for s in s2[1].segments] == ["opus", None, None, "opus", None]
    s3 = lanes["S3"]
    assert [s.segments[0].issue for s in s3] == [432, 408, 406]
    assert [s.segments[0].kind for s in s3] == ["plan", "feature", "feature"]
    # the real Point is a Point `/point` could have posted
    assert lane.check_point(POINT) == []


def test_the_example_in_the_point_skill_is_a_point_the_check_accepts() -> None:
    # /point writes lanes in the grammar its own skill shows: the two must not drift
    skill = ROOT / ".claude" / "skills" / "point" / "SKILL.md"
    block = skill.read_text(encoding="utf-8").split("```markdown", 1)[1].split("```")[0]
    example = textwrap.dedent(block)
    lanes = lane.parse_point(example)
    assert list(lanes) == ["S1", "S2", "S3"]
    assert lane.check_point(example) == []
    kinds = [seg.kind for steps in lanes.values() for s in steps for seg in s.segments]
    assert {
        "plan",
        "design-merge",
        "develop",
        "feature",
        "review",
        "merge",
        "owner",
    } <= set(kinds)
    assert lanes["S3"][1].after == ("S1.2", "S2.1")


def test_a_lane_ends_at_the_next_bold_line_and_steps_outside_a_lane_are_ignored() -> (
    None
):
    lanes = _parse(
        "S0.1 `/plan 1` before any lane",
        "**S1 — a lane**",
        "S1.1 `/feature 2` (Sonnet)",
        "",
        "S1.2 `/review` (Sonnet) → merge",
        "**End of the day**: `/point`.",
        "S1.3 `/plan 3` after the lane ended",
        "S2.1 `/plan 4` of a lane that was never opened",
    )
    assert list(lanes) == ["S1"]
    assert [s.id for s in lanes["S1"]] == ["S1.1", "S1.2"]


def test_the_model_is_read_from_the_parenthesis_after_the_command() -> None:
    lanes = _parse(
        "**S1 — models**",
        "S1.1 `/plan 1` (Opus; `/lane`) → `/develop 1` (the plan's model) → `/review` (haiku)",
        "S1.2 `/feature 2` for the thing (Sonnet; it reads the notes (twice) first)",
        "S1.3 `/plan 3` with no parenthesis",
        "S1.4 `/plan 4` (a note that mentions Opus later) and then (Haiku)",
    )
    models = [[s.model for s in step.segments] for step in lanes["S1"]]
    assert models == [
        ["opus", None, "haiku"],
        ["sonnet"],
        [None],
        # the first parenthesis after the command decides: "a note that mentions Opus"
        ["opus"],
    ]


def test_a_segment_without_a_command_takes_the_issue_of_the_line_or_the_lane() -> None:
    lanes = _parse(
        "**S1 — issues**",
        "S1.1 merge before any issue",
        "S1.2 `/plan 21` (Sonnet) → merge its design PR → `/develop 21` → `/plan 22`"
        " → merge its design PR",
        "S1.3 `/review` (Sonnet) → merge",
        "S1.4 `/review 445` (Sonnet) → merge",
    )
    issues = [[s.issue for s in step.segments] for step in lanes["S1"]]
    assert issues == [[None], [21, 21, 21, 22, 22], [22, 22], [22, 22]]
    # digits give the pr of a review; nothing else has one
    assert [[s.pr for s in step.segments] for step in lanes["S1"]][2:] == [
        [None, None],
        [445, None],
    ]


def test_a_session_with_you_is_the_owners() -> None:
    lanes = _parse(
        "**S1 — who**",
        "S1.1 `/plan 5` (Opus), **with you**: a walk",
        "S1.2 `/hardware-session nikon-ti2 before` (Opus)",
        "S1.3 `/adr` a title",
        "S1.4 `/point`",
        "S1.5 something nobody can parse",
        "S1.6 `/plan 6` with your usual model",
        "S1.7 `/develop 7` WITH YOU",
    )
    kinds = [step.segments[0].kind for step in lanes["S1"]]
    assert kinds == ["owner"] * 5 + ["plan", "owner"]


def test_after_names_the_steps_it_waits_for() -> None:
    lanes = _parse(
        "**S1 — after**",
        "S1.1 `/feature 1` (Sonnet)",
        "S1.2 after S1.1: `/feature 2` (Sonnet)",
        "S1.3 After S1.1 and S2.1: `/hardware-session nikon-ti2 before` (Opus)",
        "S1.4 after S1.1, S2.1 and S3.2: `/feature 4`",
    )
    assert [step.after for step in lanes["S1"]] == [
        (),
        ("S1.1",),
        ("S1.1", "S2.1"),
        ("S1.1", "S2.1", "S3.2"),
    ]
    assert lanes["S1"][1].segments[0].kind == "feature"


def test_check_flags_a_plan_without_a_develop_in_its_lane() -> None:
    incomplete = "**S1 — x**\nS1.1 `/plan 5` (Sonnet) → merge its design PR\n"
    (problem,) = lane.check_point(incomplete)
    assert "S1.1" in problem
    assert "#5" in problem
    assert "no develop or feature" in problem
    # a develop of another issue, or of the same issue in another lane, does not count
    other = (
        "**S1 — x**\nS1.1 `/plan 5` (Sonnet) → `/develop 6`\n"
        "**S2 — y**\nS2.1 `/develop 5`\n"
    )
    assert len(lane.check_point(other)) == 1
    # a develop or a feature later in the lane does
    assert lane.check_point("**S1 — x**\nS1.1 `/plan 5` → `/develop 5`\n") == []
    assert lane.check_point("**S1 — x**\nS1.1 `/plan 5`\nS1.2 `/feature 5`\n") == []
    # but not an earlier one
    assert lane.check_point("**S1 — x**\nS1.1 `/develop 5`\nS1.2 `/plan 5`\n")


def test_check_flags_an_after_that_names_no_step() -> None:
    draft = (
        "**S1 — x**\nS1.1 `/feature 1`\nS1.2 after S1.1 and S9.9: `/feature 2`\n"
        "**S2 — y**\nS2.1 after S1.2: `/feature 3`\n"
    )
    (problem,) = lane.check_point(draft)
    assert "S1.2" in problem
    assert "S9.9" in problem


def test_check_flags_a_lane_without_a_step_and_a_segment_without_an_issue() -> None:
    draft = (
        "**S1 — empty**\n"
        "**S2 — no issue**\nS2.1 `/review` (Sonnet) → merge → merge its design PR\n"
    )
    problems = lane.check_point(draft)
    assert problems[0].startswith("S1")
    assert "no step" in problems[0]
    assert len(problems) == 4
    assert all("names no issue" in p for p in problems[1:])
    assert lane.check_point("a Point with no lane at all") == []


# --- the state of a segment ---------------------------------------------------------


def _issue(number: int, comments: tuple = (), state: str = "OPEN") -> dict:
    return {"number": number, "state": state, "labels": [], "comments": list(comments)}


def _plan(at: str, body: str | None = None) -> dict:
    text = body or (
        "**Depends on**: none\n**Risk**: tooling\n**Suggested model**: sonnet\n\n"
        "**Ready for /develop**: yes"
    )
    return {"body": f"## Plan (design session, 2026-11-11)\n\n{text}", "createdAt": at}


def _review(verdict: str, at: str) -> dict:
    return {
        "body": f"## Review (design session, 2026-11-11)\n\n**Verdict: {verdict}**",
        "createdAt": at,
    }


def _addressed(at: str) -> dict:
    return {
        "body": "## Review addressed (develop session, 2026-11-11)\n\n- 1. fixed",
        "createdAt": at,
    }


_CHECKS = {
    "green": [{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}],
    "running": [{"name": "lint", "status": "IN_PROGRESS", "conclusion": ""}],
    "red": [{"name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"}],
    "none": [],
}


def _pr(
    number: int,
    closes: tuple[int, ...] = (),
    state: str = "OPEN",
    checks: str = "green",
    texts: tuple = (),
    at: str = "2026-11-11T16:00:00Z",
) -> dict:
    return {
        "number": number,
        "state": state,
        "isDraft": False,
        "createdAt": at,
        "closingIssuesReferences": [{"number": n} for n in closes],
        "statusCheckRollup": _CHECKS[checks],
        "reviews": [],
        "comments": list(texts),
    }


def _facts(issues=(), prs=(), refs=None, pr_numbers=()):
    return lane.Facts(
        {i["number"]: i for i in issues},
        list(prs),
        dict(refs or {}),
        frozenset(pr_numbers),
    )


def _seg(kind: str, issue: int | None = 5, pr: int | None = None, model=None):
    return lane.Segment(kind, issue, pr, model, f"{kind} {issue}")


def _state(seg, facts) -> tuple[str, str]:
    return lane.segment_state(seg, facts, T0)


def test_a_plan_is_done_only_when_posted_after_the_point() -> None:
    seg = _seg("plan")
    assert _state(seg, _facts([_issue(5)])) == ("session", "plan 5")
    old = _facts([_issue(5, (_plan(BEFORE),))])
    assert _state(seg, old) == ("session", "plan 5")
    new = _facts([_issue(5, (_plan(BEFORE), _plan(AFTER)))])
    assert _state(seg, new)[0] == "done"
    # a deviation is not a plan, whenever it was posted
    deviation = {"body": "## Plan deviation (develop session)\n\nx", "createdAt": AFTER}
    assert _state(seg, _facts([_issue(5, (deviation,))])) == ("session", "plan 5")
    assert _state(seg, _facts([_issue(5, state="CLOSED")]))[0] == "done"


def test_the_design_merge_waits_for_the_pr_on_the_ready_line() -> None:
    # #421's case: Depends on none, and the design PR is on the Ready line
    body = "**Depends on**: none\n**Risk**: tooling\n\n**Ready for /develop**: blocked by PR #433"
    issue = _issue(5, (_plan(AFTER, body),))
    seg = _seg("design-merge")
    waiting = _facts([issue], refs={433: "open"}, pr_numbers=[433])
    assert _state(seg, waiting) == ("wait", "merge #433")
    assert _state(seg, _facts([issue], refs={433: "merged"}))[0] == "done"
    assert _state(seg, _facts([issue], refs={433: "closed"}))[0] == "done"
    # a decision issue on Depends on is named as an issue, not as a merge
    both = "**Depends on**: #434\n\n**Ready for /develop**: blocked by #433"
    issue = _issue(5, (_plan(AFTER, both),))
    facts = _facts([issue], refs={433: "open", 434: "open"}, pr_numbers=[433])
    assert _state(seg, facts) == ("wait", "#434, merge #433")


def test_a_design_merge_needs_a_plan_posted_since_the_point() -> None:
    # a plan older than the Point would let the lane skip the owner's walk
    seg = _seg("design-merge")
    assert _state(seg, _facts([_issue(5)]))[0] == "owner"
    assert _state(seg, _facts([_issue(5, (_plan(BEFORE),))]))[0] == "owner"


def test_a_design_pr_closed_unmerged_stops_for_the_owner() -> None:
    body = "**Depends on**: none\n\n**Ready for /develop**: blocked by PR #433"
    issue = _issue(5, (_plan(AFTER, body),))
    facts = _facts([issue], refs={433: "closed-unmerged"}, pr_numbers=[433])
    assert _state(_seg("design-merge"), facts) == (
        "owner",
        "design PR #433 was closed unmerged: re-plan or decide",
    )
    # a develop does not start on it either
    assert _state(_seg("develop"), facts)[0] == "owner"


def test_develop_and_feature_are_done_by_a_pr_and_develop_needs_a_ready_plan() -> None:
    assert _state(_seg("develop"), _facts([_issue(5)])) == ("owner", "no plan on #5")
    ready = _facts([_issue(5, (_plan(AFTER),))])
    assert _state(_seg("develop"), ready) == ("session", "develop 5")
    assert _state(_seg("feature"), _facts([_issue(5)])) == ("session", "feature 5")
    blocked_body = "**Depends on**: #9\n\n**Ready for /develop**: yes"
    blocked = _facts([_issue(5, (_plan(AFTER, blocked_body),))], refs={9: "open"})
    assert _state(_seg("develop"), blocked) == ("wait", "#9")
    merged = _facts(
        [_issue(5, (_plan(AFTER, blocked_body),))],
        [_pr(12, (5,), state="OPEN")],
        refs={9: "closed"},
    )
    for kind in ("develop", "feature"):
        assert _state(_seg(kind), merged) == ("done", "PR #12")


def test_a_develop_does_not_start_on_an_issue_a_session_is_already_on() -> None:
    # the label is set first and the PR last: an in-progress issue with no PR is a
    # session at work, or one that died; the lane leaves it to the owner
    busy = _issue(5, (_plan(AFTER),))
    busy["labels"] = [{"name": "status: in-progress"}]
    for kind in ("develop", "feature"):
        state, detail = _state(_seg(kind), _facts([busy]))
        assert state == "owner"
        assert "#5 is status: in-progress" in detail
    # once its PR exists, it is done whatever the label says
    assert _state(_seg("develop"), _facts([busy], [_pr(12, (5,))])) == (
        "done",
        "PR #12",
    )


def test_review_waits_while_the_checks_run_and_stops_on_red_or_no_checks() -> None:
    seg = _seg("review")
    for checks, want in (
        ("running", ("wait", "CI of #12")),
        ("red", ("owner", "#12 is red: tests")),
        ("none", ("owner", "#12 has no checks: it may conflict with main")),
    ):
        # the verdict is not looked at before the checks: even a ready PR waits or stops
        texts = (_review("ready to merge", AFTER),)
        facts = _facts([_issue(5)], [_pr(12, (5,), checks=checks, texts=texts)])
        assert _state(seg, facts) == want


def test_review_runs_on_a_green_pr_closing_the_issue() -> None:
    seg = _seg("review")
    green = _facts([_issue(5)], [_pr(12, (5,))])
    assert _state(seg, green) == ("session", "review 12")
    # no PR, or a PR that closes another issue
    assert _state(seg, _facts([_issue(5)]))[0] == "owner"
    assert _state(seg, _facts([_issue(5)], [_pr(12, (6,))]))[0] == "owner"
    # a PR closed without a merge is not the issue's PR; the newest of two is
    closed = _pr(11, (5,), state="CLOSED", at="2026-11-11T17:00:00Z")
    assert _state(seg, _facts([_issue(5)], [closed]))[0] == "owner"
    two = [_pr(10, (5,), at="2026-11-11T15:00:00Z"), _pr(12, (5,)), closed]
    assert _state(seg, _facts([_issue(5)], two)) == ("session", "review 12")
    # digits name the PR directly; a merged PR is done; a ready verdict is done
    by_number = _seg("review", issue=5, pr=12)
    assert _state(by_number, _facts([_issue(5)], [_pr(12, ())])) == (
        "session",
        "review 12",
    )
    missing = _seg("review", issue=5, pr=99)
    assert _state(missing, _facts([_issue(5)], [_pr(12, (5,))])) == (
        "owner",
        "PR #99 is not among the last 100 pull requests",
    )
    assert (
        _state(seg, _facts([_issue(5)], [_pr(12, (5,), state="MERGED")]))[0] == "done"
    )
    ready = _pr(12, (5,), texts=(_review("ready to merge", AFTER),))
    assert _state(seg, _facts([_issue(5)], [ready])) == ("done", "ready to merge")


def test_changes_requested_sends_one_fix_round_then_the_owner() -> None:
    seg = _seg("review")
    first = _review("changes requested", "2026-11-11T16:10:00Z")
    one = _pr(12, (5,), texts=(first,))
    assert _state(seg, _facts([_issue(5)], [one])) == (
        "session",
        "develop 5 fix-round",
    )
    # the second "changes requested", after the first was addressed: the owner
    second = _review("changes requested", "2026-11-11T16:30:00Z")
    both = (first, _addressed("2026-11-11T16:20:00Z"), second)
    twice = _pr(12, (5,), texts=both)
    state, detail = _state(seg, _facts([_issue(5)], [twice]))
    assert state == "owner"
    assert "changes requested twice" in detail


def test_an_addressed_review_is_reviewed_again() -> None:
    texts = (
        _review("changes requested", "2026-11-11T16:10:00Z"),
        _addressed("2026-11-11T16:20:00Z"),
    )
    pr = _pr(12, (5,), texts=texts)
    assert _state(_seg("review"), _facts([_issue(5)], [pr])) == ("session", "review 12")


def test_a_review_without_a_verdict_is_the_owners() -> None:
    odd = {"body": "## Review (design session)\n\nno verdict here", "createdAt": AFTER}
    pr = _pr(12, (5,), texts=(odd,))
    assert _state(_seg("review"), _facts([_issue(5)], [pr]))[0] == "owner"


def test_a_merged_pr_completes_the_merge() -> None:
    seg = _seg("merge")
    merged = _facts([_issue(5)], [_pr(12, (5,), state="MERGED")])
    assert _state(seg, merged) == ("done", "PR #12 merged")
    assert _state(seg, _facts([_issue(5)], [_pr(12, (5,))])) == ("wait", "merge #12")
    assert _state(seg, _facts([_issue(5)]))[0] == "owner"


def test_an_owner_segment_is_never_done_by_itself() -> None:
    seg = lane.Segment("owner", 5, None, None, "`/plan 5` with you: a walk")
    state, detail = _state(seg, _facts([_issue(5, (_plan(AFTER),), state="CLOSED")]))
    assert state == "owner"
    assert detail.startswith("`/plan 5` with you")


# --- a lane ---------------------------------------------------------------------------

LANE = lane.parse_point(
    "**S1 — one issue**\n"
    "S1.1 `/plan 5` (Sonnet) → merge its design PR → `/develop 5` → `/review` (Sonnet)"
    " → merge\n"
)


def _rows(facts, start=None, lanes=LANE, lane_id="S1"):
    rows = lane.lane_status(lanes, lane_id, facts, T0, start)
    return [(seg.kind, state) for _, seg, state, _ in rows]


def test_a_later_done_segment_marks_the_earlier_ones_done() -> None:
    # the owner merged by hand before anything was recorded: nothing is left to run
    merged = _facts([_issue(5)], [_pr(12, (5,), state="MERGED")])
    assert _rows(merged) == [
        ("plan", "done"),
        ("design-merge", "done"),
        ("develop", "done"),
        ("review", "done"),
        ("merge", "done"),
    ]
    # an owner segment before a done one is done too
    owner = lane.parse_point(
        "**S1 — o**\nS1.1 `/plan 5` (Opus), **with you**: a walk\nS1.2 `/develop 5`\n"
    )
    facts = _facts([_issue(5)], [_pr(12, (5,))])
    assert _rows(facts, lanes=owner) == [("owner", "done"), ("develop", "done")]
    # with nothing done, the first segment is the next action
    assert _rows(_facts([_issue(5)]))[0] == ("plan", "session")


def test_from_takes_the_steps_before_it_as_done() -> None:
    two = lane.parse_point(
        "**S1 — two**\nS1.1 `/plan 5` (Opus), **with you**: a walk\n"
        "S1.2 `/develop 5` → `/review` (Sonnet)\n"
    )
    facts = _facts([_issue(5, (_plan(AFTER),))])
    assert _rows(facts, lanes=two) == [
        ("owner", "owner"),
        ("develop", "session"),
        # no PR yet: the review is not a session to start, and the lane stops here
        # only after the develop has run
        ("review", "owner"),
    ]
    rows = lane.lane_status(two, "S1", facts, T0, "S1.2")
    assert rows[0][2:] == ("done", "before --from")
    assert [r[2] for r in rows][1:] == ["session", "owner"]
    with pytest.raises(ValueError, match=r"S1\.9"):
        lane.lane_status(two, "S1", facts, T0, "S1.9")


def test_a_step_waits_for_the_steps_it_follows_in_other_lanes() -> None:
    lanes = lane.parse_point(
        "**S1 — first**\nS1.1 `/feature 5` (Sonnet) → `/review` (Sonnet) → merge\n"
        "**S2 — second**\nS2.1 after S1.1: `/feature 6` (Sonnet)\n"
        "S2.2 after S2.1 and S1.1: `/feature 7`\n"
    )
    pending = _facts([_issue(5), _issue(6), _issue(7)], [_pr(12, (5,))])
    rows = lane.lane_status(lanes, "S2", pending, T0)
    assert rows[0][2:] == ("wait", "after S1.1")
    done = _facts([_issue(5)], [_pr(12, (5,), state="MERGED")])
    rows = lane.lane_status(lanes, "S2", done, T0)
    assert [r[2] for r in rows] == ["session", "wait"]
    assert rows[1][3] == "after S2.1"
    # a step that names no step, or a cycle, is the owner's, not an endless wait
    lonely = lane.parse_point("**S1 — x**\nS1.1 after S9.9: `/feature 5`\n")
    assert lane.lane_status(lonely, "S1", _facts([_issue(5)]), T0)[0][2] == "owner"
    cyclic = lane.parse_point(
        "**S1 — x**\nS1.1 after S2.1: `/feature 5`\n"
        "**S2 — y**\nS2.1 after S1.1: `/feature 6`\n"
    )
    rows = lane.lane_status(cyclic, "S1", _facts([_issue(5), _issue(6)]), T0)
    assert rows[0][2] in {"owner", "wait"}


def _model(seg, skill, facts) -> str:
    return lane.session_model(seg, skill, facts)


def test_the_model_falls_back_to_the_plans_then_sonnet_and_opus_for_a_moves_hardware_review() -> (
    None
):
    plan = _plan(AFTER, "**Risk**: moves-hardware\n**Suggested model**: Opus, only for")
    facts = _facts([_issue(5, (plan,)), _issue(6)])
    # the segment's own model first
    assert _model(_seg("develop", model="haiku"), "develop", facts) == "haiku"
    # else the plan's first word, else Sonnet
    assert _model(_seg("develop"), "develop", facts) == "opus"
    assert _model(_seg("develop", 6), "develop", facts) == "sonnet"
    assert _model(_seg("feature", 6), "feature", facts) == "sonnet"
    assert _model(_seg("plan"), "plan", facts) == "sonnet"
    odd = _plan(AFTER, "**Suggested model**: maybe\n")
    assert _model(_seg("develop"), "develop", _facts([_issue(5, (odd,))])) == "sonnet"
    # a review: Opus for moves-hardware, Sonnet otherwise, never the develop's model
    assert _model(_seg("review"), "review", facts) == "opus"
    assert _model(_seg("review", 6), "review", facts) == "sonnet"
    reads = _plan(AFTER, "**Risk**: reads-hardware\n**Suggested model**: opus\n")
    reads_facts = _facts([_issue(5, (reads,))])
    assert _model(_seg("review"), "review", reads_facts) == "sonnet"
    assert _model(_seg("review", model="haiku"), "review", facts) == "haiku"
    # a fix round is a develop session started from a review segment: the review's
    # own model is not the develop's
    fix = _seg("review", model="haiku")
    assert _model(fix, "develop", facts) == "opus"


# --- the brief, the output ------------------------------------------------------------

REVIEW_BRIEF = (
    "You are a fresh session that `/lane S1` runs for one step. Invoke the Skill tool"
    " with skill `review` and args `445`, and follow it as a top-level session would,"
    " with its section *In a lane*: you run as a subagent in your own worktree"
    " (`isolation: worktree`); EnterWorktree and ExitWorktree are refused there; you"
    " cannot ask the owner anything, so where the skill would ask, stop and say what"
    " you need. End with at most five lines: the URL of what you posted (comment or"
    " PR), the verdict or state, and anything that stops the lane."
)


def test_the_brief_is_built_from_the_skill_and_its_argument_only() -> None:
    # nothing but the skill, its argument, the lane and the fix round goes in
    names = list(inspect.signature(lane.brief).parameters)
    assert names == ["skill", "arg", "lane_id", "fix_round"]
    assert lane.brief("review", 445, "S1", False) == REVIEW_BRIEF
    fix = lane.brief("develop", 5, "S2", True)
    assert "skill `develop` and args `5`" in fix
    assert "this is the fix round" in fix
    assert "this is the fix round" not in lane.brief("develop", 5, "S2", False)
    for text in (REVIEW_BRIEF, fix):
        assert text.isascii()
        assert "In a lane" in text
        assert "  " not in text
        assert "\n" not in text
    # two different sets of facts give the same brief for the same review
    lanes = lane.parse_point("**S1 — r**\nS1.1 `/feature 5` → `/review 445` (Sonnet)\n")
    point_comment = {"createdAt": T0, "id": "IC_1"}
    briefs = []
    for texts in ((), (_review("ready to merge", BEFORE),)):
        pr = _pr(445, (5,), texts=texts[:0])
        facts = _facts([_issue(5)], [pr])
        rows = lane.lane_status(lanes, "S1", facts, T0)
        briefs.append(lane.render("S1", rows, facts, point_comment, 51).lines)
    noisy = _facts([_issue(5, (_plan(AFTER, "**Risk**: moves-hardware"),))], [pr])
    rows = lane.lane_status(lanes, "S1", noisy, T0)
    briefs.append(lane.render("S1", rows, noisy, point_comment, 51).lines)
    wanted = f"BRIEF: {REVIEW_BRIEF}"
    assert [next(x for x in b if x.startswith("BRIEF:")) for b in briefs] == [
        wanted
    ] * 3


def test_the_output_has_a_header_one_line_per_segment_and_one_next_line() -> None:
    lanes = lane.parse_point(
        "**S1 — r**\nS1.1 `/feature 5` (Sonnet) → `/review` (Sonnet) → merge\n"
    )
    point_comment = {"createdAt": T0, "id": "IC_kw"}
    facts = _facts([_issue(5)], [_pr(12, (5,))])
    rows = lane.lane_status(lanes, "S1", facts, T0)
    view = lane.render("S1", rows, facts, point_comment, 51)
    assert view.lines[0] == f"lane S1 | point {T0} (comment IC_kw) | tracking #51"
    assert view.lines[1] == "S1.1 [1] feature (sonnet)  done  PR #12"
    assert view.lines[2] == "S1.1 [2] review (sonnet)  session  review 12"
    assert view.lines[3] == "S1.1 [3] merge  wait  merge #12"
    assert view.lines[4] == "NEXT: session review 12 sonnet"
    assert view.lines[5].startswith("BRIEF: You are a fresh session")
    assert len(view.lines) == 6
    assert view.next_kind == "session"
    assert [x for x in view.lines if x.startswith("NEXT:")] == [view.lines[4]]


@pytest.mark.parametrize(
    ("prs", "want", "kind"),
    [
        ([_pr(12, (5,), checks="running")], "NEXT: wait CI of #12", "wait"),
        ([_pr(12, (5,), state="MERGED")], "NEXT: done", "done"),
        (
            [_pr(12, (5,), checks="red")],
            "NEXT: owner #12 is red: tests",
            "owner",
        ),
    ],
)
def test_the_next_line_is_a_wait_an_owner_step_or_done(prs, want, kind) -> None:
    lanes = lane.parse_point(
        "**S1 — r**\nS1.1 `/feature 5` → `/review` (Sonnet) → merge\n"
    )
    facts = _facts([_issue(5)], prs)
    rows = lane.lane_status(lanes, "S1", facts, T0)
    view = lane.render("S1", rows, facts, {"createdAt": T0, "id": "x"}, 51)
    assert [x for x in view.lines if x.startswith("NEXT:")] == [want]
    assert view.next_kind == kind


def test_an_owner_step_says_where_to_resume() -> None:
    lanes = lane.parse_point(POINT)
    facts = _facts([_issue(431)])
    point_comment = {"createdAt": T0, "id": "IC_1"}
    view = lane.render(
        "S1", lane.lane_status(lanes, "S1", facts, T0), facts, point_comment, 51
    )
    nxt = [x for x in view.lines if x.startswith("NEXT:")]
    assert len(nxt) == 1
    assert nxt[0].startswith("NEXT: owner `/plan 431` (Opus), **with you**")
    # the owner's walk is done by the owner: the lane resumes after it
    assert view.lines[-1] == "RESUME: /lane S1 from S1.2"
    # the Point has em dashes and arrows: the output is ASCII (#42)
    assert all(line.isascii() for line in view.lines)
    # a red PR is fixed by the owner and the same step runs again
    one = lane.parse_point("**S1 — r**\nS1.1 `/feature 5` → `/review` (Sonnet)\n")
    red = _facts([_issue(5)], [_pr(12, (5,), checks="red")])
    rows = lane.lane_status(one, "S1", red, T0)
    view = lane.render("S1", rows, red, point_comment, 51)
    assert view.lines[-1] == "RESUME: /lane S1 from S1.1"


def test_ascii_only_transliterates_what_a_console_cannot_encode() -> None:
    text = "a \N{EM DASH} b \N{RIGHTWARDS ARROW} c 40\N{MULTIPLICATION SIGN} \N{GREEK SMALL LETTER MU}m"
    assert lane.ascii_only(text) == "a - b -> c 40x ?m"


# --- the wait and the command line -----------------------------------------------------


def _view(kind: str, point_id: str = "IC_1") -> Any:
    return lane.View([f"NEXT: {kind} x"], kind, point_id)


def _clock(monkeypatch) -> list[float]:
    slept: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(lane.time, "sleep", sleep)
    monkeypatch.setattr(lane.time, "monotonic", lambda: now[0])
    return slept


def test_wait_returns_when_the_next_action_is_no_longer_a_wait(
    monkeypatch, capsys
) -> None:
    views = iter([_view("wait"), _view("wait"), _view("session")])
    monkeypatch.setattr(lane, "collect", lambda lane_id, start: next(views))
    slept = _clock(monkeypatch)
    assert lane.run_lane("S1", None, True, 120, 28800) == 0
    assert slept == [120, 120]
    assert capsys.readouterr().out == "NEXT: session x\n"


def test_without_wait_a_wait_is_printed_once(monkeypatch, capsys) -> None:
    monkeypatch.setattr(lane, "collect", lambda lane_id, start: _view("wait"))
    slept = _clock(monkeypatch)
    assert lane.run_lane("S1", None, False, 120, 28800) == 0
    assert slept == []
    assert capsys.readouterr().out == "NEXT: wait x\n"


def test_wait_gives_up_at_the_timeout_with_exit_3(monkeypatch, capsys) -> None:
    monkeypatch.setattr(lane, "collect", lambda lane_id, start: _view("wait"))
    slept = _clock(monkeypatch)
    assert lane.run_lane("S1", None, True, 100, 250) == 3
    assert slept == [100, 100, 100]
    assert capsys.readouterr().out == "NEXT: wait x\n"


def test_a_new_point_ends_the_wait(monkeypatch, capsys) -> None:
    views = iter([_view("wait", "IC_1"), _view("wait", "IC_2")])
    monkeypatch.setattr(lane, "collect", lambda lane_id, start: next(views))
    slept = _clock(monkeypatch)
    assert lane.run_lane("S1", None, True, 120, 28800) == 0
    assert slept == [120]
    assert capsys.readouterr().out == "NEXT: wait x\n"


def test_check_cli_exits_1_and_names_the_problem(tmp_path, capsys) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(
        "**S1 — x**\nS1.1 `/plan 5` (Sonnet) → merge its design PR\n", encoding="utf-8"
    )
    assert lane.main(["--check", str(draft)]) == 1
    out = capsys.readouterr().out
    assert "S1.1" in out
    assert "#5" in out
    draft.write_text(POINT, encoding="utf-8")
    assert lane.main(["--check", str(draft)]) == 0
    assert capsys.readouterr().out == "ok\n"
    # a file that cannot be read is a usage error, not the verdict "problems found"
    assert lane.main(["--check", str(tmp_path / "missing.md")]) == 4


def test_the_exit_codes_tell_a_gh_failure_an_unknown_lane_and_a_crash_apart(
    monkeypatch, capsys
) -> None:
    def failing(lane_id, start):
        raise lane.GhError("gh pr list failed (1): nope")

    monkeypatch.setattr(lane, "collect", failing)
    assert lane.main(["S1"]) == 2
    assert "nope" in capsys.readouterr().err

    def unknown(lane_id, start):
        raise lane.NoLaneError("the point has no lane S9: S1, S2")

    monkeypatch.setattr(lane, "collect", unknown)
    assert lane.main(["S9"]) == 1
    assert "no lane S9" in capsys.readouterr().err

    def crash(lane_id, start):
        raise KeyError("boom")

    monkeypatch.setattr(lane, "collect", crash)
    monkeypatch.setattr(sys, "argv", ["lane.py", "S1"])
    assert lane._run() == 4
    assert "KeyError" in capsys.readouterr().err
    # usage errors are 4 as well: 2 means `gh` failed
    with pytest.raises(SystemExit) as caught:
        lane.main([])
    assert caught.value.code == 4
    with pytest.raises(SystemExit) as caught:
        lane.main(["S1", "--every", "0"])
    assert caught.value.code == 4


# --- GitHub ---------------------------------------------------------------------------------


def test_every_gh_child_has_no_console_a_timeout_and_utf8(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="[1]", stderr="")

    monkeypatch.setattr(lane.subprocess, "run", fake_run)
    assert lane.gh_json("issue", "list") == [1]
    assert seen["cmd"] == ["gh", "issue", "list"]
    assert seen["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
    assert seen["timeout"] == lane.GH_TIMEOUT_S
    assert seen["stdin"] == subprocess.DEVNULL


def test_a_gh_failure_carries_its_stderr(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 502\n")

    monkeypatch.setattr(lane.subprocess, "run", fake_run)
    with pytest.raises(lane.GhError, match="HTTP 502"):
        lane.gh("pr", "list")

    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(lane.subprocess, "run", hang)
    with pytest.raises(lane.GhError, match="timed out"):
        lane.gh("pr", "list")

    def missing(cmd, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(lane.subprocess, "run", missing)
    with pytest.raises(lane.GhError, match="not installed"):
        lane.gh("pr", "list")


def test_a_ref_is_open_closed_merged_or_closed_unmerged(monkeypatch) -> None:
    data = {
        1: {"state": "open"},
        2: {"state": "closed"},
        3: {"state": "open", "pull_request": {"merged_at": None}},
        4: {"state": "closed", "pull_request": {"merged_at": "2026-11-11T10:00:00Z"}},
        5: {"state": "closed", "pull_request": {"merged_at": None}},
    }
    monkeypatch.setattr(lane, "gh_json", lambda *a: data[int(a[1].rsplit("/", 1)[1])])
    got = {n: lane.ref_state("o/r", n) for n in data}
    assert got == {
        1: ("open", False),
        2: ("closed", False),
        3: ("open", True),
        4: ("merged", True),
        5: ("closed-unmerged", True),
    }


def test_the_facts_are_one_listing_of_prs_the_named_issues_and_the_plans_refs(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    body = "**Depends on**: #9\n\n**Ready for /develop**: blocked by PR #433"
    plan = _plan(AFTER, body)

    def fake(*args: str) -> Any:
        calls.append(args)
        if args[:2] == ("issue", "view"):
            return _issue(int(args[2]), (plan,))
        if args[:2] == ("pr", "list"):
            return [_pr(433, ())]
        if args[0] == "api":
            number = int(args[1].rsplit("/", 1)[1])
            if number == 433:
                return {"state": "open", "pull_request": {"merged_at": None}}
            return {"state": "closed"}
        raise AssertionError(args)

    monkeypatch.setattr(lane, "gh_json", fake)
    lanes = lane.parse_point(
        "**S1 — x**\nS1.1 `/plan 5` → `/develop 5`\n**S2 — y**\nS2.1 `/feature 6`\n"
    )
    facts = lane.fetch_facts("o/r", lanes)
    assert sorted(facts.issues) == [5, 6]
    assert facts.refs == {9: "closed", 433: "open"}
    assert 433 in facts.pr_numbers
    pr_calls = [c for c in calls if c[:2] == ("pr", "list")]
    assert len(pr_calls) == 1
    assert "--state" in pr_calls[0]
    assert pr_calls[0][pr_calls[0].index("--state") + 1] == "all"
    fields = pr_calls[0][pr_calls[0].index("--json") + 1].split(",")
    assert {"closingIssuesReferences", "statusCheckRollup", "reviews"} <= set(fields)
    # each ref is asked once, though two plans name it
    assert sum(1 for c in calls if c[0] == "api") == 2
