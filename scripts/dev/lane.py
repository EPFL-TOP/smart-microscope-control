"""The next action of one lane of the Point, for /lane.

A Point (``point.py``'s tracking issue, "Point: next steps") lists up to three
lanes, `S1`..`S3`, one session at a time inside each. This script reads the
newest ``## Point`` comment, derives from GitHub where every step of one lane
stands, and prints what the lane session does next: start a session for a step
(with the exact prompt, ``BRIEF:``), wait for something (a CI run, a merge),
hand over to the owner, or stop. The lane session decides nothing: its
decisions are the ``NEXT:`` line.

    python scripts/dev/lane.py S1 [--from S1.3] [--wait [--every 120] [--timeout 28800]]
    python scripts/dev/lane.py --check <file with a draft Point>

Exit codes: 0 done (or the next action printed); 1 an unknown lane, no Point,
or the problems that ``--check`` found; 2 a ``gh`` failure; 3 ``--wait`` timed
out; 4 a usage error or an unexpected crash (a crash must not read as the
verdict of 1, FM-82).

The grammar of a lane is fixed in ``/point`` (step 7); ``parse_point`` is its
only reader. Stdlib only, Python 3.10, ASCII output (#42): the Point's text is
transliterated before it is printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

HERE = Path(__file__).resolve().parent
GH_TIMEOUT_S = 120
# A child sharing the console can change its code page (FM-22): no `gh` child gets one.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MODELS = ("opus", "sonnet", "haiku")
PR_FIELDS = (
    "number,state,isDraft,headRefName,closingIssuesReferences,reviews,comments,"
    "statusCheckRollup,createdAt"
)


def _load_point() -> Any:
    """``point.py`` loaded from this folder: its readers of a PR are not copied.

    One reading of "ready to merge", "running" or "changes requested" in the
    project: a second copy would drift from the one the owner's point shows.
    """
    name = "smc_dev_point"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "point.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


point = _load_point()


# -- the grammar of a lane (pure) ----------------------------------------------

_ARROW = " \N{RIGHTWARDS ARROW} "
_LANE_HEAD = re.compile(r"^\*\*(S\d+)(?=[\s*])")
_STEP = re.compile(r"^(S\d+)\.(\d+)\s+(.*)$")
_AFTER = re.compile(
    r"^after\s+(S\d+\.\d+(?:\s*(?:,|and)\s*S\d+\.\d+)*)\s*:\s*", re.IGNORECASE
)
_STEP_ID = re.compile(r"S\d+\.\d+")
_PLAN = re.compile(r"`/plan\s+(\d+)`")
_DEVELOP = re.compile(r"`/develop\s+(\d+)`")
_FEATURE = re.compile(r"`/feature\s+(\d+)`")
_REVIEW = re.compile(r"`/review(?:\s+([^`]*))?`")
# A session "with you" is the owner's: a subagent cannot talk to them.
_WITH_YOU = re.compile(r"\bwith you\b", re.IGNORECASE)
_OWNER_COMMAND = re.compile(r"`/(?:hardware-session|adr|point)\b")
_COMMAND = re.compile(r"`/[\w-]+(?:\s[^`]*)?`")
_MODEL = re.compile(r"\b(opus|sonnet|haiku)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    """One session or one merge of a step: ``kind`` is how /lane treats it."""

    kind: str
    issue: int | None
    pr: int | None
    model: str | None
    text: str


@dataclass(frozen=True)
class Step:
    """One line of a lane: ``S1.2``, what it waits for, and its segments in order."""

    id: str
    after: tuple[str, ...]
    segments: tuple[Segment, ...]


def classify(text: str) -> str:
    """The kind of a segment, first match wins; anything unparsed is the owner's.

    The owner check comes first: a `/plan 431` "with you" is a walk the owner
    takes part in, which a subagent cannot do.
    """
    if _WITH_YOU.search(text) or _OWNER_COMMAND.search(text):
        return "owner"
    for kind, rx in (
        ("plan", _PLAN),
        ("develop", _DEVELOP),
        ("feature", _FEATURE),
        ("review", _REVIEW),
    ):
        if rx.search(text):
            return kind
    low = text.lower()
    if "design pr" in low and "merge" in low:
        return "design-merge"
    if low.strip().startswith("merge"):
        return "merge"
    return "owner"


def _own_issue(text: str) -> int | None:
    for rx in (_PLAN, _DEVELOP, _FEATURE):
        m = rx.search(text)
        if m:
            return int(m.group(1))
    return None


def _review_pr(text: str) -> int | None:
    m = _REVIEW.search(text)
    arg = (m.group(1) or "").strip() if m else ""
    return int(arg) if arg.isdigit() else None


def _model_of(text: str) -> str | None:
    """The model in the parenthesis that follows the segment's backticked command."""
    cmd = _COMMAND.search(text)
    if cmd is None:
        return None
    start = text.find("(", cmd.end())
    if start < 0:
        return None
    depth = 0
    end = len(text)
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    found = _MODEL.search(text[start + 1 : end])
    return found.group(1).lower() if found else None


def parse_point(body: str) -> dict[str, list[Step]]:
    """The lanes of a Point: ``{"S1": [Step, ...], ...}`` in the order written.

    A lane starts at a line beginning ``**S<k> ``; its steps are the following
    lines beginning ``S<k>.<n> ``; it ends at the next line beginning ``**``
    that is not a step. A segment's issue is its own command's, else the last
    issue seen before it on the line or in the lane's earlier steps.
    """
    lanes: dict[str, list[Step]] = {}
    current: str | None = None
    issue: int | None = None
    for raw in body.splitlines():
        line = raw.rstrip()
        head = _LANE_HEAD.match(line)
        if head:
            current = head.group(1)
            lanes.setdefault(current, [])
            issue = None
            continue
        if line.startswith("**"):
            current = None
            continue
        m = _STEP.match(line)
        if m is None or current is None or m.group(1) != current:
            continue
        rest = m.group(3)
        after: tuple[str, ...] = ()
        prefix = _AFTER.match(rest)
        if prefix:
            after = tuple(_STEP_ID.findall(prefix.group(1)))
            rest = rest[prefix.end() :]
        segments: list[Segment] = []
        for piece in rest.split(_ARROW):
            text = piece.strip()
            if not text:
                continue
            kind = classify(text)
            own = _own_issue(text)
            if own is not None:
                issue = own
            segments.append(
                Segment(
                    kind=kind,
                    issue=issue,
                    pr=_review_pr(text) if kind == "review" else None,
                    model=_model_of(text),
                    text=text,
                )
            )
        lanes[current].append(
            Step(f"{m.group(1)}.{m.group(2)}", after, tuple(segments))
        )
    return lanes


def check_point(body: str) -> list[str]:
    """One line per problem of a draft Point; [] when its lanes can be driven."""
    lanes = parse_point(body)
    known = {step.id for steps in lanes.values() for step in steps}
    problems: list[str] = []
    for lane, steps in lanes.items():
        if not steps:
            problems.append(f"{lane}: a lane with no step")
        flat = [(step, seg) for step in steps for seg in step.segments]
        for k, (step, seg) in enumerate(flat):
            if seg.kind in ("review", "merge", "design-merge") and seg.issue is None:
                problems.append(
                    f"{step.id}: a {seg.kind} segment names no issue"
                    f" (put a /plan, /develop or /feature of it earlier in the lane):"
                    f" {point.clip(seg.text, 80)}"
                )
            if seg.kind == "plan" and not any(
                later.kind in ("develop", "feature") and later.issue == seg.issue
                for _, later in flat[k + 1 :]
            ):
                problems.append(
                    f"{step.id}: the plan of #{seg.issue} has no develop or feature"
                    f" of #{seg.issue} after it in {lane} (an incomplete lane)"
                )
        for step in steps:
            problems.extend(
                f"{step.id}: after {name} names no step"
                for name in step.after
                if name not in known
            )
    return problems


# -- the facts and the state of a step (pure: the facts are passed in) ----------


@dataclass
class Facts:
    """What GitHub says, fetched once per run; the state functions never fetch.

    ``refs`` maps a ``#N`` on a plan's Depends-on or Ready line to ``open``,
    ``closed``, ``merged`` or ``closed-unmerged``; ``pr_numbers`` says which
    of them are pull requests (an open one is merged, an open issue is closed).
    """

    issues: dict[int, dict[str, Any]]
    prs: list[dict[str, Any]]
    refs: dict[int, str] = field(default_factory=dict)
    pr_numbers: frozenset[int] = frozenset()


_PLAN_HEAD = re.compile(r"## Plan(?:\s*\(|\s*$)")
_READY = re.compile(r"\*\*Ready for /develop\*\*:?\s*(.*)", re.IGNORECASE)
_NUMBER = re.compile(r"#(\d+)")


def latest_plan(issue: dict[str, Any]) -> dict[str, Any] | None:
    """The newest ``## Plan`` comment; a ``## Plan deviation`` is not a plan."""
    hits = []
    for c in issue.get("comments", []):
        body = str(c.get("body", "")).strip()
        first = body.splitlines()[0] if body else ""
        if _PLAN_HEAD.match(first):
            hits.append(c)
    return max(hits, key=lambda c: str(c.get("createdAt", "")), default=None)


def ready_refs(plan: str) -> list[int]:
    """Numbers on the plan's ``**Ready for /develop**`` line: the design PR is there."""
    for line in plan.splitlines():
        m = _READY.search(line)
        if m:
            return [int(n) for n in _NUMBER.findall(m.group(1))]
    return []


def plan_refs(plan: str) -> list[int]:
    """What a plan waits on: its Depends-on refs, then its Ready line's."""
    return list(dict.fromkeys([*point.depends_on(plan), *ready_refs(plan)]))


def _pr_state(pr: dict[str, Any]) -> str:
    return str(pr.get("state", "")).upper()


def issue_pr(facts: Facts, number: int) -> dict[str, Any] | None:
    """The newest PR that closes the issue and was not closed without a merge."""
    hits = [
        p
        for p in facts.prs
        if _pr_state(p) != "CLOSED"
        and number in {int(r["number"]) for r in p.get("closingIssuesReferences") or []}
    ]
    return max(
        hits,
        key=lambda p: (str(p.get("createdAt", "")), int(p["number"])),
        default=None,
    )


def blockers(plan: str, facts: Facts) -> tuple[list[str], list[int]]:
    """``(what is still open, the refs closed without a merge)`` for a plan's refs."""
    waiting: list[str] = []
    dead: list[int] = []
    for ref in plan_refs(plan):
        state = facts.refs.get(ref, "open")
        if state == "closed-unmerged":
            dead.append(ref)
        elif state == "open":
            is_pr = ref in facts.pr_numbers
            waiting.append(f"merge #{ref}" if is_pr else f"#{ref}")
    return waiting, dead


def _blocked(plan: str, facts: Facts) -> tuple[str, str] | None:
    waiting, dead = blockers(plan, facts)
    if dead:
        names = ", ".join(f"#{n}" for n in dead)
        return ("owner", f"design PR {names} was closed unmerged: re-plan or decide")
    if waiting:
        return ("wait", ", ".join(waiting))
    return None


def _review_state(seg: Segment, facts: Facts) -> tuple[str, str]:
    n = seg.issue
    pr = None
    if seg.pr is not None:
        pr = next((p for p in facts.prs if int(p["number"]) == seg.pr), None)
    elif n is not None:
        pr = issue_pr(facts, n)
    if pr is None:
        if seg.pr is not None:
            return ("owner", f"PR #{seg.pr} is not among the last 100 pull requests")
        return ("owner", f"no PR for #{n}")
    num = int(pr["number"])
    state = _pr_state(pr)
    if state == "MERGED":
        return ("done", f"PR #{num} merged")
    if state == "CLOSED":
        return ("owner", f"PR #{num} was closed unmerged")
    ci = point.checks(pr.get("statusCheckRollup") or [])
    if ci == "running":
        return ("wait", f"CI of #{num}")
    if ci.startswith("failing"):
        return ("owner", f"#{num} is red: {ci.removeprefix('failing: ')}")
    if ci == "no checks":
        return ("owner", f"#{num} has no checks: it may conflict with main")
    texts = point.review_texts(pr)
    found = point.verdict(texts)
    if found == "ready to merge":
        return ("done", "ready to merge")
    if found == "no design review yet" or found.endswith("needs a re-review"):
        return ("session", f"review {num}")
    if found == "changes requested":
        rounds = sum(
            1
            for t in texts
            if str(t.get("body", "")).lstrip().startswith("## Review (")
            and "Verdict: changes requested" in str(t.get("body", ""))
        )
        if rounds == 1:
            issue = n
            if issue is None:
                refs = pr.get("closingIssuesReferences") or []
                issue = int(refs[0]["number"]) if refs else num
            return ("session", f"develop {issue} fix-round")
        return (
            "owner",
            f"#{num}: changes requested twice; read the review and decide",
        )
    return ("owner", f"#{num}: {found}")


def segment_state(seg: Segment, facts: Facts, t0: str) -> tuple[str, str]:
    """``(state, detail)``: ``done``, ``session``, ``wait`` or ``owner``.

    ``t0`` is the Point comment's ``createdAt``: a plan counts as this Point's
    only when it was posted after it. ISO strings compare as strings.
    """
    if seg.kind == "owner":
        return ("owner", point.clip(seg.text, 160))
    n = seg.issue
    if n is None:
        return ("owner", f"a {seg.kind} segment names no issue")
    issue = facts.issues.get(n, {})
    plan = latest_plan(issue)
    if seg.kind == "plan":
        if str(issue.get("state", "")).upper() == "CLOSED":
            return ("done", f"#{n} is closed")
        if plan and str(plan["createdAt"]) > t0:
            return ("done", f"plan of #{n} posted since the point")
        return ("session", f"plan {n}")
    if seg.kind == "design-merge":
        if plan is None or str(plan["createdAt"]) <= t0:
            # A plan older than the Point would let a lane skip the owner's walk.
            return ("owner", f"no plan on #{n} since the point: no design PR to read")
        return _blocked(str(plan["body"]), facts) or ("done", "design PR merged")
    if seg.kind in ("develop", "feature"):
        pr = issue_pr(facts, n)
        if pr is not None:
            return ("done", f"PR #{pr['number']}")
        labels = {str(x.get("name")) for x in issue.get("labels") or []}
        if "status: in-progress" in labels:
            # A session is on it, or died on it (the label is set first, the PR last):
            # a second one would stop "not ready" and cost a subagent to say so.
            return (
                "owner",
                f"#{n} is status: in-progress: a session is on it or died;"
                " finish it by hand",
            )
        if seg.kind == "feature":
            return ("session", f"feature {n}")
        if plan is None:
            return ("owner", f"no plan on #{n}")
        return _blocked(str(plan["body"]), facts) or ("session", f"develop {n}")
    if seg.kind == "review":
        return _review_state(seg, facts)
    pr = issue_pr(facts, n)  # merge
    if pr is None:
        return ("owner", f"no PR for #{n}")
    if _pr_state(pr) == "MERGED":
        return ("done", f"PR #{pr['number']} merged")
    return ("wait", f"merge #{pr['number']}")


Row = tuple[Step, Segment, str, str]


def _after_state(
    lanes: dict[str, list[Step]],
    step: Step,
    facts: Facts,
    t0: str,
    visiting: frozenset[str],
    rows: list[Row],
) -> tuple[str, str] | None:
    """``wait after S2.3`` while a step this one follows is unfinished."""
    unfinished: list[str] = []
    for name in step.after:
        lane = name.split(".")[0]
        steps = lanes.get(lane, [])
        if name not in {s.id for s in steps}:
            return ("owner", f"after {name} names no step")
        if lane in visiting:
            done = [r for r in rows if r[0].id == name]
            if not done:
                return ("owner", f"after {name} is not an earlier step: a cycle")
            finished = all(r[2] == "done" for r in done)
        else:
            other = lane_status(lanes, lane, facts, t0, None, visiting)
            finished = all(r[2] == "done" for r in other if r[0].id == name)
        if not finished:
            unfinished.append(name)
    return ("wait", "after " + ", ".join(unfinished)) if unfinished else None


def lane_status(
    lanes: dict[str, list[Step]],
    lane_id: str,
    facts: Facts,
    t0: str,
    start: str | None = None,
    visiting: frozenset[str] = frozenset(),
) -> list[Row]:
    """Every segment of a lane with its state, in order.

    A segment followed by a done segment is done (the owner did it, or it was
    done by hand); the steps before ``start`` are done. A step that follows
    others waits for them before its first segment starts.
    """
    steps = lanes[lane_id]
    ids = [s.id for s in steps]
    if start is not None and start not in ids:
        raise ValueError(f"{start} is not a step of {lane_id}: {', '.join(ids)}")
    cut = ids.index(start) if start is not None else 0
    rows: list[Row] = []
    inside = visiting | {lane_id}
    for i, step in enumerate(steps):
        for j, seg in enumerate(step.segments):
            if i < cut:
                state, detail = "done", "before --from"
            else:
                state, detail = segment_state(seg, facts, t0)
                if j == 0 and state != "done" and step.after:
                    waiting = _after_state(lanes, step, facts, t0, inside, rows)
                    if waiting is not None:
                        state, detail = waiting
            rows.append((step, seg, state, detail))
    later_done = False
    for k in range(len(rows) - 1, -1, -1):
        step, seg, state, _ = rows[k]
        if state == "done":
            later_done = True
        elif later_done:
            rows[k] = (step, seg, "done", "done: a later segment is done")
    return rows


# -- the model, the brief, the output --------------------------------------------


def _field_word(plan: str, name: str) -> str:
    m = re.search(rf"\*\*{re.escape(name)}\*\*:?\s*([A-Za-z-]+)", plan)
    return m.group(1).lower() if m else ""


def session_model(seg: Segment, skill: str, facts: Facts) -> str:
    """The model of a session: the segment's, else the plan's, else Sonnet.

    A review with no model of its own is Opus for a plan whose risk is
    ``moves-hardware`` and Sonnet otherwise. A fix round is a develop session
    started from a review segment: that segment's model is the review's, not
    the develop's.
    """
    own = seg.model if seg.kind == skill else None
    if own:
        return own
    plan = latest_plan(facts.issues.get(seg.issue, {})) if seg.issue else None
    body = str(plan["body"]) if plan else ""
    if skill == "review":
        return "opus" if _field_word(body, "Risk") == "moves-hardware" else "sonnet"
    if skill in ("develop", "feature"):
        word = _field_word(body, "Suggested model")
        if word in MODELS:
            return word
    return "sonnet"


# The whole prompt of a step's subagent. It takes no fact from GitHub and nothing
# a previous subagent said: a review's brief cannot carry the develop's opinion of
# its own work. A test pins the text.
BRIEF = (
    "You are a fresh session that `/lane {lane_id}` runs for one step. Invoke the"
    " Skill tool with skill `{skill}` and args `{arg}`, and follow it as a top-level"
    " session would, with its section *In a lane*: you run as a subagent in your own"
    " worktree (`isolation: worktree`); EnterWorktree and ExitWorktree are refused"
    " there; you cannot ask the owner anything, so where the skill would ask, stop"
    " and say what you need.{fix} End with at most five lines: the URL of what you"
    " posted (comment or PR), the verdict or state, and anything that stops the lane."
)
_FIX_ROUND = " The PR's latest review says changes requested: this is the fix round."


def brief(skill: str, arg: int, lane_id: str, fix_round: bool) -> str:
    """The subagent's whole prompt, from the skill and its argument only."""
    return BRIEF.format(
        lane_id=lane_id, skill=skill, arg=arg, fix=_FIX_ROUND if fix_round else ""
    )


_ASCII = {
    ord("\N{EM DASH}"): "-",
    ord("\N{EN DASH}"): "-",
    ord("\N{RIGHTWARDS ARROW}"): "->",
    ord("\N{MULTIPLICATION SIGN}"): "x",
    ord("\N{MIDDLE DOT}"): "-",
    ord("\N{HORIZONTAL ELLIPSIS}"): "...",
    ord("\N{LEFT SINGLE QUOTATION MARK}"): "'",
    ord("\N{RIGHT SINGLE QUOTATION MARK}"): "'",
    ord("\N{LEFT DOUBLE QUOTATION MARK}"): '"',
    ord("\N{RIGHT DOUBLE QUOTATION MARK}"): '"',
    ord("\N{NO-BREAK SPACE}"): " ",
}


def ascii_only(text: str) -> str:
    """The text with what a Windows console cannot encode transliterated (#42)."""
    return text.translate(_ASCII).encode("ascii", "replace").decode("ascii")


@dataclass
class View:
    """One run's output: the lines to print and the kind of the next action."""

    lines: list[str]
    next_kind: str
    point_id: str


def render(
    lane_id: str,
    rows: list[Row],
    facts: Facts,
    point_comment: dict[str, Any],
    tracking_number: int,
) -> View:
    """The header, one line per segment, then exactly one ``NEXT:`` line."""
    lines = [
        f"lane {lane_id} | point {point_comment['createdAt']}"
        f" (comment {point_comment.get('id', '?')}) | tracking #{tracking_number}"
    ]
    for step, seg, state, detail in rows:
        index = next(k for k, s in enumerate(step.segments, 1) if s is seg)
        label = seg.kind
        if seg.kind in ("plan", "develop", "feature", "review"):
            label += f" ({session_model(seg, seg.kind, facts)})"
        lines.append(f"{step.id} [{index}] {label}  {state}  {detail}")
    next_kind = "done"
    tail = ["NEXT: done"]
    for step, seg, state, detail in rows:
        if state == "done":
            continue
        next_kind = state
        if state == "session":
            skill, arg, *rest = detail.split()
            model = session_model(seg, skill, facts)
            text = brief(skill, int(arg), lane_id, "fix-round" in rest)
            tail = [f"NEXT: session {skill} {arg} {model}", f"BRIEF: {text}"]
        elif state == "wait":
            tail = [f"NEXT: wait {detail}"]
        else:
            tail = [f"NEXT: owner {detail}"]
            target = _resume_step(rows, step, seg)
            if target is not None:
                tail.append(f"RESUME: /lane {lane_id} from {target}")
        break
    lines += tail
    return View(
        [ascii_only(line) for line in lines],
        next_kind,
        str(point_comment.get("id", "")),
    )


def _resume_step(rows: list[Row], step: Step, seg: Segment) -> str | None:
    """The step ``/lane <S> from <step>`` resumes at once the owner has acted.

    An owner segment is done by the owner: the lane resumes after its step (None
    when it was the last). A red PR, a missing plan or a second "changes
    requested" is fixed by the owner and the same step runs again.
    """
    if seg.kind != "owner":
        return step.id
    ids = list(dict.fromkeys(r[0].id for r in rows))
    here = ids.index(step.id)
    return ids[here + 1] if here + 1 < len(ids) else None


# -- GitHub ------------------------------------------------------------------------


class GhError(Exception):
    """A ``gh`` call failed: the run ends with exit code 2 and the stderr."""


class NoLaneError(Exception):
    """No tracking issue or no Point, or a lane the Point does not have: exit 1."""


def gh(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=GH_TIMEOUT_S,
            creationflags=_NO_WINDOW,
        )
    except FileNotFoundError:
        raise GhError("gh is not installed or not on PATH") from None
    except subprocess.TimeoutExpired:
        raise GhError(
            f"gh {' '.join(args[:2])} timed out after {GH_TIMEOUT_S} s"
        ) from None
    if proc.returncode != 0:
        raise GhError(
            f"gh {' '.join(args[:2])} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def gh_json(*args: str) -> Any:
    return json.loads(gh(*args) or "null")


def ref_state(repo: str, number: int) -> tuple[str, bool]:
    """``(state, is_pr)``: open, closed, merged or closed-unmerged."""
    data = gh_json("api", f"repos/{repo}/issues/{number}")
    pull = data.get("pull_request")
    is_open = str(data.get("state")) == "open"
    if pull:
        if pull.get("merged_at"):
            return ("merged", True)
        return ("open" if is_open else "closed-unmerged", True)
    return ("open" if is_open else "closed", False)


def fetch_point() -> tuple[int, dict[str, Any]]:
    """The tracking issue's number and its newest ``## Point`` comment."""
    issues = gh_json(
        "issue", "list", "--state", "open", "--limit", "300", "--json", "number,title"
    )
    tracking = next((i for i in issues if i["title"] == point.TRACKING_TITLE), None)
    if tracking is None:
        raise NoLaneError(f"no open issue titled {point.TRACKING_TITLE!r}")
    full = gh_json(
        "issue", "view", str(tracking["number"]), "--json", "number,comments"
    )
    newest = point.latest(full.get("comments", []), "## Point")
    if newest is None:
        raise NoLaneError(f"#{tracking['number']} has no ## Point comment")
    return int(tracking["number"]), newest


def fetch_facts(repo: str, lanes: dict[str, list[Step]]) -> Facts:
    numbers = sorted(
        {
            seg.issue
            for steps in lanes.values()
            for step in steps
            for seg in step.segments
            if seg.issue is not None
        }
    )
    issues = {
        n: gh_json("issue", "view", str(n), "--json", "number,state,labels,comments")
        for n in numbers
    }
    prs = gh_json("pr", "list", "--state", "all", "--limit", "100", "--json", PR_FIELDS)
    refs: dict[int, str] = {}
    pr_numbers = {int(p["number"]) for p in prs}
    for issue in issues.values():
        plan = latest_plan(issue)
        for ref in plan_refs(str(plan["body"])) if plan else []:
            if ref not in refs:
                refs[ref], is_pr = ref_state(repo, ref)
                if is_pr:
                    pr_numbers.add(ref)
    return Facts(issues, prs, refs, frozenset(pr_numbers))


def collect(lane_id: str, start: str | None) -> View:
    """Everything one run needs from GitHub, then its output."""
    repo = str(gh_json("repo", "view", "--json", "nameWithOwner")["nameWithOwner"])
    tracking, newest = fetch_point()
    lanes = parse_point(str(newest["body"]))
    if lane_id not in lanes:
        raise NoLaneError(
            f"the point has no lane {lane_id}: {', '.join(lanes) or 'none'}"
        )
    facts = fetch_facts(repo, lanes)
    try:
        rows = lane_status(lanes, lane_id, facts, str(newest["createdAt"]), start)
    except ValueError as exc:
        raise NoLaneError(str(exc)) from None
    return render(lane_id, rows, facts, newest, tracking)


# -- the command line ----------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """A usage error ends with 4: 2 is a ``gh`` failure here."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(4, f"{self.prog}: error: {message}\n")


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def run_lane(
    lane_id: str, start: str | None, wait: bool, every: int, timeout: int
) -> int:
    """Print the next action; with ``wait``, re-read while it is a wait.

    A new Point ends the wait too: the lane session must see its comment id
    change and stop, not keep waiting on a Point that was replaced.
    """
    deadline = time.monotonic() + timeout
    first: str | None = None
    while True:
        view = collect(lane_id, start)
        if first is None:
            first = view.point_id
        if not wait or view.next_kind != "wait" or view.point_id != first:
            print("\n".join(view.lines))
            return 0
        if time.monotonic() >= deadline:
            print("\n".join(view.lines))
            return 3
        time.sleep(every)


def main(argv: list[str] | None = None) -> int:
    ap = _Parser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("lane", nargs="?", help="the lane to run, e.g. S1")
    ap.add_argument("--check", metavar="FILE", help="check a draft Point and exit")
    ap.add_argument(
        "--from", dest="start", metavar="S<k>.<n>", help="steps before it are done"
    )
    ap.add_argument(
        "--wait", action="store_true", help="re-read while the next action is a wait"
    )
    ap.add_argument(
        "--every", type=_positive, default=120, help="seconds between two reads"
    )
    ap.add_argument(
        "--timeout", type=_positive, default=28800, help="seconds before giving up"
    )
    ns = ap.parse_args(argv)

    if ns.check:
        try:
            text = Path(ns.check).read_text(encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"cannot read {ns.check}: {exc}\n")
            return 4
        problems = check_point(text)
        print("\n".join(ascii_only(p) for p in problems) if problems else "ok")
        return 1 if problems else 0
    if not ns.lane:
        ap.error("a lane (S1, S2, ...) or --check FILE is required")
    try:
        return run_lane(ns.lane, ns.start, ns.wait, ns.every, ns.timeout)
    except GhError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except NoLaneError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


def _run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # a crash is not the verdict of exit 1 (FM-82)
        sys.stderr.write(f"unexpected {type(exc).__name__}: {exc}\n")
        return 4


if __name__ == "__main__":
    sys.exit(_run())
