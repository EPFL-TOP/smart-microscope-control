"""The two tracks of the repository (ADR-0010): whose is a path, a pull request, an issue, a lane.

A track is a territory with a lead. ``.github/CODEOWNERS`` is the map of the paths, written
from the default: the stand's is every path that no line frees, and a line with no owner is a
path of the analysis track. The issues of the analysis track carry the label
``track: analysis``; the rest are the stand's. This module is the one reader of both, for
``point.py``, ``lane.py`` and the skills; ``tests/test_two_tracks_repository.py`` pins the map
itself.

    python scripts/dev/tracks.py PATH [PATH ...]      # the track of each path
    python scripts/dev/tracks.py --pr N               # the track of a PR and its crossings

``--pr`` prints one header and one ``other track:`` line per path that is not on the track
of the PR's issue (``none`` when there is none): facts only, the verdict is ``/review``'s.
Stdlib only, Python 3.10, ASCII output (#42, FM-20); needs an authenticated ``gh`` for
``--pr``.
"""

from __future__ import annotations

import argparse
import importlib.util
import posixpath
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

STAND = "stand"
ANALYSIS = "analysis"
TRACKS = (STAND, ANALYSIS)
TRACK_LABEL = "track: analysis"
# Each track's pinned issue: its Point is posted there and its lanes leave their notes there.
TRACKING_TITLES = {STAND: "Point: next steps", ANALYSIS: "Point (analysis): next steps"}
LANE_LETTERS = {"S": STAND, "A": ANALYSIS}
# A regex fragment: the id of a lane, `S1` on the stand track and `A1` on the analysis track.
LANE_ID = r"[SA]\d+"
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"

# The regex of a CODEOWNERS line and whether the line names an owner.
Rules = list[tuple[re.Pattern[str], bool]]


def load_script(name: str, path: Path) -> Any:
    """The module at ``path`` loaded under ``name`` (once): a script's readers are not copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The CI's own reader of a branch name: a PR is the issue's by its branch under the same
# grammar that the closing references check enforces, loaded under the name `lane.py` uses so
# that there is one instance of it.
closing_refs = load_script(
    "smc_ci_check_closing_refs", HERE.parent / "ci" / "check_closing_refs.py"
)


# -- the map of the paths (pure) ----------------------------------------------


def codeowners_rules(text: str) -> list[tuple[str, list[str]]]:
    """The lines of CODEOWNERS as (pattern, owners), in file order."""
    rules: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *owners = line.split()
        rules.append((pattern, owners))
    return rules


def pattern_regex(pattern: str) -> re.Pattern[str]:
    """A CODEOWNERS pattern as gitignore reads it, for the forms ADR-0010 uses.

    A leading ``/`` anchors the pattern at the root, ``**/`` is zero or more
    directories, ``*`` does not cross a ``/``, and a pattern owns what is under it.
    A form this reading does not model is refused, so that a future line is never
    approved by a matcher that would read it differently from GitHub: a trailing
    ``*`` is one level on GitHub (``docs/*`` leaves ``docs/x/y.md`` out) and
    recursive here, and a pattern with no leading ``/`` floats under any directory
    here and not always there.
    """
    floating = pattern != "*" and not pattern.startswith("/")
    trailing_star = pattern != "*" and pattern.endswith("*")
    bare_double_star = "**" in pattern.replace("**/", "")
    if floating or trailing_star or bare_double_star:
        raise ValueError(
            f"{pattern!r} is a form this matcher does not model: write the line as"
            " ADR-0010 does, or teach the matcher the form and test it first"
        )
    body = pattern.strip("/")
    parts: list[str] = []
    i = 0
    while i < len(body):
        if body.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif body[i] == "*":
            parts.append("[^/]*")
            i += 1
        else:
            parts.append(re.escape(body[i]))
            i += 1
    prefix = "" if pattern.startswith("/") else "(?:.*/)?"
    return re.compile(f"^{prefix}{''.join(parts)}(?:/.*)?$")


def read_rules(path: Path = CODEOWNERS) -> Rules:
    """The map of CODEOWNERS, compiled; the run ends, naming the file, when it cannot be read.

    A missing file or a line of a form the matcher does not model must never read as "every
    path is the stand's" or as "every path is the analysis track's": either would send a pull
    request to the wrong lead without a word, so the tool stops instead.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from None
    except UnicodeDecodeError as exc:
        # An editor on Windows may save UTF-16 or cp1252: say which file and what to do (FM-80).
        raise SystemExit(
            f"cannot read {path}: not UTF-8 ({exc.reason}); save it as UTF-8"
        ) from None
    if text.startswith("\N{ZERO WIDTH NO-BREAK SPACE}"):
        # A byte order mark would make the first comment line read as a pattern.
        raise SystemExit(
            f"{path} starts with a byte order mark; save it as UTF-8 without one"
        )
    rules: Rules = []
    for pattern, owners in codeowners_rules(text):
        try:
            rules.append((pattern_regex(pattern), bool(owners)))
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from None
    return rules


def path_track(path: str, rules: Rules) -> str:
    r"""The track of a path: the last matching line wins, a line with no owner is the analysis'.

    A path no line matches is the stand's, and so is one whose last line names an owner:
    the default errs on the side that protects the microscopes. A Windows path (a
    backslash) and a path written from the root (a leading ``/``) are read as POSIX: the
    patterns are, and a path that missed them would fall to the stand without a word (FM-24).
    The same file written ``./src/...`` (PowerShell completes with ``.\``), ``src//...`` or with
    a ``..`` segment is one path, so the path is normalised before the lines are tried.
    """
    posix = posixpath.normpath(path.replace("\\", "/").lstrip("/"))
    analysis = False
    for regex, has_owner in rules:
        if regex.match(posix):
            analysis = not has_owner
    return ANALYSIS if analysis else STAND


def paths_track(paths: list[str], rules: Rules, changed: int | None = None) -> str:
    """The track of a set of paths: the analysis track's only when every path is.

    A pull request that touches one stand path needs the owner. ``changed`` is the number of
    files GitHub says the pull request changes: a listing shorter than it is a cut one, and a
    path that was not listed may be a stand's (FM-256), so a cut listing is read as the stand's.
    """
    if not paths or (changed is not None and len(paths) < changed):
        return STAND
    return ANALYSIS if all(path_track(p, rules) == ANALYSIS for p in paths) else STAND


def pr_track(pr: dict[str, Any], rules: Rules) -> str:
    """The track of a pull request in ``gh``'s shape: its ``files`` and ``changedFiles``."""
    paths = [str(f["path"]) for f in pr.get("files") or []]
    changed = pr.get("changedFiles")
    return paths_track(paths, rules, changed if isinstance(changed, int) else None)


def issue_track(issue: dict[str, Any]) -> str:
    """The track of an issue in ``gh``'s shape: the analysis track's only with its label."""
    names = {
        str(label.get("name")) if isinstance(label, dict) else str(label)
        for label in issue.get("labels") or []
    }
    return ANALYSIS if TRACK_LABEL in names else STAND


def other(track: str) -> str:
    """The track that is not ``track``."""
    return ANALYSIS if track == STAND else STAND


def lane_track(lane_id: str) -> str:
    """The track of a lane by its letter; ``ValueError`` for anything else than ``S<k>``, ``A<k>``."""
    if re.fullmatch(LANE_ID, lane_id) is None:
        raise ValueError(
            f"{lane_id} is not a lane: S<k> on the stand track, A<k> on the analysis track"
        )
    return LANE_LETTERS[lane_id[0]]


def pr_issues(pr: dict[str, Any]) -> list[int]:
    """The issues of a PR: what GitHub says it closes, in order, then its branch's number.

    A pull request that says ``Refs #N`` on purpose closes nothing and a design branch
    (``docs/<N>-<slug>``) never does: the branch is the link that is left (#658).
    """
    found = [int(r["number"]) for r in pr.get("closingIssuesReferences") or []]
    from_branch = closing_refs.branch_issue(str(pr.get("headRefName") or ""))
    if from_branch is not None and from_branch not in found:
        found.append(from_branch)
    return found


# -- the command line ---------------------------------------------------------


def pr_lines(
    number: int,
    rules: Rules,
    gh: Callable[..., str],
    gh_json: Callable[..., Any],
) -> list[str]:
    """The header of a pull request and the paths that are not on its issue's track.

    The diff is read whole (``gh pr diff --name-only``), so that no listing is cut. The
    track is the first issue's, which ``/review`` checks the paths against (rule 2 of
    ADR-0010); a pull request with no issue is read by its paths.
    """
    pr = gh_json(
        "pr",
        "view",
        str(number),
        "--json",
        "number,headRefName,closingIssuesReferences",
    )
    paths = [
        p.strip() for p in gh("pr", "diff", str(number), "--name-only").splitlines()
    ]
    paths = [p for p in paths if p]
    issues = pr_issues(pr)
    if issues:
        issue = gh_json("issue", "view", str(issues[0]), "--json", "number,labels")
        track = issue_track(issue)
        head = f"issue #{issues[0]}, {track} track"
    else:
        track = paths_track(paths, rules)
        head = f"no issue found, track by its paths: {track}"
    analysis = sum(1 for p in paths if path_track(p, rules) == ANALYSIS)
    counts = f"{len(paths)} paths: {analysis} analysis, {len(paths) - analysis} stand"
    lines = [f"PR #{number} | {head} | {counts}"]
    crossing = [p for p in paths if path_track(p, rules) != track]
    lines += [f"other track: {p}" for p in crossing] or ["other track: none"]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="paths, as git prints them")
    ap.add_argument(
        "--pr", type=int, metavar="N", help="a pull request of this repository"
    )
    ns = ap.parse_args(argv)
    if bool(ns.paths) == (ns.pr is not None):
        ap.error("give paths, or --pr N, and not both")
    # cmd.exe and PowerShell pass a wildcard on as it is, and bash does when it matches nothing:
    # `src/smc/selection/*.py` would read as one path of the analysis track, with api.py and
    # registry.py, which are the stand's, among the files it stands for.
    wild = [p for p in ns.paths if any(c in p for c in "*?[")]
    if wild:
        ap.error(f"name the files, a wildcard stands for several tracks: {wild[0]}")
    # point.py imports this module, so the reverse import waits for a run (no cycle at import).
    point = load_script("smc_dev_point", HERE / "point.py")
    rules = read_rules()
    if ns.pr is not None:
        lines = pr_lines(ns.pr, rules, point.gh, point.gh_json)
    else:
        lines = [f"{path_track(p, rules):<9} {p}" for p in ns.paths]
    print("\n".join(point.ascii_only(line) for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
