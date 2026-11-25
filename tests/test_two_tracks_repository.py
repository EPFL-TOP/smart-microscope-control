"""The map of the two tracks (ADR-0010, #652): CODEOWNERS, the interface-idea form and the ruleset's payload.

None of them is code that a test of the library reaches, and each fails quietly: a
CODEOWNERS line moved above the default hands every new path to the analysis side, a
renamed ``api.py`` leaves the selection contract to anyone with write access, an extra
line with no owner over ``.github/`` lets the collaborator edit the file that gates
them, GitHub drops a form it cannot parse, and a ruleset whose bypass reads ``always``
lets an administrator skip more than the code-owner rule. The trial on GitHub (the
owner, three throwaway pull requests) shows what GitHub does with the map; these tests
pin what the map is meant to say.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
FORM = ROOT / ".github" / "ISSUE_TEMPLATE" / "interface-idea.yml"
PAYLOAD = ROOT / "scripts" / "github" / "rulesets" / "code-owners.json"

OWNER = ["@clementhelsens"]

# The analysis track's paths (ADR-0010): the only lines of CODEOWNERS with no owner.
# A path joins this set through a pull request the owner approves, and this test
# module is one of the files that pull request edits.
ANALYSIS = {
    "/src/smc/selection/",
    "/src/smc/analysis/",
    "/docs/algorithms/",
    "/tests/**/test_selector_*.py",
    "/tests/analysis/",
}

# Paths that ADR-0010 announces and that do not exist yet: a line for them protects
# nothing today, and it is the analysis side's own, so it errs on the safe side.
NOT_YET = {"/src/smc/analysis/", "/tests/analysis/"}


def _tracks() -> Any:
    """``scripts/dev/tracks.py``, the one reader of CODEOWNERS (``point.py`` and ``lane.py`` share it)."""
    name = "smc_dev_tracks"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "dev" / "tracks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rules() -> list[tuple[str, list[str]]]:
    """The lines of CODEOWNERS as (pattern, owners), in file order."""
    return _tracks().codeowners_rules(CODEOWNERS.read_text(encoding="utf-8"))


def _regex(pattern: str) -> Any:
    """A CODEOWNERS pattern as gitignore reads it: ``tracks.py``'s matcher, whose table this is."""
    return _tracks().pattern_regex(pattern)


def _owners_of(path: str) -> list[str]:
    """Who GitHub asks for a path: the owners of the last matching line, or nobody."""
    owners: list[str] = []
    for pattern, rule_owners in _rules():
        if _regex(pattern).match(path):
            owners = rule_owners
    return owners


@pytest.mark.parametrize(
    ("path", "owners"),
    [
        # The default is the stand: a path nobody listed is the owner's.
        ("src/smc/hardware/bus.py", OWNER),
        ("src/smc/data/loader.py", OWNER),
        ("src/smc/newpkg/x.py", OWNER),
        ("src/smc/ui/app.py", OWNER),
        ("src/smc/plugins/x.py", OWNER),
        ("pyproject.toml", OWNER),
        ("CLAUDE.md", OWNER),
        ("profiles/nikon-ti2.toml", OWNER),
        (".claude/skills/x/SKILL.md", OWNER),
        (".github/CODEOWNERS", OWNER),
        (".github/workflows/ci.yml", OWNER),
        ("docs/adr/0010-two-tracks.md", OWNER),
        ("docs/design/algorithms.md", OWNER),
        ("scripts/dev/point.py", OWNER),
        ("scripts/github/rulesets/code-owners.json", OWNER),
        ("tests/test_selection_api.py", OWNER),
        ("src/smc/selection_extra/x.py", OWNER),
        # The contract between the tracks is given back to the owner.
        ("src/smc/selection/api.py", OWNER),
        ("src/smc/selection/registry.py", OWNER),
        # The analysis track's paths have no owner: CI alone gates them.
        ("src/smc/selection/builtin.py", []),
        ("src/smc/analysis/review.py", []),
        ("docs/algorithms/README.md", []),
        ("tests/analysis/test_review.py", []),
        # ``**/`` is zero or more directories.
        ("tests/test_selector_dense.py", []),
        ("tests/unit/test_selector_dense.py", []),
    ],
)
def test_the_default_is_the_stand_and_the_analysis_paths_have_no_owner(
    path: str, owners: list[str]
) -> None:
    assert _owners_of(path) == owners


def test_the_default_rule_comes_first_so_a_new_path_belongs_to_the_stand() -> None:
    first_pattern, first_owners = _rules()[0]
    assert (first_pattern, first_owners) == ("*", OWNER)


def test_the_ownerless_lines_are_the_analysis_tracks_and_every_other_line_is_the_owners() -> (
    None
):
    """The table above samples paths; this pins the lines, so that an extra line with
    no owner over ``.github/``, ``profiles/`` or ``docs/*.md`` is a red test and not a
    path nobody thought to sample."""
    rules = _rules()
    assert {pattern for pattern, owners in rules if not owners} == ANALYSIS
    for pattern, owners in rules:
        assert owners in ([], OWNER), f"{pattern} names {owners}, not the owner"


@pytest.mark.parametrize(
    "pattern", ["/docs/algorithms/*", "docs/algorithms/", "/docs/**", "/a**b"]
)
def test_the_matcher_refuses_a_pattern_whose_meaning_it_does_not_model(
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match="does not model"):
        _regex(pattern)


def test_codeowners_paths_that_have_an_owner_exist_so_a_rename_cannot_hand_them_over() -> (
    None
):
    """A renamed ``docs/algorithms/`` would put the collaborator's pages back under
    the owner, and a renamed ``api.py`` would hand the contract to the analysis side."""
    for pattern, _owners in _rules():
        if "*" in pattern or pattern in NOT_YET:
            continue
        assert (ROOT / pattern.strip("/")).exists(), (
            f"{pattern} is not in the repository"
        )


def test_codeowners_is_kept_with_unix_line_endings_so_no_owner_reads_with_a_carriage_return() -> (
    None
):
    """``read_text`` folds CRLF and ``strip`` swallows a ``\\r``, so the tests above
    pass on a file whose owners GitHub would read as ``@clementhelsens\\r`` (how it
    reads one is not in its docs). The rule in ``.gitattributes`` makes Git store the
    file with LF whatever the editor and ``core.autocrlf`` do."""
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert ".github/CODEOWNERS text eol=lf" in attributes
    assert b"\r" not in CODEOWNERS.read_bytes()


def _fields(text: str) -> dict[str, str]:
    """The blocks of an issue form's ``body``, by their ``id``."""
    blocks = re.split(r"^  - type: ", text, flags=re.M)[1:]
    fields: dict[str, str] = {}
    for block in blocks:
        found = re.search(r"^    id: (\w+)$", block, flags=re.M)
        if found:
            fields[found.group(1)] = block
    return fields


def test_the_interface_idea_form_asks_for_the_task_and_what_is_missing_and_takes_no_code() -> (
    None
):
    text = FORM.read_text(encoding="utf-8")
    # The keys GitHub reads. The description holds a ``: ``, so it must stay quoted
    # (unquoted, YAML refuses the file and GitHub does not offer the form), and the
    # labels line is what ``test_every_issue_template_names_labels_the_repository_has``
    # skips when it is missing.
    assert re.search(r"^name: Interface idea$", text, flags=re.M)
    assert re.search(r'^description: "[^"]+"$', text, flags=re.M)
    assert re.search(r'^title: "idea: "$', text, flags=re.M)
    assert re.search(r'^labels: \["type: feature"\]$', text, flags=re.M)
    assert re.search(r"^body:$", text, flags=re.M)
    fields = _fields(text)
    assert set(fields) == {"task", "today", "sketch"}
    for required in ("task", "today"):
        assert re.search(r"^\s+required: true$", fields[required], flags=re.M), required
    assert "required: true" not in fields["sketch"], "a sketch must stay optional"
    assert "render:" not in text, "a field that renders code asks for code"
    assert "dropdown" not in text, "an idea needs no choice from a list"


def test_the_ruleset_payload_asks_the_code_owners_and_lets_the_admin_role_bypass_pull_requests_only() -> (
    None
):
    """The whole payload, key by key: an ``exclude`` of the default branch would make
    the ruleset cover nothing and still read back as active."""
    assert json.loads(PAYLOAD.read_text(encoding="utf-8")) == {
        "name": "Code owners",
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
        "rules": [
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "dismiss_stale_reviews_on_push": False,
                    "require_code_owner_review": True,
                    "require_last_push_approval": False,
                    "required_review_thread_resolution": False,
                },
            }
        ],
        "bypass_actors": [
            {
                "actor_id": 5,
                "actor_type": "RepositoryRole",
                "bypass_mode": "pull_request",
            }
        ],
    }
