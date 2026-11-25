"""The map of the two tracks (ADR-0010, #652): CODEOWNERS, the interface-idea form and the ruleset's payload.

None of them is code that a test of the library reaches, and each fails quietly: a
CODEOWNERS line moved above the default hands every new path to the analysis side, a
renamed ``api.py`` leaves the selection contract to anyone with write access, GitHub
drops a form field it cannot parse, and a ruleset whose bypass reads ``always`` lets an
administrator skip more than the code-owner rule. The trial at the stand's side of
GitHub (the owner, three throwaway pull requests) shows what GitHub does with the map;
these tests pin what the map is meant to say.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
FORM = ROOT / ".github" / "ISSUE_TEMPLATE" / "interface-idea.yml"
PAYLOAD = ROOT / "scripts" / "github" / "rulesets" / "code-owners.json"

OWNER = ["@clementhelsens"]

# Paths that ADR-0010 announces and that do not exist yet: a line for them protects
# nothing today, and it is the analysis side's own, so it errs on the safe side.
NOT_YET = {"/src/smc/analysis/", "/tests/analysis/"}


def _rules() -> list[tuple[str, list[str]]]:
    """The lines of CODEOWNERS as (pattern, owners), in file order."""
    rules: list[tuple[str, list[str]]] = []
    for raw in CODEOWNERS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *owners = line.split()
        rules.append((pattern, owners))
    return rules


def _regex(pattern: str) -> re.Pattern[str]:
    """A CODEOWNERS pattern as gitignore reads it, for the forms ADR-0010 uses.

    A leading ``/`` anchors the pattern at the root, ``**/`` is zero or more
    directories, ``*`` does not cross a ``/``, and a pattern owns what is under it.
    """
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
        ("pyproject.toml", OWNER),
        (".claude/skills/x/SKILL.md", OWNER),
        ("docs/design/algorithms.md", OWNER),
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
    fields = _fields(text)
    for required in ("task", "today"):
        assert required in fields, f"the form has no `{required}` field"
        assert re.search(r"^\s+required: true$", fields[required], flags=re.M), required
    assert "sketch" in fields
    assert "required: true" not in fields["sketch"], "a sketch must stay optional"
    assert "render:" not in text, "a field that renders code asks for code"
    assert "dropdown" not in text, "an idea needs no choice from a list"


def test_the_ruleset_payload_asks_the_code_owners_and_lets_the_admin_role_bypass_pull_requests_only() -> (
    None
):
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    assert payload["name"] == "Code owners"
    assert payload["enforcement"] == "active"
    assert payload["conditions"]["ref_name"]["include"] == ["~DEFAULT_BRANCH"]
    (rule,) = payload["rules"]
    assert rule["type"] == "pull_request"
    assert rule["parameters"] == {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": False,
        "require_code_owner_review": True,
        "require_last_push_approval": False,
        "required_review_thread_resolution": False,
    }
    assert payload["bypass_actors"] == [
        {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "pull_request"}
    ]
