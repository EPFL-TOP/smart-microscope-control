"""The backlog files under scripts/github/ must stay consistent.

They are created on GitHub by ``scripts/github/bootstrap.py``; a typo in a
label or milestone name would fail halfway through a bootstrap run.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GH = ROOT / "scripts" / "github"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap", GH / "bootstrap.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[cls.__module__];
    # a module loaded from a path must be registered there first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bootstrap():
    return _load_bootstrap()


def test_every_issue_parses_and_uses_known_labels_and_milestones(bootstrap) -> None:
    labels = {
        lab["name"]
        for lab in json.loads((GH / "labels.json").read_text(encoding="utf-8"))
    }
    milestones = {
        ms["title"]
        for ms in json.loads((GH / "milestones.json").read_text(encoding="utf-8"))
    }
    files = sorted((GH / "issues").glob("*.md"))
    assert len(files) >= 20
    titles = []
    for path in files:
        issue = bootstrap.parse_issue(path)
        titles.append(issue.title)
        assert issue.labels, f"{path.name}: no labels"
        unknown = set(issue.labels) - labels
        assert not unknown, f"{path.name}: unknown labels {unknown}"
        if issue.milestone:
            assert issue.milestone in milestones, f"{path.name}: {issue.milestone!r}"
        assert "## Goal" in issue.body or "## Decision" in issue.body, path.name
    assert len(set(titles)) == len(titles), "duplicate issue titles"
