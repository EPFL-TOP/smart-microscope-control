"""scripts/ci/check_closing_refs.py: a PR closes only the issue it implements."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def refs():
    spec = importlib.util.spec_from_file_location(
        "check_closing_refs", ROOT / "scripts" / "ci" / "check_closing_refs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_design_pr_closes_nothing(refs) -> None:
    # #179 (docs/48-…) closed #48 through a quoted "fixes" in its body.
    assert refs.unexpected("docs/48-probe-design", [48]) == [48]
    assert refs.unexpected("adr/8-ui", [8]) == [8]


def test_a_code_pr_closes_its_own_issue_only(refs) -> None:
    assert refs.unexpected("feat/13-autofocus", [13]) == []
    assert refs.unexpected("fix/127-session0", [127, 16]) == [16]


def test_a_branch_without_a_number_closes_nothing(refs) -> None:
    assert refs.unexpected("chore/process-closing-refs", []) == []
    assert refs.unexpected("chore/process-closing-refs", [13, 16]) == [13, 16]


def test_the_number_is_the_whole_leading_number(refs) -> None:
    assert refs.allowed("fix/1310-x") == {1310}
    assert refs.allowed("fix/13x-y") == set()


def test_a_branch_names_its_issue_whatever_its_type(refs) -> None:
    # `allowed` is what a PR may close: a design branch closes nothing, yet it is
    # still the issue's (tracks.pr_issues reads the PR of a plan's design branch).
    assert refs.branch_issue("feat/653-x") == 653
    assert refs.branch_issue("docs/654-x") == 654
    assert refs.branch_issue("adr/606") == 606
    assert refs.branch_issue("main") is None
    assert refs.branch_issue("feat/x") is None
    assert refs.allowed("docs/654-x") == set()


def test_main_exits_1_and_names_the_fix(refs, capsys) -> None:
    assert refs.main(["docs/48-probe", "48"]) == 1
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "Refs #48" in out
    assert refs.main(["feat/48-probe", "#48"]) == 0
    assert refs.main([]) == 2
