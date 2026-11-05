"""scripts/ci/select_matrix.py: which test jobs a CI run starts (ADR-0009).

The job names are the contract with the required checks of ``main``: a name
that differs by one character leaves a required check pending for ever, so
two tests pin them against literals written here, not against the script.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"

# The five test jobs that `main` required before ADR-0009, written out.
FIVE_TEST_JOBS_OF_BEFORE = [
    "tests (ubuntu-latest, py3.10)",
    "tests (ubuntu-latest, py3.13)",
    "tests (windows-latest, py3.11)",
    "tests (windows-latest, py3.14)",
    "tests (macos-latest, py3.13)",
]
# What `main` requires after it, next to `lint & types`.
REQUIRED_TEST_JOBS = [
    "tests (ubuntu-latest, py3.13)",
    "tests (windows-latest, py3.14)",
]

# The variables a CI step has set on every GitHub runner (FM-84): a test that
# reads the environment removes them first, or it runs against the real CI's.
_CI_VARIABLES = (
    "GITHUB_OUTPUT",
    "GITHUB_STEP_SUMMARY",
    "GITHUB_EVENT_NAME",
    "GITHUB_REPOSITORY",
    "GITHUB_ACTIONS",
    "PR_NUMBER",
)


@pytest.fixture(scope="module")
def sel():
    spec = importlib.util.spec_from_file_location(
        "select_matrix", ROOT / "scripts" / "ci" / "select_matrix.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _names(sel, matrix) -> list[str]:
    return [sel.job_name(entry) for entry in matrix["include"]]


def _workflow_template() -> str:
    """The `name:` of the test job in ci.yml, with its `${{ matrix.* }}` holes."""
    text = CI_YML.read_text(encoding="utf-8")
    found = re.search(r"^    name: (tests \(.*\))\s*$", text, re.MULTILINE)
    assert found is not None, "ci.yml has no test job named 'tests (...)'"
    return found.group(1)


def _from_template(template: str, entry: dict[str, str]) -> str:
    return template.replace("${{ matrix.os }}", entry["os"]).replace(
        "${{ matrix.python }}", entry["python"]
    )


def test_a_push_to_main_runs_the_light_set(sel) -> None:
    matrix, reason = sel.select("push", [])
    assert _names(sel, matrix) == REQUIRED_TEST_JOBS
    assert "light" in reason


def test_a_push_to_main_ignores_labels(sel) -> None:
    # A push has no pull request: a label list that happens to be passed is not read.
    matrix, _ = sel.select("push", [sel.FULL_LABEL])
    assert _names(sel, matrix) == REQUIRED_TEST_JOBS


def test_a_pull_request_runs_the_light_set(sel) -> None:
    matrix, reason = sel.select("pull_request", ["type: chore", "status: in-review"])
    assert _names(sel, matrix) == REQUIRED_TEST_JOBS
    assert "light" in reason


def test_the_label_makes_a_pull_request_run_the_full_matrix(sel) -> None:
    matrix, reason = sel.select("pull_request", ["type: chore", sel.FULL_LABEL])
    assert _names(sel, matrix) == FIVE_TEST_JOBS_OF_BEFORE
    assert sel.FULL_LABEL in reason


def test_a_label_that_only_looks_like_the_full_label_does_not_count(sel) -> None:
    matrix, _ = sel.select("pull_request", ["ci: fullish", "ci:full", "area: ci"])
    assert _names(sel, matrix) == REQUIRED_TEST_JOBS


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_the_nightly_and_a_dispatch_run_the_full_matrix(sel, event) -> None:
    matrix, reason = sel.select(event, [])
    assert _names(sel, matrix) == FIVE_TEST_JOBS_OF_BEFORE
    assert event in reason


def test_labels_that_cannot_be_read_run_the_full_matrix(sel) -> None:
    # More testing is the safe direction when the label read fails.
    matrix, reason = sel.select("pull_request", None)
    assert _names(sel, matrix) == FIVE_TEST_JOBS_OF_BEFORE
    assert "could not" in reason


@pytest.mark.parametrize("event", ["merge_group", "release", "issues", ""])
def test_an_event_without_a_rule_is_refused(sel, event) -> None:
    # A new trigger in ci.yml must come with a decision, not a silent default.
    with pytest.raises(ValueError, match="no rule"):
        sel.select(event, [])


def test_the_light_set_is_exactly_the_required_checks(sel) -> None:
    matrix, _ = sel.select("pull_request", [])
    template = _workflow_template()
    built = [_from_template(template, entry) for entry in matrix["include"]]
    assert built == REQUIRED_TEST_JOBS
    # and the script's own name function agrees with the workflow's template
    assert _names(sel, matrix) == built


def test_the_full_matrix_is_the_five_test_jobs_of_before(sel) -> None:
    matrix, _ = sel.select("schedule", [])
    template = _workflow_template()
    built = [_from_template(template, entry) for entry in matrix["include"]]
    assert sorted(built) == sorted(FIVE_TEST_JOBS_OF_BEFORE)
    # the Python versions stay strings: "3.10" as a number would be 3.1
    assert all(isinstance(entry["python"], str) for entry in matrix["include"])


def test_the_light_set_is_a_subset_of_the_full_matrix(sel) -> None:
    light, _ = sel.select("push", [])
    full, _ = sel.select("schedule", [])
    assert set(_names(sel, light)) < set(_names(sel, full))


def test_the_workflow_keeps_the_names_of_its_jobs() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    assert re.search(r"^    name: lint & types\s*$", text, re.MULTILINE)
    assert _workflow_template() == "tests (${{ matrix.os }}, py${{ matrix.python }})"


def test_the_workflow_takes_its_matrix_from_the_select_step() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    assert "matrix: ${{ fromJSON(needs.lint.outputs.matrix) }}" in text
    assert "matrix: ${{ steps.select.outputs.matrix }}" in text
    assert "scripts/ci/select_matrix.py" in text


# --- pr_labels: the one place that talks to GitHub ---------------------------


def _completed(returncode: int, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def test_pr_labels_reads_one_label_per_line(sel, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs.get("encoding") == "utf-8"
        return _completed(0, "type: chore\r\nci: full\n\n")

    monkeypatch.setattr(sel.subprocess, "run", fake_run)
    assert sel.pr_labels("EPFL-TOP/x", 12) == ["type: chore", "ci: full"]
    assert "repos/EPFL-TOP/x/issues/12/labels" in calls[0]
    assert "--paginate" in calls[0]


def test_pr_labels_is_none_when_gh_fails(sel, monkeypatch) -> None:
    monkeypatch.setattr(sel.subprocess, "run", lambda *a, **k: _completed(1))
    assert sel.pr_labels("EPFL-TOP/x", 12) is None


def test_pr_labels_is_none_when_gh_is_missing_or_hangs(sel, monkeypatch) -> None:
    def missing(*a, **k):
        raise FileNotFoundError("gh")

    def hangs(*a, **k):
        raise subprocess.TimeoutExpired(cmd="gh", timeout=1)

    monkeypatch.setattr(sel.subprocess, "run", missing)
    assert sel.pr_labels("EPFL-TOP/x", 12) is None
    monkeypatch.setattr(sel.subprocess, "run", hangs)
    assert sel.pr_labels("EPFL-TOP/x", 12) is None


def test_an_empty_label_list_is_not_a_failed_read(sel, monkeypatch) -> None:
    # A PR with no label at all: gh prints nothing and exits 0. That is [], not None.
    monkeypatch.setattr(sel.subprocess, "run", lambda *a, **k: _completed(0, ""))
    assert sel.pr_labels("EPFL-TOP/x", 12) == []


# --- main: the outputs the workflow reads ---------------------------------


@pytest.fixture
def ci_env(monkeypatch, tmp_path):
    for name in _CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "github_output"
    summary = tmp_path / "step_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_REPOSITORY", "EPFL-TOP/smart-microscope-control")
    return output, summary


def test_main_writes_the_outputs_the_workflow_reads(
    sel, monkeypatch, ci_env, capsys
) -> None:
    output, summary = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("PR_NUMBER", "353")
    seen: list[tuple[str, int]] = []

    def labels(repo: str, number: int) -> list[str]:
        seen.append((repo, number))
        return ["ci: full"]

    monkeypatch.setattr(sel, "pr_labels", labels)

    assert sel.main() == 0

    assert seen == [("EPFL-TOP/smart-microscope-control", 353)]
    raw = output.read_bytes().decode("utf-8")
    assert (
        "\r" not in raw
    )  # the runner reads LF; a text-mode write on Windows would add CR
    lines = raw.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("matrix=")
    matrix = json.loads(lines[0].removeprefix("matrix="))
    assert [sel.job_name(e) for e in matrix["include"]] == FIVE_TEST_JOBS_OF_BEFORE
    text = summary.read_text(encoding="utf-8")
    assert "full" in text
    assert "::warning::" not in capsys.readouterr().out


def test_main_on_a_push_does_not_ask_github_for_labels(
    sel, monkeypatch, ci_env
) -> None:
    output, _ = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")

    def boom(repo: str, number: int) -> list[str]:
        raise AssertionError("a push has no labels to read")

    monkeypatch.setattr(sel, "pr_labels", boom)
    assert sel.main() == 0
    matrix = json.loads(
        output.read_text(encoding="utf-8").removeprefix("matrix=").strip()
    )
    assert [sel.job_name(e) for e in matrix["include"]] == REQUIRED_TEST_JOBS


def test_main_warns_and_runs_the_full_matrix_when_the_labels_cannot_be_read(
    sel, monkeypatch, ci_env, capsys
) -> None:
    output, summary = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("PR_NUMBER", "353")
    monkeypatch.setattr(sel, "pr_labels", lambda repo, number: None)

    assert sel.main() == 0

    assert "::warning::" in capsys.readouterr().out
    matrix = json.loads(
        output.read_text(encoding="utf-8").removeprefix("matrix=").strip()
    )
    assert [sel.job_name(e) for e in matrix["include"]] == FIVE_TEST_JOBS_OF_BEFORE
    assert "could not" in summary.read_text(encoding="utf-8")


def test_main_without_a_pull_request_number_runs_the_full_matrix(
    sel, monkeypatch, ci_env, capsys
) -> None:
    # A pull_request event with no number to ask about is an unreadable label list.
    output, _ = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setattr(
        sel, "pr_labels", lambda repo, number: pytest.fail("no number to read")
    )
    assert sel.main() == 0
    assert "::warning::" in capsys.readouterr().out
    matrix = json.loads(
        output.read_text(encoding="utf-8").removeprefix("matrix=").strip()
    )
    assert len(matrix["include"]) == 5


def test_main_refuses_an_event_without_a_rule(sel, monkeypatch, ci_env, capsys) -> None:
    output, _ = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "merge_group")
    assert sel.main() == 1
    assert "::error::" in capsys.readouterr().out
    assert not output.exists()  # no matrix is better than a guessed one


def test_main_prints_the_matrix_when_it_runs_outside_github(
    sel, monkeypatch, capsys
) -> None:
    for name in _CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert sel.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("matrix=")
    assert "windows-latest" in out
