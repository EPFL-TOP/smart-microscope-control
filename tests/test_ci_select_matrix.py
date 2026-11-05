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


def test_the_full_label_is_in_the_label_file_that_the_bootstrap_creates(sel) -> None:
    # A label the script waits for but that no repository has can never be put on a PR.
    labels = json.loads(
        (ROOT / "scripts" / "github" / "labels.json").read_text(encoding="utf-8")
    )
    assert sel.FULL_LABEL in {label["name"] for label in labels}


def test_the_workflow_keeps_the_names_of_its_jobs() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    assert re.search(r"^    name: lint & types\s*$", text, re.MULTILINE)
    assert _workflow_template() == "tests (${{ matrix.os }}, py${{ matrix.python }})"


@pytest.mark.parametrize(
    "line",
    [
        # what select_matrix.py reads: without these a PR silently runs the full matrix again
        "PR_NUMBER: ${{ github.event.pull_request.number }}",
        "GH_TOKEN: ${{ github.token }}",
        "pull-requests: read",
        # the test job is the matrix entry it is named for, on its system and its Python
        "runs-on: ${{ matrix.os }}",
        "python-version: ${{ matrix.python }}",
        "fail-fast: false",
        # the triggers: a missing one is a dead nightly, whose first run cannot be seen in a PR
        "branches: [main]",
        '- cron: "17 2 * * *"',
        "workflow_dispatch:",
        # what nightly_report.py reads back (it names a job's artifact from the job's name)
        "name: junit-${{ matrix.os }}-py${{ matrix.python }}",
        "pattern: junit-*",
        "path: junit\n",
        "retention-days: 7",
    ],
)
def test_the_workflow_keeps_the_wiring_its_scripts_read(line: str) -> None:
    text = CI_YML.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert line in text


def test_the_workflow_takes_its_matrix_from_the_select_step() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    assert "matrix: ${{ fromJSON(needs.lint.outputs.matrix) }}" in text
    assert "matrix: ${{ steps.select.outputs.matrix }}" in text
    assert "scripts/ci/select_matrix.py" in text


def _marker_expressions(text: str) -> list[str]:
    """The expressions given to ``-m`` in ``text``: quoted with ' or ", or a bare word."""
    found = re.finditer(
        r"""\s-m\s+(?:(?P<q>["'])(?P<expr>.*?)(?P=q)|(?P<bare>[^\s"']+))""", text
    )
    return [m.group("expr") or m.group("bare") for m in found]


# The marks a hardware test can carry: it is the combination, not the word, that selects it.
_HARDWARE_TESTS = [
    {"hardware"},
    {"hardware", "demo"},
    {"hardware", "wallclock"},
    {"hardware", "demo", "wallclock"},
]


def _selects_a_hardware_test(expression: str) -> bool:
    """Whether pytest's own evaluation of ``-m expression`` keeps any hardware test."""
    from _pytest.mark.expression import Expression

    compiled = Expression.compile(expression)
    return any(
        compiled.evaluate(lambda name, _marks=marks, **kw: name in _marks)
        for marks in _HARDWARE_TESTS
    )


def test_a_marker_expression_that_selects_hardware_is_found() -> None:
    # The positive control of the next test: the reader must see a bad expression and
    # the evaluation must call it bad, or the next test passes by finding nothing (FM-43).
    sample = """
        run: pytest -n auto -m "not wallclock" --junitxml=junit.xml
        run: pytest -m 'wallclock and not hardware'
        run: pytest -m "hardware"
        run: pytest -m "not wallclock or not hardware"
        run: pytest -m wallclock
        run: pytest -m "not hardware and not wallclock"
    """
    found = _marker_expressions(sample)
    assert found == [
        "not wallclock",
        "wallclock and not hardware",
        "hardware",
        "not wallclock or not hardware",
        "wallclock",
        "not hardware and not wallclock",
    ]
    assert [e for e in found if _selects_a_hardware_test(e)] == [
        "not wallclock",
        "hardware",
        "not wallclock or not hardware",
        "wallclock",  # keeps a test that is both hardware and wallclock
    ]


def test_every_marker_expression_of_the_workflow_excludes_hardware() -> None:
    # FM-192: a `-m` on the command line replaces the `-m 'not hardware'` of addopts,
    # and `-m "not wallclock"` alone selects the 71 hardware tests too.
    text = CI_YML.read_text(encoding="utf-8")
    expressions = _marker_expressions(text)
    assert (
        len(expressions) >= 2
    )  # the two parallel-run steps: a reader that finds none is blind
    assert [e for e in expressions if _selects_a_hardware_test(e)] == []


def test_the_wallclock_marker_is_registered() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    markers = text.split("markers = [", 1)[1].split("]", 1)[0]
    assert '"wallclock:' in markers
    assert '"hardware:' in markers  # the neighbour: still there, still registered


def _steps_running(text: str, command: str) -> list[str]:
    """The steps of ``ci.yml`` (as text, comments of the step included) that run ``command``.

    A step starts at a line of six spaces and a dash and ends before the next one or
    before the next job, so that what follows it in the file is not read as its own.
    """
    blocks = re.split(r"(?m)^(?=      - |  \S)", text)
    return [b for b in blocks if b.startswith("      - ") and command in b]


def test_the_workflow_runs_the_suite_in_parallel_unless_the_select_step_says_serial() -> (
    None
):
    text = CI_YML.read_text(encoding="utf-8")
    assert "serial: ${{ steps.select.outputs.serial }}" in text
    # the serial run keeps coverage and the whole suite
    (serial,) = _steps_running(text, "pytest --cov=smc")
    assert "needs.lint.outputs.serial == 'true'" in serial
    assert "-n auto" not in serial
    # the parallel run: not serial, drops coverage, leaves the wall-clock tests and hardware out
    (parallel,) = _steps_running(
        text, 'pytest -n auto -m "not hardware and not wallclock"'
    )
    assert "needs.lint.outputs.serial != 'true'" in parallel
    assert "--cov" not in parallel  # two invocations cannot share one coverage file
    (wallclock,) = _steps_running(text, 'pytest -m "wallclock and not hardware"')
    assert "needs.lint.outputs.serial != 'true'" in wallclock
    # two steps, not one multi-line run: on Windows only the last command's exit code counts
    assert parallel != wallclock


def test_the_wallclock_step_runs_even_when_the_parallel_step_failed() -> None:
    # Both steps must be able to fail the job and both must report: a red parallel step
    # must not hide a wall-clock test that also failed (and the other way round).
    text = CI_YML.read_text(encoding="utf-8")
    (step,) = _steps_running(text, 'pytest -m "wallclock and not hardware"')
    assert "!cancelled()" in step
    assert "continue-on-error" not in step


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
        # a stalled gh would hold `lint & types`, which every required check waits for
        assert 0 < kwargs.get("timeout", 0) <= 600
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


def _outputs(path: Path) -> dict[str, str]:
    """``$GITHUB_OUTPUT`` as the runner reads it: one ``key=value`` per line."""
    pairs = (
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
    )
    return {key: value for key, value in pairs}


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
    assert [line.split("=", 1)[0] for line in raw.splitlines()] == ["matrix", "serial"]
    matrix = json.loads(_outputs(output)["matrix"])
    assert [sel.job_name(e) for e in matrix["include"]] == FIVE_TEST_JOBS_OF_BEFORE
    # a pull request is run in parallel even with the full matrix: only the nightly is serial
    assert _outputs(output)["serial"] == "false"
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
    matrix = json.loads(_outputs(output)["matrix"])
    assert [sel.job_name(e) for e in matrix["include"]] == REQUIRED_TEST_JOBS
    assert _outputs(output)["serial"] == "false"


def test_main_warns_and_runs_the_full_matrix_when_the_labels_cannot_be_read(
    sel, monkeypatch, ci_env, capsys
) -> None:
    output, summary = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("PR_NUMBER", "353")
    monkeypatch.setattr(sel, "pr_labels", lambda repo, number: None)

    assert sel.main() == 0

    assert "::warning::" in capsys.readouterr().out
    matrix = json.loads(_outputs(output)["matrix"])
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
    matrix = json.loads(_outputs(output)["matrix"])
    assert len(matrix["include"]) == 5


@pytest.mark.parametrize(
    ("event", "serial"),
    [
        ("schedule", True),
        ("workflow_dispatch", True),
        ("push", False),
        ("pull_request", False),
    ],
)
def test_only_the_nightly_and_a_dispatch_run_the_suite_serially(
    sel, event, serial
) -> None:
    # The nightly keeps the whole suite in order, with coverage; a PR and a push run in parallel.
    assert sel.is_serial(event) is serial


def test_main_writes_serial_true_for_the_nightly(sel, monkeypatch, ci_env) -> None:
    output, _ = ci_env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert sel.main() == 0
    assert _outputs(output)["serial"] == "true"


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
