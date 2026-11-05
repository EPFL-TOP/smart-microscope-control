"""scripts/ci/nightly_report.py: the nightly's one issue (ADR-0009).

The report opens an issue when the full matrix fails, adds a line when the
next run is red too, and closes it on a green run. Every call to GitHub goes
through an injected ``gh``, so these tests never touch the network.
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

TODAY = "2026-11-05"
RUN_URL = "https://github.com/EPFL-TOP/smart-microscope-control/actions/runs/123"

# The variables the CI sets on every step (FM-84): removed before main() reads them.
_CI_VARIABLES = (
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ID",
    "GITHUB_SERVER_URL",
    "GITHUB_REF_NAME",
    "GITHUB_SHA",
    "GITHUB_ACTIONS",
    "LINT_RESULT",
    "TEST_RESULT",
    "JUNIT_DIR",
)


@pytest.fixture(scope="module")
def rep():
    spec = importlib.util.spec_from_file_location(
        "nightly_report", ROOT / "scripts" / "ci" / "nightly_report.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeGh:
    """Records every call and answers the few the report makes."""

    def __init__(self, issues=None, jobs=None) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.issues = issues if issues is not None else []
        self.jobs = jobs if jobs is not None else []

    def __call__(self, args: list[str], stdin: str | None = None) -> str:
        self.calls.append((list(args), stdin))
        if args[:2] == ["issue", "list"]:
            return json.dumps(self.issues)
        if args[0] == "api":
            return json.dumps({"total_count": len(self.jobs), "jobs": self.jobs})
        if args[:2] == ["issue", "create"]:
            return "https://github.com/EPFL-TOP/smart-microscope-control/issues/999\n"
        return ""

    def verbs(self) -> list[str]:
        """The writes: ``issue create``, ``issue edit`` ... (reads are not listed)."""
        return [
            " ".join(a[:2]) for a, _ in self.calls if a[0] == "issue" and a[1] != "list"
        ]


def _write_junit(directory: Path, artifact: str, xml: str) -> Path:
    folder = directory / artifact
    folder.mkdir(parents=True)
    path = folder / "junit.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _suite(*cases: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        + "".join(cases)
        + "</testsuite></testsuites>"
    )


PASS = '<testcase classname="tests.unit.test_a" name="test_fine" time="0.01"/>'
FAIL = (
    '<testcase classname="tests.unit.test_b" name="test_breaks[case-1]" time="0.02">'
    '<failure message="assert 1 == 2">trace</failure></testcase>'
)
ERROR = (
    '<testcase classname="tests.test_c" name="test_setup_dies" time="0">'
    '<error message="fixture failed">trace</error></testcase>'
)
SKIP = (
    '<testcase classname="tests.test_d" name="test_skipped" time="0">'
    '<skipped type="pytest.skip" message="no mm"/></testcase>'
)


# --- the state of the run -----------------------------------------------


@pytest.mark.parametrize(
    ("lint", "test"),
    [
        ("failure", "success"),
        ("success", "failure"),
        ("failure", "skipped"),
        ("failure", "failure"),
    ],
)
def test_a_failed_job_makes_the_nightly_red(rep, lint, test) -> None:
    assert rep.state(lint, test) == "red"


@pytest.mark.parametrize(
    ("lint", "test"),
    [
        ("cancelled", "skipped"),
        ("success", "cancelled"),
        ("success", "skipped"),
        ("skipped", "skipped"),
        ("", ""),
        ("success", "something-new"),
        (
            "success",
            "abandoned",
        ),  # a matrix entry that never got a runner (measured, #353)
    ],
)
def test_a_cancelled_or_skipped_run_is_neither_red_nor_green(rep, lint, test) -> None:
    # A run cut short says nothing about the code: it must neither open nor close the issue.
    assert rep.state(lint, test) == "unknown"


def test_a_green_run_is_green_only_when_lint_and_tests_both_succeeded(rep) -> None:
    assert rep.state("success", "success") == "green"
    assert rep.state("success", "skipped") != "green"
    assert rep.state("skipped", "success") != "green"
    assert rep.state("cancelled", "success") != "green"


def test_a_cancelled_test_job_does_not_hide_a_red_lint(rep) -> None:
    # Both hold: lint failed and the tests were then cancelled. Red wins over unknown.
    assert rep.state("failure", "cancelled") == "red"


def test_the_nightly_of_main_has_the_plain_title_and_a_branch_its_own(rep) -> None:
    assert rep.issue_title("main") == "CI nightly red"
    assert rep.issue_title("chore/x") == "CI nightly red (chore/x)"
    assert rep.TITLE == "CI nightly red"


def test_the_issue_is_found_by_its_exact_title(rep) -> None:
    issues = [
        {"number": 7, "title": "CI nightly red (chore/x)"},
        {"number": 9, "title": "CI nightly red"},
        {"number": 4, "title": "CI nightly red"},
        {"number": 3, "title": "CI nightly red, again"},
    ]
    assert (
        rep.find_issue(issues, "CI nightly red") == 4
    )  # the first it finds: the oldest
    assert rep.find_issue(issues, "CI nightly red (chore/x)") == 7
    assert rep.find_issue(issues, "CI nightly red (other)") is None
    assert rep.find_issue([], "CI nightly red") is None


# --- what is done to the issue ---------------------------------------------


def test_red_with_no_open_issue_creates_one(rep) -> None:
    gh = FakeGh()
    rep.act("red", "CI nightly red", None, "THE BODY", RUN_URL, gh, today=TODAY)
    assert gh.verbs() == ["issue create"]
    args, stdin = next((a, s) for a, s in gh.calls if a[:2] == ["issue", "create"])
    assert args[args.index("--title") + 1] == "CI nightly red"
    assert args[args.index("--label") + 1] == "area: ci"
    assert "--body-file" in args
    assert stdin == "THE BODY"


def test_red_with_an_open_issue_updates_it_and_adds_one_line(rep) -> None:
    gh = FakeGh()
    rep.act("red", "CI nightly red", 42, "NEW BODY", RUN_URL, gh, today=TODAY)
    assert gh.verbs() == ["issue edit", "issue comment"]
    edit, comment = (a for a, _ in gh.calls)
    assert edit[2] == "42"
    assert "--body-file" in edit
    assert gh.calls[0][1] == "NEW BODY"  # the body goes in on stdin
    assert comment[2] == "42"
    text = comment[comment.index("--body") + 1]
    assert text.startswith(f"## CI nightly (workflow, {TODAY})")
    assert "Still red" in text
    assert RUN_URL in text


def test_green_with_an_open_issue_closes_it(rep) -> None:
    gh = FakeGh()
    rep.act("green", "CI nightly red", 42, "ignored", RUN_URL, gh, today=TODAY)
    assert gh.verbs() == ["issue close"]
    args = gh.calls[0][0]
    assert args[2] == "42"
    text = args[args.index("--comment") + 1]
    assert text.startswith(f"## CI nightly (workflow, {TODAY})")
    assert "Green again" in text
    assert RUN_URL in text


def test_green_with_no_open_issue_does_nothing(rep) -> None:
    gh = FakeGh()
    rep.act("green", "CI nightly red", None, "ignored", RUN_URL, gh, today=TODAY)
    assert gh.calls == []


@pytest.mark.parametrize("existing", [None, 42])
def test_an_unknown_state_touches_no_issue(rep, existing) -> None:
    gh = FakeGh()
    rep.act("unknown", "CI nightly red", existing, "ignored", RUN_URL, gh, today=TODAY)
    assert gh.calls == []


# --- the failing tests ----------------------------------------------------


def test_failed_tests_come_from_the_failure_and_error_elements_of_the_junit_files(
    rep, tmp_path
) -> None:
    _write_junit(tmp_path, "junit-ubuntu-latest-py3.13", _suite(PASS, FAIL, SKIP))
    _write_junit(tmp_path, "junit-macos-latest-py3.13", _suite(PASS, ERROR, FAIL))

    found = rep.failed_tests(tmp_path)

    assert found.by_job["ubuntu-latest-py3.13"] == [
        "tests.unit.test_b::test_breaks[case-1]"
    ]
    assert sorted(found.by_job["macos-latest-py3.13"]) == [
        "tests.test_c::test_setup_dies",
        "tests.unit.test_b::test_breaks[case-1]",
    ]
    assert found.unreadable == []
    # a pass and a skip are not failures
    everything = [t for tests in found.by_job.values() for t in tests]
    assert not any("test_fine" in t or "test_skipped" in t for t in everything)


def test_a_green_job_has_no_entry_and_a_missing_directory_is_no_failure(
    rep, tmp_path
) -> None:
    _write_junit(tmp_path, "junit-ubuntu-latest-py3.13", _suite(PASS))
    assert rep.failed_tests(tmp_path).by_job == {}
    assert rep.failed_tests(tmp_path / "never-downloaded").by_job == {}


def test_a_testcase_with_no_classname_is_named_by_its_name(rep, tmp_path) -> None:
    # A collection error has no class: pytest writes the module as the name.
    case = (
        '<testcase classname="" name="tests.test_broken_import" time="0">'
        '<error message="ImportError">trace</error></testcase>'
    )
    _write_junit(tmp_path, "junit-ubuntu-latest-py3.13", _suite(case))
    found = rep.failed_tests(tmp_path)
    assert found.by_job["ubuntu-latest-py3.13"] == ["tests.test_broken_import"]


def test_a_truncated_junit_file_does_not_hide_the_failure(rep, tmp_path) -> None:
    # The job was killed while the file was written: what was read before the cut counts,
    # and the file is named as unreadable so that nobody reads the list as complete.
    cut = (
        _suite(FAIL, PASS)[: -len("</testsuite></testsuites>")] + "<testcase classname="
    )
    path = _write_junit(tmp_path, "junit-windows-latest-py3.14", cut)
    assert path.read_text(encoding="utf-8").endswith("<testcase classname=")

    found = rep.failed_tests(tmp_path)

    assert found.by_job["windows-latest-py3.14"] == [
        "tests.unit.test_b::test_breaks[case-1]"
    ]
    assert found.unreadable == ["windows-latest-py3.14"]


def test_a_syntax_error_after_a_failure_does_not_hide_it_either(rep, tmp_path) -> None:
    # Not a cut but garbage in the middle: the failure recorded before it still counts.
    broken = _suite(FAIL)[: -len("</testsuite></testsuites>")] + "<<<garbage>>>"
    _write_junit(tmp_path, "junit-macos-latest-py3.13", broken)

    found = rep.failed_tests(tmp_path)

    assert found.by_job["macos-latest-py3.13"] == [
        "tests.unit.test_b::test_breaks[case-1]"
    ]
    assert found.unreadable == ["macos-latest-py3.13"]


def test_a_junit_file_that_is_not_xml_is_skipped_with_a_line_saying_so(
    rep, tmp_path
) -> None:
    _write_junit(tmp_path, "junit-ubuntu-latest-py3.13", "not xml at all")
    found = rep.failed_tests(tmp_path)
    assert found.by_job == {}
    assert found.unreadable == ["ubuntu-latest-py3.13"]
    text = rep.body(
        TODAY,
        "main",
        "abcdef1234567",
        RUN_URL,
        [("tests (ubuntu-latest, py3.13)", "failure")],
        found,
    )
    assert "ubuntu-latest-py3.13" in text
    assert "could not be read" in text


def test_a_failed_job_without_a_junit_file_is_named_with_its_conclusion(
    rep, tmp_path
) -> None:
    # FM-179: a macOS process abort leaves no JUnit file; the job's own conclusion still names it.
    _write_junit(tmp_path, "junit-ubuntu-latest-py3.13", _suite(PASS))
    found = rep.failed_tests(tmp_path)
    jobs = [
        ("lint & types", "success"),
        ("tests (ubuntu-latest, py3.13)", "success"),
        ("tests (macos-latest, py3.13)", "failure"),
        ("nightly report", None),
    ]
    text = rep.body(TODAY, "main", "abcdef1234567", RUN_URL, jobs, found)
    assert "`tests (macos-latest, py3.13)`: failure" in text
    assert "no JUnit file" in text
    assert (
        "nightly report" not in text
    )  # the report job is still running: no conclusion yet
    assert "lint & types" not in text  # a job that succeeded is not listed


def test_a_job_that_timed_out_or_was_cancelled_is_listed_too(rep, tmp_path) -> None:
    found = rep.failed_tests(tmp_path)
    jobs = [
        ("tests (windows-latest, py3.11)", "timed_out"),
        ("tests (ubuntu-latest, py3.10)", "cancelled"),
        ("tests (ubuntu-latest, py3.13)", "skipped"),
    ]
    text = rep.body(TODAY, "main", "abcdef1234567", RUN_URL, jobs, found)
    assert "`tests (windows-latest, py3.11)`: timed_out" in text
    assert "`tests (ubuntu-latest, py3.10)`: cancelled" in text
    assert "py3.13" not in text


def test_a_long_list_of_failures_is_cut_and_says_how_many_more(rep, tmp_path) -> None:
    cases = [
        f'<testcase classname="tests.test_many" name="test_{i:03d}" time="0">'
        f'<failure message="x">t</failure></testcase>'
        for i in range(55)
    ]
    _write_junit(tmp_path, "junit-ubuntu-latest-py3.13", _suite(*cases))
    found = rep.failed_tests(tmp_path)
    text = rep.body(
        TODAY,
        "main",
        "abcdef1234567",
        RUN_URL,
        [("tests (ubuntu-latest, py3.13)", "failure")],
        found,
    )
    listed = re.findall(r"tests\.test_many::test_\d{3}", text)
    assert len(listed) == 40
    assert "and 15 more" in text


def test_a_test_that_failed_on_three_systems_is_listed_once_with_its_systems(
    rep, tmp_path
) -> None:
    for artifact in (
        "junit-ubuntu-latest-py3.13",
        "junit-windows-latest-py3.14",
        "junit-macos-latest-py3.13",
    ):
        _write_junit(tmp_path, artifact, _suite(FAIL))
    found = rep.failed_tests(tmp_path)
    text = rep.body(TODAY, "main", "abcdef1234567", RUN_URL, [], found)
    assert text.count("tests.unit.test_b::test_breaks[case-1]") == 1
    for system in (
        "ubuntu-latest-py3.13",
        "windows-latest-py3.14",
        "macos-latest-py3.13",
    ):
        assert system in text


# --- the body ---------------------------------------------------------------


def test_the_body_starts_with_the_header_that_point_reads(rep, tmp_path) -> None:
    found = rep.failed_tests(tmp_path)
    text = rep.body(
        TODAY, "main", "abcdef1234567", RUN_URL, [("lint & types", "failure")], found
    )
    assert text.startswith(f"## CI nightly (workflow, {TODAY})\n")
    assert rep.HEADER.format(date=TODAY) == f"## CI nightly (workflow, {TODAY})"
    assert RUN_URL in text
    assert "`main`" in text
    assert "abcdef1" in text
    assert "abcdef1234567" not in text  # the short sha
    assert "Fixed before the next merge of the code it names (ADR-0009)." in text


def test_the_body_names_the_ref_it_ran_on(rep, tmp_path) -> None:
    found = rep.failed_tests(tmp_path)
    text = rep.body(TODAY, "chore/x", "abcdef1234567", RUN_URL, [], found)
    assert "`chore/x`" in text


# --- main: the report job end to end ------------------------------------------


@pytest.fixture
def ci_env(monkeypatch, tmp_path):
    for name in _CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    junit = tmp_path / "junit"
    monkeypatch.setenv("GITHUB_REPOSITORY", "EPFL-TOP/smart-microscope-control")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")
    monkeypatch.setenv("JUNIT_DIR", str(junit))
    return junit


def test_main_opens_an_issue_for_a_red_run_with_the_failing_test(
    rep, monkeypatch, ci_env
) -> None:
    _write_junit(ci_env, "junit-macos-latest-py3.13", _suite(FAIL))
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "failure")
    gh = FakeGh(
        jobs=[{"name": "tests (macos-latest, py3.13)", "conclusion": "failure"}]
    )

    assert rep.main(gh=gh, today=TODAY) == 0

    assert gh.verbs() == ["issue create"]
    listing = next(a for a, _ in gh.calls if a[:2] == ["issue", "list"])
    assert listing[listing.index("--label") + 1] == "area: ci"
    assert "open" in listing
    api = next(a for a, _ in gh.calls if a[0] == "api")
    assert "repos/EPFL-TOP/smart-microscope-control/actions/runs/123/jobs" in api
    assert "--paginate" in api
    _, body = next((a, s) for a, s in gh.calls if a[:2] == ["issue", "create"])
    assert body is not None
    assert body.startswith(f"## CI nightly (workflow, {TODAY})")
    assert "tests.unit.test_b::test_breaks[case-1]" in body
    assert "/actions/runs/123" in body


def test_main_closes_the_issue_on_a_green_run(rep, monkeypatch, ci_env) -> None:
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "success")
    gh = FakeGh(issues=[{"number": 42, "title": "CI nightly red"}])
    assert rep.main(gh=gh, today=TODAY) == 0
    assert gh.verbs() == ["issue close"]
    # a green run does not read the jobs or the JUnit files
    assert not any(a[0] == "api" for a, _ in gh.calls)


def test_main_leaves_the_issue_alone_for_a_cancelled_run(
    rep, monkeypatch, ci_env
) -> None:
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "cancelled")
    gh = FakeGh(issues=[{"number": 42, "title": "CI nightly red"}])
    assert rep.main(gh=gh, today=TODAY) == 0
    assert gh.verbs() == []


def test_a_matrix_abandoned_by_a_starved_runner_is_red_when_a_job_failed(
    rep, monkeypatch, ci_env
) -> None:
    # Measured on the demo dispatch of #353: four test jobs failed, the macOS job never got a
    # runner (cancelled, no steps), and `needs.test.result` was `abandoned`, not `failure`.
    # The first version read that as unknown and opened no issue.
    _write_junit(ci_env, "junit-ubuntu-latest-py3.13", _suite(FAIL))
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "abandoned")
    gh = FakeGh(
        jobs=[
            {"name": "lint & types", "conclusion": "success"},
            {"name": "tests (ubuntu-latest, py3.13)", "conclusion": "failure"},
            {"name": "tests (windows-latest, py3.14)", "conclusion": "failure"},
            {"name": "tests (macos-latest, py3.13)", "conclusion": "cancelled"},
            {"name": "nightly report", "conclusion": None},
        ]
    )

    assert rep.main(gh=gh, today=TODAY) == 0

    assert gh.verbs() == ["issue create"]
    _, text = next((a, s) for a, s in gh.calls if a[:2] == ["issue", "create"])
    assert text is not None
    assert "`tests (ubuntu-latest, py3.13)`: failure" in text
    assert "`tests (macos-latest, py3.13)`: cancelled" in text
    assert "tests.unit.test_b::test_breaks[case-1]" in text


def test_an_abandoned_matrix_with_no_failed_job_touches_no_issue(
    rep, monkeypatch, ci_env
) -> None:
    # A starved runner alone says nothing about the code: no issue opens, none closes.
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "abandoned")
    jobs = [
        {"name": "tests (ubuntu-latest, py3.13)", "conclusion": "success"},
        {"name": "tests (macos-latest, py3.13)", "conclusion": "cancelled"},
    ]
    gh = FakeGh(issues=[{"number": 42, "title": "CI nightly red"}], jobs=jobs)
    assert rep.main(gh=gh, today=TODAY) == 0
    assert gh.verbs() == []
    gh = FakeGh(jobs=jobs)
    assert rep.main(gh=gh, today=TODAY) == 0
    assert gh.verbs() == []


def test_an_unknown_run_whose_jobs_cannot_be_listed_touches_no_issue(
    rep, monkeypatch, ci_env
) -> None:
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "abandoned")
    gh = FakeGh()
    inner = gh.__call__

    def flaky(args: list[str], stdin: str | None = None) -> str:
        if args[0] == "api":
            raise rep.GhError("HTTP 502")
        return inner(args, stdin)

    assert rep.main(gh=flaky, today=TODAY) == 0
    assert gh.verbs() == []


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_gh_sends_the_text_on_stdin_and_returns_what_gh_printed(
    rep, monkeypatch
) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return _completed(0, "printed")

    monkeypatch.setattr(rep.subprocess, "run", fake_run)

    assert rep.run_gh(["issue", "create", "--body-file", "-"], "THE BODY") == "printed"

    assert seen["cmd"] == ["gh", "issue", "create", "--body-file", "-"]
    # `--body-file -` reads the standard input: without it the issue has an empty body
    assert seen["kwargs"]["input"] == "THE BODY"
    assert seen["kwargs"]["encoding"] == "utf-8"
    assert seen["kwargs"]["text"] is True
    # a hung gh must not hold the report job for ever
    assert 0 < seen["kwargs"]["timeout"] <= 600


def test_run_gh_raises_with_what_gh_said_when_it_fails(rep, monkeypatch) -> None:
    # A failed `issue list` returning "" would read as "no open issue" and open a second one.
    monkeypatch.setattr(
        rep.subprocess, "run", lambda *a, **k: _completed(1, "", "HTTP 403: forbidden")
    )
    with pytest.raises(rep.GhError, match=r"issue list.*HTTP 403"):
        rep.run_gh(["issue", "list"], None)


def test_run_gh_raises_when_gh_is_missing_or_hangs(rep, monkeypatch) -> None:
    def missing(*a, **k):
        raise FileNotFoundError("gh")

    def hangs(*a, **k):
        raise subprocess.TimeoutExpired(cmd="gh", timeout=1)

    monkeypatch.setattr(rep.subprocess, "run", missing)
    with pytest.raises(rep.GhError):
        rep.run_gh(["issue", "list"], None)
    monkeypatch.setattr(rep.subprocess, "run", hangs)
    with pytest.raises(rep.GhError):
        rep.run_gh(["issue", "list"], None)


def test_run_gh_points_gh_at_the_repository_without_a_checkout(
    rep, monkeypatch
) -> None:
    seen: list[dict] = []

    def fake_run(cmd, **kwargs):
        seen.append(kwargs["env"])
        return _completed(0, "")

    monkeypatch.setattr(rep.subprocess, "run", fake_run)
    monkeypatch.delenv("GH_REPO", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "EPFL-TOP/smart-microscope-control")
    rep.run_gh(["issue", "list"], None)
    assert seen[-1]["GH_REPO"] == "EPFL-TOP/smart-microscope-control"
    # an explicit GH_REPO of the caller is kept
    monkeypatch.setenv("GH_REPO", "someone/else")
    rep.run_gh(["issue", "list"], None)
    assert seen[-1]["GH_REPO"] == "someone/else"


def test_the_jobs_of_every_page_are_read(rep) -> None:
    # `gh api --paginate` prints one JSON document per page, one after another.
    page1 = {"total_count": 3, "jobs": [{"name": "a", "conclusion": "success"}]}
    page2 = {"total_count": 3, "jobs": [{"name": "b", "conclusion": "failure"}]}
    page3 = {"total_count": 3, "jobs": [{"name": "c", "conclusion": None}]}

    def gh(args: list[str], stdin: str | None = None) -> str:
        assert args[0] == "api"
        assert "repos/EPFL-TOP/x/actions/runs/9/jobs" in args
        assert "--paginate" in args
        return json.dumps(page1) + json.dumps(page2) + "\n" + json.dumps(page3) + "\n"

    assert rep.list_jobs(gh, "EPFL-TOP/x", "9") == [
        ("a", "success"),
        ("b", "failure"),
        ("c", None),
    ]


def test_the_issue_listing_is_long_enough_for_every_open_ci_issue(
    rep, monkeypatch, ci_env
) -> None:
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "success")
    gh = FakeGh()
    assert rep.main(gh=gh, today=TODAY) == 0
    listing = next(a for a, _ in gh.calls if a[:2] == ["issue", "list"])
    assert int(listing[listing.index("--limit") + 1]) >= 100
    assert listing[listing.index("--state") + 1] == "open"


def test_a_timed_out_job_failed_and_a_cancelled_one_did_not(rep) -> None:
    assert rep.has_failed_job([("tests (a, py1)", "timed_out")])
    assert rep.has_failed_job([("tests (a, py1)", "failure")])
    assert not rep.has_failed_job([("tests (a, py1)", "cancelled")])
    assert not rep.has_failed_job(
        [("tests (a, py1)", "skipped"), ("lint & types", None)]
    )
    # the report's own job is left out even if it ever had a conclusion
    assert not rep.has_failed_job([("nightly report", "failure")])
    assert not rep.has_failed_job([])


def test_main_names_a_branch_in_the_title_of_a_dispatch_on_it(
    rep, monkeypatch, ci_env
) -> None:
    monkeypatch.setenv("GITHUB_REF_NAME", "chore/353-nightly-demo")
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "failure")
    gh = FakeGh(
        issues=[{"number": 5, "title": "CI nightly red"}]
    )  # main's, not this branch's
    assert rep.main(gh=gh, today=TODAY) == 0
    create = next(a for a, _ in gh.calls if a[:2] == ["issue", "create"])
    assert (
        create[create.index("--title") + 1] == "CI nightly red (chore/353-nightly-demo)"
    )


def test_main_fails_visibly_when_github_cannot_be_asked(
    rep, monkeypatch, ci_env, capsys
) -> None:
    monkeypatch.setenv("LINT_RESULT", "failure")
    monkeypatch.setenv("TEST_RESULT", "skipped")

    def broken(args: list[str], stdin: str | None = None) -> str:
        raise rep.GhError("HTTP 403")

    assert rep.main(gh=broken, today=TODAY) == 1
    assert "::error::" in capsys.readouterr().out


def test_main_survives_a_jobs_listing_that_fails_and_still_reports(
    rep, monkeypatch, ci_env
) -> None:
    # The issue matters more than its detail: no job list, but the issue is still opened.
    monkeypatch.setenv("LINT_RESULT", "success")
    monkeypatch.setenv("TEST_RESULT", "failure")
    gh = FakeGh()
    inner = gh.__call__

    def flaky(args: list[str], stdin: str | None = None) -> str:
        if args[0] == "api":
            raise rep.GhError("HTTP 502")
        return inner(args, stdin)

    assert rep.main(gh=flaky, today=TODAY) == 0
    assert gh.verbs() == ["issue create"]
    body = gh.calls[-1][1]
    assert body is not None
    assert "could not be listed" in body


# --- the job in ci.yml ----------------------------------------------------


def test_the_workflow_reports_the_nightly_with_the_permissions_it_needs() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    job = text[text.index("  nightly-report:") :]
    assert "name: nightly report" in job
    assert "needs: [lint, test]" in job
    assert "always()" in job
    assert "github.event_name == 'schedule'" in job
    assert "github.event_name == 'workflow_dispatch'" in job
    assert "issues: write" in job
    assert "actions: read" in job
    assert "scripts/ci/nightly_report.py" in job
    assert (
        "continue-on-error: true" in job
    )  # no artifact is not a reason to skip the report
    assert "needs.lint.result" in job
    assert "needs.test.result" in job
    # no other job may write issues
    assert text.count("issues: write") == 1
