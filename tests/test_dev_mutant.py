"""scripts/dev/mutant.py runs one mutant in a scratch copy; pin what it promises.

Every test works on a small repository built in ``tmp_path``, never on this
one: the script copies a repository and runs pytest in the copy, and the point
of most tests is what happens to the *original* (it is hashed before and after).
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dev" / "mutant.py"

PYPROJECT = """\
[tool.pytest.ini_options]
addopts = "-ra -m 'not hardware'"
markers = ["hardware: needs a stand"]
testpaths = ["tests"]
"""

CALC = """\
def add(a, b):
    return a + b  # sum


def total(a, b):
    return a + b + 1


def countdown(n):
    while n > 0:
        n -= 1
    return n
"""

TESTS = """\
from pathlib import Path

import pkg
import pytest
from helpers import EXPECTED
from pkg.calc import add, countdown, total


def test_add_sums():
    assert add(2, 3) == 5


def test_add_uses_the_helper():
    assert add(2, 3) == EXPECTED


def test_prints_non_ascii():
    print("caf\\u00e9 \\u00b5m \\u2713")
    assert add(2, 3) == 5


def test_add_runs():
    add(2, 3)


def test_add_or_total_is_intact():
    assert add(2, 3) == 5 or total(2, 3) == 6


def test_countdown_ends():
    assert countdown(3) == 0


def test_imports_the_copy():
    assert Path(pkg.__file__).resolve().is_relative_to(Path.cwd().resolve())


def test_always_fails():
    assert False


@pytest.mark.skip(reason="no adapters")
def test_skipped():
    assert False


@pytest.mark.hardware
def test_hardware():
    assert True
"""

CALC_FILE = "src/pkg/calc.py"
SUM = "return a + b  # sum"
SUM_BROKEN = "return a - b  # sum"
ADD_SUMS = "tests/test_calc.py::test_add_sums"
ADD_RUNS = "tests/test_calc.py::test_add_runs"
INTACT = "tests/test_calc.py::test_add_or_total_is_intact"
TOTAL = "return a + b + 1"
TOTAL_BROKEN = "return a + b + 2"


@pytest.fixture(scope="module")
def mutant():
    spec = importlib.util.spec_from_file_location("mutant_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "pkg" / "calc.py").write_text(CALC, encoding="utf-8")
    (root / "tests" / "test_calc.py").write_text(TESTS, encoding="utf-8")
    (root / "tests" / "helpers.py").write_text("EXPECTED = 5\n", encoding="utf-8")
    return root


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The directory the script's copies go to: empty means nothing was left."""
    directory = tmp_path / "scratch"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


class Result(NamedTuple):
    code: int
    out: str
    err: str


@pytest.fixture
def run(mutant, repo, scratch, capsys):
    def call(file: str, old: str, new: str, test_id: str, *options: str) -> Result:
        argv = [*options, "--root", str(repo), "--", file, old, new, test_id]
        code = mutant.main(argv)
        captured = capsys.readouterr()
        return Result(code, captured.out, captured.err)

    return call


@pytest.fixture
def run_argv(mutant, repo, scratch, capsys):
    """Like ``run`` for a command line it does not build: the caller gives all of it."""

    def call(*argv: str) -> Result:
        code = mutant.main(["--root", str(repo), *argv])
        captured = capsys.readouterr()
        return Result(code, captured.out, captured.err)

    return call


def tree_hash(root: Path) -> str:
    """Every directory and file of ``root``: names, bytes and modification times."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            digest.update(f"D {rel}\n".encode())
        else:
            digest.update(f"F {rel} {path.stat().st_mtime_ns} ".encode())
            digest.update(path.read_bytes() + b"\n")
    return digest.hexdigest()


# --- the verdicts ----------------------------------------------------------


def test_a_mutant_that_a_test_catches_is_killed(run) -> None:
    result = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS)
    assert result.code == 0
    assert result.out.startswith("KILLED")
    assert "1 failed" in result.out


def test_a_mutant_that_a_test_cannot_see_survives(run) -> None:
    result = run(CALC_FILE, SUM, SUM_BROKEN, ADD_RUNS)
    assert result.code == 1
    assert result.out.startswith("SURVIVED")


def test_the_copy_is_imported_not_the_original(run, repo, monkeypatch) -> None:
    # The editable install of this project looks like this: the original's
    # src is already on the path. The mutant must still be the code that runs.
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    result = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS)
    assert result.code == 0, result.out + result.err


def test_the_worktree_is_untouched(run, repo, scratch) -> None:
    before = tree_hash(repo)
    assert run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS).code == 0
    assert run(CALC_FILE, SUM, SUM_BROKEN, ADD_RUNS).code == 1
    assert run(CALC_FILE, "not in the file", "x", ADD_SUMS).code == 2
    assert tree_hash(repo) == before
    assert list(scratch.iterdir()) == []


# --- a pattern that is not found exactly once ------------------------------


def test_a_pattern_found_zero_times_is_an_error_not_a_survivor(run) -> None:
    result = run(CALC_FILE, "return a * b", "return a - b", ADD_RUNS)
    assert result.code == 2
    assert "pattern found 0 times in src/pkg/calc.py" in result.err
    assert "SURVIVED" not in result.out


def test_a_pattern_found_twice_exits_2_and_names_the_lines(run) -> None:
    result = run(CALC_FILE, "a + b", "a - b", ADD_RUNS)
    assert result.code == 2
    assert "pattern found 2 times in src/pkg/calc.py (lines 2, 6)" in result.err


def test_overlapping_matches_count_as_several(mutant) -> None:
    assert mutant.occurrences("aaa", "aa") == [0, 1]
    assert mutant.occurrences("abab", "ab") == [0, 2]
    assert mutant.occurrences("abc", "x") == []


def test_line_numbers_are_one_based(mutant) -> None:
    text = "a\nb\r\nc\n"
    assert mutant.line_numbers(text, [0, 2, 5]) == [1, 2, 3]


def test_old_equal_to_new_is_refused(run) -> None:
    result = run(CALC_FILE, SUM, SUM, ADD_RUNS)
    assert result.code == 2
    assert "old and new are the same" in result.err


def test_an_empty_pattern_is_refused(run) -> None:
    result = run(CALC_FILE, "", "x", ADD_RUNS)
    assert result.code == 2
    assert "old is empty" in result.err


def test_a_crlf_file_keeps_its_line_endings(run, repo, scratch) -> None:
    crlf = repo / "src" / "pkg" / "crlf.py"
    crlf.write_bytes(b"x = 1\r\ny = 2\r\nz = 3\r\n")
    before = crlf.read_bytes()

    result = run("src/pkg/crlf.py", "y = 2", "y = 9", ADD_RUNS, "--keep")

    assert result.code == 1
    (kept,) = scratch.glob("smc-mutant-*/repo/src/pkg/crlf.py")
    assert kept.read_bytes() == b"x = 1\r\ny = 9\r\nz = 3\r\n"
    assert crlf.read_bytes() == before


def test_a_multi_line_pattern_is_matched_as_written_without_newline_translation(
    run, repo
) -> None:
    (repo / "src" / "pkg" / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    lf = run("src/pkg/crlf.py", "x = 1\ny = 2", "x = 1\ny = 9", ADD_RUNS)
    assert lf.code == 2
    assert "pattern found 0 times in src/pkg/crlf.py" in lf.err
    crlf = run("src/pkg/crlf.py", "x = 1\r\ny = 2", "x = 1\r\ny = 9", ADD_RUNS)
    assert crlf.code == 1


def test_a_file_that_is_not_utf8_is_named(run, repo) -> None:
    (repo / "src" / "pkg" / "latin.py").write_bytes(b"x = '\xe9'\n")
    result = run("src/pkg/latin.py", "x", "y", ADD_RUNS)
    assert result.code == 2
    assert "src/pkg/latin.py is not valid UTF-8" in result.err


# --- where the paths may point ---------------------------------------------


def test_an_unknown_file_or_test_path_is_an_error_and_copies_nothing(
    run, scratch
) -> None:
    missing_file = run("src/pkg/nope.py", "a", "b", ADD_RUNS)
    assert missing_file.code == 2
    assert "src/pkg/nope.py is not a file in the repository" in missing_file.err
    missing_test = run(CALC_FILE, SUM, SUM_BROKEN, "tests/test_nope.py::test_x")
    assert missing_test.code == 2
    assert "test path tests/test_nope.py is not in the repository" in missing_test.err
    assert list(scratch.iterdir()) == []


def test_an_absolute_path_outside_the_repository_is_refused(
    run, tmp_path, scratch
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("a = 1\n", encoding="utf-8")
    result = run(str(outside), "a", "b", ADD_RUNS)
    assert result.code == 2
    assert "is outside the repository" in result.err
    assert outside.read_text(encoding="utf-8") == "a = 1\n"
    assert list(scratch.iterdir()) == []


def test_a_parent_directory_escape_is_refused(run) -> None:
    result = run("../repo/../outside.py", "a", "b", ADD_RUNS)
    assert result.code == 2
    assert "is outside the repository" in result.err


@pytest.mark.parametrize(
    "name", ["local/x.py", ".claude/worktrees/w/a.py", ".venv/x.py", ".git/x.py"]
)
def test_a_file_in_a_place_the_copy_leaves_out_is_refused(
    run, repo, scratch, name
) -> None:
    # Without the check the file passes every other test, the copy lacks it,
    # and writing the mutant crashes: a traceback and exit 1, which is SURVIVED.
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a = 1\n", encoding="utf-8")
    result = run(name, "a = 1", "a = 2", ADD_RUNS)
    assert result.code == 2
    assert f"{name} is in a place the copy leaves out" in result.err
    assert list(scratch.iterdir()) == []


def test_an_absolute_path_inside_the_repository_is_taken_as_relative(run, repo) -> None:
    result = run(str(repo / CALC_FILE), SUM, SUM_BROKEN, str(repo / ADD_SUMS))
    assert result.code == 0, result.out + result.err
    # The test id the run reports is the one it ran: relative, in the copy.
    assert f"[{ADD_SUMS}]" in result.out


def test_an_absolute_test_id_runs_the_copys_tests_not_the_originals(run, repo) -> None:
    # A helper the test imports is test-side code: if pytest were given the
    # original's absolute path it would import the original's, never the mutant.
    node = f"{repo / 'tests' / 'test_calc.py'}::test_add_uses_the_helper"
    result = run("tests/helpers.py", "EXPECTED = 5", "EXPECTED = 6", node)
    assert result.code == 0, result.out + result.err


def test_the_tests_run_inside_the_copy_not_in_the_original(run) -> None:
    # test_imports_the_copy needs the current directory to be the copy that
    # holds pkg; a run in the original's directory fails its baseline.
    result = run(
        CALC_FILE, SUM, SUM_BROKEN, "tests/test_calc.py::test_imports_the_copy"
    )
    assert result.code == 1, result.out + result.err
    assert result.out.startswith("SURVIVED")


def test_the_resolver_keeps_a_node_id_suffix(mutant, repo) -> None:
    assert mutant.split_test_id("tests/a.py::T::test_x[1::2]") == (
        "tests/a.py",
        "::T::test_x[1::2]",
    )
    assert mutant.split_test_id("tests") == ("tests", "")
    rel = mutant.resolve_in_repo(repo.resolve(), "src/../tests/x.py", "file")
    # as_posix: a WindowsPath never equals a PurePosixPath (FM-24).
    assert rel.as_posix() == "tests/x.py"


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        (".git", True),
        (".venv/lib/x.py", True),
        ("src/pkg/__pycache__/a.pyc", True),
        (".pytest_cache", True),
        (".mypy_cache/3.13", True),
        (".ruff_cache", True),
        (".hypothesis/examples", True),
        (".claude/worktrees/issue-1/src/a.py", True),
        ("local/surveys/a.json", True),
        (".claude/skills/develop/SKILL.md", False),
        ("src/smc/local.py", False),
        ("src/pkg/local/x.py", False),
        ("tests/test_a.py", False),
        ("pyproject.toml", False),
    ],
)
def test_what_the_copy_leaves_out(mutant, path, ignored) -> None:
    assert mutant.is_ignored(PurePosixPath(path)) is ignored


def test_directories_the_copy_leaves_out_are_not_copied(run, repo, scratch) -> None:
    for name in (
        ".git/HEAD",
        ".venv/x.txt",
        ".claude/worktrees/w/f.txt",
        "local/s.txt",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (repo / ".claude" / "skills.md").write_text("kept", encoding="utf-8")
    (repo / "src" / "pkg" / "__pycache__").mkdir()
    (repo / "src" / "pkg" / "__pycache__" / "calc.cpython-313.pyc").write_bytes(b"x")

    result = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS, "--keep")

    assert result.code == 0
    (copy,) = scratch.glob("smc-mutant-*/repo")
    assert (copy / "src" / "pkg" / "calc.py").is_file()
    assert (copy / ".claude" / "skills.md").is_file()
    for name in (".git", ".venv", ".claude/worktrees", "local", "src/pkg/__pycache__"):
        assert not (copy / name).exists(), name


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege")
def test_a_symlink_is_followed_so_the_mutant_cannot_write_through_it(
    run, repo, tmp_path, scratch
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    (repo / "src" / "pkg" / "linked.py").symlink_to(outside)

    result = run("src/pkg/linked.py", "value = 1", "value = 2", ADD_RUNS, "--keep")

    assert result.code == 1
    assert outside.read_text(encoding="utf-8") == "value = 1\n"
    (kept,) = scratch.glob("smc-mutant-*/repo/src/pkg/linked.py")
    assert not kept.is_symlink()
    assert kept.read_text(encoding="utf-8") == "value = 2\n"


# --- a run that gives no verdict -------------------------------------------


def test_a_test_that_already_fails_gives_no_verdict(run) -> None:
    result = run(CALC_FILE, SUM, SUM_BROKEN, "tests/test_calc.py::test_always_fails")
    assert result.code == 2
    assert "baseline fails" in result.err
    assert "KILLED" not in result.out
    # What explains the failure is pytest's own output, not only our summary.
    assert "FAILED tests/test_calc.py::test_always_fails" in result.err


def test_a_test_that_is_skipped_gives_no_verdict(run) -> None:
    result = run(CALC_FILE, SUM, SUM_BROKEN, "tests/test_calc.py::test_skipped")
    assert result.code == 2
    assert "baseline ran no passing test" in result.err
    assert "SURVIVED" not in result.out


def test_a_deselected_hardware_test_gives_no_verdict(run) -> None:
    result = run(CALC_FILE, SUM, SUM_BROKEN, "tests/test_calc.py::test_hardware")
    assert result.code == 2
    assert "baseline fails" in result.err
    assert "pytest exit 5" in result.err


def test_a_mutant_that_does_not_import_gives_no_verdict(run) -> None:
    result = run(CALC_FILE, SUM, "return a +", ADD_SUMS)
    assert result.code == 2
    assert "no verdict" in result.err
    assert "KILLED" not in result.out


def test_a_mutant_that_makes_the_test_skip_gives_no_verdict(run) -> None:
    result = run(
        "tests/test_calc.py",
        "def test_add_sums():",
        "@pytest.mark.skip\ndef test_add_sums():",
        ADD_SUMS,
    )
    assert result.code == 2
    assert "no verdict" in result.err
    assert "SURVIVED" not in result.out


def test_a_timeout_is_no_verdict_not_a_kill(
    mutant, repo, scratch, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MUTANT_TEST_OUTER_MARKER", "seen")
    calls = []

    def stub(command, cwd, env, timeout_s, log_path):
        calls.append((command, cwd, env, timeout_s))
        if len(calls) == 1:
            return mutant.ChildRun(0, False, "1 passed in 0.1s\n", 0.1)
        return mutant.ChildRun(-9, True, "", float(timeout_s))

    monkeypatch.setattr(mutant, "run_child", stub)
    code = mutant.main(
        ["--root", str(repo), "--timeout=5", CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "timed out after 5" in captured.err
    assert "KILLED" not in captured.out
    assert len(calls) == 2
    # What judge() hands to run_child, for both runs: the limit the user gave,
    # the copy as the directory, the outer environment (PATH, SYSTEMROOT on
    # Windows) plus the copy's own additions.
    for _command, cwd, env, timeout_s in calls:
        assert timeout_s == 5.0
        assert cwd != repo
        assert cwd.name == "repo"
        assert cwd.parent.parent == scratch.resolve()
        assert env["MUTANT_TEST_OUTER_MARKER"] == "seen"
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(cwd / "src")


def test_a_baseline_that_times_out_is_no_verdict(
    mutant, repo, scratch, monkeypatch, capsys
) -> None:
    calls = []

    def stub(command, cwd, env, timeout_s, log_path):
        calls.append(command)
        return mutant.ChildRun(
            -9, True, "", float(timeout_s), "the process tree could not be stopped"
        )

    monkeypatch.setattr(mutant, "run_child", stub)
    code = mutant.main(
        ["--root", str(repo), "--timeout=5", CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "baseline timed out after 5" in err
    # A helper that survived the kill is told, not buried (FM-35).
    assert "the process tree could not be stopped" in err
    assert len(calls) == 1, "the mutant must not run after a baseline that hangs"


def test_an_unexpected_error_is_no_verdict_not_a_survivor(
    mutant, repo, scratch, monkeypatch, capsys
) -> None:
    # An uncaught exception ends a Python process with exit 1: that is SURVIVED.
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mutant, "write_mutant", broken)
    before = tree_hash(repo)

    code = mutant.main(["--root", str(repo), CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS])

    captured = capsys.readouterr()
    assert code == 2
    assert "ERROR: unexpected OSError: disk full" in captured.err
    assert "Traceback" in captured.err
    assert "SURVIVED" not in captured.out
    assert list(scratch.iterdir()) == []
    assert tree_hash(repo) == before


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "3601", "soon"])
def test_a_timeout_that_is_not_a_sane_number_is_refused(mutant, repo, value) -> None:
    with pytest.raises(SystemExit) as exc:
        mutant.main(
            ["--root", str(repo), f"--timeout={value}", CALC_FILE, "a", "b", ADD_RUNS]
        )
    assert exc.value.code == 2


# --- the copy and its clean-up ----------------------------------------------


def test_the_copy_is_removed_by_default_and_kept_on_request(run, scratch) -> None:
    run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS)
    assert list(scratch.iterdir()) == []

    kept = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS, "--keep")
    (copy,) = scratch.glob("smc-mutant-*/repo")
    assert f"kept: {copy}" in kept.out
    assert SUM_BROKEN in (copy / CALC_FILE).read_text(encoding="utf-8")


def test_the_run_leaves_no_bytecode_and_no_pytest_cache_in_the_copy(
    run, scratch, monkeypatch
) -> None:
    # A stale .pyc once ran another mutant's code (#10): nothing may be written.
    # The variable may already be set here (this test runs under the script
    # itself when its own mutants are checked); the script must set it too.
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS, "--keep")
    (copy,) = scratch.glob("smc-mutant-*/repo")
    assert not (copy / ".pytest_cache").exists()
    assert not list(copy.rglob("__pycache__"))
    assert not list(copy.rglob("*.pyc"))


def test_the_pytest_command_is_the_documented_one(mutant) -> None:
    assert mutant.pytest_command("tests/a.py::t") == [
        sys.executable,
        "-m",
        "pytest",
        "tests/a.py::t",
        "-x",
        "-q",
        "-p",
        "no:cacheprovider",
    ]


def test_write_mutant_does_not_translate_newlines(
    mutant, tmp_path, monkeypatch
) -> None:
    # A text-mode write leaves "\r\n" alone on POSIX, so only the call shows it
    # (Windows would turn each "\n" into "\r\n" and doubled the "\r").
    calls = []
    real_open = open

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(mutant, "open", spy, raising=False)
    target = tmp_path / "f.txt"
    target.write_bytes(b"x")

    mutant.write_mutant(target, "a\r\nb\n")

    assert calls[-1].get("newline") == ""
    assert calls[-1].get("encoding") == "utf-8"
    assert target.read_bytes() == b"a\r\nb\n"


def test_write_mutant_writes_a_read_only_file(mutant, tmp_path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o444)

    mutant.write_mutant(target, "new")

    assert target.read_text(encoding="utf-8") == "new"


def test_keep_also_keeps_the_copy_of_a_run_without_verdict(run, scratch) -> None:
    result = run(
        CALC_FILE, SUM, SUM_BROKEN, "tests/test_calc.py::test_always_fails", "--keep"
    )
    assert result.code == 2
    assert len(list(scratch.glob("smc-mutant-*/repo"))) == 1


def test_ctrl_c_removes_the_copy_and_returns_130(
    mutant, repo, scratch, monkeypatch, capsys
) -> None:
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mutant, "run_child", interrupted)
    before = tree_hash(repo)

    code = mutant.main(["--root", str(repo), CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS])

    assert code == 130
    assert "interrupted" in capsys.readouterr().err
    assert list(scratch.iterdir()) == []
    assert tree_hash(repo) == before


def test_a_temp_directory_inside_the_repository_is_refused(
    mutant, repo, monkeypatch, capsys
) -> None:
    inside = repo / "tmp"
    inside.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(inside))
    before = tree_hash(repo)

    code = mutant.main(["--root", str(repo), CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS])

    assert code == 2
    assert "is inside the repository" in capsys.readouterr().err
    assert tree_hash(repo) == before


def test_a_temp_directory_containing_the_path_separator_is_refused(
    mutant, repo, tmp_path, monkeypatch, capsys
) -> None:
    odd = tmp_path / f"a{os.pathsep}b"
    try:
        odd.mkdir()
    except OSError:
        pytest.skip(f"this file system refuses a folder named {odd.name!r}")
    monkeypatch.setattr(tempfile, "tempdir", str(odd))

    code = mutant.main(["--root", str(repo), CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS])

    assert code == 2
    assert "PYTHONPATH" in capsys.readouterr().err
    assert list(odd.iterdir()) == []


# --- the child process -------------------------------------------------------


def test_the_child_environment_puts_the_copy_first_and_leaves_coverage_out(
    mutant, tmp_path
) -> None:
    copy = tmp_path / "copy"
    base = {
        "PYTHONPATH": "/old/one",
        "COV_CORE_SOURCE": "smc",
        "COV_CORE_CONFIG": "x",
        "COV_CORE_DATAFILE": "/repo/.coverage",
        "COVERAGE_PROCESS_START": "x",
        "COVERAGE_PROCESS_CONFIG": "x",
        "COVERAGE_FILE": "/repo/.coverage",
        "KEEP": "me",
    }
    env = mutant.child_env(copy, base)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(copy / "src"), "/old/one"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["KEEP"] == "me"
    assert not [name for name in env if name.startswith(("COV_CORE_", "COVERAGE_"))]
    assert base["PYTHONPATH"] == "/old/one"


@pytest.mark.parametrize("existing", [{}, {"PYTHONPATH": ""}])
def test_an_empty_pythonpath_adds_no_empty_entry(mutant, tmp_path, existing) -> None:
    # An empty entry is the current directory (FM-78).
    env = mutant.child_env(tmp_path / "copy", existing)
    assert env["PYTHONPATH"] == str(tmp_path / "copy" / "src")


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("1 passed in 0.10s", 1),
        ("FAILED x\n2 failed, 3 passed in 1.00s", 3),
        ("1 skipped in 0.05s", 0),
        ("1 xfailed in 0.05s", 0),
        ("\x1b[32m3 passed\x1b[0m\x1b[32m in 0.1s\x1b[0m", 3),
        ("no tests ran in 0.05s", 0),
        ("", 0),
    ],
)
def test_passed_count_reads_the_summary_line(mutant, text, count) -> None:
    assert mutant.passed_count(text) == count


def test_run_child_reports_the_exit_code_and_the_output(mutant, tmp_path) -> None:
    result = mutant.run_child(
        [sys.executable, "-c", "import sys; print('héllo'); sys.exit(3)"],
        tmp_path,
        {**os.environ, "PYTHONIOENCODING": "utf-8"},
        30.0,
        tmp_path / "log.txt",
    )
    assert result.returncode == 3
    assert not result.timed_out
    assert "héllo" in result.output


def test_run_child_kills_the_whole_tree_on_timeout(mutant, tmp_path) -> None:
    beat = tmp_path / "beat.txt"
    grandchild = (
        "import time\n"
        "start = time.time()\n"
        "while time.time() - start < 20:\n"
        f"    open({str(beat)!r}, 'a').write('x')\n"
        "    time.sleep(0.05)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        f"while not (os.path.exists({str(beat)!r}) and os.path.getsize({str(beat)!r})):\n"
        "    time.sleep(0.05)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    result = mutant.run_child(
        [sys.executable, str(parent)], tmp_path, dict(os.environ), 5.0, tmp_path / "l"
    )

    assert result.timed_out
    assert result.seconds < 60
    assert beat.stat().st_size > 0, "the grandchild never started: nothing was proved"
    # Killed is not gone (FM-35): the grandchild must have stopped writing.
    time.sleep(0.3)
    first = beat.stat().st_size
    time.sleep(0.5)
    assert beat.stat().st_size == first


def test_ctrl_c_during_the_wait_kills_the_child(mutant, tmp_path, monkeypatch) -> None:
    # The child leads its own session, so a terminal's Ctrl-C never reaches it:
    # only the `finally` of run_child stops it.
    created = []

    class InterruptedPopen(subprocess.Popen):
        # Only the process under test is interrupted and recorded: on Windows
        # kill_tree starts `taskkill` through subprocess.run, which builds a
        # Popen of the same (patched) class.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.under_test = self.args[0] == sys.executable
            self.interrupted = False
            if self.under_test:
                created.append(self)

        def wait(self, timeout=None):
            if self.under_test and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().wait(timeout)

    monkeypatch.setattr(mutant.subprocess, "Popen", InterruptedPopen)
    try:
        with pytest.raises(KeyboardInterrupt):
            mutant.run_child(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                tmp_path,
                dict(os.environ),
                30.0,
                tmp_path / "log.txt",
            )
        (child,) = created
        assert child.poll() is not None, "the child was left running"
    finally:
        for child in created:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


def test_run_child_waits_in_short_slices_so_that_ctrl_c_is_seen(
    mutant, tmp_path, monkeypatch
) -> None:
    # On Windows one long wait is one blocking call that a Ctrl-C cannot end.
    waits = []

    class RecordingPopen(subprocess.Popen):
        def wait(self, timeout=None):
            if self.args[0] == sys.executable:
                waits.append(timeout)
            return super().wait(timeout)

    monkeypatch.setattr(mutant.subprocess, "Popen", RecordingPopen)
    result = mutant.run_child(
        [sys.executable, "-c", "import time; time.sleep(1.5)"],
        tmp_path,
        dict(os.environ),
        30.0,
        tmp_path / "log.txt",
    )
    assert result.returncode == 0
    assert not result.timed_out
    assert len(waits) >= 2
    assert max(waits) <= 1.0


class _DeadChild:
    """What kill_tree needs of a process that is already gone."""

    pid = 4242

    def kill(self) -> None:
        pass

    def wait(self, timeout=None) -> int:
        return 0


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.CompletedProcess(["taskkill"], 1, b"", b"Access is denied"),
        FileNotFoundError("taskkill"),
    ],
    ids=["taskkill denied", "taskkill missing"],
)
def test_a_failed_taskkill_is_not_a_clean_stop(mutant, monkeypatch, outcome) -> None:
    # The direct child is dead, but a helper may not be (FM-35). Windows is
    # simulated: taskkill is stubbed, so this runs on every platform.
    def fake_run(command, **kwargs):
        assert command[:3] == ["taskkill", "/T", "/F"]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(mutant, "_WINDOWS", True)
    monkeypatch.setattr(mutant.subprocess, "run", fake_run)
    note = mutant.kill_tree(_DeadChild())
    assert "the process tree could not be stopped (taskkill" in note
    assert "a helper may still be running" in note


def test_a_taskkill_that_works_leaves_no_note(mutant, monkeypatch) -> None:
    monkeypatch.setattr(mutant, "_WINDOWS", True)
    monkeypatch.setattr(
        mutant.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, b"", b""),
    )
    assert mutant.kill_tree(_DeadChild()) == ""


# --- several replacements, one mutant ---------------------------------------

ALSO_TOTAL = ("--also", CALC_FILE, TOTAL, TOTAL_BROKEN)


def test_two_replacements_are_one_mutant_killed_only_by_both(run) -> None:
    # test_add_or_total_is_intact passes while either edit is missing: only a
    # copy that holds both can fail it, so one edit per run could never kill.
    both = run(CALC_FILE, SUM, SUM_BROKEN, INTACT, *ALSO_TOTAL)
    assert both.code == 0, both.out + both.err
    assert both.out.startswith("KILLED")
    assert run(CALC_FILE, SUM, SUM_BROKEN, INTACT).code == 1
    assert run(CALC_FILE, TOTAL, TOTAL_BROKEN, INTACT).code == 1


def test_the_order_of_the_replacements_does_not_change_the_mutant(run) -> None:
    # The main pattern is the later one in the file: the edits are applied in
    # the order of the text, not of the command line, or the file is scrambled.
    reversed_order = run(
        CALC_FILE, TOTAL, TOTAL_BROKEN, INTACT, "--also", CALC_FILE, SUM, SUM_BROKEN
    )
    assert reversed_order.code == 0, reversed_order.out + reversed_order.err


def test_a_file_given_under_two_spellings_is_edited_once_with_both_replacements(
    run, repo
) -> None:
    # Grouped by the string, the second spelling would write its own text over
    # the first one's: the copy would hold one edit and the test would survive.
    absolute = ("--also", str(repo / CALC_FILE), TOTAL, TOTAL_BROKEN)
    result = run(CALC_FILE, SUM, SUM_BROKEN, INTACT, *absolute)
    assert result.code == 0, result.out + result.err


def test_two_replacements_in_two_files_land_in_the_same_copy(
    run, repo, scratch
) -> None:
    before = tree_hash(repo)
    result = run(
        CALC_FILE,
        SUM,
        SUM_BROKEN,
        ADD_SUMS,
        "--also",
        "tests/helpers.py",
        "EXPECTED = 5",
        "EXPECTED = 6",
        "--keep",
    )
    assert result.code == 0, result.out + result.err
    (copy,) = scratch.glob("smc-mutant-*/repo")
    assert SUM_BROKEN in (copy / CALC_FILE).read_text(encoding="utf-8")
    helpers = copy / "tests" / "helpers.py"
    assert helpers.read_text(encoding="utf-8") == "EXPECTED = 6\n"
    assert tree_hash(repo) == before


def test_several_replacements_make_one_baseline_and_one_mutated_run(
    mutant, repo, scratch, monkeypatch, capsys
) -> None:
    seen = []

    def stub(command, cwd, env, timeout_s, log_path):
        seen.append((cwd / CALC_FILE).read_text(encoding="utf-8"))
        if len(seen) == 1:
            return mutant.ChildRun(0, False, "1 passed in 0.1s\n", 0.1)
        return mutant.ChildRun(1, False, "1 failed in 0.1s\n", 0.1)

    monkeypatch.setattr(mutant, "run_child", stub)
    argv = ["--root", str(repo), *ALSO_TOTAL, "--", CALC_FILE, SUM, SUM_BROKEN, INTACT]
    assert mutant.main(argv) == 0
    capsys.readouterr()
    mutated = CALC.replace(SUM, SUM_BROKEN).replace(TOTAL, TOTAL_BROKEN)
    assert seen == [CALC, mutated]


@pytest.mark.parametrize(
    ("main_old", "also_old", "message"),
    [
        (
            SUM,
            "return a * b",
            "replacement 2 of 2 ('return a * b'): pattern found 0 times "
            "in src/pkg/calc.py (expected exactly 1)",
        ),
        (
            "return a / b",
            TOTAL,
            "replacement 1 of 2 ('return a / b'): pattern found 0 times "
            "in src/pkg/calc.py (expected exactly 1)",
        ),
        (
            SUM,
            "a + b",
            "replacement 2 of 2 ('a + b'): pattern found 2 times "
            "in src/pkg/calc.py (lines 2, 6)",
        ),
    ],
    ids=["second missing", "first missing", "second twice"],
)
def test_a_replacement_not_found_exactly_once_is_no_verdict_and_is_named(
    run, scratch, main_old, also_old, message
) -> None:
    also = ("--also", CALC_FILE, also_old, "x")
    result = run(CALC_FILE, main_old, "return a % b", INTACT, *also)
    assert result.code == 2
    assert message in result.err
    assert "KILLED" not in result.out
    assert "SURVIVED" not in result.out
    assert list(scratch.iterdir()) == []


def test_a_run_with_one_replacement_keeps_its_messages(run) -> None:
    result = run(CALC_FILE, "return a * b", "return a - b", ADD_RUNS)
    assert result.code == 2
    assert result.err.startswith("ERROR: pattern found 0 times in src/pkg/calc.py")


def test_replacements_match_the_original_text_not_the_previous_one(run) -> None:
    # "a * b" exists only in what the main replacement inserts.
    also = ("--also", CALC_FILE, "a * b", "a / b")
    result = run(CALC_FILE, SUM, "return a * b  # sum", ADD_RUNS, *also)
    assert result.code == 2
    assert "replacement 2 of 2 ('a * b'): pattern found 0 times" in result.err


@pytest.mark.parametrize(
    "also_old",
    ["a + b  # sum", SUM, "def add(a, b):\n    return a"],
    ids=["inside the first", "the same", "from before the first"],
)
def test_overlapping_replacements_are_refused(run, scratch, also_old) -> None:
    also = ("--also", CALC_FILE, also_old, "x")
    result = run(CALC_FILE, SUM, SUM_BROKEN, ADD_RUNS, *also)
    assert result.code == 2
    assert "replacements 1 and 2 overlap in src/pkg/calc.py" in result.err
    assert "SURVIVED" not in result.out
    assert list(scratch.iterdir()) == []


def test_replacements_that_touch_without_overlapping_are_accepted(run, scratch) -> None:
    also = ("--also", CALC_FILE, " > 0:", " < 0:")
    result = run(CALC_FILE, "while n", "while not n", ADD_RUNS, *also, "--keep")
    assert result.code == 1, result.out + result.err
    (kept,) = scratch.glob("smc-mutant-*/repo/src/pkg/calc.py")
    assert "    while not n < 0:\n" in kept.read_text(encoding="utf-8")


def test_the_report_line_names_every_replacement(run) -> None:
    several = run(CALC_FILE, SUM, SUM_BROKEN, INTACT, *ALSO_TOTAL)
    first = f"{CALC_FILE}: {SUM!r} -> {SUM_BROKEN!r}"
    second = f"{CALC_FILE}: {TOTAL!r} -> {TOTAL_BROKEN!r}"
    assert f"{first} + {second}  [{INTACT}]" in several.out
    # One replacement prints what it always printed.
    one = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS)
    assert f"KILLED   {first}  [{ADD_SUMS}]" in one.out


# --- a pattern from a file ----------------------------------------------------


class PatternFiles(NamedTuple):
    old: Path
    new: Path


@pytest.fixture
def pattern_files(tmp_path: Path) -> PatternFiles:
    directory = tmp_path / "patterns"
    directory.mkdir()
    return PatternFiles(directory / "old.txt", directory / "new.txt")


def test_a_pattern_read_from_a_file_may_span_several_lines(
    run_argv, pattern_files
) -> None:
    pattern_files.old.write_bytes(b"def add(a, b):\n    return a + b  # sum")
    pattern_files.new.write_bytes(b"def add(a, b):\n    return a - b  # sum")
    result = run_argv(
        f"--old-file={pattern_files.old}",
        f"--new-file={pattern_files.new}",
        "--",
        CALC_FILE,
        ADD_SUMS,
    )
    assert result.code == 0, result.out + result.err
    assert result.out.startswith("KILLED")
    assert result.out.isascii()
    assert "\\n" in result.out


def test_a_pattern_file_is_used_as_written_with_its_line_endings(
    run_argv, repo, pattern_files
) -> None:
    (repo / "src" / "pkg" / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    pattern_files.new.write_bytes(b"x = 1\r\ny = 9")
    pattern_files.old.write_bytes(b"x = 1\ny = 2")
    options = (f"--old-file={pattern_files.old}", f"--new-file={pattern_files.new}")
    lf = run_argv(*options, "--", "src/pkg/crlf.py", ADD_RUNS)
    assert lf.code == 2
    assert "pattern found 0 times in src/pkg/crlf.py" in lf.err
    pattern_files.old.write_bytes(b"x = 1\r\ny = 2")
    crlf = run_argv(*options, "--", "src/pkg/crlf.py", ADD_RUNS)
    assert crlf.code == 1, crlf.out + crlf.err


def test_a_pattern_file_keeps_its_final_newline(
    run_argv, repo, pattern_files, scratch
) -> None:
    (repo / "src" / "pkg" / "lines.py").write_bytes(b"x = 1\ny = 2\nz = 3\n")
    pattern_files.old.write_bytes(b"y = 2\n")
    pattern_files.new.write_bytes(b"y = 9\n\n")
    result = run_argv(
        f"--old-file={pattern_files.old}",
        f"--new-file={pattern_files.new}",
        "--keep",
        "--",
        "src/pkg/lines.py",
        ADD_RUNS,
    )
    assert result.code == 1, result.out + result.err
    (kept,) = scratch.glob("smc-mutant-*/repo/src/pkg/lines.py")
    assert kept.read_bytes() == b"x = 1\ny = 9\n\nz = 3\n"


@pytest.mark.parametrize("side", ["old", "new"])
@pytest.mark.parametrize("kind", ["missing", "directory", "not utf-8"])
def test_a_pattern_file_that_cannot_be_read_is_no_verdict_not_a_survivor(
    run_argv, pattern_files, scratch, side, kind
) -> None:
    pattern_files.old.write_bytes(b"return a + b  # sum")
    pattern_files.new.write_bytes(b"return a - b  # sum")
    broken = getattr(pattern_files, side)
    broken.unlink()
    if kind == "directory":
        broken.mkdir()
    elif kind == "not utf-8":
        broken.write_bytes(b"x = '\xe9'")
    result = run_argv(
        f"--old-file={pattern_files.old}",
        f"--new-file={pattern_files.new}",
        "--",
        CALC_FILE,
        ADD_RUNS,
    )
    # Raised outside the handler, the error would end the process with exit 1.
    assert result.code == 2
    assert broken.name in result.err
    # The reader's own message, not the catch-all's "unexpected ..." one.
    reason = " is not valid UTF-8 (" if kind == "not utf-8" else "ERROR: cannot read "
    assert reason in result.err
    assert "unexpected" not in result.err
    assert "Traceback" not in result.err
    assert "SURVIVED" not in result.out
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--old-file=o.txt", "--", CALC_FILE, "new", ADD_RUNS],
            "--old-file and --new-file go together",
        ),
        (
            ["--new-file=n.txt", "--", CALC_FILE, "old", ADD_RUNS],
            "--old-file and --new-file go together",
        ),
        (
            [
                "--old-file=o.txt",
                "--new-file=n.txt",
                "--",
                CALC_FILE,
                SUM,
                SUM,
                ADD_RUNS,
            ],
            "with --old-file and --new-file give FILE TEST-ID",
        ),
        (["--", CALC_FILE, ADD_RUNS], "give FILE OLD NEW TEST-ID"),
        (["--", CALC_FILE, SUM, SUM_BROKEN, ADD_RUNS, "extra"], "give FILE OLD NEW"),
    ],
    ids=["old-file alone", "new-file alone", "pair and four arguments", "two", "five"],
)
def test_the_forms_of_the_command_line_are_not_mixed(
    mutant, repo, capsys, argv, message
) -> None:
    with pytest.raises(SystemExit) as exc:
        mutant.main(["--root", str(repo), *argv])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_also_file_reads_both_patterns_from_files(
    mutant, run_argv, repo, pattern_files, scratch, capsys
) -> None:
    # argparse reads "-n_items" as an option, so `--also` cannot carry it (and
    # the `--` that rescues the main pattern comes after its values).
    (repo / "src" / "pkg" / "dash.py").write_bytes(b"neg = -n_items\n")
    pattern_files.old.write_bytes(b"-n_items")
    pattern_files.new.write_bytes(b"+n_items")
    also = ["--also", "src/pkg/dash.py", "-n_items", "+n_items"]
    with pytest.raises(SystemExit) as exc:
        mutant.main(
            ["--root", str(repo), *also, "--", CALC_FILE, SUM, SUM_BROKEN, ADD_RUNS]
        )
    assert exc.value.code == 2
    capsys.readouterr()

    result = run_argv(
        "--also-file",
        "src/pkg/dash.py",
        str(pattern_files.old),
        str(pattern_files.new),
        "--keep",
        "--",
        CALC_FILE,
        SUM,
        SUM_BROKEN,
        ADD_RUNS,
    )
    assert result.code == 1, result.out + result.err
    (copy,) = scratch.glob("smc-mutant-*/repo")
    assert (copy / "src" / "pkg" / "dash.py").read_bytes() == b"neg = +n_items\n"
    assert SUM_BROKEN in (copy / CALC_FILE).read_text(encoding="utf-8")


def test_an_also_file_pattern_that_cannot_be_read_is_no_verdict(
    run_argv, pattern_files, scratch
) -> None:
    pattern_files.new.write_bytes(b"x")
    result = run_argv(
        "--also-file",
        CALC_FILE,
        str(pattern_files.old),
        str(pattern_files.new),
        "--",
        CALC_FILE,
        SUM,
        SUM_BROKEN,
        ADD_RUNS,
    )
    assert result.code == 2
    assert "old.txt" in result.err
    assert list(scratch.iterdir()) == []


# --- the command line, as a session uses it ---------------------------------


def test_one_command_line_run_exits_with_the_documented_codes(repo) -> None:
    def cli(test_id: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(repo),
                "--",
                CALC_FILE,
                SUM,
                SUM_BROKEN,
                test_id,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )

    killed = cli(ADD_SUMS)
    assert (killed.returncode, killed.stdout.split()[0]) == (0, "KILLED")
    survived = cli(ADD_RUNS)
    assert (survived.returncode, survived.stdout.split()[0]) == (1, "SURVIVED")


def test_verbose_prints_the_whole_pytest_output(run) -> None:
    quiet = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS)
    loud = run(CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS, "-v")
    assert "FAILED tests/test_calc.py::test_add_sums" not in quiet.out
    assert "FAILED tests/test_calc.py::test_add_sums" in loud.out


@pytest.mark.parametrize("encoding", ["ascii", "cp1252"])
def test_the_report_survives_a_stdout_that_cannot_encode_what_pytest_printed(
    repo, encoding
) -> None:
    # `-v` echoes the output of the failing test, which printed a "µ" (cp1252
    # has it) and an "é" plus a "✓" no console code page need have. A
    # strict stream raises UnicodeEncodeError, and a crash exits 1: SURVIVED.
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "-v",
            "--",
            CALC_FILE,
            SUM,
            SUM_BROKEN,
            "tests/test_calc.py::test_prints_non_ascii",
        ],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": encoding},
        timeout=120,
    )
    stdout = proc.stdout.decode(encoding)
    stderr = proc.stderr.decode(encoding)
    assert proc.returncode == 0, stdout + stderr
    assert "Traceback" not in stderr
    assert stdout.startswith("KILLED")


def test_the_report_line_is_ascii_whatever_the_pattern(run, repo) -> None:
    (repo / "src" / "pkg" / "units.py").write_text("unit = 'µm'\n", encoding="utf-8")
    result = run("src/pkg/units.py", "µm", "mm", ADD_RUNS)
    assert result.code == 1
    assert result.out.isascii()
    assert "\\xb5m" in result.out
