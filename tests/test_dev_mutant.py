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
from pkg.calc import add, countdown


def test_add_sums():
    assert add(2, 3) == 5


def test_add_runs():
    add(2, 3)


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


def test_an_absolute_path_inside_the_repository_is_taken_as_relative(run, repo) -> None:
    result = run(str(repo / CALC_FILE), SUM, SUM_BROKEN, str(repo / ADD_SUMS))
    assert result.code == 0, result.out + result.err


def test_the_resolver_keeps_a_node_id_suffix(mutant, repo) -> None:
    assert mutant.split_test_id("tests/a.py::T::test_x[1::2]") == (
        "tests/a.py",
        "::T::test_x[1::2]",
    )
    assert mutant.split_test_id("tests") == ("tests", "")
    rel = mutant.resolve_in_repo(repo.resolve(), "src/../tests/x.py", "file")
    assert rel == PurePosixPath("tests/x.py")


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


def test_a_timeout_is_no_verdict_not_a_kill(mutant, repo, scratch, monkeypatch, capsys):
    calls = []

    def stub(command, cwd, env, timeout_s, log_path):
        calls.append(command)
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


def test_a_baseline_that_times_out_is_no_verdict(
    mutant, repo, scratch, monkeypatch, capsys
) -> None:
    calls = []

    def stub(command, cwd, env, timeout_s, log_path):
        calls.append(command)
        return mutant.ChildRun(-9, True, "", float(timeout_s))

    monkeypatch.setattr(mutant, "run_child", stub)
    code = mutant.main(
        ["--root", str(repo), "--timeout=5", CALC_FILE, SUM, SUM_BROKEN, ADD_SUMS]
    )
    assert code == 2
    assert "baseline timed out after 5" in capsys.readouterr().err
    assert len(calls) == 1, "the mutant must not run after a baseline that hangs"


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


def test_the_report_line_is_ascii_whatever_the_pattern(run, repo) -> None:
    (repo / "src" / "pkg" / "units.py").write_text("unit = 'µm'\n", encoding="utf-8")
    result = run("src/pkg/units.py", "µm", "mm", ADD_RUNS)
    assert result.code == 1
    assert result.out.isascii()
    assert "\\xb5m" in result.out
