"""Run one mutant against one test, in a scratch copy of the repository.

"Can this test fail?" is answered by breaking the code it guards and running
the test: if it still passes, the test proves nothing. Doing that by hand cost
three tool calls per mutant and went wrong three times (a stale ``.pyc`` ran
another mutant's code, #10; a ``git checkout`` to restore a mutant discarded an
uncommitted fix, #93). This script does it in one call and never writes to the
repository it reads:

    python scripts/dev/mutant.py <file> <old> <new> <test-id>
    python scripts/dev/mutant.py -- src/smc/x.py "-1" "+1" tests/test_x.py::test_y

It copies the repository (the working tree, uncommitted edits included) to a
temporary directory, replaces ``<old>`` by ``<new>`` there, exactly once, and
runs ``python -m pytest <test-id> -x -q -p no:cacheprovider`` from inside the
copy. The copy's ``src`` goes first on ``PYTHONPATH``, so its code is imported
and not an editable install's (#80), and no bytecode is written, so a stale
``.pyc`` cannot run the previous mutant (#10).

Relative paths are taken from the repository root (like pytest's node ids); an
absolute path must lie inside it. ``<old>`` is matched as written, with no
newline translation: in a CRLF file a pattern of several lines contains
``\\r\\n``. Put ``--`` before the arguments when a pattern starts with a dash.

The test is run twice: first on the unmutated copy. Without that baseline an
interpreter without pytest (exit 1), a test that already fails, or one that is
skipped (exit 0, as when the demo adapters are missing) would read as KILLED or
SURVIVED. Measured with pytest 9.1: a test that is not found or a mutant that
does not import exits 4, a deselected test exits 5.

Exit codes, so that a loop over mutants can use them:

    0    KILLED    the test failed on the mutant: it can fail
    1    SURVIVED  the test passed on the mutant: it proves nothing about that line
    2    ERROR     no verdict: a usage error, a pattern not found exactly once,
                   a baseline that does not pass, a mutant pytest could not run,
                   a timeout (a hung test says nothing about its assertions)
    130  interrupted (the copy is removed first)

``--keep`` leaves the copy in place and prints its path. Stdlib only, Windows
included, ASCII output (a redirected stream on Windows cannot encode check
marks, #42).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

DEFAULT_TIMEOUT_S = 300.0
MAX_TIMEOUT_S = 3600.0
KILL_WAIT_S = 10.0
TAIL_LINES = 30
PATTERN_WIDTH = 60
# What the copy leaves out, wherever it is: the repository's history, virtual
# environments, caches. Bytecode in particular must never be copied (#10).
IGNORED_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
    }
)
# What a child started under coverage would write into the checkout it was
# started from: CI runs ``pytest --cov``, and the variables are inherited.
COVERAGE_VARIABLES = (
    "COVERAGE_PROCESS_START",
    "COVERAGE_PROCESS_CONFIG",
    "COVERAGE_FILE",
)
PYTEST_EXIT = {
    1: "tests failed",
    2: "interrupted",
    3: "internal error",
    4: "usage error, the test was not found or something does not import",
    5: "no tests collected, e.g. deselected by -m",
}
# A child sharing the console can change its code page (FM-22); the child of
# this script is created without one.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class MutantError(Exception):
    """A reason for exit 2, in one line, with the output that explains it."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ChildRun:
    """What one run of the child left behind."""

    returncode: int | None
    timed_out: bool
    output: str
    seconds: float
    note: str = ""


@dataclass(frozen=True)
class Plan:
    """Everything that is decided before anything is copied."""

    root: Path
    rel: PurePath
    old: str
    new: str
    mutated: str
    test_id: str


# --- pure parts -------------------------------------------------------------


def occurrences(text: str, old: str) -> list[int]:
    """Where ``old`` starts in ``text``, overlapping matches included.

    ``aa`` is found twice in ``aaa``: "exactly once" must mean once, or a
    mutant would silently hit the first of several places.
    """
    offsets = []
    start = text.find(old)
    while start != -1:
        offsets.append(start)
        start = text.find(old, start + 1)
    return offsets


def line_numbers(text: str, offsets: Sequence[int]) -> list[int]:
    """The 1-based line of each offset, to say which matches to tell apart."""
    return [text.count("\n", 0, offset) + 1 for offset in offsets]


def is_ignored(rel: PurePath) -> bool:
    """Whether a path (relative to the root) stays out of the copy.

    One predicate for the copy and for the check that the file and the test
    are not in a place the copy leaves out. ``local/`` is the untracked
    survey output, ``.claude/worktrees`` the other sessions' checkouts.
    """
    parts = rel.parts
    if any(part in IGNORED_NAMES for part in parts):
        return True
    return parts[:2] == (".claude", "worktrees") or parts[:1] == ("local",)


def resolve_in_repo(root: Path, text: str, what: str) -> PurePath:
    """A path given on the command line, as a path relative to ``root``.

    The plain path is tried first, then the resolved one (``/var`` is
    ``/private/var`` on macOS). A symlink inside the repository that points
    outside is accepted: the copy holds its content, not the link, so a
    mutant cannot write through it.
    """
    path = Path(text)
    candidate = path if path.is_absolute() else root / path
    for attempt in (Path(os.path.normpath(candidate)), candidate.resolve()):
        with contextlib.suppress(ValueError):
            return attempt.relative_to(root)
    raise MutantError(f"{what} {text} is outside the repository {root}")


def split_test_id(test_id: str) -> tuple[str, str]:
    """A node id as its path and the rest (``::Class::test[param]``, or empty)."""
    path, separator, rest = test_id.partition("::")
    return path, separator + rest if separator else ""


def last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def tail(output: str, lines: int = TAIL_LINES) -> str:
    return "\n".join(output.splitlines()[-lines:])


def passed_count(output: str) -> int:
    """The number of tests that passed, from pytest's last line; 0 if none.

    No word boundary in the pattern: a coloured summary has none between the
    colour code and the number. A last line that is not a summary reads as 0,
    which is the safe direction (an ERROR, never a verdict).
    """
    match = re.search(r"(\d+) passed", last_line(output))
    return int(match.group(1)) if match else 0


def shorten(text: str) -> str:
    """``text`` as ASCII, so that a pattern with a ``µ`` prints on any console."""
    shown = ascii(text)
    if len(shown) <= PATTERN_WIDTH:
        return shown
    return shown[: PATTERN_WIDTH - 3] + "..."


def child_env(copy: Path, base: Mapping[str, str]) -> dict[str, str]:
    """The environment of the pytest run in ``copy``.

    The copy's ``src`` comes first on ``PYTHONPATH`` and the old entries stay
    behind it. An empty value adds no entry: an empty entry is the current
    directory (FM-78). No bytecode, UTF-8 streams (FM-20/23), and none of the
    variables that would start coverage in the child.
    """
    env = {
        name: value
        for name, value in base.items()
        if name not in COVERAGE_VARIABLES and not name.startswith("COV_CORE_")
    }
    src = str(copy / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([src, existing]) if existing else src
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def pytest_command(test_id: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        test_id,
        "-x",
        "-q",
        "-p",
        "no:cacheprovider",
    ]


# --- before anything is copied ------------------------------------------------


def mutated_text(root: Path, rel: PurePath, old: str, new: str) -> str:
    """The file's text with ``old`` replaced by ``new``, once; nothing is written.

    The original is read as it is on disk (UTF-8, ``newline=""``: a CRLF file
    stays CRLF). Counting here, before any copy, means a stale pattern costs
    nothing and cannot pass as "survived".
    """
    if not old:
        raise MutantError("old is empty: there is nothing to replace")
    if old == new:
        raise MutantError("old and new are the same: that is no mutant")
    name = rel.as_posix()
    try:
        with open(root / rel, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except UnicodeDecodeError as exc:
        raise MutantError(
            f"{name} is not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
    except OSError as exc:
        raise MutantError(f"cannot read {name}: {exc}") from exc
    offsets = occurrences(text, old)
    if len(offsets) != 1:
        if not offsets:
            raise MutantError(f"pattern found 0 times in {name} (expected exactly 1)")
        lines = line_numbers(text, offsets)
        shown = ", ".join(str(line) for line in lines[:5])
        if len(lines) > 5:
            shown += ", ..."
        raise MutantError(
            f"pattern found {len(offsets)} times in {name} (lines {shown}); "
            "expected exactly 1: widen it"
        )
    start = offsets[0]
    return text[:start] + new + text[start + len(old) :]


def plan_run(args: argparse.Namespace) -> Plan:
    root = args.root.resolve()
    if not root.is_dir():
        raise MutantError(f"--root {args.root} is not a directory")
    rel = resolve_in_repo(root, args.file, "file")
    if is_ignored(rel):
        raise MutantError(f"{rel.as_posix()} is in a place the copy leaves out")
    if not (root / rel).is_file():
        raise MutantError(f"{rel.as_posix()} is not a file in the repository")
    mutated = mutated_text(root, rel, args.old, args.new)

    path_part, rest = split_test_id(args.test_id)
    if not path_part:
        raise MutantError("test-id is empty: give a file or a node id")
    test_rel = resolve_in_repo(root, path_part, "test path")
    if is_ignored(test_rel) or not (root / test_rel).exists():
        raise MutantError(f"test path {test_rel.as_posix()} is not in the repository")
    return Plan(root, rel, args.old, args.new, mutated, test_rel.as_posix() + rest)


# --- the copy -----------------------------------------------------------------


def make_scratch(root: Path) -> Path:
    """A fresh temporary directory that is safe to copy ``root`` into.

    Refused before it is created: inside ``root`` (the copy would recurse into
    itself) and with the path separator in its name (``PYTHONPATH`` would read
    it as two entries and import the original instead of the copy, FM-81).
    """
    temp = Path(tempfile.gettempdir()).resolve()
    if temp == root or root in temp.parents:
        raise MutantError(
            f"the temporary directory {temp} is inside the repository: "
            "set TMPDIR (TEMP on Windows) elsewhere"
        )
    if os.pathsep in str(temp):
        raise MutantError(
            f"the temporary directory {temp} contains {os.pathsep!r}, "
            "which PYTHONPATH cannot hold"
        )
    return Path(tempfile.mkdtemp(prefix="smc-mutant-")).resolve()


def copy_repository(root: Path, copy: Path) -> None:
    """Copy the working tree, symlinks followed: a link into the original would
    let the mutant write through it."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        rel = Path(directory).relative_to(root)
        return {name for name in names if is_ignored(rel / name)}

    try:
        shutil.copytree(root, copy, ignore=ignore, ignore_dangling_symlinks=True)
    except (OSError, shutil.Error) as exc:
        raise MutantError(f"cannot copy the repository: {exc}") from exc


def write_mutant(target: Path, text: str) -> None:
    """Write the mutated text, in the copy; a read-only file is made writable first."""
    mode = target.stat().st_mode
    if not mode & stat.S_IWUSR:
        target.chmod(mode | stat.S_IWUSR)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def remove_scratch(scratch: Path) -> None:
    shutil.rmtree(scratch, ignore_errors=True)
    if scratch.exists():
        sys.stderr.write(f"warning: could not remove {scratch}\n")


# --- the child ------------------------------------------------------------------


def kill_tree(proc: subprocess.Popen[bytes]) -> str:
    """Stop ``proc`` and what it started; the note says what could not be stopped.

    ``kill()`` alone stops only the direct child (FM-35): a test that spawns a
    helper would leave it running. POSIX: the child leads its own session, so
    its process group shares its pid. Windows: ``taskkill /T`` walks the tree.
    The wait after the kill has a timeout: a process stuck in a driver call may
    not die.
    """
    note = ""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=KILL_WAIT_S,
                creationflags=_NO_WINDOW,
                check=False,
            )
            if result.returncode != 0:
                note = f"taskkill exited {result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            note = f"taskkill failed: {exc}"
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    try:
        proc.wait(timeout=KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        note = f"{note}; the child could not be stopped".lstrip("; ")
    return note


def run_child(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    log_path: Path,
) -> ChildRun:
    """Run ``command`` with its output in a file; kill the whole tree on any exit.

    The output goes to a file, never a pipe: a helper process that inherits a
    pipe keeps it open and, on Windows, ``run(timeout=...)`` then waits without
    limit after the kill (FM-04). Timeout, Ctrl-C or any error: the tree is
    killed before the exception goes on.
    """
    started = time.monotonic()
    timed_out = False
    note = ""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW,
            start_new_session=os.name != "nt",
        )
        try:
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if proc.poll() is None:
                note = kill_tree(proc)
    output = log_path.read_text(encoding="utf-8", errors="replace")
    return ChildRun(
        proc.returncode, timed_out, output, time.monotonic() - started, note
    )


# --- the verdict -----------------------------------------------------------------


def exit_name(returncode: int | None) -> str:
    return PYTEST_EXIT.get(returncode, "") if returncode is not None else ""


def check_baseline(run: ChildRun, plan: Plan, timeout_s: float) -> None:
    if run.timed_out:
        raise MutantError(
            f"baseline timed out after {timeout_s:g} s: {plan.test_id} does not "
            "finish on the unmutated copy",
            tail(run.output),
        )
    if run.returncode != 0:
        why = f": {exit_name(run.returncode)}" if exit_name(run.returncode) else ""
        raise MutantError(
            f"baseline fails: {plan.test_id} does not pass on the unmutated copy "
            f"(pytest exit {run.returncode}{why}); a KILLED would prove nothing",
            tail(run.output),
        )
    if passed_count(run.output) < 1:
        raise MutantError(
            f"baseline ran no passing test: {plan.test_id} was skipped, "
            "xfailed or deselected (pytest exit 0, 0 passed)",
            tail(run.output),
        )


def judge(plan: Plan, scratch: Path, args: argparse.Namespace) -> int:
    copy = scratch / "repo"
    copy_repository(plan.root, copy)
    env = child_env(copy, os.environ)
    command = pytest_command(plan.test_id)
    log = scratch / "pytest.log"

    baseline = run_child(command, copy, env, args.timeout, log)
    check_baseline(baseline, plan, args.timeout)

    write_mutant(copy / plan.rel, plan.mutated)
    run = run_child(command, copy, env, args.timeout, log)
    note = f" ({run.note})" if run.note else ""
    if run.timed_out:
        raise MutantError(
            f"no verdict: the mutated run timed out after {args.timeout:g} s; "
            f"a hung test says nothing about its assertions{note}",
            tail(run.output),
        )
    if run.returncode == 1:
        verdict, code = "KILLED", 0
    elif run.returncode == 0:
        if passed_count(run.output) < 1:
            raise MutantError(
                "no verdict: the test passed nothing on the mutant "
                "(skipped or xfailed: did the mutant change that?)",
                tail(run.output),
            )
        verdict, code = "SURVIVED", 1
    else:
        why = f": {exit_name(run.returncode)}" if exit_name(run.returncode) else ""
        raise MutantError(
            f"no verdict: pytest exit {run.returncode}{why} on the mutated copy",
            tail(run.output),
        )

    write_out(
        f"{verdict:<8} {plan.rel.as_posix()}: {shorten(plan.old)} -> "
        f"{shorten(plan.new)}  [{plan.test_id}]  "
        f"({last_line(run.output)}, {run.seconds:.1f} s){note}"
    )
    if args.verbose:
        write_out(run.output.rstrip())
    return code


def execute(plan: Plan, args: argparse.Namespace) -> int:
    """Copy, run, report, and remove the copy whatever happens (Ctrl-C included)."""
    scratch = make_scratch(plan.root)
    code = 2
    try:
        try:
            code = judge(plan, scratch, args)
        except MutantError as exc:
            write_error(exc)
    finally:
        if args.keep:
            write_out(f"kept: {scratch / 'repo'}")
        else:
            remove_scratch(scratch)
    return code


# --- the command line ----------------------------------------------------------------


def write_out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def write_error(exc: MutantError) -> None:
    sys.stderr.write(f"ERROR: {exc.reason}\n")
    if exc.detail:
        sys.stderr.write(exc.detail.rstrip() + "\n")
    sys.stderr.flush()


def timeout_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or not 1.0 <= value <= MAX_TIMEOUT_S:
        raise argparse.ArgumentTypeError(
            f"{text} is not between 1 and {MAX_TIMEOUT_S:g} seconds"
        )
    return value


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace OLD by NEW once in FILE, in a scratch copy of the repository, "
            "and run TEST_ID there: KILLED if it fails, SURVIVED if it passes."
        ),
        epilog=(
            "Exit codes: 0 KILLED, 1 SURVIVED, 2 no verdict (see the module "
            "docstring), 130 interrupted. Put -- before the arguments when OLD "
            "starts with a dash."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("file", help="file to mutate, from the repository root")
    parser.add_argument("old", help="text to replace; must occur exactly once")
    parser.add_argument("new", help="its replacement")
    parser.add_argument("test_id", metavar="test-id", help="a pytest node id")
    parser.add_argument(
        "--keep", action="store_true", help="leave the copy in place and print its path"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="also print pytest's output"
    )
    parser.add_argument(
        "--timeout",
        type=timeout_seconds,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"limit of each of the two runs (default {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root(),
        help="the repository to copy (default: the one this script lives in)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Paths and pytest's own lines may not encode on a Windows console (cp1252).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    try:
        return execute(plan_run(args), args)
    except MutantError as exc:
        write_error(exc)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
