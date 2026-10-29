"""Look at the plate page: serve it on a demo stand and write what a user sees.

Nothing else looks at the rendered page: the tests build it and click on it
without a browser, and a defect that only a Bokeh session shows (the typed range
of #277) passes all of them (FM-161). This script serves ``smc ui plate`` on a
simulator stand in dry run, drives it with a headless Chromium through the states
that matter, and writes a PNG of each, plus ``summary.txt`` with the texts a
canvas hides. A session reads the PNGs with its Read tool (``/ui-check``).

    python scripts/dev/ui_snap.py [--stand {demo-sample,demo}] [--out DIR]
                                  [--width 1600] [--height 1000]
                                  [--no-results] [--keep-work]

The page can move a stage, so the script takes no profile: ``--stand`` is one of
the two demo stands, checked again against the repository's own file, which is
passed by absolute path (a name is looked up in the working directory and in
``$SMC_PROFILES``, and an operator's directory can shadow it). The page is always
served with ``--dry-run``. The chain of tools that fills the run document runs
*without* it, on the demo's simulated stage: in dry run the stage does not move,
every frame is plastic, and no state past the first three could be reached.

Exit codes: 0 every state reached, the server logged no error and stopped by
Ctrl-C; 1 the job could not be done (no playwright, no browser, the server never
served); 2 refused (the guard, or a bad option); 3 the PNGs are written and
something needs reading (a state not reached, a server or browser error, a server
that did not stop cleanly); 130 interrupted.

Stdlib only at import (``playwright`` is imported inside ``load_playwright``), ASCII
output, UTF-8 files. Measured on macOS only (#268); Windows stops the server with
``terminate()`` and says that the Ctrl-C path was not exercised.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = ROOT / "profiles"
DEFAULT_OUT = ROOT / ".claude" / "ui-check"

#: The two stands the script may serve, a constant and not a parameter (#268).
ALLOWED_STANDS = ("demo-sample", "demo")

SERVING_LINE = "serving the plate page"
#: The page redraws on a 1 s tick (m5 §3): one pause after each action.
SETTLE_MS = 1500
#: Playwright waits 30 s for a disabled control by default: a state is not worth that.
DEFAULT_TIMEOUT_MS = 10_000
STEP_TIMEOUT_S = 120.0
SERVE_TIMEOUT_S = 60.0
READY_POLL_S = 0.1
#: A wait is sliced so that Python runs its Ctrl-C handler between two of them (FM-83).
WAIT_SLICE_S = 0.25
STOP_GRACE_S = 25.0
TERMINATE_GRACE_S = 10.0
KILL_GRACE_S = 5.0

STATE_NAMES = ("empty", "wells-selected", "snap", "after-density", "with-route")

# A child sharing the console can change its code page (FM-22).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ERROR_LINE = re.compile(r"\bERROR\b")
_TRACEBACK = "Traceback (most recent call last)"
_DROPPED_FROM_CHILD_ENV = ("SMC_PROFILE", "SMC_PROFILES")


class RefusedStandError(Exception):
    """The stand is not one this script may serve (exit 2)."""


class ChainFailedError(Exception):
    """A step of the chain did not end well; the message says which and how."""


class PageNotReadyError(Exception):
    """The page did not load or never showed its STOP button (exit 1)."""


@dataclass(frozen=True)
class Step:
    """One ``smc run`` of the documented workflow (``docs/plugins/README.md``)."""

    plugin: str
    params: tuple[tuple[str, str], ...] = ()


# The plate from its two corners (A1 and H12 of the centred 96-well plate of the
# demo-sample stand): `plate-finding` would take 7 minutes of demo-stage travel.
_REFS = [
    {"well": "A1", "x_um": -49500, "y_um": 31500},
    {"well": "H12", "x_um": 49500, "y_um": -31500},
]
CHAIN = (
    Step("plate-registration", (("refs", json.dumps(_REFS, separators=(",", ":"))),)),
    Step("scan-plan", (("wells", "B2,B11"), ("coverage", "0.4"))),
    Step("scan"),
    Step("cells", (("detector", "blobs"),)),
    Step("density"),
)
ROUTE_STEP = Step("route", (("labels", json.dumps(["single", "doublet"])),))


@dataclass(frozen=True)
class StatePlan:
    """Whether a state can be reached, and when it cannot, why."""

    name: str
    reachable: bool
    reason: str | None = None


@dataclass
class StateRecord:
    """What the session read from the page after one state."""

    name: str
    reached: bool
    reason: str | None = None
    active_tab: str | None = None
    wells: str | None = None
    texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StopOutcome:
    """How the server ended, in the words that go into the summary."""

    how: str
    returncode: int | None
    seconds: float
    text: str
    clean: bool


@dataclass(frozen=True)
class StepOutcome:
    """How one ``smc run`` of the chain ended."""

    label: str
    exit_code: int | None
    seconds: float
    last_line: str
    timed_out: bool = False
    timeout_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def failure(self) -> str | None:
        """The sentence for a step that failed (``None`` when it passed)."""
        if self.ok:
            return None
        last = self.last_line or "(no output)"
        if self.timed_out:
            return f"{self.label} did not end within {self.timeout_s:g} s: {last}"
        return f"{self.label} exited {self.exit_code}: {last}"


# --- the guard and the commands --------------------------------------------


def stand_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> Path:
    """The repository's own profile file for a demo stand, as an absolute path.

    Why so many checks for two names: the page this script serves can move a
    stage, and a profile name is looked up in the working directory and in
    ``$SMC_PROFILES``, so a name alone could reach an operator's stand. The name
    must be one of the two; the file must be the checkout's own, load, call itself
    by that name and use the simulator's configuration (``config = ""``). The path,
    never the name, goes to the child.
    """
    if name not in ALLOWED_STANDS:
        raise RefusedStandError(
            f"{name!r} is not one of the demo stands; this script drives a page "
            "that can move a stage"
        )
    path = profiles_dir / f"{name}.toml"
    if not path.is_file():
        raise RefusedStandError(f"{path} does not exist")
    from smc.hardware.profile import Profile, ProfileError

    try:
        profile = Profile.load(path)
    except ProfileError as exc:
        raise RefusedStandError(
            f"{path} does not load: {_first_line(str(exc))}"
        ) from exc
    if profile.microscope.name != name:
        raise RefusedStandError(
            f"{path.name} names itself {profile.microscope.name!r}, not {name!r}"
        )
    if profile.micromanager.config != "":
        raise RefusedStandError(
            f"{path.name} names a Micro-Manager configuration "
            f"({profile.micromanager.config!r}); only the simulator's "
            '(config = "") may be served here'
        )
    return path.resolve()


def child_env(base: Mapping[str, str]) -> dict[str, str]:
    """The environment of a child: no operator profile, UTF-8 and unbuffered output.

    ``SMC_PROFILE`` and ``SMC_PROFILES`` would let a lab PC's own directory
    shadow the name a command takes; the encodings keep a redirected log
    readable on Windows (FM-20), and unbuffered output lets the ready line be
    seen as it is written.
    """
    dropped = {key.upper() for key in _DROPPED_FROM_CHILD_ENV}
    env = {key: value for key, value in base.items() if key.upper() not in dropped}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def server_command(python: str, profile: Path, run_doc: Path, port: int) -> list[str]:
    """``smc ui plate`` on the guarded profile, always in dry run."""
    return [
        python,
        "-m",
        "smc.cli",
        "ui",
        "plate",
        "-p",
        str(profile),
        "--run-doc",
        str(run_doc),
        "--dry-run",
        "--port",
        str(port),
    ]


def run_command(python: str, step: Step, profile: Path, run_doc: Path) -> list[str]:
    """One step of the chain: a real run on the demo's simulated stage."""
    command = [python, "-m", "smc.cli", "run", step.plugin]
    for key, value in step.params:
        command += ["--param", f"{key}={value}"]
    return [*command, "-p", str(profile), "--run-doc", str(run_doc)]


def plan_states(
    stand: str, *, results: bool, chain_failure: str | None = None
) -> list[StatePlan]:
    """Which states can be reached, and why not for the others (never silent).

    The last two need the results of a real run of the workflow: the demo camera
    images no sample, ``--no-results`` skips the chain, and a chain step that failed
    leaves them without their input.
    """
    plans = [StatePlan(name, True) for name in STATE_NAMES[:3]]
    if stand != "demo-sample":
        reason: str | None = (
            "the demo camera images no sample: no cells, no density; "
            "use --stand demo-sample"
        )
    elif not results:
        reason = "skipped by --no-results"
    else:
        reason = None
    if reason is not None:
        return [*plans, *(StatePlan(n, False, reason) for n in STATE_NAMES[3:])]
    if chain_failure is not None:
        return [
            *plans,
            StatePlan("after-density", False, chain_failure),
            StatePlan("with-route", False, f"needs the chain: {chain_failure}"),
        ]
    return [*plans, *(StatePlan(n, True) for n in STATE_NAMES[3:])]


# --- what is read back -----------------------------------------------------


def server_errors(text: str) -> list[str]:
    """The lines of a server log that report an error (a level, or a traceback)."""
    return [
        line
        for line in text.splitlines()
        if _ERROR_LINE.search(line) or line.startswith(_TRACEBACK)
    ]


def browser_hint(message: str) -> str | None:
    """What to run when Playwright says its browser is not there; else ``None``."""
    if "Executable doesn't exist" not in message:
        return None
    return (
        "the browser is not installed: run: python -m playwright install chromium "
        "(on Linux add --with-deps)"
    )


def format_summary(
    records: Sequence[StateRecord],
    *,
    server: str,
    server_errors: Sequence[str],
    browser_errors: Sequence[str],
    browser_warnings: int = 0,
) -> str:
    """``summary.txt``: one block per state, then how the server and browser fared.

    The plate map and the image are canvas, so they are in the PNGs only; the
    blocks hold what the page's text panes say (read through the shadow DOM).
    ``?`` marks a read the page did not allow, never a guess.
    """
    lines: list[str] = []
    for record in records:
        lines.append(f"== {record.name} ==")
        if record.reached:
            lines.append("reached")
            lines.append(f"active tab: {record.active_tab or '?'}")
            if record.wells is not None:
                lines.append(f"Wells: {record.wells!r}")
            for text in record.texts:
                lines += ["", text]
        else:
            lines.append(f"not reached: {record.reason}")
        lines.append("")
    lines.append(f"server: {server}")
    lines.append(f"server errors: {len(server_errors)}")
    lines += [f"  {line}" for line in server_errors[:3]]
    lines.append(f"browser errors: {len(browser_errors)}")
    lines += [f"  {line}" for line in browser_errors[:3]]
    lines.append(f"browser warnings: {browser_warnings}")
    return "\n".join(lines) + "\n"


def exit_code(
    records: Sequence[StateRecord],
    outcome: StopOutcome,
    server_error_lines: Sequence[str],
    browser_error_lines: Sequence[str],
) -> int:
    """0 when there is nothing to read, 3 when the session must read the output."""
    clean = (
        all(record.reached for record in records)
        and outcome.clean
        and not server_error_lines
        and not browser_error_lines
    )
    return 0 if clean else 3


# --- processes -------------------------------------------------------------


def wait_in_slices(
    child: Any, timeout_s: float, slice_s: float = WAIT_SLICE_S
) -> int | None:
    """Wait for ``child`` up to ``timeout_s`` and return its exit code, else ``None``.

    On Windows ``Popen.wait(timeout)`` is one blocking call that a Ctrl-C does not
    interrupt, so a long wait leaves the terminal frozen (FM-83): wait in slices
    against a deadline, so that Python runs its signal handler between two.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return int(child.wait(timeout=min(slice_s, remaining)))
        except subprocess.TimeoutExpired:
            continue


def _read_log(log: Path) -> str:
    try:
        return log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def wait_for_serving(
    child: Any, log: Path, timeout_s: float = SERVE_TIMEOUT_S
) -> str | None:
    """``None`` once the server's log says it serves, else why it will not.

    The child is asked on every pass: a server that died at once (a profile that
    was not found, a port taken) is a failure now, naming its exit code and the
    end of its log, not after the whole timeout (FM-44).
    """
    deadline = time.monotonic() + timeout_s
    while True:
        text = _read_log(log)
        if SERVING_LINE in text:
            return None
        code = child.poll()
        if code is not None:
            return (
                f"the server ended with exit {code} before it served; "
                f"the end of its log:\n{_tail(text)}"
            )
        if time.monotonic() >= deadline:
            return (
                f"the server did not print {SERVING_LINE!r} within {timeout_s:g} s; "
                f"the end of its log:\n{_tail(text)}"
            )
        time.sleep(READY_POLL_S)


def _escalate(child: Any) -> tuple[str, int | None]:
    """``terminate()``, then ``kill()``: how the child ended and with which code.

    Killed is not gone (FM-35): the wait after ``kill()`` is bounded, and a child
    that is still there is said so, not waited for.
    """
    child.terminate()
    code = wait_in_slices(child, TERMINATE_GRACE_S)
    if code is not None:
        return "terminated", code
    child.kill()
    code = wait_in_slices(child, KILL_GRACE_S)
    return ("killed" if code is not None else "unkillable"), code


def stop_server(
    child: Any, *, windows: bool = os.name == "nt", grace_s: float = STOP_GRACE_S
) -> StopOutcome:
    """Stop the page's server the way Ctrl-C does, and say what ended it.

    POSIX: SIGINT (what Ctrl-C sends), then ``terminate()`` and ``kill()`` after a
    grace period, and the outcome names which one ended the child: a server that
    does not end on SIGINT is a finding (FM-160). Windows: a Ctrl-C cannot be sent
    to a child, and ``CTRL_BREAK_EVENT`` ends a Python child without its
    ``finally``, so it would prove nothing more: ``terminate()``, and the text says
    that the page's Ctrl-C path was not exercised.
    """
    started = time.monotonic()
    code = child.poll()
    if code is not None:
        return StopOutcome(
            "ended", code, 0.0, f"had already ended with exit {code}", False
        )

    def elapsed() -> float:
        return time.monotonic() - started

    if windows:
        how, code = _escalate(child)
        seconds = elapsed()
        if how == "terminated":
            text = (
                "stopped by terminate() (Windows: the page's Ctrl-C path was not "
                f"exercised), exit {code}"
            )
            return StopOutcome("terminated-windows", code, seconds, text, True)
        return StopOutcome(
            how, code, seconds, _escalation_text(how, "terminate()", code), False
        )
    try:
        child.send_signal(signal.SIGINT)
    except OSError:  # it ended between the poll and the signal
        code = child.poll()
    else:
        code = wait_in_slices(child, grace_s)
    if code is not None:
        seconds = elapsed()
        text = f"stopped by SIGINT, exit {code} in {seconds:.1f} s"
        return StopOutcome("sigint", code, seconds, text, code == 0)
    how, code = _escalate(child)
    seconds = elapsed()
    lead = f"did not end on SIGINT within {grace_s:g} s; "
    return StopOutcome(
        how, code, seconds, lead + _escalation_text(how, "terminate()", code), False
    )


def _escalation_text(how: str, first: str, code: int | None) -> str:
    if how == "terminated":
        return f"terminated, exit {code}"
    if how == "killed":
        return f"did not end on {first} within {TERMINATE_GRACE_S:g} s; killed, exit {code}"
    return (
        f"did not end on {first} nor on kill() within {KILL_GRACE_S:g} s: "
        "it may still be running"
    )


def run_step(
    command: Sequence[str],
    *,
    label: str,
    log: Path,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float = STEP_TIMEOUT_S,
) -> StepOutcome:
    """Run one command of the chain and append its output to ``log``.

    The output goes to a file, never a pipe (FM-04), and a step that does not end
    in time is ended and named, so a hung plugin is a finding and not a hang.
    """
    started = time.monotonic()
    with tempfile.TemporaryFile() as capture:
        child = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=capture,
            stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW,
        )
        try:
            code = wait_in_slices(child, timeout_s)
            if code is None:
                _escalate(child)
        except BaseException:
            _escalate(child)
            raise
        capture.seek(0)
        output = capture.read().decode("utf-8", errors="replace")
    seconds = time.monotonic() - started
    shown = "timeout" if code is None else str(code)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {label} exit {shown} in {seconds:.1f} s\n{output}")
        if output and not output.endswith("\n"):
            handle.write("\n")
    last = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip()), "")
    return StepOutcome(label, code, seconds, last, code is None, timeout_s)


def free_port() -> int:
    """A port the OS has just offered on ``localhost`` (the server checks it again)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("localhost", 0))
        return int(probe.getsockname()[1])


# --- the page --------------------------------------------------------------


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), text)


@dataclass
class Stand:
    """What the states need to run the chain beside the page."""

    python: str
    profile: Path
    run_doc: Path
    chain_log: Path
    env: dict[str, str]

    def run_chain(self, steps: Sequence[Step]) -> None:
        for step in steps:
            outcome = run_step(
                run_command(self.python, step, self.profile, self.run_doc),
                label=step.plugin,
                log=self.chain_log,
                cwd=ROOT,
                env=self.env,
            )
            failure = outcome.failure()
            if failure is not None:
                raise ChainFailedError(failure)


def _act_empty(page: Any, stand: Stand) -> None:
    """The page as it opens: nothing to do."""


def _act_wells_selected(page: Any, stand: Stand) -> None:
    box = page.get_by_label("Wells")
    box.fill("A1:B2")
    box.press("Enter")


def _act_snap(page: Any, stand: Stand) -> None:
    # The tabs are dynamic: the Snap button does not exist until Image is active.
    page.get_by_text("Image", exact=True).click()
    page.get_by_role("button", name="Snap").click()
    page.locator("pre").filter(has_text="snap: ").first.wait_for()


def _act_after_density(page: Any, stand: Stand) -> None:
    stand.run_chain(CHAIN)
    # Back to the Plate tab (Refresh is not on the Image tab), and read the file.
    page.get_by_text("Plate", exact=True).first.click()
    page.get_by_role("button", name="Refresh").click()


def _act_with_route(page: Any, stand: Stand) -> None:
    stand.run_chain([ROUTE_STEP])
    page.get_by_role("button", name="Refresh").click()


ACTIONS = {
    "empty": _act_empty,
    "wells-selected": _act_wells_selected,
    "snap": _act_snap,
    "after-density": _act_after_density,
    "with-route": _act_with_route,
}


def _read_page(page: Any) -> tuple[str | None, str | None, list[str]]:
    """The active tab, the Wells box and the text panes, as the page shows them.

    ``inner_text("body")`` of a Panel page is empty (shadow DOM); the panes' own
    ``pre`` elements are not.
    """
    tab = page.locator(".bk-tab.bk-active")
    active = tab.first.inner_text().strip() if tab.count() else None
    box = page.get_by_label("Wells")
    wells = box.input_value() if box.count() == 1 else None
    texts = [t.strip() for t in page.locator("pre").all_inner_texts() if t.strip()]
    return active, wells, texts


def _outputs(out: Path, name: str) -> list[Path]:
    return [out / f"{name}.png", out / f"{name}-full.png", out / f"{name}-FAILED.png"]


def clear_outputs(out: Path) -> None:
    """Remove what an earlier run left, so that nothing in ``out`` is stale."""
    for name in STATE_NAMES:
        for path in _outputs(out, name):
            path.unlink(missing_ok=True)
    for name in ("summary.txt", "server.log", "chain.log"):
        (out / name).unlink(missing_ok=True)


def drive(
    pw: Any,
    context: Any,
    *,
    port: int,
    plans: Sequence[StatePlan],
    stand_name: str,
    results: bool,
    stand: Stand,
    out: Path,
) -> tuple[list[StateRecord], list[str], int]:
    """Walk the page through the states; return what was read, the browser's errors
    and the count of its warnings. A state that fails is named and the next goes on."""
    page = context.new_page()
    page.set_default_timeout(DEFAULT_TIMEOUT_MS)
    errors: list[str] = []
    warnings = [0]

    def on_console(message: Any) -> None:
        if message.type == "error":
            errors.append(f"console: {_first_line(message.text)}")
        elif message.type == "warning":
            warnings[0] += 1

    page.on("pageerror", lambda exc: errors.append(f"page: {_first_line(str(exc))}"))
    page.on("console", on_console)
    try:
        page.goto(f"http://localhost:{port}/")
        page.get_by_role("button", name="STOP").wait_for(state="visible")
        page.wait_for_load_state("networkidle")
    except pw.Error as exc:
        raise PageNotReadyError(_first_line(str(exc))) from exc

    records: list[StateRecord] = []
    remaining = list(plans)
    while remaining:
        plan = remaining.pop(0)
        if not plan.reachable:
            records.append(StateRecord(plan.name, False, plan.reason))
            continue
        try:
            ACTIONS[plan.name](page, stand)
            page.wait_for_timeout(SETTLE_MS)
            page.screenshot(path=str(out / f"{plan.name}.png"))
            page.screenshot(path=str(out / f"{plan.name}-full.png"), full_page=True)
            active, wells, texts = _read_page(page)
        except (pw.Error, ChainFailedError) as exc:
            reason = (
                exc.args[0]
                if isinstance(exc, ChainFailedError)
                else _first_line(str(exc))
            )
            for stale in _outputs(out, plan.name)[:2]:
                stale.unlink(missing_ok=True)
            # A page that is gone has no picture: the reason is what there is to say.
            with contextlib.suppress(pw.Error):
                page.screenshot(path=str(out / f"{plan.name}-FAILED.png"))
            records.append(StateRecord(plan.name, False, reason))
            if isinstance(exc, ChainFailedError) and plan.name == "after-density":
                later = plan_states(stand_name, results=results, chain_failure=reason)
                remaining = list(later[len(records) :])
            continue
        records.append(StateRecord(plan.name, True, None, active, wells, texts))
    return records, errors, warnings[0]


# --- the script ------------------------------------------------------------


def load_playwright() -> Any:
    """``playwright.sync_api``, or exit 1 saying how to install it."""
    try:
        import playwright.sync_api as sync_api
    except ImportError:
        print(
            "error: the playwright package is not installed; "
            'run: pip install -e ".[dev]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return sync_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stand",
        choices=ALLOWED_STANDS,
        default="demo-sample",
        help="the demo stand to serve (default: demo-sample, whose camera images a "
        "synthetic plate; demo's camera images no sample)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where the PNGs and summary.txt go (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--width", type=int, default=1600, help="viewport width, px")
    parser.add_argument("--height", type=int, default=1000, help="viewport height, px")
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="do not run the chain of tools: the last two states are not reached",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep the temporary run document and scan frames, and print where",
    )
    return parser


def _say(text: str = "") -> None:
    print(text, flush=True)


def _remove_work(work: Path, keep: bool) -> None:
    if keep:
        _say(f"kept the run document and the scan frames: {work}")
        return
    try:
        shutil.rmtree(work)
    except OSError as exc:
        _say(f"note: could not remove {work} ({exc}); remove it by hand")


def _session(
    args: argparse.Namespace, pw: Any, profile: Path, out: Path, work: Path
) -> int:
    with pw.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except pw.Error as exc:
            message = str(exc)
            print(
                f"error: the browser did not start: {_first_line(message)}",
                file=sys.stderr,
            )
            hint = browser_hint(message)
            if hint:
                print(hint, file=sys.stderr)
            return 1
        try:
            return _serve_and_drive(args, pw, browser, profile, out, work)
        finally:
            browser.close()


def _serve_and_drive(
    args: argparse.Namespace,
    pw: Any,
    browser: Any,
    profile: Path,
    out: Path,
    work: Path,
) -> int:
    port = free_port()
    env = child_env(os.environ)
    stand = Stand(sys.executable, profile, work / "run.json", out / "chain.log", env)
    plans = plan_states(args.stand, results=not args.no_results)
    log = out / "server.log"
    records: list[StateRecord] = []
    browser_errors: list[str] = []
    warnings = 0
    failure: str | None = None
    with log.open("wb") as logfile:
        server = subprocess.Popen(
            server_command(sys.executable, profile, stand.run_doc, port),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=_NO_WINDOW,
        )
        outcome: StopOutcome | None = None
        try:
            failure = wait_for_serving(server, log)
            if failure is None:
                context = browser.new_context(
                    viewport={"width": args.width, "height": args.height}
                )
                try:
                    records, browser_errors, warnings = drive(
                        pw,
                        context,
                        port=port,
                        plans=plans,
                        stand_name=args.stand,
                        results=not args.no_results,
                        stand=stand,
                        out=out,
                    )
                except PageNotReadyError as exc:
                    failure = f"the page did not come up: {exc}"
                finally:
                    context.close()
        finally:
            outcome = stop_server(server)
    server_log = _read_log(log)
    if failure is not None:
        print(f"error: {failure}", file=sys.stderr)
        print(f"server: {outcome.text}", file=sys.stderr)
        return 1
    errors = server_errors(server_log)
    summary = out / "summary.txt"
    summary.write_text(
        format_summary(
            records,
            server=outcome.text,
            server_errors=errors,
            browser_errors=browser_errors,
            browser_warnings=warnings,
        ),
        encoding="utf-8",
    )
    code = exit_code(records, outcome, errors, browser_errors)
    for record in records:
        if record.reached:
            _say(f"{record.name}: reached")
        else:
            _say(f"{record.name}: not reached: {record.reason}")
    for name in STATE_NAMES:
        for path in _outputs(out, name):
            if path.is_file():
                _say(str(path))
    for path in (summary, log, out / "chain.log"):
        if path.is_file():
            _say(str(path))
    reached = sum(1 for record in records if record.reached)
    _say(
        f"ui-check: {reached} of {len(records)} states reached; server: {outcome.text}; "
        f"{len(errors)} server error line(s); {len(browser_errors)} browser error(s); "
        + ("nothing to read" if code == 0 else "read summary.txt and the PNGs")
    )
    return code


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")  # a redirected stream may be cp1252 (FM-20)
    args = build_parser().parse_args(argv)
    # A job started in the background of a non-interactive shell hands its children
    # SIGINT as ignored, and Python then installs no handler: the server would not
    # stop on Ctrl-C (measured, FM-160). A handler set here is the default again
    # in each child.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        profile = stand_profile(args.stand)
    except RefusedStandError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(
            f"error: smc is not importable ({exc}); run this script with the "
            "checkout's own .venv",
            file=sys.stderr,
        )
        return 1
    pw = load_playwright()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    clear_outputs(out)
    work = Path(tempfile.mkdtemp(prefix="smc-ui-check-"))
    try:
        return _session(args, pw, profile, out, work)
    except KeyboardInterrupt:
        print("interrupted: the server was stopped", file=sys.stderr)
        return 130
    finally:
        _remove_work(work, args.keep_work)


if __name__ == "__main__":
    sys.exit(main())
