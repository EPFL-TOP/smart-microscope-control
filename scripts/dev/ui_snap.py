"""Look at the plate page: serve it on a demo stand and write what a user sees.

Nothing else looks at the rendered page: the tests build it and click on it
without a browser, and a defect that only a Bokeh session shows (the typed range
of #277) passes all of them (FM-161). This script serves ``smc ui plate`` on a
simulator stand in dry run, drives it with a headless Chromium through the states
that matter, and writes a PNG of each, plus ``summary.txt`` with the texts a
canvas hides. A session reads the PNGs with its Read tool (``/ui-check``).

    python scripts/dev/ui_snap.py [--stand {demo-sample,demo}] [--out DIR]
                                  [--width 1280] [--height 800]
                                  [--no-results] [--keep-work]

The page can move a stage, so the script takes no profile: ``--stand`` is one of
the two demo stands, checked again against the repository's own file, which is
passed by absolute path (a name is looked up in the working directory and in
``$SMC_PROFILES``, and an operator's directory can shadow it). The page is always
served with ``--dry-run``. The chain of tools that fills the run document runs
*without* it, on the demo's simulated stage: in dry run the stage does not move,
every frame is plastic, and no state past the first three could be reached. The two
guided states (``wells-laid-out``, ``scanned``) press the guided screens' own buttons,
*Register the plate*, *Lay out the fields* and *Start the scan*, on the dry-run page, where
each is a run that moves nothing, and so does ``results-all``, which presses the Results
screen's *Run all that are left* over the chain's results (the report and the export, which
move nothing); the script never presses a tool's own *Run*.

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
#: Playwright waits 30 s for a disabled control by default: a state is not worth that
#: (FM-162).
DEFAULT_TIMEOUT_MS = 10_000
#: How long Connect may take to show on the strip: the demo opens in 17 ms, and the
#: strip follows on the next 1 s tick.
CONNECT_TIMEOUT_MS = 15_000
STEP_TIMEOUT_S = 120.0
SERVE_TIMEOUT_S = 60.0
READY_POLL_S = 0.1
#: A wait is sliced so that Python runs its Ctrl-C handler between two of them (FM-83).
WAIT_SLICE_S = 0.25
STOP_GRACE_S = 25.0
TERMINATE_GRACE_S = 10.0
KILL_GRACE_S = 5.0

#: The first four need no chain and are reached on both stands (``expert`` is fourth for
#: that: m5 §17.8); the next four need the results of a real run of the workflow
#: (``results-all`` presses *Run all that are left* over them, m5 §17.11.4); the next
#: two are the short-page shots of D12 (m5 §17.2, §17.8), reached alone again, after them; the
#: last two walk the Wells and Scan screens by their own buttons (m5 §17.10.4, #445): they
#: need a sample to register the plate from, so ``demo-sample`` only, and nothing of the chain.
ALONE_STATES = ("empty", "wells-selected", "snap", "expert")
CHAIN_STATES = ("after-density", "with-route", "results-all", "turned")
SHORT_STATES = ("stand-620", "plate-620")
GUIDED_STATES = ("wells-laid-out", "scanned")
STATE_NAMES = (*ALONE_STATES, *CHAIN_STATES, *SHORT_STATES, *GUIDED_STATES)
REACHABLE_ALONE = len(ALONE_STATES)

#: Why the guided states are not reached on the plain demo (m5 §17.10.4): its plate has no
#: references to register from, and a registration from three centres by eye is a hand's work.
NO_SAMPLE_TO_REGISTER = "the demo has no sample to register from"

#: The viewport the page is designed for (m5 §17.3): the strip, the step bar, both
#: figures and the drive bar are on screen at 1280 x 800, the floor of a stand PC.
VIEWPORT_WIDTH_PX = 1280
VIEWPORT_HEIGHT_PX = 800

#: The page of the owner's walk of #431: a 1280 x 800 window gave a 1280 x 623 page, and the
#: Stand and Plate screens must show their title, lead, fields and first action in it (m5
#: §17.2 D12). ``stand-620`` and ``plate-620`` shoot those two screens at this height.
SHORT_VIEWPORT_HEIGHT_PX = 620

#: The screens' titles, as the page draws them (``guide.SCREEN_TITLES``, which a test pins
#: this to: this script imports nothing of smc at import).
SCREEN_TITLE_OF = {
    "stand-620": "Which stand, and which preset?",
    "plate-620": "Where is the plate?",
    # The Scan screen's title names the plan's wells (``guide.scan_title``): the two wells
    # ``wells-laid-out`` types, in plate order.
    "wells-laid-out": "Which wells?",
    "scanned": "Scan B2 and B11",
    # The Results screen's title is the screen's own name (``guide.SCREEN_TITLES``).
    "results-all": "Results",
}

#: What the Results screen says (m5 §17.4.5, §17.11), as the page draws it and a test pins to
#: ``guide`` (this script imports nothing of smc at import): the five rows' plain names in the
#: checklist's order, the chain's button and the files block's heading. A row whose tool is done
#: has the state line ``Done <date> ...``; the chain's own line says ``Done: ...`` and does not
#: match ``DONE_ROW``.
RESULT_ROWS = (
    "Count the cells",
    "Density per well",
    "Route through the densest fields",
    "Report",
    "Export for NIS-Elements",
)
RUN_ALL = "Run all that are left"
FILES_HEADING = "Where the files are"
DONE_ROW = re.compile(r"^Done \d{4}-\d\d-\d\d")
#: The last control of the drive bar (m5 §17.12 D15): the *Drive* toggle, the lowest of the bar's
#: controls (the notes under it may be below the fold). Its name starts ``Drive``; nothing else
#: button-shaped does.
_DRIVE_TOGGLE = re.compile(r"^Drive\b")

# A child sharing the console can change its code page (FM-22).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ERROR_LINE = re.compile(r"\bERROR\b")
#: The strip of a closed stand (``<stand> | not connected``); the Connect row's own
#: line says ``is not connected:``, never this.
_CLOSED_STRIP = re.compile(r"\| not connected\b")
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
    step: str | None = None
    wells: str | None = None
    texts: list[str] = field(default_factory=list)
    on_screen: list[str] = field(default_factory=list)
    #: What a state is about, read out of its texts by name (``scanned``: the progress lines
    #: and the result line), so that the summary says them and a session does not hunt for them.
    facts: list[str] = field(default_factory=list)


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

    The four of ``CHAIN_STATES`` need the results of a real run of the workflow: the demo
    camera images no sample, ``--no-results`` skips the chain, and a chain step that failed
    leaves them without their input. The two short-page states of D12 need none, and are
    reached on both stands, last.
    """
    plans = [StatePlan(name, True) for name in ALONE_STATES]
    short = [StatePlan(name, True) for name in SHORT_STATES]
    # The guided states register the plate from the sample themselves and run no chain, so
    # they are reached with ``--no-results`` and after a chain that failed: ``demo-sample``
    # only, since the plain demo has nothing to register from.
    guided = [
        StatePlan(
            name,
            stand == "demo-sample",
            None if stand == "demo-sample" else NO_SAMPLE_TO_REGISTER,
        )
        for name in GUIDED_STATES
    ]
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
        return [
            *plans,
            *(StatePlan(n, False, reason) for n in CHAIN_STATES),
            *short,
            *guided,
        ]
    if chain_failure is not None:
        return [
            *plans,
            StatePlan("after-density", False, chain_failure),
            *(
                StatePlan(name, False, f"needs the chain: {chain_failure}")
                for name in CHAIN_STATES[1:]
            ),
            *short,
            *guided,
        ]
    return [*plans, *(StatePlan(n, True) for n in CHAIN_STATES), *short, *guided]


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
            lines.append(f"step: {record.step or '?'}")
            lines += [f"on screen: {part}" for part in record.on_screen]
            lines += list(record.facts)
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
    # A Windows child writes CRLF, and a text-mode write would turn each "\n" of it
    # into "\r\n" again: normalise, and write "\n" as it is on every system.
    output = output.replace("\r\n", "\n")
    seconds = time.monotonic() - started
    shown = "timeout" if code is None else str(code)
    with log.open("a", encoding="utf-8", newline="\n") as handle:
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
    #: The viewport of the run, which a short-page state puts back after its photograph.
    viewport: tuple[int, int] = (VIEWPORT_WIDTH_PX, VIEWPORT_HEIGHT_PX)

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
    """The page as it opens, with the stand closed (m5 §7): nothing to do."""


def _connect(page: Any) -> None:
    """Click *Connect*, then wait until the strip no longer says ``not connected``.

    Only an enabled button is clicked: Playwright waits out its timeout on a
    disabled one, and a later state would then start on a page that never
    connected (FM-162). The wait has its own bound for the same reason. The name is
    exact, since *Disconnect* contains *Connect*.
    """
    button = page.get_by_role("button", name="Connect", exact=True)
    if button.is_enabled():
        button.click()
    page.locator("pre").filter(has_text=_CLOSED_STRIP).wait_for(
        state="detached", timeout=CONNECT_TIMEOUT_MS
    )


def _step_button(page: Any, number: int, title: str) -> Any:
    """A step's button by its accessible name, which starts with its number (m5 §17.5).

    The label gains ``✓`` when the step is done, so the name is matched from its
    start: ``Wells`` alone is also the box's label and the *Plate* the map's, which is
    why a bare title was ambiguous (#395).
    """
    return page.get_by_role("button", name=re.compile(rf"^{number}\. {title}"))


def _act_wells_selected(page: Any, stand: Stand) -> None:
    _connect(page)
    # The *Wells* box is on the Wells screen only: its label is one match, which ends
    # #395's double one (the Wells screen's box and the expert view's list of it).
    _step_button(page, 3, "Wells").click()
    box = page.get_by_label("Wells", exact=True)
    box.fill("A1:B2")
    box.press("Enter")


def _act_snap(page: Any, stand: Stand) -> None:
    # The image is beside the map: Snap is always in view, and no screen changes.
    page.get_by_role("button", name="Snap").click()
    page.locator("pre").filter(has_text="snap: ").first.wait_for()


def _expert_switch(page: Any) -> Any:
    return page.get_by_role("button", name="Expert view")


def _act_expert(page: Any, stand: Stand) -> None:
    """The expert switch on, and the wait for it: a pressed Toggle's button has ``bk-active``.

    Measured: ``bk-btn bk-btn-light bk-active``. The photograph is taken by the caller
    and ``_after_expert`` then puts the switch off, so that the next state starts on a
    step's screen.
    """
    _expert_switch(page).click()
    page.locator("button.bk-active").filter(has_text="Expert view").wait_for()


def _after_expert(page: Any, stand: Stand) -> None:
    """The switch off again, after the photograph (m5 §17.8); waits for the class to go."""
    _expert_switch(page).click()
    page.locator("button.bk-active").filter(has_text="Expert view").wait_for(
        state="detached"
    )


def _act_after_density(page: Any, stand: Stand) -> None:
    stand.run_chain(CHAIN)
    # The page reads the file again at the end of a snap: a redraw of the map with
    # no Refresh, the case of #297 (a redraw made while the figure is out of view is
    # no longer possible, since neither figure is ever hidden, but the picture is still
    # read for the whole plate in view and for plain tick labels, FM-174). Then the
    # Results screen, whose density row says it is done: the second of its state lines that
    # start ``Done <date>`` (``cells`` is the first), which the screen draws when it swaps in
    # and the page reads the run document again (the checklist's ``✓ density`` is in the
    # expert view only now).
    page.get_by_role("button", name="Snap").click()
    _step_button(page, 5, "Results").click()
    page.locator("pre").filter(has_text=DONE_ROW).nth(1).wait_for()


def _act_with_route(page: Any, stand: Stand) -> None:
    stand.run_chain([ROUTE_STEP])
    # Refresh is in the map's header row, always in view.
    page.get_by_role("button", name="Refresh").click()


def _act_results_all(page: Any, stand: Stand) -> None:
    """*5. Results*, then *Run all that are left*, and the wait for the strip's own ``Run all that are left: done`` (m5 §17.11.4).

    After ``with-route`` the run document holds a real ``cells``, ``density`` and ``route``,
    so on this dry-run page the chain runs what is left, ``report`` and ``nis-export``: a run
    that moves nothing (measured). The button is the screen's own and not a tool's *Run*, which
    this script never clicks. Each wait is for the text the server draws, never for a delay
    (FM-162): the density row's ``Done`` before the click, so that the click is on the redrawn
    screen, and the outcome line after it. A chain in motion, with a step running and the others
    waiting, needs a stage that moves: the one-off session the skill describes.
    """
    _step_button(page, 5, "Results").click()
    page.locator("pre").filter(has_text=DONE_ROW).nth(1).wait_for()
    page.get_by_role("button", name=RUN_ALL, exact=True).click()
    page.locator("pre").filter(has_text="Run all that are left: done").first.wait_for()


def _act_turned(page: Any, stand: Stand) -> None:
    """The plate declared turned 180° on the *Plate lies* select (m5 §16.4).

    The chain registered the plate, so the switch is a run of ``plate-orientation``:
    the state waits for its outcome line on the strip, never for a delay (FM-162).
    Compared by eye with ``with-route``: the same cross, other names. The label is
    read from the select's own options, since it is worded from the stand's sides; a
    select with no turned option fails the state on ``select_option``, named.
    """
    _step_button(page, 2, "Plate").click()
    select = page.get_by_label("Plate lies", exact=True)
    # The click is answered by the server, which swaps the screen: the select is not
    # there yet, and `all_inner_texts` does not wait (an empty read gave the default
    # label, and a timeout on `select_option`).
    select.wait_for()
    labels = [text.strip() for text in select.locator("option").all_inner_texts()]
    turned = next((t for t in labels if t.startswith("Turned 180")), "Turned 180")
    select.select_option(label=turned)
    page.locator("pre").filter(has_text="Plate orientation done").first.wait_for()


def _short_page(page: Any) -> None:
    """The page at 1280 x 620, the height of the owner's walk (m5 §17.2 D12): the width stays the run's."""
    size = page.viewport_size
    width = size["width"] if isinstance(size, dict) else VIEWPORT_WIDTH_PX
    page.set_viewport_size({"width": width, "height": SHORT_VIEWPORT_HEIGHT_PX})


def _after_short(page: Any, stand: Stand) -> None:
    """The run's viewport again after a short-page photograph, so that the next state starts as the others did."""
    width, height = stand.viewport
    page.set_viewport_size({"width": width, "height": height})


def _open_screen(page: Any, number: int, title: str, heading: str) -> None:
    """Connect, forget the last outcome, shrink the page, open the screen of step ``number`` and wait until its title is drawn.

    D12 measures the screen under the header with its outcome lines empty: the states
    before this one leave a ``snap:`` line under the strip, and *Refresh* (always in view,
    in the map's header row) is what forgets it (m5 §8.8.3), as an operator would. The
    shrink comes before the screen is opened, so that it is laid out at the short height
    the state is about, never resized under a photograph. The wait is for the screen's own
    title text, which the server draws when it swaps the screen (FM-162).
    """
    _connect(page)
    page.get_by_role("button", name="Refresh").click()
    _short_page(page)
    _step_button(page, number, title).click()
    page.locator("pre").filter(has_text=heading).first.wait_for()


def _act_stand_620(page: Any, stand: Stand) -> None:
    """*1. Stand* after Connect, at 1280 x 620 (m5 §17.8): its title and *Load* are inside the page."""
    _open_screen(page, 1, "Stand", SCREEN_TITLE_OF["stand-620"])


def _act_plate_620(page: Any, stand: Stand) -> None:
    """*2. Plate* at 1280 x 620 (m5 §17.8): its title and first *Go to* are inside the page."""
    _open_screen(page, 2, "Plate", SCREEN_TITLE_OF["plate-620"])


def _declare_upright(page: Any) -> None:
    """The *Plate lies* select back to the option that is not turned, when ``turned`` left it turned.

    The sample's centres are A1 at the corner it prints: registering them with the plate
    declared turned 180° is refused (``the fit turns the plate by 0.0°: it lies turned 180°
    from the declared orientation``), and the ``turned`` state, which runs earlier on the same
    page, leaves it so. On a registered plate the switch is a run of ``plate-orientation``, and
    the state waits for its outcome line (FM-162); a select already upright is left alone, so
    that ``--no-results`` runs, which never turn it, wait for nothing.
    """
    select = page.get_by_label("Plate lies", exact=True)
    select.wait_for()
    labels = [text.strip() for text in select.locator("option").all_inner_texts()]
    upright = next((t for t in labels if t and not t.startswith("Turned")), None)
    current = str(select.evaluate("el => el.options[el.selectedIndex].text")).strip()
    if upright is None or current == upright:
        return
    select.select_option(label=upright)
    page.locator("pre").filter(has_text="Plate orientation done").first.wait_for()


def _act_wells_laid_out(page: Any, stand: Stand) -> None:
    """The guided path to a plan, by the screens' own buttons: Connect, *Register the plate*, the two wells, *Lay out the fields* (m5 §17.10.4).

    Served in dry run, so nothing moves: the registration runs from the sample's references
    and the plan is computed, both legal in a dry run. These are the page's own buttons and
    not a tool's *Run*, which this script never clicks. Each wait is for the text the server
    draws when its run ends, never for a delay (FM-162): the strip's outcome line for the
    registration, and the known block's count for the plan (a plan of the chain is stale after
    the registration, so the count is the new one's). ``Fields: <n> in 2 wells`` is the line.
    The plate is declared upright first (``_declare_upright``): ``turned`` leaves it turned.
    """
    _connect(page)
    _step_button(page, 2, "Plate").click()
    _declare_upright(page)
    page.get_by_role("button", name="Register the plate").click()
    page.locator("pre").filter(has_text="Registration done").first.wait_for()
    _step_button(page, 3, "Wells").click()
    box = page.get_by_label("Wells", exact=True)
    box.fill("B2,B11")
    box.press("Enter")
    page.get_by_role("button", name="Lay out the fields").click()
    page.locator("pre").filter(has_text="Field layout done").first.wait_for()
    page.locator("pre").filter(
        has_text=re.compile(r"Fields: \d+ in 2 wells")
    ).first.wait_for()


def _act_scanned(page: Any, stand: Stand) -> None:
    """*4. Scan*, then *Start the scan*, and the wait for the run's own outcome line (m5 §17.10.4).

    A dry-run page walks the fields and takes no frame, so the scan ends at once with
    ``Scan done in 0.0 s``: the bar, its lines and the result line are what the PNG shows, and
    a scan in motion is the one-off session on a stand that moves (the skill says so). The
    outcome text starts ``scan done``: ``scan-plan done`` does not contain it.
    """
    _step_button(page, 4, "Scan").click()
    page.get_by_role("button", name="Start the scan").click()
    page.locator("pre").filter(has_text="Scan done").first.wait_for()


ACTIONS = {
    "empty": _act_empty,
    "wells-selected": _act_wells_selected,
    "snap": _act_snap,
    "expert": _act_expert,
    "after-density": _act_after_density,
    "with-route": _act_with_route,
    "results-all": _act_results_all,
    "turned": _act_turned,
    "stand-620": _act_stand_620,
    "plate-620": _act_plate_620,
    "wells-laid-out": _act_wells_laid_out,
    "scanned": _act_scanned,
}

#: What a state does after its photograph: ``expert`` puts its switch off, and the two
#: short-page states put the viewport back.
AFTER_ACTIONS = {
    "expert": _after_expert,
    "stand-620": _after_short,
    "plate-620": _after_short,
}


def _frame_parts(page: Any) -> list[tuple[str, Any]]:
    """The parts of the page that must be in view without a scroll (m5 §17.3), by name.

    The strip with STOP, the step bar, both figures and the last control of the drive
    bar: the page is read for them at the viewport's size (1280 x 800 by default), which
    is the floor the layout is designed for.
    """
    return [
        ("STOP", page.get_by_role("button", name="STOP")),
        ("step bar", _step_button(page, 1, "Stand")),
        ("map", page.locator(".bk-Figure").first),
        ("image", page.locator(".bk-Figure").last),
        ("drive bar", page.get_by_role("button", name=_DRIVE_TOGGLE)),
    ]


def _short_parts(page: Any, state: str) -> list[tuple[str, Any]]:
    """The parts that must be in view on a short page (m5 §17.2 D12), by name: the strip, the step bar, and the screen's title and first action.

    Not the figures and the drive bar: at 620 px they are below the fold by design, and
    the screen's own fields are read by eye in the PNG. The first action of *Stand* is
    the preset's *Load*, and of *Plate* the first row's *Go to*.
    """
    first = (
        page.get_by_role("button", name="Load", exact=True)
        if state == "stand-620"
        else page.get_by_role("button", name=re.compile(r"^Go to "))
    )
    return [
        ("STOP", page.get_by_role("button", name="STOP")),
        ("step bar", _step_button(page, 1, "Stand")),
        ("title", page.locator("pre").filter(has_text=SCREEN_TITLE_OF[state])),
        ("first action", first),
    ]


def _guided_parts(page: Any, state: str) -> list[tuple[str, Any]]:
    """The frame's parts and the guided screen's own: its title and its first action (m5 §17.10.4), by name.

    The page is at the run's 1280 x 800: the Wells screen is taller than the Stand's, so the
    summary says whether its title and its first action, *Lay out the fields* or *Start the
    scan*, are inside the page, and the figures and the drive bar as for any other state.
    """
    action = "Lay out the fields" if state == "wells-laid-out" else "Start the scan"
    return [
        *_frame_parts(page),
        ("title", page.locator("pre").filter(has_text=SCREEN_TITLE_OF[state])),
        ("first action", page.get_by_role("button", name=action)),
    ]


def _results_parts(page: Any) -> list[tuple[str, Any]]:
    """The frame's parts and the Results screen's own: its title and *Run all that are left* (m5 §17.11.4), by name.

    The page is at the run's 1280 x 800: the summary says whether the title, which names the
    screen, and the chain's button, which is the first action, are inside the page. What is
    below the fold is the PNG's to say (``results-all-full.png``): the five rows are taller
    than the page, and the screen's own panel scrolls.
    """
    return [
        *_frame_parts(page),
        ("title", page.locator("pre").filter(has_text=re.compile(r"^Results$"))),
        ("first action", page.get_by_role("button", name=RUN_ALL, exact=True)),
    ]


#: Finds the sticky header (the strip, the outcome line and the step bar) wherever Panel put it,
#: through the shadow roots, and says its height in pixels; ``null`` when there is none.
_HEADER_HEIGHT_JS = """() => {
  const found = [];
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) {
      if (getComputedStyle(el).position === 'sticky') found.push(el);
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document);
  return found.length ? found[0].getBoundingClientRect().height : null;
}"""


def _header_height_px(page: Any) -> float | None:
    """The height of the sticky header in pixels, or ``None`` when the browser says none (m5 §17.12 D15).

    What the owner's walk measured as 180 px of a 623 px page. ``None`` for a page that cannot
    answer (a fake), so that nothing is guessed.
    """
    try:
        height = page.evaluate(_HEADER_HEIGHT_JS)
    except Exception:
        return None
    if isinstance(height, bool) or not isinstance(height, int | float):
        return None
    return float(height)


def _on_screen(page: Any, state: str | None = None) -> list[str]:
    """One line for each part of ``_frame_parts`` (``_short_parts`` for a short-page state, ``_guided_parts`` for a guided one, ``_results_parts`` for ``results-all``): whether it is wholly in the viewport, and where.

    A part that is not drawn, or whose box the browser does not give, says so and is
    never guessed to be in view. A page with no viewport size (a fake) gives nothing.
    """
    viewport = page.viewport_size
    if not isinstance(viewport, dict):
        return []
    lines: list[str] = []
    if state in SHORT_STATES:
        parts = _short_parts(page, state)
    elif state in GUIDED_STATES:
        parts = _guided_parts(page, state)
    elif state == "results-all":
        parts = _results_parts(page)
    else:
        parts = _frame_parts(page)
    for name, locator in parts:
        if locator.count() == 0:
            lines.append(f"{name}: not on the page")
            continue
        box = locator.first.bounding_box()
        if box is None:
            lines.append(f"{name}: not drawn")
            continue
        top, bottom = box["y"], box["y"] + box["height"]
        left, right = box["x"], box["x"] + box["width"]
        inside = (
            top >= 0
            and left >= 0
            and bottom <= viewport["height"]
            and right <= viewport["width"]
        )
        lines.append(
            f"{name}: {'yes' if inside else 'NO'} (y {top:.0f} to {bottom:.0f} of "
            f"{viewport['height']}, x {left:.0f} to {right:.0f} of {viewport['width']})"
        )
    header = _header_height_px(page)
    if header is not None:
        lines.append(f"header: {header:.0f} px")
    return lines


def _read_page(
    page: Any, state: str | None = None
) -> tuple[str | None, str | None, list[str], list[str]]:
    """The step shown, the Wells box, the text panes and what is on screen, as the page shows them.

    ``inner_text("body")`` of a Panel page is empty (shadow DOM); the panes' own
    ``pre`` elements are not. The step is the button of the class ``smc-step-shown``
    (m5 §17.6, measured on Panel 1.10.0 and 1.9.4); with the expert view on none is,
    and it says so.
    """
    shown = page.locator(".smc-step-shown button")
    if shown.count():
        step: str | None = shown.first.inner_text().strip()
    elif page.locator("button.bk-active").filter(has_text="Expert view").count():
        step = "(the expert view)"
    else:
        step = None
    box = page.get_by_label("Wells", exact=True)
    wells = box.input_value() if box.count() == 1 else None
    texts = [t.strip() for t in page.locator("pre").all_inner_texts() if t.strip()]
    return step, wells, texts, _on_screen(page, state)


def scan_facts(texts: Sequence[str]) -> list[str]:
    """What the ``scanned`` state is about, read from the page's texts: the progress lines and the result line (m5 §17.10.4).

    The progress block's text starts ``Field <n> of <m>`` and the result line ``Scanned``; the
    first pane of each, its lines joined by `` | ``. A pane the page did not draw is said as
    ``not on the page`` and never guessed: a state that reached no scan says so here.
    """

    def first(prefix: str) -> str:
        found = next((t for t in texts if t.startswith(prefix)), None)
        return "not on the page" if found is None else " | ".join(found.splitlines())

    return [f"progress: {first('Field ')}", f"result: {first('Scanned')}"]


def results_facts(texts: Sequence[str]) -> list[str]:
    """What ``results-all`` is about, read from the page's texts: the five rows' state lines, the chain's line and the files (m5 §17.11.4).

    The page lists a row's bold name and then its state line, so a row's state is the pane after
    its name; the chain's line is the first pane that starts ``Running``, ``Done:`` or
    ``Stopped``, and the files are the pane after the block's heading, its lines joined by
    `` | ``. A pane the page did not draw is said as ``not on the page`` and never guessed: a
    state that reached no chain says so here.
    """
    missing = "not on the page"
    lines: list[str] = []
    for name in RESULT_ROWS:
        at = next((i for i, text in enumerate(texts) if text == name), None)
        state = (
            missing
            if at is None or at + 1 >= len(texts)
            else " | ".join(texts[at + 1].splitlines())
        )
        lines.append(f"row {name}: {state}")
    chain = next(
        (t for t in texts if t.startswith(("Running ", "Done:", "Stopped "))), None
    )
    lines.append(
        f"chain: {missing if chain is None else ' | '.join(chain.splitlines())}"
    )
    heading = next((i for i, text in enumerate(texts) if text == FILES_HEADING), None)
    files = (
        missing
        if heading is None or heading + 1 >= len(texts)
        else " | ".join(texts[heading + 1].splitlines())
    )
    lines.append(f"files: {files}")
    return lines


def state_facts(name: str, texts: Sequence[str]) -> list[str]:
    """The facts a state is about, by name: ``scanned`` the scan's, ``results-all`` the chain's; none for the others."""
    if name == "scanned":
        return scan_facts(texts)
    if name == "results-all":
        return results_facts(texts)
    return []


def _outputs(out: Path, name: str) -> list[Path]:
    return [out / f"{name}.png", out / f"{name}-full.png", out / f"{name}-FAILED.png"]


def _remove(path: Path) -> None:
    """Delete a stale output. A file that Windows will not release (a viewer has it
    open, an orphaned server holds the log) is a note and the run goes on: the
    state that needs that name then fails by itself, named (FM-24)."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _say(f"note: could not remove {path} ({exc})")


def clear_outputs(out: Path) -> None:
    """Remove what an earlier run left, so that nothing in ``out`` is stale."""
    for name in STATE_NAMES:
        for path in _outputs(out, name):
            _remove(path)
    for name in ("summary.txt", "server.log", "chain.log"):
        _remove(out / name)


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
            step, wells, texts, on_screen = _read_page(page, plan.name)
            after = AFTER_ACTIONS.get(plan.name)
            if after is not None:
                after(page, stand)
        except (pw.Error, ChainFailedError) as exc:
            reason = (
                exc.args[0]
                if isinstance(exc, ChainFailedError)
                else _first_line(str(exc))
            )
            for stale in _outputs(out, plan.name)[:2]:
                _remove(stale)
            # A page that is gone has no picture: the reason is what there is to say.
            with contextlib.suppress(pw.Error):
                page.screenshot(path=str(out / f"{plan.name}-FAILED.png"))
            records.append(StateRecord(plan.name, False, reason))
            if isinstance(exc, ChainFailedError) and plan.name == "after-density":
                later = plan_states(stand_name, results=results, chain_failure=reason)
                remaining = list(later[len(records) :])
            continue
        records.append(
            StateRecord(
                plan.name,
                True,
                None,
                step,
                wells,
                texts,
                on_screen,
                state_facts(plan.name, texts),
            )
        )
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
    parser.add_argument(
        "--width",
        type=int,
        default=VIEWPORT_WIDTH_PX,
        help="viewport width, px (default: the 1280 the layout is designed for)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=VIEWPORT_HEIGHT_PX,
        help="viewport height, px (default: the 800 the layout is designed for)",
    )
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="do not run the chain of tools: after-density, with-route, results-all "
        "and turned are not reached",
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
    stand = Stand(
        sys.executable,
        profile,
        work / "run.json",
        out / "chain.log",
        env,
        (args.width, args.height),
    )
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
