"""scripts/dev/ui_snap.py looks at the plate page of a stand that can move a stage.

No browser and no ``smc ui`` here: the guard, the commands the script builds, the
waits and the stop are pinned on their own; the page itself is looked at by
running the script (``/ui-check``).
"""

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="a child cannot be sent Ctrl-C on Windows"
)

#: A child that says it is ready and then waits, for at most a minute so that a
#: failing test cannot leave it behind for long (FM-86). It restores the default
#: SIGINT itself: a suite started from a background shell would otherwise hand it
#: SIGINT as ignored, which is the very failure FM-160 describes.
_WAITS_FOR_SIGINT = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)
_IGNORES_SIGINT = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)
_WAITS = "import time\nprint('ready', flush=True)\ntime.sleep(60)\n"
#: What `smc ui` does on Ctrl-C: the interrupt is caught, the page is closed, exit 0.
_EXITS_ZERO_ON_SIGINT = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
    "print('ready', flush=True)\n"
    "try:\n"
    "    time.sleep(60)\n"
    "except KeyboardInterrupt:\n"
    "    sys.exit(0)\n"
)
_IGNORES_SIGINT_AND_SIGTERM = (
    "import signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


@pytest.fixture(scope="module")
def snap():
    spec = importlib.util.spec_from_file_location(
        "ui_snap_script", ROOT / "scripts" / "dev" / "ui_snap.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def children():
    """Real children of a test; whatever is still alive at the end is killed."""
    started: list[subprocess.Popen[bytes]] = []
    yield started
    for child in started:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


@pytest.fixture
def popen_log(snap, monkeypatch):
    """Every child the script starts, so that a test can ask whether it is dead.

    Whatever is still alive at the end is killed (FM-86: a failing test must not
    leave a plugin that sleeps for a minute behind).
    """
    started: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    class Recorded(real):  # type: ignore[valid-type, misc]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(snap.subprocess, "Popen", Recorded)
    yield started
    for child in started:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def _start(children, tmp_path: Path, code: str) -> subprocess.Popen[bytes]:
    """Start ``python -c code`` and return once it has printed ``ready``."""
    log = tmp_path / f"child-{len(children)}.log"
    with log.open("wb") as out:
        child = subprocess.Popen(
            [sys.executable, "-c", code], stdout=out, stderr=subprocess.STDOUT
        )
    children.append(child)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if "ready" in log.read_text(encoding="utf-8", errors="replace"):
            return child
        assert child.poll() is None, "the child ended before it was ready"
        time.sleep(0.05)
    raise AssertionError("the child was not ready within 30 s")


class _Untouchable:
    """A profiles directory that fails the test when anything is built from it."""

    def __truediv__(self, other: object) -> Path:
        raise AssertionError(f"the guard read a file ({other!r}) before it refused")


class _Clock:
    """The clock a module reads, owned by the test (FM-44): a wait moves it."""

    def __init__(self, limit: int = 200) -> None:
        self.now = 0.0
        self.calls = 0
        self.limit = limit

    def step(self, seconds: float) -> None:
        self.calls += 1
        assert self.calls <= self.limit, "the code under test never ends"
        self.now += seconds

    def namespace(self, sleeps: list[float] | None = None) -> SimpleNamespace:
        def sleep(seconds: float) -> None:
            if sleeps is not None:
                sleeps.append(seconds)
            self.step(seconds)

        return SimpleNamespace(monotonic=lambda: self.now, sleep=sleep)


class _SlicedProc:
    """A ``Popen`` whose ``wait`` blocks for its whole timeout, as Windows does."""

    def __init__(self, clock: _Clock, ends_after: int | None) -> None:
        self.clock = clock
        self.ends_after = ends_after
        self.timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.timeouts.append(timeout)
        assert timeout is not None, "a wait with no limit"
        if self.ends_after is not None and len(self.timeouts) > self.ends_after:
            return 0
        self.clock.step(timeout)
        raise subprocess.TimeoutExpired("child", timeout)


class _Child:
    """A child as ``poll`` sees it: ``None`` for each entry of ``states``, then an exit code."""

    def __init__(self, *polls: int | None) -> None:
        self.polls = list(polls)

    def poll(self) -> int | None:
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]


class _Immortal:
    """A child that no signal ends: ``wait`` always times out, on the test's clock."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.sent: list[int] = []
        self.terminated = False
        self.killed = False

    def poll(self) -> None:
        return None

    def send_signal(self, number: int) -> None:
        self.sent.append(number)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        assert timeout is not None, "a wait with no limit"
        self.clock.step(timeout)
        raise subprocess.TimeoutExpired("child", timeout)


class _FakeError(Exception):
    """Stands for ``playwright.sync_api.Error`` in the driver's own tests."""


def _fake_page(
    *, fails_on_button: str | None = None, fails_on_fill: bool = False
) -> MagicMock:
    """A page on which every control is there, and the button whose name is
    ``fails_on_button`` (a string, or a step's ``"3. Wells"``) times out when it is
    clicked (or the Wells box when it is filled, with ``fails_on_fill``), with the
    first line Playwright gives."""
    page = MagicMock()
    page.locator.return_value.count.return_value = 0
    page.locator.return_value.all_inner_texts.return_value = ["a pane"]
    page.get_by_label.return_value.count.return_value = 0
    if fails_on_fill:
        page.get_by_label.return_value.fill.side_effect = _FakeError(
            "Locator.fill: Timeout 10000ms exceeded.\nCall log:\n  - waiting"
        )

    def get_by_role(role: str, name: object = None, **kwargs: object) -> MagicMock:
        found = MagicMock()
        text = getattr(name, "pattern", name)
        # A step's name is a regular expression, written ``^3\. Wells``: compared as
        # the name a user reads, ``3. Wells``.
        text = str(text).removeprefix("^").replace("\\.", ".")
        if fails_on_button is not None and text == fails_on_button:
            found.click.side_effect = _FakeError(
                "Locator.click: Timeout 10000ms exceeded.\nCall log:\n  - waiting"
            )
        return found

    page.get_by_role.side_effect = get_by_role
    return page


def _drive(snap, tmp_path: Path, page: MagicMock, run_chain):
    """``drive`` on a fake page, with the chain replaced by ``run_chain``."""
    context = MagicMock()
    context.new_page.return_value = page
    return snap.drive(
        SimpleNamespace(Error=_FakeError),
        context,
        port=1,
        plans=snap.plan_states("demo-sample", results=True),
        stand_name="demo-sample",
        results=True,
        stand=SimpleNamespace(run_chain=run_chain, viewport=(1280, 800)),
        out=tmp_path,
    )


# --- the guard -------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "nikon-ti2",
        "nikon-tie",
        "Demo",
        "demo-sample.toml",
        "../profiles/nikon-ti2",
        "",
        str(ROOT / "profiles" / "nikon-ti2.toml"),
    ],
)
def test_the_guard_refuses_every_stand_but_the_two_demos(snap, name) -> None:
    with pytest.raises(snap.RefusedStandError, match="not one of the demo stands"):
        snap.stand_profile(name, profiles_dir=_Untouchable())


def test_the_guard_refuses_a_demo_named_profile_that_names_a_config(
    snap, tmp_path
) -> None:
    text = (ROOT / "profiles" / "demo.toml").read_text(encoding="utf-8")
    assert 'config = ""' in text
    # The file exists, so the loader accepts the profile and the guard's own
    # check is the one that refuses it.
    (tmp_path / "x.cfg").write_text("", encoding="utf-8")
    (tmp_path / "demo.toml").write_text(
        text.replace('config = ""', 'config = "x.cfg"'), encoding="utf-8"
    )
    with pytest.raises(snap.RefusedStandError, match="Micro-Manager configuration"):
        snap.stand_profile("demo", profiles_dir=tmp_path)


def test_the_guard_refuses_a_demo_file_whose_profile_has_another_name(
    snap, tmp_path
) -> None:
    text = (ROOT / "profiles" / "demo.toml").read_text(encoding="utf-8")
    assert 'name = "demo"' in text
    (tmp_path / "demo.toml").write_text(
        text.replace('name = "demo"', 'name = "nikon-ti2"'), encoding="utf-8"
    )
    with pytest.raises(snap.RefusedStandError, match="names itself 'nikon-ti2'"):
        snap.stand_profile("demo", profiles_dir=tmp_path)


def test_the_guard_refuses_a_stand_whose_file_is_missing_or_does_not_load(
    snap, tmp_path
) -> None:
    with pytest.raises(snap.RefusedStandError, match="does not exist"):
        snap.stand_profile("demo", profiles_dir=tmp_path)
    (tmp_path / "demo.toml").write_text("[microscope\n", encoding="utf-8")
    with pytest.raises(snap.RefusedStandError, match="does not load"):
        snap.stand_profile("demo", profiles_dir=tmp_path)


@pytest.mark.parametrize("name", ["demo", "demo-sample"])
def test_the_guard_returns_the_repos_own_file_by_absolute_path(snap, name) -> None:
    path = snap.stand_profile(name)
    assert path.is_absolute()
    assert path == snap.PROFILES_DIR / f"{name}.toml"
    assert snap.PROFILES_DIR.is_relative_to(ROOT)
    assert path.is_file()


def test_the_allowed_stands_are_the_two_demos_and_nothing_else(snap) -> None:
    assert snap.ALLOWED_STANDS == ("demo-sample", "demo")


# --- the commands ----------------------------------------------------------


def test_the_server_always_runs_dry_on_the_guarded_profile(snap) -> None:
    profile = snap.stand_profile("demo-sample")
    command = snap.server_command("py", profile, Path("/work/run.json"), 5555)
    assert command[:5] == ["py", "-m", "smc.cli", "ui", "plate"]
    assert "--dry-run" in command
    assert command[command.index("-p") + 1] == str(profile)
    assert command[command.index("--run-doc") + 1] == str(Path("/work/run.json"))
    assert command[command.index("--port") + 1] == "5555"


def test_the_child_does_not_inherit_the_operators_profile(snap) -> None:
    base = {
        "SMC_PROFILE": "nikon-ti2",
        "SMC_PROFILES": "/site/profiles",
        "PATH": "/bin",
        "PYTHONIOENCODING": "cp1252",
    }
    env = snap.child_env(base)
    assert "SMC_PROFILE" not in env
    assert "SMC_PROFILES" not in env
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PATH"] == "/bin"
    assert base["SMC_PROFILE"] == "nikon-ti2"  # the caller's mapping is left alone


def test_the_chain_is_the_documented_workflow_on_the_demo_sample_stand(snap) -> None:
    profile = snap.stand_profile("demo-sample")
    run_doc = Path("/work/run.json")
    assert [step.plugin for step in snap.CHAIN] == [
        "plate-registration",
        "scan-plan",
        "scan",
        "cells",
        "density",
    ]
    assert snap.ROUTE_STEP.plugin == "route"
    for step in [*snap.CHAIN, snap.ROUTE_STEP]:
        command = snap.run_command("py", step, profile, run_doc)
        assert command[:5] == ["py", "-m", "smc.cli", "run", step.plugin]
        assert "--dry-run" not in command  # the chain writes what a real run writes
        assert command[command.index("-p") + 1] == str(profile)
        assert command[command.index("--run-doc") + 1] == str(run_doc)
    registration = snap.run_command("py", snap.CHAIN[0], profile, run_doc)
    refs = [c for c in registration if c.startswith("refs=")]
    assert len(refs) == 1
    corners = {
        ref["well"]: (ref["x_um"], ref["y_um"])
        for ref in json.loads(refs[0].removeprefix("refs="))
    }
    assert corners == {"A1": (-49500, 31500), "H12": (49500, -31500)}
    cells = snap.run_command("py", snap.CHAIN[3], profile, run_doc)
    assert "detector=blobs" in cells
    route = snap.run_command("py", snap.ROUTE_STEP, profile, run_doc)
    labels = [c for c in route if c.startswith("labels=")]
    assert len(labels) == 1
    assert json.loads(labels[0].removeprefix("labels=")) == ["single", "doublet"]


def test_the_demo_sample_profile_still_fits_the_chains_refs(snap) -> None:
    from smc.hardware.profile import Profile

    sample = Profile.load(snap.stand_profile("demo-sample")).sample
    assert sample is not None
    assert sample.plate == "96-well"
    assert sample.a1_center_xy_um is None
    assert sample.rotation_deg == 0


# --- the states ------------------------------------------------------------


def test_a_state_that_cannot_be_reached_is_named_with_its_reason(snap) -> None:
    names = [
        "empty",
        "wells-selected",
        "snap",
        "expert",
        "after-density",
        "with-route",
        # The Results screen's chain (m5 §17.11.4): between the route and the turn.
        "results-all",
        "turned",
        # The short-page shots of D12 (m5 §17.8): after them, and reached with no chain.
        "stand-620",
        "plate-620",
        # The Wells and Scan screens by their own buttons (m5 §17.10.4, #445): last.
        "wells-laid-out",
        "scanned",
    ]
    assert list(snap.STATE_NAMES) == names
    assert snap.CHAIN_STATES == ("after-density", "with-route", "results-all", "turned")
    # `expert` is the fourth, so that both stands reach it (m5 §17.8).
    alone = [True, True, True, True]
    chained = [False, False, False, False]
    short = [True, True]
    # The guided states register the plate from the sample themselves and need no chain.
    guided = [True, True]
    no_sample = [False, False]

    full = snap.plan_states("demo-sample", results=True)
    assert [p.name for p in full] == names
    assert all(p.reachable and p.reason is None for p in full)

    demo = snap.plan_states("demo", results=True)
    assert [p.name for p in demo] == names
    # `turned` needs the chain's registered plate (m5 §16.4): with the results. The two
    # short-page states need none: both stands reach them. The guided ones need a sample.
    assert [p.reachable for p in demo] == [*alone, *chained, *short, *no_sample]
    for plan in demo[4:8]:
        assert plan.reason == (
            "the demo camera images no sample: no cells, no density; "
            "use --stand demo-sample"
        )
    assert [p.reason for p in demo[10:]] == [
        "the demo has no sample to register from"
    ] * 2
    assert snap.NO_SAMPLE_TO_REGISTER == "the demo has no sample to register from"

    # With no chain, or after a chain that failed, the guided states are still reached:
    # they do not read the chain's results.
    skipped = snap.plan_states("demo-sample", results=False)
    assert [p.reachable for p in skipped] == [*alone, *chained, *short, *guided]
    assert {p.reason for p in skipped[4:8]} == {"skipped by --no-results"}
    assert [p.reason for p in skipped[10:]] == [None, None]

    failed = snap.plan_states(
        "demo-sample", results=True, chain_failure="cells exited 1: no detector"
    )
    assert [p.reachable for p in failed] == [*alone, *chained, *short, *guided]
    assert [p.name for p in failed[4:8]] == [
        "after-density",
        "with-route",
        "results-all",
        "turned",
    ]
    assert failed[4].reason == "cells exited 1: no detector"
    assert failed[5].reason is not None
    assert "cells exited 1: no detector" in failed[5].reason
    assert failed[6].reason == failed[5].reason
    assert failed[7].reason == failed[5].reason
    # None is silent: every state that is not reached says why.
    for plans in (demo, skipped, failed):
        assert all(p.reachable or p.reason for p in plans)


def test_the_exit_code_is_3_for_anything_the_session_must_read(snap) -> None:
    reached = snap.StateRecord(name="empty", reached=True)
    missed = snap.StateRecord(name="snap", reached=False, reason="boom")
    clean = snap.StopOutcome("sigint", 0, 0.1, "stopped by SIGINT, exit 0", True)
    dirty = snap.StopOutcome("sigint", -2, 0.1, "stopped by SIGINT, exit -2", False)
    assert snap.exit_code([reached], clean, [], []) == 0
    assert snap.exit_code([reached, missed], clean, [], []) == 3
    assert snap.exit_code([reached], dirty, [], []) == 3
    assert snap.exit_code([reached], clean, ["ERROR x"], []) == 3
    assert snap.exit_code([reached], clean, [], ["TypeError: y"]) == 3


# --- the processes ---------------------------------------------------------


def test_wait_for_serving_finds_the_line_in_the_log(snap, tmp_path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "booting\nserving the plate page on http://localhost:5006/ - Ctrl-C stops it\n",
        encoding="utf-8",
    )
    assert snap.wait_for_serving(_Child(None), log, timeout_s=2.0) is None


def test_wait_for_serving_reports_a_child_that_ended_at_once(
    snap, tmp_path, monkeypatch
) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(f"line {n}" for n in range(30))
        + "\nNo profile named 'demo-sample'\n",
        encoding="utf-8",
    )
    clock = _Clock()
    sleeps: list[float] = []
    monkeypatch.setattr(snap, "time", clock.namespace(sleeps))
    why = snap.wait_for_serving(_Child(None, None, 1), log, timeout_s=2.0)
    assert why is not None
    assert "exit 1" in why
    assert "No profile named 'demo-sample'" in why
    assert "line 9" not in why  # the last 20 lines, not the whole log
    assert "line 29" in why
    # It looked twice more, and not for the 2 s the timeout allowed.
    assert sleeps == [0.1, 0.1]


def test_wait_for_serving_gives_up_at_its_timeout(snap, tmp_path, monkeypatch) -> None:
    log = tmp_path / "server.log"
    log.write_text("starting\n", encoding="utf-8")
    clock = _Clock()
    monkeypatch.setattr(snap, "time", clock.namespace())
    why = snap.wait_for_serving(_Child(None), log, timeout_s=2.0)
    assert why is not None
    assert "within 2 s" in why
    assert "starting" in why


def test_waits_are_sliced_so_that_ctrl_c_gets_through(snap, monkeypatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(snap, "time", clock.namespace())
    proc = _SlicedProc(clock, ends_after=2)
    assert snap.wait_in_slices(proc, 60.0) == 0
    assert len(proc.timeouts) == 3
    assert all(t is not None and t <= 0.5 for t in proc.timeouts)


def test_a_wait_in_slices_ends_at_its_deadline(snap, monkeypatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(snap, "time", clock.namespace())
    proc = _SlicedProc(clock, ends_after=None)
    assert snap.wait_in_slices(proc, 1.0) is None
    assert proc.timeouts == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_a_ctrl_c_during_the_wait_is_not_swallowed(snap) -> None:
    class Interrupted:
        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        snap.wait_in_slices(Interrupted(), 60.0)


@POSIX_ONLY
def test_stop_server_sends_sigint_and_reports_how_it_ended(
    snap, children, tmp_path
) -> None:
    child = _start(children, tmp_path, _WAITS_FOR_SIGINT)
    outcome = snap.stop_server(child, windows=False, grace_s=20.0)
    assert outcome.how == "sigint"
    # Killed by the interrupt, its `finally` never run: an end, but not a clean one.
    assert outcome.returncode in (-2, 130)
    assert not outcome.clean
    assert outcome.text.startswith("stopped by SIGINT, exit ")
    assert child.poll() is not None


@POSIX_ONLY
def test_a_server_that_exits_0_on_sigint_is_a_clean_stop(
    snap, children, tmp_path
) -> None:
    child = _start(children, tmp_path, _EXITS_ZERO_ON_SIGINT)
    outcome = snap.stop_server(child, windows=False, grace_s=20.0)
    assert outcome.how == "sigint"
    assert outcome.returncode == 0
    assert outcome.clean
    assert outcome.text.startswith("stopped by SIGINT, exit 0 in ")
    assert child.poll() == 0


@POSIX_ONLY
def test_stop_server_terminates_a_child_that_ignores_sigint(
    snap, children, tmp_path
) -> None:
    child = _start(children, tmp_path, _IGNORES_SIGINT)
    outcome = snap.stop_server(child, windows=False, grace_s=0.5)
    assert outcome.how == "terminated"
    assert outcome.returncode == -signal.SIGTERM
    assert not outcome.clean
    assert "did not end on SIGINT within 0.5 s; terminated" in outcome.text
    assert child.poll() is not None


@POSIX_ONLY
def test_stop_server_kills_a_child_that_ignores_sigint_and_sigterm(
    snap, children, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(snap, "TERMINATE_GRACE_S", 0.5)
    child = _start(children, tmp_path, _IGNORES_SIGINT_AND_SIGTERM)
    outcome = snap.stop_server(child, windows=False, grace_s=0.5)
    assert outcome.how == "killed"
    assert outcome.returncode == -signal.SIGKILL
    assert not outcome.clean
    assert outcome.text == (
        "did not end on SIGINT within 0.5 s; "
        "did not end on terminate() within 0.5 s; killed, exit -9"
    )
    assert child.poll() is not None


def test_stop_server_says_when_a_child_cannot_be_killed_and_does_not_wait_for_it(
    snap, monkeypatch
) -> None:
    """FM-35: ``kill()`` is sent, the wait after it ends, and the text says the child may be there."""
    clock = _Clock()
    monkeypatch.setattr(snap, "time", clock.namespace())
    child = _Immortal(clock)
    outcome = snap.stop_server(child, windows=False, grace_s=1.0)
    assert outcome.how == "unkillable"
    assert child.sent == [signal.SIGINT]
    assert child.terminated
    assert child.killed
    assert not outcome.clean
    assert "may still be running" in outcome.text


def test_stop_server_on_windows_terminates_and_says_that_ctrl_c_was_not_tried(
    snap, children, tmp_path
) -> None:
    child = _start(children, tmp_path, _WAITS)
    outcome = snap.stop_server(child, windows=True, grace_s=0.5)
    assert outcome.how == "terminated-windows"
    assert "Ctrl-C path was not exercised" in outcome.text
    assert outcome.clean
    assert child.poll() is not None


def test_stop_server_says_when_the_server_had_already_ended(snap, tmp_path) -> None:
    done = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    done.wait(timeout=30)
    outcome = snap.stop_server(done, windows=False)
    assert outcome.how == "ended"
    assert outcome.text == "had already ended with exit 7"
    assert not outcome.clean


def test_a_step_that_fails_is_a_finding_with_its_last_line(snap, tmp_path) -> None:
    log = tmp_path / "chain.log"
    code = "print('first'); print('boom: no frame'); raise SystemExit(3)"
    outcome = snap.run_step(
        [sys.executable, "-c", code],
        label="cells",
        log=log,
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_s=60.0,
    )
    assert outcome.exit_code == 3
    assert not outcome.ok
    assert outcome.last_line == "boom: no frame"
    assert outcome.failure() == "cells exited 3: boom: no frame"
    text = log.read_text(encoding="utf-8")
    assert text.startswith("$ cells exit 3 in ")
    assert "boom: no frame" in text


def test_a_step_that_passes_is_appended_to_the_chain_log(snap, tmp_path) -> None:
    log = tmp_path / "chain.log"
    for label in ("scan", "density"):
        outcome = snap.run_step(
            [sys.executable, "-c", "print('ok')"],
            label=label,
            log=log,
            cwd=tmp_path,
            env=dict(os.environ),
            timeout_s=60.0,
        )
        assert outcome.ok
        assert outcome.failure() is None
    text = log.read_text(encoding="utf-8")
    assert "$ scan exit 0" in text
    assert "$ density exit 0" in text
    assert text.index("$ scan") < text.index("$ density")


def test_a_step_that_hangs_is_killed_and_named(snap, tmp_path, popen_log) -> None:
    outcome = snap.run_step(
        [sys.executable, "-c", _WAITS],
        label="scan",
        log=tmp_path / "chain.log",
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_s=1.0,
    )
    assert outcome.timed_out
    assert not outcome.ok
    failure = outcome.failure()
    assert failure is not None
    assert failure.startswith("scan did not end within 1 s")
    # A hung `scan` left running would go on driving the stage and rewriting the
    # run document while the script moves on to the next state.
    assert len(popen_log) == 1
    assert popen_log[0].poll() is not None


def test_a_ctrl_c_during_a_step_leaves_no_child_behind(
    snap, tmp_path, popen_log, monkeypatch
) -> None:
    """FM-83: the interrupt arrives from the first wait, against a real child."""
    real_wait = snap.wait_in_slices
    waits: list[float] = []

    def interrupted_once(child, timeout_s, slice_s=snap.WAIT_SLICE_S):
        waits.append(timeout_s)
        if len(waits) == 1:
            raise KeyboardInterrupt
        return real_wait(child, timeout_s, slice_s)

    monkeypatch.setattr(snap, "wait_in_slices", interrupted_once)
    with pytest.raises(KeyboardInterrupt):
        snap.run_step(
            [sys.executable, "-c", _WAITS],
            label="scan",
            log=tmp_path / "chain.log",
            cwd=tmp_path,
            env=dict(os.environ),
            timeout_s=60.0,
        )
    assert len(popen_log) == 1
    assert popen_log[0].poll() is not None


def test_a_child_that_writes_crlf_leaves_single_line_ends_in_the_chain_log(
    snap, tmp_path
) -> None:
    """A Windows child writes CRLF; a text-mode append would double the CR (`\\r\\r\\n`)."""
    log = tmp_path / "chain.log"
    code = "import sys; sys.stdout.buffer.write(b'a\\r\\nb\\r\\n')"
    outcome = snap.run_step(
        [sys.executable, "-c", code],
        label="cells",
        log=log,
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_s=60.0,
    )
    assert outcome.last_line == "b"
    data = log.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"a\nb\n")


# --- what is read back -----------------------------------------------------

# What typing `A1:B2` in the Wells box wrote to the server's log (#277), copied from
# the server.log of the script's first run, with the frames between cut out. Four
# lines name the failure, two from Panel's callback and two from Bokeh's.
_WELLS_LOG = (
    "serving the plate page on http://localhost:62757/ - Ctrl-C stops it\n"
    "2026-10-29 20:59:27,112 ERROR: panel.reactive - Callback failed for object "
    "named 'Wells' changing properties {'value': 'A1:B2', 'value_input': 'A1:B2'} \n"
    "Traceback (most recent call last):\n"
    '  File "panel/reactive.py", line 491, in _process_events\n'
    "    self.param.update(**self_params)\n"
    "RuntimeError: _pending_writes should be non-None when we have a document lock, "
    "and we should have the lock when the document changes\n"
    "ERROR:bokeh.util.asyncio:Error thrown from callback:\n"
    "ERROR:bokeh.util.asyncio:Traceback (most recent call last):\n"
    '  File "bokeh/server/session.py", line 297, in unlocked_callback\n'
    "RuntimeError: _pending_writes should be non-None when we have a document lock, "
    "and we should have the lock when the document changes\n"
)


def test_server_errors_finds_the_wells_traceback_and_ignores_a_clean_log(snap) -> None:
    found = snap.server_errors(_WELLS_LOG)
    assert len(found) == 4
    assert "Callback failed for object named 'Wells'" in found[0]
    assert found[1] == "Traceback (most recent call last):"
    assert found[2] == "ERROR:bokeh.util.asyncio:Error thrown from callback:"
    assert found[3].startswith("ERROR:bokeh.util.asyncio:Traceback")
    clean = "serving the plate page on http://localhost:5006/ - Ctrl-C stops it\n"
    assert snap.server_errors(clean) == []
    # A word that only contains the letters is not a log level.
    assert snap.server_errors("TERRORIST mentioned\nNo ERRORS here\n") == []


def test_a_missing_browser_says_what_to_run(snap) -> None:
    message = (
        "BrowserType.launch: Executable doesn't exist at "
        "/home/u/.cache/ms-playwright/chromium_headless_shell-1248/headless_shell\n"
        "Looks like Playwright was just installed or updated.\n"
        "Please run the following command to download new browsers:\n"
        "    playwright install\n"
    )
    hint = snap.browser_hint(message)
    assert hint is not None
    assert "python -m playwright install chromium" in hint
    assert "--with-deps" in hint
    assert snap.browser_hint("BrowserType.launch: Target closed") is None


def test_a_missing_playwright_says_to_install_the_dev_extra(
    snap, monkeypatch, capsys
) -> None:
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(SystemExit) as raised:
        snap.load_playwright()
    assert raised.value.code == 1
    assert 'pip install -e ".[dev]"' in capsys.readouterr().err


def test_the_summary_has_one_block_per_state_and_names_what_it_could_not_read(
    snap,
) -> None:
    reached = snap.StateRecord(
        name="empty",
        reached=True,
        step="3. Wells",
        wells="",
        texts=["idle", "no run yet"],
        on_screen=["STOP: yes (y 8 to 40 of 800, x 1100 to 1160 of 1280)"],
    )
    unread = snap.StateRecord(name="snap", reached=True, texts=[])
    missed = snap.StateRecord(
        name="after-density", reached=False, reason="cells exited 1: no frame"
    )
    text = snap.format_summary(
        [reached, unread, missed],
        server="stopped by SIGINT, exit 0 in 0.1 s",
        server_errors=["ERROR a", "Traceback (most recent call last):", "x", "y"],
        browser_errors=[],
        browser_warnings=3,
    )
    assert [line for line in text.splitlines() if line.startswith("== ")] == [
        "== empty ==",
        "== snap ==",
        "== after-density ==",
    ]
    assert (
        "== empty ==\nreached\nstep: 3. Wells\n"
        "on screen: STOP: yes (y 8 to 40 of 800, x 1100 to 1160 of 1280)\n"
        "Wells: ''\n"
    ) in text
    assert "idle\n\nno run yet" in text
    assert "== snap ==\nreached\nstep: ?\n" in text
    assert "active tab" not in text
    assert "Wells:" not in text.split("== snap ==")[1].split("== after")[0]
    assert "== after-density ==\nnot reached: cells exited 1: no frame" in text
    assert "server: stopped by SIGINT, exit 0 in 0.1 s" in text
    assert (
        "server errors: 4\n  ERROR a\n  Traceback (most recent call last):\n  x\n"
        in text
    )
    assert "  y" not in text  # the first three lines only
    assert "browser errors: 0" in text
    assert "browser warnings: 3" in text
    assert text.endswith("\n")


# --- the script as a command ----------------------------------------------


def test_help_needs_no_playwright(tmp_path) -> None:
    """``--help`` runs in a process where importing playwright is an error."""
    script = ROOT / "scripts" / "dev" / "ui_snap.py"
    code = (
        "import runpy, sys\n"
        "sys.modules['playwright'] = None\n"
        "sys.modules['playwright.sync_api'] = None\n"
        f"sys.argv = ['ui_snap.py', '--help']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--stand" in proc.stdout
    assert "--no-results" in proc.stdout


def test_a_stand_that_is_not_a_demo_is_refused_by_the_command_line(snap) -> None:
    with pytest.raises(SystemExit) as raised:
        snap.main(["--stand", "nikon-ti2"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        snap.main(["--profile", "nikon-ti2"])
    assert raised.value.code == 2


def test_a_refused_profile_file_ends_the_run_with_exit_2_and_writes_nothing(
    snap, tmp_path, monkeypatch, capsys
) -> None:
    """The guard's refusal reaches the exit code, and SIGINT is made a default again.

    ``main`` sets the default handler for itself before anything else (FM-160):
    the test starts it as a background shell would, with SIGINT ignored.
    """

    def refuse(name: str) -> Path:
        raise snap.RefusedStandError(
            "demo.toml names a Micro-Manager configuration ('x.cfg')"
        )

    monkeypatch.setattr(snap, "stand_profile", refuse)
    out = tmp_path / "out"
    before = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        code = snap.main(["--stand", "demo", "--out", str(out)])
        handler = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, before)
    assert code == 2
    assert handler is signal.default_int_handler
    assert (
        "refused: demo.toml names a Micro-Manager configuration"
        in capsys.readouterr().err
    )
    assert not out.exists()


# --- what an earlier run leaves ---------------------------------------------


def _output_names(snap) -> list[str]:
    names = [
        f"{state}{suffix}"
        for state in snap.STATE_NAMES
        for suffix in (".png", "-full.png", "-FAILED.png")
    ]
    return [*names, "summary.txt", "server.log", "chain.log"]


def test_clear_outputs_removes_every_fixed_name_and_nothing_else(
    snap, tmp_path
) -> None:
    names = _output_names(snap)
    for name in [*names, "notes.txt"]:
        (tmp_path / name).write_text("old", encoding="utf-8")
    snap.clear_outputs(tmp_path)
    assert [name for name in names if (tmp_path / name).exists()] == []
    assert (tmp_path / "notes.txt").exists()


def test_a_file_that_cannot_be_removed_is_a_note_and_the_rest_are_removed(
    snap, tmp_path, monkeypatch, capsys
) -> None:
    """Windows refuses to delete a file that a viewer holds open (FM-24)."""
    names = _output_names(snap)
    for name in names:
        (tmp_path / name).write_text("old", encoding="utf-8")
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "empty.png":
            raise PermissionError(13, "in use")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    snap.clear_outputs(tmp_path)
    said = capsys.readouterr().out
    assert "note: could not remove" in said
    assert "empty.png" in said
    assert [name for name in names if (tmp_path / name).exists()] == ["empty.png"]


# --- the driver, on a fake page ---------------------------------------------


def test_a_failed_chain_names_the_states_that_needed_it(snap, tmp_path) -> None:
    def run_chain(steps) -> None:
        raise snap.ChainFailedError("cells exited 1: x")

    records, errors, warnings = _drive(snap, tmp_path, _fake_page(), run_chain)
    assert [(r.name, r.reached) for r in records] == [
        ("empty", True),
        ("wells-selected", True),
        ("snap", True),
        ("expert", True),
        ("after-density", False),
        ("with-route", False),
        ("results-all", False),
        ("turned", False),
        # The short-page shots need no chain, and nor do the guided states.
        ("stand-620", True),
        ("plate-620", True),
        ("wells-laid-out", True),
        ("scanned", True),
    ]
    assert records[4].reason == "cells exited 1: x"
    assert records[5].reason == "needs the chain: cells exited 1: x"
    assert records[6].reason == "needs the chain: cells exited 1: x"
    assert records[7].reason == "needs the chain: cells exited 1: x"
    assert (errors, warnings) == ([], 0)


def test_a_state_whose_control_times_out_is_named_and_the_next_state_runs(
    snap, tmp_path
) -> None:
    page = _fake_page(fails_on_button="Snap")
    records, _, _ = _drive(snap, tmp_path, page, lambda steps: None)
    # `after-density` takes a Snap too (the page reads the run document again at its
    # end, #297), so it fails on the same control; `expert`, `with-route` and `results-all`
    # run.
    assert [r.reached for r in records] == [
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert records[2].reason == "Locator.click: Timeout 10000ms exceeded."
    assert records[4].reason == records[2].reason
    page.screenshot.assert_any_call(path=str(tmp_path / "snap-FAILED.png"))
    page.screenshot.assert_any_call(path=str(tmp_path / "after-density-FAILED.png"))


def test_a_step_button_that_times_out_names_the_state_and_the_next_one_runs(
    snap, tmp_path
) -> None:
    # The Wells step is clicked before the box is filled (m5 §17.8): a button that is
    # not there fails `wells-selected` alone, named.
    page = _fake_page(fails_on_button="3. Wells")
    records, _, _ = _drive(snap, tmp_path, page, lambda steps: None)
    # `wells-laid-out` clicks the same step, so it fails on the same control, named; the
    # states around them run.
    assert [r.reached for r in records] == [
        True,
        False,
        *([True] * 8),
        False,
        True,
    ]
    assert records[1].reason == "Locator.click: Timeout 10000ms exceeded."
    assert records[10].reason == records[1].reason
    page.screenshot.assert_any_call(path=str(tmp_path / "wells-selected-FAILED.png"))
    page.screenshot.assert_any_call(path=str(tmp_path / "wells-laid-out-FAILED.png"))


def test_a_state_whose_control_times_out_alone_is_the_only_one_named(
    snap, tmp_path
) -> None:
    page = _fake_page(fails_on_fill=True)
    records, _, _ = _drive(snap, tmp_path, page, lambda steps: None)
    # `wells-laid-out` types the wells in the same box: the same control, named.
    assert [r.reached for r in records] == [
        True,
        False,
        *([True] * 8),
        False,
        True,
    ]
    assert records[1].reason == "Locator.fill: Timeout 10000ms exceeded."
    assert records[10].reason == records[1].reason
    page.screenshot.assert_any_call(path=str(tmp_path / "wells-selected-FAILED.png"))


def _no_chain() -> SimpleNamespace:
    return SimpleNamespace(run_chain=lambda steps: None)


def test_the_turned_state_picks_the_selects_turned_label_and_waits_for_the_outcome(
    snap,
) -> None:
    # The label is worded from the stand's sides, so it is read from the select; the
    # state ends on the strip's outcome line, never on a delay (FM-162, m5 §16.4).
    page = MagicMock()
    select = page.get_by_label.return_value
    select.locator.return_value.all_inner_texts.return_value = [
        " A1 at the back left ",
        "Turned 180\N{DEGREE SIGN}: A1 at the front right",
    ]
    assert snap.ACTIONS["turned"] is snap._act_turned
    # The select waits to be drawn before its options are read: the click on the Plate
    # step is answered by the server, and `all_inner_texts` does not wait (an empty read
    # gave the default label and a timeout, seen at the first run of #443).
    seen: list[str] = []
    select.wait_for.side_effect = lambda **kw: seen.append("wait")
    select.locator.return_value.all_inner_texts.side_effect = lambda: (
        seen.append("read")
        or [
            " A1 at the back left ",
            "Turned 180\N{DEGREE SIGN}: A1 at the front right",
        ]
    )
    snap._act_turned(page, _no_chain())
    assert seen == ["wait", "read"]
    assert _step_names(page) == [r"^2\. Plate"]
    page.get_by_label.assert_called_once_with("Plate lies", exact=True)
    select.select_option.assert_called_once_with(
        label="Turned 180\N{DEGREE SIGN}: A1 at the front right"
    )
    page.locator.assert_any_call("pre")
    page.locator.return_value.filter.assert_any_call(has_text="plate-orientation done")
    page.locator.return_value.filter.return_value.first.wait_for.assert_called_once_with()
    page.wait_for_timeout.assert_not_called()


def _step_names(page: MagicMock) -> list[str]:
    """The pattern of each ``get_by_role("button", name=<regex>)`` the action made."""
    return [
        c.kwargs["name"].pattern
        for c in page.mock_calls
        if c[0] == "get_by_role" and hasattr(c.kwargs.get("name"), "pattern")
    ]


def test_after_density_takes_a_snap_and_then_reads_the_results_step(snap) -> None:
    """#297: the page reads the run document again at the end of a snap, with no Refresh
    (a Refresh would redraw the map and say nothing of the redraw a snap made). Then the
    Results step, whose density row says it is done: the second of the state lines that
    start ``Done <date>`` (``cells`` is the first). The wait comes after the click, since the
    rows are on that screen only, and a wait that is dropped reads the page before it has
    redrawn. The checklist's ``✓ density`` is the expert view's now."""
    page = MagicMock()
    chains: list[list[object]] = []
    snap._act_after_density(page, SimpleNamespace(run_chain=chains.append))
    assert chains == [snap.CHAIN]
    calls = page.mock_calls
    assert [call[0] for call in calls] == [
        "get_by_role",
        "get_by_role().click",
        "get_by_role",
        "get_by_role().click",
        "locator",
        "locator().filter",
        "locator().filter().nth",
        "locator().filter().nth().wait_for",
    ]
    first_role = next(c for c in calls if c[0] == "get_by_role")
    assert first_role.kwargs == {"name": "Snap"}
    assert _step_names(page) == [r"^5\. Results"]
    assert [c.kwargs for c in calls if c[0] == "locator().filter"] == [
        {"has_text": snap.DONE_ROW}
    ]
    assert [c.args for c in calls if c[0] == "locator().filter().nth"] == [(1,)]
    # No tab to click, and no Refresh.
    assert not [c for c in calls if c[0] == "get_by_text"]
    assert not [c for c in calls if c.kwargs.get("name") == "Refresh"]


def test_with_route_presses_the_refresh_in_the_maps_header(snap) -> None:
    """Refresh is in the map's header row, always in view: no screen is shown first."""
    page = MagicMock()
    chains: list[list[object]] = []
    snap._act_with_route(page, SimpleNamespace(run_chain=chains.append))
    assert chains == [[snap.ROUTE_STEP]]
    roots = [
        (c[0], c.args, c.kwargs)
        for c in page.mock_calls
        if c[0] in ("get_by_text", "get_by_role")
    ]
    assert roots == [("get_by_role", ("button",), {"name": "Refresh"})]


def test_the_step_locators_match_a_steps_name_from_its_number(snap) -> None:
    """A step's name is ``3. Wells`` and gains ``✓`` when done: it is matched from its
    start, so that the box's label *Wells* and the map's *Plate* are never one."""
    page = MagicMock()
    snap._step_button(page, 3, "Wells")
    (pattern,) = _step_names(page)
    assert pattern == r"^3\. Wells"
    name = re.compile(pattern)
    for found in ("3. Wells", "3. Wells ✓"):
        assert name.search(found)
    for other in ("Wells", "13. Wells", "2. Wells", "Plate", "3. Plate"):
        assert not name.search(other)


def test_wells_selected_opens_the_wells_step_before_it_fills_the_box(snap) -> None:
    page = MagicMock()
    order: list[str] = []

    def role(*args: object, **kwargs: object) -> MagicMock:
        name = kwargs.get("name")
        found = MagicMock()
        found.click.side_effect = lambda: order.append(
            f"click:{getattr(name, 'pattern', name)}"
        )
        return found

    page.get_by_role.side_effect = role
    box = page.get_by_label.return_value
    box.fill.side_effect = lambda text: order.append(f"fill:{text}")
    box.press.side_effect = lambda key: order.append(f"press:{key}")
    snap._act_wells_selected(page, _no_chain())
    # Connect is clicked only when enabled (FM-162); then the step, then the box.
    assert order[-3:] == [r"click:^3\. Wells", "fill:A1:B2", "press:Enter"]
    # The box is found by its exact label: *Wells* is the step's name too, with a number.
    page.get_by_label.assert_called_once_with("Wells", exact=True)


def test_snap_clicks_the_image_without_a_tab(snap) -> None:
    page = MagicMock()
    snap._act_snap(page, _no_chain())
    assert [c.kwargs.get("name") for c in page.mock_calls if c[0] == "get_by_role"] == [
        "Snap"
    ]
    assert not [c for c in page.mock_calls if c[0] == "get_by_text"]


def test_expert_waits_for_the_pressed_class_and_the_switch_goes_off_after_the_photograph(
    snap, tmp_path
) -> None:
    # A pressed Toggle's button has `bk-active` (measured, Panel 1.10.0): the state waits
    # for it, and puts the switch off again after the photograph (m5 §17.8).
    page = MagicMock()
    snap._act_expert(page, _no_chain())
    page.get_by_role.assert_called_once_with("button", name="Expert view")
    page.get_by_role.return_value.click.assert_called_once_with()
    page.locator.assert_called_once_with("button.bk-active")
    page.locator.return_value.filter.assert_called_once_with(has_text="Expert view")
    page.locator.return_value.filter.return_value.wait_for.assert_called_once_with()

    off = MagicMock()
    snap._after_expert(off, _no_chain())
    off.get_by_role.return_value.click.assert_called_once_with()
    off.locator.return_value.filter.return_value.wait_for.assert_called_once_with(
        state="detached"
    )
    assert {
        "expert": snap._after_expert,
        "stand-620": snap._after_short,
        "plate-620": snap._after_short,
    } == snap.AFTER_ACTIONS

    # In the driver: the photograph is taken before the switch goes off.
    order: list[str] = []
    drive_page = _fake_page()
    drive_page.screenshot.side_effect = lambda **kw: order.append(Path(kw["path"]).name)
    real_role = drive_page.get_by_role.side_effect

    def role(*args: object, **kwargs: object) -> MagicMock:
        found = real_role(*args, **kwargs)
        if kwargs.get("name") == "Expert view":
            found.click.side_effect = lambda: order.append("click:Expert view")
        return found

    drive_page.get_by_role.side_effect = role
    _drive(snap, tmp_path, drive_page, lambda steps: None)
    first = order.index("click:Expert view")
    assert order[first : first + 4] == [
        "click:Expert view",
        "expert.png",
        "expert-full.png",
        "click:Expert view",
    ]


def test_the_viewport_defaults_to_1280_by_800_and_the_options_stay(snap) -> None:
    args = snap.build_parser().parse_args([])
    assert (args.width, args.height) == (1280, 800)
    args = snap.build_parser().parse_args(["--width", "1600", "--height", "1000"])
    assert (args.width, args.height) == (1600, 1000)


def _located(boxes: dict[str, dict[str, float] | None]) -> list[tuple[str, MagicMock]]:
    """The frame parts as ``_frame_parts`` gives them, with a box for each name in ``boxes``.

    ``None`` is a part the browser draws no box for; a name left out is not on the page.
    """
    parts = []
    for name in ("STOP", "step bar", "map", "image", "drive bar"):
        found = MagicMock()
        found.count.return_value = 1 if name in boxes else 0
        found.first.bounding_box.return_value = boxes.get(name)
        parts.append((name, found))
    return parts


def _box(x: float, y: float, width: float, height: float) -> dict[str, float]:
    return {"x": x, "y": y, "width": width, "height": height}


def test_on_screen_says_where_each_part_is_and_never_guesses(snap, monkeypatch) -> None:
    page = MagicMock()
    page.viewport_size = {"width": 1280, "height": 800}
    parts = _located(
        {
            "STOP": _box(1200, 5, 60, 30),
            "step bar": _box(10, 70, 170, 30),
            "map": _box(400, 195, 440, 400),
            "image": _box(840, 178, 440, 460),
            # Below the fold: its bottom is past the viewport's.
            "drive bar": _box(850, 760, 100, 50),
        }
    )
    monkeypatch.setattr(snap, "_frame_parts", lambda page: parts)
    assert snap._on_screen(page) == [
        "STOP: yes (y 5 to 35 of 800, x 1200 to 1260 of 1280)",
        "step bar: yes (y 70 to 100 of 800, x 10 to 180 of 1280)",
        "map: yes (y 195 to 595 of 800, x 400 to 840 of 1280)",
        "image: yes (y 178 to 638 of 800, x 840 to 1280 of 1280)",
        "drive bar: NO (y 760 to 810 of 800, x 850 to 950 of 1280)",
    ]
    # A part the browser draws no box for, and one that is not on the page, are said so,
    # never guessed to be in view.
    parts = _located({"STOP": None})
    assert snap._on_screen(page)[:2] == [
        "STOP: not drawn",
        "step bar: not on the page",
    ]
    # A part wider than the viewport is not on screen either (a drive bar that runs past
    # the window was the first finding of #443).
    parts = _located({"STOP": _box(1200, 5, 100, 30)})
    assert snap._on_screen(page)[0].startswith("STOP: NO (")
    # A page with no viewport (a fake) says nothing.
    assert snap._on_screen(MagicMock()) == []


# --- the short-page states of D12 (m5 §17.2, §17.8, #444) -----------------------------


def test_the_short_page_states_name_the_screens_titles_the_guide_draws(snap) -> None:
    from smc.ui.guide import SCREEN_TITLES, scan_title

    # The script imports nothing of smc at import, so its copy of the titles is pinned: the
    # two short-page screens' and, for the guided states, the Wells screen's and the Scan
    # screen's, which names the two wells the state types, in plate order.
    assert {
        "stand-620": SCREEN_TITLES["stand"],
        "plate-620": SCREEN_TITLES["plate"],
        "wells-laid-out": SCREEN_TITLES["wells"],
        "scanned": scan_title(["B11", "B2"], []),
        # The Results screen's, which `results-all` reads (m5 §17.11.4).
        "results-all": SCREEN_TITLES["results"],
    } == snap.SCREEN_TITLE_OF
    assert snap.SHORT_VIEWPORT_HEIGHT_PX == 620


@pytest.mark.parametrize(
    ("action", "number", "title", "heading"),
    [
        ("_act_stand_620", 1, "Stand", "Which stand, and which preset?"),
        ("_act_plate_620", 2, "Plate", "Where is the plate?"),
    ],
)
def test_a_short_page_state_connects_forgets_the_outcome_shrinks_then_opens_its_screen(
    snap, action: str, number: int, title: str, heading: str
) -> None:
    page = MagicMock()
    page.viewport_size = {"width": 1280, "height": 800}
    order: list[str] = []
    page.set_viewport_size.side_effect = lambda size: order.append(f"resize:{size}")

    def role(*args: object, **kwargs: object) -> MagicMock:
        name = kwargs.get("name")
        found = MagicMock()
        found.is_enabled.return_value = False  # connected already: no click, FM-162
        found.click.side_effect = lambda: order.append(
            f"click:{getattr(name, 'pattern', name)}"
        )
        return found

    page.get_by_role.side_effect = role
    page.locator.return_value.filter.return_value.first.wait_for.side_effect = lambda: (
        order.append("wait:title")
    )
    getattr(snap, action)(page, _no_chain())
    # The outcome the earlier states left is forgotten with *Refresh* (D12 measures the
    # header with its outcome lines empty); the page shrinks before the screen opens, so
    # that it is laid out at 620 px and never resized under the photograph; the state ends
    # on the screen's own title (FM-162), never on a delay.
    assert order == [
        "click:Refresh",
        "resize:{'width': 1280, 'height': 620}",
        f"click:^{number}\\. {title}",
        "wait:title",
    ]
    page.locator.return_value.filter.assert_any_call(has_text=heading)
    page.wait_for_timeout.assert_not_called()


def test_a_short_page_state_puts_the_runs_viewport_back_after_its_photograph(
    snap,
) -> None:
    page = MagicMock()
    snap._after_short(page, SimpleNamespace(viewport=(1600, 1000)))
    page.set_viewport_size.assert_called_once_with({"width": 1600, "height": 1000})
    # In the driver the photograph is taken at 620 and the viewport is put back after it,
    # and the states after them see the run's size again.
    order: list[str] = []
    drive_page = _fake_page()
    drive_page.set_viewport_size.side_effect = lambda size: order.append(
        f"resize:{size['height']}"
    )
    drive_page.screenshot.side_effect = lambda **kw: order.append(Path(kw["path"]).name)
    _drive(snap, Path("."), drive_page, lambda steps: None)
    first = order.index("stand-620.png")
    assert order[first - 1] == "resize:620"
    assert order[first : first + 3] == [
        "stand-620.png",
        "stand-620-full.png",
        "resize:800",
    ]
    second = order.index("plate-620.png")
    assert order[second - 1] == "resize:620"
    assert order[second + 2] == "resize:800"
    # The short states are the last two: nothing is photographed at a short height before.
    assert order.index("turned.png") < first < second


def test_on_screen_reports_the_title_and_the_first_action_of_a_short_page(
    snap, monkeypatch
) -> None:
    page = MagicMock()
    page.viewport_size = {"width": 1280, "height": 620}
    seen: list[tuple[str, object]] = []

    def part(name: str, box: dict[str, float] | None) -> tuple[str, MagicMock]:
        found = MagicMock()
        found.count.return_value = 1
        found.first.bounding_box.return_value = box
        seen.append((name, found))
        return name, found

    parts = [
        part("STOP", _box(1200, 5, 60, 30)),
        part("step bar", _box(10, 73, 170, 31)),
        part("title", _box(10, 142, 370, 17)),
        # The Plate screen's first action, past the window's bottom: the finding of the
        # first run, when the header held a stale outcome line.
        part("first action", _box(10, 592, 73, 31)),
    ]
    monkeypatch.setattr(snap, "_short_parts", lambda page, state: parts)
    assert snap._on_screen(page, "plate-620") == [
        "STOP: yes (y 5 to 35 of 620, x 1200 to 1260 of 1280)",
        "step bar: yes (y 73 to 104 of 620, x 10 to 180 of 1280)",
        "title: yes (y 142 to 159 of 620, x 10 to 380 of 1280)",
        "first action: NO (y 592 to 623 of 620, x 10 to 83 of 1280)",
    ]


def test_the_short_parts_are_the_strip_the_step_bar_the_title_and_the_first_action(
    snap,
) -> None:
    page = MagicMock()
    stand = {n: c for n, c in snap._short_parts(page, "stand-620")}
    assert list(stand) == ["STOP", "step bar", "title", "first action"]
    page.get_by_role.assert_any_call("button", name="Load", exact=True)
    page.locator.return_value.filter.assert_any_call(
        has_text="Which stand, and which preset?"
    )
    page = MagicMock()
    snap._short_parts(page, "plate-620")
    names = [
        c.kwargs["name"].pattern
        for c in page.get_by_role.mock_calls
        if hasattr(c.kwargs.get("name"), "pattern")
    ]
    # The first *Go to* of the rows, and the Stand step's button for the step bar.
    assert r"^Go to " in names
    assert r"^1\. Stand" in names
    page.locator.return_value.filter.assert_any_call(has_text="Where is the plate?")


# --- the guided states of the Wells and Scan screens (m5 §17.10.4, #445) ----------------


UPRIGHT = "A1 at the back left"
TURNED = "Turned 180\N{DEGREE SIGN}: A1 at the front right"


def _recording_page(*, plate_lies: str = TURNED) -> tuple[MagicMock, list[str]]:
    """A page that writes what an action does, in order: a click by its name, a fill, a key, a wait by its text.

    *Connect* is enabled never, so the action does not click it (FM-162); a wait for a pane's
    text is ``wait:<text or pattern>`` and the wait for the strip to lose ``not connected`` is
    ``wait-detached:``. The *Plate lies* select offers the two options of a stand that says its
    sides, shows ``plate_lies`` (the ``turned`` state leaves it turned), and writes the option
    chosen as ``select:<label>``.
    """
    page = MagicMock()
    page.viewport_size = {"width": 1280, "height": 800}
    order: list[str] = []

    def role(*args: object, **kwargs: object) -> MagicMock:
        name = kwargs.get("name")
        found = MagicMock()
        found.is_enabled.return_value = False
        found.click.side_effect = lambda: order.append(
            f"click:{getattr(name, 'pattern', name)}"
        )
        return found

    def filtered(*, has_text: object) -> MagicMock:
        label = getattr(has_text, "pattern", has_text)
        found = MagicMock()
        found.first.wait_for.side_effect = lambda **kw: order.append(f"wait:{label}")
        found.wait_for.side_effect = lambda **kw: order.append(f"wait-detached:{label}")

        def nth(index: int) -> MagicMock:
            item = MagicMock()
            item.wait_for.side_effect = lambda **kw: order.append(
                f"wait-nth{index}:{label}"
            )
            return item

        found.nth.side_effect = nth
        return found

    page.get_by_role.side_effect = role
    page.locator.return_value.filter.side_effect = filtered
    box = page.get_by_label.return_value
    box.fill.side_effect = lambda text: order.append(f"fill:{text}")
    box.press.side_effect = lambda key: order.append(f"press:{key}")
    box.wait_for.side_effect = lambda **kw: order.append("wait-select")
    box.locator.return_value.all_inner_texts.return_value = [f" {UPRIGHT} ", TURNED]
    box.evaluate.return_value = f" {plate_lies} "
    box.select_option.side_effect = lambda label: order.append(f"select:{label}")
    return page, order


def test_the_wells_laid_out_state_clicks_the_pages_own_buttons_and_ends_on_texts(
    snap,
) -> None:
    page, order = _recording_page()
    assert snap.ACTIONS["wells-laid-out"] is snap._act_wells_laid_out
    snap._act_wells_laid_out(page, _no_chain())
    # The guided path, by the screens' own buttons: Plate, the plate declared upright (the
    # `turned` state left it turned, and the sample's centres are refused for a plate that lies
    # turned) with the wait for its outcome, *Register the plate* and the wait for its outcome
    # line, Wells, the two wells, *Lay out the fields* and the waits for its outcome and for
    # the count of the known block: every wait is for a text the server draws when its run
    # ends (FM-162), never a delay.
    assert order == [
        r"wait-detached:\| not connected\b",
        r"click:^2\. Plate",
        "wait-select",
        f"select:{UPRIGHT}",
        "wait:plate-orientation done",
        "click:Register the plate",
        "wait:plate-registration done",
        r"click:^3\. Wells",
        "fill:B2,B11",
        "press:Enter",
        "click:Lay out the fields",
        "wait:scan-plan done",
        r"wait:Fields: \d+ in 2 wells",
    ]
    page.wait_for_timeout.assert_not_called()


def test_the_wells_laid_out_state_leaves_a_plate_that_is_already_upright_alone(
    snap,
) -> None:
    # A run with `--no-results` never turns the plate: no switch, and no wait for a run that
    # was never started.
    page, order = _recording_page(plate_lies=UPRIGHT)
    snap._act_wells_laid_out(page, _no_chain())
    assert order[:4] == [
        r"wait-detached:\| not connected\b",
        r"click:^2\. Plate",
        "wait-select",
        "click:Register the plate",
    ]
    assert not [line for line in order if line.startswith("select:")]
    assert "wait:plate-orientation done" not in order
    page.wait_for_timeout.assert_not_called()


def test_the_scanned_state_starts_the_scan_and_waits_for_its_own_outcome(snap) -> None:
    page, order = _recording_page()
    assert snap.ACTIONS["scanned"] is snap._act_scanned
    snap._act_scanned(page, _no_chain())
    # `scan done` is not in `scan-plan done`: the wait is for the scan's own outcome line.
    assert order == [r"click:^4\. Scan", "click:Start the scan", "wait:scan done"]
    assert "scan done" not in "scan-plan done in 0.0 s"
    page.wait_for_timeout.assert_not_called()


def test_the_results_all_state_opens_results_waits_for_the_done_row_runs_all_and_waits_for_the_outcome(
    snap,
) -> None:
    page, order = _recording_page()
    assert snap.ACTIONS["results-all"] is snap._act_results_all
    snap._act_results_all(page, _no_chain())
    # The click on the button comes after the wait for the redrawn row (the screen is swapped by
    # the server), and the last wait is the strip's own `run all done`: no delay anywhere (FM-162).
    assert order == [
        r"click:^5\. Results",
        f"wait-nth1:{snap.DONE_ROW.pattern}",
        "click:Run all that are left",
        "wait:run all done",
    ]
    page.wait_for_timeout.assert_not_called()
    # The button is found by its whole name: no row's *Run* answers to it.
    page.get_by_role.assert_any_call("button", name="Run all that are left", exact=True)


def test_the_results_screens_words_are_the_guides(snap) -> None:
    from smc.ui.guide import (
        FILES_HEADING,
        RESULT_NAMES,
        RESULT_TOOLS,
        RUN_ALL_LABEL,
    )

    # The script imports nothing of smc at import, so its copy of the words is pinned.
    assert tuple(RESULT_NAMES[tool] for tool in RESULT_TOOLS) == snap.RESULT_ROWS
    assert snap.RUN_ALL == RUN_ALL_LABEL
    assert snap.FILES_HEADING == FILES_HEADING


def test_the_done_row_pattern_matches_a_state_line_and_not_the_chains_line(
    snap,
) -> None:
    assert snap.DONE_ROW.search("Done 2026-11-12 09:30:00 UTC")
    assert snap.DONE_ROW.search("Done 2026-11-12 09:30:00 UTC (dry run)")
    # The chain's own line is `Done: ...`, the files block's `Scan frames: ...`.
    assert not snap.DONE_ROW.search("Done: Count the cells, Density per well.")
    assert not snap.DONE_ROW.search("Ready")
    assert not snap.DONE_ROW.search("Not usable: Done 2026-11-12")


def test_the_results_parts_are_the_frames_and_the_screens_title_and_run_all(
    snap,
) -> None:
    page = MagicMock()
    parts = dict(snap._results_parts(page))
    assert list(parts) == [
        "STOP",
        "step bar",
        "map",
        "image",
        "drive bar",
        "title",
        "first action",
    ]
    page.get_by_role.assert_any_call("button", name="Run all that are left", exact=True)
    page.locator.return_value.filter.assert_any_call(has_text=re.compile(r"^Results$"))


def test_on_screen_reads_the_results_parts_of_results_all_and_only_there(
    snap, monkeypatch
) -> None:
    page = MagicMock()
    page.viewport_size = {"width": 1280, "height": 800}
    found = MagicMock()
    found.count.return_value = 1
    # The button past the page's bottom: what the summary must say as NO.
    found.first.bounding_box.return_value = _box(10, 150, 170, 31)
    title = MagicMock()
    title.count.return_value = 1
    title.first.bounding_box.return_value = _box(10, 142, 370, 17)
    monkeypatch.setattr(
        snap, "_results_parts", lambda page: [("title", title), ("first action", found)]
    )
    assert snap._on_screen(page, "results-all") == [
        "title: yes (y 142 to 159 of 800, x 10 to 380 of 1280)",
        "first action: yes (y 150 to 181 of 800, x 10 to 180 of 1280)",
    ]
    # Another state reads the frame's parts, as before.
    monkeypatch.setattr(snap, "_frame_parts", lambda page: [("STOP", title)])
    assert snap._on_screen(page, "snap") == [
        "STOP: yes (y 142 to 159 of 800, x 10 to 380 of 1280)"
    ]


def test_results_facts_say_the_rows_the_chain_line_and_the_files_or_that_they_are_not_on_the_page(
    snap,
) -> None:
    texts = [
        "Results",
        "Each step reads the one before. Run the next one, or all that are left.",
        "Done: Report, Export for NIS-Elements.",
        "Count the cells",
        "Done 2026-11-12 09:00:00 UTC",
        "Density per well",
        "Done 2026-11-12 09:00:01 UTC",
        "Route through the densest fields",
        "Done 2026-11-12 09:00:02 UTC",
        "Report",
        "Done 2026-11-12 09:00:03 UTC (dry run)",
        "Export for NIS-Elements",
        "Done 2026-11-12 09:00:04 UTC (dry run)",
        "Where the files are",
        "Scan frames: /run/scan\nReport: /run/report.html",
    ]
    assert snap.results_facts(texts) == [
        "row Count the cells: Done 2026-11-12 09:00:00 UTC",
        "row Density per well: Done 2026-11-12 09:00:01 UTC",
        "row Route through the densest fields: Done 2026-11-12 09:00:02 UTC",
        "row Report: Done 2026-11-12 09:00:03 UTC (dry run)",
        "row Export for NIS-Elements: Done 2026-11-12 09:00:04 UTC (dry run)",
        "chain: Done: Report, Export for NIS-Elements.",
        "files: Scan frames: /run/scan | Report: /run/report.html",
    ]
    # A pane the page did not draw is said, never guessed; a name that ends the list has no
    # state line after it.
    assert snap.results_facts(["Results", "Count the cells"]) == [
        "row Count the cells: not on the page",
        *(f"row {name}: not on the page" for name in snap.RESULT_ROWS[1:]),
        "chain: not on the page",
        "files: not on the page",
    ]
    assert snap.results_facts([]) == [
        *(f"row {name}: not on the page" for name in snap.RESULT_ROWS),
        "chain: not on the page",
        "files: not on the page",
    ]
    # The chain's line is the first that starts as one: running, done or stopped.
    assert snap.results_facts(["Running Count the cells (1 of 5)."])[-2] == (
        "chain: Running Count the cells (1 of 5)."
    )
    assert snap.results_facts(["Stopped at Report by STOP."])[-2] == (
        "chain: Stopped at Report by STOP."
    )


def test_the_summary_says_the_results_all_facts_in_their_order(snap) -> None:
    record = snap.StateRecord(
        name="results-all",
        reached=True,
        step="5. Results ✓",
        facts=["row Report: Done 2026-11-12 09:00:03 UTC (dry run)", "chain: Done."],
    )
    text = snap.format_summary(
        [record], server="stopped by SIGINT", server_errors=[], browser_errors=[]
    )
    lines = text.splitlines()
    start = lines.index("== results-all ==")
    assert lines[start : start + 5] == [
        "== results-all ==",
        "reached",
        "step: 5. Results ✓",
        "row Report: Done 2026-11-12 09:00:03 UTC (dry run)",
        "chain: Done.",
    ]


def test_no_action_clicks_a_tools_own_run_button(snap) -> None:
    # The skill's rule: the script presses the guided screens' three buttons, the Results
    # screen's *Run all that are left* and no tool's own *Run <tool>* (the expert view's, nor
    # a Results row's), nor Drive, STOP or Resume.
    import inspect

    for name, action in snap.ACTIONS.items():
        source = inspect.getsource(action)
        assert 'name="Run ' not in source, name
        assert "^Run " not in source, name
        for forbidden in ("Drive", "Resume", 'name="STOP"'):
            assert forbidden not in source.replace("Drive to", ""), (name, forbidden)
    buttons = set()
    for name in snap.GUIDED_STATES:
        page, order = _recording_page()
        snap.ACTIONS[name](page, _no_chain())
        buttons |= {
            line.removeprefix("click:") for line in order if line.startswith("click:")
        }
    assert {"Register the plate", "Lay out the fields", "Start the scan"} <= buttons


def test_the_guided_parts_are_the_frames_and_the_screens_title_and_first_action(
    snap,
) -> None:
    page = MagicMock()
    wells = {n: c for n, c in snap._guided_parts(page, "wells-laid-out")}
    assert list(wells) == [
        "STOP",
        "step bar",
        "map",
        "image",
        "drive bar",
        "title",
        "first action",
    ]
    page.get_by_role.assert_any_call("button", name="Lay out the fields")
    page.locator.return_value.filter.assert_any_call(has_text="Which wells?")
    page = MagicMock()
    snap._guided_parts(page, "scanned")
    page.get_by_role.assert_any_call("button", name="Start the scan")
    page.locator.return_value.filter.assert_any_call(has_text="Scan B2 and B11")


def test_on_screen_reads_the_guided_parts_of_a_guided_state_and_only_there(
    snap, monkeypatch
) -> None:
    page = MagicMock()
    page.viewport_size = {"width": 1280, "height": 800}

    def part(name: str, box: dict[str, float]) -> tuple[str, MagicMock]:
        found = MagicMock()
        found.count.return_value = 1
        found.first.bounding_box.return_value = box
        return name, found

    guided = [
        part("STOP", _box(1200, 5, 60, 30)),
        part("title", _box(10, 142, 370, 17)),
        # The first action past the page's bottom: what the summary must say as NO.
        part("first action", _box(10, 790, 120, 31)),
    ]
    monkeypatch.setattr(snap, "_guided_parts", lambda page, state: guided)
    for state in ("wells-laid-out", "scanned"):
        assert snap._on_screen(page, state) == [
            "STOP: yes (y 5 to 35 of 800, x 1200 to 1260 of 1280)",
            "title: yes (y 142 to 159 of 800, x 10 to 380 of 1280)",
            "first action: NO (y 790 to 821 of 800, x 10 to 130 of 1280)",
        ]
    # A state that is not guided reads the frame's parts, as before.
    frame = _located({"STOP": _box(1200, 5, 60, 30)})
    monkeypatch.setattr(snap, "_frame_parts", lambda page: frame)
    assert snap._on_screen(page, "snap")[0] == (
        "STOP: yes (y 5 to 35 of 800, x 1200 to 1260 of 1280)"
    )
    assert len(snap._on_screen(page, "snap")) == 5


def test_scan_facts_say_the_progress_and_the_result_or_that_they_are_not_on_the_page(
    snap,
) -> None:
    texts = [
        "Scan B2 and B11",
        "72 fields at Nikon 10X S Fluor. STOP, at the top right, stops the scan.",
        "Field 72 of 72\nElapsed 0 s",
        "Scanned in a dry run: no frame was taken\nassumed: dry run: 72 fields walked",
    ]
    assert snap.scan_facts(texts) == [
        "progress: Field 72 of 72 | Elapsed 0 s",
        "result: Scanned in a dry run: no frame was taken | assumed: dry run: 72 "
        "fields walked",
    ]
    # A pane the page did not draw is said, never guessed.
    assert snap.scan_facts(["Scan B2 and B11"]) == [
        "progress: not on the page",
        "result: not on the page",
    ]
    assert snap.scan_facts([]) == [
        "progress: not on the page",
        "result: not on the page",
    ]


def test_the_summary_says_a_states_facts_after_what_is_on_screen(snap) -> None:
    record = snap.StateRecord(
        name="scanned",
        reached=True,
        step="4. Scan",
        texts=["Field 72 of 72"],
        on_screen=["title: yes (y 142 to 159 of 800, x 10 to 380 of 1280)"],
        facts=["progress: Field 72 of 72", "result: Scanned: 72 frames, 0 failed"],
    )
    text = snap.format_summary(
        [record], server="stopped by SIGINT", server_errors=[], browser_errors=[]
    )
    lines = text.splitlines()
    start = lines.index("== scanned ==")
    assert lines[start : start + 6] == [
        "== scanned ==",
        "reached",
        "step: 4. Scan",
        "on screen: title: yes (y 142 to 159 of 800, x 10 to 380 of 1280)",
        "progress: Field 72 of 72",
        "result: Scanned: 72 frames, 0 failed",
    ]


def test_only_the_scanned_and_results_all_states_carry_facts(snap, tmp_path) -> None:
    page = _fake_page()
    records, _, _ = _drive(snap, tmp_path, page, lambda steps: None)
    by_name = {r.name: r for r in records}
    # The fake page's one pane is `a pane`: neither line is on it, and each says so.
    assert by_name["scanned"].facts == [
        "progress: not on the page",
        "result: not on the page",
    ]
    assert by_name["results-all"].facts == [
        *(f"row {name}: not on the page" for name in snap.RESULT_ROWS),
        "chain: not on the page",
        "files: not on the page",
    ]
    assert [n for n, r in by_name.items() if r.facts] == ["results-all", "scanned"]


# --- the chain against the plugins ------------------------------------------


def test_every_param_of_the_chain_is_a_param_of_its_plugin(snap) -> None:
    """A renamed key would end each `smc run` with exit 2, at the first browser run."""
    from smc.plugins.params import parse_params
    from smc.plugins.registry import load

    for step in [*snap.CHAIN, snap.ROUTE_STEP]:
        params = load(step.plugin).plugin.Params
        parse_params(params, [f"{key}={value}" for key, value in step.params])
