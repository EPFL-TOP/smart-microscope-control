"""scripts/dev/ui_snap.py looks at the plate page of a stand that can move a stage.

No browser and no ``smc ui`` here: the guard, the commands the script builds, the
waits and the stop are pinned on their own; the page itself is looked at by
running the script (``/ui-check``).
"""

import importlib.util
import json
import os
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


def _fake_page(*, fails_on_text: str | None = None) -> MagicMock:
    """A page on which every control is there, and the one named in ``fails_on_text``
    times out when it is clicked, with the first line Playwright gives."""
    page = MagicMock()
    page.locator.return_value.count.return_value = 0
    page.locator.return_value.all_inner_texts.return_value = ["a pane"]
    page.get_by_label.return_value.count.return_value = 0

    def get_by_text(text: str, exact: bool = False) -> MagicMock:
        found = MagicMock()
        if text == fails_on_text:
            error = _FakeError(
                "Locator.click: Timeout 10000ms exceeded.\nCall log:\n  - waiting"
            )
            found.click.side_effect = error
            found.first.click.side_effect = error
        return found

    page.get_by_text.side_effect = get_by_text
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
        stand=SimpleNamespace(run_chain=run_chain),
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
    names = ["empty", "wells-selected", "snap", "after-density", "with-route"]

    full = snap.plan_states("demo-sample", results=True)
    assert [p.name for p in full] == names
    assert all(p.reachable and p.reason is None for p in full)

    demo = snap.plan_states("demo", results=True)
    assert [p.name for p in demo] == names
    assert [p.reachable for p in demo] == [True, True, True, False, False]
    for plan in demo[3:]:
        assert plan.reason == (
            "the demo camera images no sample: no cells, no density; "
            "use --stand demo-sample"
        )

    skipped = snap.plan_states("demo-sample", results=False)
    assert [p.reachable for p in skipped] == [True, True, True, False, False]
    assert {p.reason for p in skipped[3:]} == {"skipped by --no-results"}

    failed = snap.plan_states(
        "demo-sample", results=True, chain_failure="cells exited 1: no detector"
    )
    assert [p.reachable for p in failed] == [True, True, True, False, False]
    assert failed[3].reason == "cells exited 1: no detector"
    assert failed[4].reason is not None
    assert "cells exited 1: no detector" in failed[4].reason
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
        active_tab="Plate",
        wells="",
        texts=["idle", "no run yet"],
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
    assert "== empty ==\nreached\nactive tab: Plate\nWells: ''\n" in text
    assert "idle\n\nno run yet" in text
    assert "== snap ==\nreached\nactive tab: ?\n" in text
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
        ("after-density", False),
        ("with-route", False),
    ]
    assert records[3].reason == "cells exited 1: x"
    assert records[4].reason == "needs the chain: cells exited 1: x"
    assert (errors, warnings) == ([], 0)


def test_a_state_whose_control_times_out_is_named_and_the_next_state_runs(
    snap, tmp_path
) -> None:
    page = _fake_page(fails_on_text="Image")
    records, _, _ = _drive(snap, tmp_path, page, lambda steps: None)
    assert [r.reached for r in records] == [True, True, False, True, True]
    assert records[2].reason == "Locator.click: Timeout 10000ms exceeded."
    page.screenshot.assert_any_call(path=str(tmp_path / "snap-FAILED.png"))


# --- the chain against the plugins ------------------------------------------


def test_every_param_of_the_chain_is_a_param_of_its_plugin(snap) -> None:
    """A renamed key would end each `smc run` with exit 2, at the first browser run."""
    from smc.plugins.params import parse_params
    from smc.plugins.registry import load

    for step in [*snap.CHAIN, snap.ROUTE_STEP]:
        params = load(step.plugin).plugin.Params
        parse_params(params, [f"{key}={value}" for key, value in step.params])
