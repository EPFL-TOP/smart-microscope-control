"""A core left open at exit is detached before the interpreter finalizes (#538).

A core nobody released is freed while the interpreter finalizes, and
``CMMCorePlus.__del__`` then calls ``registerCallback(None)``, which waits for
the notification MMCore's own thread is delivering; that thread can no longer
take the GIL, so the process never ends (measured on CI: 15 and 10 children in
300 on Linux, 3 in 300 on Windows, never on macOS). The detach therefore runs
from ``atexit``, while a handler in flight can still finish. Each child exits
without closing its core, the way an ``smc`` command or a quit ``smc ui`` does.
"""

from __future__ import annotations

import gc
import json
import logging
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

from smc.hardware import core as core_module
from smc.hardware.core import close_core, new_core

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "zeiss-observer7.toml"

#: Bounds the child, below the parent's own timeout, so that a hang fails with
#: every thread's stack instead of a bare ``TimeoutExpired`` (FM-44).
_WATCHDOG = """
import faulthandler

faulthandler.dump_traceback_later(45, exit=True)
"""

# The spy replaces the module's `detach_callbacks`, which the exit handler looks
# up when it runs: it prints, then detaches for real, so the child still exits.
_LEFT_OPEN = (
    _WATCHDOG
    + """
from smc.hardware import core as core_module

real = core_module.detach_callbacks


def spy(core):
    print("detached", id(core) == expected, flush=True)
    real(core)


core_module.detach_callbacks = spy
core = core_module.new_core()
expected = id(core)
print("open", flush=True)
"""
)


def _run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )


def test_a_core_left_open_is_detached_before_the_interpreter_finalizes() -> None:
    done = _run(_LEFT_OPEN)
    why = f"exit code {done.returncode}; stderr ends: {done.stderr[-800:]!r}"
    assert done.returncode == 0, why
    assert done.stdout.splitlines() == ["open", "detached True"], why


def test_a_released_core_is_forgotten_and_the_exit_hook_detaches_the_live_ones(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The set holds the cores weakly: tracking a core for the exit must not
    # keep a released one alive, which would defer its destructor to the
    # finalization this hook exists to make safe. pymmcore-plus logs
    # "Initialized core %s" at DEBUG and pytest would keep that record, and
    # with it the core (as in test_demo_core.py).
    caplog.set_level(logging.INFO, logger="pymmcore-plus")
    released = new_core()
    ref = weakref.ref(released)
    tracked = released in core_module._live_cores
    assert tracked
    close_core(released)
    del released
    gc.collect()
    assert ref() is None
    kept = new_core()
    try:
        detached: list[object] = []
        monkeypatch.setattr(core_module, "detach_callbacks", detached.append)
        core_module._detach_live_cores_at_exit()
        assert any(c is kept for c in detached)
    finally:
        close_core(kept)


def test_an_interrupt_during_one_detach_still_detaches_every_other_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Ctrl-C that lands as the first wait ends is no `Exception`: the cores
    # after it must still be detached, or they hang the finalization (FM-49).
    first, second = new_core(), new_core()
    tried: list[object] = []

    def detach(core: object) -> None:
        tried.append(core)
        if len(tried) == 1:
            raise KeyboardInterrupt

    try:
        monkeypatch.setattr(core_module, "detach_callbacks", detach)
        with pytest.raises(KeyboardInterrupt):
            core_module._detach_live_cores_at_exit()
        assert any(c is first for c in tried)
        assert any(c is second for c in tried)
    finally:
        monkeypatch.undo()
        close_core(first)
        close_core(second)


# The fake MTB stand of the session's own exit test, with the detach recorded
# in the fake's call list, which the first handler registered dumps last.
_MTB_THEN_DETACH = (
    _WATCHDOG
    + """
import atexit
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
# Registered before smc is imported, so that it runs after every smc handler.
atexit.register(
    lambda: out.write_text(
        json.dumps([list(map(str, c)) for c in fake.calls]), encoding="utf-8"
    )
)

from smc.hardware import Microscope
from smc.hardware import core as core_module
from smc.hardware.backends.zeiss_mtb import session
from smc.hardware.profile import Profile
from smc.testing.fake_mtb import FakeMTB, fake_stand_config

fake = FakeMTB()
real = core_module.detach_callbacks


def spy(core):
    fake._calls.add("core", "detach")
    real(core)


core_module.detach_callbacks = spy
swap = session.use_api_factory(lambda: fake)
swap.__enter__()  # never left: the process exits with the session open
profile = Profile.load(Path(sys.argv[2]))
cfg = fake_stand_config(profile.config_path(), out.parent)
mm = profile.micromanager.model_copy(update={"config": str(cfg)})
stand = Microscope.open(profile.model_copy(update={"micromanager": mm}))
stand.core.setShutterOpen("TLShutter", True)
# The hook detaches and never releases: an unload at exit would act on the
# stand after the logout. (A SWIG class refuses a spy on unloadAllDevices.)
release = core_module.close_core


def recorded_release(core):
    fake._calls.add("core", "release")
    release(core)


core_module.close_core = recorded_release
"""
)


def test_the_exit_detach_runs_after_the_mtb_session_s_exit_handler(
    tmp_path: Path,
) -> None:
    # The light first (m4 §8): the MTB session's handler is registered at the
    # login, after the import of smc.hardware.core, and atexit runs last in,
    # first out, so the logout comes before any detach.
    calls_file = tmp_path / "calls.json"
    done = _run(_MTB_THEN_DETACH, str(calls_file), str(PROFILE))
    why = f"exit code {done.returncode}; stderr ends: {done.stderr[-800:]!r}"
    assert done.returncode == 0, why
    calls = [tuple(c) for c in json.loads(calls_file.read_text(encoding="utf-8"))]
    assert calls.count(("core", "detach")) == 1, why
    assert calls.index(("FakeMTB", "logout")) < calls.index(("core", "detach"))
    assert ("core", "release") not in calls
