"""TEMPORARY (#538): run the exit test's child many times on CI and name the hang.

Removed before the PR is ready: it only exists to reproduce the hang on Linux.
Round 3, with the fix in smc.hardware.core; round 2 had: each child with and without a detach of the core's callbacks at
``atexit``, and a demo-core child with a slow handler in flight at exit.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import test_zeiss_mtb_session as t

N = int(os.environ.get("SMC_REPRO538_N", "300"))
PAR = 4
HANG_S = 25

_PRELUDE = f"""
import atexit, faulthandler, sys, time
faulthandler.dump_traceback_later({HANG_S}, exit=True)
_t0 = time.monotonic()
def _mark(what):
    sys.stderr.write("MARK %s %.2f s\\n" % (what, time.monotonic() - _t0))
    sys.stderr.flush()
atexit.register(_mark, "atexit-all-done")
"""

_DETACH = """
from smc.hardware.core import detach_callbacks as _detach
atexit.register(_detach, stand.core)
"""

_SLOW = """
import threading
from smc.hardware.core import open_core, detach_callbacks
core = open_core(None)
started = threading.Event()
def slow(*_args):
    started.set()
    time.sleep(1.0)
core.events.propertyChanged.connect(slow)
if DETACH:
    atexit.register(detach_callbacks, core)
core.setProperty("Camera", "Binning", "2")
_mark("handler started=%s" % started.wait(5))
"""


def _loop(script: Path, n: int, tmp_path: Path) -> str:
    profile = Path(__file__).resolve().parents[2] / "profiles" / "zeiss-observer7.toml"

    def one(i: int) -> tuple[int, float, int, str]:
        d = tmp_path / f"{script.stem}-run{i}"
        d.mkdir()
        t0 = time.monotonic()
        p = subprocess.run(
            [sys.executable, str(script), str(d / "calls.json"), str(profile)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        return i, time.monotonic() - t0, p.returncode, p.stderr

    bad: list[str] = []
    times: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(PAR) as ex:
        for i, dt, rc, err in ex.map(one, range(n)):
            times.append(dt)
            if rc != 0 or dt > HANG_S:
                bad.append(f"--- run {i}: {dt:.1f} s rc={rc}\n{err}")
    times.sort()
    summary = (
        f"{script.stem}: {len(bad)}/{n} bad; median {times[len(times) // 2]:.2f} s, "
        f"max {times[-1]:.2f} s"
    )
    return summary + ("\n" + "\n".join(bad[:2]) if bad else "")


@pytest.mark.parametrize("detach", [False], ids=["with-the-fix"])
def test_repro_538_the_exit_child_never_hangs(tmp_path: Path, detach: bool) -> None:
    script = tmp_path / f"zeiss-{'detach' if detach else 'asis'}.py"
    script.write_text(
        _PRELUDE + t._CHILD + (_DETACH if detach else ""), encoding="utf-8"
    )
    result = _loop(script, N, tmp_path)
    assert " 0/" in result, result
