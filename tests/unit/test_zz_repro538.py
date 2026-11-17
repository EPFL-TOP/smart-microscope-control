"""TEMPORARY (#538): run the exit test's child many times on CI and name the hang.

Removed before the PR is ready: it only exists to reproduce the hang on Linux.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

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

_EPILOGUE = """
import threading as _th
_mark("main-done threads=%r" % [(x.name, x.daemon) for x in _th.enumerate()])
"""


def test_repro_538_the_exit_child_never_hangs(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    script.write_text(_PRELUDE + t._CHILD + _EPILOGUE, encoding="utf-8")
    profile = Path(__file__).resolve().parents[2] / "profiles" / "zeiss-observer7.toml"

    def one(i: int) -> tuple[int, float, int, str]:
        d = tmp_path / f"run{i}"
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
        for i, dt, rc, err in ex.map(one, range(N)):
            times.append(dt)
            if rc != 0 or dt > HANG_S:
                bad.append(f"--- run {i}: {dt:.1f} s rc={rc}\n{err}")
    times.sort()
    summary = (
        f"{len(bad)}/{N} bad; median {times[len(times) // 2]:.2f} s, "
        f"max {times[-1]:.2f} s"
    )
    assert not bad, summary + "\n" + "\n".join(bad[:3])
