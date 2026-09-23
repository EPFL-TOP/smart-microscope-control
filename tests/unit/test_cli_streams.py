"""#42: redirected output on Windows is cp1252 and must not crash on ✓ / ✗."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smc.cli import tolerate_unencodable_output


def test_stream_fix_survives_a_cp1252_stream() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")

    tolerate_unencodable_output(stream)
    stream.write("✓ done\n")
    stream.flush()

    assert stream.encoding == "cp1252"  # the encoding itself is never changed
    assert raw.getvalue() == b"? done\n"


def test_stream_fix_leaves_utf8_streams_alone() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

    tolerate_unencodable_output(stream)

    assert stream.errors == "strict"


def test_stream_fix_ignores_streams_without_reconfigure() -> None:
    tolerate_unencodable_output(io.StringIO())  # no exception


def test_doctor_redirected_with_cp1252_exits_0(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"
    out = tmp_path / "doctor.txt"

    with out.open("wb") as f:
        done = subprocess.run(
            [sys.executable, "-m", "smc.cli", "doctor"],
            stdout=f,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
            check=False,
        )

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"loads and answers" in out.read_bytes()
