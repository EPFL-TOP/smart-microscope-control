"""#42: redirected output on Windows is cp1252 and must not crash on ✓ / ✗."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import smc.cli as smc_cli
from smc.cli import tolerate_unencodable_output


def test_stream_fix_survives_a_cp1252_stream() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")

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


def test_main_reconfigures_only_strict_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    # FM-42: --help exits via Click before the Typer callback (_main) runs,
    # so the fix has to live in main(), the process entry point, too.
    monkeypatch.setattr(smc_cli, "app", lambda: None)
    strict = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    lenient = io.TextIOWrapper(
        io.BytesIO(), encoding="utf-8", errors="backslashreplace"
    )
    monkeypatch.setattr(smc_cli.sys, "stdout", strict)
    monkeypatch.setattr(smc_cli.sys, "stderr", lenient)

    smc_cli.main()

    assert strict.errors == "replace"
    assert lenient.errors == "backslashreplace"  # already UTF-8: left alone


def test_help_survives_a_cp1252_redirect(tmp_path: Path) -> None:
    # FM-42: a subprocess test with PYTHONIOENCODING=cp1252 and stdout to a
    # file, because CliRunner captures output in a way that hides encoding
    # problems. Today's --help text is ASCII-only (rich degrades its own box
    # drawing when the stream cannot show it), so this proves no crash; the
    # in-process test above is the proof that the reconfiguration itself runs.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"
    out = tmp_path / "help.txt"

    with out.open("wb") as f:
        done = subprocess.run(
            [sys.executable, "-m", "smc.cli", "--help"],
            stdout=f,
            stderr=subprocess.PIPE,
            env=env,
            timeout=60,
            check=False,
        )

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"Usage" in out.read_bytes()


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
