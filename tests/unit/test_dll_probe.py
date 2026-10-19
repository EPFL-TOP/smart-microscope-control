"""The loader probe: what Windows says when an adapter's DLL will not load.

Every child here is ``CHILD_COMMAND`` patched to ``[sys.executable, "-c", …]``:
no real DLL is ever loaded, so the default suite stays away from vendor code
on a microscope PC (FM-40). The file the probe is asked about is a few bytes
in ``tmp_path``; the canned child never opens it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from smc.discovery import _dll_child, dll_probe, mm_inventory
from smc.discovery.dll_probe import (
    WINERROR_HINTS,
    DllProbe,
    describe_winerror,
    probe_dll,
)

NOT_FOUND = "The specified module could not be found."


@pytest.fixture
def dll(tmp_path: Path) -> Path:
    """A file that exists, standing for the adapter's DLL; no child loads it."""
    path = tmp_path / "mmgr_dal_Fake.dll"
    path.write_bytes(b"MZ")
    return path


def fake_child(
    monkeypatch: pytest.MonkeyPatch, body: str, *, windows: bool = True
) -> None:
    """Replace the probe child by a script; ``DLL`` is the path it was asked about."""
    code = "import json, os, sys, time\nDLL = sys.argv[1]\n" + body
    monkeypatch.setattr(dll_probe, "CHILD_COMMAND", [sys.executable, "-c", code])
    monkeypatch.setattr(dll_probe, "_WINDOWS", windows)


def verdict_line(**fields: object) -> str:
    return f"print(json.dumps({fields!r}), flush=True)\n"


# -- what the code means --------------------------------------------------------


def test_winerror_hints_name_the_four_loader_failures() -> None:
    assert set(WINERROR_HINTS) == {126, 127, 193, 1114}
    assert "depends on" in WINERROR_HINTS[126]
    assert "entry point" in WINERROR_HINTS[127]
    assert "architecture" in WINERROR_HINTS[193]
    assert "initialisation" in WINERROR_HINTS[1114]


def test_describe_winerror_adds_the_meaning_of_a_known_code_only() -> None:
    assert describe_winerror(126, NOT_FOUND) == (
        f"WinError 126: {NOT_FOUND} — {WINERROR_HINTS[126]}"
    )
    # An unknown code is reported as Windows said it, with no invented hint.
    assert describe_winerror(5, "Access is denied.") == "WinError 5: Access is denied."
    # No code at all (an OSError that carries none): the text alone.
    assert describe_winerror(None, "no such file") == "no such file"


def test_describe_winerror_keeps_a_localised_message_on_one_line() -> None:
    text = describe_winerror(126, "Le module spécifié\r\nest introuvable.")
    assert "\n" not in text
    assert "\r" not in text
    assert "Le module spécifié est introuvable." in text


# -- when no child is started ----------------------------------------------------


def test_off_windows_no_child_is_started(
    monkeypatch: pytest.MonkeyPatch, dll: Path, tmp_path: Path
) -> None:
    started = tmp_path / "started"
    fake_child(
        monkeypatch,
        f"open({str(started)!r}, 'w').close()\n" + verdict_line(loads=True),
        windows=False,
    )

    probe = probe_dll(dll)

    assert probe == DllProbe(
        dll, None, None, "not Windows: a .dll cannot be loaded here"
    )
    assert not started.exists()


def test_a_missing_file_starts_no_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = tmp_path / "started"
    fake_child(monkeypatch, f"open({str(started)!r}, 'w').close()\n")
    missing = tmp_path / "mmgr_dal_Absent.dll"

    probe = probe_dll(missing)

    assert probe == DllProbe(missing, None, None, f"{missing} does not exist")
    assert not started.exists()


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf")])
def test_a_timeout_that_is_not_a_finite_positive_number_is_refused(
    dll: Path, timeout_s: float
) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        probe_dll(dll, timeout_s=timeout_s)


# -- what the child says ---------------------------------------------------------


def test_the_childs_verdict_is_read(monkeypatch: pytest.MonkeyPatch, dll: Path) -> None:
    fake_child(monkeypatch, verdict_line(loads=False, winerror=126, strerror=NOT_FOUND))

    probe = probe_dll(dll)

    assert probe.path == dll
    assert probe.loads is False
    assert probe.winerror == 126
    assert probe.detail == describe_winerror(126, NOT_FOUND)
    assert WINERROR_HINTS[126] in probe.detail


def test_a_dll_that_loads_is_a_probe_that_loads(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    fake_child(monkeypatch, verdict_line(loads=True))

    probe = probe_dll(dll)

    assert probe.loads is True
    assert probe.winerror is None


def test_a_failure_without_a_code_is_still_a_failure(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    fake_child(monkeypatch, verdict_line(loads=False, winerror=None, strerror="odd"))

    probe = probe_dll(dll)

    assert (probe.loads, probe.winerror, probe.detail) == (False, None, "odd")


def test_a_verdict_written_before_a_crash_at_exit_is_kept(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    # FM-02: a vendor DllMain that crashes when the process ends costs the
    # exit code, not the answer the child had already written.
    fake_child(
        monkeypatch,
        verdict_line(loads=False, winerror=193, strerror="bad image")
        + "os._exit(139)\n",
    )

    probe = probe_dll(dll)

    assert (probe.loads, probe.winerror) == (False, 193)


def test_a_dll_that_prints_before_the_verdict_does_not_hide_it(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    # FM-01: a DLL writes to stdout while it loads, with or without a newline.
    fake_child(
        monkeypatch,
        "sys.stdout.write('banner without a newline')\n"
        "sys.stdout.write('\\n')\n"
        + verdict_line(loads=True)
        + "sys.stdout.write('and one more after')\n",
    )

    assert probe_dll(dll).loads is True


def test_a_child_that_crashes_is_reported_with_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    fake_child(monkeypatch, "os._exit(3)\n")

    probe = probe_dll(dll)

    assert probe.loads is None
    assert probe.winerror is None
    assert probe.detail == (
        f"the probe child ended with exit code "
        f"{mm_inventory.exit_detail(3, sys.platform)} and no verdict"
    )


@pytest.mark.parametrize(
    "body",
    [
        "print('hello')\n",
        "print(json.dumps({'loads': 'yes'}))\n",
        "print(json.dumps([True]))\n",
        "print('{not json')\n",
        "",
    ],
    ids=["text", "loads-not-bool", "not-an-object", "broken-json", "silence"],
)
def test_output_that_is_not_the_protocol_is_no_verdict(
    monkeypatch: pytest.MonkeyPatch, dll: Path, body: str
) -> None:
    fake_child(monkeypatch, body)

    probe = probe_dll(dll)

    assert probe.loads is None
    assert probe.detail == "the probe child ended with exit code 0 and no verdict"


def test_a_child_that_cannot_start_is_a_probe_without_a_verdict(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    monkeypatch.setattr(dll_probe, "_WINDOWS", True)
    monkeypatch.setattr(
        dll_probe, "CHILD_COMMAND", [str(dll.parent / "no-such-python")]
    )

    probe = probe_dll(dll)

    assert probe.loads is None
    assert probe.detail.startswith("the probe child could not start: ")


def test_a_hung_child_is_killed_and_reported(
    monkeypatch: pytest.MonkeyPatch, dll: Path, tmp_path: Path
) -> None:
    # FM-35: a DllMain that hangs may sit in a helper process; the whole tree
    # goes. The grandchild's own lifetime is bounded (60 s) so that a failing
    # run does not leak it (FM-86).
    heartbeat = tmp_path / "heartbeat.txt"
    fake_child(
        monkeypatch,
        rf"""
import subprocess
grandchild = (
    "import time\n"
    "for _ in range(600):\n"
    "    with open(" + repr({str(heartbeat)!r}) + ", 'a') as f:\n"
    "        f.write('x')\n"
    "        f.flush()\n"
    "    time.sleep(0.1)\n"
)
subprocess.Popen([sys.executable, "-c", grandchild])
time.sleep(60)
""",
    )

    started = time.monotonic()
    probe = probe_dll(dll, timeout_s=1)
    elapsed = time.monotonic() - started

    assert elapsed < 10
    assert probe.loads is None
    assert probe.detail == "did not return within 1 s: its initialisation hung"
    assert heartbeat.exists(), "the grandchild never started writing its heartbeat"

    # The kill only queues the signal: a read right after the probe returns can
    # still see one more tick. Wait until the heartbeat has been quiet over a
    # window of 1 s and fail only at the deadline (FM-86). os.kill(pid, 0) is
    # no liveness probe: it kills on Windows and answers "alive" for a zombie.
    deadline = time.monotonic() + 10
    last = heartbeat.stat().st_size
    assert last > 0
    while True:
        time.sleep(1.0)
        current = heartbeat.stat().st_size
        if current == last:
            return
        last = current
        if time.monotonic() >= deadline:
            pytest.fail("the heartbeat kept growing 10 s after the kill: it lives")


def test_a_child_that_cannot_be_stopped_says_so(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    # FM-35: a tree kill that failed is not "was stopped".
    fake_child(monkeypatch, "time.sleep(60)\n")
    monkeypatch.setattr(
        mm_inventory,
        "_kill_child_tree",
        lambda child: (child.kill(), "taskkill failed")[1],
    )

    probe = probe_dll(dll, timeout_s=1)

    assert probe.loads is None
    assert "did not return within 1 s" in probe.detail
    assert "could not be stopped" in probe.detail
    assert "taskkill failed" in probe.detail


def test_the_wait_is_sliced(monkeypatch: pytest.MonkeyPatch, dll: Path) -> None:
    # FM-83: on Windows one long wait is one blocking call that a Ctrl-C cannot
    # end, and the child (CREATE_NO_WINDOW) never receives the Ctrl-C.
    fake_child(monkeypatch, "time.sleep(1.5)\n" + verdict_line(loads=True))
    waits: list[float | None] = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):  # type: ignore[misc, type-arg]
        def wait(self, timeout: float | None = None) -> int:
            if self.args[0] == sys.executable:
                waits.append(timeout)
            return super().wait(timeout)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)

    assert probe_dll(dll, timeout_s=30).loads is True

    assert len(waits) >= 2
    assert all(w is not None for w in waits)
    assert max(w for w in waits if w is not None) <= 1.0


def test_an_interrupt_while_waiting_kills_the_child_tree(
    monkeypatch: pytest.MonkeyPatch, dll: Path
) -> None:
    killed: list[int] = []
    fake_child(monkeypatch, "time.sleep(60)\n")

    def interrupted(child: subprocess.Popen[bytes], timeout_s: float) -> int:
        raise KeyboardInterrupt

    def record_kill(child: subprocess.Popen[bytes]) -> None:
        killed.append(child.pid)
        child.kill()
        child.wait(timeout=10)

    monkeypatch.setattr(mm_inventory, "_wait_in_slices", interrupted)
    monkeypatch.setattr(mm_inventory, "_kill_child_tree", record_kill)

    with pytest.raises(KeyboardInterrupt):
        probe_dll(dll)

    assert len(killed) == 1


def test_the_child_runs_beside_nothing_of_ours(
    monkeypatch: pytest.MonkeyPatch, dll: Path, tmp_path: Path
) -> None:
    # FM-33 and FM-22: a neutral working directory, no console window.
    cwd_file = tmp_path / "cwd.txt"
    fake_child(
        monkeypatch,
        f"open({str(cwd_file)!r}, 'w', encoding='utf-8').write(os.getcwd())\n"
        + verdict_line(loads=True),
    )
    calls: list[dict[str, object]] = []
    real_popen = subprocess.Popen

    class SpyPopen(real_popen):  # type: ignore[misc, type-arg]
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(kwargs)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "Popen", SpyPopen)

    probe_dll(dll)

    assert Path(cwd_file.read_text(encoding="utf-8")).resolve() != Path.cwd().resolve()
    assert calls[0]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert calls[0]["stdin"] == subprocess.DEVNULL
    # Files, never pipes (FM-04): a helper that inherits a pipe keeps it open.
    assert calls[0]["stdout"] not in (subprocess.PIPE, None)
    assert calls[0]["stderr"] not in (subprocess.PIPE, None)


def test_the_adapter_folders_come_first_on_the_childs_path(
    monkeypatch: pytest.MonkeyPatch, dll: Path, tmp_path: Path
) -> None:
    out = tmp_path / "path.json"
    fake_child(
        monkeypatch,
        f"open({str(out)!r}, 'w', encoding='utf-8')"
        ".write(json.dumps(os.environ['PATH'].split(os.pathsep)))\n"
        + verdict_line(loads=True),
    )
    monkeypatch.setenv(
        "PATH", os.pathsep.join(["tail-entry", os.environ.get("PATH", "")])
    )
    first, second = tmp_path / "first", tmp_path / "second"
    # A folder whose name holds the separator reads back as two entries (FM-81).
    odd = Path(f"odd{os.pathsep}dir")
    before = os.environ["PATH"]

    probe_dll(dll, extra_path=[first, odd, second])

    seen = json.loads(out.read_text(encoding="utf-8"))
    assert seen[:2] == [str(first), str(second)]
    assert "tail-entry" in seen[2:]
    assert "odd" not in seen
    assert "dir" not in seen
    # Our own process keeps the PATH it had.
    assert os.environ["PATH"] == before


# -- the child itself -----------------------------------------------------------


def test_the_child_writes_a_winerror_as_ascii_json() -> None:
    # A French Windows says this; the child's stdout is the ANSI code page (FM-23).
    exc = OSError(22, "Le module spécifié est introuvable.")
    exc.winerror = 126  # type: ignore[attr-defined]  # only Windows sets it itself

    line = _dll_child.encode(_dll_child.verdict(exc))

    assert line.isascii()
    assert "\n" not in line
    assert json.loads(line) == {
        "loads": False,
        "winerror": 126,
        "strerror": "Le module spécifié est introuvable.",
    }


def test_the_child_says_loads_for_a_dll_that_loaded() -> None:
    assert _dll_child.verdict(None) == {"loads": True}


def test_the_child_reports_an_error_without_a_code() -> None:
    assert _dll_child.verdict(OSError("plain")) == {
        "loads": False,
        "winerror": None,
        "strerror": "plain",
    }


def test_the_child_main_prints_one_verdict_line_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dll: Path
) -> None:
    seen: list[Path] = []

    def load(path: Path) -> OSError | None:
        seen.append(path)
        return None

    monkeypatch.setattr(_dll_child, "_load", load)

    assert _dll_child.main([str(dll)]) == 0

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert [json.loads(ln) for ln in lines] == [{"loads": True}]
    assert seen == [dll]
