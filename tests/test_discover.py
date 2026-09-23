"""``smc discover`` on the demo install, and its failure modes.

The adapter work runs in a child process; these tests prove that a child
that prints, crashes or hangs costs at most what it had not yet written
(docs/design/failure-modes.md FM-01 to FM-04).

Every test here pins the adapter lists to Micro-Manager's test adapters: a
microscope PC has the full nightly installed, and listing ``NikonTi2`` from
the default suite would contact the Nikon SDK (FM-06, FM-40, FM-41).
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import smc.cli as smc_cli
import smc.discovery as discovery
import smc.discovery.report as smc_cli_report
from smc.cli import app
from smc.discovery import inventory, mm_inventory, os_inventory, vendors
from smc.discovery._mm_child import _clean_tree, _write
from smc.discovery.mm_inventory import list_adapters
from smc.discovery.models import (
    AdapterDevice,
    AdapterInfo,
    Inventory,
    SerialPort,
    SystemInfo,
)
from smc.discovery.report import JSON_NAME, TEXT_NAME
from smc.discovery.vendors import VendorHint

runner = CliRunner()

#: The adapters every Micro-Manager install ships for testing; nothing else
#: may be listed or loaded by the default suite.
TEST_ADAPTERS = ("DemoCamera", "Utilities", "SequenceTester", "NotificationTester")
#: A name no install has, for the "not installed" path.
ABSENT = "NoSuchAdapterForTests"

FTDI_PORT = SerialPort(device="COM3", vendor_id="0403", product_id="6001")


@pytest.fixture(autouse=True)
def only_test_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lab list = DemoCamera + an absent name; FTDI hints Utilities only."""
    monkeypatch.setattr(discovery, "LAB_ADAPTERS", ("DemoCamera", ABSENT))
    monkeypatch.setattr(
        vendors,
        "VENDOR_HINTS",
        (VendorHint(("usb:0403",), (), "FTDI", ("Utilities",)),),
    )


@pytest.fixture
def one_ftdi_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the OS sections by one FTDI serial port, the same on every OS."""

    def fake(notes: list[str]) -> tuple[list[SerialPort], list[object], list[object]]:
        return [FTDI_PORT.model_copy(deep=True)], [], []

    monkeypatch.setattr(os_inventory, "os_sections", fake)


@pytest.fixture
def need_mm(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


def _fake_child(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Replace the child by a script; ``RESULT`` is the --result path."""
    code = (
        "import json, os, sys, time\n"
        "RESULT = sys.argv[sys.argv.index('--result') + 1]\n"
        "def write_result():\n"
        "    with open(RESULT, 'w', encoding='utf-8') as f:\n"
        "        json.dump({'listed': [{'name': 'Fake', 'installed': True}]}, f)\n"
        + body
    )
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", code])


class _StubCore:
    """Two adapters installed; enumerating ``Broken`` raises like a missing DLL."""

    def getDeviceAdapterNames(self) -> tuple[str, ...]:  # noqa: N802
        return ("Good", "Broken")

    def getAvailableDevices(self, name: str) -> tuple[str, ...]:  # noqa: N802
        if name == "Broken":
            raise RuntimeError('Failed to load device adapter "Broken"')
        return ("Cam",)

    def getAvailableDeviceTypes(self, name: str) -> tuple[int, ...]:  # noqa: N802
        return (2,)

    def getAvailableDeviceDescriptions(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return ("A camera",)


def test_adapter_listing_records_enumeration_errors() -> None:
    good, broken, absent = list_adapters(_StubCore(), ["Good", "Broken", "Absent"])  # type: ignore[arg-type]

    assert good.error is None
    assert [(d.name, d.type) for d in good.devices] == [("Cam", "Camera")]
    assert broken.installed
    assert broken.error is not None
    assert "Broken" in broken.error
    assert absent == AdapterInfo(name="Absent", installed=False)


def test_system_section_records_the_os_build() -> None:
    # Python <= 3.11 calls Windows 11 "10"; the build tells them apart.
    info, _ = mm_inventory.system_section()

    assert info.os_build == platform.version()


def _listed(inv: Inventory) -> dict[str, AdapterInfo]:
    return {a.name: a for a in inv.adapters.listed}


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm", "one_ftdi_port")
def test_default_listing_covers_lab_and_hinted_adapters_only() -> None:
    inv = inventory()

    listed = _listed(inv)
    # The (pinned) lab list plus what the FTDI hint suggests, nothing else.
    assert set(listed) == {"DemoCamera", ABSENT, "Utilities"}
    assert listed["DemoCamera"].installed
    assert "DCam" in {d.name for d in listed["DemoCamera"].devices}
    # An absent adapter is reported as not installed, never as an error.
    assert not listed[ABSENT].installed
    assert listed[ABSENT].error is None
    # Installed but neither lab nor hinted: named, not enumerated.
    assert "SequenceTester" in inv.adapters.installed
    assert "SequenceTester" not in listed
    assert inv.probe is None


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_all_adapters_lists_every_installed_adapter() -> None:
    installed = set(mm_inventory.system_section()[1])
    if not installed <= set(TEST_ADAPTERS):
        pytest.skip("--all-adapters would load vendor adapters on this install")

    inv = inventory(all_adapters=True, include_os=False)

    listed = _listed(inv)
    assert set(inv.adapters.installed) <= set(listed)
    assert all(listed[name].installed for name in inv.adapters.installed)
    assert listed["DemoCamera"].devices
    assert not listed[ABSENT].installed


@pytest.mark.usefixtures("need_mm", "one_ftdi_port")
def test_mm_child_crash_becomes_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_child(
        monkeypatch,
        "sys.stderr.write('probing Fake/FakeCam\\n'); sys.stderr.flush()\n"
        "os._exit(3)\n",
    )

    inv = inventory()

    assert inv.adapters.listed == []
    assert inv.adapters.installed  # the names are kept
    (note,) = [n for n in inv.notes if n.startswith("adapters:")]
    assert "exited 3" in note
    assert "probing Fake/FakeCam" in note
    assert [p.device for p in inv.serial] == ["COM3"]
    assert inv.hints


@pytest.mark.usefixtures("need_mm", "one_ftdi_port")
def test_mm_child_timeout_becomes_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_child(monkeypatch, "time.sleep(30)\n")

    inv = inventory(timeout_s=1)

    assert inv.adapters.listed == []
    assert any("did not finish in 1 s" in n for n in inv.notes)
    assert [p.device for p in inv.serial] == ["COM3"]


@pytest.mark.usefixtures("need_mm")
def test_output_noise_without_newline_does_not_hide_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-01: a DLL printing while loading, with no newline before the result.
    _fake_child(
        monkeypatch,
        "sys.stdout.write('DLL says hi'); sys.stdout.flush()\n"
        "sys.stderr.write('more noise'); sys.stderr.flush()\n"
        "write_result()\n",
    )

    inv = inventory(include_os=False)

    assert [a.name for a in inv.adapters.listed] == ["Fake"]
    assert not [n for n in inv.notes if n.startswith("adapters:")]


@pytest.mark.usefixtures("need_mm")
def test_result_written_before_a_crash_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-02: a crash at teardown must not throw away a result already written.
    _fake_child(monkeypatch, "write_result()\nos._exit(3)\n")

    inv = inventory(include_os=False)

    assert [a.name for a in inv.adapters.listed] == ["Fake"]
    assert any("exited 3" in n for n in inv.notes)


@pytest.mark.usefixtures("need_mm")
def test_result_written_before_a_hang_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-02/FM-04: a child still running after its result is stopped, not waited on.
    _fake_child(monkeypatch, "write_result()\ntime.sleep(30)\n")

    inv = inventory(include_os=False, timeout_s=1)

    assert [a.name for a in inv.adapters.listed] == ["Fake"]
    assert any("did not finish in 1 s" in n for n in inv.notes)


@pytest.mark.usefixtures("need_mm")
def test_mm_child_garbage_result_becomes_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_child(
        monkeypatch,
        "open(RESULT, 'w', encoding='utf-8').write('{not json')\n",
    )

    inv = inventory(include_os=False)

    assert inv.adapters.listed == []
    assert any("could not be read" in n for n in inv.notes)


@pytest.mark.usefixtures("need_mm")
def test_mm_child_that_cannot_start_becomes_a_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [str(tmp_path / "missing")])

    inv = inventory(include_os=False)

    assert inv.adapters.listed == []
    assert any("could not start" in n for n in inv.notes)


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_probe_democamera_loads_hub_first() -> None:
    inv = inventory(probe_adapter="DemoCamera", include_os=False)

    assert inv.probe is not None
    assert inv.probe.error is None
    devices = inv.probe.devices
    assert devices[0].type == "Hub"
    assert devices[0].name == "DHub"
    assert len(devices) > 10
    failed = [(d.name, d.error) for d in devices if not d.ok]
    assert failed == []


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_probe_crash_keeps_the_listing_and_names_the_device() -> None:
    # NotificationTester aborts the process when initialised with the stock
    # test adapters: a real crash inside the child.
    inv = inventory(probe_adapter="NotificationTester", include_os=False)

    crash = [n for n in inv.notes if n.startswith("adapters:")]
    if not crash:
        pytest.skip("NotificationTester did not crash the child on this install")
    assert "DemoCamera" in _listed(inv)
    assert _listed(inv)["DemoCamera"].devices
    assert "probing NotificationTester/" in crash[0]
    assert inv.probe is None
    if sys.platform == "darwin":
        # The crash guard turned the abort into an exit code: no crash report.
        assert "exited 6 (likely SIGABRT, crash guard)" in crash[0]


def test_clean_tree_recovers_cp1252_bytes_and_escapes_lone_surrogates() -> None:
    # FM-05: pymmcore decodes adapter strings with surrogateescape.
    assert _clean_tree("caf\udce9") == "café"
    # A surrogate surrogateescape itself would never produce (outside the
    # DC80-DCFF range) must not raise: it becomes literal escape text.
    assert _clean_tree("\ud800x") == "\\ud800x"
    assert _clean_tree({"a": ["caf\udce9"], "caf\udce9": 1}) == {
        "a": ["café"],
        "café": 1,
    }
    assert _clean_tree(3) == 3


def test_result_with_surrogates_writes_valid_utf8_files(tmp_path: Path) -> None:
    result = mm_inventory.ChildResult(
        listed=[
            AdapterInfo(
                name="Vendor",
                installed=True,
                devices=[
                    AdapterDevice(name="D", type="Camera", description="caf\udce9")
                ],
            )
        ]
    )
    path = tmp_path / "result.json"

    _write(result, path)

    text = path.read_text(encoding="utf-8")  # must not raise
    parsed = mm_inventory.ChildResult.model_validate_json(text)
    assert parsed.listed[0].devices[0].description == "café"


@pytest.mark.skipif(sys.platform != "darwin", reason="the crash guard is macOS-only")
def test_crash_guard_turns_abort_into_exit_6_on_macos() -> None:
    # Without the guard abort() ends in -6 and ReportCrash opens a dialog.
    code = (
        "from smc.discovery._mm_child import _no_error_dialogs\n"
        "_no_error_dialogs()\n"
        "import os\n"
        "os.abort()\n"
    )

    done = subprocess.run([sys.executable, "-c", code], timeout=60, check=False)

    assert done.returncode == 6


def test_child_installs_its_crash_guard_before_importing_pymmcore(
    tmp_path: Path,
) -> None:
    # Run the real main() with the guard replaced by a spy that records which
    # pymmcore modules are already loaded when the guard is installed.
    code = (
        "import sys\n"
        "import smc.discovery._mm_child as child\n"
        "seen = []\n"
        "child._no_error_dialogs = lambda: seen.append(\n"
        "    sorted(m for m in sys.modules if m.startswith('pymmcore')))\n"
        "child.main(['--result', sys.argv[1]])\n"
        "print(seen)\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "result.json")],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=True,
        cwd=tmp_path,
    )

    assert done.stdout.strip().splitlines()[-1] == "[[]]"


@pytest.mark.parametrize(
    ("returncode", "platform_name", "expected"),
    [
        (6, "darwin", "6 (likely SIGABRT, crash guard)"),
        (11, "darwin", "11 (likely SIGSEGV, crash guard)"),
        (3, "darwin", "3"),
        (-6, "darwin", "-6"),
        (6, "linux", "6"),
        (6, "win32", "6"),
    ],
)
def test_exit_note_names_the_likely_signal_on_macos_only(
    returncode: int, platform_name: str, expected: str
) -> None:
    assert mm_inventory.exit_detail(returncode, platform_name) == expected


def _tiny_inventory() -> Inventory:
    return Inventory(
        system=SystemInfo(
            hostname="test",
            collected_at=datetime.now(timezone.utc),
            os="test",
            python="3",
            smc="0",
            pymmcore_plus="0",
        )
    )


def test_discover_out_unwritable_fails_in_one_line_before_the_survey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def survey_must_not_run(**_: object) -> Inventory:
        raise AssertionError("the survey ran before --out was checked")

    monkeypatch.setattr(discovery, "inventory", survey_must_not_run)
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")

    result = runner.invoke(app, ["discover", "--out", str(blocker / "survey")])

    assert result.exit_code == 1
    assert "✗" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_discover_out_files_are_written_before_the_report_is_printed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-30: a broken pipe while printing must not lose the files.
    def broken_pipe(*_: object, **__: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(discovery, "inventory", lambda **_: _tiny_inventory())
    monkeypatch.setattr(smc_cli.console, "print", broken_pipe)
    out = tmp_path / "survey"

    runner.invoke(app, ["discover", "--out", str(out)])

    assert (out / JSON_NAME).is_file()
    assert (out / TEXT_NAME).is_file()


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_discover_out_writes_utf8_json_and_text(tmp_path: Path) -> None:
    out = tmp_path / "surveys" / "nikon-ti2"

    result = runner.invoke(app, ["discover", "--out", str(out), "--no-os"])

    assert result.exit_code == 0, result.output
    inv = Inventory.model_validate_json((out / JSON_NAME).read_text(encoding="utf-8"))
    assert "DemoCamera" in _listed(inv)
    text = (out / TEXT_NAME).read_bytes().decode("utf-8")
    assert "Adapter devices" in text
    assert "DCam" in text


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_discover_runs_here_without_probe() -> None:
    result = runner.invoke(app, ["discover"])

    assert result.exit_code == 0, result.output
    assert "Installed adapters" in result.output
    assert "DemoCamera" in result.output
    assert "Probe of" not in result.output


@pytest.mark.parametrize("value", ["nan", "inf", "0", "5000"])
def test_timeout_outside_1_to_3600_s_is_refused_before_the_survey(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    def survey_must_not_run(**_: object) -> Inventory:
        raise AssertionError("the survey ran with an unbounded timeout")

    monkeypatch.setattr(discovery, "inventory", survey_must_not_run)

    result = runner.invoke(app, ["discover", "--no-os", "--timeout-s", value])

    assert result.exit_code == 2, result.output


@pytest.mark.usefixtures("need_mm")
def test_no_os_says_the_os_sections_were_not_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_child(monkeypatch, "write_result()\n")

    inv = inventory(include_os=False)

    assert any(n.startswith("os:") and "not collected" in n for n in inv.notes)


def test_a_failed_out_write_still_prints_the_survey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The survey took minutes: a disk that fills up after prepare() must cost
    # the files, not the report on screen.
    def disk_full(inv: Inventory, out_dir: Path) -> tuple[Path, Path]:
        raise OSError("disk full")

    monkeypatch.setattr(discovery, "inventory", lambda **_: _tiny_inventory())
    monkeypatch.setattr(smc_cli_report, "write", disk_full)

    result = runner.invoke(app, ["discover", "--out", str(tmp_path / "survey")])

    assert result.exit_code == 1
    assert "Installed adapters" in result.output
    assert "✗" in result.output
    assert "disk full" in result.output


class _UnkillableChild:
    """A child that ignores kill(): only a bounded wait returns."""

    #: Not a real process: killpg/taskkill must find nothing there.
    pid = 999_999_999

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            raise AssertionError("an unbounded wait would hang the survey")
        raise mm_inventory.subprocess.TimeoutExpired("child", timeout)

    def kill(self) -> None:
        pass


def test_a_child_that_survives_kill_does_not_hang_the_survey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mm_inventory.subprocess, "Popen", _UnkillableChild)

    listed, _, notes = mm_inventory.mm_section(["DemoCamera"], timeout_s=1)

    assert listed == []
    assert any("could not be stopped" in n for n in notes)


class _FakeChildProc:
    """A ``Popen``-shaped stand-in for :func:`mm_inventory._kill_child_tree`."""

    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.killed = False

    def kill(self) -> None:
        self.killed = True


def test_kill_child_tree_on_posix_uses_killpg(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both branches of _kill_child_tree only run on their own platform in CI;
    # exercised here directly, on every job, by faking sys.platform.
    monkeypatch.setattr(mm_inventory.sys, "platform", "linux")
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        mm_inventory.os, "killpg", lambda pid, sig: calls.append((pid, sig))
    )

    note = mm_inventory._kill_child_tree(_FakeChildProc())  # type: ignore[arg-type]

    assert note is None
    assert calls == [(4321, mm_inventory.signal.SIGKILL)]


def test_kill_child_tree_on_posix_ignores_a_process_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mm_inventory.sys, "platform", "linux")

    def raise_gone(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(mm_inventory.os, "killpg", raise_gone)

    assert mm_inventory._kill_child_tree(_FakeChildProc()) is None  # type: ignore[arg-type]


def test_kill_child_tree_on_windows_runs_taskkill_then_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mm_inventory.sys, "platform", "win32")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(mm_inventory.subprocess, "run", fake_run)
    child = _FakeChildProc()

    note = mm_inventory._kill_child_tree(child)  # type: ignore[arg-type]

    assert note is None
    assert calls == [["taskkill", "/T", "/F", "/PID", "4321"]]
    assert child.killed


def test_kill_child_tree_on_windows_reports_a_failing_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mm_inventory.sys, "platform", "win32")
    monkeypatch.setattr(
        mm_inventory.subprocess,
        "run",
        lambda command, **k: subprocess.CompletedProcess(
            command, 1, stdout=b"", stderr=b"not found\n"
        ),
    )
    child = _FakeChildProc()

    note = mm_inventory._kill_child_tree(child)  # type: ignore[arg-type]

    assert note is not None
    assert "not found" in note
    assert child.killed


def test_kill_child_tree_on_windows_survives_taskkill_itself_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mm_inventory.sys, "platform", "win32")

    def raise_missing(*a: object, **k: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("taskkill")

    monkeypatch.setattr(mm_inventory.subprocess, "run", raise_missing)
    child = _FakeChildProc()

    note = mm_inventory._kill_child_tree(child)  # type: ignore[arg-type]

    assert note is not None
    assert child.killed


def test_adapter_child_runs_without_a_console_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-22: a child sharing the console can repaint it.
    _fake_child(monkeypatch, "write_result()\n")
    calls: list[dict[str, object]] = []
    real_popen = mm_inventory.subprocess.Popen

    class SpyPopen(real_popen):  # type: ignore[misc, type-arg]
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(kwargs)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mm_inventory.subprocess, "Popen", SpyPopen)

    mm_inventory.mm_section(["DemoCamera"], timeout_s=10)

    assert calls
    assert calls[0]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # So the whole tree can be killed together (FM-35).
    assert calls[0]["start_new_session"] is True


def test_timeout_kills_the_adapter_childs_helper_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-35: kill() alone leaves a helper process the child started running.
    heartbeat = tmp_path / "heartbeat.txt"
    body = rf"""
write_result()
import subprocess, sys, time
heartbeat = {str(heartbeat)!r}
grandchild_code = (
    "import time\n"
    "while True:\n"
    "    with open(" + repr(heartbeat) + ", 'a') as f:\n"
    "        f.write('x')\n"
    "        f.flush()\n"
    "    time.sleep(0.1)\n"
)
subprocess.Popen([sys.executable, "-c", grandchild_code])
time.sleep(30)
"""
    _fake_child(monkeypatch, body)

    mm_inventory.mm_section(["DemoCamera"], timeout_s=1)

    assert heartbeat.exists(), "the grandchild never started writing its heartbeat"
    last = heartbeat.stat().st_size
    assert last > 0

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        time.sleep(0.5)
        current = heartbeat.stat().st_size
        assert current == last, "the heartbeat kept growing: the grandchild lives"
        last = current


class _InterruptedChild:
    """A child whose ``wait()`` is interrupted, like a Ctrl-C at the terminal."""

    #: Not a real process: killpg/taskkill must find nothing there.
    pid = 999_999_998

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        raise KeyboardInterrupt

    def kill(self) -> None:
        pass


def test_interrupt_while_waiting_kills_the_child_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-35: Ctrl-C while waiting must not leave a helper process running.
    killed: list[int] = []
    monkeypatch.setattr(mm_inventory.subprocess, "Popen", _InterruptedChild)
    if sys.platform == "win32":
        monkeypatch.setattr(
            mm_inventory.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0),
        )
        monkeypatch.setattr(
            _InterruptedChild, "kill", lambda self: killed.append(self.pid)
        )
    else:
        monkeypatch.setattr(
            mm_inventory.os, "killpg", lambda pid, sig: killed.append(pid)
        )

    with pytest.raises(KeyboardInterrupt):
        mm_inventory.mm_section(["DemoCamera"], timeout_s=5)

    assert killed == [_InterruptedChild.pid]
