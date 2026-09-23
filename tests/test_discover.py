"""``smc discover`` on the demo install, and its failure modes.

The adapter work runs in a child process; these tests prove that a child
that prints, crashes or hangs costs at most what it had not yet written
(docs/design/failure-modes.md FM-01 to FM-04).

Every test here pins the adapter lists to Micro-Manager's test adapters: a
microscope PC has the full nightly installed, and listing ``NikonTi2`` from
the default suite would contact the Nikon SDK (FM-06, FM-40, FM-41).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import smc.cli as smc_cli
import smc.discovery as discovery
from smc.cli import app
from smc.discovery import inventory, mm_inventory, os_inventory, vendors
from smc.discovery.mm_inventory import list_adapters
from smc.discovery.models import AdapterInfo, Inventory, SerialPort, SystemInfo
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
    assert [(d.name, d.type) for d in good.devices] == [("Cam", "CameraDevice")]
    assert broken.installed
    assert broken.error is not None
    assert "Broken" in broken.error
    assert absent == AdapterInfo(name="Absent", installed=False)


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
    assert devices[0].type == "HubDevice"
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
