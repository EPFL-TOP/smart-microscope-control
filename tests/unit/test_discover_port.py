"""``--port`` and the child's checkpoints (design m2 §9.5 and §9.7).

A serial port the owner names reaches a vendor driver, so it is validated
where it enters (the CLI, ``inventory``, the walk), and it travels to the
child as one argument that ``argparse`` cannot read as an option.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import smc.discovery as discovery
from smc.cli import app
from smc.discovery import _mm_child, mm_inventory, os_inventory
from smc.discovery.mm_inventory import ChildResult, check_port
from smc.discovery.models import (
    AdapterDevice,
    AdapterInfo,
    Inventory,
    ProbedDevice,
    ProbeResult,
    SystemInfo,
)

runner = CliRunner(env={"COLUMNS": "200"})


# --- check_port --------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "   ", "C" * 201, "COM3\x00", "COM3\n", "COM\t3", "COM3\x7f"]
)
def test_check_port_refuses_empty_long_and_control_characters(bad: str) -> None:
    with pytest.raises(ValueError, match="port"):
        check_port(bad)


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        (" COM3 ", "COM3"),
        ("\\\\.\\COM10", "\\\\.\\COM10"),
        ("/dev/ttyUSB0", "/dev/ttyUSB0"),
        ("-x", "-x"),
        ("C" * 200, "C" * 200),
    ],
)
def test_check_port_strips_and_accepts_what_a_serial_port_is_called(
    given: str, kept: str
) -> None:
    assert check_port(given) == kept


# --- the child's command -----------------------------------------------------------


def _argv_recording_child(monkeypatch: pytest.MonkeyPatch, argv_path: Path) -> None:
    """A child that records its arguments and writes an empty, valid result."""
    code = (
        "import json, sys\n"
        f"ARGV = {str(argv_path)!r}\n"
        "RESULT = sys.argv[sys.argv.index('--result') + 1]\n"
        "with open(ARGV, 'w', encoding='utf-8') as f:\n"
        "    json.dump(sys.argv[1:], f)\n"
        "with open(RESULT, 'w', encoding='utf-8') as f:\n"
        "    json.dump({'listed': []}, f)\n"
    )
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", code])


def test_mm_section_passes_the_port_to_the_child_as_one_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)

    mm_inventory.mm_section(["DemoCamera"], probe="Utilities", port="-x", timeout_s=30)

    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    # `--port -x` would be read by argparse as an option and refused; one
    # argument, `--port=-x`, is read as the value (design m2 §9.1).
    assert "--port=-x" in argv
    assert "--port" not in argv
    assert "-x" not in argv
    assert argv[argv.index("--probe") + 1] == "Utilities"


def test_mm_section_does_not_pass_a_port_it_was_not_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)

    mm_inventory.mm_section(["DemoCamera"], probe="Utilities", timeout_s=30)

    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    assert not [a for a in argv if a.startswith("--port")]


def test_mm_section_refuses_a_port_check_port_refuses_before_starting_a_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)

    with pytest.raises(ValueError, match="control character"):
        mm_inventory.mm_section(
            ["DemoCamera"], probe="Utilities", port="COM3\n", timeout_s=30
        )

    assert not argv_path.exists()


# --- inventory ---------------------------------------------------------------------


def test_inventory_refuses_a_port_without_a_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def survey_must_not_run(*_: object, **__: object) -> object:
        raise AssertionError("the survey ran")

    monkeypatch.setattr(mm_inventory, "system_section", survey_must_not_run)

    with pytest.raises(ValueError, match="probe_adapter"):
        discovery.inventory(probe_port="COM3", include_os=False)


def test_inventory_refuses_a_bad_port_before_the_survey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def survey_must_not_run(*_: object, **__: object) -> object:
        raise AssertionError("the survey ran")

    monkeypatch.setattr(mm_inventory, "system_section", survey_must_not_run)

    with pytest.raises(ValueError, match="empty"):
        discovery.inventory(probe_adapter="Utilities", probe_port=" ", include_os=False)


def _system() -> SystemInfo:
    from datetime import datetime, timezone

    return SystemInfo(
        hostname="host",
        collected_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
        os="os",
        python="3.13",
        smc="0",
        pymmcore_plus="0",
        mm_install="/mm",
    )


def test_inventory_sends_the_checked_port_to_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_mm_section(
        adapters: list[str], **kwargs: object
    ) -> tuple[list[AdapterInfo], ProbeResult | None, list[str]]:
        seen.update(kwargs)
        return [], None, []

    monkeypatch.setattr(mm_inventory, "system_section", lambda: (_system(), []))
    monkeypatch.setattr(mm_inventory, "mm_section", fake_mm_section)

    discovery.inventory(
        probe_adapter="Utilities", probe_port=" COM3 ", include_os=False
    )

    assert seen["port"] == "COM3"
    assert seen["probe"] == "Utilities"


# --- the child's checkpoint writer -------------------------------------------------


def _row(name: str) -> ProbedDevice:
    return ProbedDevice(name=name, type="Camera", status="ok")


def test_the_childs_checkpoint_writes_the_snapshot_as_an_incomplete_probe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    result = ChildResult(listed=[AdapterInfo(name="Fake", installed=True)])
    checkpoint = _mm_child._checkpoint_writer(result, path)

    checkpoint(ProbeResult(adapter="Fake", devices=[_row("A")], complete=False))

    written = ChildResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert written.probe is not None
    assert [d.name for d in written.probe.devices] == ["A"]
    assert written.probe.complete is False
    # The listing written before the probe is still in the file.
    assert [a.name for a in written.listed] == ["Fake"]

    checkpoint(
        ProbeResult(adapter="Fake", devices=[_row("A"), _row("B")], complete=False)
    )
    again = ChildResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert again.probe is not None
    assert [d.name for d in again.probe.devices] == ["A", "B"]


def test_a_failed_intermediate_write_is_a_progress_line_and_does_not_end_the_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def held_by_antivirus(result: ChildResult, path: Path) -> None:
        raise PermissionError("the file is held")

    monkeypatch.setattr(_mm_child, "_write", held_by_antivirus)
    checkpoint = _mm_child._checkpoint_writer(ChildResult(), tmp_path / "result.json")

    checkpoint(ProbeResult(adapter="Fake", complete=False))

    err = capsys.readouterr().err
    assert "the file is held" in err
    # Not a line that `_last_progress` reads as "what was running when it died".
    assert not err.startswith(mm_inventory._PROGRESS_PREFIXES)


def test_a_checkpoint_swallows_nothing_but_an_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(result: ChildResult, path: Path) -> None:
        raise ValueError("not a file problem")

    monkeypatch.setattr(_mm_child, "_write", broken)
    checkpoint = _mm_child._checkpoint_writer(ChildResult(), tmp_path / "result.json")

    with pytest.raises(ValueError, match="not a file problem"):
        checkpoint(ProbeResult(adapter="Fake", complete=False))


# --- the CLI -----------------------------------------------------------------------


def _inventory_that_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def survey_must_not_run(**_: object) -> Inventory:
        raise AssertionError("the survey ran")

    monkeypatch.setattr(discovery, "inventory", survey_must_not_run)


def test_discover_port_without_probe_adapter_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inventory_that_must_not_run(monkeypatch)

    result = runner.invoke(app, ["discover", "--port", "COM3", "--no-os"])

    assert result.exit_code == 2
    assert "--port needs --probe-adapter" in " ".join(result.output.split())


@pytest.mark.parametrize("bad", ["", "   ", "COM3\n", "C" * 201])
def test_discover_refuses_a_bad_port_before_the_survey(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    _inventory_that_must_not_run(monkeypatch)

    result = runner.invoke(
        app, ["discover", "--probe-adapter", "Utilities", f"--port={bad}", "--no-os"]
    )

    assert result.exit_code == 2
    assert "✗" in result.output
    assert "the port" in " ".join(result.output.split())


def _no_install(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace the two MM calls so that no Micro-Manager install is needed."""
    seen: dict[str, object] = {}

    def fake_mm_section(
        adapters: list[str], **kwargs: object
    ) -> tuple[list[AdapterInfo], ProbeResult | None, list[str]]:
        seen.update(kwargs)
        listed = [
            AdapterInfo(
                name="Utilities",
                installed=True,
                devices=[AdapterDevice(name="Serial", type="Shutter")],
            )
        ]
        return listed, ProbeResult(adapter="Utilities"), []

    monkeypatch.setattr(mm_inventory, "system_section", lambda: (_system(), []))
    monkeypatch.setattr(mm_inventory, "mm_section", fake_mm_section)
    monkeypatch.setattr(os_inventory, "os_sections", lambda notes: ([], [], []))
    return seen


def test_discover_passes_the_port_to_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _no_install(monkeypatch)

    result = runner.invoke(
        app, ["discover", "--probe-adapter", "Utilities", "--port", "COM3", "--no-os"]
    )

    assert result.exit_code == 0, result.output
    assert seen["port"] == "COM3"
    assert seen["probe"] == "Utilities"
    assert "Probing Utilities (port COM3)" in " ".join(result.output.split())


def test_discover_without_a_port_prints_the_banner_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _no_install(monkeypatch)

    result = runner.invoke(app, ["discover", "--probe-adapter", "Utilities", "--no-os"])

    assert result.exit_code == 0, result.output
    assert seen["port"] is None
    text = " ".join(result.output.split())
    assert "Probing Utilities:" in text
    assert "(port" not in text


def test_the_banner_is_text_and_never_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-36: a port or an adapter written like a rich tag must reach the screen.
    _no_install(monkeypatch)

    result = runner.invoke(
        app,
        ["discover", "--probe-adapter", "[bold]X", "--port", "[red]COM3", "--no-os"],
    )

    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "Probing [bold]X (port [red]COM3)" in text
