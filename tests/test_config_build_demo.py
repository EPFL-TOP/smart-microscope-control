"""``smc config build``: the build child, its runner and the command (design m2 §10).

The real child runs on Micro-Manager's own test adapters (``DemoCamera`` is a
hub with 19 peripherals, ``SequenceTester`` a hub with 14 and two cameras);
the failures are shown by fake children that write a result and die. Only the
test adapters are named, never a vendor's (FM-40), and every output file is
under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest
from pymmcore_plus import DeviceType
from typer.testing import CliRunner, Result

from smc.cli import app
from smc.discovery import _mm_child, diagnosis, mm_inventory
from smc.discovery.mm_inventory import BuildRun, ChildResult, build_section
from smc.discovery.models import BuildScan, BuiltDevice, ProbedDevice, ProbeResult
from smc.hardware import core as core_mod
from smc.hardware.backends import nikon

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


@pytest.fixture
def need_mm(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


def _ok(name: str, type_name: str = "Generic") -> ProbedDevice:
    return ProbedDevice(name=name, type=type_name, status="ok")


def _fake_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scan: BuildScan | None,
    *,
    stderr: str = "",
    exit_code: int = 0,
    then: str = "",
) -> None:
    """A child that writes ``scan`` as its result, says what it was doing, and ends.

    The result is copied from a file, not put on the command line: a long one
    would not fit on Windows' (FM-35's neighbour, FM-101's lesson of what a
    fixture's size may be).
    """
    payload = tmp_path / "payload.json"
    if scan is not None:
        payload.write_text(
            json.dumps(ChildResult(build=scan).model_dump(mode="json")),
            encoding="utf-8",
        )
    code = (
        "import os, shutil, sys, time\n"
        "RESULT = sys.argv[sys.argv.index('--result') + 1]\n"
        + (f"shutil.copyfile({str(payload)!r}, RESULT)\n" if scan is not None else "")
        + f"sys.stderr.write({stderr!r}); sys.stderr.flush()\n"
        + then
        + f"os._exit({exit_code})\n"
    )
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", code])


def _argv_recording_child(monkeypatch: pytest.MonkeyPatch, argv_path: Path) -> None:
    code = (
        "import json, sys\n"
        "RESULT = sys.argv[sys.argv.index('--result') + 1]\n"
        f"json.dump(sys.argv[1:], open({str(argv_path)!r}, 'w', encoding='utf-8'))\n"
        "json.dump({'listed': []}, open(RESULT, 'w', encoding='utf-8'))\n"
    )
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", code])


# --- the build child, on the demo adapters --------------------------------------------


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_the_build_child_reads_the_parent_and_the_labels_of_what_initialised() -> None:
    run = build_section("DemoCamera")

    assert run.notes == []
    assert run.stopped_at is None
    scan = run.scan
    assert scan is not None
    assert scan.complete
    assert scan.camera is None
    assert scan.stand.error is None
    ok_rows = [d.name for d in scan.stand.devices if d.status == "ok"]
    # One built device per row that initialised, in row order, under its own name.
    assert [d.label for d in scan.devices] == ok_rows
    assert all(d.name == d.label and d.library == "DemoCamera" for d in scan.devices)
    by_label = {d.label: d for d in scan.devices}
    assert by_label["DHub"].parent is None
    assert by_label["DWheel"].parent == "DHub"
    assert by_label["DWheel"].state_labels[:2] == ["State-0", "State-1"]
    assert len(by_label["DObjective"].state_labels) == 6
    assert by_label["DObjective"].state_labels[0] == "Objective-A"
    # Only a State device has labels to read.
    assert by_label["DStage"].state_labels == []


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_the_build_child_walks_the_camera_adapter_on_the_same_core() -> None:
    run = build_section("DemoCamera", camera="SequenceTester")

    scan = run.scan
    assert scan is not None
    assert scan.complete
    assert scan.camera is not None
    assert scan.camera.adapter == "SequenceTester"
    libraries = [d.library for d in scan.devices]
    # The stand's devices first, then the camera adapter's, each in row order.
    assert libraries == sorted(libraries, key=["DemoCamera", "SequenceTester"].index)
    assert {"DemoCamera", "SequenceTester"} == set(libraries)
    assert "TCamera-0" in {d.label for d in scan.devices}


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_a_skipped_device_is_not_loaded_by_the_build_child() -> None:
    run = build_section("DemoCamera", skip=["DWheel"])

    scan = run.scan
    assert scan is not None
    (row,) = [d for d in scan.stand.devices if d.name == "DWheel"]
    assert (row.status, row.reason) == ("not-probed", "left out: --skip DWheel")
    assert "DWheel" not in {d.label for d in scan.devices}


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_an_adapter_folder_that_is_not_a_folder_is_a_scan_with_an_error(
    tmp_path: Path,
) -> None:
    run = build_section("DemoCamera", search_paths=[tmp_path / "not-there"])

    # The child ends well and says what it found: a stack trace would say less.
    assert run.notes == []
    scan = run.scan
    assert scan is not None
    assert scan.complete
    assert scan.stand.error is not None
    assert "adapter search path is not a directory" in scan.stand.error
    assert scan.devices == []


# --- what reaches the child --------------------------------------------------------------


def test_every_value_reaches_the_child_as_one_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)
    folder = tmp_path / "adapters"

    build_section(
        "NikonTi2",
        camera="Cam",
        skip=["-x", " TIDiaLamp "],
        search_paths=[folder],
        timeout_s=30,
    )

    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    # `--skip -x` would be read by argparse as an option and refused.
    assert "--build=NikonTi2" in argv
    assert "--camera=Cam" in argv
    assert "--skip=-x" in argv
    assert "--skip=TIDiaLamp" in argv
    assert f"--search-path={folder}" in argv
    assert not [a for a in argv if a in ("--skip", "--build", "-x", str(folder))]


def test_a_camera_that_is_not_asked_for_is_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)

    build_section("NikonTi2", timeout_s=30)

    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    assert not [a for a in argv if a.startswith(("--camera", "--skip", "--search"))]


@pytest.mark.parametrize(
    ("skip", "why"), [(["Dia\n"], "control"), ([""], "empty"), (["ok", "  "], "empty")]
)
def test_a_skip_name_that_check_device_name_refuses_starts_no_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, skip: list[str], why: str
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)

    with pytest.raises(ValueError, match=why):
        build_section("NikonTi2", skip=skip, timeout_s=30)

    assert not argv_path.exists()


def test_an_empty_stand_name_starts_no_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)

    # `--build=` would be an adapter that is not installed, but nothing says so.
    with pytest.raises(ValueError, match="empty"):
        build_section("  ", timeout_s=30)

    assert not argv_path.exists()


# --- a child that dies, hangs or says nothing ------------------------------------------


def _partial(stand: str = "DemoCamera") -> BuildScan:
    return BuildScan(
        stand=ProbeResult(adapter=stand, devices=[_ok("DHub", "Hub")], complete=False),
        complete=False,
    )


def test_a_child_that_dies_mid_build_leaves_the_partial_scan_and_names_the_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch,
        tmp_path,
        _partial(),
        stderr="listing DemoCamera\nprobing DemoCamera/DWheel\n",
        exit_code=3,
    )

    run = build_section("DemoCamera", timeout_s=30)

    assert run.scan is not None
    assert not run.scan.complete
    assert [d.name for d in run.scan.stand.devices] == ["DHub"]
    assert run.stopped_at == "DWheel"
    (note,) = run.notes
    assert note.startswith("build: the build child exited 3")
    assert "while probing DemoCamera/DWheel" in note
    # mm_section's wording is its own: nobody keeps adapter names here.
    assert "adapter names kept" not in note
    assert "adapter child" not in note


def test_a_child_that_dies_reading_a_device_names_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch,
        tmp_path,
        _partial(),
        stderr="probing DemoCamera/DWheel\nreading DemoCamera/DHub\n",
        exit_code=3,
    )

    run = build_section("DemoCamera", timeout_s=30)

    assert run.stopped_at == "DHub"


@pytest.mark.parametrize(
    ("line", "device"),
    [
        ("probing DemoCamera/DWheel", "DWheel"),
        ("reading DemoCamera/DObjective", "DObjective"),
        # The hub is the device the walk is on while it asks for the peripherals.
        ("probing NikonTi2/Ti2-E__0: listing its peripherals", "Ti2-E__0"),
        ("probing Adapter/a device/with a slash", "a device/with a slash"),
        # A crash while listing is the adapter's, not a device's: nothing to skip.
        ("listing NikonTi2", None),
        ("probing NikonTi2", None),
        ("checkpoint skipped: the result file could not be written: x", None),
        (None, None),
    ],
)
def test_the_device_a_child_was_on_comes_from_its_last_progress_line(
    line: str | None, device: str | None
) -> None:
    assert mm_inventory._stopped_at(line) == device


def test_a_child_that_finished_names_no_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    whole = _partial()
    whole.complete = True
    _fake_child(
        monkeypatch, tmp_path, whole, stderr="reading DemoCamera/DHub\n", exit_code=0
    )

    run = build_section("DemoCamera", timeout_s=30)

    assert run.notes == []
    assert run.stopped_at is None
    assert run.scan is not None
    assert run.scan.complete


def test_a_death_after_the_whole_scan_was_written_costs_a_note_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-02: the child dies unloading the devices, after its last write.
    whole = _partial()
    whole.complete = True
    _fake_child(monkeypatch, tmp_path, whole, exit_code=6)

    run = build_section("DemoCamera", timeout_s=30)

    assert run.scan is not None
    assert run.scan.complete
    assert any("exited 6" in n for n in run.notes)


def test_a_child_that_hangs_is_stopped_and_the_device_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch,
        tmp_path,
        _partial(),
        stderr="probing DemoCamera/DWheel\n",
        then="time.sleep(30)\n",
    )

    run = build_section("DemoCamera", timeout_s=1)

    assert run.scan is not None
    assert not run.scan.complete
    assert run.stopped_at == "DWheel"
    (note,) = run.notes
    assert "build: the build child did not finish in 1 s and was stopped" in note


def test_a_child_that_says_nothing_has_a_note_and_no_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(monkeypatch, tmp_path, None, exit_code=0)

    run = build_section("DemoCamera", timeout_s=30)

    assert run.scan is None
    assert run.notes == ["build: the build child wrote no result"]
    assert run.stopped_at is None


def test_a_child_that_dies_before_its_first_write_has_a_note_and_no_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(monkeypatch, tmp_path, None, stderr="listing DemoCamera\n", exit_code=3)

    run = build_section("DemoCamera", timeout_s=30)

    assert run.scan is None
    (note,) = run.notes
    assert note == "build: the build child exited 3 while listing DemoCamera"
    # The listing is the adapter's own step: there is no device to leave out.
    assert run.stopped_at is None


def test_a_result_the_child_cannot_have_written_is_a_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mm_inventory,
        "CHILD_COMMAND",
        [
            sys.executable,
            "-c",
            "import sys; open(sys.argv[sys.argv.index('--result') + 1], 'w').write('{no')",
        ],
    )

    run = build_section("DemoCamera", timeout_s=30)

    assert run.scan is None
    (note,) = run.notes
    assert note.startswith("build: the build child's result could not be read")


def test_a_child_that_cannot_start_is_a_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [str(tmp_path / "missing")])

    run = build_section("DemoCamera", timeout_s=30)

    assert run.scan is None
    (note,) = run.notes
    assert note.startswith("build: the build child could not start")


# --- the child in process, on a stub core ------------------------------------------------


class _StubCore:
    """Adapter ``Fake``: a hub ``H`` and under it a camera, a shutter and a wheel.

    Adapter ``Cam`` is a camera adapter: a hub ``CH`` and a camera ``CCam``.
    ``dies_on`` names the device whose initialisation ends the process, as a
    vendor driver that crashes does (``SystemExit`` passes the walk's
    ``except Exception``, like the process dying under it).
    """

    def __init__(
        self,
        *,
        parent_error: bool = False,
        labels_error: bool = False,
        dies_on: str | None = None,
    ) -> None:
        self.parent_error = parent_error
        self.labels_error = labels_error
        self.dies_on = dies_on
        self.parents: dict[str, str] = {}
        self.initialised: set[str] = set()
        self.loaded: list[str] = []
        self.released = False

    _TYPES: ClassVar[dict[str, int]] = {
        "H": int(DeviceType.HubDevice),
        "Cam": int(DeviceType.CameraDevice),
        "Shut": int(DeviceType.ShutterDevice),
        "Wheel": int(DeviceType.StateDevice),
        "CH": int(DeviceType.HubDevice),
        "CCam": int(DeviceType.CameraDevice),
    }
    _ADAPTERS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Fake": ("H", "Cam", "Shut", "Wheel"),
        "Cam": ("CH", "CCam"),
    }

    def getDeviceAdapterNames(self) -> tuple[str, ...]:  # noqa: N802
        return tuple(self._ADAPTERS)

    def getAvailableDevices(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return self._ADAPTERS[name]

    def getAvailableDeviceTypes(self, name: str) -> tuple[int, ...]:  # noqa: N802
        return tuple(self._TYPES[n] for n in self._ADAPTERS[name])

    def getAvailableDeviceDescriptions(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return tuple("N/A" for _ in self._ADAPTERS[name])

    def loadDevice(self, label: str, adapter: str, device: str) -> None:  # noqa: N802
        self.loaded.append(label)

    def setParentLabel(self, label: str, parent: str) -> None:  # noqa: N802
        self.parents[label] = parent

    def hasProperty(self, label: str, prop: str) -> bool:  # noqa: N802
        return False

    def initializeDevice(self, label: str) -> None:  # noqa: N802
        if label == self.dies_on:
            raise SystemExit(9)
        self.initialised.add(label)

    def getInstalledDevices(self, hub: str) -> tuple[str, ...]:  # noqa: N802
        return ()

    def getDeviceType(self, label: str) -> int:  # noqa: N802
        return self._TYPES[label]

    def getParentLabel(self, label: str) -> str:  # noqa: N802
        if self.parent_error and label == "Cam":
            raise RuntimeError("no parent for you")
        return self.parents.get(label, "")

    def getStateLabels(self, label: str) -> tuple[str, ...]:  # noqa: N802
        if self.labels_error:
            raise RuntimeError("Cannot get position label at index 0")
        return ("Open", "Closed")


def _run_build_in_process(
    core: _StubCore,
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seen_at_release: list[ChildResult | None],
) -> ChildResult:
    def release(released: object) -> None:
        # What the file holds at the moment the devices are unloaded.
        try:
            seen_at_release.append(
                ChildResult.model_validate_json(path.read_text(encoding="utf-8"))
            )
        except OSError:
            seen_at_release.append(None)
        core.released = True

    monkeypatch.setattr(_mm_child, "close_core", release)
    result = ChildResult()
    _mm_child._build(core, result, path, "Fake", None, ())  # type: ignore[arg-type]
    return ChildResult.model_validate_json(path.read_text(encoding="utf-8"))


def _build_that_dies(
    core: _StubCore,
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    camera: str | None = None,
    stand: str = "Fake",
    skip: Sequence[str] = (),
) -> ChildResult:
    """Run the build until ``core`` ends the process; read what the file holds."""
    released: list[bool] = []
    monkeypatch.setattr(_mm_child, "close_core", lambda c: released.append(True))
    with pytest.raises(SystemExit):
        _mm_child._build(core, ChildResult(), path, stand, camera, skip)  # type: ignore[arg-type]
    # The `finally` still releases the stand: nothing holds it after a death.
    assert released == [True]
    return ChildResult.model_validate_json(path.read_text(encoding="utf-8"))


def test_a_child_that_dies_in_the_stands_walk_leaves_the_rows_it_had_finished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left = _build_that_dies(
        _StubCore(dies_on="Shut"), tmp_path / "result.json", monkeypatch
    )

    # The shutter killed it: the hub and the camera are on file, the scan is not
    # whole, and nothing in it looks like a verdict on the stand.
    scan = left.build
    assert scan is not None
    assert not scan.complete
    assert [d.name for d in scan.stand.devices] == ["H", "Cam"]
    assert not scan.stand.complete
    assert scan.camera is None
    assert scan.devices == []


def test_a_child_that_dies_in_the_cameras_walk_keeps_the_whole_stand_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left = _build_that_dies(
        _StubCore(dies_on="CCam"),
        tmp_path / "result.json",
        monkeypatch,
        camera="Cam",
    )

    scan = left.build
    assert scan is not None
    assert not scan.complete
    # The stand was finished before the camera adapter was touched.
    assert scan.stand.complete
    assert [d.name for d in scan.stand.devices] == ["H", "Cam", "Shut", "Wheel"]
    assert scan.camera is not None
    assert not scan.camera.complete
    assert [d.name for d in scan.camera.devices] == ["CH"]


def test_the_camera_walk_leaves_a_device_out_when_told_and_never_loads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core = _StubCore()
    monkeypatch.setattr(_mm_child, "close_core", lambda c: None)
    path = tmp_path / "result.json"

    _mm_child._build(core, ChildResult(), path, "Fake", "Cam", ["CCam"])  # type: ignore[arg-type]

    scan = ChildResult.model_validate_json(path.read_text(encoding="utf-8")).build
    assert scan is not None
    assert scan.camera is not None
    (row,) = [d for d in scan.camera.devices if d.name == "CCam"]
    assert (row.status, row.reason) == ("not-probed", "left out: --skip CCam")
    assert "CCam" not in core.loaded
    # The camera walk reports each step too: a crash there names its device.
    err = capsys.readouterr().err.splitlines()
    assert "listing Cam" in err
    assert "probing Cam/CH" in err
    assert "reading Cam/CH" in err


def test_both_walks_run_when_the_stand_cannot_be_walked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The camera is diagnosed in the same visit as a stand that is not there.
    monkeypatch.setattr(_mm_child, "close_core", lambda c: None)
    path = tmp_path / "result.json"

    _mm_child._build(_StubCore(), ChildResult(), path, "NoSuchAdapter", "Cam", ())  # type: ignore[arg-type]

    scan = ChildResult.model_validate_json(path.read_text(encoding="utf-8")).build
    assert scan is not None
    assert scan.complete
    assert scan.stand.error == "adapter not installed"
    assert scan.camera is not None
    assert [d.status for d in scan.camera.devices] == ["ok", "ok"]
    assert [d.label for d in scan.devices] == ["CH", "CCam"]


def test_the_whole_scan_is_written_before_the_core_is_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    core = _StubCore()
    seen: list[ChildResult | None] = []

    final = _run_build_in_process(core, tmp_path / "result.json", monkeypatch, seen)

    assert core.released
    # Unloading a vendor's devices can crash (FM-02): the file must be whole
    # by then, not rewritten after it.
    (at_release,) = seen
    assert at_release is not None
    assert at_release.build is not None
    assert at_release.build.complete
    assert at_release.build == final.build
    assert [d.label for d in at_release.build.devices] == ["H", "Cam", "Shut", "Wheel"]


def test_a_getter_that_raises_leaves_its_field_empty_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    core = _StubCore(parent_error=True, labels_error=True)

    final = _run_build_in_process(core, tmp_path / "result.json", monkeypatch, [])

    scan = final.build
    assert scan is not None
    assert scan.complete
    by_label = {d.label: d for d in scan.devices}
    # The device itself initialised: it stays, with less known about it.
    assert by_label["Cam"].parent is None
    assert by_label["Wheel"].state_labels == []
    assert by_label["Shut"].parent == "H"
    assert "Cam: its parent could not be read: no parent for you" in scan.notes
    assert (
        "Wheel: its state labels could not be read: "
        "Cannot get position label at index 0" in scan.notes
    )


def test_the_labels_of_a_state_device_are_read_in_position_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = _run_build_in_process(
        _StubCore(), tmp_path / "result.json", monkeypatch, []
    )

    assert final.build is not None
    by_label = {d.label: d for d in final.build.devices}
    assert by_label["Wheel"].state_labels == ["Open", "Closed"]
    assert by_label["Cam"].state_labels == []
    assert by_label["H"].parent is None


def test_the_child_says_which_step_it_is_on_in_the_order_the_parent_reads_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_build_in_process(_StubCore(), tmp_path / "result.json", monkeypatch, [])

    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "listing Fake"
    assert lines.index("probing Fake/H") < lines.index("probing Fake/Cam")
    # Every probing line comes before the first reading line.
    last_probing = max(i for i, line in enumerate(lines) if line.startswith("probing "))
    first_reading = min(
        i for i, line in enumerate(lines) if line.startswith("reading ")
    )
    assert last_probing < first_reading
    assert "reading Fake/Wheel" in lines


def test_a_scan_that_is_cut_short_after_the_walks_has_both_walks_and_is_not_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The child dies reading a device: the file holds the walks, not "complete".
    class Dies(_StubCore):
        def getParentLabel(self, label: str) -> str:  # noqa: N802
            raise SystemExit(7)

    path = tmp_path / "result.json"
    monkeypatch.setattr(_mm_child, "close_core", lambda core: None)

    with pytest.raises(SystemExit):
        _mm_child._build(Dies(), ChildResult(), path, "Fake", None, ())  # type: ignore[arg-type]

    left = ChildResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert left.build is not None
    assert not left.build.complete
    assert left.build.stand.devices[0].name == "H"
    assert left.build.stand.complete
    assert left.build.devices == []


# --- the command -----------------------------------------------------------------------

runner = CliRunner(env={"COLUMNS": "200"})


def _build(*args: str) -> Result:
    return runner.invoke(app, ["config", "build", *args])


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    """Whitespace-normalised, without colour: the report's lines wrap at the console's
    width (FM-37), and typer colours its own errors when CI forces colour (FM-72)."""
    return " ".join(_ANSI.sub("", text).split())


def _core_of(path: Path) -> CMMCorePlus:
    """The file as a fresh core loads it: the proof that it is a configuration."""
    return core_mod.open_core(path)


def _slots(core: CMMCorePlus) -> dict[str, str]:
    return {
        "Camera": core.getCameraDevice(),
        "XYStage": core.getXYStageDevice(),
        "Focus": core.getFocusDevice(),
        "AutoFocus": core.getAutoFocusDevice(),
        "Shutter": core.getShutterDevice(),
    }


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("utf-8", errors="surrogateescape").split("\n")


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_a_build_from_the_demo_hub_loads_back_with_its_devices_and_roles(
    tmp_path: Path,
) -> None:
    out = tmp_path / "demo.cfg"

    result = _build("--profile", "demo", "--adapter", "DemoCamera", "--out", str(out))

    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    assert f"✓ wrote {out}" in text
    assert f"check it with: smc doctor --config {out}" in text
    # MMCore picks the only camera, XY stage, autofocus and shutter by itself, so
    # a core that answers `DCam` does not show that the file said so.
    lines = _lines(out)
    for slot, label in (
        ("Camera", "DCam"),
        ("Shutter", "DShutter"),
        ("Focus", "DStage"),
        ("XYStage", "DXYStage"),
        ("AutoFocus", "DAutoFocus"),
    ):
        assert f"Property,Core,{slot},{label}" in lines
    assert "Property,Core,AutoShutter,1" in lines
    core = _core_of(out)
    try:
        # DHub and its 19 peripherals, and the Core itself.
        assert len(core.getLoadedDevices()) == 21
        assert _slots(core) == {
            "Camera": "DCam",
            "XYStage": "DXYStage",
            "Focus": "DStage",
            "AutoFocus": "DAutoFocus",
            "Shutter": "DShutter",
        }
        labels = [str(label) for label in core.getStateLabels("DObjective")]
        assert labels[0] == "Objective-A"
        assert len(labels) == 6
        assert core.getParentLabel("DCam") == "DHub"
    finally:
        core_mod.close_core(core)


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_the_file_is_written_with_lf_line_ends(tmp_path: Path) -> None:
    out = tmp_path / "demo.cfg"

    result = _build("--adapter", "DemoCamera", "--out", str(out))

    assert result.exit_code == 0, result.output
    data = out.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert data.startswith(b"# Generated by smc config build on ")
    # The CLI does not mind which OS wrote it: UTF-8, no byte-order mark.
    assert not data.startswith(b"\xef\xbb\xbf")


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_a_rebuild_carries_presets_and_pixel_sizes_and_writes_a_bak(
    tmp_path: Path,
) -> None:
    out = tmp_path / "demo.cfg"
    assert _build("--adapter", "DemoCamera", "--out", str(out)).exit_code == 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        f.write(
            "\n# added by hand\n"
            "ConfigGroup,Channel,DAPI,DWheel,Label,State-1\n"
            "ConfigGroup,Channel,DAPI,DCam,Exposure,10\n"
            "ConfigPixelSize,Res10x,DObjective,Label,Objective-B\n"
            "PixelSize_um,Res10x,0.65\n"
        )
    before = out.read_bytes()

    result = _build("--adapter", "DemoCamera", "--out", str(out), "--force")

    assert result.exit_code == 0, result.output
    bak = tmp_path / "demo.cfg.bak"
    assert bak.read_bytes() == before
    text = _flat(result.output)
    assert f"previous saved as {bak}" in text
    assert "carried over from" in text
    assert "1 preset" in text
    assert "1 pixel size" in text
    core = _core_of(out)
    try:
        assert "DAPI" in core.getAvailableConfigs("Channel")
        assert "Res10x" in core.getAvailablePixelSizeConfigs()
        assert core.getPixelSizeUmByID("Res10x") == pytest.approx(0.65)
    finally:
        core_mod.close_core(core)


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_a_forced_build_with_no_file_to_replace_writes_no_bak(tmp_path: Path) -> None:
    out = tmp_path / "demo.cfg"

    result = _build("--adapter", "DemoCamera", "--out", str(out), "--force")

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert not (tmp_path / "demo.cfg.bak").exists()
    assert "previous saved" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_the_demos_own_presets_are_set_aside_since_its_labels_differ(
    tmp_path: Path,
) -> None:
    install = core_mod.find_install()
    assert install is not None
    own = install / "MMConfig_demo.cfg"
    if not own.is_file():
        pytest.skip("this install has no MMConfig_demo.cfg")
    out = tmp_path / "demo.cfg"
    shutil.copyfile(own, out)

    first = _build("--adapter", "DemoCamera", "--out", str(out), "--force")

    assert first.exit_code == 0, first.output
    assert "! set aside:" in first.output
    lines = _lines(out)
    aside = [line for line in lines if line.startswith("# not carried: ")]
    assert aside
    # The file's labels are `Camera`, `XY`, `Objective`: none of its presets
    # may be partly active in a file that has `DCam`, `DXYStage`, `DObjective`.
    active = [line for line in lines if line and not line.startswith("#")]
    assert not [line for line in active if line.startswith("ConfigGroup")]
    assert not [line for line in active if line.startswith("ConfigPixelSize")]
    core = _core_of(out)
    try:
        assert len(core.getLoadedDevices()) == 21
        assert list(core.getAvailableConfigGroups()) == []
    finally:
        core_mod.close_core(core)

    # Again: the same lines are set aside, and still not one is lost.
    second = _build("--adapter", "DemoCamera", "--out", str(out), "--force")
    assert second.exit_code == 0, second.output
    assert [line for line in _lines(out) if line.startswith("# not carried: ")] == aside
    original = [
        line.rstrip("\r")
        for line in own.read_text(encoding="utf-8").split("\n")
        if line.startswith(("ConfigGroup", "ConfigPixelSize", "PixelSize_um", "Label,"))
    ]
    assert original
    for line in original:
        assert "# not carried: " + line in aside


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_a_skipped_device_is_left_out_of_the_file_and_said_so(tmp_path: Path) -> None:
    out = tmp_path / "demo.cfg"

    result = _build(
        "--adapter",
        "DemoCamera",
        "--skip",
        "DWheel",
        "--skip",
        "Nope",
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.output
    lines = _lines(out)
    assert "#   DemoCamera/DWheel: left out: --skip DWheel" in lines
    assert not [line for line in lines if line.startswith("Device,DWheel")]
    text = _flat(result.output)
    assert "- --skip Nope matched no device of DemoCamera" in text
    core = _core_of(out)
    try:
        assert len(core.getLoadedDevices()) == 20
    finally:
        core_mod.close_core(core)


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_a_camera_adapter_adds_its_cameras_and_their_hub_only(tmp_path: Path) -> None:
    out = tmp_path / "both.cfg"

    result = _build(
        "--adapter",
        "DemoCamera",
        "--skip",
        "DCam",
        "--camera-adapter",
        "SequenceTester",
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.output
    lines = _lines(out)
    assert "Device,THub,SequenceTester,THub" in lines
    assert "Device,TCamera-0,SequenceTester,TCamera-0" in lines
    assert "Parent,TCamera-0,THub" in lines
    # The other twelve devices of SequenceTester are not the cameras'.
    assert not [line for line in lines if line.startswith("Device,TXYStage-0")]
    assert not [line for line in lines if line.startswith("Device,TShutter-0")]
    assert (
        "#   SequenceTester/TXYStage-0: not a camera: --camera-adapter adds cameras only"
        in lines
    )
    assert "Property,Core,Camera,TCamera-0" in lines
    assert "Building from DemoCamera and SequenceTester" in _flat(result.output)
    core = _core_of(out)
    try:
        assert core.getCameraDevice() == "TCamera-0"
        assert core.getParentLabel("TCamera-0") == "THub"
        assert "TXYStage-0" not in core.getLoadedDevices()
    finally:
        core_mod.close_core(core)


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_dry_run_prints_the_configuration_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "demo.cfg"

    result = _build("--adapter", "DemoCamera", "--out", str(out), "--dry-run")

    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
    lines = result.output.splitlines()
    marker = f"--- would write {out} ---"
    assert marker in lines
    printed = lines[lines.index(marker) + 1 :]
    assert "Device,DCam,DemoCamera,DCam" in printed
    assert "Property,Core,Camera,DCam" in printed
    assert "✓ wrote" not in result.output


@pytest.mark.demo
@pytest.mark.usefixtures("need_mm")
def test_dry_run_reads_an_existing_file_to_show_what_is_carried(tmp_path: Path) -> None:
    out = tmp_path / "demo.cfg"
    out.write_text(
        "ConfigGroup,Channel,DAPI,DWheel,State,1\n"
        "ConfigGroup,Channel,Gone,Missing,State,1\n",
        encoding="utf-8",
    )
    before = out.read_bytes()

    # No --force needed to read it, and nothing is written, not even a .bak.
    result = _build("--adapter", "DemoCamera", "--out", str(out), "--dry-run")

    assert result.exit_code == 0, result.output
    assert out.read_bytes() == before
    assert not (tmp_path / "demo.cfg.bak").exists()
    text = _flat(result.output)
    assert "carried over from" in text
    assert (
        "! set aside: Channel/Gone: names Missing, which this configuration does not load"
        in text
    )
    assert "ConfigGroup,Channel,DAPI,DWheel,State,1" in result.output.splitlines()


# --- refusals before anything is initialised ---------------------------------------------


def _sentinel_child(monkeypatch: pytest.MonkeyPatch, sentinel: Path) -> None:
    """A child that proves it was started."""
    code = f"open({str(sentinel)!r}, 'w', encoding='utf-8').write('started')"
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", code])


def test_an_existing_out_without_force_is_refused_before_any_child_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)
    out = tmp_path / "x.cfg"
    out.write_bytes(b"ConfigGroup,Channel,DAPI,DWheel,State,1\r\n")

    result = _build("--adapter", "DemoCamera", "--out", str(out))

    assert result.exit_code == 2, result.output
    text = _flat(result.output)
    assert f"✗ {out} exists: pass --force to replace it" in text
    assert "it is saved as" in text
    assert str(tmp_path / "x.cfg.bak") in text
    assert not sentinel.exists()
    assert out.read_bytes() == b"ConfigGroup,Channel,DAPI,DWheel,State,1\r\n"
    assert not (tmp_path / "x.cfg.bak").exists()


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda tmp: tmp, "is a folder"),
        (lambda tmp: tmp / "no-such-folder" / "x.cfg", "does not exist"),
    ],
    ids=["a-folder", "a-folder-that-does-not-exist"],
)
def test_an_out_that_cannot_be_written_is_refused_before_any_child_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make: Callable[[Path], Path],
    message: str,
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)

    result = _build("--adapter", "DemoCamera", "--out", str(make(tmp_path)))

    assert result.exit_code == 2, result.output
    assert message in _flat(result.output)
    assert not sentinel.exists()


def test_an_out_that_exists_and_cannot_be_read_stops_before_any_child_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)
    out = tmp_path / "x.cfg"
    out.write_text("Device,A,B,C\n", encoding="utf-8")

    def refuse(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("smc.discovery.config_build.read_previous", refuse)

    result = _build("--adapter", "DemoCamera", "--out", str(out), "--force")

    assert result.exit_code == 1, result.output
    assert f"✗ cannot read {out}" in _flat(result.output)
    assert not sentinel.exists()


def test_the_adapter_is_required_when_the_profile_names_no_stand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)

    result = _build("--profile", "demo", "--out", str(tmp_path / "x.cfg"))

    assert result.exit_code == 2, result.output
    text = _flat(result.output)
    assert "✗ name the stand's adapter with --adapter" in text
    assert "demo" in text
    assert not sentinel.exists()


def test_a_camera_adapter_equal_to_the_adapter_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)

    result = _build(
        "--adapter",
        "DemoCamera",
        "--camera-adapter",
        "DemoCamera",
        "--out",
        str(tmp_path / "x.cfg"),
    )

    assert result.exit_code == 2, result.output
    assert "--camera-adapter is the stand's adapter (DemoCamera)" in _flat(
        result.output
    )
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--skip", "Dia\nLamp"],
            "--skip: the device name holds an invisible or control",
        ),
        (["--skip", " "], "--skip: the device name is empty"),
        (["--adapter", " "], "--adapter: the adapter name is empty"),
        (["--camera-adapter", "Cam\x00"], "--camera-adapter: the adapter name holds"),
        (["--timeout-s", "nan"], "--timeout-s must be a number of seconds"),
    ],
    ids=[
        "skip-control",
        "skip-empty",
        "adapter-empty",
        "camera-control",
        "timeout-nan",
    ],
)
def test_a_value_the_checks_refuse_is_a_usage_error_before_any_child_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str], message: str
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)
    base = ["--adapter", "DemoCamera", "--out", str(tmp_path / "x.cfg")]

    result = _build(*base, *args)

    assert result.exit_code == 2, result.output
    assert message in _flat(result.output)
    assert not sentinel.exists()


@pytest.mark.parametrize("value", ["0", "3601", "-5"])
def test_a_timeout_out_of_range_is_a_usage_error(tmp_path: Path, value: str) -> None:
    result = _build(
        "--adapter",
        "DemoCamera",
        "--timeout-s",
        value,
        "--out",
        str(tmp_path / "x.cfg"),
    )

    assert result.exit_code == 2, result.output


def test_the_default_timeout_is_the_builds() -> None:
    from smc.cli import _BUILD_TIMEOUT_DEFAULT_S

    assert _BUILD_TIMEOUT_DEFAULT_S == mm_inventory.BUILD_TIMEOUT_S


def test_out_is_required() -> None:
    result = _build("--adapter", "DemoCamera")

    assert result.exit_code == 2
    assert "--out" in _flat(result.output)


# --- what the stand's profile decides ----------------------------------------------------


def _profile(
    tmp_path: Path,
    config_lines: str,
    *,
    search: Path | None = None,
    roles: str = '[roles.assign]\nshutter = "Shut"\n',
) -> Path:
    """A profile file beside its configuration, which is only read for its stands."""
    (tmp_path / "scope.cfg").write_text(config_lines, encoding="utf-8")
    folders = f'["{search.as_posix()}"]' if search is not None else "[]"
    path = tmp_path / "scope.toml"
    path.write_text(
        '[microscope]\nname = "scope"\nvendor = "Test"\n\n'
        f'[micromanager]\nconfig = "scope.cfg"\nadapter_search_paths = {folders}\n\n'
        + roles,
        encoding="utf-8",
    )
    return path


def test_the_adapter_defaults_to_the_one_stand_the_profile_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_path = tmp_path / "argv.json"
    _argv_recording_child(monkeypatch, argv_path)
    folder = tmp_path / "adapters"
    folder.mkdir()
    profile = _profile(tmp_path, "Device,Ti2-E__0,NikonTi2,Ti2-E__0\n", search=folder)

    result = _build("--profile", str(profile), "--out", str(tmp_path / "x.cfg"))

    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    assert "--build=NikonTi2" in argv
    # The folders the profile's stand is opened with are the ones the build searches.
    assert f"--search-path={folder}" in argv
    assert "Building from NikonTi2:" in _flat(result.output)


@pytest.mark.parametrize(
    ("config", "why"),
    [
        ("Device,Cam,DemoCamera,DCam\n", "loads no Nikon stand"),
        (
            "Device,A,NikonTi2,A\nDevice,B,NikonTI,B\n",
            "loads 2 Nikon stands: NikonTi2, NikonTI",
        ),
    ],
    ids=["no-stand", "two-stands"],
)
def test_the_adapter_is_not_guessed_among_none_or_several_stands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: str, why: str
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)
    profile = _profile(tmp_path, config)

    result = _build("--profile", str(profile), "--out", str(tmp_path / "x.cfg"))

    assert result.exit_code == 2, result.output
    assert why in _flat(result.output)
    assert not sentinel.exists()


def test_a_profile_that_does_not_load_is_one_line_and_exit_1(tmp_path: Path) -> None:
    result = _build(
        "--profile",
        str(tmp_path / "missing.toml"),
        "--adapter",
        "DemoCamera",
        "--out",
        str(tmp_path / "x.cfg"),
    )

    assert result.exit_code == 1, result.output
    assert "✗" in result.output
    assert "missing.toml" in _flat(result.output)


def test_the_profiles_role_assignments_reach_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices = [
        _built("H", "Hub", parent=None),
        _built("Cam", "Camera"),
        _built("Shut", "Shutter"),
        _built("Dia", "Shutter"),
    ]
    scan = BuildScan(
        stand=ProbeResult(
            adapter="NikonTi2",
            devices=[
                ProbedDevice(name=d.label, type=d.type, status="ok") for d in devices
            ],
        ),
        devices=devices,
    )
    _fake_child(monkeypatch, tmp_path, scan)
    profile = _profile(tmp_path, "Device,Ti2-E__0,NikonTi2,Ti2-E__0\n")
    out = tmp_path / "x.cfg"

    result = _build("--profile", str(profile), "--out", str(out))

    assert result.exit_code == 0, result.output
    lines = _lines(out)
    # [roles.assign] shutter = "Shut" beats the heuristic, which prefers "Dia".
    assert "Property,Core,Shutter,Shut" in lines
    assert "Profile scope" in "\n".join(lines[:3])


def _built(
    label: str, type_name: str, *, parent: str | None = "H", library: str = "NikonTi2"
) -> BuiltDevice:
    return BuiltDevice(
        label=label, library=library, name=label, type=type_name, parent=parent
    )


# --- a build that does not produce a file ------------------------------------------------


def test_a_child_that_dies_mid_build_writes_nothing_and_names_the_device_to_skip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch,
        tmp_path,
        _partial(),
        stderr="listing DemoCamera\nprobing DemoCamera/DWheel\n",
        exit_code=3,
    )
    out = tmp_path / "x.cfg"

    result = _build("--adapter", "DemoCamera", "--out", str(out))

    assert result.exit_code == 1, result.output
    assert not out.exists()
    assert list(tmp_path.glob("x.cfg*")) == []
    text = _flat(result.output)
    assert "INCOMPLETE" in text
    assert "build: the build child exited 3" in text
    assert "✗ the build child stopped before the end" in text
    assert "re-run with --skip DWheel to leave it out" in text


def test_a_child_that_left_no_result_says_so_and_names_the_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch, tmp_path, None, stderr="probing DemoCamera/DWheel\n", exit_code=3
    )
    out = tmp_path / "x.cfg"

    result = _build("--adapter", "DemoCamera", "--out", str(out))

    assert result.exit_code == 1, result.output
    assert not out.exists()
    text = _flat(result.output)
    assert "✗ the build child left no result" in text
    assert "re-run with --skip DWheel to leave it out" in text


def test_a_hub_that_will_not_initialise_prints_the_causes_in_order_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = ProbedDevice(
        name="Ti2-E__0",
        type="Hub",
        status="failed",
        error="(Error message unavailable) (-10)",
    )
    scan = BuildScan(
        stand=ProbeResult(
            adapter="NikonTi2",
            devices=[hub],
            causes=diagnosis.hub_failure_causes("NikonTi2"),
        )
    )
    _fake_child(monkeypatch, tmp_path, scan)
    out = tmp_path / "x.cfg"

    result = _build("--adapter", "NikonTi2", "--out", str(out))

    assert result.exit_code == 1, result.output
    assert not out.exists()
    text = _flat(result.output)
    assert "✗ hub Ti2-E__0 of NikonTi2 did not initialise" in text
    # Another program first, then power, then the driver: by their first words.
    causes = [nikon.ONE_CONNECTION, *nikon.TI2.causes]
    heads = [_flat(cause)[:30] for cause in causes]
    positions = [text.index(head) for head in heads]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)
    assert "In order of likelihood:" in text
    assert "1. another program holds the stand" in text
    assert text.index("In order of likelihood:") < positions[0]


def test_a_refusal_does_not_touch_the_file_it_would_have_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(monkeypatch, tmp_path, _partial(), exit_code=3)
    out = tmp_path / "x.cfg"
    out.write_bytes(b"ConfigGroup,Channel,DAPI,DWheel,State,1\n")

    result = _build("--adapter", "DemoCamera", "--out", str(out), "--force")

    assert result.exit_code == 1, result.output
    assert out.read_bytes() == b"ConfigGroup,Channel,DAPI,DWheel,State,1\n"
    assert not (tmp_path / "x.cfg.bak").exists()


def test_a_failed_write_prints_the_report_all_the_same_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices = [_built("H", "Hub", parent=None), _built("Cam", "Camera")]
    scan = BuildScan(
        stand=ProbeResult(
            adapter="Fake",
            devices=[
                ProbedDevice(name=d.label, type=d.type, status="ok") for d in devices
            ],
        ),
        devices=devices,
    )
    _fake_child(monkeypatch, tmp_path, scan)

    def disk_full(out: Path, write: Callable[[Path], object]) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("smc.cli._replace", disk_full)
    out = tmp_path / "x.cfg"

    result = _build("--adapter", "Fake", "--out", str(out))

    assert result.exit_code == 1, result.output
    text = _flat(result.output)
    # The build took minutes: the report is what is left of it.
    assert "Probe of Fake" in text
    assert "Roles" in text
    assert f"✗ could not write {out}: No space left on device" in text
    assert "✓ wrote" not in text


def test_ctrl_c_while_building_exits_130_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(mm_inventory, "build_section", interrupted)
    out = tmp_path / "x.cfg"

    result = _build("--adapter", "DemoCamera", "--out", str(out))

    assert result.exit_code == 130, result.output
    assert "✗ interrupted" in _flat(result.output)
    assert not out.exists()


def test_a_message_with_markup_in_it_is_printed_as_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-36: a device name, an error or a refusal carries brackets; rich would
    # drop "[red]" and "[/]" as markup, or fail on a closing tag with no opener.
    hub = ProbedDevice(
        name="[bold]Hub[/]", type="Hub", status="failed", error="[/] not markup [red]x"
    )
    scan = BuildScan(stand=ProbeResult(adapter="[x]", devices=[hub]))
    _fake_child(monkeypatch, tmp_path, scan)

    result = _build("--adapter", "[x]", "--out", str(tmp_path / "x.cfg"))

    assert result.exit_code == 1, result.output
    text = _flat(result.output)
    assert "✗ hub [bold]Hub[/] of [x] did not initialise" in text
    assert "[/] not markup [red]x" in text
    assert "Building from [x]:" in text


def test_a_dry_run_prints_a_carried_line_that_is_not_utf8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The file keeps the byte; a UTF-8 console cannot print it, and must not die.
    devices = [
        _built("H", "Hub", parent=None, library="Fake"),
        _built("Wheel", "State", library="Fake"),
    ]
    scan = BuildScan(
        stand=ProbeResult(
            adapter="Fake",
            devices=[
                ProbedDevice(name=d.label, type=d.type, status="ok") for d in devices
            ],
        ),
        devices=devices,
    )
    _fake_child(monkeypatch, tmp_path, scan)
    out = tmp_path / "x.cfg"
    out.write_bytes(b"ConfigGroup,Channel,\xb5m,Wheel,State,1\n")

    result = _build("--adapter", "Fake", "--out", str(out), "--dry-run")

    assert result.exit_code == 0, result.output
    assert "ConfigGroup,Channel," in result.output
    assert out.read_bytes() == b"ConfigGroup,Channel,\xb5m,Wheel,State,1\n"


# --- what the file keeps, and when it is written -------------------------------------------


def _whole_scan(adapter: str, devices: Sequence[BuiltDevice]) -> BuildScan:
    return BuildScan(
        stand=ProbeResult(
            adapter=adapter,
            devices=[
                ProbedDevice(name=d.label, type=d.type, status="ok") for d in devices
            ],
        ),
        devices=list(devices),
    )


def _stand_with_a_wheel(adapter: str = "Fake") -> BuildScan:
    return _whole_scan(
        adapter,
        [
            _built("H", "Hub", parent=None, library=adapter),
            _built("Cam", "Camera", library=adapter),
            _built("Wheel", "State", library=adapter),
        ],
    )


def test_the_file_and_its_bak_keep_the_bytes_that_are_not_utf8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The write of the command itself, not a copy of it: a cp1252 `µ` is one byte
    # in the file it came from, and must be that byte in both files (FM-80).
    _fake_child(monkeypatch, tmp_path, _stand_with_a_wheel())
    out = tmp_path / "x.cfg"
    previous = b"ConfigGroup,Channel,\xb5m,Wheel,State,1\r\n"
    out.write_bytes(previous)

    result = _build("--adapter", "Fake", "--out", str(out), "--force")

    assert result.exit_code == 0, result.output
    assert (tmp_path / "x.cfg.bak").read_bytes() == previous
    data = out.read_bytes()
    assert b"ConfigGroup,Channel,\xb5m,Wheel,State,1\n" in data
    assert b"\r" not in data


def test_the_file_is_written_before_the_report_is_printed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-30: a broken pipe or a Ctrl-C while the report is printed must not lose
    # the file. The report's first line asks whether the file is there yet.
    from smc.discovery import report as report_mod

    _fake_child(monkeypatch, tmp_path, _stand_with_a_wheel())
    out = tmp_path / "x.cfg"
    on_disk_when_printed: list[bool] = []
    real = report_mod.probe_renderables

    def watching(probe: ProbeResult) -> list[object]:
        on_disk_when_printed.append(out.exists())
        return real(probe)  # type: ignore[return-value]

    monkeypatch.setattr(report_mod, "probe_renderables", watching)

    result = _build("--adapter", "Fake", "--out", str(out))

    assert result.exit_code == 0, result.output
    assert on_disk_when_printed == [True]


def test_a_folder_that_cannot_be_written_is_refused_before_any_child_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = tmp_path / "child-was-started"
    _sentinel_child(monkeypatch, sentinel)

    def read_only(*args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "TemporaryFile", read_only)

    result = _build("--adapter", "DemoCamera", "--out", str(tmp_path / "x.cfg"))

    assert result.exit_code == 1, result.output
    assert f"✗ cannot write to {tmp_path}" in _flat(result.output)
    assert not sentinel.exists()
    # A dry run writes nothing, so it does not need the folder to be writable.
    dry = _build(
        "--adapter", "DemoCamera", "--out", str(tmp_path / "x.cfg"), "--dry-run"
    )
    assert "cannot write to" not in dry.output
    assert sentinel.exists()


def test_the_options_reach_the_build_as_they_were_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[dict[str, object]] = []

    def recording(stand: str, **kwargs: object) -> BuildRun:
        seen.append({"stand": stand, **kwargs})
        return BuildRun(scan=None, notes=[], stopped_at=None)

    monkeypatch.setattr(mm_inventory, "build_section", recording)
    folder = tmp_path / "adapters"
    folder.mkdir()
    profile = _profile(tmp_path, "Device,Ti2,NikonTi2,Ti2-E__0\n", search=folder)

    result = _build(
        "--profile",
        str(profile),
        "--camera-adapter",
        "Cam",
        "--skip",
        " Dia ",
        "--skip",
        "Lamp",
        "--timeout-s",
        "7",
        "--out",
        str(tmp_path / "x.cfg"),
    )

    assert result.exit_code == 1, result.output
    assert len(seen) == 1
    (call,) = seen
    assert call["stand"] == "NikonTi2"
    assert call["camera"] == "Cam"
    assert call["skip"] == ["Dia", "Lamp"]
    assert call["timeout_s"] == 7.0
    assert [Path(p).resolve() for p in call["search_paths"]] == [folder.resolve()]  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", ["1", "3600"])
def test_the_ends_of_the_timeout_range_are_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    _fake_child(monkeypatch, tmp_path, None)

    result = _build(
        "--adapter", "Fake", "--timeout-s", value, "--out", str(tmp_path / "x.cfg")
    )

    # It ran and found no result: not a usage error.
    assert result.exit_code == 1, result.output


def test_the_profiles_role_exclusions_reach_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices = [
        _built("H", "Hub", parent=None),
        _built("Cam", "Camera"),
        _built("Dia", "Shutter"),
        _built("Shut", "Shutter"),
    ]
    _fake_child(monkeypatch, tmp_path, _whole_scan("NikonTi2", devices))
    config = "Device,Ti2-E__0,NikonTi2,Ti2-E__0\n"
    plain = tmp_path / "plain"
    excluding = tmp_path / "excluding"
    plain.mkdir()
    excluding.mkdir()
    plain_profile = _profile(plain, config, roles="")
    excluding_profile = _profile(
        excluding, config, roles='[roles.exclude]\nshutter = ["Dia"]\n'
    )

    first = _build("--profile", str(plain_profile), "--out", str(plain / "x.cfg"))
    second = _build(
        "--profile", str(excluding_profile), "--out", str(excluding / "x.cfg")
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    # Control: without the exclusion the heuristic prefers the lamp's shutter.
    assert "Property,Core,Shutter,Dia" in _lines(plain / "x.cfg")
    assert "Property,Core,Shutter,Shut" in _lines(excluding / "x.cfg")


def test_a_previous_file_with_a_blank_row_is_read_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Saved from a spreadsheet: blank rows come back as commas.
    _fake_child(monkeypatch, tmp_path, _stand_with_a_wheel())
    out = tmp_path / "x.cfg"
    out.write_text(
        "ConfigGroup,Channel,A,Wheel,State,0\r\n,,,,,\r\n,\r\n", encoding="utf-8"
    )

    result = _build("--adapter", "Fake", "--out", str(out), "--dry-run")

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert "ConfigGroup,Channel,A,Wheel,State,0" in result.output.splitlines()


def test_a_scan_whose_walks_are_cut_short_writes_nothing_whatever_it_says_of_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = _stand_with_a_wheel()
    scan.stand.complete = False
    _fake_child(monkeypatch, tmp_path, scan, stderr="probing Fake/Cam\n", exit_code=3)
    out = tmp_path / "x.cfg"

    result = _build("--adapter", "Fake", "--out", str(out))

    assert result.exit_code == 1, result.output
    assert not out.exists()
    text = _flat(result.output)
    assert "✗ the build child stopped before the end" in text
    assert "re-run with --skip Cam to leave it out" in text


# --- the .bak that cannot be made --------------------------------------------------------


def _replace_that_fails_for(
    monkeypatch: pytest.MonkeyPatch, *suffixes: str
) -> list[str]:
    """``os.replace`` as a Windows PC with antivirus on the file, for some targets."""
    real = os.replace
    refused: list[str] = []

    def replace(src: object, dst: object) -> None:
        if str(dst).endswith(suffixes):
            refused.append(str(dst))
            raise PermissionError(13, "Permission denied (held by the antivirus)")
        real(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", replace)
    return refused


def test_when_the_bak_cannot_be_made_the_new_configuration_is_kept_beside_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(monkeypatch, tmp_path, _stand_with_a_wheel())
    out = tmp_path / "x.cfg"
    previous = b"ConfigGroup,Channel,A,Wheel,State,0\n"
    out.write_bytes(previous)
    refused = _replace_that_fails_for(monkeypatch, ".bak")

    result = _build("--adapter", "Fake", "--out", str(out), "--force")

    assert result.exit_code == 1, result.output
    assert refused == [str(tmp_path / "x.cfg.bak")]
    text = _flat(result.output)
    kept = tmp_path / "x.cfg.new"
    assert f"could not save the previous file as {tmp_path / 'x.cfg.bak'}" in text
    assert f"{out} was not replaced" in text
    assert f"the new configuration is kept as {kept}" in text
    # The previous file is as it was; the result of the build is on disk.
    assert out.read_bytes() == previous
    assert "Device,Cam,Fake,Cam" in _lines(kept)
    assert "ConfigGroup,Channel,A,Wheel,State,0" in _lines(kept)
    # Nothing else is left behind: no partial file that holds the old bytes.
    assert list(tmp_path.glob(".*.part")) == []
    assert not (tmp_path / "x.cfg.bak").exists()


def test_when_neither_the_bak_nor_the_kept_file_can_be_made_the_message_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(monkeypatch, tmp_path, _stand_with_a_wheel())
    out = tmp_path / "x.cfg"
    out.write_bytes(b"Device,A,B,C\n")
    _replace_that_fails_for(monkeypatch, ".bak", ".new")

    result = _build("--adapter", "Fake", "--out", str(out), "--force")

    assert result.exit_code == 1, result.output
    text = _flat(result.output)
    assert "could not save the previous file" in text
    assert "could not be kept as" in text
    assert out.read_bytes() == b"Device,A,B,C\n"
    # The report is still on screen: it is all that is left.
    assert "Probe of Fake" in text


# --- the hint after a stop -------------------------------------------------------------


def test_a_line_that_names_no_adapter_of_the_build_is_not_a_device_to_leave_out() -> (
    None
):
    # A vendor DLL writes to the same file as the child's progress lines (FM-01).
    stop = mm_inventory._stopped_at
    assert stop("reading C:/Program Files/x.ini", ["DemoCamera"]) is None
    assert stop("probing Other/Dev", ["DemoCamera", "Cam"]) is None
    assert stop("probing Cam/Dev", ["DemoCamera", "Cam"]) == "Dev"
    assert stop("probing DemoCamera/Dev", ["DemoCamera"]) == "Dev"


def test_a_child_that_prints_like_a_progress_line_does_not_name_a_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch,
        tmp_path,
        _partial(),
        stderr="reading C:/vendor/config.ini\n",
        exit_code=3,
    )

    run = build_section("DemoCamera", timeout_s=30)

    assert run.stopped_at is None


def test_a_child_stopped_by_the_deadline_says_the_device_may_only_be_slow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_child(
        monkeypatch,
        tmp_path,
        _partial(),
        stderr="probing DemoCamera/DWheel\n",
        then="time.sleep(30)\n",
    )
    crash = tmp_path / "crash"
    crash.mkdir()

    slow = _build(
        "--adapter", "DemoCamera", "--timeout-s", "1", "--out", str(tmp_path / "a.cfg")
    )

    assert slow.exit_code == 1, slow.output
    text = _flat(slow.output)
    assert "re-run with --skip DWheel to leave it out" in text
    assert "or, if DWheel was only slow, raise --timeout-s (it was 1 s)" in text
    # A child that crashed is not slow: the advice is the other one only.
    _fake_child(
        monkeypatch,
        crash,
        _partial(),
        stderr="probing DemoCamera/DWheel\n",
        exit_code=3,
    )
    died = _build("--adapter", "DemoCamera", "--out", str(tmp_path / "b.cfg"))
    assert "re-run with --skip DWheel to leave it out" in _flat(died.output)
    assert "only slow" not in died.output


def test_the_child_writes_its_progress_in_utf8_whatever_the_console_code_page(
    tmp_path: Path,
) -> None:
    # A redirected Windows stream is cp1252, and would write the micro sign as one
    # byte the parent reads as a replacement character: the --skip hint would
    # name no device. PYTHONIOENCODING stands in for that stream (FM-42).
    stderr_file = tmp_path / "stderr.txt"
    code = (
        "from smc.discovery import _mm_child\n"
        "_mm_child._use_utf8_stderr()\n"
        "_mm_child._progress('reading Fake/Lamp \\N{MICRO SIGN}m')\n"
        "_mm_child._progress('reading Fake/Obj \\N{GREEK SMALL LETTER LAMDA}')\n"
    )
    with stderr_file.open("wb") as err:
        subprocess.run(
            [sys.executable, "-c", code],
            stderr=err,
            env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
            cwd=tmp_path,
            check=True,
            timeout=60,
        )

    assert stderr_file.read_bytes().decode("utf-8").splitlines() == [
        "reading Fake/Lamp \N{MICRO SIGN}m",
        "reading Fake/Obj \N{GREEK SMALL LETTER LAMDA}",
    ]
    last = mm_inventory._last_progress(stderr_file)
    assert mm_inventory._stopped_at(last, ["Fake"]) == (
        "Obj \N{GREEK SMALL LETTER LAMDA}"
    )


def test_the_real_child_writes_utf8_to_its_stderr_file_whatever_the_code_page(
    tmp_path: Path,
) -> None:
    # The whole child, not its helper: `main` must switch the stream. An adapter
    # that is not installed is a result, not a crash; its name is on the first
    # progress line.
    stderr_file = tmp_path / "stderr.txt"
    with stderr_file.open("wb") as err:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "smc.discovery._mm_child",
                "--result",
                str(tmp_path / "result.json"),
                "--build=Ad\N{MICRO SIGN}",
            ],
            stderr=err,
            env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
            cwd=tmp_path,
            check=True,
            timeout=120,
        )

    assert stderr_file.read_bytes() == "listing Ad\N{MICRO SIGN}\n".encode()


# --- mm_section's texts did not change when its runner was shared -----------------------------


def _section_child(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    code = (
        "import os, sys, time\n"
        "RESULT = sys.argv[sys.argv.index('--result') + 1]\n" + body
    )
    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [sys.executable, "-c", code])


def test_mm_section_keeps_the_texts_of_its_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _section_child(
        monkeypatch, "sys.stderr.write('probing Fake/Dev\\n')\nos._exit(3)\n"
    )
    _, _, crashed = mm_inventory.mm_section(["Fake"], timeout_s=30)
    # A failure and no result: the suffix the survey has always had.
    assert crashed == [
        "adapters: the adapter child exited 3 while probing Fake/Dev; "
        "device lists missing, adapter names kept"
    ]

    _section_child(monkeypatch, "os._exit(0)\n")
    _, _, silent = mm_inventory.mm_section(["Fake"], timeout_s=30)
    assert silent == ["adapters: the adapter child wrote no result"]

    _section_child(monkeypatch, "open(RESULT, 'w').write('{no')\n")
    _, _, garbled = mm_inventory.mm_section(["Fake"], timeout_s=30)
    assert len(garbled) == 1
    assert garbled[0].startswith(
        "adapters: the adapter child's result could not be read"
    )

    _section_child(monkeypatch, "time.sleep(30)\n")
    _, _, hung = mm_inventory.mm_section(["Fake"], timeout_s=1)
    assert hung == [
        "adapters: the adapter child did not finish in 1 s and was stopped; "
        "device lists missing, adapter names kept"
    ]

    monkeypatch.setattr(mm_inventory, "CHILD_COMMAND", [str(tmp_path / "missing")])
    _, _, absent = mm_inventory.mm_section(["Fake"], timeout_s=30)
    assert len(absent) == 1
    assert absent[0].startswith("adapters: the adapter child could not start")
