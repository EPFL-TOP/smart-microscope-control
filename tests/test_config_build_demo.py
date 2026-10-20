"""``smc config build``: the build child, its runner and the command (design m2 §10).

The real child runs on Micro-Manager's own test adapters (``DemoCamera`` is a
hub with 19 peripherals, ``SequenceTester`` a hub with 14 and two cameras);
the failures are shown by fake children that write a result and die. Only the
test adapters are named, never a vendor's (FM-40), and every output file is
under ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from pymmcore_plus import DeviceType

from smc.discovery import _mm_child, mm_inventory
from smc.discovery.mm_inventory import ChildResult, build_section
from smc.discovery.models import BuildScan, ProbedDevice, ProbeResult


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
    """Adapter ``Fake``: a hub ``H`` and under it a camera, a shutter and a wheel."""

    def __init__(
        self, *, parent_error: bool = False, labels_error: bool = False
    ) -> None:
        self.parent_error = parent_error
        self.labels_error = labels_error
        self.parents: dict[str, str] = {}
        self.initialised: set[str] = set()
        self.released = False

    _TYPES: ClassVar[dict[str, int]] = {
        "H": int(DeviceType.HubDevice),
        "Cam": int(DeviceType.CameraDevice),
        "Shut": int(DeviceType.ShutterDevice),
        "Wheel": int(DeviceType.StateDevice),
    }
    _LISTED = ("H", "Cam", "Shut", "Wheel")

    def getDeviceAdapterNames(self) -> tuple[str, ...]:  # noqa: N802
        return ("Fake",)

    def getAvailableDevices(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return self._LISTED

    def getAvailableDeviceTypes(self, name: str) -> tuple[int, ...]:  # noqa: N802
        return tuple(self._TYPES[n] for n in self._LISTED)

    def getAvailableDeviceDescriptions(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return tuple("N/A" for _ in self._LISTED)

    def loadDevice(self, label: str, adapter: str, device: str) -> None:  # noqa: N802
        pass

    def setParentLabel(self, label: str, parent: str) -> None:  # noqa: N802
        self.parents[label] = parent

    def hasProperty(self, label: str, prop: str) -> bool:  # noqa: N802
        return False

    def initializeDevice(self, label: str) -> None:  # noqa: N802
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
