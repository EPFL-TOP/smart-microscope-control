"""The scan checks the plate against its registration before its first field (m3 §27.5, #542).

At Ti2-E session 5 (#550, F15) the plate was lifted and put back after the 10x
registration, and a 40x scan placed its frames about 50 µm off in their wells.
The scan now re-measures two registered wells in the first well of every attempt
and refuses a plate that moved, before its first frame.

The stand and the documents are ``plate_check_stand``'s (§20.1's turned stand,
the objectives' offsets in the profile), built once per module through
``run_plugin``. Each test copies ``run.json``, plans a small scan into it with
``scan-plan``, and may then rewrite ``plate-walls``' record turned or slid
(§27.1): the plate on the stage stays where it is. The autofocus is a
``RecordingAutofocus`` whose calls and the stage's moves go into one list, so
that their order is the order the scan made them in.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from plate_check_stand import (
    FORTY,
    TEN,
    Documents,
    build_documents,
    move_registration,
    open_stand,
    profile,
)
from recording_autofocus import RecordingAutofocus
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Limits,
    ObjectiveTurret,
    XYStage,
)
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.plate_registration.plugin import (
    PlateRegistrationParams,
    PlateRegistrationPlugin,
)
from smc.plugins.plate_walls.loop import AUTOFOCUS_LEFT_OFF
from smc.plugins.scan.plugin import ScanParams, ScanPlugin
from smc.plugins.scan_plan.plugin import ScanPlanParams, ScanPlanPlugin
from smc.plugins.slots import ScanResult
from smc.run import RunDocument, StageRecord, StageStatus
from smc.testing.sample_stand import sample_refs

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

#: The end of the refusal: a resume would place new frames against the old pose.
AGAIN = (
    "register it again (plate-walls), then run scan-plan and scan again, without "
    "--resume: frames already taken are placed against the old registration"
)
#: Two fields in A1, the plan's first well; A2 is the second well of a resume.
SMALL = {"rows": 1, "columns": 2}


# ------------------------------------------------------------------ the doubles


class _Stage:
    """The stand's XY stage, recording each move in ``calls``; each method is written out."""

    def __init__(self, inner: XYStage, calls: list[str]) -> None:
        self._inner = inner
        self._calls = calls
        self.moves = 0

    def position_um(self) -> XY:
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        self._calls.append("move_to_um")
        return self._inner.move_to_um(x_um, y_um)

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("the scan makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._inner.limits_um()


class _Camera:
    """The stand's camera, counting the snaps; each method is written out."""

    def __init__(self, inner: Camera) -> None:
        self._inner = inner
        self.snaps = 0

    def snap(self) -> npt.NDArray[np.generic]:
        self.snaps += 1
        return self._inner.snap()

    def exposure_ms(self) -> float:
        return self._inner.exposure_ms()

    def set_exposure_ms(self, value_ms: float) -> float:
        return self._inner.set_exposure_ms(value_ms)

    def image_shape(self) -> tuple[int, int]:
        return self._inner.image_shape()

    def bit_depth(self) -> int:
        return self._inner.bit_depth()

    def pixel_size_um(self) -> float:
        return self._inner.pixel_size_um()


@dataclass
class _Rig:
    stand: Microscope
    stage: _Stage
    camera: _Camera
    calls: list[str]
    run_doc: Path


# ------------------------------------------------------------------ the stands


@pytest.fixture(scope="module")
def documents(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Documents:
    """The 40X's calibration, and the 10X's calibration and registration, through ``run_plugin``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    return build_documents(tmp_path_factory.mktemp("scan-plate-check"))


@pytest.fixture
def stands(mm_available: bool) -> Iterator[list[Microscope]]:
    """The stands a test opened, closed afterwards."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    opened: list[Microscope] = []
    yield opened
    for stand in opened:
        stand.close()


def _rig(
    stands: list[Microscope],
    documents: Documents,
    tmp_path: Path,
    the_profile: Profile | None = None,
    *,
    objective: str = TEN,
    dry_run: bool = False,
    autofocus: RecordingAutofocus | None = None,
    wells: str = "A1",
    **plan: Any,
) -> _Rig:
    """A stand with ``objective`` in place, a copy of ``run.json`` and a scan planned into it."""
    run_doc = tmp_path / "run.json"
    shutil.copy(documents.run, run_doc)
    stand = open_stand(
        the_profile if the_profile is not None else profile(), dry_run=dry_run
    )
    stands.append(stand)
    turret = stand.require(ObjectiveTurret)
    if turret.current() != objective:
        turret.select(objective, confirm=True)
    calls: list[str] = autofocus.calls if autofocus is not None else []
    if autofocus is not None:
        stand.override(Autofocus, autofocus)
    record = run_plugin(
        ScanPlanPlugin(),
        stand,
        ScanPlanParams(wells=wells, **{**SMALL, **plan}),
        run_doc,
    )
    assert record.status is StageStatus.done, record.error
    stage = _Stage(stand.require(XYStage), calls)
    camera = _Camera(stand.require(Camera))
    stand.override(XYStage, stage)
    stand.override(Camera, camera)
    return _Rig(stand, stage, camera, calls, run_doc)


def _scan(
    rig: _Rig,
    *,
    resume: bool = False,
    stop: threading.Event | None = None,
    on_progress: Any = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        ScanPlugin(),
        rig.stand,
        ScanParams(**params),
        rig.run_doc,
        resume=resume,
        stop=stop,
        on_progress=on_progress,
    )


def _result(record: StageRecord) -> ScanResult:
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    return ScanResult.model_validate(record.result)


def _frames_written(rig: _Rig) -> list[Path]:
    return sorted((rig.run_doc.parent / "scan").rglob("*.tif"))


def _refused(rig: _Rig, **params: Any) -> PluginRunError:
    with pytest.raises(PluginRunError) as raised:
        _scan(rig, **params)
    assert raised.value.plugin == "scan"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    stored = RunDocument.load(rig.run_doc).stages["scan"]
    assert stored.status is StageStatus.failed
    assert stored.result is None
    return raised.value


_REFUSED = re.compile(
    r"the plate moved since plate-walls registered it: A1 lies (\d+\.\d) µm from "
    r"its registered centre \([+-]\d+\.\d, [+-]\d+\.\d\) and H12 (\d+\.\d) µm "
    rf"\([+-]\d+\.\d, [+-]\d+\.\d\) under {TEN}, more than 25 µm; it turned about "
    r"([+-]\d\.\d{3})° and slid about \([+-]\d+\.\d, [+-]\d+\.\d\) µm: "
    + re.escape(AGAIN)
)


# ---------------------------------------------------------------- the params


def test_the_scan_reads_the_plate_when_it_can() -> None:
    entry = load("scan")

    assert entry.plugin.reads == ("scan-plan",)
    assert entry.plugin.reads_optional == ("plate",)


def test_the_checks_parameters_are_bounded() -> None:
    assert ScanParams().check_plate is True
    assert ScanParams().max_plate_shift_um == 25.0
    with pytest.raises(ValueError, match="max_plate_shift_um"):
        ScanParams(max_plate_shift_um=0.0)
    with pytest.raises(ValueError, match="max_plate_shift_um"):
        ScanParams(max_plate_shift_um=500.5)


# ---------------------------------------------------------------- it refuses


def test_the_scan_refuses_a_plate_moved_50_um_in_the_record_before_its_first_frame(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(stands, documents, tmp_path)
    move_registration(rig.run_doc, slide_um=(50.0, 0.0))

    refused = _refused(rig)

    match = _REFUSED.fullmatch(refused.reason)
    assert match is not None, refused.reason
    assert float(match[1]) == pytest.approx(50.0, abs=0.2)
    assert float(match[2]) == pytest.approx(50.0, abs=0.2)
    # No frame of the scan: the check's own frames only, two wells of 8.
    assert _frames_written(rig) == []
    assert rig.camera.snaps == 16
    assert RunDocument.load(rig.run_doc).stages["scan"].checkpoint is None


def test_the_scan_refuses_a_plate_turned_0_06_degrees_in_the_record_before_its_first_frame(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(stands, documents, tmp_path)
    move_registration(rig.run_doc, turn_deg=0.06)

    refused = _refused(rig)

    match = _REFUSED.fullmatch(refused.reason)
    assert match is not None, refused.reason
    # 0.06° about the origin moves A1 and H12, 58.7 mm from it, by 61.5 µm.
    assert float(match[1]) == pytest.approx(61.4, abs=0.5)
    assert float(match[2]) == pytest.approx(61.5, abs=0.5)
    assert float(match[3]) == pytest.approx(-0.06, abs=0.002)
    assert _frames_written(rig) == []


def test_the_check_runs_again_on_a_resume(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(stands, documents, tmp_path, wells="A1:A2")
    stop = threading.Event()

    def after_the_first_well(progress: Progress) -> None:
        if progress.done == 2:
            stop.set()

    record = _scan(rig, stop=stop, on_progress=after_the_first_well)
    assert record.status is StageStatus.stopped
    assert len(_frames_written(rig)) == 2
    # The plate is lifted and put back during the stop.
    move_registration(rig.run_doc, slide_um=(50.0, 0.0))
    snaps = rig.camera.snaps

    refused = _refused(rig, resume=True)

    assert _REFUSED.fullmatch(refused.reason), refused.reason
    # The resume took the check's 16 frames and none of the scan's.
    assert rig.camera.snaps == snaps + 16
    assert len(_frames_written(rig)) == 2


# ------------------------------------------------------------------ it passes


def test_a_scan_of_a_plate_that_did_not_move_records_its_check_and_says_so(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(stands, documents, tmp_path)
    seen: list[Progress] = []
    before = datetime.now(timezone.utc)

    result = _result(_scan(rig, on_progress=seen.append))

    after = datetime.now(timezone.utc)
    check = result.plate_check
    assert check is not None
    assert before <= check.checked_utc <= after
    assert [w.well for w in check.wells] == ["A1", "H12"]
    assert check.worst_um < 0.1
    assert (check.plate_objective, check.objective) == (TEN, TEN)
    assert check.shift_um == (0.0, 0.0)
    assert check.frames == 16
    assert len(result.frames) == 2
    assert rig.camera.snaps == 18
    # The check's line first, after the stand's (none here).
    assert result.assumed[0] == (
        f"the plate was checked before the first field: A1 and H12 lie within "
        f"{check.worst_um:.1f} µm of their registered centres under {TEN} (bound "
        f"25 µm), at {check.checked_utc.isoformat(timespec='seconds')}"
    )
    # Its frames are kept beside the scan's.
    assert check.folder is not None
    assert re.fullmatch(r"plate-check/[0-9a-f]{12}", check.folder)
    assert len(list((rig.run_doc.parent / check.folder).glob("*.tif"))) == 8
    # The check comes before the first field, and the scan's progress stays at 0.
    checking = [p for p in seen if p.message.startswith("checking the plate: ")]
    assert len(checking) == 8
    assert {(p.done, p.total) for p in checking} == {(0, 2)}
    assert seen[0].message == "checking the plate: A1 +X"
    assert seen.index(checking[-1]) == 7


def test_a_plate_moved_within_the_bound_passes_with_its_distance(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(stands, documents, tmp_path)
    move_registration(rig.run_doc, slide_um=(50.0, 0.0))

    result = _result(_scan(rig, max_plate_shift_um=60.0))

    assert result.plate_check is not None
    assert result.plate_check.worst_um == pytest.approx(50.0, abs=0.2)
    assert result.plate_check.max_shift_um == 60.0


# ------------------------------------------------------------------ the skips


def _skipped(
    result: ScanResult,
    line: str,
    caplog: pytest.LogCaptureFixture,
    *,
    stand_lines: tuple[str, ...] = (),
) -> None:
    """No check, its line first after the stand's, and one WARNING that says it."""
    assert result.plate_check is None
    assert result.assumed[: len(stand_lines) + 1] == (*stand_lines, line)
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.name == "smc.plugins.scan" and r.levelno == logging.WARNING
    ]
    assert warned.count(line) == 1


def test_check_plate_false_skips_the_check_and_says_so(
    stands: list[Microscope],
    documents: Documents,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = _rig(stands, documents, tmp_path)
    move_registration(rig.run_doc, slide_um=(50.0, 0.0))

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_scan(rig, check_plate=False))

    _skipped(
        result,
        "the plate was not checked against its registration (check_plate=false)",
        caplog,
    )
    assert rig.camera.snaps == 2
    assert not (rig.run_doc.parent / "plate-check").exists()


def test_a_dry_run_scan_does_not_check_the_plate_and_says_so(
    stands: list[Microscope],
    documents: Documents,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = _rig(stands, documents, tmp_path, dry_run=True)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_scan(rig))

    _skipped(
        result, "dry run: the plate was not checked against its registration", caplog
    )
    assert rig.camera.snaps == 0


def test_a_scan_under_an_objective_with_no_measured_offset_skips_the_check_and_says_so(
    stands: list[Microscope],
    documents: Documents,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = _rig(
        stands,
        documents,
        tmp_path,
        profile(forty=False),
        objective=FORTY,
        calibration=documents.forty,
    )

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_scan(rig))

    _skipped(
        result,
        f"the plate was not checked: the offset between {TEN} and {FORTY} is not "
        "measured (the profile's [objectives]), and a check would read it as a move",
        caplog,
    )
    assert rig.camera.snaps == 2


def test_a_scan_under_the_40x_with_the_profiles_offset_checks_there(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(
        stands, documents, tmp_path, objective=FORTY, calibration=documents.forty
    )

    result = _result(_scan(rig))

    check = result.plate_check
    assert check is not None
    assert (check.plate_objective, check.objective) == (TEN, FORTY)
    assert check.shift_um == pytest.approx((-37.5, 12.25), abs=1e-9)
    assert check.worst_um < 0.1


def test_a_plate_registered_by_plate_registration_is_not_checked_and_said(
    stands: list[Microscope],
    documents: Documents,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_doc = tmp_path / "run.json"
    shutil.copy(documents.run, run_doc)
    section = profile().sample
    assert section is not None
    stand = open_stand(profile())
    stands.append(stand)
    record = run_plugin(
        PlateRegistrationPlugin(),
        stand,
        PlateRegistrationParams(plate="96-well", refs=sample_refs(section)),
        run_doc,
    )
    assert record.status is StageStatus.done
    record = run_plugin(
        ScanPlanPlugin(), stand, ScanPlanParams(wells="A1", **SMALL), run_doc
    )
    assert record.status is StageStatus.done
    rig = _Rig(
        stand,
        _Stage(stand.require(XYStage), []),
        _Camera(stand.require(Camera)),
        [],
        run_doc,
    )

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_scan(rig))

    _skipped(
        result,
        "the plate was not checked: it was registered by references, which measures "
        "no well's walls (plate-walls does)",
        caplog,
    )


def test_a_plan_whose_matrix_was_not_measured_skips_the_check_and_says_so(
    stands: list[Microscope],
    documents: Documents,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = _rig(stands, documents, tmp_path, pixel_size_um=1.0)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_scan(rig))

    _skipped(
        result,
        "the plate was not checked: the plan's camera matrix is the parameter's, and "
        "a wall is looked for with a measured one only",
        caplog,
        stand_lines=(
            "the plan's pixel size was given as a parameter: not compared with the "
            "camera's",
        ),
    )


# --------------------------------------------------------------- the autofocus


def test_with_a_locked_autofocus_the_check_walks_with_it_off_and_the_first_field_engages_it_again(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    af = RecordingAutofocus(engaged=True, locked=True)
    rig = _rig(stands, documents, tmp_path, autofocus=af)
    del af.calls[:]

    result = _result(_scan(rig))

    assert result.plate_check is not None
    calls = af.calls
    first_move = calls.index("move_to_um")
    # The scan's own reads, then the first well's switch-off and its read back,
    # all before the check's first move.
    assert calls[:first_move][-3:] == ["is_engaged", "disengage", "is_engaged"]
    # The check's 16 moves come with no call to the autofocus between them.
    assert calls[first_move : first_move + 16] == ["move_to_um"] * 16
    # The first field's move, then its read and the engage after the wall.
    after = calls[first_move + 16 :]
    assert after[:3] == ["move_to_um", "is_locked", "engage"]
    assert calls.count("disengage") == 1
    assert af.engaged is True
    assert result.assumed[0].startswith(
        "the plate was checked before the first field: "
    )
    assert (
        "the autofocus was switched off before the first field of 1 wells, so that "
        "no lock crossed a well's wall, and engaged again after it on 1 fields, "
        "locked on 1 of them"
    ) in result.assumed


def test_a_refused_check_after_the_switch_off_says_the_autofocus_was_left_off(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    af = RecordingAutofocus(engaged=True, locked=True)
    rig = _rig(stands, documents, tmp_path, autofocus=af)
    move_registration(rig.run_doc, slide_um=(50.0, 0.0))

    refused = _refused(rig)

    assert refused.reason.startswith("the plate moved since plate-walls registered it")
    assert refused.reason.endswith(AGAIN + AUTOFOCUS_LEFT_OFF)
    assert "engage" not in af.calls
    assert af.engaged is False


def test_a_refused_check_without_an_autofocus_says_nothing_of_it(
    stands: list[Microscope], documents: Documents, tmp_path: Path
) -> None:
    rig = _rig(stands, documents, tmp_path)
    move_registration(rig.run_doc, slide_um=(50.0, 0.0))

    refused = _refused(rig)

    assert refused.reason.endswith(AGAIN)
    assert math.isfinite(rig.stage.position_um().x_um)
