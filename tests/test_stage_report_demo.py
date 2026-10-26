"""``stage-report`` on the demo devices: the whole plugin path, nothing moves (m3 §1.11).

It reads where the stage is, so the entry point, the parameters, the
requirements, the progress, the stop and the run document are all exercised
without sending a move. The demo stage's position is not asserted against another
open of the demo (FM-47): the readings are compared with each other and with what
the same stand reports.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage, ZStage
from smc.hardware.profile import Profile
from smc.plugins import Progress, discover, load, run_plugin
from smc.plugins.stage_report import (
    StageReading,
    StageReport,
    StageReportParams,
    StageReportResult,
)
from smc.run import RunDocument, StageStatus
from smc.testing import FakeCore

pytestmark = pytest.mark.demo


def _run(
    microscope: Microscope, tmp_path: Path, params: StageReportParams, **kw: object
) -> StageReportResult:
    record = run_plugin(StageReport(), microscope, params, tmp_path / "run.json", **kw)  # type: ignore[arg-type]
    assert record.result is not None
    return StageReportResult.model_validate(record.result)


def test_stage_report_reads_xy_and_z_on_the_demo(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    result = _run(demo_microscope, tmp_path, StageReportParams(samples=3))
    assert result.profile == "demo"
    assert len(result.readings) == 3
    assert all(reading.z_um is not None for reading in result.readings)
    here = demo_microscope.require(XYStage).position_um()
    z_here = demo_microscope.require(ZStage).position_um()
    # Nothing moved: every reading is where the stand still is.
    assert {(r.x_um, r.y_um, r.z_um) for r in result.readings} == {
        (here.x_um, here.y_um, z_here)
    }


def test_stage_report_without_z_leaves_z_none(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    result = _run(
        demo_microscope, tmp_path, StageReportParams(samples=2, include_z=False)
    )
    assert [r.z_um for r in result.readings] == [None, None]


def test_stage_report_reports_progress_per_sample(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    seen: list[Progress] = []
    _run(
        demo_microscope,
        tmp_path,
        StageReportParams(samples=3),
        on_progress=seen.append,
    )
    assert seen == [Progress("stage-report", i, 3, "read") for i in (1, 2, 3)]


def test_stage_report_stops_between_samples(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    stop = threading.Event()
    seen: list[Progress] = []

    def report(progress: Progress) -> None:
        seen.append(progress)
        stop.set()  # a UI's Stop, after the first sample

    record = run_plugin(
        StageReport(),
        demo_microscope,
        StageReportParams(samples=3),
        tmp_path / "run.json",
        on_progress=report,
        stop=stop,
    )
    assert record.status is StageStatus.stopped
    assert record.result is None
    assert [p.done for p in seen] == [1]  # one reading, then the stop
    assert RunDocument.load(tmp_path / "run.json").stages["stage-report"] == record


def test_stage_report_reads_nothing_when_stopped_before_it_starts(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # The plugin's own check before each reading, not only the context's wait.
    stop = threading.Event()
    stop.set()
    seen: list[Progress] = []
    record = run_plugin(
        StageReport(),
        demo_microscope,
        StageReportParams(samples=3),
        tmp_path / "run.json",
        on_progress=seen.append,
        stop=stop,
    )
    assert record.status is StageStatus.stopped
    assert seen == []


def test_stage_report_waits_between_two_readings_and_not_after_the_last(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    # The context's own wait, recorded and free: the plugin asks for it, once
    # per gap.
    from smc.plugins import api

    monkeypatch.setattr(api.RunContext, "wait_s", lambda self, s: waits.append(s))
    _run(
        demo_microscope,
        tmp_path,
        StageReportParams(samples=3, interval_ms=250.0),
    )
    assert waits == [0.25, 0.25]


def test_stage_report_is_registered_and_its_params_come_from_its_model() -> None:
    entry = load("stage-report")
    assert isinstance(entry.plugin, StageReport)
    assert entry.plugin.requires == (XYStage,)
    assert entry.plugin.provides == "stage-report"
    assert discover().failures == ()


def test_stage_report_on_a_stand_without_a_focus_drive_leaves_z_none(
    tmp_path: Path,
) -> None:
    # `requires` is the XY stage only: Z is read when there is a drive and left
    # unknown ("unknown beats guessed") when there is not, never a refusal.
    profile_file = tmp_path / "no-focus.toml"
    profile_file.write_text(
        '[microscope]\nname = "no-focus"\n\n[roles.exclude]\nfocus = ["Z"]\n',
        encoding="utf-8",
    )
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(
        cast("CMMCorePlus", core), Profile.load(profile_file)
    )
    assert not microscope.has(ZStage)
    result = _run(microscope, tmp_path, StageReportParams(samples=2))
    assert result.profile == "no-focus"
    assert [reading.z_um for reading in result.readings] == [None, None]
    assert all(reading.x_um is not None for reading in result.readings)


def test_a_reading_is_a_frozen_model_with_unknown_z_as_none() -> None:
    reading = StageReading(x_um=1.0, y_um=2.0, z_um=None)
    assert reading.model_dump(mode="json") == {"x_um": 1.0, "y_um": 2.0, "z_um": None}
    with pytest.raises(ValidationError, match="Instance is frozen"):
        reading.x_um = 3.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Input should be a finite number"):
        StageReading(x_um=float("nan"), y_um=2.0, z_um=None)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StageReading(x_um=1.0, y_um=2.0, z_um=None, w_um=0.0)  # type: ignore[call-arg]
