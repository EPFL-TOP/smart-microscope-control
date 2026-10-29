"""The plugin runner (m3 §1.8) and the context it hands a plugin (§1.4).

A plugin is tested through ``run_plugin`` with a run document under
``tmp_path``, never by building a context. Everything runs on ``fake_microscope``
(no Micro-Manager). The text of every refusal and of every record is asserted
exactly (FM-45); threads meet on ``Event`` and are joined with a timeout (FM-44).
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field
from pymmcore_plus import CMMCorePlus

import smc
from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage, ZStage
from smc.hardware.errors import HardwareError, SafetyRefusedError
from smc.hardware.profile import Profile
from smc.plugins import api as api_mod
from smc.plugins.api import (
    Plugin,
    PluginParams,
    PluginResult,
    Progress,
    RunContext,
    StopRequested,
)
from smc.plugins.errors import (
    PluginRunError,
    RequirementsError,
    ResumeError,
    UpstreamError,
)
from smc.plugins.runner import check_ready, run_plugin
from smc.run import (
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)
from smc.run import files as files_mod
from smc.testing import FakeCore

T0 = datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)


class Params(PluginParams):
    n: int = 1


class Result(PluginResult):
    value: int = 0


class Seen(BaseModel):
    """What a checkpoint holds in these tests."""

    done: int


class Base:
    """A plugin that returns ``Result(value=n)``; variants override ``body``."""

    name = "base"
    summary = "A test plugin."
    Params = Params
    Result = Result
    requires: tuple[type, ...] = ()
    provides = "base"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.calls = 0
        self.seen: dict[str, Any] = {}

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        self.calls += 1
        return self.body(microscope, params, ctx)

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        return Result(value=params.n)


class Echo(Base):
    name = provides = "echo"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        # What the run document shows while the plugin runs (m3 §1.8, step 4).
        loaded = RunDocument.load(ctx.run_dir / "run.json").stages["echo"]
        self.seen = {
            "status": loaded.status,
            "checkpoint": loaded.checkpoint,
            "logger": ctx.logger.name,
            "run_dir": ctx.run_dir,
            "run_doc": ctx.run_doc,
            "dry_run": ctx.dry_run,
            "plugin": ctx.plugin,
        }
        return Result(value=params.n)


class Up(Base):
    name = provides = "up"


class Reader(Base):
    name = provides = "reader"
    reads = ("up",)

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        self.seen = {"upstream": ctx.upstream("up", Result)}
        return Result(value=self.seen["upstream"].value + 1)


class Optional(Base):
    name = provides = "optional"
    reads_optional = ("timings",)

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        self.seen = {"timings": ctx.upstream_or_none("timings", Result)}
        return Result()


class NeedsBoth(Base):
    name = provides = "needs-both"
    requires = (XYStage, ZStage)


class Boom(Base):
    name = provides = "boom"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        raise HardwareError("the stage did not answer")


class Unusable(Base):
    """Does its work, saves a checkpoint, and finds nothing a reader could trust."""

    name = provides = "unusable"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        ctx.save_checkpoint(Seen(done=2))
        raise PluginRunError(self.name, "no plate found; run it again")


class Refuses(Base):
    name = provides = "refuses"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        raise SafetyRefusedError("the jog is too long", "re-run with --force")


class Interrupted(Base):
    name = provides = "interrupted"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        ctx.save_checkpoint(Seen(done=2))
        raise KeyboardInterrupt


class Stops(Base):
    name = provides = "stops"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        ctx.save_checkpoint(Seen(done=4))
        ctx.raise_if_stopped()  # not set: carries on
        raise StopRequested


class WrongResult(Base):
    name = provides = "wrong"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Any:
        return object()


class Resumable(Base):
    """Does two units of work, saves a checkpoint after each, can be told to stop."""

    name = provides = "resumable"

    def __init__(self, stop_after: int | None = None) -> None:
        super().__init__()
        self.stop_after = stop_after
        self.started_from: Seen | None = None
        self.checkpoint_at_start: dict[str, Any] | None = None

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        self.started_from = ctx.checkpoint(Seen)
        start = self.started_from.done if self.started_from else 0
        for unit in range(start, 3):
            if self.stop_after is not None and unit == self.stop_after:
                raise StopRequested
            ctx.save_checkpoint(Seen(done=unit + 1))
            # The checkpoint seen by the plugin stays the one it started from.
            assert ctx.checkpoint(Seen) == self.started_from
        return Result(value=3)


class Saves(Base):
    """Saves two checkpoints with another writer's stage added in between."""

    name = provides = "saves"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        path = ctx.run_dir / "run.json"
        ctx.save_checkpoint(Seen(done=1))
        self.seen["after_first"] = RunDocument.load(path).stages["saves"].checkpoint
        write_stage(path, _done_record("other-writer"))
        self.seen["units"] = 1
        ctx.save_checkpoint(Seen(done=2))
        self.seen["units"] = 2
        stored = RunDocument.load(path)
        self.seen["stages"] = sorted(stored.stages)
        self.seen["status"] = stored.stages["saves"].status
        self.seen["checkpoint"] = stored.stages["saves"].checkpoint
        return Result()


class ReadingResumable(Resumable):
    """A resumable stage that reads ``up``: its checkpoint is made against it."""

    name = provides = "reading-resumable"
    reads = ("up",)


class NeedsXYAndUp(Base):
    name = provides = "needs-xy-and-up"
    requires = (XYStage,)
    reads = ("up",)


class Progressing(Base):
    name = provides = "progressing"

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        for done in (1, 2, 3):
            ctx.progress(done, 3, "step")
        return Result(value=3)


class Waits(Base):
    name = provides = "waits"

    def __init__(self, seconds: float = 5.0) -> None:
        super().__init__()
        self.seconds = seconds
        self.waiting = threading.Event()

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        self.waiting.set()
        ctx.wait_s(self.seconds)
        return Result(value=1)


class AsksFor(Base):
    """Asks the context for slots it did not declare."""

    name = provides = "asks-for"
    reads_optional = ("timings",)

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        which = params.n
        if which == 1:
            ctx.upstream("nothing", Result)
        elif which == 2:
            ctx.upstream("timings", Result)  # declared optional, not required
        else:
            ctx.upstream_or_none("nothing", Result)
        return Result()


def _path(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _run(
    plugin: Base, microscope: Microscope, tmp_path: Path, n: int = 1, **kw: Any
) -> StageRecord:
    return run_plugin(plugin, microscope, Params(n=n), _path(tmp_path), **kw)


def _stored(tmp_path: Path, name: str) -> StageRecord:
    return RunDocument.load(_path(tmp_path)).stages[name]


def _done_record(
    plugin: str,
    *,
    provides: str | None = None,
    profile: str = "demo",
    status: StageStatus = StageStatus.done,
    dry_run: bool = False,
    result: dict[str, Any] | None = None,
    minute: int = 0,
) -> StageRecord:
    return StageRecord(
        plugin=plugin,
        provides=provides or plugin,
        attempt=f"{plugin}-{minute}",
        status=status,
        started_utc=T0.replace(minute=minute),
        profile=profile,
        dry_run=dry_run,
        smc_version="0.0.1",
        params={},
        result=result if result is not None else {"assumed": [], "value": 7},
    )


@pytest.fixture
def two_stage_less_stand(tmp_path: Path) -> Microscope:
    """The demo devices with the XY stage and the focus drive excluded (measured, §1.1)."""
    profile = tmp_path / "bare.toml"
    profile.write_text(
        '[microscope]\nname = "bare"\n\n[roles.exclude]\nxy_stage = ["XY"]\n'
        'focus = ["Z"]\n',
        encoding="utf-8",
    )
    core = FakeCore.demo_like()
    return Microscope.from_core(cast("CMMCorePlus", core), Profile.load(profile))


# --- outcomes ----------------------------------------------------------------


def test_a_done_run_records_params_result_reads_and_versions(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    plugin = Echo()
    before = datetime.now(timezone.utc)
    record = _run(plugin, fake_microscope, tmp_path, n=3, plugin_version="1.2.3")
    assert record.status is StageStatus.done
    assert (record.plugin, record.provides) == ("echo", "echo")
    assert record.params == {"n": 3}
    assert record.result == {"assumed": [], "value": 3}
    assert record.reads == {}
    assert record.checkpoint is None
    assert record.error is None
    assert (record.profile, record.dry_run) == ("demo", False)
    assert record.smc_version == smc.__version__
    assert record.plugin_version == "1.2.3"
    assert len(record.attempt) == 32
    assert record.finished_utc is not None
    assert before <= record.started_utc <= record.finished_utc
    assert record.started_utc.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    assert _stored(tmp_path, "echo") == record


def test_the_running_record_is_written_before_run_and_the_context_is_wired(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # A crash then shows `running`, which `unusable` reads as interrupted.
    plugin = Echo()
    _run(plugin, fake_microscope, tmp_path)
    assert plugin.seen == {
        "status": StageStatus.running,
        "checkpoint": None,
        "logger": "smc.plugins.echo",
        "run_dir": tmp_path,
        "run_doc": tmp_path / "run.json",
        "dry_run": False,
        "plugin": "echo",
    }


def test_a_reader_records_which_attempt_of_its_upstream_it_read(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    up = _run(Up(), fake_microscope, tmp_path, n=4)
    reader = Reader()
    record = _run(reader, fake_microscope, tmp_path)
    assert reader.seen["upstream"] == Result(value=4)
    assert record.result == {"assumed": [], "value": 5}
    assert record.reads == {"up": ReadRecord(plugin="up", attempt=up.attempt)}
    assert _stored(tmp_path, "reader").reads == record.reads


def test_params_of_the_wrong_type_are_a_type_error_and_write_nothing(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match=r"^echo takes Params, not Result$"):
        run_plugin(Echo(), fake_microscope, Result(), _path(tmp_path))
    assert not _path(tmp_path).exists()


def test_a_missing_capability_lists_every_one_and_writes_nothing(
    two_stage_less_stand: Microscope, tmp_path: Path
) -> None:
    plugin = NeedsBoth()
    with pytest.raises(RequirementsError) as raised:
        _run(plugin, two_stage_less_stand, tmp_path)
    error = raised.value
    assert error.missing == ("XYStage", "ZStage")
    assert len(error.details) == 2
    assert error.details[0].startswith("XYStage needs the 'xy_stage' role, which no")
    assert error.details[1].startswith("ZStage needs the 'focus' role, which no")
    assert all("[roles.assign]" in detail for detail in error.details)  # FM-36
    assert str(error).startswith(
        "needs-both needs XYStage and ZStage, which this stand does not have: "
    )
    assert plugin.calls == 0
    assert not _path(tmp_path).exists()


def test_a_capability_the_stand_has_is_not_listed_as_missing(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    record = _run(NeedsBoth(), fake_microscope, tmp_path)
    assert record.status is StageStatus.done


def test_a_missing_upstream_is_refused_before_run(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    reader = Reader()
    with pytest.raises(UpstreamError) as raised:
        _run(reader, fake_microscope, tmp_path)
    assert raised.value.reason == (
        "the run document has no up; smc run --list shows which plugin provides it"
    )
    assert (raised.value.plugin, raised.value.slot) == ("reader", "up")
    assert reader.calls == 0
    assert not _path(tmp_path).exists()


def test_an_upstream_of_another_profile_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    write_stage(_path(tmp_path), _done_record("up", profile="other"))
    reader = Reader()
    with pytest.raises(UpstreamError) as raised:
        _run(reader, fake_microscope, tmp_path)
    assert raised.value.reason == (
        "up was run on other, this is demo; run up again on this profile"
    )
    assert reader.calls == 0


def test_an_unreadable_run_document_stops_before_the_hardware_and_is_kept(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    path = _path(tmp_path)
    path.write_bytes(b"{ this is not json")
    plugin = Echo()
    with pytest.raises(RunDocumentError):
        _run(plugin, fake_microscope, tmp_path)
    assert plugin.calls == 0
    assert path.read_bytes() == b"{ this is not json"


def test_keyboard_interrupt_records_stopped_and_is_re_raised(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-71: the operator's stop is recorded, then re-raised, not turned into a result.
    with pytest.raises(KeyboardInterrupt):
        _run(Interrupted(), fake_microscope, tmp_path)
    stored = _stored(tmp_path, "interrupted")
    assert stored.status is StageStatus.stopped
    assert stored.error == "KeyboardInterrupt: Ctrl-C"
    assert stored.checkpoint == {"done": 2}
    assert stored.result is None
    assert stored.finished_utc is not None


def test_a_hardware_error_records_failed_and_is_re_raised_unchanged(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    with pytest.raises(HardwareError) as raised:
        _run(Boom(), fake_microscope, tmp_path)
    assert str(raised.value) == "the stage did not answer"
    stored = _stored(tmp_path, "boom")
    assert stored.status is StageStatus.failed
    assert stored.error == "HardwareError: the stage did not answer"
    assert stored.result is None


def test_a_plugin_run_error_ends_the_record_failed_and_keeps_the_checkpoint(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    with pytest.raises(PluginRunError):
        _run(Unusable(), fake_microscope, tmp_path)
    stored = _stored(tmp_path, "unusable")
    assert stored.status is StageStatus.failed
    assert stored.error == "PluginRunError: unusable: no plate found; run it again"
    assert stored.checkpoint == {"done": 2}
    assert stored.result is None
    # No unusable result fills the slot for a reader to trust.
    doc = RunDocument.load(_path(tmp_path))
    assert doc.unusable("unusable", profile="demo", dry_run=False) == (
        "the last run of unusable ended failed; run it again"
    )


def test_a_safety_refusal_is_not_caught_by_the_runner(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # The CLI maps it to exit 2 by its type, so it must arrive as it was.
    with pytest.raises(SafetyRefusedError) as raised:
        _run(Refuses(), fake_microscope, tmp_path)
    assert raised.value.how_to_force == "re-run with --force"
    stored = _stored(tmp_path, "refuses")
    assert stored.status is StageStatus.failed
    assert stored.error == (
        "SafetyRefusedError: the jog is too long — re-run with --force"
    )


def test_stop_requested_records_stopped_and_returns_the_record(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # A stop is an outcome, not an error.
    record = _run(Stops(), fake_microscope, tmp_path)
    assert record.status is StageStatus.stopped
    assert record.error is None
    assert record.result is None
    assert record.checkpoint == {"done": 4}
    assert record.finished_utc is not None
    assert _stored(tmp_path, "stops") == record


def test_a_wrong_result_type_records_failed(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match=r"^wrong returned object, not Result$"):
        _run(WrongResult(), fake_microscope, tmp_path)
    stored = _stored(tmp_path, "wrong")
    assert stored.status is StageStatus.failed
    assert stored.error == "TypeError: wrong returned object, not Result"


def test_an_exception_message_is_recorded_on_one_line(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    class Multi(Base):
        name = provides = "multi"

        def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
            raise HardwareError("first line\n  second   line")

    with pytest.raises(HardwareError):
        _run(Multi(), fake_microscope, tmp_path)
    assert _stored(tmp_path, "multi").error == "HardwareError: first line second line"


def test_a_rerun_replaces_the_stage_and_keeps_the_others(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    first = _run(Up(), fake_microscope, tmp_path, n=1)
    _run(Echo(), fake_microscope, tmp_path)
    second = _run(Up(), fake_microscope, tmp_path, n=2)
    stages = RunDocument.load(_path(tmp_path)).stages
    assert sorted(stages) == ["echo", "up"]
    assert stages["up"].attempt == second.attempt != first.attempt
    assert stages["up"].result == {"assumed": [], "value": 2}


def test_a_rerun_of_an_upstream_makes_its_reader_stale(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _run(Up(), fake_microscope, tmp_path)
    _run(Reader(), fake_microscope, tmp_path)
    assert not RunDocument.load(_path(tmp_path)).is_stale("reader")
    _run(Up(), fake_microscope, tmp_path)
    assert RunDocument.load(_path(tmp_path)).is_stale("reader")


# --- optional reads ----------------------------------------------------------


def test_an_unusable_optional_read_is_recorded_as_none(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="smc.plugins")
    plugin = Optional()
    record = _run(plugin, fake_microscope, tmp_path)
    assert record.reads == {"timings": None}
    assert plugin.seen == {"timings": None}
    assert (
        "optional: not using timings: the run document has no timings; smc run "
        "--list shows which plugin provides it"
    ) in caplog.messages


def _timing_failed() -> list[StageRecord]:
    return [_done_record("timing", provides="timings", status=StageStatus.failed)]


def _timing_stopped() -> list[StageRecord]:
    return [_done_record("timing", provides="timings", status=StageStatus.stopped)]


def _timing_running() -> list[StageRecord]:
    return [_done_record("timing", provides="timings", status=StageStatus.running)]


def _timing_on_another_stand() -> list[StageRecord]:
    return [_done_record("timing", provides="timings", profile="another-stand")]


def _timing_was_a_dry_run() -> list[StageRecord]:
    return [_done_record("timing", provides="timings", dry_run=True)]


def _timing_is_stale() -> list[StageRecord]:
    origin = _done_record("origin", minute=1)  # attempt origin-1
    timing = _done_record("timing", provides="timings", minute=2).model_copy(
        update={"reads": {"origin": ReadRecord(plugin="origin", attempt="origin-0")}}
    )
    return [origin, timing]


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        pytest.param(
            _timing_failed,
            "the last run of timing ended failed; run it again",
            id="failed",
        ),
        pytest.param(
            _timing_stopped,
            "the last run of timing ended stopped; run it again",
            id="stopped",
        ),
        pytest.param(
            _timing_running,
            "the last run of timing is still running or was interrupted; run it again",
            id="running",
        ),
        pytest.param(
            _timing_on_another_stand,
            "timing was run on another-stand, this is demo; run timing again on "
            "this profile",
            id="another-profile",
        ),
        pytest.param(
            _timing_was_a_dry_run,
            "timing was a dry run; run it without --dry-run",
            id="dry-run-provider",
        ),
        pytest.param(
            _timing_is_stale,
            "timing read its inputs from an earlier run; run it again",
            id="stale",
        ),
    ],
)
def test_every_way_an_optional_slot_is_unusable_is_recorded_as_none(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    records: Any,
    reason: str,
) -> None:
    # FM-106: another stand's result, a failed or interrupted run, a rehearsal
    # and a stale one are never handed to the plugin as the timings it asked for.
    caplog.set_level(logging.INFO, logger="smc.plugins")
    for record in records():
        write_stage(_path(tmp_path), record)
    plugin = Optional()
    record = _run(plugin, fake_microscope, tmp_path)
    assert record.reads == {"timings": None}
    assert plugin.seen == {"timings": None}
    assert f"optional: not using timings: {reason}" in caplog.messages


def test_a_usable_optional_read_is_recorded_and_returned(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    write_stage(
        _path(tmp_path),
        _done_record("timing", provides="timings", result={"assumed": [], "value": 9}),
    )
    plugin = Optional()
    record = _run(plugin, fake_microscope, tmp_path)
    assert record.reads == {"timings": ReadRecord(plugin="timing", attempt="timing-0")}
    assert plugin.seen == {"timings": Result(value=9)}


def test_a_usable_optional_result_that_the_model_refuses_raises_upstream_error(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    write_stage(
        _path(tmp_path),
        _done_record("timing", provides="timings", result={"unexpected": 1}),
    )
    with pytest.raises(UpstreamError) as raised:
        _run(Optional(), fake_microscope, tmp_path)
    assert raised.value.reason == (
        "the result of timing is not what optional expects (unexpected: Extra "
        "inputs are not permitted); run timing again"
    )
    assert _stored(tmp_path, "optional").status is StageStatus.failed


def test_an_upstream_that_the_model_refuses_names_slot_provider_and_error(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    write_stage(_path(tmp_path), _done_record("up", result={"value": "not a number"}))
    with pytest.raises(UpstreamError) as raised:
        _run(Reader(), fake_microscope, tmp_path)
    assert str(raised.value) == (
        "reader reads up, but the result of up is not what reader expects (value: "
        "Input should be a valid integer, unable to parse string as an integer); "
        "run up again"
    )


@pytest.mark.parametrize(
    ("which", "problem"),
    [
        pytest.param(
            1,
            "asks-for does not declare 'nothing' in reads, so the runner did not "
            "check it; add it to reads (or to reads_optional, and call "
            "upstream_or_none)",
            id="upstream-of-an-undeclared-slot",
        ),
        pytest.param(
            2,
            "asks-for does not declare 'timings' in reads, so the runner did not "
            "check it; add it to reads (or to reads_optional, and call "
            "upstream_or_none)",
            id="upstream-of-an-optional-slot",
        ),
        pytest.param(
            3,
            "asks-for does not declare 'nothing' in reads_optional; add it there "
            "(or to reads, and call upstream)",
            id="upstream-or-none-of-an-undeclared-slot",
        ),
    ],
)
def test_upstream_of_an_undeclared_slot_is_a_value_error(
    fake_microscope: Microscope, tmp_path: Path, which: int, problem: str
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(problem)}$"):
        _run(AsksFor(), fake_microscope, tmp_path, n=which)
    assert _stored(tmp_path, "asks-for").status is StageStatus.failed


# --- checkpoints -------------------------------------------------------------


def test_resume_offers_the_previous_checkpoint(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    stopped = _run(Resumable(stop_after=2), fake_microscope, tmp_path)
    assert stopped.status is StageStatus.stopped
    assert stopped.checkpoint == {"done": 2}
    plugin = Resumable()
    resumed = _run(plugin, fake_microscope, tmp_path, resume=True)
    assert plugin.started_from == Seen(done=2)
    assert resumed.status is StageStatus.done
    assert resumed.checkpoint is None  # cleared when the stage ends done
    assert resumed.attempt != stopped.attempt  # a resumed run is a new attempt


def test_a_resumed_run_that_stops_again_keeps_its_latest_checkpoint(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _run(Resumable(stop_after=1), fake_microscope, tmp_path)
    again = _run(Resumable(stop_after=2), fake_microscope, tmp_path, resume=True)
    assert again.status is StageStatus.stopped
    assert again.checkpoint == {"done": 2}


def test_checkpoint_is_none_without_resume_and_a_fresh_run_clears_the_old_one(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="smc.plugins")
    _run(Resumable(stop_after=2), fake_microscope, tmp_path)
    plugin = Resumable()
    _run(plugin, fake_microscope, tmp_path)
    assert plugin.started_from is None
    assert (
        "resumable: the previous attempt (stopped) saved a checkpoint; --resume "
        "would continue it, and this run starts afresh"
    ) in caplog.messages


def test_a_running_record_of_a_fresh_run_holds_no_checkpoint(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    class Look(Resumable):
        name = provides = "resumable"

        def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
            self.checkpoint_at_start = (
                RunDocument.load(ctx.run_dir / "run.json")
                .stages["resumable"]
                .checkpoint
            )
            return Result()

    _run(Resumable(stop_after=2), fake_microscope, tmp_path)
    fresh = Look()
    _run(fresh, fake_microscope, tmp_path)
    assert fresh.checkpoint_at_start is None
    resuming = Look()
    # Not resumed: the same call with resume=True shows the old checkpoint.
    _run(Resumable(stop_after=2), fake_microscope, tmp_path)
    _run(resuming, fake_microscope, tmp_path, resume=True)
    assert resuming.checkpoint_at_start == {"done": 2}


def test_a_checkpoint_reaches_the_document_at_once_and_reverts_no_other_stage(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # The run document, not the plugin's memory, is what a crash leaves behind.
    plugin = Saves()
    record = _run(plugin, fake_microscope, tmp_path)
    assert plugin.seen["after_first"] == {"done": 1}
    assert plugin.seen["stages"] == ["other-writer", "saves"]
    assert plugin.seen["status"] is StageStatus.running
    assert plugin.seen["checkpoint"] == {"done": 2}
    assert record.checkpoint is None
    assert sorted(RunDocument.load(_path(tmp_path)).stages) == ["other-writer", "saves"]


def test_a_checkpoint_that_cannot_be_written_ends_the_run(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace, calls = files_mod.os.replace, []

    def replace_once(src: Any, dst: Any) -> None:
        calls.append(src)
        if len(calls) > 1:  # the `running` write went through
            raise PermissionError(13, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(files_mod.os, "replace", replace_once)
    monkeypatch.setattr(files_mod.time, "sleep", lambda seconds: None)
    plugin = Saves()
    with pytest.raises(RunDocumentError) as raised:
        _run(plugin, fake_microscope, tmp_path)
    assert raised.value.reason.startswith("could not be replaced (Permission denied)")
    assert plugin.seen == {}  # not even the read after the first checkpoint


def test_resume_of_a_stage_whose_input_was_produced_again_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-106: the checkpoint was made against the first `up`; resuming would finish
    # the work on the second and record the whole run as read from it.
    _run(Up(), fake_microscope, tmp_path, n=1)
    stopped = _run(ReadingResumable(stop_after=1), fake_microscope, tmp_path)
    assert stopped.checkpoint == {"done": 1}
    _run(Up(), fake_microscope, tmp_path, n=2)
    plugin = ReadingResumable()
    with pytest.raises(ResumeError) as raised:
        _run(plugin, fake_microscope, tmp_path, resume=True)
    assert raised.value.reason == (
        "reading-resumable read its inputs from an earlier run, so its checkpoint "
        "no longer matches them; run it without --resume"
    )
    assert plugin.calls == 0
    assert _stored(tmp_path, "reading-resumable") == stopped  # untouched
    # A fresh run is the way out, and reads the current input.
    assert _run(ReadingResumable(), fake_microscope, tmp_path).status is (
        StageStatus.done
    )


def test_resume_of_a_stage_whose_inputs_are_current_is_allowed(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # The control of the test above: the same sequence without the second `up`.
    _run(Up(), fake_microscope, tmp_path)
    _run(ReadingResumable(stop_after=1), fake_microscope, tmp_path)
    resumed = _run(ReadingResumable(), fake_microscope, tmp_path, resume=True)
    assert resumed.status is StageStatus.done


def test_the_requirements_are_checked_after_what_the_stage_reads(
    two_stage_less_stand: Microscope, tmp_path: Path
) -> None:
    # Two checks are false at once (§1.8: check_ready, then the requirements): the
    # first of the order is the error, and it needs no stand to know.
    plugin = NeedsXYAndUp()
    with pytest.raises(UpstreamError):
        _run(plugin, two_stage_less_stand, tmp_path)
    assert plugin.calls == 0


def test_resume_with_other_params_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _run(Resumable(stop_after=1), fake_microscope, tmp_path, n=1)
    with pytest.raises(ResumeError) as raised:
        _run(Resumable(), fake_microscope, tmp_path, n=2, resume=True)
    assert raised.value.reason == "the parameters differ from the run being resumed: n"
    assert str(raised.value) == (
        "cannot resume resumable: the parameters differ from the run being resumed: n"
    )
    assert _stored(tmp_path, "resumable").status is StageStatus.stopped  # untouched


def test_resume_of_a_done_stage_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _run(Resumable(), fake_microscope, tmp_path)
    with pytest.raises(ResumeError) as raised:
        _run(Resumable(), fake_microscope, tmp_path, resume=True)
    assert raised.value.reason == "resumable finished; run it without --resume"


def test_resume_with_nothing_to_resume_is_refused(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    plugin = Resumable()
    with pytest.raises(ResumeError) as raised:
        _run(plugin, fake_microscope, tmp_path, resume=True)
    assert raised.value.reason == "nothing to resume"
    assert plugin.calls == 0
    assert not _path(tmp_path).exists()


def test_a_checkpoint_the_model_refuses_raises_resume_error(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    _run(Resumable(stop_after=1), fake_microscope, tmp_path)
    record = _stored(tmp_path, "resumable").model_copy(
        update={"checkpoint": {"done": "many"}}
    )
    write_stage(_path(tmp_path), record)
    with pytest.raises(ResumeError) as raised:
        _run(Resumable(), fake_microscope, tmp_path, resume=True)
    assert raised.value.reason == (
        "the saved checkpoint is not what resumable expects (done: Input should be "
        "a valid integer, unable to parse string as an integer); run it again "
        "without --resume"
    )


def _check(doc: RunDocument, plugin: Base, **kw: Any) -> None:
    check_ready(
        plugin,
        doc,
        Params(n=kw.pop("n", 1)),
        profile=kw.pop("profile", "demo"),
        dry_run=kw.pop("dry_run", False),
        resume=kw.pop("resume", False),
    )


def _doc(*records: StageRecord) -> RunDocument:
    return RunDocument(created_utc=T0, stages={r.plugin: r for r in records})


def test_check_ready_resume_refuses_another_profile_and_another_kind_of_run() -> None:
    stopped = _done_record(
        "resumable", status=StageStatus.stopped, profile="other"
    ).model_copy(update={"params": {"n": 1}, "result": None})
    with pytest.raises(ResumeError) as raised:
        _check(_doc(stopped), Resumable(), resume=True)
    assert raised.value.reason == (
        "resumable was run on other, this is demo; resume it there, or run it "
        "without --resume"
    )
    # A dry run's checkpoint stands for work that was never done: a real run must
    # not skip it. The other way round is harmless.
    dry = stopped.model_copy(update={"profile": "demo", "dry_run": True})
    with pytest.raises(ResumeError) as raised:
        _check(_doc(dry), Resumable(), resume=True)
    assert raised.value.reason == "resumable was a dry run; run it without --resume"
    _check(_doc(dry), Resumable(), resume=True, dry_run=True)
    real = stopped.model_copy(update={"profile": "demo"})
    _check(_doc(real), Resumable(), resume=True, dry_run=True)
    _check(_doc(real), Resumable(), resume=True)


def test_check_ready_names_every_parameter_that_differs() -> None:
    class Two(PluginParams):
        a: int = 1
        b: int = 1
        c: int = 1

    class TwoPlugin(Base):
        name = provides = "two"
        Params = Two

    stopped = _done_record("two", status=StageStatus.stopped).model_copy(
        update={"params": {"a": 1, "b": 2, "d": 9}, "result": None}
    )
    with pytest.raises(ResumeError) as raised:
        check_ready(
            TwoPlugin(),
            _doc(stopped),
            Two(),
            profile="demo",
            dry_run=False,
            resume=True,
        )
    assert raised.value.reason == (
        "the parameters differ from the run being resumed: b, c, d"
    )


def test_check_ready_touches_neither_files_nor_hardware(tmp_path: Path) -> None:
    # It takes a document object, never a path: the CLI calls it before the stand opens.
    _check(_doc(_done_record("up")), Reader())
    assert list(tmp_path.iterdir()) == []


def test_check_ready_reports_the_first_unusable_read_in_the_order_of_reads() -> None:
    class Two(Base):
        name = provides = "two-reads"
        reads = ("first", "second")

    with pytest.raises(UpstreamError) as raised:
        _check(_doc(), Two())
    assert raised.value.slot == "first"


# --- context -----------------------------------------------------------------


def test_a_failing_progress_callback_is_dropped_and_the_run_goes_on(
    fake_microscope: Microscope,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[Progress] = []

    def broken(progress: Progress) -> None:
        calls.append(progress)
        raise ValueError("the terminal is gone")

    record = _run(Progressing(), fake_microscope, tmp_path, on_progress=broken)
    assert record.status is StageStatus.done
    assert calls == [Progress("progressing", 1, 3, "step")]  # called once, then dropped
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name.startswith("smc.plugins")
    ]
    assert warnings == [
        "the progress callback failed (ValueError: the terminal is gone); progress "
        "is not reported any more in this run"
    ]


def test_progress_reaches_the_callback_unvalidated(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    seen: list[Progress] = []
    _run(Progressing(), fake_microscope, tmp_path, on_progress=seen.append)
    assert seen == [
        Progress("progressing", 1, 3, "step"),
        Progress("progressing", 2, 3, "step"),
        Progress("progressing", 3, 3, "step"),
    ]


def test_a_keyboard_interrupt_in_the_progress_callback_passes_through(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    def interrupt(progress: Progress) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(Progressing(), fake_microscope, tmp_path, on_progress=interrupt)
    assert _stored(tmp_path, "progressing").status is StageStatus.stopped


def test_wait_s_returns_early_when_stop_is_set(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # Threads meet on an Event, and the join has a timeout (FM-44).
    plugin = Waits(seconds=30.0)
    stop = threading.Event()
    outcome: list[StageRecord] = []
    worker = threading.Thread(
        target=lambda: outcome.append(
            _run(plugin, fake_microscope, tmp_path, stop=stop)
        ),
        daemon=True,
    )
    worker.start()
    assert plugin.waiting.wait(timeout=10)
    stop.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert [record.status for record in outcome] == [StageStatus.stopped]


def test_a_stop_set_before_the_wait_ends_it_at_once(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    stop = threading.Event()
    stop.set()
    record = _run(Waits(seconds=30.0), fake_microscope, tmp_path, stop=stop)
    assert record.status is StageStatus.stopped


class _RecordingEvent(threading.Event):
    """An event whose waits are recorded and cost nothing, so ``wait_s`` can be timed."""

    def __init__(self, clock: SimpleNamespace) -> None:
        super().__init__()
        self.clock = clock
        self.timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        self.clock.now += timeout or 0.0
        return self.is_set()


def test_wait_s_waits_in_slices_of_at_most_a_tenth_of_a_second(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-83: one long wait is not interrupted by Ctrl-C on Windows.
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(api_mod, "time", SimpleNamespace(monotonic=lambda: clock.now))
    stop = _RecordingEvent(clock)
    record = _run(Waits(seconds=0.35), fake_microscope, tmp_path, stop=stop)
    assert record.status is StageStatus.done
    timeouts = [t for t in stop.timeouts if t is not None]
    assert len(timeouts) == len(stop.timeouts) >= 4
    assert timeouts[:3] == [0.1, 0.1, 0.1]
    assert max(timeouts) <= 0.1
    assert sum(timeouts) == pytest.approx(0.35)


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_wait_s_refuses_a_negative_or_non_finite_wait(
    fake_microscope: Microscope, tmp_path: Path, seconds: float
) -> None:
    problem = f"wait_s needs a finite number of seconds, 0 or more, got {seconds!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(problem)}$"):
        _run(Waits(seconds=seconds), fake_microscope, tmp_path)


def test_should_stop_follows_the_event(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    class Asks(Base):
        name = provides = "asks"

        def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
            self.seen["before"] = ctx.should_stop()
            stop.set()
            self.seen["after"] = ctx.should_stop()
            return Result()

    stop = threading.Event()
    plugin = Asks()
    _run(plugin, fake_microscope, tmp_path, stop=stop)
    assert plugin.seen == {"before": False, "after": True}


def test_the_final_write_failing_raises_with_the_done_record(
    fake_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-34: the stage ran; the caller can still show its result.
    real_replace, calls = files_mod.os.replace, []

    def replace_once(src: Any, dst: Any) -> None:
        calls.append(src)
        if len(calls) > 1:  # the `running` write went through
            raise PermissionError(13, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(files_mod.os, "replace", replace_once)
    monkeypatch.setattr(files_mod.time, "sleep", lambda seconds: None)
    with pytest.raises(RunDocumentError) as raised:
        _run(Echo(), fake_microscope, tmp_path, n=5)
    error = raised.value
    assert error.record is not None
    assert error.record.status is StageStatus.done
    assert error.record.result == {"assumed": [], "value": 5}
    assert error.kept is not None
    assert error.kept.exists()
    assert _stored(tmp_path, "echo").status is StageStatus.running


def test_a_failed_outcome_write_does_not_hide_the_original_error(
    fake_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_replace, calls = files_mod.os.replace, []

    def replace_once(src: Any, dst: Any) -> None:
        calls.append(src)
        if len(calls) > 1:
            raise PermissionError(13, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(files_mod.os, "replace", replace_once)
    monkeypatch.setattr(files_mod.time, "sleep", lambda seconds: None)
    with pytest.raises(HardwareError) as raised:
        _run(Boom(), fake_microscope, tmp_path)
    assert str(raised.value) == "the stage did not answer"  # not the write's error
    errors = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name.startswith("smc.plugins")
    ]
    assert len(errors) == 1
    assert errors[0].startswith(
        "boom: the outcome (failed) could not be written to the run document: "
    )
    assert "could not be replaced (Permission denied)" in errors[0]
    # The document still says `running`, which `unusable` reads as interrupted.
    assert _stored(tmp_path, "boom").status is StageStatus.running


def test_a_nan_in_an_unbounded_field_is_recorded_as_null_not_written_as_nan(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # `allow_inf_nan=False` bounds the declared floats; an `Any` field is not
    # bounded, and the JSON dump turns nan into null (measured): "unknown is None".
    class Loose(PluginResult):
        extra_data: dict[str, Any] = Field(default_factory=dict)

    class LoosePlugin(Base):
        name = provides = "loose"
        Result = Loose

        def body(self, microscope: Any, params: Params, ctx: RunContext) -> Loose:
            return Loose(extra_data={"x": float("nan"), "y": float("inf")})

    record = _run(LoosePlugin(), fake_microscope, tmp_path)
    assert record.result == {"assumed": [], "extra_data": {"x": None, "y": None}}
    assert "NaN" not in _path(tmp_path).read_text(encoding="utf-8")
    assert _stored(tmp_path, "loose") == record


def test_the_context_adds_no_logging_handler(
    fake_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-46: a command's logging is undone when its context closes; the plugin
    # layer adds none of its own.
    names = ("smc", "smc.plugins", "smc.plugins.echo", "smc.plugins.runner")
    before = {
        n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level)
        for n in names
    }
    _run(Echo(), fake_microscope, tmp_path)
    after = {
        n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level)
        for n in names
    }
    assert after == before


def test_a_plugin_is_a_plugin_for_the_registry_too() -> None:
    assert isinstance(Echo(), Plugin)


# --- no cycle, no exception kept (FM-92, FM-98) ------------------------------


def _outcome_of(action: Any) -> str:
    """The type name of what ``action`` raised; the error itself is not kept."""
    try:
        action()
    except BaseException as exc:
        return type(exc).__name__
    return ""


class _FreshBoom(Base):
    name = provides = "fresh-boom"

    def __init__(self, make: Any) -> None:
        super().__init__()
        self.make = make

    def body(self, microscope: Any, params: Params, ctx: RunContext) -> Result:
        raise self.make()


@pytest.mark.parametrize(
    ("make", "outcome"),
    [
        pytest.param(lambda: HardwareError("fault"), "HardwareError", id="hardware"),
        pytest.param(
            lambda: SafetyRefusedError("too far"), "SafetyRefusedError", id="refused"
        ),
        pytest.param(KeyboardInterrupt, "KeyboardInterrupt", id="ctrl-c"),
        pytest.param(StopRequested, "", id="stop"),
    ],
)
def test_a_run_that_ends_in_an_error_frees_the_stand_without_the_cyclic_collector(
    no_cyclic_gc: None,
    tmp_path: Path,
    make: Any,
    outcome: str,
) -> None:
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)  # a kept LogRecord would hold the error
    try:
        ended, watched = _run_on_a_fresh_stand(_FreshBoom(make), tmp_path)
        assert ended == outcome
        assert [ref() is not None for ref in watched] == [False, False]
    finally:
        logging.disable(previous)


def _run_on_a_fresh_stand(
    plugin: Base, tmp_path: Path
) -> tuple[str, list[weakref.ref[Any]]]:
    """Run ``plugin`` on its own stand; the outcome and weak references to the stand.

    The stand is a local of this frame, so it is released when the frame is.
    """
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    watched: list[weakref.ref[Any]] = [weakref.ref(microscope), weakref.ref(core)]
    return _outcome_of(lambda: _run(plugin, microscope, tmp_path)), watched
