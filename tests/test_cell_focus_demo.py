"""``focus-on-cells`` on the demo-sample stand: the sweep, its refusals and its failures (m3 §21.7, §21.9).

The demo-sample's camera renders the plate where the stage is, sharp at Z = 0
and blurred by ``|z| / 10`` pixels elsewhere, with fresh noise per snap
(§21.1): a stand-in for defocus. The doubles write out every method of the
capability they stand for, because ``Microscope.override`` refuses a
``__getattr__`` delegate (#380). Positions are compared with readbacks: the
demo rounds a command.
"""

from __future__ import annotations

import errno
import logging
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import tifffile

import smc.plugins.cell_focus.plugin as focus_plugin
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, FocusStep, Limits, ZStage
from smc.hardware.errors import MicroscopeHaltedError, SafetyRefusedError
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, load, run_plugin
from smc.plugins.cell_focus.plugin import FocusOnCellsParams, FocusOnCellsPlugin
from smc.plugins.slots import CellFocusResult
from smc.run import RunDocument, StageRecord, StageStatus
from smc.run.files import temporary_path

pytestmark = pytest.mark.demo

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
NAME = "focus-on-cells"


# ------------------------------------------------------------------ the doubles


class _Z:
    """The stand's focus drive, counting moves; ``fail`` raises at a move, ``off_um`` misreports one.

    Both dicts are keyed by the move's number, counted from 1.
    """

    def __init__(self, inner: ZStage, af: Autofocus | None = None) -> None:
        self._inner = inner
        self._af = af
        self.moves: list[float] = []
        self.engaged_at_move: list[bool] = []
        self.fail: dict[int, BaseException] = {}
        self.off_um: dict[int, float] = {}
        #: The soft limits this drive reports; ``None``: the stand's.
        self.limits: Limits | None = None
        #: Called with the move's number after each move.
        self.after_move: Callable[[int], None] | None = None

    def position_um(self) -> float:
        return self._inner.position_um()

    def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float:
        assert keep_autofocus is False, "the sweep moves Z with the autofocus off"
        self.moves.append(z_um)
        if self._af is not None:
            self.engaged_at_move.append(self._af.is_engaged())
        n = len(self.moves)
        if n in self.fail:
            raise self.fail[n]
        readback = self._inner.move_to_um(z_um, keep_autofocus=keep_autofocus)
        if self.after_move is not None:
            self.after_move(n)
        return readback + self.off_um.get(n, 0.0)

    def move_by_um(
        self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
    ) -> float:
        raise AssertionError("the sweep makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> Limits | None:
        return self.limits if self.limits is not None else self._inner.limits_um()


class _Autofocus:
    """An autofocus that records every call; ``locked`` is what it reads while engaged."""

    def __init__(self, *, engaged: bool, locked: bool) -> None:
        self.engaged = engaged
        self.locked = locked
        self.calls: list[str] = []
        self.engage_result: bool = True

    def is_engaged(self) -> bool:
        self.calls.append("is_engaged")
        return self.engaged

    def is_locked(self) -> bool:
        self.calls.append("is_locked")
        return self.engaged and self.locked

    def is_in_range(self) -> bool | None:
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        self.calls.append("engage")
        self.engaged = self.locked = self.engage_result
        return self.engage_result

    def disengage(self) -> None:
        self.calls.append("disengage")
        self.engaged = self.locked = False

    def offset_native(self) -> float:
        self.calls.append("offset_native")
        return 4242.0

    def set_offset_native(self, value: float) -> float:
        raise AssertionError("the sweep never moves the offset")

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        raise AssertionError("the sweep never moves the offset")

    def offset_unit(self) -> str:
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        return None

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        raise AssertionError("the sweep never focuses through the autofocus")


class _StuckAutofocus(_Autofocus):
    """An autofocus that still reads on after it was switched off."""

    def disengage(self) -> None:
        self.calls.append("disengage")


# ------------------------------------------------------------------ the stands


@pytest.fixture
def open_stand(mm_available: bool) -> Iterator[Callable[..., Microscope]]:
    """Opens ``demo-sample``; closes what it opened."""
    opened: list[Microscope] = []

    def make(*, dry_run: bool = False) -> Microscope:
        if not mm_available:
            pytest.skip("Micro-Manager demo adapters not installed")
        stand = Microscope.open(Profile.load(DEMO_SAMPLE), dry_run=dry_run)
        opened.append(stand)
        return stand

    yield make
    for stand in opened:
        stand.close()


def _run(stand: Microscope, run_doc: Path, **params: Any) -> StageRecord:
    return run_plugin(
        FocusOnCellsPlugin(), stand, FocusOnCellsParams(**params), run_doc
    )


def _result(record: StageRecord) -> CellFocusResult:
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    return CellFocusResult.model_validate(record.result)


def _refused(stand: Microscope, run_doc: Path, **params: Any) -> str:
    with pytest.raises(PluginRunError) as raised:
        _run(stand, run_doc, **params)
    stored = RunDocument.load(run_doc).stages[NAME]
    assert stored.status is StageStatus.failed
    return raised.value.reason


def _tiffs(run_doc: Path) -> list[Path]:
    return sorted((run_doc.parent / "cell-focus").rglob("*.tif"))


def _meta(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tiff:
        stored = tiff.shaped_metadata
    assert stored is not None
    return dict(stored[0])


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


# ---------------------------------------------------------------- the tool


def test_the_entry_point_loads_focus_on_cells() -> None:
    entry = load(NAME)

    assert entry.plugin.provides == "cell-focus"
    assert entry.plugin.requires == (ZStage, focus_plugin.Camera)
    assert entry.plugin.reads == ()
    assert entry.target == "smc.plugins.cell_focus.plugin:FocusOnCellsPlugin"


def test_a_sweep_from_20_um_finds_the_sharp_plane_within_one_step(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    z = stand.require(ZStage)
    start = z.move_to_um(20.0)

    result = _result(_run(stand, run_doc, range_um=25.0, step_um=5.0))

    sweep = result.sweep
    assert sweep.centre_um == start
    assert len(sweep.z_um) == 11
    assert sweep.z_um[0] == pytest.approx(-5.0, abs=0.01)
    assert sweep.outcome == "peak"
    assert sweep.best_z_um is not None
    # Measured 0.03 (§21.1); a sweep that kept its centre would end at 20.
    assert abs(sweep.best_z_um) <= 5.0
    assert result.z_um == pytest.approx(sweep.best_z_um, abs=1.25)
    assert z.position_um() == result.z_um
    # The stack is kept, one frame per plane in the sweep's order, and the best.
    assert result.folder.startswith("cell-focus/")
    assert result.files == tuple(f"{result.folder}/{k:03d}.tif" for k in range(11))
    assert result.best_file == f"{result.folder}/best.tif"
    assert len(_tiffs(run_doc)) == 12
    for k, (relative, readback, score) in enumerate(
        zip(result.files, sweep.z_um, sweep.scores, strict=True)
    ):
        meta = _meta(run_doc.parent / relative)
        assert meta["z_um"] == readback
        assert meta["z_command_um"] == pytest.approx(-5.0 + 5.0 * k)
        assert meta["score"] == score
        assert meta["score_name"] == "brenner"
    best = _meta(run_doc.parent / result.best_file)
    assert best["z_command_um"] == sweep.best_z_um
    # The demo's autofocus is off at open: not used, not touched.
    assert result.autofocus == "off"
    assert result.pfs_offset is None
    assert result.relocked is None
    assert result.assumed == ()


def test_a_sweep_past_the_soft_limit_is_refused_before_any_move(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    # The demo's DStage refuses a Z of 500 µm, so the plan's "Z at 990 under a
    # limit of 1000" is played 10 µm under a limit the double reports.
    stand = open_stand()
    start = stand.require(ZStage).move_to_um(20.0)
    z = _Z(stand.require(ZStage))
    z.limits = Limits(-1000.0, 30.0)
    stand.override(ZStage, z)

    reason = _refused(stand, run_doc, range_um=25.0, step_um=5.0)

    # §21.6's check 4 names the first plane past the limit (first_outside).
    assert reason == (
        f"the focus sweep around {start:g} µm would reach 35 µm, outside the "
        "soft limits [-1000, 30] µm"
    )
    assert z.moves == []
    assert z.position_um() == start
    assert not (run_doc.parent / "cell-focus").exists()


def test_a_range_above_the_z_jog_limit_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    stand.profile = stand.profile.model_copy(
        update={"safety": stand.profile.safety.model_copy(update={"max_z_jog_um": 4.0})}
    )
    z = _Z(stand.require(ZStage))
    stand.override(ZStage, z)

    reason = _refused(stand, run_doc, range_um=5.0)

    assert reason == (
        "range_um=5 µm is above the stand's Z jog limit of 4 µm ([safety] "
        "max_z_jog_um): a sweep reaches no farther than a Z jog may"
    )
    assert z.moves == []


def test_a_guard_trip_ends_the_sweep_and_keeps_the_frames(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    af = _Autofocus(engaged=True, locked=True)
    stand.override(Autofocus, af)
    z = _Z(stand.require(ZStage), af)
    z.fail[3] = SafetyRefusedError("Z 3.0 µm is outside the soft limits")
    stand.override(ZStage, z)

    reason = _refused(stand, run_doc)

    assert re.fullmatch(
        r"the focus sweep ended at plane 3 of 11 \(SafetyRefusedError: Z 3\.0 µm is "
        r"outside the soft limits\); the 2 frames taken are kept in "
        r"cell-focus/[0-9a-f]{12}/; the autofocus was left off; Z was left where "
        r"it stopped",
        reason,
    ), reason
    assert len(_tiffs(run_doc)) == 2
    assert "engage" not in af.calls
    assert af.engaged is False


def test_a_locked_autofocus_is_switched_off_for_the_sweep_and_relocked(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    af = stand.require(Autofocus)
    assert af.engage() is True
    offset = af.offset_native()
    z = _Z(stand.require(ZStage), af)
    stand.override(ZStage, z)

    result = _result(_run(stand, run_doc))

    assert z.engaged_at_move
    assert not any(z.engaged_at_move)
    assert af.is_locked() is True
    assert result.relocked is True
    assert result.autofocus == "locked"
    assert result.pfs_offset == offset
    assert result.assumed == ()


def test_relock_false_leaves_it_off_at_the_best_plane(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    af = _Autofocus(engaged=True, locked=True)
    stand.override(Autofocus, af)

    result = _result(_run(stand, run_doc, relock=False))

    assert result.sweep.outcome == "peak"
    assert "engage" not in af.calls
    assert af.engaged is False
    assert result.relocked is None
    assert result.assumed == ("the autofocus was left off, at the best plane",)


def test_a_relock_that_does_not_lock_is_said_and_not_a_failure(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stand = open_stand()
    af = _Autofocus(engaged=True, locked=True)
    af.engage_result = False
    stand.override(Autofocus, af)

    with caplog.at_level(logging.WARNING, logger=f"smc.plugins.{NAME}"):
        result = _result(_run(stand, run_doc))

    assert result.relocked is False
    assert result.assumed == ("the autofocus did not lock again: it is off",)
    assert "the autofocus did not lock again: it is off" in caplog.text


def test_an_autofocus_on_and_not_locked_is_refused(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    af = _Autofocus(engaged=True, locked=False)
    stand.override(Autofocus, af)
    z = _Z(stand.require(ZStage))
    stand.override(ZStage, z)

    reason = _refused(stand, run_doc)

    assert reason == (
        "the autofocus is on and not locked: lock it on the sample, or switch it "
        "off to sweep around the Z where it is"
    )
    assert z.moves == []
    assert "disengage" not in af.calls
    assert not (run_doc.parent / "cell-focus").exists()


def test_an_autofocus_that_stays_on_is_not_swept_under(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    af = _StuckAutofocus(engaged=True, locked=True)
    stand.override(Autofocus, af)
    z = _Z(stand.require(ZStage))
    stand.override(ZStage, z)

    reason = _refused(stand, run_doc)

    assert reason.startswith(
        "the focus sweep ended at the switch-off of the autofocus (AutofocusStillOn: "
        "the autofocus still reads on after it was switched off for the focus "
        "sweep: nothing was moved); the 0 frames taken are kept in cell-focus/"
    )
    # It reads on: the text does not say it was left off.
    assert "left off" not in reason
    assert z.moves == []


def test_a_halt_during_the_sweep_is_not_a_plugin_error(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    z = _Z(stand.require(ZStage))
    z.fail[2] = MicroscopeHaltedError()
    stand.override(ZStage, z)

    with pytest.raises(MicroscopeHaltedError):
        _run(stand, run_doc)

    assert len(_tiffs(run_doc)) == 1
    assert RunDocument.load(run_doc).stages[NAME].status is StageStatus.failed


def test_a_readback_off_its_command_fails_the_sweep(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    z = _Z(stand.require(ZStage))
    z.off_um[4] = 0.3  # step 1 µm: a quarter step is 0.25 µm
    stand.override(ZStage, z)

    reason = _refused(stand, run_doc)

    assert re.fullmatch(
        r"the focus sweep ended at plane 4 of 11 \(FocusDriveOffTarget: Z read "
        r"-?\d+\.\d\d µm after a move to -?\d+\.\d\d µm, more than 0\.25 µm away\); "
        r"the 3 frames taken are kept in cell-focus/[0-9a-f]{12}/; Z was left where "
        r"it stopped",
        reason,
    ), reason
    assert len(_tiffs(run_doc)) == 3


def test_a_readback_within_a_quarter_step_passes(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand()
    z = _Z(stand.require(ZStage))
    z.off_um[4] = 0.2
    stand.override(ZStage, z)

    result = _result(_run(stand, run_doc))

    assert len(result.files) == 11


def test_a_frame_that_cannot_be_written_ends_the_run_and_says_why(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stand = open_stand()
    calls = 0
    real = focus_plugin.write_atomically

    def full_disk(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(path, data, make_parents=make_parents)

    monkeypatch.setattr(focus_plugin, "write_atomically", full_disk)

    reason = _refused(stand, run_doc)

    assert re.fullmatch(
        r"the frame cell-focus/[0-9a-f]{12}/002\.tif could not be written \(No space "
        r"left on device\): free some space if the disk is full, or check the "
        r"network share, then run focus-on-cells again",
        reason,
    ), reason
    assert len(_tiffs(run_doc)) == 2


def test_a_frame_not_written_after_the_switch_off_says_the_autofocus_is_off(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # FM-88: the run ends with PFS off and Z off its plane; the operator who
    # frees the disk and runs it again must know to lock PFS first.
    stand = open_stand()
    af = _Autofocus(engaged=True, locked=True)
    stand.override(Autofocus, af)
    calls = 0
    real = focus_plugin.write_atomically

    def full_disk(path: Path, data: bytes, *, make_parents: bool = True) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(path, data, make_parents=make_parents)

    monkeypatch.setattr(focus_plugin, "write_atomically", full_disk)

    with caplog.at_level(logging.WARNING, logger=f"smc.plugins.{NAME}"):
        reason = _refused(stand, run_doc)

    assert reason.endswith(
        "then run focus-on-cells again; the autofocus was left off; Z was left "
        "where it stopped"
    )
    assert af.engaged is False
    assert "engage" not in af.calls
    assert _left_off_warnings(caplog) == 1


def _left_off_warnings(caplog: pytest.LogCaptureFixture) -> int:
    return sum(
        r.name == f"smc.plugins.{NAME}"
        and r.levelno == logging.WARNING
        and "the autofocus was left off and Z was left where it stopped"
        in r.getMessage()
        for r in caplog.records
    )


def test_a_stop_during_the_sweep_ends_it_at_the_next_plane(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stand = open_stand()
    af = _Autofocus(engaged=True, locked=True)
    stand.override(Autofocus, af)
    z = _Z(stand.require(ZStage), af)
    stand.override(ZStage, z)
    stop = threading.Event()
    z.after_move = lambda n: stop.set() if n == 3 else None

    with caplog.at_level(logging.WARNING, logger=f"smc.plugins.{NAME}"):
        record = run_plugin(
            FocusOnCellsPlugin(), stand, FocusOnCellsParams(), run_doc, stop=stop
        )

    # A stop is an outcome, not a failure (FM-64), honoured before the 4th move.
    assert record.status is StageStatus.stopped
    assert len(z.moves) == 3
    assert len(_tiffs(run_doc)) == 3
    # Nothing is put back after a stop (FM-88), and that is said.
    assert "engage" not in af.calls
    assert af.engaged is False
    assert _left_off_warnings(caplog) == 1


def test_a_sweep_without_a_peak_moves_back_to_the_centre(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    # From 20 µm, ±5 µm never reaches the sharp plane at 0: the best is the
    # sweep's lowest plane, an edge, and the centre is kept (D4).
    stand = open_stand()
    start = stand.require(ZStage).move_to_um(20.0)

    result = _result(_run(stand, run_doc))

    assert result.sweep.outcome == "at-edge"
    assert result.sweep.best_z_um is None
    assert result.best_file is None
    assert len(result.files) == 11
    assert not (run_doc.parent / result.folder / "best.tif").exists()
    # Back at the centre, not left at the sweep's last plane (25 µm).
    assert result.z_um == pytest.approx(start, abs=0.25)


def _longest_write_chars(run_doc: Path) -> int:
    """The longest path the tool writes, as Windows counts it: ``best.tif``'s temporary file."""
    best = run_doc.parent / "cell-focus" / ("0" * 12) / "best.tif"
    return len(str(temporary_path(best)))


def test_a_frame_path_too_long_for_windows_is_refused_before_any_move(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stand = open_stand()
    z = _Z(stand.require(ZStage))
    stand.override(ZStage, z)
    n = _longest_write_chars(run_doc)
    monkeypatch.setattr(focus_plugin, "ON_WINDOWS", True)
    monkeypatch.setattr(focus_plugin, "MAX_PATH", n)

    reason = _refused(stand, run_doc)

    assert reason == (
        f"a frame's path would reach {n} characters while it is written, and "
        f"Windows takes {n - 1} at most unless long paths are enabled (not "
        "assumed): move the run document to a shorter folder, then run "
        "focus-on-cells again"
    )
    assert z.moves == []
    assert not (run_doc.parent / "cell-focus").exists()


def test_a_frame_path_one_character_under_the_limit_is_written(
    open_stand: Callable[..., Microscope],
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stand = open_stand()
    monkeypatch.setattr(focus_plugin, "ON_WINDOWS", True)
    monkeypatch.setattr(focus_plugin, "MAX_PATH", _longest_write_chars(run_doc) + 1)

    assert len(_result(_run(stand, run_doc)).files) == 11


def test_a_dry_run_moves_nothing_and_writes_nothing(
    open_stand: Callable[..., Microscope], run_doc: Path
) -> None:
    stand = open_stand(dry_run=True)
    z = stand.require(ZStage)
    start = z.position_um()

    result = _result(_run(stand, run_doc, range_um=2.0, step_um=0.5))

    assert result.sweep.z_um == tuple(start + 0.5 * i for i in range(-4, 5))
    assert result.sweep.scores == (0.0,) * 9
    assert result.sweep.outcome == "flat"
    assert result.files == ()
    assert result.best_file is None
    assert result.relocked is None
    assert result.assumed == ("dry run: 9 planes walked, no frame taken",)
    assert z.position_um() == start
    assert not (run_doc.parent / "cell-focus").exists()


def test_params_refuse_a_step_above_the_range_and_too_many_planes() -> None:
    with pytest.raises(ValueError, match=r"step_um=2 is above range_um=1"):
        FocusOnCellsParams(range_um=1.0, step_um=2.0)
    with pytest.raises(ValueError, match=r"range_um=50 with step_um=0\.1 makes 1001"):
        FocusOnCellsParams(range_um=50.0, step_um=0.1)
    FocusOnCellsParams(range_um=5.0, step_um=0.1)  # 101 planes, the bound


@pytest.mark.parametrize(
    ("given", "accepted"),
    [
        ({"range_um": 50.0}, True),
        ({"range_um": 50.1}, False),
        ({"step_um": 0.1, "range_um": 5.0}, True),
        ({"step_um": 0.09}, False),
        ({"step_um": 10.0, "range_um": 10.0}, True),
        ({"step_um": 10.1, "range_um": 20.0}, False),
    ],
)
def test_params_hold_the_bounds_from_both_sides(
    given: dict[str, float], accepted: bool
) -> None:
    if accepted:
        FocusOnCellsParams(**given)
    else:
        with pytest.raises(ValueError, match=r"should be (less|greater) than or equal"):
            FocusOnCellsParams(**given)
