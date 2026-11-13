"""``goto-well`` through ``run_plugin``: the registered plate in, one absolute move out (m3 §5, #19).

The plate slot is written by the test, as an earlier ``plate-registration`` run
would have left it: A1 at (1000, 2000) µm and the plate turned by 3°. Every
expected target is written here from the cosine and the sine of the rotation, the
9 mm pitch and the 3.2 mm half-size of a 96-well plate, never from ``wall_position``
or from the result's own fields (FM-124). The demo stage rounds a command to
0.01 µm and travels, so a position is read after the move has returned (FM-85)
and compared within 0.02 µm.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pymmcore_plus import DeviceType
from typer.testing import CliRunner

from objective_offsets_stand import (
    FORTY,
    MEASURED,
    NOT_MEASURED,
    SHIFT_TEN_TO_FORTY,
    TEN,
    TWENTY,
    opened,
    profile_with_offsets,
)
from recording_autofocus import RecordingAutofocus
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    Autofocus,
    ObjectiveOffset,
    ObjectiveTurret,
    XYStage,
    ZStage,
)
from smc.hardware.errors import (
    CoreError,
    DeviceTimeoutError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
)
from smc.hardware.profile import (
    AutofocusSection,
    HolderSection,
    MicroscopeSection,
    Profile,
    SafetySection,
)
from smc.plugins import (
    ParamsError,
    PluginRunError,
    UpstreamError,
    load,
    parse_params,
    run_plugin,
)
from smc.plugins.goto_well import GotoWellParams, GotoWellPlugin, GotoWellResult
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore, FakeDevice

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

pytestmark = pytest.mark.demo

A1 = (1000.0, 2000.0)
ROTATION_DEG = 3.0
PITCH_UM = 9000.0
HALF_UM = 3200.0
SEED_LINE = "one reference well (A1): rotation 0° assumed from the declared orientation (not-turned), not measured"
DRY_LINE = "dry run: the stage did not move; x_um and y_um are the target"
DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"


def _plate(
    *,
    method: str = "references",
    assumed: tuple[str, ...] = (),
    objective: str | None = TEN,
) -> PlateResult:
    """A registered plate; by default with the 10x, the objective the demo stand has in place."""
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=A1,
        rotation_deg=ROTATION_DEG,
        method=method,  # type: ignore[arg-type]
        residual_um=5.0 if method == "references" else None,
        scale_error=0.001 if method == "references" else None,
        n_refs=3 if method == "references" else 1,
        assumed=assumed,
        objective=objective,
    )


def _seed(
    tmp_path: Path,
    plate: PlateResult | None = None,
    *,
    dry_run: bool = False,
    profile: str = "demo",
) -> None:
    """The ``plate`` slot as an earlier ``plate-registration`` run left it, ``done``."""
    then = datetime(2026, 10, 28, 9, 30, tzinfo=timezone.utc)
    write_stage(
        tmp_path / "run.json",
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=(plate if plate is not None else _plate()).model_dump(mode="json"),
        ),
    )


def _at(well: str, ox: float, oy: float) -> tuple[float, float]:
    """The point at offset ``(ox, oy)`` from a well's centre, in the plate's axes, on the stage.

    ``well`` is a letter and a number on a 96-well plate: A1 plus (column,
    -row) pitches, turned by the rotation, as M1 §8 has it.
    """
    row, col = ord(well[0]) - ord("A"), int(well[1:]) - 1
    angle = math.radians(ROTATION_DEG)
    cos, sin = math.cos(angle), math.sin(angle)
    px, py = col * PITCH_UM + ox, -row * PITCH_UM + oy
    return (A1[0] + cos * px - sin * py, A1[1] + sin * px + cos * py)


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    **params: object,
) -> StageRecord:
    return run_plugin(
        GotoWellPlugin(),
        microscope,
        GotoWellParams(**params),  # type: ignore[arg-type]
        tmp_path / "run.json",
        stop=stop,
    )


def _result(record: StageRecord) -> GotoWellResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return GotoWellResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["goto-well"]


def _where(microscope: Microscope) -> tuple[float, float]:
    xy = microscope.require(XYStage).position_um()
    return (xy.x_um, xy.y_um)


# ---------------------------------------------------------------- the move


@pytest.mark.parametrize(
    ("well", "side", "offset"),
    [
        ("B3", "+Y", (0.0, HALF_UM)),
        ("C5", "+X", (HALF_UM, 0.0)),
        ("B2", "-Y", (0.0, -HALF_UM)),
        ("B2", "-X", (-HALF_UM, 0.0)),
    ],
)
def test_goto_well_drives_to_the_wall_point_on_a_rotated_plate(
    demo_microscope: Microscope,
    tmp_path: Path,
    well: str,
    side: str,
    offset: tuple[float, float],
) -> None:
    _seed(tmp_path)
    expected = _at(well, *offset)
    result = _result(_run(demo_microscope, tmp_path, well=well, side=side))
    assert result.well == well
    assert result.side == side
    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert (result.x_um, result.y_um) == pytest.approx(expected, abs=0.02)
    assert _where(demo_microscope) == pytest.approx(expected, abs=0.02)
    assert _stored(tmp_path).status is StageStatus.done
    assert _stored(tmp_path).provides == "goto-well"


def test_goto_well_drives_to_the_centre_by_default(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    expected = _at("B2", 0.0, 0.0)
    result = _result(_run(demo_microscope, tmp_path, well="b2"))
    assert result.well == "B2"
    assert result.side == "centre"
    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert _where(demo_microscope) == pytest.approx(expected, abs=0.02)
    assert result.assumed == ()


def test_the_readback_is_where_the_stage_is_not_the_target(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The demo stage rounds a command to 0.01 µm: the result reports the stage, and keeps the target beside it."""
    _seed(tmp_path)
    result = _result(_run(demo_microscope, tmp_path, well="A2"))
    assert (result.x_um, result.y_um) == _where(demo_microscope)
    assert (result.x_um, result.y_um) != (result.target_x_um, result.target_y_um)


# --------------------------------------------------------------- refusals


def test_goto_well_needs_a_plate(demo_microscope: Microscope, tmp_path: Path) -> None:
    before = _where(demo_microscope)
    with pytest.raises(
        UpstreamError, match="goto-well reads plate, but the run document has no plate"
    ) as info:
        _run(demo_microscope, tmp_path, well="A1")
    assert _where(demo_microscope) == before
    # Without `plate=`, the refusal also says how to drive to the estimate (m3 §5.5 D3).
    assert str(info.value) == (
        "goto-well reads plate, but the run document has no plate; smc run --list "
        "shows which plugin provides it; or give plate=<a useq plate name> to drive "
        "to the estimate of the profile's [holder]"
    )


def test_an_unknown_well_fails_and_does_not_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    before = _where(demo_microscope)
    with pytest.raises(PluginRunError) as info:
        _run(demo_microscope, tmp_path, well="Z99")
    assert str(info.value) == (
        "goto-well: well 'Z99' is not on the 96-well plate; "
        "give a well of the 96-well plate"
    )
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    assert _where(demo_microscope) == before


def test_a_target_outside_the_soft_limits_is_refused_and_does_not_move(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    profile = Profile.demo()
    limited = profile.model_copy(
        update={
            "safety": profile.safety.model_copy(
                update={"xy_soft_limits_um": ((-1000.0, 1000.0), (-500.0, 500.0))}
            )
        }
    )
    microscope = Microscope.open(limited, dry_run=False)
    try:
        _seed(tmp_path)
        before = _where(microscope)
        with pytest.raises(SafetyRefusedError) as info:
            _run(microscope, tmp_path, well="A1")  # y = 2000 µm, past the +500
        assert info.value.how_to_force == ""
        assert _stored(tmp_path).status is StageStatus.failed
        assert _stored(tmp_path).result is None
        assert _where(microscope) == before
    finally:
        microscope.close()


def test_a_halt_reaches_the_caller_and_is_not_a_refusal(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """FM-64: the emergency stop is checked by the layer before the move; the plugin does not turn it into a refusal."""
    _seed(tmp_path)
    before = _where(demo_microscope)
    demo_microscope.stop()
    try:
        with pytest.raises(MicroscopeHaltedError):
            _run(demo_microscope, tmp_path, well="A2")
    finally:
        demo_microscope.resume()
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    assert _where(demo_microscope) == before


@pytest.mark.parametrize(
    "fault",
    [
        MicroscopeHaltedError(),
        DeviceTimeoutError("xy_stage did not arrive within 30 s"),
        CoreError("the core lost the stage"),
    ],
    ids=["halt", "timeout", "core-error"],
)
def test_a_fault_of_the_move_reaches_the_runner_unchanged(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    """Nothing is caught around the move: not a halt, not a timeout, not a core error (FM-64)."""
    _seed(tmp_path)

    def fails(x_um: float, y_um: float) -> object:
        raise fault

    monkeypatch.setattr(demo_microscope.require(XYStage), "move_to_um", fails)
    with pytest.raises(type(fault)) as info:
        _run(demo_microscope, tmp_path, well="A2")
    assert info.value is fault
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None


def test_goto_well_sends_one_move_and_reads_no_position_before_it(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan: one absolute move with the target the result reports, and no read of where the stage is first."""
    _seed(tmp_path)
    stage = demo_microscope.require(XYStage)
    calls: list[tuple[object, ...]] = []
    real_move, real_position = stage.move_to_um, stage.position_um

    def move(x_um: float, y_um: float) -> object:
        calls.append(("move_to_um", x_um, y_um))
        return real_move(x_um, y_um)

    def position() -> object:
        calls.append(("position_um",))
        return real_position()

    monkeypatch.setattr(stage, "move_to_um", move)
    monkeypatch.setattr(stage, "position_um", position)
    result = _result(_run(demo_microscope, tmp_path, well="B2", side="-X"))
    assert calls == [("move_to_um", result.target_x_um, result.target_y_um)]


def test_the_entry_point_loads_goto_well() -> None:
    entry = load("goto-well")

    assert entry.plugin.provides == "goto-well"
    assert entry.plugin.requires == (XYStage,)
    # Optional since #462: without a usable plate it drives to the holder's estimate.
    assert entry.plugin.reads == ()
    assert entry.plugin.reads_optional == ("plate",)
    assert entry.target == "smc.plugins.goto_well:GotoWellPlugin"


def test_a_stop_before_the_move_sends_nothing(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    before = _where(demo_microscope)
    stop = threading.Event()
    stop.set()
    record = _run(demo_microscope, tmp_path, well="A2", stop=stop)
    assert record.status is StageStatus.stopped
    assert record.result is None
    assert _where(demo_microscope) == before


# ------------------------------------------------------- what it says


def test_a_dry_run_says_the_position_is_the_target(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    """The plate is a real record (a dry run may read one); the stage is not moved and the result says so."""
    _seed(tmp_path, dry_run=False)
    before = _where(demo_microscope_dry)
    expected = _at("C5", HALF_UM, 0.0)
    record = _run(demo_microscope_dry, tmp_path, well="C5", side="+X")
    assert record.dry_run is True
    result = _result(record)
    assert result.assumed[-1] == DRY_LINE
    assert (result.x_um, result.y_um) == pytest.approx(expected, abs=0.02)
    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert _where(demo_microscope_dry) == before


def test_a_real_run_does_not_say_it_was_a_dry_run(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    assert DRY_LINE not in _result(_run(demo_microscope, tmp_path, well="A2")).assumed


def test_a_dry_plate_cannot_drive_a_real_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A dry run of plate-registration is a record of a computation nobody checked on this stand: a real run refuses it."""
    _seed(tmp_path, dry_run=True)
    before = _where(demo_microscope)
    with pytest.raises(
        UpstreamError,
        match="plate-registration was a dry run; run it without --dry-run",
    ):
        _run(demo_microscope, tmp_path, well="A2")
    assert _where(demo_microscope) == before


def test_the_plate_s_assumptions_are_carried(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, _plate(method="seed", assumed=(SEED_LINE,)))
    result = _result(_run(demo_microscope, tmp_path, well="A2"))
    assert result.assumed == (f"plate: {SEED_LINE}",)


def test_the_plate_s_assumptions_come_before_the_dry_run_line(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, _plate(method="seed", assumed=(SEED_LINE,)))
    result = _result(_run(demo_microscope_dry, tmp_path, well="A2"))
    assert result.assumed == (f"plate: {SEED_LINE}", DRY_LINE)


# ------------------------------------- the plate in the objective's frame (#340)


class _EmptyTurret:
    """A turret at an empty position: the protocol lists them, and their label is ``""`` (FM-125)."""

    def labels(self) -> list[str]:
        return ["", TEN]

    def current(self) -> str:
        return ""

    def select(self, label: str, *, confirm: bool = False) -> str:
        raise AssertionError("goto-well never turns the turret")

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0

    def offset(self, label: str) -> ObjectiveOffset:
        return ObjectiveOffset(label, None, None, None, "")


@pytest.fixture
def stand_with_offsets(mm_available: bool) -> Iterator[Microscope]:
    """The demo devices on a profile that measures the 10x and the 40x from the 20x."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    with opened(profile_with_offsets()) as microscope:
        yield microscope


def test_goto_well_at_40x_on_a_plate_registered_at_10x_lands_shifted(
    stand_with_offsets: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(tmp_path, _plate(objective=TEN))
    stand_with_offsets.require(ObjectiveTurret).select(FORTY, confirm=True)
    # The H12 target of the 10x registration, and the shift written from the
    # profile's numbers: 40x (-32.5, 9.75) minus 10x (5.0, -2.5).
    base = _at("H12", 0.0, 0.0)
    expected = (base[0] - 37.5, base[1] + 12.25)

    with caplog.at_level(logging.INFO, logger="smc"):
        result = _result(_run(stand_with_offsets, tmp_path, well="H12"))

    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert (result.x_um, result.y_um) == pytest.approx(expected, abs=0.02)
    assert _where(stand_with_offsets) == pytest.approx(expected, abs=0.02)
    assert (result.objective, result.plate_objective) == (FORTY, TEN)
    assert result.shift_um == SHIFT_TEN_TO_FORTY
    assert result.assumed == ()  # the offset is measured: nothing to say
    # The shift and both offsets' provenance are logged at INFO, and nothing warns.
    assert (
        f"the plate registered with {TEN} is moved into the frame of {FORTY} by "
        f"(-37.50, +12.25) µm; offsets measured: {TEN} {MEASURED}, {FORTY} {MEASURED}"
    ) in [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize(
    ("well", "side", "offset"),
    [
        ("B3", "+Y", (0.0, HALF_UM)),
        ("C5", "+X", (HALF_UM, 0.0)),
        ("B2", "-Y", (0.0, -HALF_UM)),
        ("B2", "-X", (-HALF_UM, 0.0)),
    ],
)
def test_goto_well_drives_to_the_shifted_wall_point_at_40x_on_a_plate_registered_at_10x(
    stand_with_offsets: Microscope,
    tmp_path: Path,
    well: str,
    side: str,
    offset: tuple[float, float],
) -> None:
    # The wall of a well is as far from the 40x's A1 as the centre is: the plate is
    # translated whole, so the wall point moves by the same (-37.5, +12.25) µm.
    _seed(tmp_path, _plate(objective=TEN))
    stand_with_offsets.require(ObjectiveTurret).select(FORTY, confirm=True)
    base = _at(well, *offset)
    expected = (base[0] - 37.5, base[1] + 12.25)

    result = _result(_run(stand_with_offsets, tmp_path, well=well, side=side))

    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert (result.x_um, result.y_um) == pytest.approx(expected, abs=0.02)
    assert result.shift_um == SHIFT_TEN_TO_FORTY


def test_goto_well_with_the_registration_objective_is_unshifted(
    stand_with_offsets: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(tmp_path, _plate(objective=TEN))  # the 10x is in place
    expected = _at("B2", 0.0, 0.0)

    with caplog.at_level(logging.INFO, logger="smc"):
        result = _result(_run(stand_with_offsets, tmp_path, well="B2"))

    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert (result.objective, result.plate_objective) == (TEN, TEN)
    assert result.shift_um == (0.0, 0.0)
    assert result.assumed == ()
    assert not any("is moved into the frame" in r.getMessage() for r in caplog.records)


def test_goto_well_with_an_unmeasured_offset_moves_unshifted_and_says_so(
    mm_available: bool, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    line = NOT_MEASURED.format(origin=TEN, target=FORTY)
    with opened(profile_with_offsets(forty=False)) as stand:
        _seed(tmp_path, _plate(objective=TEN))
        stand.require(ObjectiveTurret).select(FORTY, confirm=True)
        expected = _at("B2", 0.0, 0.0)

        with caplog.at_level(logging.INFO, logger="smc"):
            result = _result(_run(stand, tmp_path, well="B2"))

        # Unknown is not zero: the registration's own position, said so twice.
        assert (result.target_x_um, result.target_y_um) == pytest.approx(
            expected, abs=1e-6
        )
        assert (result.objective, result.plate_objective) == (FORTY, TEN)
        assert result.shift_um is None
        assert result.assumed == (line,)
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert warnings == [line]


def test_the_plate_s_assumptions_come_before_the_frame_s_line(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    line = NOT_MEASURED.format(origin=TEN, target=FORTY)
    with opened(profile_with_offsets(forty=False)) as stand:
        _seed(tmp_path, _plate(method="seed", assumed=(SEED_LINE,), objective=TEN))
        stand.require(ObjectiveTurret).select(FORTY, confirm=True)

        result = _result(_run(stand, tmp_path, well="B2"))

    assert result.assumed == (f"plate: {SEED_LINE}", line)


def test_the_frame_s_line_comes_before_the_dry_run_line(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    # The plate, then the frame (a plate that records no objective), then the dry run.
    _seed(tmp_path, _plate(method="seed", assumed=(SEED_LINE,), objective=None))

    result = _result(_run(demo_microscope_dry, tmp_path, well="A2"))

    assert result.assumed == (
        f"plate: {SEED_LINE}",
        "the plate does not record the objective it was registered with: no offset "
        f"between objectives is applied for {TEN}",
        DRY_LINE,
    )


def test_the_log_names_the_reference_as_the_reference(
    stand_with_offsets: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A plate registered with the reference objective has no entry to quote: its offset
    # is zero by definition, and the log says that and not an empty date.
    _seed(tmp_path, _plate(objective=TWENTY))
    stand_with_offsets.require(ObjectiveTurret).select(FORTY, confirm=True)
    base = _at("B2", 0.0, 0.0)
    expected = (base[0] - 32.5, base[1] + 9.75)

    with caplog.at_level(logging.INFO, logger="smc"):
        result = _result(_run(stand_with_offsets, tmp_path, well="B2"))

    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert result.shift_um == (-32.5, 9.75)
    assert (
        f"the plate registered with {TWENTY} is moved into the frame of {FORTY} by "
        f"(-32.50, +9.75) µm; offsets measured: {TWENTY} (the reference), "
        f"{FORTY} {MEASURED}"
    ) in [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


def test_goto_well_on_a_plate_that_records_no_objective_moves_unshifted_and_says_so(
    stand_with_offsets: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, _plate(objective=None))
    expected = _at("B2", 0.0, 0.0)

    result = _result(_run(stand_with_offsets, tmp_path, well="B2"))

    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert (result.objective, result.plate_objective, result.shift_um) == (
        TEN,
        None,
        None,
    )
    assert result.assumed == (
        "the plate does not record the objective it was registered with: no offset "
        f"between objectives is applied for {TEN}",
    )


def test_goto_well_refuses_a_turret_at_an_empty_position_before_any_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    demo_microscope.override(ObjectiveTurret, _EmptyTurret())
    before = _where(demo_microscope)

    with pytest.raises(PluginRunError) as info:
        _run(demo_microscope, tmp_path, well="B2")

    assert str(info.value) == (
        "goto-well: no objective is in place: the turret is at an empty position; "
        "turn it to an objective, or give objective to plan for one"
    )
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(demo_microscope) == before


def test_a_goto_well_result_written_before_the_objective_loads_as_unknown() -> None:
    older = {
        "well": "B2",
        "side": "centre",
        "target_x_um": 1.0,
        "target_y_um": 2.0,
        "x_um": 1.0,
        "y_um": 2.0,
        "assumed": [],
    }
    result = GotoWellResult.model_validate(older)
    assert (result.objective, result.plate_objective, result.shift_um) == (
        None,
        None,
        None,
    )


# --------------------------------------------------------------- from text


def test_the_parameters_parse_from_text() -> None:
    params = parse_params(GotoWellParams, ["well=H12", "side=+X"])
    assert (params.well, params.side) == ("H12", "+X")
    assert parse_params(GotoWellParams, ["well=A1", "side=-Y"]).side == "-Y"
    assert parse_params(GotoWellParams, ["well=A1"]).side == "centre"


@pytest.mark.parametrize("word", ["left", "right", "top", "bottom"])
def test_goto_well_refuses_a_side_on_the_image_with_the_reason(word: str) -> None:
    # The same word in two frames was FM-189: `left` is refused and says why, on the
    # model and through the command line's parameters, and nothing is moved.
    reason = (
        f"side {word!r} is a side on the image: a wall is named by a stage "
        "direction, -X, +X, -Y or +Y"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        GotoWellParams(well="A1", side=word)  # type: ignore[arg-type]
    with pytest.raises(ParamsError, match=re.escape(reason)):
        parse_params(GotoWellParams, ["well=A1", f"side={word}"])


def test_goto_well_refuses_a_word_that_is_no_side() -> None:
    # Not an image word: the Literal's own error stands, and does not claim that
    # `middle` is a side on the image.
    with pytest.raises(ValueError, match="side") as raised:
        GotoWellParams(well="A1", side="middle")  # type: ignore[arg-type]
    assert "Input should be 'centre', '+X', '-X', '+Y' or '-Y'" in str(raised.value)
    assert "side on the image" not in str(raised.value)


# ------------------------------------ before registration, the holder (#462)

#: Anisotropic and away from the origin (FM-73, FM-203). A 96-well grid's centre is
#: (49 500, -31 500) µm from A1, so A1 lies at the holder minus that, not turned,
#: and at the holder plus that, turned 180°: written here, not from holder_estimate.
HOLDER_UM = (400.0, -325.0)
A1_ESTIMATED = (-49_100.0, 31_175.0)
A1_ESTIMATED_TURNED = (49_900.0, -31_825.0)
NO_PLATE = (
    "the run document has no plate; smc run --list shows which plugin provides it"
)
ESTIMATE_LINE = (
    "target estimated from the stand's holder, not registered: the 96-well centred "
    "at stage (400, -325) µm, not turned"
)
NO_OFFSET_LINE = "no objective offset applied: the holder's estimate names no objective"


def _holder_profile() -> Profile:
    return Profile.demo().model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=HOLDER_UM,
                measured="a test's holder, anisotropic and off the origin",
            )
        }
    )


@pytest.fixture
def holder_stand(mm_available: bool) -> Iterator[Microscope]:
    """The demo devices on a profile whose ``[holder]`` puts a plate at (400, -325) µm."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    microscope = Microscope.open(_holder_profile(), dry_run=False)
    try:
        yield microscope
    finally:
        microscope.close()


def test_goto_well_without_a_plate_drives_to_the_holders_estimate(
    holder_stand: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="smc"):
        record = _run(holder_stand, tmp_path, well="A1", plate="96-well")
    result = _result(record)
    assert result.from_holder is True
    assert (result.well, result.side) == ("A1", "centre")
    assert (result.target_x_um, result.target_y_um) == pytest.approx(
        A1_ESTIMATED, abs=0.01
    )
    assert (result.x_um, result.y_um) == pytest.approx(A1_ESTIMATED, abs=0.02)
    assert _where(holder_stand) == pytest.approx(A1_ESTIMATED, abs=0.02)
    assert result.objective == holder_stand.require(ObjectiveTurret).current()
    assert (result.plate_objective, result.shift_um) == (None, None)
    assert result.assumed == (
        ESTIMATE_LINE,
        f"plate not read: {NO_PLATE}",
        NO_OFFSET_LINE,
    )
    assert record.reads == {"plate": None}
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "goto-well: driving to A1 of the 96-well estimated from the stand's holder, "
        "not registered"
    ]


def test_goto_well_on_a_turned_estimate_drives_to_the_turned_well(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    result = _result(
        _run(
            holder_stand,
            tmp_path,
            well="A1",
            plate="96-well",
            orientation="turned-180",
        )
    )
    assert (result.target_x_um, result.target_y_um) == pytest.approx(
        A1_ESTIMATED_TURNED, abs=0.01
    )
    assert _where(holder_stand) == pytest.approx(A1_ESTIMATED_TURNED, abs=0.02)
    assert result.assumed[0] == (
        "target estimated from the stand's holder, not registered: the 96-well "
        "centred at stage (400, -325) µm, turned 180°"
    )


def test_goto_well_without_a_holder_names_both_missing_and_moves_nothing(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    before = _where(demo_microscope)
    with pytest.raises(UpstreamError) as info:
        _run(demo_microscope, tmp_path, well="A1", plate="96-well")
    assert str(info.value) == (
        f"goto-well reads plate, but {NO_PLATE}; and this stand's profile has no "
        "[holder] plate_centre_xy_um to estimate the plate from"
    )
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(demo_microscope) == before


def test_goto_well_refuses_a_wall_on_the_estimate_before_any_move(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    before = _where(holder_stand)
    with pytest.raises(PluginRunError) as info:
        _run(holder_stand, tmp_path, well="B2", side="+X", plate="96-well")
    assert str(info.value) == (
        "goto-well: a wall is reached through a registered plate: the holder's "
        "estimate can be a few hundred µm off; go to the centre, and register the plate"
    )
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(holder_stand) == before


@pytest.mark.parametrize(
    ("plate", "well", "problem"),
    [
        pytest.param(
            "bogus",
            "A1",
            "Unknown plate name 'bogus'. Use `useq.register_well_plates` to add new "
            "plate definitions; give a useq plate name (96-well, greiner-24-well)",
            id="unknown-plate",
        ),
        pytest.param(
            "96-WELL",
            "A1",
            "Unknown plate name '96-WELL'. Use `useq.register_well_plates` to add new "
            "plate definitions; give a useq plate name (96-well, greiner-24-well)",
            id="names-are-exact",
        ),
        pytest.param(
            "96-well",
            "Z99",
            "well 'Z99' is not on the 96-well plate; give a well of the 96-well plate",
            id="unknown-well",
        ),
    ],
)
def test_goto_well_refuses_an_unknown_plate_on_the_estimate_before_any_move(
    holder_stand: Microscope, tmp_path: Path, plate: str, well: str, problem: str
) -> None:
    before = _where(holder_stand)
    with pytest.raises(PluginRunError) as info:
        _run(holder_stand, tmp_path, well=well, plate=plate)
    assert str(info.value) == f"goto-well: {problem}"
    # Raised outside the handler: useq's error is not chained onto it (FM-98).
    assert info.value.__context__ is None
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(holder_stand) == before


def test_goto_well_refuses_an_empty_turret_on_the_estimate_before_any_move(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    holder_stand.override(ObjectiveTurret, _EmptyTurret())
    before = _where(holder_stand)
    with pytest.raises(PluginRunError, match="no objective is in place"):
        _run(holder_stand, tmp_path, well="A1", plate="96-well")
    assert _where(holder_stand) == before


def test_a_stop_before_the_estimates_move_sends_nothing(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    before = _where(holder_stand)
    stop = threading.Event()
    stop.set()
    record = _run(holder_stand, tmp_path, well="A1", plate="96-well", stop=stop)
    assert record.status is StageStatus.stopped
    assert record.result is None
    assert _where(holder_stand) == before


def test_a_halt_reaches_the_caller_on_the_estimate_too(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    before = _where(holder_stand)
    holder_stand.stop()
    try:
        with pytest.raises(MicroscopeHaltedError):
            _run(holder_stand, tmp_path, well="A1", plate="96-well")
    finally:
        holder_stand.resume()
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(holder_stand) == before


@pytest.mark.parametrize(
    "fault",
    [
        MicroscopeHaltedError(),
        DeviceTimeoutError("xy_stage did not arrive within 30 s"),
        CoreError("the core lost the stage"),
    ],
    ids=["halt", "timeout", "core-error"],
)
def test_a_fault_of_the_estimates_move_reaches_the_runner_unchanged(
    holder_stand: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    def fails(x_um: float, y_um: float) -> object:
        raise fault

    monkeypatch.setattr(holder_stand.require(XYStage), "move_to_um", fails)
    with pytest.raises(type(fault)) as info:
        _run(holder_stand, tmp_path, well="A1", plate="96-well")
    assert info.value is fault
    assert _stored(tmp_path).status is StageStatus.failed


def test_an_estimate_outside_the_soft_limits_is_refused_and_does_not_move(
    mm_available: bool, tmp_path: Path
) -> None:
    # As A1 of a 96-well estimated on the Ti2-E: the layer refuses, unforceable.
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    profile = _holder_profile()
    limited = profile.model_copy(
        update={
            "safety": profile.safety.model_copy(
                update={
                    "xy_soft_limits_um": ((-49_000.0, 49_000.0), (-29_000.0, 29_000.0))
                }
            )
        }
    )
    microscope = Microscope.open(limited, dry_run=False)
    try:
        before = _where(microscope)
        with pytest.raises(SafetyRefusedError) as info:
            _run(microscope, tmp_path, well="A1", plate="96-well")
        assert info.value.how_to_force == ""
        assert _stored(tmp_path).status is StageStatus.failed
        assert _where(microscope) == before
    finally:
        microscope.close()


def test_the_estimate_sends_one_move_and_reads_no_position_before_it(
    holder_stand: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = holder_stand.require(XYStage)
    calls: list[tuple[object, ...]] = []
    real_move, real_position = stage.move_to_um, stage.position_um

    def move(x_um: float, y_um: float) -> object:
        calls.append(("move_to_um", x_um, y_um))
        return real_move(x_um, y_um)

    def position() -> object:
        calls.append(("position_um",))
        return real_position()

    monkeypatch.setattr(stage, "move_to_um", move)
    monkeypatch.setattr(stage, "position_um", position)
    result = _result(_run(holder_stand, tmp_path, well="A1", plate="96-well"))
    assert calls == [("move_to_um", result.target_x_um, result.target_y_um)]


def test_goto_well_on_a_dry_plate_drives_to_the_estimate_and_says_why(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path, dry_run=True)
    result = _result(_run(holder_stand, tmp_path, well="A1", plate="96-well"))
    assert result.from_holder is True
    assert _where(holder_stand) == pytest.approx(A1_ESTIMATED, abs=0.02)
    assert result.assumed[1] == (
        "plate not read: plate-registration was a dry run; run it without --dry-run"
    )


def test_the_estimate_in_a_dry_run_says_the_stage_did_not_move(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    microscope = Microscope.open(_holder_profile(), dry_run=True)
    try:
        before = _where(microscope)
        result = _result(_run(microscope, tmp_path, well="A1", plate="96-well"))
        assert result.assumed == (
            ESTIMATE_LINE,
            f"plate not read: {NO_PLATE}",
            NO_OFFSET_LINE,
            DRY_LINE,
        )
        assert (result.x_um, result.y_um) == pytest.approx(A1_ESTIMATED, abs=0.01)
        assert _where(microscope) == before
    finally:
        microscope.close()


def test_goto_well_with_a_usable_plate_ignores_the_holder(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    expected = _at("B2", 0.0, 0.0)
    record = _run(
        holder_stand, tmp_path, well="B2", plate="96-well", orientation="turned-180"
    )
    result = _result(record)
    assert result.from_holder is False
    assert (result.target_x_um, result.target_y_um) == pytest.approx(expected, abs=1e-6)
    assert _where(holder_stand) == pytest.approx(expected, abs=0.02)
    assert result.assumed == ()
    assert record.reads["plate"] is not None


def test_goto_well_refuses_a_plate_that_is_not_the_registered_plate(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    before = _where(holder_stand)
    with pytest.raises(PluginRunError) as info:
        _run(holder_stand, tmp_path, well="B2", plate="384-well")
    assert str(info.value) == (
        "goto-well: the registered plate is a 96-well, not a 384-well: leave plate "
        "out, or register the 384-well"
    )
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(holder_stand) == before


def test_an_unreadable_plate_is_refused_even_with_plate_given(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    # A plate that is there and that the model refuses is not "no plate": it is
    # refused as it was, and is not taken as a reason to drive to the estimate.
    then = datetime(2026, 10, 28, 9, 30, tzinfo=timezone.utc)
    write_stage(
        tmp_path / "run.json",
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result={"plate": "96-well", "assumed": []},
        ),
    )
    before = _where(holder_stand)
    with pytest.raises(UpstreamError) as info:
        _run(holder_stand, tmp_path, well="A1", plate="96-well")
    assert str(info.value).startswith(
        "goto-well reads plate, but the result of plate-registration is not what "
        "goto-well expects ("
    )
    assert str(info.value).endswith("); run plate-registration again")
    assert _where(holder_stand) == before


def test_the_estimate_parameters_parse_from_text() -> None:
    params = parse_params(
        GotoWellParams, ["well=A1", "plate=96-well", "orientation=turned-180"]
    )
    assert (params.plate, params.orientation) == ("96-well", "turned-180")
    defaults = parse_params(GotoWellParams, ["well=A1"])
    assert (defaults.plate, defaults.orientation) == (None, "not-turned")


def test_smc_run_goto_well_drives_to_the_estimate_on_demo_sample(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path / "run.json"
    result = CliRunner(
        env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None}
    ).invoke(
        app,
        [
            "run",
            "goto-well",
            "-p",
            str(DEMO_SAMPLE),
            "--run-doc",
            str(run_doc),
            "--param",
            "well=A1",
            "--param",
            "plate=96-well",
        ],
    )
    assert result.exit_code == 0, result.output
    record = RunDocument.load(run_doc).stages["goto-well"]
    done = _result(record)
    # demo-sample's holder is at (0, 0): A1 is the grid's centre offset, negated.
    assert (done.target_x_um, done.target_y_um) == pytest.approx(
        (-49_500.0, 31_500.0), abs=0.01
    )
    assert done.from_holder is True
    assert done.assumed[0] == (
        "target estimated from the stand's holder, not registered: the 96-well "
        "centred at stage (0, 0) µm, not turned"
    )


# -------------------------------- the autofocus across a well's wall (#454)


def _switched_off_line(well: str) -> str:
    """m3 §5.6 D1's ``assumed`` line, written here and not read from the plugin."""
    return (
        "the autofocus was switched off before the move, so that no lock crossed a "
        f"well's wall: engage it again in {well}"
    )


def _switched_off_warning(well: str) -> str:
    return (
        f"goto-well: the autofocus was switched off before the move to {well}, so "
        "that no lock crosses a well's wall (m2 §12.2 D2): engage it again there"
    )


def _not_switched_off(well: str, cause: str) -> str:
    return (
        f"goto-well: the autofocus could not be switched off before the move to "
        f"{well} ({cause}): the stage did not move, and PFS may still be on: switch "
        "it off at the stand, then go to the well again"
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The plugin's own warnings: ``override`` warns too, on the layer's logger."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "smc.plugins.goto-well"
    ]


def test_goto_well_switches_an_engaged_autofocus_off_before_the_move_and_says_so_first(
    demo_microscope: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(tmp_path, _plate(method="seed", assumed=(SEED_LINE,)))
    af = RecordingAutofocus(engaged=True)
    demo_microscope.override(Autofocus, af)
    expected = _at("B2", 0.0, 0.0)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(demo_microscope, tmp_path, well="b2"))

    # The whole list: a goto-well run makes no other call on the autofocus.
    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    assert af.engaged is False
    assert _where(demo_microscope) == pytest.approx(expected, abs=0.02)
    assert result.autofocus_switched_off is True
    assert result.assumed == (_switched_off_line("B2"), f"plate: {SEED_LINE}")
    assert _warnings(caplog) == [_switched_off_warning("B2")]


def test_goto_well_switches_the_autofocus_off_before_it_sends_the_move(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order: the switch-off and its read back, then the one move, nothing between."""
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True)
    demo_microscope.override(Autofocus, af)
    stage = demo_microscope.require(XYStage)
    real_move = stage.move_to_um

    def move(x_um: float, y_um: float) -> object:
        af.calls.append("move_to_um")
        return real_move(x_um, y_um)

    monkeypatch.setattr(stage, "move_to_um", move)
    _result(_run(demo_microscope, tmp_path, well="A2"))
    assert af.calls == ["is_engaged", "disengage", "is_engaged", "move_to_um"]


def test_goto_well_to_the_estimate_switches_the_autofocus_off_and_says_so_before_the_estimate(
    holder_stand: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    af = RecordingAutofocus(engaged=True)
    holder_stand.override(Autofocus, af)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(holder_stand, tmp_path, well="a1", plate="96-well"))

    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    assert _where(holder_stand) == pytest.approx(A1_ESTIMATED, abs=0.02)
    assert result.autofocus_switched_off is True
    assert result.assumed == (
        _switched_off_line("A1"),
        ESTIMATE_LINE,
        f"plate not read: {NO_PLATE}",
        NO_OFFSET_LINE,
    )
    assert _switched_off_warning("A1") in _warnings(caplog)


def test_goto_well_leaves_an_autofocus_that_is_off_alone(
    demo_microscope: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=False)
    demo_microscope.override(Autofocus, af)

    with caplog.at_level(logging.WARNING, logger="smc"):
        result = _result(_run(demo_microscope, tmp_path, well="B2"))

    assert af.calls == ["is_engaged"]
    assert result.autofocus_switched_off is False
    assert result.assumed == ()
    assert _warnings(caplog) == []
    assert _where(demo_microscope) == pytest.approx(_at("B2", 0.0, 0.0), abs=0.02)


def test_goto_well_searching_counts_as_on(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """PFS on and searching drives Z the moment it finds its reference: it is switched off too."""
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True, locked=False)
    demo_microscope.override(Autofocus, af)

    result = _result(_run(demo_microscope, tmp_path, well="B2"))

    assert af.calls.count("disengage") == 1
    assert "is_locked" not in af.calls
    assert result.autofocus_switched_off is True


def test_goto_well_in_a_dry_run_switches_nothing(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    # A switch-off acts in a dry run (m2 §3.2), and nothing moves to carry a lock.
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True)
    demo_microscope_dry.override(Autofocus, af)

    result = _result(_run(demo_microscope_dry, tmp_path, well="B2"))

    assert af.calls == []
    assert af.engaged is True
    assert result.autofocus_switched_off is False
    assert result.assumed == (DRY_LINE,)


def test_a_stop_before_the_move_keeps_the_autofocus_on_and_sends_nothing(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True)
    demo_microscope.override(Autofocus, af)
    before = _where(demo_microscope)
    stop = threading.Event()
    stop.set()

    record = _run(demo_microscope, tmp_path, well="A2", stop=stop)

    assert record.status is StageStatus.stopped
    assert "disengage" not in af.calls
    assert af.engaged is True
    assert _where(demo_microscope) == before


def test_a_stop_before_the_estimates_move_keeps_the_autofocus_on(
    holder_stand: Microscope, tmp_path: Path
) -> None:
    af = RecordingAutofocus(engaged=True)
    holder_stand.override(Autofocus, af)
    before = _where(holder_stand)
    stop = threading.Event()
    stop.set()

    record = _run(holder_stand, tmp_path, well="A1", plate="96-well", stop=stop)

    assert record.status is StageStatus.stopped
    assert "disengage" not in af.calls
    assert af.engaged is True
    assert _where(holder_stand) == before


def test_the_estimate_switches_the_autofocus_off_before_it_sends_the_move(
    holder_stand: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    af = RecordingAutofocus(engaged=True)
    holder_stand.override(Autofocus, af)
    stage = holder_stand.require(XYStage)
    real_move = stage.move_to_um

    def move(x_um: float, y_um: float) -> object:
        af.calls.append("move_to_um")
        return real_move(x_um, y_um)

    monkeypatch.setattr(stage, "move_to_um", move)
    _result(_run(holder_stand, tmp_path, well="A1", plate="96-well"))
    assert af.calls == ["is_engaged", "disengage", "is_engaged", "move_to_um"]


def test_a_switch_off_error_of_several_lines_is_said_by_its_first(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A vendor's message can run over lines: the run's text keeps one.
    af = RecordingAutofocus(engaged=True)

    def fails() -> None:
        raise CoreError("PFS did not answer\nError in device PFS: 10")

    monkeypatch.setattr(af, "disengage", fails)
    _fails_before_the_move(
        demo_microscope, tmp_path, af, "CoreError: PFS did not answer"
    )


def _fails_before_the_move(
    microscope: Microscope, tmp_path: Path, af: RecordingAutofocus, cause: str
) -> None:
    """Unknown counts as on (FM-19): the run fails with §5.6's text whole, and nothing moved."""
    _seed(tmp_path)
    microscope.override(Autofocus, af)
    before = _where(microscope)

    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, well="a2")

    assert str(info.value) == _not_switched_off("A2", cause)
    # Raised outside the handler: the device's error is not kept on it (FM-98).
    assert info.value.__context__ is None
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.error == f"PluginRunError: {_not_switched_off('A2', cause)}"
    assert _where(microscope) == before


def test_a_switch_off_that_raises_fails_goto_well_before_the_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    af = RecordingAutofocus(engaged=True, fails="disengage")
    _fails_before_the_move(
        demo_microscope, tmp_path, af, "CoreError: disengage failed on the fake"
    )
    assert af.calls == ["is_engaged", "disengage"]


def test_a_state_read_that_raises_fails_goto_well_before_the_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    af = RecordingAutofocus(engaged=True, fails="is_engaged")
    _fails_before_the_move(
        demo_microscope, tmp_path, af, "CoreError: is_engaged failed on the fake"
    )
    assert af.calls == ["is_engaged"]


def test_a_read_after_the_switch_off_that_raises_fails_goto_well_before_the_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    af = RecordingAutofocus(engaged=True, read_fails_after_off=True)
    _fails_before_the_move(
        demo_microscope,
        tmp_path,
        af,
        "CoreError: is_engaged failed on the fake after the switch-off",
    )
    assert af.calls == ["is_engaged", "disengage", "is_engaged"]


def test_an_autofocus_that_still_reads_on_fails_goto_well_before_the_move(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True, stays_on=True)
    demo_microscope.override(Autofocus, af)
    before = _where(demo_microscope)
    text = (
        "goto-well: the autofocus still reads on after it was switched off before "
        "the move to A2: the stage did not move: switch it off at the stand, then go "
        "to the well again"
    )

    with pytest.raises(PluginRunError) as info:
        _run(demo_microscope, tmp_path, well="a2")

    assert str(info.value) == text
    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.error == f"PluginRunError: {text}"
    assert _where(demo_microscope) == before


def test_a_halted_stand_switches_the_autofocus_off_and_the_halt_reaches_the_caller(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """PFS off is the safe side; the move then raises the halt as it did (FM-64).

    The stand's stop switches the autofocus off itself; one switched on again
    while halted (at the stand) is what ``goto-well``'s own switch-off meets.
    """
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True)
    demo_microscope.override(Autofocus, af)
    before = _where(demo_microscope)
    demo_microscope.stop()
    assert af.engaged is False  # the stop's own switch-off
    af.engaged = True
    af.calls.clear()
    try:
        with pytest.raises(MicroscopeHaltedError):
            _run(demo_microscope, tmp_path, well="A2")
    finally:
        demo_microscope.resume()
    assert af.calls == ["is_engaged", "disengage", "is_engaged"]
    assert _stored(tmp_path).status is StageStatus.failed
    assert _where(demo_microscope) == before


@pytest.mark.parametrize(
    "fault",
    [MicroscopeHaltedError(), MotionStoppedError("the stage was stopped")],
    ids=["halt", "motion-stopped"],
)
def test_a_halt_from_the_switch_off_reaches_the_caller_unchanged(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    """A halt or a stop is not a switch-off that failed: it passes first (FM-64)."""
    _seed(tmp_path)
    af = RecordingAutofocus(engaged=True)
    demo_microscope.override(Autofocus, af)
    before = _where(demo_microscope)

    def fails() -> None:
        raise fault

    monkeypatch.setattr(af, "disengage", fails)
    with pytest.raises(type(fault)) as info:
        _run(demo_microscope, tmp_path, well="A2")
    assert info.value is fault
    assert _where(demo_microscope) == before


def test_a_goto_well_result_written_before_454_loads_with_the_autofocus_not_switched_off() -> (
    None
):
    older = {
        "well": "B2",
        "side": "centre",
        "target_x_um": 1.0,
        "target_y_um": 2.0,
        "x_um": 1.0,
        "y_um": 2.0,
        "assumed": [],
    }
    assert GotoWellResult.model_validate(older).autofocus_switched_off is False


def _ti2_stand() -> tuple[FakeCore, Microscope]:
    """The Ti2-E's focus and PFS on a ``FakeCore``, PFS driving Z to its offset's plane.

    Built here and not imported from ``tests/unit/test_mm_autofocus``: a module of
    ``tests/`` is collected before ``tests/unit`` is on the path. No turret, so the
    plate is moved into no objective's frame.
    """
    core = FakeCore(
        [
            FakeDevice("ZDrive", DeviceType.Stage),
            FakeDevice("PFS", DeviceType.AutoFocus),
            FakeDevice("PFSOffset", DeviceType.Stage),
            FakeDevice("XYStage", DeviceType.XYStage),
            FakeDevice("Camera-1", DeviceType.Camera),
        ],
        camera="Camera-1",
        xy_stage="XYStage",
        focus="ZDrive",
        autofocus="PFS",
    )
    profile = Profile(
        microscope=MicroscopeSection(name="ti2e-fake"),
        safety=SafetySection.model_validate(
            {"max_z_jog_um": 50.0, "max_engage_excursion_um": 50.0}
        ),
        autofocus=AutofocusSection.model_validate({"engage_drives_z_to_offset": True}),
    )
    return core, Microscope.from_core(cast("CMMCorePlus", core), profile)


def test_goto_well_leaves_z_where_pfs_held_it_when_the_next_wells_plane_lies_60_um_higher(
    tmp_path: Path,
) -> None:
    """F11's two wells on the Ti2-E fake: Z does not follow PFS over the wall (FM-223).

    The fake drives Z only at a locked read, so the test reads once more after the
    move: without the switch-off, that read moved Z to ``Z0 + 60.00`` (measured
    in #454's plan).
    """
    core, microscope = _ti2_stand()
    try:
        _seed(tmp_path, _plate(objective=None), profile="ti2e-fake")
        a1, a2 = _at("A1", 0.0, 0.0), _at("A2", 0.0, 0.0)
        boundary = (a1[0] + a2[0]) / 2.0
        microscope.require(XYStage).move_to_um(*a1)
        z = microscope.require(ZStage)
        z0 = z.position_um()
        core.autofocus_target_um = lambda: (
            z0 if core.getXPosition() < boundary else z0 + 60.0
        )
        af = microscope.require(Autofocus)
        core.enableContinuousFocus(True)  # switched on at the stand
        assert af.is_locked() is True  # seen locked in A1, where its plane is Z0
        assert z.position_um() == pytest.approx(z0, abs=0.01)

        record = _run(microscope, tmp_path, well="A2")
        locked_after = af.is_locked()  # the read at which a carried lock drives Z

        assert z.position_um() == pytest.approx(z0, abs=0.01)
        assert locked_after is False
        assert af.is_engaged() is False
        assert _where(microscope) == pytest.approx(a2, abs=0.02)
        assert _result(record).autofocus_switched_off is True
    finally:
        microscope.close()
