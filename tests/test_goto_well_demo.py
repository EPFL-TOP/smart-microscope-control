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

import pytest

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
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveOffset, ObjectiveTurret, XYStage
from smc.hardware.errors import (
    CoreError,
    DeviceTimeoutError,
    MicroscopeHaltedError,
    SafetyRefusedError,
)
from smc.hardware.profile import Profile
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

pytestmark = pytest.mark.demo

A1 = (1000.0, 2000.0)
ROTATION_DEG = 3.0
PITCH_UM = 9000.0
HALF_UM = 3200.0
SEED_LINE = "one reference well (A1): rotation 0° assumed from the declared orientation (not-turned), not measured"
DRY_LINE = "dry run: the stage did not move; x_um and y_um are the target"


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
    tmp_path: Path, plate: PlateResult | None = None, *, dry_run: bool = False
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
            profile="demo",
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
    ):
        _run(demo_microscope, tmp_path, well="A1")
    assert _where(demo_microscope) == before


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
    assert entry.plugin.reads == ("plate",)
    assert entry.plugin.reads_optional == ()
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
