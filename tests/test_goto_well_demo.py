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

import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.errors import SafetyRefusedError
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, UpstreamError, parse_params, run_plugin
from smc.plugins.goto_well import GotoWellParams, GotoWellPlugin, GotoWellResult
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

pytestmark = pytest.mark.demo

A1 = (1000.0, 2000.0)
ROTATION_DEG = 3.0
PITCH_UM = 9000.0
HALF_UM = 3200.0
SEED_LINE = "one reference well (A1): rotation 0° assumed, not measured"
DRY_LINE = "dry run: the stage did not move; x_um and y_um are the target"


def _plate(*, method: str = "references", assumed: tuple[str, ...] = ()) -> PlateResult:
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=A1,
        rotation_deg=ROTATION_DEG,
        method=method,  # type: ignore[arg-type]
        residual_um=5.0 if method == "references" else None,
        scale_error=0.001 if method == "references" else None,
        n_refs=3 if method == "references" else 1,
        assumed=assumed,
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
        ("B3", "top", (0.0, HALF_UM)),
        ("C5", "right", (HALF_UM, 0.0)),
        ("B2", "bottom", (0.0, -HALF_UM)),
        ("B2", "left", (-HALF_UM, 0.0)),
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
    record = _run(demo_microscope_dry, tmp_path, well="C5", side="right")
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


# --------------------------------------------------------------- from text


def test_the_parameters_parse_from_text() -> None:
    params = parse_params(GotoWellParams, ["well=H12", "side=left"])
    assert (params.well, params.side) == ("H12", "left")
    assert parse_params(GotoWellParams, ["well=A1"]).side == "centre"


def test_a_side_that_is_not_one_of_the_five_is_refused() -> None:
    with pytest.raises(ValueError, match="side"):
        GotoWellParams(well="A1", side="middle")  # type: ignore[arg-type]
