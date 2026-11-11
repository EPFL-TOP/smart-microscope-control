"""``plate-registration`` through ``run_plugin``: references and wall touches in, the ``plate`` slot out (m3 §5, #19).

The plugin moves no device and sends nothing (``requires = ()``); it reads the turret,
when there is one, for the objective the plate is seen with (m3 §18.3, #340). The
demo stand is here because ``run_plugin`` needs a ``Microscope``, and its dry-run
twin shows that a dry run is the same computation. Every expected value is written here from the cosine and
the sine of the rotation and the plate's 9 mm pitch and 6.4 mm wells, never read
from the result's own fields or from ``smc`` code (FM-124). The fit against
``useq``'s forward model is the unit tests'.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import useq
from useq import _plate_registry

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.errors import HardwareError
from smc.plugins import ParamsError, PluginRunError, load, parse_params, run_plugin
from smc.plugins.plate_registration.plugin import (
    PlateRegistrationParams,
    PlateRegistrationPlugin,
)
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus

pytestmark = pytest.mark.demo

NAME = "plate-registration"
PITCH_UM = 9000.0
HALF_UM = 3200.0


def _placed(
    a1: tuple[float, float], rotation_deg: float, well: str
) -> dict[str, object]:
    """A reference at a well's centre on a 96-well plate put at ``a1`` and turned by ``rotation_deg``."""
    row, col = ord(well[0]) - ord("A"), int(well[1:]) - 1
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    angle = math.radians(rotation_deg)
    return {
        "well": well,
        "x_um": a1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        "y_um": a1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    }


def _wall(well: str, side: str, x_um: float, y_um: float) -> dict[str, object]:
    return {"well": well, "side": side, "x_um": x_um, "y_um": y_um}


def _run(microscope: Microscope, tmp_path: Path, **params: object) -> StageRecord:
    return run_plugin(
        PlateRegistrationPlugin(),
        microscope,
        PlateRegistrationParams(**params),  # type: ignore[arg-type]
        tmp_path / "run.json",
    )


def _result(record: StageRecord) -> PlateResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return PlateResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages[NAME]


def _refused(
    microscope: Microscope, tmp_path: Path, **params: object
) -> PluginRunError:
    """Run and expect a ``PluginRunError``; the stored record is ``failed`` and has no result."""
    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, **params)
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    return info.value


# ------------------------------------------------------------------- the fit


def test_a_seed_fills_the_plate_slot_with_rotation_zero(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """One reference well, typed as an operator types it: A1 is where it was seen, nothing is measured."""
    result = _result(
        _run(
            demo_microscope,
            tmp_path,
            refs=[{"well": "d 06", "x_um": 50000.0, "y_um": -20000.0}],
        )
    )
    assert result.plate == "96-well"
    assert result.method == "seed"
    assert result.rotation_deg == 0.0
    assert result.residual_um is None
    assert result.scale_error is None
    assert result.n_refs == 1
    # D6 is five columns of 9 mm along and three rows down from A1.
    assert result.a1_center_xy_um == pytest.approx(
        (50000.0 - 5 * PITCH_UM, -20000.0 + 3 * PITCH_UM)
    )
    assert result.assumed == (
        "one reference well (D6): rotation 0° assumed, not measured",
    )
    assert _stored(tmp_path).status is StageStatus.done
    assert _stored(tmp_path).provides == "plate"


def test_references_fill_the_plate_slot_and_the_record_keeps_them(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    a1 = (1500.0, -900.0)
    refs = [_placed(a1, 3.0, well) for well in ("A1", "A12", "H1")]
    result = _result(_run(demo_microscope, tmp_path, refs=refs))
    assert result.method == "references"
    assert result.a1_center_xy_um == pytest.approx(a1, abs=1e-6)
    assert result.rotation_deg == pytest.approx(3.0, abs=1e-9)
    assert result.residual_um is not None
    assert result.residual_um < 1e-6
    assert result.n_refs == 3
    assert [r.well for r in result.refs] == ["A1", "A12", "H1"]
    assert result.assumed == ()
    # The audit trail: the run document's params hold every reference that made the plate.
    assert _stored(tmp_path).params["refs"] == refs


def test_a_dry_run_is_the_same_computation(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    refs = [_placed((1500.0, -900.0), 3.0, well) for well in ("A1", "A12", "H1")]
    record = _run(demo_microscope_dry, tmp_path, refs=refs)
    assert record.dry_run is True
    result = _result(record)
    assert result.a1_center_xy_um == pytest.approx((1500.0, -900.0), abs=1e-6)
    assert result.rotation_deg == pytest.approx(3.0, abs=1e-9)


def test_the_plate_and_the_plate_s_pitch_are_the_ones_asked_for(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A 384-well plate is 4.5 mm between wells: A24 is 23 pitches along, P1 15 down."""
    a1 = (-2000.0, 3000.0)
    refs = [
        {"well": "A1", "x_um": a1[0], "y_um": a1[1]},
        {"well": "A24", "x_um": a1[0] + 23 * 4500.0, "y_um": a1[1]},
        {"well": "P1", "x_um": a1[0], "y_um": a1[1] - 15 * 4500.0},
    ]
    result = _result(_run(demo_microscope, tmp_path, plate="384-well", refs=refs))
    assert result.plate == "384-well"
    assert result.a1_center_xy_um == pytest.approx(a1, abs=1e-6)
    assert result.rotation_deg == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------- the walls


def test_walls_in_stage_directions_become_a_reference(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """B2 from a -X and +X pair and one -Y wall; the touches' other coordinate is not read.

    The axis with one wall is assumed and the result says so.
    """
    a1 = (1000.0, -500.0)
    b2 = (a1[0] + PITCH_UM, a1[1] - PITCH_UM)
    walls = [
        _wall("B2", "-X", b2[0] - HALF_UM, 777.0),
        _wall("b2", "+X", b2[0] + HALF_UM, -777.0),
        _wall("B2", "-Y", 555.0, b2[1] - HALF_UM),
    ]
    result = _result(
        _run(
            demo_microscope,
            tmp_path,
            refs=[{"well": "A1", "x_um": a1[0], "y_um": a1[1]}],
            walls=walls,
        )
    )
    assert [r.well for r in result.refs] == ["A1", "B2"]
    assert (result.refs[1].x_um, result.refs[1].y_um) == pytest.approx(b2)
    assert result.a1_center_xy_um == pytest.approx(a1, abs=1e-6)
    assert result.assumed == (
        "B2: the Y centre is one wall plus the nominal half-size, 3200 µm",
    )
    assert _stored(tmp_path).params["walls"] == walls


def test_references_come_first_then_each_wall_well_in_the_order_of_its_first_touch(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    h1 = (0.0, -7 * PITCH_UM)
    a12 = (11 * PITCH_UM, 0.0)
    walls = [
        _wall("H1", "-X", h1[0] - HALF_UM, 0.0),
        _wall("A12", "-X", a12[0] - HALF_UM, 0.0),
        _wall("H1", "-Y", 0.0, h1[1] - HALF_UM),
        _wall("A12", "-Y", 0.0, a12[1] - HALF_UM),
    ]
    result = _result(
        _run(
            demo_microscope,
            tmp_path,
            refs=[{"well": "A1", "x_um": 0.0, "y_um": 0.0}],
            walls=walls,
        )
    )
    assert [r.well for r in result.refs] == ["A1", "H1", "A12"]
    assert (result.refs[1].x_um, result.refs[1].y_um) == pytest.approx(h1)
    assert (result.refs[2].x_um, result.refs[2].y_um) == pytest.approx(a12)
    assert result.assumed == (
        "H1: the X centre is one wall plus the nominal half-size, 3200 µm",
        "H1: the Y centre is one wall plus the nominal half-size, 3200 µm",
        "A12: the X centre is one wall plus the nominal half-size, 3200 µm",
        "A12: the Y centre is one wall plus the nominal half-size, 3200 µm",
    )


def test_walls_alone_make_a_seed_that_says_what_it_assumed(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    walls = [
        _wall("A1", "-X", 1000.0 - HALF_UM, 0.0),
        _wall("A1", "+X", 1000.0 + HALF_UM, 0.0),
        _wall("A1", "+Y", 0.0, 2000.0 + HALF_UM),
        _wall("A1", "-Y", 0.0, 2000.0 - HALF_UM),
    ]
    result = _result(_run(demo_microscope, tmp_path, walls=walls))
    assert result.method == "seed"
    assert result.a1_center_xy_um == pytest.approx((1000.0, 2000.0))
    assert result.assumed == (
        "one reference well (A1): rotation 0° assumed, not measured",
    )


def test_the_seed_s_line_comes_before_the_walls_lines(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """One wall per axis: the result says the rotation was assumed, then what each wall assumed."""
    walls = [
        _wall("A1", "+X", 1000.0 + HALF_UM, 0.0),
        _wall("A1", "+Y", 0.0, 2000.0 + HALF_UM),
    ]
    result = _result(_run(demo_microscope, tmp_path, walls=walls))
    assert result.a1_center_xy_um == pytest.approx((1000.0, 2000.0))
    assert result.assumed == (
        "one reference well (A1): rotation 0° assumed, not measured",
        "A1: the X centre is one wall plus the nominal half-size, 3200 µm",
        "A1: the Y centre is one wall plus the nominal half-size, 3200 µm",
    )


def test_the_assumed_half_sizes_are_each_axis_s_on_an_anisotropic_plate(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well 10 mm wide and 5 mm high: one wall in X is 5000 µm from the centre, one in Y 2500 (FM-73)."""
    monkeypatch.setitem(
        _plate_registry._PLATE_REGISTRY,
        "test-anisotropic-15",
        {
            "rows": 3,
            "columns": 5,
            "well_spacing": (12.0, 7.0),
            "well_size": (10.0, 5.0),
            "circular_wells": False,
            "name": "test-anisotropic-15",
        },
    )
    assert useq.WellPlate.from_str("test-anisotropic-15").well_size == (10.0, 5.0)
    b3 = (2 * 12000.0, -7000.0)
    result = _result(
        _run(
            demo_microscope,
            tmp_path,
            plate="test-anisotropic-15",
            refs=[{"well": "A1", "x_um": 0.0, "y_um": 0.0}],
            walls=[
                _wall("B3", "-X", b3[0] - 5000.0, 0.0),
                _wall("B3", "-Y", 0.0, b3[1] - 2500.0),
            ],
        )
    )
    assert (result.refs[1].x_um, result.refs[1].y_um) == pytest.approx(b3)
    assert result.assumed == (
        "B3: the X centre is one wall plus the nominal half-size, 5000 µm",
        "B3: the Y centre is one wall plus the nominal half-size, 2500 µm",
    )


# ------------------------------------------------------------------ refusals


def test_a_residual_beyond_300_um_fails_and_names_the_well(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    def refs(h1_x_um: float) -> list[dict[str, object]]:
        return [
            {"well": "A1", "x_um": 0.0, "y_um": 0.0},
            {"well": "A12", "x_um": 99000.0, "y_um": 0.0},
            {"well": "H1", "x_um": h1_x_um, "y_um": -63000.0},
        ]

    error = _refused(demo_microscope, tmp_path, refs=refs(700.0))
    assert "H1 is 348 µm from where the others put it: centre it again" in str(error)

    # 600 µm off is a residual of 298.4 µm: inside the limit, and reported.
    result = _result(_run(demo_microscope, tmp_path, refs=refs(600.0)))
    assert result.residual_um == pytest.approx(298.4, abs=0.1)


def test_a_centre_900_um_off_on_the_lab_plate_is_refused_and_named(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """Three wells check each other: one centre 900 µm off in X is refused and named (m5 §15 D7, #410).

    The Greiner 24-well at 19.45 mm, A1 where the Ti2-E's holder puts it, (-48 225, 28 850) µm:
    B3, B5 and C3 are written out from that pitch. The 900 µm is three times the 300 µm that
    refuses a fit, and the other two centres exact, so the worst residual is B3's (572.8 µm,
    against 351.5 for B5 and 247.2 for C3) and the message names it.
    """
    centres = {
        "B3": (-9325.0, 9400.0),
        "B5": (29575.0, 9400.0),
        "C3": (-9325.0, -10050.0),
    }

    def refs(b3_x_error_um: float) -> list[dict[str, object]]:
        return [
            {
                "well": well,
                "x_um": x + (b3_x_error_um if well == "B3" else 0.0),
                "y_um": y,
            }
            for well, (x, y) in centres.items()
        ]

    error = _refused(
        demo_microscope, tmp_path, plate="greiner-24-well", refs=refs(900.0)
    )
    assert "B3 is 573 µm from where the others put it: centre it again" in str(error)

    # Without the error the same three centres register the plate where they say it is.
    result = _result(
        _run(demo_microscope, tmp_path, plate="greiner-24-well", refs=refs(0.0))
    )
    assert result.a1_center_xy_um == pytest.approx((-48225.0, 28850.0), abs=1e-6)
    assert result.method == "references"
    assert result.n_refs == 3


def test_a_well_given_twice_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    error = _refused(
        demo_microscope,
        tmp_path,
        refs=[
            {"well": "A1", "x_um": 0.0, "y_um": 0.0},
            {"well": "a1", "x_um": 5.0, "y_um": 5.0},
        ],
    )
    assert str(error) == "plate-registration: well A1 given twice"


def test_a_well_given_both_as_a_reference_and_by_walls_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    error = _refused(
        demo_microscope,
        tmp_path,
        refs=[{"well": "A01", "x_um": 0.0, "y_um": 0.0}],
        walls=[_wall("a1", "-X", 0.0, 0.0), _wall("A1", "-Y", 0.0, 0.0)],
    )
    assert (
        str(error)
        == "plate-registration: A1 is given both as a reference and by walls: "
        "give it one way"
    )


@pytest.mark.parametrize(
    ("misplaced", "message"),
    [
        ("A1", "A1 is 436 µm from where the others put it: centre it again"),
        ("A12", "A12 is 446 µm from where the others put it: centre it again"),
        ("H1", "H1 is 348 µm from where the others put it: centre it again"),
    ],
)
def test_the_refusal_names_the_worst_reference_whichever_it_is(
    demo_microscope: Microscope, tmp_path: Path, misplaced: str, message: str
) -> None:
    """700 µm off in X on the first, a middle or the last reference: the largest name is not the worst well."""
    truth = {"A1": (0.0, 0.0), "A12": (99000.0, 0.0), "H1": (0.0, -63000.0)}
    refs = [
        {"well": name, "x_um": x + (700.0 if name == misplaced else 0.0), "y_um": y}
        for name, (x, y) in truth.items()
    ]
    error = _refused(demo_microscope, tmp_path, refs=refs)
    assert str(error) == f"plate-registration: {message}"


def test_the_first_reference_a_little_off_is_inside_the_limit(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A1 400 µm off in X is a residual of 249 µm, under 300, and it is A1 that carries it."""
    refs = [
        {"well": "A1", "x_um": 400.0, "y_um": 0.0},
        {"well": "A12", "x_um": 99000.0, "y_um": 0.0},
        {"well": "H1", "x_um": 0.0, "y_um": -63000.0},
    ]
    result = _result(_run(demo_microscope, tmp_path, refs=refs))
    assert result.residual_um == pytest.approx(249.25, abs=0.01)


@pytest.mark.parametrize("first", ["A1", "A12"])
def test_two_references_equally_far_from_the_fit_name_the_first_given(
    demo_microscope: Microscope, tmp_path: Path, first: str
) -> None:
    """A12 700 µm too far along X: each reference is 350 µm from the fit, and the tie goes to the first given.

    The numbers are whole µm and the plate is not turned, so the two distances
    are equal to the last bit on every platform, in either order.
    """
    at = {"A1": 0.0, "A12": 99700.0}
    order = (first, "A12" if first == "A1" else "A1")
    refs = [{"well": name, "x_um": at[name], "y_um": 0.0} for name in order]
    error = _refused(demo_microscope, tmp_path, refs=refs)
    assert str(error) == (
        f"plate-registration: {first} is 350 µm from where the others put it: "
        f"centre it again"
    )


def test_the_first_well_given_both_ways_is_the_one_named(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    error = _refused(
        demo_microscope,
        tmp_path,
        refs=[
            {"well": "H12", "x_um": 99000.0, "y_um": -63000.0},
            {"well": "A1", "x_um": 0.0, "y_um": 0.0},
        ],
        walls=[
            _wall("A1", "-X", 0.0, 0.0),
            _wall("A1", "-Y", 0.0, 0.0),
            _wall("H12", "-X", 0.0, 0.0),
            _wall("H12", "-Y", 0.0, 0.0),
        ],
    )
    assert str(error) == (
        "plate-registration: H12 is given both as a reference and by walls: "
        "give it one way"
    )


def test_the_entry_point_loads_plate_registration() -> None:
    entry = load("plate-registration")

    assert entry.plugin.provides == "plate"
    assert entry.plugin.requires == ()
    assert entry.plugin.reads == ()
    assert entry.plugin.reads_optional == ()
    assert (
        entry.target == "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin"
    )


def test_a_seed_keeps_its_reference_in_the_record(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    result = _result(
        _run(
            demo_microscope,
            tmp_path,
            refs=[{"well": "d 06", "x_um": 50000.0, "y_um": -20000.0}],
        )
    )
    assert [(r.well, r.x_um, r.y_um) for r in result.refs] == [
        ("D6", 50000.0, -20000.0)
    ]


# ------------------------------------------ the objective the plate was seen with (#340)

SEED = [{"well": "d 06", "x_um": 50000.0, "y_um": -20000.0}]
TEN = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
TURRET_LABELS = (
    "Nikon 40X Plan Fluor ELWD, Nikon 10X S Fluor, Objective-2, "
    "Nikon 20X Plan Fluor ELWD, Objective-4, Objective-5"
)


def test_the_registration_records_the_objective_in_place(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    result = _result(_run(demo_microscope, tmp_path, refs=SEED))

    assert result.objective == TEN
    # And the run document keeps it, for the scan that comes later.
    stored = _stored(tmp_path).result
    assert stored is not None
    assert stored["objective"] == TEN


def test_the_registration_records_the_objective_requested(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    turret = demo_microscope.require(ObjectiveTurret)

    result = _result(_run(demo_microscope, tmp_path, refs=SEED, objective=FORTY))

    assert result.objective == FORTY
    # Named, not turned: the registration touches no stage and no turret.
    assert turret.current() == TEN


def test_an_objective_not_on_the_turret_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    error = _refused(demo_microscope, tmp_path, refs=SEED, objective="Nikon 100X")

    assert str(error) == (
        f"plate-registration: objective 'Nikon 100X' is not on the turret: {TURRET_LABELS}"
    )


def test_a_stand_without_a_turret_records_no_objective_and_refuses_one_asked_for(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = demo_microscope.get
    monkeypatch.setattr(
        demo_microscope,
        "get",
        lambda wanted: None if wanted is ObjectiveTurret else real_get(wanted),
    )

    assert _result(_run(demo_microscope, tmp_path, refs=SEED)).objective is None
    error = _refused(demo_microscope, tmp_path, refs=SEED, objective=TEN)
    assert str(error) == (
        "plate-registration: the stand has no turret, so objective cannot be "
        "checked; leave it unset"
    )


def test_a_turret_at_an_empty_position_is_refused_before_the_fit(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The protocol lists empty positions, whose label is ``""``: no objective, not one called nothing (FM-125)."""
    monkeypatch.setattr(demo_microscope.require(ObjectiveTurret), "current", lambda: "")

    error = _refused(demo_microscope, tmp_path, refs=SEED)

    assert str(error) == (
        "plate-registration: no objective is in place: the turret is at an empty "
        "position; turn it to an objective, or give objective to plan for one"
    )


def test_the_objective_is_checked_before_the_references(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are wrong: no reference at all, and a turret at an empty position. The objective comes first (m3 §18.3)."""
    monkeypatch.setattr(demo_microscope.require(ObjectiveTurret), "current", lambda: "")

    error = _refused(demo_microscope, tmp_path)

    assert str(error).startswith("plate-registration: no objective is in place: ")


def test_a_turret_that_cannot_be_read_is_a_fault_and_not_no_objective(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FM-19: a read that fails is not caught and not taken as an unknown objective; no plate is written."""
    fault = HardwareError("objective_turret: getStateLabel failed (no answer)")

    def fails() -> str:
        raise fault

    monkeypatch.setattr(demo_microscope.require(ObjectiveTurret), "current", fails)

    with pytest.raises(HardwareError) as info:
        _run(demo_microscope, tmp_path, refs=SEED)

    assert info.value is fault
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None


def test_the_parameter_that_names_the_objective_says_what_it_is_for() -> None:
    field = PlateRegistrationParams.model_fields["objective"]
    assert field.default is None
    assert field.description is not None
    assert "turret label" in field.description
    assert "default the one in place" in field.description


def test_no_reference_is_refused(demo_microscope: Microscope, tmp_path: Path) -> None:
    error = _refused(demo_microscope, tmp_path)
    assert str(error) == (
        "plate-registration: no reference: give refs (a centred well) or walls "
        "(wall touches); see docs/plugins/plate-registration.md"
    )


def test_a_wall_without_its_other_axis_is_refused(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    error = _refused(
        demo_microscope,
        tmp_path,
        walls=[_wall("A1", "-X", 0.0, 0.0), _wall("A1", "+X", 6400.0, 0.0)],
    )
    assert str(error) == ("plate-registration: no Y wall for A1: give -Y and/or +Y")


def test_a_wrong_plate_type_fails_and_fills_no_slot(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    refs = [_placed((0.0, 0.0), 0.0, well) for well in ("A1", "A12", "H1")]
    error = _refused(demo_microscope, tmp_path, plate="384-well", refs=refs)
    assert "off the 384-well definition by +100.0%" in str(error)


def test_an_unknown_plate_is_refused_with_useq_s_words(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    error = _refused(
        demo_microscope,
        tmp_path,
        plate="97-well",
        refs=[{"well": "A1", "x_um": 0.0, "y_um": 0.0}],
    )
    assert "Unknown plate name '97-well'" in str(error)


def test_a_plate_turned_beyond_the_bound_fails_and_the_parameter_allows_it(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    refs = [_placed((10.0, 20.0), 17.5, well) for well in ("A1", "A12", "H1")]
    error = _refused(demo_microscope, tmp_path, refs=refs)
    assert "the fit turns the plate by 17.5°, beyond ±10°" in str(error)

    result = _result(_run(demo_microscope, tmp_path, refs=refs, max_rotation_deg=20.0))
    assert result.rotation_deg == pytest.approx(17.5, abs=1e-9)


def test_two_references_the_same_distance_apart_are_not_a_plate_turned_by_90_degrees(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """A2 given as B1: the fit is exact at 90° (FM-121), so only the bound can refuse it."""
    error = _refused(
        demo_microscope,
        tmp_path,
        refs=[
            {"well": "A1", "x_um": 0.0, "y_um": 0.0},
            {"well": "B1", "x_um": 9000.0, "y_um": 0.0},
        ],
    )
    assert "beyond ±10°" in str(error)


@pytest.mark.parametrize("bad", [0.0, -1.0, 45.5, float("nan"), float("inf")])
def test_the_rotation_bound_is_itself_bounded(bad: float) -> None:
    with pytest.raises(ValueError, match="max_rotation_deg"):
        PlateRegistrationParams(max_rotation_deg=bad)


# --------------------------------------------------------------- from text


def test_references_and_walls_parse_from_text() -> None:
    params = parse_params(
        PlateRegistrationParams,
        [
            "plate=384-well",
            'refs=[{"well": "a1", "x_um": 1, "y_um": 2}]',
            'walls=[{"well": "B2", "side": "-X", "x_um": 3, "y_um": 4}]',
            "max_rotation_deg=5",
        ],
    )
    assert params.plate == "384-well"
    assert [(r.well, r.x_um, r.y_um) for r in params.refs] == [("a1", 1.0, 2.0)]
    assert [(w.well, w.side, w.x_um, w.y_um) for w in params.walls] == [
        ("B2", "-X", 3.0, 4.0)
    ]
    assert params.max_rotation_deg == 5.0


@pytest.mark.parametrize("word", ["left", "right", "top", "bottom"])
def test_the_old_side_words_are_refused_by_the_cli_parameters(word: str) -> None:
    # `smc run plate-registration --param walls=[...]` of an earlier session names
    # the sides on the image: refused with the reason, never read in the stage's
    # frame (FM-189), and never translated.
    with pytest.raises(ParamsError) as info:
        parse_params(
            PlateRegistrationParams,
            [f'walls=[{{"well": "B2", "side": "{word}", "x_um": 3, "y_um": 4}}]'],
        )
    (problem,) = info.value.problems
    assert problem.startswith("walls.0.side: ")
    assert f"side {word!r} is a side on the image: a wall is named by a stage" in (
        problem
    )
    assert "maps them with the camera calibration" in problem


def test_a_reference_without_its_y_is_reported_by_its_path() -> None:
    with pytest.raises(ParamsError) as info:
        parse_params(PlateRegistrationParams, ['refs=[{"well": "A1", "x_um": 1}]'])
    assert info.value.problems == ("refs.0.y_um: Field required",)


def test_the_parameters_are_described() -> None:
    from smc.plugins import describe_params

    keys = [row[0] for row in describe_params(PlateRegistrationParams)]
    assert keys == ["plate", "refs", "walls", "max_rotation_deg", "objective"]
