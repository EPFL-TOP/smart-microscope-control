"""The pure part of plate registration: the fit, the wall arithmetic, the nearest well (m3 §5, #19).

Ported from ``nikon-control``'s ``tests/test_plate.py``. The ground truth is
always ``useq``'s own forward model (``WellPlatePlan.all_well_positions``),
never ``smc`` code: a test that used ``well_geometry`` as its oracle could not
catch a convention error that the fit and the geometry share (FM-124). The one
test plate that is not square, in wells or in pitch, is the only FM-73 check
there is: every registered plate has equal X and Y pitches and sizes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import pytest
import useq
from hypothesis import given, settings
from hypothesis import strategies as st
from useq import _plate_registry

from smc.plugins.plate_registration import algorithm
from smc.plugins.plate_registration.algorithm import (
    DIAMETER_TOLERANCE,
    GOOD_RESIDUAL_UM,
    SCALE_TOLERANCE,
    calibrate,
    centre_from_walls,
    nearest_well,
    nominal_offsets,
    suggested_refs,
    wall_position,
)
from smc.plugins.slots import PlateResult, WallTouch, WellRef

ANISOTROPIC = "test-anisotropic-15"
PLATES = ("6-well", "12-well", "24-well", "48-well", "96-well", "384-well", "1536-well")


@pytest.fixture
def anisotropic(monkeypatch: pytest.MonkeyPatch) -> str:
    """A 3 x 5 plate with 12 mm by 7 mm pitches and 10 mm by 5 mm wells (FM-73)."""
    monkeypatch.setitem(
        _plate_registry._PLATE_REGISTRY,
        ANISOTROPIC,
        {
            "rows": 3,
            "columns": 5,
            "well_spacing": (12.0, 7.0),
            "well_size": (10.0, 5.0),
            "circular_wells": False,
            "name": ANISOTROPIC,
        },
    )
    return ANISOTROPIC


def _measure(
    plate: str,
    a1: tuple[float, float],
    rotation: float,
    well_names: Sequence[str],
) -> list[WellRef]:
    """Ground-truth stage positions from ``useq`` for a known plate placement."""
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str(plate), a1_center_xy=a1, rotation=rotation
    )
    lookup = {str(p.name): p for p in plan.all_well_positions}
    return [
        WellRef(well=n, x_um=float(lookup[n].x), y_um=float(lookup[n].y))
        for n in well_names
    ]


def _result(
    plate: str, a1: tuple[float, float], rotation: float, *, method: str = "references"
) -> PlateResult:
    return PlateResult.model_validate(
        {
            "plate": plate,
            "a1_center_xy_um": a1,
            "rotation_deg": rotation,
            "method": method,
            "residual_um": None,
            "scale_error": None,
            "n_refs": 0,
        }
    )


def _touch(well: str, side: str, x: float, y: float) -> WallTouch:
    return WallTouch.model_validate({"well": well, "side": side, "x_um": x, "y_um": y})


# ---- nominal geometry ----------------------------------------------------


def test_nominal_offsets_match_the_plate_definition() -> None:
    off = nominal_offsets("96-well")
    assert off["A1"] == (0.0, 0.0)
    # 96-well spacing is 9 mm; stage units are µm
    assert off["A2"][0] == pytest.approx(9000.0)
    assert abs(off["B1"][1]) == pytest.approx(9000.0)
    assert len(off) == 96


def test_the_schema_exists_before_the_plate_is_located() -> None:
    """You have to point at A1 before you know where A1 is."""
    off = nominal_offsets("96-well")
    assert len(off) == 96
    assert off["A1"] == (0.0, 0.0)
    assert off["H12"] == pytest.approx((99000.0, -63000.0))


def test_nominal_offsets_use_each_axis_s_pitch(anisotropic: str) -> None:
    """B3 is two columns of 12 mm along, one row of 7 mm down: FM-73."""
    off = nominal_offsets(anisotropic)
    assert off["B3"] == pytest.approx((24000.0, -7000.0))
    assert len(off) == 15


def test_an_unknown_plate_is_useq_s_error_unchanged() -> None:
    with pytest.raises(ValueError, match="Unknown plate name 'nope'"):
        nominal_offsets("nope")


def test_suggested_refs_are_the_far_corners() -> None:
    assert suggested_refs("96-well") == ("A1", "A12", "H1")
    assert suggested_refs("384-well") == ("A1", "A24", "P1")


def test_a_single_well_plate_suggests_its_one_well_once() -> None:
    assert suggested_refs("coverslip-18mm-square") == ("A1",)


# ---- round trips ---------------------------------------------------------


@pytest.mark.parametrize(
    ("a1", "rot"),
    [
        ((0.0, 0.0), 0.0),
        ((1234.5, -6789.0), 0.0),
        ((1000.0, 2000.0), 2.0),
        ((-500.0, 400.0), -3.75),
        ((10.0, 20.0), 9.5),
    ],
)
def test_recovers_a_known_placement(a1: tuple[float, float], rot: float) -> None:
    refs = _measure("96-well", a1, rot, ["A1", "A12", "H1"])
    cal = calibrate("96-well", refs)
    assert cal.a1_center_xy_um[0] == pytest.approx(a1[0], abs=1e-6)
    assert cal.a1_center_xy_um[1] == pytest.approx(a1[1], abs=1e-6)
    assert cal.rotation_deg == pytest.approx(rot, abs=1e-9)
    assert cal.residual_um is not None
    assert cal.residual_um < 1e-6


def test_two_wells_are_enough_for_rotation() -> None:
    refs = _measure("96-well", (300.0, -200.0), 1.5, ["A1", "H12"])
    cal = calibrate("96-well", refs)
    assert cal.rotation_deg == pytest.approx(1.5, abs=1e-9)
    assert cal.method == "references"


def test_one_well_fixes_position_and_assumes_no_rotation() -> None:
    refs = _measure("96-well", (777.0, 888.0), 0.0, ["A1"])
    cal = calibrate("96-well", refs)
    assert cal.a1_center_xy_um == pytest.approx((777.0, 888.0))
    assert cal.rotation_deg == 0.0
    assert cal.method == "seed"
    assert cal.residual_um is None
    assert cal.scale_error is None
    assert cal.assumed == (
        "one reference well (A1): rotation 0° assumed, not measured",
    )


def test_one_well_that_is_not_a1_still_locates_the_plate() -> None:
    refs = _measure("96-well", (1000.0, 2000.0), 0.0, ["D6"])
    cal = calibrate("96-well", refs)
    assert cal.a1_center_xy_um[0] == pytest.approx(1000.0, abs=1e-6)
    assert cal.a1_center_xy_um[1] == pytest.approx(2000.0, abs=1e-6)


def test_calibration_feeds_straight_into_a_plan() -> None:
    refs = _measure("96-well", (1500.0, -900.0), 3.0, ["A1", "A12", "H1"])
    plan = calibrate("96-well", refs).to_plan(selected_wells=((0, 7), (0, 11)))
    got = {
        str(n): (p.x, p.y)
        for n, p in zip(
            plan.selected_well_names, plan.selected_well_positions, strict=True
        )
    }
    want = {
        r.well: (r.x_um, r.y_um)
        for r in _measure("96-well", (1500.0, -900.0), 3.0, ["A1", "H12"])
    }
    assert set(got) == set(want)
    for name, (x, y) in want.items():
        assert got[name][0] == pytest.approx(x, abs=1e-6)
        assert got[name][1] == pytest.approx(y, abs=1e-6)


@pytest.mark.parametrize("plate", ["6-well", "24-well", "96-well", "384-well"])
def test_works_across_plate_formats(plate: str) -> None:
    corners = suggested_refs(plate)
    refs = _measure(plate, (250.0, 125.0), -1.25, corners)
    cal = calibrate(plate, refs)
    assert cal.a1_center_xy_um[0] == pytest.approx(250.0, abs=1e-6)
    assert cal.rotation_deg == pytest.approx(-1.25, abs=1e-9)


# ---- noise and guards ----------------------------------------------------


def test_averages_out_centring_error() -> None:
    """Hand-centring is imprecise; extra wells should reduce the error, and
    the residual should report roughly how bad the centring was."""
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12", "H1", "H12"])
    jitter = [(12.0, -9.0), (-11.0, 8.0), (9.0, 10.0), (-10.0, -9.0)]
    noisy = [
        WellRef(well=r.well, x_um=r.x_um + dx, y_um=r.y_um + dy)
        for r, (dx, dy) in zip(refs, jitter, strict=True)
    ]
    cal = calibrate("96-well", noisy)
    assert abs(cal.a1_center_xy_um[0]) < 15
    assert abs(cal.a1_center_xy_um[1]) < 15
    assert abs(cal.rotation_deg) < 0.05
    assert cal.residual_um is not None
    assert 0 < cal.residual_um < 40  # reports the centring error honestly


def test_wrong_plate_type_is_refused_not_silently_fitted() -> None:
    """Measuring a 96-well plate but saying 384-well doubles every spacing:
    the fit would 'succeed' and put every position badly off."""
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12", "H1"])
    with pytest.raises(
        ValueError, match=re.escape("off the 384-well definition by +100.0%")
    ):
        calibrate("384-well", refs)


def test_misidentified_well_is_refused() -> None:
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12"])
    # user thought the second well was A6 when it was really A12
    bad = [refs[0], WellRef(well="A6", x_um=refs[1].x_um, y_um=refs[1].y_um)]
    with pytest.raises(
        ValueError, match=re.escape("off the 96-well definition by +120.0%")
    ):
        calibrate("96-well", bad)


def test_small_scale_error_is_tolerated_and_reported() -> None:
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12"])
    stretched = [
        refs[0],
        WellRef(well=refs[1].well, x_um=refs[1].x_um * 1.005, y_um=refs[1].y_um),
    ]
    cal = calibrate("96-well", stretched)
    assert cal.scale_error is not None
    assert cal.scale_error == pytest.approx(0.005)
    assert 0 < cal.scale_error < SCALE_TOLERANCE


def test_the_scale_is_signed_and_read_on_the_longest_baseline() -> None:
    """A12 is 99 mm from A1, A2 is 9 mm: only the long pair says -1 %."""
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A2", "A12"])
    shrunk = [
        refs[0],
        refs[1],
        WellRef(well="A12", x_um=refs[2].x_um * 0.99, y_um=refs[2].y_um),
    ]
    cal = calibrate("96-well", shrunk)
    assert cal.scale_error == pytest.approx(-0.01)


@pytest.mark.parametrize(
    ("factor", "accepted"),
    [(1.015, True), (0.985, True), (1.025, False), (0.975, False)],
)
def test_the_scale_tolerance_is_two_percent(factor: float, accepted: bool) -> None:
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12"])
    stretched = [
        refs[0],
        WellRef(well="A12", x_um=refs[1].x_um * factor, y_um=refs[1].y_um),
    ]
    if accepted:
        assert calibrate("96-well", stretched).scale_error == pytest.approx(factor - 1)
    else:
        with pytest.raises(ValueError, match="off the 96-well definition by"):
            calibrate("96-well", stretched)


def test_the_first_of_two_equally_long_baselines_sets_the_scale() -> None:
    """A1-B2 and A2-B1 are both the 12.7 mm diagonal; B2 is 1 % long, so only the first pair says so."""
    refs = [
        WellRef(well="A1", x_um=0.0, y_um=0.0),
        WellRef(well="A2", x_um=9000.0, y_um=0.0),
        WellRef(well="B1", x_um=0.0, y_um=-9000.0),
        WellRef(well="B2", x_um=9090.0, y_um=-9090.0),
    ]
    assert calibrate("96-well", refs).scale_error == pytest.approx(0.01, abs=1e-9)


def test_the_scale_is_checked_before_the_rotation() -> None:
    """A1 and a well at twice B1's distance, along X: 90° and +100 %: the scale is the cause named."""
    refs = [
        WellRef(well="A1", x_um=0.0, y_um=0.0),
        WellRef(well="B1", x_um=18000.0, y_um=0.0),
    ]
    with pytest.raises(
        ValueError, match=re.escape("off the 96-well definition by +100.0%")
    ):
        calibrate("96-well", refs)


def test_unknown_well_name_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=re.escape("well 'Z99' is not on the 96-well plate")
    ):
        calibrate("96-well", [WellRef(well="Z99", x_um=0.0, y_um=0.0)])


def test_an_unknown_well_is_named_as_typed() -> None:
    refs = [
        WellRef(well="A1", x_um=0.0, y_um=0.0),
        WellRef(well="z 99", x_um=1.0, y_um=1.0),
    ]
    with pytest.raises(
        ValueError, match=re.escape("well 'z 99' is not on the 96-well plate")
    ):
        calibrate("96-well", refs)


@pytest.mark.parametrize("again", ["a1", "A01", " A1 "])
def test_duplicate_reference_is_rejected(again: str) -> None:
    refs = [
        WellRef(well="A1", x_um=0.0, y_um=0.0),
        WellRef(well=again, x_um=5.0, y_um=5.0),
    ]
    with pytest.raises(ValueError, match="well A1 given twice"):
        calibrate("96-well", refs)


def test_empty_refs_rejected() -> None:
    with pytest.raises(ValueError, match="need at least one reference well"):
        calibrate("96-well", [])


def test_well_names_are_case_insensitive() -> None:
    cal = calibrate("96-well", [WellRef(well="a1", x_um=100.0, y_um=200.0)])
    assert cal.a1_center_xy_um == pytest.approx((100.0, 200.0))


def test_the_result_names_its_method_and_references() -> None:
    truth = _measure("96-well", (1000.0, 2000.0), 2.0, ["A1", "A12", "H1"])
    typed = ["a1", "A 12", "h01"]
    refs = [
        WellRef(well=name, x_um=t.x_um, y_um=t.y_um)
        for name, t in zip(typed, truth, strict=True)
    ]
    cal = calibrate("96-well", refs)
    assert cal.plate == "96-well"
    assert cal.method == "references"
    assert cal.n_refs == 3
    assert [r.well for r in cal.refs] == ["A1", "A12", "H1"]
    assert [(r.x_um, r.y_um) for r in cal.refs] == [(r.x_um, r.y_um) for r in refs]
    assert cal.assumed == ()


# ---- the rotation bound --------------------------------------------------


def test_two_references_the_same_distance_apart_cannot_turn_the_plate() -> None:
    """A1 and A2 typed as B1: a rigid fit turns 90° with no residual (FM-121).

    Neither the scale check nor the residual can see it, so the rotation has
    its own bound; with the bound lifted the fit is exact, which says why.
    """
    refs = [
        WellRef(well="A1", x_um=0.0, y_um=0.0),
        WellRef(well="B1", x_um=9000.0, y_um=0.0),
    ]
    with pytest.raises(ValueError, match="beyond ±10°"):
        calibrate("96-well", refs)
    cal = calibrate("96-well", refs, max_rotation_deg=180.0)
    assert cal.rotation_deg == pytest.approx(90.0, abs=1e-9)
    assert cal.residual_um is not None
    assert cal.residual_um < 1e-6
    assert cal.scale_error == pytest.approx(0.0, abs=1e-12)


def test_a_rotation_beyond_the_bound_is_refused_unless_allowed() -> None:
    refs = _measure("96-well", (10.0, 20.0), 17.5, ["A1", "A12", "H1"])
    with pytest.raises(
        ValueError, match=re.escape("the fit turns the plate by 17.5°, beyond ±10°")
    ):
        calibrate("96-well", refs)
    cal = calibrate("96-well", refs, max_rotation_deg=20.0)
    assert cal.rotation_deg == pytest.approx(17.5, abs=1e-9)
    assert cal.a1_center_xy_um == pytest.approx((10.0, 20.0), abs=1e-6)


def test_the_bound_is_on_the_size_of_the_rotation_not_its_sign() -> None:
    refs = _measure("96-well", (0.0, 0.0), -17.5, ["A1", "A12", "H1"])
    with pytest.raises(ValueError, match=re.escape("turns the plate by -17.5°")):
        calibrate("96-well", refs)
    cal = calibrate("96-well", refs, max_rotation_deg=20.0)
    assert cal.rotation_deg == pytest.approx(-17.5, abs=1e-9)


def test_a_rotation_at_the_bound_is_accepted() -> None:
    refs = _measure("96-well", (0.0, 0.0), 9.9, ["A1", "A12", "H1"])
    assert calibrate("96-well", refs).rotation_deg == pytest.approx(9.9, abs=1e-9)


# ---- the round trip against useq's forward model -------------------------

_NAMES = {plate: tuple(nominal_offsets(plate)) for plate in PLATES}


@st.composite
def _placements(
    draw: st.DrawFn,
) -> tuple[str, tuple[float, float], float, list[str], list[tuple[float, float]]]:
    plate = draw(st.sampled_from(PLATES))
    names = draw(
        st.lists(st.sampled_from(_NAMES[plate]), min_size=1, max_size=4, unique=True)
    )
    a1 = (
        draw(st.floats(-50_000.0, 50_000.0)),
        draw(st.floats(-50_000.0, 50_000.0)),
    )
    # Within ±8°, not ±10°: a fit of a draw near 10° with jitter on a short baseline
    # lands beyond the bound and is (rightly) refused.
    rotation = draw(st.floats(-8.0, 8.0))
    jitter = draw(
        st.lists(
            st.tuples(st.floats(-12.0, 12.0), st.floats(-12.0, 12.0)),
            min_size=len(names),
            max_size=len(names),
        )
    )
    return plate, a1, rotation, names, jitter


@settings(max_examples=200, deadline=None)
@given(_placements())
def test_calibration_round_trips_through_useqs_forward_model(
    placement: tuple[
        str, tuple[float, float], float, list[str], list[tuple[float, float]]
    ],
) -> None:
    plate, a1, rotation, names, jitter = placement
    truth = _measure(plate, a1, rotation, names)
    refs = [
        WellRef(well=t.well, x_um=t.x_um + dx, y_um=t.y_um + dy)
        for t, (dx, dy) in zip(truth, jitter, strict=True)
    ]
    cal = calibrate(plate, refs)
    count = len(refs)
    worst = max(math.hypot(dx, dy) for dx, dy in jitter)
    if count == 1:
        assert cal.method == "seed"
        x, y = cal.well_center_um(refs[0].well)
        assert math.dist((x, y), (refs[0].x_um, refs[0].y_um)) < 1e-6
        return
    assert cal.residual_um is not None
    assert cal.residual_um <= math.sqrt(count) * worst + 1e-6
    for t in truth:
        fitted = cal.well_center_um(t.well)
        assert (
            math.dist(fitted, (t.x_um, t.y_um)) <= (math.sqrt(count) + 1) * worst + 1e-6
        )


# ------------------------------------------ finding a centre you cannot see


def test_opposite_walls_give_the_centre_with_no_diameter_assumed() -> None:
    """At 40x a well is not visible: the field is ~333 µm, the well 6400.

    So the centre is never eyeballed: touch one wall, touch the opposite
    one, take the midpoint.
    """
    e = centre_from_walls(
        "96-well",
        "A1",
        [
            _touch("A1", "left", 1000 - 3200, 0),
            _touch("A1", "right", 1000 + 3200, 0),
            _touch("A1", "bottom", 0, 2000 - 3200),
            _touch("A1", "top", 0, 2000 + 3200),
        ],
    )
    assert (e.x_um, e.y_um) == pytest.approx((1000.0, 2000.0))
    assert e.measured_width_um == pytest.approx(6400.0)
    assert e.measured_height_um == pytest.approx(6400.0)
    assert e.assumed_axes == ()
    assert e.well == "A1"


def test_a_single_wall_per_axis_uses_the_nominal_half_size() -> None:
    e = centre_from_walls(
        "96-well",
        "a1",
        [_touch("A1", "left", -2200, 0), _touch("a1", "bottom", 0, -1200)],
    )
    assert (e.x_um, e.y_um) == pytest.approx((1000.0, 2000.0))
    assert e.assumed_axes == ("x", "y")
    assert e.measured_width_um is None
    assert e.measured_height_um is None
    assert e.well == "A1"


@pytest.mark.parametrize(
    ("side", "x", "y", "expected"),
    [
        ("left", 100.0, 0.0, (3300.0, 0.0)),
        ("right", 100.0, 0.0, (-3100.0, 0.0)),
    ],
)
def test_one_x_wall_is_a_half_size_from_the_centre(
    side: str, x: float, y: float, expected: tuple[float, float]
) -> None:
    e = centre_from_walls(
        "96-well", "A1", [_touch("A1", side, x, y), _touch("A1", "top", 0.0, 0.0)]
    )
    assert e.x_um == pytest.approx(expected[0])
    assert e.y_um == pytest.approx(-3200.0)  # top is +Y: the centre is below it
    assert e.assumed_axes == ("x", "y")


def test_a_single_wall_is_in_stage_units() -> None:
    """useq holds sizes in mm: a 384-well well is 3400 µm, so half is 1700."""
    e = centre_from_walls(
        "384-well",
        "A1",
        [_touch("A1", "left", 0.0, 0.0), _touch("A1", "bottom", 0.0, 0.0)],
    )
    assert (e.x_um, e.y_um) == pytest.approx((1700.0, 1700.0))


def test_wall_touches_read_each_axis_from_its_own_coordinate(anisotropic: str) -> None:
    """A touch's other coordinate is never read, and each axis has its own size (FM-73)."""
    cx, cy = 24000.0 + 500.0, -7000.0 + 300.0  # B3, a little off the nominal
    far = 1e6
    pair = centre_from_walls(
        anisotropic,
        "B3",
        [
            _touch("B3", "left", cx - 5000.0, far),
            _touch("B3", "right", cx + 5000.0, -far),
            _touch("B3", "bottom", far, cy - 2500.0),
            _touch("B3", "top", -far, cy + 2500.0),
        ],
    )
    assert (pair.x_um, pair.y_um) == pytest.approx((cx, cy))
    assert pair.measured_width_um == pytest.approx(10000.0)
    assert pair.measured_height_um == pytest.approx(5000.0)

    one_wall = {
        "left": centre_from_walls(
            anisotropic,
            "B3",
            [
                _touch("B3", "left", cx - 5000.0, far),
                _touch("B3", "bottom", far, cy - 2500.0),
            ],
        ),
        "right": centre_from_walls(
            anisotropic,
            "B3",
            [
                _touch("B3", "right", cx + 5000.0, far),
                _touch("B3", "top", far, cy + 2500.0),
            ],
        ),
    }
    for e in one_wall.values():
        assert (e.x_um, e.y_um) == pytest.approx((cx, cy))
        assert e.assumed_axes == ("x", "y")


def test_touching_the_wrong_well_is_caught_by_the_implied_size() -> None:
    """The free sanity check: two walls imply a size, and a size off by more than 10 % is refused."""
    touches = [
        _touch("A1", "left", -2200, 0),
        _touch("A1", "right", 7000, 0),
        _touch("A1", "bottom", 0, -1200),
        _touch("A1", "top", 0, 5200),
    ]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "A1's left and right walls are 9200 µm apart, a 96-well well is 6400 µm "
            "(+43.8%): a touch of another well, or off the wall; touch them again"
        ),
    ):
        centre_from_walls("96-well", "A1", touches)


def test_a_wrong_y_pair_is_refused_in_its_own_words() -> None:
    touches = [
        _touch("A1", "left", -3200, 0),
        _touch("A1", "right", 3200, 0),
        _touch("A1", "bottom", 0, 0),
        _touch("A1", "top", 0, 9000),
    ]
    with pytest.raises(
        ValueError,
        match=re.escape("A1's top and bottom walls are 9000 µm apart, a 96-well well"),
    ):
        centre_from_walls("96-well", "A1", touches)


def test_the_size_tolerance_is_ten_percent() -> None:
    assert DIAMETER_TOLERANCE == 0.10
    inside = centre_from_walls(
        "96-well",
        "A1",
        [
            _touch("A1", "left", 0, 0),
            _touch("A1", "right", 6400 * 1.09, 0),
            _touch("A1", "bottom", 0, 0),
            _touch("A1", "top", 0, 6400 * 0.91),
        ],
    )
    assert inside.measured_width_um == pytest.approx(6400 * 1.09)
    with pytest.raises(ValueError, match="left and right walls"):
        centre_from_walls(
            "96-well",
            "A1",
            [
                _touch("A1", "left", 0, 0),
                _touch("A1", "right", 6400 * 1.11, 0),
                _touch("A1", "bottom", 0, 0),
                _touch("A1", "top", 0, 6400),
            ],
        )


def test_swapped_opposite_walls_are_refused() -> None:
    """Right of left by -6400 µm: a size, signed, that no well has."""
    touches = [
        _touch("A1", "left", 3200, 0),
        _touch("A1", "right", -3200, 0),
        _touch("A1", "bottom", 0, -3200),
        _touch("A1", "top", 0, 3200),
    ]
    with pytest.raises(
        ValueError, match=re.escape("A1's left and right walls are -6400 µm apart")
    ):
        centre_from_walls("96-well", "A1", touches)


def test_an_axis_with_no_wall_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no Y wall for A1: give top and/or bottom"):
        centre_from_walls(
            "96-well",
            "A1",
            [_touch("A1", "left", 0, 0), _touch("A1", "right", 6400, 0)],
        )
    with pytest.raises(ValueError, match="no X wall for A1: give left and/or right"):
        centre_from_walls(
            "96-well",
            "A1",
            [_touch("A1", "bottom", 0, 0), _touch("A1", "top", 0, 6400)],
        )
    with pytest.raises(ValueError, match="give at least one wall touch for A1"):
        centre_from_walls("96-well", "A1", [])


def test_a_side_touched_twice_is_refused() -> None:
    touches = [
        _touch("A1", "left", 0, 0),
        _touch("a1", "left", 10, 0),
        _touch("A1", "bottom", 0, 0),
    ]
    with pytest.raises(ValueError, match="A1: the left wall is given twice"):
        centre_from_walls("96-well", "A1", touches)


def test_a_touch_of_another_well_is_refused() -> None:
    touches = [_touch("A1", "left", 0, 0), _touch("B2", "bottom", 0, 0)]
    with pytest.raises(ValueError, match="a touch of B2 was given for A1"):
        centre_from_walls("96-well", "A1", touches)


def test_a_well_that_is_not_on_the_plate_is_refused_first() -> None:
    with pytest.raises(
        ValueError, match=re.escape("well 'Z99' is not on the 96-well plate")
    ):
        centre_from_walls("96-well", "Z99", [])


def test_an_edge_centre_feeds_the_calibration_directly() -> None:
    refs = []
    for name, cx, cy in [
        ("A1", 1000, 2000),
        ("A12", 100000, 2000),
        ("H1", 1000, -61000),
    ]:
        e = centre_from_walls(
            "96-well",
            name,
            [
                _touch(name, "left", cx - 3200, 0),
                _touch(name, "right", cx + 3200, 0),
                _touch(name, "bottom", 0, cy - 3200),
                _touch(name, "top", 0, cy + 3200),
            ],
        )
        refs.append(WellRef(well=e.well, x_um=e.x_um, y_um=e.y_um))
    cal = calibrate("96-well", refs)
    assert cal.residual_um is not None
    assert cal.residual_um < 1.0
    assert cal.a1_center_xy_um == pytest.approx((1000.0, 2000.0))


def test_the_residual_threshold_matches_what_a_scan_can_absorb() -> None:
    """A registration error shifts the scan grid; it does not compound.

    50 µm was the old threshold and would have flagged a perfectly usable
    registration as suspect.
    """
    assert GOOD_RESIDUAL_UM >= 100.0
    # a few hundred µm of error is well inside a well's spare margin
    assert GOOD_RESIDUAL_UM < 6400.0 / 10


def test_the_constants_are_the_designs() -> None:
    assert SCALE_TOLERANCE == 0.02
    assert GOOD_RESIDUAL_UM == 300.0
    assert algorithm.MAX_ROTATION_DEG == 10.0


# ---------------------------------------------- the guided procedure


def test_one_rough_well_predicts_every_other_well() -> None:
    """A seed turns 'hunt for an invisible well' into 'press go, then nudge'."""
    seeded = calibrate("96-well", [WellRef(well="A1", x_um=12345, y_um=-6789)])
    assert seeded.well_center_um("A2") == pytest.approx((12345 + 9000, -6789))
    assert seeded.well_center_um("B1") == pytest.approx((12345, -6789 - 9000))


def test_a_seed_on_any_well_works_not_just_a1() -> None:
    seeded = calibrate("96-well", [WellRef(well="D6", x_um=50000, y_um=-20000)])
    assert seeded.well_center_um("D6") == pytest.approx((50000, -20000))
    # and A1 is placed consistently
    assert seeded.well_center_um("A1") == pytest.approx(
        (50000 - 5 * 9000, -20000 + 3 * 9000)
    )


def test_wall_targets_are_a_half_size_from_the_centre() -> None:
    cal = calibrate("96-well", [WellRef(well="A1", x_um=1000, y_um=2000)])
    cx, cy = cal.well_center_um("A1")
    for side, expect in (
        ("right", (cx + 3200, cy)),
        ("left", (cx - 3200, cy)),
        ("top", (cx, cy + 3200)),
        ("bottom", (cx, cy - 3200)),
    ):
        assert wall_position(cal, "A1", side) == pytest.approx(expect)


def test_wall_targets_follow_the_fitted_rotation() -> None:
    """A rotated plate's walls are not on the stage axes."""
    rotated = calibrate(
        "96-well",
        [
            WellRef(well="A1", x_um=0, y_um=0),
            WellRef(
                well="A12",
                x_um=99000 * math.cos(math.radians(5)),
                y_um=99000 * math.sin(math.radians(5)),
            ),
        ],
    )
    assert rotated.rotation_deg == pytest.approx(5.0, abs=0.01)
    x, y = wall_position(rotated, "A1", "right")
    assert y > 100, "the wall target ignored the plate rotation"
    assert (x, y) == pytest.approx(
        (3200 * math.cos(math.radians(5)), 3200 * math.sin(math.radians(5))), abs=0.1
    )


def test_wall_position_on_an_anisotropic_plate_uses_each_axis_s_size(
    anisotropic: str,
) -> None:
    """B3 of a plate turned by 3°: 12 mm and 7 mm pitches, half-sizes 5000 and 2500 µm."""
    theta = math.radians(3.0)
    cos, sin = math.cos(theta), math.sin(theta)
    a1 = (1000.0, 2000.0)
    result = _result(anisotropic, a1, 3.0)
    # B3 is two columns of 12 000 µm along and one row of 7 000 µm down the plate's axes.
    centre = (
        a1[0] + cos * 24000.0 - sin * -7000.0,
        a1[1] + sin * 24000.0 + cos * -7000.0,
    )
    for side, (ox, oy) in {
        "left": (-5000.0, 0.0),
        "right": (5000.0, 0.0),
        "top": (0.0, 2500.0),
        "bottom": (0.0, -2500.0),
    }.items():
        expected = (centre[0] + cos * ox - sin * oy, centre[1] + sin * ox + cos * oy)
        assert wall_position(result, "B3", side) == pytest.approx(expected, abs=1e-6)


def test_an_unknown_well_is_refused_rather_than_placed() -> None:
    cal = calibrate("96-well", [WellRef(well="A1", x_um=0, y_um=0)])
    with pytest.raises(
        ValueError, match=re.escape("well 'Z99' is not on the 96-well plate")
    ):
        wall_position(cal, "Z99", "left")


def test_an_unknown_side_is_refused() -> None:
    cal = calibrate("96-well", [WellRef(well="A1", x_um=0, y_um=0)])
    with pytest.raises(
        ValueError,
        match=re.escape("side 'middle' is not one of left, right, top, bottom"),
    ):
        wall_position(cal, "A1", "middle")


def test_capturing_references_replaces_the_seed_with_a_real_fit() -> None:
    """Each capture makes the next well easier to find than the last."""
    seeded = calibrate(
        "96-well", [WellRef(well="A1", x_um=1200, y_um=1800)]
    )  # 200 µm out
    rough = seeded.well_center_um("H12")

    fitted = calibrate(
        "96-well",
        [
            WellRef(well="A1", x_um=1000, y_um=2000),
            WellRef(well="A12", x_um=100000, y_um=2000),
            WellRef(well="H1", x_um=1000, y_um=-61000),
        ],
    )
    good = fitted.well_center_um("H12")
    truth = (100000, -61000)

    assert math.dist(good, truth) < math.dist(rough, truth)
    assert math.dist(good, truth) < 1.0


# ------------------------------------------------------------- nearest well


def test_nearest_well_names_the_well_and_its_distance() -> None:
    """A point 1000 µm from C5's centre along the plate's diagonal, on a plate turned by 2°."""
    theta = math.radians(2.0)
    cos, sin = math.cos(theta), math.sin(theta)
    a1 = (1000.0, -500.0)
    result = _result("96-well", a1, 2.0)
    # C5 is four columns of 9 mm along and two rows of 9 mm down the plate's axes.
    centre = (
        a1[0] + cos * 36000.0 - sin * -18000.0,
        a1[1] + sin * 36000.0 + cos * -18000.0,
    )
    step = 1000.0 / math.sqrt(2.0)
    point = (
        centre[0] + cos * step - sin * step,
        centre[1] + sin * step + cos * step,
    )
    name, distance = nearest_well(result, *point)
    assert name == "C5"
    assert distance == pytest.approx(1000.0, abs=1e-6)


def test_nearest_well_reads_the_y_pitch_for_y(anisotropic: str) -> None:
    """4000 µm below B3 is nearer C3 (7000 µm below) than B3: with the X pitch read for Y it would be B3 (FM-73)."""
    result = _result(anisotropic, (0.0, 0.0), 0.0)
    name, distance = nearest_well(result, 24000.0, -7000.0 - 4000.0)
    assert name == "C3"
    assert distance == pytest.approx(3000.0, abs=1e-6)


def test_nearest_well_has_no_cut_off() -> None:
    result = _result("96-well", (0.0, 0.0), 0.0)
    name, distance = nearest_well(result, -1e6, 0.0)
    assert name == "A1"
    assert distance == pytest.approx(1e6)


def test_nearest_well_on_a_tie_is_the_first_in_useq_s_order() -> None:
    """Half way between A1 and A2: A1 comes first."""
    result = _result("96-well", (0.0, 0.0), 0.0)
    name, distance = nearest_well(result, 4500.0, 0.0)
    assert name == "A1"
    assert distance == pytest.approx(4500.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nearest_well_refuses_a_point_that_is_not_finite(bad: float) -> None:
    result = _result("96-well", (0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="the point must be finite"):
        nearest_well(result, bad, 0.0)
    with pytest.raises(ValueError, match="the point must be finite"):
        nearest_well(result, 0.0, bad)
