"""The fields a selection chose, as a plan's fields (m3 §28.4).

Pure: stops built as ``RoutePosition``s on a 96-well plate turned 1.5°, the
wells' shapes from ``well_geometry`` of its plan, and the shift of the
``plate_check_stand`` profile from the 10X to the 40X, ``(-37.5, 12.25)``, which
is anisotropic so that a swapped axis or a dropped sign is seen (FM-73, FM-203).
Every expectation is written from the numbers (FM-124): the stops lie on round
positions, so a centre is the stop plus the shift exactly.
"""

from __future__ import annotations

import math

import pytest
import useq
from pydantic import ValidationError

from smc.plugins.scan_plan.plugin import ScanPlanParams
from smc.plugins.scan_plan.selection import (
    MAX_FIELDS_PER_WELL,
    ChosenFields,
    chosen_fields,
)
from smc.plugins.slots import PlannedField, RoutePosition, WellGeometry, well_geometry

SHIFT = (-37.5, 12.25)
PLAN = useq.WellPlatePlan(
    plate=useq.WellPlate.from_str("96-well"),
    a1_center_xy=(10000.0, -5000.0),
    rotation=1.5,
)
#: The 96-well plate's wells are discs of 6.4 mm.
RADIUS_UM = 3200.0


def _geometry(well: str) -> WellGeometry:
    return well_geometry(PLAN, well)


def _centre(well: str) -> tuple[float, float]:
    """The well's centre rounded to 10 µm: a round point well inside the well."""
    x_um, y_um = _geometry(well).center_um
    return round(x_um, -1), round(y_um, -1)


def _stop(
    order: int, well: str, k: int, dx_um: float = 0.0, dy_um: float = 0.0
) -> RoutePosition:
    x_um, y_um = _centre(well)
    return RoutePosition(
        order=order,
        well=well,
        name=f"{well}_r{k:03d}",
        x_um=x_um + dx_um,
        y_um=y_um + dy_um,
        z_um=1500.0,
        pfs_offset=512.0,
        counts={"cell": 3},
    )


def _chosen(
    positions: list[RoutePosition],
    wells: tuple[str, ...],
    *,
    shift_um: tuple[float, float] = SHIFT,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> ChosenFields:
    return chosen_fields(
        positions,
        wells,
        shift_um=shift_um,
        geometry_of=_geometry,
        xy_limits_um=limits,
    )


#: B3 and B2 interleaved by the route's order, given out of that order.
INTERLEAVED = [
    _stop(3, "B2", 1, 400.0, -300.0),
    _stop(0, "B3", 0, -200.0, 100.0),
    _stop(2, "B3", 1, 500.0, 600.0),
    _stop(1, "B2", 0, 0.0, 0.0),
]


def test_a_wells_fields_are_its_stops_in_the_routes_order_and_the_wells_the_plans() -> (
    None
):
    chosen = _chosen(INTERLEAVED, ("B2", "B3"))
    assert chosen.wells == ("B2", "B3")
    # B2 whole, then B3, each in the route's order (B2: orders 1, 3; B3: 0, 2).
    assert [field.stop for field in chosen.fields] == [
        "B2_r000",
        "B2_r001",
        "B3_r000",
        "B3_r001",
    ]
    # And the wells in the order given, not the route's first well.
    reversed_ = _chosen(INTERLEAVED, ("B3", "B2"))
    assert reversed_.wells == ("B3", "B2")
    assert [field.stop for field in reversed_.fields] == [
        "B3_r000",
        "B3_r001",
        "B2_r000",
        "B2_r001",
    ]


def test_a_named_well_with_no_stop_is_left_out_and_listed() -> None:
    chosen = _chosen(INTERLEAVED, ("C4", "B2", "A1", "B3"))
    assert chosen.wells == ("B2", "B3")
    assert chosen.wells_without_fields == ("C4", "A1")
    assert {field.well for field in chosen.fields} == {"B2", "B3"}
    assert chosen.fields_left_out == 0
    assert chosen.wells_left_out == ()


def test_stops_of_a_well_not_named_are_counted_and_their_wells_listed() -> None:
    positions = [*INTERLEAVED, _stop(4, "D5", 0), _stop(5, "C4", 0), _stop(6, "D5", 1)]
    chosen = _chosen(positions, ("B3",))
    assert chosen.wells == ("B3",)
    assert [field.stop for field in chosen.fields] == ["B3_r000", "B3_r001"]
    # Two of B2, two of D5, one of C4; their wells in the order first met (route order).
    assert chosen.fields_left_out == 5
    assert chosen.wells_left_out == ("B2", "D5", "C4")


def test_every_centre_is_its_stop_plus_the_shift() -> None:
    chosen = _chosen(INTERLEAVED, ("B2", "B3"))
    by_stop = {stop.name: stop for stop in INTERLEAVED}
    for field in chosen.fields:
        stop = by_stop[field.stop or ""]
        assert (field.x_um, field.y_um) == (stop.x_um - 37.5, stop.y_um + 12.25)
    # One written out in full: B2_r001 lies at B2's round centre plus (400, -300).
    b2_x, b2_y = _centre("B2")
    assert (chosen.fields[1].x_um, chosen.fields[1].y_um) == (
        b2_x + 400.0 - 37.5,
        b2_y - 300.0 + 12.25,
    )


def test_a_zero_shift_keeps_the_stops_where_they_are() -> None:
    chosen = _chosen(INTERLEAVED, ("B2",), shift_um=(0.0, 0.0))
    assert [(f.x_um, f.y_um) for f in chosen.fields] == [
        (INTERLEAVED[3].x_um, INTERLEAVED[3].y_um),
        (INTERLEAVED[0].x_um, INTERLEAVED[0].y_um),
    ]


def test_a_centre_outside_its_well_after_the_shift_is_refused_by_its_stop() -> None:
    # 20 µm inside the wall on -X before the shift; the shift's -37.5 µm takes it
    # 17.5 µm past the wall (and its +12.25 µm in Y only further).
    edge = _stop(1, "B3", 1, -(RADIUS_UM - 20.0), 0.0)
    assert _geometry("B3").contains(edge.x_um, edge.y_um)
    x_um, y_um = edge.x_um - 37.5, edge.y_um + 12.25
    assert not _geometry("B3").contains(x_um, y_um)
    positions = [_stop(0, "B3", 0), edge, _stop(2, "B3", 2, -(RADIUS_UM + 100.0))]
    with pytest.raises(ValueError, match=r"^(field|the selection) ") as refused:
        _chosen(positions, ("B3",))
    assert str(refused.value) == (
        f"field B3_r001 of the selection lies at ({x_um:.1f}, {y_um:.1f}) µm in the "
        "plan's frame, outside well B3: check the offsets of the profile's "
        "[objectives], or run route again"
    )


def test_the_first_offending_stop_in_visiting_order_is_named() -> None:
    # Both B2's and B3's stops are out; the plan visits B3 first.
    positions = [
        _stop(0, "B2", 0, RADIUS_UM + 500.0),
        _stop(1, "B3", 0, 0.0, RADIUS_UM + 500.0),
    ]
    with pytest.raises(ValueError, match=r"^field B3_r000 of the selection"):
        _chosen(positions, ("B3", "B2"))


def test_a_centre_outside_the_soft_limits_is_refused_and_one_on_the_limit_is_kept() -> (
    None
):
    a = _stop(0, "B2", 0)
    b = _stop(1, "B3", 0)
    # B2's field lies exactly on x_lo and y_lo, B3's on x_hi and y_hi (the plate
    # turns by +1.5°, so B3 is right of and above B2): the layer's own test
    # (`lo <= v <= hi`) keeps both.
    x_lo, y_lo = a.x_um - 37.5, a.y_um + 12.25
    x_hi, y_hi = b.x_um - 37.5, b.y_um + 12.25
    assert x_lo < x_hi
    assert y_lo < y_hi
    kept = _chosen([a, b], ("B2", "B3"), limits=((x_lo, x_hi), (y_lo, y_hi)))
    assert [field.stop for field in kept.fields] == ["B2_r000", "B3_r000"]

    with pytest.raises(ValueError, match=r"^(field|the selection) ") as refused:
        _chosen([a, b], ("B2", "B3"), limits=((x_lo, x_hi - 0.5), (y_lo, y_hi)))
    assert str(refused.value) == (
        f"field B3_r000 of the selection lies at ({x_hi:.1f}, {y_hi:.1f}) µm, "
        f"outside the stage's soft limits x [{x_lo:g}, {x_hi - 0.5:g}], "
        f"y [{y_lo:g}, {y_hi:g}] µm"
    )
    # Each bound on its own.
    for limits, stop in (
        (((x_lo, x_hi), (y_lo, y_hi - 0.5)), "B3_r000"),
        (((x_lo + 0.5, x_hi), (y_lo, y_hi)), "B2_r000"),
        (((x_lo, x_hi), (y_lo + 0.5, y_hi)), "B2_r000"),
    ):
        with pytest.raises(ValueError, match=rf"^field {stop} .* soft limits"):
            _chosen([a, b], ("B2", "B3"), limits=limits)


def test_no_limits_checks_no_travel() -> None:
    # A stop far from the stage's usual travel is kept when the stage has none.
    chosen = _chosen(INTERLEAVED, ("B2", "B3"), limits=None)
    assert len(chosen.fields) == 4


def test_more_than_ten_thousand_fields_in_a_well_are_refused() -> None:
    assert MAX_FIELDS_PER_WELL == 10_000
    many = [_stop(i, "B2", i) for i in range(10_001)]
    kept = _chosen(many[:10_000], ("B2",))
    assert len(kept.fields) == 10_000
    with pytest.raises(ValueError, match=r"^(field|the selection) ") as refused:
        _chosen(many, ("B2",))
    assert str(refused.value) == (
        "the selection holds 10001 fields in B2, more than 10000 a well"
    )
    # Stops of a well not named do not count.
    assert len(_chosen([*many[:10_000], _stop(10_000, "B3", 0)], ("B2",)).fields) == (
        10_000
    )


def test_fields_are_named_by_well_and_index_and_keep_their_stop() -> None:
    chosen = _chosen(INTERLEAVED, ("B3", "B2"))
    assert [(f.well, f.index, f.name, f.stop) for f in chosen.fields] == [
        ("B3", 0, "B3_0000", "B3_r000"),
        ("B3", 1, "B3_0001", "B3_r001"),
        ("B2", 0, "B2_0000", "B2_r000"),
        ("B2", 1, "B2_0001", "B2_r001"),
    ]
    assert all(isinstance(f, PlannedField) for f in chosen.fields)


def test_no_stop_in_the_named_wells_returns_no_field() -> None:
    chosen = _chosen(INTERLEAVED, ("A1", "H12"))
    assert chosen.wells == ()
    assert chosen.fields == ()
    assert chosen.wells_without_fields == ("A1", "H12")
    assert chosen.fields_left_out == 4
    assert chosen.wells_left_out == ("B3", "B2")


def test_a_rotation_of_the_plate_is_in_the_wells_shape() -> None:
    # Sanity on the fixture: the plate is turned, so B2 is not at A1 + (9000, -9000).
    x_um, y_um = _geometry("B2").center_um
    assert not math.isclose(x_um, 19000.0, abs_tol=1.0)
    assert not math.isclose(y_um, -14000.0, abs_tol=1.0)


# --- the parameters' third validator (m3 §28.5) ---------------------------------------


_GRID = {
    "rows": 2,
    "columns": 3,
    "coverage": 0.5,
    "overlap": 0.2,
    "clip_to_well": False,
}
_DEFAULTS = {
    "coverage": ScanPlanParams.model_fields["coverage"].default,
    "overlap": ScanPlanParams.model_fields["overlap"].default,
    "clip_to_well": ScanPlanParams.model_fields["clip_to_well"].default,
}


@pytest.mark.parametrize(
    ("given", "names"),
    [
        ({"rows": 2, "columns": 3}, "rows, columns"),
        ({"coverage": 0.5}, "coverage"),
        ({"overlap": 0.2}, "overlap"),
        ({"clip_to_well": False}, "clip_to_well"),
        (_GRID, "rows, columns, coverage, overlap, clip_to_well"),
    ],
    ids=["rows-columns", "coverage", "overlap", "clip_to_well", "all"],
)
def test_fields_from_refuses_the_grids_parameters_given_with_it(
    given: dict[str, object], names: str
) -> None:
    with pytest.raises(ValidationError) as refused:
        ScanPlanParams.model_validate(
            {"wells": "all", "fields_from": "run-10x.json", **given}
        )
    assert (
        f"fields_from plans the fields a selection chose: {names} are not read "
        "with it; leave them unset"
    ) in str(refused.value)
    # Without fields_from, the same parameters plan a grid.
    ScanPlanParams.model_validate({"wells": "all", **given})


@pytest.mark.parametrize("name", list(_DEFAULTS))
def test_fields_from_accepts_the_grids_parameters_at_their_defaults(name: str) -> None:
    params = ScanPlanParams.model_validate(
        {"wells": "all", "fields_from": "run-10x.json", name: _DEFAULTS[name]}
    )
    assert params.fields_from is not None
    assert params.fields_from.as_posix() == "run-10x.json"
