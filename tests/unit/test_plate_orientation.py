"""A plate's orientation in its holder: the turn is a renaming (m3 §24 D2, D6).

FM-73: the plates here have grids that are not square (24-well is 4 x 6, 96-well
8 x 12) and positions that differ in X and Y, so a swap of rows and columns, or a
mirror in place of a turn, fails. Positions are checked against ``useq``'s own
plan turned 180°, never against a hand-flipped sign.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import useq
from pydantic import ValidationError

from smc.plugins.geometry import wrap_deg
from smc.plugins.slots.plate import (
    ORIENTATION_DEG,
    PlateResult,
    WallFrame,
    WallImages,
    WellFit,
    WellRef,
    turned_well,
)

A1 = (1234.0, -567.0)
TILT_DEG = 13.0

#: Every plate ``useq`` knows here, the lab's ``greiner-24-well`` included (the
#: import of the slot registers it).
PLATES = sorted(useq.registered_well_plate_keys())


def _result(plate: str = "24-well", **update: object) -> PlateResult:
    fields: dict[str, object] = {
        "plate": plate,
        "a1_center_xy_um": A1,
        "rotation_deg": TILT_DEG,
        "method": "references",
        "residual_um": 12.5,
        "scale_error": 0.0007,
        "n_refs": 3,
        "objective": "10x",
        "assumed": ("a line of the fit, about B3",),
    }
    fields.update(update)
    return PlateResult.model_validate(fields)


def test_greiner_is_among_the_plates_checked() -> None:
    assert "greiner-24-well" in PLATES
    assert len(PLATES) >= 10


@pytest.mark.parametrize("plate", PLATES)
def test_turned_well_swaps_row_and_column_on_every_registered_plate(
    plate: str,
) -> None:
    names = np.asarray(useq.WellPlate.from_str(plate).all_well_names)
    rows, cols = names.shape
    for row in range(rows):
        for col in range(cols):
            turned = turned_well(plate, str(names[row, col]))
            assert turned == names[rows - 1 - row, cols - 1 - col]


@pytest.mark.parametrize(
    ("plate", "well", "expected"),
    [
        ("24-well", "b3", "C4"),
        ("24-well", "B3", "C4"),
        ("24-well", "A1", "D6"),
        ("96-well", "A1", "H12"),
        ("96-well", "B 03", "G10"),
        ("1536-well", "A1", "AF48"),
        ("greiner-24-well", "B3", "C4"),
    ],
)
def test_turned_well_names_the_well_at_the_same_place(
    plate: str, well: str, expected: str
) -> None:
    assert turned_well(plate, well) == expected


def test_turned_well_refuses_a_well_that_is_not_on_the_plate() -> None:
    with pytest.raises(ValueError, match=r"'E1' is not on the 24-well plate"):
        turned_well("24-well", "E1")


@pytest.mark.parametrize("plate", PLATES)
def test_a_turn_keeps_every_well_position_on_every_registered_plate(
    plate: str,
) -> None:
    fit = _result(plate)
    turned = fit.turned()
    # useq's own plan of the turned plate: each name in it lies where the fit had
    # the turned name, within 1e-6 µm (measured 4e-11).
    turned_plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=turned.a1_center_xy_um,
        rotation=180.0 + TILT_DEG,
    )
    positions = turned_plan.all_well_positions
    assert positions
    for position in positions:
        name = str(position.name)
        before = fit.well_center_um(turned_well(plate, name))
        assert (position.x, position.y) == pytest.approx(before, abs=1e-6)
        assert turned.well_center_um(name) == pytest.approx(before, abs=1e-6)
    assert turned.rotation_deg == pytest.approx(wrap_deg(TILT_DEG + 180.0))
    assert turned.orientation == "turned-180"


def test_a_turn_moves_a1_to_the_far_corner_of_a_plate_that_is_not_square() -> None:
    fit = _result("24-well", rotation_deg=0.0, a1_center_xy_um=(0.0, 0.0))
    turned = fit.turned()
    # D6 is at row 3, column 5: 5 pitches towards +X, 3 towards -Y (19 mm).
    assert turned.a1_center_xy_um == pytest.approx((5 * 19000.0, -3 * 19000.0))
    assert turned.rotation_deg == 180.0
    assert turned.well_center_um("D6") == pytest.approx((0.0, 0.0), abs=1e-6)
    # B3 of the turned plate is at C4's old place, not B3's or a mirror of it.
    assert turned.well_center_um("B3") == pytest.approx(
        fit.well_center_um("C4"), abs=1e-6
    )


def test_turning_twice_gives_the_fit_back() -> None:
    fit = _result("96-well", rotation_deg=-179.8, orientation="turned-180")
    back = fit.turned().turned()
    assert back.orientation == "turned-180"
    assert back.a1_center_xy_um == pytest.approx(fit.a1_center_xy_um, abs=1e-6)
    assert back.rotation_deg == pytest.approx(fit.rotation_deg, abs=1e-9)
    assert back.refs == fit.refs


def _wall_fit(well: str, centre: tuple[float, float]) -> WellFit:
    return WellFit(
        well=well,
        recorded_xy_um=(centre[0] + 4.0, centre[1] - 4.0),
        center_xy_um=centre,
        diameter_um=15298.4,
        size_error=0.0,
        rms_um=3.0,
        n_points=40,
        walls=tuple(
            WallFrame(side=side, x_um=0.0, y_um=0.0, n_points=10, frames=2)
            for side in ("+X", "+Y", "-X", "-Y")
        ),
    )


def test_a_turn_renames_refs_and_wall_wells_and_keeps_the_residual() -> None:
    walls = WallImages(
        objective="10x",
        pixel_size_um=1.0,
        image_shape_px=(100, 100),
        search_um=3900.0,
        wells=(_wall_fit("B3", (1.0, 2.0)), _wall_fit("D5", (3.0, 4.0))),
        frames=16,
        folder="plate-walls/tok",
        elapsed_s=16.5,
    )
    fit = _result(
        "24-well",
        method="wall-images",
        refs=(
            WellRef(well="B3", x_um=1.0, y_um=2.0),
            WellRef(well="D5", x_um=3.0, y_um=4.0),
        ),
        n_refs=2,
        walls=walls,
    )
    turned = fit.turned()
    assert [r.well for r in turned.refs] == ["C4", "A2"]
    assert [(r.x_um, r.y_um) for r in turned.refs] == [(1.0, 2.0), (3.0, 4.0)]
    assert turned.walls is not None
    assert [w.well for w in turned.walls.wells] == ["C4", "A2"]
    assert turned.walls.wells[0].walls == walls.wells[0].walls
    assert turned.walls.folder == walls.folder
    assert turned.walls.frames == walls.frames
    for name in ("method", "residual_um", "scale_error", "n_refs", "fit", "objective"):
        assert getattr(turned, name) == getattr(fit, name), name
    assert turned.assumed[: len(fit.assumed)] == fit.assumed
    assert turned.assumed[-1] == (
        "turned 180° to turned-180 after the fit: every well renamed (row r to "
        "R-1-r, column c to C-1-c), nothing measured again; a well named in a "
        "line above is the fit's name"
    )


def test_with_orientation_of_the_same_orientation_is_the_result_itself() -> None:
    fit = _result()
    assert fit.with_orientation("not-turned") is fit
    turned = fit.with_orientation("turned-180")
    assert turned.orientation == "turned-180"
    assert turned.with_orientation("turned-180") is turned


@pytest.mark.parametrize(
    ("rotation_deg", "orientation"),
    [
        (179.8, "not-turned"),
        (-179.8, "not-turned"),
        (90.5, "not-turned"),
        (0.2, "turned-180"),
        (-89.5, "turned-180"),
    ],
)
def test_a_result_whose_rotation_contradicts_its_orientation_is_refused(
    rotation_deg: float, orientation: str
) -> None:
    with pytest.raises(ValidationError, match=r"rotation_deg .* orientation"):
        _result(rotation_deg=rotation_deg, orientation=orientation)


@pytest.mark.parametrize(
    ("rotation_deg", "orientation"),
    [
        (90.0, "not-turned"),
        (-90.0, "not-turned"),
        (179.8, "turned-180"),
        (-179.8, "turned-180"),
        (180.0, "turned-180"),
        (-180.0, "turned-180"),
        (90.0, "turned-180"),
    ],
)
def test_a_rotation_within_90_degrees_of_its_orientation_is_kept(
    rotation_deg: float, orientation: str
) -> None:
    result = _result(rotation_deg=rotation_deg, orientation=orientation)
    assert result.rotation_deg == rotation_deg


def test_a_document_without_an_orientation_reads_not_turned() -> None:
    data = _result(rotation_deg=30.0).model_dump(mode="json")
    del data["orientation"]
    loaded = PlateResult.model_validate(data)
    assert loaded.orientation == "not-turned"
    assert loaded.rotation_deg == 30.0


def test_orientation_deg_names_both_orientations() -> None:
    assert dict(ORIENTATION_DEG) == {"not-turned": 0.0, "turned-180": 180.0}


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (180.0, 180.0),
        (-180.0, 180.0),
        (540.0, 180.0),
        (-180.2, 179.8),
        (359.8, -0.2),
        (0.0, 0.0),
        (-179.8, -179.8),
        (720.0, 0.0),
    ],
)
def test_wrap_deg_is_in_minus_180_exclusive_to_180_inclusive(
    angle: float, expected: float
) -> None:
    wrapped = wrap_deg(angle)
    assert -180.0 < wrapped <= 180.0
    assert wrapped == pytest.approx(expected, abs=1e-9)
    assert math.isclose(
        math.cos(math.radians(wrapped)), math.cos(math.radians(angle)), abs_tol=1e-12
    )
