"""``make_input``: a ``SelectionInput`` built in memory for the test of a selector (algorithms design §5.7, #567).

The expected values are the plate's, from ``useq`` and the geometry module, and
never read back from the input under test: B2 of a 96-well plate with A1 at the
origin is at (9000, -9000) µm, a camera of 400 x 300 µm is 4 and 3 µm a pixel
over a 100 x 100 frame, and a field is the camera's footprint.
"""

from __future__ import annotations

import math

import pytest

from smc.data import DatasetError
from smc.plugins.geometry import footprint, fov_of
from smc.plugins.slots import ScanFrame
from smc.selection.api import SelectionError, SelectionInput
from smc.selection.testing import make_input


def test_make_input_places_the_wells_the_cells_and_the_field_as_asked() -> None:
    cells = [
        ("B3", 18050.0, -8990.0, "doublet"),
        ("B2", 9010.0, -9020.0, "single"),
        ("B2", 9100.0, -9100.0, "single"),
    ]

    data = make_input(
        cells,
        wells=("B2", "B3"),
        field_um=(400.0, 300.0),
        labels=("single", "doublet"),
        fields_per_well=4,
        wall_margin_um=25.0,
        exclude_centre_um=40.0,
    )

    assert isinstance(data, SelectionInput)
    assert data.fov_um == (400.0, 300.0)
    # A camera of 4 by 3 µm a pixel, upright and not mirrored, over 100 x 100 pixels.
    assert data.matrix_um_per_px == ((4.0, 0.0), (0.0, -3.0))
    assert data.image_shape_px == (100, 100)
    assert data.scan.wells == ("B2", "B3")
    assert data.scan.geometry["B2"].center_um == pytest.approx((9000.0, -9000.0))
    assert data.scan.geometry["B3"].center_um == pytest.approx((18000.0, -9000.0))
    # The plan has one field a well, at the well's centre.
    assert data.first_field("B2") == pytest.approx((9000.0, -9000.0))
    assert data.first_field("B3") == pytest.approx((18000.0, -9000.0))
    assert [field.name for field in data.scan.plan.fields] == ["B2_0000", "B3_0000"]
    assert data.labels == ("single", "doublet")
    assert (
        data.fields_per_well,
        data.wall_margin_um,
        data.exclude_centre_um,
        data.objective,
    ) == (4, 25.0, 40.0, None)
    # The cells are the input's, in its order, and every one is a Cell the detector could have written.
    assert [(c.well, c.x_um, c.y_um, c.label) for c in data.cells] == cells
    assert all(c.score == 1.0 and c.frame == 0 for c in data.cells)


def test_make_input_has_one_label_and_one_well_by_default() -> None:
    data = make_input([("B2", 9000.0, -9000.0, "single")])

    assert data.labels == ("single",)
    assert data.scan.wells == ("B2",)
    assert data.fov_um == (100.0, 100.0)
    assert (data.fields_per_well, data.wall_margin_um, data.exclude_centre_um) == (
        10,
        0.0,
        0.0,
    )


def test_make_input_describes_a_well_with_no_cell_and_the_classes_of_the_cells() -> (
    None
):
    data = make_input(
        [("B3", 18000.0, -9000.0, "debris"), ("B3", 18010.0, -9000.0, "single")],
        wells=("B2", "B3"),
        labels=("single", "doublet"),
    )

    result = data.scan.cells_result
    assert result is not None
    # The labels asked for first, then the others in the order they appear; zeros are kept.
    assert result.classes == ("single", "doublet", "debris")
    assert result.counts == {
        "B2": {"single": 0, "doublet": 0, "debris": 0},
        "B3": {"single": 1, "doublet": 0, "debris": 1},
    }
    assert data.cells_in(18000.0, -9000.0, "B3") != ()
    assert data.cells_in(9000.0, -9000.0, "B2") == ()


def test_make_input_normalises_the_wells_as_a_person_types_them() -> None:
    data = make_input([("B2", 9000.0, -9000.0, "single")], wells=("b02",))

    assert data.scan.wells == ("B2",)


def test_make_input_follows_the_order_of_the_wells_it_was_given() -> None:
    data = make_input([], wells=("B3", "A1", "B2"))

    assert data.scan.wells == ("B3", "A1", "B2")
    assert [field.well for field in data.scan.plan.fields] == ["B3", "A1", "B2"]


def test_make_input_field_is_the_cameras_footprint_for_a_field_that_is_not_square() -> (
    None
):
    data = make_input([], field_um=(250.0, 100.0))

    # The oracle is the geometry module's, not the input's own property.
    assert fov_of(data.matrix_um_per_px, data.image_shape_px) == (250.0, 100.0)
    corners = footprint(
        0.0, 0.0, matrix=data.matrix_um_per_px, image_shape_px=data.image_shape_px
    )
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    assert (max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx((250.0, 100.0))
    assert data.scan.pixel_size_um == pytest.approx(2.5)


def test_make_input_images_raise_a_dataset_error() -> None:
    data = make_input([("B2", 9000.0, -9000.0, "single")])
    frame = ScanFrame(
        well="B2",
        index=0,
        name="B2_0000",
        file="frames/B2_0000.tif",
        x_um=9000.0,
        y_um=-9000.0,
        z_um=None,
        focus="none",
        pfs_offset=None,
        t_s=0.0,
    )

    # No frame was ever taken: asking for the pixels of one is a refusal that says so.
    assert data.scan.frames() == ()
    with pytest.raises(DatasetError, match="B2_0000"):
        data.scan.image(frame)


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        ([("C7", 0.0, 0.0, "single")], "cell 0 is in well 'C7'.*not given.*B2"),
        (
            [("B2", 9000.0, -9000.0, "single"), ("B2", math.nan, 0.0, "single")],
            "cell 1 .*not finite",
        ),
        ([("B2", 0.0, math.inf, "single")], "cell 0 .*not finite"),
    ],
)
def test_make_input_refuses_a_cell_of_a_well_it_was_not_given_and_a_non_finite_one(
    cells: list[tuple[str, float, float, str]], message: str
) -> None:
    with pytest.raises(SelectionError, match=message):
        make_input(cells)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"field_um": (0.0, 100.0)}, "field_um must be finite and positive"),
        ({"field_um": (100.0, -1.0)}, "field_um must be finite and positive"),
        ({"field_um": (math.nan, 100.0)}, "field_um must be finite and positive"),
        ({"field_um": (100.0, math.inf)}, "field_um must be finite and positive"),
        ({"fields_per_well": 0}, "fields_per_well must be at least 1"),
        ({"wall_margin_um": -1.0}, "wall_margin_um must be finite and 0 or more"),
        ({"exclude_centre_um": math.nan}, "exclude_centre_um must be finite"),
        ({"labels": ()}, "labels must name at least one label"),
        ({"labels": ("a", "b", "a")}, "labels repeat 'a'"),
        ({"wells": ("B2", "b02")}, "wells are listed twice"),
        ({"wells": ("Z99",)}, "Z99"),
        ({"plate": "no-such-plate"}, "no-such-plate"),
    ],
)
def test_make_input_names_what_it_refuses(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(SelectionError, match=message):
        make_input([], **kwargs)  # type: ignore[arg-type]
