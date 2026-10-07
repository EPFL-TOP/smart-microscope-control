"""The synthetic plate sample and its camera, without Micro-Manager (design §8, #10).

Flat samples (``texture_std=0.0, noise_std=0.0``) isolate the well geometry;
the ground truth for it is ``useq``'s own well positions, never the sample's
arithmetic, so a mm/µm slip or a wrong rotation sign cannot agree with itself.
"""

from __future__ import annotations

import math
import re

import pytest
import useq

from smc.testing.synthetic import Blob, PlateSample

FLAT = {"texture_std": 0.0, "noise_std": 0.0}


def _plan(
    plate: str, rotation_deg: float, a1: tuple[float, float]
) -> useq.WellPlatePlan:
    return useq.WellPlatePlan(plate=plate, a1_center_xy=a1, rotation=rotation_deg)


# --- geometry --------------------------------------------------------------


@pytest.mark.parametrize("rotation_deg", [0.0, 90.0, 30.0, -17.5])
@pytest.mark.parametrize("plate", ["6-well", "96-well", "384-well"])
def test_every_well_centre_is_a_well_and_half_a_pitch_away_is_plastic(
    plate: str, rotation_deg: float
) -> None:
    a1 = (1234.5, -987.6)
    sample = PlateSample(plate, a1, rotation_deg, **FLAT)
    truth = _plan(plate, rotation_deg, a1)
    pitch_um = truth.plate.well_spacing[0] * 1000.0
    half_size_um = truth.plate.well_size[0] * 1000.0 / 2.0
    # The direction of the plate's columns in stage coordinates.
    along = (math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg)))

    for p in truth.all_well_positions:
        x, y = p.x, p.y
        assert sample.well_at(x, y) == p.name
        assert (
            sample.well_at(x + along[0] * pitch_um / 2, y + along[1] * pitch_um / 2)
            is None
        )
        inner, outer = 0.95 * half_size_um, 1.05 * half_size_um
        assert sample.well_at(x + along[0] * inner, y + along[1] * inner) == p.name
        assert sample.well_at(x + along[0] * outer, y + along[1] * outer) is None


def test_rotation_direction_is_counter_clockwise() -> None:
    def at(rotation_deg: float) -> PlateSample:
        return PlateSample("96-well", (0.0, 0.0), rotation_deg, **FLAT)

    assert at(0.0).well_at(9000.0, 0.0) == "A2"
    assert at(90.0).well_at(9000.0, 0.0) == "B1"
    assert at(-90.0).well_at(9000.0, 0.0) is None


def test_the_plate_ends_where_useq_says_it_does() -> None:
    sample = PlateSample("96-well", (0.0, 0.0), 0.0, **FLAT)
    last = _plan("96-well", 0.0, (0.0, 0.0)).all_well_positions[-1]
    assert last.name == "H12"
    assert sample.well_at(last.x, last.y) == "H12"
    # One pitch past either end of the plate is outside it, not a 13th column.
    assert sample.well_at(last.x + 9000.0, last.y) is None
    assert sample.well_at(last.x, last.y - 9000.0) is None
    assert sample.well_at(-9000.0, 0.0) is None
    assert sample.well_at(0.0, 9000.0) is None


def test_a_custom_plate_with_square_wells_uses_the_square() -> None:
    plate = useq.WellPlate(
        rows=2,
        columns=2,
        well_spacing=(10.0, 10.0),
        well_size=(8.0, 8.0),
        circular_wells=False,
    )
    sample = PlateSample(plate, (0.0, 0.0), 0.0, **FLAT)
    assert sample.well_at(3900.0, -3900.0) == "A1"  # a corner: inside a square
    assert sample.well_at(4100.0, 0.0) is None


def test_blobs_and_levels_are_kept() -> None:
    blob = Blob(1.0, 2.0, radius_um=3.0, intensity=4.0)
    sample = PlateSample(blobs=[blob], well_level=10.0, plastic_level=2.0, seed=7)
    assert sample.blobs == (blob,)
    assert (sample.well_level, sample.plastic_level, sample.seed) == (10.0, 2.0, 7)
    assert sample.plate_name == "96-well"
    with pytest.raises(AttributeError):
        blob.x_um = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("make", "phrase"),
    [
        pytest.param(
            lambda: PlateSample(noise_std=-1.0),
            "noise_std must be finite and within [0, 1e+09], got -1.0",
            id="negative-noise",
        ),
        pytest.param(
            lambda: PlateSample(texture_std=math.inf),
            "texture_std must be finite and within [0, 1e+09], got inf",
            id="infinite-texture",
        ),
        pytest.param(
            lambda: PlateSample(well_level=math.nan),
            "well_level must be finite and within [0, 1e+09], got nan",
            id="nan-well-level",
        ),
        pytest.param(
            lambda: PlateSample(plastic_level=-5.0),
            "plastic_level must be finite and within [0, 1e+09], got -5.0",
            id="negative-plastic-level",
        ),
        pytest.param(
            lambda: PlateSample(rotation_deg=math.nan),
            "rotation_deg must be finite and within [-1e+06, 1e+06], got nan",
            id="nan-rotation",
        ),
        pytest.param(
            lambda: PlateSample(a1_center_xy_um=(0.0, math.inf)),
            "a1_center_xy_um[1] must be finite and within [-1e+09, 1e+09], got inf",
            id="infinite-a1",
        ),
        pytest.param(
            lambda: PlateSample(a1_center_xy_um=(1.0,)),  # type: ignore[arg-type]
            "a1_center_xy_um must be two numbers (x, y), got (1.0,)",
            id="a1-with-one-number",
        ),
        pytest.param(
            lambda: PlateSample(seed=-1),
            "seed must be a non-negative integer, got -1",
            id="negative-seed",
        ),
        pytest.param(
            lambda: PlateSample(seed=1.5),  # type: ignore[arg-type]
            "seed must be a non-negative integer, got 1.5",
            id="float-seed",
        ),
        pytest.param(
            lambda: PlateSample("glass"),
            "Unknown plate name 'glass'",
            id="unknown-plate",
        ),
        pytest.param(
            lambda: PlateSample(
                useq.WellPlate(
                    rows=1,
                    columns=2,
                    well_spacing=(5.0, 5.0),
                    well_size=(6.0, 5.0),
                    circular_wells=True,
                )
            ),
            "the plate's well_size (6.0, 5.0) mm exceeds its well_spacing "
            "(5.0, 5.0) mm: a well must lie inside its own pitch cell",
            id="well-larger-than-its-pitch",
        ),
        pytest.param(
            lambda: Blob(0.0, 0.0, radius_um=0.0, intensity=1.0),
            "radius_um must be finite and positive (at most 1e+09), got 0.0",
            id="blob-radius-zero",
        ),
        pytest.param(
            lambda: Blob(math.nan, 0.0, radius_um=1.0, intensity=1.0),
            "x_um must be finite and within [-1e+09, 1e+09], got nan",
            id="blob-nan-position",
        ),
        pytest.param(
            lambda: Blob(0.0, 0.0, radius_um=1.0, intensity=math.inf),
            "intensity must be finite and within [-1e+09, 1e+09], got inf",
            id="blob-infinite-intensity",
        ),
    ],
)
def test_invalid_parameters_are_refused(make, phrase: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=re.escape(phrase)):
        make()
