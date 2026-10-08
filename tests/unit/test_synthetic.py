"""The synthetic plate sample and its camera, without Micro-Manager (design §8, #10).

Flat samples (``texture_std=0.0, noise_std=0.0``) isolate the well geometry;
the ground truth for it is ``useq``'s own well positions, never the sample's
arithmetic, so a mm/µm slip or a wrong rotation sign cannot agree with itself.
"""

from __future__ import annotations

import itertools
import math
import re
import time
import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import useq
from pymmcore_plus import DeviceType

from smc.hardware.capabilities import Camera, XYStage, ZStage
from smc.hardware.microscope import Microscope
from smc.hardware.profile import Profile
from smc.testing import FakeCore, FakeDevice
from smc.testing.synthetic import (
    MAX_BLUR_PX,
    REFERENCE_EXPOSURE_MS,
    Blob,
    PlateSample,
    SampleCamera,
    _shape,
    _tile_values,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

FLAT = {"texture_std": 0.0, "noise_std": 0.0}
#: The defaults of ``PlateSample``, which these tests rely on.
WELL, PLASTIC = 3000, 800


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


@pytest.mark.parametrize("rotation_deg", [0.0, 90.0, 30.0, -17.5])
@pytest.mark.parametrize("plate", ["6-well", "96-well", "384-well"])
def test_a_well_is_round_or_square_as_useq_says(
    plate: str, rotation_deg: float
) -> None:
    """Probes along a diagonal, where a circle and a square of the same size part.

    1.2 half-sizes out along the diagonal is outside a circle but inside a
    square (0.85 half-sizes on each axis); 1.45 is outside both. Along the
    plate's axes, which the test above probes, they agree.
    """
    a1 = (1234.5, -987.6)
    sample = PlateSample(plate, a1, rotation_deg, **FLAT)
    truth = _plan(plate, rotation_deg, a1)
    half_size_um = truth.plate.well_size[0] * 500.0
    angle = math.radians(rotation_deg + 45.0)
    diagonal = (math.cos(angle), math.sin(angle))

    def at(p: useq.AbsolutePosition, distance_um: float) -> str | None:
        return sample.well_at(
            p.x + diagonal[0] * distance_um, p.y + diagonal[1] * distance_um
        )

    for p in truth.all_well_positions:
        assert at(p, 0.95 * half_size_um) == p.name
        # Outside a round well, still inside a square one.
        expected = None if truth.plate.circular_wells else p.name
        assert at(p, 1.2 * half_size_um) == expected
        assert at(p, 1.45 * half_size_um) is None


@pytest.mark.parametrize("rotation_deg", [0.0, 90.0, 30.0])
def test_a_plate_with_unequal_x_and_y_is_read_axis_by_axis(
    rotation_deg: float,
) -> None:
    """Every registered plate is isotropic, so this one is not: 10 x 6 mm pitch,
    8 x 4 mm square wells. An x value used for a y quantity shows here."""
    plate = useq.WellPlate(
        rows=3,
        columns=2,
        well_spacing=(10.0, 6.0),
        well_size=(8.0, 4.0),
        circular_wells=False,
    )
    a1 = (-500.0, 250.0)
    sample = PlateSample(plate, a1, rotation_deg, **FLAT)
    truth = useq.WellPlatePlan(plate=plate, a1_center_xy=a1, rotation=rotation_deg)
    half_x_um, half_y_um = 4000.0, 2000.0
    angle = math.radians(rotation_deg)
    plate_x = (math.cos(angle), math.sin(angle))  # the columns' direction
    plate_y = (-math.sin(angle), math.cos(angle))  # the rows' direction

    def at(
        p: useq.AbsolutePosition, along_x_um: float, along_y_um: float
    ) -> str | None:
        return sample.well_at(
            p.x + plate_x[0] * along_x_um + plate_y[0] * along_y_um,
            p.y + plate_x[1] * along_x_um + plate_y[1] * along_y_um,
        )

    assert len(truth.all_well_positions) == 6
    for p in truth.all_well_positions:
        assert at(p, 0.0, 0.0) == p.name
        # The edges of the well, on each axis and on both sides.
        for sign in (1.0, -1.0):
            assert at(p, sign * 0.95 * half_x_um, 0.0) == p.name
            assert at(p, sign * 1.05 * half_x_um, 0.0) is None
            assert at(p, 0.0, sign * 0.95 * half_y_um) == p.name
            assert at(p, 0.0, sign * 1.05 * half_y_um) is None
        # A square: its corner is inside; the gaps between wells are plastic.
        assert at(p, 0.95 * half_x_um, 0.95 * half_y_um) == p.name
        assert at(p, 5000.0, 0.0) is None
        assert at(p, 0.0, 3000.0) is None


def test_rotation_direction_is_counter_clockwise() -> None:
    def at(rotation_deg: float) -> PlateSample:
        return PlateSample("96-well", (0.0, 0.0), rotation_deg, **FLAT)

    assert at(0.0).well_at(9000.0, 0.0) == "A2"
    assert at(90.0).well_at(9000.0, 0.0) == "B1"
    assert at(-90.0).well_at(9000.0, 0.0) is None

    # The frames agree: B1 sits at (0, -9000) unrotated and is gone at 90 degrees.
    at_0 = at(0.0).render(0.0, -9000.0, 0.0, shape=(64, 64), pixel_size_um=1.0)
    at_90 = at(90.0).render(0.0, -9000.0, 0.0, shape=(64, 64), pixel_size_um=1.0)
    assert at_0.mean() == WELL
    assert at_90.mean() == PLASTIC


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
            lambda: PlateSample(seed=True),  # type: ignore[arg-type]
            "seed must be a non-negative integer, got True",
            id="bool-seed",
        ),
        pytest.param(
            lambda: Blob(0.0, 0.0, radius_um=0.0, intensity=1.0),
            "radius_um must be finite and within [1e-06, 1e+09], got 0.0",
            id="blob-radius-zero",
        ),
        pytest.param(
            lambda: Blob(math.nan, 0.0, radius_um=1.0, intensity=1.0),
            "x_um must be finite and within [-1e+09, 1e+09], got nan",
            id="blob-nan-position",
        ),
        pytest.param(
            lambda: Blob(0.0, math.inf, radius_um=1.0, intensity=1.0),
            "y_um must be finite and within [-1e+09, 1e+09], got inf",
            id="blob-infinite-y",
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


# --- texture ---------------------------------------------------------------


def test_texture_tiles_are_bit_exact_across_platforms() -> None:
    """The first values of the hash, pinned: numpy < 2 on Windows has a 32-bit
    default integer, and a hash written on scalars would differ there."""
    ix = np.array([0, 1, -1, 7], dtype=np.int64)
    iy = np.array([0, 0, 5, -3], dtype=np.int64)
    assert _tile_values(ix, iy, 0) == pytest.approx(
        [
            -1.7320508075688772,
            1.3278275898326373,
            -0.14943785591070433,
            -1.4697101501494696,
        ],
        abs=1e-12,
    )
    assert _tile_values(ix, iy, 1) == pytest.approx(
        [
            0.6979045179271314,
            -0.9645479464229244,
            -1.308947001715897,
            0.5873197229803402,
        ],
        abs=1e-12,
    )


def test_the_texture_hash_wraps_without_a_numpy_overflow_warning() -> None:
    ix = np.arange(-1000, 1000, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = _tile_values(ix, ix[::-1], 2**63 + 12345)
        # A 0-d index would make the products numpy scalars, which warn.
        _tile_values(np.int64(-(2**62)), np.int64(2**62), 3)
    assert abs(values.mean()) < 0.1
    assert values.std() == pytest.approx(1.0, abs=0.05)


# --- frames ----------------------------------------------------------------


def _content_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """Where the content of ``before`` went in ``after``: ``(d_col, d_row)`` pixels.

    FFT cross-correlation (mean-subtracted ``rfft2`` product with the
    conjugate, argmax, wrapped to signed): a whole-pixel answer, which is all
    these frames can give. Positive ``d_col`` is to the right, positive
    ``d_row`` is down the image.
    """
    fa = np.fft.rfft2(before - before.mean())
    fb = np.fft.rfft2(after - after.mean())
    corr = np.fft.irfft2(np.conj(fa) * fb, s=before.shape)
    row, col = np.unravel_index(np.argmax(corr), corr.shape)
    height, width = before.shape
    d_row = row - height if row > height // 2 else row
    d_col = col - width if col > width // 2 else col
    return float(d_col), float(d_row)


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """The intensity-weighted ``(col, row)`` of an image, in pixels."""
    rows, cols = np.indices(image.shape)
    total = image.sum()
    return float((cols * image).sum() / total), float((rows * image).sum() / total)


def test_frame_mean_drops_when_the_stage_crosses_a_well_wall() -> None:
    sample = PlateSample("96-well", (0.0, 0.0), 0.0, **FLAT)

    def frame_at(x_um: float) -> np.ndarray:
        return sample.render(x_um, 0.0, 0.0, shape=(64, 64), pixel_size_um=1.0)

    assert np.all(frame_at(0.0) == WELL)
    assert np.all(frame_at(4500.0) == PLASTIC)
    # The wall of A1 is at x = 3200 um (6.4 mm wells): half the frame is well.
    wall = frame_at(3200.0)
    assert PLASTIC < wall.mean() < WELL
    assert 0.4 < (wall == WELL).mean() < 0.6
    means = [frame_at(float(x)).mean() for x in range(3100, 3301, 20)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(means))
    assert means[0] > means[-1]


@pytest.mark.parametrize("pixel_size_um", [1.0, 0.5, 2.0])
def test_texture_is_deterministic_and_position_locked(pixel_size_um: float) -> None:
    sample = PlateSample(noise_std=0.0)
    x_um, y_um = 123.4, -56.7

    def frame_at(x: float, y: float, seed_sample: PlateSample = sample) -> np.ndarray:
        return seed_sample.render(
            x, y, 0.0, shape=(256, 256), pixel_size_um=pixel_size_um
        ).astype(np.float64)

    here = frame_at(x_um, y_um)
    assert np.array_equal(here, frame_at(x_um, y_um))
    assert not np.array_equal(
        here, frame_at(x_um, y_um, PlateSample(noise_std=0.0, seed=1))
    )

    # Design section 8: a +10 um X move shifts the content 10/ps columns left,
    # and a +10 um Y move shifts it as many rows down.
    expected = 10.0 / pixel_size_um
    d_col, d_row = _content_shift(here, frame_at(x_um + 10.0, y_um))
    assert d_col == pytest.approx(-expected, abs=0.5)
    assert d_row == pytest.approx(0.0, abs=0.5)
    d_col, d_row = _content_shift(here, frame_at(x_um, y_um + 10.0))
    assert d_col == pytest.approx(0.0, abs=0.5)
    assert d_row == pytest.approx(expected, abs=0.5)


def test_the_texture_has_the_asked_standard_deviation() -> None:
    sample = PlateSample(
        noise_std=0.0
    )  # well level everywhere: only the texture varies
    frame = sample.render(0.0, 0.0, 0.0, shape=(512, 512), pixel_size_um=1.0)
    assert float(frame.std()) == pytest.approx(40.0, rel=0.05)
    assert float(frame.mean()) == pytest.approx(WELL, abs=2.0)


def test_blob_appears_at_its_stage_position() -> None:
    blob = Blob(40.0, -25.0, radius_um=10.0, intensity=2000.0)
    sample = PlateSample(blobs=[blob], **FLAT)
    frame = sample.render(0.0, 0.0, 0.0, shape=(128, 128), pixel_size_um=1.0)
    col, row = _centroid(frame.astype(np.float64) - WELL)
    # Columns grow with +X from the centre (63.5); rows grow with -Y.
    assert col == pytest.approx(63.5 + 40.0, abs=0.5)
    assert row == pytest.approx(63.5 + 25.0, abs=0.5)
    # The centre falls between four pixel centres, each 0.5 * sqrt(2) um away,
    # and sigma is half the radius (5 um): the peak is 2000 * exp(-0.5 / 50).
    assert int(frame.max()) == pytest.approx(WELL + 2000.0 * math.exp(-0.01), abs=1.0)


@pytest.mark.parametrize(
    ("rotation_deg", "mirrored"), [(0.0, False), (37.0, False), (90.0, True)]
)
def test_a_blob_keeps_its_whole_gaussian_at_any_camera_rotation(
    rotation_deg: float, mirrored: bool
) -> None:
    """The integral of ``intensity * exp(-d^2 / 2 sigma^2)`` is ``intensity * 2 pi sigma^2``:
    a window or a box that is too small for the rotated frame loses the tails."""
    blob = Blob(10.0, -5.0, radius_um=10.0, intensity=2000.0)
    sample = PlateSample(blobs=[blob], **FLAT)
    frame = sample.render(
        0.0,
        0.0,
        0.0,
        shape=(160, 160),
        pixel_size_um=1.0,
        camera_rotation_deg=rotation_deg,
        mirrored=mirrored,
    )
    total = float((frame.astype(np.float64) - WELL).sum())
    # The box of +-4 sigma loses 0.006 %; a window of 1.8 radii would lose 0.15 %.
    assert total == pytest.approx(2000.0 * 2.0 * math.pi * 5.0**2, rel=5e-4)


@pytest.mark.parametrize(
    ("rotation_deg", "mirrored"), [(0.0, False), (30.0, False), (0.0, True)]
)
def test_a_blob_cut_by_the_frame_edge_matches_a_larger_frame(
    rotation_deg: float, mirrored: bool
) -> None:
    sample = PlateSample(
        blobs=[Blob(60.0, 10.0, radius_um=10.0, intensity=2000.0)], **FLAT
    )
    pose = {
        "camera_rotation_deg": rotation_deg,
        "mirrored": mirrored,
        "pixel_size_um": 1.0,
    }
    small = sample.render(0.0, 0.0, 0.0, shape=(128, 128), **pose)
    large = sample.render(0.0, 0.0, 0.0, shape=(256, 256), **pose)
    # The same centre: the small frame is the middle of the large one.
    difference = small.astype(np.int64) - large[64:192, 64:192].astype(np.int64)
    assert np.abs(difference).max() <= 1
    assert small.max() > WELL + 500  # the blob is really in view


def test_a_blob_outside_the_frame_adds_nothing() -> None:
    far = Blob(5000.0, 5000.0, radius_um=10.0, intensity=2000.0)
    sample = PlateSample(blobs=[far], **FLAT)
    frame = sample.render(0.0, 0.0, 0.0, shape=(64, 64), pixel_size_um=1.0)
    assert np.all(frame == WELL)


def test_a_blob_a_kilometre_away_over_a_tiny_pixel_adds_nothing() -> None:
    """``1e9 um / 1e-300 um`` is infinite pixels: the blob is skipped, not an
    ``OverflowError`` out of ``math.floor`` (FM-32)."""
    far = Blob(1e9, 0.0, radius_um=1.0, intensity=2000.0)
    frame = PlateSample(blobs=[far], **FLAT).render(
        0.0, 0.0, 0.0, shape=(8, 8), pixel_size_um=1e-300
    )
    assert np.all(frame == WELL)


@pytest.mark.parametrize(
    ("blob_at", "rotation_deg", "mirrored", "expected_cr"),
    [
        pytest.param((40.0, 0.0), 0.0, False, (103.5, 63.5), id="east-is-right"),
        pytest.param((0.0, 25.0), 0.0, False, (63.5, 38.5), id="north-is-up"),
        pytest.param(
            (40.0, 0.0), 90.0, False, (63.5, 103.5), id="rotated-east-is-down"
        ),
        pytest.param((40.0, 0.0), 0.0, True, (23.5, 63.5), id="mirrored-east-is-left"),
        # Mirroring is applied to the image's x axis before the rotation:
        # north, rotated 90 degrees, is on the right, and mirrored it is on
        # the left.
        pytest.param(
            (0.0, 25.0), 90.0, False, (88.5, 63.5), id="rotated-north-is-right"
        ),
        pytest.param((0.0, 25.0), 90.0, True, (38.5, 63.5), id="mirror-then-rotate"),
    ],
)
def test_camera_rotation_and_mirror_move_the_image_axes(
    blob_at: tuple[float, float],
    rotation_deg: float,
    mirrored: bool,
    expected_cr: tuple[float, float],
) -> None:
    blob = Blob(*blob_at, radius_um=8.0, intensity=2000.0)
    sample = PlateSample(blobs=[blob], **FLAT)
    frame = sample.render(
        0.0,
        0.0,
        0.0,
        shape=(128, 128),
        pixel_size_um=1.0,
        camera_rotation_deg=rotation_deg,
        mirrored=mirrored,
    )
    col, row = _centroid(frame.astype(np.float64) - WELL)
    assert (col, row) == pytest.approx(expected_cr, abs=0.5)


def test_exposure_scales_the_signal_but_not_the_noise() -> None:
    flat = PlateSample(**FLAT)
    twice = flat.render(
        0.0, 0.0, 0.0, shape=(8, 8), pixel_size_um=1.0, exposure_ms=20.0
    )
    assert np.all(twice == 2 * WELL)
    assert np.all(
        flat.render(0.0, 0.0, 0.0, shape=(8, 8), pixel_size_um=1.0, exposure_ms=0.0)
        == 0
    )

    bright = PlateSample(well_level=50000.0, **FLAT)
    saturated = bright.render(
        0.0, 0.0, 0.0, shape=(8, 8), pixel_size_um=1.0, exposure_ms=20.0
    )
    assert saturated.dtype == np.uint16
    assert np.all(saturated == 65535)

    noisy = PlateSample(texture_std=0.0, noise_std=20.0)
    for exposure_ms in (10.0, 20.0):
        frame = noisy.render(
            0.0, 0.0, 0.0, shape=(64, 64), pixel_size_um=1.0, exposure_ms=exposure_ms
        )
        assert float(frame.std()) == pytest.approx(20.0, rel=0.05)


def test_noise_differs_between_frames_and_a_seed_repeats_the_sequence() -> None:
    def frames(seed: int) -> list[np.ndarray]:
        sample = PlateSample(seed=seed)
        return [
            sample.render(0.0, 0.0, 0.0, shape=(32, 32), pixel_size_um=1.0)
            for _ in range(2)
        ]

    first, second = frames(5)
    assert not np.array_equal(first, second)
    again_first, again_second = frames(5)
    assert np.array_equal(first, again_first)
    assert np.array_equal(second, again_second)


def test_a_dark_frame_is_clipped_at_zero_not_wrapped() -> None:
    """Negative counts are clipped to 0: cast to ``uint16`` they would wrap to ~65000."""
    sample = PlateSample(
        well_level=0.0, plastic_level=0.0, texture_std=0.0, noise_std=20.0
    )
    frame = sample.render(0.0, 0.0, 0.0, shape=(64, 64), pixel_size_um=1.0)
    assert int(frame.max()) < 150  # seven sigma of the noise
    assert float((frame == 0).mean()) > 0.4  # the negative half of it


def test_a_dark_blob_darkens_to_zero_and_never_wraps() -> None:
    dark = Blob(0.0, 0.0, radius_um=10.0, intensity=-1000.0)
    sample = PlateSample(well_level=100.0, plastic_level=100.0, blobs=[dark], **FLAT)
    frame = sample.render(0.0, 0.0, 0.0, shape=(64, 64), pixel_size_um=1.0)
    assert np.all(frame[31:33, 31:33] == 0)  # 100 - 990, clipped
    assert int(frame.max()) == 100  # the corners, out of reach of the blob


@pytest.mark.parametrize("shape", [(64, 192), (192, 64)])
@pytest.mark.parametrize(
    ("z_um", "rotation_deg", "mirrored"),
    [(0.0, 0.0, False), (0.0, 30.0, True), (40.0, 0.0, False), (40.0, 30.0, True)],
)
def test_a_non_square_frame_is_the_middle_of_a_larger_square_one(
    shape: tuple[int, int], z_um: float, rotation_deg: float, mirrored: bool
) -> None:
    """Height and width swapped anywhere (blob window, blur crop, padding) shows
    only off the diagonal: every square frame is blind to it."""
    bright = Blob(25.0, -10.0, radius_um=8.0, intensity=1500.0)
    dark = Blob(-30.0, 15.0, radius_um=5.0, intensity=-400.0)
    sample = PlateSample(noise_std=0.0, blobs=[bright, dark])
    pose = {
        "camera_rotation_deg": rotation_deg,
        "mirrored": mirrored,
        "pixel_size_um": 1.0,
    }
    side = max(shape)
    large = sample.render(0.0, 0.0, z_um, shape=(side, side), **pose)
    small = sample.render(0.0, 0.0, z_um, shape=shape, **pose)

    assert small.shape == shape
    row0, col0 = (side - shape[0]) // 2, (side - shape[1]) // 2
    middle = large[row0 : row0 + shape[0], col0 : col0 + shape[1]]
    assert np.abs(small.astype(np.int64) - middle.astype(np.int64)).max() <= 1
    # ... and the blobs are really in the small frame.
    bare = PlateSample(noise_std=0.0).render(0.0, 0.0, z_um, shape=shape, **pose)
    assert (small.astype(np.int64) - bare.astype(np.int64)).max() > 300


def test_blobs_add_up_and_sit_on_the_texture() -> None:
    first = Blob(20.0, 5.0, radius_um=8.0, intensity=1200.0)
    second = Blob(26.0, 0.0, radius_um=6.0, intensity=900.0)  # overlaps the first

    def frame(sample: PlateSample) -> np.ndarray:
        return sample.render(0.0, 0.0, 0.0, shape=(96, 96), pixel_size_um=1.0).astype(
            np.int64
        )

    bump_first = frame(PlateSample(blobs=[first], **FLAT)) - WELL
    bump_second = frame(PlateSample(blobs=[second], **FLAT)) - WELL
    assert bump_first.max() > 1000
    assert bump_second.max() > 800
    # Each rounding is within half a count, so sums agree within a count or two.
    both = frame(PlateSample(blobs=[first, second], **FLAT)) - WELL
    assert np.abs(both - bump_first - bump_second).max() <= 1

    textured = frame(PlateSample(noise_std=0.0))
    textured_both = frame(PlateSample(noise_std=0.0, blobs=[first, second]))
    assert np.abs(textured_both - textured - bump_first - bump_second).max() <= 2


def test_a_texture_tile_is_four_um_wide_on_both_sides_of_zero() -> None:
    """``floor`` puts the tiles at [-4, 0) and [0, 4); truncation would make the
    one around zero 8 um wide."""
    sample = PlateSample(noise_std=0.0)
    frame = sample.render(0.0, 0.0, 0.0, shape=(8, 8), pixel_size_um=1.0)
    # Pixel centres are at -3.5 ... 3.5 um on both axes: four tiles of 4 x 4 px.
    quadrants = [frame[:4, :4], frame[:4, 4:], frame[4:, :4], frame[4:, 4:]]
    for quadrant in quadrants:
        assert np.all(quadrant == quadrant[0, 0])
    assert len({int(q[0, 0]) for q in quadrants}) == 4


def test_the_blur_grows_up_to_its_cap_and_not_beyond() -> None:
    sample = PlateSample(well_level=1000.0, plastic_level=1000.0, noise_std=0.0)

    def frame_at(z_um: float) -> np.ndarray:
        return sample.render(0.0, 0.0, z_um, shape=(64, 64), pixel_size_um=1.0)

    cap_z_um = 10.0 * MAX_BLUR_PX
    assert MAX_BLUR_PX == 32.0
    assert not np.array_equal(frame_at(cap_z_um - 10.0), frame_at(cap_z_um))
    assert np.array_equal(frame_at(cap_z_um), frame_at(cap_z_um + 10.0))
    assert np.array_equal(frame_at(cap_z_um), frame_at(5000.0))


def test_well_at_refuses_a_position_that_is_not_a_number() -> None:
    sample = PlateSample(**FLAT)
    with pytest.raises(
        ValueError,
        match=re.escape("x_um must be finite and within [-1e+09, 1e+09], got nan"),
    ):
        sample.well_at(math.nan, 0.0)
    with pytest.raises(
        ValueError,
        match=re.escape("y_um must be finite and within [-1e+09, 1e+09], got inf"),
    ):
        sample.well_at(0.0, math.inf)


def test_a_shape_bound_is_checked_in_python_ints_not_in_the_callers_dtype() -> None:
    """numpy < 2 on Windows turns ``np.array([60000, 60000])`` into int32, whose
    product wraps to a negative number below the bound; a small dtype's product
    also warns, which ``-W error`` turns into a failure (FM-32)."""
    prefix = (
        "shape must be two positive integers (height, width) with at most "
        "16777216 pixels"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=re.escape(prefix)):
            _shape((np.int32(60000), np.int32(60000)))
        assert _shape((np.uint16(300), np.uint16(300))) == (300, 300)
        assert _shape((np.int64(4096), np.int64(4096))) == (4096, 4096)


def test_a_blur_that_pads_a_frame_beyond_the_bound_is_refused_before_it_allocates() -> (
    None
):
    """One row of 2**24 pixels passes the shape bound; padded by 128 pixels on
    each side for a blur of 32 pixels it would be 4.3e9 values (FM-32)."""
    phrase = (
        "a blur of 32 px (z_um=320.0) pads shape (1, 16777216) to "
        "4311810304 pixels, more than the 33554432 allowed"
    )
    with pytest.raises(ValueError, match=re.escape(phrase)):
        PlateSample().render(0.0, 0.0, 320.0, shape=(1, 2**24), pixel_size_um=1.0)


def test_the_largest_frame_may_still_be_blurred_to_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The padding bound has room for a frame at the shape bound with the maximum
    blur: 4352 x 4352 pixels is allowed. The render stops where it would start
    to allocate."""

    class ReachedError(Exception):
        pass

    def stop(self: PlateSample, pose: object) -> None:
        raise ReachedError(pose.shape)  # type: ignore[attr-defined]

    monkeypatch.setattr(PlateSample, "_signal", stop)
    with pytest.raises(ReachedError) as reached:
        PlateSample().render(0.0, 0.0, 320.0, shape=(4096, 4096), pixel_size_um=1.0)
    assert reached.value.args == ((4352, 4352),)


def test_a_blob_radius_that_would_underflow_the_exponent_is_refused() -> None:
    """``radius_um ** 2`` underflows to 0 below about 1e-154, and the exponent
    of a pixel on the blob's centre becomes 0/0: NaN, cast to ``uint16`` (FM-32)."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "radius_um must be finite and within [1e-06, 1e+09], got 1e-200"
        ),
    ):
        Blob(0.0, 0.0, radius_um=1e-200, intensity=1000.0)
    # The smallest allowed radius renders without a warning.
    tiny = Blob(0.0, 0.0, radius_um=1e-6, intensity=1000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = PlateSample(blobs=[tiny], **FLAT).render(
            0.0, 0.0, 0.0, shape=(16, 16), pixel_size_um=1.0
        )
    assert np.all(frame == WELL)


def test_blur_grows_with_distance_from_focus() -> None:
    # Texture only: both levels equal, no noise. 512 x 512 keeps the mean of a
    # blurred frame within a count of the sharp one (128 x 128 does not).
    sample = PlateSample(well_level=1000.0, plastic_level=1000.0, noise_std=0.0)

    def frame_at(z_um: float) -> np.ndarray:
        return sample.render(0.0, 0.0, z_um, shape=(512, 512), pixel_size_um=1.0)

    sharp, mild, strong = frame_at(0.0), frame_at(20.0), frame_at(60.0)
    stds = [float(f.std()) for f in (sharp, mild, strong)]
    assert stds[0] > stds[1] > stds[2]
    # Measured on the reference implementation: 40.0, 19.5 and 7.4 (sigma 0, 2
    # and 6 px for z = 0, 20 and 60 um); a different scale of z shows here.
    assert stds == pytest.approx([40.0, 19.5, 7.4], abs=1.0)
    assert np.array_equal(frame_at(30.0), frame_at(-30.0))
    assert float(frame_at(40.0).mean()) == pytest.approx(float(sharp.mean()), abs=1.0)


def test_blur_does_not_wrap_the_frame_edges() -> None:
    """Without the padding, the FFT's periodic boundary leaks the far side in."""
    sample = PlateSample(**FLAT)
    # A 64 x 64 frame centred on the wall of A1 at x = 3200 um: well on the
    # left, plastic on the right, 31.5 um from either edge.
    frame = sample.render(3200.0, 0.0, 50.0, shape=(64, 64), pixel_size_um=1.0)
    assert np.all(np.abs(frame[:, 0].astype(np.int64) - WELL) <= 1)
    assert np.all(np.abs(frame[:, -1].astype(np.int64) - PLASTIC) <= 1)
    # The wall itself is blurred into a ramp.
    assert PLASTIC + 100 < frame[32, 32] < WELL - 100


@pytest.mark.parametrize(
    ("kwargs", "phrase"),
    [
        pytest.param(
            {"pixel_size_um": 0.0},
            "pixel_size_um must be finite and positive (at most 10000), got 0.0",
            id="pixel-size-zero",
        ),
        pytest.param(
            {"pixel_size_um": -1.0},
            "pixel_size_um must be finite and positive (at most 10000), got -1.0",
            id="pixel-size-negative",
        ),
        pytest.param(
            {"pixel_size_um": math.nan},
            "pixel_size_um must be finite and positive (at most 10000), got nan",
            id="pixel-size-nan",
        ),
        pytest.param(
            {"shape": (0, 8)},
            "shape must be two positive integers (height, width) with at most "
            "16777216 pixels, got (0, 8)",
            id="shape-with-a-zero",
        ),
        pytest.param(
            {"shape": (8,)},
            "shape must be two positive integers (height, width) with at most "
            "16777216 pixels, got (8,)",
            id="shape-of-one",
        ),
        pytest.param(
            {"shape": (8.5, 8)},
            "shape must be two positive integers (height, width) with at most "
            "16777216 pixels, got (8.5, 8)",
            id="shape-with-a-float",
        ),
        pytest.param(
            {"shape": (True, 8)},
            "shape must be two positive integers (height, width) with at most "
            "16777216 pixels, got (True, 8)",
            id="shape-with-a-bool",
        ),
        pytest.param(
            {"shape": (5000, 5000)},
            "shape must be two positive integers (height, width) with at most "
            "16777216 pixels, got (5000, 5000)",
            id="shape-too-large",
        ),
        pytest.param(
            {"x_um": math.nan},
            "x_um must be finite and within [-1e+09, 1e+09], got nan",
            id="x-nan",
        ),
        pytest.param(
            {"y_um": math.inf},
            "y_um must be finite and within [-1e+09, 1e+09], got inf",
            id="y-infinite",
        ),
        pytest.param(
            {"z_um": 1e30},
            "z_um must be finite and within [-1e+09, 1e+09], got 1e+30",
            id="z-huge",
        ),
        pytest.param(
            {"exposure_ms": -1.0},
            "exposure_ms must be finite and within [0, 1e+06], got -1.0",
            id="exposure-negative",
        ),
        pytest.param(
            {"exposure_ms": math.nan},
            "exposure_ms must be finite and within [0, 1e+06], got nan",
            id="exposure-nan",
        ),
        pytest.param(
            {"camera_rotation_deg": math.inf},
            "camera_rotation_deg must be finite and within [-1e+06, 1e+06], got inf",
            id="camera-rotation-infinite",
        ),
    ],
)
def test_render_refuses_invalid_arguments(
    kwargs: dict[str, object], phrase: str
) -> None:
    arguments: dict[str, object] = {
        "x_um": 0.0,
        "y_um": 0.0,
        "z_um": 0.0,
        "shape": (8, 8),
        "pixel_size_um": 1.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=re.escape(phrase)):
        PlateSample().render(**arguments)  # type: ignore[arg-type]


# --- the camera, on the fake stand -----------------------------------------


def _install(
    microscope: Microscope, sample: PlateSample | None = None, **kwargs: object
) -> SampleCamera:
    """A noise-free ``SampleCamera`` as the facade's camera, as the fixture does it."""
    camera = SampleCamera(
        microscope,
        sample if sample is not None else PlateSample(noise_std=0.0),
        pixel_size_um=kwargs.pop("pixel_size_um", 1.0),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    microscope.override(Camera, camera)
    return camera


@pytest.mark.parametrize("pixel_size_um", [1.0, 0.5, 2.0])
def test_sample_camera_follows_the_fake_stage(
    fake_microscope: Microscope, pixel_size_um: float
) -> None:
    camera = _install(fake_microscope, shape=(256, 256), pixel_size_um=pixel_size_um)
    stage = fake_microscope.require(XYStage)
    start = camera.snap().astype(np.float64)
    stage.move_by_um(10.0, 0.0)
    east = camera.snap().astype(np.float64)
    stage.move_by_um(0.0, 10.0)
    north = camera.snap().astype(np.float64)
    # +10 um in X moves the content 10 / ps columns left, +10 um in Y as many
    # rows down.
    pixels = 10.0 / pixel_size_um
    assert _content_shift(start, east) == pytest.approx((-pixels, 0.0), abs=0.5)
    assert _content_shift(east, north) == pytest.approx((0.0, pixels), abs=0.5)


def test_sample_camera_reads_z_from_the_stage(fake_microscope: Microscope) -> None:
    sample = PlateSample(well_level=1000.0, plastic_level=1000.0, noise_std=0.0)
    camera = _install(fake_microscope, sample, shape=(256, 256))
    sharp = camera.snap()
    fake_microscope.require(ZStage).move_to_um(30.0)
    blurred = camera.snap()
    assert float(blurred.std()) < 0.5 * float(sharp.std())


def test_sample_camera_without_a_z_stage_renders_at_z_zero() -> None:
    core = FakeCore(
        [FakeDevice("XY", DeviceType.XYStage), FakeDevice("Camera", DeviceType.Camera)],
        camera="Camera",
        xy_stage="XY",
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    try:
        assert microscope.get(ZStage) is None  # the stand really has no Z
        sample = PlateSample(noise_std=0.0)
        camera = SampleCamera(microscope, sample, pixel_size_um=1.0, shape=(64, 64))
        expected = sample.render(0.0, 0.0, 0.0, shape=(64, 64), pixel_size_um=1.0)
        assert np.array_equal(camera.snap(), expected)
    finally:
        microscope.close()


def test_sample_camera_honours_the_camera_basics(fake_microscope: Microscope) -> None:
    """The cases of ``tests/contracts/test_camera.py``, for a camera the facade holds."""
    camera = _install(fake_microscope, shape=(48, 80), pixel_size_um=0.65)
    frame = camera.snap()
    assert frame.ndim == 2
    assert frame.shape == camera.image_shape() == (48, 80)
    assert frame.dtype.itemsize * 8 >= camera.bit_depth()
    assert camera.bit_depth() == 16  # the frames use the whole uint16 range
    assert camera.pixel_size_um() == 0.65
    assert camera.exposure_ms() == REFERENCE_EXPOSURE_MS
    assert camera.set_exposure_ms(20.0) == 20.0
    assert camera.exposure_ms() == 20.0
    # The stored exposure reaches the frame: a flat 3000 doubles.
    flat = _install(fake_microscope, PlateSample(**FLAT), shape=(8, 8))
    assert np.all(flat.snap() == WELL)
    flat.set_exposure_ms(20.0)
    assert np.all(flat.snap() == 2 * WELL)
    assert fake_microscope.require(Camera) is flat


@pytest.mark.parametrize(
    ("rotation_deg", "mirrored"), [(90.0, False), (0.0, True), (90.0, True)]
)
def test_sample_camera_passes_its_rotation_and_mirror_to_the_render(
    fake_microscope: Microscope, rotation_deg: float, mirrored: bool
) -> None:
    sample = PlateSample(
        blobs=[Blob(30.0, 12.0, radius_um=6.0, intensity=1500.0)], **FLAT
    )
    camera = _install(
        fake_microscope,
        sample,
        shape=(96, 96),
        camera_rotation_deg=rotation_deg,
        mirrored=mirrored,
    )
    expected = sample.render(
        0.0,
        0.0,
        0.0,
        shape=(96, 96),
        pixel_size_um=1.0,
        camera_rotation_deg=rotation_deg,
        mirrored=mirrored,
    )
    assert np.array_equal(camera.snap(), expected)
    unturned = sample.render(0.0, 0.0, 0.0, shape=(96, 96), pixel_size_um=1.0)
    assert not np.array_equal(expected, unturned)  # the orientation does show


@pytest.mark.parametrize(
    ("kwargs", "phrase"),
    [
        pytest.param(
            {"pixel_size_um": 0.0},
            "pixel_size_um must be finite and positive (at most 10000), got 0.0",
            id="pixel-size-zero",
        ),
        pytest.param(
            {"pixel_size_um": math.nan},
            "pixel_size_um must be finite and positive (at most 10000), got nan",
            id="pixel-size-nan",
        ),
        pytest.param(
            {"shape": (0, 8)},
            "shape must be two positive integers (height, width) with at most "
            "16777216 pixels, got (0, 8)",
            id="shape-with-a-zero",
        ),
        pytest.param(
            {"camera_rotation_deg": math.nan},
            "camera_rotation_deg must be finite and within [-1e+06, 1e+06], got nan",
            id="camera-rotation-nan",
        ),
    ],
)
def test_sample_camera_refuses_invalid_arguments(
    fake_microscope: Microscope, kwargs: dict[str, object], phrase: str
) -> None:
    arguments: dict[str, object] = {"pixel_size_um": 1.0}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=re.escape(phrase)):
        SampleCamera(fake_microscope, PlateSample(), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value_ms", [-1.0, math.nan, math.inf])
def test_sample_camera_refuses_an_invalid_exposure_and_keeps_the_old_one(
    fake_microscope: Microscope, value_ms: float
) -> None:
    camera = _install(fake_microscope)
    camera.set_exposure_ms(15.0)
    with pytest.raises(
        ValueError, match=re.escape("exposure_ms must be finite and within [0, 1e+06]")
    ):
        camera.set_exposure_ms(value_ms)
    assert camera.exposure_ms() == 15.0


def test_rendering_a_512_frame_is_vectorised() -> None:
    """A tripwire for a per-pixel loop (about a second), not a speed claim: the
    50 ms target is measured and quoted in the PR, never asserted (FM-44)."""
    sample = PlateSample(blobs=[Blob(30.0, -20.0, radius_um=15.0, intensity=500.0)])
    for z_um in (0.0, 30.0):  # the second one runs the padded FFT blur
        best = min(_seconds(sample, z_um=z_um) for _ in range(5))
        assert best < 0.5, f"z={z_um}: {best:.3f} s"


def _seconds(sample: PlateSample, *, z_um: float) -> float:
    start = time.perf_counter()
    sample.render(0.0, 0.0, z_um, shape=(512, 512), pixel_size_um=1.0)
    return time.perf_counter() - start
