"""The demo-sample stand's cells and camera, without Micro-Manager (m1 §8, #260).

The cells are checked against the ground truth the sample itself gives
(``PlateSample.well_at``, which reads ``useq``'s plate geometry), never against
the arithmetic ``scatter_cells`` used to place them. The camera runs on
``fake_microscope`` with an inner stub whose frame is not square (FM-73).
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import threading
import time
import weakref
from collections import Counter
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pytest
import useq
from pymmcore_plus import DeviceType

from smc.hardware.capabilities import Camera, XYStage, ZStage
from smc.hardware.errors import HardwareError, MicroscopeHaltedError, ProfileError
from smc.hardware.microscope import Microscope
from smc.hardware.profile import Profile, SampleSection
from smc.plugins.cells.detector import BlobDetector
from smc.testing import FakeCore, FakeDevice
from smc.testing.sample_stand import (
    CELL_INTENSITY,
    CELL_REACH,
    DOUBLET_OFFSET,
    StandSampleCamera,
    centred_a1_um,
    install_sample,
    sample_lines,
    sample_of,
    scatter_cells,
)
from smc.testing.synthetic import Blob, PlateSample

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

FLAT = {"texture_std": 0.0, "noise_std": 0.0}
#: No level at all: the only thing in a frame is what the test put there.
DARK = {**FLAT, "well_level": 0.0, "plastic_level": 0.0}


def _counts_by_well(section: SampleSection) -> Counter[str]:
    """How many spots each well holds, by the sample's own ``well_at`` (the ground truth)."""
    sample = sample_of(section)
    found: Counter[str] = Counter()
    for blob in scatter_cells(section):
        well = sample.well_at(blob.x_um, blob.y_um)
        assert well is not None, f"a spot at ({blob.x_um}, {blob.y_um}) is on plastic"
        found[well] += 1
    return found


# -- where the plate lies ---------------------------------------------------------


def test_centred_a1_puts_the_plate_centre_on_the_origin() -> None:
    assert centred_a1_um("96-well") == pytest.approx((-49_500.0, 31_500.0))
    assert centred_a1_um("384-well") == pytest.approx((-51_750.0, 33_750.0))


@pytest.mark.parametrize("plate", ["6-well", "96-well", "384-well"])
def test_the_centred_plate_is_symmetric_about_the_origin_by_useq(plate: str) -> None:
    # The mean of every well's centre, from useq's own positions: the origin.
    centres = useq.WellPlatePlan(
        plate=plate,  # type: ignore[arg-type]
        a1_center_xy=centred_a1_um(plate),
    ).all_well_positions
    assert np.mean([p.x for p in centres]) == pytest.approx(0.0, abs=1e-6)
    assert np.mean([p.y for p in centres]) == pytest.approx(0.0, abs=1e-6)


def test_sample_of_puts_a1_where_the_section_says_or_centres_the_plate() -> None:
    centred = sample_of(SampleSection(cells_per_well=0))
    assert centred.a1_center_xy_um == pytest.approx((-49_500.0, 31_500.0))
    placed = sample_of(SampleSection(a1_center_xy_um=(10.0, -20.0), cells_per_well=0))
    assert placed.a1_center_xy_um == (10.0, -20.0)


# -- the cells ---------------------------------------------------------------------


def test_the_count_per_well_grows_from_half_to_one_and_a_half_the_mean() -> None:
    section = SampleSection(cells_per_well=40, doublet_fraction=0.0)
    found = _counts_by_well(section)

    # Column 1 holds half the mean, column 6 (index 5 of 12) 38.18 rounded, column 12 one and a half.
    for row in "ABCDEFGH":
        assert found[f"{row}1"] == 20
        assert found[f"{row}6"] == 38
        assert found[f"{row}12"] == 60
    assert sum(found.values()) == sum(round(40 * (0.5 + c / 11)) for c in range(12)) * 8
    # The mean over the columns stays near the number asked for.
    assert sum(found.values()) / 96 == pytest.approx(40, abs=0.5)


def test_a_three_column_plate_holds_half_the_mean_first_and_one_and_a_half_last() -> (
    None
):
    # No registered well plate has fewer than three columns, so the guard for one
    # column in `scatter_cells` (the mean, no division by zero) cannot be reached
    # through a `[sample]`: this is the smallest plate there is.
    section = SampleSection(plate="6-well", cells_per_well=10, doublet_fraction=0.0)
    found = _counts_by_well(section)
    assert found["A1"] == 5  # the first column is half the mean
    assert found["A2"] == 10
    assert found["A3"] == 15


@pytest.mark.parametrize(
    "section",
    [
        SampleSection(doublet_fraction=0.5),
        SampleSection(
            plate="384-well",
            rotation_deg=10.0,
            cells_per_well=20,
            doublet_fraction=0.5,
            cell_radius_um=20.0,
        ),
        SampleSection(
            plate="1536-well",
            cells_per_well=30,
            doublet_fraction=1.0,
            cell_radius_um=50.0,
            rotation_deg=-45.0,
        ),
        SampleSection(
            a1_center_xy_um=(1234.5, -2345.6), rotation_deg=7.0, doublet_fraction=0.5
        ),
    ],
    ids=["96-well", "384-well-turned-10", "1536-well-big-doublets", "96-well-placed"],
)
def test_every_spot_lies_inside_its_well(section: SampleSection) -> None:
    # `_counts_by_well` fails on a spot that is on plastic.
    found = _counts_by_well(section)
    assert sum(found.values()) == len(scatter_cells(section)) > 0


def test_the_wells_of_a_turned_square_plate_get_their_own_cells() -> None:
    # The count per well, by well_at, equals the formula: a spot that went to the
    # neighbour well (a rotation applied twice, or with the wrong sign) would not.
    section = SampleSection(
        plate="384-well", rotation_deg=10.0, cells_per_well=20, doublet_fraction=0.0
    )
    found = _counts_by_well(section)
    names = useq.WellPlatePlan(
        plate="384-well",  # type: ignore[arg-type]
        a1_center_xy=centred_a1_um("384-well"),
        rotation=10.0,
    ).all_well_names
    for column in range(24):
        expected = round(20 * (0.5 + column / 23))
        for row in range(16):
            assert found[str(names[row, column])] == expected


def test_cells_keep_to_the_reach_of_their_well() -> None:
    # CELL_REACH of the well's radius, so a cell and its doublet never touch the wall.
    section = SampleSection(doublet_fraction=0.0, cells_per_well=400)
    sample = sample_of(section)
    plan = useq.WellPlatePlan(
        plate="96-well",  # type: ignore[arg-type]
        a1_center_xy=centred_a1_um("96-well"),
    )
    centres = {p.name: (p.x, p.y) for p in plan.all_well_positions}
    radius_um = 6400.0 / 2
    for blob in scatter_cells(section):
        name = sample.well_at(blob.x_um, blob.y_um)
        assert name is not None
        cx, cy = centres[name]
        assert (
            math.hypot(blob.x_um - cx, blob.y_um - cy) <= CELL_REACH * radius_um + 1e-6
        )


def test_doublets_are_two_spots_either_side_of_a_centre() -> None:
    section = SampleSection(doublet_fraction=1.0, cells_per_well=10, cell_radius_um=5.0)
    singles = scatter_cells(SampleSection(doublet_fraction=0.0, cells_per_well=10))
    spots = scatter_cells(section)

    assert len(spots) == 2 * len(singles)
    for first, second in zip(spots[::2], spots[1::2], strict=True):
        # The two spots of a pair lie 2 x 0.8 x radius apart.
        assert math.hypot(
            first.x_um - second.x_um, first.y_um - second.y_um
        ) == pytest.approx(2 * DOUBLET_OFFSET * 5.0)


def test_a_spot_is_a_bright_round_blob_of_the_cell_radius() -> None:
    spots = scatter_cells(SampleSection(cells_per_well=2, cell_radius_um=7.5))
    assert spots
    assert {(b.radius_um, b.intensity) for b in spots} == {(7.5, CELL_INTENSITY)}


def test_the_same_seed_gives_the_same_cells_and_another_seed_others() -> None:
    first = scatter_cells(SampleSection(cells_per_well=5, seed=3))
    again = scatter_cells(SampleSection(cells_per_well=5, seed=3))
    other = scatter_cells(SampleSection(cells_per_well=5, seed=4))

    assert first == again
    assert first != other
    assert len(first) > 0


def test_no_cells_make_a_sample_with_no_blob() -> None:
    assert scatter_cells(SampleSection(cells_per_well=0)) == ()


def test_the_sample_keeps_the_seed_for_its_texture_and_noise() -> None:
    assert sample_of(SampleSection(seed=9)).seed == 9


def test_sample_lines_name_the_plate_and_the_limits() -> None:
    lines = sample_lines(SampleSection())

    assert len(lines) == 3
    assert lines[0] == (
        "Sample: a synthetic 96-well plate, A1 at (-49500.0, 31500.0) µm, turned 0°; "
        "40 cells per well on average, 30 % of them doublets (seed 0)"
    )
    assert "z = 0" in lines[1]
    assert "4 µm" in lines[2]
    assert "detector=blobs" in lines[2]


def test_sample_lines_follow_the_section() -> None:
    lines = sample_lines(
        SampleSection(
            plate="384-well",
            a1_center_xy_um=(1.0, 2.0),
            rotation_deg=-2.5,
            cells_per_well=8,
            doublet_fraction=0.125,
            seed=5,
        )
    )
    assert lines[0] == (
        "Sample: a synthetic 384-well plate, A1 at (1.0, 2.0) µm, turned -2.5°; "
        "8 cells per well on average, 12.5 % of them doublets (seed 5)"
    )


def test_sample_lines_are_text_a_windows_console_prints() -> None:
    # cp1252 is what a redirected doctor writes on Windows (FM-20).
    for line in sample_lines(SampleSection()):
        line.encode("cp1252")


# -- the camera ----------------------------------------------------------------------


class _Inner:
    """The stand's own camera, as far as the wrapper reads it, with a frame that is not square."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (60, 80),
        pixel_size_um: float = 1.0,
        exposure_ms: float = 10.0,
        bit_depth: int = 16,
    ) -> None:
        self.shape = shape
        self.pixel_size = pixel_size_um
        self.exposure = exposure_ms
        self.depth = bit_depth
        self.snaps = 0

    def snap(self) -> npt.NDArray[np.uint16]:
        self.snaps += 1
        return np.zeros(self.shape, dtype=np.uint16)

    def exposure_ms(self) -> float:
        return self.exposure

    def set_exposure_ms(self, value_ms: float) -> float:
        self.exposure = value_ms + 1.0  # a readback that is not the command
        return self.exposure

    def image_shape(self) -> tuple[int, int]:
        return (self.shape[0], self.shape[1])

    def bit_depth(self) -> int:
        return self.depth

    def pixel_size_um(self) -> float:
        return self.pixel_size


def _stand_camera(
    microscope: Microscope,
    inner: _Inner,
    sample: PlateSample | None = None,
    **kwargs: object,
) -> StandSampleCamera:
    return StandSampleCamera(
        microscope,
        sample if sample is not None else PlateSample(**DARK),
        inner,
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_stand_camera_renders_at_the_inner_frames_shape(
    fake_microscope: Microscope,
) -> None:
    # FM-73: a (60, 80) frame, a blob 20 um east of the stage. East is right of
    # the centre column (39.5) and on the middle rows (29.5): a height and width
    # swapped, or X and Y, put the maximum elsewhere.
    sample = PlateSample(blobs=[Blob(20.0, 0.0, 5.0, 3000.0)], **DARK)
    camera = _stand_camera(fake_microscope, _Inner(shape=(60, 80)), sample)

    frame = camera.snap()

    assert frame.shape == (60, 80)
    assert frame.dtype == np.uint16
    row, col = np.unravel_index(int(frame.argmax()), frame.shape)
    assert int(col) in (59, 60)
    assert int(row) in (29, 30)


def test_a_blob_north_of_the_stage_is_above_the_centre_row(
    fake_microscope: Microscope,
) -> None:
    sample = PlateSample(blobs=[Blob(0.0, 10.0, 3.0, 3000.0)], **DARK)
    frame = _stand_camera(fake_microscope, _Inner(shape=(60, 80)), sample).snap()

    row, col = np.unravel_index(int(frame.argmax()), frame.shape)
    assert int(row) in (19, 20)  # rows grow with -Y: +10 um is 10 rows above 29.5
    assert int(col) in (39, 40)


def test_the_stand_cameras_pose_turns_and_mirrors_the_frame(
    fake_microscope: Microscope,
) -> None:
    sample = PlateSample(blobs=[Blob(20.0, 0.0, 5.0, 3000.0)], **DARK)
    inner = _Inner(shape=(60, 80))
    mirrored = _stand_camera(fake_microscope, inner, sample, mirrored=True).snap()
    turned = _stand_camera(
        fake_microscope, inner, sample, camera_rotation_deg=90.0
    ).snap()

    _, mirrored_col = np.unravel_index(int(mirrored.argmax()), mirrored.shape)
    assert int(mirrored_col) in (19, 20)  # east is left when mirrored
    turned_row, turned_col = np.unravel_index(int(turned.argmax()), turned.shape)
    assert int(turned_col) in (39, 40)  # at 90 deg east is below the centre
    assert int(turned_row) in (49, 50)


@pytest.mark.parametrize("size", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_the_stand_camera_refuses_a_pixel_size_that_is_not_a_size(
    fake_microscope: Microscope, size: float
) -> None:
    camera = _stand_camera(fake_microscope, _Inner(pixel_size_um=size))

    with pytest.raises(HardwareError) as info:
        camera.snap()

    assert not isinstance(info.value, ValueError)
    # It names the objective in place and the key to set, and never guesses a scale.
    assert "Nikon 10X S Fluor" in str(info.value)
    assert "[camera] pixel_size_um" in str(info.value)


def test_the_stand_camera_snaps_the_inner_camera_first(
    fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The halt, the motion guard and the lock are the inner camera's: its refusal
    # must reach the caller before anything is rendered.
    sample = PlateSample(**DARK)
    rendered: list[object] = []
    monkeypatch.setattr(
        sample, "render", lambda *a, **k: rendered.append((a, k)), raising=True
    )

    class Halted(_Inner):
        def snap(self) -> npt.NDArray[np.uint16]:
            raise MicroscopeHaltedError

    camera = _stand_camera(fake_microscope, Halted(), sample)
    with pytest.raises(MicroscopeHaltedError):
        camera.snap()
    assert rendered == []


def test_the_stand_camera_snaps_the_inner_camera_once_per_frame(
    fake_microscope: Microscope,
) -> None:
    inner = _Inner()
    camera = _stand_camera(fake_microscope, inner)

    camera.snap()
    camera.snap()

    assert inner.snaps == 2


@pytest.mark.parametrize(
    ("shape", "size", "phrase"),
    [
        ((4, 4, 3), 1.0, "shape (4, 4, 3)"),
        ((8,), 1.0, "shape (8,)"),
        ((60, 80), 2e4, "pixel_size_um"),
    ],
    ids=["colour-frame", "1-d-frame", "pixel-size-too-large"],
)
def test_a_render_refusal_is_a_hardware_error(
    fake_microscope: Microscope, shape: tuple[int, ...], size: float, phrase: str
) -> None:
    camera = _stand_camera(fake_microscope, _Inner(shape=shape, pixel_size_um=size))

    with pytest.raises(HardwareError) as info:
        camera.snap()

    assert not isinstance(info.value, ValueError)
    assert phrase in str(info.value)


def test_a_position_the_sample_cannot_render_is_a_hardware_error(
    fake_microscope: Microscope,
) -> None:
    # An exposure that is not finite reaches render, which refuses with a ValueError.
    camera = _stand_camera(fake_microscope, _Inner(exposure_ms=math.nan))

    with pytest.raises(HardwareError, match="exposure_ms"):
        camera.snap()


def test_the_stand_camera_delegates_exposure_shape_depth_and_pixel_size(
    fake_microscope: Microscope,
) -> None:
    inner = _Inner(shape=(48, 80), pixel_size_um=0.65, exposure_ms=12.5, bit_depth=12)
    camera = _stand_camera(fake_microscope, inner)

    assert camera.image_shape() == (48, 80)
    assert camera.bit_depth() == 12
    assert camera.pixel_size_um() == 0.65
    assert camera.exposure_ms() == 12.5
    assert camera.set_exposure_ms(20.0) == 21.0  # the inner camera's readback
    assert camera.exposure_ms() == 21.0


def test_the_exposure_the_stand_has_is_the_one_the_frame_shows(
    fake_microscope: Microscope,
) -> None:
    # A flat well at 3000 counts at the 10 ms reference: twice that at 20 ms.
    sample = PlateSample(well_level=3000.0, plastic_level=3000.0, **FLAT)
    inner = _Inner(shape=(8, 8), exposure_ms=10.0)
    camera = _stand_camera(fake_microscope, inner, sample)
    assert np.all(camera.snap() == 3000)

    inner.exposure = 20.0

    assert np.all(camera.snap() == 6000)


def test_the_pixel_size_the_stand_has_is_the_scale_of_the_frame(
    fake_microscope: Microscope,
) -> None:
    # A blob 5 um east is 5 / ps columns right of the centre column (39.5): the
    # scale is the inner camera's, read at each snap.
    sample = PlateSample(blobs=[Blob(5.0, 0.0, 5.0, 3000.0)], **DARK)
    inner = _Inner(shape=(60, 80), pixel_size_um=0.5)
    camera = _stand_camera(fake_microscope, inner, sample)

    _, col = np.unravel_index(int(camera.snap().argmax()), (60, 80))
    assert int(col) in (49, 50)  # 39.5 + 10

    inner.pixel_size = 2.0
    _, col = np.unravel_index(int(camera.snap().argmax()), (60, 80))
    assert int(col) == 42  # 39.5 + 2.5


def test_the_stand_camera_reads_z_from_the_stage(fake_microscope: Microscope) -> None:
    sample = PlateSample(blobs=[Blob(0.0, 0.0, 5.0, 3000.0)], **DARK)
    camera = _stand_camera(fake_microscope, _Inner(shape=(64, 64)), sample)
    sharp = camera.snap()

    fake_microscope.require(ZStage).move_to_um(30.0)

    assert int(camera.snap().max()) < 0.5 * int(sharp.max())


def test_the_stand_camera_warns_once_above_the_textures_tile(
    fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    camera = _stand_camera(fake_microscope, _Inner(pixel_size_um=5.0))

    with caplog.at_level(logging.WARNING, logger="smc.testing.sample_stand"):
        camera.snap()
        camera.snap()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "aliased" in warnings[0].getMessage()


def test_the_stand_camera_does_not_warn_at_or_below_the_textures_tile(
    fake_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    camera = _stand_camera(fake_microscope, _Inner(pixel_size_um=4.0))

    with caplog.at_level(logging.WARNING, logger="smc.testing.sample_stand"):
        camera.snap()

    assert caplog.records == []


def test_the_stand_camera_refuses_once_its_microscope_is_gone(
    no_cyclic_gc: None,
) -> None:
    # FM-92: held weakly, so the facade and its core are freed by their last
    # reference, and a camera that outlives them says so instead of reading a dead stand.
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    camera = _stand_camera(microscope, _Inner())
    core.log.clear()
    del microscope

    with pytest.raises(HardwareError, match="no longer exists"):
        camera.snap()
    assert core.log == []


def test_an_installed_sample_camera_frees_the_core(no_cyclic_gc: None) -> None:
    # The facade holds the camera as its Camera: a strong reference back was a
    # cycle through the core, freed only by the cyclic collector (FM-92).
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    install_sample(microscope, SampleSection(cells_per_well=1))
    microscope.require(Camera).snap()
    watched = {"facade": weakref.ref(microscope), "core": weakref.ref(core)}

    del microscope, core

    assert {name: ref() is not None for name, ref in watched.items()} == {
        "facade": False,
        "core": False,
    }


# -- the facade installs it from the profile ------------------------------------------


def _sample_profile(**sample: object) -> Profile:
    return Profile.demo().model_copy(
        update={"sample": SampleSection(**sample)}  # type: ignore[arg-type]
    )


def test_from_core_installs_the_sample_when_the_profile_has_one(
    fake_core: FakeCore,
) -> None:
    microscope = Microscope.from_core(
        cast("CMMCorePlus", fake_core), _sample_profile(cells_per_well=0)
    )
    try:
        camera = microscope.require(Camera)
        assert isinstance(camera, StandSampleCamera)
        # The origin is plastic and E7 is a well: the frames follow the stage.
        at_origin = float(camera.snap().mean())
        microscope.require(XYStage).move_to_um(4500.0, -4500.0)
        on_a_well = float(camera.snap().mean())
        assert at_origin < 1500
        assert on_a_well > 2500
    finally:
        microscope.close()


def test_from_core_leaves_a_profile_without_a_sample_on_the_stands_own_camera(
    fake_core: FakeCore,
) -> None:
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), Profile.demo())
    try:
        assert not isinstance(microscope.require(Camera), StandSampleCamera)
    finally:
        microscope.close()


def test_a_stand_without_a_sample_never_imports_the_sample_stand() -> None:
    # In a fresh interpreter: another test may have imported the module already.
    code = (
        "import sys\n"
        "from smc.hardware.microscope import Microscope\n"
        "from smc.hardware.profile import Profile\n"
        "from smc.testing import FakeCore\n"
        "microscope = Microscope.from_core(FakeCore.demo_like(), Profile.demo())\n"
        "microscope.close()\n"
        "assert 'smc.testing.sample_stand' not in sys.modules, 'imported'\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_install_refuses_a_camera_that_is_not_the_demo_camera(
    fake_core: FakeCore,
) -> None:
    fake_core._device("Camera").library = "PVCAM"

    with pytest.raises(ProfileError, match="PVCAM") as info:
        Microscope.from_core(cast("CMMCorePlus", fake_core), _sample_profile())

    assert "DemoCamera" in str(info.value)
    # The refusal closed the facade: the connection is not left held.
    assert "unloadAllDevices()" in fake_core.log


def test_install_refuses_a_stand_with_no_camera() -> None:
    core = FakeCore([FakeDevice("XY", DeviceType.XYStage)], xy_stage="XY")

    with pytest.raises(ProfileError, match="camera role"):
        Microscope.from_core(cast("CMMCorePlus", core), _sample_profile())

    assert "unloadAllDevices()" in core.log


def test_a_failed_install_whose_close_also_fails_still_raises_the_refusal(
    fake_core: FakeCore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # FM-49: the close is attempted, and its own failure is named without
    # replacing the refusal the operator needs.
    fake_core._device("Camera").library = "PVCAM"

    def busy(self: Microscope) -> None:
        raise HardwareError("another call holds the lock")

    monkeypatch.setattr(Microscope, "close", busy)

    with (
        caplog.at_level(logging.WARNING, logger="smc.hardware.microscope"),
        pytest.raises(ProfileError, match="PVCAM"),
    ):
        Microscope.from_core(cast("CMMCorePlus", fake_core), _sample_profile())

    assert any("another call holds the lock" in r.getMessage() for r in caplog.records)


def test_install_logs_that_the_frames_are_rendered(
    fake_core: FakeCore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="smc.testing.sample_stand"):
        microscope = Microscope.from_core(
            cast("CMMCorePlus", fake_core), _sample_profile(plate="384-well")
        )
    microscope.close()

    assert any(
        "synthetic 384-well plate" in r.getMessage() and r.levelno == logging.INFO
        for r in caplog.records
    )


def test_a_keyboard_interrupt_during_the_install_still_closes_the_facade(
    fake_core: FakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-49: building the sample takes seconds on a big plate, and the operator's
    # Ctrl-C lands in it. It is a BaseException, not an Exception.
    def interrupted(microscope: Microscope, section: SampleSection) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("smc.testing.sample_stand.install_sample", interrupted)

    with pytest.raises(KeyboardInterrupt):
        Microscope.from_core(cast("CMMCorePlus", fake_core), _sample_profile())

    assert "unloadAllDevices()" in fake_core.log


# -- what the stand draws, and what the detector finds in it ----------------------------


def test_the_cell_constants_are_the_designs() -> None:
    # m1 §8 gives them. The tests that use the drawing lose a cell only when a
    # constant moves a lot, so each is pinned on its own.
    assert (CELL_INTENSITY, CELL_REACH, DOUBLET_OFFSET) == (3000.0, 0.85, 0.8)


def test_the_doublet_share_is_the_one_asked_for() -> None:
    # 3840 cells: a share of 0.3 is 0.3 +- 0.0074 (one standard deviation).
    cells = len(scatter_cells(SampleSection(doublet_fraction=0.0)))
    spots = len(scatter_cells(SampleSection(doublet_fraction=0.3)))

    assert 3800 < cells < 3880  # 96 wells at 40 on average, each rounded
    assert (spots - cells) / cells == pytest.approx(0.3, abs=0.04)


@pytest.mark.parametrize("plate", ["96-well", "384-well"])
def test_cells_are_spread_evenly_over_their_wells(plate: str) -> None:
    # Uniform over the disc (96-well) or the square (384-well) of CELL_REACH: a quarter
    # of the cells lie within half the reach, by area. A radius not taken as a square
    # root puts half of them there, and a square not centred a different share.
    section = SampleSection(plate=plate, doublet_fraction=0.0, cells_per_well=40)
    sample = sample_of(section)
    wells = useq.WellPlatePlan(
        plate=plate,  # type: ignore[arg-type]
        a1_center_xy=centred_a1_um(plate),
    )
    centres = {p.name: (p.x, p.y) for p in wells.all_well_positions}
    half_um = wells.plate.well_size[0] * 500.0
    inner = 0
    blobs = scatter_cells(section)
    for blob in blobs:
        name = sample.well_at(blob.x_um, blob.y_um)
        assert name is not None
        cx, cy = centres[name]
        dx, dy = blob.x_um - cx, blob.y_um - cy
        reach = 0.5 * CELL_REACH * half_um
        inner += (
            math.hypot(dx, dy) <= reach
            if wells.plate.circular_wells
            else max(abs(dx), abs(dy)) <= reach
        )

    assert inner / len(blobs) == pytest.approx(0.25, abs=0.04)


def test_the_blob_detector_finds_what_the_stand_draws() -> None:
    # H12 of the densest plate (750 cells, 23 per mm2): the frames around its centre
    # at 1 um per pixel hold about five cells each. Every cell that is in the frame,
    # away from its edge and from every other cell is found, with its label, at its
    # place (within 3 um), and nothing is found away from a drawn spot.
    section = SampleSection(cells_per_well=500, doublet_fraction=0.3)
    sample = sample_of(section)
    spots = scatter_cells(section)
    centre_x, centre_y = 49_500.0, -31_500.0  # H12 on the centred 96-well plate
    near = [
        s for s in spots if math.hypot(s.x_um - centre_x, s.y_um - centre_y) < 1000.0
    ]
    cells: list[tuple[float, float, str]] = []  # (x, y, label), from consecutive spots
    index = 0
    while index < len(near):
        first = near[index]
        second = near[index + 1] if index + 1 < len(near) else None
        if (
            second is not None
            and abs(
                math.hypot(first.x_um - second.x_um, first.y_um - second.y_um) - 8.0
            )
            < 1e-6
        ):
            cells.append(
                (
                    (first.x_um + second.x_um) / 2,
                    (first.y_um + second.y_um) / 2,
                    "doublet",
                )
            )
            index += 2
        else:
            cells.append((first.x_um, first.y_um, "single"))
            index += 1

    def alone(x: float, y: float) -> bool:
        return all(
            math.hypot(x - ox, y - oy) > 40.0 or (ox, oy) == (x, y)
            for ox, oy, _ in cells
        )

    matched = 0
    for dx, dy in [(-300.0, -300.0), (300.0, -300.0), (-300.0, 300.0), (300.0, 300.0)]:
        frame_x, frame_y = centre_x + dx, centre_y + dy
        found = [
            (
                frame_x + ((d.col0_px + d.col1_px) / 2 - 255.5),
                frame_y - ((d.row0_px + d.row1_px) / 2 - 255.5),
                d.label,
            )
            for d in BlobDetector().detect(
                sample.render(
                    frame_x, frame_y, 0.0, shape=(512, 512), pixel_size_um=1.0
                )
            )
        ]
        for x, y, label in cells:
            if abs(x - frame_x) <= 230 and abs(y - frame_y) <= 230 and alone(x, y):
                hits = [f for f in found if math.hypot(f[0] - x, f[1] - y) <= 3.0]
                assert [h[2] for h in hits] == [label], (x, y, label, found)
                matched += 1
        for fx, fy, _ in found:
            if abs(fx - frame_x) <= 230 and abs(fy - frame_y) <= 230:
                assert any(math.hypot(fx - x, fy - y) <= 40.0 for x, y, _ in cells), (
                    fx,
                    fy,
                )

    assert matched >= 6  # a frame with nothing in it would pass the rest


# -- the lock, and a stand with no Z ------------------------------------------------------


def test_two_snaps_at_once_never_render_at_once(
    fake_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sample owns a random generator that is not thread-safe: the render runs
    # under the camera's own lock. A render that takes 50 ms makes an overlap certain
    # when there is none.
    sample = PlateSample(**DARK)
    guard = threading.Lock()
    inside = peak = 0

    def slow_render(*args: object, **kwargs: object) -> npt.NDArray[np.uint16]:
        nonlocal inside, peak
        with guard:
            inside += 1
            peak = max(peak, inside)
        time.sleep(0.05)
        with guard:
            inside -= 1
        return np.zeros((60, 80), dtype=np.uint16)

    monkeypatch.setattr(sample, "render", slow_render)
    camera = _stand_camera(fake_microscope, _Inner(), sample)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def snap() -> None:
        try:
            barrier.wait(timeout=10)
            camera.snap()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=snap) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert peak == 1


def test_a_stand_without_a_z_stage_renders_at_z_zero() -> None:
    core = FakeCore(
        [FakeDevice("XY", DeviceType.XYStage), FakeDevice("Camera", DeviceType.Camera)],
        camera="Camera",
        xy_stage="XY",
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    try:
        assert microscope.get(ZStage) is None  # the stand really has no Z
        sample = PlateSample(blobs=[Blob(0.0, 0.0, 5.0, 3000.0)], **DARK)

        frame = _stand_camera(microscope, _Inner(shape=(64, 64)), sample).snap()

        # Sharp: a blur at some other z would spread the spot's 2900 counts.
        assert int(frame.max()) > 2500
    finally:
        microscope.close()
