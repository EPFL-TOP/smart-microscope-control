"""The synthetic sample on the demo devices: the fixture and the frames it gives (#10).

The demo's XY stage does the moving and the sample camera does the looking,
so these tests cover what the unit tests cannot: that the fixture wires a
``SampleCamera`` into a facade over the real simulator, with the demo's own
frame size and pixel size.

The demo's XY stage rounds a command (``move_by_um(10, 0)`` reads back
10.005 um), so frames are compared with the *readback*, never the command.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from smc.hardware.capabilities import Camera, XYStage
from smc.testing import Blob, PlateSample, SampleCamera

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope
    from smc.testing.fixtures import SampleMicroscopeFactory

pytestmark = pytest.mark.demo

#: The defaults of ``PlateSample``, which these tests rely on.
WELL, PLASTIC = 3000, 800


def _content_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """Where the content of ``before`` went in ``after``: ``(d_col, d_row)`` pixels.

    FFT cross-correlation (mean-subtracted ``rfft2`` product with the
    conjugate, argmax, wrapped to signed). Positive ``d_col`` is to the right,
    positive ``d_row`` is down the image.
    """
    fa = np.fft.rfft2(before - before.mean())
    fb = np.fft.rfft2(after - after.mean())
    corr = np.fft.irfft2(np.conj(fa) * fb, s=before.shape)
    row, col = np.unravel_index(np.argmax(corr), corr.shape)
    height, width = before.shape
    d_row = row - height if row > height // 2 else row
    d_col = col - width if col > width // 2 else col
    return float(d_col), float(d_row)


class _NoPixelSizeCamera:
    """A camera that reports what an uncalibrated one does: a pixel size of 0."""

    def snap(self) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.uint16)

    def exposure_ms(self) -> float:
        return 10.0

    def set_exposure_ms(self, value_ms: float) -> float:
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (4, 4)

    def bit_depth(self) -> int:
        return 16

    def pixel_size_um(self) -> float:
        return 0.0


def test_the_frame_changes_when_the_demo_stage_crosses_a_well_wall(
    demo_microscope_with_sample: SampleMicroscopeFactory,
) -> None:
    # A1 at x = -2800 um with 6.4 mm wells puts the wall at x = 400 um. The
    # fixture takes the demo's own frame size and pixel size, so this also
    # proves the defaults.
    sample = PlateSample(a1_center_xy_um=(-2800.0, 0.0), texture_std=0.0, noise_std=0.0)
    microscope = demo_microscope_with_sample(sample)
    camera = microscope.require(Camera)
    height, width = camera.image_shape()
    # The frame stays clear of the wall at both stops (FM-47: from what the
    # demo reports, not from constants).
    half_diagonal_um = camera.pixel_size_um() * np.hypot(height, width) / 2
    assert half_diagonal_um < 400.0 - 1.0
    stage = microscope.require(XYStage)

    assert np.all(camera.snap() == WELL)
    stage.move_to_um(800.0, 0.0)
    assert 800.0 - half_diagonal_um > 400.0
    assert np.all(camera.snap() == PLASTIC)


def test_sample_camera_follows_the_demo_stage(
    demo_microscope_with_sample: SampleMicroscopeFactory,
) -> None:
    microscope = demo_microscope_with_sample(PlateSample(noise_std=0.0))
    camera = microscope.require(Camera)
    stage = microscope.require(XYStage)

    before = stage.position_um()
    first = camera.snap().astype(np.float64)
    stage.move_by_um(10.0, 0.0)
    after = stage.position_um()
    second = camera.snap().astype(np.float64)

    moved_px = round((after.x_um - before.x_um) / camera.pixel_size_um())
    assert moved_px != 0
    assert after.y_um == before.y_um
    assert _content_shift(first, second) == pytest.approx((-moved_px, 0.0), abs=0.5)


def test_the_fixture_takes_shape_and_pixel_size_from_the_demo_camera(
    demo_microscope: Microscope,
    demo_microscope_with_sample: SampleMicroscopeFactory,
) -> None:
    demo_camera = demo_microscope.require(Camera)
    shape, pixel_size_um = demo_camera.image_shape(), demo_camera.pixel_size_um()
    assert pixel_size_um > 0

    microscope = demo_microscope_with_sample()
    assert microscope is demo_microscope
    camera = microscope.require(Camera)
    assert camera.image_shape() == shape
    assert camera.pixel_size_um() == pixel_size_um
    frame = camera.snap()
    assert frame.shape == shape
    # The default sample puts A1's centre at the demo stage's origin: the
    # whole frame is inside the well, at its level.
    assert float(frame.mean()) == pytest.approx(WELL, abs=10.0)

    # What the call gives wins over what the demo camera reports.
    demo_microscope_with_sample(pixel_size_um=0.5, shape=(64, 96))
    camera = microscope.require(Camera)
    assert camera.image_shape() == (64, 96)
    assert camera.pixel_size_um() == 0.5
    assert camera.snap().shape == (64, 96)


def test_the_fixture_passes_the_camera_orientation_on(
    demo_microscope_with_sample: SampleMicroscopeFactory,
) -> None:
    sample = PlateSample(
        blobs=[Blob(30.0, 12.0, radius_um=6.0, intensity=1500.0)],
        texture_std=0.0,
        noise_std=0.0,
    )
    microscope = demo_microscope_with_sample(
        sample,
        shape=(96, 96),
        pixel_size_um=1.0,
        camera_rotation_deg=90.0,
        mirrored=True,
    )
    position = microscope.require(XYStage).position_um()
    expected = sample.render(
        position.x_um,
        position.y_um,
        0.0,
        shape=(96, 96),
        pixel_size_um=1.0,
        camera_rotation_deg=90.0,
        mirrored=True,
    )
    assert np.array_equal(microscope.require(Camera).snap(), expected)


def test_the_fixture_refuses_a_camera_without_a_pixel_size_and_changes_nothing(
    demo_microscope: Microscope,
    demo_microscope_with_sample: SampleMicroscopeFactory,
) -> None:
    uncalibrated = _NoPixelSizeCamera()
    demo_microscope.override(Camera, uncalibrated)

    with pytest.raises(ValueError, match=re.escape("give pixel_size_um=")):
        demo_microscope_with_sample()
    assert demo_microscope.require(Camera) is uncalibrated

    # Giving the pixel size is the way out, and the call's shape is not needed.
    demo_microscope_with_sample(pixel_size_um=0.65)
    assert demo_microscope.require(Camera).pixel_size_um() == 0.65


def test_the_sample_camera_is_the_facades_camera(
    demo_microscope: Microscope,
    demo_microscope_with_sample: SampleMicroscopeFactory,
) -> None:
    microscope = demo_microscope_with_sample(shape=(128, 160), pixel_size_um=0.5)
    first = microscope.require(Camera)
    assert isinstance(first, SampleCamera)

    state = microscope.state()
    assert state.image_shape == (128, 160)
    assert state.pixel_size_um == 0.5
    assert state.errors == []

    # Calling the factory again replaces the camera, it does not stack them.
    microscope = demo_microscope_with_sample(shape=(32, 32), pixel_size_um=2.0)
    second = microscope.require(Camera)
    assert isinstance(second, SampleCamera)
    assert second is not first
    assert microscope.state().image_shape == (32, 32)
