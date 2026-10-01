"""The ``Camera`` contract, on every backend (design §3, §8).

A real camera rounds the exposure to its own step, so the round trip is
within 1 %. The pixel-size fallback needs a camera without a calibration,
which only the fake can be made into, so that case runs on the fake alone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, Properties

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from conftest import Envelope
    from smc.testing import FakeCore


def test_snap_is_2d_and_matches_image_shape(camera: Camera) -> None:
    frame = camera.snap()
    assert frame.ndim == 2
    assert frame.shape == camera.image_shape()


def test_snap_dtype_holds_the_bit_depth(camera: Camera) -> None:
    frame = camera.snap()
    assert frame.dtype.itemsize * 8 >= camera.bit_depth()


def test_exposure_round_trips(camera: Camera, envelope: Envelope) -> None:
    start = envelope.exposure_ms
    assert start is not None
    asked = 30.0 if abs(start - 20.0) <= 1.0 else 20.0
    assert camera.set_exposure_ms(asked) == pytest.approx(asked, rel=0.01)
    assert camera.exposure_ms() == pytest.approx(asked, rel=0.01)


def test_pixel_size_is_unknown_or_positive(camera: Camera) -> None:
    size_um = camera.pixel_size_um()
    assert size_um == 0.0 or (math.isfinite(size_um) and size_um > 0)


def test_pixel_size_falls_back_to_the_profile_for_the_objective(
    fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    # Profile.demo(): "Nikon 10X S Fluor" 0.65, "Nikon 40X Plan Fluor ELWD"
    # 0.1625, at binning 1. The demo objective starts at position 1 (10X).
    fake_core.pixel_size_um = 0.0
    camera = fake_microscope.require(Camera)
    assert camera.pixel_size_um() == pytest.approx(0.65)
    fake_microscope.require(Properties).set("Camera", "Binning", "2")
    assert camera.pixel_size_um() == pytest.approx(1.3)
    fake_core.setState("Objective", 0)
    assert camera.pixel_size_um() == pytest.approx(0.325)
    fake_core.setState("Objective", 2)  # Objective-2: not in the profile
    assert camera.pixel_size_um() == 0.0
