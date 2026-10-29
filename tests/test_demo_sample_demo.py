"""The demo-sample stand on the demo devices: the profile's camera images a plate (#260).

The facade builds the stand from ``profiles/demo-sample.toml``, so what these
tests open is what ``smc run -p demo-sample`` and ``smc ui plate -p demo-sample``
open. The thresholds sit far from the measured values (plastic 800, a well with
its cells about 3000) and from each other, and every test compares with a fixed
bound, never with the value of another open of the demo (FM-47).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from smc.hardware.capabilities import Camera, XYStage
from smc.hardware.errors import MicroscopeHaltedError
from smc.hardware.microscope import Microscope
from smc.hardware.profile import Profile
from smc.testing.sample_stand import StandSampleCamera

pytestmark = pytest.mark.demo

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
#: E7 of the centred 96-well plate: A1 at (-49 500, 31 500) um, 9 mm pitch, six
#: columns across and four rows down.
E7_UM = (4500.0, -4500.0)


@pytest.fixture
def sample_stand(mm_available: bool) -> Iterator[Microscope]:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    microscope = Microscope.open(DEMO_SAMPLE)
    try:
        yield microscope
    finally:
        microscope.close()


def test_the_profile_opens_a_stand_whose_camera_is_the_sample_camera(
    sample_stand: Microscope,
) -> None:
    camera = sample_stand.require(Camera)

    assert isinstance(camera, StandSampleCamera)
    assert sample_stand.profile.microscope.name == "demo-sample"


def test_a_frame_changes_when_the_stage_moves_onto_a_well(
    sample_stand: Microscope,
) -> None:
    camera = sample_stand.require(Camera)

    at_origin = camera.snap()
    sample_stand.require(XYStage).move_to_um(*E7_UM)
    on_a_well = camera.snap()

    # The origin is plastic (800); E7 is a well with its cells (about 3000).
    assert float(at_origin.mean()) < 1500
    assert float(on_a_well.mean()) > 2500
    # The frame has the camera's own shape: the wrapper renders at it.
    assert at_origin.shape == on_a_well.shape == camera.image_shape()


def test_a_snap_is_refused_while_halted(sample_stand: Microscope) -> None:
    # The snap goes through the Executor: a rendering that skipped it would
    # return a frame from a stopped stand.
    sample_stand.stop()

    with pytest.raises(MicroscopeHaltedError):
        sample_stand.require(Camera).snap()


def test_a_dry_run_snaps(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    microscope = Microscope.open(DEMO_SAMPLE, dry_run=True)
    try:
        frame = microscope.require(Camera).snap()
    finally:
        microscope.close()

    assert frame.ndim == 2
    assert float(frame.mean()) < 1500  # the origin: plastic


def test_the_default_demo_profile_is_unchanged(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    microscope = Microscope.open(Profile.demo())
    try:
        assert not isinstance(microscope.require(Camera), StandSampleCamera)
        assert microscope.profile.sample is None
    finally:
        microscope.close()
