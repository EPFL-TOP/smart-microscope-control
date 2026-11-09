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
from typing import Any

import pytest

from smc.hardware.capabilities import Camera, ObjectiveTurret, XYStage
from smc.hardware.errors import MicroscopeHaltedError
from smc.hardware.microscope import Microscope
from smc.hardware.profile import Profile, SampleSection
from smc.testing.sample_stand import StandSampleCamera
from smc.testing.synthetic import PlateSample

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


TEN = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The declared shift of the 40X (m3 §20.8): under it a feature is centred at its
#: position plus this, so the sample is rendered at the position minus this.
SHIFT_FORTY = (-32.5, 9.75)


@pytest.fixture
def shifted_stand(mm_available: bool) -> Iterator[Microscope]:
    """``demo-sample`` whose 40X is offset by ``SHIFT_FORTY`` and whose 10X (in place at open) has no entry."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    profile = Profile.load(DEMO_SAMPLE).model_copy(
        update={"sample": SampleSection(objective_shift_um={FORTY: SHIFT_FORTY})}
    )
    microscope = Microscope.open(profile)
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    """Where each snap of the sample camera is rendered: ``PlateSample.render``'s first two arguments.

    Two snaps at one place are not equal (the noise is drawn per snap), so the
    frames are never compared: the place the sample is rendered at is what the
    shift changes (m3 §20.1).
    """
    seen: list[tuple[float, float]] = []
    original = PlateSample.render

    def recording(
        self: PlateSample, x_um: float, y_um: float, *args: Any, **kwargs: Any
    ) -> Any:
        seen.append((x_um, y_um))
        return original(self, x_um, y_um, *args, **kwargs)

    monkeypatch.setattr(PlateSample, "render", recording)
    return seen


def test_an_objective_shift_moves_where_the_sample_is_rendered(
    shifted_stand: Microscope, renders: list[tuple[float, float]]
) -> None:
    stage = shifted_stand.require(XYStage)
    camera = shifted_stand.require(Camera)
    turret = shifted_stand.require(ObjectiveTurret)
    assert turret.current() == TEN
    stage.move_to_um(1000.0, -2000.0)

    # Under the 10X, which has no entry, the sample is rendered at the stage's
    # readback itself, never the target: the demo stage reads (123.495, -45.255)
    # after a move to (123.5, -45.25), m3 §20.1 (FM-142).
    here = stage.position_um()
    camera.snap()
    assert renders[-1] == pytest.approx((here.x_um, here.y_um), abs=1e-9)

    # Under the 40X it is rendered at the readback minus (-32.5, 9.75): a feature
    # is centred 32.5 um west and 9.75 um north of where the stage says.
    turret.select(FORTY, confirm=True)
    here = stage.position_um()
    camera.snap()
    assert renders[-1] == pytest.approx((here.x_um + 32.5, here.y_um - 9.75), abs=1e-9)


def test_a_stand_without_shifts_renders_at_the_stage_under_every_objective(
    sample_stand: Microscope, renders: list[tuple[float, float]]
) -> None:
    stage = sample_stand.require(XYStage)
    camera = sample_stand.require(Camera)
    sample_stand.require(ObjectiveTurret).select(FORTY, confirm=True)
    stage.move_to_um(1000.0, -2000.0)

    here = stage.position_um()
    camera.snap()

    assert renders[-1] == pytest.approx((here.x_um, here.y_um), abs=1e-9)


def test_the_default_demo_profile_is_unchanged(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    microscope = Microscope.open(Profile.demo())
    try:
        assert not isinstance(microscope.require(Camera), StandSampleCamera)
        assert microscope.profile.sample is None
    finally:
        microscope.close()
