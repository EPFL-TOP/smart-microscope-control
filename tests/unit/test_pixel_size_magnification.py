"""The profile's pixel size is divided by the core's magnification factor (m2 §8.10, #167).

The Ti2-E's intermediate-magnification lever (1.0x or 1.5x) scales the image
as much as the objective does, and the profile's sizes are for a factor of
1.0, the convention of MMCore's own pixel-size presets. The fake core has one
``Magnifier`` device, the lever, and no pixel-size calibration, so every size
here comes from the profile and the factor the fake reports.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest
from pymmcore_plus import DeviceType

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, ObjectiveTurret
from smc.hardware.errors import HardwareError
from smc.hardware.profile import Profile
from smc.testing import FakeCore, FakeDevice

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

LEVER = "IntermediateMagnification"
#: The demo profile's two sizes (10X 0.65 um, 40X 0.1625 um) and their labels.
TEN_X = "Nikon 10X S Fluor"
FORTY_X = "Nikon 40X Plan Fluor ELWD"


def _unknown(bad: float, objective: str = TEN_X) -> str:
    """What the camera's and the turret's pixel size say for a factor ``bad``, whole (FM-45)."""
    return (
        "the magnification factor (the product of the Magnifier devices) reads "
        f"{bad!r}, not a positive number; reporting 0.0 (unknown) for objective "
        f"{objective!r}"
    )


@pytest.fixture
def stand() -> Iterator[tuple[FakeCore, Microscope]]:
    """The demo layout plus the lever, with no pixel-size calibration in the core."""
    core = FakeCore.demo_like(
        extra_devices=[
            FakeDevice(LEVER, DeviceType.Magnifier, library="NikonTi2", name=LEVER)
        ]
    )
    core.pixel_size_um = 0.0
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    try:
        yield core, microscope
    finally:
        microscope.close()


def test_camera_fallback_divides_the_profile_size_by_the_magnification_factor(
    stand: tuple[FakeCore, Microscope],
) -> None:
    core, microscope = stand
    camera = microscope.require(Camera)
    # x / 1.0 == x: nothing changes for a stand with no magnifier or a 1.0x lever.
    assert camera.pixel_size_um() == 0.65
    core.magnification[LEVER] = 1.5
    assert camera.pixel_size_um() == pytest.approx(0.65 / 1.5)
    core.setProperty("Camera", "Binning", "2")
    assert camera.pixel_size_um() == pytest.approx(1.3 / 1.5)
    core.setStateLabel("Objective", FORTY_X)
    # Binning is still 2.
    assert camera.pixel_size_um() == pytest.approx(0.325 / 1.5)


def test_turret_pixel_size_divides_the_profile_size_by_the_magnification_factor(
    stand: tuple[FakeCore, Microscope],
) -> None:
    core, microscope = stand
    turret = microscope.require(ObjectiveTurret)
    assert turret.pixel_size_um() == 0.65
    core.magnification[LEVER] = 1.5
    assert turret.pixel_size_um() == pytest.approx(0.65 / 1.5)
    # Any label: the lever does not depend on the objective in the light path.
    assert turret.pixel_size_um(FORTY_X) == pytest.approx(0.1625 / 1.5)
    # A position the profile has no size for stays unknown.
    assert turret.pixel_size_um("Objective-2") == 0.0


def test_a_calibrated_core_value_is_not_divided_a_second_time(
    stand: tuple[FakeCore, Microscope],
) -> None:
    # MMCore's own value already includes the factor (it reads 0.625 for a
    # preset of 1.0 at 1.6x, measured), so only the profile's fallback divides.
    core, microscope = stand
    core.magnification[LEVER] = 1.5
    core.pixel_size_um = 0.37
    assert microscope.require(Camera).pixel_size_um() == 0.37


@pytest.mark.parametrize("bad", [0.0, -1.5, math.nan, math.inf])
def test_a_factor_that_is_not_a_positive_number_gives_unknown(
    stand: tuple[FakeCore, Microscope],
    bad: float,
    caplog: pytest.LogCaptureFixture,
) -> None:
    core, microscope = stand
    core.magnification[LEVER] = bad
    with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm"):
        assert microscope.require(Camera).pixel_size_um() == 0.0
        assert microscope.require(ObjectiveTurret).pixel_size_um() == 0.0
    # One warning per call, never de-duplicated, like a bad profile entry.
    assert caplog.messages == [_unknown(bad)] * 2


def test_the_warning_names_the_objective_that_was_asked_about(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    # The turret answers for any label, so the current objective (the 10X)
    # is not the one the operator asked about.
    core, microscope = stand
    core.magnification[LEVER] = 0.0
    with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm"):
        assert microscope.require(ObjectiveTurret).pixel_size_um(FORTY_X) == 0.0
    assert caplog.messages == [_unknown(0.0, FORTY_X)]


def test_a_core_pixel_size_that_is_not_finite_is_not_reported(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    # MMCore reads infinity for a factor of 0, and infinity is above 0.
    core, microscope = stand
    camera = microscope.require(Camera)
    core.magnification[LEVER] = 0.0
    core.pixel_size_um = math.inf
    with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm"):
        assert camera.pixel_size_um() == 0.0
    assert caplog.messages == [_unknown(0.0)]
    # With a valid factor the profile's size stands in for it.
    core.magnification[LEVER] = 1.0
    assert camera.pixel_size_um() == 0.65


def test_a_faulting_magnifier_is_a_hardware_error_naming_the_role(
    stand: tuple[FakeCore, Microscope],
) -> None:
    core, microscope = stand
    core.failing[LEVER] = RuntimeError("no answer")
    with pytest.raises(HardwareError) as camera_fault:
        microscope.require(Camera).pixel_size_um()
    assert str(camera_fault.value).startswith(
        "camera: getMagnificationFactor() failed ("
    )
    with pytest.raises(HardwareError) as turret_fault:
        microscope.require(ObjectiveTurret).pixel_size_um()
    assert str(turret_fault.value).startswith(
        "objective_turret: getMagnificationFactor() failed ("
    )
    # state() is the tolerant layer: an entry in errors, not an exception.
    state = microscope.state()
    assert state.pixel_size_um is None
    assert any(error.startswith("pixel_size: HardwareError") for error in state.errors)


def test_an_objective_the_profile_lacks_is_unknown_without_reading_the_magnifier(
    stand: tuple[FakeCore, Microscope],
) -> None:
    core, microscope = stand
    core.failing[LEVER] = RuntimeError("no answer")
    assert microscope.require(ObjectiveTurret).pixel_size_um("Objective-2") == 0.0
    core.setStateLabel("Objective", "Objective-2")
    assert microscope.require(Camera).pixel_size_um() == 0.0


def test_a_bad_profile_entry_is_unknown_before_the_magnifier_is_read(
    stand: tuple[FakeCore, Microscope], caplog: pytest.LogCaptureFixture
) -> None:
    # FM-43: both checks false, and the step after them observable. The profile
    # is checked first, so with the magnifier faulting the answer is still 0.0
    # and the only warning is the profile's, not a HardwareError.
    core, microscope = stand
    microscope.profile.camera.pixel_size_um[TEN_X] = math.nan
    core.failing[LEVER] = RuntimeError("no answer")
    with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm"):
        assert microscope.require(Camera).pixel_size_um() == 0.0
        assert microscope.require(ObjectiveTurret).pixel_size_um() == 0.0
    warning = (
        "pixel size for objective 'Nikon 10X S Fluor' in the profile is nan, not a "
        "positive number; reporting 0.0 (unknown). Fix it under [camera] "
        "pixel_size_um in the profile."
    )
    assert caplog.messages == [warning, warning]


def test_both_pixel_sizes_are_read_while_the_microscope_is_halted(
    stand: tuple[FakeCore, Microscope],
) -> None:
    # The factor is a read: Executor.read takes no lock and a halt does not
    # refuse it (design §13), so a stopped stand still reports its pixel size.
    # An acquire would raise MicroscopeHaltedError here (FM-15).
    core, microscope = stand
    core.magnification[LEVER] = 1.5
    microscope.stop()
    expected = pytest.approx(0.65 / 1.5)
    assert microscope.require(ObjectiveTurret).pixel_size_um() == expected
    assert microscope.require(Camera).pixel_size_um() == expected
