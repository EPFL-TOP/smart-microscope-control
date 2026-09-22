import pytest

from smc.hardware.core import CoreError
from smc.hardware.errors import (
    CapabilityMissingError,
    HardwareError,
    SafetyRefusedError,
)
from smc.hardware.roles import DeviceInfo, Role


def test_role_values_are_the_profile_keys() -> None:
    assert Role("xy_stage") is Role.xy_stage
    assert {r.value for r in Role} == {r.name for r in Role}


def test_device_info_is_immutable() -> None:
    dev = DeviceInfo(label="XY", type="XYStage")
    with pytest.raises(AttributeError):
        dev.label = "other"  # type: ignore[misc]


def test_core_error_is_a_hardware_error() -> None:
    assert issubclass(CoreError, HardwareError)


def test_capability_missing_names_the_role_and_what_is_available() -> None:
    err = CapabilityMissingError("XYStage", Role.xy_stage, ("camera", "focus"))
    assert "'xy_stage'" in str(err)
    assert "camera, focus" in str(err)
    assert "[roles.assign]" in str(err)


def test_safety_refused_carries_how_to_force() -> None:
    err = SafetyRefusedError(
        "jog of 9000 µm exceeds the 5000 µm guard", "pass force=True"
    )
    assert err.how_to_force == "pass force=True"
    assert "pass force=True" in str(err)
