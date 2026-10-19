import pickle

import pytest

from smc.hardware.core import CoreError
from smc.hardware.errors import (
    AutofocusLostError,
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


def test_a_hint_replaces_the_roles_advice() -> None:
    # m2 §8.6: Channels needs a config group, not a role.
    err = CapabilityMissingError(
        "Channels",
        None,
        ("Camera", "System"),
        hint="no config group is named for channels; name one in the profile's "
        "[channels] group",
    )
    assert str(err) == (
        "Channels is not available: no config group is named for channels; name "
        "one in the profile's [channels] group (available: Camera, System)."
    )
    assert "[roles.assign]" not in str(err)


def test_capability_missing_survives_a_worker_process() -> None:
    for error in (
        CapabilityMissingError("XYStage", Role.xy_stage, ("camera",)),
        CapabilityMissingError("Channels", None, ("Camera",), hint="name a group"),
    ):
        copy = pickle.loads(pickle.dumps(error))
        assert (str(copy), copy.role, copy.available, copy.hint) == (
            str(error),
            error.role,
            error.available,
            error.hint,
        )


def test_safety_refused_carries_how_to_force() -> None:
    err = SafetyRefusedError(
        "jog of 9000 µm exceeds the 5000 µm guard", "pass force=True"
    )
    assert err.how_to_force == "pass force=True"
    assert "pass force=True" in str(err)


def test_autofocus_lost_is_a_hardware_error_and_not_a_refusal() -> None:
    # FM-64: the move was made, so a tool that skips a target on a refusal
    # must not catch this one and carry on.
    assert issubclass(AutofocusLostError, HardwareError)
    assert not issubclass(AutofocusLostError, SafetyRefusedError)


def test_autofocus_lost_survives_a_worker_process() -> None:
    error = AutofocusLostError("z: moved, no lock", 12.0, False)
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.z_um, copy.in_range) == ("z: moved, no lock", 12.0, False)
