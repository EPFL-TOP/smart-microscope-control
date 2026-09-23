"""Role resolution end to end on Micro-Manager's demo configuration."""

import pytest

from smc.hardware.roles import DeviceInfo, Role, core_roles, devices_from_core, resolve

pytestmark = pytest.mark.demo

CORE_SLOT_ROLES = (Role.camera, Role.xy_stage, Role.focus, Role.autofocus, Role.shutter)


def test_demo_configuration_resolves_camera_xy_focus_autofocus_shutter(
    demo_core,
) -> None:
    roles = resolve(devices_from_core(demo_core), core_roles=core_roles(demo_core))
    assert {role: roles.get(role) for role in CORE_SLOT_ROLES} == {
        Role.camera: "Camera",
        Role.xy_stage: "XY",
        Role.focus: "Z",
        Role.autofocus: "Autofocus",
        Role.shutter: "White Light Shutter",
    }
    assert all(roles.sources[role] == "core" for role in CORE_SLOT_ROLES)
    assert roles.warnings == []


def test_demo_heuristics_alone_find_the_same_devices(demo_core) -> None:
    # Without the core's slots, only the type table can find them: this is
    # what proves the types devices_from_core reports are the ones it expects.
    roles = resolve(devices_from_core(demo_core))
    for role in (Role.camera, Role.xy_stage, Role.focus, Role.autofocus):
        assert roles.sources[role] == "heuristic"
    assert roles.get(Role.camera) == demo_core.getCameraDevice()
    assert roles.get(Role.xy_stage) == demo_core.getXYStageDevice()
    assert roles.get(Role.focus) == demo_core.getFocusDevice()
    assert roles.get(Role.autofocus) == demo_core.getAutoFocusDevice()
    assert roles.get(Role.objective_turret) == "Objective"
    # Two shutters and no hint to tell them apart: reported, not hidden.
    assert sorted(roles.candidates[Role.shutter]) == [
        "LED Shutter",
        "White Light Shutter",
    ]


def test_devices_from_core_reports_labels_types_and_adapters(demo_core) -> None:
    devices = {d.label: d for d in devices_from_core(demo_core)}
    assert "Core" not in devices
    assert devices["XY"] == DeviceInfo(
        label="XY",
        type="XYStage",
        library="DemoCamera",
        name="DXYStage",
        description="Demo XY stage",
    )
    assert devices["Z"].type == "Stage"
    assert devices["Objective"].type == "State"
    assert devices["DHub"].type == "Hub"
