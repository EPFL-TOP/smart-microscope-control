"""What the layer knows about the two Nikon stands (design m2 §7, #12).

Pure: the measured device lists under ``tests/data`` go through ``resolve()``,
the registry and the file checks work on ``tmp_path``, and no test loads an
adapter or a vendor DLL (FM-40).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smc.hardware.roles import DeviceInfo, Role, RoleMap, resolve

DATA = Path(__file__).resolve().parents[1] / "data"

#: The core's own slots in the lab's ``nikon-ti2.cfg`` (``Property,Core,…``).
TI2_CFG_CORE_ROLES = {
    Role.camera: "Camera-1",
    Role.shutter: "DiaLamp",
    Role.focus: "ZDrive",
    Role.xy_stage: "XYStage",
    Role.autofocus: "PFS",
}


def measured(name: str) -> list[DeviceInfo]:
    """A measured device list of ``tests/data``, as the resolver sees it."""
    data = json.loads((DATA / name).read_text(encoding="utf-8"))
    return [
        DeviceInfo(
            label=d["label"], type=d["type"], library=d["library"], name=d["label"]
        )
        for d in data["devices"]
    ]


@pytest.fixture(scope="module")
def ti2_devices() -> list[DeviceInfo]:
    return measured("nikon-ti2-devices.json")


@pytest.fixture(scope="module")
def tie_devices() -> list[DeviceInfo]:
    return measured("nikon-tie-devices.json")


def filled_labels(roles: RoleMap) -> set[str]:
    """Every label that fills a role or is a candidate for one."""
    return {c for labels in roles.candidates.values() for c in labels}


# -- roles on the two measured lists (m2 §7.8) ------------------------------------


@pytest.mark.parametrize("with_cfg", [False, True], ids=["heuristic", "cfg-core-roles"])
def test_ti2_measured_list_resolves_every_role_but_light_source(
    ti2_devices: list[DeviceInfo], with_cfg: bool
) -> None:
    roles = resolve(
        ti2_devices,
        core_roles=TI2_CFG_CORE_ROLES if with_cfg else None,
        overrides={Role.filter_turret: "FilterTurret2"},
    )
    expected = {
        Role.camera: "Camera-1",
        Role.xy_stage: "XYStage",
        Role.focus: "ZDrive",
        Role.autofocus: "PFS",
        Role.autofocus_offset: "PFSOffset",
        Role.objective_turret: "Nosepiece",
        Role.shutter: "DiaLamp",
        Role.light_path: "LightPath",
        Role.filter_turret: "FilterTurret2",
    }
    assert {r: roles.get(r) for r in expected} == expected
    # The roles with one possible device never offer a second one.
    for role in (Role.xy_stage, Role.focus, Role.objective_turret):
        assert len(roles.candidates[role]) == 1, role
    assert roles.candidates[Role.filter_turret] == ["FilterTurret2", "FilterTurret1"]
    assert roles.candidates[Role.shutter] == [
        "DiaLamp",
        "Turret1Shutter",
        "Turret2Shutter",
    ]
    assert roles.sources[Role.filter_turret] == "profile"
    # light_source is #124's: nothing is asserted about it here.
    assert "IntermediateMagnification" not in filled_labels(roles)
    assert roles.warnings == []


def test_ti2_without_the_profile_the_filter_turret_is_a_choice_between_two(
    ti2_devices: list[DeviceInfo],
) -> None:
    roles = resolve(ti2_devices, core_roles=TI2_CFG_CORE_ROLES)
    assert roles.get(Role.filter_turret) == "FilterTurret1"
    assert roles.candidates[Role.filter_turret] == ["FilterTurret1", "FilterTurret2"]


def test_ti2_xy_stage_never_offers_a_tirf_positioner(
    ti2_devices: list[DeviceInfo],
) -> None:
    roles = resolve(ti2_devices)
    assert roles.candidates[Role.xy_stage] == ["XYStage"]


def test_ti2_config_naming_pfsoffset_as_focus_falls_back_to_zdrive_with_a_warning(
    ti2_devices: list[DeviceInfo],
) -> None:
    roles = resolve(ti2_devices, core_roles={Role.focus: "PFSOffset"})
    assert roles.get(Role.focus) == "ZDrive"
    assert roles.sources[Role.focus] == "heuristic"
    [warning] = roles.warnings
    assert warning.startswith("focus: ")
    assert "'PFSOffset'" in warning
    assert "Property,Core,Focus" in warning
    assert "using 'ZDrive' (heuristic)" in warning


def test_ti2_config_naming_tirf1_as_xy_falls_back_to_xystage_with_a_warning(
    ti2_devices: list[DeviceInfo],
) -> None:
    roles = resolve(ti2_devices, core_roles={Role.xy_stage: "TIRF1"})
    assert roles.get(Role.xy_stage) == "XYStage"
    assert roles.sources[Role.xy_stage] == "heuristic"
    [warning] = roles.warnings
    assert warning.startswith("xy_stage: ")
    assert "'TIRF1'" in warning
    assert "Property,Core,XYStage" in warning
    assert "using 'XYStage' (heuristic)" in warning


def test_tie_measured_list_resolves_every_role_but_light_source(
    tie_devices: list[DeviceInfo],
) -> None:
    roles = resolve(tie_devices)
    expected = {
        Role.xy_stage: "TIXYDrive",
        Role.focus: "TIZDrive",
        Role.autofocus: "TIPFSStatus",
        Role.autofocus_offset: "TIPFSOffset",
        Role.objective_turret: "TINosePiece",
        Role.shutter: "TIEpiShutter",
        Role.light_path: "TILightPath",
    }
    assert {r: roles.get(r) for r in expected} == expected
    # The Ti-E publishes no camera: it is the camera adapter's device.
    assert roles.get(Role.camera) is None
    assert roles.candidates[Role.shutter] == [
        "TIEpiShutter",
        "TIDiaShutter",
        "TIDiaLamp",
        "TIAuxShutter",
    ]
    assert sorted(roles.candidates[Role.filter_turret]) == [
        "TIBarrierFilterWheel",
        "TIExcitationFilterWheel",
        "TIFilterBlock1",
        "TIFilterBlock2",
    ]
    assert "TIAnalyzer" not in filled_labels(roles)
    assert roles.warnings == []


def test_tie_focus_is_never_the_pfs_offset_or_the_tirf_drive(
    tie_devices: list[DeviceInfo],
) -> None:
    roles = resolve(tie_devices)
    assert roles.candidates[Role.focus] == ["TIZDrive"]


def test_tie_config_naming_titirf_as_focus_falls_back_to_tizdrive_with_a_warning(
    tie_devices: list[DeviceInfo],
) -> None:
    roles = resolve(tie_devices, core_roles={Role.focus: "TITIRF"})
    assert roles.get(Role.focus) == "TIZDrive"
    assert roles.sources[Role.focus] == "heuristic"
    [warning] = roles.warnings
    assert warning.startswith("focus: ")
    assert "'TITIRF'" in warning
    assert "Property,Core,Focus" in warning
    assert "using 'TIZDrive' (heuristic)" in warning
