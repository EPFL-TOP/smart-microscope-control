"""What the layer knows about the two Nikon stands (design m2 §7, #12).

Pure: the measured device lists under ``tests/data`` go through ``resolve()``,
the registry and the file checks work on ``tmp_path``, and no test loads an
adapter or a vendor DLL (FM-40).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from smc.discovery import dll_probe
from smc.discovery.dll_probe import DllProbe, describe_winerror
from smc.hardware.backends import nikon
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


# -- the registry (m2 §7.2) -------------------------------------------------------


def touch(path: Path, data: bytes = b"MZ") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_registry_maps_both_nikon_adapters_and_keys() -> None:
    assert nikon.stand_for_adapter("NikonTi2") is nikon.TI2
    assert nikon.stand_for_adapter("NikonTI") is nikon.TIE
    assert nikon.stand_for_adapter("DemoCamera") is None
    assert nikon.stand_for_key("ti2") is nikon.TI2
    assert nikon.stand_for_key("tie") is nikon.TIE
    # nikon-control's key for the Ti-E is "ti"; ours is "tie".
    assert nikon.stand_for_key("ti") is None
    assert nikon.STANDS == (nikon.TI2, nikon.TIE)

    assert (nikon.TI2.key, nikon.TI2.adapter, nikon.TI2.name) == (
        "ti2",
        "NikonTi2",
        "Nikon Ti2-E",
    )
    assert (nikon.TIE.key, nikon.TIE.adapter, nikon.TIE.name) == (
        "tie",
        "NikonTI",
        "Nikon Ti-E",
    )
    assert (nikon.TI2.hub, nikon.TI2.dynamic) == ("", True)
    assert (nikon.TIE.hub, nikon.TIE.dynamic) == ("TIScope", False)
    assert nikon.TI2.driver == nikon.Driver(
        dll="Ti2_Mic_Driver.dll",
        sdk_dir=r"C:\Program Files\Nikon\Ti2-SDK\bin",
        installer="Ti2 Control (the Ti2 SDK)",
        beside_adapter=True,
    )
    assert nikon.TIE.driver == nikon.Driver(
        dll="NikonTi.dll",
        sdk_dir=r"C:\Program Files\Nikon\Shared\Bin",
        installer="Nikon's Ti Setup Tool (or the TiSDKRedist package)",
        beside_adapter=False,
    )
    assert nikon.TI2.adapter_file == "mmgr_dal_NikonTi2.dll"
    assert nikon.TIE.adapter_file == "mmgr_dal_NikonTI.dll"


def test_adapter_names_are_matched_without_regard_to_case() -> None:
    # Windows finds mmgr_dal_nikonti2.dll as it finds mmgr_dal_NikonTi2.dll,
    # and NikonTi2 and NikonTI differ by more than their case.
    assert nikon.stand_for_adapter("nikonti2") is nikon.TI2
    assert nikon.stand_for_adapter("NIKONTI") is nikon.TIE
    assert nikon.stand_for_key(" TIE ") is nikon.TIE


def test_an_empty_list_means_the_sdk_on_a_ti2_and_nothing_on_a_tie() -> None:
    assert "SDK" in nikon.TI2.empty_list_meaning
    assert "could not be reached" in nikon.TI2.empty_list_meaning
    assert "proves nothing" in nikon.TIE.empty_list_meaning
    assert "only a probe does" in nikon.TIE.empty_list_meaning


def test_stands_in_config_reads_the_library_column_not_the_label(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "stand.cfg"
    lines = [
        "# Device,Commented,NikonTi2,Commented",
        "Property,Core,Camera,NikonTI",
        "Device,NikonTi2,DemoCamera,DCam",
        "Device,TIScope,NikonTI,TIScope",
        "Device,Ti2-E__0,NikonTi2,Ti2-E__0",
        "Device,TIZDrive,NikonTI,TIZDrive",
        "Parent,TIZDrive,TIScope",
    ]
    cfg.write_bytes("\r\n".join(lines).encode("utf-8") + b"\r\n")

    # A device labelled NikonTi2 whose library is DemoCamera is not a stand,
    # a comment is skipped, a stand counts once, in order of first appearance.
    assert nikon.stands_in_config(cfg) == [nikon.TIE, nikon.TI2]


def test_stands_in_config_finds_nothing_in_a_demo_configuration(tmp_path: Path) -> None:
    cfg = tmp_path / "demo.cfg"
    cfg.write_text("Device,Camera,DemoCamera,DCam\n", encoding="utf-8")

    assert nikon.stands_in_config(cfg) == []


def test_stands_in_config_survives_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    cfg = tmp_path / "stand.cfg"
    cfg.write_bytes(b"# caf\xe9 \xff\nDevice,Ti2-E__0,NikonTi2,Ti2-E__0\n")

    assert nikon.stands_in_config(cfg) == [nikon.TI2]


def test_stands_in_config_of_an_unreadable_file_is_empty(tmp_path: Path) -> None:
    assert nikon.stands_in_config(tmp_path / "missing.cfg") == []
    assert nikon.stands_in_config(tmp_path) == []


# -- where the driver is, and what the adapter finds (m2 §7.2) ----------------------

TI2_DLL = nikon.TI2.driver.dll
TIE_DLL = nikon.TIE.driver.dll


def test_ti2_driver_beside_the_adapter_is_satisfied(tmp_path: Path) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    beside = touch(mm / TI2_DLL)

    found = nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk)

    assert found == nikon.DriverLocation(beside, beside_adapter=True, satisfied=True)


def test_ti2_driver_in_a_subfolder_is_not_beside_the_adapter(tmp_path: Path) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    nested = touch(mm / "extras" / "ti2" / TI2_DLL)

    found = nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk)

    # The adapter loads it from its own folder only: a subfolder is a finding.
    assert found == nikon.DriverLocation(nested, beside_adapter=False, satisfied=False)


def test_ti2_driver_only_in_the_sdk_folder_is_not_enough(tmp_path: Path) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    mm.mkdir()
    in_sdk = touch(sdk / TI2_DLL)

    found = nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk)

    assert found == nikon.DriverLocation(in_sdk, beside_adapter=False, satisfied=False)


def test_ti2_driver_beside_the_adapter_wins_over_the_sdk_copy(tmp_path: Path) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    beside = touch(mm / TI2_DLL)
    touch(sdk / TI2_DLL)

    assert nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk).path == beside


def test_tie_driver_in_its_sdk_folder_is_enough(tmp_path: Path) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    mm.mkdir()
    in_sdk = touch(sdk / TIE_DLL)

    found = nikon.find_driver(nikon.TIE, mm, sdk_dir=sdk)

    # NikonTI finds NikonTi.dll on the system path, where Nikon's installer puts it.
    assert found == nikon.DriverLocation(in_sdk, beside_adapter=False, satisfied=True)


def test_tie_driver_beside_the_adapter_is_not_where_its_adapter_looks(
    tmp_path: Path,
) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    touch(mm / TIE_DLL)

    found = nikon.find_driver(nikon.TIE, mm, sdk_dir=sdk)

    assert found.path == mm / TIE_DLL
    assert found.satisfied is False


def test_a_driver_nowhere_is_a_location_without_a_path(tmp_path: Path) -> None:
    mm = tmp_path / "mm"
    mm.mkdir()

    for stand in nikon.STANDS:
        found = nikon.find_driver(stand, mm, sdk_dir=tmp_path / "sdk")
        assert found == nikon.DriverLocation(
            None, beside_adapter=False, satisfied=False
        )


def test_a_driver_is_looked_for_in_the_sdk_folder_when_there_is_no_adapter_folder(
    tmp_path: Path,
) -> None:
    sdk = tmp_path / "sdk"
    in_sdk = touch(sdk / TI2_DLL)

    found = nikon.find_driver(nikon.TI2, None, sdk_dir=sdk)

    assert found == nikon.DriverLocation(in_sdk, beside_adapter=False, satisfied=False)


def test_the_sdk_folder_defaults_to_the_one_the_registry_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk = tmp_path / "Ti2-SDK" / "bin"
    touch(sdk / TI2_DLL)
    stand = dataclasses.replace(
        nikon.TI2, driver=dataclasses.replace(nikon.TI2.driver, sdk_dir=str(sdk))
    )

    assert nikon.find_driver(stand, None).path == sdk / TI2_DLL


def test_a_folder_that_cannot_be_walked_is_not_found_there_and_the_search_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    mm.mkdir()
    in_sdk = touch(sdk / TI2_DLL)

    def denied(path: object) -> object:
        raise PermissionError("denied")

    with monkeypatch.context() as patched:
        patched.setattr(nikon.os, "scandir", denied)
        found = nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk)

    assert found.path == in_sdk


def test_the_search_below_the_adapter_folder_is_bounded_in_depth(
    tmp_path: Path,
) -> None:
    # FM-74: a profile that lists a drive as an adapter folder must not make
    # doctor walk the whole drive. Three folders down is found, four is not.
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    three = touch(mm / "a" / "b" / "c" / TI2_DLL)
    touch(mm / "a" / "b" / "c" / "d" / TI2_DLL)

    assert nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk).path == three
    three.unlink()
    assert nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk).path is None


def test_the_search_below_the_adapter_folder_is_bounded_in_folders_visited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    for name in ("a", "b", "c"):
        (mm / name).mkdir(parents=True)
    touch(mm / "c" / TI2_DLL)
    monkeypatch.setattr(nikon, "_MAX_FOLDERS", 3)  # mm, a and b; never c

    assert nikon.find_driver(nikon.TI2, mm, sdk_dir=sdk).path is None


def test_the_adapter_comes_from_the_first_folder_that_holds_it(tmp_path: Path) -> None:
    # FM-76: MMCore takes an adapter from the first search folder that has its
    # file and does not fall back; the driver is then looked for beside that one.
    first, second = tmp_path / "first", tmp_path / "second"
    touch(first / nikon.TI2.adapter_file)
    touch(second / nikon.TI2.adapter_file)
    touch(second / TI2_DLL)

    check = nikon.check_stand(nikon.TI2, [first, second], sdk_dir=tmp_path / "sdk")

    assert check.adapter_path == first / nikon.TI2.adapter_file
    assert check.searched == (first, second)
    assert check.driver.satisfied is False
    assert check.driver.path is None


def test_a_missing_adapter_names_every_folder_searched(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()

    check = nikon.check_stand(nikon.TI2, [first, second], sdk_dir=tmp_path / "sdk")

    assert check.adapter_path is None
    assert check.searched == (first, second)
    rows = dict(nikon.readiness(check))
    assert str(first) in rows["adapter"]
    assert str(second) in rows["adapter"]
    assert rows["adapter"].startswith("not found")


# -- the readiness rows, before opening (m2 §7.4 item 1) -----------------------------


def ready_ti2(tmp_path: Path, *, driver_beside: bool = True) -> nikon.StandCheck:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    touch(mm / nikon.TI2.adapter_file)
    touch(mm / TI2_DLL if driver_beside else sdk / TI2_DLL)
    return nikon.check_stand(nikon.TI2, [mm], sdk_dir=sdk)


def test_readiness_of_a_ready_ti2_names_the_adapter_and_the_driver_beside_it(
    tmp_path: Path,
) -> None:
    rows = nikon.readiness(ready_ti2(tmp_path))

    assert [name for name, _ in rows] == ["adapter", "driver"]
    adapter, driver = (text for _, text in rows)
    assert adapter == str(tmp_path / "mm" / nikon.TI2.adapter_file)
    assert driver == f"{TI2_DLL} beside the adapter"


def test_readiness_says_to_copy_a_misplaced_ti2_driver(tmp_path: Path) -> None:
    rows = dict(nikon.readiness(ready_ti2(tmp_path, driver_beside=False)))

    assert "smc install-driver ti2" in rows["driver"]
    assert str(tmp_path / "sdk") in rows["driver"]
    assert "not beside the adapter" in rows["driver"]


def test_readiness_says_where_to_get_a_driver_that_is_nowhere(tmp_path: Path) -> None:
    mm = tmp_path / "mm"
    touch(mm / nikon.TI2.adapter_file)
    check = nikon.check_stand(nikon.TI2, [mm], sdk_dir=tmp_path / "sdk")

    driver = dict(nikon.readiness(check))["driver"]

    assert driver == f"{TI2_DLL} not found: install {nikon.TI2.driver.installer}"


def test_readiness_of_a_tie_asks_for_its_driver_in_the_sdk_folder(
    tmp_path: Path,
) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    touch(mm / nikon.TIE.adapter_file)

    missing = nikon.check_stand(nikon.TIE, [mm], sdk_dir=sdk)
    assert dict(nikon.readiness(missing))["driver"].startswith(f"{TIE_DLL} not found")
    assert nikon.TIE.driver.installer in dict(nikon.readiness(missing))["driver"]

    touch(sdk / TIE_DLL)
    there = nikon.check_stand(nikon.TIE, [mm], sdk_dir=sdk)
    assert dict(nikon.readiness(there))["driver"] == f"{TIE_DLL} in {sdk}"


def test_readiness_without_an_adapter_does_not_send_a_ti2_driver_to_nowhere(
    tmp_path: Path,
) -> None:
    sdk = tmp_path / "sdk"
    touch(sdk / TI2_DLL)
    check = nikon.check_stand(nikon.TI2, [tmp_path / "mm"], sdk_dir=sdk)

    rows = dict(nikon.readiness(check))

    assert rows["adapter"].startswith("not found")
    # There is no adapter folder to copy it into, so the row cannot say
    # "smc install-driver ti2" as if that would help.
    assert "install-driver" not in rows["driver"]
    assert str(sdk) in rows["driver"]


# -- install-driver: copy, never move, never overwrite (m2 §7.5) ----------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_install_driver_copies_and_leaves_the_source(tmp_path: Path) -> None:
    source = touch(tmp_path / "sdk" / TI2_DLL, b"vendor driver bytes")
    mm = tmp_path / "mm"
    mm.mkdir()

    ok, message = nikon.install_driver(nikon.TI2, mm, source=source)

    assert ok is True
    assert (mm / TI2_DLL).read_bytes() == b"vendor driver bytes"
    # The vendor's own software still needs its copy.
    assert source.read_bytes() == b"vendor driver bytes"
    assert sha256(source)[:12] in message
    assert "copied" in message
    assert sorted(p.name for p in mm.iterdir()) == [TI2_DLL]


def test_install_driver_keeps_an_identical_copy_and_says_so(tmp_path: Path) -> None:
    source = touch(tmp_path / "sdk" / TI2_DLL, b"same bytes")
    mm = tmp_path / "mm"
    touch(mm / TI2_DLL, b"same bytes")

    ok, message = nikon.install_driver(nikon.TI2, mm, source=source)

    assert ok is True
    assert "already there" in message
    assert sha256(source)[:12] in message


def test_install_driver_never_overwrites_a_different_dll(tmp_path: Path) -> None:
    source = touch(tmp_path / "sdk" / TI2_DLL, b"the sdk version")
    mm = tmp_path / "mm"
    there = touch(mm / TI2_DLL, b"another version")

    ok, message = nikon.install_driver(nikon.TI2, mm, source=source)

    assert ok is False
    assert there.read_bytes() == b"another version"
    assert "differs" in message
    assert "never overwrites" in message
    # Both fingerprints, so the operator can tell which one is which.
    assert sha256(source)[:12] in message
    assert sha256(there)[:12] in message
    assert sorted(p.name for p in mm.iterdir()) == [TI2_DLL]


def test_install_driver_refuses_a_missing_source(tmp_path: Path) -> None:
    mm = tmp_path / "mm"
    mm.mkdir()
    source = tmp_path / "sdk" / TI2_DLL

    ok, message = nikon.install_driver(nikon.TI2, mm, source=source)

    assert ok is False
    assert f"{source} does not exist" in message
    assert nikon.TI2.driver.installer in message
    assert "--from" in message
    assert list(mm.iterdir()) == []


def test_install_driver_takes_the_source_from_the_sdk_folder_by_default(
    tmp_path: Path,
) -> None:
    sdk = tmp_path / "Ti2-SDK" / "bin"
    touch(sdk / TI2_DLL, b"from the sdk")
    stand = dataclasses.replace(
        nikon.TI2, driver=dataclasses.replace(nikon.TI2.driver, sdk_dir=str(sdk))
    )
    mm = tmp_path / "mm"
    mm.mkdir()

    ok, _ = nikon.install_driver(stand, mm)

    assert ok is True
    assert (mm / TI2_DLL).read_bytes() == b"from the sdk"


def test_install_driver_refuses_a_stand_whose_driver_is_on_the_system_path(
    tmp_path: Path,
) -> None:
    source = touch(tmp_path / "sdk" / TIE_DLL)
    mm = tmp_path / "mm"
    mm.mkdir()

    ok, message = nikon.install_driver(nikon.TIE, mm, source=source)

    assert ok is False
    assert "system path" in message
    assert nikon.TIE.driver.installer in message
    assert list(mm.iterdir()) == []


def test_install_driver_refuses_a_destination_that_is_not_a_folder(
    tmp_path: Path,
) -> None:
    source = touch(tmp_path / "sdk" / TI2_DLL)
    missing = tmp_path / "no-such-folder"

    ok, message = nikon.install_driver(nikon.TI2, missing, source=source)

    assert ok is False
    assert f"{missing} is not a folder" in message
    assert not missing.exists()


def test_install_driver_leaves_no_partial_file_when_the_copy_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-38: a copy that stops half-way must not leave a truncated DLL where
    # the adapter would load it, nor a partial file that nothing names.
    source = touch(tmp_path / "sdk" / TI2_DLL, b"0123456789")
    mm = tmp_path / "mm"
    mm.mkdir()

    def half_a_copy(src: Path, dst: Path) -> None:
        dst.write_bytes(src.read_bytes()[:5])
        raise PermissionError("denied")

    monkeypatch.setattr(shutil, "copy2", half_a_copy)

    ok, message = nikon.install_driver(nikon.TI2, mm, source=source)

    assert ok is False
    assert message.startswith(f"could not copy to {mm / TI2_DLL}: ")
    assert "denied" in message
    assert list(mm.iterdir()) == []


def test_install_driver_does_not_replace_a_dll_that_appeared_during_the_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Never overwrite means never: another installer may write the file between
    # our look at the destination and the rename onto it.
    source = touch(tmp_path / "sdk" / TI2_DLL, b"ours")
    mm = tmp_path / "mm"
    mm.mkdir()
    real_copy = shutil.copy2

    def copy_then_someone_else_installs(src: Path, dst: Path) -> None:
        real_copy(src, dst)
        (mm / TI2_DLL).write_bytes(b"theirs")

    monkeypatch.setattr(shutil, "copy2", copy_then_someone_else_installs)

    ok, message = nikon.install_driver(nikon.TI2, mm, source=source)

    assert ok is False
    assert (mm / TI2_DLL).read_bytes() == b"theirs"
    assert "differs" in message
    assert sorted(p.name for p in mm.iterdir()) == [TI2_DLL]


# -- what to check when a stand will not open (m2 §7.4 items 2 and 3) -------------------

NOT_FOUND = "The specified module could not be found."


def ti2_check(
    tmp_path: Path, *, adapter: bool = True, driver: str = "beside"
) -> nikon.StandCheck:
    """A Ti2 check on ``tmp_path``: the driver beside the adapter, in the SDK folder, or nowhere."""
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    mm.mkdir(parents=True, exist_ok=True)
    if adapter:
        touch(mm / nikon.TI2.adapter_file)
    if driver == "beside":
        touch(mm / TI2_DLL)
    elif driver == "sdk":
        touch(sdk / TI2_DLL)
    return nikon.check_stand(nikon.TI2, [mm], sdk_dir=sdk)


class RecordingProbe:
    """A ``dll_probe.probe_dll`` that loads nothing and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[Path]]] = []

    def __call__(
        self,
        path: Path,
        *,
        extra_path: object = (),
        timeout_s: float = dll_probe.DLL_PROBE_TIMEOUT_S,
    ) -> DllProbe:
        self.calls.append((path, list(extra_path)))  # type: ignore[call-overload]
        return DllProbe(path, True, None, "loads")


def test_deep_check_probes_the_adapter_then_the_driver_each_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    check = ti2_check(tmp_path)
    recorder = RecordingProbe()
    monkeypatch.setattr(dll_probe, "probe_dll", recorder)
    folders = [tmp_path / "mm", tmp_path / "other"]

    probes = nikon.deep_check(check, extra_path=folders)

    assert [path for path, _ in recorder.calls] == [
        tmp_path / "mm" / nikon.TI2.adapter_file,
        tmp_path / "mm" / TI2_DLL,
    ]
    # Each probe is its own call, so its own child, with the adapter folders on PATH.
    assert [extra for _, extra in recorder.calls] == [folders, folders]
    assert [probe.path for probe in probes] == [path for path, _ in recorder.calls]


def test_deep_check_probes_only_what_was_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = RecordingProbe()
    monkeypatch.setattr(dll_probe, "probe_dll", recorder)

    assert nikon.deep_check(ti2_check(tmp_path, adapter=False, driver="none")) == []
    assert recorder.calls == []

    only_adapter = ti2_check(tmp_path, driver="none")
    nikon.deep_check(only_adapter)
    assert [path.name for path, _ in recorder.calls] == [nikon.TI2.adapter_file]

    recorder.calls.clear()
    only_driver = ti2_check(tmp_path / "second", adapter=False, driver="sdk")
    nikon.deep_check(only_driver)
    assert [path.name for path, _ in recorder.calls] == [TI2_DLL]


def test_deep_check_off_windows_still_returns_its_probes_as_not_probed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dll_probe, "_WINDOWS", False)
    check = ti2_check(tmp_path)

    probes = nikon.deep_check(check)

    assert [probe.loads for probe in probes] == [None, None]
    assert all(probe.detail.startswith("not Windows") for probe in probes)
    lines = [ln for ln in nikon.diagnosis(check, probes) if ln.startswith("probe:")]
    assert len(lines) == 2
    assert all(" not probed: not Windows" in ln for ln in lines)
    assert all("will not load" not in ln for ln in lines)


def test_diagnosis_prints_findings_then_probes_then_likely_causes(
    tmp_path: Path,
) -> None:
    check = ti2_check(tmp_path, driver="sdk")
    adapter = tmp_path / "mm" / nikon.TI2.adapter_file
    probes = [
        DllProbe(adapter, True, None, "loads"),
        DllProbe(
            tmp_path / "sdk" / TI2_DLL, False, 126, describe_winerror(126, NOT_FOUND)
        ),
    ]

    lines = nikon.diagnosis(check, probes)

    kinds = [ln.split(":", 1)[0] for ln in lines]
    first_probe = kinds.index("probe")
    heading = lines.index("in order of likelihood:")
    assert kinds[0] == "found"
    assert all(kind == "found" for kind in kinds[:first_probe])
    assert all(kind == "probe" for kind in kinds[first_probe:heading])
    assert heading == first_probe + 2
    assert lines[heading + 1] == f"  1. {nikon.ONE_CONNECTION}"
    # A misplaced Ti2 driver is a certain finding: where it is, and how to fix it.
    found = [ln for ln in lines if ln.startswith("found:")]
    assert any(TI2_DLL in ln and "smc install-driver ti2" in ln for ln in found)
    assert any(str(tmp_path / "sdk") in ln for ln in found)


def test_diagnosis_says_nothing_found_when_nothing_is_certain(tmp_path: Path) -> None:
    check = ti2_check(tmp_path)  # adapter and driver both where they belong

    lines = nikon.diagnosis(check, [])

    assert not any(ln.startswith("found:") for ln in lines)
    assert lines[0] == "in order of likelihood:"


def test_diagnosis_names_a_driver_that_is_nowhere(tmp_path: Path) -> None:
    check = ti2_check(tmp_path, driver="none")

    found = [ln for ln in nikon.diagnosis(check, []) if ln.startswith("found:")]

    assert len(found) == 1
    assert TI2_DLL in found[0]
    assert nikon.TI2.driver.installer in found[0]


def test_diagnosis_of_a_missing_adapter_names_every_folder_searched(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    check = nikon.check_stand(nikon.TI2, [first, second], sdk_dir=tmp_path / "sdk")

    found = [ln for ln in nikon.diagnosis(check, []) if ln.startswith("found:")]

    assert any(
        nikon.TI2.adapter_file in ln and str(first) in ln and str(second) in ln
        for ln in found
    )


def test_diagnosis_of_a_tie_driver_outside_its_sdk_folder(tmp_path: Path) -> None:
    mm, sdk = tmp_path / "mm", tmp_path / "sdk"
    touch(mm / nikon.TIE.adapter_file)
    check = nikon.check_stand(nikon.TIE, [mm], sdk_dir=sdk)

    found = [ln for ln in nikon.diagnosis(check, []) if ln.startswith("found:")]

    assert len(found) == 1
    assert TIE_DLL in found[0]
    assert nikon.TIE.driver.installer in found[0]
    assert nikon.TIE.driver.sdk_dir in found[0]


def causes_of(lines: list[str]) -> list[str]:
    """The numbered causes, without their numbers."""
    after = lines[lines.index("in order of likelihood:") + 1 :]
    numbered = [ln.strip().split(". ", 1) for ln in after]
    assert [int(n) for n, _ in numbered] == list(range(1, len(numbered) + 1))
    return [text for _, text in numbered]


def test_likely_causes_start_with_another_program_holding_the_stand(
    tmp_path: Path,
) -> None:
    ti2 = causes_of(nikon.diagnosis(ti2_check(tmp_path), []))
    tie = causes_of(
        nikon.diagnosis(
            nikon.check_stand(nikon.TIE, [tmp_path / "mm"], sdk_dir=tmp_path / "sdk"),
            [],
        )
    )

    assert ti2[0] == nikon.ONE_CONNECTION
    assert "another program" in ti2[0]
    assert ti2[1:] == list(nikon.TI2.causes)
    assert "power" in ti2[1]
    assert "driver" in ti2[2]
    assert "Ti2 Control" in ti2[3]

    assert tie[0] == nikon.ONE_CONNECTION
    assert tie[1:] == list(nikon.TIE.causes)
    assert "USB" in tie[1]
    assert "NikonTi.dll" in tie[2]
    assert "TIDiaLamp" in tie[3]


def test_a_winerror_126_probe_reads_as_a_missing_dependency(tmp_path: Path) -> None:
    check = ti2_check(tmp_path)
    adapter = check.adapter_path
    assert adapter is not None
    probe = DllProbe(adapter, False, 126, describe_winerror(126, NOT_FOUND))

    [line] = [ln for ln in nikon.diagnosis(check, [probe]) if ln.startswith("probe:")]

    assert line == f"probe: {adapter.name} will not load: {probe.detail}"
    assert "WinError 126" in line
    assert "a DLL it depends on could not be found" in line


def test_a_probe_without_a_verdict_is_not_called_a_failure(tmp_path: Path) -> None:
    check = ti2_check(tmp_path)
    adapter = check.adapter_path
    assert adapter is not None
    probe = DllProbe(
        adapter, None, None, "did not return within 20 s: its initialisation hung"
    )

    lines = nikon.diagnosis(check, [probe])

    assert f"probe: {adapter.name} not probed: {probe.detail}" in lines
    assert not any("will not load" in ln for ln in lines)


def test_a_probe_that_loads_says_so(tmp_path: Path) -> None:
    check = ti2_check(tmp_path)
    adapter = check.adapter_path
    assert adapter is not None

    lines = nikon.diagnosis(check, [DllProbe(adapter, True, None, "loads")])

    assert f"probe: {adapter.name} loads" in lines


def test_diagnosis_without_an_adapter_does_not_point_a_ti2_driver_beside_nothing(
    tmp_path: Path,
) -> None:
    check = ti2_check(tmp_path, adapter=False, driver="sdk")

    found = [ln for ln in nikon.diagnosis(check, []) if ln.startswith("found:")]

    # Only the adapter is certain; "copy it beside the adapter" has no folder to name.
    assert len(found) == 1
    assert nikon.TI2.adapter_file in found[0]
    assert "install-driver" not in found[0]


# -- the shipped configurations (m2 §7.7) -----------------------------------------

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"


def cfg_records(path: Path, kind: str) -> list[list[str]]:
    """The fields of every ``kind`` line of a ``.cfg``; comments and blank lines skipped."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not line.startswith("#") and fields[0] == kind:
            records.append(fields)
    return records


def test_shipped_nikon_configs_load_the_measured_devices() -> None:
    ti2 = measured("nikon-ti2-devices.json")
    ti2_cfg = PROFILES_DIR / "nikon-ti2.cfg"
    assert [(f[1], f[2], f[3]) for f in cfg_records(ti2_cfg, "Device")] == [
        (d.label, d.library, d.name) for d in ti2
    ]
    # Every device but the hub and the camera hangs below the hub.
    hub = next(d.label for d in ti2 if d.type == "Hub")
    assert [(f[1], f[2]) for f in cfg_records(ti2_cfg, "Parent")] == [
        (d.label, hub) for d in ti2 if d.type not in ("Hub", "Camera")
    ]

    # TIDiaLamp has crashed Micro-Manager with some driver versions: it is left out.
    tie = [d for d in measured("nikon-tie-devices.json") if d.label != "TIDiaLamp"]
    tie_cfg = PROFILES_DIR / "nikon-tie.cfg"
    assert [(f[1], f[2], f[3]) for f in cfg_records(tie_cfg, "Device")] == [
        (d.label, d.library, d.name) for d in tie
    ]
    assert [(f[1], f[2]) for f in cfg_records(tie_cfg, "Parent")] == [
        (d.label, "TIScope") for d in tie if d.type != "Hub"
    ]


def test_shipped_nikon_configs_are_stands_of_their_own_adapter() -> None:
    assert nikon.stands_in_config(PROFILES_DIR / "nikon-ti2.cfg") == [nikon.TI2]
    assert nikon.stands_in_config(PROFILES_DIR / "nikon-tie.cfg") == [nikon.TIE]


def test_the_ti_e_configuration_says_it_is_assumed_and_has_no_core_role_lines() -> None:
    path = PROFILES_DIR / "nikon-tie.cfg"

    assert path.read_text(encoding="utf-8").startswith("# ASSUMED, never loaded")
    # The resolver fills the roles: a core slot written by hand would be a guess.
    assert [f for f in cfg_records(path, "Property") if f[1] == "Core"] == [
        ["Property", "Core", "Initialize", "0"],
        ["Property", "Core", "Initialize", "1"],
    ]


def test_shipped_nikon_configs_hold_no_port_serial_number_or_host() -> None:
    # FM-50: the repository is public.
    for name in ("nikon-ti2.cfg", "nikon-tie.cfg"):
        for line in (PROFILES_DIR / name).read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                continue
            fields = [field.strip().lower() for field in line.split(",")]
            assert "port" not in fields, (name, line)
            assert "serialnumber" not in fields, (name, line)
            assert not any(
                field.startswith("com") and field[3:].isdigit() for field in fields
            )
