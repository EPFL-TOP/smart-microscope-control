"""The Zeiss profile, its ``.cfg`` and the roles it resolves on the fake stand (m4 §9, §10)."""

from __future__ import annotations

from pathlib import Path

from smc.hardware import Microscope
from smc.hardware.capabilities import (
    LightSource,
    ObjectiveTurret,
    Shutter,
    XYStage,
    ZStage,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.testing.fake_mtb import fake_stand_config

PROFILES = Path(__file__).resolve().parents[2] / "profiles"
CFG = PROFILES / "zeiss-observer7.cfg"

#: m4 §7.1: label, class, role (None when the stand assigns it none).
DEVICES = (
    ("XYStage", "MTBXYStage", Role.xy_stage),
    ("Focus", "MTBFocus", Role.focus),
    ("Piezo", "MTBPiezo", None),
    ("Objective", "MTBObjectiveChanger", Role.objective_turret),
    ("Reflector", "MTBReflectorChanger", Role.filter_turret),
    ("Sideport", "MTBSideportChanger", Role.light_path),
    ("Condenser", "MTBCondenserChanger", None),
    ("TLShutter", "MTBTLShutter", None),
    ("Colibri", "MTBColibri", Role.shutter),
)


def test_zeiss_cfg_is_ascii() -> None:
    # UniMMCore reads the file without an encoding: cp1252 on Windows (m4 §1.2).
    CFG.read_bytes().decode("ascii")


def test_zeiss_cfg_declares_the_nine_mtb_devices() -> None:
    lines = CFG.read_text(encoding="utf-8").splitlines()
    declared = [
        tuple(line.removeprefix("#py ").split(",")[1:])
        for line in lines
        if line.startswith("#py pyDevice,")
    ]
    module = "smc.hardware.backends.zeiss_mtb.devices"
    assert declared == [(label, module, cls) for label, cls, _ in DEVICES]
    assert "Device,Camera-1,PVCAM,Camera-1" in lines
    assert "#py Property,Core,Shutter,Colibri" in lines


def test_the_fake_stand_config_keeps_only_the_py_lines_and_initialize(
    tmp_path: Path,
) -> None:
    out = fake_stand_config(CFG, tmp_path)
    kept = out.read_text(encoding="utf-8").splitlines()
    assert "Device,Camera-1,PVCAM,Camera-1" not in kept
    assert "Property,Core,Camera,Camera-1" not in kept
    assert not any(line.startswith("# ") for line in kept)
    assert kept[0] == "Property,Core,Initialize,0"
    assert "Property,Core,Initialize,1" in kept
    py_lines = [
        line
        for line in CFG.read_text(encoding="utf-8").splitlines()
        if line.startswith("#py ")
    ]
    assert [line for line in kept if line.startswith("#py ")] == py_lines
    assert len(kept) == len(py_lines) + 2


def test_zeiss_profile_loads_with_the_stop_refused_and_the_segments() -> None:
    profile = Profile.load(PROFILES / "zeiss-observer7.toml")
    assert profile.config_path() == CFG
    assert set(profile.safety.stop_refused_by) == {"xy_stage", "focus"}
    assert profile.safety.max_segment_um == 5000.0
    assert profile.light_source.intensity_property == "Intensity"


def test_zeiss_profile_resolves_every_role_on_the_fake_stand(
    zeiss_fake_microscope: Microscope,
) -> None:
    stand = zeiss_fake_microscope
    expected = {role: label for label, _, role in DEVICES if role is not None}
    expected[Role.light_source] = "Colibri"
    assert dict(stand.roles.assigned) == expected
    assert stand.core.getXYStageDevice() == "XYStage"
    assert stand.core.getFocusDevice() == "Focus"
    assert stand.core.getShutterDevice() == "Colibri"
    for capability in (
        XYStage,
        ZStage,
        ObjectiveTurret,
        Shutter,
        LightSource,
    ):
        assert stand.has(capability), capability.__name__
