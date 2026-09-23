"""Role resolution on device lists copied from docs/hardware/inventory.md.

Pure: synthetic ``DeviceInfo`` lists and a stub core; no adapter is loaded.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.hardware.roles import (
    DeviceInfo,
    Role,
    RoleMap,
    core_roles,
    devices_from_core,
    resolve,
)


def dev(label: str, type_: str, name: str = "") -> DeviceInfo:
    return DeviceInfo(label=label, type=type_, name=name)


def ti2_xy(stage_label: str = "XYStage") -> list[DeviceInfo]:
    """Nikon Ti2-E: the stage and three TIRF illuminator positioners, all XYStage."""
    tirf = [dev(f"TIRF{i}", "XYStage") for i in (1, 2, 3)]
    return [*tirf, dev(stage_label, "XYStage")]


# Nikon Ti-E: the Z drive, the PFS offset and the TIRF drive, all Stage.
TIE_STAGES = [
    dev("TIRF", "Stage"),
    dev("TIPFSOffset", "Stage"),
    dev("TIZDrive", "Stage"),
]


# -- the stands of the inventory ------------------------------------------------


@pytest.mark.parametrize("stage_label", ["XYStage", "XY"])
def test_ti2_xy_stage_is_not_a_tirf_positioner(stage_label: str) -> None:
    # "XY" matches no hint, so only the exclusion keeps TIRF1 (first by
    # label) from being picked: the dead-stage symptom of FM-13.
    roles = resolve(ti2_xy(stage_label))
    assert roles.get(Role.xy_stage) == stage_label
    assert roles.candidates[Role.xy_stage] == [stage_label]
    assert roles.warnings == []


def test_tie_focus_is_the_z_drive_not_pfs_offset_nor_tirf() -> None:
    roles = resolve(TIE_STAGES)
    assert roles.get(Role.focus) == "TIZDrive"
    assert roles.candidates[Role.focus] == ["TIZDrive"]
    assert roles.get(Role.autofocus_offset) == "TIPFSOffset"
    assert roles.candidates[Role.autofocus_offset] == ["TIPFSOffset"]


def test_config_naming_tirf_as_xy_is_corrected_with_a_warning() -> None:
    roles = resolve(ti2_xy(), core_roles={Role.xy_stage: "TIRF1"})
    assert roles.get(Role.xy_stage) == "XYStage"
    assert roles.sources[Role.xy_stage] == "heuristic"
    [warning] = roles.warnings
    assert warning.startswith("xy_stage: ")
    assert "'TIRF1'" in warning
    assert "'tirf'" in warning
    # It says how to fix it, not only what went wrong.
    assert "Property,Core,XYStage" in warning
    assert "[roles.assign] xy_stage" in warning
    assert "using 'XYStage' (heuristic)" in warning


def test_objective_turret_ignores_filter_and_condenser_turrets() -> None:
    turrets = [
        dev("Dichroic", "State"),
        dev("FilterTurret1", "State"),
        dev("CondenserTurret", "State"),
        dev("Nosepiece", "State"),
    ]
    roles = resolve(turrets)
    assert roles.get(Role.objective_turret) == "Nosepiece"
    assert roles.candidates[Role.objective_turret] == ["Nosepiece"]
    assert roles.get(Role.filter_turret) == "FilterTurret1"
    # Without a nosepiece, no other State device is taken for it.
    assert resolve(turrets[:3]).get(Role.objective_turret) is None


# -- the order of authority ---------------------------------------------------


def test_override_to_unknown_label_warns_and_falls_through() -> None:
    devices = [dev("XY", "XYStage"), dev("Z", "Stage")]
    roles = resolve(
        devices,
        core_roles={Role.xy_stage: "XY"},
        overrides={Role.xy_stage: "xy"},
    )
    assert roles.get(Role.xy_stage) == "XY"
    assert roles.sources[Role.xy_stage] == "core"
    [warning] = roles.warnings
    assert warning.startswith("xy_stage: [roles.assign] names 'xy'")
    assert "candidates: XY" in warning
    assert "using 'XY' (core)" in warning


def test_override_to_unknown_label_with_no_fallback_says_so() -> None:
    roles = resolve([dev("Cam", "Camera")], overrides={Role.focus: "ZDrive"})
    assert roles.get(Role.focus) is None
    [warning] = roles.warnings
    assert "candidates: none" in warning
    assert "leaving it unassigned" in warning


def test_override_is_honoured_even_for_an_excluded_device() -> None:
    # A human said so; the exclusions only guard the configuration and the
    # heuristics.
    roles = resolve(ti2_xy(), overrides={Role.xy_stage: "TIRF2"})
    assert roles.get(Role.xy_stage) == "TIRF2"
    assert roles.sources[Role.xy_stage] == "profile"
    assert roles.candidates[Role.xy_stage] == ["TIRF2", "XYStage"]
    assert roles.warnings == []


def test_override_of_the_wrong_type_is_used_with_a_warning() -> None:
    roles = resolve(
        [dev("Cam", "Camera"), dev("ZDrive", "Stage")],
        overrides={Role.focus: "Cam"},
    )
    assert roles.get(Role.focus) == "Cam"
    assert roles.sources[Role.focus] == "profile"
    [warning] = roles.warnings
    assert "'Cam', a Camera device" in warning
    assert "Stage devices" in warning


def test_core_slot_naming_a_device_that_is_not_loaded_falls_through() -> None:
    roles = resolve([dev("XY", "XYStage")], core_roles={Role.xy_stage: "Gone"})
    assert roles.get(Role.xy_stage) == "XY"
    assert roles.sources[Role.xy_stage] == "heuristic"
    [warning] = roles.warnings
    assert "XYStage slot names 'Gone'" in warning


def test_sources_record_where_each_role_came_from() -> None:
    devices = [
        dev("Cam", "Camera"),
        dev("Cam2", "Camera"),
        dev("XY", "XYStage"),
        dev("ZDrive", "Stage"),
    ]
    roles = resolve(
        devices,
        core_roles={Role.camera: "Cam", Role.xy_stage: "XY"},
        overrides={Role.camera: "Cam2"},
    )
    assert roles.assigned == {
        Role.camera: "Cam2",
        Role.xy_stage: "XY",
        Role.focus: "ZDrive",
    }
    assert roles.sources == {
        Role.camera: "profile",
        Role.xy_stage: "core",
        Role.focus: "heuristic",
    }
    # The assigned device comes first, whatever the heuristics say.
    assert roles.candidates[Role.camera] == ["Cam2", "Cam"]


# -- exclusions from the profile ----------------------------------------------


def test_profile_exclusion_removes_a_candidate() -> None:
    devices = [dev("ZDrive", "Stage"), dev("Piezo", "Stage")]
    assert resolve(devices).candidates[Role.focus] == ["ZDrive", "Piezo"]

    # Fragments are normalised like labels, so "Piezo" matches "piezo".
    excluded = {Role.focus: ["Piezo"]}
    assert resolve(devices, exclusions=excluded).candidates[Role.focus] == ["ZDrive"]

    # It also corrects a configuration that names the piezo as the focus.
    roles = resolve(devices, core_roles={Role.focus: "Piezo"}, exclusions=excluded)
    assert roles.get(Role.focus) == "ZDrive"
    [warning] = roles.warnings
    assert "'Piezo'" in warning
    assert "'piezo'" in warning


def test_a_bare_string_exclusion_is_one_fragment_not_letters() -> None:
    # A str is a Sequence[str], so this type-checks; read letter by letter it
    # would exclude "ZDrive" too (it contains "z", "i" and "e").
    devices = [dev("ZDrive", "Stage"), dev("Piezo", "Stage")]
    roles = resolve(devices, exclusions={Role.focus: "piezo"})
    assert roles.candidates[Role.focus] == ["ZDrive"]


def test_empty_exclusion_is_ignored_with_a_warning() -> None:
    # "" is contained in every label: honoured, it would exclude everything.
    roles = resolve([dev("ZDrive", "Stage")], exclusions={Role.focus: ["", " - "]})
    assert roles.get(Role.focus) == "ZDrive"
    assert len(roles.warnings) == 2
    assert all("[roles.exclude]" in w for w in roles.warnings)


# -- ranking and reporting ----------------------------------------------------


def test_candidates_are_reported_best_first() -> None:
    stages = [dev("Z", "Stage"), dev("Focus", "Stage"), dev("ZDrive", "Stage")]
    roles = resolve(stages)
    assert roles.get(Role.focus) == "ZDrive"
    assert roles.candidates[Role.focus] == ["ZDrive", "Focus", "Z"]
    assert roles.ambiguous() == {Role.focus: ["ZDrive", "Focus", "Z"]}


def test_device_name_breaks_a_tie_between_labels() -> None:
    # Both labels rank on "stage"; the adapter's device names decide.
    stages = [
        dev("Stage A", "XYStage", name="TIXYDrive"),
        dev("Stage B", "XYStage", name="XYStage"),
    ]
    assert resolve(stages).candidates[Role.xy_stage] == ["Stage B", "Stage A"]
    # Nothing to rank on: the label decides, ignoring case.
    cameras = [dev("B", "Camera"), dev("a", "Camera")]
    assert resolve(cameras).candidates[Role.camera] == ["a", "B"]


def test_role_without_candidate_is_absent_not_empty() -> None:
    roles = resolve([dev("Cam", "Camera"), dev("", "Stage")])
    assert roles.assigned == {Role.camera: "Cam"}
    assert roles.get(Role.focus) is None
    assert roles.candidates[Role.focus] == []
    assert roles.missing() == [r for r in Role if r is not Role.camera]
    assert set(roles.candidates) == set(Role)


def test_describe_prints_one_aligned_line_per_role() -> None:
    devices = [dev("Camera", "Camera"), dev("XY", "XYStage"), dev("Stage2", "XYStage")]
    roles = resolve(devices, core_roles={Role.camera: "Camera", Role.xy_stage: "XY"})
    lines = roles.describe()
    assert len(lines) == len(Role)
    assert lines[0] == "camera            Camera                  [core]"
    assert lines[1] == "xy stage          XY      (also: Stage2)  [core]"
    assert lines[2] == "focus             -"
    assert [line.split("  ")[0] for line in lines] == [
        r.value.replace("_", " ") for r in Role
    ]


def test_describe_without_runner_ups_has_no_gap() -> None:
    roles = resolve([dev("Cam", "Camera")])
    assert roles.describe()[0] == "camera            Cam  [heuristic]"


def test_role_map_reads_like_the_design() -> None:
    roles = RoleMap(
        assigned={Role.camera: "Cam"},
        sources={Role.camera: "core"},
        candidates={Role.camera: ["Cam", "Cam2"], Role.focus: []},
        warnings=[],
    )
    assert roles.get(Role.camera) == "Cam"
    assert roles.get(Role.xy_stage) is None
    assert roles.ambiguous() == {Role.camera: ["Cam", "Cam2"]}
    assert Role.camera not in roles.missing()


# -- the core helpers, on a stub core -----------------------------------------


class StubCore:
    """The inventory calls of an MMCore, with raw ``int`` types like pymmcore's."""

    def __init__(
        self, devices: dict[str, int], slots: dict[str, str] | None = None
    ) -> None:
        self.devices = devices
        self.slots = slots or {}

    def getLoadedDevices(self) -> tuple[str, ...]:  # noqa: N802
        return tuple(self.devices)

    def getDeviceType(self, label: str) -> int:  # noqa: N802
        return self.devices[label]

    def getDeviceLibrary(self, label: str) -> str:  # noqa: N802
        return "" if label == "Core" else "StubLib"

    def getDeviceName(self, label: str) -> str:  # noqa: N802
        return f"D{label}"

    def getDeviceDescription(self, label: str) -> str:  # noqa: N802
        return f"stub {label}"

    def getCameraDevice(self) -> str:  # noqa: N802
        return self.slots.get("camera", "")

    def getXYStageDevice(self) -> str:  # noqa: N802
        return self.slots.get("xy", "")

    def getFocusDevice(self) -> str:  # noqa: N802
        return self.slots.get("focus", "")

    def getAutoFocusDevice(self) -> str:  # noqa: N802
        return self.slots.get("autofocus", "")

    def getShutterDevice(self) -> str:  # noqa: N802
        return self.slots.get("shutter", "")


def test_devices_from_core_skips_core_and_strips_the_device_suffix() -> None:
    # MMCore DeviceType values: Core 10, XYStage 6, Camera 2, State 4.
    core = StubCore({"Core": 10, "XY": 6, "Cam": 2, "Turret": 4})
    assert devices_from_core(core) == [
        DeviceInfo("XY", "XYStage", "StubLib", "DXY", "stub XY"),
        DeviceInfo("Cam", "Camera", "StubLib", "DCam", "stub Cam"),
        DeviceInfo("Turret", "State", "StubLib", "DTurret", "stub Turret"),
    ]


def test_device_of_an_unknown_type_is_listed_not_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A device type newer than this pymmcore-plus must not end the inventory.
    core = StubCore({"Pump": 99, "XY": 6})
    with caplog.at_level(logging.WARNING, logger="smc.hardware.roles"):
        devices = devices_from_core(core)
    assert [(d.label, d.type) for d in devices] == [
        ("Pump", "Unknown"),
        ("XY", "XYStage"),
    ]
    assert "Pump" in caplog.text
    assert resolve(devices).get(Role.xy_stage) == "XY"


def test_core_roles_keeps_only_filled_slots() -> None:
    core = StubCore({}, slots={"camera": "Cam", "focus": "Z", "shutter": ""})
    assert core_roles(core) == {Role.camera: "Cam", Role.focus: "Z"}


def test_importing_roles_does_not_load_pymmcore_plus(tmp_path: Path) -> None:
    # resolve() must stay usable, and testable, without the hardware bindings.
    code = "import sys, smc.hardware.roles; print('pymmcore_plus' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        timeout=120,
        check=True,
    )
    assert result.stdout.strip() == "False"


# -- properties ---------------------------------------------------------------

# The §4 table's exclusions, written out again so the property is checked
# against the design rather than against the implementation's own table.
BUILT_IN_EXCLUDE: dict[Role, tuple[str, ...]] = {
    Role.xy_stage: ("tirf",),
    Role.focus: ("pfs", "offset", "tirf"),
    Role.objective_turret: ("filter", "condenser", "reflector"),
    Role.light_source: ("shutter",),
    Role.filter_turret: ("objective", "nose"),
}
CORE_SLOT_ROLES = (Role.camera, Role.xy_stage, Role.focus, Role.autofocus, Role.shutter)
FRAGMENTS = (
    "XY", "Stage", "Drive", "TIRF", "Z", "Focus", "PFS", "Offset", "Piezo",
    "Nose", "Piece", "Objective", "Turret", "Filter", "Condenser", "Reflector",
    "Epi", "Dia", "Shutter", "Lamp", "LED", "Light", "Path", "Port", "Laser",
    "Cam", "1", "2",
)  # fmt: skip
TYPES = (
    "Camera",
    "XYStage",
    "Stage",
    "AutoFocus",
    "State",
    "Shutter",
    "Generic",
    "Hub",
)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


labels = st.builds(
    lambda parts, sep: sep.join(parts),
    st.lists(st.sampled_from(FRAGMENTS), min_size=1, max_size=3),
    st.sampled_from(("", " ", "-", "_")),
)


@st.composite
def scenarios(
    draw: st.DrawFn,
) -> tuple[list[DeviceInfo], dict[Role, str], dict[Role, str], dict[Role, list[str]]]:
    devices = draw(
        st.lists(
            st.builds(DeviceInfo, label=labels, type=st.sampled_from(TYPES)),
            max_size=10,
            unique_by=lambda d: d.label,
        )
    )
    # "Missing" is never loaded: overrides and slots may name it.
    names = st.sampled_from([d.label for d in devices] + ["Missing"])
    slots = draw(st.dictionaries(st.sampled_from(CORE_SLOT_ROLES), names))
    overrides = draw(st.dictionaries(st.sampled_from(list(Role)), names, max_size=3))
    exclusions = draw(
        st.dictionaries(
            st.sampled_from(list(Role)),
            st.lists(st.sampled_from(FRAGMENTS), max_size=2),
            max_size=3,
        )
    )
    return devices, slots, overrides, exclusions


def banned(role: Role, exclusions: dict[Role, list[str]]) -> tuple[str, ...]:
    return BUILT_IN_EXCLUDE.get(role, ()) + tuple(
        norm(f) for f in exclusions.get(role, ())
    )


@given(scenarios())
def test_resolve_never_assigns_an_excluded_device(
    scenario: tuple[
        list[DeviceInfo], dict[Role, str], dict[Role, str], dict[Role, list[str]]
    ],
) -> None:
    devices, slots, overrides, exclusions = scenario
    roles = resolve(
        devices, core_roles=slots, overrides=overrides, exclusions=exclusions
    )
    loaded = {d.label for d in devices}
    for role, label in roles.assigned.items():
        assert label in loaded
        assert roles.candidates[role][0] == label
        if roles.sources[role] == "profile":
            # Only a human can assign an excluded device.
            assert overrides[role] == label
            continue
        hit = [b for b in banned(role, exclusions) if b in norm(label)]
        assert not hit, f"{role.value} = {label!r} although it contains {hit}"
    assert set(roles.assigned) | set(roles.missing()) == set(Role)
    assert not set(roles.assigned) & set(roles.missing())


@given(scenarios())
def test_authority_is_profile_then_core_then_heuristic(
    scenario: tuple[
        list[DeviceInfo], dict[Role, str], dict[Role, str], dict[Role, list[str]]
    ],
) -> None:
    devices, slots, overrides, exclusions = scenario
    roles = resolve(
        devices, core_roles=slots, overrides=overrides, exclusions=exclusions
    )
    loaded = {d.label for d in devices}
    for role in Role:
        wanted, slot = overrides.get(role), slots.get(role)
        if wanted in loaded:
            expected = (wanted, "profile")
        elif slot in loaded and not any(
            b in norm(slot) for b in banned(role, exclusions)
        ):
            expected = (slot, "core")
        elif roles.candidates[role]:
            expected = (roles.candidates[role][0], "heuristic")
        else:
            assert role not in roles.assigned
            continue
        assert (roles.get(role), roles.sources.get(role)) == expected
