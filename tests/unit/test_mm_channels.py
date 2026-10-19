"""``Channels`` on the Ti2-E's layout (m2 §8.5, #14).

``ti2_core()`` is the Ti2-E as session 0 read it (m2 §8.8): the devices of
``tests/data/nikon-ti2-devices.json``, the labels and the one ``Channel``
preset ``BF`` of ``profiles/nikon-ti2.cfg``, and a ``DiaLamp`` with a free
``Intensity``. ``test_mm_objective_turret.py`` and ``test_mm_light_source.py``
import it from here (``tests/`` has no packages). Everything goes through
``Microscope.from_core``, as a tool's calls would; a capture writes a copy of
the configuration in ``tmp_path``, never the repository's.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from pymmcore_plus import DeviceType

import smc.hardware.backends.mm as mm
from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, ChannelPreset, Channels
from smc.hardware.errors import (
    CapabilityMissingError,
    DeviceTimeoutError,
    HardwareError,
    MotionInProgressError,
    SafetyRefusedError,
)
from smc.hardware.profile import (
    ChannelsSection,
    MicroManagerSection,
    MicroscopeSection,
    Profile,
    SafetySection,
)
from smc.testing import FakeCore, FakeDevice, FakeProperty

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

REPO = Path(__file__).resolve().parents[2]
TI2_CFG = REPO / "profiles" / "nikon-ti2.cfg"
JOIN_S = 5.0

#: The configuration's labels of the Ti2-E's State devices (session 0).
TI2_LABELS = {
    "Nosepiece": (
        "1-Plan Fluor 4x",
        "2-Plan Fluor 10x",
        "3-Plan Apo LmbdD0.25 40x",
        "4-",
        "5-",
        "6-",
    ),
    "CondenserTurret": ("1-OPEN", "2-Ph1", "3-", "4-", "5-", "6-Shutter", "7-N2"),
    "FilterTurret1": ("1-", "2-", "3-", "4-", "5-", "6-"),
    "FilterTurret2": (
        "1-ANALY",
        "2-MXR00700 -96372 - GF",
        "3-96372 - GF96373 - YF",
        "4-96373 - YF96375 - mC",
        "5-96375 - mCMXR00712 -",
        "6-MXR00712 -",
    ),
    "LightPath": ("1-EYE", "2-R100", "3-AUX", "4-L100"),
}


#: The lab's ``BF``, as session 0 read it from ``profiles/nikon-ti2.cfg``. A
#: literal: a capture at the stand edits the committed file (profiles/README.md).
LAB_BF = (
    ("CondenserTurret", "Label", "7-N2"),
    ("FilterTurret1", "Label", "1-"),
    ("Turret1Shutter", "State", "0"),
    ("FilterTurret2", "Label", "1-ANALY"),
    ("Turret2Shutter", "State", "0"),
    ("LightPath", "Label", "4-L100"),
    ("DiaLamp", "State", "0"),
    ("DiaLamp", "Intensity", "210"),
    ("Camera-1", "Exposure", "10"),
)


def _bf_from_the_cfg() -> list[tuple[str, str, str]]:
    """``BF``, line for line from ``profiles/nikon-ti2.cfg``."""
    settings = []
    for line in TI2_CFG.read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if fields[:3] == ["ConfigGroup", "Channel", "BF"]:
            device, prop, value = fields[3:]
            settings.append((device, prop, value))
    return settings


def ti2_core() -> FakeCore:
    """The Ti2-E's layout: session 0's devices, labels, slots and ``BF`` (m2 §8.8).

    ``Nosepiece`` starts at ``2-Plan Fluor 10x``; every other State device at
    its first position, and ``DiaLamp``'s ``Intensity`` at ``"501"``, as read.
    The two turret shutters and ``DiaLamp`` carry the ``State`` property that
    ``BF`` sets; in the fake it is a plain property, not the open state.
    """
    data = json.loads(
        (REPO / "tests" / "data" / "nikon-ti2-devices.json").read_text(encoding="utf-8")
    )
    devices = []
    for entry in data["devices"]:
        label = entry["label"]
        properties: dict[str, FakeProperty] = {}
        if label == "DiaLamp":
            properties = {"Intensity": FakeProperty("501"), "State": FakeProperty("0")}
        elif label in ("Turret1Shutter", "Turret2Shutter"):
            properties = {"State": FakeProperty("0")}
        labels = TI2_LABELS.get(label, ())
        if labels:
            position = 1 if label == "Nosepiece" else 0
            properties = {
                "State": FakeProperty(str(position)),
                "Label": FakeProperty(labels[position], allowed=labels),
            }
        devices.append(
            FakeDevice(
                label,
                DeviceType[entry["type"]],
                library=entry["library"],
                name=label,
                properties=properties,
                state_labels=labels,
            )
        )
    core = FakeCore(
        devices,
        camera="Camera-1",
        xy_stage="XYStage",
        focus="ZDrive",
        autofocus="PFS",
        shutter="DiaLamp",
    )
    core.config_groups = {"Channel": {"BF": list(LAB_BF)}}
    return core


def ti2_profile(
    *,
    config: Path | None = None,
    channels: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
    device_timeout_ms: int = 60_000,
) -> Profile:
    """A profile for the layout; ``config`` is the ``.cfg`` a capture writes."""
    return Profile(
        microscope=MicroscopeSection(name="ti2e-fake"),
        micromanager=MicroManagerSection(
            config="" if config is None else str(config),
            device_timeout_ms=device_timeout_ms,
        ),
        safety=SafetySection.model_validate({"max_z_jog_um": 50.0, **(safety or {})}),
        channels=ChannelsSection.model_validate(channels or {}),
    )


@contextmanager
def ti2(
    profile: Profile | None = None,
    *,
    core: FakeCore | None = None,
    dry_run: bool = False,
) -> Iterator[tuple[FakeCore, Microscope]]:
    """A facade over ``core`` (default: ``ti2_core()``), the log cleared once built."""
    fake = core or ti2_core()
    microscope = Microscope.from_core(
        cast("CMMCorePlus", fake), profile or ti2_profile(), dry_run=dry_run
    )
    fake.log.clear()
    try:
        yield fake, microscope
    finally:
        microscope.close()


class Thread:
    """Run ``target`` in a thread and keep what it returned or raised (FM-44)."""

    def __init__(self, target: Callable[[], object]) -> None:
        self.result: object = None
        self.error: BaseException | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self._target()
        except BaseException as exc:
            self.error = exc

    def join(self) -> None:
        self._thread.join(timeout=JOIN_S)
        assert not self._thread.is_alive(), "thread did not finish: deadlock?"


def wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + JOIN_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


@contextmanager
def lock_held_by_a_snap(
    core: FakeCore, microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> Iterator[threading.Event]:
    """A snap holds the microscope lock until the block ends.

    Yields the event set once another action waits for the lock behind it.
    """
    camera, executor = microscope.require(Camera), microscope._executor
    queued, held, release = threading.Event(), threading.Event(), threading.Event()
    contention, snap = executor._contention, core.snapImage

    def noted(*args: Any, **kwargs: Any) -> None:
        queued.set()
        contention(*args, **kwargs)

    def hung_snap() -> None:
        held.set()
        release.wait(JOIN_S)
        snap()

    monkeypatch.setattr(executor, "_contention", noted)
    monkeypatch.setattr(core, "snapImage", hung_snap)
    snapper = Thread(camera.snap)
    assert held.wait(JOIN_S)
    try:
        yield queued
    finally:
        release.set()
        snapper.join()


def copy_of_the_cfg(tmp_path: Path, *, crlf: bool = False) -> Path:
    """The Ti2-E's configuration, copied to ``tmp_path`` as bytes (CRLF if asked)."""
    # LF first: a Windows checkout (core.autocrlf) gives the committed file CRLF.
    raw = TI2_CFG.read_bytes().replace(b"\r\n", b"\n")
    if crlf:
        raw = raw.replace(b"\n", b"\r\n")
    path = tmp_path / "nikon-ti2.cfg"
    path.write_bytes(raw)
    return path


def _not_written(path: Path, before: bytes) -> None:
    """``path`` holds ``before``, and no temporary file is left beside it (FM-38)."""
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


# Captured on the layout as it starts: State labels in getLoadedDevices()
# order, the lamp's knob, the core's shutter.
CAPTURED = (
    ("CondenserTurret", "Label", "1-OPEN"),
    ("FilterTurret1", "Label", "1-"),
    ("FilterTurret2", "Label", "1-ANALY"),
    ("LightPath", "Label", "1-EYE"),
    ("DiaLamp", "Intensity", "501"),
    ("Core", "Shutter", "DiaLamp"),
)


# --- select ---------------------------------------------------------------------


def test_the_layouts_bf_is_the_labs() -> None:
    # The layout's preset is the committed file's, line for line (m2 §8.1).
    # A capture at the stand that the owner commits fails this test alone.
    assert tuple(_bf_from_the_cfg()) == LAB_BF


def test_bf_reads_current_after_select_although_mmcore_formats_the_numbers() -> None:
    with ti2() as (core, microscope):
        channels = microscope.require(Channels)
        assert (channels.group(), channels.names()) == ("Channel", ["BF"])
        assert channels.current() is None  # the condenser and the lamp differ
        assert channels.select("BF") == "BF"
        assert core.log == ["setConfig('Channel', 'BF')"]
        # MMCore's string match would read no current preset here.
        assert core.getProperty("Camera-1", "Exposure") == "10.0000"
        assert channels.current() == "BF"
        assert microscope.state().channel == "BF"


def test_a_select_the_stand_does_not_follow_returns_what_it_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    core = ti2_core()
    # A set on a read-only property is ignored, as MMCore does.
    core._devices["DiaLamp"].properties["Intensity"].read_only = True
    with ti2(core=core) as (_, microscope):
        with caplog.at_level(logging.WARNING):
            assert microscope.require(Channels).select("BF") is None
        assert caplog.messages == [
            "channels: Channel BF selected, but the stand reads otherwise: "
            "DiaLamp,Intensity reads '501' (preset: '210'); it matches no preset"
        ]


def test_an_unknown_channel_is_refused_before_anything_is_sent() -> None:
    with ti2() as (core, microscope):
        with pytest.raises(
            ValueError,
            match=r"^channels: group 'Channel' has no preset 'GFP'; its presets "
            r"are: 'BF'$",
        ):
            microscope.require(Channels).select("GFP")
        assert core.log == []


def test_a_preset_that_turns_the_objective_is_refused_and_cannot_be_forced() -> None:
    core = ti2_core()
    core.defineConfig(
        "Channel", "Turn", "Nosepiece", "Label", "3-Plan Apo LmbdD0.25 40x"
    )
    core.defineConfig("Channel", "Turn", "ZDrive", "Position", "0")
    with ti2(core=core) as (_, microscope):
        with pytest.raises(SafetyRefusedError) as refused:
            microscope.require(Channels).select("Turn")
        assert refused.value.how_to_force == ""
        assert refused.value.reason == (
            "channel 'Turn' of group 'Channel' is refused: Nosepiece,Label: "
            "'Nosepiece' is the objective turret (use ObjectiveTurret.select with "
            "confirm=True); ZDrive,Position: 'ZDrive' is a Stage device (use "
            "ZStage). Edit the preset in the configuration, or move these through "
            "their capability (ZStage, XYStage, Autofocus, ObjectiveTurret)"
        )
        assert core.log == []
        assert core.getStateLabel("Nosepiece") == "2-Plan Fluor 10x"


def test_a_preset_redefined_while_select_waits_for_the_lock_is_checked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-43: the preset is read under the lock; a check made when select was
    # called would pass BF and send the redefined one.
    with ti2() as (core, microscope):
        channels = microscope.require(Channels)
        with lock_held_by_a_snap(core, microscope, monkeypatch) as queued:
            selector = Thread(lambda: channels.select("BF"))
            assert queued.wait(JOIN_S)
            core.defineConfig("Channel", "BF", "Nosepiece", "Label", "4-")
        selector.join()
        assert isinstance(selector.error, SafetyRefusedError)
        assert "'Nosepiece' is the objective turret" in selector.error.reason
        assert "setConfig('Channel', 'BF')" not in core.log
        assert core.getStateLabel("Nosepiece") == "2-Plan Fluor 10x"


def test_a_filter_turret_still_busy_at_the_deadline_stays_registered() -> None:
    with ti2(ti2_profile(device_timeout_ms=100)) as (core, microscope):
        core.busy_devices.add("FilterTurret2")
        channels = microscope.require(Channels)
        with pytest.raises(DeviceTimeoutError) as timed_out:
            channels.select("BF")
        assert str(timed_out.value).startswith(
            "channels Channel BF is still busy after 0.1 s; "
        )
        # No stop reaches a State device, so nothing was sent but the change.
        assert core.log == ["setConfig('Channel', 'BF')"]
        assert microscope.state().moving == ("channels Channel BF",)
        with pytest.raises(MotionInProgressError):
            channels.select("BF")
        microscope.resume()
        assert microscope.state().moving == ()
        core.busy_devices.clear()


def test_the_profiles_group_decides_between_two_channel_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def two_groups() -> FakeCore:
        core = ti2_core()
        core.config_groups["Channel-Epi"] = {"GFP": [("FilterTurret2", "Label", "2-")]}
        return core

    with ti2(core=two_groups()) as (_, microscope), caplog.at_level(logging.WARNING):
        assert not microscope.has(Channels)
        microscope.state()
        # Chosen once: the warning is not repeated at every state().
        assert [m for m in caplog.messages if m.startswith("channels:")] == [
            "channels: several config groups are named for channels ('Channel', "
            "'Channel-Epi') and the core's channel group ('') is none of them, so "
            "none is chosen; name one in the profile's [channels] group"
        ]
    profile = ti2_profile(channels={"group": "Channel-Epi"})
    with ti2(profile, core=two_groups()) as (_, microscope):
        channels = microscope.require(Channels)
        assert (channels.group(), channels.names()) == ("Channel-Epi", ["GFP"])


def test_a_capture_into_the_profiles_new_group_creates_it(tmp_path: Path) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_text(encoding="utf-8")
    profile = ti2_profile(config=cfg, channels={"group": "Epi"})
    with ti2(profile) as (core, microscope):
        channels = microscope.require(Channels)
        assert (channels.group(), channels.names(), channels.current()) == (
            "Epi",
            [],
            None,
        )
        preset = channels.capture("GFP")
        assert core.getAvailableConfigs("Epi") == ("GFP",)
        assert channels.select("GFP") == "GFP"
    lines = "".join(f"ConfigGroup,Epi,GFP,{d},{p},{v}\n" for d, p, v in preset.settings)
    assert cfg.read_text(encoding="utf-8") == before + lines


def test_a_preset_naming_a_device_that_did_not_load_matches_nothing() -> None:
    # FM-19: one such preset (defineConfig takes it without a word) must not
    # fail every current() and every state(), nor a select of another preset.
    core = ti2_core()
    core.defineConfig("Channel", "AAA-ghost", "Ghost", "Prop", "1")
    with ti2(core=core) as (_, microscope):
        channels = microscope.require(Channels)
        assert channels.current() is None
        assert channels.select("BF") == "BF"
        assert channels.current() == "BF"
        state = microscope.state()
        assert (state.channel, state.errors) == ("BF", [])


def test_without_a_channel_group_require_names_the_profile_key() -> None:
    core = ti2_core()
    core.config_groups = {"Camera": {}, "System": {}}
    with ti2(core=core) as (_, microscope):
        assert not microscope.has(Channels)
        assert Channels not in microscope.available()
        with pytest.raises(CapabilityMissingError) as missing:
            microscope.require(Channels)
        assert str(missing.value) == (
            "Channels is not available: no config group is named for channels; name "
            "one in the profile's [channels] group (available: Camera, System)."
        )
        assert microscope.state().channel is None


# --- capture --------------------------------------------------------------------


def test_capture_records_labels_the_lamp_intensity_and_the_core_shutter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with ti2() as (core, microscope), caplog.at_level(logging.INFO):
        preset = microscope.require(Channels).capture("GFP")
        assert preset == ChannelPreset("Channel", "GFP", CAPTURED, saved_to=None)
        assert core.getConfigData("Channel", "GFP") == CAPTURED
    assert (
        "channels: GFP not saved: the profile loads the demo configuration, which "
        "has no file"
    ) in caplog.messages


def test_capture_never_records_the_objective_turret_a_shutter_state_or_the_exposure() -> (
    None
):
    with ti2() as (_, microscope):
        preset = microscope.require(Channels).capture("GFP")
    devices = {device for device, _, _ in preset.settings}
    assert devices.isdisjoint(
        {
            "Nosepiece",  # the objective turret
            "Turret1Shutter",
            "Turret2Shutter",
            "Camera-1",  # the exposure
            "ZDrive",
            "XYStage",
            "PFS",
            "PFSOffset",
            "IntermediateMagnification",
            "Ti2-E__0",
        }
    )
    assert ("DiaLamp", "State", "0") not in preset.settings


def test_capture_records_only_state_devices_with_a_writable_label() -> None:
    # The type filter, not the fake's lack of a Label, keeps a magnifier, a
    # hub, a stage and a shutter out of the channel (m2 §8.5).
    core = ti2_core()
    for label in ("IntermediateMagnification", "Ti2-E__0", "ZDrive", "Turret1Shutter"):
        core._devices[label].properties["Label"] = FakeProperty("1.0x")
    for label in ("Wheel-RO", "Wheel-NoLabel"):
        core._devices[label] = FakeDevice(label, DeviceType.State, state_labels=("A",))
        core._devices[label].properties["Label"] = FakeProperty("A")
    core._devices["Wheel-RO"].properties["Label"].read_only = True
    del core._devices["Wheel-NoLabel"].properties["Label"]
    with ti2(core=core) as (_, microscope):
        assert microscope.require(Channels).capture("GFP").settings == CAPTURED


def test_a_capture_with_nothing_to_record_is_refused_and_keeps_the_preset(
    tmp_path: Path,
) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_bytes()
    every_state = [*TI2_LABELS, "DiaLamp"]
    core = ti2_core()
    core.setShutterDevice("")
    profile = ti2_profile(config=cfg, channels={"exclude": every_state})
    with ti2(profile, core=core) as (_, microscope):
        message = (
            "channels: nothing to capture for 'BF': no State device, light-source "
            "knob or shutter to record"
        )
        with pytest.raises(HardwareError, match=f"^{re.escape(message)}$"):
            microscope.require(Channels).capture("BF")
        assert core.log == []
        assert core.getConfigData("Channel", "BF") == LAB_BF
    _not_written(cfg, before)


def test_a_capture_records_the_stand_as_it_is_once_it_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-43: read when called, the capture would record a filter that a move
    # still holding the lock was about to change.
    with ti2() as (core, microscope):
        channels = microscope.require(Channels)
        with lock_held_by_a_snap(core, microscope, monkeypatch) as queued:
            capturer = Thread(lambda: channels.capture("GFP"))
            assert queued.wait(JOIN_S)
            core.setStateLabel("FilterTurret2", "2-MXR00700 -96372 - GF")
        capturer.join()
        assert capturer.error is None
        assert isinstance(capturer.result, ChannelPreset)
        assert (
            "FilterTurret2",
            "Label",
            "2-MXR00700 -96372 - GF",
        ) in capturer.result.settings


def test_a_preset_that_switches_auto_shutter_off_reads_current() -> None:
    core = ti2_core()
    core.config_groups["Channel"]["Manual"] = [
        ("DiaLamp", "Intensity", "501"),
        ("Core", "AutoShutter", "0"),
    ]
    with ti2(core=core) as (_, microscope):
        channels = microscope.require(Channels)
        assert channels.current() is None  # auto-shutter starts on
        assert channels.select("Manual") == "Manual"
        assert core.getAutoShutter() is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_captured_cfg_keeps_the_mode_a_plain_write_gives(tmp_path: Path) -> None:
    # FM-38: mkstemp's 0600 would survive the rename and make the file private.
    cfg = copy_of_the_cfg(tmp_path)
    old = os.umask(0o022)
    try:
        with ti2(ti2_profile(config=cfg)) as (_, microscope):
            microscope.require(Channels).capture("BF")
    finally:
        os.umask(old)
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o644


def test_the_profile_exclude_list_is_never_captured() -> None:
    profile = ti2_profile(channels={"exclude": ["CondenserTurret", "DiaLamp"]})
    with ti2(profile) as (_, microscope):
        preset = microscope.require(Channels).capture("GFP")
    assert preset.settings == (
        ("FilterTurret1", "Label", "1-"),
        ("FilterTurret2", "Label", "1-ANALY"),
        ("LightPath", "Label", "1-EYE"),
        ("Core", "Shutter", "DiaLamp"),
    )


def test_recapturing_a_preset_replaces_it_in_the_core_and_the_cfg(
    tmp_path: Path,
) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_text(encoding="utf-8")
    core = ti2_core()
    # A stale setting defineConfig would keep if the preset were not deleted.
    core.defineConfig("Channel", "BF", "Ghost", "Prop", "1")
    with ti2(ti2_profile(config=cfg), core=core) as (_, microscope):
        preset = microscope.require(Channels).capture("BF")
        assert preset == ChannelPreset("Channel", "BF", CAPTURED, saved_to=cfg)
        assert core.getConfigData("Channel", "BF") == CAPTURED
        assert core.log[0] == "deleteConfig('Channel', 'BF')"
    lines = [f"ConfigGroup,Channel,BF,{d},{p},{v}" for d, p, v in CAPTURED]
    old = [
        line
        for line in before.splitlines()
        if line.startswith("ConfigGroup,Channel,BF,")
    ]
    assert cfg.read_text(encoding="utf-8") == before.replace(
        "\n".join(old) + "\n", "\n".join(lines) + "\n"
    )


def test_a_recapture_says_which_settings_it_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The lab's BF closes the epi shutters and sets an exposure; a capture
    # records neither, so replacing BF loses them, and the WARNING says so.
    with ti2() as (_, microscope):
        with caplog.at_level(logging.WARNING):
            microscope.require(Channels).capture("BF")
        assert caplog.messages == [
            "channels: Channel BF replaced; dropped: CondenserTurret,Label,7-N2; "
            "Turret1Shutter,State,0; Turret2Shutter,State,0; LightPath,Label,4-L100; "
            "DiaLamp,State,0; DiaLamp,Intensity,210; Camera-1,Exposure,10; added: "
            "CondenserTurret,Label,1-OPEN; LightPath,Label,1-EYE; "
            "DiaLamp,Intensity,501; Core,Shutter,DiaLamp"
        ]
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            microscope.require(Channels).capture("BF")  # the same again
        assert caplog.messages == []


def test_a_capture_keeps_a_windows_cfg_crlf(tmp_path: Path) -> None:
    cfg = copy_of_the_cfg(tmp_path, crlf=True)
    with ti2(ti2_profile(config=cfg)) as (_, microscope):
        microscope.require(Channels).capture("BF")
    raw = cfg.read_bytes()
    assert b"ConfigGroup,Channel,BF,DiaLamp,Intensity,501\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_a_capture_writes_the_file_before_the_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    with ti2(ti2_profile(config=cfg)) as (core, microscope):
        channels = microscope.require(Channels)
        before = cfg.read_bytes()

        def refused(*args: object) -> None:
            raise HardwareError("disk full")

        monkeypatch.setattr(mm, "_write_atomically", refused)
        with pytest.raises(HardwareError, match=r"^disk full$"):
            channels.capture("BF")
        assert core.log == []  # the core is unchanged
        assert cfg.read_bytes() == before
        monkeypatch.undo()

        def no_define(*args: object) -> None:
            raise RuntimeError("Ghost: no such device")

        core.defineConfig = no_define  # type: ignore[method-assign]
        with pytest.raises(HardwareError) as failed:
            channels.capture("BF")
        del core.defineConfig
        assert str(failed.value).startswith(
            f"channels: 'BF' was written to {cfg}, but the running core did not take "
            f"it (channels: defineConfig('Channel', 'BF', 'CondenserTurret', 'Label', "
            f"'1-OPEN') failed (Ghost: no such device)"
        )
        assert str(failed.value).endswith(
            "and may hold part of it; the file holds the preset, and the next open "
            "loads it"
        )
        assert b"ConfigGroup,Channel,BF,DiaLamp,Intensity,501\n" in cfg.read_bytes()


def test_an_interrupted_cfg_write_leaves_the_old_file_and_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_bytes()

    def interrupted(fd: int) -> None:
        raise KeyboardInterrupt

    with ti2(ti2_profile(config=cfg)) as (core, microscope):
        monkeypatch.setattr(mm.os, "fsync", interrupted)
        with pytest.raises(KeyboardInterrupt):
            microscope.require(Channels).capture("BF")
        monkeypatch.undo()
        assert core.log == []
    _not_written(cfg, before)


def test_a_failed_rename_keeps_and_names_the_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_bytes()

    def held_open(src: object, dst: object) -> None:
        raise PermissionError(13, "Permission denied")

    with ti2(ti2_profile(config=cfg)) as (core, microscope):
        monkeypatch.setattr(mm.os, "replace", held_open)
        with pytest.raises(HardwareError) as failed:
            microscope.require(Channels).capture("BF")
        monkeypatch.undo()
        assert core.log == []
    assert cfg.read_bytes() == before
    kept = [p for p in tmp_path.iterdir() if p != cfg]
    assert len(kept) == 1
    assert str(failed.value) == (
        f"channels: {cfg} could not be replaced (Permission denied); it is as it "
        f"was, and the new text is kept as {kept[0]}. Close any program that holds "
        f"nikon-ti2.cfg open, then capture again"
    )
    assert b"ConfigGroup,Channel,BF,DiaLamp,Intensity,501\n" in kept[0].read_bytes()


def test_a_cfg_that_is_not_utf8_is_refused_without_writing(tmp_path: Path) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    cfg.write_bytes(b"# Objectif 10\xb5m\n" + cfg.read_bytes())  # cp1252's µ
    before = cfg.read_bytes()
    with ti2(ti2_profile(config=cfg)) as (core, microscope):
        with pytest.raises(HardwareError) as refused:
            microscope.require(Channels).capture("BF")
        assert core.log == []
    assert str(refused.value) == (
        f"channels: {cfg} is not UTF-8 text (line 1: invalid start byte); nothing "
        f"was written. Save it as UTF-8, then capture again"
    )
    _not_written(cfg, before)


def test_capture_in_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_bytes()
    with ti2(ti2_profile(config=cfg), dry_run=True) as (core, microscope):
        preset = microscope.require(Channels).capture("BF")
        assert preset == ChannelPreset("Channel", "BF", CAPTURED, saved_to=None)
        assert core.log == []
        assert core.getConfigData("Channel", "BF") == LAB_BF
    _not_written(cfg, before)


@pytest.mark.parametrize("name", ["", "a,b", " BF"])
def test_a_capture_name_the_cfg_cannot_hold_is_refused_before_anything_is_read(
    name: str,
) -> None:
    core = ti2_core()
    with ti2(core=core) as (_, microscope):
        channels = microscope.require(Channels)
        core.failing["CondenserTurret"] = RuntimeError("must not be read")
        with pytest.raises(ValueError, match=re.escape(f"{name!r} is refused")):
            channels.capture(name)
        core.failing.clear()
        assert core.log == []


def test_a_value_the_cfg_cannot_hold_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    cfg = copy_of_the_cfg(tmp_path)
    before = cfg.read_bytes()
    core = ti2_core()
    # A label as a vendor's adapter may name a position (FM-81).
    condenser = core._devices["CondenserTurret"]
    condenser.state_labels = ("1-Ph1,Ph2", *condenser.state_labels[1:])
    condenser.properties["Label"].value = "1-Ph1,Ph2"
    with ti2(ti2_profile(config=cfg), core=core) as (_, microscope):
        with pytest.raises(HardwareError) as refused:
            microscope.require(Channels).capture("BF")
        assert core.log == []
    assert str(refused.value).startswith(
        "CondenserTurret,Label reads '1-Ph1,Ph2', which a configuration file cannot "
        "hold"
    )
    _not_written(cfg, before)
