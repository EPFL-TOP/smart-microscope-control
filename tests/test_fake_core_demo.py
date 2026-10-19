"""``FakeCore.demo_like()`` agrees with the demo it copies (design §8, #9).

Where the fake and the demo disagree, the fake is wrong: a test that passes
on the fake must describe the same stand the demo is. Property *values* may
differ (the demo's shutter starts open on some opens, FM-47); names, flags,
the inventory, the slots and the State labels may not.
"""

from __future__ import annotations

from typing import Any

import pytest

from smc.hardware.roles import core_roles, devices_from_core
from smc.testing import FakeCore

pytestmark = pytest.mark.demo


def test_demo_like_inventory_matches_the_demo(demo_core: Any) -> None:
    fake = FakeCore.demo_like()
    assert devices_from_core(fake) == devices_from_core(demo_core)
    assert core_roles(fake) == core_roles(demo_core)
    assert fake.getLoadedDevices() == tuple(demo_core.getLoadedDevices())


def test_demo_like_properties_exist_on_the_demo_with_the_same_read_only_flag(
    demo_core: Any,
) -> None:
    fake = FakeCore.demo_like()
    checked = 0
    for label in fake.getLoadedDevices():
        demo_names = set(demo_core.getDevicePropertyNames(label))
        for name in fake.getDevicePropertyNames(label):
            assert name in demo_names, f"{label}.{name} is not a demo property"
            assert fake.isPropertyReadOnly(label, name) == bool(
                demo_core.isPropertyReadOnly(label, name)
            ), f"{label}.{name}: read-only flags differ"
            checked += 1
    # The loop must have compared something: an empty fake would pass it.
    assert checked >= 30


def test_demo_like_state_labels_match_the_demo(demo_core: Any) -> None:
    fake = FakeCore.demo_like()
    states = [d for d in devices_from_core(fake) if d.type == "State"]
    assert [d.label for d in states] == [
        "Dichroic",
        "Emission",
        "Excitation",
        "Objective",
        "Path",
        "LED",
    ]
    for device in states:
        label = device.label
        assert fake.getStateLabels(label) == tuple(demo_core.getStateLabels(label))
        assert fake.getState(label) == demo_core.getState(label), label
        assert fake.getStateLabel(label) == demo_core.getStateLabel(label), label


def _outcome(call: Any) -> tuple[str, object]:
    """What a call did: its result, or the type and message of what it raised."""
    try:
        return ("returned", call())
    except Exception as exc:
        return (type(exc).__name__, str(exc))


def test_fake_and_demo_agree_on_an_empty_autofocus_slot(demo_core: Any) -> None:
    # FM-87: the continuous-focus calls act on the slot, and with it empty
    # they fail, or not, as MMCore does (m2 §1, measured 2026-10-16).
    fake = FakeCore.demo_like()
    outcomes: list[list[tuple[str, object]]] = []
    for core in (fake, demo_core):
        assert core.getAutoFocusOffset() == 0.0
        core.setAutoFocusDevice("")
        outcomes.append(
            [
                _outcome(core.isContinuousFocusEnabled),
                _outcome(core.isContinuousFocusLocked),
                _outcome(lambda c=core: c.enableContinuousFocus(True)),
                _outcome(lambda c=core: c.enableContinuousFocus(False)),
                _outcome(core.getAutoFocusOffset),
                _outcome(lambda c=core: c.setAutoFocusOffset(1.0)),
            ]
        )
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][2] == ("RuntimeError", "Auto-focus focus device not defined.")


def test_demo_like_config_groups_match_the_demo(demo_core: Any) -> None:
    # m2 §8.8: the groups, their presets and every setting, in order.
    fake = FakeCore.demo_like()

    def groups(core: Any) -> dict[str, dict[str, list[tuple[str, ...]]]]:
        return {
            group: {
                name: [tuple(s) for s in core.getConfigData(group, name)]
                for name in core.getAvailableConfigs(group)
            }
            for group in core.getAvailableConfigGroups()
        }

    assert fake.getAvailableConfigGroups() == tuple(
        demo_core.getAvailableConfigGroups()
    )
    assert groups(fake) == groups(demo_core)
    assert fake.getChannelGroup() == demo_core.getChannelGroup() == "Channel"


def test_fake_and_demo_agree_on_the_config_group_calls(demo_core: Any) -> None:
    fake = FakeCore.demo_like()
    outcomes: list[list[tuple[str, object]]] = []
    for core in (fake, demo_core):
        outcomes.append(
            [
                _outcome(lambda c=core: tuple(c.getAvailableConfigs("Nope"))),
                _outcome(lambda c=core: c.getConfigData("Channel", "Nope")),
                _outcome(lambda c=core: c.setConfig("Channel", "Nope")),
                _outcome(lambda c=core: c.deleteConfig("Channel", "Nope")),
                _outcome(lambda c=core: c.setChannelGroup("Nope")),
                _outcome(lambda c=core: c.setStateLabel("Objective", "Nope")),
                _outcome(lambda c=core: c.setConfig("Channel-Multiband", "FITC")),
                _outcome(lambda c=core: (c.getShutterDevice(), c.getStateLabel("LED"))),
                # defineConfig merges, and a group outlives its last preset.
                _outcome(
                    lambda c=core: c.defineConfig(
                        "Channel", "Cy5", "LED", "Label", "385nm"
                    )
                ),
                _outcome(
                    lambda c=core: [tuple(s) for s in c.getConfigData("Channel", "Cy5")]
                ),
                _outcome(lambda c=core: c.deleteConfig("Objective", "10X")),
                _outcome(lambda c=core: c.deleteConfig("Objective", "20X")),
                _outcome(lambda c=core: c.deleteConfig("Objective", "40X")),
                _outcome(lambda c=core: tuple(c.getAvailableConfigGroups())),
            ]
        )
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][7] == ("returned", ("LED Shutter", "470nm"))


def test_fake_and_demo_agree_on_the_autofocus_offset_and_lock(demo_core: Any) -> None:
    fake = FakeCore.demo_like()
    seen: list[tuple[object, ...]] = []
    for core in (fake, demo_core):
        core.enableContinuousFocus(True)
        locked = (core.isContinuousFocusEnabled(), core.isContinuousFocusLocked())
        core.setAutoFocusOffset(12.5)
        offset = core.getAutoFocusOffset()
        core.enableContinuousFocus(False)
        seen.append((locked, offset, core.getAutoFocusOffset()))
    assert seen[0] == seen[1] == ((True, True), 12.5, 12.5)
