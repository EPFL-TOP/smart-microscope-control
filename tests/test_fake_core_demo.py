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
