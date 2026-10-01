"""The ``Properties`` contract, on every backend (design §3, §8).

MMCore ignores a set on a read-only property without an error (measured on
the demo, and copied by the fake), so the refusal asserted here is the
backend's own check: remove it and this contract fails (FM-43).
"""

from __future__ import annotations

import re

import pytest

from smc.hardware.capabilities import Properties
from smc.hardware.errors import HardwareError


def test_devices_excludes_core(properties: Properties) -> None:
    devices = properties.devices()
    assert devices
    assert "Core" not in devices


def test_describe_answers_for_every_device(properties: Properties) -> None:
    described = {device: properties.describe(device) for device in properties.devices()}
    for device, entries in described.items():
        assert all(entry.device == device for entry in entries), device
    # Not "every device has one": the demo's DHub has no property at all.
    assert any(described.values())


def test_read_only_property_refuses_set_and_keeps_its_value(
    properties: Properties,
) -> None:
    found = next(
        (
            info
            for device in properties.devices()
            for info in properties.describe(device)
            if info.read_only
        ),
        None,
    )
    if found is None:
        pytest.skip("this stand reports no read-only property")
    device, name = found.device, found.name
    before = properties.get(device, name)
    # The exact phrase, so no fixture text can satisfy it (FM-45).
    with pytest.raises(HardwareError, match=re.escape(f"{device}.{name} is read-only")):
        properties.set(device, name, "smc-contract")
    assert properties.get(device, name) == before
