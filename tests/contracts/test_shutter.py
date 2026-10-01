"""The ``Shutter`` contract, on every backend (design §3, §8).

At a stand, opening the shutter turns the light on: the ``microscope``
fixture closes it again at teardown if it started closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smc.hardware.capabilities import Shutter

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from conftest import Envelope


def test_open_and_close_round_trip(shutter: Shutter) -> None:
    assert shutter.set_open(True) is True
    assert shutter.is_open() is True
    assert shutter.set_open(False) is False
    assert shutter.is_open() is False


def test_auto_shutter_round_trips(shutter: Shutter, envelope: Envelope) -> None:
    start = envelope.auto_shutter
    assert start is not None
    assert shutter.set_auto_shutter(not start) is (not start)
    assert shutter.auto_shutter() is (not start)
