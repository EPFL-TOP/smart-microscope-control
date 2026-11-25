"""The focus sweep's loop: a Z read off its command is read again once (m3 §21.4, §21.11 D10, #622)."""

from __future__ import annotations

import logging
import math
from typing import cast

import pytest

from smc.hardware.capabilities import Camera, ZStage
from smc.hardware.microscope import Microscope
from smc.plugins.api import StopRequested
from smc.plugins.cell_focus.algorithm import plan_planes
from smc.plugins.cell_focus.loop import (
    REREAD_WAIT_S,
    FocusDriveOffTarget,
    move_checked,
    sweep,
)
from smc.testing import FakeCore


def _sweep(
    microscope: Microscope, waits: list[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The nine planes -4 to +4 µm around Z = 0 on the fake demo stand, brenner scored."""
    z = microscope.get(ZStage)
    camera = microscope.get(Camera)
    assert z is not None
    assert camera is not None
    return sweep(
        z,
        camera,
        plan_planes(0.0, 4.0, 1.0),
        1.0,
        "brenner",
        raise_if_stopped=lambda: None,
        wait_s=waits.append,
    )


def test_a_stale_z_read_is_read_again_and_the_sweep_goes_on(
    fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    # F17: the drive read the plane before once. The sweep keeps the second read
    # of every plane, never the first, and asks for the wait each time.
    fake_core.make_positions_stale("Z", reads=1)
    waits: list[float] = []
    readbacks, scores = _sweep(fake_microscope, waits)
    assert readbacks == tuple(float(k) for k in range(-4, 5))
    assert len(scores) == 9
    assert waits == [REREAD_WAIT_S] * 9


def test_a_z_read_stale_twice_still_fails_the_sweep(
    fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    # Once, as #622 asks: a drive that did not arrive is a failed field, 0.3 s later.
    fake_core.make_positions_stale("Z", reads=2)
    waits: list[float] = []
    with pytest.raises(FocusDriveOffTarget) as caught:
        _sweep(fake_microscope, waits)
    assert str(caught.value) == (
        "Z read 0.00 µm after a move to -4.00 µm and 0.00 µm 0.3 s later, "
        "more than 0.25 µm away"
    )
    assert waits == [0.3]


def test_a_fresh_readback_waits_for_nothing(fake_microscope: Microscope) -> None:
    waits: list[float] = []
    readbacks, _ = _sweep(fake_microscope, waits)
    assert readbacks == tuple(float(k) for k in range(-4, 5))
    assert waits == []


class _Z:
    """A ``ZStage`` double: each move reads what ``reads`` says, in order, then ``position_um`` the rest."""

    def __init__(self, reads: list[float], order: list[str]) -> None:
        self._reads = reads
        self.order = order

    def move_to_um(self, target_um: float, *, keep_autofocus: bool = True) -> float:
        assert keep_autofocus is False
        self.order.append("move")
        return self._reads.pop(0)

    def position_um(self) -> float:
        self.order.append("read")
        return self._reads.pop(0)


def _double(reads: list[float], order: list[str]) -> ZStage:
    return cast("ZStage", _Z(reads, order))


def test_move_checked_waits_before_it_reads_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order: list[str] = []
    z = _double([4.0, 5.0], order)
    with caplog.at_level(logging.INFO, logger="smc.plugins.cell_focus.loop"):
        readback = move_checked(
            z, 5.0, 0.25, wait_s=lambda s: order.append(f"wait {s:g}")
        )
    # The wait comes between the move and the read, and the read is returned.
    assert order == ["move", "wait 0.3", "read"]
    assert readback == 5.0
    assert [
        r.getMessage()
        for r in caplog.records
        if r.name == "smc.plugins.cell_focus.loop"
    ] == [
        "Z read 4.00 µm after a move to 5.00 µm, and 5.00 µm 0.3 s later: "
        "the first read was stale (FM-188)"
    ]
    assert caplog.records[0].levelno == logging.INFO


def test_move_checked_reads_the_wait_at_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("smc.plugins.cell_focus.loop.REREAD_WAIT_S", 1.5)
    waits: list[float] = []
    move_checked(_double([4.0, 5.0], []), 5.0, 0.25, wait_s=waits.append)
    assert waits == [1.5]


def test_move_checked_accepts_a_read_on_the_tolerance_and_refuses_one_past_it() -> None:
    # Both sides of the bound (FM-73), with values that differ from it.
    waits: list[float] = []
    on_it = move_checked(_double([5.25], []), 5.0, 0.25, wait_s=waits.append)
    assert on_it == 5.25
    assert waits == []
    # Just past it, the second read is asked for; and just past it again fails.
    with pytest.raises(FocusDriveOffTarget, match=r"5\.30 µm after .* 5\.30 µm 0\.3 s"):
        move_checked(_double([5.3, 5.3], []), 5.0, 0.25, wait_s=waits.append)
    assert waits == [REREAD_WAIT_S]
    again = move_checked(_double([5.3, 4.8], []), 5.0, 0.25, wait_s=waits.append)
    assert again == 4.8


def test_a_nan_readback_is_read_again_and_a_second_nan_fails() -> None:
    # FM-104: a NaN is off target, the first read and the second.
    waits: list[float] = []
    assert (
        move_checked(_double([math.nan, 5.0], []), 5.0, 0.25, wait_s=waits.append)
        == 5.0
    )
    assert waits == [REREAD_WAIT_S]
    with pytest.raises(FocusDriveOffTarget, match="nan"):
        move_checked(_double([math.nan, math.nan], []), 5.0, 0.25, wait_s=waits.append)
    assert waits == [REREAD_WAIT_S, REREAD_WAIT_S]
    # A NaN second read after a read that was off is no more accepted.
    with pytest.raises(FocusDriveOffTarget, match="nan"):
        move_checked(_double([4.0, math.nan], []), 5.0, 0.25, wait_s=waits.append)


def test_a_stop_during_the_wait_passes() -> None:
    # FM-64: the wait is the caller's, so a stop in it raises StopRequested, and
    # nothing reads Z after it.
    order: list[str] = []

    def stop(seconds: float) -> None:
        order.append("wait")
        raise StopRequested

    with pytest.raises(StopRequested):
        move_checked(_double([4.0, 5.0], order), 5.0, 0.25, wait_s=stop)
    assert order == ["move", "wait"]
