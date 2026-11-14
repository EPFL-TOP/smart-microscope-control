"""A long move cut into segments on a stand that refuses ``stop()`` (m2 §11.12, #221).

On the Ti2-E a sent move runs to its target (m2 §11.1), and its move call
returns once the move has ended (session 2, #334). A profile that sets
``[safety] max_segment_um`` / ``max_z_segment_um`` makes a long move a chain of
shorter ones, so that a stop, a halt or a Ctrl-C leaves the rest unsent. Every
stand here is a ``FakeCore`` whose stages refuse their stop, as the Ti2-E's do;
the callbacks that stop or interrupt are instance attributes, removed after
(FM-98's test hazards).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus, DeviceType

from smc.hardware import Microscope
from smc.hardware.capabilities import XY, XYStage, ZStage
from smc.hardware.errors import (
    HardwareError,
    MicroscopeHaltedError,
    MotionStoppedError,
    SafetyRefusedError,
    StopNote,
    StopOutcome,
    StopRefusedError,
    stop_notes,
)
from smc.hardware.profile import (
    MicroscopeSection,
    Profile,
    RolesSection,
    SafetySection,
)
from smc.hardware.roles import Role
from smc.hardware.safety import (
    SEGMENT_END_TOLERANCE_UM,
    Safety,
    segment_targets_um,
)
from smc.testing import FakeCore, FakeDevice

LOGGER = "smc.hardware.microscope"
#: The first of three segments of a move from (0, 0) to (2500, 0) at 1000 µm.
FIRST = 2500.0 / 3
SECOND = 5000.0 / 3


def _segmented_profile(
    *,
    xy: float | None = 1000.0,
    z: float | None = 100.0,
    roles: tuple[Role, ...] = (Role.xy_stage, Role.focus),
) -> Profile:
    """The demo profile, declaring ``roles`` refused and cutting at ``xy`` / ``z`` µm."""
    demo = Profile.demo()
    safety = SafetySection.model_validate(
        {
            **demo.safety.model_dump(),
            "stop_refused_by": [role.value for role in roles],
            "max_segment_um": xy,
            "max_z_segment_um": z,
        }
    )
    return demo.model_copy(update={"safety": safety})


@contextmanager
def _stand(
    profile: Profile | None = None, *, dry_run: bool = False
) -> Iterator[tuple[FakeCore, Microscope]]:
    """A fake demo stand whose XY and Z refuse their stop; the log cleared."""
    core = FakeCore.demo_like()
    core.stop_refused = {"XY", "Z"}
    microscope = Microscope.from_core(
        cast("CMMCorePlus", core),
        profile if profile is not None else _segmented_profile(),
        dry_run=dry_run,
    )
    core.log.clear()
    try:
        yield core, microscope
    finally:
        microscope.close()


def _xy_sends(core: FakeCore) -> list[str]:
    return [line for line in core.log if line.startswith("setXYPosition")]


def _segment_lines(messages: list[str]) -> list[str]:
    return [m for m in messages if "segment" in m and m.startswith(("xy_stage", "z:"))]


# --- the cut (D2) ----------------------------------------------------------------


def test_segment_targets_cut_a_long_move_into_equal_segments_ending_on_the_target() -> (
    None
):
    assert segment_targets_um((0.0, 0.0), (2500.0, 0.0), 1000.0) == (
        (FIRST, 0.0),
        (SECOND, 0.0),
        (2500.0, 0.0),
    )


def test_the_last_segment_ends_on_the_target_as_given_not_recomputed() -> None:
    # start + (target - start) * n / n misses this target by a rounding: the move
    # must land where it was asked to (D2).
    start, target = -3656.358, 3474.337
    assert start + (target - start) * 3 / 3 != target
    ends = segment_targets_um((start,), (target,), 2500.0)
    assert len(ends) == 3
    assert ends[-1] == (target,)


def test_a_move_no_longer_than_a_segment_is_one_segment() -> None:
    assert segment_targets_um((0.0, 0.0), (1000.0, 0.0), 1000.0) == ((1000.0, 0.0),)
    assert segment_targets_um((5.0,), (5.0,), 100.0) == ((5.0,),)
    assert len(segment_targets_um((0.0,), (100.001,), 100.0)) == 2


def test_segment_targets_measure_xy_along_the_diagonal() -> None:
    # 5000 µm along the diagonal: five segments, where max(|dx|, |dy|) would cut four.
    ends = segment_targets_um((0.0, 0.0), (3000.0, 4000.0), 1000.0)
    assert len(ends) == 5
    assert ends[0] == pytest.approx((600.0, 800.0))
    assert ends[-1] == (3000.0, 4000.0)


@pytest.mark.parametrize(
    ("start", "target", "cap"),
    [
        ((math.nan,), (1.0,), 1.0),
        ((0.0,), (math.inf,), 1.0),
        ((0.0,), (1.0,), 0.0),
        ((0.0,), (1.0,), math.nan),
        ((0.0, 0.0), (1.0,), 1.0),
    ],
)
def test_segment_targets_refuse_what_cannot_be_cut(
    start: tuple[float, ...], target: tuple[float, ...], cap: float
) -> None:
    with pytest.raises(ValueError, match=r"\S"):
        segment_targets_um(start, target, cap)


# --- XY (§11.12.5) -----------------------------------------------------------------


def test_a_move_without_the_key_sends_what_it_sent_before(
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = _segmented_profile(xy=None, z=None)
    with _stand(profile) as (core, microscope):
        reads: list[str] = []
        read = core.getXPosition

        def counted(label: str | None = None) -> float:
            reads.append("x")
            return read(label)

        core.getXPosition = counted  # type: ignore[method-assign]
        try:
            with caplog.at_level(logging.INFO, logger=LOGGER):
                landed = microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.getXPosition
        assert core.log == ["setXYPosition('XY', 2500.0, 0.0)"]
    assert landed == XY(2500.0, 0.0)
    # The readback alone: no read of the start (D1, "no extra read").
    assert reads == ["x"]
    assert "xy_stage: move_to (2500.0, 0.0) µm" in caplog.messages
    assert _segment_lines(caplog.messages) == []


def test_a_long_xy_move_is_sent_in_segments_and_lands_on_its_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _stand() as (core, microscope):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            landed = microscope.require(XYStage).move_to_um(2500.0, 0.0)
        assert core.log == [
            f"setXYPosition('XY', {FIRST!r}, 0.0)",
            f"setXYPosition('XY', {SECOND!r}, 0.0)",
            "setXYPosition('XY', 2500.0, 0.0)",
        ]
    assert landed == XY(2500.0, 0.0)
    assert _segment_lines(caplog.messages) == [
        f"xy_stage: move_to (2500.0, 0.0) µm, segment 1 of 3 to ({FIRST!r}, 0.0) µm",
        f"xy_stage: move_to (2500.0, 0.0) µm, segment 2 of 3 to ({SECOND!r}, 0.0) µm",
        "xy_stage: move_to (2500.0, 0.0) µm, segment 3 of 3 to (2500.0, 0.0) µm",
    ]


def test_a_move_of_one_segment_under_the_key_keeps_todays_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _stand() as (core, microscope):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            microscope.require(XYStage).move_to_um(1000.0, 0.0)
        assert core.log == ["setXYPosition('XY', 1000.0, 0.0)"]
    assert "xy_stage: move_to (1000.0, 0.0) µm" in caplog.messages
    assert _segment_lines(caplog.messages) == []


def test_move_by_cuts_from_the_position_read_inside_the_move() -> None:
    with _stand() as (core, microscope):
        core.setXYPosition("XY", 100.0, -40.0)
        core.log.clear()
        landed = microscope.require(XYStage).move_by_um(2500.0, 0.0)
        assert core.log == [
            f"setXYPosition('XY', {100.0 + FIRST!r}, -40.0)",
            f"setXYPosition('XY', {100.0 + SECOND!r}, -40.0)",
            "setXYPosition('XY', 2600.0, -40.0)",
        ]
    assert landed == XY(2600.0, -40.0)


def test_move_by_jog_guard_sees_the_whole_offset_not_a_segment() -> None:
    with _stand() as (core, microscope):
        with pytest.raises(SafetyRefusedError) as refused:
            microscope.require(XYStage).move_by_um(6000.0, 0.0)
        assert core.log == []
    assert "exceeds the jog limit of 5000.0 µm" in refused.value.reason


def test_a_start_that_is_not_a_number_sends_nothing() -> None:
    with _stand() as (core, microscope):
        core.setXYPosition("XY", math.nan, 0.0)
        core.log.clear()
        with pytest.raises(HardwareError) as failed:
            microscope.require(XYStage).move_to_um(2500.0, 0.0)
        assert core.log == []
    assert "read its position as (nan, 0.0) µm, not a finite position" in str(
        failed.value
    )


def test_a_stop_during_a_segment_ends_the_move_after_it_and_sends_no_more() -> None:
    with _stand() as (core, microscope):
        stage = microscope.require(XYStage)
        send = core.setXYPosition

        def stopped_on_the_way(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            with pytest.raises(StopRefusedError):
                stage.stop()

        core.setXYPosition = stopped_on_the_way  # type: ignore[method-assign]
        try:
            with pytest.raises(MotionStoppedError) as stopped:
                stage.move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert _xy_sends(core) == [f"setXYPosition('XY', {FIRST!r}, 0.0)"]
    detail = stopped.value.detail
    assert detail.startswith("segment 1 of 3: a stop was requested during the move")
    assert detail.endswith(
        "; the profile cuts a move into segments of at most 1000 µm ([safety] "
        "max_segment_um), and the segments after it were not sent"
    )
    assert isinstance(stopped.value.__cause__, MotionStoppedError)
    assert stop_notes(stopped.value) == (
        StopNote("xy_stage XY", StopOutcome.rest_not_sent),
    )


def test_a_stop_between_two_segments_refuses_the_next_before_it_is_sent() -> None:
    with _stand() as (core, microscope):
        stage = microscope.require(XYStage)
        read = core.getXPosition
        stopped_once: list[str] = []

        def stop_at_the_readback(label: str | None = None) -> float:
            if _xy_sends(core) and not stopped_once:
                stopped_once.append("stop")
                with pytest.raises(StopRefusedError):
                    stage.stop()
            return read(label)

        core.getXPosition = stop_at_the_readback  # type: ignore[method-assign]
        try:
            with pytest.raises(MotionStoppedError) as stopped:
                stage.move_to_um(2500.0, 0.0)
        finally:
            del core.getXPosition
        assert _xy_sends(core) == [f"setXYPosition('XY', {FIRST!r}, 0.0)"]
    assert stopped.value.detail == (
        "segment 2 of 3: it was stopped before it started; the profile cuts a move "
        "into segments of at most 1000 µm ([safety] max_segment_um), and the "
        "segments after it were not sent"
    )


def test_a_stop_during_the_last_segment_names_it_and_adds_nothing() -> None:
    # Nothing comes after the last segment: no "were not sent", and no note.
    with _stand() as (core, microscope):
        stage = microscope.require(XYStage)
        send = core.setXYPosition

        def stop_in_the_last(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            if len(_xy_sends(core)) == 3:
                with pytest.raises(StopRefusedError):
                    stage.stop()

        core.setXYPosition = stop_in_the_last  # type: ignore[method-assign]
        try:
            with pytest.raises(MotionStoppedError) as stopped:
                stage.move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert len(_xy_sends(core)) == 3
    assert stopped.value.detail.startswith("segment 3 of 3: a stop was requested")
    assert "were not sent" not in stopped.value.detail
    assert stop_notes(stopped.value) == ()


def test_an_emergency_stop_during_a_segment_sends_no_more_segments() -> None:
    with _stand() as (core, microscope):
        send = core.setXYPosition

        def halted_on_the_way(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            with pytest.raises(StopRefusedError):
                microscope.stop()

        core.setXYPosition = halted_on_the_way  # type: ignore[method-assign]
        try:
            with pytest.raises(MotionStoppedError) as stopped:
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert _xy_sends(core) == [f"setXYPosition('XY', {FIRST!r}, 0.0)"]
        microscope.resume()
    assert stopped.value.detail.startswith("segment 1 of 3: ")
    assert stop_notes(stopped.value) == (
        StopNote("xy_stage XY", StopOutcome.rest_not_sent),
    )


def test_a_halt_between_two_segments_sends_nothing_more() -> None:
    with _stand() as (core, microscope):
        read = core.getXPosition
        halted: list[str] = []

        def halt_at_the_readback(label: str | None = None) -> float:
            if _xy_sends(core) and not halted:
                halted.append("halt")
                with pytest.raises(StopRefusedError):
                    microscope.stop()
            return read(label)

        core.getXPosition = halt_at_the_readback  # type: ignore[method-assign]
        try:
            with pytest.raises(MicroscopeHaltedError) as refused:
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.getXPosition
        assert len(_xy_sends(core)) == 1
        microscope.resume()
    assert stop_notes(refused.value) == (
        StopNote("xy_stage XY", StopOutcome.rest_not_sent),
    )


def test_ctrl_c_as_a_segments_call_returns_notes_refused_then_rest_not_sent() -> None:
    with _stand() as (core, microscope):
        send = core.setXYPosition

        def interrupted_as_it_returns(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            raise KeyboardInterrupt

        core.setXYPosition = interrupted_as_it_returns  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert core.log == [f"setXYPosition('XY', {FIRST!r}, 0.0)", "stop('XY')"]
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.refused),
        StopNote("xy_stage XY", StopOutcome.rest_not_sent),
    )


def test_ctrl_c_between_two_segments_notes_only_rest_not_sent() -> None:
    with _stand() as (core, microscope):
        read = core.getXPosition

        def interrupted_at_the_readback(label: str | None = None) -> float:
            if _xy_sends(core):
                raise KeyboardInterrupt
            return read(label)

        core.getXPosition = interrupted_at_the_readback  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.getXPosition
        # The segment had ended (idle): no stop is sent for it.
        assert core.log == [f"setXYPosition('XY', {FIRST!r}, 0.0)"]
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.rest_not_sent),
    )


def test_ctrl_c_in_the_last_segment_notes_no_rest() -> None:
    with _stand() as (core, microscope):
        send = core.setXYPosition

        def interrupted_in_the_last(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            if len(_xy_sends(core)) == 3:
                raise KeyboardInterrupt

        core.setXYPosition = interrupted_in_the_last  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
    assert stop_notes(interrupted.value) == (
        StopNote("xy_stage XY", StopOutcome.refused),
    )


def _landing_off_by(core: FakeCore, segment: int, off_um: float) -> None:
    """Segment ``segment`` stores ``off_um`` short of its target in x: an early return."""
    send = core.setXYPosition

    def short(label: str, x_um: float, y_um: float) -> None:
        send(label, x_um, y_um)
        if len(_xy_sends(core)) == segment:
            core._xy[label] = (x_um - off_um, y_um)

    core.setXYPosition = short  # type: ignore[method-assign]


def test_a_segment_that_does_not_land_stops_the_move_with_a_diagnosis() -> None:
    with _stand() as (core, microscope):
        send = core.setXYPosition

        def half_way(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            core._xy[label] = (x_um / 2, y_um)

        core.setXYPosition = half_way  # type: ignore[method-assign]
        try:
            with pytest.raises(HardwareError) as failed:
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert len(_xy_sends(core)) == 1
    assert str(failed.value) == (
        f"`xy_stage: move_to (2500.0, 0.0) µm`: segment 1 of 3 read ({FIRST / 2!r}, "
        f"0.0) µm once its call had returned and xy_stage XY read idle, 417 µm from "
        f"its target ({FIRST!r}, 0.0) µm, beyond 1 µm; the stand may report a move "
        f"as ended before it has, so segments cannot bound its moves ([safety] "
        f"max_segment_um), and the rest of the move was not sent. Read the position "
        f"before the next move, and report it: max_segment_um may have to go (m2 "
        f"§11.12)"
    )
    assert stop_notes(failed.value) == (
        StopNote("xy_stage XY", StopOutcome.rest_not_sent),
    )


def test_a_segment_just_beyond_the_tolerance_stops_the_move() -> None:
    assert SEGMENT_END_TOLERANCE_UM == 1.0
    with _stand() as (core, microscope):
        _landing_off_by(core, 1, 1.5)
        try:
            with pytest.raises(HardwareError, match=r"1\.5 µm from its target"):
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert len(_xy_sends(core)) == 1


def test_a_segment_within_the_tolerance_goes_on() -> None:
    with _stand() as (core, microscope):
        _landing_off_by(core, 1, 0.5)
        try:
            landed = microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert len(_xy_sends(core)) == 3
    assert landed == XY(2500.0, 0.0)


def test_a_segment_exactly_at_the_tolerance_goes_on() -> None:
    with _stand() as (core, microscope):
        _landing_off_by(core, 2, 1.0)
        try:
            microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert len(_xy_sends(core)) == 3


def test_a_readback_that_is_not_a_number_stops_the_move() -> None:
    # nan compares false against every bound, so a check written `d > 1` passes it.
    with _stand() as (core, microscope):
        send = core.setXYPosition

        def unreadable(label: str, x_um: float, y_um: float) -> None:
            send(label, x_um, y_um)
            core._xy[label] = (math.nan, y_um)

        core.setXYPosition = unreadable  # type: ignore[method-assign]
        try:
            with pytest.raises(HardwareError, match="segment 1 of 3 read"):
                microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
        assert len(_xy_sends(core)) == 1


def test_the_last_segment_has_no_landing_check() -> None:
    with _stand() as (core, microscope):
        _landing_off_by(core, 3, 5.0)
        try:
            landed = microscope.require(XYStage).move_to_um(2500.0, 0.0)
        finally:
            del core.setXYPosition
    assert landed == XY(2495.0, 0.0)


def test_a_segmented_move_in_dry_run_logs_each_segment_and_sends_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _stand(dry_run=True) as (core, microscope):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            landed = microscope.require(XYStage).move_to_um(2500.0, 0.0)
        assert _xy_sends(core) == []
    assert landed == XY(2500.0, 0.0)
    assert [m for m in caplog.messages if m.startswith("[dry-run]")] == [
        f"[dry-run] xy_stage: move_to (2500.0, 0.0) µm, segment 1 of 3 to ({FIRST!r}, "
        f"0.0) µm",
        f"[dry-run] xy_stage: move_to (2500.0, 0.0) µm, segment 2 of 3 to "
        f"({SECOND!r}, 0.0) µm",
        "[dry-run] xy_stage: move_to (2500.0, 0.0) µm, segment 3 of 3 to (2500.0, "
        "0.0) µm",
    ]


# --- Z, without and under an autofocus -------------------------------------------


def test_a_long_z_move_without_an_autofocus_is_sent_in_segments() -> None:
    with _stand() as (core, microscope):
        landed = microscope.require(ZStage).move_to_um(250.0)
        assert [line for line in core.log if line.startswith("setPosition")] == [
            f"setPosition('Z', {250.0 / 3!r})",
            f"setPosition('Z', {500.0 / 3!r})",
            "setPosition('Z', 250.0)",
        ]
    assert landed == 250.0


def test_a_z_move_by_cuts_from_the_position_read_inside_the_move() -> None:
    with _stand() as (core, microscope):
        core.setPosition("Z", 10.0)
        core.log.clear()
        landed = microscope.require(ZStage).move_by_um(-150.0, force=True)
        assert core.log == ["setPosition('Z', -65.0)", "setPosition('Z', -140.0)"]
    assert landed == -140.0


OFF, ON = "enableContinuousFocus(False)", "enableContinuousFocus(True)"


@contextmanager
def _ti2_stand(
    safety: dict[str, object] | None = None,
) -> Iterator[tuple[FakeCore, Microscope]]:
    """The Ti2-E's layout, refusing every stop, cutting at 1000 / 100 µm, PFS locked."""
    core = FakeCore(
        [
            FakeDevice("ZDrive", DeviceType.Stage),
            FakeDevice("PFS", DeviceType.AutoFocus),
            FakeDevice("PFSOffset", DeviceType.Stage),
            FakeDevice("XYStage", DeviceType.XYStage),
        ],
        xy_stage="XYStage",
        focus="ZDrive",
        autofocus="PFS",
    )
    core.stop_refused = {"XYStage", "ZDrive", "PFSOffset"}
    profile = Profile(
        microscope=MicroscopeSection(name="ti2e-fake"),
        roles=RolesSection.model_validate(
            {"assign": {"autofocus": "PFS", "autofocus_offset": "PFSOffset"}}
        ),
        safety=SafetySection.model_validate(
            {
                "stop_refused_by": ["xy_stage", "focus", "autofocus_offset"],
                "max_segment_um": 1000.0,
                "max_z_segment_um": 100.0,
                **(safety or {}),
            }
        ),
    )
    microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    core.autofocus_lockable = True
    core.enableContinuousFocus(True)
    core.log.clear()
    try:
        yield core, microscope
    finally:
        microscope.close()


def test_a_segmented_z_move_under_pfs_relocks_once_after_the_last_segment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _ti2_stand() as (core, microscope):
        with caplog.at_level(logging.INFO):
            landed = microscope.require(ZStage).move_to_um(250.0)
        assert core.log == [
            OFF,
            f"setPosition('ZDrive', {250.0 / 3!r})",
            f"setPosition('ZDrive', {500.0 / 3!r})",
            "setPosition('ZDrive', 250.0)",
            ON,
        ]
        assert core.isContinuousFocusLocked() is True
    assert landed == 250.0
    assert [m for m in caplog.messages if m.startswith("autofocus:")] == [
        "autofocus: suspend for z: move_to 250.0 µm",
        "autofocus: engage after z: move_to 250.0 µm",
    ]


def test_a_stop_during_a_segmented_z_move_leaves_the_autofocus_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _ti2_stand() as (core, microscope):
        z = microscope.require(ZStage)
        send = core.setPosition

        def stopped_on_the_way(label: str, z_um: float) -> None:
            send(label, z_um)
            if label == "ZDrive":
                with pytest.raises(StopRefusedError):
                    z.stop()

        core.setPosition = stopped_on_the_way  # type: ignore[method-assign]
        try:
            with (
                caplog.at_level(logging.WARNING),
                pytest.raises(MotionStoppedError) as stopped,
            ):
                z.move_to_um(250.0)
        finally:
            del core.setPosition
        sends = [line for line in core.log if line.startswith("setPosition")]
        assert sends == [f"setPosition('ZDrive', {250.0 / 3!r})"]
        assert ON not in core.log
        assert core.isContinuousFocusEnabled() is False
    assert stopped.value.detail.startswith("segment 1 of 3: ")
    assert "([safety] max_z_segment_um)" in stopped.value.detail
    left_off = [m for m in caplog.messages if "left off, because" in m]
    assert left_off == [
        "autofocus PFS: left off, because z: move_to 250.0 µm did not complete "
        "(MotionStoppedError)"
    ]


def test_a_soft_limit_refusal_of_a_segmented_z_move_never_touches_the_autofocus() -> (
    None
):
    with _ti2_stand({"z_soft_limits_um": (-100.0, 100.0)}) as (core, microscope):
        z = microscope.require(ZStage)
        with pytest.raises(SafetyRefusedError):
            z.move_by_um(150.0, force=True)
        with pytest.raises(SafetyRefusedError):
            z.move_to_um(500.0)
        assert core.log == []
        assert core.isContinuousFocusLocked() is True


# --- Safety and the emergency stop -----------------------------------------------


def test_safety_refuses_a_segment_cap_without_its_declared_role() -> None:
    with pytest.raises(ValueError, match="does not name xy_stage") as xy:
        Safety(max_jog_um=5000.0, max_segment_um=1000.0)
    assert str(xy.value) == (
        "max_segment_um cuts a move on a stand that refuses its stop, but "
        "stop_refused_by does not name xy_stage, so a stop already ends its moves: "
        "remove max_segment_um, or declare the refusal the stand showed"
    )
    with pytest.raises(ValueError, match="does not name focus"):
        Safety(
            max_jog_um=5000.0,
            stop_refused_by=frozenset({"xy_stage"}),
            max_z_segment_um=100.0,
        )
    safety = Safety(
        max_jog_um=5000.0,
        stop_refused_by=frozenset({"xy_stage", "focus"}),
        max_segment_um=1000,
        max_z_segment_um=100.0,
    )
    assert (safety.max_segment_um, safety.max_z_segment_um) == (1000.0, 100.0)
    assert Safety(max_jog_um=5000.0).max_segment_um is None


@pytest.mark.parametrize("cap", [0.0, -1.0, math.inf, math.nan, True])
def test_safety_refuses_a_segment_cap_that_is_not_a_positive_number(
    cap: float,
) -> None:
    declared = frozenset({"xy_stage", "focus"})
    with pytest.raises(ValueError, match="max_segment_um must be finite and > 0"):
        Safety(max_jog_um=5000.0, stop_refused_by=declared, max_segment_um=cap)
    with pytest.raises(ValueError, match="max_z_segment_um must be finite and > 0"):
        Safety(max_jog_um=5000.0, stop_refused_by=declared, max_z_segment_um=cap)


_HALT_CLAUSE = (
    "; a move cut into segments ([safety] max_segment_um and max_z_segment_um) ends "
    "with its current segment, since the halt refuses the next"
)


def test_the_emergency_stop_on_a_segmented_stand_says_the_halt_refuses_the_next_segment() -> (
    None
):
    with _stand() as (_, microscope), pytest.raises(StopRefusedError) as refused:
        microscope.stop()
    assert str(refused.value) == (
        "the emergency stop halted the microscope and switched autofocus Autofocus "
        "off, but xy_stage XY and z Z refused the stop, as the profile declares "
        f"([safety] stop_refused_by): a move under way runs to its target"
        f"{_HALT_CLAUSE}. The microscope stays halted: stop the stage with the "
        "stand's own controls if it must not finish its move, then call `resume()`"
    )


def test_the_emergency_stop_names_only_the_keys_of_the_roles_that_refused() -> None:
    # Z took its stop: its key says nothing about a move under way.
    profile = _segmented_profile()
    with _stand(profile) as (core, microscope):
        core.stop_refused = {"XY"}
        with pytest.raises(StopRefusedError) as refused:
            microscope.stop()
    assert (
        "a move under way runs to its target; a move cut into segments ([safety] "
        "max_segment_um) ends with its current segment, since the halt refuses the "
        "next. The microscope"
    ) in str(refused.value)


def test_the_emergency_stop_with_a_failure_adds_the_clause_to_the_refusal() -> None:
    # Z undeclared, so its fault is a failure (m2 §11.11 E7), and it has no key.
    profile = _segmented_profile(z=None, roles=(Role.xy_stage,))
    with _stand(profile) as (core, microscope):
        core.failing["Z"] = RuntimeError("device fault")
        try:
            with pytest.raises(HardwareError) as failed:
                microscope.stop()
        finally:
            core.failing.clear()
    assert type(failed.value) is HardwareError
    assert str(failed.value).endswith(
        "; xy_stage XY refused the stop, as the profile declares ([safety] "
        "stop_refused_by), so a move under way runs to its target; a move cut into "
        "segments ([safety] max_segment_um) ends with its current segment, since the "
        "halt refuses the next. The microscope stays halted: stop the stage at the "
        "stand if it still moves, then call `resume()`"
    )


def test_the_emergency_stop_without_the_keys_has_no_segment_clause() -> None:
    with (
        _stand(_segmented_profile(xy=None, z=None)) as (_, microscope),
        pytest.raises(StopRefusedError) as refused,
    ):
        microscope.stop()
    assert "segment" not in str(refused.value)
