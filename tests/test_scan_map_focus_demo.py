"""``scan``'s map-focus steps called directly on the demo stand, with a hand-built ``MapFocus`` (m3 §21.12.5, #688).

``test_scan_demo.py`` drives whole scans, where a map of the rig's own cannot
reach a target beyond the Z jog bound without also failing check 6, and where an
anchor try's failure rows meet a stage that never fails. Here ``_anchor_try`` and
``focus_field`` are called on the demo's devices with the state the scan would
have built, so that each of their failure rows is reached on its own, and each
asserts what the try or the field recorded and that Z made no move for it (a
counting Z).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XY, Camera, Limits, XYStage, ZStage
from smc.hardware.safety import DEFAULT_MAX_Z_JOG_UM
from smc.plugins.api import RunContext
from smc.plugins.cell_focus.focus_map import FocusMap, MapPoint
from smc.plugins.cell_focus.loop import z_jog_limit_um
from smc.plugins.scan.map_focus import (
    MapFocus,
    MapStep,
    _anchor_try,
    _check_reach,
    focus_field,
)
from smc.plugins.scan.plugin import ScanParams
from smc.plugins.slots import FocusAnchor, FocusMapUse, PlannedField

pytestmark = pytest.mark.demo

#: Four locked frames of A1 at Z 0, 100 µm apart: the map reads 0 anywhere inside.
_POINTS = tuple(MapPoint("A1", x, y, 0.0) for x in (0.0, 100.0) for y in (0.0, 100.0))
_PLANNED = PlannedField(well="A1", index=0, name="A1_0000", x_um=50.0, y_um=50.0)
_WIDE = Limits(-1000.0, 1000.0)


class _CountingZ:
    """The demo's focus drive that counts its moves; ``readback_error_um`` is added to every move's readback.

    With an error the drive does not move: it reads where it was told plus the
    error, both after the move and when it is read again, as a drive that lands
    off its command does. Every method is written out (the protocol's).
    """

    def __init__(self, inner: ZStage, *, readback_error_um: float = 0.0) -> None:
        self._inner = inner
        self._error = readback_error_um
        self._lands: float | None = None
        self.moves = 0

    def position_um(self) -> float:
        return self._inner.position_um() if self._lands is None else self._lands

    def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float:
        self.moves += 1
        if self._error:
            self._lands = z_um + self._error
            return self._lands
        return self._inner.move_to_um(z_um, keep_autofocus=keep_autofocus)

    def move_by_um(
        self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
    ) -> float:
        raise AssertionError("the map makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> Limits | None:
        return self._inner.limits_um()


class _LandsOffXY:
    """The demo's XY stage that reports every landing ``off_um`` to the right of its target."""

    def __init__(self, inner: XYStage, off_um: float) -> None:
        self._inner = inner
        self._off_um = off_um

    def position_um(self) -> XY:
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        landed = self._inner.move_to_um(x_um, y_um)
        return XY(landed.x_um + self._off_um, landed.y_um)

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("the map makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._inner.limits_um()


def _ctx() -> RunContext:
    """What the map's steps read of a context: the wait, the stop check and the progress."""
    stand_in: Callable[..., None] = lambda *args, **kwargs: None  # noqa: E731
    return cast(
        RunContext,
        SimpleNamespace(
            wait_s=stand_in, raise_if_stopped=stand_in, progress=stand_in, dry_run=False
        ),
    )


def _params(step_um: float = 2.0, check_every: int = 0) -> ScanParams:
    return ScanParams(
        cell_focus="map",
        focus_map=Path("run-10x.json"),
        focus_range_um=10.0,
        focus_step_um=step_um,
        map_check_every=check_every,
    )


def _state(
    *,
    limits: Limits = _WIDE,
    z_start_um: float = 0.0,
    jog_um: float = 100.0,
    anchors: tuple[FocusAnchor, ...] = (),
    map_z_um: float = 0.0,
) -> MapFocus:
    """The attempt's state with the map at ``map_z_um`` over A1, the start anchor 0, and ``anchors`` kept."""
    use = FocusMapUse(
        run_doc="run-10x.json",
        scan_attempt="m1",
        measured="at an unknown time",
        start_anchor_um=0.0,
        anchors=(),
        start_well="A1",
    )
    return MapFocus(
        focus_map=FocusMap(
            tuple(MapPoint(p.well, p.x_um, p.y_um, map_z_um) for p in _POINTS)
        ),
        use=use,
        limits=limits,
        z_start_um=z_start_um,
        jog_um=jog_um,
        fov_side_um=1000.0,
        resumed=anchors,
    )


def _anchor(anchor_um: float) -> FocusAnchor:
    return FocusAnchor(
        well="A1", source="start", anchor_um=anchor_um, tries=(), duration_s=0.0
    )


def test_an_anchor_try_that_lands_off_target_records_stageofftarget_and_no_readback(
    demo_microscope: Microscope,
) -> None:
    z = _CountingZ(demo_microscope.require(ZStage))
    xy = _LandsOffXY(demo_microscope.require(XYStage), 50.0)

    tried, offset = _anchor_try(
        _PLANNED,
        xy=xy,
        z=z,
        camera=demo_microscope.require(Camera),
        state=_state(),
        params=_params(),
        tolerance_um=5.0,
        ctx=_ctx(),
    )

    assert tried.error == (
        "StageOffTarget: the readback is 50.0 µm from the target, more than 5.0 µm "
        "(target (50.0, 50.0), readback (100.0, 50.0))"
    )
    assert (tried.x_um, tried.y_um, tried.map_z_um) == (None, None, None)
    assert tried.sweep is None
    assert offset is None
    assert z.moves == 0


def test_an_anchor_try_whose_planes_leave_the_soft_limits_records_sweepoutsidelimits(
    demo_microscope: Microscope,
) -> None:
    z = _CountingZ(demo_microscope.require(ZStage))

    tried, offset = _anchor_try(
        _PLANNED,
        xy=demo_microscope.require(XYStage),
        z=z,
        camera=demo_microscope.require(Camera),
        state=_state(limits=Limits(-5.0, 5.0)),
        params=_params(),
        tolerance_um=5.0,
        ctx=_ctx(),
    )

    assert tried.error == (
        "SweepOutsideLimits: the focus sweep around 0 µm would reach -10 µm, "
        "outside the soft limits [-5, 5] µm"
    )
    # The move landed and the map was read: the readback and the map's Z are kept.
    assert (tried.x_um, tried.y_um, tried.map_z_um) == pytest.approx(
        (50.0, 50.0, 0.0), abs=0.01
    )
    assert tried.sweep is None
    assert offset is None
    assert z.moves == 0


def test_an_anchor_try_whose_planes_pass_the_jog_limit_records_sweepbeyondjog(
    demo_microscope: Microscope,
) -> None:
    z = _CountingZ(demo_microscope.require(ZStage))

    tried, offset = _anchor_try(
        _PLANNED,
        xy=demo_microscope.require(XYStage),
        z=z,
        camera=demo_microscope.require(Camera),
        state=_state(jog_um=5.0),
        params=_params(),
        tolerance_um=5.0,
        ctx=_ctx(),
    )

    assert tried.error == (
        "SweepBeyondJog: the anchor sweep around 0.00 µm would reach -10.00 µm, "
        "10.00 µm from the Z the scan starts at, beyond the Z jog limit of 5 µm"
    )
    assert (tried.x_um, tried.y_um, tried.map_z_um) == pytest.approx(
        (50.0, 50.0, 0.0), abs=0.01
    )
    assert tried.sweep is None
    assert offset is None
    assert z.moves == 0


def test_an_anchor_try_whose_drive_lands_off_target_records_focusdriveofftarget(
    demo_microscope: Microscope,
) -> None:
    # A step of 1 µm: a plane is landed on within 0.25 µm, and this drive reads 1 µm
    # past every command, and again 0.3 s later.
    z = _CountingZ(demo_microscope.require(ZStage), readback_error_um=1.0)

    tried, offset = _anchor_try(
        _PLANNED,
        xy=demo_microscope.require(XYStage),
        z=z,
        camera=demo_microscope.require(Camera),
        state=_state(),
        params=_params(step_um=1.0),
        tolerance_um=5.0,
        ctx=_ctx(),
    )

    assert tried.error == (
        "FocusDriveOffTarget: Z read -9.00 µm after a move to -10.00 µm and -9.00 µm "
        "0.3 s later, more than 0.25 µm away"
    )
    assert (tried.x_um, tried.y_um, tried.map_z_um) == pytest.approx(
        (50.0, 50.0, 0.0), abs=0.01
    )
    assert tried.sweep is None
    assert offset is None
    # The first plane's move is the only one: the sweep ended there.
    assert z.moves == 1


def test_a_field_target_beyond_the_jog_limit_fails_with_nothing_moved(
    demo_microscope: Microscope,
) -> None:
    # The measured case (#688's plan): the well's anchor 8.0 on a map of 0 puts the
    # target 8 µm from the Z the scan starts at, 3 µm beyond a bound of 5.
    z = _CountingZ(demo_microscope.require(ZStage))
    state = _state(jog_um=5.0, anchors=(_anchor(8.0),))

    stepped = focus_field(
        z,
        demo_microscope.require(Camera),
        state,
        _PLANNED,
        XY(50.0, 50.0),
        _params(),
        _ctx(),
    )

    assert stepped == (
        "MapBeyondJog: the focus from the map at A1_0000 is 8.00 µm, 8.00 µm from "
        "the Z the scan starts at, beyond the Z jog limit of 5 µm"
    )
    assert z.moves == 0


#: The corners of a 1 mm square of A1, all at Z 0.
_CORNERS = tuple(
    MapPoint("A1", x, y, 0.0) for x in (0.0, 1000.0) for y in (0.0, 1000.0)
)


def _planned(index: int, x_um: float, y_um: float) -> PlannedField:
    return PlannedField(
        well="A1", index=index, name=f"A1_{index:04d}", x_um=x_um, y_um=y_um
    )


def test_a_field_at_the_bound_is_not_beyond_it_and_the_first_farthest_wins_a_tie() -> (
    None
):
    # D22: strictly greater is beyond. (60, 80) is 100 µm from the corner at the
    # origin, exactly the bound of this map; (60, 80.1) is just past it.
    state = MapFocus(
        focus_map=FocusMap(_CORNERS),
        use=_state().use,
        limits=_WIDE,
        z_start_um=0.0,
        jog_um=100.0,
        fov_side_um=100.0,
    )
    fields = [
        _planned(0, 60.0, 80.0),
        _planned(1, 60.0, 80.1),
        _planned(2, 0.0, 120.0),
        _planned(3, 120.0, 0.0),
        _planned(4, 0.0, 110.0),
    ]

    _check_reach(state, [("A1", fields)], 10.0)

    # Beyond: A1_0001 (100.08), A1_0002 and A1_0003 (120 each), A1_0004 (110); the
    # farthest is the first of the two at 120.
    assert (state.planned, state.beyond) == (5, 4)
    assert (state.farthest_name, state.farthest_um) == ("A1_0002", 120.0)
    line = state.beyond_line()
    assert line is not None
    assert line.startswith("4 of 5 fields lie more than 100.0 µm from every locked")
    assert "the farthest A1_0002 at 120.0 µm" in line


def test_the_z_jog_limit_is_the_profiles_else_the_layers_default(
    demo_microscope: Microscope,
) -> None:
    safety = demo_microscope.profile.safety
    demo_microscope.profile = demo_microscope.profile.model_copy(
        update={"safety": safety.model_copy(update={"max_z_jog_um": 4.0})}
    )
    assert z_jog_limit_um(demo_microscope) == 4.0
    demo_microscope.profile = demo_microscope.profile.model_copy(
        update={"safety": safety.model_copy(update={"max_z_jog_um": None})}
    )
    assert z_jog_limit_um(demo_microscope) == DEFAULT_MAX_Z_JOG_UM


def test_the_jog_bound_of_a_field_is_measured_from_the_z_the_scan_started_at(
    demo_microscope: Microscope,
) -> None:
    # A scan that starts at Z 100 µm, on a map at 100: a target 3 µm from the start is
    # inside a bound of 5, one 6 µm from it is not. Measured from 0 (or from the
    # command before), 103 would be refused too.
    camera = demo_microscope.require(Camera)
    z = _CountingZ(demo_microscope.require(ZStage))
    near = _state(z_start_um=100.0, jog_um=5.0, map_z_um=100.0, anchors=(_anchor(3.0),))

    stepped = focus_field(z, camera, near, _PLANNED, XY(50.0, 50.0), _params(), _ctx())

    assert isinstance(stepped, MapStep)
    assert z.moves == 1
    assert z.position_um() == pytest.approx(103.0, abs=0.01)
    far = _state(z_start_um=100.0, jog_um=5.0, map_z_um=100.0, anchors=(_anchor(6.0),))
    z = _CountingZ(demo_microscope.require(ZStage))

    stepped = focus_field(z, camera, far, _PLANNED, XY(50.0, 50.0), _params(), _ctx())

    assert stepped == (
        "MapBeyondJog: the focus from the map at A1_0000 is 106.00 µm, 6.00 µm from "
        "the Z the scan starts at, beyond the Z jog limit of 5 µm"
    )
    assert z.moves == 0


def test_the_jog_bound_of_an_anchor_sweep_is_measured_from_the_z_the_scan_started_at(
    demo_microscope: Microscope,
) -> None:
    # Centred on 100 µm, a sweep of ±10 µm lies within 10 µm of a start at 100: it
    # runs. Centred on 106 µm its plane at 112 µm is 12 µm from the start: refused.
    z = _CountingZ(demo_microscope.require(ZStage))
    tried, _ = _anchor_try(
        _PLANNED,
        xy=demo_microscope.require(XYStage),
        z=z,
        camera=demo_microscope.require(Camera),
        state=_state(z_start_um=100.0, jog_um=10.0, map_z_um=100.0),
        params=_params(),
        tolerance_um=5.0,
        ctx=_ctx(),
    )

    assert tried.error is None
    assert tried.sweep is not None
    assert z.moves == 11
    z = _CountingZ(demo_microscope.require(ZStage))

    tried, offset = _anchor_try(
        _PLANNED,
        xy=demo_microscope.require(XYStage),
        z=z,
        camera=demo_microscope.require(Camera),
        state=_state(z_start_um=100.0, jog_um=10.0, map_z_um=106.0),
        params=_params(),
        tolerance_um=5.0,
        ctx=_ctx(),
    )

    assert tried.error == (
        "SweepBeyondJog: the anchor sweep around 106.00 µm would reach 112.00 µm, "
        "12.00 µm from the Z the scan starts at, beyond the Z jog limit of 10 µm"
    )
    assert (tried.sweep, offset, z.moves) == (None, None, 0)


def test_a_check_sweep_beyond_the_jog_limit_is_skipped_and_says_it_in_its_own_words(
    demo_microscope: Microscope, caplog: pytest.LogCaptureFixture
) -> None:
    # The target is within the bound (it is the start), the sweep around it is not.
    z = _CountingZ(demo_microscope.require(ZStage))
    state = _state(jog_um=5.0, anchors=(_anchor(0.0),))

    with caplog.at_level("WARNING", logger="smc.plugins.scan"):
        stepped = focus_field(
            z,
            demo_microscope.require(Camera),
            state,
            _PLANNED,
            XY(50.0, 50.0),
            _params(check_every=1),
            _ctx(),
        )

    assert isinstance(stepped, MapStep)
    assert stepped.sweep is None
    assert state.checks_skipped == 1
    assert z.moves == 1  # the target's move, and no plane
    assert [r.getMessage() for r in caplog.records] == [
        "A1_0000: the map's Z was not checked (the check sweep around 0.00 µm would "
        "reach -10.00 µm, 10.00 µm from the Z the scan starts at, beyond the Z jog "
        "limit of 5 µm)"
    ]


def test_a_field_target_at_the_jog_limit_moves(demo_microscope: Microscope) -> None:
    z = _CountingZ(demo_microscope.require(ZStage))
    state = _state(jog_um=5.0, anchors=(_anchor(5.0),))

    stepped = focus_field(
        z,
        demo_microscope.require(Camera),
        state,
        _PLANNED,
        XY(50.0, 50.0),
        _params(),
        _ctx(),
    )

    assert isinstance(stepped, MapStep)
    assert (stepped.map_z_um, stepped.anchor_um, stepped.sweep) == (0.0, 5.0, None)
    assert z.moves == 1
    assert z.position_um() == pytest.approx(5.0, abs=0.01)
