"""The plate page's XY jog with the autofocus on: kept inside a well, switched off over a wall (m5 §6.6, #454).

The rule is pure and tested on the demo 96-well estimate: the holder at (0, 0),
A1 at (-49 500, 31 500) µm, a disc of radius 3200 µm, so 2700 µm inside the
500 µm margin. Every number is written here, never read from ``jog_crossing``.

The page runs on ``FakeCore.demo_like()`` with a ``RecordingAutofocus``. The
tick (``status()``) reads the autofocus too, so a switch-off is counted on
``calls.count("disengage")``, never on the whole list; "moves nothing" is the
stage's position read after the outcome. Texts are asserted whole (FM-45).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus
from test_plate_page_drive import FrameCamera

from recording_autofocus import RecordingAutofocus
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, XYStage
from smc.hardware.errors import CoreError
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.slots import (
    PlateResult,
    WellGeometry,
    WellRef,
    holder_estimate,
    well_geometry,
)
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import JOG_WALL_MARGIN_UM, PlatePage, jog_crossing

A1 = (-49_500.0, 31_500.0)
A2 = (-40_500.0, 31_500.0)
NOT_PLACED = "the map does not place the plate"
SWITCHED = (
    "stage jog: the autofocus was switched off before the move ({reason}), so that "
    "no lock crosses a well's wall"
)


def _geometries() -> tuple[WellGeometry, ...]:
    plan = holder_estimate("96-well", (0.0, 0.0), "not-turned")
    return tuple(
        well_geometry(plan, str(name))
        for name in np.asarray(plan.all_well_names).ravel()
    )


def _from_a1(dx_um: float) -> tuple[float, float]:
    return (A1[0] + dx_um, A1[1])


# --- the rule ----------------------------------------------------------------------


def test_the_margin_is_500_um() -> None:
    assert JOG_WALL_MARGIN_UM == 500.0


def test_a_jog_inside_one_well_stays() -> None:
    assert jog_crossing(_geometries(), True, A1, _from_a1(1000.0)) is None


def test_a_jog_that_ends_2600_um_from_the_centre_stays_and_one_at_2800_leaves() -> None:
    geometries = _geometries()
    assert jog_crossing(geometries, True, A1, _from_a1(2600.0)) is None
    assert jog_crossing(geometries, True, A1, _from_a1(2800.0)) == "it leaves A1"


def test_a_jog_into_the_next_well_leaves_the_first() -> None:
    assert jog_crossing(_geometries(), True, A1, A2) == "it leaves A1"


def test_a_jog_that_starts_on_the_plastic_may_cross() -> None:
    # Half a pitch from A1's centre: on the plastic between A1 and A2.
    between = _from_a1(4500.0)
    assert (
        jog_crossing(_geometries(), True, between, _from_a1(4600.0))
        == "it does not start 500 µm inside a well"
    )


def test_a_jog_that_starts_near_a_wall_may_cross() -> None:
    # 2800 µm from the centre: inside the well, but not 500 µm inside it.
    assert (
        jog_crossing(_geometries(), True, _from_a1(2800.0), _from_a1(2700.0))
        == "it does not start 500 µm inside a well"
    )


def test_a_jog_on_a_map_that_does_not_place_the_plate_may_cross() -> None:
    assert jog_crossing(_geometries(), False, A1, _from_a1(10.0)) == NOT_PLACED


# --- the page ----------------------------------------------------------------------


def _profile(*, holder: bool = True) -> Profile:
    demo = Profile.demo()
    if not holder:
        return demo
    return demo.model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=(0.0, 0.0), measured="a test's holder at the origin"
            )
        }
    )


#: A plate registered with A1 at (1000, -500) µm and not turned, in the run document.
REGISTERED_A1 = (1000.0, -500.0)


def _registered(method: str = "references") -> PlateResult:
    seeded = method == "seed"
    return PlateResult(
        plate="96-well",
        a1_center_xy_um=REGISTERED_A1,
        rotation_deg=0.0,
        method=method,  # type: ignore[arg-type]
        residual_um=None if seeded else 2.0,
        scale_error=None if seeded else 0.001,
        n_refs=1 if seeded else 2,
        refs=(WellRef(well="A1", x_um=REGISTERED_A1[0], y_um=REGISTERED_A1[1]),)
        if seeded
        else (),
    )


def _register(run_doc: Path, plate: PlateResult) -> None:
    """``plate`` as an earlier ``plate-registration`` left it, ``done`` and of this stand."""
    then = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={"plate": plate.plate},
            result=plate.model_dump(mode="json"),
        ),
    )


class Stand:
    """A page on the fake demo stand, the stage at A1's centre, PFS a recording fake."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        holder: bool = True,
        choose: bool = True,
        dry_run: bool = False,
        af: RecordingAutofocus | None = None,
        start: tuple[float, float] = A1,
        registered: PlateResult | None = None,
    ) -> None:
        self.core = FakeCore.demo_like()
        self.microscope = Microscope.from_core(
            cast("CMMCorePlus", self.core), _profile(holder=holder), dry_run=dry_run
        )
        self.core.setXYPosition("XY", *start)  # the store: a dry run moves nothing
        if registered is not None:
            _register(tmp_path / "run.json", registered)
        self.af = af if af is not None else RecordingAutofocus(engaged=True)
        self.microscope.override(Autofocus, self.af)
        self.page = PlatePage(self.microscope, tmp_path / "run.json", entry_points=[])
        if choose:
            self.page.choose_plate("96-well")

    def where(self) -> tuple[float, float]:
        xy = self.microscope.require(XYStage).position_um()
        return (xy.x_um, xy.y_um)

    def jog(self, dx_um: float, dy_um: float = 0.0) -> str:
        self.page.jog_xy(dx_um, dy_um)
        assert self.page.join(10.0)
        outcome = self.page.status().outcome
        assert outcome is not None
        return outcome

    def close(self) -> None:
        self.core.failing.clear()  # the kept error holds its traceback (FM-98)
        self.page.close()
        self.microscope.close()


@pytest.fixture
def stands(tmp_path: Path) -> Iterator[Callable[..., Stand]]:
    made: list[Stand] = []

    def make(**kwargs: object) -> Stand:
        stand = Stand(tmp_path, **kwargs)  # type: ignore[arg-type]
        made.append(stand)
        return stand

    try:
        yield make
    finally:
        for stand in made:
            stand.close()


def test_a_jog_inside_one_well_keeps_the_autofocus_on(
    stands: Callable[..., Stand],
) -> None:
    stand = stands()
    outcome = stand.jog(1000.0)
    assert outcome == "stage jog: XY (-48500.00, 31500.00) µm"
    assert stand.af.calls.count("disengage") == 0
    assert stand.af.engaged is True
    assert stand.where() == _from_a1(1000.0)


def test_a_jog_that_leaves_a_well_switches_the_autofocus_off_first_and_says_so(
    stands: Callable[..., Stand],
) -> None:
    stand = stands()
    outcome = stand.jog(3000.0)
    assert outcome == (
        f"{SWITCHED.format(reason='it leaves A1')}: engage it again here; "
        "XY (-46500.00, 31500.00) µm"
    )
    assert stand.af.calls.count("disengage") == 1
    assert stand.af.engaged is False
    assert stand.where() == _from_a1(3000.0)


def test_the_switch_off_comes_before_the_jog(
    stands: Callable[..., Stand], monkeypatch: pytest.MonkeyPatch
) -> None:
    stand = stands()
    xy = stand.microscope.require(XYStage)
    real = xy.move_by_um
    seen: list[bool] = []

    def move_by_um(dx_um: float, dy_um: float, *, force: bool = False) -> object:
        seen.append(stand.af.engaged)
        return real(dx_um, dy_um, force=force)

    monkeypatch.setattr(xy, "move_by_um", move_by_um)
    stand.jog(3000.0)
    assert seen == [False]


def test_a_jog_in_y_that_leaves_a_well_switches_the_autofocus_off(
    stands: Callable[..., Stand],
) -> None:
    stand = stands()
    outcome = stand.jog(0.0, 3000.0)
    assert outcome == (
        f"{SWITCHED.format(reason='it leaves A1')}: engage it again here; "
        "XY (-49500.00, 34500.00) µm"
    )
    assert stand.af.calls.count("disengage") == 1


def test_a_jog_in_y_inside_one_well_keeps_the_autofocus_on(
    stands: Callable[..., Stand],
) -> None:
    stand = stands()
    assert stand.jog(0.0, -1000.0) == "stage jog: XY (-49500.00, 30500.00) µm"
    assert stand.af.calls.count("disengage") == 0


@pytest.mark.parametrize(
    ("dx_um", "dy_um", "leaves"),
    [
        # From 2000 µm right of A1's centre, the wall is 700 µm away on +X...
        (1000.0, 0.0, True),
        # ...and 4700 µm away on -X: the step's sign is the jog's.
        (-1000.0, 0.0, False),
        # A Y step from there ends 2236 µm from the centre: X and Y are not swapped.
        (0.0, 1000.0, False),
    ],
    ids=["toward-the-wall", "away-from-it", "across-in-y"],
)
def test_a_jog_from_off_the_centre_is_judged_at_its_own_end(
    stands: Callable[..., Stand], dx_um: float, dy_um: float, leaves: bool
) -> None:
    stand = stands(start=_from_a1(2000.0))
    outcome = stand.jog(dx_um, dy_um)
    assert outcome.startswith(SWITCHED.format(reason="it leaves A1")) is leaves
    assert stand.af.calls.count("disengage") == (1 if leaves else 0)


@pytest.mark.parametrize("method", ["references", "seed"])
def test_a_registered_plate_places_the_map_for_the_jog(
    stands: Callable[..., Stand], method: str
) -> None:
    # No holder: without the registration the map would not place the plate.
    stand = stands(holder=False, start=REGISTERED_A1, registered=_registered(method))
    assert stand.jog(1000.0) == "stage jog: XY (2000.00, -500.00) µm"
    assert stand.af.calls.count("disengage") == 0
    outcome = stand.jog(2000.0)  # ends 3000 µm from A1's centre
    assert outcome.startswith(f"{SWITCHED.format(reason='it leaves A1')}: ")
    assert stand.af.calls.count("disengage") == 1


def test_a_jog_on_an_unplaced_map_switches_the_autofocus_off(
    stands: Callable[..., Stand],
) -> None:
    stand = stands(holder=False)
    outcome = stand.jog(1000.0)
    assert outcome == (
        f"{SWITCHED.format(reason=NOT_PLACED)}: engage it again here; "
        "XY (-48500.00, 31500.00) µm"
    )
    assert stand.af.calls.count("disengage") == 1


def test_a_jog_on_the_placeholder_plate_switches_the_autofocus_off(
    stands: Callable[..., Stand],
) -> None:
    # The holder places a plate, but not one the operator chose.
    stand = stands(choose=False)
    outcome = stand.jog(1000.0)
    assert outcome.startswith(f"{SWITCHED.format(reason=NOT_PLACED)}: ")
    assert stand.af.calls.count("disengage") == 1


def test_a_jog_on_an_unreadable_run_document_switches_the_autofocus_off(
    stands: Callable[..., Stand], tmp_path: Path
) -> None:
    stand = stands()
    (tmp_path / "run.json").write_text("{not json", encoding="utf-8")
    outcome = stand.jog(1000.0)
    assert outcome.startswith(f"{SWITCHED.format(reason=NOT_PLACED)}: ")
    assert stand.af.calls.count("disengage") == 1


def test_a_jog_with_the_autofocus_off_switches_nothing(
    stands: Callable[..., Stand],
) -> None:
    stand = stands(af=RecordingAutofocus(engaged=False))
    outcome = stand.jog(3000.0)
    assert outcome == "stage jog: XY (-46500.00, 31500.00) µm"
    assert stand.af.calls.count("disengage") == 0


def test_a_jog_in_a_dry_run_switches_nothing(stands: Callable[..., Stand]) -> None:
    # A switch-off acts in a dry run (m2 §3.2), and nothing moves to carry a lock.
    stand = stands(dry_run=True)
    outcome = stand.jog(3000.0)
    assert not outcome.startswith("stage jog: the autofocus")
    assert stand.af.calls.count("disengage") == 0
    assert stand.af.engaged is True
    assert stand.where() == A1


def _not_switched(cause: str) -> str:
    return (
        f"stage jog: the autofocus could not be switched off before the move "
        f"({cause}): the stage did not move, and PFS may still be on: switch it off, "
        "then jog again"
    )


def test_a_switch_off_that_raises_during_a_jog_moves_nothing_and_says_so(
    stands: Callable[..., Stand],
) -> None:
    stand = stands(af=RecordingAutofocus(engaged=True, fails="disengage"))
    outcome = stand.jog(3000.0)
    assert outcome == _not_switched("CoreError: disengage failed on the fake")
    assert stand.where() == A1


def test_a_state_read_that_raises_during_a_jog_moves_nothing(
    stands: Callable[..., Stand],
) -> None:
    stand = stands(af=RecordingAutofocus(engaged=True, fails="is_engaged"))
    outcome = stand.jog(1000.0)  # inside the well: the read alone decides
    assert outcome == _not_switched("CoreError: is_engaged failed on the fake")
    assert stand.af.calls.count("disengage") == 0
    assert stand.where() == A1


def test_a_read_after_the_switch_off_that_raises_during_a_jog_moves_nothing(
    stands: Callable[..., Stand],
) -> None:
    stand = stands(af=RecordingAutofocus(engaged=True, read_fails_after_off=True))
    outcome = stand.jog(3000.0)
    assert outcome == _not_switched(
        "CoreError: is_engaged failed on the fake after the switch-off"
    )
    assert stand.where() == A1


def test_a_position_read_that_raises_with_the_autofocus_on_moves_nothing(
    stands: Callable[..., Stand],
) -> None:
    # The jog cannot be judged: it is not made, and the autofocus is left as it is.
    stand = stands()
    stand.core.failing["XY"] = CoreError("the stage did not answer")
    outcome = stand.jog(1000.0)
    stand.core.failing.clear()
    assert outcome == _not_switched("CoreError: the stage did not answer")
    assert stand.af.calls.count("disengage") == 0
    assert stand.where() == A1


def test_an_autofocus_that_still_reads_on_after_the_switch_off_moves_nothing(
    stands: Callable[..., Stand],
) -> None:
    stand = stands(af=RecordingAutofocus(engaged=True, stays_on=True))
    outcome = stand.jog(3000.0)
    assert outcome == (
        "stage jog: the autofocus still reads on after it was switched off before the "
        "move: the stage did not move: switch it off at the stand, then jog again"
    )
    assert stand.af.calls.count("disengage") == 1
    assert stand.where() == A1


def test_a_jog_refused_after_the_switch_off_says_both(
    stands: Callable[..., Stand],
) -> None:
    # Past the jog limit (5000 µm): the switch-off is made and said, then the refusal.
    stand = stands()
    outcome = stand.jog(6000.0)
    assert outcome == (
        f"{SWITCHED.format(reason='it leaves A1')}; then the jog failed: "
        "SafetyRefusedError: jog (6000.0, 0.0) µm exceeds the jog limit of 5000.0 µm "
        "— pass force=True"
    )
    assert stand.af.calls.count("disengage") == 1
    assert stand.af.engaged is False
    assert stand.where() == A1


def test_a_halted_stand_switches_the_autofocus_off_and_says_the_jog_failed(
    stands: Callable[..., Stand],
) -> None:
    stand = stands()
    stand.microscope.stop()
    stand.af.engaged = True  # the stop switched it off; on again at the stand
    try:
        outcome = stand.jog(3000.0)
    finally:
        stand.microscope.resume()
    assert outcome.startswith(
        f"{SWITCHED.format(reason='it leaves A1')}; then the jog failed: "
        "MicroscopeHaltedError: "
    )
    assert stand.af.engaged is False
    assert stand.where() == A1


def _wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, f"{what} not seen within {timeout_s} s"
        time.sleep(0.005)


def test_a_jog_that_leaves_a_well_during_live_switches_the_autofocus_off_between_two_frames(
    stands: Callable[..., Stand],
) -> None:
    stand = stands()
    stand.microscope.override(Camera, FrameCamera())
    page = stand.page
    expected = (
        f"{SWITCHED.format(reason='it leaves A1')}: engage it again here; "
        "XY (-46500.00, 31500.00) µm"
    )
    page.live(True)
    try:
        _wait_for(
            lambda: (frame := page.frame()) is not None and frame.number > 1,
            "a second frame",
        )
        before = page.activities_ended
        page.jog_xy(3000.0, 0.0)
        _wait_for(
            lambda: page.status().outcome == expected,
            f"the outcome (last: {page.status().outcome!r})",
        )
        at_outcome = page.frame()
        assert at_outcome is not None
        _wait_for(
            lambda: (
                (frame := page.frame()) is not None and frame.number > at_outcome.number
            ),
            "a frame after the outcome",
        )
        after = page.frame()
        assert after is not None
        # A frame taken after the outcome shows where the stage went (FM-96).
        assert after.xy_um == _from_a1(3000.0)
        assert page.is_live
        assert page.activities_ended == before  # handed to live, not an activity
    finally:
        page.live(False)
    assert page.join(10.0)
    assert stand.af.calls.count("disengage") == 1
    assert stand.af.engaged is False
