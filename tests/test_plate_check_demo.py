"""``plate-check`` on the demo-sample stand: a plate that did not move passes, one moved in the record is refused (m3 §27.6, #542).

The stand and the documents are ``plate_check_stand``'s (§20.1's turned stand,
with the objectives' offsets in the profile): a module fixture builds them once
through ``run_plugin``, and each test copies ``run.json``. A plate "moved" is
``plate-walls``' record rewritten turned and slid (§27.1), while the plate on the
stage stays where it is: the check reads the move with its sign reversed. The
numbers expected are written out from the move, never read from the code
(FM-203); positions are compared with the stage's readback, never the target
(FM-142), and the demo's snaps differ per snap, so a distance is held below a
bound and never compared equal.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from plate_check_stand import (
    FORTY,
    TEN,
    Documents,
    build_documents,
    move_registration,
    open_stand,
    profile,
    truth,
)
from recording_autofocus import RecordingAutofocus
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Limits,
    ObjectiveTurret,
    XYStage,
)
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, Progress, load, run_plugin
from smc.plugins.plate_check.plugin import AGAIN, PlateCheckParams, PlateCheckPlugin
from smc.plugins.plate_registration.plugin import (
    PlateRegistrationParams,
    PlateRegistrationPlugin,
)
from smc.plugins.plate_walls.loop import AUTOFOCUS_LEFT_OFF, AUTOFOCUS_OFF_LINE
from smc.plugins.slots import PlateCheckResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.sample_stand import sample_refs

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

NAME = "plate-check"
NO_LIMITS_LINE = (
    "the stage reports no soft limits, so the frames' moves were not checked "
    "against any travel"
)


# ------------------------------------------------------------------ the doubles


class _Stage:
    """The stand's XY stage, counting the moves; each method is written out."""

    def __init__(self, inner: XYStage, calls: list[str]) -> None:
        self._inner = inner
        self._calls = calls
        self.moves = 0

    def position_um(self) -> XY:
        return self._inner.position_um()

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        self.moves += 1
        self._calls.append("move_to_um")
        return self._inner.move_to_um(x_um, y_um)

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        raise AssertionError("plate-check makes no relative move")

    def wait(self, timeout_s: float | None = None) -> None:
        self._inner.wait(timeout_s)

    def is_busy(self) -> bool:
        return self._inner.is_busy()

    def stop(self) -> None:
        self._inner.stop()

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return self._inner.limits_um()


class _Camera:
    """The stand's camera, counting the snaps; each method is written out."""

    def __init__(self, inner: Camera) -> None:
        self._inner = inner
        self.snaps = 0

    def snap(self) -> npt.NDArray[np.generic]:
        self.snaps += 1
        return self._inner.snap()

    def exposure_ms(self) -> float:
        return self._inner.exposure_ms()

    def set_exposure_ms(self, value_ms: float) -> float:
        return self._inner.set_exposure_ms(value_ms)

    def image_shape(self) -> tuple[int, int]:
        return self._inner.image_shape()

    def bit_depth(self) -> int:
        return self._inner.bit_depth()

    def pixel_size_um(self) -> float:
        return self._inner.pixel_size_um()


@dataclass
class _Rig:
    """A stand with the doubles that count what it does, and the autofocus' calls with the moves among them."""

    stand: Microscope
    stage: _Stage
    camera: _Camera
    calls: list[str]


# ------------------------------------------------------------------ the stands


@pytest.fixture(scope="module")
def documents(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Documents:
    """The 40X's calibration, and the 10X's calibration and registration, through ``run_plugin``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    return build_documents(tmp_path_factory.mktemp("plate-check"))


@pytest.fixture
def run_doc(documents: Documents, tmp_path: Path) -> Path:
    """A copy of the registration's ``run.json``: each test writes its own result."""
    copy = tmp_path / "run.json"
    shutil.copy(documents.run, copy)
    return copy


@pytest.fixture
def stands(mm_available: bool) -> Iterator[list[Microscope]]:
    """The stands a test opened, closed afterwards."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    opened: list[Microscope] = []
    yield opened
    for stand in opened:
        stand.close()


def _rig(
    stands: list[Microscope],
    the_profile: Profile | None = None,
    *,
    objective: str = TEN,
    dry_run: bool = False,
    autofocus: RecordingAutofocus | None = None,
) -> _Rig:
    """A stand with ``objective`` in place and the counting doubles; the autofocus's calls and the moves in one list."""
    stand = open_stand(
        the_profile if the_profile is not None else profile(), dry_run=dry_run
    )
    stands.append(stand)
    turret = stand.require(ObjectiveTurret)
    if turret.current() != objective:
        turret.select(objective, confirm=True)
    calls: list[str] = autofocus.calls if autofocus is not None else []
    if autofocus is not None:
        stand.override(Autofocus, autofocus)
    stage = _Stage(stand.require(XYStage), calls)
    camera = _Camera(stand.require(Camera))
    stand.override(XYStage, stage)
    stand.override(Camera, camera)
    return _Rig(stand, stage, camera, calls)


def _run(
    rig: _Rig,
    run_doc: Path,
    *,
    stop: threading.Event | None = None,
    on_progress: Any = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        PlateCheckPlugin(),
        rig.stand,
        PlateCheckParams(**params),
        run_doc,
        stop=stop,
        on_progress=on_progress,
    )


def _result(record: StageRecord) -> PlateCheckResult:
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    return PlateCheckResult.model_validate(record.result)


def _refused(rig: _Rig, run_doc: Path, **params: Any) -> PluginRunError:
    with pytest.raises(PluginRunError) as raised:
        _run(rig, run_doc, **params)
    assert raised.value.plugin == NAME
    # FM-98: a refusal is raised outside the handler of the error it stands for.
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    stored = RunDocument.load(run_doc).stages[NAME]
    assert stored.status is StageStatus.failed
    assert stored.result is None
    return raised.value


def _nothing_moved(rig: _Rig) -> None:
    assert rig.stage.moves == 0
    assert rig.camera.snaps == 0


def _here(rig: _Rig) -> tuple[float, float]:
    position = rig.stage.position_um()
    return (position.x_um, position.y_um)


# ---------------------------------------------------------------- registration


def test_the_entry_point_loads_plate_check() -> None:
    entry = load(NAME)

    assert entry.plugin.provides == "plate-check"
    assert entry.plugin.requires == (XYStage, Camera)
    assert entry.plugin.reads == ("plate",)
    assert entry.plugin.reads_optional == ("camera-calibration",)
    assert entry.target == "smc.plugins.plate_check.plugin:PlateCheckPlugin"


def test_the_parameters_are_checked_before_the_stand_opens() -> None:
    with pytest.raises(ValueError, match="wells"):
        PlateCheckParams(wells=tuple("ABCDEFG"))
    with pytest.raises(ValueError, match="max_shift_um"):
        PlateCheckParams(max_shift_um=0.0)
    with pytest.raises(ValueError, match="max_shift_um"):
        PlateCheckParams(max_shift_um=500.5)
    with pytest.raises(ValueError, match="search_um"):
        PlateCheckParams(search_um=1000.5)
    assert PlateCheckParams().max_shift_um == 25.0


# -------------------------------------------------------------------- it passes


def test_a_plate_that_did_not_move_passes_under_its_own_objective(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)
    before = datetime.now(timezone.utc)

    result = _result(_run(rig, run_doc))

    after = datetime.now(timezone.utc)
    check = result.check
    # The pair farthest apart of A1, H12 and A12 is the diagonal, in the
    # registration's order.
    assert [w.well for w in check.wells] == ["A1", "H12"]
    assert check.worst_um < 0.1
    assert check.worst_um == max(w.distance_um for w in check.wells)
    # Two walls a side (the walk's first frame finds the wall, the centred one
    # measures it), four sides, two wells.
    assert check.frames == 16
    assert rig.camera.snaps == 16
    assert before <= check.checked_utc <= after
    assert (check.plate, check.plate_objective, check.objective) == (
        "96-well",
        TEN,
        TEN,
    )
    assert check.shift_um == (0.0, 0.0)
    assert check.max_shift_um == 25.0
    assert abs(check.turn_deg) < 0.001
    assert math.hypot(*check.slide_xy_um) < 0.1
    for well in check.wells:
        # Under its own objective the expected centre is the registered one, and both
        # are where the 10X centres the well: the true centre plus its shift.
        assert well.expected_xy_um == well.registered_xy_um
        assert well.fit.recorded_xy_um == well.expected_xy_um
        assert well.registered_xy_um == pytest.approx(
            (truth(well.well)[0] + 5.0, truth(well.well)[1] - 2.5), abs=0.5
        )
        assert well.distance_um == pytest.approx(math.hypot(*well.difference_xy_um))
    assert re.fullmatch(r"plate-check/[0-9a-f]{12}", check.folder or "")
    assert result.assumed == (
        NO_LIMITS_LINE,
        f"the plate was checked under {TEN}: A1 and H12 within "
        f"{check.worst_um:.1f} µm of their registered centres (bound 25 µm)",
    )
    # The stage ends on the last well's fitted centre (FM-142: its readback).
    assert _here(rig) == pytest.approx(check.wells[-1].fit.center_xy_um, abs=0.1)


def test_the_check_is_recorded_in_its_slot_with_when(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)

    _run(rig, run_doc)

    document = RunDocument.load(run_doc)
    record = document.stages[NAME]
    assert record.provides == "plate-check"
    assert (
        document.unusable("plate-check", profile=record.profile, dry_run=False) is None
    )
    assert record.result is not None
    stored = PlateCheckResult.model_validate(record.result)
    assert stored.check.checked_utc.tzinfo is not None


def test_a_plate_that_did_not_move_passes_under_the_40x_with_the_profiles_offset(
    stands: list[Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(stands, objective=FORTY)

    result = _result(_run(rig, run_doc, calibration=documents.forty))

    check = result.check
    assert (check.plate_objective, check.objective) == (TEN, FORTY)
    # The profile's offsets from the 20X: the 40X's (-32.5, 9.75) minus the 10X's
    # (5.0, -2.5), written out.
    assert check.shift_um == pytest.approx((-37.5, 12.25), abs=1e-9)
    assert check.worst_um < 0.1
    for well in check.wells:
        assert well.expected_xy_um == pytest.approx(
            (well.registered_xy_um[0] - 37.5, well.registered_xy_um[1] + 12.25),
            abs=1e-9,
        )
    assert result.assumed[-1].startswith(f"the plate was checked under {FORTY}: ")


def test_the_wells_given_are_measured_in_their_order(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)

    result = _result(_run(rig, run_doc, wells=("a12", "A01", "h12")))

    assert [w.well for w in result.check.wells] == ["A12", "A1", "H12"]
    assert result.assumed[-1].startswith(
        f"the plate was checked under {TEN}: A12, A1 and H12 within "
    )


def test_progress_counts_every_side_of_both_wells(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)
    seen: list[Progress] = []

    _run(rig, run_doc, on_progress=seen.append)

    assert [(p.done, p.total) for p in seen] == [(n, 8) for n in range(1, 9)]
    assert seen[0].message == "A1 +X"


def test_no_frame_is_written_without_save_frames(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)

    result = _result(_run(rig, run_doc, save_frames=False))

    assert result.check.folder is None
    assert not (run_doc.parent / "plate-check").exists()


# ------------------------------------------------------------------ it refuses


def test_a_plate_moved_50_um_in_the_record_is_refused_naming_both_wells(
    stands: list[Microscope], run_doc: Path, caplog: pytest.LogCaptureFixture
) -> None:
    move_registration(run_doc, slide_um=(50.0, 0.0))
    rig = _rig(stands)

    with caplog.at_level(logging.WARNING, logger="smc"):
        refused = _refused(rig, run_doc)

    # The record says the wells lie 50 µm further +X than they do: each well is
    # found 50 µm -X of where it was expected.
    match = re.fullmatch(
        r"the plate moved since plate-walls registered it: A1 lies (\d+\.\d) µm from "
        r"its registered centre \(([+-]\d+\.\d), ([+-]\d+\.\d)\) and H12 (\d+\.\d) µm "
        rf"\(([+-]\d+\.\d), ([+-]\d+\.\d)\) under {TEN}, more than 25 µm; it turned "
        r"about ([+-]\d\.\d{3})° and slid about \(([+-]\d+\.\d), ([+-]\d+\.\d)\) µm: "
        + re.escape(AGAIN),
        refused.reason,
    )
    assert match is not None, refused.reason
    numbers = [float(n) for n in match.groups()]
    assert numbers[0] == pytest.approx(50.0, abs=0.2)
    assert numbers[1:3] == pytest.approx([-50.0, 0.0], abs=0.2)
    assert numbers[3] == pytest.approx(50.0, abs=0.2)
    assert numbers[4:6] == pytest.approx([-50.0, 0.0], abs=0.2)
    assert numbers[6] == pytest.approx(0.0, abs=0.002)
    assert numbers[7:9] == pytest.approx([-50.0, 0.0], abs=0.2)
    # Both wells were measured before the verdict, and the stage is left there.
    assert rig.camera.snaps == 16
    assert any("plate-check stopped at H12's" in m for m in caplog.messages)


def test_a_plate_turned_0_06_degrees_in_the_record_is_refused(
    stands: list[Microscope], run_doc: Path
) -> None:
    move_registration(run_doc, turn_deg=0.06)
    rig = _rig(stands)

    refused = _refused(rig, run_doc)

    match = re.search(
        r"A1 lies (\d+\.\d) µm .* and H12 (\d+\.\d) µm .* it turned about "
        r"([+-]\d\.\d{3})°",
        refused.reason,
    )
    assert match is not None, refused.reason
    # A turn of 0.06° about the origin moves a well 58.7 mm away by 61.5 µm
    # (2 * r * sin(0.03°)), both A1 and H12 alike; the check reads it back.
    assert float(match[1]) == pytest.approx(61.4, abs=0.5)
    assert float(match[2]) == pytest.approx(61.5, abs=0.5)
    assert float(match[3]) == pytest.approx(-0.06, abs=0.002)
    assert refused.reason.endswith(AGAIN)


def test_a_larger_bound_lets_the_moved_plate_through(
    stands: list[Microscope], run_doc: Path
) -> None:
    move_registration(run_doc, slide_um=(50.0, 0.0))
    rig = _rig(stands)

    result = _result(_run(rig, run_doc, max_shift_um=60.0))

    assert result.check.worst_um == pytest.approx(50.0, abs=0.2)
    assert result.check.max_shift_um == 60.0


def test_an_unmeasured_offset_is_refused_before_anything_moves(
    stands: list[Microscope], run_doc: Path, documents: Documents
) -> None:
    rig = _rig(stands, profile(forty=False), objective=FORTY)

    refused = _refused(rig, run_doc, calibration=documents.forty)

    assert refused.reason == (
        f"the offset between {TEN} and {FORTY} is not measured (the profile's "
        f"[objectives]): check the plate with {TEN} in place, or measure the offsets "
        "(objective-offsets)"
    )
    _nothing_moved(rig)


def test_a_plate_registered_by_eye_is_refused_before_anything_moves(
    stands: list[Microscope], tmp_path: Path, documents: Documents
) -> None:
    # Positions nobody measured: a difference of them would be a difference of
    # guesses. The 10X's calibration is given from the registration's document, so
    # that the refusal is the plate's and not the missing calibration's.
    eye = tmp_path / "eye.json"
    rig = _rig(stands)
    section = profile().sample
    assert section is not None
    record = run_plugin(
        PlateRegistrationPlugin(),
        rig.stand,
        PlateRegistrationParams(plate="96-well", refs=sample_refs(section)),
        eye,
    )
    assert record.status is StageStatus.done

    refused = _refused(rig, eye, calibration=documents.run)

    assert refused.reason == (
        "the plate was registered with references: the check compares the centres "
        "plate-walls measured, so register it with plate-walls"
    )
    _nothing_moved(rig)


def test_a_well_the_registration_did_not_measure_is_refused(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)

    refused = _refused(rig, run_doc, wells=("a1", "B2"))

    assert refused.reason == (
        "B2 was not measured by the registration: give wells among A1, H12, A12"
    )
    _nothing_moved(rig)


def test_one_well_is_not_enough_and_a_well_given_twice_is_refused(
    stands: list[Microscope], run_doc: Path
) -> None:
    rig = _rig(stands)

    one = _refused(rig, run_doc, wells=("A1",))
    twice = _refused(rig, run_doc, wells=("A1", "a01"))

    assert one.reason == (
        "give at least two wells: one well shows a slide, and not a turn about itself"
    )
    assert twice.reason == "A1 is given twice in wells"
    _nothing_moved(rig)


def test_a_dry_run_is_refused(stands: list[Microscope], run_doc: Path) -> None:
    rig = _rig(stands, dry_run=True)

    refused = _refused(rig, run_doc)

    assert refused.reason == (
        "a dry run cannot image the walls; run it without --dry-run"
    )
    _nothing_moved(rig)


def test_the_10xs_calibration_under_the_40x_is_refused_before_anything_moves(
    stands: list[Microscope], run_doc: Path
) -> None:
    # The run document's own calibration is the registration's, of the 10X: under
    # the 40X it would look for every wall with the wrong scale and turn.
    rig = _rig(stands, objective=FORTY)

    refused = _refused(rig, run_doc)

    assert refused.reason == (
        f"the camera calibration is not of this stand: it is of {TEN}, the camera "
        f"is at {FORTY}; run camera-calibration with {FORTY} in place, on a field "
        "with texture: a well's wall will do"
    )
    _nothing_moved(rig)


def test_no_calibration_at_all_is_refused_with_what_to_give(
    stands: list[Microscope], run_doc: Path
) -> None:
    document = RunDocument.load(run_doc)
    record = document.stages["camera-calibration"]
    # A calibration that did not end done is no calibration (the runner does not read it).
    write_stage(
        run_doc,
        StageRecord.model_validate(
            {**record.model_dump(mode="json"), "status": "failed", "result": None}
        ),
    )
    rig = _rig(stands)

    refused = _refused(rig, run_doc)

    assert refused.reason == (
        f"no camera calibration of {TEN}: give calibration=<a run document whose "
        "camera-calibration was made with it in place> (m5 §18.6)"
    )
    _nothing_moved(rig)


# --------------------------------------------------------------- the autofocus


def test_an_engaged_autofocus_is_switched_off_before_the_first_move_and_said_first(
    stands: list[Microscope], run_doc: Path
) -> None:
    af = RecordingAutofocus(engaged=True)
    rig = _rig(stands, autofocus=af)

    result = _result(_run(rig, run_doc))

    # Switched off and read back once, before any move; never engaged again.
    first_move = af.calls.index("move_to_um")
    assert af.calls[:first_move] == ["is_engaged", "disengage", "is_engaged"]
    assert af.calls.count("disengage") == 1
    assert "engage" not in af.calls
    assert af.engaged is False
    assert result.assumed[0] == AUTOFOCUS_OFF_LINE
    assert result.assumed[1] == NO_LIMITS_LINE


def test_a_refusal_after_the_switch_off_says_the_autofocus_was_left_off(
    stands: list[Microscope], run_doc: Path
) -> None:
    move_registration(run_doc, slide_um=(50.0, 0.0))
    af = RecordingAutofocus(engaged=True)
    rig = _rig(stands, autofocus=af)

    refused = _refused(rig, run_doc)

    assert refused.reason.startswith("the plate moved since plate-walls registered it")
    assert refused.reason.endswith(AGAIN + AUTOFOCUS_LEFT_OFF)
    assert af.engaged is False


def test_a_refusal_before_anything_moves_leaves_the_autofocus_on(
    stands: list[Microscope], run_doc: Path
) -> None:
    af = RecordingAutofocus(engaged=True)
    rig = _rig(stands, autofocus=af)

    refused = _refused(rig, run_doc, wells=("A1", "B2"))

    assert not refused.reason.endswith(AUTOFOCUS_LEFT_OFF)
    assert "disengage" not in af.calls
    assert af.engaged is True
    _nothing_moved(rig)


def test_a_stop_before_the_first_move_keeps_the_autofocus_on(
    stands: list[Microscope], run_doc: Path
) -> None:
    stop = threading.Event()
    stop.set()
    af = RecordingAutofocus(engaged=True)
    rig = _rig(stands, autofocus=af)

    record = _run(rig, run_doc, stop=stop)

    assert record.status is StageStatus.stopped
    assert "disengage" not in af.calls
    _nothing_moved(rig)
