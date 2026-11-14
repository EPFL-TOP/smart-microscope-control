"""The plate page measures the offsets between objectives on the demo-sample stand, through the view's own handlers (m5 §18, #382).

The stand is ``demo-sample`` turned as m3 §20.1 turned it (camera 30° and mirrored, plate
1.2°) with a declared shift per objective: 10X ``(5.0, -2.5)``, 20X ``(0, 0)``, 40X
``(-32.5, 9.75)``. The plate is registered under the 10X, so the offsets to recover are
written out from the shifts and never read from the code (FM-124, FM-203): 40X from 10X
``(-37.5, 12.25)``, 20X from 10X ``(-5.0, 2.5)``. They are compared within 1.0 µm, a pixel of
the 10X.

A module fixture builds the documents once, through ``run_plugin`` (the seam): the 40X's and
the 20X's calibration, one document each, and the 10X's calibration and ``plate-walls`` (A1,
H12, A12) into ``run.json``. Each test copies ``run.json`` and a copy of the profile into its
own folder, opens a stand with the demo stage's ``Velocity`` raised on its own core before
the page is built over it (the demo moves at 10 mm/s, and a run would take a minute), and
drives the view: a click is ``button.clicks += 1``. No assertion rests on a duration.
"""

from __future__ import annotations

import math
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, run_plugin
from smc.plugins.camera_calibration.plugin import (
    CameraCalibrationParams,
    CameraCalibrationPlugin,
)
from smc.plugins.objective_offsets.plugin import (
    ObjectiveOffsetsParams,
    ObjectiveOffsetsPlugin,
)
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.registry import GROUP
from smc.plugins.slots import ObjectiveOffsetsResult, WellRef
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage
from smc.ui.guide import OFFSETS_DRY_RUN, OFFSETS_NOT_TURNED_BACK
from smc.ui.plate_page import PlatePageView

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The shift of each objective, declared (m3 §20.8): a feature is centred at its stage
#: position plus this.
SHIFT = {TEN: [5.0, -2.5], TWENTY: [0.0, 0.0], FORTY: [-32.5, 9.75]}
PLATE_DEG = 1.2
#: The demo-sample's A1 (a 96-well plate centred on the origin) and its pitch.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
INDEX = {"A1": (0, 0), "H12": (11, 7), "A12": (11, 0)}
#: Where the operator "centred" each well: up to 300 µm off its true centre.
ROUGH_UM = {"A1": (300.0, -200.0), "H12": (-250.0, 150.0), "A12": (-100.0, 200.0)}
TOOLS = {
    "objective-offsets": "smc.plugins.objective_offsets.plugin:ObjectiveOffsetsPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
}
SESSION = "session 3 (#412)"


def _truth(well: str) -> tuple[float, float]:
    """A well's true centre: ``A1 + R(theta) * (col * pitch, -row * pitch)``, as the plate is placed."""
    col, row = INDEX[well]
    angle = math.radians(PLATE_DEG)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    return (
        A1_UM[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_UM[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _profile_text() -> str:
    """``demo-sample.toml`` with its ``[sample]`` turned as m3 §20.1 turned it: a profile file's text."""
    data = tomllib.loads(DEMO_SAMPLE.read_text(encoding="utf-8"))
    data["sample"].update(
        {
            "rotation_deg": PLATE_DEG,
            "camera_rotation_deg": 30.0,
            "mirrored": True,
            "objective_shift_um": SHIFT,
        }
    )
    return tomli_w.dumps(data)


def _entry_points(*names: str) -> list[EntryPoint]:
    return [EntryPoint(name=n, value=TOOLS[n], group=GROUP) for n in names]


@dataclass(frozen=True)
class Documents:
    """The run documents the tests start from."""

    run: Path
    twenty: Path
    forty: Path


@pytest.fixture(scope="module")
def documents(
    mm_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Documents:
    """The 40X's and the 20X's calibration, and the registration under the 10X, through ``run_plugin``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("page-offsets")
    profile = folder / "stand.toml"
    profile.write_text(_profile_text(), encoding="utf-8")
    docs = Documents(
        folder / "run.json", folder / "cal-20x.json", folder / "cal-40x.json"
    )
    stand = Microscope.open(Profile.load(profile))
    try:
        stand.core.setProperty("XY", "Velocity", "100000")
        turret = stand.require(ObjectiveTurret)
        assert turret.current() == TEN
        for label, path in ((TWENTY, docs.twenty), (FORTY, docs.forty)):
            turret.select(label, confirm=True)
            record = run_plugin(
                CameraCalibrationPlugin(), stand, CameraCalibrationParams(), path
            )
            assert record.status is StageStatus.done
        turret.select(TEN, confirm=True)
        record = run_plugin(
            CameraCalibrationPlugin(), stand, CameraCalibrationParams(), docs.run
        )
        assert record.status is StageStatus.done
        refs = tuple(
            WellRef(
                well=well,
                x_um=_truth(well)[0] + ROUGH_UM[well][0],
                y_um=_truth(well)[1] + ROUGH_UM[well][1],
            )
            for well in ("A1", "H12", "A12")
        )
        record = run_plugin(
            PlateWallsPlugin(),
            stand,
            PlateWallsParams(plate="96-well", refs=refs),
            docs.run,
        )
        assert record.status is StageStatus.done
    finally:
        stand.close()
    return docs


@dataclass(frozen=True)
class Rig:
    """One test's page, its view, its stand and the files it may write."""

    page: PlatePage
    view: PlatePageView
    stand: Microscope
    run_doc: Path
    profile: Path
    documents: Documents


@pytest.fixture
def make_rig(documents: Documents, tmp_path: Path) -> Iterator[Callable[..., Rig]]:
    """Opens the stand on a copy of the profile, a page over it on a copy of ``run.json``, and a view; closes what it opened."""
    opened: list[tuple[PlatePage, Microscope]] = []

    def make(*, dry_run: bool = False) -> Rig:
        run_doc = tmp_path / "run.json"
        shutil.copy(documents.run, run_doc)
        profile = tmp_path / "stand.toml"
        profile.write_text(_profile_text(), encoding="utf-8")
        stand = Microscope.open(Profile.load(profile), dry_run=dry_run)
        # The demo stage moves at its `Velocity` property, 10 mm/s by default.
        stand.core.setProperty("XY", "Velocity", "100000")
        page = PlatePage(stand, run_doc, entry_points=_entry_points(*TOOLS))
        opened.append((page, stand))
        view = PlatePageView(page)
        view.show_step("plate")
        view.refresh()
        view.tick()
        return Rig(page, view, stand, run_doc, profile, documents)

    try:
        yield make
    finally:
        for page, stand in opened:
            page.close()
            stand.close()


def _click(button: Any) -> None:
    button.clicks += 1


def _typed(*paths: Path) -> str:
    return "\n".join(str(path) for path in paths)


def _xy(line: str) -> tuple[float, float]:
    """The ``xy (x, y) µm`` of a preview line."""
    found = re.search(r"xy \(([+-]\d+\.\d+), ([+-]\d+\.\d+)\) µm", line)
    assert found is not None, line
    return float(found.group(1)), float(found.group(2))


def _within(found: tuple[float, float], declared: tuple[float, float]) -> None:
    assert abs(found[0] - declared[0]) <= 1.0, (found, declared)
    assert abs(found[1] - declared[1]) <= 1.0, (found, declared)


def test_the_page_measures_the_samples_offsets_after_the_confirmation(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    view, page, stand = rig.view, rig.page, rig.stand
    turret = stand.require(ObjectiveTurret)

    # The plate was registered from its walls with the 10X: the block is offered, and says so.
    assert view.offsets_box.visible is True
    assert TEN in view.offsets_lead_text.object
    assert view.objective_in_place_text.object == f"Objective in place: {TEN}"

    # The two documents typed, the 40X's first: the turret turns in this order.
    view.calibrations_input.value = _typed(rig.documents.forty, rig.documents.twenty)
    typed = [
        re.search(r": calibrated at (.+) on \d{4}-\d{2}-\d{2}$", line)
        for line in view.calibration_reads_text.object.splitlines()
    ]
    assert [m.group(1) for m in typed if m is not None] == [FORTY, TWENTY]
    _click(view.offsets_measure_button)

    # The first click only asks: the turret still reads the 10X.
    assert view._offsets_armed == (FORTY, TWENTY)
    assert view.offsets_confirm_box.visible is True
    assert turret.current() == TEN
    assert "objective-offsets" not in RunDocument.load(rig.run_doc).stages

    _click(view.offsets_turn_button)

    seen: list[str] = []
    started = time.monotonic()
    while not page.join(0.01):
        view.tick()
        line = view.objective_in_place_text.object
        if not seen or seen[-1] != line:
            seen.append(line)
        assert time.monotonic() - started < 300, "the run did not end"
    view.tick()
    view.refresh()

    # The in-place line moved with each turn, in the order of the documents, and the run
    # line said nothing once the run had ended.
    in_place = [line.removeprefix("Objective in place: ") for line in seen]
    assert FORTY in in_place
    assert TWENTY in in_place
    assert in_place.index(FORTY) < in_place.index(TWENTY)
    assert view.offsets_run_text.visible is False
    record = RunDocument.load(rig.run_doc).stages["objective-offsets"]
    assert record.status is StageStatus.done, page.status().outcome
    result = ObjectiveOffsetsResult.model_validate(record.result)
    shifts = {m.objective: m.shift_xy_um for m in result.measures}
    _within(shifts[FORTY], (-37.5, 12.25))
    _within(shifts[TWENTY], (-5.0, 2.5))
    assert [m.objective for m in result.measures] == [FORTY, TWENTY]
    # Nothing turned the turret back: it is at the last target, and the line says so.
    assert turret.current() == TWENTY
    assert view.objective_in_place_text.object == (
        f"Objective in place: {TWENTY} {OFFSETS_NOT_TURNED_BACK}"
    )

    # The card names each target, and the preview says what the profile held (nothing).
    card_lines = view.offsets_result_text.object.splitlines()
    assert any(
        line.strip().startswith(f"{FORTY}: shift from {TEN} (") for line in card_lines
    )
    assert any(
        line.strip().startswith(f"{TWENTY}: shift from {TEN} (") for line in card_lines
    )
    assert view.offsets_write_button.visible is True
    assert view.offsets_write_button.label == "Write these offsets into stand.toml"
    view.offsets_session_input.value = SESSION
    preview = view.offsets_plan_text.object.splitlines()
    assert preview[0] == f"Writing these replaces what {rig.profile} holds:"
    xy_lines = {
        label: _xy(next(line for line in preview if line.startswith(f"'{label}'  xy")))
        for label in (FORTY, TWENTY)
    }
    _within(xy_lines[FORTY], (-37.5, 12.25))
    _within(xy_lines[TWENTY], (-5.0, 2.5))
    assert all("was unknown" in line for line in preview if "  xy (" in line)
    assert preview[-1].startswith('measured = "')
    assert SESSION in preview[-1]

    # *Write* puts the block into the copy of the profile, and nowhere else.
    before = rig.profile.read_bytes()
    _click(view.offsets_write_button)
    after = rig.profile.read_bytes()
    assert after != before
    assert after.startswith(before)
    objectives = Profile.load(rig.profile).objectives
    assert objectives.reference == TEN
    forty = objectives.offsets[FORTY].xy_offset_um
    twenty = objectives.offsets[TWENTY].xy_offset_um
    assert forty is not None
    assert twenty is not None
    _within(forty, (-37.5, 12.25))
    _within(twenty, (-5.0, 2.5))
    assert SESSION in objectives.offsets[FORTY].measured
    assert page.status().outcome == "Offsets written into stand.toml"
    assert view.offsets_write_note.object.startswith(f"Written into {rig.profile}.")
    # The page keeps the offsets it was started with: its own profile is not reloaded.
    assert page._profile.objectives.offsets == {}


def test_without_the_confirmation_nothing_turns(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    turret = rig.stand.require(ObjectiveTurret)
    before = rig.run_doc.read_bytes()

    # The block's pairs and no `confirm_turns`: the tool's own refusal, a failed record.
    step = rig.page.offsets_step(_typed(rig.documents.forty, rig.documents.twenty))
    assert step.wait is None
    rig.page.run("objective-offsets", list(step.pairs))
    assert rig.page.join(30.0)

    record = RunDocument.load(rig.run_doc).stages["objective-offsets"]
    assert record.status is StageStatus.failed
    assert record.error is not None
    assert "give confirm_turns=true" in record.error
    assert turret.current() == TEN
    assert (
        rig.run_doc.read_bytes() != before
    )  # the failed record, and nothing else changed
    assert RunDocument.load(rig.run_doc).provider("plate") is not None


def test_a_stop_during_the_walk_says_stopped_and_leaves_the_turret_where_it_is(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    view, page = rig.view, rig.page
    turret = rig.stand.require(ObjectiveTurret)
    view.calibrations_input.value = _typed(rig.documents.forty, rig.documents.twenty)
    _click(view.offsets_measure_button)
    _click(view.offsets_turn_button)

    # STOP once the first report is in: the turret has turned to the 40X by then (FM-64, FM-88).
    started = time.monotonic()
    while page.status().progress is None:
        assert time.monotonic() - started < 120, "no report came"
        threading.Event().wait(0.01)
    page.stop()
    assert page.join(60.0)
    view.tick()
    view.refresh()

    outcome = page.status().outcome or ""
    where = turret.current()
    threading.Event().wait(0.3)
    record = RunDocument.load(rig.run_doc).stages["objective-offsets"]
    assert record.status in (StageStatus.stopped, StageStatus.failed)
    assert "stopped" in outcome.lower()
    assert where != TEN
    # Nothing turned it back, and nothing moves it after the stop.
    assert turret.current() == where
    assert view.objective_in_place_text.object == (
        f"Objective in place: {where} {OFFSETS_NOT_TURNED_BACK}"
    )


def test_a_dry_run_page_on_a_walls_plate_refuses_and_turns_nothing(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(dry_run=True)
    view, page = rig.view, rig.page
    turret = rig.stand.require(ObjectiveTurret)
    before = rig.run_doc.read_bytes()

    # The block is offered (a plate from the walls), and says why it waits.
    assert view.offsets_box.visible is True
    assert view.offsets_wait_text.object == OFFSETS_DRY_RUN
    pairs = [
        f'targets=["{FORTY}"]',
        f'calibrations=["{rig.documents.forty.as_posix()}"]',
        "confirm_turns=true",
    ]
    page.run("objective-offsets", pairs)

    assert page.status().running is None
    assert page.status().outcome == f"Objective offsets: {OFFSETS_DRY_RUN}"
    assert rig.run_doc.read_bytes() == before
    assert turret.current() == TEN

    # The page's text is the tool's own: run through the seam on this stand, on a copy of the
    # document, the tool says it.
    probe = rig.run_doc.with_name("probe.json")
    shutil.copy(rig.run_doc, probe)
    with pytest.raises(PluginRunError) as raised:
        run_plugin(
            ObjectiveOffsetsPlugin(),
            rig.stand,
            ObjectiveOffsetsParams(
                targets=(FORTY,),
                calibrations=(rig.documents.forty,),
                confirm_turns=True,
            ),
            probe,
        )
    assert raised.value.reason == OFFSETS_DRY_RUN
    # ... and it is a failed record, which is what the page's refusal keeps from replacing a
    # usable result (the probe is a copy: the page's own document was not touched).
    failed = RunDocument.load(probe).stages["objective-offsets"]
    assert failed.status is StageStatus.failed


def test_the_page_registered_from_the_sample_has_no_block(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    # The dry-run `demo-sample` page of `/ui-check`: its plate is registered from the sample's
    # centres, not from walls, so the tool has nothing to measure from (D10).
    stand = Microscope.open(Profile.load(DEMO_SAMPLE), dry_run=True)
    page = PlatePage(
        stand,
        tmp_path / "run.json",
        entry_points=_entry_points("objective-offsets", "plate-registration"),
    )
    try:
        view = PlatePageView(page)
        view.show_step("plate")
        page.choose_plate("96-well")
        pairs, wait = page.registration_step()
        assert wait is None
        page.run("plate-registration", list(pairs))
        assert page.join(60.0)
        view.refresh()
        view.tick()

        assert page.view().registered is not None
        assert page.view().offsets is None
        assert view.offsets_box.visible is False
        assert view.offsets_wait_text.object == ""
    finally:
        page.close()
        stand.close()
