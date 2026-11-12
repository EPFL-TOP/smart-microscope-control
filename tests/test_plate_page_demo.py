"""The plate page on Micro-Manager's demo devices (m3 §15, #197; m5 §6, #263).

A click in drive mode and the Drive region are what the page does that moves a
stage, so they are proved on the demo stage: the click finds the well from the
registered plate, ``goto-well`` runs through the layer, and the stage reads back
within 0.5 µm of where the forward model of m3 §2.1 puts the well's centre. The
demo stage travels, so the position is read after the facade's wait (FM-85).
"""

from __future__ import annotations

import math
import os
import re
import statistics
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret, XYStage, ZStage
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.ui import controller as controller_module
from smc.ui.controller import PlatePage
from smc.ui.frames import Frame

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

GOTO_WELL = EntryPoint(
    name="goto-well", value="smc.plugins.goto_well:GotoWellPlugin", group=GROUP
)
PITCH_UM = 9000.0


def _seed(run_doc: Path, a1: tuple[float, float], rotation_deg: float) -> None:
    """The ``plate`` slot as an earlier ``plate-registration`` run left it, ``done``."""
    then = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=a1,
        rotation_deg=rotation_deg,
        method="references",
        residual_um=2.0,
        scale_error=0.001,
        n_refs=2,
    )
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=plate.model_dump(mode="json"),
        ),
    )


@pytest.mark.parametrize(
    ("well", "rotation_deg"),
    [("A1", 0.0), ("C5", 3.0)],
    ids=["a1-straight", "c5-turned"],
)
def test_a_drive_click_moves_the_demo_stage_to_the_well(
    demo_microscope: Microscope, tmp_path: Path, well: str, rotation_deg: float
) -> None:
    a1 = (1000.0, -500.0)
    run_doc = tmp_path / "run.json"
    _seed(run_doc, a1, rotation_deg)
    row, col = ord(well[0]) - ord("A"), int(well[1:]) - 1
    angle = math.radians(rotation_deg)
    dx, dy = col * PITCH_UM, -row * PITCH_UM
    target = (
        a1[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        a1[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )
    page = PlatePage(demo_microscope, run_doc, entry_points=[GOTO_WELL])
    try:
        assert page.click(*target, drive=True) == well
        assert page.join(30.0)
        status = page.status()
        assert status.outcome is not None
        assert status.outcome.startswith("goto-well done in ")
        stage = demo_microscope.require(XYStage)
        stage.wait(10.0)
        position = stage.position_um()
        assert (position.x_um, position.y_um) == pytest.approx(target, abs=0.5)
        assert page.selection == ()
        assert page.current == well
    finally:
        page.close()


def test_a_snap_on_the_demo_camera_is_a_512_square_uint16_frame(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # Measured on the demo adapter: 512 x 512, 16 bits, 10.8 ms at its default exposure.
    page = PlatePage(demo_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        page.snap()
        assert page.join(10.0)
        frame = page.frame()
        assert frame is not None, page.status().outcome
        assert frame.summary.startswith("512x512 uint16, exposure ")
        assert page.status().outcome == f"snap: {frame.summary}"
        assert frame.image.shape == (512, 512)
        assert frame.image.dtype == np.uint16
        assert frame.bit_depth == 16
        assert not frame.image.flags.writeable
        # A snap writes nothing: not a file, not the run document.
        assert list(tmp_path.iterdir()) == []
    finally:
        page.close()


@pytest.mark.parametrize("stall", [False, True], ids=["steady", "one-stall"])
@pytest.mark.wallclock
def test_live_without_its_cap_runs_at_least_five_frames_a_second_on_the_demo_camera(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stall: bool,
) -> None:
    # The demo camera snaps in about 11 ms (10.9 ms here; for the macOS runner inferred
    # from the gap after the stall in the CI log of #311). Five a second is 0.2 s a
    # frame. The period is set to 0: that runner returned timed waits about 0.1 s late
    # (inferred, not probed), which ran the capped loop at 4.5 a second, so the cap's
    # wait is proved on an owned clock in test_plate_page_controller and this test
    # measures what the page does per frame. A stall moves one gap, and the case says
    # no more than that.
    frames = 30
    monkeypatch.setattr(controller_module, "LIVE_PERIOD_S", 0.0)
    original = PlatePage._take_frame
    starts: list[float] = []
    numbers: list[int] = []

    def recording(self: PlatePage) -> Frame | str:
        starts.append(time.perf_counter())
        if stall and len(starts) == 3:
            time.sleep(1.2)
        taken = original(self)
        if isinstance(taken, Frame):
            numbers.append(taken.number)
        if len(starts) == frames:
            self.live(False)
        return taken

    monkeypatch.setattr(PlatePage, "_take_frame", recording)
    page = PlatePage(demo_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        page.live(True)
        # A hang guard and not the measurement: thirty frames at the frame step, and the
        # 1.2 s stall of one case, are far below 30 s.
        assert page.join(30.0), f"{len(starts)} frames; {page.status().outcome}"
        # The outcome is the cause when a frame failed or live was refused.
        assert numbers == list(range(1, frames + 1)), page.status().outcome
        per_frame_s = np.diff(starts).tolist()
        median_s = statistics.median(per_frame_s)
        assert median_s <= 0.2, f"median {median_s:.3f} s a frame in {per_frame_s}"
        if stall:
            # The stall is in the data, so the median above survived it.
            assert max(per_frame_s) >= 1.0, per_frame_s
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith(f"live ended: {frames} frames"), outcome
        assert not page.status().halted
    finally:
        page.close()


def test_a_demo_frame_is_drawn_by_the_page(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    if os.environ.get("SMC_REQUIRE_UI"):
        import panel  # noqa: F401  (CI must fail, not skip, when the extra is missing)
    else:
        pytest.importorskip("panel")
    from smc.ui.plate_page import PlatePageView

    page = PlatePage(demo_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        view = PlatePageView(page)
        view.snap_button.clicks += 1
        assert page.join(10.0)
        view.tick()
        drawn = view.image_source.data
        assert drawn["image"][0].shape == (512, 512)
        assert drawn["image"][0].dtype == np.uint8
        assert list(drawn["dw"]) == [512]
        assert list(drawn["dh"]) == [512]
        assert view.image_figure.title.text.startswith("512x512 uint16")
        assert view.layout.get_root() is not None
    finally:
        page.close()


def test_the_page_builds_on_the_demo(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    if os.environ.get("SMC_REQUIRE_UI"):
        import panel  # noqa: F401  (CI must fail, not skip, when the extra is missing)
    else:
        pytest.importorskip("panel")
    from smc.ui.plate_page import PlatePageView, build

    # The plugins are listed, not read from the checkout's install (FM-112).
    page = PlatePage(demo_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        assert build(page).get_root() is not None
        view = PlatePageView(page)
        assert view.status_text.object.startswith("demo | XY (0.0, 0.0) µm | Z ")
        assert len(view.wells_source.data["name"]) == 96
        assert not tmp_path.joinpath(
            "run.json"
        ).exists()  # building the page writes nothing
    finally:
        page.close()


def _wait_for_outcome(page: PlatePage, prefix: str, timeout_s: float = 10.0) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        outcome = page.status().outcome
        if outcome is not None and outcome.startswith(prefix):
            return outcome
        assert time.monotonic() < deadline, f"{prefix!r} not seen: {outcome!r}"
        time.sleep(0.005)


def test_the_drive_region_moves_the_demo_stand(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    if os.environ.get("SMC_REQUIRE_UI"):
        import panel  # noqa: F401  (CI must fail, not skip, when the extra is missing)
    else:
        pytest.importorskip("panel")
    from smc.ui.plate_page import PlatePageView

    page = PlatePage(demo_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        view = PlatePageView(page)
        # The demo's LED is a State device: the light is the one control it lacks.
        assert view.drive_note.object == (
            "light: the light source LED has no on/off that smc knows"
        )
        assert view.light_on_button.disabled
        view.xy_step_select.value = 100.0
        view.jog_buttons["right"].clicks += 1
        assert page.join(10.0)
        stage = demo_microscope.require(XYStage)
        stage.wait(10.0)
        position = stage.position_um()
        # Measured: the demo lands at (100.005, 0.0) for a 100 µm jog (FM-85).
        assert (position.x_um, position.y_um) == pytest.approx((100.0, 0.0), abs=0.5)
        view.z_step_select.value = 5.0
        view.z_buttons["up"].clicks += 1
        assert page.join(10.0)
        assert demo_microscope.require(ZStage).position_um() == pytest.approx(
            5.0, abs=0.5
        )
        view.objective_select.value = "Nikon 20X Plan Fluor ELWD"
        view.objective_button.clicks += 1
        assert demo_microscope.require(ObjectiveTurret).current() == (
            "Nikon 10X S Fluor"
        )
        view.objective_confirm_button.clicks += 1
        assert page.join(10.0)
        assert page.status().outcome == (
            "objective set: objective 'Nikon 20X Plan Fluor ELWD'"
        )
        view.autofocus_on_button.clicks += 1
        assert page.join(10.0)
        assert page.status().outcome == "autofocus engage: autofocus locked"
        view.autofocus_off_button.clicks += 1
        assert page.join(10.0)
        assert page.status().outcome == "autofocus disengage: autofocus off"
        assert view.message.object == ""
    finally:
        page.close()


def test_a_jog_during_live_on_the_demo_moves_the_stage_and_live_goes_on(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    page = PlatePage(demo_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        page.live(True)
        deadline = time.monotonic() + 10.0
        while page.frame() is None:
            assert time.monotonic() < deadline, page.status().outcome
            time.sleep(0.005)
        before = page.activities_ended
        page.jog_xy(100.0, 0.0)
        outcome = _wait_for_outcome(page, "stage jog: ")
        assert re.fullmatch(r"stage jog: XY \(100\.\d\d, -?0\.\d\d\) µm", outcome), (
            outcome
        )
        at_outcome = page.frame()
        assert at_outcome is not None
        deadline = time.monotonic() + 10.0
        while (frame := page.frame()) is None or frame.number <= at_outcome.number:
            assert time.monotonic() < deadline, page.status().outcome
            time.sleep(0.005)
        # A frame taken after the jog shows where it went (FM-96).
        assert frame.xy_um is not None
        assert frame.xy_um == pytest.approx((100.0, 0.0), abs=0.5)
        assert page.is_live
        assert page.activities_ended == before
        page.live(False)
        assert page.join(10.0)
        assert not page.status().halted
    finally:
        page.close()


# -- where the stage is on the plate, before any registration (m5 §15, #410) ------------

REGISTRATION = EntryPoint(
    name="plate-registration",
    value="smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    group=GROUP,
)
HOLDER_CENTRE_UM = (400.0, -325.0)

#: The Greiner 24-well at 19.45 mm with its A1 at (-48 225, 28 850) µm, where a holder
#: centred at (400, -325) µm puts it: three wells written out from that pitch (FM-124).
LAB_CENTRES_UM = {
    "B3": (-9325.0, 9400.0),
    "B5": (29575.0, 9400.0),
    "C3": (-9325.0, -10050.0),
}
#: What an eye gives, per well: the three centres are off by these, in µm.
BY_EYE_UM = {"B3": (100.0, -60.0), "B5": (-80.0, 90.0), "C3": (50.0, 110.0)}


@pytest.fixture
def holder_microscope(mm_available: bool) -> Iterator[Microscope]:
    """The demo stand with a profile whose ``[holder]`` says where its plate lies."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    profile = Profile.demo().model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=HOLDER_CENTRE_UM,
                measured="2026-11-10, sessions 2 and 3 (#334, #405)",
            )
        }
    )
    microscope = Microscope.open(profile, dry_run=False)
    try:
        yield microscope
    finally:
        microscope.close()


def test_three_captured_centres_register_the_lab_plate_from_the_page(
    holder_microscope: Microscope, tmp_path: Path
) -> None:
    run_doc = tmp_path / "run.json"
    page = PlatePage(holder_microscope, run_doc, entry_points=[REGISTRATION])
    try:
        page.choose_plate("greiner-24-well")
        assert page.view().pose.kind == "estimated"
        stage = holder_microscope.require(XYStage)
        for well, (x_um, y_um) in LAB_CENTRES_UM.items():
            # The stage is centred by eye on the well of a plate lying where the holder says.
            error = BY_EYE_UM[well]
            stage.move_to_um(x_um + error[0], y_um + error[1])
            stage.wait(10.0)
            page.capture_centre(well)
        assert [r.well for r in page.refs] == ["B3", "B5", "C3"]

        # The page's own pairs: the plate and the three centres it recorded.
        page.run("plate-registration", page.page_params("plate-registration"))
        assert page.join(30.0)
        record = RunDocument.load(run_doc).stages["plate-registration"]

        assert record.status is StageStatus.done, record.error
        assert record.result is not None
        plate = PlateResult.model_validate(record.result)
        assert plate.method == "references"
        assert plate.n_refs == 3
        assert plate.plate == "greiner-24-well"
        # Three centres off by about a hundred µm: a residual of 121.6 µm, below the
        # 300 µm that refuses a fit, and A1 within 55 µm of where the holder put it.
        assert plate.residual_um == pytest.approx(121.6, abs=0.5)
        assert math.dist(plate.a1_center_xy_um, (-48225.0, 28850.0)) < 55.0
        # The page's map is the registered plate's now, and says so.
        assert page.view().pose.kind == "registered"
    finally:
        page.close()


def test_a_drive_click_on_an_estimated_map_drives_to_the_estimate_and_says_so(
    holder_microscope: Microscope, tmp_path: Path
) -> None:
    # FM-217's check (m3 §5.5): with no plate registered, a drive click on the well the
    # map shows runs `goto-well` to the holder's estimate of its centre, and the record,
    # the strip and the line beside the toggle say that it was estimated.
    page = PlatePage(holder_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        page.choose_plate("96-well")
        view = page.view()
        assert view.pose.kind == "estimated"
        b2 = next(w.center_um for w in view.wells if w.well == "B2")
        stage = holder_microscope.require(XYStage)

        assert page.click(*b2, drive=True) == "B2"
        assert page.join(30.0)

        # B2 is A1 plus one pitch each way, A1 the holder's centre minus the 96-well
        # grid's centre offset (49 500, -31 500) µm: written by hand.
        after = stage.position_um()
        assert (after.x_um, after.y_um) == pytest.approx(
            (-40_100.0, 22_175.0), abs=0.02
        )
        status = page.status()
        said = "assumed: target estimated from the stand's holder, not registered"
        assert status.outcome is not None
        assert said in status.outcome
        assert said in status.goto
        record = RunDocument.load(tmp_path / "run.json").stages["goto-well"]
        assert record.status is StageStatus.done
        assert record.result is not None
        assert record.result["from_holder"] is True
    finally:
        page.close()


def test_the_cross_follows_the_demo_stage_in_live_and_after_a_drive(
    holder_microscope: Microscope, tmp_path: Path
) -> None:
    if os.environ.get("SMC_REQUIRE_UI"):
        import panel  # noqa: F401  (CI must fail, not skip, when the extra is missing)
    else:
        pytest.importorskip("panel")
    from smc.ui.plate_page import PlatePageView

    page = PlatePage(holder_microscope, tmp_path / "run.json", entry_points=[GOTO_WELL])
    try:
        view = PlatePageView(page)
        page.live(True)
        deadline = time.monotonic() + 10.0
        while page.frame() is None:
            assert time.monotonic() < deadline, page.status().outcome
            time.sleep(0.005)
        # A jog during Live: a frame taken after it carries where the stage went.
        page.jog_xy(100.0, 0.0)
        _wait_for_outcome(page, "stage jog: ")
        at_jog = page.frame()
        assert at_jog is not None
        deadline = time.monotonic() + 10.0
        while (frame := page.frame()) is None or frame.number <= at_jog.number:
            assert time.monotonic() < deadline, page.status().outcome
            time.sleep(0.005)
        view.frame_tick()
        assert page.is_live
        assert list(view.stage_source.data["x"]) == pytest.approx([100.0], abs=0.5)
        assert list(view.stage_source.data["y"]) == pytest.approx([0.0], abs=0.5)
        page.live(False)
        assert page.join(10.0)

        # Live is over: a drive moves the stage and the next tick draws it there.
        page.jog_xy(0.0, 50.0)
        assert page.join(10.0)
        view.tick()
        assert list(view.stage_source.data["x"]) == pytest.approx([100.0], abs=0.5)
        assert list(view.stage_source.data["y"]) == pytest.approx([50.0], abs=0.5)
        # It is on the estimated map, so the label says what it rests on.
        assert len(view.stage_source.data["text"]) == 1
    finally:
        page.close()
