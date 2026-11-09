"""The plate page's Panel layer, without a browser (m3 §15, #197).

``figure._trigger_event(Tap(...))`` is what a tap raises, ``button.clicks += 1``
what a click raises, and ``source.data`` what the map draws: the page is tested
through the methods and the models the way a session drives them. What no test
here does is run a browser (the tap of a real tab, the look, the periodic
callback of a session): that is for the owner at the stand (#198).

Panel's ``PendingDeprecationWarning`` is an error here: a widget spelling that
warns on the version the lab installs (``name=``, ``button_type=``) fails the
test at the call, and is not filtered.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import re
import socket
import string
import threading
import time
import urllib.request
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

import numpy as np
import useq
from bokeh.events import Tap
from bokeh.models import Circle, Line, Rect
from pymmcore_plus import CMMCorePlus

import smc.ui.plate_page as plate_page
from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, LightSource, XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.algorithm import plan_fields
from smc.plugins.slots import (
    DensityResult,
    DensityWell,
    PlateResult,
    ScanPlanResult,
    WallFrame,
    WallImages,
    WellFit,
)
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import (
    LIVE_PERIOD_S,
    PageServerError,
    PageStatus,
    PlatePage,
    ResultCard,
)
from smc.ui.frames import display_image
from smc.ui.guide import (
    NO_CALIBRATION_FOR_SIDES,
    WALLS_DRY_RUN,
    WALLS_NEED_CALIBRATION,
    centres_text,
    wall_touches_note,
    walls_card_lines,
    walls_centres_problem,
)
from smc.ui.plate_page import PlatePageView, build, card_text, serve

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
A1_OFF = (1000.0, -500.0)
TURN_DEG = 7.0
PITCH_96_UM = 9000.0

REAL = {
    "camera-calibration": "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
    "plate-walls": "smc.plugins.plate_walls.plugin:PlateWallsPlugin",
    "scan-plan": "smc.plugins.scan_plan.plugin:ScanPlanPlugin",
}


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class SlowScan:
    """Registered as ``scan`` (a name ``STAGES`` shows): runs until released or stopped."""

    name = provides = "scan"
    summary = "A test scan that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        SlowScan.started.set()
        while not SlowScan.release.is_set():
            ctx.wait_s(0.05)
        return NoResult()


class CheckpointingScan(SlowScan):
    """A ``scan`` that saves a checkpoint, then waits for the stop: a scan after its first well."""

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        ctx.save_checkpoint(NoResult())
        SlowScan.started.set()
        while True:
            ctx.wait_s(0.05)


class QuickTool:
    """A tool that ends at once: no tick of another tab can fall inside its run."""

    name = provides = "quick"
    summary = "A test plugin that ends at once."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        return NoResult()


def _entry_points(*names: str) -> list[EntryPoint]:
    found = [EntryPoint(name=n, value=REAL[n], group=GROUP) for n in names]
    found.append(EntryPoint(name="scan", value=f"{__name__}:SlowScan", group=GROUP))
    return found


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    SlowScan.started = threading.Event()
    SlowScan.release = threading.Event()
    try:
        yield
    finally:
        SlowScan.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    controller = PlatePage(
        fake_microscope,
        run_doc,
        entry_points=_entry_points("goto-well", "plate-registration", "scan-plan"),
    )
    try:
        yield controller
    finally:
        controller.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _seed(
    run_doc: Path,
    plugin: str,
    provides: str,
    result: PluginResult,
    *,
    minute: int = 0,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A done record of ``plugin``; a plate result's params name its plate, as the runner writes them (m5 §11.3).

    ``reads`` names the attempt of each slot the record read (``a<minute>`` of the
    provider): a record that read an older attempt than the current one is stale.
    """
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt=f"a{minute}",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={"plate": result.plate} if isinstance(result, PlateResult) else {},
            result=result.model_dump(mode="json"),
            reads=reads or {},
        ),
    )


def _plate(plate: str = "96-well") -> PlateResult:
    return PlateResult(
        plate=plate,
        a1_center_xy_um=A1_OFF,
        rotation_deg=TURN_DEG,
        method="references",
        residual_um=2.0,
        scale_error=0.001,
        n_refs=2,
    )


def _centre(
    well: str, pitch_um: float = PITCH_96_UM, rotation_deg: float = TURN_DEG
) -> tuple[float, float]:
    """The forward model of m3 §2.1 for a plate with A1 at ``A1_OFF``, written out."""
    row = string.ascii_uppercase.index(well[0])
    col = int(well[1:]) - 1
    angle = math.radians(rotation_deg)
    dx, dy = col * pitch_um, -row * pitch_um
    return (
        A1_OFF[0] + math.cos(angle) * dx - math.sin(angle) * dy,
        A1_OFF[1] + math.sin(angle) * dx + math.cos(angle) * dy,
    )


def _tap(view: PlatePageView, x: float, y: float) -> None:
    view.figure._trigger_event(Tap(view.figure, x=x, y=y, sx=0, sy=0))


def _wells_by_name(view: PlatePageView) -> dict[str, dict[str, Any]]:
    data = view.wells_source.data
    return {
        name: {key: data[key][i] for key in data} for i, name in enumerate(data["name"])
    }


def _status(**changes: Any) -> PageStatus:
    base: dict[str, Any] = {
        "stand": "demo",
        "dry_run": False,
        "running": None,
        "progress": None,
        "outcome": None,
        "stop_note": None,
        "xy_um": (1000.0, -500.0),
        "z_um": 12.5,
        "objective": "Nikon 10X S Fluor",
        "autofocus": "autofocus off",
        "light_on": None,
        "moving": (),
        "halted": False,
        "stop_refused": (),
        "problems": (),
    }
    return PageStatus(**{**base, **changes})


# --- building ---------------------------------------------------------------------


def test_the_page_builds(page: PlatePage) -> None:
    layout = build(page)
    assert layout.get_root() is not None


def test_the_page_builds_every_tool_the_registry_holds_in_workflow_order(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    # `goto-well` is no stage of the workflow: it follows under `More`.
    assert list(view.tool_runs) == [
        "plate-registration",
        "scan-plan",
        "scan",
        "goto-well",
    ]
    assert list(view.tool_fields["scan-plan"]) == [
        "objective",
        "overlap",
        "coverage",
        "serpentine",
        "rows",
        "columns",
        "clip_to_well",
        "pixel_size_um",
    ]
    assert "wells" not in view.tool_fields["scan-plan"]
    assert set(view.tool_fields["plate-registration"]) == {"max_rotation_deg"}


# --- the image, Snap and the result cards ----------------------------------------------


class FrameCamera:
    """The six methods of ``Camera``, handing out ``frame``; a blocking snap is released by the test."""

    def __init__(
        self, frame: np.ndarray, *, bit_depth: int = 16, block: bool = False
    ) -> None:
        self.frame = frame
        self._bit_depth = bit_depth
        self._block = block
        self.entered = threading.Event()
        self.release = threading.Event()

    def snap(self) -> np.ndarray:
        self.entered.set()
        if self._block:
            assert self.release.wait(10), "the test never released the snap"
        return self.frame

    def exposure_ms(self) -> float:
        return 5.0

    def set_exposure_ms(self, value_ms: float) -> float:
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (int(self.frame.shape[0]), int(self.frame.shape[1]))

    def bit_depth(self) -> int:
        return self._bit_depth

    def pixel_size_um(self) -> float:
        return 0.65


def _rows_columns(rows: int, columns: int) -> np.ndarray:
    """``row * 1000 + column``: not symmetric, so a flip or a transposition shows."""
    grid = np.arange(rows)[:, None] * 1000 + np.arange(columns)[None, :]
    return grid.astype(np.uint16)


def _snap_and_draw(view: PlatePageView, page: PlatePage) -> None:
    view.snap_button.clicks += 1
    assert page.join(10.0)
    view.tick()


def test_a_snap_is_drawn_by_the_next_tick_flipped_and_at_the_frames_size(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10), bit_depth=14))
    view = PlatePageView(page)
    assert view.image_figure.title.text == "no frame yet"
    assert view.tabs.active == 0
    _snap_and_draw(view, page)
    frame = page.frame()
    assert frame is not None
    drawn = view.image_source.data
    # Bokeh draws the first row at the bottom: the camera's top row is the last.
    expected = display_image(frame.image, bit_depth=14)[::-1]
    assert drawn["image"][0].shape == (6, 10)
    assert (drawn["image"][0] == expected).all()
    assert (drawn["image"][0][-1] == display_image(frame.image, bit_depth=14)[0]).all()
    assert list(drawn["dw"]) == [10]
    assert list(drawn["dh"]) == [6]
    assert view.image_figure.title.text == frame.summary
    assert view.tabs.active == 1
    # The models with a frame in them still build, with the image tab active.
    assert view.layout.get_root() is not None


def test_a_flat_frame_is_drawn_against_the_cameras_own_bit_depth(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A saturated 12-bit frame is white and a mid-grey one is mid-grey: the camera's
    # bit depth reaches `display_image`, or 4095 of 16 bits would be near black.
    camera = FrameCamera(np.full((6, 10), 4095, dtype=np.uint16), bit_depth=12)
    fake_microscope.override(Camera, camera)
    view = PlatePageView(page)
    _snap_and_draw(view, page)
    assert (view.image_source.data["image"][0] == 255).all()
    camera.frame = np.full((6, 10), 2048, dtype=np.uint16)
    _snap_and_draw(view, page)
    assert (view.image_source.data["image"][0] == 127).all()
    camera.frame = np.zeros((6, 10), dtype=np.uint16)
    _snap_and_draw(view, page)
    assert (view.image_source.data["image"][0] == 0).all()


def test_the_tabs_are_not_dynamic_so_the_plate_figure_keeps_its_view_and_its_frame(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    assert list(view.tabs._names) == ["Plate", "Image"]
    # A dynamic tab destroys the figure's view when another tab is shown, and the
    # figure's DataRange1d keeps the bounds of the dead view: the map is then framed
    # on the union of every plate it drew, and Reset restores that (#297). Only
    # /ui-check shows it (pytest has no browser): its `after-density` state.
    assert view.tabs.dynamic is False
    assert view.tabs.active == 0


def test_both_figures_stay_in_the_model_tree_whichever_tab_is_shown(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    root = view.layout.get_root()
    # A figure that leaves the model tree has its view destroyed in the browser, and
    # its ranges keep the bounds of the dead one (FM-174): what a dynamic tab does to
    # the figure of the tab not shown. The parameter pinned above is how it is avoided.
    for active in (1, 0, 1):
        view.tabs.active = active
        in_tree = {model.id for model in root.references()}
        assert view.figure.id in in_tree
        assert view.image_figure.id in in_tree


def test_a_tool_that_ends_between_two_ticks_of_another_tab_is_drawn_there_too(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    quick = EntryPoint(name="quick", value=f"{__name__}:QuickTool", group=GROUP)
    controller = PlatePage(fake_microscope, run_doc, entry_points=[quick])
    try:
        first = PlatePageView(controller)
        second = PlatePageView(controller)
        second.tick()
        assert second.tool_cards["quick"].object == ""
        first.on_run("quick")
        assert controller.join(10.0)
        # The run was over before `second` looked: it was never seen running.
        second.tick()
        first.tick()
        assert first.tool_cards["quick"].object.startswith("done ")
        assert second.tool_cards["quick"].object == first.tool_cards["quick"].object
    finally:
        controller.close()


def test_the_axes_are_the_frames_own_pixels_at_any_stride(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # 2048 rows by 512 columns is shown as 1024 x 256 (a stride of 2): the extent
    # is the camera's 512 x 2048, or the image would be drawn half its size (FM-136).
    fake_microscope.override(Camera, FrameCamera(_rows_columns(2048, 512)))
    view = PlatePageView(page)
    _snap_and_draw(view, page)
    drawn = view.image_source.data
    assert drawn["image"][0].shape == (1024, 256)
    assert list(drawn["dw"]) == [512]
    assert list(drawn["dh"]) == [2048]


def test_a_tick_without_a_new_frame_draws_nothing(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    _snap_and_draw(view, page)
    before = view.image_source.data
    view.tick()
    view.tick()
    assert view.image_source.data is before
    # A new frame is a new draw.
    _snap_and_draw(view, page)
    assert view.image_source.data is not before


def test_a_frame_snapped_in_another_tab_is_drawn_by_the_next_tick(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    first = PlatePageView(page)
    second = PlatePageView(page)
    _snap_and_draw(first, page)
    assert list(second.image_source.data["dw"]) == []
    second.tick()
    assert list(second.image_source.data["dw"]) == [10]
    assert list(second.image_source.data["dh"]) == [6]
    # Only the tab that clicked is taken to the image.
    assert second.tabs.active == 0
    # A tab opened after the snap draws it at once.
    third = PlatePageView(page)
    assert list(third.image_source.data["dw"]) == [10]


def test_snap_is_disabled_with_the_reason_on_a_stand_without_a_camera(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    assert not view.snap_button.disabled
    assert view.snap_note.object == ""
    real_has = fake_microscope.has

    def has(capability: type) -> bool:
        return capability is not Camera and real_has(capability)

    with _replaced(fake_microscope, "has", has):
        view = PlatePageView(page)
        assert view.snap_button.disabled
        assert view.snap_note.object == "this stand has no Camera"


def test_snap_and_every_run_button_are_disabled_while_an_activity_goes(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    camera = FrameCamera(_rows_columns(6, 10), block=True)
    fake_microscope.override(Camera, camera)
    view = PlatePageView(page)
    try:
        view.snap_button.clicks += 1
        assert camera.entered.wait(5)
        view.tick()
        assert view.snap_button.disabled
        assert view.tool_runs
        assert all(button.disabled for button in view.tool_runs.values())
        assert view.run_text.object == "snap is running"
    finally:
        camera.release.set()
    assert page.join(10.0)
    view.tick()
    assert not view.snap_button.disabled
    assert not view.tool_runs["scan"].disabled
    # And the other way round: a run in flight disables Snap.
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.tick()
    assert view.snap_button.disabled
    view.on_snap()
    assert view.message.object == "scan is running"


def test_a_refused_snap_is_the_pages_message_and_not_an_exception(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.on_snap()
    assert view.message.object == "scan is running"
    assert page.frame() is None


def _wait_for_frame_number(
    page: PlatePage, number: int, timeout_s: float = 5.0
) -> None:
    """Poll the controller until it holds frame ``number`` or later; bounded (FM-44)."""
    deadline = time.monotonic() + timeout_s
    while True:
        frame = page.frame()
        if frame is not None and frame.number >= number:
            return
        assert time.monotonic() < deadline, (
            f"frame {number} not seen within {timeout_s} s: {page.status().outcome}"
        )
        time.sleep(0.005)


def test_live_button_starts_and_ends_live_and_says_which(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    assert (view.live_button.label, view.live_button.disabled) == ("Live", False)
    assert view.tabs.active == 0
    view.live_button.clicks += 1
    assert page.is_live
    assert view.live_button.label == "End live"
    assert view.tabs.active == 1
    view.live_button.clicks += 1
    assert page.join(10.0)
    view.tick()
    assert view.live_button.label == "Live"
    assert not page.is_live
    # Ending live is not STOP: the stand is not halted.
    assert not page.status().halted


def test_live_is_disabled_with_the_reason_on_a_stand_without_a_camera(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    real_has = fake_microscope.has

    def has(capability: type) -> bool:
        return capability is not Camera and real_has(capability)

    with _replaced(fake_microscope, "has", has):
        view = PlatePageView(page)
        assert view.live_button.disabled
        assert view.live_button.label == "Live"
        assert view.snap_note.object == "this stand has no Camera"


def test_live_stays_enabled_while_it_goes_and_is_disabled_while_another_activity_goes(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    view.on_live()
    try:
        assert view.live_button.label == "End live"
        assert not view.live_button.disabled
        # Snap and every Run button are disabled as they are for any activity.
        assert view.snap_button.disabled
        assert view.tool_runs
        assert all(button.disabled for button in view.tool_runs.values())
    finally:
        view.on_live()
    assert page.join(10.0)
    view.tick()
    assert not view.live_button.disabled
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.tick()
    assert view.live_button.disabled
    assert view.live_button.label == "Live"
    # A click that cannot start says why, as a refused snap does.
    view.on_live()
    assert view.message.object == "scan is running"
    assert not page.is_live


def test_the_run_line_says_live_and_its_frames(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    view.on_live()
    try:
        _wait_for_frame_number(page, 3)
        view.tick()
        assert re.fullmatch(r"live: \d+ frames", view.run_text.object), (
            view.run_text.object
        )
    finally:
        view.on_live()
    assert page.join(10.0)


def test_frame_tick_draws_a_new_frame_and_reads_nothing_else(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    counts = {"view": 0, "status": 0}
    real_view, real_status = page.view, page.status

    def counting_view() -> Any:
        counts["view"] += 1
        return real_view()

    def counting_status() -> Any:
        counts["status"] += 1
        return real_status()

    with (
        _replaced(page, "view", counting_view),
        _replaced(page, "status", counting_status),
    ):
        page.snap()
        assert page.join(10.0)
        view.frame_tick()
        drawn = view.image_source.data
        assert list(drawn["dw"]) == [10]
        assert list(drawn["dh"]) == [6]
        # The same number: nothing is drawn, the very same data stays.
        view.frame_tick()
        view.frame_tick()
        assert view.image_source.data is drawn
        page.snap()
        assert page.join(10.0)
        view.frame_tick()
        assert view.image_source.data is not drawn
    assert counts == {"view": 0, "status": 0}


class _ReadSpy:
    """Stands for the controller and notes the name of every attribute the view reads."""

    def __init__(self, page: PlatePage) -> None:
        self._page = page
        self.read: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.read.append(name)
        return getattr(self._page, name)


def test_frame_tick_reads_nothing_but_the_frame_from_the_controller(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # FM-145: it runs ten times a second on the loop that handles STOP. Not the status,
    # not the view, and not the buttons either (`lacks`, `is_live`): `tick` has those.
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    page.snap()
    assert page.join(10.0)
    spy = _ReadSpy(page)
    view.page = cast("PlatePage", spy)
    view.frame_tick()  # a new frame: drawn
    view.frame_tick()  # the same one: nothing
    assert list(view.image_source.data["dw"]) == [10]
    assert spy.read == ["frame", "frame"]


def test_a_closed_tab_does_not_end_live(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # FM-146: the controller is the server's, and live is its activity, not the tab's.
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    first = PlatePageView(page)
    first.on_live()
    PlatePageView(page)  # a second tab, built and dropped
    del first  # and the first one closed
    gc.collect()
    frame = page.frame()
    seen = 0 if frame is None else frame.number
    _wait_for_frame_number(page, seen + 2)
    assert page.is_live
    assert not page.status().halted
    page.live(False)
    assert page.join(10.0)


class _FaultableCamera(FrameCamera):
    """A camera that counts its snaps and fails each one once ``fail`` is set."""

    def __init__(self) -> None:
        super().__init__(_rows_columns(6, 10))
        self.snaps = 0
        self.fail = threading.Event()

    def snap(self) -> np.ndarray:
        self.snaps += 1
        if self.fail.is_set():
            raise RuntimeError("the camera timed out")
        return super().snap()


def test_a_click_on_a_stale_end_live_ends_nothing_and_starts_no_live(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # FM-177: the click acts on the label it shows, and no tick has redrawn it since
    # the fault: this test never ticks between the fault and the click.
    camera = _FaultableCamera()
    fake_microscope.override(Camera, camera)
    view = PlatePageView(page)
    view.on_live()
    assert view.live_button.label == "End live"
    _wait_for_frame_number(page, 2)
    camera.fail.set()
    assert page.join(10.0)
    assert not page.is_live
    # No tick since: the label is the stale one this test is about.
    assert view.live_button.label == "End live"
    snaps, outcome = camera.snaps, page.status().outcome
    assert outcome is not None
    assert outcome.startswith("live ended after ")
    view.on_live()
    assert page.join(10.0)
    assert camera.snaps == snaps
    assert not page.is_live
    assert page.status().outcome == outcome
    assert view.message.object == ""
    assert view.live_button.label == "Live"


def test_a_click_on_a_stale_live_is_refused_and_does_not_end_the_other_tabs_live(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    first, second = PlatePageView(page), PlatePageView(page)
    first.on_live()
    try:
        assert page.is_live
        assert second.live_button.label == "Live"
        second.on_live()
        assert second.message.object == "live is running"
        frame = page.frame()
        seen = 0 if frame is None else frame.number
        _wait_for_frame_number(page, seen + 2)
        assert page.is_live
        assert second.live_button.label == "End live"
    finally:
        page.live(False)
    assert page.join(10.0)


def _card(**changes: Any) -> ResultCard:
    base: dict[str, Any] = {
        "plugin": "scan",
        "status": "done",
        "when_utc": "2026-10-29 09:03:00",
        "profile": "demo",
        "dry_run": False,
        "lines": ("n: 3", "text: hello"),
        "assumed": ("pixel size from the profile",),
        "error": None,
    }
    return ResultCard(**{**base, **changes})


def test_card_text_says_status_time_profile_and_dry_run() -> None:
    assert card_text(None) == ""
    assert card_text(None, stand="demo") == ""
    assert card_text(_card(), stand="demo") == (
        "done 2026-10-29 09:03:00 UTC\n"
        "  n: 3\n"
        "  text: hello\n"
        "  in the run document\n"
        "assumed: pixel size from the profile"
    )
    assert card_text(
        _card(profile="lightsheet", dry_run=True, assumed=("a", "b")), stand="demo"
    ) == (
        "done 2026-10-29 09:03:00 UTC on lightsheet (dry run)\n"
        "  n: 3\n"
        "  text: hello\n"
        "  in the run document\n"
        "assumed: a\n"
        "assumed: b"
    )
    assert card_text(
        _card(status="failed", lines=(), assumed=(), error="the camera failed"),
        stand="demo",
    ) == ("failed 2026-10-29 09:03:00 UTC\nerror: the camera failed")
    # A stand that is not known: the card never claims to be this stand's.
    assert card_text(_card(lines=(), assumed=())) == (
        "done 2026-10-29 09:03:00 UTC on demo\n  in the run document"
    )


def test_card_text_ends_a_stale_card_with_stale_after_dry_run() -> None:
    assert card_text(
        _card(dry_run=True, stale=True, lines=(), assumed=()), stand="demo"
    ) == ("done 2026-10-29 09:03:00 UTC (dry run) (stale)\n  in the run document")
    assert card_text(_card(stale=True, lines=(), assumed=()), stand="demo") == (
        "done 2026-10-29 09:03:00 UTC (stale)\n  in the run document"
    )
    assert card_text(
        _card(profile="lightsheet", stale=True, lines=(), assumed=()), stand="demo"
    ) == ("done 2026-10-29 09:03:00 UTC on lightsheet (stale)\n  in the run document")


def test_card_text_shows_files_then_says_in_the_run_document_without_one() -> None:
    # The files follow the values and come before the assumptions; each is
    # indented as a value is (m5 §8.4).
    assert card_text(
        _card(files=("file: /runs/a/report/t.html", "folder: /runs/a/scan/t")),
        stand="demo",
    ) == (
        "done 2026-10-29 09:03:00 UTC\n"
        "  n: 3\n"
        "  text: hello\n"
        "  file: /runs/a/report/t.html\n"
        "  folder: /runs/a/scan/t\n"
        "assumed: pixel size from the profile"
    )
    # Every done card says where its result is: its files, else the document.
    assert card_text(_card(), stand="demo").splitlines()[3] == "  in the run document"
    # A card that did not end done has no result to point at, and says nothing of one.
    for status in ("failed", "stopped", "running"):
        text = card_text(
            _card(status=status, lines=(), assumed=(), error="x"), stand="demo"
        )
        assert "in the run document" not in text
        assert "file" not in text


def _seed_raw(
    run_doc: Path,
    plugin: str,
    result: dict[str, Any],
    *,
    profile: str = "demo",
    status: StageStatus = StageStatus.done,
    error: str | None = None,
) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=plugin,
            attempt="a0",
            status=status,
            started_utc=T0,
            finished_utc=T0 + timedelta(minutes=3),
            profile=profile,
            dry_run=False,
            smc_version="0",
            params={},
            result=result,
            error=error,
        ),
    )


def test_the_card_text_of_a_worst_case_record_is_bounded(
    page: PlatePage, run_doc: Path
) -> None:
    # 2000 assumptions of 700 characters, 2000 keys, an error of 100,000 and a profile
    # of 5,000: a card of 620,000 characters before the bounds (m5 §2.4).
    result: dict[str, Any] = {"assumed": ["a" * 700] * 2000}
    result.update({f"key{i}": i for i in range(2000)})
    _seed_raw(run_doc, "scan", result, profile="p" * 5000, error="e" * 100_000)
    text = PlatePageView(page).tool_cards["scan"].object
    assert len(text) < 7_500
    assert text.splitlines()[-1] == f"error: {'e' * 600}…"
    assert "assumed: … 1992 more" in text.splitlines()


def test_a_result_card_is_plain_text_under_its_tool(
    page: PlatePage, run_doc: Path
) -> None:
    # Markup in a result must reach the screen as the words it is: a Str pane, not
    # Markdown (FM-139).
    _seed_raw(
        run_doc,
        "scan",
        {"assumed": ["<i>[bold]guess[/bold]</i>"], "note": "<b>[red]x</b>"},
    )
    view = PlatePageView(page)
    pane = view.tool_cards["scan"]
    assert type(pane) is pn.pane.Str
    assert pane.object == (
        "done 2026-10-29 09:03:00 UTC\n"
        "  note: <b>[red]x</b>\n"
        "  in the run document\n"
        "assumed: <i>[bold]guess[/bold]</i>"
    )
    # Under its Run button (and the note that explains a disabled one), in its own
    # section; no other tool shows it.
    run = view.tool_runs["scan"]
    (accordion,) = view.layout.select(pn.Accordion)
    (section,) = [s for stage in accordion.objects for s in stage if run in s]
    assert list(section).index(pane) == list(section).index(run) + 2
    assert view.tool_cards["goto-well"].object == ""


def test_a_card_of_another_stand_names_it_and_one_of_this_stand_does_not(
    page: PlatePage, run_doc: Path
) -> None:
    # The first tick reads the stand's name; the cards the first refresh drew
    # without it are drawn again then.
    _seed_raw(run_doc, "scan", {"n": 1}, profile="lightsheet")
    _seed_raw(run_doc, "scan-plan", {"n": 2}, profile="demo")
    view = PlatePageView(page)
    assert view.tool_cards["scan"].object.splitlines()[0] == (
        "done 2026-10-29 09:03:00 UTC on lightsheet"
    )
    assert view.tool_cards["scan-plan"].object.splitlines()[0] == (
        "done 2026-10-29 09:03:00 UTC"
    )


def test_a_card_follows_the_run_documents_end_of_a_run(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.tool_cards["scan"].object == ""
    view.on_run("scan")
    assert SlowScan.started.wait(5)
    view.tick()
    SlowScan.release.set()
    assert page.join(10.0)
    view.tick()
    text = view.tool_cards["scan"].object
    assert text.startswith("done ")
    # The stub names no file: the card says where its result is all the same.
    assert text.splitlines()[0].endswith(" UTC")
    assert text.endswith("\n  in the run document")


def test_tools_under_more_are_built(page: PlatePage) -> None:
    view = PlatePageView(page)
    # `goto-well` is no stage of the workflow: a section like the others, in the
    # last card of the accordion, with its required field as a row of its own.
    assert "goto-well" in view.tool_runs
    assert "well" in view.tool_fields["goto-well"]
    (accordion,) = view.layout.select(pn.Accordion)
    assert len(accordion.objects) == 4
    last = [b for b in accordion.objects[-1].select(pn.widgets.Button)]
    assert [b for b in last if b is view.tool_runs["goto-well"]]
    for earlier in accordion.objects[:-1]:
        assert view.tool_runs["goto-well"] not in earlier.select(pn.widgets.Button)


def test_a_plugin_whose_parameters_cannot_be_described_leaves_the_page_building(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    ghost = EntryPoint(name="ghost", value="no_such_module_xyz:Plugin", group=GROUP)
    controller = PlatePage(
        fake_microscope,
        run_doc,
        entry_points=[ghost, *_entry_points("goto-well")],
    )
    try:
        view = PlatePageView(controller)
        assert "ghost" not in view.tool_runs
        assert view.problems_text.object == (
            "plugin ghost: could not be imported: ModuleNotFoundError: No module "
            "named 'no_such_module_xyz'"
        )
        assert view.layout.get_root() is not None
    finally:
        controller.close()


# --- the map ----------------------------------------------------------------------


def test_the_map_draws_wells_in_stage_axes_with_the_plates_rotation(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    wells = _wells_by_name(view)
    assert len(wells) == 96
    for name, drawn in wells.items():
        expected = _centre(name)
        assert (drawn["x"], drawn["y"]) == pytest.approx(expected, abs=1e-6)
        assert drawn["angle"] == pytest.approx(math.radians(TURN_DEG))
        assert (drawn["w"], drawn["h"]) == pytest.approx((6400.0, 6400.0))
        assert drawn["r"] == pytest.approx(3200.0)
    # Stage axes, Y up: row A is above row H.
    assert wells["A1"]["y"] > wells["H1"]["y"]
    assert wells["A12"]["x"] > wells["A1"]["x"]


def test_round_wells_are_circles_and_square_ones_rectangles(
    run_doc: Path, page: PlatePage
) -> None:
    view = PlatePageView(page)
    assert isinstance(view.wells_renderer.glyph, Circle)
    view.plate_select.value = "384-well"
    assert isinstance(view.wells_renderer.glyph, Rect)
    drawing = [r for r in view.figure.renderers if r.data_source is view.wells_source]
    assert len(drawing) == 1
    assert view.hover.renderers == [view.wells_renderer]
    view.plate_select.value = "96-well"
    assert isinstance(view.wells_renderer.glyph, Circle)
    assert (
        len([r for r in view.figure.renderers if r.data_source is view.wells_source])
        == 1
    )


def test_the_map_says_whether_the_plate_is_registered(
    run_doc: Path, page: PlatePage
) -> None:
    view = PlatePageView(page)
    assert view.figure.title.text == (
        "96-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )
    _seed(run_doc, "plate-registration", "plate", _plate())
    view.refresh()
    assert view.figure.title.text == "96-well: the registered plate"
    view.plate_select.value = "384-well"
    assert view.figure.title.text == (
        "384-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )


def test_names_are_drawn_only_up_to_96_wells(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.names.visible
    view.plate_select.value = "384-well"
    assert not view.names.visible
    view.plate_select.value = "24-well"
    assert view.names.visible


def test_a_tap_selects_the_well_under_it(run_doc: Path, page: PlatePage) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    _tap(view, *_centre("C5"))
    assert page.selection == ("C5",)
    wells = _wells_by_name(view)
    assert wells["C5"]["outline_width"] == 3
    assert {w["outline_width"] for n, w in wells.items() if n != "C5"} == {1}
    assert wells["C5"]["outline"] != wells["C4"]["outline"]
    _tap(view, *_centre("C5"))
    assert page.selection == ()
    assert {w["outline_width"] for w in _wells_by_name(view).values()} == {1}


def test_a_tap_on_no_well_does_nothing(page: PlatePage) -> None:
    view = PlatePageView(page)
    _tap(view, -50000.0, 50000.0)
    assert page.selection == ()
    assert view.message.object == ""


def test_a_typed_range_selects_and_draws(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.select_input.value = "A1:A3"
    assert page.selection == ("A1", "A2", "A3")
    wells = _wells_by_name(view)
    assert [n for n, w in wells.items() if w["outline_width"] == 3] == [
        "A1",
        "A2",
        "A3",
    ]


# --- the handlers and the document lock (#277, FM-161) ----------------------------------
#
# A handler that raised in a real session when Enter was pressed in the Wells box passed
# every test above: they set the widget on a view that no session owns, where nothing
# holds a lock. These tests cannot open a session (a `pull_session` client does not
# reproduce the interleaving, #277); they prove the pieces that make a session work, and
# `/ui-check` looks at the page in a browser.


def _type_wells(view: PlatePageView) -> None:
    view.select_input.value = "A1:A3"


def _click_refresh(view: PlatePageView) -> None:
    view.refresh_button.clicks += 1


def _tap_nowhere(view: PlatePageView) -> None:
    _tap(view, 0.0, 0.0)


_HANDLER_TRIGGERS = pytest.mark.parametrize(
    "trigger",
    [_type_wells, _click_refresh, _tap_nowhere],
    ids=["typed-range", "refresh-button", "tap"],
)


@_HANDLER_TRIGGERS
def test_a_handler_runs_inside_panels_unlocked_context(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch, trigger: Callable[..., None]
) -> None:
    view = PlatePageView(page)
    active = {"now": False}
    seen: list[bool] = []

    @contextmanager
    def unlocked() -> Iterator[None]:
        active["now"] = True
        try:
            yield
        finally:
            active["now"] = False

    def redraw() -> None:
        seen.append(active["now"])

    monkeypatch.setattr(pn.io, "unlocked", unlocked)
    with _replaced(view, "refresh", redraw):
        trigger(view)
    assert seen
    assert all(seen)


@_HANDLER_TRIGGERS
def test_an_unexpected_failure_in_a_handler_is_a_line_of_the_page(
    page: PlatePage,
    handler_failures: list[logging.LogRecord],
    trigger: Callable[..., None],
) -> None:
    view = PlatePageView(page)

    def redraw() -> None:
        raise KeyError("boom")

    with _replaced(view, "refresh", redraw):
        trigger(view)
    assert view.message.object.startswith("that action failed on the page: KeyError: ")
    assert "boom" in view.message.object
    assert len(handler_failures) == 1
    assert handler_failures[0].levelno == logging.ERROR
    assert handler_failures[0].name.startswith("smc.ui")
    assert handler_failures[0].exc_info is not None
    handler_failures.clear()  # expected here: the fixture fails a test that leaves one


@pytest.mark.parametrize(
    ("size", "cut"), [(10_000, True), (301, True), (300, False), (299, False)]
)
def test_the_failure_line_of_a_handler_is_bounded(
    page: PlatePage,
    handler_failures: list[logging.LogRecord],
    size: int,
    cut: bool,
) -> None:
    view = PlatePageView(page)
    text = "x" * size

    def redraw() -> None:
        raise OSError(text)

    with _replaced(view, "refresh", redraw):
        view.refresh_button.clicks += 1
    assert len(handler_failures) == 1
    handler_failures.clear()
    prefix = "that action failed on the page: OSError: "
    suffix = " (see the server's log)"
    line = view.message.object
    assert len(line) <= 400
    assert line.startswith(prefix)
    assert line.endswith(suffix)
    body = line[len(prefix) : -len(suffix)]
    assert body == ("x" * 300 + "…" if cut else text)


def test_a_keyboard_interrupt_in_a_handler_is_not_swallowed(page: PlatePage) -> None:
    view = PlatePageView(page)

    def interrupted() -> None:
        raise KeyboardInterrupt

    with _replaced(view, "refresh", interrupted), pytest.raises(KeyboardInterrupt):
        view.refresh_button.clicks += 1
    assert view.message.object == ""


def _page_widgets(view: PlatePageView) -> list[Any]:
    """Every widget of the view: the layout's, the tool column's too, and the view's attributes.

    ``Viewable.select`` does not go into the panels of an ``Accordion`` (measured: 26
    widgets, and none of the four Run buttons, Capture centre, Capture wall or Forget),
    so each panel is walked, and the widgets the view holds as attributes (a widget or a
    dict of them) are added: a container this walk does not know cannot hide one.
    """
    found: dict[int, Any] = {id(w): w for w in view.layout.select(pn.widgets.Widget)}
    for accordion in view.layout.select(pn.Accordion):
        for panel in accordion.objects:
            found.update({id(w): w for w in panel.select(pn.widgets.Widget)})
    for value in vars(view).values():
        for item in value.values() if isinstance(value, dict) else [value]:
            if isinstance(item, pn.widgets.Widget):
                found[id(item)] = item
    return list(found.values())


def _placed(view: PlatePageView) -> set[int]:
    """The id of every object the layout holds, panes included, the Accordion's panels too.

    ``_page_widgets`` adds what the view holds as attributes, so a widget deleted from
    the layout is still found there: it cannot tell a widget the operator sees from one
    only a test reads. This walk goes through the layout alone.
    """
    found = {id(o) for o in view.layout.select()}
    for accordion in view.layout.select(pn.Accordion):
        for panel in accordion.objects:
            found.update(id(o) for o in panel.select())
    return found


def test_the_widgets_that_ask_and_explain_are_on_the_page(page: PlatePage) -> None:
    # The tests of the plate's ask and of the captures drive these widgets through
    # attributes (`clicks += 1`, `.visible`, `.object`), which a layout that does not
    # hold them answers all the same: the operator would never see the question, the
    # button or the note (adversarial review of #332).
    view = PlatePageView(page)
    placed = _placed(view)
    for name in (
        "plate_note",
        "plate_confirm_button",
        "capture_note",
        # What each side of the image is, and what an arrow does to it (m5 §13.2): an
        # operator who never sees them names a wall on the image and gets the stage's.
        "image_note",
        "drive_image_note",
        "capture_centre_button",
        "capture_wall_button",
        "capture_side_select",
        "forget_button",
    ):
        assert id(getattr(view, name)) in placed, f"{name} is not in the layout"


def test_every_handler_the_page_registers_is_guarded(page: PlatePage) -> None:
    view = PlatePageView(page)
    registered: list[Any] = [
        watcher.fn
        for widget in _page_widgets(view)
        for by_name in widget.param.watchers.values()
        for watchers in by_name.values()
        for watcher in watchers
    ]
    registered += view.figure._event_callbacks["tap"]
    # Panel registers watchers of its own on the same widgets: ours are the functions
    # of this module, which `functools.wraps` leaves a `__wrapped__` on.
    ours = [
        fn for fn in registered if getattr(fn, "__module__", "") == plate_page.__name__
    ]
    # 30 widgets' handlers and the tap at the time of writing: a floor that a lost
    # container (the Accordion's 7) would go under.
    assert len(ours) >= 30, "the test found too few handlers to check"
    unguarded = [
        getattr(fn, "__qualname__", repr(fn))
        for fn in ours
        if not hasattr(fn, "__wrapped__")
    ]
    assert unguarded == []


def test_a_handler_failure_is_kept_nowhere(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = PlatePageView(page)
    refs: list[weakref.ref[BaseException]] = []

    def redraw() -> None:
        raise _track(_RefusedError("boom"), refs)

    # A log record keeps its `exc_info`, and pytest keeps the records: what is asked
    # here is what the guard itself keeps (FM-98).
    monkeypatch.setattr(plate_page.log, "exception", lambda *args, **kwargs: None)
    gc.disable()
    try:
        with _replaced(view, "refresh", redraw):
            view.refresh_button.clicks += 1
        assert view.message.object.startswith(
            "that action failed on the page: _RefusedError: boom"
        )
        assert len(refs) == 1
        assert (
            refs[0]() is None
        )  # dead without the cyclic collector: nothing holds it (FM-98)
    finally:
        gc.enable()


def test_a_tap_in_drive_mode_moves_once(
    run_doc: Path, page: PlatePage, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    stage = fake_microscope.require(XYStage)
    assert not view.drive_text.visible
    view.drive_toggle.value = True
    assert view.drive_text.visible
    assert view.drive_text.object == (
        "DRIVE: the next click on a well moves the stage there"
    )
    _tap(view, *_centre("C5"))
    assert page.join(10.0)
    assert stage.position_um().x_um == pytest.approx(_centre("C5")[0], abs=1e-6)
    assert stage.position_um().y_um == pytest.approx(_centre("C5")[1], abs=1e-6)
    assert not view.drive_toggle.value
    assert not view.drive_text.visible
    assert page.selection == ()
    moved = stage.position_um()
    attempt = RunDocument.load(run_doc).stages["goto-well"].attempt
    # The second tap selects: the toggle went back to select mode, so nothing moves.
    _tap(view, *_centre("A1"))
    assert page.selection == ("A1",)
    assert page.status().running is None
    assert stage.position_um() == moved
    assert RunDocument.load(run_doc).stages["goto-well"].attempt == attempt


def test_a_drive_tap_that_is_refused_says_why_and_disarms_the_toggle(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    view.drive_toggle.value = True
    # No plate is registered: `goto-well` itself refuses, and the refusal is the outcome.
    _tap(view, 0.0, 0.0)
    assert page.join(10.0)
    assert page.status().outcome == (
        "goto-well: UpstreamError: goto-well reads plate, but the run document has "
        "no plate; smc run --list shows which plugin provides it"
    )
    assert stage.position_um() == before
    assert not view.drive_toggle.value


def test_a_drive_tap_on_another_plates_map_is_refused_before_anything_moves(
    run_doc: Path, page: PlatePage, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate("384-well"))
    view = PlatePageView(page)
    page.choose_plate("96-well")
    view.refresh()
    stage = fake_microscope.require(XYStage)
    before = stage.position_um()
    view.drive_toggle.value = True
    _tap(view, 36000.0, -18000.0)
    assert view.message.object == (
        "the registered plate is a 384-well; calibrate the 96-well plate again, "
        "or choose 384-well"
    )
    assert page.status().running is None
    assert stage.position_um() == before
    assert not view.drive_toggle.value
    assert "goto-well" not in RunDocument.load(run_doc).stages


def test_density_colours_the_wells_and_unknown_is_grey(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    _seed(
        run_doc,
        "density",
        "density",
        DensityResult(
            labels=("live", "dead"),
            area_grid_um=10.0,
            wells=tuple(
                DensityWell(
                    well=well,
                    fields=1,
                    failed_fields=0,
                    area_mm2=2.0,
                    counts={"live": n, "dead": 1},
                    outside_well=0,
                    per_mm2={"live": n / 2.0, "dead": 0.5},
                )
                for well, n in (("A1", 2), ("A2", 8))
            ),
        ),
        minute=1,
    )
    view = PlatePageView(page)
    assert view.label_select.options == ["live", "dead"]
    assert view.label_select.value == "live"
    wells = _wells_by_name(view)
    assert (wells["A1"]["value"], wells["A2"]["value"]) == (1.0, 4.0)
    assert math.isnan(wells["B1"]["value"])
    assert view.mapper.low == 0.0
    assert view.mapper.high == 4.0
    assert view.mapper.nan_color == "#eeeeee"
    assert wells["A2"]["tip"] == "A2 — live: 8 (4.0/mm²), dead: 1 (0.5/mm²)"
    assert wells["B1"]["tip"] == "B1"
    view.label_select.value = "dead"
    wells = _wells_by_name(view)
    assert (wells["A1"]["value"], wells["A2"]["value"]) == (0.5, 0.5)
    assert view.mapper.high == 0.5


def test_a_selected_well_keeps_its_density_colour(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    page.select_text("A1")
    view.refresh()
    assert _wells_by_name(view)["A1"]["outline_width"] == 3


def test_the_route_is_drawn_in_order(run_doc: Path, page: PlatePage) -> None:
    from smc.plugins.slots import RouteResult

    _seed(
        run_doc,
        "route",
        "route",
        RouteResult.model_validate(
            {
                "objective": None,
                "matrix_um_per_px": [[1.0, 0.0], [0.0, 1.0]],
                "image_shape_px": [10, 10],
                "fov_um": [10.0, 10.0],
                "labels": [],
                "positions": [
                    {
                        "order": order,
                        "well": "A1",
                        "name": f"A1_{order}",
                        "x_um": x,
                        "y_um": y,
                        "z_um": None,
                        "pfs_offset": None,
                        "counts": {},
                    }
                    for order, x, y in ((2, 30.0, 3.0), (0, 10.0, 1.0), (1, 20.0, 2.0))
                ],
                "uncovered": {},
                "distance_um": 20.0,
                "travel_s": None,
            }
        ),
    )
    view = PlatePageView(page)
    assert view.route_source.data["x"] == [10.0, 20.0, 30.0]
    assert view.route_source.data["y"] == [1.0, 2.0, 3.0]
    assert view.route_source.data["order"] == [0, 1, 2]


# --- the scan plan's fields on the map (m5 §14, #320) ---------------------------------


class PlanParams(PluginParams):
    wells: str = ""


class PlanTool:
    """Registered as ``scan-plan``: ends at once with a plan, so a second tab can see its end."""

    name = provides = "scan-plan"
    summary = "A test scan plan."
    Params = PlanParams
    Result = ScanPlanResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self, microscope: Any, params: PlanParams, ctx: RunContext
    ) -> ScanPlanResult:
        return _scan_plan_result()


def _scan_plan_result(
    wells: tuple[str, ...] = ("B11", "B2", "A1"),
    *,
    fov_um: tuple[float, float] = (512.0, 512.0),
) -> ScanPlanResult:
    """A scan plan of 2 x 2 fields in each of ``wells`` (visiting order, not alphabetical), as test_report.py builds one."""
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    half_w, half_h = fov_um[0] / 2, fov_um[1] / 2
    plate_plan, fields = plan_fields(
        plate,
        list(wells),
        fov_um=fov_um,
        footprint_of=lambda x, y: [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ],
        rows=2,
        columns=2,
        overlap=0.1,
        clip_to_well=True,
    )
    return ScanPlanResult(
        plate_plan=plate_plan,
        wells=wells,
        objective="Nikon 10X S Fluor",
        pixel_size_um=1.0,
        matrix_um_per_px=((1.0, 0.0), (0.0, -1.0)),
        matrix_source="parameter",
        image_shape_px=(512, 512),
        fov_um=fov_um,
        overlap=0.1,
        rows=2,
        columns=2,
        fields=fields,
        estimate_s=None,
    )


def test_the_map_draws_a_rectangle_for_each_planned_field_and_the_title_says_so(
    run_doc: Path, page: PlatePage
) -> None:
    result = _scan_plan_result(fov_um=(600.0, 400.0))
    assert len(result.fields) == 12
    _seed(run_doc, "scan-plan", "scan-plan", result)
    view = PlatePageView(page)
    data = view.fields_source.data
    # Width and height are the field of view's own two sides, in the plan's order.
    assert data["x"] == [f.x_um for f in result.fields]
    assert data["y"] == [f.y_um for f in result.fields]
    assert data["w"] == [600.0] * 12
    assert data["h"] == [400.0] * 12
    assert view.figure.title.text == (
        "96-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
        "; scan plan: 12 fields in 3 wells"
    )


def test_the_title_of_a_registered_plate_gets_the_same_suffix(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    _seed(run_doc, "scan-plan", "scan-plan", _scan_plan_result(), minute=1)
    view = PlatePageView(page)
    assert view.figure.title.text == (
        "96-well: the registered plate; scan plan: 12 fields in 3 wells"
    )
    # A page with no plan says nothing of one.
    run_doc.unlink()
    view.refresh()
    assert (
        view.figure.title.text
        == "96-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )


def test_a_plan_over_the_bound_draws_no_field_and_the_title_says_why(
    run_doc: Path, page: PlatePage
) -> None:
    base = _scan_plan_result()
    big = base.model_copy(
        update={"fields": tuple(base.fields[i % 12] for i in range(5001))}
    )
    _seed(run_doc, "scan-plan", "scan-plan", big)
    view = PlatePageView(page)
    assert view.fields_source.data == {"x": [], "y": [], "w": [], "h": []}
    assert view.figure.title.text.endswith(
        "; scan plan: 5001 fields in 3 wells, too many to draw (over 5000)"
    )


@pytest.mark.parametrize("how", ["stale", "missing"])
def test_a_stale_or_missing_plan_clears_the_fields_on_the_next_refresh(
    run_doc: Path, page: PlatePage, how: str
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    _seed(
        run_doc,
        "scan-plan",
        "scan-plan",
        _scan_plan_result(),
        minute=1,
        reads={"plate": ReadRecord(plugin="plate-registration", attempt="a0")},
    )
    view = PlatePageView(page)
    assert len(view.fields_source.data["x"]) == 12
    if how == "stale":
        # The plate was registered again: the plan read an attempt that is gone.
        _seed(run_doc, "plate-registration", "plate", _plate(), minute=2)
    else:
        run_doc.unlink()
    view.refresh()
    assert view.fields_source.data == {"x": [], "y": [], "w": [], "h": []}
    assert "scan plan" not in view.figure.title.text


def test_the_fields_are_drawn_under_the_route_and_the_wells_keep_the_hover(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)

    def order() -> tuple[int, int]:
        renderers = view.figure.renderers
        fields = [r for r in renderers if r.data_source is view.fields_source]
        lines = [
            r
            for r in renderers
            if r.data_source is view.route_source and isinstance(r.glyph, Line)
        ]
        assert len(fields) == len(lines) == 1
        assert isinstance(fields[0].glyph, Rect)
        assert fields[0] not in view.hover.renderers
        return renderers.index(fields[0]), renderers.index(lines[0])

    # The route is drawn over the fields: later in the list is on top at one level.
    fields_at, route_at = order()
    assert fields_at < route_at
    assert view.hover.renderers == [view.wells_renderer]
    assert view.wells_renderer.level == "underlay"
    # A plate with square wells swaps the wells' renderer, and the fields stay where they were.
    view.plate_select.value = "384-well"
    fields_at, route_at = order()
    assert fields_at < route_at
    assert view.hover.renderers == [view.wells_renderer]


def test_a_second_tab_draws_the_plan_the_first_ran(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-151: the plan was made through the first tab, and the second draws it at its tick.
    planner = EntryPoint(name="scan-plan", value=f"{__name__}:PlanTool", group=GROUP)
    controller = PlatePage(fake_microscope, run_doc, entry_points=[planner])
    try:
        first, second = PlatePageView(controller), PlatePageView(controller)
        controller.select_text("B11,B2,A1")
        assert second.fields_source.data["x"] == []
        first.on_run("scan-plan")
        assert controller.join(10.0)
        first.tick()
        second.tick()
        expected = _scan_plan_result()
        assert len(second.fields_source.data["x"]) == 12
        assert second.fields_source.data["x"] == [f.x_um for f in expected.fields]
        assert second.fields_source.data == first.fields_source.data
        assert second.figure.title.text.endswith("; scan plan: 12 fields in 3 wells")
    finally:
        controller.close()


def test_the_stage_is_a_cross_at_its_position(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    fake_microscope.require(XYStage).move_to_um(1234.0, -567.0)
    view = PlatePageView(page)
    assert view.stage_source.data["x"] == [1234.0]
    assert view.stage_source.data["y"] == [-567.0]


# --- the strip, STOP and the tools ---------------------------------------------------


def test_a_position_that_rounds_to_zero_is_printed_without_a_sign() -> None:
    # The demo stage reads back -0.0 at its origin; the CLI prints it as 0.00 too.
    line = plate_page.status_line(_status(xy_um=(-0.0, -0.04), z_um=-0.0))
    assert line.startswith("demo | XY (0.0, 0.0) µm | Z 0.00 µm | ")
    line = plate_page.status_line(_status(xy_um=(-0.06, 0.0), z_um=-0.001))
    assert line.startswith("demo | XY (-0.1, 0.0) µm | Z 0.00 µm | ")


def test_the_strip_starts_with_the_stand() -> None:
    assert plate_page.status_line(_status(stand="ti2e-bench")).startswith(
        "ti2e-bench | XY ("
    )


@pytest.mark.parametrize(
    ("dry_run", "line"),
    [(False, "demo | not connected"), (True, "demo | not connected | DRY RUN")],
)
def test_status_line_of_a_closed_stand_says_not_connected_and_nothing_else(
    dry_run: bool, line: str
) -> None:
    # Whatever else the status holds: a closed stand has no position, no halt and
    # no move to report, and STOP's refusal is said once the stand is held.
    closed = _status(
        connected=False,
        dry_run=dry_run,
        xy_um=None,
        z_um=None,
        objective=None,
        autofocus="?",
        moving=("xy_stage",),
        halted=True,
        stop_refused=("xy_stage",),
    )
    assert plate_page.status_line(closed) == line


def test_the_strip_of_a_session_starts_with_the_stand(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.status_text.object.startswith("demo | XY (")


def test_the_stand_is_one_line_and_unknown_is_a_question_mark() -> None:
    assert plate_page.status_line(_status()) == (
        "demo | XY (1000.0, -500.0) µm | Z 12.50 µm | Nikon 10X S Fluor | "
        "autofocus off | light ?"
    )
    assert plate_page.status_line(
        _status(
            xy_um=None,
            z_um=None,
            objective=None,
            light_on=True,
            dry_run=True,
            moving=("xy_stage",),
            halted=True,
            stop_refused=("xy_stage", "focus"),
        )
    ) == (
        "demo | XY ? | Z ? | objective ? | autofocus off | light on | DRY RUN | "
        "moving: xy_stage | HALTED | STOP is refused by xy_stage, focus on this "
        "stand: a move under way runs to its target"
    )


def test_the_run_is_its_progress_its_outcome_and_the_last_stop() -> None:
    from smc.plugins.api import Progress

    assert plate_page.run_text(_status()) == ""
    assert plate_page.run_text(
        _status(
            running="scan",
            progress=Progress("scan", 3, 12, "well B2"),
            outcome="cells done in 2.0 s",
            stop_note="scan told to stop; stopped: the stand is halted (Resume clears the halt)",
        )
    ) == (
        "scan: 3/12 well B2\n"
        "cells done in 2.0 s\n"
        "scan told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert plate_page.run_text(_status(running="scan")) == "scan is running"
    assert (
        plate_page.run_text(
            _status(running="scan", progress=Progress("scan", 4, None, ""))
        )
        == "scan: 4"
    )


def test_stop_is_always_offered_and_resume_only_when_halted(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert not view.stop_button.disabled
    assert not view.resume_button.visible
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.tick()
    assert not view.stop_button.disabled
    view.on_stop()
    assert page.join(10.0)
    view.tick()
    assert view.resume_button.visible
    assert not view.stop_button.disabled
    assert view.run_text.object == (
        "scan stopped\n"
        "scan told to stop; stopped: the stand is halted (Resume clears the halt)"
    )
    assert "HALTED" in view.status_text.object
    view.on_resume()
    assert not view.resume_button.visible
    assert view.run_text.object == "scan stopped"


def test_stop_button_stops_through_a_click(page: PlatePage) -> None:
    view = PlatePageView(page)
    view.stop_button.clicks += 1
    assert page.status().halted
    assert view.resume_button.visible
    view.resume_button.clicks += 1
    assert not page.status().halted
    assert not view.resume_button.visible


def test_a_blocked_tool_is_disabled_with_its_reason(page: PlatePage) -> None:
    page.choose_plate("96-well")
    view = PlatePageView(page)
    reason = (
        "the run document has no plate; smc run --list shows which plugin provides it"
    )
    assert view.tool_runs["scan-plan"].disabled
    assert view.tool_notes["scan-plan"].object == reason
    assert not view.tool_runs["plate-registration"].disabled
    assert view.tool_notes["plate-registration"].object == ""
    assert not view.tool_runs["scan"].disabled


def test_the_run_button_of_a_stopped_stage_is_disabled_with_the_resume_command(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # #252: a Run after STOP would start the scan afresh and drop its checkpoint.
    scan = EntryPoint(name="scan", value=f"{__name__}:CheckpointingScan", group=GROUP)
    page = PlatePage(fake_microscope, run_doc, entry_points=[scan])
    try:
        view = PlatePageView(page)
        assert not view.tool_runs["scan"].disabled
        assert view.tool_notes["scan"].object == ""
        view.on_run("scan")
        assert SlowScan.started.wait(5)
        view.on_stop()
        assert page.join(10.0)
        # The end of the run redraws the view: the button is disabled by that tick,
        # not only after a Refresh.
        view.tick()
        assert view.tool_runs["scan"].disabled
        assert view.tool_notes["scan"].object == (
            "stopped with a checkpoint: continue it with `smc run scan --resume` "
            "(close this page first); `smc run scan` without --resume starts afresh "
            "and drops it"
        )
        # A click that gets through anyway (a stale tab) is refused by the controller
        # and the record is what the stop left.
        stopped = run_doc.read_bytes()
        view.on_run("scan")
        assert view.run_text.object.startswith(
            "scan: stopped with a checkpoint: continue it with `smc run scan --resume`"
        )
        assert run_doc.read_bytes() == stopped
    finally:
        page.close()


def test_a_tools_note_wraps_where_the_text_is_the_command_that_continues(
    page: PlatePage,
) -> None:
    # Measured in Chromium (#252, /ui-check): a `white-space` in `styles` lands on the
    # pane's host while the `<pre>` inside its shadow root keeps `pre`, so a reason
    # of 190 characters ran past the column and ended at `smc run sca`. The rule is
    # on the `<pre>`; a test cannot see a rendering, only that it is still there.
    view = PlatePageView(page)
    for note in view.tool_notes.values():
        assert note.stylesheets == ["pre { white-space: pre-wrap; }"]
        assert "white-space" not in note.styles


def test_a_running_tool_disables_every_run_button_until_it_ends(
    page: PlatePage,
) -> None:
    page.choose_plate("96-well")
    view = PlatePageView(page)
    view.on_run("scan")
    assert SlowScan.started.wait(5)
    assert all(button.disabled for button in view.tool_runs.values())
    assert view.run_text.object == "scan is running"
    SlowScan.release.set()
    assert page.join(10.0)
    view.tick()
    # The end of a run redraws the view: the run buttons are the view's blocked set again.
    assert not view.tool_runs["plate-registration"].disabled
    assert not view.tool_runs["scan"].disabled
    assert view.tool_runs["scan-plan"].disabled
    assert view.run_text.object.startswith("scan done in ")


def test_a_run_that_ends_redraws_the_view(run_doc: Path, page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.label_select.options == []
    page.run("scan", [])
    assert SlowScan.started.wait(5)
    view.tick()
    _seed(
        run_doc,
        "density",
        "density",
        DensityResult(labels=("live",), area_grid_um=10.0, wells=()),
        minute=1,
    )
    assert (
        view.label_select.options == []
    )  # not read yet: the tick never reads the view
    SlowScan.release.set()
    assert page.join(10.0)
    view.tick()
    assert view.label_select.options == ["live"]


def test_an_empty_field_is_not_sent(page: PlatePage) -> None:
    view = PlatePageView(page)
    page.select_text("A1")
    sent: list[tuple[str, list[str]]] = []

    def record(plugin: str, params: list[str]) -> None:
        sent.append((plugin, list(params)))

    view.tool_fields["scan-plan"]["overlap"].value = "0.2"
    view.tool_fields["scan-plan"]["coverage"].value = "  "
    with _replaced(page, "run", record):
        view.on_run("scan-plan")
    assert sent == [("scan-plan", [*page.page_params("scan-plan"), "overlap=0.2"])]
    assert sent == [("scan-plan", ["wells=A1", "overlap=0.2"])]


def test_the_page_shows_a_message_and_keeps_no_exception(page: PlatePage) -> None:
    view = PlatePageView(page)
    # A run that is refused for the operator's mistake is a message, not an error.
    view.on_run("scan-plan")
    assert view.message.object == "select at least one well first"
    page.select_text("A1")
    view.on_select_text("Z99")
    assert view.message.object == "wells not understood on a 96-well plate: 'Z99'"
    assert page.selection == ("A1",)
    view.on_select_text("A2")
    assert view.message.object == ""
    assert page.selection == ("A2",)


def test_a_handler_keeps_no_error_it_showed(page: PlatePage) -> None:
    view = PlatePageView(page)
    refs: list[weakref.ref[BaseException]] = []

    def refuse(text: str) -> None:
        raise _track(_RefusedError("refused"), refs)

    gc.disable()
    try:
        with _replaced(page, "select_text", refuse):
            view.on_select_text("A1")
        assert view.message.object == "refused"
        assert len(refs) == 1
        assert (
            refs[0]() is None
        )  # dead without the cyclic collector: nothing holds it (FM-98)
    finally:
        gc.enable()


class _RefusedError(ValueError):
    """A subclass: a builtin exception cannot be weakly referenced, its subclass can."""


def _track(
    error: _RefusedError, refs: list[weakref.ref[BaseException]]
) -> _RefusedError:
    refs.append(weakref.ref(error))
    return error


def test_capture_buttons_act_on_the_current_well(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)  # the Ti2-E's: the image's top is stage -Y
    view = PlatePageView(page)
    view.on_capture("centre")
    assert view.message.object == "click a well on the map first"
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, 0.0, 0.0)
    assert page.current == "A1"
    view.capture_centre_button.clicks += 1
    view.capture_side_select.value = "top"
    view.capture_wall_button.clicks += 1
    assert [(r.well, r.x_um, r.y_um) for r in page.refs] == [("A1", 10.0, 20.0)]
    # The side is the stage's, mapped from the image's `top` with the calibration.
    assert [(w.well, w.side) for w in page.walls] == [("A1", "-Y")]
    assert (
        view.captures_text.object == "A1 centre (10.0, 20.0)\nA1 -Y wall (10.0, 20.0)"
    )
    view.forget_button.clicks += 1
    assert (page.refs, page.walls) == ((), ())
    assert view.captures_text.object == "no capture yet"
    view.on_capture(
        "nonsense"
    )  # not a button: the handler still refuses it, by message
    assert (
        view.message.object == "cannot capture 'nonsense'; capture a centre or a wall"
    )


# --- a disabled Run says why beside the button (m5 §11.2, #332) ---------------------

LIVE_NOTE = "Live is running: press End live, then run the tool."
SCAN_PLAN_REASON = (
    "the run document has no plate; smc run --list shows which plugin provides it"
)


def test_a_tool_run_greyed_in_live_says_so_beside_the_button(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    page.choose_plate("96-well")
    # A capture made, so that the next step has no reason of its own (it would be
    # stuck on `CAPTURE_WELLS`, which holds longer than any activity).
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    page.capture_centre("A1")
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view = PlatePageView(page)
    assert view.tool_notes["scan"].object == ""
    view.on_live()
    try:
        assert view.tool_runs["scan"].disabled
        assert view.tool_notes["scan"].object == LIVE_NOTE
        assert view.tool_notes["plate-registration"].object == LIVE_NOTE
        # The next step is plate-registration here: its note says the same.
        assert view.next_button.disabled
        assert view.next_note.object == LIVE_NOTE
        # A tool with a reason of its own keeps it: it holds longer than the activity
        # (both of these read the plate, which the run document does not have yet).
        assert view.tool_notes["scan-plan"].object == SCAN_PLAN_REASON
        assert view.tool_notes["goto-well"].object == SCAN_PLAN_REASON
    finally:
        view.on_live()
    assert page.join(10.0)
    view.tick()
    assert view.tool_notes["scan"].object == ""
    assert view.next_note.object == ""


def test_a_run_in_progress_says_what_it_waits_for(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    page.choose_plate("96-well")
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    page.capture_centre("A1")
    view = PlatePageView(page)
    view.on_run("scan")
    assert SlowScan.started.wait(5)
    view.tick()
    try:
        waiting = "scan is running: wait for it to end, or press STOP."
        assert view.tool_notes["plate-registration"].object == waiting
        assert view.tool_notes["scan"].object == waiting
        assert view.next_note.object == waiting
    finally:
        SlowScan.release.set()
        assert page.join(10.0)
    view.tick()
    assert view.tool_notes["plate-registration"].object == ""


def test_a_closed_stand_reason_wins_over_the_busy_note(
    run_doc: Path, fake_microscope: Microscope
) -> None:
    gate, entered = threading.Event(), threading.Event()
    stands: list[Microscope] = []

    def slow_opener() -> Microscope:
        entered.set()
        assert gate.wait(5)
        stand = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()), Profile.demo()
        )
        stands.append(stand)
        return stand

    closed = PlatePage.closed(
        Profile.demo(),
        run_doc,
        opener=slow_opener,
        entry_points=_entry_points("plate-registration", "scan-plan"),
    )
    closed.choose_plate("96-well")
    view = PlatePageView(closed)
    try:
        view.on_connect()
        assert entered.wait(5)
        view.tick()
        assert closed.status().running == "connect"
        # `connect` goes, and the stand is not there yet: the reason that holds
        # longer is the one said, not "connect is running".
        assert view.tool_notes["scan"].object == "the stand is not connected"
        assert view.next_note.object == "the stand is not connected"
    finally:
        gate.set()
        assert closed.join(5)
        closed.close()
        closed.release()
        for stand in stands:
            stand.close()


def test_the_replace_note_is_drawn_under_the_tool_that_would_replace_the_plate(
    run_doc: Path, page: PlatePage, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    # The page was built before the record: the operator named the plate since.
    page.choose_plate("96-well")
    view = PlatePageView(page)
    when = T0.astimezone().strftime("%Y-%m-%d %H:%M")
    note = (
        f"Run replaces the plate registered {when} (references, 2 wells). "
        "A failed run leaves no plate; the captures stay, so Forget the wrong one "
        "and Run again to get a plate back."
    )
    assert view.tool_notes["plate-registration"].object == note
    # The other tools say nothing of it: a note under every tool teaches none is read.
    for other in ("scan-plan", "scan", "goto-well"):
        assert note not in view.tool_notes[other].object
    # Under a busy note, after a newline: both are said.
    fake_microscope.override(Camera, FrameCamera(_rows_columns(6, 10)))
    view.on_live()
    try:
        assert view.tool_notes["plate-registration"].object == f"{LIVE_NOTE}\n{note}"
    finally:
        view.on_live()
    assert page.join(10.0)


# --- the plate the page asks for (m5 §11.3, #332) -----------------------------------

PLATE_ASK = (
    "Which plate is on the stage? 96-well is only a placeholder: choose the plate, "
    "or press the button if it is a 96-well."
)
PLATE_NOT_CHOSEN_TEXT = (
    "choose the plate on the stage first: the page shows a 96-well placeholder"
)


def test_an_unchosen_plate_shows_the_note_and_the_confirm_button(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    assert not page.plate_chosen
    assert view.plate_note.visible
    assert view.plate_note.object == PLATE_ASK
    assert view.plate_confirm_button.visible
    assert view.plate_confirm_button.label == "The plate is a 96-well"
    # The select's options and value are unchanged: the placeholder is shown, named.
    assert view.plate_select.value == "96-well"
    # The tool that takes the plate says why it is disabled, beside its button.
    assert view.tool_runs["plate-registration"].disabled
    assert view.tool_notes["plate-registration"].object == PLATE_NOT_CHOSEN_TEXT
    assert view.next_note.object == PLATE_NOT_CHOSEN_TEXT


def test_the_confirm_button_keeps_the_plate_and_hides_the_note(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, 0.0, 0.0)
    view.on_capture("centre")
    view.plate_confirm_button.clicks += 1
    assert (page.plate, page.plate_chosen) == ("96-well", True)
    assert not view.plate_note.visible
    assert not view.plate_confirm_button.visible
    # The same plate: the captures and the selection are kept, and the tool runs.
    assert [r.well for r in page.refs] == ["A1"]
    assert page.selection == ("A1",)
    assert not view.tool_runs["plate-registration"].disabled
    assert view.tool_notes["plate-registration"].object == ""
    assert view.message.object == ""


def test_choosing_another_plate_hides_the_note_too(page: PlatePage) -> None:
    view = PlatePageView(page)
    assert view.plate_note.visible
    view.plate_select.value = "24-well"
    assert (page.plate, page.plate_chosen) == ("24-well", True)
    assert not view.plate_note.visible
    assert not view.plate_confirm_button.visible


def test_a_second_tab_redraws_when_the_plate_is_confirmed(page: PlatePage) -> None:
    first, second = PlatePageView(page), PlatePageView(page)
    assert second.plate_note.visible
    first.plate_confirm_button.clicks += 1
    # Nothing else changed: not the plate, the selection, the captures. The shared
    # state carries `plate_chosen`, so the other tab's tick redraws (FM-151).
    assert second.plate_note.visible
    second.tick()
    assert not second.plate_note.visible
    assert not second.plate_confirm_button.visible
    assert not second.tool_runs["plate-registration"].disabled


def test_a_plate_the_run_document_names_needs_no_confirmation(
    run_doc: Path, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate("24-well"))
    named = PlatePage(
        fake_microscope, run_doc, entry_points=_entry_points("plate-registration")
    )
    try:
        view = PlatePageView(named)
        assert (named.plate, named.plate_chosen) == ("24-well", True)
        assert not view.plate_note.visible
        assert not view.plate_confirm_button.visible
    finally:
        named.close()


# --- the captures name the well they record (m5 §11.1, #332) -----------------------


def _nominal(well: str) -> tuple[float, float]:
    """The centre of ``well`` on the nominal 96-well map (A1 at 0, 0, not turned), written out."""
    row = string.ascii_uppercase.index(well[0])
    col = int(well[1:]) - 1
    return (col * PITCH_96_UM, -row * PITCH_96_UM)


def _capture_labels(view: PlatePageView) -> tuple[str, str, str]:
    return (
        view.capture_centre_button.label,
        view.capture_wall_button.label,
        view.forget_button.label,
    )


NO_WELL_NOTE = (
    "click a well on the map first: a capture records the well you clicked last"
)


def test_the_capture_buttons_name_the_well_and_follow_the_wall_select(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    _tap(view, *_nominal("B3"))
    assert page.current == "B3"
    assert _capture_labels(view) == (
        "Capture centre of B3",
        "Capture B3 wall at image left",
        "Forget B3",
    )
    # The select redraws the label at once: no tick between the change and the read.
    view.capture_side_select.value = "top"
    assert view.capture_wall_button.label == "Capture B3 wall at image top"
    assert view.capture_centre_button.label == "Capture centre of B3"
    assert view.capture_note.object == ""


def test_the_capture_buttons_are_disabled_with_no_well_and_the_note_says_why(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)
    assert _capture_labels(view) == ("Capture centre", "Capture wall", "Forget")
    assert view.capture_centre_button.disabled
    assert view.capture_wall_button.disabled
    assert view.forget_button.disabled
    assert view.capture_note.object == NO_WELL_NOTE
    # The select still redraws, and names nothing while there is no well.
    view.capture_side_select.value = "right"
    assert view.capture_wall_button.label == "Capture wall"
    _tap(view, *_nominal("A2"))
    assert not view.capture_centre_button.disabled
    assert not view.capture_wall_button.disabled
    assert not view.forget_button.disabled
    assert view.capture_note.object == ""


def test_the_capture_buttons_stay_disabled_with_the_stand_closed_whatever_the_well(
    run_doc: Path, fake_microscope: Microscope
) -> None:
    closed = PlatePage.closed(
        fake_microscope.profile,
        run_doc,
        entry_points=_entry_points("plate-registration"),
    )
    try:
        view = PlatePageView(closed)
        _tap(view, *_nominal("B3"))
        assert closed.current == "B3"
        # The captures read the stage: the *Connect* row says why, and the label still
        # names the well so that the operator sees what Connect will enable.
        assert view.capture_centre_button.disabled
        assert view.capture_wall_button.disabled
        assert view.capture_centre_button.label == "Capture centre of B3"
        # *Forget* reads no stage: only a missing well disables it.
        assert not view.forget_button.disabled
    finally:
        closed.close()


def test_a_press_on_buttons_drawn_with_no_well_says_to_click_one(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A disabled button still reaches the handler (FM-169). The buttons were drawn with
    # no well; another tab has since made B3 current, and this tab has not ticked: what
    # the label says (no well) is what the press answers, and it records nothing.
    view = PlatePageView(page)
    assert view.capture_centre_button.label == "Capture centre"
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    page.click(*_nominal("B3"), drive=False)
    view.on_capture("centre")
    assert view.message.object == "click a well on the map first"
    assert page.refs == ()
    # The redraw that followed names B3: a second press records it.
    assert view.capture_centre_button.label == "Capture centre of B3"
    view.on_capture("centre")
    assert [r.well for r in page.refs] == ["B3"]


def test_a_deselecting_tap_disables_the_capture_buttons(page: PlatePage) -> None:
    view = PlatePageView(page)
    _tap(view, *_nominal("B3"))
    assert not view.capture_centre_button.disabled
    _tap(view, *_nominal("B3"))
    assert page.current is None
    assert _capture_labels(view) == ("Capture centre", "Capture wall", "Forget")
    assert view.capture_centre_button.disabled
    assert view.capture_wall_button.disabled
    assert view.forget_button.disabled
    assert view.capture_note.object == NO_WELL_NOTE


def test_a_capture_clicked_on_a_stale_label_records_nothing(
    page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # FM-177: the label said B3; another tab's click (here, the controller's own
    # `click`) made B4 current before this tab's tick. The wall recorded under B4
    # at the position the operator took for B3's is the finding of #332.
    seed_camera_calibration(run_doc)
    view = PlatePageView(page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    page.click(*_nominal("B4"), drive=False)
    assert page.current == "B4"
    assert view.capture_wall_button.label == "Capture B3 wall at image left"
    stale = (
        "the well to record changed from B3 to B4: check the button, then press "
        "it again"
    )
    view.on_capture("wall")
    assert view.message.object == stale
    assert (page.refs, page.walls) == ((), ())
    # The view redrew: the label now names what a press would record.
    assert view.capture_wall_button.label == "Capture B4 wall at image left"
    view.on_capture("wall")
    assert view.message.object == ""
    assert [(w.well, w.side) for w in page.walls] == [("B4", "+X")]


def test_a_capture_of_a_centre_clicked_on_a_stale_label_records_nothing(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    page.click(*_nominal("B4"), drive=False)
    view.on_capture("centre")
    assert view.message.object == (
        "the well to record changed from B3 to B4: check the button, then press "
        "it again"
    )
    assert page.refs == ()


def test_a_forget_clicked_on_a_stale_label_forgets_nothing(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    view.on_capture("centre")
    _tap(view, *_nominal("B4"))
    view.on_capture("centre")
    assert [r.well for r in page.refs] == ["B3", "B4"]
    page.click(*_nominal("C1"), drive=False)
    assert view.forget_button.label == "Forget B4"
    view.on_forget()
    assert view.message.object == (
        "the well to record changed from B4 to C1: check the button, then press "
        "it again"
    )
    # Neither B4's capture nor B3's went: the click recorded no decision.
    assert [r.well for r in page.refs] == ["B3", "B4"]
    assert view.forget_button.label == "Forget C1"


def test_a_capture_clicked_after_current_was_dropped_says_none(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    page.toggle("B3")  # another tab deselected it: `current` is dropped
    assert page.current is None
    view.on_capture("centre")
    assert view.message.object == (
        "the well to record changed from B3 to none: check the button, then press "
        "it again"
    )
    assert page.refs == ()
    # The redraw followed: the buttons are the plain three, disabled, with the note.
    assert _capture_labels(view) == ("Capture centre", "Capture wall", "Forget")
    assert view.capture_note.object == NO_WELL_NOTE
    view.on_forget()
    assert view.message.object == "click a well on the map first"


def test_a_capture_after_a_tick_records_the_well_on_the_label(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    page.click(*_nominal("B4"), drive=False)
    # The tick sees the shared state changed and redraws, as it does for another tab.
    view.tick()
    assert view.capture_centre_button.label == "Capture centre of B4"
    view.on_capture("centre")
    assert view.message.object == ""
    assert [(r.well, r.x_um, r.y_um) for r in page.refs] == [("B4", 10.0, 20.0)]


# --- plate-walls on the page: its centre captures, its Run, its card (m5 §12, #346) ----


@pytest.fixture
def walls_page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    """A page with the calibration, the registration and the wall registration: the *Set up* stage's three."""
    controller = PlatePage(
        fake_microscope,
        run_doc,
        entry_points=_entry_points(
            "camera-calibration", "plate-registration", "plate-walls"
        ),
    )
    try:
        yield controller
    finally:
        controller.close()


def _section(view: PlatePageView, tool: str) -> Any:
    """The tool's section of the accordion: the column whose first line is its name."""
    for panel in view.accordion.objects:
        for column in panel.select(pn.Column):
            first = column.objects[0] if column.objects else None
            if isinstance(first, pn.pane.Str) and first.object == tool:
                return column
    raise AssertionError(f"no section for {tool}")


WALLS_WIDGETS = (
    "walls_capture_button",
    "walls_forget_button",
    "walls_capture_note",
    "walls_captures_text",
    "walls_touches_note",
)
REGISTRATION_WIDGETS = (
    "capture_centre_button",
    "forget_button",
    "capture_wall_button",
    "capture_side_select",
    "capture_note",
    "captures_text",
)


def test_plate_walls_section_holds_the_centre_captures_and_not_the_wall_ones(
    walls_page: PlatePage,
) -> None:
    view = PlatePageView(walls_page)
    walls = {id(o) for o in _section(view, "plate-walls").select()}
    registration = {id(o) for o in _section(view, "plate-registration").select()}
    for name in WALLS_WIDGETS:
        assert id(getattr(view, name)) in walls, f"{name} is not in plate-walls"
        assert id(getattr(view, name)) not in registration
    for name in REGISTRATION_WIDGETS:
        # The registration keeps every widget it has, by the same names (§12.5), and the
        # wall captures are its own: `plate-walls` images the walls itself.
        assert id(getattr(view, name)) in registration, f"{name} left the registration"
        assert id(getattr(view, name)) not in walls
    # §12.5's order, top to bottom: the summary, the two buttons, the note, the centres,
    # the wall note, the Advanced fold, Run, the tool's note, the card.
    objects = _section(view, "plate-walls").objects
    assert objects[0].object == "plate-walls"
    buttons = objects[2]
    assert [id(o) for o in buttons.objects] == [
        id(view.walls_capture_button),
        id(view.walls_forget_button),
    ]
    assert objects[3] is view.walls_capture_note
    assert objects[4] is view.walls_captures_text
    assert objects[5] is view.walls_touches_note
    assert isinstance(objects[6], pn.Card)
    assert objects[7] is view.tool_runs["plate-walls"]
    assert objects[8] is view.tool_notes["plate-walls"]
    assert objects[9] is view.tool_cards["plate-walls"]


def _walls_labels(view: PlatePageView) -> tuple[str, str]:
    return view.walls_capture_button.label, view.walls_forget_button.label


def test_the_walls_buttons_name_the_well_and_follow_a_tap(
    walls_page: PlatePage,
) -> None:
    view = PlatePageView(walls_page)
    assert _walls_labels(view) == ("Capture centre", "Forget")
    _tap(view, *_nominal("B3"))
    assert _capture_labels(view)[0] == "Capture centre of B3"
    # The same well on both sets of buttons, from the one `current` they were drawn for.
    assert _walls_labels(view) == ("Capture centre of B3", "Forget B3")
    assert view.walls_capture_note.object == ""
    _tap(view, *_nominal("D5"))
    assert _walls_labels(view) == ("Capture centre of D5", "Forget D5")
    # A press records the well the label names, whichever set it is on.
    view.walls_capture_button.clicks += 1
    assert [r.well for r in walls_page.refs] == ["D5"]


def test_the_walls_buttons_are_disabled_with_no_well_and_with_the_stand_closed(
    walls_page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    view = PlatePageView(walls_page)
    assert view.walls_capture_button.disabled
    assert view.walls_forget_button.disabled
    assert view.walls_capture_note.object == NO_WELL_NOTE
    _tap(view, *_nominal("A2"))
    assert not view.walls_capture_button.disabled
    assert not view.walls_forget_button.disabled
    assert view.walls_capture_note.object == ""
    _tap(view, *_nominal("A2"))
    assert view.walls_capture_button.disabled
    assert _walls_labels(view) == ("Capture centre", "Forget")
    assert view.walls_capture_note.object == NO_WELL_NOTE

    closed = PlatePage.closed(
        fake_microscope.profile,
        run_doc,
        entry_points=_entry_points("camera-calibration", "plate-walls"),
    )
    try:
        shut = PlatePageView(closed)
        _tap(shut, *_nominal("B3"))
        # The captures read the stage: disabled with the stand closed, the label naming
        # the well all the same; *Forget* reads no stage.
        assert shut.walls_capture_button.disabled
        assert shut.walls_capture_button.label == "Capture centre of B3"
        assert not shut.walls_forget_button.disabled
    finally:
        closed.close()


def test_a_walls_capture_clicked_on_a_stale_label_records_nothing(
    walls_page: PlatePage, fake_microscope: Microscope
) -> None:
    # FM-177 for the second set of buttons: the label said B3, another tab's click made
    # B4 current before this tab's tick. `page.click` moves `current` with no tick.
    view = PlatePageView(walls_page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    walls_page.click(*_nominal("B4"), drive=False)
    assert walls_page.current == "B4"
    assert view.walls_capture_button.label == "Capture centre of B3"
    view.walls_capture_button.clicks += 1
    assert view.message.object == (
        "the well to record changed from B3 to B4: check the button, then press "
        "it again"
    )
    assert walls_page.refs == ()
    # The view redrew both sets for the well a press would record now.
    assert _walls_labels(view) == ("Capture centre of B4", "Forget B4")
    view.walls_capture_button.clicks += 1
    assert view.message.object == ""
    assert [r.well for r in walls_page.refs] == ["B4"]


def test_a_walls_forget_clicked_on_a_stale_label_forgets_nothing(
    walls_page: PlatePage, fake_microscope: Microscope
) -> None:
    view = PlatePageView(walls_page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(view, *_nominal("B3"))
    view.walls_capture_button.clicks += 1
    assert [r.well for r in walls_page.refs] == ["B3"]
    walls_page.click(*_nominal("C1"), drive=False)
    assert view.walls_forget_button.label == "Forget B3"
    view.walls_forget_button.clicks += 1
    assert view.message.object == (
        "the well to record changed from B3 to C1: check the button, then press "
        "it again"
    )
    assert [r.well for r in walls_page.refs] == ["B3"]
    assert view.walls_forget_button.label == "Forget C1"


def test_the_centres_and_the_wall_note_are_drawn_for_the_captures_the_page_holds(
    walls_page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    seed_camera_calibration(run_doc)
    view = PlatePageView(walls_page)
    assert view.walls_captures_text.object == "no centre captured yet"
    assert view.walls_touches_note.object == ""
    stage = fake_microscope.require(XYStage)
    stage.move_to_um(10182.0, -10087.9)
    _tap(view, *_nominal("B3"))
    view.walls_capture_button.clicks += 1
    stage.move_to_um(1.0, 2.0)
    _tap(view, *_nominal("D5"))
    view.walls_capture_button.clicks += 1
    assert view.walls_captures_text.object == centres_text(walls_page.refs)
    assert view.walls_captures_text.object == (
        "centres captured (2):\n  B3 (10182.0, -10087.9) µm\n  D5 (1.0, 2.0) µm"
    )
    # A wall touch is the registration's: said once, where the operator wonders.
    view.capture_wall_button.clicks += 1
    assert view.walls_touches_note.object == wall_touches_note(1)
    assert "1 wall capture is for plate-registration" in view.walls_touches_note.object
    # And forgetting the well clears its centre and its wall.
    view.walls_forget_button.clicks += 1
    assert view.walls_touches_note.object == ""
    assert (
        view.walls_captures_text.object
        == "centres captured (1):\n  B3 (10182.0, -10087.9) µm"
    )


def test_a_second_tab_redraws_the_walls_captures(
    walls_page: PlatePage,
    fake_microscope: Microscope,
    run_doc: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    # FM-151: a capture made through another tab is in the controller, and this tab
    # draws it at its next tick.
    seed_camera_calibration(run_doc)
    one, two = PlatePageView(walls_page), PlatePageView(walls_page)
    fake_microscope.require(XYStage).move_to_um(10.0, 20.0)
    _tap(one, *_nominal("B3"))
    one.walls_capture_button.clicks += 1
    assert two.walls_captures_text.object == "no centre captured yet"
    two.tick()
    assert two.walls_captures_text.object == centres_text(walls_page.refs)
    assert two.walls_capture_button.label == "Capture centre of B3"
    walls_page.capture_wall("B3", "left")
    two.tick()
    assert two.walls_touches_note.object == wall_touches_note(1)


def test_the_walls_run_is_disabled_with_the_reason_beside_it_and_enabled_when_nothing_blocks(
    walls_page: PlatePage, run_doc: Path, fake_microscope: Microscope
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    # The fixture's page opened on an empty document: the plate is what a record names.
    walls_page.choose_plate("96-well")
    view = PlatePageView(walls_page)
    run, note = view.tool_runs["plate-walls"], view.tool_notes["plate-walls"]
    stage = fake_microscope.require(XYStage)
    when = T0.astimezone().strftime("%Y-%m-%d %H:%M")
    replace = (
        f"Run replaces the plate registered {when} (references, 2 wells). "
        "A failed run leaves no plate; the captures stay, so Forget the wrong one "
        "and Run again to get a plate back."
    )
    # No calibration, no centre: the page's sentence for the calibration comes first,
    # and the replace note is below it.
    assert run.disabled
    assert note.object == f"{WALLS_NEED_CALIBRATION}\n{replace}"
    _seed(run_doc, "camera-calibration", "camera-calibration", NoResult(), minute=1)
    view.refresh()
    assert run.disabled
    assert note.object == f"{walls_centres_problem(0)}\n{replace}"
    for index, well in enumerate(["B3", "D5"]):
        stage.move_to_um(9000.0 * index, 0.0)
        # The registered plate is on the map: the wells are where it puts them.
        _tap(view, *_centre(well))
        view.walls_capture_button.clicks += 1
    assert [r.well for r in walls_page.refs] == ["B3", "D5"]
    assert not run.disabled
    assert note.object == replace


def _walls_plate() -> PlateResult:
    """What ``plate-walls`` writes for one well: four frames on each of its four sides."""
    fit = WellFit(
        well="B3",
        recorded_xy_um=(10186.0, -10092.0),
        center_xy_um=(10182.04, -10087.93),
        diameter_um=15298.4,
        size_error=0.0,
        rms_um=3.0,
        n_points=40,
        walls=tuple(
            WallFrame(side=side, x_um=0.0, y_um=0.0, n_points=10, frames=4)
            for side in ("+X", "+Y", "-X", "-Y")
        ),
    )
    return PlateResult(
        plate="24-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.1234,
        method="wall-images",
        residual_um=3.24,
        scale_error=0.0,
        n_refs=1,
        walls=WallImages(
            objective="10x",
            pixel_size_um=1.0,
            image_shape_px=(100, 100),
            search_um=3900.0,
            wells=(fit,),
            frames=16,
            folder="plate-walls/tok",
            elapsed_s=16.5,
        ),
    )


def test_the_card_text_of_a_walls_result_lists_the_wells(
    walls_page: PlatePage, run_doc: Path
) -> None:
    result = _walls_plate()
    _seed(run_doc, "plate-walls", "plate", result)
    view = PlatePageView(walls_page)
    text = view.tool_cards["plate-walls"].object
    # The wells, then the plate, as the controller's card says them; not the result's
    # first keys with the walls cut away.
    for line in walls_card_lines(result):
        assert line in text
    assert "B3: centre (10182.0, -10087.9) µm, 15298 µm across, 16 frames" in text
    assert f"folder: {run_doc.parent / 'plate-walls' / 'tok'}" in text


def test_a_dry_run_page_says_so_beside_the_walls_run(
    run_doc: Path, fake_core: FakeCore
) -> None:
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        controller = PlatePage(
            dry,
            run_doc,
            entry_points=_entry_points("camera-calibration", "plate-walls"),
        )
        try:
            controller.choose_plate("96-well")
            view = PlatePageView(controller)
            assert view.tool_runs["plate-walls"].disabled
            assert view.tool_notes["plate-walls"].object == WALLS_DRY_RUN
        finally:
            controller.close()
    finally:
        dry.close()


def test_no_widget_is_placed_twice_in_the_page(walls_page: PlatePage) -> None:
    # A Panel widget placed twice in one layout is drawn once (measured: one Bokeh model
    # for two places), so the second place is empty and nothing fails. The tree is the
    # check, not a look at the page: every object of the layout is found once.
    view = PlatePageView(walls_page)
    found = _placements(view)
    assert len(found) == len(set(found))


def _placements(view: PlatePageView) -> list[int]:
    """The id of every object of the layout, once for each place it is in: the Accordion's panels too."""
    found = [id(o) for o in view.layout.select()]
    for accordion in view.layout.select(pn.Accordion):
        for panel in accordion.objects:
            found.extend(id(o) for o in panel.select())
    return found


def test_the_check_for_a_widget_placed_twice_can_fail(walls_page: PlatePage) -> None:
    # The check above must be able to fail: a widget put in two places is found twice.
    view = PlatePageView(walls_page)
    _section(view, "plate-registration").append(view.walls_forget_button)
    found = _placements(view)
    assert len(found) - len(set(found)) == 1
    assert found.count(id(view.walls_forget_button)) == 2


def test_the_checklist_says_what_each_slot_is(run_doc: Path, page: PlatePage) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    assert view.checklist_text.object == "\n".join(
        [
            "· camera-calibration",
            "✓ plate",
            "· timings",
            "· scan-plan",
            "· scan",
            "· cells",
            "· density",
            "· route",
            "· report",
            "· nis-export",
        ]
    )


def test_tick_does_not_rebuild_the_view(page: PlatePage) -> None:
    view = PlatePageView(page)
    counts = {"view": 0, "status": 0}
    real_view, real_status = page.view, page.status

    def counting_view() -> Any:
        counts["view"] += 1
        return real_view()

    def counting_status() -> Any:
        counts["status"] += 1
        return real_status()

    with (
        _replaced(page, "view", counting_view),
        _replaced(page, "status", counting_status),
    ):
        for _ in range(10):
            view.tick()
    assert counts == {"view": 0, "status": 10}


def test_a_problem_is_shown_above_the_map(run_doc: Path, page: PlatePage) -> None:
    run_doc.write_bytes(b"not json")
    view = PlatePageView(page)
    assert view.problems_text.object == (
        f"run document: {run_doc}: is not valid JSON: Expecting value (line 1, "
        f"column 1); it was left as it is, restore it from a copy"
    )


def test_a_closed_or_reloaded_tab_does_not_stop_a_run(page: PlatePage) -> None:
    first = PlatePageView(page)
    first.on_run("scan")
    assert SlowScan.started.wait(5)
    PlatePageView(page)  # a second tab, built and dropped
    del first  # and the first one closed
    gc.collect()
    status = page.status()
    assert status.running == "scan"
    assert not status.halted
    assert RunDocument.load(page._run_doc).stages["scan"].status is StageStatus.running
    SlowScan.release.set()
    assert page.join(10.0)
    assert RunDocument.load(page._run_doc).stages["scan"].status is StageStatus.done


def test_the_plate_choice_redraws_the_map(page: PlatePage) -> None:
    view = PlatePageView(page)
    page.select_text("A1")
    view.plate_select.value = "24-well"
    assert page.plate == "24-well"
    assert page.selection == ()
    assert len(_wells_by_name(view)) == len(
        useq.WellPlatePlan(
            plate=useq.WellPlate.from_str("24-well"), a1_center_xy=(0.0, 0.0)
        ).all_well_names.ravel()
    )
    assert view.plate_select.value == "24-well"


# --- serve ----------------------------------------------------------------------------


class _Thread:
    """What ``pn.serve(..., threaded=True)`` returns, as far as ``serve`` uses it."""

    def __init__(
        self,
        *,
        alive: bool,
        interrupt: bool = False,
        events: list[str] | None = None,
        stop_waits_for: threading.Event | None = None,
    ) -> None:
        self.alive = alive
        self.interrupt = interrupt
        self.events = events if events is not None else []
        # Panel's `stop` ends with a join that has no limit: this one waits for the test.
        self.stop_waits_for = stop_waits_for
        self.calls: list[tuple[str, float | None]] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))
        if self.interrupt and timeout == 0.1:
            self.interrupt = False
            raise KeyboardInterrupt  # raised from the stub, never a signal (FM-44)

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self.events.append("server stop")
        if self.stop_waits_for is not None:
            self.stop_waits_for.wait(30)
        self.alive = False


@pytest.fixture
def served_thread(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., list[str]]]:
    """Put a ``_Thread`` where ``pn.serve`` would; the list is what happened, in order.

    The page's own ``close`` and the console's ``_say`` are recorded in it, so a test
    reads the order of the stop of the run and the stop of the server (FM-94: the
    assertion is after, never inside a callback).
    """
    released: list[threading.Event] = []

    def install(
        thread: _Thread, *, say: Callable[[str], None] | None = None
    ) -> list[str]:
        events = thread.events

        def close() -> None:
            events.append("page close")

        monkeypatch.setattr(pn, "serve", lambda apps, **kwargs: thread)
        monkeypatch.setattr(
            "smc.cli._say", say or (lambda text: events.append(f"say: {text}"))
        )
        setattr(page, "close", close)  # noqa: B010  (an instance attribute, removed below)
        if thread.stop_waits_for is not None:
            released.append(thread.stop_waits_for)
        return events

    try:
        yield install
    finally:
        for event in released:
            event.set()  # a stopper thread a failed test left behind ends
        if "close" in vars(page):
            delattr(page, "close")


def test_serve_returns_on_ctrl_c_and_stops_the_server(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    thread = _Thread(alive=True, interrupt=True, events=events)
    served: list[dict[str, Any]] = []
    said: list[str] = []
    bokeh_levels: list[int] = []
    bokeh_log = logging.getLogger("bokeh")

    def fake_serve(apps: dict[str, Any], **kwargs: Any) -> _Thread:
        served.append({"apps": list(apps), **kwargs})
        bokeh_levels.append(
            bokeh_log.level
        )  # asserted after: the call is not ours (FM-94)
        return thread

    monkeypatch.setattr(pn, "serve", fake_serve)
    monkeypatch.setattr("smc.cli._say", said.append)
    before = bokeh_log.level
    with _replaced(page, "close", lambda: events.append("page close")):
        serve(page, port=5123)
    # The run is told to stop before the server is touched (it can wait on a busy
    # tick, measured at twice the tick), and the server is then stopped and waited for.
    assert events == ["page close", "server stop"]
    assert thread.calls[0] == ("join", 0.1)
    assert ("stop", None) in thread.calls
    assert served == [
        {
            "apps": [""],
            "address": "localhost",
            "port": 5123,
            "show": False,
            "threaded": True,
            "verbose": False,
        }
    ]
    assert said == [
        "serving the plate page on http://localhost:5123/ — Ctrl-C stops it"
    ]
    # Bokeh is quiet while the page serves (its console would block the loop that
    # handles STOP), and its level is put back.
    assert bokeh_levels == [logging.WARNING]
    assert bokeh_log.level == before


def test_serve_says_when_the_server_stopped_on_its_own(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pn, "serve", lambda apps, **kwargs: _Thread(alive=False))
    monkeypatch.setattr("smc.cli._say", lambda text: None)
    before = logging.getLogger("bokeh").level
    with pytest.raises(PageServerError) as stopped:
        serve(page, port=5006)
    assert str(stopped.value) == (
        "the plate page's server stopped on its own (is port 5006 taken?)"
    )
    assert logging.getLogger("bokeh").level == before


_ANNOUNCED = "say: serving the plate page on http://localhost:5006/ — Ctrl-C stops it"


def test_a_server_that_will_not_stop_is_waited_for_a_while_and_said(
    page: PlatePage,
    served_thread: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Panel's `stop` ends with a join that has no limit, and the server waits for the
    # callback in flight: asked directly it would hold the console for as long as a slow
    # device read. The stub's `stop` waits for the test, as that join would.
    release = threading.Event()
    thread = _Thread(alive=True, interrupt=True, stop_waits_for=release)
    events = served_thread(thread)
    monkeypatch.setattr(plate_page, "SERVER_STOP_WAIT_S", 1.0)
    started = time.monotonic()
    serve(page, port=5006)
    waited = time.monotonic() - started
    release.set()
    assert events == [
        _ANNOUNCED,
        "page close",
        "server stop",
        "say: the plate page's server did not stop within 1 s",
    ]
    assert waited < 10  # not the 30 s that the stub's stop would take


def test_a_ctrl_c_before_the_server_is_announced_still_stops_the_run_and_the_server(
    page: PlatePage, served_thread: Callable[..., list[str]]
) -> None:
    def say(text: str) -> None:
        raise KeyboardInterrupt  # from the stub, never a signal (FM-44)

    thread = _Thread(alive=True)
    events = served_thread(thread, say=say)
    serve(page, port=5006)
    assert events == ["page close", "server stop"]


def test_a_console_that_fails_still_stops_the_server(
    page: PlatePage, served_thread: Callable[..., list[str]]
) -> None:
    def say(text: str) -> None:
        raise BrokenPipeError("the console is closed")

    thread = _Thread(alive=True)
    events = served_thread(thread, say=say)
    with pytest.raises(BrokenPipeError):
        serve(page, port=5006)
    # Panel's thread is not a daemon: left running it would keep the process alive.
    assert events == ["server stop"]


def test_a_second_ctrl_c_while_the_run_is_stopped_still_stops_the_server(
    page: PlatePage, served_thread: Callable[..., list[str]]
) -> None:
    thread = _Thread(alive=True, interrupt=True)
    events = served_thread(thread)

    def close() -> None:
        events.append("page close")
        raise KeyboardInterrupt  # the operator's second Ctrl-C (FM-71)

    with _replaced(page, "close", close), pytest.raises(KeyboardInterrupt):
        serve(page, port=5006)
    assert events == [_ANNOUNCED, "page close", "server stop"]


def test_the_served_page_answers(page: PlatePage) -> None:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    thread = pn.serve(
        {"": lambda: build(page)},
        address="localhost",
        port=port,
        show=False,
        threaded=True,
    )
    try:
        deadline = time.monotonic() + 20
        status = 0
        body = b""
        while time.monotonic() < deadline:
            try:
                # No proxy: a PC's proxy settings (the environment, and Windows' Internet
                # Options) would send a request for localhost to the proxy.
                with direct.open(f"http://localhost:{port}/", timeout=20) as reply:
                    status, body = reply.status, reply.read()
                break
            except OSError:
                time.sleep(0.2)
        assert status == 200
        assert b"bokeh" in body.lower()
    finally:
        thread.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()


# --- the widgets are wired to what they say -----------------------------------------


def test_a_wall_pressed_with_no_calibration_says_why_and_records_nothing(
    page: PlatePage,
) -> None:
    # The press is the guard, not the button's state (FM-169): with no camera
    # calibration no stage side can be named, and the page says what to run.
    view = PlatePageView(page)
    _tap(view, *_nominal("B3"))
    view.capture_wall_button.clicks += 1
    assert view.message.object == NO_CALIBRATION_FOR_SIDES
    assert page.walls == ()
    assert view.captures_text.object == "no capture yet"


def test_a_drive_tap_to_a_wall_with_no_calibration_says_why_and_drives_nowhere(
    run_doc: Path, page: PlatePage
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    view = PlatePageView(page)
    view.side_select.value = "left"
    view.drive_toggle.value = True
    _tap(view, *_centre("C5"))
    assert view.message.object == NO_CALIBRATION_FOR_SIDES
    # One click, even a refused one: a stage that stays armed is the danger.
    assert not view.drive_toggle.value
    assert "goto-well" not in RunDocument.load(run_doc).stages
    assert page.current is None


def test_a_drive_tap_goes_to_the_wall_the_side_select_says(
    run_doc: Path,
    page: PlatePage,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed(run_doc, "plate-registration", "plate", _plate())
    seed_camera_calibration(run_doc)  # the Ti2-E's: the image's left is stage +X
    view = PlatePageView(page)
    view.side_select.value = "left"
    view.drive_toggle.value = True
    _tap(view, *_centre("C5"))
    assert page.join(10.0)
    record = RunDocument.load(run_doc).stages["goto-well"]
    # `goto-well` gets the stage's side: the page named the wall on the image.
    assert record.params == {"well": "C5", "side": "+X"}
    assert record.status is StageStatus.done


def test_each_run_button_starts_its_own_tool(run_doc: Path, page: PlatePage) -> None:
    page.choose_plate("96-well")
    view = PlatePageView(page)
    page.select_text("A1")
    sent: list[tuple[str, list[str]]] = []

    def record(plugin: str, params: list[str]) -> None:
        sent.append((plugin, list(params)))

    with _replaced(page, "run", record):
        for name in ("plate-registration", "scan-plan", "scan"):
            sent.clear()
            view.tool_runs[
                name
            ].disabled = False  # a click is what is tested, not `blocked`
            view.tool_runs[name].clicks += 1
            assert [plugin for plugin, _ in sent] == [name]


def test_the_refresh_button_redraws_the_view_after_the_run_document_changed(
    run_doc: Path, page: PlatePage
) -> None:
    view = PlatePageView(page)
    unregistered = (
        "96-well: nominal map (A1 at 0, 0, not turned), the plate is not registered"
    )
    assert view.figure.title.text == unregistered
    _seed(run_doc, "plate-registration", "plate", _plate())
    assert view.figure.title.text == unregistered  # a tick never reads the view
    view.refresh_button.clicks += 1
    assert view.figure.title.text == "96-well: the registered plate"
    assert "✓ plate" in view.checklist_text.object


def test_build_registers_the_tick_every_second_and_the_frame_tick_every_tenth(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: list[tuple[Any, Any]] = []

    def recorder(callback: Any, period: Any = 500, *args: Any, **kwargs: Any) -> None:
        registered.append((callback, period))

    monkeypatch.setattr(pn.state, "add_periodic_callback", recorder)
    build(page)
    assert len(registered) == 2
    views = {callback.__self__ for callback, _period in registered}
    assert len(views) == 1  # both are the one session's own
    assert isinstance(next(iter(views)), PlatePageView)
    # The status poll is one second, and no faster: a scan must not be slowed by it.
    # The frame tick is a tenth: the live view's period, and it reads only the frame.
    assert {(callback.__func__, period) for callback, period in registered} == {
        (PlatePageView.tick, 1000),
        (PlatePageView.frame_tick, 100),
    }
    assert plate_page.FRAME_TICK_MS / 1000 == LIVE_PERIOD_S


def test_a_device_error_reaches_the_problems_line_after_a_tick(
    page: PlatePage, fake_core: FakeCore
) -> None:
    def refuses(*args: object, **kwargs: object) -> list[float]:
        raise RuntimeError("the stage does not answer")

    with _replaced(fake_core, "getXYPosition", refuses):
        view = PlatePageView(page)
        view.tick()
        problems = page.status().problems
    assert problems == (
        "xy: HardwareError: xy_stage: getXPosition('XY') failed (the stage does not "
        "answer); check the device: power, cable, another program holding it",
    )
    assert view.problems_text.object == "\n".join(problems)
    assert view.status_text.object.startswith("demo | XY ? | ")


def test_the_light_that_is_on_is_said_on(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    class LampOn:
        def is_on(self) -> bool:
            return True

    real_get = fake_microscope.get

    def get(capability: object) -> object:
        return LampOn() if capability is LightSource else real_get(capability)

    with _replaced(fake_microscope, "get", get):
        view = PlatePageView(page)
        status = page.status()
    assert status.light_on is True
    assert "light on" in view.status_text.object


def test_a_stage_that_is_moving_is_listed(
    page: PlatePage, fake_core: FakeCore, fake_microscope: Microscope
) -> None:
    label = fake_core.getXYStageDevice()
    fake_core.busy_devices.add(label)
    stage = fake_microscope.require(XYStage)
    mover = threading.Thread(
        target=lambda: stage.move_to_um(5.0, 5.0), daemon=True, name="smc-test-move"
    )
    mover.start()
    seen: tuple[str, ...] = ()
    try:
        deadline = time.monotonic() + 5.0
        while not seen and time.monotonic() < deadline:
            seen = page.status().moving
            time.sleep(0.01)
        assert seen == ("xy_stage XY",)
        view = PlatePageView(page)
        assert "moving: xy_stage XY" in view.status_text.object
    finally:
        fake_core.busy_devices.discard(label)
        mover.join(10.0)
    assert not mover.is_alive()


def test_a_dry_run_is_said_on_the_strip(run_doc: Path, fake_core: FakeCore) -> None:
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    controller = PlatePage(dry, run_doc, entry_points=_entry_points("goto-well"))
    try:
        assert controller.status().dry_run
        assert "DRY RUN" in PlatePageView(controller).status_text.object
    finally:
        controller.close()
        dry.close()


def test_a_selection_made_in_another_tab_is_drawn_by_the_next_tick(
    page: PlatePage,
) -> None:
    # The controller is the server's: a tab that does not redraw would run
    # `scan-plan` with wells it does not show.
    first, second = PlatePageView(page), PlatePageView(page)
    _tap(first, *_centre("A1"))
    assert page.selection == ("A1",)
    assert _wells_by_name(second)["A1"]["outline_width"] == 1
    second.tick()
    assert _wells_by_name(second)["A1"]["outline_width"] == 3
    page.choose_plate("24-well")
    second.tick()
    assert second.plate_select.value == "24-well"
    assert len(_wells_by_name(second)) == 24


def test_a_tick_redraws_the_view_only_when_something_changed(page: PlatePage) -> None:
    view = PlatePageView(page)
    counted = {"view": 0}
    real_view = page.view

    def counting_view() -> Any:
        counted["view"] += 1
        return real_view()

    with _replaced(page, "view", counting_view):
        view.tick()
        view.tick()
        assert counted["view"] == 0
        page.select_text("A1")
        view.tick()
        assert counted["view"] == 1
        view.tick()
        assert counted["view"] == 1
