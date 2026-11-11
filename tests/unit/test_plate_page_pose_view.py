"""The plate page draws the map's pose and a cross that follows the stage (m5 §15 D1, D4, D5; #410).

The Panel layer without a browser, as ``test_plate_page.py`` drives it: ``view.tick()`` and
``view.frame_tick()`` are what the two periodic callbacks call, ``stage_source.data`` is what the
cross draws and ``figure.title.text`` what the map says. The cross has two writers (the tick, once a
second, and the frame tick, ten times a second while Live runs), and the tests that prove they never
leave it at an older position script the controller's Live state, its frame and the stage's
position, so that no test waits for a camera.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from bokeh.models import LabelSet
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import Camera, XYStage
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.slots import PlateResult, WellRef
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import PageStatus, PlatePage
from smc.ui.frames import Frame
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

UM = "\N{MICRO SIGN}m"
DEG = "\N{DEGREE SIGN}"
HOLDER_CENTRE = (400.0, -325.0)
THEN = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
#: A1 of the 96-well plate as the estimate places it (m5 §15 D3): the holder's centre
#: minus the grid's offset from A1, (49 500, -31 500) µm.
ESTIMATED_A1 = (-49_100.0, 31_175.0)
PITCH_UM = 9000.0
B3 = (ESTIMATED_A1[0] + 2 * PITCH_UM, ESTIMATED_A1[1] - PITCH_UM)

ESTIMATED_TITLE = (
    "96-well: estimated from the stand's holder, not registered: "
    f"centred at stage (400, -325) {UM}, not turned, A1 at the top left of this map"
)
UNPLACED_TITLE = (
    "96-well: not registered, and this stand does not say where its plate lies: "
    "A1 is drawn at stage (0, 0), so the cross shows the stage and not a well"
)


def _profile(holder: bool) -> Profile:
    demo = Profile.demo()
    if not holder:
        return demo
    return demo.model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=HOLDER_CENTRE,
                measured="2026-11-10, sessions 2 and 3 (#334, #405)",
            )
        }
    )


def _controller(
    core: FakeCore, run_doc: Path, *, holder: bool
) -> Iterator[tuple[PlatePage, Microscope]]:
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _profile(holder))
    page = PlatePage(microscope, run_doc, entry_points=[])
    try:
        yield page, microscope
    finally:
        page.close()
        microscope.close()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def stand(fake_core: FakeCore, run_doc: Path) -> Iterator[tuple[PlatePage, Microscope]]:
    """The page and its stand, on a profile with a ``[holder]``."""
    yield from _controller(fake_core, run_doc, holder=True)


@pytest.fixture
def page(stand: tuple[PlatePage, Microscope]) -> PlatePage:
    return stand[0]


@pytest.fixture
def bare_page(fake_core: FakeCore, run_doc: Path) -> Iterator[PlatePage]:
    for controller, _ in _controller(fake_core, run_doc, holder=False):
        yield controller


class _Scripted:
    """The controller as one view reads it, with Live, the frame and the stage's position scripted.

    Everything else is the real controller's, so the view, the pose and the wells are the
    page's own. ``reads`` is the name of every attribute the view asked the controller for,
    in order: it shows what a callback reads (FM-145).
    """

    def __init__(self, page: PlatePage) -> None:
        self._page = page
        self.live = False
        self.frame_now: Frame | None = None
        self.stage_xy: tuple[float, float] | None = None
        self.reads: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.reads.append(name)
        return getattr(self._page, name)

    @property
    def is_live(self) -> bool:
        self.reads.append("is_live")
        return self.live

    def frame(self) -> Frame | None:
        self.reads.append("frame")
        return self.frame_now

    def status(self) -> PageStatus:
        self.reads.append("status")
        return replace(self._page.status(), xy_um=self.stage_xy)


def _scripted(page: PlatePage) -> tuple[PlatePageView, _Scripted]:
    scripted = _Scripted(page)
    view = PlatePageView(cast("PlatePage", scripted))
    return view, scripted


def _frame(number: int, xy_um: tuple[float, float] | None) -> Frame:
    return Frame(
        number=number,
        image=np.zeros((4, 4), dtype=np.uint16),
        bit_depth=16,
        exposure_ms=5.0,
        pixel_size_um=0.65,
        xy_um=xy_um,
        taken_utc=THEN,
        summary=f"frame {number}",
    )


def _cross(view: PlatePageView) -> dict[str, list[Any]]:
    return {key: list(value) for key, value in view.stage_source.data.items()}


def _register(
    run_doc: Path,
    *,
    a1: tuple[float, float],
    method: str = "references",
    refs: tuple[WellRef, ...] = (),
) -> None:
    seeded = method == "seed"
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=a1,
        rotation_deg=0.0,
        method=method,  # type: ignore[arg-type]
        residual_um=None if seeded else 2.0,
        scale_error=None if seeded else 0.001,
        n_refs=len(refs) if seeded else 2,
        refs=refs,
    )
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=THEN,
            finished_utc=THEN,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={"plate": plate.plate},
            result=plate.model_dump(mode="json"),
        ),
    )


# -- the title says which pose the map is drawn in ---------------------------------


def test_the_title_says_which_pose_the_map_is_drawn_in(
    page: PlatePage, bare_page: PlatePage, run_doc: Path
) -> None:
    assert PlatePageView(bare_page).figure.title.text == UNPLACED_TITLE

    view = PlatePageView(page)
    assert view.figure.title.text == ESTIMATED_TITLE

    seed = (WellRef(well="B3", x_um=B3[0], y_um=B3[1]),)
    _register(run_doc, a1=ESTIMATED_A1, method="seed", refs=seed)
    view.refresh()
    assert view.figure.title.text == (
        "96-well: placed from one well (B3), not registered: "
        f"the rotation is assumed 0{DEG}, not measured"
    )

    _register(run_doc, a1=ESTIMATED_A1)
    view.refresh()
    assert view.figure.title.text == "96-well: the registered plate"


def test_the_estimated_map_draws_the_wells_where_the_holder_puts_them(
    page: PlatePage,
) -> None:
    view = PlatePageView(page)

    data = view.wells_source.data
    index = list(data["name"]).index("A1")
    assert (data["x"][index], data["y"][index]) == pytest.approx(ESTIMATED_A1, abs=0.01)


# -- the cross is labelled with the well it is over --------------------------------


def test_the_cross_is_labelled_with_the_well_it_is_over(
    page: PlatePage, run_doc: Path
) -> None:
    view, scripted = _scripted(page)
    scripted.stage_xy = B3

    view.tick()
    assert _cross(view) == {"x": [B3[0]], "y": [B3[1]], "text": ["over B3 (estimated)"]}

    # A registration that places the plate where the estimate did: the same well, no
    # longer an estimate. The label follows the redraw of the view at once, not the next tick.
    _register(run_doc, a1=ESTIMATED_A1)
    view.refresh()
    assert _cross(view)["text"] == ["over B3"]
    view.tick()
    assert _cross(view)["text"] == ["over B3"]


def test_a_seed_labels_the_cross_with_its_assumption(
    page: PlatePage, run_doc: Path
) -> None:
    _register(
        run_doc,
        a1=ESTIMATED_A1,
        method="seed",
        refs=(WellRef(well="B3", x_um=B3[0], y_um=B3[1]),),
    )
    view, scripted = _scripted(page)
    scripted.stage_xy = B3

    view.tick()

    assert _cross(view)["text"] == ["over B3 (rotation assumed)"]


def test_the_cross_has_no_label_where_the_stand_does_not_say_where_its_plate_lies(
    bare_page: PlatePage,
) -> None:
    view, scripted = _scripted(bare_page)
    # The stage at the nominal A1: the unplaced map draws A1 there, and names no well.
    scripted.stage_xy = (0.0, 0.0)

    view.tick()

    assert _cross(view) == {"x": [0.0], "y": [0.0], "text": [""]}


def test_the_label_before_the_first_refresh_is_empty_and_asks_for_no_view(
    page: PlatePage,
) -> None:
    # FM-145: the label is computed from the last drawn view, and a tick never builds one.
    view, scripted = _scripted(page)
    view._view = None
    scripted.stage_xy = B3
    built: list[None] = []
    real_view = page.view

    def counting_view() -> Any:
        built.append(None)
        return real_view()

    page.view = counting_view  # type: ignore[method-assign]
    try:
        view._seen_state = view._shared_state()  # no other tab changed anything
        view.tick()
    finally:
        del page.view

    assert built == []
    assert _cross(view) == {"x": [B3[0]], "y": [B3[1]], "text": [""]}


def test_the_stage_columns_always_have_one_length(page: PlatePage) -> None:
    view, scripted = _scripted(page)
    # Before any tick: three columns, none with a row.
    assert _cross(view) == {"x": [], "y": [], "text": []}

    # A closed stand reads no position: three empty lists, never one that is longer.
    scripted.stage_xy = None
    view.tick()
    assert _cross(view) == {"x": [], "y": [], "text": []}

    # A position on no well: a one-row cross with an empty label, still of one length.
    scripted.stage_xy = (0.0, 0.0)
    view.tick()
    assert {key: len(column) for key, column in _cross(view).items()} == {
        "x": 1,
        "y": 1,
        "text": 1,
    }
    scripted.stage_xy = None
    view.tick()
    assert _cross(view) == {"x": [], "y": [], "text": []}


# -- the two writers of the cross --------------------------------------------------


def test_while_live_runs_the_cross_follows_the_frames_not_the_tick(
    page: PlatePage,
) -> None:
    view, scripted = _scripted(page)
    p1, p2 = B3, (B3[0] + 500.0, B3[1] - 300.0)
    scripted.live = True
    scripted.frame_now = _frame(1, p1)
    scripted.stage_xy = p1

    view.frame_tick()
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p1[0]], [p1[1]])

    # The stage has moved since that frame: the tick leaves the cross to the frames.
    scripted.stage_xy = p2
    view.tick()
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p1[0]], [p1[1]])

    # The next frame carries the new position.
    scripted.frame_now = _frame(2, p2)
    view.frame_tick()
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p2[0]], [p2[1]])

    # Live ends: the tick draws the stage again, and no frame draws it any more.
    p3 = (p2[0] + 100.0, p2[1])
    scripted.live = False
    scripted.stage_xy = p3
    view.tick()
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p3[0]], [p3[1]])
    scripted.frame_now = _frame(3, p1)
    view.frame_tick()
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p3[0]], [p3[1]])


def test_live_that_has_drawn_no_frame_yet_leaves_the_cross_to_the_tick(
    page: PlatePage,
) -> None:
    view, scripted = _scripted(page)
    scripted.live = True
    scripted.stage_xy = B3

    view.tick()

    assert (_cross(view)["x"], _cross(view)["y"]) == ([B3[0]], [B3[1]])


def test_a_snap_outside_live_never_moves_the_cross(page: PlatePage) -> None:
    view, scripted = _scripted(page)
    here, there = B3, (B3[0] + 2 * PITCH_UM, B3[1])
    scripted.stage_xy = here
    view.tick()

    # A snap's frame is old by the time it is drawn: the stage may be anywhere since.
    scripted.frame_now = _frame(1, there)
    view.frame_tick()
    assert _cross(view)["x"] == [here[0]]
    view.tick()
    assert _cross(view)["x"] == [here[0]]
    assert view.image_figure.title.text == "frame 1"


def test_a_frame_without_a_position_leaves_the_cross_where_it_was(
    page: PlatePage,
) -> None:
    view, scripted = _scripted(page)
    p1, p2 = B3, (B3[0] + 500.0, B3[1])
    scripted.live = True
    scripted.frame_now = _frame(1, p1)
    view.frame_tick()

    scripted.frame_now = _frame(2, None)
    view.frame_tick()
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p1[0]], [p1[1]])

    # The flag stayed as it was: the tick still leaves the cross to the frames.
    scripted.stage_xy = p2
    view.tick()
    assert _cross(view)["x"] == [p1[0]]


def test_a_live_frame_that_the_tick_draws_still_moves_the_cross(
    page: PlatePage,
) -> None:
    # The tick draws a new frame too, and now and then it is the first to see one: the
    # picture shown must not be newer than the cross that follows the frames.
    view, scripted = _scripted(page)
    p1, p2 = B3, (B3[0] + 500.0, B3[1] - 300.0)
    scripted.live = True
    scripted.frame_now = _frame(1, p1)
    view.frame_tick()

    scripted.frame_now = _frame(2, p2)
    # The status says another place: in Live it is not what the cross follows.
    scripted.stage_xy = (p2[0] + 700.0, p2[1])
    view.tick()  # draws frame 2 before the frame tick does
    view.frame_tick()  # finds nothing new

    assert view.image_figure.title.text == "frame 2"
    assert (_cross(view)["x"], _cross(view)["y"]) == ([p2[0]], [p2[1]])


def test_a_snap_that_the_tick_draws_outside_live_never_moves_the_cross(
    page: PlatePage,
) -> None:
    view, scripted = _scripted(page)
    here, there = B3, (B3[0] + 2 * PITCH_UM, B3[1])
    scripted.stage_xy = here
    scripted.frame_now = _frame(1, there)

    view.tick()  # draws the snap's frame, and the cross from the status

    assert view.image_figure.title.text == "frame 1"
    assert (_cross(view)["x"], _cross(view)["y"]) == ([here[0]], [here[1]])


def test_frame_tick_in_live_reads_the_frame_and_whether_live_runs_and_nothing_else(
    page: PlatePage,
) -> None:
    # FM-145: it runs ten times a second on the loop that handles STOP, so the cross it
    # moves comes from the frame it has drawn: not the status (a device read) and not the
    # view (339 ms for 1536 wells). A frame already drawn asks nothing more than the frame.
    view, scripted = _scripted(page)
    scripted.live = True
    scripted.frame_now = _frame(1, B3)
    scripted.reads.clear()

    view.frame_tick()

    assert scripted.reads == ["frame", "is_live"]
    assert (_cross(view)["x"], _cross(view)["y"]) == ([B3[0]], [B3[1]])

    scripted.reads.clear()
    view.frame_tick()
    assert scripted.reads == ["frame"]


def test_the_label_is_a_labelset_over_the_stage_columns(page: PlatePage) -> None:
    # The label the operator reads is this layout, wired to the cross's own source and to
    # its `text` column; the data tests above would pass with the wrong column drawn.
    view = PlatePageView(page)

    labels = [
        r
        for r in view.figure.center
        if isinstance(r, LabelSet) and r.source is view.stage_source
    ]

    assert len(labels) == 1
    label = labels[0]
    assert (label.x, label.y, label.text) == ("x", "y", "text")
    assert (label.x_offset, label.y_offset) == (12, 12)
    # The look the plan fixes: small, bold, on a white ground that lets the map show.
    assert label.text_font_size == "11px"
    assert label.text_font_style == "bold"
    assert (label.background_fill_color, label.background_fill_alpha) == (
        "#ffffff",
        0.7,
    )


def test_a_second_tab_draws_the_same_pose_and_cross(
    page: PlatePage, run_doc: Path
) -> None:
    # FM-151: the controller is the server's, and every tab draws what it holds.
    first, first_script = _scripted(page)
    second, second_script = _scripted(page)
    for script in (first_script, second_script):
        script.stage_xy = B3

    first.tick()
    second.tick()
    assert first.figure.title.text == second.figure.title.text == ESTIMATED_TITLE
    assert _cross(first) == _cross(second)
    assert _cross(second)["text"] == ["over B3 (estimated)"]

    _register(run_doc, a1=ESTIMATED_A1)
    first.refresh()
    second.refresh()
    assert first.figure.title.text == second.figure.title.text
    assert first.figure.title.text == "96-well: the registered plate"
    assert _cross(first) == _cross(second)
    assert _cross(second)["text"] == ["over B3"]

    # One tab chooses another plate: the other's tick sees the shared state change and
    # redraws the same pose (the registration is of the 96-well: this one is estimated).
    first.plate_select.value = "greiner-24-well"
    second.tick()
    assert second.figure.title.text.startswith(
        "greiner-24-well: estimated from the stand's holder"
    )
    assert second.figure.title.text == first.figure.title.text
    assert _cross(first) == _cross(second)


def test_live_on_the_fake_stand_draws_the_cross_from_a_frame(
    stand: tuple[PlatePage, Microscope],
) -> None:
    # The real Live: its frame carries the stage's position, read right after the snap.
    page, microscope = stand
    microscope.override(Camera, _SteadyCamera())
    stage = microscope.require(XYStage)
    stage.move_to_um(B3[0], B3[1])
    stage.wait(5.0)
    view = PlatePageView(page)

    page.live(True)
    try:
        _wait_for_frame(page)
        view.frame_tick()
        assert page.is_live
        assert _cross(view)["x"] == pytest.approx([B3[0]], abs=0.5)
        assert _cross(view)["text"] == ["over B3 (estimated)"]
    finally:
        page.live(False)
    assert page.join(10.0)


class _SteadyCamera:
    """The six methods of ``Camera``: a small frame, at once."""

    def snap(self) -> np.ndarray:
        return np.zeros((4, 6), dtype=np.uint16)

    def exposure_ms(self) -> float:
        return 5.0

    def set_exposure_ms(self, value_ms: float) -> float:
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (4, 6)

    def bit_depth(self) -> int:
        return 16

    def pixel_size_um(self) -> float:
        return 0.65


def _wait_for_frame(page: PlatePage, timeout_s: float = 10.0) -> None:
    """Poll the controller until it holds a frame; bounded (FM-44)."""
    deadline = time.monotonic() + timeout_s
    while page.frame() is None:
        assert time.monotonic() < deadline, "no frame within the time"
        time.sleep(0.005)
